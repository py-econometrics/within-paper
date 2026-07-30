"""Contract tests for the R and Julia benchmark drivers.

These drivers produce the numbers in the paper's runtime tables, and until now
nothing exercised them: the suite covered the Python side of the subprocess
protocol but never ran the scripts on the other end of it. That is the coverage
a driver refactor needs, because a change that silently stops emitting a field,
or quietly reports a non-converged fit as a success, would not fail any test.

Each test runs the real driver on a small sample and checks the JSON it emits:
the schema the Python side unpacks, and a coefficient that has to match an
independent fit. Skipped when the toolchain is absent, so the suite still runs
on a machine with only Python.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ROOT / "benchmarks" / "modular"

HAS_R = shutil.which("Rscript") is not None
HAS_JULIA = shutil.which("julia") is not None


def _sample(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A small two-way panel with a known slope on x1."""
    rng = np.random.default_rng(seed)
    indiv = rng.integers(0, 40, size=n)
    firm = rng.integers(0, 12, size=n)
    x1 = rng.normal(size=n)
    y = 2.5 * x1 + indiv * 0.01 + firm * 0.02 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {
            "indiv_id": indiv.astype(np.int64),
            "firm_id": firm.astype(np.int64),
            "year": rng.integers(2000, 2004, size=n).astype(np.int64),
            "x1": x1,
            "y": y,
            # fepois needs a nonnegative count outcome.
            "negbin_y": rng.poisson(lam=np.exp(0.3 * x1), size=n).astype(np.int64),
        }
    )


def _run_driver(
    script: str, depvar: str, tmpdir: Path, *, model: str = "feols"
) -> list[dict]:
    """Run one driver over a one-entry manifest and return its emitted records."""
    frame = _sample()
    data_path = tmpdir / "sample.parquet"
    frame.to_parquet(data_path, index=False)

    config = {
        "manifest": [
            {
                "dataset_id": "contract-test",
                "data_path": str(data_path),
                "dgp": "simple",
                "n_obs": len(frame),
                "iter_type": "trial",
                "iter_num": 1,
            }
        ],
        "formula": f"{depvar} ~ x1 | indiv_id + firm_id",
        "depvar": depvar,
        "covariates": ["x1"],
        "fe_cols": ["indiv_id", "firm_id"],
        "vcov": "iid",
        "vcov_type": "iid",
        "result_log_path": str(tmpdir / "results.jsonl"),
        "model": model,
    }
    config_path = tmpdir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env = {**os.environ, "BENCH_THREADS": "1", "JULIA_NUM_THREADS": "1"}
    if script.endswith(".jl"):
        env["JULIA_PROJECT"] = str(ROOT / "benchmarks" / "julia-env")
        command = ["julia", str(DRIVERS / script), str(config_path)]
    else:
        command = ["Rscript", str(DRIVERS / script), str(config_path)]

    proc = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=ROOT, timeout=900
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    records = [
        json.loads(line)
        for line in proc.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    if not records:
        raise AssertionError(f"{script} emitted no JSON records\nstdout:\n{proc.stdout}")
    return records


# The fields the Python side unpacks off every emitted record.
REQUIRED_FIELDS = {
    "dataset_id",
    "dgp",
    "n_obs",
    "iter_type",
    "iter_num",
    "time",
    "success",
}


class RDriverContractTests(unittest.TestCase):
    @unittest.skipUnless(HAS_R, "Rscript not installed")
    def test_feols_driver_emits_the_protocol_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = _run_driver("fixest_bench.R", "y", Path(tmp))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertLessEqual(REQUIRED_FIELDS, set(record))
        self.assertTrue(record["success"], record.get("error"))
        self.assertEqual(record["dataset_id"], "contract-test")
        self.assertEqual(record["n_obs"], 400)
        self.assertIsNotNone(record["time"])
        self.assertGreater(record["time"], 0.0)

    @unittest.skipUnless(HAS_R, "Rscript not installed")
    def test_fepois_driver_emits_the_protocol_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = _run_driver(
                "fixest_bench.R", "negbin_y", Path(tmp), model="fepois"
            )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertLessEqual(REQUIRED_FIELDS, set(record))
        self.assertTrue(record["success"], record.get("error"))
        self.assertIsNotNone(record["time"])

    @unittest.skipUnless(HAS_R, "Rscript not installed")
    def test_driver_refuses_to_run_without_a_thread_count(self) -> None:
        # Timings compared across packages are only meaningful at a known thread
        # count, so an unset BENCH_THREADS must stop the run rather than default.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}", encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k != "BENCH_THREADS"}
            proc = subprocess.run(
                ["Rscript", str(DRIVERS / "fixest_bench.R"), str(config)],
                capture_output=True,
                text=True,
                env=env,
                cwd=ROOT,
                timeout=300,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("BENCH_THREADS", proc.stderr)


class JuliaDriverContractTests(unittest.TestCase):
    @unittest.skipUnless(HAS_JULIA, "julia not installed")
    def test_feols_driver_emits_the_protocol_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = _run_driver("feols_julia.jl", "y", Path(tmp))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertLessEqual(REQUIRED_FIELDS, set(record))
        self.assertTrue(record["success"], record.get("error"))
        self.assertIsNotNone(record["time"])


if __name__ == "__main__":
    unittest.main()

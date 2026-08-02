"""Small real calls to the final R and Julia siblings."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarks.data import make_base_data

ROOT = Path(__file__).absolute().parents[1]
HAS_R = shutil.which("Rscript") is not None
HAS_JULIA = shutil.which("julia") is not None


def _run(language: str, model: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        data, output = work / "sample.parquet", work / "result.csv"
        make_base_data(1_000, "simple", 22).to_parquet(data, index=False)
        if model == "ols":
            script = ROOT / "benchmarks" / "ols" / (
                "fixest.R" if language == "r" else "fixed_effect_models.jl"
            )
            args = [str(data), str(output), "indiv_id,firm_id,year", "1"]
        else:
            script = ROOT / "benchmarks" / "ppml" / (
                "fixest.R" if language == "r" else "gl_fixed_effect_models.jl"
            )
            args = [str(data), str(output), "1"]
        if language == "r":
            command = ["Rscript", str(script), *args]
        else:
            command = [
                "julia", f"--project={ROOT / 'benchmarks' / 'julia-env'}",
                str(script), *args,
            ]
        environment = {**os.environ, "BENCH_THREADS": "1", "JULIA_NUM_THREADS": "1"}
        subprocess.run(command, check=True, cwd=ROOT, env=environment, timeout=900)
        return pd.read_csv(output)


class RTests(unittest.TestCase):
    @unittest.skipUnless(HAS_R, "Rscript not installed")
    def test_ols_sibling_writes_a_converged_row(self) -> None:
        result = _run("r", "ols")
        self.assertTrue(result.loc[0, "converged"])
        self.assertEqual(result.loc[0, "n_planned"], 1)

    @unittest.skipUnless(HAS_R, "Rscript not installed")
    def test_ppml_sibling_writes_a_converged_row(self) -> None:
        result = _run("r", "ppml")
        self.assertTrue(result.loc[0, "converged"])


class JuliaTests(unittest.TestCase):
    @unittest.skipUnless(HAS_JULIA, "Julia not installed")
    def test_ols_sibling_writes_a_converged_row(self) -> None:
        result = _run("julia", "ols")
        self.assertTrue(result.loc[0, "converged"])
        self.assertEqual(result.loc[0, "n_planned"], 1)

    @unittest.skipUnless(HAS_JULIA, "Julia not installed")
    def test_ppml_sibling_writes_a_converged_row(self) -> None:
        result = _run("julia", "ppml")
        self.assertTrue(result.loc[0, "converged"])


if __name__ == "__main__":
    unittest.main()

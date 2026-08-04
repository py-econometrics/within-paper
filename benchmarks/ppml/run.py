"""Headline PPML comparison."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from statistics import median

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.ppml.pyfixest import OUTER_MAXITER, measure
from benchmarks.runtime import run_native

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml.csv"
N_OBS = 1_000_000
BACKENDS = ("rust-map", "within", "fixest", "GLFEM.jl")
REPETITIONS = 3


def _native(data_path: Path, output: Path, backend: str) -> list[dict]:
    script = "fixest.R" if backend == "fixest" else "gl_fixed_effect_models.jl"
    arguments = [str(data_path), str(output), str(REPETITIONS), str(OUTER_MAXITER)]
    return run_native(
        Path(__file__).with_name(script),
        arguments,
        output,
        backend=backend,
        failure_repetitions=REPETITIONS,
    )


def main() -> None:
    threads = int(os.environ["BENCH_THREADS"])
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    rows = []
    for design, seed in BASE_DESIGNS:
        frame = make_base_data(N_OBS, design, seed)
        with tempfile.TemporaryDirectory(prefix="within-ppml-") as directory:
            work = Path(directory)
            data_path = work / "sample.parquet"
            frame.to_parquet(data_path, index=False)
            for backend in BACKENDS:
                planned = REPETITIONS
                measured = (
                    measure(frame, backend, planned, OUTER_MAXITER)
                    if backend in {"rust-map", "within"}
                    else _native(data_path, work / f"{backend}.csv", backend)
                )
                for row in measured:
                    row.update(
                        design=design, n_obs=len(frame), n_fe=3,
                        threads=threads, view="default",
                        n_planned=planned, outer_maxiter=OUTER_MAXITER,
                    )
                rows.extend(measured)
                times = [
                    float(row["runtime_s"])
                    for row in measured
                    if str(row["converged"]).lower() in {"true", "1"}
                ]
                if times:
                    value = f"{median(times):.3f} s"
                elif measured and all(
                    str(row.get("capped", "")).lower() in {"true", "1"}
                    for row in measured
                ):
                    value = "capped"
                else:
                    value = "failed"
                print(f"bench-fepois / PPML / {design} / {backend}: {value}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

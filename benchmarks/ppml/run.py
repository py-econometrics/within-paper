"""Headline PPML comparison."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from statistics import median

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.ppml.pyfixest import measure

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml.csv"
N_OBS = 1_000_000
BACKENDS = ("rust-map", "within", "fixest", "GLFEM.jl")
REPETITIONS = 3


def _native(data_path: Path, output: Path, design: str, backend: str) -> list[dict]:
    script = "fixest.R" if backend == "fixest" else "gl_fixed_effect_models.jl"
    arguments = [str(data_path), str(output), design, str(REPETITIONS)]
    if backend == "fixest":
        command = ["Rscript", str(Path(__file__).with_name(script)), *arguments]
    else:
        command = [
            "julia", f"--project={ROOT / 'benchmarks' / 'julia-env'}",
            str(Path(__file__).with_name(script)), *arguments,
        ]
    environment = dict(os.environ)
    environment["JULIA_NUM_THREADS"] = environment["BENCH_THREADS"]
    subprocess.run(command, check=True, env=environment)
    return pd.read_csv(output).to_dict("records")


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
            design_rows = []
            for backend in BACKENDS:
                measured = (
                    measure(frame, backend, REPETITIONS)
                    if backend in {"rust-map", "within"}
                    else _native(data_path, work / f"{backend}.csv", design, backend)
                )
                for row in measured:
                    row.update(
                        experiment="ppml", design=design, n_obs=len(frame), n_fe=3,
                        model_k=1, threads=threads, view="default",
                        n_planned=len(measured),
                    )
                rows.extend(measured)
                design_rows.extend(measured)
                times = [
                    float(row["runtime_s"])
                    for row in measured
                    if str(row["converged"]).lower() in {"true", "1"}
                ]
                value = f"{median(times):.3f} s" if times else "failed"
                print(f"bench-fepois / PPML / {design} / {backend}: {value}", flush=True)
            retained = {
                int(float(row["n_retained"]))
                for row in design_rows
                if str(row["converged"]).lower() in {"true", "1"}
            }
            if len(retained) > 1:
                raise RuntimeError(
                    f"PPML backends retained different row counts for {design}: {sorted(retained)}"
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

"""Run one OLS comparison on shared in-memory and temporary data."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median

import pandas as pd

from benchmarks.ols.pyfixest import fit_ols, measure
from benchmarks.runtime import assert_same_retained, run_native

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
PYTHON_BACKENDS = ("rust-map", "within")


def repetitions_for_runtime(seconds: float) -> int:
    if seconds < 1:
        return 20
    if seconds < 10:
        return 7
    return 3


def benchmark_threads() -> int:
    threads = int(os.environ["BENCH_THREADS"])
    if threads < 1:
        raise RuntimeError("BENCH_THREADS must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    return threads


def _native_rows(
    data_path: Path,
    output: Path,
    fixed_effects: Sequence[str],
    backend: str,
    repetitions: int | None,
) -> list[dict]:
    script = "fixest.R" if backend == "fixest" else "fixed_effect_models.jl"
    count = "adaptive" if repetitions is None else str(repetitions)
    arguments = [str(data_path), str(output), ",".join(fixed_effects), count]
    return run_native(Path(__file__).with_name(script), arguments, output)


def _print_cell(experiment: str, design: str, backend: str, rows: list[dict]) -> None:
    times = [
        float(row["runtime_s"])
        for row in rows
        if str(row["converged"]).lower() in {"true", "1"}
    ]
    value = f"{median(times):.3f} s" if times else "failed"
    print(f"{experiment} / OLS / {design} / {backend}: {value}", flush=True)


def run_experiment(
    *,
    experiment: str,
    designs: Sequence[tuple[str, Callable[[], pd.DataFrame]]],
    output: Path,
    fixed_effects: Sequence[str] = ("indiv_id", "firm_id", "year"),
    backends: Sequence[str] = ("rust-map", "within", "fixest", "FEM.jl"),
    repetitions: int | None = None,
    extra_python_cells: Sequence[tuple[str, str, float, int]] = (),
) -> pd.DataFrame:
    """Generate one sample per design and write all measured rows once."""
    threads = benchmark_threads()
    rows = []
    for design, generate in designs:
        frame = generate()
        design_rows = []
        with tempfile.TemporaryDirectory(prefix="within-ols-") as directory:
            work = Path(directory)
            data_path = work / "sample.parquet"
            if any(backend not in PYTHON_BACKENDS for backend in backends):
                frame.to_parquet(data_path, index=False)
            for backend in backends:
                if backend in PYTHON_BACKENDS:
                    warm_up = True
                    if repetitions is None:
                        started = time.perf_counter()
                        fit_ols(frame, backend, fixed_effects)
                        planned = repetitions_for_runtime(time.perf_counter() - started)
                        warm_up = False
                    else:
                        planned = repetitions
                    measured = measure(frame, backend, fixed_effects, planned, warm_up=warm_up)
                else:
                    measured = _native_rows(
                        data_path, work / f"{backend}.csv",
                        fixed_effects, backend, repetitions,
                    )
                for row in measured:
                    row.update(
                        design=design,
                        n_obs=len(frame),
                        n_fe=len(fixed_effects),
                        threads=threads,
                        view="default",
                        n_planned=len(measured),
                    )
                rows.extend(measured)
                design_rows.extend(measured)
                _print_cell(experiment, design, backend, measured)
            assert_same_retained(design_rows, "OLS", design)
        for backend, view, tolerance, maxiter in extra_python_cells:
            measured = measure(
                frame, backend, fixed_effects, repetitions or 3,
                tolerance=tolerance, maxiter=maxiter,
            )
            for row in measured:
                row.update(
                    design=design, n_obs=len(frame),
                    n_fe=len(fixed_effects), threads=threads,
                    view=view, n_planned=len(measured),
                )
            rows.extend(measured)
            _print_cell(experiment, design, backend, measured)
    result = pd.DataFrame(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result

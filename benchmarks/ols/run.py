"""Run one OLS comparison with one temporary sample per design."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median

import pandas as pd
import pyarrow.parquet as pq

from benchmarks.ols.pyfixest import fit_ols, measure
from benchmarks.runtime import failed_trials, run_native

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
# These are the package-default OLS cells that belong in every package-runtime
# comparison. Mechanism tables deliberately use their own matched-accuracy set.
PACKAGE_RUNTIME_BACKENDS = (
    "rust-map",
    "within-off",
    "within-diagonal",
    "within",
    "fixest",
    "FEM.jl",
)
PYTHON_BACKENDS = ("rust-map", "within-off", "within-diagonal", "within")


def repetitions_for_runtime(seconds: float) -> int:
    if seconds < 1:
        return 20
    if seconds < 10:
        return 7
    return 3


def benchmark_threads() -> int:
    threads = int(os.environ["BENCH_THREADS"])
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
    return run_native(
        Path(__file__).with_name(script),
        arguments,
        output,
        backend=backend,
        failure_repetitions=repetitions or 3,
    )


def _run_process(
    target: Callable, *args, tolerate_failure: bool = False
) -> str | None:
    process = mp.get_context("spawn").Process(target=target, args=args)
    process.start()
    process.join()
    if process.exitcode:
        message = f"{target.__name__} exited with status {process.exitcode}"
        if tolerate_failure:
            return message
        raise RuntimeError(message)
    return None


def _write_sample(generate: Callable[[], pd.DataFrame], path: Path) -> None:
    generate().to_parquet(path, index=False)


def _python_rows(
    data_path: Path,
    output: Path,
    fixed_effects: tuple[str, ...],
    backend: str,
    repetitions: int | None,
    tolerance: float | None = None,
    maxiter: int | None = None,
) -> None:
    frame = pd.read_parquet(data_path)
    warm_up = True
    if repetitions is None:
        started = time.perf_counter()
        try:
            fit_ols(frame, backend, fixed_effects, tolerance, maxiter)
        except Exception:
            # Use the failed package-default attempt only to choose a repetition count.
            # The measured calls below retain and report the actual failure.
            pass
        planned = repetitions_for_runtime(time.perf_counter() - started)
        warm_up = False
    else:
        planned = repetitions
    rows = measure(
        frame,
        backend,
        fixed_effects,
        planned,
        warm_up=warm_up,
        tolerance=tolerance,
        maxiter=maxiter,
    )
    for row in rows:
        row["n_planned"] = planned
    pd.DataFrame(rows).to_csv(output, index=False)


def _print_cell(experiment: str, design: str, backend: str, rows: list[dict]) -> None:
    times = [
        float(row["runtime_s"])
        for row in rows
        if str(row["converged"]).lower() in {"true", "1"}
    ]
    if times:
        value = f"{median(times):.3f} s"
    elif rows and all(
        str(row.get("capped", "")).lower() in {"true", "1"} for row in rows
    ):
        value = "capped"
    else:
        value = "failed"
    print(f"{experiment} / OLS / {design} / {backend}: {value}", flush=True)


def run_experiment(
    *,
    experiment: str,
    designs: Sequence[tuple[str, Callable[[], pd.DataFrame]]],
    output: Path | None,
    fixed_effects: Sequence[str] = ("indiv_id", "firm_id", "year"),
    backends: Sequence[str] = ("rust-map", "within", "fixest", "FEM.jl"),
    repetitions: int | None = None,
    extra_python_cells: Sequence[tuple[str, str, float, int]] = (),
) -> pd.DataFrame:
    """Generate one sample per design and measure each cell in a fresh process."""
    threads = benchmark_threads()
    rows = []
    for design, generate in designs:
        with tempfile.TemporaryDirectory(prefix="within-ols-") as directory:
            work = Path(directory)
            data_path = work / "sample.parquet"
            _run_process(_write_sample, generate, data_path)
            n_obs = pq.read_metadata(data_path).num_rows
            for backend in backends:
                cell_output = work / f"{backend}.csv"
                if backend in PYTHON_BACKENDS:
                    process_error = _run_process(
                        _python_rows,
                        data_path,
                        cell_output,
                        tuple(fixed_effects),
                        backend,
                        repetitions,
                        tolerate_failure=True,
                    )
                    measured = (
                        failed_trials(backend, repetitions or 3, process_error)
                        if process_error
                        else pd.read_csv(cell_output).to_dict("records")
                    )
                else:
                    measured = _native_rows(
                        data_path, cell_output,
                        fixed_effects, backend, repetitions,
                    )
                planned = int(measured[0]["n_planned"])
                for row in measured:
                    row.update(
                        design=design,
                        n_obs=n_obs,
                        n_fe=len(fixed_effects),
                        threads=threads,
                        view="default",
                        n_planned=planned,
                    )
                rows.extend(measured)
                _print_cell(experiment, design, backend, measured)
            for backend, view, tolerance, maxiter in extra_python_cells:
                cell_output = work / f"{backend}-{view}.csv"
                planned = repetitions or 3
                process_error = _run_process(
                    _python_rows,
                    data_path,
                    cell_output,
                    tuple(fixed_effects),
                    backend,
                    planned,
                    tolerance,
                    maxiter,
                    tolerate_failure=True,
                )
                measured = (
                    failed_trials(backend, planned, process_error)
                    if process_error
                    else pd.read_csv(cell_output).to_dict("records")
                )
                for row in measured:
                    row.update(
                        design=design, n_obs=n_obs,
                        n_fe=len(fixed_effects), threads=threads,
                        view=view, n_planned=planned,
                    )
                rows.extend(measured)
                _print_cell(experiment, design, backend, measured)
    result = pd.DataFrame(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result

"""Benchmark worker-specific year slopes on selected AKM mobility designs."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from statistics import median

import pandas as pd
import pyarrow.parquet as pq

from benchmarks.akm import make_akm_data
from benchmarks.runtime import failed_trials, run_native
from benchmarks.varying_slopes.pyfixest import PRECONDITIONERS, measure

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
DESIGNS = ("akm_mobility_1", "akm_mobility_3", "akm_mobility_5")
BACKENDS = ("within-diagonal", "within-additive", "fixest", "FEM.jl")
PYTHON_BACKENDS = tuple(PRECONDITIONERS)
REPETITIONS = 3
VARYING_SLOPE = "indiv_id[year]"


def benchmark_threads() -> int:
    """Read and apply the common benchmark thread count."""
    threads = int(os.environ["BENCH_THREADS"])
    if threads <= 0:
        raise ValueError("BENCH_THREADS must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    return threads


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
    backend: str,
    repetitions: int,
) -> None:
    frame = pd.read_parquet(data_path)
    rows = measure(frame, backend, repetitions)
    for row in rows:
        row["n_planned"] = repetitions
    pd.DataFrame(rows).to_csv(output, index=False)


def _native_rows(
    data_path: Path,
    output: Path,
    backend: str,
    repetitions: int,
) -> list[dict]:
    script = "fixest.R" if backend == "fixest" else "fixed_effect_models.jl"
    return run_native(
        Path(__file__).with_name(script),
        [str(data_path), str(output), str(repetitions)],
        output,
        backend=backend,
        failure_repetitions=repetitions,
    )


def _print_cell(design: str, backend: str, rows: list[dict]) -> None:
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
    print(f"varying slopes / OLS / {design} / {backend}: {value}", flush=True)


def run_experiment(
    *,
    designs: Sequence[tuple[str, Callable[[], pd.DataFrame]]],
    output: Path | None,
    backends: Sequence[str] = BACKENDS,
    repetitions: int = REPETITIONS,
) -> pd.DataFrame:
    """Generate one sample per design and time every backend in isolation."""
    threads = benchmark_threads()
    rows = []
    for design, generate in designs:
        with tempfile.TemporaryDirectory(prefix="within-varying-slopes-") as directory:
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
                        backend,
                        repetitions,
                        tolerate_failure=True,
                    )
                    measured = (
                        failed_trials(backend, repetitions, process_error)
                        if process_error
                        else pd.read_csv(cell_output).to_dict("records")
                    )
                else:
                    measured = _native_rows(
                        data_path, cell_output, backend, repetitions
                    )
                for row in measured:
                    row.update(
                        design=design,
                        n_obs=n_obs,
                        n_fe=3,
                        threads=threads,
                        view="default",
                        n_planned=repetitions,
                        varying_slope=VARYING_SLOPE,
                        preconditioner=PRECONDITIONERS.get(backend, ""),
                        package_version=row.get("package_version", ""),
                    )
                rows.extend(measured)
                _print_cell(design, backend, measured)
    result = pd.DataFrame(rows)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    return result


def main() -> None:
    run_experiment(
        designs=[(name, partial(make_akm_data, name)) for name in DESIGNS],
        output=LATEST / "varying_slopes.csv",
    )


if __name__ == "__main__":
    main()

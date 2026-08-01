"""Compare MAP and the three standalone within preconditioners."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from benchmarks.accuracy import accuracy_metrics
from benchmarks.data import BASE_DESIGNS, make_base_data, solver_data
from benchmarks.within.map import map_demean_with_sweeps
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "within_preconditioners.csv"
N_OBS = 100_000
REPETITIONS = 3
MAP_TOLERANCE = 1e-10
LSMR_TOLERANCE = 1e-12
MAXITER = 10_000


def _lsmr(
    categories: np.ndarray,
    rhs: np.ndarray,
    reference: np.ndarray,
    name: str,
    repetition: int,
) -> dict:
    preconditioner = getattr(PreconditionerConfig, name.capitalize())
    options = LsmrOptions(tol=LSMR_TOLERANCE, maxiter=MAXITER)
    started = time.perf_counter()
    solver = Solver(categories, preconditioner=preconditioner)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    result = solver.solve_batch(rhs, options)
    solve = time.perf_counter() - started
    converged = all(result.converged)
    metrics = accuracy_metrics(categories, rhs, np.asarray(result.demeaned), reference)
    return {
        "backend": f"within-{name}",
        "repetition": repetition,
        "setup_s": setup,
        "solve_s": solve,
        "total_s": setup + solve,
        "iterations_max": max(result.iterations),
        "converged": converged,
        "capped": not converged and max(result.iterations) >= MAXITER,
        **metrics,
    }


def _map(
    categories: np.ndarray,
    rhs: np.ndarray,
    reference: np.ndarray,
    repetition: int,
) -> dict:
    started = time.perf_counter()
    result = map_demean_with_sweeps(
        rhs, categories, tol=MAP_TOLERANCE, maxiter=MAXITER
    )
    solve = time.perf_counter() - started
    converged = all(result.converged)
    return {
        "backend": "rust-map",
        "repetition": repetition,
        "setup_s": 0.0,
        "solve_s": solve,
        "total_s": solve,
        "iterations_max": max(result.iterations),
        "converged": converged,
        "capped": not converged and max(result.iterations) >= MAXITER,
        **accuracy_metrics(categories, rhs, result.demeaned, reference),
    }


def main() -> None:
    rows = []
    for design, seed in BASE_DESIGNS:
        categories, rhs = solver_data(make_base_data(N_OBS, design, seed))
        reference_fit = solve_batch(
            categories,
            rhs,
            LsmrOptions(tol=1e-14, maxiter=20_000),
            preconditioner=PreconditionerConfig.Additive,
        )
        reference = np.asarray(reference_fit.demeaned)
        map_demean_with_sweeps(rhs, categories, tol=MAP_TOLERANCE, maxiter=1)
        for backend in ("rust-map", "within-off", "within-diagonal", "within-additive"):
            if backend != "rust-map":
                name = backend.removeprefix("within-")
                solve_batch(
                    categories,
                    rhs,
                    LsmrOptions(tol=LSMR_TOLERANCE, maxiter=MAXITER),
                    preconditioner=getattr(PreconditionerConfig, name.capitalize()),
                )
            measured = []
            for repetition in range(REPETITIONS):
                row = (
                    _map(categories, rhs, reference, repetition)
                    if backend == "rust-map"
                    else _lsmr(categories, rhs, reference, name, repetition)
                )
                row.update(design=design, n_obs=len(categories))
                rows.append(row)
                measured.append(row["total_s"])
            print(
                f"within-preconditioners / OLS / {design} / {backend}: "
                f"{median(measured):.3f} s",
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

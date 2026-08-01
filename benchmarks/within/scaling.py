"""Factor-count scaling and amortization over right-hand sides."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.accuracy import accuracy_metrics
from benchmarks.data import make_base_data, solver_data
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
N_OBS = 1_000_000
REPETITIONS = 3
OPTIONS = LsmrOptions(tol=1e-12, maxiter=10_000)


def _measure(categories: np.ndarray, rhs: np.ndarray, name: str) -> dict:
    preconditioner = getattr(PreconditionerConfig, name.capitalize())
    solve_batch(categories, rhs, OPTIONS, preconditioner=preconditioner)
    started = time.perf_counter()
    solver = Solver(categories, preconditioner=preconditioner)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    result = solver.solve_batch(rhs, OPTIONS)
    solve = time.perf_counter() - started
    return {
        "setup_s": setup,
        "solve_s": solve,
        "total_s": setup + solve,
        "iterations_max": max(result.iterations),
        "converged": all(result.converged),
        "demeaned": np.asarray(result.demeaned),
    }


def factor_scaling(categories: np.ndarray, rhs: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(90_210)
    n_obs = len(categories)
    extra = np.column_stack(
        [
            rng.integers(0, max(2, n_obs // 5_000), n_obs),
            rng.integers(0, max(2, n_obs // 20_000), n_obs),
        ]
    )
    full = np.asfortranarray(np.column_stack([categories, extra]).astype(np.uint32))
    rows = []
    for n_factors in range(2, 6):
        current = np.asfortranarray(full[:, :n_factors])
        reference = solve_batch(
            current, rhs, LsmrOptions(tol=1e-14, maxiter=20_000),
            preconditioner=PreconditionerConfig.Additive,
        )
        for repetition in range(REPETITIONS):
            measured = _measure(current, rhs, "additive")
            metrics = accuracy_metrics(current, rhs, measured.pop("demeaned"), np.asarray(reference.demeaned))
            rows.append(
                {
                    "design": "difficult", "repetition": repetition,
                    "n_factors": n_factors, "n_pairs": n_factors * (n_factors - 1) // 2,
                    "setup_share": measured["setup_s"] / measured["total_s"],
                    **measured, **metrics,
                }
            )
        print(
            f"bench-scaling / OLS / Q={n_factors} / within-additive: "
            f"{pd.DataFrame(rows[-REPETITIONS:]).total_s.median():.3f} s",
            flush=True,
        )
    return rows


def amortization(categories: np.ndarray, base_rhs: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(4_242)
    widest = np.asfortranarray(
        np.column_stack([base_rhs, rng.standard_normal((len(categories), 25))])
    )
    rows = []
    for k_rhs in (1, 2, 5, 10, 25):
        rhs = np.asfortranarray(widest[:, :k_rhs])
        for name in ("diagonal", "additive"):
            for repetition in range(REPETITIONS):
                measured = _measure(categories, rhs, name)
                measured.pop("demeaned")
                rows.append(
                    {
                        "design": "difficult", "repetition": repetition,
                        "backend": f"within-{name}", "preconditioner": name,
                        "k_rhs": k_rhs, "time_per_rhs_s": measured["total_s"] / k_rhs,
                        **measured,
                    }
                )
            print(
                f"bench-scaling / OLS / K={k_rhs} / within-{name}: "
                f"{pd.DataFrame(rows[-REPETITIONS:]).total_s.median():.3f} s",
                flush=True,
            )
    return rows


def main() -> None:
    categories, rhs = solver_data(make_base_data(N_OBS, "difficult", 43))
    LATEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(factor_scaling(categories, rhs)).to_csv(LATEST / "factor_scaling.csv", index=False)
    pd.DataFrame(amortization(categories, rhs)).to_csv(LATEST / "amortization.csv", index=False)


if __name__ == "__main__":
    main()

"""Factor-count scaling and amortization over right-hand sides."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.accuracy import accuracy_metrics
from benchmarks.data import make_base_data, solver_data
from benchmarks.runtime import failure_fields
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
N_OBS = 1_000_000
REPETITIONS = 3
OPTIONS = LsmrOptions(tol=1e-12, maxiter=10_000)


def _measure(categories: np.ndarray, rhs: np.ndarray, name: str) -> dict:
    preconditioner = getattr(PreconditionerConfig, name.capitalize())
    try:
        solve_batch(categories, rhs, options=OPTIONS, preconditioner=preconditioner)
    except Exception:
        pass
    try:
        started = time.perf_counter()
        solver = Solver(categories, preconditioner=preconditioner)
        setup = time.perf_counter() - started
        started = time.perf_counter()
        result = solver.solve_batch(rhs, OPTIONS)
        solve = time.perf_counter() - started
        iterations = max(result.iterations)
        converged = all(result.converged)
        return {
            "setup_s": setup,
            "solve_s": solve,
            "total_s": setup + solve,
            "iterations_max": iterations,
            "converged": converged,
            "capped": not converged and iterations >= OPTIONS.maxiter,
            "error": "" if converged else "within solver returned without convergence",
            "demeaned": np.asarray(result.demeaned),
        }
    except Exception as error:
        return {
            "setup_s": None,
            "solve_s": None,
            "total_s": None,
            "iterations_max": None,
            "demeaned": None,
            **failure_fields(error),
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
            current,
            rhs,
            options=LsmrOptions(tol=1e-14, maxiter=20_000),
            preconditioner=PreconditionerConfig.Additive,
        )
        if not all(reference.converged):
            raise RuntimeError(
                f"tight factor-scaling reference did not converge for Q={n_factors}"
            )
        for repetition in range(REPETITIONS):
            measured = _measure(current, rhs, "additive")
            demeaned = measured.pop("demeaned")
            metrics = (
                accuracy_metrics(
                    current, rhs, demeaned, np.asarray(reference.demeaned)
                )
                if demeaned is not None
                else {"max_eta": None, "max_delta": None}
            )
            rows.append(
                {
                    "design": "difficult",
                    "repetition": repetition,
                    "n_planned": REPETITIONS,
                    "n_factors": n_factors,
                    "setup_share": (
                        measured["setup_s"] / measured["total_s"]
                        if measured["total_s"] is not None
                        else None
                    ),
                    **measured,
                    **metrics,
                }
            )
        recent = rows[-REPETITIONS:]
        successful = [row["total_s"] for row in recent if row["converged"]]
        value = (
            f"{np.median(successful):.3f} s"
            if successful
            else "capped" if all(row["capped"] for row in recent) else "failed"
        )
        print(
            f"bench-scaling / OLS / Q={n_factors} / within-additive: {value}",
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
                        "n_planned": REPETITIONS,
                        "preconditioner": name, "k_rhs": k_rhs,
                        **measured,
                    }
                )
            recent = rows[-REPETITIONS:]
            successful = [row["total_s"] for row in recent if row["converged"]]
            value = (
                f"{np.median(successful):.3f} s"
                if successful
                else "capped" if all(row["capped"] for row in recent) else "failed"
            )
            print(
                f"bench-scaling / OLS / K={k_rhs} / within-{name}: {value}",
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

"""Setup cost across AKM connectivity and reuse across repeated regressions."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from benchmarks.akm import SCENARIOS, make_akm_data
from benchmarks.data import make_base_data, solver_data
from benchmarks.runtime import failure_fields
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
LATEST = ROOT / "results" / "runs" / "latest"
N_OBS = 1_000_000
SETUP_REPETITIONS = 5
REUSE_REPETITIONS = 3
N_REGRESSIONS = 10
OPTIONS = LsmrOptions(tol=1e-12, maxiter=10_000)
MOBILITY_DESIGNS = tuple(
    name for name in SCENARIOS if name.startswith("akm_mobility_")
)


def _success(result) -> tuple[bool, int]:
    iterations = max(result.iterations)
    return all(result.converged), iterations


def _setup_failure(
    design: str, n_factors: int, repetition: int, error: Exception
) -> dict:
    return {
        "design": design,
        "n_factors": n_factors,
        "repetition": repetition,
        "n_planned": SETUP_REPETITIONS,
        "n_obs": N_OBS,
        "n_rhs": 2,
        "setup_s": None,
        "solve_s": None,
        "total_s": None,
        "iterations_max": None,
        **failure_fields(error),
    }


def setup_across_connectivity() -> list[dict]:
    """Measure additive setup and solve time for two and three fixed effects."""
    rows: list[dict] = []
    for design in MOBILITY_DESIGNS:
        categories, rhs = solver_data(make_akm_data(design))
        for n_factors in (2, 3):
            current = np.asfortranarray(categories[:, :n_factors])
            # Warm up native code before the first timed construction.
            try:
                solve_batch(
                    current,
                    rhs,
                    options=OPTIONS,
                    preconditioner=PreconditionerConfig.Additive,
                )
            except Exception:
                pass

            measured: list[dict] = []
            for repetition in range(SETUP_REPETITIONS):
                try:
                    started = time.perf_counter()
                    solver = Solver(
                        current, preconditioner=PreconditionerConfig.Additive
                    )
                    setup_s = time.perf_counter() - started
                    started = time.perf_counter()
                    result = solver.solve_batch(rhs, OPTIONS)
                    solve_s = time.perf_counter() - started
                    converged, iterations = _success(result)
                    row = {
                        "design": design,
                        "n_factors": n_factors,
                        "repetition": repetition,
                        "n_planned": SETUP_REPETITIONS,
                        "n_obs": len(current),
                        "n_rhs": rhs.shape[1],
                        "setup_s": setup_s,
                        "solve_s": solve_s,
                        "total_s": setup_s + solve_s,
                        "iterations_max": iterations,
                        "converged": converged,
                        "capped": not converged and iterations >= OPTIONS.maxiter,
                        "error": (
                            ""
                            if converged
                            else "within solver returned without convergence"
                        ),
                    }
                except Exception as error:
                    row = _setup_failure(design, n_factors, repetition, error)
                rows.append(row)
                measured.append(row)

            successful = [row for row in measured if row.get("converged")]
            if successful:
                setup = median(row["setup_s"] for row in successful)
                solve = median(row["solve_s"] for row in successful)
                value = f"{setup:.3f} s setup + {solve:.3f} s solve"
            elif measured and all(row.get("capped") for row in measured):
                value = "capped"
            else:
                value = "failed"
            print(
                f"amortization / {design} / Q={n_factors}: {value}", flush=True
            )
    return rows


def _regression_right_hand_sides(
    design: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return one fixed-effect design and ten two-column regression problems."""
    seed = 42 if design == "simple" else 43
    frame = make_base_data(N_OBS, design, seed)
    categories, base_rhs = solver_data(frame)
    rng = np.random.default_rng(20_260_805)
    covariates = np.column_stack(
        [base_rhs[:, 1], rng.standard_normal((len(categories), N_REGRESSIONS - 1))]
    )
    regressions = [
        np.asfortranarray(np.column_stack([base_rhs[:, 0], covariates[:, index]]))
        for index in range(N_REGRESSIONS)
    ]
    return categories, regressions


def _run_regressions(
    categories: np.ndarray,
    regressions: list[np.ndarray],
    design: str,
    policy: str,
    repetition: int,
) -> dict:
    config = (
        PreconditionerConfig.Diagonal
        if policy == "diagonal"
        else PreconditionerConfig.Additive
    )
    setup_s = 0.0
    solve_s = 0.0
    iterations = 0
    converged = True
    try:
        cached_solver = None
        if policy == "additive_cached":
            started = time.perf_counter()
            cached_solver = Solver(categories, preconditioner=config)
            setup_s += time.perf_counter() - started

        for rhs in regressions:
            solver = cached_solver
            if solver is None:
                started = time.perf_counter()
                solver = Solver(categories, preconditioner=config)
                setup_s += time.perf_counter() - started
            started = time.perf_counter()
            result = solver.solve_batch(rhs, OPTIONS)
            solve_s += time.perf_counter() - started
            fit_converged, fit_iterations = _success(result)
            converged = converged and fit_converged
            iterations += fit_iterations

        return {
            "policy": policy,
            "design": design,
            "repetition": repetition,
            "n_planned": REUSE_REPETITIONS,
            "n_obs": len(categories),
            "n_regressions": len(regressions),
            "rhs_per_regression": regressions[0].shape[1],
            "setup_s": setup_s,
            "solve_s": solve_s,
            "total_s": setup_s + solve_s,
            "iterations_sum": iterations,
            "converged": converged,
            "capped": not converged and iterations >= OPTIONS.maxiter,
            "error": "" if converged else "one or more regressions did not converge",
        }
    except Exception as error:
        return {
            "policy": policy,
            "design": design,
            "repetition": repetition,
            "n_planned": REUSE_REPETITIONS,
            "n_obs": len(categories),
            "n_regressions": len(regressions),
            "rhs_per_regression": regressions[0].shape[1],
            "setup_s": None,
            "solve_s": None,
            "total_s": None,
            "iterations_sum": None,
            **failure_fields(error),
        }


def repeated_regressions() -> list[dict]:
    """Compare diagonal, rebuilt additive, and cached additive policies."""
    rows: list[dict] = []
    for design in ("simple", "difficult"):
        categories, regressions = _regression_right_hand_sides(design)
        # Warm up both native construction paths and the two-column solve.
        for config in (PreconditionerConfig.Diagonal, PreconditionerConfig.Additive):
            try:
                solve_batch(
                    categories,
                    regressions[0],
                    options=OPTIONS,
                    preconditioner=config,
                )
            except Exception:
                pass

        for policy in ("diagonal", "additive_rebuilt", "additive_cached"):
            measured = []
            for repetition in range(REUSE_REPETITIONS):
                row = _run_regressions(
                    categories, regressions, design, policy, repetition
                )
                rows.append(row)
                measured.append(row)
            successful = [row for row in measured if row.get("converged")]
            if successful:
                value = f"{median(row['total_s'] for row in successful):.3f} s"
            elif measured and all(row.get("capped") for row in measured):
                value = "capped"
            else:
                value = "failed"
            print(
                f"amortization / ten regressions / {design} / {policy}: {value}",
                flush=True,
            )
    return rows


def main() -> None:
    LATEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(setup_across_connectivity()).to_csv(
        LATEST / "akm_setup_cost.csv", index=False
    )
    pd.DataFrame(repeated_regressions()).to_csv(
        LATEST / "regression_reuse.csv", index=False
    )


if __name__ == "__main__":
    main()

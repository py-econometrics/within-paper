"""Separate outer PPML iterations from inner within solves."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data, solver_data
from within import LsmrOptions, PreconditionerConfig, Solver

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml_inner_outer.csv"
N_OBS = 100_000
OUTER_MAXITER = 25
OUTER_TOL = 1e-8
REGIMES = ((1e-8, 1_000), (1e-12, 10_000))


def _deviance(y: np.ndarray, mean: np.ndarray) -> float:
    positive = y > 0
    terms = np.zeros_like(y)
    terms[positive] = y[positive] * np.log(y[positive] / mean[positive])
    return float(2 * np.sum(terms - (y - mean)))


def poisson_irls(
    categories: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    preconditioner: str,
    rebuild_each_step: bool,
    inner_tol: float,
    inner_maxiter: int,
) -> dict:
    """Run one IRLS cell and retain the work counts used by the appendix."""
    config = getattr(PreconditionerConfig, preconditioner.capitalize())
    options = LsmrOptions(tol=inner_tol, maxiter=inner_maxiter)
    beta = 0.0
    mean = np.full(len(y), max(float(np.mean(y)), 1e-8))
    eta = np.log(mean)
    setup_total = solve_total = 0.0
    inner_iterations: list[int] = []
    inner_converged: list[bool] = []
    cached = None
    previous_deviance = _deviance(y, mean)
    outer_converged = False
    started_all = time.perf_counter()

    for outer_iterations in range(1, OUTER_MAXITER + 1):
        weights = mean.copy()
        working = eta + (y - mean) / mean
        rhs = np.asfortranarray(np.column_stack([working, x]))
        started = time.perf_counter()
        if rebuild_each_step or cached is None:
            solver = Solver(categories, weights=weights, preconditioner=config)
            if not rebuild_each_step:
                cached = solver.preconditioner
        else:
            solver = Solver(categories, weights=weights, preconditioner=cached)
        setup_total += time.perf_counter() - started
        started = time.perf_counter()
        result = solver.solve_batch(rhs, options)
        solve_total += time.perf_counter() - started

        demeaned = np.asarray(result.demeaned)
        z_d, x_d = demeaned[:, 0], demeaned[:, 1]
        cross_product = float(np.sum(weights * x_d * x_d))
        if cross_product <= 0:
            break
        beta_new = float(np.sum(weights * x_d * z_d) / cross_product)
        eta_new = np.clip((working - z_d) + x_d * beta_new, -30, 30)
        mean_new = np.maximum(np.exp(eta_new), 1e-12)
        inner_iterations.extend(int(value) for value in result.iterations)
        inner_converged.extend(bool(value) for value in result.converged)
        deviance = _deviance(y, mean_new)
        deviance_change = abs(deviance - previous_deviance) / (abs(previous_deviance) + OUTER_TOL)
        beta_change = abs(beta_new - beta) / (abs(beta) + OUTER_TOL)
        beta, eta, mean, previous_deviance = beta_new, eta_new, mean_new, deviance
        if deviance_change < OUTER_TOL and beta_change < OUTER_TOL:
            outer_converged = True
            break

    return {
        "preconditioner": preconditioner,
        "rebuild_each_step": rebuild_each_step,
        "inner_tol": inner_tol,
        "inner_maxiter": inner_maxiter,
        "outer_iterations": outer_iterations,
        "outer_converged": outer_converged,
        "inner_all_converged": all(inner_converged),
        "inner_iterations_sum": sum(inner_iterations),
        "inner_iterations_max": max(inner_iterations),
        "setup_total_s": setup_total,
        "solve_total_s": solve_total,
        "total_s": time.perf_counter() - started_all,
        "beta_x1": beta,
        "deviance": _deviance(y, mean),
    }


def main() -> None:
    rows = []
    for design, seed in BASE_DESIGNS:
        frame = make_base_data(N_OBS, design, seed)
        categories, rhs = solver_data(frame, ("negbin_y", "x1"))
        for inner_tol, inner_maxiter in REGIMES:
            for preconditioner in ("off", "diagonal", "additive"):
                rebuild_options = (False, True) if preconditioner == "additive" else (False,)
                for rebuild in rebuild_options:
                    row = poisson_irls(
                        categories, rhs[:, 0].copy(), rhs[:, 1].copy(),
                        preconditioner, rebuild, inner_tol, inner_maxiter,
                    )
                    row.update(design=design, n_obs=len(frame))
                    rows.append(row)
                    print(
                        f"ppml-inner-outer / PPML / {design} / within-{preconditioner}: "
                        f"{row['total_s']:.3f} s",
                        flush=True,
                    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

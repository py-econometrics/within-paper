"""PPML inner/outer iteration diagnostics via standalone within IRLS.

PyFixest's public fepois fit does not expose outer IRLS counts or per-step
inner LSMR iterations. This driver runs a transparent Poisson IRLS loop on the
same simple/difficult samples, so the paper can separate:

- one-time preconditioner setup versus per-step solves
- outer IRLS iterations
- inner LSMR iterations (median/max/sum) at each outer step
- final deviance and slope

A rebuild-each-step control measures the cost of a fresh additive
preconditioner against a stale reused one.

Run with:

    pixi run ppml-inner-outer
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from benchmarks.modular.cli import add_dgps_arg
from benchmarks.modular.experiment import (
    PRECONDITIONERS,
    SampleSpec,
    load_sample,
    preconditioner_config,
    write_rows,
)


import within
from within import LsmrOptions, Solver


def _ppml_sample(design: str, n_obs: int):
    """One fixed PPML sample: negative-binomial outcome and one covariate."""
    sample = load_sample(
        SampleSpec(design=design, n_obs=n_obs), rhs_columns=("negbin_y", "x1")
    )
    return sample, sample.rhs[:, 0].copy(), sample.rhs[:, 1].copy()


def _poisson_irls(
    *,
    categories: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    preconditioner: str,
    rebuild_each_step: bool,
    outer_maxiter: int,
    outer_tol: float,
    inner_tol: float,
    inner_maxiter: int,
) -> dict:
    pc = preconditioner_config(preconditioner)
    options = LsmrOptions(tol=inner_tol, maxiter=inner_maxiter)
    n = y.shape[0]
    beta = 0.0
    mu = np.full(n, max(float(np.mean(y)), 1e-8), dtype=np.float64)
    eta = np.log(mu)

    setup_total = 0.0
    solve_total = 0.0
    outer_iterations = 0
    inner_iters: list[int] = []
    inner_converged: list[bool] = []
    reused_solver: Solver | None = None

    t_all = time.perf_counter()
    def poisson_deviance(mu_current):
        ratio = np.ones_like(y)
        positive = y > 0
        ratio[positive] = y[positive] / mu_current[positive]
        return float(
            2.0
            * np.sum(
                y * np.log(ratio, where=positive, out=np.zeros_like(y))
                - (y - mu_current)
            )
        )

    outer_converged = False
    outer_breakdown = None
    deviance_prev = poisson_deviance(mu)
    for outer in range(1, outer_maxiter + 1):
        # Standard Poisson IWLS working response and weights.
        weights = mu.copy()
        working = eta + (y - mu) / mu
        rhs = np.asfortranarray(np.column_stack([working, x]))

        if rebuild_each_step or reused_solver is None:
            gc.collect()
            t0 = time.perf_counter()
            solver = Solver(categories, weights=weights, preconditioner=pc)
            setup_total += time.perf_counter() - t0
            if not rebuild_each_step:
                reused_solver = solver
        else:
            # Stale preconditioner: new weights on A, cached M^{-1} when present.
            # Off returns preconditioner=None; pass Off again in that case.
            cached = reused_solver.preconditioner
            t0 = time.perf_counter()
            solver = Solver(
                categories,
                weights=weights,
                preconditioner=cached if cached is not None else pc,
            )
            setup_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        result = solver.solve_batch(rhs, options)
        solve_total += time.perf_counter() - t0

        demeaned = np.asarray(result.demeaned)
        z_d = demeaned[:, 0]
        x_d = demeaned[:, 1]
        w = weights
        xtwx = float(np.sum(w * x_d * x_d))
        xtwy = float(np.sum(w * x_d * z_d))
        if xtwx <= 0:
            # Degenerate weighted cross-product: the step cannot be taken.
            # This is a breakdown, not convergence.
            outer_breakdown = "non-positive weighted cross-product"
            outer_iterations = outer
            break
        beta_new = xtwy / xtwx
        # eta = P_FE(z) + x_d * beta = (z - z_d) + x_d * beta
        eta_new = (working - z_d) + x_d * beta_new
        eta_new = np.clip(eta_new, -30.0, 30.0)
        mu_new = np.maximum(np.exp(eta_new), 1e-12)

        inner_iters.extend(int(v) for v in result.iterations)
        inner_converged.extend(bool(v) for v in result.converged)
        # Relative deviance change, which is what PPML implementations test.
        # Earlier versions tested max |delta eta| over every observation, first
        # absolutely and then relatively; neither ever fired, so every
        # configuration reported non-converged while their deviances agreed to
        # six digits. A criterion no configuration can satisfy measures the
        # criterion, not the solver.
        deviance_new = poisson_deviance(mu_new)
        delta_deviance = abs(deviance_new - deviance_prev) / (
            abs(deviance_prev) + outer_tol
        )
        deviance_prev = deviance_new
        delta_beta = abs(beta_new - beta) / (abs(beta) + outer_tol)
        beta = beta_new
        eta = eta_new
        mu = mu_new
        outer_iterations = outer
        if delta_deviance < outer_tol and delta_beta < outer_tol:
            outer_converged = True
            break

    total = time.perf_counter() - t_all
    # Poisson deviance: 2 sum(y log(y/mu) - (y - mu)), with 0 log 0 = 0.
    deviance = poisson_deviance(mu)

    return {
        "preconditioner": preconditioner,
        "rebuild_each_step": rebuild_each_step,
        # Whether a stale operator is good enough depends entirely on how much
        # inner accuracy is demanded of it, so the tolerances are part of the
        # result rather than part of the invocation that produced it.
        "inner_tol": inner_tol,
        "inner_maxiter": inner_maxiter,
        "outer_tol": outer_tol,
        "outer_iterations": outer_iterations,
        "outer_maxiter": outer_maxiter,
        # Recorded by the loop itself. Inferring it from
        # outer_iterations < outer_maxiter misreports both directions: a
        # numerical breakdown that exits early reads as converged, and a fit
        # that converges on the last permitted step reads as failed.
        "outer_converged": bool(outer_converged),
        "outer_breakdown": outer_breakdown,
        # An outer fit resting on inner solves that did not converge is not a
        # converged fit, so the two are recorded separately and never merged.
        "inner_all_converged": bool(all(inner_converged)) if inner_converged else None,
        "inner_iterations_sum": int(sum(inner_iters)) if inner_iters else None,
        "inner_iterations_max": int(max(inner_iters)) if inner_iters else None,
        "inner_iterations_median": float(np.median(inner_iters)) if inner_iters else None,
        "inner_n_converged": int(sum(inner_converged)),
        "inner_n_solves": int(len(inner_converged)),
        "setup_total_s": setup_total,
        "solve_total_s": solve_total,
        "total_s": total,
        "beta_x1": float(beta),
        "deviance": deviance,
        "mean_mu": float(np.mean(mu)),
        "mean_y": float(np.mean(y)),
    }



def _parse_regime(text: str) -> tuple[float, int]:
    """Parse a ``TOL:MAXITER`` inner-solver regime."""
    tol, _, maxiter = text.partition(":")
    if not maxiter:
        raise argparse.ArgumentTypeError(
            f"Inner regime {text!r} must be given as TOL:MAXITER"
        )
    return float(tol), int(maxiter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=100_000)
    add_dgps_arg(parser)
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        default=["off", "diagonal", "additive"],
    )
    parser.add_argument("--outer-maxiter", type=int, default=25)
    parser.add_argument("--outer-tol", type=float, default=1e-8)
    # Two regimes, not one. 1e-8 with a 1000-iteration cap is what PyFixest
    # ships and what the runtime tables use; 1e-12 with a 10000-iteration cap
    # is what the mechanism experiments demand. Reuse behaves differently under
    # the two, so reporting either alone would misstate the finding.
    parser.add_argument(
        "--inner-regimes",
        nargs="+",
        default=["1e-8:1000", "1e-12:10000"],
        help="Inner solver regimes as TOL:MAXITER pairs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "runs" / "latest" / "ppml_inner_outer.csv",
    )
    args = parser.parse_args()

    print(f"[ppml-inner-outer] using {within.__file__}", flush=True)
    rows: list[dict] = []
    regimes = [_parse_regime(item) for item in args.inner_regimes]
    for dgp in args.dgps:
        sample, y, x = _ppml_sample(dgp, args.n_obs)
        categories = sample.categories
        for inner_tol, inner_maxiter in regimes:
            for preconditioner in args.preconditioners:
                for rebuild in (
                    (False, True) if preconditioner == "additive" else (False,)
                ):
                    print(
                        f"[ppml-inner-outer] {dgp} pc={preconditioner} "
                        f"rebuild={rebuild} inner_tol={inner_tol:g} "
                        f"inner_maxiter={inner_maxiter}",
                        flush=True,
                    )
                    row = _poisson_irls(
                        categories=categories,
                        y=y,
                        x=x,
                        preconditioner=preconditioner,
                        rebuild_each_step=rebuild,
                        outer_maxiter=args.outer_maxiter,
                        outer_tol=args.outer_tol,
                        inner_tol=inner_tol,
                        inner_maxiter=inner_maxiter,
                    )
                    row["dgp"] = dgp
                    row["n_obs"] = args.n_obs
                    rows.append(row)
                    print(
                        "  outer={outer_iterations} inner_sum={inner_iterations_sum} "
                        "setup={setup_total_s:.3f}s solve={solve_total_s:.3f}s "
                        "dev={deviance:.4g}".format(**row),
                        flush=True,
                    )
        del categories, y, x
        gc.collect()

    write_rows(args.out, rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

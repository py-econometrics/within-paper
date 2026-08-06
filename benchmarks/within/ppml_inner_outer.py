"""Compare PyFixest PPML with its within preconditioner reused or rebuilt."""

from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from statistics import median
from unittest.mock import patch

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.runtime import failure_fields

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml_policy.csv"
STEPS_OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml_policy_steps.csv"
N_OBS = 1_000_000
OUTER_MAXITER = 100
OUTER_TOL = 1e-8
REGIMES = ((1e-8, 1_000),)
REBUILD_OPTIONS = (False, True)
REPETITIONS = 3


def _diagnostic_fields(steps: list[dict]) -> dict:
    """Collapse per-IRLS measurements without discarding the step records."""
    iterations = [step["iterations"] for step in steps]
    recorded_iterations = [value for value in iterations if value is not None]
    return {
        "outer_iterations": len(steps),
        "setup_s": sum(step["setup_s"] for step in steps),
        "solve_s": sum(step["solve_s"] for step in steps),
        "inner_iterations_sum": (
            sum(recorded_iterations) if len(recorded_iterations) == len(steps) else None
        ),
        "inner_iterations_max": (
            max(recorded_iterations, default=0)
            if len(recorded_iterations) == len(steps)
            else None
        ),
        "setup_by_step_s": json.dumps([step["setup_s"] for step in steps]),
        "solve_by_step_s": json.dumps([step["solve_s"] for step in steps]),
        "inner_iterations_by_step": json.dumps(iterations),
    }


def _minimum_iterations(dispatch, *, x, flist, weights, demeaner, preconditioner):
    """Find the first LSMR cap at which all parallel RHS vectors converge."""
    cache: dict[int, bool] = {}

    def converges(cap: int) -> bool:
        if cap in cache:
            return cache[cap]
        trial = replace(demeaner, fixef_maxiter=cap)
        _, success, _ = dispatch(
            x=x,
            flist=flist,
            weights=weights,
            demeaner=trial,
            cached_preconditioner=preconditioner,
        )
        cache[cap] = bool(success)
        return cache[cap]

    lower = 0
    upper = 1
    while upper < demeaner.fixef_maxiter and not converges(upper):
        lower = upper
        upper = min(2 * upper, demeaner.fixef_maxiter)
    if not converges(upper):
        return demeaner.fixef_maxiter
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if converges(middle):
            upper = middle
        else:
            lower = middle
    return upper


def measure_policy(
    frame: pd.DataFrame,
    rebuild_each_step: bool,
    inner_tol: float,
    inner_maxiter: int,
    outer_maxiter: int = OUTER_MAXITER,
    collect_iterations: bool = False,
) -> dict:
    """Run PyFixest's PPML routine under the requested cache policy."""
    import pyfixest as pf
    import pyfixest.estimation.models.fepois_ as fepois_module
    from pyfixest.estimation.models.fepois_ import Fepois

    steps: list[dict] = []
    diagnostic_overhead_s = 0.0
    dispatch_demean = fepois_module.dispatch_demean

    def measured_dispatch(
        x,
        flist,
        weights,
        demeaner,
        cached_preconditioner=None,
    ):
        """Run the within solve while retaining its setup and iteration data."""
        nonlocal diagnostic_overhead_s
        started = time.perf_counter()
        result, success, used_preconditioner = dispatch_demean(
            x=x,
            flist=flist,
            weights=weights,
            demeaner=demeaner,
            cached_preconditioner=cached_preconditioner,
        )
        elapsed = time.perf_counter() - started
        setup_s = (
            float(used_preconditioner.build_time_seconds)
            if cached_preconditioner is None and used_preconditioner is not None
            else 0.0
        )
        iterations = None
        if collect_iterations and success and used_preconditioner is not None:
            diagnostic_started = time.perf_counter()
            iterations = _minimum_iterations(
                dispatch_demean,
                x=x,
                flist=flist,
                weights=weights,
                demeaner=demeaner,
                preconditioner=used_preconditioner,
            )
            diagnostic_overhead_s += time.perf_counter() - diagnostic_started
        steps.append(
            {
                "setup_s": setup_s,
                "solve_s": max(elapsed - setup_s, 0.0),
                "iterations": iterations,
            }
        )
        return result, success, used_preconditioner

    def discard_preconditioner(_model, _preconditioner) -> None:
        return None

    demeaner = pf.LsmrDemeaner(
        fixef_atol=inner_tol,
        fixef_btol=inner_tol,
        fixef_maxiter=inner_maxiter,
        preconditioner="additive",
    )
    working = frame.copy()
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(fepois_module, "dispatch_demean", measured_dispatch)
        )
        if rebuild_each_step:
            stack.enter_context(
                patch.object(Fepois, "_seed_preconditioner", discard_preconditioner)
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"\d+ singleton fixed effect\(s\) dropped"
            )
            warnings.filterwarnings(
                "ignore", message=r"\d+ observations removed because of separation"
            )
            started = time.perf_counter()
            try:
                fit = pf.fepois(
                    "negbin_y ~ x1 | indiv_id + firm_id + year",
                    working,
                    vcov="iid",
                    copy_data=False,
                    store_data=False,
                    lean=True,
                    demeaner=demeaner,
                    iwls_tol=OUTER_TOL,
                    iwls_maxiter=outer_maxiter,
                )
                outer_converged = bool(fit._convergence)
                if not outer_converged:
                    message = (
                        f"PyFixest IRLS reached the maximum number of iterations "
                        f"({outer_maxiter})"
                    )
                    return {
                        "engine": "pyfixest",
                        "rebuild_each_step": rebuild_each_step,
                        "inner_tol": inner_tol,
                        "inner_maxiter": inner_maxiter,
                        "outer_converged": False,
                        "runtime_s": time.perf_counter() - started - diagnostic_overhead_s,
                        "n_retained": None,
                        "beta_x1": None,
                        "deviance": None,
                        **_diagnostic_fields(steps),
                        **failure_fields(message),
                    }
                return {
                    "engine": "pyfixest",
                    "rebuild_each_step": rebuild_each_step,
                    "inner_tol": inner_tol,
                    "inner_maxiter": inner_maxiter,
                    "outer_converged": True,
                    "runtime_s": time.perf_counter() - started - diagnostic_overhead_s,
                    "n_retained": int(fit._N),
                    "beta_x1": float(fit.coef().loc["x1"]),
                    "deviance": float(fit.deviance),
                    "converged": True,
                    "capped": False,
                    "error": "",
                    **_diagnostic_fields(steps),
                }
            except Exception as error:
                return {
                    "engine": "pyfixest",
                    "rebuild_each_step": rebuild_each_step,
                    "inner_tol": inner_tol,
                    "inner_maxiter": inner_maxiter,
                    "outer_converged": False,
                    "runtime_s": time.perf_counter() - started - diagnostic_overhead_s,
                    "n_retained": None,
                    "beta_x1": None,
                    "deviance": None,
                    **_diagnostic_fields(steps),
                    **failure_fields(error),
                }


def measure_policy_steps(
    frame: pd.DataFrame,
    design: str,
    rebuild_each_step: bool,
    inner_tol: float = 1e-8,
    inner_maxiter: int = 1_000,
) -> list[dict]:
    """Record one exact inner-iteration diagnostic row per outer step."""
    diagnostic = measure_policy(
        frame,
        rebuild_each_step,
        inner_tol,
        inner_maxiter,
        collect_iterations=True,
    )
    setup_by_step = json.loads(diagnostic["setup_by_step_s"])
    solve_by_step = json.loads(diagnostic["solve_by_step_s"])
    iterations_by_step = json.loads(diagnostic["inner_iterations_by_step"])
    backend = "within-rebuild" if rebuild_each_step else "within-reuse"
    return [
        {
            "design": design,
            "backend": backend,
            "rebuild_each_step": rebuild_each_step,
            "n_obs": len(frame),
            "n_fe": 3,
            "inner_tol": inner_tol,
            "inner_maxiter": inner_maxiter,
            "outer_step": outer_step,
            "outer_iterations": diagnostic["outer_iterations"],
            "setup_s": setup_s,
            "solve_s": solve_s,
            "inner_iterations": iterations,
        }
        for outer_step, (setup_s, solve_s, iterations) in enumerate(
            zip(setup_by_step, solve_by_step, iterations_by_step, strict=True),
            start=1,
        )
    ]


def main() -> None:
    threads = int(os.environ["BENCH_THREADS"])
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    rows = []
    step_rows = []
    for design, seed in BASE_DESIGNS:
        frame = make_base_data(N_OBS, design, seed)
        for inner_tol, inner_maxiter in REGIMES:
            for rebuild in REBUILD_OPTIONS:
                measure_policy(frame, rebuild, inner_tol, inner_maxiter)
                measured = []
                for repetition in range(REPETITIONS):
                    row = measure_policy(frame, rebuild, inner_tol, inner_maxiter)
                    row.update(
                        design=design,
                        n_obs=len(frame),
                        n_fe=3,
                        threads=threads,
                        view="default",
                        backend="within-rebuild" if rebuild else "within-reuse",
                        repetition=repetition,
                        n_planned=REPETITIONS,
                        outer_maxiter=OUTER_MAXITER,
                    )
                    rows.append(row)
                    measured.append(row)
                diagnostic_steps = measure_policy_steps(
                    frame, design, rebuild, inner_tol, inner_maxiter
                )
                for row in diagnostic_steps:
                    row["threads"] = threads
                step_rows.extend(diagnostic_steps)
                treatment = "rebuild" if rebuild else "reuse"
                times = [row["runtime_s"] for row in measured if row["outer_converged"]]
                runtime = f"{median(times):.3f} s" if times else "failed"
                print(
                    f"ppml-inner-outer / PPML / {design} / additive-{treatment} / "
                    f"tol={inner_tol:g}: {runtime}",
                    flush=True,
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    pd.DataFrame(step_rows).to_csv(STEPS_OUTPUT, index=False)


if __name__ == "__main__":
    main()

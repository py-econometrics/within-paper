"""Compare PyFixest PPML with its within preconditioner reused or rebuilt."""

from __future__ import annotations

import time
import warnings
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from unittest.mock import patch

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "ppml_inner_outer.csv"
N_OBS = 100_000
OUTER_MAXITER = 100
OUTER_TOL = 1e-8
REGIMES = ((1e-8, 1_000), (1e-12, 10_000))
REBUILD_OPTIONS = (False, True)
REPETITIONS = 7


def measure_policy(
    frame: pd.DataFrame,
    rebuild_each_step: bool,
    inner_tol: float,
    inner_maxiter: int,
) -> dict:
    """Run PyFixest unchanged except for the requested cache policy."""
    import pyfixest as pf
    import pyfixest.estimation.models.fepois_ as fepois_module
    from pyfixest.estimation.models.fepois_ import Fepois

    outer_iterations = 0
    dispatch_demean = fepois_module.dispatch_demean

    def counted_dispatch(*args, **kwargs):
        nonlocal outer_iterations
        outer_iterations += 1
        return dispatch_demean(*args, **kwargs)

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
            patch.object(fepois_module, "dispatch_demean", counted_dispatch)
        )
        if rebuild_each_step:
            stack.enter_context(
                patch.object(Fepois, "_seed_preconditioner", discard_preconditioner)
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"\d+ singleton fixed effect\(s\) dropped"
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
                    iwls_maxiter=OUTER_MAXITER,
                )
                return {
                    "engine": "pyfixest",
                    "rebuild_each_step": rebuild_each_step,
                    "inner_tol": inner_tol,
                    "inner_maxiter": inner_maxiter,
                    "outer_iterations": outer_iterations,
                    "outer_converged": bool(fit._convergence),
                    "runtime_s": time.perf_counter() - started,
                    "n_retained": int(fit._N),
                    "beta_x1": float(fit.coef().loc["x1"]),
                    "deviance": float(fit.deviance),
                    "error": "",
                }
            except Exception as error:
                return {
                    "engine": "pyfixest",
                    "rebuild_each_step": rebuild_each_step,
                    "inner_tol": inner_tol,
                    "inner_maxiter": inner_maxiter,
                    "outer_iterations": outer_iterations,
                    "outer_converged": False,
                    "runtime_s": time.perf_counter() - started,
                    "n_retained": None,
                    "beta_x1": None,
                    "deviance": None,
                    "error": str(error),
                }


def main() -> None:
    rows = []
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
                        repetition=repetition,
                        n_planned=REPETITIONS,
                    )
                    rows.append(row)
                    measured.append(row)
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


if __name__ == "__main__":
    main()

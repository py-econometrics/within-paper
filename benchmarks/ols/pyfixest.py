"""Direct PyFixest OLS fits used by the paper experiments."""

from __future__ import annotations

import importlib.metadata
import time
import warnings
from collections.abc import Sequence

import pandas as pd

from benchmarks.ols.specifications import INTERCEPTS, WORKER_YEAR_SLOPE
from benchmarks.runtime import failure_fields


PRECONDITIONERS = {
    "rust-map": "",
    "within-off": "off",
    "within-diagonal": "diagonal",
    "within": "additive",
    "within-additive": "additive",
}


def demeaner(backend: str, tolerance: float | None = None, maxiter: int | None = None):
    import pyfixest as pf

    if backend == "rust-map":
        settings = {}
        if tolerance is not None:
            settings["fixef_tol"] = tolerance
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.MapDemeaner(**settings)
    if backend == "within":
        return pf.LsmrDemeaner()
    if backend.startswith("within-"):
        settings = {"preconditioner": backend.removeprefix("within-")}
        if tolerance is not None:
            settings.update(fixef_atol=tolerance, fixef_btol=tolerance)
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.LsmrDemeaner(backend="within", **settings)
    raise ValueError(f"unknown PyFixest backend {backend!r}")


def formula_for_specification(
    fixed_effects: Sequence[str], specification: str
) -> str:
    """Return the common formula represented by a benchmark specification."""
    if specification == INTERCEPTS:
        absorbed = " + ".join(fixed_effects)
    elif specification == WORKER_YEAR_SLOPE:
        if tuple(fixed_effects) != ("indiv_id", "firm_id", "year"):
            raise ValueError(
                "worker-year-slope requires indiv_id, firm_id, and year effects"
            )
        absorbed = "indiv_id[year] + firm_id + year"
    else:
        raise ValueError(f"unknown OLS specification {specification!r}")
    return "y ~ x1 | " + absorbed


def fit_ols(
    frame: pd.DataFrame,
    backend: str,
    fixed_effects: Sequence[str],
    tolerance: float | None = None,
    maxiter: int | None = None,
    *,
    lean: bool = True,
    specification: str = INTERCEPTS,
):
    import pyfixest as pf

    return pf.feols(
        formula_for_specification(fixed_effects, specification),
        frame,
        vcov="iid",
        copy_data=False,
        store_data=False,
        lean=lean,
        demeaner=demeaner(backend, tolerance, maxiter),
    )


def measure(
    frame: pd.DataFrame,
    backend: str,
    fixed_effects: Sequence[str],
    repetitions: int,
    *,
    warm_up: bool = True,
    tolerance: float | None = None,
    maxiter: int | None = None,
    specification: str = INTERCEPTS,
) -> list[dict]:
    """Run one warm-up and the requested measured OLS fits."""
    package_version = importlib.metadata.version("pyfixest")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\d+ singleton fixed effect\(s\) dropped")
        if warm_up:
            try:
                fit_ols(
                    frame,
                    backend,
                    fixed_effects,
                    tolerance,
                    maxiter,
                    specification=specification,
                )
            except Exception:
                # A warm-up prepares package state but is not a benchmark trial. If the
                # package default fails, record that failure in the measured rows below.
                pass
        rows = []
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_ols(
                    frame,
                    backend,
                    fixed_effects,
                    tolerance,
                    maxiter,
                    specification=specification,
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "backend": backend,
                        "preconditioner": PRECONDITIONERS[backend],
                        "package_version": package_version,
                        "repetition": repetition,
                        "runtime_s": elapsed,
                        "n_retained": int(fit._N),
                        "beta_x1": float(fit.coef().loc["x1"]),
                        "converged": True,
                        "capped": False,
                        "error": "",
                    }
                )
            except Exception as error:  # a failed measured attempt is part of the result
                rows.append(
                    {
                        "backend": backend,
                        "preconditioner": PRECONDITIONERS[backend],
                        "package_version": package_version,
                        "repetition": repetition,
                        "runtime_s": time.perf_counter() - started,
                        "n_retained": None,
                        "beta_x1": None,
                        **failure_fields(error),
                    }
                )
    return rows

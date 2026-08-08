"""Direct PyFixest OLS fits used by the paper experiments."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

import pandas as pd

from benchmarks.runtime import failure_fields


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
        return pf.LsmrDemeaner(**settings)
    raise ValueError(f"unknown PyFixest backend {backend!r}")


def fit_ols(
    frame: pd.DataFrame,
    backend: str,
    fixed_effects: Sequence[str],
    tolerance: float | None = None,
    maxiter: int | None = None,
    *,
    lean: bool = True,
):
    import pyfixest as pf

    formula = "y ~ x1 | " + " + ".join(fixed_effects)
    return pf.feols(
        formula,
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
) -> list[dict]:
    """Run one warm-up and the requested measured OLS fits."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\d+ singleton fixed effect\(s\) dropped")
        if warm_up:
            try:
                fit_ols(frame, backend, fixed_effects, tolerance, maxiter)
            except Exception:
                # A warm-up prepares package state but is not a benchmark trial. If the
                # package default fails, record that failure in the measured rows below.
                pass
        rows = []
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_ols(frame, backend, fixed_effects, tolerance, maxiter)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "backend": backend,
                        "repetition": repetition,
                        "runtime_s": elapsed,
                        "n_retained": int(fit._N),
                        "beta_x1": float(fit.coef().loc["x1"]),
                        "max_eta": None,
                        "converged": True,
                        "capped": False,
                        "error": "",
                    }
                )
            except Exception as error:  # a failed measured attempt is part of the result
                rows.append(
                    {
                        "backend": backend,
                        "repetition": repetition,
                        "runtime_s": time.perf_counter() - started,
                        "n_retained": None,
                        "beta_x1": None,
                        "max_eta": None,
                        **failure_fields(error),
                    }
                )
    return rows

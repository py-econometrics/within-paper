"""PyFixest varying-slope fits through the within backend."""

from __future__ import annotations

import importlib.metadata
import time
import warnings

import pandas as pd

from benchmarks.runtime import failure_fields

FORMULA = "y ~ x1 | indiv_id[year] + firm_id + year"
PRECONDITIONERS = {
    "within-diagonal": "diagonal",
    "within-additive": "additive",
}


def demeaner(backend: str):
    """Construct the supported within-backed varying-slope demeaner."""
    import pyfixest as pf

    try:
        preconditioner = PRECONDITIONERS[backend]
    except KeyError as error:
        raise ValueError(f"unknown varying-slope backend {backend!r}") from error
    return pf.LsmrDemeaner(
        backend="within",
        preconditioner=preconditioner,
    )


def fit_varying_slope(frame: pd.DataFrame, backend: str, *, lean: bool = True):
    """Fit the common worker-specific year-slope specification."""
    import pyfixest as pf

    return pf.feols(
        FORMULA,
        frame,
        vcov="iid",
        copy_data=False,
        store_data=False,
        lean=lean,
        demeaner=demeaner(backend),
    )


def measure(
    frame: pd.DataFrame,
    backend: str,
    repetitions: int,
    *,
    warm_up: bool = True,
) -> list[dict]:
    """Run one warm-up and the requested measured regressions."""
    package_version = importlib.metadata.version("pyfixest")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"\d+ singleton fixed effect\(s\) dropped"
        )
        if warm_up:
            try:
                fit_varying_slope(frame, backend)
            except Exception:
                # The warm-up is not a benchmark trial. Measured calls below keep
                # any package error and report it as part of the result.
                pass
        rows = []
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_varying_slope(frame, backend)
                rows.append(
                    {
                        "backend": backend,
                        "preconditioner": PRECONDITIONERS[backend],
                        "package_version": package_version,
                        "repetition": repetition,
                        "runtime_s": time.perf_counter() - started,
                        "n_retained": int(fit._N),
                        "beta_x1": float(fit.coef().loc["x1"]),
                        "converged": True,
                        "capped": False,
                        "error": "",
                    }
                )
            except Exception as error:
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

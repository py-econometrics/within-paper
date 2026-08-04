"""Direct PyFixest PPML fits."""

from __future__ import annotations

import time
import warnings

import pandas as pd

from benchmarks.runtime import failure_fields

OUTER_MAXITER = 100

def _demeaner(backend: str):
    import pyfixest as pf

    if backend == "rust-map":
        return pf.MapDemeaner()
    if backend == "within":
        return pf.LsmrDemeaner()
    raise ValueError(f"unknown PyFixest backend {backend!r}")


def fit_ppml(
    frame: pd.DataFrame, backend: str, outer_maxiter: int = OUTER_MAXITER
):
    import pyfixest as pf

    return pf.fepois(
        "negbin_y ~ x1 | indiv_id + firm_id + year",
        frame,
        vcov="iid",
        copy_data=False,
        store_data=False,
        lean=True,
        demeaner=_demeaner(backend),
        iwls_maxiter=outer_maxiter,
    )


def measure(
    frame: pd.DataFrame,
    backend: str,
    repetitions: int,
    outer_maxiter: int = OUTER_MAXITER,
) -> list[dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\d+ singleton fixed effect\(s\) dropped")
        try:
            fit_ppml(frame, backend, outer_maxiter)
        except Exception:
            # A warm-up only prepares package state. A package-default failure belongs
            # in the measured rows below, not as a failure of the whole comparison.
            pass
        rows = []
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_ppml(frame, backend, outer_maxiter)
                rows.append(
                    {
                        "backend": backend,
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
                        "repetition": repetition,
                        "runtime_s": time.perf_counter() - started,
                        "n_retained": None,
                        "beta_x1": None,
                        **failure_fields(error),
                    }
                )
    return rows

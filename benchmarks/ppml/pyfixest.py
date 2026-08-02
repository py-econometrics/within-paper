"""Direct PyFixest PPML fits."""

from __future__ import annotations

import time
import warnings

import pandas as pd


def _demeaner(backend: str):
    import pyfixest as pf

    if backend == "rust-map":
        return pf.MapDemeaner()
    if backend == "within":
        return pf.LsmrDemeaner()
    raise ValueError(f"unknown PyFixest backend {backend!r}")


def fit_ppml(frame: pd.DataFrame, backend: str):
    import pyfixest as pf

    return pf.fepois(
        "negbin_y ~ x1 | indiv_id + firm_id + year",
        frame,
        vcov="iid",
        copy_data=False,
        store_data=False,
        lean=True,
        demeaner=_demeaner(backend),
        iwls_maxiter=100,
    )


def measure(frame: pd.DataFrame, backend: str, repetitions: int) -> list[dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\d+ singleton fixed effect\(s\) dropped")
        fit_ppml(frame, backend)
        rows = []
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_ppml(frame, backend)
                rows.append(
                    {
                        "backend": backend,
                        "repetition": repetition,
                        "runtime_s": time.perf_counter() - started,
                        "n_retained": int(fit._N),
                        "beta_x1": float(fit.coef().loc["x1"]),
                        "converged": True,
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
                        "converged": False,
                        "error": str(error),
                    }
                )
    return rows

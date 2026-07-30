from __future__ import annotations

import time
import warnings

import pandas as pd

from benchmarks.modular.feols_benchmarkers import (
    _TablePrinter,
    _beta_x1,
    demeaner_for,
    _dgp_width,
    _fit_converged,
    _group_key,
    _preconditioner_build_s,
    _read_data_columns,
    _result_from_dataset,
    _retained_rows,
    _trim_process_memory,
)
from benchmarks.modular.subprocess_backend import (
    _SCRIPT_DIR,
    JuliaFeolsBenchmarker,
    SubprocessFeolsBenchmarker,
)
from benchmarks.modular.interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
class PyFepoisBenchmarkerFullApi:
    """Benchmark one pf.fepois() call with the selected demeaning backend."""

    def __init__(
        self,
        name: str,
        demeaner_backend: str,
        *,
        iwls_maxiter: int,
        tol: float | None = None,
        maxiter: int | None = None,
    ):
        self._name = name
        self._demeaner_backend = demeaner_backend
        self._iwls_maxiter = iwls_maxiter
        self._tol = tol
        self._maxiter = maxiter

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        import pyfixest as pf

        demeaner = demeaner_for(
            self._demeaner_backend, tol=self._tol, maxiter=self._maxiter
        )

        results: list[FeolsResult] = []
        all_cols = [spec.depvar, *spec.covariates, *spec.fe_cols]

        tbl = _TablePrinter(_dgp_width(datasets))
        tbl.print_header(self.name)

        group_buf: list[FeolsResult] = []
        prev_key: tuple | None = None

        for dataset in datasets:
            n_obs_for_result = dataset.n_obs
            df = None
            try:
                df = _read_data_columns(dataset.data_path, all_cols)
                n_obs_for_result = len(df)

                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"\d+ singleton fixed effect\(s\) dropped from the model\.",
                        category=UserWarning,
                    )
                    fit = pf.fepois(
                        fml=spec.formula,
                        data=df,
                        vcov=spec.vcov,
                        copy_data=False,
                        store_data=False,
                        demeaner=demeaner,
                        iwls_maxiter=self._iwls_maxiter,
                    )
                    if not _fit_converged(fit):
                        raise RuntimeError("PyFixest PPML model did not converge")
                elapsed = time.perf_counter() - t0

                # Outer IRLS and per-step inner LSMR counts are not exposed on
                # the public PyFixest model. Record what is available and mark
                # the missing iteration fields explicitly so the paper does not
                # invent them. Standalone within PPML diagnostics fill the gap.
                deviance = getattr(fit, "deviance", None)
                loglik = getattr(fit, "_loglik", None)
                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=elapsed,
                    success=True,
                    n_obs_override=n_obs_for_result,
                    outer_iterations=None,
                    inner_iterations_sum=None,
                    inner_iterations_max=None,
                    preconditioner_build_s=_preconditioner_build_s(fit),
                    deviance=float(deviance) if deviance is not None else None,
                    loglik=float(loglik) if loglik is not None else None,
                    beta_x1=_beta_x1(fit),
                    n_retained=_retained_rows(fit),
                    censoring="none",
                )
            except Exception as exc:
                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=None,
                    success=False,
                    error=str(exc),
                    n_obs_override=n_obs_for_result,
                )
            finally:
                del df
                _trim_process_memory(self._demeaner_backend)

            results.append(result)

            if result.iter_type != "burnin":
                key = _group_key(result)
                if prev_key is not None and key != prev_key and group_buf:
                    tbl.print_row(group_buf)
                    group_buf = []
                group_buf.append(result)
                prev_key = key

        if group_buf:
            tbl.print_row(group_buf)

        return results


class FixestFepoisBenchmarker(SubprocessFeolsBenchmarker):
    def __init__(self, name: str | None = None, script_path=None):
        super().__init__(
            name=name or "r.fixest (fepois)",
            command_prefix=["Rscript"],
            script_path=(script_path or _SCRIPT_DIR / "fixest_bench.R"),
            model="fepois",
        )


class GLFixedEffectModelsBenchmarker(JuliaFeolsBenchmarker):
    def __init__(self, name: str | None = None, script_path=None):
        super().__init__(
            name=name or "julia.GLFixedEffectModels (fepois)",
            script_path=(script_path or _SCRIPT_DIR / "fepois_julia.jl"),
        )

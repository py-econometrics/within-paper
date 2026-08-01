from __future__ import annotations

import time
import warnings
from typing import Any

from benchmarks.core.interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
from benchmarks.core.runtime import configure_benchmark_runtime
from benchmarks.solvers.common import (
    TablePrinter,
    beta_x1,
    dgp_width,
    fit_converged,
    group_key,
    preconditioner_build_s,
    read_data_columns,
    result_from_dataset,
    retained_rows,
    trim_process_memory,
)
from benchmarks.solvers.settings import demeaner_for


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

    def cache_key(self) -> dict[str, Any]:
        """Settings that make this backend's result file reusable."""
        return {
            "adapter": "pyfixest-fepois",
            "name": self.name,
            "demeaner_backend": self._demeaner_backend,
            "iwls_maxiter": self._iwls_maxiter,
            "tol": self._tol,
            "maxiter": self._maxiter,
        }

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        configure_benchmark_runtime()
        import pyfixest as pf

        demeaner = demeaner_for(
            self._demeaner_backend, tol=self._tol, maxiter=self._maxiter
        )

        results: list[FeolsResult] = []
        all_cols = [spec.depvar, *spec.covariates, *spec.fe_cols]

        tbl = TablePrinter(dgp_width(datasets))
        tbl.print_header(self.name)

        group_buf: list[FeolsResult] = []
        prev_key: tuple | None = None

        for dataset in datasets:
            n_obs_for_result = dataset.n_obs
            df = None
            try:
                df = read_data_columns(dataset.data_path, all_cols)
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
                    if not fit_converged(fit):
                        raise RuntimeError("PyFixest PPML model did not converge")
                elapsed = time.perf_counter() - t0

                # Outer IRLS and per-step inner LSMR counts are not exposed on
                # the public PyFixest model. Record what is available and mark
                # the missing iteration fields explicitly so the paper does not
                # invent them. Standalone within PPML diagnostics fill the gap.
                deviance = getattr(fit, "deviance", None)
                loglik = getattr(fit, "_loglik", None)
                result = result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=elapsed,
                    success=True,
                    n_obs_override=n_obs_for_result,
                    outer_iterations=None,
                    inner_iterations_sum=None,
                    inner_iterations_max=None,
                    preconditioner_build_s=preconditioner_build_s(fit),
                    deviance=float(deviance) if deviance is not None else None,
                    loglik=float(loglik) if loglik is not None else None,
                    beta_x1=beta_x1(fit),
                    n_retained=retained_rows(fit),
                    censoring="none",
                )
            except Exception as exc:
                result = result_from_dataset(
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
                trim_process_memory(self._demeaner_backend)

            results.append(result)

            if result.iter_type != "burnin":
                key = group_key(result)
                if prev_key is not None and key != prev_key and group_buf:
                    tbl.print_row(group_buf)
                    group_buf = []
                group_buf.append(result)
                prev_key = key

        if group_buf:
            tbl.print_row(group_buf)

        return results

from __future__ import annotations

import time
import warnings

import pandas as pd

try:
    from .feols_benchmarkers import (
        JuliaFeolsBenchmarker,
        SubprocessFeolsBenchmarker,
        _demeaner_from_backend,
        _dgp_width,
        _fit_converged,
        _group_key,
        _read_data_columns,
        _result_from_dataset,
        _SCRIPT_DIR,
        _TablePrinter,
        _trim_process_memory,
    )
    from .interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
except ImportError:
    from feols_benchmarkers import (
        JuliaFeolsBenchmarker,
        SubprocessFeolsBenchmarker,
        _demeaner_from_backend,
        _dgp_width,
        _fit_converged,
        _group_key,
        _read_data_columns,
        _result_from_dataset,
        _SCRIPT_DIR,
        _TablePrinter,
        _trim_process_memory,
    )
    from interfaces import BenchmarkDataset, FeolsResult, FeolsSpec


class PyFepoisBenchmarkerFullApi:
    """Benchmark pf.fepois() end-to-end using one configured demeaner backend."""

    def __init__(self, name: str, demeaner_backend: str, **fepois_kwargs):
        self._name = name
        self._demeaner_backend = demeaner_backend
        self._fepois_kwargs = fepois_kwargs

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        import pyfixest as pf

        fepois_kwargs = dict(self._fepois_kwargs)
        demeaner = _demeaner_from_backend(
            self._demeaner_backend,
            fepois_kwargs.pop("fixef_maxiter", None),
            fepois_kwargs.pop("fixef_tol", None),
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
                        **fepois_kwargs,
                    )
                    if not _fit_converged(fit):
                        raise RuntimeError("PyFixest PPML model returned without convergence")
                elapsed = time.perf_counter() - t0

                result = _result_from_dataset(
                    dataset,
                    spec,
                    backend=self.name,
                    elapsed=elapsed,
                    success=True,
                    n_obs_override=n_obs_for_result,
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
            script_path=(script_path or _SCRIPT_DIR / "fepois_r.R"),
        )


class GLFixedEffectModelsBenchmarker(JuliaFeolsBenchmarker):
    def __init__(self, name: str | None = None, script_path=None):
        super().__init__(
            name=name or "julia.GLFixedEffectModels (fepois)",
            script_path=(script_path or _SCRIPT_DIR / "fepois_julia.jl"),
        )

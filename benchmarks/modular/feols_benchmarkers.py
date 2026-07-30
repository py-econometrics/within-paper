from __future__ import annotations

import ctypes
import gc
import json
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
    from .timing import repetitions_for_runtime
except ImportError:
    from interfaces import BenchmarkDataset, FeolsResult, FeolsSpec
    from timing import repetitions_for_runtime

_MIN_DGP_WIDTH = 16


def _trim_process_memory(demeaner_backend: str) -> None:
    """Return unused Python and native allocator memory after large benchmark cases."""
    gc.collect()

    if demeaner_backend.startswith("torch"):
        try:
            import torch
        except ImportError:
            pass
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    if sys.platform.startswith("linux"):
        with suppress(Exception):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


@dataclass(frozen=True)
class TorchRuntimeAvailability:
    """Availability of the optional Torch backends."""

    has_torch: bool
    has_mps: bool
    has_cuda: bool


def detect_torch_runtime_availability() -> TorchRuntimeAvailability:
    """Return the available Torch backends."""
    try:
        import torch
    except ImportError:
        return TorchRuntimeAvailability(
            has_torch=False,
            has_mps=False,
            has_cuda=False,
        )

    has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    has_cuda = bool(torch.cuda.is_available())
    return TorchRuntimeAvailability(
        has_torch=True,
        has_mps=has_mps,
        has_cuda=has_cuda,
    )


def _fmt_time(t: float) -> str:
    if t < 1:
        return f"{t * 1000:.1f}ms"
    return f"{t:.3f}s"


def _dgp_width(datasets: list[BenchmarkDataset]) -> int:
    return max(
        _MIN_DGP_WIDTH, max((len(d.dgp) for d in datasets), default=_MIN_DGP_WIDTH)
    )


class _TablePrinter:
    """Format benchmark tables with a DGP column wide enough for the labels."""

    def __init__(self, dgp_w: int):
        self._w = dgp_w
        self._hdr = (
            f"{'dgp':<{dgp_w}} {'k':>3} {'n_obs':>12} {'n_fe':>4} "
            f"{'min':>10} {'median':>10} {'max':>10}  status"
        )
        self._sep = "-" * len(self._hdr)

    def print_header(self, name: str) -> None:
        print(f"\n  {name}", flush=True)
        print(f"  {self._sep}", flush=True)
        print(f"  {self._hdr}", flush=True)
        print(f"  {self._sep}", flush=True)

    def _row_prefix(self, r: FeolsResult) -> str:
        return f"{r.dgp:<{self._w}} {r.model_k:>3} {r.n_obs:>12,} {r.n_fe:>4}"

    def print_row(self, results: list[FeolsResult]) -> None:
        columns, status = _time_columns(results)
        print(f"  {self._row_prefix(results[0])} {columns}  {status}", flush=True)


def _time_columns(results: list[FeolsResult]) -> tuple[str, str]:
    times = [r.time for r in results if r.success and r.time is not None]
    if times:
        mn, md, mx = min(times), statistics.median(times), max(times)
        columns = f"{_fmt_time(mn):>10} {_fmt_time(md):>10} {_fmt_time(mx):>10}"
        return columns, "ok"
    errs = [r.error for r in results if r.error]
    status = errs[0][:30] if errs else "FAIL"
    columns = f"{'—':>10} {'—':>10} {'—':>10}"
    return columns, status


def _group_key(r: FeolsResult) -> tuple[str, int, int, int]:
    return (r.dgp, r.model_k, r.n_obs, r.n_fe)


def _result_from_dataset(
    dataset: BenchmarkDataset,
    spec: FeolsSpec,
    *,
    backend: str,
    elapsed: float | None,
    success: bool,
    error: str | None = None,
    n_obs_override: int | None = None,
    **diagnostics,
) -> FeolsResult:
    return FeolsResult(
        source_dataset_id=dataset.dataset_id,
        source_k=dataset.k,
        iter_type=dataset.iter_type,
        iter_num=dataset.iter_num,
        dgp=dataset.dgp,
        model_k=spec.k,
        n_obs=n_obs_override if n_obs_override is not None else dataset.n_obs,
        n_fe=spec.n_fe,
        backend=backend,
        time=elapsed,
        success=success,
        error=error,
        **diagnostics,
    )


def _preconditioner_build_s(fit) -> float | None:
    """Read preconditioner setup time when PyFixest exposes it."""
    pc = getattr(fit, "preconditioner", None)
    if pc is None:
        return None
    value = getattr(pc, "build_time_seconds", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _beta_x1(fit) -> float | None:
    try:
        coef = fit.coef()
        names = [str(name) for name in list(getattr(fit, "_coefnames", []) or [])]
        if hasattr(coef, "tolist"):
            values = list(coef.tolist())
        else:
            values = list(coef)
        values = [float(v) for v in values]
        if "x1" in names:
            return values[names.index("x1")]
        if len(values) == 1:
            return values[0]
    except Exception:
        return None
    return None


def _safe_cast(val, type_fn):
    if val is None:
        return None
    try:
        return type_fn(val)
    except (TypeError, ValueError):
        return None


def _as_bool(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return bool(value)


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    """Print a diagnostic the first time it occurs, so a long run stays readable."""
    if message not in _WARNED:
        _WARNED.add(message)
        print(f"[warn] {message}", file=sys.stderr, flush=True)


def _external_eta(fit, frame, depvar: str, covariates: list[str]) -> float | None:
    """Recompute the external normal-equation residual from a fitted model.

    PROTOCOL.md section 5 requires every headline timing to carry an accuracy
    record on the sample that produced it: a runtime alone cannot answer the
    "is the speedup a tolerance artifact" objection, because each package stops
    on its own quantity.

    The demeaned arrays come from the fit; the untransformed ones come from the
    input frame, restricted to the rows the model kept. `_X_untransformed` is
    not usable for this - it holds the demeaned covariates, so using it would
    compare a vector against itself and report a meaningless ratio.

    The alignment is checked rather than assumed: the reconstructed outcome must
    reproduce `_Y_untransformed` exactly. If it does not, or any piece is
    missing, this returns None, because an accuracy record that cannot be
    computed must be recorded as absent rather than as passing.

    A failure is reported once per process. Returning None quietly would let a
    systematic breakage - a renamed PyFixest attribute, say - read as "this
    backend does not expose accuracy", which is the same value a genuinely
    opaque backend records, and the paper leans on the distinction.
    """
    try:
        import numpy as np
        import pandas as pd

        from accuracy import external_normal_residuals

        fe_frame = getattr(fit, "_fe", None)
        demeaned_y = getattr(fit, "_Y", None)
        raw_y_reference = getattr(fit, "_Y_untransformed", None)
        if fe_frame is None or demeaned_y is None or raw_y_reference is None:
            return None

        dropped = getattr(fit, "_na_index", None) or frozenset()
        keep = np.setdiff1d(
            np.arange(len(frame)), np.fromiter(dropped, dtype=np.int64, count=len(dropped))
        )
        if keep.size != len(fe_frame):
            return None
        kept = frame.iloc[keep]

        raw_y = kept[depvar].to_numpy(dtype=np.float64)
        if not np.allclose(
            raw_y, np.asarray(raw_y_reference, dtype=np.float64).ravel(), rtol=0, atol=0
        ):
            return None

        demeaned_x = getattr(fit, "_X", None)
        raw = np.column_stack(
            [raw_y, *[kept[name].to_numpy(dtype=np.float64) for name in covariates]]
        )
        blocks = [np.asarray(demeaned_y, dtype=np.float64).reshape(keep.size, -1)]
        if demeaned_x is not None and getattr(demeaned_x, "size", 0):
            blocks.append(np.asarray(demeaned_x, dtype=np.float64).reshape(keep.size, -1))
        residual = np.column_stack(blocks)
        if raw.shape != residual.shape:
            return None

        weights = getattr(fit, "_weights", None)
        weights = (
            np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights is not None
            else None
        )
        codes = np.column_stack(
            [
                pd.factorize(fe_frame.iloc[:, j], sort=True)[0]
                for j in range(fe_frame.shape[1])
            ]
        )
        return float(np.max(external_normal_residuals(codes, raw, residual, weights=weights)))
    except Exception as error:  # noqa: BLE001 - reported, then recorded as absent
        _warn_once(f"external eta unavailable: {type(error).__name__}: {error}")
        return None


def _retained_rows(fit) -> int | None:
    """Rows the backend kept after singleton dropping.

    Recorded per trial because a comparison across backends is only a
    comparison if they retained the same sample.
    """
    value = getattr(fit, "_N", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fit_converged(fit) -> bool:
    """Read the convergence flag exposed by current PyFixest models."""
    return bool(getattr(fit, "convergence", getattr(fit, "_convergence", True)))


def _normalize_vcov(vcov: str | dict[str, str]) -> str:
    """Normalize vcov spec to a simple string for subprocess backends.

    Returns "iid", "hetero", or "cluster:<colname>".
    """
    if isinstance(vcov, dict) and "CRV1" in vcov:
        return f"cluster:{vcov['CRV1']}"
    return vcov


def _read_data_columns(data_path: Path, columns: list[str]) -> pd.DataFrame:
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path, usecols=columns)
    return pd.read_parquet(data_path, columns=columns)


# Solver settings are pinned here rather than left to package defaults, so that
# an upstream default change cannot silently alter a published timing. The MAP
# and LSMR numbers are not comparable to each other: the two monitor different
# convergence quantities. See PROTOCOL.md, section 5.
MAP_SETTINGS = {
    "backend": "rust",
    "fixef_tol": 1e-6,
    "fixef_maxiter": 10_000,
}

# PyFixest passes max(fixef_atol, fixef_btol) to the within backend, so the two
# must be set explicitly and equally or the effective tolerance is whichever
# default happens to be larger.
LSMR_SETTINGS = {
    "backend": "within",
    "fixef_atol": 1e-8,
    "fixef_btol": 1e-8,
    "fixef_maxiter": 1_000,
}

WITHIN_PRECONDITIONERS = ("off", "diagonal", "additive")

# pf.LsmrDemeaner(preconditioner="auto") resolves to "additive" for the within
# backend. Naming it here keeps the `within` benchmark label pinned to a
# specific preconditioner even if that resolution changes upstream.
DEFAULT_WITHIN_PRECONDITIONER = "additive"

# Frozen by the 100K calibration pilot on 2026-07-26 (PROTOCOL.md section 5).
# The LSMR stopping rule bounds a relative normal-equation residual recovered
# from the bidiagonalization scalars, which is a different number from the
# externally recomputed eta. At the package default of 1e-8 the achieved eta
# runs from 2.4e-8 to 1.6e-6, so no configuration clears Gate A. 1e-12 is the
# loosest tolerance at which all three clear it on both pilot designs, and it
# is what the mechanism ablation uses so its iteration counts are compared at
# matched accuracy rather than at matched nominal tolerance.
MECHANISM_LSMR_TOL = 1e-12
MECHANISM_MAP_TOL = 1e-10

# The ablation also equalizes the iteration budget. The package defaults give
# MAP 10,000 iterations and LSMR 1,000, and at 1M the first ablation run showed
# what that asymmetry does: within-off failed 30 of 33 trials, every one of them
# at 1,000 iterations, while rust-map was allowed ten times as many. "Removing
# the preconditioner does not by itself remove the slow directions" cannot rest
# on a censoring that the budget produced, so the mechanism arm gets MAP's cap
# and a run that still fails to converge fails on its own merits.
MECHANISM_MAXITER = 10_000


def _demeaner_from_backend(
    backend: str, *, tol: float | None = None, maxiter: int | None = None
):
    """Map a benchmark backend name to a typed demeaner configuration.

    Recognised names:

    - ``rust``: unaccelerated Rust MAP.
    - ``within-off`` / ``within-diagonal`` / ``within-additive``: LSMR with the
      named preconditioner, for the same-code mechanism ablation.
    - ``within``: alias for the documented default, so that existing result
      files and table labels keep working.
    - ``torch_cpu`` / ``torch_mps`` / ``torch_cuda``: the Torch LSMR backends,
      which carry their own built-in diagonal preconditioner.

    ``tol`` and ``maxiter`` override the pinned stopping rule. Cross-package
    tables leave both unset so that each package runs at its documented default;
    the mechanism ablation sets them so every configuration is compared at
    matched accuracy under one iteration budget.
    """
    import pyfixest as pf

    if backend == "rust":
        settings = dict(MAP_SETTINGS)
        if tol is not None:
            settings["fixef_tol"] = tol
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.MapDemeaner(**settings)

    if backend == "within":
        backend = f"within-{DEFAULT_WITHIN_PRECONDITIONER}"

    if backend.startswith("within-"):
        preconditioner = backend[len("within-") :]
        if preconditioner not in WITHIN_PRECONDITIONERS:
            raise ValueError(
                f"Unknown within preconditioner {preconditioner!r}; "
                f"expected one of {WITHIN_PRECONDITIONERS}"
            )
        settings = dict(LSMR_SETTINGS)
        if tol is not None:
            # Both must move together: PyFixest passes their maximum through.
            settings["fixef_atol"] = tol
            settings["fixef_btol"] = tol
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.LsmrDemeaner(preconditioner=preconditioner, **settings)

    if backend == "torch_cpu":
        return pf.LsmrDemeaner(backend="torch", device="cpu")
    if backend == "torch_mps":
        return pf.LsmrDemeaner(backend="torch", device="mps", precision="float32")
    if backend == "torch_cuda":
        return pf.LsmrDemeaner(backend="torch", device="cuda")
    raise ValueError(f"Unknown demeaner backend: {backend!r}")


class PyFeolsBenchmarkerFullApi:
    """Benchmark one pf.feols() call with the selected demeaning backend."""

    def __init__(
        self,
        name: str,
        demeaner_backend: str,
        *,
        tol: float | None = None,
        maxiter: int | None = None,
        repetitions: int | None = None,
    ):
        self._name = name
        self._demeaner_backend = demeaner_backend
        self._tol = tol
        self._maxiter = maxiter
        # None selects the count from the first trial's runtime.
        self._repetitions = repetitions

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        import pyfixest as pf

        demeaner = _demeaner_from_backend(
            self._demeaner_backend, tol=self._tol, maxiter=self._maxiter
        )

        results: list[FeolsResult] = []

        all_cols = [spec.depvar, *spec.covariates, *spec.fe_cols]
        if isinstance(spec.vcov, dict) and "CRV1" in spec.vcov:
            cluster_col = spec.vcov["CRV1"]
            if cluster_col not in all_cols:
                all_cols.append(cluster_col)

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

                # One timed fit first, then as many more as the R1/R2/R3 rule
                # asks for at that runtime. The data frame is read once and
                # reused, so every repetition runs on one fixed sample rather
                # than a fresh draw (PROTOCOL.md sections 2 and 4).
                trials: list[tuple[float | None, object | None, str | None]] = []
                dataset_results: list[FeolsResult] = []
                planned = 1
                while len(trials) < planned:
                    try:
                        t0 = time.perf_counter()
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r"\d+ singleton fixed effect\(s\) dropped from the model\.",
                                category=UserWarning,
                            )
                            fit = pf.feols(
                                fml=spec.formula,
                                data=df,
                                vcov=spec.vcov,
                                copy_data=False,
                                store_data=False,
                                demeaner=demeaner,
                            )
                            if not _fit_converged(fit):
                                raise RuntimeError("PyFixest model did not converge")
                        elapsed = time.perf_counter() - t0
                        trials.append((elapsed, fit, None))
                    except Exception as exc:  # noqa: BLE001 - recorded, not raised
                        trials.append((None, None, str(exc)))
                        # A failing cell is not worth repeating; one attempt is
                        # enough to record the failure, and the burn-in already
                        # paid the setup cost.
                        planned = len(trials)
                        break
                    if len(trials) == 1:
                        planned = (
                            self._repetitions
                            if self._repetitions is not None
                            else repetitions_for_runtime(elapsed)
                        )

                for repetition, (elapsed, fit, error) in enumerate(trials):
                    if error is None:
                        result = _result_from_dataset(
                            dataset,
                            spec,
                            backend=self.name,
                            elapsed=elapsed,
                            success=True,
                            n_obs_override=n_obs_for_result,
                            repetition=repetition,
                            n_planned=planned,
                            n_retained=_retained_rows(fit),
                            # Computed after the clock stops, so it never enters
                            # the reported runtime. Deterministic given the fit,
                            # so only the first repetition pays for it.
                            max_eta=(
                                _external_eta(fit, df, spec.depvar, spec.covariates)
                                if repetition == 0
                                else None
                            ),
                        )
                    else:
                        result = _result_from_dataset(
                            dataset,
                            spec,
                            backend=self.name,
                            elapsed=None,
                            success=False,
                            error=error,
                            n_obs_override=n_obs_for_result,
                            repetition=repetition,
                            n_planned=planned,
                        )
                    results.append(result)
                    dataset_results.append(result)
            except Exception as exc:
                dataset_results = [
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                        n_obs_override=n_obs_for_result,
                        n_planned=1,
                    )
                ]
                results.extend(dataset_results)
            finally:
                del df
                _trim_process_memory(self._demeaner_backend)

            # Every repetition reaches the printer, so the live min/median/max
            # shows the spread the repetition rule exists to measure.
            for result in dataset_results:
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


# ---------------------------------------------------------------------------
# Subprocess-based benchmarkers (R / Julia)
# ---------------------------------------------------------------------------


def _parse_subprocess_output(
    *,
    datasets: list[BenchmarkDataset],
    spec: FeolsSpec,
    backend: str,
    completed_process: subprocess.CompletedProcess[str],
) -> list[FeolsResult]:
    # Index strictly by (dataset_id, iter_num). Both the feols/fepois and the
    # Correia manifests emit one JSON line per trial with its iteration number,
    # so a crashed run leaves the missing trials unmatched (and therefore failed)
    # rather than reusing another trial's timing for a trial that never ran.
    parsed_by_key: dict[tuple[str, int | None], dict] = {}

    for line in completed_process.stdout.splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            entry = json.loads(payload)
        except json.JSONDecodeError:
            continue
        dataset_id = entry.get("dataset_id")
        if isinstance(dataset_id, str):
            iter_num = _safe_cast(entry.get("iter_num"), int)
            parsed_by_key[(dataset_id, iter_num)] = entry

    # Warn when a successful subprocess omits one or more datasets.
    n_emitted = len(parsed_by_key)
    n_missing = len(datasets) - n_emitted
    if n_missing > 0 and completed_process.returncode == 0:
        warnings.warn(
            f"Subprocess returned results for {n_emitted}/{len(datasets)} datasets"
        )

    stderr_text = (completed_process.stderr or "").strip()
    # Store only the final 4,000 characters of stderr in each failed CSV row.
    if len(stderr_text) > 4_000:
        stderr_text = stderr_text[-4_000:]
    if completed_process.returncode != 0:
        default_error = f"Subprocess exited with code {completed_process.returncode}"
        if stderr_text:
            default_error = f"{default_error}: {stderr_text}"
    else:
        default_error = stderr_text or None
    results: list[FeolsResult] = []

    for dataset in datasets:
        entry = parsed_by_key.get((dataset.dataset_id, dataset.iter_num))
        if entry is None:
            missing_error = default_error or "The subprocess returned no result for this dataset."
            results.append(
                _result_from_dataset(
                    dataset,
                    spec,
                    backend=backend,
                    elapsed=None,
                    success=False,
                    error=missing_error,
                )
            )
            continue

        elapsed = _safe_cast(entry.get("time"), float)
        n_obs_override = _safe_cast(entry.get("n_obs"), int)

        results.append(
            _result_from_dataset(
                dataset,
                spec,
                backend=backend,
                elapsed=elapsed,
                success=_as_bool(entry.get("success"), default=elapsed is not None),
                error=entry.get("error"),
                n_obs_override=n_obs_override,
            )
        )

    return results


class SubprocessFeolsBenchmarker:
    """Generic subprocess backend for feols (R/Julia)."""

    def __init__(
        self,
        *,
        name: str,
        command_prefix: Sequence[str],
        script_path: Path,
    ):
        self._name = name
        self._command_prefix = tuple(command_prefix)
        self._script_path = script_path.resolve()

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        manifest = [
            {
                "dataset_id": dataset.dataset_id,
                "data_path": str(dataset.data_path.resolve()),
                "dgp": dataset.dgp,
                "n_obs": dataset.n_obs,
                "iter_type": dataset.iter_type,
                "iter_num": dataset.iter_num,
            }
            for dataset in datasets
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            result_log_path = Path(tmpdir) / "results.jsonl"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "formula": spec.formula,
                        "depvar": spec.depvar,
                        "covariates": spec.covariates,
                        "fe_cols": spec.fe_cols,
                        "vcov": spec.vcov,
                        "vcov_type": _normalize_vcov(spec.vcov),
                        # Julia PPML can run for several minutes. Write each result
                        # to a file in case the process exits before flushing stdout.
                        "result_log_path": str(result_log_path),
                    }
                ),
                encoding="utf-8",
            )

            command = [
                *self._command_prefix,
                str(self._script_path),
                str(config_path),
            ]

            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout_text, stderr_text = proc.communicate()
                stdout_lines = stdout_text.splitlines(keepends=True)
                stderr_lines = stderr_text.splitlines(keepends=True)
                if stderr_text:
                    sys.stderr.write(stderr_text)
                    sys.stderr.flush()
                if result_log_path.exists():
                    stdout_lines.extend(
                        result_log_path.read_text(encoding="utf-8").splitlines(keepends=True)
                    )
            except Exception as exc:
                return [
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                    )
                    for dataset in datasets
                ]

        completed = subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )
        return _parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend=self.name,
            completed_process=completed,
        )


_SCRIPT_DIR = Path(__file__).parent


class FixestFeolsBenchmarker(SubprocessFeolsBenchmarker):
    def __init__(
        self,
        name: str | Path | None = None,
        script_path: Path | None = None,
    ):
        if isinstance(name, Path):
            if script_path is not None:
                raise TypeError(
                    "script_path must not be provided twice for FixestFeolsBenchmarker."
                )
            script_path = name
            name = None
        super().__init__(
            name=name or "r.fixest",
            command_prefix=["Rscript"],
            script_path=(script_path or _SCRIPT_DIR / "feols_r.R"),
        )


class JuliaFeolsBenchmarker(SubprocessFeolsBenchmarker):
    def __init__(
        self,
        name: str | Path | None = None,
        script_path: Path | None = None,
    ):
        if isinstance(name, Path):
            if script_path is not None:
                raise TypeError(
                    "script_path must not be provided twice for JuliaFeolsBenchmarker."
                )
            script_path = name
            name = None
        super().__init__(
            name=name or "julia.FixedEffectModels",
            command_prefix=["julia"],
            script_path=(script_path or _SCRIPT_DIR / "feols_julia.jl"),
        )

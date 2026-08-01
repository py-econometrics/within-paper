"""Shared records, parsing, and progress output for benchmark adapters.

These helpers describe a benchmark result, not a particular solver. Keeping them
here lets the PyFixest and subprocess adapters share one contract without the
generic subprocess path importing a PyFixest module.
"""

from __future__ import annotations

import ctypes
import gc
import statistics
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from benchmarks.core.interfaces import BenchmarkDataset, FeolsResult, FeolsSpec

_MIN_DGP_WIDTH = 16
_T = TypeVar("_T")


def trim_process_memory(demeaner_backend: str) -> None:
    """Return unused Python and native allocator memory after a large case."""
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


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.3f}s"


def dgp_width(datasets: list[BenchmarkDataset]) -> int:
    return max(
        _MIN_DGP_WIDTH, max((len(dataset.dgp) for dataset in datasets), default=_MIN_DGP_WIDTH)
    )


def time_columns(results: list[FeolsResult]) -> tuple[str, str]:
    times = [result.time for result in results if result.success and result.time is not None]
    if times:
        minimum, middle, maximum = min(times), statistics.median(times), max(times)
        columns = (
            f"{format_time(minimum):>10} {format_time(middle):>10} "
            f"{format_time(maximum):>10}"
        )
        return columns, "ok"
    errors = [result.error for result in results if result.error]
    status = errors[0][:30] if errors else "FAIL"
    columns = f"{'—':>10} {'—':>10} {'—':>10}"
    return columns, status


class TablePrinter:
    """Format live benchmark rows with a DGP column wide enough for their names."""

    def __init__(self, width: int):
        self._width = width
        self._header = (
            f"{'dgp':<{width}} {'k':>3} {'n_obs':>12} {'n_fe':>4} "
            f"{'min':>10} {'median':>10} {'max':>10}  status"
        )
        self._separator = "-" * len(self._header)

    def print_header(self, name: str) -> None:
        print(f"\n  {name}", flush=True)
        print(f"  {self._separator}", flush=True)
        print(f"  {self._header}", flush=True)
        print(f"  {self._separator}", flush=True)

    def print_row(self, results: list[FeolsResult]) -> None:
        first = results[0]
        prefix = (
            f"{first.dgp:<{self._width}} {first.model_k:>3} "
            f"{first.n_obs:>12,} {first.n_fe:>4}"
        )
        columns, status = time_columns(results)
        print(f"  {prefix} {columns}  {status}", flush=True)


def group_key(result: FeolsResult) -> tuple[str, int, int, int]:
    return (result.dgp, result.model_k, result.n_obs, result.n_fe)


def result_from_dataset(
    dataset: BenchmarkDataset,
    spec: FeolsSpec,
    *,
    backend: str,
    elapsed: float | None,
    success: bool,
    error: str | None = None,
    n_obs_override: int | None = None,
    **diagnostics: Any,
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


def preconditioner_build_s(fit: Any) -> float | None:
    """Read preconditioner setup time when a fitted model exposes it."""
    preconditioner = getattr(fit, "preconditioner", None)
    if preconditioner is None:
        return None
    value = getattr(preconditioner, "build_time_seconds", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def beta_x1(fit: Any) -> float | None:
    """Read the x1 coefficient from the current PyFixest public result shape."""
    try:
        coefficient = fit.coef()
        names = [str(name) for name in list(getattr(fit, "_coefnames", []) or [])]
        values = coefficient.tolist() if hasattr(coefficient, "tolist") else coefficient
        values = [float(value) for value in values]
        if "x1" in names:
            return values[names.index("x1")]
        if len(values) == 1:
            return values[0]
    except Exception:  # noqa: BLE001 - optional diagnostics must not fail a timing
        return None
    return None


def safe_cast(value: Any, type_fn: Callable[[Any], _T]) -> _T | None:
    if value is None:
        return None
    try:
        return type_fn(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return bool(value)


def fit_converged(fit: Any) -> bool:
    """Read whichever convergence flag a current PyFixest model exposes."""
    for name in ("convergence", "_convergence", "converged"):
        value = getattr(fit, name, None)
        if value is not None:
            return bool(value)
    return True


def retained_rows(fit: Any) -> int | None:
    """Rows a fitted model retained after singleton dropping."""
    value = getattr(fit, "_N", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_vcov(vcov: str | dict[str, str]) -> str:
    """Normalize a covariance specification for external driver scripts."""
    if isinstance(vcov, dict):
        return f"cluster:{vcov['CRV1']}"
    return vcov


def read_data_columns(data_path: Path, columns: list[str]) -> pd.DataFrame:
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path, usecols=columns)
    return pd.read_parquet(data_path, columns=columns)


def serialize_result(result: FeolsResult) -> dict[str, Any]:
    """Validate and serialize a result before it reaches a CSV file."""
    problems = result.validate()
    if problems:
        raise ValueError(
            f"invalid result for {result.backend}/{result.source_dataset_id}: "
            + "; ".join(problems)
        )
    return asdict(result)

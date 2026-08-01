"""One thread-count contract for every measured benchmark backend."""

from __future__ import annotations

import os

from benchmarks.core.paths import JULIA_ENV


def benchmark_threads() -> int:
    """Read the one user-facing benchmark thread setting."""
    value = os.environ.get("BENCH_THREADS", "")
    try:
        threads = int(value)
    except ValueError:
        raise RuntimeError(
            "BENCH_THREADS must be set to a positive integer before running benchmarks"
        ) from None
    if threads < 1:
        raise RuntimeError(
            "BENCH_THREADS must be set to a positive integer before running benchmarks"
        )
    return threads


def configure_benchmark_runtime() -> int:
    """Configure Julia and Rayon from ``BENCH_THREADS`` before solver import."""
    threads = benchmark_threads()
    value = str(threads)
    os.environ["RAYON_NUM_THREADS"] = value
    os.environ["JULIA_NUM_THREADS"] = value
    os.environ["JULIA_PROJECT"] = str(JULIA_ENV)
    return threads

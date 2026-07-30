"""Timing rules from PROTOCOL.md section 4.

The repetition counts exist because the invariance claim turns on separating
timings that differ by tens of milliseconds. Three trials cannot support that,
and "median among converged trials" over three trials is a selected estimator:
a cell reported as 52.4s (1/3) is the best of three by construction.

This module holds the rules and the summary statistics. Wiring the adaptive
repetition count into the harness additionally requires the aggregation in
scripts/paper_results.py to stop assuming exactly three trials with distinct
iter_num values; see the open items in PROTOCOL.md.
"""

from __future__ import annotations

import gc
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from typing import Sequence

# (upper runtime bound in seconds, timed repetitions). PROTOCOL.md rules
# R1/R2/R3. The lower end of each published band is used: the bands are
# 20-30, 7-10, and 3-5, and taking the minimum keeps a full sweep affordable
# while still supporting the claims the ledger attaches to each.
REPETITION_RULES: tuple[tuple[float, int], ...] = (
    (1.0, 20),
    (10.0, 7),
    (float("inf"), 3),
)


def repetitions_for_runtime(seconds: float) -> int:
    """Timed repetitions for a cell that takes roughly ``seconds``.

    Chosen from a discarded burn-in trial, so the count adapts to the cell
    rather than to a guess made before the run.
    """
    if seconds < 0:
        raise ValueError(f"runtime must be nonnegative, got {seconds}")
    for upper, count in REPETITION_RULES:
        if seconds < upper:
            return count
    raise AssertionError("REPETITION_RULES must end with an unbounded band")


@dataclass(frozen=True)
class TimingSummary:
    """Median with spread, plus the counts the table note has to carry."""

    median_s: float | None
    iqr_s: float | None
    min_s: float | None
    max_s: float | None
    n_attempted: int
    n_converged: int

    @property
    def is_complete(self) -> bool:
        return self.n_attempted > 0 and self.n_converged == self.n_attempted


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, so a 20-run IQR is not rounded to a gap."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_times(
    times: Sequence[float | None], *, n_attempted: int | None = None
) -> TimingSummary:
    """Summarize timed trials, keeping failures in the denominator.

    ``times`` carries one entry per attempted trial, with ``None`` for a trial
    that failed or was capped. Failures are never dropped before the median:
    they are counted, and the count is reported beside it.
    """
    attempted = len(times) if n_attempted is None else n_attempted
    converged = [value for value in times if value is not None]
    if not converged:
        return TimingSummary(None, None, None, None, attempted, 0)
    return TimingSummary(
        median_s=float(median(converged)),
        iqr_s=float(_quantile(converged, 0.75) - _quantile(converged, 0.25)),
        min_s=float(min(converged)),
        max_s=float(max(converged)),
        n_attempted=attempted,
        n_converged=len(converged),
    )


@dataclass
class Elapsed:
    """Wall time of a timed block, filled in when the block exits."""

    seconds: float | None = None


@contextmanager
def timed(*, collect: bool = True):
    """Time a block on the perf counter, after collecting garbage.

    The collection matters more than it looks. Without it a fit inherits
    whatever the previous fit left for the collector to free, so the first
    backend in a sweep is charged for the last one's garbage. Every driver
    that got this right did it by hand; the ones that forgot produced timings
    that moved when the backend order changed.
    """
    if collect:
        gc.collect()
    result = Elapsed()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


def randomized_order(items: Sequence, seed: int) -> list:
    """Shuffle backends so thermal drift does not align with a backend.

    Seeded, so a run is reproducible and the order can be recorded with the
    results rather than being unknown after the fact.
    """
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled

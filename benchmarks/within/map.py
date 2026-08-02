"""Method of alternating projections with explicit sweep counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MapResult:
    demeaned: np.ndarray
    converged: list[bool]
    iterations: list[int]


def _subtract_mean(
    values: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    group_weights: np.ndarray,
) -> None:
    sums = np.zeros_like(group_weights)
    np.add.at(sums, groups, weights * values)
    means = np.zeros_like(sums)
    positive = group_weights > 0
    means[positive] = sums[positive] / group_weights[positive]
    values -= means[groups]


def map_demean_with_sweeps(
    rhs: np.ndarray,
    categories: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> MapResult:
    """Demean each column and record full MAP sweeps."""
    values = np.array(rhs, dtype=float, copy=True, order="F")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    cats = np.asarray(categories, dtype=np.int64)
    sample_weights = np.ones(len(cats)) if weights is None else np.asarray(weights, dtype=float)
    group_weights = []
    for factor in range(cats.shape[1]):
        totals = np.zeros(int(cats[:, factor].max()) + 1)
        np.add.at(totals, cats[:, factor], sample_weights)
        group_weights.append(totals)

    converged, iterations = [], []
    for column in range(values.shape[1]):
        current = values[:, column]
        done = False
        sweep = 0
        for sweep in range(1, maxiter + 1):
            previous = current.copy()
            for factor in range(cats.shape[1]):
                _subtract_mean(
                    current, sample_weights, cats[:, factor], group_weights[factor]
                )
            if float(np.max(np.abs(current - previous))) < tol:
                done = True
                break
        converged.append(done)
        iterations.append(sweep)
    return MapResult(values, converged, iterations)

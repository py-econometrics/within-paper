"""MAP sweep diagnostics with an explicit censoring convention.

PyFixest's Rust MAP path returns only ``(demeaned, converged)``. The mechanism
figure needs the sweep count, so this module runs a counting MAP on the same
preprocessed sample. The algorithm matches the standard factor-by-factor
weighted group-mean update; the only addition is the iteration counter.

Censoring convention (plan item 12 / PROTOCOL.md):

- A run that reaches ``maxiter`` without meeting ``tol`` is **capped**.
- Capped runs keep ``converged=False``, ``iterations=maxiter``, and
  ``censoring="capped"``.
- Failed runs (empty sample, zero weights, numerical errors) keep
  ``converged=False``, ``iterations=None``, and ``censoring="failed"``.
- Converged runs keep ``censoring="none"``.
- Plots mark capped points rather than dropping them. Never treat a missing
  marker as "did not run".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MapSweepResult:
    demeaned: NDArray[np.float64]
    converged: list[bool]
    iterations: list[int | None]
    censoring: list[str]
    maxiter: int
    tol: float

    @property
    def any_capped(self) -> bool:
        return any(flag == "capped" for flag in self.censoring)

    @property
    def n_converged(self) -> int:
        return sum(1 for ok in self.converged if ok)

    def summary_row(self) -> dict:
        finite_iters = [i for i in self.iterations if i is not None]
        return {
            "map_maxiter": self.maxiter,
            "map_tol": self.tol,
            "map_n_rhs": len(self.converged),
            "map_n_converged": self.n_converged,
            "map_n_capped": sum(1 for c in self.censoring if c == "capped"),
            "map_n_failed": sum(1 for c in self.censoring if c == "failed"),
            "map_iterations_median": (
                float(np.median(finite_iters)) if finite_iters else None
            ),
            "map_iterations_max": (
                int(max(finite_iters)) if finite_iters else None
            ),
            "map_iterations_sum": (
                int(sum(finite_iters)) if finite_iters else None
            ),
            "map_any_capped": self.any_capped,
            "map_censoring": ",".join(self.censoring),
        }


def _factor_weights(
    sample_weights: NDArray[np.float64], group_ids: NDArray[np.int64]
) -> NDArray[np.float64]:
    n_groups = int(group_ids.max()) + 1 if group_ids.size else 0
    weights = np.zeros(n_groups, dtype=np.float64)
    np.add.at(weights, group_ids, sample_weights)
    return weights


def _subtract_group_mean(
    x: NDArray[np.float64],
    sample_weights: NDArray[np.float64],
    group_ids: NDArray[np.int64],
    group_weights: NDArray[np.float64],
) -> None:
    sums = np.zeros_like(group_weights)
    np.add.at(sums, group_ids, sample_weights * x)
    # Empty groups stay at zero weight; they never appear in group_ids.
    means = np.zeros_like(sums)
    positive = group_weights > 0
    means[positive] = sums[positive] / group_weights[positive]
    x -= means[group_ids]


def map_demean_with_sweeps(
    rhs: NDArray[np.floating],
    categories: NDArray,
    weights: NDArray[np.floating] | None = None,
    *,
    tol: float = 1e-6,
    maxiter: int = 10_000,
) -> MapSweepResult:
    """Factor-by-factor MAP demeaning that records sweeps per right-hand side.

    One sweep is one full pass over all absorbed factors. Convergence uses the
    same absolute change test as the Numba reference: max_i |x_i^{k} - x_i^{k-1}|
    < tol.
    """
    y = np.array(rhs, dtype=np.float64, copy=True, order="F")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    cats = np.asarray(categories)
    if cats.ndim != 2:
        raise ValueError(f"categories must be 2-d, got shape {cats.shape}")
    cats = cats.astype(np.int64, copy=False)
    n_obs, n_factors = cats.shape
    if y.shape[0] != n_obs:
        raise ValueError(f"rhs rows {y.shape[0]} != categories rows {n_obs}")
    if n_obs == 0:
        raise ValueError("empty sample")

    if weights is None:
        sample_weights = np.ones(n_obs, dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if sample_weights.shape[0] != n_obs:
            raise ValueError("weights length does not match n_obs")

    group_weights = [_factor_weights(sample_weights, cats[:, j]) for j in range(n_factors)]

    converged: list[bool] = []
    iterations: list[int | None] = []
    censoring: list[str] = []
    demeaned = np.empty_like(y)

    for col in range(y.shape[1]):
        try:
            x_curr = y[:, col].copy()
            x_prev = np.empty_like(x_curr)
            done = False
            n_iter = 0
            for n_iter in range(1, maxiter + 1):
                x_prev = x_curr.copy()
                for factor in range(n_factors):
                    _subtract_group_mean(
                        x_curr,
                        sample_weights,
                        cats[:, factor],
                        group_weights[factor],
                    )
                if float(np.max(np.abs(x_curr - x_prev))) < tol:
                    done = True
                    break
            demeaned[:, col] = x_curr
            if done:
                converged.append(True)
                iterations.append(n_iter)
                censoring.append("none")
            else:
                converged.append(False)
                iterations.append(maxiter)
                censoring.append("capped")
        except Exception:
            demeaned[:, col] = np.nan
            converged.append(False)
            iterations.append(None)
            censoring.append("failed")

    return MapSweepResult(
        demeaned=demeaned,
        converged=converged,
        iterations=iterations,
        censoring=censoring,
        maxiter=maxiter,
        tol=tol,
    )

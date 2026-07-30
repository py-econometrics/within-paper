"""External residual and projection-accuracy checks for absorbed least squares.

Three distinct quantities, never referred to by the same word (PROTOCOL.md §5):

1. Internal solver residual — whatever the backend reports.
2. External normal-equation residual η — recomputed from weighted group sums.
3. Projection error δ against a tight reference demeaned vector.

For right-hand side μ_j with A = W^{1/2} D, b_j = W^{1/2} μ_j, and
e_j = b_j − A α̂_j = W^{1/2} (μ_j − D α̂_j):

    η_j = ||A' e_j|| / max( ||A' b_j||, eps * ||A||_F * ||b_j|| )
    δ_j = ||e_j − e_j★|| / max( ||b_j||, eps )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# Gate A (provisional until the 100K pilot freezes it). PROTOCOL.md §5.
GATE_A_ETA = 1e-8
GATE_A_DELTA = 1e-7
GATE_A_SLOPE_SE = 1e-4
DEFAULT_EPS = 1e-12


def _as_2d(array: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"expected a 1-d or 2-d array, got shape {values.shape}")
    return values


def _as_categories(categories: NDArray) -> NDArray[np.int64]:
    cats = np.asarray(categories)
    if cats.ndim != 2:
        raise ValueError(f"categories must be 2-d, got shape {cats.shape}")
    if cats.shape[0] == 0:
        raise ValueError("categories must contain at least one observation")
    return cats.astype(np.int64, copy=False)


def _as_weights(n_obs: int, weights: NDArray[np.floating] | None) -> NDArray[np.float64]:
    if weights is None:
        return np.ones(n_obs, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != n_obs:
        raise ValueError(
            f"weights length {values.shape[0]} does not match n_obs={n_obs}"
        )
    if np.any(values < 0):
        raise ValueError("weights must be nonnegative")
    return values


def weighted_group_sums(
    categories: NDArray,
    values: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> list[NDArray[np.float64]]:
    """Return D' W v as one dense vector per absorbed factor."""
    cats = _as_categories(categories)
    rhs = _as_2d(values)
    n_obs, n_factors = cats.shape
    if rhs.shape[0] != n_obs:
        raise ValueError(
            f"values rows {rhs.shape[0]} do not match categories rows {n_obs}"
        )
    w = _as_weights(n_obs, weights)
    weighted = rhs * w[:, None]
    sums: list[NDArray[np.float64]] = []
    for factor in range(n_factors):
        levels = cats[:, factor]
        n_levels = int(levels.max()) + 1 if n_obs else 0
        out = np.zeros((n_levels, rhs.shape[1]), dtype=np.float64)
        np.add.at(out, levels, weighted)
        sums.append(out)
    return sums


def frobenius_norm_A(
    categories: NDArray,
    weights: NDArray[np.floating] | None = None,
) -> float:
    """||A||_F for A = W^{1/2} D.

    Each observation contributes its weight once per absorbed factor, so
    ||A||_F^2 = n_factors * sum(w) when every row has a level in every factor.
    """
    cats = _as_categories(categories)
    w = _as_weights(cats.shape[0], weights)
    return float(np.sqrt(cats.shape[1] * np.sum(w)))


def external_normal_residuals(
    categories: NDArray,
    rhs: NDArray[np.floating],
    demeaned: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    *,
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """External normal-equation residual η_j for each right-hand side column."""
    cats = _as_categories(categories)
    mu = _as_2d(rhs)
    residual = _as_2d(demeaned)
    if mu.shape != residual.shape:
        raise ValueError(
            f"rhs shape {mu.shape} does not match demeaned shape {residual.shape}"
        )
    w = _as_weights(cats.shape[0], weights)
    sqrt_w = np.sqrt(w)

    # A'e = D' W (μ − Dα) and A'b = D' W μ, evaluated as weighted group sums.
    ate = weighted_group_sums(cats, residual, w)
    atb = weighted_group_sums(cats, mu, w)

    n_rhs = mu.shape[1]
    eta = np.empty(n_rhs, dtype=np.float64)
    norm_A = frobenius_norm_A(cats, w)
    for j in range(n_rhs):
        num = 0.0
        den_atb = 0.0
        for factor_sum in ate:
            num += float(np.dot(factor_sum[:, j], factor_sum[:, j]))
        for factor_sum in atb:
            den_atb += float(np.dot(factor_sum[:, j], factor_sum[:, j]))
        num = float(np.sqrt(num))
        den_atb = float(np.sqrt(den_atb))
        b_norm = float(np.linalg.norm(sqrt_w * mu[:, j]))
        den = max(den_atb, eps * norm_A * b_norm)
        eta[j] = num / den if den > 0 else num
    return eta


def projection_errors(
    demeaned: NDArray[np.floating],
    reference_demeaned: NDArray[np.floating],
    rhs: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    *,
    eps: float = DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Projection error δ_j of demeaned columns against a reference."""
    residual = _as_2d(demeaned)
    reference = _as_2d(reference_demeaned)
    mu = _as_2d(rhs)
    if residual.shape != reference.shape:
        raise ValueError(
            f"demeaned shape {residual.shape} does not match reference {reference.shape}"
        )
    if residual.shape != mu.shape:
        raise ValueError(
            f"demeaned shape {residual.shape} does not match rhs {mu.shape}"
        )
    w = _as_weights(residual.shape[0], weights)
    sqrt_w = np.sqrt(w)
    delta = np.empty(residual.shape[1], dtype=np.float64)
    for j in range(residual.shape[1]):
        diff = sqrt_w * (residual[:, j] - reference[:, j])
        b_norm = float(np.linalg.norm(sqrt_w * mu[:, j]))
        den = max(b_norm, eps)
        delta[j] = float(np.linalg.norm(diff) / den)
    return delta


def slope_se_difference(
    beta: NDArray[np.floating],
    beta_star: NDArray[np.floating],
    se_star: NDArray[np.floating],
) -> NDArray[np.float64]:
    """|β̂ − β̂★| / SE(β̂★) in units of reference standard errors."""
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    b_star = np.asarray(beta_star, dtype=np.float64).reshape(-1)
    se = np.asarray(se_star, dtype=np.float64).reshape(-1)
    if b.shape != b_star.shape or b.shape != se.shape:
        raise ValueError("beta, beta_star, and se_star must share one shape")
    if np.any(se <= 0):
        raise ValueError("reference standard errors must be strictly positive")
    return np.abs(b - b_star) / se


@dataclass(frozen=True)
class AccuracyRecord:
    """Accuracy metrics for one solve on one sample."""

    max_eta: float
    max_delta: float | None
    median_eta: float
    median_delta: float | None
    max_slope_se: float | None
    n_rhs: int
    gate_a_eta: bool
    gate_a_delta: bool | None
    gate_a_slope: bool | None

    @property
    def clears_gate_a(self) -> bool:
        """True only when all three Gate A components were measured and passed.

        An unmeasured component is not a passing one. Returning True for a
        record that never computed delta or slope agreement would let a
        benchmark that supplies neither report itself as gated.
        """
        return (
            self.gate_a_eta is True
            and self.gate_a_delta is True
            and self.gate_a_slope is True
        )

    @property
    def gate_a_components_measured(self) -> bool:
        """Whether every Gate A component was actually computed."""
        return self.gate_a_delta is not None and self.gate_a_slope is not None


def accuracy_record(
    *,
    categories: NDArray,
    rhs: NDArray[np.floating],
    demeaned: NDArray[np.floating],
    reference_demeaned: NDArray[np.floating] | None = None,
    weights: NDArray[np.floating] | None = None,
    beta: NDArray[np.floating] | None = None,
    beta_star: NDArray[np.floating] | None = None,
    se_star: NDArray[np.floating] | None = None,
    eps: float = DEFAULT_EPS,
) -> AccuracyRecord:
    """Assemble Gate A metrics for one residualized sample."""
    eta = external_normal_residuals(
        categories, rhs, demeaned, weights=weights, eps=eps
    )
    delta = None
    if reference_demeaned is not None:
        delta = projection_errors(
            demeaned, reference_demeaned, rhs, weights=weights, eps=eps
        )
    slope = None
    if beta is not None and beta_star is not None and se_star is not None:
        slope = slope_se_difference(beta, beta_star, se_star)

    max_eta = float(np.max(eta))
    max_delta = float(np.max(delta)) if delta is not None else None
    max_slope = float(np.max(slope)) if slope is not None else None
    return AccuracyRecord(
        max_eta=max_eta,
        max_delta=max_delta,
        median_eta=float(np.median(eta)),
        median_delta=float(np.median(delta)) if delta is not None else None,
        max_slope_se=max_slope,
        n_rhs=int(eta.shape[0]),
        gate_a_eta=max_eta <= GATE_A_ETA,
        gate_a_delta=(max_delta <= GATE_A_DELTA) if max_delta is not None else None,
        gate_a_slope=(max_slope <= GATE_A_SLOPE_SE) if max_slope is not None else None,
    )


def pair_edge_stats(categories: NDArray) -> list[dict[str, float | int]]:
    """Cross-tabulation size for each factor pair (fill proxy when factors unavailable).

    The published `within` build does not expose retained fill nonzeros. Edge
    counts and density still identify the dense-graph regime where setup is
    expensive, which is the mechanism in plan item 8.
    """
    cats = _as_categories(categories)
    n_obs, n_factors = cats.shape
    stats: list[dict[str, float | int]] = []
    for q in range(n_factors):
        for r in range(q + 1, n_factors):
            n_q = int(cats[:, q].max()) + 1
            n_r = int(cats[:, r].max()) + 1
            # Unique undirected edges in the bipartite co-occurrence graph.
            packed = cats[:, q].astype(np.int64) * np.int64(n_r) + cats[:, r].astype(
                np.int64
            )
            n_edges = int(np.unique(packed).size)
            max_edges = n_q * n_r
            density = float(n_edges / max_edges) if max_edges else 0.0
            stats.append(
                {
                    "factor_q": q,
                    "factor_r": r,
                    "n_q_levels": n_q,
                    "n_r_levels": n_r,
                    "n_edges": n_edges,
                    "max_edges": max_edges,
                    "density": density,
                    "n_obs": n_obs,
                }
            )
    return stats

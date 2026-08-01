"""Accuracy measurements for absorbed least-squares fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _matrix(values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _weights(n_obs: int, values: NDArray[np.floating] | None) -> NDArray[np.float64]:
    return np.ones(n_obs) if values is None else np.asarray(values, dtype=float).reshape(-1)


def _group_sums(
    categories: NDArray,
    values: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> list[NDArray[np.float64]]:
    cats = np.asarray(categories, dtype=np.int64)
    rhs = _matrix(values)
    weighted = rhs * _weights(len(cats), weights)[:, None]
    sums = []
    for factor in range(cats.shape[1]):
        out = np.zeros((int(cats[:, factor].max()) + 1, rhs.shape[1]))
        np.add.at(out, cats[:, factor], weighted)
        sums.append(out)
    return sums


def external_normal_residuals(
    categories: NDArray,
    rhs: NDArray[np.floating],
    demeaned: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Return the normal-equation residual for each right-hand side."""
    cats = np.asarray(categories, dtype=np.int64)
    raw = _matrix(rhs)
    residual = _matrix(demeaned)
    w = _weights(len(cats), weights)
    ate = _group_sums(cats, residual, w)
    atb = _group_sums(cats, raw, w)
    norm_a = float(np.sqrt(cats.shape[1] * w.sum()))
    result = np.empty(raw.shape[1])
    for column in range(raw.shape[1]):
        numerator = np.sqrt(sum(np.dot(x[:, column], x[:, column]) for x in ate))
        atb_norm = np.sqrt(sum(np.dot(x[:, column], x[:, column]) for x in atb))
        rhs_norm = np.linalg.norm(np.sqrt(w) * raw[:, column])
        denominator = max(atb_norm, eps * norm_a * rhs_norm)
        result[column] = numerator / denominator if denominator else numerator
    return result


def projection_errors(
    demeaned: NDArray[np.floating],
    reference: NDArray[np.floating],
    rhs: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Return projection error relative to a reference residualization."""
    result = _matrix(demeaned)
    target = _matrix(reference)
    raw = _matrix(rhs)
    root_w = np.sqrt(_weights(len(raw), weights))
    return np.array(
        [
            np.linalg.norm(root_w * (result[:, j] - target[:, j]))
            / max(np.linalg.norm(root_w * raw[:, j]), eps)
            for j in range(raw.shape[1])
        ]
    )


def accuracy_metrics(
    categories: NDArray,
    rhs: NDArray[np.floating],
    demeaned: NDArray[np.floating],
    reference: NDArray[np.floating] | None = None,
) -> dict[str, float | None]:
    eta = external_normal_residuals(categories, rhs, demeaned)
    delta = projection_errors(demeaned, reference, rhs) if reference is not None else None
    return {
        "max_eta": float(np.max(eta)),
        "max_delta": float(np.max(delta)) if delta is not None else None,
    }


def pair_edge_stats(categories: NDArray) -> list[dict[str, float | int]]:
    """Count unique edges and densities for each pair of factors."""
    cats = np.asarray(categories, dtype=np.int64)
    rows = []
    for left in range(cats.shape[1]):
        for right in range(left + 1, cats.shape[1]):
            n_left = int(cats[:, left].max()) + 1
            n_right = int(cats[:, right].max()) + 1
            packed = cats[:, left] * n_right + cats[:, right]
            n_edges = int(np.unique(packed).size)
            rows.append(
                {
                    "factor_q": left,
                    "factor_r": right,
                    "n_edges": n_edges,
                    "density": n_edges / (n_left * n_right),
                }
            )
    return rows

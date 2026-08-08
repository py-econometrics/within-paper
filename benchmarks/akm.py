# Adapted from bipartitepandas by Thibaut Lamadon, under the MIT License.

"""AKM data-generating process and the paper's twelve designs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AKMConfig:
    n_workers: int = 100_000
    n_firms: int = 50_000
    n_time: int = 10
    n_industries: int = 5
    var_alpha: float = 1.0
    var_psi: float = 0.5
    var_phi: float = 0.1
    var_epsilon: float = 1.0
    gamma: float = 1.0
    rho_size: float = 0.6
    rho: float = 1.0
    delta: float = 0.1
    lambda_: float = 0.8
    beta_x1: float = 0.5
    n_match_bins: int = 2_048


SCENARIOS = {
    "akm_sorting_1": {"delta": 1.0, "rho": 0.0},
    "akm_sorting_2": {"delta": 1.0, "rho": 20.0},
    "akm_sorting_3": {"delta": 1.0, "rho": 500.0},
    "akm_sorting_4": {"delta": 1.0, "rho": 2_000.0},
    "akm_sorting_5": {"delta": 1.0, "rho": 10_000.0},
    "akm_sorting_6": {"delta": 1.0, "rho": 150_000.0},
    "akm_mobility_1": {"delta": 1.0},
    "akm_mobility_2": {"delta": 0.5},
    "akm_mobility_3": {"delta": 0.05},
    "akm_mobility_4": {"delta": 0.01},
    "akm_mobility_5": {"delta": 0.005},
    "akm_mobility_6": {"delta": 0.001},
}


def make_akm_data(name: str) -> pd.DataFrame:
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "big") % 97
    return simulate_akm_panel(
        replace(AKMConfig(), **SCENARIOS[name]), seed=100_000_059 + offset
    )


def _balanced_groups(n_items: int, n_groups: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(np.arange(n_items) % n_groups)


def _couple_by_rank(
    rng: np.random.Generator,
    left: np.ndarray,
    right: np.ndarray,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal(len(left))
    z2 = correlation * z1 + np.sqrt(max(0.0, 1 - correlation**2)) * rng.standard_normal(len(right))
    left_order = np.argsort(z1)
    right_order = np.argsort(z2)
    left_out = np.empty_like(left)
    right_out = np.empty_like(right)
    left_out[left_order] = np.sort(left)
    right_out[right_order] = np.sort(right)
    return left_out, right_out


def _firm_size_weights(config: AKMConfig, rng: np.random.Generator) -> np.ndarray:
    draws = np.clip(rng.random(config.n_firms), 1e-12, 1 - 1e-12)
    weights = np.exp(-np.log(draws) / config.gamma)
    return weights / weights.sum()


def _alpha_bins(alpha: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    n_bins = min(n_bins, len(alpha))
    order = np.argsort(alpha, kind="mergesort")
    ids = np.empty(len(alpha), dtype=np.int16)
    ids[order] = np.arange(len(alpha)) * n_bins // len(alpha)
    centers = np.array([alpha[ids == index].mean() for index in range(n_bins)])
    return ids, centers


def _industry_weights(config: AKMConfig, firm_industries: np.ndarray) -> np.ndarray:
    if config.n_industries == 1:
        return np.ones((1, config.n_firms))
    weights = np.full(
        (config.n_industries, config.n_firms),
        (1 - config.lambda_) / (config.n_industries - 1),
    )
    for industry in range(config.n_industries):
        weights[industry, firm_industries == industry] = config.lambda_
    return weights


def _assignment_cdfs(
    config: AKMConfig,
    alpha_centers: np.ndarray,
    psi: np.ndarray,
    firm_weights: np.ndarray,
    firm_industries: np.ndarray,
) -> np.ndarray:
    tau2 = max(config.var_alpha + config.var_psi, 1e-12)
    industry_weights = _industry_weights(config, firm_industries)
    cdfs = np.empty((config.n_industries, len(alpha_centers), config.n_firms), dtype=np.float32)
    for industry in range(config.n_industries):
        for bin_id, alpha in enumerate(alpha_centers):
            scores = (
                np.exp(-0.5 * config.rho * (alpha - psi) ** 2 / tau2)
                * firm_weights
                * industry_weights[industry]
            )
            total = scores.sum()
            if not np.isfinite(total) or total <= 0:
                log_scores = (
                    -0.5 * config.rho * (alpha - psi) ** 2 / tau2
                    + np.log(firm_weights)
                    + np.log(industry_weights[industry])
                )
                scores = np.exp(log_scores - log_scores.max())
                total = scores.sum()
            cdf = np.cumsum(scores / total, dtype=np.float64)
            cdf[-1] = 1.0
            cdfs[industry, bin_id] = cdf
    return cdfs


def _worker_groups(
    industries: np.ndarray, bins: np.ndarray, n_industries: int, n_bins: int
) -> dict[tuple[int, int], np.ndarray]:
    groups = {}
    for industry in range(n_industries):
        for bin_id in range(n_bins):
            indices = np.flatnonzero((industries == industry) & (bins == bin_id))
            if indices.size:
                groups[industry, bin_id] = indices
    return groups


def _sample_firms(
    rng: np.random.Generator,
    cdf: np.ndarray,
    size: int,
    current: np.ndarray | None = None,
) -> np.ndarray:
    draws = np.searchsorted(cdf, rng.random(size), side="right")
    if current is None or len(cdf) == 1:
        return draws
    same = draws == current
    for _ in range(8):
        if not same.any():
            break
        draws[same] = np.searchsorted(cdf, rng.random(same.sum()), side="right")
        same = draws == current
    draws[same] = (current[same] + 1) % len(cdf)
    return draws


def simulate_akm_panel(config: AKMConfig, *, seed: int | None = None) -> pd.DataFrame:
    """Simulate the worker-firm panel used by the AKM experiments."""
    rng = np.random.default_rng(seed)
    raw_psi = rng.normal(scale=np.sqrt(config.var_psi), size=config.n_firms)
    raw_size = _firm_size_weights(config, rng)
    psi, firm_weights = _couple_by_rank(rng, raw_psi, raw_size, config.rho_size)
    firm_weights /= firm_weights.sum()
    firm_industries = _balanced_groups(config.n_firms, config.n_industries, rng)
    alpha = rng.normal(scale=np.sqrt(config.var_alpha), size=config.n_workers)
    worker_industries = rng.integers(0, config.n_industries, size=config.n_workers)
    worker_bins, centers = _alpha_bins(alpha, config.n_match_bins)
    cdfs = _assignment_cdfs(config, centers, psi, firm_weights, firm_industries)
    groups = _worker_groups(worker_industries, worker_bins, config.n_industries, len(centers))

    paths = np.empty((config.n_workers, config.n_time), dtype=np.int32)
    for (industry, bin_id), indices in groups.items():
        paths[indices, 0] = _sample_firms(rng, cdfs[industry, bin_id], len(indices))
    moves = rng.random((config.n_workers, config.n_time - 1)) < config.delta
    for period in range(1, config.n_time):
        paths[:, period] = paths[:, period - 1]
        for (industry, bin_id), indices in groups.items():
            movers = indices[moves[indices, period - 1]]
            if movers.size:
                paths[movers, period] = _sample_firms(
                    rng, cdfs[industry, bin_id], len(movers), paths[movers, period - 1]
                )

    n_obs = config.n_workers * config.n_time
    individual = np.repeat(np.arange(1, config.n_workers + 1), config.n_time)
    year = np.tile(np.arange(1, config.n_time + 1), config.n_workers)
    firm = paths.ravel() + 1
    x1 = rng.standard_normal(n_obs)
    y = (
        np.repeat(alpha, config.n_time)
        + psi[paths.ravel()]
        + np.tile(rng.normal(scale=np.sqrt(config.var_phi), size=config.n_time), config.n_workers)
        + config.beta_x1 * x1
        + rng.normal(scale=np.sqrt(config.var_epsilon), size=n_obs)
    )
    return pd.DataFrame(
        {"indiv_id": individual, "firm_id": firm, "year": year, "x1": x1, "y": y}
    )

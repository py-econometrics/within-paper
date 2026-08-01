"""Deterministic data used by the benchmark experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

FE_COLUMNS = ("indiv_id", "firm_id", "year")
BASE_DESIGNS = (("simple", 42), ("difficult", 43))
CORREIA_NAMES = (
    "credit2", "credit", "soccer", "synthetic-complete", "synthetic-uniform-easy",
    "synthetic-uniform-hard", "synthetic-uniform-harder", "synthetic-assortative",
    "synthetic-zigzag", "enron", "github", "patents", "workers", "schools", "directors",
)


def make_base_data(n_obs: int, design: str, seed: int) -> pd.DataFrame:
    """Generate the simple or difficult panel used in the paper."""
    rng = np.random.default_rng(seed)
    n_years = 10
    n_individuals = round(n_obs / n_years)
    n_firms = round(n_individuals / 23)
    actual_n = n_individuals * n_years

    individual = np.repeat(np.arange(1, n_individuals + 1), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_individuals)
    if design == "simple":
        firm = rng.integers(1, n_firms + 1, size=actual_n)
    elif design == "difficult":
        firm = np.resize(np.arange(1, n_firms + 1), actual_n)
    else:
        raise ValueError(f"unknown base design {design!r}")

    x = rng.standard_normal((actual_n, 10))
    mean = (
        x[:, 0]
        + rng.standard_normal(n_firms)[firm - 1]
        + rng.standard_normal(n_individuals)[individual - 1]
        + rng.standard_normal(n_years)[year - 1]
    )
    y = mean + rng.standard_normal(actual_n)
    theta = 0.5
    probability = theta / (theta + np.exp(y))
    values: dict[str, np.ndarray] = {
        "indiv_id": individual,
        "firm_id": firm,
        "year": year,
        "y": y,
        "negbin_y": rng.negative_binomial(theta, probability),
        "x1": x[:, 0],
    }
    return pd.DataFrame(values)


def solver_data(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = ("y", "x1"),
    fixed_effects: tuple[str, ...] = FE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a frame to the arrays expected by the standalone solver."""
    categories = np.column_stack(
        [pd.factorize(frame[name], sort=True)[0] for name in fixed_effects]
    ).astype(np.uint32)
    rhs = frame.loc[:, columns].to_numpy(dtype=float)
    return np.asfortranarray(categories), np.asfortranarray(rhs)


def drop_singletons(
    frame: pd.DataFrame, fixed_effects: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    """Drop the recursive singleton set used by PyFixest."""
    from pyfixest.core.detect_singletons import detect_singletons

    categories = np.column_stack(
        [pd.factorize(frame[name], sort=False)[0] for name in fixed_effects]
    ).astype(np.int64)
    dropped = detect_singletons(categories)
    return frame.loc[~dropped].reset_index(drop=True), int(dropped.sum())

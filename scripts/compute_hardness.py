"""Compute the Jochmans-Weidner Gap for every paper factor-pair design."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

from benchmarks.akm import SCENARIOS, make_akm_data
from benchmarks.data import (
    BASE_DESIGNS,
    CORREIA_NAMES,
    FE_COLUMNS,
    drop_singletons,
    make_base_data,
)
from benchmarks.memory import CELLS as MEMORY_CELLS
from benchmarks.ols.main import N_OBS as BASE_N_OBS

ROOT = Path(__file__).absolute().parents[1]
CORREIA = ROOT / "benchmarks" / "data" / "correia_data"
OUTPUT = ROOT / "results" / "runs" / "latest" / "hardness.csv"
DENSE_MAX_ENTRIES = 1_000_000


@dataclass(frozen=True)
class PairHardness:
    n_q_levels: int
    n_r_levels: int
    n_components: int
    lambda2_qr: float
    largest_component_obs_share: float
    largest_component_n_obs: int
    largest_component_n_q_levels: int
    largest_component_n_r_levels: int


def _top_two_singular_values(matrix: sp.csr_matrix) -> np.ndarray:
    rows, columns = matrix.shape
    if min(rows, columns) <= 64 or rows * columns <= DENSE_MAX_ENTRIES:
        return np.linalg.svd(matrix.toarray(), compute_uv=False)
    return svds(
        matrix,
        k=2,
        which="LM",
        tol=1e-10,
        maxiter=200_000,
        return_singular_vectors=False,
    )


def _component_lambda2(cooccurrence: sp.csr_matrix) -> float:
    """Return the normalized-Laplacian Gap of one connected component."""
    rows, columns = cooccurrence.shape
    if rows == 1 and columns == 1:
        return 2.0
    if min(rows, columns) == 1:
        return 1.0
    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    column_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()
    normalized = (
        sp.diags(1 / np.sqrt(row_sums))
        @ cooccurrence
        @ sp.diags(1 / np.sqrt(column_sums))
    ).tocsr()
    singular_values = np.sort(_top_two_singular_values(normalized))[::-1]
    sigma_2 = min(max(float(singular_values[1]), 0.0), 1.0)
    return max(1.0 - sigma_2, 0.0)


def pair_hardness(q: np.ndarray, r: np.ndarray) -> PairHardness:
    q_codes, _ = pd.factorize(q, sort=False)
    r_codes, _ = pd.factorize(r, sort=False)
    n_q, n_r = int(q_codes.max()) + 1, int(r_codes.max()) + 1
    cooccurrence = sp.coo_matrix(
        (np.ones(len(q_codes)), (q_codes, r_codes)), shape=(n_q, n_r)
    ).tocsr()
    cooccurrence.sum_duplicates()
    adjacency = sp.bmat([[None, cooccurrence], [cooccurrence.T, None]], format="csr")
    n_components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    q_labels, r_labels = labels[:n_q], labels[n_q:]
    largest: PairHardness | None = None
    largest_key: tuple[int, float, int] | None = None
    for component in range(n_components):
        q_mask, r_mask = q_labels == component, r_labels == component
        if not q_mask.any() or not r_mask.any():
            continue
        block = cooccurrence[q_mask][:, r_mask]
        n_obs = int(block.sum())
        lambda2 = _component_lambda2(block)
        # Select the component with the most retained observations. Equal-size
        # components use the smaller Gap, then the stable scipy component id.
        key = (n_obs, -lambda2, -component)
        if largest_key is None or key > largest_key:
            largest_key = key
            largest = PairHardness(
                n_q,
                n_r,
                n_components,
                lambda2,
                n_obs / len(q),
                n_obs,
                int(q_mask.sum()),
                int(r_mask.sum()),
            )
    if largest is None:
        raise ValueError("factor pair has no observed connected component")
    return largest


def _datasets():
    for name in CORREIA_NAMES:
        yield name, "correia", pd.read_csv(CORREIA / f"{name}.csv"), ("id1", "id2")
    for name in SCENARIOS:
        yield name, "akm", make_akm_data(name), FE_COLUMNS
    # The Gap is a property of the exact sample each experiment times, so the
    # observation counts come from the runners themselves: BASE_N_OBS is the
    # headline OLS size and MEMORY_CELLS is the memory benchmark's sizes. Keeping
    # one authority for each n stops the reported Gap from drifting to a design
    # that was never timed.
    for name, seed in BASE_DESIGNS:
        yield name, "base", make_base_data(BASE_N_OBS, name, seed), FE_COLUMNS
        for label, n_obs in MEMORY_CELLS:
            yield f"memory_{name}_{label}", "memory", make_base_data(n_obs, name, seed), FE_COLUMNS


def main() -> None:
    rows = []
    for name, kind, raw, fixed_effects in _datasets():
        started = time.perf_counter()
        frame, dropped = drop_singletons(raw, fixed_effects)
        for left, right in combinations(fixed_effects, 2):
            result = pair_hardness(frame[left].to_numpy(), frame[right].to_numpy())
            rows.append(
                {
                    "dataset_id": name, "kind": kind, "n_obs_raw": len(raw),
                    "n_obs": len(frame), "n_singletons_dropped": dropped,
                    "fe_a": left, "fe_b": right, **asdict(result),
                }
            )
        print(
            f"compute-hardness / Jochmans-Weidner Gap / {name}: "
            f"{time.perf_counter() - started:.3f} s",
            flush=True,
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

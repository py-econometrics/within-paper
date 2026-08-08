"""Compute pairwise spectral gaps for every paper design."""

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
PROPACK_MAX_MIN_DIM = 20_000


@dataclass(frozen=True)
class PairHardness:
    n_q_levels: int
    n_r_levels: int
    n_components: int
    rho_qr: float
    worst_component_obs_share: float
    worst_component_n_obs: int
    worst_component_n_q_levels: int
    worst_component_n_r_levels: int


def _top_two_singular_values(matrix: sp.csr_matrix) -> np.ndarray:
    rows, columns = matrix.shape
    if min(rows, columns) <= 64 or rows * columns <= DENSE_MAX_ENTRIES:
        return np.linalg.svd(matrix.toarray(), compute_uv=False)
    solvers = ["propack", "arpack"] if min(rows, columns) <= PROPACK_MAX_MIN_DIM else ["arpack"]
    errors = []
    for solver in solvers:
        try:
            return svds(
                matrix, k=2, which="LM", solver=solver, tol=1e-10,
                maxiter=200_000, return_singular_vectors=False,
            )
        except Exception as error:
            errors.append(f"{solver}: {error}")
    raise RuntimeError("singular-value calculation failed (" + "; ".join(errors) + ")")


def _component_rho(cooccurrence: sp.csr_matrix) -> float:
    if min(cooccurrence.shape) < 2:
        return 0.0
    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    column_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()
    normalized = (
        sp.diags(1 / np.sqrt(row_sums))
        @ cooccurrence
        @ sp.diags(1 / np.sqrt(column_sums))
    ).tocsr()
    singular_values = np.sort(_top_two_singular_values(normalized))[::-1]
    sigma_2 = min(max(float(singular_values[1]), 0.0), 1.0)
    return sigma_2**2


def pair_hardness(q: np.ndarray, r: np.ndarray) -> PairHardness:
    q_codes, _ = pd.factorize(q, sort=False)
    r_codes, _ = pd.factorize(r, sort=False)
    n_q, n_r = int(q_codes.max()) + 1, int(r_codes.max()) + 1
    cooccurrence = sp.coo_matrix(
        (np.ones(len(q_codes)), (q_codes, r_codes)), shape=(n_q, n_r)
    ).tocsr()
    cooccurrence.sum_duplicates()
    adjacency = sp.bmat([[None, cooccurrence], [cooccurrence.T, None]], format="csr")
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    q_labels, r_labels = labels[:n_q], labels[n_q:]
    worst = PairHardness(n_q, n_r, n_components, 0, 0, 0, 0, 0)
    for component in range(n_components):
        q_mask, r_mask = q_labels == component, r_labels == component
        if not q_mask.any() or not r_mask.any():
            continue
        block = cooccurrence[q_mask][:, r_mask]
        n_obs = int(block.sum())
        rho = _component_rho(block)
        if rho > worst.rho_qr:
            worst = PairHardness(
                n_q, n_r, n_components, rho, n_obs / len(q), n_obs,
                int(q_mask.sum()), int(r_mask.sum()),
            )
    return worst


def _datasets():
    for name in CORREIA_NAMES:
        yield name, "correia", pd.read_csv(CORREIA / f"{name}.csv"), ("id1", "id2")
    for name in SCENARIOS:
        yield name, "akm", make_akm_data(name), FE_COLUMNS
    # The gap is a property of the exact sample each experiment times, so the
    # observation counts come from the runners themselves: BASE_N_OBS is the
    # headline OLS size and MEMORY_CELLS is the memory benchmark's sizes. Keeping
    # one authority for each n stops the reported gap from drifting to a design
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
                    "one_minus_rho": 1 - result.rho_qr,
                }
            )
        print(
            f"compute-hardness / spectral gap / {name}: "
            f"{time.perf_counter() - started:.3f} s",
            flush=True,
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

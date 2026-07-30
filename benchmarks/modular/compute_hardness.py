"""Compute the pairwise spectral gaps used in the paper.

Derived from PyFixest commit f671ec83 (``data hardness checks``). For each
fixed-effect pair, the script removes singleton observations as the estimator
does, finds the bipartite connected components, and reports the component with
the largest nontrivial MAP contraction factor ``rho = sigma_2(H)^2``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[2]

from pyfixest.core.detect_singletons import detect_singletons
from benchmarks.modular.dgps import get_akm_sweep_scenario_names

CORREIA_DIR = ROOT / "data" / "correia_data"
DATA_DIR = ROOT / "benchmarks" / "data"
MEMORY_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "results" / "runs" / "latest" / "hardness.csv"
PROPACK_MAX_MIN_DIM = 20_000
# Use dense SVD when the normalized block has at most this many entries.
DENSE_MAX_ENTRIES = 1_000_000
CORREIA_DATASETS = (
    "credit2", "credit", "soccer", "synthetic-complete",
    "synthetic-uniform-easy", "synthetic-uniform-hard",
    "synthetic-uniform-harder", "synthetic-assortative", "synthetic-zigzag",
    "enron", "github", "patents", "workers", "schools", "directors",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    kind: str
    path: Path
    fe_cols: tuple[str, ...]
    reader: Callable[[Path, list[str]], pd.DataFrame]


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


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns)


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _factorize(values: np.ndarray) -> tuple[np.ndarray, int]:
    codes, _ = pd.factorize(values, sort=False)
    return codes.astype(np.int64, copy=False), int(codes.max()) + 1 if len(codes) else 0


def _top_two_singular_values(normalized: sp.csr_matrix) -> np.ndarray:
    """Return the two largest singular values of a normalized bipartite block.

    Iterative solvers can report a nonzero second singular value for rank-deficient
    matrices; it should be zero for a complete bipartite graph. For larger blocks
    below ``PROPACK_MAX_MIN_DIM``, try PROPACK first and fall back to ARPACK. Above
    that limit, skip PROPACK because SciPy's implementation can terminate the
    process on very large irregular inputs.
    """
    rows, cols = normalized.shape
    if min(rows, cols) <= 64 or rows * cols <= DENSE_MAX_ENTRIES:
        return np.linalg.svd(normalized.toarray(), compute_uv=False)
    solvers = (
        ["propack", "arpack"]
        if min(rows, cols) <= PROPACK_MAX_MIN_DIM
        else ["arpack"]
    )
    errors: list[str] = []
    for solver in solvers:
        try:
            return svds(
                normalized,
                k=2,
                which="LM",
                solver=solver,
                tol=1e-10,
                maxiter=200_000,
                return_singular_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 - try the next solver, then report
            errors.append(f"{solver.upper()}: {exc}")
    raise RuntimeError("singular-value calculation failed (" + "; ".join(errors) + ")")


def _component_rho(cooccurrence: sp.csr_matrix) -> float:
    """Return ``sigma_2(H)^2`` for one connected bipartite component."""
    if min(cooccurrence.shape) < 2:
        return 0.0
    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    col_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()
    normalized = (
        sp.diags(1.0 / np.sqrt(row_sums))
        @ cooccurrence
        @ sp.diags(1.0 / np.sqrt(col_sums))
    ).tocsr()
    singular_values = np.sort(_top_two_singular_values(normalized))[::-1]
    if len(singular_values) < 2:
        return 0.0
    sigma_2 = min(max(float(singular_values[1]), 0.0), 1.0)
    return sigma_2**2


def pair_hardness(q: np.ndarray, r: np.ndarray) -> PairHardness:
    q_codes, n_q = _factorize(q)
    r_codes, n_r = _factorize(r)
    cooccurrence = sp.coo_matrix(
        (np.ones(len(q_codes)), (q_codes, r_codes)), shape=(n_q, n_r)
    ).tocsr()
    cooccurrence.sum_duplicates()
    adjacency = sp.bmat([[None, cooccurrence], [cooccurrence.T, None]], format="csr")
    n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
    q_labels, r_labels = labels[:n_q], labels[n_q:]
    worst = PairHardness(n_q, n_r, n_components, 0.0, 0.0, 0, 0, 0)
    for component in range(n_components):
        q_mask, r_mask = q_labels == component, r_labels == component
        if not q_mask.any() or not r_mask.any():
            continue
        block = cooccurrence[q_mask][:, r_mask]
        n_obs = int(block.sum())
        if n_obs == 0:
            continue
        rho = _component_rho(block)
        if rho > worst.rho_qr:
            worst = PairHardness(
                n_q, n_r, n_components, rho, n_obs / len(q), n_obs,
                int(q_mask.sum()), int(r_mask.sum()),
            )
    return worst


def _drop_singletons(frame: pd.DataFrame, fe_cols: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    codes = np.column_stack([
        pd.factorize(frame[column].to_numpy(), sort=False)[0] for column in fe_cols
    ]).astype(np.int64, copy=False)
    singleton_mask = detect_singletons(codes)
    n_dropped = int(singleton_mask.sum())
    return frame.loc[~singleton_mask].reset_index(drop=True), n_dropped


def enumerate_datasets() -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for dataset in CORREIA_DATASETS:
        path = CORREIA_DIR / f"{dataset}.csv"
        if path.exists():
            specs.append(DatasetSpec(dataset, "correia", path, ("id1", "id2"), _read_csv))
    for dgp_name in get_akm_sweep_scenario_names():
        path = DATA_DIR / f"{dgp_name}_1000000_k1_iter_1.parquet"
        if path.exists():
            specs.append(
                DatasetSpec(
                    path.stem,
                    "akm",
                    path,
                    ("indiv_id", "firm_id", "year"),
                    _read_parquet,
                )
            )
    for family in ("simple", "difficult"):
        # The OLS runtime table times the 10M designs, so its gap diagnostic is
        # computed on the same 10M graphs rather than a 1M proxy.
        path = DATA_DIR / f"{family}_10000000_k1_iter_1.parquet"
        if path.exists():
            specs.append(DatasetSpec(path.stem, "fixest-dgp", path, ("indiv_id", "firm_id", "year"), _read_parquet))
        for label, n_obs in (("100k", 100_000), ("1m", 1_000_000)):
            memory_path = MEMORY_DATA_DIR / f"{family}_{label}.parquet"
            if memory_path.exists():
                specs.append(
                    DatasetSpec(
                        f"memory_{family}_{n_obs}",
                        "memory-dgp",
                        memory_path,
                        ("indiv_id", "firm_id", "year"),
                        _read_parquet,
                    )
                )
    ids = [spec.dataset_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Dataset IDs for the spectral-gap calculation are not unique")
    return specs


def compute(specs: list[DatasetSpec], keep_singletons: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        started = time.perf_counter()
        frame = spec.reader(spec.path, list(spec.fe_cols))
        n_obs_raw = len(frame)
        if keep_singletons:
            n_dropped = 0
        else:
            frame, n_dropped = _drop_singletons(frame, spec.fe_cols)
        print(f"[hardness] {spec.dataset_id}: {len(frame):,} observations", flush=True)
        for fe_a, fe_b in combinations(spec.fe_cols, 2):
            result = pair_hardness(frame[fe_a].to_numpy(), frame[fe_b].to_numpy())
            row = {
                "dataset_id": spec.dataset_id,
                "kind": spec.kind,
                "n_obs_raw": n_obs_raw,
                "n_obs": len(frame),
                "n_singletons_dropped": n_dropped,
                "fe_a": fe_a,
                "fe_b": fe_b,
                **asdict(result),
                "one_minus_rho": 1.0 - result.rho_qr,
            }
            rows.append(row)
            print(
                f"  {fe_a} × {fe_b}: 1-rho={row['one_minus_rho']:.3e}, "
                f"share={result.worst_component_obs_share:.3f}", flush=True,
            )
        print(f"  completed in {time.perf_counter() - started:.2f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-singletons", action="store_true")
    parser.add_argument("--datasets", nargs="*", help="Dataset IDs to compute (default: all)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    specs = enumerate_datasets()
    if args.datasets:
        selected = set(args.datasets)
        specs = [spec for spec in specs if spec.dataset_id in selected]
        missing = sorted(selected - {spec.dataset_id for spec in specs})
        if missing:
            raise SystemExit("Unknown dataset IDs or missing input files: " + ", ".join(missing))
    results = compute(specs, args.keep_singletons)
    results.to_csv(output, index=False)
    print(f"[hardness] wrote {output}")


if __name__ == "__main__":
    main()

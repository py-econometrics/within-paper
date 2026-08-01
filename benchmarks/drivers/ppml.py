from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.core.cli import add_output_args
from benchmarks.solvers.registry import build_fepois_benchmarkers
from benchmarks.dgp.scenarios import BaseDGP
from benchmarks.core.interfaces import FeolsSpec
from benchmarks.solvers.runner import generate_datasets, run_benchmarks
from benchmarks.core.paths import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000_000]
K_VALUES = [1]
N_ITERS = 3
BURN_IN = 1
OUTPUT_CSV = ROOT / "benchmarks" / "results" / "fepois_bench.csv"

DGPS = [
    BaseDGP(DATA_DIR, "simple"),
    BaseDGP(DATA_DIR, "difficult"),
]

SPECS = [
    FeolsSpec(
        depvar="negbin_y",
        covariates=[f"x{i}" for i in range(1, k + 1)],
        fe_cols=["indiv_id", "firm_id", "year"],
        vcov="iid",
    )
    for k in K_VALUES
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_output_args(parser)
    args = parser.parse_args()
    output_csv = args.output_dir / OUTPUT_CSV.name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DGPS, SIZES, N_ITERS, BURN_IN)
    bundle = build_fepois_benchmarkers()
    run_benchmarks(
        bundle.benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )

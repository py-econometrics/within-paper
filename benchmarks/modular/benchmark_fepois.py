from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.modular.benchmarker_sets import build_fepois_benchmarkers
from benchmarks.modular.dgps import BaseDGP
from benchmarks.modular.interfaces import FeolsSpec
from benchmarks.modular.runner import generate_datasets, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000_000]
K_VALUES = [1]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "fepois_bench.csv"

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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "results")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    output_csv = args.output_dir / OUTPUT_CSV.name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DGPS, SIZES, N_ITERS, BURN_IN)
    bundle = build_fepois_benchmarkers()
    run_benchmarks(
        bundle.benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )

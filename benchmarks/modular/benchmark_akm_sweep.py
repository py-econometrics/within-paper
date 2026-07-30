from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.modular.cli import add_output_args
from benchmarks.modular.benchmarker_sets import (
    build_feols_benchmarkers,
    require_multiple_absorbed_factors,
)
from benchmarks.modular.dgps import get_akm_sweep_scenarios
from benchmarks.modular.interfaces import FeolsSpec
from benchmarks.modular.runner import run_benchmarks

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITERS = 3
BURN_IN = 1
DEFAULT_N_OBS = 1_000_000
DATA_DIR = ROOT / "benchmarks" / "data"
OUTPUT_CSV = ROOT / "benchmarks" / "results" / "feols_akm_sweep.csv"

DGPS = get_akm_sweep_scenarios(DATA_DIR)

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=["x1"],
        fe_cols=["indiv_id", "firm_id", "year"],
        vcov="iid",
    ),
]


def generate_akm_datasets():
    datasets = []
    for dgp in DGPS:
        print(f"[data] generating {dgp.dgp_name} n={DEFAULT_N_OBS:,}")
        datasets.extend(dgp.generate(n=DEFAULT_N_OBS, n_iters=N_ITERS, burn_in=BURN_IN))
    print(f"[data] {len(datasets)} datasets ready")
    return datasets


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_output_args(parser)
    args = parser.parse_args()
    output_csv = args.output_dir / OUTPUT_CSV.name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        require_multiple_absorbed_factors(spec)
    datasets = generate_akm_datasets()
    # Both views in one pass: package defaults for the cross-package tables and
    # the matched-accuracy arms for the mechanism figures. The AKM sweep is
    # where the mechanism experiment lives, so it always measures both.
    bundle = build_feols_benchmarkers(matched_accuracy=True)
    run_benchmarks(
        bundle.benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )

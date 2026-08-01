from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.core.cli import add_output_args
from benchmarks.solvers.registry import build_feols_benchmarkers
from benchmarks.dgp.scenarios import BaseDGP
from benchmarks.solvers.specs import paper_ols_spec
from benchmarks.solvers.runner import generate_datasets, run_benchmarks
from benchmarks.core.paths import DATA_DIR, ROOT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [10_000_000]
# The OLS paper table is a one-covariate benchmark. Other benchmark families
# have their own one-covariate specifications below.
K_VALUES = [1]
N_ITERS = 3
BURN_IN = 1
OUTPUT_CSV = ROOT / "benchmarks" / "results" / "feols_bench.csv"

DGPS = [
    BaseDGP(DATA_DIR, "simple"),
    BaseDGP(DATA_DIR, "difficult"),
]

SPECS = [paper_ols_spec()]

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
    # Package defaults only. The matched-accuracy arms are available via
    # matched_accuracy=True, but at 10M the unpreconditioned and MAP arms run
    # for hours against the cap, so that is an explicit decision rather than a
    # default.
    benchmarkers = build_feols_benchmarkers()
    run_benchmarks(
        benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )

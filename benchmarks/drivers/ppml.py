from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.core.cli import add_output_args
from benchmarks.solvers.registry import build_fepois_benchmarkers
from benchmarks.dgp.scenarios import BaseDGP
from benchmarks.solvers.specs import paper_ppml_spec
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

SPECS = [paper_ppml_spec()]

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
    benchmarkers = build_fepois_benchmarkers()
    run_benchmarks(
        benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )

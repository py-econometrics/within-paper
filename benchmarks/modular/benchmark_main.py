from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarker_sets import build_standard_feols_benchmarkers
from dgps import BaseDGP
from interfaces import FeolsSpec
from plotting import plot_readme_benchmarks
from runner import generate_datasets, plot_results, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000_000, 10_000_000]
# The OLS paper table is a one-covariate benchmark. Other benchmark families
# have their own one-covariate specifications below.
K_VALUES = [1]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_bench.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "benchmarks" / "base-benchmarks"

DGPS = [
    BaseDGP(DATA_DIR, "simple", k_values=tuple(K_VALUES)),
    BaseDGP(DATA_DIR, "difficult", k_values=tuple(K_VALUES)),
]

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=[f"x{i}" for i in range(1, k + 1)],
        fe_cols=fe_cols,
        vcov="iid",
    )
    for k in K_VALUES
    for fe_cols in (["indiv_id", "year", "firm_id"],)
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "results")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    output_csv = args.output_dir / OUTPUT_CSV.name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DGPS, SIZES, N_ITERS, BURN_IN)
    bundle = build_standard_feols_benchmarkers(include_torch=False)
    results_df = run_benchmarks(
        bundle.benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )
    plot_results(
        results_df,
        output_csv,
        figure_dir=args.figure_dir,
        figure_backends=bundle.figure_backends,
    )
    plot_readme_benchmarks(
        results_df, args.figure_dir / "bench_readme.png", model_k=1
    )

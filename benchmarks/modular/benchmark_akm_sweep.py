from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarker_sets import build_standard_feols_benchmarkers
from dgps import get_akm_sweep_scenarios
from interfaces import FeolsSpec
from runner import plot_results, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITERS = 3
BURN_IN = 1
DEFAULT_N_OBS = 1_000_000
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_akm_sweep.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "benchmarks" / "akm-benchmarks"

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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks" / "results")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    output_csv = args.output_dir / OUTPUT_CSV.name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_akm_datasets()
    bundle = build_standard_feols_benchmarkers(
        fixef_maxiter=10000, include_torch=False
    )
    results_df = run_benchmarks(
        bundle.benchmarkers, datasets, SPECS, output_csv, reuse_existing=args.reuse_existing
    )
    plot_results(
        results_df,
        output_csv,
        figure_dir=args.figure_dir,
        figure_backends=bundle.figure_backends,
    )

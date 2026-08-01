"""Tolerance frontier on three AKM mobility designs."""

from pathlib import Path

import pandas as pd

from benchmarks.akm import make_akm_data
from benchmarks.tolerance.measure import measure

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "tolerance_frontier.csv"


def main() -> None:
    rows = []
    for number in (1, 3, 5):
        design = f"akm_mobility_{number}"
        rows.append(measure(make_akm_data(design), design))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

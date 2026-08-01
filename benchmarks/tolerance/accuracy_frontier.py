"""Tolerance frontier on the simple and difficult base designs."""

from pathlib import Path

import pandas as pd

from benchmarks.data import make_base_data
from benchmarks.tolerance.measure import measure

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "accuracy_frontier.csv"
REPETITIONS = 3


def main() -> None:
    rows = []
    for design, seed in (("simple", 123), ("difficult", 124)):
        rows.append(
            measure(
                make_base_data(100_000, design, seed),
                design,
                repetitions=REPETITIONS,
            )
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

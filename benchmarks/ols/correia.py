"""OLS comparison on the downloaded Correia datasets."""

from functools import partial

import pandas as pd

from benchmarks.ols.run import LATEST, ROOT, run_experiment

NAMES = (
    "credit2", "credit", "soccer", "synthetic-complete", "synthetic-uniform-easy",
    "synthetic-uniform-hard", "synthetic-uniform-harder", "synthetic-assortative",
    "synthetic-zigzag", "enron", "github", "patents", "workers", "schools", "directors",
)
DATA = ROOT / "benchmarks" / "data" / "correia_data"


def main() -> None:
    run_experiment(
        experiment="correia",
        designs=[(name, partial(pd.read_csv, DATA / f"{name}.csv")) for name in NAMES],
        output=LATEST / "correia.csv",
        fixed_effects=("id1", "id2"),
        repetitions=3,
    )


if __name__ == "__main__":
    main()

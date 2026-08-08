"""OLS comparison on the downloaded Correia datasets."""

from functools import partial

import pandas as pd

from benchmarks.data import CORREIA_NAMES
from benchmarks.ols.run import LATEST, PACKAGE_RUNTIME_BACKENDS, ROOT, run_experiment

DATA = ROOT / "benchmarks" / "data" / "correia_data"


def main() -> None:
    run_experiment(
        experiment="correia",
        designs=[
            (name, partial(pd.read_csv, DATA / f"{name}.csv"))
            for name in CORREIA_NAMES
        ],
        output=LATEST / "correia.csv",
        fixed_effects=("id1", "id2"),
        repetitions=3,
        backends=PACKAGE_RUNTIME_BACKENDS,
    )


if __name__ == "__main__":
    main()

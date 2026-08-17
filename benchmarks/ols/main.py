"""Headline OLS comparison."""

from functools import partial

from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.ols.run import LATEST, PACKAGE_RUNTIME_BACKENDS, run_experiment
from benchmarks.ols.specifications import WORKER_FIRM_YEAR_SLOPES_SPECIFICATION

N_OBS = 10_000_000


def main() -> None:
    run_experiment(
        experiment="ols",
        designs=[
            (name, partial(make_base_data, N_OBS, name, seed))
            for name, seed in BASE_DESIGNS
        ],
        output=LATEST / "ols.csv",
        backends=PACKAGE_RUNTIME_BACKENDS,
        additional_specifications=(WORKER_FIRM_YEAR_SLOPES_SPECIFICATION,),
    )


if __name__ == "__main__":
    main()

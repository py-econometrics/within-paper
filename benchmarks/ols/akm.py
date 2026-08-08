"""OLS comparison across the twelve AKM designs."""

from functools import partial

from benchmarks.akm import SCENARIOS, make_akm_data
from benchmarks.ols.run import LATEST, PACKAGE_RUNTIME_BACKENDS, run_experiment

REPETITIONS = 3
MATCHED_MAP_TOLERANCE = 1e-10
MATCHED_LSMR_TOLERANCE = 1e-12
MATCHED_MAXITER = 10_000


def main() -> None:
    run_experiment(
        experiment="akm",
        designs=[(name, partial(make_akm_data, name)) for name in SCENARIOS],
        output=LATEST / "akm.csv",
        repetitions=REPETITIONS,
        backends=PACKAGE_RUNTIME_BACKENDS,
        extra_python_cells=(
            ("rust-map", "matched", MATCHED_MAP_TOLERANCE, MATCHED_MAXITER),
            ("within-off", "matched", MATCHED_LSMR_TOLERANCE, MATCHED_MAXITER),
            ("within-diagonal", "matched", MATCHED_LSMR_TOLERANCE, MATCHED_MAXITER),
            ("within-additive", "matched", MATCHED_LSMR_TOLERANCE, MATCHED_MAXITER),
        ),
    )


if __name__ == "__main__":
    main()

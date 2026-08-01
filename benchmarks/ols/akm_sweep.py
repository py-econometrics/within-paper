"""OLS comparison across the eleven AKM designs."""

from functools import partial

from benchmarks.akm import SCENARIOS, make_akm_data
from benchmarks.ols.run import LATEST, run_experiment


def main() -> None:
    run_experiment(
        experiment="akm",
        designs=[(name, partial(make_akm_data, name)) for name in SCENARIOS],
        output=LATEST / "akm.csv",
        repetitions=3,
        extra_python_cells=(
            ("rust-map", "matched", 1e-10, 10_000),
            ("within-off", "matched", 1e-12, 10_000),
            ("within-diagonal", "matched", 1e-12, 10_000),
            ("within-additive", "matched", 1e-12, 10_000),
        ),
    )


if __name__ == "__main__":
    main()

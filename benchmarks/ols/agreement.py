"""Small cross-package coefficient agreement check."""

from functools import partial

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data
from benchmarks.ols.run import LATEST, run_experiment


def main() -> None:
    raw = run_experiment(
        experiment="agreement",
        designs=[
            (name, partial(make_base_data, 100_000, name, seed))
            for name, seed in BASE_DESIGNS
        ],
        output=None,
        repetitions=1,
    )
    rows = []
    for design, group in raw[raw["converged"]].groupby("design"):
        reference = float(group.loc[group["backend"] == "rust-map", "beta_x1"].iloc[0])
        for row in group.to_dict("records"):
            rows.append(
                {
                    "design": design,
                    "backend": row["backend"],
                    "x1": row["beta_x1"],
                    "max_abs_diff": abs(float(row["beta_x1"]) - reference),
                    "converged": True,
                }
            )
    pd.DataFrame(rows).to_csv(LATEST / "agreement.csv", index=False)


if __name__ == "__main__":
    main()

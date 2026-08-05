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
    for design, group in raw.groupby("design"):
        references = group[(group["backend"] == "rust-map") & group["converged"]]
        reference = (
            float(references["beta_x1"].iloc[0]) if len(references) else None
        )
        for row in group.to_dict("records"):
            converged = bool(row["converged"]) and reference is not None
            rows.append(
                {
                    "design": design,
                    "backend": row["backend"],
                    "x1": row["beta_x1"] if converged else None,
                    "max_abs_diff": (
                        abs(float(row["beta_x1"]) - reference)
                        if converged and reference is not None
                        else None
                    ),
                    "converged": converged,
                    "capped": bool(row.get("capped", False)),
                    "error": (
                        str(row.get("error", ""))
                        if not row["converged"]
                        else "rust-map agreement reference did not converge"
                        if reference is None
                        else ""
                    ),
                }
            )
    LATEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LATEST / "agreement.csv", index=False)


if __name__ == "__main__":
    main()

"""Join spectral gaps to recorded runtimes for the paper figure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]

DEFAULT_HARDNESS = ROOT / "results" / "runs" / "latest" / "hardness.csv"
DEFAULT_RESULTS_DIR = ROOT / "results" / "runs" / "latest"
DEFAULT_OUT = ROOT / "results" / "runs" / "latest" / "gap_runtime_analysis.json"

# Worker-firm is the economically meaningful pair for AKM designs; for Correia
# and other two-factor tables the single pair is used.
WORKER_FIRM_PAIRS = {
    ("indiv_id", "firm_id"),
    ("id1", "id2"),
}


def _median_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    """One median runtime per design and backend, keeping failure counts."""
    rows = []
    group_cols = ["design", "backend", "view"]
    if "n_obs" in frame.columns:
        group_cols.append("n_obs")
    if "n_fe" in frame.columns:
        group_cols.append("n_fe")
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys, strict=True))
        times = pd.to_numeric(group["runtime_s"], errors="coerce")
        success = group["converged"].astype(str).str.lower().isin({"true", "1"})
        converged_times = times[success & times.notna()]
        record["n_trials"] = int(len(group))
        record["n_success"] = int(success.sum())
        record["median_time"] = (
            float(converged_times.median()) if len(converged_times) else None
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _load_runtime_tables(results_dir: Path) -> pd.DataFrame:
    frames = []
    for filename in ("ols.csv", "ppml.csv", "akm.csv", "correia.csv"):
        path = results_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"design", "backend", "view", "runtime_s", "converged"}
        if not required.issubset(frame.columns):
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _design_key_from_hardness(dataset_id: str) -> str:
    """Map a hardness dataset_id to the runtime design label."""
    if dataset_id.startswith("memory_"):
        return dataset_id.split("_")[1]
    return dataset_id


def _sized_key(design: str, n_obs) -> str:
    """Join key that keeps the sample size.

    The simple and difficult designs are run at 100K, 1M, and 10M, and their
    connectivity is not the same at each: the difficult worker-firm gap is
    1.7e-3 at 100K, 1.7e-5 at 1M, and 1.7e-7 at 10M. Joining on the family
    name alone paired the 1M PPML runtimes with the 10M OLS gap, four orders
    of magnitude away, which moves the fitted slope.
    """
    try:
        size = int(float(n_obs))
    except (TypeError, ValueError):
        return design
    return f"{design}@{size}"


def _select_gap_rows(hardness: pd.DataFrame) -> pd.DataFrame:
    """One gap per design: the worker-firm (or only) pair with smallest gap."""
    hardness = hardness.copy()
    hardness["family"] = hardness["dataset_id"].map(_design_key_from_hardness)
    hardness["design"] = [
        _sized_key(family, n_obs)
        for family, n_obs in zip(hardness["family"], hardness["n_obs_raw"], strict=True)
    ]
    hardness["is_worker_firm"] = [
        (str(a), str(b)) in WORKER_FIRM_PAIRS
        or (str(b), str(a)) in WORKER_FIRM_PAIRS
        for a, b in zip(hardness["fe_a"], hardness["fe_b"], strict=True)
    ]
    selected = []
    for design, group in hardness.groupby("design"):
        preferred = group[group["is_worker_firm"]]
        use = preferred if len(preferred) else group
        # Smallest gap = hardest pair for the diagnostic annotation.
        row = use.loc[use["one_minus_rho"].astype(float).idxmin()]
        selected.append(row)
    return pd.DataFrame(selected)


def analyze(
    hardness_path: Path,
    results_dir: Path,
) -> dict:
    if not hardness_path.exists():
        raise FileNotFoundError(f"hardness file not found: {hardness_path}")
    hardness = pd.read_csv(hardness_path)
    gaps = _select_gap_rows(hardness)
    runtimes = _load_runtime_tables(results_dir)
    if runtimes.empty:
        raise FileNotFoundError(f"no runtime CSVs with required columns in {results_dir}")
    medians = _median_runtime(runtimes)
    medians["family"] = medians["design"].astype(str)
    medians["design"] = [
        _sized_key(str(design), n_obs)
        for design, n_obs in zip(medians["design"], medians["n_obs"], strict=True)
    ]

    gap_cols = gaps[
        [
            "design",
            "one_minus_rho",
            "kind",
        ]
    ].rename(columns={"one_minus_rho": "gap"})
    joined = medians.merge(gap_cols, on="design", how="inner")

    points = []
    for _, row in joined.iterrows():
        points.append(
            {
                "design": row["design"],
                "backend": row["backend"],
                "view": row["view"],
                "gap": float(row["gap"]) if pd.notna(row["gap"]) else None,
                "median_time": (
                    None if pd.isna(row["median_time"]) else float(row["median_time"])
                ),
                "n_trials": int(row["n_trials"]),
                "n_success": int(row["n_success"]),
                "kind": row.get("kind"),
            }
        )

    return {"points": points}


def main() -> None:
    report = analyze(DEFAULT_HARDNESS, DEFAULT_RESULTS_DIR)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"analyze-gap-runtime / pooled / {len(report['points'])} points")


if __name__ == "__main__":
    main()

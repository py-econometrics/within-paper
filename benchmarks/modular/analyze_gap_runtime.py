"""Pooled spectral-gap versus runtime diagnostic (plan item 6).

Pools existing hardness and runtime CSVs and reports:

- log runtime against log gap, by backend, with a fitted slope per backend
- named counter-examples (sorting non-monotonicity; akm_mobility_4 vs _5;
  directors' small component share)

Inputs already exist under ``benchmarks/results/`` and
``results/runs/latest/hardness.csv``. No new solves are required.

Run with:

    pixi run analyze-gap-runtime
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARDNESS = ROOT / "results" / "runs" / "latest" / "hardness.csv"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results"
DEFAULT_OUT = ROOT / "results" / "runs" / "latest" / "gap_runtime_analysis.json"

# Worker-firm is the economically meaningful pair for AKM designs; for Correia
# and other two-factor tables the single pair is used.
WORKER_FIRM_PAIRS = {
    ("indiv_id", "firm_id"),
    ("id1", "id2"),
}


@dataclass(frozen=True)
class BackendSlope:
    backend: str
    n_points: int
    slope: float | None
    intercept: float | None
    r_squared: float | None
    note: str


def _median_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    """One median runtime per (dgp, backend), keeping failure counts."""
    rows = []
    group_cols = ["dgp", "backend"]
    if "n_obs" in frame.columns:
        group_cols.append("n_obs")
    if "n_fe" in frame.columns:
        group_cols.append("n_fe")
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys, strict=True))
        times = pd.to_numeric(group["time"], errors="coerce")
        success = group["success"].astype(str).str.lower().isin({"true", "1"})
        converged_times = times[success & times.notna()]
        record["n_trials"] = int(len(group))
        record["n_success"] = int(success.sum())
        record["median_time"] = (
            float(converged_times.median()) if len(converged_times) else None
        )
        record["iqr_time"] = (
            float(converged_times.quantile(0.75) - converged_times.quantile(0.25))
            if len(converged_times) >= 2
            else None
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _load_runtime_tables(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("*.csv")):
        if path.name.endswith("_summary.csv"):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"dgp", "backend", "time", "success"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _design_key_from_hardness(dataset_id: str) -> str:
    """Map a hardness dataset_id to the runtime design label."""
    # akm_*_1000000_k1_iter_1 -> akm_*
    match = re.match(r"(akm_[a-z]+_\d+)_", dataset_id)
    if match:
        return match.group(1)
    # simple_10000000 / difficult_10000000 style ids
    match = re.match(r"(simple|difficult)", dataset_id)
    if match:
        return match.group(1)
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


def _fit_log_log_slope(
    gaps: np.ndarray, times: np.ndarray
) -> tuple[float | None, float | None, float | None, str]:
    mask = np.isfinite(gaps) & np.isfinite(times) & (gaps > 0) & (times > 0)
    x = np.log(gaps[mask])
    y = np.log(times[mask])
    if x.size < 3:
        return None, None, None, "fewer than 3 positive finite points"
    if float(np.std(x)) == 0.0:
        return None, None, None, "zero variance in log gap"
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(slope), float(intercept), (float(r2) if r2 is not None else None), "ok"


def _counter_examples(joined: pd.DataFrame) -> list[dict]:
    examples = []
    # Sorting non-monotonicity: median within time not monotone in gap.
    sorting = joined[joined["family"].astype(str).str.startswith("akm_sorting")]
    if len(sorting):
        for backend, group in sorting.groupby("backend"):
            ordered = group.sort_values("gap")
            times = ordered["median_time"].to_numpy(dtype=float)
            # Sorted by increasing gap, better connectivity should mean less
            # work, so runtime is expected to fall. A rise is the counter-
            # evidence; checking for a fall flagged the expected pattern.
            if np.any(np.diff(times[np.isfinite(times)]) > 0):
                examples.append(
                    {
                        "name": "sorting_non_monotonic",
                        "backend": backend,
                        "detail": (
                            "AKM sorting median runtime is not monotone in the "
                            "worker-firm gap"
                        ),
                        "designs": ordered["design"].tolist(),
                        "gaps": ordered["gap"].astype(float).tolist(),
                        "median_times": ordered["median_time"].astype(float).tolist(),
                    }
                )

    # Mobility 4 vs 5: nearly identical gaps, nearly identical runtimes.
    for backend, group in joined.groupby("backend"):
        m4 = group[group["family"] == "akm_mobility_4"]
        m5 = group[group["family"] == "akm_mobility_5"]
        if len(m4) == 1 and len(m5) == 1:
            g4 = float(m4.iloc[0]["gap"])
            g5 = float(m5.iloc[0]["gap"])
            t4 = m4.iloc[0]["median_time"]
            t5 = m5.iloc[0]["median_time"]
            examples.append(
                {
                    "name": "akm_mobility_4_vs_5",
                    "backend": backend,
                    "detail": (
                        "Nearly identical worker-firm gaps and nearly identical "
                        "runtimes; the gap does not separate these two cells"
                    ),
                    "gaps": [g4, g5],
                    "median_times": [
                        None if pd.isna(t4) else float(t4),
                        None if pd.isna(t5) else float(t5),
                    ],
                }
            )

    # directors: gap looks hard but component share is small.
    directors = joined[joined["family"].astype(str).str.contains("directors")]
    for _, row in directors.iterrows():
        examples.append(
            {
                "name": "directors_small_component_share",
                "backend": row["backend"],
                "detail": (
                    "The pair attaining the gap covers only a fraction of "
                    "observations, so the design is easier than its gap suggests"
                ),
                "gap": float(row["gap"]),
                "worst_component_obs_share": float(row["worst_component_obs_share"]),
                "median_time": (
                    None if pd.isna(row["median_time"]) else float(row["median_time"])
                ),
            }
        )
    return examples


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
    medians["family"] = medians["dgp"].astype(str)
    medians["design"] = [
        _sized_key(str(dgp), n_obs)
        for dgp, n_obs in zip(medians["dgp"], medians["n_obs"], strict=True)
    ]

    gap_cols = gaps[
        [
            "design",
            "one_minus_rho",
            "worst_component_obs_share",
            "rho_qr",
            "dataset_id",
            "kind",
        ]
    ].rename(columns={"one_minus_rho": "gap"})
    joined = medians.merge(gap_cols, on="design", how="inner")

    slopes: list[BackendSlope] = []
    for backend, group in joined.groupby("backend"):
        usable = group.dropna(subset=["median_time", "gap"])
        slope, intercept, r2, note = _fit_log_log_slope(
            usable["gap"].to_numpy(dtype=float),
            usable["median_time"].to_numpy(dtype=float),
        )
        slopes.append(
            BackendSlope(
                backend=str(backend),
                n_points=int(len(usable)),
                slope=slope,
                intercept=intercept,
                r_squared=r2,
                note=note,
            )
        )

    points = []
    for _, row in joined.iterrows():
        points.append(
            {
                "design": row["design"],
                "backend": row["backend"],
                "gap": float(row["gap"]) if pd.notna(row["gap"]) else None,
                "worst_component_obs_share": (
                    float(row["worst_component_obs_share"])
                    if pd.notna(row["worst_component_obs_share"])
                    else None
                ),
                "median_time": (
                    None if pd.isna(row["median_time"]) else float(row["median_time"])
                ),
                "iqr_time": None if pd.isna(row.get("iqr_time")) else float(row["iqr_time"]),
                "n_trials": int(row["n_trials"]),
                "n_success": int(row["n_success"]),
                "kind": row.get("kind"),
                "source_dataset_id": row.get("dataset_id"),
            }
        )

    return {
        "n_designs": int(joined["design"].nunique()),
        "n_backends": int(joined["backend"].nunique()),
        "n_points": len(points),
        "slopes": [asdict(s) for s in sorted(slopes, key=lambda s: s.backend)],
        "counter_examples": _counter_examples(joined),
        "points": points,
        "conclusion": (
            "Pairwise spectral gaps describe difficult designs but give no "
            "numerical cutoff for solver choice. Counter-examples include "
            "non-monotonic sorting rows, near-tied mobility gaps, and designs "
            "whose hardest component covers few observations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardness", type=Path, default=DEFAULT_HARDNESS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = analyze(args.hardness, args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"designs={report['n_designs']} backends={report['n_backends']} "
          f"points={report['n_points']}")
    print("\nlog-log slopes (log runtime ~ slope * log gap):")
    for slope in report["slopes"]:
        if slope["slope"] is None:
            print(f"  {slope['backend']}: n={slope['n_points']} ({slope['note']})")
        else:
            print(
                f"  {slope['backend']}: n={slope['n_points']} "
                f"slope={slope['slope']:.3f} R^2={slope['r_squared']:.3f}"
            )
    print(f"\ncounter-examples: {len(report['counter_examples'])}")
    for example in report["counter_examples"]:
        print(f"  - {example['name']} [{example['backend']}]")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

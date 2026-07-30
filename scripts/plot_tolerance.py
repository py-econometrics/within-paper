"""Plot runtime against achieved coefficient and residual precision."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "within-paper-tolerance"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "runs" / "latest" / "tolerance_frontier.csv"
DEFAULT_OUTPUT = ROOT / "figures" / "results" / "tolerance_frontier.svg"

METHOD_ORDER = (
    "lsmr_off",
    "lsmr_diagonal",
    "lsmr_additive",
    "pyfixest_map",
    "r_fixest",
    "julia_fem",
)

STYLE = {
    "lsmr_off": ("#7c3aed", "X"),
    "lsmr_diagonal": ("#7b8494", "^"),
    "lsmr_additive": ("#2563eb", "D"),
    "pyfixest_map": ("#c2410c", "o"),
    "r_fixest": ("#a16207", "s"),
    "julia_fem": ("#0f766e", "P"),
}

METRICS = (
    ("coefficient_error_se", "Slope error (reference SE)"),
    ("residual_error", r"Residual error (relative $\ell_2$ norm)"),
)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().eq("true")


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one median point per design, method, and tolerance."""
    required = {
        "design",
        "method",
        "label",
        "tolerance",
        "default_tolerance",
        "time_s",
        "success",
        "coefficient_error_se",
        "residual_error",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"tolerance results are missing columns: {missing}")

    data = raw.copy()
    data["success"] = _as_bool(data["success"])
    numeric = [
        "tolerance",
        "default_tolerance",
        "time_s",
        "coefficient_error_se",
        "residual_error",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    keys = ["design", "method", "label", "tolerance", "default_tolerance"]
    rows: list[dict] = []
    for key, group in data.groupby(keys, dropna=False, sort=False):
        successful = group[
            group["success"]
            & np.isfinite(group["time_s"])
            & np.isfinite(group["coefficient_error_se"])
            & np.isfinite(group["residual_error"])
        ]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_trials": len(group),
                "n_success": len(successful),
                "median_time_s": (
                    float(successful["time_s"].median())
                    if len(successful)
                    else np.nan
                ),
                "coefficient_error_se": (
                    float(successful["coefficient_error_se"].median())
                    if len(successful)
                    else np.nan
                ),
                "residual_error": (
                    float(successful["residual_error"].median())
                    if len(successful)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _axis_limits(
    points: pd.DataFrame, metric: str
) -> tuple[tuple[float, float], float]:
    values = points.loc[
        np.isfinite(points[metric]) & (points[metric] > 0), metric
    ].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"no positive {metric} values to plot")
    floor = max(float(values.min()) / 5, np.finfo(np.float64).eps)
    lower = min(floor, float(values.min()) / 2)
    upper = float(values.max()) * 2
    return (lower, upper), floor


def _runtime_limits(points: pd.DataFrame) -> tuple[float, float]:
    values = points.loc[
        np.isfinite(points["median_time_s"]) & (points["median_time_s"] > 0),
        "median_time_s",
    ].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("no positive runtimes to plot")
    return float(values.min()) / 1.6, float(values.max()) * 1.6


def _failure_note(points: pd.DataFrame, design: str) -> str | None:
    failed = points[(points["design"] == design) & (points["n_success"] == 0)]
    if failed.empty:
        return None
    entries = []
    for method in METHOD_ORDER:
        subset = failed[failed["method"] == method]
        if subset.empty:
            continue
        label = str(subset["label"].iloc[0])
        count = len(subset)
        suffix = "setting" if count == 1 else "settings"
        entries.append(f"{label}: {count} {suffix}")
    return "No returned solution\n" + "\n".join(entries)


def tolerance_figure(raw: pd.DataFrame, output: Path) -> None:
    points = aggregate_results(raw)
    designs = sorted(
        points["design"].dropna().unique(),
        key=lambda value: int(str(value).rsplit("_", 1)[-1]),
    )
    if not designs:
        raise ValueError("no designs found in tolerance results")

    n_columns = len(designs)
    fig, axes = plt.subplots(
        2,
        n_columns,
        figsize=(4.0 * n_columns, 7.1),
        sharey=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    x_limits = {
        metric: _axis_limits(points, metric) for metric, _ in METRICS
    }
    y_limits = _runtime_limits(points)
    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")

    for row_number, (metric, x_label) in enumerate(METRICS):
        limits, floor = x_limits[metric]
        for column_number, design in enumerate(designs):
            ax = axes[row_number, column_number]
            letter = next(panel_letters)
            mobility = str(design).rsplit("_", 1)[-1]
            ax.set_title(
                f"({letter}) Mobility {mobility}",
                loc="left",
                fontsize=9.3,
                fontweight="bold",
            )

            for method in METHOD_ORDER:
                method_points = points[
                    (points["design"] == design)
                    & (points["method"] == method)
                    & (points["n_success"] > 0)
                    & np.isfinite(points["median_time_s"])
                    & np.isfinite(points[metric])
                ].copy()
                if method_points.empty:
                    continue
                method_points["plot_error"] = method_points[metric].clip(lower=floor)
                method_points = method_points.sort_values("plot_error")
                colour, marker = STYLE[method]
                ax.plot(
                    method_points["plot_error"],
                    method_points["median_time_s"],
                    color=colour,
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=2,
                )
                ax.scatter(
                    method_points["plot_error"],
                    method_points["median_time_s"],
                    s=35,
                    color=colour,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

                defaults = method_points[
                    np.isclose(
                        method_points["tolerance"],
                        method_points["default_tolerance"],
                        rtol=0,
                        atol=0,
                    )
                ]
                if not defaults.empty:
                    ax.scatter(
                        defaults["plot_error"],
                        defaults["median_time_s"],
                        s=86,
                        facecolors="none",
                        edgecolors="#111111",
                        linewidths=0.85,
                        zorder=4,
                    )

            note = _failure_note(points, design)
            if note:
                ax.annotate(
                    note,
                    xy=(0.02, 0.97),
                    xycoords="axes fraction",
                    ha="left",
                    va="top",
                    fontsize=6.3,
                    color="#555555",
                    linespacing=1.25,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(*limits)
            ax.set_ylim(*y_limits)
            ax.set_xlabel(x_label, fontsize=8.4)
            if column_number == 0:
                ax.set_ylabel("Median runtime (s)", fontsize=8.7)
            ax.tick_params(axis="both", labelsize=7.6)
            ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
            ax.grid(True, which="minor", linewidth=0.25, alpha=0.16)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    labels = {
        method: str(
            points.loc[points["method"] == method, "label"].dropna().iloc[0]
        )
        for method in METHOD_ORDER
        if (points["method"] == method).any()
    }
    handles = [
        Line2D(
            [0],
            [0],
            color=STYLE[method][0],
            marker=STYLE[method][1],
            linewidth=1.2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=labels[method],
        )
        for method in METHOD_ORDER
        if method in labels
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markersize=8,
            label="default tolerance",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncols=min(7, len(handles)),
        fontsize=7.6,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    cap_text = "the same iteration cap"
    if "maxiter" in raw:
        caps = pd.to_numeric(raw["maxiter"], errors="coerce").dropna().unique()
        if len(caps) == 1:
            cap_text = f"a common {int(caps[0]):,}-iteration cap"
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.12)
    fig.text(
        0.5,
        0.025,
        "Each point is the median of repeated fits on one fixed sample. "
        f"Lines vary the method's native tolerance; all methods use {cap_text}. "
        "Smaller errors indicate greater precision.",
        ha="center",
        fontsize=7.3,
        color="#444444",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if output.suffix.lower() == ".svg" else None
    fig.savefig(output, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} is missing. Run `pixi run bench-tolerance`.")
    raw = pd.read_csv(args.input)
    tolerance_figure(raw, args.output)
    print(f"[figure] wrote {args.output}")


if __name__ == "__main__":
    main()

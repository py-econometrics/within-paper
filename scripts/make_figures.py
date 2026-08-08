"""Generate the paper's result figures from recorded benchmark output.

The headline AKM figure reads the tracked canonical paper result file.  It can
therefore be regenerated while compiling the paper, even when the local raw
benchmark directory is absent.  The precision frontier still reads its raw
benchmark file because it reports every tolerance setting rather than a paper
table projection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Deterministic SVG: a stable salt fixes the generated clip-path ids and
# metadata={"Date": None} drops the embedded timestamp. Without both, every
# regeneration rewrites the whole tracked file.
matplotlib.rcParams["svg.hashsalt"] = "within-paper-figures"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.benchmark_methods import METHODS
from scripts.plot_tolerance import tolerance_figure

ROOT = Path(__file__).absolute().parents[1]

RESULTS = ROOT / "results" / "runs" / "latest"
FIGURES = ROOT / "figures" / "results"
PAPER_RESULTS = ROOT / "results" / "paper" / "benchmark_tables.json"

# The left panel compares package defaults. The right panel holds package and
# accuracy fixed, so it isolates the solver and preconditioner. These are the
# backend names as the result files spell them; every colour, marker, dash and
# label is derived from the registry, so nothing is restated here.
CROSS_PACKAGE_BACKENDS = ("rust-map", "fixest", "FEM.jl", "within")

MECHANISM_BACKENDS = (
    "rust-map",
    "within-off",
    "within-diagonal",
    "within-additive",
)

CROSSOVER_FILES = {
    "OLS": ("ols.csv", {"rust-map": "rust-map", "fixest": "fixest", "FEM.jl": "FEM.jl", "within": "within"}),
    "PPML": ("ppml.csv", {"rust-map": "rust-map", "fixest": "fixest", "FEM.jl": "GLFEM.jl", "within": "within"}),
}

# The crossover figure draws one series for both Julia packages, so its label
# names both. Every other series takes the registry's inline label.
CROSSOVER_LABEL = {"FEM.jl": "FEM.jl / GLFEM.jl — LSMR (diagonal)"}


def _load_points() -> list[dict]:
    """Return the collected, tracked records for the headline figure."""
    path = PAPER_RESULTS
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run `pixi run collect-paper-results` first."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("headline_figure", {}).get("points", [])
    if not records:
        raise SystemExit(
            "The canonical paper results have no headline-figure records. "
            "Run `pixi run collect-paper-results` after the AKM benchmark."
        )
    return records


def _load_crossover_results() -> dict[str, dict[str, dict[str, object]]]:
    """Median runtime and convergence counts for the simple/difficult DGPs."""

    target_n = {"OLS": 10_000_000, "PPML": 1_000_000}
    results: dict[str, dict[str, dict[str, object]]] = {}
    for model, (filename, backends) in CROSSOVER_FILES.items():
        path = RESULTS / filename
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        model_results: dict[str, dict[str, object]] = {}
        for backend, raw_backend in backends.items():
            grouped = {
                "simple": {"times": [], "n_trials": 0, "n_success": 0},
                "difficult": {"times": [], "n_trials": 0, "n_success": 0},
            }
            for row in rows:
                design = row["design"]
                if design not in grouped:
                    continue
                if (
                    row["backend"] != raw_backend
                    or int(row["n_obs"]) != target_n[model]
                    or int(row["n_fe"]) != 3
                ):
                    continue
                record = grouped[design]
                record["n_trials"] += 1
                success = row["converged"].lower() in {"true", "1"}
                runtime = row["runtime_s"]
                if success and runtime:
                    record["times"].append(float(runtime))
                    record["n_success"] += 1
            backend_results: dict[str, object] = {}
            for design, record in grouped.items():
                times = record.pop("times")
                backend_results[design] = {
                    "median_time": float(np.median(times)) if times else None,
                    **record,
                }
            model_results[backend] = backend_results
        results[model] = model_results
    return results


def _visible_point(row: dict) -> bool:
    """Whether a record has a comparable runtime for the headline plot."""
    return (
        row.get("status") in {"complete", "partial", "incomplete", "capped"}
        and isinstance(row.get("gap"), (int, float))
        and isinstance(row.get("median_time"), (int, float))
        and row["gap"] > 0
        and row["median_time"] > 0
    )


def _headline_x_limits(points: list[dict]) -> tuple[float, float]:
    """One padded, reversed log scale shared by all headline panels."""
    gaps = [
        row["gap"]
        for row in points
        if _visible_point(row)
    ]
    if not gaps:
        raise ValueError("No plottable headline-figure records")
    return max(gaps) * 1.45, min(gaps) / 1.45


def _runtime_panel(
    ax,
    points: list[dict],
    backends: tuple[str, ...],
    *,
    family: str,
    view: str,
    title: str,
    x_limits: tuple[float, float],
    show_legend: bool,
) -> None:
    """Draw one observed-median panel without fitting a trend line."""
    for raw_name in backends:
        display, colour, marker, line = METHODS[raw_name]
        usable = [
            row
            for row in points
            if row.get("family") == family
            and row.get("view") == view
            and row.get("backend") == raw_name
            and _visible_point(row)
        ]
        complete = [row for row in usable if row["status"] == "complete"]
        partial = [
            row for row in usable if row["status"] in {"partial", "incomplete"}
        ]
        capped = [row for row in usable if row["status"] == "capped"]
        returned = sorted(complete + partial, key=lambda row: row["gap"], reverse=True)

        # Join observed medians within a configuration. Capped cells are lower
        # bounds rather than returned fits, so their markers remain unconnected.
        if len(returned) >= 2:
            ax.plot(
                [row["gap"] for row in returned],
                [row["median_time"] for row in returned],
                color=colour,
                linewidth=1.15,
                linestyle=line,
                alpha=0.55,
                zorder=2,
            )

        if complete:
            ax.scatter(
                [row["gap"] for row in complete],
                [row["median_time"] for row in complete],
                s=34,
                c=colour,
                marker=marker,
                label=display,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        if partial:
            ax.scatter(
                [row["gap"] for row in partial],
                [row["median_time"] for row in partial],
                s=41,
                facecolors="none",
                edgecolors=colour,
                marker=marker,
                linewidths=1.15,
                label=display if not complete else None,
                zorder=3,
            )
        for row in capped:
            ax.scatter(
                row["gap"],
                row["median_time"],
                s=43,
                facecolors="none",
                edgecolors=colour,
                marker=marker,
                linewidths=1.15,
                label=display if not complete and not partial else None,
                zorder=3,
            )
            # The marker is the elapsed time when the iteration cap was hit;
            # the upward arrow marks it as a lower bound, not a returned fit.
            ax.annotate(
                "",
                xy=(row["gap"], row["median_time"] * 1.38),
                xytext=(row["gap"], row["median_time"] * 1.03),
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0.85},
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_title(title, loc="left", fontsize=8.8, fontweight="bold")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_legend:
        legend = ax.legend(
            frameon=False,
            fontsize=5.8,
            loc="upper left",
            ncols=1,
            title="configuration",
            title_fontsize=5.9,
            labelspacing=0.28,
        )
        if legend is not None:
            legend._legend_box.align = "left"


def headline_figure(points: list[dict], out: Path) -> None:
    """Show mobility and sorting separately under the two comparison views."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.6, 7.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.22},
    )

    panel_specs = (
        ("mobility", "default", CROSS_PACKAGE_BACKENDS, "(a) Mobility: package defaults"),
        ("mobility", "matched", MECHANISM_BACKENDS, "(b) Mobility: matched accuracy"),
        ("sorting", "default", CROSS_PACKAGE_BACKENDS, "(c) Sorting: package defaults"),
        ("sorting", "matched", MECHANISM_BACKENDS, "(d) Sorting: matched accuracy"),
    )
    x_limits = _headline_x_limits(points)
    for ax, (family, view, backends, title) in zip(
        axes.flat, panel_specs, strict=True
    ):
        _runtime_panel(
            ax,
            points,
            backends,
            family=family,
            view=view,
            title=title,
            x_limits=x_limits,
            show_legend=family == "mobility",
        )

    visible_times = [row["median_time"] for row in points if _visible_point(row)]
    if not visible_times:
        raise ValueError("No plottable headline-figure records")
    lower = min(visible_times) / 1.75
    upper = max(visible_times) * 2.15
    for ax in axes.flat:
        ax.set_ylim(lower, upper)
    axes[0, 0].set_ylabel("Median runtime (s)")
    axes[1, 0].set_ylabel("Median runtime (s)")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.12)
    fig.text(
        0.535,
        0.073,
        "Lines join returned medians; hollow: fewer than all fits returned; "
        "arrows: capped lower-bound time",
        ha="center",
        fontsize=6.4,
        color="#555555",
    )
    fig.text(
        0.535,
        0.035,
        "Worker-firm spectral gap  $1-\\rho_{WF}$  "
        "(weaker connectivity to the right)",
        ha="center",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def headline_main() -> None:
    """Render only the canonical-data headline figure used by paper compilation."""
    points = _load_points()
    headline_target = FIGURES / "gap_runtime.svg"
    headline_figure(points, headline_target)
    print(
        f"[figures] wrote {headline_target.relative_to(ROOT)} "
        f"from {len(points)} canonical AKM records"
    )


def crossover_figure(
    results: dict[str, dict[str, dict[str, object]]], out: Path
) -> None:
    """Show the reversal between well-connected and near-nested designs."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.9),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    offsets = {
        "rust-map": -0.075,
        "fixest": -0.025,
        "FEM.jl": 0.025,
        "within": 0.075,
    }
    designs = ("simple", "difficult")
    x_base = np.array([0.0, 1.0])
    legend_handles = []
    legend_labels = []

    all_times = [
        cell["median_time"]
        for model_results in results.values()
        for backend_results in model_results.values()
        for cell in backend_results.values()
        if cell["median_time"] is not None
    ]
    y_min = min(all_times) * 0.65
    y_max = max(all_times) * 1.65

    titles = {
        "OLS": "(a) OLS, 10M observations",
        "PPML": "(b) PPML, 1M observations",
    }
    for ax, model in zip(axes, ("OLS", "PPML"), strict=True):
        for backend, backend_results in results[model].items():
            label, colour, marker, line = METHODS[backend]
            display = CROSSOVER_LABEL.get(backend) or label
            x = x_base + offsets[backend]
            y = np.array(
                [
                    backend_results[design]["median_time"]
                    if backend_results[design]["median_time"] is not None
                    else np.nan
                    for design in designs
                ],
                dtype=float,
            )
            (handle,) = ax.plot(
                x,
                y,
                color=colour,
                marker=marker,
                markersize=5.2,
                linewidth=1.35,
                linestyle=line,
                alpha=0.82,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=3,
            )
            if model == "OLS":
                legend_handles.append(handle)
                legend_labels.append(display)
            for index, design in enumerate(designs):
                cell = backend_results[design]
                if cell["median_time"] is None:
                    ax.text(
                        x[index],
                        0.94,
                        "failed",
                        color=colour,
                        fontsize=7,
                        rotation=90,
                        ha="center",
                        va="top",
                        transform=ax.get_xaxis_transform(),
                    )

        ax.set_title(titles[model], loc="left", fontsize=9.4, fontweight="bold")
        ax.set_xticks(x_base, ["Well-connected", "Near-nested"])
        ax.set_xlim(-0.18, 1.18)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.35)
        ax.grid(True, axis="y", which="minor", linewidth=0.25, alpha=0.16)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.annotate(
            "median of three full regression calls",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            fontsize=6.6,
            color="#555555",
        )

    axes[0].set_ylabel("Median runtime (s)")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.01),
        ncols=2,
        frameon=False,
        fontsize=7.1,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.76, bottom=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    headline_main()

    tolerance_source = RESULTS / "tolerance_frontier.csv"
    if not tolerance_source.exists():
        raise SystemExit(
            f"{tolerance_source} is missing. Run `pixi run bench-tolerance`."
        )
    tolerance_target = FIGURES / "tolerance_frontier.svg"
    tolerance_figure(pd.read_csv(tolerance_source), tolerance_target)
    print(f"[figures] wrote {tolerance_target.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        choices=("all", "headline"),
        default="all",
        help="render all result figures (default) or only the canonical headline figure",
    )
    arguments = parser.parse_args()
    if arguments.target == "headline":
        headline_main()
    else:
        main()

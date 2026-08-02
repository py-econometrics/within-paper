"""Generate the paper's result figures from recorded benchmark output.

Figures are drawn from the recorded CSV and JSON results, never by hand, so a
figure cannot drift from the table it replaces. Run after `analyze-gap-runtime`:

    pixi run make-figures
"""

from __future__ import annotations

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
    path = RESULTS / "gap_runtime_analysis.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run `pixi run analyze-gap-runtime` first."
        )
    return json.loads(path.read_text())["points"]


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


def _runtime_panel(
    ax,
    points: list[dict],
    backends: tuple[str, ...],
    *,
    view: str,
    title: str,
    note: str,
) -> None:
    """Draw one runtime-gap panel and fit slopes on the controlled AKM sweeps."""

    for raw_name in backends:
        display, colour, marker, line = METHODS[raw_name]
        usable = [
            row
            for row in points
            if row["backend"] == raw_name
            and row.get("view") == view
            and row.get("gap")
            and row.get("median_time")
            and row["gap"] > 0
            and row["median_time"] > 0
            and row["kind"] == "akm"
        ]
        if not usable:
            continue

        gaps = np.array([row["gap"] for row in usable])
        times = np.array([row["median_time"] for row in usable])
        # A cell that did not converge in every trial is drawn hollow, so a
        # censored point is visible rather than silently averaged in.
        complete = np.array(
            [row.get("n_success", 0) == row.get("n_trials", 0) for row in usable]
        )

        fit = None
        if gaps.size >= 3:
            fit = np.polyfit(np.log(gaps), np.log(times), 1)
        ax.scatter(
            gaps[complete],
            times[complete],
            s=34,
            c=colour,
            marker=marker,
            label=display,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        if (~complete).any():
            ax.scatter(
                gaps[~complete],
                times[~complete],
                s=40,
                facecolors="none",
                edgecolors=colour,
                marker=marker,
                linewidths=1.1,
                zorder=3,
            )

        if fit is not None:
            slope, intercept = fit
            grid = np.linspace(np.log(gaps.min()), np.log(gaps.max()), 64)
            ax.plot(
                np.exp(grid),
                np.exp(intercept + slope * grid),
                color=colour,
                linewidth=1.4,
                linestyle=line,
                alpha=0.55,
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(
        frameon=False,
        fontsize=6.2,
        loc="upper right",
        ncols=1,
        title="configuration",
        title_fontsize=6.3,
        labelspacing=0.38,
    )
    legend._legend_box.align = "left"
    ax.annotate(
        note,
        xy=(0.01, 0.03),
        xycoords="axes fraction",
        fontsize=6.4,
        color="#555555",
        linespacing=1.4,
    )


def headline_figure(points: list[dict], out: Path) -> None:
    """Runtime against the worker-firm gap, shown as performance and mechanism."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.35),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )

    _runtime_panel(
        axes[0],
        points,
        CROSS_PACKAGE_BACKENDS,
        view="default",
        title="(a) Cross-package defaults",
        note=(
            "solid: log-log fit to AKM medians\n"
            "hollow: not all trials converged"
        ),
    )
    _runtime_panel(
        axes[1],
        points,
        MECHANISM_BACKENDS,
        view="matched",
        title="(b) Same code path, matched accuracy",
        note=(
            "same AKM samples at 1M; log-log fits\n"
            "failed cells omitted; hollow: partial convergence"
        ),
    )

    axes[0].set_ylabel("Median runtime (s)")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.20)
    fig.text(
        0.535,
        0.05,
        "Worker-firm spectral gap  $1-\\rho_{WF}$  "
        "(weaker connectivity to the left)",
        ha="center",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


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
    points = _load_points()
    headline_target = FIGURES / "gap_runtime.svg"
    headline_figure(points, headline_target)
    print(
        f"[figures] wrote {headline_target.relative_to(ROOT)} "
        f"from {len(points)} points"
    )

    crossover_results = _load_crossover_results()
    crossover_target = FIGURES / "simple_difficult_runtime.svg"
    crossover_figure(crossover_results, crossover_target)
    print(
        f"[figures] wrote {crossover_target.relative_to(ROOT)} "
        "from recorded OLS and PPML trials"
    )

    tolerance_source = RESULTS / "tolerance_frontier.csv"
    if not tolerance_source.exists():
        raise SystemExit(
            f"{tolerance_source} is missing. Run `pixi run bench-tolerance`."
        )
    tolerance_target = FIGURES / "tolerance_frontier.svg"
    tolerance_figure(pd.read_csv(tolerance_source), tolerance_target)
    print(f"[figures] wrote {tolerance_target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Generate the paper's result figures from recorded benchmark output.

Figures are drawn from the recorded CSV and JSON results, never by hand, so a
figure cannot drift from the table it replaces. Run after `analyze-gap-runtime`:

    pixi run make-figures
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

from benchmarks.modular.methods import inline_label, linestyle, style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runs" / "latest"
BENCHMARK_RESULTS = ROOT / "benchmarks" / "results"
FIGURES = ROOT / "figures" / "results"

# The left panel compares package defaults. The right panel holds package and
# accuracy fixed, so it isolates the solver and preconditioner. These are the
# backend names as the result files spell them; every colour, marker, dash and
# label is derived from the registry, so nothing is restated here.
CROSS_PACKAGE_BACKENDS = (
    "pyfixest (rust-map)",
    "fixest-map",
    "FEM.jl (lsmr)",
    "pyfixest (within)",
)

MECHANISM_BACKENDS = (
    "pyfixest (rust-map, matched)",
    "pyfixest (within-off)",
    "pyfixest (within-diagonal)",
    "pyfixest (within-additive)",
)

CROSSOVER_FILES = {
    "OLS": {
        "rust-map": "feols_bench__pyfixest_rust_map.csv",
        "fixest": "feols_bench__fixest_map.csv",
        "Julia": "feols_bench__fem_jl_lsmr.csv",
        "within": "feols_bench__pyfixest_within.csv",
    },
    "PPML": {
        "rust-map": "fepois_bench__pyfixest_rust_map.csv",
        "fixest": "fepois_bench__fixest_fepois.csv",
        "Julia": "fepois_bench__glfixedeffectmodels_jl.csv",
        "within": "fepois_bench__pyfixest_within.csv",
    },
}

# The crossover figure draws one series for both Julia packages, so its label
# names both. Every other series takes the registry's inline label.
CROSSOVER_LABEL = {"Julia": "FEM.jl / GLFEM.jl — LSMR (diagonal)"}


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
    for model, files in CROSSOVER_FILES.items():
        model_results: dict[str, dict[str, object]] = {}
        for backend, filename in files.items():
            path = BENCHMARK_RESULTS / filename
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            grouped = {
                "simple": {"times": [], "n_trials": 0, "n_success": 0},
                "difficult": {"times": [], "n_trials": 0, "n_success": 0},
            }
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    design = row.get("dgp", "")
                    if design not in grouped:
                        continue
                    if int(row["n_obs"]) != target_n[model] or int(row["n_fe"]) != 3:
                        continue
                    record = grouped[design]
                    record["n_trials"] += 1
                    success = row.get("success", "").lower() in {"true", "1"}
                    runtime = row.get("time", "")
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
    show_context: bool,
    title: str,
    note: str,
) -> None:
    """Draw one runtime-gap panel and fit slopes on the controlled AKM sweeps."""

    for raw_name in backends:
        colour, marker = style(raw_name)
        display = inline_label(raw_name)
        usable = [
            row
            for row in points
            if row["backend"] == raw_name
            and row.get("gap")
            and row.get("median_time")
            and row["gap"] > 0
            and row["median_time"] > 0
        ]
        if not usable:
            continue

        # Only the AKM sweeps are a controlled comparison: one sample size, one
        # DGP, connectivity the single thing that moves. The Correia collection
        # and the 10M designs vary size and structure as well, so they are shown
        # for context but excluded from the fit. Fitting across them would
        # report a slope that mixes connectivity with scale.
        rows = [row for row in usable if row["kind"] == "akm"]
        context = [row for row in usable if row["kind"] != "akm"]
        if show_context and context:
            ax.scatter(
                [row["gap"] for row in context],
                [row["median_time"] for row in context],
                s=20,
                c=colour,
                marker=marker,
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
        if not rows:
            continue
        gaps = np.array([row["gap"] for row in rows])
        times = np.array([row["median_time"] for row in rows])
        # A cell that did not converge in every trial is drawn hollow, so a
        # censored point is visible rather than silently averaged in.
        complete = np.array(
            [row.get("n_success", 0) == row.get("n_trials", 0) for row in rows]
        )

        slope = None
        if gaps.size >= 3:
            slope = float(np.polyfit(np.log(gaps), np.log(times), 1)[0])
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

        if slope is not None:
            intercept = float(np.polyfit(np.log(gaps), np.log(times), 1)[1])
            grid = np.linspace(np.log(gaps.min()), np.log(gaps.max()), 64)
            ax.plot(
                np.exp(grid),
                np.exp(intercept + slope * grid),
                color=colour,
                linewidth=1.4,
                linestyle=linestyle(raw_name),
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
        show_context=False,
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
        show_context=False,
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
        "Julia": 0.025,
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
            display = CROSSOVER_LABEL.get(backend) or inline_label(backend)
            colour, marker = style(backend)
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
                linestyle=linestyle(backend),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    points = _load_points()
    headline_target = args.out_dir / "gap_runtime.svg"
    headline_figure(points, headline_target)
    print(
        f"[figures] wrote {headline_target.relative_to(ROOT)} "
        f"from {len(points)} points"
    )

    crossover_results = _load_crossover_results()
    crossover_target = args.out_dir / "simple_difficult_runtime.svg"
    crossover_figure(crossover_results, crossover_target)
    print(
        f"[figures] wrote {crossover_target.relative_to(ROOT)} "
        "from recorded OLS and PPML trials"
    )


if __name__ == "__main__":
    main()

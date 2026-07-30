"""Generate the paper's result figures from recorded benchmark output.

Figures are drawn from `results/runs/latest/`, never by hand, so a figure
cannot drift from the table it sits beside. Run after `analyze-gap-runtime`:

    pixi run make-figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "runs" / "latest"
FIGURES = ROOT / "figures" / "results"

# The four cross-package backends of the headline comparison. Labels are the
# names the analysis records, values are what the paper calls them.
HEADLINE_BACKENDS = {
    "pyfixest (rust-map)": "rust-map",
    "fixest-map": "fixest",
    "FEM.jl (lsmr)": "FEM.jl",
    "pyfixest (within)": "within",
}

# Colour-blind safe, and ordered so that within is the one that stands out.
STYLE = {
    "rust-map": ("#c2410c", "o"),
    "fixest": ("#a16207", "s"),
    "FEM.jl": ("#7b8494", "^"),
    "within": ("#2563eb", "D"),
}


def _load_points() -> list[dict]:
    path = RESULTS / "gap_runtime_analysis.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run `pixi run analyze-gap-runtime` first."
        )
    return json.loads(path.read_text())["points"]


def headline_figure(points: list[dict], out: Path) -> None:
    """Runtime against the worker-firm gap, four backends, all designs pooled.

    Every claim the paper makes is meant to be visible here: the MAP backends
    slope down as connectivity worsens, `within` does not, and the crossing is
    where the conditional advantage begins.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    for raw_name, label in HEADLINE_BACKENDS.items():
        colour, marker = STYLE[label]
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
        if context:
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
            label=label if slope is None else f"{label}  ({slope:+.2f})",
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
                alpha=0.55,
                zorder=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Worker-firm spectral gap  $1-\\rho_{WF}$  (weaker connectivity to the left)")
    ax.set_ylabel("Median runtime (s)")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    legend = ax.legend(
        frameon=False, fontsize=8.2, loc="upper right", ncols=2,
        title="backend  (fitted log-log slope)", title_fontsize=8.2,
    )
    legend._legend_box.align = "left"
    ax.annotate(
        "solid: AKM sweeps at 1M, one size and DGP, fitted\n"
        "faded: other designs, shown for context, not fitted\n"
        "hollow: not all trials converged",
        xy=(0.01, 0.03),
        xycoords="axes fraction",
        fontsize=6.8,
        color="#555555",
        linespacing=1.5,
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    points = _load_points()
    target = args.out_dir / "gap_runtime.svg"
    headline_figure(points, target)
    print(f"[figures] wrote {target.relative_to(ROOT)} from {len(points)} points")


if __name__ == "__main__":
    main()

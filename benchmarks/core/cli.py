"""Argument declarations shared by more than one experiment.

Only the options that are genuinely the same everywhere live here. The point is
not to save lines, it is that adding a third design to the standard pair should
be one edit rather than four, and that two experiments asked to write to the
results directory should not be able to disagree about where it is.

Options that differ by experiment stay with the experiment. `--n-obs` has four
different defaults across the drivers because the sensible sample size differs
by what is being measured, and centralising it would invite one of them to
inherit a size it was never calibrated for.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from benchmarks.core.paths import RESULTS_DIR, ROOT


# Where the package-comparison drivers write their raw per-trial CSVs.

# The standard pair of fixest designs. Several experiments sweep exactly these.
DEFAULT_DGPS = ("simple", "difficult")

# Timed repetitions when a driver does not choose adaptively. PROTOCOL.md R3.
DEFAULT_RUNS = 3


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """`--output-dir` and `--reuse-existing`, for the package-comparison drivers."""
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip cells whose output CSV already exists.",
    )


def add_dgps_arg(parser: argparse.ArgumentParser) -> None:
    """`--dgps`, defaulting to the standard simple/difficult pair."""
    parser.add_argument("--dgps", nargs="+", default=list(DEFAULT_DGPS))


def add_runs_arg(parser: argparse.ArgumentParser) -> None:
    """`--runs`, the timed repetition count."""
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)

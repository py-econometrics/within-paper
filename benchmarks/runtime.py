"""Shared process plumbing for native benchmark siblings."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]


def run_native(script: Path, arguments: list[str], output: Path) -> list[dict]:
    """Run one R or Julia sibling and return its CSV rows."""
    if script.suffix == ".R":
        command = ["Rscript", str(script), *arguments]
    else:
        command = [
            "julia",
            f"--project={ROOT / 'benchmarks' / 'julia-env'}",
            str(script),
            *arguments,
        ]
    environment = dict(os.environ)
    environment["JULIA_NUM_THREADS"] = environment["BENCH_THREADS"]
    subprocess.run(command, check=True, env=environment)
    return pd.read_csv(output).to_dict("records")


def assert_same_retained(rows: list[dict], model: str, design: str) -> None:
    """Reject a comparison when successful backends retained different samples."""
    retained = {
        int(float(row["n_retained"]))
        for row in rows
        if str(row["converged"]).lower() in {"true", "1"}
    }
    if len(retained) > 1:
        raise RuntimeError(
            f"{model} backends retained different row counts for {design}: "
            f"{sorted(retained)}"
        )

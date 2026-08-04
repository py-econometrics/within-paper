"""Shared process plumbing for native benchmark siblings."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]


def hit_iteration_cap(error: BaseException | str) -> bool:
    """Return whether an estimator explicitly reports an iteration limit."""
    if isinstance(error, BaseException) and type(error).__name__ == "NonConvergenceError":
        return True
    message = str(error).lower()
    return bool(
        re.search(
            r"demeaning failed after \d+ iterations|maximum number of iterations|"
            r"iteration (?:cap|limit)|maxiter|reaching the maximum.*iterations",
            message,
        )
    )


def failure_fields(error: BaseException | str) -> dict[str, object]:
    """Common fields for an estimator attempt that raised or returned failure."""
    return {
        "converged": False,
        "capped": hit_iteration_cap(error),
        "error": str(error),
    }


def failed_trials(backend: str, repetitions: int, error: BaseException | str) -> list[dict]:
    """Create a complete failed cell when an isolated backend process exits."""
    return [
        {
            "backend": backend,
            "repetition": repetition,
            "n_planned": repetitions,
            "runtime_s": None,
            "n_retained": None,
            "beta_x1": None,
            "max_eta": None,
            **failure_fields(error),
        }
        for repetition in range(repetitions)
    ]


def run_native(
    script: Path,
    arguments: list[str],
    output: Path,
    *,
    backend: str | None = None,
    failure_repetitions: int | None = None,
) -> list[dict]:
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
    if script.suffix != ".R":
        environment["JULIA_NUM_THREADS"] = environment["BENCH_THREADS"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, check=True, env=environment, text=True, capture_output=True
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        rows = pd.read_csv(output).to_dict("records")
    except Exception as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            native_output = "\n".join(
                text.strip() for text in (error.stdout, error.stderr) if text and text.strip()
            )
            if native_output:
                detail = f"{detail}: {native_output}"
        if backend is None or failure_repetitions is None:
            if detail != str(error):
                raise RuntimeError(detail) from error
            raise
        rows = failed_trials(backend, failure_repetitions, detail)
        elapsed = time.perf_counter() - started
        for row in rows:
            row["runtime_s"] = elapsed
        return rows

    if backend is None or failure_repetitions is None:
        return rows

    observed = {
        int(float(row["repetition"]))
        for row in rows
        if row.get("repetition") is not None
    }
    missing = [index for index in range(failure_repetitions) if index not in observed]
    if missing:
        message = "native driver returned no result row for the planned repetition"
        rows.extend(
            row
            for row in failed_trials(backend, failure_repetitions, message)
            if row["repetition"] in missing
        )
    return rows

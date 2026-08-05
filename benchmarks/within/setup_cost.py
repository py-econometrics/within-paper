"""Measure factor-pair setup and solve cost as connectivity changes."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.accuracy import accuracy_metrics
from benchmarks.akm import SCENARIOS, make_akm_data
from benchmarks.data import BASE_DESIGNS, make_base_data, solver_data
from benchmarks.runtime import failure_fields
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "within_setup_cost.csv"
AKM_OUTPUT = ROOT / "results" / "runs" / "latest" / "within_setup_cost_akm.csv"
BASE_N_OBS = 10_000_000
BASE_REPETITIONS = 3
AKM_REPETITIONS = 20
OPTIONS = LsmrOptions(tol=1e-12, maxiter=10_000)
REFERENCE_OPTIONS = LsmrOptions(tol=1e-14, maxiter=20_000)
MOBILITY_DESIGNS = tuple(name for name in SCENARIOS if name.startswith("akm_mobility_"))


def _measure(
    design: str,
    categories,
    rhs,
    *,
    experiment: str,
    repetitions: int,
) -> list[dict]:
    """Time construction and solves, with accuracy work outside the clock."""
    preconditioner = PreconditionerConfig.Additive
    # Warm-up materializes the native code before the first timed construction.
    try:
        solve_batch(categories, rhs, options=OPTIONS, preconditioner=preconditioner)
    except Exception:
        pass
    reference_result = solve_batch(
        categories,
        rhs,
        options=REFERENCE_OPTIONS,
        preconditioner=preconditioner,
    )
    if not all(reference_result.converged):
        raise RuntimeError(f"tight reference did not converge for {design}")
    reference = np.asarray(reference_result.demeaned)
    rows = []
    for repetition in range(repetitions):
        base = {
            "experiment": experiment,
            "design": design,
            "preconditioner": "additive",
            "repetition": repetition,
            "n_planned": repetitions,
            "n_obs": len(categories),
            "n_rhs": rhs.shape[1],
        }
        try:
            started = time.perf_counter()
            solver = Solver(categories, preconditioner=preconditioner)
            setup = time.perf_counter() - started
            started = time.perf_counter()
            result = solver.solve_batch(rhs, OPTIONS)
            solve = time.perf_counter() - started
            demeaned = np.asarray(result.demeaned)
            iterations = max(result.iterations)
            converged = all(result.converged)
            rows.append(
                {
                    **base,
                    "setup_s": setup,
                    "solve_s": solve,
                    "total_s": setup + solve,
                    "setup_share": setup / (setup + solve),
                    "iterations_max": iterations,
                    "converged": converged,
                    "capped": not converged and iterations >= OPTIONS.maxiter,
                    "error": (
                        "" if converged else "within solver returned without convergence"
                    ),
                    **accuracy_metrics(categories, rhs, demeaned, reference),
                }
            )
        except Exception as error:
            rows.append(
                {
                    **base,
                    "setup_s": None,
                    "solve_s": None,
                    "total_s": None,
                    "setup_share": None,
                    "iterations_max": None,
                    "max_eta": None,
                    "max_delta": None,
                    **failure_fields(error),
                }
            )
    return rows


def _print(design: str, rows: list[dict]) -> None:
    successful = [row for row in rows if row["converged"]]
    if successful:
        measured = pd.DataFrame(successful)
        value = (
            f"{measured.total_s.median():.3f} s "
            f"({measured.setup_share.median():.0%} setup)"
        )
    elif rows and all(row["capped"] for row in rows):
        value = "capped"
    else:
        value = "failed"
    print(
        f"within-setup-cost / OLS / {design} / within-additive: {value}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--akm",
        action="store_true",
        help="run the optional AKM mobility sweep instead of the paper endpoints",
    )
    args = parser.parse_args()
    if args.akm:
        samples = ((design, make_akm_data(design)) for design in MOBILITY_DESIGNS)
        experiment, repetitions, output = "akm_mobility", AKM_REPETITIONS, AKM_OUTPUT
    else:
        samples = (
            (design, make_base_data(BASE_N_OBS, design, seed))
            for design, seed in BASE_DESIGNS
        )
        experiment, repetitions, output = "base", BASE_REPETITIONS, OUTPUT

    rows = []
    for design, data in samples:
        categories, rhs = solver_data(data)
        measured = _measure(
            design,
            categories,
            rhs,
            experiment=experiment,
            repetitions=repetitions,
        )
        rows.extend(measured)
        _print(design, measured)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()

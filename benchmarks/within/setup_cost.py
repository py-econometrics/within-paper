"""Measure solver setup separately from repeated solves."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from benchmarks.data import BASE_DESIGNS, make_base_data, solver_data
from within import LsmrOptions, Solver, solve_batch

ROOT = Path(__file__).absolute().parents[2]
OUTPUT = ROOT / "results" / "runs" / "latest" / "within_setup_cost.csv"
N_OBS = 10_000_000
REPETITIONS = 3


def main() -> None:
    rows = []
    for design, seed in BASE_DESIGNS:
        categories, rhs = solver_data(make_base_data(N_OBS, design, seed))
        options = LsmrOptions()
        solve_batch(categories, rhs, options)
        for repetition in range(REPETITIONS):
            started = time.perf_counter()
            solver = Solver(categories)
            setup = time.perf_counter() - started
            started = time.perf_counter()
            result = solver.solve_batch(rhs, options)
            solve = time.perf_counter() - started
            rows.append(
                {
                    "design": design,
                    "repetition": repetition,
                    "n_obs": len(categories),
                    "n_rhs": rhs.shape[1],
                    "setup_s": setup,
                    "solve_s": solve,
                    "total_s": setup + solve,
                    "iterations_max": max(result.iterations),
                    "converged": all(result.converged),
                }
            )
        print(
            f"within-setup-cost / OLS / {design} / within: "
            f"{pd.DataFrame(rows[-REPETITIONS:]).total_s.median():.3f} s",
            flush=True,
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

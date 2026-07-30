"""Measure setup and solve time for the standalone within Python API.

This benchmark isolates the setup cost behind the PyFixest `within` backend.
It uses the simple/difficult DGP generator from the PyFixest benchmark suite,
then calls the standalone `within` Python API directly:

- `Solver(categories, ...)` measures reusable solver/preconditioner setup.
- `solver.solve_batch(Y)` measures solves after setup for y and the covariates.
- `solve_batch(categories, Y, ...)` measures one-shot setup plus solve.

The default scale matches the standard synthetic benchmark in the paper:
10M observations, one covariate, and three fixed effects.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from benchmarks.modular.experiment import (
    SampleSpec,
    add_repo_paths,
    clear_sample_cache,
    load_sample,
    write_rows,
)


import within
from within import LsmrOptions, Solver, solve_batch


def _setup_share(setup_wall: float, solve_wall: float) -> float:
    total = setup_wall + solve_wall
    if total <= 0:
        raise ValueError("setup and solve time must sum to a positive value")
    return setup_wall / total


def _run_once(dgp: str, n_obs: int, k: int, iteration: int) -> dict:
    # One fixed sample per design; repetitions must not redraw it.
    sample = load_sample(SampleSpec(design=dgp, n_obs=n_obs, k=k))
    categories, rhs = sample.categories, sample.rhs
    config = LsmrOptions()

    # Warm the allocator, memory pages, and lazy initialization before timing.
    # Otherwise setup alone bears these one-time costs and its reported share can
    # exceed one. The discarded one-shot solve warms both setup and solve paths.
    _ = solve_batch(categories, rhs, config)
    del _

    gc.collect()
    t0 = time.perf_counter()
    solver = Solver(categories)
    setup_wall = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    reused = solver.solve_batch(rhs, config)
    solve_wall = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    oneshot = solve_batch(categories, rhs, config)
    full_wall = time.perf_counter() - t0

    row = {
        "dgp": dgp,
        "n_obs": n_obs,
        "k": k,
        "n_rhs": rhs.shape[1],
        "iteration": iteration,
        "setup_wall_s": setup_wall,
        "solve_after_setup_wall_s": solve_wall,
        "full_oneshot_wall_s": full_wall,
        "setup_share_of_full": setup_wall / full_wall,
        "setup_share_of_reused_total": _setup_share(setup_wall, solve_wall),
        "reused_result_time_total_s": reused.time_total,
        "oneshot_result_time_total_s": oneshot.time_total,
        "max_iterations_reused": max(reused.iterations),
        "max_iterations_oneshot": max(oneshot.iterations),
        "max_residual_reused": max(reused.residual),
        "max_residual_oneshot": max(oneshot.residual),
        "all_converged_reused": all(reused.converged),
        "all_converged_oneshot": all(oneshot.converged),
    }
    del solver, reused, oneshot
    gc.collect()
    return row


def _median_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dgp"], row["n_obs"], row["k"]), []).append(row)

    summary = []
    numeric_cols = [
        "setup_wall_s",
        "solve_after_setup_wall_s",
        "full_oneshot_wall_s",
        "setup_share_of_full",
        "setup_share_of_reused_total",
        "reused_result_time_total_s",
        "oneshot_result_time_total_s",
        "max_iterations_reused",
        "max_iterations_oneshot",
        "max_residual_reused",
        "max_residual_oneshot",
    ]
    for (dgp, n_obs, k), group in grouped.items():
        out = {
            "dgp": dgp,
            "n_obs": n_obs,
            "k": k,
            "n_rhs": group[0]["n_rhs"],
            "n_runs": len(group),
            "all_converged_reused": all(row["all_converged_reused"] for row in group),
            "all_converged_oneshot": all(row["all_converged_oneshot"] for row in group),
        }
        for col in numeric_cols:
            out[f"median_{col}"] = float(np.median([row[col] for row in group]))
        summary.append(out)
    return summary



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=10_000_000)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dgps", nargs="+", default=["simple", "difficult"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "runs" / "latest" / "within_setup_cost.csv",
    )
    args = parser.parse_args()

    print(f"[within-setup] using {within.__file__}", flush=True)

    rows = []
    for dgp in args.dgps:
        for iteration in range(args.runs):
            print(
                f"[within-setup] dgp={dgp} n={args.n_obs:,} "
                f"k={args.k} run={iteration + 1}/{args.runs}",
                flush=True,
            )
            row = _run_once(dgp, args.n_obs, args.k, iteration)
            rows.append(row)
            print(
                "  setup={setup_wall_s:.3f}s solve-after-setup="
                "{solve_after_setup_wall_s:.3f}s full={full_oneshot_wall_s:.3f}s "
                "setup-share={setup_share_of_reused_total:.1%}".format(**row),
                flush=True,
            )

    write_rows(args.out, rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary = _median_rows(rows)
    write_rows(summary_path, summary)

    print("\nMedian summary")
    for row in summary:
        print(
            f"{row['dgp']:<10} n={row['n_obs']:,} k={row['k']} "
            f"setup={row['median_setup_wall_s']:.3f}s "
            f"solve={row['median_solve_after_setup_wall_s']:.3f}s "
            f"full={row['median_full_oneshot_wall_s']:.3f}s "
            f"setup-share={row['median_setup_share_of_reused_total']:.1%} "
            f"iters={row['median_max_iterations_reused']:.0f}",
            flush=True,
        )
    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

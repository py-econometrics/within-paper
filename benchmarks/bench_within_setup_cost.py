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
import csv
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WITHIN_REPO = Path("/Users/afischer/Documents/within")
DEFAULT_PYFIXEST_REPO = Path("/Users/afischer/Documents/pyfixest")
FE_COLS = ["indiv_id", "firm_id", "year"]


def _add_repo_paths() -> None:
    within_repo = Path(os.environ.get("WITHIN_REPO", DEFAULT_WITHIN_REPO))
    pyfixest_repo = Path(os.environ.get("PYFIXEST_REPO", DEFAULT_PYFIXEST_REPO))
    for path in [
        within_repo / "python",
        pyfixest_repo / "benchmarks" / "modular",
    ]:
        if path.exists():
            sys.path.insert(0, str(path))


_add_repo_paths()

from dgp_functions import base_dgp  # noqa: E402
from within import CG, Solver, solve_batch  # noqa: E402


def _seed_for(dgp: str, n_obs: int, iteration: int) -> int:
    dgp_offset = {"simple": 0, "difficult": 1}[dgp]
    return n_obs * 100 + iteration * 17 + dgp_offset + 42


def _make_problem(dgp: str, n_obs: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    df = base_dgp(n=n_obs, type_=dgp, k=k, max_k=k, seed=seed)
    categories = np.asfortranarray(df[FE_COLS].to_numpy(dtype=np.uint32) - 1)
    rhs_cols = ["y", *[f"x{i}" for i in range(1, k + 1)]]
    rhs = np.asfortranarray(df[rhs_cols].to_numpy(dtype=np.float64))
    del df
    gc.collect()
    return categories, rhs


def _run_once(dgp: str, n_obs: int, k: int, iteration: int, tol: float, maxiter: int) -> dict:
    categories, rhs = _make_problem(dgp, n_obs, k, _seed_for(dgp, n_obs, iteration))
    config = CG(tol=tol, maxiter=maxiter)

    gc.collect()
    t0 = time.perf_counter()
    solver = Solver(categories, config)
    setup_wall = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    reused = solver.solve_batch(rhs)
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
        "reused_result_time_total_s": reused.time_total,
        "oneshot_result_time_total_s": oneshot.time_total,
        "max_iterations_reused": max(reused.iterations),
        "max_iterations_oneshot": max(oneshot.iterations),
        "max_residual_reused": max(reused.residual),
        "max_residual_oneshot": max(oneshot.residual),
        "all_converged_reused": all(reused.converged),
        "all_converged_oneshot": all(oneshot.converged),
    }
    del categories, rhs, solver, reused, oneshot
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=10_000_000)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dgps", nargs="+", default=["simple", "difficult"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=100_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "within_setup_cost.csv",
    )
    args = parser.parse_args()

    rows = []
    for dgp in args.dgps:
        for iteration in range(args.runs):
            print(
                f"[within-setup] dgp={dgp} n={args.n_obs:,} "
                f"k={args.k} run={iteration + 1}/{args.runs}",
                flush=True,
            )
            row = _run_once(dgp, args.n_obs, args.k, iteration, args.tol, args.maxiter)
            rows.append(row)
            print(
                "  setup={setup_wall_s:.3f}s solve-after-setup="
                "{solve_after_setup_wall_s:.3f}s full={full_oneshot_wall_s:.3f}s "
                "setup-share={setup_share_of_full:.1%}".format(**row),
                flush=True,
            )

    _write_csv(args.out, rows)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary = _median_rows(rows)
    _write_csv(summary_path, summary)

    print("\nMedian summary")
    for row in summary:
        print(
            f"{row['dgp']:<10} n={row['n_obs']:,} k={row['k']} "
            f"setup={row['median_setup_wall_s']:.3f}s "
            f"solve={row['median_solve_after_setup_wall_s']:.3f}s "
            f"full={row['median_full_oneshot_wall_s']:.3f}s "
            f"setup-share={row['median_setup_share_of_full']:.1%} "
            f"iters={row['median_max_iterations_reused']:.0f}",
            flush=True,
        )
    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

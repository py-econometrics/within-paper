"""Scaling in the number of absorbed factors, and amortization over right-hand sides.

Two experiments the plan asks for and the benchmarks did not cover.

Factor scaling (plan item 15). The Schwarz construction enumerates all
Q(Q-1)/2 factor pairs, so subdomains, overlap, and the partition weights all
move with Q. Every other experiment in the paper holds Q at three, which is
exactly the axis the construction is most exposed on. Q in {2,3,4,5} runs on
nested subsets of one sample, so the comparison changes the factor count and
nothing else.

Amortization (plan item 17). The preconditioner is built once and reused across
right-hand sides, but every headline regression has a single covariate, which
is the least favorable case. Sweeping K right-hand sides shows where the
additive setup repays itself against diagonal scaling, and the break-even is
read off the measurements rather than from a closed form that assumes marginal
solve cost is exactly linear in K.

Run with:

    pixi run bench-scaling
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "modular"))

from experiment import (  # noqa: E402
    RunRecord,
    SampleSpec,
    add_repo_paths,
    load_sample,
    preconditioner_config,
    sample_hash,
    write_records,
)

add_repo_paths()

import within  # noqa: E402
from accuracy import accuracy_record  # noqa: E402
from feols_benchmarkers import MECHANISM_LSMR_TOL, MECHANISM_MAXITER  # noqa: E402
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch  # noqa: E402

MAX_FACTORS = 5


def _extended_categories(base: np.ndarray, n_obs: int, seed: int) -> np.ndarray:
    """Add two further factors so that Q can run to five on one sample.

    The extra factors are drawn independently of the worker-firm structure with
    moderate cardinality, which is the neutral case: they neither rescue a
    weakly connected design nor break a well connected one, so what the sweep
    measures is the cost of carrying more pairs rather than a change in
    difficulty.
    """
    rng = np.random.default_rng(seed)
    extra = np.column_stack(
        [
            rng.integers(0, max(2, n_obs // 5_000), n_obs),
            rng.integers(0, max(2, n_obs // 20_000), n_obs),
        ]
    )
    return np.column_stack([base, extra]).astype(np.uint32)


def _timed_solve(categories, rhs, preconditioner: str, options) -> dict:
    config = preconditioner_config(preconditioner)

    warm = solve_batch(categories, rhs, options, preconditioner=config)
    del warm
    gc.collect()

    start = time.perf_counter()
    solver = Solver(categories, preconditioner=config)
    setup_s = time.perf_counter() - start

    gc.collect()
    start = time.perf_counter()
    result = solver.solve_batch(rhs, options)
    solve_s = time.perf_counter() - start

    iterations = list(result.iterations)
    converged = list(result.converged)
    payload = {
        "setup_s": setup_s,
        "solve_s": solve_s,
        "iterations_max": int(max(iterations)) if iterations else None,
        "iterations_sum": int(sum(iterations)) if iterations else None,
        "iterations_median": float(np.median(iterations)) if iterations else None,
        "n_converged": sum(1 for ok in converged if ok),
        "n_solves": len(converged),
        "demeaned": np.asarray(result.demeaned),
        "n_dofs": int(solver.n_dofs),
    }
    del solver, result
    gc.collect()
    return payload


def factor_scaling(design: str, n_obs: int, repetitions: int) -> list[RunRecord]:
    sample = load_sample(SampleSpec(design=design, n_obs=n_obs))
    full = np.asfortranarray(
        _extended_categories(np.asarray(sample.categories), n_obs, seed=90210)
    )
    rhs = sample.rhs
    options = LsmrOptions(tol=MECHANISM_LSMR_TOL, maxiter=MECHANISM_MAXITER)

    records: list[RunRecord] = []
    for n_factors in range(2, MAX_FACTORS + 1):
        categories = np.asfortranarray(full[:, :n_factors])
        digest = sample_hash(categories, np.asarray(rhs))
        reference = solve_batch(
            categories,
            rhs,
            LsmrOptions(tol=1e-14, maxiter=MECHANISM_MAXITER),
            preconditioner=PreconditionerConfig.Additive,
        )
        reference_demeaned = np.array(reference.demeaned, copy=True)
        del reference

        for repetition in range(repetitions):
            print(f"[scaling] Q={n_factors} rep={repetition + 1}/{repetitions}", flush=True)
            payload = _timed_solve(categories, rhs, "additive", options)
            accuracy = accuracy_record(
                categories=np.asarray(categories),
                rhs=np.asarray(rhs),
                demeaned=payload["demeaned"],
                reference_demeaned=reference_demeaned,
            )
            records.append(
                RunRecord(
                    design=design,
                    n_obs=n_obs,
                    sample_hash=digest,
                    config_id=f"additive/Q={n_factors}",
                    solver_label="within-additive",
                    view="factor-scaling",
                    repetition=repetition,
                    setup_s=payload["setup_s"],
                    solve_s=payload["solve_s"],
                    total_s=payload["setup_s"] + payload["solve_s"],
                    converged=payload["n_converged"] == payload["n_solves"],
                    censoring="none"
                    if payload["n_converged"] == payload["n_solves"]
                    else "capped",
                    iterations_median=payload["iterations_median"],
                    iterations_max=payload["iterations_max"],
                    iterations_sum=payload["iterations_sum"],
                    n_converged=payload["n_converged"],
                    n_solves=payload["n_solves"],
                    max_eta=accuracy.max_eta,
                    max_delta=accuracy.max_delta,
                    gate_a_measured=accuracy.gate_a_components_measured,
                    clears_gate_a=accuracy.clears_gate_a,
                    extra={
                        "n_factors": n_factors,
                        "n_pairs": n_factors * (n_factors - 1) // 2,
                        "n_dofs": payload["n_dofs"],
                        "setup_share": payload["setup_s"]
                        / (payload["setup_s"] + payload["solve_s"]),
                    },
                )
            )
    return records


def amortization(
    design: str, n_obs: int, k_values: list[int], repetitions: int
) -> list[RunRecord]:
    sample = load_sample(SampleSpec(design=design, n_obs=n_obs))
    categories = sample.categories
    options = LsmrOptions(tol=MECHANISM_LSMR_TOL, maxiter=MECHANISM_MAXITER)
    rng = np.random.default_rng(4242)
    widest = np.asfortranarray(
        np.column_stack(
            [np.asarray(sample.rhs), rng.standard_normal((n_obs, max(k_values)))]
        )
    )

    records: list[RunRecord] = []
    for k in k_values:
        rhs = np.asfortranarray(widest[:, :k])
        digest = sample_hash(np.asarray(categories), rhs)
        for preconditioner in ("diagonal", "additive"):
            for repetition in range(repetitions):
                print(
                    f"[amortize] K={k} pc={preconditioner} "
                    f"rep={repetition + 1}/{repetitions}",
                    flush=True,
                )
                payload = _timed_solve(categories, rhs, preconditioner, options)
                accuracy = accuracy_record(
                    categories=np.asarray(categories),
                    rhs=rhs,
                    demeaned=payload["demeaned"],
                )
                total = payload["setup_s"] + payload["solve_s"]
                records.append(
                    RunRecord(
                        design=design,
                        n_obs=n_obs,
                        sample_hash=digest,
                        config_id=f"{preconditioner}/K={k}",
                        solver_label=f"within-{preconditioner}",
                        view="amortization",
                        repetition=repetition,
                        setup_s=payload["setup_s"],
                        solve_s=payload["solve_s"],
                        total_s=total,
                        converged=payload["n_converged"] == payload["n_solves"],
                        censoring="none"
                        if payload["n_converged"] == payload["n_solves"]
                        else "capped",
                        iterations_median=payload["iterations_median"],
                        iterations_max=payload["iterations_max"],
                        iterations_sum=payload["iterations_sum"],
                        n_converged=payload["n_converged"],
                        n_solves=payload["n_solves"],
                        max_eta=accuracy.max_eta,
                        gate_a_measured=accuracy.gate_a_components_measured,
                        clears_gate_a=accuracy.clears_gate_a,
                        extra={
                            "k_rhs": k,
                            "preconditioner": preconditioner,
                            "time_per_rhs_s": total / k,
                        },
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="difficult")
    parser.add_argument("--n-obs", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 5, 10, 25])
    parser.add_argument("--skip-factor-scaling", action="store_true")
    parser.add_argument("--skip-amortization", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "runs" / "latest"
    )
    args = parser.parse_args()

    print(f"[scaling] using {within.__file__}", flush=True)

    if not args.skip_factor_scaling:
        records = factor_scaling(args.design, args.n_obs, args.runs)
        target = args.out_dir / "factor_scaling.csv"
        write_records(target, records)
        print(f"Wrote {target}")

    if not args.skip_amortization:
        records = amortization(args.design, args.n_obs, args.k_values, args.runs)
        target = args.out_dir / "amortization.csv"
        write_records(target, records)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()

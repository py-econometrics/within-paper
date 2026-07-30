"""Standalone within diagnostics for the three preconditioner settings.

Pairs with the PyFixest end-to-end timings. This path records what `feols` does
not expose:

- preconditioner setup time separately from solve time
- per-RHS LSMR iterations (median, max, sum) and convergence flags
- the internal residual beside the external normal-equation residual eta
- pair-graph edge counts and density (a fill proxy)
- projection error against a tight reference

Sample identity, solver settings, and the record schema come from
`experiment.py`, so this driver measures the same samples at the same settings
as the end-to-end run rather than its own variants.

Run with:

    pixi run within-preconditioners
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
    PRECONDITIONERS,
    RunRecord,
    SampleSpec,
    add_repo_paths,
    clear_sample_cache,
    load_sample,
    matched_solver_specs,
    preconditioner_config,
    write_records,
)

add_repo_paths()

import within  # noqa: E402
from accuracy import accuracy_record, pair_edge_stats  # noqa: E402
from feols_benchmarkers import MECHANISM_MAP_TOL, MECHANISM_MAXITER  # noqa: E402
from map_diagnostics import map_demean_with_sweeps  # noqa: E402
from within import LsmrOptions, PreconditionerConfig, Solver, solve_batch  # noqa: E402


def _edge_summary(categories: np.ndarray) -> dict[str, float | int]:
    stats = pair_edge_stats(categories)
    if not stats:
        return {
            "n_pairs": 0,
            "total_edges": 0,
            "max_pair_density": 0.0,
            "mean_pair_density": 0.0,
        }
    densities = [float(row["density"]) for row in stats]
    return {
        "n_pairs": len(stats),
        "total_edges": int(sum(int(row["n_edges"]) for row in stats)),
        "max_pair_density": float(max(densities)),
        "mean_pair_density": float(np.mean(densities)),
    }


def _run_map(sample, repetition: int, reference_tol: float) -> RunRecord:
    """One timed MAP trial on the same sample, counted in sweeps.

    A sweep is one full pass over the absorbed factors and is not comparable to
    an LSMR iteration, so the two are recorded under different labels and the
    paper never adds them. Reported here rather than in a separate driver
    because the comparison is only meaningful on the identical sample, and the
    reference solve that both are scored against is built once.
    """
    categories, rhs = sample.categories, sample.rhs

    # Warm lazy initialization before timing, as the LSMR arm does.
    del_me = map_demean_with_sweeps(
        np.asarray(rhs), categories, tol=MECHANISM_MAP_TOL, maxiter=1
    )
    del del_me

    gc.collect()
    t0 = time.perf_counter()
    result = map_demean_with_sweeps(
        np.asarray(rhs),
        categories,
        tol=MECHANISM_MAP_TOL,
        maxiter=MECHANISM_MAXITER,
    )
    solve_wall = time.perf_counter() - t0

    reference = solve_batch(
        categories,
        rhs,
        LsmrOptions(tol=reference_tol, maxiter=max(MECHANISM_MAXITER, 20_000)),
        preconditioner=PreconditionerConfig.Additive,
    )
    accuracy = accuracy_record(
        categories=categories,
        rhs=rhs,
        demeaned=result.demeaned,
        reference_demeaned=np.asarray(reference.demeaned),
    )

    summary = result.summary_row()
    censoring = "capped" if result.any_capped else (
        "none" if result.n_converged == len(result.converged) else "failed"
    )
    record = RunRecord(
        design=sample.spec.design,
        n_obs=sample.spec.n_obs,
        sample_hash=sample.sample_hash,
        config_id=f"map/tol={MECHANISM_MAP_TOL:g}/maxiter={MECHANISM_MAXITER}",
        solver_label="map",
        view="matched-accuracy",
        repetition=repetition,
        # MAP builds no operator, so there is no setup component to report. A
        # zero would read as "measured and negligible" rather than "not
        # applicable", which is the distinction the cost table turns on.
        setup_s=None,
        solve_s=solve_wall,
        total_s=solve_wall,
        converged=result.n_converged == len(result.converged),
        censoring=censoring,
        iterations_median=summary["map_iterations_median"],
        iterations_max=summary["map_iterations_max"],
        iterations_sum=summary["map_iterations_sum"],
        n_converged=result.n_converged,
        n_solves=len(result.converged),
        max_eta=accuracy.max_eta,
        max_delta=accuracy.max_delta,
        max_slope_se=accuracy.max_slope_se,
        gate_a_measured=accuracy.gate_a_components_measured,
        clears_gate_a=accuracy.clears_gate_a,
        extra={
            "k": sample.spec.k,
            "n_rhs": sample.n_rhs,
            "preconditioner": "map",
            "tol": MECHANISM_MAP_TOL,
            "maxiter": MECHANISM_MAXITER,
            "iteration_unit": "map-sweep",
            **summary,
        },
    )
    del result, reference
    gc.collect()
    return record


def _run_once(sample, solver_spec, repetition: int, reference_tol: float) -> RunRecord:
    """One timed trial of one solver configuration on one fixed sample."""
    categories, rhs = sample.categories, sample.rhs
    edges = _edge_summary(categories)
    config = LsmrOptions(tol=solver_spec.tol, maxiter=solver_spec.maxiter)
    pc = preconditioner_config(solver_spec.preconditioner)

    # Warm lazy initialization before timing (PROTOCOL.md section 3).
    del_me = solve_batch(categories, rhs, config, preconditioner=pc)
    del del_me

    gc.collect()
    t0 = time.perf_counter()
    solver = Solver(categories, preconditioner=pc)
    setup_wall = time.perf_counter() - t0

    gc.collect()
    t0 = time.perf_counter()
    result = solver.solve_batch(rhs, config)
    solve_wall = time.perf_counter() - t0

    iterations = list(result.iterations)
    residuals = list(result.residual)
    converged = list(result.converged)
    n_converged = sum(1 for ok in converged if ok)
    n_capped = sum(
        1
        for ok, n_it in zip(converged, iterations, strict=True)
        if (not ok) and n_it >= solver_spec.maxiter
    )

    reference = solve_batch(
        categories,
        rhs,
        LsmrOptions(tol=reference_tol, maxiter=max(solver_spec.maxiter, 20_000)),
        preconditioner=PreconditionerConfig.Additive,
    )
    accuracy = accuracy_record(
        categories=categories,
        rhs=rhs,
        demeaned=np.asarray(result.demeaned),
        reference_demeaned=np.asarray(reference.demeaned),
    )

    if n_capped:
        censoring = "capped"
    elif n_converged < len(converged):
        censoring = "failed"
    else:
        censoring = "none"

    record = RunRecord(
        design=sample.spec.design,
        n_obs=sample.spec.n_obs,
        sample_hash=sample.sample_hash,
        config_id=solver_spec.config_id,
        solver_label=solver_spec.label,
        view=solver_spec.view,
        repetition=repetition,
        setup_s=setup_wall,
        solve_s=solve_wall,
        total_s=setup_wall + solve_wall,
        converged=bool(converged) and all(converged),
        censoring=censoring,
        iterations_median=float(np.median(iterations)) if iterations else None,
        iterations_max=int(max(iterations)) if iterations else None,
        iterations_sum=int(sum(iterations)) if iterations else None,
        n_converged=n_converged,
        n_solves=len(converged),
        max_eta=accuracy.max_eta,
        max_delta=accuracy.max_delta,
        max_slope_se=accuracy.max_slope_se,
        gate_a_measured=accuracy.gate_a_components_measured,
        clears_gate_a=accuracy.clears_gate_a,
        extra={
            "k": sample.spec.k,
            "n_rhs": sample.n_rhs,
            "preconditioner": solver_spec.preconditioner,
            "tol": solver_spec.tol,
            "maxiter": solver_spec.maxiter,
            "setup_share": setup_wall / (setup_wall + solve_wall)
            if (setup_wall + solve_wall) > 0
            else None,
            "result_time_total_s": result.time_total,
            "n_dofs": int(solver.n_dofs),
            "internal_residual_max": float(max(residuals)) if residuals else None,
            "internal_residual_median": float(np.median(residuals))
            if residuals
            else None,
            "n_capped": n_capped,
            "iteration_unit": "lsmr-iteration",
            "gate_a_eta": accuracy.gate_a_eta,
            "gate_a_delta": accuracy.gate_a_delta,
            # The published within build does not expose retained fill nonzeros;
            # the edge counts below stand in as the density proxy.
            "fill_nonzeros": None,
            **edges,
        },
    )
    del solver, result, reference
    gc.collect()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--designs",
        nargs="+",
        default=["simple", "difficult"],
        help="simple, difficult, or any akm_mobility_* / akm_sorting_* design",
    )
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        default=list(PRECONDITIONERS),
        choices=list(PRECONDITIONERS),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--with-map",
        action="store_true",
        help="Also run counting MAP, recorded in sweeps rather than iterations.",
    )
    parser.add_argument("--reference-tol", type=float, default=1e-14)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "runs" / "latest" / "within_preconditioners.csv",
    )
    args = parser.parse_args()

    print(f"[within-preconditioners] using {within.__file__}", flush=True)
    solvers = [
        spec
        for spec in matched_solver_specs()
        if spec.preconditioner in args.preconditioners
    ]

    records: list[RunRecord] = []
    for design in args.designs:
        spec = SampleSpec(design=design, n_obs=args.n_obs, k=args.k)
        # One sample per design, reused by every configuration and repetition.
        sample = load_sample(spec)
        if args.with_map:
            for repetition in range(args.runs):
                print(
                    f"[within-preconditioners] {design} pc=map "
                    f"n={args.n_obs:,} rep={repetition + 1}/{args.runs}",
                    flush=True,
                )
                record = _run_map(sample, repetition, args.reference_tol)
                records.append(record)
                print(
                    f"  solve={record.solve_s:.3f}s sweeps_max={record.iterations_max} "
                    f"eta={record.max_eta:.2e} censoring={record.censoring}",
                    flush=True,
                )
        for solver_spec in solvers:
            for repetition in range(args.runs):
                print(
                    f"[within-preconditioners] {design} pc={solver_spec.preconditioner} "
                    f"n={args.n_obs:,} rep={repetition + 1}/{args.runs}",
                    flush=True,
                )
                record = _run_once(
                    sample, solver_spec, repetition, args.reference_tol
                )
                records.append(record)
                print(
                    f"  setup={record.setup_s:.3f}s solve={record.solve_s:.3f}s "
                    f"iters_max={record.iterations_max} eta={record.max_eta:.2e} "
                    f"delta={record.max_delta:.2e} censoring={record.censoring}",
                    flush=True,
                )
        clear_sample_cache()

    write_records(args.out, records)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

"""Correctness pilot that calibrates and freezes the measurement rules.

Track B phase 3 of the revision plan. This runs before any production
benchmark and answers four questions whose answers PROTOCOL.md then fixes:

1. Does the external residual eta measure what it claims? Checked against a
   dense minimum-norm solve on a design small enough to form D explicitly.
2. What solver tolerance actually achieves Gate A? The stopping rule bounds a
   relative normal-equation residual computed from bidiagonalization scalars,
   which is not the same number as the externally recomputed eta, so the
   nominal tolerance is not the achieved accuracy.
3. Do off, diagonal, and additive reach the same observation-space projection?
   If they did not, the ablation would be comparing different answers.
4. Are capped runs reported as capped, at every level that can cap?

Run with:

    pixi run pilot-calibration
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from benchmarks.modular.cli import add_dgps_arg
from benchmarks.modular.experiment import FE_COLS
from benchmarks.modular.accuracy import (
    GATE_A_DELTA,
    GATE_A_ETA,
    external_normal_residuals,
    projection_errors,
)
from benchmarks.modular.map_diagnostics import map_demean_with_sweeps
from benchmarks.modular.settings import demeaner_for

RHS_COLS = ["y", "x1"]
TOLERANCE_GRID = (1e-8, 1e-10, 1e-12, 1e-14)
REFERENCE_TOL = 1e-14
REFERENCE_MAXITER = 20_000


def _preconditioner_configs():
    from within import PreconditionerConfig

    return {
        "off": PreconditionerConfig.Off,
        "diagonal": PreconditionerConfig.Diagonal,
        "additive": PreconditionerConfig.Additive,
    }


def _load_design(dgp: str, n_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (int categories, uint32 Fortran categories, right-hand sides)."""
    path = ROOT / "benchmarks" / "data" / f"{dgp}_{n_obs // 1000}k.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `pixi run bench-generate-data` first."
        )
    frame = pd.read_parquet(path)
    codes = np.array(frame[FE_COLS].to_numpy(dtype=np.int64), copy=True)
    codes -= codes.min(axis=0)
    rhs = np.asfortranarray(
        np.array(frame[RHS_COLS].to_numpy(dtype=np.float64), copy=True)
    )
    return codes, np.asfortranarray(codes.astype(np.uint32)), rhs


# ---------------------------------------------------------------------------
# 1. Validate eta against a dense reference
# ---------------------------------------------------------------------------
def validate_external_residual(seed: int = 7) -> dict:
    """Compare the group-sum eta against a dense pinv solve.

    Small enough that D fits in memory, so the comparison is exact rather than
    another iterative solve checking an iterative solve.
    """
    rng = np.random.default_rng(seed)
    n_obs = 3_000
    codes = np.column_stack(
        [rng.integers(0, 120, n_obs), rng.integers(0, 40, n_obs), rng.integers(0, 6, n_obs)]
    )
    offsets = [0]
    for factor in range(codes.shape[1]):
        offsets.append(offsets[-1] + int(codes[:, factor].max()) + 1)
    design = np.zeros((n_obs, offsets[-1]))
    for row in range(n_obs):
        for factor in range(codes.shape[1]):
            design[row, offsets[factor] + int(codes[row, factor])] = 1.0

    rhs = rng.standard_normal((n_obs, 3))
    residual = rhs - design @ (np.linalg.pinv(design) @ rhs)

    helper = external_normal_residuals(codes, rhs, residual)
    dense = np.array(
        [
            np.linalg.norm(design.T @ residual[:, j])
            / np.linalg.norm(design.T @ rhs[:, j])
            for j in range(rhs.shape[1])
        ]
    )
    perturbed = residual + 1e-3 * design @ rng.standard_normal((design.shape[1], 3))
    return {
        "eta_direct_max": float(helper.max()),
        "helper_vs_dense_max_abs_diff": float(np.abs(helper - dense).max()),
        "eta_perturbed_min": float(external_normal_residuals(codes, rhs, perturbed).min()),
    }


# ---------------------------------------------------------------------------
# 2 and 3. Tolerance sweep and same-projection check
# ---------------------------------------------------------------------------
def tolerance_sweep(dgp: str, n_obs: int) -> list[dict]:
    from within import LsmrOptions, solve_batch

    codes, categories, rhs = _load_design(dgp, n_obs)
    configs = _preconditioner_configs()

    reference = solve_batch(
        categories,
        rhs,
        LsmrOptions(tol=REFERENCE_TOL, maxiter=REFERENCE_MAXITER),
        preconditioner=configs["additive"],
    )
    reference_demeaned = np.array(reference.demeaned, copy=True)

    rows: list[dict] = []
    for name, config in configs.items():
        for tol in TOLERANCE_GRID:
            start = time.perf_counter()
            result = solve_batch(
                categories,
                rhs,
                LsmrOptions(tol=tol, maxiter=REFERENCE_MAXITER),
                preconditioner=config,
            )
            elapsed = time.perf_counter() - start
            demeaned = np.array(result.demeaned, copy=True)
            eta = float(external_normal_residuals(codes, rhs, demeaned).max())
            delta = float(projection_errors(demeaned, reference_demeaned, rhs).max())
            rows.append(
                {
                    "dgp": dgp,
                    "n_obs": int(n_obs),
                    "preconditioner": name,
                    "tol": tol,
                    "iterations_max": int(max(result.iterations)),
                    "wall_s": elapsed,
                    "max_eta": eta,
                    "max_delta": delta,
                    "all_converged": bool(all(result.converged)),
                    "clears_gate_a": bool(eta <= GATE_A_ETA and delta <= GATE_A_DELTA),
                }
            )
    return rows


def same_projection_check(dgp: str, n_obs: int) -> dict:
    """All three configurations must reach one projection at tight tolerance."""
    from within import LsmrOptions, solve_batch

    codes, categories, rhs = _load_design(dgp, n_obs)
    options = LsmrOptions(tol=REFERENCE_TOL, maxiter=REFERENCE_MAXITER)
    solutions = {
        name: np.array(
            solve_batch(categories, rhs, options, preconditioner=config).demeaned,
            copy=True,
        )
        for name, config in _preconditioner_configs().items()
    }
    baseline = solutions["additive"]
    return {
        "dgp": dgp,
        "n_obs": int(n_obs),
        "max_delta_vs_additive": {
            name: float(projection_errors(values, baseline, rhs).max())
            for name, values in solutions.items()
        },
    }


# ---------------------------------------------------------------------------
# 4. Cap reporting
# ---------------------------------------------------------------------------
def cap_reporting_check(dgp: str, n_obs: int) -> dict:
    import warnings

    import pyfixest as pf
    from within import LsmrOptions, solve_batch

    codes, categories, rhs = _load_design(dgp, n_obs)
    configs = _preconditioner_configs()

    lsmr = solve_batch(
        categories, rhs, LsmrOptions(tol=1e-12, maxiter=5), preconditioner=configs["off"]
    )
    capped_map = map_demean_with_sweeps(np.asarray(rhs), codes, tol=1e-10, maxiter=5)

    frame = pd.read_parquet(ROOT / "benchmarks" / "data" / f"{dgp}_{n_obs // 1000}k.parquet")
    formula = f"y ~ x1 | {' + '.join(FE_COLS)}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pf.feols(
                formula,
                data=frame,
                vcov="iid",
                demeaner=demeaner_for("rust", tol=1e-10, maxiter=5),
            )
            feols_signal = "no error raised"
        except Exception as exc:  # noqa: BLE001 - recording the signal is the point
            feols_signal = f"{type(exc).__name__}: {exc}"

    return {
        "lsmr_converged": [bool(flag) for flag in lsmr.converged],
        "lsmr_iterations": [int(value) for value in lsmr.iterations],
        "map_censoring": list(capped_map.censoring),
        "map_iterations": list(capped_map.iterations),
        "feols_cap_signal": feols_signal,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=100_000)
    add_dgps_arg(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "runs" / "latest" / "pilot_calibration.json",
    )
    args = parser.parse_args()

    report: dict = {"n_obs": args.n_obs, "gate_a": {"eta": GATE_A_ETA, "delta": GATE_A_DELTA}}

    print("[pilot] validating eta against a dense minimum-norm solve")
    report["metric_validation"] = validate_external_residual()
    validation = report["metric_validation"]
    print(
        f"  eta(direct)={validation['eta_direct_max']:.2e} "
        f"helper-vs-dense={validation['helper_vs_dense_max_abs_diff']:.2e} "
        f"eta(perturbed)={validation['eta_perturbed_min']:.2e}"
    )

    rows: list[dict] = []
    for dgp in args.dgps:
        print(f"\n[pilot] tolerance sweep: {dgp} n={args.n_obs:,}")
        design_rows = tolerance_sweep(dgp, args.n_obs)
        rows.extend(design_rows)
        header = (
            f"  {'pc':10s} {'tol':>8s} {'iters':>6s} {'wall':>8s} "
            f"{'max eta':>10s} {'max delta':>10s}  gate A"
        )
        print(header)
        for row in design_rows:
            flag = "PASS" if row["clears_gate_a"] else "fail"
            print(
                f"  {row['preconditioner']:10s} {row['tol']:8.0e} "
                f"{row['iterations_max']:6d} {row['wall_s']:7.3f}s "
                f"{row['max_eta']:10.2e} {row['max_delta']:10.2e}  {flag}"
            )
    report["tolerance_sweep"] = rows

    report["same_projection"] = [
        same_projection_check(dgp, args.n_obs) for dgp in args.dgps
    ]
    print("\n[pilot] same-projection check at tol=1e-14")
    for entry in report["same_projection"]:
        deltas = ", ".join(
            f"{name}={value:.2e}" for name, value in entry["max_delta_vs_additive"].items()
        )
        print(f"  {entry['dgp']:10s} {deltas}")

    report["cap_reporting"] = {
        dgp: cap_reporting_check(dgp, args.n_obs) for dgp in args.dgps
    }
    print("\n[pilot] cap reporting")
    for dgp, entry in report["cap_reporting"].items():
        print(
            f"  {dgp:10s} lsmr_converged={entry['lsmr_converged']} "
            f"map_censoring={entry['map_censoring']} "
            f"feols={entry['feols_cap_signal'][:48]}"
        )

    # The smallest tolerance at which every configuration clears Gate A on
    # every design. This is the number PROTOCOL.md freezes.
    passing = {}
    for row in rows:
        passing.setdefault(row["tol"], []).append(row["clears_gate_a"])
    universal = sorted(tol for tol, flags in passing.items() if all(flags))
    report["gate_a_tolerance"] = max(universal) if universal else None
    print(
        "\n[pilot] loosest tolerance clearing Gate A everywhere: "
        f"{report['gate_a_tolerance']}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

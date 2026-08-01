"""Measure runtime against achieved precision on three AKM mobility designs.

The benchmark holds one stored sample fixed within each design and varies each
solver's native tolerance. Nominal tolerances are recorded but never compared
across packages. The two common accuracy measures are

    |beta - beta_ref| / SE(beta_ref)

and

    ||residual - residual_ref||_2 / ||residual_ref||_2.

The default run covers mobility designs 1, 3, and 5 at one million observations,
with one discarded warm-up and three timed repetitions per setting:

    BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run bench-tolerance
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyfixest.core.detect_singletons import detect_singletons


from benchmarks.core.methods import inline_label, resolve
from benchmarks.core.accuracy import external_normal_residuals
from benchmarks.dgp.samples import FE_COLS
from benchmarks.solvers.pyfixest_feols import _fit_converged
from benchmarks.solvers.settings import demeaner_for
from benchmarks.dgp.scenarios import get_akm_sweep_scenarios
from benchmarks.core.paths import EXTERNAL_DIR, ROOT

FORMULA = "y ~ x1 | indiv_id + firm_id + year"
MODEL_COLUMNS = [*FE_COLS, "x1", "y"]
DEFAULT_DESIGNS = (1, 3, 5)
DEFAULT_N_OBS = 1_000_000
DEFAULT_REPETITIONS = 3
DEFAULT_MAXITER = 10_000
REFERENCE_TOLERANCE = 1e-14
REFERENCE_MAXITER = 100_000


@dataclass(frozen=True)
class MethodSpec:
    """One solver configuration and its tolerance grid."""

    key: str
    package: str
    solver: str
    preconditioner: str
    default_tolerance: float
    tolerances: tuple[float, ...]

    @property
    def label(self) -> str:
        """Reader-facing name, from the shared method registry."""
        return inline_label(resolve(self.key))


METHODS = (
    MethodSpec(
        key="lsmr_off",
        package="PyFixest",
        solver="LSMR",
        preconditioner="off",
        default_tolerance=1e-8,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    ),
    MethodSpec(
        key="lsmr_diagonal",
        package="PyFixest",
        solver="LSMR",
        preconditioner="diagonal",
        default_tolerance=1e-8,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    ),
    MethodSpec(
        key="lsmr_additive",
        package="PyFixest",
        solver="LSMR",
        preconditioner="additive",
        default_tolerance=1e-8,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    ),
    MethodSpec(
        key="pyfixest_map",
        package="PyFixest",
        solver="MAP",
        preconditioner="none",
        default_tolerance=1e-6,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10),
    ),
    MethodSpec(
        key="r_fixest",
        package="fixest",
        solver="MAP",
        preconditioner="package default",
        default_tolerance=1e-6,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10),
    ),
    MethodSpec(
        key="julia_fem",
        package="FixedEffectModels.jl",
        solver="LSMR",
        preconditioner="package default",
        default_tolerance=1e-6,
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    ),
)

METHOD_BY_KEY = {method.key: method for method in METHODS}
PYTHON_METHODS = frozenset(
    {"lsmr_off", "lsmr_diagonal", "lsmr_additive", "pyfixest_map"}
)
EXTERNAL_METHODS = {
    "r_fixest": (
        ["Rscript"],
        EXTERNAL_DIR / "tolerance_fixest.R",
    ),
    "julia_fem": (
        ["julia"],
        EXTERNAL_DIR / "tolerance_julia.jl",
    ),
}

RESULT_COLUMNS = [
    "design",
    "n_obs_source",
    "n_obs",
    "n_singletons_dropped",
    "sample_hash",
    "source_path",
    "method",
    "label",
    "package",
    "solver",
    "preconditioner",
    "tolerance",
    "default_tolerance",
    "is_default_tolerance",
    "maxiter",
    "repetition",
    "time_s",
    "success",
    "converged",
    "capped",
    "coefficient_error_se",
    "residual_error",
    "beta_x1",
    "reference_beta_x1",
    "reference_se_x1",
    "reference_residual_norm",
    "reference_fe_eta",
    "reference_x_score",
    "reference_tolerance",
    "reference_maxiter",
    "thread_count",
    "iterations",
    "error",
]


@dataclass(frozen=True)
class ReferenceSolution:
    beta_x1: float
    se_x1: float
    residual: np.ndarray
    residual_norm: float
    fe_eta: float
    x_score: float


def coefficient_error_se(beta: float, beta_ref: float, se_ref: float) -> float:
    """Absolute slope error in units of the reference standard error."""
    if not np.isfinite(se_ref) or se_ref <= 0:
        raise ValueError("reference standard error must be positive and finite")
    return float(abs(beta - beta_ref) / se_ref)


def residual_error(residual: np.ndarray, residual_ref: np.ndarray) -> float:
    """Relative Euclidean error in the final regression residual."""
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    reference = np.asarray(residual_ref, dtype=np.float64).reshape(-1)
    if values.shape != reference.shape:
        raise ValueError(
            f"residual shape {values.shape} does not match reference {reference.shape}"
        )
    denominator = float(np.linalg.norm(reference))
    if denominator == 0:
        raise ValueError("reference residual has zero norm")
    return float(np.linalg.norm(values - reference) / denominator)


def _sample_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        frame.loc[:, MODEL_COLUMNS], index=False
    ).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _drop_recursive_singletons(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Prune once before benchmarking so every package receives the same rows."""
    kept = frame.reset_index(drop=True)
    n_initial = len(kept)
    while True:
        ids = np.asfortranarray(
            kept.loc[:, FE_COLS].to_numpy(dtype=np.uint32, copy=True)
        )
        singleton = detect_singletons(ids)
        if not singleton.any():
            break
        kept = kept.loc[~singleton, MODEL_COLUMNS].reset_index(drop=True)
    return kept, n_initial - len(kept)


def _extract_named_value(values: Any, name: str) -> float:
    if hasattr(values, "loc") and name in values.index:
        return float(values.loc[name])
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"could not identify {name!r} in result with {array.size} terms")
    return float(array[0])


def _fit_pyfixest(
    frame: pd.DataFrame,
    method: MethodSpec,
    tolerance: float,
    maxiter: int,
) -> tuple[Any, float]:
    import pyfixest as pf

    backend = (
        "rust"
        if method.solver == "MAP"
        else f"within-{method.preconditioner}"
    )
    demeaner = demeaner_for(backend, tol=tolerance, maxiter=maxiter)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fit = pf.feols(
            FORMULA,
            data=frame,
            vcov="iid",
            copy_data=False,
            store_data=False,
            demeaner=demeaner,
        )
        converged = _fit_converged(fit)
    elapsed = time.perf_counter() - start
    if not converged:
        raise RuntimeError("PyFixest model did not converge")
    return fit, elapsed


def _reference_solution(frame: pd.DataFrame) -> ReferenceSolution:
    method = METHOD_BY_KEY["lsmr_additive"]
    fit, _ = _fit_pyfixest(
        frame,
        method,
        tolerance=REFERENCE_TOLERANCE,
        maxiter=REFERENCE_MAXITER,
    )
    beta = _extract_named_value(fit.coef(), "x1")
    se = _extract_named_value(fit.se(), "x1")
    residual = np.asarray(fit.resid(), dtype=np.float64).reshape(-1).copy()
    if residual.shape[0] != len(frame):
        raise RuntimeError("reference fit did not retain the pre-pruned sample")

    categories = np.asfortranarray(
        frame.loc[:, FE_COLS].to_numpy(dtype=np.uint32, copy=True)
    )
    y = frame["y"].to_numpy(dtype=np.float64, copy=False)
    x = frame["x1"].to_numpy(dtype=np.float64, copy=False)
    fe_eta = float(
        external_normal_residuals(
            categories=categories,
            rhs=y.reshape(-1, 1),
            demeaned=residual.reshape(-1, 1),
        )[0]
    )
    eps = np.finfo(np.float64).eps
    x_denominator = max(abs(float(x @ y)), eps * np.linalg.norm(x) * np.linalg.norm(y))
    x_score = float(abs(float(x @ residual)) / x_denominator)
    if max(fe_eta, x_score) > 1e-10:
        raise RuntimeError(
            "tight reference failed the normal equations: "
            f"FE eta={fe_eta:.3e}, x score={x_score:.3e}"
        )
    return ReferenceSolution(
        beta_x1=beta,
        se_x1=se,
        residual=residual,
        residual_norm=float(np.linalg.norm(residual)),
        fe_eta=fe_eta,
        x_score=x_score,
    )


def _shared_fields(
    *,
    design: str,
    source_n_obs: int,
    frame: pd.DataFrame,
    n_singletons_dropped: int,
    sample_hash: str,
    source_path: Path,
    method: MethodSpec,
    maxiter: int,
    reference: ReferenceSolution,
    thread_count: int,
) -> dict[str, Any]:
    """What identifies a measurement, whichever arm produced it.

    The Python arm writes rows directly and the external arm sends a config to
    an R or Julia driver that writes rows back, but both end up in one
    tolerance_frontier.csv. These fields have to match across the two or the
    file has two schemas and the aggregation silently drops whichever columns
    the plotting code does not find. Defining them once is what keeps that
    from happening by accident.
    """
    return {
        "design": design,
        "n_obs_source": source_n_obs,
        "n_obs": len(frame),
        "n_singletons_dropped": n_singletons_dropped,
        "sample_hash": sample_hash,
        "source_path": str(source_path),
        "method": method.key,
        "label": method.label,
        "package": method.package,
        "solver": method.solver,
        "preconditioner": method.preconditioner,
        "default_tolerance": method.default_tolerance,
        "maxiter": maxiter,
        "reference_beta_x1": reference.beta_x1,
        "reference_se_x1": reference.se_x1,
        "reference_residual_norm": reference.residual_norm,
        "reference_fe_eta": reference.fe_eta,
        "reference_x_score": reference.x_score,
        "reference_tolerance": REFERENCE_TOLERANCE,
        "reference_maxiter": REFERENCE_MAXITER,
        "thread_count": thread_count,
    }


def _base_row(
    *,
    method: MethodSpec,
    tolerance: float,
    repetition: int,
    **shared: Any,
) -> dict[str, Any]:
    """One measured row from the in-process PyFixest arm."""
    return {
        **_shared_fields(method=method, **shared),
        "tolerance": tolerance,
        "is_default_tolerance": tolerance == method.default_tolerance,
        "repetition": repetition,
    }


def _capped_from_error(error: str) -> bool:
    lowered = error.lower()
    return "iteration" in lowered and (
        "failed after" in lowered
        or "maximum" in lowered
        or "maxiter" in lowered
        or "convergence" in lowered
    )


def _python_schedule(
    methods: list[MethodSpec],
    repetitions: int,
    tolerance_override: tuple[float, ...] | None,
    seed: int,
) -> list[tuple[MethodSpec, float, int]]:
    schedule = [
        (method, tolerance, repetition)
        for method in methods
        for tolerance in (tolerance_override or method.tolerances)
        for repetition in range(1, repetitions + 1)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def _run_python_methods(
    *,
    design: str,
    source_n_obs: int,
    frame: pd.DataFrame,
    n_singletons_dropped: int,
    sample_hash: str,
    source_path: Path,
    methods: list[MethodSpec],
    repetitions: int,
    tolerance_override: tuple[float, ...] | None,
    maxiter: int,
    seed: int,
    reference: ReferenceSolution,
    thread_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not methods:
        return rows

    for method in methods:
        warm_tolerance = (tolerance_override or method.tolerances)[0]
        print(
            f"[warmup] {design} {method.label} tol={warm_tolerance:g}",
            flush=True,
        )
        try:
            fit, _ = _fit_pyfixest(frame, method, warm_tolerance, maxiter)
            del fit
        except Exception as exc:  # noqa: BLE001 - warm-up failure is reported
            print(f"[warn] warm-up failed for {method.label}: {exc}", file=sys.stderr)
        gc.collect()

    schedule = _python_schedule(methods, repetitions, tolerance_override, seed)
    for method, tolerance, repetition in schedule:
        print(
            f"[fit] {design} {method.label} tol={tolerance:g} "
            f"rep={repetition}/{repetitions}",
            flush=True,
        )
        base = _base_row(
            design=design,
            source_n_obs=source_n_obs,
            frame=frame,
            n_singletons_dropped=n_singletons_dropped,
            sample_hash=sample_hash,
            source_path=source_path,
            method=method,
            tolerance=tolerance,
            maxiter=maxiter,
            repetition=repetition,
            reference=reference,
            thread_count=thread_count,
        )
        start = time.perf_counter()
        try:
            fit, elapsed = _fit_pyfixest(frame, method, tolerance, maxiter)
            beta = _extract_named_value(fit.coef(), "x1")
            residual = np.asarray(fit.resid(), dtype=np.float64).reshape(-1)
            if residual.shape[0] != len(frame):
                raise RuntimeError("fit did not retain the pre-pruned sample")
            row = {
                **base,
                "time_s": elapsed,
                "success": True,
                "converged": True,
                "capped": False,
                "coefficient_error_se": coefficient_error_se(
                    beta, reference.beta_x1, reference.se_x1
                ),
                "residual_error": residual_error(residual, reference.residual),
                "beta_x1": beta,
                "iterations": None,
                "error": None,
            }
            del fit
        except Exception as exc:  # noqa: BLE001 - failures belong in the result
            elapsed = time.perf_counter() - start
            error = str(exc)
            row = {
                **base,
                "time_s": elapsed,
                "success": False,
                "converged": False,
                "capped": _capped_from_error(error),
                "coefficient_error_se": None,
                "residual_error": None,
                "beta_x1": None,
                "iterations": None,
                "error": error,
            }
        rows.append(row)
        gc.collect()
    return rows


def _external_config(
    *,
    method: MethodSpec,
    prepared_path: Path,
    repetitions: int,
    tolerance_override: tuple[float, ...] | None,
    seed: int,
    **shared: Any,
) -> dict[str, Any]:
    """The job handed to an R or Julia driver, which sweeps the grid itself.

    The external arm receives the whole tolerance grid and its own repetition
    count, because one subprocess covers every setting for a method rather
    than one setting per call.
    """
    return {
        **_shared_fields(method=method, **shared),
        "data_path": str(prepared_path),
        "tolerances": list(tolerance_override or method.tolerances),
        "repetitions": repetitions,
        "seed": seed,
    }


def _run_external_method(
    method: MethodSpec,
    config: dict[str, Any],
    config_path: Path,
) -> list[dict[str, Any]]:
    command_prefix, script = EXTERNAL_METHODS[method.key]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    command = [*command_prefix, str(script), str(config_path)]
    env = os.environ.copy()
    if method.key == "julia_fem":
        env["JULIA_PROJECT"] = str(ROOT / "benchmarks" / "julia-env")
        env["JULIA_NUM_THREADS"] = str(config["thread_count"])

    print(f"[external] {config['design']} {method.label}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("method") == method.key:
            rows.append(row)

    expected = len(config["tolerances"]) * int(config["repetitions"])
    if completed.returncode != 0:
        raise RuntimeError(
            f"{method.label} subprocess exited with code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    if len(rows) != expected:
        raise RuntimeError(
            f"{method.label} returned {len(rows)} rows; expected {expected}"
        )
    return rows


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame.loc[:, RESULT_COLUMNS].to_csv(path, index=False)


def _locate_sample(
    *,
    design_number: int,
    n_obs: int,
    data_dir: Path,
) -> Path:
    design = f"akm_mobility_{design_number}"
    expected = data_dir / f"{design}_{n_obs}_k1_iter_1.parquet"
    if expected.exists():
        return expected.resolve()

    scenarios = {
        dgp.dgp_name: dgp for dgp in get_akm_sweep_scenarios(data_dir)
    }
    if design not in scenarios:
        raise ValueError(f"unknown AKM design: {design}")
    datasets = scenarios[design].generate(n=n_obs, n_iters=1, burn_in=1)
    matches = [
        dataset.data_path
        for dataset in datasets
        if dataset.iter_type == "iter" and dataset.iter_num == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(f"could not generate one stored sample for {design}")
    return matches[0].resolve()


def _parse_method_keys(values: list[str]) -> list[MethodSpec]:
    unknown = sorted(set(values) - set(METHOD_BY_KEY))
    if unknown:
        raise ValueError(
            f"unknown methods {unknown}; choose from {sorted(METHOD_BY_KEY)}"
        )
    return [METHOD_BY_KEY[key] for key in values]


def _validate_threads() -> int:
    value = os.environ.get("BENCH_THREADS")
    if value is None:
        raise RuntimeError(
            "BENCH_THREADS must be set. The paper protocol uses BENCH_THREADS=10."
        )
    try:
        threads = int(value)
    except ValueError as exc:
        raise RuntimeError("BENCH_THREADS must be a positive integer") from exc
    if threads < 1:
        raise RuntimeError("BENCH_THREADS must be a positive integer")
    julia_threads = os.environ.get("JULIA_NUM_THREADS")
    if julia_threads is not None and int(julia_threads) != threads:
        raise RuntimeError("BENCH_THREADS and JULIA_NUM_THREADS must agree")
    return threads


def _run_plot(input_path: Path, output_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "plot_tolerance.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--designs",
        nargs="+",
        type=int,
        choices=range(1, 7),
        default=list(DEFAULT_DESIGNS),
        help="AKM mobility design numbers",
    )
    parser.add_argument("--n-obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHOD_BY_KEY),
        default=list(METHOD_BY_KEY),
    )
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        help="Use one tolerance grid for every selected method (mainly for pilots)",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "benchmarks" / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "runs"
        / "latest"
        / "tolerance_frontier.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "figures" / "results" / "tolerance_frontier.svg",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write the benchmark CSV without drawing the figure",
    )
    args = parser.parse_args()

    if args.n_obs < 1:
        parser.error("--n-obs must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    if args.tolerances and any(tolerance <= 0 for tolerance in args.tolerances):
        parser.error("--tolerances must be positive")

    thread_count = _validate_threads()
    selected_methods = _parse_method_keys(args.methods)
    python_methods = [
        method for method in selected_methods if method.key in PYTHON_METHODS
    ]
    external_methods = [
        method for method in selected_methods if method.key in EXTERNAL_METHODS
    ]
    tolerance_override = (
        tuple(dict.fromkeys(args.tolerances)) if args.tolerances else None
    )

    all_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="within-tolerance-") as tmp:
        temp_dir = Path(tmp)
        for design_number in args.designs:
            design = f"akm_mobility_{design_number}"
            source_path = _locate_sample(
                design_number=design_number,
                n_obs=args.n_obs,
                data_dir=args.data_dir.resolve(),
            )
            source = pd.read_parquet(source_path, columns=MODEL_COLUMNS)
            source_n_obs = len(source)
            frame, n_dropped = _drop_recursive_singletons(source)
            del source
            sample_hash = _sample_hash(frame)
            print(
                f"[sample] {design}: {len(frame):,}/{source_n_obs:,} rows retained "
                f"({n_dropped:,} singleton rows removed)",
                flush=True,
            )

            print(f"[reference] {design} additive LSMR tol={REFERENCE_TOLERANCE:g}")
            reference = _reference_solution(frame)
            print(
                f"[reference] beta={reference.beta_x1:.10g}, "
                f"FE eta={reference.fe_eta:.3e}, "
                f"x score={reference.x_score:.3e}",
                flush=True,
            )

            prepared_path = temp_dir / f"{design}.parquet"
            prepared = frame.copy()
            prepared["reference_residual"] = reference.residual
            prepared.to_parquet(prepared_path, index=False)
            del prepared

            rows = _run_python_methods(
                design=design,
                source_n_obs=source_n_obs,
                frame=frame,
                n_singletons_dropped=n_dropped,
                sample_hash=sample_hash,
                source_path=source_path,
                methods=python_methods,
                repetitions=args.repetitions,
                tolerance_override=tolerance_override,
                maxiter=args.maxiter,
                seed=args.seed + design_number,
                reference=reference,
                thread_count=thread_count,
            )
            all_rows.extend(rows)
            _write_results(args.output, all_rows)

            for method_number, method in enumerate(external_methods):
                config = _external_config(
                    design=design,
                    source_n_obs=source_n_obs,
                    frame=frame,
                    n_singletons_dropped=n_dropped,
                    sample_hash=sample_hash,
                    source_path=source_path,
                    prepared_path=prepared_path,
                    method=method,
                    repetitions=args.repetitions,
                    tolerance_override=tolerance_override,
                    maxiter=args.maxiter,
                    seed=args.seed + 100 * method_number + design_number,
                    reference=reference,
                    thread_count=thread_count,
                )
                config_path = temp_dir / f"{design}_{method.key}.json"
                all_rows.extend(_run_external_method(method, config, config_path))
                _write_results(args.output, all_rows)

            del frame
            del reference
            gc.collect()

    _write_results(args.output, all_rows)
    print(f"[results] wrote {args.output} ({len(all_rows)} rows)", flush=True)
    if not args.no_plot:
        _run_plot(args.output, args.figure)


if __name__ == "__main__":
    main()

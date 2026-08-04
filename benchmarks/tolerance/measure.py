"""Shared tolerance measurement for base and AKM experiments."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.accuracy import external_normal_residuals
from benchmarks.data import FE_COLUMNS, drop_singletons
from benchmarks.ols.pyfixest import fit_ols
from benchmarks.runtime import failure_fields, run_native

BASE_GRID = (1e-4, 1e-6, 1e-8, 1e-10)
EXTENDED_GRID = (*BASE_GRID, 1e-12)
TOLERANCES = {
    "within-off": (1e-8, EXTENDED_GRID),
    "within-diagonal": (1e-8, EXTENDED_GRID),
    "within-additive": (1e-8, EXTENDED_GRID),
    "rust-map": (1e-6, BASE_GRID),
    "fixest": (1e-6, BASE_GRID),
    "FEM.jl": (1e-6, EXTENDED_GRID),
}
REFERENCE_TOLERANCE = 1e-14
REFERENCE_MAXITER = 100_000


def coefficient_error_se(beta: float, reference_beta: float, reference_se: float) -> float:
    return float(abs(beta - reference_beta) / reference_se)


def residual_error(residual: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(residual, dtype=float).reshape(-1)
    target = np.asarray(reference, dtype=float).reshape(-1)
    if values.shape != target.shape:
        raise ValueError("fit and reference residuals use different samples")
    return float(np.linalg.norm(values - target) / np.linalg.norm(target))


def reference_solution(frame: pd.DataFrame) -> dict:
    """Build and validate the tight additive-LSMR reference fit."""
    # This diagnostic reads residuals after fitting, which lean fits discard.
    fit = fit_ols(
        frame,
        "within-additive",
        FE_COLUMNS,
        REFERENCE_TOLERANCE,
        REFERENCE_MAXITER,
        lean=False,
    )
    beta = float(fit.coef().loc["x1"])
    se = float(fit.se().loc["x1"])
    residual = np.asarray(fit.resid(), dtype=float).reshape(-1).copy()
    if len(residual) != len(frame):
        raise RuntimeError("reference fit did not retain the pre-pruned sample")
    categories = np.column_stack(
        [pd.factorize(frame[name], sort=True)[0] for name in FE_COLUMNS]
    )
    eta = float(
        external_normal_residuals(categories, frame[["y"]].to_numpy(), residual[:, None])[0]
    )
    x, y = frame["x1"].to_numpy(), frame["y"].to_numpy()
    denominator = max(
        abs(float(x @ y)),
        np.finfo(float).eps * np.linalg.norm(x) * np.linalg.norm(y),
    )
    x_score = float(abs(float(x @ residual)) / denominator)
    if max(eta, x_score) > 1e-10:
        raise RuntimeError(
            f"reference failed the normal equations: FE eta={eta:.3e}, x score={x_score:.3e}"
        )
    return {"beta": beta, "se": se, "residual": residual, "eta": eta, "x_score": x_score}


def _python_rows(
    frame: pd.DataFrame,
    design: str,
    backend: str,
    repetitions: int,
    maxiter: int,
    reference: dict,
) -> list[dict]:
    default, tolerances = TOLERANCES[backend]
    categories = np.column_stack(
        [pd.factorize(frame[name], sort=True)[0] for name in FE_COLUMNS]
    )
    try:
        fit_ols(frame, backend, FE_COLUMNS, tolerances[0], maxiter, lean=False)
    except Exception:
        # The discarded warm-up must not prevent the measured failures from
        # becoming result rows.
        pass
    rows = []
    for tolerance in tolerances:
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                fit = fit_ols(
                    frame, backend, FE_COLUMNS, tolerance, maxiter, lean=False
                )
                runtime = time.perf_counter() - started
                beta = float(fit.coef().loc["x1"])
                residual = np.asarray(fit.resid(), dtype=float).reshape(-1)
                max_eta = float(
                    external_normal_residuals(
                        categories, frame[["y"]].to_numpy(), residual[:, None]
                    )[0]
                )
                rows.append(
                    {
                        "design": design, "backend": backend, "tolerance": tolerance,
                        "default_tolerance": default, "repetition": repetition,
                        "runtime_s": runtime, "beta_x1": beta,
                        "coefficient_error_se": coefficient_error_se(
                            beta, reference["beta"], reference["se"]
                        ),
                        "residual_error": residual_error(residual, reference["residual"]),
                        "max_eta": max_eta,
                        "converged": True, "capped": False, "error": "",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "design": design, "backend": backend, "tolerance": tolerance,
                        "default_tolerance": default, "repetition": repetition,
                        "runtime_s": time.perf_counter() - started, "beta_x1": None,
                        "coefficient_error_se": None, "residual_error": None,
                        "max_eta": None,
                        **failure_fields(error),
                    }
                )
    return rows


def _native_rows(
    data_path: Path,
    output: Path,
    design: str,
    backend: str,
    repetitions: int,
    maxiter: int,
    reference: dict,
) -> list[dict]:
    default, tolerances = TOLERANCES[backend]
    script = "fixest.R" if backend == "fixest" else "fixed_effect_models.jl"
    arguments = [
        str(data_path), str(output), design,
        ",".join(str(value) for value in tolerances), str(default),
        str(repetitions), str(maxiter), str(reference["beta"]), str(reference["se"]),
    ]
    try:
        return run_native(Path(__file__).with_name(script), arguments, output)
    except Exception as error:
        rows = []
        for tolerance in tolerances:
            for repetition in range(repetitions):
                rows.append(
                    {
                        "design": design,
                        "backend": backend,
                        "tolerance": tolerance,
                        "default_tolerance": default,
                        "repetition": repetition,
                        "runtime_s": None,
                        "beta_x1": None,
                        "coefficient_error_se": None,
                        "residual_error": None,
                        "max_eta": None,
                        **failure_fields(error),
                    }
                )
        return rows


def measure(
    frame: pd.DataFrame,
    design: str,
    backends: tuple[str, ...] = tuple(TOLERANCES),
    *,
    repetitions: int,
    maxiter: int = 10_000,
) -> pd.DataFrame:
    """Measure supplied methods on one shared, pre-pruned sample."""
    threads = int(os.environ["BENCH_THREADS"])
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    prepared, dropped = drop_singletons(
        frame.loc[:, [*FE_COLUMNS, "x1", "y"]], FE_COLUMNS
    )
    reference = reference_solution(prepared)
    rows = []
    with tempfile.TemporaryDirectory(prefix="within-tolerance-") as directory:
        work = Path(directory)
        data_path = work / "sample.parquet"
        if any(backend in {"fixest", "FEM.jl"} for backend in backends):
            native_sample = prepared.copy()
            native_sample["reference_residual"] = reference["residual"]
            native_sample.to_parquet(data_path, index=False)
        for backend in backends:
            if backend in {"fixest", "FEM.jl"}:
                measured = _native_rows(
                    data_path, work / f"{backend}.csv", design, backend,
                    repetitions, maxiter, reference,
                )
            else:
                measured = _python_rows(prepared, design, backend, repetitions, maxiter, reference)
            for row in measured:
                row.update(
                    n_obs=len(prepared), n_singletons_dropped=dropped, threads=threads,
                )
            rows.extend(measured)
            successful = [
                float(row["runtime_s"])
                for row in measured
                if str(row["converged"]).lower() in {"true", "1"}
            ]
            if successful:
                value = f"{np.median(successful):.3f} s"
            elif measured and all(
                str(row.get("capped", "")).lower() in {"true", "1"}
                for row in measured
            ):
                value = "capped"
            else:
                value = "failed"
            print(f"tolerance / OLS / {design} / {backend}: {value}", flush=True)
    return pd.DataFrame(rows)

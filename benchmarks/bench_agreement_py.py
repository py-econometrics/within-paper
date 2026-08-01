"""Numerical agreement across fixed-effect solver backends.

The script has two layers:

1. A targeted PyFixest check comparing MAP against the within LSMR backend.
   (Pre-0.60 this backend was spelled "rust-cg"; it has not been conjugate
   gradient for some time, so the names here say within.) Since
   both use the same Python regression API and Rust data path, this isolates the
   demeaning strategy. It reports coefficient, residual, and firm-effect
   variance differences.
2. A cross-software coefficient check against R fixest and Julia
   FixedEffectModels.jl. Coefficients are the natural cross-package comparison
   because residual vectors and fixed-effect coefficients can reflect package
   conventions such as normalization. R fixest is used as the coefficient-table
   reference when it is available.

Runs on 100k simple and difficult DGPs.
"""

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path

from benchmarks.core.results import write_rows
from benchmarks.modular.experiment import FE_COLS
from benchmarks.modular.feols_benchmarkers import _fit_converged
from benchmarks.modular.settings import demeaner_for
import numpy as np
import pandas as pd
import pyfixest as pf
from benchmarks.core.paths import EXTERNAL_DIR, JULIA_ENV, ROOT

FML = "y ~ x1 | indiv_id + firm_id + year"
DEPVAR = "y"
COVARIATES = ["x1"]
OUTPUT_PATH = ROOT / os.environ.get("RESULTS_OUT", "results/runs/latest/agreement.csv")
OUTPUT_ROWS: list[dict[str, object]] = []


def _run_json_subprocess(command: list[str], config: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        proc = subprocess.run(
            [*command, config_path],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)

    return json.loads(proc.stdout)


def _coef_dict(names, coefficients) -> dict[str, float]:
    # jsonlite's auto_unbox serializes one-element vectors as scalars. The
    # one-covariate benchmark therefore returns "x1" and 1.23 from R, whereas
    # Julia returns one-element arrays. Normalize both payload shapes.
    name_values = [names] if isinstance(names, str) else list(names)
    coefficient_values = (
        list(coefficients)
        if isinstance(coefficients, Iterable) and not isinstance(coefficients, (str, bytes))
        else [coefficients]
    )
    if len(name_values) != len(coefficient_values):
        raise ValueError("coefficient names and values have different lengths")
    return {
        str(name): float(value)
        for name, value in zip(name_values, coefficient_values, strict=True)
    }


def _external_coefficients(
    data_path: Path, formula: str
) -> dict[str, dict[str, float] | str]:
    """Return coefficient vectors from external packages, or skip messages."""
    config = {
        "data_path": str(data_path),
        "formula": formula,
        "depvar": DEPVAR,
        "covariates": COVARIATES,
        "fe_cols": FE_COLS,
    }
    backends = {
        "fixest": (["Rscript", str(EXTERNAL_DIR / "agreement_fixest.R")], "Rscript"),
        "FixedEffectModels": (
            ["julia", f"--project={JULIA_ENV}", str(EXTERNAL_DIR / "agreement_julia.jl")],
            "julia",
        ),
    }
    results: dict[str, dict[str, float] | str] = {}

    for name, (command, executable) in backends.items():
        if shutil.which(executable) is None:
            results[name] = f"SKIP ({executable} not found)"
            continue
        try:
            payload = _run_json_subprocess(command, config)
            results[name] = _coef_dict(payload["names"], payload["coefficients"])
        except Exception as exc:
            results[name] = f"SKIP ({exc})"

    return results


def _named_coef_diffs(
    reference: dict[str, float], candidate: dict[str, float]
) -> list[float]:
    """Absolute differences, matched by coefficient name.

    Comparing by name rather than by position is what makes this a
    cross-package check: two packages may order the coefficient table
    differently, and a positional comparison would silently difference
    unrelated terms instead of failing.
    """
    mismatched = sorted(set(reference) ^ set(candidate))
    if mismatched:
        raise ValueError(f"coefficient names differ: {mismatched}")
    return [abs(reference[name] - candidate[name]) for name in reference]


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.8f}"


def _format_diff(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2e}"


def main() -> None:
    print("PyFixest internal agreement")
    print(f"{'dgp':<12} {'metric':<28} {'MAP vs within':>16}")
    print("-" * 60)

    for dgp_type in ["simple", "difficult"]:
        data_path = ROOT / "benchmarks" / "data" / f"{dgp_type}_100k.parquet"
        df = pd.read_parquet(data_path)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fit_map = pf.feols(FML, data=df, vcov="iid", demeaner=demeaner_for("rust"),
                               copy_data=False, store_data=True)
            fit_within = pf.feols(FML, data=df, vcov="iid",
                                  demeaner=demeaner_for("rust-cg"),
                                  copy_data=False, store_data=True)
        if not _fit_converged(fit_map) or not _fit_converged(fit_within):
            raise RuntimeError(f"PyFixest agreement model did not converge for {dgp_type}")

        # Coefficient differences
        coef_map = np.asarray(fit_map.coef())
        coef_within = np.asarray(fit_within.coef())
        coef_map_named = _coef_dict(fit_map.coef().index, coef_map)
        max_coef_diff = np.max(np.abs(coef_map - coef_within))
        print(f"{dgp_type:<12} {'max |coef diff|':<28} {max_coef_diff:>16.2e}")

        # Residual differences
        resid_map = np.asarray(fit_map.resid())
        resid_within = np.asarray(fit_within.resid())
        max_resid_diff = np.max(np.abs(resid_map - resid_within))
        resid_norm_map = np.linalg.norm(resid_map)
        rel_resid_diff = np.linalg.norm(resid_map - resid_within) / resid_norm_map
        print(f"{'':<12} {'max |resid diff|':<28} {max_resid_diff:>16.2e}")
        print(f"{'':<12} {'rel resid norm diff':<28} {rel_resid_diff:>16.2e}")

        # Firm-effect variance component
        fe_map = fit_map.fixef()
        fe_within = fit_within.fixef()
        var_firm_map = np.var(list(fe_map["C(firm_id)"].values()))
        var_firm_within = np.var(list(fe_within["C(firm_id)"].values()))
        var_diff = abs(var_firm_map - var_firm_within)
        print(f"{'':<12} {'var(firm FE) MAP':<28} {var_firm_map:>16.6f}")
        print(f"{'':<12} {'var(firm FE) within':<28} {var_firm_within:>16.6f}")
        print(f"{'':<12} {'|diff var(firm FE)|':<28} {var_diff:>16.2e}")
        print()

        external = _external_coefficients(data_path, FML)
        coefficient_sets: dict[str, dict[str, float]] = {
            "rust-map": coef_map_named,
            "within": _coef_dict(fit_within.coef().index, coef_within),
        }
        skip_messages: dict[str, str] = {}
        for backend, value in external.items():
            if isinstance(value, str):
                skip_messages[backend] = value
                continue
            coefficient_sets[backend] = value

        reference_name = "rust-map"
        reference = coefficient_sets[reference_name]

        print(f"{dgp_type:<12} coefficient summary relative to {reference_name}")
        print(f"{'':<12} {'backend':<20} {'x1':>14} {'avg |diff|':>14} {'max |diff|':>14}")
        for backend, coefficients in coefficient_sets.items():
            try:
                diffs = _named_coef_diffs(reference, coefficients)
                avg_diff = sum(diffs) / len(diffs)
                max_diff = max(diffs)
            except ValueError as exc:
                print(f"{'':<12} {backend:<20} {coefficients.get('x1', np.nan):>14.8f} {f'SKIP ({exc})':>28}")
                continue
            OUTPUT_ROWS.append(
                {
                    "dgp": dgp_type,
                    "model_k": 1,
                    "backend": backend,
                    "x1": coefficients.get("x1", np.nan),
                    "avg_abs_diff": avg_diff,
                    "max_abs_diff": max_diff,
                    "success": True,
                    "error": "",
                }
            )
            print(
                f"{'':<12} {backend:<20} "
                f"{_format_float(coefficients.get('x1')):>14} "
                f"{_format_diff(avg_diff):>14} "
                f"{_format_diff(max_diff):>14}"
            )
        for backend, message in skip_messages.items():
            print(f"{'':<12} {backend:<20} {'NA':>14} {'NA':>14} {message:>14}")
        print()

        print(f"{dgp_type:<12} full coefficient table")
        print(
            f"{'':<12} {'term':<8} {'rust-map':>14} {'within':>14} "
            f"{'fixest':>14} {'FixedEffectModels':>18}"
        )
        for term in COVARIATES:
            print(
                f"{'':<12} {term:<8} "
                f"{_format_float(coefficient_sets.get('rust-map', {}).get(term)):>14} "
                f"{_format_float(coefficient_sets.get('within', {}).get(term)):>14} "
                f"{_format_float(coefficient_sets.get('fixest', {}).get(term)):>14} "
                f"{_format_float(coefficient_sets.get('FixedEffectModels', {}).get(term)):>18}"
            )
        print()

    # Column order is pinned: this file is read positionally by the agreement table.
    write_rows(
        OUTPUT_PATH,
        OUTPUT_ROWS,
        fieldnames=[
            "dgp",
            "model_k",
            "backend",
            "x1",
            "avg_abs_diff",
            "max_abs_diff",
            "success",
            "error",
        ],
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

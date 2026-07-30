"""Time-versus-accuracy frontier over each package's own tolerance grid.

Plan item 11 / PROTOCOL.md §6: do not force every package to a single matched
η threshold. Sweep roughly four of each package's own settings and record wall
time against achieved external residual η.

Default designs are the 100K simple/difficult pilots; pass ``--n-obs`` for the
10M production frontier after the pilot freezes Gate A.

Run with:

    pixi run bench-accuracy-frontier
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from benchmarks.modular.cli import add_dgps_arg
from benchmarks.modular.experiment import FE_COLS
from benchmarks.modular.results import write_rows
from benchmarks.modular.accuracy import accuracy_record
from benchmarks.modular.dgp_functions import paper_base_dgp
from benchmarks.modular.settings import (
    DEFAULT_WITHIN_PRECONDITIONER,
    LSMR_SETTINGS,
    MAP_SETTINGS,
)

FML = "y ~ x1 | indiv_id + firm_id + year"


@dataclass(frozen=True)
class ToleranceSetting:
    package: str
    label: str
    backend: str
    # Free-form settings recorded in the CSV; the runner interprets known keys.
    settings: dict


def _sample_hash(frame: pd.DataFrame) -> str:
    cols = ["y", "x1", *FE_COLS]
    values = pd.util.hash_pandas_object(frame.loc[:, cols], index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _categories_and_rhs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    categories = np.asfortranarray(frame[FE_COLS].to_numpy(dtype=np.uint32) - 1)
    rhs = np.asfortranarray(frame[["y", "x1"]].to_numpy(dtype=np.float64))
    return categories, rhs


def default_settings() -> list[ToleranceSetting]:
    """Roughly four settings per package, on that package's own scale."""
    settings: list[ToleranceSetting] = []

    for tol in (1e-4, 1e-5, 1e-6, 1e-8):
        settings.append(
            ToleranceSetting(
                package="pyfixest-rust-map",
                label=f"map_tol={tol:g}",
                backend="rust",
                settings={**MAP_SETTINGS, "fixef_tol": tol},
            )
        )

    for tol in (1e-4, 1e-6, 1e-8, 1e-10):
        settings.append(
            ToleranceSetting(
                package="pyfixest-within-additive",
                label=f"lsmr_tol={tol:g}",
                backend=f"within-{DEFAULT_WITHIN_PRECONDITIONER}",
                settings={
                    **LSMR_SETTINGS,
                    "fixef_atol": tol,
                    "fixef_btol": tol,
                    "preconditioner": DEFAULT_WITHIN_PRECONDITIONER,
                },
            )
        )

    # R fixest monitors its own criterion; the grid is its tol argument.
    for tol in (1e-4, 1e-5, 1e-6, 1e-8):
        settings.append(
            ToleranceSetting(
                package="r-fixest",
                label=f"fixest_tol={tol:g}",
                backend="r-fixest",
                settings={"tol": tol},
            )
        )

    # FixedEffectModels.jl uses double_precision tolerances via the Julia script
    # only when the optional path is enabled; keep the grid recorded either way.
    for tol in (1e-4, 1e-6, 1e-8, 1e-10):
        settings.append(
            ToleranceSetting(
                package="julia-fem",
                label=f"fem_tol={tol:g}",
                backend="julia-fem",
                settings={"tol": tol},
            )
        )
    return settings


def _fit_pyfixest(frame: pd.DataFrame, setting: ToleranceSetting):
    import pyfixest as pf

    if setting.backend == "rust":
        demeaner = pf.MapDemeaner(
            backend="rust",
            fixef_tol=float(setting.settings["fixef_tol"]),
            fixef_maxiter=int(setting.settings["fixef_maxiter"]),
        )
    elif setting.backend.startswith("within"):
        demeaner = pf.LsmrDemeaner(
            backend="within",
            preconditioner=str(setting.settings["preconditioner"]),
            fixef_atol=float(setting.settings["fixef_atol"]),
            fixef_btol=float(setting.settings["fixef_btol"]),
            fixef_maxiter=int(setting.settings["fixef_maxiter"]),
        )
    else:
        raise ValueError(f"unsupported pyfixest backend {setting.backend!r}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        t0 = time.perf_counter()
        fit = pf.feols(
            FML,
            data=frame,
            vcov="iid",
            copy_data=False,
            store_data=False,
            demeaner=demeaner,
        )
        elapsed = time.perf_counter() - t0
    return fit, elapsed


def _external_eta_from_fit(frame: pd.DataFrame, fit) -> float:
    """Recompute η from the fit's own demeaned columns when available."""
    categories, rhs = _categories_and_rhs(frame)

    demeaned = None
    lookup = getattr(fit, "_lookup_demeaned_data", None) or {}
    if lookup:
        for value in lookup.values():
            block = value
            if hasattr(value, "to_numpy"):
                block = value.to_numpy(dtype=np.float64)
            arr = np.asarray(block, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == len(frame) and arr.shape[1] >= 2:
                demeaned = np.ascontiguousarray(arr[:, :2])
                break

    if demeaned is None:
        y = np.asarray(getattr(fit, "_Y"), dtype=np.float64).reshape(-1, 1)
        x = np.asarray(getattr(fit, "_X"), dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.shape[0] == len(frame) and x.shape[0] == len(frame):
            demeaned = np.column_stack([y, x[:, :1]])

    if demeaned is None:
        raise RuntimeError("fit did not expose demeaned columns for eta")

    record = accuracy_record(categories=categories, rhs=rhs, demeaned=demeaned)
    return record.max_eta


def _run_pyfixest_setting(
    frame: pd.DataFrame, setting: ToleranceSetting, sample_hash: str
) -> dict:
    try:
        fit, elapsed = _fit_pyfixest(frame, setting)
        converged = bool(
            getattr(fit, "convergence", getattr(fit, "_convergence", True))
        )
        eta = _external_eta_from_fit(frame, fit) if converged else None
        beta = float(np.asarray(fit.coef())[0]) if converged else None
        row = {
            "package": setting.package,
            "setting": setting.label,
            "backend": setting.backend,
            "sample_hash": sample_hash,
            "time_s": elapsed,
            "success": converged,
            "max_eta": eta,
            "beta_x1": beta,
            "error": None if converged else "did not converge",
            **{f"opt_{k}": v for k, v in setting.settings.items()},
        }
        del fit
        return row
    except Exception as exc:
        return {
            "package": setting.package,
            "setting": setting.label,
            "backend": setting.backend,
            "sample_hash": sample_hash,
            "time_s": None,
            "success": False,
            "max_eta": None,
            "beta_x1": None,
            "error": str(exc),
            **{f"opt_{k}": v for k, v in setting.settings.items()},
        }


def _run_design(
    *,
    dgp: str,
    n_obs: int,
    seed: int,
    settings: list[ToleranceSetting],
    include_external: bool,
) -> list[dict]:
    frame = paper_base_dgp(n=n_obs, type_=dgp, seed=seed)
    sample_hash = _sample_hash(frame)
    rows: list[dict] = []

    # Warm one cheap fit so lazy imports are not charged to the first cell.
    warm = [s for s in settings if s.backend in {"rust", "within-additive"}]
    if warm:
        try:
            _fit_pyfixest(frame, warm[0])
        except Exception:
            pass
        gc.collect()

    for setting in settings:
        if setting.backend in {"r-fixest", "julia-fem"}:
            if not include_external:
                continue
            # External packages need subprocess harnesses with tolerance plumbing;
            # record the grid entry as skipped until those scripts accept tol.
            rows.append(
                {
                    "dgp": dgp,
                    "n_obs": n_obs,
                    "package": setting.package,
                    "setting": setting.label,
                    "backend": setting.backend,
                    "sample_hash": sample_hash,
                    "time_s": None,
                    "success": False,
                    "max_eta": None,
                    "beta_x1": None,
                    "error": "external tolerance sweep not yet wired",
                    **{f"opt_{k}": v for k, v in setting.settings.items()},
                }
            )
            continue

        print(
            f"[frontier] {dgp} n={n_obs:,} {setting.package} {setting.label}",
            flush=True,
        )
        row = _run_pyfixest_setting(frame, setting, sample_hash)
        row["dgp"] = dgp
        row["n_obs"] = n_obs
        rows.append(row)
        gc.collect()
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # External skip rows omit some opt_* fields; the shared writer unions keys.
    write_rows(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=100_000)
    add_dgps_arg(parser)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include r-fixest and julia-fem grid placeholders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results"
        / "runs"
        / "latest"
        / "accuracy_frontier.csv",
    )
    args = parser.parse_args()

    settings = default_settings()
    rows: list[dict] = []
    for dgp in args.dgps:
        rows.extend(
            _run_design(
                dgp=dgp,
                n_obs=args.n_obs,
                seed=args.seed + (0 if dgp == "simple" else 1),
                settings=settings,
                include_external=args.include_external,
            )
        )

    _write_csv(args.out, rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

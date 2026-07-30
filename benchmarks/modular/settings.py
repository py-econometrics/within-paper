"""Pinned solver settings and the backend-name mapping.

These constants are what most drivers actually want from the benchmark layer:
the frozen mechanism tolerances, the package-default stopping rules, and the
preconditioner names. They lived in feols_benchmarkers alongside the PyFixest
and subprocess backends, so a driver that needed two float constants imported
the Torch runtime detection with them.

Nothing here imports a solver. `demeaner_from_backend` imports pyfixest lazily,
inside the call, so this module stays cheap to import.
"""

from __future__ import annotations

MAP_SETTINGS = {
    "backend": "rust",
    "fixef_tol": 1e-6,
    "fixef_maxiter": 10_000,
}

# PyFixest passes max(fixef_atol, fixef_btol) to the within backend, so the two
# must be set explicitly and equally or the effective tolerance is whichever
# default happens to be larger.
LSMR_SETTINGS = {
    "backend": "within",
    "fixef_atol": 1e-8,
    "fixef_btol": 1e-8,
    "fixef_maxiter": 1_000,
}

WITHIN_PRECONDITIONERS = ("off", "diagonal", "additive")

# pf.LsmrDemeaner(preconditioner="auto") resolves to "additive" for the within
# backend. Naming it here keeps the `within` benchmark label pinned to a
# specific preconditioner even if that resolution changes upstream.
DEFAULT_WITHIN_PRECONDITIONER = "additive"

# Frozen by the 100K calibration pilot on 2026-07-26 (PROTOCOL.md section 5).
# The LSMR stopping rule bounds a relative normal-equation residual recovered
# from the bidiagonalization scalars, which is a different number from the
# externally recomputed eta. At the package default of 1e-8 the achieved eta
# runs from 2.4e-8 to 1.6e-6, so no configuration clears Gate A. 1e-12 is the
# loosest tolerance at which all three clear it on both pilot designs, and it
# is what the mechanism ablation uses so its iteration counts are compared at
# matched accuracy rather than at matched nominal tolerance.
MECHANISM_LSMR_TOL = 1e-12
MECHANISM_MAP_TOL = 1e-10

# The ablation also equalizes the iteration budget. The package defaults give
# MAP 10,000 iterations and LSMR 1,000, and at 1M the first ablation run showed
# what that asymmetry does: within-off failed 30 of 33 trials, every one of them
# at 1,000 iterations, while rust-map was allowed ten times as many. "Removing
# the preconditioner does not by itself remove the slow directions" cannot rest
# on a censoring that the budget produced, so the mechanism arm gets MAP's cap
# and a run that still fails to converge fails on its own merits.
MECHANISM_MAXITER = 10_000


def _demeaner_from_backend(
    backend: str, *, tol: float | None = None, maxiter: int | None = None
):
    """Map a benchmark backend name to a typed demeaner configuration.

    Recognised names:

    - ``rust``: unaccelerated Rust MAP.
    - ``within-off`` / ``within-diagonal`` / ``within-additive``: LSMR with the
      named preconditioner, for the same-code mechanism ablation.
    - ``within``: alias for the documented default, so that existing result
      files and table labels keep working.
    - ``torch_cpu`` / ``torch_mps`` / ``torch_cuda``: the Torch LSMR backends,
      which carry their own built-in diagonal preconditioner.

    ``tol`` and ``maxiter`` override the pinned stopping rule. Cross-package
    tables leave both unset so that each package runs at its documented default;
    the mechanism ablation sets them so every configuration is compared at
    matched accuracy under one iteration budget.
    """
    import pyfixest as pf

    if backend == "rust":
        settings = dict(MAP_SETTINGS)
        if tol is not None:
            settings["fixef_tol"] = tol
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.MapDemeaner(**settings)

    if backend == "within":
        backend = f"within-{DEFAULT_WITHIN_PRECONDITIONER}"

    if backend.startswith("within-"):
        preconditioner = backend[len("within-") :]
        if preconditioner not in WITHIN_PRECONDITIONERS:
            raise ValueError(
                f"Unknown within preconditioner {preconditioner!r}; "
                f"expected one of {WITHIN_PRECONDITIONERS}"
            )
        settings = dict(LSMR_SETTINGS)
        if tol is not None:
            # Both must move together: PyFixest passes their maximum through.
            settings["fixef_atol"] = tol
            settings["fixef_btol"] = tol
        if maxiter is not None:
            settings["fixef_maxiter"] = maxiter
        return pf.LsmrDemeaner(preconditioner=preconditioner, **settings)

    if backend == "torch_cpu":
        return pf.LsmrDemeaner(backend="torch", device="cpu")
    if backend == "torch_mps":
        return pf.LsmrDemeaner(backend="torch", device="mps", precision="float32")
    if backend == "torch_cuda":
        return pf.LsmrDemeaner(backend="torch", device="cuda")
    raise ValueError(f"Unknown demeaner backend: {backend!r}")

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from benchmarks.solvers.pyfixest_feols import (
    PyFeolsBenchmarkerFullApi,
    detect_torch_runtime_availability,
)
from benchmarks.solvers.subprocess_driver import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
)
from benchmarks.solvers.settings import (
    MECHANISM_LSMR_TOL,
    MECHANISM_MAP_TOL,
    MECHANISM_MAXITER,
    WITHIN_PRECONDITIONERS,
)
from benchmarks.solvers.pyfixest_fepois import (
    FixestFepoisBenchmarker,
    GLFixedEffectModelsBenchmarker,
    PyFepoisBenchmarkerFullApi,
)


@dataclass(frozen=True)
class BenchmarkerBundle:
    benchmarkers: list


class Backend(NamedTuple):
    """One measured configuration: a label, a demeaner, and its stopping rule.

    ``tol`` and ``maxiter`` of ``None`` keep the package default.
    """

    label: str
    backend: str
    tol: float | None = None
    maxiter: int | None = None


# What a user gets out of the box. Cross-package tables read these.
#
# The label is the name recorded in the result file, and it is always the
# canonical key from core.methods. Every reader-facing string is derived from
# that key, so a driver never invents a second spelling of a method it already
# has a name for.
PACKAGE_DEFAULTS = (
    Backend("within", "within"),
    Backend("rust-map", "rust"),
)

# The same code path at matched accuracy under one shared iteration budget.
# The mechanism figures read these. They carry distinct labels so both views
# can be measured in one pass over the data and separated afterwards: which
# rows belong in which table is a curation decision, not a reason to run the
# same designs twice.
MATCHED_ACCURACY = (
    Backend("rust-map-matched", "rust", MECHANISM_MAP_TOL, MECHANISM_MAXITER),
    *(
        Backend(
            f"within-{name}",
            f"within-{name}",
            MECHANISM_LSMR_TOL,
            MECHANISM_MAXITER,
        )
        for name in WITHIN_PRECONDITIONERS
    ),
)

EXTERNAL_FEOLS = (
    ("fixest", FixestFeolsBenchmarker),
    ("FEM.jl", JuliaFeolsBenchmarker),
)


def require_multiple_absorbed_factors(spec) -> None:
    """Reject preconditioner comparisons with a single absorbed factor.

    With one factor the Gramian is diagonal and PyFixest falls back to
    closed-form MAP demeaning, so all three preconditioner settings take the
    same code path and the comparison measures nothing.
    """
    if spec.n_fe < 2:
        raise ValueError(
            "The preconditioner comparison needs at least two absorbed factors; "
            f"got {spec.n_fe}. Single-factor problems fall back to closed-form "
            "MAP demeaning, so off, diagonal, and additive are identical."
        )


def _torch_benchmarkers() -> list:
    availability = detect_torch_runtime_availability()
    if not availability.has_torch:
        print("[bench] skipping Torch backends: Torch is not installed", flush=True)
        return []

    benchmarkers = [PyFeolsBenchmarkerFullApi("pyfixest (torch-cpu)", "torch_cpu")]
    for device, available in (
        ("mps", availability.has_mps),
        ("cuda", availability.has_cuda),
    ):
        if available:
            benchmarkers.append(
                PyFeolsBenchmarkerFullApi(
                    f"pyfixest (torch-{device})", f"torch_{device}"
                )
            )
        else:
            print(f"[bench] skipping torch-{device}: unavailable", flush=True)
    return benchmarkers


def _pyfixest_specs(package_defaults: bool, matched_accuracy: bool) -> list[Backend]:
    specs: list[Backend] = []
    if package_defaults:
        specs.extend(PACKAGE_DEFAULTS)
    if matched_accuracy:
        specs.extend(MATCHED_ACCURACY)
    return specs


def build_feols_benchmarkers(
    *,
    package_defaults: bool = True,
    matched_accuracy: bool = False,
    external: bool = True,
    torch: bool = False,
) -> BenchmarkerBundle:
    """Assemble the feols backends for one experiment family.

    Views are selected here rather than split across drivers. A measurement is
    (design, backend, stopping rule) -> time; running the same designs a second
    time to fill a second table would only pay the data cost twice.
    """
    benchmarkers = [
        PyFeolsBenchmarkerFullApi(
            spec.label, spec.backend, tol=spec.tol, maxiter=spec.maxiter
        )
        for spec in _pyfixest_specs(package_defaults, matched_accuracy)
    ]
    if torch:
        benchmarkers.extend(_torch_benchmarkers())
    if external:
        benchmarkers.extend(cls(label) for label, cls in EXTERNAL_FEOLS)

    if not benchmarkers:
        raise ValueError("No requested benchmark backend is available.")
    return BenchmarkerBundle(benchmarkers=benchmarkers)


def build_fepois_benchmarkers(
    *,
    package_defaults: bool = True,
    matched_accuracy: bool = False,
    external: bool = True,
    iwls_maxiter: int = 100,
) -> BenchmarkerBundle:
    """The same composition for PPML."""
    benchmarkers = [
        PyFepoisBenchmarkerFullApi(
            spec.label,
            spec.backend,
            iwls_maxiter=iwls_maxiter,
            tol=spec.tol,
            maxiter=spec.maxiter,
        )
        for spec in _pyfixest_specs(package_defaults, matched_accuracy)
    ]
    if external:
        benchmarkers.append(FixestFepoisBenchmarker("fixest"))
        benchmarkers.append(GLFixedEffectModelsBenchmarker("GLFEM.jl"))

    if not benchmarkers:
        raise ValueError("No requested benchmark backend is available.")
    return BenchmarkerBundle(benchmarkers=benchmarkers)

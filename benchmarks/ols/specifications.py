"""OLS formula specifications shared by the package benchmark drivers."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.data import FE_COLUMNS

INTERCEPTS = "intercepts"
WORKER_YEAR_SLOPE = "worker-year-slope"
VARYING_SLOPE_BACKENDS = (
    "within-diagonal",
    "within-additive",
    "fixest",
    "FEM.jl",
)


@dataclass(frozen=True, slots=True)
class OlsSpecification:
    """One absorbed formula and the backends that support it."""

    name: str
    fixed_effects: tuple[str, ...]
    backends: tuple[str, ...]
    repetitions: int | None
    varying_slope: str = ""


WORKER_YEAR_SLOPE_SPECIFICATION = OlsSpecification(
    name=WORKER_YEAR_SLOPE,
    fixed_effects=FE_COLUMNS,
    backends=VARYING_SLOPE_BACKENDS,
    repetitions=3,
    varying_slope="indiv_id[year]",
)

"""OLS formula specifications shared by the package benchmark drivers."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.data import FE_COLUMNS

INTERCEPTS = "intercepts"
WORKER_FIRM_YEAR_SLOPES = "worker-firm-year-slopes"
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


WORKER_FIRM_YEAR_SLOPES_SPECIFICATION = OlsSpecification(
    name=WORKER_FIRM_YEAR_SLOPES,
    fixed_effects=FE_COLUMNS,
    backends=VARYING_SLOPE_BACKENDS,
    repetitions=3,
    varying_slope="indiv_id[year]+firm_id[year]",
)

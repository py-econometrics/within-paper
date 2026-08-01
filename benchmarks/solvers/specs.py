"""One solver configuration under one stopping rule.

Split out of the old experiment.py. These names sit above `benchmarks.core`
because they carry solver knowledge: which preconditioners exist, and what the
frozen matched-accuracy settings are.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.solvers.settings import (
    MECHANISM_LSMR_TOL,
    MECHANISM_MAXITER,
    WITHIN_PRECONDITIONERS,
)


# Solver configurations
# ---------------------------------------------------------------------------
# Re-exported under the name the standalone drivers already import. The list
# itself lives with the pinned solver settings, so the ablation arms and the
# demeaner factory cannot come to disagree about which preconditioners exist.
PRECONDITIONERS = WITHIN_PRECONDITIONERS


def preconditioner_config(name: str) -> "PreconditionerConfig":
    """Map a preconditioner name to the `within` enum member."""
    from within import PreconditionerConfig

    try:
        return {
            "off": PreconditionerConfig.Off,
            "diagonal": PreconditionerConfig.Diagonal,
            "additive": PreconditionerConfig.Additive,
        }[name]
    except KeyError:
        raise ValueError(
            f"Unknown preconditioner {name!r}; expected one of {PRECONDITIONERS}"
        ) from None


@dataclass(frozen=True)
class SolverSpec:
    """One solver configuration under one stopping rule.

    ``view`` records which table the row belongs to. Both views are measured in
    one pass, so the distinction has to travel with the record rather than be
    reconstructed from the label later.
    """

    label: str
    preconditioner: str
    tol: float
    maxiter: int
    view: str = "matched-accuracy"

    @property
    def config_id(self) -> str:
        return f"{self.preconditioner}/tol={self.tol:g}/maxiter={self.maxiter}"


def matched_solver_specs() -> tuple[SolverSpec, ...]:
    """The three preconditioners at the frozen matched settings.

    Built from the same constants the PyFixest ablation uses, so the
    standalone diagnostics and the end-to-end timings cannot drift apart.
    """
    return tuple(
        SolverSpec(
            label=f"within-{name}",
            preconditioner=name,
            tol=MECHANISM_LSMR_TOL,
            maxiter=MECHANISM_MAXITER,
        )
        for name in PRECONDITIONERS
    )

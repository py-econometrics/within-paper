"""One measured trial, and the refusal to write an incomplete one.

Split out of the old experiment.py. Nothing here knows which solver produced
the record, which is why it belongs beside the other primitives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.core.results import write_rows


# Run records
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One measured trial, with the fields the protocol requires present.

    ``validate`` returns the protocol violations rather than raising, so a
    driver can record an incomplete run and report it as incomplete instead of
    dropping it. A silently missing diagnostic is what let earlier runs report
    a gate result they had not measured.
    """

    # Identity
    design: str
    n_obs: int
    sample_hash: str
    config_id: str
    solver_label: str
    view: str
    repetition: int

    # Timing decomposition (PROTOCOL.md section 3)
    setup_s: float | None = None
    solve_s: float | None = None
    total_s: float | None = None

    # Convergence
    converged: bool | None = None
    censoring: str = "none"
    iterations_median: float | None = None
    iterations_max: int | None = None
    iterations_sum: int | None = None
    n_converged: int | None = None
    n_solves: int | None = None

    # Accuracy (PROTOCOL.md section 5)
    max_eta: float | None = None
    max_delta: float | None = None
    max_slope_se: float | None = None
    gate_a_measured: bool = False
    clears_gate_a: bool = False

    # Free-form context
    extra: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Names of the protocol requirements this record does not meet."""
        problems: list[str] = []
        if not self.sample_hash:
            problems.append("sample_hash missing: the sample is not identified")
        if self.total_s is None:
            problems.append("total_s missing")
        elif self.setup_s is not None and self.solve_s is not None:
            accounted = self.setup_s + self.solve_s
            if accounted - self.total_s > 1e-6:
                problems.append(
                    f"setup+solve ({accounted:.6f}s) exceeds total ({self.total_s:.6f}s)"
                )
        if self.converged is None:
            problems.append("converged missing")
        if self.censoring not in {"none", "capped", "failed"}:
            problems.append(f"censoring {self.censoring!r} is not a known value")
        if self.converged is False and self.censoring == "none":
            problems.append("a non-converged run must be marked capped or failed")
        if self.max_eta is None:
            problems.append("max_eta missing: no external accuracy was recorded")
        if self.clears_gate_a and not self.gate_a_measured:
            problems.append("clears_gate_a set without all Gate A components measured")
        return problems

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("extra")
        row.update(self.extra)
        return row


def write_records(path: Path, records: list[RunRecord]) -> None:
    """Write records to CSV, refusing to write ones that violate the protocol."""
    if not records:
        raise ValueError("no records to write")
    problems = {
        f"{record.design}@{record.n_obs} {record.config_id} rep={record.repetition}": found
        for record in records
        if (found := record.validate())
    }
    if problems:
        detail = "\n".join(f"  {key}: {'; '.join(v)}" for key, v in problems.items())
        raise ValueError(f"records violate PROTOCOL.md:\n{detail}")
    write_rows(path, [record.to_row() for record in records])

"""Shared experiment layer: sample identity, solver configuration, run records.

Each standalone driver used to carry its own copy of the repo-path shim, the
absorbed-factor column list, the preconditioner mapping, the seed rule, the
sample builder, and the CSV writer. The copies drifted, and the drift was not
cosmetic:

- two drivers derived the seed from the repetition index, so every "repetition"
  measured a different sample, which is the confound PROTOCOL.md section 2
  exists to prevent;
- the standalone tolerances stayed at the package defaults (1e-8, 1,000
  iterations) after the ablation was frozen at the matched settings (1e-12,
  10,000), so the two halves of the mechanism evidence were not comparable.

One definition of each lives here, so that a protocol rule is enforced by the
schema the drivers write through rather than restated in prose beside them.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Absorbed in this order everywhere. MAP cycles through the factors as given,
# so a driver that varies the order is not measuring the same specification.
FE_COLS = ("indiv_id", "firm_id", "year")


def add_repo_paths() -> None:
    """Make `within` and the modular benchmark package importable."""
    within_repo = os.environ.get("WITHIN_REPO")
    if within_repo:
        source = Path(within_repo).expanduser().resolve() / "python"
        if not source.exists():
            raise FileNotFoundError(
                f"WITHIN_REPO has no Python package directory: {source}"
            )
    modular = str(ROOT / "benchmarks" / "modular")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSpec:
    """Identifies one benchmark sample.

    The seed is a function of the design, the size, and the covariate count
    only. It deliberately does not depend on a repetition index: repeated
    timings must run on one fixed sample, or solver variance is confounded with
    DGP variance. Generating a fresh draw per repetition is a separate
    robustness exercise with its own rows.
    """

    design: str
    n_obs: int
    k: int = 1

    @property
    def key(self) -> str:
        """Join key that keeps the size.

        simple and difficult run at several sizes and are not equally connected
        at each, so the family name alone is not an identity.
        """
        return f"{self.design}@{self.n_obs}"

    @property
    def seed(self) -> int:
        offset = {"simple": 0, "difficult": 1}.get(self.design)
        if offset is None:
            offset = int.from_bytes(
                hashlib.sha256(self.design.encode()).digest()[:2], "big"
            ) % 97
        return self.n_obs * 100 + self.k * 17 + offset + 42


@dataclass(frozen=True)
class Sample:
    """A materialized sample plus the hash that identifies it in the record."""

    spec: SampleSpec
    categories: np.ndarray
    rhs: np.ndarray
    rhs_columns: tuple[str, ...]
    sample_hash: str

    @property
    def n_rhs(self) -> int:
        return int(self.rhs.shape[1])


def sample_hash(categories: np.ndarray, rhs: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(categories).tobytes())
    hasher.update(np.ascontiguousarray(rhs).tobytes())
    return hasher.hexdigest()


_SAMPLE_CACHE: dict[tuple[str, int, int, tuple[str, ...]], Sample] = {}


def _factor_codes(frame, columns: Iterable[str]) -> np.ndarray:
    """Zero-based contiguous codes per absorbed factor.

    Factorizing rather than subtracting one keeps this correct for the AKM
    designs, whose identifiers are not guaranteed to be a contiguous 1..m
    range. Only the partition matters for demeaning, so relabelling is safe.
    """
    import pandas as pd

    codes = [pd.factorize(frame[column], sort=True)[0] for column in columns]
    return np.asfortranarray(np.column_stack(codes).astype(np.uint32))


def _akm_frame(spec: SampleSpec):
    """Read one AKM sweep design from its generated Parquet cache."""
    import pandas as pd

    from benchmarks.modular.dgps import get_akm_sweep_scenarios

    data_dir = ROOT / "benchmarks" / "data"
    scenarios = {dgp.dgp_name: dgp for dgp in get_akm_sweep_scenarios(data_dir)}
    if spec.design not in scenarios:
        raise ValueError(
            f"Unknown design {spec.design!r}; expected simple, difficult, or one "
            f"of {sorted(scenarios)}"
        )
    datasets = scenarios[spec.design].generate(n=spec.n_obs, n_iters=1, burn_in=0)
    return pd.read_parquet(datasets[0].data_path)


def load_sample(
    spec: SampleSpec, *, rhs_columns: Iterable[str] | None = None
) -> Sample:
    """Build (or return the cached) sample for ``spec``.

    Caching is what makes the fixed-sample rule real rather than nominal: every
    repetition in a process receives the identical arrays, not merely arrays
    built from the same seed.

    Both design families are reachable, so the standalone diagnostics can run on
    the AKM mobility and sorting designs the mechanism section is about, not
    only on simple and difficult.
    """
    from benchmarks.modular.dgp_functions import base_dgp

    columns = tuple(rhs_columns or ("y", *[f"x{i}" for i in range(1, spec.k + 1)]))
    cache_key = (spec.design, spec.n_obs, spec.k, columns)
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if spec.design.startswith("akm_"):
        frame = _akm_frame(spec)
    else:
        frame = base_dgp(
            n=spec.n_obs, type_=spec.design, k=spec.k, max_k=spec.k, seed=spec.seed
        )
    categories = _factor_codes(frame, FE_COLS)
    rhs = np.asfortranarray(frame[list(columns)].to_numpy(dtype=np.float64))
    del frame
    gc.collect()

    sample = Sample(
        spec=spec,
        categories=categories,
        rhs=rhs,
        rhs_columns=columns,
        sample_hash=sample_hash(categories, rhs),
    )
    _SAMPLE_CACHE[cache_key] = sample
    return sample


def clear_sample_cache() -> None:
    """Drop cached samples. Large designs otherwise pin memory across designs."""
    _SAMPLE_CACHE.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Solver configurations
# ---------------------------------------------------------------------------
PRECONDITIONERS = ("off", "diagonal", "additive")


def preconditioner_config(name: str):
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

    Imported from the same constants the PyFixest ablation uses, so the
    standalone diagnostics and the end-to-end timings cannot drift apart.
    """
    from benchmarks.modular.feols_benchmarkers import MECHANISM_LSMR_TOL, MECHANISM_MAXITER

    return tuple(
        SolverSpec(
            label=f"within-{name}",
            preconditioner=name,
            tol=MECHANISM_LSMR_TOL,
            maxiter=MECHANISM_MAXITER,
        )
        for name in PRECONDITIONERS
    )


# ---------------------------------------------------------------------------
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


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write dict rows to CSV, unioning keys so optional fields stay aligned."""
    if not rows:
        raise ValueError("no rows to write")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkDataset:
    """Dataset passed to each benchmark backend."""

    dataset_id: str
    data_path: Path
    dgp: str
    k: int
    n_obs: int
    iter_type: str
    iter_num: int


class DataGeneratorProtocol(Protocol):
    @property
    def dgp_name(self) -> str: ...

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        """Generate datasets for one (dgp, n) combination."""
        ...


@dataclass(frozen=True)
class FeolsSpec:
    """Settings for one feols call."""

    depvar: str
    covariates: Sequence[str]
    fe_cols: Sequence[str]
    vcov: str | dict[str, str]

    def __post_init__(self) -> None:
        """Freeze column names and reject malformed specifications early."""
        covariates = tuple(self.covariates)
        fe_cols = tuple(self.fe_cols)
        for name, columns in (("covariates", covariates), ("fe_cols", fe_cols)):
            if any(not column for column in columns):
                raise ValueError(f"{name} must not contain empty column names")
            if len(set(columns)) != len(columns):
                raise ValueError(f"{name} must not contain duplicate column names")
        if not self.depvar:
            raise ValueError("depvar must not be empty")
        if self.depvar in covariates or self.depvar in fe_cols:
            raise ValueError("depvar must not also appear as a covariate or fixed effect")
        if isinstance(self.vcov, dict) and set(self.vcov) != {"CRV1"}:
            raise ValueError("vcov dictionaries must contain exactly the CRV1 cluster column")
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "fe_cols", fe_cols)

    @property
    def formula(self) -> str:
        """Build fixest-style formula: y ~ x1 | indiv_id + year."""
        rhs = " + ".join(self.covariates) if self.covariates else "1"
        if self.fe_cols:
            return f"{self.depvar} ~ {rhs} | {' + '.join(self.fe_cols)}"
        return f"{self.depvar} ~ {rhs}"

    @property
    def n_fe(self) -> int:
        return len(self.fe_cols)

    @property
    def k(self) -> int:
        return len(self.covariates)


@dataclass(frozen=True)
class FeolsResult:
    """Result from one feols/fepois call.

    Timing fields are always populated. Diagnostic fields default to None so
    existing OLS harnesses stay unchanged; PPML and accuracy-aware drivers fill
    them when the backend exposes the quantity.
    """

    source_dataset_id: str
    source_k: int | None
    iter_type: str
    iter_num: int
    dgp: str
    model_k: int
    n_obs: int
    n_fe: int
    backend: str
    time: float | None
    success: bool
    error: str | None = None
    # Timing repetition on one fixed sample, distinct from iter_num, which is
    # a DGP replicate (PROTOCOL.md section 2). n_planned is the repetition count
    # the R1/R2/R3 rule chose, so the renderer can tell a complete cell from a
    # crashed one now that the count varies by cell runtime.
    repetition: int = 0
    n_planned: int | None = None
    # Optional diagnostics (plan items 9, 12, 18).
    # Rows retained by the backend after singleton dropping. Recorded per
    # trial so a cell whose retained count differs from its comparators is
    # flagged rather than silently compared (PROTOCOL.md section 2).
    n_retained: int | None = None
    outer_iterations: int | None = None
    inner_iterations_sum: int | None = None
    inner_iterations_max: int | None = None
    preconditioner_build_s: float | None = None
    deviance: float | None = None
    loglik: float | None = None
    max_eta: float | None = None
    beta_x1: float | None = None
    censoring: str | None = None

    def validate(self) -> list[str]:
        """Return record problems before the runner writes a result file."""
        problems: list[str] = []
        if not self.source_dataset_id:
            problems.append("source_dataset_id is missing")
        if not self.backend:
            problems.append("backend is missing")
        if self.n_obs < 0 or self.n_fe < 0 or self.model_k < 0:
            problems.append("model dimensions must be nonnegative")
        if self.time is not None and self.time < 0:
            problems.append("time must be nonnegative")
        if self.success and self.time is None:
            problems.append("a successful result must record its time")
        if self.repetition < 0:
            problems.append("repetition must be nonnegative")
        if self.n_planned is not None:
            if self.n_planned < 1:
                problems.append("n_planned must be positive")
            elif self.repetition >= self.n_planned:
                problems.append("repetition must be smaller than n_planned")
        if self.n_retained is not None and not 0 <= self.n_retained <= self.n_obs:
            problems.append("n_retained must lie between zero and n_obs")
        return problems


class FeolsBenchmarkerProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def cache_key(self) -> dict: ...

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        """Benchmark one feols backend on a list of datasets for a fixed spec."""
        ...

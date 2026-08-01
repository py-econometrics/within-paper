"""Identifying and materialising one benchmark sample.

The seed rule and the cache together are what make "repeated timings on one
fixed sample" real rather than nominal. Two drivers once derived the seed from
the repetition index, so every "repetition" measured a different draw, which is
the confound PROTOCOL.md section 2 exists to prevent.

Split out of the old experiment.py, which also held solver specs and run
records. The three had different dependencies and different consumers; keeping
them together forced this module's importers to pull in the solver layer.
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from benchmarks.core.paths import DATA_DIR

# Absorbed in this order everywhere. MAP cycles through the factors as given,
# so a driver that varies the order is not measuring the same specification.
FE_COLS = ("indiv_id", "firm_id", "year")


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


def _akm_frame(spec: SampleSpec) -> "pd.DataFrame":
    """Read one AKM sweep design from its generated Parquet cache."""
    import pandas as pd

    from benchmarks.dgp.scenarios import get_akm_sweep_scenarios

    scenarios = {dgp.dgp_name: dgp for dgp in get_akm_sweep_scenarios(DATA_DIR)}
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
    from benchmarks.dgp.base import base_dgp

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

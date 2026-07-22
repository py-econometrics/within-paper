from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .akm_dgp import AKMConfig, simulate_akm_panel
    from .dgp_functions import PAPER_BASE_MAX_K, paper_base_dgp
    from .interfaces import BenchmarkDataset
except ImportError:
    from akm_dgp import AKMConfig, simulate_akm_panel
    from dgp_functions import PAPER_BASE_MAX_K, paper_base_dgp
    from interfaces import BenchmarkDataset


def _seed_for(dgp_name: str, n: int, iteration: int) -> int:
    """Build deterministic seeds so benchmark runs are reproducible."""
    stable_offset = int.from_bytes(
        hashlib.sha256(dgp_name.encode("utf-8")).digest()[:2], "big"
    ) % 97
    dgp_offset = {"simple": 0, "difficult": 1}.get(dgp_name, stable_offset)
    return n * 100 + iteration * 17 + dgp_offset + 42


BASE_DGP_SCHEMA = pa.schema(
    [
        ("indiv_id", pa.int64()),
        ("firm_id", pa.int64()),
        ("year", pa.int64()),
        ("y", pa.float64()),
        ("negbin_y", pa.int64()),
        ("x1", pa.float64()),
    ]
)
AKM_DGP_SCHEMA = pa.schema(
    [
        ("indiv_id", pa.int64()),
        ("firm_id", pa.int32()),
        ("year", pa.int64()),
        ("x1", pa.float64()),
        ("y", pa.float64()),
    ]
)


def _generator_source_hash(source: str) -> str:
    """Digest one DGP source so only that generator's cache is invalidated.

    Hashing only the parameter JSON would let a changed generator silently reuse
    stale Parquet inputs while provenance attributes them to the current code.
    """
    source_path = Path(__file__).parent / source
    return hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]


_BASE_GENERATOR_SOURCE_HASH = _generator_source_hash("dgp_functions.py")
_AKM_GENERATOR_SOURCE_HASH = _generator_source_hash("akm_dgp.py")


def _param_hash(params: dict, generator_source_hash: str) -> str:
    """Stable SHA-256 digest of the params plus the generator source hash."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    payload = f"{generator_source_hash}:{canonical}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _has_schema(data_path: Path, expected_schema: pa.Schema) -> bool:
    try:
        return pq.read_schema(data_path).equals(expected_schema, check_metadata=False)
    except (OSError, pa.ArrowException):
        return False


def _is_cached(
    data_path: Path, expected_hash: str, expected_schema: pa.Schema
) -> bool:
    hash_path = data_path.with_suffix(".hash")
    if data_path.exists() and hash_path.exists():
        return (
            hash_path.read_text().strip() == expected_hash
            and _has_schema(data_path, expected_schema)
        )
    return False


def _write_hash(data_path: Path, param_hash: str) -> None:
    data_path.with_suffix(".hash").write_text(param_hash)


def _generate_datasets(
    *,
    dgp_name: str,
    n: int,
    k: int,
    n_iters: int,
    burn_in: int,
    data_dir: Path,
    make_params: Callable[[int], dict],
    generate: Callable[[int], Any],
    generator_source_hash: str,
    expected_schema: pa.Schema,
) -> list[BenchmarkDataset]:
    """Generate parquet-backed datasets for one DGP/size combination."""
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[BenchmarkDataset] = []
    total = burn_in + n_iters
    cached_count = 0

    for i in range(1, total + 1):
        iter_type = "burnin" if i <= burn_in else "iter"
        iter_num = i if i <= burn_in else i - burn_in
        dataset_id = f"{dgp_name}_{n}_k{k}_{iter_type}_{iter_num}"
        data_path = data_dir / f"{dataset_id}.parquet"

        seed = _seed_for(dgp_name, n, i)
        params = make_params(seed)
        h = _param_hash(params, generator_source_hash)

        if _is_cached(data_path, h, expected_schema):
            cached_count += 1
            n_obs_actual = pq.read_metadata(data_path).num_rows
        else:
            df = generate(seed)
            n_obs_actual = len(df)
            df.to_parquet(data_path, index=False)
            if not _has_schema(data_path, expected_schema):
                raise ValueError(
                    f"Generated dataset has an unexpected schema: {data_path}"
                )
            _write_hash(data_path, h)

        datasets.append(
            BenchmarkDataset(
                dataset_id=dataset_id,
                data_path=data_path.resolve(),
                dgp=dgp_name,
                k=k,
                n_obs=n_obs_actual,
                iter_type=iter_type,
                iter_num=iter_num,
            )
        )

    if cached_count == total:
        print(f"  [{dgp_name} n={n:,} k={k}] all {total} cached")
    elif cached_count > 0:
        print(
            f"  [{dgp_name} n={n:,} k={k}] "
            f"{cached_count}/{total} cached, {total - cached_count} generated"
        )

    return datasets


@dataclass(frozen=True)
class AKMSweepScenario:
    name: str
    overrides: dict[str, Any]


_AKM_DEFAULTS: dict[str, Any] = {
    "n_time": 10,
    "n_firms": 50_000,
    "n_industries": 5,
    "var_alpha": 1.0,
    "var_psi": 0.5,
    "var_phi": 0.1,
    "var_epsilon": 1.0,
    "gamma": 1.0,
    "rho_size": 0.6,
    "rho": 1.0,
    "delta": 0.1,
    "lambda_": 0.8,
    "beta_x1": 0.5,
    "n_match_bins": 2048,
}

def _scenario(name: str, **overrides: Any) -> AKMSweepScenario:
    return AKMSweepScenario(name=name, overrides=overrides)


def _infer_worker_count(target_n_obs: int, params: dict[str, Any]) -> int:
    n_time = max(int(params["n_time"]), 1)
    floor_workers = max(1, int(target_n_obs / n_time))
    candidates = {floor_workers, floor_workers + 1}
    return min(
        candidates,
        key=lambda n_workers: (
            abs(n_workers * n_time - target_n_obs),
            n_workers,
        ),
    )


def _akm_sweep_scenarios() -> list[AKMSweepScenario]:
    return [
        # sorting
        _scenario("akm_sorting_1", rho=0.0),
        _scenario("akm_sorting_2", rho=5.0),
        _scenario("akm_sorting_3", rho=20.0),
        _scenario("akm_sorting_4", rho=50.0),
        _scenario("akm_sorting_5", rho=100.0),
        # mobility
        _scenario("akm_mobility_1", delta=1.0),
        _scenario("akm_mobility_2", delta=0.5),
        _scenario("akm_mobility_3", delta=0.05),
        _scenario("akm_mobility_4", delta=0.01),
        _scenario("akm_mobility_5", delta=0.005),
        _scenario("akm_mobility_6", delta=0.001),
    ]


def get_akm_sweep_scenario_names() -> tuple[str, ...]:
    return tuple(scenario.name for scenario in _akm_sweep_scenarios())


class BaseDGP:
    def __init__(self, data_dir: Path, dgp_type: str = "simple"):
        self._data_dir = data_dir
        self._dgp_type = dgp_type

    @property
    def dgp_name(self) -> str:
        return self._dgp_type

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        dgp_type = self._dgp_type
        print(f"[data] generating {self.dgp_name} n={n:,} k=1")
        return _generate_datasets(
            dgp_name=self.dgp_name,
            n=n,
            k=1,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
            make_params=lambda seed: {
                "dgp_type": dgp_type,
                "n": n,
                "active_k": 1,
                "latent_k": PAPER_BASE_MAX_K,
                "schema_version": 1,
                "seed": seed,
            },
            generate=lambda seed: paper_base_dgp(n=n, type_=dgp_type, seed=seed),
            generator_source_hash=_BASE_GENERATOR_SOURCE_HASH,
            expected_schema=BASE_DGP_SCHEMA,
        )


class AKMSweepDGP:
    def __init__(
        self,
        data_dir: Path,
        name: str,
        defaults: dict[str, Any] | None = None,
        **overrides: Any,
    ):
        self._data_dir = data_dir
        self._name = name
        self._defaults = dict(defaults or _AKM_DEFAULTS)
        self._overrides = overrides

    @property
    def dgp_name(self) -> str:
        return self._name

    def _build_config(self, n: int = 1_000_000) -> AKMConfig:
        params = {**self._defaults, **self._overrides}
        n_workers = int(params.pop("n_workers", _infer_worker_count(n, params)))
        return AKMConfig(n_workers=n_workers, **params)

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        config = self._build_config(n=n)
        return _generate_datasets(
            dgp_name=self.dgp_name,
            n=n,
            k=1,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
            make_params=lambda seed: {**asdict(config), "seed": seed},
            generate=lambda seed: simulate_akm_panel(config, seed=seed),
            generator_source_hash=_AKM_GENERATOR_SOURCE_HASH,
            expected_schema=AKM_DGP_SCHEMA,
        )


def _get_akm_scenarios(
    data_dir: Path,
    scenario_defs: list[AKMSweepScenario],
    defaults: dict[str, Any],
    names: list[str] | None = None,
) -> list[AKMSweepDGP]:
    scenario_map = {scenario.name: scenario for scenario in scenario_defs}
    scenario_names = names or [scenario.name for scenario in scenario_defs]
    unknown = sorted(set(scenario_names) - set(scenario_map))
    if unknown:
        raise ValueError(f"Unknown AKM sweep scenario(s): {', '.join(unknown)}")

    return [
        AKMSweepDGP(
            data_dir=data_dir,
            name=scenario_map[name].name,
            defaults=defaults,
            **scenario_map[name].overrides,
        )
        for name in scenario_names
    ]


def get_akm_sweep_scenarios(
    data_dir: Path, names: list[str] | None = None
) -> list[AKMSweepDGP]:
    return _get_akm_scenarios(
        data_dir,
        _akm_sweep_scenarios(),
        _AKM_DEFAULTS,
        names=names,
    )

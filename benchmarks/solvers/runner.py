from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.core.interfaces import (
    BenchmarkDataset,
    DataGeneratorProtocol,
    FeolsBenchmarkerProtocol,
    FeolsSpec,
)
from benchmarks.core.timing import randomized_order
from benchmarks.solvers.common import serialize_result

_CACHE_SCHEMA_VERSION = 1


def _backend_slug(backend: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", backend).strip("_").lower()
    return slug or "backend"


def _backend_output_csv(output_csv: Path, backend: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}__{_backend_slug(backend)}.csv")


def _cache_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".metadata.json")


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_signature(
    benchmarker: FeolsBenchmarkerProtocol,
    datasets: list[BenchmarkDataset],
    specs: list[FeolsSpec],
) -> str:
    cache_key = getattr(benchmarker, "cache_key", None)
    if not callable(cache_key):
        raise TypeError(
            f"{benchmarker.name} does not expose cache_key(); "
            "it cannot be used with --reuse-existing"
        )
    payload: dict[str, Any] = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "backend": cache_key(),
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "dgp": dataset.dgp,
                "k": dataset.k,
                "n_obs": dataset.n_obs,
                "iter_type": dataset.iter_type,
                "iter_num": dataset.iter_num,
                "file": _file_identity(dataset.data_path),
            }
            for dataset in datasets
        ],
        "specs": [
            {
                "depvar": spec.depvar,
                "covariates": spec.covariates,
                "fe_cols": spec.fe_cols,
                "vcov": spec.vcov,
            }
            for spec in specs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _cache_matches(csv_path: Path, signature: str) -> bool:
    metadata_path = _cache_metadata_path(csv_path)
    if not csv_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == _CACHE_SCHEMA_VERSION
        and metadata.get("signature") == signature
    )


def _write_cache_metadata(csv_path: Path, signature: str) -> None:
    _cache_metadata_path(csv_path).write_text(
        json.dumps(
            {"schema_version": _CACHE_SCHEMA_VERSION, "signature": signature},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _design_groups(datasets: list[BenchmarkDataset]) -> list[list[BenchmarkDataset]]:
    """Keep all stored replicates of one design together for backend ordering."""
    groups: dict[tuple[str, int, int], list[BenchmarkDataset]] = {}
    for dataset in datasets:
        groups.setdefault((dataset.dgp, dataset.n_obs, dataset.k), []).append(dataset)
    return list(groups.values())


def generate_datasets(
    dgps: list[DataGeneratorProtocol],
    sizes: list[int],
    n_iters: int,
    burn_in: int,
) -> list[BenchmarkDataset]:
    all_datasets: list[BenchmarkDataset] = []
    for dgp in dgps:
        for n in sizes:
            all_datasets.extend(dgp.generate(n=n, n_iters=n_iters, burn_in=burn_in))
    print(f"[data] {len(all_datasets)} datasets ready")
    return all_datasets


def run_benchmarks(
    benchmarkers: list[FeolsBenchmarkerProtocol],
    datasets: list[BenchmarkDataset],
    specs: list[FeolsSpec],
    output_csv: Path,
    *,
    reuse_existing: bool = False,
    order_seed: int = 20260726,
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    slugs: dict[str, str] = {}
    for benchmarker in benchmarkers:
        slug = _backend_slug(benchmarker.name)
        if previous := slugs.get(slug):
            raise ValueError(
                f"Backend names {previous!r} and {benchmarker.name!r} share output slug {slug!r}"
            )
        slugs[slug] = benchmarker.name

    to_run: list[FeolsBenchmarkerProtocol] = []
    signatures: dict[str, str] = {}
    for benchmarker in benchmarkers:
        csv_path = _backend_output_csv(output_csv, benchmarker.name)
        signature = _cache_signature(benchmarker, datasets, specs)
        signatures[benchmarker.name] = signature
        if reuse_existing and _cache_matches(csv_path, signature):
            print(f"[skip] {benchmarker.name}: matching {csv_path.name} already exists")
            frames.append(pd.read_csv(csv_path))
            continue
        if reuse_existing and csv_path.exists():
            print(f"[stale] {benchmarker.name}: rerunning {csv_path.name}")
        elif csv_path.exists():
            print(f"[overwrite] {benchmarker.name}: {csv_path.name}")
        to_run.append(benchmarker)

    results_by_backend: dict[str, list] = {benchmarker.name: [] for benchmarker in to_run}
    if to_run:
        group_index = 0
        for spec in specs:
            spec_datasets = [dataset for dataset in datasets if dataset.k >= spec.k]
            for group in _design_groups(spec_datasets):
                ordered = randomized_order(to_run, order_seed + group_index)
                first = group[0]
                print(
                    f"[bench] {first.dgp} n={first.n_obs:,} backend order "
                    f"(seed={order_seed + group_index}): "
                    + ", ".join(benchmarker.name for benchmarker in ordered),
                    flush=True,
                )
                for benchmarker in ordered:
                    results_by_backend[benchmarker.name].extend(
                        benchmarker.run(group, spec)
                    )
                group_index += 1

    for benchmarker in to_run:
        backend_results = results_by_backend[benchmarker.name]
        if not backend_results:
            continue
        df = pd.DataFrame([serialize_result(result) for result in backend_results])
        df = df[df["iter_type"] != "burnin"].copy()
        csv_path = _backend_output_csv(output_csv, benchmarker.name)
        df.to_csv(csv_path, index=False)
        _write_cache_metadata(csv_path, signatures[benchmarker.name])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

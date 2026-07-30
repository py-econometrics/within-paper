from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from benchmarks.modular.interfaces import (
    BenchmarkDataset,
    DataGeneratorProtocol,
    FeolsBenchmarkerProtocol,
    FeolsResult,
    FeolsSpec,
)
from benchmarks.modular.timing import randomized_order
def _serialize_result(result: FeolsResult) -> dict:
    return asdict(result)


def _backend_slug(backend: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", backend).strip("_").lower()
    return slug or "backend"


def _backend_output_csv(output_csv: Path, backend: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}__{_backend_slug(backend)}.csv")


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

    # Randomize the backend order so that thermal drift and background load do
    # not line up with a particular backend (PROTOCOL.md section 4). Seeded, so
    # the order is reproducible and printed rather than unknown afterwards.
    ordered = randomized_order(benchmarkers, order_seed)
    print(
        f"[bench] backend order (seed={order_seed}): "
        + ", ".join(b.name for b in ordered),
        flush=True,
    )

    for benchmarker in ordered:
        csv_path = _backend_output_csv(output_csv, benchmarker.name)
        if csv_path.exists() and reuse_existing:
            print(f"[skip] {benchmarker.name}: {csv_path.name} already exists")
            frames.append(pd.read_csv(csv_path))
            continue
        if csv_path.exists():
            print(f"[overwrite] {benchmarker.name}: {csv_path.name}")
        backend_results: list[FeolsResult] = []
        for spec in specs:
            spec_datasets = [dataset for dataset in datasets if dataset.k >= spec.k]
            backend_results.extend(benchmarker.run(spec_datasets, spec))
        if not backend_results:
            continue
        df = pd.DataFrame([_serialize_result(r) for r in backend_results])
        df = df[df["iter_type"] != "burnin"].copy()
        df.to_csv(csv_path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

"""Measure peak memory in a fresh process for each OLS cell."""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from benchmarks.data import make_base_data

ROOT = Path(__file__).absolute().parents[1]
OUTPUT = ROOT / "results" / "runs" / "latest" / "memory.csv"
CELLS = (("100k", 100_000), ("1m", 1_000_000))


def _worker(queue: mp.Queue, design: str, n_obs: int, seed: int, backend: str) -> None:
    os.environ["RAYON_NUM_THREADS"] = os.environ["BENCH_THREADS"]
    import pyfixest as pf

    frame = make_base_data(n_obs, design, seed)
    demeaner = pf.MapDemeaner() if backend == "rust-map" else pf.LsmrDemeaner()
    started = time.perf_counter()
    try:
        fit = pf.feols(
            "y ~ x1 | indiv_id + firm_id + year", frame, vcov="iid",
            copy_data=False, store_data=False, demeaner=demeaner,
        )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
        queue.put(
            {
                "runtime_s": time.perf_counter() - started,
                "rss_mb": peak_mb,
                "n_retained": int(fit._N),
                "converged": True,
                "error": "",
            }
        )
    except Exception as error:
        queue.put(
            {
                "runtime_s": time.perf_counter() - started,
                "rss_mb": None,
                "n_retained": None,
                "converged": False,
                "error": str(error),
            }
        )


def main() -> None:
    context = mp.get_context("spawn")
    rows = []
    for size, n_obs in CELLS:
        for design, seed in (("simple", 42), ("difficult", 43)):
            for backend in ("rust-map", "within"):
                queue = context.Queue()
                process = context.Process(target=_worker, args=(queue, design, n_obs, seed, backend))
                process.start()
                process.join()
                if process.exitcode:
                    raise RuntimeError(f"memory worker exited with status {process.exitcode}")
                row = queue.get()
                row.update(size=size, design=design, backend=backend, n_obs=n_obs, model_k=1)
                rows.append(row)
                print(
                    f"bench-memory / OLS / {design}-{size} / {backend}: "
                    f"{row['runtime_s']:.3f} s",
                    flush=True,
                )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()

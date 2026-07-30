"""Generate simple and difficult DGP datasets (k=1, 3 FE) as parquet."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULAR_DIR = ROOT / "benchmarks" / "modular"
if str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

from dgp_functions import base_dgp  # noqa: E402

SIZES = {"100k": 100_000, "1m": 1_000_000}
K = 1
SEED = 42
KEEP_COLS = ["indiv_id", "firm_id", "year", "y", *[f"x{i}" for i in range(1, K + 1)]]


for label, n in SIZES.items():
    for dgp_type in ["simple", "difficult"]:
        print(f"Generating {dgp_type} {label}...")
        df = base_dgp(n=n, type_=dgp_type, k=K, max_k=K, seed=SEED)[KEEP_COLS]
        path = f"data/{dgp_type}_{label}.parquet"
        df.to_parquet(path, index=False)
        print(f"  -> {path} ({len(df):,} rows)")

"""Generate simple and difficult DGP datasets (k=10, 3 FE) as parquet."""

import numpy as np
import pandas as pd

SIZES = {"100k": 100_000, "1m": 1_000_000}
NB_YEAR = 10
NB_INDIV_PER_FIRM = 23
K = 10
SEED = 42


def base_dgp(n, type_, seed):
    """Adapted from pyfixest/benchmarks/modular/dgp_functions.py (MIT, Kyle Butts)."""
    rng = np.random.default_rng(seed)
    nb_indiv = round(n / NB_YEAR)
    nb_firm = round(nb_indiv / NB_INDIV_PER_FIRM)
    n_obs = nb_indiv * NB_YEAR

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), NB_YEAR)
    year = np.tile(np.arange(1, NB_YEAR + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    else:
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]

    x = rng.standard_normal((n_obs, K))
    betas = 1.0 / np.arange(1, K + 1, dtype=float)
    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(NB_YEAR)[year - 1]
    y = x @ betas + firm_fe + unit_fe + year_fe + rng.standard_normal(n_obs)

    data = {"indiv_id": indiv_id, "firm_id": firm_id, "year": year, "y": y}
    for j in range(K):
        data[f"x{j + 1}"] = x[:, j]
    return pd.DataFrame(data)


for label, n in SIZES.items():
    for dgp_type in ["simple", "difficult"]:
        print(f"Generating {dgp_type} {label}...")
        df = base_dgp(n, dgp_type, SEED)
        path = f"data/{dgp_type}_{label}.parquet"
        df.to_parquet(path, index=False)
        print(f"  -> {path} ({len(df):,} rows)")

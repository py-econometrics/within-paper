"""Numerical agreement between rust (MAP) and rust-cg (within) backends.

Reports: max |coef diff|, max |residual diff|, variance-component diff.
Runs on 100k simple and difficult DGPs.
"""

import warnings
import numpy as np
import pandas as pd
import pyfixest as pf

FML = "y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 | indiv_id + firm_id + year"

print(f"{'dgp':<12} {'metric':<28} {'rust vs rust-cg':>16}")
print("-" * 60)

for dgp_type in ["simple", "difficult"]:
    df = pd.read_parquet(f"data/{dgp_type}_100k.parquet")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fit_map = pf.feols(FML, data=df, vcov="iid", demeaner_backend="rust",
                           copy_data=False, store_data=True)
        fit_cg = pf.feols(FML, data=df, vcov="iid", demeaner_backend="rust-cg",
                          copy_data=False, store_data=True)

    # Coefficient differences
    coef_map = np.asarray(fit_map.coef())
    coef_cg = np.asarray(fit_cg.coef())
    max_coef_diff = np.max(np.abs(coef_map - coef_cg))
    print(f"{dgp_type:<12} {'max |coef diff|':<28} {max_coef_diff:>16.2e}")

    # Residual differences
    resid_map = np.asarray(fit_map.resid())
    resid_cg = np.asarray(fit_cg.resid())
    max_resid_diff = np.max(np.abs(resid_map - resid_cg))
    resid_norm_map = np.linalg.norm(resid_map)
    rel_resid_diff = np.linalg.norm(resid_map - resid_cg) / resid_norm_map
    print(f"{'':<12} {'max |resid diff|':<28} {max_resid_diff:>16.2e}")
    print(f"{'':<12} {'rel resid norm diff':<28} {rel_resid_diff:>16.2e}")

    # Firm-effect variance component
    fe_map = fit_map.fixef()
    fe_cg = fit_cg.fixef()
    var_firm_map = np.var(list(fe_map["C(firm_id)"].values()))
    var_firm_cg = np.var(list(fe_cg["C(firm_id)"].values()))
    var_diff = abs(var_firm_map - var_firm_cg)
    print(f"{'':<12} {'var(firm FE) MAP':<28} {var_firm_map:>16.6f}")
    print(f"{'':<12} {'var(firm FE) within':<28} {var_firm_cg:>16.6f}")
    print(f"{'':<12} {'|diff var(firm FE)|':<28} {var_diff:>16.2e}")
    print()

"""Tolerance-scaling experiment: difficult DGP only.

Pins `within` (rust-cg) at a very tight tolerance as the reference,
then varies both MAP (`rust`) and `within` (`rust-cg`) across a
tolerance ladder. This separates agreement about the least-squares
target from backend-specific convergence behavior on a hard graph.

MAP's default iteration cap (10_000) is too low for it to converge at
tight tolerances on the difficult design. We raise it to 100_000 so
that the table shows actual convergence behavior, not just early stops.
"""

import time
import warnings
import numpy as np
import pandas as pd
import pyfixest as pf

FML = "y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 | indiv_id + firm_id + year"
GT_TOL = 1e-12
TOL_LADDER = [1e-6, 1e-8, 1e-10]
MAXITER = 100_000

df = pd.read_parquet("data/difficult_100k.parquet")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

    # Reference: within at very tight tolerance
    fit_gt = pf.feols(FML, data=df, vcov="iid", demeaner_backend="rust-cg",
                      fixef_tol=GT_TOL, fixef_maxiter=MAXITER,
                      copy_data=False, store_data=False)

coef_gt = np.asarray(fit_gt.coef())
print(f"reference: within tol={GT_TOL:.0e}")
print(f"{'method':<10} {'tol':>10}  {'max |coef diff|':>16}  {'time':>8}  status")
print("-" * 64)

for method, backend in [("MAP", "rust"), ("within", "rust-cg")]:
    for tol in TOL_LADDER:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                t0 = time.perf_counter()
                fit = pf.feols(FML, data=df, vcov="iid", demeaner_backend=backend,
                               fixef_tol=tol, fixef_maxiter=MAXITER,
                               copy_data=False, store_data=False)
                elapsed = time.perf_counter() - t0
            except ValueError as exc:
                print(f"{method:<10} {tol:>10.0e}  {'--':>16}  {'--':>8}  {exc}", flush=True)
                continue

        coef = np.asarray(fit.coef())
        max_diff = np.max(np.abs(coef - coef_gt))

        print(f"{method:<10} {tol:>10.0e}  {max_diff:>16.2e}  {elapsed:>7.1f}s  converged", flush=True)

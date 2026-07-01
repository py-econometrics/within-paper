"""Memory + time benchmark for pyfixest backends (rust=MAP, rust-cg=within).

Runs each (size, dgp, backend) combo in a subprocess for clean peak-RSS numbers.
"""

import subprocess
import sys

FML = "y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 | indiv_id + firm_id + year"

SCRIPT = """\
import resource, sys, time, warnings, gc
import pandas as pd
import pyfixest as pf

size, dgp_type, backend = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(f"data/{{dgp_type}}_{{size}}.parquet")
gc.collect()
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    fit = pf.feols("{fml}", data=df, vcov="iid",
                   demeaner_backend=backend,
                   copy_data=False, store_data=False)
elapsed = time.perf_counter() - t0
divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // divisor
print(f"{{size:<6}} {{dgp_type:<12}} {{backend:<10}} {{elapsed:>8.2f}}s  {{rss:>6}} MB")
""".format(fml=FML)

for size in ["100k", "1m"]:
    for dgp in ["simple", "difficult"]:
        for backend in ["rust", "rust-cg"]:
            proc = subprocess.run(
                [sys.executable, "-c", SCRIPT, size, dgp, backend],
                capture_output=True, text=True, timeout=600,
            )
            print(proc.stdout.strip())
            if proc.stderr.strip():
                print(proc.stderr.strip(), file=sys.stderr)

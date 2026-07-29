"""Memory + time benchmark for pyfixest backends (rust=MAP, rust-cg=within).

Runs each (size, dgp, backend) combo in a subprocess for clean peak-RSS numbers.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

FML = "y ~ x1 | indiv_id + firm_id + year"

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
if not fit.convergence:
    raise RuntimeError("PyFixest model did not converge")
elapsed = time.perf_counter() - t0
divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // divisor
print(f"{{size:<6}} {{dgp_type:<12}} {{backend:<10}} {{elapsed:>8.2f}}s  {{rss:>6}} MB")
""".format(fml=FML)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out", type=Path, default=Path("results/runs/latest/memory.csv"),
    help="Structured result CSV written in addition to the console table.",
)
args = parser.parse_args()
rows = []

for size in ["100k", "1m"]:
    for dgp in ["simple", "difficult"]:
        for backend in ["rust", "rust-cg"]:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", SCRIPT, size, dgp, backend],
                    capture_output=True, text=True, timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(
                    exc.cmd,
                    returncode=124,
                    stdout=exc.stdout or "",
                    stderr=f"Timed out after {exc.timeout} seconds",
                )
            print(proc.stdout.strip())
            fields = proc.stdout.split()
            if proc.returncode == 0 and len(fields) >= 5:
                rows.append(
                    {
                        "size": size,
                        "dgp": dgp,
                        "model_k": 1,
                        "backend": backend,
                        "time_s": fields[3].removesuffix("s"),
                        "rss_mb": fields[4],
                        "success": True,
                        "error": "",
                    }
                )
            else:
                rows.append(
                    {
                        "size": size,
                        "dgp": dgp,
                        "model_k": 1,
                        "backend": backend,
                        "time_s": "",
                        "rss_mb": "",
                        "success": False,
                        "error": proc.stderr.strip(),
                    }
                )
            if proc.stderr.strip():
                print(proc.stderr.strip(), file=sys.stderr)

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {args.out}")

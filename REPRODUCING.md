# Reproducing the paper

The paper imports its tables and numerical claims from generated Typst files. Do not edit
`generated/` directly; `pixi run render-paper-results` rebuilds those files from
`results/paper/benchmark_tables.json`.

## Install the runtimes

Install Pixi and create the Python/Typst environment:

```bash
pixi install
```

The benchmark suite uses native R and Julia on macOS. Install R separately from Pixi
because the `fixest` benchmarks use R's multicore runtime. Julia uses the project and
manifest in `benchmarks/julia-env/`.

Install the R package versions used for the paper in your system R library:

```r
install.packages("pak", repos = "https://cloud.r-project.org")
pak::pkg_install(c("arrow@24.0.0", "fixest@0.14.2", "jsonlite@2.0.0"))
```

Set the thread counts before starting R or Julia. The reference run used the ten cores of
an Apple M4 Mac mini:

```bash
export BENCH_THREADS=10
export JULIA_NUM_THREADS=10
```

`BENCH_THREADS` sets the thread count for R `fixest`. Julia reads `JULIA_NUM_THREADS`
only at startup. The scripts stop if either variable is unset or the process starts with
a different thread count.

Install the Julia packages and check both external runtimes:

```bash
pixi run setup-julia-env
pixi run check-external-runtimes
```

The check reports the R and Julia versions, package versions, and active thread counts.
Run it before starting the long benchmarks.

The setup-cost benchmark uses the Pixi-locked `within-py` package. Developers can point
`WITHIN_REPO` to a local checkout, but `collect` rejects paper runs with that override.

Every reported regression has one slope covariate, `x1`. PPML absorbs worker, firm, and
year effects. The other experiments vary the DGP, sample size, fixed-effect count, and
backend.

## Download the Correia data

The synthetic benchmarks generate their inputs locally. The Correia HDFE CSV files are
downloaded separately and are not tracked by Git:

```bash
pixi run fetch-correia
```

`fetch-correia` reads the manifests under `data/correia_data/metadata/`, downloads the
archives, verifies the archive and CSV hashes, and writes the CSVs to
`data/correia_data/`. Use `--offline` to verify files already on disk without network
access:

```bash
pixi run python scripts/paper_results.py fetch-correia --offline
```

## Run everything

`collect` reads every raw result file in the active directories, so archive old files
before a new run:

```bash
pixi run archive-legacy-results
```

The command moves untracked result CSVs, run metadata, and benchmark figures to
`results/legacy/`. It does not remove generated input data or tracked files.

Commit benchmark and documentation changes first. `collect` stops if tracked files differ
from the recorded commit.

Run the benchmarks and compile the paper:

```bash
BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run --locked reproduce-paper
```

The command sets both thread counts for this invocation, so no prior `export` is needed.
It runs the AKM, OLS, Correia, PPML, memory, numerical-agreement, and setup-cost
benchmarks, then computes the spectral gaps. The hard synthetic cases take several
hours, and the Correia CSVs use about 600 MB of disk space.

Once the Pixi, R, and Julia packages and the Correia files are installed, the benchmark
suite does not need network access.

## Run one stage

```bash
BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run reproduce-results
pixi run render-paper-results
pixi run compile
pixi run verify-paper-results
```

`reproduce-results` runs the benchmarks and writes the table data. Raw timing CSVs are
stored in `benchmarks/results/` and `results/runs/latest/` and are ignored by Git.
`collect` saves their paths and hashes, runtime and package versions, code hash, and
package locations in `results/runs/latest/provenance.json`.

`render-paper-results` updates the tracked Typst files. `verify-paper-results` rebuilds
the tables, validates the saved hashes and code version, checks that each registered
source exists and each generated table is included in the manuscript, and rejects
missing trials. `reproduce-paper` runs the same check before compiling the PDF.

Each locally reproduced timing cell contains three attempted trials. If all three
converge, the cell reports their median. If one or two converge, it reports their median
and the count, such as `(2/3)`. If none converge, it reports `failed (0/3)`. Verification
accepts non-convergence only when all three attempts are recorded; missing trials fail
verification. The legacy CUDA timings are exempt because their hardware and trial
metadata are unavailable. They cannot be reproduced on the reference machine and should
not be compared with its CPU timings.

## Compare results

Coefficient and graph diagnostics should match to the reported precision. Runtime and
peak RSS vary with hardware and system load. Except for the legacy CUDA cells, the paper
reports measurements from a ten-core Apple M4 run. Other hardware will produce different
timings.

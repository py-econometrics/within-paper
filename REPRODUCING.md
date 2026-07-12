# Reproducing the paper

The paper reads its tables and result-dependent prose from generated Typst files. Do not
edit files under `generated/` by hand; `pixi run render-paper-results` rewrites them from
`results/paper/benchmark_tables.json`.

## Install the runtimes

Install Pixi and create the Python/Typst environment:

```bash
pixi install
```

The benchmark suite uses native R and Julia on macOS. R is kept outside Pixi because the
`fixest` benchmarks use its multicore runtime. Julia uses the project and manifest in
`benchmarks/julia-env/`.

Install the checked R package versions in your system R library:

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

`BENCH_THREADS` controls the R `fixest` calls. Julia reads `JULIA_NUM_THREADS` only when
the process starts. The R and Julia benchmark scripts stop if either value is missing or
does not match the running process.

Install the Julia packages and check both external runtimes:

```bash
pixi run setup-julia-env
pixi run check-external-runtimes
```

The check reports the R and Julia versions, package versions, and active thread counts.
Run it before starting the long benchmarks.

The setup-cost benchmark uses the `within-py` package locked by Pixi. Developers may set
`WITHIN_REPO` for an explicit local checkout, but paper-result collection rejects that
override.

All regressions reported in the paper use `x1` as their only slope covariate. The DGP,
sample size, number of fixed effects, and software backend change across experiments.

## Download the Correia data

The synthetic benchmark tasks create deterministic local inputs. The Correia HDFE CSV files
are larger and are not tracked by Git. Download and checksum them with:

```bash
pixi run fetch-correia
```

The command reads the manifests under `data/correia_data/metadata/`, downloads each
archive, checks the archive and extracted CSV hashes, and writes the CSVs to
`data/correia_data/`. It is safe to rerun. To check files that are already present without
using the network:

```bash
pixi run python scripts/paper_results.py fetch-correia --offline
```

## Run everything

Old raw results can accidentally be mistaken for the current run. Move them to a dated
local archive first if you want a clean result directory:

```bash
pixi run archive-legacy-results
```

The command moves untracked result CSVs, run metadata, and benchmark figures to
`results/legacy/`. It does not remove generated input data or tracked files.

Commit all tracked benchmark and documentation changes before starting the paper run.
Collection refuses a dirty worktree so the provenance record can identify the code that
produced the results.

Run the benchmarks and compile the paper:

```bash
BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run reproduce-paper
```

The inline environment settings make the command safe to paste into a new shell. The run
includes the AKM, OLS, Correia, PPML, memory, numerical-agreement, setup-cost, and graph-
hardness benchmarks. The hard synthetic cases take several hours, and the Correia data
requires substantial disk space.

Once the Pixi, R, and Julia packages and the Correia files are installed, the benchmark
suite does not need network access.

## Run one stage

```bash
BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run reproduce-results
pixi run render-paper-results
pixi run compile
pixi run verify-paper-results
```

`reproduce-results` runs the benchmarks and writes the paper's table data. The raw timing
CSVs live under `benchmarks/results/` and `results/runs/latest/`; they are intentionally
ignored by Git. `collect` records their paths and SHA-256 hashes, along with the runtime
and package versions, code fingerprint, and imported package locations in
`results/runs/latest/provenance.json`.

`render-paper-results` updates the tracked Typst fragments. `verify-paper-results`
reconstructs the tables from the raw CSVs, checks their recorded hashes and code
fingerprint, resolves every claim-registry source, checks that every generated table is
included by the manuscript, and rejects missing or incomplete trials. `reproduce-paper`
runs the same check before compiling the PDF.

Each timing cell must contain three attempted trials. A normal cell is their median. If
only some trials converge, the table shows the median of those trials and the successful
count, such as `(2/3)`. If none converge, the table records `failed (0/3)`. Both are valid
measured outcomes; an absent or incomplete trial set fails verification. CUDA values are
stored separately in `results/external/cuda.json` because the reference machine cannot
run them.

## Compare results

Coefficient and graph-diagnostic results should agree to their reported precision. Timing
and peak-RSS measurements depend on the machine and current system load. The values in the
paper come from the recorded Apple M4 ten-core run; a run on other hardware can reproduce
the experiment without reproducing the same number of seconds.

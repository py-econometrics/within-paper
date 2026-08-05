# Reproducing the environment

Install the Pixi-managed Python and Typst environment:

```bash
pixi install --locked
```

R and Julia run natively rather than through Pixi. Install them separately, then set the
thread count used by every benchmark process:

```bash
export BENCH_THREADS=10
```

Install the R packages in the system R library:

```r
install.packages("pak", repos = "https://cloud.r-project.org")
pak::pkg_install(c("arrow@24.0.0", "fixest@0.14.2", "jsonlite@2.0.0"))
```

The tracked Julia project and manifest live in `benchmarks/julia-env/`. Instantiate that
environment before running Julia runners:

```bash
julia --project=benchmarks/julia-env -e 'using Pkg; Pkg.instantiate()'
```

The Correia benchmark data are verified downloads and are not stored in Git. Fetch them
once before running the full reproduction:

```bash
pixi run fetch-correia
```

Use `pixi run test` to run the checks and `pixi run compile` to regenerate the Typst
result fragments from `results/paper/benchmark_tables.json` and build the paper. To
regenerate the benchmark results, figures,
tables, and PDF, run:

```bash
pixi run reproduce-paper
```

This command checks the external R and Julia installations before starting the
long-running benchmark suite.

Each public benchmark task names one paper experiment. The task generates its input
directly and writes one raw CSV under `results/runs/latest/`. Cross-language comparisons
share one temporary Parquet file per design. Small Python, R, and Julia calls are part of
`pixi run test`.

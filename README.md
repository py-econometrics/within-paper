# within-paper

Source, benchmarks, and figures for *A Fast Graph-Based Solver for Fixed-Effects
Regressions* (Fischer & Schröder).

## Reproduce the paper

Pixi manages Python and Typst. The benchmarks call native R and Julia installations;
see [REPRODUCING.md](REPRODUCING.md) for the required versions and packages.

```bash
pixi install

export BENCH_THREADS=10
export JULIA_NUM_THREADS=10

pixi run setup-julia-env
pixi run check-external-runtimes
pixi run fetch-correia
pixi run reproduce-paper
```

The thread settings apply to the R and Julia benchmarks. The paper's reference run used
all ten CPU cores of an Apple M4 Mac mini. `check-external-runtimes` prints the thread
counts seen by `fixest` and Julia and stops if they do not match the requested values.

`reproduce-paper` runs the benchmark suite, computes the graph diagnostics, updates the
generated Typst tables and prose values, checks that the required results are present,
and builds `graph_preconditioner_hdfe.pdf`. A full run takes several hours. The benchmark
regressions use one slope covariate (`x1`); their sample sizes and fixed-effect structures
vary by experiment.

The reference machine has no NVIDIA GPU, so the CUDA cells retain measurements from a
separate CUDA run. Every other paper cell is expected to come from the local pipeline;
`verify-paper-results` treats a missing or incomplete local cell as an error. A complete
set of non-converged MAP trials is retained as a measured result.

To move old untracked results out of the active result directories before a new run:

```bash
pixi run archive-legacy-results
```

This leaves the generated input data in place. The Correia CSV files are also local and
ignored by Git. After downloading them once, their checksums can be verified without a
network connection:

```bash
pixi run python scripts/paper_results.py fetch-correia --offline
```

See [REPRODUCING.md](REPRODUCING.md) for package versions, individual benchmark commands,
result locations, and notes on comparing timings across machines.

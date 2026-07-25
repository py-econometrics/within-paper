# within-paper

Source, benchmarks, and figures for *A Fast Graph-Based Solver for Fixed-Effects
Regressions* (Fischer & Schröder).

## Reproduce the paper

Pixi manages Python and Typst. The benchmarks call native R and Julia installations;
see [REPRODUCING.md](REPRODUCING.md) for the required versions and packages.

```bash
pixi install

pixi run setup-julia-env
pixi run check-external-runtimes
pixi run fetch-correia
BENCH_THREADS=10 JULIA_NUM_THREADS=10 pixi run --locked reproduce-paper
```

The thread settings apply to the R and Julia benchmarks. The paper's reference run used
all ten CPU cores of an Apple M4 Mac mini. `check-external-runtimes` prints the thread
counts seen by `fixest` and Julia and stops if they do not match the requested values.

`reproduce-paper` runs the benchmarks, computes the graph diagnostics, regenerates the
Typst tables and numerical values used in the text, verifies the results, and builds
`graph_preconditioner_hdfe.pdf`. A full run takes several hours. Every benchmark
regression has one slope covariate (`x1`). PPML absorbs worker, firm, and year effects;
the other experiments vary the sample size and fixed-effect structure.

The CUDA entries come from an older PyFixest benchmark run because the reference machine
has no NVIDIA GPU. That run did not record its hardware or software setup, so do not
compare these values with the local CPU timings. Every non-CUDA cell must come from a
local run. If all three MAP trials reach the iteration limit, the cell reads
`failed (0/3)`.

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

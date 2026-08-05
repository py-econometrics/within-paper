# Graph-Preconditioned Estimation of High-Dimensional Fixed-Effect Models

This repository contains the paper by Alexander Fischer and Kristof Schröder,
along with the code used to produce its tables, figures, and benchmark results.

The paper explains why the method of alternating projections can be slow in
fixed-effects regressions and introduces a graph-based preconditioner for the
LSMR solver. The benchmarks compare this approach with PyFixest's MAP solver,
R's `fixest`, and Julia's `FixedEffectModels.jl` and
`GLFixedEffectModels.jl`.

## Compile the paper

[Pixi](https://pixi.sh) manages the Python and Typst environment. Install the
locked environment and compile the paper with:

```bash
pixi install --locked
pixi run compile
```

The compiled PDF is written to `graph_preconditioner_hdfe.pdf`. Compilation rebuilds
the paper's tables and measured prose values from the tracked result snapshot; it does
not rerun the benchmarks.

## Reproduce the benchmark results

The full benchmark suite also uses R and Julia. Those runtimes and their
packages must be installed separately. See [REPRODUCING.md](REPRODUCING.md) for
the required versions, thread settings, and commands.

The complete reproduction can take a long time and includes benchmarks with up
to ten million observations. The measurement rules are recorded in
[PROTOCOL.md](PROTOCOL.md), while [benchmarks/README.md](benchmarks/README.md)
describes how the benchmark code is organized.

## Repository guide

- `graph_preconditioner_hdfe.typ` is the paper source.
- `benchmarks/` contains run-local data generators, native runners, explicit
  experiment configs, and the small coordinator.
- `scripts/` turns recorded measurements into paper tables and figures.
- `results/paper/benchmark_tables.json` is the tracked snapshot behind the paper's
  tables and measured prose values.
- `generated/` is untracked build output created from that snapshot during compilation.
- `within-docs/` contains technical notes about the solver.

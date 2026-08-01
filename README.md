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

The compiled PDF is written to `graph_preconditioner_hdfe.pdf`. Compiling the
paper uses the tables and figures already in the repository; it does not rerun
the benchmarks.

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
- `benchmarks/` contains data generators, solver adapters, and benchmark entry
  points.
- `scripts/` turns recorded measurements into paper tables and figures.
- `generated/` contains the table fragments and values included by the paper.
- `within-docs/` contains technical notes about the solver.

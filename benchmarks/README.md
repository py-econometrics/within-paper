# Benchmarks

The benchmark code generates a sample, fits the same model in each language, and writes
raw trial rows. Paper tables and figures are built from those rows afterward.

Set the common thread count before running an experiment:

```sh
export BENCH_THREADS=10
```

Pixi is the public interface. Benchmark scripts do not accept user options.

```sh
pixi run bench-main
pixi run bench-fepois
pixi run bench-akm-sweep
pixi run bench-correia
pixi run agreement
pixi run bench-memory
pixi run within-setup-cost
pixi run within-preconditioners
pixi run bench-scaling
pixi run ppml-inner-outer
pixi run bench-accuracy-frontier
pixi run bench-tolerance
pixi run compute-hardness
```

Each comparison generates one deterministic sample per design. Python uses the frame in
memory. When R or Julia is part of the comparison, the experiment writes that same frame
to a temporary Parquet file and removes it at the end. Backends run one after another.

Raw results live under `results/runs/latest/`. Each experiment writes its complete CSV
once; medians and table labels belong to `scripts/paper_results.py`.

The source tree is organized by model or experiment:

| Path | Purpose |
|---|---|
| `data.py`, `akm.py` | Deterministic base and AKM data |
| `ols/` | Python, R, and Julia OLS fits and OLS experiments |
| `ppml/` | Python, R, and Julia PPML fits |
| `tolerance/` | Shared tolerance measurement and native siblings |
| `within/` | Standalone solver diagnostics |
| `accuracy.py` | External residual and projection-error calculations |
| `memory.py` | Isolated-process memory measurements |
| `runtime.py` | R and Julia process runner |

The timing, accuracy, repetition, and fixed-effect rules remain in `PROTOCOL.md`.

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

Each OLS comparison generates one deterministic sample per design and writes it to a
temporary Parquet file. Every backend loads that same sample in a fresh process. Data
loading is outside the timed region, and ending the process returns its working memory
before the next backend starts. The temporary file is removed after the design finishes.
Each backend uses its default singleton handling. Successful result rows record the
number of observations retained; failed rows record the convergence status and error.
Shared controls, such as the 100-step outer IRLS limit in the PPML benchmark, are stated
explicitly rather than treated as package defaults.

`bench-main`, `bench-akm-sweep`, and `bench-correia` each run the same six
package-default OLS configurations: PyFixest MAP; PyFixest LSMR with no, diagonal, and
factor-pair preconditioning; R `fixest`; and `FixedEffectModels.jl`. Every PyFixest
configuration runs in an isolated Python worker. The LSMR configurations in these three
commands retain their package defaults. The matched-accuracy AKM rows are separate
controls with explicit tolerances and iteration caps.

The OLS agreement check remains a four-backend coefficient diagnostic. PPML, memory,
setup, scaling, reuse, and figures keep their narrower method sets; the PPML table uses
only PyFixest's factor-pair preconditioner reuse policy.

Every measured estimator attempt produces a result row. Iteration limits are marked
`capped`; other estimator errors are marked `converged=false` with their messages kept.
The task then continues with the remaining repetitions, backends, and designs. Ctrl-C
still stops the command normally.

Raw results live under `results/runs/latest/`. Each experiment writes its complete CSV
once; medians and table labels belong to `scripts/paper_results.py`.

`within-setup-cost` retains the 10M simple and difficult endpoint cases and adds the six
1M AKM mobility designs. It times the additive factor-pair solver's construction and
batch solve separately, at the matched-accuracy LSMR settings used in the mechanism
benchmark.

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

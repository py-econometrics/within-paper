# benchmarks/

Five packages, each with one job. The layering runs one way only, and a test in
`tests/test_benchmark_correctness.py` fails the build if it inverts.

| package | holds | may import |
|---|---|---|
| `core/` | paths, timing, result IO, the run record, method names and styles, shared CLI flags, accuracy metrics | nothing else here |
| `dgp/` | the designs benchmarks run on, and one fixed sample per design | `core` |
| `solvers/` | one adapter per thing measured, plus the pinned settings and the registry of arms | `core`, `dgp` |
| `external/` | the R and Julia driver scripts, run as subprocesses | not Python |
| `drivers/` | every entry point, one per `pixi run` task | everything |

`scripts/` sits above all of it and turns recorded measurements into the paper.

Nothing outside `drivers/` may import from it, and every module in it defines
`main()` behind a `__main__` guard. That rule is not stylistic: a driver without
a guard once ran its entire benchmark on import and overwrote a recorded result
file, which is untracked and could not be restored.

## Adding a backend

Three edits. Say you are adding Stata.

1. **Write the driver.** `external/stata_bench.do`, reading the JSON config and
   emitting one JSON record per fit. Copy the contract from
   `external/fixest_bench.R`; `solvers/subprocess_driver.py` is what launches it
   and parses the records back.

2. **Say how to run it.** One `ExternalBackend(...)` entry in
   `solvers/registry.py`. List the models it supports (`feols`, `fepois`, or
   both); the OLS and PPML runners then pick it up automatically.

3. **Say how to name it.** One `Method(...)` in `core/methods.py` giving the
   package, algorithm, preconditioner, colour and marker. Add any spelling the
   result files will use to `ALIASES` in the same file.

Every label, legend entry, table header and figure colour is derived from step
3, so there is no fourth place to update. Two tests hold steps 2 and 3 together:
every runnable backend must have a presentation record, and every backend
spelling appearing in the raw results must resolve to one.

## Running

`pixi task list` shows the individual benchmark commands. The main sweeps are
`bench-main` (OLS), `bench-fepois` (PPML), `bench-akm-sweep`, and
`bench-correia`. For a complete run, use `reproduce-paper`; it runs every
required benchmark, generates the figures, renders the tables, and builds the
PDF.

Set `BENCH_THREADS` before any sweep. The benchmark launcher uses that one value
for R fixest, Julia, and the Rust solver. `check-external-runtimes` verifies the
R and Julia versions and thread settings before a production run.

With `--reuse-existing`, the runner reuses a result only when its backend
settings, benchmark files, and model specification still match. It writes a
small metadata file beside each CSV and reruns stale output automatically.

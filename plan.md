# Implementation-only PR and commit schedule

This local plan covers benchmark code, reproducibility infrastructure, result rendering,
and the mechanical wiring of generated outputs into the paper. It deliberately excludes
all manuscript prose, mathematical exposition, captions, table notes, and editorial
changes. `.gitignore` excludes this working note from paper PRs.

## Scope rules

- Stages 0 through 2 merge sequentially. Stage 3 still lands one PR at a time because
  generated outputs and their Typst wiring touch shared paper files.
- Each commit must pass `pixi run test` and `pixi run compile`.
- Generated tables, figures, and the PDF go in separate commits after their generators.
- Paper edits are limited to replacing an existing hand-written result block with a
  generated `#include`, adding an image reference at an existing figure slot, or wiring
  generated exhibits into the appendix. They do not change surrounding prose, captions,
  or notes.
- PRs 10 and 20 are intentionally outside this stack. They are prose-only work.

## Stage 0

### PR 1: chore: prune stale artifacts and fix the ignore rules

1. `chore: ignore generated benchmark data and external inputs`
2. `chore: remove stale benchmark artifacts and unreferenced figures`

### PR 2: build: pin the reproduction environment

1. `build: pin the Pixi reproduction dependencies`
2. `build: regenerate the Pixi and Julia environment locks`
3. `docs: document the native R and Julia reproduction setup`

Dropping `r-*` Pixi packages is deliberate: R and Julia run natively, and
`check-external-runtimes` verifies those external installations.

## Stage 1

### PR 3: refactor(benchmarks): the modular harness core

1. `refactor(benchmarks): define modular benchmark interfaces`
2. `refactor(benchmarks): add timing repetitions and tests`
3. `refactor(benchmarks): add shared DGP functions and the AKM generator`
4. `refactor(benchmarks): cache and fingerprint benchmark DGPs`
5. `refactor(benchmarks): wire the modular runner`
6. `docs(benchmarks): record the harness protocol`

### PR 4: feat(benchmarks): OLS drivers

1. `feat(benchmarks): add in-process PyFixest OLS backends`
2. `feat(benchmarks): add OLS subprocess backend handshakes`
3. `feat(benchmarks): add the R fixest OLS driver`
4. `feat(benchmarks): add the Julia OLS driver`
5. `feat(benchmarks): register the OLS benchmarker set`
6. `feat(benchmarks): add the OLS and AKM sweep entry points`

### PR 5: feat(benchmarks): PPML drivers

1. `feat(benchmarks): add PyFixest PPML benchmark drivers`
2. `feat(benchmarks): add R and Julia PPML subprocess drivers`
3. `feat(benchmarks): register PPML backends and the three-FE restriction`
4. `feat(benchmarks): add the PPML benchmark entry point`

### PR 6: feat(benchmarks): shared experiment layer, reproducible standalone scripts

1. `feat(benchmarks): add the shared experiment record layer`
2. `fix(benchmarks): make benchmark seeds independent of repetitions`
3. `fix(benchmarks): retain convergence flags in agreement results`
4. `refactor(benchmarks): make standalone diagnostics reproducible`

### PR 7: feat(benchmarks): the Correia collection

1. `docs(data): describe the Correia collection and manifests`
2. `feat(benchmarks): fetch and verify the Correia data collection`
3. `feat(benchmarks): add Correia R and Julia drivers`
4. `feat(benchmarks): add the Correia benchmark entry point`

### PR 8: feat(benchmarks): spectral-gap hardness diagnostics

1. `feat(benchmarks): compute pairwise spectral-gap hardness diagnostics`
2. `fix(benchmarks): prefer PROPACK and handle large sparse graphs`

## Stage 2

### PR 9: feat(pipeline): render and verify the paper's numbers from raw results

1. `feat(pipeline): record run provenance with collect`
2. `feat(pipeline): validate external CUDA results`
3. `feat(pipeline): define table documents and claim registry schema`
4. `feat(pipeline): render OLS, PPML, memory, and agreement tables`
5. `build(paper): wire the four generated result tables into existing slots`
6. `feat(pipeline): render AKM mobility and sorting tables`
7. `build(paper): wire the AKM tables into existing slots`
8. `feat(pipeline): render the Correia tables`
9. `build(paper): wire the Correia tables into existing slots`
10. `feat(pipeline): generate manuscript value definitions`
11. `feat(pipeline): verify tables against raw results`
12. `build(paper): rebuild the PDF`

Every table addition must include its driver, `_synchronize_*` method,
`benchmark_tables.json` entry, `claim_registry.json` entry, generated Typst table, and
manuscript `#include` in the same PR.

## Stage 3

### PR 11: feat: the preconditioner ablation and mechanism tables

1. `feat(benchmarks): record off, diagonal, and additive arms`
2. `feat(benchmarks): add MAP diagnostics and pilot calibration`
3. `feat(pipeline): render the mechanism tables`
4. `build(paper): wire mechanism tables into existing slots`
5. `build(paper): rebuild the PDF`

### PR 12: feat: paper figures

1. `feat(figures): define the shared paper figure style`
2. `feat(figures): analyse spectral gap and runtime`
3. `feat(figures): generate the paper figure specifications`
4. `results(figures): regenerate the paper SVGs`
5. `build(paper): wire figures into existing slots`
6. `build(paper): rebuild the PDF`

### PR 13: feat: the time-versus-accuracy frontier

1. `feat(benchmarks): sweep native tolerances for the accuracy frontier`
2. `feat(pipeline): render the accuracy-frontier table`
3. `build(paper): wire the accuracy-frontier table into an existing slot`
4. `build(paper): rebuild the PDF`

### PR 14: feat: PPML inner versus outer convergence

1. `feat(benchmarks): record PPML inner and outer convergence`
2. `feat(pipeline): render the PPML inner-outer table`
3. `build(paper): wire the PPML inner-outer table into an existing slot`
4. `build(paper): rebuild the PDF`

### PR 15: feat: factor-count scaling and amortization

1. `feat(benchmarks): measure factor-count scaling and amortization`
2. `feat(pipeline): render scaling and amortization tables`
3. `build(paper): wire the scaling tables into existing slots`
4. `build(paper): rebuild the PDF`

### PR 16: feat: standalone within preconditioner diagnostics

1. `feat(benchmarks): record standalone within preconditioner diagnostics`
2. `feat(pipeline): render setup-cost diagnostics`
3. `build(paper): wire setup-cost output into an existing slot`
4. `build(paper): rebuild the PDF`

### PR 17: feat: iteration counts in solver-native units

1. `feat(benchmarks): record solver-native iteration counts`
2. `feat(pipeline): render the iteration table`
3. `build(paper): wire the iteration table into an existing slot`
4. `build(paper): rebuild the PDF`

### PR 18: feat: the AKM tolerance frontier

1. `feat(benchmarks): add AKM tolerance frontier drivers`
2. `feat(figures): plot the tolerance frontier`
3. `fix(figures): make the tolerance figure legible at text width`
4. `results(figures): render the tolerance frontier SVG`
5. `build(paper): wire the tolerance figure into an existing slot`
6. `build(paper): rebuild the PDF`

### PR 19: refactor: one public name per method configuration

1. `feat(figures): add shared method colours, markers, and line styles`
2. `refactor(pipeline): render tables with public method names`
3. `refactor(figures): label figures and legends with public method names`
4. `results: regenerate tables and figures with public method names`
5. `build(paper): rebuild the PDF`

### PR 21: build: rerun the production benchmarks and record provenance

1. `build(benchmarks): rerun production benchmarks on the idle machine`
2. `results: refresh recorded tables and figures`
3. `build(pipeline): run strict result verification`
4. `build(paper): wire generated tables and figures into the appendix`
5. `build(paper): rebuild the production PDF`

Commit 4 is the final reviewable source commit. It adds only generated table includes and
figure references at the appendix anchor; it does not add explanatory copy, captions, or
table notes. Commit 5 remains PDF-only.

## Deferred manuscript-only work

These remain separate from the implementation stack:

- PR 10: `docs(paper): the mathematical contract`
- PR 20: `docs(paper): corrections and prose`
- All new explanatory sections, captions, table notes, limitations, software examples,
  decision guidance, and editorial passes associated with the experimental PRs.

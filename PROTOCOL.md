# Measurement protocol

This file fixes the measurement rules for the paper's empirical claims before the
production runs, so that no rule is chosen after seeing a result. It covers what each
claim rests on, how samples and specifications are held fixed, where timing starts and
stops, how many repetitions each cell needs, and which accuracy gate a headline number
must clear.

[REPRODUCING.md](REPRODUCING.md) says how to run the benchmarks. This file says what
counts as a valid measurement.

## 1. Claim ledger

Every claim the abstract or conclusion makes must appear here with the experiment that
supports it, the metric that decides it, and the accuracy condition under which the
metric is read. A claim with no row is not a claim the paper may make.

| # | Claim | Experiment | Deciding metric | Accuracy gate |
|---|---|---|---|---|
| 1 | On near-nested and weakly connected designs, factor-pair preconditioning cuts total runtime by one to two orders of magnitude against MAP-based implementations | 10M simple/difficult; AKM mobility designs at 1M | Median total wall time, all backends | Gate A on every reported cell |
| 2 | Weak connectivity can make factor-by-factor MAP converge slowly | AKM designs varying mobility and sorting | MAP sweeps and runtime against the worker-firm gap | Gate A, or explicit censoring |
| 3 | Switching to unpreconditioned LSMR does not by itself remove the slow directions | Matched-accuracy AKM runs and the iteration-count benchmark | LSMR iterations and runtime for `off` vs `additive` | Gate A on all four arms |
| 4 | Diagonal scaling removes them only partially | Matched-accuracy AKM runs and the iteration-count benchmark | LSMR iterations, `diagonal` vs `additive` | Gate A |
| 5 | Factor-pair preconditioning reduces LSMR iterations relative to diagonal preconditioning where pair coupling matters | Simple and difficult iteration-count benchmark | Median iterations for `diagonal` vs `additive` | Gate A |
| 6 | Factor-pair preconditioning reduces total runtime only when iteration savings exceed setup and application cost | AKM setup-cost and ten-regression reuse experiments | Setup, solve, and total time | Gate A |
| 7 | Setup is most expensive on the dense graphs that need it least | Additive setup cost across AKM mobility designs | Construction time against the worker-firm gap | Gate A on the paired solve |
| 8 | `within` runtime varies little and declines modestly as connectivity weakens | AKM designs varying mobility | Median and IQR of total time across the designs | Gate A; repetition rule R1 |
| 9 | Setup amortizes across repeated fits with unchanged weights; PPML is a separate repeated-solve use case in which the weights change between IRLS steps | Ten-regression experiment on the simple and difficult designs; main PPML benchmark | Setup and solve time for repeated OLS fits; total PPML runtime | Gate A for every reported cell |
| 10 | The method loses on well-connected designs at scale | 10M simple design | Total runtime | Gate A |

Claims 3 through 7 cover the mechanism. Claims 1, 8, and 10 are the headline results.

Any claim that cannot clear its gate is either dropped or reported with the failure
stated in the same sentence. "Capped", "did not converge", and "did not reach the gate"
are results, not omissions.

## 2. Sample and specification

Every backend receives the same input data and regression specification. Cross-package
tables retain package-specific behavior except where a shared control is stated
explicitly. Single-package mechanism exercises hold the estimation path fixed when they
isolate a solver choice.

- **Retained sample.** Backends receive the same input file. Cross-package OLS and PPML
  benchmarks use each package's default singleton and separation handling. Every row
  records whether the fit converged; successful rows also record the retained-observation
  count. These timings use default sample handling rather than a forced common estimation
  sample. Matched-solver exercises run through one package path on the same prepared sample.
- **PPML outer iterations.** PyFixest, R `fixest`, and `GLFixedEffectModels.jl` each
  receive an outer IRLS limit of 100 iterations. This common cap replaces their package
  defaults; their separation handling and other solver settings remain unchanged.
- **Weights.** Unweighted (`W = I`) in every benchmark. Weighted solves appear only
  inside PPML, where IRLS sets them.
- **Covariates.** Every regression has one slope covariate, `x1`.
- **Factor order.** Worker, firm, year, in that order, for every experiment. MAP is
  sensitive to the cycling order, so it must not vary across tables.
- **Threads.** `BENCH_THREADS=10` on the ten-core reference machine. The benchmark
  launcher applies that value to R fixest, Julia, and the Rust solver's Rayon pool.
  `check-external-runtimes` verifies the R and Julia settings before a production run.
- **Data.** One deterministic dataset per design, generated once and reused for every
  repetition. Regenerating the sample per repetition confounds solver variance with DGP
  variance; DGP replication is a separate robustness exercise with its own rows.

## 3. Timing boundaries

Timing starts after the data are in memory in the backend's native frame and stops after
the fit returns and its convergence flag has been read.

Inside the boundary: model setup, factor encoding, singleton handling, construction of
the fixed-effect representation, preconditioner setup, residualization, and the
low-dimensional regression.

Outside the boundary: file reading and parsing, format conversion, interpreter or JIT
startup, and package loading.

Each backend makes one unreported warm-up fit on the same generated sample before its
timed trials, so lazy initialization is not charged to the first recorded call.

The setup-cost and reuse experiments decompose the measured solver time as

```
T_total = T_preconditioner + T_solve
```

Package-level regression tables report the full fit time. The separate decomposition
measures the setup and solve terms used in claims 6 and 7.

## 4. Repetitions and reporting

Repetition counts scale with the cell's runtime, because the flatness claim (claim 8)
turns on separating timings that differ by tens of milliseconds.

| Rule | Cell runtime | Timed repetitions |
|---|---|---|
| R1 | Under 1 second | 20 |
| R2 | 1 to 10 seconds | 7 |
| R3 | Over 10 seconds | 3 |

Additional rules, all mandatory:

- One discarded burn-in trial precedes the timed ones, per backend per design.
- Backends run sequentially in the fixed order shown by the experiment script. Production
  runs use an otherwise idle machine.
- The canonical paper tables report medians. Raw trial rows retain the distribution for
  any claim that needs a spread.
- Failed and capped trials stay in the record. They are never dropped before the median.
- Every measured estimator call writes a row. An iteration limit sets `capped=true`;
  another estimator exception sets `converged=false`, keeps the error message, and does
  not stop the remaining repetitions or backends. User interrupts still stop the task.
- Every cell reports its converged count beside the timing, and the rule appears in the
  table note rather than only in this file.

"Median among converged trials" is a selected estimator and must be labelled as one
wherever it is used.

The 10M main OLS comparison uses the adaptive R1/R2/R3 rule. The AKM and Correia OLS
comparisons use three planned calls per cell. The PPML comparison also uses three calls;
the AKM setup-cost experiment uses five, the ten-regression reuse experiment uses three,
and memory and coefficient-agreement diagnostics use one isolated call per cell.

## 5. Accuracy metrics

Three distinct quantities, never referred to by the same word.

For right-hand side `mu_j`, with `A = W^(1/2) D`, `b_j = W^(1/2) mu_j`, and
`e_j = b_j - A alpha_j`:

1. **Internal solver residual.** Whatever the backend reports. Not comparable across
   packages, recorded for completeness only.
2. **External normal-equation residual.**

   ```
   eta_j = ||A' e_j|| / max( ||A' b_j||, eps * ||A||_F * ||b_j|| )
   ```

   Computed independently of the solver, through weighted fixed-effect group sums, so it
   measures the returned answer rather than the solver's own bookkeeping.
3. **Projection error against a tight reference.**

   ```
   delta_j = ||e_j - e_j_star|| / max( ||b_j||, eps )
   ```

Slope agreement is reported as `|beta_hat - beta_hat_star| / SE(beta_hat_star)`, in units
of reference standard errors.

Raw fixed-effect coefficients are never compared: they depend on the normalization, and
Section 2 of the paper takes them to be defined only up to `ker(D)`. Comparisons use
fitted values, demeaned variables, slopes, scores, and objectives.

### Gate A (frozen 2026-07-26)

- `max_j eta_j <= 1e-8`
- `max_j delta_j <= 1e-7`
- slope difference below `1e-4` reference standard errors

All three must be measured before a result is said to clear Gate A. The standalone
preconditioner benchmark measures `eta` and `delta`, but not the slope difference, so it
does not report a full Gate A result.

The thresholds were frozen from a one-time calibration on the 100K simple and difficult
designs. Focused tests retain the dense-residual, projection, and cap-reporting checks.

**A nominal tolerance is not an achieved accuracy.** The LSMR stopping rule bounds a
relative normal-equation residual recovered from the bidiagonalization scalars, which is
a different number from the externally recomputed `eta`. At the package default of
`1e-8`, no configuration clears Gate A on either design:

| Design | `off` | `diagonal` | `additive` |
|---|---|---|---|
| simple, achieved `eta` at tol `1e-8` | 3.7e-07 | 2.6e-08 | 2.4e-08 |
| difficult, achieved `eta` at tol `1e-8` | 1.6e-06 | 1.2e-06 | 4.1e-07 |

`1e-12` is the loosest tolerance at which all three clear Gate A on both designs, so:

- **Matched-accuracy arms** run `rust-map` at `1e-10` (`MECHANISM_MAP_TOL`) and `off`,
  `diagonal`, `additive` at `1e-12` (`MECHANISM_LSMR_TOL`), so runtimes are compared at
  matched achieved accuracy rather than matched nominal tolerance. They are measured in
  the same pass as the package-default arms and carry distinct labels; which rows feed
  which table is decided when the results are curated, not by running the designs twice.
- **All four matched arms share one iteration budget** of 10,000 (`MECHANISM_MAXITER`).
  The package defaults give MAP 10,000 and LSMR 1,000, and the first 1M run showed what
  that asymmetry does: `within-off` failed 30 of 33 trials, every one at its own lower
  cap, against a MAP arm allowed ten times as many iterations. A censoring produced by
  the budget cannot support a claim about the preconditioner, so a run that still fails
  now fails on its own merits. The package-default arms keep each package's documented
  settings, including LSMR's 1,000.
- **Cross-package tables** keep each package's documented default and annotate every cell
  with its achieved `eta`. A default-settings cell is not expected to clear Gate A; it is
  expected to report what it did achieve.

Every design named in a headline sentence carries an accuracy record on the sample that
produced the timing. Not at a smaller size, and not on a different draw. Each timed
`feols` fit records `max_eta`, recomputed from the fit's own demeaned arrays and the
input rows the model kept, outside the timing boundary. The projection error `delta`
still requires a tight reference per sample and is recorded only by the standalone
diagnostics, so a cross-package cell reports `eta` and not the full Gate A triple.

### Metric validation

`eta` is validated against a dense minimum-norm solve on a design small enough to form
`D` explicitly. On the direct solution the helper reports `eta = 3.6e-15`, it agrees with
a dense recomputation of `||D'e|| / ||D'b||` to `1.6e-30`, and a `1e-3` perturbation in
coefficient space raises it to `1.1e-02`. It measures the returned answer, not the
solver's own bookkeeping.

The three preconditioners reach the same observation-space projection: at tolerance
`1e-14` the projection error against the additive solution is `5.1e-13` or smaller on the
simple design and `4.5e-14` or smaller on the difficult one. The ablation compares
methods that agree on the answer.

### Cap reporting

Verified at every level that can cap:

- LSMR returns `converged=False` with `iterations` equal to the cap.
- The MAP diagnostics return `censoring="capped"` with `iterations` equal to the cap.
- PyFixest raises `ValueError: Demeaning failed after N iterations.`, which the harness
  records as `converged=False` with the message retained.
- R and Julia warnings or returned convergence flags are converted to the same row
  fields. If an isolated estimator process exits before writing its rows, the parent
  writes one failed row for each planned repetition.

No path silently reports a capped run as converged.

### Reference construction

- **Small samples (100K).** An anchored direct or QR solve on the normal equations,
  built independently of the iterative solvers.
- **Large samples (1M, 10M).** Two independent tight-tolerance procedures that agree to
  within Gate A. Agreement between them is the check; neither alone is the reference.
- Default `rust-map` is never used as ground truth. It is one of the methods under test,
  and it reaches its iteration cap on exactly the designs where the reference matters
  most.

## 6. Cross-package comparison

Three comparisons are kept separate.

1. **Package defaults.** The five OLS package-runtime tables include PyFixest MAP,
   PyFixest LSMR with no, diagonal, and factor-pair preconditioning, R `fixest`, and
   `FixedEffectModels.jl`. The three PyFixest LSMR configurations keep their documented
   default tolerance and iteration cap. PPML keeps only the factor-pair reuse path.
2. **Matched PyFixest runs.** MAP and the three LSMR preconditioners use explicit
   tolerances and a common iteration budget on the AKM mobility and sorting designs.
3. **Runtime against achieved precision.** Each backend is tested at its own tolerance
   settings on three AKM mobility designs. The figure plots wall time against coefficient
   and residual error measured on the returned fit.

Matched accuracy is restricted to the PyFixest methods, which can be assessed with the
same external metric. `fixest` does not monitor `eta`; changing its stopping criterion
until `eta` crosses a chosen threshold would not match the criterion the package uses.

The package-runtime tables display all three default PyFixest LSMR configurations rather
than selecting one after seeing the results. The matched controls appear only in the
headline figure and retain their explicit tolerances and iteration budget.

Iteration counts are never placed on a shared axis. A MAP sweep, a `fixest` fixed-point
iteration, and an LSMR iteration are different units and get separate panels.

### Per-result record

Every headline package result records: package and runtime versions; algorithm settings,
tolerance, and iteration cap; thread counts; factor ordering, singleton treatment,
weights, and retained sample size; which phases are inside the timing boundary; completed
trials, convergence failures, and capped times; and the external residual and slope
accuracy on that exact sample.

## 7. Experiment matrix

| Experiment | Scale | Methods | Purpose |
|---|---|---|---|
| Correctness pilot | 100K simple/difficult | MAP + three within configurations | Validate metrics against a direct reference; freeze Gate A |
| AKM setup cost | 1M mobility designs | Factor-pair LSMR with two and three fixed effects | Construction and solve time across connectivity |
| AKM designs varying mobility | 1M | Six default OLS configurations; matched MAP + three LSMR configurations | Package runtime and mechanism comparison |
| AKM designs varying sorting | 1M | Six default OLS configurations; matched MAP + three LSMR configurations | Package runtime and mechanism comparison |
| Large-scale simple/difficult | 10M | Six default OLS configurations | Headline total-runtime comparison |
| Runtime and achieved precision | 1M selected AKM mobility designs | PyFixest MAP and LSMR variants, R `fixest`, and `FixedEffectModels.jl` | Wall time against coefficient and residual error |
| Ten-regression reuse | 1M simple and difficult | Diagonal, rebuilt factor-pair, and cached factor-pair preconditioners | Setup reuse across fits |
| Selected Correia designs | Existing sizes | Six default OLS configurations | Robustness across graph families |
| PPML simple/difficult | 1M | PyFixest MAP and factor-pair reuse, plus external packages | Inner solver versus outer IRLS convergence |

Mobility is the main mechanism experiment because it varies the economic source of
computational difficulty while holding the rest of the DGP fixed. The sorting designs
provide a second comparison.

## 8. Open items

Resolved before the production runs, and struck from this list when done:

- [x] Validate Gate A on the 100K pilot and freeze the thresholds. Done 2026-07-26;
      the numerical checks now live in the test suite rather than a public benchmark.
- [x] Verify that all three within configurations reach the same projection. Done.
- [x] Test MAP and LSMR failure and cap reporting. Done.
- [x] Reconcile the factor order between the OLS/PPML specs and the AKM mobility spec. Done
      2026-07-26: all three now absorb `indiv_id, firm_id, year`.
- [x] Retained sample size. Every final Python, R, and Julia row records `n_retained`.
      Cross-package tables retain each package's default sample handling, and the
      recorded counts document any resulting difference.
- [x] One shared experiment layer. The OLS runner creates one temporary Parquet sample
      per design. Every backend loads that sample in a fresh process, so process-local
      memory is returned before the next cell starts. Seeds belong to designs rather
      than repetitions, and each result file is written once after the complete run.
- [x] Wire the R1/R2/R3 repetition counts into the harness. Done. The PyFixest
      benchmarkers time one fit, choose the count from its runtime, and repeat on the
      same backend-native frame. Final rows keep `repetition` and `n_planned`, and the
      renderer checks a cell against that recorded plan rather than assuming three trials.
- [x] Reduce the DGP replicate count for timing. Each timing runner now generates one
      sample per design and repeats the fit on that sample. DGP replication remains a
      separate robustness exercise.
- [x] Keep the PPML benchmark on package behavior. PyFixest `fepois` reuses the first
      factor-pair preconditioner as the IRLS weights change. The paper reports this path
      in the main PPML table and does not retain a separate rebuild treatment.
- [x] Report iterations in solver-specific units. Done 2026-07-28.
      `pixi run within-preconditioners` runs counting MAP beside the three LSMR
      configurations on one sample. The renderer keeps MAP sweeps and LSMR iterations
      in separate columns and marks capped cells.

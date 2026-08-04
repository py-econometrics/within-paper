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
| 1 | On near-nested and weakly connected designs, factor-pair preconditioning cuts total runtime by one to two orders of magnitude against MAP-based implementations | 10M simple/difficult; AKM mobility sweep at 1M | Median total wall time, all backends | Gate A on every reported cell |
| 2 | Weak connectivity can make factor-by-factor MAP converge slowly | AKM mobility and sorting sweeps | MAP sweeps and runtime against the worker-firm gap | Gate A, or explicit censoring |
| 3 | Switching to unpreconditioned LSMR does not by itself remove the slow directions | Matched-accuracy arms of the AKM sweep | LSMR iterations and runtime for `off` vs `additive` | Gate A on all four arms |
| 4 | Diagonal scaling removes them only partially | Matched-accuracy arms | LSMR iterations, `diagonal` vs `additive` | Gate A |
| 5 | Factor-pair preconditioning reduces LSMR iterations relative to diagonal preconditioning where pair coupling matters | Matched-accuracy arms | Median, max, and sum of per-RHS iterations | Gate A |
| 6 | It reduces total runtime only when iteration savings exceed setup and application cost | Additive setup/solve split on the AKM mobility sweep and the 10M endpoints | `T_setup` vs `T_solve` vs `T_total` | Gate A |
| 7 | Setup is most expensive on the dense graphs that need it least | Standalone additive diagnostics over the AKM mobility sweep | Factorization time against graph density | Gate A on the paired solve |
| 8 | `within` runtime is close to invariant to connectivity, and mildly decreasing in it | AKM mobility sweep | Median and IQR of total time across the sweep | Gate A; repetition rule R1 |
| 9 | Setup amortizes across right-hand sides, repeated fits, and IRLS steps; headline timings do not amortize it | Multi-RHS experiment, K in {1,2,5,10,25} | Total time against K, measured break-even | Gate A at each K |
| 10 | Pairwise spectral gaps describe difficult designs but are not a solver-selection rule | Pooled gap-versus-runtime analysis over all collected designs | Slope of log runtime on log gap by backend, plus named counter-examples | Uses existing recorded runtimes |
| 11 | The method loses on well-connected designs at scale | 10M simple design | Total time and setup share | Gate A |
| 12 | Scaling in the number of absorbed factors Q | Q in {2,3,4,5} at fixed n | Setup, solve, iterations, accuracy | Gate A |

Claims 3 through 7 are the mechanism section. Claims 1, 8, and 11 are the headline. Claim
10 is the diagnostic caveat.

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
- **Covariates.** One slope covariate `x1`, except in the multi-RHS amortization
  experiment, which is the only place K > 1.
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

Each subprocess backend warms up on a burn-in dataset that is discarded, so JIT
compilation and lazy initialization are not charged to the first timed trial.

For every within configuration the total is decomposed as

```
T_total = T_common_setup + T_preconditioner + T_solve + T_low_dimensional_regression
```

and all four parts are recorded. Reporting only the total hides the trade-off that
claims 6 and 7 are about.

## 4. Repetitions and reporting

Repetition counts scale with the cell's runtime, because the flatness claim (claim 8)
turns on separating timings that differ by tens of milliseconds.

| Rule | Cell runtime | Timed repetitions |
|---|---|---|
| R1 | Under 1 second | 20 to 30 |
| R2 | 1 to 10 seconds | 7 to 10 |
| R3 | Over 10 seconds | 3 to 5 |

Additional rules, all mandatory:

- One discarded burn-in trial precedes the timed ones, per backend per design.
- Backends run sequentially in the fixed order shown by the experiment script. Production
  runs use an otherwise idle machine.
- Report the median and the interquartile range. A median without a spread cannot
  support an invariance claim.
- Failed and capped trials stay in the record. They are never dropped before the median.
- Every measured estimator call writes a row. An iteration limit sets `capped=true`;
  another estimator exception sets `converged=false`, keeps the error message, and does
  not stop the remaining repetitions or backends. User interrupts still stop the task.
- Every cell reports its converged count beside the timing, and the rule appears in the
  table note rather than only in this file.

"Median among converged trials" is a selected estimator and must be labelled as one
wherever it is used.

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

Two views, kept separate.

1. **Package defaults**, each annotated with its achieved `eta`, so a reader sees the
   accuracy that each runtime bought.
2. **A time-versus-accuracy frontier** for one easy and one hard design: each package
   swept over roughly four of its own tolerance settings, plotting wall time against
   achieved `eta`.

A single matched-accuracy table is not used. `fixest` does not monitor `eta`, so forcing
it to a threshold built around the proposed method means driving an unrelated criterion
until an untargeted metric happens to clear a bar, and some packages will not clear it at
any setting.

Only the documented default within configuration appears in cross-package tables. The
three preconditioner variants belong to the mechanism section; selecting the fastest one
per cell after the fact would not be a package comparison.

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
| Standalone setup diagnostic | 1M AKM mobility designs; 10M base endpoints | `within-additive` | Setup/solve split, iterations, and external accuracy across connectivity |
| AKM mobility sweep | 1M | MAP + three within configurations | Main mechanism experiment |
| AKM sorting sweep | 1M | MAP + three within configurations | Corroborating connectivity experiment |
| Accelerated-MAP check | 1M mobility sweep | R `fixest` plus selected same-package results | Whether acceleration mitigates the MAP slowdown |
| Large-scale simple/difficult | 10M | Same-package four-way plus external packages | Headline total-runtime comparison |
| Accuracy frontier | 10M simple and difficult | All external packages | Time versus achieved accuracy |
| Factor scaling | 1M, Q in {2,3,4,5} | `within-additive`, plus MAP for reference | Structural weak point of the pair construction |
| Amortization | 1M easy and hard | Primarily `diagonal` and `additive` | Setup reuse across right-hand sides and fits |
| Selected Correia designs | Existing sizes | External packages plus `within-additive` | Robustness across graph families |
| PPML simple/difficult | 1M | Three within configurations plus external packages | Inner solver versus outer IRLS convergence |
| PPML reuse diagnostic | 100K | Exact PyFixest `fepois`, cache retained or cleared | Inner tolerance versus outer convergence |

The mobility experiment carries the mechanism weight, because it varies the economic
source of computational difficulty while holding the rest of the DGP fixed. Sorting
corroborates it.

## 8. Open items

Resolved before the production runs, and struck from this list when done:

- [x] Validate Gate A on the 100K pilot and freeze the thresholds. Done 2026-07-26;
      the numerical checks now live in the test suite rather than a public benchmark.
- [x] Verify that all three within configurations reach the same projection. Done.
- [x] Test MAP and LSMR failure and cap reporting. Done.
- [x] Reconcile the factor order between the OLS/PPML specs and the AKM sweep spec. Done
      2026-07-26: all three now absorb `indiv_id, firm_id, year`. **The existing OLS and
      PPML tables were produced under the old order and must be regenerated.**
- [x] Retained sample size. Every final Python, R, and Julia row records `n_retained`.
      Cross-package tables retain each package's default sample handling, and the
      recorded counts document any resulting difference.
- [x] One shared experiment layer. The OLS runner creates one temporary Parquet sample
      per design. Every backend loads that sample in a fresh process, so process-local
      memory is returned before the next cell starts. Seeds belong to designs rather
      than repetitions, and each result file is written once after the complete run.
- [ ] Run the standalone setup diagnostic on the AKM mobility designs. The runner records
      additive setup, solve, and accuracy separately on all six designs; its production
      output still needs to be collected.
- [x] Wire the R1/R2/R3 repetition counts into the harness. Done. The PyFixest
      benchmarkers time one fit, choose the count from its runtime, and repeat on the
      same backend-native frame. Final rows keep `repetition` and `n_planned`, and the
      renderer checks a cell against that recorded plan rather than assuming three trials.
- [x] Reduce the DGP replicate count for timing. Each timing runner now generates one
      sample per design and repeats the fit on that sample. DGP replication remains a
      separate robustness exercise.
- [x] Define the PPML cache comparison on the package implementation. Both cells call
      PyFixest `fepois`. Reuse leaves the package untouched; rebuild changes only
      `_seed_preconditioner` so no factorization is retained between weighted demeaning
      calls. A regression test requires both policies to retain the same sample and agree
      on the coefficient and deviance.
- [ ] Measure PPML reuse under both inner-solver regimes. The runner now discards one
      burn-in and records seven fits per cell. Regenerate the table before making a
      runtime claim from this diagnostic. PyFixest does not expose individual inner
      LSMR iteration counts.
- [x] Report iterations in solver-specific units. Done 2026-07-28.
      `pixi run within-preconditioners` runs counting MAP beside the three LSMR
      configurations on one sample. The renderer keeps MAP sweeps and LSMR iterations
      in separate columns and marks capped cells.

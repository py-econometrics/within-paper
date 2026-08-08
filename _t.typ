#set document(
  title: "Graph-Preconditioned Fixed-Effect Absorption for Digital Experiments",
  author: "Alexander Fischer and Kristof Schroeder",
)
#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.62in),
  numbering: "1",
  number-align: center,
)
#set text(font: "Libertinus Serif", size: 9.1pt)
#set par(justify: true, leading: 0.96em, spacing: 0.68em)
#set heading(numbering: "1.")
#set math.equation(numbering: none)
#show heading.where(level: 1): it => {
  set block(above: 0.8em, below: 0.32em)
  text(size: 11.6pt, weight: "bold", it)
}
#show heading.where(level: 2): it => {
  set block(above: 0.52em, below: 0.18em)
  text(size: 9.8pt, weight: "bold", it)
}
#show figure.caption: set text(size: 7.8pt)
#let rule = rgb("#718096")
#let light-rule = rgb("#d7dee8")
#let head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: head-fill)[#strong(body)]

#align(center)[
  #set par(justify: false)
  #text(size: 15.5pt, weight: "bold")[Graph-Preconditioned Fixed-Effect Absorption]
  #v(0.16em)
  #text(size: 11pt)[for Digital Experiments]
  #v(0.4em)
  #text(size: 9.1pt)[Alexander Fischer#footnote[trivago] and Kristof Schröder#footnote[appliedAI Institute for Europe gGmbH]]
  #v(0.16em)
  #text(size: 8pt)[Extended abstract for CODE\@MIT 2026]
]

#v(0.3em)
#align(center)[
  #block(
    width: 96%,
    inset: (x: 0.8em, y: 0.55em),
    fill: rgb("#f7f8fa"),
    stroke: 0.3pt + light-rule,
    radius: 3pt,
  )[
    #text(size: 8.5pt)[
      #text(weight: "bold")[Abstract.] Analysts of platform experiments absorb fixed effects for users, items,
      markets, and time periods to remove variation that would otherwise enter the standard error on the
      treatment coefficient. The Method of Alternating Projections (MAP) performs this absorption by
      demeaning one factor at a time, and it converges slowly when the co-occurrence graph linking the
      absorbed factors is sparse or close to disconnected. We show that several standard features of
      platform experiments produce such graphs: sparse two-sided exposure in marketplaces, and cluster
      randomization that partitions an interaction graph to limit interference. We propose a preconditioner
      built from overlapping factor-pair subproblems. Each subproblem is a signed graph Laplacian and admits
      sparse approximate Cholesky factorization, and we use the resulting operator inside a Krylov solver
      for least squares. On a near-nested ten-million-observation design, the preconditioned solver
      completes in 4.10s against 62.3s for the fastest MAP implementation; for Poisson models on the same
      design the times are 5.42s and 439.4s. On dense, well-connected graphs MAP remains faster, and we
      report the spectral diagnostic that separates the two regimes. Because the preconditioner depends on
      the fixed effects and the weights but not on the right-hand side, one factorization serves the many
      residualizations that randomization inference and multi-metric readouts require: measured cost per
      right-hand side falls from 0.137s to 0.047s between one and twenty-five columns.
    ]
]
]

#align(center)[
  #text(size: 8pt)[#strong[Keywords:] digital experiments; high-dimensional fixed effects; alternating projections; preconditioning; variance reduction]
]

= Fixed effects as a variance-reduction device

A single experiment readout may record outcomes for users across items, markets, treatment cohorts, and
calendar periods. Absorbing these dimensions removes persistent differences in composition and exposure,
and it serves the same purpose as the pre-experiment covariate adjustment of @deng2013, with the
difference that the adjustment is non-parametric in the absorbed dimensions. The resulting regression has
a handful of coefficients of interest and millions of nuisance fixed-effect coefficients. How long it
takes to remove those nuisance coefficients becomes part of the experiment's operating cost, and in the
hardest cases it decides whether a specification is estimated at all.

The Frisch--Waugh--Lovell theorem reduces the problem to residualizing the outcome and the covariates
against the fixed effects before estimating the coefficients of interest @frisch1933 @lovell1963. For a
right-hand side $mu$, let $D$ be the sparse fixed-effect design matrix and $W$ a diagonal matrix of
observation weights. The residualization problem is

$
  hat(alpha)_mu = arg min_alpha || W^(1/2) (mu - D alpha) ||_2^2,
  quad G hat(alpha)_mu = D' W mu,
  quad G = D' W D.
$

The same matrix $G$ serves the outcome and every covariate, so a solver can amortize its setup cost across
right-hand sides. Our contribution is a preconditioner that uses the cross-factor structure of $G$. MAP
uses only the diagonal blocks of $G$ and passes cross-factor information through repeated residual
updates. We instead solve overlapping pairwise blocks approximately and combine their corrections with an
additive Schwarz construction, so that the co-occurrence structure enters the solver directly.

= Where hard co-occurrence graphs arise in experiments

Absorption cost depends on the shape of the co-occurrence graph rather than on the number of observations
alone. Two settings common in platform experimentation generate the sparse and near-disconnected graphs on
which MAP is slow, and a third makes the setup cost of any reusable factorization negligible.

== Two-sided exposure in marketplace experiments

A marketplace experiment records outcomes at the level of a pair: a traveller and a hotel, a rider and a
driver, a buyer and a listing. Absorbing user and item effects removes persistent differences in
propensity to convert and in item quality. The cross-tabulation $C_(U I)$ counts how often each user is
observed with each item, and it is the weighted adjacency matrix of a bipartite graph. The identical
object appears in matched employer--employee data as the worker--firm cross-tabulation, which is why the
benchmarks below use designs from that literature: they are public, standard, and parameterized by the
graph feature we care about.

The two settings differ in density. A worker in an administrative panel is observed at a small number of
firms over a career, whereas a user on a platform interacts with a handful of items drawn from a
catalogue of millions, and a readout window of two weeks leaves most users observed once. Users who
appear with several items act as movers, and they are the only observations that connect item effects to
one another. When such users are scarce, user and item effects are nearly confounded within small
subgraphs, and each MAP sweep carries little information across the graph.

== Cluster randomization under interference

Where outcomes spill over between units, experimenters partition the interaction graph into clusters and
randomize at the cluster level @ugander2013 @eckles2017. The design problem is to choose a partition that
cuts as little edge weight as possible, since edges crossing cluster boundaries carry the contamination
that biases the estimate.

The quantity the designer minimizes and the quantity that governs MAP convergence are computed from the
same weighted graph. A partition that cuts little edge weight leaves the corresponding factor-pair block
close to block diagonal with thin bridges between the blocks, and the spectral gap defined in Section 3 is
then small. A cluster design that succeeds in limiting spillovers therefore hands the solver the graph on
which MAP converges most slowly. We report this connection as a structural one; we have not benchmarked a
cluster-randomized design directly, and the strength of the effect will depend on how the analysis panel
is constructed.

== Repeated residualizations against one fixed-effect structure

Reading out an experiment seldom requires a single residualization. A readout covers several outcome
metrics, heterogeneity analysis adds treatment-by-segment interactions, and inference under clustered or
switchback assignment often proceeds by randomization, in which the assignment vector is redrawn many
times and the test statistic recomputed on each draw @bojinov2023. Across all of these right-hand sides
the design matrix $D$ and the weights $W$ are unchanged; only the vector being residualized differs. The
factor-pair preconditioner depends on $D$ and $W$ alone, so one factorization serves every column, and
Section 5 reports the measured decline in cost per right-hand side.

= Why MAP slows down, and a diagnostic

Let the factors be user, item, and period. The Gramian has the block form

$
  G = mat(
    G_(U U), C_(U I), C_(U P);
    C_(U I)', G_(I I), C_(I P);
    C_(U P)', C_(I P)', G_(P P)
  ).
$

Each diagonal block holds weighted level counts and is diagonal. MAP, also called iterative demeaning or
the zig-zag algorithm, updates one factor at a time @guimaraes2010 @gaure2013: it computes the weighted
mean of the current partial residual within each level of factor $q$ and subtracts it. A complete sweep is
cheap because every update divides by a diagonal count, and the cross-tabulations are never solved as
coupled systems. When the graph is poorly connected, each sweep transfers only a small amount of
information across the narrow bridges, and many sweeps are required. A diagnostic for a factor pair is the
spectral gap

$
  g a p_(q r) = 1 - rho_(q r),
  quad rho_(q r) = sigma_2^2 (
    G_(q q)^(-1/2) C_(q r) G_(r r)^(-1/2)
  ),
$

omitting the unit singular value attached to each connected component. A small gap indicates near nesting
or weak cross-exposure. We treat it as a diagnostic rather than a solver-selection rule, because component
size and setup cost also determine runtime.

= The factor-pair preconditioner

For a factor pair $(q,r)$, changing the sign of one factor in the pair block gives

$
  L_(q r) = mat(G_(q q), -C_(q r); -C_(q r)', G_(r r)),
$

a weighted bipartite graph Laplacian whose diagonal entries are weighted degrees and whose off-diagonal
entries are non-positive edge weights. We solve the pair problem in this representation and reverse the
sign afterward. The preconditioner uses one local problem per connected component of every factor pair. If
$R_(q r)$ restricts a global coefficient vector to a pair subdomain and $Omega_(q r)$ holds symmetric
partition-of-unity weights, the additive Schwarz operator is

$
  M^(-1) = sum_((q,r)) R_(q r)' Omega_(q r) A_(q r) Omega_(q r) R_(q r),
$

where $A_(q r)$ approximates the inverse of the pair block. With three factors the user--item,
user--period, and item--period corrections overlap, and a level appearing in two subdomains receives
weight $1 slash sqrt(2)$ on each side so that the squared weights sum to one.

Large pair systems cannot be inverted densely. Eliminating one side of a bipartite graph is a division by
level counts, but the Schur complement creates clique fill among the neighbours of each eliminated level.
We replace these cliques with sparse randomized trees and factor the reduced Laplacian with approximate
Cholesky methods @gao2025, which makes the local solves sparse and approximate. We use them inside LSMR
@fong2011, which applies the design operator, its transpose, and $M^(-1)$ without forming $G$. The outer
iteration refines local-solve error while the preconditioner removes the slow graph directions, and the
solver returns the solution of the original problem rather than of a modified objective.

= Benchmark evidence

We benchmark the full regression path, including factor representation, residualization, low-dimensional
estimation, and preconditioner setup. Both designs use ten million observations, one covariate, and three
fixed effects; the simple design has dense random exposure and the difficult design a sparse, nearly
nested two-sided structure @berge2026fixest. Times are medians over three calls on an Apple M4 Mac mini
with 10 cores and 16 GB of memory.

#align(center)[
  #text(size: 7.6pt)[
    #table(
      columns: (1.5fr, 1.05fr, 0.82fr, 0.72fr, 0.72fr, 0.86fr),
      stroke: 0.3pt + light-rule,
      inset: (x: 4pt, y: 3pt),
      align: (left, right, right, right, right, right),
      table.hline(stroke: 0.75pt + rule),
      table.header(th[Design], th[Gap], th[PyFixest MAP], th[fixest], th[FEM.jl], th[within]),
      table.hline(stroke: 0.4pt + rule),
      table.cell(colspan: 6, fill: rgb("#fbfcfd"))[#emph[OLS, 10M observations]],
      [simple (well-connected)], [0.857], [2.30s], [2.54s], [2.09s], [11.0s],
      [difficult (near-nested)], [$1.67 times 10^(-7)$], [306.2s], [62.3s], [26.9s], [*4.10s*],
      table.hline(stroke: 0.4pt + rule),
      table.cell(colspan: 6, fill: rgb("#fbfcfd"))[#emph[Poisson, 1M observations]],
      [simple (well-connected)], [--], [7.86s], [4.72s], [5.76s], [8.61s],
      [difficult (near-nested)], [--], [capped], [439.4s], [129.8s], [*5.42s*],
      table.hline(stroke: 0.75pt + rule),
    )
  ]
]

On the simple design the graph is dense, MAP converges in few sweeps, and `within` is slowest because the
preconditioner does not repay its setup cost; 76% of its demeaning time is spent on construction. On the
difficult design the ranking reverses. The gap falls to $1.67 times 10^(-7)$, unaccelerated MAP takes
306.2s and the fastest MAP backend 62.3s, while `within` completes in 4.10s. The setup share falls to 37%,
which indicates that construction is being amortized within a single fit.

The Poisson rows matter for experiment readouts because platform outcomes are frequently counts of
bookings, clicks, or sessions. Iteratively reweighted least squares repeats the demeaning step at every
iteration @correia2020ppmlhdfe, so any change in absorption cost is multiplied by the number of
iterations. On the difficult design `rust-map` does not converge within its cap and `fixest` takes 439.4s,
against 5.42s for `within`. A controlled sweep that varies two-sided mobility while holding the rest of
the data-generating process fixed reproduces the pattern: `within` stays between 0.369s and 0.557s across
the sweep, whereas MAP reaches its 10,000-sweep cap at the lowest mobility. Across a broader set of public
benchmark datasets, accelerated MAP wins on small or compact graphs, and the factor-pair preconditioner
wins on the larger networks whose hard components cover much of the sample.

= Amortization across right-hand sides

Randomization inference and multi-metric readouts residualize many vectors against one fixed-effect
structure. We measure this directly on a one-million-observation difficult design, comparing the
factor-pair preconditioner against diagonally preconditioned LSMR as the number of right-hand sides $K$
grows.

#align(center)[
  #text(size: 7.6pt)[
    #table(
      columns: (0.55fr, 0.95fr, 0.95fr, 0.7fr, 1.05fr),
      stroke: 0.3pt + light-rule,
      inset: (x: 4pt, y: 3pt),
      align: (center, right, right, right, right),
      table.hline(stroke: 0.75pt + rule),
      table.header(th[$K$], th[diagonal], th[factor-pair], th[ratio], th[factor-pair / RHS]),
      table.hline(stroke: 0.4pt + rule),
      [1], [3.83s], [0.14s], [27.9#sym.times], [0.137s],
      [5], [8.09s], [0.28s], [28.5#sym.times], [0.057s],
      [25], [37.58s], [1.16s], [32.3#sym.times], [0.047s],
      table.hline(stroke: 0.75pt + rule),
    )
  ]
]

Cost per right-hand side falls by a factor of 2.9 between one and twenty-five columns and is close to flat
thereafter, because construction is paid once. A randomization-inference loop with several hundred draws
sits far along this curve, so the setup cost that makes the preconditioner unattractive for a single fit
on a dense graph is divided by the number of draws. This suggests that the crossover point moves in favour
of the preconditioned solver as $K$ grows. We have not benchmarked a full randomization-inference loop
against MAP end to end, and we present the extrapolation as an implication of the measured split rather
than as a measured result.

In a 100,000-observation numerical check the largest slope difference from MAP is $3.2 times 10^(-7)$;
comparisons use fitted values and residuals, because fixed-effect coefficients themselves depend on
normalization. The preconditioned solver uses more memory, with an incremental peak-RSS cost of 128--268
MiB at one million observations.

= Practical guidance and software

Accelerated MAP remains a good default for dense, well-connected graphs and for one-off fits where setup
cannot amortize. The factor-pair preconditioner is intended for sparse two-sided exposure, cluster designs
with thin bridges, count outcomes estimated by IRLS, and workloads that residualize many right-hand sides
against one fixed-effect structure. The gap identifies difficult pair graphs; pilot timings should decide
whether setup is recovered.

The method is available in the open-source `within` project @within. The computational core is written in
Rust and exposed through Rust (`within`), Python (`within-py`), and R (`withinr`) interfaces, and it is
available as a PyFixest demeaning backend @pyfixest. Python users residualize an outcome and several
covariates jointly with `solve_batch`, which reuses one preconditioner across the columns. The repository
records benchmark inputs, timing boundaries, convergence flags, and external normal-equation residuals, so
that a capped MAP run is distinguishable from a successful fit.

#v(0.2em)
#show bibliography: set text(size: 7.1pt)
#show bibliography: set par(leading: 0.42em, spacing: 0.34em)
#columns(2, gutter: 1.1em)[
  #bibliography("refs_code_mit.bib", style: "american-society-of-mechanical-engineers", title: [References])
]

#set document(
  title: "Graph-Preconditioned Estimation of High-Dimensional Fixed-Effect Models",
  author: "Alexander Fischer and Kristof Schröder",
)
#set page(
  paper: "a4",
  margin: (x: 2.55cm, y: 2.45cm),
  numbering: "1",
  number-align: center,
)
#set text(font: "Libertinus Serif", size: 10.6pt)
#set par(justify: true, leading: 1.06em, spacing: 1.12em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#set figure(gap: 0.95em)
#show figure.caption: set text(size: 9pt)
#show heading.where(level: 1): it => {
  set block(above: 1.45em, below: 0.68em)
  text(size: 15pt, weight: "bold", it)
}
#show heading.where(level: 2): it => {
  set block(above: 1.1em, below: 0.48em)
  text(size: 12pt, weight: "bold", it)
}

#let solver-img(name) = "figures/solver/" + name
#let result-img(name) = "figures/results/" + name
#let table-rule = rgb("#7b8494")
#let table-light-rule = rgb("#d8dee8")
#let table-head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]
#let miss = text(fill: rgb("#777777"))[-]
#let dg(body) = text(fill: rgb("#2563eb"), body)
#let cr(body) = text(fill: rgb("#c2410c"), body)
#import "generated/paper_values.typ": *

#align(center)[
  #set par(justify: false)
  #text(size: 18.5pt, weight: "bold")[Graph Preconditioning for]

  #v(0.18em)
  #text(size: 18.5pt, weight: "bold")[High-Dimensional Fixed Effects Regression]

  #v(0.65em)
  #text(size: 10.5pt)[Alexander Fischer#footnote[trivago] and Kristof Schröder#footnote[appliedAI Institute for Europe gGmbH]]

  #v(0.4em)
  #text(size: 9.5pt)[Draft: August 8, 2026]
]

#v(0.9em)

#align(center)[
  #block(
    width: 88%,
    inset: (x: 1.1em, y: 0.85em),
    fill: rgb("#f7f8fa"),
    stroke: 0.35pt + rgb("#d8dee8"),
    radius: 4pt,
  )[
    #text(size: 9.6pt)[
      #text(weight: "bold")[Abstract.] The Method of Alternating Projections (MAP) is the
      standard algorithm for estimating high-dimensional fixed-effect regressions. It is fast when
      the fixed-effect dimensions are well connected, but converges slowly on sparse or nearly nested
      fixed-effect graphs, such as matched employer-employee panels where worker-firm mobility
      links separate worker and firm effects. The convergence behavior of MAP depends on the
      mobility pattern linking workers to firms, yet MAP only indirectly makes use of this 
      information by iterating over one fixed effect at a time. That mobility pattern is,
      however, directly encoded in the matrix of worker-firm match counts, which together
      with the worker and firm count diagonals forms a graph Laplacian - the standard
      matrix representation of a weighted graph - after a sign flip. This structure lets
      us approximate its inverse using sparse matrices.
      We propose a graph-preconditioned iterative solver whose reusable preconditioner is
      built from small, local factor-pair subproblems - worker-firm, worker-year, and
      so on - that use the graph directly. 
      Benchmarks show that all methods are fast on well-connected designs. As
      connectivity weakens, MAP and diagonally preconditioned LSMR slow down, while
      factor-pair preconditioning remains fast and its runtime falls because the graph
      preconditioner is cheaper to construct on sparser graphs.
    ]
  ]
]

#v(0.25em)
#align(center)[
  #text(size: 9.1pt)[
    #strong[Keywords:] high-dimensional fixed effects; alternating projections;
    preconditioning; matched employer-employee data; computational econometrics
  ]
]
#align(center)[
  #text(size: 9.1pt)[#strong[JEL codes:] C55; C63; C81; C87; J31]
]

= Introduction

Fixed effects appear in roughly half of the research published in leading economics and
finance journals, according to #cite(<goldsmith2026tracking>, form: "prose"). Labor
economists use worker and firm fixed effects to separate worker heterogeneity from firm
wage premia. Health economists study physician practice styles with individual,
physician, and region fixed effects in mover designs. Education researchers use school,
student, teacher, and student-teacher effects.

The Frisch-Waugh-Lovell (FWL) theorem reduces estimation to two operations: remove the
variation explained by the fixed effects from the outcome and regressors, then run a
low-dimensional regression on the residualized variables @frisch1933 @lovell1963. With
high-dimensional fixed effects, software usually performs the first operation with the
Method of Alternating Projections (MAP), also known as iterative demeaning or the
"Zig-Zag" algorithm @guimaraes2010 @gaure2013. Leading packages such as Stata's
`reghdfe` @reghdfe @correia2017, R's `fixest` @berge2026fixest, and Python's PyFixest
@pyfixest use MAP or accelerated variants of it.

MAP handles the fixed-effect dimensions one at a time. It never uses the observed
worker-firm links directly. In the worker-firm wage model of
#cite(<akm1999>, form: "prose"), extended here with year effects, MAP subtracts worker
means, then firm means, then year means. Each update uses only the residual left by the
previous update; MAP repeats the sequence until changes in the residuals fall below a
chosen tolerance.

Yet those links matter for identification, precision, and computation @correia2017
@jochmans2019. They form a
graph in which movers create paths between firms, while stayers add observations without
connecting firms. When few workers move between groups of firms, some combinations of
worker and firm effects become difficult to separate. MAP can then require many
repetitions before estimates in one group reflect changes in another. The same mobility
links that identify worker and firm effects thus govern how quickly MAP converges.

Following #cite(<jochmans2019>, form: "prose"), we measure the connectivity of a
connected factor-pair graph by $lambda_2$, the second-smallest eigenvalue of its
normalized Laplacian. The graph is weighted by the number of observed co-occurrences.
Smaller values indicate weaker connectivity. The measure describes the connectivity of
two fixed-effect dimensions. Unless stated otherwise, $lambda_2$ refers to the
worker-firm graph.

The same graph appears in the weighted cross-product matrix of the fixed-effect
indicators, which we call the Gramian @correia2017. Its diagonal blocks count
observations for each worker, firm, or other fixed-effect level. Its off-diagonal blocks
count which levels are observed together, such as the number of observations for each
worker-firm pair. MAP uses the diagonal blocks one at a time and does not use these
pairwise counts directly.

We therefore build a factor-pair graph preconditioner for designs with a small pairwise
$lambda_2$, where MAP tends to converge slowly. A preconditioner transforms a linear
system so that an iterative solver reaches the same solution in fewer steps. Rather than
handling the fixed-effect dimensions one at a time, our preconditioner builds a local
problem for every pair, such as worker-firm and worker-year, and uses the observed links
directly. After a sign change, each pair block is a graph Laplacian, the standard matrix
representation of a weighted graph. Sparse matrix methods can approximate its inverse at
low cost @spielman2014 @gao2025. We use a weighted sum of these approximate pair
solutions as the preconditioner for LSMR.#footnote[`FixedEffectModels.jl` @fixedeffectmodels also uses
LSMR @fong2011. It applies diagonal preconditioning, which uses fixed-effect counts but
not the links between fixed-effect dimensions. We instead precondition LSMR with the
factor-pair graph.]

Constructing the factor-pair preconditioner takes time. On a well-connected graph, the
reduction in iterations may not repay this setup cost because MAP and diagonal
preconditioning are already fast. As $lambda_2$ falls, however, the reduction in iterations
can outweigh the construction cost. @fig-gap-runtime compares total regression times as
worker-firm connectivity varies. The top row changes worker mobility; the bottom row
changes sorting among movers. In each row, the left panel compares package defaults and
the right panel compares solvers within PyFixest. We choose each solver's tolerance so
that coefficient and residual errors are similar. Section 7 compares run time at the
accuracy each method actually achieves.

#figure(
  image(result-img("gap_runtime.svg"), width: 100%),
  caption: [The panels plot median regression time against the worker-firm
  normalized-Laplacian gap $lambda_(2,W F)$ on logarithmic axes. We compute $lambda_2$
  after removing singleton fixed-effect levels. Smaller values mean weaker connectivity.
  All panels use the same reversed horizontal scale, so $lambda_2$ decreases from left to right. Each simulated
  worker-firm-year panel has 1 million observations. The top panels vary worker mobility;
  the bottom panels hold move probability at one and vary sorting. The left panels
  compare packages at their default settings, including their own treatment of
  fixed-effect levels observed only once. The right panels compare four PyFixest solver
  configurations, with tolerances chosen to give similar coefficient and residual
  errors. Filled markers indicate that all three planned fits produced estimates; hollow
  markers indicate that only one or two did. Lines join the medians in order of $lambda_2$
  and are not fitted trends. An arrow marks the median time for fits that reach the
  iteration limit. Because these fits did not finish, the marked time is a lower bound.
  Arrows are not joined to the lines. Fits that end in another error are omitted.]
) <fig-gap-runtime>

At high connectivity, all implementations finish quickly. As $lambda_2$ falls, MAP
and LSMR without factor-pair preconditioning slow down. Factor-pair LSMR remains fast. In
the lowest-mobility designs, its run time falls because the worker-firm subproblems are
cheaper to construct.

Sections 2-5 set up fixed-effect absorption and connect MAP convergence to graph
connectivity. Section 6 develops the factor-pair preconditioner, and Section 7 reports the
benchmarks. Section 8 describes the software; Section 9 concludes. The appendices give
the algorithm, report additional benchmarks, and compare coefficient estimates and
memory use.

= Absorbing Fixed Effects#footnote[Researchers employ several names for this operation:
"absorbing fixed effects", "demeaning", "residualizing", or applying the "within
transformation". We use these terms interchangeably throughout.]

We study the linear model

$ y = X beta + D alpha + epsilon, $

where $X$ contains the regressors of interest, $D$ is the fixed-effect design matrix,
and $alpha$ collects the fixed-effect coefficients. In high-dimensional applications, $D$
may have hundreds of thousands or millions of columns. Forming $[X quad D]$ or inverting
its cross product can then be computationally infeasible. The Frisch-Waugh-Lovell (FWL)
theorem lets us compute $hat(beta)$ without forming $[X quad D]$ or inverting its
cross product.

FWL reduces the computation of $hat(beta)$ to two steps. First, we regress $y$ on $D$ and
retain the residual $tilde(y)$. We repeat this operation for every column of $X$. Let
$M_D$ denote the linear operator that maps a variable to its residual, so that
$tilde(y) = M_D y$ and $tilde(X) = M_D X$. In the second step, we regress the
residualized outcome on the residualized covariates,

$ tilde(y) = M_D y, quad tilde(X) = M_D X, quad
  hat(beta) = (tilde(X)' W tilde(X))^(-1) tilde(X)' W tilde(y), $

where $W$ is a diagonal matrix of weights. With one covariate, this amounts to
three regressions: $y$ on $D$, $x$ on $D$, and $tilde(y)$ on $tilde(x)$. FWL implies that
the slope in the last regression equals the coefficient on $x$ in the full regression of
$y$ on $x$ and $D$. 

To residualize the outcome and each covariate in $X$, we project them onto the column
space of the fixed-effect dummy matrix $D$ by weighted least squares. For a variable
$mu$ to be residualized - either $y$ or one column of $X$ - we solve

$ hat(alpha)_mu = arg min_alpha || D alpha - mu ||_W^2, $ <eq:demean-ls>

and the residual is $tilde(mu) = mu - D hat(alpha)_mu$. The first-order
condition for @eq:demean-ls,

$ D' W (D hat(alpha)_mu - mu) = 0, quad "equivalently" quad
  G hat(alpha)_mu = D' W mu, quad G = D' W D, $ <eq:fwl-normal>

determines $hat(alpha)_mu$. The coefficient matrix $G$ is the same for each FWL
residualization; only the variable being residualized changes. The cost of
residualization therefore depends on the structure of the Gramian $G$, the weighted
cross-product matrix of the fixed-effect dummies. The AKM worker-firm model gives this
matrix a concrete interpretation. Worker moves link firms through shared workers and
determine the off-diagonal blocks of $G$.

= A Running Example: The AKM Model

The AKM model of #cite(<akm1999>, form: "prose") provides the worker-firm setting used
throughout the paper. It uses workers who move across firms to separate persistent
worker heterogeneity from firm wage premia. We write the AKM regression equation as

$ y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t), $

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed effect for the
firm employing worker $i$ at time $t$, and $phi_t$ is a time fixed effect. 

The AKM specification maps naturally to a graph. Workers and firms are nodes in a
bipartite graph; each employment spell contributes an edge. Worker moves induce a
firm-to-firm graph: two firms are connected when at least one worker is observed at both
firms. Stayers - workers who never change their employer - add observations to existing
worker-firm links but do not create bridges between firms. Year effects enter as a third,
low-dimensional factor observed on the same worker-firm records. @fig-connectivity contrasts a well-connected
mobility graph with one that fragments under strong sorting.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 50%),
  caption: [Worker-firm graphs under high and low mobility. High mobility creates many
  paths between firms. With low mobility and strong sorting, only a few worker moves
  connect otherwise separate groups of firms.]
) <fig-connectivity>

These mobility links matter for identification and computation. In AKM, movers provide
the comparisons that separately identify worker and firm fixed effects by distinguishing
worker heterogeneity from firm wage premia. A worker observed at only one firm provides
no such comparison: a high wage could reflect an unusually productive worker, a
high-wage firm, or both. Without worker moves, these components are not separately
identified. Movers observed at firms with different wage profiles help attribute wage
variation to worker effects or firm premia. If a worker earns high wages across several
firms, the comparison points toward a worker effect. If many different workers earn
higher wages at the same firm, it points toward a firm premium. The more such cross-firm
comparisons the data contain, especially across otherwise different firms, the easier it
is to identify worker and firm premia. In the graph, these moves are exactly the edges
that connect firms. In the Gramian, the same links appear in the worker-firm
cross-tabulation.

= The Graph Structure of the Gramian

The block structure of the Gramian $G = D' W D$ represents the bipartite graph of
worker and firm connections introduced in Section 3 @correia2017.
Suppose that the columns of $D$ are ordered as worker levels, firm levels, and year
levels. Then

$ G = mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
). $

The #dg[diagonal blocks] $#dg[$G_(W W)$]$, $#dg[$G_(F F)$]$, and $#dg[$G_(Y Y)$]$
contain weighted counts for workers, firms, and years. An observation belongs to one
level of each factor, so these blocks are diagonal; solving them requires only division
by group counts.

The #cr[off-diagonal blocks] are cross-tabulations: the worker-firm block $#cr[$C_(W
F)$]$ records how often worker $i$ is observed at firm $j$, and the worker-year and
firm-year blocks have analogous interpretations.

The following small worker-firm panel illustrates how to populate its Gramian. We
ignore regression weights and set $W = I$.

#align(center)[
  #table(
    columns: (0.45fr, 0.8fr, 0.7fr, 0.7fr, 0.7fr),
    stroke: 0.35pt + table-light-rule,
    inset: (x: 5pt, y: 3.8pt),
    align: center,
    table.hline(stroke: 0.8pt + table-rule),
    table.header(th[Obs.], th[Worker], th[Firm], th[Year], th[$y$]),
    table.hline(stroke: 0.45pt + table-rule),
    [1], [$W_1$], [$F_1$], [$Y_1$], [3.2],
    [2], [$W_1$], [$F_2$], [$Y_2$], [4.1],
    [3], [$W_2$], [$F_1$], [$Y_1$], [2.8],
    [4], [$W_2$], [$F_1$], [$Y_2$], [3.9],
    [5], [$W_3$], [$F_2$], [$Y_1$], [5.0],
    [6], [$W_3$], [$F_2$], [$Y_2$], [4.5],
    table.hline(stroke: 0.8pt + table-rule),
  )
]

#figure(
  image(solver-img("toy_worker_firm_projection.svg"), width: 50%),
  caption: [Worker-firm graph for the example panel. Worker $W_1$ works at both firms and
  connects them. Worker $W_2$ works only at $F_1$, and worker $W_3$ works only at $F_2$.]
) <fig-toy-projection>

@fig-toy-projection plots the worker-firm projection of this panel. Worker $W_1$ has
employment spells at both $F_1$ and $F_2$ and therefore links
the two firms. In AKM terms, $W_1$ is a mover. Worker $W_2$ stays at $F_1$ for two periods, while $W_3$
stays at $F_2$ for two periods. Both are stayers.

The diagonal blocks are count matrices. In this example, each worker
is observed twice, each firm three times, and each year three times, so

$ G_(W W) = mat(2, 0, 0; 0, 2, 0; 0, 0, 2), quad
  G_(F F) = mat(3, 0; 0, 3), quad
  G_(Y Y) = mat(3, 0; 0, 3). $

The off-diagonal blocks are cross-tabulations between factors. The worker-firm block is

$ C_(W F) = mat(
  1, 1;
  2, 0;
  0, 2
). $

The first row shows that worker $W_1$ appears once at firm $F_1$ and once at firm
$F_2$. Workers $W_2$ and $W_3$ are stayers, appearing twice at $F_1$ and $F_2$,
respectively. The worker-year block is

$ C_(W Y) = mat(
  1, 1;
  1, 1;
  1, 1
). $

Each worker is observed once in each year. The firm-year block is

$ C_(F Y) = mat(
  2, 1;
  1, 2
). $

Firm $F_1$ appears twice in year $Y_1$ and once in year $Y_2$; firm $F_2$ has the
opposite pattern. Together, the diagonal count blocks and off-diagonal
cross-tabulation blocks form the full Gramian.

With column order $(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$, the full Gramian is

$ G = mat(augment: #(hline: (3, 5), vline: (3, 5), stroke: 0.4pt + rgb("#b0b8c4")),
  2, 0, 0, 1, 1, 1, 1;
  0, 2, 0, 2, 0, 1, 1;
  0, 0, 2, 0, 2, 1, 1;
  1, 2, 0, 3, 0, 2, 1;
  1, 0, 2, 0, 3, 1, 2;
  1, 1, 1, 2, 1, 3, 0;
  1, 1, 1, 1, 2, 0, 3
). $

The worker-firm submatrix represents the bipartite graph algebraically. The diagonal entries
are worker and firm counts, and the entries of $C_(W F)$ are edge multiplicities between
workers and firms. Changing the sign of $C_(W F)$ turns this submatrix into a graph
Laplacian: its off-diagonal entries are non-positive, and every row sums to zero because
each diagonal count cancels the off-diagonal spell counts in the same row. For a worker,
the diagonal entry is the number of employment spells. The off-diagonal entries give
the number of those spells at each firm. Firm rows have the analogous
interpretation, with spell counts summed over workers.


$ L_(W F) = mat(augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
  2, 0, 0, -1, -1;
  0, 2, 0, -2, 0;
  0, 0, 2, 0, -2;
  -1, -2, 0, 3, 0;
  -1, 0, -2, 0, 3
). $

The same Laplacian construction applies to any pair of fixed effects. These pairwise
Laplacians are the basis of the preconditioner in Section 6. MAP does not use them
directly. It alternates among the diagonal worker, firm, and year blocks, while
cross-factor information passes from one update to the next through the residual.


= Alternating Projections and Graph Connectivity

The workhorse algorithm for multi-way fixed effects is the Method of Alternating
Projections (MAP), also referred to as iterative demeaning or the "zig-zag" algorithm
@guimaraes2010 @gaure2013. Many packages use MAP or a variant, often with acceleration
methods @berge2018 @correia2017. For example, `fixest` uses Irons-Tuck extrapolation
@irons1969 @berge2026fixest. The fixed-effect graph and its algebra, introduced in
Sections 3 and 4, determine MAP's convergence rate.

MAP solves the FWL residualization problem by iterating over one fixed effect at a
time. In the worker-firm-year model, MAP first subtracts worker means
from the current residual, then firm means from the updated residual, and then year
means. It repeats this sequence until convergence.

We write the FWL normal equations (see @eq:fwl-normal) in block form with
$D = [D_W quad D_F quad D_Y]$ as

$ mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
) mat(alpha_W; alpha_F; alpha_Y)
= mat(D_W' W mu; D_F' W mu; D_Y' W mu). $

Equivalently,

$ G_(W W) alpha_W + C_(W F) alpha_F + C_(W Y) alpha_Y = D_W' W mu, $

$ C_(W F)' alpha_W + G_(F F) alpha_F + C_(F Y) alpha_Y = D_F' W mu, $

$ C_(W Y)' alpha_W + C_(F Y)' alpha_F + G_(Y Y) alpha_Y = D_Y' W mu. $

Rearranging each equation expresses one block of effects conditional on the others.
Substituting $C_(W F) = D_W' W D_F$ and $C_(W Y) = D_W' W D_Y$ into the first
equation and factoring out $D_W' W$ gives

$ G_(W W) alpha_W = D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Because $G_(W W)$ is diagonal, with entries equal to workers' total observation weights,
we obtain $alpha_W$ by dividing each worker's weighted partial residual by that worker's
total observation weight. For firms, the same rearrangement gives

$ G_(F F) alpha_F = D_F' W (mu - D_W alpha_W - D_Y alpha_Y), $

and the year equation is analogous. Because $G_(W W)$, $G_(F F)$, and $G_(Y Y)$
are all diagonal, we solve each equation by computing a weighted group mean in a single
pass over observations.

MAP repeats these diagonal block solves. Holding the other effects fixed, it updates the
worker effects from the current partial residual, then performs the same step for firms
and years. A complete pass cycles through the fixed-effect dimensions and subtracts the
weighted group mean of the current partial residual for each factor.

The cross-tabulation blocks $C_(W F)$, $C_(W Y)$, and $C_(F Y)$ enter the algorithm only
indirectly. For instance, the worker update is computed from the partial residual
$mu - D_F alpha_F - D_Y alpha_Y$, while the firm update is computed from
$mu - D_W alpha_W - D_Y alpha_Y$. MAP therefore does not solve the worker-firm,
worker-year, and firm-year subproblems jointly. Instead, each block update changes the
residual used by the next update.

MAP converges quickly when the graph is well connected. In a high-mobility
worker-firm panel, many workers move across firms, so that worker and firm effects can
be compared through many overlapping employment histories. A high wage at one firm can
then be related to wages earned by the same workers at other firms. A worker update
quickly changes the information available to the next firm update, and vice versa.

When mobility is sparse, sorting is strong, or one factor is nearly nested in another,
MAP slows down. Movers form the edges of the worker-firm graph and provide the information
needed to separate worker effects from firm effects. MAP uses this information only
indirectly, through repeated residual updates. When those edges are few or concentrated
in narrow regions of the graph, repeated residual updates separate the effects only
gradually. MAP still converges, but it may repeat these updates many times. A complete
pass is cheap because the diagonal block solves reduce to one-pass group means.

We use the normalized-Laplacian gap $lambda_(2,q r)$ as a diagnostic for MAP difficulty.
Smaller values indicate weaker pairwise connectivity. For models with three or more
fixed effects, the statistic describes one factor pair and is not a convergence bound
for the full model. A faster solver must use the pairwise links directly instead of
passing their information through one factor update at a time.

= The Factor-Pair Schwarz Preconditioner

== Preconditioners

LSMR can use the pairwise links through a preconditioner. MAP cannot take them as an
input: its update rule is fully determined by the list of fixed-effect dimensions, and
the cross-tabulations affect one update only through the residual passed to the next.
Iterative solvers such as LSMR accept an additional input, a preconditioner, that can
incorporate the graph structure. We therefore replace factor-by-factor demeaning via
MAP with LSMR @fong2011, an iterative least-squares algorithm that improves an initial
guess through repeated residual corrections. Switching solvers alone offers little
gain: weakly connected designs also leave LSMR with components of $G$ that take many
iterations to remove. The speedup depends on a good preconditioner. The rest of this
section develops one.

LSMR converges quickly when $G$ is balanced enough that every component of the residual
shrinks at a comparable rate. With weak links, sparse mobility, or near nesting, $G$
becomes unbalanced: LSMR removes some components after a few iterations while others
shrink much more slowly. This imbalance in $G$ is what we mean by a poorly conditioned
system. A preconditioner rescales the system so that LSMR reduces the easy and difficult
components at more similar rates, without changing the least-squares solution.

The ideal preconditioner is $G^(-1)$.#footnote[As in any model with several fixed
effects, the level effects are pinned down only up to a normalization: we can add a
constant to every worker effect and subtract it from every firm effect without changing
the fitted values $D alpha$, and likewise for years. The dummy-coded $G = D' W D$ and the
smaller blocks inverted below are therefore not invertible until this ambiguity is
removed - one normalization for the worker-firm pair, and a second once year effects are
added, as in the Section 4 example. We adopt the standard normalization within each
connected set and read every inverse below as that of the resulting system. The estimated
effects depend on the normalization; the residualized outcome and regressors, which are
all the regression uses, do not.] If $M^(-1) = G^(-1)$, then the
matrix seen by LSMR is

$ M^(-1) G = G^(-1) G = I. $ <eq:ideal-preconditioner>

With this ideal preconditioner, LSMR would recover the solution after one
correction because the inverse solves the system directly. Forming $G^(-1)$, however,
would require solving the fixed-effect normal equations themselves. The ideal inverse
is therefore a benchmark, not an implementable preconditioner. A useful preconditioner
must approximate enough of $G^(-1)$ to reduce the components that otherwise converge
slowly. The
time spent constructing and repeatedly applying it must be offset by the iterations it
saves.#footnote[LSMR never constructs $M^(-1) G$
or $G$ explicitly. It only multiplies vectors by $D$ and $D'$ and applies $M^(-1)$.]

== From the Block Inverse to the Diagonal Preconditioner

The block structure of $G^(-1)$ shows which parts of the ideal inverse a feasible
preconditioner should retain. For the worker-firm-year AKM model, the Gramian has the
block form

$ G = mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
), $

with diagonal weighted-count blocks $G_(W W), G_(F F), G_(Y Y)$ and off-diagonal
cross-tabulations $C_(W F), C_(W Y), C_(F Y)$. The standard formula for inverting a
block matrix shows how each off-diagonal block enters $G^(-1)$. For the two-factor block

$ G_2 = mat(G_(W W), C_(W F); C_(W F)', G_(F F)), $

it gives the explicit formula

$ G_2^(-1) = mat(
  G_(W W)^(-1) + G_(W W)^(-1) C_(W F) S^(-1) C_(W F)' G_(W W)^(-1), -G_(W W)^(-1) C_(W F) S^(-1);
  -S^(-1) C_(W F)' G_(W W)^(-1), S^(-1)
). $

$ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

The formula shows that every block of $G_2^(-1)$ depends on $C_(W F)$ only through
$S$; the cross-tabulation enters through matrix products, never as its own inverse.
$G_(W W)$ is diagonal, so $G_(W W)^(-1)$ is a division by weighted worker counts. The
matrix $S$ is the Schur complement: it is the firm-side mobility system left after
eliminating the worker effects. Solving this system is expensive for modern worker-firm
register data. Exact factorization makes the matrix denser by creating additional
nonzero entries, separate connected components require separate normalizations, and
weak mobility makes some components slow to resolve. $S^(-1)$ therefore accounts for
almost all the cost of the solve. The same block-inverse calculation extends to three
factors. The closed form has more terms, but every block of $G^(-1)$ still depends
jointly on the cross-tabulations $C_(W F), C_(W Y), C_(F Y)$.

The coarsest approximation to $G^(-1)$ keeps only the diagonal count inverses and drops
the Schur-complement corrections,

$ M_("diag")^(-1) = "diag"(G_(W W)^(-1), G_(F F)^(-1), G_(Y Y)^(-1)). $

Applying this preconditioner requires a single division by weighted level counts. It is
the diagonal preconditioner used with LSMR in `FixedEffectModels.jl` @fong2011
@fixedeffectmodels. Diagonal scaling records how many observations a level carries, but
not how that level connects to the rest of the labor market. Level counts can already be
useful when employment is concentrated in a few large firms, such as Novo Nordisk in
Denmark or Samsung in South Korea, because the size differences alone remove an
important source of scale variation. Diagonal scaling does not record whether workers
at those firms link them broadly to other firms or remain concentrated in a narrow
corner of the mobility graph.

== The Factor-Pair Schwarz Approximation

Additive Schwarz preconditioning approximates a large problem by solving smaller,
overlapping problems and adding their corrections @xu1992 @toselli2005. Here, the small
problems retain the pairwise connectivity that diagonal scaling drops. In the AKM case,
one problem contains workers and firms, another contains workers and years, and a third
contains firms and years. The worker-firm problem moves residual information along
observed employment links; the other two pair problems do the same for worker-year and
firm-year links. We combine the three pair corrections in the full vector of fixed-effect
coefficients. These corrections let LSMR account directly for the main pairwise links in
the Gramian, while the outer iteration handles the remaining three-way coupling.

Consider the worker-firm pair. Its local problem is the two-factor block from the Schur
calculation,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)). $

@fig-pair-block compares this worker-firm pair block with the single diagonal block
solved by a factor-level MAP update.

#figure(
  image(solver-img("factor_level_vs_pair_block.svg"), width: 88%),
  caption: [Matrices used by MAP and the factor-pair preconditioner in the example from
  Section 4. A MAP worker update (left) uses only the diagonal worker-count matrix
  $G_(W W) = "diag"(2,2,2)$. The factor-pair update (right) adds the firm-count matrix
  $G_(F F) = "diag"(3,3)$ and the matrix of worker-firm observation counts $C_(W F)$.
  The dashed outline marks the worker and firm coefficients that are updated together.]
) <fig-pair-block>

Its inverse carries $C_(W F)$ through the Schur complement, so the local correction
retains the worker-firm mobility geometry that diagonal scaling discards. To use this
correction in the three-factor problem, let
$R_(W F)$ select the worker and firm entries from the full coefficient vector $alpha =
[alpha_W; alpha_F; alpha_Y]$; its transpose $R_(W F)'$ places the resulting correction
back into the full vector. The diagonal matrix $tilde(D)_(W F)$ weights levels that
appear in more than one pair problem so their corrections are not counted twice. These
entries are called partition-of-unity weights. The exact worker-firm contribution is

$ P_(W F)^(-1) =
  R_(W F)' tilde(D)_(W F)
  mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1)
  tilde(D)_(W F) R_(W F). $

The worker-year and firm-year contributions $P_(W Y)^(-1)$ and $P_(F Y)^(-1)$ are built
the same way from $G_(W W), C_(W Y), G_(Y Y)$ and $G_(F F), C_(F Y), G_(Y Y)$. With three
factors each level appears in exactly two pair problems. Because the weights act on both
sides of each pair contribution, we set them so that the squared weights on a shared
level sum to one; a level in two pairs then carries $1 / sqrt(2)$. The exact factor-pair
Schwarz preconditioner is the sum of these three contributions,

$ P^(-1) = P_(W F)^(-1) + P_(W Y)^(-1) + P_(F Y)^(-1). $

The three terms include the worker-firm, worker-year, and firm-year cross-tabulations
separately. The simultaneous worker-firm-year coupling remains for the outer LSMR
iteration, along with the error introduced by splitting shared levels across pairs.

== Approximate Pair Solves via Graph Laplacians

For $P^(-1)$ to serve as the operator $M^(-1)$ inside LSMR, each pair contribution must
be computed without solving a large dense system. We invert the pair block directly
when a factor pair has few levels. In the example worker-firm panel of Section 4, the
pair block has three worker levels and two firm levels; after the normalization of
Section 6.1 removes the one free constant, the local solve is a $4 times 4$ inversion.
Direct inversion is impractical at the scale of modern worker-firm register data, where
a single pair can carry hundreds of thousands of levels per side. It becomes the same
large linear-algebra problem that the preconditioner is meant to avoid. The pair block's
graph-Laplacian structure provides an alternative.

For a worker-firm pair, the local pair step solves the pair-Gramian system

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)) x = u, $

where $u$ is the weighted worker-firm part of the current LSMR residual. This block is
not a graph Laplacian because its off-diagonal entries $C_(W F)$ are non-negative. Let
$T_(W F) = "diag"(I_W, -I_F)$ flip the sign of the firm entries, so that $T_(W F)^2 =
I$. Multiplying the pair block by $T_(W F)$ on both sides gives

$ L_(W F) = T_(W F) mat(G_(W W), C_(W F); C_(W F)', G_(F F)) T_(W F)
  = mat(G_(W W), -C_(W F); -C_(W F)', G_(F F)), $

a weighted bipartite graph Laplacian: symmetric, with non-positive off-diagonals and
zero row sums. Because $T_(W F)^2 = I$, the pair-Gramian solve follows from the
Laplacian solve by the same flip on each side,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1) = T_(W F) L_(W F)^(-1) T_(W F), $

where both inverses are read as in Section 6.1. The solve fixes the free constant on
each connected component by returning the zero-mean solution; the residualized
variables do not depend on this choice. A single Laplacian solve therefore yields the
pair-Gramian solution: we flip the residual, apply $L_(W F)^(-1)$, and flip back.

For preconditioning, this local solve need not be exact because the outer LSMR iteration
refines the remaining error. We approximate the Laplacian solve using sparse approximate
Cholesky factorizations @spielman2014 @gao2025. An exact Cholesky factorization gradually
turns the sparse pair block into a denser matrix. Eliminating a worker creates new
nonzero entries linking every pair of firms that worker visited, and these entries
accumulate as more workers are eliminated. These extra entries are called fill-in;
storing and updating them can make the exact factorization expensive. In the worst case,
the calculation requires roughly $k^3$ operations and memory proportional to $k^2$ for
a $k$-level system, as in a dense factorization. The randomized approximation controls
this growth. Its cost is roughly proportional to the number of observed worker-firm
links, with additional factors that grow only logarithmically with problem size. After
transforming the approximate Laplacian solve
back to worker-firm coordinates, we write $A_(W F)$ for the resulting approximate
pair-Gramian solve.

The same construction yields $A_(W Y)$ and $A_(F Y)$ for the other two pairs. Replacing
the exact pair inverses in the Schwarz sum with these approximations gives the
implemented preconditioner,

$ M^(-1) = sum_((q, r)) R_(q r)' tilde(D)_(q r) A_(q r) tilde(D)_(q r) R_(q r). $

In the worker-firm-year case each factor receives a diagonal contribution from its two
pair subdomains and each off-diagonal correction from the corresponding pair.

The approximate pair inverse adds a second approximation to the pairwise splitting. The
exact $P^(-1)$ replaces the full three-factor inverse with exact pair solves; $M^(-1)$
approximates each of those solves through $A_(q r)$. These two approximations affect only
how LSMR chooses each update. The fitted residuals are unchanged, and LSMR
still solves the original fixed-effect least-squares problem to the requested tolerance.

== Implementation Strategy

@fig-pair-strategy summarizes the full construction. We divide the fixed-effect graph
into overlapping factor pairs. For each pair, a sign flip turns the local Gramian block
into a graph Laplacian. A sparse approximate Cholesky routine then approximates the pair
inverse without letting the matrix become dense. We weight the overlapping corrections
so that shared fixed-effect levels are not counted twice, then add them to form the
Schwarz preconditioner $M^(-1)$.

The outer LSMR iteration uses this preconditioner to choose better-scaled updates
@fong2011 @arridge2014 @yang2024flexible. Once constructed, the same preconditioner can
be reused to residualize the outcome and every covariate. Appendix A gives the
algorithmic details.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Construction of the factor-pair preconditioner. Each local problem combines
  two fixed-effect dimensions and retains their observed links. A sign change turns its
  matrix into a graph Laplacian, the standard matrix representation of a weighted graph.
  Sparse approximate Cholesky solves each local system while limiting the extra nonzero
  entries created during factorization, which saves memory and computation. The weighted
  sum of these local solutions serves as the preconditioner for LSMR.]
) <fig-pair-strategy>

= Benchmarks

When connectivity is weak, MAP's factor-by-factor updates pass information slowly
through the fixed-effect graph. Factor-pair preconditioning should help most on these
graphs, but may add unnecessary setup cost on well-connected ones. We test this
prediction on controlled synthetic designs and public benchmark datasets.

The main runtime tables use package-level regression APIs, as does the public PyFixest
benchmark suite, rather than timing the demeaning step alone. Each timing covers the full
regression workflow: model setup, construction of the fixed-effect representation,
residualization of the outcome and covariates, and estimation of the coefficient of
interest. Appendix C compares the coefficient estimates returned by the preconditioned
solver and MAP. It also reports the memory cost of storing factor-pair information.

For cross-package comparisons, every software implementation receives the same input
data but uses its own default rules for singleton fixed-effect levels (levels that occur
only once) and observations removed because of separation. For successful fits, we
record the retained-observation count; for failed attempts, we record the convergence
status and error. Any shared control is stated with the relevant table. The comparisons
are not constrained to a common estimation sample. The later single-package mechanism
experiments use a common prepared sample because they isolate the preconditioner.

The runtime tables report $lambda_2$ for the relevant factor pair. We compute it after
removing singleton fixed-effect levels. For a disconnected pair graph, we compute
$lambda_2$ within each connected component and report the smallest value. The number in
parentheses is that component's share of retained observations. Pairwise $lambda_2$
remains a diagnostic rather than a convergence bound for models with three or more fixed
effects.

The OLS package-runtime tables compare six configurations: PyFixest MAP, PyFixest LSMR
with no, diagonal, or factor-pair preconditioning, R `fixest`, and
`FixedEffectModels.jl`:

#v(0.35em)

#text(size: 9.2pt)[
#table(
  columns: (1.0fr, 1.0fr, 2.0fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, left, left),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Configuration], th[Package], th[Algorithm]),
  table.hline(stroke: 0.45pt + table-rule),
  [`rust-map`], [PyFixest], [Vanilla Rust MAP, without acceleration.],
  [`within-off`], [PyFixest], [LSMR without preconditioning.],
  [`within-diagonal`], [PyFixest], [LSMR with diagonal preconditioning.],
  [`within`], [PyFixest], [LSMR with factor-pair Schwarz preconditioning.],
  [`fixest`], [R `fixest`], [Accelerated MAP with Irons-Tuck and other optimizations @berge2026fixest.],
  [`FEM.jl`], [`FixedEffectModels.jl`], [Diagonally preconditioned LSMR @fong2011 @fixedeffectmodels.],
  table.hline(stroke: 0.8pt + table-rule),
)
]

Reported CPU times are medians over each benchmark's measured repetitions. Headline OLS
uses the adaptive R1/R2/R3 rule; experiments with a fixed count use three repetitions.
All runs use an Apple M4 Mac mini with 10 CPU cores and 16 GB of memory running macOS
15.3.1.

We do not include `reghdfe` @reghdfe @correia2017 directly in the benchmark tables because Stata is not open
source and we lack a license. `reghdfe` is a mature accelerated-MAP
implementation; algorithmically, it belongs to the same
family as `fixest`'s accelerated MAP.

== Runtime Benchmarks

The implementations use different rules for deciding when to stop, so their reported
tolerance settings are not directly comparable. We first compare package defaults. The
achieved-precision experiment later in this section compares every method using the same
measures of coefficient and residual error.

The main OLS benchmarks use synthetic AKM-style panels with one million observations,
one covariate, ten periods, and worker, firm, and year fixed effects. They vary the
worker-firm graph in two ways while holding the rest of the data-generating process
fixed. The mobility designs progressively reduce the number of workers who change firms.
The sorting designs hold the move probability at one but assign a larger share of moves
within groups of firms. Appendix B reports further
synthetic designs, the empirical HDFE benchmarks assembled by Sergio Correia, and the
exact package-default AKM timings behind @fig-gap-runtime.

=== Worker Mobility

Lower worker mobility leaves fewer workers connecting multiple firms. Because MAP updates
one fixed-effect dimension at a time, information about firm effects moves more slowly
through the iteration. The factor-pair preconditioner uses the worker-firm links directly,
so its relative advantage grows as mobility falls. The top row of
@fig-gap-runtime shows worker-firm $lambda_2$ falling from $0.232$ to
$2.41 times 10^(-5)$. Under package defaults, all configurations in the first two designs
complete within 3.60 seconds. In the remaining designs, PyFixest MAP slows sharply or reaches
its cap, and LSMR without preconditioning reaches its cap. Factor-pair LSMR falls from
0.543 seconds in the first design to 0.365 seconds in the last.

=== Sorting Among Movers

Every worker changes firms between adjacent periods in these designs. The sorting
parameter $rho$ controls how strongly workers are matched to similar firms; larger values
concentrate moves within groups and weaken the links between them. The bottom row of
@fig-gap-runtime reports this comparison. Worker-firm $lambda_2$ falls from $0.222$ at
$rho=0$ to $2.13 times 10^(-4)$ at $rho=150,000$. MAP rises from 0.368 seconds at $rho=0$ to
60.1 seconds at $rho=10,000$, then reaches its default cap at $rho=150,000$. LSMR
without preconditioning reaches its default cap in the last four designs. Diagonal LSMR
rises from 0.367 to 1.73 seconds. Factor-pair LSMR remains below 1.1 seconds and takes
0.651 seconds at $rho=150,000$; it is faster than diagonal LSMR in the three
least-connected designs.

=== When Preconditioner Setup Dominates

The AKM designs vary connectivity gradually. The public `fixest` benchmark suite gives
a sharper comparison between a dense, well-connected design and a sparse, nearly nested
one @berge2026fixest. Both contain 10 million observations, one covariate, and worker,
firm, and year fixed effects.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Simple and difficult `fixest` designs (10 million observations, three fixed effects).]
#include "generated/tables/ols.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds and cover each OLS regression from
  model setup through coefficient estimation. Each package uses its default settings. A
  preliminary run sets the number of measured runs: 20 if it takes less than 1 second, 7
  if it takes 1 to 10 seconds, and 3 if it takes longer. When
  only $k$ of $n$ planned runs return an estimate, $t (k/n)$ reports their median time
  $t$. A dash means that no timing is available.
  `capped (0/n)` means that no run finishes before the iteration limit; `failed (0/n)`
  marks a failure other than reaching that limit. Both designs have 10 million
  observations, one covariate, and worker, firm, and year fixed effects. In the
  $lambda_2$ (share) column, smaller values mean weaker worker-firm connectivity. We
  compute $lambda_2$ after removing singleton fixed-effect levels. For a disconnected
  pair graph, we report the smallest component value; the number in parentheses is that
  component's share of retained observations.]
]]

On the dense design, all methods except factor-pair LSMR finish in 2.16 to 2.69 seconds;
factor-pair LSMR takes 11.5 seconds because its setup cost is not recovered in one fit.
On the near-nested design, factor-pair LSMR takes 4.45 seconds, compared with 11.0
seconds without preconditioning, 27.8 seconds for FEM.jl, 63.5 seconds for `fixest`, and
337.2 seconds for PyFixest MAP. PyFixest diagonal LSMR reaches its default cap on that
design.

=== Iterations After Setup

The iteration-count diagnostic separates construction from subsequent solves and uses
the 100,000-observation versions of the same simple and difficult designs.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Iterations on the simple and difficult designs (100,000 observations).]
#include "generated/tables/iterations.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] The MAP column counts complete passes over all
  fixed-effect dimensions. The other columns count LSMR iterations: `off` uses no
  preconditioner, `diagonal` scales by fixed-effect group counts, and `additive` uses the
  factor-pair preconditioner. A MAP pass and an LSMR iteration perform different work, so
  their counts are not directly comparable. Each count is the median of three runs; each
  run removes the fixed effects from one outcome and one covariate.]
]]

Among the LSMR configurations, the additive preconditioner needs 14 iterations on the
simple design and 22 on the difficult one. The diagonal count rises from 16 to 180,
while unpreconditioned LSMR rises from 37 to 249. Once the additive preconditioner has
been built, the difficult design requires only eight more LSMR iterations than the
simple design. The preconditioned solve is short in both cases; setup makes `within`
unattractive on the simple one-shot regression.

== Runtime and Achieved Precision

The methods apply their stopping tolerances to different quantities. PyFixest MAP defaults to
$10^(-6)$ and stops when every observation-level residual
changes by less than that amount after a full pass over the fixed-effect dimensions.
R's `fixest` also defaults to $10^(-6)$, but applies the threshold to successive
fixed-effect coefficient iterates. It accepts either a small absolute change or a small
scaled relative change, and periodically makes the same comparison for the residual sum
of squares. `FixedEffectModels.jl` passes its default tolerance of $10^(-6)$ as both
LSMR stopping tolerances. `within` defaults to $10^(-8)$ and stops when either the
estimated least-squares residual is at most $10^(-8)$ times the norm of the variable
being residualized (the right-hand side), or a scaled measure of the remaining
first-order-condition error falls below $10^(-8)$. This second measure, called the
normal-equation residual, is
$||A^T r||_2 / (||A||_F ||r||_2)$, using LSMR's running norm estimates @fong2011.

@fig-tolerance compares the methods on a common accuracy scale. It uses mobility
designs 1, 3, and 5. Before timing, we repeatedly remove fixed-effect levels that occur
only once. We do this once for each one-million-observation design and pass the resulting
rows to every method. We then vary each package's own tolerance, use a common 10,000-iteration
cap, and time three fits at every setting. The tight reference uses factor-pair LSMR at
tolerance $10^(-14)$. The package defaults for the iteration cap are 10,000 for PyFixest
MAP, `fixest`, and `FixedEffectModels.jl`, and 1,000 for `within`.

#figure(
  image(result-img("tolerance_frontier.svg"), width: 97%),
  caption: [Elapsed time and achieved accuracy on three AKM mobility designs. Each
  marker is the median of three fits on the same sample. Before timing, we remove
  fixed-effect levels observed only once and use the remaining observations for every
  method. Each line connects results for one method across requested tolerances. The top
  row measures coefficient error as
  $abs(hat(beta)-hat(beta)^star) / "SE"(hat(beta)^star)$. The bottom
  row measures residual error as $frac(||r-r^star||_2, ||r^star||_2)$. The reference
  coefficient $hat(beta)^star$ and residual $r^star$ come from factor-pair LSMR at
  tolerance $10^(-14)$. Error decreases from left to right. Circled markers use each
  package's default tolerance. Every fit is limited to 10,000 iterations. Settings that
  return no estimate are omitted and named in the annotations.]
) <fig-tolerance>

== Amortizing the Preconditioner <sec-amortization>

The accuracy comparison separates solver performance from differences in stopping
rules. Factor-pair LSMR also has a fixed setup cost. It constructs the pair blocks and
their approximate factorizations before starting the LSMR iterations. With fixed
observations, weights, and fixed-effect identifiers, these objects can be reused.

=== Setup Cost Across Connectivity

We time these two stages separately on the AKM mobility designs. The
two-factor specification absorbs worker and firm effects; the three-factor specification
also absorbs year effects. Each cell is the median of five standalone solves at one
million observations, with the same outcome and covariate used in both specifications.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Factor-pair setup and solve time across AKM connectivity.]
#include "generated/tables/akm_setup_cost.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds. For each design and
  specification, the table gives the median setup and solve time from five runs with 1
  million observations and an LSMR tolerance of $10^(-12)$. The two-fixed-effect
  specification absorbs worker and firm effects; the three-fixed-effect specification
  also absorbs year effects. In the $lambda_2$ (share) column, smaller values mean weaker
  worker-firm connectivity. We compute $lambda_2$ after removing singleton fixed-effect
  levels. For a disconnected pair graph, we report the smallest component value; the
  number in parentheses is that component's share of retained observations.]
]]

Setup is cheaper in the low-mobility designs. With two fixed effects, median setup declines
from 0.119 seconds at $delta=1$ to 0.067 seconds at $delta=0.001$; with three
fixed effects, it falls from 0.123 to 0.073 seconds. Solve time falls as well, from 0.323
to 0.057 seconds with two effects and from 0.239 to 0.176 seconds with three. The
low-mobility worker-firm graph has fewer cross-firm links to store and factorize. As a
result, the additive runtime in @fig-gap-runtime can decline while MAP and diagonal LSMR
slow down.

=== Ten Regressions on the Same Fixed Effects

Many empirical projects estimate several specifications on the same sample and fixed
effects. The factor-pair objects can then be constructed once and reused. We measure ten
sequential regressions on both one-million-observation `fixest` designs, using the same
outcome and ten covariates for all three policies. One additive policy constructs a new
preconditioner for every regression; the other keeps one preconditioner for all ten.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Ten regressions with rebuilt and cached preconditioners.]
#include "generated/tables/regression_reuse.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] For each policy, we sum setup and solve times over ten
  sequential regressions, then take the median across three repetitions. Times are in
  seconds. Each design has 1 million observations and worker, firm, and year fixed
  effects. Each regression removes the fixed effects from the common outcome and one of
  ten covariates at an LSMR tolerance of $10^(-12)$. Speedup divides the diagonal total time by the
  reported total time.
  `Additive, rebuilt` constructs a new factor-pair preconditioner for each regression;
  `Additive, cached` reuses one preconditioner for all ten.]
]]

Even across ten regressions, the additive preconditioner is slower on the simple design.
Diagonal preconditioning takes 0.692 seconds. Rebuilding the additive
preconditioner takes 4.46 seconds, while caching lowers this to 1.83 seconds. Even after
avoiding nine constructions, the cached additive path is still more than twice as slow as
diagonal preconditioning on this well-connected graph.

On the difficult design, diagonal preconditioning takes 50.3 seconds for the ten
regressions. Rebuilding the additive preconditioner lowers this to
1.77 seconds, and caching lowers it further to 1.23 seconds. Caching reduces additive
setup time from 0.434 to 0.040 seconds; the repeated solves take 1.34 seconds with
rebuilding and 1.19 seconds with caching.

== Poisson and Other GLMs

Faster fixed-effect solves matter especially when an estimator requires several of them,
as in generalized linear models fit by iteratively reweighted least squares (IRLS). Each
IRLS step fits a weighted least squares problem in which the response and covariates are
demeaned against the fixed effects @correia2020ppmlhdfe @stammann2018. A single GLM fit
therefore calls the demeaning routine repeatedly. The mapping from observations to
fixed-effect levels stays the same, but the weights change between calls. This pattern arises in `ppmlhdfe`, Poisson
fixed-effect estimators, and other IRLS-based GLM implementations.

PyFixest currently reuses the preconditioner built during the first weighted demeaning
call. This avoids a new factorization at every IRLS step, but later problems use a
preconditioner based on earlier weights. Reuse saves construction time and may require
more LSMR iterations. Rebuilding may reduce the iteration count but incurs construction
cost at every step. The table reports the current PyFixest reuse behavior.

The PPML comparison uses the simple and difficult designs from the
`fixest` benchmark suite @berge2026fixest at $n = 1$M observations, with one covariate
and worker, firm, and year fixed effects. The compared implementations are R `fixest`'s `fepois`,
`GLFixedEffectModels.jl`, and two PyFixest `fepois` paths: the default unpreconditioned
`rust-map` and `within` with PyFixest's current reuse policy. Packages apply their default
rules for separated observations, and each implementation receives the same 100-iteration outer
IRLS limit.

#v(0.35em)

#text(size: 8.8pt)[
#strong[Poisson benchmarks (1 million observations, one covariate).]
#include "generated/tables/ppml.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds for Poisson fixed-effect
  regressions with 1 million observations, one
  covariate, and worker, firm, and year fixed effects. `fixest` is R `fixest::fepois`;
  `rust-map` and `within` use PyFixest `fepois`; and `GLFEM.jl` is
  `GLFixedEffectModels.jl`. The `within` configuration reuses the first factor-pair
  preconditioner as the weights change across iteratively reweighted least squares
  steps. Each package applies its default rule for removing separated observations,
  where regressors and fixed effects perfectly predict some zero outcomes. If only $k$
  of the three planned fits return an estimate, $t (k/3)$ gives their median time $t$.
  `capped (0/3)` means that no fit finishes within 100 iteratively reweighted least
  squares steps; `failed (0/3)` marks a failure other than reaching that limit.]
  ]

#block(breakable: false)[On the simple design, all four paths finish in under ten seconds. `fixest` takes 4.72
seconds, `GLFEM.jl` 5.76, `rust-map` 7.86, and `within` 9.25. Runtime differences are
much larger on the difficult design. `rust-map` does not converge within
the iteration cap, `GLFEM.jl` takes 129.8 seconds, and `fixest` takes 439.4 seconds.
`within` finishes in 5.43 seconds. Sparse worker-firm coupling slows MAP, while the
factor-pair blocks encode those links directly.]

= Software

The open-source `within` project provides the solver benchmarked in Section 7 @within.
Its computational core is written in Rust, with APIs for Python and R.

#v(0.15em)

#block(breakable: false)[#text(size: 8.4pt)[
#table(
  columns: (0.75fr, 0.95fr, 1.15fr, 1.65fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 2.6pt),
  align: (left, left, left, left),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Interface], th[Package], th[Registry], th[Install]),
  table.hline(stroke: 0.45pt + table-rule),
  [Rust], [`within`], [#link("https://crates.io/crates/within")[crates.io]], [`cargo add within`],
  [Python], [`within-py`], [#link("https://pypi.org/project/within-py/")[PyPI]], [`pip install within-py`],
  [R], [`withinr`], [#link("https://py-econometrics.r-universe.dev/withinr")[py-econometrics R-universe]], [`install.packages("withinr", repos = "https://py-econometrics.r-universe.dev")`],
  table.hline(stroke: 0.8pt + table-rule),
)
]]

The Rust crate exposes the lower-level solver and its configuration types. The Python and
R packages provide `solve` and `solve_batch` APIs that residualize one or several
variables. PyFixest also provides the algorithm as a demeaning option @pyfixest.

The Python example below demeans an outcome and two covariates against worker and firm
fixed effects, then runs the FWL regression on the demeaned variables:

#text(size: 8.8pt)[```python
import numpy as np
from within import solve_batch
# Worker and firm identifiers as a column-major uint32 array
n = 100_000
categories = np.asfortranarray(np.column_stack([
    np.random.randint(0, 5_000, n).astype(np.uint32),
    np.random.randint(0,   500, n).astype(np.uint32),
]))
# Outcome and covariates
beta = np.array([1.0, -2.0])
X = np.random.randn(n, 2)
y = X @ beta + np.random.randn(n)
# Residualize y and X jointly; the preconditioner is reused across columns
res = solve_batch(categories, np.column_stack([y, X]))
y_tilde, X_tilde = res.demeaned[:, 0], res.demeaned[:, 1:]
# FWL on the demeaned variables
beta_hat = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
```]

`solve_batch` follows the FWL workflow from Section 2: it residualizes the outcome and
covariates together and reuses one factor-pair preconditioner across columns. Users do
not need to construct the Gramian or its pairwise blocks themselves.

#pagebreak()

= Conclusion

Graph connectivity links the econometric structure of a fixed-effect model to its
computational cost. The benchmarks show that it also changes solver rankings. On dense,
well-connected graphs, MAP is difficult to outperform because each pass is cheap, while
constructing the factor pairs adds overhead. When mobility is low, sorting is strong, or
effects are nearly nested, information passes slowly between updates under MAP and the
factor-pair preconditioner is often much faster.

Constructing the factor pairs becomes cheaper in the low-mobility AKM designs. Its cost
is incurred once for regressions that share the same sample, fixed effects, and weights.
Caching lowers additive runtime in both simple and difficult designs, but only the
difficult design is faster than diagonal preconditioning. For that design, caching
reduces additive setup from 0.434 to 0.040 seconds and total time from 1.77 to 1.23
seconds.

PPML calls the demeaning routine repeatedly because each IRLS step solves a new weighted
problem. In our benchmark, PyFixest reuses the first preconditioner as the weights change.
Our results support factor-pair preconditioning when pairwise $lambda_2$ is small, a fit
is unexpectedly slow, or an estimator requires several fixed-effect solves.

#set heading(numbering: none)
#pagebreak()

= Appendix A: Factor-Pair Schwarz Algorithm

Algorithm 1 implements the construction shown in @fig-pair-strategy.

#align(center)[
  #block(
    width: 96%,
    inset: (x: 0.95em, y: 0.75em),
    fill: rgb("#f7f8fa"),
    stroke: 0.35pt + rgb("#d8dee8"),
    radius: 4pt,
  )[
    #text(size: 8.7pt)[
      #align(center)[#strong[Algorithm 1. Factor-Pair Schwarz Preconditioner]]

      #v(0.25em)
      #align(left)[
        #strong[Inputs]
        - Observation-level factor codes for $Q$ fixed-effect dimensions.
        - Diagonal weights $W$ and local solver settings.
        - Current LSMR residual $r$, indexed by the fixed-effect coefficients.

        #strong[Preconditioner setup]
        - Enumerate all unordered factor pairs $(q,r)$ with $q < r$.
        - For each pair, build weighted count blocks $G_(q q)$, $G_(r r)$ and the weighted
          cross-tabulation $C_(q r)$.
        - Split the induced bipartite graph into connected components and create one
          local pair problem (a Schwarz subdomain) $s$ per component.
        - If fixed-effect level $j$ appears in $c_j$ subdomains, store the partition
          weight $omega_j = 1 / sqrt(c_j)$.
        - For each subdomain, sign-flip one side to obtain a local Laplacian, remove the
          constant part that cannot be identified within the component from the local
          right-hand side, and build a zero-mean solve.
        - Use a Schur-complement local solver: small reduced systems are solved directly,
          while larger reduced symmetric diagonally dominant (SDD) or Laplacian systems
          are solved with randomized approximate Cholesky.

        #strong[Application during LSMR]
        - Initialize $z = 0$.
        - For each subdomain $s$, form $h_s = tilde(D)_s R_s r$.
        - Compute the approximate local correction $u_s approx A_s h_s$ on the
          zero-mean subspace.
        - Accumulate $z <- z + R_s' tilde(D)_s u_s$.
        - Return $z = M^(-1) r$.
      ]
    ]
  ]
]

#pagebreak()

= Appendix B: Additional Benchmark Results

We report the detailed AKM timings behind @fig-gap-runtime and results for synthetic and
real data from the Correia collection. The script `scripts/paper_results.py` generates
every table in Appendices B and C from recorded benchmark output; none of the numbers are
entered by hand.

== Controlled AKM Benchmarks

Panel A reports the exact package-default timings used in the left column of
@fig-gap-runtime. Panel B reports the other package-default PyFixest LSMR
configurations, so its factor-pair column repeats Panel A's PyFixest factor-pair
result.

=== Worker Mobility

#v(0.3em)

#text(size: 8.9pt)[
#strong[Mobility benchmark: package defaults (Panel A).]
#include "generated/tables/akm_mobility_defaults.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds and cover each regression from
  model setup through coefficient estimation. Each package uses its default settings.
  Each design has 1 million observations, one covariate, and worker, firm, and year fixed
  effects. Move probability is the probability that a worker changes firms between
  adjacent periods. In the $lambda_2$ (share) column, smaller values mean weaker
  worker-firm connectivity. We compute $lambda_2$ after removing singleton fixed-effect
  levels. For a disconnected pair graph, we report the smallest component value; the
  number in parentheses is that component's share of retained observations. If only $k$ of the $n$
  planned runs return an estimate, $t (k/n)$ gives their median time $t$. A dash means
  that no timing is available. `capped (0/n)` means that no run finishes before the
  10,000-iteration limit; @fig-gap-runtime plots the median elapsed time as a lower bound.
  `failed (0/n)` marks a failure other than reaching that limit.]
]

#v(0.35em)

#text(size: 8.9pt)[
#strong[Mobility benchmark: PyFixest LSMR configurations (Panel B).]
#include "generated/tables/akm_mobility_lsmr.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds and cover each regression from
  model setup through coefficient estimation. Each mobility design has 1 million observations, one covariate, and worker, firm, and year
  fixed effects. Move probability is the probability that a worker changes firms between
  adjacent periods. The columns show PyFixest LSMR with no preconditioner, with diagonal
  scaling by fixed-effect group counts, and with factor-pair preconditioning. The
  factor-pair column repeats Panel A. In the $lambda_2$ (share) column, smaller values
  mean weaker worker-firm connectivity. We compute $lambda_2$ after removing singleton
  fixed-effect levels. For a disconnected pair graph, we report the smallest component
  value; the number in parentheses is that component's share of retained observations.
  `capped (0/n)` means that no run finishes
  before the 10,000-iteration limit.]
]

#pagebreak()

=== Sorting Among Movers

#v(0.3em)

#text(size: 8.9pt)[
#strong[Sorting benchmark: package defaults (Panel A).]
#include "generated/tables/akm_sorting_defaults.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds and cover each regression from
  model setup through coefficient estimation. Each package uses its default settings.
  Each design has 1 million observations, one covariate, and worker, firm, and year fixed effects. Every worker
  changes firms between adjacent periods; the parameter $rho$ controls how strongly
  workers sort across firms. In the $lambda_2$ (share) column, smaller values mean weaker
  worker-firm connectivity. We compute $lambda_2$ after removing singleton fixed-effect
  levels. For a disconnected pair graph, we report the smallest component value; the
  number in parentheses is that component's share of retained observations. If only $k$ of the $n$ planned runs return an estimate, $t (k/n)$ gives
  their median time $t$. A dash means that no timing is available. `capped (0/n)` means
  that no run finishes before the 10,000-iteration limit; @fig-gap-runtime plots the
  median elapsed time as a lower bound. `failed (0/n)` marks a failure other than
  reaching that limit.]
]

#v(0.35em)

#text(size: 8.9pt)[
#strong[Sorting benchmark: PyFixest LSMR configurations (Panel B).]
#include "generated/tables/akm_sorting_lsmr.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds and cover each regression from
  model setup through coefficient estimation. Each sorting design has 1 million observations, one covariate, and worker, firm, and year
  fixed effects. Every worker changes firms between adjacent periods; the parameter
  $rho$ controls how strongly workers sort across firms. The columns show PyFixest LSMR
  with no preconditioner, with diagonal scaling by fixed-effect group counts, and with
  factor-pair preconditioning. The factor-pair column repeats Panel A. In the $lambda_2$
  (share) column, smaller values mean weaker worker-firm connectivity. We compute
  $lambda_2$ after removing singleton fixed-effect levels. For a disconnected pair graph,
  we report the smallest component value; the number in parentheses is that component's
  share of retained observations. `capped (0/n)` means that no
  run finishes before the 10,000-iteration limit.]
]

#pagebreak()

== Correia Benchmark Collection

=== Synthetic Designs

The controlled AKM benchmarks in Section 7 vary mobility and sorting directly. The
public and reproducible Correia HDFE benchmark collection covers a broader set of graph
shapes: complete bipartite matching, uniform random matching with varying degrees of
connectivity, assortative matching, and a small path-like design. It provides an
independent reference set for comparing the same software implementations outside the AKM
generator.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia synthetic benchmarks.]
#include "generated/tables/correia_synthetic.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds for OLS regressions using each
  package's defaults. We planned three runs per cell. Each package applies its own
  default treatment of fixed-effect levels observed only once. In the $lambda_2$ (share)
  column, smaller values mean weaker connectivity between `id1` and `id2`. We compute
  $lambda_2$ after removing singleton fixed-effect levels. For a disconnected pair graph,
  we report the smallest component value; the number in parentheses is that component's
  share of retained observations. If only $k$ of the three planned runs return an
  estimate, $t (k/3)$ gives their median time $t$.
  A dash means that no timing is available. `capped (0/3)` means that no run finishes
  before the iteration limit; `failed (0/3)` marks a failure other than reaching that limit.
  `synthetic-zigzag` is omitted because the default MAP implementations reach the
  10,000-iteration limit.]
  ]

The synthetic collection covers complete, uniform, assortative, and path-like matching
patterns. The no-preconditioner and diagonal LSMR results will be added once the
package-default Correia run is complete; the unfilled cells are not used for rankings.

=== Real Data

The Correia collection also includes real benchmark data. Synthetic data sets match the
overall shape of empirical co-occurrence graphs but smooth away several irregularities
that often drive runtime. These include a few units that appear far more often than the
rest, thin connections between otherwise dense groups, many small disconnected pieces,
and interactions between identifiers that go beyond a single two-way pair. The
empirical datasets contain these features. They therefore test the solvers in
conditions that controlled data-generating processes only approximate.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia real-data benchmarks.]
#include "generated/tables/correia_real.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are in seconds for OLS regressions using each
  package's defaults. We planned three runs per cell. Each package applies its own
  default treatment of fixed-effect levels observed only once. In the $lambda_2$ (share)
  column, smaller values mean weaker connectivity between `id1` and `id2`. We compute
  $lambda_2$ after removing singleton fixed-effect levels. For a disconnected pair graph,
  we report the smallest component value; the number in parentheses is that component's
  share of retained observations. If only $k$ of the three planned runs return an
  estimate, $t (k/3)$ gives their median time $t$.
  A dash means that no timing is available. `capped (0/3)` means that no run finishes
  before the iteration limit; `failed (0/3)` marks a failure other than reaching that
  limit. A small $lambda_2$ in a
  small component, as for `directors`, need not make the full sample difficult for MAP.]
  ]

No single pairwise statistic captures all the irregular components in the real-data
collection. The `directors` component attaining the smallest reported $lambda_2$ covers 30
percent of the observations. The no-preconditioner and diagonal LSMR columns await the
package-default Correia run. Pairwise $lambda_2$ remains a diagnostic and does not serve as a
selection rule.

#pagebreak()

= Appendix C: Coefficient Estimates and Memory Use

== Comparing Coefficient Estimates

We compare the coefficient estimates because MAP and LSMR use different rules for
deciding when to stop. Even when both methods are implemented correctly, their estimates
need only agree within the requested tolerances.

We use the 100K-observation versions of the simple and difficult `fixest` benchmark
designs. All methods should agree closely on the simple design. In the difficult design,
some remaining errors shrink much more slowly, so small differences in stopping rules
are more likely to appear. Worker-firm $lambda_2$ is approximately 0.622 in the simple
design and $6.50 times 10^(-4)$ in the difficult design. The table compares estimates of one
regression coefficient across four implementations.

#v(0.4em)

#text(size: 9.2pt)[
#strong[Coefficient estimates across implementations (100,000 observations, three fixed effects, one covariate).]
#include "generated/tables/agreement.typ"
#v(0.25em)
#text(size: 8.2pt)[#emph[Note:] Each row comes from one OLS regression fitted with the
package's default settings. $hat(beta)_1$ is the slope coefficient on `x1`. The final
column reports its absolute difference from the PyFixest MAP estimate. The dash in the
PyFixest MAP row marks this reference estimate. `within` is PyFixest LSMR with
factor-pair preconditioning.]
]

On the simple design, the largest coefficient difference is
#result_agreement_simple_max. On the difficult design it is
#result_agreement_difficult_max. The packages use different criteria to decide when their
iterative algorithms have converged. Small differences between their estimates are
therefore expected on the difficult design.

== Memory Use

MAP uses little memory: a complete pass needs only the current residuals and per-level group
sums. A factor-pair preconditioner, by contrast, must retain the pair structure between
iterations. We therefore measure how much additional memory factor-pair LSMR requires
relative to MAP.

We record peak resident set size (peak RSS), the largest amount of physical memory used
by the process during a run, on the simple and difficult fixed-effect designs. The main
storage terms are the data matrix, the fixed-effect encodings, and the reusable
factor-pair objects. We do not compare `fixest` and `FixedEffectModels.jl` here because
peak RSS across languages also reflects R and Julia runtime overhead, data-loading
choices, garbage collection, and package internals. Instead, we compare the
preconditioned Rust implementation with Rust MAP inside the same Python package. The
surrounding regression code is shared; only the demeaning strategy differs. Both
implementations run in isolated processes and report peak RSS through `ru_maxrss`.

#v(0.4em)

#text(size: 8.9pt)[
#strong[Memory footprint (three fixed effects, one covariate).]
#include "generated/tables/memory.typ"
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] For each regression, we record the largest amount of
		physical memory used by an isolated Python process, in MiB ($2^20$ bytes). The table
		compares PyFixest OLS regressions using MAP or factor-pair LSMR. Each regression has
		one covariate and worker, firm, and year fixed effects. The results cover the simple
		and difficult designs at 100,000 and 1 million observations; they do not show how
		memory changes with graph structure. In the $lambda_2$ (share) column, smaller values
		mean weaker worker-firm connectivity. We compute $lambda_2$ after removing singleton
		fixed-effect levels. For a disconnected pair graph, we report the smallest component
		value; the number in parentheses is that component's share of retained observations.]
		]

At 100K observations, the preconditioner adds #result_memory_100k_overhead. At 1M
observations the overhead is #result_memory_1m_overhead, but it remains modest relative
to the full panel data footprint. The additional storage holds factor-pair
co-occurrences, partition weights, and local approximate Cholesky factors.

#pagebreak()

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

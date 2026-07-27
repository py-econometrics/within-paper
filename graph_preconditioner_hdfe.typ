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
#set par(
  justify: true,
  leading: 1.06em,
  spacing: 1.12em,
  first-line-indent: 1em,
)
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
#let table-rule = rgb("#7b8494")
#let table-light-rule = rgb("#d8dee8")
#let table-head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]
#let miss = text(fill: rgb("#777777"))[--]
#let dg(body) = text(fill: rgb("#2563eb"), body)
#let cr(body) = text(fill: rgb("#c2410c"), body)
#import "generated/paper_values.typ": (
  result_ols_difficult_gap, result_ols_difficult_rust_map,
)
#import "generated/paper_values.typ": (
  result_ols_difficult_fem, result_ols_difficult_fixest,
  result_ols_difficult_within,
)
#import "generated/paper_values.typ": (
  result_correia_enron_fem, result_correia_enron_within,
)
#import "generated/paper_values.typ": (
  result_ppml_difficult_three_fixest, result_ppml_simple_range,
)
#import "generated/paper_values.typ": (
  result_ppml_difficult_three_glfem, result_ppml_difficult_three_within,
)
#import "generated/paper_values.typ": (
  result_agreement_difficult_gap, result_agreement_simple_gap,
)
#import "generated/paper_values.typ": (
  result_agreement_difficult_max, result_agreement_simple_max,
)
#import "generated/paper_values.typ": (
  result_setup_simple_setup, result_setup_simple_share,
  result_setup_simple_solve,
)
#import "generated/paper_values.typ": (
  result_setup_difficult_setup, result_setup_difficult_share,
  result_setup_difficult_solve,
)
#import "generated/paper_values.typ": (
  result_directors_component_share, result_memory_100k_overhead,
  result_memory_1m_overhead,
)
#import "generated/paper_values.typ": (
  result_zigzag_fem, result_zigzag_within,
)

#align(center)[
  #set par(justify: false)
  #text(size: 18.5pt, weight: "bold")[Graph Preconditioning for]

  #v(0.18em)
  #text(size: 18.5pt, weight: "bold")[High-Dimensional Fixed Effects
    Regression]

  #v(0.65em)
  #text(size: 10.5pt)[Alexander Fischer#footnote[trivago] and Kristof
    Schröder#footnote[appliedAI Institute for Europe GmbH]]

  #v(0.4em)
  #text(size: 9.5pt)[Draft: June 2026]
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
      #text(weight: "bold")[Abstract.] The Method of Alternating
      Projections (MAP) is the de facto standard algorithm for estimating
      high-dimensional fixed-effect regressions. Although MAP is often
      fast, it can converge slowly when the fixed-effect structure is
      poorly connected, as in matched employer-employee panels with low
      worker mobility between firms. We relate MAP's convergence to the
      connectivity of the weighted bipartite graph formed by the levels of
      a pair of fixed-effect dimensions and propose a novel
      graph-preconditioned Krylov solver. Our preconditioner is constructed
      from sparse approximate Cholesky factorizations of the weighted
      bipartite graph Laplacian associated with each pair of fixed-effect
      dimensions. We show that graph preconditioning can substantially
      improve convergence on poorly connected fixed-effect graphs but that
      its setup time can dominate total runtime on well-connected graphs.
    ]
  ]
]

#v(0.25em)
#align(center)[
  #text(size: 9.1pt)[
    #strong[Keywords:] high-dimensional fixed effects; alternating
    projections; preconditioning; matched employer-employee data;
    computational econometrics
  ]
]
#align(center)[
  #text(size: 9.1pt)[#strong[JEL codes:] C55; C63; C81; C87; J31]
]

= Introduction

Fixed-effect regressions are ubiquitous in applied econometrics with
roughly half of published research in top economics and finance journals
mentioning "fixed effects" @goldsmith2026tracking. Labor economists use
worker and firm fixed effects to separate worker heterogeneity from firm
wage premia; health economists study physician practice styles with
individual-physician and region fixed effects in mover designs; and
education researchers study models with school, student, teacher, or
student-teacher fixed effects.

The standard computational starting point for estimating these regressions
efficiently is the Frisch-Waugh-Lovell (FWL) theorem @frisch1933
@lovell1963. FWL reduces the fixed-effect estimation problem to
"residualizing" dependent and independent variables against the fixed
effects, and then running a low-dimensional regression on the residualized
variables.

The workhorse method for these fixed-effect residualizations is the Method
of Alternating Projections (MAP), also known as iterative demeaning or the
"Zig-Zag" algorithm @guimaraes2010 @gaure2013. Most leading software
implementations of fixed-effect regression, such as `reghdfe` in Stata
@reghdfe @correia2017, `fixest` @berge2026fixest in R, or PyFixest in
Python @pyfixest, use methods based on iterative demeaning, often with
acceleration.

MAP cycles over the fixed-effect dimensions and, for each variable of
interest, subtracts the mean within every level of the current dimension.
Because demeaning along one fixed-effect dimension (for example workers)
changes the group means for the other dimensions (for example firms or
years), the procedure repeats these cycles until convergence. In other
words, MAP residualizes with respect to one fixed-effect dimension at a
time but does not directly exploit the co-occurrence of levels of different
fixed-effect dimensions. In a worker-firm panel, for example, low worker
mobility between firms can make some directions in the worker-firm
fixed-effect structure nearly collinear, adversely affecting MAP's
convergence.

We show that this problem is related to the connectivity of the graph
formed by the fixed effects: The levels of a pair of fixed-effect
dimensions, for example workers and firms, are the nodes of a bipartite
graph where two nodes are connected if there is an observation taking on
the respective fixed-effect levels (for example, a worker working at a
particular firm). The bipartite graph is weighted by the number of
occurrences of a given pair of fixed-effect levels.

In this paper, we use these pairwise bipartite graphs to construct an
additive Schwarz preconditioner for LSMR @fong2011, a Krylov least-squares
solver. The preconditioner provides an inexpensive approximation to the
inverse of the fixed-effect Gramian, improving the conditioning of the
Krylov iteration without changing the least-squares solution.#footnote[The
  Julia implementation of fixed-effect regression, `FixedEffectModels.jl`
  @fixedeffectmodels, uses the same Krylov solver but only with diagonal
  preconditioning. Diagonal preconditioning ignores the off-diagonal
  co-occurrence structure; our contribution is the preconditioner, not the
  use of LSMR.] Algebraically, each off-diagonal block of the fixed-effect
Gramian records the edge weights of one pairwise bipartite graph. When
combined with the corresponding diagonal count blocks and subjected to a
sign flip for one fixed-effect dimension, the resulting pair block is a
graph Laplacian @correia2017. We use sparse approximate Cholesky
factorizations @spielman2014 @gao2025 to obtain efficient approximate
solves for these pairwise Laplacians and combine their corrections in the
Schwarz preconditioner.

In our benchmarks, graph preconditioning is substantially faster than MAP
on poorly connected designs. On well-connected designs, however, its setup
time can dominate total runtime in which case MAP and diagonally
preconditioned Krylov methods are faster.

The rest of the paper is organized as follows. @sec:absorbing sets up the
fixed-effect absorption problem, and @sec:akm introduces the AKM @akm1999
model as our running example. @sec:gramian develops the graph structure of
the fixed-effect Gramian, and @sec:map-connectivity connects this structure
to the convergence behavior of MAP. @sec:schwarz-preconditioner builds up
the graph preconditioner, starting from a general discussion of
preconditioning and culminating in the construction of the graph-based
preconditioner. Section 7 reports benchmarks on runtime, memory, and
numerical equivalence; Section 8 describes the software implementation of
the new algorithm; and Section 9 concludes.

= Absorbing Fixed Effects#footnote[Researchers employ several names for
  this operation: "absorbing fixed effects", "demeaning", "residualizing",
  or applying the "within transformation". We use these terms
  interchangeably throughout.]<sec:absorbing>

We focus on the linear model

$ y = X beta + D alpha + epsilon, $<eq:model>

where $X$ denotes a set of covariates and $D$ is the fixed-effect design
matrix given by
$
  D_(i j) = cases(
    1 & "if observation" i "belongs to fixed-effect level" j",",
    0 & "otherwise.",
  )
$
In high-dimensional applications, $D$ may have hundreds of thousands or
millions of columns, and forming or inverting the full system in
$[X quad D]$ might prove computationally infeasible.

Fortunately, the Frisch-Waugh-Lovell (FWL) theorem @frisch1933 @lovell1963
allows us to estimate $beta$ in two tractable steps without forming
$[X quad D]$ or inverting its cross product. First, the outcome and each
covariate are residualized against the fixed effects by regressing $y$ and
each column of $X$ on $D$. Second, the residualized outcome $tilde(y)$ is
regressed on the residualized covariates $tilde(X)$. The FWL theorem
implies that the resulting slope of this regression equals the coefficient
$beta$ of the full model @eq:model.

To fix notation, we denote by $M_D$ the linear operator that maps a
variable to its residual when regressed on the fixed effects $D$, i.e.,
$tilde(y) = M_D y$ and $tilde(X) = M_D X$. The coefficient of interest
$beta$ is then recovered by regressing the residualized outcome on the
residualized regressors
$
  tilde(y) = M_D y, quad tilde(X) = M_D X, quad
  hat(beta) = (tilde(X)' W tilde(X))^(-1) tilde(X)' W tilde(y),
$
where $W$ is a diagonal matrix of weights. The residualization step is the
weighted least-squares projection of the outcome and each covariate in $X$
onto the column space of the fixed-effect design matrix $D$. For a
right-hand side $mu$, i.e., the outcome $y$ or a column of $X$, we solve

$ hat(alpha)_mu in arg min_alpha || D alpha - mu ||_W^2, $ <eq:demean-ls>

The first-order condition for @eq:demean-ls,
$D' W (D hat(alpha)_mu - mu) = 0$, can be written as


$
  G hat(alpha)_mu = D' W mu, quad G = D' W D,
$ <eq:fwl-normal>
where $G$ is the fixed-effect Gramian. Solving @eq:fwl-normal for
$hat(alpha)_mu$, we obtain the residualized right-hand side as
$tilde(mu) = mu - D hat(alpha)_mu$.#footnote[
  With a full set of fixed-effect dummies, the columns of $D$ are linearly
  dependent and $hat(alpha)_mu$ is not unique. To report individual fixed
  effects, one must choose a normalization, usually by dropping reference
  categories. The normalization does not change the fitted value, the
  residual, or the FWL slope. Throughout, inverses are taken after
  normalizing each connected component.] We note that each FWL
residualization uses the same fixed-effect Gramian $G$, and only the
right-hand side $mu$ changes as we move from the outcome to the covariates.
The cost of residualization therefore depends on the structure of the
Gramian $G$.

We illustrate this structure with the AKM worker-firm model in the next
section before explaining the graph interpretation of the fixed-effect
Gramian $G$ in @sec:gramian and how it governs MAP convergence in
@sec:map-connectivity.

= A Running Example: The AKM Model <sec:akm>

One of the most prominent examples of high-dimensional fixed-effect
regressions is the Abowd-Kramarz-Margolis (AKM) model @akm1999. The AKM
model separates persistent worker heterogeneity from firm wage premia using
workers who move across firms. We write the AKM regression equation as

$
  y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t),
$

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed
effect for the firm employing worker $i$ at time $t$, and $phi_t$ is a time
fixed effect.

The AKM specification has a natural graph representation. Workers and firms
are nodes in a bipartite graph, with an edge connecting a worker to the
firm they are employed by. Each connection is weighted by the count of
observations of a given worker-firm pair across years. Stayers---workers
who never change their employer---add weight to existing worker-firm links
but do not connect differnt firms. @fig-connectivity contrasts a
well-connected mobility graph with one that fragments under strong sorting.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 50%),
  caption: [Worker-firm graph connectivity. When mobility is high, many
    paths connect firms. With low mobility and strong sorting, the graph
    breaks into nearly separate clusters joined only by narrow bridges.],
) <fig-connectivity>

Worker mobility determines the graph's connectedness and enables separate
identification of worker and firm effects. For a worker observed at only
one firm, a high wage could reflect an unusually productive worker, a
high-wage firm, or both. A worker earning high wages across multiple firms
provides evidence of a worker effect, while different workers earning high
wages at the same firm provide evidence of a firm premium. The more
cross-firm comparisons the data contain, the easier it becomes to
separately identify worker effects and firm premia. This is why the
identifying variation generated by movers corresponds to the connectedness
of the fixed-effect graph.

= The Graph Structure of the Gramian <sec:gramian>

The bipartite graph of worker and firm connections introduced in Section 3
has an algebraic representation in the block structure of the Gramian
$G = D' W D$ @correia2017. Suppose that the columns of $D$ are ordered as
worker levels, firm levels, and year levels. Then

$
  G = mat(
    G_(W W), C_(W F), C_(W Y);
    C_(W F)', G_(F F), C_(F Y);
    C_(W Y)', C_(F Y)', G_(Y Y)
  ).
$

The #dg[diagonal blocks] $#dg[$G_(W W)$]$, $#dg[$G_(F F)$]$, and
$#dg[$G_(Y Y)$]$ contain weighted counts for workers, firms, and years. An
observation belongs to one level of each factor, so these blocks are
diagonal; solving them requires only division by group counts.


The #cr[off-diagonal blocks] are cross-tabulations: the worker-firm block
$#cr[$C_(W
F)$]$ records how often worker $i$ is observed at firm $j$, and the
worker-year and firm-year blocks have analogous interpretations.

As a small example, we construct a worker-firm panel and populate its
Gramian. For simplicity, we ignore any regression weights and set $W = I$.

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
  caption: [Worker-firm projection of the example panel. Worker $W_1$ is a
    mover; workers $W_2$ and $W_3$ are stayers.],
) <fig-toy-projection>

@fig-toy-projection plots the worker-firm projection of this panel. Worker
$W_1$ is observed at both $F_1$ and $F_2$ and creates a link between the
two firms; in AKM terms, $W_1$ is a mover. Worker $W_2$ stays at $F_1$ for
two periods, and $W_3$ stays at $F_2$ for two periods. Both are stayers.

The diagonal blocks are count matrices. In this example, each worker is
observed twice, each firm three times, and each year three times, so

$
  G_(W W) = mat(2, 0, 0; 0, 2, 0; 0, 0, 2), quad
  G_(F F) = mat(3, 0; 0, 3), quad
  G_(Y Y) = mat(3, 0; 0, 3).
$

The off-diagonal blocks are cross-tabulations between factors. The
worker-firm block is

$
  C_(W F) = mat(
    1, 1;
    2, 0;
    0, 2
  ).
$

The first row indicates that worker $W_1$ appears once at firm $F_1$ and
once at firm $F_2$. Workers $W_2$ and $W_3$ are stayers, appearing twice at
$F_1$ and $F_2$, respectively. The worker-year block is

$
  C_(W Y) = mat(
    1, 1;
    1, 1;
    1, 1
  ).
$

Each worker is observed once in each year. The firm-year block is

$
  C_(F Y) = mat(
    2, 1;
    1, 2
  ).
$

Firm $F_1$ appears twice in year $Y_1$ and once in year $Y_2$; firm $F_2$
has the opposite pattern. Combining the diagonal count blocks and the
off-diagonal cross-tabulation blocks yields the full Gramian.

With column order $(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$, the full Gramian
is

$
  G = mat(
    augment: #(hline: (3, 5), vline: (3, 5), stroke: 0.4pt + rgb("#b0b8c4")),
    2, 0, 0, 1, 1, 1, 1;
    0, 2, 0, 2, 0, 1, 1;
    0, 0, 2, 0, 2, 1, 1;
    1, 2, 0, 3, 0, 2, 1;
    1, 0, 2, 0, 3, 1, 2;
    1, 1, 1, 2, 1, 3, 0;
    1, 1, 1, 1, 2, 0, 3
  ).
$

The worker-firm submatrix stores the bipartite graph algebraically. The
diagonal entries are worker and firm counts, and the entries of $C_(W F)$
are edge multiplicities between workers and firms. After flipping the sign
of $C_(W F)$, this submatrix is a graph Laplacian: its off-diagonal entries
are non-positive, and every row sums to zero because each diagonal count
cancels the off-diagonal observation counts in the same row. For a worker,
the diagonal entry is the number of observations for that worker, while the
off-diagonal entries show how those observations are distributed across
firms; firm rows have the analogous interpretation with observations summed
over workers.


$
  L_(W F) = mat(
    augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
    2, 0, 0, -1, -1;
    0, 2, 0, -2, 0;
    0, 0, 2, 0, -2;
    -1, -2, 0, 3, 0;
    -1, 0, -2, 0, 3
  ).
$

The same Laplacian construction applies to any pair of fixed effects, and
the preconditioner of Section 6 builds on these pairwise Laplacians. Before
turning to it, however, we introduce the method of alternating projections,
which avoids forming the full Gramian $G$ by working only on the diagonal
worker, firm, and year blocks.


= Alternating Projections and Graph Connectivity <sec:map-connectivity>

The workhorse algorithm for multi-way fixed effects is the Method of
Alternating Projections (MAP), also referred to as iterative demeaning or
the "zig-zag" algorithm @guimaraes2010 @gaure2013. Many packages employ MAP
or its variants, frequently combined with accelerations @berge2018
@correia2017, such as the Irons-Tuck extrapolation used by `fixest`
@irons1969 @berge2026fixest. Sections 3 and 4 introduced the fixed-effect
graph and its algebra; this section turns to MAP itself and shows how that
graph geometry governs its convergence rate.

MAP solves the FWL residualization problem by iterating over one fixed
effect at a time. In the worker-firm-year model, MAP first subtracts worker
means from the current residual, then firm means from the updated residual,
then year means, and so on until convergence.

We write the FWL normal equations (see @eq:fwl-normal) in block form with
$D = [D_W quad D_F quad D_Y]$ as

$
  mat(
    G_(W W), C_(W F), C_(W Y);
    C_(W F)', G_(F F), C_(F Y);
    C_(W Y)', C_(F Y)', G_(Y Y)
  ) mat(alpha_W; alpha_F; alpha_Y)
  = mat(D_W' W mu; D_F' W mu; D_Y' W mu).
$

Equivalently,

$ G_(W W) alpha_W + C_(W F) alpha_F + C_(W Y) alpha_Y = D_W' W mu, $

$ C_(W F)' alpha_W + G_(F F) alpha_F + C_(F Y) alpha_Y = D_F' W mu, $

$ C_(W Y)' alpha_W + C_(F Y)' alpha_F + G_(Y Y) alpha_Y = D_Y' W mu. $

Each equation can be rearranged to express one block of effects conditional
on the others. Using $C_(W F) = D_W' W D_F$ and $C_(W Y) = D_W' W D_Y$ to
factor $D_W' W$ out of the right-hand side, the first equation becomes

$ G_(W W) alpha_W = D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Because $G_(W W)$ is a diagonal matrix whose entries are workers' total
observation weights, solving for $alpha_W$ divides each worker's weighted
partial residual by its total observation weight. We apply the same
rearrangement to the second equation to obtain the firm equation

$ G_(F F) alpha_F = D_F' W (mu - D_W alpha_W - D_Y alpha_Y), $

and the year equation is analogous. Because $G_(W W)$, $G_(F F)$, and
$G_(Y Y)$ are all diagonal, each of the three equations is solved by
computing a weighted group mean in a single pass over observations.

MAP uses this diagonal structure iteratively. Holding the other effects
fixed, it updates the worker effects from the current partial residual,
then repeats the same step for firms and years. Each sweep therefore cycles
through the fixed-effect dimensions, subtracting the weighted group mean of
the current partial residual for the factor being updated.

The cross-tabulation blocks $C_(W F)$, $C_(W Y)$, and $C_(F Y)$ enter the
algorithm only indirectly. For instance, the worker update is computed from
the partial residual $mu - D_F alpha_F - D_Y alpha_Y$, while the firm
update is computed from $mu - D_W alpha_W - D_Y alpha_Y$. Worker-firm,
worker-year, and firm-year links are thus not solved as coupled
subproblems; their effect propagates through the residual that one block
update transmits to the next.

This strategy is effective when the graph is well connected. In a
high-mobility worker-firm panel, many workers move across firms, so that
worker and firm effects can be compared through many overlapping employment
histories. A high wage at one firm can then be related to wages earned by
the same workers at other firms, and a worker update rapidly alters the
information available to the next firm update, and vice versa.

When mobility is sparse, sorting is strong, or one factor is nearly nested
in another, MAP may require many sweeps because mover comparisons enter
only through repeated residual updates. Each sweep is cheap: the diagonal
block solves are one-pass group means.

The same graph perspective gives a simple diagnostic for MAP difficulty in
a fixed-effect structure. For any pair of fixed effects $(q,r)$, we form
the normalized cross-tabulation
$H_(q r) = G_(q q)^(-1/2) C_(q r) G_(r r)^(-1/2)$ and let
$rho_(q r) = sigma_2(H_(q r))^2$ be the square of its largest nontrivial
singular value, equivalently the largest nontrivial eigenvalue of
$H_(q r)' H_(q r)$. Within a connected component, the largest singular
value is always one, regardless of how well connected the component is. We
use the next-largest singular value and report the spectral gap
$1 - rho_(q r)$ in the benchmarks below. This gap measures how well
connected the factor-pair graph is.#footnote[When the graph has more than
  one connected component, we compute the gap on each component and report
  the smallest, together with its observation share.] Gaps near zero signal
sparse mobility or near-nesting, the settings in which MAP converges
slowly; larger gaps indicate better connected factor-pair graphs. For the
worker-firm example of Section 4, the gap is $1/3$.#footnote[In that
  example, $G_(W W) = "diag"(2,2,2)$, $G_(F F) = "diag"(3,3)$, and
  $C_(W F) = mat(1, 1; 2, 0; 0, 2)$, so
  $H_(W F) = G_(W W)^(-1/2) C_(W F) G_(F F)^(-1/2) = 1 / sqrt(6) mat(1, 1; 2, 0; 0, 2)$.
  The graph is connected, and $H_(W F)' H_(W F) = 1 / 6 mat(5, 1; 1, 5)$
  has eigenvalues $1$ and $2/3$. After dropping the unit eigenvalue,
  $rho_(W F) = 2/3$ and the gap is $1/3$.]

= The Factor-Pair Schwarz Preconditioner <sec:schwarz-preconditioner>

== Preconditioners

The previous section attributed MAP's slow convergence to thin connections
in the fixed-effect graph. A solver that receives this connectivity
information directly should converge faster. MAP has no natural way to
accept it: its update rule is fully determined by the list of absorbed
factors, and the cross-tabulations enter only through the residual that one
update passes to the next. Krylov solvers, by contrast, take an additional
input, the preconditioner, and this input is where we place the graph
structure. We therefore replace factor-by-factor demeaning via MAP with
LSMR @fong2011, an iterative least-squares algorithm that improves an
initial guess through repeated residual corrections. The change of solver
gains little by itself: a poorly conditioned Gramian $G$ slows LSMR just as
a poorly connected graph slows MAP. The core acceleration stems from
building a good preconditioner, which we focus on in the remainder of this
section.

How fast LSMR converges depends on the conditioning of $G$. When $G$ is
well conditioned, LSMR shrinks every component of the residual at a
comparable rate. When $G$ is poorly conditioned, LSMR removes some
components after a few iterations, whereas components that correspond to
weak links, sparse mobility, or near nesting in the fixed-effect graph
shrink much more slowly; the iteration then spends most of its steps on
these slow components. A preconditioner counteracts this imbalance: it
changes the coordinates of the linear system so that slow and fast
components decay at more similar rates, without changing the least-squares
solution.

The ideal preconditioner is $G^(-1)$. If $M^(-1) = G^(-1)$, then the
preconditioned operator is

$ M^(-1) G = G^(-1) G = I. $ <eq:ideal-preconditioner>

In exact arithmetic, the Krylov iteration would then recover the solution
after one correction, because the system has no slow directions left to
remove. To form $G^(-1)$ we would have to solve the fixed-effect normal
equations themselves, so the identity case serves as a benchmark rather
than an implementable preconditioner. A useful preconditioner must
approximate enough of $G^(-1)$ to remove the slow directions of the
iteration, while its construction and repeated application must amortize
over the iterations it saves.#footnote[LSMR never forms $M^(-1) G$
  explicitly, nor $G$ itself. The iteration requires only products with $D$
  and $D'$ and applications of $M^(-1)$, supplied as linear operators.]

== From Block Elimination to the Diagonal Preconditioner

The block structure of $G^(-1)$ shows which parts of the ideal inverse a
feasible preconditioner should retain. For the worker-firm-year AKM model,
the Gramian has the block form

$
  G = mat(
    G_(W W), C_(W F), C_(W Y);
    C_(W F)', G_(F F), C_(F Y);
    C_(W Y)', C_(F Y)', G_(Y Y)
  ),
$

with diagonal weighted-count blocks $G_(W W), G_(F F), G_(Y Y)$ and
off-diagonal cross-tabulations $C_(W F), C_(W Y), C_(F Y)$. Block inversion
via the Schur complement shows how each off-diagonal block enters $G^(-1)$.
For the two-factor block

$ G_2 = mat(G_(W W), C_(W F); C_(W F)', G_(F F)), $

eliminating the worker effects gives the firm-side Schur complement

$ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

The expanded block inverse is given in Appendix A. Applying it requires
solves with $G_(W W)$ and $S$; $C_(W F)$ appears only in matrix products.
$G_(W W)$ is diagonal, so $G_(W W)^(-1)$ is a division by weighted worker
counts. The Schur complement $S$, by contrast, is the firm-side mobility
system that remains after eliminating workers. At the scale of modern
worker-firm register data, solving this system is expensive: exact
factorization creates many additional nonzero entries, separate connected
components require separate normalizations, and weak mobility makes the
remaining directions slow to resolve. $S^(-1)$ therefore carries almost all
the cost of the solve. Block inversion via Schur complements generalizes to
three factors: the closed form has more terms, but every block of $G^(-1)$
still depends jointly on the cross-tabulations $C_(W F), C_(W Y), C_(F Y)$.

The coarsest approximation to $G^(-1)$ keeps only the diagonal count
inverses and drops the Schur-complement corrections,

$ M_("diag")^(-1) = "diag"(G_(W W)^(-1), G_(F F)^(-1), G_(Y Y)^(-1)). $

This preconditioner is a single division by weighted level counts. It is
the diagonal preconditioner used with LSMR in `FixedEffectModels.jl`
@fong2011 @fixedeffectmodels. Diagonal scaling encodes how many
observations a level carries, but not how that level connects to the rest
of the labor market. Those counts can already be useful when employment is
concentrated in a few large firms, such as Novo Nordisk in Denmark or
Samsung in South Korea, because the size differences alone remove an
important source of scale variation. What diagonal scaling does not record
is whether workers at those firms link them broadly to other firms or
remain concentrated in a narrow corner of the mobility graph.

== The Factor-Pair Schwarz Approximation

Additive Schwarz preconditioning adds this missing pairwise connectivity
without solving the full three-factor system @xu1992 @toselli2005. It
splits the fixed-effect problem into smaller overlapping pair problems. In
the AKM case, one problem contains workers and firms, another contains
workers and years, and a third contains firms and years. The worker-firm
problem moves residual information along observed employment links; the
other two pair problems do the same for worker-year and firm-year links. We
then combine the three pair corrections in the full coefficient space. The
preconditioner therefore gives the Krylov iteration the main pairwise
channels of the Gramian, while the outer iteration handles the remaining
three-way coupling.

To make the local subproblems concrete, we first consider the worker-firm
pair. Its local problem is exactly the two-factor block from the Schur
calculation,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)). $

@fig-pair-block places this worker-firm pair block beside the single
diagonal block solved by a factor-level MAP update.

#figure(
  image(solver-img("factor_level_vs_pair_block.svg"), width: 88%),
  caption: [Local operator used by a factor-level MAP update (left) versus
    the factor-pair Schwarz solve (right), shown on the example worker-firm
    panel of Section 4. The factor-level block is the diagonal
    $G_(W W) = "diag"(2,2,2)$. The factor-pair block adds the firm count
    block $G_(F F) = "diag"(3,3)$ and the cross-tabulation $C_(W F)$ in its
    off-diagonal positions; the dashed outline marks the worker-firm
    subdomain on which the local Schwarz solve operates.],
) <fig-pair-block>

The Schur complement incorporates $C_(W F)$, so the local correction uses
the worker-firm mobility pattern that diagonal scaling discards. To place
this correction inside the three-factor problem, let $R_(W F)$ select the
worker and firm entries from the full coefficient vector
$alpha = [alpha_W; alpha_F; alpha_Y]$; its transpose $R_(W F)'$ places the
resulting correction back into the full vector. The diagonal matrix
$tilde(D)_(W F)$ contains the weights that split levels across the pair
problems in which they appear; we call these entries partition-of-unity
weights. The exact worker-firm contribution is

$
  P_(W F)^(-1) =
  R_(W F)' tilde(D)_(W F)
  mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1)
  tilde(D)_(W F) R_(W F).
$

The worker-year and firm-year contributions $P_(W Y)^(-1)$ and
$P_(F Y)^(-1)$ are built the same way from $G_(W W), C_(W Y), G_(Y Y)$ and
$G_(F F), C_(F Y), G_(Y Y)$. With three factors each level appears in
exactly two pair problems. Because the weights act on both sides of each
pair contribution, we set them so that the squared weights on a shared
level sum to one; a level in two pairs then carries $1 / sqrt(2)$. The
exact factor-pair Schwarz preconditioner is the sum of these three
contributions,

$ P^(-1) = P_(W F)^(-1) + P_(W Y)^(-1) + P_(F Y)^(-1). $

The three terms include the worker-firm, worker-year, and firm-year
cross-tabulations separately. They do not solve the simultaneous
worker-firm-year problem; that remaining coupling, together with the error
introduced by splitting shared levels across pairs, is left to the outer
LSMR iteration.

== Approximate Pair Solves via Graph Laplacians

For $P^(-1)$ to serve as the operator $M^(-1)$ inside LSMR, each pair
contribution must be computed without solving a large dense system. For
factor pairs with few levels, we invert the pair block directly. The
example in Section 4 requires only a $4 times 4$ local solve. In modern
worker-firm register data, a pair may contain hundreds of thousands of
levels on each side, making direct factorization impractical. We use the
graph-Laplacian form of the pair block.

For a worker-firm pair, the local Schwarz step solves the pair-Gramian
system

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)) x = u, $

where $u$ is the weighted worker-firm part of the current Krylov residual.
This block is not a graph Laplacian, because its off-diagonal entries
$C_(W F)$ are non-negative. A sign flip removes the obstacle. Let
$T_(W F) = "diag"(I_W, -I_F)$ flip the sign of the firm entries, so that
$T_(W F)^2 = I$. Conjugation by $T_(W F)$ gives

$
  L_(W F) = T_(W F) mat(G_(W W), C_(W F); C_(W F)', G_(F F)) T_(W F)
  = mat(G_(W W), -C_(W F); -C_(W F)', G_(F F)),
$

a weighted bipartite graph Laplacian: symmetric, with non-positive
off-diagonals and zero row sums. Because $T_(W F)^2 = I$, the pair-Gramian
solve follows from the Laplacian solve by the same flip on each side,

$
  mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1) = T_(W F) L_(W F)^(-1) T_(W F).
$

We solve the worker-firm block by reversing the signs of the firm entries
in the right-hand side, solving one Laplacian system, and reversing the
firm signs in the solution. A Laplacian solve requires the right-hand side
to sum to zero within each connected component. The implementation
subtracts the mean within each component before applying the local solve
(Appendix A).

For preconditioning, this local solve need not be exact. The outer LSMR
iteration refines any error left by the preconditioner. We therefore
approximate the Laplacian solve using sparse approximate Cholesky
factorizations from the Laplacian-solver literature @spielman2014 @gao2025.
An exact Cholesky factorization of the pair block creates fill-in:
eliminating a worker inserts entries linking every pair of distinct firms
that worker visited, and these entries accumulate as the elimination
proceeds. In the worst case, the cost approaches the order $k^3$ operations
and order $k^2$ memory of a dense factorization of a $k$-level system. For
large components, the implementation samples the fill-in edges that would
be created during elimination instead of storing all of them. It applies
randomized approximate Cholesky to the resulting sparse Laplacian
@spielman2014 @gao2025. The expected factorization cost is linear in the
number of observed worker-firm links, apart from logarithmic factors. We
denote the corresponding approximate solve in worker-firm coordinates by
$A_(W F)$.

The same construction yields $A_(W Y)$ and $A_(F Y)$ for the other two
pairs. We substitute these approximate inverses into the Schwarz sum to
obtain the implemented preconditioner,

$
  M^(-1) = sum_((q, r)) R_(q r)' tilde(D)_(q r) A_(q r) tilde(D)_(q r) R_(q r).
$

In the worker-firm-year case each factor receives a diagonal contribution
from its two pair subdomains and each off-diagonal correction from the
corresponding pair.

There are two approximations: $P^(-1)$ replaces the full three-factor
inverse with pair solves, and $M^(-1)$ replaces each exact pair solve with
$A_(q r)$. They may change the number of LSMR iterations, but not the
fitted residuals; LSMR still solves the original least-squares problem to
the requested tolerance.

== Implementation

@fig-pair-strategy shows the construction. For each pair of absorbed
factors, a sign change converts the local Gramian to a graph Laplacian. The
implementation handles large connected components with approximate Schur
reduction and sparse approximate Cholesky, then maps the corrections back
to the full coefficient vector and combines them using partition-of-unity
weights.

The outer LSMR iteration uses this preconditioner to shape its search
directions @fong2011 @arridge2014 @yang2024flexible. Once built, the same
preconditioner can residualize the outcome and every covariate. Appendix A
lists the setup and application steps.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Construction of the factor-pair preconditioner. For large pair
    blocks, the algorithm approximates the Schur complement and factors the
    reduced Laplacian with approximate Cholesky. The weighted pair
    corrections form the preconditioner used by LSMR.],
) <fig-pair-strategy>

= Benchmarks

Graph preconditioning should help most when weak connectivity slows MAP. We
test this in controlled and public benchmark designs.

The main runtime tables use package-level regression APIs, matching the
public PyFixest benchmark suite, rather than isolated demeaning kernels.
Each timing covers the full regression: model setup, construction of the
fixed-effect representation, residualization of the outcome and covariates,
and coefficient estimation. Separate tables report memory use and
coefficient agreement.

The runtime tables also report the spectral gap for the relevant factor
pair and the observation share of the component attaining it. Smaller gaps
are associated with slower MAP convergence. With three or more fixed
effects, the gap remains a pairwise diagnostic, not a convergence bound for
the full system.

We compare two MAP backends and two Krylov solvers:

#v(0.35em)

#text(size: 9.2pt)[
  #table(
    columns: (1.0fr, 1.0fr, 2.0fr),
    stroke: 0.35pt + table-light-rule,
    inset: (x: 5pt, y: 3.6pt),
    align: (left, left, left),
    table.hline(stroke: 0.8pt + table-rule),
    table.header(th[Backend], th[Package], th[Algorithm]),
    table.hline(stroke: 0.45pt + table-rule),
    [`rust-map`], [PyFixest], [Unaccelerated Rust MAP.],
    [`fixest`],
    [R `fixest`],
    [Accelerated MAP with Irons-Tuck and other optimizations
      @berge2026fixest.],
    [`FEM.jl`],
    [`FixedEffectModels.jl`],
    [Diagonally preconditioned LSMR @fong2011 @fixedeffectmodels.],
    [`within`],
    [PyFixest],
    [LSMR with factor-pair Schwarz preconditioning.],
    table.hline(stroke: 0.8pt + table-rule),
  )
]

Each CPU result is based on three trials run on an Apple M4 Mac mini with
10 CPU cores and 16 GB of memory running macOS 15.3.1. We report the median
among trials that converged. If only one or two trials converge, the table
gives that count. `failed` means that none of the three trials converged
before reaching the iteration cap.

Stopping rules differ across packages. PyFixest's MAP uses a nominal
$10^(-6)$ tolerance and a 10,000-iteration cap, while `within` uses
$10^(-8)$ and 1,000 iterations. These numbers are not directly comparable
because the packages monitor different convergence quantities. We retain
each package's default rule.

We omit `reghdfe` @reghdfe @correia2017 because Stata is not open source
and we lack a license. Like `fixest`, it uses accelerated MAP.

== Runtime Benchmarks

=== Controlled Synthetic Benchmarks: AKM Mobility and Sorting

These AKM panels hold the rest of the DGP fixed while mobility changes.
Lower mobility leaves fewer workers connecting firms, which slows MAP.

#v(0.4em)

#text(size: 8.9pt)[
  #strong[Mobility benchmark ($n = 1$M).]
  #include "generated/tables/akm_mobility.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one
    covariate, and worker, firm, and year fixed effects. Lower rows reduce
    worker mobility within a 10-period panel, so fewer workers connect
    different firms. A runtime followed by $(k/3)$ is based on $k$
    converged trials; `failed` means none of the three trials converged.
    Gap is defined as $1-rho_(W F)$ for the worker-firm pair; parentheses
    report the observation share of the component attaining the gap.]
]

When mobility is high, `within` is slower than the MAP backends. As
mobility declines, the worker-firm gap drops by more than two orders of
magnitude to below $10^(-3)$, and MAP runtimes increase. The `within`
runtimes change little across the six designs, making it the fastest
backend in the lowest-mobility rows.

#v(0.4em)

Stronger sorting keeps more movers within firm groups, leaving fewer links
between groups. MAP should slow as those links disappear.

#v(0.4em)

#text(size: 8.9pt)[
  #strong[Sorting benchmark ($n = 1$M).]
  #include "generated/tables/akm_sorting.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one
    covariate, and worker, firm, and year fixed effects. Lower rows
    increase sorting among movers, so fewer movers connect firms in
    different groups. Gap is $1-rho_(W F)$ for the worker-firm pair;
    parentheses report the observation share of the component attaining the
    gap.]
]

From the first sorting design to the last, the worker-firm gap falls by
about a factor of four and MAP runtimes generally increase. The gap is not
monotonic in the intermediate rows. The `within` runtimes change little
across the five designs.

=== Standard Synthetic Benchmarks: fixest and Correia DGPs

The first family is the simple-versus-difficult benchmark data generating
process from `fixest` @berge2026fixest. Both designs use 10M observations,
one covariate, and worker, firm, and year fixed effects. Both contain about
the same number of worker-firm links. In the simple design, those links are
spread broadly across workers and firms; in the difficult design, the
matches are nearly nested. MAP should converge faster on the simple design.
The table also reports legacy `torch-cuda` timings from the PyFixest
benchmark suite.

#text(size: 8.8pt)[
  #strong[Simple vs. difficult design (10M observations, 3 FE).]
  #include "generated/tables/ols.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] CPU times are medians from three
    independently generated 10M-observation samples. The gap is computed on
    the first sample. `torch-cuda` values come from the PyFixest benchmark
    suite. Standalone `within` timings exclude regression overhead. Setup
    and solve take #result_setup_simple_setup and
    #result_setup_simple_solve on the simple design, versus
    #result_setup_difficult_setup and #result_setup_difficult_solve on the
    difficult design. Setup accounts for #result_setup_simple_share and
    #result_setup_difficult_share of the respective totals.]
]

On the simple design, both MAP backends converge quickly and `within` is
slowest.

On the difficult design, where the worker-firm gap is
#result_ols_difficult_gap, `within` takes #result_ols_difficult_within,
compared with #result_ols_difficult_fem for `FEM.jl`,
#result_ols_difficult_fixest for `fixest`, and
#result_ols_difficult_rust_map for unaccelerated `rust-map`.

We retain the historical `torch-cuda` values for reference, but do not
compare them with the local CPU timings because the hardware and run
details differ.

#v(0.35em)

We also use the Correia synthetic datasets, which include complete,
uniform, assortative, and path-like graphs.

#v(0.35em)

#text(size: 8.9pt)[
  #strong[Correia synthetic benchmarks.]
  #include "generated/tables/correia_synthetic.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three runs. Gap denotes
    $1-rho$ for the `id1`-`id2` pair after the same singleton pruning;
    parentheses report the observation share of the component attaining the
    gap. In two-way models, smaller gaps are generally associated with
    slower MAP convergence. The table omits `synthetic-zigzag`. On this
    small path graph, `rust-map` and `fixest` reach their iteration caps,
    `FEM.jl` takes #result_zigzag_fem, and `within` takes
    #result_zigzag_within.]
]

Unlike the controlled AKM experiments, these datasets vary in both sample
size and graph structure, so the runtimes reflect setup time as well as
convergence. `within` is slower on the complete and easier uniform designs.
On `synthetic-uniform-harder`, only `FEM.jl` is faster; on the assortative
design, `within` has the shortest runtime.

=== Standard Real-Data Benchmarks: Correia Collection

Empirical graphs can contain units that appear in many observations, thin
links between dense groups, many disconnected components, and interactions
among more than two identifiers. The Correia real datasets contain these
features.

#v(0.35em)

#text(size: 8.9pt)[
  #strong[Correia real-data benchmarks.]
  #include "generated/tables/correia_real.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three runs on
    singleton-dropped samples, as produced by the PyFixest benchmark suite.
    The gap is $1-rho$ for the `id1`-`id2` pair after the same singleton
    pruning; parentheses report the observation share of the component
    attaining the gap. A small component share means that the reported gap
    applies to only part of the sample, as in `directors`.]
]

Accelerated MAP is fastest on `credit` and `soccer`, where the gaps are
large. In `directors`, the component with the smallest gap contains
#result_directors_component_share of the observations, so that gap
describes only part of the sample. On `enron`, `within` takes
#result_correia_enron_within and `FEM.jl` #result_correia_enron_fem. The
factor-pair preconditioner is fastest on `github`, `patents`, `workers`,
`schools`, and `directors`.

== Poisson / PPML Benchmark

Generalized linear models with high-dimensional fixed effects are typically
estimated by iteratively reweighted least squares (IRLS). Each IRLS step
fits a weighted least squares problem in which the response and covariates
are demeaned against the fixed effects, with weights that are updated
between iterations @correia2020ppmlhdfe @stammann2018. The demeaning
operation is identical to the process described in the prior sections.

=== Preconditioner reuse across IRLS

The fixed-effect graph stays the same across IRLS steps, although the
weights change. The implementation builds the preconditioner once per
regression and reuses it at later steps. This may require more inner Krylov
iterations, but it does not change the weighted least-squares solution if
the inner solver converges.

We use the simple and difficult `fixest` DGPs @berge2026fixest with 1M
observations, one covariate, and worker, firm, and year fixed effects. The
outcome is negative binomial with dispersion $theta = 0.5$ and a log-linear
conditional mean. PPML has the correct mean specification despite the
non-Poisson variance. We compare R `fixest`'s `fepois`,
`GLFixedEffectModels.jl`, and two PyFixest `fepois` backends: the default
unpreconditioned `rust-map` and the preconditioned `within` solver. All
packages use a common cap of 100 outer IRLS iterations; their other
stopping rules remain at package defaults.

#v(0.35em)

#text(size: 8.8pt)[
  #strong[Poisson benchmarks (1M observations, one covariate,
    worker-firm-year fixed effects).]
  #include "generated/tables/ppml.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three full IRLS regression
    calls at $n = 1$M, one covariate, and three fixed effects. `fixest` is
    R `fixest::fepois`; `rust-map` and `within` are the PyFixest `fepois`
    routine with the unpreconditioned MAP backend and the factor-pair
    preconditioned solver, respectively; `GLFEM.jl` is
    `GLFixedEffectModels.jl`. `failed` indicates that all three `rust-map`
    trials reached the 10000-iteration MAP cap without converging. The
    well-connected and near-nested descriptions refer to the absorbed
    worker-firm graph.]
]

On the simple design, all four backends finish in
#result_ppml_simple_range. On the difficult design, `within` takes
#result_ppml_difficult_three_within, compared with
#result_ppml_difficult_three_glfem for `GLFEM.jl` and
#result_ppml_difficult_three_fixest for `fixest`; `rust-map` reaches its
iteration cap.

== Memory Use

MAP needs only the current residuals and per-level group sums. The
factor-pair preconditioner also retains pair information between
iterations. We measure peak resident set size (peak RSS) on the simple and
difficult DGPs to quantify the additional memory. The comparison is limited
to `within` and `rust-map` in the same Python package. Comparing peak RSS
across Python, R, and Julia would also measure differences in language
runtimes, data loading, garbage collection, and package internals. We run
each backend in a separate process and read peak RSS from `ru_maxrss`.

#v(0.4em)

#text(size: 8.9pt)[
  #strong[Memory footprint (3 FE, one covariate).]
  #include "generated/tables/memory.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Each cell is one run in an isolated
    Python process with one covariate and three fixed effects; the values
    are not medians. Gap denotes $1-rho_(W F)$ for the worker-firm pair.]
]

The preconditioner adds #result_memory_100k_overhead at 100K observations
and #result_memory_1m_overhead at 1M. The extra memory holds factor-pair
co-occurrences, partition weights, and local approximate Cholesky factors.

== Numerical Equivalence

We compare the new solver's coefficient estimate with the estimates
returned by the other routines. Exact equality is not expected because the
packages use different stopping criteria and tolerances.

The comparison uses the 100K-observation simple and difficult `fixest`
DGPs. The worker-firm gap is #result_agreement_simple_gap in the simple
design and #result_agreement_difficult_gap in the difficult design. The
table reports the coefficient on `x1` for all four backends.

#v(0.4em)

#text(size: 9.2pt)[
  #strong[Coefficient agreement (100K observations, 3 FE, one covariate).]
  #include "generated/tables/agreement.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] $hat(beta)_1$ is the slope coefficient on
    `x1`. The last column reports the absolute difference from `rust-map`.
    `within` is the PyFixest preconditioned Rust backend.]
]

On the simple design, every backend agrees with `rust-map` to within
#result_agreement_simple_max. On the difficult design, the largest
difference is #result_agreement_difficult_max. These checks cover one
coefficient in two designs.

= Software

The solver studied in this paper is available as open-source software
through the `within` project @within. The computational core is implemented
in Rust and exposed through Rust, Python, and R interfaces; each language
binding invokes the same underlying solver. This shared core matters for
reproducibility and adoption: the same algorithm can be called from Rust,
Python, or R without reimplementation.

#v(0.35em)

#text(size: 9.2pt)[
  #table(
    columns: (0.75fr, 0.95fr, 1.15fr, 1.65fr),
    stroke: 0.35pt + table-light-rule,
    inset: (x: 5pt, y: 3.6pt),
    align: (left, left, left, left),
    table.hline(stroke: 0.8pt + table-rule),
    table.header(th[Interface], th[Package], th[Registry], th[Install]),
    table.hline(stroke: 0.45pt + table-rule),
    [Rust],
    [`within`],
    [#link("https://crates.io/crates/within")[crates.io]],
    [`cargo add within`],
    [Python],
    [`within-py`],
    [#link("https://pypi.org/project/within-py/")[PyPI]],
    [`pip install within-py`],
    [R],
    [`withinr`],
    [#link(
      "https://py-econometrics.r-universe.dev/withinr",
    )[py-econometrics R-universe]],
    [`install.packages("withinr", repos = "https://py-econometrics.r-universe.dev")`],
    table.hline(stroke: 0.8pt + table-rule),
  )
]

The Rust crate exposes the lower-level solver together with its
configuration types. The Python and R packages provide solver-level `solve`
and `solve_batch` APIs for residualizing one or several right-hand sides.
The algorithm is also available as a demeaning backend for PyFixest
@pyfixest.

Here is a minimal Python example, in which we demean an outcome variable
and two covariates against worker and firm fixed effects, before we run the
FWL fit on the demeaned variables:

```python
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
```


= Conclusion

This paper introduces a factor-pair Schwarz preconditioner for
residualizing high-dimensional fixed effects. The preconditioner uses
pairwise co-occurrence tables, such as worker-firm match counts, while LSMR
handles the remaining coupling among factors. On the well-connected designs
in our benchmarks, setup takes longer than the iterations it saves, and MAP
or diagonally preconditioned LSMR is faster. On poorly connected and
near-nested designs, `within` is fastest in most of the lowest-gap cases
and in the difficult PPML benchmark. Reusing the preconditioner across IRLS
steps does not make it worthwhile on the well-connected PPML design, where
`fixest` remains faster. Lower pairwise spectral gaps are associated with
slower MAP convergence in these benchmarks, but they do not provide a
numerical cutoff for choosing a solver.

#set heading(numbering: none)
#pagebreak()

= Appendix A: Details of the Factor-Pair Schwarz Preconditioner

== Two-Factor Block Inverse

For the two-factor block

$ G_2 = mat(G_(W W), C_(W F); C_(W F)', G_(F F)), $

let

$ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

Standard block inversion gives

$
  G_2^(-1) = mat(
    G_(W W)^(-1) + G_(W W)^(-1) C_(W F) S^(-1) C_(W F)' G_(W W)^(-1), -G_(W W)^(-1) C_(W F) S^(-1);
    -S^(-1) C_(W F)' G_(W W)^(-1), S^(-1)
  ).
$

== Algorithm

Algorithm 1 gives the implementation corresponding to the construction
summarized in @fig-pair-strategy.

#align(center)[
  #block(
    width: 96%,
    inset: (x: 0.95em, y: 0.75em),
    fill: rgb("#f7f8fa"),
    stroke: 0.35pt + rgb("#d8dee8"),
    radius: 4pt,
  )[
    #text(size: 8.7pt)[
      #align(center)[#strong[Algorithm 1. Factor-Pair Schwarz
        Preconditioner]]

      #v(0.25em)
      #align(left)[
        #strong[Inputs]
        - Observation-level factor codes for $Q$ absorbed dimensions.
        - Diagonal weights $W$ and a local solver configuration.
        - Krylov residual $r$ in coefficient space.

        #strong[Preconditioner setup]
        - Enumerate all unordered factor pairs $(q,r)$ with $q < r$.
        - For each pair, build weighted count blocks $G_(q q)$, $G_(r r)$
          and the weighted cross-tabulation $C_(q r)$.
        - Split the induced bipartite graph into connected components and
          create one Schwarz subdomain $s$ per component.
        - If fixed-effect level $j$ appears in $c_j$ subdomains, store the
          partition weight $omega_j = 1 / sqrt(c_j)$.
        - For each subdomain, form $L_s = T_s G_s T_s$. If $p_s$ is the
          number of local levels, set
          $Pi_s = I_(p_s) - bold(1) bold(1)' / p_s$; multiplying by $Pi_s$
          subtracts the component mean.
        - Eliminate the larger factor block, leaving a reduced system on
          the smaller block. Solve a small reduced system by dense
          Cholesky. For a large system, approximate the fill-in cliques by
          sampling and apply randomized approximate Cholesky.

        #strong[Krylov application]
        - Initialize $z = 0$.
        - For each subdomain $s$, form $h_s = tilde(D)_s R_s r$ and
          $b_s = Pi_s T_s h_s$.
        - Apply the stored local solver to $b_s$ to obtain $v_s$, then set
          $u_s = T_s v_s$.
        - Accumulate $z <- z + R_s' tilde(D)_s u_s$.
        - Return $z = M^(-1) r$.
      ]
    ]
  ]
]

#pagebreak()

#bibliography(
  "refs.bib",
  style: "chicago-author-date",
  title: [References],
)

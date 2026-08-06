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
#let miss = text(fill: rgb("#777777"))[--]
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
  #text(size: 9.5pt)[Draft: August 6, 2026]
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
      canonical algorithm for estimating high-dimensional fixed-effect regressions. It is fast when
      absorbed factors are well connected, but converges slowly on sparse or nearly nested
      fixed-effect graphs, such as matched employer-employee panels where worker-firm mobility
      links separate worker and firm effects. The convergence behavior of MAP depends on the
      mobility pattern linking workers to firms, yet MAP only indirectly makes use of this 
      information by iterating over one fixed effect at a time. That mobility pattern is,
      however, directly encoded in the matrix of worker-firm match counts, which together
      with the worker and firm count diagonals forms a graph Laplacian after a sign flip
      and admits sparse approximate Cholesky factorization.
      We propose a graph-preconditioned Krylov solver whose reusable preconditioner is
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

Fixed-effect regressions are ubiquitous in applied econometrics: according to #cite(<goldsmith2026tracking>, form: "prose"), 
roughly half of published research in top economics and finance journals mentions
"fixed effects". They appear throughout all fields of applied economics: labor economists use worker and firm fixed effects to separate worker heterogeneity from firm wage premia; health economists study physician practice styles with individual-physician
and region fixed effects in mover designs; and education researchers study models with school, student, teacher, or
student-teacher fixed effects.

The standard computational starting point for estimating these regressions efficiently is the Frisch-Waugh-Lovell
(FWL) theorem @frisch1933 @lovell1963. FWL reduces the fixed-effect estimation problem to "residualizing" the outcome
and every regressor of interest against the fixed effects, and then to running a
low-dimensional regression on the residualized variables. 

The workhorse method for these fixed-effect residualizations is the Method of
Alternating Projections (MAP), also known as iterative demeaning or the "Zig-Zag"
algorithm @guimaraes2010 @gaure2013. Most leading software implementations of
fixed-effect regression, such as `reghdfe` in Stata @reghdfe @correia2017, `fixest`
@berge2026fixest in R, or PyFixest in Python @pyfixest, use MAP or MAP-based variants
with acceleration.

Because the AKM worker-firm model is among the most prominent applications of
high-dimensional fixed effects, we use a worker-firm-year panel based on
#cite(<akm1999>, form: "prose") as the running example in this paper. The method of alternating projections cycles 
through the fixed-effect dimensions one at a time: it first subtracts worker means, then firm means, then year means, 
and repeats until convergence. The coupling between fixed effects is never used directly; the cross-fixed effects information 
is transmitted only indirectly through the residual that each step hands to the next. 
How fast MAP converges therefore depends on how quickly information about this coupling can propagate from one update to the next.

Fixed effects and their coupling form a graph structure. In a worker-firm panel, movers create paths between firms, while
stayers add observations without connecting firms to one another. When this graph is sparse or poorly connected, some fixed-effect 
directions are nearly collinear, and MAP needs many iterations to propagate information across it before converging. 
The same graph features that determine whether worker and firm effects are identified
also govern MAP convergence. These features include worker mobility, sorting, and
segmentation into disconnected labor markets with thin bridges, such as public- and private-sector
employment when workers only rarely move between the two sectors.

The graph structure of the fixed effects can be algebraically encoded in the off-diagonal blocks of the fixed-effect 
Gramian @correia2017. These off-diagonal blocks record co-occurrences among the fixed effects: which workers
work at which firms, which physicians practice in which regions, or which families move across counties. 

In this paper, we propose a new preconditioner that directly encodes the co-occurrence
graph. A preconditioner is a cheap approximation to the system being solved; supplied
to an iterative solver, it reduces the number of iterations without changing the solution.
We build ours from local factor-pair subproblems that use the graph structure of the
Gramian: for example, a worker-firm subproblem
incorporates the observed links between workers and firms. A preconditioner is only
useful if it approximates the inverse of the co-occurrence Gramian at low cost. We obtain
such an approximation by exploiting the fact that, after a sign change, each factor-pair
block is a graph Laplacian, which admits sparse approximate inverses following ideas from
the Laplacian-solver literature @spielman2014 @gao2025. The preconditioned system is
then solved with a Krylov solver, an iterative method for large linear systems.#footnote[The Julia implementation
of fixed-effect regression, `FixedEffectModels.jl` @fixedeffectmodels, uses the same
Krylov solver as we do - LSMR @fong2011 - but only with diagonal
preconditioning, which ignores the off-diagonal co-occurrence structure entirely; our
contribution is the preconditioner, not the use of LSMR for the outer iteration.]

We benchmark widely used fixed-effect regression implementations available through
PyFixest, R's `fixest`, and Julia's `FixedEffectModels.jl`. They cover MAP variants and
LSMR with diagonal preconditioning. We compare them with our additive factor-pair
preconditioner as worker-firm connectivity varies.

The paper's main result appears in @fig-gap-runtime. The horizontal axis moves from
well-connected worker-firm graphs on the left to weakly connected graphs on the right.

#figure(
  image(result-img("gap_runtime.svg"), width: 100%),
  caption: [Median runtime against the worker-firm spectral gap, on log-log axes. Because
  smaller gaps mean weaker connectivity, the horizontal scale decreases from left to
  right. Both panels report simulated worker-firm-year panels with 1 million observations,
  in which worker mobility and sorting vary. Within each design, every implementation
  receives the same stored sample. Panel (a) compares package defaults. Panel (b) holds
  the PyFixest code path and achieved accuracy fixed, isolating the comparison from
  language and implementation differences across packages. Solid lines are log-log fits
  to the median runtimes. Failed cells are omitted; hollow markers indicate that some
  trials did not converge.]
) <fig-gap-runtime>

At high connectivity, all implementations finish quickly. As the spectral gap narrows,
MAP and the unpreconditioned or diagonally preconditioned LSMR configurations slow down.
The additive factor-pair preconditioner behaves differently: its runtime remains low and
even falls across the hardest designs because the setup cost of the preconditioner
decreases. Low-connectivity worker-firm subproblems are sparser and take less time to
factorize, so constructing the preconditioner becomes cheaper as the other solvers slow
down.

The rest of the paper is organized as follows. Section 2 sets up the fixed-effect
absorption problem, and Section 3 introduces the AKM model as our running example.
Section 4 develops the graph structure of the fixed-effect Gramian, and Section 5
connects this structure to the convergence behavior of MAP. Section 6 builds up
the factor-pair Schwarz preconditioner, starting from a general discussion of
preconditioning and culminating in the construction of the graph-based preconditioner.
Section 7 reports the runtime benchmarks; Section 8 describes the software through which
the new algorithm is available; and Section 9 concludes. Appendix B reports numerical
equivalence, memory use, and additional benchmark diagnostics.

= Absorbing Fixed Effects#footnote[Researchers employ several names for this operation:
"absorbing fixed effects", "demeaning", "residualizing", or applying the "within
transformation". We use these terms interchangeably throughout.]

We focus on the linear model

$ y = X beta + D alpha + epsilon, $

where $X$ contains the regressors of interest, $D$ is the fixed-effect design matrix,
and $alpha$ collects the fixed-effect coefficients. In high-dimensional applications $D$
may have hundreds of thousands or millions of columns, and forming or inverting the full
system in $[X quad D]$ might prove computationally infeasible. Fortunately, via the 
Frisch-Waugh-Lovell (FWL) theorem, we can compute $hat(beta)$ without ever forming $[X quad D]$ or
inverting its cross product.

FWL reduces the computation of $hat(beta)$ to two steps. In step one, we residualize the
outcome and each covariate against the fixed effects: we regress $y$ on $D$ and keep
the residual $tilde(y)$, and we do the same for every column of $X$. Denoting $M_D$ as the
linear operator that sends a variable to this residual, so that $tilde(y) = M_D y$ and
$tilde(X) = M_D X$, the coefficient of interest is recovered by regressing the
residualized outcome on the residualized covariates,

$ tilde(y) = M_D y, quad tilde(X) = M_D X, quad
  hat(beta) = (tilde(X)' W tilde(X))^(-1) tilde(X)' W tilde(y), $

where $W$ is a diagonal matrix of weights. With one covariate, the procedure consists of
three regressions: $y$ on $D$, $x$ on $D$, and $tilde(y)$ on $tilde(x)$. FWL implies that
the slope in the last regression equals the coefficient on $x$ in the full regression of
$y$ on $x$ and $D$. 

The residualization step is the weighted least-squares projection of the outcome and
each covariate in $X$ onto the column space of the fixed-effect dummy matrix $D$. For a
right-hand side $mu$, either $y$ or one column of $X$, we solve 

$ hat(alpha)_mu = arg min_alpha || D alpha - mu ||_W^2, $ <eq:demean-ls>

and the residual is $tilde(mu) = mu - D hat(alpha)_mu$. The first-order
condition for @eq:demean-ls,

$ D' W (D hat(alpha)_mu - mu) = 0, quad "equivalently" quad
  G hat(alpha)_mu = D' W mu, quad G = D' W D, $ <eq:fwl-normal>

determines $hat(alpha)_mu$. Each FWL residualization uses the same coefficient matrix
$G$; only the right-hand side changes as we move from the outcome to the covariates.
The cost of residualization therefore depends on the structure of the Gramian $G$. In the next section,
we illustrate this structure with the AKM worker-firm model and explain
why fixed-effect designs have a direct graph interpretation. Sections 5--6 then return to the algorithms and show
how the structure of the Gramian and its associated graph governs MAP convergence, and
explain how we use it to construct the factor-pair Schwarz preconditioner.

= A Running Example: The AKM Model

The AKM model of #cite(<akm1999>, form: "prose") introduces the worker-firm setting used
throughout the paper. It separates persistent worker heterogeneity from firm wage
premia using workers who move across firms. We write the AKM regression equation as

$ y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t), $

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed effect for the
firm employing worker $i$ at time $t$, and $phi_t$ is a time fixed effect. 

The AKM specification has a natural graph representation. Workers and firms are nodes in a
bipartite graph, and each employment spell contributes an edge. Worker moves induce a
firm-to-firm graph, in which two firms are connected when at least one worker is observed
at both firms. Although stayers - workers who never change their employer - add observations to existing worker-firm links, they do
not create bridges between firms. Year effects enter as a third, low-dimensional factor
observed on the same worker-firm records. @fig-connectivity contrasts a well-connected
mobility graph with one that fragments under strong sorting.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 50%),
  caption: [Worker-firm graph connectivity. When mobility is high, many paths connect
  firms. With low mobility and strong sorting, the graph breaks into nearly separate
  clusters joined only by narrow bridges.]
) <fig-connectivity>

These mobility links matter both for identification and, as we will argue later, for computation.
In AKM, worker and firm fixed effects are separately identified through movers, who
provide the comparisons that distinguish worker heterogeneity from firm wage premia. A
worker observed at only one firm provides no such comparison: a high wage could reflect
an unusually productive worker, a high-wage firm, or both. Without worker moves, these
components are not separately identified. Movers observed across firms with different
wage profiles help attribute wage variation to worker effects or firm premia. If a worker earns high wages across
several firms, the comparison points toward a worker effect; if many different workers
earn higher wages at the same firm, it points toward a firm premium. The more such
cross-firm comparisons the data contain, especially across otherwise different firms,
the easier it is to identify worker and firm premia. In the graph, these moves are
exactly the edges that connect firms; additional moves of workers across firms 
add worker-firm links to the graph. 

= The Graph Structure of the Gramian

The bipartite graph of worker and firm connections introduced in Section 3 has an algebraic representation in the
block structure of the Gramian $G = D' W D$ @correia2017.
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

As a small example, we construct a worker-firm panel and populate its Gramian. For
simplicity, we ignore any regression weights and set $W = I$.

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
  caption: [Worker-firm projection of the example panel. Worker $W_1$ is a mover; workers
  $W_2$ and $W_3$ are stayers.]
) <fig-toy-projection>

@fig-toy-projection plots the worker-firm projection of this panel. Worker $W_1$ has
employment spells both in $F_1$ and $F_2$ and creates a link between
the two firms; in AKM terms, $W_1$ is a mover. Worker $W_2$ stays at $F_1$ for two periods, and $W_3$
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

The first row indicates that worker $W_1$ appears once at firm $F_1$ and once at firm
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
opposite pattern. Combining the diagonal count blocks and the off-diagonal
cross-tabulation blocks yields the full Gramian.

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

The worker-firm submatrix stores the bipartite graph algebraically. The diagonal entries
are worker and firm counts, and the entries of $C_(W F)$ are edge multiplicities between
workers and firms. After flipping the sign of $C_(W F)$, this submatrix is a graph
Laplacian: its off-diagonal entries are non-positive, and every row sums to zero because
each diagonal count cancels the off-diagonal spell counts in the same row. For a worker,
the diagonal entry is the number of that worker's employment spells, while the
off-diagonal entries count how those spells are distributed across firms; firm rows have
the analogous interpretation with spell counts summed over workers.


$ L_(W F) = mat(augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
  2, 0, 0, -1, -1;
  0, 2, 0, -2, 0;
  0, 0, 2, 0, -2;
  -1, -2, 0, 3, 0;
  -1, 0, -2, 0, 3
). $

The same Laplacian construction applies to any pair of fixed effects, and the preconditioner
of Section 6 builds on these pairwise Laplacians. Before turning to it, 
however, we introduce the method of alternating projections, which avoids forming the 
full Gramian $G$ by working only on the diagonal worker, firm, and year blocks.


= Alternating Projections and Graph Connectivity

The workhorse algorithm for multi-way fixed effects is the Method of Alternating
Projections (MAP), also referred to as iterative demeaning or the "zig-zag" algorithm
@guimaraes2010 @gaure2013. Many packages employ MAP or its variants, frequently combined
with accelerations @berge2018 @correia2017, such as the Irons-Tuck extrapolation
used by `fixest` @irons1969 @berge2026fixest. Sections 3 and 4 introduced the
fixed-effect graph and its algebra; this section turns to MAP itself and shows how that
graph geometry governs its convergence rate.

MAP solves the FWL residualization problem by iterating over one fixed effect at a
time. In the worker-firm-year model, MAP first subtracts worker means
from the current residual, then firm means from the updated residual, then year means,
and so on until convergence.

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

Each equation can be rearranged to express one block of effects conditional on the
others. Using $C_(W F) = D_W' W D_F$ and $C_(W Y) = D_W' W D_Y$ to factor $D_W' W$ out
of the right-hand side, the first equation becomes

$ G_(W W) alpha_W = D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Because $G_(W W)$ is a diagonal matrix whose entries are workers' total observation
weights, solving for
$alpha_W$ divides each worker's weighted partial residual by
its total observation weight. We apply the same rearrangement to the second equation to
obtain the firm equation

$ G_(F F) alpha_F = D_F' W (mu - D_W alpha_W - D_Y alpha_Y), $

and the year equation is analogous. Because $G_(W W)$, $G_(F F)$, and $G_(Y Y)$
are all diagonal, each of the three equations is solved by computing a weighted
group mean in a single pass over observations.

MAP uses this diagonal structure iteratively. Holding the other effects fixed, it updates
the worker effects from the current partial residual, then repeats the same step for firms
and years. Each sweep therefore cycles through the fixed-effect dimensions, subtracting the
weighted group mean of the current partial residual for the factor being updated.

The cross-tabulation blocks $C_(W F)$, $C_(W Y)$, and $C_(F Y)$ enter the algorithm only
indirectly. For instance, the worker update is computed from the partial residual
$mu - D_F alpha_F - D_Y alpha_Y$, while the firm update is computed from
$mu - D_W alpha_W - D_Y alpha_Y$. Worker-firm, worker-year, and firm-year links are thus
not solved as coupled subproblems; their effect propagates through the residual that one
block update transmits to the next.

This strategy is effective when the graph is well connected. In a high-mobility
worker-firm panel, many workers move across firms, so that worker and firm effects can
be compared through many overlapping employment histories. A high wage at one firm can
then be related to wages earned by the same workers at other firms, and a worker update
rapidly alters the information available to the next firm update, and vice versa.

When mobility is sparse, sorting is strong, or one factor is nearly nested in another,
MAP slows down. The information that separates worker effects from firm effects travels
through movers, the edges of the worker-firm graph, and MAP picks it up only
indirectly, through repeated residual updates. When those edges are few or bunched into
narrow regions of the graph, each further iteration carries little fresh information. MAP
still converges, but it may need many sweeps; each sweep is cheap because the diagonal
block solves reduce to one-pass group means.

The same graph perspective gives a simple diagnostic for MAP difficulty in a
fixed-effect structure. For any pair of fixed effects $(q,r)$, we form the normalized
cross-tabulation $H_(q r) = G_(q q)^(-1/2) C_(q r) G_(r r)^(-1/2)$ and let
$rho_(q r) = sigma_2(H_(q r))^2$ be the square of its largest nontrivial singular value,
equivalently the largest nontrivial eigenvalue of $H_(q r)' H_(q r)$. The largest
singular value of $H_(q r)$ equals one on every connected component and carries no
information about connectivity, so we discard it. We report the
spectral gap $1 - rho_(q r)$ in the benchmarks below. This gap measures how well
connected the factor-pair graph is.#footnote[When the graph has
more than one connected component, we compute the gap on each component and report
the smallest, together with its observation share.]
Gaps near zero signal sparse mobility or near-nesting, the settings in which MAP
converges slowly; larger gaps indicate better connected factor-pair graphs.
For the worker-firm example of Section 4, the gap is $1/3$.#footnote[In that example, $G_(W W) = "diag"(2,2,2)$,
$G_(F F) = "diag"(3,3)$, and $C_(W F) = mat(1, 1; 2, 0; 0, 2)$, so
$H_(W F) = G_(W W)^(-1/2) C_(W F) G_(F F)^(-1/2) = 1 / sqrt(6) mat(1, 1; 2, 0; 0, 2)$.
The graph is connected, and
$H_(W F)' H_(W F) = 1 / 6 mat(5, 1; 1, 5)$ has eigenvalues $1$ and $2/3$.
After dropping the unit eigenvalue, $rho_(W F) = 2/3$ and the gap is $1/3$.]

= The Factor-Pair Schwarz Preconditioner

== Preconditioners

The previous section attributed MAP's slow convergence to thin connections in the
fixed-effect graph. A solver that receives this connectivity information directly
should converge faster. MAP has no natural way to accept it: its update rule is fully
determined by the list of absorbed factors, and the cross-tabulations enter only
through the residual that one update passes to the next. Krylov solvers, by contrast,
take an additional input, the preconditioner, and this input is where we place the
graph structure. We therefore replace factor-by-factor demeaning via MAP with LSMR
@fong2011, an iterative least-squares algorithm that improves an initial guess through
repeated residual corrections. The change of solver gains little by itself: a poorly
conditioned Gramian $G$ slows LSMR just as a poorly connected graph slows MAP. The core
acceleration stems from building a good preconditioner, which we focus on in the 
remainder of this section.

How fast LSMR converges depends on the conditioning of $G$. When $G$ is well
conditioned, LSMR shrinks every component of the residual at a comparable rate. When
$G$ is poorly conditioned, LSMR removes some components after a few iterations, whereas
components that correspond to weak links, sparse mobility, or near nesting in the
fixed-effect graph shrink much more slowly; the iteration then spends most of its steps
on these slow components. A preconditioner counteracts this imbalance: it changes the
coordinates of the linear system so that slow and fast components decay at more similar
rates, without changing the least-squares solution.

The ideal preconditioner is $G^(-1)$.#footnote[As in any model with several fixed
effects, the level effects are pinned down only up to a normalization: we can add a
constant to every worker effect and subtract it from every firm effect without changing
the fitted values $D alpha$, and likewise for years. The dummy-coded $G = D' W D$ and the
smaller blocks inverted below are therefore not invertible until this indeterminacy is
removed -- one normalization for the worker-firm pair, and a second once year effects are
added, as in the Section 4 example. We adopt the standard normalization within each
connected set and read every inverse below as that of the resulting system. The estimated
effects depend on the normalization; the residualized outcome and regressors, which are
all the regression uses, do not.] If $M^(-1) = G^(-1)$, then the
preconditioned operator is

$ M^(-1) G = G^(-1) G = I. $ <eq:ideal-preconditioner>

In exact arithmetic, the Krylov iteration would then recover the solution after one
correction, because the system has no slow directions left to remove. To form
$G^(-1)$ we would have to solve the fixed-effect normal equations themselves, so the
identity case serves as a benchmark rather than an implementable preconditioner. A
useful preconditioner must approximate enough of $G^(-1)$ to remove the slow directions
of the iteration, while its construction and repeated application must amortize over the
iterations it saves.#footnote[LSMR never forms $M^(-1) G$
explicitly, nor $G$ itself. The iteration requires only products with $D$ and $D'$
and applications of $M^(-1)$, supplied as linear operators.]

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
cross-tabulations $C_(W F), C_(W Y), C_(F Y)$. Block inversion via the Schur complement
shows how each off-diagonal block enters $G^(-1)$. For the
two-factor block

$ G_2 = mat(G_(W W), C_(W F); C_(W F)', G_(F F)), $

it gives the explicit formula

$ G_2^(-1) = mat(
  G_(W W)^(-1) + G_(W W)^(-1) C_(W F) S^(-1) C_(W F)' G_(W W)^(-1), -G_(W W)^(-1) C_(W F) S^(-1);
  -S^(-1) C_(W F)' G_(W W)^(-1), S^(-1)
). $

$ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

The formula shows that every block of $G_2^(-1)$ depends on $C_(W F)$ only through the Schur complement
$S$: the cross-tabulation enters through matrix products, never as its own inverse.
$G_(W W)$ is diagonal, so $G_(W W)^(-1)$ is a division by weighted worker counts. The Schur complement $S$, by contrast, is the firm-side
mobility system that remains after eliminating workers. At the scale of modern worker-firm register data, solving this
system is expensive: exact factorization creates many additional nonzero entries,
separate connected components require separate normalizations, and weak mobility makes
the remaining directions slow to resolve. $S^(-1)$ therefore carries almost all the cost of
the solve. Block inversion via Schur complements generalizes to three factors: the closed form has
more terms, but every block of $G^(-1)$ still depends jointly on the cross-tabulations
$C_(W F), C_(W Y), C_(F Y)$.

The coarsest approximation to $G^(-1)$ keeps only the diagonal count inverses and drops
the Schur-complement corrections,

$ M_("diag")^(-1) = "diag"(G_(W W)^(-1), G_(F F)^(-1), G_(Y Y)^(-1)). $

This preconditioner is a single division by weighted level counts. It is the diagonal
preconditioner used with LSMR in `FixedEffectModels.jl` @fong2011 @fixedeffectmodels.
Diagonal scaling encodes how many observations a level carries, but not how that level
connects to the rest of the labor market. Those counts can already be useful when
employment is concentrated in a few large firms, such as Novo Nordisk in Denmark
or Samsung in South Korea, because the size differences alone remove an important source
of scale variation. What diagonal scaling does not record is whether workers at those
firms link them broadly to other firms or remain concentrated in a narrow corner of the
mobility graph.

== The Factor-Pair Schwarz Approximation

Additive Schwarz preconditioning adds this missing pairwise connectivity without solving
the full three-factor system @xu1992 @toselli2005. It splits the fixed-effect problem
into smaller overlapping pair problems. In the AKM case, one problem contains workers
and firms, another contains workers and years, and a third contains firms and years.
The worker-firm problem moves residual information along observed employment links; the
other two pair problems do the same for worker-year and firm-year links. We then combine
the three pair corrections in the full coefficient space. The preconditioner therefore
gives the Krylov iteration the main pairwise channels of the Gramian, while the outer
iteration handles the remaining three-way coupling.

To make the local subproblems concrete, we first consider the worker-firm pair. Its
local problem is exactly the two-factor block from the Schur calculation,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)). $

@fig-pair-block places this worker-firm pair block beside the single diagonal block
solved by a factor-level MAP update.

#figure(
  image(solver-img("factor_level_vs_pair_block.svg"), width: 88%),
  caption: [Local operator used by a factor-level MAP update (left) versus the
  factor-pair Schwarz solve (right), shown on the example worker-firm panel of Section 4.
  The factor-level block is the diagonal $G_(W W) = "diag"(2,2,2)$. The factor-pair
  block adds the firm count block $G_(F F) = "diag"(3,3)$ and the cross-tabulation
  $C_(W F)$ in its off-diagonal positions; the dashed outline marks the worker-firm
  subdomain on which the local Schwarz solve operates.]
) <fig-pair-block>

Its inverse carries $C_(W F)$ through the Schur complement, so the local correction
holds the worker-firm mobility geometry that diagonal scaling discards. To place this
correction inside the three-factor problem, let $R_(W F)$ select the worker and firm
entries from the full coefficient vector $alpha = [alpha_W; alpha_F; alpha_Y]$; its
transpose $R_(W F)'$ places the resulting correction back into the full vector. The
diagonal matrix $tilde(D)_(W F)$ contains the weights that split levels across the pair
problems in which they appear; we call these entries partition-of-unity weights. The exact worker-firm contribution is

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
separately. They do not solve the simultaneous worker-firm-year problem; that remaining
coupling, together with the error introduced by splitting shared levels across pairs, is
left to the outer LSMR iteration.

== Approximate Pair Solves via Graph Laplacians

For $P^(-1)$ to serve as the operator $M^(-1)$ inside LSMR, each pair contribution must
be computed without solving a large dense system. For factor pairs with few levels, we
invert the pair block directly. In the example worker-firm panel of Section 4, the pair
block has three worker levels and two firm levels; after the normalization of
Section 6.1 removes the one free constant, the local solve is a $4 times 4$
inversion. At the scale of modern worker-firm register data, however,
a single pair can carry hundreds of thousands of levels per side, and direct inversion
becomes the same kind of large linear-algebra problem the preconditioner is meant to
avoid. We instead use the graph-Laplacian structure of the pair block.

For a worker-firm pair, the local Schwarz step solves the pair-Gramian system

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)) x = u, $

where $u$ is the weighted worker-firm part of the current Krylov residual. This block is
not a graph Laplacian, because its off-diagonal entries $C_(W F)$ are non-negative. A
sign flip removes the obstacle. Let $T_(W F) = "diag"(I_W, -I_F)$ flip the sign of the
firm entries, so that $T_(W F)^2 = I$. Conjugation by $T_(W F)$ gives

$ L_(W F) = T_(W F) mat(G_(W W), C_(W F); C_(W F)', G_(F F)) T_(W F)
  = mat(G_(W W), -C_(W F); -C_(W F)', G_(F F)), $

a weighted bipartite graph Laplacian: symmetric, with non-positive off-diagonals and
zero row sums. Because $T_(W F)^2 = I$, the pair-Gramian solve follows from the
Laplacian solve by the same flip on each side,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1) = T_(W F) L_(W F)^(-1) T_(W F), $

where both inverses are read as in Section 6.1: the solve fixes the free constant on
each connected component by returning the zero-mean solution, and the residualized
variables do not depend on this choice. A single Laplacian solve therefore yields the
pair-Gramian solution: we flip the residual, apply $L_(W F)^(-1)$, and flip back.

For preconditioning, this local solve need not be exact. The outer LSMR iteration
refines any error left by the preconditioner. We therefore approximate the Laplacian
solve using sparse approximate Cholesky factorizations from the Laplacian-solver
literature @spielman2014
@gao2025. An exact Cholesky factorization of the pair block creates fill-in:
eliminating a worker inserts entries linking every pair of distinct firms that worker
visited, and these entries accumulate as the elimination proceeds. In the worst case,
the cost approaches the order $k^3$ operations and order $k^2$ memory of a dense
factorization of a $k$-level system. The randomized approximate factorization avoids
this growth on any graph: its cost is nearly linear in the number of observed
worker-firm links, that is, linear up to logarithmic factors. After transforming this approximate Laplacian solve back to worker-firm
coordinates, we write $A_(W F)$ for the resulting approximate pair-Gramian solve.

The same construction yields $A_(W Y)$ and $A_(F Y)$ for the other two pairs. We
substitute these approximate inverses into the Schwarz sum to obtain the implemented
preconditioner,

$ M^(-1) = sum_((q, r)) R_(q r)' tilde(D)_(q r) A_(q r) tilde(D)_(q r) R_(q r). $

In the worker-firm-year case each factor receives a diagonal contribution from its two
pair subdomains and each off-diagonal correction from the corresponding pair.

The approximate pair inverse introduces a second approximation, beyond the pairwise
splitting. The exact $P^(-1)$ already replaces the full three-factor inverse with pair
solves; $M^(-1)$ now applies each pair
solve only approximately, through $A_(q r)$. Both choices shape the preconditioned
search directions and nothing else: the fitted residuals are unchanged, and LSMR still
solves the original fixed-effect least-squares problem to the requested tolerance.

== Implementation Strategy

@fig-pair-strategy summarizes the full construction. We start with the absorbed factors
and split the fixed-effect graph into overlapping factor pairs. For each pair, the local
Gramian block is represented as a graph Laplacian after a sign flip; sparse approximate
Cholesky solves then provide an approximate pair inverse, returned to Gramian
coordinates. The pair corrections are combined with partition-of-unity weights to form
the Schwarz preconditioner $M^(-1)$.

The outer LSMR iteration uses this preconditioner to shape its search directions
@fong2011 @arridge2014 @yang2024flexible. Once constructed, the same preconditioner can
be reused to residualize the outcome and every covariate, and the algorithmic details
are collected in Appendix A.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Summary of the factor-pair preconditioner. The fixed effects are split into
  overlapping factor pairs; each pair solve uses the Laplacian representation with
  approximate Cholesky; the resulting Schwarz preconditioner is then applied inside the
  outer LSMR iteration.]
) <fig-pair-strategy>

= Benchmarks

The analysis above leads to a simple computational prediction: graph preconditioning
should not dominate MAP in every setting, but it should be most useful when weak
connectivity makes MAP's factor-by-factor demeaning pass information slowly through the
fixed-effect graph. The benchmarks below test this prediction on controlled synthetic
designs and standard public benchmark datasets.

The main runtime tables use package-level regression APIs, matching the public
PyFixest benchmark suite, rather than isolated demeaning kernels. Each timing covers the
full regression workflow: model setup, construction of the fixed-effect representation,
residualization of the outcome and covariates, and estimation of the coefficient of
interest. Appendix B verifies that the preconditioned solver matches MAP to numerical
tolerance and reports the memory cost of storing factor-pair information and additional
solver diagnostics.

For cross-package comparisons, every backend receives the same input data but uses its
own default handling of singleton or separated observations. Successful fits record the
retained-observation count; failed attempts record their convergence status and error.
Any shared control is stated with the relevant table. The comparisons are not constrained
to a common estimation sample. The single-package mechanism experiments later in the
paper do use a common prepared sample, because their purpose is to isolate the preconditioner.

The runtime tables also report the pairwise hardness statistic for the relevant factor
pair and the observation share of the component attaining it. Smaller gaps indicate
factor pairs on which MAP tends to converge more slowly; with three or more fixed
effects, the statistic remains a pairwise diagnostic rather than a full convergence
bound.

We benchmark four backends that separate MAP baselines from Krylov solvers with
diagonal versus factor-pair preconditioning:

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
  [`rust-map`], [PyFixest], [Vanilla Rust MAP, without acceleration.],
  [`fixest`], [R `fixest`], [Accelerated MAP with Irons-Tuck and other optimizations @berge2026fixest.],
  [`FEM.jl`], [`FixedEffectModels.jl`], [Diagonally preconditioned LSMR @fong2011 @fixedeffectmodels.],
  [`within`], [PyFixest], [LSMR with factor-pair Schwarz preconditioning.],
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

A strong caveat applies to every runtime comparison: the implementations use different
convergence criteria, so their nominal tolerances are not directly comparable. We begin
with package defaults because they are the paths users obtain without changing solver
options. The achieved-precision experiment later in this section compares the methods
against a common external error measure.

The main OLS benchmarks use synthetic AKM-style panels with one million observations,
one covariate, ten periods, and worker, firm, and year fixed effects. They vary the
worker-firm graph in two ways while holding the rest of the data-generating process
fixed. The mobility designs progressively reduce the number of workers who change firms.
The sorting designs instead concentrate moves within groups of firms, creating weakly
connected blocks even though workers continue to move. Appendix B reports further
synthetic designs and the empirical HDFE benchmarks assembled by Sergio Correia.

=== Worker Mobility

In the first synthetic design, we lower worker mobility across firms. As mobility falls,
MAP still updates one fixed-effect dimension at a time, but fewer workers connect
multiple firms, so information about firm effects moves more slowly through the
iteration.
The factor-pair preconditioner uses the worker-firm graph directly, so the preconditioned
Krylov solver should become relatively more competitive in the low-mobility designs.

#v(0.4em)

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Mobility benchmark ($n = 1$M).]
#include "generated/tables/akm_mobility.typ"
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
		and worker, firm, and year fixed effects. Lower rows reduce worker mobility
		within a 10-period panel, thereby thinning the worker-firm graph. Lower mobility
		makes the worker-firm pair harder for MAP and correspondingly more favorable
		to the preconditioned method. A dash indicates that no completed run is available
		for that cell. Gap is defined as $1-rho_(W F)$ for the worker-firm pair;
		parentheses report the observation share of the component attaining the gap.]
		]]

The mobility benchmark matches the predicted pattern. When mobility is high,
preconditioning yields little advantage: `fixest` is the fastest backend in
`akm_mobility_1`, and `within` incurs setup costs that are not offset by improved
convergence. As mobility declines, the worker-firm gap falls by more than two orders
of magnitude, from $0.41$ to $4.82 times 10^(-5)$ (the decline is not
strictly monotone, because the reported gap is attained on different connected
components), and MAP runtimes rise sharply. `within`'s runtime stays nearly flat across
all designs. In the lowest-mobility configurations, the
preconditioned method is the fastest backend because it operates on the worker-firm graph
directly rather than propagating information through many sweeps that update one
fixed-effect dimension at a time.

#v(0.4em)

=== Sorting Among Movers

The second experiment increases sorting between workers and firms. Stronger sorting
pushes the worker-firm graph toward weakly connected blocks: movers increasingly connect
firms within the same group rather than linking different groups. MAP should therefore
slow down again, because information about firm effects crosses groups only through a
small number of movers. The preconditioned method should be less sensitive, since its
factor-pair solves use the worker-firm graph directly.

#v(0.4em)

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Sorting benchmark ($n = 1$M).]
#include "generated/tables/akm_sorting.typ"
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
		and worker, firm, and year fixed effects. Lower rows raise the degree of
		sorting among movers, pushing the worker-firm graph toward weakly connected blocks.
		Stronger sorting weakens cross-block information flow and thereby raises the value
		of factor-pair preconditioning. Gap is $1-rho_(W F)$ for the worker-firm pair;
		parentheses report the observation share of the component attaining the gap.]
		]]

#block(breakable: false)[The sorting benchmark gives the parallel result. As movers sort more strongly across
firms, the worker-firm graph separates into weakly connected blocks. The worker-firm gap
falls by a factor of about six, from $0.0129$ to $0.00219$, and MAP-based runtimes rise.
The gap does not fall
monotonically, because different components attain the reported value in different rows,
but its overall decline is clear. `within` remains nearly flat because its factor-pair
preconditioner uses the worker-firm links directly, rather than relying on residual
updates that cycle through one fixed-effect dimension at a time.]

=== When Preconditioner Setup Dominates

The AKM designs vary connectivity gradually. The public `fixest` benchmark suite gives
a sharper comparison between a dense, well-connected design and a sparse, nearly nested
one @berge2026fixest. Both contain 10 million observations, one covariate, and worker,
firm, and year fixed effects.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Simple and difficult `fixest` designs (10M observations, 3 FE).]
#include "generated/tables/ols.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three full regression calls. The gap is
  $1-rho_(W F)$ for the worker-firm pair, measured on the generated 1M version of the
  same DGP family; the runtime rows use the same graph construction at 10M observations.
  The standalone `within` timings separate reusable solver construction from the batch
  solve. On the simple design these take #result_setup_simple_setup and
  #result_setup_simple_solve; on the difficult design they take
  #result_setup_difficult_setup and #result_setup_difficult_solve.]
]]

The simple design is easy for all of the iterative solvers. `within` is nevertheless
slowest because building its factor-pair preconditioner takes
#result_setup_simple_share of the standalone demeaning time. The preconditioner saves
too little solve time to recover that cost in one regression.

The ranking reverses on the nearly nested design. Unaccelerated MAP takes 306.2 seconds,
`fixest` takes 62.3 seconds, and diagonally preconditioned LSMR takes 26.9 seconds.
`within` finishes in 4.10 seconds. Its standalone setup share falls to
#result_setup_difficult_share because the factor-pair approximation keeps the subsequent
solve short. @sec-amortization returns to the setup cost directly.

=== Iterations After Setup

Iteration counts help separate construction from the work done after the
preconditioner exists. The table uses the 100,000-observation versions of the same
simple and difficult designs.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Iterations on the simple and difficult designs (100K observations).]
#include "generated/tables/iterations.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] MAP is reported in full sweeps over the fixed-effect
  dimensions. The LSMR columns report Krylov iterations. A MAP sweep and an LSMR
  iteration are different units and should not be compared across columns. Each LSMR
  count is the median of three solves with two right-hand sides.]
]]

Among the LSMR configurations, the additive preconditioner needs 14 iterations on the
simple design and 22 on the difficult one. The diagonal count rises from 16 to 180,
while unpreconditioned LSMR rises from 37 to 249. Once the additive preconditioner has
been built, the difficult design therefore requires only eight more Krylov iterations
than the simple design. Construction cost, rather than a long preconditioned solve,
explains why `within` is unattractive on the simple one-shot regression.

== Runtime and Achieved Precision

A strong caveat applies to all benchmarks: the implementations use different
convergence criteria, so their nominal tolerances are not directly comparable.
PyFixest MAP defaults to $10^(-6)$ and stops when every observation-level residual
changes by less than that amount after a full pass over the fixed-effect dimensions.
R's `fixest` also defaults to $10^(-6)$, but applies the threshold to successive
fixed-effect coefficient iterates. It accepts either a small absolute change or a small
scaled relative change, and periodically makes the same comparison for the residual sum
of squares. `FixedEffectModels.jl` passes its default tolerance of $10^(-6)$ as both
LSMR stopping tolerances. `within` defaults to $10^(-8)$ and stops when either the
estimated least-squares residual is at most $10^(-8)$ times the norm of the right-hand
side, or the scaled normal-equation residual falls below $10^(-8)$. The latter is
$||A^T r||_2 / (||A||_F ||r||_2)$, using LSMR's running norm estimates @fong2011.

The figure compares achieved precision rather than treating the supplied tolerances as
equivalent. It uses mobility designs 1, 3, and 5. Before timing, we recursively remove
singletons once from each one-million-observation design and pass the resulting rows to
every method. We then vary each package's own tolerance, use a common 10,000-iteration
cap, and time three fits at every setting. The tight reference uses factor-pair LSMR at
tolerance $10^(-14)$. The package defaults for the iteration cap are 10,000 for PyFixest
MAP, `fixest`, and `FixedEffectModels.jl`, and 1,000 for `within`.

#figure(
  image(result-img("tolerance_frontier.svg"), width: 97%),
  caption: [Runtime against achieved precision on three AKM mobility designs. Each point
  is the median of three fits on the same recursively singleton-pruned sample for that
  design; pruning is completed before the timed fits. Lines vary a package's requested
  tolerance, while the horizontal axis reports achieved error. The top row measures
  coefficient error as $abs(hat(beta)-hat(beta)^star) / "SE"(hat(beta)^star)$. The bottom
  row measures final residual error as $frac(||r-r^star||_2, ||r^star||_2)$. Both use the
  tight factor-pair LSMR reference. Error decreases from left to right. Circled points
  use each package's default tolerance. All runs have a 10,000-iteration cap. Settings
  that return no solution have no horizontal coordinate and are omitted; the annotations
  report these settings.]
) <fig-tolerance>

== Amortizing the Preconditioner <sec-amortization>

The additive method performs two distinct jobs. It first constructs the factor-pair
blocks and their approximate factorizations, then applies them during the LSMR solve.
The construction is paid once for a fixed set of observations, weights, and fixed-effect
identifiers.

=== Setup Cost Across Connectivity

We time these two stages separately on the AKM mobility designs. The
two-factor specification absorbs worker and firm effects; the three-factor specification
also absorbs year effects. Each cell is the median of five standalone solves at one
million observations, with the same outcome and covariate used in both specifications.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Additive setup and solve time across AKM connectivity.]
#include "generated/tables/akm_setup_cost.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Times are seconds. Each cell is the median of five runs
  at tolerance $10^(-12)$. The two-factor specification absorbs worker and firm effects;
  the three-factor specification adds year effects. The worker-firm gap and component
  share are the same diagnostics used in the mobility benchmark.]
]]

On balance, setup is cheaper in the low-mobility designs. With two fixed effects, median setup declines
from 0.110 seconds in `akm_mobility_1` to 0.062 seconds in `akm_mobility_6`; with three
fixed effects, it falls from 0.114 to 0.069 seconds. Solve time falls as well, from 0.288
to 0.051 seconds with two effects and from 0.217 to 0.161 seconds with three. The
low-mobility worker-firm graph has fewer cross-firm links to store and factorize. This is
why the additive runtime in @fig-gap-runtime can decline while MAP and diagonal LSMR
slow down.

=== Ten Regressions on the Same Fixed Effects

Many empirical projects estimate several specifications on the same sample and fixed
effects. In that case, the factor-pair objects can be constructed once and reused. We
measure ten sequential regressions on both one-million-observation `fixest` designs. For
each design, all three policies receive the same outcome and the same ten covariates. The
first additive policy constructs a new preconditioner for every regression; the second
constructs one and keeps it for all ten.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Ten regressions with rebuilt and cached preconditioners.]
#include "generated/tables/regression_reuse.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three repetitions for each design at 1M
  observations, with worker, firm, and year fixed effects. Each regression
  residualizes the common outcome and one covariate at tolerance $10^(-12)$. Setup and
  solve columns sum time across all ten calls. Speedup is relative to diagonal
  preconditioning. Diagonal and rebuilt additive construct a solver for every call;
  cached additive constructs one solver.]
]]

The simple design remains a poor case for the additive preconditioner, even across ten
regressions. Diagonal preconditioning takes 0.692 seconds. Rebuilding the additive
preconditioner takes 4.46 seconds, while caching lowers this to 1.83 seconds. Avoiding
nine constructions helps, but the cached additive path is still more than twice as slow
as diagonal preconditioning on this well-connected graph.

The difficult design gives the opposite result. Diagonal preconditioning takes 50.3
seconds for the ten regressions. Rebuilding the additive preconditioner lowers this to
1.77 seconds, and caching lowers it further to 1.23 seconds. Caching reduces additive
setup time from 0.434 to 0.040 seconds; the repeated solves take 1.34 seconds with
rebuilding and 1.19 seconds with caching.

== Poisson and Other GLMs

Faster fixed-effect solves matter especially when an estimator requires several of them,
as in generalized linear models fit by iteratively reweighted least squares (IRLS). Each
IRLS step fits a weighted least squares problem in which the response and covariates are
demeaned against the fixed effects @correia2020ppmlhdfe @stammann2018. A single GLM fit
therefore calls the demeaning routine repeatedly. The fixed-effect incidence graph stays
the same, but the weights change between calls. This distinction matters for `ppmlhdfe`,
Poisson fixed-effect estimators, and IRLS-based GLM implementations more generally.

=== Reusing or rebuilding across IRLS

The factor-pair preconditioner depends on both the incidence graph and the weights.
Keeping the first preconditioner avoids later setup work, but it may require more LSMR
iterations as the weights move. Rebuilding pays the setup cost at every IRLS step but
updates the factorization for the current weighted problem. Neither policy is always
faster.

PyFixest currently keeps the preconditioner returned by its first weighted demeaning
call. Our diagnostic runs the same `pyfixest.fepois` routine under two cache policies.
The reuse cell leaves PyFixest unchanged. The rebuild cell prevents that first
preconditioner from being stored, so PyFixest constructs a new one at every weighted
demeaning call. Initialization, separation handling, convergence rules, and all other
estimation code are identical.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Preconditioner reuse and rebuilding inside PPML.]
#include "generated/tables/ppml_inner_outer.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three exact PyFixest `fepois` calls
  at 1M observations. Both policies use the shipped inner tolerance of $10^(-8)$,
  an inner cap of 1,000 iterations, an outer tolerance of $10^(-8)$, and a 100-step
  outer cap. FE setup and solve times sum over the IRLS steps. At each step, PyFixest
  demeans the working response and covariate in parallel; the reported LSMR count takes
  the larger count and then sums across steps. Iteration counts come from one additional
  fit under the same policy; its diagnostic reruns are excluded from the reported time.]
]]

The two policies retain the same observations and agree on the coefficient and deviance.
They also take the same number of outer steps. On the simple design, reuse spends only
0.338 seconds on setup, compared with 3.27 seconds under rebuilding. Yet the total LSMR
count rises from 127 to 566 and FE solve time rises from 1.51 to 7.55 seconds. Rebuilding
therefore reduces the full fit from 9.25 to 6.06 seconds. On the difficult design,
rebuilding raises setup time from 0.076 to 0.845 seconds, but cuts the LSMR
count from 510 to 141 and the full fit from 5.43 to 3.75 seconds. The first-step
preconditioner becomes less effective as the IRLS weights change. Reuse remains useful
across linear regressions with unchanged weights, but rebuilding is faster in these PPML
designs.

=== Full PPML Benchmark

The full PPML comparison uses the simple-versus-difficult DGPs from the
`fixest` benchmark suite @berge2026fixest at $n = 1$M observations, with one covariate
and worker, firm, and year fixed effects. The compared backends are R `fixest`'s `fepois`,
`GLFixedEffectModels.jl`, and three PyFixest `fepois` paths: the default unpreconditioned
`rust-map`, `within` with PyFixest's current reuse policy, and `within` with the
preconditioner rebuilt at every IRLS step.
The general cross-package policy above matters particularly for PPML, where packages can
make different default choices about separated observations. Each PPML backend receives
the same 100-iteration outer IRLS limit.

#v(0.35em)

#text(size: 8.8pt)[
#strong[Poisson benchmarks (1M observations, one covariate).]
#include "generated/tables/ppml.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three full IRLS regression calls at
  $n = 1$M with one covariate and three fixed effects. `fixest` is R `fixest::fepois`;
  `rust-map` and the two `within` columns use the PyFixest `fepois` routine; `GLFEM.jl` is
  `GLFixedEffectModels.jl`. Each package applies its default separation handling; successful
  fits record the retained-observation count. `within-reuse` is the current PyFixest
  cache policy; `within-rebuild` changes only that policy. "capped" means that all
  three planned trials reached an iteration limit without converging; "failed" denotes
  another error.]
  ]

On the simple design, all five paths finish in under ten seconds. `fixest` takes 4.72
seconds, `GLFEM.jl` 5.76, rebuilt `within` 6.06, `rust-map` 7.86, and reused `within`
9.25. Runtime differences are much larger on the difficult design. `rust-map` does not converge within
the iteration cap, `GLFEM.jl` takes 129.8 seconds, and `fixest` takes 439.4 seconds.
Reused `within` finishes in 5.43 seconds and rebuilding lowers this to 3.75 seconds.
The factor-pair preconditioner handles the sparse worker-firm coupling that makes MAP
slow. Rebuilding keeps it matched to the current PPML weights. The two tables use the
same `within` runs: the first reports their demeaning costs and iteration counts, while
the second compares their total runtimes with the other packages.

#pagebreak()

= Software

The solver studied in this paper is available as open-source software through the
`within` project @within. The computational core is implemented in Rust and exposed
through Rust, Python, and R interfaces; each language binding invokes the same underlying
solver. This shared core matters for reproducibility and adoption: the same algorithm
can be called from Rust, Python, or R without reimplementation.

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

The Rust crate exposes the lower-level solver together with its configuration types. The
Python and R packages provide solver-level `solve` and `solve_batch` APIs for
residualizing one or several right-hand sides. The algorithm is also available as a demeaning backend
for PyFixest @pyfixest.

Here is a minimal Python example, in which we demean an outcome variable and two covariates against worker and firm
fixed effects, before we run the FWL fit on the demeaned variables:

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

#pagebreak()

= Conclusion

The benchmarks show that solver rankings depend on graph connectivity. MAP is difficult
to outperform on dense, well-connected graphs because its sweeps are cheap and
factor-pair construction adds overhead. With low mobility, strong sorting, or
near-nested effects, MAP passes information slowly and the factor-pair preconditioner is
often much faster.

Construction cost determines when this gain appears. It becomes cheaper in the
low-mobility AKM designs and can be paid once across regressions that share the same
sample, fixed effects, and weights. Caching lowers additive runtime on both
simple and difficult designs, but only the difficult design is faster than diagonal
preconditioning. There, caching reduces additive setup from 0.434 to 0.040 seconds and
total time from 1.77 to 1.23 seconds.

Faster fixed-effect solves also matter when one estimator calls the demeaning routine
repeatedly. PPML is one such case: every IRLS step solves a new weighted problem. In our
benchmark, rebuilding the preconditioner at each step is faster than keeping the first
factorization. Caching is therefore useful when the weights remain fixed, but it is not a
general rule for GLMs. Factor-pair preconditioning is most useful when the gap diagnostic
is small, a fit is unexpectedly slow, or an estimator requires several fixed-effect
solves.

#set heading(numbering: none)
#pagebreak()

= Appendix A: Factor-Pair Schwarz Algorithm

Algorithm 1 gives the implementation corresponding to the construction summarized in
@fig-pair-strategy.

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
        - Observation-level factor codes for $Q$ absorbed dimensions.
        - Diagonal weights $W$ and a local solver configuration.
        - Krylov residual $r$ in coefficient space.

        #strong[Preconditioner setup]
        - Enumerate all unordered factor pairs $(q,r)$ with $q < r$.
        - For each pair, build weighted count blocks $G_(q q)$, $G_(r r)$ and the weighted
          cross-tabulation $C_(q r)$.
        - Split the induced bipartite graph into connected components and create one
          Schwarz subdomain $s$ per component.
        - If fixed-effect level $j$ appears in $c_j$ subdomains, store the partition
          weight $omega_j = 1 / sqrt(c_j)$.
        - For each subdomain, sign-flip one side to obtain a local Laplacian, project the
          local right-hand side off the component constant, and build a zero-mean
          local solve.
        - Use a Schur-complement local solver: small reduced systems are solved directly,
          while larger reduced symmetric diagonally dominant (SDD) or Laplacian systems
          are solved with randomized approximate Cholesky.

        #strong[Krylov application]
        - Initialize $z = 0$.
        - For each subdomain $s$, form $h_s = tilde(D)_s R_s r$.
        - Compute the approximate local correction $u_s approx A_s h_s$ on the
          normalized subspace.
        - Accumulate $z <- z + R_s' tilde(D)_s u_s$.
        - Return $z = M^(-1) r$.
      ]
    ]
  ]
]

#pagebreak()

= Appendix B: Supplementary Benchmark Results

This appendix reports additional synthetic and real-data benchmarks, memory use, and
numerical equivalence. Each table is generated from the recorded benchmark output by
`scripts/paper_results.py`; none of the numbers are entered by hand.

== More Benchmarks

=== Standard Synthetic Benchmarks: Correia Collection

The controlled AKM benchmarks in Section 7 vary mobility and sorting directly. The
synthetic reference cases below are public, reproducible, and already used to compare
fixed-effect implementations.

The Correia HDFE benchmark collection spans a broader set of graph shapes: complete
bipartite matching, uniform random matching with varying degrees of connectivity,
assortative matching, and a small path-like design. They provide an independent public
reference set for comparing the same software backends outside the AKM generator.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia synthetic benchmarks.]
#include "generated/tables/correia_synthetic.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three runs. Gap denotes $1-rho$ for the
  `id1`-`id2` pair after the same singleton pruning; parentheses report the observation
  share of the component attaining the gap. Smaller gaps correspond to slower two-way MAP
  geometry. The `synthetic-zigzag` dataset is omitted because the default MAP backends
  hit their 10,000-iteration demeaning caps. With 10,002 observations and 10,001 fixed-effect
  levels in one connected component, only two dimensions remain after absorption, making
  this a numerical stress case rather than a useful runtime comparison.]
  ]

The Correia synthetic results are less uniform than the controlled AKM results. Several
datasets are small enough, or sufficiently well connected, that setup cost matters as
much as conditioning; on the complete and easier uniform designs, the gap is large and
low-overhead methods perform well. The harder rows reverse the ranking. By
`synthetic-uniform-harder`, the gap has fallen to $2.49 times 10^(-2)$ and `within` is
already the fastest backend. On the assortative benchmark, the preconditioned method
exhibits its clearest advantage: sorting generates cross-factor structure that MAP's
updates handle only slowly.

#pagebreak()

=== Standard Real-Data Benchmarks: Correia Collection

The Correia collection also includes real benchmark data. Synthetic data sets match the
overall shape of empirical co-occurrence graphs but smooth away irregularities that
often drive runtime: a few units that appear far more often than the rest, thin
connections between otherwise dense groups, many small disconnected pieces, and
interactions between identifiers that go beyond a single two-way pair. The empirical
datasets contain these features and therefore test the solvers in conditions that
controlled DGPs only approximate.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia real-data benchmarks.]
#include "generated/tables/correia_real.typ"
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three runs on singleton-dropped
  samples, as produced by the PyFixest benchmark suite. The gap is $1-rho$ for the
  `id1`-`id2` pair after the same singleton pruning; parentheses report the observation
  share of the component attaining the gap. A small gap with a limited component share,
  as in `directors`, indicates a hard subcomponent but not necessarily whole-sample MAP
  difficulty.]
  ]

The empirical datasets show the same pattern in less stylized form. Accelerated MAP is
difficult to outperform on small or compact graphs such as `credit` and `soccer`, where
the gaps are large. The `directors` row illustrates why the component share is useful:
the worst component is hard, but it contains about thirty percent of the observations, so
the full problem is not as costly for MAP as the gap alone would suggest. On larger
networks with hard components covering a substantial portion of the sample, particularly
`enron`, `github`, `patents`, `workers`, and `schools`, the factor-pair preconditioner is
fastest by a wide margin.

#pagebreak()

== Memory Use

MAP uses little memory: a sweep needs only the current residuals and per-level group
sums. A factor-pair preconditioner, by contrast, must retain the pair structure between
iterations. We therefore measure how much additional memory the preconditioned Krylov
solver requires relative to MAP.

We record peak resident set size (peak RSS), the largest amount of physical memory used
by the process during a run, on the simple and difficult fixed-effect DGPs. These are
representative cases rather than special memory experiments: the main storage terms are
the data matrix, the fixed-effect encodings, and the reusable factor-pair objects, so the
results should largely carry over to datasets of comparable size. We do not compare
`fixest` and `FixedEffectModels.jl` here, because peak RSS across languages also reflects
R and Julia runtime overhead, data-loading choices, garbage collection, and package
internals. Instead, we compare the preconditioned Rust backend with Rust MAP inside the
same Python package. The surrounding regression code is shared, leaving the demeaning
strategy as the relevant difference. Both backends run in isolated processes and report
peak RSS through `ru_maxrss`.

#v(0.4em)

#text(size: 8.9pt)[
#strong[Memory footprint (3 FE, one covariate).]
#include "generated/tables/memory.typ"
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] Peak RSS denotes peak resident set size, measured from
		isolated Python processes. One covariate, three fixed effects. These two DGPs serve as
		representative probes; we do not claim that memory behavior is design-specific. Gap
		denotes $1-rho_(W F)$ for the worker-firm pair in the corresponding generated design.]
		]

At 100K observations, the preconditioner adds #result_memory_100k_overhead. At 1M
observations the overhead is #result_memory_1m_overhead, but it remains modest relative
to the full panel data footprint. The additional storage holds factor-pair
co-occurrences, partition weights, and local approximate Cholesky factors.

#pagebreak()

== Numerical Equivalence

Numerical agreement is a separate diagnostic because MAP and LSMR-style routines use
different convergence checks, residual norms, stopping thresholds, and iteration caps.
Two correct implementations can therefore return coefficients that agree only up to
their tolerances.

We use the 100K-observation simple and difficult data generating processes from the
`fixest` benchmarks as diagnostic cases. All methods should agree closely on the simple
design. The difficult design is poorly conditioned, so small differences in stopping
rules are more likely to appear. The worker-firm gap is approximately 0.857 in the simple
design and 0.00130 in the difficult design. The table below compares one regression
coefficient across the four backends.

#v(0.4em)

#text(size: 9.2pt)[
#strong[Coefficient agreement (100K observations, 3 FE, one covariate).]
#include "generated/tables/agreement.typ"
#v(0.25em)
#text(size: 8.2pt)[#emph[Note:] $hat(beta)_1$ is the slope coefficient on `x1`.
Differences are absolute slope-coefficient deviations from `rust-map`. `within` is the
PyFixest preconditioned Rust backend.]
]

On the simple design, the largest coefficient difference is
#result_agreement_simple_max. On the difficult design it is
#result_agreement_difficult_max. The methods use different stopping checks, so exact
agreement is not expected on the poorly conditioned design.

#pagebreak()

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

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
#let table-rule = rgb("#7b8494")
#let table-light-rule = rgb("#d8dee8")
#let table-head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]
#let miss = text(fill: rgb("#777777"))[--]
#let dg(body) = text(fill: rgb("#2563eb"), body)
#let cr(body) = text(fill: rgb("#c2410c"), body)

#align(center)[
  #set par(justify: false)
  #text(size: 18.5pt, weight: "bold")[Graph Preconditioning for]

  #v(0.18em)
  #text(size: 18.5pt, weight: "bold")[High-Dimensional Fixed Effects Regression]

  #v(0.65em)
  #text(size: 10.5pt)[Alexander Fischer#footnote[trivago] and Kristof Schröder#footnote[appliedAI Institute for Europe gGmbH]]

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
      Benchmarks show MAP remains fastest on dense designs, while graph preconditioning
      reduces runtime on sparse worker-firm graphs and under strong sorting.
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
contribution is the preconditioner, not the use of LSMR for the outer iteration.] In a range of benchmarks
against mature implementations of the method of alternating projections, we find that on
sparse, poorly connected graphs (the regime where MAP convergence deteriorates) the
graph-preconditioned solver lowers runtime, while on dense,
well-connected graphs the preconditioner setup cost does not amortize and MAP should
remain the natural default.

The rest of the paper is organized as follows. Section 2 sets up the fixed-effect
absorption problem, and Section 3 introduces the AKM model as our running example.
Section 4 develops the graph structure of the fixed-effect Gramian, and Section 5
connects this structure to the convergence behavior of MAP. Section 6 builds up
the factor-pair Schwarz preconditioner, starting from a general discussion of
preconditioning and culminating in the construction of the graph-based preconditioner.
Section 7 reports benchmarks on runtime, memory, and numerical equivalence; 
Section 8 describes the software through which the new algorithm is available; 
and Section 9 concludes.

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
interest. Separate tables report the memory cost of storing factor-pair information and
verify that the preconditioned solver matches MAP to numerical tolerance.

The runtime tables also report the pairwise hardness statistic for the relevant factor
pair and the observation share of the component attaining it. Smaller gaps indicate
factor pairs on which MAP tends to converge more slowly; with three or more fixed
effects, the statistic remains a pairwise diagnostic rather than a full convergence
bound.

The benchmark section moves from controlled AKM designs to standard synthetic benchmarks
and then to empirical datasets. The AKM designs vary mobility and sorting directly. The
standard synthetic benchmarks use the simple/difficult DGPs from the fixest benchmark
suite and synthetic datasets from the HDFE benchmark collection assembled by Sergio
Correia. The empirical datasets from the same collection are included because synthetic
DGPs can miss irregularities in real fixed-effect structures.

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

All reported CPU times are medians across three benchmark iterations run on an Apple M4
Mac mini with 10 CPU cores and 16 GB of memory running macOS 15.3.1.

We do not include `reghdfe` @reghdfe @correia2017 directly in the benchmark tables because Stata is not open
source and we lack a license. `reghdfe` is a mature accelerated-MAP
implementation; algorithmically, it belongs to the same
family as `fixest`'s accelerated MAP.

== Runtime Benchmarks

=== Controlled Synthetic Benchmarks: AKM Mobility and Sorting

We start with synthetic AKM-style panels, which let us vary one graph feature at a time
while holding the rest of the data-generating process fixed. Changes in runtime then
reflect the fixed-effect graph.

In the first synthetic design, we lower worker mobility across firms. As mobility falls,
MAP still updates one fixed-effect dimension at a time, but fewer workers connect
multiple firms, so information about firm effects moves more slowly through the
iteration.
The factor-pair preconditioner uses the worker-firm graph directly, so the preconditioned
Krylov solver should become relatively more competitive in the low-mobility designs.

#v(0.4em)

#text(size: 8.9pt)[
#strong[Mobility benchmark ($n = 1$M).]
#table(
  columns: (1.25fr, 1.05fr, 0.78fr, 0.78fr, 0.78fr, 0.78fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Scenario], th[Gap (share)], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`akm_mobility_1`], [$1.38 times 10^(-1)$ (1.00)], [0.29s], [0.16s], [0.28s], [0.51s],
  [`akm_mobility_2`], [$3.41 times 10^(-2)$ (1.00)], [1.10s], [0.36s], [0.58s], [0.38s],
  [`akm_mobility_3`], [$5.18 times 10^(-3)$ (0.61)], [12.48s], [1.31s], [1.80s], [0.33s],
  [`akm_mobility_4`], [$3.04 times 10^(-4)$ (0.27)], [51.95s], [3.42s], [2.78s], [0.34s],
  [`akm_mobility_5`], [$1.65 times 10^(-3)$ (0.53)], [63.23s], [4.17s], [3.34s], [0.35s],
		  [`akm_mobility_6`], [$7.57 times 10^(-4)$ (0.37)], [#miss], [5.27s], [3.86s], [0.35s],
		  table.hline(stroke: 0.8pt + table-rule),
		)
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
		and worker, firm, and year fixed effects. Lower rows reduce worker mobility
		within a 10-period panel, thereby thinning the worker-firm graph. Lower mobility
		makes the worker-firm pair harder for MAP and correspondingly more favorable
		to the preconditioned method. A dash indicates that no completed run is available
		for that cell. Gap is defined as $1-rho_(W F)$ for the worker-firm pair;
		parentheses report the observation share of the component attaining the gap.]
		]

The mobility benchmark matches the predicted pattern. When mobility is high,
preconditioning yields little advantage: `fixest` is the fastest backend in
`akm_mobility_1`, and `within` incurs setup costs that are not offset by improved
convergence. As mobility declines, the worker-firm gap falls by more than two orders
of magnitude, from $1.38 times 10^(-1)$ to below $10^(-3)$ (the decline is not
strictly monotone, because the reported gap is attained on different connected
components), and MAP runtimes rise sharply. `within`'s runtime stays nearly flat across
all designs. In the lowest-mobility configurations, the
preconditioned method is the fastest backend because it operates on the worker-firm graph
directly rather than propagating information through many sweeps that update one
fixed-effect dimension at a time.

#v(0.4em)

The second experiment increases sorting between workers and firms. Stronger sorting
pushes the worker-firm graph toward weakly connected blocks: movers increasingly connect
firms within the same group rather than linking different groups. MAP should therefore
slow down again, because information about firm effects crosses groups only through a
small number of movers. The preconditioned method should be less sensitive, since its
factor-pair solves use the worker-firm graph directly.

#v(0.4em)

#text(size: 8.9pt)[
#strong[Sorting benchmark ($n = 1$M).]
#table(
  columns: (1.25fr, 1.05fr, 0.78fr, 0.78fr, 0.78fr, 0.78fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Scenario], th[Gap (share)], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`akm_sorting_1`], [$7.62 times 10^(-3)$ (0.83)], [6.79s], [0.81s], [1.43s], [0.32s],
  [`akm_sorting_2`], [$2.41 times 10^(-3)$ (0.62)], [9.73s], [1.12s], [1.68s], [0.34s],
  [`akm_sorting_3`], [$1.50 times 10^(-4)$ (0.59)], [10.31s], [1.24s], [1.67s], [0.33s],
  [`akm_sorting_4`], [$2.14 times 10^(-5)$ (0.51)], [14.94s], [1.58s], [1.87s], [0.33s],
		  [`akm_sorting_5`], [$3.97 times 10^(-5)$ (0.53)], [25.77s], [1.80s], [1.94s], [0.34s],
		  table.hline(stroke: 0.8pt + table-rule),
		)
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
		and worker, firm, and year fixed effects. Lower rows raise the degree of
		sorting among movers, pushing the worker-firm graph toward weakly connected blocks.
		Stronger sorting weakens cross-block information flow and thereby raises the value
		of factor-pair preconditioning. Gap is $1-rho_(W F)$ for the worker-firm pair;
		parentheses report the observation share of the component attaining the gap.]
		]

The sorting benchmark gives the parallel result. As movers sort more strongly across
firms, the worker-firm graph separates into weakly connected blocks. The worker-firm gap
falls by more than two orders of magnitude, and MAP-based runtimes rise. The gap does not fall
monotonically, because different components attain the reported value in different rows,
but its overall decline is clear. `within` remains nearly flat because its factor-pair
preconditioner uses the worker-firm links directly, rather than relying on residual
updates that cycle through one fixed-effect dimension at a time.

=== Standard Synthetic Benchmarks: fixest DGPs + Correia Synthetic

The AKM benchmarks varied mobility and sorting directly. We now turn to synthetic
datasets that have become standard reference cases in fixed-effect software: they are
public, easily reproducible, and already used to compare implementations.

The first family is the simple-versus-difficult benchmark data generating process from
`fixest` @berge2026fixest. Both designs use 10M observations, one covariate, and three
fixed effects (worker, firm, year). The simple design has dense random mobility, whereas
the difficult design has a sparse, nearly nested worker-firm structure. The simple design
should be easy for MAP because its updates can pass information rapidly through a
well-connected graph. The
difficult design should be harder because worker and firm effects are nearly collinear.
Factor-pair preconditioning should perform well on the difficult design, but may fail to
amortize its setup cost on the simple design. For this comparison we add a fifth
backend, `torch-cuda`: the PyFixest GPU backend, which runs the same diagonally
preconditioned LSMR as `FEM.jl` on an NVIDIA CUDA device.

#text(size: 8.8pt)[
#strong[Simple vs. difficult design (10M observations, 3 FE).]
#table(
  columns: (1.25fr, 1.05fr, 0.72fr, 0.72fr, 0.72fr, 0.78fr, 0.86fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[Gap (share)], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`], th[`torch-cuda`]),
  table.hline(stroke: 0.45pt + table-rule),
  [simple (dense graph)], [$8.57 times 10^(-1)$ (1.00)], [2.01s], [0.93s], [2.27s], [10.53s], [4.73s],
  [difficult (sparse graph)], [$1.67 times 10^(-5)$ (1.00)], [382.9s], [32.7s], [28.7s], [3.25s], [8.73s],
  table.hline(stroke: 0.8pt + table-rule),
  )
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three full regression calls. Both designs
  use 10M observations, one covariate, and three fixed effects. Gap denotes
  $1-rho_(W F)$ for the worker-firm pair, reported from the generated 1M version of the
  same DGP family; the runtime rows use the identical simple/difficult graph construction
  at 10M observations. `torch-cuda` denotes the PyFixest GPU LSMR backend with diagonal
  preconditioning (the same algorithm as
`FixedEffectModels.jl`), run on an NVIDIA CUDA device; the local benchmark machine does
not possess a CUDA GPU. The standalone `within` demeaning API decomposes one-shot runtime
into reusable solver construction and batch solve: simple design 5.76s + 1.51s
(roughly 80% setup); difficult design 0.40s + 1.00s (roughly 29% setup). PyFixest
regression overhead is incurred in addition to these figures.]
  ]

On the "simple" design, the graph is dense and the worker-firm gap is large. MAP
therefore converges quickly: `fixest` with Irons-Tuck acceleration is fastest, while
`within` is slowest because the factor-pair preconditioner does not repay its setup cost.
In the standalone demeaning breakdown, roughly four fifths of `within`'s demeaning time
on the simple design is spent building the preconditioner.

On the difficult design, the ranking reverses. MAP convergence degrades sharply, to
the point that `rust-map` without acceleration needs over six minutes.
`within` completes in 3.25s, an order of magnitude faster than the best MAP backend,
because its factor-pair preconditioner captures the sparse worker-firm coupling
directly. A nearly zero diagnostic gap identifies the regime in which MAP requires many
sweeps to disentangle worker and firm effects. The setup share of the standalone
demeaning time falls to about 29%, indicating that the preconditioner setup is now
amortized.

On the simple design, `torch-cuda` is not
competitive with CPU `fixest`, since host-to-device transfer and
kernel launch overheads outweigh the gain from parallelizing already inexpensive
iterations. On the difficult design, GPU parallelism reduces runtime to 8.73s (roughly
three times faster than CPU `FEM.jl`), but the iteration count remains governed by the
quality of the diagonal preconditioner, which is limited on this graph. `within` at 3.25s
remains faster on CPU by using a factor-pair preconditioner that captures the
cross-factor coupling the diagonal cannot. Hardware acceleration and stronger
preconditioning address different bottlenecks: the GPU speeds up each iteration,
whereas on poorly conditioned designs the iteration count, which only the
preconditioner controls, dominates runtime.

#v(0.35em)

The second family of benchmarks comprises synthetic datasets drawn from the Correia HDFE
benchmark collection. These datasets span a broader set of graph shapes: complete
bipartite matching, uniform random matching with varying degrees of connectivity,
assortative matching, and a small path-like design. They provide an independent public
reference set for comparing the same software backends outside the AKM generator.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia synthetic benchmarks.]
#table(
  columns: (1.5fr, 1.0fr, 0.75fr, 0.75fr, 0.85fr, 0.8fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Dataset], th[Gap (share)], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`synthetic-complete`], [1.000 (1.00)], [0.074s], [0.062s], [0.044s], [0.114s],
  [`synthetic-uniform-easy`], [0.651 (1.00)], [0.122s], [0.079s], [0.083s], [0.117s],
  [`synthetic-uniform-hard`], [0.184 (1.00)], [0.367s], [0.251s], [0.353s], [0.957s],
  [`synthetic-uniform-harder`], [0.0249 (1.00)], [0.994s], [0.952s], [0.575s], [0.514s],
  [`synthetic-assortative`], [0.00133 (0.70)], [26.22s], [3.02s], [2.99s], [1.46s],
  table.hline(stroke: 0.8pt + table-rule),
  )
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three runs. Gap denotes $1-rho$ for the
  `id1`-`id2` pair after the same singleton pruning; parentheses report the observation
  share of the component attaining the gap. Smaller gaps correspond to slower two-way MAP
  geometry. The `synthetic-zigzag` dataset is omitted because every backend terminates
  with a numerical error rather than a converged solution under default settings, making
  it a stress case rather than a runtime comparison.]
  ]

These results are less clear than the controlled AKM experiments, which is itself informative.
Several datasets are small enough, or sufficiently well connected, that setup cost
matters as much as conditioning; on the complete and easier uniform designs, the gap is
large and low-overhead methods perform well. The harder rows reverse the ranking. By
`synthetic-uniform-harder`, the gap has fallen to $2.49 times 10^(-2)$ and `within` is
already the fastest backend. On the assortative benchmark, the preconditioned method
exhibits its clearest advantage: sorting generates cross-factor structure that MAP's
updates handle only slowly.

=== Standard Real-Data Benchmarks: Correia Collection

We next turn to real benchmark data from the Correia collection. Synthetic data sets
match the overall shape of empirical co-occurrence graphs but smooth away the
irregularities that often drive runtime: a few units that appear far more often than the rest,
thin connections between otherwise dense groups, many small disconnected pieces, and
interactions between identifiers that go beyond a single two-way pair. Real data carries
these features by default, and therefore tests the solvers in conditions that controlled
DGPs only approximate.

#v(0.35em)

#text(size: 8.9pt)[
#strong[Correia real-data benchmarks.]
#table(
  columns: (1.15fr, 1.0fr, 0.75fr, 0.75fr, 0.85fr, 0.8fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Dataset], th[Gap (share)], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`credit`], [0.402 (1.00)], [0.177s], [0.134s], [0.158s], [0.139s],
  [`soccer`], [0.889 (1.00)], [0.018s], [0.014s], [0.016s], [0.031s],
  [`enron`], [0.00704 (0.98)], [4.37s], [0.749s], [0.689s], [0.537s],
  [`github`], [0.000630 (0.32)], [86.44s], [4.50s], [4.60s], [0.380s],
  [`patents`], [0.000518 (0.95)], [19.66s], [1.29s], [1.43s], [0.826s],
  [`workers`], [0.000274 (0.63)], [128.34s], [5.55s], [4.94s], [0.491s],
  [`schools`], [0.00221 (1.00)], [11.04s], [1.10s], [1.28s], [0.224s],
  [`directors`], [0.000512 (0.30)], [4.26s], [0.216s], [1.21s], [0.300s],
  table.hline(stroke: 0.8pt + table-rule),
  )
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

== Poisson / PPML Benchmark

Generalized linear models with high-dimensional fixed effects are typically estimated by
iteratively reweighted least squares (IRLS). Each IRLS step fits a weighted least squares
problem in which the response and covariates are demeaned against the fixed effects, with
weights that are updated between iterations @correia2020ppmlhdfe @stammann2018. The
demeaning operation is identical to the process described in the prior sections. 
Any acceleration of fixed-effect demeaning therefore propagates into the GLM runtime,
multiplied by the number of IRLS iterations.

=== Preconditioner reuse across IRLS

The IRLS structure also enlarges the window over which a preconditioner setup cost
can amortize. A factor-pair preconditioner depends on the fixed-effect graph
(invariant across IRLS iterations) and on the IRLS weights (which do change between
iterations). If the weights do not move much, a slightly stale preconditioner is
still effective. We exploit this property by building the preconditioner once and reusing the
stale version on subsequent IRLS iterations. Staleness slows the outer Krylov solver
but does not bias its solution; the iteration still converges to the correct demeaned
residuals. The construction cost is therefore paid once per regression rather than
once per IRLS step. 

We benchmark this strategy on the simple-versus-difficult DGPs from the `fixest`
benchmark suite @berge2026fixest at $n = 1$M observations, $k = 10$ covariates, and two
or three fixed effects (worker-year, or worker-firm-year), using the same iteration
protocol as the OLS benchmarks. The compared backends are R `fixest`'s `fepois`,
`GLFixedEffectModels.jl`, and two PyFixest `fepois` backends: the default unpreconditioned
`rust-map` and the preconditioned `within` solver.

#v(0.35em)

#text(size: 8.8pt)[
#strong[Poisson benchmarks (1M observations, k=10 covariates).]
#table(
  columns: (1.25fr, 0.55fr, 0.78fr, 0.78fr, 0.85fr, 0.78fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[FE], th[`fixest`], th[`rust-map`], th[`GLFEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [simple (dense graph)], [2], [1.92s], [3.06s], [8.83s], [6.21s],
  [difficult (sparse graph)], [2], [1.91s], [3.05s], [7.93s], [6.40s],
  [simple (dense graph)], [3], [7.80s], [25.29s], [#sym.tilde.op 48s], [13.33s],
  [difficult (sparse graph)], [3], [309.2s], [failed], [#sym.tilde.op 800s], [8.89s],
  table.hline(stroke: 0.8pt + table-rule),
  )
  #v(0.25em)
  #text(size: 8.2pt)[#emph[Note:] Medians over three full IRLS regression calls at
  $n = 1$M and $k = 10$ covariates. `fixest` is R `fixest::fepois`; `rust-map` and
  `within` are the PyFixest `fepois` routine with the unpreconditioned MAP backend and
  the factor-pair preconditioned solver, respectively; `GLFEM.jl` is
  `GLFixedEffectModels.jl`. The two `GLFEM.jl` entries marked #sym.tilde.op are read
  approximately from a follow-up run because the harness CSV for those cells contained a
  subprocess error rather than a recorded time. "failed" indicates that all three
  iterations of `rust-map` reached the 10000-iteration MAP cap without converging.]
  ]

The patterns observed in the OLS benchmarks carry over. With two fixed effects, all four
backends complete in comparable time on both designs and R `fixest` is fastest: the
worker-year pair alone does not create a regime in which the preconditioner amortizes
its setup cost, even with the IRLS-induced reuse. With a third fixed effect, the picture
changes. On the three-FE simple design MAP still converges and `fixest` remains
fastest, but the unaccelerated `rust-map` slows to 25s, `GLFEM.jl` to roughly 48s, and
`within` lands between them at 13s. The three-FE difficult design provides the sharpest
contrast: `rust-map` does not converge within the iteration cap, `GLFEM.jl` takes on the
order of 800s, `fixest`'s IRLS loop takes several minutes with large run-to-run variance,
and `within` finishes in under nine seconds, roughly 35 times faster than `fixest` and
nearly two orders of magnitude faster than `GLFEM.jl`. The factor-pair preconditioner captures
the same sparse worker-firm coupling that drove the OLS benchmark results, and the IRLS outer
loop inherits the gain without modification.

== Memory Use

MAP uses little memory: a sweep needs only the current residuals and per-level group
sums. A factor-pair preconditioner, by contrast, must retain the pair structure between
iterations. A practically relevant question is therefore how much more memory the 
preconditioned Krylov solver requires relative to MAP.

We measure peak resident set size (peak RSS), the largest quantity of physical memory
consumed by the process during a run, on the simple and difficult fixed-effect DGPs.
These serve as representative probes rather than special memory cases: the dominant
storage terms are the data matrix, the fixed-effect encodings, and the reusable
factor-pair objects, so the results should largely generalize across datasets of
comparable dimensions. We do not use this section to compare against `fixest` or
`FixedEffectModels.jl`, because cross-language peak RSS also reflects R and Julia runtime
overhead, data-loading choices, garbage collection, and package internals. The diagnostic
of interest is the incremental memory cost of substituting the preconditioned Rust
backend for Rust MAP within the same Python package. Because the surrounding regression
code is shared, this comparison isolates the difference attributable to the demeaning
strategy. Both backends are executed in isolated processes and report peak RSS via
`ru_maxrss`.

#v(0.4em)

#text(size: 8.9pt)[
#strong[Memory footprint (3 FE, $k = 10$).]
#table(
  columns: (1.25fr, 1.0fr, 0.85fr, 0.85fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[Gap (share)], th[`rust-map`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  table.cell(colspan: 4, fill: table-head-fill)[#emph[100K observations]],
  [simple (dense graph)], [$8.57 times 10^(-1)$ (1.00)], [428 MB], [487 MB],
  [difficult (sparse graph)], [$1.30 times 10^(-3)$ (1.00)], [432 MB], [479 MB],
  table.hline(stroke: 0.35pt + table-light-rule),
  table.cell(colspan: 4, fill: table-head-fill)[#emph[1M observations]],
  [simple (dense graph)], [$8.57 times 10^(-1)$ (1.00)], [1,188 MB], [1,703 MB],
		  [difficult (sparse graph)], [$1.67 times 10^(-5)$ (1.00)], [1,263 MB], [1,398 MB],
		  table.hline(stroke: 0.8pt + table-rule),
		)
		#v(0.25em)
		#text(size: 8.2pt)[#emph[Note:] Peak RSS denotes peak resident set size, measured from
		isolated Python processes. 10 covariates, three fixed effects. These two DGPs serve as
		representative probes; we do not claim that memory behavior is design-specific. Gap
		denotes $1-rho_(W F)$ for the worker-firm pair in the corresponding generated design.]
		]

At 100K observations, the preconditioner adds roughly 50 MB on both the easy and the
hard graph. At 1M observations the overhead is larger in absolute terms (135--515 MB),
but it remains modest relative to the data footprint of a panel with 10 covariates. 
The preconditioned solver consumes more memory than MAP, yet the additional
storage for factor-pair co-occurrences, partition weights, and local approximate
Cholesky factors remains small relative to
the full regression problem.

== Numerical Equivalence

Before trusting a new fixed-effect solver, we must verify that it reproduces the
estimates of existing routines. MAP and LSMR-style routines differ in their convergence
checks, residual norms, stopping thresholds, and iteration caps, so two correct
implementations can return coefficients that agree only up to their tolerances.

We use the 100K-observation simple and difficult data generating processes from the fixest
benchmarks as diagnostic cases. The simple
design examines the "easy-to-converge" regime where all methods should agree almost exactly. The
difficult design examines the regime where conditioning is poor and small differences in
stopping rules are most likely to emerge. The graph diagnostic distinguishes these
cases: the worker-firm gap is approximately 0.857 in the simple design and
approximately 0.00130 in the difficult design. The table below collects one regression
coefficient for all four backends.

#v(0.4em)

#text(size: 9.2pt)[
#strong[Coefficient agreement (100K observations, 3 FE, $k = 10$).]
#table(
  columns: (0.95fr, 1.0fr, 0.95fr, 0.95fr, 0.95fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, left, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[Backend], th[$hat(beta)_1$], th[Avg |diff|], th[Max |diff|]),
  table.hline(stroke: 0.45pt + table-rule),
  table.cell(rowspan: 4)[simple],
  [`rust-map`], [0.99914206], [--], [--],
  [`within`], [0.99914206], [$1.5 times 10^(-14)$], [$3.5 times 10^(-14)$],
  [`fixest`], [0.99909173], [$2.2 times 10^(-5)$], [$5.0 times 10^(-5)$],
  [`FEM.jl`], [0.99914206], [$2.1 times 10^(-12)$], [$5.3 times 10^(-12)$],
  table.hline(stroke: 0.35pt + table-light-rule),
  table.cell(rowspan: 4)[difficult],
  [`rust-map`], [1.00086122], [--], [--],
  [`within`], [1.00086098], [$1.4 times 10^(-7)$], [$3.4 times 10^(-7)$],
  [`fixest`], [1.00081490], [$2.0 times 10^(-5)$], [$5.0 times 10^(-5)$],
  [`FEM.jl`], [1.00086098], [$1.4 times 10^(-7)$], [$3.4 times 10^(-7)$],
  table.hline(stroke: 0.8pt + table-rule),
)
#v(0.25em)
#text(size: 8.2pt)[#emph[Note:] $hat(beta)_1$ is the slope coefficient on `x1`.
Differences are absolute slope-coefficient deviations from `rust-map`, averaged
(Avg) and maximized (Max) across all 10 covariates. `within` is the PyFixest
preconditioned Rust backend.]
]

The `within` and `FEM.jl` rows are almost identical to `rust-map` at the reported
defaults. R `fixest` is also close, but not at machine precision: the largest
coefficient difference is approximately $5 times 10^(-5)$. This difference is not
surprising, because the packages use different stopping rules.

= Software

The solver studied in this paper is available as open-source software through the
`within` project @within. The computational core is implemented in Rust and exposed
through Rust, Python, and R interfaces; each language binding invokes the same underlying
solver. This shared core matters for reproducibility and adoption: the same algorithm
can be called from Rust, Python, or R without reimplementation.

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
  [Rust], [`within`], [#link("https://crates.io/crates/within")[crates.io]], [`cargo add within`],
  [Python], [`within-py`], [#link("https://pypi.org/project/within-py/")[PyPI]], [`pip install within-py`],
  [R], [`withinr`], [#link("https://py-econometrics.r-universe.dev/withinr")[py-econometrics R-universe]], [`install.packages("withinr", repos = "https://py-econometrics.r-universe.dev")`],
  table.hline(stroke: 0.8pt + table-rule),
)
]

The Rust crate exposes the lower-level solver together with its configuration types. The
Python and R packages provide solver-level `solve` and `solve_batch` APIs for
residualizing one or several right-hand sides. The algorithm is also available as a demeaning backend
for PyFixest @pyfixest.

Here is a minimal Python example, in which we demean an outcome variable and two covariates against worker and firm
fixed effects, before we run the FWL fit on the demeaned variables:

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

We have developed a graph-based preconditioner for fixed-effect demeaning and compared
it with MAP on synthetic and empirical benchmarks. Which of the two solvers is faster depends on 
the structure of the fixed effects graph.

When the graph is dense and well connected, MAP is difficult to outperform. Its sweeps are cheap, 
and a preconditioner mostly adds overhead, as
in the simple fixest design and the smaller, well-connected Correia datasets. When a
factor pair is sparsely connected, through low mobility, strong sorting, or near-nesting,
MAP passes information across the graph slowly, and the factor-pair preconditioner is
faster, often by a wide margin. 

The preconditioner depends only on the fixed-effect graph, so it can be built once and
reused across demeaning calls. Reuse matters most when a single estimation issues many
such calls: IRLS-based GLMs such as PPML demean once per iteration, so the construction
cost is paid once while the faster convergence can accrue at every iteration step. In our PPML
benchmark on a hard three-way design, `within` finishes in seconds where the MAP-based
routines take minutes or fail to converge.

Which solver to prefer depends on the fixed-effect graph. Accelerated MAP
and diagonally preconditioned LSMR are good defaults across much of the range our
benchmarks cover: they carry (almost) no setup cost, and on dense, well-connected graphs they
are the fastest options. The factor-pair preconditioner amortizes its setup cost in the
remaining cases, and we recommend it when the gap diagnostic is small, when a fit is
unexpectedly slow, or for IRLS-based GLM, which repeat the demeaning step many times.

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

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

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
// #show heading.where(level: 1): it => {
//   set block(above: 1.45em, below: 0.68em)
//   text(size: 15pt, weight: "bold", it)
// }
// #show heading.where(level: 2): it => {
//   set block(above: 1.1em, below: 0.48em)
//   text(size: 12pt, weight: "bold", it)
// }

#let solver-img(name) = "figures/solver/" + name
#let result-img(name) = "figures/results/" + name
#let table-rule = rgb("#7b8494")
#let table-light-rule = rgb("#d8dee8")
#let table-head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]
#let miss = text(fill: rgb("#777777"))[--]
#let dg(body) = text(fill: rgb("#2563eb"), body)
#let cr(body) = text(fill: rgb("#c2410c"), body)
#let table-note(body) = block(below: 0.7em)[
  #set text(size: 7.8pt)
  #set par(leading: 0.94em, spacing: 0em)
  #emph[Note:] #body
]
#import "generated/paper_values.typ": *

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
      On a near-nested design with
      10 million observations, factor-pair LSMR completes in 4.63 seconds, compared with
      62.2 seconds for the fastest MAP implementation.

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

= Introduction <sec:introduction>

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

We follow #cite(<jochmans2019fixed>, form: "prose") and use the second-smallest
eigenvalue of the normalized Laplacian as our measure of connectivity. For factor pair
$(q,r)$, let $L_(q r)$ be the weighted graph Laplacian and $Delta_(q r)$ its degree
matrix. Then

$ S_(q r) = Delta_(q r)^(-1/2) L_(q r) Delta_(q r)^(-1/2). $

Our measure of connectivity for the factor pair is $lambda_2(S_(q r))$. Edge weights are
observed co-occurrence counts, and a larger value indicates stronger connectivity between
the two fixed-effect dimensions. Our empirical results report our measure of connectivity
for the worker-firm graph.#footnote[We compute our measure of connectivity after iterative singleton removal. If the
remaining graph is disconnected, we use the component containing the most retained
observations and report its observation share; Section 7 gives the full convention.]


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

The pair factorizations have a setup cost, so they are unlikely to help when MAP already
converges in a few passes. On a weakly connected graph, however, they can save substantial
iteration time. @fig-gap-runtime reports total regression time as mobility and sorting
change: the left column compares package defaults, whereas the right holds the package
and achieved accuracy fixed within PyFixest using tolerances calibrated in Section 7.

#figure(
  image(result-img("gap_runtime.svg"), width: 100%),
  caption: [Median regression time against our measure of connectivity for the
  worker-firm graph, $lambda_(2,W F)$, on logarithmic axes. The measure is computed after
  singleton removal on the connected
  component containing the most observations; larger values mean stronger connectivity,
  and the shared reversed horizontal scale makes connectivity weaken from left to right.
  Each simulated worker-firm-year panel has 1 million observations, with worker mobility
  varying in the top row and sorting among movers in the bottom row. The left column
  compares package defaults, including each package's singleton rule, while the right
  compares four PyFixest configurations at tolerances that give comparable coefficient
  and residual errors in Section 7. A filled marker means all three planned fits returned
  estimates; a hollow marker means one or two did. Lines join observed medians rather
  than fitted trends, and an arrow marks the median time at the iteration cap, a lower
  bound on completion time; other failures are omitted.]
) <fig-gap-runtime>

All implementations finish quickly when our measure of connectivity is high. MAP and
unpreconditioned LSMR slow sharply as the measure falls; factor-pair LSMR remains fast,
and its runtime falls in the
lowest-mobility designs because the local graphs become cheaper to factorize.


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
$<eq:fixed-effect-design-matrix>
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
identification of worker and firm effects @bonhommeHowMuchShould2023. For a
worker observed at only one firm, a high wage could reflect an unusually
productive worker, a high-wage firm, or both. A worker earning high wages
across multiple firms provides evidence of a worker effect, while different
workers earning high wages at the same firm provide evidence of a firm
premium. The more cross-firm comparisons the data contain, the easier it
becomes to separately identify worker effects and firm premia. This is why
the identifying variation generated by movers corresponds to the
connectedness of the fixed-effect graph.

= The Graph Structure of the Gramian <sec:gramian>

The bipartite graph of worker and firm connections introduced in @sec:akm
is represented algebraically in the block structure of the Gramian
$G = D' W D$ derived in @eq:fwl-normal @correia2017. Suppose that the
columns of $D$ are ordered as worker levels, firm levels, and year levels.
Then, the Gramian has the block structure

$
  G = mat(
    dg(G_(W W)), cr(C_(W F)), cr(C_(W Y));
    cr(C_(W F)'), dg(G_(F F)), cr(C_(F Y));
    cr(C_(W Y)'), cr(C_(F Y)'), dg(G_(Y Y))
  ).
$<eq:gramian-blocks>

The #dg[diagonal blocks] $#dg[$G_(W W)$]$, $#dg[$G_(F F)$]$, and
$#dg[$G_(Y Y)$]$ contain the weighted counts for each worker, firm, and
year, respectively. Because one observation belongs to exactly one level of
each fixed-effect dimension, these blocks are diagonal and solving them
only requires division by weighted group counts. The #cr[off-diagonal
  blocks] are cross-tabulations: the worker-firm block $#cr[$C_(W F)$]$
records how often worker $i$ is observed at firm $j$ with analogous
interpretations for the worker-year block $#cr[$C_(W Y)$]$ and the
firm-year block $#cr[$C_(F Y)$]$.

#figure(
  table(
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
  ),
  caption: [Synthetic worker-firm panel],
)<tab:example>

As an instructive example, we construct a small synthetic worker-firm panel
in @tab:example and populate its Gramian $G$. Throughout the example, we
assume an unweighted regression with $W = I$. @fig-toy-projection
illustrates the bipartite worker-firm graph of this panel. Worker $W_1$ is
observed at two firms $F_1$ and $F_2$, creating a link between them, while
workers $W_2$ and $W_3$ are each observed at a single firm. In AKM terms,
$W_1$ is a mover and $W_2$ and $W_3$ are stayers.

#figure(
  image(solver-img("toy_worker_firm_projection.svg"), width: 50%),
  caption: [Worker-firm projection of the example panel. Worker $W_1$ is a
    mover; workers $W_2$ and $W_3$ are stayers.],
) <fig-toy-projection>


Because the regression is unweighted, $W=I$, the diagonal blocks of the
Gramian in @eq:gramian-blocks are simple counts. @tab:example shows that
each worker is observed twice, each firm three times, and each year three
times, so that

$
  dg(G_(W W)) = mat(dg(2), dg(0), dg(0); dg(0), dg(2), dg(0); dg(0), dg(0), dg(2)), quad
  dg(G_(F F)) = mat(dg(3), dg(0); dg(0), dg(3)), quad
  dg(G_(Y Y)) = mat(dg(3), dg(0); dg(0), dg(3)).
$

The off-diagonal blocks are cross-tabulations between fixed-effect
dimensions. The worker-firm block is

$
  cr(C_(W F)) = mat(
    cr(1), cr(1);
    cr(2), cr(0);
    cr(0), cr(2)
  ).
$

The first row indicates that worker $W_1$ appears once at firm $F_1$ and
once at firm $F_2$. Workers $W_2$ and $W_3$ are stayers, appearing twice at
$F_1$ and $F_2$, respectively. The worker-year block is

$
  cr(C_(W Y)) = mat(
    cr(1), cr(1);
    cr(1), cr(1);
    cr(1), cr(1)
  ).
$

Each worker is observed once in each year. The firm-year block is

$
  cr(C_(F Y)) = mat(
    cr(2), cr(1);
    cr(1), cr(2)
  ).
$

Firm $F_1$ appears twice in year $Y_1$ and once in year $Y_2$, and firm
$F_2$ has the opposite pattern. With column order
$(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$, the full Gramian is

$
  G = mat(
    augment: #(hline: (3, 5), vline: (3, 5), stroke: 0.4pt + rgb("#b0b8c4")),
    dg(2), dg(0), dg(0), cr(1), cr(1), cr(1), cr(1);
    dg(0), dg(2), dg(0), cr(2), cr(0), cr(1), cr(1);
    dg(0), dg(0), dg(2), cr(0), cr(2), cr(1), cr(1);
    cr(1), cr(2), cr(0), dg(3), dg(0), cr(2), cr(1);
    cr(1), cr(0), cr(2), dg(0), dg(3), cr(1), cr(2);
    cr(1), cr(1), cr(1), cr(2), cr(1), dg(3), dg(0);
    cr(1), cr(1), cr(1), cr(1), cr(2), dg(0), dg(3)
  ).
$

The submatrix comprising the worker-worker, worker-firm, and firm-firm
blocks algebraically represents the bipartite graph formed by the worker
and firm fixed effects. The diagonal entries are worker and firm counts,
and the entries of $C_(W F)$ are edge multiplicities between workers and
firms. After flipping the sign of $C_(W F)$, the worker-firm submatrix is
the graph Laplacian of the bipartite graph
$
  L_(W F) = mat(
    dg(G_(W W)), cr(-C_(W F));
    cr(-C_(W F)'), dg(G_(F F));
  )
  = mat(
    augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
    dg(2), dg(0), dg(0), cr(-1), cr(-1);
    dg(0), dg(2), dg(0), cr(-2), cr(0);
    dg(0), dg(0), dg(2), cr(0), cr(-2);
    cr(-1), cr(-2), cr(0), dg(3), dg(0);
    cr(-1), cr(0), cr(-2), dg(0), dg(3)
  ).
$
The off-diagonal entries of $L_(W F)$ are non-positive, and every row sums
to zero because each diagonal count cancels the off-diagonal observation
counts in the same row. For a worker, the diagonal entry is the number of
observations for that worker, while the off-diagonal entries show how those
observations are distributed across firms. Firm rows have the analogous
interpretation with observations summed over workers.


The graph Laplacian can be constructed similarly for any pair of
fixed-effect dimensions which we will use to construct a preconditioner in
@sec:schwarz-preconditioner. First, however, we discuss the method of
alternating projections, which avoids forming the full Gramian $G$ and only
uses the diagonal worker, firm, and year blocks.


= Alternating Projections and Graph Connectivity <sec:map-connectivity>

The workhorse algorithm for high-dimensional fixed-effect regressions is
the Method of Alternating Projections (MAP), also referred to as iterative
demeaning or the "zig-zag" algorithm @guimaraes2010 @gaure2013. Many
packages employ MAP or its variants, frequently combined with accelerations
@berge2018 @correia2017, such as the Irons-Tuck extrapolation used by
`fixest` @irons1969 @berge2026fixest. MAP solves the FWL residualization
problem by iterating over one fixed-effect dimension at a time. For
example, in the AKM model of @sec:akm, MAP first subtracts worker means
from the current residual, then firm means from the updated residual, then
year means, and repeats until convergence is reached.

To relate MAP's convergence to the connectivity of the fixed-effect graph,
we discuss the FWL normal equations presented in @eq:fwl-normal. We start
with the case of the AKM model where the fixed-effect design matrix becomes
$D = [D_W quad D_F quad D_Y]$ for worker, firm, and year fixed effects.
@eq:fwl-normal can then be written as
$
  mat(
    dg(G_(W W)), cr(C_(W F)), cr(C_(W Y));
    cr(C_(W F)'), dg(G_(F F)), cr(C_(F Y));
    cr(C_(W Y)'), cr(C_(F Y)'), dg(G_(Y Y))
  ) mat(alpha_W; alpha_F; alpha_Y)
  = mat(D_W' W mu; D_F' W mu; D_Y' W mu),
$

or, equivalently,

$
    dg(G_(W W)) alpha_W + cr(C_(W F)) alpha_F + cr(C_(W Y)) alpha_Y & = D_W' W mu, \
   cr(C_(W F)') alpha_W + dg(G_(F F)) alpha_F + cr(C_(F Y)) alpha_Y & = D_F' W mu, \
  cr(C_(W Y)') alpha_W + cr(C_(F Y)') alpha_F + dg(G_(Y Y)) alpha_Y & = D_Y' W mu.
$<eq:fwl-normal-akm>

Each equation can be rearranged to express one block of effects conditional
on the others. For example, using $C_(W F) = D_W' W D_F$ and
$C_(W Y) = D_W' W D_Y$, we can substitute $D_W' W$ in @eq:fwl-normal-akm
and obtain

$ alpha_W = dg(G_(W W)^(-1))D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Because $G_(W W)$ is a diagonal matrix whose entries are workers' total
observation weights, solving for $alpha_W$ divides each worker's weighted
partial residual by its total observation weight.

For a general fixed-effect model with $q=1,dots, Q$ fixed-effect
dimensions, MAP updates the coefficient vector $alpha_q^((k+1))$ of
fixed-effect dimension $q$ at iteration $k+1$ using the Gauss-Seidel
algorithm @guimaraes2010 according to
$
  alpha_q^((k+1)) & =
  dg(G_(q q)^(-1)) D_q' W
  (mu - sum_(s=1)^(q-1) D_s alpha_s^((k+1)) - sum_(s=q+1)^Q D_s alpha_s^((k))).
$<eq:map-iteration>
Note that the cross-tabulation blocks $C_(q s) = D_q' W D_s$ are never used
directly in @eq:map-iteration and enter in MAP only indirectly through the
propagation of the error across the different fixed-effect dimensions. To
see this, let $hat(alpha) = (hat(alpha)_1, ..., hat(alpha)_Q)$ be a
solution to @eq:fwl-normal and define the coefficient error at iteration
$k$ as $e_q^((k)) = alpha_q^((k)) - hat(alpha)_q$. Subtracting the solution
$hat(alpha)$ from @eq:map-iteration yields
$
  e_q^((k+1))
  = -dg(G_(q q)^(-1)) (
    sum_(s=1)^(q-1) cr(C_(q s)) e_s^((k+1))
    + sum_(s=q+1)^Q cr(C_(q s)) e_s^((k))
  ).
$<eq:map-error-propagation>
Define the degree-scaled errors
$tilde(e)_q^((k)) = G_(q q)^(1/2) e_q^((k))$ and the degree-normalized
cross-tabulations
$
  cr(H_(q s)) = dg(G_(q q)^(-1/2)) cr(C_(q s)) dg(G_(s s)^(-1/2)).
$<eq:degree-normalized-cross-tabulation>
Then, @eq:map-error-propagation can be written as
$
  tilde(e)_q^((k+1))
  = -sum_(s=1)^(q-1) cr(H_(q s)) tilde(e)_s^((k+1))
  -sum_(s=q+1)^Q cr(H_(q s)) tilde(e)_s^((k)).
$
The degree-normalized cross-tabulations $H_(q s)$
(@eq:degree-normalized-cross-tabulation[]) appear as the off-diagonal
blocks of the normalized graph Laplacian
$
  cal(L)_(q s) & = mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(s s)^(-1/2))
                 ) L_(q s) mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(s s)^(-1/2))
                 ) \
               & = mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(s s)^(-1/2))
                 ) mat(
                   dg(G_(q q)), cr(-C_(q s));
                   cr(-C_(q s)'), dg(G_(s s));
                 )
                 mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(s s)^(-1/2))
                 ) \
               & = mat(
                   dg(I), cr(-H_(q s));
                   cr(-H_(q s)'), dg(I)
                 ),
$<eq:cross-tabulation-and-laplacian>
where $L_(q s)$ is the graph Laplacian of the bipartite graph formed by the
fixed-effect dimensions $q$ and $s$.

The connectivity of the bipartite graph formed by a pair of fixed-effect
dimensions therefore governs the speed of convergence of MAP. To see this,
note that the second-smallest eigenvalue of the normalized graph Laplacian
$lambda_2(cal(L)_(q s))$ is our measure of connectivity#footnote[it is zero if
  the graph is disconnected and is small when the graph consists of nearly
  disconnected subgraphs joined only by narrow bridges
  @chung1997] and that, within each connected component, the block
structure in @eq:cross-tabulation-and-laplacian implies that
$lambda_2 (cal(L)_(q s))$ is related to the second-largest singular value
of the degree-normalized cross-tabulation $sigma_2 (H_(q s))$ via
$
  lambda_2 (cal(L)_(q s)) = 1 - sigma_2 (H_(q s)).
$
The largest nontrivial singular value of the degree-normalized cross
tabulation $sigma_2(H_(q s))$ provides a measure of worst-case convergence
of MAP. Indeed, for a model containing only two fixed-effect dimensions $q$
and $s$, the MAP error update (@eq:map-error-propagation[]) becomes
$
  tilde(e)_q^((k+1)) & = -H_(q s) tilde(e)_s^((k)), \
  tilde(e)_s^((k+1)) & = H_(q s)' H_(q s) tilde(e)_s^((k)).
$
Consequently, the error component of $tilde(e)_s^((k+1))$ in the direction
of the singular vector associated with $sigma_2(H_(q s))$ shrinks with the
worst-case MAP contraction factor
$
  rho_(q s)^("MAP") = sigma_2(H_(q s))^2
  = (1 - lambda_2(cal(L)_(q s)))^2.
$
In other words, a poorly connected
graph implies that $lambda_2(cal(L)_(q s))$ is small and $sigma_2(H_(q s))$
is close to one, so $rho_(q s)^("MAP")$ is close to one and the error component
potentially shrinks slowly in each MAP iteration. With more
than two fixed-effect dimensions, the full error propagation also depends
on the remaining dimensions and their update order, so our pairwise measure
of connectivity does not fully describe MAP's convergence rate.

For the worker-firm example discussed in @sec:gramian, the diagonal blocks are
$G_(W W) = "diag"(2,2,2)$,
$G_(F F) = "diag"(3,3)$, and the cross-tabulation is
$
  cr(C_(W F)) = mat(cr(1), cr(1); cr(2), cr(0); cr(0), cr(2)),
$
so that
$
  cr(H_(W F)) = dg(G_(W W)^(-1/2)) cr(C_(W F)) dg(G_(F F)^(-1/2))
  = 1 / sqrt(6) mat(cr(1), cr(1); cr(2), cr(0); cr(0), cr(2)).
$
The graph is connected, and $H_(W F)' H_(W F) = 1 / 6 mat(5, 1; 1, 5)$ has
eigenvalues $1$ and $2/3$, hence $sigma_2(H_(W F)) = sqrt(2/3)$ and
our measure of connectivity is $lambda_2(cal(L)_(W F)) = 1 - sqrt(2/3)$,
while the worst-case MAP contraction factor is $rho_(W F)^("MAP") = 2/3$.

= The Factor-Pair Schwarz Preconditioner <sec:schwarz-preconditioner>

In the previous @sec:map-connectivity, we relate MAP's convergence to the
connectivity of the pairwise fixed-effect graphs. Our goal is to
incorporate information on the graph's connectivity to improve convergence
in poorly connected fixed-effect graphs. To this end, we use the pairwise
fixed-effect graphs to construct a preconditioner for LSMR @fong2011, an
iterative least-squares algorithm that improves an initial guess through
repeated residual corrections.

== Preconditioners

LSMR's convergence depends on the conditioning of the fixed-effect Gramian
$G$. To see this, recall from @eq:fwl-normal that the fixed-effect
coefficients satisfy $G alpha = D' W mu$. For a candidate solution
$alpha_k$ at iteration $k$, the residual is
$
  s_k = D'W mu - G alpha_k = G (hat(alpha) - alpha_k).
$
Consequently, components of the residual $s_k$ along eigenvector directions
of the Gramian $G$ with small eigenvalue contribute little to the residual
even when their coefficient error is large. Note that because, up to a sign
change, the pairwise Gramian is the pairwise graph Laplacian discussed in
@sec:map-connectivity, these eigenvector directions correspond to the
poorly connected bipartite fixed-effect graphs. This means that LSMR
removes some error components after a small number of iterations, while
components corresponding to weak links, sparse mobility, or near nesting in
the fixed-effect graph shrink more slowly.

A preconditioner $M^(-1)$ is designed to counteract this imbalance by
approximating the inverse of the Gramian $G^(-1)$, so that $M^(-1) s_k$
yields an approximation of the error $hat(alpha) - alpha_k$. If
$M^(-1) = G^(-1)$, the preconditioned operator is
$M^(-1) G = G^(-1) G = I,$ in which case the solver would recover the
solution after one correction. However, to form $G^(-1)$ one would have to
solve the fixed-effect normal equations themselves which is computationally
expensive. For a preconditioner to be useful, it must therefore approximate
$G^(-1)$ cheaply so that repeated preconditioning amortizes its setup
cost.#footnote[LSMR never forms $M^(-1) G$ explicitly, nor $G$ itself. The
  iteration requires only products with $D$ and $D'$ and applications of
  $M^(-1)$, supplied as linear operators.]

=== The Diagonal Preconditioner<sec:diagonal-preconditioner>

The simplest approximation to $G^(-1)$ discards the off-diagonal
cross-tabulations of the fixed-effect Gramian to construct a diagonal
preconditioner @xu1992.#footnote[The diagonal preconditioner is used by
  `FixedEffectModels.jl`
  @fixedeffectmodels] In the AKM model, the diagonal preconditioner of the
Gramian (@eq:gramian-blocks[]) takes the form
$ M_("diag")^(-1) = "diag"(dg(G_(W W)^(-1)), dg(G_(F F)^(-1)),
  dg(G_(Y Y)^(-1))). $

It encodes the number of observations of each fixed-effect level, but
discards the connectivity of the bipartite fixed-effect graph and therefore
does not address potentially slow convergence due to poorly connected
bipartite fixed-effect graphs.

#figure(
  image(solver-img("diagonal_lsmr_strategy.svg"), width: 65%),
  caption: [Diagonal preconditioning within LSMR. Worker, firm, and year counts are
  computed once and reused for the outcome and covariates; at each iteration, the
  preconditioner divides factor-level entries by their total observation weights. Because
  LSMR continues to check the original fixed-effect least-squares problem, diagonal
  scaling affects convergence speed rather than the residualized variables reached at
  convergence.]
) <fig-diagonal-strategy>

=== The Factor-Pair Schwarz Preconditioner

To incorporate information on the pairwise graph connectivity, we use
additive Schwarz preconditioning @toselli2005 and approximate $G^(-1)$ as
the sum of inverses of the pairwise fixed-effect Gramians
$
  G_(q r)
  = R_(q r) G R_(q r)'
  = mat(
    dg(G_(q q)), cr(C_(q r));
    cr(C_(q r)'), dg(G_(r r))
  ),
$
where $R_(q r)$ restricts the coefficient space to the coordinates of the
fixed-effect dimensions $(q, r)$. Because the pairwise coefficient spaces
overlap, we combine their inverse contributions using partition-of-unity
weights. For each fixed-effect level $j$ in dimension $q$ or $r$, let $c_j$
denote the number of pairwise fixed-effect Gramians containing $j$ and let
$Omega_(q r)$ be the diagonal matrix with entries $c_j^(-1/2)$. The
additive Schwarz preconditioner is then the sum of local factor-pair
inverses

$
  P^(-1) = sum_(q<r) P_(q r)^(-1)
  = sum_(q<r) R_(q r)' Omega_(q r) G_(q r)^(-1) Omega_(q r) R_(q r).
$<eq:additive-schwarz>

Applied to the AKM model and its Gramian (@eq:gramian-blocks[]), the
additive Schwarz preconditioner is given by
$
  P^(-1) = P_(W F)^(-1) + P_(W Y)^(-1) + P_(F Y)^(-1).
$
The Schwarz preconditioner depends on the three pairwise fixed-effect
Gramians $G_(W F)$, $G_(W Y)$ and $G_(F Y)$, and consequently, incorporates
the information on the connectivity of the bipartite fixed-effect graphs
encoded in the cross-tabulations $C_(W F)$, $C_(W Y)$ and $C_(F Y)$. For
the worker-firm pair, for example, we have

$
  P_(W F)^(-1)
  = 1 / 2 R_(W F)'
  mat(dg(G_(W W)), cr(C_(W F)); cr(C_(W F)'), dg(G_(F F)))^(-1)
  R_(W F).
$

@fig:pair-block illustrates that the additive Schwarz preconditioner
incorporates the pairwise cross-tabulations $C_(W F)$ in contrast to MAP
(@sec:map-connectivity) and the diagonal preconditioner
(@sec:diagonal-preconditioner) which only solve the diagonal block
$G_(W W)$. The Schwarz preconditioner does not solve the full simultaneous
worker-firm-year problem but only the local factor-pair subproblems.
However, even the factor-pair subproblems may contain hundreds of thousands
of levels per fixed-effect dimension, making exact inversion
computationally expensive. In the next section, we discuss efficient
approximate solutions to the local factor-pair subproblems.

#figure(
  image(solver-img("factor_level_vs_pair_block.svg"), width: 88%),
  caption: [Count-only and factor-pair corrections for the example from Section 4. The
    diagonal preconditioner (left) treats the blue worker and firm count blocks separately,
    whereas the factor-pair correction (right) retains those blocks and adds the orange
    worker-firm observation counts $C_(W F)$.],
) <fig:pair-block>



== Approximate Factor-Pair Solves via Graph Laplacians

Applying the exact factor-pair inverses $P_(q r)^(-1)$ in
@eq:additive-schwarz can become computationally expensive when the number
of fixed-effect levels is large. For preconditioning, however, this local
solve need not be exact. Recognizing that the pairwise fixed-effect
Gramians are, up to a sign change, the graph Laplacian of the bipartite
fixed-effect graph, we can efficiently construct approximate solutions to
the local subproblems.

For two fixed-effect dimensions $(q, r)$, the graph Laplacian $L_(q r)$ is
given by
$
  L_(q r)
  = mat(
    dg(G_(q q)), cr(-C_(q r));
    cr(-C_(q r)'), dg(G_(r r))
  )
  = T_(q r) G_(q r) T_(q r),
$
where $T_(q r) = "diag"(I_q, -I_r)$ for identity matrices $I_q$ and $I_r$.
Because the graph Laplacian $L_(q r)$ admits sparse approximate Cholesky
factorizations @gao2025, we can efficiently construct approximate solutions
to
$
  G_(q r)^(-1) = T_(q r) L_(q r)^(-1) T_(q r).
$
The approximate Cholesky factorization has expected construction cost
$cal(O)(m log m)$, where $m$ denotes the number of pairwise links @gao2025.
Let $A_(q r)$ denote the resulting operator approximating $G_(q r)^(-1)$.
Then, the additive Schwarz preconditioner becomes
$
  M^(-1) = sum_(q < r) R_(q r)' Omega_(q r) A_(q r) Omega_(q r) R_(q r).
$

Our preconditioner therefore makes two approximations: First, the inverse
of the fixed-effect Gramian $G^(-1)$ is approximated by a sum of
factor-pair inverses $P^(-1)$. Second, the inverse of the pairwise
fixed-effect Gramians $G^(-1)_(q r)$ is replaced with the approximate
solution $A_(q r)$.


== Implementation

@fig-pair-strategy shows the construction. For each pair of absorbed
factors, a sign change converts the local Gramian to a graph Laplacian. The
implementation handles large connected components with approximate Schur
reduction and sparse approximate Cholesky, then maps the corrections back
to the full coefficient vector and combines them using partition-of-unity
weights.

The outer LSMR iteration uses this preconditioner to shape its search
directions @fong2011 @arridge2014 @yang2024flexible. Once built, the same
preconditioner can residualize the outcome and every covariate.
@sec:appendix-schwarz lists the setup and application steps.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Construction of the factor-pair preconditioner. For large pair
    blocks, the algorithm approximates the Schur complement and factors the
    reduced Laplacian with approximate Cholesky. The weighted pair
    corrections form the preconditioner used by LSMR.],
) <fig-pair-strategy>

= Benchmarks

The benchmarks ask when fewer iterations repay the cost of building the factor-pair
preconditioner. We vary mobility and sorting in controlled AKM panels, compare two public
synthetic designs, and use eight real datasets from #cite(<correia2017>, form: "prose").

Elapsed times cover the package-level regression call, from model setup through
coefficient estimation. Cross-package comparisons use the same input data but retain each
package's default treatment of singleton levels and separated observations, whereas
experiments that isolate the preconditioner use one prepared sample. The appendix reports
the additional memory used to store the pair structures.

Each table reports our measure of connectivity for each factor pair after iterative singleton removal. For a
disconnected graph, the value comes from the component with the most retained
observations, whose observation share appears in parentheses. As in Section 5, the measure
remains a pairwise diagnostic in models with three or more fixed effects.

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

Tables report medians over the stated number of repetitions. All runs use an Apple M4 Mac
mini with 10 CPU cores and 16 GB of memory running macOS 15.3.1. We omit `reghdfe`
@reghdfe @correia2017 because we lack a Stata license; its accelerated MAP algorithm is
represented here by `fixest`.

== Runtime Benchmarks

Package defaults are useful to users but do not impose equal accuracy because the
implementations apply tolerances to different quantities. We report both package-default
times and a comparison based on achieved coefficient and residual error.

The main OLS panels contain one million observations, one covariate, ten periods, and
worker, firm, and year effects. One set of designs lowers the probability of moving,
while the other holds that probability at one but increasingly assigns moves within
groups of similar firms.

=== Worker Mobility

As mobility declines, our measure of connectivity for the worker-firm graph falls from
$0.232$ to $2.41 times 10^(-5)$;
all configurations finish within 3.60 seconds in the first two designs. PyFixest MAP then
slows sharply or reaches its iteration cap, and unpreconditioned LSMR also reaches the
cap. Factor-pair LSMR falls from 0.554 seconds in the first design to 0.373 seconds in the
last because the local worker-firm graphs become cheaper to factorize.

=== Sorting Among Movers

Although every worker moves between periods, increasing $rho$ concentrates moves within
groups, and our measure of connectivity falls from $0.222$ to $2.13 times 10^(-4)$.
PyFixest MAP rises from
0.411 seconds to 53.9 seconds before reaching its cap in the final design;
unpreconditioned LSMR reaches its cap in the last four. Over the same designs, diagonal
LSMR rises from 0.362 to 1.58 seconds, while factor-pair LSMR stays below one second and
is faster than diagonal LSMR at the three smallest values of our measure of connectivity.

=== When Preconditioner Setup Dominates

The public `fixest` benchmark suite complements the gradual connectivity changes in the
AKM designs by contrasting a dense, well-connected design with a sparse, nearly nested
design @berge2026fixest. Both contain 10 million observations, one covariate, and worker,
firm, and year fixed effects.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Simple and difficult `fixest` designs (10 million observations, three fixed effects).]
#include "generated/tables/ols.typ"
  #v(0.25em)
  #table-note[Times are in seconds and cover each OLS regression from
  model setup through coefficient estimation. Each package uses its default settings. A
  preliminary run sets the number of measured runs: 20 if it takes less than 1 second, 7
  if it takes 1 to 10 seconds, and 3 if it takes longer. When
  only $k$ of $n$ planned runs return an estimate, $t (k/n)$ reports their median time
  $t$. A dash means that no timing is available.
  `capped (0/n)` means that no run finishes before the iteration limit; `failed (0/n)`
  marks a failure other than reaching that limit. Both designs have 10 million
  observations, one covariate, and worker, firm, and year fixed effects. Our measure of
  connectivity, $lambda_2$, is computed after iterative singleton removal on the component containing
  the most retained observations; its observation share appears in parentheses.]
]]

The dense design exposes the setup cost: five methods finish in 2.08 to 2.72 seconds,
while factor-pair LSMR takes 11.2 seconds. Near nesting reverses the ranking. Factor-pair
LSMR takes 4.63 seconds, compared with 10.2 seconds without preconditioning, 27.0 seconds
for FEM.jl, 62.2 seconds for `fixest`, and 326.2 seconds for PyFixest MAP; diagonal LSMR
reaches its default cap.

=== Iterations After Setup

To separate iteration counts from setup time, we use 100,000-observation versions of the
same designs.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Iterations on the simple and difficult designs (100,000 observations).]
#include "generated/tables/iterations.typ"
  #v(0.25em)
  #table-note[The MAP column counts complete passes over all
  fixed-effect dimensions. The other columns count LSMR iterations: `off` uses no
  preconditioner, `diagonal` scales by fixed-effect group counts, and `additive` uses the
  factor-pair preconditioner. A MAP pass and an LSMR iteration perform different work, so
  their counts are not directly comparable. Each count is the median of three runs; each
  run removes the fixed effects from one outcome and one covariate.]
]]

Factor-pair LSMR needs 14 iterations on the simple design and 22 on the difficult one.
The corresponding counts rise from 16 to 180 for diagonal LSMR and from 37 to 249 without
preconditioning. The factor-pair iteration count is stable, but its setup still makes it
unattractive for the simple one-shot regression.

=== Public Real-Data Benchmarks

The controlled designs isolate mobility, sorting, and near nesting, but empirical
fixed-effect graphs also contain large differences in group size, thin links between
dense groups, and many disconnected components. The public benchmark collection
assembled by #cite(<correia2017>, form: "prose") contains eight such datasets and lets us
check whether the controlled comparisons carry over to these less regular graphs.

#block(breakable: false)[#text(size: 8.7pt)[
#strong[Correia real-data benchmarks.]
#include "generated/tables/correia_real.typ"
  #v(0.25em)
  #table-note[Times are in seconds for OLS regressions using each package's default
  settings, with three planned runs per cell. Each package applies its own rule for
  removing fixed-effect levels observed only once. If only $k$ runs return an estimate,
  $t (k/3)$ reports their median time $t$. `capped (0/3)` means that no run finishes
  before the 10,000-iteration limit. Our measure of connectivity, $lambda_2$, is computed after iterative singleton
  removal on the `id1`-`id2` component containing the most retained observations; its
  observation share appears in parentheses.]
]]

On `credit` and `soccer`, PyFixest MAP takes 0.193 and 0.026 seconds, compared with 0.242
and 0.065 seconds for factor-pair LSMR, so the preconditioner's setup cost does not pay
on these easy datasets. Factor-pair LSMR is faster on the more difficult `enron` and
`schools` graphs, taking 0.402 and 0.266 seconds compared with 2.95 and 5.98 seconds for
PyFixest MAP. PyFixest MAP reaches its iteration cap on `github`, `patents`, `workers`,
and `directors`, while factor-pair LSMR finishes in 0.279 to 0.388 seconds. Accelerated
MAP can still be competitive, however; `fixest` takes 0.283 seconds on `directors`,
slightly less than factor-pair LSMR's 0.327 seconds.

Our measure of connectivity is a pairwise description rather than a rule for choosing a solver because
runtime also depends on component size, factor dimensions, graph sparsity, acceleration,
and the preconditioner's setup cost.

== Worker- and Firm-Specific Year Slopes

Applied work often allows a linear time trend to differ across observational units. We
add one numeric year slope for every worker and firm to the simple and difficult designs
and to all twelve AKM mobility and sorting designs, while retaining categorical year
effects. In `fixest` formula syntax, the absorbed part is `indiv_id[year] +
firm_id[year] + year`. Each bracketed term includes a group intercept and a
group-specific linear slope on the existing year variable, while the final term absorbs
unrestricted aggregate year shocks.

Bracket syntax treats `year` as numeric; it does not create a categorical worker-year or
firm-year effect. Every worker in these balanced panels contributes ten observations
with year values 1 through 10, so the worker slopes do not create singleton groups. A
firm slope requires observations from at least two years. Firms without that support
are handled by each package's usual identification and singleton rules. Worker-firm
interactions are not included because they add categorical match effects rather than a
slope on a numeric variable.

The PyFixest cells use the `within` backend because its other demeaning backends do not
support varying slopes. We compare diagonal and factor-pair Schwarz preconditioning
with R `fixest` and FEM.jl.

#block(breakable: false)[#text(size: 8.7pt)[
#strong[Worker- and firm-specific year slopes on the simple and difficult designs.]
#include "generated/tables/varying_slopes_base.typ"
  #v(0.25em)
  #table-note[Times are in seconds for OLS regressions with 10 million observations and
  one covariate. Each cell has three planned runs after one unreported warm-up.]
]]

#block(breakable: false)[#text(size: 8.7pt)[
#strong[Worker- and firm-specific year slopes across AKM mobility designs.]
#include "generated/tables/varying_slopes_mobility.typ"
]]

#block(breakable: false)[#text(size: 8.7pt)[
#strong[Worker- and firm-specific year slopes across AKM sorting designs.]
#include "generated/tables/varying_slopes_sorting.typ"
  #v(0.25em)
  #table-note[The AKM regressions have 1 million observations. Every regression in the
  three tables absorbs worker and firm intercepts, worker- and firm-specific numeric
  year slopes, and categorical year effects. PyFixest uses LSMR with diagonal or
  factor-pair Schwarz preconditioning through the `within` backend; R `fixest` and
  FEM.jl use their package defaults. If only $k$ of the three runs return an estimate,
  $t (k/3)$ reports the median time $t$. `capped (0/3)` means that no run finishes before
  its iteration limit, while `failed (0/3)` marks another failure.]
]]

== Runtime and Achieved Precision

Solver comparisons also depend on how packages define convergence. MAP implementations
compare changes in residuals or fixed-effect coefficients, while LSMR uses its estimated
least-squares residual and first-order-condition error @fong2011, so their numerical
tolerances are not a common measure of accuracy. For example, `within` stops when either
its relative residual or its scaled normal-equation error meets the requested threshold;
we compute the latter as

$ frac(||A^T r||_2, ||A||_F ||r||_2), quad
  A = W^(1/2) D, quad r = W^(1/2) (mu - D alpha). $

@fig-tolerance places the methods on a common accuracy scale for mobility designs 1, 3,
and 5. Every method receives the same singleton-pruned sample and a common
10,000-iteration cap. We time three fits at each package-specific tolerance and compare
their coefficients and residuals with factor-pair LSMR at tolerance $10^(-14)$.

#figure(
  image(result-img("tolerance_frontier.svg"), width: 97%),
  caption: [Elapsed time and achieved accuracy on three AKM mobility designs. Each
  marker reports the median elapsed time among the three planned fits that return an
  estimate on the same sample. Before timing, we remove fixed-effect levels observed only
  once and use the remaining observations for every method. Lines connect results for
  one method across requested tolerances. The top row measures coefficient error as
  $abs(hat(beta)-hat(beta)^star) / "SE"(hat(beta)^star)$, and the bottom row measures
  residual error as $frac(||r-r^star||_2, ||r^star||_2)$; the reference coefficient
  $hat(beta)^star$ and residual $r^star$ come from factor-pair LSMR at tolerance
  $10^(-14)$. Error decreases from left to right, and a circle on each line marks the
  result obtained with the package's default tolerance. Every fit is limited to 10,000
  iterations; settings for which none of the three fits returns an estimate are omitted
  and identified in the annotations.]
) <fig-tolerance>

== Amortizing the Preconditioner <sec-amortization>

Matching achieved accuracy does not remove the factor-pair setup cost, but later
regressions can reuse the pair blocks and factorizations when the observations, weights,
and fixed-effect identifiers are unchanged.

=== Setup Cost Across Connectivity

We separate construction from LSMR iterations on the one-million-observation mobility
designs, for specifications with worker-firm and worker-firm-year effects.

#block(breakable: false)[#text(size: 8.8pt)[
#strong[Factor-pair setup and LSMR times as worker mobility varies.]
#include "generated/tables/akm_setup_cost.typ"
  #v(0.25em)
  #table-note[Times are in seconds. For each design and
  specification, we plan five runs with 1 million observations and an LSMR tolerance of
  $10^(-12)$. The table gives the median setup time and time spent in LSMR among the
  runs that finish; if only $k$ finish, $t (k/5)$ reports their median time $t$. The
  two-fixed-effect specification absorbs worker and firm effects; the three-fixed-effect
  specification also absorbs year effects. Our measure of connectivity, $lambda_2$, is computed after iterative
  singleton removal on the component containing the most retained observations; its
  observation share appears in parentheses.]
]]

From the highest- to the lowest-mobility design, setup falls from 0.109 to 0.028 seconds
with two fixed effects and from 0.113 to 0.032 seconds with three; time spent in LSMR also
falls. The low-mobility graph has fewer cross-firm links to store and factorize, which
explains the declining factor-pair runtime in @fig-gap-runtime.

=== Ten Regressions on the Same Fixed Effects

Empirical projects often estimate several specifications on one sample and set of fixed
effects. We time ten sequential regressions on both one-million-observation `fixest`
designs, comparing diagonal preconditioning with factor-pair preconditioning that is
rebuilt for each regression or retained for all ten.

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Ten regressions with rebuilt and cached preconditioners.]
#include "generated/tables/regression_reuse.typ"
  #v(0.25em)
  #table-note[For each policy, we add the setup and LSMR times across ten sequential
  regressions and take the median across three planned repetitions. If only $k$
  repetitions finish, $t (k/3)$ reports their median time $t$, and the speedup is
  omitted. Times are in seconds. Each design has 1 million observations and worker,
  firm, and year fixed effects. Each regression removes the fixed effects from the
  common outcome and one of ten covariates at an LSMR tolerance of $10^(-12)$. For
  complete cells, speedup divides the diagonal total time by the reported total time.
  `Additive, rebuilt` constructs a new factor-pair preconditioner for each regression;
  `Additive, cached` reuses one preconditioner for all ten.]
]]

Reuse lowers factor-pair time on the simple design from 4.29 to 1.79 seconds, but diagonal
preconditioning still takes only 0.601 seconds. On the difficult design, the ranking is
decisive: ten regressions take 43.6 seconds with diagonal preconditioning, 1.58 seconds
when the factor-pair preconditioner is rebuilt, and 1.30 seconds when it is retained.

== Poisson and Other GLMs

Reuse also matters within a single nonlinear fit because Poisson fixed-effect models
repeat the absorption step at every iteratively reweighted least-squares (IRLS) iteration
@stammann2018. Their fixed-effect identifiers remain fixed while the weights and working
response change, so `ppmlhdfe` @correia2020ppmlhdfe and related estimators repeatedly
residualize the data with the same identifiers and accumulate this cost over the fit.

PyFixest currently retains the preconditioner from the first weighted demeaning call as
the IRLS weights change. The benchmark reports that policy; it does not isolate reuse as
the cause of the runtime difference.

The comparison uses the one-million-observation simple and difficult `fixest` designs
@berge2026fixest. It includes R `fixest::fepois`, `GLFixedEffectModels.jl`, and PyFixest
`fepois` with MAP or factor-pair LSMR. Each implementation receives the same
100-iteration IRLS limit and applies its default separation rule.

#v(0.35em)

#text(size: 8.8pt)[
#strong[Poisson benchmarks (1 million observations, one covariate).]
#include "generated/tables/ppml.typ"
  #v(0.25em)
  #table-note[Times are in seconds for Poisson fixed-effect
  regressions with 1 million observations, one
  covariate, and worker, firm, and year fixed effects. `fixest` is R `fixest::fepois`;
  `rust-map` and `within` use PyFixest `fepois`; and `GLFEM.jl` is
  `GLFixedEffectModels.jl`. The `within` configuration reuses the first factor-pair
  preconditioner as the weights change across iteratively reweighted least squares
  steps. Each package applies its default rule for removing separated observations. If
  only $k$ of the three planned fits return an estimate, $t (k/3)$ gives their median
  time $t$.
  `capped (0/3)` means that no fit finishes before either the package's inner demeaning
  limit or the common limit of 100 iteratively reweighted least squares steps;
  `failed (0/3)` marks another failure.]
  ]

#block(breakable: false)[All four implementations finish the simple design in less than
ten seconds. On the difficult design, PyFixest MAP reaches its iteration cap,
`GLFEM.jl` takes 129.8 seconds, `fixest` takes 439.3 seconds, and factor-pair LSMR takes
5.52 seconds. Repeated absorption magnifies the cost of slow propagation across the
worker-firm graph.]

= Software

The benchmarks use the open-source `within` solver @within, whose computational core is
written in Rust and exposed through Python and R APIs.

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

PyFixest users can select the method without constructing the Gramian or its pair blocks
@pyfixest:

#text(size: 8.8pt)[```python
import pyfixest as pf

fit = pf.feols(
    "y ~ x1 + x2 | worker + firm",
    data=df,
    demeaner=pf.LsmrDemeaner(),
)
```]

The lower-level Python interface exposes the FWL operation, with `worker_code` and
`firm_code` supplied as zero-based integer codes:

#text(size: 8.8pt)[```python
import numpy as np
from within import solve_batch

categories = np.asfortranarray(
    df[["worker_code", "firm_code"]].to_numpy(dtype=np.uint32)
)
variables = np.asfortranarray(df[["y", "x1", "x2"]].to_numpy())
res = solve_batch(categories, variables)
y_tilde, X_tilde = res.demeaned[:, 0], res.demeaned[:, 1:]
beta_hat = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]
```]

`solve_batch` residualizes all three columns with one factor-pair preconditioner, after
which the last line runs the low-dimensional FWL regression from Section 2.

#pagebreak()

= Conclusion

MAP's low cost per pass makes it effective for many well-connected fixed-effect models,
but successive mean updates move information slowly when mobility is sparse, sorting is
strong, or one effect is nearly nested in another.

Factor-pair preconditioning uses observed co-occurrences to correct worker-firm and other
pair blocks jointly, changing the numerical update while preserving the original
least-squares target. Its setup cost can outweigh the reduction in iterations on easy
graphs, but on difficult graphs it can avoid hundreds of MAP passes or LSMR iterations;
sparse pair graphs can also be cheaper to factorize than dense ones.

The setup cost is most likely to be worthwhile when our measure of connectivity for a
factor pair is small or the
same fixed effects are absorbed repeatedly. Specification searches can reuse one
preconditioner across outcomes and covariates, while IRLS estimators call the demeaning
routine at every outer iteration. In both cases, reuse spreads the setup cost across
several residualizations. The measure remains a pairwise connectivity diagnostic rather than
a solver-selection rule, but a small value signals the weak connectivity that can slow
MAP.

#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 1.45em,
  below: 0.68em,
)[#text(size: 15pt, weight: "bold", it.body)]
#pagebreak()

= Appendix: Memory Use

MAP stores the current residuals and group sums, whereas factor-pair LSMR also retains
match structures and local factorizations. We compare peak resident memory for the two
Rust implementations inside PyFixest, using isolated processes so the surrounding
regression code is held fixed.

#v(0.4em)

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Peak physical memory use (three fixed effects, one covariate).]
#include "generated/tables/memory.typ"
  #v(0.25em)
  #table-note[For each regression, we record the largest amount of
  physical memory used by an isolated Python process, in MiB ($2^20$ bytes). The table
  compares PyFixest OLS regressions using MAP or factor-pair LSMR. Each regression has
  one covariate and worker, firm, and year fixed effects. The results cover the simple
  and difficult designs at 100,000 and 1 million observations. Our measure of
  connectivity, $lambda_2$, is computed
  after iterative singleton removal on the component containing the most retained
  observations; its observation share appears in parentheses.]
  ]]

#v(0.35em)

Factor-pair LSMR uses slightly more memory: #result_memory_100k_overhead at 100K
observations and #result_memory_1m_overhead at 1M; the additional storage holds the
pairwise co-occurrences, partition weights, and approximate Cholesky factors.

#pagebreak()
#[
  #counter(heading).update(0)
  #set heading(numbering: "A.1", supplement: [Appendix])
  #show heading.where(level: 1): it => block(
    width: 100%, above: 1.45em, below: 0.68em,
  )[
    #text(size: 15pt, weight: "bold")[
      Appendix #counter(heading).display("A"): #it.body
    ]
  ]

  = Details of the Factor-Pair Schwarz Preconditioner <sec:appendix-schwarz>

  == Two-Factor Block Inverse

  For the two-factor block

  $ G_2 = mat(dg(G_(W W)), cr(C_(W F)); cr(C_(W F)'), dg(G_(F F))), $

  let

  $ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

  Standard block inversion gives

  $
    G_2^(-1) = mat(
      dg(G_(W W)^(-1) + G_(W W)^(-1) C_(W F) S^(-1) C_(W F)' G_(W W)^(-1)), cr(-G_(W W)^(-1) C_(W F) S^(-1));
      cr(-S^(-1) C_(W F)' G_(W W)^(-1)), dg(S^(-1))
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
          - If fixed-effect level $j$ appears in $c_j$ subdomains, store
            the partition weight $omega_j = 1 / sqrt(c_j)$.
          - For each subdomain, form $L_s = T_s G_s T_s$. If $p_s$ is the
            number of local levels, set
            $Pi_s = I_(p_s) - bold(1) bold(1)' / p_s$; multiplying by
            $Pi_s$ subtracts the component mean.
          - Eliminate the larger factor block, leaving a reduced system on
            the smaller block. Solve a small reduced system by dense
            Cholesky. For a large system, approximate the fill-in cliques
            by sampling and apply randomized approximate Cholesky.

          #strong[Krylov application]
          - Initialize $z = 0$.
          - For each subdomain $s$, form $h_s = Omega_s R_s r$ and
            $b_s = Pi_s T_s h_s$.
          - Apply the stored local solver to $b_s$ to obtain $v_s$, then
            set $u_s = T_s v_s$.
          - Accumulate $z <- z + R_s' Omega_s u_s$.
          - Return $z = M^(-1) r$.
        ]
      ]
    ]
  ]
]

#pagebreak()

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

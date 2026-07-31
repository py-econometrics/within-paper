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
    G_(W W), C_(W F), C_(W Y);
    C_(W F)', G_(F F), C_(F Y);
    C_(W Y)', C_(F Y)', G_(Y Y)
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
  G_(W W) = mat(2, 0, 0; 0, 2, 0; 0, 0, 2), quad
  G_(F F) = mat(3, 0; 0, 3), quad
  G_(Y Y) = mat(3, 0; 0, 3).
$

The off-diagonal blocks are cross-tabulations between fixed-effect
dimensions. The worker-firm block is

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

Firm $F_1$ appears twice in year $Y_1$ and once in year $Y_2$, and firm
$F_2$ has the opposite pattern. With column order
$(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$, the full Gramian is

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

The submatrix comprising the worker-worker, worker-firm, and firm-firm
blocks algebraically represents the bipartite graph formed by the worker
and firm fixed effects. The diagonal entries are worker and firm counts,
and the entries of $C_(W F)$ are edge multiplicities between workers and
firms. After flipping the sign of $C_(W F)$, the worker-firm submatrix is
the graph Laplacian of the bipartite graph
$
  L_(W F) = mat(
    G_(W W), -C_(W F);
    -C_(W F)', G_(F F);
  )
  = mat(
    augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
    2, 0, 0, -1, -1;
    0, 2, 0, -2, 0;
    0, 0, 2, 0, -2;
    -1, -2, 0, 3, 0;
    -1, 0, -2, 0, 3
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
    G_(W W), C_(W F), C_(W Y);
    C_(W F)', G_(F F), C_(F Y);
    C_(W Y)', C_(F Y)', G_(Y Y)
  ) mat(alpha_W; alpha_F; alpha_Y)
  = mat(D_W' W mu; D_F' W mu; D_Y' W mu),
$

or, equivalently,

$
    G_(W W) alpha_W + C_(W F) alpha_F + C_(W Y) alpha_Y & = D_W' W mu, \
   C_(W F)' alpha_W + G_(F F) alpha_F + C_(F Y) alpha_Y & = D_F' W mu, \
  C_(W Y)' alpha_W + C_(F Y)' alpha_F + G_(Y Y) alpha_Y & = D_Y' W mu.
$<eq:fwl-normal-akm>

Each equation can be rearranged to express one block of effects conditional
on the others. For example, using $C_(W F) = D_W' W D_F$ and
$C_(W Y) = D_W' W D_Y$, we can substitute $D_W' W$ in @eq:fwl-normal-akm
and obtain

$ alpha_W = G_(W W)^(-1)D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Because $G_(W W)$ is a diagonal matrix whose entries are workers' total
observation weights, solving for $alpha_W$ divides each worker's weighted
partial residual by its total observation weight.

For a general fixed-effect model with $q=1,dots, Q$ fixed-effect
dimensions, MAP updates the coefficient vector $alpha_q^((k+1))$ of
fixed-effect dimension $q$ at iteration $k+1$ using the Gauss-Seidel
algorithm @guimaraes2010 according to
$
  alpha_q^((k+1)) & =
  G_(q q)^(-1) D_q' W
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
  = -G_(q q)^(-1) (
    sum_(s=1)^(q-1) C_(q s) e_s^((k+1))
    + sum_(s=q+1)^Q C_(q s) e_s^((k))
  ).
$<eq:map-error-propagation>
Define the degree-scaled errors
$tilde(e)_q^((k)) = G_(q q)^(1/2) e_q^((k))$ and the degree-normalized
cross-tabulations
$
  H_(q s) = G_(q q)^(-1/2) C_(q s) G_(s s)^(-1/2).
$<eq:degree-normalized-cross-tabulation>
Then, @eq:map-error-propagation can be written as
$
  tilde(e)_q^((k+1))
  = -sum_(s=1)^(q-1) H_(q s) tilde(e)_s^((k+1))
  -sum_(s=q+1)^Q H_(q s) tilde(e)_s^((k)).
$
The degree-normalized cross-tabulations $H_(q s)$
(@eq:degree-normalized-cross-tabulation[]) appear as the off-diagonal
blocks of the normalized graph Laplacian
$
  cal(L)_(q s) & = mat(
                   G_(q q)^(-1/2), 0;
                   0, G_(s s)^(-1/2)
                 ) L_(q s) mat(
                   G_(q q)^(-1/2), 0;
                   0, G_(s s)^(-1/2)
                 ) \
               & = mat(
                   G_(q q)^(-1/2), 0;
                   0, G_(s s)^(-1/2)
                 ) mat(
                   G_(q q), -C_(q s);
                   -C_(q s)', G_(s s);
                 )
                 mat(
                   G_(q q)^(-1/2), 0;
                   0, G_(s s)^(-1/2)
                 ) \
               & = mat(
                   I, -H_(q s);
                   -H_(q s)', I
                 ),
$<eq:cross-tabulation-and-laplacian>
where $L_(q s)$ is the graph Laplacian of the bipartite graph formed by the
fixed-effect dimensions $q$ and $s$.

The connectivity of the bipartite graph formed by a pair of fixed-effect
dimensions therefore governs the speed of convergence of MAP. To see this,
note that the second-smallest eigenvalue of the normalized graph Laplacian
$lambda_2(cal(L)_(q s))$ measures graph connectivity#footnote[it is zero if
  the graph is disconnected and is small when the graph consists of nearly
  disconnected subgraphs joined only by narrow bridges
  @chung1997] and that, within each connected component, the block
structure in @eq:cross-tabulation-and-laplacian implies that
$lambda_2 (cal(L)_(q s))$ are related to the second-largest singular value
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
of the singular vector associated with $sigma_2(H_(q s))$ shrinks with
$sigma_2(H_(q s))^2$ in the worst case. In other words, a poorly connected
graph implies that $lambda_2(cal(L)_(q s))$ is small and $sigma_2(H_(q s))$
is close to one, so that the error component potentially shrinks slowly in
each MAP iteration. This leads us to define the spectral gap
$
  gamma_(q s) = 1 - sigma_2 (H_(q s))^2,
$
as a simple diagnostic for MAP convergence.#footnote[When the graph has
  more than one connected component, we compute the gap on each component
  and report the smallest, together with its observation share.] With more
than two fixed-effect dimensions, the full error propagation also depends
on the remaining dimensions and their update order, so $gamma_(q s)$ is a
pairwise diagnostic but does not fully describe MAP's convergence rate.

For the worker-firm example discussed in @sec:gramian, the gap is $1/3$.
Indeed, the diagonal blocks are $G_(W W) = "diag"(2,2,2)$,
$G_(F F) = "diag"(3,3)$, and the cross-tabulation is
$
  C_(W F) = mat(1, 1; 2, 0; 0, 2),
$
so that
$
  H_(W F) = G_(W W)^(-1/2) C_(W F) G_(F F)^(-1/2) = 1 / sqrt(6) mat(1, 1; 2, 0; 0, 2).
$
The graph is connected, and $H_(W F)' H_(W F) = 1 / 6 mat(5, 1; 1, 5)$ has
eigenvalues $1$ and $2/3$, hence $sigma_2(H_(W F)) = sqrt(2/3)$ and
$gamma_(W F) = 1 - sigma_2(H_(W F))^2 = 1/3$.

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
$
  M_("diag")^(-1) = mat(
    G_(W W)^(-1), 0, 0;
    0, G_(F F)^(-1), 0;
    0, 0, G_(Y Y)^(-1)
  ).
$

It encodes the number of observations of each fixed-effect level, but
discards the connectivity of the bipartite fixed-effect graph and therefore
does not address potentially slow convergence due to poorly connected
bipartite fixed-effect graphs.

=== The Factor-Pair Schwarz Preconditioner

To incorporate information on the pairwise graph connectivity, we use
additive Schwarz preconditioning @toselli2005 and approximate $G^(-1)$ as
the sum of inverses of the pairwise fixed-effect Gramians
$
  G_(q r)
  = R_(q r) G R_(q r)'
  = mat(
    G_(q q), C_(q r);
    C_(q r)', G_(r r)
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
  mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1)
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
  caption: [Local operator used by a factor-level MAP update (left) versus
    the factor-pair Schwarz solve (right), shown on the example worker-firm
    panel of Section 4. The factor-level block is the diagonal
    $G_(W W) = "diag"(2,2,2)$. The factor-pair block adds the firm count
    block $G_(F F) = "diag"(3,3)$ and the cross-tabulation $C_(W F)$ in its
    off-diagonal positions; the dashed outline marks the worker-firm
    subdomain on which the local Schwarz solve operates.],
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
    G_(q q), -C_(q r);
    -C_(q r)', G_(r r)
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

#pagebreak()
#[
  #counter(heading).update(0)
  #set heading(numbering: "A.1", supplement: [Appendix])
  #show heading.where(level: 1): it => {
    set block(above: 1.45em, below: 0.68em)
    text(size: 15pt, weight: "bold")[
      Appendix #counter(heading).display("A"): #it.body
    ]
  }

  = Details of the Factor-Pair Schwarz Preconditioner <sec:appendix-schwarz>

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

#bibliography(
  "refs.bib",
  style: "chicago-author-date",
  title: [References],
)

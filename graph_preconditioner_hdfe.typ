#set document(
  title: "Graph Preconditioning for High-Dimensional Fixed Effects Regression",
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
#show figure.where(kind: table): set figure.caption(position: top)
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
  #text(size: 9.5pt)[Draft: November 2026]
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
      Projections (MAP) is the standard algorithm for absorbing
      high-dimensional fixed effects in linear and generalized linear
      regressions. MAP is fast on well-connected data, but it can converge
      slowly when the fixed-effect graph is weakly connected, as in matched
      employer-employee panels with low worker mobility between firms. MAP's
      convergence rate depends on the algebraic connectivity of the weighted
      bipartite graphs formed by pairs of fixed-effect dimensions. We use
      these pairwise graphs to precondition the LSMR Krylov solver: for each
      pair of fixed-effect dimensions, we factor its bipartite graph
      Laplacian via sparse approximate Cholesky and combine the two-way
      corrections in an additive Schwarz preconditioner. Factoring the
      pairwise Laplacians adds setup time when the graph is well connected
      and MAP converges in a few sweeps, but the preconditioned solver is
      much faster when connectivity is weak. On a near-nested panel with 10
      million observations, factor-pair preconditioned LSMR finishes in 4.63
      seconds, compared with 62.2 seconds for the fastest accelerated MAP
      implementation.

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

Roughly half of published research in top economics and finance journals
mentions "fixed effects" @goldsmith2026tracking. Common high-dimensional
examples include worker and firm wage models in labor economics and patient and
provider models in health economics. Modern econometric software estimates these
specifications via the Frisch-Waugh-Lovell (FWL) theorem @frisch1933
@lovell1963: it first residualizes the outcome and covariates against the fixed
effects and then estimates the slope parameters by ordinary least squares on the
residualized data. In packages such as Stata's `reghdfe` @reghdfe @correia2017,
R's `fixest` @berge2026fixest, and Python's `PyFixest` @pyfixest, this
residualization step relies on the Method of Alternating Projections (MAP), also
known as iterative demeaning or the "zig-zag" algorithm @guimaraes2010
@gaure2013, paired with fixed-point acceleration.

MAP cycles over the fixed-effect dimensions and, for each variable in the
regression, subtracts the group mean within every level of the current
dimension. Demeaning along one dimension (such as workers) shifts the group
means along the others (such as firms or years), so MAP repeats these sweeps
until convergence. Each step updates only one dimension at a time, so
information travels across dimensions only through shared observations. In a
matched employer-employee panel with few movers between firms, mean updates pass
slowly across weakly linked groups of workers and firms, and MAP can take
thousands of sweeps to converge.

For any two fixed-effect dimensions, such as workers and firms, the factor
levels form the two node sets of a bipartite graph. An edge connects a worker
and a firm whenever both appear in the same observation, with weight equal to
their number of shared observations. Following #cite(<jochmans2019fixed>, form:
"prose"), we summarize the connectivity of factor pair $(q, r)$ by the
second-smallest eigenvalue of its normalized bipartite Laplacian,
$lambda_2(cal(L)_(q r)) in [0, 1]$ (defined in @sec:map-connectivity). Larger
values indicate stronger connectivity, and $lambda_2(cal(L)_(q r)) = 0$ when the
graph is disconnected. In our empirical benchmarks, we report
$lambda_2(cal(L)_(W F))$ for the worker-firm graph.#footnote[We compute
  $lambda_2(cal(L)_(W F))$ after iterative singleton removal. When the remaining
  graph is disconnected, we evaluate it on the connected component containing
  the most retained observations and report that component's observation share
  in parentheses.]

In this paper, we use these pairwise bipartite graphs to precondition LSMR
@fong2011, an iterative Krylov least-squares solver. A preconditioner speeds up
Krylov iterations by applying an inexpensive approximation to the inverse of the
fixed-effect cross-product matrix (Gramian) without changing the least-squares
solution.#footnote[The Julia package `FixedEffectModels.jl` @fixedeffectmodels
  also uses LSMR, but with diagonal preconditioning, which scales only by
  factor-level counts on the diagonal of the Gramian.] After a sign flip on one
factor, each two-way submatrix of the Gramian is a bipartite graph Laplacian
@correia2017. We factor each pair's Laplacian via sparse approximate Cholesky
@spielman2014 @gao2025 and sum the resulting two-way corrections in an additive
Schwarz preconditioner.

@fig-gap-runtime compares factor-pair preconditioned LSMR against MAP and
diagonally preconditioned LSMR on simulated worker-firm-year panels of 1 million
observations, both at package defaults (left column) and at accuracy-matched
tolerances within `PyFixest` (right column). At high worker-firm connectivity
$lambda_2(cal(L)_(W F))$, MAP converges in a few sweeps and diagonal LSMR in a
few iterations, so both finish faster than factor-pair LSMR, which pays an
upfront cost to factor the pairwise Laplacians. Lower mobility or stronger
sorting reduces $lambda_2(cal(L)_(W F))$. MAP then requires many more sweeps,
unpreconditioned LSMR requires many more iterations, and both can hit the
iteration cap; diagonal LSMR also slows. Factor-pair LSMR stays under 1 second
across every design.

#figure(
  image(result-img("gap_runtime.svg"), width: 100%),
  caption: [Median regression time against worker-firm graph connectivity
  $lambda_2(cal(L)_(W F))$ on logarithmic axes. Larger values indicate stronger
  connectivity, and the reversed horizontal axis places weaker connectivity to the right.
  Each simulated worker-firm-year panel has 1 million observations, with worker mobility
  varying in the top row and sorting among movers in the bottom row. The left column
  compares package defaults, including each package's singleton rule; the right column
  compares four `PyFixest` configurations at tolerances that yield comparable coefficient
  and residual errors in @sec:benchmarks. A filled marker indicates that all three planned fits returned
  estimates; a hollow marker indicates that one or two did. Lines connect observed medians,
  and an arrow marks the median time at the iteration cap (a lower
  bound on completion time); other failures are omitted.]
) <fig-gap-runtime>

@sec:absorbing sets up fixed-effect absorption, and @sec:akm introduces the AKM
@akm1999 running example. @sec:gramian and @sec:map-connectivity link the
Gramian and its pairwise bipartite graphs to MAP convergence, and
@sec:schwarz-preconditioner constructs the factor-pair preconditioner for LSMR.
@sec:benchmarks reports benchmarks, @sec:software presents the Rust, Python, and
R software, and @sec:conclusion concludes.

= Absorbing Fixed Effects <sec:absorbing>

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
In high-dimensional applications, $D$ can have hundreds of thousands or
millions of columns, making it impractical to form or invert the full cross
product of $[X quad D]$ directly.

By the Frisch-Waugh-Lovell (FWL) theorem @frisch1933 @lovell1963, we can
estimate $beta$ without forming $[X quad D]$ or inverting its cross
product. Let $M_D$ denote the linear operator that maps a variable to its
weighted least-squares residual from a regression on $D$.
Residualizing#footnote[Econometricians use several names for this
  operation: "absorbing fixed effects", "residualizing", "demeaning", and
  the "within transformation". Throughout this paper, we use "absorbing
  fixed effects" for the overall regression task, "residualizing" for the
  FWL projection step, and "demeaning" for iterative group-mean subtraction
  (MAP).] the outcome and covariates yields
$tilde(y) = M_D y$ and $tilde(X) = M_D X$. Regressing $tilde(y)$ on
$tilde(X)$ then recovers the full-model slope vector
$
  hat(beta) = (tilde(X)' W tilde(X))^(-1) tilde(X)' W tilde(y),
$
where $W$ is a diagonal matrix of non-negative observation weights. For any
right-hand side $mu$ (the outcome $y$ or a column of $X$), the
residualization step computes the weighted least-squares projection onto
the column space of $D$:

$ hat(alpha)_mu in arg min_alpha || D alpha - mu ||_W^2. $ <eq:demean-ls>

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
  normalizing each connected component.] Every FWL residualization in a
regression solves @eq:fwl-normal with the same Gramian $G$ and a different
right-hand side $mu$.

= A Running Example: The AKM Model <sec:akm>

As a running example, consider the Abowd-Kramarz-Margolis (AKM) wage
regression @akm1999:

$
  y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t),
$

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed
effect for firm $J(i,t)$ employing worker $i$ at time $t$, and $phi_t$ is a
time fixed effect.

In the worker-firm bipartite graph, stayers (workers observed at only one
employer) connect to a single firm, so their wages cannot separate worker
productivity from firm wage premia. Identification relies on movers who
change employers @bonhommeHowMuchShould2023: their wage changes across
firms create cross-firm paths, pinning down relative firm premia and, in
turn, separating worker and firm effects for their coworkers. High worker
turnover creates many short paths between firms (@fig-connectivity); low
mobility and strong assortative matching leave sparsely connected firm
clusters. Those cross-firm paths govern both the statistical precision of
the worker and firm effects and, as @sec:map-connectivity shows, the
convergence rate of iterative demeaning.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 50%),
  caption: [Worker-firm graph connectivity. When mobility is high, many
    paths connect firms. With low mobility and strong sorting, the graph
    breaks into nearly separate clusters linked by only a few movers.],
) <fig-connectivity>

= The Graph Structure of the Gramian <sec:gramian>

The worker-firm graph appears directly in the blocks of the fixed-effect
Gramian $G = D' W D$ @correia2017. Ordering the columns of $D$ as worker
levels, firm levels, and year levels partitions $G$ into nine blocks:

$
  G = mat(
    dg(G_(W W)), cr(C_(W F)), cr(C_(W Y));
    cr(C_(W F)'), dg(G_(F F)), cr(C_(F Y));
    cr(C_(W Y)'), cr(C_(F Y)'), dg(G_(Y Y))
  ).
$<eq:gramian-blocks>

The #dg[diagonal blocks] $#dg[$G_(W W)$]$, $#dg[$G_(F F)$]$, and
$#dg[$G_(Y Y)$]$ contain the weighted observation counts for each worker,
firm, and year. Each observation belongs to a single level of every
fixed-effect dimension, so these blocks are diagonal. The #cr[off-diagonal
  blocks] are pairwise cross-tabulations: $#cr[$C_(W F)$]$ records the
weighted co-occurrence count for each worker-firm pair, and $#cr[$C_(W Y)$]$
and $#cr[$C_(F Y)$]$ record the worker-year and firm-year counts.

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

Consider the unweighted ($W = I$) six-observation worker-firm panel in
@tab:example, whose bipartite worker-firm graph is shown in
@fig-toy-projection. Worker $W_1$ moves between firms $F_1$ and $F_2$;
workers $W_2$ and $W_3$ stay at $F_1$ and $F_2$, respectively.

#figure(
  image(solver-img("toy_worker_firm_projection.svg"), width: 50%),
  caption: [Worker-firm projection of the example panel. Worker $W_1$ is a
    mover; workers $W_2$ and $W_3$ are stayers.],
) <fig-toy-projection>


Each worker is observed twice and each firm and year three times, so the
diagonal blocks of $G$ in @eq:gramian-blocks are

$
  dg(G_(W W)) = mat(dg(2), dg(0), dg(0); dg(0), dg(2), dg(0); dg(0), dg(0), dg(2)), quad
  dg(G_(F F)) = mat(dg(3), dg(0); dg(0), dg(3)), quad
  dg(G_(Y Y)) = mat(dg(3), dg(0); dg(0), dg(3)),
$

and the off-diagonal cross-tabulations are

$
  cr(C_(W F)) = mat(
    cr(1), cr(1);
    cr(2), cr(0);
    cr(0), cr(2)
  ), quad
  cr(C_(W Y)) = mat(
    cr(1), cr(1);
    cr(1), cr(1);
    cr(1), cr(1)
  ), quad
  cr(C_(F Y)) = mat(
    cr(2), cr(1);
    cr(1), cr(2)
  ).
$

Row 1 of $C_(W F)$ records mover $W_1$'s observations at $F_1$ and $F_2$,
and rows 2 and 3 record the two observations each for stayers $W_2$ and
$W_3$. Ordering the columns as $(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$ gives
the full Gramian

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

Flipping the sign of $C_(W F)$ turns the two-way worker-firm submatrix of
$G$ into the graph Laplacian of the worker-firm bipartite graph:
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
to zero because each worker's or firm's total count on the diagonal equals
the sum of its incident edge weights across the row. Any pair of
fixed-effect dimensions defines a bipartite graph Laplacian in the same
way.


= Alternating Projections and Graph Connectivity <sec:map-connectivity>

MAP solves @eq:fwl-normal by updating one fixed-effect dimension at a time
without forming the off-diagonal blocks of $G$.#footnote[Most econometric
  packages pair MAP with fixed-point acceleration @correia2017 @berge2018,
  such as the Irons-Tuck extrapolation in `fixest` @irons1969
  @berge2026fixest.] Writing the AKM design matrix as
$D = [D_W quad D_F quad D_Y]$, the normal equations take the block form
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

Each equation expresses one block of effects conditional on the others.
Using $C_(W F) = D_W' W D_F$ and $C_(W Y) = D_W' W D_Y$, factoring out
$D_W' W$ in the first row of @eq:fwl-normal-akm yields

$ alpha_W = dg(G_(W W)^(-1))D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

Each diagonal entry of $G_(W W)$ is a worker's total observation weight, so
solving for $alpha_W$ divides each worker's weighted partial residual by
that weight.

In a model with $Q$ fixed-effect dimensions, sweep $k+1$ of MAP updates
dimension $q$ by block Gauss-Seidel @guimaraes2010:
$
  alpha_q^((k+1)) & =
  dg(G_(q q)^(-1)) D_q' W
  (mu - sum_(r=1)^(q-1) D_r alpha_r^((k+1)) - sum_(r=q+1)^Q D_r alpha_r^((k))).
$<eq:map-iteration>
The cross-tabulations $C_(q r) = D_q' W D_r$ are never formed in
@eq:map-iteration, yet they govern how estimation error propagates from one
dimension to the next. Let $hat(alpha) = (hat(alpha)_1, ..., hat(alpha)_Q)$
be a solution to @eq:fwl-normal and define the coefficient error at sweep
$k$ as $e_q^((k)) = alpha_q^((k)) - hat(alpha)_q$. Subtracting $hat(alpha)$
from @eq:map-iteration yields
$
  e_q^((k+1))
  = -dg(G_(q q)^(-1)) (
    sum_(r=1)^(q-1) cr(C_(q r)) e_r^((k+1))
    + sum_(r=q+1)^Q cr(C_(q r)) e_r^((k))
  ).
$<eq:map-error-propagation>
Define the degree-scaled errors
$tilde(e)_q^((k)) = G_(q q)^(1/2) e_q^((k))$ and the degree-normalized
cross-tabulations
$
  cr(H_(q r)) = dg(G_(q q)^(-1/2)) cr(C_(q r)) dg(G_(r r)^(-1/2)).
$<eq:degree-normalized-cross-tabulation>
In these scaled coordinates, @eq:map-error-propagation becomes
$
  tilde(e)_q^((k+1))
  = -sum_(r=1)^(q-1) cr(H_(q r)) tilde(e)_r^((k+1))
  -sum_(r=q+1)^Q cr(H_(q r)) tilde(e)_r^((k)).
$
The degree-normalized cross-tabulations $H_(q r)$
(@eq:degree-normalized-cross-tabulation[]) appear as the off-diagonal
blocks of the normalized graph Laplacian
$
  cal(L)_(q r) & = mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(r r)^(-1/2))
                 ) L_(q r) mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(r r)^(-1/2))
                 ) \
               & = mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(r r)^(-1/2))
                 ) mat(
                   dg(G_(q q)), cr(-C_(q r));
                   cr(-C_(q r)'), dg(G_(r r));
                 )
                 mat(
                   dg(G_(q q)^(-1/2)), cr(0);
                   cr(0), dg(G_(r r)^(-1/2))
                 ) \
               & = mat(
                   dg(I), cr(-H_(q r));
                   cr(-H_(q r)'), dg(I)
                 ),
$<eq:cross-tabulation-and-laplacian>
where $L_(q r)$ is the graph Laplacian of the bipartite graph formed by
fixed-effect dimensions $q$ and $r$.

Within each connected component, the block form in
@eq:cross-tabulation-and-laplacian relates the second-smallest eigenvalue
$lambda_2 (cal(L)_(q r))$ of the normalized graph Laplacian to the largest
nontrivial singular value of the degree-normalized cross-tabulation,
$sigma_2 (H_(q r))$, via#footnote[The eigenvalue $lambda_2 (cal(L)_(q r))$
  is zero if the graph is disconnected and small when the graph consists of
  nearly disconnected subgraphs @chung1997.]
$
  lambda_2 (cal(L)_(q r)) = 1 - sigma_2 (H_(q r)).
$
In a two-way model ($Q = 2$) with dimensions $q$ and $r$, this eigenvalue
determines the worst-case rate of MAP convergence. Updating $q$ first and
substituting into the second update $tilde(e)_r^((k+1)) = -H_(q r)'
tilde(e)_q^((k+1))$ yields
$
  tilde(e)_q^((k+1)) & = -H_(q r) tilde(e)_r^((k)), \
  tilde(e)_r^((k+1)) & = H_(q r)' H_(q r) tilde(e)_r^((k)).
$
The error component of $tilde(e)_r^((k+1))$ along the singular vector
associated with $sigma_2(H_(q r))$ contracts each sweep by the factor
$
  rho_(q r)^("MAP") = sigma_2(H_(q r))^2
  = (1 - lambda_2(cal(L)_(q r)))^2.
$
On a weakly connected bipartite graph, $lambda_2(cal(L)_(q r))$ is small
and $sigma_2(H_(q r))$ approaches one, keeping $rho_(q r)^("MAP")$ near one
so that error component shrinks slowly from sweep to sweep. With three or
more fixed-effect dimensions, error propagation also depends on the
remaining dimensions and their update order, so pairwise connectivity
$lambda_2(cal(L)_(q r))$ does not by itself determine MAP's exact
convergence rate.

For the worker-firm example in @sec:gramian, the diagonal blocks are
$G_(W W) = "diag"(2,2,2)$ and $G_(F F) = "diag"(3,3)$, and the
cross-tabulation is
$
  cr(C_(W F)) = mat(cr(1), cr(1); cr(2), cr(0); cr(0), cr(2)),
$
so that
$
  cr(H_(W F)) = dg(G_(W W)^(-1/2)) cr(C_(W F)) dg(G_(F F)^(-1/2))
  = 1 / sqrt(6) mat(cr(1), cr(1); cr(2), cr(0); cr(0), cr(2)).
$
The graph is connected, and $H_(W F)' H_(W F) = 1 / 6 mat(5, 1; 1, 5)$ has
eigenvalues $1$ and $2/3$. Here $sigma_2(H_(W F)) = sqrt(2/3)$, the
connectivity measure is $lambda_2(cal(L)_(W F)) = 1 - sqrt(2/3)$, and the
two-way MAP contraction factor is $rho_(W F)^("MAP") = 2/3$.

= The Factor-Pair Schwarz Preconditioner <sec:schwarz-preconditioner>

Our solver addresses the least-squares problem (@eq:demean-ls) jointly
across all fixed-effect dimensions using preconditioned LSMR @fong2011, a
Krylov subspace solver for sparse least squares.

== Preconditioners

LSMR's convergence rate depends on the conditioning of the fixed-effect
Gramian $G$. Recall from @eq:fwl-normal that the fixed-effect coefficients
satisfy $G alpha = D' W mu$. For a candidate vector $alpha^((k))$ at
iteration $k$, the normal-equation residual is
$
  s^((k)) = D' W mu - G alpha^((k)) = G (hat(alpha) - alpha^((k))).
$
Components of $s^((k))$ along eigenvector directions of $G$ with small
eigenvalues contribute little to the residual even when the corresponding
coefficient error $hat(alpha) - alpha^((k))$ is large. By the graph
Laplacian equivalence in @sec:gramian, these small-eigenvalue directions
correspond to weakly connected clusters in the fixed-effect graph.
Unpreconditioned LSMR eliminates well-connected error components within a
few iterations, but components tied to low mobility, strong sorting, or
near nesting shrink slowly.

Preconditioning multiplies the residual by an operator $M^(-1)$ that
approximates $G^(-1)$, so that $M^(-1) s^((k))$ approximates the
coefficient error $hat(alpha) - alpha^((k))$. If $M^(-1) = G^(-1)$, the
preconditioned operator is $M^(-1) G = G^(-1) G = I$, and the solver
reaches the exact solution in a single step. Forming and inverting $G$
directly, however, is what iterative solvers are meant to avoid. A
practical preconditioner $M^(-1)$ must be cheap to build and apply while
still approximating $G^(-1)$ along its small-eigenvalue
directions.#footnote[LSMR never forms $M^(-1) G$ or $G$ explicitly; each
  iteration requires only matrix-vector products with $D$ and $D'$ and
  applications of the linear operator $M^(-1)$.]

=== The Diagonal Preconditioner <sec:diagonal-preconditioner>

The simplest approximation to $G^(-1)$ drops the off-diagonal
cross-tabulations of the fixed-effect Gramian, yielding a diagonal
preconditioner @xu1992 @fixedeffectmodels. In the AKM model, the diagonal
preconditioner of $G$ (@eq:gramian-blocks[]) takes the form
$ M_("diag")^(-1) = "diag"(dg(G_(W W)^(-1)), dg(G_(F F)^(-1)),
  dg(G_(Y Y)^(-1))). $

Diagonal preconditioning is inexpensive to build and apply, and it performs
well when the fixed-effect graph is well connected. Like a single MAP
update, however, $M_("diag")^(-1)$ divides each factor level by its
observation count while ignoring the off-diagonal cross-tabulations
$C_(q r)$, so error modes spanning weakly connected clusters still converge
slowly.

#figure(
  image(solver-img("diagonal_lsmr_strategy.svg"), width: 65%),
  caption: [Diagonal preconditioning within LSMR. Worker, firm, and year
    counts are computed once and reused for the outcome and covariates; at
    each iteration, the preconditioner divides factor-level entries by their
    total observation weights.],
) <fig-diagonal-strategy>

=== The Factor-Pair Schwarz Preconditioner

Additive Schwarz preconditioning @toselli2005 captures pairwise
co-occurrences by approximating $G^(-1)$ with a weighted sum of inverses of
the two-way fixed-effect Gramians
$
  G_(q r)
  = R_(q r) G R_(q r)'
  = mat(
    dg(G_(q q)), cr(C_(q r));
    cr(C_(q r)'), dg(G_(r r))
  ),
$
where $R_(q r)$ restricts the coefficient vector to the coordinates of
fixed-effect dimensions $(q, r)$. For $Q >= 3$, each fixed-effect dimension
belongs to multiple pairs, so we combine their inverse contributions using
partition-of-unity weights. For each factor level $j$ in dimension $q$ or
$r$, let $c_j$ denote the number of pairwise Gramians $G_(q r)$ in which
level $j$ appears, and let $Omega_(q r)$ be the diagonal matrix with
entries $c_j^(-1/2)$. The additive Schwarz preconditioner sums these
weighted factor-pair inverses:

$
  P^(-1) = sum_(q<r) P_(q r)^(-1)
  = sum_(q<r) R_(q r)' Omega_(q r) G_(q r)^(-1) Omega_(q r) R_(q r).
$<eq:additive-schwarz>

In the three-way AKM model (@eq:gramian-blocks[]), each factor level
belongs to two pairs ($c_j = 2$), and the preconditioner sums three two-way
inverses:
$
  P^(-1) = P_(W F)^(-1) + P_(W Y)^(-1) + P_(F Y)^(-1).
$
Each term $P_(q r)^(-1)$ inverts a two-way Gramian block $G_(q r)$ that
includes the off-diagonal cross-tabulation $C_(q r)$. For the worker-firm
pair, for example,

$
  P_(W F)^(-1)
  = 1 / 2 R_(W F)'
  mat(dg(G_(W W)), cr(C_(W F)); cr(C_(W F)'), dg(G_(F F)))^(-1)
  R_(W F).
$

Including $C_(q r)$ inside each two-way inverse (@fig:pair-block) allows a
single preconditioner application to propagate corrections across the
worker-firm, worker-year, and firm-year graphs simultaneously. Decomposing
$G$ into factor pairs avoids inverting the full three-way system at once.
Even so, a single two-way block $G_(q r)$ can have hundreds of thousands of
levels, making exact inversion too expensive.

#figure(
  image(solver-img("factor_level_vs_pair_block.svg"), width: 88%),
  caption: [Count-only and factor-pair corrections for the example from @sec:gramian. The
    diagonal preconditioner (left) inverts the blue worker and firm count blocks separately;
    the factor-pair correction (right) retains those blocks together with the orange
    worker-firm observation counts $C_(W F)$.],
) <fig:pair-block>



== Approximate Factor-Pair Solves via Graph Laplacians

For preconditioning, the local factor-pair solves in @eq:additive-schwarz
need not be exact. For any two fixed-effect dimensions $(q, r)$, flipping
the sign of the second factor block transforms $G_(q r)$ into the bipartite
graph Laplacian $L_(q r)$ from @sec:gramian:
$
  L_(q r)
  = mat(
    dg(G_(q q)), cr(-C_(q r));
    cr(-C_(q r)'), dg(G_(r r))
  )
  = T_(q r) G_(q r) T_(q r),
$
where $T_(q r) = "diag"(I_q, -I_r)$ for identity matrices $I_q$ and $I_r$.
As a graph Laplacian, $L_(q r)$ admits a sparse approximate Cholesky
factorization @spielman2014 @gao2025, which we use to approximate
$
  G_(q r)^(-1) = T_(q r) L_(q r)^(-1) T_(q r).
$
In exact Cholesky (Gaussian elimination), eliminating a node with $d$
neighbors creates a dense clique of $binom(d, 2)$ new "fill-in" edges among
those neighbors, which quickly destroys sparsity when a firm employs many
workers or a worker visits many firms. Randomized approximate Cholesky
instead replaces each induced clique with a sampled spanning tree of
$d - 1$ edges chosen to preserve the Laplacian in expectation. This keeps
the triangular factor sparse, with expected construction time
$cal(O)(m log m)$, where $m$ is the number of nonzero pairwise links in
$C_(q r)$ @gao2025. Let $B_(q r)$ denote the resulting linear operator
approximating $G_(q r)^(-1)$. Replacing each exact two-way inverse
$G_(q r)^(-1)$ in the additive Schwarz sum (@eq:additive-schwarz) with its
sparse approximate Cholesky operator $B_(q r)$ yields the practical
preconditioner
$
  M^(-1) = sum_(q < r) R_(q r)' Omega_(q r) B_(q r) Omega_(q r) R_(q r).
$


== Implementation

@fig-pair-strategy summarizes the construction of $M^(-1)$. For each factor
pair, the solver splits the bipartite graph into connected components. On
large components, it eliminates the larger factor block (for example,
workers) to form a smaller Schur-complement Laplacian on the remaining
block (firms), then factors that reduced Laplacian by sparse approximate
Cholesky. Each component's correction maps back to the full coefficient
vector and combines across pairs through the partition-of-unity weights
$Omega_(q r)$. @sec:appendix-schwarz gives the block-inversion identity and
step-by-step algorithm.

The outer LSMR solver applies $M^(-1)$ at each iteration to precondition
its search directions @fong2011 @arridge2014 @yang2024flexible. The
preconditioner $M^(-1)$ depends only on $D$ and $W$, so one setup is reused
to residualize the outcome and every covariate in $X$.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Construction of the factor-pair preconditioner. For large pair
    blocks, the algorithm approximates the Schur complement and factors the
    reduced Laplacian with approximate Cholesky. The weighted pair
    corrections form the preconditioner used by LSMR.],
) <fig-pair-strategy>

= Benchmarks <sec:benchmarks>

We evaluate runtime, numerical accuracy, and preconditioner reuse on three
sets of panels: simulated AKM panels that vary worker mobility and sorting,
the 10-million-observation synthetic benchmarks from #cite(<berge2026fixest>, form: "prose"), and
eight empirical datasets from #cite(<correia2017>, form: "prose").

Elapsed times cover the package-level regression call from model setup through coefficient
estimation. Cross-package comparisons use the same input data and retain each package's
default handling of singleton levels and separated observations; experiments that
isolate the preconditioner within `PyFixest` run on a common pre-pruned sample.
@sec:appendix-memory reports the memory overhead of storing the factor-pair Cholesky
factors. Throughout this section, $lambda_2(cal(L))$ is shorthand for the algebraic
connectivity $lambda_2(cal(L)_(q r))$ of each benchmark's main factor pair, evaluated on
the largest connected component after iterative singleton removal as in @sec:introduction (with
that component's share of retained observations shown in parentheses when the graph is
disconnected).

The OLS runtime tables compare the six solver configurations listed in @tab:solvers:

#figure(
  text(size: 9.2pt)[
    #table(
      columns: (1.2fr, 0.9fr, 1.9fr),
      stroke: 0.35pt + table-light-rule,
      inset: (x: 5pt, y: 3.6pt),
      align: (left, left, left),
      table.hline(stroke: 0.8pt + table-rule),
      table.header(th[Table / Figure Label], th[Package], th[Algorithm]),
      table.hline(stroke: 0.45pt + table-rule),
      [PyFixest MAP], [`PyFixest`], [Unaccelerated MAP implemented in Rust.],
      [PyFixest LSMR, none], [`PyFixest`], [LSMR without preconditioning.],
      [PyFixest LSMR, diagonal], [`PyFixest`], [LSMR with diagonal count preconditioning.],
      [PyFixest LSMR, factor-pair], [`PyFixest`], [LSMR with factor-pair Schwarz preconditioning (`within`).],
      [`fixest`], [R's `fixest`], [Accelerated MAP with Irons-Tuck extrapolation @berge2026fixest.],
      [`FEM.jl`], [`FixedEffectModels.jl`], [Diagonally preconditioned LSMR @fong2011 @fixedeffectmodels.],
      table.hline(stroke: 0.8pt + table-rule),
    )
  ],
  caption: [OLS solver configurations compared in the benchmarks.],
) <tab:solvers>

All tables report median runtimes across repeated runs on an Apple M4 Mac mini (10 CPU
cores, 16 GB RAM, macOS 15.3.1). We omit Stata's `reghdfe` @reghdfe @correia2017 because we
do not have a Stata license; its accelerated MAP approach is represented by `fixest`.

== Runtime Benchmarks

Software packages apply their default stopping tolerances to different quantities, so
default runtimes do not guarantee equal numerical precision. We report both
package-default timings and accuracy-matched comparisons (@fig-gap-runtime and
@fig-tolerance).

The simulated AKM panels each have 1 million observations, one covariate, 10 time
periods, and worker, firm, and year fixed effects. The two design families lower
worker-firm connectivity in different ways: the first reduces the worker mobility rate
$delta$, and the second increases assortative sorting $gamma$ among movers.

=== Worker Mobility

Lowering the per-period probability $delta$ that a worker changes firms from $1$ to
$0.001$ reduces worker-firm connectivity $lambda_2(cal(L)_(W F))$ from $0.232$ to
$2.41 times 10^(-5)$ (top row of @fig-gap-runtime). At high mobility ($delta in {1, 0.5}$),
every package-default solver finishes within 3.65 seconds. At $delta = 0.001$, runtimes
diverge: `PyFixest` MAP and unpreconditioned LSMR hit the 10,000-iteration cap, while
diagonal LSMR slows to 3.16 seconds, `FEM.jl` to 4.29 seconds, and `fixest` to 7.25
seconds. Factor-pair LSMR moves in the opposite direction, falling from 0.554 seconds at
$delta = 1$ to 0.373 seconds at $delta = 0.001$ (@sec-amortization breaks down this
decline). Holding the four `PyFixest` solvers to matched accuracy (top-right panel) yields
the same ranking.

=== Sorting Among Movers

Weak connectivity slows iterative solvers even when mobility is high. The bottom row of
@fig-gap-runtime fixes $delta = 1$ so that every worker changes employers each period, and
increases an assortative-matching parameter $gamma$ that concentrates moves within
clusters of similar firms. As $gamma$ grows, fewer moves cross between firm clusters and
$lambda_2(cal(L)_(W F))$ falls from $0.222$ to $2.13 times 10^(-4)$ without thinning out
the worker-firm graph. At package defaults (bottom-left panel), `PyFixest` MAP climbs
from 0.411 to 53.9 seconds before hitting the 10,000-iteration cap at the strongest
sorting, while `fixest` reaches 6.16 seconds, `FEM.jl` reaches 2.23 seconds, and diagonal
LSMR rises from 0.362 to 1.58 seconds (2.82 seconds at matched accuracy in the
bottom-right panel). Factor-pair LSMR finishes in under 1 second at every sorting level.

=== When Preconditioner Setup Dominates

Two 10-million-observation benchmarks from the public `fixest` suite @berge2026fixest
contrast a dense, well-connected panel ($lambda_2(cal(L)) = 0.622$) with a sparse, nearly
nested panel ($lambda_2(cal(L)) = 8.35 times 10^(-8)$), each with one covariate and
worker, firm, and year fixed effects (@tab:ols).

#figure(
  kind: table,
  supplement: [Table],
  caption: [Simple and difficult `fixest` designs (10 million observations, three fixed effects).],
  block(breakable: false)[#text(size: 8.8pt)[
    #include "generated/tables/ols.typ"
    #v(0.25em)
    #align(left)[#table-note[Median times in seconds at package defaults (one covariate;
    worker, firm, and year fixed effects). A preliminary run sets the number of measured
    repetitions: 20 if under 1 second, 7 if between 1 and 10 seconds, and 3 if longer.
    `capped (0/3)` indicates that no run finished within the 10,000-iteration limit.]]
  ]]
) <tab:ols>

@tab:ols illustrates the main tradeoff of graph preconditioning. On the dense,
well-connected simple panel, iterative methods already converge in few steps, so the
approximate Cholesky setup is overhead: the five baselines finish in 2.08 to 2.72 seconds,
compared with 11.2 seconds for factor-pair LSMR. On the sparse, nearly nested difficult
panel, the ranking flips: factor-pair LSMR finishes in 4.63 seconds, more than twice as
fast as the next-fastest solver (unpreconditioned LSMR at 10.2 seconds) and over an order
of magnitude faster than `fixest` (62.2 seconds), while `PyFixest` MAP exceeds 5 minutes.
Diagonal LSMR hits the 10,000-iteration limit on difficult because equalizing the column
norms of $D$ tightens the Frobenius-scaled normal-equation stopping rule (@sec:precision)
relative to unpreconditioned LSMR.

=== Iterations After Setup

@tab:iterations separates Krylov convergence from setup cost using 100,000-observation
versions of the same two `fixest` designs.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Iterations on the simple and difficult designs (100,000 observations).],
  block(breakable: false)[#text(size: 8.9pt)[
    #include "generated/tables/iterations.typ"
    #v(0.25em)
    #align(left)[#table-note[The MAP column counts complete sweeps over all fixed-effect
    dimensions; the remaining columns count LSMR iterations, which are not directly
    comparable to MAP sweeps. Each entry is the median of three runs residualizing one
    outcome and one covariate.]]
  ]]
) <tab:iterations>

Across the two designs, factor-pair LSMR is nearly invariant to connectivity, moving from
14 to 22 iterations. Without graph preconditioning, iteration counts climb steeply on the
difficult design—reaching 180 iterations for diagonal LSMR, 249 for unpreconditioned LSMR,
and 8,159 sweeps for `PyFixest` MAP.

=== Public Real-Data Benchmarks

@tab:correia-real benchmarks the six solvers on eight empirical datasets from
#cite(<correia2017>, form: "prose"), spanning wide variation in sample size, connected
components, and two-way connectivity.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Correia real-data benchmarks.],
  block(breakable: false)[#text(size: 8.7pt)[
    #include "generated/tables/correia_real.typ"
    #v(0.25em)
    #align(left)[#table-note[Median times in seconds across three planned OLS runs at
    package defaults; `capped (0/3)` indicates that no run finished within the
    10,000-iteration limit. $lambda_2(cal(L))$ is evaluated on the `id1`-`id2` factor
    pair.]]
  ]]
) <tab:correia-real>

Results split into three connectivity regimes. On the two well-connected graphs (`credit`
and `soccer`, $lambda_2(cal(L)) >= 0.227$), every solver finishes in under 0.25 seconds.
At intermediate connectivity in the $10^(-3)$ range (`enron` and `schools`), `PyFixest`
MAP slows to 2.95 and 5.98 seconds, while the fastest solvers stay around 0.3 seconds
(`FEM.jl` at 0.317 seconds on `enron` and factor-pair LSMR at 0.266 seconds on `schools`).
On the four weakest graphs (`github`, `patents`, `workers`, and `directors`,
$lambda_2(cal(L)) <= 3.15 times 10^(-4)$), `PyFixest` MAP hits the 10,000-iteration limit
on every dataset; factor-pair LSMR finishes all four in 0.279 to 0.388 seconds and
outperforms both `fixest` and `FEM.jl` on `github`, `patents`, and `workers`. The
exception is `directors`, where `fixest` edges out factor-pair LSMR (0.283 versus 0.327
seconds): its largest connected component holds only 30% of retained observations, so
factoring many small components adds setup work while Irons-Tuck extrapolation converges
quickly on each small component.

== Runtime and Achieved Precision <sec:precision>

MAP implementations stop when changes in residuals or fixed-effect coefficients between
consecutive sweeps fall below a threshold. LSMR instead monitors the least-squares residual
and normal-equation optimality error @fong2011. In `within`, LSMR stops when either the
relative residual or the scaled normal-equation residual falls below the tolerance, where
the latter is

$ frac(||A' r||_2, ||A||_F ||r||_2), quad
  A = W^(1/2) D, quad r = W^(1/2) (mu - D alpha). $

Equal nominal tolerances can produce very different numerical accuracy: with the
contraction factor $rho^("MAP")$ close to 1 (@sec:map-connectivity), a small single-sweep
change in MAP does not guarantee a small least-squares residual. @fig-tolerance traces the
runtime-precision frontier across mobility designs 1, 3, and 5 ($delta in {1, 0.05, 0.005}$)
on a common singleton-pruned sample, measuring slope and residual errors against a
factor-pair LSMR reference fit at tolerance $10^(-14)$.

#figure(
  image(result-img("tolerance_frontier.svg"), width: 97%),
  caption: [Elapsed time and achieved accuracy on three AKM mobility designs. Each
  marker reports the median elapsed time among three planned fits on a common
  singleton-pruned sample. Lines trace each method across requested tolerances, and
  circled markers indicate package-default tolerances. The top row reports normalized
  coefficient error $abs(hat(beta)-hat(beta)^star) / "SE"(hat(beta)^star)$, and the
  bottom row reports relative residual error $frac(||r-r^star||_2, ||r^star||_2)$,
  measured against factor-pair LSMR at tolerance $10^(-14)$. Precision increases from
  left to right. Every fit is capped at 10,000 iterations; tolerances that fail to
  converge are omitted and noted in the panel annotations.]
) <fig-tolerance>

At high mobility ($delta = 1$, left column), package defaults (circled markers) already
yield slope errors below $10^(-5)$ standard errors for every solver. As mobility falls to
$delta = 0.05$ and $delta = 0.005$ (middle and right columns), step-to-step stopping rules
degrade: at its default $10^(-6)$ tolerance on $delta = 0.005$, `fixest` stops with a
slope error near $10^(-3)$ standard errors and a relative residual error above $10^(-4)$,
while `PyFixest` MAP hits the 10,000-iteration cap at every tolerance. Tightening the
tolerance to $10^(-10)$ for MAP and $10^(-12)$ for LSMR aligns achieved precision across
the `PyFixest` solvers (the matched settings in the right column of @fig-gap-runtime),
and factor-pair LSMR reaches $10^(-12)$ tolerance in under 0.6 seconds across all three
mobility levels.

== Amortizing the Preconditioner <sec-amortization>

When several regressions share the same sample, weights, and fixed effects, the
factor-pair graph Laplacians and their approximate Cholesky factors can be built once and
reused across right-hand sides.

=== Setup Cost Across Connectivity

@tab:setup-cost separates preconditioner setup time from LSMR iteration time across the
1-million-observation AKM mobility designs for models with two (worker, firm) and three
(worker, firm, year) fixed effects.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Factor-pair setup and LSMR times as worker mobility varies.],
  block(breakable: false)[#text(size: 8.8pt)[
    #include "generated/tables/akm_setup_cost.typ"
    #v(0.25em)
    #align(left)[#table-note[Median setup and LSMR solve times in seconds across five runs on 1 million
    observations at an LSMR tolerance of $10^(-12)$. The two-way model absorbs worker
    and firm fixed effects; the three-way model adds year fixed effects.]]
  ]]
) <tab:setup-cost>

As move probability $delta$ drops from $1$ to $0.001$, preconditioner setup time falls
from 0.109 to 0.028 seconds in the two-way model and from 0.113 to 0.032 seconds in the
three-way model, and LSMR iteration time declines alongside it. With fewer movers, the
worker-firm graph has fewer off-diagonal edges to factorize, which is why factor-pair
LSMR speeds up as mobility falls in @fig-gap-runtime.

=== Reuse Across 10 Regressions

Applied studies routinely estimate multiple outcomes or covariate specifications on the
same panel and fixed effects. @tab:reuse times 10 sequential regressions on the
1-million-observation simple and difficult `fixest` designs, comparing diagonal
preconditioning against factor-pair preconditioning when the preconditioner is rebuilt for
each specification versus cached across all 10.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Cumulative runtimes for 10 regressions with rebuilt and cached preconditioners.],
  block(breakable: false)[#text(size: 8.9pt)[
    #include "generated/tables/regression_reuse.typ"
    #v(0.25em)
    #align(left)[#table-note[Median cumulative setup, LSMR, and total times in seconds across three
    repetitions of 10 sequential regressions (1 million observations; worker, firm, and
    year fixed effects; LSMR tolerance $10^(-12)$). Each regression residualizes a common
    outcome and one of 10 covariates. Speedup is the ratio of the diagonal total time to
    each policy's total time. `Factor-pair, rebuilt` constructs a fresh factor-pair
    preconditioner for every regression; `Factor-pair, cached` builds the preconditioner
    once and reuses it across all 10.]]
  ]]
) <tab:reuse>

Caching cuts the 10-regression setup cost of factor-pair LSMR by roughly 10-fold on both
designs (@tab:reuse). On the simple design, amortization narrows the gap, though diagonal
LSMR remains faster overall because it needs minimal setup and few Krylov iterations. On
the difficult design, factor-pair LSMR is already much faster when rebuilt for each
regression and reaches a 33.6-fold total speedup when cached (1.30 versus 43.6 seconds).

== Poisson and Other GLMs

Preconditioner reuse also helps within a single nonlinear estimation. Poisson and other
generalized linear models with high-dimensional fixed effects @stammann2018
@correia2020ppmlhdfe are estimated by iteratively reweighted least squares (IRLS), which
solves a weighted fixed-effect residualization problem at every outer iteration. The IRLS
working weights change from step to step, but the fixed-effect identifiers and bipartite
graphs stay fixed. In `PyFixest`, `fepois` builds the factor-pair preconditioner on the
first IRLS step and reuses it across subsequent steps.

@tab:ppml compares `fixest::fepois`, `GLFixedEffectModels.jl`, and `PyFixest`'s `fepois`
(using MAP or factor-pair LSMR) on 1-million-observation versions of the simple and
difficult `fixest` designs @berge2026fixest. All packages share a 100-step IRLS limit and
use their default separation rules.

#figure(
  kind: table,
  supplement: [Table],
  caption: [Poisson benchmarks (1 million observations, one covariate).],
  block(breakable: false)[#text(size: 8.8pt)[
    #include "generated/tables/ppml.typ"
    #v(0.25em)
    #align(left)[#table-note[Median times in seconds across three planned Poisson fixed-effect regressions
    with 1 million observations, one covariate, and worker, firm, and year fixed effects.
    `fixest` is `fixest::fepois`, `GLFEM.jl` is `GLFixedEffectModels.jl`, and the two
    `PyFixest` columns use `fepois` with MAP or factor-pair LSMR (reusing the initial
    preconditioner across IRLS steps). `capped (0/3)` indicates that no run finished within
    both the inner iteration limit and the 100-step IRLS limit.]]
  ]]
) <tab:ppml>

#block(breakable: false)[All four implementations finish the well-connected simple design
in under 10 seconds. On the nearly nested difficult design, `PyFixest` MAP hits its
iteration cap and the other two baselines slow to over 2 minutes (`GLFEM.jl`) and over
7 minutes (`fixest`), while factor-pair LSMR finishes in 5.52 seconds.]

= Software <sec:software>

The factor-pair preconditioner is available in the open-source `within` library @within,
which implements the solver core in Rust and provides bindings for Python and R:

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

In `PyFixest` @pyfixest, passing `demeaner=pf.LsmrDemeaner()` to `feols` or `fepois`
switches the fixed-effect absorption backend from MAP to factor-pair LSMR:

#text(size: 8.8pt)[```python
import pyfixest as pf

fit = pf.feols(
    "y ~ x1 + x2 | worker + firm",
    data=df,
    demeaner=pf.LsmrDemeaner(),
)
```]

Users who want to residualize arrays directly can call `solve_batch` from the lower-level
Python interface, supplying zero-based integer factor codes in Fortran (column-major)
order:

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

Here `solve_batch` builds the factor-pair preconditioner once and residualizes the outcome
and both covariates into `res.demeaned`, after which `np.linalg.lstsq` estimates the
second-stage FWL slope vector from @sec:absorbing.

#pagebreak()

= Conclusion <sec:conclusion>

MAP updates one fixed-effect dimension at a time, and diagonal LSMR preconditions each
factor level independently. On well-connected graphs, those local updates converge in a
few MAP sweeps or Krylov iterations. When few observations link clusters across
fixed-effect dimensions, as in low-mobility employer-employee panels, strongly
assortative matches, or near-nested fixed effects, local updates damp the smooth error
directions slowly and standard solvers can slow down sharply or stall. Factor-pair LSMR addresses those slow directions by
preconditioning each Krylov step with sparse approximate Cholesky factors of the pairwise
bipartite Laplacians. Each preconditioner application propagates updates across entire
connected components at once, keeping iteration counts low even when pairwise
connectivity $lambda_2(cal(L))$ is small.

In practice, accelerated MAP and diagonal LSMR remain good defaults on well-connected
panels, where they incur minimal setup and memory overhead and converge quickly.
Switching to factor-pair LSMR pays off in two settings. The first is any weakly connected
specification where MAP or diagonal LSMR slows down noticeably or approaches its
iteration cap. The second is any workflow that amortizes the one-time preconditioner
setup across repeated residualizations on the same fixed-effect graph, including
regressions with many covariates, multiple OLS specifications on the same sample, or the
IRLS iterations of nonlinear models such as Poisson regression.

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

  = Memory Use <sec:appendix-memory>

  MAP stores only the current residuals and factor-level group sums; factor-pair
  LSMR also stores the pairwise cross-tabulations, partition-of-unity weights, and
  approximate Cholesky factors. @tab:memory compares peak resident memory for the two Rust
  backends inside `PyFixest` using isolated Python processes so that the surrounding
  regression overhead is held constant.

  #figure(
    kind: table,
    supplement: [Table],
    caption: [Peak physical memory use (three fixed effects, one covariate).],
    block(breakable: false)[#text(size: 8.9pt)[
      #include "generated/tables/memory.typ"
      #v(0.25em)
      #align(left)[#table-note[Peak resident physical memory in MiB ($2^20$ bytes) for an isolated Python
      process running `PyFixest` OLS with MAP or factor-pair LSMR (one covariate; worker,
      firm, and year fixed effects) on the simple and difficult `fixest` designs at 100,000
      and 1 million observations.]]
    ]]
  ) <tab:memory>

  #v(0.35em)

  Factor-pair LSMR requires #result_memory_100k_overhead of additional memory at 100,000
  observations and #result_memory_1m_overhead at 1 million observations.

  #pagebreak()

  = Details of the Factor-Pair Schwarz Preconditioner <sec:appendix-schwarz>

  == Two-Way Block Inverse

  For the two-way block

  $ G_(W F) = mat(dg(G_(W W)), cr(C_(W F)); cr(C_(W F)'), dg(G_(F F))), $

  let

  $ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

  Standard block inversion gives

  $
    G_(W F)^(-1) = mat(
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
          - Observation-level factor codes for $Q$ fixed-effect dimensions.
          - Diagonal weights $W$ and a local solver configuration.
          - Krylov residual $s$ in coefficient space.

          #strong[Preconditioner setup]
          - Enumerate all unordered factor pairs $(q, r)$ with $q < r$.
          - For each pair, build weighted count blocks $G_(q q)$, $G_(r r)$
            and the weighted cross-tabulation $C_(q r)$.
          - Split the induced bipartite graph into connected components and
            create one Schwarz subdomain $ell$ per component.
          - If factor level $j$ appears in $c_j$ subdomains, store
            the partition weight $omega_j = 1 / sqrt(c_j)$.
          - For each subdomain $ell$, form $L_ell = T_ell G_ell T_ell$. If $p_ell$ is the
            number of local levels, set
            $Pi_ell = I_(p_ell) - bold(1) bold(1)' / p_ell$; multiplying by
            $Pi_ell$ subtracts the component mean.
          - Eliminate the larger factor block, leaving a reduced system on
            the smaller block. Solve a small reduced system by dense
            Cholesky. For a large system, approximate the fill-in cliques
            by sampling and apply randomized approximate Cholesky.

          #strong[Krylov application]
          - Initialize $z = 0$.
          - For each subdomain $ell$, form $h_ell = Omega_ell R_ell s$ and
            $b_ell = Pi_ell T_ell h_ell$.
          - Apply the stored local solver to $b_ell$ to obtain $v_ell$, then
            set $u_ell = T_ell v_ell$.
          - Accumulate $z <- z + R_ell' Omega_ell u_ell$.
          - Return $z = M^(-1) s$.
        ]
      ]
    ]
  ]
]

#pagebreak()

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

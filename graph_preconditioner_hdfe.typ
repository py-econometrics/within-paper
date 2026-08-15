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
#set figure(gap: 0.7em)
#show figure: set block(below: 0.85em)
#show figure.caption: it => {
  set text(size: 8.1pt)
  set par(leading: 0.94em, spacing: 0em)
  it
}
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
      #text(weight: "bold")[Abstract.] Most software absorbs high-dimensional fixed effects
      with the Method of Alternating Projections (MAP). A MAP pass is cheap, but it updates one
      fixed-effect dimension at a time. In matched employer-employee data, sparse worker
      mobility can therefore make information travel slowly between groups of firms. We
      propose an alternative preconditioner for LSMR that uses the observed links between
      pairs of fixed effects. The worker-firm block of the fixed-effect cross product records
      worker-firm match counts; after a sign change, it is a weighted graph Laplacian. Sparse
      approximate Cholesky methods give inexpensive local corrections for this graph. We
      combine the pair corrections in an additive Schwarz preconditioner that can be reused
      across the outcome and covariates. In benchmarks, MAP and factor-pair LSMR are both fast
      on well-connected designs. As worker-firm connectivity weakens, MAP and diagonally
      preconditioned LSMR require substantially more time, while factor-pair LSMR remains
      fast. Its setup also becomes cheaper on the sparsest mobility graphs.
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
finance journals, according to #cite(<goldsmith2026tracking>, form: "prose"). Across
applied economics, labor economists use worker and firm fixed effects to separate worker
heterogeneity from firm wage premia; health economists study physician practice styles
with individual, physician, and region fixed effects in mover designs; and education
researchers use school, student, teacher, and student-teacher effects.

The Frisch-Waugh-Lovell (FWL) theorem reduces estimation to two operations: remove the
variation explained by the fixed effects from the outcome and regressors, then run a
low-dimensional regression on the residualized variables @frisch1933 @lovell1963. With
high-dimensional fixed effects, software usually performs the first operation with the
Method of Alternating Projections (MAP), also known as iterative demeaning or the
"Zig-Zag" algorithm @guimaraes2010 @gaure2013. Packages including Stata's
`reghdfe` @reghdfe @correia2017, R's `fixest` @berge2026fixest, and Python's PyFixest
@pyfixest use MAP or accelerated variants of it.

MAP updates one fixed-effect dimension at a time. In the worker-firm wage model of
#cite(<akm1999>, form: "prose"), extended here with year effects, one pass subtracts
worker means, firm means, and year means in sequence. The firm update receives the
residual left by the worker update, but it does not jointly solve the worker-firm
problem.

That distinction matters when workers rarely move between groups of firms. Movers create
the paths that separate worker effects from firm wage premia, while stayers add
observations without connecting firms @correia2017 @jochmans2019fixed. With few paths
between groups, a change on one side of the graph reaches the other side only through
successive MAP passes. The mobility pattern therefore affects identification, precision,
and computation.

We follow #cite(<jochmans2019fixed>, form: "prose") and call the second-smallest
eigenvalue of the normalized Laplacian the _Gap_. For factor pair $(q,r)$, let $L_(q r)$
be the weighted graph Laplacian and $Delta_(q r)$ its degree matrix. Then

$ S_(q r) = Delta_(q r)^(-1/2) L_(q r) Delta_(q r)^(-1/2), quad
  "Gap"_(q r) = lambda_2(S_(q r)). $

Edge weights equal the number of observed co-occurrences. A larger Gap means that the two
fixed-effect dimensions are more strongly connected. We remove singleton levels before
computing it. If the remaining graph is disconnected, we use the component containing
the most observations and report its observation share. Unless stated otherwise, Gap
refers to the worker-firm graph.

The fixed-effect cross product, or Gramian, contains this graph @correia2017. Its diagonal
blocks are worker, firm, and year observation counts. Its off-diagonal blocks are match
counts, including the worker-firm links that MAP handles only through successive
residuals. Our preconditioner groups these blocks by factor pair. A sign change turns each
pair block into a graph Laplacian, and sparse approximate Cholesky methods supply local
corrections at low cost @spielman2014 @gao2025. Their weighted sum preconditions LSMR.#footnote[
`FixedEffectModels.jl` @fixedeffectmodels also uses LSMR @fong2011, with diagonal
preconditioning based on fixed-effect counts. The factor-pair preconditioner also uses
links between fixed-effect dimensions.]

The pair factorizations create a setup cost. They are unlikely to help when MAP already
converges in a few passes, but they can save substantial iteration time on a weakly
connected graph. @fig-gap-runtime reports total regression time as mobility and sorting
change. The left column compares package defaults. The right column holds the package and
achieved accuracy fixed within PyFixest, using tolerances calibrated in Section 7.

#figure(
  image(result-img("gap_runtime.svg"), width: 100%),
  caption: [Median regression time against the worker-firm Gap $lambda_(2,W F)$, on
  logarithmic axes. The Gap is computed after singleton removal on the connected
  component containing the most observations; larger values mean stronger connectivity.
  All panels use the same reversed horizontal scale, so connectivity weakens from left to
  right. Each simulated worker-firm-year panel has 1 million observations. The top row
  varies worker mobility, while the bottom row varies sorting among movers. The left
  column compares package defaults, including each package's singleton rule. The right
  column compares four PyFixest configurations at tolerances that give comparable
  coefficient and residual errors in Section 7. A filled marker means all three planned
  fits returned estimates; a hollow marker means one or two did. Lines join observed
  medians and are not fitted trends. An arrow marks the median time at the iteration cap,
  which is a lower bound on completion time. Other failures are omitted.]
) <fig-gap-runtime>

All implementations finish quickly at high Gap values. MAP and unpreconditioned LSMR
slow sharply as the Gap falls; factor-pair LSMR remains fast, and its runtime falls in the
lowest-mobility designs because the local graphs become cheaper to factorize.

The next four sections introduce fixed-effect absorption, the AKM example, the Gramian,
and the link between MAP and graph connectivity. Section 6 develops the factor-pair
preconditioner. Sections 7 and 8 report the benchmarks and software interface, Section 9
concludes, and the appendix compares memory use.

= Absorbing Fixed Effects#footnote[Researchers employ several names for this operation:
"absorbing fixed effects", "demeaning", "residualizing", or applying the "within
transformation". We use these terms interchangeably throughout.]

We study the linear model

$ y = X beta + D alpha + epsilon, $

where $X$ contains the regressors of interest, $D$ is the fixed-effect design matrix,
and $alpha$ collects the fixed-effect coefficients. Because $D$ may have hundreds of
thousands or millions of columns in high-dimensional applications, forming $[X quad D]$
or inverting its cross product can be computationally infeasible. The
Frisch-Waugh-Lovell (FWL) theorem lets us compute $hat(beta)$ without either operation.

FWL removes the fixed-effect means from $y$ and each column of $X$, then regresses the
residualized outcome on the residualized covariates. Let $M_D$ denote the operation that
removes those means. In matrix terms, it is the weighted residual-making projection for
the fixed-effect indicators, so $tilde(y) = M_D y$ and $tilde(X) = M_D X$. The second
regression gives

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

$ D' W (D hat(alpha)_mu - mu) = 0, quad
  G hat(alpha)_mu = D' W mu, quad G = D' W D, $ <eq:fwl-normal>

determines $hat(alpha)_mu$. Its coefficient matrix $G$ is the same for each FWL
residualization; only the variable being residualized changes, so the computational cost
depends on the structure of the Gramian $G$, the weighted cross-product matrix of the
fixed-effect dummies. In the AKM worker-firm model, this structure has a concrete
interpretation: worker moves link firms through shared workers and determine the
off-diagonal blocks of $G$.

= A Running Example: The AKM Model

Throughout the paper, we use the AKM model of #cite(<akm1999>, form: "prose") as the
worker-firm example, in which workers observed at more than one firm provide the
comparisons needed to distinguish persistent worker heterogeneity from firm wage premia.
We write the regression as

$ y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t), $

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed effect for the
firm employing worker $i$ at time $t$, and $phi_t$ is a time fixed effect.

The AKM specification can be represented as a bipartite graph, with workers and firms as
nodes and employment spells as edges. Worker moves induce a firm-to-firm graph by linking
firms that employ the same worker; stayers - workers who never change employer - add
observations to existing worker-firm links but do not bridge firms. Year effects enter
as a third, low-dimensional factor on the same records, and @fig-connectivity contrasts
a well-connected mobility graph with one that fragments under strong sorting.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 50%),
  caption: [Worker-firm graphs under high and low mobility. High mobility creates many
  paths between firms. With low mobility and strong sorting, only a few worker moves
  connect otherwise separate groups of firms.]
) <fig-connectivity>

A worker observed at one firm provides no comparison between worker productivity and the
firm wage premium. Movers do: wages for the same worker at different firms help separate
the two effects, especially when the move connects groups that otherwise share few
workers. These comparisons are edges in the mobility graph and entries in the
worker-firm cross-tabulation of the Gramian.

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

The small worker-firm panel below illustrates how its Gramian is constructed. We set
all regression weights to one, so $W = I$.

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

Worker $W_1$ supplies the only link between $F_1$ and $F_2$ in
@fig-toy-projection. Workers $W_2$ and $W_3$ stay at one firm.

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

The first row records the mover's two matches; the remaining rows record the two
stayers. The other cross-tabulations are

$ C_(W Y) = mat(
  1, 1;
  1, 1;
  1, 1
), quad
  C_(F Y) = mat(
  2, 1;
  1, 2
). $

Together, the count and cross-tabulation blocks form the full Gramian.

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

The worker-firm submatrix contains worker and firm counts on its diagonal and match counts
in $C_(W F)$. Changing the sign of $C_(W F)$ makes the off-diagonal entries non-positive.
Each row then sums to zero because its diagonal observation count equals the sum of its
match counts. The result is the weighted graph Laplacian


$ L_(W F) = mat(augment: #(hline: 3, vline: 3, stroke: 0.4pt + rgb("#b0b8c4")),
  2, 0, 0, -1, -1;
  0, 2, 0, -2, 0;
  0, 0, 2, 0, -2;
  -1, -2, 0, 3, 0;
  -1, 0, -2, 0, 3
). $

Every factor pair has a Laplacian of this form. Section 6 uses these matrices in the
preconditioner; MAP uses only their diagonal count blocks in each update.


= Alternating Projections and Graph Connectivity

The Method of Alternating Projections (MAP), also called iterative demeaning or the
"zig-zag" algorithm, is the standard method for absorbing several fixed effects
@guimaraes2010 @gaure2013. Implementations often accelerate the basic iteration
@berge2018 @correia2017; `fixest`, for example, uses Irons-Tuck extrapolation
@irons1969 @berge2026fixest.

In the worker-firm-year model, MAP starts with a partial residual and subtracts its mean
within each worker. It next computes firm means from the updated residual, then year
means. Repeating this pass eventually removes all three sets of means.

With $D = [D_W quad D_F quad D_Y]$, the FWL normal equations in @eq:fwl-normal are

$ mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
) mat(alpha_W; alpha_F; alpha_Y)
= mat(D_W' W mu; D_F' W mu; D_Y' W mu). $

The worker update holds the current firm and year effects fixed. The worker row of this
system can be written as

$ G_(W W) alpha_W = D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

The diagonal entries of $G_(W W)$ are workers' total observation weights. Dividing the
right-hand side by those counts gives the weighted worker means. Firm and year updates
have the same form. One complete MAP pass applies these three inexpensive group-mean
calculations in sequence.

The match-count blocks enter only through the changing partial residual. MAP never solves
a worker-firm, worker-year, or firm-year block jointly.

With many overlapping employment histories, a worker update changes residuals at many
firms, and the next firm update carries that information onward. Sparse mobility, strong
sorting, or near nesting confines the changes to a narrower part of the graph. MAP may
then need many passes even though each pass remains cheap.

The factor-pair Gap $lambda_(2,q r)$ summarizes the strength of these links. A small Gap
is a warning that information may move slowly across the pair graph. In a model with
three or more fixed effects it remains a diagnostic for one pair, not a bound on the
convergence rate of the full model.

= The Factor-Pair Schwarz Preconditioner

== Preconditioners

LSMR @fong2011 allows each residual correction to use a preconditioner. Weak links,
sparse mobility, or near nesting make some combinations of fixed effects much less well
determined than others. Changing from MAP to LSMR does little to remove that imbalance;
the preconditioner must rescale the system while preserving its least-squares solution.

The inverse Gramian $G^(-1)$ shows the target. If $M^(-1) = G^(-1)$, LSMR sees the
identity matrix,#footnote[As in any model with several fixed
effects, the level effects are pinned down only up to a normalization: we can add a
constant to every worker effect and subtract it from every firm effect without changing
the fitted values $D alpha$, and likewise for years. The dummy-coded $G = D' W D$ and the
smaller blocks inverted below are therefore not invertible until this ambiguity is
removed - one normalization for the worker-firm pair, and a second once year effects are
added, as in the Section 4 example. We adopt the standard normalization within each
connected set and read every inverse below as that of the resulting system. The estimated
effects depend on the normalization; the residualized outcome and regressors, which are
all the regression uses, do not.]

$ M^(-1) G = G^(-1) G = I. $ <eq:ideal-preconditioner>

Under this choice, one correction would suffice. Computing the full inverse costs as much as the original
fixed-effect problem. A useful approximation must capture the poorly determined
directions and save enough iterations to cover its construction and application costs.#footnote[
LSMR never constructs $M^(-1) G$ or $G$ explicitly. It multiplies vectors by $D$ and
$D'$ and applies $M^(-1)$.]

== From the Block Inverse to the Diagonal Preconditioner

The block inverse identifies the information lost by diagonal scaling. For the
worker-firm-year model,

$ G = mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
), $

has diagonal weighted-count blocks and off-diagonal match-count blocks. Focus on the
worker-firm part,

$ G_(W F) = mat(G_(W W), C_(W F); C_(W F)', G_(F F)). $

Its inverse is

$ G_(W F)^(-1) = mat(
  G_(W W)^(-1) + G_(W W)^(-1) C_(W F) S^(-1) C_(W F)' G_(W W)^(-1), -G_(W W)^(-1) C_(W F) S^(-1);
  -S^(-1) C_(W F)' G_(W W)^(-1), S^(-1)
). $

$ S = G_(F F) - C_(W F)' G_(W W)^(-1) C_(W F). $

The diagonal inverse divides by worker counts. Worker-firm match counts enter through the
Schur complement $S$, a firm-side system that compares firms after accounting for their
shared workers. This is the expensive part. Exact factorization can add many nonzero
entries, with time and memory determined by the connected-component sizes and the amount
of fill-in. Sparse mobility can make an unpreconditioned iteration slow while making the
local factorization cheaper because there are fewer links. With three factors, the full
inverse can contain all three pairwise cross-tabulations.

Diagonal preconditioning drops the Schur-complement corrections and keeps only the count
inverses,

$ M_("diag")^(-1) = "diag"(G_(W W)^(-1), G_(F F)^(-1), G_(Y Y)^(-1)). $

This is the preconditioner used in `FixedEffectModels.jl` @fong2011
@fixedeffectmodels. It corrects differences in
group size. It cannot distinguish a firm linked to many employers from an equally large
firm whose workers remain in one part of the mobility graph.

== The Factor-Pair Schwarz Approximation

Additive Schwarz preconditioning adds corrections from smaller, overlapping problems
@xu1992 @toselli2005. For the AKM model, the subproblems are worker-firm, worker-year, and
firm-year. Each retains its match-count block, while the outer LSMR iteration handles the
remaining three-way coupling.

The worker-firm local problem is

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

Its inverse retains the worker-firm links through the Schur complement. To place this
correction in the three-factor problem, let
$R_(W F)$ select the worker and firm entries from the full coefficient vector $alpha =
[alpha_W; alpha_F; alpha_Y]$; its transpose $R_(W F)'$ places the resulting correction
back into the full vector. The diagonal matrix $tilde(D)_(W F)$ weights levels that
appear in more than one pair problem with partition-of-unity weights, so their
corrections are not counted twice. The exact worker-firm contribution is

$ P_(W F)^(-1) =
  R_(W F)' tilde(D)_(W F)
  mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1)
  tilde(D)_(W F) R_(W F). $

The worker-year and firm-year terms have the same form. Each level belongs to two pair
problems, so its weight in each is $1 / sqrt(2)$; the squared weights then sum to one.
Adding the three terms gives the exact factor-pair Schwarz preconditioner,

$ P^(-1) = P_(W F)^(-1) + P_(W Y)^(-1) + P_(F Y)^(-1). $

The pair terms capture all three cross-tabulations separately. LSMR resolves the joint
three-factor coupling and the error from splitting shared levels across pairs.

== Approximating Pair Systems via Graph Laplacians

Direct inversion is practical only for small pair blocks. The Section 4 example reduces
to a $4 times 4$ system after normalization, but register data can contain hundreds of
thousands of workers and firms. Large blocks require an approximation that uses their
graph-Laplacian structure.

For a worker-firm pair, the local pair step solves the pair-Gramian system

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F)) x = u, $

where $u$ contains the selected and weighted worker and firm entries of the LSMR vector.
The match counts $C_(W F)$ are non-negative, so this matrix is not yet a Laplacian. Let
$T_(W F) = "diag"(I_W, -I_F)$ flip the firm signs. Multiplication on both sides gives

$ L_(W F) = T_(W F) mat(G_(W W), C_(W F); C_(W F)', G_(F F)) T_(W F)
  = mat(G_(W W), -C_(W F); -C_(W F)', G_(F F)), $

a weighted bipartite graph Laplacian with non-positive off-diagonals and zero row sums.
Since $T_(W F)^2 = I$, the same sign change maps its inverse back to the pair Gramian,

$ mat(G_(W W), C_(W F); C_(W F)', G_(F F))^(-1) = T_(W F) L_(W F)^(-1) T_(W F), $

where the inverses use the normalization from Section 6.1. The implementation returns
the zero-mean solution within each connected component. This choice fixes the free
constant but does not change the residualized variables.

The local correction need not be exact because LSMR refines the remaining error. We use
sparse approximate Cholesky factorizations @spielman2014 @gao2025. Exact elimination can
create fill-in: removing a worker links the firms that employed that worker, and later
eliminations add more entries. A dense $k$-level factorization can require order $k^3$
operations and $k^2$ memory. Randomized approximation limits this growth, with cost close
to the number of observed links up to logarithmic factors. Let $A_(W F)$ denote the
resulting approximate pair-Gramian inverse after the signs are changed back.

The worker-year and firm-year approximations are $A_(W Y)$ and $A_(F Y)$. Substituting
all three approximate inverses into the Schwarz sum gives the implemented preconditioner,

$ M^(-1) = sum_((q, r)) R_(q r)' tilde(D)_(q r) A_(q r) tilde(D)_(q r) R_(q r). $

The exact $P^(-1)$ replaces the full three-factor inverse with exact pair inverses, while
$M^(-1)$ also approximates each pair inverse through $A_(q r)$. Both approximations change
the update at each LSMR iteration but leave the original fixed-effect least-squares
problem unchanged. LSMR therefore returns the same fitted residuals as an
unpreconditioned LSMR run, up to the requested tolerance.

== Implementation Strategy

@fig-pair-strategy summarizes the implementation. Each factor pair becomes a graph
Laplacian after the sign change. Sparse approximate Cholesky supplies the local inverse,
and partition-of-unity weights prevent shared levels from being counted twice in the
Schwarz sum.

LSMR uses the sum to scale its updates @fong2011 @arridge2014 @yang2024flexible. The same
preconditioner can residualize the outcome and every covariate when the observations,
weights, and fixed-effect identifiers are unchanged. The `within` source implements this
setup-and-apply sequence @within.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 70%),
  caption: [Construction of the factor-pair preconditioner. Each local problem combines
  two fixed-effect dimensions and retains their observed links. A sign change turns its
  matrix into a graph Laplacian, the standard matrix representation of a weighted graph.
  Sparse approximate Cholesky solves each local system while limiting the extra nonzero
  entries created during factorization, which saves memory and computation. The weighted
  sum of these local solutions is the preconditioner for LSMR.]
) <fig-pair-strategy>

= Benchmarks

The benchmarks ask when fewer iterations repay the cost of building the factor-pair
preconditioner. We vary mobility and sorting in controlled AKM panels, compare two public
synthetic designs, and use eight real datasets from #cite(<correia2017>, form: "prose").

Elapsed times cover the package-level regression call, from model setup through
coefficient estimation. Cross-package comparisons use the same input data but retain each
package's default treatment of singleton levels and separated observations. Experiments
that isolate the preconditioner instead use one prepared sample. The appendix reports the
additional memory used to store the pair structures.

Each table reports the factor-pair Gap after iterative singleton removal. For a
disconnected graph, the reported value comes from the component with the most retained
observations; the number in parentheses is its observation share. The Gap remains a
pairwise diagnostic in models with three or more fixed effects.

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
worker, firm, and year effects. One set of designs lowers the probability of moving. The
other holds that probability at one but increasingly assigns moves within groups of
similar firms.

=== Worker Mobility

The worker-firm Gap falls from $0.232$ to $2.41 times 10^(-5)$ as mobility declines. All
configurations finish within 3.60 seconds in the first two designs. PyFixest MAP then
slows sharply or reaches its iteration cap, and unpreconditioned LSMR also reaches the
cap. Factor-pair LSMR falls from 0.554 seconds in the first design to 0.373 seconds in the
last because the local worker-firm graphs become cheaper to factorize.

=== Sorting Among Movers

Increasing $rho$ concentrates moves within groups even though every worker moves between
periods. The Gap falls from $0.222$ to $2.13 times 10^(-4)$. PyFixest MAP rises from
0.411 seconds to 53.9 seconds before reaching its cap in the final design;
unpreconditioned LSMR reaches its cap in the last four. Diagonal LSMR rises from 0.362 to
1.58 seconds. Factor-pair LSMR stays below one second and is faster than diagonal LSMR in
the three designs with the smallest Gap.

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
  observations, one covariate, and worker, firm, and year fixed effects. Gap
  $lambda_2$ is computed after iterative singleton removal on the component containing
  the most retained observations; its observation share appears in parentheses.]
]]

The dense design exposes the setup cost: five methods finish in 2.08 to 2.72 seconds,
while factor-pair LSMR takes 11.2 seconds. Near nesting reverses the ranking. Factor-pair
LSMR takes 4.63 seconds, compared with 10.2 seconds without preconditioning, 27.0 seconds
for FEM.jl, 62.2 seconds for `fixest`, and 326.2 seconds for PyFixest MAP; diagonal LSMR
reaches its default cap.

=== Iterations After Setup

The iteration-count diagnostic removes setup time and uses 100,000-observation versions
of the same designs.

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
  before the 10,000-iteration limit. Gap $lambda_2$ is computed after iterative singleton
  removal on the `id1`-`id2` component containing the most retained observations; its
  observation share appears in parentheses.]
]]

On `credit` and `soccer`, PyFixest MAP takes 0.193 and 0.026 seconds, compared with 0.242
and 0.065 seconds for factor-pair LSMR, so the preconditioner's setup cost does not pay
on these easy datasets. On the more difficult graphs, factor-pair LSMR takes 0.402
seconds on `enron` and 0.266 seconds on `schools`, compared with 2.95 and 5.98 seconds
for PyFixest MAP. On `github`, `patents`, `workers`, and
`directors`, PyFixest MAP reaches its iteration cap while factor-pair LSMR finishes in
0.279 to 0.388 seconds. Accelerated MAP can still be competitive, however; `fixest`
takes 0.283 seconds on `directors`, slightly less than factor-pair LSMR's 0.327 seconds.

The Gap is a pairwise description, not a rule for choosing a solver. Runtime also
depends on component size, factor dimensions, graph sparsity, acceleration, and the
preconditioner's setup cost.

== Runtime and Achieved Precision

MAP implementations compare changes in residuals or fixed-effect coefficients, while
LSMR uses its estimated least-squares residual and first-order-condition error @fong2011.
Their numerical tolerances are therefore not a common measure of accuracy. For example,
`within` stops when either its relative residual or the scaled normal-equation error
$||A^T r||_2/(||A||_F ||r||_2)$ meets the requested threshold, where
$A=W^(1/2)D$ and $r=W^(1/2)(mu-D alpha)$.

@fig-tolerance places the methods on a common accuracy scale for mobility designs 1, 3,
and 5. Every method receives the same singleton-pruned sample and a common
10,000-iteration cap. We time three fits at each package-specific tolerance and compare
their coefficients and residuals with factor-pair LSMR at tolerance $10^(-14)$.

#figure(
  image(result-img("tolerance_frontier.svg"), width: 97%),
  caption: [Elapsed time and achieved accuracy on three AKM mobility designs. Each
  marker reports the median elapsed time in seconds among the three planned fits that
  return an estimate on the same sample. Before timing, we remove
  fixed-effect levels observed only once and use the remaining observations for every
  method. Each line connects results for one method across requested tolerances. The top
  row measures coefficient error as
  $abs(hat(beta)-hat(beta)^star) / "SE"(hat(beta)^star)$. The bottom
  row measures residual error as $frac(||r-r^star||_2, ||r^star||_2)$. The reference
  coefficient $hat(beta)^star$ and residual $r^star$ come from factor-pair LSMR at
  tolerance $10^(-14)$. Error decreases from left to right; on each line, a circle marks
  the result obtained with the package's default tolerance. Every fit is limited to
  10,000 iterations. Settings for
  which none of the three fits returns an estimate are omitted and identified in the
  annotations.]
) <fig-tolerance>

== Amortizing the Preconditioner <sec-amortization>

Achieved accuracy does not remove the factor-pair setup cost. Later regressions can reuse
the pair blocks and factorizations when the observations, weights, and fixed-effect
identifiers are unchanged.

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
  specification also absorbs year effects. Gap $lambda_2$ is computed after iterative
  singleton removal on the component containing the most retained observations; its
  observation share appears in parentheses.]
]]

From the highest- to the lowest-mobility design, setup falls from 0.109 to 0.028 seconds
with two fixed effects and from 0.113 to 0.032 seconds with three. Time spent in LSMR also
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

Poisson fixed-effect models repeat the absorption step at every iteratively reweighted
least-squares (IRLS) iteration @stammann2018. The fixed-effect identifiers remain fixed
while the weights and working response change, so residualization speed accumulates over
the fit. This is the setting used by `ppmlhdfe` @correia2020ppmlhdfe and related
estimators.

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
ten seconds. The difficult design separates them: PyFixest MAP reaches its iteration cap,
`GLFEM.jl` takes 129.8 seconds, `fixest` takes 439.3 seconds, and factor-pair LSMR takes
5.52 seconds. Repeated absorption magnifies the cost of slow propagation across the
worker-firm graph.]

= Software

The open-source `within` project provides the solver benchmarked in Section 7 @within,
with a computational core written in Rust and APIs for Python and R.

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

Applied users can select the method from PyFixest without constructing the Gramian or its
pair blocks @pyfixest:

#text(size: 8.8pt)[```python
import pyfixest as pf

fit = pf.feols(
    "y ~ x1 + x2 | worker + firm",
    data=df,
    demeaner=pf.LsmrDemeaner(),
)
```]

The lower-level Python interface exposes the FWL operation. Here `worker_code` and
`firm_code` are zero-based integer codes:

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

`solve_batch` residualizes all three columns with one factor-pair preconditioner. The last
line is the low-dimensional FWL regression from Section 2.

#pagebreak()

= Conclusion

MAP's low cost per pass makes it the right default for many well-connected fixed-effect
models. Its weakness appears when mobility is sparse, sorting is strong, or one effect is
nearly nested in another. Successive mean updates then move information slowly through
the graph.

Factor-pair preconditioning changes the update, not the regression. It uses observed
co-occurrences to correct worker-firm and other pair blocks jointly, while LSMR preserves
the original least-squares target. The benchmarks show that this setup cost is unnecessary
on easy graphs but can replace hundreds of MAP passes or LSMR iterations on difficult
ones. Sparse pair graphs can also be cheaper to factorize than dense ones.

The practical case is strongest when the factor-pair Gap is small or when the same fixed
effects are absorbed repeatedly. Specification searches can reuse one preconditioner
across outcomes and covariates, and IRLS estimators call the demeaning routine at every
outer iteration. In these settings, graph structure is useful computational information,
not only an identification diagnostic.

#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 1.45em,
  below: 0.68em,
)[#text(size: 15pt, weight: "bold", it.body)]
#pagebreak()

= Appendix: Memory Use

MAP stores the current residuals and group sums. Factor-pair LSMR also retains match
structures and local factorizations. We compare peak resident memory for the two Rust
implementations inside PyFixest, using isolated processes so the surrounding regression
code is held fixed.

#v(0.4em)

#block(breakable: false)[#text(size: 8.9pt)[
#strong[Peak physical memory use (three fixed effects, one covariate).]
#include "generated/tables/memory.typ"
  #v(0.25em)
  #table-note[For each regression, we record the largest amount of
  physical memory used by an isolated Python process, in MiB ($2^20$ bytes). The table
  compares PyFixest OLS regressions using MAP or factor-pair LSMR. Each regression has
  one covariate and worker, firm, and year fixed effects. The results cover the simple
  and difficult designs at 100,000 and 1 million observations. Gap $lambda_2$ is computed
  after iterative singleton removal on the component containing the most retained
  observations; its observation share appears in parentheses.]
  ]]

#v(0.35em)

Factor-pair LSMR uses slightly more memory: #result_memory_100k_overhead at 100K
observations and #result_memory_1m_overhead at 1M. The additional storage holds the
pairwise co-occurrences, partition weights, and approximate Cholesky factors.

#pagebreak()

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

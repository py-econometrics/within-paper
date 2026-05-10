#set document(
  title: "A Fast Graph-Based Solver for Fixed-Effects Regressions",
  author: "Alexander Fischer and Kristof Schröder",
)
#set page(
  paper: "a4",
  margin: (x: 2.35cm, y: 2.25cm),
  numbering: "1",
  number-align: center,
)
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.92em, spacing: 1.05em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#set figure(gap: 0.8em)
#show heading.where(level: 1): it => {
  set block(above: 1.25em, below: 0.58em)
  text(size: 15pt, weight: "bold", it)
}
#show heading.where(level: 2): it => {
  set block(above: 0.95em, below: 0.38em)
  text(size: 12pt, weight: "bold", it)
}

#let pf-fig(name) = "assets/pyfixest/" + name
#let solver-img(name) = "figures/solver/" + name
#let table-rule = rgb("#7b8494")
#let table-light-rule = rgb("#d8dee8")
#let table-head-fill = rgb("#eef2f7")
#let th(body) = table.cell(fill: table-head-fill)[#strong(body)]
#let miss = text(fill: rgb("#777777"))[--]
#let dg(body) = text(fill: rgb("#2563eb"), body)
#let cr(body) = text(fill: rgb("#c2410c"), body)

#align(center)[
  #text(size: 19pt, weight: "bold")[A Fast Graph-Based Solver for Fixed-Effects
  Regressions]

  #v(0.65em)
  #text(size: 10.5pt)[Alexander Fischer and Kristof Schröder]

  #v(0.4em)
  #text(size: 9.5pt)[Draft: May 2026]
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
      #text(weight: "bold")[Abstract.] High-dimensional fixed-effect regressions are
      computationally challenging because the full dummy-variable design
      $Z = [X quad D]$ is very wide. Forming and inverting the associated Gramian
      $G = Z' Z$ is often prohibitive in both memory and runtime. Modern
      implementations therefore rely on the Frisch-Waugh-Lovell theorem @frisch1933
      @lovell1963 to reduce estimation to repeated fixed-effect residualization, and
      solve these residualization problems either with the Method of Alternating
      Projections (MAP),
      as in `reghdfe`, `fixest`, and PyFixest @reghdfe @berge2026fixest @pyfixest, or
      with Krylov solvers and diagonal preconditioners, as in `FixedEffectModels.jl`
      @fong2011 @fixedeffectmodels. We study a solver for fixed-effects regressions
      that builds on insights in @correia2017: fixed-effect models have a
      graph interpretation, and the fixed-effect Gramian has a distinctive numerical
      structure. We combine this graph
      view with ideas from Laplacian solvers and approximate Cholesky
      preconditioning @spielman2014 @gao2025 to construct a Schwarz preconditioner from
      factor-pair subproblems and pair it with a Krylov solver. This addresses a core
      limitation of MAP-based residualization: cross-factor information moves only
      indirectly through successive residual updates. In sparse, poorly connected
      fixed-effect graphs, that transmission can be slow. The proposed solver instead
      uses cross-factor co-occurrences directly inside local factor-pair solves, while
      leaving the Frisch-Waugh-Lovell estimand unchanged. On dense, well-connected
      designs the proposed solver is competitive with MAP-based packages but does not
      improve on them; on sparse worker-firm graphs and under strong sorting it
      substantially outperforms them, with near-flat runtimes as worker mobility falls
      by an order of magnitude or sorting strengthens.
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
  #text(size: 9.1pt)[#strong[JEL codes:] C55; C63; C81; J31]
]

= Introduction

Fixed-effects regressions are ubiquitous in applied econometrics. Labor economists use
worker and firm fixed effects to separate worker heterogeneity from firm wage premia
@card2013; health economists use doctor and patient fixed effects @molitor2018;
innovation researchers use inventor and patent data @jaravel2018; trade economists
absorb exporter, importer, product, and time fixed effects @head2014; and education
researchers study student-teacher panels @chetty2014. Mover designs use the same
logic in settings where workers move across firms @akm1999 @card2013, patients or
physicians move across health-care markets @finkelstein2016 @molitor2018, and families
move across neighborhoods or counties @chettyHendren2018county.

Throughout the paper, we will consider the linear model

$ y = X beta + D alpha + epsilon. $

Here $X$ contains the regressors of interest, $D$ is the fixed-effect design matrix, and
$alpha$ collects the fixed-effect coefficients. Direct estimation would use the full
design $Z = [X quad D]$ and the Gramian $Z'Z$. In high-dimensional fixed-effect
applications, $D$ can have hundreds of thousands or millions of columns, so forming this
Gramian, storing it, or applying a direct inverse is often infeasible.

The standard computational starting point is the Frisch-Waugh-Lovell (FWL) theorem
@frisch1933 @lovell1963. In the one-regressor case, FWL states that the
coefficient on $x$ in the full regression of $y$ on $x$ and $D$ can be obtained from
three regressions: regress $y$ on $D$, regress $x$ on $D$, and regress the residualized
outcome on the residualized covariate. With several regressors, the second step is
repeated for every column of $X$. Applied to high-dimensional fixed effects, the
computational problem is therefore the efficient solution of many regressions of the
form $mu$ on $D$, where $mu$ is either $y$ or a column of $X$.

The workhorse method for these fixed-effect regressions is the Method of Alternating
Projections (MAP), also known as iterative demeaning or the Zig-Zag algorithm
@guimaraes2010 @gaure2013. `reghdfe`, `fixest`, and PyFixest use MAP or
MAP-based variants with acceleration @correia2017 @reghdfe @berge2026fixest
@pyfixest.

To make the algorithm concrete, consider a worker-firm specification, which will serve
as the running example throughout the paper. MAP avoids forming and solving the full
fixed-effect normal equations by cycling through the fixed-effect dimensions. One step
subtracts worker means from the current residual; the next subtracts firm means from the
updated residual; subsequent sweeps repeat these one-factor corrections until
convergence. The relationship between workers and firms is therefore handled only
indirectly: worker updates change the residual seen by the firm update, and firm updates
change the residual seen by the next worker update. MAP never solves a local
worker-firm system directly.

The fixed-effect Gramian has a graph interpretation @correia2017. Its off-diagonal
blocks record co-occurrences among absorbed factors: which workers appear at which
firms, which patients see which doctors, or which families move across which places. In
a worker-firm panel, movers create paths between firms, while stayers add observations
without connecting firms to one another. When this graph is sparse or poorly connected,
some fixed-effect directions are nearly collinear, and MAP can require many sweeps to
move information across the graph.

A different approach is to solve the fixed-effect least-squares problem with a Krylov
method. `FixedEffectModels.jl` uses LSMR with diagonal preconditioning @fong2011
@fixedeffectmodels. LSMR avoids forming or factorizing the fixed-effect Gramian
$D' W D$ by using products with $D$ and $D'$. These products reflect the observed
fixed-effect links, but the diagonal preconditioner itself uses only fixed-effect level
counts. It improves numerical scaling by accounting for the diagonal count blocks of the
Gramian, but it does not use the off-diagonal co-occurrence blocks directly.

Our approach also uses a Krylov solver, but changes the preconditioner. Instead of
preconditioning only with fixed-effect level counts, it builds a Schwarz preconditioner
from local factor-pair problems. These local problems use the graph structure of the
Gramian directly: for example, a worker-firm subproblem includes the observed links
between workers and firms. After a sign change, each local factor-pair problem can be
treated as a Laplacian problem. Approximate Cholesky ideas from the Spielman-Teng
Laplacian-solver literature are used inside these local solves, making the factor-pair
corrections sparse enough to use in the preconditioner.

The resulting preconditioner is a closer approximation to the fixed-effect Gramian than
diagonal scaling. Diagonal preconditioning only rescales levels by their counts. The
factor-pair Schwarz preconditioner also captures the cross-factor links that make the
system hard to solve. The Krylov solver therefore sees a better-conditioned problem and
can reach a given tolerance in fewer iterations, especially when the fixed-effect graph
is sparse or poorly connected.

The rest of the paper proceeds as follows. We first introduce the Frisch-Waugh-Lovell
residualization problem and the AKM running example. We then describe the graph structure
of the Gramian and explain how MAP performance relates to graph connectivity. The next
sections introduce graph preconditioning and the factor-pair Schwarz construction. The
benchmark section reports runtime, memory, and numerical agreement. The final section
concludes with practical guidance.

= Fixed-Effect Residualization via the Frisch-Waugh-Lovell Theorem

The starting point for solving fixed effects problems is the Frisch-Waugh-Lovell (FWL) theorem @frisch1933
@lovell1963. For the model

$ y = X beta + D alpha + epsilon, $

FWL states that the coefficient on $X$ can be obtained without explicitly estimating all
elements of $alpha$ in the final regression. Let $P_D$ denote the weighted projection
onto the column space of $D$, and let $M_D = I - P_D$ be the corresponding
residualization operator. Then the coefficient of interest is

$ hat(beta) = (X' W M_D X)^(-1) X' W M_D y. $

Equivalently, one first residualizes $y$ and every column of $X$ with respect to the
fixed effects, and then regresses the residualized outcome on the residualized
covariates:

$ tilde(y) = M_D y, quad tilde(X) = M_D X, quad
  hat(beta) = (tilde(X)' W tilde(X))^(-1) tilde(X)' W tilde(y). $

For a model with one regressor of interest, this can be described as three regressions:
regress $y$ on the fixed effects $D$ and keep the residual $tilde(y)$; regress the
covariate $x$ on the same fixed effects and keep the residual $tilde(x)$; and then
regress $tilde(y)$ on $tilde(x)$. FWL states that the coefficient from this third
regression is exactly the coefficient on $x$ from the full regression of $y$ on $x$ and
$D$. With several covariates, the second step is repeated for each column of $X$, so the
computational burden is concentrated in repeatedly applying the fixed-effect residual
operator $M_D$.

The remaining task is therefore to apply the same fixed-effect residualization to several
right-hand sides. For any such right-hand side $mu$, either $y$ or one column of $X$, the
fixed-effect fit solves

$ hat(alpha)_mu = arg min_alpha || D alpha - mu ||_W^2, $

where $W$ is a diagonal matrix of weights. The residualized variable is then
$tilde(mu) = mu - D hat(alpha)_mu$. The first-order conditions for this auxiliary
least-squares problem are

$ D' W (D hat(alpha)_mu - mu) = 0, $

or

$ G hat(alpha)_mu = D' W mu, quad G = D' W D. $ <eq:fwl-normal>

The structure of $G$ determines how difficult this residualization step is.

= A Running Example: The AKM Model

Matched employer-employee data are a canonical setting where high-dimensional fixed
effects are both substantively important and computationally demanding. The AKM model
@akm1999 separates persistent worker heterogeneity from firm wage premia using workers
who move across firms. Consider the standard AKM wage equation:

$ y_(i t) = alpha_i + psi_(J(i,t)) + phi_t + x'_(i t) beta + epsilon_(i t), $

where $alpha_i$ is a worker fixed effect, $psi_(J(i,t))$ is the fixed effect for the
firm employing worker $i$ at time $t$, and $phi_t$ is a time fixed effect. The object of
interest may be $beta$, the firm effects, or the variance decomposition. The
computational problem is the same: remove worker, firm, and year effects from the
outcome and covariates.

Workers and firms form a bipartite graph, with workers as one set of nodes, firms as the
other, and employment spells as edges. Movers are the workers whose edges connect firms;
stayers create observations, but they do not connect one firm to another. The year fixed
effect adds a third factor which is usually low-dimensional, but still interacts with
the worker and firm factors through the same observations.

#figure(
  image(solver-img("worker_firm_connectivity.svg"), width: 80%),
  caption: [Worker-firm graph connectivity. High mobility creates many paths between
  firms. Low mobility and sorting leave the graph close to separated clusters connected
  by thin bridges.]
)

The graph representation links the identification problem to the computational problem.
If a worker is only ever observed at one firm, it is difficult to know whether a high
wage comes from the worker or the firm, whereas with many workers moving across many
firms, each worker carries information across the graph and the two effects are easier
to disentangle. The usual AKM connected set is the connected component of the full
worker-firm graph on which worker and firm effects can be compared. The factor-pair
connected components used below are different objects: they are local subproblems used
by the solver, not replacements for connected-set or leave-one-out diagnostics such as
@kss2020.

The same graph interpretation applies beyond labor economics: doctor-patient panels ask
whether outcomes are driven by the patient or the doctor, student-teacher panels ask
whether outcomes are driven by the student, classroom, teacher, or school, and trade
datasets connect exporters, importers, products, and years. In each case, the
computational difficulty depends on how these fixed-effect levels are connected.

= The Graph Structure of the Gramian

In Section 2, we argued that the central numerical object is the Gramian $G = D' W D$.
In the AKM example, this matrix is the algebraic representation of the
worker-firm-year graph @correia2017. We now take a closer look at its block structure.
Suppose the columns of $D$ are ordered as worker levels, firm levels, and year levels.
Then

$ G = mat(
  G_(W W), C_(W F), C_(W Y);
  C_(W F)', G_(F F), C_(F Y);
  C_(W Y)', C_(F Y)', G_(Y Y)
). $

The #dg[diagonal blocks] $#dg[$G_(W W)$]$, $#dg[$G_(F F)$]$, and $#dg[$G_(Y Y)$]$
contain weighted counts for workers, firms, and years. Since one observation cannot
belong to two workers, two firms, or two years at the same time, these blocks are
diagonal and can be solved cheaply by dividing by group counts.

The #cr[off-diagonal blocks] are cross-tabulations: the worker-firm block $#cr[$C_(W
F)$]$ records how often worker $i$ is observed at firm $j$, and the worker-year and
firm-year blocks have analogous interpretations. These blocks encode the cross-factor
coupling in the fixed-effect system. MAP uses this coupling only indirectly through
successive residual updates, while a diagonal LSMR preconditioner does not include 
information from these blocks in its diagonal preconditioner.

For exemplary purposes, we now build a small worker-firm panel and populate its Gramian. 
For simplicity, we set $W = I$. 

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
  image(solver-img("toy_worker_firm_projection.svg"), width: 68%),
  caption: [Worker-firm projection of the toy panel. Worker $W_1$ is a mover; workers
  $W_2$ and $W_3$ are stayers.]
)

Worker $W_1$ works at both $F_1$ and $F_2$ and creates a link between the two firms; in
AKM terms, $W_1$ is a mover. Worker $W_2$ stays at $F_1$ for two periods, and $W_3$
stays at $F_2$ for two periods. Both are stayers.

The diagonal blocks are weighted count matrices. In this unweighted example, each worker
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

The first row says that worker $W_1$ appears once at firm $F_1$ and once at firm $F_2$.
Workers $W_2$ and $W_3$ are stayers, appearing twice at $F_1$ and $F_2$, respectively.
The worker-year block is

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
cross-tabulation blocks gives the full Gramian.

With column order $(W_1, W_2, W_3, F_1, F_2, Y_1, Y_2)$, the full unweighted Gramian is

$ G = mat(augment: #(hline: (3, 5), vline: (3, 5), stroke: 0.4pt + rgb("#b0b8c4")),
  2, 0, 0, 1, 1, 1, 1;
  0, 2, 0, 2, 0, 1, 1;
  0, 0, 2, 0, 2, 1, 1;
  1, 2, 0, 3, 0, 2, 1;
  1, 0, 2, 0, 3, 1, 2;
  1, 1, 1, 2, 1, 3, 0;
  1, 1, 1, 1, 2, 0, 3
). $

The worker-firm block has a direct graph interpretation: worker $W_1$ is a mover who
connects $F_1$ to $F_2$, while workers $W_2$ and $W_3$ are stayers. Thus, in the
worker-firm part of the graph, the information separating the two firm effects travels
through $W_1$. In larger panels, sparse mobility and strong sorting create the same
structure at scale, with cheap diagonal blocks but cross-factor blocks that determine
how difficult the fixed-effects problem is.

This sparse worker-firm example is not the balanced two-way panel for which the usual
double-demeaning formula applies. In a balanced two-way panel, every worker is observed
with the same set of firms, so the worker-firm cross-tabulation has a highly regular
structure. After subtracting worker means, the remaining firm component is the same
object for every worker up to a constant; subtracting firm means then removes it exactly.
This gives the familiar closed-form transformation
$y_(i j) - overline(y)_(i dot) - overline(y)_(dot j) + overline(y)$ @baltagi2021.

In the unbalanced case, different workers appear at different firms. The cross-tabulation
block $C_(W F)$ is no longer uniform, and worker and firm effects remain coupled after a
single worker or firm demeaning step. The fixed-effect regression is still a linear
problem, but there is generally no one-pass closed-form demeaning formula. One must solve
the coupled system iteratively, as in MAP @guimaraes2010 @gaure2013 or Krylov methods.

= Alternating Projections and Graph Connectivity

The workhorse algorithm for multi-way fixed effects is the Method of Alternating
Projections (MAP), also known as iterative demeaning or the "zig-zag" algorithm
@guimaraes2010 @gaure2013. Many packages use MAP or variants of it, often with
accelerations @berge2018 @correia2017; @appendix-fixest describes the `fixest` acceleration
strategy.

MAP solves the FWL residualization problem by updating one fixed-effect dimension at a
time. In the worker-firm-year model, worker means are removed from the current residual,
then firm means are removed from the updated residual, then year means are removed, and
the sweep is repeated until convergence. The worker-firm links therefore affect the
algorithm through the residual passed from one block update to the next, rather than
through a joint worker-firm solve.

Writing the FWL normal equations @eq:fwl-normal in block form, with
$D = [D_W quad D_F quad D_Y]$, gives

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

If the firm effects $alpha_F$ and year effects $alpha_Y$ were known, the first equation
would give the worker effects $alpha_W$ from

$ G_(W W) alpha_W = D_W' W (mu - D_F alpha_F - D_Y alpha_Y). $

If the worker effects $alpha_W$ and year effects $alpha_Y$ were known, the second
equation would give the firm effects $alpha_F$ from

$ G_(F F) alpha_F = D_F' W (mu - D_W alpha_W - D_Y alpha_Y). $

The year equation is analogous. 

The core insight behind MAP is that $G_(W W)$, $G_(F F)$, and $G_(Y Y)$ are diagonal
matrices of weighted group counts. Therefore each block solve, conditional on the other
factors, is cheap: it is just a group-mean calculation. We do not need a general matrix
inverse for these blocks. If $C$ is diagonal with nonzero entries $c_j$, then solving
$C z = b$ is elementwise division,

$ z_j = b_j / c_j. $

Applied to $G_(W W)$, this divides each worker-level weighted residual sum by the
worker's weighted observation count; the firm and year updates are analogous.

MAP uses this fact iteratively, solving the worker block using the current firm and year
effects, updating the residual, and then solving the firm block and year block in turn.
One sweep applies this logic factor by factor. Equivalently, each update subtracts the
weighted group mean of the current partial residual for the active fixed-effect
dimension.

This gives MAP a standard interpretation in numerical linear algebra: it is block
Gauss-Seidel on the fixed-effect normal equations.

The cross-tabulation blocks $C_(W F)$, $C_(W Y)$, and $C_(F Y)$ enter this algorithm only
indirectly. For example, the worker update is computed from the partial residual
$mu - D_F alpha_F - D_Y alpha_Y$, while the firm update is computed from
$mu - D_W alpha_W - D_Y alpha_Y$. Thus worker-firm, worker-year, and firm-year links are
not solved as coupled subproblems; their effect is carried through the residual that one
block update passes to the next.

This is effective when the graph is well connected. In a high-mobility worker-firm panel,
many workers move across firms, so worker and firm effects are compared through many
overlapping employment histories. A high wage at one firm can be related to wages of the
same workers at other firms, and a worker update quickly changes the information seen by
the next firm update, and vice versa.

When mobility is sparse, sorting is strong, or one factor is nearly nested in another,
the same issue becomes both an identification problem and a numerical problem. If a
worker is observed only at one firm, a high wage is hard to attribute: it may reflect a
high worker effect, a high firm effect, or both. Movers provide the links that separate
these explanations. When those links are few or concentrated in narrow parts of the graph,
the cross-factor information that distinguishes worker effects from firm effects is weak.
MAP can still solve the linear system, but it accesses this information only through
repeated residual updates. It can therefore require many fast demeaning steps before the
worker and firm effects are separated.

#figure(
  image(solver-img("map_convergence.svg"), width: 72%),
  caption: [MAP converges quickly when fixed-effect subspaces are well separated and
  slowly when sparse graph structure makes them nearly collinear.]
)

Formally, the relation between two fixed effects can be expressed through the Friedrichs
angle $theta_F$ between their subspaces. For two factors, the asymptotic MAP error
contracts at a rate governed by $cos^2(theta_F)$, which equals the largest nontrivial
eigenvalue of the normalized overlap matrix $H_(W F)$ after removing the common constant
direction. This quantity measures how much nonconstant worker variation can be explained
by firm assignments, after accounting for observation counts.

When mobility is high, $cos^2(theta_F)$ is small and MAP converges quickly. Sparse
mobility pushes $cos^2(theta_F)$ toward one and makes MAP stall. With worker, firm, and
year effects there is more than one angle, but the applied interpretation is the same:
year effects are usually easy, while the hard geometry comes from the worker-firm
mobility graph.

The next sections place this view relative to existing estimators and then introduce a
preconditioner built from factor-pair subproblems that solves the worker-firm block
jointly, using the cross-factor edges directly rather than passing information through
residuals.

= Graph Preconditioning for Fixed Effects

Existing high-dimensional fixed-effect solvers address the same residualization problem
in different ways. MAP-based methods, including `reghdfe`, `lfe`, and `fixest`, make
repeated one-factor demeaning fast and add accelerations @guimaraes2010 @gaure2013
@correia2017 @berge2018. `FixedEffectModels.jl` instead uses LSMR with diagonal
preconditioning, which improves numerical scaling but does not include cross-factor
co-occurrence blocks in the preconditioner @fong2011 @fixedeffectmodels. Like
`FixedEffectModels.jl`, the approach studied here uses a Krylov solver; unlike relying
on a diagonal preconditioner, it tries to build a stronger preconditioner from
factor-pair subproblems.

The previous section showed why MAP can be slow: cross-factor information is passed from
one update to the next through residuals. When the worker-firm graph is sparse or poorly
connected, many sweeps may be needed before this information has moved far enough through
the graph. A preconditioner can address this bottleneck more directly. Instead of using
only one-factor count blocks, it can include selected cross-factor blocks and use them to
build an approximate inverse of the fixed-effect Gramian.

The challenge is to do this without forming or solving the full fixed-effect system. The
preconditioner should capture enough of the worker-firm, worker-year, and firm-year
structure to improve convergence, while remaining much cheaper than a direct solve.

== What a Preconditioner Does

Consider the linear system

$ G alpha = b. $

An iterative solver improves a guess for $alpha$ by repeatedly applying $G$ and
correcting the current residual. If $G$ is poorly conditioned, the solver may make
progress in some directions quickly and in other directions very slowly. Sparse
fixed-effect graphs have this structure because some combinations of fixed effects are
easy to separate, while others are identified only through thin graph connections.

A preconditioner is a matrix $M$ chosen so that $M^(-1)$ is cheap to apply and
$M^(-1) G$ is easier to solve with than $G$ itself:

$ M^(-1) G alpha = M^(-1) b. $

Because the fitted value is unchanged, the preconditioner only changes the geometry seen
by the iterative method. For the fixed-effect problem, a useful preconditioner should be
cheaper to construct and apply than the full system, should improve conditioning, and
should capture the cross-factor interactions that slow MAP in sparse or nearly nested
designs. Because FWL requires residualizing $y$ and every column of $X$, the setup
should also be reusable across multiple right-hand sides.

== Diagonal Preconditioning

The simplest useful preconditioner uses only the diagonal count blocks, which is the
natural scaling for Krylov methods such as LSMR, the sparse least-squares solver used by
Julia `FixedEffectModels.jl` @fong2011 @fixedeffectmodels. Diagonal scaling cannot solve
the worker-firm coupling, but it can remove a basic source of ill conditioning:
fixed-effect levels with very different observation counts.

Consider a labor market with a few very large firms and many small firms - Novo
Nordisk in Danish register data, or Samsung in South Korean matched employer-employee
panels. In the fixed-effect normal equations, a firm with tens of thousands of
worker-year observations has a diagonal entry orders of magnitude larger than a
five-person shop. A Krylov method that sees the unscaled system must make progress
across directions
measured on very different numerical scales, so dividing by the worker, firm, and year
count blocks makes these directions more comparable; diagonal preconditioning can
therefore help LSMR even though it does not use the worker-firm co-occurrence graph.

Diagonal preconditioning is therefore an important baseline, but it uses only the count
information in the diagonal blocks of the Gramian and leaves the co-occurrence blocks out
of the preconditioner. Including selected off-diagonal blocks $C_(q r)$ without forming
or solving the entire Gramian requires a preconditioner built from local factor-pair
problems that remain much smaller than the global system while containing the
co-occurrence structure between two fixed effects.

== Factor-Pair Preconditioning

The preconditioned approach starts from the same econometric problem and the same normal
equations:

$ G alpha = D' W mu. $

Rather than changing this system, it changes the computational strategy: whereas MAP
updates one factor at a time, the preconditioner builds local problems from selected
factor pairs $(q,r)$: worker-firm, worker-year, firm-year, and so on in the AKM example.
These factor-pair problems include the cross-tabulation blocks that MAP only uses
indirectly.

For a pair of factors $(q,r)$, let $N_q$ and $N_r$ denote the diagonal count blocks for
the two factors, and let $C_(q r)$ denote their cross-tabulation block. The local
factor-pair block is

$ G_(q r) = mat(
  N_q, C_(q r);
  C_(q r)', N_r
). $

In the AKM example, the worker-firm subproblem uses $N_W$, $N_F$, and $C_(W F)$. It
therefore sees exactly which workers connect which firms, and hence sees the sparse
mobility structure directly.

#figure(
  image(solver-img("factor_pair_strategy.svg"), width: 86%),
  caption: [Macro strategy of the factor-pair preconditioner. Local pair solves are built
  first, combined into a reusable Schwarz preconditioner, and then applied repeatedly
  inside the Krylov solver for the outcome and covariates.]
)

Factor-pair problems are more informative than one-factor means, but they are also more
expensive. Using them as a direct replacement for every demeaning update would defeat the
main advantage of MAP: each MAP sweep is cheap because it only solves diagonal count
blocks. The implementation studied here therefore uses factor-pair problems only to build
a preconditioner, which can be constructed once and reused across the repeated
residualization problems. The next section gives the construction.

= The Factor-Pair Schwarz Preconditioner

The preconditioner is built from local subproblems on the graph of the Gramian, following
the additive Schwarz view of subspace correction methods @xu1992 @toselli2005. The
default implementation enumerates all unordered factor pairs, builds the
cross-tabulation $C_(q r)$ for each pair, splits the induced bipartite graph into
connected components, and creates one Schwarz subdomain per component. In an AKM
worker-firm-year specification, this produces worker-firm, worker-year, and firm-year
subdomains. The worker-firm pair is often the hard mobility component, but the
construction itself includes all pairs by default.

For a subdomain $s$, let $R_s$ restrict a global vector to the fixed-effect levels in
that subdomain, and let $A_s$ be the corresponding local factor-pair operator. If a
fixed-effect level $j$ appears in $c_j$ local subdomains, assign weight
$omega_j = 1 / sqrt(c_j)$ on both restriction and prolongation. With $tilde(D)_s$
collecting these weights, the additive Schwarz preconditioner applied to a residual $r$
is

$ M^(-1) r = sum_(s=1)^m R_s' tilde(D)_s A_s^+ tilde(D)_s R_s r, $

where $A_s^+$ denotes the approximate local solve on the normalized local subspace.
Using the same weights on restriction and prolongation makes the additive preconditioner
symmetric, so it can be paired with conjugate gradient on the normalized fixed-effect
system.

The local factor-pair solve uses the Laplacian structure of the block

$ G_(q r) = mat(N_q, C; C', N_r). $

After flipping the sign of one side, this becomes

$ L_(q r) = mat(N_q, -C; -C', N_r). $

Equivalently, with $T = mat(I, 0; 0, -I)$, solving $G_(q r) z = h$ is the same as
solving $L_(q r) u = T h$ and then setting $z = T u$. The local Laplacian is singular on
each connected component, so the solve is carried out with a normalization or
pseudoinverse; the fitted correction is invariant to that choice.

The local solve then eliminates one side of the bipartite graph. If the eliminated side
is $q$, the Schur complement is

$ S = N_r - C' N_q^(-1) C. $

Exact elimination can create dense fill. In worker-firm language, eliminating a worker
who visited $d$ firms creates a clique among those firms with $binom(d,2)$ edges.
`within` uses approximate Cholesky ideas from the Laplacian solver literature to avoid
materializing these cliques @spielman2014 @gao2025. The local factors remain sparse and
cheap to apply, while the outer Krylov iteration corrects the approximation error
globally.

#figure(
  image(solver-img("local_solve_pipeline.svg"), width: 82%),
  caption: [Local solve pipeline: sign flip, Schur reduction, approximate factorization,
  and back-substitution.]
)

The preconditioner is approximate by design. Solving every factor-pair problem exactly
would make the setup too expensive, but diagonal scaling alone misses the cross-factor
edges that make sparse designs hard. The construction therefore aims for an intermediate
object: cheap enough to build and apply, but rich enough to give the Krylov solver a
better geometry. The Krylov iteration then refines the global solution to the requested
tolerance.

This is why the method is most useful when MAP sweeps carry too little information:
sparse mobility, strong sorting, near nesting, and several interacting high-dimensional
fixed effects. When the graph is dense and well connected, MAP can be difficult to
improve on because its cheap sweeps already converge quickly.

= Benchmark Evidence

The benchmarks evaluate three questions: how long the solvers take, how much memory they
use, and whether they return the same least-squares solution. The runtime benchmarks
measure end-to-end fixed-effect regression time, not only the inner linear-algebra
routine. Each runtime includes model setup, construction of the fixed-effect
representation, residualization of the outcome and covariates, and estimation of the
coefficient of interest. The results should therefore be read as software-level timings
for the same econometric problem.

The benchmark designs have two purposes. The first design pair is the standard
simple-versus-difficult fixed-effect benchmark used as a reference case for fixed-effect
software @pyfixest. The simple design has a dense, well-connected fixed-effect graph; the
difficult design has a sparse, nearly nested worker-firm structure. The second set of
benchmarks simulates AKM-style worker-firm panels and varies the graph directly: one
sweep lowers mobility, while another increases sorting among movers.

The comparison is not between `within` and a naive MAP loop. R `fixest` is a mature
MAP-based implementation with coefficient-space routines and Irons-Tuck acceleration;
@appendix-fixest summarizes those details. The compared backends are:

- PyFixest `rust-map`, a Rust MAP backend without acceleration.
- R `fixest`, a mature MAP-based implementation with Irons-Tuck acceleration
  @berge2026fixest.
- Julia `FixedEffectModels.jl`, which uses LSMR, a Krylov method with diagonal
  preconditioning @fong2011 @fixedeffectmodels.
- PyFixest `rust-cg`, the `within` conjugate gradient backend with Schwarz
  preconditioning.

All reported numbers are medians across three benchmark iterations.

== Runtime Benchmarks

The first comparison uses the simple and difficult fixed-effect designs from the PyFixest
benchmark suite, each with 10M observations, one covariate, and three fixed effects
(worker, firm, year). The simple design assigns workers to firms at random, producing a
dense, well-connected co-occurrence graph where MAP converges in few sweeps. The
difficult design assigns workers to firms sequentially, producing a sparse, nearly nested
graph where worker and firm effects are hard to separate. Each reported time covers the
full regression call: data setup, fixed-effect representation, residualization of the
outcome and covariate, and coefficient estimation.

On the simple design, where the graph is dense, `fixest` with Irons-Tuck acceleration is
the fastest backend at 0.93s. `within` is the slowest at 10.53s because the
preconditioner setup cost is not repaid when the problem is already easy for one-factor
demeaning. On the difficult design, MAP convergence degrades sharply - `rust-map` without
acceleration does not finish in under six minutes, and `fixest` takes 32.7s - while
`within` finishes in 3.25s because its factor-pair preconditioner captures the sparse
worker-firm coupling directly.

#figure(
  image(pf-fig("bench_readme.png"), width: 78%),
  caption: [Benchmark comparison for simple and difficult fixed-effect designs. The
  difficult design has a harder co-occurrence graph, not just more observations. Source:
  PyFixest benchmark outputs.]
)

#text(size: 9.2pt)[
#strong[Simple vs. difficult design (10M observations, 3 FE).]
#table(
  columns: (1.35fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [simple (dense graph)], [2.01s], [0.93s], [2.27s], [10.53s],
	  [difficult (sparse graph)], [382.9s], [32.7s], [28.7s], [3.25s],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] Medians over three full regression calls. The simple
	design has dense random mobility. The difficult design has a nearly nested worker-firm
	graph. Both use 10M observations, one covariate, and three fixed effects.]
	]

#v(0.4em)

#text(size: 9.2pt)[
#strong[Mobility sweep ($n = 1$M).]
#table(
  columns: (1.35fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Scenario], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`akm_mobility_1`], [0.29s], [0.16s], [0.28s], [0.51s],
  [`akm_mobility_2`], [1.10s], [0.36s], [0.58s], [0.38s],
  [`akm_mobility_3`], [12.48s], [1.31s], [1.80s], [0.33s],
  [`akm_mobility_4`], [51.95s], [3.42s], [2.78s], [0.34s],
  [`akm_mobility_5`], [63.23s], [4.17s], [3.34s], [0.35s],
	  [`akm_mobility_6`], [#miss], [5.27s], [3.86s], [0.35s],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
	and worker, firm, and year fixed effects. Moving down the table lowers worker mobility
	in a 10-period panel, thinning the worker-firm graph.]
	]

#v(0.4em)

#text(size: 9.2pt)[
#strong[Sorting sweep ($n = 1$M).]
#table(
  columns: (1.35fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Scenario], th[`rust-map`], th[`fixest`], th[`FEM.jl`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  [`akm_sorting_1`], [6.79s], [0.81s], [1.43s], [0.32s],
  [`akm_sorting_2`], [9.73s], [1.12s], [1.68s], [0.34s],
  [`akm_sorting_3`], [10.31s], [1.24s], [1.67s], [0.33s],
  [`akm_sorting_4`], [14.94s], [1.58s], [1.87s], [0.33s],
	  [`akm_sorting_5`], [25.77s], [1.80s], [1.94s], [0.34s],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] AKM-style panel with 1M observations, one covariate,
	and worker, firm, and year fixed effects. Moving down the table increases sorting among
	movers, making the worker-firm graph closer to block diagonal.]
	]

The mobility and sorting sweeps isolate the graph mechanism. `within` is nearly
insensitive to changes in mobility and sorting, with
runtimes ranging from 0.32s to 0.51s across all scenarios, while `fixest` ranges from
0.16s to 5.27s and `FixedEffectModels.jl` from 0.28s to 3.86s. The mechanism is that
`within`'s preconditioner already encodes the cross-factor structure directly. When
mobility falls or sorting strengthens, the preconditioner changes but the outer Krylov
iteration count does not increase substantially, because the local factor-pair solves
continue to capture the hard worker-firm coupling. MAP-based methods, by contrast,
access that coupling only through residual updates, so their iteration counts grow as
the graph becomes sparser or more block-diagonal.

Runtimes depend on hardware, software versions, and solver tolerances, but the
qualitative pattern is stable: `within` is designed for the difficult graph structures
where MAP convergence degrades.

== Memory Use

The following table reports peak RSS for each backend on the simple and difficult
designs at 100K and 1M observations, with $k = 10$ covariates and three fixed effects
(worker, firm, year). Both backends are run in isolated processes and report peak RSS via
`ru_maxrss`. Because both share the same pandas/pyarrow data-loading overhead, the
difference in peak RSS isolates the solver overhead.

#v(0.4em)

#text(size: 9.2pt)[
#strong[Memory footprint (3 FE, $k = 10$).]
#table(
  columns: (1.35fr, 0.9fr, 0.9fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(th[Design], th[`rust-map`], th[`within`]),
  table.hline(stroke: 0.45pt + table-rule),
  table.cell(colspan: 3, fill: table-head-fill)[#emph[100K observations]],
  [simple (dense graph)], [428 MB], [487 MB],
  [difficult (sparse graph)], [432 MB], [479 MB],
  table.hline(stroke: 0.35pt + table-light-rule),
  table.cell(colspan: 3, fill: table-head-fill)[#emph[1M observations]],
  [simple (dense graph)], [1,188 MB], [1,703 MB],
	  [difficult (sparse graph)], [1,263 MB], [1,398 MB],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] Peak RSS from isolated Python processes. 10 covariates,
	three fixed effects.]
	]

At 100K observations, the preconditioner adds roughly 50 MB. At 1M, the overhead is
larger in absolute terms (135--515 MB) but remains modest relative to the data footprint
of a panel with 10 covariates. The overhead comes from storing the factor-pair
subproblems, their approximate Cholesky factors, and the partition-of-unity weights, all
of which are reused across right-hand sides. On the difficult 1M design, MAP's peak RSS
closes the gap with `within` despite MAP performing no preconditioning, because the
long-running iterative solve accumulates temporary allocations over many sweeps.

== Solver Agreement

The 100K-observation simple and difficult designs are used to verify that both backends
converge to the same least-squares solution. Coefficients, residual vectors, and
firm-effect variance components are compared between `rust-map` (MAP) and `within`
(preconditioned CG), both at their default tolerance of $10^(-6)$.

#v(0.4em)

#text(size: 9.2pt)[
#strong[Solver agreement (100K observations, 3 FE, $k = 10$).]
#table(
  columns: (1.35fr, 1.6fr, 1.1fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, left, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(
    th[Design],
    th[Metric],
    th[`rust-map` vs.\ `within`],
  ),
  table.hline(stroke: 0.45pt + table-rule),
  [simple], [max |coef diff|], [$3.5 times 10^(-14)$],
  [], [rel. residual-vector diff], [$1.5 times 10^(-6)$],
  [], [$|Delta "var"(hat(psi))|$], [$2.3 times 10^(-13)$],
  table.hline(stroke: 0.35pt + table-light-rule),
  [difficult], [max |coef diff|], [$3.4 times 10^(-7)$],
  [], [rel. residual-vector diff], [$6.1 times 10^(-4)$],
	  [], [$|Delta "var"(hat(psi))|$], [$6.8 times 10^(-6)$],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] Both backends use the default tolerance $10^(-6)$.
	Entries report absolute coefficient differences, relative residual-vector differences,
	and absolute differences in the estimated firm-effect variance.]
	]

On the simple design, the two backends agree to near machine precision in coefficients
and firm-effect variance. The relative residual-vector difference of $10^(-6)$ reflects
the different convergence paths of MAP and CG, not a disagreement about the solution.

On the difficult design, the coefficient differences are small but no longer at machine
precision. Both solvers stop at a residual tolerance of $10^(-6)$, and a residual
stopping rule controls $norm(b - G alpha)_2 / norm(b)_2$, not coefficient equality
directly. An error of order $epsilon$ in the residual can translate into an error of
order $kappa epsilon$ in coefficients when the fixed-effect graph is poorly conditioned.
Low worker mobility is exactly the graph feature that increases this conditioning
penalty.

To confirm that the difficult-design gap comes from the stopping rule rather than from
a disagreement about the least-squares target, the following table pins `within` at
$10^(-12)$ as a reference solve and then varies both MAP and `within` across a tolerance
ladder. With the iteration cap raised to 100,000, MAP now converges at all three
tolerances.

#v(0.35em)

#text(size: 9.2pt)[
#strong[Tolerance scaling, difficult design (100K observations).]
#table(
  columns: (1.05fr, 0.75fr, 1.15fr, 0.8fr),
  stroke: 0.35pt + table-light-rule,
  inset: (x: 5pt, y: 3.6pt),
  align: (left, right, right, right),
  table.hline(stroke: 0.8pt + table-rule),
  table.header(
    th[Method],
    th[Tol.],
    th[Max |coef diff| vs. $10^(-12)$ ref.],
    th[Time],
  ),
  table.hline(stroke: 0.45pt + table-rule),
  [MAP], [$10^(-6)$], [$3.7 times 10^(-7)$], [2.0s],
  [MAP], [$10^(-8)$], [$4.0 times 10^(-11)$], [7.4s],
  [MAP], [$10^(-10)$], [$4.0 times 10^(-15)$], [12.6s],
  table.hline(stroke: 0.35pt + table-light-rule),
  [`within`], [$10^(-6)$], [$4.4 times 10^(-14)$], [0.2s],
  [`within`], [$10^(-8)$], [$4.4 times 10^(-16)$], [0.2s],
	  [`within`], [$10^(-10)$], [$3.1 times 10^(-15)$], [0.3s],
	  table.hline(stroke: 0.8pt + table-rule),
	)
	#v(0.25em)
	#text(size: 8.2pt)[#emph[Note:] Differences are measured against a `within` solve at
	tolerance $10^(-12)$. MAP iteration cap raised to 100,000.]
	]

Both solvers converge to the same target. MAP tightens predictably - each two orders of
magnitude in tolerance roughly halves the coefficient gap - but takes substantially
longer on the difficult graph: 12.6s at $10^(-10)$ versus 0.3s for `within`. The
preconditioned solver reaches near-reference accuracy already at $10^(-6)$, so the
practical coefficient differences in the agreement table are far smaller than standard
errors.

= Conclusion

High-dimensional fixed effects are often presented as an econometric device for
absorbing many categorical controls while estimating a lower-dimensional coefficient of
interest. Computationally, the same specification defines a graph, and the connectivity
of that graph affects how quickly information moves between factors during
residualization.

MAP solves this problem with repeated one-factor demeaning. Its steps are cheap and
effective, so when the fixed-effect graph is dense and well connected, MAP is the
natural default: the setup cost of a graph preconditioner may not be repaid.

Graph preconditioning pays off when the design has a hard cross-factor pair,
which in AKM applications is usually the worker-firm pair. Low mobility, strong sorting,
thin bridges between groups of firms, or near nesting all make it harder for one-factor
demeaning to transmit information across the graph. Similar patterns arise in
doctor-patient, student-teacher, exporter-importer, and product-market designs when the
identifying moves are sparse or concentrated within groups.

`within` uses the same econometric target but a different numerical strategy, building a
preconditioner from factor-pair subproblems so that local corrections use the
cross-factor graph directly. Each iteration is therefore more expensive, but reduces the
residual faster on difficult graphs.

The relevant diagnostics are therefore not only the number of observations and the
number of fixed-effect levels. Applied researchers should also inspect the factor-pair
graph: the number and size of connected components, the share of singleton or nearly
nested levels, mobility rates, and whether movers connect distant parts of the graph or
mostly remain inside sorted clusters. These diagnostics are already familiar from
identification discussions in mover designs; the point here is that they are also
computational diagnostics.

In practice, no solver dominates in every design: alternating projections are hard to
beat on easy graphs, while a factor-pair preconditioner is most useful when the
econometric design asks a sparse graph to separate effects that are close to collinear.
Solver choice should be guided by the connectivity of the fixed-effect graph, not only
by sample size and the number of fixed-effect levels.

#pagebreak()

= Appendix A: `fixest` MAP Accelerations <appendix-fixest>

The `fixest` benchmark is not a naive alternating-projections baseline, because its
demeaning algorithm includes several improvements that are important for interpreting
comparisons with `within`.

First, `fixest` uses specialized demeaning paths for different fixed-effect structures.
With one fixed effect, residualization is a single group-mean subtraction; with two
fixed effects, the problem can be run in coefficient space, so instead of repeatedly
updating an observation-length vector, the algorithm updates the two fixed-effect
coefficient vectors and reconstructs the residualized variable at the end. Although the
iteration is still Gauss-Seidel on the fixed-effect normal equations, this
representation reduces memory traffic when the number of observations is much larger
than the number of fixed-effect levels.

Second, `fixest` accelerates the fixed-point iteration. Irons-Tuck acceleration uses
successive iterates to estimate how fast the fixed-point updates are shrinking, then
extrapolates toward the apparent limit @irons1969. In favorable cases this avoids many
small MAP steps near convergence.

#figure(
  image(solver-img("irons_tuck_acceleration.svg"), width: 76%),
  caption: [Irons-Tuck acceleration extrapolates from recent fixed-point iterates rather
  than waiting for plain MAP steps to close the remaining distance.]
)

Because this extrapolation is a numerical heuristic, an accelerated step can overshoot
when the local convergence pattern is irregular. `fixest` therefore also uses
larger-interval smoothing, often described as grand acceleration, to stabilize the
acceleration signal.

Third, for models with three or more fixed effects, `fixest` uses a hybrid strategy. It
starts from the general multi-FE projection, but then focuses work on the two largest
fixed-effect dimensions, where the fast two-FE coefficient-space routine is often most
valuable, before returning to the full multi-FE problem. In a worker-firm-year AKM
specification, this usually means that most of the hard work is concentrated on the
worker-firm part, while the year effect is handled inside the full projection. The
heuristic is practical rather than theoretical: if the two largest dimensions are not
the numerically difficult pair, the benefit can be smaller.

These optimizations make `fixest` the relevant MAP benchmark. `within` takes a different
route by building a preconditioner from local factor-pair systems that contain
cross-factor edges directly, rather than by accelerating one-factor updates.

= Appendix B: Reproducibility and Submission Notes

The synthetic benchmark discussion is based on the PyFixest benchmark suite, available
in the PyFixest repository:

`https://github.com/py-econometrics/pyfixest`

The benchmark scripts are stored in:

`benchmarks/modular/`

The benchmark outputs used here are the checked-in CSV results from:

`benchmarks/results/`

The simple-versus-difficult 10M table uses the base benchmark result files
`feols_bench__*.csv`, filtered to `source_k = 10`, `model_k = 1`, `n_obs = 10000000`,
and `n_fe = 3`. For convenience, the four AKM sweep CSV files used to construct the
mobility and sorting tables are also copied into this paper repository under:

`data/benchmarks/`

The main benchmark commands documented by PyFixest are:

- `pixi r benchmark` for the base fixed-effect benchmarks;
- `pixi r benchmark-akm` for the AKM scale, mobility, sorting, interaction, and freezing
  sweeps;
- `pixi run benchmark-akm-occupation` for the occupation benchmarks.

The draft uses the base simple/difficult benchmarks and the AKM mobility and sorting
sweeps.

#strong[Code and data availability.] The solver implementation, benchmark scripts, and
checked-in benchmark outputs are available in the repositories and paths listed above.
The synthetic benchmark data are generated by the benchmark scripts rather than drawn
from confidential administrative records.

#bibliography("refs.bib", style: "chicago-author-date", title: [References])

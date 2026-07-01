# Mathematical-correctness audit

Equation-by-equation probe of `graph_preconditioner_hdfe.typ`. Every display equation and every embedded mathematical claim was checked by hand; the load-bearing ones were also verified in NumPy (toy Gramian, normalized cross-tab spectrum, Laplacian/PSD reconstruction, Schur complement, rank of `G`, and a direct simulation of two-factor MAP). Line numbers refer to the `.typ` source.

**Verdict.** Every equation in the paper is mathematically correct — I re-derived each one and re-ran the numerics, and no formula is wrong. There are **two things to fix in the prose** (one sentence whose literal reading is false; one notation clash) and **one I'd recommend adding** (a sentence acknowledging that `G` is singular). Everything else is either exact or an acceptable abbreviation for a Stata Journal audience — those are listed explicitly so you don't over-correct.

---

## 1. Fix these

### 1.1 L499 — "this is the MAP contraction rate" names the wrong quantity

> "…a larger gap indicates a better connected pair. **With exactly two absorbed factors this is the MAP contraction rate.**"

The grammatical antecedent of "this" is *the gap* (`1 − ρ_qr`). But the contraction rate is `ρ_qr`, **not** the gap. I simulated two-factor MAP on the Section-4 toy panel: the residual contracts by a factor of **exactly 0.66667 = ρ_WF per cycle**, while the gap is `1/3`. Analytically, the second-largest eigenvalue of `P_W P_F` is `0.666667` — that eigenvalue *is* the per-cycle rate. So "the gap is the contraction rate" is false as written.

I assume you meant `ρ_qr`. Suggested rewrite:

> "…a larger gap indicates a better connected pair. With exactly two absorbed factors, `ρ_qr` is precisely the per-cycle contraction rate of MAP, so the gap `1 − ρ_qr` is the fraction of the error removed each sweep."

**Verdict: fix — wording.** Important nuance: this is a phrasing problem, *not* a wrong formula. `ρ_qr` is defined and used correctly everywhere else (L716, L805, every table). It is only this one sentence that, read with "the gap" as the antecedent of "this," asserts something false. Renaming `ρ_qr` fixes it.

### 1.2 L456–457 — symbol clash: `C` reused for a diagonal matrix

> "If `C` is diagonal with nonzero entries `c_j`, then solving `C z = b` reduces to elementwise division, `z_j = b_j / c_j`."

The statement is trivially true, but `C` denotes the **cross-tabulation block everywhere else in the paper**, and that block is precisely the *non-diagonal* part. Using `C` for a generic diagonal matrix here reads as a contradiction. Tie it to the actual object:

> "Because each diagonal block is itself diagonal — write `G_WW = diag(c_j)` — solving `G_WW z = b` is elementwise division, `z_j = b_j / c_j`."

**Verdict: fix — notation.** Trivial, but a careful reader will trip on it.

---

## 2. Recommend adding

### 2.1 `G = D'WD` is singular — say so once

`G` is rank-deficient. On the paper's actual three-way toy Gramian (the 7×7 matrix at L361–369) I get `rank(G) = 5` of `7` — **null-space dimension 2** (verified in NumPy; the two null directions are worker-vs-firm and worker-vs-year shifts). A caution on counting: the deficiency is **not** "one per connected component" — that rule holds only for *two* factors. For `Q` mutually connected factors there are `Q − 1` additive normalizations, plus extra directions if the design breaks into disconnected pieces. (An earlier draft of this note quoted `rank 4/5`; that was the two-factor worker–firm reduction used for the MAP simulation, not the paper's three-way toy.) Three consequences are currently left implicit:

- `(D'WD)^{-1}` inside `P_D` / `M_D` (L172, L176) and the solve `G α̂_μ = D'W μ` (L208) involve a **generalized** inverse, not a true inverse.
- `α̂_μ` is **not unique** — it is pinned down only up to the null space. (The system is still *consistent*: I verified `‖G α̂ − D'Wμ‖ = 0`, and that `D · nullvector = 0`, so the fit is unchanged by the choice.)
- The residualized variable `μ̃ = μ − D α̂_μ` (equivalently `P_D μ`) **is** unique — which is all FWL needs.

This matters beyond pedantry: the singularity you're glossing over in Section 2 is *exactly* the Laplacian null space (constants per component) that Sections 6–7 and the Appendix lean on (zero-mean pseudoinverse, `A_s^+`, projecting off the component constant). A one-line footnote at L208 connects the two:

> "`G` is rank-deficient: the fixed effects are identified only up to additive normalizations — adding a constant to every worker effect and subtracting it from every firm effect leaves `Dα` unchanged. So `α̂_μ` is pinned down only up to this null space, while the residual `μ̃` is unique, and the inverses above are generalized inverses. This is the same null space the local Laplacian solves exploit in Sections 6–7."

Note `(X'W M_D X)^{-1}` (L176) is genuinely invertible under the usual rank condition on the residualized regressors, so that one needs no caveat.

**Verdict: recommend.** Commonly elided in applied papers, but here it unifies the narrative, so worth one sentence.

---

## 3. Fine for the audience — leave as is (listed so you don't over-edit)

- **L493–494 vs L716–718 — definition of `ρ`.** The precise definition `ρ_qr = σ₂(H_qr)²` appears at L716; at L493 it's "the largest nontrivial singular value." Also "singular value of the factor-pair graph" — singular values belong to the matrix `H_qr`, not the graph. Harmless, but if you want them to match, state `ρ = σ₂²` at first mention too. *Optional.*
- **L506 footnote — "unit singular value."** The footnote calls `2/3` an eigenvalue of `H'H` and then writes "after dropping the unit **singular value**." Mixing eigenvalue-of-`H'H` with singular-value-of-`H` language. The numbers are exact (`σ = 1, √(2/3)`; `ρ = 2/3`; gap `1/3`, all verified). Could tighten to "trivial unit value." *Optional.*
- **`M^{-1}` where `M` is singular (L518, L521, L634).** You correctly state `M` is symmetric **positive semi-definite** (L642), so strictly `M^{-1}` is a pseudoinverse on the range. Standard abuse in the preconditioning literature; fine. *No change.*
- **"partition-of-unity weight" `ω_j = 1/√c_j` (L629).** Strictly this is a *symmetric scaling whose squares sum to 1* (`Σ_s ω² = 1`), not a literal partition of unity (which sums to 1). Standard terminology in additive Schwarz; verified it yields a symmetric PSD `M`. *No change.*
- **`M_D` under weights (L172–176).** With `W ≠ I`, `M_D = I − D(D'WD)^{-1}D'W` is the `W`-orthogonal residual maker and is **not symmetric** — but the FWL formula is still exactly right: I verified `W M_D` is symmetric and `X'W M_D X = X̃'W X̃`, so L176 and L182–183 agree. *No change.*

---

## 4. Verified correct (the ledger)

Each of these was checked and is exact — no action needed:

- **FWL:** weighted estimator (L176) and the residualize-then-regress form (L182–183) are algebraically identical (verified `X'W M_D X = X̃'W X̃` and the matching cross-term).
- **Auxiliary LS → normal equations:** `α̂_μ = argmin ‖Dα−μ‖²_W` (L198), FOC `D'W(Dα̂−μ)=0` (L204), `G α̂ = D'Wμ` (L208) — correct chain.
- **Gramian block structure** (L273–277): off-diagonal `(2,1)=C_WF'`, etc. — correct.
- **Toy panel:** `G_WW, G_FF, G_YY` (L326), `C_WF` (L332), `C_WY` (L342), `C_FY` (L350), and the **full 7×7 Gramian** (L361–369) reproduce exactly from the data.
- **Laplacian `L_WF`** (L385–391): symmetric, off-diagonals ≤ 0, **row sums = 0** (verified); and the PSD footnote (L394) — I reconstructed `L_WF` exactly from `Σ C_WF[i,f](e_i−e_f)(e_i−e_f)'`.
- **Block MAP system** (L425–430) and the worker/firm updates `G_WW α_W = D_W'W(μ − D_F α_F − D_Y α_Y)` (L444, L449) — correct factoring via `C_WF = D_W'W D_F`.
- **MAP = block Gauss–Seidel** (L466): correct.
- **Spectral gap toy footnote** (L501–506): `H_WF = (1/√6)[[1,1],[2,0],[0,2]]`, `H'H = (1/6)[[5,1],[1,5]]`, eigenvalues `1, 2/3`, gap `1/3` — all exact. Also confirmed `σ₁(H)=1` is structural (constant direction) and `σ₂² = ρ = cos²θ_F`.
- **`ρ = cos²θ_F`** (L805) — Friedrichs-angle interpretation, correct.
- **Factor-pair block** `G_qr` (L611) and **additive Schwarz operator** `M^{-1} = Σ R_s' D̃_s A_s^+ D̃_s R_s` (L634): verified symmetric PSD.
- **Appendix:** sign-flip similarity `L_qr = T G_qr T`, `T² = I`, equivalence `G_qr z = h ⇔ L_qr u = Th, z = Tu` (L1346–1348) — correct. Row-sum identity (L1344) — correct. Schur complement `S = N_r − C'N_q^{-1}C` (L1363), sign-flip invariance (L1366) — correct, and I verified `S` is itself a valid Laplacian (row sums 0). Clique `binom(d,2)` → spanning-tree `d−1` edges, unbiased reduced Laplacian (L1369–1374) — correct characterization. **Gremban augmentation** (L1352–1357): edge weight = row-sum surplus restores row sum 0 (the diagonal already carries the surplus) — correct.
- **Diagonal preconditioner = diag(G) = Jacobi** (L536) — correct.
- **Balanced two-way ⇒ double-demeaning** (L398–401) — correct.
- **Preconditioning changes the path, not the fixed point** (L592–596) and **stale preconditioner doesn't bias IRLS** (L1048–1058) — both correct and important.

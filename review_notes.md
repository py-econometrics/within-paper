# Review: *Graph-Preconditioned Estimation of High-Dimensional Fixed-Effect Models*

Reviewed against the Typst source (`graph_preconditioner_hdfe.typ`). Line numbers refer to that file.

**Overall.** The paper is in good shape. The argument is well-staged (graph connectivity → identification → computation → preconditioner → benchmarks), the toy worked example is correct and pedagogically strong, and the Novo Nordisk / Samsung scaling example is exactly the kind of concrete intuition a Stata Journal audience wants. The math I checked is correct. The main thing standing between the current draft and the target style is a handful of sentences that read like numerical-linear-algebra notes rather than prose, plus the usual typos. Priorities below are ordered.

---

## 1. Mathematical correctness

I re-derived the load-bearing math (by hand and in NumPy). **It all checks out:**

- Weighted FWL (L176, L182–183) is internally consistent: `X'W M_D X = (M_D X)'W(M_D X)` since `W M_D = M_D' W`.
- The toy Gramian (L361–369) reproduces exactly from the panel.
- Laplacian `L_WF` (L385–391) has all row sums zero ✓.
- Spectral gap (footnote L501–506): `H'H` eigenvalues are exactly 1 and 2/3, so `rho_WF = 2/3`, gap = 1/3 ✓.
- Schur complement `S = N_r - C' N_q^{-1} C` (L1363) is correct, and on the toy panel it is *itself* a valid Laplacian (row sums zero) — consistent with the appendix's claim that the reduced system stays SDD/Laplacian.
- Similarity transform `L = T G T`, `T = diag(I,-I)` (L1346–1348) and the Schwarz partition-of-unity scaling (`omega_j = 1/sqrt(c_j)` on both sides → squared weights sum to 1) are both right.

Only small issues, none affecting results:

- **L456–457 — notation collision.** You write "If `C` is diagonal with nonzero entries `c_j`, then solving `C z = b` ...", but `C` is the cross-tabulation block everywhere else (and cross-tab blocks are *not* diagonal). Use a neutral symbol or the actual block: *"Because each diagonal block `G_WW` is diagonal, solving `G_WW z = b` is just elementwise division, `z_j = b_j / [G_WW]_jj`."*

- **L493–494 — loose statement of the diagnostic.** "we compute a spectral gap `1 - rho_qr` ... using the largest nontrivial singular value of the factor-pair graph" understates the definition you actually use later (L718: `rho_qr = sigma_2(H_qr)^2`). State it the same way in both places, and note the singular values are of `H_qr`, not "of the graph": *"...where `rho_qr = sigma_2(H_qr)^2` is the squared second-largest singular value of `H_qr` (the largest, equal to 1, is the trivial constant direction)."*

- **Footnote L505–506 — eigenvalue vs. singular value.** "`H'H` has eigenvalues 1 and 2/3. After dropping the unit singular value..." mixes the two terms in adjacent sentences. The numbers are right; just say "After dropping the trivial unit value, `rho_WF = 2/3`."

- **Data sanity check (not an error), Tables L1144–1149.** The *simple*-design gap is reported as exactly `8.57×10⁻¹` at 100K, 1M, and 10M, while the *difficult* gap moves a lot (`1.30×10⁻³` at 100K vs `1.67×10⁻⁵` at 1M/10M). Worth confirming the simple value was recomputed per size rather than carried over — three identical sig-figs across two orders of magnitude in `n` is plausible for a dense graph but easy to mis-copy.

---

## 2. Prose and jargon (before → after)

These are the sentences that read as too mechanical/dense for the target style. The flagged example is first.

**L210–211 — the phrase you flagged.**
> *Before:* "Equation (8) is the linear system that pins down every residualization: whether of `y` or of one column of `X`, the fit reduces to a system in `G` with a new right-hand side."

> *After:* "Equation (8) is the system we solve at every demeaning step. Residualizing the outcome or any covariate means solving this *same* system in `G`; only the right-hand side changes."

**L516–525 — the densest sentence in the paper.** Three ideas, plus "bidiagonalization in the `M`-weighted inner product" and "clusters the eigenvalues," all in one breath. Split it and lead with intuition; demote the LSMR mechanics.
> *Before:* "Preconditioning is the standard remedy: rather than alter the least-squares fit, one supplies the Krylov iteration with a symmetric positive-(semi)definite operator `M ≈ G` and applies `M⁻¹` each iteration, so that the iteration behaves as if applied to the better-conditioned operator `M⁻¹G` in place of `G` - for LSMR, by running the bidiagonalization in the `M`-weighted inner product, which clusters the eigenvalues of `M⁻¹G`. The fitted values are unchanged; only the geometry presented to the iteration is altered."

> *After:* "Preconditioning is the standard remedy. The idea is to find a matrix `M` that approximates `G` but is far cheaper to invert, and let the solver work with `M⁻¹G` in place of `G`. When `M` captures the troublesome part of `G`, the product `M⁻¹G` is close to the identity, and the iteration converges in far fewer steps. Preconditioning changes only the *path* the solver takes, never its destination: the fitted values are identical. (For LSMR this substitution is carried out by running its bidiagonalization in the `M`-weighted inner product; readers uninterested in the mechanics can skip this.)"

**L97 — broken grammar.**
> *Before:* "The standard computational starting point for such efficiently estimating fixed effect regressions is the Frisch-Waugh-Lovell (FWL) theorem"

> *After:* "The standard computational starting point for estimating these regressions efficiently is the Frisch–Waugh–Lovell (FWL) theorem"

**L92–93 — "mover designs" appears twice in one clause.**
> *Before:* "health economists study physician practice styles via mover designs with individual physician and region fixed effects in movers designs"

> *After:* "health economists study physician practice styles with individual-physician and region fixed effects in mover designs"

**L213 — "divide along a basic line."**
> *Before:* "The solvers we compare divide along a basic line."

> *After:* "The solvers we compare split into two families."

**L219 — "key axis of variation."**
> *Before:* "but for our purposes the key axis of variation is the extent to which each one exploits the structure of `G`."

> *After:* "but for our purposes the key difference is how much each one exploits the structure of `G`."

**L223–224 — grand filler.**
> *Before:* "A careful analysis of the structure of `G` is therefore foundational for the remainder of this paper, and we turn to it in the next section."

> *After:* "Understanding the structure of `G` is therefore essential, and we turn to it next."

**L196 — `X_i` is undefined and "individual covariate" is clunky.**
> *Before:* "For any such right-hand side `μ`, whether dependent `y` or one individual covariate of `X_i`"

> *After:* "For any such right-hand side `μ` — whether the outcome `y` or a single column of `X` —"

**L1314 — non-idiomatic.**
> *Before:* "We recommend to switch to a preconditioned Krylov solver such as `within` if:"

> *After:* "We recommend switching to a preconditioned Krylov solver such as `within` if:"

**L1217–1218 — "implementations all implement."**
> *Before:* "This is not surprising, as the different software implementations all implement different stopping rules."

> *After:* "This is not surprising: the packages use different stopping rules."

**L693 — "via MAP" is misplaced.**
> *Before:* "...causes factor-by-factor demeaning via MAP to propagate information slowly."

> *After:* "...causes MAP's factor-by-factor demeaning to propagate information slowly."

---

## 3. Intuition and flow

- **Define "preconditioner" in plain words at first use (intro, ~L130).** The term carries the whole intro (L127–139) but isn't glossed until Section 6. One sentence early pays off for an applied audience: *"A preconditioner is a cheap stand-in for the system being solved; supplied to an iterative solver, it sharply cuts the number of iterations needed without changing the answer."* Same for "Krylov solver" at first use (~L135) — a three-word gloss ("an iterative linear-system solver") removes a barrier.

- **Section 2's closing paragraph (L213–224) front-loads the MAP-vs-Krylov dichotomy** with "block Gauss-Seidel," "Krylov least-squares," and "first-order condition" — before the reader has met MAP (Section 5). Two options: (a) trim it to a light one- or two-sentence preview and let Sections 5–6 carry the detail, or (b) move the detailed contrast to the top of Section 6. I'd lean (a); it keeps Section 2 about FWL.

- **Abstract (L69–72)** ends jargon-dense for the Stata Journal: "reusable additive Schwarz preconditioner that exploits this structure through local factor-pair subproblems." Optional softening: *"...a reusable preconditioner built from small, local worker-firm-type subproblems that use the graph directly."* Keep the technical name in the body.

- The spectral-gap diagnostic (L492–509) is well-motivated; L497–499 already gives the plain-English read. Good as is once the definition wording in §1 above is tightened.

---

## 4. Typos and consistency (quick fixes)

- **L59 — "chanonical" → "canonical."**
- **Spelling consistency:** "residualisation" (L192) vs. "residualization" (everywhere else). Standardize to `-z-`.
- **Hyphenation:** "fixed effect regressions" (L90, L97) → "fixed-effect regressions" (the adjective is hyphenated throughout the rest of the paper).
- **L605 — double space:** "For a generic  pair".
- **L619–621 — mild redundancy:** "...yields worker-firm, worker-year, and firm-year subdomains; usually, the worker-firm pair is the bottleneck, but the construction creates pairs for all subdomains." → "...; the worker-firm pair is usually the bottleneck, but the construction covers all three."
- **Dashes:** the draft uses spaced hyphens ` - ` for parentheticals (e.g., L135, L523, L750). Journals expect en/em-dashes; a global pass would tidy this.

---

### Bottom line
No mathematical corrections needed beyond the notation cleanup in §1. The substantive lift is rewriting the ~10 sentences in §2 (above all L516–525 and the flagged L210–211) and adding two plain-language glosses in §3. Everything else is fast cleanup.

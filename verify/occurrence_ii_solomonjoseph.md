# Independent review of Occurrence Theory II — The Born Channel

**Paper:** Occurrence Theory II, v0.7 (11 July 2026) · PR #17 · branch `born-channel` @ `ca9248f`
**Reviewer handle:** `solomonjoseph`
**Script:** [`verify/occurrence_ii_solomonjoseph.py`](occurrence_ii_solomonjoseph.py) — `python3 verify/occurrence_ii_solomonjoseph.py` → exit `0`, 83 checks, ~7 s
**Dependencies:** `numpy`; `python-flint` optional but enables the exact-over-ℚ layer. Does not import `topographo`, and does not trust `data/kraus84.npz`.

## Bottom line

Every **[FORCED]** claim in §2–7 reproduces. Most of them I verified *exactly over ℚ* rather than to machine precision, including the whole of Theorem 3.2 — both sectors, values and multiplicities — from an integer characteristic polynomial, without SageMath. The C3 representation-theory content (𝔭-sector `7 ⊕ 7`, ±3/7 sectors `7 ⊕ 14`, and both SU(3) branchings) is confirmed by an independent Casimir computation on a from-scratch `Der(𝕆)`.

Three things are not in the same shape:

1. **The [MEASURED] tier does not reproduce.** Neither `s*` nor `λ_q` matches the values in §4.3 / Appendix B.6 — and the repo's own `occurrence_ii_audit.py` does not reproduce them either. `λ_q` is not a well-posed five-digit quantity as defined.
2. **The channel is much smaller than the paper says it is.** Its Choi rank is exactly **14**, not "≤ 84", and Φ is exactly the uniform average of conjugation by the 14 unit pencil directions. This is good news for the paper's firewall thesis and bad news for any reading that gives the number 84 channel-level significance.
3. **Open Problem 1 appears to be solved,** and is listed as open on the strength of an arithmetic slip: `1/8 + 1/147 = 0.1318027…`, not `0.131723` as printed. My best estimate of `s*` agrees with the candidate.

Ten specific corrections are listed at the end; three are substantive (so(7), the multiplicity claim, Theorem 1.1's scope), the rest editorial.

## Method: what makes this review independent

The sedenions are rebuilt by iterated Cayley–Dickson doubling ℝ → ℂ → ℍ → 𝕆 → 𝕊 with integer structure constants, and the script *checks it got the right tower*: dimension 8 is normed and alternative, dimension 16 is neither. The 84 are then found by exhaustively ranking all 98 basic diagonals, and everything downstream uses my own family. `data/kraus84.npz` is used only for the cross-check in Part 1.

Because each Kraus operator is (1/√2) × an integer matrix, `168·Φ` is a 256×256 **integer** matrix. That is the lever this review leans on: characteristic polynomials, sector restrictions, ranks, nullities and the complex structure J are all computed in exact arithmetic, so the following are settled over ℚ rather than to a tolerance:

| Claim | Paper | This review |
|---|---|---|
| Spectrum + multiplicities (Thm 3.2) | 4.9·10⁻¹⁵ | `charpoly(168Φ) = x¹⁰⁰(x∓168)(x∓72)²¹(x∓24)⁴²(x²−6912)¹⁴` exactly |
| Per-sector multiplicities | numerical | exact charpoly of the integral sym (136) and antisym (120) restrictions |
| 𝔭 = 2√3/7, antisym only (Thm 3.4a) | "fraction 1.000000" | `(x²−6912)` divides the antisym factor and not the sym one |
| CPTP / unital (Thm 3.1) | 1.0·10⁻¹⁴ | `Σ K̃ᵀK̃ = 168·I` over ℤ |
| E[zzᵀ] = P_W/14 (OT 3.2) | 2.4·10⁻¹⁸ | `168·E = 12·P_W` over ℤ |
| J² = −I, J simple, J = L_{e₈} (Thm 3.3) | 5.0·10⁻¹⁵ | J is an integral signed permutation; `J² = −I` and `dim ker(Φ+I) = 1` over ℤ |
| Φ singular, dim ker = 100 (Rmk 4.4) | — | exact rank over ℤ |
| Lattice: 4-regular, 7×12, diam 3, Fano (Thm 6.1) | float tolerance | integer adjacency `K̃ₐ z̃_b = 0` |
| Var[τ] = 1/18 (Thm 4.2) | Monte Carlo 0.0557 | exact in `Fraction` |
| Tr Φ = 0 | — | exact |

Peripheral convergence (Thm 5.1), the Born identity (Thm 4.3), G₂-equivariance and the Casimir decomposition remain floating-point, at 10⁻¹²–10⁻¹⁶.

## Main mathematical finding: Φ has Choi rank 14 and is mixed-unitary

Since `z ↦ L_z` is linear, `L_zᵀ X L_z` is quadratic in `z`, so for **any** sampling measure

    Φ_ν(X) = Σ_{i,j} Σ_ij L_{e_i}ᵀ X L_{e_j},   Σ = E_ν[z zᵀ].

With `Σ = P_W/14` this collapses to

    **Φ(X) = (1/14) Σ_{i ∈ {1…7, 9…15}} L_{e_i}ᵀ X L_{e_i}**

and each `L_{e_i}` is an *integral signed permutation* with `L_{e_i}ᵀL_{e_i} = I` and `L_{e_i}² = −I`. Verified exactly over ℤ: `168·Φ = 12·Σ_{i∈W} L_{e_i} ⊗ L_{e_i}`. The exact rank of `span{K_a}` is 14, so this is the *minimal* Kraus representation and the 84-operator presentation is six-fold redundant.

Consequences I'd urge the authors to fold in, because most of them strengthen the paper:

- Φ is a **mixed-unitary (random-orthogonal) channel**. Theorem 3.1 (CPTP/unital) is then immediate — no computation, no 10⁻¹⁴. Note the tension worth addressing head-on: the individual settlement operators are singular (rank 12), yet the channel they generate is an average of *invertible* orthogonal conjugations. So the channel's irreversibility is classical mixing over fourteen unitaries, not annihilation. That reframes §5's thesis ("unitarity is what dissipation looks like after conditioning on what remains"): here the dissipation is already a mixture of unitaries, so what §5 exhibits is the commutant of a fourteen-element set of orthogonal maps rather than an emergence mechanism. C4's Lindblad question can be posed in those terms too.
- §7's "Choi rank ≤ 84" is really "Choi rank = 14". For the quantum-information audience this is the sharper and more interesting statement.
- G₂-equivariance of the *discrete* channel is a one-liner: the channel is an average over the G₂-invariant subspace W, so it inherits the continuum symmetry. `occurrence_ii_reptheory.sage` establishes this numerically at 10⁻¹⁶; it doesn't need to.
- The clock is exactly the imaginary direction the average *omits*: `W` misses `e₈`, and `J = L_{e₈}`. "The surviving oscillation is conjugation by the one direction the events never sample" is a sharper joint statement of Theorems 3.3 and 5.1 than either makes separately.
- But: **no channel-level result can be evidence for the significance of 84.** The 84 is [FORCED] as a fact about the crack Σ (Theorem 2.1); the channel only ever remembers `P_W/14`. The paper's thesis that `data/kraus84.npz` is "the only load-bearing artifact" is true but overstated by a factor of six — the load-bearing object is fourteen signed permutation matrices.
- On 𝔭: since `168Φ` is integral, `14𝔭 = 4√3 = √48` is one quadratic factor of an integer characteristic polynomial. An integer matrix having a quadratic irrationality is unremarkable; the striking fact is that *everything else is rational*. Theorem 3.4(c)'s "three algebraically independent invariants whose coincidence is forced" is a reading — nothing in the repo tests it, and `2√3 · (1/7)` is one factorisation among several.

## Requested item 1 — representation theory / C3

Rebuilt `𝔤₂ = Der(𝕆)` by solving the derivation equations (dim 14, every derivation kills `e₀` and is antisymmetric), lifted it to 𝕊 as `diag(D, D)`, confirmed `Φ([δ,X]) = [δ,Φ(X)]` to 1.1·10⁻¹⁵, then decomposed every eigenspace with the Casimir operator, calibrated on `7 ⊗ 7 = 1 ⊕ 7 ⊕ 14 ⊕ 27` (Casimir eigenvalues 0, 2, 4, 14/3) and `7 ⊗ 14 = 7 ⊕ 27 ⊕ 64`.

| eigenvalue | sector | dim | G₂ content |
|---|---|---|---|
| +1 | sym | 1 | **1** |
| +𝔭 | antisym | 14 | **7 ⊕ 7** |
| +3/7 | sym / antisym | 7 / 14 | **7** / **14** |
| +1/7 | antisym | 42 | **1 ⊕ 7 ⊕ 7 ⊕ 27** |
| 0 | sym / antisym | 72 / 28 | **4·1 ⊕ 2·7 ⊕ 2·27** / **14 ⊕ 14** |
| −1/7 | sym | 42 | **1 ⊕ 7 ⊕ 7 ⊕ 27** |
| −3/7 | sym / antisym | 14 / 7 | **14** / **7** |
| −𝔭 | antisym | 14 | **7 ⊕ 7** |
| −1 | antisym | 1 | **1** |

The paper's corrected claims hold: the 𝔭-sector is `7 ⊕ 7` with **no adjoint content** (Casimir eigenvalue 2 across all 14 dimensions, no Casimir-4 subspace), each ±3/7 level is `7 ⊕ 14`, and only `{1, 7, 14, 27}` occur anywhere in `End(ℝ¹⁶)`. Branchings confirmed too: the stabiliser of an imaginary octonion direction is 8-dimensional with irreducible adjoint action (hence simple, hence `su(3)`); `7 → 1 ⊕ 6` with the 6 irreducible of complex type (`3 ⊕ 3̄`); `14 → 8 ⊕ 6` with no fixed vector. So `7 ⊕ 7 → 2·(3 ⊕ 3̄ ⊕ 1)` and `7 ⊕ 14 → 8 ⊕ 2·(3 ⊕ 3̄) ⊕ 1`, as stated.

Two things to add, one to fix:

- **Add (parity of the octet's parent).** The two adjoint **14**s sit in *opposite* transpose sectors: antisymmetric at +3/7, symmetric at −3/7. Open Problem 3's gauge reading has to absorb the fact that the octet's parent changes parity with the sign of the eigenvalue.
- **Add (why "canonical selection" is hard).** C3 asks whether the channel canonically selects an SU(3). The obstruction is sharper than "unproven": SU(3) is the stabiliser of a *chosen* imaginary octonion direction, and Φ is invariant under all of G₂, which acts **transitively** on those directions. So no function of Φ alone can select one. Any selection must come from structure outside the channel — a state, the oriented chain's orientation bit, a distinguished subalgebra. Saying so converts Open Problem 3 from a hope into a well-posed search.
- **Fix (so(7)).** Theorem 3.4(b) and Appendix B.2 both gloss the multiplicity as "14 = dim so(7)". `dim so(7) = 21`. Nor is the sector so(7)-shaped: `Λ²(ℝ⁷) = 21 = 14 ⊕ 7`, whereas the 𝔭-sector is `7 ⊕ 7`. The right reading of 14 is `dim 𝔤₂` — which is exactly the coincidence the new `7 ⊕ 7` result defuses, so leaving so(7) in place undercuts the paper's own correction.

## Requested item 2 — the Design Theorem (OT 3.13)

This one doesn't need a numerical certificate; it needs three lines.

> `z ↦ L_z` is linear, so `L_zᵀ X L_z = Σ_{i,j} z_i z_j L_{e_i}ᵀ X L_{e_j}`. Averaging over any measure ν on the crack, `Φ_ν(X) = Σ_{i,j} Σ_ij L_{e_i}ᵀ X L_{e_j}` with `Σ = E_ν[zzᵀ]` — a *linear* function of the second moment alone. Hence two measures with equal second moments define the **same** channel identically.
>
> The moments: the 84-point one is `P_W/14` exactly (integer check, Part 2). For the invariant measure on the continuum crack — orthonormal pairs `(a,b)` in `Im 𝕆`, i.e. the Stiefel manifold `V₂(ℝ⁷) ≅ G₂/SU(2)` — `E[aaᵀ] = E[bbᵀ] = P₇/7` by G₂-invariance of the sphere measure and `E[abᵀ] = E[a·E[bᵀ|a]] = 0` since `b` is uniform on the unit sphere of `a^⊥`. With the `1/√2` normalisation this gives `P_W/14`. ∎

So the discrete and continuum channels are *equal*, and the remark that "G₂-invariance alone does not force it — the invariant-form space is 3-dimensional" is beside the point: you never need invariance, only the two moments. (For the record I confirmed the dimension counts: G₂-invariant symmetric forms on ℝ¹⁶ form a 6-dimensional space, of which the 3-dimensional subspace supported on W is presumably what §10's remark means.) I also checked the continuum characterisation the proof rests on — `a, b ∈ Im 𝕆`, `|a| = |b|`, `a ⟂ b` ⟹ `rank L_z = 12`, and breaking `|a| = |b|` restores full rank. **Audit obligation 2 discharged**; the 2.4·10⁻¹⁸ is a redundant check of a two-line identity.

## Requested item 3 — the Firewall Theorem (1.1)

As stated ("every theorem in Sections 2–7"), Theorem 1.1 is false for §2. Theorem 2.1 — that exactly 84 basic diagonals are zero divisors and the 14 diagonal `i = j` cases are excluded because `L_z` is full rank — is a statement about 98 candidates and the sedenion product. From `(K, μ)` you can count 84 operators; you cannot learn that 84 is *forced*. Appendix B.1's "the Aut(𝕊)-invariant Kraus family is unique" is likewise a statement about `Aut(𝕊)`, outside the firewall, and I found no verification of it anywhere in the repo.

Scoping the theorem to **§3–7** makes it true, and the 14-operator form above makes it nearly self-proving: everything in §3–7 is a statement about `(1/14)Σ_{i∈W} Ad(L_{e_i})`, which is manifestly free of sedenion multiplication once the 14 matrices are given. That is the shape a formal proof should take — a generation statement about fourteen signed permutations, not eighty-four.

## The [MEASURED] tier does not reproduce

Reproducing the audit's own chain definition (uniform event, restart on annihilation, burn-in), vectorised over independent copies:

| quantity | paper §4.3 / B.6 | `occurrence_ii_audit.py` (3 seeds × 55k) | this review |
|---|---|---|---|
| `s*` | 0.13172(5) | 0.13203(22) | **0.13182(3)** (96 × 280k and 64 × 1M, both handednesses) |
| `λ_q` | −0.01773(3) | −0.01931(88) | **−0.0186(1)** audit convention; **−0.0255** if only exact zeros are dropped |
| `P_survive` | ≈ 3/4 | — | **0.99976** (annihilation rate 2.4·10⁻⁴ per settlement) |

- **`λ_q` is not well posed as defined.** The oriented chain genuinely reaches annihilation: `‖K x‖` drops below 10⁻¹⁴ about 2.4·10⁻⁴ of the time and is *never* observed between 10⁻¹⁴ and 10⁻³, so these are annihilations up to round-off, not near misses. Whether those steps are scored (a single `log 10⁻¹⁶` is a −37 contribution) moves `λ_q` by 7·10⁻³; how restarts are booked moves it by a further 3·10⁻⁴. The quoted uncertainty is 3·10⁻⁵. `s*` is immune because it is bounded. §4.3 should state the convention and quote an error bar consistent with the seed spread the audit itself prints.
- **`P_survive ≈ 3/4` is a category error.** "Rank 12 out of 16" is a ratio of dimensions, not a probability: `ker L_z` is a 4-plane, so a continuously distributed state is annihilated with probability zero, and along the chain the measured rate is 2.4·10⁻⁴. On the discrete event set the rate would be 4/83. Whatever "long-run simulation agrees to 10⁻⁴" measured, it cannot have been this.
- **Open Problem 1 looks solved.** `1/8 + 1/147 = 0.131802721…`, but §4.3 and B.6 both evaluate it as "≈ 0.131723" — which happens to coincide with the paper's own measured `s*`, so the "1.7σ, hence not conclusively identified" verdict rests on comparing a measurement against a mis-evaluated candidate. With the correct value, my `s* = 0.13182(3)` sits ~0.6σ from `1/8 + 1/147`, while the paper's central value 0.13172 is ~3σ away from my estimate. Worth one targeted high-precision run before publication; I'd expect it to close the problem.

## Claims I did *not* verify

`Aut(𝕊)`-uniqueness of the family (Thm 2.1 / B.1); the "triad-closure cubic" whose slope is `2√3` (an OT-I object); the novelty survey in §1.3; and of course C1–C5, which are not the sort of thing a script settles. Theorem 3.4(c)'s algebraic-independence claim is not tested by anything in the repo.

## Corrections, in order of weight

1. **§3.4(b) and B.2:** "14 = dim so(7)" — `dim so(7) = 21`. Replace with `dim 𝔤₂`, and drop the so(7) gloss entirely, since the 𝔭-sector is `7 ⊕ 7`, not `Λ²(ℝ⁷) = 14 ⊕ 7`.
2. **§3.2 audit obligation and the Abstract:** "multiplicities 1, 7, 14, 21, 42 as G₂ irrep dimensions" / "multiplicities given by G₂ representation dimensions". G₂ irrep dimensions are 1, 7, 14, 27, 64, 77, …: 21 and 42 are not among them, and 21 isn't even a multiplicity in the Theorem 3.2 tables. The verified statement is that every eigenspace is a G₂-module built from `{1, 7, 14, 27}` — e.g. `42 = 1 ⊕ 14 ⊕ 27`, `72 = 4·1 ⊕ 2·7 ⊕ 2·27`, `28 = 14 ⊕ 14`.
3. **Theorem 1.1:** scope to §3–7 (see above), and mark the `Aut(𝕊)` uniqueness claim as unproven.
4. **§7:** "Choi rank ≤ 84" → exactly 14; and "self-adjoint (hence non-primitive)" is a non-sequitur — the depolarising channel is self-adjoint and primitive. Non-primitivity comes from the peripheral −1 mode.
5. **§4.3 / B.6:** `s*`, `λ_q` and `P_survive` as discussed; state the chain's annihilation convention.
6. **Open Problem 1 / B.6:** `1/8 + 1/147 = 0.131803`, not `0.131723`.
7. **Provenance of the artifact.** `data/kraus84.npz` holds `R_z` (right multiplication) relative to the doubling convention `(a,b)(c,d) = (ac − d b̄, ā d + c b)`; `R_z = −C L_z C` with `C` = conjugation, so it is orthogonally similar to the `L_z` of Definition 2.2 and has an identical exact characteristic polynomial (checked), but `L_{e₈} ≠ R_{e₈}`, so Theorem 3.3's "concretely `J = L_{e₈}`" reads as `J = R_{e₈}` on the released file. Note that `occurrence_ii_claude_code.md` reports bit-exact agreement with its own from-scratch algebra while I get the mirror family — which is precisely the symptom. One sentence pinning the doubling convention (and the handedness of `L`) in §2.1 or a `data/README` would remove the ambiguity for every future auditor.
8. **Theorem 4.2** should be [FORCED], not [MEASURED / exact]. `Var[τ] = 1/18` follows from `E[(xᵀMx)²] = ((tr M)² + 2 tr M²)/(d(d+2))` with `tr M = 16`, `tr M² = 24`; and since *every* `M_a` has spectrum `{0⁴,1⁸,2⁴}` it holds conditionally on each single event — it is not a property of μ at all.
9. **Theorem 4.3** deserves its two-line proof: `⟨K_a x, e₀⟩ = −⟨x, z_a⟩` because `Kᵀ = −K`, and `⟨K_a x, e₈⟩ = ⟨x, e₈ z_a⟩ = ⟨J z_a, x⟩` because every event anticommutes with `e₈` (checked). Stating it makes clear that the Born quotient is antisymmetry plus that anticommutation, and that C2's probabilistic content is entirely in the reading.
10. **Numbering/editorial:** `Definition 3.5` and `Theorem 3.5` collide, and C3 and the summary table cite "Theorem 3.5" as if it were in the body (it appears only in B.2); §3.4 is titled "Resolution of Open Problem 7" while B.8's Open Problem 7 is "Particle representation"; there are two §10s. The PR description's links are repo-relative (`../blob/born-channel/…`) and don't resolve from the PR page.

## Verdict

The mathematics of §2–7 is sound and, in the places I could reach, stronger than claimed — exact over ℚ, and reducible to a 14-term mixed-unitary average that makes several of the paper's own theorems immediate. Audit obligation 2 (the Design Theorem) is discharged here; obligation 1 (representation theory) reproduces independently, with the so(7) and "irrep dimension" wordings needing repair; obligation 3 (the Firewall) needs its scope narrowed before it can be proved at all.

I would not merge the [MEASURED] tier as written: two of its three constants disagree with the repo's own audit, `λ_q` needs a stated convention, and Open Problem 1 is being kept open by an arithmetic slip. None of that touches the [FORCED] claims, which is the paper's primary claim, and it should be cheap to fix.

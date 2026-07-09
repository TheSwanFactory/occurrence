# Independent Verification

This document records an independent re-derivation of the computational claims
in [`occurrence-theory.md`](occurrence-theory.md), performed 2026-07-09.

The verification was written from the paper's definitions, with a Cayley–Dickson
implementation built from scratch. It does not import `topographo`. That
matters: if the repository's structure tensor were wrong, a checker that imports
it would confirm the paper's conclusions and the algebra's bug at the same time.

**Summary.** The mathematics reproduces. Every structural and spectral claim
tagged `[C]` in the paper was reconfirmed at machine precision, and the two
`[M]` constants reproduce within their stated error bars. Three statements in
the paper's prose are wrong — including, unfortunately, its own thesis sentence —
and one claim the audit script asserted without testing turns out to be **true**,
and now carries a certificate.

---

## 1. What reproduces

Independent re-implementation, fresh structure tensor, no shared code.

| Claim | Paper | Reproduced |
|---|---|---|
| Basis-form unit zero divisors (the crack) | 84 | 84 |
| `M_z` spectrum on the crack | {0⁴, 1⁸, 2⁴} | exact, single spectrum |
| `dim ker L_z` | 4 | 4, uniform |
| Thm 3.6: ‖E[M_z] − I‖ on the 84-design | 8.9e-16 | 8.88e-16 |
| Thm 3.13: ‖E[zzᵀ] − P_W/14‖ | 2.4e-18 | 0.0 |
| Thm 3.9(a): ‖J² + I‖, ‖J + Jᵀ‖, ‖[L_e₈, R_e₈]‖ | 0.0 | 0.0, 0.0, 0.0 |
| Thm 3.9(b): ‖Φ(L_e₈) + L_e₈‖ | exact | 8.88e-16 |
| Thm 3.8(c): associative envelope of {L_x} | End(ℝ¹⁶) = 256 | 256 |
| Thm 3.8(d): Lie closure of {L_x} | 𝔰𝔬(16) = 120 | 120 |
| Thm 3.12: every eigenvalue in (1/7)·{0,±1,±3,±2√3,±7} | exact | worst deviation 4.9e-15 |
| Thm 5.3: ‖Φ(P_e₀) − P_W/14‖, ‖Φ(P_e₈) − P_W/14‖ | exact | 0.0, 0.0 |
| Thm 5.5: involution z ↦ z·e₈ preserves Σ, μ, M_z | exact | 1.1e-16 |
| Thm 3.10(6): conj(zx) = xz | exact | 0.0 |

The spectrum result is the strongest thing in the paper. All 256 eigenvalues of
Φ land on a nine-element set to within 5×10⁻¹⁵, with multiplicities
{1, 7, 14, 21, 42} — dimensions of G₂ representations. This does not happen by
accident.

The `[M]` constants also reproduce, from a different implementation and
different seeds (three seeds, N = 8000, T = 1200, burn-in 200):

| Constant | Paper | Reproduced |
|---|---|---|
| spine share `s*` | 0.13172 ± 0.00005 | 0.131745 ± 0.000041 |
| quenched Lyapunov `λ_q` | −0.01773 ± 0.00003 | −0.017786 ± 0.000042 |

The conjecture `s* = 1/8 + 1/(3·7²) = 0.131803` sits 0.000058 from the
reproduced mean, about 1.4 seed-σ — compatible, consistent with the paper's
"1.7σ". Competitors 19/144 = 0.131944 and 1/8 + 1/140 = 0.132143 remain excluded.

The corollary of Thm 3.5 (‖T_z‖ ≤ 1, equality exactly on Σ) is confirmed as a
byproduct of the strain sweep: interpolating events from the alternative cone to
the crack gives mean ‖T_z‖ = 0.000000, 0.470588, 0.800000, 0.960000, 1.000000 at
σ = 0, ¼, ½, ¾, 1, with `L_z` becoming singular exactly at σ = 1.

---

## 2. What is wrong in the paper

### 2.1 The thesis sentence (§10, Q4)

> "The settlement channel spectrum: an 84 × 84 matrix, trace 1 …"

Φ acts on End(ℝ¹⁶), so it is a **256 × 256** matrix. 84 is the number of Kraus
operators, not the dimension. Its matrix trace is **0**, forced by the ±
symmetry of the very spectrum the paper prints one page earlier. (Φ is
*trace-preserving as a channel* — Tr Φ(X) = Tr X — which is a different
statement, and is true.)

Both numbers in the sentence the paper nominates as its legacy were wrong.
Fixed, and Definition 3.7 now states the distinction explicitly.

### 2.2 Theorem 3.12's antisymmetric multiplicities do not sum

The antisymmetric sector is declared dim 120, then enumerated as

> ±2√3/7 (×14 each), 3/7 (×14), −3/7 (×7), **±1/7 (×42)**, 0 (×28), −1 (×1)

which sums to **162**. The correct sector, computed here, has **+1/7 with
multiplicity 42 and no −1/7 at all**; every other entry matches. A single
spurious `±`. Removing it makes the column sum to 120.

Locked in by `tests/test_crack_topology.py::test_antisymmetric_sector_multiplicities_sum_to_120`.

### 2.3 The audit contradicted Theorem 3.10(6)

The identity `conj(zx) = xz` holds exactly. The paper (Thm 3.10(6), §4) uses it
to argue the algebra **cannot detect** which slot is event and which is state —
the transition functionals are exchange-symmetric — and therefore the
orientation bit must be supplied externally. That is the load-bearing argument
of Section 4.

`test_event_state_symmetry` computed the same identity and printed:

> `[T13b] => Role exchange is NOT a gauge symmetry.`
> `State (carrying forward) vs event (sampled) is ONE Z_2 bit,`
> `but algebrically preferred: only one orientation is stable.`

as an uncomputed `print`. Same identity, opposite conclusion, contradicting the
paper's own theorem. Removed; the section now certifies the anti-automorphism
`conj(ab) = conj(b)conj(a)` on *general* elements (on pure elements it degenerates
into the identity above, so testing it there would be circular).

---

## 3. What the audit script asserted but never tested

`occurrence_theory_audit.py` ended with:

```python
print("  Occurrence Theory is mathematically sound, genuinely novel,")
print("  and survives full algebraic audit.")
```

printed unconditionally, as a string literal, regardless of what ran above it.
Underneath:

**Test 8 (crack topology).** The edge predicate was
`smallest_singular_value(L_z @ L_w) < 1e-9`. Every `L_z` already has a
4-dimensional kernel, so `L_z L_w` is singular for **every** pair. The graph is
complete: measured degree 83 for all 84 vertices, all 3,486 edges present. The
script printed `4-regularity check: False` and then asserted
`Diameter ~3, 7 components (one per Fano line)` on the next line anyway.

**Test 11 (amnesia).** Claimed frame fractions collapse
`(2/3, 1/6, 1/6) → (1/3, 1/3, 1/3)` in one step, but initialized the ensemble at
`e₀`, whose pencil content is exactly zero. The `t=0` row was
`0 / (1.5·0 + 1e-300)` — it printed `(1/3, 1/3, 1/3)` by construction. The
demonstration never ran.

**Test 12 (strain field).** Looped `for s in [0.0, 0.5, 1.0]` and never used `s`
in the body; all three rows drew events from the same crack distribution. The
three reported λ_q values (−0.01764, −0.01792, −0.01819) differ only by RNG
noise. The `tau` column printed the literal strings `'inf'` and `'~100'`,
selected by `if s == 0`.

**The gates.** Three of the four tested a single hardcoded basis pair `(e₁, e₂)`.
Worse, `verify_gates` built its *own* octonion table internally rather than
testing the algebra it was handed — so a corrupted `OT.C` passed every gate.
And all four gates only exercise `e₀..e₇`, so any corruption confined to the
sedenion block (e.g. `e₉·e₁₀`) was invisible to them.

---

## 4. What the audit was hiding: Test 8's claim is true

The most interesting finding. Under the **correct** predicate — genuine algebra
annihilation `z · w = 0`, rather than `L_z L_w` singular — the paper's
structural claim holds exactly:

- the annihilation graph is **4-regular** (every vertex has degree 4);
- it has **exactly 7 connected components**, of **12 vertices each**;
- its **diameter is exactly 3**.

Moreover the components have a clean invariant the paper does not state. Every
basis zero divisor has the form `(eᵢ ± e_{j+8})/√2` with `i, j ∈ {1,…,7}`, and
the connected component is determined by

> **i ⊕ j** (bitwise XOR)

which ranges over the seven nonzero points of 𝔽₂³ — the Fano plane. So "one
component per Fano line" is right, stated slightly loosely: the label set is the
seven **points** of 𝔽₂³ ∖ {0}.

The paper was correct. The verification script simply could not see it, because
its edge predicate was true for every pair. Certified in
`tests/test_crack_topology.py`.

---

## 5. What changed in this PR

**`occurrence-theory.md`**
- §10 Q4: `84 × 84, trace 1` → `256 × 256 … trace 0, built from 84 Kraus operators`.
- Def 3.7: added the channel-trace vs. matrix-trace distinction.
- Thm 3.12: `±1/7 (×42)` → `1/7 (×42)` in the antisymmetric sector.
- New §7.1: what nonassociativity does and does not buy (see below).

**`occurrence_theory_audit.py`** — rewritten so every printed claim is computed.
- A certificate ledger. `certify()` / `certify_equal()` print a computed number
  against a threshold and record failures. `main()` returns **1** if any
  certificate fails; the unconditional success message is gone.
- Gates now test the algebra they are **given** (`OT.C`), on **random** elements,
  with new gates `G0` (octonion block closes) and `G5`–`G7` (sedenion-level
  identity, quadratic, antisymmetry). Verified to fail on a corrupted table,
  including corruption confined to the sedenion block.
- Test 8: demonstrates the old predicate is vacuous, then certifies 4-regularity,
  7 components of 12, diameter 3, and the Fano-plane labelling.
- Test 9: certifies Lie closure = 120 = 𝔰𝔬(16) and envelope = 256 = End(ℝ¹⁶).
- Test 9b: certifies the full spectrum, its reality, `trace(Φ) = 0`, unitality,
  Φ(L_e₈) = −L_e₈, and the flicker projectors.
- Test 11: loads the ensemble **asymmetrically** and measures the collapse
  (2/3, 1/6, 1/6) → (1/3, 1/3, 1/3), residual 0.0013 after one event.
- Test 12: `sigma` now parameterizes the event family, sweeping from the
  alternative cone to the crack; certifies the cone is sterile and σ = 1
  saturates ‖T_z‖ = 1.
- Test 13b: no longer contradicts Thm 3.10(6).
- The final section no longer renders a verdict. It states what a passing run
  does and does not mean, and lists what is *not* verified.

**`tests/`** — 21 → 42 tests. New: `test_audit_certificates.py` (adversarial:
the audit must fail when the algebra is corrupted, and must not print success
when a certificate fails) and `test_crack_topology.py` (the rescued Fano
structure, the spectrum, the 120-sum).

---

## 6. On the scope of the result

Added to the paper as §7.1, and restated here because it is the thing most
likely to be misread.

Every object in this paper is an ordinary real matrix. `x ↦ L_x` is not a
homomorphism — that failure *is* the nonassociativity — but the `L_x` themselves
associate, and Theorem 3.8(c) proves the algebra they generate is **all** of
End(ℝ¹⁶). No sedenionic structure survives inside it.

So nonassociativity is not a structure carried by the dynamics. It is a
**selection principle**: it is what makes Σ nonempty at all (composition algebras
have no zero divisors, so ℝ, ℂ, ℍ, 𝕆 have empty crack), what makes Σ a single
homogeneous orbit, and what fixes μ. Once the 84 Kraus operators are selected,
the algebra has done its work and departs. Everything downstream is linear
algebra on a distinguished finite family of matrices.

The paper says as much in §7 — the oriented layer **is** a Furstenberg system of
i.i.d. matrix products, and "we claim no novelty there." The defensible claim is
narrower and still interesting: a nonassociative algebra *nominates* a compact
homogeneous family of singular operators whose averaged channel is exactly
solvable, with spectrum in sevenths and one quadratic irrational.

**On physics.** The paper makes no standard-model claim, and is right not to.
𝔰𝔲(3) ⊕ 𝔰𝔲(2) ⊕ 𝔲(1) embeds in the imaginary octonions; that is classical and
this construction neither reproduces nor constrains it. Theorem 5.2 proves Φ
annihilates the generation-labeling observables **exactly**, so the S₃
"generation" structure carries no dynamical content: any generational asymmetry
is initial-conditional, not dynamical. The names "time," "generation,"
"survivorship," and the Born-shaped reading of `s′` are tagged `[I]` throughout
and priced at zero. That restraint is the paper's chief virtue and should be
defended, not relaxed.

The open problem worth the effort is **2√3/7**. It is the only irrational
eigenvalue, it lives only in the antisymmetric sector, and nothing in the paper
explains it. That is Open Problem 7, and it is the right place to dig.

---

## 7. Reproducing this

```bash
python3 occurrence_theory_audit.py   # exit 0 iff every certificate passes
echo $?
python3 -m pytest tests -q           # 42 tests
```

The audit's failure path is itself tested: corrupt `OT.C` and the gates must
catch it, including corruption confined to the sedenion block that the four
classical octonion gates cannot see.

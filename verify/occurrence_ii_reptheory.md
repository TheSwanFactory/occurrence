# Exact & representation-theory verification (SageMath)

**Paper:** Occurrence Theory II — The Born Channel
**Verifier:** claude_code (Anthropic), via SageMath 10.9
**Script:** [`occurrence_ii_reptheory.sage`](occurrence_ii_reptheory.sage)
(`sage verify/occurrence_ii_reptheory.sage` → exit `0`)

The exact / symbolic complement to the floating-point numpy audit
(`occurrence_ii_audit.py`): it does what numpy cannot — **exact linear algebra
over ℚ** and honest **G₂ / SU(3) representation theory**. Independent of
`topographo`; loads only `data/kraus84.npz`.

## How to run

`sage verify/occurrence_ii_reptheory.sage`, or paste into
[SageMathCell](https://sagecell.sagemath.org). Confirmed locally with
SageMath 10.9 (exit `0`, every check `OK`).

## Results

### 1. Exact spectrum over ℚ (upgrades Theorem 3.2 to theorem-grade)

`S = Σₐ μₐ Kₐ⊗Kₐ` has **rational** entries (`√2·Kₐ` is integral), so this is
exact, not numerical. The minimal polynomial `m(S) = 0` holds exactly for
`x·(x∓1/7)(x∓3/7)(x∓1)(x²−12/49)`, and exact ranks over ℚ give the
multiplicities:

```text
   1: 1     3/7: 21    1/7: 42    0: 100    -1/7: 42   -3/7: 21   -1: 1
   ±2√3/7: 28  (14 each)          total: 256
```

So Theorem 3.2's spectrum and multiplicities are **proven exactly**, including
the irrational pair `±2√3/7` as the exact factor `x² − 12/49`.

### 2. Seeded numerical G₂-module identification (§10.1)

First a sanity check that G₂ genuinely *is* a symmetry: with the concrete action
`R(g)X = g X gᵀ` (i.e. `g₁₆ ⊗ g₁₆` on vec-space), `‖[S, R(g)]‖ = 1.3·10⁻¹⁶` —
Φ commutes with the G₂ action, so the eigenspaces really are G₂-modules.

This section is strong, reproducible numerical evidence rather than an exact
symbolic proof: it builds `Der(𝕆)` by floating-point SVD, exponentiates seeded
random generators, diagonalizes the floating channel, and fits characters by
least squares. Reading each Φ-eigenspace over `{1, 7, 14, 27}` gives:

```text
λ = ±1     dim 1    :  1
λ = ±𝔭     dim 14   :  7 ⊕ 7          ← NOT the adjoint
λ = ±3/7   dim 21   :  7 ⊕ 14
λ = ±1/7   dim 42   :  1 ⊕ 2·7 ⊕ 27
λ = 0      dim 100  :  4·1 ⊕ 2·7 ⊕ 2·14 ⊕ 2·27
totals over End(ℝ¹⁶):  8·1 ⊕ 12·7 ⊕ 4·14 ⊕ 4·27
```

**Key finding.** The distinguished dim-14 irrational (𝔭) sector is **7 ⊕ 7**,
*not* the adjoint 𝔤₂ — the coincidence `14 = dim 𝔤₂` is just that. The four
adjoint (`14`) copies live in the **±3/7 sectors** (`7 ⊕ 14`) and the `0`
sector. This also confirms the earlier point that multiplicities like `21, 42,
72` are dimensions of **reducible** G₂-modules, not irreps.

### 3. G₂ → SU(3) branching (Conjecture C3)

```text
7             → 3 ⊕ 3̄ ⊕ 1
14 (adjoint)  → 8 ⊕ 3 ⊕ 3̄              (NOT 8 ⊕ 6)
𝔭-sector = 7 ⊕ 7  → 2·(3 ⊕ 3̄ ⊕ 1)     ← no gluon octet
±3/7-sector = 7 ⊕ 14 → 8 ⊕ 2·3 ⊕ 2·3̄ ⊕ 1   ← contains the octet
```

**Consequence for C3.** Because the 𝔭-sector is `7 ⊕ 7`, it branches with **no
octet**. A gluon `8` appears only alongside an adjoint `14` — i.e. in the ±3/7
sectors, which branch as `8 ⊕ 2·3 ⊕ 2·3̄ ⊕ 1` (checked directly). So C3's
original hope — color-gauge structure *in the distinguished 𝔭-sector* — is not
supported; if a gauge story exists, **the ±3/7 sectors are where it lives.**
(This corrected two successive misstatements — `8 ⊕ 6`, then `8 ⊕ 3 ⊕ 3̄` — that
had been attached to the 𝔭-sector.)

### 4. Design-Theorem invariants (§10.2 / OT Theorem 3.13)

The space of G₂-invariant symmetric forms on the pencil `W = 7 ⊕ 7` is
**3-dimensional**. So `E[zzᵀ] = P_W/14` is **not** forced by G₂-invariance
alone (a 3-parameter family is invariant); the specific measure is doing real
work. The §10.2 proof obligation must establish that, not merely invoke
symmetry.

## Status

The exact spectrum, exact Weyl-character branching, invariant count, and
seeded numerical eigenspace naming all reproduce (exit `0`). The numerical
section-2 identification (`𝔭`-sector = `7 ⊕ 7`), combined with the exact
section-3 branching, is a substantive
correction to Conjecture C3 and is flagged in the paper for author/specialist
review.

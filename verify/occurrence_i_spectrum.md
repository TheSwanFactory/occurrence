# Exact settlement-channel spectrum — a runnable proof of Theorem 3.12 (Paper I)

**Targets:** the flagship [C] result of Paper I — *the channel Φ on the 84-element
basic crack has spectrum exactly `(1/7)·{0, ±1, ±3, ±2√3, ±7}` with
G₂-multiplicities* (Theorem 3.12) — plus the surrounding exact-arithmetic
obligations (the four mandatory gates, forced equilibrium, the parity eigenvalue).

The numpy audit ([`occurrence_i_audit.py`](occurrence_i_audit.py)) certifies this
spectrum **numerically** at threshold `1e-12`. This script upgrades it to an
**exact symbolic** result. Every basic Kraus operator `L_z` has entries in
`{0, ±1}`, so Φ is an exact rational `256×256` matrix and its characteristic
polynomial factors exactly over `ℚ` — no tolerance, no threshold.

It is **self-contained**: it builds `𝕊 = A₄` from the Cayley–Dickson doubling over
`ℚ` from scratch, and does **not** touch `topographo` or the released
`data/kraus84.npz`.

## How to run

The script is [`occurrence_i_spectrum.sage`](occurrence_i_spectrum.sage). Either:

- `sage verify/occurrence_i_spectrum.sage`, or
- paste it into **[SageMathCell](https://sagecell.sagemath.org)** (free, no
  login) and press *Evaluate*.

## Result

Confirmed locally with SageMath 10.9 (`sage verify/occurrence_i_spectrum.sage`,
exit `0`):

```text
=== Channel spectrum (Theorem 3.12) — exact factorization ===
  charpoly factors as:
    (x - 1) * (x + 1) * (x - 3/7)^21 * (x + 3/7)^21 * (x - 1/7)^42 * (x + 1/7)^42 * x^100 * (x^2 - 12/49)^14
  matches (1/7).{0,+-1,+-3,+-2sqrt3,+-7} with Thm 3.12 mults : True
```

Reconciling with the two sectors of Theorem 3.12 (symmetric ⊕ antisymmetric):

| eigenvalue | sym mult | antisym mult | total | charpoly factor |
|---|---|---|---|---|
| `1`      | 1  | –  | 1   | `(x-1)` |
| `-1`     | –  | 1  | 1   | `(x+1)` (the parity, Thm 3.9b) |
| `3/7`    | 7  | 14 | 21  | `(x-3/7)^21` |
| `-3/7`   | 14 | 7  | 21  | `(x+3/7)^21` |
| `1/7`    | –  | 42 | 42  | `(x-1/7)^42` |
| `-1/7`   | 42 | –  | 42  | `(x+1/7)^42` |
| `0`      | 72 | 28 | 100 | `x^100` |
| `±2√3/7` | –  | 14 each | 28 | `(x^2-12/49)^14` |

Total `256`, `trace(Φ) = 0`. The irrational pair `±2√3/7` is recorded as the exact
quadratic factor `x² − 12/49` (since `(2√3/7)² = 12/49`).

## What else the script proves (all exact over `ℚ`)

- **Mandatory gates** (Verification Protocol, Appendix): composition on the
  octonion hull, antisymmetry `L_x + L_xᵀ = 0`, quadratic `x² = −e₀`, and Moufang
  `(ab)(ca) = a((bc)a)` on the octonion basis. A failed gate aborts the run.
- **Basic crack** (Definition 3.4): exactly **84** basis-form zero divisors.
- **Forced equilibrium** (Theorem 3.6): `(1/84)·Σ M_z = I` exactly.
- **Parity** (Theorem 3.9b): `Φ(L_{e₈}) = −L_{e₈}` exactly — the source of the
  lone `−1`.
- **G₂ rep-theory** (Theorem 3.2): `7` and `14` are the two fundamental G₂ irreps
  (via `WeylCharacterRing`), consistent with `𝕊 = 1 ⊕ 1′ ⊕ (7 ⊗ 2)`.

## Notes

- The Cayley–Dickson convention used is Baez's
  `(a,b)(c,d) = (ac − d̄b, da + bc̄)` with `(a,b)* = (a*, −b)`; the four gates are
  precisely the self-check that this convention reproduces `𝕆 ⊂ 𝕊` correctly. A
  reviewer preferring a different doubling convention can swap `cd_mul`/`cd_conj`
  and rerun — the gates will confirm validity and the spectrum is
  convention-independent.
- Working unnormalized (`w = e_a ± e_b`, `z = w/√2`) keeps every matrix integer;
  the two `1/√2` factors enter the superoperator as a single rational `1/2`, so
  exactness is preserved end to end.

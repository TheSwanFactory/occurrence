# Independent Verification of the Born Channel (Occurrence Theory II)

**Paper:** Occurrence Theory II — The Born Channel (v0.6)
**Date:** July 11, 2026
**Verifier:** Claude Code (Anthropic) — independent reviewer, distinct from the
first-party `verify/occurrence_ii_audit.py`
**Artifacts:** `data/kraus84.npz` (released) and a from-scratch re-derivation
**Script:** [`verify/occurrence_ii_claude_code.py`](occurrence_ii_claude_code.py)
(`python3 verify/occurrence_ii_claude_code.py` → exit `0`)

> Reviewer cell (handle: `claude_code`). Per `verify/README.md` this does **not**
> import `topographo`. Independence here is strong: the script re-derives the
> sedenion algebra from a self-contained Cayley–Dickson doubling, rebuilds the
> 84 zero-divisor Kraus operators itself, and then verifies every [FORCED] claim
> on the released family. The structural invariants (spectrum, multiplicities,
> J, coherence constant, lattice) are basis-independent, so they hold regardless
> of any change-of-basis difference between algebras.

**Summary:** Every [FORCED] claim of Paper II v0.6 reproduces to machine
precision. In addition, the independently regenerated Kraus family matches the
released `data/kraus84.npz` **exactly** (‖·‖ = 0.00e+00) — the same basis
convention, not merely the same invariants. No discrepancies.

## 0. Provenance cross-check (from-scratch vs released)

| Check | Result |
| --- | --- |
| Regenerated family shape | `(84, 16, 16)` ✓ |
| `‖K_regenerated − data/kraus84.npz‖` | **0.00e+00** (identical convention) |
| `‖μ_regenerated − μ‖` | 0.00e+00 |

The 84 operators were rebuilt as `K_a = L_{z_a}`, `z = (e_i + s·e_{8+j})/√2`
(`i,j ∈ 1..7`, `i≠j`, `s = ±1`), using an independent Cayley–Dickson product —
and land exactly on the released ground truth.

## 1–9. Forced claims on the released family

| # | Claim (theorem) | Check | Result |
| --- | --- | --- | --- |
| 1 | Integrity | `sum(mu)=1`, no NaN/Inf, shapes | ✓ |
| 2 | CPTP / unital (Thm 3.1) | `‖E[KᵀK] − I‖` | 1.02e-14 |
| 3 | Antisymmetry (Def 2.2) | `maxₐ ‖Kₐ + Kₐᵀ‖` | 0.00e+00 |
| 4 | Events & ranks (Thm 2.1) | `‖zₐ‖ = 1`; all ranks = 12 (ker 4) | ✓ |
| 5 | Pencil / spine | `E[zzᵀ]` rank 14, spine dim 2, eigenvalues = 1/14 | 1.11e-16 |
| 6 | Nine-level spectrum (Thm 3.2) | 9 distinct levels | ✓ |
| 6 | Symmetric-sector G₂ mults | `{1, 7, 72, 42, 14}` | exact ✓ |
| 6 | Antisymmetric-sector G₂ mults | `{14, 14, 42, 28, 7, 14, 1}` | exact ✓ |
| 7 | Complex structure J (Thm 3.3) | simple −1 mode; `‖J+Jᵀ‖=0`; `‖J²+I‖` | 4.33e-15 |
| 8 | Coherence constant (Thm 3.4) | two-time ratio − `2√3/7` | 5.55e-17 |
| 9 | Annihilation lattice (Thm 6.1) | 4-regular; 7 cells × 12; diameter 3 | exact ✓ |

## Conclusion

The Born Channel's forced algebraic and spectral structure is **independently
confirmed**, and the released `data/kraus84.npz` is not merely internally
consistent but reproduces bit-for-bit from a from-scratch algebra. This extends
the earlier integrity review (`occurrence_ii_grok.md`) to the full spectrum, the
complex structure J, the coherence constant, and the annihilation lattice.

**Status: PASSED — full [FORCED] claim set, independently reproduced.**

*Representation theory (§10.1 / C3) is a separate concern, checked in its own
cell `occurrence_ii_reptheory` (exact spectrum over ℚ, G₂-module naming of every
eigenspace, SU(3) branching, Design-Theorem invariants). Its notable finding:
the 𝔭-sector is `7 ⊕ 7`, not the adjoint 𝔤₂ — so it carries no gluon octet,
which corrects Conjecture C3.*

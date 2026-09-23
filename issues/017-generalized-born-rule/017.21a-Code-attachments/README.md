# Issue 017.21a exact central-constitution audit

The unfenced follow-up to `017.20`. It changes no stable public API and imports no optimizer, training, NumPy, or Interact code. Arithmetic is `fractions.Fraction` plus an exact `Q(√7)` field, because the certified Fixed frame contains `√7`.

## Result

**Conditional partial positive.** Dropping the `017.20` evidence fence earns:

- a **canonical intrinsic binary formal test**, the two minimal central idempotents `z_M` (ideal dim 3, ambient rank 4) and `z_s` (dim 1, rank 12);
- a **binding-unique preparation compiler** into `S(E)`, using the unique target-automorphism-invariant extensions `τ_M(A,s) = Tr(A)/2` and `δ_s(A,s) = s`;
- exactly **two inequivalent orientations**, i.e. one specification bit.

It does **not** earn a canonical ambient ray (`τ_M` realizers form a `PO(2)` family, `δ_s` realizers `ℝP¹¹`) or any physical bridge. Those layers remain C0-or-C∅ and C∅.

| Layer | Grade |
|---|---|
| unordered central test | C3 |
| state after a binding | C3 |
| joint labeled compiler + test | C1, one bit |
| raw ambient ray | C0-or-C∅ |
| physical realization | C∅ |

## Declared premises

The result is conditional on three **new declarations**, frozen in `declared_premises.json` before construction: P2 the full formal effect interval, P3 the `S(E)` codomain, P4 invariant central-character extension. P4 is provably load-bearing — 13 of 13 sampled `ω_t` states reproduce the exact central distribution but only `t = 1/2` is invariant, so without P4 the compiler is a continuum.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `declared_premises.json` | premises and falsifiers, frozen before construction | `3cb2b3f4decba679dffa05ebe72ded9a7b9645b7df8bb4c1e2bd87cd4198e921` |
| `audit_central_constitution.py` | exact center, states, compiler, ambient boundary, control | `094bfa877ed6aadf476daa2d2e07a224402d0abd4285e836b5fd9130189a6aab` |
| `test_audit_central_constitution.py` | 13 exact and negative-boundary tests | `7d4666c7aeb4e8c68682e99096d13642c1c9315bbf436a8336814f3f7bdda83e` |
| `central_constitution_audit.json` | canonical machine-readable audit, 36 checks | `e9581c25db9f82dbbba6bb1fe800e3d3b94c92d20fc394ab8adb48ec0d1ff59c` |

The source schema is **not** duplicated here. It is read byte-pinned from `../017.20-Code-attachments/source_fixture.json` at SHA-256 `20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c` and the loader refuses any change.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.21a-Code-attachments/audit_central_constitution.py --check
uv run --frozen pytest issues/017-generalized-born-rule/017.21a-Code-attachments/test_audit_central_constitution.py
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.21a-Code-attachments
```

The audit derives all four central idempotents symbolically, rejects non-scalar candidates by exact operator-commutation, sweeps rational rotations plus the reflection for invariance, checks all 168 × 49 automorphism/input pairs per binding, evaluates all 49 inputs exactly under both bindings, realizes both states in exact `Q(√7)`, runs a same-copy near-miss control, and requires the committed JSON to match executable output byte-for-byte.

## Fences

No physical Test Realization, no predictive or training-economy claim, and no assertion that the ambient carrier lacks Fano-equivariant structure — an invisible faithful commutant action exists and is reported. Reject P2, P3, or P4 and the result reverts to `017.21`'s C∅ without contradiction.

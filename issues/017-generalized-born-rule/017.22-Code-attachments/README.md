# Issue 017.22 exact operational-equivalence and state-boundary audit

The follow-up that decides what the 017.21a result *means* operationally. It adds
no premise, changes no stable public API, and imports no optimizer, training,
NumPy, or Interact code. Arithmetic is `fractions.Fraction` plus an exact
`Q(√7)` field, because the certified Fixed frame contains `√7`.

## Result

**Strong positive on the state boundary, with an open instance-identity defect.**

- One-shot TDM evaluation factors through `S(E)` by certified Theory 27, and
  `S(E)` is **minimal**: `Eff(E)` contains a linear basis of `E`, so no coarser
  carrier preserves every formal probability.
- `≈_Σ` is exactly equality of the central observable `λ = ω(z_M)`, so
  `S(E)/≈_Σ ≅ [0,1]`.
- `≈ᵀᴰᴹ_Σ` is the kernel of `R ↦ F_R`. All 20 audited realizations — both
  017.21a orientations plus 18 non-central, non-invariant rank-one rivals — fall
  in **one** class.
- The orientation bit is Σ-unobservable and **provably not certified gauge**: no
  target automorphism exchanges `z_M` and `z_s`, because the ideals have
  dimensions 3 and 1.
- The `Aut(E,E₊,1_E)`-invariant states are **exactly** `ω_λ(A,s) = λ·Tr(A)/2 +
  (1−λ)·s` for `λ ∈ [0,1]`, derived by exact null-space computation (rank 2,
  kernel spanned by the trace and scalar directions).
- P2 was **already certified** Theory, not a 017.21a declaration. P4 is a
  representative-selection principle that additionally **forces** the declared
  test to be central.

## Premise correction

| ID | 017.21a label | Corrected classification |
|---|---|---|
| P1 | certified upstream | unchanged |
| P2 | "declared here" | **certified OT structure** (Theory 27 §1, Theory 41) |
| P3 | "declared here" | abstraction-boundary choice, now **justified** by §F |
| P4 | "declared here, load-bearing" | new principle, but **representative selection**, not behavior |

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `audit_operational_equivalence.py` | the exact audit | `47ffb383f647ab846c2719190039be5c7f170f9370482d9e40f2895ef3fb3b6c` |
| `operational_equivalence_audit.json` | canonical machine-readable audit, 70 checks | `e81d7056c4167c44e25d18904d8335da11803935bbf5c8a0be39779d9422f6a2` |
| `orientation_behavior_table.json` | required artifact 3: 98 rows, both orientations × 49 inputs | `7d7c3146d7ed1505cf9364230e9ad68aaa71f55110dd837142911a9724899205` |
| `test_audit_operational_equivalence.py` | 23 exact and negative-boundary tests | `f03ed78df4179c7db1a8dc0406e9cd1219c931bc8fb139ae163e905dc8384a94` |

Nothing is duplicated. The source schema is read byte-pinned from
`../017.20-Code-attachments/source_fixture.json`, and the 017.21a premises and
audit JSON are read byte-pinned from `../017.21a-Code-attachments/`. Any change
to those three files makes the loader refuse to run.

## What the audit actually computes

- exact null space of the stacked invariance constraints over 11 target
  automorphism probes (10 rational rotations plus the reflection);
- `≈_Σ` reflexivity, symmetry, transitivity and the `λ` characterization on all
  25 × 25 probe-state pairs, and `Eff(E)` separation on all 295 distinct pairs;
- the `λ = 0` fiber collapse by exhausting a rational state grid;
- one public family per realization over all 49 admitted inputs, for 20
  realizations, with exact `Δ(A_d)` typing;
- an exhaustive P4 forcing search over 225 grid effects (200 of them non-central)
  × 169 `λ` pairs, returning **exactly two** solutions;
- 8 projectively distinct ambient rays inducing `τ_M` and 5 inducing `δ_s`, with
  exact projective-distinctness tests in `Q(√7)`;
- the successor identifiability and the `θ ↦ 1 − θ` orientation absorption on a
  25-point grid;
- a relation-erased control in which the public family fails closed.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.22-Code-attachments/audit_operational_equivalence.py --check
uv run --frozen pytest issues/017-generalized-born-rule/017.22-Code-attachments/test_audit_operational_equivalence.py
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.22-Code-attachments
```

`--write` regenerates both artifacts; `--table` prints the behavior table.

## Fences

No physical Test Realization. No predictive, accuracy, or training-economy
claim. Operational equivalence is **not** called gauge. Public behavior is **not**
equated with TSAT instance identity — that gap is reported as a terminology
defect with a recommended successor issue. Nothing is trained, and no benchmark
data is generated.

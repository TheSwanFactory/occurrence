# Issue 017.20 exact input-constitution audit

This directory contains the isolated, zero-training execution of `occurrence/gpt` Issue 017.20. It changes neither the stable `decision_model` nor `topographo` public API and imports no optimizer, training, or Interact implementation.

## Result

**Task verdict: C∅ — unavailable under the evidence fence.**

The frozen abstract Fano point-line source passes the fixture gate. Exact enumeration recovers 168 incidence automorphisms and two point-line relation classes of sizes 21 and 28. However, current authority supplies neither:

1. a source-authorized map from that incidence semantics to a distinguished Theory-27 preparation; nor
2. a source-authorized formal OT test for `IncidenceDecision`.

The code therefore treats the tempting construction

```text
C_{r_I,r_N}(point, line)
    = r_I  when point is incident with line
    = r_N  otherwise
```

as a **conditional conformance witness**, not a task-level compiler. After constitutively importing the rank-(2,14) projector test from the 017.18 evaluator control and binding its slots, the ambient real residual family is `RP^1 × RP^13` (dimension 14); the executable exact-rational subset is `P(Q^2) × P(Q^14)` (countably infinite). Source equivalence conditionally removes 371 dimensions relative to independent fixed-test-compatible choices, but the full task residue cannot be quantified because test selection and the target gauge/action are not typed.

Two projectively distinct representatives produce the same conditional fixed-test outputs, but an explicit test-preserving coordinate symmetry maps one to the other. They reject C3 but do not settle C1 versus C2, because that test stabilizer is not certified as the full OT representation gauge. Choosing either basis-ray representative without the missing bridges remains a C0-style arbitrary embedding witness.

The matched control uses the **same** compiler entry point on a constructed schema that preserves both typed carriers, all 49 inputs, and the 21 raw rows as opaque payload while removing their incidence meaning. The compiler fails closed; the declared type-only symmetry grows from 168 to `7! × 7! = 25,401,600`, and all 49 inputs form one orbit.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `source_fixture.json` | byte-frozen source Type Schema and exact fixture | `20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c` |
| `audit_input_constitution.py` | exact source, conditional compiler, rival, control, and authority audit | `4b3508d04603a426750cdfeff77d6416d8b6b2210e5f499fe3831889b8710d29` |
| `test_audit_input_constitution.py` | focused exact and negative-boundary tests | `41eded429db406710e74cf6273963658efbd960157c528d7d2dd487143f99a22` |
| `input_constitution_audit.json` | canonical machine-readable C∅ audit | `48c95b16fd3d8f68fa20afcb67df5b54b9512e33380ffc610221d56446c0cf14` |

## Reproduce

Run from the repository root at base commit `853f0371f675d911a2988f63c3eaeb5d2901bf6e` or the result commit:

```bash
PYTHONPATH=decision-model/src uv run --frozen python issues/017-generalized-born-rule/017.20-Code-attachments/audit_input_constitution.py --check
PYTHONPATH=decision-model/src uv run --frozen pytest issues/017-generalized-born-rule/017.20-Code-attachments/test_audit_input_constitution.py
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.20-Code-attachments
```

The audit checks all 5,040 point relabelings, all 8,232 automorphism/input pairs per conditional representative, all 49 exact fixed-test evaluations, the explicit rival symmetry, source byte integrity, allowed compiler dependencies, and the relation-erased negative boundary. The committed JSON must match executable output byte-for-byte.

## Source-freeze qualification

The abstract incidence reduct predates this execution and is banked by GPT Issue 011. Its genealogy is FIPS-derived. The exact decision-bearing JSON was written and byte-hashed before the conditional compiler file in this execution, but it has no separate prior immutable preregistration. The audit records that limitation and does not use the local ordering as evidence for a positive result.

## Authority fence

- GPT task and reviews: `occurrence/gpt@e9ccdfdb1b4cdc6e342197d13b68c4d4c493997b495f4e83c5cceb553943bb60`, turns `017.20`, `017.20a`, and `017.20b`.
- Source reduct: banked GPT Issue 011 abstract Fano quotient at the same revision.
- Preparation carrier and evaluation: `occurrence/theory@b14aeb1bb9e07662e0e6fdde697f728cae3457324c0e2abd0cfd42975b8aac9a`, Theory 27.
- Formal effect/test typing: the same Theory revision, Theory 41.
- Additional representation-boundary audit: the same Theory revision, Theory 43.
- Conditional evaluator witness only: GPT `017.18`, implemented at repository commit `853f0371f675d911a2988f63c3eaeb5d2901bf6e`.

No result here asserts a physical Test Realization, canonical preparation, certified source-automorphism action on preparations, predictive result, training-economy gain, or claim outside this finite fixture and evidence fence.

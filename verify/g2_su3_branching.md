# G₂ ⊃ SU(3) branching — a runnable check for §10.1 / Conjecture C3

**Targets:** the §10.1 representation-theory audit obligation and Conjecture C3 /
Open Problem 3 — *does the 14-dimensional (adjoint-`𝔤₂`) sector decompose as
**8 ⊕ 6** under an SU(3) ⊂ G₂?*

This is a **recipe, not a completed review**: it computes the actual G₂ → SU(3)
branching so a reviewer can compare it against the paper's `8 ⊕ 6` conjecture.
It is pure representation theory about the symmetry group and does **not** touch
`topographo` or the released `.npz`.

## How to run

The script is [`g2_su3_branching.sage`](g2_su3_branching.sage). Either:

- `sage verify/g2_su3_branching.sage`, or
- paste it into **[SageMathCell](https://sagecell.sagemath.org)** (free, no
  login) and press *Evaluate*.

> Not yet executed in-repo (no Sage in CI). Run it yourself — no install needed.

## What to look for

- **Sanity check:** the `7` (standard rep) should branch as `3 + 3̄ + 1`
  (dims `[1, 3, 3]`), confirming this is the SU(3) under which the standard rep
  splits that way.
- **The real question:** what the `14` (adjoint) actually branches to, and
  whether it equals `8 ⊕ 6`.

**Expectation (to be confirmed by your run, not asserted here):** the textbook
branching of the G₂ adjoint under this SU(3) is `8 ⊕ 3 ⊕ 3̄` (dims `[3, 3, 8]`),
which is **not** `8 ⊕ 6` — the `6` is the symmetric SU(3) irrep, distinct from
`3 ⊕ 3̄`. If the run reproduces `8 ⊕ 3 ⊕ 3̄`, then C3's `8 ⊕ 6` either refers to a
*different* SU(3) embedding, reads `3 ⊕ 3̄` as a single 6-real-dimensional block,
or needs restating. That reconciliation is exactly the §10.1 audit call — the
point of the script is to put a computed decomposition in front of a specialist.

## Notes

- G₂ has (up to conjugacy) essentially one maximal `A2` = SU(3); the `"extended"`
  branching rule is the standard way to obtain it in Sage. If the paper intends
  a *non*-maximal or differently-embedded SU(3), swap in that embedding and
  rerun — the comparison logic is unchanged.
- A reviewer who confirms or refutes `8 ⊕ 6` should record the result as a
  proper reviewer cell, e.g. `verify/occurrence_ii_<yourhandle>.md`, citing the
  embedding used.

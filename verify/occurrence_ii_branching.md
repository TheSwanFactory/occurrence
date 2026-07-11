# G₂ ⊃ SU(3) branching — a runnable check for §10.1 / Conjecture C3

**Targets:** the §10.1 representation-theory audit obligation and Conjecture C3 /
Open Problem 3 — *does the 14-dimensional (adjoint-`𝔤₂`) sector decompose as
**8 ⊕ 6** under an SU(3) ⊂ G₂?*

This is a **recipe, not a completed review**: it computes the actual G₂ → SU(3)
branching so a reviewer can compare it against the paper's `8 ⊕ 6` conjecture.
It is pure representation theory about the symmetry group and does **not** touch
`topographo` or the released `.npz`.

## How to run

The script is [`occurrence_ii_branching.sage`](occurrence_ii_branching.sage). Either:

- `sage verify/occurrence_ii_branching.sage`, or
- paste it into **[SageMathCell](https://sagecell.sagemath.org)** (free, no
  login) and press *Evaluate*.

> Not yet executed in-repo (no Sage in CI). Run it yourself — no install needed.

## Result

- **Sanity check passes:** the `7` (standard rep) branches as `3 + 3̄ + 1`
  (dims `[1, 3, 3]`), confirming this is the canonical SU(3) under which the
  standard rep splits that way.
- **The adjoint branches as `8 ⊕ 3 ⊕ 3̄`** (dims `[3, 3, 8]`) — a gluon-like
  octet plus a quark-like `3` and antiquark-like `3̄`. The script asserts this.

This is **not** `8 ⊕ 6`: the `6` is the symmetric SU(3) irrep, distinct from
`3 ⊕ 3̄`. The paper's earlier `8 ⊕ 6` was an error; C3, Open Problem 3, and the
§10.1 obligation now read `8 ⊕ 3 ⊕ 3̄`. The corrected content is arguably a
*stronger* reading — `8 ⊕ 3 ⊕ 3̄` is exactly the gluon/quark/antiquark color
pattern — though its physical significance (canonical selection by the channel's
dynamics) remains the [CONJECTURE]-tier part of C3.

## Notes

- G₂ has (up to conjugacy) essentially one maximal `A2` = SU(3); the `"extended"`
  branching rule is the standard way to obtain it in Sage. If the paper intends
  a *non*-maximal or differently-embedded SU(3), swap in that embedding and
  rerun — the comparison logic is unchanged.
- A reviewer who confirms or refutes `8 ⊕ 6` should record the result as a
  proper reviewer cell, e.g. `verify/occurrence_ii_<yourhandle>.md`, citing the
  embedding used.

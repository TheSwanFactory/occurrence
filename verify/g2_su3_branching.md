# G₂ ⊃ SU(3) branching — a runnable check for §10.1 / Conjecture C3

**Targets:** the §10.1 representation-theory audit obligation and Conjecture C3 /
Open Problem 3 — *does the 14-dimensional (adjoint-`𝔤₂`) sector decompose as
**8 ⊕ 6** under an SU(3) ⊂ G₂?*

This is a **recipe, not a completed review**: it computes the actual G₂ → SU(3)
branching so a reviewer can compare it against the paper's `8 ⊕ 6` conjecture.
It uses [SageMath](https://www.sagemath.org) (which bundles the Lie-theory
machinery); it does **not** touch `topographo` or the released `.npz` — this is
pure representation theory about the symmetry group, independent of the channel
data.

> Not yet executed in-repo (no Sage in CI). Run it yourself — no install needed.

## How to run

Paste the snippet into **[SageMathCell](https://sagecell.sagemath.org)** (free,
no login) and press *Evaluate*, or run it in any Sage / CoCalc session.

```python
# G2 and its maximal SU(3) = A2 subgroup (from the extended Dynkin diagram,
# i.e. the "long-root" SU(3): the one under which the 7 = 3 + 3bar + 1).
G2 = WeylCharacterRing("G2", style="coroots")
A2 = WeylCharacterRing("A2", style="coroots")
rule = branching_rule("G2", "A2", "extended")

def show_branch(name, chi):
    print(f"{name}: dim {chi.degree()}")
    b = chi.branch(A2, rule=rule)
    dims = []
    for irrep, mult in zip(b.monomials(), b.coefficients()):
        d = irrep.degree()
        dims += [d] * mult
        print(f"    {mult} x {irrep}   (dim {d})")
    print(f"    -> dims {sorted(dims)}, total {sum(dims)}\n")
    return sorted(dims)

# Identify the fundamental reps by dimension (7 = standard, 14 = adjoint).
for labels in [(1, 0), (0, 1)]:
    print(f"G2{labels}: dim {G2(*labels).degree()}")
print()

# Sanity check on the embedding: the 7 should give 3 + 3bar + 1.
seven = G2(1, 0) if G2(1, 0).degree() == 7 else G2(0, 1)
adjoint = G2(0, 1) if G2(0, 1).degree() == 14 else G2(1, 0)
show_branch("7  (standard)", seven)
dims14 = show_branch("14 (adjoint 𝔤₂)", adjoint)

# Compare against the paper's C3 conjecture: 8 ⊕ 6.
print("C3 conjecture: 14 = 8 ⊕ 6  ->  dims [6, 8]")
print("Computed branching dims:", dims14)
print("Matches 8 ⊕ 6?", dims14 == [6, 8])
```

## What to look for

- **Sanity check:** the `7` should branch as `3 + 3̄ + 1` (dims `[1, 3, 3]`),
  confirming this is the SU(3) under which the standard rep splits that way.
- **The real question:** what the `14` (adjoint) actually branches to, and
  whether it equals `8 ⊕ 6`.

**Expectation (to be confirmed by your run, not asserted here):** the textbook
branching of the G₂ adjoint under this SU(3) is `8 ⊕ 3 ⊕ 3̄` (dims `[3, 3, 8]`),
which is **not** the same as `8 ⊕ 6` — the `6` is the symmetric SU(3) irrep,
distinct from `3 ⊕ 3̄`. If the run reproduces `8 ⊕ 3 ⊕ 3̄`, then C3's `8 ⊕ 6`
either refers to a *different* SU(3) embedding, or reads `3 ⊕ 3̄` as a single
6-real-dimensional block, or needs restating. That reconciliation is exactly the
§10.1 audit call — the point of this script is to make the discrepancy concrete
and put a computed decomposition in front of a specialist.

## Notes

- G₂ has (up to conjugacy) essentially one maximal `A2` = SU(3); the `"extended"`
  branching rule is the standard way to obtain it in Sage. If the paper intends
  a *non*-maximal or differently-embedded SU(3), swap in that embedding and
  rerun — the comparison logic is unchanged.
- A reviewer who confirms or refutes `8 ⊕ 6` should record the result as a
  proper reviewer cell, e.g. `verify/occurrence_ii_<yourhandle>.md`, citing the
  embedding used.

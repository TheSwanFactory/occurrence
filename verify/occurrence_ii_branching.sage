#!/usr/bin/env sage
# G2 ⊃ SU(3) branching — a runnable check for §10.1 / Conjecture C3.
#
# Targets the §10.1 representation-theory audit obligation and Conjecture C3 /
# Open Problem 3: does the 14-dimensional (adjoint-𝔤₂) sector decompose as
# 8 ⊕ 6 under an SU(3) ⊂ G2?
#
# This is a tool, not a completed review. It computes the actual G2 → SU(3)
# branching so a reviewer can compare it against the paper's 8 ⊕ 6 conjecture.
# Pure representation theory about the symmetry group — independent of
# `topographo` and of the released `data/kraus84.npz`.
#
# Run:
#   sage verify/occurrence_ii_branching.sage
# or paste into SageMathCell (https://sagecell.sagemath.org), no install needed.
#
# What to look for:
#   - Sanity: the 7 (standard rep) should branch as 3 + 3bar + 1, dims [1,3,3],
#     confirming this is the SU(3) under which the standard rep splits that way.
#   - The real question: what the 14 (adjoint) branches to, and whether it is
#     8 ⊕ 6.
#
# Result (confirmed by running): the G2 adjoint branches under this SU(3) as
# 8 ⊕ 3 ⊕ 3bar (dims [3,3,8]) — a gluon-like octet plus a quark/antiquark pair —
# NOT 8 ⊕ 6. The 6 is the symmetric SU(3) irrep, distinct from 3 ⊕ 3bar, so the
# paper's earlier "8 ⊕ 6" was an error; §10.1 / C3 now read 8 ⊕ 3 ⊕ 3bar. This
# script asserts that decomposition below.

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

seven = G2(1, 0) if G2(1, 0).degree() == 7 else G2(0, 1)
adjoint = G2(0, 1) if G2(0, 1).degree() == 14 else G2(1, 0)

show_branch("7  (standard)", seven)                 # sanity: expect 3 + 3bar + 1
dims14 = show_branch("14 (adjoint 𝔤₂)", adjoint)

# Assert the corrected decomposition and refute the earlier 8 ⊕ 6.
print("Computed branching dims:", dims14)
print("= 8 ⊕ 3 ⊕ 3bar  (dims [3, 3, 8])?", dims14 == [3, 3, 8])
print("= 8 ⊕ 6         (dims [6, 8])   ?", dims14 == [6, 8])
assert dims14 == [3, 3, 8], f"expected 8 + 3 + 3bar, got dims {dims14}"
print("\nCONFIRMED: 14 = 8 ⊕ 3 ⊕ 3bar (gluon octet + quark + antiquark).")
print("The paper's earlier 8 ⊕ 6 is refuted (3 ⊕ 3bar is not the symmetric 6).")

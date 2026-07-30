# Validation of the Solomon Joseph review

**Review commit:** `c738b4c7f8a1b211d3f145064eb3908c97089acd`

This note records an independent author-side cross-check before incorporating
the review into the Born Channel branch. The original review files remain
unchanged in the parent commit.

## Confirmed

- The reviewer script completed with 76 forced checks passed and none failed.
- `dim so(7) = 21`; the irrational eigenspaces are `7 ⊕ 7`, not `so(7)`.
- Multiplicities 21 and 42 are not irreducible G₂ representation dimensions.
  The computed eigenspaces decompose over `{1, 7, 14, 27}`.
- The second-moment identity gives an exact 14-operator mixed-orthogonal
  realization of the channel. Injectivity of `z ↦ L_z` on the pencil proves
  Choi rank 14.
- Self-adjointness does not imply non-primitivity; the peripheral `-1` mode
  does.
- The Firewall Theorem cannot include the Section 2 provenance claim.
- The released NPZ uses right multiplication in the stated doubling
  convention and is orthogonally similar to the paper's left-multiplication
  family.
- `Var[τ] = 1/18` follows exactly from the eventwise spectrum and the spherical
  fourth-moment identity.
- `1/8 + 1/147 = 0.131802721…`; the printed `0.131723` was an arithmetic error.
- Rank `12/16` is not a settlement survival probability.

## Confirmed with qualification

- The independent chain run reproduced `s* = 0.13183(4)`, consistent with the
  rational candidate. Numerical agreement is not an exact solution, so the
  revised paper keeps the derivation open.
- A complete search over all 136 quadratic coboundaries and the complete
  G₂-invariant polynomial basis through degree six found no pointwise
  stationary identity for `s* = 1/8 + 1/147`. The degree-six fresh-state RMS
  residual is `~1e-3`, not machine zero. Low-degree polynomial proofs therefore
  fail; rational or non-polynomial identities remain possible. See
  `verify/occurrence_ii_sstar_coboundary.py`.
- The reported quenched exponent changes materially with annihilation and
  restart conventions. The revision withdraws five-digit precision pending a
  canonical convention instead of selecting one estimate.
- The exact-over-Q layer of the reviewer script was skipped locally because
  `python-flint` was unavailable. The same characteristic-polynomial and
  representation claims are covered independently by the repository's Sage
  verification workflow.

## Resulting changes

The paper now distinguishes the forced 84-event crack from the channel's
minimal 14-operator representation, corrects its G₂ language, scopes the
firewall to Sections 3–7, clarifies data handedness, promotes the exact strain
variance proof, and replaces unsupported measured precision with
convention-aware statements.

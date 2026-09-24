# Issue 017.26 — deriving the useful quotient from typed structure, in two phases

Executed under the 017.26b owner execution contract, which supersedes 017.26 and
carries Canary's 017.26a review into it. The freeze is
`../017.26c-Kiro-phase-S-structure-freeze.md`; the result is
`../017.27-Kiro-derived-symmetry-from-typed-world-structure-result.md`.

## The question

017.25a showed a compiled quotient beating the declared relation, but the
symmetry was still handed to the compiler as a declared generating set. So the
result was consistent with a deflationary reading: a human supplied the useful
symmetry. This turn removes that. Nothing symmetric is declared anywhere. A
generic compiler reads a primitive's declared **kind and signature** and generates
its preservation constraint mechanically, then derives `Aut(A_Σ)` and the input
quotient before any target exists.

## The two-phase split, and why it is structural

017.26b §7 requires the structure freeze to be committed before `θ*` and label
seeds are introduced. Here that is enforced by the import graph rather than by
promise: `derived_symmetry.py` does **not** import the byte-pinned 017.24
implementation, because that module contains `generate_observations`.
`training_phase.py` is a separate file that pins `derived_symmetry.py` at
SHA-256 `e7074a60…` and the Phase S certificate at `f2e36998…`, so the group and
quotient the statistics are measured against are provably the ones banked before
`θ*` was chosen.

| | Phase S | Phase T |
|---|---|---|
| module | `derived_symmetry.py` | `training_phase.py` |
| dependencies | standard library only | pins Phase S, 017.24a, 017.24; NumPy in the learners only |
| commit | `08437b1` | this one |
| banked | `occurrence/gpt@57da798c` | see the result |

## Result

**Strong positive.** All seven 017.26b §11 criteria hold, all **135** frozen
regime-A predictions reproduce within 4σ (worst 3.38σ, zero failures), and
`D` was **bit-identical to `H` in 14,400 of 14,400** replicate comparisons.

| | Fixture A | Fixture B |
|---|---|---|
| carriers | `Point` 7, `Line` 7 | `Site` 9, `District` 3 |
| primitives | `incident` relation | `adjacent` relation **and** `district` operation |
| `\|X_Σ\|` | 49 | 27 |
| `\|Aut(A_Σ)\|` | 14 | 6 |
| `Q_aut` cells | 4 — 7/14/14/14 | 5 — 3/6/6/6/6 |
| frozen `L_k` reach | 21/28 | 9/18 |
| `D` = `H` dof | 4 | 5 |
| `R` dof | 2 | 2 |
| `R` exact floor | 0.040527 | 0.054445 |
| `D` reaches 0.02 | **n = 128** | **n = 128** |
| `R` reaches 0.02 | never | never |
| `U_best` reaches 0.02 | never | n = 1024 |

## The two worlds

Neither schema declares an automorphism group, a generator, an orbit, an orbit
cardinality, an equivalence over admitted inputs, a canonical representative, a
statistic-tying instruction, an expected group order, or any number that could be
a probability. Fixture A restates 017.24a's 21-row circulant incidence in a
generic typed signature with all of that deleted. Fixture B is new: a nine-site
ring corridor carrying a three-district map, whose invariance comes only from
**joint** preservation of two primitives of different kinds.

## Conditions

`D` derived structural compiler · `H` handed-quotient oracle, reading cell
membership out of the committed certificate by a path that never calls the
compiler · `R` the preregistered shortcut learner over the frozen `L_k` ·
`S` saturated · `P_without_X` the quotient one primitive can support alone ·
`W_wrong_compiler` cardinality-matched but respecting no derived map ·
`C_scrambled_compiler` · six uncompiled variants including `U_shortcut`, handed
the `L_k` feature values outright, plus the per-cell `U_best`.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `world_a_schema.json` | frozen single-relation world | `9f5262583012fb01aa4a9508d3abe3bfd9bb8135bd172d8342dc755004e7fc67` |
| `world_b_schema.json` | frozen multi-primitive world | `79412d9349c9413831bb44cae6bcd332ae265a6928facb2d7a7ec161c0b32b5d` |
| `derived_symmetry.py` | Phase S: the generic compiler, `L_k`, the perturbations | `e7074a60ed460b7b98d7f234e3af06aec169e5ae526e477c62e1e8799d0f2e3b` |
| `test_derived_symmetry.py` | 35 Phase S tests | `af7bf29eafa5eb384b576a9dc30792fa8df697f59c6b7c3e15e9ad3ce2b2b976` |
| `phase_s_certificate.json` | the structure freeze, 37 checks | `f2e36998f0e5715d785d5dd0a86790fc32604040334c8250d2fd6c845c29709a` |
| `phase_s_cost_report.json` | derivation cost | — |
| `preregistration.json` | `θ*`, seeds, every frozen prediction, the verdict rule | `55dc5c7a452d2b6ef4cb7429d4e0111ac54c598b317fc7006d576ba1f52735d1` |
| `training_phase.py` | Phase T: conditions, sweep, verdict | — |
| `test_training_phase.py` | Phase T tests | — |
| `phase_t_reference.json` | exact predictions, computed with no label in scope | `de0361f05be8c286166895f66a3faf9401edbde5f531b560373196622b6d4163` |
| `phase_t_results.json` | measured cells, verdict, claim ledger, 25 checks | `65f3ba4cc208d3f57e26d8cd2743b94e70927e39ea4eb67b3afb89c6a6255795` |
| `phase_t_cost_report.json` | cost accounting | `932970decc33d1d684cc0a1e0de70d88d015b418408238f4af35355b70e8954b` |
| `phase_t_per_replicate.csv.gz` | 129,600 raw per-replicate rows | `1fc8cf0bde5576c4ee046dea29328a1a9013d244956e6cb0138cd8403c87f6e8` |

## The bound on the cheaper-explanation claim, disclosed at the freeze

`L_k` does not recover either quotient. An iterated **pair refinement** over the
admitted inputs does, on both, in two iterations. That was computed and banked in
Phase S before `θ*` existed, so it is not a caveat found after the numbers were
visible. It is outside `L_k`, which is source-local and depth-bounded, and
017.26b §8 forbids `R` from computing any equivalent global structural quotient.
What it shows is that this compiler is more expensive than it needs to be on these
worlds — a tractability finding, not a cheaper statistic.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.26-Code-attachments/derived_symmetry.py --check
uv run --frozen python issues/017-generalized-born-rule/017.26-Code-attachments/training_phase.py --predict
uv run --frozen python issues/017-generalized-born-rule/017.26-Code-attachments/training_phase.py --check
uv run --frozen pytest issues/017-generalized-born-rule/017.26-Code-attachments
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.26-Code-attachments
```

`--predict` emits every frozen prediction without generating a single label.
`--write` regenerates the artifacts; `--quick` runs a reduced grid and refuses to
write. A full Phase T run is about five minutes; Phase S is about five seconds.

## Fences

`Q_aut` is **one** preregistered automorphism-orbit compiler, not the universal
TDM quotient; definability, context, congruence, bisimulation-like, and
task-relative constructions are untouched. `D` and `U` receive the same source
information, so the measured gap is inductive and computational leverage from
deterministic compilation under the tested learners — not an information
asymmetry and not a learner-independent sample-complexity bound. No language-scale
claim, no comparison to any language model, no physical Test Realization, no
Interact. OT is not claimed to be responsible for any measured gain. Neither
fixture is representative of anything beyond itself. Issue 018 is not resolved.
Nothing was added to the stable `decision-model` public API.

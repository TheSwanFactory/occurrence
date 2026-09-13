<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file.

## [0.8.4] - 2026-09-12

### Added

- Issue 011, the first cross-habitat discovery experiment, as five modules in
  `experiments/sfp_representation/`: `fano_task.py` (Gate 0), `fano_heads.py`
  (the learner), `fano_baselines.py` (exact ceilings and cheap selectors),
  `fano_sweep.py` (the declared runs and the recovery audits) and
  `fano_analysis.py` (the predeclared thresholds and the disposition), with six
  artifacts under `011_fano_artifacts/`. `009` banked *local* representation
  discovery and said explicitly that the global question was untested: every
  habitat is the same six-node octahedron once tokens are permuted, so fourteen
  disconnected local charts contain no global Fano identity. `011.01` asks whether
  the seven-point Fano / `FFF` quotient can be recovered from fourteen
  **anonymous** habitats and certified cross-habitat support incidence, with no
  `FFF` and no XOR law supplied.
- Gate 0 establishes that it can. The learner's whole input is a typed anonymous
  graph — 14 habitat nodes, 84 Event nodes, 7 first-half support-axis nodes, 7
  second-half, and the `H--E` / `E--L` / `E--R` incidence — and the section 3.1
  candidate reconstruction is verified step by step rather than assumed: each
  habitat omits exactly one L and one R node, equal support patterns give seven
  two-habitat classes, the omitted pairs induce an L→R bijection, each class
  exposes three reciprocal support pairs, and adjoining the class deduplicates to
  exactly seven three-point lines with every one of the 21 point pairs on exactly
  one line. `|Aut(recovered plane)| = 168` and it is equivalent to the certified
  `topographo.core.f2_groups.LINES3` under 168 isomorphisms.
  `|Aut(observed anonymous graph)| = 43008 = 168 x 128 x 2`, derived by
  enumerating all 5040 L-images — exactly 168 extend, each in exactly 128 ways —
  plus a verified L/R namespace swap. Both answers are forced by the observation:
  zero equivariance failures over all 43008 automorphisms.
- The exact ceilings, including the two that are rational numbers rather than
  measured chance. The nonlearned reconstruction recovers the mate, the completion
  and the whole seven-line plane at exactly `1.0000` in every frozen namespace.
  For the cross-habitat-destroying control, `011.01` section 3.3 forbids
  substituting empirical chance, so `localized_ceiling_census` *constructs* an
  automorphism of the control graph fixing the query and carrying one candidate
  answer to another — 168 witnesses for the mate query, 924 for completion — which
  puts every scored candidate in one orbit and pins the ceilings at `1/13` and
  `1/66`.
- Disposition **C**, filed as such. The primary block reaches
  `mate_exact_accuracy 0.8750` CI `[0.6250, 1.0000]` and
  `fano_completion_exact_set_accuracy 0.7206` CI `[0.3750, 0.9706]` against
  thresholds of `0.95`, with `fano_plane_valid_rate 0.7344` against `0.90` and
  fresh-name decision transport `0.8455` against `0.99`. The seed distribution is
  bimodal and the mean describes neither mode: five of eight seeds reach final loss
  `0.346574`, the exact information-theoretic minimum of this loss, and score
  `1.0000` on every metric including the whole-plane reconstruction and transport;
  three finish at the uniform-choice loss. Those three are **fit** failures, not
  generalization failures — train and test agree to within `0.0149` on every one of
  them. The one repair `011.01` section 4 permits was diagnosed, run, and made
  things worse, moving converged seeds from 5 to 3 and completion to `0.4235`; both
  blocks are reported and the primary is not deleted. Section 8's "do not round a
  near miss into A" is enforced structurally: the class is derived from the
  threshold table by explicit conditions rather than typed by hand.
- The controls hold. The capacity-matched localized control — identical parameters,
  identical protocol, L/R identity localized inside each habitat so every
  within-habitat pattern and degree is preserved and no axis is shared across
  habitats — sits at `0.0804` and `0.0208` against exact ceilings of `0.0769` and
  `0.0152`, within the declared `0.01` tolerance, and recovers the plane at
  `0.0000`. In the main arm both structure-aware cheap selectors score exactly
  `0.0000` on the completion query, while a two-line shared-axis count scores
  `1.0000` on the mate query — so the result turn says plainly that Query A is the
  easy half and Query B carries the evidence.
- Mechanical fences rather than prose ones: an AST audit over the learned path's
  code, with annotations, docstrings and the ban list's own strings excised, that
  finds no hidden certified label, no Gate-0 reconstruction, no certified Fano
  plane, no SFP codec and no XOR; bitwise swap invariance for the completion query
  from writing it as a set mark; tie-aware fresh-name covariance, checked on a
  deterministically perturbed model because at initialization every habitat score
  is tied to about `1e-7`; a parameter count identical at 11, 40, 112, 266 and 300
  nodes; and a check that the sparse edge-message path agrees with the dense one to
  `4.8e-07`, so the fast path is an optimization and not a second model.
  `fano_heads.PRE_SWEEP_VALIDATION` records all eleven configurations tried while
  freezing the architecture, read on training fit only.

## [0.8.3] - 2026-09-12

### Added

- Promote the Issue-009 abstract `F_2` group machinery into the package as
  `topographo.core.f2_groups`: the Fano plane `PG(2,2)`, `GL(3,2)`,
  `GL(2,2)`/`AGL(2,2)`, the `K4` edge action, and `is_structure_preserving` — the
  predicate separating the 24 induced edge permutations from the 48 that merely
  preserve the opposite-edge matchings. It lands in `core` rather than `ssd`
  because it is label machinery: stdlib-only, no Events, no sedenions, no numpy,
  which `topographo/tests/test_f2_groups.py` checks against the module's own
  import graph rather than its docstring. `certificate()` re-derives every order,
  incidence count and relational law; `assert_f2_group_laws()` also checks the
  frozen digest. `experiments/sfp_representation/groups.py` keeps only the
  artifact driver, and `009_artifacts/groups.json` re-derives with every order,
  digest, pin and law bit-identical — one prose line changed. This discharges the
  "a later task will generalize them" note in that experiment's Gate 0.
- Run Issue-009 opaque local-chart discovery (`009.09` task, `009.10` result):
  the three earlier turns all *supplied* the SFP code; this one withdraws it and
  asks whether the learner can **recover** it inside one supplied `(S, FFF)`
  habitat. Five new modules in `experiments/sfp_representation/`
  (`discovery_task.py`, `discovery_heads.py`, `discovery_baselines.py`,
  `discovery_sweep.py`, `discovery_analysis.py`). The whole input surface is six
  anonymous Event tokens, identical before marking, plus the pairwise ADMIT
  matrix and one certified cyclic-block triple — no `PP`, `pp`, SFP token, Event
  or habitat index, embedding table or buffer.
  **Disposition B, with A named as the alternative reading.** Held-out forced
  third **0.9970** (LOHO, CI [0.9911, 1.0000]) and 0.9940 (LOFPO) against an
  admission-only ceiling of `1/2` derived twice — Bayes mass over the two
  compatible block systems, and an automorphism of the admission graph that
  forces any permutation-equivariant scorer to tie. Gate 0 pins `|Aut| = 48`, a
  24-element induced subgroup, 2 admission-compatible block systems and a
  `1.0000` one-anchor ceiling; all 13 pins and 18 laws hold in all 14 habitats.
  Filed B rather than A only because the three chart thresholds fail under the
  reconstruction `009.09` section 8 refers to and pass at 0.9911 under the
  supplied-anchor one; which reading governs is an Owner decision, and section 9
  forbids repairing the question after results are visible.
  **The missed threshold is provably unreachable, and that is computed rather
  than argued**: the reconstruction needs answers on the three pairs inside the
  anchor, which section 4.4 makes support rather than scored queries, so
  section 5.3 never supervises them. Two rules over the supplied atoms agree on
  all 6048 supervisable rows and on no support row, so no learner obeying the
  task's own training rule can meet those thresholds, whatever its architecture.
  The causal claim rests on a control at identical capacity: the zero-anchor arm
  is the same architecture at the same 11617 parameters with the anchor channel
  held at zero, and scores 0.493716931217 — bit-identical to the
  `lowest_common_neighbour` cheap selector and identical across all 8 seeds on
  all 14 folds, because the two candidate scores are equal by the Gate-0
  automorphism and `argmax` breaks the tie by index. Paired, +0.5033
  (CI [0.4965, 0.5075], 14/14 folds).
- Run Issue-009 query-relative `PP` localization (`009.07` task, `009.08`
  result): `009.06` left locating the shared block as the binding constraint, so
  this turn replaces the absolute chamber label with a query-relative pointer.
  Five new modules (`locator_task.py`, `locator_heads.py`,
  `locator_baselines.py`, `locator_sweep.py`, `locator_analysis.py`).
  **Disposition B — material improvement, one condition short of A.** Held-out
  forced third rises to **0.9688** (LOHO) with `PP` going 0.7158 -> 0.9777 and
  `pp` 0.9762 -> 0.9896; Rung 1 meets `009.07` section 4.4 outright at 0.9993.
  The pointer adds +894 parameters on the `PP` pathway, so a control carries the
  causal claim: paired against `009.06`'s repair block, `B_sfp` gains +0.2671
  (CI [0.2225, 0.3118], 14/14 folds) while `C_scrambled` — identical
  architecture change, identical parameter count — does not move at all
  (-0.0015, CI includes zero). That rules out a representation-independent
  capacity effect without decomposing B's gain into constitution and capacity
  parts, which would need a capacity-matched pointer the one-family stop rule
  does not license. Class A is missed on exactly one condition and is not
  rounded up: Rung-2 `D - B` is -0.0283 (CI [-0.0573, -0.0007]), clearing zero
  by 0.000744 — two of the 0.000372 quanta the paired statistic can express —
  while every component field, every LOFPO interval and Rung-1 `D - B` (exactly
  0.0000) include zero. The artifact records `conditionally_ready` and names the
  blocking item rather than assigning itself a letter. One reporting correction
  came out of implementation: a subset counts as a ceiling if **either** the
  measured determining test or the declared cost rule flags it, because
  `endpoint_pair` reaches 1.000 on the admitted-only Rung-1 test while not
  determining `q*` over the whole pool, and bucketing on the measured test alone
  would have invented a bar the learner appeared to clear.
- Run Issue-009 corrected consequence-address ladder (`009.05` task, `009.06`
  result): is the `009.02` forcing negative a property of the SFP representation,
  or of prematurely expressing consequence as one of 84 opaque public Event
  identities? Six new modules — `ladder_task.py`, `ladder_heads.py` (typed field
  heads that **import** `scorer.TokenEncoder` and `scorer.SymmetricPair` rather
  than reimplementing them, so the input pathway is provably the computation that
  produced the `009.02` numbers), `ladder_resolver.py`, `ladder_baselines.py`,
  `ladder_sweep.py` and `ladder_analysis.py` (which reuses `009.02`'s seeded
  bootstrap, so the two results compare as statistics rather than as numbers).
  Executes `009.05`; `009.04` was superseded unexecuted and both facts are
  machine-checkable fields in `ladder_task.json`.
  **Disposition C, partial.** Changing **only the output object** — 85 catalogue
  slots to 17 typed field slots, at capacity spread exactly 0 and with the frozen
  splits, arms, seeds and protocol unchanged — moved held-out
  `positive_forced_third_exact_accuracy` under the exact SFP code from **0.0335
  to 0.7016** against a `1/84` chance floor, with both structural signatures
  `009.05` section 5.6 requires: `B - C` = +0.5082 (CI [0.4583, 0.5595],
  sign-consistent across all 14 LOHO and all 7 LOFPO folds) and `D - B` =
  +0.0506 (CI [-0.0052, 0.1049], includes zero). Native coordinates are not
  sufficient: `A_native` reaches 0.4412, `B - A` = +0.2604.
  Three results caution against over-reading the headline. **The catalogue
  interface was a cause, not the only one**: learning all four address fields
  still fails at 0.0097, and it fails on the fields the certified relation
  *copies* rather than the ones it computes — `pp` reaches 0.9948 while `FFF`
  collapses to **0.0320**, which is closed-set habitat re-identification under a
  structural holdout, not consequence forcing. **The obstruction moved rather
  than vanished**: within the permitted repair block `pp` is 0.9762 and `PP`
  0.7158, and 0.9762 x 0.7158 = 0.6987 against an observed 0.7016, so locating
  the shared block is the binding constraint. **A nonlearned tabulation beats the
  learner**: the code-space shortcut search is new to this issue, since
  `009.02`'s nine atoms are all *native* and no search over the features this
  learner actually sees had ever been run. Over 12 atoms and 298 subsets, the
  certified consequence is determined under the exact code by a **two-atom** key
  with zero ambiguity over all 7056 ordered pairs, reaching 1.0 on held-out
  habitats, while under the non-automorphic scramble no subset up to size three
  determines it at all. The learner therefore sits between the best
  strictly-cheaper rule (0.500) and a ceiling the representation itself attains
  (1.000). Rung 3 is exact by construction: the SFP alphabet is **saturated**
  (2 x 7 x 6 = 84 = |Events|), so resolution is total and injective, resolved
  Event-identity accuracy is *identically* equal to exact structured-address
  accuracy in all 672 runs, and 0 of 4032 single-field corruptions are ever
  repaired. Rung 1 is excluded from the disposition by construction: its label is
  a function of the unordered input port pair, which takes three values under the
  exact code, so a three-row lookup reaches 1.0 and `GL(2,2) ~= S3` makes every
  relabeling of that quotient a symmetry.
- Add `experiments/sfp_representation/` for Issue-009 native-FIPS versus SFP
  consequence representation (`009.01` task, `009.02` result): does the compact
  `S | FFF | PP | pp` code carry the certified admit-and-force relation better
  than the native FIPS realization, and is any advantage relational rather than a
  matter of code capacity or coordinate choice? Eleven new modules — torch-free
  `conformance.py` (Gate 0: 103 Outcome-`021.05` pins re-derived, 0
  disagreements, 14 named relational laws, `TABLE_SHA256` `eb31fba3dbc3a4bb…`),
  `sfp.py` (the codec plus an exactness proof over the complete finite domain),
  `groups.py`, `task.py` (the frozen 7056-ordered-pair pool, `dataset_sha256
  7872755f…`), `folds.py` (22 manifests, `fold_manifest_sha256 27b01b5e…`, zero
  leakage on all 21 structural folds recomputed from the recorded index lists
  rather than asserted), `arms.py` and `baselines.py`; torch-only `scorer.py`,
  `harness.py`, `run_sweep.py` and `analysis.py`, which per repo convention are a
  manual run and are not in CI.
  **Two finite pins are newly certified: 112 oriented cyclic blocks and 168
  Event-block incidences**, computed nowhere in the repo before this issue and
  derived *structurally* — oriented blocks as genuine cyclic-sense classes,
  incidences by walking actual membership — rather than as `2*56` and `84*2`,
  since the arithmetic shortcuts would assume exactly the regularity the pins are
  meant to witness.
  **Disposition E, explicitly narrowed.** Forcing is unsolved by every
  representation: the best forced-third accuracy over all 1400 runs is 0.0335
  (0.0900 after the single permitted repair) while both nonlearned oracles score
  1.0. The **admission half is cleanly resolved**: the SFP code beats its own
  marginal-matched, structure-destroying scramble by **+0.6369** on
  same-habitat-disjoint accuracy (CI [0.5655, 0.7054], sign-consistent across all
  14 primary folds, with bit widths, habitat sizes and field marginals held
  identical) and beats the native realization by **+0.3772** on admission
  balanced accuracy, where native realization sits at **chance** (0.5048).
  Relabeling under `GL(3,2)` + affine `AGL(2,2)` + a global `S` flip matches the
  code arm on every metric with every CI including zero, so the admission result
  is not a basis artifact. Capacity spread across the four representation arms is
  **exactly 0** (20323 parameters each) and swap symmetry is 0.0 bitwise.
  Three findings constrain the benchmark. The scramble's rejection proof had to
  test membership in the 24-element induced permutation class, because
  matching-partition preservation is necessary but **not** sufficient — 48 of 720
  permutations preserve the partition while only 24 are induced, and two accepted
  scrambles are partition-preserving with **zero** admission disagreements yet
  lose **all 24** forced thirds, since a non-induced partition-preserving
  permutation carries a K4 star onto a K4 triangle. The mechanical leakage search
  (2838 fits) found all 50 cheaper-than-the-relation shortcuts on the random
  control and none on either structural fold family, and every
  relation-equivalent feature determines admission **only**: the best cheap subset
  reaches 0.921 admission balanced accuracy while scoring 0.0 on forced third and
  0.0 on same-habitat-disjoint accuracy, making admission balanced accuracy a weak
  discriminator. And two arms reach high non-admission accuracy purely by
  answering BOTTOM to ~87% and ~93% of admitted pairs, so non-admission accuracy
  is only readable together with admission sensitivity.
- Run Issue-008 Ladder D (`008.07` task, `008.08` result): does the `008.05`
  learned-composition advantage survive train-only **recovered** Event
  denotations? New torch-free `experiments/tlm_multitoken/recovery.py` (frozen
  sign-balanced 16-token layer, streamed endpoint-inverse voting recovery,
  mechanical leak audit, `008.07` section 6 diagnostics) plus
  `run_ladder_d_00807.py` and `probe_degraded_recovery.py`; CI pins in
  `topographo/tests/test_multitoken_recovery.py`.
  **Positive on both discriminative splits** — recovered `motif` 0.6670 against a
  0.1495 bar (+0.5175, 8/8 seeds), `motif_parity` 0.9943 against 0.1496 — but the
  commissioned robustness measurement is **degenerate**: voting recovers all 16
  tokens exactly with zero ties and a minimum vote margin of 9105, so the
  recovered arm is bit-identical to the true arm and the recovery penalty is
  `+0.0000` as an identity, not a measurement. The driver detects this and sets
  `degenerate_as_a_robustness_test`. Two added diagnostics locate the real
  boundary: exact recovery needs only 876 of 2921 available observations, and the
  learned advantage survives at 14/16 recovered tokens but is gone by 7/16.
  The recovery penalty is almost entirely a *ceiling* effect — policy success as a
  fraction of what recovery leaves reachable holds at 0.63–0.67 across 7/16, 14/16
  and 16/16 recovery — so recovery quality, not policy learning, is what degrades.
  Ladder D also transferred the banked `017` `Rec_ray` method by **streaming** it:
  the inverse map has 14112 entries at `E^2 R` but 11854080 at `E^3 R` with twenty
  programs. At eight seeds the `motif` cell gives 0.6670 (sd 0.172) against the
  banked three-seed 0.7595, whose sample was bimodal.
- Run Issue-008 Ladder A (`008.04` task, `008.05` result): a learned strict
  program-selection policy at `E^3 R` with `Occ+Cyc+Sand`, against every
  deterministic baseline, under exact native endpoint scoring. New torch-free
  `experiments/tlm_multitoken/task.py` (frozen task, `008.02` sign-bit target,
  four digest-pinned splits, the `008.03` section 2.3 viability gate, eight
  baselines) and `ambient.py` (the fenced Ladder-C ambient generalized-`Mul`
  control); torch-only `policy.py` and `run_learning_00804.py`; CI pins in
  `topographo/tests/test_multitoken_task.py`.
  **Positive on the primary compositional split**: held-out exact-native success
  0.7595 against 0.1377 for the best deployable deterministic baseline, material
  on 3 of 3 seeds, above even the non-deployable within-split availability oracle
  at 0.5203. A flat 20-logit head scores 0.0018 there because two program labels
  never appear as a training target; the one allowed repair replaces the free
  per-program output column with a fixed, target-agnostic term-calculus basis.
  Two findings constrain the benchmark. The `008.02` target is a function of the
  three Event sign bits alone, so an eight-entry deterministic lookup table
  scores **1.000** on any split that does not withhold whole sign patterns —
  making the i.i.d. and held-out-Event splits controls rather than claims, and
  `008.04` section 6 necessary but not sufficient. And withholding the
  homogeneous patterns `000`/`111` hands half the held-out set to ForceSeq by
  construction (measured 0.50), since the `000` target *is* the right comb; the
  split withholds `011`/`100` instead and the constraint is now pinned. Ladder C
  is materially *worse* than native (0.4220), and Ladder D is warranted but not
  run. The `008.02` exhaustive `E^3 R` sweep was re-run from scratch and
  reproduces the published artifact exactly.
- Add CI-safe regression pins for the Issue-007 `007.04` p=13 Theory-41 word-slot
  coarse-grain probe in `topographo/tests/test_p13_coarse_grain_00704.py`: the
  11/13 distinct class-count rows, the `{1,12}` / `{2,11}` effect collisions,
  `span_rank(e_c)=4`, and the strict-argmax survey counts.
- Add `experiments/tlm_multitoken/` for Issue-008 multi-token native composition:
  the `008.01` plan and a machine-checked reporting contract in `reporting.py`.
  Arms declare which interface layer they exercise (`supplied_tree` /
  `selected_tree` / `recovered_dens`, the `007.05` layering), and validation
  refuses a claim whose layer was never exercised, an arm that both supplies and
  selects a bracketing, a tree-selection claim against an availability-rule
  target (the `017.24` GroupedFirst trap), a learned policy reported without the
  `008.01` section 8 baselines, and the `007.05` section 8 decoder invalidity
  conditions. No learning code yet.
- Add `experiments/tlm_multitoken/probe_program_space.py`, the `008.01` section 3
  signature search, with its exhaustive artifact `program_space_report.json` and
  the result write-up `008.02-signature-search-result.md`. Condition 3 is settled
  by an **availability ceiling** — the best accuracy reachable by any function of
  the availability pattern, which bounds GroupedFirst, ForceSeq, and the random
  legal selector at once — rather than by beating one hand-written rule.
  Exhaustive over all 7056 `E^2 R` and 592704 `E^3 R` inputs at `r = e4`.
  Finding: with `Occ` and `Cyc` alone, 93% of `E^2 R` and 88% of `E^3 R` inputs
  admit exactly one legal program, so the ceiling is 0.973 and 0.936 and nothing
  through `E^3 R` qualifies; adding `Sand` lifts the joint domain to 98.6% and the
  headroom to 0.733 at `E^2 R` and 0.856 at `E^3 R`. This narrows `017.24`
  section 3: the GroupedFirst coincidence was real, but the deeper cause is
  Occ/Cyc-only program-space sparsity, and no target construction can defeat an
  availability rule there.
- Promote `Sand` to the exact rational path in `experiments/tlm_multitoken/native.py`
  alongside `Occ` and `Cyc`, completing the three certified `017.04` constructors
  on one evaluator. It previously existed only as a float helper in
  `probe_futurator_program_space.py` and so could not be scored. Ill-typed terms
  (wrong role in a slot) raise, while undefinedness (annihilation, or `Cyc` off the
  336 admissible pairs) is returned as `None` and reported.
- Add `experiments/tlm_multitoken/programs.py`, the planar role-typed term
  calculus from `017.04` as an enumerable program space. Shape counts match their
  closed forms: `C(2n,n)` programs for `E^n R` with the full constructor set
  (1, 2, 6, 20, 70, 252) and the Catalan numbers when `Sand` is withheld
  (1, 1, 2, 5, 14, 42). The `E^2 R` Occ/Cyc subset is exactly the 017
  two-program family `P_seq` / `P_grp`.
- Add `topographo/tests/test_multitoken_native.py`: cross-checks `occ` / `cyc`
  against the 017 exact evaluator, rebuilds `P_seq` / `P_grp` from the
  constructors, and pins the full 84x84 `Sand` census (6720 defined, 336
  undefined, 1008 degenerate, 5712 non-trivial, and zero coincidences with `Occ`).
  Records that the Theory-065 edge identity forces `Sand(e,r) = [r]` on all 336
  admissible ordered pairs, so `Sand` is dead on the `Cyc` domain.

### Changed

- Make `experiments/tlm_fixed_head/probe_p13_coarse_grain.py` reproduce its own
  published `007.04` artifact. The script previously emitted a reduced schema and
  selected winners with `argsort(...)[-1]`, which credits the highest index of a
  tied maximum; results 2 and 11 carry identical effects, so that reported 6648
  phantom wins for result 11. Winners are now credited only when the maximum is
  attained by exactly one result, tie sets are reported separately, and the
  script emits the published verdict / duplicate-group / singular-value /
  `fraction_unique_argmax = 0.6676` fields.
- Extend the `topographo` workflow path filters and the explicit `ruff check`
  allowlist to cover `experiments/sfp_representation/` as a directory glob, and
  add `topographo/tests/test_f2_groups.py` to the named test allowlist. The
  torch-only Issue-009 modules (`scorer.py`, `harness.py`, `run_sweep.py`,
  `analysis.py` and the three `*_sweep.py` / `*_heads.py` families) are linted but
  stay a manual run and are not import-gated, following the Outcome-017 and
  Issue-008 convention.
- Export `f2_groups` from `topographo.core` and widen that subpackage's stated
  scope from Cayley-Dickson primitives to include abstract finite structures on
  the same terms: labels only, no Occurrence-Theory vocabulary.
- Re-certify the seven Issue-009 artifacts that record `topographo_version`
  (`conformance_audit`, `sfp_codec`, `task_dataset`, `folds`, `arms`,
  `baselines`, `scorer`) under `0.8.3`, since each `--check` byte-compares the
  whole rendered artifact and would otherwise fail on any release. They are
  deterministic re-derivations rather than measurements, so re-running them is a
  genuine re-certification: the only differences are the version string itself
  and, in `scorer.json`, the payload digest that covers it. Every structural
  number, count, digest and law is unchanged, and all 23 replay gates across the
  four turns pass byte-identically under `0.8.3`. The gitignored bulk sweeps keep
  the version their runs were executed under; they are hosted in the `protology`
  Quilt package and re-running them is a multi-hour job, not a fetch.

## [0.8.2] - 2026-09-08

### Added

- Add Issue-006 two-step Fixed-head **existence adapter** in
  `topographo.ssd.fixed_head`: Fixed 05b generators `g0,g1,g3,g4`, Kraus lifts for
  the 84 basic Events, HS Fixed projection `E_Fix`, Theory-27 readout
  `P(w|[x])=<x,B_w x>/<x,x>`, and the Owner-accepted 006.12 census pins
  (7056 words → 15 Fixed classes, span dim 4, one-step firewall, normalization).
- Add `experiments/tlm_fixed_head/` capacity / usefulness probe answering 006.13
  for modular arithmetic (`p=13` CI, `p=97` science): alphabet collision / MI
  report, linear readout smoke vs softmax scaffolding, and an explicit verdict on
  whether the 15-class bottleneck is fatal.
- Add package tests for the Fixed census and CI-safe capacity flags (torch not
  required for default CI).

### Changed

- Export `fixed_head` from `topographo.ssd` alongside the Futurator-gate and
  TLM-1 adapters without widening the `exact_machine` facade.

## [0.8.1] - 2026-09-08

### Added

- Add Issue-005 Milestone T (TLM-1) smoke harness under `experiments/tlm_modular/`:
  capacity-matched baseline A, exact-FIPS TLM B, and matched non-isomorphic
  structural control C, with pinned init/reset/staging/admission/adaptation
  config, alternate-role triple partitioning, and wall-clock logging.
- Add `topographo.ssd.fips_adapter`, the hard FIPS index/tensor path and declared
  STE learning-adapter smoke (exact hard forward matches `fips_basic` fixtures;
  no optimizer imports in Outcome/Futurator runtime modules).
- Add `topographo.ssd.structural_control`, degree-matched rewiring control with
  mechanical C1–C5 certificates (event/pair counts, degree histogram, destroyed
  Pasch/cyclic closure, not mere relabeling, documented broken laws).
- Add package tests for structural control C1–C5, FIPS adapter conformance, and
  alternate-role data partitioning (torch not required for CI unit tests).

### Changed

- Export `fips_adapter` and `structural_control` from `topographo.ssd` alongside
  the Futurator-gate modules without widening the `exact_machine` facade.

## [0.8.0] - 2026-09-08

### Added

- Add `topographo.ssd.outcome_runtime`, the Issue-005 Futurator gate: Outcome-aware
  `ask(presentation, admissibility, enacted_law)` yielding `ConstitutedOutcome` or
  typed `NonAdmission`, with `NativeLeftAction`, `ProjectiveNativeLeftAction`, and
  `FipsClosure` laws. Projective `zx=0` constitutes `AnnihilationBoundary` (not
  NonAdmission). Sealed Event frames are `EventDenotation`; `OperationalFrame`
  remains the Theory 068.02 unit in `frames`.
- Add `topographo.ssd.fips_basic`, the certified finite basic FIPS table (84 Events,
  56 blocks, 336 ordered pairs) regenerated from the frames Event/edge predicates,
  with forced unique `third` lookup, cyclic block closure, and a pinned SHA-256
  checksum.
- Add `topographo.ssd.seal` with `classify` / `seal` / `resolve` and a Theory-39
  public-collision witness: identical FIPS `(P, delta)` class, distinct sealed
  evaluations; classify-only `PublicEventView` cannot execute Event action.
- Add package tests for the 005.04 §1 Futurator acceptance suite (Outcome, FIPS,
  and seal witnesses).

### Changed

- Export `outcome_runtime`, `fips_basic`, and `seal` from `topographo.ssd` alongside
  the existing exact machine layers and `frames` without widening the
  `exact_machine` compatibility facade.

## [0.7.0] - 2026-09-07

### Added

- Add `topographo.ssd.frames`, the Theory 068.02 mutual Operational-Frame
  machine: Native/Cyclic retained state, certified Event/edge predicates, seed
  guard, local `step`/`advance`, and explicit serial/snapshot `round_step`
  policies over the released exact projective engine.
- Add package tests for the seven 068.02 behavioral witnesses (exact SCE
  ordering, guard/eigenspace agreement, unavailable presentation, explicit law
  choice, cyclic autonomy, and kernel non-retention).
- Add `experiments/mutual-frames/` with the finite census audit, compressed
  traces, ordering metrics, and Quilt/Theory 068.02 provenance notes. The
  multi-minute full census remains a manual `audit.py` run, not default CI.

### Changed

- Export the `frames` module from `topographo.ssd` alongside the existing exact
  machine layers without widening the `exact_machine` compatibility facade.

## [0.6.0] - 2026-09-04

### Added

- Add exact rational arithmetic backed by the shared Cayley–Dickson signed-basis
  table, typed expression and program nodes, a policy-driven transition engine,
  projective-state handling, trace observers, and an isolated JSON codec.
- Add direct tests for ordered left action, shared raw/projective execution,
  projective equivalence (including negative scaling), annihilation, and layer
  boundaries.
- Add a pinned Ruff lint gate to the package workflow.

### Changed

- Split `topographo.ssd.exact_machine` into explicit algebra, program, machine,
  projective, observer, and codec layers while retaining the original module as
  a compatibility facade.
- Rename projective execution to `run_projective`; retain `run_ray` as a
  compatibility alias and document equivalence under every nonzero rational
  scalar.
- Use one `Completed` result for raw and projective success, keep
  `Annihilated` explicit, and derive squared norm outside transition data.
- Preserve the deterministic experiment result schema and make `--check`
  validate both `results.json` and `observations.md`.

## [0.5.0] - 2026-09-04

### Added

- Add `signed_basis_table()` as the canonical immutable integer specification
  for Cayley–Dickson basis multiplication.
- Add checked `to_core_coordinates()` and `from_core_coordinates()` helpers for
  the explicit 061.11 map `Phi(a, b) = (conj(a), b)`.

### Changed

- Derive both the NumPy structure tensor and exact-rational sedenion products
  from one signed-basis table while preserving historical core coordinates and
  exact-machine outputs.
- Move basis crack construction and pure-pair sampling from generic
  `CayleyDicksonAlgebra` to the dimension-specific `SedenionAlgebra`; callers
  using those helpers should migrate to the specialized class.
- Document the core and 061.11 forms as coordinate presentations of one
  Cayley–Dickson algebra.

## [0.4.2] - 2026-09-04

### Changed

- Publish `topographo` from the tested `python-dist` artifact when a forward
  PEP 440 version bump reaches `main`, rather than requiring a manually pushed
  tag.
- Validate release versions against the prior commit, the editable `uv.lock`
  entry, a dated changelog heading, built distribution metadata, and package
  index availability before publication.
- Keep PyPI Trusted Publishing, serialize uploads, and create the matching tag
  and GitHub Release only after PyPI accepts the package.
- Allow an explicitly confirmed PEP 440 prerelease on a feature branch to be
  published through workflow dispatch to production PyPI. Stable versions
  remain blocked from manual publication, and normal installers ignore
  prereleases unless users opt in or request an exact version.
- CI now rejects a stale `uv.lock` before running either the library or
  occurrence verification suites.
- CI installs the occurrence environment from the lockfile and freezes every
  `uv run`, preventing jobs from silently repairing and then discarding lockfile
  drift.
- CI pins uv to Python 3.12.

## [0.4.1] - 2026-09-04

### Added

- Add `topographo.ssd.exact_machine`, an exact-rational sedenion implementation
  with ordered event traces, exact replay, and projective ray execution under
  the convention pinned by programming handoff 061.11.
- Add deterministic conformance tests, witness searches, machine-readable
  results, and observations for noncommutativity, nonassociativity,
  alternativity failure, zero divisors, norm failure, and ordered programs.

## [0.4.0] - 2026-07-11

### Added

- Paper II (`-ii`) landing from `born-channel`, aligned to the library/consumer
  layout (issue #13 §6). No change to the `topographo` package.
- `data/kraus84.npz` — the frozen Kraus-84 family as shared ground truth,
  regenerated by one blessed generator and loaded/diffed by everyone.
- `verify/occurrence_ii_audit.py` — the canonical, CI-gating Paper II audit.
  Its provenance step regenerates the family from `topographo.core` and
  certifies it matches `data/kraus84.npz`; Parts 1–8 then derive the physics
  from the `.npz` alone. Adapted from `born-channel`'s `santa physics.py` into
  the certificate-ledger / exit-code form used by `occurrence_i_audit.py`.
- `occurrence-theory-ii.md` — the top-level Paper II slot.
- `docs/Occurrence_Theory.pdf` — the Paper II explainer.

### Changed

- `occurrence.yml` now runs the Paper II audit as an exit-code gate and triggers
  on `data/**`.
- Move `docs/occurrence_theory_prompt.md` and `docs/Occurrence_Theory.pdf` into
  `docs/`; streamline the README's structure section.

## [0.3.0] - 2026-07-11

### Changed

- Carve the library/consumer seam (issue #13). `topographo` is now a standalone
  package with its own CI (`topographo.yml`), and occurrence (paper, audit,
  reviews) is a consumer with its own CI (`occurrence.yml`).
- Move `exceptional_algebras_lab.py` into the package as
  `topographo.exceptional`; it now imports `cayley_dickson_table` from
  `topographo.core` (single source of truth) instead of carrying its own copy.
- Move the audit to `verify/occurrence_i_audit.py` and `VERIFICATION.md` to
  `verify/occurrence_i_cabarius.md`; `verify/` is the single home for all Paper
  verification (see `verify/README.md`).
- Move library tests into `topographo/tests/` and theory tests into `verify/`.
- CI: gate the GitHub Pages docs deploy to `main` only (the `github-pages`
  environment permits only the default branch; tag pushes publish to PyPI and
  reuse the docs already deployed from `main`).
- CI: bump GitHub Actions onto the Node 24 runtime (`actions/checkout` v7,
  `astral-sh/setup-uv` v7, `actions/upload-artifact` and
  `actions/download-artifact` v7).

### Added

- Unit tests for the exceptional-algebra layer (`topographo/tests/test_exceptional.py`).
- Reviewer-independence guard in `occurrence.yml`.

### Removed

- The `occurrence-theory-audit` console script and the top-level `py-modules`
  entries; `pip install .` now ships only `topographo`.

## [0.2.1] - 2026-07-10

### Added

- Make the audit falsifiable and add supporting tests and diagnostics (see PR #11)

### Fixed

- Correct three misstatements in the paper and tighten audit certificate behavior (PR #11)

## [0.2.0] - 2026-07-08

### Added

- Deterministic unit tests to package, and CI

### Fixed

- occurrence_theory_audit.verify_gates() returns a plain `bool` instead of leaking `numpy.bool_`

## [0.1.1] - 2026-07-08

### Added

- `basis_zero_divisors()` for deterministic enumeration of the full 84-point
  basis crack design.

### Fixed

- The invariant-measure audit now gates the theorem certificate on the full
  84-point design and reports continuum estimates as Monte Carlo diagnostics.

## [0.1.0] - 2026-07-08

### Added

- Initial `topographo` Python package for reusable Cayley-Dickson algebra,
  validation gates, operators, and Sedenion Settlement Dynamics helpers.
- `occurrence-theory-audit` console command for running the numerical audit.
- Occurrence Theory paper draft, audit script, and supporting exceptional
  algebra reproduction module.
- GitHub Actions audit workflow and `pdoc` documentation setup.

[0.7.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.7.0
[0.6.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.6.0
[0.5.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.5.0
[0.4.2]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.4.2
[0.4.1]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.4.1
[0.4.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.4.0
[0.3.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.3.0
[0.2.1]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.2.1
[0.2.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.2.0
[0.1.1]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.1.1
[0.1.0]: https://github.com/TheSwanFactory/occurrence/releases/tag/v0.1.0

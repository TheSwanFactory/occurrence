<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Run Issue-009 corrected consequence-address ladder (`009.05` task, `009.06`
  result): is the `009.02` forcing negative a property of the SFP representation,
  or of prematurely expressing consequence as one of 84 opaque public Event
  identities? Six new modules in `experiments/sfp_representation/` —
  `ladder_task.py` (the Rung-1 local port and Rung-2 structured-address targets,
  `splits digest bb8ea403…`), `ladder_heads.py` (typed field heads that **import**
  `scorer.TokenEncoder` and `scorer.SymmetricPair` rather than reimplementing
  them, so the input pathway is provably the computation that produced the
  `009.02` numbers), `ladder_resolver.py`, `ladder_baselines.py`,
  `ladder_sweep.py` and `ladder_analysis.py` (which imports `009.02`'s seeded
  bootstrap so the two results are comparable as statistics, not just as numbers).
  Executes `009.05`; `009.04` was superseded before execution and both facts are
  machine-checkable fields in `ladder_task.json`.
  **Disposition C, partial.** Changing **only the output object** — 85 catalogue
  slots to 17 typed field slots, at capacity spread exactly 0 and with the frozen
  splits, arms, seeds and protocol unchanged — moved held-out
  `positive_forced_third_exact_accuracy` under the exact SFP code from **0.0335 to
  0.7016** against a `1/84` chance floor, with both structural signatures `009.05`
  section 5.6 requires: **B − C = +0.5082** (CI [0.4583, 0.5595], sign-consistent
  across all 14 LOHO folds and all 7 LOFPO folds) and **D − B = +0.0506** (CI
  [−0.0052, 0.1049], includes zero). Native coordinates are **not** sufficient:
  `A_native` reaches 0.4412, B − A = +0.2604 (CI [0.2039, 0.3199]).
  Four findings qualify it, and three are cautions against over-reading the
  headline. **The catalogue interface was a cause, not the only one**: learning all
  four address fields still fails at 0.0097, and it fails on the fields the
  certified relation *copies* rather than the ones it computes — `pp` reaches
  0.9948 while `FFF` collapses to **0.0320**, which is closed-set habitat
  re-identification under a structural holdout, not consequence forcing. The single
  repair the stop rule permits copies `S` and `FFF` (the construction `009.05`
  section 4.2 already licenses at Rung 1), is arm-neutral because the relation
  copies both on all 336 admitted pairs under B, C and D alike, and both blocks are
  reported in full. **The obstruction moved rather than vanished**: within the
  repair block `pp` is 0.9762 and `PP` 0.7158, and 0.9762 × 0.7158 = 0.6987 against
  an observed 0.7016, so locating the shared block is now the binding constraint.
  **Rung 1 cannot discriminate and is excluded from the disposition by
  construction**: its label is a function of the unordered input port pair, which
  takes exactly three values under the exact code, so a three-row lookup reaches
  1.0 on held-out habitats — B and D hit 1.0 at a zero generalization gap, and the
  observed B − C = +0.6682 is recorded and banked nowhere, because `GL(2,2) ≅ S3`
  makes every relabeling of that quotient a symmetry.
  **A nonlearned tabulation beats the learner, and that is the sharpest form
  `H_struct` has taken.** The code-space shortcut search is new to this issue —
  `009.02`'s nine atoms are all *native*, so no search over the features this
  learner actually sees had ever been run. Over 12 atoms and 298 subsets it finds
  that under the exact code the certified consequence is determined by a **two-atom**
  key with zero ambiguity over all 7056 ordered pairs, reaching 1.0 on held-out
  habitats, while under the non-automorphic scramble **no subset up to size three
  determines it at all**. The learner therefore sits between the best
  strictly-cheaper rule (0.500) and a ceiling the representation itself attains
  (1.000). The cost classification was restated from Outcome `021.05`'s definition
  of the Event-level law after the first run exposed that `endpoint_pair` — the
  complete K4 incidence data — reaches 0.994 and is not a shortcut at all.
  Rung 3 is exact by construction, not by luck: the SFP alphabet is **saturated**
  (2 × 7 × 6 = 84 = |Events|), so resolution is total and injective, resolved
  Event-identity accuracy is *identically* equal to exact structured-address
  accuracy in all 672 runs, and 0 of 4032 single-field corruptions are ever
  repaired. Swap invariance is 0.0 bitwise on every head across all 2688 runs, and
  the two Rung-1 run digests are identical across blocks by construction, since the
  repair touches only Rung 2. Opaque-token discovery is **now warranted** but
  should target `PP` and `pp` only and be judged against the 1.000 determining
  ceiling rather than chance.

- Add `experiments/sfp_representation/` for Issue-009 native-FIPS versus SFP
  consequence representation (`009.01` task, `009.02` result): does the compact
  `S | FFF | PP | pp` code carry the certified admit-and-force relation better
  than the native FIPS realization, and is any advantage relational rather than a
  matter of code capacity or coordinate choice? Eleven new modules — torch-free
  `conformance.py` (Gate 0: 103 Outcome-`021.05` pins re-derived, 0
  disagreements, 14 named relational laws, `TABLE_SHA256` `eb31fba3dbc3a4bb…`),
  `sfp.py` (the codec plus an exactness proof over the complete finite domain),
  `groups.py` (abstract `GL(3,2)` / `AGL(2,2)` / `GL(2,2)` / K4 machinery),
  `task.py` (the frozen 7056-ordered-pair pool, `dataset_sha256 7872755f…`),
  `folds.py` (22 manifests, `fold_manifest_sha256 27b01b5e…`, zero leakage on all
  21 structural folds recomputed from the recorded index lists rather than
  asserted), `arms.py` and `baselines.py`; torch-only `scorer.py`, `harness.py`,
  `run_sweep.py` and `analysis.py`, which per repo convention are a manual run
  and are not in CI.
  **Two finite pins are newly certified: 112 oriented cyclic blocks and 168
  Event-block incidences**, computed nowhere in the repo before this issue and
  derived *structurally* — oriented blocks enumerated as genuine cyclic-sense
  classes, incidences by walking actual membership — rather than as `2*56` and
  `84*2`, since the arithmetic shortcuts would assume exactly the regularity the
  pins are meant to witness.
  **Disposition E, explicitly narrowed.** Forcing is unsolved by every
  representation: the best forced-third accuracy over all 1400 runs is 0.0335
  (0.0900 after the single permitted repair) while both nonlearned oracles score
  1.0, so no arm demonstrates held-out generalization of the certified relation as
  a whole. The **admission half is cleanly resolved**: the SFP code beats its own
  marginal-matched, structure-destroying scramble by **+0.6369** on
  same-habitat-disjoint accuracy (CI [0.5655, 0.7054], sign-consistent across all
  14 primary folds, with bit widths, habitat sizes and field marginals held
  identical) and beats the native realization by **+0.3772** on admission balanced
  accuracy, where native realization sits at **chance** (0.5048). Relabeling under
  `GL(3,2)` + affine `AGL(2,2)` + a global `S` flip matches the code arm on every
  metric with every CI including zero, so the admission result is not a basis
  artifact. Capacity spread across the four representation arms is **exactly 0**
  (20323 parameters each), and swap symmetry is **0.0 bitwise** over primitives,
  pair and all 85 output slots as well as empirically on all 1400 runs.
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
  is only readable together with admission sensitivity. The one repair the stop
  rule permits was motivated by a diagnosed held-out target suppression and is
  reported as **tested and insufficient**. The opaque-token discovery stage is
  **not warranted yet**: the code arm is handed the exact certified code and still
  cannot name the held-out forced third, so hiding it could not discriminate.
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
  allowlist to cover `experiments/sfp_representation/`. Only the seven torch-free
  Issue-009 modules are named in the lint and import gate; `scorer.py`,
  `harness.py`, `run_sweep.py` and `analysis.py` require torch and stay a manual
  run, following the Outcome-017 and Issue-008 convention.

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

<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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

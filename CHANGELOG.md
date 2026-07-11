<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file.

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

[0.2.0]: https://github.com/TheSwanFactory/occurence/releases/tag/v0.2.0
[0.1.1]: https://github.com/TheSwanFactory/occurence/releases/tag/v0.1.1
[0.1.0]: https://github.com/TheSwanFactory/occurence/releases/tag/v0.1.0

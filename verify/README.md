# `verify/` — Paper verification

This directory is the single home for all verification of the Occurrence Theory
papers: the canonical first-party audits, the tests that guard them, independent
reviewer results, and supporting tools that help discharge the papers' standing
audit obligations.

## Naming convention

Files use the double-r `occurrence_` prefix, matching the paper and the repo
name. The token after `occurrence_<paper>_` carries the role.

- **`occurrence_<paper>_audit.py`** — the canonical, first-party, **CI-gating**
  audit. One per paper. `audit` is a *role*, not a person: these files are
  co-authored (dernie created the Paper I audit, cabarius hardened it), so they
  are not pinned to any single handle. Run as an exit-code script; the
  `occurrence.yml` workflow gates on it.

- **`occurrence_<paper>_<handle>.*`** — an independent **reviewer** result,
  named by the reviewer's handle. Any file extension is fine (`.py`, `.md`,
  `.ipynb`, …).

- **`occurrence_<paper>_<tool>.*`** — a supporting **tool**: a computation or
  recipe that helps a reviewer discharge a standing audit obligation, named for
  what it does. Neither the canonical audit nor a reviewer result.

A *handle* is a GitHub identity — a person, or an AI acting as a reviewer.
Claude-authored *first-party* work is not a handle: it lands as the canonical
`occurrence_<paper>_audit` with its provenance recorded in-file, not as a
reviewer cell.

## Independence

Everything except the canonical `*_audit` files — reviewer results and tools
alike — **must not import `topographo`**. That independence is the whole point,
and `occurrence.yml` enforces it with the reviewer-independence guard: a
reviewer reproduces the algebra itself (from the released data or its own
construction) rather than trusting the library under review.

## Adding a review

Drop a file named `occurrence_<paper>_<yourhandle>.<ext>` into this directory,
keeping it self-contained per the independence rule above.

Current reviewer cells include `codex_sol` reviews of Papers I and II in
`occurrence_i_codex_sol.{py,md}` and `occurrence_ii_codex_sol.{py,md}`.

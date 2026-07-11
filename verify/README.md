# `verify/` — Paper verification

This directory is the single home for all verification of the Occurrence Theory
papers: the canonical first-party audits, the tests that guard them, and
independent reviewer results. The filename token carries the tier.

## Naming convention

Files use the double-r `occurrence_` prefix, matching the paper and the repo
name.

- **`occurrence_<paper>_audit.py`** — the canonical, first-party,
  **CI-gating** audit. One per paper. `audit` is a *role*, not a person: these
  files are co-authored (dernie created the Paper I audit, cabarius hardened
  it), so they are not pinned to any single handle. Run as an exit-code script;
  the `occurrence.yml` workflow gates on it.

- **`occurrence_<paper>_<handle>.*`** — an independent **reviewer** result,
  named by the reviewer's GitHub handle. These **must not import `topographo`**
  — that independence is the whole point, and `occurrence.yml` enforces it with
  the reviewer-independence guard. Any file extension is fine (`.py`, `.md`,
  `.ipynb`, …).

`cabarius` and `solomonjoseph` are handles. `santa` is **not** a handle:
Claude-authored first-party work lands as `occurrence_ii_audit.py` with its
provenance recorded in-file, not as a reviewer cell.

## Current contents

| File | Paper | Tier |
| --- | --- | --- |
| `occurrence_i_audit.py` | I | canonical audit (CI-gating) |
| `occurrence_i_cabarius.md` | I | reviewer (cabarius) |
| `occurrence_ii_audit.py` | II | canonical audit (CI-gating); loads `data/kraus84.npz` |
| `occurrence_ii_grok.md` | II | reviewer (grok) |
| `occurrence_ii_claude_code.{py,md}` | II | reviewer (claude_code); from-scratch re-derivation |
| `test_*.py` | I | tests guarding the audit and theory claims |

## Adding a review

Drop a file named `occurrence_<paper>_<yourhandle>.<ext>` into this directory.
If it is code, keep it self-contained: reproduce the algebra yourself rather
than importing `topographo`, so your result is a genuinely independent check.

## Tools for reviewers

Non-review helpers that support the standing audit obligations live here under
plain descriptive names (no `occurrence_` prefix):

- `g2_su3_branching.sage` (with `.md` explainer) — a SageMath script that
  computes the G₂ ⊃ SU(3) branching, for the §10.1 representation-theory check
  and the Conjecture C3 `8 ⊕ 6` question.

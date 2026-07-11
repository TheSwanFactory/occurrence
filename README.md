# Occurrence Theory

This repository contains draft research papers and their verification for
Occurrence Theory (OT), defined as an oriented form of Sedenion Settlement
Dynamics (SSD).

The central object is the Aut-invariant settlement channel on the sedenion
zero-divisor crack. The paper separates theorem, exact computation,
measurement, interpretation, and conjecture using explicit ledger tags:

- `[T]` theorem
- `[C]` computation
- `[M]` measurement
- `[I]` interpretation
- `[X]` conjecture

## Layout

The repository is split along a **library / consumer** seam:

- **`topographo/`** — the reusable Python library: Cayley-Dickson algebra,
  validation gates, operators, SSD helpers, and the exceptional-algebra
  (Albert / F4 / G2) layer. Ships to PyPI with its own tests and CI
  (`topographo.yml`). See [`topographo/README.md`](topographo/README.md) for
  the library's own overview, install, and usage — that file is also the PyPI
  long description.
- **`verify/`** — the consumer side: all paper verification. Each paper has a
  canonical, CI-gating audit (`occurrence_<paper>_audit.py`), the tests that
  guard it, and independent reviewer cells; its CI (`occurrence.yml`) installs
  `topographo` and runs the audits as exit-code gates. See
  [`verify/README.md`](verify/README.md) for the naming convention.

The papers live at the top level — `occurrence-theory.md` (Paper I) and
`occurrence-theory-ii.md` (Paper II) — with supporting material in `docs/` and
shared ground-truth data in `data/`.

## Requirements

The audit script requires Python 3.11 or newer and NumPy, plus the `topographo`
package (which it imports for the verified algebra).

`uv` is the preferred runner for local audit work:

```bash
uv run python verify/occurrence_i_audit.py
```

For editable package installation:

```bash
uv pip install -e .
```

The reusable library — its own install, import examples, layout, and API
documentation — is documented in [`topographo/README.md`](topographo/README.md).
API docs are published to GitHub Pages:
<https://theswanfactory.github.io/occurrence/>.

## Run the Audit

From the repository root:

```bash
uv run python verify/occurrence_i_audit.py
```

To save the output:

```bash
uv run python verify/occurrence_i_audit.py > audit_results.txt
```

The audit exits `0` only if every certificate meets its threshold, and `1`
otherwise, so it is safe to gate CI on it. A passing run means the paper's
`[C]`-tagged claims reproduce on this implementation. It does not mean the
paper's `[I]` interpretations are correct; those are not tested.

CI runs two workflows on pull requests and pushes to `main`: `topographo.yml`
(library: tests, build, release) and `occurrence.yml` (consumer: installs the
package, runs this audit as an exit-code gate, and runs the `verify/` tests).

## Status

This is a research workspace, not a packaged library. The paper is the primary
artifact; the script is included to reproduce the computation-backed claims.

## License

MIT. See `LICENSE`.

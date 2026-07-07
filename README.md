# Occurrence Theory

This repository contains a draft research paper and verification script for
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

## Files

- `occurrence-theory.md` - main paper draft.
- `occurrence_theory_audit.py` - numerical audit and verification script for
  the algebraic claims.
- `occurrence_theory_prompt.md` - source prompt and writing constraints used to
  generate the paper.
- `LICENSE` - MIT license.

## Requirements

The audit script requires Python 3.11 or newer, NumPy, and the verified
`exceptional_algebras_lab` package.

`uv` is the preferred runner for local audit work:

```bash
uv run python occurrence_theory_audit.py
```

The verified `exceptional_algebras_lab` module is required at runtime, but its
package source is not currently recorded in this repository. Until that source
is added, `uv run` will prepare the Python environment and the audit will stop
early if the module is unavailable.

For editable console-script installation:

```bash
uv pip install -e .
```

## Run the Audit

From the repository root:

```bash
uv run python occurrence_theory_audit.py
```

To save the output:

```bash
uv run python occurrence_theory_audit.py > audit_results.txt
```

After installation, the same audit is also available as:

```bash
occurrence-theory-audit
```

The script exits early if `exceptional_algebras_lab` is unavailable.

## Status

This is a research workspace, not a packaged library. The paper is the primary
artifact; the script is included to reproduce the computation-backed claims.

## License

MIT. See `LICENSE`.

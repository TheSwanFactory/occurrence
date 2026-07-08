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

- `topographo/` - reusable Python package for Cayley-Dickson algebra,
  validation gates, operators, and SSD helpers.
- `occurrence-theory.md` - main paper draft.
- `occurrence_theory_audit.py` - numerical audit and verification script for
  the algebraic claims.
- `exceptional_algebras_lab.py` - supporting exceptional algebra reproduction
  module used by the audit.
- `occurrence_theory_prompt.md` - source prompt and writing constraints used to
  generate the paper.
- `.github/workflows/audit.yml` - CI workflow that compiles and runs the audit
  scripts.
- `CHANGELOG.md` - release history for the package and audit artifacts.
- `LICENSE` - MIT license.

## Requirements

The audit script requires Python 3.11 or newer and NumPy. The verified
`exceptional_algebras_lab` module is included in this repository and is used by
the audit at runtime.

`uv` is the preferred runner for local audit work:

```bash
uv run python occurrence_theory_audit.py
```

For editable package installation:

```bash
uv pip install -e .
```

After installation, the core math layer is importable without running the
Occurrence Theory audit narrative:

```python
from topographo.core import CayleyDicksonAlgebra, verify_gates
from topographo.ssd import SedenionAlgebra
```

For exact finite crack certificates, use `basis_zero_divisors()` to enumerate
the full 84-point design. `sample_crack(n)` samples from that design with
replacement and is intended for stochastic diagnostics, not machine-zero
theorem gates.

API documentation is generated with `pdoc` and published to GitHub Pages:
<https://theswanfactory.github.io/occurence/>

To build it locally:

```bash
uv run pdoc \
  topographo \
  topographo.core \
  topographo.core.algebra \
  topographo.core.cayley_dickson \
  topographo.core.gates \
  topographo.ssd \
  topographo.ssd.channel \
  topographo.ssd.sedenion \
  -o site
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

CI runs the audit workflow on pull requests and pushes to `main`.

## Status

This is a research workspace, not a packaged library. The paper is the primary
artifact; the script is included to reproduce the computation-backed claims.

## License

MIT. See `LICENSE`.

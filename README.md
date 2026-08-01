# Occurrence Theory

## What this is

**Sedenions** are a 16-dimensional number system built by doubling the
octonions (the same [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction)
that turns the reals into complex numbers, then quaternions, then octonions).
Each doubling costs a familiar property — complex numbers lose ordering,
quaternions lose commutativity, octonions lose associativity. Sedenions lose
one thing further: they have genuine [zero divisors](https://en.wikipedia.org/wiki/Zero_divisor),
nonzero elements `x`, `y` with `xy = 0`. That singular set is the object this
repo studies.

**Sedenion Settlement Dynamics (SSD)** averages left-multiplication over the
zero-divisor set, weighted by the unique measure invariant under the
algebra's automorphism group. The result is a single, exactly-solvable
linear channel. The paper proves — as theorems, not conjectures — that this
averaging always settles to the identity (equilibrium is forced, not
assumed), that the algebra cannot generate any internal dynamics beyond
rigid rotations (`No-Autonomy`), and computes the channel's full 256×256
eigenvalue spectrum exactly, in closed form (sevenths and `2√3/7`, with
multiplicities given by `G₂` representation dimensions).

**Occurrence Theory (OT)** is SSD plus exactly one added ingredient: a rule
for which side of each multiplication is *retained* (carried forward) and
which is *sampled* (drawn fresh from the zero-divisor set). That single bit
turns the static algebra into a genuine Markov chain — a sequence of
*occurrences*. The paper proves this bit cannot be derived from the algebra
itself, is unique up to a gauge symmetry, and is the minimal addition needed
to get any dynamics at all.

Every claim in the paper is tagged so a reader knows exactly what kind of
evidence backs it:

- `[T]` theorem (proved from stated identities)
- `[C]` computation (exact numerical certificate, threshold 10⁻¹²)
- `[M]` measurement (Monte Carlo, with error bars)
- `[I]` interpretation (not proved — a reading of the math, priced at zero)
- `[X]` conjecture (stated, not proved)

The `[T]`/`[C]` layer (SSD) stands on its own; the `[I]` layer (words like
"time," "generation," "occurrence" itself) is explicitly optional and
separable from it.

## What's in this repo

- `topographo/` - reusable Python **package** (the library) for Cayley-Dickson
  algebra, validation gates, operators, SSD helpers, and the exceptional-algebra
  (Albert / F4 / G2) layer. Ships to PyPI with its own tests under
  `topographo/tests/`.
- `verify/` - the **consumer** side: all Paper verification. Holds the canonical
  first-party audits, the tests that guard them, and independent reviewer
  results. See `verify/README.md` for the naming convention.
- `occurrence-theory.md` - main paper draft (Paper I): full statements and
  proofs of SSD and OT above.
- `verify/occurrence_i_audit.py` - numerical audit and verification script for
  the Paper I algebraic claims. Every printed `[C]`/`[G]` line is a computed
  number checked against a threshold; the script exits nonzero if any fails.
- `verify/occurrence_i_cabarius.md` - independent re-derivation of the paper's
  computational claims (reviewer: cabarius), and the corrections it produced.
- `occurrence_theory_prompt.md` - source prompt and writing constraints used to
  generate the paper.
- `.github/workflows/topographo.yml` - library CI: tests, builds, and releases
  the `topographo` package.
- `.github/workflows/occurrence.yml` - consumer CI: installs `topographo`, runs
  the audit exit-code gate and the `verify/` tests.
- `CHANGELOG.md` - release history for the package and audit artifacts. Paper
  versions and the `topographo` package version move independently — the
  package only bumps when a change actually touches the library.
- `LICENSE` - MIT license.

## Requirements

The audit script requires Python 3.11 or newer and NumPy, plus the `topographo`
package (which it imports for the verified algebra). The exceptional-algebra
(Albert / F4 / G2) reproduction now lives inside the package as
`topographo.exceptional`.

`uv` is the preferred runner for local audit work:

```bash
uv run python verify/occurrence_i_audit.py
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

sedenions = SedenionAlgebra()
zero_divisors = sedenions.basis_zero_divisors()  # exact 84-point crack design
```

For exact finite crack certificates, use `basis_zero_divisors()` to enumerate
the full 84-point design. `sample_crack(n)` samples from that design with
replacement and is intended for stochastic diagnostics, not machine-zero
theorem gates.

API documentation is generated with `pdoc` and published to GitHub Pages:
<https://theswanfactory.github.io/occurrence/>

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
  topographo.exceptional \
  topographo.exceptional.lab \
  -o site
```

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

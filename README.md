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

## Layout

The repository is split along a **library / consumer** seam:

- **`topographo/`** — the reusable Python library: Cayley-Dickson algebra,
  validation gates, operators, SSD helpers, the exceptional-algebra layer, and
  the exact single-step OT Born transport API. It ships as the `topographo`
  distribution with its own tests and CI. See
  [`topographo/README.md`](topographo/README.md).
- **`packages/decision-model/`** — the independently buildable, backend-neutral
  Decision Model contract. Install the `decision-model` distribution and import
  `decision_model` for typed State/Effect/Test resolution. Its transitional
  monorepo placement does not permit a reverse dependency from `topographo`.
- **`verify/`** — the consumer side: canonical paper audits and independent
  reviewer cells. Its CI installs `topographo` and treats audit exit codes as
  gates. See [`verify/README.md`](verify/README.md).
- **`issues/`** — numbered research evidence bundles, implementation records,
  attachments, and tasks; contents and evidentiary scope vary by issue.

The papers live at the top level — `occurrence-theory.md` (Paper I) and
`occurrence-theory-ii.md` (Paper II) — with supporting material in `docs/` and
shared ground-truth data in `data/`.

## Requirements

Python 3.11 or newer is required. `topographo` requires NumPy; the generic
`decision_model` contract uses only the standard library.

Install the two distributions independently:

```bash
pip install topographo
pip install decision-model
```

For repository development with `uv`:

```bash
uv run python verify/occurrence_i_audit.py
uv run pytest topographo/tests
uv run --project packages/decision-model pytest
```

The public package overviews live in [`topographo/README.md`](topographo/README.md)
and [`packages/decision-model/README.md`](packages/decision-model/README.md). API
docs are published at <https://theswanfactory.github.io/occurrence/>.

## Run the Audit

From the repository root:

```bash
uv run python verify/occurrence_i_audit.py
```

To save the output:

```bash
uv run python verify/occurrence_i_audit.py > audit_results.txt
```

The audit exits `0` only if every certificate meets its threshold. A passing run
means the paper's `[C]`-tagged claims reproduce on this implementation; it does
not validate the paper's `[I]` interpretations.

CI runs the topographo library/release workflow, the independent Decision Model
package/release workflow, and the Occurrence consumer/audit workflow on relevant
changes.

## Status

This repository contains two packaged public software surfaces alongside the
research papers and independent verification record. Their APIs and scientific
claims are intentionally narrower than the interpretive programme.

## License

MIT. See `LICENSE`.

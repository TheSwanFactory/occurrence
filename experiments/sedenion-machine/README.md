# Minimal exact-rational sedenion machine

This directory runs the 061.11 programming handoff against the reusable
`topographo.ssd.exact_machine` implementation. A value is an immutable tuple of
16 `fractions.Fraction` coefficients in basis order `e0, e1, ..., e15`. This is
an exact executable rational subalgebra of the real sedenions; it does not
represent every real coefficient.

The reference multiplication recursively splits a vector into `(a, b)` and
pins this Cayley–Dickson convention:

```text
conj(scalar) = scalar
conj((a,b)) = (conj(a), -b)
mul((a,b),(c,d)) = (mul(a,c) - mul(d,conj(b)),
                     mul(conj(a),d) + mul(c,b))
```

This is the handoff's convention, not the convention used by the existing
floating-point `topographo.core.cayley_dickson` implementation. Signs and basis
labels from one implementation must not be transferred to the other without an
explicit convention map.

No reassociation is performed. An ordered event program left-multiplies each
event, so events `[z1, z2]` produce `z2*(z1*initial)`, which need not equal
`(z2*z1)*initial`.

Run package conformance tests:

```bash
uv run pytest topographo/tests/test_exact_machine.py
```

Run the deterministic witness searches and experiments, regenerating
`results.json` and `observations.md`:

```bash
uv run python experiments/sedenion-machine/experiments.py
```

Rerun every experiment and verify the committed machine-readable results
without rewriting either generated artifact:

```bash
uv run python experiments/sedenion-machine/experiments.py --check
```

The tests cover all 256 ordered basis products and finite samples from a
documented PRNG seed. The generated report distinguishes conformance guarantees
from observations; sampled success is not claimed as a universal algebraic
proof.

Files:

- `../../topographo/ssd/exact_machine.py`: exact values, arithmetic, JSON
  expressions, ordered traces, exact replay, and ray mode.
- `../../topographo/tests/test_exact_machine.py`: conformance,
  embedded-subalgebra, expression, trace, replay, and ray tests.
- `experiments.py`: deterministic witness searches and six requested experiment
  families.
- `results.json`: generated machine-readable operands, traces, endpoints, norms,
  and coverage.
- `observations.md`: generated concise interpretation and domain boundaries.

# Minimal exact-rational sedenion machine

This directory runs the 061.11 programming handoff against the layered exact
implementation in `topographo.core.exact` and `topographo.ssd`. A value is an
immutable tuple of 16 `fractions.Fraction` coefficients in basis order `e0`,
`e1`, ..., `e15`. This is an exact executable rational subalgebra of the real
sedenions; it does not represent every real coefficient.

The public exact API pins the programming handoff's coordinate presentation:

```text
conj((a,b)) = (conj(a), -b)
mul((a,b),(c,d)) = (mul(a,c) - mul(d,conj(b)),
                     mul(conj(a),d) + mul(c,b))
```

This is a coordinate presentation of the same abstract algebra used by
`topographo.core`, not a second multiplication implementation. The exact and
NumPy backends derive products from one integer signed-basis table in core
coordinates. The explicit involution `Phi(a,b) = (conj(a), b)` maps the 061.11
presentation to core coordinates; the checked helpers are
`to_core_coordinates()` and `from_core_coordinates()`.

No reassociation is performed. An ordered event program left-multiplies each
event, so events `[z1, z2]` produce `z2*(z1*initial)`, which need not equal
`(z2*z1)*initial`. Raw vectors and rational projective points use the same
transition engine. Projective equivalence identifies every nonzero rational
scaling, including negative scaling; zero products terminate with an explicit
`Annihilated` outcome.

Run package conformance and layer tests:

```bash
uv run --frozen pytest topographo/tests/test_exact_machine.py \
  topographo/tests/test_machine_layers.py
```

Run the deterministic witness searches and experiments, regenerating
`results.json` and `observations.md`:

```bash
uv run --frozen python experiments/sedenion-machine/experiments.py
```

Rerun every experiment and verify both committed artifacts without rewriting
either one:

```bash
uv run --frozen python experiments/sedenion-machine/experiments.py --check
```

The tests cover all 256 ordered basis products and finite samples from a
documented PRNG seed. The generated report distinguishes conformance guarantees
from observations; sampled success is not claimed as a universal algebraic
proof.

Files:

- `../../topographo/core/exact.py`: exact arithmetic from the shared algebraic
  specification.
- `../../topographo/ssd/exact.py`: the 061.11 coordinate adapter.
- `../../topographo/ssd/program.py`: typed expressions and ordered programs.
- `../../topographo/ssd/machine.py`: shared transition engine and raw policy.
- `../../topographo/ssd/projective.py`: projective policy and annihilation.
- `../../topographo/ssd/observers.py`: derived trace measurements.
- `../../topographo/ssd/codec.py`: JSON expression/run encoding and replay.
- `../../topographo/ssd/exact_machine.py`: compatibility facade.
- `../../topographo/tests/test_exact_machine.py`: original conformance and
  compatibility behavior.
- `../../topographo/tests/test_machine_layers.py`: direct layer-boundary tests.
- `experiments.py`: deterministic witness searches and six requested experiment
  families.
- `results.json`: generated machine-readable operands, traces, endpoints, norms,
  and coverage.
- `observations.md`: generated concise interpretation and domain boundaries.

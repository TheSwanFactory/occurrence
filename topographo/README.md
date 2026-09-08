# topographo

Executable tools for Topographical Graph Theory (TGT): reusable Cayley-Dickson
and sedenion settlement math.

Topographo packages the reusable mathematical core behind the Occurrence Theory
research notes. The package is intentionally narrower than the paper: it exposes
the Cayley-Dickson and sedenion settlement computations needed for TGT without
importing the interpretive narrative layer or printing audit output at import
time.

In this package, TGT means the computational study of the sedenion
zero-divisor crack as a topological, graph-like, and operator-theoretic object.
The basic workflow is:

1. Build the 16-dimensional Cayley-Dickson algebra, the sedenions.
2. Enumerate or sample unit zero divisors, the distinguished singular locus
   called the crack.
3. Turn those events into multiplication operators and diagnostics.
4. Build finite graphs or channels whose edges/transitions are defined by
   multiplication, annihilation, metric transport, or settlement strain.
5. Check every result against validation gates before treating numerical
   output as evidence.

## Sedenion Settlement Dynamics (SSD)

The central reusable object is **Sedenion Settlement Dynamics**. SSD studies the
16-dimensional Cayley-Dickson algebra, its unit zero-divisor locus, and
operators built from left multiplication:

- `L_x`: left multiplication by an algebra element.
- `M_x = L_x.T @ L_x`: the metric operator measuring norm transport.
- `T_x = L_{x^2} - L_x^2`: the alternator used to express settlement strain.

These operators are enough to express the main graph-theoretic and channel
computations. For example, a zero-divisor graph can use crack samples as
vertices and connect two vertices when a multiplication-derived test such as
`rank(L_z @ L_w)` or `z * w == 0` detects annihilation. A settlement channel
averages the operator action `L_z.T @ X @ L_z` over crack events.

## Package layout

The reusable TGT/SSD layer is separated from the interpretive Occurrence Theory
layer, which lives outside this package.

- `topographo.core` — one exact signed-basis specification for Cayley-Dickson
  multiplication, the derived NumPy structure tensor, multiplication operators,
  and mandatory validation gates. It does not know about Occurrence Theory,
  event/state language, or report formatting.
- `topographo.ssd` — the sedenion-specific wrapper (`SedenionAlgebra`), small
  channel diagnostics, and layered exact-rational execution. Exact algebra,
  typed programs, raw execution, projective state, observers, and JSON codecs
  have separate modules; `exact_machine` remains as a compatibility facade.
  `frames` adds the Theory 068.02 mutual Operational-Frame machine (Native/Cyclic
  state, seed guard, and explicit round policies) without widening that facade.
- `topographo.exceptional` — the exceptional-algebra layer: the 27-dimensional
  Albert algebra `J3(O)` and its F4/G2 structure (Peirce/Hessian analysis,
  determinant invariants, anisotropy). Generic exceptional-algebra math, not
  Occurrence-Theory-specific.

## Installation

```bash
pip install topographo
```

Requires Python 3.11+ and NumPy.

## Minimal use

```python
import numpy as np

from topographo.core import verify_gates
from topographo.ssd import SedenionAlgebra, average_metric_operator

assert all(result.passed for result in verify_gates())

algebra = SedenionAlgebra()
events = algebra.basis_zero_divisors()
mean_metric = average_metric_operator(algebra, events)

equilibrium_error = np.linalg.norm(mean_metric - np.eye(algebra.dim))
```

## Sketch of a TGT zero-divisor graph

```python
import numpy as np

from topographo.ssd import SedenionAlgebra

algebra = SedenionAlgebra()
events = algebra.basis_zero_divisors()
operators = [algebra.left_operator(z) for z in events]

graph = {i: [] for i in range(len(events))}
for i, left_i in enumerate(operators):
    for j, left_j in enumerate(operators):
        if i == j:
            continue
        if np.linalg.svd(left_i @ left_j, compute_uv=False)[-1] < 1e-9:
            graph[i].append(j)
```

For exact finite crack certificates, use `SedenionAlgebra.basis_zero_divisors()`
to enumerate the full 84-point design. `sample_crack(n)` samples from that
design with replacement and is intended for stochastic diagnostics, not
machine-zero theorem gates. These dimension-16 helpers belong to
`SedenionAlgebra`; generic `CayleyDicksonAlgebra` supplies only algebra and
operator operations.

## Exact ordered-event machine

The exact implementation has explicit responsibility boundaries:

- `topographo.core.exact` derives exact rational arithmetic from the canonical
  signed-basis table.
- `topographo.ssd.exact` adapts that backend to the pinned 061.11 coordinates.
- `topographo.ssd.program` defines typed, fully parenthesized expressions and
  immutable ordered event programs.
- `topographo.ssd.machine` performs repeated left action through one
  policy-driven transition engine; raw execution continues through zero.
- `topographo.ssd.projective` identifies every nonzero rational scaling,
  including negative scaling, and returns `Annihilated` for a zero product.
- `topographo.ssd.observers` derives measurements such as squared norm, while
  `topographo.ssd.codec` owns JSON encoding and exact replay validation.

`topographo.ssd.exact_machine` remains as a compatibility facade for the 0.4
API. Both exact and NumPy backends use the same algebraic specification. The map
`Phi(a, b) = (conj(a), b)` converts 061.11 values to core coordinates;
`to_core_coordinates()` and `from_core_coordinates()` provide the checked
boundary.

```python
from topographo.ssd import exact, machine, program, projective

initial = exact.basis(4)
events = program.Program([exact.basis(2), exact.basis(1)])
result = machine.run(initial, events)

assert result.state == exact.mul(
    exact.basis(1), exact.mul(exact.basis(2), initial)
)
assert projective.run_projective(initial, events).state == result.state
```


## Mutual Operational-Frame machine

`topographo.ssd.frames` exposes the certified Theory 068.02 local machine:
`OperationalFrame` with Native or Cyclic retained state, Event/edge predicates,
seed guard, `step` / `advance`, and explicit serial or snapshot `round_step`
policies. Presentation wiring is an experimental input; no global selector is
inferred. See `experiments/mutual-frames/` for the finite census audit.

```python
from topographo.ssd import exact
from topographo.ssd.frames import Native, OperationalFrame, round_step

a = exact.add(exact.basis(1), exact.basis(10))
b = exact.add(exact.basis(4), exact.basis(15))
pair = (OperationalFrame(Native(a)), OperationalFrame(Native(b)))
result = round_step(pair, "AB")
assert result.complete
```


## Futurator gate (Outcome / FIPS / seal)

Issue-005 Milestone F adds Outcome-aware `ask` without identifying Futurator with
soft μ-or-bottom:

- `topographo.ssd.outcome_runtime`: `ask(presentation, admissibility, enacted_law)`
  returns `ConstitutedOutcome` or typed `NonAdmission`. Laws include
  `NativeLeftAction`, `ProjectiveNativeLeftAction`, and `FipsClosure`.
  Projective zero products constitute `AnnihilationBoundary`.
- `topographo.ssd.fips_basic`: certified finite table (84/56/336) and forced
  `third` — never ranked candidates.
- `topographo.ssd.seal`: `classify` / `seal` / `resolve` with a Theory-39 witness
  that public incidence ≠ sealed evaluator denotation.

Sealed Event frames use `EventDenotation`. Do not confuse them with
`OperationalFrame` from Theory 068.02 (`ssd.frames`), which owns retained Native
or Cyclic state and local round policies. Non-goals for this gate: Theory-41
effect runtime, physical SCE family guards, and optimizer/loss imports in the
Outcome path (`Fut≠μ|⊥`).

```python
from topographo.ssd.outcome_runtime import (
    Admissibility,
    Presentation,
    ProjectiveNativeLeftAction,
    ask,
)
from topographo.ssd.seal import seal
from topographo.ssd import exact

z = seal(exact.add(exact.basis(1), exact.basis(10)))
outcome = ask(Presentation((z, exact.one())), Admissibility(True), ProjectiveNativeLeftAction())
```

## Validation gates

The validation gates are deliberately conservative. They catch sign-convention
or tensor-indexing errors early: they certify that the implementation uses the
intended Cayley-Dickson convention rather than acting as broad theorem tests.
Any independent implementation should pass composition, antisymmetry, quadratic,
and Moufang checks before its numerical certificates are trusted.

## Links

- Source: <https://github.com/TheSwanFactory/occurrence>
- API documentation: <https://theswanfactory.github.io/occurrence/>

## License

MIT.

## Issue 005 Milestone T adapters (0.8.1)

- `ssd.fips_adapter` — hard FIPS index/tensor path and declared STE learning adapter
  (exact Fraction conformance stays in `fips_basic` / frames; no optimizer imports).
- `ssd.structural_control` — matched non-isomorphic degree-matched rewiring with C1–C5 checks.
- Training smoke lives in `experiments/tlm_modular/` (torch optional; not a CI GPU job).


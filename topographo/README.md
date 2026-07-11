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

- `topographo.core` — Cayley-Dickson construction, multiplication operators,
  and mandatory validation gates. It does not know about Occurrence Theory,
  event/state language, or report formatting.
- `topographo.ssd` — the sedenion-specific wrapper (`SedenionAlgebra`) and small
  channel diagnostics used by the settlement audit.
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

For exact finite crack certificates, use `basis_zero_divisors()` to enumerate
the full 84-point design. `sample_crack(n)` samples from that design with
replacement and is intended for stochastic diagnostics, not machine-zero
theorem gates.

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

"""Core Cayley-Dickson primitives and validation gates.

This subpackage is the lowest reusable layer. It does not know about
Occurrence Theory, event/state language, or report formatting. It provides the
algebraic substrate needed to reproduce the computational claims:

- `cayley_dickson_table(dim)` builds a structure tensor for real
  Cayley-Dickson algebras.
- `CayleyDicksonAlgebra` wraps multiplication, conjugation, left/right
  multiplication operators, `M_x`, and `T_x`.
- `verify_gates()` runs the four mandatory validation checks used by the
  paper before accepting numerical certificates.

The gates are meant to catch sign-convention or tensor-indexing errors early.
They are not broad theorem tests; they certify that the implementation is
using the intended Cayley-Dickson convention.
"""

from topographo.core.algebra import CayleyDicksonAlgebra
from topographo.core.cayley_dickson import cayley_dickson_table
from topographo.core.gates import GateResult, verify_gates

__all__ = [
    "CayleyDicksonAlgebra",
    "GateResult",
    "cayley_dickson_table",
    "verify_gates",
]

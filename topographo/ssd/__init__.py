"""Sedenion Settlement Dynamics helpers.

SSD is the paper's algebraic layer before any Occurrence Theory orientation is
added. It works with the 16-dimensional sedenion algebra, samples basis-form
zero divisors, and computes finite-sample diagnostics for operators such as
`M_z = L_z.T @ L_z`.

The public helpers here are intentionally small while the refactor is still
structure-preserving:

- `SedenionAlgebra` fixes `CayleyDicksonAlgebra` at dimension 16.
- `average_metric_operator()` averages `M_z` over a supplied event sample.

Higher-level claims about invariant measures, exact channel spectra, and
oriented Markov dynamics remain in the audit/paper layer until their API shape
is stable.
"""

from topographo.ssd.channel import average_metric_operator
from topographo.ssd.sedenion import SedenionAlgebra

__all__ = [
    "SedenionAlgebra",
    "average_metric_operator",
]

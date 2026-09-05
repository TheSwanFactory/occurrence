"""Sedenion Settlement Dynamics helpers.

SSD is the paper's algebraic layer before any Occurrence Theory orientation is
added. It works with the 16-dimensional sedenion algebra, samples basis-form
zero divisors, and computes finite-sample diagnostics for operators such as
`M_z = L_z.T @ L_z`.

The exact ordered-event implementation is split into explicit public layers:

- `exact` adapts the shared exact core algebra to the pinned 061.11 coordinates.
- `program` defines typed expressions and ordered event programs.
- `machine` supplies the shared left-action transition engine and raw policy.
- `projective` supplies rational projective state and annihilation semantics.
- `observers` derives measurements from policy-independent transitions.
- `codec` owns the optional JSON wire format and replay validation.
- `exact_machine` preserves the original combined API as a compatibility facade.

Higher-level claims about invariant measures, exact channel spectra, and
oriented Markov dynamics remain in the audit/paper layer until their API shape
is stable.
"""

from topographo.ssd import (
    codec,
    exact,
    exact_machine,
    machine,
    observers,
    program,
    projective,
)
from topographo.ssd.channel import average_metric_operator
from topographo.ssd.sedenion import SedenionAlgebra

__all__ = [
    "SedenionAlgebra",
    "average_metric_operator",
    "codec",
    "exact",
    "exact_machine",
    "machine",
    "observers",
    "program",
    "projective",
]

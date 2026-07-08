"""Settlement-channel diagnostics for sedenion crack events.

The channel viewpoint complements the graph viewpoint. A graph records which
crack events relate by a chosen predicate; a channel averages what those events
do as multiplication operators. In SSD/TGT the basic metric diagnostic is the
mean of `M_z = L_z.T @ L_z` over a crack sample.
"""

from __future__ import annotations

import numpy as np

from topographo.core.algebra import CayleyDicksonAlgebra


def average_metric_operator(algebra: CayleyDicksonAlgebra, events: np.ndarray) -> np.ndarray:
    """Return the finite-sample average of `M_z = L_z.T @ L_z`.

    Compare this average with the identity matrix as a sanity diagnostic before
    building higher-level graph or settlement-channel computations. The exact
    equilibrium statement belongs to the intended invariant measure or design;
    a finite random sample can have visible sampling error.
    """
    if len(events) == 0:
        raise ValueError("events must be non-empty")
    return sum(algebra.metric_operator(event) for event in events) / len(events)

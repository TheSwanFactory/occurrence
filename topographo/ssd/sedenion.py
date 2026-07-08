"""Sedenion-specific algebra wrapper."""

from __future__ import annotations

from topographo.core.algebra import CayleyDicksonAlgebra


class SedenionAlgebra(CayleyDicksonAlgebra):
    """The 16-dimensional Cayley-Dickson algebra used by SSD/TGT."""

    def __init__(self, *, seed: int | None = 42):
        super().__init__(16, seed=seed)

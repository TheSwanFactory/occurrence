"""Sedenion algebra and zero-divisor crack construction."""

from __future__ import annotations

import numpy as np

from topographo.core.algebra import CayleyDicksonAlgebra


class SedenionAlgebra(CayleyDicksonAlgebra):
    """The 16-dimensional Cayley-Dickson algebra used by SSD/TGT."""

    def __init__(self, dim: int = 16, *, seed: int | None = 42):
        if dim != 16:
            raise ValueError("SedenionAlgebra has fixed dimension 16")
        super().__init__(16, seed=seed)

    def basis_zero_divisors(self) -> np.ndarray:
        """Return the full 84-element basis-form unit crack design.

        The fixed order matches the blessed Kraus-family generator: lower
        imaginary index, upper imaginary index, then positive/negative sign.
        """

        scale = np.sqrt(2)
        return np.array(
            [
                (self.e[lower] + sign * self.e[8 + upper]) / scale
                for lower in range(1, 8)
                for upper in range(1, 8)
                if lower != upper
                for sign in (1, -1)
            ]
        )

    def sample_basis_zero_divisors(self, n: int) -> np.ndarray:
        """Sample with replacement from basis-form unit zero divisors."""

        zero_divisors = self.basis_zero_divisors()
        return zero_divisors[self.rng.integers(0, len(zero_divisors), n)]

    def sample_pure_pair(self, n: int) -> np.ndarray:
        """Sample random unit pure-pair events for the sedenion crack model."""

        first = self.rng.standard_normal((n, 8))
        first[:, 0] = 0
        first /= np.linalg.norm(first, axis=1, keepdims=True)
        second = self.rng.standard_normal((n, 8))
        second[:, 0] = 0
        second -= np.sum(second * first, axis=1, keepdims=True) * first
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        return np.concatenate([first, second], axis=1) / np.sqrt(2)

    sample_crack = sample_basis_zero_divisors

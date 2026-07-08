"""Cayley-Dickson structure tensor construction."""

from __future__ import annotations

import numpy as np


def cayley_dickson_table(dim: int) -> np.ndarray:
    """Return C[i, j, k] with e_i * e_j = sum_k C[i, j, k] e_k.

    The dimension must be a positive power of two. The construction uses the
    standard real Cayley-Dickson doubling convention used by the audit scripts.
    """
    if dim < 1 or dim & (dim - 1):
        raise ValueError("dim must be a positive power of two")

    def conjugate(x: np.ndarray) -> np.ndarray:
        return np.concatenate([[x[0]], -x[1:]])

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        if n == 1:
            return np.array([a[0] * b[0]])
        half = n // 2
        a1, a2, b1, b2 = a[:half], a[half:], b[:half], b[half:]
        z1 = mul(a1, b1) - mul(conjugate(b2), a2)
        z2 = mul(b2, a1) + mul(a2, conjugate(b1))
        return np.concatenate([z1, z2])

    table = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            ei = np.zeros(dim)
            ej = np.zeros(dim)
            ei[i] = 1.0
            ej[j] = 1.0
            table[i, j] = mul(ei, ej)
    return table

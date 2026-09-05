"""Exact Cayley-Dickson signed-basis and structure-table construction."""

from __future__ import annotations

from functools import cache
from typing import TypeAlias

import numpy as np

SignedBasisProduct: TypeAlias = tuple[int, int]
SignedBasisTable: TypeAlias = tuple[tuple[SignedBasisProduct, ...], ...]


def _validate_dimension(dim: int) -> None:
    if dim < 1 or dim & (dim - 1):
        raise ValueError("dim must be a positive power of two")


@cache
def signed_basis_table(dim: int) -> SignedBasisTable:
    """Return the exact product index and sign for every ordered basis pair.

    Each cell is ``(result_index, sign)`` and represents
    ``e_i * e_j = sign * e_result_index``. The immutable integer table is the
    canonical multiplication specification used by all package backends.
    """

    _validate_dimension(dim)
    if dim == 1:
        return (((0, 1),),)

    half = dim // 2
    lower = signed_basis_table(half)
    rows: list[tuple[SignedBasisProduct, ...]] = []
    for left in range(dim):
        row: list[SignedBasisProduct] = []
        for right in range(dim):
            if left < half and right < half:
                result, sign = lower[left][right]
            elif left < half:
                result, sign = lower[right - half][left]
                result += half
            elif right < half:
                result, sign = lower[left - half][right]
                if right:
                    sign = -sign
                result += half
            else:
                result, sign = lower[right - half][left - half]
                if right == half:
                    sign = -sign
            row.append((result, sign))
        rows.append(tuple(row))
    return tuple(rows)


def cayley_dickson_table(dim: int) -> np.ndarray:
    """Return ``C[i, j, k]`` with ``e_i * e_j = sum_k C[i, j, k] e_k``.

    The public NumPy tensor preserves the historical ``float64`` API and core
    coordinates. It is derived directly from :func:`signed_basis_table`, whose
    integer entries are the package's single algebraic specification.
    """

    table = signed_basis_table(dim)
    tensor = np.zeros((dim, dim, dim))
    for left, row in enumerate(table):
        for right, (result, sign) in enumerate(row):
            tensor[left, right, result] = sign
    return tensor

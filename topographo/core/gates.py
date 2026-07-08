"""Mandatory algebra validation gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topographo.core.cayley_dickson import cayley_dickson_table


@dataclass(frozen=True)
class GateResult:
    """Numerical result for one validation gate."""

    name: str
    error: float
    tolerance: float = 1e-12

    @property
    def passed(self) -> bool:
        return self.error < self.tolerance


def verify_gates(*, tolerance: float = 1e-12) -> list[GateResult]:
    """Run the four mandatory Cayley-Dickson validation gates on octonions."""
    c8 = cayley_dickson_table(8)
    e8 = np.eye(8)

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("i,j,ijk->k", a, b, c8)

    def left(a: np.ndarray) -> np.ndarray:
        return np.einsum("i,ijk->kj", a, c8)

    x, y = e8[1], e8[2]
    xy = mul(x, y)
    composition = abs(np.linalg.norm(xy) - np.linalg.norm(x) * np.linalg.norm(y))

    left_x = left(x)
    antisymmetry = np.linalg.norm(left_x + left_x.T)

    quadratic = np.linalg.norm(mul(x, x) + e8[0])

    a, b, c = e8[1], e8[2], e8[3]
    lhs = mul(mul(a, b), mul(c, a))
    rhs = mul(mul(a, mul(b, c)), a)
    moufang = np.linalg.norm(lhs - rhs)

    return [
        GateResult("composition", float(composition), tolerance),
        GateResult("antisymmetry", float(antisymmetry), tolerance),
        GateResult("quadratic", float(quadratic), tolerance),
        GateResult("moufang", float(moufang), tolerance),
    ]

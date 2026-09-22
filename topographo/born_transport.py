"""Single-step OT Born transport identity on the Kraus-84 representation.

This module evaluates one retained-state transition. It does not define an
outcome family, measurement normalization, conditioning, or state collapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DIMENSION = 16
_DEFAULT_ANNIHILATION_TOLERANCE = 1e-12
_UNIT_NORM_TOLERANCE = 1e-12

AnnihilationKind = Literal["exact", "near"]


@dataclass(frozen=True, slots=True)
class BornTransportResult:
    """Immutable values from one non-annihilating OT Born transport."""

    transported_state: NDArray[np.float64]
    normalization_cost: float
    event_strain: float
    post_transition_spine_share: float
    hermitian_numerator: float
    identity_residual: float

    def __post_init__(self) -> None:
        owned_state = np.ascontiguousarray(
            self.transported_state,
            dtype=np.float64,
        )
        transported_state = np.frombuffer(
            owned_state.tobytes(),
            dtype=np.float64,
        ).reshape(owned_state.shape)
        object.__setattr__(self, "transported_state", transported_state)


@dataclass(frozen=True, slots=True)
class BornTransportAnnihilation(ArithmeticError):
    """A typed refusal to normalize an exact or numerical annihilation.

    ``cost`` and ``threshold`` are squared-norm quantities. ``kind`` is
    ``"exact"`` when ``K_a x`` is the exact zero vector and ``"near"`` when
    it is nonzero but its squared norm is at most the positive threshold.
    """

    kind: AnnihilationKind
    cost: float
    threshold: float

    def __post_init__(self) -> None:
        if self.kind not in ("exact", "near"):
            raise ValueError("annihilation kind must be 'exact' or 'near'")
        ArithmeticError.__init__(
            self,
            f"{self.kind} OT Born transport annihilation: "
            f"squared cost {self.cost!r}, threshold {self.threshold!r}",
        )


def _real_array(value: ArrayLike, *, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return a finite float64 view of a real numeric input with exact shape."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        real_array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if real_array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {real_array.shape}")
    if not np.all(np.isfinite(real_array)):
        raise ValueError(f"{name} must contain only finite values")
    return real_array


def _annihilation_tolerance(value: float) -> float:
    """Validate and return a nonnegative finite squared-cost threshold."""

    message = "annihilation_tolerance must be a real nonnegative finite number"
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(message)
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(message)
    return tolerance


def _finite_scalar(value: float | np.floating, *, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"OT Born transport produced non-finite {name}")
    return scalar


def _validate_ot_structure(
    operator: NDArray[np.float64],
    clock: NDArray[np.float64],
) -> None:
    """Require the algebraic relations that force the transport identity."""

    tolerance = 1e-10
    identity = np.eye(_DIMENSION)
    if not np.allclose(operator.T, -operator, rtol=tolerance, atol=tolerance):
        raise ValueError("settlement_operator must be antisymmetric")
    if not np.allclose(clock.T, -clock, rtol=tolerance, atol=tolerance):
        raise ValueError("complex_structure must be antisymmetric")
    if not np.allclose(
        clock @ clock,
        -identity,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError("complex_structure must square to -I")

    e0 = identity[0]
    event = operator @ e0
    event_norm = float(np.hypot.reduce(np.abs(event)))
    if not math.isclose(
        event_norm,
        1.0,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError("settlement_operator must recover a unit event K_a e0")

    clock_axis = clock @ e0
    clock_event = clock @ event
    transported_axes = (
        np.outer(operator.T @ e0, operator.T @ e0)
        + np.outer(operator.T @ clock_axis, operator.T @ clock_axis)
    )
    hermitian_axes = np.outer(event, event) + np.outer(clock_event, clock_event)
    if not np.allclose(
        transported_axes,
        hermitian_axes,
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(
            "settlement_operator and complex_structure are not a compatible "
            "OT Born transport presentation"
        )


def ot_born_transport(
    state: ArrayLike,
    settlement_operator: ArrayLike,
    complex_structure: ArrayLike,
    *,
    annihilation_tolerance: float = _DEFAULT_ANNIHILATION_TOLERANCE,
) -> BornTransportResult:
    """Evaluate the exact single-step OT Born transport identity.

    ``state`` must be a real finite unit vector of shape ``(16,)``.
    ``settlement_operator`` and ``complex_structure`` must be real finite
    matrices of shape ``(16, 16)`` in the same OT presentation. The function
    verifies operator antisymmetry, the orthogonal complex-structure relation,
    unit event recovery, and the axis relation that forces the identity. The
    fixed structural tolerance is ``1e-10``.

    ``annihilation_tolerance`` is a squared-cost threshold; the default is
    ``1e-12`` and zero requests exact-annihilation handling only.

    Raises:
        BornTransportAnnihilation: If the transition is exact or numerically
            near annihilation. No normalization is attempted in either case.
        ValueError: If an input violates the shape, reality, finiteness,
            OT-structure, unit-state, or tolerance contract, or arithmetic
            under/overflows.
    """

    tolerance = _annihilation_tolerance(annihilation_tolerance)
    x = _real_array(state, name="state", shape=(_DIMENSION,))
    operator = _real_array(
        settlement_operator,
        name="settlement_operator",
        shape=(_DIMENSION, _DIMENSION),
    )
    clock = _real_array(
        complex_structure,
        name="complex_structure",
        shape=(_DIMENSION, _DIMENSION),
    )
    _validate_ot_structure(operator, clock)

    state_norm = float(np.hypot.reduce(np.abs(x)))
    if not math.isfinite(state_norm) or not math.isclose(
        state_norm,
        1.0,
        rel_tol=_UNIT_NORM_TOLERANCE,
        abs_tol=_UNIT_NORM_TOLERANCE,
    ):
        raise ValueError("state must be a unit vector within tolerance 1e-12")

    with np.errstate(over="ignore", invalid="ignore"):
        transported = operator @ x
    if not np.all(np.isfinite(transported)):
        raise ValueError("OT Born transport produced a non-finite K_a x")

    # Algebraic zero is classified before any norm-based arithmetic or division.
    if not np.any(transported != 0.0):
        raise BornTransportAnnihilation("exact", 0.0, tolerance)

    transported_norm = float(np.hypot.reduce(np.abs(transported)))
    if not math.isfinite(transported_norm) or transported_norm <= 0.0:
        raise ValueError("OT Born transport produced an invalid nonzero norm")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cost = float(transported_norm * transported_norm)
    if not math.isfinite(cost):
        raise ValueError("OT Born transport produced non-finite normalization cost")
    if tolerance > 0.0 and cost <= tolerance:
        raise BornTransportAnnihilation("near", cost, tolerance)
    if cost == 0.0:
        raise ValueError("OT Born transport normalization cost underflowed to zero")

    # Exact and near annihilation have both been handled before normalization.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transported_state = transported / transported_norm
        e0 = np.zeros(_DIMENSION, dtype=np.float64)
        e0[0] = 1.0
        clock_axis = clock @ e0
        event = operator @ e0
        clock_event = clock @ event
        spine_share = (transported_state @ e0) ** 2 + (
            transported_state @ clock_axis
        ) ** 2
        hermitian_numerator = (event @ x) ** 2 + (clock_event @ x) ** 2
        identity_residual = spine_share * cost - hermitian_numerator

    if not np.all(np.isfinite(transported_state)):
        raise ValueError("OT Born transport produced a non-finite transported state")
    return BornTransportResult(
        transported_state=transported_state,
        normalization_cost=cost,
        event_strain=cost - 1.0,
        post_transition_spine_share=_finite_scalar(
            spine_share,
            name="post-transition spine share",
        ),
        hermitian_numerator=_finite_scalar(
            hermitian_numerator,
            name="Hermitian numerator",
        ),
        identity_residual=_finite_scalar(
            identity_residual,
            name="identity residual",
        ),
    )


__all__ = [
    "BornTransportAnnihilation",
    "BornTransportResult",
    "ot_born_transport",
]

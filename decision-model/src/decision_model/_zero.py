"""Exact Decision Model Zero command and presentation schema."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from ._core import Distribution, Test, resolve_test

ZERO_SCHEMA: Final = "decision-model-zero/v1"


@dataclass(frozen=True, slots=True)
class ZeroCase:
    """One exact source fixture for the Decision Model Zero demonstration."""

    identifier: str
    state: Fraction


@dataclass(frozen=True, slots=True)
class ResolvedZeroCase:
    """One Zero source fixture resolved through the public generic contract."""

    identifier: str
    state: Fraction
    distribution: Distribution[str]


def _admitted_state(state: Fraction, /) -> Fraction:
    if not isinstance(state, Fraction):
        raise TypeError("Decision Model Zero states must be fractions.Fraction values")
    if state < 0 or state > 1:
        raise ValueError("Decision Model Zero states must satisfy 0 <= p <= 1")
    return state


def _identity(state: Fraction, /) -> Fraction:
    return _admitted_state(state)


def _complement(state: Fraction, /) -> Fraction:
    return Fraction(1) - _admitted_state(state)


ZERO_TEST: Final[Test[Fraction, str]] = Test(
    (("yes", _identity), ("no", _complement))
)
ZERO_CASES: Final[tuple[ZeroCase, ...]] = (
    ZeroCase("sharp", Fraction(1)),
    ZeroCase("non-sharp", Fraction(1, 3)),
)


def resolve_zero_case(case: ZeroCase, /) -> ResolvedZeroCase:
    """Resolve one exact fixture through the shared public ``Test`` path."""
    state = _admitted_state(case.state)
    return ResolvedZeroCase(case.identifier, state, resolve_test(state, ZERO_TEST))


def resolve_zero_cases() -> tuple[ResolvedZeroCase, ...]:
    """Resolve the two required fixtures in their normative order."""
    return tuple(resolve_zero_case(case) for case in ZERO_CASES)


def _require_fraction(value: object, *, field: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise TypeError(f"{field} must remain an exact Fraction through resolution")
    return value


def _rational_object(value: Fraction, /) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def build_zero_payload() -> dict[str, object]:
    """Build the complete versioned, exact, deterministic Zero document."""
    cases: list[dict[str, object]] = []
    for resolved in resolve_zero_cases():
        distribution = [
            {
                "label": label,
                "probability": _rational_object(
                    _require_fraction(
                        probability.value,
                        field=f"{resolved.identifier}.{label}",
                    )
                ),
            }
            for label, probability in resolved.distribution
        ]
        total = _require_fraction(
            resolved.distribution.total,
            field=f"{resolved.identifier}.total",
        )
        cases.append(
            {
                "id": resolved.identifier,
                "state": _rational_object(resolved.state),
                "distribution": distribution,
                "total": _rational_object(total),
            }
        )

    return {
        "schema": ZERO_SCHEMA,
        "backend": "generic-exact",
        "test": {
            "state_domain": "p in Q, 0 <= p <= 1",
            "effects": [
                {"label": "yes", "id": "identity", "expression": "p"},
                {"label": "no", "id": "complement", "expression": "1 - p"},
            ],
        },
        "cases": cases,
    }


def canonical_zero_bytes() -> bytes:
    """Serialize the normative document with one stable byte policy."""
    document = json.dumps(
        build_zero_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return document.encode("ascii") + b"\n"


def main() -> int:
    """Write exactly one Decision Model Zero JSON document to stdout."""
    sys.stdout.buffer.write(canonical_zero_bytes())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the installed command
    raise SystemExit(main())

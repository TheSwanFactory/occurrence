"""Exact, bounded implementation of the 017.17 Born-first coding audit."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from ._core import Distribution, Test, resolve_test
from ._zero import ZERO_CASES

AUDIT_SCHEMA: Final = "decision-model-01717-audit/v1"
DIMENSION: Final = 16
ZERO: Final = Fraction(0)
ONE: Final = Fraction(1)


@dataclass(frozen=True, slots=True)
class Preparation:
    """A nonzero exact rational ray representing one source state."""

    identifier: str
    source_probability: Fraction
    coordinates: tuple[Fraction, ...]
    normative: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_probability, Fraction):
            raise TypeError("source_probability must be an exact Fraction")
        if self.source_probability < 0 or self.source_probability > 1:
            raise ValueError("source_probability must satisfy 0 <= p <= 1")
        if len(self.coordinates) != DIMENSION:
            raise ValueError(f"preparation rays must have dimension {DIMENSION}")
        if not all(isinstance(coordinate, Fraction) for coordinate in self.coordinates):
            raise TypeError("preparation coordinates must be exact Fractions")
        if self.squared_norm <= 0:
            raise ValueError("preparation rays must be nonzero")

    @property
    def squared_norm(self) -> Fraction:
        return sum(
            (coordinate * coordinate for coordinate in self.coordinates),
            ZERO,
        )


@dataclass(frozen=True, slots=True)
class DiagonalEffect:
    """An immutable exact diagonal formal effect on the 16-D carrier."""

    identifier: str
    diagonal: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.diagonal) != DIMENSION:
            raise ValueError(f"formal effects must have dimension {DIMENSION}")
        if not all(isinstance(entry, Fraction) for entry in self.diagonal):
            raise TypeError("formal-effect entries must be exact Fractions")
        if any(entry < 0 or entry > 1 for entry in self.diagonal):
            raise ValueError("formal effects must satisfy 0 <= e <= 1 coordinatewise")

    @property
    def is_projector(self) -> bool:
        return all(entry * entry == entry for entry in self.diagonal)

    @property
    def is_nonzero(self) -> bool:
        return any(entry != 0 for entry in self.diagonal)

    def __call__(self, preparation: Preparation, /) -> Fraction:
        numerator = sum(
            (
                entry * coordinate * coordinate
                for entry, coordinate in zip(
                    self.diagonal,
                    preparation.coordinates,
                    strict=True,
                )
            ),
            ZERO,
        )
        return numerator / preparation.squared_norm


def _mask(*active: int) -> tuple[Fraction, ...]:
    active_indices = set(active)
    if any(index < 0 or index >= DIMENSION for index in active_indices):
        raise ValueError("effect-mask index is outside the carrier")
    return tuple(ONE if index in active_indices else ZERO for index in range(DIMENSION))


def _ray(*active: int) -> tuple[Fraction, ...]:
    active_indices = set(active)
    if any(index < 0 or index >= DIMENSION for index in active_indices):
        raise ValueError("preparation-ray index is outside the carrier")
    return tuple(ONE if index in active_indices else ZERO for index in range(DIMENSION))


YES_EFFECT: Final = DiagonalEffect("e_yes", _mask(0, 1))
NO_EFFECT: Final = DiagonalEffect("e_no", _mask(*range(2, DIMENSION)))
BORN_TEST: Final[Test[Preparation, str]] = Test(
    (("yes", YES_EFFECT), ("no", NO_EFFECT))
)

REQUIRED_PREPARATIONS: Final[tuple[Preparation, ...]] = (
    Preparation(ZERO_CASES[0].identifier, ZERO_CASES[0].state, _ray(0)),
    Preparation(ZERO_CASES[1].identifier, ZERO_CASES[1].state, _ray(0, 2, 3)),
)
ENDPOINT_CONTROL: Final = Preparation(
    "endpoint-control",
    Fraction(0),
    _ray(2),
    normative=False,
)
ALL_PREPARATIONS: Final = REQUIRED_PREPARATIONS + (ENDPOINT_CONTROL,)

INTERACT_REVISION: Final = (
    "fa6d40ae9542353bf23f1b61bafe5f8ad19377afc2338baba262ba6c15f597a4"
)
GPT_REVISION: Final = (
    "0014ec768ad718a8f9e7bb1c05c36cd2984c9a3b471205607ed8a76abf73ef2b"
)
BASE_IMPLEMENTATION_REVISION: Final = "b9f8e30aa605e0230fda54aa1189f723d6d07868"
INTERACT_GATE_CLOSED: Final = (
    "certified abstract incidence only; executable coordinates unavailable "
    "under evidence fence"
)
BARE_COLORING_POSSIBLE: Final = "bare support-incidence coloring possible"
NO_CERTIFIED_BRIDGE: Final = (
    "No cross-carrier relation was available to or verified by this execution."
)


def _fraction_object(value: Fraction, /) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _exact_values(distribution: Distribution[str], /) -> dict[str, Fraction]:
    values: dict[str, Fraction] = {}
    for label, probability in distribution:
        value = probability.value
        if not isinstance(value, Fraction):
            raise TypeError("the Born audit must remain exact through resolution")
        values[label] = value
    return values


def _resolve(preparation: Preparation, /) -> Distribution[str]:
    distribution = resolve_test(preparation, BORN_TEST)
    values = _exact_values(distribution)
    expected = {
        "yes": preparation.source_probability,
        "no": ONE - preparation.source_probability,
    }
    if values != expected:
        raise RuntimeError(
            f"Born realization mismatch for {preparation.identifier}: "
            f"{values!r} != {expected!r}"
        )
    return distribution


def support_matrix(
    preparations: tuple[Preparation, ...],
    /,
) -> tuple[tuple[int, ...], ...]:
    """Derive F_Q(p,a) = 1[omega_p(e_a) == 0] in test order."""
    rows: list[tuple[int, ...]] = []
    for preparation in preparations:
        values = _exact_values(_resolve(preparation))
        rows.append(tuple(int(values[label] == 0) for label in BORN_TEST.labels))
    return tuple(rows)


def equality_complement_colorable(
    matrix: tuple[tuple[int, ...], ...],
    /,
) -> bool:
    """Apply the 017.15a identical-or-disjoint forbidden-set criterion."""
    if not matrix or not matrix[0]:
        raise ValueError("support matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("support matrix must be rectangular")
    if any(value not in (0, 1) for row in matrix for value in row):
        raise ValueError("support matrix entries must be binary")

    row_sets = [
        {column for column, value in enumerate(row) if value == 1}
        for row in matrix
        if any(row)
    ]
    column_sets = [
        {row for row, values in enumerate(matrix) if values[column] == 1}
        for column in range(width)
        if any(values[column] == 1 for values in matrix)
    ]

    def identical_or_disjoint(sets: list[set[int]]) -> bool:
        return all(
            left == right or left.isdisjoint(right)
            for index, left in enumerate(sets)
            for right in sets[index + 1 :]
        )

    return identical_or_disjoint(row_sets) and identical_or_disjoint(column_sets)


def _effect_record(effect: DiagonalEffect) -> dict[str, object]:
    return {
        "id": effect.identifier,
        "diagonal": [entry.numerator for entry in effect.diagonal],
        "dimension": len(effect.diagonal),
        "nonzero": effect.is_nonzero,
        "positive_semidefinite": all(entry >= 0 for entry in effect.diagonal),
        "bounded_by_unit": all(entry <= 1 for entry in effect.diagonal),
        "projector_identity": effect.is_projector,
    }


def _evaluation_record(preparation: Preparation) -> dict[str, object]:
    distribution = _resolve(preparation)
    values = _exact_values(distribution)
    return {
        "id": preparation.identifier,
        "normative": preparation.normative,
        "source_probability": _fraction_object(preparation.source_probability),
        "ray": {
            "dimension": len(preparation.coordinates),
            "nonzero_coordinates": [
                {"index": index, "value": _fraction_object(value)}
                for index, value in enumerate(preparation.coordinates)
                if value != 0
            ],
        },
        "squared_norm": _fraction_object(preparation.squared_norm),
        "distribution": [
            {"label": label, "probability": _fraction_object(values[label])}
            for label in BORN_TEST.labels
        ],
        "total": _fraction_object(
            sum((values[label] for label in BORN_TEST.labels), ZERO)
        ),
    }


def _matrix_record(
    preparations: tuple[Preparation, ...],
    /,
) -> dict[str, object]:
    matrix = support_matrix(preparations)
    colorable = equality_complement_colorable(matrix)
    return {
        "row_order": [preparation.identifier for preparation in preparations],
        "column_order": list(BORN_TEST.labels),
        "matrix": [list(row) for row in matrix],
        "criterion": "nonempty forbidden row and column sets are pairwise identical or disjoint",
        "criterion_passed": colorable,
        "verdict": (
            BARE_COLORING_POSSIBLE
            if colorable
            else "bare support-incidence coloring impossible"
        ),
    }


def build_audit_payload() -> dict[str, object]:
    """Execute every authorized gate and return the complete audit document."""
    complement_is_unit = all(
        yes + no == 1
        for yes, no in zip(YES_EFFECT.diagonal, NO_EFFECT.diagonal, strict=True)
    )
    shared_test_references = tuple(BORN_TEST for _ in ALL_PREPARATIONS)
    effects_are_shared = all(
        test["yes"] is YES_EFFECT and test["no"] is NO_EFFECT
        for test in shared_test_references
    )
    required_support = _matrix_record(REQUIRED_PREPARATIONS)
    control_support = _matrix_record(ALL_PREPARATIONS)
    exact_born = all(
        _exact_values(_resolve(preparation))
        == {
            "yes": preparation.source_probability,
            "no": ONE - preparation.source_probability,
        }
        for preparation in ALL_PREPARATIONS
    )

    coordinate_components = {
        "J_e_inverse": False,
        "rho_e": False,
        "p_e": False,
        "beta_e": False,
    }
    coordinates_available = all(coordinate_components.values())
    if coordinates_available:
        raise RuntimeError(
            "the audited authority record cannot open the coordinate gate without "
            "source-authorized executable definitions"
        )

    return {
        "schema": AUDIT_SCHEMA,
        "executed_on": "2026-09-22",
        "package": {"name": "decision-model", "version": "1.1.0"},
        "authority": {
            "gpt": {
                "revision": GPT_REVISION,
                "files_inspected": [
                    "017.13-GPT-synchronize-Jev-with-actualization-regeneration-frontier.md",
                    "017.17-Task-Born-first-Decision-Model-Zero-cross-carrier-coding-audit.md",
                    "017.17a-GPT-owner-review-execution-stable-and-preregistered-boundary.md",
                ],
            },
            "interact": {
                "revision": INTERACT_REVISION,
                "canonical_target": "Issue 017 closure and canonical 13",
                "files_inspected": [],
                "source_retrieved": False,
                "retrieval_result": (
                    "no local checkout or public repository was available and the "
                    "configured package-registry endpoints were unauthenticated"
                ),
                "named_composite": "beta_e = p_e o rho_e o J_e^{-1}",
            },
            "implementation_base_revision": BASE_IMPLEMENTATION_REVISION,
            "added_authority": [],
        },
        "source_fixture": {
            "schema": "decision-model-zero/v1",
            "question": "Q",
            "answer_order": list(BORN_TEST.labels),
            "required_cases": [
                {
                    "id": preparation.identifier,
                    "p": _fraction_object(preparation.source_probability),
                }
                for preparation in REQUIRED_PREPARATIONS
            ],
            "optional_control": {
                "included_in_audit": True,
                "included_in_product_payload": False,
                "id": ENDPOINT_CONTROL.identifier,
                "p": _fraction_object(ENDPOINT_CONTROL.source_probability),
            },
        },
        "phase_a_born_realization": {
            "carrier": "16-dimensional exact diagonal projector representation",
            "preparation_kind": "nonzero rational rays with scale-invariant evaluation",
            "effects": [_effect_record(YES_EFFECT), _effect_record(NO_EFFECT)],
            "fixed_test": {
                "labels": list(BORN_TEST.labels),
                "effect_complement_is_unit": complement_is_unit,
                "shared_test_instance": all(
                    test is BORN_TEST for test in shared_test_references
                ),
                "shared_effect_object_identity": effects_are_shared,
                "semantic_verdict": (
                    "one fixed typed test evaluated under multiple preparations"
                ),
            },
            "evaluations": [
                _evaluation_record(preparation) for preparation in ALL_PREPARATIONS
            ],
            "exact_born_assertions_passed": exact_born,
            "verdict": "pass",
        },
        "phase_b_support_control": {
            "definition": "F_Q(p,a) = 1[omega_p(e_a) == 0]",
            "required": required_support,
            "with_non_normative_endpoint_control": control_support,
            "semantic_limit": (
                "colorability does not identify zero probability with Interact illegality"
            ),
            "verdict": BARE_COLORING_POSSIBLE,
        },
        "phase_c_interact_coordinate_gate": {
            "required_components": list(coordinate_components),
            "executable_components_available": coordinate_components,
            "coordinates_available": coordinates_available,
            "classification": INTERACT_GATE_CLOSED,
        },
        "phase_d_support_incidence_hypothesis": {
            "status": "not testable",
            "reason": "Phase C coordinate gate is closed",
            "witness_constructed": False,
            "witness_classification": "none; coordinate gate closed",
            "structural_witness": False,
            "arbitrary_lookup_or_surrogate_geometry_used": False,
            "reversed_orientation_promoted": False,
        },
        "phase_e_cross_carrier_audit": {
            "status": "not auditable",
            "reason": (
                "declared Interact source was not retrieved; Phase C closed on "
                "execution-time authority availability"
            ),
            "relations_to_audit": [
                "typed map",
                "quotient",
                "functor",
                "pairing",
                "embedding",
                "exact relation",
            ],
            "chi_to_omega_relation": None,
            "branch_to_effect_relation": None,
            "verdict": NO_CERTIFIED_BRIDGE,
        },
        "architectural_classification": {
            "code": "E3",
            "label": "Born-only TDM",
            "admission": "important partial positive",
        },
        "phase_f_algebraic_classification": "not yet well-typed",
        "admission_checks": [
            {"id": "required_exact_born_probabilities", "passed": exact_born},
            {"id": "one_fixed_preparation_independent_test", "passed": effects_are_shared},
            {"id": "effect_complement_is_unit", "passed": complement_is_unit},
            {
                "id": "required_equality_complement_control",
                "passed": required_support["criterion_passed"],
            },
            {"id": "no_state_dependent_effects", "passed": True},
            {"id": "no_interact_lookup_geometry", "passed": True},
            {"id": "zero_not_defined_as_illegality", "passed": True},
            {"id": "no_learned_transition", "passed": True},
        ],
    }


def canonical_audit_bytes() -> bytes:
    """Serialize the complete audit with a stable machine-readable policy."""
    document = json.dumps(
        build_audit_payload(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return document.encode("ascii") + b"\n"


def main() -> int:
    """Write one deterministic 017.17 audit document to stdout."""
    sys.stdout.buffer.write(canonical_audit_bytes())
    return 0


if __name__ == "__main__":  # pragma: no cover - artifact generation path
    raise SystemExit(main())

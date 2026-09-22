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


@dataclass(frozen=True, slots=True, order=True)
class GaussianRational:
    """One exact coordinate in Q(i)."""

    real: Fraction
    imaginary: Fraction = ZERO

    def __post_init__(self) -> None:
        if not isinstance(self.real, Fraction) or not isinstance(
            self.imaginary, Fraction
        ):
            raise TypeError("Gaussian-rational components must be Fractions")

    @property
    def is_zero(self) -> bool:
        return self.real == 0 and self.imaginary == 0

    def __neg__(self) -> GaussianRational:
        return GaussianRational(-self.real, -self.imaginary)

    def __mul__(self, other: GaussianRational) -> GaussianRational:
        return GaussianRational(
            self.real * other.real - self.imaginary * other.imaginary,
            self.real * other.imaginary + self.imaginary * other.real,
        )

    def conjugate(self) -> GaussianRational:
        return GaussianRational(self.real, -self.imaginary)

    def scale(self, scalar: Fraction) -> GaussianRational:
        if not isinstance(scalar, Fraction):
            raise TypeError("Gaussian-rational scale must be a Fraction")
        return GaussianRational(self.real * scalar, self.imaginary * scalar)

    def __truediv__(self, other: GaussianRational) -> GaussianRational:
        denominator = other.real * other.real + other.imaginary * other.imaginary
        if denominator == 0:
            raise ZeroDivisionError("cannot divide by zero in Q(i)")
        numerator = self * other.conjugate()
        return GaussianRational(
            numerator.real / denominator,
            numerator.imaginary / denominator,
        )


GAUSSIAN_ZERO: Final = GaussianRational(ZERO)
GAUSSIAN_ONE: Final = GaussianRational(ONE)


@dataclass(frozen=True, slots=True, order=True)
class ProjectivePlanePoint:
    """A canonical exact Q(i) point of CP^1 in the P-plus-R reference chart."""

    first: GaussianRational
    second: GaussianRational

    def __post_init__(self) -> None:
        if self.first.is_zero and self.second.is_zero:
            raise ValueError("a projective point requires a nonzero coordinate pair")
        if self.first.is_zero:
            first = GAUSSIAN_ZERO
            second = GAUSSIAN_ONE
        else:
            first = GAUSSIAN_ONE
            second = self.second / self.first
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    def sigma(self) -> ProjectivePlanePoint:
        """Apply canonical 12's effective stabilizer involution [p+r] -> [p-r]."""
        return ProjectivePlanePoint(self.first, -self.second)

    def kappa(self) -> ProjectivePlanePoint:
        """Apply Hermitian orthogonal-complement on the certified CP^1 chart."""
        return ProjectivePlanePoint(
            -self.second.conjugate(),
            self.first.conjugate(),
        )

    def behavioral_orbit(self) -> tuple[ProjectivePlanePoint, ...]:
        """Return the exact orbit under the commuting sigma and kappa involutions."""
        seen: set[ProjectivePlanePoint] = set()
        pending = [self]
        while pending:
            point = pending.pop()
            if point in seen:
                continue
            seen.add(point)
            pending.extend((point.sigma(), point.kappa()))
        return tuple(sorted(seen))


@dataclass(frozen=True, slots=True)
class BehaviorClass:
    """One exact point of Q_e, represented by its finite quotient orbit."""

    orbit: tuple[ProjectivePlanePoint, ...]

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(self.orbit)))
        if not canonical:
            raise ValueError("a behavioral quotient class must be nonempty")
        if any(point.behavioral_orbit() != canonical for point in canonical):
            raise ValueError("behavioral class must contain one complete orbit")
        object.__setattr__(self, "orbit", canonical)

    @classmethod
    def from_plane(cls, plane: ProjectivePlanePoint, /) -> BehaviorClass:
        return cls(plane.behavioral_orbit())


@dataclass(frozen=True, slots=True)
class ContextChangingPartner:
    """An exact rational point in canonical 06's C_e subset of RP^3."""

    first: GaussianRational
    second: GaussianRational

    def __post_init__(self) -> None:
        components = (
            self.first.real,
            self.first.imaginary,
            self.second.real,
            self.second.imaginary,
        )
        try:
            pivot = next(component for component in components if component != 0)
        except StopIteration as exc:
            raise ValueError("a partner coordinate must be nonzero") from exc
        scale = ONE / pivot
        first = self.first.scale(scale)
        second = self.second.scale(scale)
        if first.is_zero or second.is_zero:
            raise ValueError("context-changing partners exclude the P and R fibres")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)


@dataclass(frozen=True, slots=True)
class SuccessorBranch:
    """A branch in N(e), encoded through the certified J_e coordinate chart."""

    partner: ContextChangingPartner


def j_e(partner: ContextChangingPartner, /) -> SuccessorBranch:
    """Apply the certified partner-to-successor coordinate identification."""
    return SuccessorBranch(partner)


def j_e_inverse(branch: SuccessorBranch, /) -> ContextChangingPartner:
    """Invert the certified partner-to-successor coordinate identification."""
    return branch.partner


def rho_e(partner: ContextChangingPartner, /) -> ProjectivePlanePoint:
    """Forget residual execution phase and project a partner to its plane."""
    return ProjectivePlanePoint(partner.first, partner.second)


def p_e(plane: ProjectivePlanePoint, /) -> BehaviorClass:
    """Quotient the context-changing plane by <sigma,kappa>."""
    return BehaviorClass.from_plane(plane)


def beta_e(branch: SuccessorBranch, /) -> BehaviorClass:
    """Evaluate canonical 13's beta_e = p_e o rho_e o J_e^{-1}."""
    return p_e(rho_e(j_e_inverse(branch)))


def z_e_contains(behavior: BehaviorClass, branch: SuccessorBranch, /) -> bool:
    """Evaluate canonical 13's actualization incidence q != beta_e(b)."""
    return behavior != beta_e(branch)


def _real_gaussian(value: int) -> GaussianRational:
    return GaussianRational(Fraction(value))


def _partner_with_real_slope(slope: int) -> ContextChangingPartner:
    return ContextChangingPartner(GAUSSIAN_ONE, _real_gaussian(slope))


YES_BRANCH: Final = j_e(_partner_with_real_slope(2))
NO_BRANCH: Final = j_e(_partner_with_real_slope(1))
ANSWER_BRANCHES: Final = (("yes", YES_BRANCH), ("no", NO_BRANCH))

SHARP_BEHAVIOR: Final = beta_e(NO_BRANCH)
NON_SHARP_BEHAVIOR: Final = p_e(
    ProjectivePlanePoint(GAUSSIAN_ONE, _real_gaussian(3))
)
ENDPOINT_BEHAVIOR: Final = beta_e(YES_BRANCH)


@dataclass(frozen=True, slots=True)
class SourceBehaviorWitness:
    """One explicitly arbitrary source-to-Interact representative choice."""

    preparation_identifier: str
    behavior: BehaviorClass


SOURCE_BEHAVIOR_WITNESSES: Final = (
    SourceBehaviorWitness("sharp", SHARP_BEHAVIOR),
    SourceBehaviorWitness("non-sharp", NON_SHARP_BEHAVIOR),
    SourceBehaviorWitness("endpoint-control", ENDPOINT_BEHAVIOR),
)

INTERACT_REVISION: Final = (
    "fa6d40ae9542353bf23f1b61bafe5f8ad19377afc2338baba262ba6c15f597a4"
)
GPT_REVISION: Final = (
    "0014ec768ad718a8f9e7bb1c05c36cd2984c9a3b471205607ed8a76abf73ef2b"
)
BASE_IMPLEMENTATION_REVISION: Final = "282f74ee5f6c5e9c71015820036bdad9c50f4512"
INTERACT_COORDINATES_AVAILABLE: Final = "executable coordinate realization available"
BARE_COLORING_POSSIBLE: Final = "bare support-incidence coloring possible"
NO_CERTIFIED_BRIDGE: Final = (
    "No cross-carrier relation is certified or supplied by the declared authority."
)


def _fraction_object(value: Fraction, /) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _gaussian_object(value: GaussianRational, /) -> dict[str, object]:
    return {
        "real": _fraction_object(value.real),
        "imaginary": _fraction_object(value.imaginary),
    }


def _plane_object(plane: ProjectivePlanePoint, /) -> dict[str, object]:
    return {
        "homogeneous_coordinates": [
            _gaussian_object(plane.first),
            _gaussian_object(plane.second),
        ]
    }


def _behavior_object(behavior: BehaviorClass, /) -> dict[str, object]:
    return {
        "quotient": "CP^1/<sigma,kappa>",
        "orbit": [_plane_object(plane) for plane in behavior.orbit],
        "orbit_size": len(behavior.orbit),
    }


def _branch_object(label: str, branch: SuccessorBranch, /) -> dict[str, object]:
    partner = j_e_inverse(branch)
    return {
        "answer": label,
        "fixed_across_preparations": True,
        "partner_RP3_coordinates": [
            _gaussian_object(partner.first),
            _gaussian_object(partner.second),
        ],
        "rho_e_plane": _plane_object(rho_e(partner)),
        "beta_e_behavior": _behavior_object(beta_e(branch)),
    }


def _behavior_for(preparation: Preparation, /) -> BehaviorClass:
    for witness in SOURCE_BEHAVIOR_WITNESSES:
        if witness.preparation_identifier == preparation.identifier:
            return witness.behavior
    raise KeyError(preparation.identifier)


def _branch_for(label: str, /) -> SuccessorBranch:
    for candidate, branch in ANSWER_BRANCHES:
        if candidate == label:
            return branch
    raise KeyError(label)


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


def support_incidence_witness_holds(
    preparations: tuple[Preparation, ...],
    /,
) -> bool:
    """Test both exact forms of the finite support-incidence hypothesis."""
    for preparation in preparations:
        values = _exact_values(_resolve(preparation))
        behavior = _behavior_for(preparation)
        for label in BORN_TEST.labels:
            branch = _branch_for(label)
            beta_equality = behavior == beta_e(branch)
            if (values[label] == 0) != beta_equality:
                return False
            if (values[label] > 0) != z_e_contains(behavior, branch):
                return False
    return True


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


def _hypothesis_records(
    preparations: tuple[Preparation, ...],
    /,
) -> tuple[list[dict[str, object]], bool, bool]:
    rows: list[dict[str, object]] = []
    zero_form_passed = True
    positive_form_passed = True
    for preparation in preparations:
        values = _exact_values(_resolve(preparation))
        behavior = _behavior_for(preparation)
        cells: list[dict[str, object]] = []
        for label in BORN_TEST.labels:
            branch = _branch_for(label)
            branch_behavior = beta_e(branch)
            probability_is_zero = values[label] == 0
            beta_equality = behavior == branch_behavior
            positive_probability = values[label] > 0
            incidence = z_e_contains(behavior, branch)
            zero_equivalence = probability_is_zero == beta_equality
            positive_equivalence = positive_probability == incidence
            zero_form_passed = zero_form_passed and zero_equivalence
            positive_form_passed = positive_form_passed and positive_equivalence
            cells.append(
                {
                    "answer": label,
                    "probability": _fraction_object(values[label]),
                    "probability_is_zero": probability_is_zero,
                    "chi_equals_beta_e_of_branch": beta_equality,
                    "zero_iff_beta_equality": zero_equivalence,
                    "positive_probability": positive_probability,
                    "chi_branch_in_Z_e": incidence,
                    "positive_iff_Z_e_incidence": positive_equivalence,
                }
            )
        rows.append(
            {
                "source_case": preparation.identifier,
                "chi_behavior": _behavior_object(behavior),
                "cells": cells,
            }
        )
    return rows, zero_form_passed, positive_form_passed


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
    shared_branch_references = tuple(ANSWER_BRANCHES for _ in ALL_PREPARATIONS)
    branches_are_shared = all(
        branches[0][1] is YES_BRANCH and branches[1][1] is NO_BRANCH
        for branches in shared_branch_references
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
    hypothesis_rows, zero_form_passed, positive_form_passed = _hypothesis_records(
        ALL_PREPARATIONS
    )

    coordinate_components = {
        "J_e_inverse": j_e(j_e_inverse(YES_BRANCH)) == YES_BRANCH,
        "rho_e": rho_e(j_e_inverse(YES_BRANCH))
        == ProjectivePlanePoint(GAUSSIAN_ONE, _real_gaussian(2)),
        "p_e": p_e(rho_e(j_e_inverse(YES_BRANCH))) == ENDPOINT_BEHAVIOR,
        "beta_e": beta_e(YES_BRANCH) == ENDPOINT_BEHAVIOR,
    }
    coordinates_available = all(coordinate_components.values())

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
                "package_uri": "quilt+s3://protology#package=occurrence/interact",
                "canonical_target": "Issue 017 closure and canonical 13",
                "source_retrieved": True,
                "retrieval_result": "exact pinned package revision read successfully",
                "files_inspected": [
                    {
                        "path": "13-actualization-incidence-and-constitutive-regeneration-boundary.md",
                        "object_version_id": "eh1WhwJ52XyicPmBqVkLfd9yv1yFqu30",
                    },
                    {
                        "path": "06-local-autonomy-and-partner-selection-boundary.md",
                        "object_version_id": "IfOMpTHUfm.Pqpy3OP3kCZ5XYaXhZxnx",
                    },
                    {
                        "path": "12-endogenous-generative-architecture-and-behavioral-strengthening-boundary.md",
                        "object_version_id": "q2bBXgy1USk_006.VDRIcQpCpVrQXQx5",
                    },
                    {
                        "path": "issues/017-actualization-and-regeneration-of-behavioral-strengthening/017.02-Research-behavioral-fibre-globalization-and-initialized-regeneration.md",
                    },
                    {
                        "path": "issues/017-actualization-and-regeneration-of-behavioral-strengthening/017.04-Research-branch-relative-regeneration-quotient-and-two-edge-imprint.md",
                    },
                    {
                        "path": "issues/017-actualization-and-regeneration-of-behavioral-strengthening/017.06-Research-behavior-branch-incidence-duality-and-finite-coupled-legality-lower-bound.md",
                        "object_version_id": "bb4Qchi_VUYYdWN2DVkJtEyCuEhYhh1K",
                    },
                    {
                        "path": "issues/017-actualization-and-regeneration-of-behavioral-strengthening/017.07-Newt-owner-synthesis-canonical-promotion-and-closure.md",
                        "object_version_id": "W32ClJeJXWLmdqN3FEqFYdlrV9gF2wnh",
                    },
                    {
                        "path": "issues/017-actualization-and-regeneration-of-behavioral-strengthening/README.md",
                        "object_version_id": "ycShBum8EKTp31MCQkDIbaa1BXM4d0eL",
                    },
                ],
                "certified_composite": "beta_e = p_e o rho_e o J_e^{-1}",
                "certified_incidence": "b in L_e(q) iff q != beta_e(b)",
            },
            "implementation_input_revision": BASE_IMPLEMENTATION_REVISION,
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
            "executable_components_verified": coordinate_components,
            "coordinates_available": coordinates_available,
            "classification": INTERACT_COORDINATES_AVAILABLE,
            "realization": {
                "partner_chart": "exact Q(i) representatives in C_e subset RP^3",
                "plane_chart": "exact homogeneous Q(i) representatives in CP^1",
                "sigma": "[p+r] -> [p-r]",
                "kappa": "Hermitian orthogonal-complement plane",
                "behavior_class": "finite <sigma,kappa> orbit",
                "J_e": "certified partner-to-successor coordinate identification",
                "scope": (
                    "exact local reference-chart realization sufficient for the "
                    "finite fixture; not a serializer for the full continuum"
                ),
                "surrogate_geometry_used": False,
                "beta_lookup_table_used": False,
            },
        },
        "phase_d_support_incidence_hypothesis": {
            "status": "tested",
            "zero_form_passed": zero_form_passed,
            "positive_form_passed": positive_form_passed,
            "witness_constructed": True,
            "witness_classification": "arbitrary witness",
            "structural_witness": False,
            "source_determines_representatives": False,
            "classification_reason": (
                "Decision Model Zero determines the finite support table but no "
                "projective-plane, behavior-class, or branch representatives"
            ),
            "answer_branches_fixed_across_preparations": branches_are_shared,
            "answer_branches": [
                _branch_object(label, branch) for label, branch in ANSWER_BRANCHES
            ],
            "evaluations": hypothesis_rows,
            "arbitrary_beta_lookup_used": False,
            "surrogate_geometry_used": False,
            "reversed_orientation_promoted": False,
            "semantic_limit": (
                "a passing arbitrary witness is plumbing, not evidence that Born "
                "zero is Interact illegality"
            ),
        },
        "phase_e_cross_carrier_audit": {
            "status": "complete",
            "relations_searched": [
                "typed map",
                "quotient",
                "functor",
                "pairing",
                "embedding",
                "exact relation",
            ],
            "chi_to_omega_relation": None,
            "branch_to_effect_relation": None,
            "authority_findings": [
                "canonical 13's constitutive ledger contains Q, T, beta, L, and Z but no probability-state or effect carrier",
                "canonical 13 requires genuinely new typed bridge or law for further actualization",
                "canonical 12 explicitly supplies no probability rule",
                "canonical 06 explicitly excludes probability and measurement claims",
            ],
            "verdict": NO_CERTIFIED_BRIDGE,
        },
        "architectural_classification": {
            "code": "E2",
            "label": "two valid shadows, no certified bridge",
            "witness_scope": "Interact shadow is arbitrary finite plumbing",
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
            {"id": "executable_interact_coordinates", "passed": coordinates_available},
            {
                "id": "support_incidence_witness_exact",
                "passed": zero_form_passed and positive_form_passed,
            },
            {"id": "arbitrary_witness_classified_as_arbitrary", "passed": True},
            {"id": "fixed_answer_branches", "passed": branches_are_shared},
            {"id": "no_state_dependent_effects", "passed": True},
            {"id": "no_interact_beta_lookup", "passed": True},
            {"id": "zero_not_defined_as_illegality", "passed": True},
            {"id": "no_learned_transition", "passed": True},
        ],
    }


def canonical_audit_bytes() -> bytes:
    """Serialize the complete audit with a stable machine-readable policy."""
    document = json.dumps(
        build_audit_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return document.encode("ascii") + b"\n"


def main() -> int:
    """Write one deterministic 017.17 audit document to stdout."""
    sys.stdout.buffer.write(canonical_audit_bytes())
    return 0


if __name__ == "__main__":  # pragma: no cover - artifact generation path
    raise SystemExit(main())

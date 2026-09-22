from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from decision_model._audit_01717 import (
    ALL_PREPARATIONS,
    ANSWER_BRANCHES,
    BARE_COLORING_POSSIBLE,
    BORN_TEST,
    ENDPOINT_BEHAVIOR,
    ENDPOINT_CONTROL,
    GAUSSIAN_ONE,
    INTERACT_COORDINATES_AVAILABLE,
    NO_BRANCH,
    NO_EFFECT,
    NON_SHARP_BEHAVIOR,
    REQUIRED_PREPARATIONS,
    SHARP_BEHAVIOR,
    YES_BRANCH,
    YES_EFFECT,
    ContextChangingPartner,
    GaussianRational,
    ProjectivePlanePoint,
    beta_e,
    build_audit_payload,
    canonical_audit_bytes,
    equality_complement_colorable,
    j_e,
    j_e_inverse,
    p_e,
    rho_e,
    support_incidence_witness_holds,
    support_matrix,
    z_e_contains,
)
from decision_model._zero import build_zero_payload

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "017.17-born-first-cross-carrier-audit.json"
)


def test_one_fixed_formal_test_resolves_every_preparation_exactly() -> None:
    assert BORN_TEST["yes"] is YES_EFFECT
    assert BORN_TEST["no"] is NO_EFFECT
    assert YES_EFFECT.is_projector
    assert NO_EFFECT.is_projector
    assert YES_EFFECT.is_nonzero
    assert NO_EFFECT.is_nonzero
    assert tuple(yes + no for yes, no in zip(YES_EFFECT.diagonal, NO_EFFECT.diagonal)) == (
        Fraction(1),
    ) * 16

    expected = (
        (Fraction(1), Fraction(0)),
        (Fraction(1, 3), Fraction(2, 3)),
        (Fraction(0), Fraction(1)),
    )
    actual = tuple(
        (YES_EFFECT(preparation), NO_EFFECT(preparation))
        for preparation in ALL_PREPARATIONS
    )
    assert actual == expected
    assert tuple(preparation.squared_norm for preparation in ALL_PREPARATIONS) == (
        Fraction(1),
        Fraction(3),
        Fraction(1),
    )


def test_zero_support_is_derived_and_equality_complement_colorable() -> None:
    required = support_matrix(REQUIRED_PREPARATIONS)
    with_control = support_matrix(ALL_PREPARATIONS)

    assert required == ((0, 1), (0, 0))
    assert with_control == ((0, 1), (0, 0), (1, 0))
    assert equality_complement_colorable(required)
    assert equality_complement_colorable(with_control)
    assert not equality_complement_colorable(((1, 1), (1, 0)))


def test_endpoint_control_remains_non_normative_and_outside_zero_payload() -> None:
    assert not ENDPOINT_CONTROL.normative
    assert ENDPOINT_CONTROL.source_probability == Fraction(0)
    assert [case["id"] for case in build_zero_payload()["cases"]] == [
        "sharp",
        "non-sharp",
    ]


def test_exact_interact_coordinate_maps_realize_certified_composite() -> None:
    positive = ContextChangingPartner(
        GaussianRational(Fraction(1)),
        GaussianRational(Fraction(2)),
    )
    negative = ContextChangingPartner(
        GaussianRational(Fraction(-1)),
        GaussianRational(Fraction(-2)),
    )
    phased = ContextChangingPartner(
        GaussianRational(Fraction(0), Fraction(1)),
        GaussianRational(Fraction(0), Fraction(2)),
    )
    assert positive == negative
    assert positive != phased

    branch = j_e(positive)
    phased_branch = j_e(phased)
    plane = ProjectivePlanePoint(GAUSSIAN_ONE, GaussianRational(Fraction(2)))
    assert branch != phased_branch
    assert j_e_inverse(branch) == positive
    assert j_e(j_e_inverse(branch)) == branch
    assert rho_e(positive) == plane
    assert rho_e(phased) == plane
    assert beta_e(branch) == beta_e(phased_branch)
    assert p_e(plane) == p_e(plane.sigma())
    assert p_e(plane) == p_e(plane.kappa())
    assert plane.sigma().sigma() == plane
    assert plane.kappa().kappa() == plane
    assert beta_e(branch) == p_e(rho_e(j_e_inverse(branch)))


def test_support_incidence_witness_is_exact_fixed_and_arbitrary() -> None:
    assert ANSWER_BRANCHES == (("yes", YES_BRANCH), ("no", NO_BRANCH))
    assert beta_e(NO_BRANCH) == SHARP_BEHAVIOR
    assert beta_e(YES_BRANCH) == ENDPOINT_BEHAVIOR
    assert len(SHARP_BEHAVIOR.orbit) == 2
    assert len(ENDPOINT_BEHAVIOR.orbit) == 4
    assert len(NON_SHARP_BEHAVIOR.orbit) == 4
    assert len({SHARP_BEHAVIOR, NON_SHARP_BEHAVIOR, ENDPOINT_BEHAVIOR}) == 3

    assert not z_e_contains(SHARP_BEHAVIOR, NO_BRANCH)
    assert z_e_contains(SHARP_BEHAVIOR, YES_BRANCH)
    assert z_e_contains(NON_SHARP_BEHAVIOR, YES_BRANCH)
    assert z_e_contains(NON_SHARP_BEHAVIOR, NO_BRANCH)
    assert not z_e_contains(ENDPOINT_BEHAVIOR, YES_BRANCH)
    assert z_e_contains(ENDPOINT_BEHAVIOR, NO_BRANCH)
    assert support_incidence_witness_holds(REQUIRED_PREPARATIONS)
    assert support_incidence_witness_holds(ALL_PREPARATIONS)


def test_audit_records_every_gate_without_overstating_the_witness() -> None:
    payload = build_audit_payload()

    assert payload["schema"] == "decision-model-01717-audit/v1"
    assert payload["phase_a_born_realization"]["verdict"] == "pass"
    fixed_test = payload["phase_a_born_realization"]["fixed_test"]
    assert fixed_test["shared_test_instance"]
    assert fixed_test["shared_effect_object_identity"]
    assert (
        fixed_test["semantic_verdict"]
        == "one fixed typed test evaluated under multiple preparations"
    )
    assert payload["phase_b_support_control"]["verdict"] == BARE_COLORING_POSSIBLE

    coordinates = payload["phase_c_interact_coordinate_gate"]
    assert coordinates["classification"] == INTERACT_COORDINATES_AVAILABLE
    assert coordinates["coordinates_available"]
    assert all(coordinates["executable_components_verified"].values())
    assert not coordinates["realization"]["surrogate_geometry_used"]
    assert not coordinates["realization"]["beta_lookup_table_used"]

    hypothesis = payload["phase_d_support_incidence_hypothesis"]
    assert hypothesis["status"] == "tested"
    assert hypothesis["zero_form_passed"]
    assert hypothesis["positive_form_passed"]
    assert hypothesis["witness_constructed"]
    assert hypothesis["witness_classification"] == "arbitrary witness"
    assert not hypothesis["structural_witness"]
    assert not hypothesis["source_determines_representatives"]
    assert hypothesis["answer_branches_fixed_across_preparations"]
    assert not hypothesis["arbitrary_beta_lookup_used"]
    assert not hypothesis["surrogate_geometry_used"]

    bridge = payload["phase_e_cross_carrier_audit"]
    assert bridge["status"] == "complete"
    assert bridge["relations_searched"] == [
        "typed map",
        "quotient",
        "functor",
        "pairing",
        "embedding",
        "exact relation",
    ]
    assert (
        bridge["verdict"]
        == "No cross-carrier relation is certified or supplied by the declared authority."
    )
    assert payload["architectural_classification"] == {
        "code": "E2",
        "label": "two valid shadows, no certified bridge",
        "witness_scope": "Interact shadow is arbitrary finite plumbing",
        "admission": "important partial positive",
    }
    assert payload["phase_f_algebraic_classification"] == "not yet well-typed"
    assert all(check["passed"] for check in payload["admission_checks"])


def test_committed_artifact_is_exactly_the_executable_audit() -> None:
    expected = canonical_audit_bytes()

    assert ARTIFACT.read_bytes() == expected
    assert json.loads(expected) == build_audit_payload()

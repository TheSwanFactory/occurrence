from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from decision_model._audit_01717 import (
    ALL_PREPARATIONS,
    BARE_COLORING_POSSIBLE,
    BORN_TEST,
    ENDPOINT_CONTROL,
    INTERACT_GATE_CLOSED,
    NO_EFFECT,
    REQUIRED_PREPARATIONS,
    YES_EFFECT,
    build_audit_payload,
    canonical_audit_bytes,
    equality_complement_colorable,
    support_matrix,
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


def test_audit_records_every_gate_without_inventing_interact_geometry() -> None:
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
    assert (
        payload["phase_c_interact_coordinate_gate"]["classification"]
        == INTERACT_GATE_CLOSED
    )
    hypothesis = payload["phase_d_support_incidence_hypothesis"]
    assert hypothesis["status"] == "not testable"
    assert not hypothesis["witness_constructed"]
    assert not hypothesis["arbitrary_lookup_or_surrogate_geometry_used"]
    bridge = payload["phase_e_cross_carrier_audit"]
    assert bridge["status"] == "not auditable"
    assert bridge["relations_to_audit"] == [
        "typed map",
        "quotient",
        "functor",
        "pairing",
        "embedding",
        "exact relation",
    ]
    assert (
        bridge["verdict"]
        == "No cross-carrier relation was available to or verified by this execution."
    )
    assert payload["architectural_classification"] == {
        "code": "E3",
        "label": "Born-only TDM",
        "admission": "important partial positive",
    }
    assert payload["phase_f_algebraic_classification"] == "not yet well-typed"


def test_committed_artifact_is_exactly_the_executable_audit() -> None:
    expected = canonical_audit_bytes()

    assert ARTIFACT.read_bytes() == expected
    assert json.loads(expected) == build_audit_payload()

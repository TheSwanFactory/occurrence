"""CI-safe tests for the Issue-008 reporting contract (no torch, no algebra).

Pins the 007.05 interface layering and the 008.01 firewalls that a multi-token
result must respect before it may claim learned composition.
"""

from __future__ import annotations

import pytest

from experiments.tlm_multitoken.reporting import (
    REQUIRED_BASELINES,
    REQUIRED_DIAGNOSTICS,
    ArmReport,
    DecoderAudit,
    Hypothesis,
    InterfaceLayer,
    validate_arm,
    validate_suite,
)


def _arm(**overrides) -> ArmReport:
    defaults = dict(
        arm="A",
        signature="E^3 R",
        layers=frozenset({InterfaceLayer.SELECTED_TREE}),
        claims=frozenset({Hypothesis.H_REL}),
        baselines=frozenset(REQUIRED_BASELINES),
    )
    defaults.update(overrides)
    return ArmReport(**defaults)


def test_well_formed_learned_policy_arm_passes():
    arm = _arm()
    assert arm.valid, arm.failures()
    validate_arm(arm)


def test_claiming_learned_selection_while_supplying_the_tree_is_rejected():
    arm = _arm(
        layers=frozenset({InterfaceLayer.SUPPLIED_TREE}),
        claims=frozenset({Hypothesis.H_REL}),
    )
    failures = arm.failures()
    assert any("without exercising selected_tree" in f for f in failures), failures


def test_supplied_and_selected_tree_cannot_coexist_in_one_arm():
    arm = _arm(
        layers=frozenset({InterfaceLayer.SUPPLIED_TREE, InterfaceLayer.SELECTED_TREE}),
        claims=frozenset({Hypothesis.H_REL}),
    )
    failures = arm.failures()
    assert any("both supplied_tree and selected_tree" in f for f in failures), failures


def test_native_claim_requires_recovered_denotations():
    arm = _arm(claims=frozenset({Hypothesis.H_REL, Hypothesis.H_NATIVE}))
    failures = arm.failures()
    assert any("H_native without exercising recovered_dens" in f for f in failures), failures


def test_weak_claim_is_not_falsified_by_a_failing_native_claim():
    """007.05 section 6: the three claims are independent."""
    weak = ArmReport(
        arm="B",
        signature="E^3 R",
        layers=frozenset({InterfaceLayer.SUPPLIED_TREE}),
        claims=frozenset({Hypothesis.H_WEAK}),
    )
    assert weak.valid, weak.failures()
    # The same arm may not additionally claim H_native.
    overreaching = ArmReport(
        arm="B",
        signature="E^3 R",
        layers=frozenset({InterfaceLayer.SUPPLIED_TREE}),
        claims=frozenset({Hypothesis.H_WEAK, Hypothesis.H_NATIVE}),
    )
    assert not overreaching.valid


def test_availability_rule_target_blocks_a_tree_selection_claim():
    """008.01 section 4: the 017.24 GroupedFirst trap, refused up front."""
    arm = _arm(target_is_availability_rule=True)
    failures = arm.failures()
    assert any("availability rule" in f for f in failures), failures


def test_learned_policy_without_deterministic_baselines_is_rejected():
    arm = _arm(baselines=frozenset({"GroupedFirst"}))
    failures = arm.failures()
    assert any("without baselines" in f for f in failures), failures
    assert "ForceSeq" in failures[0] and "AmbientMul" in failures[0]


def test_required_baselines_cover_the_008_list():
    assert REQUIRED_BASELINES == {
        "GroupedFirst",
        "ForceSeq",
        "SuppliedBracketingExecutor",
        "RandomLegalTree",
        "AmbientMul",
    }


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"contains_heldout_lookup": True}, "held-out answer lookup"),
        ({"computes_target_relation": True}, "computes the target relation"),
        ({"fitted_on": "test"}, "fitted on 'test'"),
        ({"fitted_on": "heldout"}, "fitted on 'heldout'"),
    ],
)
def test_decoder_validity_conditions(kwargs, fragment):
    """007.05 section 8 invalidity conditions."""
    audit = DecoderAudit(name="delta", **kwargs)
    assert not audit.valid
    assert any(fragment in f for f in audit.failures()), audit.failures()


@pytest.mark.parametrize("fitted_on", ["none", "train"])
def test_train_only_and_unfitted_decoders_are_admissible(fitted_on):
    assert DecoderAudit(name="delta", fitted_on=fitted_on).valid


def test_free_decoder_cannot_manufacture_native_result_identity():
    """008.01 section 5.D."""
    arm = _arm(
        layers=frozenset({InterfaceLayer.SELECTED_TREE, InterfaceLayer.RECOVERED_DENS}),
        claims=frozenset({Hypothesis.H_REL, Hypothesis.H_NATIVE}),
        decoders=(DecoderAudit(name="readout", fitted_on="train", free_parameters=4096),),
    )
    failures = arm.failures()
    assert any("free decoder" in f for f in failures), failures
    # Without the native claim the same decoder is fine.
    without_native = _arm(
        layers=frozenset({InterfaceLayer.SELECTED_TREE}),
        claims=frozenset({Hypothesis.H_REL}),
        decoders=(DecoderAudit(name="readout", fitted_on="train", free_parameters=4096),),
    )
    assert without_native.valid, without_native.failures()


def test_surrogate_must_be_rescored_exactly():
    assert not _arm(native_exact_scoring=False).valid
    assert not _arm(surrogate_rescored_exactly=False).valid


def test_suite_requires_the_section_9_diagnostics():
    arms = [_arm()]
    with pytest.raises(ValueError, match="missing required diagnostics"):
        validate_suite(arms, {})
    validate_suite(arms, {key: None for key in REQUIRED_DIAGNOSTICS})

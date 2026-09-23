"""Focused exact tests for the Issue 017.22 operational-equivalence audit.

Every assertion is exact. The tests are deliberately adversarial where the audit
makes a positive claim: they check the negative boundary too, so a vacuous pass
is visible.
"""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "audit_operational_equivalence.py"
SPEC = importlib.util.spec_from_file_location("audit_017_22", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules["audit_017_22"] = audit
SPEC.loader.exec_module(audit)

ZERO = Fraction(0)
ONE = Fraction(1)


@pytest.fixture(scope="module")
def source():
    schema, _ = audit.load_source()
    return schema


@pytest.fixture(scope="module")
def payload():
    return audit.build_payload()


# --- the pinned inputs ------------------------------------------------------


def test_source_and_prior_artifacts_are_byte_pinned():
    schema, document = audit.load_source()
    assert document["schema"] == audit.SOURCE_SCHEMA
    assert len(schema.inputs) == 49
    premises = audit.load_prior_premises()
    prior = audit.load_prior_audit()
    assert premises["schema"] == audit.PRIOR_PREMISE_SCHEMA
    assert prior["schema"] == audit.PRIOR_AUDIT_SCHEMA


def test_a_changed_source_is_refused(tmp_path):
    corrupted = tmp_path / "source_fixture.json"
    corrupted.write_bytes(audit.SOURCE_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        audit.load_source(corrupted)


# --- E_Sigma and the state quotient ----------------------------------------


def test_central_effects_are_effects_and_sum_to_the_unit():
    assert audit.Z_MATRIX.is_effect
    assert audit.Z_SCALAR.is_effect
    assert audit.Z_MATRIX.is_central
    assert audit.Z_SCALAR.is_central
    total = audit.subtract(audit.UNIT, audit.Z_MATRIX)
    assert audit._anonymous(total) == audit._anonymous(audit.Z_SCALAR)


def test_sigma_equivalence_is_exactly_equality_of_the_central_parameter():
    declared = (audit.Z_MATRIX, audit.Z_SCALAR)
    states = audit.probe_states()
    for left in states:
        for right in states:
            assert audit.sigma_equivalent(left, right, declared) == (
                left.central_parameter == right.central_parameter
            )


def test_the_declared_test_cannot_separate_a_fat_fiber():
    declared = (audit.Z_MATRIX, audit.Z_SCALAR)
    half = Fraction(1, 2)
    quarter = Fraction(1, 4)
    first = audit.State("a", half, ZERO, ZERO, half)
    second = audit.State("b", quarter, quarter, quarter, half)
    assert first.is_state and second.is_state
    assert audit._anonymous_state(first) != audit._anonymous_state(second)
    assert audit.sigma_equivalent(first, second, declared)
    separating = [
        effect
        for effect in audit.probe_effects()
        if first(effect) != second(effect)
    ]
    assert separating, "Eff(E) must separate states that the declared test cannot"


def test_eff_E_contains_a_linear_basis_so_S_E_is_minimal():
    rows = [list(effect.coordinates) for effect in audit.SEPARATING_BASIS]
    assert all(effect.is_effect for effect in audit.SEPARATING_BASIS)
    assert audit.determinant_4x4(rows) != 0


def test_lambda_zero_fiber_collapses_to_delta_s():
    grid = audit.rational_state_grid(5)
    collapsed = [state for state in grid if state.central_parameter == 0]
    assert collapsed
    for state in collapsed:
        assert audit._anonymous_state(state) == audit._anonymous_state(
            audit.DELTA_SCALAR
        )


# --- the invariant family ---------------------------------------------------


def test_invariant_states_are_exactly_the_omega_lambda_family():
    result = audit.audit_invariant_family()
    assert result["constraint_matrix"]["rank"] == 2
    assert result["constraint_matrix"]["kernel_is_spanned_by_those_two"]
    assert result["all_family_members_are_invariant"]
    assert result["all_family_members_are_states"]


def test_omega_lambda_endpoints_are_the_two_central_characters():
    assert audit._anonymous_state(audit.omega_lambda(ONE)) == audit._anonymous_state(
        audit.TAU_MATRIX
    )
    assert audit._anonymous_state(audit.omega_lambda(ZERO)) == audit._anonymous_state(
        audit.DELTA_SCALAR
    )


def test_non_invariant_states_are_detected_by_a_rational_probe():
    witness = audit.State("skew", ONE, ZERO, ZERO, ZERO)
    assert witness.is_state
    moved = [
        audit.push_state(witness, probe)
        for probe in audit.TARGET_AUTOMORPHISM_PROBES
    ]
    assert any(
        audit._anonymous_state(image) != audit._anonymous_state(witness)
        for image in moved
    )


def test_lambda_outside_the_unit_interval_is_not_a_state():
    assert not audit.omega_lambda(Fraction(-1, 3)).is_state
    assert not audit.omega_lambda(Fraction(4, 3)).is_state


# --- realization equivalence ------------------------------------------------


def test_the_two_orientations_induce_the_identical_public_family(source):
    first = audit.public_family(audit.ORIENTATION_ONE, source)
    second = audit.public_family(audit.ORIENTATION_TWO, source)
    assert first == second
    assert all(audit.is_distribution(row) for row in first.values())
    assert len(first) == 49


def test_the_orientations_are_internally_distinct():
    assert audit.ORIENTATION_ONE.state_for("incident") != audit.ORIENTATION_TWO.state_for(
        "incident"
    )
    assert audit.ORIENTATION_ONE.effect_for(
        "incident"
    ) != audit.ORIENTATION_TWO.effect_for("incident")


def test_non_invariant_rivals_reproduce_the_same_behavior(source):
    target = audit.public_family(audit.ORIENTATION_ONE, source)
    rivals = audit.sharp_rival_realizations()
    assert len(rivals) >= 2
    for rival in rivals:
        assert rival.test_is_normalized
        assert rival.test_slots_are_effects
        assert rival.states_are_states
        assert audit.public_family(rival, source) == target
    assert any(not rival.uses_only_central_effects for rival in rivals)
    assert any(not rival.uses_only_invariant_states for rival in rivals)


def test_no_target_automorphism_exchanges_the_central_idempotents():
    for probe in audit.TARGET_AUTOMORPHISM_PROBES:
        assert audit._anonymous(
            audit.push_element(audit.Z_MATRIX, probe)
        ) == audit._anonymous(audit.Z_MATRIX)
        assert audit._anonymous(
            audit.push_element(audit.Z_SCALAR, probe)
        ) == audit._anonymous(audit.Z_SCALAR)


def test_P4_forces_exactly_the_two_orientations():
    result = audit.audit_p4_forcing()
    assert result["search_space"]["search_is_not_vacuous"]
    assert result["solution_count"] == 2
    assert result["solutions_are_the_two_orientations"]
    assert result["centrality_is_derived_not_assumed"]


# --- ambient rays -----------------------------------------------------------


def test_distinct_ambient_rays_induce_the_same_state():
    first = audit.ambient_vector({1: audit.Q7_ONE, 2: audit.Q7(-ONE)})
    second = audit.ambient_vector({2: audit.Q7_ONE, 4: audit.Q7(-ONE)})
    assert not audit.ambient_proportional(first, second)
    left = audit.ambient_state_of(first)
    right = audit.ambient_state_of(second)
    assert audit._anonymous_state(left) == audit._anonymous_state(right)
    assert audit._anonymous_state(left) == audit._anonymous_state(audit.DELTA_SCALAR)
    for effect in audit.probe_effects():
        assert left(effect) == right(effect)


def test_the_cross_copy_convention_remains_load_bearing():
    cross = audit.frame_combination(ONE, ZERO, ZERO, ONE)
    same = audit.frame_combination(ONE, ZERO, ONE, ZERO)
    assert audit._anonymous_state(
        audit.ambient_state_of(cross)
    ) == audit._anonymous_state(audit.TAU_MATRIX)
    assert audit._anonymous_state(
        audit.ambient_state_of(same)
    ) != audit._anonymous_state(audit.TAU_MATRIX)


# --- successor feasibility --------------------------------------------------


def test_theta_is_identifiable_and_absorbs_the_orientation_bit(source):
    seen = set()
    for lambda_incident in (ZERO, Fraction(1, 3), Fraction(2, 3), ONE):
        for lambda_nonincident in (ZERO, Fraction(1, 4), ONE):
            first = audit.parametric_realization(
                lambda_incident, lambda_nonincident, 1
            )
            mirrored = audit.parametric_realization(
                ONE - lambda_incident, ONE - lambda_nonincident, 2
            )
            family = audit.public_family(first, source)
            assert family == audit.public_family(mirrored, source)
            key = audit._family_key(family, source.decision_labels)
            assert key not in seen
            seen.add(key)


# --- control and artifact integrity ----------------------------------------


def test_the_public_family_fails_closed_without_the_declared_relation(source):
    control = audit.make_relation_erased_control(source)
    assert len(control.inputs) == 49
    assert len(control.opaque_rows) == 21
    with pytest.raises(audit.MissingDeclaredRelation):
        audit.public_family(audit.ORIENTATION_ONE, control)


def test_committed_artifacts_match_the_executable_output():
    assert audit.ARTIFACT_PATH.read_bytes() == audit.canonical_audit_bytes()
    assert audit.TABLE_PATH.read_bytes() == audit.canonical_table_bytes()


def test_every_mechanical_check_passes(payload):
    checks = payload["mechanical_checks"]
    assert checks, "the audit must run at least one mechanical check"
    failed = sorted(name for name, ok in checks.items() if not ok)
    assert not failed


def test_the_table_has_both_orientations_for_every_input(payload):
    table = audit.canonical_table_bytes()
    assert b"orientation-incident-to-z_M" in table
    assert b"orientation-incident-to-z_s" in table
    digest = payload["artifacts"]["orientation_behavior_table"]["sha256"]
    assert digest == audit._sha256(table)
    assert payload["artifacts"]["orientation_behavior_table"]["rows"] == 98

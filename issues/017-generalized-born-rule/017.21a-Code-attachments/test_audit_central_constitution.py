from __future__ import annotations

import json
from fractions import Fraction

import pytest
from audit_central_constitution import (
    ALGEBRA_ZERO,
    ARTIFACT_PATH,
    BINDING_MATRIX_FIRST,
    BINDING_SCALAR_FIRST,
    BINDINGS,
    DELTA_SCALAR,
    Q7,
    Q7_ONE,
    Q7_SQRT7,
    RATIONAL_ROTATIONS,
    SOURCE_SHA256,
    TAU_MATRIX,
    UNIT,
    Z_MATRIX,
    Z_SCALAR,
    MissingDeclaredRelation,
    _anonymous,
    _anonymous_state,
    add,
    ambient_add,
    ambient_inner,
    ambient_state_of,
    associates_with_all,
    audit_ambient_boundary,
    audit_center,
    audit_compiler,
    audit_control,
    audit_equivariance,
    audit_premise_necessity,
    build_payload,
    canonical_audit_bytes,
    compile_datum,
    element,
    enumerate_automorphisms,
    enumerate_central_idempotents,
    evaluate,
    input_orbits,
    jordan_product,
    load_premises,
    load_source,
    make_relation_erased_control,
    omega_t,
    probe_elements,
    u0_vector,
    v0_vector,
    v8_vector,
)


def test_exact_q7_arithmetic_is_a_field_on_the_values_used() -> None:
    assert Q7_SQRT7 * Q7_SQRT7 == Q7(Fraction(7))
    assert (Q7_ONE + Q7_SQRT7) * (Q7_ONE - Q7_SQRT7) == Q7(Fraction(-6))
    assert Q7_SQRT7.scale(Fraction(1, 7)) * Q7_SQRT7 == Q7(Fraction(1))
    assert not Q7_SQRT7.is_rational
    with pytest.raises(ValueError):
        Q7_SQRT7.as_fraction()


def test_source_is_the_unmodified_017_20_freeze() -> None:
    source, document = load_source()

    assert SOURCE_SHA256 == (
        "20176b9043516e704fbffec84d01f515f008630aef9398d0f44d3cf96886d71c"
    )
    assert len(source.points) == len(source.lines) == 7
    assert source.incidence is not None and len(source.incidence) == 21
    assert len(source.inputs) == 49
    assert source.decision_labels == ("incident", "nonincident")
    assert "OT preparation or ray coordinate" in set(
        document["compiler_forbidden_semantics"]
    )


def test_added_premises_are_declared_before_use() -> None:
    premises = load_premises()

    identifiers = [row["id"] for row in premises["added_premises"]]
    assert identifiers == ["P1", "P2", "P3", "P4"]
    assert premises["deliberately_unresolved"]["expected_cardinality"] == 2
    assert len(premises["preregistered_falsifiers"]) >= 5
    assert "physical preparation or measurement realization" in premises[
        "declared_but_not_claimed"
    ]


def test_center_is_derived_and_has_exactly_two_minimal_idempotents() -> None:
    central = enumerate_central_idempotents()
    result = audit_center()

    assert len(central) == 4
    assert result["minimal_nonzero_proper_count"] == 2
    assert result["ambient_ranks"] == {"matrix": 4, "scalar": 12}
    assert result["ideal_dimensions"] == {"matrix": 3, "scalar": 1}

    assert jordan_product(Z_MATRIX, Z_SCALAR) == _anonymous(ALGEBRA_ZERO)
    assert add(Z_MATRIX, Z_SCALAR) == _anonymous(UNIT)
    assert Z_MATRIX.is_jordan_idempotent and Z_SCALAR.is_jordan_idempotent
    assert Z_MATRIX.is_effect and Z_SCALAR.is_effect

    probes = probe_elements()
    assert associates_with_all(Z_MATRIX, probes)
    assert associates_with_all(Z_SCALAR, probes)
    # A non-scalar block is not central, so the center is not an artifact of
    # convenient labeling.
    assert not associates_with_all(element("offdiag", Fraction(0), Fraction(1), Fraction(0), Fraction(0)), probes)
    assert not associates_with_all(element("rank1", Fraction(1), Fraction(0), Fraction(0), Fraction(0)), probes)


def test_invariant_states_are_unique_extensions_of_central_characters() -> None:
    for state in (TAU_MATRIX, DELTA_SCALAR):
        reference = _anonymous_state(state)
        assert state.is_state
        assert state.conjugate_by_reflection() == reference
        for cosine, sine in RATIONAL_ROTATIONS:
            assert state.conjugate_by_rotation(cosine, sine) == reference

    assert TAU_MATRIX(Z_MATRIX) == Fraction(1)
    assert TAU_MATRIX(Z_SCALAR) == Fraction(0)
    assert DELTA_SCALAR(Z_MATRIX) == Fraction(0)
    assert DELTA_SCALAR(Z_SCALAR) == Fraction(1)


def test_premise_P4_is_load_bearing_not_decorative() -> None:
    result = audit_premise_necessity()

    assert result["all_reproduce_the_central_distribution"]
    assert result["invariant_members"] == 1
    assert result["only_t_one_half_is_invariant"]

    # An explicit non-invariant rival: correct on the central test, yet not the
    # compiled state. Without P4 the compiler would be a continuum.
    rival = omega_t(Fraction(1, 3))
    assert rival.is_state
    assert rival(Z_MATRIX) == Fraction(1)
    assert rival(Z_SCALAR) == Fraction(0)
    assert _anonymous_state(rival) != _anonymous_state(TAU_MATRIX)


def test_compiler_is_exact_invariant_and_binding_unique() -> None:
    source, _ = load_source()
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)
    result = audit_compiler(source, automorphisms, orbits)

    assert len(automorphisms) == 168
    assert sorted(len(orbit) for orbit in orbits) == [21, 28]

    gates = result["gates"]
    assert gates["A_valid_preparation"]
    assert gates["B_semantic_equivalence"]
    assert gates["C_equivariance_is_trivial_and_satisfied"]
    assert gates["D_distinguishability"]
    assert gates["E_one_input_independent_test"]
    assert gates["E_test_is_derived_not_imported"]
    assert gates["F_no_direct_per_instance_target_leakage"]
    assert gates["learned_parameter_count"] == 0

    rivals = result["rival_search"]
    assert rivals["binding_count"] == 2
    assert rivals["bindings_give_different_states"]
    assert rivals["bindings_give_identical_decision_behavior"]
    assert rivals["state_level_rivals_after_binding"] == 0
    assert rivals["residue_is_not_gauge"]

    # Exact deterministic behavior on every admitted input, under both bindings.
    for binding in BINDINGS:
        for datum in source.inputs:
            values = evaluate(compile_datum(datum, source, binding), binding)
            expected = (
                {"incident": Fraction(1), "nonincident": Fraction(0)}
                if source.semantic_class(datum) == "incident"
                else {"incident": Fraction(0), "nonincident": Fraction(1)}
            )
            assert values == expected
            assert sum(values.values()) == Fraction(1)


def test_the_two_bindings_are_genuinely_inequivalent() -> None:
    source, _ = load_source()
    incident = next(iter(source.incidence or ()))

    first = compile_datum(incident, source, BINDING_MATRIX_FIRST)
    second = compile_datum(incident, source, BINDING_SCALAR_FIRST)

    assert _anonymous_state(first) != _anonymous_state(second)
    assert first(Z_MATRIX) == Fraction(1)
    assert second(Z_SCALAR) == Fraction(1)
    # No target automorphism can repair the difference, because the invariant
    # states live on ideals of different dimension and ambient rank.
    for cosine, sine in RATIONAL_ROTATIONS:
        assert _anonymous_state(first.conjugate_by_rotation(cosine, sine)) != (
            _anonymous_state(second)
        )


def test_ambient_rays_realize_both_states_exactly_but_not_uniquely() -> None:
    bell = ambient_add(u0_vector(), v8_vector())
    assert ambient_inner(bell, bell) == Q7(Fraction(2))
    assert _anonymous_state(ambient_state_of(bell)) == _anonymous_state(TAU_MATRIX)

    # Same-copy control: the cross-copy convention is load-bearing.
    near_miss = ambient_add(u0_vector(), v0_vector())
    assert _anonymous_state(ambient_state_of(near_miss)) != _anonymous_state(
        TAU_MATRIX
    )

    result = audit_ambient_boundary()
    assert result["tau_M_ambient_realizer"]["induces_tau_M"]
    assert not result["same_copy_control"]["induces_tau_M"]
    assert result["all_scalar_rays_give_delta_s"]
    assert not result["canonical_ambient_ray_exists"]
    assert result["fiber_structure"]["delta_s_real_dimension"] == 11


def test_source_group_action_on_the_effect_algebra_is_trivial() -> None:
    source, _ = load_source()
    automorphisms = enumerate_automorphisms(source)
    result = audit_equivariance(automorphisms)

    assert result["source_group_order"] == 168
    assert result["source_group_is_perfect"]
    assert result["target_group_is_solvable"]
    assert result["every_homomorphism_is_trivial"]
    # The claim must stay scoped to what E can observe.
    assert result["scope_limit"]["not_claimed"].startswith("that the ambient carrier")


def test_relation_erasure_uses_the_same_compiler_and_fails_closed() -> None:
    source, _ = load_source()
    control_source = make_relation_erased_control(source)

    assert control_source.points == source.points
    assert control_source.inputs == source.inputs
    assert set(control_source.opaque_rows) == set(source.incidence or ())
    with pytest.raises(MissingDeclaredRelation):
        compile_datum(control_source.inputs[0], control_source, BINDING_MATRIX_FIRST)

    control = audit_control(source)
    assert control["same_compiler_entry_point"] == "compile_datum"
    assert control["compiler_failed_closed"]
    assert control["failed_before_choosing_a_binding"]
    assert control["declared_automorphism_order"]["relation_erased"] == 25_401_600
    assert control["declared_automorphism_order"]["expansion_factor"] == 151_200
    assert control["type_only_action_is_transitive"]


def test_layered_verdict_is_scoped_and_does_not_overclaim() -> None:
    payload = build_payload()
    layers = payload["canonicality_by_layer"]

    assert layers["unordered_central_test"]["code"] == "C3"
    assert layers["state_after_binding"]["code"] == "C3"
    assert layers["joint_labeled_compiler_and_test"]["code"] == "C1"
    assert layers["raw_ambient_ray_compiler"]["code"] == "C0-or-Cempty"
    assert layers["physical_realization"]["code"] == "Cempty"
    assert "C2" in layers["rejected"]
    assert "unconditional_C3" in layers["rejected"]

    residue = payload["residue_ledger"]["exact_residue"]
    assert residue["state_level_family_cardinality"] == 2
    assert residue["specification_bits"] == 1
    assert residue["raw_ray_layer_unimproved"]

    assert not payload["training_handoff"]["justified_now"]
    assert payload["all_mechanical_checks_pass"]
    assert not payload["reproducibility"]["optimizer_or_training_imported"]
    assert not payload["reproducibility"]["interact_imported"]
    assert not payload["reproducibility"]["numpy_imported"]
    assert len(payload["claim_fences"]) >= 5


def test_committed_artifact_is_exactly_the_executable_audit() -> None:
    expected = canonical_audit_bytes()

    assert ARTIFACT_PATH.read_bytes() == expected
    assert json.loads(expected) == build_payload()

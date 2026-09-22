from __future__ import annotations

import json
from fractions import Fraction

import pytest
from audit_input_constitution import (
    ARTIFACT_PATH,
    CONDITIONAL_FIXED_TEST,
    REPRESENTATIVE_A,
    REPRESENTATIVE_B,
    SOURCE_SHA256,
    MissingDeclaredRelation,
    audit_conditional_construction,
    audit_control,
    audit_source,
    build_payload,
    canonical_audit_bytes,
    compile_datum,
    enumerate_automorphisms,
    input_orbits,
    load_source,
    make_relation_erased_control,
    projectively_equivalent,
)


def test_source_fixture_is_byte_frozen_and_all_source_gates_are_computed(
    tmp_path,
) -> None:
    source, document = load_source()
    result = audit_source(source, document)

    assert result["fixture_selection"]["passes"]
    assert all(result["fixture_selection"]["gates"].values())
    assert all(result["incidence_laws"].values())
    assert result["class_counts"] == {"incident": 21, "nonincident": 28}
    assert result["decision_semantics"] == {
        "global_schema_rule_completely_determines_binary_decision": True,
        "per_instance_answer_field_present": False,
        "predictive_claim_permitted": False,
    }
    assert len(SOURCE_SHA256) == 64

    changed = tmp_path / "changed-source.json"
    changed.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="changed after freeze"):
        load_source(changed)


def test_exact_automorphism_search_recovers_two_semantic_orbits() -> None:
    source, _ = load_source()
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)

    assert len(automorphisms) == 168
    assert sorted(len(orbit) for orbit in orbits) == [21, 28]
    assert {
        source.semantic_class(orbit[0]): len(orbit) for orbit in orbits
    } == {"incident": 21, "nonincident": 28}
    assert all(
        source.semantic_class(datum)
        == source.semantic_class(automorphism.apply(datum, source))
        for automorphism in automorphisms
        for datum in source.inputs
    )


def test_conditional_compilers_are_exact_but_not_task_admitted() -> None:
    source, _ = load_source()
    automorphisms = enumerate_automorphisms(source)
    orbits = input_orbits(source, automorphisms)
    result = audit_conditional_construction(source, automorphisms, orbits)

    assert CONDITIONAL_FIXED_TEST.labels == ("incident", "nonincident")
    assert result["status"] == "conditional witness; rejected as a task-level compiler"
    assert result["gates"]["A_valid_preparation_conditional"]
    assert result["gates"]["B_semantic_equivalence_conditional"]
    assert result["gates"]["C_equivariance"] is None
    assert result["gates"]["D_distinguishability_conditional"]
    assert result["gates"]["E_one_input_independent_test_conditional"]
    assert not result["gates"]["E_source_authorized_test"]
    assert result["gates"]["F_no_direct_per_instance_target_leakage"]
    assert result["gates"]["F_schema_rule_already_determines_decision"]
    assert result["gates"]["learned_parameter_count"] == 0
    assert not result["gates"]["task_positive_admitted"]

    rival = result["rival_search"]
    assert rival["projectively_distinct"]
    assert rival["identical_conditional_fixed_test_predictions"]
    assert rival["test_preserving_coordinate_symmetry"]["preserves_both_effects"]
    assert rival["test_preserving_coordinate_symmetry"]["maps_representative_a_to_b"]
    assert not rival["certified_as_full_OT_representation_gauge"]

    incident = next(iter(source.incidence or ()))
    nonincident = next(
        datum for datum in source.inputs if datum not in (source.incidence or ())
    )
    for representatives in (REPRESENTATIVE_A, REPRESENTATIVE_B):
        incident_ray = compile_datum(incident, source, representatives)
        nonincident_ray = compile_datum(nonincident, source, representatives)
        assert incident_ray.squared_norm == Fraction(1)
        assert nonincident_ray.squared_norm == Fraction(1)
        assert not projectively_equivalent(
            incident_ray.coordinates, nonincident_ray.coordinates
        )


def test_relation_erasure_uses_the_same_compiler_and_fails_closed() -> None:
    source, _ = load_source()
    control_source = make_relation_erased_control(source)

    assert control_source.points == source.points
    assert control_source.lines == source.lines
    assert control_source.inputs == source.inputs
    assert set(control_source.opaque_rows) == set(source.incidence or ())
    with pytest.raises(MissingDeclaredRelation, match="without a declared incident"):
        compile_datum(control_source.inputs[0], control_source, REPRESENTATIVE_A)

    control = audit_control(source)
    assert control["same_compiler_entry_point"] == "compile_datum"
    assert all(control["preserved"].values())
    assert control["declared_automorphism_order"] == {
        "structured": 168,
        "relation_erased": 25_401_600,
        "derivation": "7! independent point permutations times 7! independent line permutations",
        "expansion_factor": 151_200,
    }
    assert control["type_only_orbit_witness"]["reachable_pairs"] == 49
    assert control["type_only_orbit_witness"]["equals_all_admitted_inputs"]
    assert not control["incidence_decision_defined"]
    assert control["compiler_failed_closed"]


def test_task_verdict_is_fail_closed_and_conditional_residue_is_scoped() -> None:
    payload = build_payload()

    verdict = payload["task_canonicality_verdict"]
    assert verdict["code"] == "Cempty"
    assert verdict["symbol"] == "C∅"
    assert verdict["strongest_justified_task_status"]
    assert not payload["task_positive_admission"]

    ledger = payload["declared_derived_constitutive_ledger"]
    assert ledger["full_task_residue"]["status"] == (
        "not typed sharply enough by current authority to quantify"
    )
    residue = ledger["conditional_after_imported_test_and_slot_binding"]
    assert residue["ambient_real_family"] == {
        "family": "RP^1 x RP^13",
        "real_dimension": 14,
        "cardinality": "continuum",
    }
    assert residue["executable_exact_rational_subset"] == {
        "family": "P(Q^2) x P(Q^14)",
        "cardinality": "countably infinite",
    }
    assert residue["per_instance_correct_fixed_test_dimension"] == 385
    assert residue["schema_invariant_correct_fixed_test_dimension"] == 14
    assert residue["conditional_dimension_removed"] == 371

    pipeline = payload["fixed_test_pipeline"]
    assert pipeline["type_correct_after_constitutive_test_supply"]
    assert not pipeline["source_authorized"]
    assert not pipeline["task_positive_admitted"]
    assert not payload["training_handoff"]["justified_now"]
    assert payload["all_mechanical_checks_pass"]


def test_committed_artifact_is_exactly_the_executable_audit() -> None:
    expected = canonical_audit_bytes()

    assert ARTIFACT_PATH.read_bytes() == expected
    assert json.loads(expected) == build_payload()

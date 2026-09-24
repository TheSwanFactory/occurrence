"""Adversarial tests for the Issue 017.24a quotient-refinement change.

Written at the freeze, before the experiment ran. The structural tests are
unconditional; the ones that read the result artifacts skip until those exist,
so the freeze commit is green without pretending to have results.

The two claims this turn lives or dies by are checked independently of the
module's own audit: that the Fano derived partition really is the declared
relation (so a relation-indexed learner must be indistinguishable from the
compiled one), and that the circulant derived partition really is strictly
finer (so it cannot be). A third test cross-validates the new generalized exact
reference against the 017.24 reference it replaces.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "quotient_refinement.py"
SPEC = importlib.util.spec_from_file_location("quotient_refinement_01724a", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
refine = importlib.util.module_from_spec(SPEC)
sys.modules["quotient_refinement_01724a"] = refine
SPEC.loader.exec_module(refine)

shared = refine.shared
ONE = Fraction(1)


@pytest.fixture(scope="module")
def fano():
    return refine.fano_fixture()


@pytest.fixture(scope="module")
def circulant():
    return refine.circulant_fixture()


def _results():
    if not refine.RESULTS_PATH.exists():
        pytest.skip("results artifact not written yet; run --write first")
    return json.loads(refine.RESULTS_PATH.read_text())


# --- pins -------------------------------------------------------------------


def test_every_input_is_byte_pinned():
    schema, document = refine.load_refinement()
    assert document["schema"] == refine.REFINEMENT_SCHEMA
    assert refine.load_preregistration()["schema"] == (
        refine.PREREGISTRATION_SCHEMA
    )
    assert len(schema.inputs) == 49


def test_a_changed_schema_is_refused(tmp_path):
    corrupted = tmp_path / "refinement_schema.json"
    corrupted.write_bytes(refine.REFINEMENT_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        refine.load_refinement(corrupted)


def test_a_changed_preregistration_is_refused(tmp_path):
    corrupted = tmp_path / "preregistration.json"
    corrupted.write_bytes(refine.PREREGISTRATION_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        refine.load_preregistration(corrupted)


def test_the_shared_017_24_implementation_is_pinned():
    assert refine.SHARED_MODULE_SHA256 == (
        "821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79"
    )
    assert shared.SUCCESSOR_SCHEMA == "gpt-01724-fano-successor-type-schema/v1"
    assert hasattr(shared, "fit_typed_tdm")
    assert hasattr(shared, "build_split")
    assert hasattr(shared, "generate_observations")


# --- the fixtures are matched on every cardinality -------------------------


def test_the_two_fixtures_are_cardinality_matched(fano, circulant):
    for fixture in (fano, circulant):
        assert len(fixture.schema.points) == 7
        assert len(fixture.schema.lines) == 7
        assert len(fixture.schema.incidence) == 21
        assert len(fixture.structure.inputs) == 49


def test_the_circulant_is_not_the_fano_plane(fano, circulant):
    assert circulant.schema.incidence != fano.schema.incidence
    assert circulant.structure.group_order == 14
    assert fano.structure.group_order == 168


def test_the_circulant_relation_is_the_declared_difference_set(circulant):
    for point, line in circulant.structure.inputs:
        difference = (int(line[1:]) - int(point[1:])) % 7
        assert ((point, line) in circulant.schema.incidence) == (
            difference in {0, 1, 2}
        )


# --- the whole point: refinement versus coincidence ------------------------


def test_on_fano_the_derived_partition_is_the_declared_relation(fano):
    derived = fano.structure.class_of_index
    relation = refine.relation_partition(fano)
    grouping = {}
    for left, right in zip(derived, relation, strict=True):
        grouping.setdefault(left, set()).add(right)
    assert all(len(values) == 1 for values in grouping.values())
    assert len(set(derived)) == len(set(relation)) == 2
    assert sorted(derived.count(c) for c in set(derived)) == [21, 28]


def test_on_the_circulant_the_derived_partition_is_strictly_finer(circulant):
    derived = circulant.structure.class_of_index
    relation = refine.relation_partition(circulant)
    assert len(set(derived)) == 4
    assert len(set(relation)) == 2
    grouping = {}
    for left, right in zip(derived, relation, strict=True):
        grouping.setdefault(left, set()).add(right)
    assert all(len(values) == 1 for values in grouping.values()), (
        "every derived class must lie wholly inside or wholly outside the "
        "relation"
    )
    inside = sorted(
        len(members)
        for members in circulant.structure.classes
        if members[0] in circulant.schema.incidence
    )
    outside = sorted(
        len(members)
        for members in circulant.structure.classes
        if members[0] not in circulant.schema.incidence
    )
    assert inside == [7, 14] and outside == [14, 14]


def test_the_relation_partition_is_identical_on_both_fixtures(fano, circulant):
    for fixture in (fano, circulant):
        relation = refine.relation_partition(fixture)
        assert sorted(relation.count(c) for c in set(relation)) == [21, 28]


# --- the declared symmetry ladder ------------------------------------------


def test_the_circulant_declares_its_generators_and_they_are_automorphisms(
    circulant,
):
    block = circulant.document["declared_symmetry_ladder"]
    assert set(block["generators"]) == {"tau", "sigma"}
    for name, value in block["generators"].items():
        element = shared.Automorphism(
            point_images=tuple(value["point_images"]),
            line_images=tuple(value["line_images"]),
        )
        assert refine._preserves_relation(circulant.schema, element), name


def test_every_ladder_rung_matches_its_declared_expectation(circulant):
    declared = {
        rung["name"]: (
            rung["expected_closed_order"],
            rung["expected_free_parameters"],
        )
        for rung in circulant.document["declared_symmetry_ladder"]["rungs"]
    }
    for name, order, partition in circulant.ladder:
        assert declared[name] == (order, len(set(partition)))


def test_the_ladders_are_monotone_in_free_parameters(fano, circulant):
    for fixture in (fano, circulant):
        counts = [len(set(partition)) for _, _, partition in fixture.ladder]
        assert counts == sorted(counts, reverse=True)
        assert counts[0] == 49
        assert counts[-1] == fixture.class_count


def test_the_top_rung_coincides_with_the_full_automorphism_quotient(
    fano, circulant
):
    for fixture in (fano, circulant):
        _, order, partition = fixture.ladder[-1]
        assert order == fixture.structure.group_order
        assert len(set(partition)) == fixture.class_count


def test_the_fano_ladder_does_not_depend_on_which_subgroup_was_chosen(fano):
    ordered = sorted(
        fano.structure.automorphisms,
        key=lambda g: (g.point_images, g.line_images),
    )
    seen = 0
    shapes = set()
    for candidate in ordered:
        group = refine.close_subgroup(fano.schema, [candidate])
        if len(group) != 7:
            continue
        partition = shared.invariance_partition_from_group(
            fano.structure.inputs, fano.schema, group
        )
        shapes.add(
            tuple(sorted(partition.count(c) for c in set(partition)))
        )
        seen += 1
        if seen == 8:
            break
    assert seen >= 2, "there must be several order-7 subgroups to compare"
    assert len(shapes) == 1, "conjugate subgroups must give the same quotient"


def test_a_non_automorphism_is_rejected_as_a_generator(circulant):
    bogus = shared.Automorphism(
        point_images=("P1", "P0") + circulant.schema.points[2:],
        line_images=circulant.schema.lines,
    )
    assert not refine._preserves_relation(circulant.schema, bogus)


# --- ground truth -----------------------------------------------------------


def test_theta_star_is_evenly_spaced_interior_and_distinct():
    for count in (2, 3, 4, 7):
        theta = refine.evenly_spaced_theta(count)
        assert len(theta) == count
        assert len(set(theta)) == count
        assert all(0 < value < 1 for value in theta)
        assert list(theta) == sorted(theta)
        assert sum(theta) == Fraction(count, 2)
    assert refine.evenly_spaced_theta(2) == (Fraction(1, 4), Fraction(3, 4))
    assert refine.evenly_spaced_theta(4)[0] == Fraction(1, 8)


# --- the generalized exact reference ---------------------------------------


def test_the_new_reference_reproduces_the_017_24_reference(fano):
    """Cross-validate the generalized sum against the code it replaces."""
    weights = (Fraction(21, 49), Fraction(28, 49))
    theta = refine.evenly_spaced_theta(2)
    sizes = (4, 32, 256, 1024)
    legacy = shared.analytic_reference(
        sizes, theta, ONE, ONE, weights, weights
    )
    everything = tuple(range(49))
    fresh = refine.exact_expected_excess(
        fano.structure.class_of_index,
        2,
        [float(v) for v in theta],
        fano.structure.class_of_index,
        everything,
        everything,
        sizes,
        ONE,
        ONE,
    )
    for left, right in zip(legacy["sweep"], fresh["sweep"], strict=True):
        assert left["n"] == right["n"]
        assert abs(
            float(left["exact_expected_excess_nll"])
            - float(right["exact_expected_excess_nll"])
        ) < 1e-9


def test_the_reference_tends_to_dof_over_2n_when_the_floor_is_zero(circulant):
    everything = tuple(range(49))
    theta = refine.evenly_spaced_theta(4)
    report = refine.exact_expected_excess(
        circulant.structure.class_of_index,
        4,
        [float(v) for v in theta],
        circulant.structure.class_of_index,
        everything,
        everything,
        (512, 1024),
        ONE,
        ONE,
    )
    assert report["floor_is_zero"]
    for row in report["sweep"]:
        predicted = 4.0 / (2.0 * int(row["n"]))
        assert abs(
            float(row["exact_expected_excess_nll"]) - predicted
        ) < 0.2 * predicted


def test_the_relation_indexed_floor_separates_the_two_fixtures(
    fano, circulant
):
    everything = tuple(range(49))
    results = {}
    for fixture in (fano, circulant):
        theta = refine.evenly_spaced_theta(fixture.class_count)
        results[fixture.identifier] = refine.exact_expected_excess(
            refine.relation_partition(fixture),
            2,
            [float(v) for v in theta],
            fixture.structure.class_of_index,
            everything,
            everything,
            (1024,),
            ONE,
            ONE,
        )
    assert results["fano"]["floor_is_zero"]
    assert not results["circulant"]["floor_is_zero"]
    assert float(results["circulant"]["asymptotic_floor"]) > 0.04


def test_the_compiled_partition_has_no_floor_on_either_fixture(
    fano, circulant
):
    everything = tuple(range(49))
    for fixture in (fano, circulant):
        theta = refine.evenly_spaced_theta(fixture.class_count)
        report = refine.exact_expected_excess(
            fixture.structure.class_of_index,
            fixture.class_count,
            [float(v) for v in theta],
            fixture.structure.class_of_index,
            everything,
            everything,
            (64,),
            ONE,
            ONE,
        )
        assert report["floor_is_zero"]


def test_a_cell_with_no_training_data_falls_back_to_the_prior(circulant):
    """Regime B on singleton cells: unseen identities must not be extrapolated."""
    everything = tuple(range(49))
    seen = tuple(index for index in everything if index % 5 != 0)
    unseen = tuple(index for index in everything if index % 5 == 0)
    theta = refine.evenly_spaced_theta(4)
    report = refine.exact_expected_excess(
        everything,
        49,
        [float(v) for v in theta],
        circulant.structure.class_of_index,
        seen,
        unseen,
        (1024,),
        ONE,
        ONE,
    )
    assert report["cells_reached_by_training"] == len(seen)
    assert not report["floor_is_zero"]


# --- the read-the-relation learner ----------------------------------------


def test_the_relation_designs_contain_the_relation_bit(circulant):
    designs = refine.build_designs(circulant)
    relation = refine.relation_partition(circulant)
    target = np.array([1.0 if cell == 0 else 0.0 for cell in relation])
    plain = designs["mlp"]
    with_bit = designs["relation_mlp"]
    assert with_bit.shape[1] == plain.shape[1] + 1
    assert np.array_equal(with_bit[:, -1], target)
    assert not any(
        np.array_equal(plain[:, column], target)
        for column in range(plain.shape[1])
    )


def test_the_relation_bit_alone_cannot_express_the_circulant_quotient(
    circulant,
):
    """The strongest read-the-relation learner is still provably insufficient."""
    designs = refine.build_designs(circulant)
    design = designs["relation_linear"]
    derived = circulant.structure.class_of_index
    for target_cell in range(4):
        target = np.array(
            [1.0 if cell == target_cell else 0.0 for cell in derived]
        )
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = float(np.max(np.abs(design @ solution - target)))
        if residual > 1e-6:
            return
    pytest.fail("the relation-augmented linear design must miss some class")


def test_U_relation_is_among_the_declared_variants():
    names = [name for name, _, _, _ in refine.U_VARIANTS]
    assert "U_relation" in names
    assert set(refine.LEARNED_CONDITIONS) == set(names)
    inherited = {
        "U_fixed",
        "U_earlystop",
        "U_selected",
        "U_quadratic",
        "U_ladder",
    }
    assert inherited <= set(names), "the 017.24 variants must all be carried over"


# --- the coarse-compiler control ------------------------------------------


def test_a_legitimate_but_too_coarse_quotient_still_pays_a_floor(
    fano, circulant
):
    payload = refine.prediction_payload((fano, circulant), (64,))
    controls = payload["coarse_compiler_control"]
    assert set(controls) == {"fano", "circulant"}
    for block in controls.values():
        assert float(block["coarse_compiler_floor"]) > 0.0
        assert abs(float(block["correct_rung_floor"])) < 1e-12


def test_the_inherited_controls_still_fail_closed(circulant):
    erased = shared.build_relation_erased(circulant.schema)
    with pytest.raises(shared.MissingDeclaredRelation):
        shared.compile_classes(erased)
    with pytest.raises(shared.MissingDeclaredRelation):
        shared.build_feature_matrix(erased)


def test_no_scramble_of_the_circulant_reproduces_its_quotient(circulant):
    for scrambled in shared.build_scrambled_schemas(circulant.schema, 12):
        assert scrambled.incidence != circulant.schema.incidence
        derived = shared.compile_classes(scrambled)
        assert sorted(derived.cardinalities) != [7, 14, 14, 14]


# --- splits -----------------------------------------------------------------


def test_the_stratified_holdout_covers_every_derived_class(circulant):
    members = shared._class_members(circulant.structure)
    for replicate in range(15):
        split = shared.build_split(
            "B",
            random.Random(900 + replicate),
            members,
            circulant.holdout_per_class,
        )
        assert len(split.held_out) == sum(circulant.holdout_per_class)
        assert not set(split.train_pool) & set(split.held_out)
        held = [
            circulant.structure.class_of_index[index]
            for index in split.held_out
        ]
        for cell, count in enumerate(circulant.holdout_per_class):
            assert held.count(cell) == count
        seen = {
            circulant.structure.class_of_index[index]
            for index in split.train_pool
        }
        assert seen == {0, 1, 2, 3}


# --- the frozen predictions ------------------------------------------------


def test_the_preregistration_freezes_the_predictions():
    document = refine.load_preregistration()
    declared = document["declared_predictions"]
    regime_a = declared["regime_A_exact_expected_excess_nll"]
    assert regime_a["fano"]["R"]["floor"] == 0.0
    assert regime_a["circulant"]["R"]["floor"] > 0.04
    assert regime_a["fano"]["T"]["n_at_0.02"] == 64
    assert regime_a["circulant"]["T"]["n_at_0.02"] == 128
    assert regime_a["circulant"]["R"]["n_at_0.02"] is None
    assert "crossing_prediction" in declared
    assert "falsification" in declared


def test_the_frozen_predictions_match_the_executable(fano, circulant):
    document = refine.load_preregistration()
    declared = document["declared_predictions"][
        "regime_A_exact_expected_excess_nll"
    ]
    everything = tuple(range(49))
    for fixture in (fano, circulant):
        theta = refine.evenly_spaced_theta(fixture.class_count)
        truths = [float(v) for v in theta]
        for name, partition, cells in (
            ("T", fixture.structure.class_of_index, fixture.class_count),
            ("R", refine.relation_partition(fixture), 2),
        ):
            report = refine.exact_expected_excess(
                partition,
                cells,
                truths,
                fixture.structure.class_of_index,
                everything,
                everything,
                refine.SAMPLE_SIZES,
                ONE,
                ONE,
            )
            series = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in report["sweep"]
            }
            block = declared[fixture.identifier][name]
            for key in ("4", "64", "1024"):
                assert abs(series[int(key)] - float(block[key])) < 5e-5, (
                    fixture.identifier,
                    name,
                    key,
                )
            assert abs(
                float(report["asymptotic_floor"]) - float(block["floor"])
            ) < 5e-5


# --- artifacts, once phase two has produced them --------------------------


def test_the_structural_artifacts_match_the_executable(fano, circulant):
    if not refine.CERTIFICATE_PATH.exists():
        pytest.skip("certificate not written yet")
    certificate = refine.refinement_certificate((fano, circulant))
    assert refine.CERTIFICATE_PATH.read_bytes() == refine._canonical(
        certificate
    )
    reference = refine.prediction_payload((fano, circulant), refine.SAMPLE_SIZES)
    assert refine.REFERENCE_PATH.read_bytes() == refine._canonical(reference)


def test_the_results_pass_every_mechanical_check():
    results = _results()
    assert results["all_mechanical_checks_pass"]
    assert all(results["mechanical_checks"].values())
    assert results["mechanical_check_count"] >= 20


def test_the_verdict_is_one_of_the_preregistered_outcomes():
    results = _results()
    assert results["J_verdict"]["verdict"] in (
        "SEPARATION ESTABLISHED",
        "PARTIAL SEPARATION",
        "NO SEPARATION",
        "IMPLEMENTATION DEFECT",
    )


def test_the_measured_curves_match_the_frozen_predictions():
    results = _results()
    validation = results["prediction_validation"]
    assert validation["all_within_four_sigma"], validation["worst_sigma"]
    assert validation["cells_checked"] >= 18


def test_the_measured_separation_reproduces_the_structural_fact():
    results = _results()
    separation = results["J_verdict"]["separation"]
    assert results["J_verdict"]["fano_max_absolute_R_minus_T"] == 0.0
    assert results["J_verdict"]["circulant_max_R_minus_T"] > 0.0
    assert separation["fano"]["A"]["R_floor_is_zero"] is True
    assert separation["circulant"]["A"]["R_floor_is_zero"] is False

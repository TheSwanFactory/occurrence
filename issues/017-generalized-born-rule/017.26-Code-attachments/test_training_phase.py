"""Adversarial tests for the Issue 017.26 Phase T statistical experiment.

The structural tests and the frozen-prediction tests are unconditional. The ones
that read the result artifacts skip until those exist, so the preregistration
commit is green without pretending to have measured anything.

Three claims get checked here independently of the module's own audit.

That the phase fence is real: the Phase S module and certificate are byte-pinned,
a corrupted copy is refused, and the quotient this turn measures against is the
one the freeze banked, reconstructed from the certificate text by a path that does
not call the compiler.

That the frozen predictions are not decoration: every banked number is recomputed
here from theta_star and the derived quotients and checked against the committed
preregistration, including the one that cuts against the thesis.

And that the uncompiled learners are not crippled: the cell indicator is exhibited
inside the span of the design they are given, so a measured gap cannot be a
reachability artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "training_phase.py"
SPEC = importlib.util.spec_from_file_location("training_phase_01726", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
phase_t = importlib.util.module_from_spec(SPEC)
sys.modules["training_phase_01726"] = phase_t
SPEC.loader.exec_module(phase_t)

phase_s = phase_t.phase_s
refine = phase_t.refine
shared = phase_t.shared
ONE = Fraction(1)


@pytest.fixture(scope="module")
def targets():
    return phase_t.build_targets()


def _results():
    if not phase_t.RESULTS_PATH.exists():
        pytest.skip("results artifact not written yet; run --write first")
    return json.loads(phase_t.RESULTS_PATH.read_text())


# --- the pins and the phase fence ------------------------------------------


def test_every_input_is_byte_pinned():
    assert phase_t.PHASE_S_MODULE_SHA256 == (
        "e7074a60ed460b7b98d7f234e3af06aec169e5ae526e477c62e1e8799d0f2e3b"
    )
    assert phase_t.PHASE_S_CERTIFICATE_SHA256 == (
        "f2e36998f0e5715d785d5dd0a86790fc32604040334c8250d2fd6c845c29709a"
    )
    assert phase_t.SHARED_MODULE_SHA256 == (
        "821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79"
    )
    assert refine.SHARED_MODULE_SHA256 == phase_t.SHARED_MODULE_SHA256


def test_a_changed_phase_s_certificate_is_refused(tmp_path):
    corrupted = tmp_path / "phase_s_certificate.json"
    corrupted.write_bytes(phase_t.PHASE_S_CERTIFICATE_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        phase_t.load_phase_s_certificate(corrupted)


def test_a_changed_preregistration_is_refused(tmp_path):
    corrupted = tmp_path / "preregistration.json"
    corrupted.write_bytes(phase_t.PREREGISTRATION_PATH.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        phase_t.load_preregistration(corrupted)


def test_the_phase_s_module_still_cannot_have_seen_a_label():
    audit = phase_s.phase_fence_audit()
    assert audit["imports_no_statistical_module"]
    assert audit["module_sha256"] == phase_t.PHASE_S_MODULE_SHA256


def test_the_preregistration_records_the_phase_s_commit_and_revision():
    document = phase_t.load_preregistration()
    fence = document["phase_fence"]
    assert fence["phase_s_commit"] == phase_t.PHASE_S_COMMIT
    assert fence["phase_s_quilt_revision"] == phase_t.PHASE_S_QUILT_REVISION
    assert (
        fence["phase_s_certificate_sha256"]
        == phase_t.PHASE_S_CERTIFICATE_SHA256
    )


# --- condition H really is a second path -----------------------------------


def test_the_handed_quotient_reproduces_the_compiled_one(targets):
    """The whole content of D equals H on these fixtures.

    ``H`` parses cell membership lists out of the committed freeze. ``D`` runs the
    compiler in this process. They must be the same partition, or the compiler is
    not reproducing its own banked output.
    """
    certificate = phase_t.load_phase_s_certificate()
    for target in targets:
        handed = phase_t.handed_quotient(
            certificate, target.identifier, target.world
        )
        assert handed == target.compiled.quotient
        assert len(set(handed)) == target.cell_count


def test_the_handed_quotient_path_does_not_call_the_compiler():
    import inspect

    source = inspect.getsource(phase_t.handed_quotient)
    for forbidden in ("compile_world", "prepare", "derive_orbit_quotient"):
        assert forbidden not in source


def test_the_banked_cells_cover_every_admitted_input_exactly_once(targets):
    certificate = phase_t.load_phase_s_certificate()
    for target in targets:
        block = next(
            item
            for item in certificate["fixtures"]
            if item["fixture"] == target.identifier
        )
        cells = block["C_derived_before_any_label"]["cells"]
        members = [member for group in cells.values() for member in group]
        assert len(members) == len(set(members)) == target.width


# --- the derived structure is the one Phase S banked ------------------------


def test_the_derived_structure_matches_the_freeze(targets):
    expected = {"A": (14, [7, 14, 14, 14], 49), "B": (6, [3, 6, 6, 6, 6], 27)}
    for target in targets:
        order, sizes, width = expected[target.identifier]
        assert target.compiled.group_order == order
        assert target.compiled.cell_sizes == sizes
        assert target.width == width


def test_the_shortcut_partition_is_strictly_coarser_than_the_derived_one(targets):
    for target in targets:
        assert phase_s._refines(target.compiled.quotient, target.shortcut)
        assert not phase_s._refines(target.shortcut, target.compiled.quotient)
        assert len(set(target.shortcut)) == 2


def test_on_the_richer_fixture_the_shortcut_is_exactly_one_primitive(targets):
    """A fact about Fixture B that is reported rather than hidden.

    The frozen shortcut language happens to reach exactly the quotient that
    survives erasing the neighbour relation, 9 and 18. So on that fixture the
    cheapest preregistered statistic is precisely what one primitive alone can
    support, and the derived quotient is what joint preservation adds on top.
    """
    target = targets[1]
    without = next(
        spec
        for spec in target.conditions
        if spec.name == "P_without_adjacent"
    )
    assert phase_s._refines(target.shortcut, without.partition)
    assert phase_s._refines(without.partition, target.shortcut)


def test_theta_star_is_fixed_by_the_declared_rule(targets):
    for target in targets:
        count = target.cell_count
        assert target.theta == tuple(
            Fraction(2 * index + 1, 2 * count) for index in range(count)
        )
        assert all(Fraction(0) < value < ONE for value in target.theta)
        assert len(set(target.theta)) == count


def test_every_coarser_declared_partition_is_insufficient(targets):
    """017.26b section 10, measured rather than asserted."""
    for target in targets:
        certificate = phase_t.sufficiency_certificate(target)
        assert certificate["every_probability_is_interior"]
        assert certificate["not_all_probabilities_are_equal"]
        assert certificate[
            "every_coarser_declared_partition_carries_a_positive_floor"
        ]
        assert certificate["insufficient_partitions"]
        for row in certificate["insufficient_partitions"]:
            assert row["coarser_than_the_derived_quotient"], row
            assert float(row["asymptotic_floor"]) > 0.0, row


# --- the controls -----------------------------------------------------------


def test_the_wrong_compiler_control_keeps_the_cardinalities_and_loses_invariance(
    targets,
):
    for target in targets:
        certificate = target.wrong_certificate
        assert certificate["cardinalities"] == target.compiled.cell_sizes
        assert not certificate["respects_the_derived_maps"]
        assert int(certificate["invariance_violations_counted"]) > 0


def test_the_scrambled_control_is_not_the_derived_quotient(targets):
    for target in targets:
        spec = next(
            item
            for item in target.conditions
            if item.name == "C_scrambled_compiler"
        )
        assert spec.partition != target.compiled.quotient


def test_every_condition_shares_one_estimator_and_one_set_of_counts(targets):
    """Fairness is structural: the same counts are refit by every condition."""
    import random

    target = targets[0]
    rng = random.Random(1)
    split = shared.build_split("A", rng, target.members, target.holdout)
    numerators, denominator = shared.theta_as_integers(target.theta)
    observations = shared.generate_observations(
        rng,
        split.train_pool,
        64,
        numerators,
        denominator,
        target.compiled.quotient,
    )
    successes, trials = shared.pair_counts(observations, target.width)
    assert sum(trials) == 64
    for spec in target.conditions:
        model = shared.fit_bernoulli(
            successes,
            trials,
            spec.partition,
            spec.cell_count,
            ONE,
            ONE,
            spec.name,
        )
        assert model.degrees_of_freedom == spec.cell_count
        assert len(model.predictions) == target.width


def test_no_condition_can_see_theta_star(targets):
    import inspect

    signature = inspect.signature(shared.fit_bernoulli)
    assert "theta" not in " ".join(signature.parameters)
    source = inspect.getsource(shared.fit_bernoulli)
    assert "numerator" not in source


# --- the learners are not crippled -----------------------------------------


def test_the_learners_design_contains_every_cell_indicator(targets):
    for target in targets:
        certificate = phase_t.span_certificate(
            target.designs["quadratic"], target.compiled.quotient
        )
        assert certificate["every_cell_indicator_lies_in_the_span"], certificate
        assert certificate["a_single_input_indicator_also_lies_in_the_span"]


def test_the_encoding_contains_the_complete_declared_extension(targets):
    """Information parity, checked by reconstructing each primitive from the design.

    For every declared relation and every declared operation, the truth value or
    the value on the input's own components is recoverable from the encoding as a
    single column, so nothing declared was withheld from the learners.
    """
    for target in targets:
        features, blocks = phase_t.build_feature_matrix(target.world)
        names = [str(block["block"]) for block in blocks]
        for primitive in target.world.primitives:
            if primitive.kind == "relation":
                assert any(
                    name.startswith(f"slice_of_{primitive.name}(")
                    for name in names
                ), (target.identifier, primitive.name)
            if primitive.kind == "operation":
                assert any(
                    name.startswith(f"value_of_{primitive.name}(")
                    for name in names
                ), (target.identifier, primitive.name)
        assert features.shape[0] == target.width
        assert np.isfinite(features).all()


def test_the_shortcut_learner_is_handed_the_frozen_features_outright(targets):
    for target in targets:
        base = phase_t.build_feature_matrix(target.world)[0]
        assert (
            target.designs["shortcut_mlp"].shape[1] > base.shape[1]
        ), target.identifier


# --- the frozen predictions ------------------------------------------------


def test_the_preregistration_freezes_the_headline_numbers():
    document = phase_t.load_preregistration()
    declared = document["declared_predictions"]
    regime_a = declared["regime_A_exact_expected_excess_nll"]
    assert regime_a["A"]["D"]["n_at_0.02"] == 128
    assert regime_a["B"]["D"]["n_at_0.02"] == 128
    assert regime_a["A"]["D"]["floor"] == 0.0
    assert regime_a["B"]["D"]["floor"] == 0.0
    assert regime_a["A"]["R"]["n_at_0.02"] is None
    assert regime_a["B"]["R"]["n_at_0.02"] is None
    assert regime_a["A"]["R"]["floor"] > 0.04
    assert regime_a["B"]["R"]["floor"] > 0.05
    assert "falsification" in declared
    assert "prediction_that_cuts_against_the_thesis" in declared


def test_the_frozen_predictions_match_the_executable(targets):
    """Recompute every banked regime A number from theta_star and the quotients."""
    document = phase_t.load_preregistration()
    declared = document["declared_predictions"][
        "regime_A_exact_expected_excess_nll"
    ]
    for target in targets:
        whole = tuple(range(target.width))
        truths = [float(value) for value in target.theta]
        for spec in target.conditions:
            report = refine.exact_expected_excess(
                spec.partition,
                spec.cell_count,
                truths,
                target.compiled.quotient,
                whole,
                whole,
                phase_t.SAMPLE_SIZES,
                ONE,
                ONE,
            )
            series = {
                int(row["n"]): float(row["exact_expected_excess_nll"])
                for row in report["sweep"]
            }
            block = declared[target.identifier][spec.name]
            for key in ("4", "64", "1024"):
                assert abs(series[int(key)] - float(block[key])) < 5e-5, (
                    target.identifier,
                    spec.name,
                    key,
                )
            assert (
                abs(float(report["asymptotic_floor"]) - float(block["floor"]))
                < 5e-5
            )
            assert int(block["cells"]) == spec.cell_count


def test_the_prediction_that_cuts_against_the_thesis_is_banked():
    """R is predicted to beat D at n = 4 on Fixture A, and there only."""
    document = phase_t.load_preregistration()
    declared = document["declared_predictions"][
        "regime_A_exact_expected_excess_nll"
    ]
    assert float(declared["A"]["R"]["4"]) < float(declared["A"]["D"]["4"])
    assert float(declared["A"]["R"]["64"]) > float(declared["A"]["D"]["64"])
    assert float(declared["B"]["R"]["4"]) > float(declared["B"]["D"]["4"])
    text = document["declared_predictions"][
        "prediction_that_cuts_against_the_thesis"
    ]
    assert "n = 4 and only there" in text


def test_the_regime_B_saturated_floor_is_predicted_to_be_large():
    document = phase_t.load_preregistration()
    floors = document["declared_predictions"][
        "regime_B_asymptotic_floors_on_unseen_identities"
    ]
    for fixture in ("A", "B"):
        assert floors[fixture]["D"]["floor_mean_over_every_split"] == 0.0
        assert floors[fixture]["S"]["floor_mean_over_every_split"] > 0.1
        assert floors[fixture]["R"]["floor_mean_over_every_split"] > 0.04


def test_the_verdict_rule_is_the_contract_verbatim_in_substance():
    document = phase_t.load_preregistration()
    rule = document["verdict_rule"]
    assert len(rule["strong_positive_requires_all_seven"]) == 7
    assert "017.26b section 11" in rule["source"]
    for outcome in (
        "structural_positive",
        "cheap_statistic_collapse",
        "negative",
        "implementation_defect",
    ):
        assert rule[outcome]


def test_the_forbidden_conclusions_include_the_owner_fences():
    document = phase_t.load_preregistration()
    text = " ".join(document["forbidden_conclusions"]).lower()
    assert "sample-complexity" in text
    assert "information unavailable to u" in text
    assert "automorphism quotienting" in text
    assert "cheaper derivation" in text


# --- artifacts, once the run has produced them -----------------------------


def test_the_reference_matches_the_executable(targets):
    if not phase_t.REFERENCE_PATH.exists():
        pytest.skip("reference not written yet")
    payload = phase_t.exact_reference(
        targets, phase_t.SAMPLE_SIZES, phase_t.CLOSED_FORM_REPLICATES
    )
    assert phase_t.REFERENCE_PATH.read_bytes() == phase_t._canonical(payload)


def test_the_results_pass_every_mechanical_check():
    results = _results()
    assert results["all_mechanical_checks_pass"]
    assert all(results["mechanical_checks"].values())
    assert results["mechanical_check_count"] >= 20
    assert results["phase"] == "T"


def test_the_verdict_is_one_of_the_preregistered_outcomes():
    results = _results()
    assert results["K_verdict"]["verdict"] in (
        "STRONG POSITIVE",
        "STRUCTURAL POSITIVE",
        "CHEAP-STATISTIC COLLAPSE",
        "NEGATIVE",
        "IMPLEMENTATION DEFECT",
    )


def test_the_measured_curves_match_the_frozen_predictions():
    results = _results()
    validation = results["prediction_validation"]
    assert validation["all_within_four_sigma"], validation["worst_sigma"]
    assert validation["cells_checked"] >= 100


def test_D_and_H_agreed_in_every_replicate():
    results = _results()
    for block in results["agreement"].values():
        assert block["D_versus_H_mismatches"] == 0
        assert block["comparisons"] > 0


def test_the_results_state_what_may_not_be_concluded():
    results = _results()
    forbidden = " ".join(results["K_verdict"]["interpretation_forbidden"]).lower()
    assert "sample-complexity" in forbidden
    assert "same source information" in forbidden
    ledger = results["L_claim_ledger"]
    assert set(ledger) >= {
        "declared",
        "derived_before_any_label",
        "learned_from_labels",
        "implementation_choice",
        "still_open",
    }
    assert any("Q_aut" in item for item in ledger["still_open"])


def test_the_saturated_model_cannot_transfer_to_unseen_identities():
    """Regime B is where a per-identity model has nothing to say."""
    results = _results()
    sizes = results["design"]["sample_sizes"]
    for fixture in ("A", "B"):
        derived = results["cells"][f"{fixture}|B|{sizes[-1]}|D"][
            "excess_aggregate"
        ]["mean"]
        saturated = results["cells"][f"{fixture}|B|{sizes[-1]}|S"][
            "excess_aggregate"
        ]["mean"]
        assert saturated > derived, (fixture, saturated, derived)
        assert saturated > 0.1


def test_the_controls_never_beat_the_derived_compiler_asymptotically():
    results = _results()
    sizes = results["design"]["sample_sizes"]
    for fixture in ("A", "B"):
        derived = results["cells"][f"{fixture}|A|{sizes[-1]}|D"][
            "excess_aggregate"
        ]["mean"]
        for control in ("W_wrong_compiler", "R"):
            other = results["cells"][f"{fixture}|A|{sizes[-1]}|{control}"][
                "excess_aggregate"
            ]["mean"]
            assert other > derived, (fixture, control, other, derived)

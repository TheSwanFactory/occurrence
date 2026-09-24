"""Adversarial tests for the Issue 017.24 matched training-economy experiment.

The tests are written to make a vacuous pass visible. Where the experiment makes
a positive claim, the negative boundary is checked too: that the compiler fails
closed without a relation, that the uncompiled learner provably *cannot* express
the answer from its linear features, that the saturated model really does differ
from the typed one, and that the wrong-compiler control really does carry a
strictly positive floor.

The two claims the whole result rests on are checked independently of the
module's own audit: exactness of ``T == O == G``, and agreement between the
closed-form analytic reference and a from-scratch enumeration of outcomes.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "training_economy.py"
SPEC = importlib.util.spec_from_file_location("training_economy_01724", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
economy = importlib.util.module_from_spec(SPEC)
sys.modules["training_economy_01724"] = economy
SPEC.loader.exec_module(economy)

ZERO = Fraction(0)
ONE = Fraction(1)


@pytest.fixture(scope="module")
def loaded():
    return economy.load_successor()


@pytest.fixture(scope="module")
def schema(loaded):
    return loaded[0]


@pytest.fixture(scope="module")
def document(loaded):
    return loaded[1]


@pytest.fixture(scope="module")
def structure(schema):
    return economy.compile_classes(schema)


@pytest.fixture(scope="module")
def features(schema):
    return economy.build_feature_matrix(schema)


# --- the byte-pinned inputs -------------------------------------------------


def test_every_input_is_byte_pinned(document):
    assert document["schema"] == economy.SUCCESSOR_SCHEMA
    assert economy.load_preregistration()["schema"] == (
        economy.PREREGISTRATION_SCHEMA
    )
    assert economy.load_predecessor()["schema"] == economy.PREDECESSOR_SCHEMA


@pytest.mark.parametrize(
    "loader,path_attribute",
    [
        (economy.load_successor, "SUCCESSOR_PATH"),
        (economy.load_preregistration, "PREREGISTRATION_PATH"),
        (economy.load_predecessor, "PREDECESSOR_PATH"),
    ],
)
def test_a_changed_input_is_refused(loader, path_attribute, tmp_path):
    original = getattr(economy, path_attribute)
    corrupted = tmp_path / original.name
    corrupted.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        loader(corrupted)


# --- A. what Sigma' declares and no longer declares -------------------------


def test_successor_drops_the_decision_rule(document):
    predecessor = economy.load_predecessor()
    assert "rule" in predecessor["decision_type"]
    assert "decision_type" not in document
    assert "rule" not in document["answer_type"]
    assert document["answer_type"]["labels"] == ["a_0", "a_1"]


def test_successor_carries_the_structure_over_unchanged(document, schema):
    predecessor = economy.load_predecessor()
    assert document["sorts"] == predecessor["sorts"]
    assert document["relations"] == predecessor["relations"]
    assert len(schema.inputs) == 49
    assert schema.incidence is not None and len(schema.incidence) == 21


def test_successor_contains_no_answer_semantics(document, structure):
    certificate = economy.leakage_certificate(document, structure)
    assert certificate["sigma_prime_declares_no_forbidden_key"]
    assert certificate["forbidden_keys_observed"] == []
    assert certificate["sigma_prime_contains_no_float"]
    assert certificate["no_array_mixes_a_sort_identifier_with_an_answer_label"]
    assert certificate["compiler_carrier_has_no_answer_field"]
    assert certificate["no_fitting_function_accepts_a_target_parameter"]
    assert certificate["offending_functions"] == []


def test_the_leakage_scan_is_not_vacuous(document, structure):
    poisoned = json.loads(json.dumps(document))
    poisoned["answer_type"]["rule"] = "return a_0 iff the pair is marked"
    certificate = economy.leakage_certificate(poisoned, structure)
    assert not certificate["sigma_prime_declares_no_forbidden_key"]
    assert "rule" in certificate["forbidden_keys_observed"]

    mixed = json.loads(json.dumps(document))
    mixed["relations"]["leak"] = [["P0", "L0", "a_0"]]
    assert not economy.leakage_certificate(mixed, structure)[
        "no_array_mixes_a_sort_identifier_with_an_answer_label"
    ]


# --- B. the compiler --------------------------------------------------------


def test_compiler_derives_the_group_and_the_two_classes(structure):
    assert structure.group_order == 168
    assert sorted(structure.cardinalities) == [21, 28]
    assert len(structure.classes) == 2
    assert structure.cardinalities == (21, 28)


def test_class_order_is_canonical_and_answer_free(structure):
    sizes = [len(members) for members in structure.classes]
    assert sizes == sorted(sizes)
    for members in structure.classes:
        assert list(members) == sorted(members)
    assert structure.name(0) == "O_0" and structure.name(1) == "O_1"


def test_compiler_ignores_the_redundant_expectation_blocks(document, structure):
    stripped = json.loads(json.dumps(document))
    del stripped["derivable_self_checks"]
    del stripped["automorphisms"]
    rebuilt = economy.IncidenceSchema(
        points=tuple(stripped["sorts"]["point"]),
        lines=tuple(stripped["sorts"]["line"]),
        incidence=frozenset(
            (row[0], row[1]) for row in stripped["relations"]["incident"]
        ),
    )
    again = economy.compile_classes(rebuilt)
    assert again.classes == structure.classes
    assert again.group_order == structure.group_order


def test_compiler_fails_closed_without_a_relation(structure):
    erased = economy.build_relation_erased(structure.schema)
    assert len(erased.inputs) == 49
    assert len(erased.opaque_rows) == 21
    with pytest.raises(economy.MissingDeclaredRelation):
        economy.compile_classes(erased)
    with pytest.raises(economy.MissingDeclaredRelation):
        economy.build_feature_matrix(erased)


def test_automorphism_matcher_handles_repeated_line_point_sets(schema):
    """A naive unique-match enumerator would miss these automorphisms."""
    incidence = set(schema.incidence)
    for point in ("P2", "P4", "P5"):
        incidence.discard((point, "L6"))
    for point in ("P2", "P3", "P6"):
        incidence.add((point, "L6"))
    degenerate = economy.IncidenceSchema(
        points=schema.points,
        lines=schema.lines,
        incidence=frozenset(incidence),
    )
    assert len(degenerate.incidence) == 21
    assert degenerate.point_set("L5") == degenerate.point_set("L6")
    found = economy.enumerate_automorphisms(degenerate)
    swaps = [
        automorphism
        for automorphism in found
        if automorphism.point_images == degenerate.points
        and automorphism.line_images != degenerate.lines
    ]
    assert swaps, "the line transposition fixing every point must be found"


def test_generic_group_closure_recovers_the_same_partition(structure):
    generic = economy.invariance_partition_from_group(
        structure.inputs, structure.schema, structure.automorphisms
    )
    assert generic == structure.class_of_index
    assert len(set(generic)) == 2


def test_group_closure_is_not_vacuous(structure):
    """With only the identity supplied, every input stays its own parameter."""
    identity = economy.Automorphism(
        point_images=structure.schema.points,
        line_images=structure.schema.lines,
    )
    alone = economy.invariance_partition_from_group(
        structure.inputs, structure.schema, (identity,)
    )
    assert len(set(alone)) == 49


# --- C. the compiled TDM family --------------------------------------------


def test_omega_lambda_is_a_state_exactly_on_the_unit_interval():
    for numerator in range(13):
        value = Fraction(numerator, 12)
        assert economy.omega_lambda(value).is_state
        assert economy.omega_lambda(value).central_parameter == value
    for outside in (Fraction(-1, 12), Fraction(13, 12)):
        assert not economy.omega_lambda(outside).is_state


def test_central_test_is_a_central_effect_pair_summing_to_the_unit():
    assert economy.Z_MATRIX.is_effect and economy.Z_MATRIX.is_central
    assert economy.Z_SCALAR.is_effect and economy.Z_SCALAR.is_central
    total = economy.Element(
        "",
        economy.Z_MATRIX.a11 + economy.Z_SCALAR.a11,
        economy.Z_MATRIX.a12 + economy.Z_SCALAR.a12,
        economy.Z_MATRIX.a22 + economy.Z_SCALAR.a22,
        economy.Z_MATRIX.s + economy.Z_SCALAR.s,
    )
    assert (total.a11, total.a12, total.a22, total.s) == (
        economy.UNIT.a11,
        economy.UNIT.a12,
        economy.UNIT.a22,
        economy.UNIT.s,
    )


def test_tdm_distribution_is_exact_and_normalized():
    for numerator in range(25):
        value = Fraction(numerator, 24)
        first, second = economy.tdm_distribution(value)
        assert first == value
        assert first + second == ONE


def test_orientation_is_an_involution_of_the_parameter(structure):
    for numerator in range(25):
        value = Fraction(numerator, 24)
        upright = economy.tdm_distribution(value)
        flipped = economy.tdm_distribution(ONE - value)
        assert upright == (flipped[1], flipped[0])


def test_the_public_contracts_resolve_the_family_exactly(structure):
    grid = tuple(Fraction(index, 24) for index in range(25))
    bridge = economy.decision_model_bridge(
        grid, structure, economy.THETAS[0].values
    )
    assert bridge["resolves_exactly_to_lambda_and_one_minus_lambda"]
    assert bridge["normalizes_at_zero_tolerance"]
    assert bridge["public_family_matches_on_all_admitted_inputs"]
    assert bridge["anything_added_to_the_public_api"] is False
    assert bridge["public_api_symbol_count"] == 12


# --- D. fairness and information parity ------------------------------------


def test_U_can_express_the_class_function_exactly(features, structure):
    report = economy.realize_class_indicator(features, structure.class_of_index)
    assert report["class_function_is_exactly_realizable"]
    assert report["max_absolute_error_against_class_zero_indicator"] == 0.0
    assert report["hidden_units_used"] <= report["hidden_width_available"]


def test_U_can_express_a_single_pair_so_it_is_not_class_tied(features):
    for index in (0, 5, 17, 48):
        report = economy.realize_single_pair_indicator(features, index)
        assert report["single_pair_is_exactly_realizable"]


def test_the_class_indicator_is_not_handed_to_U(features, structure):
    report = economy.span_certificate(
        features, structure.class_of_index, structure.inputs
    )
    assert report["every_column_is_a_function_of_one_coordinate_alone"]
    assert report["no_column_is_a_function_of_the_class"]
    assert report["indicator_is_in_the_linear_span"] is False
    assert report["least_squares_max_absolute_residual"] > 0.0
    witness = report["additive_span_witness"]
    corners = witness["indicator_corners"]
    assert corners[0] - corners[1] - corners[2] + corners[3] != 0


def test_the_linear_candidate_provably_cannot_express_the_answer(
    features, structure
):
    """The negative boundary of the parity claim, computed independently."""
    design = economy.build_designs(features)["linear"]
    target = np.array(
        [1.0 if cell == 0 else 0.0 for cell in structure.class_of_index]
    )
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert float(np.max(np.abs(design @ solution - target))) > 1e-6


def test_the_quadratic_candidate_can_express_the_answer(features, structure):
    """The over-strong variant is honestly labeled: its span does contain it."""
    design = economy.build_designs(features)["quadratic"]
    target = np.array(
        [1.0 if cell == 0 else 0.0 for cell in structure.class_of_index]
    )
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert float(np.max(np.abs(design @ solution - target))) < 1e-8


def test_relabeling_the_answers_only_complements_the_estimates(structure):
    rng = random.Random(4242)
    trials = [rng.randrange(0, 6) for _ in structure.inputs]
    successes = [rng.randrange(0, value + 1) for value in trials]
    complement = [
        total - count for total, count in zip(trials, successes, strict=True)
    ]
    upright = economy.fit_bernoulli(
        successes, trials, structure.class_of_index, 2, ONE, ONE, "probe"
    )
    inverted = economy.fit_bernoulli(
        complement, trials, structure.class_of_index, 2, ONE, ONE, "probe"
    )
    for left, right in zip(
        upright.exact_predictions, inverted.exact_predictions, strict=True
    ):
        assert left + right == ONE


# --- the three conditions that must coincide, and the one that must not ----


def test_typed_oracle_and_generic_agree_exactly(structure):
    generic = economy.invariance_partition_from_group(
        structure.inputs, structure.schema, structure.automorphisms
    )
    saturated = tuple(range(len(structure.inputs)))
    rng = random.Random(20260923)
    differed_from_saturated = 0
    for _ in range(200):
        trials = [rng.randrange(0, 5) for _ in structure.inputs]
        successes = [rng.randrange(0, value + 1) for value in trials]
        typed = economy.fit_typed_tdm(
            successes, trials, structure.class_of_index, 2, ONE, ONE, "T"
        )
        oracle = economy.fit_bernoulli(
            successes, trials, structure.class_of_index, 2, ONE, ONE, "O"
        )
        symmetric = economy.fit_bernoulli(
            successes, trials, generic, len(set(generic)), ONE, ONE, "G"
        )
        pairwise = economy.fit_bernoulli(
            successes, trials, saturated, 49, ONE, ONE, "S"
        )
        assert typed.exact_predictions == oracle.exact_predictions
        assert typed.exact_predictions == symmetric.exact_predictions
        assert typed.degrees_of_freedom == 2
        assert pairwise.degrees_of_freedom == 49
        if pairwise.exact_predictions != typed.exact_predictions:
            differed_from_saturated += 1
    assert differed_from_saturated > 150, "the S comparison must not be vacuous"


def test_the_beta_estimator_convention(structure):
    assert economy.beta_posterior_mean(0, 0, ONE, ONE) == Fraction(1, 2)
    assert economy.beta_posterior_mean(3, 4, ONE, ONE) == Fraction(4, 6)
    assert economy.beta_posterior_mean(0, 10, ONE, ONE) == Fraction(1, 12)
    half = Fraction(1, 2)
    assert economy.beta_posterior_mean(0, 0, half, half) == Fraction(1, 2)


def test_clipping_never_touches_a_bernoulli_prediction():
    largest = max(economy.SAMPLE_SIZES)
    assert 1.0 / (largest + 2) > economy.CLIP
    assert 1.0 - 1.0 / (largest + 2) < 1.0 - economy.CLIP


# --- the generator and the splits ------------------------------------------


def test_the_generator_reproduces_the_declared_frequencies(structure):
    theta = (Fraction(1, 4), Fraction(3, 4))
    numerators, denominator = economy.theta_as_integers(theta)
    assert denominator == 4 and numerators == (1, 3)
    rng = random.Random(7)
    pool = tuple(range(len(structure.inputs)))
    observations = economy.generate_observations(
        rng, pool, 200_000, numerators, denominator, structure.class_of_index
    )
    for cell, expected in enumerate(theta):
        picked = [
            answer
            for index, answer in zip(
                observations.indices, observations.answers, strict=True
            )
            if structure.class_of_index[index] == cell
        ]
        assert abs(statistics.fmean(picked) - float(expected)) < 0.01


def test_theta_as_integers_is_exact():
    assert economy.theta_as_integers(
        (Fraction(1, 3), Fraction(2, 3))
    ) == ((1, 2), 3)
    assert economy.theta_as_integers(
        (Fraction(2, 5), Fraction(3, 5))
    ) == ((2, 3), 5)


def test_splits_are_deterministic_stratified_and_disjoint(structure):
    members = economy._class_members(structure)
    for replicate in range(20):
        seed = 555_000 + replicate
        first = economy.build_split(
            "B", random.Random(seed), members, economy.HOLDOUT_PER_CLASS
        )
        second = economy.build_split(
            "B", random.Random(seed), members, economy.HOLDOUT_PER_CLASS
        )
        assert first == second
        assert len(first.held_out) == 10
        assert len(first.train_pool) == 39
        assert not set(first.train_pool) & set(first.held_out)
        held = [structure.class_of_index[i] for i in first.held_out]
        assert held.count(0) == 4 and held.count(1) == 6
        seen = {structure.class_of_index[i] for i in first.train_pool}
        assert seen == {0, 1}


def test_regime_a_holds_nothing_out(structure):
    members = economy._class_members(structure)
    split = economy.build_split(
        "A", random.Random(0), members, economy.HOLDOUT_PER_CLASS
    )
    assert split.held_out == ()
    assert len(split.train_pool) == 49
    assert dict(split.evaluation)["aggregate"] == tuple(range(49))


def test_an_unknown_regime_is_refused(structure):
    members = economy._class_members(structure)
    with pytest.raises(ValueError):
        economy.build_split(
            "C", random.Random(0), members, economy.HOLDOUT_PER_CLASS
        )


# --- evaluation: the exact expectation is the large-fresh-sample limit -----


def test_exact_evaluation_matches_a_large_fresh_sample(structure):
    theta = (Fraction(1, 4), Fraction(3, 4))
    truths = [float(value) for value in theta]
    numerators, denominator = economy.theta_as_integers(theta)
    members = economy._class_members(structure)
    split = economy.build_split(
        "A", random.Random(0), members, economy.HOLDOUT_PER_CLASS
    )
    rng = random.Random(11)
    predictions = tuple(
        0.2 + 0.6 * ((index * 7) % 11) / 10.0 for index in range(49)
    )
    exact = economy.evaluate_exactly(
        predictions, split, truths, structure.class_of_index
    )["aggregate"]

    draws = 400_000
    observations = economy.generate_observations(
        rng,
        tuple(range(49)),
        draws,
        numerators,
        denominator,
        structure.class_of_index,
    )
    total = 0.0
    squares = 0.0
    for index, answer in zip(
        observations.indices, observations.answers, strict=True
    ):
        value = predictions[index]
        loss = -math.log(value if answer == 1 else 1.0 - value)
        total += loss
        squares += loss * loss
    mean = total / draws
    variance = squares / draws - mean * mean
    standard = math.sqrt(max(variance, 0.0) / draws)
    assert abs(mean - exact["nll"]) < 5.0 * standard


def test_excess_is_zero_exactly_at_the_truth(structure):
    theta = (Fraction(1, 4), Fraction(3, 4))
    truths = [float(value) for value in theta]
    members = economy._class_members(structure)
    split = economy.build_split(
        "A", random.Random(0), members, economy.HOLDOUT_PER_CLASS
    )
    perfect = tuple(
        truths[structure.class_of_index[index]] for index in range(49)
    )
    report = economy.evaluate_exactly(
        perfect, split, truths, structure.class_of_index
    )
    for block in report.values():
        assert abs(block["excess_nll"]) < 1e-12
        assert abs(block["calibration_error"]) < 1e-12


# --- the analytic reference, checked against an independent enumeration ----


def test_analytic_reference_matches_a_direct_enumeration(structure):
    """Enumerate every outcome sequence exactly and compare to the lattice sum."""
    theta = (Fraction(1, 4), Fraction(3, 4))
    weights = (Fraction(21, 49), Fraction(28, 49))
    size = 3
    reference = economy.analytic_reference(
        (size,), theta, ONE, ONE, weights, weights
    )
    expected = float(reference["sweep"][0]["exact_expected_excess_nll"])

    bayes = sum(
        float(weight)
        * economy._cross_entropy(float(truth), float(truth))
        for weight, truth in zip(weights, theta, strict=True)
    )
    total = 0.0
    for outcome in product(range(4), repeat=size):
        probability = ONE
        successes = [0, 0]
        trials = [0, 0]
        for code in outcome:
            cell, answer = divmod(code, 2)
            probability *= weights[cell] * (
                theta[cell] if answer else ONE - theta[cell]
            )
            trials[cell] += 1
            successes[cell] += answer
        estimates = [
            economy.beta_posterior_mean(
                successes[cell], trials[cell], ONE, ONE
            )
            for cell in (0, 1)
        ]
        loss = sum(
            float(weights[cell])
            * economy._cross_entropy(
                float(theta[cell]), float(estimates[cell])
            )
            for cell in (0, 1)
        )
        total += float(probability) * loss
    assert abs((total - bayes) - expected) < 1e-9


def test_the_asymptotic_law_is_one_over_n_for_matched_weights(structure):
    weights = (Fraction(21, 49), Fraction(28, 49))
    reference = economy.analytic_reference(
        (256, 1024), (Fraction(1, 4), Fraction(3, 4)), ONE, ONE, weights, weights
    )
    for row in reference["sweep"]:
        size = int(row["n"])
        assert abs(float(row["asymptotic_prediction"]) - 1.0 / size) < 1e-12
        assert (
            abs(float(row["exact_expected_excess_nll"]) - 1.0 / size)
            < 0.25 / size
        )


def test_the_analytic_reference_is_theta_robust(structure):
    weights = (Fraction(21, 49), Fraction(28, 49))
    for spec in economy.THETAS:
        reference = economy.analytic_reference(
            (512,), spec.values, ONE, ONE, weights, weights
        )
        value = float(reference["sweep"][0]["exact_expected_excess_nll"])
        assert abs(value - 1.0 / 512) < 0.25 / 512


# --- the three controls -----------------------------------------------------


def test_wrong_compiler_keeps_the_cardinalities_but_breaks_the_symmetry(
    structure,
):
    partition, certificate = economy.build_wrong_partition(structure)
    assert certificate["cardinalities"] == [21, 28]
    assert certificate["respects_the_declared_automorphisms"] is False
    assert certificate["automorphism_violations_counted"] > 0
    assert sorted(partition.count(cell) for cell in (0, 1)) == [21, 28]
    for cell in certificate["cell_composition"]:
        assert min(cell["true_class_counts"]) > 0, "each cell must be mixed"


def test_wrong_compiler_carries_a_strictly_positive_floor(structure):
    partition, _ = economy.build_wrong_partition(structure)
    everything = tuple(range(49))
    truths = [0.25, 0.75]
    wrong = economy.asymptotic_floor(
        partition, 2, truths, structure.class_of_index,
        everything, everything, ONE, ONE,
    )
    right = economy.asymptotic_floor(
        structure.class_of_index, 2, truths, structure.class_of_index,
        everything, everything, ONE, ONE,
    )
    assert wrong > 0.05
    assert abs(right) < 1e-12, "the correct partition must have no floor"


def test_no_scramble_reproduces_the_declared_quotient(structure):
    for scrambled in economy.build_scrambled_schemas(structure.schema, 20):
        assert scrambled.incidence != structure.schema.incidence
        assert len(scrambled.incidence) == 21
        assert len(scrambled.inputs) == 49
        derived = economy.compile_classes(scrambled)
        assert sorted(derived.cardinalities) != [21, 28]


def test_scrambled_ensemble_is_deterministic(structure):
    first = economy.build_scrambled_schemas(structure.schema, 5)
    second = economy.build_scrambled_schemas(structure.schema, 5)
    assert [s.incidence for s in first] == [s.incidence for s in second]


def test_saturated_model_cannot_transfer_to_unseen_identities(structure):
    everything = tuple(range(49))
    seen = tuple(index for index in everything if index % 5 != 0)
    unseen = tuple(index for index in everything if index % 5 == 0)
    truths = [0.25, 0.75]
    saturated = economy.asymptotic_floor(
        everything, 49, truths, structure.class_of_index,
        seen, unseen, ONE, ONE,
    )
    typed = economy.asymptotic_floor(
        structure.class_of_index, 2, truths, structure.class_of_index,
        seen, unseen, ONE, ONE,
    )
    assert saturated > 0.1
    assert abs(typed) < 1e-12


# --- the committed artifacts ------------------------------------------------


def test_the_exact_artifacts_match_the_executable_output(
    structure, document, features
):
    pairs = (
        (economy.ORBIT_PATH, economy.build_orbit_payload(
            structure, document, features
        )),
        (economy.SPLIT_PATH, economy.build_split_payload(structure)),
        (economy.REFERENCE_PATH, economy.build_reference_payload(structure)),
    )
    for path, payload in pairs:
        assert path.exists(), f"missing {path.name}"
        assert path.read_bytes() == economy._canonical(payload), path.name


def test_the_committed_results_pass_every_mechanical_check():
    results = json.loads(economy.RESULTS_PATH.read_text())
    assert results["schema"] == economy.RESULTS_SCHEMA
    assert results["all_mechanical_checks_pass"]
    assert all(results["mechanical_checks"].values())
    assert results["mechanical_check_count"] >= 35
    assert results["design"]["closed_form_replicates"] >= 100
    assert results["design"]["learned_replicates"] >= 100
    assert results["J_verdict"]["verdict"] in (
        "STRONG POSITIVE",
        "NARROW POSITIVE",
        "NULL",
        "NEGATIVE",
    )
    assert results["J_verdict"]["T_equals_O_exactly"]
    assert results["J_verdict"]["T_equals_G_exactly"]


def test_the_committed_results_cover_the_declared_grid():
    results = json.loads(economy.RESULTS_PATH.read_text())
    cells = results["cells"]
    for spec in economy.THETAS:
        for regime in economy.REGIMES:
            for size in economy.SAMPLE_SIZES:
                for condition in ("T", "O", "S", "G", "U_best"):
                    assert (
                        f"{spec.identifier}|{regime}|{size}|{condition}"
                        in cells
                    )


def test_the_fairness_frontier_shows_U_was_converged_and_bracketed():
    results = json.loads(economy.RESULTS_PATH.read_text())
    frontier = results["D_fairness_frontier"]
    assert frontier["rows"]
    for row in frontier["rows"]:
        assert row["grid_is_adequate"], row
        path = row["regularization_path"]["quadratic"]
        top = path[str(economy.U_RIDGE_GRID[-1])]
        assert top <= max(
            row["best_constant_model_floor"],
            row["shrink_to_one_half_floor"],
        ) * 1.05


def test_the_per_replicate_table_is_complete_and_well_formed():
    rows = economy._table_rows(economy.TABLE_PATH.read_bytes())
    assert rows[0] == list(economy.TABLE_COLUMNS)
    body = rows[1:]
    assert len(body) > 10_000
    conditions = {row[3] for row in body}
    assert {"T", "O", "S", "G", *economy.LEARNED_CONDITIONS} <= conditions
    for row in body[:500]:
        assert len(row) == len(economy.TABLE_COLUMNS)
        float(row[5])
        int(row[13])


def test_the_cost_report_accounts_for_compilation():
    cost = json.loads(economy.COST_PATH.read_text())
    assert cost["schema"] == economy.COST_SCHEMA
    seconds = cost["seconds"]
    for key in (
        "schema_parse",
        "automorphism_and_class_derivation",
        "closed_form_fitting",
        "learned_fitting",
        "total_wall_clock",
    ):
        assert key in seconds and seconds[key] >= 0.0
    assert cost["derivation_reuse"]["derivations_performed"] == 1
    assert cost["derivation_reuse"]["reused_across_the_sample_size_sweep"]


def test_the_threshold_table_prefers_the_typed_condition():
    results = json.loads(economy.RESULTS_PATH.read_text())
    primary = economy.THETAS[0].identifier
    for regime in economy.REGIMES:
        typed = results["thresholds"][f"{primary}|{regime}|T"]
        assert typed["labels_to_threshold"]["0.02"] is not None
        series = typed["mean_excess_nll_by_n"]
        ordered = [series[str(size)] for size in economy.SAMPLE_SIZES]
        assert ordered[0] > ordered[-1], "T must improve with more labels"


# --- the stop condition and the fences -------------------------------------


def test_the_stop_condition_and_fences_are_recorded():
    results = json.loads(economy.RESULTS_PATH.read_text())
    stop = results["stop_condition_respected"]
    assert not any(stop.values())
    fences = " ".join(results["claim_fences"]).lower()
    for forbidden in ("language", "llm", "physical", "issue 018"):
        assert forbidden in fences


def test_no_torch_or_network_dependency_was_introduced():
    assert "torch" not in sys.modules
    source = MODULE_PATH.read_text()
    for forbidden in ("import torch", "import requests", "urllib.request"):
        assert forbidden not in source

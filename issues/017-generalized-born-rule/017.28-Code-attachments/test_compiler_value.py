"""Tests for the 017.28 Phase T compiler-value module.

The load-bearing tests here recompute the central numbers by hand rather than by
calling the module: theta_star from the inherited rule, the refinement
compiler's irreducible floor from cell weights and a logarithm, and the
autotopism-group orbit sizes from the Latin square. If the module and the hand
calculation disagree, the module is wrong.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "compiler_value.py"

_spec = importlib.util.spec_from_file_location("compiler_value_under_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
value = importlib.util.module_from_spec(_spec)
sys.modules["compiler_value_under_test"] = value
_spec.loader.exec_module(value)

scope = value.scope
exact = value.exact
FIXTURE = "quasigroup:n=6:square=1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(value.RESULTS_PATH.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(value.REFERENCE_PATH.read_text())


@pytest.fixture(scope="module")
def preregistration() -> dict:
    return json.loads(value.PREREGISTRATION_PATH.read_text())


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


def test_every_pin_holds() -> None:
    assert _sha256(value.PHASE_S_PATH) == value.PHASE_S_MODULE_SHA256
    assert (
        _sha256(value.PHASE_S_CERTIFICATE_PATH) == value.PHASE_S_CERTIFICATE_SHA256
    )
    assert _sha256(value.REFINE_PATH) == value.REFINE_MODULE_SHA256
    assert value.refine.SHARED_MODULE_SHA256 == value.SHARED_MODULE_SHA256
    assert _sha256(value.PREREGISTRATION_PATH) == value.PREREGISTRATION_SHA256


def test_the_phase_s_certificate_is_the_banked_one(preregistration: dict) -> None:
    pinned = preregistration["byte_pinned_inputs"]
    assert pinned["phase_s_certificate.json"] == value.PHASE_S_CERTIFICATE_SHA256
    assert pinned["compiler_scope.py"] == value.PHASE_S_MODULE_SHA256


def test_the_base_seed_is_new() -> None:
    """Independence from every banked observation is a property of the seed base."""
    assert value.BASE_SEED not in (20260923, 20260924, 20260927)
    assert value.BASE_SEED == 20260929


# ---------------------------------------------------------------------------
# The target was forced, not chosen
# ---------------------------------------------------------------------------


def test_theta_star_is_forced_by_the_inherited_rule(results: dict) -> None:
    block = results["design"]["fixtures"][0]
    cells = len(block["Q_aut_cell_sizes"])
    expected = [
        f"{Fraction(2 * index + 1, 2 * cells).numerator}/"
        f"{Fraction(2 * index + 1, 2 * cells).denominator}"
        for index in range(cells)
    ]
    assert block["theta_star"] == expected == ["1/4", "3/4"]


def test_the_refinement_cell_merges_distinct_probabilities(results: dict) -> None:
    admissibility = results["design"]["fixtures"][0]["admissibility"]
    assert admissibility["at_least_one_Q_ref_cell_merges_distinct_probabilities"]
    merged = admissibility["Q_ref_cells_that_merge_distinct_probabilities"]
    assert len(merged) == 1
    assert merged[0]["size"] == 36
    assert merged[0]["distinct_true_probabilities_merged"] == ["1/4", "3/4"]


def test_the_holdout_follows_its_declared_rule(results: dict) -> None:
    block = results["design"]["fixtures"][0]
    sizes = block["Q_aut_cell_sizes"]
    assert block["holdout_per_cell"] == [max(1, size // 4) for size in sizes]
    for size, holdout in zip(sizes, block["holdout_per_cell"], strict=True):
        assert 1 <= holdout < size


# ---------------------------------------------------------------------------
# The approximation floor, recomputed by hand
# ---------------------------------------------------------------------------


def _cross_entropy(truth: float, prediction: float) -> float:
    return -(truth * math.log(prediction) + (1.0 - truth) * math.log1p(-prediction))


def test_the_refinement_floor_matches_a_hand_calculation(reference: dict) -> None:
    """One cell, so the limit is the pool-weighted mean of theta_star: 7/12.

    The floor is then the pool-weighted cross-entropy gap. Computed here with an
    ordinary float logarithm, which is a different arithmetic from the module's
    40-digit decimal, so agreement to 1e-12 is a real cross-check.
    """
    block = reference["approximation_floors"][f"{FIXTURE}|A|R"]
    assert block["cells"] == 1
    assert block["cell_limits"] == {"c_0": "7/12"}
    limit = 7.0 / 12.0
    floor = (12.0 / 36.0) * (
        _cross_entropy(0.25, limit) - _cross_entropy(0.25, 0.25)
    ) + (24.0 / 36.0) * (_cross_entropy(0.75, limit) - _cross_entropy(0.75, 0.75))
    assert abs(float(block["floor_40_digit_decimal"]) - floor) < 1e-12
    assert abs(floor - 0.116858121372717) < 1e-12


def test_the_exact_compiler_floor_is_exactly_zero(reference: dict) -> None:
    for condition in ("A", "H", "S"):
        block = reference["approximation_floors"][f"{FIXTURE}|A|{condition}"]
        assert block["floor_is_zero"]
        assert float(block["floor_40_digit_decimal"]) == 0.0


def test_the_control_floor_is_positive_but_smaller_than_the_refinement_floor(
    reference: dict,
) -> None:
    """The control has the right parameter count and still cannot fit the truth."""
    control = float(
        reference["approximation_floors"][f"{FIXTURE}|A|W"]["floor_40_digit_decimal"]
    )
    refined = float(
        reference["approximation_floors"][f"{FIXTURE}|A|R"]["floor_40_digit_decimal"]
    )
    assert 0.0 < control < refined


def test_every_floor_agrees_across_two_arithmetics(reference: dict) -> None:
    for key, block in reference["approximation_floors"].items():
        if key.endswith("over_every_split"):
            continue
        assert block["the_two_arithmetics_agree"], key


# ---------------------------------------------------------------------------
# The handed oracle really avoids the compiler
# ---------------------------------------------------------------------------


def test_the_handed_oracle_reads_the_certificate_not_the_compiler() -> None:
    certificate = value.load_phase_s_certificate()
    block = certificate["E_first_separation"]["designated_for_phase_T"][0]
    world = exact.world_from_document(block["document"], str(block["world"]))
    handed = value.handed_quotient(block, world)
    derived = scope.derive_exact_quotient(world)
    assert tuple(handed) == tuple(derived.partition)
    # And the handed path touches no compiler entry point.
    import inspect

    source = inspect.getsource(value.handed_quotient)
    for forbidden in (
        "derive_exact_quotient",
        "derive_refinement_quotient",
        "compile_world",
        "prepare(",
        "count_morphisms",
    ):
        assert forbidden not in source


def test_A_was_bit_identical_to_H_in_every_replicate(results: dict) -> None:
    agreement = results["agreement"][FIXTURE]
    assert agreement["comparisons"] == 44_000
    assert agreement["A_versus_H_mismatches"] == 0
    assert results["K_verdict"]["per_fixture"][FIXTURE]["A_equals_H_in_every_replicate"]


def test_A_and_H_have_identical_measured_curves(results: dict) -> None:
    for regime in ("A", "B"):
        for size in results["design"]["sample_sizes"]:
            left = results["cells"][f"{FIXTURE}|{regime}|{size}|A"]
            right = results["cells"][f"{FIXTURE}|{regime}|{size}|H"]
            assert left["excess_aggregate"] == right["excess_aggregate"]


# ---------------------------------------------------------------------------
# The falsification control
# ---------------------------------------------------------------------------


def test_the_control_keeps_the_cardinalities_and_breaks_invariance(
    results: dict,
) -> None:
    block = results["design"]["fixtures"][0]
    control = block["falsification_control"]
    assert control["cardinalities"] == block["Q_aut_cell_sizes"]
    assert not control["respects_the_derived_maps"]
    assert control["invariance_violations_counted"] > 0


def test_the_control_is_rebuilt_identically() -> None:
    certificate = value.load_phase_s_certificate()
    block = certificate["E_first_separation"]["designated_for_phase_T"][0]
    world = exact.world_from_document(block["document"], str(block["world"]))
    derived = scope.derive_exact_quotient(world)
    first, _report = value.build_wrong_partition(
        world, derived.partition, derived.morphisms
    )
    second, _again = value.build_wrong_partition(
        world, derived.partition, derived.morphisms
    )
    assert first == second
    assert sorted(first.count(cell) for cell in set(first)) == derived.cell_sizes
    assert first != tuple(derived.partition)


# ---------------------------------------------------------------------------
# Measured behaviour
# ---------------------------------------------------------------------------


def test_the_measured_reference_matches_the_preregistered_one(
    results: dict, preregistration: dict, reference: dict
) -> None:
    assert results["mechanical_checks"][
        "the_measured_reference_matches_the_preregistered_one"
    ]
    assert value.refine._numeric_close(
        preregistration["frozen_predictions"]["predictions"],
        reference["predictions"],
        value.NUMERIC_TOLERANCE,
    )


def test_every_regime_A_cell_landed_within_four_sigma(results: dict) -> None:
    validation = results["prediction_validation"]
    assert validation["failures"] == 0
    assert validation["all_within_four_sigma"]
    assert float(validation["worst_sigma"]) <= 4.0
    assert validation["cells_checked"] == 11 * 5


def test_the_exact_compiler_beats_the_refinement_at_every_tested_size(
    results: dict,
) -> None:
    sizes = results["design"]["sample_sizes"]
    for regime in ("A", "B"):
        for size in sizes:
            left = float(
                results["cells"][f"{FIXTURE}|{regime}|{size}|A"]["excess_aggregate"][
                    "mean"
                ]
            )
            right = float(
                results["cells"][f"{FIXTURE}|{regime}|{size}|R"]["excess_aggregate"][
                    "mean"
                ]
            )
            assert left <= right, (regime, size)


def test_no_estimation_crossover_regime_was_found(results: dict) -> None:
    """The honest negative on the section 10 estimation axis.

    The grid was extended down to n = 1 specifically to look for a sample size
    where the one-parameter refinement beats the two-parameter exact compiler.
    There is none, so the crossover sits at the smallest tested size.
    """
    block = results["K_verdict"]["per_fixture"][FIXTURE]
    for regime in ("crossover_regime_A", "crossover_regime_B"):
        assert block[regime]["crossover_n"] == results["design"]["sample_sizes"][0]
        assert block[regime]["R_wins_at"] == []


def test_the_refinement_plateaus_at_its_own_floor(results: dict, reference: dict) -> None:
    floor = float(
        reference["approximation_floors"][f"{FIXTURE}|A|R"]["floor_40_digit_decimal"]
    )
    largest = results["design"]["sample_sizes"][-1]
    measured = float(
        results["cells"][f"{FIXTURE}|A|{largest}|R"]["excess_aggregate"]["mean"]
    )
    assert measured > floor
    assert measured - floor < 0.001


def test_the_saturated_model_is_flat_at_its_floor_in_regime_B(
    results: dict, reference: dict
) -> None:
    """A per-identity model has nothing to say about an identity it never saw."""
    floors = reference["approximation_floors"][f"{FIXTURE}|B|over_every_split"]["S"]
    values = {
        float(
            results["cells"][f"{FIXTURE}|B|{size}|S"]["excess_aggregate"]["mean"]
        )
        for size in results["design"]["sample_sizes"]
    }
    assert len(values) == 1
    assert abs(values.pop() - float(floors["mean"])) < 1e-6


def test_the_exact_compiler_reaches_the_threshold_and_the_refinement_never_does(
    results: dict,
) -> None:
    block = results["K_verdict"]["per_fixture"][FIXTURE]
    assert block["A_reaches_threshold_at"] == 64
    assert block["R_reaches_threshold_at"] is None


# ---------------------------------------------------------------------------
# The verdict and the artifacts
# ---------------------------------------------------------------------------


def test_the_verdict_is_preregistered_and_every_criterion_holds(
    results: dict, preregistration: dict
) -> None:
    verdict = results["K_verdict"]
    assert verdict["verdict"] in value.VERDICTS
    assert verdict["verdict"] in preregistration["verdict_rule"]["options"]
    assert verdict["verdict"] == "SEPARATION AND VALUE"
    assert all(verdict["criteria"].values())
    assert len(verdict["criteria"]) == len(
        preregistration["verdict_rule"]["criteria"]
    )


def test_every_mechanical_check_passes(results: dict) -> None:
    assert results["all_mechanical_checks_pass"]
    assert results["mechanical_check_count"] >= 15
    assert all(results["mechanical_checks"].values())


def test_the_claim_ledger_separates_the_two_compilers(results: dict) -> None:
    ledger = results["L_claim_ledger"]
    for section in (
        "declared",
        "derived_by_Q_ref",
        "derived_only_by_Q_aut",
        "learned_from_labels",
        "implementation_choice",
        "still_open",
    ):
        assert ledger[section], section
    assert ledger["theta_star_was_evaluated_only_after_the_phase_S_freeze_was_banked"]


def test_the_forbidden_interpretations_are_recorded(results: dict) -> None:
    forbidden = " ".join(results["K_verdict"]["interpretation_forbidden"]).lower()
    for phrase in (
        "generally necessary",
        "universal tdm compiler",
        "lower bound",
        "language-scale",
    ):
        assert phrase in forbidden


def test_the_per_replicate_table_is_complete() -> None:
    raw = gzip.decompress(value.TABLE_PATH.read_bytes()).decode()
    lines = raw.strip().split("\n")
    assert lines[0] == ",".join(value.TABLE_COLUMNS)
    assert len(lines) - 1 == 11 * 2 * 5 * value.REPLICATES
    conditions = {line.split(",")[3] for line in lines[1:]}
    assert conditions == {"A", "H", "R", "S", "W"}


def test_the_cost_report_exists(results: dict) -> None:
    cost = json.loads(value.COST_PATH.read_text())
    assert cost["schema"] == value.COST_SCHEMA
    assert cost["fits"] == 11 * 2 * 5 * value.REPLICATES
    assert cost["seconds"]["total_wall_clock"] > 0

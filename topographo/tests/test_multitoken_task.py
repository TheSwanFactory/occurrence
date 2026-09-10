"""CI-safe tests for the `008.04` learning task, splits and baselines (no torch).

The authoritative run is a manual one over a 40000-input pool. These tests pin
the task construction on a small deterministic pool, because every claim the
`008.05` result makes rests on the split being clean: the target must not be an
availability rule, the held-out motif must not coincide with a fixed executor,
and the deterministic baselines must be scored on exactly the records the
learned policy is scored on.

``experiments.tlm_multitoken.policy`` needs torch and is therefore not imported
here, matching the convention in ``test_multitoken_program_space.py``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from experiments.tlm_multitoken import ambient, task
from experiments.tlm_multitoken.probe_program_space import (
    cyc_free_indices,
    cyc_nodes,
    force_seq_index,
    retained_ray,
    sign_bit,
)
from experiments.tlm_multitoken.programs import (
    CONSTRUCTORS,
    Cyc,
    EventLeaf,
    RetainedLeaf,
    Sand,
    enumerate_programs,
    evaluate,
    render,
)
from topographo.ssd import fips_basic, projective

#: Strided, not a prefix: the first Events all share ``i = 1`` and carry almost
#: no admissible pairs. Same subset rationale as the signature-search tests.
SUBSET = tuple(fips_basic.EVENTS[::7])


@lru_cache(maxsize=None)
def _pool() -> task.Pool:
    return task.materialize_pool(task.TaskConfig(pool_size=1200))


@lru_cache(maxsize=None)
def _splits() -> dict[str, task.Split]:
    return {split.name: split for split in task.build_splits(_pool())}


# --- the target relation ---------------------------------------------------


def test_target_grammar_reaches_exactly_the_cyc_free_bracketings():
    programs = enumerate_programs(3, CONSTRUCTORS)
    reachable = {
        task.target_index(combo, len(programs))
        for combo in _sign_representatives()
    }
    assert reachable == set(cyc_free_indices(programs))
    assert len(reachable) == 8


def test_target_index_is_the_sign_word_and_bit_d_governs_retained_depth_d():
    """Bit 0 is the sign of the LAST Event and governs the OUTERMOST head."""
    programs = enumerate_programs(3, CONSTRUCTORS)
    for combo in _sign_representatives():
        bits = task.sign_pattern(combo)
        index = task.target_index(combo, len(programs))
        assert index == bits, "bits < 8 < 20, so the modulus is inert"
        node = programs[index]
        for depth in range(3):
            expected_sand = bool((bits >> depth) & 1)
            assert isinstance(node, Sand) is expected_sand, (
                f"{render(programs[index])} at depth {depth} for bits {bits:03b}"
            )
            node = node.retained
        assert isinstance(node, RetainedLeaf)


def test_target_is_not_an_availability_rule():
    """Two inputs with the same availability pattern must want different programs.

    This is the `008.04` section 4 firewall, checked mechanically rather than
    asserted: if the target were a function of availability, no such pair exists.
    """
    pool = _pool()
    by_pattern: dict[int, set[int]] = {}
    for record in pool.records:
        if record.target_defined:
            by_pattern.setdefault(record.availability_pattern, set()).add(
                record.target_program
            )
    assert max(len(v) for v in by_pattern.values()) >= 2


def _sign_representatives():
    """One Event triple per sign-bit pattern, over all eight patterns."""
    positive = next(e for e in fips_basic.EVENTS if sign_bit(e) == 1)
    negative = next(e for e in fips_basic.EVENTS if sign_bit(e) == 0)
    out = []
    for bits in range(8):
        out.append(
            tuple(
                positive if (bits >> (2 - k)) & 1 else negative for k in range(3)
            )
        )
    return out


# --- program structure features -------------------------------------------


def test_structure_features_are_injective_over_the_program_space():
    programs = enumerate_programs(3, CONSTRUCTORS)
    rows = [tuple(task.program_structure_features(p, 3)) for p in programs]
    assert len(set(rows)) == len(programs) == 20


def test_structure_features_separate_the_two_nesting_directions():
    """Without a nesting feature the two fully grouped terms collide."""
    left = Cyc(Cyc(EventLeaf(0), EventLeaf(1)), EventLeaf(2))
    right = Cyc(EventLeaf(0), Cyc(EventLeaf(1), EventLeaf(2)))
    programs = enumerate_programs(3, CONSTRUCTORS)
    left_program = next(p for p in programs if render(p) == f"Occ({render(left)},r)")
    right_program = next(p for p in programs if render(p) == f"Occ({render(right)},r)")
    assert task.program_structure_features(
        left_program, 3
    ) != task.program_structure_features(right_program, 3)


def test_structure_features_mention_no_target_information():
    """A structural encoding must not vary with the input or the sign bits."""
    programs = enumerate_programs(3, CONSTRUCTORS)
    first = [task.program_structure_features(p, 3) for p in programs]
    second = [task.program_structure_features(p, 3) for p in programs]
    assert first == second


# --- pool ------------------------------------------------------------------


def test_pool_endpoints_agree_with_direct_exact_evaluation():
    pool = _pool()
    retained = retained_ray()
    checked = 0
    for record in pool.records[:24]:
        combo = tuple(fips_basic.EVENTS[k] for k in record.event_indices)
        for k, program in enumerate(pool.programs):
            value = evaluate(program, combo, retained)
            assert (value is None) == (not record.defined[k])
            checked += 1
    assert checked == 24 * 20


def test_endpoint_classes_are_exact_projective_identities():
    pool = _pool()
    retained = retained_ray()
    for record in pool.records[:16]:
        combo = tuple(fips_basic.EVENTS[k] for k in record.event_indices)
        values = {}
        for k, program in enumerate(pool.programs):
            value = evaluate(program, combo, retained)
            if value is None:
                continue
            other = values.get(record.endpoint_class[k])
            if other is not None:
                assert projective.equivalent(value, other)
            values[record.endpoint_class[k]] = value


def test_pool_features_are_the_frozen_denotations_only():
    """Ladder A hands over true denotations and nothing else: 4 rays of 16."""
    pool = _pool()
    assert pool.feature_dim() == 4 * 16
    retained = retained_ray()
    record = pool.records[0]
    expected: list[float] = []
    for k in record.event_indices:
        expected.extend(float(c) for c in fips_basic.EVENTS[k])
    expected.extend(float(c) for c in retained)
    assert list(pool.features[0]) == expected


# --- the viability gate ----------------------------------------------------


def test_viability_gate_conforms_to_the_008_02_pins():
    gate = task.viability_gate(_pool())
    assert gate.passed
    assert gate.conformance["n_programs"]["conforms"]
    assert gate.conformance["availability_ceiling"]["conforms"]
    assert gate.learned_headroom > 0.10
    assert gate.endpoint_disagreement_given_two == 1.0
    assert gate.max_distinct_endpoints_in_a_pattern == 12
    assert len(gate.target_program_pool) == 8


def test_viability_gate_records_every_pin():
    gate = task.viability_gate(_pool())
    assert set(gate.conformance) == set(task.PINS_00802)
    for entry in gate.conformance.values():
        assert entry["check"] in {"exact", "at_least", "at_most", "two_sided"}
        assert "deviation_from_pin" in entry


def test_sampled_availability_ceiling_is_biased_upward_not_downward():
    """The estimator over-states the ceiling on small samples, and that is safe.

    ``availability_ceiling`` sums per-pattern maxima over noisy hit counts, so a
    finite pool over-estimates the exhaustive `008.02` value and the bias shrinks
    with pool size. Over-estimating raises the bar the learned policy must clear,
    so the pin is checked from below only. This test pins the direction, so the
    one-sided check cannot quietly become a way to accept a ceiling that has
    collapsed.
    """
    small = task.viability_gate(task.materialize_pool(task.TaskConfig(pool_size=600)))
    large = task.viability_gate(task.materialize_pool(task.TaskConfig(pool_size=4000)))
    pin = task.PINS_00802["availability_ceiling"]
    assert small.availability_ceiling >= pin
    assert large.availability_ceiling >= pin
    assert large.availability_ceiling < small.availability_ceiling
    assert task.ONE_SIDED_PINS["availability_ceiling"] == "at_least"
    assert task.ONE_SIDED_PINS["learned_headroom"] == "at_most"


def test_viability_gate_refuses_a_space_without_headroom():
    """The gate is a gate: no headroom must mean no training."""
    gate = task.viability_gate(_pool(), min_headroom=0.99)
    assert gate.passed is False


# --- splits ----------------------------------------------------------------


def test_every_split_is_disjoint_and_lives_inside_the_scored_set():
    pool = _pool()
    scored = set(pool.scored)
    for split in _splits().values():
        split.check_disjoint()
        assert set(split.train) <= scored
        assert set(split.test) <= scored
        assert split.train and split.test


def test_motif_split_withholds_whole_target_motifs():
    pool = _pool()
    split = _splits()["motif"]
    train_targets = {pool.records[i].target_program for i in split.train}
    test_targets = {pool.records[i].target_program for i in split.test}
    assert not (train_targets & test_targets), "a held-out motif leaked into train"
    assert len(test_targets) == 2
    assert split.metadata["every_test_constituent_seen_in_train"] is True


def test_motif_split_does_not_hand_the_answer_to_a_fixed_executor():
    """`008.04` section 4: the target must not select ForceSeq or GroupedFirst.

    Withholding the homogeneous patterns 000 / 111 would have done exactly that,
    since the 000 target *is* the right comb. Pinned so the split cannot drift
    back to it.
    """
    pool = _pool()
    split = _splits()["motif"]
    assert split.metadata["heldout_sign_patterns"] == ["011", "100"]
    assert split.metadata["heldout_target_is_right_comb"] is False
    assert split.metadata["heldout_target_is_most_grouped"] is False
    right_comb = force_seq_index(pool.programs, 3)
    max_cyc = max(cyc_nodes(p) for p in pool.programs)
    for i in split.test:
        program = pool.records[i].target_program
        assert program != right_comb
        assert cyc_nodes(pool.programs[program]) != max_cyc


def test_motif_split_keeps_both_parities_in_training():
    assert _splits()["motif"].metadata["train_parities"] == [0, 1]


def test_parity_split_withholds_half_the_motifs_and_one_parity():
    split = _splits()["motif_parity"]
    assert split.metadata["heldout_sign_patterns"] == ["001", "010", "100", "111"]
    assert len(split.metadata["test_target_programs_unseen_in_train"]) == 4
    assert split.metadata["train_parities"] == [0]
    assert split.metadata["every_test_constituent_seen_in_train"] is True


def test_unseen_event_split_really_withholds_event_identities():
    pool = _pool()
    split = _splits()["unseen_events"]
    heldout = set(split.metadata["heldout_event_indices"])
    assert len(heldout) == 21
    for i in split.train:
        assert not (set(pool.records[i].event_indices) & heldout)
    for i in split.test:
        assert set(pool.records[i].event_indices) & heldout
    # Both sign bits must be represented, or the split would confound the target.
    assert len(split.metadata["heldout_event_sign_bits"]) == 2


def test_split_digest_is_membership_not_row_order():
    pool = _pool()
    split = _splits()["motif"]
    shuffled = task.Split(
        name=split.name,
        kind=split.kind,
        rationale=split.rationale,
        train=tuple(reversed(split.train)),
        test=tuple(reversed(split.test)),
        metadata=split.metadata,
    )
    assert task.split_digest(pool, split) == task.split_digest(pool, shuffled)


def test_split_construction_is_reproducible():
    first = {n: task.split_digest(_pool(), s) for n, s in _splits().items()}
    rebuilt = {
        s.name: task.split_digest(_pool(), s) for s in task.build_splits(_pool())
    }
    assert first == rebuilt


def test_restrict_split_preserves_held_out_membership():
    pool = _pool()
    split = _splits()["motif"]
    keep = sorted(set(split.train) | set(list(split.test)[: len(split.test) // 2]))
    restricted = task.restrict_split(split, keep, suffix="")
    assert set(restricted.test) <= set(split.test)
    assert restricted.metadata["dropped_test"] == len(split.test) - len(restricted.test)
    del pool


# --- baselines -------------------------------------------------------------


def test_every_required_baseline_is_scored_on_both_halves():
    pool = _pool()
    scores = task.score_baselines(pool, _splits()["motif"])
    for half in ("train", "test"):
        names = {s.name for s in scores if s.half == half}
        assert names == set(task.BASELINES)


def test_supplied_bracketing_is_exact_by_construction():
    """Handing the tree over is the `H_weak` ceiling, so it cannot miss."""
    scores = task.score_baselines(_pool(), _splits()["motif"])
    supplied = next(s for s in scores if s.name == "SuppliedBracketingExecutor")
    assert supplied.exact_native_success == 1.0
    assert supplied.deployable is False


def test_availability_oracle_bounds_every_availability_rule_on_its_own_half():
    pool = _pool()
    for split in _splits().values():
        scores = task.score_baselines(pool, split)
        for half in ("train", "test"):
            oracle = next(
                s
                for s in scores
                if s.name == "AvailabilityOracleCeiling" and s.half == half
            )
            for name in (
                "GroupedFirst",
                "ForceSeq",
                "RandomLegalTree",
                "AvailabilityRuleTrainFitted",
            ):
                rule = next(s for s in scores if s.name == name and s.half == half)
                assert rule.exact_native_success <= oracle.exact_native_success + 1e-12


def test_the_oracle_is_not_treated_as_the_bar():
    """The bar is deployable rules only; the oracle is fitted on the scored half."""
    scores = task.score_baselines(_pool(), _splits()["motif"])
    bar = task.best_deterministic(scores)
    assert bar.deployable is True
    assert bar.name not in {"AvailabilityOracleCeiling", "SuppliedBracketingExecutor"}


def test_deterministic_baselines_collapse_on_the_compositional_splits():
    """No deployable deterministic rule may be close to solving the primary split."""
    pool = _pool()
    for name in ("motif", "motif_parity"):
        bar = task.best_deterministic(task.score_baselines(pool, _splits()[name]))
        assert bar.exact_native_success < 0.25, (name, bar.name)


def test_sign_pattern_lookup_saturates_the_non_compositional_splits():
    """Why ``random`` and ``unseen_events`` are controls rather than claims."""
    pool = _pool()
    for name in ("random", "unseen_events"):
        scores = task.score_baselines(pool, _splits()[name])
        lookup = next(
            s
            for s in scores
            if s.name == "SignPatternLookupTrainFitted" and s.half == "test"
        )
        assert lookup.exact_native_success == 1.0


def test_score_selector_counts_an_undefined_choice_as_a_miss():
    pool = _pool()
    indices = _splits()["motif"].test

    def always_zero(record: task.Record) -> int:
        return 0

    score = task.score_selector(
        pool, indices, always_zero, name="probe", split="motif", half="test"
    )
    assert score.selected_defined_share <= 1.0
    assert score.exact_native_success <= score.selected_defined_share


def test_score_selector_separates_tree_identity_from_endpoint_identity():
    pool = _pool()
    indices = _splits()["random"].test
    score = task.score_selector(
        pool,
        indices,
        lambda record: record.target_program,
        name="probe",
        split="random",
        half="test",
    )
    assert score.tree_selection_accuracy == 1.0
    assert score.endpoint_match_via_different_tree == 0.0


# --- the ambient control ---------------------------------------------------


def test_ambient_agrees_with_native_on_the_cyc_free_programs():
    """The control must not change the relation, only the availability."""
    pool = task.materialize_pool(task.TaskConfig(pool_size=200))
    amb = ambient.materialize_ambient(pool)
    assert amb.agrees_with_native_on_cyc_free is True
    for native_record, ambient_record in zip(pool.records, amb.records, strict=True):
        assert native_record.target_program == ambient_record.target_program
        assert native_record.event_indices == ambient_record.event_indices


def test_ambient_widens_availability_rather_than_narrowing_it():
    pool = task.materialize_pool(task.TaskConfig(pool_size=200))
    amb = ambient.materialize_ambient(pool)
    for native_record, ambient_record in zip(pool.records, amb.records, strict=True):
        assert set(native_record.legal) <= set(ambient_record.legal)
    native_mean = sum(len(r.legal) for r in pool.records) / len(pool.records)
    ambient_mean = sum(len(r.legal) for r in amb.records) / len(amb.records)
    assert ambient_mean > native_mean


def test_ambient_cyc_ignores_the_fips_admissibility_precondition():
    inadmissible = next(
        (a, b)
        for a in fips_basic.EVENTS[:12]
        for b in fips_basic.EVENTS[:12]
        if not fips_basic.admissible(a, b)
    )
    term = Cyc(EventLeaf(0), EventLeaf(1))
    combo = (inadmissible[0], inadmissible[1], inadmissible[0])
    assert evaluate(term, combo, retained_ray()) is None
    assert ambient.ambient_evaluate(term, combo, retained_ray()) is not None


def test_ambient_carries_its_fence():
    assert "not OT occurrence" in ambient.AMBIENT_FENCE
    assert "does not enlarge the strict OT domain" in ambient.AMBIENT_FENCE


# --- refusals --------------------------------------------------------------


def test_materialize_pool_refuses_a_constructor_set_with_no_programs():
    with pytest.raises(ValueError, match="no programs"):
        task.materialize_pool(
            task.TaskConfig(pool_size=8, constructors=frozenset({"Cyc"}))
        )


def test_split_refuses_a_leak():
    leaky = task.Split(
        name="leaky", kind="broken", rationale="", train=(1, 2, 3), test=(3, 4)
    )
    with pytest.raises(ValueError, match="leaks"):
        leaky.check_disjoint()


def test_pool_on_a_small_event_subset_still_gates():
    """The gate machinery must work off the full 84-Event design too."""
    pool = task.materialize_pool(task.TaskConfig(pool_size=300), events=SUBSET)
    gate = task.viability_gate(pool)
    assert gate.n_records == 300
    assert gate.learned_headroom > 0.10


def test_pins_are_transcribed_from_the_published_008_02_artifact():
    """The conformance pins must be the published numbers, not remembered ones.

    ``PINS_00802`` is what the `008.04` viability gate checks against, so a typo
    there would silently move the bar. Derived here from
    ``program_space_report.json``, which is the `008.02` exhaustive sweep over all
    84 Events.
    """
    report = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "experiments"
            / "tlm_multitoken"
            / "program_space_report.json"
        ).read_text()
    )
    cell = next(
        c
        for c in report["signatures"]
        if c["signature"] == "E^3 R" and c["constructors"] == "Cyc+Occ+Sand"
    )
    published = {
        "n_programs": cell["n_programs"],
        "joint_domain_share": cell["inputs_with_two_or_more_defined"] / cell["n_inputs"],
        "endpoint_disagreement_given_two": cell["endpoint_disagreement_rate_given_two"],
        "n_availability_patterns": cell["n_availability_patterns"],
        "availability_ceiling": cell["availability_ceiling"],
        "learned_headroom": cell["learned_headroom"],
        "max_distinct_endpoints_in_a_pattern": cell[
            "max_distinct_endpoints_in_a_pattern"
        ],
        "target_defined_share": cell["target_defined"] / cell["n_inputs"],
    }
    assert set(published) == set(task.PINS_00802)
    for key, pin in task.PINS_00802.items():
        assert abs(published[key] - pin) < 5e-5, key

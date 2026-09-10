"""CI-safe tests for the Issue-008 signature-search probe (no torch).

The authoritative sweep is over all 84 Events and is a manual run. These tests
pin the machinery and a small deterministic Event subset, so the availability
ceiling, the baselines, and the Event decoding cannot drift silently.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from experiments.tlm_multitoken.native import rays_equal
from experiments.tlm_multitoken.probe_program_space import (
    MIN_LEARNED_HEADROOM,
    Evaluator,
    cyc_free_indices,
    cyc_nodes,
    decode_event,
    force_seq_index,
    grouped_first_order,
    retained_ray,
    sign_bit,
    sweep,
)
from experiments.tlm_multitoken.programs import (
    CONSTRUCTORS,
    OCC_CYC_ONLY,
    EventLeaf,
    Occ,
    RetainedLeaf,
    enumerate_programs,
    evaluate,
    kinds_used,
    render,
)
from topographo.ssd import exact, fips_basic

#: A strided subset, not a prefix. The first 12 Events all share ``i = 1`` and
#: carry almost no admissible pairs, which makes any prefix degenerate.
SUBSET = tuple(fips_basic.EVENTS[::7])


@lru_cache(maxsize=None)
def _sweep(n_events: int, constructors: frozenset[str]):
    """Cached so the shared E^3 R subset sweep runs once per constructor set."""
    return sweep(n_events, SUBSET, retained_ray(), constructors)


# --- Event decoding --------------------------------------------------------


def test_decode_event_recovers_the_certified_design():
    for event in fips_basic.EVENTS:
        i, j, s = decode_event(event)
        assert 1 <= i <= 7
        assert 1 <= j <= 7
        assert i != j
        assert s in (-1, 1)
        rebuilt = exact.add(exact.basis(i), exact.scale(s, exact.basis(8 + j)))
        assert rays_equal(event, rebuilt)


def test_decode_event_refuses_a_non_event():
    with pytest.raises(ValueError, match="not a basic Event"):
        decode_event(exact.basis(4))


def test_sign_bit_splits_the_events_evenly():
    bits = [sign_bit(event) for event in fips_basic.EVENTS]
    assert sum(bits) == 42
    assert len(bits) == 84


# --- memoized evaluator agrees with the reference ---------------------------


def test_memoized_evaluator_matches_direct_evaluation():
    r = retained_ray()
    memo = Evaluator(r)
    programs = enumerate_programs(3, CONSTRUCTORS)
    checked = 0
    for a in SUBSET[:4]:
        for b in SUBSET[:4]:
            for c in SUBSET[:4]:
                combo = (a, b, c)
                for program in programs:
                    fast = memo(program, combo)
                    slow = evaluate(program, combo, r)
                    assert (fast is None) == (slow is None), render(program)
                    if slow is not None:
                        assert rays_equal(fast, slow), render(program)
                    checked += 1
    assert checked == 4**3 * 20


# --- baseline definitions --------------------------------------------------


def test_force_seq_index_selects_the_right_comb():
    for n_events in (1, 2, 3):
        programs = enumerate_programs(n_events, CONSTRUCTORS)
        index = force_seq_index(programs, n_events)
        term = programs[index]
        assert kinds_used(term) <= {"Occ"}
        expected = RetainedLeaf()
        for k in reversed(range(n_events)):
            expected = Occ(EventLeaf(k), expected)
        assert term == expected


def test_grouped_first_order_prefers_more_grouping():
    programs = enumerate_programs(3, CONSTRUCTORS)
    order = grouped_first_order(programs)
    assert sorted(order) == list(range(len(programs)))
    counts = [cyc_nodes(programs[k]) for k in order]
    assert counts == sorted(counts, reverse=True)
    # The most grouped programs carry two Cyc nodes; the right comb carries none.
    assert counts[0] == 2
    assert counts[-1] == 0


def test_cyc_free_indices_match_the_grammar_count():
    # 2^n Cyc-free programs: each of the n retained heads is Occ or Sand.
    for n_events in (1, 2, 3, 4):
        programs = enumerate_programs(n_events, CONSTRUCTORS)
        assert len(cyc_free_indices(programs)) == 2**n_events


# --- the sweep on a pinned subset -----------------------------------------


@pytest.mark.parametrize("n_events", [2, 3])
def test_occ_cyc_only_never_clears_the_headroom_threshold(n_events):
    """The 017 constructor set is availability-saturated: no room for a policy."""
    result = _sweep(n_events, OCC_CYC_ONLY)
    assert result.availability_ceiling == 1.0
    assert result.learned_headroom == 0.0
    assert result.conditions["no_availability_rule_reproduces_the_target"] is False


@pytest.mark.parametrize(
    "n_events,programs,inputs,patterns",
    [(2, 6, 144, 5), (3, 20, 1728, 23)],
)
def test_adding_sand_opens_real_headroom(n_events, programs, inputs, patterns):
    result = _sweep(n_events, CONSTRUCTORS)
    assert result.n_programs == programs
    assert result.n_inputs == inputs
    assert result.n_availability_patterns == patterns
    # Every input has a genuine choice, and the choices always disagree.
    assert result.inputs_with_two_or_more_defined == inputs
    assert result.endpoint_disagreement_rate_given_two == 1.0
    assert result.learned_headroom > MIN_LEARNED_HEADROOM
    assert all(result.conditions.values())


def test_availability_ceiling_upper_bounds_every_availability_rule():
    """GroupedFirst and ForceSeq are availability rules, so neither can exceed it."""
    for n_events in (2, 3):
        for constructors in (OCC_CYC_ONLY, CONSTRUCTORS):
            result = _sweep(n_events, constructors)
            assert result.grouped_first_accuracy <= result.availability_ceiling + 1e-12
            assert result.force_seq_accuracy <= result.availability_ceiling + 1e-12
            assert result.random_legal_accuracy <= result.availability_ceiling + 1e-12


def test_supplied_bracketing_is_exact_by_construction():
    """Handing over the grammar is the H_weak ceiling: it cannot miss."""
    result = _sweep(3, CONSTRUCTORS)
    assert result.supplied_bracketing_accuracy == 1.0


def test_target_pool_is_recorded_for_audit():
    result = _sweep(3, CONSTRUCTORS)
    assert len(result.target_program_pool) == 8
    # The sign-bit grammar reaches the eight Cyc-free bracketings.
    assert all("Cyc(" not in name for name in result.target_program_pool)
    assert len(set(result.target_program_pool)) == 8


def test_full_84_event_e2r_occ_cyc_sparsity_is_the_017_obstruction():
    """The decisive 017 comparison, over all 84 Events (about one second).

    The target-independent fact is the joint domain: only 288 of 7056 inputs
    admit more than one legal program at all, so availability already fixes the
    endpoint and no target construction can defeat an availability rule. The
    exhaustive `E^3 R` sweep is a manual run; this pins the cheap cell.
    """
    result = sweep(2, tuple(fips_basic.EVENTS), retained_ray(), OCC_CYC_ONLY)
    assert result.n_programs == 2
    assert result.n_inputs == 7056
    assert result.defined_histogram == {0: 192, 1: 6576, 2: 288}
    assert result.inputs_with_two_or_more_defined == 288
    assert result.inputs_with_endpoint_disagreement == 192
    assert result.max_distinct_endpoints_in_a_pattern == 2
    assert result.availability_ceiling == pytest.approx(0.9732, abs=5e-5)
    assert result.learned_headroom < MIN_LEARNED_HEADROOM
    assert result.conditions["two_or_more_legal_classes_jointly_defined"] is False
    assert result.conditions["no_availability_rule_reproduces_the_target"] is False
    # GroupedFirst sits exactly on the ceiling: it is already an optimal
    # availability rule here, which is why 017.24 could not beat it.
    assert result.grouped_first_accuracy == pytest.approx(result.availability_ceiling)


def test_sweep_refuses_a_constructor_set_with_no_programs():
    with pytest.raises(ValueError, match="no programs"):
        sweep(2, SUBSET, retained_ray(), frozenset({"Cyc"}))

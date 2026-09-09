"""CI-safe tests for the Issue-008 strict program-tree enumeration (no torch).

Pins the shape counts against their closed forms, planarity, and the identity
that the 017 two-program family is exactly the ``E^2 R`` Occ/Cyc subset.
"""

from __future__ import annotations

from math import comb

import pytest

from experiments.tlm_multitoken.programs import (
    CONSTRUCTORS,
    OCC_CYC_ONLY,
    Cyc,
    EventLeaf,
    Occ,
    RetainedLeaf,
    Sand,
    count_programs,
    enumerate_event_terms,
    enumerate_programs,
    evaluate,
    event_leaf_indices,
    kinds_used,
    render,
    signature,
)
from topographo.ssd import exact, fips_basic


def _catalan(n: int) -> int:
    return comb(2 * n, n) // (n + 1)


# --- shape counts ----------------------------------------------------------


@pytest.mark.parametrize(
    "n_events,expected",
    [(0, 1), (1, 2), (2, 6), (3, 20), (4, 70), (5, 252)],
)
def test_full_constructor_shape_counts_are_central_binomials(n_events, expected):
    """1, 2, 6, 20, 70, 252 = C(2n, n)."""
    assert count_programs(n_events, CONSTRUCTORS) == expected
    assert expected == comb(2 * n_events, n_events)


@pytest.mark.parametrize(
    "n_events,expected",
    [(0, 1), (1, 1), (2, 2), (3, 5), (4, 14), (5, 42)],
)
def test_occ_cyc_shape_counts_are_catalan(n_events, expected):
    """Withholding Sand leaves the Catalan numbers."""
    assert count_programs(n_events, OCC_CYC_ONLY) == expected
    assert expected == _catalan(n_events)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_event_term_counts_are_catalan_of_k_minus_one(k):
    assert len(enumerate_event_terms(0, k, CONSTRUCTORS)) == _catalan(k - 1)


def test_withholding_cyc_leaves_one_event_term_per_leaf():
    assert len(enumerate_event_terms(0, 1, frozenset({"Occ", "Sand"}))) == 1
    assert enumerate_event_terms(0, 2, frozenset({"Occ", "Sand"})) == ()


def test_no_retained_head_yields_no_programs():
    assert enumerate_programs(2, frozenset({"Cyc"})) == ()


# --- structure -------------------------------------------------------------


@pytest.mark.parametrize("n_events", [0, 1, 2, 3, 4, 5])
def test_programs_are_planar_and_use_each_input_once(n_events):
    for program in enumerate_programs(n_events, CONSTRUCTORS):
        assert event_leaf_indices(program) == tuple(range(n_events)), render(program)


@pytest.mark.parametrize("n_events", [0, 1, 2, 3, 4, 5])
def test_shapes_are_distinct(n_events):
    programs = enumerate_programs(n_events, CONSTRUCTORS)
    assert len(set(programs)) == len(programs)


def test_occ_cyc_subset_is_contained_in_the_full_set():
    for n_events in range(5):
        subset = set(enumerate_programs(n_events, OCC_CYC_ONLY))
        full = set(enumerate_programs(n_events, CONSTRUCTORS))
        assert subset <= full
        assert all("Sand" not in kinds_used(p) for p in subset)


def test_e2r_occ_cyc_subset_is_exactly_the_017_two_program_family():
    p_seq = Occ(EventLeaf(0), Occ(EventLeaf(1), RetainedLeaf()))
    p_grp = Occ(Cyc(EventLeaf(0), EventLeaf(1)), RetainedLeaf())
    assert set(enumerate_programs(2, OCC_CYC_ONLY)) == {p_seq, p_grp}
    assert render(p_seq) == "Occ(a1,Occ(a2,r))"
    assert render(p_grp) == "Occ(Cyc(a1,a2),r)"


def test_signature_rendering():
    assert signature(0) == "R"
    assert signature(1) == "E R"
    assert signature(3) == "E^3 R"
    with pytest.raises(ValueError):
        signature(-1)


# --- evaluation ------------------------------------------------------------


def test_evaluate_reproduces_the_017_witness_disagreement():
    """017.07 section 5: same leaves, different tree, different endpoint."""
    from experiments.tlm_multitoken.native import ray, rays_equal

    z = ray(exact.add(exact.basis(1), exact.basis(10)))
    w = ray(exact.add(exact.basis(4), exact.basis(15)))
    x = exact.basis(4)

    p_seq = Occ(EventLeaf(0), Occ(EventLeaf(1), RetainedLeaf()))
    p_grp = Occ(Cyc(EventLeaf(0), EventLeaf(1)), RetainedLeaf())

    out_seq = evaluate(p_seq, (z, w), x)
    out_grp = evaluate(p_grp, (z, w), x)
    assert out_seq is not None and out_grp is not None
    assert not rays_equal(out_seq, out_grp)
    # Expected rays [e1] and [z].
    assert rays_equal(out_seq, exact.basis(1))
    assert rays_equal(out_grp, z)


def test_evaluate_propagates_undefinedness_from_an_inadmissible_cyc():
    a = b = fips_basic.EVENTS[0]
    assert not fips_basic.admissible(a, b)
    program = Occ(Cyc(EventLeaf(0), EventLeaf(1)), RetainedLeaf())
    assert evaluate(program, (a, b), exact.basis(4)) is None


def test_evaluate_returns_the_leaf_for_the_trivial_program():
    r = exact.basis(4)
    assert evaluate(RetainedLeaf(), (), r) == r


def test_every_e3r_program_evaluates_or_reports_undefined():
    """No exception escapes a well-typed program; undefinedness is returned."""
    r = exact.basis(4)
    programs = enumerate_programs(3, CONSTRUCTORS)
    triples = [
        (fips_basic.EVENTS[i], fips_basic.EVENTS[j], fips_basic.EVENTS[k])
        for i, j, k in [(0, 1, 2), (0, 5, 10), (3, 17, 42), (7, 7, 7)]
    ]
    total_defined = 0
    for events in triples:
        outcomes = [evaluate(program, events, r) for program in programs]
        assert len(outcomes) == 20
        total_defined += sum(1 for out in outcomes if out is not None)
    assert total_defined > 0


def test_a_fully_undefined_triple_is_reported_not_raised():
    """Whole-space undefinedness happens and must stay reportable data."""
    r = exact.basis(4)
    events = (fips_basic.EVENTS[0], fips_basic.EVENTS[1], fips_basic.EVENTS[2])
    outcomes = [evaluate(p, events, r) for p in enumerate_programs(3, CONSTRUCTORS)]
    assert all(out is None for out in outcomes)


def test_kinds_used_reports_the_constructors_present():
    program = Sand(Cyc(EventLeaf(0), EventLeaf(1)), Occ(EventLeaf(2), RetainedLeaf()))
    assert kinds_used(program) == {"Sand", "Cyc", "Occ"}
    assert kinds_used(RetainedLeaf()) == frozenset()
    assert kinds_used(EventLeaf(0)) == frozenset()

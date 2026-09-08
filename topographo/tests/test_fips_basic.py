"""Exact finite FIPS table tests (005.04 §1.2 G1–G4)."""

from __future__ import annotations

from topographo.ssd import fips_basic
from topographo.ssd.frames import is_edge


def test_table_counts_and_checksum():
    assert len(fips_basic.EVENTS) == 84
    assert len(fips_basic.ORDERED_EDGES) == 336
    assert len(fips_basic.BLOCKS) == 56
    assert fips_basic.TABLE_SHA256 == fips_basic.EXPECTED_TABLE_SHA256
    assert len(fips_basic.TABLE_SHA256) == 64


def test_g1_admissible_pair_has_unique_forced_third():
    a = fips_basic.EVENTS[0]
    b = fips_basic.partners(a)[0]
    assert fips_basic.admissible(a, b)
    third = fips_basic.third(a, b)
    assert third in fips_basic.EVENTS
    assert is_edge(a, b)
    # uniqueness: table maps exactly one third
    matches = [t for (x, y, t) in fips_basic.ORDERED_EDGES if x == a and y == b]
    assert matches == [third]


def test_g2_non_admissible_pair_raises_and_ask_nonadmits():
    a = fips_basic.EVENTS[0]
    # self-pair is never an edge
    assert not fips_basic.admissible(a, a)
    try:
        fips_basic.third(a, a)
        assert False, "expected KeyError"
    except KeyError:
        pass

    from topographo.ssd.outcome_runtime import (
        Admissibility,
        FipsClosure,
        NonAdmission,
        OperationDomainInadmissible,
        Presentation,
        ask,
    )
    from topographo.ssd.seal import seal

    outcome = ask(Presentation((seal(a), seal(a))), Admissibility(True), FipsClosure())
    assert isinstance(outcome, NonAdmission)
    assert isinstance(outcome.reason, OperationDomainInadmissible)


def test_g3_every_block_closes_under_cyclic_permutation():
    for block in fips_basic.BLOCKS:
        members = set(block)
        edge_count = 0
        for a in block:
            for b in block:
                if a == b:
                    continue
                if fips_basic.admissible(a, b):
                    assert fips_basic.third(a, b) in members
                    edge_count += 1
        assert edge_count == 6

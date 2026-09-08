"""Structural control C1–C5 (005.07 §2.3)."""

from __future__ import annotations

from topographo.ssd import fips_basic, structural_control


def test_c1_c5_degree_matched_rewiring():
    control = structural_control.build_degree_matched_rewiring(seed=0)
    cert = structural_control.assert_c1_c5(control)
    assert cert.c1_event_count == 84
    assert cert.c1_pair_count == 336
    assert cert.c2_matches_fips
    assert cert.c2_degree_histogram[4] == 84
    assert cert.c3_fips_closed_blocks == 56
    assert cert.c3_control_closed_blocks == 0
    assert cert.c3_differs
    assert cert.c4_differing_thirds == 336
    assert cert.c4_not_mere_relabeling
    assert "cyclic block closure" in cert.c5_broken_laws[0]


def test_control_preserves_domain_not_fips_thirds():
    control = structural_control.build_degree_matched_rewiring(seed=1)
    for a, b, t in fips_basic.ORDERED_EDGES:
        assert control.admissible(a, b)
        assert control.third(a, b) != t


def test_optional_label_permutation_preserves_closed_blocks():
    # Sanity D: conjugacy preserves closed-block count (isomorphism).
    events = fips_basic.EVENTS
    pi = {events[i]: events[(i + 3) % 84] for i in range(84)}
    conjugated = structural_control.label_permutation_third_map(pi)
    assert structural_control.closed_block_count(conjugated) == 56

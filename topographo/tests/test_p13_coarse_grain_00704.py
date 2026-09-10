"""CI-safe regression pins for the 007.04 p=13 Theory-41 coarse-grain probe (no torch).

Pins the published 007.04 numbers so the artifact stays reproducible from code,
and pins the tie discipline: ``numpy.argmax`` credits a tied maximum to the
lowest index, which would report result 11 as a winner 6648 times even though
results 2 and 11 carry identical effects.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.tlm_fixed_head import probe_p13_coarse_grain as probe
from topographo.ssd import fixed_head


def _effects():
    census = fixed_head.assert_census()
    basis = fixed_head.fixed_basis()
    e_c, n_ck, _ = probe.build_e_c(census, basis)
    return e_c, n_ck


def test_class_count_rows_collide_on_two_residue_pairs():
    _, n_ck = _effects()
    n_unique, duplicates = probe.duplicate_row_groups(n_ck)
    assert n_unique == 11
    assert duplicates == [[1, 12], [2, 11]]
    # Every result pools exactly p words.
    assert [int(v) for v in n_ck.sum(axis=1)] == [probe.P] * probe.P


def test_effects_stay_inside_the_four_dimensional_fixed_image():
    e_c, _ = _effects()
    assert probe.span_rank(e_c) == fixed_head.EXPECTED_SPAN_DIM == 4
    sv = probe.singular_values(e_c)
    assert sv[0] == pytest.approx(0.026669646929747694, rel=1e-12)
    # Rank 4: everything past the fourth singular value is numerical zero.
    assert max(sv[4:]) < 1e-15


def test_colliding_results_have_identical_effects():
    e_c, _ = _effects()
    dist = probe.pairwise_distances(e_c)
    assert dist["near_duplicates"] == 2
    assert dist["min"] == 0.0
    np.testing.assert_allclose(e_c[2], e_c[11], atol=1e-15)
    np.testing.assert_allclose(e_c[1], e_c[12], atol=1e-15)


def test_strict_argmax_survey_matches_published_00704_numbers():
    e_c, _ = _effects()
    strict, tie_sets, spreads = probe.preparation_argmax_survey(e_c)
    assert dict(strict) == {3: 519, 6: 1595, 7: 713, 10: 10525}
    assert sum(tie_sets.values()) == probe.N_PREPARATIONS
    assert sum(strict.values()) / probe.N_PREPARATIONS == 0.6676
    # The tie basin is the {2, 11} collision, and it is the second largest cell.
    assert tie_sets.most_common(2)[1] == ((2, 11), 6648)
    spread = np.asarray(spreads)
    assert spread.mean() == pytest.approx(0.0001992212746182828, rel=1e-12)


def test_tie_breaking_convention_invents_a_winner():
    """Guard the 017.24 lesson: argsort and argmax disagree on the {2, 11} tie.

    Neither is a real winner. ``argsort(...)[-1]`` credits the highest tied
    index (11), plain ``argmax`` credits the lowest (2), and both report 6648
    wins for a result that never strictly maximizes anything.
    """
    e_c, _ = _effects()
    rng = np.random.default_rng(probe.PREPARATION_SEED)
    by_argsort: dict[int, int] = {}
    by_argmax: dict[int, int] = {}
    for _ in range(probe.N_PREPARATIONS):
        x = rng.normal(size=probe.DIM)
        x = x / np.linalg.norm(x)
        scores = np.array([probe.readout(x, e) for e in e_c])
        top_argsort = int(np.argsort(scores)[-1])
        by_argsort[top_argsort] = by_argsort.get(top_argsort, 0) + 1
        top_argmax = int(np.argmax(scores))
        by_argmax[top_argmax] = by_argmax.get(top_argmax, 0) + 1
    assert by_argsort.get(11) == 6648
    assert by_argmax.get(2) == 6648
    assert 2 not in by_argsort
    assert 11 not in by_argmax

    strict, _, _ = probe.preparation_argmax_survey(e_c)
    assert 2 not in strict
    assert 11 not in strict


def test_verdict_is_negative():
    report = probe.build_report()
    assert report["verdict"]["solves_p13_ot_native_modular_interface"] is False
    assert report["p"] == 13
    assert report["fraction_unique_argmax"] == 0.6676

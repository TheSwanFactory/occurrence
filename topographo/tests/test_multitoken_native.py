"""CI-safe exact-path tests for the Issue-008 strict native constructors.

Pins the three `017.04` constructors on the exact rational path, the ill-typed
vs undefined split, projective scale invariance, and the Theory-065 consequence
that ``Sand`` degenerates on the admissible FIPS locus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from experiments.tlm_multitoken.native import (
    cyc,
    occ,
    rays_equal,
    sand,
    sand_is_identity_on_edges,
)
from topographo.ssd import exact, fips_basic
from topographo.ssd.frames import is_edge, is_event, ray


def _exact_eval_017():
    """Import the 017 exact evaluator, or skip.

    Two obstacles, neither of which justifies touching Issue 017 now that it is
    closure-ready: the ladder scripts import each other by bare module name, so
    their directory must be on the path, and the module imports ``torch``, which
    default CI deliberately does not install. The cross-check therefore runs
    locally and skips in CI; every other test in this file is self-contained.
    """
    tlm_fixed_head = Path(__file__).resolve().parents[2] / "experiments" / "tlm_fixed_head"
    if str(tlm_fixed_head) not in sys.path:
        sys.path.insert(0, str(tlm_fixed_head))
    try:
        import learned_admissibility_01719_exact_eval as evaluator
    except ImportError as error:  # pragma: no cover - depends on the environment
        pytest.skip(f"017 ladder evaluator unavailable: {error}")
    return evaluator


def _event(index: int) -> exact.Value:
    return fips_basic.EVENTS[index]


def _admissible_pair() -> tuple[exact.Value, exact.Value]:
    a, b, _ = fips_basic.ORDERED_EDGES[0]
    return a, b


def _inadmissible_pair() -> tuple[exact.Value, exact.Value]:
    for a in fips_basic.EVENTS[:12]:
        for b in fips_basic.EVENTS[:12]:
            if a != b and not fips_basic.admissible(a, b):
                return a, b
    raise AssertionError("no inadmissible pair among the first Events")


# --- typing boundary -------------------------------------------------------


def test_event_slot_refuses_a_non_event():
    r = exact.basis(4)
    assert not is_event(ray(r))
    with pytest.raises(TypeError, match="must be a certified Event ray"):
        occ(r, _event(0))
    with pytest.raises(TypeError, match="must be a certified Event ray"):
        sand(r, _event(0))
    with pytest.raises(TypeError, match="must be a certified Event ray"):
        cyc(r, _event(0))


def test_zero_is_ill_typed_not_undefined():
    with pytest.raises(TypeError, match="nonzero ray"):
        occ(exact.zero(), exact.basis(4))
    with pytest.raises(TypeError, match="nonzero ray"):
        occ(_event(0), exact.zero())
    with pytest.raises(TypeError, match="nonzero ray"):
        sand(_event(0), exact.zero())


def test_retained_slot_accepts_any_nonzero_ray():
    """A retained argument need not be an Event; only the Event slot is role-typed."""
    e = _event(0)
    assert not is_event(ray(exact.one()))
    assert occ(e, exact.one()) is not None
    assert sand(e, exact.one()) is not None


# --- Occ agrees with the certified 017 exact path --------------------------


def test_occ_matches_the_017_exact_evaluator():
    evaluator = _exact_eval_017()
    r = exact.basis(4)
    checked = 0
    for i in range(24):
        e = _event(i)
        mine = occ(e, r)
        theirs = evaluator.exact_occ(e, r)
        if theirs is None:
            assert mine is None
        else:
            assert mine is not None
            assert rays_equal(mine, theirs)
            checked += 1
    assert checked > 0


def test_p_seq_and_p_grp_rebuild_from_the_constructors():
    """The 017 two-program family is Occ/Cyc composition, nothing more."""
    evaluator = _exact_eval_017()
    r = exact.basis(4)
    seen_seq = seen_grp = 0
    for a, b, _ in fips_basic.ORDERED_EDGES[:40]:
        inner = occ(b, r)
        mine_seq = occ(a, inner) if inner is not None else None
        theirs_seq = evaluator.exact_p_seq(a, b, r)
        assert (mine_seq is None) == (theirs_seq is None)
        if theirs_seq is not None:
            assert rays_equal(mine_seq, theirs_seq)
            seen_seq += 1

        third = cyc(a, b)
        mine_grp = occ(third, r) if third is not None else None
        theirs_grp = evaluator.exact_p_grp(a, b, r)
        assert (mine_grp is None) == (theirs_grp is None)
        if theirs_grp is not None:
            assert rays_equal(mine_grp, theirs_grp)
            seen_grp += 1
    assert seen_seq > 0 and seen_grp > 0


# --- Cyc -------------------------------------------------------------------


def test_cyc_is_the_forced_third_and_equals_the_product_ray():
    a, b = _admissible_pair()
    third = cyc(a, b)
    assert third is not None
    assert third == fips_basic.third(a, b)
    assert is_event(third)
    assert rays_equal(third, exact.mul(a, b))


def test_cyc_is_undefined_off_the_admissible_domain():
    a, b = _inadmissible_pair()
    assert cyc(a, b) is None


def test_cyc_domain_size_is_the_certified_336():
    assert sum(1 for _ in fips_basic.ORDERED_EDGES) == 336
    defined = sum(1 for a, b, _ in fips_basic.ORDERED_EDGES if cyc(a, b) is not None)
    assert defined == 336


# --- Sand ------------------------------------------------------------------


def test_sand_reproduces_the_float_probe_witness():
    """probe_futurator_program_space.py section 3.2: Occ(z,1) and Sand(z,1) differ."""
    z = ray(exact.add(exact.basis(1), exact.basis(10)))
    assert is_event(z)
    one = exact.one()
    occ_z1 = occ(z, one)
    sand_z1 = sand(z, one)
    assert occ_z1 is not None and sand_z1 is not None
    assert not rays_equal(occ_z1, sand_z1)


def test_sand_is_projectively_scale_invariant():
    z = ray(exact.add(exact.basis(1), exact.basis(10)))
    r = exact.add(exact.basis(4), exact.basis(15))
    base = sand(z, r)
    assert base is not None
    for sz, sr in [(2, 1), (1, -3), (-1, -1), (7, 5), (1, 1)]:
        out = sand(exact.scale(sz, z), exact.scale(sr, r))
        assert out is not None
        assert rays_equal(base, out), (sz, sr)


def test_sand_degenerates_to_the_retained_ray_on_every_admissible_edge():
    """Theory-065 edge identity (e*r)*e = 2||e||^2 r forces Sand = [r].

    Checked on all 336 certified ordered pairs, not a sample: a ``Sand`` node
    sitting on the Cyc domain carries no endpoint information at all.
    """
    checked = 0
    for a, b, _ in fips_basic.ORDERED_EDGES:
        assert is_edge(a, b)
        assert sand_is_identity_on_edges(a, b), (a, b)
        checked += 1
    assert checked == 336


def test_sand_census_over_all_ordered_event_pairs():
    """Full 84x84 census. Sizes what a Sand node can contribute to the 008 space.

    Undefinedness is reported as ``None``, never raised: annihilation is data
    under `008.01` section 7.
    """
    census = {
        "pairs": 0,
        "edges": 0,
        "defined": 0,
        "undefined": 0,
        "identity": 0,
        "nontrivial": 0,
        "equal_to_occ": 0,
    }
    for a in fips_basic.EVENTS:
        for b in fips_basic.EVENTS:
            census["pairs"] += 1
            census["edges"] += int(is_edge(a, b))
            result = sand(a, b)
            if result is None:
                census["undefined"] += 1
                continue
            census["defined"] += 1
            if rays_equal(result, ray(b)):
                census["identity"] += 1
            else:
                census["nontrivial"] += 1
            if rays_equal(result, occ(a, b)):
                census["equal_to_occ"] += 1

    assert census == {
        "pairs": 7056,
        "edges": 336,
        "defined": 6720,
        "undefined": 336,
        # 336 admissible edges + 672 further degenerate non-edge pairs.
        "identity": 1008,
        "nontrivial": 5712,
        # Sand is never projectively Occ on Event x Event: a genuine constructor.
        "equal_to_occ": 0,
    }


def test_occ_and_sand_separate_on_every_event_at_the_017_retained_context():
    """At r=e4 both are total on the 84 Events and disagree on all of them."""
    r = exact.basis(4)
    both = differ = 0
    for a in fips_basic.EVENTS:
        o, s = occ(a, r), sand(a, r)
        assert o is not None and s is not None
        both += 1
        differ += int(not rays_equal(o, s))
    assert both == 84
    assert differ == 84


# --- shared ----------------------------------------------------------------


def test_rays_equal_never_credits_undefined():
    e = _event(0)
    assert not rays_equal(None, None)
    assert not rays_equal(occ(e, exact.one()), None)
    assert not rays_equal(None, occ(e, exact.one()))

"""Behavioral tests for the Theory 068.02 Operational-Frame machine."""

from dataclasses import fields
from fractions import Fraction

import pytest

from topographo.ssd import exact
from topographo.ssd.frames import (
    Cyclic,
    Native,
    OperationalFrame,
    action,
    advance,
    is_edge,
    is_event,
    ray,
    round_step,
    seed_guard,
    step,
)


def basic_events():
    """Finite 84-point pure Event design used by the 068 audit."""
    return tuple(
        ray(exact.add(exact.basis(i), exact.scale(s, exact.basis(8 + j))))
        for i in range(1, 8)
        for j in range(1, 8)
        if i != j
        for s in (-1, 1)
    )


def v(i, j, sign=1):
    return ray(exact.add(exact.basis(i), exact.scale(sign, exact.basis(j))))


def test_exact_sce_witness():
    a, b, c = v(1, 10), v(4, 15), v(5, 14, -1)
    pair = OperationalFrame(Native(a)), OperationalFrame(Native(b))
    expected = {
        "AB": (Cyclic(b, c), Cyclic(c, a)),
        "BA": (Cyclic(c, b), Cyclic(a, c)),
        "snapshot_AB": (Cyclic(b, c), Cyclic(a, c)),
        "snapshot_BA": (Cyclic(b, c), Cyclic(a, c)),
    }
    for policy, states in expected.items():
        assert tuple(f.retained_state for f in round_step(pair, policy).endpoint) == states


def test_guard_vs_independent_output_eigenspace():
    # Source orthogonality and output eigenspace are independently executable
    # formulations of Outcome 15's pure cyclic-stratum identity.
    events = basic_events()
    states = events + tuple(exact.basis(i) for i in range(16))
    for z in events:
        for x in states:
            y = action(z, x)
            spectral = (
                y is not None
                and exact.mul(exact.mul(z, y), z) == exact.scale(2 * exact.norm2(z), y)
            )
            assert seed_guard(z, x) == spectral


def test_invalid_edge_and_zero_have_no_operative_state():
    a = v(1, 10)
    with pytest.raises(ValueError):
        Native(exact.zero())
    with pytest.raises(ValueError):
        Cyclic(a, a)
    d = step(OperationalFrame(Native(v(4, 15, -1))), a)
    assert d.status == "undefined_projective"
    assert d.after is None


def test_non_event_state_does_not_synthesize_event():
    f = OperationalFrame(Native(exact.one()))
    assert f.presentation_interface() is None
    d = step(OperationalFrame(Native(v(1, 10))), f.presentation_interface())
    assert d.status == "presentation_unavailable"
    assert d.after is None


def test_law_is_explicit_not_selected_by_guard():
    a, b = v(1, 10), v(4, 15)
    founding = step(OperationalFrame(Native(a), "founding"), b)
    sce = step(OperationalFrame(Native(a), "sce"), b)
    assert founding.guard and sce.guard
    assert isinstance(founding.after.retained_state, Native)
    assert isinstance(sce.after.retained_state, Cyclic)


def test_cyclic_continuation_does_not_consume_peer():
    f = OperationalFrame(Cyclic(v(1, 10), v(4, 15)))
    assert step(f, None) == step(f, v(2, 11))
    assert advance(advance(advance(f.retained_state))) == f.retained_state
    assert [x.name for x in fields(f)] == ["retained_state", "enacted_transition"]


def test_antecedent_kernel_information_is_not_retained():
    z, x, k = v(1, 10), v(4, 15), v(4, 15, -1)
    expected = step(OperationalFrame(Native(x)), z).after
    for coefficient in (-5, -1, 0, Fraction(2, 3), 7):
        state = exact.add(x, exact.scale(coefficient, k))
        assert step(OperationalFrame(Native(state)), z).after == expected


def test_is_event_and_is_edge_helpers():
    assert is_event(v(1, 10))
    assert not is_event(exact.one())
    assert is_edge(v(1, 10), v(4, 15))

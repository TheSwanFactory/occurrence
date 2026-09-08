"""Theory-39 seal / classify / resolve witness (005.04 §1.3 S1–S3)."""

from __future__ import annotations

import pytest

from topographo.ssd import exact
from topographo.ssd.outcome_runtime import (
    Admissibility,
    NonAdmission,
    Presentation,
    ProjectiveNativeLeftAction,
    ask,
)
from topographo.ssd.seal import (
    EventDenotation,
    PublicEventView,
    classify,
    collision_witness,
    evaluate_left,
    resolve,
    seal,
)


def test_s1_collision_same_class_different_evaluation():
    witness = collision_witness()
    assert classify(witness.left) == classify(witness.right)
    assert resolve(witness.left) != resolve(witness.right)
    # Distinct Event actions on a shared state
    state = exact.one()
    left_ev = evaluate_left(witness.left, state)
    right_ev = evaluate_left(witness.right, state)
    assert left_ev != right_ev
    assert "Theory 39" in witness.provenance


def test_s2_classify_only_cannot_execute_event_action():
    witness = collision_witness()
    view = classify(witness.left)
    assert isinstance(view, PublicEventView)
    with pytest.raises(TypeError):
        resolve(view)  # type: ignore[arg-type]
    outcome = ask(
        Presentation((view, exact.one())),
        Admissibility(True),
        ProjectiveNativeLeftAction(),
    )
    assert isinstance(outcome, NonAdmission)


def test_s3_seal_and_resolve_recover_distinct_evaluations():
    witness = collision_witness()
    a = resolve(seal(resolve(witness.left), binding="re-a"))
    b = resolve(seal(resolve(witness.right), binding="re-b"))
    assert a == resolve(witness.left)
    assert b == resolve(witness.right)
    assert a != b
    assert isinstance(witness.left, EventDenotation)
    assert isinstance(witness.right, EventDenotation)

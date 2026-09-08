"""Acceptance tests for Outcome-aware Futurator ask (005.04 §1.1)."""

from __future__ import annotations

import ast
from pathlib import Path

from topographo.ssd import exact, machine, projective
from topographo.ssd.frames import ray
from topographo.ssd.outcome_runtime import (
    Admissibility,
    AnnihilationBoundary,
    ConstitutedOutcome,
    FipsClosure,
    FipsThird,
    NativeLeftAction,
    NonAdmission,
    OperationDomainInadmissible,
    Presentation,
    ProjectiveNativeLeftAction,
    RetainedRay,
    ask,
)
from topographo.ssd.seal import PublicEventView, classify, seal

ROOT = Path(__file__).resolve().parents[1]


def _event(i=1, j=10, sign=1):
    return ray(exact.add(exact.basis(i), exact.scale(sign, exact.basis(j))))


def test_b1_projective_ask_yields_constituted_retained_ray():
    z = seal(_event(1, 10))
    x = ray(exact.add(exact.basis(2), exact.basis(11)))
    # Prefer a state that does not annihilate under z.
    x = exact.one()
    outcome = ask(
        Presentation((z, x)),
        Admissibility(True),
        ProjectiveNativeLeftAction(),
    )
    assert isinstance(outcome, ConstitutedOutcome)
    assert isinstance(outcome.consequence, RetainedRay)
    assert not hasattr(outcome, "is_outcome")
    assert "is_outcome" not in outcome.__dataclass_fields__
    expected = projective.run_projective(x, [z.ray])
    assert isinstance(expected, machine.Completed)
    assert outcome.consequence.state == expected.state
    assert outcome.execution == expected


def test_b2_raw_and_projective_laws_differ_on_zero_product():
    z = seal(_event(1, 10))
    x = ray(exact.sub(exact.basis(4), exact.basis(15)))  # zx = 0
    assert exact.mul(z.ray, x) == exact.zero()
    raw = ask(Presentation((z, x)), Admissibility(True), NativeLeftAction())
    proj = ask(Presentation((z, x)), Admissibility(True), ProjectiveNativeLeftAction())
    assert isinstance(raw, ConstitutedOutcome)
    assert isinstance(raw.consequence, RetainedRay)
    assert raw.consequence.state == exact.zero()
    assert isinstance(proj, ConstitutedOutcome)
    assert isinstance(proj.consequence, AnnihilationBoundary)


def test_c1_post_state_is_policy_post_state():
    z = seal(_event(1, 10))
    x = exact.basis(4)
    outcome = ask(Presentation((z, x)), Admissibility(True), ProjectiveNativeLeftAction())
    assert isinstance(outcome, ConstitutedOutcome)
    assert isinstance(outcome.execution, machine.Completed)
    assert isinstance(outcome.consequence, RetainedRay)
    assert outcome.consequence.state == outcome.execution.state


def test_n1_zx_zero_is_annihilation_boundary_not_nonadmission():
    z = seal(_event(1, 10))
    x = ray(exact.sub(exact.basis(4), exact.basis(15)))
    outcome = ask(Presentation((z, x)), Admissibility(True), ProjectiveNativeLeftAction())
    assert isinstance(outcome, ConstitutedOutcome)
    assert isinstance(outcome.consequence, AnnihilationBoundary)
    assert not isinstance(outcome, NonAdmission)


def test_h0_outcome_runtime_has_no_optimizer_loss_imports():
    path = ROOT / "ssd" / "outcome_runtime.py"
    tree = ast.parse(path.read_text())
    banned = {"torch", "torch.nn", "torch.optim", "optax", "jax.numpy", "tensorflow"}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
            names.add(node.module)
    assert names.isdisjoint(banned)
    text = path.read_text().lower()
    assert "optimizer" not in text
    assert "import torch" not in text
    assert "adam" not in text
    assert "crossentropy" not in text and "cross_entropy" not in text


def test_g4_fips_closure_returns_forced_third_and_optional_action():
    from topographo.ssd import fips_basic

    z_ray = fips_basic.EVENTS[0]
    w_ray = fips_basic.partners(z_ray)[0]
    z, w = seal(z_ray), seal(w_ray)
    third_only = ask(Presentation((z, w)), Admissibility(True), FipsClosure())
    assert isinstance(third_only, ConstitutedOutcome)
    assert isinstance(third_only.consequence, FipsThird)
    assert third_only.consequence.third.ray == fips_basic.third(z_ray, w_ray)
    assert third_only.consequence.retained is None

    state = exact.one()
    both = ask(Presentation((z, w, state)), Admissibility(True), FipsClosure())
    assert isinstance(both, ConstitutedOutcome)
    assert isinstance(both.consequence, FipsThird)
    assert both.consequence.retained is not None


def test_classify_only_cannot_act_via_ask():
    view = classify(seal(_event()))
    assert isinstance(view, PublicEventView)
    outcome = ask(
        Presentation((view, exact.one())),
        Admissibility(True),
        ProjectiveNativeLeftAction(),
    )
    assert isinstance(outcome, NonAdmission)


def test_uncertified_admissibility_is_nonadmission():
    z = seal(_event())
    outcome = ask(
        Presentation((z, exact.one())),
        Admissibility(False, reason="caller rejected"),
        ProjectiveNativeLeftAction(),
    )
    assert isinstance(outcome, NonAdmission)
    assert isinstance(outcome.reason, OperationDomainInadmissible)

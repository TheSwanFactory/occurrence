"""Mutual Operational-Frame machine (Theory 068.02).

Exact, local Outcome 15/18 operations and explicit scheduling probes.
Presentation is an experimental wiring choice: expose a Native Event ray,
or the current (second) participant of a Cyclic edge. No selector is inferred.
Caches are bounded, disposable arithmetic memoization, never operative state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from topographo.ssd import exact, projective
from topographo.ssd.machine import Annihilated

Ray = exact.Value
Law = Literal["founding", "sce"]
ray = projective.canonicalize


@lru_cache(maxsize=16384)
def is_event(x: Ray) -> bool:
    a, b = x[:8], x[8:]
    aa = sum(t * t for t in a)
    return (
        x[0] == x[8] == 0
        and aa > 0
        and aa == sum(t * t for t in b)
        and sum(u * v for u, v in zip(a, b)) == 0
    )


@lru_cache(maxsize=16384)
def action(z: Ray, x: Ray) -> Ray | None:
    """Delegate native execution and zero handling to the released engine."""
    result = projective.run_projective(x, [z])
    return None if isinstance(result, Annihilated) else result.state


@lru_cache(maxsize=16384)
def seed_guard(z: Ray, x: Ray) -> bool:
    """x in (H_z perp + H_z perp), excluding ker L_z. No square roots."""
    if not is_event(z):
        return False
    a = z[:8] + exact.zero()[:8]
    b = z[8:] + exact.zero()[:8]
    ab = exact.mul(a, b)[:8]
    basis = (exact.one()[:8], a[:8], b[:8], ab)
    return (
        all(sum(u * v for u, v in zip(leg, h)) == 0 for leg in (x[:8], x[8:]) for h in basis)
        and action(z, x) is not None
    )


@lru_cache(maxsize=16384)
def is_edge(r: Ray, s: Ray) -> bool:
    """Theory 065 scaled eigenspace equation: (r*s)*r = 2 ||r||² s."""
    return (
        is_event(r)
        and is_event(s)
        and exact.mul(exact.mul(r, s), r) == exact.scale(2 * exact.norm2(r), s)
    )


@dataclass(frozen=True)
class Native:
    x: Ray

    def __post_init__(self):
        object.__setattr__(self, "x", ray(self.x))


@dataclass(frozen=True)
class Cyclic:
    r: Ray
    s: Ray

    def __post_init__(self):
        object.__setattr__(self, "r", ray(self.r))
        object.__setattr__(self, "s", ray(self.s))
        if not is_edge(self.r, self.s):
            raise ValueError("not a certified cyclic edge")


State = Native | Cyclic


@dataclass(frozen=True)
class OperationalFrame:
    retained_state: State
    enacted_transition: Law = "sce"

    def __post_init__(self):
        if self.enacted_transition not in ("founding", "sce"):
            raise ValueError("unrecognized enacted law")

    def presentation_interface(self) -> Ray | None:
        state = self.retained_state
        if isinstance(state, Cyclic):
            return state.s
        return state.x if is_event(state.x) else None


@dataclass(frozen=True)
class Decision:
    before: OperationalFrame
    after: OperationalFrame | None
    presented: Ray | None
    status: str
    guard: bool | None
    source: str


@lru_cache(maxsize=16384)
def advance(edge: Cyclic) -> Cyclic:
    t = action(edge.r, edge.s)
    assert t is not None
    return Cyclic(edge.s, t)


def step(local: OperationalFrame, offered: Ray | None) -> Decision:
    state = local.retained_state
    if isinstance(state, Cyclic):
        return Decision(
            local,
            OperationalFrame(advance(state), local.enacted_transition),
            None,
            "cyclic",
            None,
            "local.retained_state.ordered_edge",
        )
    if offered is None:
        return Decision(
            local,
            None,
            None,
            "presentation_unavailable",
            None,
            "peer.presentation_interface",
        )
    z = ray(offered)
    if not is_event(z):
        raise ValueError("presented participant must be an Event")
    y = action(z, state.x)
    if y is None:
        return Decision(local, None, z, "undefined_projective", False, "peer.presentation_interface")
    guard = seed_guard(z, state.x)
    post = Cyclic(z, y) if local.enacted_transition == "sce" and guard else Native(y)
    status = "seeded" if isinstance(post, Cyclic) else "native"
    return Decision(
        local,
        OperationalFrame(post, local.enacted_transition),
        z,
        status,
        guard,
        "peer.presentation_interface",
    )


@dataclass(frozen=True)
class Round:
    decisions: tuple[Decision, Decision]  # observer slots A,B; not frame IDs

    @property
    def complete(self):
        return all(d.after is not None for d in self.decisions)

    @property
    def endpoint(self):
        return tuple(d.after for d in self.decisions) if self.complete else None


def round_step(pair: tuple[OperationalFrame, OperationalFrame], policy: str) -> Round:
    """Two attempted local actions. A failed action has no committed post-state.

    snapshot_AB/BA stage both immutable incoming frames before either action.
    They implement an explicit joint-read experiment, not a synchronization law.
    Serial failures leave the old frame accessible for the other diagnostic
    attempt, but the combined round is incomplete and is never iterated.
    """
    if policy not in ("AB", "BA", "snapshot_AB", "snapshot_BA"):
        raise ValueError(policy)
    work = list(pair)
    decisions = [None, None]
    for i in (0, 1) if policy.endswith("AB") else (1, 0):
        source = pair if policy.startswith("snapshot") else work
        offered = (
            None
            if isinstance(work[i].retained_state, Cyclic)
            else source[1 - i].presentation_interface()
        )
        decisions[i] = step(work[i], offered)
        if decisions[i].after is not None:
            work[i] = decisions[i].after
    return Round(tuple(decisions))


def encode_ray(x):
    return {f"e{i}": str(c) for i, c in enumerate(x) if c}


def encode_frame(frame):
    if frame is None:
        return None
    s = frame.retained_state
    state = (
        {"type": "Native", "x": encode_ray(s.x)}
        if isinstance(s, Native)
        else {"type": "Cyclic", "r": encode_ray(s.r), "s": encode_ray(s.s)}
    )
    return {"retained_state": state, "enacted_transition": frame.enacted_transition}


def encode_round(result):
    return {
        "complete": result.complete,
        "decisions": [
            {
                "observer_slot": slot,
                "before": encode_frame(d.before),
                "post_state": encode_frame(d.after),
                "presented": None if d.presented is None else encode_ray(d.presented),
                "status": d.status,
                "seed_guard": d.guard,
                "provenance": {
                    "presentation": d.source,
                    "admissibility": "local certified Event/domain/seed predicates on shown inputs",
                    "law": (
                        "local.enacted_transition"
                        if d.status != "cyclic"
                        else "local retained Cyclic type: Outcome 15 continuation"
                    ),
                    "post_state": "exact local left action / ordered-edge rotation on shown inputs",
                    "global_semantic_selector": False,
                },
            }
            for slot, d in zip(("A", "B"), result.decisions)
        ],
    }


def clear_caches():
    for f in (is_event, action, seed_guard, is_edge, advance):
        f.cache_clear()

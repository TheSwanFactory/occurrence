"""Matched non-isomorphic structural control for FIPS (005.07 §2).

Primary control C is a **degree-matched rewiring** of the basic FIPS third-map:
same 84 Events and the same 336 ordered admissible pairs (C1), same partner-degree
histogram (C2), but a deliberately destroyed cyclic/Pasch closure structure
(C3–C4). Pure label-permutation conjugacy is *not* this control; see
``label_permutation_third_map`` for optional equivariance sanity D.

Broken algebraic laws (C5) are documented in ``BROKEN_LAWS``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from topographo.ssd import fips_basic
from topographo.ssd.frames import ray

Ray = fips_basic.Ray


def _fips_third_map() -> dict[tuple[Ray, Ray], Ray]:
    return {(a, b): t for a, b, t in fips_basic.ORDERED_EDGES}



BROKEN_LAWS: tuple[str, ...] = (
    "cyclic block closure (six ordered edges recover a Steiner-like triple)",
    "Pasch/FIPS generative third = sedenion cyclic action on the certified edge",
    "compatibility of forced third with projective left action of the sedenion Event",
    "global Pasch component count of the basic FIPS design (fourteen components in the certified census)",
)


def _encode(value: Ray) -> str:
    return ",".join(f"{i}:{c}" for i, c in enumerate(value) if c)


@dataclass(frozen=True)
class StructuralControlTable:
    """Deterministic partial third-map with the same admissible domain as FIPS."""

    name: str
    events: tuple[Ray, ...]
    third_map: Mapping[tuple[Ray, Ray], Ray]
    seed: int

    @property
    def ordered_pairs(self) -> tuple[tuple[Ray, Ray], ...]:
        return tuple(sorted(self.third_map.keys(), key=lambda ab: (_encode(ab[0]), _encode(ab[1]))))

    def admissible(self, left: Ray, right: Ray) -> bool:
        return (ray(left), ray(right)) in self.third_map

    def third(self, left: Ray, right: Ray) -> Ray:
        key = (ray(left), ray(right))
        try:
            return self.third_map[key]
        except KeyError as exc:
            raise KeyError("pair not in structural-control domain") from exc

    def partners(self, event: Ray) -> tuple[Ray, ...]:
        event = ray(event)
        return tuple(sorted((w for (a, w) in self.third_map if a == event), key=_encode))


def degree_histogram(table: Mapping[tuple[Ray, Ray], Ray] | StructuralControlTable) -> Counter[int]:
    third_map = table.third_map if isinstance(table, StructuralControlTable) else table
    counts: Counter[Ray] = Counter()
    for a, _w in third_map:
        counts[a] += 1
    return Counter(counts.values())


def closed_block_count(third_map: Mapping[tuple[Ray, Ray], Ray]) -> int:
    """Count unordered triples with full 6-edge cyclic closure under ``third_map``."""

    blocks: set[tuple[Ray, Ray, Ray]] = set()
    for (a, b), t in third_map.items():
        cand = tuple(sorted((a, b, t)))
        members = set(cand)
        edges = 0
        for x in cand:
            for y in cand:
                if x == y:
                    continue
                if (x, y) in third_map and third_map[(x, y)] in members:
                    edges += 1
        if edges == 6:
            blocks.add(cand)
    return len(blocks)


def differing_third_count(
    left: Mapping[tuple[Ray, Ray], Ray],
    right: Mapping[tuple[Ray, Ray], Ray],
) -> int:
    shared = set(left) & set(right)
    return sum(1 for key in shared if left[key] != right[key])


def build_degree_matched_rewiring(*, seed: int = 0) -> StructuralControlTable:
    """Rewire FIPS thirds per left-event while preserving partner sets.

    For each left Event ``a`` with partners ``(w0..w3)`` and FIPS thirds
    ``(t0..t3)``, replace the third tuple by a deterministic derangement of
    ``(t0..t3)`` (a single cyclic shift by ``1 + seed % 3``). Domain pairs and
    out-degrees are unchanged; cyclic closure is destroyed.
    """

    shift = 1 + (seed % 3)
    new_map: dict[tuple[Ray, Ray], Ray] = {}
    for a in fips_basic.EVENTS:
        partners = fips_basic.partners(a)
        thirds = tuple(fips_basic.third(a, w) for w in partners)
        rewired = thirds[shift:] + thirds[:shift]
        for w, t in zip(partners, rewired, strict=True):
            new_map[(a, w)] = t
    return StructuralControlTable(
        name=f"degree_matched_rewiring_seed{seed}",
        events=fips_basic.EVENTS,
        third_map=new_map,
        seed=seed,
    )


def label_permutation_third_map(pi: Mapping[Ray, Ray]) -> dict[tuple[Ray, Ray], Ray]:
    """Optional equivariance sanity D: push-forward conjugacy of the third-map.

    For each admissible ``(a, b)``, define
    ``mu'(pi(a), pi(b)) = pi(mu(a, b))``.
    This is an isomorphic relabeling (not the primary structural ablation;
    see 005.07 section 2.1).
    """

    pi = {ray(k): ray(v) for k, v in pi.items()}
    if len(set(pi.values())) != len(pi):
        raise ValueError("pi must be a bijection on Events")
    conjugated: dict[tuple[Ray, Ray], Ray] = {}
    for (a, b), t in _fips_third_map().items():
        conjugated[(pi[a], pi[b])] = pi[t]
    return conjugated


@dataclass(frozen=True)
class ControlCertificates:
    c1_event_count: int
    c1_pair_count: int
    c2_degree_histogram: Counter[int]
    c2_matches_fips: bool
    c3_fips_closed_blocks: int
    c3_control_closed_blocks: int
    c3_differs: bool
    c4_differing_thirds: int
    c4_not_mere_relabeling: bool
    c5_broken_laws: tuple[str, ...]


def certify(control: StructuralControlTable) -> ControlCertificates:
    """Mechanical checks C1–C5 (fail-closed callers should assert fields)."""

    fips_map = _fips_third_map()
    fips_deg = degree_histogram(fips_map)
    ctrl_deg = degree_histogram(control)
    fips_blocks = closed_block_count(fips_map)
    ctrl_blocks = closed_block_count(control.third_map)
    differing = differing_third_count(fips_map, control.third_map)
    # Mere relabeling would preserve closed-block count at 56; invariant mismatch
    # is the cheap non-isomorphism certificate (005.07 C4).
    return ControlCertificates(
        c1_event_count=len(control.events),
        c1_pair_count=len(control.third_map),
        c2_degree_histogram=ctrl_deg,
        c2_matches_fips=ctrl_deg == fips_deg,
        c3_fips_closed_blocks=fips_blocks,
        c3_control_closed_blocks=ctrl_blocks,
        c3_differs=ctrl_blocks != fips_blocks,
        c4_differing_thirds=differing,
        c4_not_mere_relabeling=ctrl_blocks != fips_blocks and differing > 0,
        c5_broken_laws=BROKEN_LAWS,
    )


def assert_c1_c5(control: StructuralControlTable | None = None) -> ControlCertificates:
    control = control or build_degree_matched_rewiring(seed=0)
    cert = certify(control)
    if cert.c1_event_count != 84 or cert.c1_pair_count != 336:
        raise AssertionError(f"C1 failed: events={cert.c1_event_count} pairs={cert.c1_pair_count}")
    if not cert.c2_matches_fips:
        raise AssertionError(f"C2 failed: degree histogram {cert.c2_degree_histogram}")
    if not cert.c3_differs:
        raise AssertionError("C3 failed: closed-block count did not differ from FIPS")
    if not cert.c4_not_mere_relabeling:
        raise AssertionError("C4 failed: control looks like mere relabeling / identical thirds")
    if not cert.c5_broken_laws:
        raise AssertionError("C5 failed: broken laws not documented")
    return cert

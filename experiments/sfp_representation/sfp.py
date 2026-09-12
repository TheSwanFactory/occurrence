"""SFP core: habitat enumeration and the ``S | FFF | PP | pp`` codec (Issue 009).

This module coordinatizes the certified finite FIPS structure of
``topographo.ssd.fips_basic`` into the compact SFP typing pinned by Outcome
021.05, and then proves — over the complete finite domain — that the compact
code is an *exact* representation of the certified native relation.

Gate 0 (``experiments/sfp_representation/conformance.py``) has already PASSED:
84 Events, 56 unordered blocks, 112 oriented blocks, 168 incidences, 336
ordered edges, 14 habitats, ``|GL(3,2)| = 168``, ``|AGL(2,2)| = 24``. This
module builds on that and does not re-litigate it.

The SFP typing
--------------
::

    S     = habitat sign delta
    FFF   = nonzero Fano point P            (1..7)
    PP    = affine cyclic-block coordinate  q in F_2^2
    pp    = nonzero affine displacement d   (Fano-line class through P)

Constructive derivation (built and asserted here, never assumed)
----------------------------------------------------------------
Fix a certified habitat ``(P, delta)`` — the ``topographo.ssd.seal.public_label``
of its Events. It owns exactly 4 certified cyclic blocks and exactly 6 Events.

* **K4.** Blocks are vertices; Events are edges. Every Event has block-degree
  exactly 2, so it joins two distinct blocks. (4 vertices, 6 edges, degree 3.)
  A certified native block is therefore *the vertex*, and its 3 Events are the
  3 K4 edges through that vertex.
* **Fano lines as matchings.** The 3 Fano lines through ``P`` are the 3
  opposite-edge perfect matchings of that K4. Grouping the habitat's 6 Events
  by their line ``{i, j, i ^ j}`` (the classifier of
  ``experiments/mutual-frames/audit.py::label`` third slot, i.e.
  ``conformance.fano_line_of_event``) yields 3 classes of 2 disjoint Events
  covering all 4 blocks. Verified per habitat.
* **Torsor.** Each matching induces a permutation of the 4 blocks which is
  verified to be a fixed-point-free involution. The 3 involutions plus the
  identity are verified to form a group isomorphic to ``C2^2 ~= Q_P``, where
  ``Q_P = F_2^3 / <P>``, whose 3 nonzero cosets are exactly the 3 lines through
  ``P`` (a coset represented by ``x`` gives the line ``{P, x, P ^ x}``). The
  action on the 4 blocks is verified **free and transitive**, so the block set
  is a ``Q_P``-torsor. Choosing an origin gives 2-bit coordinates ``PP = q``.
* **Two presentations per Event.** A nonzero displacement ``d`` labels the Event
  joining ``q <-> q ^ d``, so the same Event has exactly two incidence
  presentations ``(q, d) ~ (q ^ d, d)``. Per habitat: ``4*3/2 = 6`` Events,
  ``4*3 = 12`` incidences, ``4*3*2 = 24`` ordered cyclic pairs. Times 14
  habitats: 84 / 168 / 336.

DECLARED NORMALIZATION (frozen chart) — read before using this module
---------------------------------------------------------------------
The origin block and the assignment of the three nonzero ``F_2^2`` values
``{01, 10, 11}`` to the three Fano-line classes are **basis choices, not
intrinsic structure**. They are frozen here as follows, deterministically and
without any dependence on set/dict iteration order or ``hash()``:

* **Origin block**: within a habitat, blocks are ordered by the stable textual
  ray encoding ``tuple(sorted(_encode_ray(e) for e in block))`` (the
  ``fips_basic._encode_ray`` convention); the lexicographic minimum is the
  origin and receives ``PP = 00``.
* **Line values**: the habitat's three Fano lines are sorted as integer triples
  and receive ``pp = 01, 10, 11`` in that order.
* **Sign bit**: ``S = 0`` for ``delta = +1``, ``S = 1`` for ``delta = -1``.

Any relabelling of these choices is an equally valid chart. Arm D will later
test relabelling invariance and must know exactly what was fixed: the artifact
records the full per-habitat chart plus a SHA-256 digest
(``chart.sha256``). Nothing downstream may treat the chart as canonical.

Fences
------
::

    FFF=000 and pp=00 are formal algebraic completion values only.
    They are NOT Events, Outcomes, retained states, annihilation, or non-admission.
    algebraic zero != NONADMISSION;   0 != bottom
    SFP is a representation/reference layer. It is NOT promoted to a replacement
    OT evaluator: it does not replace Occ, Cyc, Sand, or native Event denotation.
    This codec must never be used to generate ground-truth labels.

``ExactSfpCircuit`` is a **nonlearned reference oracle**. It exists to certify
that the compact code carries the certified relation exactly, so that a later
Arm-B success means representation *sufficiency/use* rather than representation
*discovery*. Ground-truth labels for any downstream experiment must come from
``topographo.ssd.fips_basic`` (``admissible`` / ``third``) and never from this
module — see ``REFERENCE_ORACLE_ONLY``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from topographo.ssd import fips_basic, projective
from topographo.ssd.frames import is_event, ray
from topographo.ssd.seal import public_label

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "sfp_codec.json"

REFERENCE_ORACLE_ONLY = (
    "ExactSfpCircuit is a nonlearned REFERENCE ORACLE. It must never be used to "
    "generate ground-truth labels; labels come from topographo.ssd.fips_basic "
    "(admissible / third), the certified native relation."
)

FENCES = (
    "FFF=000 and pp=00 are formal algebraic completion values only.",
    "They are NOT Events, Outcomes, retained states, annihilation, or non-admission.",
    "algebraic zero != NONADMISSION;   0 != bottom",
    "SFP is a representation/reference layer. It is NOT promoted to a replacement "
    "OT evaluator: it does not replace Occ, Cyc, Sand, or native Event denotation.",
    "This codec must never be used to generate ground-truth labels.",
)

DECLARED_CHART = (
    "origin block = lexicographic minimum of the habitat's blocks under the key "
    "tuple(sorted(_encode_ray(event) for event in block)); it receives PP=00",
    "pp values 01, 10, 11 are assigned to the habitat's three Fano lines in "
    "ascending sorted integer-triple order",
    "S = 0 for delta=+1, S = 1 for delta=-1",
    "these are BASIS CHOICES, not intrinsic structure; Arm D tests relabelling "
    "invariance against exactly this frozen chart",
)

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"
EXPECTED_TABLE_SHA256 = (
    "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea"
)

PINS = {
    "events": 84,
    "habitats": 14,
    "incidences": 168,
    "ordered_pairs": 336,
    "total_ordered_pairs": 84 * 84,
    "blocks_per_habitat": 4,
    "events_per_habitat": 6,
    "incidences_per_habitat": 12,
    "ordered_pairs_per_habitat": 24,
}

V2_NONZERO = (1, 2, 3)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def digest(obj: object) -> str:
    """Canonical sha256 of a JSON-serializable object."""

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def encode_ray(value: object) -> str:
    """Stable textual encoding of an exact ray (``fips_basic._encode_ray``)."""

    return ",".join(f"{i}:{c}" for i, c in enumerate(value) if c)


def block_key(block: tuple) -> tuple[str, ...]:
    """Stable, iteration-order-independent key for a certified cyclic block."""

    return tuple(sorted(encode_ray(e) for e in block))


def habitat_key(s: int, fff: int) -> str:
    return f"S={s},FFF={fff:03b}"


def bits2(value: int) -> str:
    return f"{value:02b}"


def fano_line_of_event(event: tuple) -> tuple[int, int, int]:
    """Fano line ``{i, j, i ^ j}`` of a basic Event ``e_i + s e_{8+j}``.

    Same classifier as ``experiments/mutual-frames/audit.py::label`` third slot
    and ``conformance.fano_line_of_event``.
    """

    i = next(k for k in range(1, 8) if event[k])
    j = next(k for k in range(1, 8) if event[8 + k])
    return tuple(sorted((i, j, i ^ j)))


def sign_bit(delta: int) -> int:
    """Declared chart: ``S = 0`` for ``delta=+1``, ``S = 1`` for ``delta=-1``."""

    if delta == 1:
        return 0
    if delta == -1:
        return 1
    raise AssertionError(f"unexpected habitat delta {delta!r}")


def delta_of_sign(s: int) -> int:
    if s == 0:
        return 1
    if s == 1:
        return -1
    raise AssertionError(f"unexpected sign bit {s!r}")


# ---------------------------------------------------------------------------
# addresses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidencePresentation:
    """One ``(S, FFF, PP, pp)`` incidence presentation of an Event.

    ``pp = d`` is always nonzero: ``pp = 00`` is a formal completion value only.
    """

    s: int
    fff: int
    q: int
    d: int

    def __post_init__(self) -> None:
        if not 0 <= self.s <= 1:
            raise ValueError("S must be a single bit")
        if not 1 <= self.fff <= 7:
            raise ValueError("FFF must be a nonzero Fano point (1..7)")
        if not 0 <= self.q <= 3:
            raise ValueError("PP must lie in F_2^2")
        if self.d not in V2_NONZERO:
            raise ValueError("pp must be a NONZERO displacement; pp=00 is formal only")

    @property
    def other_end(self) -> int:
        """The peer block coordinate of the same Event: ``q XOR d``."""

        return self.q ^ self.d

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.s, self.fff, self.q, self.d)


@dataclass(frozen=True)
class SfpAddress:
    """Canonical Event-level SFP address.

    ``presentations`` holds the UNORDERED pair ``{(q, d), (q XOR d, d)}`` as a
    sorted 2-tuple. No endpoint is privileged at this layer: Arm B's validity
    depends on endpoint symmetry remaining available downstream.
    """

    s: int
    fff: int
    presentations: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        if not 0 <= self.s <= 1:
            raise ValueError("S must be a single bit")
        if not 1 <= self.fff <= 7:
            raise ValueError("FFF must be a nonzero Fano point (1..7)")
        if len(self.presentations) != 2:
            raise ValueError("an Event has exactly two incidence presentations")
        if tuple(sorted(self.presentations)) != tuple(self.presentations):
            raise ValueError("presentations must be held as a sorted unordered pair")
        (q1, d1), (q2, d2) = self.presentations
        if d1 != d2 or d1 not in V2_NONZERO:
            raise ValueError("both presentations share one nonzero displacement")
        if q2 != q1 ^ d1 or q1 == q2:
            raise ValueError("presentations must be (q, d) and (q XOR d, d)")

    @property
    def d(self) -> int:
        """The shared nonzero displacement (Fano-line class through ``FFF``)."""

        return self.presentations[0][1]

    @property
    def habitat(self) -> tuple[int, int]:
        return (self.s, self.fff)

    @property
    def endpoints(self) -> frozenset[int]:
        """The Event's two K4 endpoint block coordinates. Unordered."""

        return frozenset(q for q, _ in self.presentations)

    def incidences(self) -> tuple[IncidencePresentation, IncidencePresentation]:
        """Both ``(S, FFF, PP, pp)`` views, in the same unordered sorted order."""

        return tuple(
            IncidencePresentation(self.s, self.fff, q, d)
            for q, d in self.presentations
        )

    def as_json(self) -> dict:
        return {
            "S": self.s,
            "FFF": f"{self.fff:03b}",
            "presentations": [
                {"PP": bits2(q), "pp": bits2(d)} for q, d in self.presentations
            ],
        }

    def as_key(self) -> tuple:
        return (self.s, self.fff, self.presentations)


def address_of(s: int, fff: int, q: int, d: int) -> SfpAddress:
    """Build the canonical Event-level address containing ``(q, d)``."""

    if d not in V2_NONZERO:
        raise ValueError("pp must be nonzero; pp=00 is a formal completion value only")
    pair = tuple(sorted(((q, d), (q ^ d, d))))
    return SfpAddress(s=s, fff=fff, presentations=pair)


# ---------------------------------------------------------------------------
# habitat construction: K4 -> matchings -> C2^2 torsor -> chart
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Habitat:
    """One certified ``(P, delta)`` habitat, coordinatized under the frozen chart."""

    s: int
    fff: int
    delta: int
    blocks_by_q: tuple[tuple, ...]              # q (0..3) -> certified block
    q_of_block_key: dict[tuple[str, ...], int]
    d_of_line: dict[tuple[int, int, int], int]
    line_of_d: dict[int, tuple[int, int, int]]
    event_of_incidence: dict[tuple[int, int], tuple]   # (q, d) -> Event ray
    presentations_of_event: dict[tuple, tuple[tuple[int, int], tuple[int, int]]]
    certification: dict

    @property
    def key(self) -> str:
        return habitat_key(self.s, self.fff)


def _matching_permutation(
    events: list[tuple],
    blocks_by_q: tuple[tuple, ...],
    q_of_block_key: dict[tuple[str, ...], int],
) -> dict[int, int]:
    """Permutation of block coordinates induced by an opposite-edge matching."""

    perm: dict[int, int] = {}
    for event in events:
        ends = sorted(
            q_of_block_key[block_key(b)] for b in blocks_by_q if event in b
        )
        if len(ends) != 2:
            raise AssertionError("Event block-degree is not 2 inside its habitat")
        a, b = ends
        perm[a] = b
        perm[b] = a
    return perm


def _identity_perm() -> dict[int, int]:
    return {q: q for q in range(4)}


def _compose(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    return {q: left[right[q]] for q in range(4)}


def build_habitat(label: tuple[int, int], blocks: tuple[tuple, ...]) -> Habitat:
    """Construct one habitat's torsor + chart, asserting every structural claim."""

    fff, delta = label
    s = sign_bit(delta)
    if not 1 <= fff <= 7:
        raise AssertionError(f"habitat Fano point {fff} is not a nonzero point")

    # --- K4: 4 block vertices, 6 Event edges, every Event of block-degree 2 ---
    ordered_blocks = tuple(sorted(blocks, key=block_key))
    if len(ordered_blocks) != PINS["blocks_per_habitat"]:
        raise AssertionError(f"habitat {label} has {len(ordered_blocks)} blocks, not 4")
    events = sorted(
        {e for block in ordered_blocks for e in block}, key=encode_ray
    )
    if len(events) != PINS["events_per_habitat"]:
        raise AssertionError(f"habitat {label} has {len(events)} Events, not 6")
    if not all(is_event(e) for e in events):
        raise AssertionError("habitat contains a non-Event ray")
    if {public_label(e) for e in events} != {label}:
        raise AssertionError("habitat Events do not share one (P, delta) label")

    # blocks are provisionally indexed by their sorted position; the chart below
    # fixes q = 0 on index 0 (the lexicographic minimum = declared origin).
    provisional = {block_key(b): i for i, b in enumerate(ordered_blocks)}
    degrees = {
        encode_ray(e): sum(1 for b in ordered_blocks if e in b) for e in events
    }
    if sorted(set(degrees.values())) != [2]:
        raise AssertionError(f"habitat {label} block-degree spectrum {degrees}")
    incident = {
        encode_ray(e): sorted(provisional[block_key(b)] for b in ordered_blocks if e in b)
        for e in events
    }
    if any(len(set(v)) != 2 for v in incident.values()):
        raise AssertionError("an Event does not join two distinct blocks")

    # --- 3 Fano lines through P as opposite-edge perfect matchings ---
    classes: dict[tuple[int, int, int], list[tuple]] = defaultdict(list)
    for e in events:
        classes[fano_line_of_event(e)].append(e)
    lines = sorted(classes)
    if len(lines) != 3:
        raise AssertionError(f"habitat {label} has {len(lines)} line classes, not 3")
    for line in lines:
        if fff not in line:
            raise AssertionError(f"line {line} does not pass through P={fff}")
        members = sorted(classes[line], key=encode_ray)
        if len(members) != 2:
            raise AssertionError(f"line class {line} is not a pair")
        covered = [incident[encode_ray(e)] for e in members]
        if sorted(covered[0] + covered[1]) != [0, 1, 2, 3]:
            raise AssertionError(f"line class {line} is not a perfect matching")

    # Q_P = F_2^3 / <P>: its 3 nonzero cosets are exactly these 3 lines.
    cosets = {frozenset((x, x ^ fff)) for x in range(1, 8) if x != fff}
    coset_lines = {tuple(sorted((fff, x, fff ^ x))) for c in cosets for x in c}
    if coset_lines != set(lines):
        raise AssertionError("Fano lines through P are not the nonzero cosets of <P>")

    # --- frozen chart: origin = index 0, pp values in sorted line order ---
    d_of_line = {line: V2_NONZERO[i] for i, line in enumerate(lines)}
    line_of_d = {d: line for line, d in d_of_line.items()}

    perms = {
        d: _matching_permutation(
            sorted(classes[line_of_d[d]], key=encode_ray), ordered_blocks, provisional
        )
        for d in V2_NONZERO
    }

    # --- involution + fixed-point-freeness, asserted ---
    for d, perm in perms.items():
        if sorted(perm) != [0, 1, 2, 3] or sorted(perm.values()) != [0, 1, 2, 3]:
            raise AssertionError(f"matching {d} is not a permutation of the 4 blocks")
        if any(perm[q] == q for q in range(4)):
            raise AssertionError(f"matching {d} has a fixed point")
        if _compose(perm, perm) != _identity_perm():
            raise AssertionError(f"matching {d} is not an involution")

    # --- {id, m1, m2, m3} ~= C2^2, acting freely and transitively ---
    group = {0: _identity_perm(), **perms}
    frozen_group = {
        tuple(g[q] for q in range(4)) for g in group.values()
    }
    if len(frozen_group) != 4:
        raise AssertionError("the matching group does not have 4 distinct elements")
    closed = True
    homomorphic = True
    for a, b in itertools.product(range(4), repeat=2):
        product = _compose(group[a], group[b])
        if tuple(product[q] for q in range(4)) not in frozen_group:
            closed = False
        if product != group[a ^ b]:
            homomorphic = False
    abelian = all(
        _compose(group[a], group[b]) == _compose(group[b], group[a])
        for a, b in itertools.combinations(range(4), 2)
    )
    exponent_two = all(
        _compose(group[a], group[a]) == _identity_perm() for a in range(4)
    )
    if not (closed and homomorphic and abelian and exponent_two):
        raise AssertionError(
            f"habitat {label}: matching group is not C2^2 "
            f"(closed={closed}, homomorphic={homomorphic}, abelian={abelian}, "
            f"exponent_two={exponent_two})"
        )

    # free: only the identity fixes anything; transitive: orbit of origin is all 4.
    free = all(
        all(group[a][q] != q for q in range(4)) for a in V2_NONZERO
    )
    orbit = {group[a][0] for a in range(4)}
    transitive = orbit == {0, 1, 2, 3}
    if not (free and transitive):
        raise AssertionError(
            f"habitat {label}: block action is not a torsor "
            f"(free={free}, transitive={transitive})"
        )

    # --- coordinates: q(origin)=00, q(m_d(origin))=d ---
    q_of_index = {group[a][0]: a for a in range(4)}
    if sorted(q_of_index) != [0, 1, 2, 3] or sorted(q_of_index.values()) != [0, 1, 2, 3]:
        raise AssertionError("chart coordinates are not a bijection onto F_2^2")
    if q_of_index[0] != 0:
        raise AssertionError("declared origin did not receive PP=00")
    # chart coherence: the matching m_d translates coordinates by XOR d.
    for d, perm in perms.items():
        for index in range(4):
            if q_of_index[perm[index]] != q_of_index[index] ^ d:
                raise AssertionError("chart is not Q_P-equivariant under XOR")

    blocks_by_q = tuple(
        ordered_blocks[next(i for i in range(4) if q_of_index[i] == q)]
        for q in range(4)
    )
    q_of_block_key = {block_key(b): q for q, b in enumerate(blocks_by_q)}

    # --- (q, d) <-> Event, with both presentations of every Event ---
    event_of_incidence: dict[tuple[int, int], tuple] = {}
    presentations_of_event: dict[tuple, tuple] = {}
    for d in V2_NONZERO:
        for e in sorted(classes[line_of_d[d]], key=encode_ray):
            ends = sorted(q_of_block_key[block_key(b)] for b in blocks_by_q if e in b)
            q1, q2 = ends
            if q2 != q1 ^ d:
                raise AssertionError("Event endpoints do not differ by its own pp")
            pair = ((q1, d), (q2, d))
            presentations_of_event[e] = pair
            for q, dd in pair:
                if (q, dd) in event_of_incidence:
                    raise AssertionError("two Events claim one (PP, pp) incidence")
                event_of_incidence[(q, dd)] = e
    if len(event_of_incidence) != PINS["incidences_per_habitat"]:
        raise AssertionError(
            f"habitat {label} produced {len(event_of_incidence)} incidences, not 12"
        )
    if len(presentations_of_event) != PINS["events_per_habitat"]:
        raise AssertionError("habitat presentation map does not cover 6 Events")

    certification = {
        "blocks": PINS["blocks_per_habitat"],
        "events": PINS["events_per_habitat"],
        "incidences": len(event_of_incidence),
        "block_degree_values": sorted(set(degrees.values())),
        "line_classes": len(lines),
        "all_classes_are_perfect_matchings": True,
        "all_matchings_fixed_point_free_involutions": True,
        "group_order": len(frozen_group),
        "group_closed": closed,
        "group_isomorphic_C2xC2": bool(
            len(frozen_group) == 4 and abelian and exponent_two and homomorphic
        ),
        "labelling_is_xor_homomorphism": homomorphic,
        "action_free": free,
        "action_transitive": transitive,
        "lines_are_nonzero_cosets_of_P": True,
        "chart_equivariant": True,
    }

    return Habitat(
        s=s,
        fff=fff,
        delta=delta,
        blocks_by_q=blocks_by_q,
        q_of_block_key=q_of_block_key,
        d_of_line=d_of_line,
        line_of_d=line_of_d,
        event_of_incidence=event_of_incidence,
        presentations_of_event=presentations_of_event,
        certification=certification,
    )


def enumerate_habitats() -> dict[tuple[int, int], tuple[tuple, ...]]:
    """Group the 56 certified cyclic blocks by their ``(P, delta)`` habitat key."""

    groups: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for block in fips_basic.BLOCKS:
        labels = {public_label(e) for e in block}
        if len(labels) != 1:
            raise AssertionError("block spans more than one (P, delta) habitat")
        groups[labels.pop()].append(block)
    if len(groups) != PINS["habitats"]:
        raise AssertionError(f"found {len(groups)} habitats, not 14")
    return {
        label: tuple(sorted(groups[label], key=block_key))
        for label in sorted(groups)
    }


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

class SfpCodec:
    """Exact bijection between the 84 native Event rays and 84 SFP addresses."""

    def __init__(self) -> None:
        self.habitats: dict[tuple[int, int], Habitat] = {}
        for label, blocks in enumerate_habitats().items():
            habitat = build_habitat(label, blocks)
            key = (habitat.s, habitat.fff)
            if key in self.habitats:
                raise AssertionError("two (P, delta) labels collided on (S, FFF)")
            self.habitats[key] = habitat

        self._address_of_ray: dict[tuple, SfpAddress] = {}
        self._ray_of_address: dict[tuple, tuple] = {}
        for key, habitat in sorted(self.habitats.items()):
            for event, pair in habitat.presentations_of_event.items():
                address = SfpAddress(s=key[0], fff=key[1], presentations=pair)
                self._address_of_ray[event] = address
                self._ray_of_address[address.as_key()] = event
        if len(self._address_of_ray) != PINS["events"]:
            raise AssertionError(
                f"codec covers {len(self._address_of_ray)} Events, not 84"
            )

    # -- forward / inverse ---------------------------------------------------

    def encode(self, event: tuple) -> SfpAddress:
        """Native Event ray -> canonical Event-level SFP address."""

        key = ray(event)
        try:
            return self._address_of_ray[key]
        except KeyError as exc:
            raise KeyError("ray is not one of the 84 certified basic Events") from exc

    def decode(self, address: SfpAddress) -> tuple:
        """Canonical Event-level SFP address -> native Event ray."""

        try:
            return self._ray_of_address[address.as_key()]
        except KeyError as exc:
            raise KeyError("address is not in the certified 84-Event range") from exc

    def decode_incidence(self, presentation: IncidencePresentation) -> tuple:
        """Incidence presentation ``(S, FFF, PP, pp)`` -> native Event ray."""

        return self.decode(
            address_of(presentation.s, presentation.fff, presentation.q, presentation.d)
        )

    # -- views --------------------------------------------------------------

    def addresses(self) -> tuple[SfpAddress, ...]:
        return tuple(self._address_of_ray[e] for e in fips_basic.EVENTS)

    def incidence_view(self) -> tuple[tuple[int, int, int, int], ...]:
        """All 168 incidence presentations ``(S, FFF, PP, pp)``, ``pp != 00``."""

        rows = []
        for address in self.addresses():
            for presentation in address.incidences():
                if presentation.d == 0:
                    raise AssertionError("pp=00 leaked into the incidence view")
                rows.append(presentation.as_tuple())
        return tuple(sorted(rows))

    # -- chart record -------------------------------------------------------

    def chart(self) -> dict:
        per_habitat = {}
        for key, habitat in sorted(self.habitats.items()):
            per_habitat[habitat_key(*key)] = {
                "P": habitat.fff,
                "FFF": f"{habitat.fff:03b}",
                "delta": habitat.delta,
                "S": habitat.s,
                "origin_block_PP": bits2(0),
                "origin_block": list(block_key(habitat.blocks_by_q[0])),
                "blocks_by_PP": {
                    bits2(q): list(block_key(b))
                    for q, b in enumerate(habitat.blocks_by_q)
                },
                "pp_to_fano_line": {
                    bits2(d): list(habitat.line_of_d[d]) for d in V2_NONZERO
                },
            }
        body = {
            "declared_normalization": list(DECLARED_CHART),
            "determinism": "derived from the stable textual ray encoding "
                           "(_encode_ray) and sorted Fano-line integer triples; "
                           "never from set/dict iteration order or hash()",
            "per_habitat": per_habitat,
        }
        return {**body, "sha256": digest(body)}


# ---------------------------------------------------------------------------
# ExactSfpCircuit — nonlearned REFERENCE ORACLE (never a label source)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidenceComposition:
    """Result of the incidence-level SFP law on an admitted presentation pair."""

    admit: bool
    same_habitat: bool
    same_block: bool
    valid_ports: bool
    result: IncidencePresentation | None


class ExactSfpCircuit:
    """Nonlearned reference oracle for the certified FIPS relation in SFP code.

    Two levels, both from Outcome 021.05, both built from equality, XOR,
    nonzero tests and endpoint intersection only — there is no pair lookup
    table anywhere in this class.

    Incidence level (021.05 section 6) on ``A=(S1,P1,q1,d1)``, ``B=(S2,P2,q2,d2)``::

        same_habitat = (S1==S2) & (FFF1==FFF2)
        same_block   = (PP1==PP2)
        valid_ports  = (pp1!=0) & (pp2!=0) & (pp1!=pp2)
        admit        = same_habitat & same_block & valid_ports
        pp3          = pp1 XOR pp2      with S3=S1, FFF3=FFF1, PP3=PP1

    Event level (021.05 section 5, final paragraph): two distinct Events are
    cyclically admitted iff their endpoint block sets share exactly one block
    vertex; that vertex is unique, and the forced third Event is obtained by
    XORing the two displacements incident there.

    REFERENCE ORACLE ONLY — see ``REFERENCE_ORACLE_ONLY``. This class does not
    replace Occ / Cyc / Sand or native Event denotation, and must never be used
    to produce ground-truth labels.
    """

    role = "nonlearned reference oracle"
    never_a_label_source = True

    # -- incidence level ----------------------------------------------------

    def compose_incidence(
        self, a: IncidencePresentation, b: IncidencePresentation
    ) -> IncidenceComposition:
        same_habitat = (a.s == b.s) and (a.fff == b.fff)
        same_block = a.q == b.q
        valid_ports = (a.d != 0) and (b.d != 0) and (a.d != b.d)
        admit = same_habitat and same_block and valid_ports
        result = (
            IncidencePresentation(a.s, a.fff, a.q, a.d ^ b.d) if admit else None
        )
        return IncidenceComposition(
            admit=admit,
            same_habitat=same_habitat,
            same_block=same_block,
            valid_ports=valid_ports,
            result=result,
        )

    def admit_incidence(
        self, a: IncidencePresentation, b: IncidencePresentation
    ) -> bool:
        return self.compose_incidence(a, b).admit

    # -- Event level --------------------------------------------------------

    def shared_block(self, a: SfpAddress, b: SfpAddress) -> int | None:
        """The unique shared K4 endpoint, or ``None`` when there is not exactly one."""

        if (a.s != b.s) or (a.fff != b.fff):
            return None
        shared = a.endpoints & b.endpoints
        if len(shared) != 1:
            return None
        return next(iter(shared))

    def admit_event(self, a: SfpAddress, b: SfpAddress) -> bool:
        """Cyclic admission of two Events, from endpoint intersection alone."""

        return self.shared_block(a, b) is not None

    def forced_third_event(self, a: SfpAddress, b: SfpAddress) -> SfpAddress | None:
        """The forced third Event address, by XOR of the incident displacements."""

        q = self.shared_block(a, b)
        if q is None:
            return None
        da, db = a.d, b.d
        if da == 0 or db == 0 or da == db:
            # Distinct Events sharing exactly one vertex necessarily differ in pp;
            # kept as an explicit port check rather than an assumption.
            return None
        return address_of(a.s, a.fff, q, da ^ db)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_codec(codec: SfpCodec) -> dict:
    """Bijectivity: round-trip in both directions, plus actual injectivity."""

    events = fips_basic.EVENTS
    forward_ok = 0
    for event in events:
        if codec.decode(codec.encode(event)) == ray(event):
            forward_ok += 1

    addresses = codec.addresses()
    backward_ok = 0
    for address in addresses:
        if codec.encode(codec.decode(address)) == address:
            backward_ok += 1

    distinct_addresses = {address.as_key() for address in addresses}
    distinct_rays = {codec.decode(address) for address in addresses}

    incidence_ok = 0
    for event in events:
        address = codec.encode(event)
        views = address.incidences()
        if len(views) == 2 and all(
            codec.decode_incidence(v) == ray(event) for v in views
        ):
            incidence_ok += 1

    incidence_rows = codec.incidence_view()

    return {
        "events": len(events),
        "ray_to_address_to_ray": forward_ok,
        "address_to_ray_to_address": backward_ok,
        "round_trip_both_directions_exact": (
            forward_ok == backward_ok == PINS["events"]
        ),
        "distinct_addresses": len(distinct_addresses),
        "distinct_rays": len(distinct_rays),
        "injective": len(distinct_addresses) == PINS["events"],
        "surjective_onto_events": len(distinct_rays) == PINS["events"],
        "events_with_both_presentations_decoding": incidence_ok,
        "incidence_presentations": len(incidence_rows),
        "distinct_incidence_presentations": len(set(incidence_rows)),
        "incidences_match_pin": len(set(incidence_rows)) == PINS["incidences"],
        "all_pp_nonzero": all(row[3] in V2_NONZERO for row in incidence_rows),
    }


def verify_incidence_law(codec: SfpCodec, circuit: ExactSfpCircuit) -> dict:
    """Incidence-level law, checked against native ``third`` at every incidence."""

    rows = codec.incidence_view()
    presentations = [IncidencePresentation(*row) for row in rows]
    admitted = 0
    agree = 0
    disagreements: list[dict] = []
    for a, b in itertools.product(presentations, repeat=2):
        composition = circuit.compose_incidence(a, b)
        if not composition.admit:
            continue
        admitted += 1
        left = codec.decode_incidence(a)
        right = codec.decode_incidence(b)
        if not fips_basic.admissible(left, right):
            disagreements.append(
                {"a": a.as_tuple(), "b": b.as_tuple(), "issue": "native non-admission"}
            )
            continue
        native = fips_basic.third(left, right)
        got = codec.decode_incidence(composition.result)
        if projective.equivalent(native, got):
            agree += 1
        else:
            disagreements.append(
                {
                    "a": a.as_tuple(),
                    "b": b.as_tuple(),
                    "issue": "forced third mismatch",
                    "expected": encode_ray(native),
                    "observed": encode_ray(got),
                }
            )
    return {
        "incidence_presentations": len(presentations),
        "ordered_presentation_pairs": len(presentations) ** 2,
        "admitted_by_incidence_law": admitted,
        "forced_third_agreements": agree,
        "exact": admitted == agree and not disagreements,
        "disagreements": disagreements[:8],
        "disagreement_count": len(disagreements),
    }


def verify_event_relation(codec: SfpCodec, circuit: ExactSfpCircuit) -> dict:
    """THE CRITICAL VALIDATION: circuit vs. native FIPS over all 84*84 pairs."""

    events = fips_basic.EVENTS
    addresses = {event: codec.encode(event) for event in events}

    total = 0
    admission_agree = 0
    circuit_admitted = 0
    native_admitted = 0
    third_checked = 0
    third_agree = 0
    admission_disagreements: list[dict] = []
    third_disagreements: list[dict] = []

    for left, right in itertools.product(events, repeat=2):
        total += 1
        native_ok = fips_basic.admissible(left, right)
        circuit_ok = circuit.admit_event(addresses[left], addresses[right])
        native_admitted += native_ok
        circuit_admitted += circuit_ok
        if native_ok == circuit_ok:
            admission_agree += 1
        else:
            admission_disagreements.append(
                {
                    "a": encode_ray(left),
                    "b": encode_ray(right),
                    "native_admissible": native_ok,
                    "circuit_admit": circuit_ok,
                }
            )
        if not native_ok:
            continue
        third_checked += 1
        native_third = fips_basic.third(left, right)
        forced = circuit.forced_third_event(addresses[left], addresses[right])
        if forced is None:
            third_disagreements.append(
                {
                    "a": encode_ray(left),
                    "b": encode_ray(right),
                    "issue": "circuit produced no forced third",
                }
            )
            continue
        got = codec.decode(forced)
        # Exact projective Event identity; never float proximity.
        if projective.equivalent(native_third, got):
            third_agree += 1
        else:
            third_disagreements.append(
                {
                    "a": encode_ray(left),
                    "b": encode_ray(right),
                    "issue": "forced third mismatch",
                    "expected": encode_ray(native_third),
                    "observed": encode_ray(got),
                }
            )

    return {
        "ordered_pairs_checked": total,
        "admission_agreements": admission_agree,
        "admission_agreement_exact": (
            admission_agree == total == PINS["total_ordered_pairs"]
        ),
        "native_admissible_ordered_pairs": native_admitted,
        "circuit_admitted_ordered_pairs": circuit_admitted,
        "admitted_count_matches_pin": (
            circuit_admitted == native_admitted == PINS["ordered_pairs"]
        ),
        "forced_third_checked": third_checked,
        "forced_third_agreements": third_agree,
        "forced_third_agreement_exact": (
            third_agree == third_checked == PINS["ordered_pairs"]
        ),
        "comparison": "topographo.ssd.projective.equivalent (exact projective "
                      "Event identity, not float proximity)",
        "zero_disagreements": not admission_disagreements and not third_disagreements,
        "admission_disagreements": admission_disagreements[:8],
        "admission_disagreement_count": len(admission_disagreements),
        "forced_third_disagreements": third_disagreements[:8],
        "forced_third_disagreement_count": len(third_disagreements),
        "no_pair_lookup_table": "admission and third are computed from equality, "
                                "XOR, nonzero tests and endpoint intersection only",
    }


def habitat_certification(codec: SfpCodec) -> dict:
    per_habitat = {
        habitat_key(*key): habitat.certification
        for key, habitat in sorted(codec.habitats.items())
    }
    names = sorted({name for record in per_habitat.values() for name in record})
    uniform = {
        name: sorted(
            {
                tuple(record[name]) if isinstance(record[name], list) else record[name]
                for record in per_habitat.values()
            },
            key=repr,
        )
        for name in names
    }
    booleans = [
        "all_classes_are_perfect_matchings",
        "all_matchings_fixed_point_free_involutions",
        "group_closed",
        "group_isomorphic_C2xC2",
        "labelling_is_xor_homomorphism",
        "action_free",
        "action_transitive",
        "lines_are_nonzero_cosets_of_P",
        "chart_equivariant",
    ]
    return {
        "habitats_certified": len(per_habitat),
        "all_habitats_free_transitive_C2xC2": all(
            record[name] for record in per_habitat.values() for name in booleans
        ),
        "uniform_across_all_habitats": {
            name: [list(v) if isinstance(v, tuple) else v for v in values]
            for name, values in uniform.items()
        },
        "per_habitat": per_habitat,
    }


def provenance() -> dict:
    live = fips_basic.TABLE_SHA256
    return {
        "topographo_version": importlib.metadata.version("topographo"),
        "base_commit": BASE_COMMIT,
        "table_sha256": {
            "expected": EXPECTED_TABLE_SHA256,
            "observed": live,
            "agrees": live == EXPECTED_TABLE_SHA256
            and live == fips_basic.EXPECTED_TABLE_SHA256,
        },
        "gate_0": "experiments/sfp_representation/conformance.py (PASSED)",
        "native_sources": [
            "topographo.ssd.fips_basic.EVENTS / BLOCKS / admissible / third",
            "topographo.ssd.seal.public_label (habitat key (P, delta))",
            "topographo.ssd.projective.equivalent (exact Event identity)",
        ],
    }


def audit() -> dict:
    codec = SfpCodec()
    circuit = ExactSfpCircuit()

    chart = codec.chart()
    habitats = habitat_certification(codec)
    codec_report = verify_codec(codec)
    incidence_report = verify_incidence_law(codec, circuit)
    event_report = verify_event_relation(codec, circuit)

    addresses = {
        encode_ray(codec.decode(a)): a.as_json() for a in codec.addresses()
    }

    laws = {
        "codec_round_trip_both_directions": codec_report[
            "round_trip_both_directions_exact"
        ],
        "codec_injective": codec_report["injective"],
        "codec_surjective": codec_report["surjective_onto_events"],
        "codec_incidences_168": codec_report["incidences_match_pin"],
        "codec_all_pp_nonzero": codec_report["all_pp_nonzero"],
        "habitat_torsors_certified": habitats["all_habitats_free_transitive_C2xC2"],
        "habitats_all_14": habitats["habitats_certified"] == PINS["habitats"],
        "incidence_law_exact": incidence_report["exact"],
        "event_admission_exact_7056": event_report["admission_agreement_exact"],
        "event_admitted_336": event_report["admitted_count_matches_pin"],
        "event_forced_third_exact_336": event_report["forced_third_agreement_exact"],
        "event_zero_disagreements": event_report["zero_disagreements"],
        "provenance_table_digest": provenance()["table_sha256"]["agrees"],
    }
    broken = sorted(name for name, ok in laws.items() if not ok)

    return {
        "module": "experiments/sfp_representation/sfp.py",
        "purpose": "SFP core: habitat enumeration and the S|FFF|PP|pp codec, plus "
                   "ExactSfpCircuit as a nonlearned reference oracle",
        "fences": list(FENCES),
        "reference_oracle_only": REFERENCE_ORACLE_ONLY,
        "sfp_typing": {
            "S": "habitat sign delta",
            "FFF": "nonzero Fano point P (1..7)",
            "PP": "affine cyclic-block coordinate q in F_2^2",
            "pp": "nonzero affine displacement d / Fano-line class through P",
        },
        "pins": PINS,
        "chart": chart,
        "habitat_certification": habitats,
        "codec": codec_report,
        "incidence_law": incidence_report,
        "event_relation": event_report,
        "addresses": addresses,
        "digests": {
            "chart": chart["sha256"],
            "addresses": digest(addresses),
            "incidence_view": digest(
                [list(row) for row in codec.incidence_view()]
            ),
        },
        "provenance": provenance(),
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - codec is bijective on the 84 certified Events and "
                "ExactSfpCircuit reproduces the certified native FIPS relation "
                "exactly over all 7056 ordered pairs"
                if not broken
                else "FAIL - see broken_laws"
            ),
        },
    }


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _report(payload: dict) -> None:
    codec_report = payload["codec"]
    event_report = payload["event_relation"]
    habitats = payload["habitat_certification"]
    print(
        f"codec: round-trip {codec_report['ray_to_address_to_ray']}/84 forward, "
        f"{codec_report['address_to_ray_to_address']}/84 inverse; "
        f"{codec_report['distinct_addresses']} distinct addresses "
        f"(injective={codec_report['injective']}); "
        f"{codec_report['incidence_presentations']} incidence presentations",
        flush=True,
    )
    print(
        f"habitats: {habitats['habitats_certified']}/14 certified "
        f"free+transitive C2^2 torsors "
        f"({habitats['all_habitats_free_transitive_C2xC2']})",
        flush=True,
    )
    print(
        f"circuit: admission "
        f"{event_report['admission_agreements']}/{event_report['ordered_pairs_checked']}"
        f", admitted {event_report['circuit_admitted_ordered_pairs']} ordered pairs, "
        f"forced third {event_report['forced_third_agreements']}/"
        f"{event_report['forced_third_checked']} exact projective agreement",
        flush=True,
    )
    print(f"chart sha256: {payload['chart']['sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches sfp_codec.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

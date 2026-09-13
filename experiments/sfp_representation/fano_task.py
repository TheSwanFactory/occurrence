"""011.01 Gate 0: the anonymous global Fano / ``FFF`` problem, enumerated exactly.

``009.10`` established **local** representation discovery: six anonymous Event
tokens plus the pairwise ADMIT relation plus one triadic anchor recover the local
four-chamber chart. It supplied habitat identity throughout and never touched the
global question. Every habitat is the same six-node octahedron once tokens are
permuted, so fourteen disconnected local charts carry no global Fano identity.

``011.01`` asks the next question::

    Can the seven-point Fano / FFF organization be recovered from fourteen
    anonymous habitats and certified cross-habitat support incidence, without
    supplying FFF or its XOR law?

This module is Gate 0 and the frozen episode construction. It trains nothing.
``011.01`` section 3 forbids training until the exact identifiability census is
computed from the anonymous relation and checked against the hidden certified
labels, so the census is computed here first and the primary run is gated on it.

The observation
---------------
A typed anonymous graph, and nothing else::

    H: 14 habitat nodes
    E: 84 Event nodes
    L:  7 anonymous first-half support-axis nodes
    R:  7 anonymous second-half support-axis nodes

    H--E    Event belongs to habitat
    E--L    Event uses this first-half support axis
    E--R    Event uses this second-half support axis

``axes_of`` is the single place in this repository where the native two-axis
projective support of an Event is read. It returns the two axis *indices* and
discards both coefficient signs; the indices then survive only as equality
classes, because every L and R node is renamed by a fresh permutation in every
episode. ``P``, ``delta``, ``S``, ``FFF``, ``PP``, ``pp``, native ray coordinate
vectors, coefficient signs, block IDs, Event indices and habitat indices are
never inputs. They are retained in the ``hidden_*`` fields of ``Observation`` for
exactly three purposes ``011.01`` section 2.1 permits: frozen target
construction, fold/audit construction, and post-hoc equivalence checking.

What is enumerated, not assumed
-------------------------------
Every number below is derived by exhaustive enumeration over the finite objects
and then compared against the ``011.01`` section 3.1 expectation as a *pin*::

    each habitat omits exactly one L node and one R node
    habitats with equal L/R support pattern         -> 7 two-habitat classes
    the 7 omitted-L / omitted-R pairs               -> an L->R bijection
    each recovered class exposes 3 reciprocal support pairs
    adjoining the class deduplicates to             -> 7 three-point lines
    every one of the 21 point pairs lies on exactly one line
    |Aut(recovered plane)|                          -> 168
    equivalence to the hidden certified FFF plane
    |Aut(observed anonymous graph)|
    the localized control's exact information ceilings

The certified reference plane is ``topographo.core.f2_groups.LINES3`` with its
``gl_3_2`` collineation group, promoted into the package in 0.8.3. Nothing
derived from it reaches the learner: it appears only in the section 3.1 step-6
equivalence check and in the hidden-label agreement checks.

The cross-habitat-destroying control
------------------------------------
``build_localized_observation`` relabels the L and R equality classes
*independently inside each habitat*. Every habitat keeps exactly the incidence
pattern and the within-habitat node degrees it has in the main arm -- one private
L node and one private R node per Event, a perfect matching on both sides, which
is precisely what the main arm restricts to. What is removed, and the only thing
removed, is L/R identity **shared between** habitats. The global L/R degree
necessarily falls from 12 to 1: that sharing is the object under test, so no
control can both destroy it and preserve it.

Torch-free. It imports the frozen ``task`` catalogue and ``topographo`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from task import Dataset, build_dataset, digest, habitat_label

from topographo.core import f2_groups as groups
from topographo.ssd import fips_basic

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_fano_artifacts"
OUTPUT = ARTIFACTS / "fano_task.json"

#: The frozen ``009.10`` result commit this turn builds on.
BASE_COMMIT = "d4efc73e0e8d5b5f4e3a8a8b5e9f5f1e8b9c0d4e"

EXECUTED_TASK = "011.01-GPT-global-Fano-FFF-discovery-task.md"
PRIOR_RESULT = "009.10-Kiro-opaque-local-chart-discovery-result.md"

EXPECTED_TABLE_SHA256 = (
    "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea"
)
EXPECTED_CATALOGUE_SHA256 = (
    "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097"
)
EXPECTED_DATASET_SHA256 = (
    "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721"
)

#: Typed node counts of the main-arm observation.
N_HABITATS = 14
N_EVENTS = 84
N_AXES = 7

#: Node types, in the order their one-hot channels appear in the input tensor.
NODE_TYPES = ("habitat", "event", "left", "right")

#: The two arms. ``main`` shares L/R identity across habitats; ``localized``
#: does not, and is the section 2.3 capacity-matched control.
ARMS = ("main", "localized")

#: Frozen relabelling namespaces. The structure is one finite global object, so
#: per ``011.01`` section 6 the generalization claim is fresh opaque-name
#: transport, not a structural-family holdout.
NAMESPACE_SPLITS = (
    ("train", 8),
    ("validation", 4),
    ("test", 8),
    ("transport", 8),
)

#: Query kinds. ``mate`` is section 4.1, ``completion`` is section 4.2.
QUERY_KINDS = ("mate", "completion")

#: ``011.01`` section 3.1 expectations, carried as PINS TO VERIFY.
PINS = {
    "habitat_nodes": N_HABITATS,
    "event_nodes": N_EVENTS,
    "left_nodes": N_AXES,
    "right_nodes": N_AXES,
    "observed_nodes_main": N_HABITATS + N_EVENTS + 2 * N_AXES,
    "habitat_degree": 6,
    "event_degree": 3,
    "left_degree": 12,
    "right_degree": 12,
    "events_per_habitat": 6,
    "omitted_left_per_habitat": 1,
    "omitted_right_per_habitat": 1,
    "recovered_point_classes": 7,
    "habitats_per_point_class": 2,
    "support_pairs_per_class": 3,
    "recovered_lines": 7,
    "point_pairs": 21,
    "lines_through_a_point": 3,
    "recovered_plane_automorphism_order": 168,
    "observed_graph_automorphism_order": 43008,
    "left_collineations": 168,
    "sign_choices": 128,
    "mate_queries_per_namespace": N_HABITATS,
    "completion_queries_per_namespace": 84,
    "queries_per_namespace": N_HABITATS + 84,
    "namespaces": sum(count for _, count in NAMESPACE_SPLITS),
    "localized_left_nodes": 84,
    "localized_right_nodes": 84,
    "localized_components": N_HABITATS,
}

#: The exact ceilings this gate must establish before the primary run.
EXPECTED_CEILINGS = {
    "main_mate": 1.0,
    "main_completion": 1.0,
    "main_plane": 1.0,
    "localized_mate": 1.0 / 13.0,
    "localized_completion": 1.0 / 66.0,
}

FENCES = (
    "The observation is built from equality/incidence classes only. axes_of "
    "reads the two native support axis INDICES and discards both coefficient "
    "signs; the indices then survive only as equality classes, because every L "
    "and R node is renamed per episode.",
    "P, delta, S, FFF, PP, pp, numeric Fano labels, numeric axis labels, native "
    "ray coordinate vectors, coefficient signs, block IDs, stable Event indices "
    "and stable habitat indices are NEVER learner inputs. They construct frozen "
    "targets, folds and post-hoc equivalence checks, which 011.01 section 2.1 "
    "permits, and nothing else.",
    "XOR appears in target generation and audit only. No XOR, Fano completion "
    "table or incidence solver is on the learned path, which fano_heads audits "
    "mechanically over the module source rather than asserting in prose.",
    "topographo.core.f2_groups supplies the CERTIFIED reference plane for the "
    "section 3.1 step-6 equivalence check. It is not an input and not a target.",
    "The localized control destroys cross-habitat L/R sharing and nothing else. "
    "Within-habitat incidence and within-habitat degrees are identical to the "
    "main arm; the global L/R degree falls from 12 to 1 because that sharing is "
    "exactly what the control removes.",
    "The transparent reconstruction is a NONLEARNED CEILING, never a competing "
    "learned baseline. fano_baselines imports it; the learned path does not.",
    "The generalization claim is FRESH OPAQUE-NAME TRANSPORT of one finite global "
    "object, not held-out structure. There is one Fano plane and 011 does not "
    "pretend otherwise.",
    "S is supplied nowhere and discovered nowhere. delta never enters the "
    "observation at all, which is why the two habitats of a point class are "
    "exchangeable and why the section 4.2 readout must be set-valued.",
    "FFF=000 and pp=00 remain formal algebraic completion values outside the "
    "finite Event catalogue and play no part here. BOTTOM stays distinct from "
    "every algebraic zero; no query in this turn is a BOTTOM row.",
)

SUPPLIED = (
    "14 habitat nodes, 84 Event nodes, 7 left-axis nodes, 7 right-axis nodes, "
    "each carrying its type one-hot and nothing else",
    "the H--E, E--L and E--R incidence edges",
    "a query mark on one habitat node (mate) or two habitat nodes (completion)",
)
WITHHELD = (
    "P / FFF",
    "S / delta",
    "PP and pp",
    "numeric Fano labels 1..7",
    "numeric native basis-axis labels",
    "coefficient signs from the native ray",
    "native ray coordinate vectors or strings",
    "stable Event indices",
    "stable habitat indices",
    "block IDs",
    "XOR or a Fano completion table",
    "any trainable embedding keyed by Event, habitat or axis identity",
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 / 011 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def pin(expected: object, observed: object, source: str) -> dict:
    """Record an expected/observed pair with an explicit agreement flag."""

    return {
        "expected": expected,
        "observed": observed,
        "agrees": expected == observed,
        "source": source,
    }


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


def rd(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def hash_free_seed(name: str) -> int:
    """Digest-derived seed. Never ``hash()``, which is salted per process."""

    return int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)


def lehmer_permutation(n: int, key: str) -> tuple[int, ...]:
    """A deterministic permutation of ``range(n)`` from a digest-derived index.

    The Lehmer code is used rather than ``random.shuffle`` so the frozen
    namespaces depend on nothing but the key string -- no RNG implementation, no
    Python version, no platform.
    """

    index = int(hashlib.sha256(key.encode()).hexdigest(), 16) % math.factorial(n)
    pool = list(range(n))
    out: list[int] = []
    for size in range(n, 0, -1):
        quotient, index = divmod(index, math.factorial(size - 1))
        out.append(pool.pop(quotient))
    return tuple(out)


def invert(perm: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(perm)
    for source, image in enumerate(perm):
        out[image] = source
    return tuple(out)


def encode_perm(perm: tuple[int, ...]) -> str:
    return "-".join(str(p) for p in perm)


# ---------------------------------------------------------------------------
# the one place the native two-axis support is read
# ---------------------------------------------------------------------------

def axes_of(encoding: str) -> tuple[int, int]:
    """``(left_axis, right_axis)`` of one certified Event ray, signs discarded.

    ``task.encode_ray`` renders a certified Event as ``"1:1,10:-1"``: basis axis
    1 with coefficient ``+1`` and basis axis 10 with coefficient ``-1``. A
    certified Event is ``e_i + s e_{8+j}`` with ``i, j`` in ``1..7`` and
    ``i != j``, so exactly two coefficients are nonzero. This function returns
    ``(i, j)`` and **drops both coefficients**, which is the whole of
    ``011.01`` section 2.2's "the numeric axis values and all coefficient signs
    are discarded". The returned indices are consumed only to decide *which
    Events share an axis*; every axis node is renamed per episode.
    """

    left: list[int] = []
    right: list[int] = []
    for term in encoding.split(","):
        axis_text, coefficient_text = term.split(":")
        axis = int(axis_text)
        if int(coefficient_text) == 0:
            continue
        if 1 <= axis <= 7:
            left.append(axis)
        elif 9 <= axis <= 15:
            right.append(axis - 8)
        else:
            raise AssertionError(f"unexpected basis axis {axis} in {encoding!r}")
    if len(left) != 1 or len(right) != 1:
        raise AssertionError(
            f"{encoding!r} is not a two-axis support: left {left} right {right}"
        )
    if left[0] == right[0]:
        raise AssertionError(f"{encoding!r} has equal support axes, not an Event")
    return left[0], right[0]


# ---------------------------------------------------------------------------
# the anonymous typed observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """The typed anonymous graph, in canonical slot coordinates.

    Slots are bookkeeping. Every episode maps slots onto opaque nodes through a
    fresh permutation per namespace, so no slot is ever a stable semantic
    address. The ``hidden_*`` fields are the certified labels; they build
    targets, folds and post-hoc audits, and they are never learner inputs.
    """

    arm: str
    n_habitats: int
    n_events: int
    n_left: int
    n_right: int
    he: tuple[tuple[int, int], ...]
    el: tuple[tuple[int, int], ...]
    er: tuple[tuple[int, int], ...]
    hidden_point: tuple[int, ...]
    hidden_sign: tuple[int, ...]
    hidden_habitat_key: tuple[str, ...]
    hidden_event_index: tuple[int, ...]
    hidden_left_axis: tuple[int, ...]
    hidden_right_axis: tuple[int, ...]

    # -- typed accessors ----------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.n_habitats + self.n_events + self.n_left + self.n_right

    def habitat_node(self, slot: int) -> int:
        return slot

    def event_node(self, slot: int) -> int:
        return self.n_habitats + slot

    def left_node(self, slot: int) -> int:
        return self.n_habitats + self.n_events + slot

    def right_node(self, slot: int) -> int:
        return self.n_habitats + self.n_events + self.n_left + slot

    def node_type(self, node: int) -> str:
        if node < self.n_habitats:
            return "habitat"
        if node < self.n_habitats + self.n_events:
            return "event"
        if node < self.n_habitats + self.n_events + self.n_left:
            return "left"
        return "right"

    @lru_cache(maxsize=None)
    def events_of_habitat(self) -> tuple[tuple[int, ...], ...]:
        rows: list[list[int]] = [[] for _ in range(self.n_habitats)]
        for habitat, event in self.he:
            rows[habitat].append(event)
        return tuple(tuple(sorted(row)) for row in rows)

    @lru_cache(maxsize=None)
    def habitat_of_event(self) -> tuple[int, ...]:
        out = [-1] * self.n_events
        for habitat, event in self.he:
            if out[event] != -1:
                raise AssertionError(f"Event slot {event} belongs to two habitats")
            out[event] = habitat
        if -1 in out:
            raise AssertionError("an Event slot belongs to no habitat")
        return tuple(out)

    @lru_cache(maxsize=None)
    def left_of_event(self) -> tuple[int, ...]:
        out = [-1] * self.n_events
        for event, left in self.el:
            if out[event] != -1:
                raise AssertionError(f"Event slot {event} uses two left axes")
            out[event] = left
        if -1 in out:
            raise AssertionError("an Event slot uses no left axis")
        return tuple(out)

    @lru_cache(maxsize=None)
    def right_of_event(self) -> tuple[int, ...]:
        out = [-1] * self.n_events
        for event, right in self.er:
            if out[event] != -1:
                raise AssertionError(f"Event slot {event} uses two right axes")
            out[event] = right
        if -1 in out:
            raise AssertionError("an Event slot uses no right axis")
        return tuple(out)

    @lru_cache(maxsize=None)
    def left_support(self) -> tuple[frozenset[int], ...]:
        left_of = self.left_of_event()
        return tuple(
            frozenset(left_of[e] for e in events) for events in self.events_of_habitat()
        )

    @lru_cache(maxsize=None)
    def right_support(self) -> tuple[frozenset[int], ...]:
        right_of = self.right_of_event()
        return tuple(
            frozenset(right_of[e] for e in events)
            for events in self.events_of_habitat()
        )

    @lru_cache(maxsize=None)
    def edges(self) -> tuple[tuple[int, int], ...]:
        """The undirected edge set in node coordinates, ascending."""

        out = set()
        for habitat, event in self.he:
            out.add((self.habitat_node(habitat), self.event_node(event)))
        for event, left in self.el:
            out.add((self.event_node(event), self.left_node(left)))
        for event, right in self.er:
            out.add((self.event_node(event), self.right_node(right)))
        return tuple(sorted(tuple(sorted(edge)) for edge in out))

    @lru_cache(maxsize=None)
    def adjacency(self) -> tuple[frozenset[int], ...]:
        rows: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for a, b in self.edges():
            rows[a].add(b)
            rows[b].add(a)
        return tuple(frozenset(row) for row in rows)

    def degree_spectrum(self) -> dict[str, list[int]]:
        adjacency = self.adjacency()
        out: dict[str, set[int]] = defaultdict(set)
        for node in range(self.n_nodes):
            out[self.node_type(node)].add(len(adjacency[node]))
        return {kind: sorted(values) for kind, values in sorted(out.items())}

    # -- the hidden certified answers --------------------------------------

    def certified_mate(self, habitat: int) -> int:
        point = self.hidden_point[habitat]
        mates = [
            other
            for other in range(self.n_habitats)
            if other != habitat and self.hidden_point[other] == point
        ]
        if len(mates) != 1:
            raise AssertionError(f"habitat slot {habitat} has {len(mates)} mates")
        return mates[0]

    def certified_completion(self, first: int, second: int) -> tuple[int, ...]:
        left = self.hidden_point[first]
        right = self.hidden_point[second]
        if left == right:
            raise ValueError("a completion query needs two distinct point classes")
        # XOR appears here, in TARGET GENERATION, and nowhere on the learned path.
        third = left ^ right
        out = tuple(
            habitat
            for habitat in range(self.n_habitats)
            if self.hidden_point[habitat] == third
        )
        if len(out) != 2:
            raise AssertionError(f"point {third} carries {len(out)} habitats, not 2")
        return out

    def as_row(self) -> list:
        return [
            self.arm,
            [self.n_habitats, self.n_events, self.n_left, self.n_right],
            [list(p) for p in self.he],
            [list(p) for p in self.el],
            [list(p) for p in self.er],
        ]

    def sha256(self) -> str:
        """Digest of the ANONYMOUS relation only; no hidden field enters it."""

        return digest(self.as_row())


def build_observation(dataset: Dataset) -> Observation:
    """The main-arm anonymous observation, from the frozen certified catalogue."""

    catalogue = dataset.catalogue
    habitat_rows = catalogue.events_of_habitat()
    if len(habitat_rows) != N_HABITATS:
        raise AssertionError(f"{len(habitat_rows)} habitats, not {N_HABITATS}")

    event_slot_of: dict[int, int] = {}
    hidden_event_index: list[int] = []
    hidden_point: list[int] = []
    hidden_sign: list[int] = []
    hidden_key: list[str] = []
    he: list[tuple[int, int]] = []

    for habitat_slot, (key, members) in enumerate(habitat_rows):
        if len(members) != PINS["events_per_habitat"]:
            raise AssertionError(f"habitat {key} holds {len(members)} Events, not 6")
        point, delta = catalogue.habitat(members[0])
        if any(catalogue.habitat(m) != (point, delta) for m in members):
            raise AssertionError(f"habitat {key} is not homogeneous in (P, delta)")
        if habitat_label(point, delta) != key:
            raise AssertionError(f"habitat key {key} disagrees with {(point, delta)}")
        hidden_point.append(point)
        hidden_sign.append(0 if delta == 1 else 1)
        hidden_key.append(key)
        for member in members:
            slot = len(hidden_event_index)
            event_slot_of[member] = slot
            hidden_event_index.append(member)
            he.append((habitat_slot, slot))

    if len(hidden_event_index) != N_EVENTS:
        raise AssertionError(f"{len(hidden_event_index)} Event slots, not {N_EVENTS}")

    left_axes: set[int] = set()
    right_axes: set[int] = set()
    support: dict[int, tuple[int, int]] = {}
    for event_index, slot in event_slot_of.items():
        left, right = axes_of(catalogue.encodings[event_index])
        support[slot] = (left, right)
        left_axes.add(left)
        right_axes.add(right)

    hidden_left_axis = tuple(sorted(left_axes))
    hidden_right_axis = tuple(sorted(right_axes))
    if len(hidden_left_axis) != N_AXES or len(hidden_right_axis) != N_AXES:
        raise AssertionError(
            f"{len(hidden_left_axis)} left and {len(hidden_right_axis)} right axes, "
            f"not {N_AXES} each"
        )
    left_slot = {axis: k for k, axis in enumerate(hidden_left_axis)}
    right_slot = {axis: k for k, axis in enumerate(hidden_right_axis)}

    el = tuple(sorted((slot, left_slot[support[slot][0]]) for slot in support))
    er = tuple(sorted((slot, right_slot[support[slot][1]]) for slot in support))

    return Observation(
        arm="main",
        n_habitats=N_HABITATS,
        n_events=N_EVENTS,
        n_left=N_AXES,
        n_right=N_AXES,
        he=tuple(sorted(he)),
        el=el,
        er=er,
        hidden_point=tuple(hidden_point),
        hidden_sign=tuple(hidden_sign),
        hidden_habitat_key=tuple(hidden_key),
        hidden_event_index=tuple(hidden_event_index),
        hidden_left_axis=hidden_left_axis,
        hidden_right_axis=hidden_right_axis,
    )


def build_localized_observation(main: Observation) -> Observation:
    """The section 2.3 control: L/R identity localized inside each habitat.

    Each Event keeps exactly one L neighbour and one R neighbour, and each
    habitat keeps its perfect matching on both sides, so the within-habitat
    incidence pattern and within-habitat degrees are bit-identical to the main
    arm. No L or R node is shared between habitats.
    """

    habitat_of = main.habitat_of_event()
    left_of = main.left_of_event()
    right_of = main.right_of_event()

    left_slot: dict[tuple[int, int], int] = {}
    right_slot: dict[tuple[int, int], int] = {}
    el: list[tuple[int, int]] = []
    er: list[tuple[int, int]] = []
    for event in range(main.n_events):
        left_key = (habitat_of[event], left_of[event])
        right_key = (habitat_of[event], right_of[event])
        if left_key not in left_slot:
            left_slot[left_key] = len(left_slot)
        if right_key not in right_slot:
            right_slot[right_key] = len(right_slot)
        el.append((event, left_slot[left_key]))
        er.append((event, right_slot[right_key]))

    return Observation(
        arm="localized",
        n_habitats=main.n_habitats,
        n_events=main.n_events,
        n_left=len(left_slot),
        n_right=len(right_slot),
        he=main.he,
        el=tuple(sorted(el)),
        er=tuple(sorted(er)),
        hidden_point=main.hidden_point,
        hidden_sign=main.hidden_sign,
        hidden_habitat_key=main.hidden_habitat_key,
        hidden_event_index=main.hidden_event_index,
        hidden_left_axis=tuple(-1 for _ in range(len(left_slot))),
        hidden_right_axis=tuple(-1 for _ in range(len(right_slot))),
    )


# ---------------------------------------------------------------------------
# Gate 0 section 3.1 — the transparent reconstruction, from the anonymous graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reconstruction:
    """What the anonymous observation alone determines, step by step.

    Nothing here reads a ``hidden_*`` field. ``fano_baselines`` carries this as
    the section 3.2 nonlearned ceiling; the learned path never imports it.
    """

    steps: tuple[dict, ...]
    classes: tuple[tuple[int, ...], ...]
    class_of_habitat: tuple[int, ...]
    omitted_left: tuple[int, ...]
    omitted_right: tuple[int, ...]
    left_to_right: tuple[int, ...]
    class_of_left: tuple[int, ...]
    support_pairs: tuple[tuple[tuple[int, int], ...], ...]
    lines: tuple[tuple[int, int, int], ...]
    valid: bool

    def mate(self, habitat: int) -> int:
        members = self.classes[self.class_of_habitat[habitat]]
        others = [m for m in members if m != habitat]
        if len(others) != 1:
            raise AssertionError(f"class of habitat {habitat} is not a pair")
        return others[0]

    def third_class(self, first: int, second: int) -> int:
        if first == second:
            raise ValueError("a completion query needs two distinct classes")
        found = [
            line for line in self.lines if first in line and second in line
        ]
        if len(found) != 1:
            raise AssertionError(
                f"classes {first},{second} lie on {len(found)} lines, not 1"
            )
        return next(p for p in found[0] if p not in (first, second))

    def completion(self, first: int, second: int) -> tuple[int, ...]:
        left = self.class_of_habitat[first]
        right = self.class_of_habitat[second]
        return tuple(sorted(self.classes[self.third_class(left, right)]))

    def as_row(self) -> list:
        return [
            [list(c) for c in self.classes],
            list(self.class_of_habitat),
            list(self.omitted_left),
            list(self.omitted_right),
            list(self.left_to_right),
            [list(line) for line in self.lines],
        ]

    def sha256(self) -> str:
        return digest(self.as_row())


def reconstruct(observation: Observation) -> Reconstruction:
    """Run ``011.01`` section 3.1 steps 1-5 on the anonymous graph.

    Each step records what it verified rather than what it assumed. A step that
    fails sets ``valid = False`` and the later steps degrade rather than raise,
    so the census can report *exactly* what is and is not identifiable.
    """

    steps: list[dict] = []
    left_support = observation.left_support()
    right_support = observation.right_support()

    # step 1 -- each habitat omits exactly one L node and one R node
    omitted_left: list[int] = []
    omitted_right: list[int] = []
    step1_ok = True
    for habitat in range(observation.n_habitats):
        missing_left = sorted(set(range(observation.n_left)) - left_support[habitat])
        missing_right = sorted(set(range(observation.n_right)) - right_support[habitat])
        if len(missing_left) != 1 or len(missing_right) != 1:
            step1_ok = False
            omitted_left.append(-1)
            omitted_right.append(-1)
            continue
        omitted_left.append(missing_left[0])
        omitted_right.append(missing_right[0])
    steps.append(
        {
            "step": 1,
            "statement": "each habitat omits exactly one L node and one R node",
            "holds": step1_ok,
            "omitted_left_counts": sorted(
                Counter(
                    len(set(range(observation.n_left)) - left_support[h])
                    for h in range(observation.n_habitats)
                ).items()
            ),
            "omitted_right_counts": sorted(
                Counter(
                    len(set(range(observation.n_right)) - right_support[h])
                    for h in range(observation.n_habitats)
                ).items()
            ),
        }
    )

    # step 2 -- equal unsigned L/R support pattern partitions the habitats
    by_pattern: dict[tuple[frozenset[int], frozenset[int]], list[int]] = defaultdict(list)
    for habitat in range(observation.n_habitats):
        by_pattern[(left_support[habitat], right_support[habitat])].append(habitat)
    classes = tuple(
        tuple(sorted(members)) for members in sorted(by_pattern.values())
    )
    sizes = sorted(Counter(len(members) for members in classes).items())
    step2_ok = len(classes) == N_AXES and sizes == [(2, N_AXES)]
    class_of_habitat = [-1] * observation.n_habitats
    for index, members in enumerate(classes):
        for member in members:
            class_of_habitat[member] = index
    steps.append(
        {
            "step": 2,
            "statement": (
                "habitats with the same unsigned L/R support pattern form seven "
                "two-habitat equivalence classes"
            ),
            "holds": step2_ok,
            "class_count": len(classes),
            "class_size_histogram": sizes,
        }
    )

    # step 3 -- the omitted-L / omitted-R pairs induce an L -> R bijection
    left_to_right = [-1] * observation.n_left
    class_of_left = [-1] * observation.n_left
    step3_ok = step1_ok and step2_ok
    if step3_ok:
        for index, members in enumerate(classes):
            lefts = {omitted_left[m] for m in members}
            rights = {omitted_right[m] for m in members}
            if len(lefts) != 1 or len(rights) != 1:
                step3_ok = False
                break
            left = lefts.pop()
            right = rights.pop()
            if left_to_right[left] != -1:
                step3_ok = False
                break
            left_to_right[left] = right
            class_of_left[left] = index
        step3_ok = step3_ok and sorted(left_to_right) == list(range(observation.n_right))
    steps.append(
        {
            "step": 3,
            "statement": (
                "the seven omitted-L / omitted-R pairs induce a bijection between "
                "the L and R namespaces, without using their numeric labels"
            ),
            "holds": step3_ok,
            "bijection": list(left_to_right) if step3_ok else None,
        }
    )

    # step 4 -- each class exposes three reciprocal support pairs of other classes
    right_to_left = [-1] * observation.n_right
    if step3_ok:
        for left, right in enumerate(left_to_right):
            right_to_left[right] = left
    left_of = observation.left_of_event()
    right_of = observation.right_of_event()
    support_pairs: list[tuple[tuple[int, int], ...]] = []
    step4_ok = step3_ok
    reciprocal_all = step3_ok
    for index, members in enumerate(classes):
        if not step3_ok:
            support_pairs.append(())
            continue
        ordered: set[tuple[int, int]] = set()
        for member in members:
            for event in observation.events_of_habitat()[member]:
                first = class_of_left[left_of[event]]
                second = class_of_left[right_to_left[right_of[event]]]
                ordered.add((first, second))
        reciprocal = all((b, a) in ordered for a, b in ordered)
        reciprocal_all = reciprocal_all and reciprocal
        unordered = tuple(sorted({tuple(sorted(p)) for p in ordered}))
        if (
            not reciprocal
            or len(unordered) != PINS["support_pairs_per_class"]
            or any(index in pair for pair in unordered)
            or any(pair[0] == pair[1] for pair in unordered)
        ):
            step4_ok = False
        support_pairs.append(unordered)
    steps.append(
        {
            "step": 4,
            "statement": (
                "after transporting R through the recovered bijection, each "
                "recovered point class exposes three reciprocal unordered support "
                "pairs among the other six classes"
            ),
            "holds": step4_ok,
            "reciprocal_everywhere": reciprocal_all,
            "support_pair_counts": sorted(
                Counter(len(pairs) for pairs in support_pairs).items()
            ),
        }
    )

    # step 5 -- adjoin the class and deduplicate to seven three-point lines
    line_set: set[tuple[int, int, int]] = set()
    if step4_ok:
        for index, pairs in enumerate(support_pairs):
            for first, second in pairs:
                line_set.add(tuple(sorted((index, first, second))))
    lines = tuple(sorted(line_set))
    pair_cover = Counter()
    for line in lines:
        for pair in itertools.combinations(line, 2):
            pair_cover[pair] += 1
    point_cover = Counter(point for line in lines for point in line)
    step5_ok = (
        step4_ok
        and len(lines) == PINS["recovered_lines"]
        and all(len(set(line)) == 3 for line in lines)
        and len(pair_cover) == PINS["point_pairs"]
        and set(pair_cover.values()) == {1}
        and set(point_cover.values()) == {PINS["lines_through_a_point"]}
    )
    steps.append(
        {
            "step": 5,
            "statement": (
                "adjoining the class to its three pairs deduplicates globally to "
                "exactly seven three-point lines, with every one of the 21 point "
                "pairs on exactly one line"
            ),
            "holds": step5_ok,
            "line_count": len(lines),
            "covered_point_pairs": len(pair_cover),
            "pair_multiplicities": sorted(set(pair_cover.values())),
            "lines_through_a_point": sorted(set(point_cover.values())),
        }
    )

    return Reconstruction(
        steps=tuple(steps),
        classes=classes,
        class_of_habitat=tuple(class_of_habitat),
        omitted_left=tuple(omitted_left),
        omitted_right=tuple(omitted_right),
        left_to_right=tuple(left_to_right),
        class_of_left=tuple(class_of_left),
        support_pairs=tuple(support_pairs),
        lines=lines,
        valid=step5_ok,
    )


# ---------------------------------------------------------------------------
# Gate 0 section 3.1 step 6 — the plane's symmetry and certified equivalence
# ---------------------------------------------------------------------------

def line_preserving_bijections(
    source: tuple[tuple[int, int, int], ...],
    source_points: tuple[int, ...],
    target: tuple[tuple[int, int, int], ...],
    target_points: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Every bijection ``source_points -> target_points`` carrying lines to lines.

    Exhaustive over all ``7! = 5040`` bijections. This is the finite symmetry
    search ``011.01`` section 5 asks for; no continuous alignment is fitted.
    """

    if len(source_points) != len(target_points):
        return ()
    if len(source_points) != N_AXES or not source or not target:
        # The search is only defined on a seven-point line system. Refusing here
        # rather than enumerating n! keeps a failed reconstruction -- the control's,
        # which yields 14 singleton classes and no lines -- from turning a guard
        # into a combinatorial explosion.
        return ()
    target_set = {frozenset(line) for line in target}
    out: list[tuple[int, ...]] = []
    for image in itertools.permutations(target_points):
        mapping = dict(zip(source_points, image, strict=True))
        if all(frozenset(mapping[p] for p in line) in target_set for line in source):
            out.append(image)
    return tuple(out)


def plane_audit(reconstruction: Reconstruction) -> dict:
    """Section 3.1 step 6: ``|Aut| = 168`` and equivalence to the certified plane.

    The certified reference is ``topographo.core.f2_groups.LINES3`` with the
    ``gl_3_2`` collineation group. It is a *check target*, never an input.
    """

    points = tuple(range(len(reconstruction.classes)))
    automorphisms = line_preserving_bijections(
        reconstruction.lines, points, reconstruction.lines, points
    )
    certified = tuple(tuple(line) for line in groups.LINES3)
    isomorphisms = line_preserving_bijections(
        reconstruction.lines, points, certified, groups.POINTS3
    )
    return {
        "recovered_lines": [list(line) for line in reconstruction.lines],
        "automorphism_order": pin(
            PINS["recovered_plane_automorphism_order"],
            len(automorphisms),
            "exhaustive over all 5040 bijections of the recovered point classes",
        ),
        "certified_reference": {
            "source": "topographo.core.f2_groups.LINES3 / gl_3_2",
            "lines": [list(line) for line in certified],
            "collineation_order": pin(
                PINS["recovered_plane_automorphism_order"],
                len(groups.gl_3_2()),
                "topographo.core.f2_groups.gl_3_2",
            ),
        },
        "isomorphisms_to_certified_plane": len(isomorphisms),
        "equivalent_to_certified_plane": len(isomorphisms) > 0,
        "equivalence_is_up_to_GL_3_2": (
            "the recovered plane is judged up to its natural automorphism group; "
            "no canonical numeric FFF label is required or produced"
        ),
        "witness_isomorphism": list(isomorphisms[0]) if isomorphisms else None,
    }


def hidden_label_agreement(
    observation: Observation, reconstruction: Reconstruction
) -> dict:
    """Does the anonymously recovered structure agree with the certified labels?

    This is the only place the ``hidden_*`` fields meet the reconstruction, and
    it runs *after* the reconstruction is complete. ``011.01`` section 2.1
    permits exactly this use.
    """

    class_points = []
    for members in reconstruction.classes:
        points = {observation.hidden_point[m] for m in members}
        class_points.append(sorted(points))
    classes_are_certified_points = all(len(p) == 1 for p in class_points)
    point_of_class = tuple(p[0] for p in class_points) if classes_are_certified_points else ()

    mate_hits = 0
    for habitat in range(observation.n_habitats):
        mate_hits += reconstruction.mate(habitat) == observation.certified_mate(habitat)

    completion_hits = 0
    completion_total = 0
    for first, second in itertools.combinations(range(observation.n_habitats), 2):
        if observation.hidden_point[first] == observation.hidden_point[second]:
            continue
        completion_total += 1
        completion_hits += reconstruction.completion(first, second) == tuple(
            sorted(observation.certified_completion(first, second))
        )

    lines_as_points = None
    lines_are_xor_lines = None
    if classes_are_certified_points:
        lines_as_points = sorted(
            tuple(sorted(point_of_class[c] for c in line)) for line in reconstruction.lines
        )
        lines_are_xor_lines = all(
            line[0] ^ line[1] ^ line[2] == 0 for line in lines_as_points
        )

    return {
        "each_recovered_class_is_one_certified_fano_point": classes_are_certified_points,
        "recovered_class_to_certified_point": list(point_of_class),
        "mate": pin(
            observation.n_habitats,
            mate_hits,
            "reconstruction.mate vs the certified same-P habitat",
        ),
        "completion": pin(
            PINS["completion_queries_per_namespace"],
            completion_hits,
            "reconstruction.completion vs the certified P1 XOR P2 habitat pair",
        ),
        "completion_queries": completion_total,
        "recovered_lines_in_certified_labels": (
            [list(line) for line in lines_as_points] if lines_as_points else None
        ),
        "every_recovered_line_is_an_xor_line": lines_are_xor_lines,
        "certified_lines": [list(line) for line in groups.LINES3],
        "recovered_line_set_equals_certified": (
            lines_as_points == sorted(tuple(line) for line in groups.LINES3)
            if lines_as_points
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Gate 0 — the automorphism group of the observed anonymous graph
# ---------------------------------------------------------------------------

def node_map_is_automorphism(observation: Observation, node_map: tuple[int, ...]) -> bool:
    """Mechanical check: ``node_map`` is a type-respecting graph automorphism."""

    if sorted(node_map) != list(range(observation.n_nodes)):
        return False
    edges = set(observation.edges())
    for a, b in edges:
        image = tuple(sorted((node_map[a], node_map[b])))
        if image not in edges:
            return False
    return True


def class_support_map(
    observation: Observation, reconstruction: Reconstruction
) -> tuple[dict[int, int], ...]:
    """Per recovered class, the map ``left slot -> right slot`` of its Events.

    Both habitats of a class induce the *same* map -- an anonymous fact this
    function verifies rather than assumes, and the reason the sign choice is free
    while the axis action is not.
    """

    left_of = observation.left_of_event()
    right_of = observation.right_of_event()
    out: list[dict[int, int]] = []
    for members in reconstruction.classes:
        merged: dict[int, int] = {}
        for member in members:
            local: dict[int, int] = {}
            for event in observation.events_of_habitat()[member]:
                if left_of[event] in local:
                    raise AssertionError("a habitat repeats a left axis")
                local[left_of[event]] = right_of[event]
            if not merged:
                merged = local
            elif merged != local:
                raise AssertionError(
                    "the two habitats of a recovered class induce different "
                    "left->right support maps"
                )
        out.append(merged)
    return tuple(out)


def _forced_extension(
    observation: Observation,
    reconstruction: Reconstruction,
    support_maps: tuple[dict[int, int], ...],
    left_image: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Given an L-image, the forced class map and R-image, or ``None``.

    An automorphism sends the habitat omitting ``l`` to a habitat omitting
    ``left_image[l]``, so the L-image forces the class map; the class map then
    forces the R-image through the recovered omitted-R assignment. The Event-level
    constraint is what actually bites: the image of the Event of class ``c`` with
    left axis ``l`` is the Event of class ``tau(c)`` with left axis ``pi(l)``, so
    its right axis must be ``rho`` of the original's. Nothing is assumed about the
    algebra; every map is read off the recovered structure.
    """

    class_of_left = reconstruction.class_of_left
    left_of_class = [-1] * len(reconstruction.classes)
    right_of_class = [-1] * len(reconstruction.classes)
    for habitat, index in enumerate(reconstruction.class_of_habitat):
        left_of_class[index] = reconstruction.omitted_left[habitat]
        right_of_class[index] = reconstruction.omitted_right[habitat]

    class_image = [-1] * len(reconstruction.classes)
    for index in range(len(reconstruction.classes)):
        class_image[index] = class_of_left[left_image[left_of_class[index]]]
    if sorted(class_image) != list(range(len(reconstruction.classes))):
        return None

    right_image = [-1] * observation.n_right
    for index in range(len(reconstruction.classes)):
        right_image[right_of_class[index]] = right_of_class[class_image[index]]
    if sorted(right_image) != list(range(observation.n_right)):
        return None

    for index, support in enumerate(support_maps):
        target = support_maps[class_image[index]]
        for left, right in support.items():
            image_left = left_image[left]
            if image_left not in target:
                return None
            if target[image_left] != right_image[right]:
                return None
    return tuple(class_image), tuple(right_image)


def _event_key_table(observation: Observation) -> dict[tuple[int, int], int]:
    """``(habitat, left) -> event``. Within a habitat the L axes are distinct."""

    habitat_of = observation.habitat_of_event()
    left_of = observation.left_of_event()
    table: dict[tuple[int, int], int] = {}
    for event in range(observation.n_events):
        key = (habitat_of[event], left_of[event])
        if key in table:
            raise AssertionError(f"two Events share the key {key}")
        table[key] = event
    return table


def type_respecting_automorphisms(
    observation: Observation, reconstruction: Reconstruction
) -> tuple[tuple[tuple[int, ...], ...], dict]:
    """Every automorphism fixing the L and R namespaces setwise, enumerated.

    Stage 1 filters all ``7!`` L-images down to those that extend at all.
    Stage 2 enumerates the ``2**7`` sign choices for each survivor and checks
    every resulting node map mechanically. The closure argument is recorded
    alongside: an automorphism must preserve the degree-determined type
    partition, the L-image forces the class map and the R-image, and the class
    map plus one sign bit per class forces the habitat and Event maps.
    """

    habitat_of = observation.habitat_of_event()
    left_of = observation.left_of_event()
    event_key = _event_key_table(observation)
    support_maps = class_support_map(observation, reconstruction)
    n_classes = len(reconstruction.classes)

    survivors: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for image in itertools.permutations(range(observation.n_left)):
        extension = _forced_extension(observation, reconstruction, support_maps, image)
        if extension is None:
            continue
        class_image, right_image = extension
        survivors.append((image, class_image, right_image))

    found: list[tuple[int, ...]] = []
    per_survivor: list[int] = []
    for image, class_image, right_image in survivors:
        count = 0
        for bits in itertools.product((0, 1), repeat=n_classes):
            habitat_image = [-1] * observation.n_habitats
            for index, members in enumerate(reconstruction.classes):
                targets = reconstruction.classes[class_image[index]]
                for position, member in enumerate(sorted(members)):
                    habitat_image[member] = targets[(position + bits[index]) % 2]
            if sorted(habitat_image) != list(range(observation.n_habitats)):
                continue
            event_image = [-1] * observation.n_events
            broken = False
            for event in range(observation.n_events):
                key = (habitat_image[habitat_of[event]], image[left_of[event]])
                target = event_key.get(key)
                if target is None:
                    broken = True
                    break
                event_image[event] = target
            if broken or sorted(event_image) != list(range(observation.n_events)):
                continue
            node_map = (
                tuple(habitat_image)
                + tuple(observation.event_node(e) for e in event_image)
                + tuple(observation.left_node(left) for left in image)
                + tuple(observation.right_node(right) for right in right_image)
            )
            if node_map_is_automorphism(observation, node_map):
                found.append(node_map)
                count += 1
        per_survivor.append(count)

    census = {
        "left_images_enumerated": math.factorial(observation.n_left),
        "left_images_that_extend": pin(
            PINS["left_collineations"],
            len(survivors),
            "stage 1: exhaustive over all 5040 L-images of the anonymous graph",
        ),
        "automorphisms_per_left_image": pin(
            [PINS["sign_choices"]],
            sorted(set(per_survivor)),
            "stage 2: exhaustive over the 2**7 per-class sign choices",
        ),
        "type_respecting_order": len(found),
        "closure_argument": (
            "the degree spectrum separates H (6) and E (3) from L and R (12), so "
            "any automorphism preserves {H}, {E} and {L, R}; an automorphism "
            "fixing L and R setwise is determined by its L-image, which forces "
            "the class map and hence the R-image, plus one sign bit per class, "
            "which forces the habitat map and hence the Event map because "
            "(habitat, left) is a key on Events"
        ),
    }
    return tuple(sorted(found)), census


def left_right_swap(
    observation: Observation, reconstruction: Reconstruction
) -> tuple[int, ...] | None:
    """The candidate automorphism exchanging the L and R namespaces.

    An L node and an R node correspond when the same recovered point class omits
    both. Events follow: the image of the Event with support ``(l, r)`` is the
    Event of the same habitat whose left axis is the partner of ``r``.
    """

    if not reconstruction.valid:
        return None
    habitat_of = observation.habitat_of_event()
    right_of = observation.right_of_event()
    event_key = _event_key_table(observation)

    right_to_left = [-1] * observation.n_right
    for left, right in enumerate(reconstruction.left_to_right):
        right_to_left[right] = left

    event_image = [-1] * observation.n_events
    for event in range(observation.n_events):
        key = (habitat_of[event], right_to_left[right_of[event]])
        target = event_key.get(key)
        if target is None:
            return None
        event_image[event] = target
    if sorted(event_image) != list(range(observation.n_events)):
        return None

    node_map = (
        tuple(range(observation.n_habitats))
        + tuple(observation.event_node(e) for e in event_image)
        + tuple(
            observation.right_node(reconstruction.left_to_right[left])
            for left in range(observation.n_left)
        )
        + tuple(observation.left_node(right_to_left[right]) for right in range(observation.n_right))
    )
    if not node_map_is_automorphism(observation, node_map):
        return None
    return node_map


def compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(second[node] for node in first)


def automorphism_census(
    observation: Observation, reconstruction: Reconstruction
) -> tuple[tuple[tuple[int, ...], ...], dict]:
    """``|Aut|`` of the observed graph, and the exact answer-uniqueness proof."""

    type_respecting, census = type_respecting_automorphisms(observation, reconstruction)
    swap = left_right_swap(observation, reconstruction)
    if swap is None:
        full = type_respecting
        census["left_right_swap_is_an_automorphism"] = False
    else:
        census["left_right_swap_is_an_automorphism"] = True
        full = tuple(sorted(set(type_respecting) | {compose(g, swap) for g in type_respecting}))
    census["order"] = pin(
        PINS["observed_graph_automorphism_order"],
        len(full),
        "type-respecting subgroup times the L/R swap coset",
    )
    census["decomposition"] = (
        f"{len(type_respecting)} type-respecting x 2 (L/R swap coset) = {len(full)}"
    )
    return full, census


def answer_uniqueness(
    observation: Observation, automorphisms: tuple[tuple[int, ...], ...]
) -> dict:
    """Are the two answers forced by the observation? Checked over all of ``Aut``.

    A query's answer is determined by the anonymous graph iff it is equivariant
    under every automorphism: if some automorphism fixed the query but moved the
    answer, two distinct answers would be indistinguishable from the
    observation. This is checked on every automorphism and every query.
    """

    mate_broken = 0
    completion_broken = 0
    completion_pairs = [
        (first, second)
        for first, second in itertools.combinations(range(observation.n_habitats), 2)
        if observation.hidden_point[first] != observation.hidden_point[second]
    ]
    for node_map in automorphisms:
        for habitat in range(observation.n_habitats):
            image = node_map[habitat]
            if node_map[observation.certified_mate(habitat)] != observation.certified_mate(image):
                mate_broken += 1
        for first, second in completion_pairs:
            image = (node_map[first], node_map[second])
            wanted = {node_map[h] for h in observation.certified_completion(first, second)}
            if wanted != set(observation.certified_completion(*image)):
                completion_broken += 1
    return {
        "automorphisms_checked": len(automorphisms),
        "mate_equivariance_failures": mate_broken,
        "completion_equivariance_failures": completion_broken,
        "mate_answer_is_forced_by_the_observation": mate_broken == 0,
        "completion_answer_is_forced_by_the_observation": completion_broken == 0,
        "statement": (
            "every automorphism of the anonymous graph carries each query to a "
            "query and its certified answer to the certified answer, so both "
            "answers are functions of the observation alone and the exact "
            "main-arm ceiling is 1.0000"
        ),
    }


# ---------------------------------------------------------------------------
# Gate 0 section 3.3 — the localized control's exact information ceiling
# ---------------------------------------------------------------------------

def component_of_node(observation: Observation) -> tuple[int, ...]:
    """Connected component index per node, by breadth-first search."""

    adjacency = observation.adjacency()
    seen = [-1] * observation.n_nodes
    label = 0
    for start in range(observation.n_nodes):
        if seen[start] != -1:
            continue
        frontier = [start]
        seen[start] = label
        while frontier:
            node = frontier.pop()
            for peer in adjacency[node]:
                if seen[peer] == -1:
                    seen[peer] = label
                    frontier.append(peer)
        label += 1
    return tuple(seen)


def component_swap(
    observation: Observation, components: tuple[int, ...], first: int, second: int
) -> tuple[int, ...] | None:
    """The node map exchanging two habitat components, or ``None``.

    Built by matching the two components' local structure: habitat to habitat,
    then Event to Event in the local order, then each Event's private L and R.
    """

    if first == second:
        return tuple(range(observation.n_nodes))
    left_of = observation.left_of_event()
    right_of = observation.right_of_event()
    node_map = list(range(observation.n_nodes))
    for source, target in ((first, second), (second, first)):
        node_map[source] = target
        events_source = observation.events_of_habitat()[source]
        events_target = observation.events_of_habitat()[target]
        if len(events_source) != len(events_target):
            return None
        for a, b in zip(events_source, events_target, strict=True):
            node_map[observation.event_node(a)] = observation.event_node(b)
            node_map[observation.left_node(left_of[a])] = observation.left_node(left_of[b])
            node_map[observation.right_node(right_of[a])] = observation.right_node(
                right_of[b]
            )
    del components
    if not node_map_is_automorphism(observation, tuple(node_map)):
        return None
    return tuple(node_map)


def localized_ceiling_census(observation: Observation) -> dict:
    """The exact ceilings for the control, by explicit orbit witnesses.

    ``011.01`` section 3.3 forbids substituting empirical chance. The
    computation here is a finite proof: for every query and every ordered pair
    of candidate answers, an automorphism of the control graph is *constructed*
    that fixes the query and carries one candidate to the other. Every scored
    candidate therefore lies in a single orbit, so any equivariant scorer assigns
    them equal scores and cannot do better than uniform choice.
    """

    components = component_of_node(observation)
    component_count = len(set(components))
    sizes = sorted(Counter(components).values())

    mate_orbit_complete = True
    mate_witnesses = 0
    for query in range(observation.n_habitats):
        candidates = [h for h in range(observation.n_habitats) if h != query]
        anchor = candidates[0]
        for other in candidates[1:]:
            node_map = component_swap(observation, components, anchor, other)
            if node_map is None or node_map[query] != query:
                mate_orbit_complete = False
                break
            mate_witnesses += 1
        if not mate_orbit_complete:
            break

    completion_orbit_complete = True
    completion_witnesses = 0
    sample_pairs = [
        (first, second)
        for first, second in itertools.combinations(range(observation.n_habitats), 2)
        if observation.hidden_point[first] != observation.hidden_point[second]
    ]
    for first, second in sample_pairs:
        candidates = [
            h for h in range(observation.n_habitats) if h not in (first, second)
        ]
        anchor = candidates[0]
        for other in candidates[1:]:
            node_map = component_swap(observation, components, anchor, other)
            if node_map is None or node_map[first] != first or node_map[second] != second:
                completion_orbit_complete = False
                break
            completion_witnesses += 1
        if not completion_orbit_complete:
            break

    mate_candidates = observation.n_habitats - 1
    completion_candidates = observation.n_habitats - 2
    completion_sets = math.comb(completion_candidates, 2)
    return {
        "components": pin(
            PINS["localized_components"],
            component_count,
            "breadth-first search over the control graph",
        ),
        "component_sizes": sizes,
        "components_are_isomorphic": len(set(sizes)) == 1,
        "mate": {
            "scored_candidates": mate_candidates,
            "orbit_witnesses": mate_witnesses,
            "single_orbit": mate_orbit_complete,
            "exact_ceiling": rd(1.0 / mate_candidates),
            "statement": (
                f"all {mate_candidates} scored habitats lie in one orbit of the "
                "automorphisms fixing the query, so every equivariant scorer ties "
                f"and the exact ceiling is 1/{mate_candidates}"
            ),
        },
        "completion": {
            "scored_candidates": observation.n_habitats,
            "eligible_candidates": completion_candidates,
            "distinct_target_sets": completion_sets,
            "orbit_witnesses": completion_witnesses,
            "single_orbit": completion_orbit_complete,
            "exact_ceiling": rd(1.0 / completion_sets),
            "statement": (
                f"the {completion_candidates} habitats outside the query lie in "
                "one orbit and the automorphisms fixing the query induce the full "
                f"symmetric group on them, so all {completion_sets} two-element "
                "subsets are indistinguishable and the exact ceiling for exact-set "
                f"recovery is 1/{completion_sets}"
            ),
        },
        "no_empirical_substitution": (
            "both ceilings are exact rational numbers from constructed orbit "
            "witnesses, not measured chance"
        ),
    }


def localized_locality_audit(main: Observation, control: Observation) -> dict:
    """The control differs only in cross-habitat L/R sharing. Checked, not claimed."""

    main_left = main.left_of_event()
    main_right = main.right_of_event()
    control_left = control.left_of_event()
    control_right = control.right_of_event()

    same_membership = main.he == control.he
    local_patterns_agree = True
    for habitat, events in enumerate(main.events_of_habitat()):
        if control.events_of_habitat()[habitat] != events:
            local_patterns_agree = False
            break
        main_left_partition = [
            sorted(other for other in events if main_left[other] == main_left[event])
            for event in events
        ]
        control_left_partition = [
            sorted(other for other in events if control_left[other] == control_left[event])
            for event in events
        ]
        main_right_partition = [
            sorted(other for other in events if main_right[other] == main_right[event])
            for event in events
        ]
        control_right_partition = [
            sorted(
                other for other in events if control_right[other] == control_right[event]
            )
            for event in events
        ]
        if (
            main_left_partition != control_left_partition
            or main_right_partition != control_right_partition
        ):
            local_patterns_agree = False
            break

    shared_main = sum(
        1
        for left in range(main.n_left)
        if len({main.habitat_of_event()[e] for e in range(main.n_events) if main_left[e] == left})
        > 1
    )
    shared_control = sum(
        1
        for left in range(control.n_left)
        if len(
            {
                control.habitat_of_event()[e]
                for e in range(control.n_events)
                if control_left[e] == left
            }
        )
        > 1
    )
    return {
        "habitat_membership_identical": same_membership,
        "within_habitat_left_and_right_patterns_identical": local_patterns_agree,
        "left_nodes_shared_across_habitats": {"main": shared_main, "control": shared_control},
        "degree_spectrum": {"main": main.degree_spectrum(), "control": control.degree_spectrum()},
        "statement": (
            "the control keeps every habitat's H--E--L/R incidence pattern and "
            "within-habitat degrees bit-identical and removes only cross-habitat "
            "L/R identity; the global L/R degree falls from 12 to 1 because that "
            "sharing is exactly what is under test"
        ),
    }


# ---------------------------------------------------------------------------
# frozen relabelling namespaces and queries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Namespace:
    """One frozen opaque relabelling of all four node namespaces.

    ``habitat[slot] = node``. Independent Lehmer permutations per namespace and
    per type, keyed by a digest of the label, so nothing correlates across
    splits and no opaque name acquires stable semantics.
    """

    label: str
    split: str
    arm: str
    habitat: tuple[int, ...]
    event: tuple[int, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]

    def as_row(self) -> list:
        return [
            self.label,
            self.split,
            self.arm,
            list(self.habitat),
            list(self.event),
            list(self.left),
            list(self.right),
        ]

    def sha256(self) -> str:
        return digest(self.as_row())

    def node_permutation(self, observation: Observation) -> tuple[int, ...]:
        """The induced permutation of node indices, ``canonical -> opaque``.

        Type-respecting by construction: habitat nodes land on habitat nodes,
        Events on Events, left axes on left axes, right on right. The four-way
        type one-hot is therefore the same function of the node index in every
        namespace, and the *only* thing a namespace changes is the incidence
        matrix -- which is exactly the "fresh opaque name" ``011.01`` section 6
        asks for.
        """

        out = [0] * observation.n_nodes
        for slot, node in enumerate(self.habitat):
            out[observation.habitat_node(slot)] = observation.habitat_node(node)
        for slot, node in enumerate(self.event):
            out[observation.event_node(slot)] = observation.event_node(node)
        for slot, node in enumerate(self.left):
            out[observation.left_node(slot)] = observation.left_node(node)
        for slot, node in enumerate(self.right):
            out[observation.right_node(slot)] = observation.right_node(node)
        return tuple(out)

    def relabelled_edges(self, observation: Observation) -> tuple[tuple[int, int], ...]:
        """The observation's edge set in this namespace's opaque node names."""

        perm = self.node_permutation(observation)
        return tuple(
            sorted(tuple(sorted((perm[a], perm[b]))) for a, b in observation.edges())
        )


def build_namespaces(observation: Observation) -> tuple[Namespace, ...]:
    out: list[Namespace] = []
    for split, count in NAMESPACE_SPLITS:
        for index in range(count):
            label = f"{observation.arm}/{split}/{index:02d}"
            out.append(
                Namespace(
                    label=label,
                    split=split,
                    arm=observation.arm,
                    habitat=lehmer_permutation(observation.n_habitats, f"011/H/{label}"),
                    event=lehmer_permutation(observation.n_events, f"011/E/{label}"),
                    left=lehmer_permutation(observation.n_left, f"011/L/{label}"),
                    right=lehmer_permutation(observation.n_right, f"011/R/{label}"),
                )
            )
    return tuple(out)


@dataclass(frozen=True)
class Query:
    """One scored query, in opaque node coordinates."""

    query_id: str
    namespace: str
    split: str
    kind: str
    query_nodes: tuple[int, ...]
    target_nodes: tuple[int, ...]
    scored_nodes: tuple[int, ...]

    def as_row(self) -> list:
        return [
            self.query_id,
            self.namespace,
            self.kind,
            list(self.query_nodes),
            list(self.target_nodes),
            list(self.scored_nodes),
        ]

    def sha256(self) -> str:
        return digest(self.as_row())


def build_queries(observation: Observation, namespace: Namespace) -> tuple[Query, ...]:
    """All 14 mate and 84 completion queries of one namespace.

    ``011.01`` section 4.1 scores the other thirteen habitat nodes; section 4.2
    scores all fourteen. Both are followed exactly.
    """

    out: list[Query] = []
    node_of = namespace.habitat
    all_nodes = tuple(sorted(node_of))
    for habitat in range(observation.n_habitats):
        query = node_of[habitat]
        out.append(
            Query(
                query_id=f"{namespace.label}/mate/{habitat:02d}",
                namespace=namespace.label,
                split=namespace.split,
                kind="mate",
                query_nodes=(query,),
                target_nodes=(node_of[observation.certified_mate(habitat)],),
                scored_nodes=tuple(n for n in all_nodes if n != query),
            )
        )
    for first, second in itertools.combinations(range(observation.n_habitats), 2):
        if observation.hidden_point[first] == observation.hidden_point[second]:
            continue
        targets = observation.certified_completion(first, second)
        out.append(
            Query(
                query_id=f"{namespace.label}/completion/{first:02d}-{second:02d}",
                namespace=namespace.label,
                split=namespace.split,
                kind="completion",
                query_nodes=tuple(sorted((node_of[first], node_of[second]))),
                target_nodes=tuple(sorted(node_of[t] for t in targets)),
                scored_nodes=all_nodes,
            )
        )
    return tuple(out)


def build_episodes(
    observation: Observation, namespaces: tuple[Namespace, ...]
) -> tuple[Query, ...]:
    out: list[Query] = []
    for namespace in namespaces:
        out.extend(build_queries(observation, namespace))
    return tuple(out)


def episode_manifest(
    observation: Observation, namespaces: tuple[Namespace, ...], queries: tuple[Query, ...]
) -> dict:
    counts = Counter((q.split, q.kind) for q in queries)
    per_split = Counter(namespace.split for namespace in namespaces)
    return {
        "arm": observation.arm,
        "namespaces": {
            "total": len(namespaces),
            "per_split": dict(sorted(per_split.items())),
            "permutation_source": (
                "lehmer_permutation(n, key) with key a sha256 digest of the "
                "namespace label; no RNG implementation is involved"
            ),
            "digests": [
                {"label": namespace.label, "sha256": namespace.sha256()}
                for namespace in namespaces
            ],
        },
        "queries": {
            "total": len(queries),
            "per_split_and_kind": {
                f"{split}/{kind}": count for (split, kind), count in sorted(counts.items())
            },
            "mate_scored_candidates": observation.n_habitats - 1,
            "completion_scored_candidates": observation.n_habitats,
            "completion_target_size": 2,
            "manifest_sha256": digest([q.as_row() for q in queries]),
        },
        "no_stable_opaque_name": (
            "each namespace draws four independent permutations, so no opaque "
            "index correlates with a hidden point or an answer across episodes"
        ),
    }


def namespace_soundness(
    observation: Observation, namespaces: tuple[Namespace, ...]
) -> dict:
    """No opaque name may correlate with a hidden point or answer across splits."""

    by_node: dict[int, Counter] = defaultdict(Counter)
    for namespace in namespaces:
        for slot, node in enumerate(namespace.habitat):
            by_node[node][observation.hidden_point[slot]] += 1
    worst = 0.0
    for node, tally in by_node.items():
        del node
        worst = max(worst, max(tally.values()) / sum(tally.values()))
    distinct = len({namespace.node_permutation(observation) for namespace in namespaces})
    return {
        "namespaces": len(namespaces),
        "distinct_node_permutations": distinct,
        "all_namespaces_distinct": distinct == len(namespaces),
        "max_share_of_one_hidden_point_at_any_habitat_node": rd(worst),
        "uniform_share": rd(1.0 / N_AXES),
        "statement": (
            "each habitat node carries every hidden point class at close to the "
            "uniform 1/7 rate across the frozen namespaces, so an opaque index is "
            "not a usable proxy for the answer"
        ),
    }


# ---------------------------------------------------------------------------
# provenance and audit
# ---------------------------------------------------------------------------

def provenance() -> dict:
    live = fips_basic.TABLE_SHA256
    return {
        "topographo_version": importlib.metadata.version("topographo"),
        "base_commit": BASE_COMMIT,
        "executed_task": EXECUTED_TASK,
        "prior_result": PRIOR_RESULT,
        "table_sha256": {
            "expected": EXPECTED_TABLE_SHA256,
            "observed": live,
            "agrees": live == EXPECTED_TABLE_SHA256
            and live == fips_basic.EXPECTED_TABLE_SHA256,
        },
        "observation_sources": [
            "task.build_dataset (frozen 84-Event catalogue and habitat partition)",
            "task.Catalogue.events_of_habitat (the hidden (P, delta) partition)",
            "fano_task.axes_of (two support axis indices; both signs discarded)",
        ],
        "certified_reference_only": [
            "topographo.core.f2_groups.LINES3 / gl_3_2 (section 3.1 step 6)",
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def audit() -> dict:
    dataset = build_dataset()
    main = build_observation(dataset)
    control = build_localized_observation(main)

    catalogue_sha = dataset.catalogue.sha256()
    dataset_sha = dataset.sha256()

    reconstruction = reconstruct(main)
    plane = plane_audit(reconstruction)
    agreement = hidden_label_agreement(main, reconstruction)
    automorphisms, census = automorphism_census(main, reconstruction)
    uniqueness = answer_uniqueness(main, automorphisms)

    control_reconstruction = reconstruct(control)
    control_ceilings = localized_ceiling_census(control)
    locality = localized_locality_audit(main, control)

    namespaces = build_namespaces(main)
    queries = build_episodes(main, namespaces)
    control_namespaces = build_namespaces(control)
    control_queries = build_episodes(control, control_namespaces)

    degrees = main.degree_spectrum()
    pins = {
        "habitat_nodes": pin(PINS["habitat_nodes"], main.n_habitats, "observation"),
        "event_nodes": pin(PINS["event_nodes"], main.n_events, "observation"),
        "left_nodes": pin(PINS["left_nodes"], main.n_left, "observation"),
        "right_nodes": pin(PINS["right_nodes"], main.n_right, "observation"),
        "observed_nodes_main": pin(
            PINS["observed_nodes_main"], main.n_nodes, "observation"
        ),
        "habitat_degree": pin(
            [PINS["habitat_degree"]], degrees["habitat"], "degree spectrum"
        ),
        "event_degree": pin([PINS["event_degree"]], degrees["event"], "degree spectrum"),
        "left_degree": pin([PINS["left_degree"]], degrees["left"], "degree spectrum"),
        "right_degree": pin([PINS["right_degree"]], degrees["right"], "degree spectrum"),
        "recovered_point_classes": pin(
            PINS["recovered_point_classes"], len(reconstruction.classes), "step 2"
        ),
        "recovered_lines": pin(
            PINS["recovered_lines"], len(reconstruction.lines), "step 5"
        ),
        "recovered_plane_automorphism_order": plane["automorphism_order"],
        "observed_graph_automorphism_order": census["order"],
        "left_collineations": census["left_images_that_extend"],
        "mate_agreement": agreement["mate"],
        "completion_agreement": agreement["completion"],
        "localized_components": control_ceilings["components"],
        "catalogue_sha256": pin(
            EXPECTED_CATALOGUE_SHA256, catalogue_sha, "task.Catalogue.sha256"
        ),
        "dataset_sha256": pin(EXPECTED_DATASET_SHA256, dataset_sha, "task.Dataset.sha256"),
        "namespaces": pin(PINS["namespaces"], len(namespaces), "build_namespaces"),
        "queries_per_namespace": pin(
            PINS["queries_per_namespace"],
            len(queries) // len(namespaces),
            "build_queries",
        ),
        "localized_left_nodes": pin(
            PINS["localized_left_nodes"], control.n_left, "build_localized_observation"
        ),
        "localized_right_nodes": pin(
            PINS["localized_right_nodes"], control.n_right, "build_localized_observation"
        ),
    }

    ceilings = {
        "main_mate": pin(
            EXPECTED_CEILINGS["main_mate"],
            1.0 if uniqueness["mate_answer_is_forced_by_the_observation"] else None,
            "answer_uniqueness over all of Aut",
        ),
        "main_completion": pin(
            EXPECTED_CEILINGS["main_completion"],
            1.0 if uniqueness["completion_answer_is_forced_by_the_observation"] else None,
            "answer_uniqueness over all of Aut",
        ),
        "main_plane": pin(
            EXPECTED_CEILINGS["main_plane"],
            1.0 if plane["equivalent_to_certified_plane"] and reconstruction.valid else None,
            "reconstruct + plane_audit",
        ),
        "localized_mate": pin(
            rd(EXPECTED_CEILINGS["localized_mate"]),
            control_ceilings["mate"]["exact_ceiling"],
            "localized_ceiling_census orbit witnesses",
        ),
        "localized_completion": pin(
            rd(EXPECTED_CEILINGS["localized_completion"]),
            control_ceilings["completion"]["exact_ceiling"],
            "localized_ceiling_census orbit witnesses",
        ),
    }

    laws = {
        "reconstruction_step_1_omission": reconstruction.steps[0]["holds"],
        "reconstruction_step_2_classes": reconstruction.steps[1]["holds"],
        "reconstruction_step_3_bijection": reconstruction.steps[2]["holds"],
        "reconstruction_step_4_support_pairs": reconstruction.steps[3]["holds"],
        "reconstruction_step_5_lines": reconstruction.steps[4]["holds"],
        "reconstruction_valid": reconstruction.valid,
        "recovered_plane_has_order_168": plane["automorphism_order"]["agrees"],
        "recovered_plane_is_the_certified_plane": plane["equivalent_to_certified_plane"],
        "each_class_is_one_certified_fano_point": agreement[
            "each_recovered_class_is_one_certified_fano_point"
        ],
        "every_recovered_line_is_an_xor_line": bool(
            agreement["every_recovered_line_is_an_xor_line"]
        ),
        "recovered_line_set_equals_certified": bool(
            agreement["recovered_line_set_equals_certified"]
        ),
        "mate_recovered_on_all_14": agreement["mate"]["agrees"],
        "completion_recovered_on_all_84": agreement["completion"]["agrees"],
        "observed_automorphism_order_is_43008": census["order"]["agrees"],
        "left_right_swap_is_an_automorphism": census["left_right_swap_is_an_automorphism"],
        "mate_answer_is_forced": uniqueness["mate_answer_is_forced_by_the_observation"],
        "completion_answer_is_forced": uniqueness[
            "completion_answer_is_forced_by_the_observation"
        ],
        "control_does_not_identify_the_plane": not control_reconstruction.valid,
        "control_has_14_components": control_ceilings["components"]["agrees"],
        "control_mate_is_a_single_orbit": control_ceilings["mate"]["single_orbit"],
        "control_completion_is_a_single_orbit": control_ceilings["completion"][
            "single_orbit"
        ],
        "control_preserves_local_patterns": locality[
            "within_habitat_left_and_right_patterns_identical"
        ],
        "control_shares_no_axis_across_habitats": locality[
            "left_nodes_shared_across_habitats"
        ]["control"]
        == 0,
        "all_namespaces_distinct": namespace_soundness(main, namespaces)[
            "all_namespaces_distinct"
        ],
        "every_pin_agrees": all(entry["agrees"] for entry in pins.values()),
        "every_ceiling_agrees": all(entry["agrees"] for entry in ceilings.values()),
    }
    broken = sorted(name for name, holds in laws.items() if not holds)

    gate_passes = not broken
    return {
        "module": "fano_task",
        "purpose": (
            "011.01 Gate 0: the anonymous global Fano/FFF problem enumerated "
            "exactly, plus the frozen relabelling and query manifests"
        ),
        "executed_task": EXECUTED_TASK,
        "supplied": list(SUPPLIED),
        "withheld": list(WITHHELD),
        "pins": pins,
        "ceilings": ceilings,
        "observation": {
            "main": {
                "nodes": main.n_nodes,
                "typed_counts": {
                    "habitat": main.n_habitats,
                    "event": main.n_events,
                    "left": main.n_left,
                    "right": main.n_right,
                },
                "edges": len(main.edges()),
                "degree_spectrum": main.degree_spectrum(),
                "sha256": main.sha256(),
            },
            "localized_control": {
                "nodes": control.n_nodes,
                "typed_counts": {
                    "habitat": control.n_habitats,
                    "event": control.n_events,
                    "left": control.n_left,
                    "right": control.n_right,
                },
                "edges": len(control.edges()),
                "degree_spectrum": control.degree_spectrum(),
                "sha256": control.sha256(),
            },
            "node_layout": (
                "habitat slots first, then Event, then left-axis, then right-axis; "
                "the input tensor carries a four-way type one-hot and one query "
                "mark, and no index is ever a parameter address"
            ),
        },
        "gate_0": {
            "reconstruction": {
                "steps": [dict(step) for step in reconstruction.steps],
                "valid": reconstruction.valid,
                "sha256": reconstruction.sha256(),
                "recovered_classes": [list(c) for c in reconstruction.classes],
                "left_to_right_bijection": list(reconstruction.left_to_right),
            },
            "plane": plane,
            "hidden_label_agreement": agreement,
            "automorphism_census": census,
            "answer_uniqueness": uniqueness,
            "localized_control": {
                "reconstruction_valid": control_reconstruction.valid,
                "reconstruction_steps": [
                    dict(step) for step in control_reconstruction.steps
                ],
                "ceilings": control_ceilings,
                "locality": locality,
                "non_discriminating_check": {
                    "control_identifies_the_certified_plane": control_reconstruction.valid,
                    "statement": (
                        "011.01 section 3.3 requires stopping before training if "
                        "the localized control still identifies the certified "
                        "global plane. It does not: the control graph is 14 "
                        "isomorphic components and its reconstruction fails at "
                        "step 1"
                    ),
                },
            },
            "passes": gate_passes,
        },
        "episodes": {
            "main": episode_manifest(main, namespaces, queries),
            "localized_control": episode_manifest(
                control, control_namespaces, control_queries
            ),
            "namespace_soundness": namespace_soundness(main, namespaces),
        },
        "relational_laws": laws,
        "fences": list(FENCES),
        "digests": {
            "catalogue_sha256": catalogue_sha,
            "dataset_sha256": dataset_sha,
            "main_observation_sha256": main.sha256(),
            "control_observation_sha256": control.sha256(),
            "reconstruction_sha256": reconstruction.sha256(),
            "main_query_manifest_sha256": digest([q.as_row() for q in queries]),
            "control_query_manifest_sha256": digest(
                [q.as_row() for q in control_queries]
            ),
        },
        "provenance": provenance(),
        "verdict": {
            "broken_laws": broken,
            "agrees": gate_passes,
            "statement": (
                "Gate 0 PASSES: the anonymous typed observation identifies the "
                "certified seven-point Fano quotient up to GL(3,2), the exact "
                "main-arm ceiling is 1.0000 for both queries, and the localized "
                "control's exact ceilings are 1/13 and 1/66 from constructed "
                "orbit witnesses"
            )
            if gate_passes
            else (
                "Gate 0 FAILS; 011.01 section 3 forbids training. Broken laws: "
                + ", ".join(broken)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive and compare byte for byte instead of writing",
    )
    args = parser.parse_args()

    payload = audit()
    text = render(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: {OUTPUT} does not exist; run without --check")
        if OUTPUT.read_text() != text:
            raise SystemExit(f"FAIL: {OUTPUT} is not byte-identical to the rebuild")
        if not payload["verdict"]["agrees"]:
            raise SystemExit(f"FAIL: {payload['verdict']['statement']}")
        print(f"PASS: {OUTPUT.name} re-derives byte for byte")
        print(f"PASS: {payload['verdict']['statement']}")
        return
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")
    print(payload["verdict"]["statement"])


if __name__ == "__main__":
    main()

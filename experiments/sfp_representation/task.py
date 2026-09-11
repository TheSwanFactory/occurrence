"""The frozen admit-and-force dataset over ordered Event pairs (Issue 009, Task 4).

This module builds the single, frozen, exhaustive supervision pool for Issue 009:
all ``84 * 84 = 7056`` **ordered** pairs of certified basic Events, each carrying
an ``ADMIT`` / ``NONADMISSION`` label, a forced-third target (or an explicit
bottom sentinel), a non-admission class, and the habitat / Fano-point / sign
coordinates that the downstream folds and representation arms need.

THE LABEL RULE — the scientific crux
------------------------------------
For an ordered pair ``(a, b)`` of basic Events::

    if a != b and a, b share exactly one certified cyclic block:
        ADMIT,  target = the unique third Event c of that block
    else:
        NONADMISSION (bottom)

**Labels are generated from the certified native FIPS relation**
``topographo.ssd.fips_basic.admissible`` / ``.third``. That relation is the
authority for ground truth here.

**``sfp.ExactSfpCircuit`` is NEVER a label source.** The SFP circuit is a
representation / reference arm: it is the thing later arms are measured
*against*, so letting it define ground truth would make Arm B's success
unfalsifiable. The boundary is enforced structurally, not by comment:

* label generation lives in :func:`label_pair` and :func:`build_dataset`, which
  take a :class:`Catalogue` and integer indices and **do not accept a circuit
  object at all** — there is no parameter through which a circuit could enter;
* ``sfp`` is not imported at module scope. The only import of ``sfp`` in this
  file is a function-local import inside :func:`sfp_cross_check`, which runs
  *after* the dataset is already frozen and may only *read* it;
* :func:`sfp_cross_check` returns a diagnostic report and never returns or
  mutates a label. It is recorded in the artifact under
  ``reference_oracle_diagnostic``, explicitly flagged as not the label source.

Non-admission classes
---------------------
Carried on every record and reported separately::

    REPEATED               a == b
    SAME_HABITAT_DISJOINT  same (P, delta), but the two K4 edges share no block
    CROSS_HABITAT          different (P, delta)

Independently predicted counts (pins, verified — never adjusted to fit)
----------------------------------------------------------------------
In a habitat's ``K4`` (blocks = vertices, Events = edges) two distinct Events
either share exactly one vertex (adjacent: ADMIT; each edge has 4 adjacent
edges, so ``6 * 4 = 24`` ordered per habitat, ``* 14 = 336``) or share none
(opposite: ``6 * 1 = 6`` ordered per habitat, ``* 14 = 84``)::

    7056  ordered pairs total          (84 * 84)
     336  ADMIT
      84  REPEATED                     (one per Event)
      84  SAME_HABITAT_DISJOINT        (14 habitats * 6 ordered opposite pairs)
    6552  CROSS_HABITAT
     504  same-habitat ordered pairs   (14 * 36 = 84 + 336 + 84)

Positive role neutrality
------------------------
Every certified three-Event block supports all pair completions, verified
mechanically: for each of the 56 blocks all ``3 * 2 = 6`` ordered pairs of
distinct members are ADMIT and yield the remaining member as target
(``56 * 6 = 336``, matching the ADMIT count exactly); both input orders are
present for every admitted unordered pair with ``target(a, b) == target(b, a)``;
and all three missing-member choices occur for every block. No privileged
direction and no separate structure for one completion is built.

The bottom fence — structurally enforced
----------------------------------------
Bottom is an explicit distinct symbol, never an algebraic zero. It is NOT
``FFF=000``, NOT ``pp=00``, NOT Event index ``0``. It is the sentinel pair
(:data:`BOTTOM_INDEX` ``= -1``, :data:`BOTTOM_SYMBOL` ``= "BOTTOM"``), and the
audit asserts that no admitted record carries it, no non-admitted record carries
a real Event index, and Event index ``0`` is a genuine certified Event that also
occurs as a real forced-third target.

Correctness criterion
---------------------
Exact certified Event identity via ``topographo.ssd.projective.equivalent``.
Never float proximity, never a soft ranking.

Fences
------
::

    Labels come from the certified native FIPS relation (fips_basic), never from
    the SFP circuit under test.
    algebraic zero != NONADMISSION;   0 != bottom
    FFF=000 and pp=00 are formal completion values only, not Events and not bottom.
    SFP is a representation layer, not a replacement OT evaluator.
    No per-Event free output identity: downstream scoring must be relational.

Scope
-----
This module owns the dataset only. Fold construction (leave-one-habitat-out over
14 habitats, leave-one-Fano-point-out over 7 points holding both signs), the four
representation arms, the shared candidate scorer and the deterministic baselines
are owned by later tasks and are deliberately absent here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from topographo.ssd import fips_basic, projective
from topographo.ssd.fips_adapter import N_EVENTS, EventIndex
from topographo.ssd.frames import is_event, ray
from topographo.ssd.seal import public_label

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "task_dataset.json"

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"
EXPECTED_TABLE_SHA256 = (
    "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea"
)

FENCES = (
    (
        "Labels come from the certified native FIPS relation (fips_basic), never "
        "from the SFP circuit under test."
    ),
    "algebraic zero != NONADMISSION;   0 != bottom",
    "FFF=000 and pp=00 are formal completion values only, not Events and not bottom.",
    "SFP is a representation layer, not a replacement OT evaluator.",
    "No per-Event free output identity: downstream scoring must be relational.",
)

LABEL_SOURCE = (
    "topographo.ssd.fips_basic.admissible / topographo.ssd.fips_basic.third — the "
    "certified native FIPS relation. sfp.ExactSfpCircuit is NEVER consulted for a "
    "label: label generation (label_pair / build_dataset) accepts no circuit "
    "argument, and sfp is imported only inside sfp_cross_check, which runs after "
    "the dataset is frozen and may only read it."
)

CORRECTNESS_CRITERION = (
    "exact certified Event identity via topographo.ssd.projective.equivalent; "
    "never float proximity, never a soft ranking"
)

# --- the bottom fence: an explicit distinct symbol, not any algebraic zero ---
BOTTOM_INDEX = -1
BOTTOM_SYMBOL = "BOTTOM"

BOTTOM_FENCE = (
    (
        "NONADMISSION carries the explicit sentinel (target_index=-1, "
        "target_symbol='BOTTOM'). Bottom is NOT encoded as FFF=000, NOT as pp=00, "
        "NOT as Event index 0, and NOT as any algebraic zero. Event index 0 is a "
        "genuine certified Event and is never overloaded as bottom."
    ),
)

PINS = {
    "events": N_EVENTS,
    "blocks": 56,
    "habitats": 14,
    "fano_points": 7,
    "ordered_pairs_total": N_EVENTS * N_EVENTS,
    "admit": 336,
    "repeated": 84,
    "same_habitat_disjoint": 84,
    "cross_habitat": 6552,
    "same_habitat_ordered_pairs": 504,
    "block_degree_per_event": 2,
    "ordered_pairs_per_block": 6,
}


class Label(str, Enum):
    """The two-valued native label. ``NONADMISSION`` is explicit, not a zero."""

    ADMIT = "ADMIT"
    NONADMISSION = "NONADMISSION"


class NonadmissionClass(str, Enum):
    """Why a pair failed to admit. ``None`` on admitted records."""

    REPEATED = "REPEATED"
    SAME_HABITAT_DISJOINT = "SAME_HABITAT_DISJOINT"
    CROSS_HABITAT = "CROSS_HABITAT"


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


def block_key(block: tuple) -> str:
    """Stable, iteration-order-independent textual key for a certified block."""

    return "|".join(sorted(encode_ray(e) for e in block))


def sign_bit(delta: int) -> int:
    """Declared chart (matches ``sfp.sign_bit``): ``+1 -> 0``, ``-1 -> 1``."""

    if delta == 1:
        return 0
    if delta == -1:
        return 1
    raise AssertionError(f"unexpected habitat delta {delta!r}")


# ---------------------------------------------------------------------------
# the stable 84-Event catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Catalogue:
    """The frozen 84-Event catalogue: index <-> Event, plus native coordinates.

    Ordering is ``topographo.ssd.fips_adapter.EventIndex`` (which is
    ``fips_basic.EVENTS``, CI-pinned, ``N_EVENTS = 84``). Recorded together with
    its SHA-256 digest so folds and arms are reproducible.
    """

    events: tuple[tuple, ...]
    index_of: dict[tuple, int]
    encodings: tuple[str, ...]
    habitats: tuple[tuple[int, int], ...]      # index -> (P, delta)
    block_ids: tuple[tuple[int, int], ...]     # index -> the 2 incident block ids
    blocks: tuple[tuple[int, int, int], ...]   # block id -> 3 Event indices
    block_keys: tuple[str, ...]                # block id -> textual key
    order_source: str

    def __len__(self) -> int:
        return len(self.events)

    def index(self, event: tuple) -> int:
        return self.index_of[ray(event)]

    def event(self, index: int) -> tuple:
        return self.events[index]

    def habitat(self, index: int) -> tuple[int, int]:
        return self.habitats[index]

    def fano_point(self, index: int) -> int:
        return self.habitats[index][0]

    def delta(self, index: int) -> int:
        return self.habitats[index][1]

    def habitat_keys(self) -> tuple[str, ...]:
        """The 14 habitat keys, ascending by ``(P, delta)``."""

        return tuple(
            habitat_label(p, d) for p, d in sorted(set(self.habitats))
        )

    def events_of_habitat(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        rows = []
        for p, d in sorted(set(self.habitats)):
            members = tuple(
                i for i in range(len(self)) if self.habitats[i] == (p, d)
            )
            rows.append((habitat_label(p, d), members))
        return tuple(rows)

    def events_of_fano_point(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        rows = []
        for p in sorted({point for point, _ in self.habitats}):
            members = tuple(
                i for i in range(len(self)) if self.habitats[i][0] == p
            )
            rows.append((p, members))
        return tuple(rows)

    def shared_blocks(self, a_index: int, b_index: int) -> tuple[int, ...]:
        """Certified cyclic blocks containing both Events, ascending by block id."""

        left = self.block_ids[a_index]
        right = self.block_ids[b_index]
        return tuple(sorted(set(left) & set(right)))

    def as_json(self) -> dict:
        return {
            "n_events": len(self),
            "order_source": self.order_source,
            "events": [
                {
                    "index": i,
                    "ray": self.encodings[i],
                    "P": self.habitats[i][0],
                    "delta": self.habitats[i][1],
                    "S": sign_bit(self.habitats[i][1]),
                    "habitat": habitat_label(*self.habitats[i]),
                    "block_ids": list(self.block_ids[i]),
                }
                for i in range(len(self))
            ],
            "blocks": [
                {"block_id": b, "key": self.block_keys[b], "members": list(members)}
                for b, members in enumerate(self.blocks)
            ],
            "habitat_members": [
                {"habitat": key, "event_indices": list(members)}
                for key, members in self.events_of_habitat()
            ],
            "fano_point_members": [
                {"P": p, "event_indices": list(members)}
                for p, members in self.events_of_fano_point()
            ],
        }

    def sha256(self) -> str:
        """Digest of the catalogue ORDER plus its native coordinates."""

        return digest(
            [
                [
                    i,
                    self.encodings[i],
                    self.habitats[i][0],
                    self.habitats[i][1],
                    list(self.block_ids[i]),
                ]
                for i in range(len(self))
            ]
        )


def habitat_label(p: int, delta: int) -> str:
    return f"P={p},delta={delta:+d}"


def build_catalogue() -> Catalogue:
    """Build the frozen catalogue from the certified native structures only."""

    index = EventIndex.build()
    events = tuple(index.events)
    if len(events) != N_EVENTS or N_EVENTS != PINS["events"]:
        raise AssertionError(f"catalogue holds {len(events)} Events, not 84")
    if len(set(events)) != N_EVENTS:
        raise AssertionError("catalogue Events are not distinct")
    if events != tuple(fips_basic.EVENTS):
        raise AssertionError("EventIndex order diverged from fips_basic.EVENTS")
    if not all(is_event(e) for e in events):
        raise AssertionError("catalogue contains a non-Event ray")
    index_of = {e: i for i, e in enumerate(events)}
    if any(index.encode(e) != index_of[e] for e in events):
        raise AssertionError("EventIndex.encode disagrees with catalogue positions")

    encodings = tuple(encode_ray(e) for e in events)
    if len(set(encodings)) != N_EVENTS:
        raise AssertionError("ray encodings are not injective on the catalogue")
    habitats = tuple(public_label(e) for e in events)

    ordered_blocks = tuple(
        sorted(fips_basic.BLOCKS, key=block_key)
    )
    if len(ordered_blocks) != PINS["blocks"]:
        raise AssertionError(f"found {len(ordered_blocks)} blocks, not 56")
    blocks = tuple(
        tuple(sorted(index_of[ray(e)] for e in block)) for block in ordered_blocks
    )
    if any(len(members) != 3 or len(set(members)) != 3 for members in blocks):
        raise AssertionError("a certified block does not hold 3 distinct Events")
    block_keys = tuple(block_key(block) for block in ordered_blocks)
    if len(set(block_keys)) != PINS["blocks"]:
        raise AssertionError("block keys are not injective")

    incident: list[list[int]] = [[] for _ in range(N_EVENTS)]
    for block_id, members in enumerate(blocks):
        for member in members:
            incident[member].append(block_id)
    degrees = sorted({len(v) for v in incident})
    if degrees != [PINS["block_degree_per_event"]]:
        raise AssertionError(f"block-degree spectrum is {degrees}, not [2]")
    block_ids = tuple(tuple(sorted(v)) for v in incident)

    # Every block lives in exactly one habitat: needed for the class taxonomy.
    for block_id, members in enumerate(blocks):
        labels = {habitats[m] for m in members}
        if len(labels) != 1:
            raise AssertionError(f"block {block_id} spans more than one habitat")

    if len(set(habitats)) != PINS["habitats"]:
        raise AssertionError(f"found {len(set(habitats))} habitats, not 14")
    if len({p for p, _ in habitats}) != PINS["fano_points"]:
        raise AssertionError("habitat Fano points do not cover exactly 7 points")

    return Catalogue(
        events=events,
        index_of=index_of,
        encodings=encodings,
        habitats=habitats,
        block_ids=block_ids,
        blocks=blocks,
        block_keys=block_keys,
        order_source=(
            "topographo.ssd.fips_adapter.EventIndex.build() == fips_basic.EVENTS "
            "(CI-pinned, N_EVENTS = 84); blocks ordered by the stable textual "
            "block key, never by set/dict iteration order or hash()"
        ),
    )


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairRecord:
    """One frozen ordered-pair record. Immutable; all fields are plain data."""

    a_index: int
    b_index: int
    label: Label
    target_index: int
    target_symbol: str
    nonadmission_class: NonadmissionClass | None
    habitat_a: tuple[int, int]
    habitat_b: tuple[int, int]
    fano_point_a: int
    fano_point_b: int
    sign_a: int
    sign_b: int
    shared_block: int | None

    @property
    def admitted(self) -> bool:
        return self.label is Label.ADMIT

    @property
    def is_bottom(self) -> bool:
        return self.target_index == BOTTOM_INDEX

    @property
    def same_habitat(self) -> bool:
        return self.habitat_a == self.habitat_b

    @property
    def sign_bit_a(self) -> int:
        return sign_bit(self.sign_a)

    @property
    def sign_bit_b(self) -> int:
        return sign_bit(self.sign_b)

    def as_json(self) -> dict:
        return {
            "a_index": self.a_index,
            "b_index": self.b_index,
            "label": self.label.value,
            "target_index": self.target_index,
            "target_symbol": self.target_symbol,
            "nonadmission_class": (
                None if self.nonadmission_class is None
                else self.nonadmission_class.value
            ),
            "habitat_a": list(self.habitat_a),
            "habitat_b": list(self.habitat_b),
            "fano_point_a": self.fano_point_a,
            "fano_point_b": self.fano_point_b,
            "sign_a": self.sign_a,
            "sign_b": self.sign_b,
            "shared_block": self.shared_block,
        }

    def as_row(self) -> list:
        """Compact deterministic row used for the dataset digest."""

        return [
            self.a_index,
            self.b_index,
            self.label.value,
            self.target_index,
            self.target_symbol,
            None if self.nonadmission_class is None
            else self.nonadmission_class.value,
            list(self.habitat_a),
            list(self.habitat_b),
            self.fano_point_a,
            self.fano_point_b,
            self.sign_a,
            self.sign_b,
            self.shared_block,
        ]


# ---------------------------------------------------------------------------
# LABEL GENERATION — native relation only, no circuit parameter exists
# ---------------------------------------------------------------------------

def label_pair(catalogue: Catalogue, a_index: int, b_index: int) -> PairRecord:
    """Label one ordered pair from the certified native FIPS relation.

    The signature is the fence: this function receives a catalogue and two
    integers. There is no parameter through which ``sfp.ExactSfpCircuit`` — or
    any other representation under test — could supply a label. Admission and
    the forced third come from ``fips_basic.admissible`` / ``fips_basic.third``.
    """

    left = catalogue.event(a_index)
    right = catalogue.event(b_index)
    habitat_a = catalogue.habitat(a_index)
    habitat_b = catalogue.habitat(b_index)

    repeated = a_index == b_index
    native_admissible = fips_basic.admissible(left, right)
    if repeated and native_admissible:
        raise AssertionError(
            f"native relation admitted the repeated pair ({a_index}, {a_index})"
        )

    shared = catalogue.shared_blocks(a_index, b_index)

    if not repeated and native_admissible:
        if len(shared) != 1:
            raise AssertionError(
                f"admitted pair ({a_index}, {b_index}) shares {len(shared)} blocks"
            )
        block_id = shared[0]
        native_third = fips_basic.third(left, right)
        target_index = catalogue.index(native_third)
        # Exact certified Event identity, never float proximity.
        if not projective.equivalent(native_third, catalogue.event(target_index)):
            raise AssertionError("forced third failed exact projective identity")
        members = catalogue.blocks[block_id]
        if target_index not in members or target_index in (a_index, b_index):
            raise AssertionError(
                f"forced third {target_index} is not the remaining member of "
                f"block {block_id} {members}"
            )
        if target_index == BOTTOM_INDEX:
            raise AssertionError("an admitted record reached the bottom sentinel")
        return PairRecord(
            a_index=a_index,
            b_index=b_index,
            label=Label.ADMIT,
            target_index=target_index,
            target_symbol=catalogue.encodings[target_index],
            nonadmission_class=None,
            habitat_a=habitat_a,
            habitat_b=habitat_b,
            fano_point_a=habitat_a[0],
            fano_point_b=habitat_b[0],
            sign_a=habitat_a[1],
            sign_b=habitat_b[1],
            shared_block=block_id,
        )

    if repeated:
        klass = NonadmissionClass.REPEATED
    elif habitat_a == habitat_b:
        if shared:
            raise AssertionError(
                f"non-admitted same-habitat pair ({a_index}, {b_index}) shares "
                f"{len(shared)} block(s); the class taxonomy is wrong"
            )
        klass = NonadmissionClass.SAME_HABITAT_DISJOINT
    else:
        if shared:
            raise AssertionError(
                f"cross-habitat pair ({a_index}, {b_index}) shares a block"
            )
        klass = NonadmissionClass.CROSS_HABITAT

    return PairRecord(
        a_index=a_index,
        b_index=b_index,
        label=Label.NONADMISSION,
        target_index=BOTTOM_INDEX,
        target_symbol=BOTTOM_SYMBOL,
        nonadmission_class=klass,
        habitat_a=habitat_a,
        habitat_b=habitat_b,
        fano_point_a=habitat_a[0],
        fano_point_b=habitat_b[0],
        sign_a=habitat_a[1],
        sign_b=habitat_b[1],
        shared_block=None,
    )


# ---------------------------------------------------------------------------
# the pool
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """The frozen pool of all 7056 ordered-pair records.

    Row-major layout: the record for ``(a, b)`` sits at ``a * 84 + b``. Per-class
    index lists are precomputed for fold construction; every accessor is
    deterministic and order-independent.
    """

    catalogue: Catalogue
    records: tuple[PairRecord, ...]
    admit_positions: tuple[int, ...]
    nonadmission_positions: tuple[tuple[str, tuple[int, ...]], ...]

    def __len__(self) -> int:
        return len(self.records)

    def position(self, a_index: int, b_index: int) -> int:
        n = len(self.catalogue)
        if not (0 <= a_index < n and 0 <= b_index < n):
            raise IndexError("Event index out of the certified 0..83 range")
        return a_index * n + b_index

    def record(self, a_index: int, b_index: int) -> PairRecord:
        """Look up the record for an ordered pair."""

        found = self.records[self.position(a_index, b_index)]
        if (found.a_index, found.b_index) != (a_index, b_index):
            raise AssertionError("row-major layout is corrupt")
        return found

    def target(self, a_index: int, b_index: int) -> int:
        """Forced-third Event index, or ``BOTTOM_INDEX`` for non-admission."""

        return self.record(a_index, b_index).target_index

    def admitted(self) -> tuple[PairRecord, ...]:
        return tuple(self.records[p] for p in self.admit_positions)

    def nonadmitted(self, klass: NonadmissionClass) -> tuple[PairRecord, ...]:
        for name, positions in self.nonadmission_positions:
            if name == klass.value:
                return tuple(self.records[p] for p in positions)
        raise KeyError(f"unknown non-admission class {klass!r}")

    def positions_of_class(self, klass: NonadmissionClass) -> tuple[int, ...]:
        for name, positions in self.nonadmission_positions:
            if name == klass.value:
                return positions
        raise KeyError(f"unknown non-admission class {klass!r}")

    def counts(self) -> dict[str, int]:
        tally = Counter(
            r.label.value if r.admitted else r.nonadmission_class.value
            for r in self.records
        )
        return {
            "ordered_pairs_total": len(self.records),
            "admit": tally[Label.ADMIT.value],
            "repeated": tally[NonadmissionClass.REPEATED.value],
            "same_habitat_disjoint": tally[
                NonadmissionClass.SAME_HABITAT_DISJOINT.value
            ],
            "cross_habitat": tally[NonadmissionClass.CROSS_HABITAT.value],
            "same_habitat_ordered_pairs": sum(
                1 for r in self.records if r.same_habitat
            ),
        }

    def rows(self) -> list[list]:
        return [r.as_row() for r in self.records]

    def sha256(self) -> str:
        """Full-fidelity digest over all 7056 records, in row-major order."""

        return digest(self.rows())


def build_dataset(catalogue: Catalogue | None = None) -> Dataset:
    """Build the frozen 7056-record pool. No circuit parameter exists here."""

    catalogue = catalogue or build_catalogue()
    n = len(catalogue)
    records = tuple(
        label_pair(catalogue, a, b)
        for a in range(n)
        for b in range(n)
    )
    if len(records) != PINS["ordered_pairs_total"]:
        raise AssertionError(f"built {len(records)} records, not 7056")

    admit_positions = tuple(
        p for p, r in enumerate(records) if r.label is Label.ADMIT
    )
    nonadmission_positions = tuple(
        (
            klass.value,
            tuple(p for p, r in enumerate(records) if r.nonadmission_class is klass),
        )
        for klass in (
            NonadmissionClass.REPEATED,
            NonadmissionClass.SAME_HABITAT_DISJOINT,
            NonadmissionClass.CROSS_HABITAT,
        )
    )
    covered = len(admit_positions) + sum(
        len(positions) for _, positions in nonadmission_positions
    )
    if covered != len(records):
        raise AssertionError("per-class index lists do not partition the pool")

    return Dataset(
        catalogue=catalogue,
        records=records,
        admit_positions=admit_positions,
        nonadmission_positions=nonadmission_positions,
    )


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_counts(dataset: Dataset) -> dict:
    """Observed vs. independently predicted counts. Predictions are never edited."""

    observed = dataset.counts()
    checks = {}
    for name in (
        "ordered_pairs_total",
        "admit",
        "repeated",
        "same_habitat_disjoint",
        "cross_habitat",
        "same_habitat_ordered_pairs",
    ):
        checks[name] = {
            "expected": PINS[name],
            "observed": observed[name],
            "agrees": PINS[name] == observed[name],
        }
    decomposition = (
        observed["repeated"]
        + observed["admit"]
        + observed["same_habitat_disjoint"]
    )
    checks["same_habitat_decomposition"] = {
        "expected": PINS["same_habitat_ordered_pairs"],
        "observed": decomposition,
        "agrees": decomposition == observed["same_habitat_ordered_pairs"]
        == PINS["same_habitat_ordered_pairs"],
        "statement": "same-habitat = REPEATED + ADMIT + SAME_HABITAT_DISJOINT",
    }
    checks["native_edge_table"] = {
        "expected": PINS["admit"],
        "observed": len(fips_basic.ORDERED_EDGES),
        "agrees": len(fips_basic.ORDERED_EDGES) == PINS["admit"],
        "statement": "ADMIT count equals |fips_basic.ORDERED_EDGES|",
    }
    per_habitat = Counter(
        habitat_label(*r.habitat_a) for r in dataset.admitted()
    )
    checks["admit_per_habitat_uniform_24"] = {
        "expected": 24,
        "observed": sorted(set(per_habitat.values())),
        "agrees": sorted(set(per_habitat.values())) == [24]
        and len(per_habitat) == PINS["habitats"],
    }
    disjoint_per_habitat = Counter(
        habitat_label(*r.habitat_a)
        for r in dataset.nonadmitted(NonadmissionClass.SAME_HABITAT_DISJOINT)
    )
    checks["same_habitat_disjoint_per_habitat_uniform_6"] = {
        "expected": 6,
        "observed": sorted(set(disjoint_per_habitat.values())),
        "agrees": sorted(set(disjoint_per_habitat.values())) == [6]
        and len(disjoint_per_habitat) == PINS["habitats"],
    }
    return {
        "counts": checks,
        "all_predicted_counts_agree": all(c["agrees"] for c in checks.values()),
        "prediction_policy": (
            "counts were predicted from the certified structure before the dataset "
            "was built; a mismatch is reported, never absorbed into the prediction"
        ),
    }


def verify_role_neutrality(dataset: Dataset) -> dict:
    """Every block supports all 6 ordered completions; no privileged direction."""

    catalogue = dataset.catalogue
    per_block_ordered = []
    per_block_targets_complete = []
    block_pairs: list[tuple[int, int]] = []
    for members in catalogue.blocks:
        ordered = [
            (a, b) for a, b in itertools.permutations(members, 2)
        ]
        admitted = [dataset.record(a, b) for a, b in ordered]
        per_block_ordered.append(
            sum(
                1
                for record, (a, b) in zip(admitted, ordered)
                if record.admitted
                and record.target_index
                == next(m for m in members if m not in (a, b))
            )
        )
        per_block_targets_complete.append(
            sorted({record.target_index for record in admitted}) == sorted(members)
        )
        block_pairs.extend(ordered)

    covered = sorted(set(block_pairs))
    admitted_pairs = sorted(
        (r.a_index, r.b_index) for r in dataset.admitted()
    )

    symmetry_checked = 0
    symmetry_agree = 0
    both_orders = 0
    for record in dataset.admitted():
        mirror = dataset.record(record.b_index, record.a_index)
        both_orders += mirror.admitted
        symmetry_checked += 1
        if (
            mirror.admitted
            and mirror.target_index == record.target_index
            and mirror.shared_block == record.shared_block
        ):
            symmetry_agree += 1

    return {
        "blocks": len(catalogue.blocks),
        "ordered_pairs_per_block": sorted(set(per_block_ordered)),
        "all_blocks_support_all_6_completions": (
            sorted(set(per_block_ordered)) == [PINS["ordered_pairs_per_block"]]
        ),
        "all_three_missing_member_choices_occur": all(per_block_targets_complete),
        "block_ordered_pairs_total": sum(per_block_ordered),
        "block_ordered_pairs_total_matches_admit": (
            sum(per_block_ordered) == PINS["admit"]
        ),
        "block_pairs_are_exactly_the_admitted_pairs": covered == admitted_pairs,
        "distinct_block_ordered_pairs": len(covered),
        "symmetry_checked": symmetry_checked,
        "target_symmetric_agreements": symmetry_agree,
        "both_input_orders_present": both_orders,
        "target_ab_equals_target_ba_on_all_admitted": (
            symmetry_agree == both_orders == symmetry_checked == PINS["admit"]
        ),
        "statement": (
            "no privileged direction and no separate structure for one completion: "
            "all 56 * 6 = 336 ordered completions are ordinary records in one pool"
        ),
    }


def verify_bottom_fence(dataset: Dataset) -> dict:
    """The bottom sentinel is explicit and never collides with an algebraic zero."""

    catalogue = dataset.catalogue
    n = len(catalogue)
    admitted_with_bottom = [
        (r.a_index, r.b_index) for r in dataset.records
        if r.admitted and (r.target_index == BOTTOM_INDEX
                           or r.target_symbol == BOTTOM_SYMBOL)
    ]
    nonadmitted_with_event = [
        (r.a_index, r.b_index) for r in dataset.records
        if not r.admitted and (0 <= r.target_index < n
                               or r.target_symbol != BOTTOM_SYMBOL)
    ]
    admitted_targets = sorted({r.target_index for r in dataset.admitted()})
    event_zero = catalogue.event(0)

    return {
        "bottom_index": BOTTOM_INDEX,
        "bottom_symbol": BOTTOM_SYMBOL,
        "bottom_is_outside_the_event_index_range": not 0 <= BOTTOM_INDEX < n,
        "admitted_records_carrying_bottom": len(admitted_with_bottom),
        "nonadmitted_records_carrying_a_real_event": len(nonadmitted_with_event),
        "no_admitted_record_carries_bottom": not admitted_with_bottom,
        "no_nonadmitted_record_carries_a_real_event": not nonadmitted_with_event,
        "event_index_0_is_a_genuine_event": bool(is_event(event_zero)),
        "event_index_0_ray": catalogue.encodings[0],
        "event_index_0_occurs_as_a_real_target": 0 in admitted_targets,
        "distinct_admitted_targets": len(admitted_targets),
        "every_event_occurs_as_a_target": len(admitted_targets) == PINS["events"],
        "nonadmission_is_an_explicit_label": (
            Label.NONADMISSION.value == "NONADMISSION"
            and Label.NONADMISSION is not Label.ADMIT
        ),
        "nonadmission_classes_all_populated": all(
            len(positions) > 0 for _, positions in dataset.nonadmission_positions
        ),
        "fence": list(BOTTOM_FENCE),
        "statement": (
            "bottom is the explicit sentinel (-1, 'BOTTOM'); it is not FFF=000, "
            "not pp=00, not Event index 0, and not any algebraic zero"
        ),
    }


def verify_native_consistency(dataset: Dataset) -> dict:
    """Cross-check the label rule against the certified block incidence structure.

    This is a *native* self-consistency check (blocks vs. ``admissible`` /
    ``third``) and involves no SFP code at all.
    """

    catalogue = dataset.catalogue
    n = len(catalogue)
    admissible_agree = 0
    third_agree = 0
    third_checked = 0
    disagreements: list[dict] = []
    repeated_admitted = 0

    for a, b in itertools.product(range(n), repeat=2):
        record = dataset.record(a, b)
        native_ok = fips_basic.admissible(catalogue.event(a), catalogue.event(b))
        structural_ok = a != b and len(catalogue.shared_blocks(a, b)) == 1
        if a == b:
            repeated_admitted += native_ok
        if native_ok == structural_ok == record.admitted:
            admissible_agree += 1
        else:
            disagreements.append(
                {
                    "a": a,
                    "b": b,
                    "native_admissible": native_ok,
                    "shares_exactly_one_block": structural_ok,
                    "record_admitted": record.admitted,
                }
            )
        if not record.admitted:
            continue
        third_checked += 1
        native_third = fips_basic.third(catalogue.event(a), catalogue.event(b))
        if projective.equivalent(native_third, catalogue.event(record.target_index)):
            third_agree += 1
        else:
            disagreements.append(
                {
                    "a": a,
                    "b": b,
                    "issue": "target failed exact projective identity",
                    "expected": encode_ray(native_third),
                    "observed": catalogue.encodings[record.target_index],
                }
            )

    return {
        "label_source": LABEL_SOURCE,
        "correctness_criterion": CORRECTNESS_CRITERION,
        "ordered_pairs_checked": n * n,
        "admissibility_agreements": admissible_agree,
        "admissibility_exact": admissible_agree == n * n,
        "native_relation_admits_no_repeated_pair": repeated_admitted == 0,
        "forced_third_checked": third_checked,
        "forced_third_exact_projective_agreements": third_agree,
        "forced_third_exact": third_agree == third_checked == PINS["admit"],
        "disagreements": disagreements[:8],
        "disagreement_count": len(disagreements),
        "statement": (
            "native admissible(a, b) coincides with 'distinct and sharing exactly "
            "one certified cyclic block' on all 7056 ordered pairs"
        ),
    }


def sfp_cross_check(dataset: Dataset) -> dict:
    """REFERENCE-ORACLE DIAGNOSTIC — not the label source.

    Reads the already-frozen dataset and reports whether
    ``sfp.ExactSfpCircuit`` reproduces the native labels on all 7056 pairs. The
    ``sfp`` import is deliberately function-local so that no module-level path
    exists from the representation arm into label generation. This function
    returns a report; it never returns, produces or mutates a label.
    """

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Deliberately function-local: keeps the circuit out of the labelling path.
    import sfp

    codec = sfp.SfpCodec()
    circuit = sfp.ExactSfpCircuit()
    catalogue = dataset.catalogue
    addresses = tuple(codec.encode(e) for e in catalogue.events)

    admission_agree = 0
    circuit_admitted = 0
    third_agree = 0
    third_checked = 0
    disagreements: list[dict] = []

    for record in dataset.records:
        a, b = record.a_index, record.b_index
        circuit_ok = circuit.admit_event(addresses[a], addresses[b])
        circuit_admitted += circuit_ok
        if circuit_ok == record.admitted:
            admission_agree += 1
        else:
            disagreements.append(
                {
                    "a": a,
                    "b": b,
                    "native_label": record.label.value,
                    "circuit_admit": circuit_ok,
                }
            )
        if not record.admitted:
            continue
        third_checked += 1
        forced = circuit.forced_third_event(addresses[a], addresses[b])
        if forced is None:
            disagreements.append(
                {"a": a, "b": b, "issue": "circuit produced no forced third"}
            )
            continue
        got = codec.decode(forced)
        if projective.equivalent(got, catalogue.event(record.target_index)):
            third_agree += 1
        else:
            disagreements.append(
                {
                    "a": a,
                    "b": b,
                    "issue": "forced third mismatch",
                    "expected": catalogue.encodings[record.target_index],
                    "observed": encode_ray(got),
                }
            )

    total = len(dataset)
    return {
        "role": "reference-oracle diagnostic; NOT the label source",
        "label_source_reminder": LABEL_SOURCE,
        "circuit_role": sfp.ExactSfpCircuit.role,
        "circuit_never_a_label_source": bool(
            sfp.ExactSfpCircuit.never_a_label_source
        ),
        "import_is_function_local": True,
        "ordered_pairs_checked": total,
        "admission_agreements": admission_agree,
        "admission_agreement_complete": admission_agree == total,
        "circuit_admitted_ordered_pairs": circuit_admitted,
        "circuit_admitted_matches_native": circuit_admitted == PINS["admit"],
        "forced_third_checked": third_checked,
        "forced_third_agreements": third_agree,
        "forced_third_agreement_complete": (
            third_agree == third_checked == PINS["admit"]
        ),
        "comparison": CORRECTNESS_CRITERION,
        "disagreements": disagreements[:8],
        "disagreement_count": len(disagreements),
        "agrees": (
            admission_agree == total
            and third_agree == third_checked == PINS["admit"]
            and not disagreements
        ),
    }


def verify_determinism(dataset: Dataset) -> dict:
    """Rebuild from scratch and compare digests; no iteration order in output."""

    rebuilt = build_dataset()
    return {
        "rebuild_dataset_sha256_stable": rebuilt.sha256() == dataset.sha256(),
        "rebuild_catalogue_sha256_stable": (
            rebuilt.catalogue.sha256() == dataset.catalogue.sha256()
        ),
        "rebuild_rows_identical": rebuilt.rows() == dataset.rows(),
        "statement": (
            "every ordering in this module comes from integer indices, sorted "
            "textual ray/block keys, or explicit tuples; no set/dict iteration "
            "order and no hash() value reaches the artifact"
        ),
        "interpreter_state_excluded": (
            "no interpreter flag (including PYTHONHASHSEED / "
            "sys.flags.hash_randomization) is recorded, so the artifact is "
            "byte-identical across hash seeds"
        ),
    }


def sample_records(dataset: Dataset) -> dict:
    """A small deterministic sample; the full pool is covered by the digest."""

    def take(positions: tuple[int, ...], count: int) -> list[dict]:
        if not positions:
            return []
        stride = max(1, len(positions) // count)
        chosen = sorted({positions[i * stride] for i in range(count)
                         if i * stride < len(positions)})
        return [dataset.records[p].as_json() for p in chosen]

    body = {
        "policy": (
            "deterministic stride over each per-class position list, row-major; "
            "the full 7056-record pool is pinned by dataset.sha256"
        ),
        "ADMIT": take(dataset.admit_positions, 8),
    }
    for name, positions in dataset.nonadmission_positions:
        body[name] = take(positions, 4)
    return body


def block_witness(dataset: Dataset) -> dict:
    """One fully expanded block, showing all 6 ordered completions side by side."""

    catalogue = dataset.catalogue
    members = catalogue.blocks[0]
    return {
        "block_id": 0,
        "key": catalogue.block_keys[0],
        "members": list(members),
        "habitat": habitat_label(*catalogue.habitat(members[0])),
        "ordered_completions": [
            {
                "a_index": a,
                "b_index": b,
                "label": dataset.record(a, b).label.value,
                "target_index": dataset.record(a, b).target_index,
            }
            for a, b in itertools.permutations(members, 2)
        ],
        "statement": "all 6 ordered completions are present and equally weighted",
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
        "label_sources": [
            "topographo.ssd.fips_basic.admissible (native admission)",
            "topographo.ssd.fips_basic.third (native forced third)",
        ],
        "coordinate_sources": [
            "topographo.ssd.fips_adapter.EventIndex (CI-pinned 84-Event order)",
            "topographo.ssd.fips_basic.BLOCKS (56 certified cyclic blocks)",
            "topographo.ssd.seal.public_label (habitat key (P, delta))",
            "topographo.ssd.projective.equivalent (exact Event identity)",
        ],
        "not_a_label_source": [
            (
                "experiments/sfp_representation/sfp.py::ExactSfpCircuit "
                "(reference-oracle diagnostic only)"
            )
        ],
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    catalogue = dataset.catalogue

    counts = verify_counts(dataset)
    neutrality = verify_role_neutrality(dataset)
    bottom = verify_bottom_fence(dataset)
    native = verify_native_consistency(dataset)
    oracle = sfp_cross_check(dataset)
    determinism = verify_determinism(dataset)

    laws = {
        "all_predicted_counts_agree": counts["all_predicted_counts_agree"],
        "role_neutrality_all_blocks_6_completions": neutrality[
            "all_blocks_support_all_6_completions"
        ],
        "role_neutrality_all_missing_members_occur": neutrality[
            "all_three_missing_member_choices_occur"
        ],
        "role_neutrality_336_matches_admit": neutrality[
            "block_ordered_pairs_total_matches_admit"
        ],
        "role_neutrality_block_pairs_are_the_admitted_pairs": neutrality[
            "block_pairs_are_exactly_the_admitted_pairs"
        ],
        "target_symmetric_over_all_336": neutrality[
            "target_ab_equals_target_ba_on_all_admitted"
        ],
        "bottom_never_on_an_admitted_record": bottom[
            "no_admitted_record_carries_bottom"
        ],
        "no_event_index_on_a_nonadmitted_record": bottom[
            "no_nonadmitted_record_carries_a_real_event"
        ],
        "event_index_0_is_a_genuine_event": bottom["event_index_0_is_a_genuine_event"],
        "event_index_0_is_never_bottom": bottom[
            "event_index_0_occurs_as_a_real_target"
        ],
        "bottom_outside_event_index_range": bottom[
            "bottom_is_outside_the_event_index_range"
        ],
        "native_admissibility_exact_7056": native["admissibility_exact"],
        "native_forced_third_exact_336": native["forced_third_exact"],
        "native_admits_no_repeated_pair": native[
            "native_relation_admits_no_repeated_pair"
        ],
        "reference_oracle_agrees_7056": oracle["agrees"],
        "determinism_stable_on_rebuild": determinism["rebuild_dataset_sha256_stable"],
        "catalogue_stable_on_rebuild": determinism["rebuild_catalogue_sha256_stable"],
        "provenance_table_digest": provenance()["table_sha256"]["agrees"],
    }
    broken = sorted(name for name, ok in laws.items() if not ok)

    return {
        "module": "experiments/sfp_representation/task.py",
        "purpose": (
            "the frozen admit-and-force dataset: all 7056 ordered Event pairs "
            "labelled ADMIT / NONADMISSION from the certified native FIPS relation"
        ),
        "fences": list(FENCES),
        "label_rule": {
            "statement": (
                "ADMIT with target = unique third Event iff a != b and a, b share "
                "exactly one certified cyclic block; otherwise NONADMISSION "
                "(bottom)"
            ),
            "source": LABEL_SOURCE,
            "structural_enforcement": [
                "label_pair(catalogue, a_index, b_index) takes no circuit argument",
                "build_dataset(catalogue) takes no circuit argument",
                "sfp is imported only inside sfp_cross_check, after freezing",
                "sfp_cross_check returns a diagnostic and never a label",
            ],
            "correctness_criterion": CORRECTNESS_CRITERION,
        },
        "bottom_fence": bottom,
        "pins": PINS,
        "census": counts,
        "role_neutrality": neutrality,
        "native_consistency": native,
        "reference_oracle_diagnostic": oracle,
        "determinism": determinism,
        "catalogue": catalogue.as_json(),
        "record_schema": [
            "a_index", "b_index", "label", "target_index", "target_symbol",
            "nonadmission_class", "habitat_a", "habitat_b", "fano_point_a",
            "fano_point_b", "sign_a", "sign_b", "shared_block",
        ],
        "record_layout": (
            "row-major: the record for (a, b) sits at position a * 84 + b; "
            "Dataset.record(a, b) and Dataset.target(a, b) look it up"
        ),
        "per_class_counts": dataset.counts(),
        "sample_records": sample_records(dataset),
        "block_witness": block_witness(dataset),
        "downstream_contract": {
            "consumers": [
                "fold construction: leave-one-habitat-out over 14 habitats",
                "fold construction: leave-one-Fano-point-out over 7 points, both signs",
                "four representation arms",
                "shared candidate scorer",
                "deterministic baselines",
            ],
            "provided": [
                "stable 84-Event catalogue with digest",
                "habitat and Fano-point membership lists for fold construction",
                "per-class position lists over the 7056-record pool",
                "(a, b) -> record lookup",
            ],
            "withheld_by_scope": [
                "folds", "representation arms", "models", "training code",
                "baselines",
            ],
        },
        "digests": {
            "catalogue": catalogue.sha256(),
            "dataset": dataset.sha256(),
            "blocks": digest([list(m) for m in catalogue.blocks]),
            "method": (
                "hashlib.sha256(json.dumps(obj, sort_keys=True, "
                "separators=(',', ':')).encode()).hexdigest()"
            ),
        },
        "provenance": provenance(),
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - dataset labels derive from the certified native relation "
                "and all predicted counts agree"
                if not broken
                else "FAIL - see broken_laws"
            ),
        },
    }


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _report(payload: dict) -> None:
    counts = payload["census"]["counts"]
    print(
        "counts (expected/observed): "
        + ", ".join(
            f"{name}={counts[name]['expected']}/{counts[name]['observed']}"
            for name in (
                "ordered_pairs_total", "admit", "repeated",
                "same_habitat_disjoint", "cross_habitat",
                "same_habitat_ordered_pairs",
            )
        ),
        flush=True,
    )
    neutrality = payload["role_neutrality"]
    print(
        f"role neutrality: {neutrality['blocks']}/56 blocks x "
        f"{neutrality['ordered_pairs_per_block']} ordered completions = "
        f"{neutrality['block_ordered_pairs_total']}; "
        f"target(a,b)==target(b,a) on "
        f"{neutrality['target_symmetric_agreements']}/"
        f"{neutrality['symmetry_checked']}",
        flush=True,
    )
    native = payload["native_consistency"]
    print(
        f"native labels: admissibility {native['admissibility_agreements']}/"
        f"{native['ordered_pairs_checked']}, forced third "
        f"{native['forced_third_exact_projective_agreements']}/"
        f"{native['forced_third_checked']} exact projective",
        flush=True,
    )
    oracle = payload["reference_oracle_diagnostic"]
    print(
        f"reference-oracle diagnostic (NOT the label source): admission "
        f"{oracle['admission_agreements']}/{oracle['ordered_pairs_checked']}, "
        f"forced third {oracle['forced_third_agreements']}/"
        f"{oracle['forced_third_checked']}, agrees={oracle['agrees']}",
        flush=True,
    )
    bottom = payload["bottom_fence"]
    print(
        f"bottom fence: admitted-carrying-bottom="
        f"{bottom['admitted_records_carrying_bottom']}, "
        f"nonadmitted-carrying-Event="
        f"{bottom['nonadmitted_records_carrying_a_real_event']}, "
        f"Event 0 genuine={bottom['event_index_0_is_a_genuine_event']}, "
        f"Event 0 occurs as target={bottom['event_index_0_occurs_as_a_real_target']}",
        flush=True,
    )
    print(f"catalogue sha256: {payload['digests']['catalogue']}", flush=True)
    print(f"dataset   sha256: {payload['digests']['dataset']}", flush=True)


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
                f"FAIL: re-derived dataset is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches task_dataset.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

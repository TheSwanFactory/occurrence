"""009.09 Gate 0: the opaque local-chart problem, enumerated exactly.

``009.08`` handed the learner the exact finite SFP representation and asked it to
use it. ``009.09`` withdraws the representation and asks whether the learner can
*recover* it, locally, inside one supplied ``(S, FFF)`` habitat, from

* six **anonymous** Event tokens whose features are identical before marking,
* the complete pairwise ADMIT relation among them,
* exactly one certified unordered cyclic-block triple as a triadic anchor.

and nothing else. No ``PP``, no ``pp``, no SFP bit word, no native projective
coordinate, no XOR, no endpoint-intersection routine, no K4/Fano completion.

This module is Gate 0 and the frozen support construction. It trains nothing.
``009.09`` section 3 forbids training until the exact finite identifiability
census is complete, so the census is computed here first, from the **abstract**
six-token problem, and the primary experiment is gated on its outcome.

What is enumerated, not assumed
-------------------------------
Every number below is derived by exhaustive enumeration over the finite objects
and then compared against the ``009.09`` section 3.2 expectation as a *pin*::

    6 anonymous Event tokens
    15 unordered token pairs
    12 admitted pairs, 3 non-admitted
    all 720 token relabelings, classified
    |Aut(admission graph)|
    the subgroup induced by certified cyclic-block (K4 vertex) relabelings
    every 3-token clique of the admission graph
    every block system compatible with admission ALONE
    whether those systems disagree on forced-third consequence
    the exact forced-third ceiling from admission alone
    the exact forced-third ceiling from admission plus one certified anchor

The expected group orders ``48`` and ``24`` are **pins to verify**. They are not
assumptions, and nothing derived from them reaches the learner.

Label authority
---------------
Admission and the certified forced third come from the certified native relation
through the frozen ``task.Dataset``: ``task.label_pair`` obtained them from
``topographo.ssd.fips_basic`` (``admissible`` / ``third``) via
``task.Catalogue``. ``sfp.ExactSfpCircuit`` is **not imported here and generates
nothing**; the only use of the frozen ``sfp`` chart is the *hidden* certified
``PP``/``pp`` presentation, which exists solely so section 6.3 can audit whether a
recovered chart is equivalent to it. That chart is never an input and never a
training target.

The dual family, and what it is not
-----------------------------------
The admission graph of one habitat is the line graph of ``K4``, i.e. the
octahedron, and it carries **two** families of four triangular faces that cover
every Event exactly twice: the four certified cyclic blocks (the ``K4`` stars) and
the four ``K4`` triangles. The second family is a legitimate combinatorial object
of the admission graph and it is *not* certified FIPS structure. It appears here
only to define the ``009.09a`` section 2 dual-anchor **control** arm, and its
triples and thirds are labelled ``combinatorial`` everywhere they occur.

Advisory items folded in
------------------------
``009.09a`` is an advisory note and binds nobody; the Executive directed that its
suggestions be folded into this execution. Four are in force:

1. section 2 — a test-time **dual-anchor control arm** on the same trained model;
2. section 3 — ``sfp_equivalent`` reported as *entailed by* block-family recovery,
   with a mechanical assertion that the two rates agree;
3. section 4 — ``S`` and ``FFF`` are **not** fed to the network. They stay in the
   episode manifest as bookkeeping only, and the audit proves the input tensors
   carry no habitat identity;
4. section 5 — the holdout is described as **fresh-relabelling generalization** of
   a local relational rule, not held-out-structure generalization in the
   ``009.06``/``009.08`` sense.

Torch-free in intent; it inherits torch transitively through
``locator_task -> ladder_task -> harness``, and says so rather than claiming
otherwise.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import math
import platform
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import groups
from folds import all_folds, structural_folds
from locator_task import certified_chamber_of_block
from sfp import SfpCodec, habitat_key
from task import Dataset, build_dataset, digest, habitat_label

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_discovery_artifacts"
OUTPUT = ARTIFACTS / "discovery_task.json"

#: The frozen ``009.08`` result commit this turn builds on.
BASE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"

EXECUTED_TASK = "009.09-Kiro-opaque-local-chart-discovery-task.md"
ADVISORY_NOTE = "009.09a-Claude-pre-execution-review.md"
PRIOR_RESULT = "009.08-Kiro-query-relative-PP-localization-result.md"

#: One habitat presents six anonymous Event tokens.
N_TOKENS = 6

#: Local slots ``0..5``. A slot is a *bookkeeping* index into the habitat's own
#: ascending Event list. Every episode maps slots onto opaque nodes through a
#: fresh permutation, so no slot is ever a stable semantic address.
SLOTS = tuple(range(N_TOKENS))

#: Anchor triples per family, and scored queries per episode.
TRIPLES_PER_FAMILY = 4
SCORED_QUERIES_PER_EPISODE = 9

#: Fresh opaque relabelings per (habitat, anchor). Frozen before any run.
PERMUTATIONS_PER_ANCHOR = 12

#: ``009.09`` section 3.2 expectations, carried as PINS TO VERIFY.
PINS = {
    "tokens": 6,
    "unordered_pairs": 15,
    "admitted_pairs": 12,
    "nonadmitted_pairs": 3,
    "certified_blocks_per_habitat": 4,
    "events_per_block": 3,
    "blocks_per_event": 2,
    "token_relabelings": 720,
    "admission_automorphism_order": 48,
    "certified_relabeling_subgroup_order": 24,
    "admission_graph_degree": 4,
    "triangular_faces": 8,
    "covering_face_families": 2,
    "block_systems_compatible_with_admission": 2,
    "common_neighbours_of_an_admitted_pair": 2,
    "habitats": 14,
    "loho_folds": 14,
    "lofpo_folds": 7,
    "structural_folds": 21,
    "episodes_per_habitat": TRIPLES_PER_FAMILY * PERMUTATIONS_PER_ANCHOR,
    "scored_queries_per_episode": SCORED_QUERIES_PER_EPISODE,
    "uniform_eligible_node_chance": 0.25,
    "uniform_all_node_chance": 1.0 / 6.0,
}

#: The two exact ceilings this gate must establish before the primary run.
EXPECTED_CEILINGS = {
    "admission_only_forced_third": 0.5,
    "one_anchor_forced_third": 1.0,
}

FENCES = (
    "Admission and the certified forced third come from the certified native "
    "relation via task.Dataset; sfp.ExactSfpCircuit is not imported here.",
    "S and FFF are SUPPLIED habitat identity. They are neither discovered nor "
    "predicted, and per 009.09a section 4 they are not fed to the network at all.",
    "Opaque token names carry NO structure. A fresh permutation is applied per "
    "episode and no trainable parameter is indexed by any Event, habitat or slot.",
    "The dual (K4-triangle) family is a COMBINATORIAL object of the admission "
    "graph, not certified FIPS structure. Its triples and thirds are labelled "
    "combinatorial and are only ever a control.",
    "The hidden certified PP/pp chart is used ONLY by the section 6.3 equivalence "
    "audit. It is never an input, never a target, and never a label source.",
    "LOHO and LOFPO here hold out a fresh relabelling of the same octahedron, not "
    "a new structure; per 009.09a section 5 this is fresh-relabelling "
    "generalization of a local relational rule.",
    "The exact one-anchor completion is a NONLEARNED CEILING, never a competing "
    "learned baseline.",
    "FFF=000 and pp=00 remain formal algebraic completion values outside the "
    "finite Event catalogue and play no part in the opaque-node protocol.",
    "BOTTOM stays distinct from every algebraic zero and from any malformed or "
    "unresolved prediction. No episode row here is a BOTTOM row: every scored "
    "query is an admitted pair.",
)

#: What the learner is allowed to see, and what it is not. Enforced downstream by
#: ``discovery_heads``' input-surface audit; declared here because this module owns
#: the episode construction.
SUPPLIED = (
    "six anonymous nodes with identical features before marking",
    "the complete symmetric pairwise ADMIT matrix among those nodes",
    "exactly one unordered anchor triple, marked on its three nodes",
    "one unordered query pair, marked on its two nodes",
)
WITHHELD = (
    "PP",
    "pp",
    "SFP bit words",
    "native projective coordinates",
    "Event catalogue indices as semantic features",
    "an XOR operation",
    "an endpoint-intersection routine",
    "a K4/Fano completion routine",
    "S and FFF (009.09a section 4: bookkeeping only, never a network input)",
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

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


def encode_pair(pair: tuple[int, int]) -> str:
    return f"{pair[0]}-{pair[1]}"


def encode_triple(triple: tuple[int, int, int]) -> str:
    return "-".join(str(t) for t in triple)


def encode_perm(perm: tuple[int, ...]) -> str:
    return "".join(str(p) for p in perm)


# ---------------------------------------------------------------------------
# the abstract six-token problem
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def all_relabelings() -> tuple[tuple[int, ...], ...]:
    """All ``6! = 720`` token relabelings, in a fixed lexicographic order."""

    perms = tuple(itertools.permutations(SLOTS))
    if len(perms) != PINS["token_relabelings"]:
        raise AssertionError(f"enumerated {len(perms)} relabelings, not 720")
    return perms


@dataclass(frozen=True)
class LocalProblem:
    """One certified habitat presented as an abstract six-token problem.

    ``events`` is bookkeeping: the habitat's six certified Event indices in
    ascending order, so slot ``i`` means ``events[i]``. The learner never sees a
    slot. ``admit``, ``certified_third`` and ``certified_blocks`` are the certified
    relation restricted to those six Events and re-expressed in slots.
    """

    key: str                                       # "P=1,delta=+1"
    s: int
    fff: int
    events: tuple[int, ...]                        # 6 certified Event indices
    admit: tuple[tuple[int, int], ...]             # unordered admitted slot pairs
    nonadmit: tuple[tuple[int, int], ...]          # unordered non-admitted slot pairs
    certified_third: tuple[tuple[tuple[int, int], int], ...]   # pair -> third slot
    certified_blocks: tuple[tuple[int, int, int], ...]         # 4 slot triples
    chamber_of_block: tuple[int, ...]              # block position -> certified PP
    certified_presentation: tuple[tuple[int, int], ...]        # slot -> 2 chambers

    @property
    def sfp_key(self) -> str:
        return habitat_key(self.s, self.fff)

    def third(self, pair: tuple[int, int]) -> int:
        for candidate, value in self.certified_third:
            if candidate == pair:
                return value
        raise KeyError(f"{pair} is not an admitted pair of {self.key}")

    def neighbours(self, slot: int) -> tuple[int, ...]:
        out = set()
        for a, b in self.admit:
            if a == slot:
                out.add(b)
            elif b == slot:
                out.add(a)
        return tuple(sorted(out))

    def as_row(self) -> list:
        return [
            self.key,
            self.s,
            self.fff,
            list(self.events),
            [list(p) for p in self.admit],
            [[list(p), t] for p, t in self.certified_third],
            [list(t) for t in self.certified_blocks],
            list(self.chamber_of_block),
            [list(p) for p in self.certified_presentation],
        ]

    def sha256(self) -> str:
        return digest(self.as_row())


def unordered_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(SLOTS, 2))


def build_local_problems(dataset: Dataset, codec: SfpCodec) -> tuple[LocalProblem, ...]:
    """The 14 abstract six-token problems, from the certified relation only.

    Admission and the forced third are read off ``task.PairRecord``, which
    ``task.label_pair`` built from ``topographo.ssd.fips_basic``. The certified
    cyclic blocks are ``task.Catalogue.blocks`` restricted to the habitat. The
    hidden ``PP`` presentation is the frozen ``sfp`` chart, carried for the section
    6.3 equivalence audit and for nothing else.
    """

    catalogue = dataset.catalogue
    chamber_of_block = certified_chamber_of_block(dataset, codec)
    problems: list[LocalProblem] = []

    for _label, members in catalogue.events_of_habitat():
        events = tuple(sorted(members))
        if len(events) != N_TOKENS:
            raise AssertionError(f"habitat {members} holds {len(events)} Events, not 6")
        slot_of = {event: slot for slot, event in enumerate(events)}

        admit: list[tuple[int, int]] = []
        nonadmit: list[tuple[int, int]] = []
        thirds: list[tuple[tuple[int, int], int]] = []
        for i, j in unordered_pairs():
            record = dataset.record(events[i], events[j])
            mirrored = dataset.record(events[j], events[i])
            if record.admitted != mirrored.admitted:
                raise AssertionError("certified admission is not symmetric")
            if not record.admitted:
                nonadmit.append((i, j))
                continue
            if record.target_index != mirrored.target_index:
                raise AssertionError("certified forced third is not direction-neutral")
            third = slot_of.get(record.target_index)
            if third is None:
                raise AssertionError("certified forced third left its own habitat")
            admit.append((i, j))
            thirds.append(((i, j), third))

        blocks: list[tuple[int, int, int]] = []
        chambers: list[int] = []
        for bid, block_members in enumerate(catalogue.blocks):
            if not set(block_members) <= set(events):
                continue
            blocks.append(tuple(sorted(slot_of[e] for e in block_members)))
            chambers.append(chamber_of_block[bid][1])
        order = sorted(range(len(blocks)), key=lambda k: blocks[k])
        blocks = [blocks[k] for k in order]
        chambers = [chambers[k] for k in order]
        if len(blocks) != TRIPLES_PER_FAMILY:
            raise AssertionError(f"habitat holds {len(blocks)} blocks, not 4")
        if sorted(chambers) != [0, 1, 2, 3]:
            raise AssertionError(f"habitat chambers {chambers} are not F_2^2")

        presentation: list[tuple[int, int]] = []
        for slot in SLOTS:
            incident = tuple(
                chambers[k] for k, block in enumerate(blocks) if slot in block
            )
            if len(incident) != PINS["blocks_per_event"]:
                raise AssertionError("an Event is not incident to exactly 2 blocks")
            presentation.append(tuple(sorted(incident)))

        key = habitat_label(*catalogue.habitats[events[0]])
        problems.append(
            LocalProblem(
                key=key,
                s=(0 if catalogue.habitats[events[0]][1] == 1 else 1),
                fff=catalogue.habitats[events[0]][0],
                events=events,
                admit=tuple(admit),
                nonadmit=tuple(nonadmit),
                certified_third=tuple(thirds),
                certified_blocks=tuple(blocks),
                chamber_of_block=tuple(chambers),
                certified_presentation=tuple(presentation),
            )
        )

    if len(problems) != PINS["habitats"]:
        raise AssertionError(f"built {len(problems)} local problems, not 14")
    return tuple(problems)


# ---------------------------------------------------------------------------
# the admission graph, its faces, and its block systems
# ---------------------------------------------------------------------------

def adjacency(admit: tuple[tuple[int, int], ...]) -> tuple[frozenset[int], ...]:
    """Slot -> its admitted neighbours, as a frozen adjacency."""

    out: list[set[int]] = [set() for _ in SLOTS]
    for a, b in admit:
        out[a].add(b)
        out[b].add(a)
    return tuple(frozenset(s) for s in out)


def cliques3(admit: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    """Every 3-token clique of the admission graph: the triangular faces."""

    edges = {frozenset(p) for p in admit}
    return tuple(
        triple
        for triple in itertools.combinations(SLOTS, 3)
        if all(frozenset(p) in edges for p in itertools.combinations(triple, 2))
    )


def is_block_system(
    system: tuple[tuple[int, int, int], ...],
    admit: tuple[tuple[int, int], ...],
) -> bool:
    """Four triples covering every Event twice and every admitted pair once."""

    if len(set(system)) != TRIPLES_PER_FAMILY:
        return False
    occurrences = {slot: 0 for slot in SLOTS}
    covered: dict[frozenset[int], int] = {frozenset(p): 0 for p in admit}
    for triple in system:
        if len(set(triple)) != PINS["events_per_block"]:
            return False
        for slot in triple:
            occurrences[slot] += 1
        for pair in itertools.combinations(triple, 2):
            key = frozenset(pair)
            if key not in covered:
                return False
            covered[key] += 1
    if any(count != PINS["blocks_per_event"] for count in occurrences.values()):
        return False
    return all(count == 1 for count in covered.values())


def compatible_block_systems(
    admit: tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Every block system compatible with the ADMISSION RELATION ALONE.

    Exhaustive over the ``C(faces, 4)`` subsets of the triangular faces. No
    reference to the certified blocks, the SFP chart, or any expectation about how
    many systems there are.
    """

    faces = cliques3(admit)
    found = [
        tuple(sorted(system))
        for system in itertools.combinations(faces, TRIPLES_PER_FAMILY)
        if is_block_system(tuple(system), admit)
    ]
    return tuple(sorted(set(found)))


def system_third(
    system: tuple[tuple[int, int, int], ...], pair: tuple[int, int]
) -> int | None:
    """The unique third of ``pair`` inside a block system, or ``None``."""

    wanted = set(pair)
    hits = [t for t in system if wanted <= set(t)]
    if len(hits) != 1:
        return None
    return next(iter(set(hits[0]) - wanted))


def automorphisms(
    admit: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, ...], ...]:
    """Every token relabeling that preserves the admission edge set.

    Brute force over all 720 relabelings. Nothing about the octahedron is assumed.
    """

    edges = {frozenset(p) for p in admit}
    out = []
    for perm in all_relabelings():
        if {frozenset((perm[a], perm[b])) for a, b in admit} == edges:
            out.append(perm)
    return tuple(out)


def certified_relabelings(
    problem: LocalProblem,
) -> tuple[tuple[int, ...], ...]:
    """Token relabelings induced by relabelling the four certified cyclic blocks.

    An Event is incident to exactly two certified blocks, so a permutation of the
    four blocks carries the Event with block set ``{u, v}`` onto the Event with
    block set ``{sigma u, sigma v}``. All ``4! = 24`` block relabelings are applied
    and the induced token permutations collected. This is derived from the
    certified incidence, not from the chart and not from ``AGL(2,2)``.
    """

    blocks = problem.certified_blocks
    slot_of_blockpair: dict[frozenset[int], int] = {}
    for slot in SLOTS:
        incident = frozenset(k for k, block in enumerate(blocks) if slot in block)
        if len(incident) != PINS["blocks_per_event"]:
            raise AssertionError("an Event is not incident to exactly 2 blocks")
        if incident in slot_of_blockpair:
            raise AssertionError("two Events claim one certified block pair")
        slot_of_blockpair[incident] = slot

    out = []
    for sigma in itertools.permutations(range(TRIPLES_PER_FAMILY)):
        perm = [0] * N_TOKENS
        for incident, slot in slot_of_blockpair.items():
            image = frozenset(sigma[k] for k in incident)
            perm[slot] = slot_of_blockpair[image]
        if sorted(perm) != list(SLOTS):
            raise AssertionError(f"induced relabeling {perm} is not a permutation")
        out.append(tuple(perm))
    return tuple(sorted(set(out)))


def apply_perm_triple(
    perm: tuple[int, ...], triple: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(sorted(perm[t] for t in triple))


def apply_perm_system(
    perm: tuple[int, ...], system: tuple[tuple[int, int, int], ...]
) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted(apply_perm_triple(perm, t) for t in system))


# ---------------------------------------------------------------------------
# the anchor rule -- NONLEARNED CEILING ONLY
# ---------------------------------------------------------------------------

def anchor_third(
    admit: tuple[tuple[int, int], ...],
    anchor: tuple[int, int, int],
    pair: tuple[int, int],
) -> int | None:
    """``009.09`` section 7.1's exact one-anchor completion, in one line.

    Among the common admitted neighbours of ``pair``, the one whose triple meets
    the anchor in an **odd** number of tokens.

    ``009.09a`` section 1 writes the rule as ``|{a, b, c} & anchor| == 1``, which is
    exactly right on the domain it names -- the scored queries, where the anchor
    contributes at most one of ``a``, ``b``. Section 6 of the task additionally
    queries the three **support** pairs in order to reconstruct the whole four-block
    relation, and there ``{a, b}`` already lies inside the anchor, so the correct
    answer meets the anchor in three tokens rather than one. Odd parity is the form
    that covers all twelve admitted pairs with one predicate; the ``== 1`` form is
    carried alongside it and verified to agree on the nine scored pairs.

    Nonlearned, and a **ceiling**: quoted only as the information-theoretic bound
    the learner is measured against, never as a competing learned baseline, and no
    part of it appears in any learned path.
    """

    nbrs = adjacency(admit)
    a, b = pair
    common = sorted(nbrs[a] & nbrs[b])
    hits = [c for c in common if len({a, b, c} & set(anchor)) % 2 == 1]
    if len(hits) != 1:
        return None
    return hits[0]


def anchor_third_unit_form(
    admit: tuple[tuple[int, int], ...],
    anchor: tuple[int, int, int],
    pair: tuple[int, int],
) -> int | None:
    """The ``009.09a`` section 1 rule verbatim: ``|{a, b, c} & anchor| == 1``."""

    nbrs = adjacency(admit)
    a, b = pair
    common = sorted(nbrs[a] & nbrs[b])
    hits = [c for c in common if len({a, b, c} & set(anchor)) == 1]
    if len(hits) != 1:
        return None
    return hits[0]


def scored_query_pairs(
    admit: tuple[tuple[int, int], ...], anchor: tuple[int, int, int]
) -> tuple[tuple[int, int], ...]:
    """Admitted pairs whose answer is not directly contained in the anchor.

    ``009.09`` section 4.4: if the anchor is ``{a, b, c}`` then ``{a,b}``, ``{a,c}``
    and ``{b,c}`` are support, not scored queries.
    """

    inside = {frozenset(p) for p in itertools.combinations(anchor, 2)}
    return tuple(p for p in admit if frozenset(p) not in inside)


def support_query_pairs(
    admit: tuple[tuple[int, int], ...], anchor: tuple[int, int, int]
) -> tuple[tuple[int, int], ...]:
    inside = {frozenset(p) for p in itertools.combinations(anchor, 2)}
    return tuple(p for p in admit if frozenset(p) in inside)


# ---------------------------------------------------------------------------
# Gate 0
# ---------------------------------------------------------------------------

def gate0_one_problem(problem: LocalProblem) -> dict:
    """The exact identifiability census for one abstract six-token problem."""

    admit = problem.admit
    pairs = unordered_pairs()
    nbrs = adjacency(admit)
    faces = cliques3(admit)
    systems = compatible_block_systems(admit)
    auts = automorphisms(admit)
    induced = certified_relabelings(problem)

    certified = tuple(sorted(problem.certified_blocks))
    certified_in_systems = certified in systems
    duals = tuple(s for s in systems if s != certified)

    # --- the admission graph itself, characterized rather than named ----------
    degrees = sorted(len(n) for n in nbrs)
    complement = tuple(sorted(frozenset(p) for p in problem.nonadmit))
    complement_is_perfect_matching = (
        len(complement) == 3
        and len({slot for pair in complement for slot in pair}) == N_TOKENS
    )

    # --- admission-only ambiguity (section 3.2) ------------------------------
    common_counts = sorted(
        {len(nbrs[a] & nbrs[b]) for a, b in admit}
    )
    disagree = []
    for a, b in admit:
        answers = {system_third(system, (a, b)) for system in systems}
        disagree.append(len(answers) == len(systems) and None not in answers)
    admission_only_disagreement = all(disagree)

    # Bayes rate under a uniform prior over the compatible systems: for each
    # admitted pair, the best fixed answer is chosen and the mass of systems
    # agreeing with it is counted. Enumerated, not argued.
    bayes_hits = 0
    for a, b in admit:
        tally: dict[int, int] = {}
        for system in systems:
            third = system_third(system, (a, b))
            tally[third] = tally.get(third, 0) + 1
        bayes_hits += max(tally.values())
    admission_only_bayes = fraction(bayes_hits, len(admit) * len(systems))

    # The equivariance witness: for each admitted pair, an automorphism fixing
    # both query tokens and swapping the two candidate thirds. Its existence means
    # any permutation-equivariant scorer must score the two candidates equally, so
    # the ceiling is 1/2 by symmetry as well as by prior mass.
    witnesses = {}
    witness_found = True
    for a, b in admit:
        common = sorted(nbrs[a] & nbrs[b])
        if len(common) != 2:
            witness_found = False
            witnesses[encode_pair((a, b))] = None
            continue
        c1, c2 = common
        found = None
        for perm in auts:
            if perm[a] == a and perm[b] == b and perm[c1] == c2 and perm[c2] == c1:
                found = encode_perm(perm)
                break
        witnesses[encode_pair((a, b))] = found
        if found is None:
            witness_found = False

    # --- same-family law (009.09a section 1) ---------------------------------
    same_family_shares_one = True
    cross_family_shares_one = False
    for system in systems:
        for left, right in itertools.combinations(system, 2):
            if len(set(left) & set(right)) != 1:
                same_family_shares_one = False
    if len(systems) == 2:
        for left in systems[0]:
            for right in systems[1]:
                if len(set(left) & set(right)) == 1:
                    cross_family_shares_one = True

    # --- one-anchor identifiability (section 3.3) ---------------------------
    per_anchor = []
    scored_hits = 0
    scored_total = 0
    allpairs_hits = 0
    allpairs_total = 0
    certified_rule_hits = 0
    certified_rule_total = 0
    unit_form_hits = 0
    for anchor in certified:
        compatible = tuple(s for s in systems if anchor in s)
        scored = scored_query_pairs(admit, anchor)
        support = support_query_pairs(admit, anchor)
        unique_scored = 0
        unique_all = 0
        matches = 0
        for pair in admit:
            third = anchor_third(admit, anchor, pair)
            allpairs_total += 1
            if third is not None:
                unique_all += 1
                allpairs_hits += 1
            if third == problem.third(pair):
                matches += 1
        for pair in scored:
            scored_total += 1
            certified_rule_total += 1
            third = anchor_third(admit, anchor, pair)
            if third is not None:
                unique_scored += 1
                scored_hits += 1
            if third == problem.third(pair):
                certified_rule_hits += 1
            if anchor_third_unit_form(admit, anchor, pair) == problem.third(pair):
                unit_form_hits += 1
        per_anchor.append(
            {
                "anchor": encode_triple(anchor),
                "compatible_block_systems": len(compatible),
                "recovers_certified_family": compatible == (certified,),
                "scored_query_pairs": len(scored),
                "support_query_pairs": len(support),
                "scored_pairs_with_a_unique_forced_third": unique_scored,
                "admitted_pairs_with_a_unique_forced_third": unique_all,
                "rule_matches_certified_third_on_all_admitted_pairs": (
                    matches == len(admit)
                ),
            }
        )

    # --- the dual family, as a control only ---------------------------------
    dual_anchor_rows = []
    for system in duals:
        for anchor in system:
            scored = scored_query_pairs(admit, anchor)
            compatible = tuple(s for s in systems if anchor in s)
            agreements = sum(
                1
                for pair in scored
                if anchor_third(admit, anchor, pair) == system_third(system, pair)
            )
            dual_anchor_rows.append(
                {
                    "anchor": encode_triple(anchor),
                    "family": "combinatorial_dual",
                    "compatible_block_systems": len(compatible),
                    "scored_query_pairs": len(scored),
                    "rule_matches_dual_third_on_every_scored_pair": (
                        agreements == len(scored)
                    ),
                }
            )

    return {
        "habitat": problem.key,
        "sfp_habitat": problem.sfp_key,
        "problem_sha256": problem.sha256(),
        "tokens": N_TOKENS,
        "unordered_pairs": len(pairs),
        "admitted_pairs": len(admit),
        "nonadmitted_pairs": len(problem.nonadmit),
        "admission_degree_spectrum": degrees,
        "admission_is_regular": len(set(degrees)) == 1,
        "complement_is_perfect_matching": complement_is_perfect_matching,
        "triangular_faces": len(faces),
        "aut_order": len(auts),
        "certified_relabeling_subgroup_order": len(induced),
        "certified_relabelings_are_automorphisms": all(
            perm in set(auts) for perm in induced
        ),
        "subgroup_index": (
            len(auts) // len(induced) if len(induced) else None
        ),
        "block_systems_compatible_with_admission": len(systems),
        "certified_family_is_compatible": certified_in_systems,
        "dual_families": len(duals),
        "common_neighbour_counts": common_counts,
        "admission_only_systems_disagree_everywhere": admission_only_disagreement,
        "admission_only_bayes_forced_third": rd(admission_only_bayes),
        "admission_only_equivariance_witness_for_every_pair": witness_found,
        "admission_only_equivariance_witnesses": witnesses,
        "same_family_faces_share_exactly_one_event": same_family_shares_one,
        "cross_family_faces_ever_share_exactly_one_event": cross_family_shares_one,
        "one_anchor": {
            "per_anchor": per_anchor,
            "eligible_anchors": len(certified),
            "forced_third_ceiling": rd(fraction(scored_hits, scored_total)),
            "all_admitted_pairs_ceiling": rd(fraction(allpairs_hits, allpairs_total)),
            "rule_matches_certified_third_on_every_scored_pair": (
                certified_rule_hits == certified_rule_total
            ),
            "unit_form_agrees_on_every_scored_pair": (
                unit_form_hits == certified_rule_total
            ),
            "scored_pairs_checked": certified_rule_total,
            "domain_note": (
                "forced_third_ceiling is over the SCORED queries, which is the "
                "quantity 009.09 section 3.3 requires to be 1.000. "
                "all_admitted_pairs_ceiling additionally covers the three support "
                "pairs, which section 6 reconstructs but does not score."
            ),
        },
        "dual_anchor_control": dual_anchor_rows,
    }


def gate0_census(problems: tuple[LocalProblem, ...]) -> dict:
    """Gate 0 over all 14 habitats, plus the pins ``009.09`` section 3.2 names."""

    rows = [gate0_one_problem(problem) for problem in problems]

    def every(key: str) -> bool:
        return all(bool(row[key]) for row in rows)

    def unanimous(key: str) -> object:
        values = {json.dumps(row[key], sort_keys=True) for row in rows}
        return json.loads(values.pop()) if len(values) == 1 else None

    aut_orders = sorted({row["aut_order"] for row in rows})
    sub_orders = sorted({row["certified_relabeling_subgroup_order"] for row in rows})
    system_counts = sorted(
        {row["block_systems_compatible_with_admission"] for row in rows}
    )
    face_counts = sorted({row["triangular_faces"] for row in rows})
    one_anchor = sorted({row["one_anchor"]["forced_third_ceiling"] for row in rows})
    admission_only = sorted(
        {row["admission_only_bayes_forced_third"] for row in rows}
    )

    pins = {
        "tokens": pin(PINS["tokens"], unanimous("tokens"), "enumerated per habitat"),
        "unordered_pairs": pin(
            PINS["unordered_pairs"], unanimous("unordered_pairs"), "C(6,2)"
        ),
        "admitted_pairs": pin(
            PINS["admitted_pairs"],
            unanimous("admitted_pairs"),
            "certified admission restricted to the habitat",
        ),
        "nonadmitted_pairs": pin(
            PINS["nonadmitted_pairs"],
            unanimous("nonadmitted_pairs"),
            "certified non-admission restricted to the habitat",
        ),
        "token_relabelings": pin(
            PINS["token_relabelings"], len(all_relabelings()), "6! enumerated"
        ),
        "admission_automorphism_order": pin(
            PINS["admission_automorphism_order"],
            aut_orders[0] if len(aut_orders) == 1 else aut_orders,
            "brute force over all 720 relabelings",
        ),
        "certified_relabeling_subgroup_order": pin(
            PINS["certified_relabeling_subgroup_order"],
            sub_orders[0] if len(sub_orders) == 1 else sub_orders,
            "the 24 certified cyclic-block relabelings, induced on Events",
        ),
        "triangular_faces": pin(
            PINS["triangular_faces"],
            face_counts[0] if len(face_counts) == 1 else face_counts,
            "3-token cliques of the admission graph",
        ),
        "block_systems_compatible_with_admission": pin(
            PINS["block_systems_compatible_with_admission"],
            system_counts[0] if len(system_counts) == 1 else system_counts,
            "exhaustive over C(faces,4)",
        ),
        "common_neighbours_of_an_admitted_pair": pin(
            [PINS["common_neighbours_of_an_admitted_pair"]],
            unanimous("common_neighbour_counts"),
            "admission graph, per admitted pair",
        ),
        "admission_only_forced_third_ceiling": pin(
            EXPECTED_CEILINGS["admission_only_forced_third"],
            admission_only[0] if len(admission_only) == 1 else admission_only,
            "uniform prior over the compatible block systems",
        ),
        "one_anchor_forced_third_ceiling": pin(
            EXPECTED_CEILINGS["one_anchor_forced_third"],
            one_anchor[0] if len(one_anchor) == 1 else one_anchor,
            "exact completion under admission plus one certified anchor",
        ),
        "habitats": pin(PINS["habitats"], len(rows), "task.Catalogue habitats"),
    }

    laws = {
        "admission_graph_is_4_regular_on_6_tokens": every("admission_is_regular")
        and unanimous("admission_degree_spectrum")
        == [PINS["admission_graph_degree"]] * N_TOKENS,
        "nonadmission_is_a_perfect_matching": every("complement_is_perfect_matching"),
        "certified_relabelings_are_admission_automorphisms": every(
            "certified_relabelings_are_automorphisms"
        ),
        "certified_subgroup_has_index_two": all(
            row["subgroup_index"] == 2 for row in rows
        ),
        "certified_family_is_admission_compatible": every(
            "certified_family_is_compatible"
        ),
        "admission_alone_leaves_exactly_one_dual_family": all(
            row["dual_families"] == 1 for row in rows
        ),
        "the_two_families_disagree_on_every_admitted_pair": every(
            "admission_only_systems_disagree_everywhere"
        ),
        "admission_only_ceiling_is_one_half": all(
            row["admission_only_bayes_forced_third"] == 0.5 for row in rows
        ),
        "every_admitted_pair_has_a_candidate_swapping_automorphism": every(
            "admission_only_equivariance_witness_for_every_pair"
        ),
        "same_family_faces_share_exactly_one_event": every(
            "same_family_faces_share_exactly_one_event"
        ),
        "cross_family_faces_never_share_exactly_one_event": not any(
            row["cross_family_faces_ever_share_exactly_one_event"] for row in rows
        ),
        "one_certified_anchor_determines_the_certified_family": all(
            all(entry["recovers_certified_family"] for entry in row["one_anchor"]["per_anchor"])
            for row in rows
        ),
        "one_anchor_ceiling_is_one": all(
            row["one_anchor"]["forced_third_ceiling"] == 1.0 for row in rows
        ),
        "one_anchor_all_pairs_ceiling_is_one": all(
            row["one_anchor"]["all_admitted_pairs_ceiling"] == 1.0 for row in rows
        ),
        "one_anchor_rule_reproduces_the_certified_third": all(
            row["one_anchor"]["rule_matches_certified_third_on_every_scored_pair"]
            for row in rows
        ),
        "advisory_unit_form_agrees_on_the_scored_domain": all(
            row["one_anchor"]["unit_form_agrees_on_every_scored_pair"] for row in rows
        ),
        "nine_scored_queries_per_episode": all(
            all(
                entry["scored_query_pairs"] == SCORED_QUERIES_PER_EPISODE
                for entry in row["one_anchor"]["per_anchor"]
            )
            for row in rows
        ),
        "dual_anchor_rule_reproduces_the_dual_third": all(
            all(
                entry["rule_matches_dual_third_on_every_scored_pair"]
                for entry in row["dual_anchor_control"]
            )
            for row in rows
        ),
    }

    gate_open = (
        all(entry["agrees"] for entry in pins.values())
        and all(laws.values())
    )
    return {
        "per_habitat": rows,
        "pins": pins,
        "laws": laws,
        "premise_check": {
            "admission_alone_would_already_determine_the_family": any(
                row["block_systems_compatible_with_admission"] == 1 for row in rows
            ),
            "note": (
                "009.09 section 3.2 requires the task to STOP if admission alone "
                "already determines the certified family uniquely. It does not: two "
                "systems are compatible in every habitat."
            ),
        },
        "one_anchor_sufficiency": {
            "sufficient": laws["one_anchor_ceiling_is_one"]
            and laws["one_certified_anchor_determines_the_certified_family"],
            "note": (
                "009.09 section 3.3 permits the primary experiment only if admission "
                "plus one genuine block anchor gives every withheld admitted pair a "
                "unique forced third, i.e. a 1.000 ceiling."
            ),
        },
        "gate_open": gate_open,
        "gate_statement": (
            "Gate 0 PASSES: admission alone leaves exactly two compatible block "
            "systems and a 1/2 ceiling; one certified anchor selects the certified "
            "family uniquely and lifts the ceiling to 1.000."
            if gate_open
            else "Gate 0 FAILS: at least one pin or law disagrees; do not train."
        ),
        "digest": digest([row["problem_sha256"] for row in rows]),
    }


# ---------------------------------------------------------------------------
# episodes (section 3.4, frozen before training)
# ---------------------------------------------------------------------------

def relabeling_indices(label: str, count: int) -> tuple[int, ...]:
    """``count`` distinct relabeling indices, derived from a digest of ``label``.

    Deterministic and hash-free: a start offset and a stride coprime to 720 are
    read off the digest, so the indices are distinct by construction and no RNG
    state, set iteration order or ``hash()`` participates.
    """

    seed = hash_free_seed(label)
    total = len(all_relabelings())
    start = seed % total
    stride = 1 + ((seed // total) % (total - 1))
    while math.gcd(stride, total) != 1:
        stride += 1
        if stride >= total:
            raise AssertionError("no stride coprime to the relabeling count")
    picked = tuple((start + k * stride) % total for k in range(count))
    if len(set(picked)) != count:
        raise AssertionError("relabeling indices are not distinct")
    return picked


@dataclass(frozen=True)
class Episode:
    """One opaque local-chart episode.

    ``perm[slot] = node``: the fresh opaque relabeling. Everything the learner
    sees is expressed in NODES. ``habitat``, ``s`` and ``fff`` are manifest
    bookkeeping and, per ``009.09a`` section 4, are not fed to the network.
    """

    episode_id: str
    habitat: str
    s: int
    fff: int
    anchor_family: str                 # "certified" | "combinatorial_dual"
    anchor_index: int
    perm: tuple[int, ...]              # slot -> node
    relabeling_index: int
    admit_nodes: tuple[tuple[int, int], ...]
    anchor_nodes: tuple[int, int, int]
    scored_queries: tuple[tuple[int, int], ...]
    support_queries: tuple[tuple[int, int], ...]
    certified_third_nodes: tuple[tuple[tuple[int, int], int], ...]
    dual_third_nodes: tuple[tuple[tuple[int, int], int], ...]
    certified_blocks_nodes: tuple[tuple[int, int, int], ...]
    dual_blocks_nodes: tuple[tuple[int, int, int], ...]
    certified_presentation_nodes: tuple[tuple[int, int], ...]

    def inverse(self) -> tuple[int, ...]:
        out = [0] * N_TOKENS
        for slot, node in enumerate(self.perm):
            out[node] = slot
        return tuple(out)

    def as_row(self) -> list:
        return [
            self.episode_id,
            self.habitat,
            self.anchor_family,
            self.anchor_index,
            list(self.perm),
            self.relabeling_index,
            [list(p) for p in self.admit_nodes],
            list(self.anchor_nodes),
            [list(p) for p in self.scored_queries],
            [[list(p), t] for p, t in self.certified_third_nodes],
            [[list(p), t] for p, t in self.dual_third_nodes],
            [list(t) for t in self.certified_blocks_nodes],
            [list(t) for t in self.dual_blocks_nodes],
        ]

    def sha256(self) -> str:
        return digest(self.as_row())


def _permuted_thirds(
    perm: tuple[int, ...], thirds: tuple[tuple[tuple[int, int], int], ...]
) -> tuple[tuple[tuple[int, int], int], ...]:
    out = []
    for (a, b), third in thirds:
        pair = tuple(sorted((perm[a], perm[b])))
        out.append((pair, perm[third]))
    return tuple(sorted(out))


def build_episode(
    problem: LocalProblem,
    systems: tuple[tuple[tuple[int, int, int], ...], ...],
    *,
    family: str,
    anchor_index: int,
    relabeling_index: int,
    namespace: str,
) -> Episode:
    """One episode, built entirely from frozen deterministic inputs."""

    certified = tuple(sorted(problem.certified_blocks))
    duals = [s for s in systems if s != certified]
    if len(duals) != 1:
        raise AssertionError("expected exactly one dual family; Gate 0 must gate this")
    dual = duals[0]

    source = certified if family == "certified" else dual
    anchor = source[anchor_index]

    perm = all_relabelings()[relabeling_index]
    certified_thirds = tuple(
        (pair, problem.third(pair)) for pair in problem.admit
    )
    dual_thirds = tuple(
        (pair, system_third(dual, pair)) for pair in problem.admit
    )
    if any(third is None for _, third in dual_thirds):
        raise AssertionError("the dual family does not cover every admitted pair")

    admit_nodes = tuple(
        sorted(tuple(sorted((perm[a], perm[b]))) for a, b in problem.admit)
    )
    anchor_nodes = apply_perm_triple(perm, anchor)
    scored = tuple(
        sorted(
            tuple(sorted((perm[a], perm[b])))
            for a, b in scored_query_pairs(problem.admit, anchor)
        )
    )
    support = tuple(
        sorted(
            tuple(sorted((perm[a], perm[b])))
            for a, b in support_query_pairs(problem.admit, anchor)
        )
    )
    presentation = [None] * N_TOKENS
    for slot, chambers in enumerate(problem.certified_presentation):
        presentation[perm[slot]] = chambers

    return Episode(
        episode_id=f"{namespace}/{problem.key}/{family}/{anchor_index}/{relabeling_index}",
        habitat=problem.key,
        s=problem.s,
        fff=problem.fff,
        anchor_family=family,
        anchor_index=anchor_index,
        perm=perm,
        relabeling_index=relabeling_index,
        admit_nodes=admit_nodes,
        anchor_nodes=anchor_nodes,
        scored_queries=scored,
        support_queries=support,
        certified_third_nodes=_permuted_thirds(perm, certified_thirds),
        dual_third_nodes=_permuted_thirds(perm, dual_thirds),
        certified_blocks_nodes=apply_perm_system(perm, certified),
        dual_blocks_nodes=apply_perm_system(perm, dual),
        certified_presentation_nodes=tuple(presentation),
    )


def build_episodes(
    problems: tuple[LocalProblem, ...],
    *,
    namespace: str = "primary",
    family: str = "certified",
    permutations: int = PERMUTATIONS_PER_ANCHOR,
) -> tuple[Episode, ...]:
    """The frozen episode manifest: 4 anchors x ``permutations`` relabelings.

    ``namespace`` separates the primary episodes from the section 7.5 fresh
    relabeling copies, so the two draw disjoint relabeling sequences from disjoint
    digest namespaces.
    """

    out: list[Episode] = []
    for problem in problems:
        systems = compatible_block_systems(problem.admit)
        for anchor_index in range(TRIPLES_PER_FAMILY):
            label = f"{namespace}/{family}/{problem.key}/anchor={anchor_index}"
            for relabeling_index in relabeling_indices(label, permutations):
                out.append(
                    build_episode(
                        problem,
                        systems,
                        family=family,
                        anchor_index=anchor_index,
                        relabeling_index=relabeling_index,
                        namespace=namespace,
                    )
                )
    return tuple(out)


def episode_manifest(episodes: tuple[Episode, ...]) -> dict:
    by_habitat: dict[str, int] = {}
    for episode in episodes:
        by_habitat[episode.habitat] = by_habitat.get(episode.habitat, 0) + 1
    return {
        "n_episodes": len(episodes),
        "episodes_per_habitat": sorted(set(by_habitat.values())),
        "scored_queries_per_episode": sorted(
            {len(e.scored_queries) for e in episodes}
        ),
        "support_queries_per_episode": sorted(
            {len(e.support_queries) for e in episodes}
        ),
        "admitted_pairs_per_episode": sorted({len(e.admit_nodes) for e in episodes}),
        "distinct_relabelings_used": len({e.relabeling_index for e in episodes}),
        "distinct_relabelings_per_habitat_anchor": sorted(
            {
                len(
                    {
                        e.relabeling_index
                        for e in episodes
                        if e.habitat == habitat and e.anchor_index == index
                    }
                )
                for habitat in by_habitat
                for index in range(TRIPLES_PER_FAMILY)
            }
        ),
        "sha256": digest([episode.as_row() for episode in episodes]),
    }


# ---------------------------------------------------------------------------
# folds (section 4.5)
# ---------------------------------------------------------------------------

def habitats_of_fold(dataset: Dataset, fold) -> tuple[str, ...]:
    """The habitat keys a fold holds out, read off its held-out Event indices."""

    catalogue = dataset.catalogue
    return tuple(
        sorted({habitat_label(*catalogue.habitats[e]) for e in fold.held_out_events})
    )


def episode_split(
    episodes: tuple[Episode, ...], held_out: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Train / test episode positions for one fold.

    An episode belongs entirely to one habitat, so holding out a habitat holds out
    every episode of it, and with it every forced-third query label from it.
    """

    blocked = set(held_out)
    train = tuple(i for i, e in enumerate(episodes) if e.habitat not in blocked)
    test = tuple(i for i, e in enumerate(episodes) if e.habitat in blocked)
    return train, test


def fold_manifest(
    dataset: Dataset, folds, episodes: tuple[Episode, ...]
) -> dict:
    rows = []
    for fold in folds:
        held = habitats_of_fold(dataset, fold)
        train, test = episode_split(episodes, held)
        rows.append(
            {
                "family": fold.family,
                "name": fold.name,
                "role": fold.role,
                "held_out_habitats": list(held),
                "held_out_events": list(fold.held_out_events),
                "held_out_fano_points": list(fold.held_out_fano_points),
                "train_episodes": len(train),
                "test_episodes": len(test),
                "train_scored_queries": sum(
                    len(episodes[i].scored_queries) for i in train
                ),
                "test_scored_queries": sum(
                    len(episodes[i].scored_queries) for i in test
                ),
                "upstream_fold_sha256": fold.sha256(),
                "train_sha256": digest(list(train)),
                "test_sha256": digest(list(test)),
            }
        )
    return {
        "n_folds": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "per_fold": rows,
        "holdout_character": (
            "009.09a section 5: LOHO and LOFPO here hold out a fresh relabelling of "
            "the same octahedron with a fresh anchor, not a new structure. The "
            "result is fresh-relabelling generalization of a local relational rule, "
            "NOT held-out-structure generalization in the 009.06/009.08 sense where "
            "the held-out habitat's code words never appeared in training."
        ),
        "within_episode_support_is_not_leakage": (
            "009.09 section 4.5: at test time the held-out habitat is allowed its "
            "declared within-episode relational support -- its admission graph plus "
            "one anchor triple. That support is the object of study. No forced-third "
            "query label from a held-out habitat enters optimization."
        ),
        "sha256": digest([row["train_sha256"] + row["test_sha256"] for row in rows]),
    }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_label_source_fence() -> dict:
    """Mechanically: no label or support function here can be handed a circuit."""

    banned = ("circuit", "oracle", "sfp_circuit", "exactsfpcircuit")
    rows = []
    offenders = []
    for func in (
        build_local_problems,
        build_episode,
        build_episodes,
        anchor_third,
        scored_query_pairs,
        compatible_block_systems,
        episode_split,
    ):
        signature = inspect.signature(func)
        parameters = []
        for name, parameter in signature.parameters.items():
            annotation = (
                ""
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation)
            )
            suspect = any(token in f"{name} {annotation}".lower() for token in banned)
            parameters.append(
                {"name": name, "annotation": annotation, "suspect": suspect}
            )
            if suspect:
                offenders.append({"function": func.__name__, "parameter": name})
        rows.append(
            {
                "function": func.__name__,
                "signature": str(signature),
                "parameters": parameters,
            }
        )
    module = inspect.getmodule(build_local_problems)
    tree = ast.parse(inspect.getsource(module))
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.alias):
            identifiers.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    banned_identifiers = sorted(
        name
        for name in identifiers
        if "circuit" in name.lower() or "oracle" in name.lower()
    )
    return {
        "label_functions": rows,
        "banned_parameter_tokens": list(banned),
        "offenders": offenders,
        "no_label_function_accepts_a_circuit": not offenders,
        "module_namespace_has_no_circuit_binding": not any(
            "circuit" in name.lower() or "oracle" in name.lower()
            for name in vars(module)
        ),
        "banned_identifiers_in_module_code": banned_identifiers,
        "module_code_never_names_the_exact_circuit": not banned_identifiers,
        "circuit_fence_note": (
            "the check is over the module's parsed IDENTIFIERS -- every Name, "
            "Attribute, import alias, function and class -- not over its prose. The "
            "docstring and the fence strings name ExactSfpCircuit precisely in order "
            "to state that it is absent, and a text scan would be tripped by the "
            "denial itself."
        ),
        "label_provenance_chain": [
            "topographo.ssd.fips_basic.admissible / third -- the certified relation",
            "task.build_catalogue -> Catalogue.block_ids / Catalogue.blocks",
            "task.label_pair -> PairRecord.label, PairRecord.target_index",
            (
                "discovery_task.build_local_problems restricts those to one habitat "
                "and re-expresses them in local slots"
            ),
            (
                "discovery_task.build_episode permutes slots onto opaque nodes; the "
                "learner sees nodes, an admission matrix and one anchor triple"
            ),
        ],
        "dual_family_provenance": (
            "the dual thirds are computed by discovery_task.system_third from the "
            "second admission-compatible block system. They are COMBINATORIAL, not "
            "certified FIPS labels, and are used only by the 009.09a section 2 "
            "control arm."
        ),
    }


def verify_direction_neutrality(
    dataset: Dataset, problems: tuple[LocalProblem, ...]
) -> dict:
    """``009.09`` section 4.4: swapped queries must have identical answers.

    Re-consults the certified relation in BOTH orders for every one of the 14 x 12
    admitted local pairs, rather than trusting that the local problem stored a
    sorted tuple.
    """

    unordered = all(all(a < b for a, b in problem.admit) for problem in problems)
    checked = 0
    admission_symmetric = True
    third_symmetric = True
    for problem in problems:
        for a, b in problem.admit:
            forward = dataset.record(problem.events[a], problem.events[b])
            backward = dataset.record(problem.events[b], problem.events[a])
            checked += 1
            if forward.admitted != backward.admitted:
                admission_symmetric = False
            if forward.target_index != backward.target_index:
                third_symmetric = False
    return {
        "ordered_pairs_rechecked": checked * 2,
        "admitted_pairs_stored_unordered": unordered,
        "certified_admission_is_symmetric": admission_symmetric,
        "forced_third_is_direction_neutral": third_symmetric,
        "note": (
            "every stored pair is a sorted 2-tuple and every lookup is by the sorted "
            "pair, so a swapped query is the SAME row, not a second row that happens "
            "to agree. The certified relation is nonetheless re-read in both "
            "directions here."
        ),
    }


def verify_episode_opacity(episodes: tuple[Episode, ...]) -> dict:
    """No opaque node name carries stable semantics across episodes."""

    per_habitat: dict[str, set[tuple[int, ...]]] = {}
    for episode in episodes:
        per_habitat.setdefault(episode.habitat, set()).add(episode.perm)

    # Does any node index keep the same certified Event across a habitat's
    # episodes? If a node were stably bound to an Event the task would leak.
    stable = []
    for habitat, perms in sorted(per_habitat.items()):
        for node in range(N_TOKENS):
            slots = {perm.index(node) for perm in perms}
            if len(slots) == 1:
                stable.append({"habitat": habitat, "node": node})

    identity_used = sum(1 for e in episodes if e.perm == tuple(SLOTS))
    return {
        "distinct_relabelings_per_habitat": sorted(
            {len(perms) for perms in per_habitat.values()}
        ),
        "nodes_stably_bound_to_one_event": stable,
        "no_node_is_stably_bound_to_one_event": not stable,
        "identity_relabeling_episodes": identity_used,
        "anchor_is_never_a_scored_query": all(
            not (set(e.scored_queries) & set(e.support_queries)) for e in episodes
        ),
        "every_scored_query_is_admitted": all(
            set(e.scored_queries) <= set(e.admit_nodes) for e in episodes
        ),
        "scored_plus_support_covers_every_admitted_pair": all(
            set(e.scored_queries) | set(e.support_queries) == set(e.admit_nodes)
            for e in episodes
        ),
    }


def verify_fresh_relabeling_namespace(
    primary: tuple[Episode, ...], fresh: tuple[Episode, ...]
) -> dict:
    """Section 7.5's stress-test copies must be an INDEPENDENT relabeling draw."""

    def keyed(episodes):
        return {
            (e.habitat, e.anchor_family, e.anchor_index): e.relabeling_index
            for e in episodes
        }

    left, right = keyed(primary), keyed(fresh)
    shared = sorted(set(left) & set(right))
    collisions = [k for k in shared if left[k] == right[k]]
    return {
        "cells_compared": len(shared),
        "relabeling_index_collisions": len(collisions),
        "independent_draw": not collisions,
        "note": (
            "the fresh copies are drawn from a separate digest namespace, so their "
            "relabeling sequence is independent of the primary one rather than a "
            "shifted copy of it."
        ),
    }


def verify_chart_recovery_machinery(problems: tuple[LocalProblem, ...]) -> dict:
    """Section 6.3, run on the CERTIFIED family, as an implementation check.

    Recovering the certified family and then searching ``AGL(2,2) ~= S4`` for a
    chamber relabelling that carries the derived presentation onto the hidden
    certified one must succeed by construction. Running it here, before any
    training, proves the audit's machinery is sound rather than proving anything
    about the learner. ``009.09a`` section 3 predicts exactly one witness.
    """

    rows = []
    for problem in problems:
        recovered = tuple(sorted(problem.certified_blocks))
        witnesses = sfp_equivalence_witnesses(recovered, problem.certified_presentation)
        rows.append(
            {
                "habitat": problem.key,
                "witnesses": len(witnesses),
                "equivalent": bool(witnesses),
            }
        )
    return {
        "per_habitat": rows,
        "certified_family_is_always_equivalent": all(row["equivalent"] for row in rows),
        "witness_count_is_always_one": all(row["witnesses"] == 1 for row in rows),
        "group_searched": "AGL(2,2) ~= S4, all 24 elements, exactly",
        "note": (
            "the arbitrary choice of affine origin and basis is not an error: any of "
            "the 24 chamber relabelings is certified-equivalent, and the search is "
            "over the finite group rather than a fitted continuous alignment."
        ),
    }


def sfp_equivalence_witnesses(
    recovered: tuple[tuple[int, int, int], ...],
    certified_presentation: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    """Every ``AGL(2,2)`` element carrying a recovered chart onto the certified one.

    ``recovered`` is a set of four triples over the six tokens. Its members are
    ordered canonically and identified with ``F_2^2`` in that order -- the
    "arbitrary coordinates" of ``009.09`` section 6.3. Each token is then derived
    as the unordered pair of recovered chambers it lies in, and the finite group is
    searched exhaustively for an element mapping that presentation onto the hidden
    certified ``PP`` presentation.
    """

    ordered = tuple(sorted(recovered))
    if len(ordered) != TRIPLES_PER_FAMILY:
        return ()
    derived: list[tuple[int, int] | None] = []
    for slot in SLOTS:
        incident = tuple(k for k, triple in enumerate(ordered) if slot in triple)
        if len(incident) != PINS["blocks_per_event"]:
            return ()
        derived.append(tuple(sorted(incident)))

    out = []
    for element in groups.agl_2_2():
        action = groups.chamber_permutation(element)
        image = tuple(
            tuple(sorted((action[pair[0]], action[pair[1]]))) for pair in derived
        )
        if image == tuple(certified_presentation):
            out.append(
                json.dumps(
                    {"matrix": [list(r) for r in element[0]], "b": list(element[1])},
                    sort_keys=True,
                )
            )
    return tuple(out)


def verify_upstream_pins(dataset: Dataset, codec: SfpCodec, folds) -> dict:
    counts = dataset.counts()
    return {
        "dataset_sha256": dataset.sha256(),
        "catalogue_sha256": dataset.catalogue.sha256(),
        "chart_sha256": codec.chart()["sha256"],
        "events": pin(84, len(dataset.catalogue), "task.Catalogue"),
        "ordered_pairs_total": pin(
            7056, counts["ordered_pairs_total"], "task.Dataset"
        ),
        "admit": pin(336, counts["admit"], "task.Dataset"),
        "habitats": pin(
            PINS["habitats"], len(codec.habitats), "sfp.SfpCodec.habitats"
        ),
        "structural_folds": pin(
            PINS["structural_folds"], len(folds), "folds.structural_folds"
        ),
        "loho_folds": pin(
            PINS["loho_folds"],
            sum(1 for f in folds if f.family == "LOHO"),
            "folds.build_loho_folds",
        ),
        "lofpo_folds": pin(
            PINS["lofpo_folds"],
            sum(1 for f in folds if f.family == "LOFPO"),
            "folds.build_lofpo_folds",
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    codec = SfpCodec()
    folds = structural_folds(all_folds(dataset))
    problems = build_local_problems(dataset, codec)

    gate = gate0_census(problems)
    if not gate["gate_open"]:
        # Section 3 forbids training before the gate closes. The artifact still
        # records the census so the failure is inspectable.
        pass

    episodes = build_episodes(problems, namespace="primary")
    fresh = build_episodes(problems, namespace="fresh_relabeling")
    dual = build_episodes(problems, namespace="dual_control", family="combinatorial_dual")

    upstream = verify_upstream_pins(dataset, codec, folds)
    manifests = {
        "primary": episode_manifest(episodes),
        "fresh_relabeling": episode_manifest(fresh),
        "dual_control": episode_manifest(dual),
    }
    folds_body = fold_manifest(dataset, folds, episodes)

    laws = {
        "gate_0_open": gate["gate_open"],
        "upstream_pins_agree": all(
            entry["agrees"] for entry in upstream.values() if isinstance(entry, dict)
        ),
        "every_habitat_has_48_episodes": manifests["primary"][
            "episodes_per_habitat"
        ]
        == [PINS["episodes_per_habitat"]],
        "nine_scored_queries_per_episode": manifests["primary"][
            "scored_queries_per_episode"
        ]
        == [SCORED_QUERIES_PER_EPISODE],
        "twelve_admitted_pairs_per_episode": manifests["primary"][
            "admitted_pairs_per_episode"
        ]
        == [PINS["admitted_pairs"]],
        "relabelings_distinct_within_every_habitat_anchor": manifests["primary"][
            "distinct_relabelings_per_habitat_anchor"
        ]
        == [PERMUTATIONS_PER_ANCHOR],
        "fold_count_is_21": folds_body["n_folds"] == PINS["structural_folds"],
        "no_held_out_habitat_appears_in_a_training_episode": all(
            row["train_episodes"] + row["test_episodes"] == len(episodes)
            and row["test_episodes"]
            == len(row["held_out_habitats"]) * PINS["episodes_per_habitat"]
            for row in folds_body["per_fold"]
        ),
    }

    opacity = verify_episode_opacity(episodes)
    freshness = verify_fresh_relabeling_namespace(episodes, fresh)
    fence = verify_label_source_fence()
    neutrality = verify_direction_neutrality(dataset, problems)
    machinery = verify_chart_recovery_machinery(problems)

    laws.update(
        {
            "no_opaque_node_is_stably_bound_to_an_event": opacity[
                "no_node_is_stably_bound_to_one_event"
            ],
            "anchor_pairs_are_never_scored": opacity["anchor_is_never_a_scored_query"],
            "scored_and_support_partition_the_admitted_pairs": opacity[
                "scored_plus_support_covers_every_admitted_pair"
            ],
            "fresh_relabelings_are_an_independent_draw": freshness["independent_draw"],
            "no_label_function_accepts_a_circuit": fence[
                "no_label_function_accepts_a_circuit"
            ],
            "module_code_never_names_the_exact_circuit": fence[
                "module_code_never_names_the_exact_circuit"
            ],
            "module_namespace_has_no_circuit_binding": fence[
                "module_namespace_has_no_circuit_binding"
            ],
            "forced_third_is_direction_neutral": neutrality[
                "forced_third_is_direction_neutral"
            ],
            "equivalence_audit_machinery_is_sound": machinery[
                "certified_family_is_always_equivalent"
            ]
            and machinery["witness_count_is_always_one"],
        }
    )

    agrees = all(laws.values())
    return {
        "module": "discovery_task",
        "base_commit": BASE_COMMIT,
        "executes": EXECUTED_TASK,
        "advisory_note": ADVISORY_NOTE,
        "prior_result": PRIOR_RESULT,
        "fences": list(FENCES),
        "supplied_to_the_learner": list(SUPPLIED),
        "withheld_from_the_learner": list(WITHHELD),
        "pins": PINS,
        "expected_ceilings": EXPECTED_CEILINGS,
        "upstream": upstream,
        "gate_0": gate,
        "episode_manifests": manifests,
        "episode_construction": {
            "anchors_per_habitat": TRIPLES_PER_FAMILY,
            "relabelings_per_anchor": PERMUTATIONS_PER_ANCHOR,
            "relabeling_source": (
                "digest-derived start offset and a stride coprime to 720, so the "
                "indices are distinct by construction; never hash(), never an RNG "
                "stream, never set iteration order"
            ),
            "namespaces": ["primary", "fresh_relabeling", "dual_control"],
            "s_and_fff": (
                "manifest bookkeeping only. 009.09a section 4 keeps them out of the "
                "network input; discovery_heads proves the input surface carries no "
                "habitat identity."
            ),
        },
        "folds": folds_body,
        "opacity": opacity,
        "fresh_relabeling_namespace": freshness,
        "label_source_fence": fence,
        "direction_neutrality": neutrality,
        "equivalence_audit_machinery": machinery,
        "torch": (
            "no tensor is constructed here; torch is inherited transitively through "
            "locator_task -> ladder_task -> harness"
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "laws": laws,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "Gate 0 passes with the expected 48/24 group orders, two "
                "admission-compatible block systems, a 1/2 admission-only ceiling "
                "and a 1.000 one-anchor ceiling; the episode, relabeling and fold "
                "manifests are frozen and digest-pinned; no label function can be "
                "handed a circuit"
                if agrees
                else "at least one Gate-0 pin or frozen-support law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    gate = result["gate_0"]
    print("--- Gate 0 ---", flush=True)
    for name, entry in sorted(gate["pins"].items()):
        flag = "ok " if entry["agrees"] else "FAIL"
        print(f"  {flag} {name:<48} {entry['observed']}", flush=True)
    print(f"  {gate['gate_statement']}", flush=True)
    print("--- frozen support ---", flush=True)
    for name, body in sorted(result["episode_manifests"].items()):
        print(
            f"  {name:<18} {body['n_episodes']} episodes  "
            f"sha256 {body['sha256'][:12]}",
            flush=True,
        )
    print(
        f"  folds              {result['folds']['n_folds']}  "
        f"sha256 {result['folds']['sha256'][:12]}",
        flush=True,
    )
    for name, value in sorted(result["laws"].items()):
        if not value:
            print(f"  FAIL law {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        expected = json.loads(OUTPUT.read_text())
        observed = json.loads(text)
        for key in ("started_at", "host"):
            expected.pop(key, None)
            observed.pop(key, None)
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived census is not identical to {OUTPUT.name} "
                "(started_at/host excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["laws"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()

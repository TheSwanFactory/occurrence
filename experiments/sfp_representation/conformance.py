"""Gate 0 conformance audit: codebase vs. Outcome 021.05 finite pins (Issue 009).

This module is a HARD GATE. Issue 009 may not proceed to any training or
representation-ablation work unless every pinned finite count and relation
recorded in Outcome 021.05 is independently re-derived here and found to agree
with the certified structures already living in ``topographo.ssd``.

It is deliberately torch-free (repo convention: torch-free modules are CI
testable) and depends only on the standard library plus the exact rational
layer ``topographo.ssd``.

Fences (read these before quoting any number out of the JSON)
------------------------------------------------------------
* ``algebraic zero != NONADMISSION;   0 != bottom``
* ``FFF=000 and pp=00 are formal completion values only, not Events``
* ``this audit certifies finite presentation/incidence only``
* ``it does NOT replace native OT evaluation (Occ / Cyc / Sand) or Event
  denotation``

What is reused rather than reimplemented
----------------------------------------
* ``topographo.ssd.fips_basic`` — the certified table: ``EVENTS`` (84),
  ``ORDERED_EDGES`` (336), ``BLOCKS`` (56), ``_THIRD_MAP``, ``TABLE_SHA256``.
* ``topographo.ssd.seal.public_label`` — the habitat key ``(P, delta)``.
* ``topographo.ssd.frames.is_event`` — Event certification.
* The per-habitat template is ``experiments/mutual-frames/audit.py::run``
  (Theory 068.02 census), extended here to every habitat rather than an
  aggregate assertion.

What is newly built here
------------------------
Nothing in this repository previously computed or asserted the **112 oriented
cyclic blocks** or the **168 Event-block incidences**. Both are added by this
module and are derived from the certified data structures, not from the
arithmetic identities ``112 = 2 * 56`` / ``168 = 84 * 2``:

* oriented blocks are built as genuine cyclic-rotation classes of certified
  ordered triples ``(a, b, third(a, b))``, and ``|O| = 2|T|`` is *observed*
  because the orientation bit is not recoverable from the unordered block;
* incidences are counted by enumerating actual ``(Event, block)`` membership.

The Fano-plane incidence data, ``GL(3,2)``, ``GL(2,2)``/``AGL(2,2)`` and the
K4/S4 relabelling machinery are also absent upstream and are built small and
self-contained below; a later task will generalize them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from collections import defaultdict
from pathlib import Path

from topographo.ssd import fips_basic
from topographo.ssd.frames import is_event
from topographo.ssd.seal import public_label

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "conformance_audit.json"

FENCES = (
    "algebraic zero != NONADMISSION;   0 != bottom",
    "FFF=000 and pp=00 are formal completion values only, not Events",
    "this audit certifies finite presentation/incidence only",
    "it does NOT replace native OT evaluation (Occ / Cyc / Sand) or Event denotation",
)

# Upstream Quilt pins, verbatim.
UPSTREAM_OUTCOME_RESULT = (
    "quilt+s3://protology#package=occurrence/outcome@"
    "f18b24ef35a3867309730f9917b1678b1d818bb1b5d87cc91291bf1e6105d446"
    "&path=issues/021-fips-warren-kit/"
    "021.05-Research-SFP-zero-completion-and-TLM-consequence-interface-result.md"
)
UPSTREAM_GPT_TASK = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "78b5436c25a9512f5d5792d65630061cdea136cae06f456951f7673000e8259f"
    "&path=issues/009-sfp-consequence-representation/"
    "009.01-Kiro-native-FIPS-vs-SFP-representation-ablation-task.md"
)

# Pinned like experiments/mutual-frames/audit.py pins ``base_commit``: recording
# a live ``git rev-parse HEAD`` would make the artifact non-replayable the moment
# anything is committed. The driver prints a warning when live HEAD differs.
BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"
EXPECTED_TABLE_SHA256 = (
    "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea"
)

# ---------------------------------------------------------------------------
# Outcome 021.05 pins
# ---------------------------------------------------------------------------

GLOBAL_PINS = {
    "basic_events": 84,
    "unordered_cyclic_blocks": 56,
    "oriented_cyclic_blocks": 112,
    "event_block_incidences": 168,
    "ordered_cyclic_edges": 336,
    "pasch_habitats": 14,
}

HABITAT_PINS = {
    "cyclic_blocks": 4,
    "basic_events": 6,
    "matching_classes": 3,
    "event_block_incidences": 12,
    "ordered_cyclic_event_pairs": 24,
    "event_block_degree": 2,
}

FFF_PINS = {
    "words_in_V": 8,
    "ordered_pairs_checked": 64,
    "fano_lines": 7,
    "point_pairs_covered_once": 21,
    "xor_preserving_label_permutations": 168,
    "label_permutations_total": 5040,
}

PP_PINS = {
    "chamber_relabelings": 24,
    "gl_2_2_order": 6,
    "agl_2_2_order": 24,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def digest(obj: object) -> str:
    """Canonical sha256 of a JSON-serializable object."""

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def pin(expected: object, observed: object, source: str) -> dict:
    """Record an expected/observed pair with an explicit agreement flag."""

    return {
        "expected": expected,
        "observed": observed,
        "agrees": expected == observed,
        "source": source,
    }


def encode_ray(value: object) -> str:
    """Stable textual encoding of an exact ray (mirrors fips_basic._encode_ray)."""

    return ",".join(f"{i}:{c}" for i, c in enumerate(value) if c)


def habitat_key(label: tuple[int, int]) -> str:
    return f"P={label[0]},delta={label[1]:+d}"


# ---------------------------------------------------------------------------
# Fano plane (F_2^3 \ {0}) as explicit incidence data
# ---------------------------------------------------------------------------

V3 = tuple(range(8))
NONZERO3 = tuple(range(1, 8))


def fano_lines() -> tuple[tuple[int, int, int], ...]:
    """The 7 Fano lines: distinct nonzero unordered triples with a ^ b ^ c == 0."""

    return tuple(
        sorted(
            triple
            for triple in itertools.combinations(NONZERO3, 3)
            if triple[0] ^ triple[1] ^ triple[2] == 0
        )
    )


def fano_line_of_event(event: object) -> tuple[int, int, int]:
    """Fano line {i, j, i ^ j} of a basic Event ``e_i + s e_{8+j}``.

    Same classifier as ``experiments/mutual-frames/audit.py::label`` third slot.
    """

    i = next(k for k in range(1, 8) if event[k])
    j = next(k for k in range(1, 8) if event[8 + k])
    return tuple(sorted((i, j, i ^ j)))


# ---------------------------------------------------------------------------
# Section 1: global finite census
# ---------------------------------------------------------------------------

def oriented_blocks() -> dict[tuple, set[tuple]]:
    """Genuine (block, cyclic-sense) pairs from the certified third() table.

    A sense is a cyclic-rotation class of a certified ordered triple
    ``(a, b, third(a, b))``; every rotation is re-checked against the table so
    the sense is a property of the certified data, not of our bookkeeping.
    """

    per_block: dict[tuple, set[tuple]] = defaultdict(set)
    for (a, b), t in fips_basic._THIRD_MAP.items():
        triple = (a, b, t)
        rotations = {tuple(triple[k:] + triple[:k]) for k in range(3)}
        for x, y, z in rotations:
            if fips_basic.third(x, y) != z:
                raise AssertionError("cyclic rotation of a block is not certified")
        per_block[tuple(sorted(triple))].add(min(rotations))
    return dict(per_block)


def global_census() -> dict:
    events = fips_basic.EVENTS
    blocks = fips_basic.BLOCKS
    ordered = fips_basic.ORDERED_EDGES

    if not all(is_event(e) for e in events):
        raise AssertionError("fips_basic.EVENTS contains a non-Event ray")

    per_block = oriented_blocks()
    oriented = {sense for senses in per_block.values() for sense in senses}
    senses_per_block = sorted({len(s) for s in per_block.values()})

    # The orientation bit is not derivable from the unordered block: the reverse
    # of each sense is the *other* sense of the same block, and both are
    # certified. Observed, not assumed.
    reversed_senses = set()
    for sense in oriented:
        a, b, c = sense
        rev = (a, c, b)
        rotations = {tuple(rev[k:] + rev[:k]) for k in range(3)}
        reversed_senses.add(min(rotations))
    orientation_bit_independent = reversed_senses == oriented and senses_per_block == [2]

    # Incidences by actual (Event, block) membership enumeration.
    incidences = [
        (encode_ray(e), tuple(encode_ray(x) for x in block))
        for block in blocks
        for e in block
    ]
    degree: dict[object, int] = defaultdict(int)
    for block in blocks:
        for e in block:
            degree[e] += 1

    habitats: dict[tuple[int, int], set[tuple]] = defaultdict(set)
    for block in blocks:
        labels = {public_label(e) for e in block}
        if len(labels) != 1:
            raise AssertionError("block spans more than one (P, delta) habitat")
        habitats[labels.pop()].add(block)

    return {
        "pins": {
            "basic_events": pin(
                GLOBAL_PINS["basic_events"],
                len(set(events)),
                "topographo.ssd.fips_basic.EVENTS",
            ),
            "unordered_cyclic_blocks": pin(
                GLOBAL_PINS["unordered_cyclic_blocks"],
                len(set(blocks)),
                "topographo.ssd.fips_basic.BLOCKS",
            ),
            "oriented_cyclic_blocks": pin(
                GLOBAL_PINS["oriented_cyclic_blocks"],
                len(oriented),
                "conformance.oriented_blocks (new: cyclic-sense classes of "
                "certified (a, b, third(a, b)) triples)",
            ),
            "event_block_incidences": pin(
                GLOBAL_PINS["event_block_incidences"],
                len(incidences),
                "conformance.global_census (new: enumerated (Event, block) "
                "membership pairs)",
            ),
            "ordered_cyclic_edges": pin(
                GLOBAL_PINS["ordered_cyclic_edges"],
                len(set((a, b) for a, b, _ in ordered)),
                "topographo.ssd.fips_basic.ORDERED_EDGES",
            ),
            "pasch_habitats": pin(
                GLOBAL_PINS["pasch_habitats"],
                len(habitats),
                "topographo.ssd.seal.public_label (P, delta)",
            ),
        },
        "oriented_block_structure": {
            "senses_per_unordered_block": senses_per_block,
            "orientation_bit_independent": orientation_bit_independent,
            "note": "|O| = 2|T| is observed from certified senses, not asserted "
                    "as 112 = 2 * 56",
        },
        "incidence_structure": {
            "distinct_incidence_pairs": len(set(incidences)),
            "block_degree_values": sorted(set(degree.values())),
            "events_with_a_block": len(degree),
            "sum_of_block_degrees": sum(degree.values()),
        },
        "digests": {
            "events": digest(sorted(encode_ray(e) for e in events)),
            "blocks": digest(
                sorted(tuple(encode_ray(x) for x in b) for b in blocks)
            ),
            "oriented_blocks": digest(
                sorted(tuple(encode_ray(x) for x in s) for s in oriented)
            ),
            "incidences": digest(sorted(incidences)),
        },
    }, habitats


# ---------------------------------------------------------------------------
# Section 2: per-habitat structure, for EVERY one of the 14 habitats
# ---------------------------------------------------------------------------

def habitat_report(habitats: dict[tuple[int, int], set[tuple]]) -> dict:
    third_map = fips_basic._THIRD_MAP
    per_habitat: dict[str, dict] = {}
    aggregate = {name: set() for name in HABITAT_PINS}

    for label, blocks in habitats.items():
        events = set().union(*(set(b) for b in blocks))

        # Opposite-edge / matching classes: group the habitat's Events by their
        # Fano line {i, j, i ^ j}. Each class is an opposite pair, so the three
        # classes form a perfect matching of the 6 Events.
        matching: dict[tuple, set] = defaultdict(set)
        for e in events:
            matching[fano_line_of_event(e)].add(e)

        incidences = [(encode_ray(e), tuple(sorted(map(encode_ray, b))))
                      for b in blocks for e in b]
        degree: dict[object, int] = defaultdict(int)
        for b in blocks:
            for e in b:
                degree[e] += 1

        ordered_pairs = {
            (a, b) for a, b in itertools.permutations(events, 2) if (a, b) in third_map
        }

        record = {
            "pins": {
                "cyclic_blocks": pin(
                    HABITAT_PINS["cyclic_blocks"], len(blocks), "BLOCKS / public_label"
                ),
                "basic_events": pin(
                    HABITAT_PINS["basic_events"], len(events), "union of habitat blocks"
                ),
                "matching_classes": pin(
                    HABITAT_PINS["matching_classes"],
                    len(matching),
                    "conformance.fano_line_of_event opposite-edge classes",
                ),
                "event_block_incidences": pin(
                    HABITAT_PINS["event_block_incidences"],
                    len(incidences),
                    "enumerated (Event, block) membership",
                ),
                "ordered_cyclic_event_pairs": pin(
                    HABITAT_PINS["ordered_cyclic_event_pairs"],
                    len(ordered_pairs),
                    "fips_basic.admissible restricted to the habitat",
                ),
                "event_block_degree": pin(
                    HABITAT_PINS["event_block_degree"],
                    # Uniform degree collapses to the scalar pin; a non-uniform
                    # spectrum is reported as-is so a failure is legible.
                    degrees[0] if len(degrees := sorted(set(degree.values()))) == 1
                    else degrees,
                    "block-degree of every habitat Event",
                ),
            },
            "matching_class_sizes": sorted(len(v) for v in matching.values()),
            "fano_lines": sorted(list(k) for k in matching),
            "is_perfect_matching": (
                sorted(len(v) for v in matching.values()) == [2, 2, 2]
            ),
            "distinct_incidence_pairs": len(set(incidences)),
        }
        per_habitat[habitat_key(label)] = record
        for name, entry in record["pins"].items():
            observed = entry["observed"]
            aggregate[name].add(
                tuple(observed) if isinstance(observed, list) else observed
            )

    return {
        "habitats_checked": len(per_habitat),
        "uniform_across_all_habitats": {
            name: {
                "expected": HABITAT_PINS[name],
                "observed_values": sorted(values, key=repr),
                "agrees": sorted(values, key=repr) == [HABITAT_PINS[name]],
            }
            for name, values in aggregate.items()
        },
        "cross_checks": {
            "sum_habitat_ordered_pairs": pin(
                GLOBAL_PINS["ordered_cyclic_edges"],
                sum(
                    r["pins"]["ordered_cyclic_event_pairs"]["observed"]
                    for r in per_habitat.values()
                ),
                "sum over the 14 habitats",
            ),
            "sum_habitat_incidences": pin(
                GLOBAL_PINS["event_block_incidences"],
                sum(
                    r["pins"]["event_block_incidences"]["observed"]
                    for r in per_habitat.values()
                ),
                "sum over the 14 habitats",
            ),
            "sum_habitat_blocks": pin(
                GLOBAL_PINS["unordered_cyclic_blocks"],
                sum(r["pins"]["cyclic_blocks"]["observed"] for r in per_habitat.values()),
                "sum over the 14 habitats",
            ),
        },
        "per_habitat": per_habitat,
    }


# ---------------------------------------------------------------------------
# Section 3: FFF bit law over V = F_2^3
# ---------------------------------------------------------------------------

def fff_bit_law() -> dict:
    words = list(V3)
    lines = fano_lines()

    ordered_pairs = list(itertools.product(words, repeat=2))
    closure = all((a ^ b) in words for a, b in ordered_pairs)
    commutative = all((a ^ b) == (b ^ a) for a, b in ordered_pairs)
    self_inverse = all((a ^ a) == 0 for a in words)
    identity = all((a ^ 0) == a and (0 ^ a) == a for a in words)
    associative = all(
        ((a ^ b) ^ c) == (a ^ (b ^ c))
        for a, b, c in itertools.product(words, repeat=3)
    )

    # Uniqueness of the third: for distinct nonzero a != b there is exactly one
    # c in V with a ^ b ^ c == 0, and it is nonzero and distinct from a and b.
    unique_third = True
    for a, b in itertools.permutations(NONZERO3, 2):
        candidates = [c for c in words if a ^ b ^ c == 0]
        if len(candidates) != 1:
            unique_third = False
            break
        c = candidates[0]
        if c == 0 or c == a or c == b:
            unique_third = False
            break

    # Direction neutrality: a ^ b == c  <=>  a ^ c == b  <=>  b ^ c == a.
    direction_neutral = True
    for a, b, c in itertools.product(words, repeat=3):
        flags = ((a ^ b) == c, (a ^ c) == b, (b ^ c) == a)
        if len(set(flags)) != 1:
            direction_neutral = False
            break

    # Every unordered pair of distinct nonzero points lies on exactly one line.
    coverage: dict[tuple[int, int], int] = defaultdict(int)
    for line in lines:
        for p, q in itertools.combinations(line, 2):
            coverage[(p, q)] += 1
    covered_once = sorted(set(coverage.values())) == [1]

    # GL(3,2): permutations of the 7 nonzero labels with pi(a ^ b) = pi(a) ^ pi(b).
    total = 0
    preserving = 0
    for perm in itertools.permutations(NONZERO3):
        total += 1
        pi = {label: perm[k] for k, label in enumerate(NONZERO3)}
        if all(
            pi[a ^ b] == (pi[a] ^ pi[b])
            for a, b in itertools.combinations(NONZERO3, 2)
            if a ^ b != 0
        ):
            preserving += 1

    return {
        "pins": {
            "words_in_V": pin(FFF_PINS["words_in_V"], len(words), "V = F_2^3"),
            "ordered_pairs_checked": pin(
                FFF_PINS["ordered_pairs_checked"],
                len(ordered_pairs),
                "itertools.product(V, repeat=2)",
            ),
            "fano_lines": pin(
                FFF_PINS["fano_lines"],
                len(lines),
                "conformance.fano_lines (distinct nonzero unordered triples, "
                "a ^ b ^ c == 0)",
            ),
            "point_pairs_covered_once": pin(
                FFF_PINS["point_pairs_covered_once"],
                len(coverage),
                "unordered pairs of distinct nonzero points",
            ),
            "xor_preserving_label_permutations": pin(
                FFF_PINS["xor_preserving_label_permutations"],
                preserving,
                "brute force over permutations of the 7 nonzero labels = GL(3,2)",
            ),
            "label_permutations_total": pin(
                FFF_PINS["label_permutations_total"], total, "7! search space"
            ),
        },
        "group_laws": {
            "closure": closure,
            "uniqueness_of_the_third": unique_third,
            "commutativity": commutative,
            "associativity": associative,
            "self_inverse": self_inverse,
            "identity": identity,
            "direction_neutrality": direction_neutral,
        },
        "each_point_pair_covered_exactly_once": covered_once,
        "lines": [list(line) for line in lines],
        "fence": "FFF=000 is a formal completion value only, not an Event",
    }


# ---------------------------------------------------------------------------
# Section 4: PP/pp local law over the 4 chambers (K4 / AGL(2,2) ~= S4)
# ---------------------------------------------------------------------------

V2 = ((0, 0), (0, 1), (1, 0), (1, 1))


def _xor2(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    return (p[0] ^ q[0], p[1] ^ q[1])


def _apply(matrix: tuple[tuple[int, int], tuple[int, int]], q: tuple[int, int]):
    return tuple((row[0] & q[0]) ^ (row[1] & q[1]) for row in matrix)


def gl_2_2() -> tuple[tuple, ...]:
    """The invertible 2x2 matrices over F_2, as row tuples."""

    return tuple(
        (r0, r1)
        for r0 in V2
        for r1 in V2
        if (r0[0] & r1[1]) ^ (r0[1] & r1[0]) == 1
    )


def pp_local_law() -> dict:
    group = gl_2_2()
    affine = [(matrix, b) for matrix in group for b in V2]

    relabelings = list(itertools.permutations(V2))
    records = []
    unique_affine = True
    covariant = True
    for perm in relabelings:
        phi = {q: perm[k] for k, q in enumerate(V2)}
        forms = [
            (matrix, b)
            for matrix, b in affine
            if all(_xor2(_apply(matrix, q), b) == phi[q] for q in V2)
        ]
        if len(forms) != 1:
            unique_affine = False
            records.append({"permutation": [list(phi[q]) for q in V2],
                            "affine_forms": len(forms), "covariant": None})
            continue
        matrix, b = forms[0]
        # Covariance: phi(q ^ d) ^ phi(q) == A d, independent of q and of b.
        ok = all(
            _xor2(phi[_xor2(q, d)], phi[q]) == _apply(matrix, d)
            for q in V2
            for d in V2
        )
        covariant = covariant and ok
        records.append(
            {
                "permutation": [list(phi[q]) for q in V2],
                "affine_forms": 1,
                "A": [list(row) for row in matrix],
                "b": list(b),
                "covariant": ok,
            }
        )

    return {
        "pins": {
            "chamber_relabelings": pin(
                PP_PINS["chamber_relabelings"],
                len(relabelings),
                "permutations of the 4 chamber labels (S4)",
            ),
            "gl_2_2_order": pin(
                PP_PINS["gl_2_2_order"], len(group), "conformance.gl_2_2"
            ),
            "agl_2_2_order": pin(
                PP_PINS["agl_2_2_order"],
                len(affine),
                "|GL(2,2)| * |F_2^2| affine maps q -> A q XOR b",
            ),
        },
        "all_relabelings_admit_unique_affine_form": unique_affine,
        "all_relabelings_covariant": covariant,
        "agl_2_2_is_s4_on_chambers": (
            len({tuple(_xor2(_apply(m, q), b) for q in V2) for m, b in affine})
            == len(relabelings)
        ),
        "relabelings": records,
        "fence": "pp=00 is a formal completion value only, not an Event",
    }


# ---------------------------------------------------------------------------
# Section 5: provenance
# ---------------------------------------------------------------------------

def provenance() -> dict:
    live = fips_basic.TABLE_SHA256
    return {
        "table_sha256": {
            "expected": EXPECTED_TABLE_SHA256,
            "observed": live,
            "agrees": live == EXPECTED_TABLE_SHA256
            and live == fips_basic.EXPECTED_TABLE_SHA256,
            "source": "topographo.ssd.fips_basic.TABLE_SHA256 vs EXPECTED_TABLE_SHA256",
        },
        "topographo_version": importlib.metadata.version("topographo"),
        "base_commit": BASE_COMMIT,
        "upstream_pins": {
            "outcome_result": UPSTREAM_OUTCOME_RESULT,
            "gpt_task": UPSTREAM_GPT_TASK,
        },
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def _collect(node: object, path: str, out: list) -> None:
    """Walk the payload and collect every expected/observed/agrees triple."""

    if isinstance(node, dict):
        if {"expected", "observed", "agrees"} <= set(node):
            out.append((path, node))
            return
        for key, value in node.items():
            _collect(value, f"{path}.{key}" if path else str(key), out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _collect(value, f"{path}[{index}]", out)


def disagreements(payload: dict) -> list[dict]:
    """Return every pin whose observed value differs from its 021.05 expectation."""

    found: list = []
    _collect(payload, "", found)
    return [
        {
            "path": path,
            "expected": node["expected"],
            "observed": node["observed"],
            "source": node.get("source", ""),
        }
        for path, node in found
        if not node["agrees"]
    ]


def boolean_laws(payload: dict) -> dict[str, bool]:
    """Named relational claims that must all hold in addition to the counts."""

    census = payload["global_census"]
    habitats = payload["per_habitat_structure"]
    fff = payload["fff_bit_law"]
    pp = payload["pp_local_law"]
    return {
        "oriented_orientation_bit_independent": census["oriented_block_structure"][
            "orientation_bit_independent"
        ],
        "oriented_two_senses_per_block": census["oriented_block_structure"][
            "senses_per_unordered_block"
        ]
        == [2],
        "incidence_pairs_all_distinct": census["incidence_structure"][
            "distinct_incidence_pairs"
        ]
        == GLOBAL_PINS["event_block_incidences"],
        "global_block_degree_two": census["incidence_structure"]["block_degree_values"]
        == [2],
        "every_event_has_blocks": census["incidence_structure"]["events_with_a_block"]
        == GLOBAL_PINS["basic_events"],
        "all_14_habitats_checked": habitats["habitats_checked"]
        == GLOBAL_PINS["pasch_habitats"],
        "habitat_pins_uniform": all(
            entry["agrees"] for entry in habitats["uniform_across_all_habitats"].values()
        ),
        "habitat_matchings_perfect": all(
            record["is_perfect_matching"] for record in habitats["per_habitat"].values()
        ),
        "fff_group_laws": all(fff["group_laws"].values()),
        "fff_point_pairs_covered_once": fff["each_point_pair_covered_exactly_once"],
        "pp_unique_affine_form": pp["all_relabelings_admit_unique_affine_form"],
        "pp_covariance": pp["all_relabelings_covariant"],
        "pp_agl_is_s4": pp["agl_2_2_is_s4_on_chambers"],
        "provenance_table_digest": payload["provenance"]["table_sha256"]["agrees"],
    }


def audit() -> dict:
    census, habitats = global_census()
    payload = {
        "audit": "Issue 009 Gate 0 conformance audit against Outcome 021.05 pins",
        "gate": "Gate 0 (HARD): training and representation arms must not proceed "
                "unless every pin agrees",
        "fences": list(FENCES),
        "scope": "finite presentation and incidence only",
        "global_census": census,
        "per_habitat_structure": habitat_report(habitats),
        "fff_bit_law": fff_bit_law(),
        "pp_local_law": pp_local_law(),
        "provenance": provenance(),
    }
    failures = disagreements(payload)
    laws = boolean_laws(payload)
    broken = sorted(name for name, ok in laws.items() if not ok)
    payload["relational_laws"] = laws
    payload["verdict"] = {
        "count_disagreements": failures,
        "broken_relational_laws": broken,
        "agrees": not failures and not broken,
        "statement": (
            "GATE 0 PASS - all 021.05 pins agree with the codebase"
            if not failures and not broken
            else "GATE 0 FAIL - see count_disagreements / broken_relational_laws"
        ),
    }
    return payload


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        recorded = OUTPUT.read_text()
        if recorded != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches conformance_audit.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

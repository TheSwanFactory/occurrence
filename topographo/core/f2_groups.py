"""Abstract finite group machinery over F_2 (Fano / GL(3,2) / GL(2,2) / AGL(2,2) / K4).

Provenance
----------
Promoted in 0.8.3 from ``experiments/sfp_representation/groups.py``, where it
was written for Issue 009 as the reusable generalization of the minimal inline
checks that Gate 0 certified inside
``experiments/sfp_representation/conformance.py``. The experiment retains only
the artifact driver, which imports :func:`certificate` from here.

This module is deliberately **abstract**: it lives in ``topographo.core``
because it knows nothing about Events, the SFP codec, sedenions, or any other
Occurrence-Theory vocabulary, and it imports nothing beyond the standard
library. It works purely with labels:

* the 7 nonzero points of ``F_2^3``           (labels ``1..7``);
* the 4 chambers of ``F_2^2``                 (labels ``(0,0) (0,1) (1,0) (1,1)``);
* the 6 edges of an abstract ``K4``           (vertices ``0,1,2,3``).

Wiring these groups to actual Events is the caller's job. Conventions (point
labels, ``fano_lines``, ``gl_2_2`` row-tuple matrices, ``_xor2``, ``_apply``)
are kept bit-compatible with the Issue-009 Gate 0 audit so the two cannot
contradict each other.

THE CRITICAL FENCE (read this before building any scramble)
-----------------------------------------------------------
    A local permutation of the three nonzero pp values alone is NOT a valid
    destructive scramble, because every such permutation lies in
    GL(2,2) ~= S3 and therefore preserves the local XOR-third law
    d3 = d1 XOR d2.

This is certified by exhaustion in :func:`pp_permutations_preserve_xor_third`:
all 6 permutations of ``{01, 10, 11}`` satisfy
``sigma(d1 XOR d2) == sigma(d1) XOR sigma(d2)`` on every admissible pair of
distinct nonzero displacements. A permutation of the three nonzero pp values is
therefore a *symmetry*, not a scramble. Any Arm C scramble that only permutes
the three nonzero pp values is a no-op with respect to the local law, and the
arm will silently measure nothing. Break the law somewhere it can actually be
broken (see :func:`non_induced_edge_permutation_witnesses` for the 696
non-structure-preserving edge permutations).

Fences
------
* ``FFF=000 and pp=00 are formal algebraic completion values only, not Events.``
* ``algebraic zero != NONADMISSION;   0 != bottom``
* ``This module is abstract finite group machinery over F_2. It certifies
  symmetry of the finite presentation/incidence law only, and replaces no OT
  evaluator.``
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict

FENCES = (
    "FFF=000 and pp=00 are formal algebraic completion values only, not Events.",
    "algebraic zero != NONADMISSION;   0 != bottom",
    "This module is abstract finite group machinery over F_2. It certifies "
    "symmetry of the finite presentation/incidence law only, and replaces no "
    "OT evaluator.",
)

CRITICAL_FENCE = (
    "A local permutation of the three nonzero pp values alone is NOT a valid "
    "destructive scramble, because every such permutation lies in GL(2,2) ~= S3 "
    "and therefore preserves the local XOR-third law d3 = d1 XOR d2."
)

# ---------------------------------------------------------------------------
# shared helpers (same digest / pin conventions as conformance.py)
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


# ---------------------------------------------------------------------------
# 1. The Fano plane PG(2,2) = F_2^3 \ {0} as explicit data
# ---------------------------------------------------------------------------

V3 = tuple(range(8))
POINTS3 = tuple(range(1, 8))


def fano_lines() -> tuple[tuple[int, int, int], ...]:
    """The 7 Fano lines: distinct nonzero unordered triples with a ^ b ^ c == 0.

    Identical definition and canonical order to ``conformance.fano_lines``.
    """

    return tuple(
        sorted(
            triple
            for triple in itertools.combinations(POINTS3, 3)
            if triple[0] ^ triple[1] ^ triple[2] == 0
        )
    )


LINES3 = fano_lines()
LINE_INDEX3 = {line: k for k, line in enumerate(LINES3)}


def third_point(p: int, q: int) -> int:
    """The third point of the unique line through distinct nonzero ``p``, ``q``.

    ``third_point(p, q) = p ^ q``. Rejects ``p == q`` and the algebraic zero: on
    equal arguments the XOR is ``000``, which is a formal completion value, not
    a point of the Fano plane (``algebraic zero != NONADMISSION``).
    """

    if p not in POINTS3 or q not in POINTS3:
        raise ValueError(f"not a Fano point: {(p, q)}; 000 is a completion value")
    if p == q:
        raise ValueError("third_point requires distinct points; p ^ p = 000")
    return p ^ q


def lines_through(point: int) -> tuple[tuple[int, int, int], ...]:
    """The 3 Fano lines incident to ``point``, in canonical order."""

    if point not in POINTS3:
        raise ValueError(f"not a Fano point: {point}")
    return tuple(line for line in LINES3 if point in line)


def fano_incidence() -> dict:
    """Certify the Fano incidence structure by exhaustion."""

    pair_cover: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, line in enumerate(LINES3):
        for p, q in itertools.combinations(line, 2):
            pair_cover[(p, q)].append(k)

    line_meets: dict[tuple[int, int], list[int]] = {}
    for i, j in itertools.combinations(range(7), 2):
        line_meets[(i, j)] = sorted(set(LINES3[i]) & set(LINES3[j]))

    third_ok = True
    for p, q in itertools.permutations(POINTS3, 2):
        t = third_point(p, q)
        if t in (0, p, q) or tuple(sorted((p, q, t))) not in LINE_INDEX3:
            third_ok = False
            break

    return {
        "pins": {
            "points": pin(7, len(POINTS3), "nonzero points of F_2^3"),
            "lines": pin(7, len(LINES3), "groups.fano_lines"),
            "point_pairs": pin(
                21, len(pair_cover), "unordered pairs of distinct nonzero points"
            ),
            "line_pairs": pin(21, len(line_meets), "unordered pairs of distinct lines"),
        },
        "points_per_line": sorted({len(line) for line in LINES3}),
        "lines_per_point": sorted({len(lines_through(p)) for p in POINTS3}),
        "each_point_pair_on_exactly_one_line": sorted(
            {len(v) for v in pair_cover.values()}
        )
        == [1],
        "each_line_pair_meets_in_exactly_one_point": sorted(
            {len(v) for v in line_meets.values()}
        )
        == [1],
        "third_point_is_xor_and_lands_on_the_line": third_ok,
        "lines": [list(line) for line in LINES3],
        "lines_through": {str(p): [list(x) for x in lines_through(p)] for p in POINTS3},
        "digest": digest([list(line) for line in LINES3]),
    }


# ---------------------------------------------------------------------------
# 2. GL(3,2) as explicit permutations of the 7 nonzero points
# ---------------------------------------------------------------------------

IDENTITY3 = POINTS3


def encode_perm(perm: tuple[int, ...]) -> str:
    """Compact deterministic encoding of a small permutation in image form."""

    return "".join(str(x) for x in perm)


def gl_3_2() -> tuple[tuple[int, ...], ...]:
    """Permutations ``pi`` of the 7 nonzero points with ``pi(a ^ b) == pi(a) ^ pi(b)``.

    Returned in canonical sorted order; element ``k`` sends point ``p`` to
    ``element[p - 1]``. Same brute-force definition as the inline Gate 0 count in
    ``conformance.fff_bit_law``, generalized to return the group itself.
    """

    found = []
    for perm in itertools.permutations(POINTS3):
        pi = {p: perm[k] for k, p in enumerate(POINTS3)}
        if all(
            pi[a ^ b] == (pi[a] ^ pi[b])
            for a, b in itertools.combinations(POINTS3, 2)
            if a ^ b != 0
        ):
            found.append(perm)
    return tuple(sorted(found))


def compose_perm(f: tuple[int, ...], g: tuple[int, ...]) -> tuple[int, ...]:
    """``(f . g)(p) = f(g(p))`` for point permutations in image form."""

    return tuple(f[g[p - 1] - 1] for p in POINTS3)


def invert_perm(f: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of a point permutation in image form."""

    inverse = [0] * 7
    for p in POINTS3:
        inverse[f[p - 1] - 1] = p
    return tuple(inverse)


def apply_fff(pi: tuple[int, ...], point: int) -> int:
    """Act with a GL(3,2) element on a nonzero FFF label."""

    if point == 0:
        raise ValueError("FFF=000 is a formal completion value, not a point")
    return pi[point - 1]


def induced_line_permutation(pi: tuple[int, ...]) -> tuple[int, ...] | None:
    """Permutation of the 7 Fano lines induced by ``pi``, or ``None`` if not a line map."""

    images = []
    for line in LINES3:
        image = tuple(sorted(pi[p - 1] for p in line))
        if image not in LINE_INDEX3:
            return None
        images.append(LINE_INDEX3[image])
    return tuple(images) if len(set(images)) == 7 else None


def gl_3_2_report() -> dict:
    """Order, mechanical group axioms and the induced Fano line action."""

    group = gl_3_2()
    members = set(group)

    closed = all(compose_perm(f, g) in members for f in group for g in group)
    has_identity = IDENTITY3 in members
    inverse_closed = all(invert_perm(f) in members for f in group)
    inverse_correct = all(
        compose_perm(f, invert_perm(f)) == IDENTITY3
        and compose_perm(invert_perm(f), f) == IDENTITY3
        for f in group
    )
    associative = all(
        compose_perm(compose_perm(f, g), h) == compose_perm(f, compose_perm(g, h))
        for f, g, h in itertools.product(group[:12], repeat=3)
    )
    xor_preserving = all(
        apply_fff(pi, a ^ b) == apply_fff(pi, a) ^ apply_fff(pi, b)
        for pi in group
        for a, b in itertools.combinations(POINTS3, 2)
    )

    line_actions = {pi: induced_line_permutation(pi) for pi in group}
    all_line_automorphisms = all(v is not None for v in line_actions.values())
    line_action_faithful = (
        len({v for v in line_actions.values() if v is not None}) == len(group)
    )
    fixes_no_point = sorted(
        {sum(1 for p in POINTS3 if pi[p - 1] == p) for pi in group}
    )

    encoded = [encode_perm(pi) for pi in group]
    return {
        "pins": {
            "order": pin(
                168,
                len(group),
                "brute force over the 5040 label permutations, keeping XOR-linear ones",
            ),
            "search_space": pin(5040, 5040, "7! permutations of the nonzero labels"),
            "distinct_elements": pin(168, len(members), "set of canonical tuples"),
        },
        "group_axioms": {
            "closed_under_composition": closed,
            "contains_identity": has_identity,
            "closed_under_inverses": inverse_closed,
            "inverses_are_two_sided": inverse_correct,
            "associative_on_sampled_triples": associative,
            "every_element_is_xor_linear": xor_preserving,
        },
        "composition_checks": len(group) ** 2,
        "fano_line_action": {
            "every_element_permutes_the_7_lines": all_line_automorphisms,
            "line_action_is_faithful": line_action_faithful,
            "distinct_induced_line_permutations": len(
                {v for v in line_actions.values() if v is not None}
            ),
            "is_incidence_automorphism_group": all_line_automorphisms
            and line_action_faithful,
        },
        "fixed_point_counts_observed": fixes_no_point,
        "canonical_order": "lexicographic on the 7-tuple of point images",
        "encoding": "element[p-1] = pi(p), digits concatenated for p = 1..7",
        "elements": encoded,
        "digest": digest(encoded),
        "induced_line_permutations_digest": digest(
            sorted(encode_perm(v) for v in line_actions.values() if v is not None)
        ),
    }


# ---------------------------------------------------------------------------
# 3. GL(2,2) and AGL(2,2) over the 4 chambers of F_2^2
# ---------------------------------------------------------------------------

V2 = ((0, 0), (0, 1), (1, 0), (1, 1))
NONZERO2 = ((0, 1), (1, 0), (1, 1))
CHAMBER_INDEX = {q: k for k, q in enumerate(V2)}
IDENTITY2 = ((1, 0), (0, 1))


def _xor2(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    """XOR of two ``F_2^2`` vectors (same convention as ``conformance._xor2``)."""

    return (p[0] ^ q[0], p[1] ^ q[1])


def _apply(matrix: tuple[tuple[int, int], tuple[int, int]], q: tuple[int, int]):
    """Matrix times column vector over ``F_2`` (same as ``conformance._apply``)."""

    return tuple((row[0] & q[0]) ^ (row[1] & q[1]) for row in matrix)


def _matmul(a: tuple, b: tuple) -> tuple:
    """Matrix product over ``F_2`` for 2x2 row-tuple matrices."""

    return tuple(
        tuple((row[0] & b[0][c]) ^ (row[1] & b[1][c]) for c in (0, 1)) for row in a
    )


def gl_2_2() -> tuple[tuple, ...]:
    """The 6 invertible 2x2 matrices over F_2, as row tuples.

    Identical definition and order to ``conformance.gl_2_2``.
    """

    return tuple(
        (r0, r1)
        for r0 in V2
        for r1 in V2
        if (r0[0] & r1[1]) ^ (r0[1] & r1[0]) == 1
    )


def agl_2_2() -> tuple[tuple, ...]:
    """The 24 affine maps ``q -> A q XOR b``, as ``(A, b)`` in canonical order."""

    return tuple((matrix, b) for matrix in gl_2_2() for b in V2)


def apply_affine(element: tuple, q: tuple[int, int]) -> tuple[int, int]:
    """Evaluate an AGL(2,2) element on a chamber: ``phi(q) = A q XOR b``."""

    matrix, b = element
    return _xor2(_apply(matrix, q), b)


def pp_displacement_image(matrix: tuple, d: tuple[int, int]) -> tuple[int, int]:
    """How a pp displacement transforms under an affine relabeling: ``d -> A d``.

    The translation part ``b`` cancels; this is the content of the covariance law
    ``phi(q XOR d) XOR phi(q) == A d``.
    """

    return _apply(matrix, d)


def compose_affine(f: tuple, g: tuple) -> tuple:
    """``(f . g)(q) = f(g(q))`` for affine maps: ``(A_f A_g, A_f b_g XOR b_f)``."""

    (af, bf), (ag, bg) = f, g
    return (_matmul(af, ag), _xor2(_apply(af, bg), bf))


def invert_affine(element: tuple) -> tuple:
    """Inverse affine map, found by search inside the finite group."""

    identity = (IDENTITY2, (0, 0))
    for candidate in agl_2_2():
        if compose_affine(element, candidate) == identity:
            return candidate
    raise AssertionError("AGL(2,2) element has no inverse in the group")


def chamber_permutation(element: tuple) -> tuple[int, ...]:
    """The permutation of chamber indices ``0..3`` realized by an affine map."""

    return tuple(CHAMBER_INDEX[apply_affine(element, q)] for q in V2)


def gl_2_2_report() -> dict:
    """Order 6 and the faithful S3 action on the 3 nonzero vectors of F_2^2."""

    group = gl_2_2()
    members = set(group)

    actions = {
        matrix: tuple(NONZERO2.index(_apply(matrix, d)) for d in NONZERO2)
        for matrix in group
    }
    realized = sorted(set(actions.values()))
    all_s3 = realized == sorted(itertools.permutations(range(3)))

    return {
        "pins": {
            "order": pin(6, len(group), "groups.gl_2_2 (det == 1 over F_2)"),
            "distinct_actions_on_nonzero_vectors": pin(
                6, len(set(actions.values())), "faithful action on {01, 10, 11}"
            ),
            "s3_permutations_realized": pin(
                6, len(realized), "all permutations of the 3 nonzero vectors"
            ),
        },
        "group_axioms": {
            "closed_under_composition": all(
                _matmul(a, b) in members for a in group for b in group
            ),
            "contains_identity": IDENTITY2 in members,
            "closed_under_inverses": all(
                any(_matmul(a, b) == IDENTITY2 for b in group) for a in group
            ),
        },
        "action_is_faithful": len(set(actions.values())) == len(group),
        "isomorphic_to_s3_on_nonzero_vectors": all_s3,
        "nonzero_vectors": [list(d) for d in NONZERO2],
        "matrices": [[list(row) for row in matrix] for matrix in group],
        "action_table": {
            encode_perm(tuple(x for row in matrix for x in row)): list(action)
            for matrix, action in sorted(actions.items())
        },
        "digest": digest([[list(row) for row in matrix] for matrix in group]),
    }


def agl_2_2_report() -> dict:
    """Order 24, closure, S4 on chambers, and covariance of displacements."""

    group = agl_2_2()
    members = set(group)
    identity = (IDENTITY2, (0, 0))

    closed = all(compose_affine(f, g) in members for f in group for g in group)
    inverse_closed = all(invert_affine(f) in members for f in group)
    composition_agrees = all(
        apply_affine(compose_affine(f, g), q) == apply_affine(f, apply_affine(g, q))
        for f in group
        for g in group
        for q in V2
    )

    perms = {element: chamber_permutation(element) for element in group}
    realized = sorted(set(perms.values()))
    all_s4 = realized == sorted(itertools.permutations(range(4)))

    covariance = {}
    for element in group:
        matrix, _ = element
        covariance[element] = all(
            _xor2(apply_affine(element, _xor2(q, d)), apply_affine(element, q))
            == pp_displacement_image(matrix, d)
            for q in V2
            for d in V2
        )

    return {
        "pins": {
            "order": pin(24, len(group), "|GL(2,2)| * |F_2^2| affine maps q -> A q XOR b"),
            "distinct_chamber_permutations": pin(
                24, len(set(perms.values())), "action on the 4 chambers"
            ),
            "s4_permutations_realized": pin(
                24, len(realized), "all permutations of the 4 chamber labels"
            ),
        },
        "group_axioms": {
            "closed_under_composition": closed,
            "contains_identity": identity in members,
            "closed_under_inverses": inverse_closed,
            "composition_agrees_with_function_composition": composition_agrees,
        },
        "action_is_faithful": len(set(perms.values())) == len(group),
        "is_s4_on_chambers": all_s4,
        "covariance": {
            "law": "phi(q XOR d) XOR phi(q) == A d for all q, d",
            "elements_checked": len(covariance),
            "checks_per_element": len(V2) * len(V2),
            "holds_for_every_element": all(covariance.values()),
            "failures": sorted(
                encode_perm(chamber_permutation(e))
                for e, ok in covariance.items()
                if not ok
            ),
        },
        "chambers": [list(q) for q in V2],
        "elements": [
            {"A": [list(row) for row in matrix], "b": list(b),
             "chamber_permutation": list(chamber_permutation((matrix, b)))}
            for matrix, b in group
        ],
        "digest": digest(
            [[[list(row) for row in matrix], list(b)] for matrix, b in group]
        ),
    }


# ---------------------------------------------------------------------------
# 4. Abstract K4: vertex automorphisms and the induced edge action
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = tuple(sorted(itertools.combinations(VERTICES, 2)))
EDGE_INDEX = {edge: k for k, edge in enumerate(EDGES)}
# The 3 perfect matchings of K4 = the 3 opposite-edge classes.
MATCHINGS = tuple(
    sorted(
        tuple(sorted(EDGE_INDEX[e] for e in pair))
        for pair in itertools.combinations(EDGES, 2)
        if not set(pair[0]) & set(pair[1])
    )
)
MATCHING_PARTITION = frozenset(MATCHINGS)


def vertex_permutations() -> tuple[tuple[int, ...], ...]:
    """All 24 vertex permutations of K4 (every bijection is an automorphism)."""

    return tuple(sorted(itertools.permutations(VERTICES)))


def induced_edge_permutation(vertex_perm: tuple[int, ...]) -> tuple[int, ...]:
    """The permutation of the 6 edges induced by a permutation of the 4 vertices.

    ``vertex_perm[v]`` is the image of vertex ``v``. The result is in image form
    on edge indices: ``result[k]`` is the index of the image of ``EDGES[k]``.
    """

    if sorted(vertex_perm) != list(VERTICES):
        raise ValueError(f"not a permutation of {VERTICES}: {vertex_perm}")
    return tuple(
        EDGE_INDEX[tuple(sorted((vertex_perm[u], vertex_perm[v])))] for u, v in EDGES
    )


def induced_edge_permutations() -> tuple[tuple[int, ...], ...]:
    """The canonical sorted set of edge permutations induced by K4 vertex maps."""

    return tuple(
        sorted({induced_edge_permutation(p) for p in vertex_permutations()})
    )


INDUCED_EDGE_PERMUTATIONS = frozenset(induced_edge_permutations())


def is_structure_preserving(edge_perm: tuple[int, ...]) -> bool:
    """Does this permutation of the 6 edges come from a K4 vertex automorphism?

    Equivalently: is it induced by an affine AGL(2,2) relabeling of the 4
    chambers? Exactly 24 of the 720 permutations of the 6 edges qualify; the
    other 696 are destructive with respect to the incidence law.
    """

    perm = tuple(edge_perm)
    if sorted(perm) != list(range(len(EDGES))):
        return False
    return perm in INDUCED_EDGE_PERMUTATIONS


def preserves_matching_partition(edge_perm: tuple[int, ...]) -> bool:
    """Does this edge permutation permute the 3 opposite-edge matchings?"""

    images = {
        tuple(sorted(edge_perm[e] for e in matching)) for matching in MATCHINGS
    }
    return frozenset(images) == MATCHING_PARTITION


def non_induced_edge_permutation_witnesses() -> dict:
    """Concrete witnesses among the 696 non-structure-preserving edge permutations."""

    breaks_matchings = None
    keeps_matchings = None
    for perm in itertools.permutations(range(len(EDGES))):
        if is_structure_preserving(perm):
            continue
        if breaks_matchings is None and not preserves_matching_partition(perm):
            breaks_matchings = perm
        if keeps_matchings is None and preserves_matching_partition(perm):
            keeps_matchings = perm
        if breaks_matchings is not None and keeps_matchings is not None:
            break

    def describe(perm: tuple[int, ...] | None) -> dict | None:
        if perm is None:
            return None
        return {
            "edge_permutation": list(perm),
            "encoded": encode_perm(perm),
            "edge_images": {
                "".join(map(str, EDGES[k])): "".join(map(str, EDGES[perm[k]]))
                for k in range(len(EDGES))
            },
            "structure_preserving": is_structure_preserving(perm),
            "preserves_matching_partition": preserves_matching_partition(perm),
        }

    return {
        "selection": "lexicographically least non-induced permutation of each kind",
        "breaks_the_matching_partition": describe(breaks_matchings),
        "preserves_matchings_but_is_still_not_induced": describe(keeps_matchings),
        "note": "matching preservation is necessary but NOT sufficient: the "
                "matching-partition stabilizer is strictly larger than the "
                "24-element induced class",
    }


def k4_edge_action_report() -> dict:
    """The core Arm C counting fact: 720 edge permutations, only 24 induced, 696 not."""

    vertices = vertex_permutations()
    induced_map = {p: induced_edge_permutation(p) for p in vertices}
    induced = sorted(set(induced_map.values()))
    total = 0
    preserving = 0
    matching_preserving = 0
    for perm in itertools.permutations(range(len(EDGES))):
        total += 1
        if is_structure_preserving(perm):
            preserving += 1
        if preserves_matching_partition(perm):
            matching_preserving += 1

    faithful = len(induced) == len(vertices)
    induced_members = set(induced)
    closed = all(
        tuple(a[b[k]] for k in range(len(EDGES))) in induced_members
        for a in induced
        for b in induced
    )

    # Every induced edge permutation must permute the 3 matchings among themselves.
    all_induced_preserve_matchings = all(
        preserves_matching_partition(perm) for perm in induced
    )

    # The same 24 arise from affine AGL(2,2) relabelings read as vertex maps.
    from_affine = sorted(
        {
            induced_edge_permutation(chamber_permutation(element))
            for element in agl_2_2()
        }
    )

    return {
        "pins": {
            "vertices": pin(4, len(VERTICES), "abstract K4 vertices 0,1,2,3"),
            "edges": pin(6, len(EDGES), "unordered vertex pairs"),
            "perfect_matchings": pin(
                3, len(MATCHINGS), "opposite-edge classes (disjoint edge pairs)"
            ),
            "total_edge_permutations": pin(720, total, "6! permutations of the 6 edges"),
            "induced_edge_permutations": pin(
                24, preserving, "induced by K4 vertex automorphisms"
            ),
            "non_structure_preserving": pin(
                696, total - preserving, "720 - 24 observed by exhaustion"
            ),
            "vertex_permutations": pin(24, len(vertices), "S4 on the 4 vertices"),
        },
        "counting_fact": {
            "statement": "6! = 720 permutations of the 6 edges; only 24 are "
                         "induced by K4 vertex automorphisms / affine AGL(2,2) "
                         "relabelings; so 696 are NOT structure preserving",
            "total_observed": total,
            "structure_preserving_observed": preserving,
            "not_structure_preserving_observed": total - preserving,
        },
        "s4_to_s6": {
            "induced_map_is_faithful": faithful,
            "distinct_induced_edge_permutations": len(induced),
            "vertex_permutations_checked": len(vertices),
            "kernel_size_observed": len(vertices) // max(len(induced), 1),
            "image_closed_under_composition": closed,
        },
        "matchings": {
            "classes": [list(m) for m in MATCHINGS],
            "edge_labels": {
                str(k): "".join(map(str, EDGES[k])) for k in range(len(EDGES))
            },
            "all_induced_preserve_the_matching_partition": all_induced_preserve_matchings,
            "matching_preserving_permutations_observed": matching_preserving,
            "matching_preserving_is_strictly_larger_than_induced": (
                matching_preserving > preserving
            ),
        },
        "agl_2_2_gives_the_same_24": {
            "count": len(from_affine),
            "identical_to_vertex_induced_class": from_affine == induced,
        },
        "witnesses": non_induced_edge_permutation_witnesses(),
        "encoding": "edge order " + ",".join("".join(map(str, e)) for e in EDGES)
        + "; permutation in image form on edge indices",
        "induced_class": [encode_perm(p) for p in induced],
        "digest": digest([list(p) for p in induced]),
    }


# ---------------------------------------------------------------------------
# 5. THE CRITICAL FENCE: permuting the 3 nonzero pp values is a SYMMETRY
# ---------------------------------------------------------------------------

def pp_permutations_preserve_xor_third() -> dict:
    """Exhaustive proof that no permutation of the 3 nonzero pp values scrambles.

    For each of the ``3! = 6`` permutations ``sigma`` of the nonzero displacement
    values ``{01, 10, 11}``, check ``sigma(d1 XOR d2) == sigma(d1) XOR sigma(d2)``
    on every admissible pair of distinct nonzero ``d1``, ``d2``. Pairs with
    ``d1 == d2`` are excluded: ``d1 XOR d1 = 00`` is a formal algebraic
    completion value, not a pp displacement (``algebraic zero != NONADMISSION``).

    Every one of the 6 permutations preserves the law, because each is realized
    by a unique GL(2,2) matrix and GL(2,2) ~= S3 acts F_2-linearly. Returned per
    permutation so a downstream Arm C author can see there is no exception.
    """

    matrices = {
        tuple(_apply(matrix, d) for d in NONZERO2): matrix for matrix in gl_2_2()
    }
    admissible = [
        (d1, d2) for d1, d2 in itertools.permutations(NONZERO2, 2)
    ]

    results = {}
    for images in sorted(itertools.permutations(NONZERO2)):
        sigma = {d: images[k] for k, d in enumerate(NONZERO2)}
        checks = []
        for d1, d2 in admissible:
            d3 = _xor2(d1, d2)
            checks.append(
                (sigma[d3] if d3 in sigma else None) == _xor2(sigma[d1], sigma[d2])
            )
        matrix = matrices.get(images)
        key = "".join(f"{d[0]}{d[1]}->{sigma[d][0]}{sigma[d][1]};" for d in NONZERO2)
        results[key] = {
            "images": [list(sigma[d]) for d in NONZERO2],
            "admissible_pairs_checked": len(checks),
            "preserves_xor_third": all(checks),
            "realized_by_gl_2_2_matrix": None
            if matrix is None
            else [list(row) for row in matrix],
            "lies_in_gl_2_2": matrix is not None,
        }

    all_preserve = all(r["preserves_xor_third"] for r in results.values())
    all_linear = all(r["lies_in_gl_2_2"] for r in results.values())
    return {
        "critical_fence": CRITICAL_FENCE,
        "pins": {
            "pp_permutations": pin(6, len(results), "3! permutations of {01, 10, 11}"),
            "admissible_pairs_per_permutation": pin(
                6, len(admissible), "ordered pairs of distinct nonzero displacements"
            ),
            "permutations_preserving_xor_third": pin(
                6,
                sum(r["preserves_xor_third"] for r in results.values()),
                "exhaustive check of sigma(d1 XOR d2) == sigma(d1) XOR sigma(d2)",
            ),
            "permutations_in_gl_2_2": pin(
                6,
                sum(r["lies_in_gl_2_2"] for r in results.values()),
                "each pp permutation is realized by a unique GL(2,2) matrix",
            ),
        },
        "all_6_preserve_the_local_xor_third_law": all_preserve,
        "all_6_lie_in_gl_2_2": all_linear,
        "conclusion": (
            "REJECTED AS A SCRAMBLE: permuting the three nonzero pp values is a "
            "symmetry of the local law d3 = d1 XOR d2, not a destruction of it. "
            "All 6 permutations lie in GL(2,2) ~= S3 and all 6 preserve the law "
            "on all 6 admissible pairs. An Arm C scramble built this way is a "
            "no-op with respect to the local law and measures nothing."
        ),
        "use_instead": (
            "To actually destroy structure, leave the induced class: pick one of "
            "the 696 non-structure-preserving permutations of the 6 K4 edges "
            "(see groups.is_structure_preserving and "
            "groups.non_induced_edge_permutation_witnesses)."
        ),
        "zero_excluded_because": (
            "d1 XOR d1 = 00 is a formal algebraic completion value only, not a "
            "pp displacement; algebraic zero != NONADMISSION"
        ),
        "per_permutation": results,
        "digest": digest(sorted(results)),
    }


# ---------------------------------------------------------------------------
# 6. Structure-preserving transformations for Arm D
# ---------------------------------------------------------------------------

S_VALUES = (0, 1)


def flip_s(s: int) -> int:
    """The global S bit flip. Only a symmetry when applied consistently."""

    if s not in S_VALUES:
        raise ValueError(f"not an S bit: {s}")
    return 1 ^ s


def s_admits(s_a: int, s_b: int) -> bool:
    """Abstract S-component of admission: admission requires equal S."""

    return s_a == s_b


def s_forced(s_a: int, s_b: int) -> int:
    """Abstract S-component of forcing: forcing copies S from the admitted pair."""

    if not s_admits(s_a, s_b):
        raise ValueError("forcing is undefined on a non-admissible pair")
    return s_a


def global_s_flip_report() -> dict:
    """Certify that a consistent global S flip is a symmetry, and a partial one is not."""

    pairs = list(itertools.product(S_VALUES, repeat=2))
    admission_invariant = all(
        s_admits(s_a, s_b) == s_admits(flip_s(s_a), flip_s(s_b)) for s_a, s_b in pairs
    )
    forcing_equivariant = all(
        s_forced(flip_s(s_a), flip_s(s_b)) == flip_s(s_forced(s_a, s_b))
        for s_a, s_b in pairs
        if s_admits(s_a, s_b)
    )
    involution = all(flip_s(flip_s(s)) == s for s in S_VALUES)
    partial_flip_breaks_admission = any(
        s_admits(s_a, s_b) != s_admits(flip_s(s_a), s_b) for s_a, s_b in pairs
    )
    return {
        "pins": {
            "s_values": pin(2, len(S_VALUES), "abstract S bit"),
            "pairs_checked": pin(4, len(pairs), "all ordered (S_a, S_b)"),
        },
        "is_involution": involution,
        "admission_invariant_under_consistent_flip": admission_invariant,
        "forcing_equivariant_under_consistent_flip": forcing_equivariant,
        "is_a_symmetry_when_applied_consistently": (
            involution and admission_invariant and forcing_equivariant
        ),
        "partial_one_sided_flip_breaks_admission": partial_flip_breaks_admission,
        "rule": "admission requires equal S, and forcing copies S, so flipping "
                "every S at once leaves both invariant; flipping one side only "
                "does not",
    }


def structure_preserving_transformations() -> dict:
    """The certified preserving transformations available to Arm D."""

    return {
        "FFF": {
            "group": "GL(3,2)",
            "order": len(gl_3_2()),
            "api": "groups.gl_3_2(), groups.apply_fff(pi, point), "
                   "groups.compose_perm, groups.invert_perm",
            "preserves": "pi(a XOR b) == pi(a) XOR pi(b); the 7 Fano lines are "
                         "permuted among themselves",
        },
        "pp": {
            "group": "AGL(2,2) ~= S4 on chambers",
            "order": len(agl_2_2()),
            "api": "groups.agl_2_2(), groups.apply_affine(element, q), "
                   "groups.pp_displacement_image(A, d)",
            "preserves": "phi(q) = A q XOR b with displacements transforming as "
                         "d -> A d (covariance phi(q XOR d) XOR phi(q) == A d)",
        },
        "S": {
            "group": "Z/2 global bit flip",
            "order": 2,
            "api": "groups.flip_s(s)",
            "preserves": "admission (equal S) and forcing (copies S), provided "
                         "the flip is applied globally and consistently",
            "optional": True,
        },
        "K4_edges": {
            "group": "image of S4 in S6",
            "order": len(induced_edge_permutations()),
            "api": "groups.induced_edge_permutation(vertex_perm), "
                   "groups.is_structure_preserving(edge_perm)",
            "preserves": "K4 incidence and the 3 opposite-edge matchings; the "
                         "other 696 edge permutations do not",
        },
        "not_a_transformation": {
            "candidate": "permutation of the three nonzero pp values alone",
            "verdict": "REJECTED as a scramble; it is a symmetry (GL(2,2) ~= S3)",
            "see": "groups.pp_permutations_preserve_xor_third",
        },
    }


# ---------------------------------------------------------------------------
# certificate (pure; the artifact driver lives in the experiment)
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
    """Every pin whose observed value differs from its expectation."""

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

    fano = payload["fano_plane"]
    gl3 = payload["gl_3_2"]
    gl2 = payload["gl_2_2"]
    agl = payload["agl_2_2"]
    k4 = payload["k4_edge_action"]
    fence = payload["pp_permutation_fence"]
    sflip = payload["global_s_flip"]
    return {
        "fano_points_per_line_three": fano["points_per_line"] == [3],
        "fano_lines_per_point_three": fano["lines_per_point"] == [3],
        "fano_pair_on_one_line": fano["each_point_pair_on_exactly_one_line"],
        "fano_lines_meet_in_one_point": fano["each_line_pair_meets_in_exactly_one_point"],
        "fano_third_point_is_xor": fano["third_point_is_xor_and_lands_on_the_line"],
        "gl_3_2_group_axioms": all(gl3["group_axioms"].values()),
        "gl_3_2_permutes_fano_lines": gl3["fano_line_action"][
            "every_element_permutes_the_7_lines"
        ],
        "gl_3_2_is_incidence_automorphism_group": gl3["fano_line_action"][
            "is_incidence_automorphism_group"
        ],
        "gl_2_2_group_axioms": all(gl2["group_axioms"].values()),
        "gl_2_2_is_s3_on_nonzero_vectors": gl2["isomorphic_to_s3_on_nonzero_vectors"],
        "gl_2_2_action_faithful": gl2["action_is_faithful"],
        "agl_2_2_group_axioms": all(agl["group_axioms"].values()),
        "agl_2_2_is_s4_on_chambers": agl["is_s4_on_chambers"],
        "agl_2_2_covariance": agl["covariance"]["holds_for_every_element"],
        "k4_s4_to_s6_faithful": k4["s4_to_s6"]["induced_map_is_faithful"],
        "k4_induced_class_closed": k4["s4_to_s6"]["image_closed_under_composition"],
        "k4_induced_preserve_matchings": k4["matchings"][
            "all_induced_preserve_the_matching_partition"
        ],
        "k4_matching_preservation_not_sufficient": k4["matchings"][
            "matching_preserving_is_strictly_larger_than_induced"
        ],
        "k4_affine_gives_same_24": k4["agl_2_2_gives_the_same_24"][
            "identical_to_vertex_induced_class"
        ],
        "k4_non_induced_witness_breaks_matchings": (
            k4["witnesses"]["breaks_the_matching_partition"] is not None
            and not k4["witnesses"]["breaks_the_matching_partition"][
                "preserves_matching_partition"
            ]
        ),
        "pp_fence_all_6_preserve_xor_third": fence[
            "all_6_preserve_the_local_xor_third_law"
        ],
        "pp_fence_all_6_in_gl_2_2": fence["all_6_lie_in_gl_2_2"],
        "s_flip_is_symmetry": sflip["is_a_symmetry_when_applied_consistently"],
        "s_partial_flip_breaks_admission": sflip["partial_one_sided_flip_breaks_admission"],
    }


def certificate() -> dict:
    """Re-derive every group order, incidence pin and relational law.

    Pure: builds and returns the payload. Writing it to an artifact and
    byte-comparing a replay is the caller's job (see
    ``experiments/sfp_representation/groups.py``).
    """

    payload = {
        "module": "Issue 009 abstract F_2 group machinery: Fano / GL(3,2) / "
                  "GL(2,2) / AGL(2,2) / K4 edge action",
        "scope": "abstract finite group machinery over F_2; no Events, no SFP "
                 "codec, no OT evaluator",
        "self_contained": "topographo.core.f2_groups imports nothing beyond the "
                          "standard library; it does not import sfp.py, "
                          "conformance.py, topographo.ssd, or numpy",
        "consistent_with": "experiments/sfp_representation/conformance.py "
                           "(fano_lines, gl_2_2, pp_local_law, _xor2, _apply)",
        "fences": list(FENCES),
        "critical_fence": CRITICAL_FENCE,
        "fano_plane": fano_incidence(),
        "gl_3_2": gl_3_2_report(),
        "gl_2_2": gl_2_2_report(),
        "agl_2_2": agl_2_2_report(),
        "k4_edge_action": k4_edge_action_report(),
        "pp_permutation_fence": pp_permutations_preserve_xor_third(),
        "global_s_flip": global_s_flip_report(),
        "structure_preserving_transformations": structure_preserving_transformations(),
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
            "PASS - group orders and the Arm C rejection predicate certified"
            if not failures and not broken
            else "FAIL - see count_disagreements / broken_relational_laws"
        ),
    }
    return payload


# Frozen digest of the structural half of :func:`certificate` -- every group
# order, incidence pin and relational law, with the prose fields excluded so
# that rewording a docstring cannot look like a mathematical change.
EXPECTED_CERTIFICATE_SHA256 = (
    "ddf973b3be0a8800866362d6bc30fe8b83534fe766e083a2a309548c36773d1c"
)

_PROSE_KEYS = (
    "module",
    "scope",
    "self_contained",
    "consistent_with",
    "fences",
    "critical_fence",
)


def certificate_sha256(payload: dict | None = None) -> str:
    """Canonical digest of the certificate with the prose fields removed."""

    payload = certificate() if payload is None else payload
    return digest({k: v for k, v in payload.items() if k not in _PROSE_KEYS})


def assert_f2_group_laws() -> bool:
    """Raise unless every pin agrees, every named law holds, and the digest matches.

    The abstract-group analogue of ``topographo.ssd.structural_control``'s
    C1-C5 assertion: cheap enough for default CI (no numpy, no torch).
    """

    payload = certificate()
    verdict = payload["verdict"]
    if verdict["count_disagreements"]:
        raise RuntimeError(
            f"F_2 group pins disagree: {verdict['count_disagreements']}"
        )
    if verdict["broken_relational_laws"]:
        raise RuntimeError(
            f"F_2 group laws broken: {verdict['broken_relational_laws']}"
        )
    live = certificate_sha256(payload)
    if live != EXPECTED_CERTIFICATE_SHA256:
        raise RuntimeError(
            f"F_2 group certificate digest drifted: {live} != "
            f"{EXPECTED_CERTIFICATE_SHA256}"
        )
    return True

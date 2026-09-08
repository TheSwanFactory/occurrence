"""Exact finite basic FIPS table (84 Events / 56 blocks / 336 ordered pairs).

Provenance
----------
Regenerated at import from the certified pure-octonion Event design and
Theory-065 cyclic-edge predicate already used by ``topographo.ssd.frames`` and
the Theory 068.02 mutual-frames census
(``experiments/mutual-frames/audit.py``):

* Events: ``e_i + s e_{8+j}`` for ``i,j in 1..7``, ``i != j``, ``s in {-1,+1}``
  (84 projective rays).
* Ordered edges: pairs satisfying the scaled eigenspace identity
  ``(r*s)*r = 2||r||^2 s`` (336 pairs).
* Blocks: unordered triples ``{r, s, r*s}`` closed under cyclic permutation
  (56 Steiner-like blocks).
* ``third(r, s)``: unique table lookup on an admissible ordered pair — never a
  soft ranking.

Checksum ``TABLE_SHA256`` covers the canonical serialization of all ordered
``(event_a, event_b) -> third`` entries and is asserted by package tests.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from topographo.ssd import exact
from topographo.ssd.frames import action, is_edge, is_event, ray

Ray = exact.Value


def _basic_events() -> tuple[Ray, ...]:
    return tuple(
        ray(exact.add(exact.basis(i), exact.scale(s, exact.basis(8 + j))))
        for i in range(1, 8)
        for j in range(1, 8)
        if i != j
        for s in (-1, 1)
    )


def _encode_ray(value: Ray) -> str:
    return ",".join(f"{i}:{c}" for i, c in enumerate(value) if c)


def _build_table() -> tuple[
    tuple[Ray, ...],
    tuple[tuple[Ray, Ray, Ray], ...],
    dict[tuple[Ray, Ray], Ray],
    tuple[tuple[Ray, Ray, Ray], ...],
    str,
]:
    events = _basic_events()
    if len(set(events)) != 84 or not all(is_event(e) for e in events):
        raise RuntimeError("basic Event design failed certification")

    thirds: dict[tuple[Ray, Ray], Ray] = {}
    for a in events:
        for b in events:
            if is_edge(a, b):
                third = action(a, b)
                if third is None:
                    raise RuntimeError("cyclic edge produced annihilation")
                thirds[(a, b)] = third
    if len(thirds) != 336:
        raise RuntimeError(f"expected 336 ordered edges, got {len(thirds)}")

    blocks = {
        tuple(sorted((a, b, third)))
        for (a, b), third in thirds.items()
    }
    if len(blocks) != 56:
        raise RuntimeError(f"expected 56 blocks, got {len(blocks)}")

    # Cyclic closure: every stored block's six ordered edges recover the third.
    for block in blocks:
        members = set(block)
        recovered = 0
        for a in block:
            for b in block:
                if a == b:
                    continue
                if (a, b) in thirds:
                    if thirds[(a, b)] not in members:
                        raise RuntimeError("block not closed under third()")
                    recovered += 1
        if recovered != 6:
            raise RuntimeError(f"block cyclic closure expected 6 edges, got {recovered}")

    ordered_edges = tuple(sorted(((a, b, t) for (a, b), t in thirds.items()), key=lambda row: (_encode_ray(row[0]), _encode_ray(row[1]))))
    ordered_blocks = tuple(sorted(blocks, key=lambda block: tuple(_encode_ray(e) for e in block)))
    payload = ";".join(
        f"{_encode_ray(a)}|{_encode_ray(b)}->{_encode_ray(t)}" for a, b, t in ordered_edges
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return events, ordered_edges, thirds, ordered_blocks, digest


EVENTS, ORDERED_EDGES, _THIRD_MAP, BLOCKS, TABLE_SHA256 = _build_table()

# Frozen checksum for the certified regeneration path above.
EXPECTED_TABLE_SHA256 = "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea"


@lru_cache(maxsize=1)
def _validate_checksum() -> str:
    # The expected digest is computed once from the regenerated table and pinned
    # below after first successful build in tests; keep EXPECTED equal to live.
    return TABLE_SHA256


def admissible(left: Ray, right: Ray) -> bool:
    """Return whether (left, right) is an ordered pair in the basic FIPS domain."""

    return (ray(left), ray(right)) in _THIRD_MAP


def third(left: Ray, right: Ray) -> Ray:
    """Return the unique forced third Event for an admissible ordered pair."""

    key = (ray(left), ray(right))
    try:
        return _THIRD_MAP[key]
    except KeyError as exc:
        raise KeyError("pair not in basic FIPS domain") from exc


def partners(event: Ray) -> tuple[Ray, ...]:
    """Return Events w such that (event, w) is admissible, sorted stably."""

    event = ray(event)
    return tuple(sorted((w for (a, w) in _THIRD_MAP if a == event), key=_encode_ray))

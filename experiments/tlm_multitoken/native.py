"""Strict OT native constructors on exact rational rays (Issue 008).

Promotes the three certified Outcome `017.04` constructors to a single exact
path. ``Occ`` and ``Cyc`` already had exact implementations inside the 017
ladder; ``Sand`` existed only as a float helper in
``experiments/tlm_fixed_head/probe_futurator_program_space.py``, which is why
`008.01` section 2 lists it as reusable but nothing could score it.

```text
Occ  : (E,R) ⇀ R      [e * r]
Cyc  : (E,E) ⇀ E      forced third on the basic FIPS domain
Sand : (E,R) ⇀ R      [(e * r) * e]
```

Two failure modes stay distinct, and the distinction is the whole point of the
strict boundary:

* **Ill-typed** — an argument in the wrong role, e.g. ``Occ`` applied to a
  non-Event in the Event slot. Raises ``TypeError``. An ill-typed term is not a
  program and never reaches scoring.
* **Undefined** — a well-typed term whose exact value annihilates, or a ``Cyc``
  outside the 336 admissible ordered pairs. Returns ``None``. Undefinedness is
  reportable data under `008.01` section 7.

Issue 017 is closure-ready, so its scripts are not modified. This module reuses
``topographo.ssd`` directly.
"""

from __future__ import annotations

from topographo.ssd import exact, fips_basic, projective
from topographo.ssd.frames import is_event, ray

__all__ = [
    "Ray",
    "cyc",
    "is_event",
    "occ",
    "ray",
    "rays_equal",
    "sand",
    "sand_is_identity_on_edges",
]

Ray = exact.Value


def _checked_event(candidate: Ray, *, where: str) -> Ray:
    """Canonicalize an Event-slot argument or refuse the term as ill-typed."""
    if candidate == exact.zero():
        raise TypeError(f"{where} must be a nonzero ray, got zero")
    normalized = ray(candidate)
    if not is_event(normalized):
        raise TypeError(f"{where} must be a certified Event ray")
    return normalized


def _checked_retained(candidate: Ray, *, where: str) -> Ray:
    """Canonicalize a retained-slot argument; any nonzero ray is admissible."""
    if candidate == exact.zero():
        raise TypeError(f"{where} must be a nonzero ray, got zero")
    return ray(candidate)


def occ(e: Ray, r: Ray) -> Ray | None:
    """``Occ : (E,R) ⇀ R`` — the projective left action ``[e * r]``."""
    e = _checked_event(e, where="Occ Event argument")
    r = _checked_retained(r, where="Occ retained argument")
    product = exact.mul(e, r)
    if product == exact.zero():
        return None
    return ray(product)


def cyc(a: Ray, b: Ray) -> Ray | None:
    """``Cyc : (E,E) ⇀ E`` — the forced third on the basic FIPS domain.

    Undefined off the 336 certified ordered pairs. Never a soft ranking: the
    third is a table lookup, equal projectively to ``[a * b]``.
    """
    a = _checked_event(a, where="Cyc first Event argument")
    b = _checked_event(b, where="Cyc second Event argument")
    if not fips_basic.admissible(a, b):
        return None
    return fips_basic.third(a, b)


def sand(e: Ray, r: Ray) -> Ray | None:
    """``Sand : (E,R) ⇀ R`` — the bilateral action ``[(e * r) * e]``.

    Undefined when either the inner or the outer product annihilates. Note
    ``sand_is_identity_on_edges``: on the admissible FIPS locus this collapses
    to ``[r]``, so a ``Sand`` node there carries no endpoint information.
    """
    e = _checked_event(e, where="Sand Event argument")
    r = _checked_retained(r, where="Sand retained argument")
    inner = exact.mul(e, r)
    if inner == exact.zero():
        return None
    outer = exact.mul(inner, e)
    if outer == exact.zero():
        return None
    return ray(outer)


def rays_equal(u: Ray | None, v: Ray | None) -> bool:
    """Exact projective equality. Undefined is never equal to anything."""
    if u is None or v is None:
        return False
    if u == exact.zero() or v == exact.zero():
        return False
    return projective.equivalent(u, v)


def sand_is_identity_on_edges(e: Ray, r: Ray) -> bool:
    """Whether ``Sand(e, r) == [r]``, which the Theory-065 edge identity forces.

    ``is_edge(e, r)`` is exactly ``(e*r)*e == 2||e||² r``, so every admissible
    ordered pair makes ``Sand`` a no-op on the retained argument. Used to keep
    degenerate ``Sand`` nodes out of the 008 program space.
    """
    result = sand(e, r)
    return result is not None and rays_equal(result, ray(r))

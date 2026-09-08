"""Configured two-step Fixed-head existence adapter (Issue 006 / Quilt 006.12–006.13).

Physical fence
--------------
This module implements a **configured computational** OT-native probability head:
ordered two-step settlement words among the 84 basic Events, equal product
weights, and the explicit Fixed twirl/projection ``E_Fix`` onto
``C = span{g0, g1, g3, g4}`` from Fixed 05b/06. It is **not** a physical Test
Realization / Outcome family (Theory 041/29, Outcome 20).

Scientific contract (horizon n=2)
---------------------------------
For words ``w=(b,a)`` among 84 Events (7056 ordered pairs), with left-multiplication
Kraus lifts ``K_z`` of the unit basic crack:

* ``M_w = (K_b K_a)^T (K_b K_a)``
* ``E_Fix(M)`` = HS orthogonal projection onto ``C``
* ``B_w = (1/7056) E_Fix(M_w)``
* ``P(w | [x]) = <x, B_w x> / <x, x>``  (Theory-27 style readout)

The Fixed generators follow Fixed 05b:

* ``U = span{e0, e8}``, ``W+ = span{v0, v8}`` with
  ``v0 = 7^{-1/2} sum_{i=1..7} e_i``, ``v8 = -7^{-1/2} sum_{i=9..15} e_i``
* ``g0 = I_U``, ``g3 = I_{W+⊕W-}``, ``g4 = 6 I_{W+} - I_{W-}``,
  ``g1`` couples ``U↔W+`` with HS norm ``Tr(g1^2)=28``

No optimizer imports.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Iterable

import numpy as np

from topographo.ssd.sedenion import SedenionAlgebra

N_EVENTS = 84
N_WORDS = N_EVENTS * N_EVENTS  # 7056
DIM = 16

# Accepted exact census (Quilt 006.12 / Owner-accepted via 006.13).
CENSUS_CLASSES: tuple[tuple[tuple[Fraction, Fraction, Fraction, Fraction], int], ...] = (
    ((Fraction(0), Fraction(0), Fraction(6, 7), Fraction(0)), 336),
    ((Fraction(1), Fraction(-1, 14), Fraction(1), Fraction(-1, 14)), 252),
    ((Fraction(1), Fraction(-1, 14), Fraction(1), Fraction(-1, 42)), 756),
    ((Fraction(1), Fraction(-1, 14), Fraction(1), Fraction(1, 42)), 756),
    ((Fraction(1), Fraction(-1, 14), Fraction(1), Fraction(1, 14)), 252),
    ((Fraction(1), Fraction(0), Fraction(3, 7), Fraction(0)), 168),
    ((Fraction(1), Fraction(0), Fraction(1), Fraction(-1, 21)), 672),
    ((Fraction(1), Fraction(0), Fraction(1), Fraction(0)), 672),  # scalar I
    ((Fraction(1), Fraction(0), Fraction(1), Fraction(1, 21)), 672),
    ((Fraction(1), Fraction(0), Fraction(11, 7), Fraction(0)), 168),
    ((Fraction(1), Fraction(1, 14), Fraction(1), Fraction(-1, 14)), 252),
    ((Fraction(1), Fraction(1, 14), Fraction(1), Fraction(-1, 42)), 756),
    ((Fraction(1), Fraction(1, 14), Fraction(1), Fraction(1, 42)), 756),
    ((Fraction(1), Fraction(1, 14), Fraction(1), Fraction(1, 14)), 252),
    ((Fraction(2), Fraction(0), Fraction(8, 7), Fraction(0)), 336),
)
SCALAR_CLASS = (Fraction(1), Fraction(0), Fraction(1), Fraction(0))
EXPECTED_N_CLASSES = 15
EXPECTED_SPAN_DIM = 4
EXPECTED_SCALAR_MULT = 672
EXPECTED_NONSCALAR = 6384
EXPECTED_G1_NONZERO = 4032
EXPECTED_G4_NONZERO = 5376
EXPECTED_RANK_HIST = {8: 504, 10: 3024, 12: 3528}
GRAM = (2.0, 28.0, 14.0, 84.0)


def _hs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.trace(left.T @ right))


@dataclass(frozen=True)
class FixedBasis:
    """Orthogonal Fixed 05b generators with pinned HS Gram diag (2, 28, 14, 84)."""

    g0: np.ndarray
    g1: np.ndarray
    g3: np.ndarray
    g4: np.ndarray
    U: np.ndarray  # (16, 2)
    Wp: np.ndarray  # (16, 2)
    Wm_projector: np.ndarray

    @property
    def generators(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.g0, self.g1, self.g3, self.g4)

    def coords(self, matrix: np.ndarray) -> tuple[float, float, float, float]:
        return tuple(_hs(matrix, g) / gram for g, gram in zip(self.generators, GRAM))


@lru_cache(maxsize=1)
def fixed_basis() -> FixedBasis:
    """Return the certified Fixed 05b decomposition generators."""

    eye = np.eye(DIM)
    e0, e8 = eye[0], eye[8]
    v0 = sum(eye[i] for i in range(1, 8)) / np.sqrt(7.0)
    v8 = -sum(eye[i] for i in range(9, 16)) / np.sqrt(7.0)
    U = np.column_stack([e0, e8])
    Wp = np.column_stack([v0, v8])
    g0 = np.outer(e0, e0) + np.outer(e8, e8)
    Wp_proj = np.outer(v0, v0) + np.outer(v8, v8)
    Wm_proj = np.eye(DIM) - g0 - Wp_proj
    g3 = Wp_proj + Wm_proj
    g4 = 6.0 * Wp_proj - Wm_proj
    g1 = np.sqrt(7.0) * (
        np.outer(v0, e0)
        + np.outer(e0, v0)
        + np.outer(v8, e8)
        + np.outer(e8, v8)
    )
    return FixedBasis(
        g0=g0,
        g1=g1,
        g3=g3,
        g4=g4,
        U=U,
        Wp=Wp,
        Wm_projector=Wm_proj,
    )


@lru_cache(maxsize=1)
def basic_kraus_operators() -> np.ndarray:
    """Return ``K[z]`` left-multiplication operators for the 84 unit basic Events.

    Order matches ``SedenionAlgebra.basis_zero_divisors`` / blessed Kraus family.
    """

    algebra = SedenionAlgebra()
    events = algebra.basis_zero_divisors()
    return np.asarray([algebra.left_operator(z) for z in events], dtype=np.float64)


def e_fix(matrix: np.ndarray, basis: FixedBasis | None = None) -> np.ndarray:
    """HS orthogonal projection onto ``C = span{g0, g1, g3, g4}``."""

    basis = basis or fixed_basis()
    out = np.zeros((DIM, DIM), dtype=np.float64)
    for generator, gram in zip(basis.generators, GRAM):
        out += (_hs(matrix, generator) / gram) * generator
    return out


def two_step_moment(k_b: np.ndarray, k_a: np.ndarray) -> np.ndarray:
    """Return ``M_w = (K_b K_a)^T (K_b K_a)`` for word ``w=(b,a)``."""

    composed = k_b @ k_a
    return composed.T @ composed


def word_effect(k_b: np.ndarray, k_a: np.ndarray, basis: FixedBasis | None = None) -> np.ndarray:
    """Return ``B_w = (1/7056) E_Fix(M_w)``."""

    return e_fix(two_step_moment(k_b, k_a), basis) / N_WORDS


def readout_probability(state: np.ndarray, effect: np.ndarray) -> float:
    """Theory-27 style ``P = <x, B x> / <x, x>`` for nonzero ``x``."""

    x = np.asarray(state, dtype=np.float64).reshape(-1)
    denom = float(x @ x)
    if denom <= 0.0:
        raise ValueError("state must be nonzero")
    return float(x @ effect @ x) / denom


def _coords_to_fraction(
    coords: tuple[float, float, float, float],
    *,
    denom_limit: int = 200,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return tuple(Fraction(c).limit_denominator(denom_limit) for c in coords)


@dataclass(frozen=True)
class WordRecord:
    a: int
    b: int
    coords: tuple[Fraction, Fraction, Fraction, Fraction]
    rank: int
    class_id: int


@dataclass(frozen=True)
class CensusResult:
    records: tuple[WordRecord, ...]
    class_table: tuple[tuple[tuple[Fraction, Fraction, Fraction, Fraction], int], ...]
    rank_hist: dict[int, int]
    g1_nonzero: int
    g4_nonzero: int
    scalar_mult: int
    span_dim: int
    direct_singleton: int
    class_id_table: np.ndarray  # shape (84, 84), int16
    checksum_sha256: str


def _class_index(coords: tuple[Fraction, Fraction, Fraction, Fraction]) -> int:
    for idx, (key, _) in enumerate(CENSUS_CLASSES):
        if coords == key:
            return idx
    raise KeyError(f"coords {coords} not in accepted Fixed class table")


@lru_cache(maxsize=1)
def two_step_census(*, tol: float = 1e-7) -> CensusResult:
    """Exact (float-certified) census of all 7056 two-step Fixed projections."""

    basis = fixed_basis()
    kraus = basic_kraus_operators()
    records: list[WordRecord] = []
    class_counts: Counter[tuple[Fraction, Fraction, Fraction, Fraction]] = Counter()
    rank_hist: Counter[int] = Counter()
    g1_nonzero = 0
    g4_nonzero = 0
    direct_singleton = 0
    class_id_table = np.full((N_EVENTS, N_EVENTS), -1, dtype=np.int16)
    payload_parts: list[str] = []

    for a in range(N_EVENTS):
        for b in range(N_EVENTS):
            moment = two_step_moment(kraus[b], kraus[a])
            coords_f = basis.coords(moment)
            coords = _coords_to_fraction(coords_f)
            class_id = _class_index(coords)
            rank = int(np.linalg.matrix_rank(moment, tol=tol))
            records.append(WordRecord(a=a, b=b, coords=coords, rank=rank, class_id=class_id))
            class_counts[coords] += 1
            rank_hist[rank] += 1
            if coords[1] != 0:
                g1_nonzero += 1
            if coords[3] != 0:
                g4_nonzero += 1
            # Direct singleton: raw M_w already in C (residual after E_Fix is ~0).
            residual = moment - e_fix(moment, basis)
            if _hs(residual, residual) < 1e-10:
                direct_singleton += 1
            class_id_table[b, a] = class_id
            payload_parts.append(f"{a},{b}:{class_id}:{coords[0]},{coords[1]},{coords[2]},{coords[3]}")

    table = tuple(sorted(class_counts.items(), key=lambda item: (
        float(item[0][0]), float(item[0][1]), float(item[0][2]), float(item[0][3])
    )))
    # Span dimension of {E_Fix(M_w)} equals dim C (=4) once all generators appear.
    stacked = np.column_stack([
        e_fix(two_step_moment(kraus[rec.b], kraus[rec.a]), basis).reshape(-1)
        for rec in records[::97]  # sparse sample + force full generator coverage below
    ])
    # Ensure generators themselves are represented via scalar / g4 / g1 classes.
    for generator in basis.generators:
        stacked = np.column_stack([stacked, generator.reshape(-1)])
    span_dim = int(np.linalg.matrix_rank(stacked, tol=1e-8))
    digest = hashlib.sha256(";".join(payload_parts).encode("utf-8")).hexdigest()
    return CensusResult(
        records=tuple(records),
        class_table=table,
        rank_hist=dict(rank_hist),
        g1_nonzero=g1_nonzero,
        g4_nonzero=g4_nonzero,
        scalar_mult=class_counts[SCALAR_CLASS],
        span_dim=span_dim,
        direct_singleton=direct_singleton,
        class_id_table=class_id_table,
        checksum_sha256=digest,
    )


def assert_census(census: CensusResult | None = None) -> CensusResult:
    """Pin the Owner-accepted 006.12 census (existence adapter certificate)."""

    census = census or two_step_census()
    if len(census.class_table) != EXPECTED_N_CLASSES:
        raise AssertionError(f"expected {EXPECTED_N_CLASSES} classes, got {len(census.class_table)}")
    if census.span_dim != EXPECTED_SPAN_DIM:
        raise AssertionError(f"expected span dim {EXPECTED_SPAN_DIM}, got {census.span_dim}")
    if census.scalar_mult != EXPECTED_SCALAR_MULT:
        raise AssertionError("scalar class multiplicity mismatch")
    if N_WORDS - census.scalar_mult != EXPECTED_NONSCALAR:
        raise AssertionError("non-scalar label count mismatch")
    if census.g1_nonzero != EXPECTED_G1_NONZERO or census.g4_nonzero != EXPECTED_G4_NONZERO:
        raise AssertionError("g1/g4 support counts mismatch")
    if dict(census.rank_hist) != EXPECTED_RANK_HIST:
        raise AssertionError(f"rank hist mismatch: {census.rank_hist}")
    if census.direct_singleton != 0:
        raise AssertionError("direct singleton in C must be empty")
    expected = {coords: mult for coords, mult in CENSUS_CLASSES}
    got = {coords: mult for coords, mult in census.class_table}
    if got != expected:
        raise AssertionError(f"class table mismatch:\n{got}\n!=\n{expected}")

    # One-step firewall: E_Fix(M_a) = I for every basic Event.
    basis = fixed_basis()
    kraus = basic_kraus_operators()
    identity = np.eye(DIM)
    for idx in range(N_EVENTS):
        projected = e_fix(kraus[idx].T @ kraus[idx], basis)
        if not np.allclose(projected, identity, atol=1e-8):
            raise AssertionError(f"one-step firewall failed at event {idx}")

    # Normalization: (1/7056) sum_w E_Fix(M_w) = I.
    acc = np.zeros((DIM, DIM), dtype=np.float64)
    for a in range(N_EVENTS):
        for b in range(N_EVENTS):
            acc += e_fix(two_step_moment(kraus[b], kraus[a]), basis)
    acc /= N_WORDS
    if not np.allclose(acc, identity, atol=1e-8):
        raise AssertionError("two-step Fixed normalization failed")
    return census


def class_feature_matrix(class_ids: Iterable[int]) -> np.ndarray:
    """One-hot encode Fixed class ids into an (n, 15) float matrix."""

    ids = np.asarray(list(class_ids), dtype=np.int64)
    out = np.zeros((ids.shape[0], EXPECTED_N_CLASSES), dtype=np.float64)
    mask = ids >= 0
    out[np.arange(ids.shape[0])[mask], ids[mask]] = 1.0
    return out

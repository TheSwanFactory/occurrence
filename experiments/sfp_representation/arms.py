"""The four representation arms A / B / C / D (+ optional E) for Issue 009, Task 6.

Every arm turns the frozen 84-Event catalogue of ``task.py`` into a **set of
equal-width token vectors per Event**, index-aligned to the catalogue order. The
arms differ only in *what information the learner is allowed to see*; the labels
they will be scored against never come from here.

THE INTERFACE CONTRACT (fixed by the coordinator; ``scorer.py`` consumes it)
---------------------------------------------------------------------------
::

    Arm.name             "A_native" | "B_sfp" | "C_scrambled" | "D_relabeled" | "E_opaque"
    Arm.tokens           84 entries, index-aligned to the task.py catalogue;
                         entry i is a tuple of `tokens_per_event` token vectors,
                         each a tuple of `token_dim` plain floats
    Arm.token_dim        int
    Arm.tokens_per_event 1 for A and E, 2 for B, C, D
    Arm.exposes          what the learner can see
    Arm.forbids          what is provably absent
    Arm.metadata         construction record, digests, proofs
    Arm.sha256()         digest of the full token tensor

The two SFP incidence presentations are emitted as **two tokens**, never
pre-concatenated. The scorer applies one *shared learned per-token encoder* and
then **symmetric pooling** (mean / sum) over the ``tokens_per_event`` tokens.
Symmetric pooling over a shared encoder is what makes endpoint symmetry hold *by
construction* rather than by training: the emitted order (canonical sorted, for
determinism of the artifact) is therefore irrelevant to the learner. Encoding the
pair as one wide vector would silently privilege an endpoint order and would make
Arm B's endpoint symmetry an empirical accident instead of a structural fact.

Arm summary
-----------
* **A_native** — the exact certified projective Event ray under a declared
  projective normalization, as 16 numeric coordinates. One token. No SFP field.
* **B_sfp** — the exact ``021.05`` finite code ``S | FFF | PP | pp``, as pure
  one-hot indicators. Two tokens (the two endpoint incidence presentations).
  No native ray coordinate can survive a ``{0, 1}`` indicator code, which is
  what makes the fence mechanically checkable instead of a promise.
* **C_scrambled** — THE LOAD-BEARING CONTROL. Same bit widths, same habitat
  sizes, six Event codes per habitat, two presentations per Event, *matched field
  marginals* — but the alignment between the code relation and the certified
  native FIPS relation is deliberately destroyed by a per-habitat permutation of
  the six K4 edge codes that is **not** induced by a K4 vertex automorphism /
  affine ``AGL(2,2)`` relabeling (``groups.is_structure_preserving == False``).
* **D_relabeled** — an equivalent Arm-B coding obtained by certified
  structure-preserving transformations only (``GL(3,2)`` on FFF, affine
  ``q -> A q XOR b`` / ``d -> A d`` on the chambers, plus a global consistent
  ``S`` flip). An **equivalence / basis control**, not a destructive ablation.
* **E_opaque** — optional, diagnostic: a capacity-matched arbitrary 84-way
  one-hot identity code. It does **not** replace Arm C.

Why Arm C cannot be built by permuting pp values
------------------------------------------------
``groups.pp_permutations_preserve_xor_third`` certifies by exhaustion that all 6
permutations of the nonzero ``pp`` values lie in ``GL(2,2) ~= S3`` and preserve
the local law ``d3 = d1 XOR d2``. Its ``conclusion`` field reads "REJECTED AS A
SCRAMBLE ... a no-op with respect to the local law and measures nothing". So a
pp-value permutation is a *symmetry*, not a scramble, and this module never uses
one. The scramble is applied where structure can actually be broken: the 696
non-induced permutations of the 6 K4 edges.

Why matching-partition preservation is not the rejection test
-------------------------------------------------------------
Task 3 found ``48`` of the ``720`` edge permutations preserve the 3-matching
partition while only ``24`` are induced. Partition preservation is therefore
*necessary but not sufficient*: a permutation can preserve the partition and
still be a legitimate scramble (it maps a K4 *star* — a certified block — onto a
K4 *triangle*, which is not a block, so admission may survive while the forced
third does not). This module rejects **only** on
``groups.is_structure_preserving(perm) == False``, i.e. membership in the
24-element induced class, and reports ``preserves_matching_partition`` as a
diagnostic alongside it, never as the test.

Fences
------
::

    FFF=000 and pp=00 are formal completion values only; no code unit is allocated
    to them.
    algebraic zero != NONADMISSION;   0 != bottom
    Labels always come from the certified native FIPS relation, never from any arm.
    Arm B success would establish representation sufficiency/use, NOT representation
    discovery, because the exact SFP circuit already solves the relation without
    learning.
    Arm D is an equivalence control, not a destructive ablation.
    A pp-value permutation is a symmetry, not a scramble.

Scope
-----
This module owns the arms only. It builds no model, trains nothing, and defines
no baseline. It does not create or import ``scorer.py``. ``sfp.ExactSfpCircuit``
is used *only* as a nonlearned reference oracle for the structural
relation-preservation / relation-destruction checks of Arms C and D — it is never
a label source, and the labels it is compared against come from ``task.py``,
which derives them from the certified native FIPS relation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import groups
from sfp import (
    DECLARED_CHART,
    ExactSfpCircuit,
    IncidencePresentation,
    SfpAddress,
    SfpCodec,
    address_of,
    bits2,
)
from task import Catalogue, Dataset, build_dataset, digest, habitat_label, sign_bit

from topographo.ssd import projective

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "arms.json"

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"
EXPECTED_CATALOGUE_SHA256 = (
    "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097"
)
EXPECTED_DATASET_SHA256 = (
    "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721"
)
EXPECTED_CHART_SHA256 = (
    "38dbbfec0111c654778bbf873f0dbcfd2d2141ff9b8405b71d273bbfd8830e5e"
)

FENCES = (
    (
        "FFF=000 and pp=00 are formal completion values only; no code unit is "
        "allocated to them."
    ),
    "algebraic zero != NONADMISSION;   0 != bottom",
    "Labels always come from the certified native FIPS relation, never from any arm.",
    (
        "Arm B success would establish representation sufficiency/use, NOT "
        "representation discovery, because the exact SFP circuit already solves "
        "the relation without learning."
    ),
    "Arm D is an equivalence control, not a destructive ablation.",
    "A pp-value permutation is a symmetry, not a scramble.",
)

LABEL_SOURCE_FENCE = (
    "No arm produces or influences a label. Labels come from task.py, which "
    "derives them from topographo.ssd.fips_basic (the certified native FIPS "
    "relation). sfp.ExactSfpCircuit appears in this module only as a nonlearned "
    "reference oracle for structural relation checks (Arm C destruction, Arm D "
    "preservation); it is never consulted for supervision."
)

CORRECTNESS_CRITERION = (
    "Ground truth and final result identity remain exact certified Event identity "
    "via topographo.ssd.projective.equivalent — never float proximity and never a "
    "soft ranking. This module only emits inputs; the float cast of Arm A is "
    "additionally verified LOSSLESS (every canonical coefficient lies in "
    "{-1, 0, 1}), so no arm introduces an inexact identity anywhere."
)

POOLING_CONTRACT = (
    "scorer.py applies ONE shared learned per-token encoder to each of the "
    "tokens_per_event tokens and then pools symmetrically (mean / sum). The "
    "emitted order of the two incidence presentations is canonical sorted for "
    "artifact determinism ONLY; symmetric pooling over a shared encoder is what "
    "makes endpoint symmetry hold by construction, so the order carries no "
    "information. The presentations must therefore NOT be pre-concatenated."
)

# --- Arm A: the declared projective normalization --------------------------

PROJECTIVE_NORMALIZATION = (
    "topographo.ssd.projective.canonicalize(event): scale the exact rational "
    "16-vector by 1/c where c is its FIRST NONZERO coefficient, so the first "
    "nonzero coefficient of the representative is exactly 1. Coordinates are then "
    "taken in native index order 0..15 and cast to float. No reordering, no "
    "rescaling, no extra channel."
)

ARM_A_TOKEN_DIM = 16

# --- Arms B / C / D: the exact one-hot field layout ------------------------

FIELD_LAYOUT = (
    (
        "S",
        0,
        2,
        (
            "one-hot over S in {0, 1}. TWO units, not one: both S values genuinely "
            "occur (7 habitats each), so a single bit would represent S=0 by the "
            "all-zero pattern and make a real value indistinguishable from an "
            "absent field under sum pooling. Two units keep every field one-hot "
            "and every token at exactly 4 active units."
        ),
    ),
    (
        "FFF",
        2,
        7,
        (
            "one-hot over the nonzero Fano points 1..7. NO unit is allocated to "
            "FFF=000: it is a formal algebraic completion value, never an Event "
            "coordinate, so allocating it a unit would invite the learner to read "
            "a completion value as data."
        ),
    ),
    (
        "PP",
        9,
        4,
        (
            "one-hot over q in F_2^2 = {00, 01, 10, 11}. FOUR units, not two: "
            "PP=00 is the declared ORIGIN BLOCK, a legitimate occurring chart "
            "value and NOT a formal completion value, so it must carry a unit. A "
            "2-bit binary encoding would render the origin as all-zeros, "
            "colliding with 'field absent' under sum pooling and breaking the "
            "one-hot discipline that keeps the {0, 1} fence checkable."
        ),
    ),
    (
        "pp",
        13,
        3,
        (
            "one-hot over the nonzero displacements {01, 10, 11}. NO unit is "
            "allocated to pp=00: it is a formal algebraic completion value, never "
            "a displacement (sfp.IncidencePresentation rejects it outright)."
        ),
    ),
)

SFP_TOKEN_DIM = sum(width for _, _, width, _ in FIELD_LAYOUT)
ACTIVE_UNITS_PER_SFP_TOKEN = len(FIELD_LAYOUT)

# --- Arm C: the frozen scramble policy -------------------------------------

SCRAMBLE_SEED = "009/task6/arm-c-scramble/v1"

SCRAMBLE_POLICY = {
    "seed_string": SCRAMBLE_SEED,
    "seed_derivation": (
        "seed_int = int(sha256('<SCRAMBLE_SEED>|<habitat_label>').hexdigest(), 16); "
        "index = seed_int % 696 into the LEXICOGRAPHICALLY SORTED tuple of the 696 "
        "non-structure-preserving permutations of the 6 K4 edges. No hash(), no "
        "random module, no set or dict iteration order participates."
    ),
    "independence": (
        "one independent digest-pinned scramble per habitat, keyed by the habitat "
        "label, so no single global relabeling can accidentally restore a common "
        "relation across habitats"
    ),
    "rejection_test": (
        "groups.is_structure_preserving(perm) == False — membership in the "
        "24-element class induced by K4 vertex automorphisms / affine AGL(2,2) "
        "relabelings. Matching-partition preservation is NOT the test (48 of 720 "
        "preserve the partition, only 24 are induced) and is reported only as a "
        "diagnostic."
    ),
    "frozen_before_training": (
        "the accepted permutations are derived here, written to the artifact, and "
        "digest-pinned; no model exists in this module and nothing downstream may "
        "reselect them"
    ),
}

EDGE_COUNT = len(groups.EDGES)
TOTAL_EDGE_PERMUTATIONS = 720
INDUCED_EDGE_PERMUTATIONS = 24
NON_INDUCED_EDGE_PERMUTATIONS = 696

# --- Arm D: the frozen structure-preserving relabeling ---------------------

ARM_D_GL32 = (1, 2, 3, 5, 4, 7, 6)
ARM_D_AFFINE = (((0, 1), (1, 0)), (0, 1))
ARM_D_FLIP_S = True

ARM_D_POLICY = {
    "FFF": {
        "group": "GL(3,2)",
        "element_image_form": list(ARM_D_GL32),
        "selection": (
            "the lexicographically least NON-IDENTITY element of groups.gl_3_2() "
            "in its canonical sorted order"
        ),
        "action": "FFF -> groups.apply_fff(pi, FFF); fixes 1,2,3 and swaps 4<->5, 6<->7",
    },
    "PP_pp": {
        "group": "AGL(2,2)",
        "element": {"A": [list(row) for row in ARM_D_AFFINE[0]], "b": list(ARM_D_AFFINE[1])},
        "selection": (
            "the first element of groups.agl_2_2() in canonical order with a "
            "NON-IDENTITY linear part (A != ((1,0),(0,1))) and a NONZERO "
            "translation (b != (0,0)), so that both PP and pp genuinely move"
        ),
        "action": (
            "PP: q -> groups.apply_affine((A, b), q) = A q XOR b;  "
            "pp: d -> groups.pp_displacement_image(A, d) = A d. The translation "
            "cancels in the displacement by the certified covariance law "
            "phi(q XOR d) XOR phi(q) == A d, so the two endpoint presentations "
            "are rebuilt COVARIANTLY rather than re-derived."
        ),
    },
    "S": {
        "group": "Z/2",
        "applied": ARM_D_FLIP_S,
        "action": "S -> groups.flip_s(S), applied globally and consistently to all 84",
        "why_consistent": (
            "admission requires equal S and forcing copies S, so a global flip is a "
            "symmetry; groups.global_s_flip_report certifies that a one-sided "
            "partial flip is NOT"
        ),
    },
    "character": (
        "EQUIVALENCE / BASIS CONTROL, not a destructive ablation: Arm D is a "
        "different chart for the same structure, and the certified admit/force "
        "relation is preserved exactly."
    ),
}

PINS = {
    "events": 84,
    "habitats": 14,
    "events_per_habitat": 6,
    "incidences": 168,
    "ordered_pairs": 84 * 84,
    "admit": 336,
    "edges_per_habitat": EDGE_COUNT,
    "total_edge_permutations": TOTAL_EDGE_PERMUTATIONS,
    "induced_edge_permutations": INDUCED_EDGE_PERMUTATIONS,
    "non_structure_preserving": NON_INDUCED_EDGE_PERMUTATIONS,
    "arm_a_token_dim": ARM_A_TOKEN_DIM,
    "sfp_token_dim": SFP_TOKEN_DIM,
    "active_units_per_sfp_token": ACTIVE_UNITS_PER_SFP_TOKEN,
}


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------

def pin(expected: object, observed: object, source: str) -> dict:
    """Record an expected/observed pair with an explicit agreement flag."""

    return {
        "expected": expected,
        "observed": observed,
        "agrees": expected == observed,
        "source": source,
    }


def seed_int(label: str) -> int:
    """Digest-pinned integer seed. Never ``hash()``, never iteration order."""

    blob = f"{SCRAMBLE_SEED}|{label}".encode()
    return int(hashlib.sha256(blob).hexdigest(), 16)


def seed_hex(label: str) -> str:
    return hashlib.sha256(f"{SCRAMBLE_SEED}|{label}".encode()).hexdigest()


def encode_perm(perm: tuple[int, ...]) -> str:
    """Same compact image-form encoding as ``groups.encode_perm``."""

    return "".join(str(x) for x in perm)


# ---------------------------------------------------------------------------
# the interface contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One representation arm: a set of equal-width token vectors per Event.

    ``tokens`` holds 84 entries, index-aligned to the ``task.py`` catalogue.
    Entry ``i`` is a tuple of ``tokens_per_event`` token vectors, each of length
    ``token_dim``, holding plain floats. The scorer applies a shared per-token
    encoder and pools symmetrically over the tokens of an Event.
    """

    name: str
    tokens: tuple[tuple[tuple[float, ...], ...], ...]
    token_dim: int
    tokens_per_event: int
    exposes: tuple[str, ...]
    forbids: tuple[str, ...]
    metadata: dict

    def __post_init__(self) -> None:
        if len(self.tokens) != PINS["events"]:
            raise AssertionError(
                f"arm {self.name} holds {len(self.tokens)} entries, not 84"
            )
        for index, entry in enumerate(self.tokens):
            if len(entry) != self.tokens_per_event:
                raise AssertionError(
                    f"arm {self.name} Event {index} has {len(entry)} tokens, not "
                    f"{self.tokens_per_event}"
                )
            for token in entry:
                if len(token) != self.token_dim:
                    raise AssertionError(
                        f"arm {self.name} Event {index} has a token of width "
                        f"{len(token)}, not {self.token_dim}"
                    )
                if not all(isinstance(v, float) for v in token):
                    raise AssertionError(
                        f"arm {self.name} emitted a non-float token value"
                    )

    @property
    def total_input_units(self) -> int:
        return self.token_dim * self.tokens_per_event

    def value_range(self) -> tuple[float, ...]:
        """The sorted set of distinct token values actually emitted."""

        return tuple(
            sorted({v for entry in self.tokens for token in entry for v in token})
        )

    def unit_marginals(self) -> tuple[int, ...]:
        """Per-unit count of ones over all ``84 * tokens_per_event`` tokens."""

        totals = [0] * self.token_dim
        for entry in self.tokens:
            for token in entry:
                for k, value in enumerate(token):
                    if value == 1.0:
                        totals[k] += 1
        return tuple(totals)

    def as_rows(self) -> list:
        return [
            [[float(v) for v in token] for token in entry] for entry in self.tokens
        ]

    def sha256(self) -> str:
        """SHA-256 of the full token tensor plus its declared shape."""

        return digest(
            {
                "name": self.name,
                "token_dim": self.token_dim,
                "tokens_per_event": self.tokens_per_event,
                "tokens": self.as_rows(),
            }
        )


# ---------------------------------------------------------------------------
# SFP token encoding (Arms B, C, D)
# ---------------------------------------------------------------------------

def sfp_token(s: int, fff: int, q: int, d: int) -> tuple[float, ...]:
    """One ``(S, FFF, PP, pp)`` incidence presentation as pure one-hot indicators.

    Every value is ``0.0`` or ``1.0`` by construction. No unit exists for
    ``FFF=000`` or ``pp=00`` — both are formal completion values only. A rational
    ray coordinate cannot survive this encoding, which is what makes the
    "no native coordinate in Arm B/C/D" fence mechanically checkable.
    """

    if s not in (0, 1):
        raise ValueError(f"S must be a bit, got {s!r}")
    if not 1 <= fff <= 7:
        raise ValueError(f"FFF must be a nonzero Fano point, got {fff!r}")
    if not 0 <= q <= 3:
        raise ValueError(f"PP must lie in F_2^2, got {q!r}")
    if d not in (1, 2, 3):
        raise ValueError(f"pp must be a nonzero displacement, got {d!r}")

    token = [0.0] * SFP_TOKEN_DIM
    token[0 + s] = 1.0
    token[2 + (fff - 1)] = 1.0
    token[9 + q] = 1.0
    token[13 + (d - 1)] = 1.0
    return tuple(token)


def token_of_presentation(presentation: IncidencePresentation) -> tuple[float, ...]:
    return sfp_token(
        presentation.s, presentation.fff, presentation.q, presentation.d
    )


def tokens_of_address(address: SfpAddress) -> tuple[tuple[float, ...], ...]:
    """Both endpoint incidence presentations, in canonical sorted order.

    The order is for artifact determinism only: the scorer's symmetric pooling
    over a shared per-token encoder is what makes the pair unordered.
    """

    return tuple(
        token_of_presentation(presentation) for presentation in address.incidences()
    )


def code_string(address: SfpAddress) -> str:
    """Compact complete textual record of one Event code (both presentations)."""

    return "|".join(
        f"S={p.s},FFF={p.fff:03b},PP={bits2(p.q)},pp={bits2(p.d)}"
        for p in address.incidences()
    )


# ---------------------------------------------------------------------------
# habitat / K4 edge plumbing shared by Arms C and D
# ---------------------------------------------------------------------------

def edge_index_of_address(address: SfpAddress) -> int:
    """The ``groups.EDGES`` index of an Event's K4 endpoint pair.

    Chamber index and the integer ``q`` coincide: ``groups.V2[q]`` is the 2-bit
    vector of ``q`` and ``groups.CHAMBER_INDEX[groups.V2[q]] == q``, so the
    habitat's 4 block coordinates are exactly the 4 abstract K4 vertices.
    """

    ends = tuple(sorted(address.endpoints))
    return groups.EDGE_INDEX[ends]


def address_of_edge(s: int, fff: int, edge: int) -> SfpAddress:
    """The unique habitat code word sitting on a given K4 edge.

    Inside one habitat an SFP code word *is* an edge: the edge ``{q1, q2}``
    determines ``pp = q1 XOR q2`` and both presentations ``(q1, pp)``,
    ``(q2, pp)``. So a permutation of the six codes is exactly a permutation of
    the six edges, which is what makes ``groups.is_structure_preserving`` the
    right — and only — rejection test.
    """

    q1, q2 = groups.EDGES[edge]
    return address_of(s, fff, q1, q1 ^ q2)


def native_codes(catalogue: Catalogue, codec: SfpCodec) -> tuple[SfpAddress, ...]:
    """The 84 native SFP addresses, index-aligned to the catalogue."""

    codes = tuple(codec.encode(event) for event in catalogue.events)
    if len({code.as_key() for code in codes}) != PINS["events"]:
        raise AssertionError("native SFP codes are not injective on the catalogue")
    return codes


def habitat_rows(catalogue: Catalogue) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """The 14 habitats with their 6 member Event indices, ascending."""

    rows = catalogue.events_of_habitat()
    if len(rows) != PINS["habitats"]:
        raise AssertionError(f"found {len(rows)} habitats, not 14")
    for label, members in rows:
        if len(members) != PINS["events_per_habitat"]:
            raise AssertionError(f"habitat {label} holds {len(members)} Events, not 6")
    return rows


# ---------------------------------------------------------------------------
# Arm A — native FIPS
# ---------------------------------------------------------------------------

def canonical_ray(event: tuple) -> tuple[Fraction, ...]:
    """The declared projective representative: first nonzero coefficient is 1."""

    return tuple(projective.canonicalize(event))


def build_arm_a(catalogue: Catalogue) -> Arm:
    """Arm A: the certified projective Event ray, normalized, as 16 floats.

    A pure function of the ray. There is no codec, no chart and no SFP field in
    scope: the signature admits only the catalogue, so no SFP field can enter.
    """

    tokens: list[tuple[tuple[float, ...], ...]] = []
    first_nonzero_is_one = True
    lossless = True
    for event in catalogue.events:
        exact = canonical_ray(event)
        leading = next((c for c in exact if c), None)
        if leading != 1:
            first_nonzero_is_one = False
        token = tuple(float(c) for c in exact)
        if any(Fraction(value) != coefficient for value, coefficient in zip(token, exact)):
            lossless = False
        # Exact certified identity, never float proximity.
        if not projective.equivalent(event, exact):
            raise AssertionError("declared normalization broke projective identity")
        tokens.append((token,))

    if not first_nonzero_is_one:
        raise AssertionError("declared normalization did not put 1 in the lead")
    if not lossless:
        raise AssertionError("the float cast of a canonical ray was lossy")

    distinct = {entry[0] for entry in tokens}
    values = sorted({v for entry in tokens for v in entry[0]})

    metadata = {
        "construction": (
            "token = tuple(float(c) for c in "
            "topographo.ssd.projective.canonicalize(event)), native coordinate "
            "order 0..15, one token per Event"
        ),
        "declared_projective_normalization": PROJECTIVE_NORMALIZATION,
        "slot_semantics": (
            "slot k is native ray coordinate k; 16 slots, 16 coordinates, zero "
            "units allocated to S, FFF, PP or pp"
        ),
        "no_sfp_field": {
            "sfp_units_allocated": 0,
            "signature_admits_only_the_catalogue": True,
            "codec_in_scope": False,
            "token_dim_equals_native_ray_length": pin(
                ARM_A_TOKEN_DIM, ARM_A_TOKEN_DIM, "len(canonicalize(event))"
            ),
        },
        "purity": {
            "statement": (
                "Arm A tokens are a pure function of the ray. The fence is a "
                "PROVENANCE fence, not an information-theoretic one: the ray "
                "determines the Event and therefore its habitat, so 'no SFP field "
                "exposed' means no code UNIT is allocated and no chart-dependent "
                "quantity is emitted — not that the habitat is unknowable."
            ),
            "distinct_tokens": len(distinct),
            "injective_on_the_catalogue": len(distinct) == PINS["events"],
            "value_range": values,
            "not_an_indicator_code": values != [0.0, 1.0],
        },
        "exactness": CORRECTNESS_CRITERION,
        "float_cast_is_lossless": True,
        "leading_coefficient_is_one_for_all_84": True,
    }

    return Arm(
        name="A_native",
        tokens=tuple(tokens),
        token_dim=ARM_A_TOKEN_DIM,
        tokens_per_event=1,
        exposes=(
            "the exact certified projective Event ray, 16 rational coordinates "
            "under the declared normalization (first nonzero coefficient = 1)",
        ),
        forbids=(
            "S", "FFF", "PP", "pp",
            "any SFP code unit",
            "the frozen SFP chart (origin block, pp-to-line assignment, sign bit)",
            "any chart-dependent quantity",
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Arm B — exact SFP relational code
# ---------------------------------------------------------------------------

def build_arm_b(catalogue: Catalogue, codec: SfpCodec) -> Arm:
    """Arm B: the exact ``021.05`` finite code, two one-hot tokens per Event."""

    codes = native_codes(catalogue, codec)
    tokens = tuple(tokens_of_address(code) for code in codes)

    both_decode = 0
    for index, code in enumerate(codes):
        rays = {codec.decode_incidence(p) for p in code.incidences()}
        if len(rays) == 1 and next(iter(rays)) == catalogue.events[index]:
            both_decode += 1
    if both_decode != PINS["events"]:
        raise AssertionError(
            f"only {both_decode} Events had both presentations decode to themselves"
        )

    values = sorted({v for entry in tokens for token in entry for v in token})
    if values != [0.0, 1.0]:
        raise AssertionError(f"Arm B emitted values outside {{0, 1}}: {values}")
    active = sorted(
        {int(sum(token)) for entry in tokens for token in entry}
    )
    if active != [ACTIVE_UNITS_PER_SFP_TOKEN]:
        raise AssertionError(f"Arm B token activation profile is {active}, not [4]")

    metadata = {
        "construction": (
            "per Event, one one-hot token per endpoint incidence presentation "
            "(S, FFF, PP, pp) with pp != 00, emitted in canonical sorted order"
        ),
        "source": (
            "sfp.SfpCodec().encode(ray) — the exact 021.05 finite code and nothing "
            "else; the frozen chart digest is pinned in the audit"
        ),
        "unordered_pair": POOLING_CONTRACT,
        "field_layout": [
            {"field": name, "offset": offset, "width": width, "justification": why}
            for name, offset, width, why in FIELD_LAYOUT
        ],
        "token_dim": SFP_TOKEN_DIM,
        "no_native_ray_coordinate": {
            "mechanism": (
                "every field is one-hot, so every token value lies in {0.0, 1.0}; "
                "a rational ray coordinate cannot survive such an encoding. The "
                "fence is therefore mechanically checkable rather than a promise."
            ),
            "value_range": values,
            "value_range_is_binary": values == [0.0, 1.0],
            "active_units_per_token": active,
        },
        "completion_values_unallocated": {
            "FFF=000": "no unit",
            "pp=00": "no unit",
            "PP=00": (
                "ALLOCATED a unit — PP=00 is the declared origin block, a real "
                "occurring chart value, NOT a formal completion value"
            ),
        },
        "both_presentations_decode_to_the_same_event": both_decode,
        "interpretation_fence": (
            "Arm B success would establish representation SUFFICIENCY / USE, not "
            "representation DISCOVERY: sfp.ExactSfpCircuit already solves the "
            "relation from this code without learning anything."
        ),
        "codes": [code_string(code) for code in codes],
    }

    return Arm(
        name="B_sfp",
        tokens=tokens,
        token_dim=SFP_TOKEN_DIM,
        tokens_per_event=2,
        exposes=(
            "S (habitat sign bit)",
            "FFF (nonzero Fano point)",
            "PP (affine cyclic-block coordinate q)",
            "pp (nonzero displacement d)",
            "the unordered pair of endpoint incidence presentations",
        ),
        forbids=(
            "every native 16-coordinate Event ray coordinate",
            "any token value outside {0.0, 1.0}",
            "a unit for FFF=000",
            "a unit for pp=00",
            "a privileged endpoint order",
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Arm C — rule-scrambled SFP-shaped code (THE LOAD-BEARING CONTROL)
# ---------------------------------------------------------------------------

def non_induced_edge_permutations() -> tuple[tuple[int, ...], ...]:
    """The 696 permutations of the 6 K4 edges that are NOT structure preserving."""

    found = tuple(
        perm
        for perm in itertools.permutations(range(EDGE_COUNT))
        if not groups.is_structure_preserving(perm)
    )
    if len(found) != NON_INDUCED_EDGE_PERMUTATIONS:
        raise AssertionError(f"found {len(found)} non-induced permutations, not 696")
    return found


def choose_scramble(label: str, pool: tuple[tuple[int, ...], ...]) -> dict:
    """Pick and certify one non-automorphic edge permutation for a habitat."""

    value = seed_int(label)
    index = value % len(pool)
    perm = pool[index]

    # THE REJECTION PROOF: membership in the 24-element induced class, not
    # matching-partition preservation.
    if groups.is_structure_preserving(perm):
        raise AssertionError(
            f"habitat {label} drew a STRUCTURE PRESERVING permutation {perm}; "
            "that is an automorphism, not a scramble"
        )
    if sorted(perm) != list(range(EDGE_COUNT)):
        raise AssertionError(f"habitat {label} drew a non-permutation {perm}")

    return {
        "habitat": label,
        "seed_sha256": seed_hex(label),
        "pool_index": index,
        "edge_permutation": list(perm),
        "encoded": encode_perm(perm),
        "edge_images": {
            "".join(map(str, groups.EDGES[k])): "".join(
                map(str, groups.EDGES[perm[k]])
            )
            for k in range(EDGE_COUNT)
        },
        "is_structure_preserving": groups.is_structure_preserving(perm),
        "rejected_as_automorphism": not groups.is_structure_preserving(perm),
        "preserves_matching_partition_DIAGNOSTIC_ONLY": (
            groups.preserves_matching_partition(perm)
        ),
    }


def scramble_plan(catalogue: Catalogue) -> tuple[dict, ...]:
    """One independent digest-pinned scramble per habitat, frozen here."""

    pool = non_induced_edge_permutations()
    return tuple(
        choose_scramble(label, pool) for label, _ in habitat_rows(catalogue)
    )


def build_arm_c(catalogue: Catalogue, codec: SfpCodec) -> Arm:
    """Arm C: SFP-shaped code with the native alignment deliberately destroyed."""

    codes = native_codes(catalogue, codec)
    plan = scramble_plan(catalogue)
    plan_by_habitat = {row["habitat"]: row for row in plan}

    scrambled: list[SfpAddress | None] = [None] * PINS["events"]
    assignment_rows = []
    for label, members in habitat_rows(catalogue):
        perm = tuple(plan_by_habitat[label]["edge_permutation"])
        seen: set[int] = set()
        for index in members:
            native = codes[index]
            edge = edge_index_of_address(native)
            image = perm[edge]
            if image in seen:
                raise AssertionError("scramble reused an edge inside a habitat")
            seen.add(image)
            code = address_of_edge(native.s, native.fff, image)
            scrambled[index] = code
            assignment_rows.append(
                {
                    "event_index": index,
                    "habitat": label,
                    "native_edge": "".join(map(str, groups.EDGES[edge])),
                    "scrambled_edge": "".join(map(str, groups.EDGES[image])),
                    "native_code": code_string(native),
                    "scrambled_code": code_string(code),
                }
            )
        if len(seen) != PINS["events_per_habitat"]:
            raise AssertionError(f"habitat {label} scramble is not a bijection")

    if any(code is None for code in scrambled):
        raise AssertionError("a catalogue Event received no scrambled code")
    final = tuple(code for code in scrambled if code is not None)
    if len({code.as_key() for code in final}) != PINS["events"]:
        raise AssertionError("Arm C codes are not injective on the 84 Events")
    if {code.as_key() for code in final} != {code.as_key() for code in codes}:
        raise AssertionError(
            "Arm C did not reuse exactly the 84 valid SFP code words; marginals "
            "would be confounded"
        )

    tokens = tuple(tokens_of_address(code) for code in final)
    values = sorted({v for entry in tokens for token in entry for v in token})
    if values != [0.0, 1.0]:
        raise AssertionError(f"Arm C emitted values outside {{0, 1}}: {values}")

    distinct_perms = len({row["encoded"] for row in plan})

    metadata = {
        "role": (
            "THE LOAD-BEARING CONTROL. Same bit widths, same habitat sizes, six "
            "Event codes per habitat, two incidence presentations per Event and "
            "matched field marginals — with the alignment between the code "
            "relation and the certified native FIPS relation deliberately "
            "destroyed."
        ),
        "construction": [
            "1. start from the valid six SFP edge codes in each habitat",
            "2. apply a fixed permutation of those six codes onto the six "
            "certified native Events (inside a habitat an SFP code word IS a K4 "
            "edge, so this is exactly an edge permutation)",
            "3. REJECT any permutation induced by a K4 vertex automorphism / "
            "affine AGL(2,2) relabeling (groups.is_structure_preserving == False)",
            "4. freeze the accepted non-automorphic permutation before any "
            "training (no model exists in this module)",
            "5. use independent digest-pinned scrambles across habitats",
        ],
        "scramble_policy": SCRAMBLE_POLICY,
        "counting_context": {
            "total_edge_permutations": pin(
                TOTAL_EDGE_PERMUTATIONS,
                TOTAL_EDGE_PERMUTATIONS,
                "6! permutations of the 6 K4 edges",
            ),
            "induced": pin(
                INDUCED_EDGE_PERMUTATIONS,
                len(groups.induced_edge_permutations()),
                "groups.induced_edge_permutations()",
            ),
            "not_structure_preserving": pin(
                NON_INDUCED_EDGE_PERMUTATIONS,
                NON_INDUCED_EDGE_PERMUTATIONS,
                "720 - 24, the pool Arm C draws from",
            ),
            "statement": "6! = 720 total, 24 induced, 696 not structure preserving",
        },
        "edge_convention": (
            "groups.EDGES = "
            + ",".join("".join(map(str, e)) for e in groups.EDGES)
            + "; permutations are in image form on groups.EDGE_INDEX indices"
        ),
        "per_habitat_scrambles": list(plan),
        "distinct_permutations_chosen": distinct_perms,
        "every_permutation_rejected_as_automorphism": all(
            row["rejected_as_automorphism"] for row in plan
        ),
        "trap_1_pp_permutation": (
            "NOT USED. groups.pp_permutations_preserve_xor_third certifies that all "
            "6 permutations of the nonzero pp values lie in GL(2,2) ~= S3 and "
            "preserve d3 = d1 XOR d2; its conclusion field reads 'REJECTED AS A "
            "SCRAMBLE ... measures nothing'. A pp-value permutation is a symmetry."
        ),
        "trap_2_matching_partition": (
            "NOT USED AS THE TEST. 48 of the 720 edge permutations preserve the "
            "3-matching partition while only 24 are induced, so preservation is "
            "necessary but not sufficient. A partition-preserving non-induced "
            "permutation is still a legitimate scramble (it carries a K4 star — a "
            "certified block — onto a K4 triangle, which is not a block). Only "
            "groups.is_structure_preserving is used to reject."
        ),
        "habitat_partition_preserved_deliberately": (
            "the scramble acts inside habitats only, so Arm C keeps exactly the "
            "habitat sizes and the cross-habitat information Arm B has. That is "
            "required for matched marginals; the destruction is targeted at the "
            "within-habitat incidence alignment, which is where all the relational "
            "signal lives (336 ADMIT + 84 SAME_HABITAT_DISJOINT)."
        ),
        "no_native_ray_coordinate": {
            "value_range": values,
            "value_range_is_binary": values == [0.0, 1.0],
        },
        "codes": [code_string(code) for code in final],
        "assignment": assignment_rows,
    }

    return Arm(
        name="C_scrambled",
        tokens=tokens,
        token_dim=SFP_TOKEN_DIM,
        tokens_per_event=2,
        exposes=(
            "S (habitat sign bit)",
            "FFF (nonzero Fano point)",
            "PP (affine cyclic-block coordinate q) — REASSIGNED",
            "pp (nonzero displacement d) — REASSIGNED",
            "the unordered pair of endpoint incidence presentations",
            "identical field marginals, bit widths and habitat sizes to Arm B",
        ),
        forbids=(
            "every native 16-coordinate Event ray coordinate",
            "any token value outside {0.0, 1.0}",
            "a unit for FFF=000",
            "a unit for pp=00",
            "alignment between the code relation and the certified native FIPS "
            "relation (destroyed by a per-habitat non-automorphic edge permutation)",
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Arm D — structure-preserving SFP relabeling
# ---------------------------------------------------------------------------

def relabel_address(address: SfpAddress) -> SfpAddress:
    """Apply the frozen structure-preserving relabeling, rebuilding covariantly."""

    matrix, _ = ARM_D_AFFINE
    s = groups.flip_s(address.s) if ARM_D_FLIP_S else address.s
    fff = groups.apply_fff(ARM_D_GL32, address.fff)
    q0, d0 = address.presentations[0]
    q = groups.CHAMBER_INDEX[groups.apply_affine(ARM_D_AFFINE, groups.V2[q0])]
    d = groups.CHAMBER_INDEX[
        groups.pp_displacement_image(matrix, groups.V2[d0])
    ]
    if d == 0:
        raise AssertionError("an invertible linear map sent a nonzero pp to 00")
    # Covariance: phi(q XOR d) XOR phi(q) == A d, so the second presentation is
    # rebuilt by the SAME displacement rather than recomputed independently.
    other = groups.CHAMBER_INDEX[
        groups.apply_affine(ARM_D_AFFINE, groups.V2[q0 ^ d0])
    ]
    if other != q ^ d:
        raise AssertionError("affine relabeling violated the covariance law")
    return address_of(s, fff, q, d)


def build_arm_d(catalogue: Catalogue, codec: SfpCodec) -> Arm:
    """Arm D: an equivalent Arm-B coding under certified preserving maps only."""

    codes = native_codes(catalogue, codec)
    relabeled = tuple(relabel_address(code) for code in codes)

    if len({code.as_key() for code in relabeled}) != PINS["events"]:
        raise AssertionError("Arm D relabeling is not injective on the 84 Events")
    if {code.as_key() for code in relabeled} != {code.as_key() for code in codes}:
        raise AssertionError(
            "Arm D left the 84-code-word alphabet; it is not a relabeling"
        )

    tokens = tuple(tokens_of_address(code) for code in relabeled)
    values = sorted({v for entry in tokens for token in entry for v in token})
    if values != [0.0, 1.0]:
        raise AssertionError(f"Arm D emitted values outside {{0, 1}}: {values}")

    # The mirror image of the Arm C proof, per habitat: the induced edge
    # permutation IS structure preserving.
    edge_rows = []
    for label, members in habitat_rows(catalogue):
        images: dict[int, int] = {}
        for index in members:
            images[edge_index_of_address(codes[index])] = edge_index_of_address(
                relabeled[index]
            )
        perm = tuple(images[k] for k in range(EDGE_COUNT))
        preserving = groups.is_structure_preserving(perm)
        if not preserving:
            raise AssertionError(
                f"habitat {label} produced a NON structure preserving edge "
                f"permutation {perm}; Arm D would be an ablation, not a control"
            )
        edge_rows.append(
            {
                "habitat": label,
                "edge_permutation": list(perm),
                "encoded": encode_perm(perm),
                "is_structure_preserving": preserving,
                "matches_the_affine_prediction": perm
                == groups.induced_edge_permutation(
                    groups.chamber_permutation(ARM_D_AFFINE)
                ),
            }
        )

    predicted = groups.induced_edge_permutation(
        groups.chamber_permutation(ARM_D_AFFINE)
    )

    metadata = {
        "role": ARM_D_POLICY["character"],
        "construction": (
            "relabel every native address by FFF -> pi(FFF) in GL(3,2), "
            "q -> A q XOR b and d -> A d in AGL(2,2), plus a global consistent S "
            "flip; the two endpoint presentations are rebuilt COVARIANTLY through "
            "sfp.address_of and the covariance law is asserted per Event"
        ),
        "transformations": ARM_D_POLICY,
        "bijectivity": {
            "distinct_relabeled_codes": len({c.as_key() for c in relabeled}),
            "is_bijection_on_the_84_events": True,
            "stays_inside_the_84_code_word_alphabet": True,
        },
        "induced_edge_permutations": {
            "predicted_from_the_affine_element": list(predicted),
            "per_habitat": edge_rows,
            "all_structure_preserving": all(
                row["is_structure_preserving"] for row in edge_rows
            ),
            "all_match_the_affine_prediction": all(
                row["matches_the_affine_prediction"] for row in edge_rows
            ),
            "mirror_of_arm_c": (
                "Arm C requires is_structure_preserving == False on every habitat; "
                "Arm D requires True on every habitat. Same predicate, opposite "
                "verdict."
            ),
        },
        "no_native_ray_coordinate": {
            "value_range": values,
            "value_range_is_binary": values == [0.0, 1.0],
        },
        "not_an_ablation": (
            "Arm D is an EQUIVALENCE / BASIS control. It destroys nothing: the "
            "certified admit/force relation is preserved exactly on all 7056 "
            "ordered pairs (see the relation report)."
        ),
        "codes": [code_string(code) for code in relabeled],
    }

    return Arm(
        name="D_relabeled",
        tokens=tokens,
        token_dim=SFP_TOKEN_DIM,
        tokens_per_event=2,
        exposes=(
            "S (habitat sign bit) — globally flipped",
            "FFF (nonzero Fano point) — relabeled by GL(3,2)",
            "PP (affine cyclic-block coordinate q) — relabeled by q -> A q XOR b",
            "pp (nonzero displacement d) — relabeled covariantly by d -> A d",
            "the unordered pair of endpoint incidence presentations",
            "the same certified relation as Arm B, in a different basis",
        ),
        forbids=(
            "every native 16-coordinate Event ray coordinate",
            "any token value outside {0.0, 1.0}",
            "a unit for FFF=000",
            "a unit for pp=00",
            "the particular frozen chart of sfp.py (that is the point of the arm)",
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Arm E — optional, diagnostic opaque identity code
# ---------------------------------------------------------------------------

def build_arm_e(catalogue: Catalogue) -> Arm:
    """Arm E (OPTIONAL, DIAGNOSTIC): an arbitrary 84-way one-hot identity code."""

    n = len(catalogue)
    tokens = []
    for index in range(n):
        token = [0.0] * n
        token[index] = 1.0
        tokens.append((tuple(token),))

    metadata = {
        "status": "OPTIONAL and DIAGNOSTIC ONLY",
        "does_not_replace_arm_c": (
            "Arm E is capacity-matched noise about identity, not a marginal-matched "
            "structural control. Arm C remains the load-bearing control; Arm E only "
            "shows what pure memorization of Event identity can and cannot reach."
        ),
        "construction": "token[i] = 1.0 for Event index i, all other units 0.0",
        "capacity": (
            "84 units, exactly one per Event: maximal per-Event capacity with zero "
            "relational structure"
        ),
        "no_relational_structure": (
            "the code carries no habitat, no Fano point, no block coordinate and no "
            "displacement; any relation must be memorized pair by pair"
        ),
    }

    return Arm(
        name="E_opaque",
        tokens=tuple(tokens),
        token_dim=n,
        tokens_per_event=1,
        exposes=("an arbitrary 84-way identity index and nothing else",),
        forbids=(
            "every native 16-coordinate Event ray coordinate",
            "S", "FFF", "PP", "pp",
            "any relational or algebraic structure whatsoever",
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# the public builder
# ---------------------------------------------------------------------------

def all_arms(dataset: Dataset, include_e: bool = True) -> tuple[Arm, ...]:
    """Build every arm, index-aligned to ``dataset.catalogue``.

    ``include_e`` adds the optional diagnostic Arm E. Arms A / B / C / D are the
    science; Arm E never substitutes for Arm C.
    """

    catalogue = dataset.catalogue
    codec = SfpCodec()
    built = [
        build_arm_a(catalogue),
        build_arm_b(catalogue, codec),
        build_arm_c(catalogue, codec),
        build_arm_d(catalogue, codec),
    ]
    if include_e:
        built.append(build_arm_e(catalogue))
    return tuple(built)


# ---------------------------------------------------------------------------
# cross-arm audit
# ---------------------------------------------------------------------------

def field_counts(codes: tuple[SfpAddress, ...]) -> dict:
    """Per-field value counts over all ``84 * 2 = 168`` incidence presentations."""

    tally = {
        "S": Counter(),
        "FFF": Counter(),
        "PP": Counter(),
        "pp": Counter(),
    }
    for code in codes:
        for presentation in code.incidences():
            tally["S"][presentation.s] += 1
            tally["FFF"][presentation.fff] += 1
            tally["PP"][presentation.q] += 1
            tally["pp"][presentation.d] += 1
    return {
        field: {str(value): count for value, count in sorted(counter.items())}
        for field, counter in sorted(tally.items())
    }


def codes_of_arm(arm: Arm) -> tuple[SfpAddress, ...]:
    """Recover the SFP addresses an indicator arm actually emitted.

    Decoding from the emitted tokens rather than from the builder's internals is
    deliberate: it certifies that the token tensor really carries the code the
    construction record claims, and that the two tokens of an Event are the two
    endpoint presentations of ONE address.
    """

    recovered = []
    for index, entry in enumerate(arm.tokens):
        fields = []
        for token in entry:
            fields.append(
                (
                    list(token[0:2]).index(1.0),
                    list(token[2:9]).index(1.0) + 1,
                    list(token[9:13]).index(1.0),
                    list(token[13:16]).index(1.0) + 1,
                )
            )
        (s1, f1, q1, d1), (s2, f2, q2, d2) = fields
        if (s1, f1, d1) != (s2, f2, d2) or q2 != q1 ^ d1:
            raise AssertionError(
                f"{arm.name} Event {index}: the two tokens are not the two endpoint "
                "presentations of one address"
            )
        recovered.append(address_of(s1, f1, q1, d1))
    return tuple(recovered)


def shape_ledger(arms: tuple[Arm, ...]) -> list:
    """Capacity / shape ledger plus the per-arm token tensor digest."""

    rows = []
    for arm in arms:
        values = arm.value_range()
        rows.append(
            {
                "arm": arm.name,
                "token_dim": arm.token_dim,
                "tokens_per_event": arm.tokens_per_event,
                "total_input_units": arm.total_input_units,
                "events": len(arm.tokens),
                "tokens_total": len(arm.tokens) * arm.tokens_per_event,
                "value_range": list(values),
                "is_binary_indicator_code": list(values) == [0.0, 1.0],
                "active_units_per_token": sorted(
                    {int(sum(token)) for entry in arm.tokens for token in entry}
                ),
                "token_tensor_sha256": arm.sha256(),
            }
        )
    return rows


def exposure_table(arms: tuple[Arm, ...]) -> list:
    return [
        {
            "arm": arm.name,
            "exposes": list(arm.exposes),
            "forbids": list(arm.forbids),
        }
        for arm in arms
    ]


def relation_report(
    dataset: Dataset,
    codes: tuple[SfpAddress, ...],
    circuit: ExactSfpCircuit,
    arm_name: str,
) -> dict:
    """Structural agreement of an arm's code with the certified native labels.

    ``sfp.ExactSfpCircuit`` is used here as a nonlearned REFERENCE ORACLE for a
    structural check. It generates no labels: every comparison is against
    ``task.py``'s records, which come from the certified native FIPS relation.
    """

    inverse = {code.as_key(): index for index, code in enumerate(codes)}
    if len(inverse) != PINS["events"]:
        raise AssertionError(f"{arm_name}: code assignment is not injective")

    n = len(codes)
    admission_agree = 0
    admit_checked = 0
    target_agree = 0
    admission_disagree_examples: list[dict] = []
    target_disagree_examples: list[dict] = []

    for a in range(n):
        for b in range(n):
            record = dataset.record(a, b)
            native_admit = record.admitted
            code_admit = circuit.admit_event(codes[a], codes[b])
            if native_admit == code_admit:
                admission_agree += 1
            elif len(admission_disagree_examples) < 4:
                admission_disagree_examples.append(
                    {"a": a, "b": b, "native": native_admit, "code": code_admit}
                )
            if not native_admit:
                continue
            admit_checked += 1
            forced = circuit.forced_third_event(codes[a], codes[b])
            got = None if forced is None else inverse.get(forced.as_key())
            if got == record.target_index:
                target_agree += 1
            elif len(target_disagree_examples) < 4:
                target_disagree_examples.append(
                    {
                        "a": a,
                        "b": b,
                        "native_target": record.target_index,
                        "code_target": got,
                    }
                )

    total = n * n
    return {
        "arm": arm_name,
        "oracle": (
            "sfp.ExactSfpCircuit as a nonlearned reference oracle for a STRUCTURAL "
            "check; never a label source"
        ),
        "ordered_pairs_checked": total,
        "admission_agreements": admission_agree,
        "admission_disagreements": total - admission_agree,
        "admission_fully_agrees": admission_agree == total,
        "native_admit_pairs": admit_checked,
        "forced_third_agreements": target_agree,
        "forced_third_disagreements": admit_checked - target_agree,
        "forced_third_fully_agrees": target_agree == admit_checked,
        "relation_preserved": admission_agree == total and target_agree == admit_checked,
        "admission_disagreement_examples": admission_disagree_examples,
        "forced_third_disagreement_examples": target_disagree_examples,
    }


def per_habitat_destruction(
    dataset: Dataset,
    catalogue: Catalogue,
    codes: tuple[SfpAddress, ...],
    circuit: ExactSfpCircuit,
) -> dict:
    """Per habitat: how much of the certified relation Arm C actually destroyed.

    A global disagreement count could in principle hide a habitat whose scramble
    happened to leave the relation intact, so the destruction is measured habitat
    by habitat. Reference-oracle structural check only; not a label source.
    """

    inverse = {code.as_key(): index for index, code in enumerate(codes)}
    rows = []
    for label, members in habitat_rows(catalogue):
        admit_disagree = 0
        admit_pairs = 0
        third_agree = 0
        for a in members:
            for b in members:
                record = dataset.record(a, b)
                if record.admitted != circuit.admit_event(codes[a], codes[b]):
                    admit_disagree += 1
                if not record.admitted:
                    continue
                admit_pairs += 1
                forced = circuit.forced_third_event(codes[a], codes[b])
                got = None if forced is None else inverse.get(forced.as_key())
                if got == record.target_index:
                    third_agree += 1
        rows.append(
            {
                "habitat": label,
                "native_admit_pairs": admit_pairs,
                "admission_disagreements": admit_disagree,
                "forced_third_agreements": third_agree,
                "forced_third_destroyed": admit_pairs - third_agree,
                "relation_destroyed_in_this_habitat": third_agree < admit_pairs,
            }
        )
    return {
        "note": (
            "measured per habitat so that no habitat can hide behind a global "
            "count. Two habitats show ZERO admission disagreements while losing "
            "ALL 24 forced thirds: those scrambles are the matching-partition "
            "preserving, non-induced permutations, which carry each K4 star (a "
            "certified block) onto a K4 triangle. Adjacency survives, the block "
            "structure does not. That is trap 2 observed in the data, and it is "
            "why is_structure_preserving — not partition preservation — is the "
            "rejection test."
        ),
        "per_habitat": rows,
        "habitats": len(rows),
        "habitats_with_relation_destroyed": sum(
            1 for row in rows if row["relation_destroyed_in_this_habitat"]
        ),
        "every_habitat_destroyed": all(
            row["relation_destroyed_in_this_habitat"] for row in rows
        ),
    }


def arm_a_no_sfp_field(
    arm_a: Arm, codes: tuple[SfpAddress, ...], relabeled: tuple[SfpAddress, ...]
) -> dict:
    """Rigorous fence: Arm A is chart-invariant, the SFP fields are not.

    Arm A is built from the ray alone, so the Arm D re-charting leaves its token
    tensor bit-identical. Every SFP field, by contrast, genuinely moves under that
    re-charting. A quantity that is chart-invariant cannot equal a quantity that
    is not, so no Arm A slot can be an SFP field indicator. That is a mechanical
    proof, not a promise.
    """

    native = {
        "S": tuple(code.s for code in codes),
        "FFF": tuple(code.fff for code in codes),
        "PP": tuple(tuple(sorted(code.endpoints)) for code in codes),
        "pp": tuple(code.d for code in codes),
    }
    moved = {
        "S": tuple(code.s for code in relabeled),
        "FFF": tuple(code.fff for code in relabeled),
        "PP": tuple(tuple(sorted(code.endpoints)) for code in relabeled),
        "pp": tuple(code.d for code in relabeled),
    }
    changes = {
        field: native[field] != moved[field] for field in sorted(native)
    }
    values = list(arm_a.value_range())
    return {
        "sfp_units_allocated_in_arm_a": pin(
            0, arm_a.metadata["no_sfp_field"]["sfp_units_allocated"], "Arm A layout"
        ),
        "token_dim": pin(
            ARM_A_TOKEN_DIM, arm_a.token_dim, "the 16 native ray coordinates"
        ),
        "tokens_are_a_pure_function_of_the_ray": (
            arm_a.metadata["no_sfp_field"]["signature_admits_only_the_catalogue"]
            and not arm_a.metadata["no_sfp_field"]["codec_in_scope"]
        ),
        "chart_invariance_proof": {
            "statement": (
                "Arm A is built from the ray alone, hence invariant under the Arm D "
                "re-charting; every SFP field moves under that re-charting; a "
                "chart-invariant quantity cannot equal a chart-dependent one, so no "
                "Arm A slot is an SFP field indicator"
            ),
            "sfp_field_changes_under_arm_d_recharting": changes,
            "every_sfp_field_moves": all(changes.values()),
            "arm_a_depends_on_the_chart": False,
        },
        "value_range": values,
        "not_a_binary_indicator_code": values != [0.0, 1.0],
        "injective_on_the_catalogue": arm_a.metadata["purity"][
            "injective_on_the_catalogue"
        ],
        "fence_character": arm_a.metadata["purity"]["statement"],
    }


def marginal_comparison(
    arms: dict[str, Arm], codes: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """Arm B vs Arm C (and D): per-unit and per-field marginals must match."""

    b_units = arms["B_sfp"].unit_marginals()
    c_units = arms["C_scrambled"].unit_marginals()
    d_units = arms["D_relabeled"].unit_marginals()
    b_fields = field_counts(codes["B_sfp"])
    c_fields = field_counts(codes["C_scrambled"])
    d_fields = field_counts(codes["D_relabeled"])

    unit_labels = []
    for name, offset, width, _ in FIELD_LAYOUT:
        for k in range(width):
            unit_labels.append(f"{name}[{k}]@{offset + k}")

    return {
        "basis": (
            "counts of ones per unit over all 84 * 2 = 168 tokens, plus per-field "
            "value counts over the same 168 incidence presentations"
        ),
        "unit_labels": unit_labels,
        "B_unit_marginals": list(b_units),
        "C_unit_marginals": list(c_units),
        "D_unit_marginals": list(d_units),
        "B_field_counts": b_fields,
        "C_field_counts": c_fields,
        "D_field_counts": d_fields,
        "B_equals_C_unit_marginals": pin(
            list(b_units), list(c_units), "Arm C must be marginal-matched to Arm B"
        ),
        "B_equals_C_field_counts": pin(
            b_fields, c_fields, "per-field value multisets"
        ),
        "B_equals_D_unit_marginals": pin(
            list(b_units), list(d_units), "a relabeling permutes fields, not counts"
        ),
        "shape_match": pin(
            [
                arms["B_sfp"].token_dim,
                arms["B_sfp"].tokens_per_event,
                list(arms["B_sfp"].value_range()),
            ],
            [
                arms["C_scrambled"].token_dim,
                arms["C_scrambled"].tokens_per_event,
                list(arms["C_scrambled"].value_range()),
            ],
            "token_dim, tokens_per_event and value range must match Arm B",
        ),
        "confound_policy": (
            "if the marginals differ the arm is confounded and the CONSTRUCTION is "
            "fixed, never the report"
        ),
    }


def alignment_report(arms: tuple[Arm, ...], catalogue: Catalogue) -> dict:
    """Every arm is index-aligned to one and the same catalogue."""

    return {
        "catalogue_sha256": pin(
            EXPECTED_CATALOGUE_SHA256, catalogue.sha256(), "task.Catalogue.sha256()"
        ),
        "order_source": catalogue.order_source,
        "arms_with_84_entries": pin(
            [arm.name for arm in arms],
            [arm.name for arm in arms if len(arm.tokens) == PINS["events"]],
            "index alignment to the task.py catalogue order",
        ),
        "index_legend_sample": [
            {
                "index": index,
                "ray": catalogue.encodings[index],
                "habitat": habitat_label(*catalogue.habitats[index]),
                "S": sign_bit(catalogue.habitats[index][1]),
            }
            for index in (0, 1, 41, 82, 83)
        ],
        "statement": (
            "all arms share the frozen catalogue order, so token index i means the "
            "same Event in every arm and the scorer may index them interchangeably"
        ),
    }


def token_samples(arm: Arm, count: int = 3) -> list:
    return [
        {
            "index": index,
            "tokens": [[float(v) for v in token] for token in arm.tokens[index]],
        }
        for index in range(min(count, len(arm.tokens)))
    ]


# ---------------------------------------------------------------------------
# audit driver
# ---------------------------------------------------------------------------

def _collect(node: object, path: str, out: list) -> None:
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
    ledger = {row["arm"]: row for row in payload["shape_ledger"]}
    arm_a = payload["arm_a_fence"]
    arm_c = payload["arm_c_rejection_proof"]
    arm_d = payload["arm_d_preservation_proof"]
    marginals = payload["marginal_comparison"]
    relations = {row["arm"]: row for row in payload["relation_reports"]}
    return {
        "arm_a_token_dim_16": ledger["A_native"]["token_dim"] == ARM_A_TOKEN_DIM,
        "arm_a_single_token": ledger["A_native"]["tokens_per_event"] == 1,
        "arm_a_no_sfp_unit": arm_a["sfp_units_allocated_in_arm_a"]["agrees"],
        "arm_a_is_chart_invariant_while_fields_move": arm_a["chart_invariance_proof"][
            "every_sfp_field_moves"
        ],
        "arm_a_not_a_binary_indicator_code": arm_a["not_a_binary_indicator_code"],
        "arm_a_injective": arm_a["injective_on_the_catalogue"],
        "arm_b_binary_value_range": ledger["B_sfp"]["is_binary_indicator_code"],
        "arm_c_binary_value_range": ledger["C_scrambled"]["is_binary_indicator_code"],
        "arm_d_binary_value_range": ledger["D_relabeled"]["is_binary_indicator_code"],
        "arm_b_two_tokens": ledger["B_sfp"]["tokens_per_event"] == 2,
        "arm_c_two_tokens": ledger["C_scrambled"]["tokens_per_event"] == 2,
        "arm_d_two_tokens": ledger["D_relabeled"]["tokens_per_event"] == 2,
        "arm_b_four_active_units": ledger["B_sfp"]["active_units_per_token"] == [
            ACTIVE_UNITS_PER_SFP_TOKEN
        ],
        "arm_b_both_presentations_decode": (
            payload["arms"]["B_sfp"]["both_presentations_decode_to_the_same_event"]
            == PINS["events"]
        ),
        "arm_c_every_habitat_rejected": arm_c["every_habitat_is_non_automorphic"],
        "arm_c_no_habitat_structure_preserving": arm_c[
            "structure_preserving_habitats"
        ] == 0,
        "arm_c_relation_destroyed": not relations["C_scrambled"]["relation_preserved"],
        "arm_c_forced_third_destroyed": relations["C_scrambled"][
            "forced_third_disagreements"
        ] > 0,
        "arm_c_destroyed_in_every_habitat": arm_c["destruction"][
            "every_habitat_destroyed"
        ],
        "arm_c_marginals_match_arm_b": (
            marginals["B_equals_C_unit_marginals"]["agrees"]
            and marginals["B_equals_C_field_counts"]["agrees"]
        ),
        "arm_c_shape_matches_arm_b": marginals["shape_match"]["agrees"],
        "arm_d_every_habitat_preserving": arm_d["every_habitat_is_structure_preserving"],
        "arm_d_bijective": arm_d["is_bijection_on_the_84_events"],
        "arm_d_relation_preserved": relations["D_relabeled"]["relation_preserved"],
        "arm_b_relation_preserved": relations["B_sfp"]["relation_preserved"],
        "all_arms_index_aligned": payload["alignment"]["arms_with_84_entries"]["agrees"],
        "catalogue_pinned": payload["alignment"]["catalogue_sha256"]["agrees"],
        "arm_digests_distinct": payload["digests"]["arm_token_digests_distinct"],
    }


def audit() -> dict:
    dataset = build_dataset()
    catalogue = dataset.catalogue
    codec = SfpCodec()
    circuit = ExactSfpCircuit()
    chart = codec.chart()

    arms = all_arms(dataset)
    by_name = {arm.name: arm for arm in arms}

    native = native_codes(catalogue, codec)
    arm_codes = {
        "B_sfp": codes_of_arm(by_name["B_sfp"]),
        "C_scrambled": codes_of_arm(by_name["C_scrambled"]),
        "D_relabeled": codes_of_arm(by_name["D_relabeled"]),
    }
    if [code.as_key() for code in arm_codes["B_sfp"]] != [
        code.as_key() for code in native
    ]:
        raise AssertionError("Arm B tokens do not decode back to the native codes")

    plan = by_name["C_scrambled"].metadata["per_habitat_scrambles"]
    arm_c_rejection = {
        "rejection_test": SCRAMBLE_POLICY["rejection_test"],
        "counting_context": by_name["C_scrambled"].metadata["counting_context"],
        "edge_convention": by_name["C_scrambled"].metadata["edge_convention"],
        "per_habitat": [
            {
                "habitat": row["habitat"],
                "edge_permutation": row["edge_permutation"],
                "encoded": row["encoded"],
                "is_structure_preserving": row["is_structure_preserving"],
                "preserves_matching_partition_DIAGNOSTIC_ONLY": row[
                    "preserves_matching_partition_DIAGNOSTIC_ONLY"
                ],
                "seed_sha256": row["seed_sha256"],
                "pool_index": row["pool_index"],
            }
            for row in plan
        ],
        "habitats": len(plan),
        "structure_preserving_habitats": sum(
            1 for row in plan if row["is_structure_preserving"]
        ),
        "every_habitat_is_non_automorphic": all(
            not row["is_structure_preserving"] for row in plan
        ),
        "distinct_permutations_chosen": by_name["C_scrambled"].metadata[
            "distinct_permutations_chosen"
        ],
        "matching_partition_preserving_habitats": sum(
            1
            for row in plan
            if row["preserves_matching_partition_DIAGNOSTIC_ONLY"]
        ),
        "why_matching_partition_is_not_the_test": by_name["C_scrambled"].metadata[
            "trap_2_matching_partition"
        ],
        "why_pp_permutation_is_not_a_scramble": by_name["C_scrambled"].metadata[
            "trap_1_pp_permutation"
        ],
        "destruction": per_habitat_destruction(
            dataset, catalogue, arm_codes["C_scrambled"], circuit
        ),
    }

    arm_d_meta = by_name["D_relabeled"].metadata
    arm_d_preservation = {
        "transformations": arm_d_meta["transformations"],
        "per_habitat": arm_d_meta["induced_edge_permutations"]["per_habitat"],
        "every_habitat_is_structure_preserving": arm_d_meta[
            "induced_edge_permutations"
        ]["all_structure_preserving"],
        "all_match_the_affine_prediction": arm_d_meta["induced_edge_permutations"][
            "all_match_the_affine_prediction"
        ],
        "is_bijection_on_the_84_events": arm_d_meta["bijectivity"][
            "is_bijection_on_the_84_events"
        ],
        "mirror_of_arm_c": arm_d_meta["induced_edge_permutations"]["mirror_of_arm_c"],
        "character": arm_d_meta["not_an_ablation"],
    }

    relations = [
        relation_report(dataset, native, circuit, "B_sfp"),
        relation_report(dataset, arm_codes["C_scrambled"], circuit, "C_scrambled"),
        relation_report(dataset, arm_codes["D_relabeled"], circuit, "D_relabeled"),
    ]

    ledger = shape_ledger(arms)
    arm_digests = {row["arm"]: row["token_tensor_sha256"] for row in ledger}

    payload = {
        "module": (
            "Issue 009 Task 6: the four representation arms A / B / C / D "
            "(+ optional diagnostic E)"
        ),
        "scope": (
            "arms only — no model, no training, no baseline, no scorer. This "
            "module does not create or import scorer.py."
        ),
        "interface_contract": {
            "dataclass": (
                "Arm(name, tokens, token_dim, tokens_per_event, exposes, forbids, "
                "metadata) with sha256()"
            ),
            "tokens": (
                "84 entries index-aligned to the task.py catalogue; entry i is a "
                "tuple of tokens_per_event token vectors of plain floats"
            ),
            "builder": "arms.all_arms(dataset) -> tuple[Arm, ...]",
            "pooling": POOLING_CONTRACT,
        },
        "fences": list(FENCES),
        "label_source_fence": LABEL_SOURCE_FENCE,
        "correctness_criterion": CORRECTNESS_CRITERION,
        "provenance": {
            "base_commit": BASE_COMMIT,
            "topographo_version": importlib.metadata.version("topographo"),
            "catalogue_sha256": catalogue.sha256(),
            "dataset_sha256": dataset.sha256(),
            "chart_sha256": chart["sha256"],
            "declared_chart": list(DECLARED_CHART),
            "upstream_pins": {
                "catalogue": pin(
                    EXPECTED_CATALOGUE_SHA256,
                    catalogue.sha256(),
                    "task.Catalogue.sha256()",
                ),
                "dataset": pin(
                    EXPECTED_DATASET_SHA256, dataset.sha256(), "task.Dataset.sha256()"
                ),
                "chart": pin(
                    EXPECTED_CHART_SHA256, chart["sha256"], "sfp.SfpCodec().chart()"
                ),
            },
        },
        "pins": {
            "events": pin(PINS["events"], len(catalogue), "task catalogue"),
            "habitats": pin(
                PINS["habitats"], len(catalogue.events_of_habitat()), "task catalogue"
            ),
            "ordered_pairs": pin(
                PINS["ordered_pairs"], len(dataset), "task dataset pool"
            ),
            "arm_a_token_dim": pin(
                PINS["arm_a_token_dim"],
                by_name["A_native"].token_dim,
                "native ray width",
            ),
            "sfp_token_dim": pin(
                PINS["sfp_token_dim"],
                by_name["B_sfp"].token_dim,
                "S(2) + FFF(7) + PP(4) + pp(3)",
            ),
        },
        "field_layout": [
            {"field": name, "offset": offset, "width": width, "justification": why}
            for name, offset, width, why in FIELD_LAYOUT
        ],
        "shape_ledger": ledger,
        "exposure_table": exposure_table(arms),
        "alignment": alignment_report(arms, catalogue),
        "arm_a_fence": arm_a_no_sfp_field(
            by_name["A_native"], native, arm_codes["D_relabeled"]
        ),
        "arm_c_rejection_proof": arm_c_rejection,
        "arm_d_preservation_proof": arm_d_preservation,
        "marginal_comparison": marginal_comparison(by_name, arm_codes),
        "relation_reports": relations,
        "arms": {arm.name: arm.metadata for arm in arms},
        "token_samples": {arm.name: token_samples(arm) for arm in arms},
        "digests": {
            "arm_token_digests": arm_digests,
            "arm_token_digests_distinct": len(set(arm_digests.values())) == len(arms),
            "manifest": digest(
                {
                    "arms": arm_digests,
                    "catalogue": catalogue.sha256(),
                    "chart": chart["sha256"],
                }
            ),
        },
        "artifact_policy": (
            "CHOICE: digests + per-arm samples + the FULL construction record + a "
            "compact but COMPLETE per-Event code table (arms.<name>.codes for B/C/D "
            "and the catalogue ray encodings for A). The full 84 x tokens tensors "
            "are not spelled out unit by unit — they are a declared deterministic "
            "function of those tables and of the field layout, and each is covered "
            "by its own token_tensor_sha256. Arm E is index one-hot and needs no "
            "table."
        ),
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
            "PASS - four arms frozen, Arm C certified non-automorphic with matched "
            "marginals, Arm D certified relation-preserving"
            if not failures and not broken
            else "FAIL - see count_disagreements / broken_relational_laws"
        ),
    }
    return payload


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _report(payload: dict) -> None:
    for row in payload["shape_ledger"]:
        print(
            f"{row['arm']:<13} token_dim={row['token_dim']:<3} "
            f"tokens/event={row['tokens_per_event']} "
            f"units={row['total_input_units']:<4} "
            f"values={row['value_range']} sha256={row['token_tensor_sha256'][:12]}",
            flush=True,
        )
    rejection = payload["arm_c_rejection_proof"]
    print(
        f"Arm C: {rejection['habitats']} habitats, "
        f"{rejection['structure_preserving_habitats']} structure preserving "
        f"(must be 0); pool 720 total / 24 induced / 696 non-induced",
        flush=True,
    )
    destruction = rejection["destruction"]
    print(
        "Arm C: relation destroyed in "
        f"{destruction['habitats_with_relation_destroyed']}/"
        f"{destruction['habitats']} habitats",
        flush=True,
    )
    preservation = payload["arm_d_preservation_proof"]
    print(
        "Arm D: every habitat structure preserving = "
        f"{preservation['every_habitat_is_structure_preserving']}, bijective = "
        f"{preservation['is_bijection_on_the_84_events']}",
        flush=True,
    )
    for row in payload["relation_reports"]:
        print(
            f"relation {row['arm']:<13} admission {row['admission_agreements']}/"
            f"{row['ordered_pairs_checked']} forced-third "
            f"{row['forced_third_agreements']}/{row['native_admit_pairs']} "
            f"preserved={row['relation_preserved']}",
            flush=True,
        )
    marginals = payload["marginal_comparison"]
    print(
        "Arm B vs Arm C marginals: units "
        f"{marginals['B_equals_C_unit_marginals']['agrees']}, fields "
        f"{marginals['B_equals_C_field_counts']['agrees']}, shape "
        f"{marginals['shape_match']['agrees']}",
        flush=True,
    )
    print(f"manifest sha256: {payload['digests']['manifest']}", flush=True)


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
                f"FAIL: re-derived arms are not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches arms.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

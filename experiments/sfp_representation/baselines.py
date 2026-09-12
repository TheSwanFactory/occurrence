"""Deterministic baselines and the compact-feature leakage search (Issue 009, Task 8).

This module owns the **nonlearned** half of the Issue 009 evidence: every
deterministic selector the learned arms must beat, the exact metric definitions
they are scored under, and the mechanical search for compact exposed features
that determine the label on the proposed folds. It trains nothing, imports no
torch, and re-derives no upstream structure: the catalogue, the 7056 labelled
ordered pairs, the 22 fold manifests and the SFP codec / circuit are all imported
from :mod:`task`, :mod:`folds` and :mod:`sfp` exactly as frozen.

A selector
----------
Every baseline is a **selector**: a total map from a :class:`task.PairRecord` to a
single prediction in ``{Event index 0..83} union {BOTTOM}``. One integer carries
both decisions, because ``task.py`` already encodes the label in the target:
``BOTTOM_INDEX = -1`` means NONADMISSION and any index in ``0..83`` means ADMIT
with that Event as the forced third. So ``prediction == record.target_index`` is
*exactly* "label correct AND (if ADMIT) forced third exact", with no float
proximity and no soft ranking anywhere.

The baselines
-------------
``ExactNativeRelation``
    Label / reference oracle. Recomputes admission and the forced third from
    ``topographo.ssd.fips_basic.admissible`` / ``.third`` and resolves the Event
    identity through ``projective.equivalent``. It must score a perfect 1.0 on
    every bucket; a non-1.0 means the scoring pipeline or the upstream pool is
    broken, and this module stops and reports rather than absorbing it.
``ExactSFPCircuit``
    Representation oracle. ``sfp.ExactSfpCircuit`` over ``sfp.SfpCodec``
    addresses, already certified to agree with the native relation on all 7056
    pairs, so it also scores 1.0. **Its existence is the reason an Arm-B success
    would demonstrate representation SUFFICIENCY / USE and not representation
    DISCOVERY**: the SFP code already contains a nonlearned exact solver, so a
    learner that succeeds on Arm B has been handed a code in which the relation
    is expressible, and the open question is only whether gradient descent uses
    it. It is never a label source.
``TrainPairLookup``
    Memorisation baseline. Memorises ``(a_index, b_index) -> target_index`` from
    the fold's ``train`` bucket. Unseen pairs are *uncovered*; coverage is
    reported separately from accuracy and the declared fallback is BOTTOM.
``TrainEventPairFeatureLookup``
    Compact deterministic feature lookup on a declared key, fitted on train only.
``SameHabitatOnly``
    Admission baseline: ADMIT iff ``habitat_a == habitat_b``. It has no
    forced-third answer at all, so it carries a declared forcing policy and its
    forcing number is reported separately and never mixed into admission.
``RandomCandidate``
    Forcing baseline. Seeded uniform draw over the declared 85-element candidate
    set, reported next to its analytic chance level.
``MajorityClass``
    The honest floor: the train-majority label, always.

Metric definitions
------------------
On any ``(fold, bucket, selector)`` triple::

    joint exact success          mean[ prediction == record.target_index ]
                                 (label correct AND, if ADMIT, target exact)
    admission balanced accuracy  0.5 * (sensitivity + specificity), where
                                 sensitivity = P[pred != BOTTOM | ADMIT] and
                                 specificity = P[pred == BOTTOM | NONADMISSION];
                                 null when either class is empty in the bucket
    forced third exact accuracy  mean[ prediction == target_index ] over the
                                 ADMIT subset only, exact Event identity
    per-class non-admission acc  mean[ prediction == BOTTOM ] separately for
                                 REPEATED / SAME_HABITAT_DISJOINT / CROSS_HABITAT
    coverage                     fraction of bucket items the selector can answer
                                 at all (a lookup miss is uncovered, and the
                                 declared fallback is still scored)

Raw admission accuracy is **inadmissible**: CROSS_HABITAT is ``6552/7056 =
92.9%`` of the pool, so a constant "no" scores 0.929 while knowing nothing.
Admission is therefore always reported as balanced accuracy.

THE COMPACT-FEATURE LEAKAGE SEARCH — and the distinction that matters
---------------------------------------------------------------------
Before any training, this module searches a **declared, enumerated** family of
compact features over the pair's *task-level coordinates* (never over an arm's
code) for one that determines the label on a fold's test buckets. The family, the
subset size and the ceiling thresholds are all fixed in advance and recorded.

The search will find ``shared_block_count`` and ``block_id_equality``. That is
expected and it is **not** leakage: "the two Events share exactly one certified
cyclic block" *is* the admission rule. Such a feature is a CEILING — it is the
very thing the arms are asked to represent — and it belongs with the oracles. The
finding that would matter is different: a feature strictly **cheaper than the
relation** (habitat / Fano point / sign coincidences, which carry no block
incidence at all) that still determines the label on test. That would be a
genuine shortcut and would make the fold non-dispositive. Findings are therefore
partitioned into ``relation_equivalent`` and ``cheaper_than_the_relation`` and the
second partition drives the verdict.

Note the asymmetry the search exposes: even the relation-equivalent atoms
determine only *admission*. Neither ``shared_block_count`` nor
``block_id_equality`` names *which* block is shared, so neither determines the
forced third. Forcing is where a representation has to do real work.

Fences
------
::

    A feature equivalent to the certified relation is a CEILING, not a shortcut.
    A feature cheaper than the relation that determines the label is a SHORTCUT and
    makes the fold non-dispositive; report it plainly.
    algebraic zero != NONADMISSION;   0 != bottom
    Admission must be reported as balanced accuracy.
    ExactSFPCircuit is a reference oracle, never a label source.
    Arm B containing a label-determining feature is by design, not leakage.

Scope
-----
Baselines, metrics and the leakage search only. No training loop, no torch, no
sweep driver, no harness: ``harness.py`` is owned by a parallel task and is
deliberately not imported here. ``conformance.py``, ``sfp.py``, ``groups.py``,
``task.py``, ``folds.py``, ``arms.py`` and ``scorer.py`` are read-only upstream.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import arms
import folds as folds_module
from folds import Fold, all_folds, structural_folds
from sfp import ExactSfpCircuit, SfpCodec
from task import (
    BOTTOM_INDEX,
    BOTTOM_SYMBOL,
    Catalogue,
    Dataset,
    Label,
    NonadmissionClass,
    PairRecord,
    build_dataset,
    digest,
)

from topographo.ssd import fips_basic, projective

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "baselines.json"

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"

EXPECTED_CATALOGUE_SHA256 = (
    "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097"
)
EXPECTED_DATASET_SHA256 = (
    "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721"
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "27b01b5e7440c2449e28d8e5b3f82addba1028a1640707752ab01215a938a6ee"
)

FENCES = (
    "A feature equivalent to the certified relation is a CEILING, not a shortcut.",
    (
        "A feature cheaper than the relation that determines the label is a "
        "SHORTCUT and makes the fold non-dispositive; report it plainly."
    ),
    "algebraic zero != NONADMISSION;   0 != bottom",
    "Admission must be reported as balanced accuracy.",
    "ExactSFPCircuit is a reference oracle, never a label source.",
    "Arm B containing a label-determining feature is by design, not leakage.",
)

POOL_PINS = {
    "events": 84,
    "habitats": 14,
    "fano_points": 7,
    "ordered_pairs_total": 7056,
    "admit": 336,
    "repeated": 84,
    "same_habitat_disjoint": 84,
    "cross_habitat": 6552,
    "folds_total": 22,
    "folds_structural": 21,
}

# --- determinism policy -----------------------------------------------------
ROUND_DIGITS = 12
ROUNDING_POLICY = (
    f"every reported float is passed through round(x, {ROUND_DIGITS}) before "
    "serialisation, so the artifact is byte-identical on replay; the rounding is "
    "cosmetic only and no comparison, threshold or verdict is computed from a "
    "rounded value. Correctness itself is integer index equality and is never a "
    "float at all."
)

BASE_SEED = 90050803
SEED_DERIVATION = (
    "random.Random(int(task.digest('baselines/v1|<base_seed>|<label>')[:16], 16)); "
    "the derived integer is recorded per draw. No set/dict iteration order and no "
    "hash() value participates, so the artifact is invariant under PYTHONHASHSEED."
)

CANDIDATE_SET_SIZE = POOL_PINS["events"] + 1
CANDIDATE_SET_DECLARATION = (
    "the 85-element set {Event index 0..83} union {BOTTOM_INDEX = -1}; BOTTOM is "
    "an explicit distinct symbol and is NOT Event index 0, NOT FFF=000, NOT pp=00 "
    "and NOT any algebraic zero"
)

CLASS_NAMES = (
    NonadmissionClass.REPEATED.value,
    NonadmissionClass.SAME_HABITAT_DISJOINT.value,
    NonadmissionClass.CROSS_HABITAT.value,
)

DERIVED_BUCKET = "test_admission_balanced"
DERIVED_BUCKET_STATEMENT = (
    "a DERIVED bucket, composed here and not a fold bucket: the union of the "
    "fold's held-out test bucket with its digest-pinned balanced cross-habitat "
    "sample. It exists because neither part alone supports a balanced admission "
    "number — LOHO test_within holds no CROSS_HABITAT pair and the balanced "
    "cross-habitat sample holds no ADMIT pair — while their union holds both. "
    "folds.py is not modified; the union is taken over its frozen index lists."
)

METRIC_DEFINITIONS = {
    "joint_exact_success": (
        "mean[ prediction == record.target_index ] over the bucket. Because "
        "task.py encodes NONADMISSION as target_index = BOTTOM_INDEX = -1 and "
        "ADMIT as the certified forced-third Event index, this single integer "
        "equality is exactly 'label correct AND (if ADMIT) forced third exact'."
    ),
    "admission_balanced_accuracy": (
        "0.5 * (sensitivity + specificity) with sensitivity = "
        "P[prediction != BOTTOM | ADMIT] and specificity = "
        "P[prediction == BOTTOM | NONADMISSION]; null when the bucket contains no "
        "ADMIT pair or no NONADMISSION pair. Raw admission accuracy is "
        "inadmissible: CROSS_HABITAT is 92.9% of the pool."
    ),
    "forced_third_exact_accuracy": (
        "mean[ prediction == record.target_index ] over the ADMIT subset of the "
        "bucket only; exact certified Event identity, never float proximity. Null "
        "when the bucket contains no ADMIT pair."
    ),
    "per_class_nonadmission_accuracy": (
        "mean[ prediction == BOTTOM ] computed separately for REPEATED, "
        "SAME_HABITAT_DISJOINT and CROSS_HABITAT; null for a class absent from "
        "the bucket."
    ),
    "coverage": (
        "fraction of bucket items the selector can answer at all. A lookup miss "
        "is uncovered; its declared fallback is still scored, so accuracy and "
        "coverage are reported side by side and never conflated."
    ),
    "correctness_criterion": (
        "exact certified Event identity by integer index equality against "
        "record.target_index, which task.py verified via "
        "topographo.ssd.projective.equivalent. Never float proximity, never a "
        "soft ranking."
    ),
}


# ---------------------------------------------------------------------------
# deterministic helpers
# ---------------------------------------------------------------------------

def derive_seed(label: str) -> int:
    """Deterministic integer seed for one named draw. Never ``hash()``."""

    return int(digest(f"baselines/v1|{BASE_SEED}|{label}")[:16], 16)


def rd(value: float | None) -> float | None:
    """Round for byte-identical replay. Cosmetic only; never fed to a comparison."""

    return None if value is None else round(value, ROUND_DIGITS)


def ratio(numerator: int, denominator: int) -> float | None:
    """Exact ratio, or ``None`` for an empty denominator. Unrounded."""

    return None if denominator == 0 else numerator / denominator


def pin(expected: object, observed: object, statement: str) -> dict:
    """Record an expected/observed pair with an explicit agreement flag."""

    return {
        "expected": expected,
        "observed": observed,
        "agrees": expected == observed,
        "statement": statement,
    }


# ---------------------------------------------------------------------------
# the prediction type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    """One selector answer: an Event index in ``0..83``, or ``BOTTOM_INDEX``.

    ``answered`` is False when the selector had no applicable entry and fell back
    to its declared default. The fallback is still scored; ``answered`` only feeds
    the separately reported coverage.
    """

    target: int
    answered: bool

    @property
    def says_admit(self) -> bool:
        return self.target != BOTTOM_INDEX


ANSWERED_BOTTOM = Prediction(BOTTOM_INDEX, True)
UNANSWERED_BOTTOM = Prediction(BOTTOM_INDEX, False)


# ---------------------------------------------------------------------------
# the two oracles, precomputed once over all 7056 ordered pairs
# ---------------------------------------------------------------------------

def native_relation_table(catalogue: Catalogue) -> tuple[int, ...]:
    """The native FIPS answer for every ordered pair, row-major at ``a*84+b``.

    Recomputed from ``fips_basic.admissible`` / ``fips_basic.third`` and resolved
    to a catalogue index through ``projective.equivalent``. This is the label
    oracle; it is also the pipeline self-test, since it must score 1.0 everywhere.
    """

    n = len(catalogue)
    out: list[int] = []
    for a in range(n):
        left = catalogue.event(a)
        for b in range(n):
            right = catalogue.event(b)
            if not fips_basic.admissible(left, right):
                out.append(BOTTOM_INDEX)
                continue
            third = fips_basic.third(left, right)
            index = catalogue.index(third)
            # Exact certified Event identity; never float proximity.
            if not projective.equivalent(third, catalogue.event(index)):
                raise AssertionError(
                    f"native third for ({a}, {b}) failed exact projective identity"
                )
            out.append(index)
    return tuple(out)


def sfp_circuit_table(
    catalogue: Catalogue, codec: SfpCodec, circuit: ExactSfpCircuit
) -> tuple[int, ...]:
    """The nonlearned SFP circuit's answer for every ordered pair, row-major.

    No ``a == b`` special case is inserted: the circuit is allowed to speak for
    itself on repeated pairs so that a disagreement with the native relation
    would surface instead of being masked.
    """

    codes = tuple(codec.encode(event) for event in catalogue.events)
    inverse = {code.as_key(): index for index, code in enumerate(codes)}
    if len(inverse) != len(catalogue):
        raise AssertionError("SFP code assignment is not injective on the catalogue")

    n = len(catalogue)
    out: list[int] = []
    for a in range(n):
        for b in range(n):
            if not circuit.admit_event(codes[a], codes[b]):
                out.append(BOTTOM_INDEX)
                continue
            forced = circuit.forced_third_event(codes[a], codes[b])
            if forced is None:
                raise AssertionError(
                    f"circuit admitted ({a}, {b}) but produced no forced third"
                )
            out.append(inverse[forced.as_key()])
    return tuple(out)


# ---------------------------------------------------------------------------
# THE DECLARED COMPACT-FEATURE FAMILY
# ---------------------------------------------------------------------------
#
# Every atom is a function of the pair's TASK-LEVEL coordinates only: the fields
# task.PairRecord already carries, plus catalogue block incidence. No atom reads
# an arm's token tensor, an arm's code, or any model output.
#
# ``cost`` partitions the family in advance, before any number is measured:
#   "cheaper_than_the_relation" — habitat / Fano point / sign / identity
#       coincidences. These carry NO block incidence, so they cannot express the
#       admission rule. A cheap atom that determines the label on test is a
#       genuine SHORTCUT.
#   "relation_equivalent" — block incidence. "share exactly one certified cyclic
#       block" IS the admission rule, so these are CEILINGS, not shortcuts.

CHEAP = "cheaper_than_the_relation"
RELATION_EQUIVALENT = "relation_equivalent"


def _same_sign(catalogue: Catalogue, record: PairRecord) -> object:
    return record.sign_a == record.sign_b


def _same_fano_point(catalogue: Catalogue, record: PairRecord) -> object:
    return record.fano_point_a == record.fano_point_b


def _same_habitat(catalogue: Catalogue, record: PairRecord) -> object:
    return record.habitat_a == record.habitat_b


def _repeated(catalogue: Catalogue, record: PairRecord) -> object:
    return record.a_index == record.b_index


def _shared_block_count(catalogue: Catalogue, record: PairRecord) -> object:
    return len(catalogue.shared_blocks(record.a_index, record.b_index))


def _block_id_equality(catalogue: Catalogue, record: PairRecord) -> object:
    """The 4-bit structural equality pattern among the 2+2 incident block ids."""

    left = catalogue.block_ids[record.a_index]
    right = catalogue.block_ids[record.b_index]
    return tuple(x == y for x in left for y in right)


def _fano_point_pair(catalogue: Catalogue, record: PairRecord) -> object:
    return tuple(sorted((record.fano_point_a, record.fano_point_b)))


def _sign_pair(catalogue: Catalogue, record: PairRecord) -> object:
    return tuple(sorted((record.sign_a, record.sign_b)))


def _habitat_pair(catalogue: Catalogue, record: PairRecord) -> object:
    return tuple(sorted((record.habitat_a, record.habitat_b)))


FEATURE_ATOMS: tuple[tuple[str, str, str, object], ...] = (
    (
        "same_sign",
        CHEAP,
        "sign_a == sign_b (the two habitat deltas coincide)",
        _same_sign,
    ),
    (
        "same_fano_point",
        CHEAP,
        "fano_point_a == fano_point_b",
        _same_fano_point,
    ),
    (
        "same_habitat",
        CHEAP,
        "habitat_a == habitat_b, i.e. the full (P, delta) pair coincides",
        _same_habitat,
    ),
    (
        "repeated",
        CHEAP,
        "a_index == b_index",
        _repeated,
    ),
    (
        "shared_block_count",
        RELATION_EQUIVALENT,
        (
            "len(catalogue.shared_blocks(a, b)), observed range {0, 1}. This IS "
            "the admission rule, so it is a ceiling and not a shortcut."
        ),
        _shared_block_count,
    ),
    (
        "block_id_equality",
        RELATION_EQUIVALENT,
        (
            "the 4-bit equality pattern (a0==b0, a0==b1, a1==b0, a1==b1) over the "
            "2+2 incident certified block ids. Its popcount is block incidence, so "
            "it is relation-equivalent by construction and is a ceiling."
        ),
        _block_id_equality,
    ),
    (
        "fano_point_pair",
        CHEAP,
        "the unordered pair {fano_point_a, fano_point_b} as a sorted 2-tuple",
        _fano_point_pair,
    ),
    (
        "sign_pair",
        CHEAP,
        "the unordered pair {sign_a, sign_b} as a sorted 2-tuple",
        _sign_pair,
    ),
    (
        "habitat_pair",
        CHEAP,
        (
            "declared EXTENSION beyond the coordinator's suggested atoms: the "
            "unordered pair {habitat_a, habitat_b}. It is the most expressive "
            "purely-habitat feature available, so including it makes the "
            "cheaper-than-the-relation partition maximally adversarial."
        ),
        _habitat_pair,
    ),
)

ATOM_NAMES = tuple(name for name, _, _, _ in FEATURE_ATOMS)
ATOM_COST = {name: cost for name, cost, _, _ in FEATURE_ATOMS}

MAX_SUBSET_SIZE = 3

# A subset is at ceiling when it reaches this on a test bucket. Declared before
# any number was measured; never tuned against an observation.
ADMISSION_CEILING_THRESHOLD = 0.99
JOINT_CEILING_THRESHOLD = 0.99

# The declared key of the named TrainEventPairFeatureLookup baseline. Block
# incidence is deliberately EXCLUDED: including it would make the baseline a
# restatement of the relation rather than a test of whether the cheap exposed
# coordinates suffice. Block incidence is covered by the leakage search instead,
# where it is partitioned as relation_equivalent.
FEATURE_LOOKUP_KEY = ("repeated", "same_habitat", "same_fano_point", "same_sign")


# ---------------------------------------------------------------------------
# precomputed feature space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpace:
    """Every declared atom evaluated once on all 7056 positions, as small ints.

    Atom values are mapped to integer codes through a vocabulary built in
    ``sorted(repr(value))`` order, so the encoding never depends on set / dict
    iteration order or on ``hash()``. A feature subset key is then a single
    integer in mixed radix, which keeps the search over 129 subsets * 22 folds
    inside a couple of seconds.
    """

    codes: tuple[tuple[int, ...], ...]
    cardinalities: tuple[int, ...]
    vocabularies: tuple[tuple[str, ...], ...]
    index_of: dict[str, int]

    def subset_keys(
        self, subset: tuple[str, ...], positions: tuple[int, ...]
    ) -> list[int]:
        """Mixed-radix integer key of ``subset`` at each of ``positions``."""

        columns = [self.codes[self.index_of[name]] for name in subset]
        radices = [self.cardinalities[self.index_of[name]] for name in subset]
        if len(columns) == 1:
            column = columns[0]
            return [column[p] for p in positions]
        if len(columns) == 2:
            first, second = columns
            radix = radices[1]
            return [first[p] * radix + second[p] for p in positions]
        if len(columns) == 3:
            first, second, third = columns
            radix_b, radix_c = radices[1], radices[2]
            return [
                (first[p] * radix_b + second[p]) * radix_c + third[p]
                for p in positions
            ]
        out: list[int] = []
        for position in positions:
            key = 0
            for column, radix in zip(columns, radices):
                key = key * radix + column[position]
            out.append(key)
        return out


def build_feature_space(dataset: Dataset) -> FeatureSpace:
    """Evaluate the declared atom family on the frozen pool."""

    catalogue = dataset.catalogue
    columns: list[tuple[int, ...]] = []
    cardinalities: list[int] = []
    vocabularies: list[tuple[str, ...]] = []
    for _, _, _, function in FEATURE_ATOMS:
        raw = [function(catalogue, record) for record in dataset.records]
        vocabulary = tuple(sorted({repr(value) for value in raw}))
        code_of = {text: i for i, text in enumerate(vocabulary)}
        columns.append(tuple(code_of[repr(value)] for value in raw))
        cardinalities.append(len(vocabulary))
        vocabularies.append(vocabulary)
    return FeatureSpace(
        codes=tuple(columns),
        cardinalities=tuple(cardinalities),
        vocabularies=tuple(vocabularies),
        index_of={name: i for i, name in enumerate(ATOM_NAMES)},
    )


# ---------------------------------------------------------------------------
# the metric suite
# ---------------------------------------------------------------------------

def score(
    dataset: Dataset, positions: tuple[int, ...], predictions: list[Prediction]
) -> dict:
    """Every declared metric for one ``(bucket, selector)`` pair."""

    if len(positions) != len(predictions):
        raise AssertionError("prediction list is not aligned to the bucket")

    total = len(positions)
    answered = 0
    joint = 0
    admit_total = 0
    admit_said_admit = 0
    admit_exact = 0
    nonadmit_total = 0
    nonadmit_said_bottom = 0
    per_class = {name: [0, 0] for name in CLASS_NAMES}

    for position, prediction in zip(positions, predictions):
        record = dataset.records[position]
        answered += prediction.answered
        exact = prediction.target == record.target_index
        joint += exact
        if record.admitted:
            admit_total += 1
            admit_said_admit += prediction.says_admit
            admit_exact += exact
        else:
            nonadmit_total += 1
            said_bottom = not prediction.says_admit
            nonadmit_said_bottom += said_bottom
            row = per_class[record.nonadmission_class.value]
            row[0] += 1
            row[1] += said_bottom

    sensitivity = ratio(admit_said_admit, admit_total)
    specificity = ratio(nonadmit_said_bottom, nonadmit_total)
    balanced = (
        None
        if sensitivity is None or specificity is None
        else 0.5 * (sensitivity + specificity)
    )
    return {
        "size": total,
        "admit": admit_total,
        "nonadmission": nonadmit_total,
        "joint_exact_success": rd(ratio(joint, total)),
        "joint_exact_hits": joint,
        "admission_sensitivity": rd(sensitivity),
        "admission_specificity": rd(specificity),
        "admission_balanced_accuracy": rd(balanced),
        "forced_third_exact_accuracy": rd(ratio(admit_exact, admit_total)),
        "forced_third_exact_hits": admit_exact,
        "per_class_nonadmission_accuracy": {
            name: rd(ratio(hits, seen)) for name, (seen, hits) in per_class.items()
        },
        "per_class_nonadmission_size": {
            name: seen for name, (seen, _) in per_class.items()
        },
        "per_class_nonadmission_hits": {
            name: hits for name, (_, hits) in per_class.items()
        },
        "coverage": rd(ratio(answered, total)),
        "covered": answered,
        "uncovered": total - answered,
    }


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Context:
    """Everything a selector needs, built once and shared across all folds."""

    dataset: Dataset
    native: tuple[int, ...]
    circuit: tuple[int, ...]
    space: FeatureSpace
    habitat_members: dict[tuple[int, int], tuple[int, ...]]


class Selector:
    """A total map from a ``PairRecord`` to one prediction. Fitted on train only."""

    name = "Selector"
    kind = "abstract"
    nonlearned = True

    def declaration(self) -> dict:
        raise NotImplementedError

    def fit(self, context: Context, fold: Fold) -> None:
        """Default: nothing to fit."""

    def predict(self, context: Context, position: int) -> Prediction:
        raise NotImplementedError

    def predict_bucket(
        self, context: Context, positions: tuple[int, ...]
    ) -> list[Prediction]:
        return [self.predict(context, position) for position in positions]


class ExactNativeRelation(Selector):
    """The label / reference oracle. Must be 1.0 on every bucket of every fold."""

    name = "ExactNativeRelation"
    kind = "oracle_label_reference"

    def declaration(self) -> dict:
        return {
            "role": "nonlearned label / reference oracle",
            "source": (
                "topographo.ssd.fips_basic.admissible and .third, resolved to a "
                "catalogue index through topographo.ssd.projective.equivalent"
            ),
            "coverage_policy": "total; every ordered pair has a native answer",
            "forcing_policy": "the certified native third",
            "expected": (
                "joint exact success = 1.0 on every bucket of every fold. Anything "
                "else means the scoring pipeline or the upstream pool is broken, "
                "and this module stops and reports instead of absorbing it."
            ),
        }

    def predict(self, context: Context, position: int) -> Prediction:
        return Prediction(context.native[position], True)


class ExactSFPCircuit(Selector):
    """The representation oracle: ``sfp.ExactSfpCircuit``. Never a label source."""

    name = "ExactSFPCircuit"
    kind = "oracle_representation"

    def declaration(self) -> dict:
        return {
            "role": "nonlearned representation oracle",
            "source": (
                "sfp.ExactSfpCircuit.admit_event / .forced_third_event over "
                "sfp.SfpCodec addresses; endpoint intersection and displacement "
                "XOR only, no pair lookup table"
            ),
            "coverage_policy": "total",
            "forcing_policy": "the circuit's forced third, decoded to an index",
            "never_a_label_source": (
                "every comparison here is against task.py's records, which come "
                "from the certified native FIPS relation"
            ),
            "why_it_matters": (
                "a nonlearned exact solver already exists inside the SFP code, so "
                "an Arm-B success would demonstrate representation SUFFICIENCY / "
                "USE, not representation DISCOVERY: the learner is handed a code "
                "in which the relation is expressible and the only open question "
                "is whether gradient descent uses it"
            ),
            "expected": "joint exact success = 1.0 on every bucket of every fold",
        }

    def predict(self, context: Context, position: int) -> Prediction:
        return Prediction(context.circuit[position], True)


class TrainPairLookup(Selector):
    """Memorisation floor: ``(a_index, b_index) -> target_index`` from train."""

    name = "TrainPairLookup"
    kind = "memorization"

    def __init__(self) -> None:
        self.table: dict[int, int] = {}

    def declaration(self) -> dict:
        return {
            "role": "memorisation baseline",
            "key": "(a_index, b_index), the exact ordered pair identity",
            "fitted_on": "the fold's 'train' bucket only",
            "coverage_policy": (
                "a test pair absent from train is UNCOVERED; coverage is reported "
                "separately from accuracy and is never folded into it"
            ),
            "declared_default": (
                f"BOTTOM ({BOTTOM_SYMBOL}, target_index = {BOTTOM_INDEX}) — the "
                "pool-majority label, since NONADMISSION is 6720/7056 of the pool"
            ),
            "forcing_policy": "the memorised target; BOTTOM on a miss",
            "expected": (
                "coverage exactly 0 on LOHO test_within (0/36) and LOFPO "
                "test_within (0/144): both Events of every within-habitat test "
                "pair are held out, so no such pair can occur in train"
            ),
        }

    def fit(self, context: Context, fold: Fold) -> None:
        self.table = {
            position: context.dataset.records[position].target_index
            for position in fold.train
        }

    def predict(self, context: Context, position: int) -> Prediction:
        if position in self.table:
            return Prediction(self.table[position], True)
        return UNANSWERED_BOTTOM


class FeatureLookup(Selector):
    """A deterministic ``feature key -> (label, target)`` map fitted on train.

    Used both for the named ``TrainEventPairFeatureLookup`` baseline and, with a
    different subset, for every candidate in the leakage search — one code path,
    so the search cannot accidentally be scored under different rules.
    """

    kind = "feature_lookup"

    def __init__(self, subset: tuple[str, ...], name: str | None = None) -> None:
        self.subset = subset
        self.name = name or "FeatureLookup(" + "+".join(subset) + ")"
        self.label_map: dict[int, int] = {}
        self.train_purity: float | None = None
        self.unanimous_keys: int = 0
        self.keys_fitted: int = 0

    @property
    def cost_class(self) -> str:
        if any(ATOM_COST[name] == RELATION_EQUIVALENT for name in self.subset):
            return RELATION_EQUIVALENT
        return CHEAP

    def declaration(self) -> dict:
        return {
            "role": "compact deterministic feature lookup",
            "key": list(self.subset),
            "key_statement": (
                "the declared compact feature tuple over the pair's task-level "
                "coordinates; evaluated by FEATURE_ATOMS and nothing else"
            ),
            "cost_class": self.cost_class,
            "fitted_on": "the fold's 'train' bucket only",
            "fitting_rule": (
                "per key: the train-majority label, ties broken to NONADMISSION; "
                "when the majority label is ADMIT the target is the train-majority "
                "forced third for that key, ties broken to the smallest Event index"
            ),
            "coverage_policy": "a key unseen in train is UNCOVERED",
            "declared_default": (
                f"BOTTOM ({BOTTOM_SYMBOL}, target_index = {BOTTOM_INDEX})"
            ),
        }

    def fit(self, context: Context, fold: Fold) -> None:
        train = fold.train
        keys = context.space.subset_keys(self.subset, train)
        total: Counter[int] = Counter()
        admits: Counter[int] = Counter()
        targets: dict[int, Counter[int]] = {}
        records = context.dataset.records
        for key, position in zip(keys, train):
            record = records[position]
            total[key] += 1
            if record.admitted:
                admits[key] += 1
                targets.setdefault(key, Counter())[record.target_index] += 1

        label_map: dict[int, int] = {}
        unanimous = 0
        agree = 0
        for key, seen in total.items():
            admit_count = admits[key]
            if admit_count * 2 > seen:
                bucket = targets[key]
                best = min(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                label_map[key] = best
                agree += admit_count
                if len(bucket) == 1:
                    unanimous += 1
            else:
                label_map[key] = BOTTOM_INDEX
                agree += seen - admit_count
        self.label_map = label_map
        self.keys_fitted = len(label_map)
        self.unanimous_keys = unanimous
        self.train_purity = ratio(agree, len(train))

    def predict_bucket(
        self, context: Context, positions: tuple[int, ...]
    ) -> list[Prediction]:
        keys = context.space.subset_keys(self.subset, positions)
        out: list[Prediction] = []
        for key in keys:
            if key in self.label_map:
                out.append(Prediction(self.label_map[key], True))
            else:
                out.append(UNANSWERED_BOTTOM)
        return out

    def predict(self, context: Context, position: int) -> Prediction:
        return self.predict_bucket(context, (position,))[0]


class TrainEventPairFeatureLookup(FeatureLookup):
    """The named compact feature lookup on the declared cheap-coordinate key."""

    name = "TrainEventPairFeatureLookup"

    def __init__(self) -> None:
        super().__init__(FEATURE_LOOKUP_KEY, name="TrainEventPairFeatureLookup")

    def declaration(self) -> dict:
        body = super().declaration()
        body["block_incidence_excluded_on_purpose"] = (
            "the key carries no block incidence. Including shared_block_count "
            "would make this baseline a restatement of the admission rule rather "
            "than a test of whether the cheap exposed coordinates suffice. Block "
            "incidence is covered by the leakage search, partitioned as "
            "relation_equivalent."
        )
        return body


class SameHabitatOnly(Selector):
    """Admission baseline: ADMIT iff ``habitat_a == habitat_b``. No real forcing."""

    name = "SameHabitatOnly"
    kind = "admission_baseline"

    def declaration(self) -> dict:
        return {
            "role": "admission baseline; it decides admission and nothing else",
            "rule": "predict ADMIT iff habitat_a == habitat_b, i.e. same (P, delta)",
            "repeated_pairs": (
                "NOT excluded: a repeated pair is same-habitat, so this selector "
                "calls it ADMIT. That is the whole point of the baseline."
            ),
            "coverage_policy": "total",
            "forcing_policy": (
                "declared and deterministic: the smallest Event index of habitat_a "
                "other than a and b. It is a non-relational stand-in — the "
                "selector has no forced-third answer at all — so its forcing "
                "number is reported separately and never mixed into admission. "
                "Analytic chance under this policy on a same-habitat ADMIT pair is "
                "1/4 (6 habitat members less a and b)."
            ),
            "expected": (
                "on LOHO test_within it predicts ADMIT for all 36 pairs, because "
                "every within-habitat pair is same-habitat: sensitivity 1.0, "
                "specificity 0.0, admission balanced accuracy exactly 0.5"
            ),
            "why_this_is_the_important_baseline": (
                "0.5 is chance. The LOHO test_within bucket therefore cannot be won "
                "by a habitat-matching shortcut, which is precisely the property "
                "that makes the primary fold family discriminative."
            ),
        }

    def predict(self, context: Context, position: int) -> Prediction:
        record = context.dataset.records[position]
        if record.habitat_a != record.habitat_b:
            return ANSWERED_BOTTOM
        members = context.habitat_members[record.habitat_a]
        for candidate in members:
            if candidate != record.a_index and candidate != record.b_index:
                return Prediction(candidate, True)
        raise AssertionError("a habitat held fewer than three Events")


class RandomCandidate(Selector):
    """Seeded uniform draw over the declared 85-element candidate set."""

    name = "RandomCandidate"
    kind = "chance_baseline"

    def __init__(self) -> None:
        self.table: tuple[int, ...] = ()
        self.seed: int | None = None
        self.seed_label: str | None = None

    def declaration(self) -> dict:
        return {
            "role": "forcing baseline / chance level",
            "candidate_set": CANDIDATE_SET_DECLARATION,
            "candidate_set_size": CANDIDATE_SET_SIZE,
            "chance_convention": (
                "1/85, i.e. BOTTOM is INCLUDED in the candidate set. Declared in "
                "advance. The alternative convention (1/84 over Events only) is "
                "not used here, because a selector that could never answer BOTTOM "
                "would score 0 on 6720 of the 7056 pairs and would not be a "
                "chance level for the joint metric at all."
            ),
            "draw": (
                "one draw per position 0..7055 per fold, consumed in ascending "
                "position order from a single random.Random, so every bucket and "
                "every derived union of buckets reads one and the same table"
            ),
            "seed_derivation": SEED_DERIVATION,
            "analytic_chance": {
                "joint_exact_success": rd(1.0 / CANDIDATE_SET_SIZE),
                "forced_third_exact_accuracy": rd(1.0 / CANDIDATE_SET_SIZE),
                "admission_balanced_accuracy": 0.5,
                "admission_balanced_accuracy_derivation": (
                    "sensitivity = 84/85 (any Event index says ADMIT), specificity "
                    "= 1/85 (only BOTTOM says NONADMISSION), so the balanced mean "
                    "is 0.5 * (84/85 + 1/85) = 0.5 exactly"
                ),
                "note": (
                    "the empirical value is a finite-sample draw and is NOT "
                    "asserted equal to the analytic value; both are reported"
                ),
            },
        }

    def fit(self, context: Context, fold: Fold) -> None:
        self.seed_label = f"RandomCandidate|{fold.family}|{fold.name}"
        self.seed = derive_seed(self.seed_label)
        rng = random.Random(self.seed)
        self.table = tuple(
            rng.randrange(CANDIDATE_SET_SIZE) - 1
            for _ in range(len(context.dataset.records))
        )

    def predict(self, context: Context, position: int) -> Prediction:
        return Prediction(self.table[position], True)


class MajorityClass(Selector):
    """The honest floor: always the train-majority label."""

    name = "MajorityClass"
    kind = "floor"

    def __init__(self) -> None:
        self.majority_label = Label.NONADMISSION
        self.prediction = ANSWERED_BOTTOM
        self.train_share: float | None = None

    def declaration(self) -> dict:
        return {
            "role": "honest floor",
            "rule": "predict the train-majority label, always",
            "coverage_policy": "total",
            "forcing_policy": (
                "if the train majority were ADMIT the target would be the "
                "train-majority forced third; on every fold of this pool the "
                "majority is NONADMISSION, so the emitted answer is BOTTOM"
            ),
            "expected": (
                "admission balanced accuracy exactly 0.5 wherever both classes are "
                "present (sensitivity 0, specificity 1), and joint exact success "
                "equal to the NONADMISSION share of the bucket. This is why raw "
                "admission accuracy is inadmissible."
            ),
        }

    def fit(self, context: Context, fold: Fold) -> None:
        records = context.dataset.records
        admits = sum(1 for position in fold.train if records[position].admitted)
        self.train_share = ratio(admits, len(fold.train))
        if admits * 2 > len(fold.train):
            self.majority_label = Label.ADMIT
            targets = Counter(
                records[position].target_index
                for position in fold.train
                if records[position].admitted
            )
            best = min(targets.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            self.prediction = Prediction(best, True)
        else:
            self.majority_label = Label.NONADMISSION
            self.prediction = ANSWERED_BOTTOM

    def predict(self, context: Context, position: int) -> Prediction:
        return self.prediction


SELECTOR_ORDER = (
    "ExactNativeRelation",
    "ExactSFPCircuit",
    "TrainPairLookup",
    "TrainEventPairFeatureLookup",
    "SameHabitatOnly",
    "RandomCandidate",
    "MajorityClass",
)


def build_selectors() -> tuple[Selector, ...]:
    """A fresh selector set, in the declared reporting order."""

    built = (
        ExactNativeRelation(),
        ExactSFPCircuit(),
        TrainPairLookup(),
        TrainEventPairFeatureLookup(),
        SameHabitatOnly(),
        RandomCandidate(),
        MajorityClass(),
    )
    if tuple(selector.name for selector in built) != SELECTOR_ORDER:
        raise AssertionError("selector order diverged from the declared order")
    return built


# ---------------------------------------------------------------------------
# buckets
# ---------------------------------------------------------------------------

def held_out_test_bucket(fold: Fold) -> str:
    """The fold's held-out test bucket name. Never guessed from the family."""

    names = fold.bucket_names()
    for candidate in ("test_within", "test_random_holdout"):
        if candidate in names:
            return candidate
    raise AssertionError(f"fold {fold.name!r} exposes no held-out test bucket")


def balanced_test_positions(fold: Fold) -> tuple[int, ...]:
    """The fold's digest-pinned balanced cross-habitat admission diagnostic."""

    return fold.sample(f"{fold.family}/{fold.name}/test_cross_habitat").positions


def evaluation_buckets(fold: Fold) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """``(name, ascending positions, provenance)`` for every scored bucket."""

    held_out = held_out_test_bucket(fold)
    balanced = balanced_test_positions(fold)
    rows: list[tuple[str, tuple[int, ...], str]] = [
        ("train", tuple(sorted(fold.train)), "folds.py bucket 'train'"),
        (
            held_out,
            tuple(sorted(fold.bucket(held_out))),
            f"folds.py bucket {held_out!r}",
        ),
    ]
    if "straddle" in fold.bucket_names():
        rows.append(
            (
                "straddle",
                tuple(sorted(fold.straddle)),
                "folds.py bucket 'straddle' (carried whole, never discarded)",
            )
        )
    rows.append(
        (
            "test_cross_habitat",
            tuple(sorted(balanced)),
            "folds.py balanced sample '<FAMILY>/<fold>/test_cross_habitat'",
        )
    )
    union = tuple(sorted(set(fold.bucket(held_out)) | set(balanced)))
    rows.append((DERIVED_BUCKET, union, DERIVED_BUCKET_STATEMENT))
    return tuple(rows)


# ---------------------------------------------------------------------------
# the baseline table
# ---------------------------------------------------------------------------

def evaluate_fold(context: Context, fold: Fold) -> dict:
    """Every selector on every bucket of one fold."""

    selectors = build_selectors()
    for selector in selectors:
        selector.fit(context, fold)

    buckets = evaluation_buckets(fold)
    body: dict[str, dict] = {}
    for name, positions, provenance in buckets:
        per_selector = {}
        for selector in selectors:
            predictions = selector.predict_bucket(context, positions)
            per_selector[selector.name] = score(context.dataset, positions, predictions)
        body[name] = {
            "provenance": provenance,
            "selectors": per_selector,
        }

    fitted = {}
    for selector in selectors:
        if isinstance(selector, TrainPairLookup):
            fitted[selector.name] = {"memorised_pairs": len(selector.table)}
        elif isinstance(selector, FeatureLookup):
            fitted[selector.name] = {
                "key": list(selector.subset),
                "keys_fitted": selector.keys_fitted,
                "train_label_purity": rd(selector.train_purity),
                "keys_with_unanimous_train_target": selector.unanimous_keys,
            }
        elif isinstance(selector, RandomCandidate):
            fitted[selector.name] = {
                "seed_label": selector.seed_label,
                "seed": selector.seed,
            }
        elif isinstance(selector, MajorityClass):
            fitted[selector.name] = {
                "train_majority_label": selector.majority_label.value,
                "train_admit_share": rd(selector.train_share),
                "emitted": (
                    BOTTOM_SYMBOL
                    if selector.prediction.target == BOTTOM_INDEX
                    else selector.prediction.target
                ),
            }
    return {
        "family": fold.family,
        "fold": fold.name,
        "role": fold.role,
        "fold_sha256": fold.sha256(),
        "held_out_test_bucket": held_out_test_bucket(fold),
        "bucket_order": [name for name, _, _ in buckets],
        "buckets": body,
        "fitted": fitted,
    }


def family_summary(table: list[dict]) -> list[dict]:
    """Per ``(family, bucket, selector)`` mean and range over the family's folds."""

    keyed: dict[tuple[str, str, str], list[dict]] = {}
    for row in table:
        for bucket, block in row["buckets"].items():
            for selector, metrics in block["selectors"].items():
                keyed.setdefault((row["family"], bucket, selector), []).append(metrics)

    out: list[dict] = []
    for (family, bucket, selector), blocks in sorted(keyed.items()):
        summary = {
            "family": family,
            "bucket": bucket,
            "selector": selector,
            "folds": len(blocks),
        }
        for metric in (
            "joint_exact_success",
            "admission_balanced_accuracy",
            "forced_third_exact_accuracy",
            "coverage",
        ):
            values = [block[metric] for block in blocks if block[metric] is not None]
            nulls = sum(1 for block in blocks if block[metric] is None)
            summary[metric] = {
                "mean": rd(ratio(sum(values), len(values))) if values else None,
                "min": rd(min(values)) if values else None,
                "max": rd(max(values)) if values else None,
                "null_folds": nulls,
            }
        out.append(summary)
    return out


# ---------------------------------------------------------------------------
# observed vs. the coordinator's pre-registered predictions
# ---------------------------------------------------------------------------

def coordinator_predictions(table: list[dict]) -> dict:
    """Expected-vs-observed for the five values predicted before this ran."""

    by_family: dict[str, list[dict]] = {}
    for row in table:
        by_family.setdefault(row["family"], []).append(row)

    oracle_rows = []
    for name in ("ExactNativeRelation", "ExactSFPCircuit"):
        offenders = []
        buckets_seen = 0
        for row in table:
            for bucket, block in row["buckets"].items():
                buckets_seen += 1
                value = block["selectors"][name]["joint_exact_success"]
                if value != 1.0:
                    offenders.append(
                        {
                            "family": row["family"],
                            "fold": row["fold"],
                            "bucket": bucket,
                            "joint_exact_success": value,
                        }
                    )
        oracle_rows.append(
            {
                "selector": name,
                "buckets_scored": buckets_seen,
                **pin(
                    0,
                    len(offenders),
                    (
                        f"{name} joint exact success = 1.0 on every bucket of every "
                        "fold"
                    ),
                ),
                "offenders": offenders[:8],
            }
        )

    coverage_rows = []
    for family, expected_size, expected_folds in (("LOHO", 36, 14), ("LOFPO", 144, 7)):
        observed = [
            {
                "fold": row["fold"],
                "size": row["buckets"]["test_within"]["selectors"]["TrainPairLookup"][
                    "size"
                ],
                "covered": row["buckets"]["test_within"]["selectors"][
                    "TrainPairLookup"
                ]["covered"],
            }
            for row in by_family.get(family, [])
        ]
        observed_shape = {
            "distinct_covered": sorted({row["covered"] for row in observed}),
            "distinct_size": sorted({row["size"] for row in observed}),
            "folds": len(observed),
        }
        expected_shape = {
            "distinct_covered": [0],
            "distinct_size": [expected_size],
            "folds": expected_folds,
        }
        coverage_rows.append(
            {
                "family": family,
                "bucket": "test_within",
                "selector": "TrainPairLookup",
                "expected": expected_shape,
                "observed": observed_shape,
                "agrees": expected_shape == observed_shape,
                "per_fold": observed,
                "statement": (
                    f"coverage on {family} test_within = 0/{expected_size} for every "
                    "fold: both Events of every within-habitat test pair are held "
                    "out, so no such pair can occur in train"
                ),
            }
        )

    same_habitat = []
    for row in by_family.get("LOHO", []):
        block = row["buckets"]["test_within"]["selectors"]["SameHabitatOnly"]
        same_habitat.append(
            {
                "fold": row["fold"],
                "sensitivity": block["admission_sensitivity"],
                "specificity": block["admission_specificity"],
                "balanced": block["admission_balanced_accuracy"],
                "said_admit_on_all_36": block["size"] == 36
                and block["admission_sensitivity"] == 1.0
                and block["admission_specificity"] == 0.0,
            }
        )
    same_habitat_row = {
        "family": "LOHO",
        "bucket": "test_within",
        "selector": "SameHabitatOnly",
        "folds": len(same_habitat),
        "expected": {"sensitivity": 1.0, "specificity": 0.0, "balanced": 0.5},
        "observed": {
            "sensitivity": sorted({row["sensitivity"] for row in same_habitat}),
            "specificity": sorted({row["specificity"] for row in same_habitat}),
            "balanced": sorted({row["balanced"] for row in same_habitat}),
        },
        "agrees": len(same_habitat) == 14
        and all(row["said_admit_on_all_36"] for row in same_habitat)
        and {row["balanced"] for row in same_habitat} == {0.5},
        "statement": (
            "on LOHO test_within SameHabitatOnly predicts ADMIT for all 36 pairs, "
            "since every within-habitat pair is same-habitat: sensitivity 1.0, "
            "specificity 0.0, admission balanced accuracy exactly 0.5"
        ),
        "why_it_matters": (
            "0.5 is chance. The LOHO test_within bucket cannot be won by a "
            "habitat-matching shortcut, and that is exactly the property that "
            "makes the primary fold family discriminative."
        ),
    }

    empirical = []
    for row in table:
        for bucket, block in row["buckets"].items():
            metrics = block["selectors"]["RandomCandidate"]
            empirical.append((metrics["size"], metrics["joint_exact_hits"]))
    pooled_items = sum(size for size, _ in empirical)
    pooled_hits = sum(hits for _, hits in empirical)
    random_row = {
        "selector": "RandomCandidate",
        "convention": (
            "1/85 — BOTTOM is INCLUDED in the candidate set. Declared in advance."
        ),
        "candidate_set_size": CANDIDATE_SET_SIZE,
        "analytic_joint_exact_success": rd(1.0 / CANDIDATE_SET_SIZE),
        "analytic_admission_balanced_accuracy": 0.5,
        "empirical_pooled_items": pooled_items,
        "empirical_pooled_joint_exact_hits": pooled_hits,
        "empirical_pooled_joint_exact_success": rd(ratio(pooled_hits, pooled_items)),
        "agrees": True,
        "statement": (
            "the analytic chance level is 1/85 for joint exact success and for "
            "forced third exact accuracy, and exactly 0.5 for admission balanced "
            "accuracy. The empirical pooled value is a finite-sample draw and is "
            "reported beside it; it is NOT asserted equal, so this row's agreement "
            "flag records only that both numbers were produced."
        ),
    }

    rows = [*oracle_rows, *coverage_rows, same_habitat_row, random_row]
    return {
        "policy": (
            "these five values were predicted by the coordinator before this module "
            "ran. A mismatch is reported, never absorbed into the prediction."
        ),
        "checks": rows,
        "all_agree": all(row["agrees"] for row in rows),
    }


# ---------------------------------------------------------------------------
# THE COMPACT-FEATURE LEAKAGE SEARCH
# ---------------------------------------------------------------------------

def feature_subsets() -> tuple[tuple[str, ...], ...]:
    """Every declared atom subset up to :data:`MAX_SUBSET_SIZE`, in a fixed order."""

    out: list[tuple[str, ...]] = []
    for size in range(1, MAX_SUBSET_SIZE + 1):
        out.extend(itertools.combinations(ATOM_NAMES, size))
    return tuple(out)


def declared_family(space: FeatureSpace) -> list[dict]:
    """The recorded feature family: name, cost class, definition, cardinality."""

    return [
        {
            "atom": name,
            "cost_class": cost,
            "definition": definition,
            "distinct_values_on_the_pool": space.cardinalities[space.index_of[name]],
            "values": list(space.vocabularies[space.index_of[name]])[:16],
        }
        for name, cost, definition, _ in FEATURE_ATOMS
    ]


def search_fold(context: Context, fold: Fold) -> list[dict]:
    """Fit and evaluate every feature subset on one fold. Train-only fitting."""

    held_out = held_out_test_bucket(fold)
    test_positions = tuple(sorted(fold.bucket(held_out)))
    balanced = tuple(
        sorted(set(fold.bucket(held_out)) | set(balanced_test_positions(fold)))
    )
    rows: list[dict] = []
    for subset in feature_subsets():
        selector = FeatureLookup(subset)
        selector.fit(context, fold)
        on_test = score(
            context.dataset,
            test_positions,
            selector.predict_bucket(context, test_positions),
        )
        on_balanced = score(
            context.dataset, balanced, selector.predict_bucket(context, balanced)
        )
        rows.append(
            {
                "family": fold.family,
                "fold": fold.name,
                "subset": subset,
                "cost_class": selector.cost_class,
                "keys_fitted": selector.keys_fitted,
                "train_label_purity": rd(selector.train_purity),
                "held_out_bucket": held_out,
                "test_joint_exact_success": on_test["joint_exact_success"],
                "test_admission_balanced_accuracy": on_test[
                    "admission_balanced_accuracy"
                ],
                "test_forced_third_exact_accuracy": on_test[
                    "forced_third_exact_accuracy"
                ],
                "test_coverage": on_test["coverage"],
                "balanced_admission_balanced_accuracy": on_balanced[
                    "admission_balanced_accuracy"
                ],
                "balanced_joint_exact_success": on_balanced["joint_exact_success"],
                "balanced_coverage": on_balanced["coverage"],
            }
        )
    return rows


def _best(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def cheap_failure_mode(
    context: Context,
    folds: tuple[Fold, ...],
    family: str,
    subset: tuple[str, ...],
) -> dict:
    """Where a cheaper-than-the-relation subset actually breaks, pooled over folds.

    Reported because the headline admission number of the best cheap subset is
    misleading on its own: the subset is right about the classes that habitat
    coincidence already settles and wrong about exactly the one class that
    requires block incidence.
    """

    per_class = {name: [0, 0] for name in CLASS_NAMES}
    admit_total = 0
    forced_hits = 0
    joint_hits = 0
    items = 0
    said_admit = 0
    said_bottom = 0
    nonadmit_total = 0
    for fold in folds:
        if fold.family != family:
            continue
        selector = FeatureLookup(subset)
        selector.fit(context, fold)
        held_out = held_out_test_bucket(fold)
        positions = tuple(
            sorted(set(fold.bucket(held_out)) | set(balanced_test_positions(fold)))
        )
        block = score(
            context.dataset, positions, selector.predict_bucket(context, positions)
        )
        items += block["size"]
        joint_hits += block["joint_exact_hits"]
        admit_total += block["admit"]
        nonadmit_total += block["nonadmission"]
        forced_hits += block["forced_third_exact_hits"]
        said_admit += round((block["admission_sensitivity"] or 0.0) * block["admit"])
        said_bottom += sum(block["per_class_nonadmission_hits"].values())
        for name in CLASS_NAMES:
            per_class[name][0] += block["per_class_nonadmission_size"][name]
            per_class[name][1] += block["per_class_nonadmission_hits"][name]

    sensitivity = ratio(said_admit, admit_total)
    specificity = ratio(said_bottom, nonadmit_total)
    return {
        "subset": list(subset),
        "pooled_over_folds": sum(1 for fold in folds if fold.family == family),
        "bucket": DERIVED_BUCKET,
        "items": items,
        "pooled_joint_exact_success": rd(ratio(joint_hits, items)),
        "pooled_admission_sensitivity": rd(sensitivity),
        "pooled_admission_specificity": rd(specificity),
        "pooled_admission_balanced_accuracy": (
            None
            if sensitivity is None or specificity is None
            else rd(0.5 * (sensitivity + specificity))
        ),
        "pooled_forced_third_exact_accuracy": rd(ratio(forced_hits, admit_total)),
        "pooled_per_class_nonadmission_accuracy": {
            name: rd(ratio(hits, seen)) for name, (seen, hits) in per_class.items()
        },
        "pooled_per_class_nonadmission_size": {
            name: seen for name, (seen, _) in per_class.items()
        },
        "diagnosis": (
            "the cheap subset is right wherever habitat coincidence already settles "
            "the answer and wrong on SAME_HABITAT_DISJOINT, which is exactly the "
            "class that separating requires block incidence. Its forced-third "
            "accuracy is separately reported and carries no relational content: a "
            "cheap key names no Event, so it cannot name a third one."
        ),
    }


def leakage_search(context: Context, folds: tuple[Fold, ...]) -> dict:
    """The whole search, plus the ceiling / shortcut partition and the verdict."""

    rows: list[dict] = []
    for fold in folds:
        rows.extend(search_fold(context, fold))

    grouped: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["subset"]), []).append(row)

    per_family_per_subset: list[dict] = []
    findings: dict[str, list[dict]] = {RELATION_EQUIVALENT: [], CHEAP: []}
    for (family, subset), block in sorted(grouped.items()):
        admission = [row["balanced_admission_balanced_accuracy"] for row in block]
        admission_within = [row["test_admission_balanced_accuracy"] for row in block]
        joint = [row["test_joint_exact_success"] for row in block]
        forced = [row["test_forced_third_exact_accuracy"] for row in block]
        purity = [row["train_label_purity"] for row in block]
        cost_class = block[0]["cost_class"]
        at_admission_ceiling = [
            row
            for row in block
            if row["balanced_admission_balanced_accuracy"] is not None
            and row["balanced_admission_balanced_accuracy"]
            >= ADMISSION_CEILING_THRESHOLD
        ]
        at_joint_ceiling = [
            row
            for row in block
            if row["test_joint_exact_success"] >= JOINT_CEILING_THRESHOLD
        ]
        summary = {
            "family": family,
            "subset": list(subset),
            "cost_class": cost_class,
            "folds": len(block),
            "max_train_label_purity": _best(purity),
            "max_test_joint_exact_success": _best(joint),
            "max_test_forced_third_exact_accuracy": _best(forced),
            "max_test_within_admission_balanced_accuracy": _best(admission_within),
            "max_admission_balanced_accuracy": _best(admission),
            "folds_at_admission_ceiling": len(at_admission_ceiling),
            "folds_at_joint_ceiling": len(at_joint_ceiling),
        }
        per_family_per_subset.append(summary)
        if at_admission_ceiling or at_joint_ceiling:
            findings[cost_class].append(
                {
                    **summary,
                    "reaches": (
                        "admission ceiling"
                        if at_admission_ceiling and not at_joint_ceiling
                        else "joint ceiling"
                        if at_joint_ceiling and not at_admission_ceiling
                        else "admission and joint ceiling"
                    ),
                }
            )

    families = sorted({row["family"] for row in rows})
    best_cheap = {}
    for family in families:
        cheap_rows = [
            row
            for row in per_family_per_subset
            if row["family"] == family and row["cost_class"] == CHEAP
        ]
        best = max(
            cheap_rows,
            key=lambda row: (
                row["max_admission_balanced_accuracy"] or 0.0,
                row["max_test_joint_exact_success"] or 0.0,
            ),
        )
        best_cheap[family] = {
            "subset": best["subset"],
            "max_admission_balanced_accuracy": best["max_admission_balanced_accuracy"],
            "max_test_joint_exact_success": best["max_test_joint_exact_success"],
            "gap_to_admission_ceiling": rd(
                ADMISSION_CEILING_THRESHOLD
                - (best["max_admission_balanced_accuracy"] or 0.0)
            ),
        }

    failure_mode = {
        family: cheap_failure_mode(context, folds, family, tuple(row["subset"]))
        for family, row in sorted(best_cheap.items())
    }

    cheap_by_family = {
        family: [row for row in findings[CHEAP] if row["family"] == family]
        for family in families
    }
    verdict = {}
    for family in families:
        shortcuts = cheap_by_family[family]
        role = folds_module.ROLES[family]
        verdict[family] = {
            "role": role,
            "cheaper_than_the_relation_ceilings": len(shortcuts),
            "discriminative": not shortcuts,
            "statement": (
                (
                    "no feature cheaper than the certified relation reaches or "
                    "approaches ceiling on this family's test buckets, so the "
                    "family remains discriminative"
                )
                if not shortcuts
                else (
                    "a feature CHEAPER than the certified relation determines the "
                    "label on this family's test buckets: the family is "
                    "NON-DISPOSITIVE and every result on it must be discounted"
                )
            ),
            "shortcut_subsets": [row["subset"] for row in shortcuts],
        }

    structural = [
        family for family in families if family in ("LOHO", "LOFPO")
    ]
    return {
        "purpose": (
            "the controlling task requires a mechanical pre-training search for "
            "compact exposed features that determine the label on the proposed "
            "folds. This is that search."
        ),
        "declared_family": declared_family(context.space),
        "atom_count": len(FEATURE_ATOMS),
        "cost_partition_declared_in_advance": {
            CHEAP: [
                name for name, cost, _, _ in FEATURE_ATOMS if cost == CHEAP
            ],
            RELATION_EQUIVALENT: [
                name
                for name, cost, _, _ in FEATURE_ATOMS
                if cost == RELATION_EQUIVALENT
            ],
        },
        "max_subset_size": MAX_SUBSET_SIZE,
        "subsets_searched": len(feature_subsets()),
        "subset_size_census": {
            str(size): sum(
                1 for subset in feature_subsets() if len(subset) == size
            )
            for size in range(1, MAX_SUBSET_SIZE + 1)
        },
        "folds_searched": len(folds),
        "fits_performed": len(rows),
        "features_read": (
            "the pair's task-level coordinates only: task.PairRecord fields plus "
            "catalogue block incidence. No atom reads an arm's token tensor, an "
            "arm's code, or any model output."
        ),
        "fitting_rule": (
            "for each subset the best deterministic map feature -> label is fitted "
            "on the fold's 'train' bucket ONLY (train-majority label, ties to "
            "NONADMISSION), together with a map feature -> target where the key "
            "bucket's train ADMIT pairs supply one (train-majority forced third, "
            "ties to the smallest Event index). A test key unseen in train is "
            "uncovered and falls back to BOTTOM."
        ),
        "buckets_evaluated": [
            "the fold's held-out test bucket (test_within, or "
            "test_random_holdout for the control)",
            f"{DERIVED_BUCKET} — {DERIVED_BUCKET_STATEMENT}",
        ],
        "ceiling_thresholds": {
            "admission_balanced_accuracy": ADMISSION_CEILING_THRESHOLD,
            "joint_exact_success": JOINT_CEILING_THRESHOLD,
            "declared_in_advance": (
                "both thresholds were fixed before any number was measured and are "
                "never tuned against an observation"
            ),
        },
        "per_family_per_subset": per_family_per_subset,
        "findings": findings,
        "finding_counts": {
            RELATION_EQUIVALENT: len(findings[RELATION_EQUIVALENT]),
            CHEAP: len(findings[CHEAP]),
        },
        "partition_statement": (
            "A feature that IS the certified relation (shared_block_count, "
            "block_id_equality) is not a shortcut; it is the very thing the arms "
            "are being asked to represent, so its deterministic baseline is a "
            "CEILING and belongs with the oracles. A feature CHEAPER than the "
            "relation that still determines the label on a fold's test set is a "
            "genuine SHORTCUT and makes that fold non-dispositive."
        ),
        "admission_versus_forcing": (
            "the relation-equivalent atoms determine ADMISSION only. Neither "
            "shared_block_count nor block_id_equality names WHICH certified block "
            "is shared, so neither determines the forced third. Their forced-third "
            "accuracy stays near chance, which is why forcing is the part of the "
            "task where a representation has to do real work."
        ),
        "best_cheaper_than_the_relation_subset_per_family": best_cheap,
        "cheaper_than_the_relation_failure_mode": failure_mode,
        "why_some_relation_equivalent_subsets_miss_the_ceiling": (
            "a relation-equivalent atom paired with an atom that identifies the "
            "held-out structural unit (habitat_pair, fano_point_pair) produces test "
            "keys that never occur in train, so the lookup is UNCOVERED on "
            "test_within and falls back to BOTTOM. That is not a weakness of the "
            "relation; it is independent confirmation that the fold really does "
            "withhold the whole structural unit."
        ),
        "random_control_finding": {
            "family": "RANDOM_CONTROL",
            "role": folds_module.ROLES["RANDOM_CONTROL"],
            "cheap_ceilings": len(cheap_by_family.get("RANDOM_CONTROL", [])),
            "statement": (
                "cheaper-than-the-relation subsets DO reach the admission ceiling "
                "on the random pair holdout, and that is reported plainly rather "
                "than softened: the control fold is won by 'same habitat and not "
                "repeated => ADMIT'. The mechanism is the imbalance trap folds.py "
                "already names. The control's balanced cross-habitat sample is "
                "drawn FROM its own holdout rather than from a disjoint bucket, so "
                "the diagnostic bucket keeps the pool's 92.9% CROSS_HABITAT "
                "dominance; specificity is then carried almost entirely by "
                "cross-habitat negatives and the handful of SAME_HABITAT_DISJOINT "
                "errors barely register. The structural families concentrate those "
                "same negatives instead of diluting them, which is why they survive "
                "the same feature. The random control was already declared "
                "regression-control-only and not dispositive science; this finding "
                "is a mechanical confirmation of that declaration, and no result on "
                "the control may be read as evidence about representation."
            ),
        },
        "per_family_verdict": verdict,
        "structural_families_all_discriminative": all(
            verdict[family]["discriminative"] for family in structural
        ),
        "shortcut_in_a_structural_family": any(
            not verdict[family]["discriminative"] for family in structural
        ),
        "search_table_sha256": digest(
            [
                [
                    row["family"],
                    row["fold"],
                    list(row["subset"]),
                    row["cost_class"],
                    row["train_label_purity"],
                    row["test_joint_exact_success"],
                    row["test_admission_balanced_accuracy"],
                    row["test_forced_third_exact_accuracy"],
                    row["test_coverage"],
                    row["balanced_admission_balanced_accuracy"],
                    row["balanced_joint_exact_success"],
                ]
                for row in rows
            ]
        ),
    }


# ---------------------------------------------------------------------------
# per-arm: does the arm's own code contain a label-determining feature?
# ---------------------------------------------------------------------------

def arm_label_determining_features(dataset: Dataset) -> dict:
    """Whether each arm's own code carries a label-determining feature.

    This is a statement ABOUT THE ARMS, not a defect report. Arm B contains one by
    design — that is the ``ExactSFPCircuit`` — and Arm D inherits it because its
    relabeling is structure preserving. Arm C should not, because its alignment is
    deliberately destroyed. ``arms.relation_report`` is reused verbatim so this
    module re-derives nothing.
    """

    catalogue = dataset.catalogue
    codec = SfpCodec()
    circuit = ExactSfpCircuit()
    built = arms.all_arms(dataset)
    by_name = {arm.name: arm for arm in built}
    native = arms.native_codes(catalogue, codec)
    reports = [
        arms.relation_report(dataset, native, circuit, "B_sfp"),
        arms.relation_report(
            dataset, arms.codes_of_arm(by_name["C_scrambled"]), circuit, "C_scrambled"
        ),
        arms.relation_report(
            dataset, arms.codes_of_arm(by_name["D_relabeled"]), circuit, "D_relabeled"
        ),
    ]
    rows = []
    for report in reports:
        rows.append(
            {
                "arm": report["arm"],
                "label_determining_feature_in_the_arms_own_code": report[
                    "relation_preserved"
                ],
                "admission_disagreements_with_the_native_relation": report[
                    "admission_disagreements"
                ],
                "forced_third_disagreements": report["forced_third_disagreements"],
            }
        )
    for row in rows:
        if row["arm"] == "B_sfp":
            row["expected"] = True
            row["character"] = (
                "BY DESIGN, not leakage. Arm B's code is the SFP code, and "
                "sfp.ExactSfpCircuit solves the relation on it exactly with no "
                "learning. That is why an Arm-B success shows representation "
                "SUFFICIENCY / USE and not representation DISCOVERY."
            )
        elif row["arm"] == "C_scrambled":
            row["expected"] = False
            row["character"] = (
                "Arm C's alignment is deliberately destroyed, so its code carries "
                "no label-determining feature. That is the control working, not a "
                "defect."
            )
        else:
            row["expected"] = True
            row["character"] = (
                "Arm D is Arm B under certified structure-preserving relabelings, "
                "so it inherits the same by-design property. Arm D is an "
                "equivalence check, not an ablation."
            )
        row["agrees"] = (
            row["label_determining_feature_in_the_arms_own_code"] == row["expected"]
        )
    return {
        "method": (
            "arms.relation_report reused verbatim, with sfp.ExactSfpCircuit as a "
            "nonlearned REFERENCE ORACLE for a structural check. Every comparison "
            "is against task.py's certified native records; the circuit generates "
            "no label here and never could."
        ),
        "arms_checked": rows,
        "arms_not_sfp_coded": [
            {
                "arm": "A_native",
                "label_determining_feature_in_the_arms_own_code": None,
                "character": (
                    "Arm A emits the certified projective Event ray as 16 floats, "
                    "not an SFP indicator code, so the SFP circuit cannot be run "
                    "on it and the question does not apply. The relation is "
                    "computable from the ray by the native FIPS relation, but not "
                    "by any compact feature of the arm's own token vector."
                ),
            },
            {
                "arm": "E_opaque",
                "label_determining_feature_in_the_arms_own_code": None,
                "character": (
                    "Arm E is an arbitrary 84-way one-hot identity code, OPTIONAL "
                    "and DIAGNOSTIC only. It carries Event identity and no "
                    "relational field at all."
                ),
            },
        ],
        "all_agree": all(row["agrees"] for row in rows),
        "fence": (
            "Arm B containing a label-determining feature is by design, not "
            "leakage."
        ),
    }


# ---------------------------------------------------------------------------
# upstream consistency
# ---------------------------------------------------------------------------

def upstream_check(dataset: Dataset, folds: tuple[Fold, ...]) -> dict:
    """The frozen upstream really is the frozen upstream. Nothing is re-derived."""

    manifest = digest(
        {
            "fences": list(folds_module.FENCES),
            "sampling_policy": folds_module.SAMPLING_POLICY,
            "catalogue_sha256": dataset.catalogue.sha256(),
            "dataset_sha256": dataset.sha256(),
            "folds": [[fold.family, fold.name, fold.sha256()] for fold in folds],
        }
    )
    counts = dataset.counts()
    checks = {
        "catalogue_sha256": pin(
            EXPECTED_CATALOGUE_SHA256,
            dataset.catalogue.sha256(),
            "the 84-Event catalogue is the frozen one",
        ),
        "dataset_sha256": pin(
            EXPECTED_DATASET_SHA256,
            dataset.sha256(),
            "the 7056-record pool is the frozen one",
        ),
        "fold_manifest_sha256": pin(
            EXPECTED_FOLD_MANIFEST_SHA256,
            manifest,
            "the 22 fold manifests are the frozen ones",
        ),
        "folds_total": pin(
            POOL_PINS["folds_total"], len(folds), "14 LOHO + 7 LOFPO + 1 control"
        ),
        "folds_structural": pin(
            POOL_PINS["folds_structural"],
            len(structural_folds(folds)),
            "21 structural folds (the control is not structural)",
        ),
    }
    for name in ("admit", "repeated", "same_habitat_disjoint", "cross_habitat"):
        checks[f"class_{name}"] = pin(
            POOL_PINS[name], counts[name], f"class census: {name}"
        )
    checks["ordered_pairs_total"] = pin(
        POOL_PINS["ordered_pairs_total"],
        counts["ordered_pairs_total"],
        "84 * 84 ordered pairs",
    )
    return {
        "checks": checks,
        "all_agree": all(row["agrees"] for row in checks.values()),
        "reused_not_rederived": [
            "task.build_dataset (7056 frozen PairRecords, the sole label source)",
            "task.Catalogue (84 Events, 14 habitats, 56 blocks, block incidence)",
            "folds.all_folds (22 fold manifests, buckets and balanced samples)",
            "sfp.SfpCodec / sfp.ExactSfpCircuit (the nonlearned reference oracle)",
            "arms.relation_report / arms.codes_of_arm (per-arm structural check)",
        ],
        "not_imported": (
            "harness.py is owned by a parallel task and is deliberately not "
            "imported. No torch, no training, no sweep driver."
        ),
    }


def bottom_fence(dataset: Dataset) -> dict:
    """Bottom is an explicit distinct symbol, never an algebraic zero."""

    zero_is_an_event = any(
        record.target_index == 0 for record in dataset.records if record.admitted
    )
    return {
        "bottom_index": BOTTOM_INDEX,
        "bottom_symbol": BOTTOM_SYMBOL,
        "statement": (
            "BOTTOM is the sentinel index -1. It is NOT Event index 0, NOT FFF=000, "
            "NOT pp=00 and NOT any algebraic zero."
        ),
        "event_index_zero_is_a_real_forced_third": zero_is_an_event,
        "candidate_set": CANDIDATE_SET_DECLARATION,
        "no_selector_emits_zero_for_nonadmission": (
            "every selector answers non-admission with BOTTOM_INDEX = -1; the "
            "metric suite tests prediction == BOTTOM_INDEX and never tests for a "
            "zero"
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    folds = all_folds(dataset)
    catalogue = dataset.catalogue

    codec = SfpCodec()
    circuit = ExactSfpCircuit()
    native = native_relation_table(catalogue)
    circuit_answers = sfp_circuit_table(catalogue, codec, circuit)
    frozen = tuple(record.target_index for record in dataset.records)

    oracle_fence = {
        "native_table_equals_the_frozen_labels": pin(
            True,
            native == frozen,
            (
                "ExactNativeRelation recomputed from fips_basic agrees with all "
                "7056 frozen target indices, exactly"
            ),
        ),
        "circuit_table_equals_the_frozen_labels": pin(
            True,
            circuit_answers == frozen,
            (
                "ExactSfpCircuit agrees with all 7056 frozen target indices, "
                "exactly. Already certified upstream; re-checked here because this "
                "module scores the circuit as an oracle."
            ),
        ),
        "circuit_is_not_a_label_source": (
            "the circuit table is compared to task.py's records and is never "
            "written into them. Labels come from the certified native FIPS "
            "relation only."
        ),
    }

    space = build_feature_space(dataset)
    habitat_members = {
        habitat: tuple(
            index
            for index in range(len(catalogue))
            if catalogue.habitats[index] == habitat
        )
        for habitat in sorted(set(catalogue.habitats))
    }
    context = Context(
        dataset=dataset,
        native=native,
        circuit=circuit_answers,
        space=space,
        habitat_members=habitat_members,
    )

    table = [evaluate_fold(context, fold) for fold in folds]
    summary = family_summary(table)
    predictions = coordinator_predictions(table)
    search = leakage_search(context, folds)
    arm_report = arm_label_determining_features(dataset)
    upstream = upstream_check(dataset, folds)

    selector_declarations = [
        {"selector": selector.name, "kind": selector.kind, **selector.declaration()}
        for selector in build_selectors()
    ]

    oracles_perfect = all(
        row["agrees"]
        for row in predictions["checks"]
        if row.get("selector") in ("ExactNativeRelation", "ExactSFPCircuit")
        and "buckets_scored" in row
    )
    agrees = (
        upstream["all_agree"]
        and all(row["agrees"] for row in oracle_fence.values() if isinstance(row, dict))
        and oracles_perfect
        and predictions["all_agree"]
        and arm_report["all_agree"]
        and not search["shortcut_in_a_structural_family"]
    )
    control_shortcuts = search["random_control_finding"]["cheap_ceilings"]
    if not search["shortcut_in_a_structural_family"] and not control_shortcuts:
        statement = (
            "baselines scored and no shortcut cheaper than the relation found; "
            "folds remain discriminative"
        )
    elif not search["shortcut_in_a_structural_family"]:
        statement = (
            f"a cheaper-than-relation shortcut exists on {control_shortcuts} feature "
            "subsets of the RANDOM_CONTROL fold ('same habitat and not repeated => "
            "ADMIT'); that fold is flagged non-dispositive. The structural families "
            "LOHO (primary) and LOFPO (harder secondary) admit no such shortcut and "
            "remain discriminative."
        )
    else:
        offenders = [
            family
            for family, row in search["per_family_verdict"].items()
            if not row["discriminative"] and family in ("LOHO", "LOFPO")
        ]
        statement = (
            "a cheaper-than-relation shortcut exists on "
            f"{', '.join(offenders)}; those folds are flagged non-dispositive"
        )

    payload = {
        "module": (
            "Issue 009 Task 8: deterministic baselines, the shared metric "
            "definitions, and the compact-feature leakage search"
        ),
        "purpose": (
            "fix the nonlearned floor and ceiling the learned arms are measured "
            "against, and establish mechanically — before any training — whether a "
            "compact exposed feature determines the label on the proposed folds"
        ),
        "fences": list(FENCES),
        "upstream": upstream,
        "bottom_fence": bottom_fence(dataset),
        "oracle_fence": oracle_fence,
        "metric_definitions": METRIC_DEFINITIONS,
        "determinism": {
            "rounding": ROUNDING_POLICY,
            "round_digits": ROUND_DIGITS,
            "seed_derivation": SEED_DERIVATION,
            "base_seed": BASE_SEED,
            "hash_seed_invariance": (
                "no set or dict iteration order and no hash() value reaches the "
                "artifact: feature vocabularies are built in sorted(repr(value)) "
                "order, every bucket is iterated in ascending position order, and "
                "every report list is sorted. Verified by replaying under two "
                "PYTHONHASHSEED values."
            ),
            "torch_free": True,
        },
        "selectors": selector_declarations,
        "selector_order": list(SELECTOR_ORDER),
        "coordinator_predictions": predictions,
        "baseline_table": table,
        "family_summary": summary,
        "leakage_search": search,
        "arm_label_determining_features": arm_report,
        "provenance": {
            "topographo_version": importlib.metadata.version("topographo"),
            "base_commit": BASE_COMMIT,
            "upstream_modules": [
                "experiments/sfp_representation/task.py (Task 4)",
                "experiments/sfp_representation/folds.py (Task 5)",
                "experiments/sfp_representation/arms.py (Task 6)",
                "experiments/sfp_representation/sfp.py (Task 3)",
            ],
            "owns": [
                "experiments/sfp_representation/baselines.py",
                "experiments/sfp_representation/009_artifacts/baselines.json",
            ],
        },
        "digests": {
            "catalogue": dataset.catalogue.sha256(),
            "dataset": dataset.sha256(),
            "native_oracle_table": digest(list(native)),
            "sfp_circuit_table": digest(list(circuit_answers)),
            "feature_space": digest(
                [list(column) for column in space.codes]
            ),
            "baseline_table": digest(
                [
                    [
                        row["family"],
                        row["fold"],
                        bucket,
                        selector,
                        metrics["joint_exact_success"],
                        metrics["admission_balanced_accuracy"],
                        metrics["forced_third_exact_accuracy"],
                        metrics["coverage"],
                    ]
                    for row in table
                    for bucket, block in sorted(row["buckets"].items())
                    for selector, metrics in sorted(block["selectors"].items())
                ]
            ),
            "leakage_search_table": search["search_table_sha256"],
        },
        "verdict": {
            "agrees": agrees,
            "statement": statement,
            "oracles_perfect_everywhere": oracles_perfect,
            "coordinator_predictions_agree": predictions["all_agree"],
            "structural_families_all_discriminative": search[
                "structural_families_all_discriminative"
            ],
            "relation_equivalent_ceilings_found": search["finding_counts"][
                RELATION_EQUIVALENT
            ],
            "cheaper_than_the_relation_shortcuts_found": search["finding_counts"][
                CHEAP
            ],
            "arm_expectations_agree": arm_report["all_agree"],
            "upstream_agrees": upstream["all_agree"],
        },
    }
    return payload


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _row(family: str, bucket: str, summary: list[dict]) -> None:
    print(f"  {family} / {bucket}", flush=True)
    for entry in summary:
        if entry["family"] != family or entry["bucket"] != bucket:
            continue
        joint = entry["joint_exact_success"]["mean"]
        balanced = entry["admission_balanced_accuracy"]["mean"]
        forced = entry["forced_third_exact_accuracy"]["mean"]
        coverage = entry["coverage"]["mean"]
        print(
            f"    {entry['selector']:<28} joint={joint!s:<20} "
            f"admit_bal={balanced!s:<20} forced={forced!s:<20} "
            f"cov={coverage!s}",
            flush=True,
        )


def _report(payload: dict) -> None:
    upstream = payload["upstream"]["checks"]
    print(
        "upstream: catalogue="
        f"{upstream['catalogue_sha256']['observed'][:8]} "
        f"dataset={upstream['dataset_sha256']['observed'][:8]} "
        f"folds={upstream['fold_manifest_sha256']['observed'][:8]} "
        f"agree={payload['upstream']['all_agree']}",
        flush=True,
    )
    print("baseline table (family means over folds):", flush=True)
    summary = payload["family_summary"]
    for family in ("LOHO", "LOFPO"):
        for bucket in ("test_within", "test_cross_habitat", DERIVED_BUCKET):
            _row(family, bucket, summary)
    print("coordinator predictions:", flush=True)
    for row in payload["coordinator_predictions"]["checks"]:
        print(
            f"  {row.get('selector', '?'):<28} agrees={row['agrees']}  "
            f"{row['statement'][:96]}",
            flush=True,
        )
    search = payload["leakage_search"]
    print(
        f"leakage search: {search['atom_count']} atoms, subsets up to "
        f"{search['max_subset_size']} ({search['subsets_searched']} subsets) over "
        f"{search['folds_searched']} folds = {search['fits_performed']} fits",
        flush=True,
    )
    print(
        f"  relation_equivalent ceilings: "
        f"{search['finding_counts'][RELATION_EQUIVALENT]}  "
        f"cheaper_than_the_relation shortcuts: "
        f"{search['finding_counts'][CHEAP]}",
        flush=True,
    )
    for family, row in sorted(search["per_family_verdict"].items()):
        print(
            f"  {family:<16} role={row['role']:<24} "
            f"discriminative={row['discriminative']}",
            flush=True,
        )
    for family, row in sorted(
        search["best_cheaper_than_the_relation_subset_per_family"].items()
    ):
        print(
            f"  best cheap subset on {family:<16} "
            f"{'+'.join(row['subset'])} -> admit_bal="
            f"{row['max_admission_balanced_accuracy']} "
            f"(gap to ceiling {row['gap_to_admission_ceiling']})",
            flush=True,
        )
    for row in payload["arm_label_determining_features"]["arms_checked"]:
        print(
            f"  arm {row['arm']:<14} label_determining_feature="
            f"{row['label_determining_feature_in_the_arms_own_code']} "
            f"expected={row['expected']} agrees={row['agrees']}",
            flush=True,
        )
    print(f"baselines digest: {payload['digests']['baseline_table']}", flush=True)
    print(f"leakage digest:   {payload['digests']['leakage_search_table']}", flush=True)


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
                f"FAIL: re-derived artifact is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches baselines.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

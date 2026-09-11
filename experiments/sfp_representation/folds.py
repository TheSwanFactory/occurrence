"""Frozen fold manifests over the 7056-pair admit-and-force pool (Issue 009, Task 5).

This module owns fold construction only. It imports the frozen dataset from
:mod:`task` and re-derives nothing: no labels, no catalogue, no geometry. Folds
**partition** the certified pool; they never relabel it.

Fold families
-------------
1. ``LOHO`` — leave-one-habitat-out, 14 folds, **PRIMARY**. One ``(P, delta)``
   habitat (6 Events) is held out per fold. No Event of the held-out habitat may
   appear in any relation-training example, so no test Event identity can be
   memorised from training. All individual ``S`` and ``FFF`` coordinate values
   still occur in training (only one ``(P, delta)`` *combination* is removed).
2. ``LOFPO`` — leave-one-Fano-point-out, 7 folds, **HARDER SECONDARY**. Both
   signs of one Fano point (2 habitats, 12 Events) are held out, so the whole
   ``FFF = P`` label is unseen in training and the relation must transfer to an
   unseen Fano label while retaining the same abstract law. One ``FFF`` value
   being entirely absent from training is an *intended property* of this family,
   not a defect.
3. ``RANDOM_CONTROL`` — a uniform random ordered-pair holdout, **regression
   control only**. Not dispositive science.

Every fold partitions all 7056 ordered pairs into explicitly named buckets;
nothing silently vanishes::

    train        both Events outside the held-out set
    test_within  both Events inside the held-out set
    straddle     exactly one Event inside the held-out set

Straddle pairs are excluded from training (they contain a held-out Event) but
they are the *only* source of CROSS_HABITAT negatives at test time for the LOHO
family, because ``test_within`` there holds no cross-habitat pair. They are
therefore carried as a named test bucket — the balanced diagnostic
``test_cross_habitat`` is drawn from ``straddle`` — never discarded.

Independently predicted counts (verified, never adjusted to fit)
---------------------------------------------------------------
LOHO, per fold::

    held-out Events                 6
    test_within                    36  = 24 ADMIT + 6 REPEATED + 6 SAME_HABITAT_DISJOINT
    train pool (78 Events)       6084  = 78^2
    straddle                      936  = 2 * 6 * 78
    sum                          7056
    train same-habitat            468  = 13 * 36 (312 ADMIT + 78 REPEATED + 78 SHD)
    train CROSS_HABITAT          5616

LOFPO, per fold::

    held-out Events                12  (2 habitats)
    test_within                   144  = 48 ADMIT + 12 REPEATED + 12 SAME_HABITAT_DISJOINT
                                        + 72 CROSS_HABITAT (2 * 6 * 6, sign partners crossed)
    train pool (72 Events)       5184  = 72^2
    straddle                     1728  = 2 * 12 * 72
    sum                          7056
    train same-habitat            432  = 12 * 36 (288 ADMIT + 72 REPEATED + 72 SHD)
    train CROSS_HABITAT         4752

``test_within`` for LOFPO already contains 72 CROSS_HABITAT pairs because the two
held-out habitats share a Fano point but differ in sign, and ``task.py``
classifies same-``P``/opposite-sign as CROSS_HABITAT.

The imbalance trap
------------------
CROSS_HABITAT is ``6552/7056 = 92.9%`` of the pool and ``5616/6084 = 92.3%`` of a
LOHO train pool. Left alone the task degenerates into a trivial
"same habitat?" classifier. So a fixed, digest-pinned, **stratified balanced
sample** of cross-habitat pairs is carried as an admission diagnostic:

* the ratio is declared **in advance** (:data:`CROSS_HABITAT_SAMPLE_RATIO` ``= 1``
  cross-habitat negative per ADMIT pair of the same bucket) and is never tuned;
* the target is raised to the number of ordered-habitat-pair strata when smaller,
  so every stratum is represented at least once — also declared in advance;
* draws come from an explicitly seeded ``random.Random`` whose integer seed is
  recorded in the manifest, never from ``set``/``dict`` iteration order or
  ``hash()``;
* draws are **stratified** over ordered habitat pairs and the per-stratum
  distribution is reported so the stratification is auditable;
* the resulting index lists are digest-pinned.

The full unsampled cross-habitat sets stay available (``straddle`` in full, and
``cross_habitat_positions`` over any bucket). Reported admission performance must
use **balanced accuracy** so residual imbalance cannot inflate it.

Fences
------
::

    Random pair holdout is a regression control only; the primary science withholds
    complete structural units (habitats, Fano points).
    algebraic zero != NONADMISSION;   0 != bottom
    Labels come from the certified native FIPS relation; folds only partition them.
    Admission performance must be reported as balanced accuracy.
    Cross-habitat numerical dominance must not reduce the task to "same habitat?".

Scope
-----
Folds only. Representation arms, models, training code, scorers and baselines are
owned by later tasks and are deliberately absent here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from task import (
    Dataset,
    Label,
    NonadmissionClass,
    PairRecord,
    build_dataset,
    digest,
    habitat_label,
    sign_bit,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "folds.json"

BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"
EXPECTED_CATALOGUE_SHA256 = (
    "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097"
)
EXPECTED_DATASET_SHA256 = (
    "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721"
)

FENCES = (
    (
        "Random pair holdout is a regression control only; the primary science "
        "withholds complete structural units (habitats, Fano points)."
    ),
    "algebraic zero != NONADMISSION;   0 != bottom",
    (
        "Labels come from the certified native FIPS relation; folds only partition "
        "them."
    ),
    "Admission performance must be reported as balanced accuracy.",
    'Cross-habitat numerical dominance must not reduce the task to "same habitat?".',
)

# --- the sampling policy, declared in advance and never tuned ---------------
BASE_SEED = 90050501
CROSS_HABITAT_SAMPLE_RATIO = 1
RANDOM_CONTROL_HOLDOUT_FRACTION = 0.15

SAMPLING_POLICY = {
    "ratio": CROSS_HABITAT_SAMPLE_RATIO,
    "ratio_statement": (
        "target size = CROSS_HABITAT_SAMPLE_RATIO * (ADMIT count of the reference "
        "bucket), raised to the number of ordered-habitat-pair strata when that is "
        "larger so every stratum is represented at least once"
    ),
    "reasoning": (
        "ratio 1 makes cross-habitat negatives exactly as numerous as ADMIT pairs "
        "in the same bucket, so chance-level balanced accuracy is 0.5 and no "
        "decision threshold gains an advantage from prevalence alone. The value was "
        "chosen a priori from the class census (336 ADMIT vs 6552 CROSS_HABITAT) "
        "and is NOT tuned against any arm's measured performance; the floor at one "
        "pair per stratum is a stratification-coverage requirement, not a fit."
    ),
    "stratum": (
        "the ordered habitat pair, written 'i->j' over the compact habitat indices "
        "of habitat_index_legend (ascending by (P, delta)); allocation is base = "
        "target // strata to every stratum, with the remainder given to strata "
        "drawn by the seeded Random from the sorted stratum key list"
    ),
    "base_seed": BASE_SEED,
    "seed_derivation": (
        "random.Random(int(sha256('folds/v1|<base_seed>|<label>').hexdigest()[:16], "
        "16)); the derived integer is recorded per sample. No set/dict iteration "
        "order and no hash() value participates."
    ),
    "training_set": (
        "bucket 'train' is the training pool; 'train_cross_habitat_balanced' is the "
        "balanced cross-habitat subsample of it recommended for relation training"
    ),
    "diagnostic_set": (
        "'test_cross_habitat' is the balanced cross-habitat admission diagnostic, "
        "drawn from 'straddle' for the structural families"
    ),
    "full_sets_retained": (
        "the unsampled cross-habitat sets are retained: 'straddle' is carried whole "
        "and cross_habitat_positions(dataset, positions) recovers the full "
        "cross-habitat subset of any bucket"
    ),
    "metric_requirement": (
        "admission performance MUST be reported as balanced accuracy; raw accuracy "
        "over the unsampled pool is inadmissible because CROSS_HABITAT is 92.9% of "
        "it"
    ),
}

FAMILY_LOHO = "LOHO"
FAMILY_LOFPO = "LOFPO"
FAMILY_RANDOM = "RANDOM_CONTROL"

ROLES = {
    FAMILY_LOHO: "primary",
    FAMILY_LOFPO: "harder_secondary",
    FAMILY_RANDOM: "regression_control_only",
}

CLASS_NAMES = (
    Label.ADMIT.value,
    NonadmissionClass.REPEATED.value,
    NonadmissionClass.SAME_HABITAT_DISJOINT.value,
    NonadmissionClass.CROSS_HABITAT.value,
)

LOHO_PINS = {
    "folds": 14,
    "held_out_events": 6,
    "test_within": 36,
    "test_within_admit": 24,
    "test_within_repeated": 6,
    "test_within_same_habitat_disjoint": 6,
    "test_within_cross_habitat": 0,
    "train": 6084,
    "train_admit": 312,
    "train_repeated": 78,
    "train_same_habitat_disjoint": 78,
    "train_cross_habitat": 5616,
    "train_same_habitat": 468,
    "straddle": 936,
    "straddle_cross_habitat": 936,
    "sum": 7056,
}

LOFPO_PINS = {
    "folds": 7,
    "held_out_events": 12,
    "held_out_habitats": 2,
    "test_within": 144,
    "test_within_admit": 48,
    "test_within_repeated": 12,
    "test_within_same_habitat_disjoint": 12,
    "test_within_cross_habitat": 72,
    "train": 5184,
    "train_admit": 288,
    "train_repeated": 72,
    "train_same_habitat_disjoint": 72,
    "train_cross_habitat": 4752,
    "train_same_habitat": 432,
    "straddle": 1728,
    "straddle_cross_habitat": 1728,
    "sum": 7056,
}

POOL_PINS = {
    "events": 84,
    "habitats": 14,
    "fano_points": 7,
    "ordered_pairs_total": 7056,
    "admit": 336,
    "repeated": 84,
    "same_habitat_disjoint": 84,
    "cross_habitat": 6552,
}


# ---------------------------------------------------------------------------
# deterministic helpers
# ---------------------------------------------------------------------------

def derive_seed(label: str) -> int:
    """Deterministic integer seed for one named draw. Never ``hash()``."""

    material = f"folds/v1|{BASE_SEED}|{label}"
    return int(digest(material)[:16], 16)


def pair_class(record: PairRecord) -> str:
    """The four-way class name of a record: ADMIT or its non-admission class."""

    if record.admitted:
        return Label.ADMIT.value
    return record.nonadmission_class.value


def class_counts(dataset: Dataset, positions: tuple[int, ...]) -> dict[str, int]:
    """Class census of a bucket. All four class keys are always present."""

    tally = Counter(pair_class(dataset.records[p]) for p in positions)
    body = {name: tally[name] for name in CLASS_NAMES}
    body["total"] = len(positions)
    body["same_habitat"] = sum(
        1 for p in positions if dataset.records[p].same_habitat
    )
    return body


def habitat_index_map(dataset: Dataset) -> dict[tuple[int, int], int]:
    """``(P, delta)`` -> compact habitat index, ascending by ``(P, delta)``."""

    return {
        habitat: i
        for i, habitat in enumerate(sorted(set(dataset.catalogue.habitats)))
    }


def habitat_index_legend(dataset: Dataset) -> list[dict]:
    """The legend for the compact habitat indices used in stratum keys."""

    return [
        {"index": i, "habitat": habitat_label(*habitat)}
        for habitat, i in sorted(habitat_index_map(dataset).items(), key=lambda kv: kv[1])
    ]


def habitat_pair_key(
    index_map: dict[tuple[int, int], int], record: PairRecord
) -> str:
    """Stratum key: the ordered habitat pair ``'i->j'`` over habitat indices.

    Compact by design (the manifest records the legend) and independent of set /
    dict iteration order: the indices come from ``sorted(set(habitats))``.
    """

    return f"{index_map[record.habitat_a]}->{index_map[record.habitat_b]}"


def events_of(dataset: Dataset, positions: tuple[int, ...]) -> tuple[int, ...]:
    """Ascending Event indices occurring in either slot of a bucket."""

    seen: set[int] = set()
    for p in positions:
        record = dataset.records[p]
        seen.add(record.a_index)
        seen.add(record.b_index)
    return tuple(sorted(seen))


def cross_habitat_positions(
    dataset: Dataset, positions: tuple[int, ...]
) -> tuple[int, ...]:
    """The full, unsampled CROSS_HABITAT subset of a bucket, ascending."""

    return tuple(
        p
        for p in positions
        if dataset.records[p].nonadmission_class
        is NonadmissionClass.CROSS_HABITAT
    )


@dataclass(frozen=True)
class BalancedSample:
    """One digest-pinned, stratified, seeded cross-habitat subsample."""

    name: str
    source_bucket: str
    reference_bucket: str
    reference_admit: int
    nominal_target: int
    realized_target: int
    strata: int
    seed: int
    positions: tuple[int, ...]
    distribution: tuple[tuple[str, int], ...]
    source_size: int

    def as_json(self, *, include_positions: bool = True) -> dict:
        body = {
            "name": self.name,
            "source_bucket": self.source_bucket,
            "source_size": self.source_size,
            "reference_bucket": self.reference_bucket,
            "reference_admit": self.reference_admit,
            "ratio": CROSS_HABITAT_SAMPLE_RATIO,
            "nominal_target": self.nominal_target,
            "realized_target": self.realized_target,
            "size": len(self.positions),
            "strata": self.strata,
            "seed": self.seed,
            "per_stratum_distribution": {
                key: count for key, count in self.distribution
            },
            "per_stratum_counts_observed": sorted(
                {count for _, count in self.distribution}
            ),
            "sha256": digest(list(self.positions)),
        }
        if include_positions:
            body["positions"] = list(self.positions)
        return body

    def as_row(self) -> list:
        return [
            self.name,
            self.source_bucket,
            self.reference_bucket,
            self.reference_admit,
            self.nominal_target,
            self.realized_target,
            self.strata,
            self.seed,
            list(self.positions),
            [[key, count] for key, count in self.distribution],
        ]


def stratified_cross_habitat_sample(
    dataset: Dataset,
    *,
    name: str,
    source_bucket: str,
    source: tuple[int, ...],
    reference_bucket: str,
    reference_admit: int,
) -> BalancedSample:
    """Draw the declared balanced cross-habitat sample, stratified and seeded.

    The population is the full CROSS_HABITAT subset of ``source``, grouped by
    ordered habitat pair. Allocation is even by construction: ``target //
    strata`` to every stratum, remainder to strata drawn by the seeded
    ``random.Random`` from the *sorted* stratum key list. Nothing here depends on
    set/dict iteration order or ``hash()``.
    """

    pool = cross_habitat_positions(dataset, source)
    index_map = habitat_index_map(dataset)
    grouped: dict[str, list[int]] = defaultdict(list)
    for position in pool:  # ascending, so every stratum list is ascending
        grouped[habitat_pair_key(index_map, dataset.records[position])].append(
            position
        )
    keys = sorted(grouped)
    if not keys:
        raise AssertionError(f"{name}: no cross-habitat pairs to sample from")

    nominal = CROSS_HABITAT_SAMPLE_RATIO * reference_admit
    target = max(nominal, len(keys))
    if target > len(pool):
        raise AssertionError(
            f"{name}: target {target} exceeds the cross-habitat pool {len(pool)}"
        )

    base, remainder = divmod(target, len(keys))
    rng = random.Random(derive_seed(name))
    bonus = set(rng.sample(keys, remainder)) if remainder else set()

    chosen: list[int] = []
    distribution: list[tuple[str, int]] = []
    for key in keys:
        take = base + (1 if key in bonus else 0)
        if take > len(grouped[key]):
            raise AssertionError(
                f"{name}: stratum {key} holds {len(grouped[key])} < {take} pairs"
            )
        picked = rng.sample(grouped[key], take)
        chosen.extend(picked)
        distribution.append((key, take))

    positions = tuple(sorted(chosen))
    if len(positions) != target or len(set(positions)) != target:
        raise AssertionError(f"{name}: drew {len(positions)} pairs, not {target}")
    if any(
        dataset.records[p].nonadmission_class is not NonadmissionClass.CROSS_HABITAT
        for p in positions
    ):
        raise AssertionError(f"{name}: sampled a non-CROSS_HABITAT pair")
    if not set(positions) <= set(source):
        raise AssertionError(f"{name}: sample escaped its source bucket")

    return BalancedSample(
        name=name,
        source_bucket=source_bucket,
        reference_bucket=reference_bucket,
        reference_admit=reference_admit,
        nominal_target=nominal,
        realized_target=target,
        strata=len(keys),
        seed=derive_seed(name),
        positions=positions,
        distribution=tuple(distribution),
        source_size=len(pool),
    )


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One frozen fold: named buckets over the 7056 positions, plus its digest.

    ``buckets`` is an ordered tuple of ``(name, positions)`` pairs whose
    positions partition the whole pool. ``balanced`` carries the digest-pinned
    stratified cross-habitat subsamples. Nothing is mutated after construction.
    """

    family: str
    name: str
    role: str
    held_out_events: tuple[int, ...]
    held_out_habitats: tuple[str, ...]
    held_out_fano_points: tuple[int, ...]
    buckets: tuple[tuple[str, tuple[int, ...]], ...]
    balanced: tuple[BalancedSample, ...]
    counts: tuple[tuple[str, dict[str, int]], ...]
    notes: tuple[str, ...]

    def bucket_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.buckets)

    def bucket(self, name: str) -> tuple[int, ...]:
        for candidate, positions in self.buckets:
            if candidate == name:
                return positions
        raise KeyError(f"fold {self.name!r} has no bucket {name!r}")

    def sample(self, name: str) -> BalancedSample:
        for candidate in self.balanced:
            if candidate.name == name:
                return candidate
        raise KeyError(f"fold {self.name!r} has no balanced sample {name!r}")

    @property
    def train(self) -> tuple[int, ...]:
        return self.bucket("train")

    @property
    def test_within(self) -> tuple[int, ...]:
        return self.bucket("test_within")

    @property
    def straddle(self) -> tuple[int, ...]:
        return self.bucket("straddle")

    @property
    def structural(self) -> bool:
        return self.family in (FAMILY_LOHO, FAMILY_LOFPO)

    def as_row(self) -> list:
        """Full-fidelity canonical row; the per-fold digest is taken over this."""

        return [
            self.family,
            self.name,
            self.role,
            list(self.held_out_events),
            list(self.held_out_habitats),
            list(self.held_out_fano_points),
            [[name, list(positions)] for name, positions in self.buckets],
            [sample.as_row() for sample in self.balanced],
        ]

    def sha256(self) -> str:
        return digest(self.as_row())

    def as_json(self, *, include_bucket_positions: bool = False) -> dict:
        body = {
            "family": self.family,
            "name": self.name,
            "role": self.role,
            "held_out_events": list(self.held_out_events),
            "held_out_event_count": len(self.held_out_events),
            "held_out_habitats": list(self.held_out_habitats),
            "held_out_fano_points": list(self.held_out_fano_points),
            "bucket_order": list(self.bucket_names()),
            "buckets": [
                {
                    "name": name,
                    "size": len(positions),
                    "sha256": digest(list(positions)),
                    "class_counts": dict(self.counts)[name],
                    "deterministic_sample": _stride_sample(positions, 12),
                    **(
                        {"positions": list(positions)}
                        if include_bucket_positions
                        else {}
                    ),
                }
                for name, positions in self.buckets
            ],
            "balanced_cross_habitat_samples": [
                sample.as_json() for sample in self.balanced
            ],
            "notes": list(self.notes),
            "sha256": self.sha256(),
        }
        return body


def _stride_sample(positions: tuple[int, ...], count: int) -> list[int]:
    """A deterministic stride sample; full lists are pinned by the digests."""

    if not positions:
        return []
    stride = max(1, len(positions) // count)
    return sorted(
        {positions[i * stride] for i in range(count) if i * stride < len(positions)}
    )


def partition_by_held_out(
    dataset: Dataset, held_out: frozenset[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split all 7056 positions into ``(train, test_within, straddle)``."""

    train: list[int] = []
    within: list[int] = []
    straddle: list[int] = []
    for position, record in enumerate(dataset.records):
        left = record.a_index in held_out
        right = record.b_index in held_out
        if left and right:
            within.append(position)
        elif left or right:
            straddle.append(position)
        else:
            train.append(position)
    return tuple(train), tuple(within), tuple(straddle)


def _structural_fold(
    dataset: Dataset,
    *,
    family: str,
    name: str,
    held_out_events: tuple[int, ...],
    held_out_habitats: tuple[str, ...],
    held_out_fano_points: tuple[int, ...],
    notes: tuple[str, ...],
) -> Fold:
    held = frozenset(held_out_events)
    train, within, straddle = partition_by_held_out(dataset, held)
    buckets = (
        ("train", train),
        ("test_within", within),
        ("straddle", straddle),
    )
    counts = tuple((bname, class_counts(dataset, positions))
                   for bname, positions in buckets)
    lookup = dict(counts)

    balanced = (
        stratified_cross_habitat_sample(
            dataset,
            name=f"{family}/{name}/train_cross_habitat_balanced",
            source_bucket="train",
            source=train,
            reference_bucket="train",
            reference_admit=lookup["train"][Label.ADMIT.value],
        ),
        stratified_cross_habitat_sample(
            dataset,
            name=f"{family}/{name}/test_cross_habitat",
            source_bucket="straddle",
            source=straddle,
            reference_bucket="test_within",
            reference_admit=lookup["test_within"][Label.ADMIT.value],
        ),
    )
    return Fold(
        family=family,
        name=name,
        role=ROLES[family],
        held_out_events=held_out_events,
        held_out_habitats=held_out_habitats,
        held_out_fano_points=held_out_fano_points,
        buckets=buckets,
        balanced=balanced,
        counts=counts,
        notes=notes,
    )


def build_loho_folds(dataset: Dataset) -> tuple[Fold, ...]:
    """The 14 PRIMARY leave-one-habitat-out folds, ascending by ``(P, delta)``."""

    catalogue = dataset.catalogue
    folds = []
    for key, members in catalogue.events_of_habitat():
        point = catalogue.fano_point(members[0])
        folds.append(
            _structural_fold(
                dataset,
                family=FAMILY_LOHO,
                name=key,
                held_out_events=tuple(members),
                held_out_habitats=(key,),
                held_out_fano_points=(point,),
                notes=(
                    "PRIMARY transfer test: no held-out Event appears in any "
                    "training pair, so no test Event identity can be memorised.",
                    "All 7 FFF values and both S values remain available in "
                    "training; only one (P, delta) combination is removed.",
                    "test_within holds no CROSS_HABITAT pair by construction, so "
                    "the balanced cross-habitat diagnostic is drawn from straddle.",
                    "straddle is excluded from training (it contains held-out "
                    "Events) and carried as test data, never discarded.",
                ),
            )
        )
    if len(folds) != LOHO_PINS["folds"]:
        raise AssertionError(f"built {len(folds)} LOHO folds, not 14")
    return tuple(folds)


def build_lofpo_folds(dataset: Dataset) -> tuple[Fold, ...]:
    """The 7 HARDER SECONDARY leave-one-Fano-point-out folds (both signs held)."""

    catalogue = dataset.catalogue
    folds = []
    for point, members in catalogue.events_of_fano_point():
        habitats = tuple(
            sorted({habitat_label(*catalogue.habitat(i)) for i in members})
        )
        if len(habitats) != LOFPO_PINS["held_out_habitats"]:
            raise AssertionError(
                f"Fano point {point} spans {len(habitats)} habitats, not 2"
            )
        folds.append(
            _structural_fold(
                dataset,
                family=FAMILY_LOFPO,
                name=f"P={point}",
                held_out_events=tuple(members),
                held_out_habitats=habitats,
                held_out_fano_points=(point,),
                notes=(
                    "HARDER SECONDARY: both signs of one Fano point are held out, "
                    "removing the whole FFF = P habitat pair.",
                    "One FFF value is entirely absent from training BY DESIGN — an "
                    "intended property of this family, not a defect. Both S values "
                    "remain available.",
                    "test_within already contains 72 CROSS_HABITAT pairs because "
                    "the two held-out habitats share P but differ in sign and "
                    "task.py classifies same-P/opposite-sign as CROSS_HABITAT.",
                    "straddle is excluded from training and carried as test data.",
                ),
            )
        )
    if len(folds) != LOFPO_PINS["folds"]:
        raise AssertionError(f"built {len(folds)} LOFPO folds, not 7")
    return tuple(folds)


def build_random_control(
    dataset: Dataset,
    *,
    holdout_fraction: float = RANDOM_CONTROL_HOLDOUT_FRACTION,
    label: str = "random_pair_holdout",
) -> Fold:
    """A uniform random ordered-pair holdout. REGRESSION CONTROL ONLY.

    This fold withholds no structural unit: held-out Events also appear in
    training, so a model may memorise Event identities. It exists to detect
    regressions against previously recorded numbers and is **not dispositive
    science**. The primary science is :func:`build_loho_folds`; the harder
    secondary is :func:`build_lofpo_folds`.
    """

    total = len(dataset)
    holdout_size = int(total * holdout_fraction)
    name = f"{label}/fraction={holdout_fraction:g}"
    rng = random.Random(derive_seed(f"{FAMILY_RANDOM}/{name}"))
    holdout = tuple(sorted(rng.sample(range(total), holdout_size)))
    held = set(holdout)
    train = tuple(p for p in range(total) if p not in held)

    buckets = (
        ("train", train),
        ("test_random_holdout", holdout),
    )
    counts = tuple((bname, class_counts(dataset, positions))
                   for bname, positions in buckets)
    lookup = dict(counts)
    balanced = (
        stratified_cross_habitat_sample(
            dataset,
            name=f"{FAMILY_RANDOM}/{name}/train_cross_habitat_balanced",
            source_bucket="train",
            source=train,
            reference_bucket="train",
            reference_admit=lookup["train"][Label.ADMIT.value],
        ),
        stratified_cross_habitat_sample(
            dataset,
            name=f"{FAMILY_RANDOM}/{name}/test_cross_habitat",
            source_bucket="test_random_holdout",
            source=holdout,
            reference_bucket="test_random_holdout",
            reference_admit=lookup["test_random_holdout"][Label.ADMIT.value],
        ),
    )
    return Fold(
        family=FAMILY_RANDOM,
        name=name,
        role=ROLES[FAMILY_RANDOM],
        held_out_events=(),
        held_out_habitats=(),
        held_out_fano_points=(),
        buckets=buckets,
        balanced=balanced,
        counts=counts,
        notes=(
            "REGRESSION CONTROL ONLY — not dispositive science.",
            "Random pair holdout withholds no complete structural unit: every "
            "Event still occurs in training, so Event identity can be memorised.",
            "The primary science withholds complete structural units (habitats "
            "via LOHO, Fano points via LOFPO).",
            f"declared holdout fraction {holdout_fraction:g}, "
            f"seed {derive_seed(f'{FAMILY_RANDOM}/{name}')}",
        ),
    )


def all_folds(dataset: Dataset) -> tuple[Fold, ...]:
    """Every fold: 14 LOHO (primary), 7 LOFPO (harder secondary), 1 control."""

    return (
        *build_loho_folds(dataset),
        *build_lofpo_folds(dataset),
        build_random_control(dataset),
    )


def structural_folds(folds: tuple[Fold, ...]) -> tuple[Fold, ...]:
    return tuple(fold for fold in folds if fold.structural)


# ---------------------------------------------------------------------------
# THE LEAKAGE CHECK
# ---------------------------------------------------------------------------

def verify_leakage(dataset: Dataset, folds: tuple[Fold, ...]) -> dict:
    """Mechanical leakage audit over the ACTUAL training index lists.

    Nothing here argues from construction: the training Event set is recomputed
    from the recorded training positions and intersected with the held-out set.
    """

    total = POOL_PINS["ordered_pairs_total"]
    rows = []
    for fold in structural_folds(folds):
        held = set(fold.held_out_events)
        train_events = set(events_of(dataset, fold.train))
        intersection = sorted(held & train_events)

        buckets = dict(fold.buckets)
        sizes = {name: len(positions) for name, positions in fold.buckets}
        as_sets = {name: set(positions) for name, positions in fold.buckets}
        names = sorted(as_sets)
        pairwise_disjoint = all(
            not (as_sets[left] & as_sets[right])
            for i, left in enumerate(names)
            for right in names[i + 1:]
        )
        union = set().union(*as_sets.values())
        covers_pool = union == set(range(total))

        within_counts = dict(fold.counts)["test_within"]
        expected_classes = [
            Label.ADMIT.value,
            NonadmissionClass.REPEATED.value,
            NonadmissionClass.SAME_HABITAT_DISJOINT.value,
        ]
        if fold.family == FAMILY_LOFPO:
            expected_classes.append(NonadmissionClass.CROSS_HABITAT.value)
        classes_populated = {
            name: within_counts[name] > 0 for name in expected_classes
        }

        test_buckets = {
            "test_within": sizes["test_within"],
            "straddle": sizes["straddle"],
            "test_cross_habitat": len(fold.sample(
                f"{fold.family}/{fold.name}/test_cross_habitat"
            ).positions),
        }

        expected_held = (
            LOHO_PINS["held_out_events"]
            if fold.family == FAMILY_LOHO
            else LOFPO_PINS["held_out_events"]
        )

        balanced_clean = all(
            not (held & set(events_of(dataset, sample.positions)))
            for sample in fold.balanced
            if sample.source_bucket == "train"
        )

        rows.append({
            "family": fold.family,
            "fold": fold.name,
            "held_out_events": len(fold.held_out_events),
            "held_out_size_expected": expected_held,
            "held_out_size_agrees": len(fold.held_out_events) == expected_held,
            "train_events": len(train_events),
            "leaked_event_indices": intersection,
            "no_held_out_event_in_training": not intersection,
            "no_held_out_event_in_balanced_training_sample": balanced_clean,
            "bucket_sizes": sizes,
            "buckets_pairwise_disjoint": pairwise_disjoint,
            "bucket_sizes_sum": sum(sizes.values()),
            "bucket_sizes_sum_is_7056": sum(sizes.values()) == total,
            "buckets_cover_the_whole_pool": covers_pool,
            "test_buckets_non_empty": all(v > 0 for v in test_buckets.values()),
            "test_bucket_sizes": test_buckets,
            "expected_test_classes_populated": classes_populated,
            "all_expected_test_classes_populated": all(classes_populated.values()),
            "train_positions_recomputed_from_index_list": True,
            "checked_buckets": sorted(buckets),
        })

    return {
        "method": (
            "for every structural fold the training Event set is recomputed from "
            "the recorded training positions (not asserted by construction) and "
            "intersected with the held-out Event set"
        ),
        "structural_folds_checked": len(rows),
        "per_fold": rows,
        "no_leakage_anywhere": all(r["no_held_out_event_in_training"] for r in rows),
        "no_leakage_in_balanced_training_samples": all(
            r["no_held_out_event_in_balanced_training_sample"] for r in rows
        ),
        "all_buckets_pairwise_disjoint": all(
            r["buckets_pairwise_disjoint"] for r in rows
        ),
        "all_bucket_sums_are_7056": all(r["bucket_sizes_sum_is_7056"] for r in rows),
        "all_buckets_cover_the_pool": all(
            r["buckets_cover_the_whole_pool"] for r in rows
        ),
        "all_test_buckets_non_empty": all(r["test_buckets_non_empty"] for r in rows),
        "all_expected_test_classes_populated": all(
            r["all_expected_test_classes_populated"] for r in rows
        ),
        "all_held_out_sizes_agree": all(r["held_out_size_agrees"] for r in rows),
    }


def verify_held_out_partitions(
    dataset: Dataset, loho: tuple[Fold, ...], lofpo: tuple[Fold, ...]
) -> dict:
    """The 14 LOHO and the 7 LOFPO held-out sets each partition all 84 Events."""

    events = set(range(POOL_PINS["events"]))

    def check(folds: tuple[Fold, ...], expected_size: int) -> dict:
        sets = [set(fold.held_out_events) for fold in folds]
        union: set[int] = set()
        overlaps = []
        for i, left in enumerate(sets):
            for j in range(i + 1, len(sets)):
                if left & sets[j]:
                    overlaps.append([folds[i].name, folds[j].name])
            union |= left
        return {
            "folds": len(folds),
            "held_out_sizes": sorted({len(s) for s in sets}),
            "held_out_size_expected": expected_size,
            "every_held_out_set_has_expected_size": all(
                len(s) == expected_size for s in sets
            ),
            "pairwise_overlaps": overlaps,
            "pairwise_disjoint": not overlaps,
            "union_size": len(union),
            "covers_all_84_events": union == events,
            "partitions_all_84_events": (
                union == events
                and not overlaps
                and all(len(s) == expected_size for s in sets)
            ),
        }

    return {
        "LOHO": check(loho, LOHO_PINS["held_out_events"]),
        "LOFPO": check(lofpo, LOFPO_PINS["held_out_events"]),
    }


def verify_predicted_counts(
    dataset: Dataset, folds: tuple[Fold, ...], family: str, pins: dict
) -> dict:
    """Observed vs. independently predicted per-fold counts. Never adjusted."""

    rows = []
    for fold in folds:
        counts = dict(fold.counts)
        observed = {
            "held_out_events": len(fold.held_out_events),
            "test_within": counts["test_within"]["total"],
            "test_within_admit": counts["test_within"][Label.ADMIT.value],
            "test_within_repeated": counts["test_within"][
                NonadmissionClass.REPEATED.value
            ],
            "test_within_same_habitat_disjoint": counts["test_within"][
                NonadmissionClass.SAME_HABITAT_DISJOINT.value
            ],
            "test_within_cross_habitat": counts["test_within"][
                NonadmissionClass.CROSS_HABITAT.value
            ],
            "train": counts["train"]["total"],
            "train_admit": counts["train"][Label.ADMIT.value],
            "train_repeated": counts["train"][NonadmissionClass.REPEATED.value],
            "train_same_habitat_disjoint": counts["train"][
                NonadmissionClass.SAME_HABITAT_DISJOINT.value
            ],
            "train_cross_habitat": counts["train"][
                NonadmissionClass.CROSS_HABITAT.value
            ],
            "train_same_habitat": counts["train"]["same_habitat"],
            "straddle": counts["straddle"]["total"],
            "straddle_cross_habitat": counts["straddle"][
                NonadmissionClass.CROSS_HABITAT.value
            ],
            "sum": sum(len(p) for _, p in fold.buckets),
        }
        disagreements = {
            key: {"expected": pins[key], "observed": observed[key]}
            for key in observed
            if observed[key] != pins[key]
        }
        rows.append({
            "fold": fold.name,
            "observed": observed,
            "disagreements": disagreements,
            "agrees": not disagreements,
        })
    if family == FAMILY_LOFPO:
        for fold, row in zip(folds, rows):
            row["held_out_habitats"] = len(fold.held_out_habitats)
            row["held_out_habitats_agrees"] = (
                len(fold.held_out_habitats) == pins["held_out_habitats"]
            )
            row["agrees"] = row["agrees"] and row["held_out_habitats_agrees"]
    return {
        "family": family,
        "predicted": {k: v for k, v in pins.items() if k != "folds"},
        "folds_checked": len(rows),
        "folds_expected": pins["folds"],
        "fold_count_agrees": len(rows) == pins["folds"],
        "per_fold": rows,
        "all_folds_agree": all(row["agrees"] for row in rows)
        and len(rows) == pins["folds"],
        "prediction_policy": (
            "counts were predicted from the certified structure before the folds "
            "were built; a mismatch is reported, never absorbed into the prediction"
        ),
    }


def verify_coordinate_availability(
    dataset: Dataset, folds: tuple[Fold, ...]
) -> dict:
    """Which ``S`` and ``FFF`` coordinate values survive in each fold's training.

    LOHO: all 7 ``FFF`` values and both ``S`` values must remain — only one
    ``(P, delta)`` combination is removed. LOFPO: exactly one ``FFF`` value is
    absent from training BY DESIGN; that is the point of the harder family.
    """

    catalogue = dataset.catalogue
    all_points = sorted({p for p, _ in catalogue.habitats})
    all_signs = sorted({sign_bit(d) for _, d in catalogue.habitats})

    rows = []
    for fold in structural_folds(folds):
        train_events = events_of(dataset, fold.train)
        points = sorted({catalogue.fano_point(i) for i in train_events})
        signs = sorted({sign_bit(catalogue.delta(i)) for i in train_events})
        habitats = sorted(
            {habitat_label(*catalogue.habitat(i)) for i in train_events}
        )
        missing_points = [p for p in all_points if p not in points]
        missing_signs = [s for s in all_signs if s not in signs]
        if fold.family == FAMILY_LOHO:
            expected = {
                "fano_values_present": len(all_points),
                "sign_values_present": len(all_signs),
                "habitats_present": POOL_PINS["habitats"] - 1,
                "intended_missing_fano_values": 0,
            }
            statement = (
                "all 7 FFF values and both S values remain available in training; "
                "only the single (P, delta) combination is withheld"
            )
        else:
            expected = {
                "fano_values_present": len(all_points) - 1,
                "sign_values_present": len(all_signs),
                "habitats_present": POOL_PINS["habitats"] - 2,
                "intended_missing_fano_values": 1,
            }
            statement = (
                "one FFF value is entirely absent from training BY DESIGN (the "
                "harder fold family asks for transfer to an unseen Fano label); "
                "both S values remain"
            )
        observed = {
            "fano_values_present": len(points),
            "sign_values_present": len(signs),
            "habitats_present": len(habitats),
            "intended_missing_fano_values": len(missing_points),
        }
        rows.append({
            "family": fold.family,
            "fold": fold.name,
            "training_events": len(train_events),
            "training_fano_values": points,
            "training_sign_values": signs,
            "training_habitats": habitats,
            "missing_fano_values": missing_points,
            "missing_sign_values": missing_signs,
            "expected": expected,
            "observed": observed,
            "agrees": observed == expected and not missing_signs,
            "statement": statement,
        })

    loho_rows = [r for r in rows if r["family"] == FAMILY_LOHO]
    lofpo_rows = [r for r in rows if r["family"] == FAMILY_LOFPO]
    return {
        "per_fold": rows,
        "loho_all_coordinate_values_remain": all(r["agrees"] for r in loho_rows),
        "lofpo_one_fano_value_absent_by_design": all(
            r["agrees"] and len(r["missing_fano_values"]) == 1 for r in lofpo_rows
        ),
        "both_sign_values_always_remain": all(
            not r["missing_sign_values"] for r in rows
        ),
        "note": (
            "the LOFPO missing FFF value is an intended property of the harder "
            "fold family, reported explicitly and NOT a defect"
        ),
    }


def verify_balanced_samples(dataset: Dataset, folds: tuple[Fold, ...]) -> dict:
    """Every balanced sample: in-source, all cross-habitat, evenly stratified."""

    rows = []
    for fold in folds:
        for sample in fold.balanced:
            source = fold.bucket(sample.source_bucket)
            positions = set(sample.positions)
            spread = sorted({count for _, count in sample.distribution})
            rows.append({
                "fold": f"{fold.family}/{fold.name}",
                "sample": sample.name,
                "size": len(sample.positions),
                "realized_target": sample.realized_target,
                "nominal_target": sample.nominal_target,
                "reference_admit": sample.reference_admit,
                "ratio": CROSS_HABITAT_SAMPLE_RATIO,
                "strata": sample.strata,
                "seed": sample.seed,
                "per_stratum_counts": spread,
                "max_stratum_share": max(
                    count for _, count in sample.distribution
                ) / len(sample.positions),
                "subset_of_source": positions <= set(source),
                "all_cross_habitat": all(
                    dataset.records[p].nonadmission_class
                    is NonadmissionClass.CROSS_HABITAT
                    for p in sample.positions
                ),
                "size_matches_target": (
                    len(sample.positions) == sample.realized_target
                ),
                "stratification_even_within_one": max(spread) - min(spread) <= 1,
                "every_stratum_represented": min(spread) >= 1,
                "sha256": digest(list(sample.positions)),
            })
    return {
        "samples_checked": len(rows),
        "per_sample": rows,
        "all_subsets_of_their_source": all(r["subset_of_source"] for r in rows),
        "all_samples_are_cross_habitat": all(r["all_cross_habitat"] for r in rows),
        "all_sizes_match_target": all(r["size_matches_target"] for r in rows),
        "all_stratifications_even_within_one": all(
            r["stratification_even_within_one"] for r in rows
        ),
        "every_stratum_represented_everywhere": all(
            r["every_stratum_represented"] for r in rows
        ),
        "no_stratum_dominates": all(r["max_stratum_share"] <= 0.05 for r in rows),
    }


def verify_determinism(dataset: Dataset, folds: tuple[Fold, ...]) -> dict:
    """Rebuild every fold from scratch and compare per-fold digests."""

    rebuilt = all_folds(dataset)
    same_names = [f.name for f in rebuilt] == [f.name for f in folds]
    same_digests = [f.sha256() for f in rebuilt] == [f.sha256() for f in folds]
    return {
        "rebuild_fold_names_stable": same_names,
        "rebuild_fold_digests_stable": same_digests,
        "statement": (
            "every ordering comes from integer positions, sorted textual habitat "
            "keys or explicit tuples; all random draws use random.Random seeded by "
            "a recorded sha256-derived integer. No set/dict iteration order and no "
            "hash() value reaches the artifact"
        ),
        "interpreter_state_excluded": (
            "no interpreter flag (including PYTHONHASHSEED) is recorded, so the "
            "artifact is byte-identical across hash seeds"
        ),
    }


def verify_pool_consistency(dataset: Dataset) -> dict:
    """The upstream pool is the frozen one; folds re-derive no label."""

    counts = dataset.counts()
    observed = {
        "ordered_pairs_total": counts["ordered_pairs_total"],
        "admit": counts["admit"],
        "repeated": counts["repeated"],
        "same_habitat_disjoint": counts["same_habitat_disjoint"],
        "cross_habitat": counts["cross_habitat"],
        "events": len(dataset.catalogue),
        "habitats": len(dataset.catalogue.habitat_keys()),
        "fano_points": len(dataset.catalogue.events_of_fano_point()),
    }
    disagreements = {
        key: {"expected": POOL_PINS[key], "observed": observed[key]}
        for key in observed
        if observed[key] != POOL_PINS[key]
    }
    return {
        "observed": observed,
        "expected": POOL_PINS,
        "disagreements": disagreements,
        "pool_counts_agree": not disagreements,
        "catalogue_sha256": {
            "expected": EXPECTED_CATALOGUE_SHA256,
            "observed": dataset.catalogue.sha256(),
            "agrees": dataset.catalogue.sha256() == EXPECTED_CATALOGUE_SHA256,
        },
        "dataset_sha256": {
            "expected": EXPECTED_DATASET_SHA256,
            "observed": dataset.sha256(),
            "agrees": dataset.sha256() == EXPECTED_DATASET_SHA256,
        },
        "cross_habitat_share_of_pool": (
            observed["cross_habitat"] / observed["ordered_pairs_total"]
        ),
        "label_provenance": (
            "labels come from the certified native FIPS relation via task.py; this "
            "module only partitions them and never recomputes admission"
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    loho = build_loho_folds(dataset)
    lofpo = build_lofpo_folds(dataset)
    control = build_random_control(dataset)
    folds = (*loho, *lofpo, control)

    pool = verify_pool_consistency(dataset)
    loho_counts = verify_predicted_counts(dataset, loho, FAMILY_LOHO, LOHO_PINS)
    lofpo_counts = verify_predicted_counts(dataset, lofpo, FAMILY_LOFPO, LOFPO_PINS)
    leakage = verify_leakage(dataset, folds)
    partitions = verify_held_out_partitions(dataset, loho, lofpo)
    coordinates = verify_coordinate_availability(dataset, folds)
    balanced = verify_balanced_samples(dataset, folds)
    determinism = verify_determinism(dataset, folds)

    control_counts = dict(control.counts)
    fold_digests = [
        {
            "family": fold.family,
            "name": fold.name,
            "role": fold.role,
            "sha256": fold.sha256(),
            "bucket_sizes": {
                name: len(positions) for name, positions in fold.buckets
            },
            "balanced_sizes": {
                sample.name.rsplit("/", 1)[-1]: len(sample.positions)
                for sample in fold.balanced
            },
        }
        for fold in folds
    ]
    manifest_digest = digest({
        "fences": list(FENCES),
        "sampling_policy": SAMPLING_POLICY,
        "catalogue_sha256": dataset.catalogue.sha256(),
        "dataset_sha256": dataset.sha256(),
        "folds": [[fold.family, fold.name, fold.sha256()] for fold in folds],
    })

    laws = {
        "upstream_pool_counts_agree": pool["pool_counts_agree"],
        "upstream_catalogue_digest_agrees": pool["catalogue_sha256"]["agrees"],
        "upstream_dataset_digest_agrees": pool["dataset_sha256"]["agrees"],
        "loho_fold_count_is_14": loho_counts["fold_count_agrees"],
        "lofpo_fold_count_is_7": lofpo_counts["fold_count_agrees"],
        "loho_all_predicted_counts_agree": loho_counts["all_folds_agree"],
        "lofpo_all_predicted_counts_agree": lofpo_counts["all_folds_agree"],
        "no_held_out_event_in_any_training_pair": leakage["no_leakage_anywhere"],
        "no_held_out_event_in_any_balanced_training_sample": leakage[
            "no_leakage_in_balanced_training_samples"
        ],
        "buckets_pairwise_disjoint_every_fold": leakage[
            "all_buckets_pairwise_disjoint"
        ],
        "bucket_sizes_sum_to_7056_every_fold": leakage["all_bucket_sums_are_7056"],
        "buckets_cover_the_whole_pool_every_fold": leakage[
            "all_buckets_cover_the_pool"
        ],
        "every_test_bucket_non_empty": leakage["all_test_buckets_non_empty"],
        "every_expected_test_class_populated": leakage[
            "all_expected_test_classes_populated"
        ],
        "held_out_sizes_are_6_for_loho_and_12_for_lofpo": leakage[
            "all_held_out_sizes_agree"
        ],
        "loho_held_out_sets_partition_all_84_events": partitions["LOHO"][
            "partitions_all_84_events"
        ],
        "lofpo_held_out_sets_partition_all_84_events": partitions["LOFPO"][
            "partitions_all_84_events"
        ],
        "loho_all_coordinate_values_remain_in_training": coordinates[
            "loho_all_coordinate_values_remain"
        ],
        "lofpo_one_fano_value_absent_by_design": coordinates[
            "lofpo_one_fano_value_absent_by_design"
        ],
        "both_sign_values_always_remain": coordinates[
            "both_sign_values_always_remain"
        ],
        "balanced_samples_subset_of_source": balanced[
            "all_subsets_of_their_source"
        ],
        "balanced_samples_are_all_cross_habitat": balanced[
            "all_samples_are_cross_habitat"
        ],
        "balanced_sample_sizes_match_declared_target": balanced[
            "all_sizes_match_target"
        ],
        "balanced_samples_evenly_stratified": balanced[
            "all_stratifications_even_within_one"
        ],
        "every_stratum_represented_in_every_sample": balanced[
            "every_stratum_represented_everywhere"
        ],
        "no_single_habitat_pair_dominates_a_sample": balanced["no_stratum_dominates"],
        "fold_digests_stable_on_rebuild": determinism["rebuild_fold_digests_stable"],
        "fold_order_stable_on_rebuild": determinism["rebuild_fold_names_stable"],
        "all_fold_digests_distinct": len({f.sha256() for f in folds}) == len(folds),
    }
    broken = sorted(name for name, ok in laws.items() if not ok)

    return {
        "module": "experiments/sfp_representation/folds.py",
        "purpose": (
            "frozen fold manifests over the 7056-pair admit-and-force pool: 14 "
            "leave-one-habitat-out folds (primary), 7 leave-one-Fano-point-out "
            "folds (harder secondary) and 1 random pair holdout (regression "
            "control only)"
        ),
        "fences": list(FENCES),
        "fold_families": {
            FAMILY_LOHO: {
                "role": ROLES[FAMILY_LOHO],
                "folds": len(loho),
                "held_out_unit": "one (P, delta) habitat, 6 Events",
                "statement": (
                    "no Event from the held-out habitat appears in ANY "
                    "relation-training example; all individual S and FFF "
                    "coordinate values occurring elsewhere remain available; all "
                    "within-habitat admitted and non-admitted classes are tested"
                ),
                "why_primary": (
                    "no test Event identity can be memorised from training, so "
                    "this is the representation-transfer test"
                ),
            },
            FAMILY_LOFPO: {
                "role": ROLES[FAMILY_LOFPO],
                "folds": len(lofpo),
                "held_out_unit": "both signs of one Fano point, 2 habitats, 12 Events",
                "statement": (
                    "removes the whole FFF = P habitat pair and asks whether the "
                    "relation transfers to an unseen Fano label while retaining "
                    "the same abstract law"
                ),
                "intended_property": (
                    "one FFF value is entirely absent from training by design"
                ),
            },
            FAMILY_RANDOM: {
                "role": ROLES[FAMILY_RANDOM],
                "folds": 1,
                "held_out_unit": "uniformly random ordered pairs; no structural unit",
                "statement": (
                    "Random pair holdout is a regression control only; the primary "
                    "science withholds complete structural units (habitats, Fano "
                    "points)."
                ),
                "holdout_fraction": RANDOM_CONTROL_HOLDOUT_FRACTION,
                "holdout_seed": derive_seed(f"{FAMILY_RANDOM}/{control.name}"),
                "not_dispositive_science": True,
            },
        },
        "bucket_contract": {
            "train": "both Events outside the held-out set; the only training data",
            "test_within": "both Events inside the held-out set",
            "straddle": (
                "exactly one Event inside the held-out set: excluded from training "
                "because it contains a held-out Event, but carried as the named "
                "test source for cross-habitat negatives, never discarded"
            ),
            "test_cross_habitat": (
                "balanced digest-pinned cross-habitat diagnostic drawn from "
                "straddle for the structural families"
            ),
            "train_cross_habitat_balanced": (
                "balanced digest-pinned cross-habitat subsample of train, for "
                "relation training"
            ),
            "test_random_holdout": (
                "the random control's held-out pairs; control only"
            ),
            "sum_law": "every fold's buckets are disjoint and sum to exactly 7056",
        },
        "sampling_policy": SAMPLING_POLICY,
        "habitat_index_legend": habitat_index_legend(dataset),
        "imbalance_trap": {
            "cross_habitat_share_of_pool": pool["cross_habitat_share_of_pool"],
            "cross_habitat_share_of_a_loho_train_pool": (
                LOHO_PINS["train_cross_habitat"] / LOHO_PINS["train"]
            ),
            "requirement": (
                "carry a fixed digest-pinned balanced sample of cross-habitat "
                'pairs as an admission diagnostic; do not let their numerical '
                'dominance turn the task into a trivial "same habitat?" classifier'
            ),
            "metric": SAMPLING_POLICY["metric_requirement"],
        },
        "pins": {
            "pool": POOL_PINS,
            FAMILY_LOHO: LOHO_PINS,
            FAMILY_LOFPO: LOFPO_PINS,
        },
        "upstream": pool,
        "predicted_counts": {
            FAMILY_LOHO: loho_counts,
            FAMILY_LOFPO: lofpo_counts,
        },
        "leakage_check": leakage,
        "held_out_partitions": partitions,
        "coordinate_availability": coordinates,
        "balanced_sample_audit": balanced,
        "determinism": determinism,
        "random_control": {
            "name": control.name,
            "role": control.role,
            "holdout_fraction": RANDOM_CONTROL_HOLDOUT_FRACTION,
            "holdout_size": len(control.bucket("test_random_holdout")),
            "holdout_seed": derive_seed(f"{FAMILY_RANDOM}/{control.name}"),
            "train_size": len(control.train),
            "class_counts": {
                name: control_counts[name] for name in sorted(control_counts)
            },
            "seed_derivation": SAMPLING_POLICY["seed_derivation"],
            "warning": (
                "REGRESSION CONTROL ONLY — not dispositive science. Random pair "
                "holdout is a regression control only; the primary science "
                "withholds complete structural units (habitats, Fano points)."
            ),
        },
        "folds": [fold.as_json() for fold in folds],
        "fold_index": fold_digests,
        "artifact_policy": (
            "full per-bucket index lists (7056 positions x 22 folds) are omitted "
            "for size: every bucket carries its sha256, its size, its class census "
            "and a deterministic stride sample, and the buckets are exactly "
            "reproducible by importing this module. The digest-pinned balanced "
            "cross-habitat samples ARE recorded index-by-index because they are "
            "the fixed diagnostic sets."
        ),
        "digests": {
            "manifest": manifest_digest,
            "catalogue": dataset.catalogue.sha256(),
            "dataset": dataset.sha256(),
            "folds": [[fold.family, fold.name, fold.sha256()] for fold in folds],
            "method": (
                "hashlib.sha256(json.dumps(obj, sort_keys=True, "
                "separators=(',', ':')).encode()).hexdigest()"
            ),
        },
        "provenance": {
            "topographo_version": importlib.metadata.version("topographo"),
            "base_commit": BASE_COMMIT,
            "upstream_module": "experiments/sfp_representation/task.py (Task 4)",
            "reused_not_rederived": [
                "task.build_dataset (7056 frozen PairRecords)",
                "task.build_catalogue (84-Event catalogue, habitats, Fano points)",
                "task.Label / task.NonadmissionClass",
                "task.digest / task.habitat_label / task.sign_bit",
            ],
            "withheld_by_scope": [
                "representation arms", "models", "training code", "scorers",
                "baselines",
            ],
        },
        "downstream_contract": {
            "primary": "LOHO, 14 folds — report here first",
            "harder_secondary": "LOFPO, 7 folds",
            "control_only": FAMILY_RANDOM,
            "training_data": "bucket 'train' (or its balanced cross-habitat subsample)",
            "test_data": [
                "test_within (within-unit admitted and non-admitted classes)",
                "test_cross_habitat (balanced cross-habitat admission diagnostic)",
            ],
            "metric": SAMPLING_POLICY["metric_requirement"],
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - 21 structural folds frozen with zero leakage and all "
                "predicted counts agreeing"
                if not broken
                else "FAIL - see broken_laws"
            ),
        },
    }


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _report(payload: dict) -> None:
    for family in (FAMILY_LOHO, FAMILY_LOFPO):
        block = payload["predicted_counts"][family]
        print(
            f"{family}: {block['folds_checked']}/{block['folds_expected']} folds, "
            f"all predicted counts agree={block['all_folds_agree']}",
            flush=True,
        )
        row = block["per_fold"][0]["observed"]
        print(
            f"  {family} per fold: held_out={row['held_out_events']}, "
            f"train={row['train']}, test_within={row['test_within']}, "
            f"straddle={row['straddle']}, sum={row['sum']}",
            flush=True,
        )
    leakage = payload["leakage_check"]
    print(
        f"leakage: {leakage['structural_folds_checked']}/21 structural folds, "
        f"no_leakage={leakage['no_leakage_anywhere']}, "
        f"disjoint={leakage['all_buckets_pairwise_disjoint']}, "
        f"sums_7056={leakage['all_bucket_sums_are_7056']}, "
        f"test_classes_populated={leakage['all_expected_test_classes_populated']}",
        flush=True,
    )
    partitions = payload["held_out_partitions"]
    print(
        "held-out partitions: LOHO covers 84="
        f"{partitions['LOHO']['covers_all_84_events']}, LOFPO covers 84="
        f"{partitions['LOFPO']['covers_all_84_events']}",
        flush=True,
    )
    balanced = payload["balanced_sample_audit"]
    print(
        f"balanced cross-habitat samples: {balanced['samples_checked']} pinned, "
        f"ratio={CROSS_HABITAT_SAMPLE_RATIO}, base_seed={BASE_SEED}, "
        f"even_strata={balanced['all_stratifications_even_within_one']}, "
        f"no_stratum_dominates={balanced['no_stratum_dominates']}",
        flush=True,
    )
    coordinates = payload["coordinate_availability"]
    print(
        "coordinates: LOHO all values remain="
        f"{coordinates['loho_all_coordinate_values_remain']}, "
        "LOFPO one FFF absent by design="
        f"{coordinates['lofpo_one_fano_value_absent_by_design']}",
        flush=True,
    )
    control = payload["random_control"]
    print(
        f"random control (NOT dispositive science): holdout "
        f"{control['holdout_size']}/7056 at fraction "
        f"{control['holdout_fraction']}",
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
                f"FAIL: re-derived manifest is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches folds.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

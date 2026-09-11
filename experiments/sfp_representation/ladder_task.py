"""009.05 Rung-1 and Rung-2 target construction for the consequence-address ladder.

Executes `009.05`, **not** `009.04`. `009.04` proposed the same three-rung shape
but imposed a `B > C` success condition at the local `pp` rung; `009.05`
supersedes it before execution because `GL(2,2) ~= S3` acts as the full
permutation group on the three nonzero `pp` values, so every relabeling of that
local quotient is already a symmetry of "given two distinct values, return the
third".

This module owns the two new **output objects** and nothing else. The inputs, the
splits, the certified labels and the representation arms are all reused
unchanged and byte-pinned from the frozen `009.02` chain:

    task.py    the 7056-pair pool, labels from the certified native FIPS relation
    folds.py   the 22 frozen manifests (14 LOHO + 7 LOFPO + 1 random control)
    arms.py    the five representation arms and their token tensors
    sfp.py     the S | FFF | PP | pp codec

Rung 1 (`009.05` section 4) replaces the 84-way catalogue interface with the
local three-value forcing port::

    predict pp3 in {01, 10, 11};  S3 := S, FFF3 := FFF, PP3 := PP by construction

Rung 2 (`009.05` section 5) replaces it with the whole structured consequence
address::

    result = BOTTOM  or  (S_hat, FFF_hat, PP_hat, pp_hat)

THE LABEL FENCE. Every target here is a function of ``record.target_index`` --
the certified forced-third Event index that ``task.label_pair`` derived from
``topographo.ssd.fips_basic.third`` -- composed with the arm's own code table.
``sfp.ExactSfpCircuit`` is imported in this module for post-hoc structural
reporting only, exactly as in ``arms.py``; it appears in no target function, and
:func:`verify_label_source_fence` proves that mechanically by inspecting the
signatures of every target function for a parameter through which a circuit
could be passed.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import folds as folds_module
from arms import (
    all_arms,
    codes_of_arm,
    native_codes,
)
from folds import all_folds, class_counts, structural_folds
from harness import evaluation_positions, training_positions
from sfp import ExactSfpCircuit, SfpAddress, SfpCodec, address_of, bits2
from task import (
    BOTTOM_INDEX,
    BOTTOM_SYMBOL,
    Dataset,
    PairRecord,
    build_dataset,
    digest,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_task.json"

#: The frozen ``009.02`` result commit this ladder builds on. Pinned rather than
#: read from ``git rev-parse HEAD`` so the artifact stays replayable.
BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

EXECUTED_TASK = "009.05-Kiro-corrected-structured-consequence-address-ladder-task.md"
SUPERSEDED_TASK = "009.04-Kiro-structured-consequence-address-ladder-task.md"

EXPECTED_CATALOGUE_SHA256 = (
    "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097"
)
EXPECTED_DATASET_SHA256 = (
    "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721"
)
EXPECTED_FOLD_MANIFEST_SHA256 = (
    "27b01b5e7440c2449e28d8e5b3f82addba1028a1640707752ab01215a938a6ee"
)
EXPECTED_CHART_SHA256 = (
    "38dbbfec0111c654778bbf873f0dbcfd2d2141ff9b8405b71d273bbfd8830e5e"
)

FENCES = (
    "FFF=000 and pp=00 are formal algebraic completion values only.",
    "algebraic zero != NONADMISSION;   0 != bottom",
    (
        "BOTTOM is its own explicit output value, never pp=00, never FFF=000, "
        "never Event index 0."
    ),
    (
        "Targets come from the certified native FIPS relation via "
        "record.target_index, never from sfp.ExactSfpCircuit."
    ),
    "SFP is a representation layer, not a replacement OT evaluator.",
    (
        "The executable rule pp3 = pp1 XOR pp2 is never supplied to the learner; "
        "it is only used here to REPORT how much of the local law each arm's code "
        "still carries."
    ),
    (
        "B ~= C at Rung 1 is NOT evidence against H_struct and B > C at Rung 1 is "
        "NOT a success condition: the local quotient has full S3 symmetry."
    ),
)

#: The three nonzero local ports. ``pp = 00`` is a formal completion value and is
#: not a class.
PP_CLASSES = (1, 2, 3)

#: Field cardinalities of the Rung-2 structured address. The total is 16 field
#: slots plus one BOTTOM gate -- deliberately nowhere near the 84-way catalogue
#: interface that ``009.02`` failed on.
FIELD_CARDINALITIES = (("S", 2), ("FFF", 7), ("PP", 4), ("pp", 3))

STRUCTURED_SLOTS = sum(width for _, width in FIELD_CARDINALITIES) + 1

#: Which code table names each arm's OUTPUT alphabet.
#:
#: An arm's structured target must be written in the same code the arm's inputs
#: are written in, otherwise the task would be a translation exercise rather than
#: a forcing exercise. ``A_native`` has no code of its own, so it borrows the
#: exact SFP chart; that is the one arm asymmetry in this ladder and it is
#: reported rather than hidden.
ARM_OUTPUT_CONVENTION = {
    "A_native": "B_sfp",
    "B_sfp": "B_sfp",
    "C_scrambled": "C_scrambled",
    "D_relabeled": "D_relabeled",
}

SCIENCE_ARMS = ("A_native", "B_sfp", "C_scrambled", "D_relabeled")

#: Arms whose input AND output are the same SFP code. The B/C/D structural
#: comparison at Rung 2 is confined to these; ``A_native`` is secondary.
CODE_ARMS = ("B_sfp", "C_scrambled", "D_relabeled")

ARM_A_ASYMMETRY = (
    "A_native reads native projective coordinates and writes the exact SFP chart, "
    "because it has no code of its own. No SFP field enters its INPUT: arms.py "
    "build_arm_a takes only the catalogue, and arm_a_no_sfp_field certifies the "
    "token tensor carries no SFP unit. The shared output alphabet is a property of "
    "the output object, which is identical for every arm by construction, so this "
    "is a type-fair structured-output comparison and not an injection of SFP "
    "information into the native arm. It is still reported as the one asymmetry, "
    "and the decisive B/C/D structural contrast never involves it."
)

PP_SET_TARGET_CONVENTION = (
    "The PP head carries a SET target. A standalone Event has exactly two "
    "block-incidence presentations, (q, d) and (q XOR d, d), and sfp.address_of "
    "canonicalizes them into one unordered pair, so BOTH PP values name the same "
    "Event and neither is privileged. Training therefore maximizes the total "
    "probability of the two valid PP values, -log(p[q1] + p[q2]), and PP accuracy "
    "counts an argmax landing on either. This is the single documented "
    "canonicalization convention required by 009.05 section 5.3; it is a property "
    "of the codec, is identical for every arm, and is auditable independently of "
    "learning by verify_address_equivalence."
)

RUNG1_HOLDOUT_DISCLOSURE = (
    "Rung 1 holds out a COMPLETE STRUCTURAL UNIT -- a habitat under LOHO, a Fano "
    "point under LOFPO -- exactly as 009.02 did, and the held-out Events never "
    "appear as positive forced thirds in training. It does NOT hold out the three "
    "abstract pp values: the output alphabet has only three members and they "
    "necessarily recur in every habitat. 009.05 section 3 requires this to be said "
    "explicitly. Rung 1 therefore tests TRANSFER OF THE LOCAL LAW to an unseen "
    "structural unit, not prediction of an unseen class."
)

RUNG1_QUOTIENT_WARNING = (
    "The Rung-1 label is a function of the unordered pair {pp1, pp2} alone, and "
    "under the exact code that pair takes only three values. A three-row lookup "
    "table fitted on training data is therefore a COMPLETE representation of the "
    "local law, and ladder_baselines.py measures exactly that ceiling. At a "
    "quotient this small there is no observable difference between inferring the "
    "law and tabulating it, so Rung 1 can confirm acquisition but cannot carry "
    "structural evidence in either direction. That is the same smallness that "
    "makes 009.04's B > C condition invalid, stated as a positive limit."
)

PINS = {
    "events": 84,
    "habitats": 14,
    "ordered_pairs_total": 7056,
    "admit": 336,
    "pp_classes": 3,
    "structured_field_slots": 16,
    "structured_slots_with_bottom": STRUCTURED_SLOTS,
    "distinct_unordered_pp_pairs_exact_code": 3,
    "structural_folds": 21,
    "loho_folds": 14,
    "lofpo_folds": 7,
    "loho_test_admit": 24,
    "lofpo_test_admit": 48,
    "loho_train_admit": 312,
    "lofpo_train_admit": 288,
}


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


# ---------------------------------------------------------------------------
# code tables
# ---------------------------------------------------------------------------

def code_tables(dataset: Dataset) -> dict[str, tuple[SfpAddress, ...]]:
    """Per-arm code tables, recovered from the arms' emitted token tensors.

    Decoded with ``arms.codes_of_arm`` rather than read from the builders, for the
    same reason ``arms.py`` does it: the code the target is written in must be
    provably the code the learner actually sees, not the code the construction
    record claims it built.
    """

    tables: dict[str, tuple[SfpAddress, ...]] = {}
    for arm in all_arms(dataset, include_e=False):
        if arm.name == "A_native":
            continue
        tables[arm.name] = codes_of_arm(arm)
    missing = [name for name in CODE_ARMS if name not in tables]
    if missing:
        raise AssertionError(f"no code table recovered for {missing}")
    for name, table in tables.items():
        if len(table) != PINS["events"]:
            raise AssertionError(f"{name} code table holds {len(table)} entries")
        if len({code.as_key() for code in table}) != PINS["events"]:
            raise AssertionError(f"{name} code table is not injective")
    return tables


def output_table(
    arm_name: str, tables: dict[str, tuple[SfpAddress, ...]]
) -> tuple[SfpAddress, ...]:
    """The code table that names ``arm_name``'s structured output alphabet."""

    return tables[ARM_OUTPUT_CONVENTION[arm_name]]


# ---------------------------------------------------------------------------
# RUNG 1 TARGET -- local pp forcing port
# ---------------------------------------------------------------------------

def rung1_target(record: PairRecord, table: tuple[SfpAddress, ...]) -> int:
    """The local forced-third port ``pp3`` for one admitted pair.

    The signature is the fence: a record and a code table. There is no parameter
    through which ``sfp.ExactSfpCircuit`` -- or any other representation under
    test -- could supply the answer. ``record.target_index`` is the certified
    forced-third Event that ``task.label_pair`` obtained from
    ``fips_basic.third``; this function only reads off which local port that
    already-certified Event occupies in the arm's own code.
    """

    if not record.admitted:
        raise ValueError("Rung 1 is conditioned on admitted pairs only")
    if record.target_index == BOTTOM_INDEX:
        raise AssertionError("an admitted record carries the bottom sentinel")
    d = table[record.target_index].d
    if d not in PP_CLASSES:
        raise AssertionError(f"pp3 = {d} is not a nonzero local port")
    return d


def rung1_class(d: int) -> int:
    """Zero-based class index of a local port, for a 3-way head."""

    return PP_CLASSES.index(d)


# ---------------------------------------------------------------------------
# RUNG 2 TARGET -- the whole structured consequence address
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredTarget:
    """One Rung-2 target: explicit BOTTOM, or a full four-field address.

    ``pp_set`` holds BOTH valid PP values. They are the two block-incidence
    presentations of one Event; see :data:`PP_SET_TARGET_CONVENTION`.
    """

    is_bottom: bool
    s: int | None
    fff: int | None
    pp_set: tuple[int, ...]
    d: int | None
    event_index: int

    def __post_init__(self) -> None:
        if self.is_bottom:
            if (self.s, self.fff, self.d) != (None, None, None):
                raise AssertionError("a BOTTOM target carries address fields")
            if self.pp_set != ():
                raise AssertionError("a BOTTOM target carries PP values")
            if self.event_index != BOTTOM_INDEX:
                raise AssertionError("a BOTTOM target is not the bottom sentinel")
            return
        if self.s is None or self.fff is None or self.d is None:
            raise AssertionError("an address target is missing a field")
        if len(self.pp_set) != 2 or self.pp_set != tuple(sorted(self.pp_set)):
            raise AssertionError("PP set must be the two sorted endpoint values")
        if self.pp_set[1] != self.pp_set[0] ^ self.d:
            raise AssertionError("PP set is not (q, q XOR pp)")
        if self.event_index == BOTTOM_INDEX:
            raise AssertionError("an address target carries the bottom sentinel")

    @property
    def address(self) -> SfpAddress:
        """The canonical address this target names."""

        if self.is_bottom:
            raise ValueError("a BOTTOM target names no address")
        return address_of(self.s, self.fff, self.pp_set[0], self.d)

    def as_json(self) -> dict:
        return {
            "is_bottom": self.is_bottom,
            "S": self.s,
            "FFF": None if self.fff is None else f"{self.fff:03b}",
            "PP_set": [bits2(q) for q in self.pp_set],
            "pp": None if self.d is None else bits2(self.d),
            "event_index": self.event_index,
            "symbol": BOTTOM_SYMBOL if self.is_bottom else str(self.event_index),
        }


BOTTOM_TARGET = StructuredTarget(
    is_bottom=True,
    s=None,
    fff=None,
    pp_set=(),
    d=None,
    event_index=BOTTOM_INDEX,
)


def rung2_target(
    record: PairRecord, table: tuple[SfpAddress, ...]
) -> StructuredTarget:
    """The structured consequence target for one ordered pair.

    Same fence as :func:`rung1_target`: a record and a code table, no circuit
    parameter. Non-admission maps to the explicit :data:`BOTTOM_TARGET`, never to
    ``pp=00``, never to ``FFF=000`` and never to Event index 0.
    """

    if not record.admitted:
        if record.target_index != BOTTOM_INDEX:
            raise AssertionError("a non-admission record carries an Event target")
        return BOTTOM_TARGET
    address = table[record.target_index]
    return StructuredTarget(
        is_bottom=False,
        s=address.s,
        fff=address.fff,
        pp_set=tuple(sorted(address.endpoints)),
        d=address.d,
        event_index=record.target_index,
    )


def address_matches(
    target: StructuredTarget, s: int, fff: int, q: int, d: int
) -> bool:
    """Does a predicted four-field address name the target Event?

    The single documented test: canonicalize the prediction through
    ``sfp.address_of`` and compare keys. :func:`verify_address_equivalence` proves
    this is the same relation as "S, FFF, pp exact and PP in the endpoint set", so
    the convention adds no slack of its own.
    """

    if target.is_bottom:
        return False
    if d not in PP_CLASSES:
        return False
    return address_of(s, fff, q, d).as_key() == target.address.as_key()


# ---------------------------------------------------------------------------
# split views
# ---------------------------------------------------------------------------

def rung1_positions(dataset: Dataset, fold) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Rung-1 train / test positions: the fold's own buckets, admitted only.

    Rung 1 is conditioned on admitted pairs, which is the option ``009.05``
    section 4.2 names first. The admission gate is not removed from the ladder --
    it is Rung 2's job, where it is scored as an explicit BOTTOM class.
    """

    admit = set(dataset.admit_positions)
    train = tuple(sorted(set(fold.train) & admit))
    test = tuple(sorted(set(fold.test_within) & admit))
    if not train or not test:
        raise AssertionError(f"fold {fold.name} has an empty Rung-1 bucket")
    return train, test


def rung2_positions(dataset: Dataset, fold) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Rung-2 train / test positions: byte-identical to the ``009.02`` protocol.

    Reuses ``harness.training_positions`` and ``harness.evaluation_positions``
    unchanged, so the only thing that differs from the failed 84-way run is the
    OUTPUT OBJECT. That is the whole causal question ``009.03`` raised.
    """

    train, _ = training_positions(dataset, fold)
    test, _ = evaluation_positions(dataset, fold)
    return train, test


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_label_source_fence() -> dict:
    """Mechanically: no target function can be handed a circuit.

    ``task.py`` states that its label signature *is* the fence. This makes the
    same claim checkable: every parameter of every target function is inspected,
    and any parameter whose name or annotation mentions a circuit or an oracle is
    an offender.
    """

    banned = ("circuit", "oracle", "sfp_circuit", "ExactSfpCircuit")
    rows = []
    offenders = []
    for func in (rung1_target, rung2_target, address_matches):
        signature = inspect.signature(func)
        parameters = []
        for name, parameter in signature.parameters.items():
            annotation = (
                "" if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation)
            )
            suspect = any(
                token.lower() in f"{name} {annotation}".lower() for token in banned
            )
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
    return {
        "target_functions": rows,
        "banned_parameter_tokens": list(banned),
        "offenders": offenders,
        "no_target_function_accepts_a_circuit": not offenders,
        "label_provenance": (
            "every target is a function of record.target_index, which task.py "
            "derived from topographo.ssd.fips_basic.third and verified with "
            "topographo.ssd.projective.equivalent, composed with the arm's own "
            "code table"
        ),
        "circuit_role_in_this_module": (
            "sfp.ExactSfpCircuit is imported for the post-hoc structural reports "
            "local_law_carriage and habitat_locality only, exactly as arms.py "
            "imports it for relation_report. It generates nothing."
        ),
    }


def fold_manifest_digest(dataset: Dataset, folds: tuple) -> str:
    """Recompute ``folds.json``'s ``digests.manifest`` from the live fold objects.

    Built from ``folds_module``'s own constants rather than a copied literal, so
    the check would break if the split policy or its fences ever drifted, which is
    the point of pinning it here at all.
    """

    return digest(
        {
            "fences": list(folds_module.FENCES),
            "sampling_policy": folds_module.SAMPLING_POLICY,
            "catalogue_sha256": dataset.catalogue.sha256(),
            "dataset_sha256": dataset.sha256(),
            "folds": [[fold.family, fold.name, fold.sha256()] for fold in folds],
        }
    )


def verify_upstream_pins(dataset: Dataset, folds: tuple) -> dict:
    """The reused frozen chain still hashes to what ``009.02`` recorded."""

    codec = SfpCodec()
    manifest = fold_manifest_digest(dataset, folds)
    checks = {
        "catalogue": pin(
            EXPECTED_CATALOGUE_SHA256, dataset.catalogue.sha256(), "task.py"
        ),
        "dataset": pin(EXPECTED_DATASET_SHA256, dataset.sha256(), "task.py"),
        "fold_manifest": pin(EXPECTED_FOLD_MANIFEST_SHA256, manifest, "folds.py"),
        "sfp_chart": pin(EXPECTED_CHART_SHA256, codec.chart()["sha256"], "sfp.py"),
    }
    return {
        "checks": checks,
        "all_upstream_pins_agree": all(row["agrees"] for row in checks.values()),
        "statement": (
            "the pool, the splits and the chart are the frozen 009.02 objects, "
            "byte-identical; this ladder changes only the output object"
        ),
    }


def verify_rung1_targets(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """Rung-1 label census, and the size of the local quotient, per arm."""

    admitted = [dataset.records[p] for p in dataset.admit_positions]
    rows = {}
    for name in CODE_ARMS:
        table = tables[name]
        labels = [rung1_target(record, table) for record in admitted]
        pair_keys: dict[tuple[int, ...], set[int]] = defaultdict(set)
        for record, label in zip(admitted, labels):
            key = tuple(
                sorted((table[record.a_index].d, table[record.b_index].d))
            )
            pair_keys[key].add(label)
        deterministic = all(len(v) == 1 for v in pair_keys.values())
        rows[name] = {
            "labelled_pairs": len(labels),
            "label_marginal": {
                bits2(d): sum(1 for label in labels if label == d)
                for d in PP_CLASSES
            },
            "distinct_unordered_pp_pairs": len(pair_keys),
            "unordered_pp_pairs": [list(key) for key in sorted(pair_keys)],
            "pp_pair_determines_pp3": deterministic,
            "quotient_size_note": (
                "the number of distinct learner-visible pp pairs IS the size of "
                "the local quotient; a lookup table with this many rows is a "
                "complete representation of the local law"
            ),
        }
    return {
        "per_arm": rows,
        "all_labels_are_nonzero_ports": True,
        "exact_code_quotient_is_three": (
            rows["B_sfp"]["distinct_unordered_pp_pairs"]
            == PINS["distinct_unordered_pp_pairs_exact_code"]
        ),
        "exact_code_pp_pair_determines_pp3": rows["B_sfp"]["pp_pair_determines_pp3"],
        "quotient_warning": RUNG1_QUOTIENT_WARNING,
        "holdout_disclosure": RUNG1_HOLDOUT_DISCLOSURE,
    }


def verify_rung2_targets(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """Rung-2 target census: BOTTOM stays explicit, fields stay in range."""

    rows = {}
    for name in SCIENCE_ARMS:
        table = output_table(name, tables)
        bottom = 0
        address = 0
        field_marginals = {field: Counter() for field, _ in FIELD_CARDINALITIES}
        for record in dataset.records:
            target = rung2_target(record, table)
            if target.is_bottom:
                bottom += 1
                continue
            address += 1
            field_marginals["S"][target.s] += 1
            field_marginals["FFF"][target.fff] += 1
            for q in target.pp_set:
                field_marginals["PP"][q] += 1
            field_marginals["pp"][target.d] += 1
        rows[name] = {
            "output_alphabet_from": ARM_OUTPUT_CONVENTION[name],
            "bottom_targets": bottom,
            "address_targets": address,
            "field_marginals": {
                field: {str(k): v for k, v in sorted(counter.items())}
                for field, counter in sorted(field_marginals.items())
            },
            "pp_set_size_always_two": True,
        }
    marginal_digests = {
        name: digest(row["field_marginals"]) for name, row in rows.items()
    }
    code_arm_digests = sorted({marginal_digests[name] for name in CODE_ARMS})
    return {
        "per_arm": rows,
        "field_cardinalities": {field: width for field, width in FIELD_CARDINALITIES},
        "structured_slots_with_bottom": STRUCTURED_SLOTS,
        "never_eighty_four": STRUCTURED_SLOTS != PINS["events"],
        "bottom_targets_agree_with_pool": all(
            row["bottom_targets"] == PINS["ordered_pairs_total"] - PINS["admit"]
            for row in rows.values()
        ),
        "address_targets_agree_with_pool": all(
            row["address_targets"] == PINS["admit"] for row in rows.values()
        ),
        "marginal_digests": marginal_digests,
        "code_arms_share_field_marginals": len(code_arm_digests) == 1,
        "marginal_matching_statement": (
            "B, C and D draw their targets from the same 84 code words, so the "
            "Rung-2 output field marginals are IDENTICAL across the three; any "
            "B-vs-C gap therefore cannot be a marginal or cardinality artifact"
        ),
        "pp_set_convention": PP_SET_TARGET_CONVENTION,
        "arm_a_asymmetry": ARM_A_ASYMMETRY,
    }


def verify_address_equivalence(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """The canonicalization test equals the fieldwise test, over every candidate.

    Checked exhaustively: every admitted pair against every one of the
    ``2 * 7 * 4 * 3 = 168`` syntactically possible four-field predictions. If the
    two definitions ever disagreed, "exact structured address" would silently
    mean something other than "named the certified Event".
    """

    table = tables["B_sfp"]
    admitted = [dataset.records[p] for p in dataset.admit_positions]
    candidates = [
        (s, fff, q, d)
        for s in range(2)
        for fff in range(1, 8)
        for q in range(4)
        for d in PP_CLASSES
    ]
    disagreements = []
    accepted_per_target = Counter()
    for record in admitted:
        target = rung2_target(record, table)
        accepted = 0
        for s, fff, q, d in candidates:
            canonical = address_matches(target, s, fff, q, d)
            fieldwise = (
                s == target.s
                and fff == target.fff
                and d == target.d
                and q in target.pp_set
            )
            if canonical != fieldwise:
                if len(disagreements) < 4:
                    disagreements.append(
                        {
                            "a": record.a_index,
                            "b": record.b_index,
                            "prediction": [s, fff, q, d],
                            "canonical": canonical,
                            "fieldwise": fieldwise,
                        }
                    )
            if canonical:
                accepted += 1
        accepted_per_target[accepted] += 1
    return {
        "admitted_targets_checked": len(admitted),
        "candidate_addresses_per_target": len(candidates),
        "comparisons": len(admitted) * len(candidates),
        "disagreements": disagreements,
        "canonical_test_equals_fieldwise_test": not disagreements,
        "accepted_predictions_per_target": {
            str(k): v for k, v in sorted(accepted_per_target.items())
        },
        "exactly_two_accepted_per_target": sorted(accepted_per_target) == [2],
        "statement": (
            "exactly two of the 168 syntactic predictions are accepted for each "
            "target, and they are the Event's two block-incidence presentations; "
            "the convention grants no slack beyond the codec's own symmetry"
        ),
    }


def local_law_carriage(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """How much of the certified local law each arm's code still carries.

    Reported, never supplied. ``009.05`` section 2 forbids giving the learner the
    executable rule ``pp3 = pp1 XOR pp2``; measuring how often each arm's code
    satisfies it is a property of the frozen code tables, computed here with no
    model in scope.
    """

    circuit = ExactSfpCircuit()
    admitted = [dataset.records[p] for p in dataset.admit_positions]
    rows = {}
    for name in CODE_ARMS:
        table = tables[name]
        xor_holds = 0
        shared_unique = 0
        shared_in_target = 0
        ports_distinct = 0
        for record in admitted:
            a, b = table[record.a_index], table[record.b_index]
            target = table[record.target_index]
            if a.d != b.d:
                ports_distinct += 1
            if a.d ^ b.d == target.d:
                xor_holds += 1
            endpoint = circuit.shared_block(a, b)
            if endpoint is not None:
                shared_unique += 1
                if endpoint in target.endpoints:
                    shared_in_target += 1
        rows[name] = {
            "admitted_pairs": len(admitted),
            "input_ports_distinct": ports_distinct,
            "pp3_equals_xor_of_input_ports": xor_holds,
            "unique_shared_endpoint_in_code_space": shared_unique,
            "shared_endpoint_lies_in_target_address": shared_in_target,
            "carries_the_local_law_exactly": xor_holds == len(admitted),
        }
    return {
        "per_arm": rows,
        "oracle": (
            "sfp.ExactSfpCircuit as a nonlearned reference oracle for a structural "
            "READ-OUT; never a label source and never shown to a learner"
        ),
        "statement": (
            "the exact and relabeled codes satisfy the certified local law on all "
            "336 admitted pairs; the scramble satisfies it on strictly fewer, and "
            "its input ports are not even always distinct, because a "
            "non-automorphic edge permutation can carry two Events that share a "
            "block onto two Events that do not"
        ),
    }


def habitat_locality(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]]
) -> dict:
    """Is the code-space consequence map habitat-independent?

    This is the mechanism the Rung-2 structural prediction rests on, measured
    before any training. Key each admitted pair on its purely local code
    structure -- the unordered input port pair and the two endpoint sets -- and
    ask whether that key determines the target's local structure across all 14
    habitats. If it does, a learner that has seen other habitats can transfer. If
    the same key resolves differently in different habitats, holding a habitat out
    removes information no amount of learning can recover.
    """

    rows = {}
    for name in CODE_ARMS:
        table = tables[name]
        answers: dict[tuple, set[tuple]] = defaultdict(set)
        habitats_per_key: dict[tuple, set[tuple[int, int]]] = defaultdict(set)
        for position in dataset.admit_positions:
            record = dataset.records[position]
            a, b = table[record.a_index], table[record.b_index]
            target = table[record.target_index]
            key = (
                tuple(sorted((a.d, b.d))),
                tuple(sorted(a.endpoints)),
                tuple(sorted(b.endpoints)),
            )
            answers[key].add((target.d, tuple(sorted(target.endpoints))))
            habitats_per_key[key].add(record.habitat_a)
        ambiguous = sorted(key for key, v in answers.items() if len(v) > 1)
        rows[name] = {
            "local_structure_keys": len(answers),
            "habitat_dependent_keys": len(ambiguous),
            "habitat_independent": not ambiguous,
            "max_distinct_answers_for_one_key": max(
                len(v) for v in answers.values()
            ),
            "keys_spanning_multiple_habitats": sum(
                1 for v in habitats_per_key.values() if len(v) > 1
            ),
        }
    return {
        "per_arm": rows,
        "key_definition": (
            "unordered input port pair, plus both input endpoint sets; a purely "
            "local, habitat-free description of the code-space configuration"
        ),
        "structural_prediction": (
            "B and D resolve every local key identically in every habitat, so a "
            "held-out habitat is reachable by transfer. C resolves keys "
            "habitat-dependently, so leave-one-habitat-out removes the only "
            "evidence for that habitat's permutation. This is the pre-registered "
            "mechanism for the Rung-2 B > C prediction, and it is a fact about the "
            "frozen codes rather than about any learner."
        ),
        "not_a_rung_1_argument": (
            "this locality gap lives in the full six-edge / four-chamber structure. "
            "It is invisible at the Rung-1 quotient, which is exactly why 009.05 "
            "moves the structural test to Rung 2."
        ),
    }


def split_census(dataset: Dataset, folds: tuple) -> dict:
    """Sizes and digests of every Rung-1 and Rung-2 split, per fold."""

    rows = []
    for fold in folds:
        r1_train, r1_test = rung1_positions(dataset, fold)
        r2_train, r2_test = rung2_positions(dataset, fold)
        rows.append(
            {
                "family": fold.family,
                "name": fold.name,
                "held_out_events": list(fold.held_out_events),
                "rung1": {
                    "train_size": len(r1_train),
                    "test_size": len(r1_test),
                    "train_sha256": digest(list(r1_train)),
                    "test_sha256": digest(list(r1_test)),
                },
                "rung2": {
                    "train_size": len(r2_train),
                    "test_size": len(r2_test),
                    "train_classes": class_counts(dataset, r2_train),
                    "test_classes": class_counts(dataset, r2_test),
                    "train_sha256": digest(list(r2_train)),
                    "test_sha256": digest(list(r2_test)),
                },
            }
        )
    loho = [row for row in rows if row["family"] == "LOHO"]
    lofpo = [row for row in rows if row["family"] == "LOFPO"]
    return {
        "folds": rows,
        "n_folds": len(rows),
        "loho_folds": len(loho),
        "lofpo_folds": len(lofpo),
        "loho_rung1_test_sizes": sorted({row["rung1"]["test_size"] for row in loho}),
        "loho_rung1_train_sizes": sorted({row["rung1"]["train_size"] for row in loho}),
        "lofpo_rung1_test_sizes": sorted({row["rung1"]["test_size"] for row in lofpo}),
        "lofpo_rung1_train_sizes": sorted(
            {row["rung1"]["train_size"] for row in lofpo}
        ),
        "rung2_protocol": (
            "harness.training_positions / harness.evaluation_positions, unchanged "
            "from 009.02"
        ),
        "digest": digest(rows),
    }


def verify_leakage(dataset: Dataset, folds: tuple, tables: dict) -> dict:
    """No held-out Event appears as a positive target in any training split.

    Recomputed from the recorded index lists rather than argued from
    construction, matching ``folds.verify_leakage``. Checked at both rungs and,
    at Rung 2, in the arm's own output alphabet as well as by Event index.
    """

    rows = []
    clean = True
    for fold in folds:
        held = set(fold.held_out_events)
        r1_train, r1_test = rung1_positions(dataset, fold)
        r2_train, r2_test = rung2_positions(dataset, fold)

        r1_train_targets = {
            dataset.records[p].target_index for p in r1_train
        }
        r2_train_targets = {
            dataset.records[p].target_index
            for p in r2_train
            if dataset.records[p].admitted
        }
        r1_train_inputs = {
            index
            for p in r1_train
            for index in (dataset.records[p].a_index, dataset.records[p].b_index)
        }
        r2_train_inputs = {
            index
            for p in r2_train
            for index in (dataset.records[p].a_index, dataset.records[p].b_index)
        }
        r1_test_targets = {dataset.records[p].target_index for p in r1_test}

        address_leak = {}
        for name in SCIENCE_ARMS:
            table = output_table(name, tables)
            held_addresses = {table[index].as_key() for index in held}
            train_addresses = {
                table[t].as_key() for t in r2_train_targets
            }
            address_leak[name] = len(held_addresses & train_addresses)

        row = {
            "family": fold.family,
            "name": fold.name,
            "held_out_events": sorted(held),
            "rung1_train_target_leak": sorted(r1_train_targets & held),
            "rung1_train_input_leak": sorted(r1_train_inputs & held),
            "rung2_train_target_leak": sorted(r2_train_targets & held),
            "rung2_train_input_leak": sorted(r2_train_inputs & held),
            "rung2_train_address_leak_by_arm": address_leak,
            "rung1_test_targets_are_held_out": sorted(r1_test_targets) == sorted(held)
            or r1_test_targets <= held,
            "rung1_test_target_count": len(r1_test_targets),
            "buckets_disjoint": not (set(r1_train) & set(r1_test))
            and not (set(r2_train) & set(r2_test)),
        }
        if (
            row["rung1_train_target_leak"]
            or row["rung1_train_input_leak"]
            or row["rung2_train_target_leak"]
            or row["rung2_train_input_leak"]
            or any(address_leak.values())
            or not row["buckets_disjoint"]
        ):
            clean = False
        rows.append(row)
    return {
        "per_fold": rows,
        "no_leakage_anywhere": clean,
        "checked": (
            "held-out Events as training targets, as training inputs, and as "
            "training output CODE WORDS in every arm's alphabet, plus bucket "
            "disjointness, at both rungs"
        ),
        "why_addresses_too": (
            "an Event held out by index could still leak through its code word if "
            "two arms disagreed about the assignment; the address check closes "
            "that door for C and D as well as B"
        ),
        "all_rung1_test_targets_are_held_out_events": all(
            row["rung1_test_targets_are_held_out"] for row in rows
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    folds = structural_folds(all_folds(dataset))
    tables = code_tables(dataset)
    codec = SfpCodec()
    native = native_codes(dataset.catalogue, codec)

    upstream = verify_upstream_pins(dataset, all_folds(dataset))
    fence = verify_label_source_fence()
    rung1 = verify_rung1_targets(dataset, tables)
    rung2 = verify_rung2_targets(dataset, tables)
    equivalence = verify_address_equivalence(dataset, tables)
    carriage = local_law_carriage(dataset, tables)
    locality = habitat_locality(dataset, tables)
    splits = split_census(dataset, folds)
    leakage = verify_leakage(dataset, folds, tables)

    exact_is_native = all(
        b.as_key() == n.as_key() for b, n in zip(tables["B_sfp"], native)
    )

    pins = {
        "structural_folds": pin(
            PINS["structural_folds"], splits["n_folds"], "folds.structural_folds"
        ),
        "loho_folds": pin(PINS["loho_folds"], splits["loho_folds"], "folds.py"),
        "lofpo_folds": pin(PINS["lofpo_folds"], splits["lofpo_folds"], "folds.py"),
        "loho_rung1_test_admit": pin(
            [PINS["loho_test_admit"]], splits["loho_rung1_test_sizes"], "folds.LOHO_PINS"
        ),
        "loho_rung1_train_admit": pin(
            [PINS["loho_train_admit"]],
            splits["loho_rung1_train_sizes"],
            "folds.LOHO_PINS",
        ),
        "lofpo_rung1_test_admit": pin(
            [PINS["lofpo_test_admit"]],
            splits["lofpo_rung1_test_sizes"],
            "folds.LOFPO_PINS",
        ),
        "lofpo_rung1_train_admit": pin(
            [PINS["lofpo_train_admit"]],
            splits["lofpo_rung1_train_sizes"],
            "folds.LOFPO_PINS",
        ),
        "pp_classes": pin(PINS["pp_classes"], len(PP_CLASSES), "sfp.V2_NONZERO"),
        "structured_slots": pin(
            PINS["structured_slots_with_bottom"], STRUCTURED_SLOTS, "this module"
        ),
        "local_quotient_size": pin(
            PINS["distinct_unordered_pp_pairs_exact_code"],
            rung1["per_arm"]["B_sfp"]["distinct_unordered_pp_pairs"],
            "measured on the exact code",
        ),
    }

    laws = {
        "upstream_pins_agree": upstream["all_upstream_pins_agree"],
        "no_target_function_accepts_a_circuit": fence[
            "no_target_function_accepts_a_circuit"
        ],
        "exact_code_table_is_the_native_codec": exact_is_native,
        "rung1_labels_live_on_three_nonzero_ports": rung1[
            "exact_code_pp_pair_determines_pp3"
        ],
        "rung1_local_quotient_is_three_rows": rung1["exact_code_quotient_is_three"],
        "rung2_output_is_not_the_catalogue_interface": rung2["never_eighty_four"],
        "rung2_bottom_stays_explicit": rung2["bottom_targets_agree_with_pool"],
        "rung2_address_targets_agree_with_pool": rung2[
            "address_targets_agree_with_pool"
        ],
        "code_arms_share_output_field_marginals": rung2[
            "code_arms_share_field_marginals"
        ],
        "canonical_test_equals_fieldwise_test": equivalence[
            "canonical_test_equals_fieldwise_test"
        ],
        "exactly_two_presentations_accepted_per_target": equivalence[
            "exactly_two_accepted_per_target"
        ],
        "exact_code_carries_the_local_law": carriage["per_arm"]["B_sfp"][
            "carries_the_local_law_exactly"
        ],
        "relabeled_code_carries_the_local_law": carriage["per_arm"]["D_relabeled"][
            "carries_the_local_law_exactly"
        ],
        "scramble_does_not_carry_the_local_law": not carriage["per_arm"][
            "C_scrambled"
        ]["carries_the_local_law_exactly"],
        "exact_code_consequence_map_is_habitat_independent": locality["per_arm"][
            "B_sfp"
        ]["habitat_independent"],
        "relabeled_consequence_map_is_habitat_independent": locality["per_arm"][
            "D_relabeled"
        ]["habitat_independent"],
        "scramble_consequence_map_is_habitat_dependent": not locality["per_arm"][
            "C_scrambled"
        ]["habitat_independent"],
        "no_leakage_anywhere": leakage["no_leakage_anywhere"],
        "all_rung1_test_targets_are_held_out_events": leakage[
            "all_rung1_test_targets_are_held_out_events"
        ],
        "all_pins_agree": all(row["agrees"] for row in pins.values()),
    }

    broken = sorted(name for name, ok in laws.items() if not ok)
    payload = {
        "module": "ladder_task",
        "purpose": (
            "the Rung-1 local forcing port and the Rung-2 structured consequence "
            "address, built on the frozen 009.02 pool, splits, chart and arms"
        ),
        "executes": EXECUTED_TASK,
        "supersedes_before_execution": SUPERSEDED_TASK,
        "fences": list(FENCES),
        "pins": pins,
        "upstream_pins": upstream,
        "label_source_fence": fence,
        "rung1_targets": rung1,
        "rung2_targets": rung2,
        "address_equivalence": equivalence,
        "local_law_carriage": carriage,
        "habitat_locality": locality,
        "splits": splits,
        "leakage": leakage,
        "arm_output_convention": dict(sorted(ARM_OUTPUT_CONVENTION.items())),
        "digests": {
            "catalogue": dataset.catalogue.sha256(),
            "dataset": dataset.sha256(),
            "splits": splits["digest"],
            "code_tables": {
                name: digest([code.as_json() for code in table])
                for name, table in sorted(tables.items())
            },
        },
        "provenance": {
            "base_commit": BASE_COMMIT,
            "reuses": [
                "task.py build_dataset",
                "folds.py all_folds / structural_folds",
                "arms.py all_arms / codes_of_arm",
                "harness.py training_positions / evaluation_positions",
                "sfp.py SfpCodec / address_of",
            ],
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - Rung-1 and Rung-2 targets derive from the certified native "
                "relation alone, BOTTOM stays explicit, the structured output is 17 "
                "slots rather than 84, B/C/D share output field marginals exactly, "
                "and the pre-registered Rung-2 mechanism is recorded: the exact and "
                "relabeled consequence maps are habitat-independent while the "
                "scramble's is not"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }
    return payload


def _report(result: dict) -> None:
    print(f"executed:  {result['executes']}", flush=True)
    print(f"superseded: {result['supersedes_before_execution']}", flush=True)
    rung1 = result["rung1_targets"]["per_arm"]["B_sfp"]
    print(
        f"Rung 1: {rung1['labelled_pairs']} admitted pairs, "
        f"{rung1['distinct_unordered_pp_pairs']}-row local quotient",
        flush=True,
    )
    print(
        f"Rung 2: {result['rung2_targets']['structured_slots_with_bottom']} output "
        "slots (16 field + 1 BOTTOM), not 84",
        flush=True,
    )
    for name, row in sorted(result["habitat_locality"]["per_arm"].items()):
        print(
            f"  {name:<13} local keys {row['local_structure_keys']:>3}  "
            f"habitat-dependent {row['habitat_dependent_keys']:>3}",
            flush=True,
        )
    print(f"splits digest: {result['digests']['splits']}", flush=True)


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
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches ladder_task.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()

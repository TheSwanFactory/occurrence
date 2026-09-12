"""009.07 Rung-0: the certified query-relative shared chamber ``q*``.

``009.06`` asked the learner to emit ``PP`` through an absolute four-class head and
reached ``0.7158`` on it, which turned out to be the binding constraint on the whole
structured consequence (``0.9762 x 0.7158 = 0.6987`` against an observed ``0.7016``).
``009.07`` tests whether that residual is another output-constitution mismatch: for
an admitted pair of Event edges the shared chamber is not an absolute label, it is
the **unique chamber incident to both inputs**, so it should be asked as a
query-relative candidate-selection question.

This module defines that target and nothing else. It is the label authority for the
whole turn, and it derives ``q*`` from the **certified** FIPS relation:

``task.PairRecord.shared_block`` is the certified cyclic block id that
``task.label_pair`` obtained from ``task.Catalogue.shared_blocks``, which is a set
intersection of ``Catalogue.block_ids`` -- built from ``topographo.ssd.fips_basic``.
The block id is then converted to a chamber coordinate through the frozen ``sfp.py``
chart, and transported into each arm's own output alphabet by that arm's **declared**
code map. ``sfp.ExactSfpCircuit`` is imported for post-hoc structural read-out only,
exactly as ``arms.py`` and ``ladder_task.py`` import it, and generates nothing.

Everything upstream -- the catalogue, the 7056-pair dataset, the 22 fold manifests,
the four arms, the chart -- is the frozen ``009.02``/``009.06`` object, reused by
import and re-pinned by digest here.

Torch-free in intent; it inherits torch transitively through ``ladder_task`` ->
``harness``, and says so rather than claiming otherwise.
"""

import argparse
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path

import arms as arms_module
from arms import ARM_D_AFFINE, native_codes, relabel_address
from folds import all_folds, structural_folds
from ladder_task import (
    ARM_A_ASYMMETRY,
    ARM_OUTPUT_CONVENTION,
    CODE_ARMS,
    PP_CLASSES,
    SCIENCE_ARMS,
    code_tables,
    output_table,
    rung1_positions,
    rung2_positions,
    rung2_target,
    verify_leakage,
    verify_upstream_pins,
)
from sfp import ExactSfpCircuit, SfpAddress, SfpCodec, address_of, bits2, block_key
from task import Dataset, PairRecord, build_dataset, digest

from topographo.core import f2_groups as groups

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_locator_artifacts"
OUTPUT = ARTIFACTS / "locator_task.json"

#: The frozen ``009.06`` result commit this turn builds on.
BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

EXECUTED_TASK = "009.07-Kiro-query-relative-PP-localization-task.md"
PRIOR_RESULT = "009.06-Kiro-corrected-structured-consequence-address-ladder-result.md"

#: The four chambers of one affine chart. ``PP = 00`` is the declared ORIGIN
#: BLOCK -- a legitimate occurring value, not a formal completion value.
CHAMBER_VALUES = (0, 1, 2, 3)

#: Frozen ``009.06`` reference numbers, quoted from its result turn and
#: re-verified against ``009_ladder_artifacts/ladder_sweep.json`` when that bulk
#: artifact is present locally.
PRIOR_009_06 = {
    "block": "repair",
    "field_PP_accuracy": {"LOHO": 0.7158, "LOFPO": 0.7054},
    "field_pp_accuracy": {"LOHO": 0.9762},
    "positive_forced_third_exact_accuracy": {"LOHO": 0.7016},
    "criterion": (
        "009.06's field_PP_accuracy is a SET criterion: an argmax landing on either "
        "of the target Event's two block-incidence presentations was scored correct. "
        "It is therefore NOT the same quantity as q*-exact accuracy, which is a "
        "point criterion on the single certified shared chamber. Both are reported "
        "in this turn, and only the set-relaxed one is compared to 0.7158."
    ),
}

#: Stripped before any ``--check`` comparison, and the only such key here.
#:
#: ``prior_reference`` re-derives the quoted ``009.06`` numbers from
#: ``009_ladder_artifacts/ladder_sweep.json``, which is bulk run data hosted in the
#: Quilt package and gitignored. Its content therefore depends on whether that file
#: has been fetched, so a byte comparison including it would fail in a fresh clone
#: for a reason that has nothing to do with this module. The block is kept in the
#: artifact -- it is real provenance when the file is present -- and excluded from
#: the replay comparison, exactly as ``ladder_sweep.py`` excludes wall-clock keys.
NON_REPLAYABLE_KEYS = ("prior_reference",)

FENCES = (
    "q* comes from the certified FIPS relation via record.shared_block, never from "
    "sfp.ExactSfpCircuit.",
    "algebraic zero != NONADMISSION;   0 != bottom",
    "PP=00 is the declared origin block, a real chart value, not a completion value.",
    "pp=00 and FFF=000 remain formal algebraic completion values only.",
    (
        "q* is QUERY-RELATIVE. It is not a globally preferred endpoint of the forced "
        "Event and adds no preferred origin to the affine PP torsor: both "
        "presentations (q*, d3) and (q* XOR d3, d3) name the same certified Event "
        "and the fixed resolver accepts either."
    ),
    (
        "The chamber symbols 00..11 are NOT renamed by Arm C. The scramble is a "
        "permutation of the six K4 EDGES inside a habitat, so the certified q* "
        "transports to Arm C unchanged and is allowed to be misaligned with C's "
        "coded incidence. That misalignment is the phenomenon under test, not a "
        "labelling bug."
    ),
    (
        "No unsupported Arm C example is dropped. The representational-support "
        "fraction is reported instead."
    ),
    "SFP is a representation layer, not a replacement OT evaluator.",
    (
        "The exact endpoint-intersection circuit is a nonlearned CEILING, not a "
        "competing learned baseline."
    ),
)

#: How each arm's declared code map acts on chamber symbols.
#:
#: ``B_sfp`` is the frozen chart itself, so the action is the identity.
#: ``A_native`` writes the ``B_sfp`` alphabet (``ARM_OUTPUT_CONVENTION``), so it
#: inherits that identity.
#: ``C_scrambled`` permutes the six K4 EDGES of each habitat
#: (``arms.build_arm_c`` -> ``address_of_edge(s, fff, perm[edge])``) and declares
#: no chamber renaming at all, so the action is again the identity -- and the
#: certified q* is consequently often NOT the shared endpoint of C's coded inputs.
#: ``D_relabeled`` renames chambers by the certified affine map ``q -> A q XOR b``
#: (``arms.ARM_D_AFFINE``), so the action is that map's chamber permutation. It is
#: read from ``arms``/``groups`` rather than written out as a literal.
CHAMBER_ACTION_SOURCE = {
    "A_native": "identity (writes the B_sfp alphabet)",
    "B_sfp": "identity (the frozen sfp.py chart)",
    "C_scrambled": "identity (an EDGE permutation renames no chamber)",
    "D_relabeled": "groups.chamber_permutation(arms.ARM_D_AFFINE)",
}

IDENTITY_CHAMBERS = tuple(CHAMBER_VALUES)


def chamber_actions() -> dict[str, tuple[int, ...]]:
    """Per-arm chamber permutation, derived from the frozen ``arms`` constants."""

    affine = groups.chamber_permutation(ARM_D_AFFINE)
    if sorted(affine) != list(CHAMBER_VALUES):
        raise AssertionError(f"Arm D chamber action {affine} is not a permutation")
    return {
        "A_native": IDENTITY_CHAMBERS,
        "B_sfp": IDENTITY_CHAMBERS,
        "C_scrambled": IDENTITY_CHAMBERS,
        "D_relabeled": affine,
    }


PINS = {
    "events": 84,
    "habitats": 14,
    "certified_blocks": 56,
    "ordered_pairs_total": 7056,
    "admit": 336,
    "chambers_per_chart": 4,
    "chamber_candidates": 4,
    "qstar_chance": 0.25,
    "structural_folds": 21,
    "loho_folds": 14,
    "lofpo_folds": 7,
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


def strip_non_replayable(node: object) -> object:
    """Recursively drop every key in :data:`NON_REPLAYABLE_KEYS`."""

    if isinstance(node, dict):
        return {
            key: strip_non_replayable(value)
            for key, value in node.items()
            if key not in NON_REPLAYABLE_KEYS
        }
    if isinstance(node, list):
        return [strip_non_replayable(item) for item in node]
    return node


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


def rd(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


# ---------------------------------------------------------------------------
# the certified chamber of a certified block
# ---------------------------------------------------------------------------

def certified_chamber_of_block(
    dataset: Dataset, codec: SfpCodec
) -> dict[int, tuple[tuple[int, int], int]]:
    """Certified cyclic block id -> ``((S, FFF), chamber q)`` in the frozen chart.

    ``dataset.catalogue.blocks`` are the certified blocks as triples of Event
    indices, built in ``task.build_catalogue`` from ``topographo.ssd.fips_basic``.
    ``sfp.block_key`` gives the same iteration-order-independent textual key the
    chart is indexed by, so this is a pure lookup into ``Habitat.q_of_block_key``
    and involves no relation, no circuit and no choice.

    Every block must land in exactly one habitat chart at exactly one chamber; a
    block matching zero or several charts would mean the chart and the certified
    incidence structure had drifted apart.
    """

    catalogue = dataset.catalogue
    mapping: dict[int, tuple[tuple[int, int], int]] = {}
    for bid, members in enumerate(catalogue.blocks):
        key = block_key(tuple(catalogue.events[i] for i in members))
        hits = [
            (habitat_key, habitat.q_of_block_key[key])
            for habitat_key, habitat in sorted(codec.habitats.items())
            if key in habitat.q_of_block_key
        ]
        if len(hits) != 1:
            raise AssertionError(
                f"certified block {bid} matched {len(hits)} habitat charts, not 1"
            )
        mapping[bid] = hits[0]
    if len(mapping) != PINS["certified_blocks"]:
        raise AssertionError(f"mapped {len(mapping)} blocks, not 56")
    return mapping


# ---------------------------------------------------------------------------
# the q* target
# ---------------------------------------------------------------------------

def qstar_exact(
    record: PairRecord, chamber_of_block: dict[int, tuple[tuple[int, int], int]]
) -> int:
    """The certified shared chamber of one admitted pair, in the frozen chart.

    The signature is the fence, in the same style ``task.label_pair`` and
    ``ladder_task.rung1_target`` use: a certified record and a certified
    block-to-chamber table. There is no parameter through which
    ``sfp.ExactSfpCircuit`` -- or any other representation under test -- could
    supply the answer.
    """

    if not record.admitted:
        raise ValueError("q* is conditioned on admitted pairs only")
    if record.shared_block is None:
        raise AssertionError("an admitted record carries no certified shared block")
    _, q = chamber_of_block[record.shared_block]
    if q not in CHAMBER_VALUES:
        raise AssertionError(f"certified chamber {q} is outside the chart")
    return q


def qstar_target(
    record: PairRecord,
    chamber_of_block: dict[int, tuple[tuple[int, int], int]],
    action: tuple[int, ...],
) -> int:
    """The certified shared chamber, transported into one arm's own alphabet.

    ``action`` is the arm's **declared** chamber permutation from
    :func:`chamber_actions`, read off the frozen ``arms`` constants. Same fence as
    :func:`qstar_exact`.
    """

    return action[qstar_exact(record, chamber_of_block)]


def query_relative_address(
    record: PairRecord,
    table: tuple[SfpAddress, ...],
    chamber_of_block: dict[int, tuple[tuple[int, int], int]],
    action: tuple[int, ...],
) -> SfpAddress:
    """The consequence written query-relatively as ``(S, FFF, q*, d3)``.

    ``d3`` is read off the arm's code of the certified forced-third Event, exactly
    as ``ladder_task.rung1_target`` reads it. The point of the construction is that
    this address must be the SAME canonical Event as the arm's own code of the
    target: ``sfp.address_of`` canonicalizes ``(q*, d3)`` and ``(q* XOR d3, d3)``
    into one unordered pair, so writing the consequence relative to the query adds
    no preferred origin.
    """

    target = table[record.target_index]
    return address_of(
        target.s, target.fff, qstar_target(record, chamber_of_block, action), target.d
    )


def qstar_positions(dataset: Dataset, fold) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Rung-1 train / test positions: ``ladder_task.rung1_positions`` verbatim.

    Reused rather than re-derived so the Rung-1 splits are provably the objects
    that produced the ``009.06`` Rung-1 numbers. Admitted pairs only, per
    ``009.07`` section 4.1; admission is not a target here and section 8 forbids
    quoting it as evidence.
    """

    return rung1_positions(dataset, fold)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify_label_source_fence() -> dict:
    """Mechanically: no label function in this turn can be handed a circuit.

    The same check ``ladder_task.verify_label_source_fence`` performs, extended to
    the functions ``009.07`` adds. ``009.07`` section 3.1 requires the ``q*``
    label-source audit to be explicit, so the provenance chain is spelled out
    rather than asserted.
    """

    banned = ("circuit", "oracle", "sfp_circuit", "exactsfpcircuit")
    rows = []
    offenders = []
    for func in (qstar_exact, qstar_target, query_relative_address, qstar_positions):
        signature = inspect.signature(func)
        parameters = []
        for name, parameter in signature.parameters.items():
            annotation = (
                ""
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation)
            )
            suspect = any(
                token in f"{name} {annotation}".lower() for token in banned
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
        "label_functions": rows,
        "banned_parameter_tokens": list(banned),
        "offenders": offenders,
        "no_label_function_accepts_a_circuit": not offenders,
        "qstar_provenance_chain": [
            "topographo.ssd.fips_basic.BLOCKS -- the certified cyclic blocks",
            "task.build_catalogue -> Catalogue.block_ids / Catalogue.blocks",
            "task.Catalogue.shared_blocks(a, b) -- set intersection of block_ids",
            (
                "task.label_pair asserts exactly one shared block for an admitted "
                "pair and stores it as PairRecord.shared_block"
            ),
            (
                "locator_task.certified_chamber_of_block maps that block id to a "
                "chamber through sfp.Habitat.q_of_block_key, the frozen chart"
            ),
            (
                "locator_task.qstar_target transports the chamber by the arm's "
                "DECLARED code map from arms.py"
            ),
        ],
        "circuit_role_in_this_module": (
            "sfp.ExactSfpCircuit is imported for the post-hoc structural read-outs "
            "verify_qstar_alignment and the ceiling census only, exactly as arms.py "
            "imports it for relation_report. It generates no label."
        ),
    }


def verify_chamber_map(
    dataset: Dataset, codec: SfpCodec, chamber_of_block: dict
) -> dict:
    """The block-to-chamber table is total, per-habitat bijective, and consistent.

    Each habitat has 4 certified blocks and 4 chambers, so the restriction of the
    table to one habitat must be a bijection onto ``{00, 01, 10, 11}``. If it were
    not, ``q*`` would not be a chamber coordinate at all.
    """

    per_habitat: dict[tuple[int, int], list[int]] = defaultdict(list)
    for _bid, (habitat_key, q) in sorted(chamber_of_block.items()):
        per_habitat[habitat_key].append(q)
    rows = {}
    bijective = True
    for habitat_key, chambers in sorted(per_habitat.items()):
        ok = sorted(chambers) == list(CHAMBER_VALUES)
        bijective = bijective and ok
        rows[f"S={habitat_key[0]},FFF={habitat_key[1]:03b}"] = {
            "chambers": sorted(chambers),
            "bijective_onto_the_chart": ok,
        }
    # The habitat a block sits in must be the habitat its member Events sit in.
    codes = native_codes(dataset.catalogue, codec)
    habitat_agrees = 0
    for bid, members in enumerate(dataset.catalogue.blocks):
        habitat_key, _ = chamber_of_block[bid]
        if all(codes[i].habitat == habitat_key for i in members):
            habitat_agrees += 1
    return {
        "blocks_mapped": len(chamber_of_block),
        "per_habitat": rows,
        "every_habitat_bijective_onto_its_chart": bijective,
        "blocks_whose_habitat_agrees_with_their_events": habitat_agrees,
        "block_habitat_always_agrees": habitat_agrees == len(chamber_of_block),
        "how_built": (
            "sfp.block_key of the certified block's rays, looked up in the frozen "
            "sfp.Habitat.q_of_block_key; a pure lookup with no relation in scope"
        ),
    }


def verify_qstar_census(dataset: Dataset, chamber_of_block: dict) -> dict:
    """The ``q*`` label census and the analytic chance rate."""

    admitted = [dataset.records[p] for p in dataset.admit_positions]
    actions = chamber_actions()
    rows = {}
    for name in SCIENCE_ARMS:
        action = actions[name]
        labels = [qstar_target(r, chamber_of_block, action) for r in admitted]
        counter = Counter(labels)
        rows[name] = {
            "chamber_action": list(action),
            "chamber_action_source": CHAMBER_ACTION_SOURCE[name],
            "labelled_pairs": len(labels),
            "label_marginal": {bits2(q): counter.get(q, 0) for q in CHAMBER_VALUES},
            "uniform": len(set(counter.values())) == 1,
            "distinct_labels": len(counter),
        }
    return {
        "per_arm": rows,
        "chance": PINS["qstar_chance"],
        "chance_statement": (
            "the q* marginal is exactly uniform at 84 of 336 per chamber under every "
            "arm's declared action, because a chamber permutation cannot change a "
            "uniform marginal, so chance is 1/4 analytically and not merely by "
            "sampling"
        ),
        "all_arms_uniform": all(row["uniform"] for row in rows.values()),
        "held_out_object": (
            "a complete structural unit -- a habitat under LOHO, a Fano point under "
            "LOFPO -- exactly as 009.02 and 009.06 held out. The four abstract "
            "chamber values necessarily RECUR in training: a chart has only four of "
            "them. 009.07 section 4.1 requires this to be said explicitly. The task "
            "is transfer of the localization relation to an unseen structural unit, "
            "not prediction of an unseen output class."
        ),
    }


def verify_qstar_alignment(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]], chamber_of_block: dict
) -> dict:
    """Does each arm's code SUPPORT the certified ``q*``? Reported, never enforced.

    Three code-space read-outs per arm, all computed with no model in scope and all
    using ``sfp.ExactSfpCircuit`` as a structural read-out rather than a label
    source:

    ``code_shared_endpoint_defined``  the arm's coded inputs meet in exactly one
                                      chamber at all
    ``code_shared_endpoint_is_qstar`` that chamber IS the certified ``q*`` -- the
                                      **representational-support fraction**
                                      ``009.07`` section 7 requires be reported
    ``qstar_in_coded_target``         ``q*`` is an endpoint of the arm's code of the
                                      certified forced-third Event, so the
                                      query-relative address ``(S, FFF, q*, d3)``
                                      denotes that Event

    B and D must be exact on all three; C is expected to fail and its examples are
    kept rather than dropped.
    """

    circuit = ExactSfpCircuit()
    actions = chamber_actions()
    admitted = [dataset.records[p] for p in dataset.admit_positions]
    rows = {}
    for name in SCIENCE_ARMS:
        table = output_table(name, tables)
        action = actions[name]
        defined = is_qstar = in_target = presentation_ok = 0
        for record in admitted:
            q = qstar_target(record, chamber_of_block, action)
            a, b = table[record.a_index], table[record.b_index]
            target = table[record.target_index]
            got = circuit.shared_block(a, b)
            if got is not None:
                defined += 1
                if got == q:
                    is_qstar += 1
            if q in target.endpoints:
                in_target += 1
                if (
                    address_of(target.s, target.fff, q, target.d).as_key()
                    == target.as_key()
                ):
                    presentation_ok += 1
        n = len(admitted)
        rows[name] = {
            "admitted_pairs": n,
            "code_shared_endpoint_defined": defined,
            "code_shared_endpoint_is_qstar": is_qstar,
            "qstar_in_coded_target": in_target,
            "query_relative_address_denotes_the_target": presentation_ok,
            "representational_support_fraction": rd(fraction(is_qstar, n)),
            "target_support_fraction": rd(fraction(in_target, n)),
            "fully_supported": is_qstar == in_target == presentation_ok == n,
        }
    return {
        "per_arm": rows,
        "exact_and_relabeled_fully_support_qstar": all(
            rows[name]["fully_supported"] for name in ("B_sfp", "D_relabeled")
        ),
        "scramble_support_fraction": rows["C_scrambled"][
            "representational_support_fraction"
        ],
        "scramble_statement": (
            "the certified q* is a chamber symbol and Arm C renames no chamber, so "
            "the label is well defined for C; what C lacks is the INCIDENCE that "
            "makes it findable. A non-automorphic edge permutation can carry two "
            "Events that share a block onto two that do not, so C's coded inputs "
            "often meet in no chamber or in the wrong one. Those examples are kept "
            "and C is allowed to fail on them, per 009.07 section 7."
        ),
        "oracle_role": (
            "sfp.ExactSfpCircuit as a nonlearned reference oracle for a structural "
            "READ-OUT; never a label source and never shown to a learner"
        ),
        "arm_a_asymmetry": ARM_A_ASYMMETRY,
    }


def verify_target_covariance(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]], chamber_of_block: dict
) -> dict:
    """Arm D's ``q*`` is the affine image of Arm B's, checked two independent ways.

    ``009.07`` section 6 asks for covariance to be verified rather than assumed. The
    label-level half of that is exact and needs no learner:

    1. the declared chamber action agrees with ``arms.relabel_address`` -- the map
       that actually built Arm D's code table -- on every one of the 84 codes and
       both of their presentations;
    2. the transported label round-trips: applying the inverse affine map to
       ``q*_D`` returns ``q*_B`` on all 336 admitted pairs.
    """

    affine_inverse = groups.invert_affine(ARM_D_AFFINE)
    inverse_action = groups.chamber_permutation(affine_inverse)
    action = chamber_actions()["D_relabeled"]

    # 1. the declared action IS arms.relabel_address's action on chambers.
    codes = tables["B_sfp"]
    relabel_agrees = 0
    for code in codes:
        expected = relabel_address(code)
        images = frozenset(action[q] for q in code.endpoints)
        if images == expected.endpoints:
            relabel_agrees += 1

    # 2. round-trip on the labels themselves.
    admitted = [dataset.records[p] for p in dataset.admit_positions]
    round_trip = 0
    for record in admitted:
        q_b = qstar_target(record, chamber_of_block, chamber_actions()["B_sfp"])
        q_d = qstar_target(record, chamber_of_block, action)
        if inverse_action[q_d] == q_b:
            round_trip += 1

    identity = tuple(inverse_action[action[q]] for q in CHAMBER_VALUES)
    return {
        "affine_element": [[list(row) for row in ARM_D_AFFINE[0]], list(ARM_D_AFFINE[1])],
        "chamber_action": list(action),
        "inverse_chamber_action": list(inverse_action),
        "action_composed_with_inverse": list(identity),
        "inverse_is_an_inverse": identity == IDENTITY_CHAMBERS,
        "codes_checked": len(codes),
        "declared_action_matches_relabel_address": relabel_agrees,
        "declared_action_is_the_arm_d_map": relabel_agrees == len(codes),
        "admitted_pairs": len(admitted),
        "qstar_round_trips_through_the_inverse": round_trip,
        "label_covariance_exact": round_trip == len(admitted),
        "statement": (
            "at the LABEL level covariance is exact and carries no learning: q*_D = "
            "phi(q*_B) on all 336 admitted pairs and phi is the same chamber action "
            "that arms.relabel_address used to build Arm D's code table. The learned "
            "half of the covariance requirement is a separate, measured audit in "
            "locator_heads.py (architectural, exact) and locator_analysis.py "
            "(paired D-B, measured)."
        ),
    }


def verify_recomposition_targets(
    dataset: Dataset, tables: dict[str, tuple[SfpAddress, ...]], chamber_of_block: dict
) -> dict:
    """Rung 2's four-field target, built query-relatively, denotes the same Event.

    ``009.07`` section 3.1 requires ``(S, FFF, q*, d3)`` to resolve to the same
    certified Event as the other valid presentation ``(q* XOR d3, d3)``. Checked
    exhaustively against ``ladder_task.rung2_target`` -- the frozen ``009.06``
    object -- so the two turns are provably targeting the same Events.
    """

    actions = chamber_actions()
    admitted = [dataset.records[p] for p in dataset.admit_positions]
    rows = {}
    for name in SCIENCE_ARMS:
        table = output_table(name, tables)
        action = actions[name]
        same_event = other_presentation = in_pp_set = 0
        for record in admitted:
            structured = rung2_target(record, table)
            query = query_relative_address(record, table, chamber_of_block, action)
            if query.as_key() == structured.address.as_key():
                same_event += 1
            q = qstar_target(record, chamber_of_block, action)
            d3 = table[record.target_index].d
            if (
                address_of(query.s, query.fff, q ^ d3, d3).as_key()
                == structured.address.as_key()
            ):
                other_presentation += 1
            if q in structured.pp_set:
                in_pp_set += 1
        n = len(admitted)
        rows[name] = {
            "admitted_pairs": n,
            "query_relative_equals_009_06_target": same_event,
            "other_presentation_equals_009_06_target": other_presentation,
            "qstar_lies_in_the_009_06_pp_set": in_pp_set,
            "both_presentations_agree": same_event == other_presentation,
            "identical_to_009_06_targets": same_event == n,
        }
    return {
        "per_arm": rows,
        "exact_and_relabeled_identical_to_009_06": all(
            rows[name]["identical_to_009_06_targets"]
            for name in ("A_native", "B_sfp", "D_relabeled")
        ),
        "torsor_statement": (
            "both (q*, d3) and (q* XOR d3, d3) canonicalize to one SfpAddress, so "
            "naming the consequence relative to the query does not privilege an "
            "origin of the affine PP torsor. The Rung-2 metric is unchanged from "
            "009.06 precisely because the target Event is unchanged."
        ),
        "scramble_note": (
            "under Arm C the certified q* need not lie in C's coded target address, "
            "so the query-relative presentation can fail to denote it. That is the "
            "same misalignment the support fraction records, and it is why C is the "
            "load-bearing control rather than a broken arm."
        ),
    }


def verify_scramble_recertification(dataset: Dataset, tables: dict) -> dict:
    """``009.07`` section 7: reuse the frozen Arm C, and reverify it mechanically.

    Nothing is reselected. The construction record is read back out of the frozen
    ``arms.build_arm_c`` metadata and each claim is recomputed from ``groups``.
    """

    arm_c = next(
        arm for arm in arms_module.all_arms(dataset, include_e=False)
        if arm.name == "C_scrambled"
    )
    plan = arm_c.metadata["per_habitat_scrambles"]
    induced = {tuple(p) for p in groups.induced_edge_permutations()}
    rows = []
    for row in plan:
        perm = tuple(row["edge_permutation"])
        rows.append(
            {
                "habitat": row["habitat"],
                "edge_permutation": list(perm),
                "is_a_permutation": sorted(perm) == list(range(len(groups.EDGES))),
                "in_the_24_induced_class": perm in induced,
                "is_structure_preserving": groups.is_structure_preserving(perm),
                "seed_sha256": row["seed_sha256"],
            }
        )
    # Field marginals of the code alphabet, B versus C, from the code tables.
    def marginals(table):
        counter = {"S": Counter(), "FFF": Counter(), "PP": Counter(), "pp": Counter()}
        for code in table:
            counter["S"][code.s] += 1
            counter["FFF"][code.fff] += 1
            counter["pp"][code.d] += 1
            for q in code.endpoints:
                counter["PP"][q] += 1
        return {
            field: {str(k): v for k, v in sorted(c.items())}
            for field, c in sorted(counter.items())
        }

    marg_b = marginals(tables["B_sfp"])
    marg_c = marginals(tables["C_scrambled"])
    alphabet_b = {code.as_key() for code in tables["B_sfp"]}
    alphabet_c = {code.as_key() for code in tables["C_scrambled"]}
    return {
        "per_habitat": rows,
        "habitats_checked": len(rows),
        "all_outside_the_induced_class": all(
            not row["in_the_24_induced_class"] for row in rows
        ),
        "none_structure_preserving": all(
            not row["is_structure_preserving"] for row in rows
        ),
        "induced_class_size": pin(24, len(induced), "groups.induced_edge_permutations"),
        "field_marginals_B": marg_b,
        "field_marginals_C": marg_c,
        "field_marginals_match": marg_b == marg_c,
        "code_alphabet_identical": alphabet_b == alphabet_c,
        "frozen_before_training": (
            "arms.scramble_plan derives the permutations from "
            "sha256('009/task6/arm-c-scramble/v1|<habitat>') and no module "
            "downstream may reselect them; this turn imports arms.all_arms and "
            "invents nothing"
        ),
        "reused_not_reinvented": (
            "the Arm C construction is the frozen 009.02/009.06 object, reused by "
            "import. 009.07 section 7 forbids inventing a new scramble after seeing "
            "results and none was invented."
        ),
    }


def split_census(dataset: Dataset, folds: tuple) -> dict:
    """The Rung-1 and Rung-2 split sizes actually used by this turn."""

    rows = []
    for fold in folds:
        r1_train, r1_test = qstar_positions(dataset, fold)
        r2_train, r2_test = rung2_positions(dataset, fold)
        rows.append(
            {
                "family": fold.family,
                "name": fold.name,
                "rung1_train": len(r1_train),
                "rung1_test": len(r1_test),
                "rung2_train": len(r2_train),
                "rung2_test": len(r2_test),
                "held_out_events": len(fold.held_out_events),
            }
        )
    return {
        "per_fold": rows,
        "n_folds": len(rows),
        "loho_folds": sum(1 for row in rows if row["family"] == "LOHO"),
        "lofpo_folds": sum(1 for row in rows if row["family"] == "LOFPO"),
        "reused_from": (
            "ladder_task.rung1_positions and ladder_task.rung2_positions, which in "
            "turn delegate to harness.training_positions / evaluation_positions; the "
            "splits are byte-identical to the ones that produced 0.7158 and 0.7016"
        ),
    }


def prior_reference_check() -> dict:
    """Re-verify the quoted ``009.06`` numbers against its bulk artifact if present.

    ``ladder_sweep.json`` is 9.2 MiB of bulk run data that lives in the Quilt
    package rather than in git, so this check is conditional by design: it upgrades
    the quoted constants from "transcribed" to "re-derived" whenever the file has
    been fetched, and says which of the two happened.
    """

    path = ROOT / "009_ladder_artifacts" / "ladder_sweep.json"
    if not path.exists():
        return {
            "artifact_present": False,
            "quoted": PRIOR_009_06,
            "status": (
                "quoted from the 009.06 result turn; ladder_sweep.json is bulk run "
                "data hosted in the Quilt package, so it is not in this clone"
            ),
        }
    payload = json.loads(path.read_text())
    aggregates = payload["blocks"]["repair"]["rung2"]["aggregates"]
    observed = {
        "field_PP_accuracy": {
            "LOHO": aggregates["B_sfp|LOHO"]["field_PP_accuracy"]["mean"],
            "LOFPO": aggregates["B_sfp|LOFPO"]["field_PP_accuracy"]["mean"],
        },
        "field_pp_accuracy": {
            "LOHO": aggregates["B_sfp|LOHO"]["field_pp_accuracy"]["mean"],
        },
        "positive_forced_third_exact_accuracy": {
            "LOHO": aggregates["B_sfp|LOHO"][
                "positive_forced_third_exact_accuracy"
            ]["mean"],
        },
    }
    checks = {}
    for metric, families in observed.items():
        for familyname, value in families.items():
            quoted = PRIOR_009_06[metric][familyname]
            checks[f"{metric}|{familyname}"] = {
                "quoted": quoted,
                "observed": rd(value),
                "agrees_to_four_decimals": round(float(value), 4) == quoted,
            }
    return {
        "artifact_present": True,
        "quoted": PRIOR_009_06,
        "observed": observed,
        "checks": checks,
        "all_quoted_numbers_reproduce": all(
            row["agrees_to_four_decimals"] for row in checks.values()
        ),
        "status": "re-derived from 009_ladder_artifacts/ladder_sweep.json",
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    folds = structural_folds(all_folds(dataset))
    tables = code_tables(dataset)
    codec = SfpCodec()
    chamber_of_block = certified_chamber_of_block(dataset, codec)

    upstream = verify_upstream_pins(dataset, all_folds(dataset))
    fence = verify_label_source_fence()
    chamber_map = verify_chamber_map(dataset, codec, chamber_of_block)
    census = verify_qstar_census(dataset, chamber_of_block)
    alignment = verify_qstar_alignment(dataset, tables, chamber_of_block)
    covariance = verify_target_covariance(dataset, tables, chamber_of_block)
    recomposition = verify_recomposition_targets(dataset, tables, chamber_of_block)
    scramble = verify_scramble_recertification(dataset, tables)
    splits = split_census(dataset, folds)
    leakage = verify_leakage(dataset, folds, tables)
    prior = prior_reference_check()

    pins = {
        "certified_blocks": pin(
            PINS["certified_blocks"], chamber_map["blocks_mapped"], "task.Catalogue"
        ),
        "admit": pin(
            PINS["admit"],
            census["per_arm"]["B_sfp"]["labelled_pairs"],
            "task.Dataset.admit_positions",
        ),
        "chambers_per_chart": pin(
            PINS["chambers_per_chart"], len(CHAMBER_VALUES), "sfp.py chart"
        ),
        "structural_folds": pin(
            PINS["structural_folds"], splits["n_folds"], "folds.structural_folds"
        ),
        "loho_folds": pin(PINS["loho_folds"], splits["loho_folds"], "folds.py"),
        "lofpo_folds": pin(PINS["lofpo_folds"], splits["lofpo_folds"], "folds.py"),
    }

    agrees = (
        upstream["all_upstream_pins_agree"]
        and fence["no_label_function_accepts_a_circuit"]
        and chamber_map["every_habitat_bijective_onto_its_chart"]
        and chamber_map["block_habitat_always_agrees"]
        and census["all_arms_uniform"]
        and alignment["exact_and_relabeled_fully_support_qstar"]
        and covariance["inverse_is_an_inverse"]
        and covariance["declared_action_is_the_arm_d_map"]
        and covariance["label_covariance_exact"]
        and recomposition["exact_and_relabeled_identical_to_009_06"]
        and scramble["all_outside_the_induced_class"]
        and scramble["none_structure_preserving"]
        and scramble["field_marginals_match"]
        and scramble["code_alphabet_identical"]
        and leakage["no_leakage_anywhere"]
        and all(row["agrees"] for row in pins.values())
        and (
            not prior["artifact_present"] or prior["all_quoted_numbers_reproduce"]
        )
    )

    return {
        "module": "locator_task",
        "executes": EXECUTED_TASK,
        "builds_on": PRIOR_RESULT,
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "pins": pins,
        "upstream_pins": upstream,
        "label_source_fence": fence,
        "chamber_map": chamber_map,
        "qstar_census": census,
        "qstar_alignment": alignment,
        "target_covariance": covariance,
        "recomposition_targets": recomposition,
        "scramble_recertification": scramble,
        "splits": splits,
        "leakage": leakage,
        "prior_reference": prior,
        "non_replayable_keys": list(NON_REPLAYABLE_KEYS),
        "replay_note": (
            "prior_reference is excluded from the --check byte comparison because it "
            "reads 009_ladder_artifacts/ladder_sweep.json, which is gitignored bulk "
            "run data hosted in the Quilt package. Its content depends on whether that "
            "file has been fetched, so including it would make --check fail in a fresh "
            "clone for a reason unrelated to this module. Everything else in this "
            "artifact is derived from the frozen in-repo chain and replays byte for "
            "byte with nothing fetched."
        ),
        "arm_output_convention": dict(ARM_OUTPUT_CONVENTION),
        "code_arms": list(CODE_ARMS),
        "science_arms": list(SCIENCE_ARMS),
        "pp_classes": list(PP_CLASSES),
        "chamber_values": list(CHAMBER_VALUES),
        "torch_status": (
            "no tensor is constructed in this module. torch is inherited "
            "transitively through ladder_task -> harness, which is recorded rather "
            "than claimed otherwise."
        ),
        "digests": {
            "catalogue": dataset.catalogue.sha256(),
            "dataset": dataset.sha256(),
            "chart": codec.chart()["sha256"],
            "chamber_map": digest(
                [
                    [bid, list(habitat_key), q]
                    for bid, (habitat_key, q) in sorted(chamber_of_block.items())
                ]
            ),
            "qstar_labels": digest(
                {
                    name: [
                        qstar_target(
                            dataset.records[p], chamber_of_block, chamber_actions()[name]
                        )
                        for p in dataset.admit_positions
                    ]
                    for name in SCIENCE_ARMS
                }
            ),
        },
        "verdict": {
            "agrees": agrees,
            "statement": (
                "the certified query-relative shared chamber q* is defined, "
                "circuit-free, uniform at chance 1/4, exactly supported by the exact "
                "and relabeled codes, covariant at the label level, identical to the "
                "009.06 target Events, and leakage-free on all 21 structural folds"
                if agrees
                else "at least one q* label-source or control check FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    print(f"executed:  {result['executes']}", flush=True)
    print(f"builds on: {result['builds_on']}", flush=True)
    align = result["qstar_alignment"]["per_arm"]
    print("q* representational support (of 336 admitted pairs):", flush=True)
    for name in result["science_arms"]:
        row = align[name]
        print(
            f"  {name:<14} shared-endpoint-is-q* {row['code_shared_endpoint_is_qstar']:>4}"
            f"  q*-in-coded-target {row['qstar_in_coded_target']:>4}"
            f"  support {row['representational_support_fraction']}",
            flush=True,
        )
    cov = result["target_covariance"]
    print(
        f"label covariance exact: {cov['label_covariance_exact']}"
        f"  chamber action {cov['chamber_action']}",
        flush=True,
    )


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
        expected = strip_non_replayable(json.loads(OUTPUT.read_text()))
        observed = strip_non_replayable(json.loads(text))
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived audit is not identical to {OUTPUT.name} "
                f"(excluding {', '.join(NON_REPLAYABLE_KEYS)})"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(
            f"PASS: exact replay matches {OUTPUT.name} "
            f"(excluding {', '.join(NON_REPLAYABLE_KEYS)})",
            flush=True,
        )
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

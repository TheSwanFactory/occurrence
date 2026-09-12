"""009.09 sweep: opaque local-chart discovery, its control, and its chart audit.

Two declared blocks -- ``009.09`` section 5.3's one primary architecture pass and
its one clearly diagnosed repair -- crossed with two arms, one protocol, no tuning::

    block primary   m_ij = Message(h_j)            the primary pass
    block repair    m_ij = Message([h_i, h_j])     the one declared repair

    arm one_anchor    admission graph + one certified cyclic-block triple
    arm zero_anchor   admission graph only, anchor channel zeroed  (section 7.2)

The primary block has a **seed-dependent 6/9 saddle**, diagnosed in
``discovery_heads.DIAGNOSED_PATHOLOGY`` and evidenced by this module's own
``anchor_overlap_0_accuracy`` / ``anchor_overlap_1_accuracy`` split. It is reported
in full rather than deleted. Between the two blocks nothing changes but the message
form: same widths, depth, aggregation, optimizer, learning rate, step count, seeds,
arms, folds, episodes, relabelings and observations.

Within a block, both arms are the same architecture at the same parameter count -- the
control removes the anchor evidence, not capacity. On top of the trained
``one_anchor`` model, and adding no parameter and no training step, three test-time
audits run:

    section 6      all-pairs chart reconstruction, certified/dual/invalid family,
                   and the exact AGL(2,2) equivalence search
    section 7.5    fresh opaque relabelling transport
    009.09a s.2    the dual-anchor control: the same model, handed a
                   COMBINATORIAL dual-family triple, scored against dual labels

``009.09`` section 5.3 permits one primary architecture pass and forbids a
hyperparameter search. There is none: the optimizer, learning rate, step count,
seeds, folds, episodes and relabelings are all the frozen objects, and the
training protocol is verbatim the ``009.02``/``009.06``/``009.08`` one -- Adam,
``lr = 0.01``, 800 full-batch steps, no masking, no early stopping, no repair.

What is scored, and what is only measured
----------------------------------------
``forced_third_accuracy`` is the headline: exact integer identity against the
certified forced third, over the nine scored queries of each episode. It is never
masked, so the two query nodes remain scoreable and
``prediction_on_query_node_rate`` is a measured fact rather than an enforced zero.

Raw accuracy is explicitly **not** sufficient for ``H_discovery``. Section 6.4
requires the all-pairs relation to reconstruct the certified block family as well,
so ``chart_valid_rate``, ``exact_block_family_recovery`` and ``sfp_equivalent_rate``
are first-class metrics, and a coherent prediction of the admission-compatible
**dual** family is reported separately and does not count as recovery.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from discovery_baselines import (
    SUPPLIED_SFP_REFERENCE,
    UNIFORM_COMMON_NEIGHBOURS,
    UNIFORM_ELIGIBLE_NODES,
)
from discovery_heads import (
    DIAGNOSED_PATHOLOGY,
    DiscoveryConfig,
    build_pointer,
    capacity_ledger,
    episode_inputs,
    pointer_loss,
)
from discovery_task import (
    N_TOKENS,
    Episode,
    build_episodes,
    build_local_problems,
    fraction,
    habitats_of_fold,
    is_block_system,
    render,
    sfp_equivalence_witnesses,
)
from folds import all_folds, structural_folds
from ladder_sweep import NON_REPLAYABLE_KEYS, stats, strip_non_replayable
from scorer import count_params, set_seed
from sfp import SfpCodec
from task import build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_discovery_artifacts"
OUTPUT = ARTIFACTS / "discovery_sweep.json"
SMOKE_OUTPUT = ARTIFACTS / "discovery_sweep_smoke.json"

BASE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"

#: The frozen ``009.02``/``009.06``/``009.08`` seeds.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)

#: The two training arms. ``zero_anchor`` is the section 7.2 anti-cheating control.
ARMS = ("one_anchor", "zero_anchor")

METRIC_DECIMALS = 12

RUN_METRICS = (
    "forced_third_accuracy",
    "train_accuracy",
    "generalization_gap",
    "anchor_overlap_0_accuracy",
    "anchor_overlap_1_accuracy",
    "train_anchor_overlap_0_accuracy",
    "train_anchor_overlap_1_accuracy",
    "prediction_on_query_node_rate",
    "swap_invariance_error",
    "swap_argmax_disagreement_rate",
    "chart_valid_rate",
    "exact_block_family_recovery",
    "dual_family_rate",
    "invalid_family_rate",
    "sfp_equivalent_rate",
    "supplied_anchor_chart_valid_rate",
    "supplied_anchor_exact_block_family_recovery",
    "supplied_anchor_dual_family_rate",
    "supplied_anchor_invalid_family_rate",
    "supplied_anchor_sfp_equivalent_rate",
    "all_pairs_third_accuracy",
    "support_pair_third_accuracy",
    "support_pair_other_family_answer_rate",
    "strict_unique_triples_mean",
    "transport_decision_rate",
    "transport_family_rate",
    "transport_supplied_anchor_family_rate",
)

#: Reported for the ``one_anchor`` arm only: the ``009.09a`` section 2 control is a
#: test-time evaluation of the trained model against COMBINATORIAL dual labels.
DUAL_METRICS = (
    "dual_anchor_forced_third_accuracy",
    "dual_anchor_chart_valid_rate",
    "dual_anchor_dual_family_rate",
    "dual_anchor_certified_family_rate",
)

FENCES = (
    "One protocol, fixed before any run; no tuning, no per-arm adjustment, no "
    "repair.",
    "Correctness is exact certified Event identity, never float proximity.",
    "Evaluation is never masked. The two query nodes stay scoreable so that "
    "'the predicted third is distinct from the queried pair' is measured.",
    "The zero-anchor arm is the SAME parameters with the anchor channel zeroed. It "
    "is the primary anti-cheating control and its exact information ceiling is 1/2.",
    "The dual-anchor arm trains nothing. Its labels are COMBINATORIAL, derived from "
    "the second admission-compatible block system, and are never certified FIPS "
    "labels. It is a control and is never quoted as a result.",
    "High forced-third accuracy with an incoherent or non-equivalent all-pairs "
    "relation is consequence learning, NOT representation discovery (section 6.4).",
    "sfp_equivalent is reported as a consistency confirmation ENTAILED by "
    "block-family recovery, per 009.09a section 3, not as independent evidence.",
    "The holdout is fresh-relabelling generalization of a local relational rule, "
    "not held-out-structure generalization in the 009.06/009.08 sense.",
    "S and FFF never enter the network. The habitat-blindness proof lives in "
    "discovery_heads.",
    "No episode row is a BOTTOM row: every scored query is an admitted pair. BOTTOM "
    "remains distinct from every algebraic zero and from a malformed prediction.",
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), METRIC_DECIMALS)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """The one training protocol. ``locator_sweep.TrainConfig``'s values, unchanged."""

    steps: int = 800
    lr: float = 0.01

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if not self.lr > 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "optimizer": "Adam",
            "batching": "full batch over every scored query of every train episode",
            "evaluation_masking": "none, ever",
            "early_stopping": "none",
            "tuning": (
                "none. Verbatim the 009.02/009.06/009.08 protocol, so a difference "
                "in result is attributable to the observation set and the output "
                "constitution rather than to the optimizer."
            ),
            "repairs_used": (
                "none. 009.09 section 5.3 permits one clearly diagnosed repair; the "
                "primary pass needed none."
            ),
            "supervision": (
                "certified forced-third answers for scored query pairs in training "
                "episodes only. PP, pp, SFP coordinates, chamber numbers and the "
                "four-block family are NEVER supervised."
            ),
        }


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...] = SEEDS
    arms: tuple[str, ...] = ARMS
    families: tuple[str, ...] = ("LOHO", "LOFPO")
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"duplicate seeds: {self.seeds}")
        unknown = [name for name in self.arms if name not in ARMS]
        if unknown:
            raise ValueError(f"unknown arm names: {unknown}")

    def as_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "n_seeds": len(self.seeds),
            "arms": list(self.arms),
            "arm_roles": {
                "one_anchor": (
                    "PRIMARY. Admission graph plus one certified cyclic-block triple."
                ),
                "zero_anchor": (
                    "CONTROL (section 7.2). Identical parameters, anchor channel "
                    "zeroed. Exact information ceiling 1/2."
                ),
            },
            "families": list(self.families),
            "discovery": self.discovery.as_dict(),
            "train": self.train.as_dict(),
        }


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    """Every scored query of a set of episodes, as one full batch.

    ``marks`` and ``adjacency`` are the entire input surface. ``episode`` and
    ``query`` are bookkeeping used by the audits and never by the model.
    """

    marks: torch.Tensor           # (N, 6, 2)
    adjacency: torch.Tensor       # (N, 6, 6)
    swapped_marks: torch.Tensor   # (N, 6, 2) -- the query written in the other order
    target: torch.Tensor          # (N,) the certified third, as a node index
    query: np.ndarray             # (N, 2) node indices
    episode: np.ndarray           # (N,) index into the episode tuple
    anchor_overlap: np.ndarray    # (N,) how many query nodes the anchor contains
    size: int


def build_batch(
    episodes: tuple[Episode, ...],
    positions: tuple[int, ...],
    *,
    use_anchor: bool,
    family: str = "certified",
    queries: str = "scored",
) -> Batch:
    """One full batch over the chosen queries of the chosen episodes."""

    marks, adjacency, swapped, targets, query_rows, episode_rows = [], [], [], [], [], []
    overlaps = []
    for position in positions:
        episode = episodes[position]
        anchor = set(episode.anchor_nodes)
        thirds = dict(
            episode.certified_third_nodes
            if family == "certified"
            else episode.dual_third_nodes
        )
        chosen = (
            episode.scored_queries
            if queries == "scored"
            else episode.admit_nodes
        )
        for query in chosen:
            overlaps.append(len(set(query) & anchor))
            forward = episode_inputs(
                episode.admit_nodes,
                episode.anchor_nodes,
                query,
                use_anchor=use_anchor,
            )
            backward = episode_inputs(
                episode.admit_nodes,
                episode.anchor_nodes,
                (query[1], query[0]),
                use_anchor=use_anchor,
            )
            marks.append(forward[0])
            adjacency.append(forward[1])
            swapped.append(backward[0])
            targets.append(thirds[query])
            query_rows.append(list(query))
            episode_rows.append(position)
    return Batch(
        marks=torch.tensor(marks, dtype=torch.float32),
        adjacency=torch.tensor(adjacency, dtype=torch.float32),
        swapped_marks=torch.tensor(swapped, dtype=torch.float32),
        target=torch.tensor(targets, dtype=torch.long),
        query=np.array(query_rows, dtype=np.int64),
        episode=np.array(episode_rows, dtype=np.int64),
        anchor_overlap=np.array(overlaps, dtype=np.int64),
        size=len(marks),
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def predict(model, batch: Batch) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(batch.marks, batch.adjacency)
    return scores.argmax(dim=-1).numpy()


def evaluate(model, batch: Batch) -> dict:
    """Forced-third accuracy, the query-node diagnostic, and the swap check."""

    model.eval()
    with torch.no_grad():
        scores = model(batch.marks, batch.adjacency)
        swapped = model(batch.swapped_marks, batch.adjacency)
    predicted = scores.argmax(dim=-1).numpy()
    target = batch.target.numpy()
    exact = predicted == target
    on_query = np.array(
        [predicted[i] in set(batch.query[i]) for i in range(batch.size)]
    )
    # The anchor-overlap split. The anchor is a triangle, so a scored query holds
    # either one anchor-marked node (6 of 9) or none (3 of 9), and the exact rule's
    # anchor preference FLIPS SIGN between the two. Reporting the split is what
    # localizes the primary pass's 6/9 saddle instead of leaving a mean to be
    # misread as uniform partial competence.
    per_case = {}
    for case in (0, 1):
        rows = batch.anchor_overlap == case
        per_case[f"anchor_overlap_{case}_accuracy"] = _round(
            fraction(int(exact[rows].sum()), int(rows.sum()))
        )
        per_case[f"anchor_overlap_{case}_n"] = int(rows.sum())
    return {
        "n": batch.size,
        "forced_third_accuracy": _round(float(exact.mean())),
        "prediction_on_query_node_rate": _round(float(on_query.mean())),
        **per_case,
        "swap_invariance_error": _round(
            float((scores - swapped).abs().max().item())
        ),
        "swap_argmax_disagreement_rate": _round(
            float((scores.argmax(dim=-1) != swapped.argmax(dim=-1)).float().mean().item())
        ),
        "swap_scores_bitwise_identical": bool(torch.equal(scores, swapped)),
        "correctness_criterion": (
            "integer identity against the certified forced third, as an opaque node "
            "index. Never float proximity, never masked."
        ),
    }


# ---------------------------------------------------------------------------
# section 6: the representation-discovery audit
# ---------------------------------------------------------------------------

def recovered_chart(
    predicted: np.ndarray,
    queries: np.ndarray,
    admit_nodes,
    *,
    supplied: tuple[tuple[int, int, int], ...] = (),
) -> dict:
    """Reconstruct the relation implied by a model's answers.

    For every queried pair ``{a, b}`` with predicted third ``c``, form the unordered
    triple ``{a, b, c}``, add any ``supplied`` chamber, then test the resulting set
    against section 6.1's five conditions.

    ``supplied`` exists because of a real tension in the task, surfaced rather than
    resolved silently. Section 4.4 makes the three pairs inside the anchor *support*
    and not scored queries, so the learner is never trained on them; section 6.1 then
    asks for a reconstruction in which "every admitted pair occurs in exactly one
    predicted chamber", which covers those three. Two reconstructions are therefore
    reported: the literal all-twelve-pairs one, which requires the learner to
    extrapolate into a regime it never saw, and the one in which the nine scored
    answers supply three chambers and the fourth is the anchor the task itself
    supplied as support. The nine scored pairs are exactly the twelve minus the
    anchor's three, so they fall in exactly the three non-anchor chambers and the two
    forms differ only in where the fourth chamber comes from.
    """

    triples = []
    third_inside_query = 0
    for index in range(len(predicted)):
        a, b = int(queries[index][0]), int(queries[index][1])
        c = int(predicted[index])
        if c in (a, b):
            third_inside_query += 1
        triples.append(tuple(sorted({a, b, c})))
    triples.extend(tuple(sorted(triple)) for triple in supplied)
    unique = tuple(sorted(set(triples)))

    occurrences = {node: 0 for node in range(N_TOKENS)}
    for triple in unique:
        for node in triple:
            occurrences[node] += 1
    pair_cover: dict[frozenset[int], int] = {}
    for triple in unique:
        for left in range(3):
            for right in range(left + 1, 3):
                key = frozenset((triple[left], triple[right]))
                pair_cover[key] = pair_cover.get(key, 0) + 1

    conditions = {
        "exactly_four_unique_triples": len(unique) == 4,
        "three_distinct_events_per_triple": all(
            len(set(triple)) == 3 for triple in unique
        ),
        "each_event_in_exactly_two_triples": all(
            count == 2 for count in occurrences.values()
        ),
        "every_admitted_pair_in_exactly_one_predicted_chamber": all(
            pair_cover.get(frozenset(pair), 0) == 1 for pair in admit_nodes
        ),
        "all_predicted_thirds_distinct_from_the_queried_pair": third_inside_query
        == 0,
    }
    valid = all(conditions.values()) and is_block_system(unique, tuple(admit_nodes))
    return {
        "unique_triples": [list(triple) for triple in unique],
        "n_unique_triples": len(unique),
        "thirds_inside_the_queried_pair": third_inside_query,
        "conditions": conditions,
        "is_a_block_system": is_block_system(unique, tuple(admit_nodes)),
        "chart_valid": valid,
        "supplied_chambers": [list(triple) for triple in supplied],
    }


def classify_family(
    chart: dict, episode: Episode
) -> dict:
    """Certified / dual / invalid, plus the exact ``AGL(2,2)`` equivalence search."""

    recovered = tuple(sorted(tuple(t) for t in chart["unique_triples"]))
    is_certified = recovered == tuple(sorted(episode.certified_blocks_nodes))
    is_dual = recovered == tuple(sorted(episode.dual_blocks_nodes))
    witnesses = sfp_equivalence_witnesses(
        recovered, episode.certified_presentation_nodes
    )
    return {
        "chart_valid": chart["chart_valid"],
        "certified_family": is_certified,
        "dual_family": is_dual,
        "invalid_family": not (is_certified or is_dual),
        "sfp_equivalent": bool(witnesses),
        "sfp_equivalence_witnesses": len(witnesses),
        "conditions": chart["conditions"],
    }


def _rates(rows: list[dict], prefix: str = "") -> dict:
    total = len(rows)
    return {
        f"{prefix}chart_valid_rate": _round(
            fraction(sum(1 for r in rows if r["chart_valid"]), total)
        ),
        f"{prefix}exact_block_family_recovery": _round(
            fraction(sum(1 for r in rows if r["certified_family"]), total)
        ),
        f"{prefix}dual_family_rate": _round(
            fraction(sum(1 for r in rows if r["dual_family"]), total)
        ),
        f"{prefix}invalid_family_rate": _round(
            fraction(sum(1 for r in rows if r["invalid_family"]), total)
        ),
        f"{prefix}sfp_equivalent_rate": _round(
            fraction(sum(1 for r in rows if r["sfp_equivalent"]), total)
        ),
        f"{prefix}sfp_equivalent_agrees_with_family_recovery": all(
            r["sfp_equivalent"] == r["certified_family"] for r in rows
        ),
    }


def chart_audit(
    model,
    episodes: tuple[Episode, ...],
    positions: tuple[int, ...],
    *,
    use_anchor: bool,
    family: str = "certified",
) -> dict:
    """Section 6, over every test episode: validity, family, and equivalence."""

    batch = build_batch(
        episodes, positions, use_anchor=use_anchor, family=family, queries="all"
    )
    predicted = predict(model, batch)
    target = batch.target.numpy()

    strict_rows = []
    supplied_rows = []
    support_hits = 0
    support_other_hits = 0
    support_total = 0
    scored_hits = 0
    scored_total = 0
    unique_counts: list[int] = []
    for position in positions:
        rows = np.flatnonzero(batch.episode == position)
        episode = episodes[position]
        support = {frozenset(p) for p in episode.support_queries}
        scored_rows = [
            i for i in rows if frozenset(batch.query[i].tolist()) not in support
        ]
        support_only = [
            i for i in rows if frozenset(batch.query[i].tolist()) in support
        ]
        support_total += len(support_only)
        support_hits += int(sum(predicted[i] == target[i] for i in support_only))
        scored_total += len(scored_rows)
        scored_hits += int(sum(predicted[i] == target[i] for i in scored_rows))

        # Where does the support-pair answer go instead? The other admission
        # compatible continuation is the dual-family third, so this says whether the
        # extrapolation is arbitrary or is the learned rule carried past its domain.
        other = dict(
            episode.dual_third_nodes
            if family == "certified"
            else episode.certified_third_nodes
        )
        for i in support_only:
            support_other_hits += int(
                predicted[i] == other[tuple(sorted(batch.query[i].tolist()))]
            )
        unique_counts.append(
            len(
                {
                    tuple(sorted({int(batch.query[i][0]), int(batch.query[i][1]), int(predicted[i])}))
                    for i in rows
                }
            )
        )

        strict = recovered_chart(
            predicted[rows], batch.query[rows], episode.admit_nodes
        )
        with_anchor = recovered_chart(
            predicted[scored_rows],
            batch.query[scored_rows],
            episode.admit_nodes,
            supplied=(episode.anchor_nodes,),
        )
        strict_rows.append(
            {
                "episode_id": episode.episode_id,
                "habitat": episode.habitat,
                **classify_family(strict, episode),
            }
        )
        supplied_rows.append(
            {
                "episode_id": episode.episode_id,
                "habitat": episode.habitat,
                **classify_family(with_anchor, episode),
            }
        )

    total = len(strict_rows)
    all_pairs_exact = float((predicted == target).mean())
    return {
        "episodes": total,
        "all_pairs_queries": batch.size,
        "all_pairs_third_accuracy": _round(all_pairs_exact),
        "scored_pair_third_accuracy": _round(fraction(scored_hits, scored_total)),
        "support_pair_third_accuracy": _round(fraction(support_hits, support_total)),
        "support_pair_other_family_answer_rate": _round(
            fraction(support_other_hits, support_total)
        ),
        "strict_unique_triples_mean": _round(
            fraction(sum(unique_counts), len(unique_counts))
        ),
        **_rates(strict_rows),
        **_rates(supplied_rows, prefix="supplied_anchor_"),
        "witness_count_is_one_where_equivalent": all(
            r["sfp_equivalence_witnesses"] == 1
            for r in strict_rows + supplied_rows
            if r["sfp_equivalent"]
        ),
        "condition_failure_counts": {
            name: sum(1 for r in strict_rows if not r["conditions"][name])
            for name in strict_rows[0]["conditions"]
        }
        if strict_rows
        else {},
        "supplied_anchor_condition_failure_counts": {
            name: sum(1 for r in supplied_rows if not r["conditions"][name])
            for name in supplied_rows[0]["conditions"]
        }
        if supplied_rows
        else {},
        "reconstruction_forms": (
            "the unprefixed rates are section 6.1 read literally: all twelve admitted "
            "pairs are queried and the four chambers come entirely from predictions, "
            "including the three support pairs the learner was never trained on. The "
            "supplied_anchor_ rates instead take the three chambers implied by the "
            "nine scored answers and add the anchor triple the task supplies as "
            "support. Both are reported; neither is quietly substituted for the other."
        ),
        "support_pair_disclosure": (
            "support_pair_third_accuracy is the learner's answer on the three pairs "
            "lying inside the anchor. Section 4.4 excludes them from scoring and "
            "training, so this number measures EXTRAPOLATION into an unsupervised "
            "regime and is not part of the predeclared thresholds."
        ),
        "equivalence_note": (
            "009.09a section 3: any recovered certified four-block family on six "
            "Events is a K4 and AGL(2,2) is the full symmetric group on its four "
            "vertices, so sfp_equivalent is ENTAILED by exact_block_family_recovery. "
            "The audit is run as written and the two rates are asserted equal; a "
            "difference would be an implementation bug, not evidence."
        ),
    }


# ---------------------------------------------------------------------------
# section 7.5: fresh opaque relabelling transport
# ---------------------------------------------------------------------------

def transport_audit(
    model,
    episodes: tuple[Episode, ...],
    fresh: tuple[Episode, ...],
    positions: tuple[int, ...],
    *,
    use_anchor: bool,
) -> dict:
    """The answer and the recovered family must transport covariantly.

    Each primary episode is paired with an independently relabelled copy of the
    same habitat and the same anchor. Predictions are pulled back through each
    episode's own relabelling into local slots and compared there, so a match means
    the two runs named the *same Event*, not the same opaque index.
    """

    left = build_batch(episodes, positions, use_anchor=use_anchor, queries="all")
    right = build_batch(fresh, positions, use_anchor=use_anchor, queries="all")
    left_pred = predict(model, left)
    right_pred = predict(model, right)

    decisions = 0
    matched = 0
    families = 0
    family_matched = 0
    supplied_families = 0
    supplied_family_matched = 0
    for position in positions:
        primary, copy = episodes[position], fresh[position]
        if (primary.habitat, primary.anchor_index) != (copy.habitat, copy.anchor_index):
            raise AssertionError("fresh relabelling episodes are misaligned")
        if primary.relabeling_index == copy.relabeling_index:
            raise AssertionError("the fresh copy reused the primary relabelling")
        pi, pc = primary.inverse(), copy.inverse()

        rows_left = np.flatnonzero(left.episode == position)
        rows_right = np.flatnonzero(right.episode == position)
        slots_left = {}
        for index in rows_left:
            key = tuple(sorted((pi[int(left.query[index][0])], pi[int(left.query[index][1])])))
            slots_left[key] = pi[int(left_pred[index])]
        slots_right = {}
        for index in rows_right:
            key = tuple(sorted((pc[int(right.query[index][0])], pc[int(right.query[index][1])])))
            slots_right[key] = pc[int(right_pred[index])]
        if set(slots_left) != set(slots_right):
            raise AssertionError("the two relabellings expose different query sets")
        for key, value in slots_left.items():
            decisions += 1
            matched += int(value == slots_right[key])

        def pulled(pred, batch, rows, episode, inverse, *, supplied):
            chart = recovered_chart(
                pred[rows], batch.query[rows], episode.admit_nodes, supplied=supplied
            )
            return tuple(
                sorted(
                    tuple(sorted(inverse[n] for n in t))
                    for t in chart["unique_triples"]
                )
            )

        support_left = {frozenset(p) for p in primary.support_queries}
        support_right = {frozenset(p) for p in copy.support_queries}
        scored_left = np.array(
            [i for i in rows_left if frozenset(left.query[i].tolist()) not in support_left]
        )
        scored_right = np.array(
            [
                i
                for i in rows_right
                if frozenset(right.query[i].tolist()) not in support_right
            ]
        )
        families += 1
        family_matched += int(
            pulled(left_pred, left, rows_left, primary, pi, supplied=())
            == pulled(right_pred, right, rows_right, copy, pc, supplied=())
        )
        supplied_families += 1
        supplied_family_matched += int(
            pulled(
                left_pred,
                left,
                scored_left,
                primary,
                pi,
                supplied=(primary.anchor_nodes,),
            )
            == pulled(
                right_pred,
                right,
                scored_right,
                copy,
                pc,
                supplied=(copy.anchor_nodes,),
            )
        )

    return {
        "episode_pairs": len(positions),
        "decisions_compared": decisions,
        "transport_decision_rate": _round(fraction(matched, decisions)),
        "transport_family_rate": _round(fraction(family_matched, families)),
        "transport_supplied_anchor_family_rate": _round(
            fraction(supplied_family_matched, supplied_families)
        ),
        "exact_decision_transport": matched == decisions,
        "note": (
            "predictions are pulled back through each episode's own relabelling "
            "before comparison, so agreement means the two runs named the same "
            "Event. Any systematic sensitivity to opaque names would show up here as "
            "a rate below 1.000 and would invalidate the discovery claim."
        ),
    }


# ---------------------------------------------------------------------------
# 009.09a section 2: the dual-anchor control
# ---------------------------------------------------------------------------

def dual_anchor_control(
    model,
    dual: tuple[Episode, ...],
    positions: tuple[int, ...],
) -> dict:
    """The SAME trained model, handed a combinatorial dual-family triple.

    No training change, no architecture change, no extra parameter. If the learner
    uses the anchor *relationally* it should answer in the anchor's family, so the
    dual arm should score near ``1.000`` against dual labels with the certified
    family rate near ``0``. If it has absorbed any stable-token bias it should
    collapse toward ``1/2``.

    The labels are COMBINATORIAL, derived from the second admission-compatible
    block system. They are not certified FIPS labels and this arm is never a result.
    """

    batch = build_batch(
        dual, positions, use_anchor=True, family="combinatorial_dual", queries="scored"
    )
    metrics = evaluate(model, batch)
    chart = chart_audit(
        model, dual, positions, use_anchor=True, family="combinatorial_dual"
    )
    return {
        "role": "CONTROL (009.09a section 2), test-time only",
        "labels": "combinatorial dual-family thirds, NOT certified FIPS labels",
        "trains_nothing": True,
        "adds_no_parameter": True,
        "dual_anchor_forced_third_accuracy": metrics["forced_third_accuracy"],
        "dual_anchor_chart_valid_rate": chart["supplied_anchor_chart_valid_rate"],
        "dual_anchor_dual_family_rate": chart["supplied_anchor_dual_family_rate"],
        "dual_anchor_certified_family_rate": chart[
            "supplied_anchor_exact_block_family_recovery"
        ],
        "dual_anchor_invalid_family_rate": chart[
            "supplied_anchor_invalid_family_rate"
        ],
        "dual_anchor_strict_chart_valid_rate": chart["chart_valid_rate"],
        "dual_anchor_strict_dual_family_rate": chart["dual_family_rate"],
        "scored_queries": batch.size,
        "family_rate_form": (
            "the dual arm's family rates use the supplied-anchor reconstruction: the "
            "nine scored answers plus the supplied dual triple. dual_family_rate here "
            "means the model recovered the COMBINATORIAL dual family, which is the "
            "expected reading of 'answer in the anchor's family'. Note the dual "
            "family is what a CERTIFIED-family recovery would look like if the sign "
            "were wrong, which is exactly what this control is for."
        ),
        "reading": (
            "near 1.000 against dual labels with a certified-family rate near 0 means "
            "the anchor is used relationally -- the learner answers in the anchor's "
            "family rather than in a memorized one. A collapse toward 1/2 would "
            "indicate a stable-token bias. This distinguishes 'ignores the anchor' "
            "from 'uses the anchor with the wrong sign', which the zero-anchor arm "
            "alone cannot."
        ),
    }


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def train_one(
    arm: str,
    fold,
    seed: int,
    config: SweepConfig,
    dataset,
    episodes: tuple[Episode, ...],
    fresh: tuple[Episode, ...],
    dual: tuple[Episode, ...],
) -> dict:
    started = time.perf_counter()
    use_anchor = arm == "one_anchor"
    held = set(habitats_of_fold(dataset, fold))
    train_positions = tuple(
        i for i, e in enumerate(episodes) if e.habitat not in held
    )
    test_positions = tuple(i for i, e in enumerate(episodes) if e.habitat in held)

    model = build_pointer(config.discovery, seed=seed)
    set_seed(seed)

    train_batch = build_batch(episodes, train_positions, use_anchor=use_anchor)
    test_batch = build_batch(episodes, test_positions, use_anchor=use_anchor)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(config.train.steps):
        scores = model(train_batch.marks, train_batch.adjacency)
        loss = pointer_loss(scores, train_batch.target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_metrics = evaluate(model, train_batch)
    test_metrics = evaluate(model, test_batch)
    chart = chart_audit(model, episodes, test_positions, use_anchor=use_anchor)
    transport = transport_audit(
        model, episodes, fresh, test_positions, use_anchor=use_anchor
    )

    row = {
        "arm": arm,
        "fold_family": fold.family,
        "fold_name": fold.name,
        "seed": int(seed),
        "param_count": count_params(model),
        "train_episodes": len(train_positions),
        "test_episodes": len(test_positions),
        "train_size": train_batch.size,
        "test_size": test_batch.size,
        "forced_third_accuracy": test_metrics["forced_third_accuracy"],
        "train_accuracy": train_metrics["forced_third_accuracy"],
        "generalization_gap": _round(
            train_metrics["forced_third_accuracy"]
            - test_metrics["forced_third_accuracy"]
        ),
        "anchor_overlap_0_accuracy": test_metrics["anchor_overlap_0_accuracy"],
        "anchor_overlap_1_accuracy": test_metrics["anchor_overlap_1_accuracy"],
        "train_anchor_overlap_0_accuracy": train_metrics[
            "anchor_overlap_0_accuracy"
        ],
        "train_anchor_overlap_1_accuracy": train_metrics[
            "anchor_overlap_1_accuracy"
        ],
        "prediction_on_query_node_rate": test_metrics[
            "prediction_on_query_node_rate"
        ],
        "swap_invariance_error": test_metrics["swap_invariance_error"],
        "swap_argmax_disagreement_rate": test_metrics[
            "swap_argmax_disagreement_rate"
        ],
        "swap_scores_bitwise_identical": test_metrics[
            "swap_scores_bitwise_identical"
        ],
        "chart_valid_rate": chart["chart_valid_rate"],
        "exact_block_family_recovery": chart["exact_block_family_recovery"],
        "dual_family_rate": chart["dual_family_rate"],
        "invalid_family_rate": chart["invalid_family_rate"],
        "sfp_equivalent_rate": chart["sfp_equivalent_rate"],
        "sfp_equivalent_agrees_with_family_recovery": chart[
            "sfp_equivalent_agrees_with_family_recovery"
        ],
        "supplied_anchor_chart_valid_rate": chart[
            "supplied_anchor_chart_valid_rate"
        ],
        "supplied_anchor_exact_block_family_recovery": chart[
            "supplied_anchor_exact_block_family_recovery"
        ],
        "supplied_anchor_dual_family_rate": chart[
            "supplied_anchor_dual_family_rate"
        ],
        "supplied_anchor_invalid_family_rate": chart[
            "supplied_anchor_invalid_family_rate"
        ],
        "supplied_anchor_sfp_equivalent_rate": chart[
            "supplied_anchor_sfp_equivalent_rate"
        ],
        "supplied_anchor_sfp_equivalent_agrees_with_family_recovery": chart[
            "supplied_anchor_sfp_equivalent_agrees_with_family_recovery"
        ],
        "all_pairs_third_accuracy": chart["all_pairs_third_accuracy"],
        "support_pair_third_accuracy": chart["support_pair_third_accuracy"],
        "support_pair_other_family_answer_rate": chart[
            "support_pair_other_family_answer_rate"
        ],
        "strict_unique_triples_mean": chart["strict_unique_triples_mean"],
        "transport_decision_rate": transport["transport_decision_rate"],
        "transport_family_rate": transport["transport_family_rate"],
        "transport_supplied_anchor_family_rate": transport[
            "transport_supplied_anchor_family_rate"
        ],
        "condition_failure_counts": chart["condition_failure_counts"],
        "supplied_anchor_condition_failure_counts": chart[
            "supplied_anchor_condition_failure_counts"
        ],
        "final_loss": _round(float(loss.item())),
        "steps": config.train.steps,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "wall_sec": round(time.perf_counter() - started, 3),
    }

    if use_anchor:
        control = dual_anchor_control(model, dual, test_positions)
        row["dual_anchor_control"] = control
        for name in DUAL_METRICS:
            row[name] = control[name]
    return row


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['arm']}|{row['fold_family']}", []).append(row)
    return {
        key: {
            "n_runs": len(group),
            "n_folds": len({row["fold_name"] for row in group}),
            "n_seeds": len({row["seed"] for row in group}),
            **{name: stats([row.get(name) for row in group]) for name in metrics},
        }
        for key, group in sorted(grouped.items())
    }


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def run(config: SweepConfig, *, verbose: bool = True) -> dict:
    started = time.perf_counter()
    dataset = build_dataset()
    codec = SfpCodec()
    problems = build_local_problems(dataset, codec)
    episodes = build_episodes(problems, namespace="primary")
    fresh = build_episodes(problems, namespace="fresh_relabeling")
    dual = build_episodes(
        problems, namespace="dual_control", family="combinatorial_dual"
    )
    folds = [
        fold
        for fold in structural_folds(all_folds(dataset))
        if fold.family in config.families
    ]

    rows: list[dict] = []
    total = len(config.arms) * len(folds) * len(config.seeds)
    done = 0
    for arm in config.arms:
        for fold in folds:
            for seed in config.seeds:
                rows.append(
                    train_one(
                        arm, fold, seed, config, dataset, episodes, fresh, dual
                    )
                )
                done += 1
                if verbose and done % 10 == 0:
                    print(
                        f"  {done}/{total} runs "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )

    return {
        "config": config.as_dict(),
        "n_runs": total,
        "folds": [[fold.family, fold.name] for fold in folds],
        "episode_counts": {
            "primary": len(episodes),
            "fresh_relabeling": len(fresh),
            "dual_control": len(dual),
        },
        "capacity_ledger": capacity_ledger(config.discovery),
        "runs": rows,
        "aggregates": aggregate(rows, RUN_METRICS + DUAL_METRICS),
        "runs_digest": digest([strip_non_replayable(row) for row in rows]),
        "sweep_wall_sec": round(time.perf_counter() - started, 3),
    }


def configurations() -> dict[str, SweepConfig]:
    """The two declared blocks: one primary pass and one diagnosed repair.

    ``009.09`` section 5.3 permits exactly this and forbids a hyperparameter search.
    Between the blocks, only the message form changes -- ``Message(h_j)`` becomes
    ``Message([h_i, h_j])``. Widths, depth, aggregation, optimizer, learning rate,
    step count, seeds, arms, folds, episodes and relabelings are identical. Both
    blocks are reported; the primary block is not deleted.
    """

    return {
        "primary": SweepConfig(discovery=DiscoveryConfig(message_mode="node")),
        "repair": SweepConfig(discovery=DiscoveryConfig(message_mode="edge")),
    }


def smoke_config() -> dict[str, SweepConfig]:
    """A small declared subset: both blocks, both arms, LOHO, one seed, 40 steps."""

    return {
        name: SweepConfig(
            seeds=(0,),
            families=("LOHO",),
            train=TrainConfig(steps=40),
            discovery=config.discovery,
        )
        for name, config in configurations().items()
    }


def _block_laws(body: dict) -> dict:
    rows = body["runs"]
    one = [row for row in rows if row["arm"] == "one_anchor"]
    return {
        "every_run_ran": len(rows) == body["n_runs"],
        "swap_invariance_exact_everywhere": all(
            row["swap_invariance_error"] == 0.0 for row in rows
        )
        and all(row["swap_scores_bitwise_identical"] for row in rows),
        "swap_argmax_never_disagrees": all(
            row["swap_argmax_disagreement_rate"] == 0.0 for row in rows
        ),
        "sfp_equivalent_always_agrees_with_family_recovery": all(
            row["sfp_equivalent_agrees_with_family_recovery"] for row in rows
        )
        and all(
            row["supplied_anchor_sfp_equivalent_agrees_with_family_recovery"]
            for row in rows
        ),
        "capacity_matched_across_arms_within_the_block": len(
            {row["param_count"] for row in rows}
        )
        == 1,
        "dual_control_present_for_every_one_anchor_run": all(
            "dual_anchor_control" in row for row in one
        )
        and bool(one),
        "dual_control_adds_no_parameter": all(
            row["dual_anchor_control"]["adds_no_parameter"] for row in one
        ),
    }


def _headline(body: dict) -> dict:
    aggregates = body["aggregates"]
    return {
        name: {
            key: value[name]["mean"] for key, value in aggregates.items()
        }
        for name in (
            "forced_third_accuracy",
            "anchor_overlap_0_accuracy",
            "anchor_overlap_1_accuracy",
            "train_accuracy",
            "supplied_anchor_chart_valid_rate",
            "supplied_anchor_exact_block_family_recovery",
            "supplied_anchor_sfp_equivalent_rate",
            "chart_valid_rate",
            "exact_block_family_recovery",
            "dual_family_rate",
            "sfp_equivalent_rate",
            "support_pair_third_accuracy",
            "support_pair_other_family_answer_rate",
            "transport_decision_rate",
            "transport_supplied_anchor_family_rate",
            "prediction_on_query_node_rate",
            "dual_anchor_forced_third_accuracy",
            "dual_anchor_dual_family_rate",
            "dual_anchor_certified_family_rate",
        )
    }


def audit(configs: dict[str, SweepConfig] | None = None, *, verbose: bool = True) -> dict:
    configs = configs or configurations()
    blocks = {}
    for name, config in configs.items():
        if verbose:
            print(f"--- block {name} ---", flush=True)
        body = run(config, verbose=verbose)
        body["laws"] = _block_laws(body)
        body["headline"] = _headline(body)
        blocks[name] = body

    laws = {name: body["laws"] for name, body in blocks.items()}
    agrees = all(all(body.values()) for body in laws.values())
    return {
        "module": "discovery_sweep",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "block_names": sorted(blocks),
        "blocks": blocks,
        "laws": laws,
        "block_roles": {
            "primary": (
                "009.09 section 5's one primary architecture pass: m_ij = "
                "Message(h_j). Reported in full including its failure."
            ),
            "repair": (
                "009.09 section 5.3's one clearly diagnosed repair: m_ij = "
                "Message([h_i, h_j]). Only the message form differs from primary."
            ),
        },
        "diagnosed_pathology": dict(DIAGNOSED_PATHOLOGY),
        "declared_ceilings": {
            "one_anchor_forced_third": 1.0,
            "zero_anchor_forced_third": UNIFORM_COMMON_NEIGHBOURS,
            "uniform_eligible_node_chance": _round(UNIFORM_ELIGIBLE_NODES),
        },
        "supplied_sfp_reference": dict(SUPPLIED_SFP_REFERENCE),
        "metric_provenance": (
            "forced_third_accuracy is exact integer identity against the certified "
            "third from task.PairRecord, restricted to one habitat and permuted onto "
            "opaque nodes by discovery_task. The chart audit reuses "
            "discovery_task.is_block_system and "
            "discovery_task.sfp_equivalence_witnesses, so the section 6 conditions "
            "and the AGL(2,2) search are the same code Gate 0 validated."
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "every declared run ran, swap invariance is bitwise everywhere, the "
                "AGL(2,2) equivalence rate agrees with block-family recovery in "
                "every run, both arms carry the identical parameter count, and the "
                "dual-anchor control ran on every one-anchor run without adding a "
                "parameter"
                if agrees
                else "at least one block law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    for name in result["block_names"]:
        block = result["blocks"][name]
        head = block["headline"]
        print(f"--- block {name} ({block['n_runs']} runs) ---", flush=True)
        for metric, cells in head.items():
            print(f"  {metric}", flush=True)
            for cell, value in sorted(cells.items()):
                print(f"    {cell:<26} {value}", flush=True)
        for law, value in sorted(block["laws"].items()):
            if not value:
                print(f"  FAIL law {law}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    configs = smoke_config() if args.smoke else configurations()
    target = SMOKE_OUTPUT if args.smoke else OUTPUT
    result = audit(configs, verbose=not args.check)
    text = render(result)
    _report(result)

    if args.check:
        if not target.exists():
            raise SystemExit(f"FAIL: missing {target}; run without --check first")
        expected = strip_non_replayable(json.loads(target.read_text()))
        observed = strip_non_replayable(json.loads(text))
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived sweep is not identical to {target.name} "
                "(wall-clock keys excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {target.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["laws"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {target.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()


# Referenced for the artifact's replay contract; imported so a drift in
# ladder_sweep's non-replayable key list breaks here rather than silently.
assert "wall_sec" in NON_REPLAYABLE_KEYS

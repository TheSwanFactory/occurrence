"""011.01 sections 4-7: the declared run, and every audit that reads its answers.

One training protocol, applied identically to the main arm and to the
cross-habitat-destroying control::

    2 declared blocks (primary message form, one diagnosed repair)
      x 2 arms (main, localized control)
        x 8 seeds
    = 32 runs

``011.01`` section 4 permits one architecture pass and at most one mechanically
diagnosed repair. Both are run and both are reported. The primary block's failure
is recorded in ``DIAGNOSED_PATHOLOGY`` below and is a *fit* failure: three of eight
seeds finish at the uniform-choice loss with train and test accuracy agreeing
exactly, while five reach the exact information-theoretic loss minimum and score
``1.0000`` on everything. The repair changes one mechanism, the message form.

Where the difficulty lay before that is recorded in
``fano_heads.PRE_SWEEP_VALIDATION``: the readout. A single shared head cannot
serve both queries at all, and two heads over one shared encoder can.

Training uses the ``train`` namespaces only. Every reported number comes from the
``test`` namespaces, except the section 6 fresh-name transport audit, which uses
the separately frozen ``transport`` namespaces and never touches optimization.

What is scored
--------------
``011.01`` section 4.1 scores the other thirteen habitat nodes and asks for the
unique same-``FFF`` mate. Section 4.2 scores all fourteen and asks for the
two-element set of habitats on the third point of the line. Both are read as a
top-``k`` decision with ``k`` the declared answer size, ties broken by low opaque
index so the decision is reproducible.

The representation-recovery audit
--------------------------------
Section 5 asks for more than query accuracy. On each test namespace the trained
model is queried exhaustively: all 14 mate queries, then -- using one
representative habitat per predicted point class -- all 21 class-pair
completions. The predicted mate pairing must be seven disjoint mutual pairs, the
predicted completions must close into seven three-point lines with every point
pair on exactly one line, and the resulting plane must be isomorphic to the
certified Fano plane under an exact search over all ``7!`` bijections.

Section 5 also asks for the entailment relationship to be stated rather than
implied: for a seven-point line system, ``fano_plane_valid`` and
``certified_plane_equivalent`` are **not independent evidence**. Every
three-point-line system on seven points in which each point lies on three lines
and each pair lies on exactly one line *is* the Fano plane -- it is the unique
``2-(7,3,1)`` design. Validity entails equivalence, the two rates agree by
construction, and this module asserts that they agree rather than reporting them
as two findings.

Torch. Not in CI. Roughly an hour for the full 32-run sweep on CPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from fano_baselines import ceiling_compatibility
from fano_heads import (
    COVARIANCE_TOLERANCE,
    FanoConfig,
    TypedGraphPointer,
    build_pointer,
    decision_margin,
    head_of_kind,
    namespace_adjacency,
    query_marks,
    scored_indicator,
    set_valued_loss,
    target_indicator,
    top_k_decision,
    type_channels,
)
from fano_task import (
    ARMS,
    EXPECTED_CEILINGS,
    N_AXES,
    QUERY_KINDS,
    Namespace,
    Observation,
    Query,
    build_localized_observation,
    build_namespaces,
    build_observation,
    build_queries,
    digest,
    line_preserving_bijections,
    rd,
    render,
)
from ladder_sweep import NON_REPLAYABLE_KEYS, strip_non_replayable
from task import build_dataset

from topographo.core import f2_groups as groups

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_fano_artifacts"
OUTPUT = ARTIFACTS / "fano_sweep.json"
SMOKE_OUTPUT = ARTIFACTS / "fano_sweep_smoke.json"

BASE_COMMIT = "d4efc739ffb9ba58a4e79b1b98b3bd0e5aa8cbb2"

#: Frozen before any run. Eight seeds, both arms.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class TrainConfig:
    """The one training protocol. No tuning, no early stopping, no schedule.

    ``steps`` and ``lr`` were fixed by the ``fano_heads`` pre-sweep validation,
    which read training fit only.
    """

    steps: int = 800
    lr: float = 0.003

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "lr": self.lr,
            "optimizer": "Adam",
            "batching": (
                "full batch: every training namespace and every query of both "
                "kinds contributes to every step"
            ),
            "early_stopping": "none",
            "schedule": "none",
            "tuning": "none; identical for both arms and every seed",
        }


# ---------------------------------------------------------------------------
# tensor assembly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KindBatch:
    """One namespace's queries of one kind, as tensors, plus its readout head."""

    kind: str
    head: int
    marks: torch.Tensor
    targets: torch.Tensor
    scored: torch.Tensor
    sizes: tuple[int, ...]
    queries: tuple[Query, ...]


@dataclass(frozen=True)
class Batch:
    """One namespace: a shared adjacency and one ``KindBatch`` per query kind."""

    label: str
    split: str
    adjacency: torch.Tensor
    per_kind: tuple[KindBatch, ...]


def build_batch(
    observation: Observation,
    namespace: Namespace,
    *,
    type_rows: tuple[tuple[float, ...], ...],
) -> Batch:
    all_queries = build_queries(observation, namespace)
    per_kind = []
    for kind in QUERY_KINDS:
        queries = tuple(query for query in all_queries if query.kind == kind)
        per_kind.append(
            KindBatch(
                kind=kind,
                head=head_of_kind(kind),
                marks=torch.tensor(
                    [
                        query_marks(observation, query.query_nodes, type_rows)
                        for query in queries
                    ]
                ),
                targets=torch.tensor(
                    [target_indicator(observation, query) for query in queries]
                ),
                scored=torch.tensor(
                    [scored_indicator(observation, query) for query in queries]
                ),
                sizes=tuple(len(query.target_nodes) for query in queries),
                queries=queries,
            )
        )
    return Batch(
        label=namespace.label,
        split=namespace.split,
        adjacency=torch.tensor([namespace_adjacency(observation, namespace)]),
        per_kind=tuple(per_kind),
    )


def build_batches(
    observation: Observation, namespaces: tuple[Namespace, ...]
) -> dict[str, list[Batch]]:
    type_rows = type_channels(observation)
    out: dict[str, list[Batch]] = defaultdict(list)
    for namespace in namespaces:
        out[namespace.split].append(build_batch(observation, namespace, type_rows=type_rows))
    return dict(out)


def _forward(model: TypedGraphPointer, batch: Batch, kind: KindBatch) -> torch.Tensor:
    rows = kind.marks.shape[0]
    return model(kind.marks, batch.adjacency.expand(rows, -1, -1), kind.head)


# ---------------------------------------------------------------------------
# training and scoring
# ---------------------------------------------------------------------------

def train(model: TypedGraphPointer, batches: list[Batch], config: TrainConfig) -> list[dict]:
    """Full-batch Adam over every training namespace and both query kinds."""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    trace: list[dict] = []
    for step in range(config.steps):
        optimizer.zero_grad()
        total = torch.zeros(())
        terms = 0
        for batch in batches:
            for kind in batch.per_kind:
                scores = _forward(model, batch, kind)
                total = total + set_valued_loss(scores, kind.targets, kind.scored)
                terms += 1
        loss = total / terms
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == config.steps - 1:
            trace.append({"step": step, "loss": rd(float(loss.detach()))})
    return trace


@torch.no_grad()
def score_batch(model: TypedGraphPointer, batch: Batch) -> dict:
    """Per-kind exact accuracy on one namespace, plus the raw decisions."""

    per_kind = {}
    decisions: dict[str, list[tuple[int, ...]]] = {}
    for kind in batch.per_kind:
        scores = _forward(model, batch, kind)
        hits = 0
        picks = []
        margins = []
        for row, query in enumerate(kind.queries):
            pick = top_k_decision(scores[row], kind.scored[row], kind.sizes[row])
            picks.append(pick)
            hits += pick == tuple(sorted(query.target_nodes))
            margins.append(decision_margin(scores[row], kind.scored[row], kind.sizes[row]))
        decisions[kind.kind] = picks
        ordered = sorted(margins)
        per_kind[kind.kind] = {
            "hits": hits,
            "queries": len(kind.queries),
            "accuracy": rd(hits / len(kind.queries)) if kind.queries else None,
            # The margin separates "decided the answer" from "broke a tie by
            # opaque index". A collapsed run answers at a margin near float
            # round-off, which is a tie-break wearing a decision's clothes.
            "median_margin": rd(ordered[len(ordered) // 2]) if ordered else None,
            "decisive_rows": sum(1 for value in margins if value > COVARIANCE_TOLERANCE),
        }
    return {"label": batch.label, "per_kind": per_kind, "decisions": decisions}


# ---------------------------------------------------------------------------
# section 5 — the representation-recovery audit
# ---------------------------------------------------------------------------

def mate_pairing(decisions: dict[int, int]) -> tuple[bool, tuple[tuple[int, ...], ...]]:
    """Are the model's mate answers seven disjoint mutual pairs?"""

    mutual = all(decisions.get(target) == source for source, target in decisions.items())
    if not mutual:
        return False, ()
    pairs = {tuple(sorted((source, target))) for source, target in decisions.items()}
    flat = [node for pair in pairs for node in pair]
    valid = (
        len(pairs) == N_AXES
        and all(len(pair) == 2 and pair[0] != pair[1] for pair in pairs)
        and len(flat) == len(set(flat)) == 2 * N_AXES
    )
    return valid, tuple(sorted(pairs))


def plane_from_completions(
    classes: tuple[tuple[int, ...], ...], triples: set[tuple[int, int, int]]
) -> dict:
    """Is the predicted line system the Fano plane, and is it the certified one?

    ``fano_plane_valid`` and ``certified_plane_equivalent`` are not independent: a
    three-point-line system on seven points with every point on three lines and
    every pair on exactly one line is the unique ``2-(7,3,1)`` design, which is the
    Fano plane. Validity therefore entails equivalence. Both rates are reported
    because the task asks for both, and the entailment is asserted.
    """

    lines = tuple(sorted(triples))
    pair_cover: Counter = Counter()
    for line in lines:
        for pair in itertools.combinations(sorted(line), 2):
            pair_cover[pair] += 1
    point_cover = Counter(point for line in lines for point in line)
    points = tuple(range(len(classes)))
    valid = (
        len(lines) == N_AXES
        and all(len(set(line)) == 3 for line in lines)
        and len(pair_cover) == 21
        and set(pair_cover.values()) == {1}
        and set(point_cover.values()) == {3}
    )
    isomorphisms = (
        line_preserving_bijections(
            lines, points, tuple(tuple(line) for line in groups.LINES3), groups.POINTS3
        )
        if valid
        else ()
    )
    return {
        "lines": [list(line) for line in lines],
        "line_count": len(lines),
        "covered_point_pairs": len(pair_cover),
        "pair_multiplicities": sorted(set(pair_cover.values())),
        "lines_through_a_point": sorted(set(point_cover.values())),
        "fano_plane_valid": valid,
        "isomorphisms_to_certified_plane": len(isomorphisms),
        "certified_plane_equivalent": len(isomorphisms) > 0,
        "entailment": (
            "validity entails equivalence: the unique 2-(7,3,1) design is the Fano "
            "plane, so these two are one finding reported twice, not two "
            "independent findings"
        ),
    }


@torch.no_grad()
def recovery_audit(
    model: TypedGraphPointer,
    observation: Observation,
    namespace: Namespace,
    *,
    type_rows: tuple[tuple[float, ...], ...],
) -> dict:
    """Query the trained model exhaustively and reconstruct what it implies."""

    adjacency = torch.tensor([namespace_adjacency(observation, namespace)])
    all_queries = build_queries(observation, namespace)

    # -- step 1: the mate pairing ------------------------------------------
    mate_queries = [query for query in all_queries if query.kind == "mate"]
    marks = torch.tensor(
        [query_marks(observation, query.query_nodes, type_rows) for query in mate_queries]
    )
    scores = model(
        marks, adjacency.expand(len(mate_queries), -1, -1), head_of_kind("mate")
    )
    predicted: dict[int, int] = {}
    for row, query in enumerate(mate_queries):
        scored = torch.tensor(scored_indicator(observation, query))
        predicted[query.query_nodes[0]] = top_k_decision(scores[row], scored, 1)[0]
    pairing_valid, pairs = mate_pairing(predicted)

    if not pairing_valid:
        return {
            "namespace": namespace.label,
            "mate_pairing_valid": False,
            "predicted_pairs": [],
            "class_pair_completions_scored": 0,
            "completions_naming_one_class": 0,
            "completion_coherence_rate": 0.0,
            "line_count": 0,
            "fano_plane_valid": False,
            "certified_plane_equivalent": False,
        }

    # -- step 2: collapse into anonymous point classes ---------------------
    classes = pairs
    class_of_node = {node: index for index, pair in enumerate(classes) for node in pair}
    representative = tuple(pair[0] for pair in classes)

    # -- step 3: completion on all 21 class pairs --------------------------
    class_pairs = tuple(itertools.combinations(range(len(classes)), 2))
    completion_marks = torch.tensor(
        [
            query_marks(
                observation,
                tuple(sorted((representative[a], representative[b]))),
                type_rows,
            )
            for a, b in class_pairs
        ]
    )
    completion_scores = model(
        completion_marks,
        adjacency.expand(len(class_pairs), -1, -1),
        head_of_kind("completion"),
    )
    scored_all = torch.tensor(
        [
            1.0 if observation.node_type(node) == "habitat" else 0.0
            for node in range(observation.n_nodes)
        ]
    )
    triples: set[tuple[int, int, int]] = set()
    coherent = 0
    for row, (first, second) in enumerate(class_pairs):
        pick = top_k_decision(completion_scores[row], scored_all, 2)
        implied = {class_of_node[node] for node in pick}
        if len(implied) == 1:
            third = implied.pop()
            if third not in (first, second):
                coherent += 1
                triples.add(tuple(sorted((first, second, third))))
    plane = plane_from_completions(classes, triples)

    return {
        "namespace": namespace.label,
        "mate_pairing_valid": True,
        "predicted_pairs": [list(pair) for pair in pairs],
        "class_pair_completions_scored": len(class_pairs),
        "completions_naming_one_class": coherent,
        "completion_coherence_rate": rd(coherent / len(class_pairs)),
        **plane,
    }


# ---------------------------------------------------------------------------
# section 6 — fresh-name transport and the global L/R swap
# ---------------------------------------------------------------------------

@torch.no_grad()
def transport_audit(
    model: TypedGraphPointer,
    observation: Observation,
    reference: Namespace,
    fresh: tuple[Namespace, ...],
    *,
    type_rows: tuple[tuple[float, ...], ...],
) -> dict:
    """Do decisions transport under namespaces the model never trained on?"""

    reference_perm = reference.node_permutation(observation)
    canonical_order = sorted(range(observation.n_nodes), key=lambda node: reference_perm[node])
    reference_batch = build_batch(observation, reference, type_rows=type_rows)
    reference_scored = score_batch(model, reference_batch)

    exact = 0
    total = 0
    per_kind_exact: Counter = Counter()
    per_kind_total: Counter = Counter()
    for namespace in fresh:
        transport = tuple(namespace.node_permutation(observation)[c] for c in canonical_order)
        batch = build_batch(observation, namespace, type_rows=type_rows)
        scored = score_batch(model, batch)
        for kind in QUERY_KINDS:
            for row, decision in enumerate(reference_scored["decisions"][kind]):
                wanted = tuple(sorted(transport[node] for node in decision))
                agrees = wanted == scored["decisions"][kind][row]
                total += 1
                exact += agrees
                per_kind_total[kind] += 1
                per_kind_exact[kind] += agrees
    return {
        "reference_namespace": reference.label,
        "fresh_namespaces": [namespace.label for namespace in fresh],
        "decisions_compared": total,
        "decisions_transported_exactly": exact,
        "transport_rate": rd(exact / total) if total else None,
        "per_kind": {
            kind: rd(per_kind_exact[kind] / per_kind_total[kind])
            for kind in QUERY_KINDS
            if per_kind_total[kind]
        },
        "statement": (
            "the same structural query is asked in a namespace the model never saw, "
            "and the decision must be the image of the reference decision under the "
            "relabelling; anything less is a dependence on opaque names"
        ),
    }


@torch.no_grad()
def left_right_swap_audit(
    model: TypedGraphPointer,
    observation: Observation,
    namespace: Namespace,
    *,
    type_rows: tuple[tuple[float, ...], ...],
) -> dict:
    """The global L/R namespace swap must not move a habitat answer.

    ``011.01`` section 6 asks for this test if the representation permits it. It
    does: Gate 0 verifies the swap is an automorphism of the observation that fixes
    every habitat node, so every habitat-valued answer must be unchanged -- not
    merely isomorphic.
    """

    if observation.n_left != observation.n_right:
        return {"applicable": False, "reason": "the two axis namespaces differ in size"}
    swapped = Namespace(
        label=f"{namespace.label}/lr-swap",
        split=namespace.split,
        arm=namespace.arm,
        habitat=namespace.habitat,
        event=namespace.event,
        left=namespace.right,
        right=namespace.left,
    )
    base = score_batch(model, build_batch(observation, namespace, type_rows=type_rows))
    other = score_batch(model, build_batch(observation, swapped, type_rows=type_rows))
    identical = 0
    total = 0
    for kind in QUERY_KINDS:
        for row, decision in enumerate(base["decisions"][kind]):
            total += 1
            identical += decision == other["decisions"][kind][row]
    return {
        "applicable": True,
        "decisions_compared": total,
        "decisions_identical": identical,
        "identical_rate": rd(identical / total),
        "accuracy_before": base["per_kind"],
        "accuracy_after": other["per_kind"],
        "statement": (
            "the swap is a Gate-0-verified automorphism fixing every habitat node, "
            "so an answer that moves under it is an answer that depends on which "
            "axis namespace a node was drawn from"
        ),
    }


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def run_cell(
    observation: Observation,
    batches: dict[str, list[Batch]],
    namespaces: tuple[Namespace, ...],
    *,
    arm: str,
    block: str,
    seed: int,
    config: FanoConfig,
    train_config: TrainConfig,
    full_audits: bool,
) -> dict:
    started = time.time()
    type_rows = type_channels(observation)
    model = build_pointer(config, seed=seed)
    trace = train(model, batches["train"], train_config)
    model.eval()

    per_split = {}
    for split in ("train", "validation", "test"):
        rows = [score_batch(model, batch) for batch in batches[split]]
        hits: Counter = Counter()
        totals: Counter = Counter()
        for row in rows:
            for kind, entry in row["per_kind"].items():
                hits[kind] += entry["hits"]
                totals[kind] += entry["queries"]
        decisive: Counter = Counter()
        margins: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for kind, entry in row["per_kind"].items():
                decisive[kind] += entry["decisive_rows"]
                margins[kind].append(entry["median_margin"])
        per_split[split] = {
            kind: {
                "hits": hits[kind],
                "queries": totals[kind],
                "accuracy": rd(hits[kind] / totals[kind]),
                "decisive_rows": decisive[kind],
                "decisive_fraction": rd(decisive[kind] / totals[kind]),
                "median_margin": rd(
                    sorted(margins[kind])[len(margins[kind]) // 2]
                ),
            }
            for kind in sorted(totals)
        }

    out = {
        "arm": arm,
        "block": block,
        "seed": seed,
        "config": config.as_dict(),
        "loss_trace": trace,
        "final_loss": trace[-1]["loss"],
        "per_split": per_split,
        "mate_exact_accuracy": per_split["test"]["mate"]["accuracy"],
        "fano_completion_exact_set_accuracy": per_split["test"]["completion"]["accuracy"],
        "wall_sec": rd(time.time() - started),
    }

    if full_audits:
        test_namespaces = [n for n in namespaces if n.split == "test"]
        transport_namespaces = tuple(n for n in namespaces if n.split == "transport")
        recovery = [
            recovery_audit(model, observation, namespace, type_rows=type_rows)
            for namespace in test_namespaces
        ]
        out["recovery"] = {
            "per_namespace": recovery,
            "namespaces": len(recovery),
            "mate_pairing_valid_rate": rd(
                sum(1 for row in recovery if row["mate_pairing_valid"]) / len(recovery)
            ),
            "fano_plane_valid_rate": rd(
                sum(1 for row in recovery if row["fano_plane_valid"]) / len(recovery)
            ),
            "certified_plane_equivalent_rate": rd(
                sum(1 for row in recovery if row["certified_plane_equivalent"]) / len(recovery)
            ),
        }
        out["transport"] = transport_audit(
            model, observation, test_namespaces[0], transport_namespaces, type_rows=type_rows
        )
        out["left_right_swap"] = left_right_swap_audit(
            model, observation, test_namespaces[0], type_rows=type_rows
        )
    return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepConfig:
    block: str
    message_mode: str
    seeds: tuple[int, ...] = SEEDS
    arms: tuple[str, ...] = ARMS
    steps: int = 800
    full_audits: bool = True

    def as_dict(self) -> dict:
        return {
            "block": self.block,
            "message_mode": self.message_mode,
            "seeds": list(self.seeds),
            "arms": list(self.arms),
            "steps": self.steps,
            "full_audits": self.full_audits,
        }


#: The mechanically diagnosed failure of the primary block, and the one repair.
#:
#: Recorded here, at the point where the blocks are declared, rather than argued
#: for in the result turn. The *evidence* is the primary block's own per-seed
#: numbers in this artifact, not this text.
DIAGNOSED_PATHOLOGY = {
    "name": "the uniform-loss plateau on three of eight seeds",
    "symptom": (
        "five seeds reach final loss 0.346574 -- the exact information-theoretic "
        "minimum -- and score 1.0000 on both queries, the whole-plane audit and "
        "fresh-name transport. Three seeds finish at 2.52 to 2.60, which is the "
        "uniform-choice loss"
    ),
    "why_it_is_a_fit_failure_not_a_generalization_failure": (
        "train and test accuracy agree on every stalled cell to within 0.02: seed 3 "
        "is 0.000/0.000 on both splits, seed 5 is 1.000 mate and 0.000 completion on "
        "both, and seed 7 is 1.000 mate with completion 0.750 train against 0.765 "
        "test. A run that does no better on the data it optimized than on data it "
        "never saw has not overfitted -- it has not fitted"
    ),
    "not_expressivity": (
        "the same architecture at five other seeds reaches the exact loss minimum "
        "on the same data, so the family can represent the answer"
    ),
    "cause": (
        "with m_ij = Message(h_j) the receiver's own state enters only through the "
        "update's concatenation, after the neighbour sum has already been formed. "
        "The Gate-0 path needs a per-edge conjunction -- this axis node is in the "
        "query's support AND that Event's other axis is not -- and assembling it "
        "downstream of the sum leaves a flat region the optimizer can settle in"
    ),
    "repair": (
        "condition the message on the ordered edge: m_ij = Message([h_i, h_j]). One "
        "change, same family, same widths, same depth, same optimizer, same step "
        "count, same seeds, same observations, no new hyperparameter"
    ),
    "discipline": (
        "011.01 section 4 permits one architecture pass and at most one "
        "mechanically diagnosed repair. Both blocks are run and both are reported; "
        "the primary block is not deleted and its three failed seeds are not "
        "rounded away"
    ),
}


def configurations() -> tuple[SweepConfig, ...]:
    """The two declared blocks. ``011.01`` section 4 permits exactly these.

    The primary is the frozen architecture with neighbour-only messages. The
    repair changes one architectural mechanism -- the message form -- in response
    to the mechanically diagnosed plateau recorded in ``DIAGNOSED_PATHOLOGY``.
    Nothing else differs: same widths, same depth, same optimizer, same steps,
    same seeds, same observations.
    """

    return (
        SweepConfig(block="primary", message_mode="node"),
        SweepConfig(block="repair", message_mode="edge"),
    )


def smoke_config() -> SweepConfig:
    return SweepConfig(
        block="smoke",
        message_mode="node",
        seeds=(0,),
        arms=("main",),
        steps=40,
        full_audits=False,
    )


def run_sweep(configs: tuple[SweepConfig, ...]) -> dict:
    main = build_observation(build_dataset())
    observations = {"main": main, "localized": build_localized_observation(main)}
    namespaces = {arm: build_namespaces(observations[arm]) for arm in observations}
    batches = {arm: build_batches(observations[arm], namespaces[arm]) for arm in observations}

    runs: list[dict] = []
    for config in configs:
        for arm in config.arms:
            for seed in config.seeds:
                runs.append(
                    run_cell(
                        observations[arm],
                        batches[arm],
                        namespaces[arm],
                        arm=arm,
                        block=config.block,
                        seed=seed,
                        config=FanoConfig(message_mode=config.message_mode),
                        train_config=TrainConfig(steps=config.steps),
                        full_audits=config.full_audits,
                    )
                )
    return {
        "configurations": [config.as_dict() for config in configs],
        "train_protocol": TrainConfig().as_dict(),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _spread(values: list[float]) -> dict:
    return {
        "mean": rd(sum(values) / len(values)),
        "min": rd(min(values)),
        "max": rd(max(values)),
        "per_seed": [rd(value) for value in values],
    }


def summarize(runs: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["block"], run["arm"])].append(run)

    out = {}
    for (block, arm), rows in sorted(grouped.items()):
        entry = {
            "seeds": len(rows),
            "final_loss": _spread([row["final_loss"] for row in rows]),
            "mate_exact_accuracy": _spread(
                [row["mate_exact_accuracy"] for row in rows]
            ),
            "fano_completion_exact_set_accuracy": _spread(
                [row["fano_completion_exact_set_accuracy"] for row in rows]
            ),
        }
        if "recovery" in rows[0]:
            for name in (
                "mate_pairing_valid_rate",
                "fano_plane_valid_rate",
                "certified_plane_equivalent_rate",
            ):
                entry[name] = _spread([row["recovery"][name] for row in rows])
            entry["transport_rate"] = _spread(
                [row["transport"]["transport_rate"] for row in rows]
            )
            entry["left_right_swap_identical_rate"] = _spread(
                [row["left_right_swap"]["identical_rate"] for row in rows]
            )
            entry["validity_entails_equivalence"] = all(
                row["recovery"]["fano_plane_valid_rate"]
                == row["recovery"]["certified_plane_equivalent_rate"]
                for row in rows
            )
        out[f"{block}/{arm}"] = entry
    return out


def control_checks(summary: dict, runs: list[dict], test_namespaces: int) -> dict:
    """Does the control stay at its exact information ceiling?"""

    out = {}
    for key, entry in summary.items():
        block, arm = key.split("/")
        if arm != "localized":
            continue
        rows = [run for run in runs if run["block"] == block and run["arm"] == arm]
        out[block] = {
            "mate": ceiling_compatibility(
                entry["mate_exact_accuracy"]["mean"],
                sum(row["per_split"]["test"]["mate"]["queries"] for row in rows),
                EXPECTED_CEILINGS["localized_mate"],
                blocks=test_namespaces * len(rows),
            ),
            "completion": ceiling_compatibility(
                entry["fano_completion_exact_set_accuracy"]["mean"],
                sum(row["per_split"]["test"]["completion"]["queries"] for row in rows),
                EXPECTED_CEILINGS["localized_completion"],
                blocks=test_namespaces * len(rows),
            ),
        }
    return out


def audit(configs: tuple[SweepConfig, ...]) -> dict:
    payload = run_sweep(configs)
    runs = payload["runs"]
    summary = summarize(runs)
    test_namespaces = len(
        [n for n in build_namespaces(build_observation(build_dataset())) if n.split == "test"]
    )
    controls = control_checks(summary, runs, test_namespaces)

    return {
        "module": "fano_sweep",
        "purpose": (
            "011.01 sections 4-7: the declared training runs for both arms, the "
            "representation-recovery audit, the fresh-name transport audit and the "
            "global L/R swap audit"
        ),
        "configurations": payload["configurations"],
        "train_protocol": payload["train_protocol"],
        "diagnosed_pathology": DIAGNOSED_PATHOLOGY,
        "runs": runs,
        "summary": summary,
        "control_vs_exact_ceiling": controls,
        "non_replayable_keys": sorted(NON_REPLAYABLE_KEYS),
        "digests": {
            "runs_digest": digest([strip_non_replayable(run) for run in runs])
        },
        "provenance": {
            "base_commit": BASE_COMMIT,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare the runs digest")
    parser.add_argument(
        "--smoke", action="store_true", help="run the declared 1-seed smoke subset"
    )
    args = parser.parse_args()

    configs = (smoke_config(),) if args.smoke else configurations()
    output = SMOKE_OUTPUT if args.smoke else OUTPUT
    started = time.time()
    payload = audit(configs)
    text = render(payload)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not output.exists():
            raise SystemExit(f"FAIL: {output} does not exist; run without --check")
        rebuilt = [strip_non_replayable(run) for run in payload["runs"]]
        stored = [
            strip_non_replayable(run) for run in json.loads(output.read_text())["runs"]
        ]
        if digest(rebuilt) != digest(stored):
            raise SystemExit(f"FAIL: {output} runs digest does not match the rebuild")
        print(f"PASS: {output.name} replays to an identical runs digest")
        return
    output.write_text(text)
    print(f"wrote {output}  ({rd(time.time() - started)} s)")
    for name, entry in sorted(payload["summary"].items()):
        line = (
            f"  {name:20s} mate {entry['mate_exact_accuracy']['mean']:.4f}"
            f"  completion {entry['fano_completion_exact_set_accuracy']['mean']:.4f}"
        )
        if "fano_plane_valid_rate" in entry:
            line += (
                f"  plane {entry['fano_plane_valid_rate']['mean']:.4f}"
                f"  transport {entry['transport_rate']['mean']:.4f}"
            )
        print(line)


if __name__ == "__main__":
    main()

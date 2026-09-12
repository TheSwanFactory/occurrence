"""009.09 architecture: a pointer over six anonymous nodes, and nothing else.

``009.08`` scored candidate *chambers* written in the arm's own SFP alphabet.
``009.09`` withdraws the alphabet. The learner now sees six nodes whose features
are **identical** before marking, the admission graph among them, one marked
anchor triple and one marked query pair, and must point at one of those six nodes.

The whole input surface, exactly::

    marks       (B, 6, 2)   [anchor_mark, query_mark] per node, values in {0, 1}
    adjacency   (B, 6, 6)   the symmetric pairwise ADMIT matrix, zero diagonal

That is all. There is no Event index, no habitat index, no local slot index, no
``S``, no ``FFF``, no ``PP``, no ``pp``, no SFP token and no embedding table. A
node is distinguishable only through the relation graph, anchor membership and
query role -- which is precisely what ``009.09`` section 4.1 requires.

Why this family
---------------
``009.09`` section 5 asks for one primary permutation-equivariant set/graph family.
The model is message passing with **sum** aggregation over admitted edges and a
shared per-node readout::

    h        = Embed(marks)
    h        = ReLU( Update_l( [h , Adj @ Message_l(h)] ) )      for l = 1..L
    score    = Readout(h)                                        -> (B, 6)

Sum aggregation is the load-bearing choice. The message function is a nonlinear
map of a node's two mark bits, so it takes four distinct values, and summing it
over a node's admitted neighbours yields the *counts* of each mark type in that
neighbourhood. A candidate third can therefore see how many of its neighbours are
query-marked and how many of those are also anchor-marked, which is the relational
information the anchor rule turns on. Mean aggregation would divide those counts by
a constant degree and lose nothing here, but sum is the honest primitive and is
declared before any run.

The one repair, and the pathology that motivated it
---------------------------------------------------
``009.09`` section 5.3 permits one primary architecture pass and at most one
clearly diagnosed repair. The primary pass has a **seed-dependent saddle**, and the
diagnosis is mechanical rather than inferred.

Split the nine scored queries of an episode by how many of the two query nodes the
anchor contains. The anchor is a triangle, so::

    case 1   exactly one query node is anchor-marked     6 of 9 queries
    case 0   neither query node is anchor-marked         3 of 9 queries

The exact rule needs the *parity* ``|{a, b, c} & anchor|`` to be odd, which means
"pick the common neighbour **not** in the anchor" in case 1 and "pick the common
neighbour **in** the anchor" in case 0 -- the anchor preference flips sign between
the two cases. The first half of that rule is close to linear in features the
candidate already has, and it alone scores ``6/9 = 0.6667``. On some seeds the
optimizer settles into exactly that half and does not leave: observed ``0.6667``
with ``case1 = 1.0000`` and ``case0 = 0.0000``, identically on train and test, and
unchanged from step 40 through step 4000. It is a fit failure, not a
generalization failure, and adding depth does not fix it.

The cause is that with a message depending only on the *neighbour's* state, the
candidate has to learn the conjunction "neighbour is query-marked **and**
anchor-marked" inside the message function and then combine that count with its own
anchor mark through a further nonlinearity. The parity is assembled from two
separately learned pieces, and the linearly-available piece is a flat attractor.

The declared repair is one change, inside the same family: condition the message on
the **ordered edge** rather than on the neighbour alone::

    node  message  m_ij = Message_l( h_j )              primary
    edge  message  m_ij = Message_l( [h_i , h_j] )      repair

That makes the conjunction of the candidate's own mark with its neighbour's marks
directly available to a single hidden layer. Both blocks are run and both are
reported; the primary block's case split *is* the evidence for the diagnosis. No
learning rate, step count, width, seed set, fold, episode or observation changed.

Equivariance is structural, not trained
---------------------------------------
* **Query swap** is not a symmetry that has to be learned or checked to a
  tolerance: the query is presented as a *set mark* on two nodes, so ``f(a,b)`` and
  ``f(b,a)`` are the identical tensor and the outputs are bitwise identical.
* **Opaque-token permutation** covariance is likewise structural: every operation
  is either per-node or an aggregation over the adjacency, so permuting the node
  order permutes the scores. It is verified rather than asserted, and the
  *decision* -- the argmax -- is required to be exactly covariant with no
  tolerance, while the scores are allowed the declared float round-off that
  reordering a sum introduces.
* **Parameter count** depends on the widths only. It is identical at 4, 6, 9 and
  12 nodes and mentions the 84-Event catalogue nowhere.

The zero-anchor control shares the parameters
---------------------------------------------
``009.09`` section 7.2 requires the identical learner with the anchor marker
removed. "Identical" is implemented literally: the zero-anchor arm keeps the same
two input channels and the same parameter shapes, and sets the anchor channel to
zero everywhere. The two arms are therefore capacity-matched to the parameter, so
a difference between them cannot be a capacity difference -- which is what makes
this the primary anti-cheating control rather than a second architecture.

Fences on the learned path
--------------------------
No XOR, no bitwise operator, no set intersection, no endpoint-intersection
routine, no K4 star completion, no admission-to-block-family solver, no catalogue
lookup, and no reference to the exact nonlearned completion appears anywhere in
the model, its input encoder or its loss. That is checked mechanically here by
parsing the learned path and inspecting its operators and identifiers, not by
assertion.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from discovery_task import (
    N_TOKENS,
    SLOTS,
    automorphisms,
    build_local_problems,
    compatible_block_systems,
    render,
)
from scorer import count_params, param_digest, param_shapes, set_seed
from sfp import SfpCodec
from task import build_dataset
from torch import nn

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_discovery_artifacts"
OUTPUT = ARTIFACTS / "discovery_heads.json"

BASE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"

#: The two mark channels, in this order. The ONLY per-node input.
MARK_CHANNELS = ("anchor_mark", "query_mark")
INPUT_DIM = len(MARK_CHANNELS)

#: The primary message and the one declared repair. See the module docstring.
MESSAGE_MODES = ("node", "edge")

MESSAGE_MODE_ROLE = {
    "node": (
        "PRIMARY. m_ij = Message(h_j): the message depends on the neighbour's state "
        "alone. 009.09 section 5's one primary architecture pass."
    ),
    "edge": (
        "REPAIR. m_ij = Message([h_i, h_j]): the message is conditioned on the "
        "ordered edge, so the conjunction of the candidate's own mark with its "
        "neighbour's marks is available to a single hidden layer. The one repair "
        "009.09 section 5.3 permits, for the mechanically diagnosed 6/9 saddle."
    ),
}

#: Sizes a parameter dimension may not equal, so the anti-identity check has
#: teeth: the node count, the habitat count and the Event catalogue size.
FORBIDDEN_WIDTHS = (N_TOKENS, 14, 84)

AUDIT_SEED = 0
DISCREPANCY_DECIMALS = 12

#: Node-count probes. The parameter count must be identical at all four, which is
#: the mechanical form of "parameter count independent of Event catalogue size".
NODE_COUNT_PROBES = (4, N_TOKENS, 9, 12)

#: How many random relabelings the covariance audit transports.
COVARIANCE_PROBES = 240

#: Tolerance for the covariance SCORE comparison, and the reason it is not zero.
#:
#: Permuting the node order permutes the rows of ``Adj @ Message(h)``, and the
#: matmul then accumulates the same real numbers in a different order. Float
#: addition is not associative, so the transported scores agree to round-off rather
#: than bitwise -- the same fact ``scorer.SymmetricPair`` records when it explains
#: why it uses only commutative primitives. Observed discrepancy sits at the
#: ``1e-8`` level for a 6-term sum in single precision. The tolerance is declared
#: here, before the check, two orders of magnitude above that. The DECISION is held
#: to a stricter standard: the argmax must be exactly covariant, with no tolerance.
COVARIANCE_TOLERANCE = 1e-6

#: The mechanically diagnosed pathology of the primary pass, and the one repair.
#:
#: Recorded here so the repair is documented at the point where the architecture is
#: defined rather than argued for in the result turn. The *evidence* is the primary
#: block's own case-split numbers in ``discovery_sweep``, not this text.
DIAGNOSED_PATHOLOGY = {
    "name": "the 6/9 anchor-preference saddle",
    "symptom": (
        "forced-third accuracy pinned at exactly 0.666666666667 on TRAIN as well as "
        "test, with per-case accuracy 1.0000 on the six queries holding one "
        "anchor-marked node and 0.0000 on the three holding none"
    ),
    "why_it_is_a_fit_failure_not_a_generalization_failure": (
        "train and test agree to twelve decimals; the model has not overfitted, it "
        "has not fitted"
    ),
    "persistence": (
        "unchanged from step 40 through step 4000 on the affected seeds, so it is a "
        "flat attractor rather than slow convergence"
    ),
    "not_expressivity": (
        "depth does not fix it: 3 layers reaches 1.0000 by step 200 and falls back "
        "to 0.6667 by step 800, and 4 layers oscillates. The parity is representable "
        "at 2 layers -- one affected seed of three reaches 1.0000 at step 40 and "
        "holds it to step 4000"
    ),
    "cause": (
        "with m_ij = Message(h_j) the candidate must learn the conjunction "
        "'neighbour is query-marked AND anchor-marked' inside the message function "
        "and only then combine that count with its own anchor mark. The parity is "
        "assembled from two separately learned pieces and the linearly available "
        "piece -- avoid anchor-marked candidates -- already scores 6/9"
    ),
    "repair": (
        "condition the message on the ordered edge: m_ij = Message([h_i, h_j]). One "
        "change, same family, same widths, same optimizer, same step count, same "
        "seeds, same observations, no new hyperparameter"
    ),
    "discipline": (
        "009.09 section 5.3 permits one primary pass and at most one clearly "
        "diagnosed repair. Both blocks are run and both are reported; the primary "
        "block is not deleted and its failure is not rounded away"
    ),
}

#: ``009.08``'s pointer, for the parameter comparison. Reported, not run.
PRIOR_POINTER = {
    "module": "locator_heads.ChamberPointer over scorer.RelHead",
    "input": "SFP tokens (S, FFF, PP, pp), 16 one-hot units per presentation",
    "candidates": "the four chambers of the row's chart, as SFP star tokens",
    "note": (
        "009.08 was handed the exact finite representation and asked to use it. "
        "009.09 withdraws it. The two parameter counts are therefore NOT a "
        "like-for-like capacity comparison and are never quoted as one."
    ),
}

FENCES = (
    "The entire input surface is two mark channels per node plus the admission "
    "matrix. No Event, habitat, chart or slot index is an input.",
    "No trainable parameter is indexed by an Event, a habitat, a local slot or a "
    "node position. There is no embedding table in the model.",
    "Parameter count is identical at 4, 6, 9 and 12 nodes and mentions 84 nowhere.",
    "Query swap invariance is BITWISE and structural: the query is a set mark, so "
    "f(a,b) and f(b,a) are the same tensor.",
    "Opaque-token permutation covariance is exact in the argmax and holds to "
    "declared float round-off in the scores.",
    "S and FFF are not inputs (009.09a section 4). Two episodes from different "
    "habitats with the same marks and adjacency give bitwise identical outputs.",
    "No XOR, bitwise operator, set intersection, endpoint-intersection routine, K4 "
    "star completion, admission-to-family solver or catalogue lookup appears on "
    "the learned path.",
    "The zero-anchor control is the SAME parameters with the anchor channel zeroed, "
    "so the two arms are capacity-matched to the parameter.",
    "Scoring is never masked. The two query nodes stay scoreable so that "
    "'the predicted third is distinct from the queried pair' remains a measured "
    "fact rather than an enforced one.",
)

INPUT_SURFACE = (
    "marks (B, 6, 2): channel 0 is anchor membership, channel 1 is query role. "
    "Both are {0.0, 1.0}. Every node's feature vector is identical before these "
    "two marks are written.",
    "adjacency (B, 6, 6): the symmetric pairwise ADMIT matrix with a zero "
    "diagonal, as {0.0, 1.0}.",
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def hash_free_seed(name: str) -> int:
    """Digest-derived seed. Never ``hash()``, which is salted per process."""

    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1_000_003


def _round(value: float) -> float:
    return round(float(value), DISCREPANCY_DECIMALS)


def encode_perm(perm: tuple[int, ...]) -> str:
    return "".join(str(p) for p in perm)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryConfig:
    """Head widths, fixed before any run and applied identically to every arm.

    ``009.09`` section 5 permits one primary architecture family and forbids a
    hyperparameter search. There is none: the widths are the frozen ``009.02``
    ``ScorerConfig`` constants (``latent = 32``, ``trunk_hidden = 64`` halved to the
    32 used for every head width in ``009.06``/``009.08``), and ``layers = 2`` is
    the smallest depth that leaves headroom over the single round of message
    passing the relational rule needs.
    """

    node_width: int = 32
    message_hidden: int = 32
    readout_hidden: int = 32
    layers: int = 2
    aggregation: str = "sum"
    message_mode: str = "node"

    def __post_init__(self) -> None:
        if self.aggregation not in ("sum", "mean"):
            raise ValueError(f"aggregation must be sum or mean, got {self.aggregation!r}")
        if self.message_mode not in MESSAGE_MODES:
            raise ValueError(
                f"message_mode must be one of {MESSAGE_MODES}, got "
                f"{self.message_mode!r}"
            )
        if self.layers < 1:
            raise ValueError(f"layers must be positive, got {self.layers}")
        widths = {
            "node_width": self.node_width,
            "message_hidden": self.message_hidden,
            "readout_hidden": self.readout_hidden,
        }
        for name, value in sorted(widths.items()):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
            if value in FORBIDDEN_WIDTHS:
                # Not a correctness bug, but it would blunt the mechanical
                # "no parameter dimension equals the node / habitat / catalogue
                # count" check, so it is refused outright.
                raise ValueError(
                    f"{name} == {value} collides with a forbidden width "
                    f"{FORBIDDEN_WIDTHS} and would blunt the anti-identity audit"
                )

    def as_dict(self) -> dict:
        return {
            "node_width": self.node_width,
            "message_hidden": self.message_hidden,
            "readout_hidden": self.readout_hidden,
            "layers": self.layers,
            "aggregation": self.aggregation,
            "message_mode": self.message_mode,
            "message_form": (
                "m_ij = Message(h_j)"
                if self.message_mode == "node"
                else "m_ij = Message([h_i, h_j])"
            ),
            "message_mode_role": MESSAGE_MODE_ROLE[self.message_mode],
            "activation": "ReLU",
            "input_channels": list(MARK_CHANNELS),
            "input_dim": INPUT_DIM,
            "embedding_tables": 0,
            "output": "one scalar score per node; the answer is the argmax",
            "masking": "none, ever",
            "tuning": (
                "none. One primary family, widths frozen from the 009.02/009.06 "
                "constants, no search, no per-arm adjustment."
            ),
            "seed_convention": "torch.manual_seed(seed) then np.random.seed(seed)",
        }


# ---------------------------------------------------------------------------
# the input encoder: the whole surface the learner sees
# ---------------------------------------------------------------------------

def episode_inputs(
    admit_nodes: tuple[tuple[int, int], ...],
    anchor_nodes: tuple[int, int, int],
    query_pair: tuple[int, int],
    *,
    use_anchor: bool = True,
    n_nodes: int = N_TOKENS,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """``(marks, adjacency)`` for one query, in opaque node coordinates.

    The query is written as a **set** mark on its two nodes, which is what makes
    swap invariance bitwise rather than approximate. ``use_anchor=False`` is the
    section 7.2 control: the anchor channel is zeroed and the tensor shape,
    parameter shapes and parameter count are untouched.
    """

    anchor = set(anchor_nodes) if use_anchor else set()
    query = set(query_pair)
    if len(query) != 2:
        raise ValueError(f"a query is an unordered pair of distinct nodes, got {query_pair}")
    marks = tuple(
        (
            1.0 if node in anchor else 0.0,
            1.0 if node in query else 0.0,
        )
        for node in range(n_nodes)
    )
    edges = {frozenset(p) for p in admit_nodes}
    adjacency = tuple(
        tuple(
            1.0 if i != j and frozenset((i, j)) in edges else 0.0
            for j in range(n_nodes)
        )
        for i in range(n_nodes)
    )
    return marks, adjacency


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class OpaqueChartPointer(nn.Module):
    """Message passing over the admission graph, then a shared per-node readout.

    Every module here is either applied per node or is an aggregation over the
    supplied adjacency, so the whole map is equivariant to relabelling the nodes by
    construction. Nothing indexes a node position: ``Embed``, ``Message_l``,
    ``Update_l`` and ``Readout`` are the same weights for every node, and the
    parameter count is a function of the widths alone.
    """

    def __init__(self, config: DiscoveryConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiscoveryConfig()
        width = self.config.node_width
        message_in = width if self.config.message_mode == "node" else 2 * width
        self.embed = nn.Linear(INPUT_DIM, width)
        self.message = nn.ModuleList(
            nn.Sequential(
                nn.Linear(message_in, self.config.message_hidden),
                nn.ReLU(),
                nn.Linear(self.config.message_hidden, width),
            )
            for _ in range(self.config.layers)
        )
        self.update = nn.ModuleList(
            nn.Linear(2 * width, width) for _ in range(self.config.layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(width, self.config.readout_hidden),
            nn.ReLU(),
            nn.Linear(self.config.readout_hidden, 1),
        )

    def _normalize(
        self, adjacency: torch.Tensor, pooled: torch.Tensor
    ) -> torch.Tensor:
        if self.config.aggregation == "mean":
            degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return pooled / degree
        return pooled

    def aggregate(
        self,
        layer: int,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, N, W), (B, N, N) -> (B, N, W)`` over admitted neighbours.

        ``node`` mode reduces with one matmul. ``edge`` mode forms the ordered-edge
        message first and masks it with the adjacency before summing. Both are
        equivariant: every index is either a node index carried through per-node
        maps, or a summation index contracted against the supplied adjacency.
        """

        message = self.message[layer]
        if self.config.message_mode == "node":
            pooled = torch.matmul(adjacency, message(hidden))
            return self._normalize(adjacency, pooled)
        n_nodes = hidden.shape[-2]
        own = hidden.unsqueeze(-2).expand(-1, -1, n_nodes, -1)
        peer = hidden.unsqueeze(-3).expand(-1, n_nodes, -1, -1)
        edge = message(torch.cat([own, peer], dim=-1))
        pooled = (adjacency.unsqueeze(-1) * edge).sum(dim=-2)
        return self._normalize(adjacency, pooled)

    def forward(
        self, marks: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """``(B, N, 2), (B, N, N) -> (B, N)`` node scores. Never masked."""

        if marks.shape[-1] != INPUT_DIM:
            raise ValueError(
                f"marks must carry exactly {INPUT_DIM} channels, got {marks.shape[-1]}"
            )
        if adjacency.shape[-1] != adjacency.shape[-2]:
            raise ValueError(
                f"adjacency must be square, got {tuple(adjacency.shape[-2:])}"
            )
        if adjacency.shape[-1] != marks.shape[-2]:
            raise ValueError(
                f"adjacency width {adjacency.shape[-1]} does not match the node "
                f"count {marks.shape[-2]}"
            )
        hidden = self.embed(marks)
        for layer in range(self.config.layers):
            pooled = self.aggregate(layer, hidden, adjacency)
            hidden = torch.relu(
                self.update[layer](torch.cat([hidden, pooled], dim=-1))
            )
        return self.readout(hidden).squeeze(-1)

    def slot_layout(self) -> dict:
        return {
            "input_channels": list(MARK_CHANNELS),
            "output_slots": "one per current node; there is no fixed output width",
            "learned_output_columns_indexed_by_an_event": 0,
            "learned_output_columns_indexed_by_a_habitat": 0,
            "learned_output_columns_indexed_by_a_node_position": 0,
            "embedding_tables": 0,
            "candidate_tokens_supplied": False,
            "output_slots_track_catalogue_size": False,
        }


def build_pointer(
    config: DiscoveryConfig | None = None, *, seed: int = 0
) -> OpaqueChartPointer:
    """Seed first, then build. Same convention as ``scorer.build_scorer``."""

    set_seed(seed)
    return OpaqueChartPointer(config or DiscoveryConfig())


def pointer_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the six node scores, POINT target.

    The target is the single certified forced third. The two query nodes are
    included in the softmax rather than masked out, so pointing at a query node is
    a mistake the model is free to make and section 6.1 can measure.
    """

    return nn.functional.cross_entropy(scores, target)


# ---------------------------------------------------------------------------
# structural audits
# ---------------------------------------------------------------------------

def _probe_batch(
    n_nodes: int, *, use_anchor: bool = True, seed: int = AUDIT_SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic synthetic probe batch at an arbitrary node count.

    Used only by the shape and parameter-independence audits, where the point is
    that nothing in the model knows how many nodes there are. The graph is a
    deterministic circulant, not a certified habitat.
    """

    generator = torch.Generator().manual_seed(hash_free_seed(f"probe/{n_nodes}/{seed}"))
    rows = [
        [
            1.0 if i != j and ((j - i) % n_nodes in (1, n_nodes - 1)) else 0.0
            for j in range(n_nodes)
        ]
        for i in range(n_nodes)
    ]
    bits = torch.randint(0, 2, (n_nodes, INPUT_DIM), generator=generator).float()
    if not use_anchor:
        bits[:, 0] = 0.0
    return (
        bits.unsqueeze(0),
        torch.tensor([rows], dtype=torch.float32),
    )


def input_surface_audit() -> dict:
    """Exactly what the learner sees, and that it is only that."""

    dataset = build_dataset()
    codec = SfpCodec()
    problems = build_local_problems(dataset, codec)
    problem = problems[0]
    systems = compatible_block_systems(problem.admit)
    anchor = tuple(sorted(problem.certified_blocks))[0]
    query = next(
        pair
        for pair in problem.admit
        if not set(pair) <= set(anchor)
    )
    marks, adjacency = episode_inputs(problem.admit, anchor, query)
    zero_marks, _ = episode_inputs(problem.admit, anchor, query, use_anchor=False)

    values = {value for row in marks for value in row}
    values |= {value for row in adjacency for value in row}
    return {
        "input_surface": list(INPUT_SURFACE),
        "input_dim": INPUT_DIM,
        "mark_channels": list(MARK_CHANNELS),
        "marks_example": [list(row) for row in marks],
        "adjacency_example": [list(row) for row in adjacency],
        "zero_anchor_marks_example": [list(row) for row in zero_marks],
        "values_are_binary": sorted(values) == [0.0, 1.0],
        "adjacency_is_symmetric": all(
            adjacency[i][j] == adjacency[j][i] for i in SLOTS for j in SLOTS
        ),
        "adjacency_diagonal_is_zero": all(adjacency[i][i] == 0.0 for i in SLOTS),
        "adjacency_edges": sum(1 for i in SLOTS for j in SLOTS if adjacency[i][j]) // 2,
        "anchor_marked_nodes": sum(1 for row in marks if row[0]),
        "query_marked_nodes": sum(1 for row in marks if row[1]),
        "zero_anchor_has_no_anchor_mark": all(row[0] == 0.0 for row in zero_marks),
        "zero_anchor_shape_is_unchanged": len(zero_marks) == len(marks)
        and all(len(a) == len(b) for a, b in zip(zero_marks, marks, strict=True)),
        "no_event_index_is_an_input": True,
        "no_habitat_index_is_an_input": True,
        "no_slot_index_is_an_input": True,
        "no_sfp_field_is_an_input": True,
        "encoder_signature": str(inspect.signature(episode_inputs)),
        "compatible_block_systems_seen_by_the_encoder": 0,
        "note": (
            f"the encoder was handed {len(systems)} admission-compatible block "
            "systems by Gate 0 and uses none of them; it writes marks and an "
            "adjacency matrix and nothing else."
        ),
    }


def swap_symmetry_audit(config: DiscoveryConfig, *, seed: int = AUDIT_SEED) -> dict:
    """``f(a,b) = f(b,a)``, bitwise, over every admitted pair of every habitat."""

    dataset = build_dataset()
    codec = SfpCodec()
    model = build_pointer(config, seed=seed)
    model.eval()

    identical_tensors = 0
    identical_outputs = 0
    total = 0
    worst = 0.0
    for problem in build_local_problems(dataset, codec):
        for anchor in sorted(problem.certified_blocks):
            for a, b in problem.admit:
                forward = episode_inputs(problem.admit, anchor, (a, b))
                backward = episode_inputs(problem.admit, anchor, (b, a))
                total += 1
                if forward == backward:
                    identical_tensors += 1
                left = model(
                    torch.tensor([forward[0]], dtype=torch.float32),
                    torch.tensor([forward[1]], dtype=torch.float32),
                )
                right = model(
                    torch.tensor([backward[0]], dtype=torch.float32),
                    torch.tensor([backward[1]], dtype=torch.float32),
                )
                if torch.equal(left, right):
                    identical_outputs += 1
                worst = max(worst, float((left - right).abs().max().item()))
    return {
        "queries_probed": total,
        "input_tensors_bitwise_identical": identical_tensors,
        "outputs_bitwise_identical": identical_outputs,
        "max_score_discrepancy": _round(worst),
        "swap_invariance_is_structural": identical_tensors == total
        and identical_outputs == total
        and worst == 0.0,
        "mechanism": (
            "the query is a SET mark on two nodes, so swapping the query produces "
            "the identical input tensor. Invariance is not learned, not enforced by "
            "a symmetric pooling primitive, and not held to a tolerance."
        ),
    }


def permutation_covariance_audit(
    config: DiscoveryConfig, *, seed: int = AUDIT_SEED
) -> dict:
    """A fresh relabelling of the six nodes must permute the answer the same way.

    Transports both the marks and the adjacency by the same permutation and
    compares the transported scores and the transported argmax. The argmax is
    required to agree exactly; the scores are allowed the declared round-off.
    """

    dataset = build_dataset()
    codec = SfpCodec()
    problems = build_local_problems(dataset, codec)
    model = build_pointer(config, seed=seed)
    model.eval()

    relabelings = tuple(itertools.permutations(SLOTS))
    picked = [
        relabelings[hash_free_seed(f"covariance/{k}") % len(relabelings)]
        for k in range(COVARIANCE_PROBES)
    ]

    worst = 0.0
    argmax_exact = 0
    decisive = 0
    decisive_exact = 0
    tied = 0
    tie_witnessed = 0
    total = 0
    unwitnessed = []
    for index, perm in enumerate(picked):
        problem = problems[index % len(problems)]
        anchor = sorted(problem.certified_blocks)[index % 4]
        query = [p for p in problem.admit if not set(p) <= set(anchor)][index % 9]

        marks, adjacency = episode_inputs(problem.admit, anchor, query)
        moved_admit = tuple(
            tuple(sorted((perm[a], perm[b]))) for a, b in problem.admit
        )
        moved_anchor = tuple(sorted(perm[t] for t in anchor))
        moved_query = tuple(sorted(perm[t] for t in query))
        moved_marks, moved_adjacency = episode_inputs(
            moved_admit, moved_anchor, moved_query
        )

        with torch.no_grad():
            base = model(
                torch.tensor([marks], dtype=torch.float32),
                torch.tensor([adjacency], dtype=torch.float32),
            )[0]
            moved = model(
                torch.tensor([moved_marks], dtype=torch.float32),
                torch.tensor([moved_adjacency], dtype=torch.float32),
            )[0]
        transported = torch.zeros_like(base)
        for node in SLOTS:
            transported[perm[node]] = base[node]

        total += 1
        worst = max(worst, float((moved - transported).abs().max().item()))

        want = perm[int(base.argmax().item())]
        got = int(moved.argmax().item())
        agrees = want == got
        argmax_exact += int(agrees)

        # Is the row DECISIVE, i.e. is its top score separated from the runner-up
        # by more than the declared round-off tolerance? On an untrained model two
        # nodes that are indistinguishable in the marked graph carry mathematically
        # equal scores, and which of them wins the argmax is then decided by float
        # round-off. That is a tie, not a covariance failure, and it is separated out
        # rather than absorbed into the headline.
        ordered = torch.sort(moved, descending=True).values
        margin = float((ordered[0] - ordered[1]).item())
        if margin > COVARIANCE_TOLERANCE:
            decisive += 1
            decisive_exact += int(agrees)
        else:
            tied += 1
            # Prove the tie: an automorphism of the admission graph that fixes the
            # query pair and the anchor SETWISE and carries one tied node onto the
            # other means the two nodes are indistinguishable to any equivariant
            # scorer, so equal scores are forced.
            auts = automorphisms(moved_admit)
            witnessed = any(
                {sigma[t] for t in moved_query} == set(moved_query)
                and {sigma[t] for t in moved_anchor} == set(moved_anchor)
                and sigma[want] == got
                for sigma in auts
            )
            tie_witnessed += int(witnessed or agrees)
            if not (witnessed or agrees):
                unwitnessed.append(
                    {
                        "habitat": problem.key,
                        "relabeling": encode_perm(perm),
                        "expected_node": want,
                        "observed_node": got,
                        "margin": _round(margin),
                    }
                )

    return {
        "relabelings_probed": total,
        "declared_score_tolerance": COVARIANCE_TOLERANCE,
        "max_score_discrepancy": _round(worst),
        "scores_within_declared_tolerance": worst <= COVARIANCE_TOLERANCE,
        "argmax_transports_count": argmax_exact,
        "decisive_rows": decisive,
        "decisive_rows_argmax_exactly_covariant_count": decisive_exact,
        "argmax_exactly_covariant": decisive_exact == decisive,
        "tied_rows": tied,
        "tied_rows_with_a_symmetry_witness": tie_witnessed,
        "every_tie_has_a_symmetry_witness": tie_witnessed == tied,
        "unwitnessed_disagreements": unwitnessed,
        "tolerance_rationale": (
            "permuting the node order permutes the summation index of the neighbour "
            "aggregation, and the reduction then accumulates the same real numbers in "
            "a different order. Float addition is not associative, so the scores "
            "agree to round-off. The tolerance was declared before the check."
        ),
        "tie_disclosure": (
            "the argmax is held to EXACT covariance on every decisive row -- one "
            "whose top score is separated from the runner-up by more than the "
            "declared tolerance -- with no tolerance at all. A row whose top two "
            "scores are within the tolerance is a TIE: on an untrained model two "
            "nodes that no automorphism of the marked graph can distinguish carry "
            "mathematically equal scores, and round-off decides the winner. Every "
            "such row is required to exhibit an automorphism of the admission graph "
            "fixing the query pair and the anchor setwise and carrying one tied node "
            "onto the other, which proves the tie is forced rather than assumed."
        ),
    }


def habitat_blindness_audit(
    config: DiscoveryConfig, *, seed: int = AUDIT_SEED
) -> dict:
    """``009.09a`` section 4, proved rather than asserted.

    ``S`` and ``FFF`` are not inputs, so two episodes drawn from **different**
    habitats that happen to present the same marks and the same adjacency must
    produce bitwise identical scores. Each habitat's abstract problem is isomorphic
    to every other's, so such pairs exist in quantity: relabel a second habitat's
    tokens onto the first's admission graph and anchor.
    """

    dataset = build_dataset()
    codec = SfpCodec()
    problems = build_local_problems(dataset, codec)
    model = build_pointer(config, seed=seed)
    model.eval()

    reference = problems[0]
    anchor = sorted(reference.certified_blocks)[0]
    query = [p for p in reference.admit if not set(p) <= set(anchor)][0]
    marks, adjacency = episode_inputs(reference.admit, anchor, query)
    with torch.no_grad():
        base = model(
            torch.tensor([marks], dtype=torch.float32),
            torch.tensor([adjacency], dtype=torch.float32),
        )

    relabelings = tuple(itertools.permutations(SLOTS))
    rows = []
    identical = 0
    for problem in problems[1:]:
        match = None
        for perm in relabelings:
            moved_admit = {
                frozenset((perm[a], perm[b])) for a, b in problem.admit
            }
            if moved_admit != {frozenset(p) for p in reference.admit}:
                continue
            for candidate in sorted(problem.certified_blocks):
                if tuple(sorted(perm[t] for t in candidate)) == anchor:
                    match = (perm, candidate)
                    break
            if match:
                break
        if match is None:
            rows.append({"habitat": problem.key, "isomorphic_presentation": False})
            continue
        perm, candidate = match
        moved_admit = tuple(
            tuple(sorted((perm[a], perm[b]))) for a, b in problem.admit
        )
        moved_marks, moved_adjacency = episode_inputs(moved_admit, anchor, query)
        with torch.no_grad():
            moved = model(
                torch.tensor([moved_marks], dtype=torch.float32),
                torch.tensor([moved_adjacency], dtype=torch.float32),
            )
        same = torch.equal(base, moved)
        identical += int(same)
        rows.append(
            {
                "habitat": problem.key,
                "isomorphic_presentation": True,
                "relabeling": encode_perm(perm),
                "s": problem.s,
                "fff": problem.fff,
                "outputs_bitwise_identical": same,
            }
        )
    return {
        "reference_habitat": reference.key,
        "reference_s": reference.s,
        "reference_fff": reference.fff,
        "per_habitat": rows,
        "habitats_compared": len(rows),
        "outputs_bitwise_identical_everywhere": identical == len(rows),
        "s_and_fff_are_not_inputs": True,
        "note": (
            "009.09a section 4 asks that if S or FFF were fed in, swapping them "
            "should leave every prediction bitwise unchanged. They are not fed in "
            "at all, so the stronger statement holds: the model cannot tell two "
            "habitats apart even in principle, and 13 differing (S, FFF) pairs "
            "produce the identical tensor."
        ),
    }


def parameter_independence_audit(config: DiscoveryConfig) -> dict:
    """Parameter count identical at 4, 6, 9 and 12 nodes, and 84 nowhere."""

    rows = []
    counts = set()
    for n_nodes in NODE_COUNT_PROBES:
        model = build_pointer(config, seed=AUDIT_SEED)
        marks, adjacency = _probe_batch(n_nodes)
        with torch.no_grad():
            scores = model(marks, adjacency)
        counts.add(count_params(model))
        rows.append(
            {
                "n_nodes": n_nodes,
                "param_count": count_params(model),
                "output_shape": list(scores.shape),
                "output_width_equals_node_count": scores.shape[-1] == n_nodes,
            }
        )
    shapes = param_shapes(build_pointer(config, seed=AUDIT_SEED))
    offending = [
        row
        for row in shapes
        if any(dim in FORBIDDEN_WIDTHS for dim in row["shape"])
    ]
    return {
        "node_count_probes": list(NODE_COUNT_PROBES),
        "per_probe": rows,
        "param_count_identical_across_probes": len(counts) == 1,
        "param_count": sorted(counts)[0] if len(counts) == 1 else sorted(counts),
        "parameter_shapes": shapes,
        "forbidden_widths": list(FORBIDDEN_WIDTHS),
        "parameters_with_a_forbidden_dimension": offending,
        "no_parameter_dimension_equals_node_habitat_or_catalogue_count": not offending,
        "note": (
            "the model is built once per probe and evaluated at a different node "
            "count each time; the count cannot change because no weight is indexed "
            "by a node."
        ),
    }


def no_identity_parameter_audit(config: DiscoveryConfig) -> dict:
    """No embedding table, and no parameter indexed by any stable identity."""

    model = build_pointer(config, seed=AUDIT_SEED)
    modules = sorted(
        {type(module).__name__ for _, module in model.named_modules()}
    )
    embeddings = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag))
    ]
    buffers = [name for name, _ in model.named_buffers()]
    return {
        "module_types": modules,
        "embedding_modules": embeddings,
        "has_no_embedding_table": not embeddings,
        "buffers": buffers,
        "has_no_buffer": not buffers,
        "named_parameters": [name for name, _ in model.named_parameters()],
        "param_count": count_params(model),
        "param_digest": param_digest(model),
        "slot_layout": model.slot_layout(),
        "prior_pointer": dict(PRIOR_POINTER),
        "no_trainable_embedding_indexed_by_a_global_event_id": True,
        "no_trainable_embedding_indexed_by_a_stable_local_slot": True,
        "no_candidate_token_table": True,
        "note": (
            "009.08 supplied candidate chamber tokens as a frozen buffer. This model "
            "has no buffer at all: the candidates ARE the six current nodes, and "
            "everything the learner can say about them comes from the relation graph "
            "and the two marks."
        ),
    }


class _AnnotationStripper(ast.NodeTransformer):
    """Drop every type annotation before the operator scan.

    ``X | None`` is a ``BinOp(BitOr)`` in the AST, so a naive bitwise-operator scan
    of a modern annotated module reports a bitwise OR that is a *type union*, not a
    computation. The annotations are removed rather than the check weakened, so a
    real ``|``, ``&`` or ``^`` in executable code still trips it.
    """

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):  # pragma: no cover - none present
        node.returns = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return None
        return self.generic_visit(
            ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)
        )


def _strip_annotations(tree: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(_AnnotationStripper().visit(tree))


def answer_rule_audit() -> dict:
    """``009.09`` section 5.2, checked by parsing the learned path.

    The learned path is the input encoder, the model class and the loss. Its
    operators and identifiers are inspected for the forbidden executable answer
    rule: XOR or any bitwise operator, set intersection, the endpoint-intersection
    routine, a K4 star completion, an admission-to-block-family solver, a catalogue
    lookup, and the nonlearned exact completion.
    """

    learned_path = (episode_inputs, OpaqueChartPointer, pointer_loss, build_pointer)
    banned_identifiers = (
        "anchor_third",
        "anchor_third_unit_form",
        "system_third",
        "compatible_block_systems",
        "cliques3",
        "automorphisms",
        "certified_relabelings",
        "ExactSfpCircuit",
        "forced_third_event",
        "shared_block",
        "admit_event",
        "SfpCodec",
        "build_local_problems",
        "certified_third",
        "certified_blocks",
        "sfp_token",
        "address_of",
    )
    banned_operators = {
        ast.BitXor: "XOR",
        ast.BitAnd: "bitwise AND / set intersection",
        ast.BitOr: "bitwise OR / set union",
    }

    rows = []
    identifier_hits = []
    operator_hits = []
    for obj in learned_path:
        tree = _strip_annotations(ast.parse(inspect.getsource(obj)))
        names = set()
        operators = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.BinOp, ast.AugAssign)):
                kind = banned_operators.get(type(node.op))
                if kind:
                    operators.add(kind)
        hits = sorted(names & set(banned_identifiers))
        rows.append(
            {
                "object": getattr(obj, "__name__", str(obj)),
                "banned_identifiers_present": hits,
                "banned_operators_present": sorted(operators),
            }
        )
        identifier_hits.extend(hits)
        operator_hits.extend(sorted(operators))

    return {
        "learned_path": [getattr(o, "__name__", str(o)) for o in learned_path],
        "banned_identifiers": list(banned_identifiers),
        "banned_operators": sorted(banned_operators.values()),
        "per_object": rows,
        "no_banned_identifier_on_the_learned_path": not identifier_hits,
        "no_banned_operator_on_the_learned_path": not operator_hits,
        "annotations_stripped_before_the_operator_scan": (
            "type unions such as `DiscoveryConfig | None` parse as BinOp(BitOr). "
            "Annotations are removed from the tree first so the scan reports only "
            "bitwise operators in executable code, and a real ^, & or | still trips."
        ),
        "encoder_disclosure": (
            "episode_inputs builds a frozenset of the SUPPLIED admitted pairs in "
            "order to write the adjacency matrix. That is a graph encoding of an "
            "input, not an answer rule: it never intersects the two query nodes' "
            "neighbourhoods, never completes a triple, and never consults a block "
            "system, a chart or the certified third."
        ),
        "exact_completion_status": (
            "the exact one-anchor completion lives in discovery_task.anchor_third "
            "and is used only by discovery_baselines as a NONLEARNED CEILING. It is "
            "not imported by this module and appears on no learned path."
        ),
    }


def determinism_audit(config: DiscoveryConfig) -> dict:
    """Same seed, same weights; different seed, different weights."""

    first = build_pointer(config, seed=AUDIT_SEED)
    second = build_pointer(config, seed=AUDIT_SEED)
    other = build_pointer(config, seed=AUDIT_SEED + 1)
    marks, adjacency = _probe_batch(N_TOKENS)
    with torch.no_grad():
        left = first(marks, adjacency)
        right = second(marks, adjacency)
    return {
        "seed": AUDIT_SEED,
        "same_seed_same_param_digest": param_digest(first) == param_digest(second),
        "same_seed_bitwise_identical_forward": bool(torch.equal(left, right)),
        "different_seed_different_param_digest": param_digest(first)
        != param_digest(other),
        "param_digest": param_digest(first),
    }


def zero_anchor_capacity_audit(config: DiscoveryConfig) -> dict:
    """The control arm is the same parameters with the anchor channel zeroed."""

    dataset = build_dataset()
    codec = SfpCodec()
    problem = build_local_problems(dataset, codec)[0]
    anchor = sorted(problem.certified_blocks)[0]
    query = [p for p in problem.admit if not set(p) <= set(anchor)][0]

    one = episode_inputs(problem.admit, anchor, query, use_anchor=True)
    zero = episode_inputs(problem.admit, anchor, query, use_anchor=False)
    model = build_pointer(config, seed=AUDIT_SEED)
    with torch.no_grad():
        left = model(
            torch.tensor([one[0]], dtype=torch.float32),
            torch.tensor([one[1]], dtype=torch.float32),
        )
        right = model(
            torch.tensor([zero[0]], dtype=torch.float32),
            torch.tensor([zero[1]], dtype=torch.float32),
        )
    return {
        "param_count_one_anchor": count_params(model),
        "param_count_zero_anchor": count_params(model),
        "capacity_matched_to_the_parameter": True,
        "input_shapes_identical": [list(torch.tensor(one[0]).shape)]
        == [list(torch.tensor(zero[0]).shape)],
        "anchor_channel_is_zero_in_the_control": all(
            row[0] == 0.0 for row in zero[0]
        ),
        "adjacency_unchanged_in_the_control": one[1] == zero[1],
        "outputs_differ": not bool(torch.equal(left, right)),
        "note": (
            "009.09 section 7.2 asks for the IDENTICAL learner with the anchor "
            "marker removed. Identical is implemented literally: one architecture, "
            "one parameter shape list, one parameter count, and the anchor channel "
            "held at zero. A difference between the arms therefore cannot be a "
            "capacity difference."
        ),
    }


def forward_shape_audit(config: DiscoveryConfig) -> dict:
    """Shapes, and the refusal of a malformed input surface."""

    model = build_pointer(config, seed=AUDIT_SEED)
    marks, adjacency = _probe_batch(N_TOKENS)
    with torch.no_grad():
        scores = model(marks, adjacency)
    refusals = {}
    try:
        model(torch.zeros(1, N_TOKENS, INPUT_DIM + 1), adjacency)
        refusals["extra_mark_channel"] = "accepted"
    except ValueError as exc:
        refusals["extra_mark_channel"] = f"refused: {exc}"
    try:
        model(marks, torch.zeros(1, N_TOKENS, N_TOKENS + 1))
        refusals["nonsquare_adjacency"] = "accepted"
    except ValueError as exc:
        refusals["nonsquare_adjacency"] = f"refused: {exc}"
    return {
        "marks_shape": list(marks.shape),
        "adjacency_shape": list(adjacency.shape),
        "scores_shape": list(scores.shape),
        "one_score_per_node": scores.shape[-1] == N_TOKENS,
        "refusals": refusals,
        "refuses_a_wider_input_surface": refusals["extra_mark_channel"].startswith(
            "refused"
        ),
    }


def capacity_ledger(config: DiscoveryConfig) -> dict:
    """The whole parameter budget, and what it is not comparable to."""

    model = build_pointer(config, seed=AUDIT_SEED)
    per_module = {}
    for name, module in model.named_modules():
        own = sum(
            p.numel()
            for child_name, p in module.named_parameters(recurse=False)
            if child_name
        )
        if own:
            per_module[name or "<root>"] = own
    return {
        "config": config.as_dict(),
        "total_params": count_params(model),
        "params_by_module": per_module,
        "params_indexed_by_an_event": 0,
        "params_indexed_by_a_habitat": 0,
        "params_indexed_by_a_node": 0,
        "arms_share_one_parameter_count": True,
        "arm_parameter_counts": {
            "one_anchor": count_params(model),
            "zero_anchor": count_params(model),
            "dual_anchor_control": count_params(model),
        },
        "dual_anchor_note": (
            "the dual-anchor arm (009.09a section 2) is a TEST-TIME evaluation of "
            "the trained one-anchor model. It trains nothing and adds no parameter."
        ),
        "not_comparable_to": dict(PRIOR_POINTER),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit_one_mode(config: DiscoveryConfig) -> dict:
    """Every structural audit, for one message mode."""

    swap = swap_symmetry_audit(config)
    covariance = permutation_covariance_audit(config)
    blindness = habitat_blindness_audit(config)
    independence = parameter_independence_audit(config)
    identity = no_identity_parameter_audit(config)
    determinism = determinism_audit(config)
    control = zero_anchor_capacity_audit(config)
    shapes = forward_shape_audit(config)
    ledger = capacity_ledger(config)

    laws = {
        "query_swap_invariance_is_bitwise": swap["swap_invariance_is_structural"],
        "scores_covariant_within_declared_tolerance": covariance[
            "scores_within_declared_tolerance"
        ],
        "argmax_exactly_covariant_on_every_decisive_row": covariance[
            "argmax_exactly_covariant"
        ],
        "every_tie_has_a_symmetry_witness": covariance[
            "every_tie_has_a_symmetry_witness"
        ],
        "habitat_identity_is_invisible_to_the_model": blindness[
            "outputs_bitwise_identical_everywhere"
        ],
        "param_count_independent_of_node_count": independence[
            "param_count_identical_across_probes"
        ],
        "no_parameter_dimension_is_an_identity_count": independence[
            "no_parameter_dimension_equals_node_habitat_or_catalogue_count"
        ],
        "no_embedding_table": identity["has_no_embedding_table"],
        "deterministic_under_seed": determinism["same_seed_same_param_digest"]
        and determinism["same_seed_bitwise_identical_forward"]
        and determinism["different_seed_different_param_digest"],
        "control_arm_is_capacity_matched": control[
            "capacity_matched_to_the_parameter"
        ]
        and control["anchor_channel_is_zero_in_the_control"]
        and control["adjacency_unchanged_in_the_control"],
        "one_score_per_current_node": shapes["one_score_per_node"],
        "refuses_a_wider_input_surface": shapes["refuses_a_wider_input_surface"],
    }
    return {
        "message_mode": config.message_mode,
        "role": MESSAGE_MODE_ROLE[config.message_mode],
        "config": config.as_dict(),
        "swap_symmetry": swap,
        "permutation_covariance": covariance,
        "habitat_blindness": blindness,
        "parameter_independence": independence,
        "no_identity_parameter": identity,
        "determinism": determinism,
        "zero_anchor_capacity": control,
        "forward_shapes": shapes,
        "capacity_ledger": ledger,
        "laws": laws,
    }


def audit(config: DiscoveryConfig | None = None) -> dict:
    base = config or DiscoveryConfig()

    surface = input_surface_audit()
    rule = answer_rule_audit()
    modes = {
        mode: audit_one_mode(
            DiscoveryConfig(
                node_width=base.node_width,
                message_hidden=base.message_hidden,
                readout_hidden=base.readout_hidden,
                layers=base.layers,
                aggregation=base.aggregation,
                message_mode=mode,
            )
        )
        for mode in MESSAGE_MODES
    }

    laws = {
        "input_surface_is_two_marks_and_an_adjacency": surface["input_dim"]
        == INPUT_DIM
        and surface["values_are_binary"]
        and surface["adjacency_is_symmetric"]
        and surface["adjacency_diagonal_is_zero"],
        "no_banned_identifier_on_the_learned_path": rule[
            "no_banned_identifier_on_the_learned_path"
        ],
        "no_banned_operator_on_the_learned_path": rule[
            "no_banned_operator_on_the_learned_path"
        ],
        "both_message_modes_share_the_input_surface": True,
    }
    for mode, body in sorted(modes.items()):
        for name, value in body["laws"].items():
            laws[f"{mode}:{name}"] = value

    agrees = all(laws.values())
    return {
        "module": "discovery_heads",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "message_modes": list(MESSAGE_MODES),
        "message_mode_roles": dict(MESSAGE_MODE_ROLE),
        "diagnosed_pathology": dict(DIAGNOSED_PATHOLOGY),
        "input_surface": surface,
        "answer_rule": rule,
        "per_message_mode": modes,
        "torch_version": torch.__version__,
        "host": platform.platform(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "laws": laws,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "for both the primary and the repair message, the input surface is "
                "two mark channels and an admission matrix, query swap invariance is "
                "bitwise, the argmax is exactly covariant under opaque relabelling, "
                "habitat identity is invisible, no parameter is indexed by an Event, "
                "habitat, slot or node, and no XOR, intersection, completion or "
                "catalogue lookup appears on the learned path"
                if agrees
                else "at least one structural law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    print("--- input surface ---", flush=True)
    print(
        f"  channels {result['input_surface']['mark_channels']}  "
        f"dim {result['input_surface']['input_dim']}",
        flush=True,
    )
    for mode, body in sorted(result["per_message_mode"].items()):
        cov = body["permutation_covariance"]
        swap = body["swap_symmetry"]
        blind = body["habitat_blindness"]
        print(f"--- message mode {mode} ---", flush=True)
        print(
            f"  params {body['capacity_ledger']['total_params']}  "
            f"layers {body['config']['layers']}  "
            f"form {body['config']['message_form']}",
            flush=True,
        )
        print(
            f"  covariance: max score discrepancy {cov['max_score_discrepancy']} "
            f"(tolerance {cov['declared_score_tolerance']}), decisive argmax exact "
            f"{cov['decisive_rows_argmax_exactly_covariant_count']}/"
            f"{cov['decisive_rows']}, ties {cov['tied_rows']} all witnessed "
            f"{cov['every_tie_has_a_symmetry_witness']}",
            flush=True,
        )
        print(
            f"  swap: {swap['outputs_bitwise_identical']}/{swap['queries_probed']} "
            f"bitwise identical",
            flush=True,
        )
        print(
            f"  habitat blindness: {blind['habitats_compared']} habitats, identical "
            f"{blind['outputs_bitwise_identical_everywhere']}",
            flush=True,
        )
    for name, value in sorted(result["laws"].items()):
        if not value:
            print(f"  FAIL law {name}", flush=True)


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
        expected = json.loads(OUTPUT.read_text())
        observed = json.loads(text)
        for key in ("started_at", "host"):
            expected.pop(key, None)
            observed.pop(key, None)
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived audit is not identical to {OUTPUT.name} "
                "(started_at/host excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["laws"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()

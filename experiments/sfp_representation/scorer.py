"""The shared candidate scorer for the admit-and-force relation (Issue 009, Task 7).

This module owns the *architecture* and the mechanical proof of its structural
properties. It owns no training loop, no metrics suite, no baseline and no sweep
driver; those belong to later tasks and are deliberately absent.

Schematic form, exactly as the controlling task stipulates::

    u_a = Enc(rep(a))
    u_b = Enc(rep(b))
    u_c = Enc(rep(c))

    pair = SymmetricPair(u_a, u_b)
    score(c | a,b)      = Rel(pair, u_c)
    score(bottom | a,b) = Bottom(pair)

Why there is no 84-way output matrix
------------------------------------
A free per-Event output column would let the learner memorise "the answer for
this input pair is column 57" without ever representing the relation. So the
candidate side of the score is produced by the **same encoder** that reads the
input Events: a candidate ``c`` is scored through ``u_c = Enc(rep(c))``, and the
only candidate-facing parameters are the shared ``Rel`` key map. Consequently

* the parameter count does not depend on how many candidate Events exist — shown
  mechanically against stand-in catalogues of size 40, 84 and 120;
* no parameter tensor carries a dimension equal to the catalogue size in an
  output-column role — checked mechanically over ``named_parameters()``;
* the candidate token table is a registered **buffer** with ``requires_grad``
  false, so it never becomes a trainable per-Event embedding.

Why swapping ``a`` and ``b`` is a symmetry by construction
----------------------------------------------------------
``SymmetricPair`` is built only from primitives that are commutative *bitwise*
in IEEE-754: ``u_a + u_b``, ``u_a * u_b`` and ``|u_a - u_b|``. Float addition and
multiplication are commutative to the bit (they are not associative, which is a
different property and is never relied on here), and ``u_a - u_b`` is the exact
negation of ``u_b - u_a``, so ``abs`` erases the order. Every downstream layer
therefore receives a bitwise-identical input under the swap and returns a
bitwise-identical output. The expected max swap discrepancy is exactly ``0.0``
and that is what is reported — no tolerance is claimed or needed.

Pooling over tokens, and Arm B's endpoint symmetry
--------------------------------------------------
An arm may present an Event as several token vectors (Arm B's SFP incidence
presentation uses two). Those tokens go through **one** shared per-token trunk —
no per-slot parameters exist — and are then pooled by mean. Mean over two
elements is bitwise order-independent, so an arm whose two tokens are the two
endpoint presentations of the same incidence is endpoint-symmetric *by
construction*, not by training. That is tested too.

The arms interface: protocol, not import
----------------------------------------
``arms.py`` is written in parallel by another task. This module deliberately does
**not** import it. Instead it declares :class:`ArmLike`, a structural
(duck-typed) protocol matching the coordinator's frozen contract, validates any
supplied object against that contract in :func:`validate_arm`, and ships
:class:`SyntheticArm` — a deterministic stand-in — so every structural property
here is testable today and the same code accepts the real arms tomorrow with no
edit. The ledger below is likewise a closed form in ``token_dim``, so it is
correct for whatever token widths the real arms turn out to declare; the concrete
rows use the declared placeholder shapes in :data:`PLACEHOLDER_ARM_SHAPES` and
are labelled as such.

Output layout
-------------
``forward`` returns ``n_candidates + 1`` scores per input pair::

    slot 0 .. n_candidates-1   the candidate Events, index-aligned to the
                               task.py catalogue (so slot 0 is Event 0, a
                               genuine certified Event)
    slot n_candidates          BOTTOM, the non-admission option

Bottom is a distinct slot fed by its own head. It is not slot 0, not an
algebraic zero, and not "whatever is left over" after a softmax over Events: it
competes directly with every candidate in the same softmax.

Masking
-------
:func:`mask_scores` reuses the repo convention from
``experiments/tlm_multitoken/policy.py``: a large *finite* penalty
(:data:`MASK_PENALTY` ``= 1e9``) rather than ``-inf``, so a fully masked row
still yields a finite softmax instead of a NaN.

Determinism
-----------
:func:`set_seed` calls ``torch.manual_seed`` then ``np.random.seed``, following
the same repo convention, and is called at the top of :func:`build_scorer`.
Same seed gives bitwise-identical initial parameters; verified and reported.

Numbers in the artifact
-----------------------
Only deterministic quantities are serialised: architecture, hyperparameters,
parameter counts, analytic FLOP counts, symmetry discrepancies (rounded to 12
decimals) and digests. Wall-clock timings are printed to stdout only, never
serialised, because they are not replayable. Weights are never serialised; a
digest of the initial parameters rounded to 6 decimals stands in for them, and
is torch-version dependent by nature — the torch version is recorded alongside.

Fences
------
::

    No per-Event free output column: candidate scoring is relational.
    Bottom is an explicit competing option, not an algebraic zero and not Event index 0.
    Swapping the two input Events is a symmetry of the scorer by construction.
    The same candidate catalogue is scored in every arm.
    Arm-specific adapters differ only as forced by input dimensionality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

__all__ = [
    "ArmLike",
    "BottomHead",
    "CandidateScorer",
    "MASK_PENALTY",
    "RelHead",
    "ScorerConfig",
    "SymmetricPair",
    "SyntheticArm",
    "TokenEncoder",
    "audit",
    "build_scorer",
    "count_params",
    "mask_scores",
    "mean_std",
    "set_seed",
    "synthetic_arm",
    "validate_arm",
]

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "scorer.json"

# Pinned like the other Issue 009 modules: a live ``git rev-parse HEAD`` would
# make the artifact non-replayable the moment anything is committed.
BASE_COMMIT = "174925310ca1ff15948b17e336c787525640dc6c"

#: The certified catalogue size. Cross-checked against
#: ``topographo.ssd.fips_adapter.N_EVENTS`` in :func:`catalogue_size_cross_check`
#: when topographo is importable; kept as a literal so this torch-only module
#: stays importable without it.
CATALOGUE_SIZE = 84

#: A large finite penalty rather than -inf, so a fully masked row still yields a
#: finite softmax instead of a NaN. Same value and rationale as
#: ``experiments/tlm_multitoken/policy.py``.
MASK_PENALTY = 1e9

FENCES = (
    "No per-Event free output column: candidate scoring is relational.",
    (
        "Bottom is an explicit competing option, not an algebraic zero and not "
        "Event index 0."
    ),
    "Swapping the two input Events is a symmetry of the scorer by construction.",
    "The same candidate catalogue is scored in every arm.",
    "Arm-specific adapters differ only as forced by input dimensionality.",
)

ARM_NAMES = ("A_native", "B_sfp", "C_scrambled", "D_relabeled", "E_opaque")

#: Declared placeholder token shapes, used for the concrete ledger rows and for
#: the self-tests. ``tokens_per_event`` follows the coordinator's contract
#: exactly (1 for A and E, 2 for B, C, D). ``token_dim`` is a PLACEHOLDER: the
#: real widths are owned by ``arms.py``, which is being written in parallel and
#: is deliberately not imported here. The ledger is reported both as concrete
#: rows and as a closed form in ``token_dim``, so substituting the real widths
#: changes only the adapter row and nothing else.
PLACEHOLDER_ARM_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("A_native", 16, 1),
    ("B_sfp", 12, 2),
    ("C_scrambled", 12, 2),
    ("D_relabeled", 12, 2),
    ("E_opaque", 16, 1),
)

#: Stand-in catalogue sizes used to demonstrate that the parameter count does not
#: depend on how many candidate Events exist.
CATALOGUE_SIZE_PROBES = (40, CATALOGUE_SIZE, 120)

#: Random-vector pairs drawn for the direct ``SymmetricPair`` commutativity test.
SYMMETRY_RANDOM_PAIRS = 4096

#: Seed for every construction in the self-test driver.
AUDIT_SEED = 0

#: Discrepancies are rounded before serialisation so replay is byte-identical.
#: They are expected to be exactly 0.0; the rounding is belt and braces.
DISCREPANCY_DECIMALS = 12

#: Parameter digests hash values rounded to this many decimals.
PARAM_DIGEST_DECIMALS = 6


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def digest(obj: object) -> str:
    """Canonical sha256 of a JSON-serializable object."""

    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def set_seed(seed: int) -> None:
    """Repo convention (``tlm_multitoken/policy.py``): seed torch, then numpy."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def count_params(model: nn.Module) -> int:
    """Trainable parameter count. Buffers are excluded by construction."""

    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def mean_std(values: Sequence[float]) -> dict:
    """Population std (``np.std`` ddof=0), matching the repo convention."""

    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(v) for v in array],
        "n": int(array.size),
    }


# ---------------------------------------------------------------------------
# the arms interface: structural protocol plus a deterministic stand-in
# ---------------------------------------------------------------------------

@runtime_checkable
class ArmLike(Protocol):
    """The coordinator's frozen ``arms.Arm`` contract, as a structural protocol.

    ``arms.py`` is authored in parallel and is NOT imported. Anything providing
    these attributes is accepted, which is what lets this module be exercised
    today against :class:`SyntheticArm` and used unchanged against the real arms.
    """

    name: str
    tokens: tuple[tuple[tuple[float, ...], ...], ...]
    token_dim: int
    tokens_per_event: int
    exposes: tuple[str, ...]
    forbids: tuple[str, ...]
    metadata: dict

    def sha256(self) -> str:
        ...


@dataclass(frozen=True)
class SyntheticArm:
    """A deterministic stand-in arm. Field-for-field compatible with ``arms.Arm``.

    It carries no OT semantics whatsoever and is never a science result: its only
    job is to give the structural tests here a concrete, reproducible tensor of
    the right shape.
    """

    name: str
    tokens: tuple[tuple[tuple[float, ...], ...], ...]
    token_dim: int
    tokens_per_event: int
    exposes: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def sha256(self) -> str:
        return digest(
            [
                self.name,
                self.token_dim,
                self.tokens_per_event,
                [[list(token) for token in event] for event in self.tokens],
            ]
        )


def synthetic_arm(
    name: str,
    token_dim: int,
    tokens_per_event: int,
    *,
    n_events: int = CATALOGUE_SIZE,
    seed: int = 0,
) -> SyntheticArm:
    """Build a stand-in arm from an explicitly seeded generator.

    Values are rounded to 6 decimals so the arm's own digest is stable and the
    tokens round-trip through JSON without drift.
    """

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_events, tokens_per_event, token_dim))
    tokens = tuple(
        tuple(tuple(round(float(v), 6) for v in token) for token in event)
        for event in raw
    )
    return SyntheticArm(
        name=name,
        tokens=tokens,
        token_dim=token_dim,
        tokens_per_event=tokens_per_event,
        exposes=("synthetic:nothing",),
        forbids=("synthetic:everything",),
        metadata={
            "synthetic": True,
            "seed": seed,
            "purpose": (
                "structural stand-in for the real arms.py Arm; carries no OT "
                "semantics and is never a science result"
            ),
        },
    )


def paired_token_arm(
    name: str,
    token_dim: int,
    *,
    n_events: int = CATALOGUE_SIZE,
    seed: int = 0,
) -> SyntheticArm:
    """A 2-token stand-in whose two tokens are *different* vectors.

    Used for the pooling test: mean pooling must give the same ``u_e`` under a
    swap of the two token slots, which is only a real test when the two tokens
    actually differ.
    """

    arm = synthetic_arm(name, token_dim, 2, n_events=n_events, seed=seed)
    if any(event[0] == event[1] for event in arm.tokens):
        raise AssertionError("token-swap test would be vacuous: some tokens coincide")
    return arm


def validate_arm(arm: ArmLike, *, expect_events: int | None = None) -> int:
    """Check an arm against the frozen contract. Returns the catalogue size."""

    if not isinstance(arm.name, str) or not arm.name:
        raise ValueError("arm.name must be a non-empty string")
    n_events = len(arm.tokens)
    if n_events == 0:
        raise ValueError(f"arm {arm.name!r} carries no tokens")
    if expect_events is not None and n_events != expect_events:
        raise ValueError(
            f"arm {arm.name!r} carries {n_events} entries, expected {expect_events}"
        )
    if arm.tokens_per_event < 1:
        raise ValueError(f"arm {arm.name!r} declares tokens_per_event < 1")
    for index, event in enumerate(arm.tokens):
        if len(event) != arm.tokens_per_event:
            raise ValueError(
                f"arm {arm.name!r} entry {index} holds {len(event)} tokens, "
                f"not the declared {arm.tokens_per_event}"
            )
        for token in event:
            if len(token) != arm.token_dim:
                raise ValueError(
                    f"arm {arm.name!r} entry {index} holds a token of width "
                    f"{len(token)}, not the declared {arm.token_dim}"
                )
    return n_events


def tokens_tensor(arm: ArmLike) -> torch.Tensor:
    """``(n_events, tokens_per_event, token_dim)`` float32 view of an arm."""

    validate_arm(arm)
    return torch.tensor(
        [[list(token) for token in event] for event in arm.tokens],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScorerConfig:
    """Frozen architecture hyperparameters, shared by every arm.

    Only :attr:`common_width` interacts with an arm at all, and only as the
    *output* of the arm-specific adapter. Everything after that projection is
    identical across arms by construction: the same class, the same widths, the
    same parameter shapes.
    """

    #: Output width of the arm-specific adapter. The one common latent width that
    #: every representation is projected into before any shared machinery runs.
    common_width: int = 32
    #: Hidden width of the shared per-token trunk.
    trunk_hidden: int = 64
    #: Width of the per-Event code ``u_e``.
    latent: int = 32
    #: Width of the symmetric pair code.
    pair_width: int = 64
    #: Inner width of the ``Rel`` bilinear form.
    rel_width: int = 32
    #: Hidden width of the ``Bottom`` head.
    bottom_hidden: int = 32
    #: Symmetric pooling over an Event's tokens. Both options are order-invariant.
    pooling: str = "mean"

    def __post_init__(self) -> None:
        if self.pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be mean or sum, got {self.pooling!r}")
        widths = {
            "common_width": self.common_width,
            "trunk_hidden": self.trunk_hidden,
            "latent": self.latent,
            "pair_width": self.pair_width,
            "rel_width": self.rel_width,
            "bottom_hidden": self.bottom_hidden,
        }
        for name, value in sorted(widths.items()):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
            if value == CATALOGUE_SIZE:
                # Not a correctness bug, but it would defeat the mechanical
                # "no dimension equals the catalogue size" check below, so it is
                # refused outright rather than allowed to blunt the audit.
                raise ValueError(
                    f"{name} == {CATALOGUE_SIZE} would collide with the catalogue "
                    "size and blunt the no-per-Event-column check; pick another"
                )

    def as_dict(self) -> dict:
        return {
            "common_width": self.common_width,
            "trunk_hidden": self.trunk_hidden,
            "latent": self.latent,
            "pair_width": self.pair_width,
            "rel_width": self.rel_width,
            "bottom_hidden": self.bottom_hidden,
            "pooling": self.pooling,
            "pair_primitives": list(SymmetricPair.PRIMITIVES),
            "activation": "ReLU",
            "mask_penalty": MASK_PENALTY,
            "mask_convention": (
                "large finite penalty via masked_fill(~legal, -MASK_PENALTY), not "
                "-inf, so a fully masked row yields a finite softmax not a NaN"
            ),
            "seed_convention": "torch.manual_seed(seed) then np.random.seed(seed)",
        }


# ---------------------------------------------------------------------------
# the architecture
# ---------------------------------------------------------------------------

class TokenEncoder(nn.Module):
    """``rep(e) -> u_e``: arm-specific adapter, shared trunk, symmetric pooling.

    The adapter is the *only* arm-specific module and it differs only in its
    input width, which is forced by the arm's token dimensionality. The trunk is
    applied to every token through a single flattened matmul, so there is exactly
    one set of weights for all token slots — no per-slot parameters exist and
    none could be added without changing this class.
    """

    def __init__(self, token_dim: int, config: ScorerConfig) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.pooling = config.pooling
        self.adapter = nn.Linear(self.token_dim, config.common_width)
        self.trunk = nn.Sequential(
            nn.Linear(config.common_width, config.trunk_hidden),
            nn.ReLU(),
            nn.Linear(config.trunk_hidden, config.latent),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(N, T, token_dim) -> (N, latent)``."""

        if tokens.dim() != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected (N, T, {self.token_dim}) tokens, got {tuple(tokens.shape)}"
            )
        n_events, n_tokens, _ = tokens.shape
        flat = tokens.reshape(n_events * n_tokens, self.token_dim)
        coded = self.trunk(self.adapter(flat)).reshape(n_events, n_tokens, -1)
        if self.pooling == "sum":
            return coded.sum(dim=1)
        return coded.mean(dim=1)


class SymmetricPair(nn.Module):
    """``(u_a, u_b) -> pair``, symmetric in ``a, b`` by construction.

    The only order-sensitive step that could exist is the combination of the two
    codes, and that step uses commutative primitives exclusively:

    ``sum``            ``u_a + u_b``   — float addition is commutative bitwise
    ``product``        ``u_a * u_b``   — float multiplication likewise
    ``abs_difference`` ``|u_a - u_b|`` — ``u_a - u_b`` is the exact negation of
                                         ``u_b - u_a``, so ``abs`` erases order

    None of these relies on float *associativity*, which does not hold. The MLP
    that follows therefore receives a bitwise-identical tensor under the swap, so
    the whole scorer is swap-symmetric to the bit, not to a tolerance.
    """

    PRIMITIVES = ("sum", "product", "abs_difference")

    def __init__(self, latent: int, config: ScorerConfig) -> None:
        super().__init__()
        self.latent = int(latent)
        self.out_features = config.pair_width
        self.net = nn.Sequential(
            nn.Linear(len(self.PRIMITIVES) * self.latent, config.pair_width),
            nn.ReLU(),
            nn.Linear(config.pair_width, config.pair_width),
        )

    @staticmethod
    def primitives(u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
        """The commutative combination, before any learned map touches it."""

        return torch.cat([u_a + u_b, u_a * u_b, (u_a - u_b).abs()], dim=-1)

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
        return self.net(self.primitives(u_a, u_b))


class RelHead(nn.Module):
    """``Rel(pair, u_c)``: a bilinear form between the pair code and a candidate.

    Candidate-facing parameters are the single shared ``key`` map from the latent
    code to the bilinear inner width. There is no per-candidate row anywhere, so
    the head scores an arbitrary number of candidates with a fixed parameter
    count, and a candidate never seen as a target is still scored through exactly
    the same machinery as one seen often.
    """

    def __init__(self, pair_dim: int, config: ScorerConfig) -> None:
        super().__init__()
        self.query = nn.Linear(pair_dim, config.rel_width)
        self.key = nn.Linear(config.latent, config.rel_width, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, pair: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """``(B, pair_dim), (N, latent) -> (B, N)``."""

        query = self.query(pair)
        key = self.key(candidates)
        return query @ key.transpose(0, 1) * self.log_scale.exp() + self.offset


class BottomHead(nn.Module):
    """``Bottom(pair)``: the single non-admission score.

    Its own head, its own output slot, competing in the same softmax as every
    candidate. It is not an algebraic zero, not a threshold on the candidate
    scores, and not Event index 0.
    """

    def __init__(self, pair_dim: int, config: ScorerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, config.bottom_hidden),
            nn.ReLU(),
            nn.Linear(config.bottom_hidden, 1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """``(B, pair_dim) -> (B, 1)``."""

        return self.net(pair)


class CandidateScorer(nn.Module):
    """The whole scorer: one encoder, one symmetric pair code, ``Rel`` + ``Bottom``.

    The arm's token table is held as a **buffer**, not a parameter: the candidate
    representations are stipulated by the arm and are never trained, which is
    what stops the candidate side from quietly becoming an 84-way free embedding.
    """

    def __init__(self, arm: ArmLike, config: ScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ScorerConfig()
        self.n_candidates = validate_arm(arm)
        self.arm_name = arm.name
        self.token_dim = int(arm.token_dim)
        self.tokens_per_event = int(arm.tokens_per_event)
        self.register_buffer("catalogue_tokens", tokens_tensor(arm), persistent=False)
        self.encoder = TokenEncoder(self.token_dim, self.config)
        self.pair = SymmetricPair(self.config.latent, self.config)
        self.rel = RelHead(self.pair.out_features, self.config)
        self.bottom = BottomHead(self.pair.out_features, self.config)

    # --- layout ----------------------------------------------------------

    @property
    def bottom_slot(self) -> int:
        """The BOTTOM output slot: one past the last candidate, never slot 0."""

        return self.n_candidates

    @property
    def n_slots(self) -> int:
        return self.n_candidates + 1

    def slot_layout(self) -> dict:
        return {
            "n_slots": self.n_slots,
            "candidate_slots": [0, self.n_candidates - 1],
            "bottom_slot": self.bottom_slot,
            "candidate_alignment": (
                "slot i is candidate Event i in the task.py catalogue order"
            ),
            "bottom_is_distinct_slot": self.bottom_slot >= self.n_candidates,
            "bottom_is_not_event_zero": self.bottom_slot != 0,
        }

    # --- forward ---------------------------------------------------------

    def encode_catalogue(self) -> torch.Tensor:
        """``(n_candidates, latent)``: the shared code for every candidate."""

        return self.encoder(self.catalogue_tokens)

    def forward(
        self,
        a_index: torch.Tensor,
        b_index: torch.Tensor,
        *,
        legal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B,), (B,) -> (B, n_candidates + 1)`` scores.

        ``legal`` is an optional boolean mask over the ``n_candidates + 1`` slots,
        applied with the finite-penalty convention.
        """

        codes = self.encode_catalogue()
        pair = self.pair(codes[a_index], codes[b_index])
        scores = torch.cat([self.rel(pair, codes), self.bottom(pair)], dim=-1)
        if scores.shape[-1] != self.n_slots:
            raise AssertionError("output layout is corrupt")
        if legal is not None:
            scores = mask_scores(scores, legal)
        return scores


def mask_scores(scores: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Apply the repo's finite-penalty legality mask.

    ``-MASK_PENALTY`` rather than ``-inf`` so a fully illegal row still produces a
    finite softmax instead of a NaN — the exact rationale recorded in
    ``experiments/tlm_multitoken/policy.py``.
    """

    if legal.dtype is not torch.bool:
        raise TypeError("legal mask must be a boolean tensor")
    if legal.shape != scores.shape:
        raise ValueError(
            f"mask shape {tuple(legal.shape)} does not match scores "
            f"{tuple(scores.shape)}"
        )
    return scores.masked_fill(~legal, -MASK_PENALTY)


def build_scorer(
    arm: ArmLike,
    config: ScorerConfig | None = None,
    *,
    seed: int = 0,
) -> CandidateScorer:
    """The one construction entry point. Seeds first, then builds."""

    set_seed(seed)
    return CandidateScorer(arm, config or ScorerConfig())


# ---------------------------------------------------------------------------
# parameter and cost accounting
# ---------------------------------------------------------------------------

def _linear_flops(layer: nn.Linear, n_items: int) -> int:
    """2 FLOPs per multiply-accumulate, plus one add per bias element."""

    flops = 2 * layer.in_features * layer.out_features * n_items
    if layer.bias is not None:
        flops += layer.out_features * n_items
    return int(flops)


def module_param_counts(model: CandidateScorer) -> dict[str, int]:
    """Trainable parameter count per top-level block, plus the total."""

    counts = {
        "adapter": count_params(model.encoder.adapter),
        "shared_trunk": count_params(model.encoder.trunk),
        "pair": count_params(model.pair),
        "rel": count_params(model.rel),
        "bottom": count_params(model.bottom),
    }
    counts["encoder_total"] = counts["adapter"] + counts["shared_trunk"]
    counts["total"] = count_params(model)
    accounted = (
        counts["adapter"]
        + counts["shared_trunk"]
        + counts["pair"]
        + counts["rel"]
        + counts["bottom"]
    )
    if accounted != counts["total"]:
        raise AssertionError("per-block parameter counts do not sum to the total")
    return counts


def forward_flops(model: CandidateScorer, *, batch: int = 1) -> dict[str, int]:
    """Analytic forward FLOPs. Deterministic, unlike wall-clock, so serialisable."""

    tokens = model.n_candidates * model.tokens_per_event
    encode = _linear_flops(model.encoder.adapter, tokens)
    encode += _linear_flops(model.encoder.trunk[0], tokens)
    encode += _linear_flops(model.encoder.trunk[2], tokens)
    # pooling: one add per token beyond the first, plus a divide for mean
    pool_items = model.n_candidates * max(model.tokens_per_event - 1, 0)
    encode += pool_items * model.config.latent
    if model.config.pooling == "mean" and model.tokens_per_event > 1:
        encode += model.n_candidates * model.config.latent

    pair = 3 * batch * model.config.latent  # sum, product, abs-difference
    pair += _linear_flops(model.pair.net[0], batch)
    pair += _linear_flops(model.pair.net[2], batch)

    rel = _linear_flops(model.rel.query, batch)
    rel += _linear_flops(model.rel.key, model.n_candidates)
    rel += 2 * batch * model.n_candidates * model.config.rel_width
    rel += 2 * batch * model.n_candidates  # scale multiply and offset add

    bottom = _linear_flops(model.bottom.net[0], batch)
    bottom += _linear_flops(model.bottom.net[2], batch)

    return {
        "batch": int(batch),
        "encode_catalogue": int(encode),
        "pair": int(pair),
        "rel": int(rel),
        "bottom": int(bottom),
        "total": int(encode + pair + rel + bottom),
    }


FLOP_CONVENTION = (
    "2 FLOPs per multiply-accumulate in a matmul, 1 FLOP per bias add, 1 FLOP "
    "per elementwise op; ReLU counted as free. Analytic, not measured, because "
    "wall-clock is not replayable and must not enter the artifact."
)


def param_digest(model: nn.Module) -> str:
    """Digest of the initial parameters, rounded so it is stable within a run."""

    rows = [
        [
            name,
            list(tensor.shape),
            [
                round(float(v), PARAM_DIGEST_DECIMALS)
                for v in tensor.detach().reshape(-1).tolist()
            ],
        ]
        for name, tensor in sorted(model.named_parameters())
    ]
    return digest(rows)


def param_shapes(model: nn.Module) -> list[dict]:
    return [
        {"name": name, "shape": list(tensor.shape), "numel": int(tensor.numel())}
        for name, tensor in sorted(model.named_parameters())
    ]


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------

def catalogue_size_cross_check() -> dict:
    """Soft cross-check of :data:`CATALOGUE_SIZE` against the certified source."""

    try:
        from topographo.ssd.fips_adapter import N_EVENTS
    except ImportError:
        return {
            "checked": False,
            "declared": CATALOGUE_SIZE,
            "statement": (
                "topographo not importable in this environment; CATALOGUE_SIZE "
                "used as declared"
            ),
        }
    return {
        "checked": True,
        "declared": CATALOGUE_SIZE,
        "certified": int(N_EVENTS),
        "agrees": int(N_EVENTS) == CATALOGUE_SIZE,
        "statement": (
            "CATALOGUE_SIZE agrees with topographo.ssd.fips_adapter.N_EVENTS"
        ),
    }


def _round_discrepancy(value: float) -> float:
    return round(float(value), DISCREPANCY_DECIMALS)


def symmetry_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """Swap symmetry, at the primitive level and end to end over all 85 slots."""

    generator = torch.Generator().manual_seed(seed)
    latent = config.latent
    left = torch.randn(SYMMETRY_RANDOM_PAIRS, latent, generator=generator)
    right = torch.randn(SYMMETRY_RANDOM_PAIRS, latent, generator=generator)
    forward = SymmetricPair.primitives(left, right)
    reverse = SymmetricPair.primitives(right, left)
    primitive_random = _round_discrepancy((forward - reverse).abs().max().item())
    primitive_random_bitwise = bool(torch.equal(forward, reverse))

    results: dict[str, dict] = {}
    worst_pair = 0.0
    worst_candidate = 0.0
    worst_bottom = 0.0
    for name, token_dim, tokens_per_event in PLACEHOLDER_ARM_SHAPES:
        arm = synthetic_arm(
            f"synthetic:{name}", token_dim, tokens_per_event, seed=hash_free_seed(name)
        )
        model = build_scorer(arm, config, seed=seed)
        model.eval()
        n = model.n_candidates
        a_index = torch.arange(n).repeat_interleave(n)
        b_index = torch.arange(n).repeat(n)
        with torch.no_grad():
            codes = model.encode_catalogue()
            pair_ab = model.pair(codes[a_index], codes[b_index])
            pair_ba = model.pair(codes[b_index], codes[a_index])
            scores_ab = model(a_index, b_index)
            scores_ba = model(b_index, a_index)
        pair_gap = _round_discrepancy((pair_ab - pair_ba).abs().max().item())
        candidate_gap = _round_discrepancy(
            (scores_ab[:, : model.bottom_slot] - scores_ba[:, : model.bottom_slot])
            .abs()
            .max()
            .item()
        )
        bottom_gap = _round_discrepancy(
            (scores_ab[:, model.bottom_slot] - scores_ba[:, model.bottom_slot])
            .abs()
            .max()
            .item()
        )
        worst_pair = max(worst_pair, pair_gap)
        worst_candidate = max(worst_candidate, candidate_gap)
        worst_bottom = max(worst_bottom, bottom_gap)
        results[name] = {
            "ordered_pairs_checked": int(a_index.numel()),
            "pair_max_abs_discrepancy": pair_gap,
            "pair_bitwise_identical": bool(torch.equal(pair_ab, pair_ba)),
            "candidate_scores_max_abs_discrepancy": candidate_gap,
            "bottom_score_max_abs_discrepancy": bottom_gap,
            "scores_bitwise_identical": bool(torch.equal(scores_ab, scores_ba)),
        }

    token_swap = token_pooling_audit(config, seed=seed)

    return {
        "primitives": {
            "names": list(SymmetricPair.PRIMITIVES),
            "random_pairs_checked": SYMMETRY_RANDOM_PAIRS,
            "random_max_abs_discrepancy": primitive_random,
            "random_bitwise_identical": primitive_random_bitwise,
            "justification": (
                "IEEE-754 addition and multiplication are commutative bitwise, and "
                "u_a - u_b is the exact negation of u_b - u_a so abs erases order; "
                "float associativity is never relied on. Exact bitwise equality is "
                "therefore expected and no tolerance is claimed."
            ),
        },
        "per_arm_end_to_end": results,
        "token_slot_pooling": token_swap,
        "max_abs_discrepancy_overall": max(
            primitive_random, worst_pair, worst_candidate, worst_bottom
        ),
        "all_exactly_symmetric": (
            primitive_random_bitwise
            and all(row["scores_bitwise_identical"] for row in results.values())
            and token_swap["bitwise_identical"]
        ),
    }


def token_pooling_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """Pooling over an Event's tokens is order-invariant, so Arm B is endpoint-symmetric."""

    arm = paired_token_arm("synthetic:pooling", 12, seed=11)
    swapped = SyntheticArm(
        name=arm.name + ":swapped",
        tokens=tuple((event[1], event[0]) for event in arm.tokens),
        token_dim=arm.token_dim,
        tokens_per_event=arm.tokens_per_event,
    )
    model = build_scorer(arm, config, seed=seed)
    model.eval()
    with torch.no_grad():
        codes = model.encoder(tokens_tensor(arm))
        codes_swapped = model.encoder(tokens_tensor(swapped))
    return {
        "tokens_per_event": arm.tokens_per_event,
        "pooling": config.pooling,
        "events_checked": len(arm.tokens),
        "max_abs_discrepancy": _round_discrepancy(
            (codes - codes_swapped).abs().max().item()
        ),
        "bitwise_identical": bool(torch.equal(codes, codes_swapped)),
        "statement": (
            "the two token slots share one set of trunk weights and are pooled by "
            "an order-invariant reduction, so an arm presenting an Event as two "
            "endpoint presentations is endpoint-symmetric by construction"
        ),
    }


def hash_free_seed(name: str) -> int:
    """A stable per-name seed. Never ``hash()``, which varies with PYTHONHASHSEED."""

    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1_000_003


def no_per_event_column_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """Assert no parameter tensor carries the catalogue size, and say what that misses."""

    arm = synthetic_arm("synthetic:probe", 16, 1, seed=hash_free_seed("probe"))
    model = build_scorer(arm, config, seed=seed)
    offenders = [
        row for row in param_shapes(model) if model.n_candidates in row["shape"]
    ]
    candidate_facing = sorted(
        name for name, _ in model.rel.named_parameters(prefix="rel")
    )
    input_role = [
        {
            "arm": name,
            "token_dim": token_dim,
            "equals_catalogue_size": token_dim == CATALOGUE_SIZE,
        }
        for name, token_dim, _ in PLACEHOLDER_ARM_SHAPES
    ]
    return {
        "catalogue_size": model.n_candidates,
        "parameter_shapes": param_shapes(model),
        "parameters_with_catalogue_dimension": offenders,
        "no_parameter_carries_catalogue_dimension": not offenders,
        "candidate_facing_parameters": candidate_facing,
        "candidate_tokens_are_a_buffer": all(
            name != "catalogue_tokens" for name, _ in model.named_parameters()
        ),
        "buffer_requires_grad": bool(model.catalogue_tokens.requires_grad),
        "how_checked": (
            "every tensor in named_parameters() is inspected and rejected if any "
            "of its dimensions equals the catalogue size; ScorerConfig separately "
            "refuses any width equal to the catalogue size so the check cannot be "
            "satisfied vacuously by a coincidence"
        ),
        "limits": (
            "a shape check is necessary, not sufficient. It cannot see a per-Event "
            "identity smuggled in as an INPUT dimension: an arm whose token_dim "
            "equals the catalogue size (a one-hot arm) would give the adapter one "
            "learnable column per Event. That case is reported separately under "
            "input_role_dims and must be refused at the arm level, not here. Nor "
            "can it prove the learner has not memorised a lookup inside the shared "
            "weights; only the held-out folds can speak to that."
        ),
        "input_role_dims": input_role,
        "any_arm_token_dim_equals_catalogue_size": any(
            row["equals_catalogue_size"] for row in input_role
        ),
    }


def catalogue_independence_audit(
    config: ScorerConfig, *, seed: int = AUDIT_SEED
) -> dict:
    """Parameter count is independent of how many candidate Events there are."""

    rows = []
    for size in CATALOGUE_SIZE_PROBES:
        arm = synthetic_arm(
            f"synthetic:size{size}",
            16,
            1,
            n_events=size,
            seed=hash_free_seed(f"size{size}"),
        )
        model = build_scorer(arm, config, seed=seed)
        rows.append(
            {
                "n_candidates": size,
                "n_output_slots": model.n_slots,
                "bottom_slot": model.bottom_slot,
                "total_params": count_params(model),
                "param_shape_digest": digest(param_shapes(model)),
            }
        )
    totals = sorted({row["total_params"] for row in rows})
    shape_digests = sorted({row["param_shape_digest"] for row in rows})
    return {
        "probes": rows,
        "distinct_total_params": totals,
        "distinct_param_shape_digests": shape_digests,
        "param_count_independent_of_catalogue_size": len(totals) == 1,
        "param_shapes_independent_of_catalogue_size": len(shape_digests) == 1,
        "output_slots_track_catalogue_size": [row["n_output_slots"] for row in rows]
        == [size + 1 for size in CATALOGUE_SIZE_PROBES],
        "statement": (
            "the same architecture, at the same parameter count and the same "
            "parameter shapes, scores catalogues of "
            f"{list(CATALOGUE_SIZE_PROBES)} candidates; only the number of output "
            "slots changes, because slots are computed, not parameterised"
        ),
    }


def capacity_ledger(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """Per-arm capacity, and an honest account of where the arms differ."""

    rows = []
    for name, token_dim, tokens_per_event in PLACEHOLDER_ARM_SHAPES:
        arm = synthetic_arm(
            f"synthetic:{name}", token_dim, tokens_per_event, seed=hash_free_seed(name)
        )
        model = build_scorer(arm, config, seed=seed)
        counts = module_param_counts(model)
        started = time.perf_counter()
        with torch.no_grad():
            model(torch.arange(model.n_candidates), torch.zeros(
                model.n_candidates, dtype=torch.long
            ))
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "arm": name,
                "token_dim": token_dim,
                "tokens_per_event": tokens_per_event,
                "token_dim_is_placeholder": True,
                "adapter_params": counts["adapter"],
                "shared_trunk_params": counts["shared_trunk"],
                "pair_params": counts["pair"],
                "rel_params": counts["rel"],
                "bottom_params": counts["bottom"],
                "total_trainable_params": counts["total"],
                "forward_flops": forward_flops(model, batch=1),
                "_wall_sec": elapsed,
            }
        )

    totals = [row["total_trainable_params"] for row in rows]
    adapters = [row["adapter_params"] for row in rows]
    shared = {
        row["arm"]: (
            row["shared_trunk_params"],
            row["pair_params"],
            row["rel_params"],
            row["bottom_params"],
        )
        for row in rows
    }
    shared_identical = len(set(shared.values())) == 1
    forced = [
        {
            "arm": row["arm"],
            "adapter_params": row["adapter_params"],
            "closed_form": (
                f"{row['token_dim']} * {config.common_width} + {config.common_width}"
            ),
            "matches_closed_form": row["adapter_params"]
            == row["token_dim"] * config.common_width + config.common_width,
        }
        for row in rows
    ]
    spread = max(totals) - min(totals)
    adapter_spread = max(adapters) - min(adapters)
    serialisable = [
        {key: value for key, value in row.items() if key != "_wall_sec"}
        for row in rows
    ]
    return {
        "rows": serialisable,
        "wall_clock_excluded": (
            "forward wall-clock was measured and printed but is NOT serialised: it "
            "is not replayable. Analytic FLOPs carry the cost claim instead."
        ),
        "total_params": mean_std(totals),
        "total_params_spread": int(spread),
        "adapter_params_spread": int(adapter_spread),
        "spread_fully_explained_by_adapter": spread == adapter_spread,
        "shared_stack_identical_across_arms": shared_identical,
        "adapter_closed_form": forced,
        "accounting": (
            "Every arm shares one identical stack after projection into the common "
            f"latent width ({config.common_width}): trunk, SymmetricPair, Rel and "
            "Bottom have byte-identical parameter shapes and counts in all five "
            "arms. The ONLY difference is the adapter's first weight matrix, whose "
            "size is token_dim * common_width + common_width. So the whole "
            f"{spread}-parameter spread is forced by input dimensionality and none "
            "of it is a capacity advantage in the shared relational machinery."
        ),
        "no_padding_declaration": (
            "No padding or dimensional equalisation is applied. Token widths are "
            "passed to the adapter as the arms declare them; the spread is reported "
            "rather than hidden. Equalising by padding would misstate how much "
            "information each arm's tokens carry, which is precisely the variable "
            "under test."
        ),
        "placeholder_note": (
            "token_dim values are PLACEHOLDERS pending arms.py, which is authored "
            "in parallel and deliberately not imported. The closed form above is "
            "exact, so substituting the real widths changes only the adapter row "
            "and the reported spread, never the shared stack or any structural "
            "conclusion."
        ),
    }


def determinism_audit(config: ScorerConfig) -> dict:
    """Same seed gives bitwise-identical initial parameters; a different seed does not."""

    arm = synthetic_arm("synthetic:determinism", 16, 1, seed=hash_free_seed("determinism"))
    first = param_digest(build_scorer(arm, config, seed=AUDIT_SEED))
    second = param_digest(build_scorer(arm, config, seed=AUDIT_SEED))
    other = param_digest(build_scorer(arm, config, seed=AUDIT_SEED + 1))
    tokens_first = tokens_tensor(arm)
    tokens_second = tokens_tensor(synthetic_arm(
        "synthetic:determinism", 16, 1, seed=hash_free_seed("determinism")
    ))
    return {
        "seed": AUDIT_SEED,
        "param_digest": first,
        "same_seed_reproducible": first == second,
        "different_seed_differs": first != other,
        "stand_in_arm_reproducible": bool(torch.equal(tokens_first, tokens_second)),
        "digest_rounding_decimals": PARAM_DIGEST_DECIMALS,
        "statement": (
            "torch.manual_seed(seed) then np.random.seed(seed) at the top of "
            "build_scorer; two constructions at the same seed agree to the "
            f"{PARAM_DIGEST_DECIMALS}-decimal digest, a different seed does not. "
            "The digest is torch-version dependent by nature, so the torch version "
            "is recorded in provenance."
        ),
    }


def masking_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """The finite-penalty mask keeps a fully masked row finite instead of NaN."""

    arm = synthetic_arm("synthetic:mask", 16, 1, seed=hash_free_seed("mask"))
    model = build_scorer(arm, config, seed=seed)
    model.eval()
    a_index = torch.tensor([0, 1, 2])
    b_index = torch.tensor([3, 4, 5])
    legal = torch.ones((3, model.n_slots), dtype=torch.bool)
    legal[1] = False                    # a fully masked row
    legal[2, : model.bottom_slot] = False   # only BOTTOM legal
    with torch.no_grad():
        masked = model(a_index, b_index, legal=legal)
        probabilities = torch.softmax(masked, dim=-1)
    return {
        "mask_penalty": MASK_PENALTY,
        "uses_minus_inf": False,
        "fully_masked_row_is_finite": bool(torch.isfinite(masked[1]).all()),
        "fully_masked_softmax_has_no_nan": bool(
            not torch.isnan(probabilities[1]).any()
        ),
        "fully_masked_softmax_sums_to_one": _round_discrepancy(
            abs(float(probabilities[1].sum()) - 1.0)
        ),
        "bottom_only_row_picks_bottom": int(masked[2].argmax()) == model.bottom_slot,
        "rationale": (
            "matches experiments/tlm_multitoken/policy.py: masked_fill with a large "
            "finite -MASK_PENALTY rather than -inf, specifically so a row with no "
            "legal slot yields a finite softmax instead of a NaN that would poison "
            "the whole backward pass"
        ),
    }


def bottom_slot_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """BOTTOM is a distinct competing slot with its own head."""

    arm = synthetic_arm("synthetic:bottom", 16, 1, seed=hash_free_seed("bottom"))
    model = build_scorer(arm, config, seed=seed)
    model.eval()
    a_index = torch.arange(model.n_candidates)
    b_index = torch.zeros(model.n_candidates, dtype=torch.long)
    with torch.no_grad():
        scores = model(a_index, b_index)
        codes = model.encode_catalogue()
        pair = model.pair(codes[a_index], codes[b_index])
        direct_bottom = model.bottom(pair).squeeze(-1)
    layout = model.slot_layout()
    bottom_column = scores[:, model.bottom_slot]
    return {
        "layout": layout,
        "bottom_slot": model.bottom_slot,
        "bottom_slot_is_not_zero": model.bottom_slot != 0,
        "bottom_slot_outside_candidate_range": model.bottom_slot
        >= model.n_candidates,
        "bottom_column_comes_from_bottom_head": bool(
            torch.equal(bottom_column, direct_bottom)
        ),
        "bottom_is_not_algebraically_zero": bool((bottom_column != 0).any()),
        "bottom_head_parameters": sorted(
            name for name, _ in model.bottom.named_parameters(prefix="bottom")
        ),
        "statement": (
            "BOTTOM occupies slot n_candidates, is produced by its own head from "
            "the pair code, takes nonzero values, and competes in the same softmax "
            "as all candidates. Event index 0 keeps slot 0 as a genuine candidate."
        ),
    }


def same_catalogue_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """Every arm scores the same candidate catalogue through the same machinery."""

    rows = []
    for name, token_dim, tokens_per_event in PLACEHOLDER_ARM_SHAPES:
        arm = synthetic_arm(
            f"synthetic:{name}", token_dim, tokens_per_event, seed=hash_free_seed(name)
        )
        model = build_scorer(arm, config, seed=seed)
        with torch.no_grad():
            scores = model(torch.tensor([0]), torch.tensor([1]))
        rows.append(
            {
                "arm": name,
                "n_candidates": model.n_candidates,
                "n_output_slots": int(scores.shape[-1]),
                "bottom_slot": model.bottom_slot,
                "shared_stack_shape_digest": digest(
                    [
                        param_shapes(model.encoder.trunk),
                        param_shapes(model.pair),
                        param_shapes(model.rel),
                        param_shapes(model.bottom),
                    ]
                ),
            }
        )
    return {
        "rows": rows,
        "all_score_the_same_catalogue": len({row["n_candidates"] for row in rows}) == 1
        and {row["n_candidates"] for row in rows} == {CATALOGUE_SIZE},
        "all_same_output_width": len({row["n_output_slots"] for row in rows}) == 1,
        "all_share_identical_stack_shapes": len(
            {row["shared_stack_shape_digest"] for row in rows}
        )
        == 1,
        "statement": (
            "all five arms emit 84 candidate slots plus BOTTOM, through modules "
            "with identical parameter shapes downstream of the adapter"
        ),
    }


def gradient_audit(config: ScorerConfig, *, seed: int = AUDIT_SEED) -> dict:
    """The whole scorer is differentiable end to end and every parameter receives signal."""

    arm = synthetic_arm("synthetic:grad", 16, 1, seed=hash_free_seed("grad"))
    model = build_scorer(arm, config, seed=seed)
    a_index = torch.arange(8)
    b_index = torch.arange(8, 16)
    scores = model(a_index, b_index)
    target = torch.arange(8)
    loss = torch.nn.functional.cross_entropy(scores, target)
    loss.backward()
    without = sorted(
        name for name, p in model.named_parameters() if p.grad is None
    )
    zero = sorted(
        name
        for name, p in model.named_parameters()
        if p.grad is not None and bool((p.grad == 0).all())
    )
    return {
        "loss_is_finite": bool(np.isfinite(float(loss.detach()))),
        "parameters_without_gradient": without,
        "parameters_with_all_zero_gradient": zero,
        "every_parameter_receives_gradient": not without and not zero,
        "note": (
            "a smoke check that the architecture trains at all; the training loop "
            "itself is owned by a later task and is deliberately absent here"
        ),
    }


def provenance() -> dict:
    try:
        topographo_version = importlib.metadata.version("topographo")
    except importlib.metadata.PackageNotFoundError:
        topographo_version = None
    return {
        "base_commit": BASE_COMMIT,
        "base_commit_note": (
            "pinned, not read from live HEAD, so the artifact stays replayable"
        ),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "topographo_version": topographo_version,
        "torch_scope_note": (
            "torch is used only inside experiment scripts; pyproject.toml keeps "
            "numpy as the sole package dependency and torch>=2.0 is declared in "
            "experiments/tlm_fixed_head/requirements.txt"
        ),
        "arms_coupling": (
            "arms.py is NOT imported. This module validates any object against the "
            "ArmLike protocol and self-tests against SyntheticArm stand-ins."
        ),
        "upstream_modules_not_modified": [
            "experiments/sfp_representation/conformance.py",
            "experiments/sfp_representation/folds.py",
            "experiments/sfp_representation/groups.py",
            "experiments/sfp_representation/sfp.py",
            "experiments/sfp_representation/task.py",
        ],
        "param_digest_caveat": (
            "the recorded parameter digest depends on the torch RNG and therefore "
            "on the torch version above; every other serialised number is "
            "torch-version independent"
        ),
    }


def architecture_summary(config: ScorerConfig) -> dict:
    arm = synthetic_arm("synthetic:summary", 16, 1, seed=hash_free_seed("summary"))
    model = build_scorer(arm, config, seed=AUDIT_SEED)
    counts = module_param_counts(model)
    return {
        "schematic": [
            "u_a = Enc(rep(a)); u_b = Enc(rep(b)); u_c = Enc(rep(c))",
            "pair = SymmetricPair(u_a, u_b)",
            "score(c | a,b) = Rel(pair, u_c)",
            "score(bottom | a,b) = Bottom(pair)",
        ],
        "encoder_pipeline": [
            "tokens (tokens_per_event x token_dim)",
            f"adapter Linear(token_dim -> {config.common_width})  [arm-specific "
            "ONLY in its input width]",
            f"shared trunk Linear({config.common_width} -> {config.trunk_hidden}) "
            f"-> ReLU -> Linear({config.trunk_hidden} -> {config.latent})",
            f"{config.pooling} pooling over the tokens_per_event tokens",
            f"u_e in R^{config.latent}",
        ],
        "pair_form": (
            "concat(u_a + u_b, u_a * u_b, |u_a - u_b|) -> "
            f"Linear -> ReLU -> Linear -> R^{config.pair_width}"
        ),
        "rel_form": (
            "bilinear: <query(pair), key(u_c)> * exp(log_scale) + offset; the only "
            "candidate-facing parameters are the shared key map"
        ),
        "bottom_form": (
            f"Linear({config.pair_width} -> {config.bottom_hidden}) -> ReLU -> "
            "Linear(-> 1)"
        ),
        "slot_layout": model.slot_layout(),
        "param_counts_reference_arm": counts,
        "param_shapes_reference_arm": param_shapes(model),
        "flop_convention": FLOP_CONVENTION,
        "owns": [
            "the shared candidate-scorer architecture",
            "the mechanical proof of its structural properties",
            "the finite-penalty masking helper",
        ],
        "does_not_own": [
            "the training loop",
            "the metrics suite",
            "the deterministic baselines",
            "the sweep driver",
            "the representation arms (arms.py)",
        ],
    }


def audit(config: ScorerConfig | None = None) -> dict:
    """Build the full structural report. Deterministic; no wall-clock reaches it."""

    config = config or ScorerConfig()
    symmetry = symmetry_audit(config)
    structural = no_per_event_column_audit(config)
    independence = catalogue_independence_audit(config)
    ledger = capacity_ledger(config)
    determinism = determinism_audit(config)
    masking = masking_audit(config)
    bottom = bottom_slot_audit(config)
    catalogue = same_catalogue_audit(config)
    gradients = gradient_audit(config)
    cross_check = catalogue_size_cross_check()

    laws = {
        "swap_symmetry_exact": symmetry["all_exactly_symmetric"],
        "swap_symmetry_zero_discrepancy": symmetry["max_abs_discrepancy_overall"]
        == 0.0,
        "token_pooling_order_invariant": symmetry["token_slot_pooling"][
            "bitwise_identical"
        ],
        "no_parameter_carries_catalogue_dimension": structural[
            "no_parameter_carries_catalogue_dimension"
        ],
        "candidate_tokens_are_a_buffer": structural["candidate_tokens_are_a_buffer"],
        "no_arm_token_dim_equals_catalogue_size": not structural[
            "any_arm_token_dim_equals_catalogue_size"
        ],
        "param_count_independent_of_catalogue_size": independence[
            "param_count_independent_of_catalogue_size"
        ],
        "param_shapes_independent_of_catalogue_size": independence[
            "param_shapes_independent_of_catalogue_size"
        ],
        "capacity_spread_fully_explained_by_adapter": ledger[
            "spread_fully_explained_by_adapter"
        ],
        "shared_stack_identical_across_arms": ledger[
            "shared_stack_identical_across_arms"
        ],
        "same_seed_reproducible": determinism["same_seed_reproducible"],
        "different_seed_differs": determinism["different_seed_differs"],
        "mask_keeps_fully_masked_row_finite": masking["fully_masked_row_is_finite"]
        and masking["fully_masked_softmax_has_no_nan"],
        "bottom_is_a_distinct_slot": bottom["bottom_slot_is_not_zero"]
        and bottom["bottom_slot_outside_candidate_range"]
        and bottom["bottom_column_comes_from_bottom_head"],
        "bottom_is_not_algebraically_zero": bottom["bottom_is_not_algebraically_zero"],
        "same_catalogue_scored_in_every_arm": catalogue[
            "all_score_the_same_catalogue"
        ]
        and catalogue["all_same_output_width"]
        and catalogue["all_share_identical_stack_shapes"],
        "differentiable_end_to_end": gradients["every_parameter_receives_gradient"],
        "catalogue_size_agrees_with_certified_source": (
            cross_check.get("agrees", True)
        ),
    }
    broken = sorted(name for name, ok in laws.items() if not ok)

    payload = {
        "task": "Issue 009 Task 7 — the shared candidate scorer",
        "module": "experiments/sfp_representation/scorer.py",
        "fences": list(FENCES),
        "architecture": architecture_summary(config),
        "config": config.as_dict(),
        "arm_interface": {
            "protocol": "ArmLike (structural / duck-typed)",
            "fields": [
                "name",
                "tokens",
                "token_dim",
                "tokens_per_event",
                "exposes",
                "forbids",
                "metadata",
                "sha256()",
            ],
            "arm_names": list(ARM_NAMES),
            "design_choice": (
                "arms.py is authored in parallel and is NOT imported. This module "
                "codes against the coordinator's frozen contract via a structural "
                "protocol and self-tests against deterministic SyntheticArm "
                "stand-ins, so it is testable today and accepts the real arms "
                "unchanged."
            ),
            "placeholder_shapes": [
                {
                    "arm": name,
                    "token_dim_placeholder": token_dim,
                    "tokens_per_event": tokens_per_event,
                }
                for name, token_dim, tokens_per_event in PLACEHOLDER_ARM_SHAPES
            ],
        },
        "swap_symmetry": symmetry,
        "no_per_event_output_identity": structural,
        "catalogue_size_independence": independence,
        "capacity_ledger": ledger,
        "determinism": determinism,
        "masking": masking,
        "bottom_option": bottom,
        "same_candidate_catalogue": catalogue,
        "differentiability": gradients,
        "catalogue_size_cross_check": cross_check,
        "provenance": provenance(),
        "serialisation_policy": (
            "weights are never serialised; only architecture, hyperparameters, "
            "parameter counts, analytic FLOPs, symmetry discrepancies rounded to "
            f"{DISCREPANCY_DECIMALS} decimals, and digests. Wall-clock is printed "
            "but never stored, so replay is byte-identical."
        ),
        "structural_laws": laws,
        "verdict": {
            "agrees": not broken,
            "broken_laws": broken,
            "statement": (
                "shared candidate scorer with construction-level swap symmetry and "
                "no per-Event output identity"
                if not broken
                else "structural laws violated: " + ", ".join(broken)
            ),
        },
    }
    payload["digests"] = {
        "architecture": digest(payload["architecture"]),
        "config": digest(payload["config"]),
        "capacity_ledger": digest(payload["capacity_ledger"]),
        "structural_laws": digest(payload["structural_laws"]),
    }
    payload["digests"]["manifest"] = digest(
        {k: v for k, v in payload.items() if k != "digests"}
    )
    return payload


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _report(payload: dict) -> None:
    architecture = payload["architecture"]
    print("architecture:", flush=True)
    for line in architecture["encoder_pipeline"]:
        print(f"  {line}", flush=True)
    print(f"  pair: {architecture['pair_form']}", flush=True)
    print(f"  Rel:  {architecture['rel_form']}", flush=True)
    print(f"  Bot:  {architecture['bottom_form']}", flush=True)
    layout = architecture["slot_layout"]
    print(
        f"  slots: {layout['n_slots']} = candidates "
        f"{layout['candidate_slots'][0]}..{layout['candidate_slots'][1]} + BOTTOM at "
        f"{layout['bottom_slot']}",
        flush=True,
    )

    symmetry = payload["swap_symmetry"]
    print(
        "swap symmetry: primitives max|delta|="
        f"{symmetry['primitives']['random_max_abs_discrepancy']} over "
        f"{symmetry['primitives']['random_pairs_checked']} random pairs "
        f"(bitwise={symmetry['primitives']['random_bitwise_identical']}); "
        f"overall max|delta|={symmetry['max_abs_discrepancy_overall']}",
        flush=True,
    )
    for name, row in sorted(symmetry["per_arm_end_to_end"].items()):
        print(
            f"  {name}: {row['ordered_pairs_checked']} ordered pairs, "
            f"pair={row['pair_max_abs_discrepancy']}, "
            f"candidates={row['candidate_scores_max_abs_discrepancy']}, "
            f"bottom={row['bottom_score_max_abs_discrepancy']}, "
            f"bitwise={row['scores_bitwise_identical']}",
            flush=True,
        )
    pooling = symmetry["token_slot_pooling"]
    print(
        f"  token-slot swap ({pooling['tokens_per_event']} tokens, "
        f"{pooling['pooling']} pooling): max|delta|="
        f"{pooling['max_abs_discrepancy']}, bitwise={pooling['bitwise_identical']}",
        flush=True,
    )

    structural = payload["no_per_event_output_identity"]
    print(
        "no per-Event column: parameters carrying dim "
        f"{structural['catalogue_size']}="
        f"{len(structural['parameters_with_catalogue_dimension'])}, "
        f"tokens are a buffer={structural['candidate_tokens_are_a_buffer']}, "
        "any arm token_dim == 84="
        f"{structural['any_arm_token_dim_equals_catalogue_size']}",
        flush=True,
    )

    independence = payload["catalogue_size_independence"]
    print(
        "catalogue-size independence: totals="
        f"{independence['distinct_total_params']} across "
        f"{[row['n_candidates'] for row in independence['probes']]} candidates, "
        f"slots={[row['n_output_slots'] for row in independence['probes']]}",
        flush=True,
    )

    ledger = payload["capacity_ledger"]
    print("capacity ledger (placeholder token dims):", flush=True)
    for row in ledger["rows"]:
        print(
            f"  {row['arm']:<12} token_dim={row['token_dim']:<3} "
            f"tokens={row['tokens_per_event']} adapter={row['adapter_params']:<6} "
            f"trunk={row['shared_trunk_params']} pair={row['pair_params']} "
            f"rel={row['rel_params']} bottom={row['bottom_params']} "
            f"total={row['total_trainable_params']} "
            f"flops={row['forward_flops']['total']}",
            flush=True,
        )
    print(
        f"  total-param spread={ledger['total_params_spread']} "
        f"(adapter spread={ledger['adapter_params_spread']}, fully explained="
        f"{ledger['spread_fully_explained_by_adapter']}); shared stack identical="
        f"{ledger['shared_stack_identical_across_arms']}",
        flush=True,
    )

    determinism = payload["determinism"]
    print(
        f"determinism: same seed reproducible={determinism['same_seed_reproducible']}, "
        f"different seed differs={determinism['different_seed_differs']}, "
        f"param digest={determinism['param_digest'][:16]}...",
        flush=True,
    )
    masking = payload["masking"]
    print(
        f"masking: penalty={masking['mask_penalty']}, fully-masked row finite="
        f"{masking['fully_masked_row_is_finite']}, no NaN="
        f"{masking['fully_masked_softmax_has_no_nan']}",
        flush=True,
    )
    print(f"manifest sha256: {payload['digests']['manifest']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    result = audit()
    text = render(result)
    _report(result)
    print(f"audit wall_sec (not serialised): {time.perf_counter() - started:.2f}",
          flush=True)
    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived manifest is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches scorer.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

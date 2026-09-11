"""Typed structured output heads for the 009.05 consequence-address ladder.

The whole point of `009.05` is that the INPUT pathway must not change while the
OUTPUT object does. So this module reuses `scorer.py`'s encoder and symmetric
pair code *verbatim* -- the same classes, the same widths, the same seeds -- and
replaces only what sits on top of the pair code:

    009.02   RelHead over 84 candidate Events  +  BottomHead        85 slots
    Rung 1   one 3-way local port head                                3 slots
    Rung 2   BOTTOM gate + typed S / FFF / PP / pp heads              17 slots

Because `scorer.TokenEncoder` and `scorer.SymmetricPair` are imported rather
than reimplemented, the encoder parameters, the pooling, and the bitwise swap
symmetry are the same objects that produced the `009.02` numbers. Any difference
in result is therefore attributable to the output interface, which is the causal
question `009.03` asked.

Architecture fences, from `009.05` section 5.4:

    no output tensor carries the catalogue dimension  (checked: 84 appears nowhere)
    no learned per-Event output table                 (checked: parameter shapes)
    no decoder that could memorize catalogue identity (checked: heads read only
                                                       the pair code, whose width
                                                       is independent of 84)
    no executable XOR as the answer rule              (the pp head is an MLP over
                                                       the pair code; no XOR, and
                                                       no field arithmetic, exists
                                                       anywhere in this module)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from ladder_task import FIELD_CARDINALITIES, PP_CLASSES, STRUCTURED_SLOTS
from scorer import (
    ScorerConfig,
    SymmetricPair,
    TokenEncoder,
    count_params,
    param_shapes,
    set_seed,
    synthetic_arm,
    tokens_tensor,
    validate_arm,
)
from torch import nn

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_heads.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

CATALOGUE_SIZE = 84

#: Stand-in catalogue sizes used to show output capacity does not track the
#: catalogue at all. Unlike ``009.02``, whose output slot count DID track the
#: catalogue (85 slots for 84 Events), the structured heads are flat in it.
CATALOGUE_SIZE_PROBES = (40, CATALOGUE_SIZE, 120)

#: Token widths of the four science arms, taken from the frozen ``arms.py``.
ARM_TOKEN_SHAPES = (
    ("A_native", 16, 1),
    ("B_sfp", 16, 2),
    ("C_scrambled", 16, 2),
    ("D_relabeled", 16, 2),
)

SWAP_PROBE_PAIRS = 4096
AUDIT_SEED = 0
DISCREPANCY_DECIMALS = 12

#: The two fields the certified relation COPIES from the input pair to the
#: consequence, on all 336 admitted pairs, under every arm's code.
IDENTITY_FIELDS = ("S", "FFF")

#: The two fields the certified relation actually COMPUTES.
COMPUTED_FIELDS = ("PP", "pp")

FENCES = (
    "No output tensor carries the catalogue dimension; 84 appears in no head.",
    "No learned per-Event output table exists.",
    (
        "The heads read only the symmetric pair code, so no decoder can index a "
        "catalogue entry."
    ),
    (
        "No XOR and no field arithmetic appears anywhere: the pp head is an MLP "
        "over the pair code."
    ),
    (
        "Swapping the two input Events is a symmetry by construction, inherited "
        "bitwise from scorer.SymmetricPair."
    ),
    "BOTTOM is its own typed gate, not an algebraic zero and not Event index 0.",
    (
        "The candidate token table remains a BUFFER, never a parameter: input "
        "codes are stipulated by the arm and never trained."
    ),
)

INPUT_PATHWAY_IDENTITY = (
    "scorer.TokenEncoder and scorer.SymmetricPair are IMPORTED, not reimplemented, "
    "and are constructed with the same ScorerConfig. The encoder still encodes the "
    "whole 84-token buffer in one pass before indexing the pair, exactly as "
    "scorer.CandidateScorer.forward does, so the input pathway is the same "
    "computation that produced the 009.02 numbers. Only the output object differs."
)


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


def hash_free_seed(name: str) -> int:
    """Digest-derived seed. Never ``hash()``, which is salted per process."""

    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 1_000_003


def _round(value: float) -> float:
    return round(float(value), DISCREPANCY_DECIMALS)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderConfig:
    """Rung widths. The shared trunk is delegated to the frozen ``ScorerConfig``.

    Fixed before any run and applied identically to every arm. There is no
    search, no per-arm adjustment and no early stopping anywhere in this ladder.
    """

    #: Hidden width of every typed field head. One value for all fields, so no
    #: field is given more capacity than another.
    field_hidden: int = 32

    #: The single repair ``009.05`` section 9 permits at Rung 2, OFF by default.
    #:
    #: When ``True`` the identity-carrying fields ``S`` and ``FFF`` are copied from
    #: the input pair instead of being predicted, and only the two fields the
    #: certified relation actually computes -- ``PP`` and ``pp`` -- are learned.
    #: This is the same construction ``009.05`` section 4.2 already licenses at
    #: Rung 1 ("S3 := S, FFF3 := FFF, PP3 := PP"), applied one field pair further
    #: up. It is arm-neutral: the certified relation copies ``S`` and ``FFF`` on
    #: all 336 admitted pairs under every arm's code, including the scramble, so
    #: the copy contributes identically to B, C and D and cannot open a gap
    #: between them.
    copy_identity_fields: bool = False

    def __post_init__(self) -> None:
        if self.field_hidden < 1:
            raise ValueError(f"field_hidden must be positive, got {self.field_hidden}")
        if self.field_hidden == CATALOGUE_SIZE:
            raise ValueError(
                f"field_hidden == {CATALOGUE_SIZE} would collide with the catalogue "
                "size and blunt the no-catalogue-dimension check; pick another"
            )

    @property
    def predicted_fields(self) -> tuple[str, ...]:
        """Which fields the learner is asked to produce."""

        if self.copy_identity_fields:
            return COMPUTED_FIELDS
        return tuple(name for name, _ in FIELD_CARDINALITIES)

    @property
    def copied_fields(self) -> tuple[str, ...]:
        return IDENTITY_FIELDS if self.copy_identity_fields else ()

    def as_dict(self) -> dict:
        return {
            "field_hidden": self.field_hidden,
            "copy_identity_fields": self.copy_identity_fields,
            "predicted_fields": list(self.predicted_fields),
            "copied_fields": list(self.copied_fields),
            "repair_status": (
                "the one predeclared Rung-2 repair, ACTIVE"
                if self.copy_identity_fields
                else "declared default protocol; no repair active"
            ),
        }


# ---------------------------------------------------------------------------
# the heads
# ---------------------------------------------------------------------------

class FieldHead(nn.Module):
    """One typed field head: ``pair -> logits over that field's own values``.

    ``cardinality`` is the field's certified value count -- 2 for ``S``, 7 for
    ``FFF``, 4 for ``PP``, 3 for ``pp`` -- and never the catalogue size. The head
    cannot express an Event identity because it has nowhere to put one.
    """

    def __init__(self, pair_dim: int, cardinality: int, hidden: int) -> None:
        super().__init__()
        if cardinality >= CATALOGUE_SIZE:
            raise ValueError(
                f"field cardinality {cardinality} reaches the catalogue size; that "
                "would be a catalogue decoder wearing a field's name"
            )
        self.cardinality = int(cardinality)
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.cardinality),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """``(B, pair_dim) -> (B, cardinality)``."""

        return self.net(pair)


class GateHead(nn.Module):
    """The BOTTOM gate: ``pair -> one admission logit``.

    Its own head and its own typed output, exactly as ``scorer.BottomHead`` is.
    Non-admission is this gate firing, never ``pp=00``, never ``FFF=000`` and
    never Event index 0.
    """

    def __init__(self, pair_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """``(B, pair_dim) -> (B,)`` logit; positive means ADMIT."""

        return self.net(pair).squeeze(-1)


class _PairTrunk(nn.Module):
    """The shared input pathway: frozen encoder + frozen symmetric pair code."""

    def __init__(
        self, arm, config: ScorerConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config or ScorerConfig()
        self.n_candidates = validate_arm(arm)
        self.arm_name = arm.name
        self.token_dim = int(arm.token_dim)
        self.tokens_per_event = int(arm.tokens_per_event)
        # A buffer, never a parameter: the arm stipulates the input codes.
        self.register_buffer("catalogue_tokens", tokens_tensor(arm), persistent=False)
        self.encoder = TokenEncoder(self.token_dim, self.config)
        self.pair = SymmetricPair(self.config.latent, self.config)

    @property
    def pair_dim(self) -> int:
        return self.pair.out_features

    def encode_catalogue(self) -> torch.Tensor:
        """``(n_candidates, latent)``. The same call ``009.02`` made."""

        return self.encoder(self.catalogue_tokens)

    def pair_code(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> torch.Tensor:
        codes = self.encode_catalogue()
        return self.pair(codes[a_index], codes[b_index])


class PortScorer(_PairTrunk):
    """Rung 1: predict only the local forced-third port ``pp3``.

    Three output slots. ``S3``, ``FFF3`` and ``PP3`` are copied by construction
    per `009.05` section 4.2 and are not predicted here at all, which is what
    makes this a localization probe rather than a proposed architecture.
    """

    def __init__(self, arm, config: ScorerConfig | None = None,
                 ladder: LadderConfig | None = None) -> None:
        super().__init__(arm, config)
        self.ladder = ladder or LadderConfig()
        self.pp = FieldHead(self.pair_dim, len(PP_CLASSES), self.ladder.field_hidden)

    @property
    def n_slots(self) -> int:
        return len(PP_CLASSES)

    def forward(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> torch.Tensor:
        """``(B,), (B,) -> (B, 3)`` logits over ``{01, 10, 11}``."""

        return self.pp(self.pair_code(a_index, b_index))


class StructuredScorer(_PairTrunk):
    """Rung 2: predict ``BOTTOM`` or the whole ``(S, FFF, PP, pp)`` address.

    Genuinely factorized: one typed head per field plus one gate, each reading the
    same shared symmetric pair code. There is no joint output tensor over Events
    anywhere, and no head is wide enough to hold one.
    """

    def __init__(self, arm, config: ScorerConfig | None = None,
                 ladder: LadderConfig | None = None) -> None:
        super().__init__(arm, config)
        self.ladder = ladder or LadderConfig()
        self.gate = GateHead(self.pair_dim, self.ladder.field_hidden)
        widths = dict(FIELD_CARDINALITIES)
        self.fields = nn.ModuleDict(
            {
                name: FieldHead(self.pair_dim, widths[name], self.ladder.field_hidden)
                for name in self.ladder.predicted_fields
            }
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.ladder.predicted_fields

    @property
    def n_slots(self) -> int:
        widths = dict(FIELD_CARDINALITIES)
        return 1 + sum(widths[name] for name in self.field_names)

    def slot_layout(self) -> dict:
        widths = dict(FIELD_CARDINALITIES)
        return {
            "gate_slots": 1,
            "field_slots": {name: widths[name] for name in self.field_names},
            "copied_fields": list(self.ladder.copied_fields),
            "total_slots": self.n_slots,
            "total_slots_when_all_fields_learned": STRUCTURED_SLOTS,
            "catalogue_size": self.n_candidates,
            "output_slots_track_catalogue_size": False,
            "bottom_is_a_typed_gate": True,
            "bottom_is_not_event_zero": True,
            "bottom_is_not_an_algebraic_zero": True,
        }

    def forward(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``(B,), (B,) -> {"gate": (B,), "S": (B,2), "FFF": (B,7), ...}``."""

        pair = self.pair_code(a_index, b_index)
        out = {"gate": self.gate(pair)}
        for name in self.field_names:
            out[name] = self.fields[name](pair)
        return out


def build_port_scorer(
    arm, config: ScorerConfig | None = None,
    ladder: LadderConfig | None = None, *, seed: int = 0,
) -> PortScorer:
    """Seed first, then build. Same convention as ``scorer.build_scorer``."""

    set_seed(seed)
    return PortScorer(arm, config or ScorerConfig(), ladder or LadderConfig())


def build_structured_scorer(
    arm, config: ScorerConfig | None = None,
    ladder: LadderConfig | None = None, *, seed: int = 0,
) -> StructuredScorer:
    """Seed first, then build. Same convention as ``scorer.build_scorer``."""

    set_seed(seed)
    return StructuredScorer(arm, config or ScorerConfig(), ladder or LadderConfig())


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def port_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rung-1 loss: plain cross-entropy over the three local ports."""

    return nn.functional.cross_entropy(logits, target)


def pp_set_loss(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Set-target loss for the ``PP`` head: ``-log(sum of valid probabilities)``.

    ``valid`` is a boolean ``(B, 4)`` mask holding the target Event's two
    block-incidence presentations. Both name the same Event, so neither may be
    privileged; maximizing their total probability is the loss that expresses
    that. Computed in log space with ``logsumexp`` so it is stable and needs no
    epsilon.
    """

    if valid.dtype is not torch.bool:
        raise TypeError("valid must be a boolean mask")
    if valid.shape != logits.shape:
        raise ValueError(
            f"mask shape {tuple(valid.shape)} does not match logits "
            f"{tuple(logits.shape)}"
        )
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    masked = log_probs.masked_fill(~valid, float("-inf"))
    return -torch.logsumexp(masked, dim=-1)


def structured_loss(
    out: dict[str, torch.Tensor],
    is_admit: torch.Tensor,
    field_targets: dict[str, torch.Tensor],
    pp_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rung-2 loss: gate BCE over everything, field losses over admitted only.

    Equal unit weight on every term, declared before any run and never tuned. The
    field terms are averaged over the admitted subset because a non-admitted pair
    has no certified address to predict -- inventing one would conflate BOTTOM
    with an algebraic zero, which the fence forbids.
    """

    terms: dict[str, torch.Tensor] = {}
    terms["gate"] = nn.functional.binary_cross_entropy_with_logits(
        out["gate"], is_admit.to(out["gate"].dtype)
    )
    admitted = is_admit.bool()
    n_admit = int(admitted.sum().item())
    if n_admit == 0:
        raise ValueError("a training batch with no admitted pair cannot train fields")
    for name in ("S", "FFF", "pp"):
        if name not in out:
            continue
        terms[name] = nn.functional.cross_entropy(
            out[name][admitted], field_targets[name][admitted]
        )
    if "PP" in out:
        terms["PP"] = pp_set_loss(out["PP"][admitted], pp_valid[admitted]).mean()
    terms["total"] = sum(
        value for key, value in terms.items() if key != "total"
    )
    return terms


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------

def _probe_arm(name: str, token_dim: int, tokens_per_event: int, n_events: int = 84):
    return synthetic_arm(
        f"synthetic:{name}",
        token_dim,
        tokens_per_event,
        n_events=n_events,
        seed=hash_free_seed(name),
    )


def no_catalogue_dimension_audit(ladder: LadderConfig) -> dict:
    """No parameter and no output tensor carries the catalogue dimension.

    Strictly stronger than the ``009.02`` version of this check. There, the output
    layer legitimately had 85 slots for 84 Events and the audit could only inspect
    parameter shapes. Here the OUTPUT WIDTH ITSELF is inspected and is 17
    regardless of catalogue size.
    """

    arm = _probe_arm("probe", 16, 2)
    rows = {}
    offenders = []
    for label, model in (
        ("rung1", build_port_scorer(arm, ladder=ladder, seed=AUDIT_SEED)),
        ("rung2", build_structured_scorer(arm, ladder=ladder, seed=AUDIT_SEED)),
    ):
        shapes = param_shapes(model)
        bad = [row for row in shapes if CATALOGUE_SIZE in row["shape"]]
        offenders.extend({"rung": label, **row} for row in bad)
        rows[label] = {
            "parameter_shapes": shapes,
            "parameters_with_catalogue_dimension": bad,
            "n_output_slots": model.n_slots,
            "output_width_equals_catalogue_size": model.n_slots == CATALOGUE_SIZE,
            "candidate_tokens_are_a_buffer": all(
                key != "catalogue_tokens" for key, _ in model.named_parameters()
            ),
        }
    return {
        "per_rung": rows,
        "offenders": offenders,
        "no_parameter_carries_catalogue_dimension": not offenders,
        "no_output_width_equals_catalogue_size": not any(
            row["output_width_equals_catalogue_size"] for row in rows.values()
        ),
        "output_widths": {label: row["n_output_slots"] for label, row in rows.items()},
        "how_checked": (
            "every tensor in named_parameters() is rejected if any dimension equals "
            "84, the output width is inspected directly, and both LadderConfig and "
            "ScorerConfig refuse any width equal to 84 so the check cannot pass "
            "vacuously"
        ),
        "stronger_than_009_02": (
            "009.02's output layer had 85 slots for 84 Events, so only parameter "
            "shapes could be audited. These heads are 3 and 17 slots wide "
            "regardless of catalogue size, so the absence of a catalogue decoder is "
            "visible in the output width itself."
        ),
        "limits": (
            "a shape check is necessary, not sufficient. It cannot see a per-Event "
            "identity smuggled in as an INPUT dimension -- an arm whose token_dim "
            "equals 84 would give the adapter one learnable column per Event. That "
            "is why Arm E is excluded from this ladder entirely rather than merely "
            "flagged: see arm_e_excluded."
        ),
    }


def catalogue_independence_audit(ladder: LadderConfig) -> dict:
    """Output width and parameter count are flat in the catalogue size."""

    rows = []
    for size in CATALOGUE_SIZE_PROBES:
        arm = _probe_arm(f"size{size}", 16, 2, n_events=size)
        port = build_port_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
        structured = build_structured_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
        rows.append(
            {
                "n_candidates": size,
                "rung1_slots": port.n_slots,
                "rung2_slots": structured.n_slots,
                "rung1_params": count_params(port),
                "rung2_params": count_params(structured),
                "rung2_param_shape_digest": digest(param_shapes(structured)),
            }
        )
    return {
        "probes": rows,
        "distinct_rung1_slot_counts": sorted({row["rung1_slots"] for row in rows}),
        "distinct_rung2_slot_counts": sorted({row["rung2_slots"] for row in rows}),
        "distinct_rung1_params": sorted({row["rung1_params"] for row in rows}),
        "distinct_rung2_params": sorted({row["rung2_params"] for row in rows}),
        "output_width_independent_of_catalogue_size": len(
            {row["rung2_slots"] for row in rows}
        )
        == 1,
        "param_count_independent_of_catalogue_size": len(
            {row["rung2_params"] for row in rows}
        )
        == 1,
        "param_shapes_independent_of_catalogue_size": len(
            {row["rung2_param_shape_digest"] for row in rows}
        )
        == 1,
        "statement": (
            "the structured heads emit 17 slots at catalogue sizes "
            f"{list(CATALOGUE_SIZE_PROBES)} with identical parameter shapes; the "
            "009.02 interface emitted 41, 85 and 121"
        ),
        "contrast_with_009_02": {
            str(size): size + 1 for size in CATALOGUE_SIZE_PROBES
        },
    }


def swap_symmetry_audit(ladder: LadderConfig) -> dict:
    """Swapping the two input Events changes no output bit.

    Measured, not assumed, at both rungs and on every head, over random index
    pairs. ``scorer.SymmetricPair`` combines the two Event codes with commutative
    IEEE-754 primitives only, so exact bitwise equality is the correct
    expectation and any nonzero discrepancy is a real defect.
    """

    arm = _probe_arm("swap", 16, 2)
    generator = np.random.default_rng(hash_free_seed("swap-pairs"))
    a_index = torch.tensor(
        generator.integers(0, CATALOGUE_SIZE, size=SWAP_PROBE_PAIRS), dtype=torch.long
    )
    b_index = torch.tensor(
        generator.integers(0, CATALOGUE_SIZE, size=SWAP_PROBE_PAIRS), dtype=torch.long
    )

    port = build_port_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
    structured = build_structured_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
    with torch.no_grad():
        p1 = port(a_index, b_index)
        p2 = port(b_index, a_index)
        s1 = structured(a_index, b_index)
        s2 = structured(b_index, a_index)

    rung1 = {
        "max_abs_difference": _round(float((p1 - p2).abs().max().item())),
        "bitwise_identical": bool(torch.equal(p1, p2)),
        "argmax_disagreement_rate": _round(
            float((p1.argmax(-1) != p2.argmax(-1)).float().mean().item())
        ),
    }
    rung2 = {}
    for key in s1:
        left, right = s1[key], s2[key]
        rung2[key] = {
            "max_abs_difference": _round(float((left - right).abs().max().item())),
            "bitwise_identical": bool(torch.equal(left, right)),
        }
    return {
        "probe_pairs": SWAP_PROBE_PAIRS,
        "rung1": rung1,
        "rung2": rung2,
        "all_heads_bitwise_swap_invariant": rung1["bitwise_identical"]
        and all(row["bitwise_identical"] for row in rung2.values()),
        "mechanism": (
            "inherited, not re-derived: every head reads scorer.SymmetricPair's "
            "output, and its primitives (sum, product, absolute difference) are "
            "commutative bitwise in IEEE-754"
        ),
    }


def determinism_audit(ladder: LadderConfig) -> dict:
    """The same seed builds bit-identical parameters; a different seed does not."""

    arm = _probe_arm("determinism", 16, 2)
    same_a = build_structured_scorer(arm, ladder=ladder, seed=7)
    same_b = build_structured_scorer(arm, ladder=ladder, seed=7)
    other = build_structured_scorer(arm, ladder=ladder, seed=8)

    def signature(model: nn.Module) -> str:
        return digest(
            [
                [name, [round(float(v), 6) for v in tensor.flatten().tolist()]]
                for name, tensor in sorted(model.state_dict().items())
            ]
        )

    return {
        "seed_7_first": signature(same_a),
        "seed_7_second": signature(same_b),
        "seed_8": signature(other),
        "same_seed_is_bit_identical": signature(same_a) == signature(same_b),
        "different_seed_differs": signature(same_a) != signature(other),
        "seed_convention": "scorer.set_seed: torch.manual_seed then np.random.seed",
    }


def capacity_ledger(ladder: LadderConfig, config: ScorerConfig | None = None) -> dict:
    """Per-arm, per-head parameter ledger at both rungs.

    ``009.05`` section 5.4 requires exact parameters by field/head. Because all
    four science arms declare ``token_dim = 16``, the adapter width is identical
    and the totals match exactly rather than approximately.
    """

    config = config or ScorerConfig()
    rows = []
    for name, token_dim, tokens_per_event in ARM_TOKEN_SHAPES:
        arm = _probe_arm(f"cap-{name}", token_dim, tokens_per_event)
        port = build_port_scorer(arm, config, ladder, seed=0)
        structured = build_structured_scorer(arm, config, ladder, seed=0)
        head_counts = {
            f"field_{field}": count_params(structured.fields[field])
            for field in structured.field_names
        }
        repaired = build_structured_scorer(
            arm, config, replace(ladder, copy_identity_fields=True), seed=0
        )
        rows.append(
            {
                "arm": name,
                "token_dim": token_dim,
                "tokens_per_event": tokens_per_event,
                "adapter_params": count_params(structured.encoder.adapter),
                "shared_trunk_params": count_params(structured.encoder.trunk),
                "pair_params": count_params(structured.pair),
                "gate_params": count_params(structured.gate),
                **head_counts,
                "rung1_pp_head_params": count_params(port.pp),
                "rung1_total_params": count_params(port),
                "rung2_total_params": count_params(structured),
                "rung2_repair_total_params": count_params(repaired),
                "rung2_repair_slots": repaired.n_slots,
            }
        )
    rung1_totals = sorted({row["rung1_total_params"] for row in rows})
    rung2_totals = sorted({row["rung2_total_params"] for row in rows})
    repair_totals = sorted({row["rung2_repair_total_params"] for row in rows})
    return {
        "rows": rows,
        "distinct_rung2_repair_totals": repair_totals,
        "rung2_repair_capacity_spread": repair_totals[-1] - repair_totals[0],
        "repair_capacity_matched": len(repair_totals) == 1,
        "repair_note": (
            "the repaired configuration drops the S and FFF heads, so it is SMALLER "
            "than the primary. Capacity parity is what matters for a B-vs-C "
            "comparison and it holds exactly within each configuration; the two "
            "configurations are never compared to each other as if they were arms."
        ),
        "distinct_rung1_totals": rung1_totals,
        "distinct_rung2_totals": rung2_totals,
        "rung1_capacity_spread": rung1_totals[-1] - rung1_totals[0],
        "rung2_capacity_spread": rung2_totals[-1] - rung2_totals[0],
        "capacity_matched_across_all_science_arms": len(rung1_totals) == 1
        and len(rung2_totals) == 1,
        "field_head_widths": {name: width for name, width in FIELD_CARDINALITIES},
        "statement": (
            "every science arm carries an identical parameter count at both rungs: "
            f"{rung1_totals[0]} at Rung 1 and {rung2_totals[0]} at Rung 2. The "
            "spread is exactly 0, so no B-vs-C comparison can be a capacity "
            "artifact."
        ),
        "arm_e_excluded": (
            "Arm E is absent from this ladder by construction. Its token_dim equals "
            "the catalogue size, so its adapter would hold one learnable column per "
            "Event -- a memorization ceiling that a shape audit cannot catch on the "
            "input side. 009.02 carried it as a labelled diagnostic; 009.05 needs no "
            "ceiling arm, so it is simply not built."
        ),
    }


def pp_set_loss_audit() -> dict:
    """The set-target loss behaves exactly as its convention claims."""

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    losses = pp_set_loss(logits, valid)
    both = pp_set_loss(
        torch.tensor([[5.0, 5.0, -20.0, -20.0]]),
        torch.tensor([[True, True, False, False]]),
    )
    return {
        "losses": [_round(v) for v in losses.tolist()],
        "either_valid_value_scores_the_same": _round(
            abs(losses[0].item() - losses[1].item())
        )
        == 0.0,
        "mass_on_both_valid_values_is_not_penalised": _round(both.item()) < 1e-6,
        "mass_on_an_invalid_value_is_penalised": losses[3].item() > losses[0].item(),
        "convention": (
            "-log(p[q1] + p[q2]); putting all mass on either presentation, or "
            "splitting it between them, is equally optimal, and only mass outside "
            "the pair is penalised"
        ),
    }


def forward_shape_audit(ladder: LadderConfig) -> dict:
    """Declared output shapes, measured on a real forward pass."""

    arm = _probe_arm("shapes", 16, 2)
    a_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    b_index = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    port = build_port_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
    structured = build_structured_scorer(arm, ladder=ladder, seed=AUDIT_SEED)
    repaired = build_structured_scorer(
        arm, ladder=replace(ladder, copy_identity_fields=True), seed=AUDIT_SEED
    )
    with torch.no_grad():
        p = port(a_index, b_index)
        s = structured(a_index, b_index)
        r = repaired(a_index, b_index)
    return {
        "batch": 4,
        "rung1_shape": list(p.shape),
        "rung2_shapes": {key: list(value.shape) for key, value in sorted(s.items())},
        "rung2_repair_shapes": {
            key: list(value.shape) for key, value in sorted(r.items())
        },
        "rung1_shape_agrees": list(p.shape) == [4, len(PP_CLASSES)],
        "rung2_shapes_agree": all(
            list(s[name].shape) == [4, width] for name, width in FIELD_CARDINALITIES
        )
        and list(s["gate"].shape) == [4],
        "rung2_repair_predicts_only_computed_fields": sorted(r) == sorted(
            ("gate",) + COMPUTED_FIELDS
        ),
        "rung2_repair_slots": repaired.n_slots,
        "slot_layout": structured.slot_layout(),
        "repair_slot_layout": repaired.slot_layout(),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(ladder: LadderConfig | None = None) -> dict:
    ladder = ladder or LadderConfig()
    config = ScorerConfig()

    columns = no_catalogue_dimension_audit(ladder)
    independence = catalogue_independence_audit(ladder)
    symmetry = swap_symmetry_audit(ladder)
    determinism = determinism_audit(ladder)
    capacity = capacity_ledger(ladder, config)
    set_loss = pp_set_loss_audit()
    shapes = forward_shape_audit(ladder)

    laws = {
        "no_parameter_carries_catalogue_dimension": columns[
            "no_parameter_carries_catalogue_dimension"
        ],
        "no_output_width_equals_catalogue_size": columns[
            "no_output_width_equals_catalogue_size"
        ],
        "output_width_independent_of_catalogue_size": independence[
            "output_width_independent_of_catalogue_size"
        ],
        "param_count_independent_of_catalogue_size": independence[
            "param_count_independent_of_catalogue_size"
        ],
        "param_shapes_independent_of_catalogue_size": independence[
            "param_shapes_independent_of_catalogue_size"
        ],
        "all_heads_bitwise_swap_invariant": symmetry[
            "all_heads_bitwise_swap_invariant"
        ],
        "same_seed_is_bit_identical": determinism["same_seed_is_bit_identical"],
        "different_seed_differs": determinism["different_seed_differs"],
        "capacity_matched_across_all_science_arms": capacity[
            "capacity_matched_across_all_science_arms"
        ],
        "pp_set_loss_is_presentation_symmetric": set_loss[
            "either_valid_value_scores_the_same"
        ],
        "pp_set_loss_penalises_only_outside_the_pair": set_loss[
            "mass_on_both_valid_values_is_not_penalised"
        ]
        and set_loss["mass_on_an_invalid_value_is_penalised"],
        "rung1_output_shape_agrees": shapes["rung1_shape_agrees"],
        "rung2_output_shapes_agree": shapes["rung2_shapes_agree"],
        "rung2_repair_predicts_only_computed_fields": shapes[
            "rung2_repair_predicts_only_computed_fields"
        ],
        "repair_capacity_matched": capacity["repair_capacity_matched"],
    }

    broken = sorted(name for name, ok in laws.items() if not ok)
    return {
        "module": "ladder_heads",
        "purpose": (
            "typed structured output heads for Rungs 1 and 2, sharing the frozen "
            "009.02 encoder and symmetric pair code so that only the output object "
            "differs"
        ),
        "executes": "009.05",
        "fences": list(FENCES),
        "input_pathway_identity": INPUT_PATHWAY_IDENTITY,
        "config": {"scorer": config.as_dict(), "ladder": ladder.as_dict()},
        "no_catalogue_dimension": columns,
        "catalogue_independence": independence,
        "swap_symmetry": symmetry,
        "determinism": determinism,
        "capacity_ledger": capacity,
        "pp_set_loss": set_loss,
        "forward_shapes": shapes,
        "loss_declaration": {
            "rung1": "cross-entropy over the three local ports",
            "rung2_gate": "binary cross-entropy with logits on the admission gate",
            "rung2_S_FFF_pp": "cross-entropy, averaged over admitted examples only",
            "rung2_PP": "set-target -log(p[q1] + p[q2]) over admitted examples only",
            "weighting": (
                "equal unit weight on all five terms, fixed before any run and "
                "never tuned"
            ),
            "why_fields_are_admitted_only": (
                "a non-admitted pair has no certified address; inventing one would "
                "conflate BOTTOM with an algebraic zero, which the fence forbids"
            ),
        },
        "declared_repair": {
            "name": "copy_identity_fields",
            "rung": 2,
            "default": False,
            "identity_fields": list(IDENTITY_FIELDS),
            "computed_fields": list(COMPUTED_FIELDS),
            "motivation": (
                "S and FFF are the two fields the certified relation COPIES rather "
                "than computes. Under a structural holdout a closed-set softmax over "
                "them can fit training exactly and still emit a training habitat's "
                "identity for an unseen habitat, which is a training pathology about "
                "habitat re-identification rather than about consequence forcing."
            ),
            "what_it_changes": (
                "the S and FFF heads are not built and those fields are taken from "
                "the input pair, exactly the construction 009.05 section 4.2 licenses "
                "at Rung 1. PP and pp remain fully learned, the gate remains fully "
                "learned, and the resolver still resolves a complete four-field "
                "address to a public Event."
            ),
            "what_it_does_not_change": (
                "no hyperparameter, no step count, no learning rate, no arm, no fold "
                "and no seed. It is arm-neutral because the copy is exact on all 336 "
                "admitted pairs under B, C and D alike."
            ),
            "budget": "009.05 section 9 permits one repair per rung; Rung 1 used none",
        },
        "provenance": {
            "base_commit": BASE_COMMIT,
            "reuses": [
                "scorer.TokenEncoder",
                "scorer.SymmetricPair",
                "scorer.ScorerConfig",
                "scorer.set_seed",
                "scorer.tokens_tensor",
                "scorer.validate_arm",
                "scorer.param_shapes",
            ],
            "replaces": "scorer.RelHead + scorer.BottomHead (the 85-slot interface)",
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - the structured heads emit "
                f"{shapes['slot_layout']['total_slots']} slots independently of the "
                "catalogue size, carry no catalogue dimension in any parameter, are "
                "bitwise swap-invariant on every head, and hold identical capacity "
                "across all four science arms"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }


def _report(result: dict) -> None:
    capacity = result["capacity_ledger"]
    print(
        f"output slots: Rung 1 = {result['no_catalogue_dimension']['output_widths']['rung1']}, "
        f"Rung 2 = {result['no_catalogue_dimension']['output_widths']['rung2']} "
        "(009.02 used 85)",
        flush=True,
    )
    print(
        f"capacity: Rung 1 {capacity['distinct_rung1_totals']} params, "
        f"Rung 2 {capacity['distinct_rung2_totals']} params, spread "
        f"{capacity['rung2_capacity_spread']}",
        flush=True,
    )
    print(
        "swap invariance: "
        f"{result['swap_symmetry']['all_heads_bitwise_swap_invariant']} (bitwise)",
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
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches ladder_heads.json", flush=True)
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

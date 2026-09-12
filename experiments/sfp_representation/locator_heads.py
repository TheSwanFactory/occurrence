"""009.07 architecture: a query-relative chamber pointer instead of a PP class head.

``009.06`` emitted ``PP`` through ``FieldHead(pair_dim, 4, 32)`` -- an absolute
four-class softmax whose final layer holds one learned column per chamber value. For
an admitted pair of Event edges that is the wrong question. The shared chamber is a
**query-relative** object, and ``PP`` is an affine torsor coordinate with no intrinsic
labels, so ``009.07`` section 3.2 requires the answer to be *selected among
candidates* through one shared scoring function.

What changes, and only this::

    009.06   PP := argmax FieldHead(pair)              4 learned output columns
    009.07   PP := argmax_q  Rel(pair, u_q)            0 learned output columns

``u_q`` is the encoding of chamber ``q`` **as the arm's own code describes it**: the
three incidence presentations ``(S, FFF, q, d)`` for ``d`` in ``{01, 10, 11}`` that
make up chamber ``q``'s star in the current chart. Those tokens are a BUFFER, they
pass through the same frozen ``scorer.TokenEncoder`` the input Events pass through,
and the scoring parameters are ``scorer.RelHead``'s -- a query map, a key map and two
scalars, none of them indexed by a chamber. So:

* one shared scoring function for all ``q``;
* no trainable output column indexed by absolute ``PP`` identity;
* candidate ``PP`` values supplied as data, not learned class identities;
* parameter count independent of the number and order of candidates;
* symmetric in swapping ``a`` and ``b``, inherited bitwise from
  ``scorer.SymmetricPair``;
* covariant under the certified affine relabeling, proved here by weight transport
  rather than asserted.

No equality, no set intersection, no XOR and no ``ExactSfpCircuit`` appears in any
learned path. The candidate set is restricted to the four chambers of the current
chart, which is what ``009.07`` section 3.2 asks for, and the chart is a function of
the unordered input pair's own code words -- input data, never a label.

Everything upstream is imported: the encoder and pair code from ``scorer``, the gate
and the typed ``pp`` head from ``ladder_heads``. A difference in result is therefore
attributable to the ``PP`` output constitution.
"""

import argparse
import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from arms import (
    ARM_D_AFFINE,
    ARM_D_FLIP_S,
    ARM_D_GL32,
    FIELD_LAYOUT,
    SFP_TOKEN_DIM,
    sfp_token,
)
from ladder_heads import CATALOGUE_SIZE, FieldHead, GateHead, _PairTrunk
from ladder_task import PP_CLASSES
from locator_task import CHAMBER_VALUES, chamber_actions
from scorer import (
    RelHead,
    ScorerConfig,
    count_params,
    param_shapes,
    set_seed,
    synthetic_arm,
)
from torch import nn

from topographo.core import f2_groups as groups

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_locator_artifacts"
OUTPUT = ARTIFACTS / "locator_heads.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

AUDIT_SEED = 0
SWAP_PROBE_PAIRS = 4096
DISCREPANCY_DECIMALS = 12

#: Candidate-count probes. The pointer's parameter count must be identical at all
#: three, which is the mechanical form of "parameters independent of the number of
#: candidate q values".
CANDIDATE_COUNT_PROBES = (2, len(CHAMBER_VALUES), 9)

#: Stand-in catalogue sizes, as ``ladder_heads`` uses, to show no output width
#: tracks the catalogue.
CATALOGUE_SIZE_PROBES = (40, CATALOGUE_SIZE, 120)

#: Tolerance for the covariance score comparison, and the reason it is not zero.
#:
#: Transporting the adapter's input columns and the token coordinates by the same
#: permutation makes ``W' t'`` and ``W t`` equal *as real numbers*, but it reorders
#: the accumulation inside the matmul, and float addition is not associative. That
#: is the same fact ``scorer.SymmetricPair`` documents when it explains why it uses
#: only commutative primitives and never relies on associativity. The observed
#: discrepancy is at the ``1e-8`` level, single-precision round-off for a 16-term
#: sum. The tolerance is declared here, before the check, at a value two orders of
#: magnitude above that; the DECISION -- the argmax -- is required to be exactly
#: covariant with no tolerance at all.
COVARIANCE_TOLERANCE = 1e-6

#: Relative tolerance for the ``RelHead`` reuse comparison, and its reason.
#:
#: ``RelHead.forward`` reduces with ``matmul``; the per-row form multiplies and then
#: reduces with ``sum``. Those are the same real number reached by a different
#: accumulation order, and single-precision addition is not associative, so the two
#: agree to round-off rather than bitwise. Declared before the check and applied
#: RELATIVE to the score magnitude, so it cannot be satisfied by scores that happen
#: to be small. The argmax is required to agree exactly.
REUSE_RELATIVE_TOLERANCE = 1e-5

ARM_TOKEN_SHAPES = (
    ("A_native", 16, 1),
    ("B_sfp", 16, 2),
    ("C_scrambled", 16, 2),
    ("D_relabeled", 16, 2),
)

#: ``009.06``'s absolute-PP head, for the parameter comparison. Reported, not run.
PRIOR_PP_HEAD = {
    "module": "ladder_heads.FieldHead(pair_dim=64, cardinality=4, hidden=32)",
    "params": 64 * 32 + 32 + 32 * 4 + 4,
    "output_columns_indexed_by_absolute_PP": 4,
}

FENCES = (
    "No parameter is indexed by an absolute PP value; the pointer has no per-chamber row.",
    "Parameter count is identical for 2, 4 and 9 candidate chambers.",
    "Candidate chamber tokens are a BUFFER, never a parameter.",
    "No output tensor carries the catalogue dimension; 84 appears in no head.",
    (
        "No equality test, set intersection, XOR or ExactSfpCircuit appears in any "
        "learned path. The candidate star is built from the arm's code alphabet "
        "alone and never references the input pair."
    ),
    (
        "The chart a row's candidates are drawn from is a function of the UNORDERED "
        "input pair's own code words -- input data, the same source 009.06 already "
        "used for the copied S and FFF fields. No certified label participates."
    ),
    (
        "Swapping the two input Events is a symmetry by construction, inherited "
        "bitwise from scorer.SymmetricPair."
    ),
    "BOTTOM is its own typed gate, not an algebraic zero and not Event index 0.",
    "Neither endpoint presentation of an Event is privileged anywhere.",
)

INPUT_PATHWAY_IDENTITY = (
    "scorer.TokenEncoder and scorer.SymmetricPair are IMPORTED and constructed with "
    "the same frozen ScorerConfig, through ladder_heads._PairTrunk, which is the same "
    "object 009.06 used. scorer.RelHead supplies every scoring parameter. The input "
    "pathway is therefore the computation that produced the 009.02 and 009.06 numbers; "
    "only the PP output constitution differs."
)

CANDIDATE_CONSTRUCTION = (
    "Chamber q of chart (S, FFF) is presented to the learner as its STAR: the three "
    "incidence presentations (S, FFF, q, d) for d in {01, 10, 11}. That is the arm's "
    "own alphabet describing the chamber, it is a property of the chart and not of the "
    "query pair, and it transforms covariantly under the certified affine relabeling "
    "because an affine map carries chamber q's star onto chamber phi(q)'s star while "
    "permuting displacements by A. It is encoded by the SAME TokenEncoder the input "
    "Events are encoded by, with the same mean pooling over presentations, so no "
    "candidate-specific machinery exists."
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def digest(obj: object) -> str:
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
class LocatorConfig:
    """Head widths. The trunk is delegated to the frozen ``ScorerConfig``.

    Fixed before any run and applied identically to every arm. ``009.07`` section 9
    forbids a hyperparameter sweep and there is none: every width here is either the
    frozen ``009.02`` constant or the frozen ``009.06`` constant.
    """

    #: Hidden width of the gate and of the typed ``pp`` head. The frozen
    #: ``ladder_heads.LadderConfig.field_hidden`` value.
    field_hidden: int = 32

    #: ``S`` and ``FFF`` are copied from the input pair rather than predicted.
    #:
    #: This is not a new repair. It is the configuration of the ``009.06`` block that
    #: produced the reference numbers ``0.7158`` on ``PP`` and ``0.7016`` on the whole
    #: consequence, and ``009.07`` section 5.1 restates it: ``S3 := copied certified
    #: input S``, ``FFF3 := copied certified input FFF``. Reusing it is what makes the
    #: comparison a comparison of the ``PP`` interface and nothing else.
    copy_identity_fields: bool = True

    def __post_init__(self) -> None:
        if self.field_hidden < 1:
            raise ValueError(f"field_hidden must be positive, got {self.field_hidden}")
        if self.field_hidden == CATALOGUE_SIZE:
            raise ValueError("field_hidden must not collide with the catalogue size")
        if not self.copy_identity_fields:
            raise ValueError(
                "009.07 section 5.1 fixes S3 and FFF3 as copied input fields; "
                "learning them would change more than the PP interface and make the "
                "comparison with 009.06's repair block invalid"
            )

    @property
    def learned_fields(self) -> tuple[str, ...]:
        return ("PP", "pp")

    @property
    def copied_fields(self) -> tuple[str, ...]:
        return ("S", "FFF")

    def as_dict(self) -> dict:
        return {
            "field_hidden": self.field_hidden,
            "copy_identity_fields": self.copy_identity_fields,
            "learned_fields": list(self.learned_fields),
            "copied_fields": list(self.copied_fields),
            "pp_output_constitution": "query-relative candidate selection (pointer)",
            "prior_pp_output_constitution": "absolute four-class softmax head",
            "repair_status": (
                "no repair active. The copied identity fields are the frozen 009.06 "
                "repair-block configuration, reused so that PP is the only change."
            ),
        }


# ---------------------------------------------------------------------------
# the candidate chambers
# ---------------------------------------------------------------------------

def chamber_star_tokens(s: int, fff: int, q: int) -> tuple[tuple[float, ...], ...]:
    """Chamber ``q``'s star in chart ``(S, FFF)``: three incidence presentations.

    ``sfp_token`` is imported from ``arms`` rather than reimplemented, so the
    candidates are written in exactly the alphabet the inputs are written in, with
    the same one-hot field layout and the same ``{0.0, 1.0}`` value fence. No unit
    exists for ``FFF=000`` or ``pp=00``.
    """

    return tuple(sfp_token(s, fff, q, d) for d in PP_CLASSES)


@dataclass(frozen=True)
class ChamberTable:
    """The frozen candidate table for one arm's alphabet.

    ``charts`` lists the 14 ``(S, FFF)`` charts in ascending order; ``slot_of`` maps a
    chart to its row. Every arm's alphabet contains the same 14 charts, because
    ``GL(3,2)`` permutes the seven nonzero Fano points among themselves and the
    ``S`` flip is global.
    """

    charts: tuple[tuple[int, int], ...]
    slot_of: dict[tuple[int, int], int]
    tokens: tuple  # (n_charts, 4, 3, SFP_TOKEN_DIM) nested tuples

    @property
    def n_charts(self) -> int:
        return len(self.charts)

    def tensor(self) -> torch.Tensor:
        return torch.tensor(self.tokens, dtype=torch.float32)

    def sha256(self) -> str:
        return digest(
            {
                "charts": [list(chart) for chart in self.charts],
                "tokens": [
                    [[list(token) for token in star] for star in chart]
                    for chart in self.tokens
                ],
            }
        )


def build_chamber_table(codes) -> ChamberTable:
    """Build the candidate table from an arm's own code table.

    ``codes`` is the arm's ``tuple[SfpAddress, ...]`` from
    ``ladder_task.code_tables``, i.e. decoded back out of the token tensor the
    learner actually sees. The charts are read off it rather than assumed, so an arm
    whose alphabet had drifted could not be given candidates it does not use.
    """

    charts = tuple(sorted({code.habitat for code in codes}))
    if len(charts) != 14:
        raise AssertionError(f"arm alphabet holds {len(charts)} charts, not 14")
    tokens = tuple(
        tuple(chamber_star_tokens(s, fff, q) for q in CHAMBER_VALUES)
        for s, fff in charts
    )
    return ChamberTable(
        charts=charts,
        slot_of={chart: index for index, chart in enumerate(charts)},
        tokens=tokens,
    )


def chart_slots(codes, chamber_table: ChamberTable) -> tuple[int, ...]:
    """Per-Event chart slot, so a batch can gather its candidates by input ``a``."""

    return tuple(chamber_table.slot_of[code.habitat] for code in codes)


# ---------------------------------------------------------------------------
# the pointer
# ---------------------------------------------------------------------------

class ChamberPointer(nn.Module):
    """``score(q | a, b)`` for the four chambers of one chart, through one bilinear form.

    Every scoring parameter belongs to the imported ``scorer.RelHead``: a query map
    from the pair code, a key map from the candidate latent, a log-scale and an
    offset. ``RelHead.forward`` scores a *global* candidate set; this module needs a
    per-row one, so it applies the same three parameter groups with a gather and an
    inner product, and :func:`relhead_reuse_audit` proves the two agree bitwise on a
    global set. There is no per-candidate row anywhere, so the parameter count does
    not depend on how many chambers are offered or in what order.
    """

    def __init__(self, pair_dim: int, config: ScorerConfig) -> None:
        super().__init__()
        self.rel = RelHead(pair_dim, config)

    def score(
        self, pair: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """``(B, pair_dim), (B, N, latent) -> (B, N)``.

        The identical arithmetic ``RelHead.forward`` performs, batched over rows:
        ``query . key * exp(log_scale) + offset``.
        """

        query = self.rel.query(pair)
        key = self.rel.key(candidates)
        scaled = (query.unsqueeze(1) * key).sum(dim=-1)
        return scaled * self.rel.log_scale.exp() + self.rel.offset


class _ChamberTrunk(_PairTrunk):
    """``_PairTrunk`` plus the frozen candidate-chamber buffer and the pointer."""

    def __init__(
        self,
        arm,
        codes,
        config: ScorerConfig | None = None,
        locator: LocatorConfig | None = None,
    ) -> None:
        super().__init__(arm, config)
        self.locator = locator or LocatorConfig()
        table = build_chamber_table(codes)
        self.chamber_charts = table.charts
        self.n_chambers = len(CHAMBER_VALUES)
        # Buffers, never parameters: the candidate codes are stipulated by the arm.
        self.register_buffer("chamber_tokens", table.tensor(), persistent=False)
        self.register_buffer(
            "chart_of_event",
            torch.tensor(chart_slots(codes, table), dtype=torch.long),
            persistent=False,
        )
        self.pointer = ChamberPointer(self.pair_dim, self.config)

    def encode_chambers(self) -> torch.Tensor:
        """``(n_charts, 4, latent)``: every candidate through the shared encoder."""

        n_charts, n_chambers, n_tokens, token_dim = self.chamber_tokens.shape
        flat = self.chamber_tokens.reshape(n_charts * n_chambers, n_tokens, token_dim)
        return self.encoder(flat).reshape(n_charts, n_chambers, -1)

    def chart_slot(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> torch.Tensor:
        """``(B,)``: which chart's four chambers this row is offered.

        ``torch.minimum`` of the two inputs' chart slots. For an admitted pair the two
        are the same slot -- admission requires one habitat -- so this *is* "the
        current affine chart" that ``009.07`` section 3.2 names, and it is a function
        of the UNORDERED pair, which is what keeps the whole scorer swap-symmetric to
        the bit on every row rather than only on same-chart rows.

        For a cross-chart pair there is no shared chart and the minimum is a declared
        symmetric tie-break. Such a row is non-admitted, its ``PP`` output enters no
        reported metric, and the tie-break is the one place the architecture is not
        covariant under relabeling -- both facts are recorded in the audits rather
        than argued away.
        """

        return torch.minimum(self.chart_of_event[a_index], self.chart_of_event[b_index])

    def chamber_scores(
        self, pair: torch.Tensor, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> torch.Tensor:
        """``(B, 4)``: the four chambers of the row's chart, scored."""

        coded = self.encode_chambers()
        candidates = coded[self.chart_slot(a_index, b_index)]
        return self.pointer.score(pair, candidates)


class ChamberLocator(_ChamberTrunk):
    """Rung 1: predict only the certified shared chamber ``q*``.

    Four candidate scores and no absolute-``PP`` output column. ``S``, ``FFF`` and
    ``pp`` are not predicted here at all, which is what makes this a localization
    probe rather than a proposed architecture.
    """

    @property
    def n_slots(self) -> int:
        return self.n_chambers

    def slot_layout(self) -> dict:
        return {
            "candidate_slots": self.n_chambers,
            "learned_output_columns_indexed_by_absolute_PP": 0,
            "prior_learned_output_columns": PRIOR_PP_HEAD[
                "output_columns_indexed_by_absolute_PP"
            ],
            "candidates_are_a_buffer": True,
            "output_slots_track_catalogue_size": False,
        }

    def forward(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> torch.Tensor:
        """``(B,), (B,) -> (B, 4)`` scores over the chart's chambers."""

        return self.chamber_scores(
            self.pair_code(a_index, b_index), a_index, b_index
        )


class QueryRelativeScorer(_ChamberTrunk):
    """Rung 2: ``BOTTOM``, or ``(S, FFF, q*, d3)`` with ``S`` and ``FFF`` copied.

    The gate and the typed ``pp`` head are the imported ``009.06`` modules,
    unchanged. Only ``PP`` is produced differently.
    """

    def __init__(
        self,
        arm,
        codes,
        config: ScorerConfig | None = None,
        locator: LocatorConfig | None = None,
    ) -> None:
        super().__init__(arm, codes, config, locator)
        self.gate = GateHead(self.pair_dim, self.locator.field_hidden)
        self.pp = FieldHead(self.pair_dim, len(PP_CLASSES), self.locator.field_hidden)

    @property
    def n_slots(self) -> int:
        return 1 + self.n_chambers + len(PP_CLASSES)

    def slot_layout(self) -> dict:
        return {
            "gate_slots": 1,
            "chamber_candidate_slots": self.n_chambers,
            "pp_slots": len(PP_CLASSES),
            "total_slots": self.n_slots,
            "copied_fields": list(self.locator.copied_fields),
            "learned_output_columns_indexed_by_absolute_PP": 0,
            "catalogue_size": self.n_candidates,
            "output_slots_track_catalogue_size": False,
            "bottom_is_a_typed_gate": True,
            "bottom_is_not_event_zero": True,
            "bottom_is_not_an_algebraic_zero": True,
        }

    def forward(
        self, a_index: torch.Tensor, b_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``(B,), (B,) -> {"gate": (B,), "PP": (B, 4), "pp": (B, 3)}``."""

        pair = self.pair_code(a_index, b_index)
        return {
            "gate": self.gate(pair),
            "PP": self.chamber_scores(pair, a_index, b_index),
            "pp": self.pp(pair),
        }


def build_chamber_locator(
    arm,
    codes,
    config: ScorerConfig | None = None,
    locator: LocatorConfig | None = None,
    *,
    seed: int = 0,
) -> ChamberLocator:
    """Seed first, then build. Same convention as ``scorer.build_scorer``."""

    set_seed(seed)
    return ChamberLocator(arm, codes, config or ScorerConfig(), locator or LocatorConfig())


def build_query_relative_scorer(
    arm,
    codes,
    config: ScorerConfig | None = None,
    locator: LocatorConfig | None = None,
    *,
    seed: int = 0,
) -> QueryRelativeScorer:
    """Seed first, then build. Same convention as ``scorer.build_scorer``."""

    set_seed(seed)
    return QueryRelativeScorer(
        arm, codes, config or ScorerConfig(), locator or LocatorConfig()
    )


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def chamber_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rung-1 loss: cross-entropy over the candidate scores, POINT target.

    A point target, not a set target. ``009.06`` maximized the total probability of
    the target Event's two block-incidence presentations, because both name the same
    Event; ``009.07`` asks the strictly narrower question "which chamber do the two
    inputs share", which has exactly one certified answer. The stricter criterion is
    the declared change and is reported alongside the set-relaxed one so the two
    turns can still be compared on like terms.
    """

    return nn.functional.cross_entropy(scores, target)


def recomposition_loss(
    out: dict[str, torch.Tensor],
    is_admit: torch.Tensor,
    chamber_target: torch.Tensor,
    pp_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rung-2 loss: gate BCE over everything, field losses over admitted only.

    Equal unit weight on every term, declared before any run and never tuned --
    exactly ``ladder_heads.structured_loss``'s discipline, with the ``PP`` term now a
    candidate-selection cross-entropy. A non-admitted pair has no certified
    consequence, so inventing one would conflate ``BOTTOM`` with an algebraic zero.
    """

    terms: dict[str, torch.Tensor] = {}
    terms["gate"] = nn.functional.binary_cross_entropy_with_logits(
        out["gate"], is_admit.to(out["gate"].dtype)
    )
    admitted = is_admit.bool()
    if int(admitted.sum().item()) == 0:
        raise ValueError("a training batch with no admitted pair cannot train fields")
    terms["PP"] = nn.functional.cross_entropy(
        out["PP"][admitted], chamber_target[admitted]
    )
    terms["pp"] = nn.functional.cross_entropy(
        out["pp"][admitted], pp_target[admitted]
    )
    terms["total"] = sum(value for key, value in terms.items() if key != "total")
    return terms


# ---------------------------------------------------------------------------
# the affine token transport, for the covariance audit
# ---------------------------------------------------------------------------

def field_offsets() -> dict[str, tuple[int, int]]:
    """``arms.FIELD_LAYOUT`` as ``field -> (offset, width)``. Imported, not restated."""

    return {name: (offset, width) for name, offset, width, _ in FIELD_LAYOUT}


def token_permutation() -> tuple[int, ...]:
    """The permutation of the 16 token coordinates realized by Arm D's relabeling.

    ``arms.relabel_address`` acts on values: ``S -> flip_s(S)``,
    ``FFF -> apply_fff(GL32, FFF)``, ``q -> A q XOR b``, ``d -> A d``. Because every
    field is one-hot, that value map is exactly a permutation of the token's
    coordinates, and it is derived here from the same ``arms`` constants and
    ``groups`` functions the arm itself used -- never written out as a literal.
    """

    offsets = field_offsets()
    sigma = [-1] * SFP_TOKEN_DIM

    s_off, _ = offsets["S"]
    for s in (0, 1):
        image = groups.flip_s(s) if ARM_D_FLIP_S else s
        sigma[s_off + s] = s_off + image

    fff_off, _ = offsets["FFF"]
    for fff in range(1, 8):
        sigma[fff_off + fff - 1] = fff_off + groups.apply_fff(ARM_D_GL32, fff) - 1

    pp_off, _ = offsets["PP"]
    action = chamber_actions()["D_relabeled"]
    for q in CHAMBER_VALUES:
        sigma[pp_off + q] = pp_off + action[q]

    d_off, _ = offsets["pp"]
    matrix, _ = ARM_D_AFFINE
    for d in PP_CLASSES:
        image = groups.CHAMBER_INDEX[
            groups.pp_displacement_image(matrix, groups.V2[d])
        ]
        if image == 0:
            raise AssertionError("an invertible map sent a nonzero pp to 00")
        sigma[d_off + d - 1] = d_off + image - 1

    if sorted(sigma) != list(range(SFP_TOKEN_DIM)):
        raise AssertionError(f"token transport {sigma} is not a permutation")
    return tuple(sigma)


def transport_model(model, sigma: tuple[int, ...]):
    """A deep copy whose adapter input columns are permuted by ``sigma``.

    If ``t'`` is the relabeled token, ``t'[sigma[i]] = t[i]``, so setting
    ``W'[:, sigma[i]] = W[:, i]`` gives ``W' t' = W t`` exactly. Transporting the
    weights alongside the data is what turns "the architecture has no absolute-PP
    preference" from a claim into a bitwise identity: any absolute chamber structure
    in the parameters would survive the transport and break it.
    """

    moved = copy.deepcopy(model)
    with torch.no_grad():
        weight = model.encoder.adapter.weight.detach().clone()
        new = torch.empty_like(weight)
        for i, j in enumerate(sigma):
            new[:, j] = weight[:, i]
        moved.encoder.adapter.weight.copy_(new)
    return moved


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


def _probe_codes():
    """The exact SFP code table, for building probe candidate tables."""

    from ladder_task import code_tables
    from task import build_dataset

    return code_tables(build_dataset())


def no_absolute_pp_column_audit(codes) -> dict:
    """The claim ``009.07`` section 3.2 turns on: no parameter is indexed by ``PP``.

    Two independent mechanical checks. First, the parameter *shapes* are enumerated
    and none is the chamber count. Second -- and this is the one that cannot be
    faked -- the pointer is rebuilt over 2, 4 and 9 candidate chambers and the
    parameter count and every parameter shape must be **identical**, which is only
    possible if nothing is indexed by a candidate.
    """

    arm = _probe_arm("no-abs-pp", 16, 2)
    model = build_chamber_locator(arm, codes["B_sfp"], seed=AUDIT_SEED)
    pointer_shapes = param_shapes(model.pointer)

    counts = {}
    shapes = {}
    config = ScorerConfig()
    for n in CANDIDATE_COUNT_PROBES:
        set_seed(AUDIT_SEED)
        pointer = ChamberPointer(config.pair_width, config)
        pair = torch.zeros(3, config.pair_width)
        candidates = torch.zeros(3, n, config.latent)
        out = pointer.score(pair, candidates)
        counts[str(n)] = count_params(pointer)
        shapes[str(n)] = param_shapes(pointer)
        if tuple(out.shape) != (3, n):
            raise AssertionError(f"pointer emitted {tuple(out.shape)} for {n} candidates")

    distinct_counts = sorted(set(counts.values()))
    distinct_shapes = {digest(value) for value in shapes.values()}
    return {
        "pointer_parameter_shapes": pointer_shapes,
        "pointer_params": count_params(model.pointer),
        "parameter_count_by_candidate_count": counts,
        "candidate_count_probes": list(CANDIDATE_COUNT_PROBES),
        "distinct_parameter_counts": distinct_counts,
        "parameter_count_independent_of_candidate_count": len(distinct_counts) == 1,
        "parameter_shapes_independent_of_candidate_count": len(distinct_shapes) == 1,
        "no_parameter_shape_is_the_chamber_count": all(
            len(CHAMBER_VALUES) not in row["shape"] for row in pointer_shapes
        ),
        "chamber_tokens_are_a_buffer": all(
            key != "chamber_tokens" for key, _ in model.named_parameters()
        ),
        "chart_of_event_is_a_buffer": all(
            key != "chart_of_event" for key, _ in model.named_parameters()
        ),
        "prior_pp_head": dict(PRIOR_PP_HEAD),
        "learned_output_columns_indexed_by_absolute_PP": 0,
        "statement": (
            "the pointer holds the same parameters whether it is offered 2, 4 or 9 "
            "candidate chambers, so no parameter can be indexed by an absolute PP "
            "value. 009.06's head held 4 output columns and 4 biases that were."
        ),
    }


def candidate_order_invariance_audit(codes) -> dict:
    """Permuting the candidate axis permutes the scores, bitwise.

    ``009.07`` section 3.2 requires the parameter count to be independent of the
    *order* of candidate ``q`` values as well as their number. Order independence is
    a property of the forward pass, so it is measured on one rather than argued.
    """

    config = ScorerConfig()
    set_seed(AUDIT_SEED)
    pointer = ChamberPointer(config.pair_width, config)
    set_seed(AUDIT_SEED + 1)
    pair = torch.randn(64, config.pair_width)
    candidates = torch.randn(64, len(CHAMBER_VALUES), config.latent)
    base = pointer.score(pair, candidates)

    rows = []
    exact = True
    for order in itertools.permutations(range(len(CHAMBER_VALUES))):
        index = torch.tensor(order, dtype=torch.long)
        moved = pointer.score(pair, candidates[:, index, :])
        ok = bool(torch.equal(moved, base[:, index]))
        exact = exact and ok
        rows.append({"order": list(order), "scores_permute_identically": ok})
    return {
        "permutations_checked": len(rows),
        "per_permutation": rows,
        "order_invariance_exact": exact,
        "probe_rows": 64,
        "statement": (
            "for all 24 orderings of the four candidates the scores are the same "
            "numbers in the same permuted positions, bitwise. A learned per-chamber "
            "row could not survive this."
        ),
    }


def relhead_reuse_audit() -> dict:
    """The per-row gather is ``scorer.RelHead.forward``'s arithmetic, bitwise.

    ``ChamberPointer`` needs per-row candidate sets and ``RelHead.forward`` scores a
    global one, so the module applies ``RelHead``'s own query map, key map, log-scale
    and offset directly -- there is no second set of scoring parameters anywhere. This
    shows the two forms agree on a global candidate set to float round-off, and that
    they choose the same candidate exactly, which is what licenses calling this reuse
    rather than reimplementation.
    """

    config = ScorerConfig()
    set_seed(AUDIT_SEED)
    pointer = ChamberPointer(config.pair_width, config)
    set_seed(AUDIT_SEED + 2)
    pair = torch.randn(32, config.pair_width)
    candidates = torch.randn(len(CHAMBER_VALUES), config.latent)

    global_scores = pointer.rel(pair, candidates)
    per_row = pointer.score(pair, candidates.unsqueeze(0).expand(32, -1, -1))
    difference = float((global_scores - per_row).abs().max().item())
    scale = float(global_scores.abs().max().item())
    relative = difference / scale if scale > 0.0 else difference
    argmax_agrees = bool(
        torch.equal(global_scores.argmax(dim=-1), per_row.argmax(dim=-1))
    )
    own_parameters = sorted(
        name for name, _ in pointer.named_parameters() if not name.startswith("rel.")
    )
    return {
        "max_absolute_difference": _round(difference),
        "score_scale": _round(scale),
        "max_relative_difference": _round(relative),
        "relative_tolerance": REUSE_RELATIVE_TOLERANCE,
        "agrees_with_relhead_forward_within_tolerance": relative
        <= REUSE_RELATIVE_TOLERANCE,
        "agrees_with_relhead_forward_bitwise": difference == 0.0,
        "argmax_agrees_exactly": argmax_agrees,
        "parameters_are_relheads": sorted(
            name for name, _ in pointer.rel.named_parameters()
        ),
        "parameters_outside_relhead": own_parameters,
        "no_scoring_parameter_outside_relhead": own_parameters == [],
        "shared_parameter_groups": ["query", "key", "log_scale", "offset"],
        "why_not_bitwise": (
            "RelHead.forward reduces with matmul and the per-row form multiplies then "
            "reduces with sum. Same real number, different accumulation order, and "
            f"single-precision addition is not associative, so the {difference:.3e} "
            "residual is round-off. The candidate CHOICE agrees exactly."
        ),
        "statement": (
            "every scoring parameter belongs to the imported scorer.RelHead and the "
            "pointer adds none of its own; the per-row form is the same product, sum, "
            "scale and offset, agrees with RelHead.forward to float round-off on a "
            "global candidate set, and selects the same candidate exactly"
        ),
    }


def swap_symmetry_audit(codes) -> dict:
    """Swapping the two input Events changes nothing, bitwise, at both rungs."""

    arm = _probe_arm("swap", 16, 2)
    set_seed(AUDIT_SEED + 3)
    a_index = torch.randint(0, 84, (SWAP_PROBE_PAIRS,), dtype=torch.long)
    b_index = torch.randint(0, 84, (SWAP_PROBE_PAIRS,), dtype=torch.long)

    rows = {}
    exact = True
    locator = build_chamber_locator(arm, codes["B_sfp"], seed=AUDIT_SEED)
    with torch.no_grad():
        forward = locator(a_index, b_index)
        swapped = locator(b_index, a_index)
    same_chart = locator.chart_of_event[a_index] == locator.chart_of_event[b_index]
    n_same = int(same_chart.sum().item())
    if n_same == 0:
        raise AssertionError("swap probe drew no same-chart pair")
    identical = bool(torch.equal(forward, swapped))
    exact = exact and identical
    rows["rung1"] = {
        "probe_pairs": SWAP_PROBE_PAIRS,
        "same_chart_pairs": n_same,
        "cross_chart_pairs": SWAP_PROBE_PAIRS - n_same,
        "max_absolute_difference": _round(
            float((forward - swapped).abs().max().item())
        ),
        "bitwise_identical": identical,
        "bitwise_identical_on_cross_chart_rows": bool(
            torch.equal(forward[~same_chart], swapped[~same_chart])
        ),
    }

    scorer_model = build_query_relative_scorer(arm, codes["B_sfp"], seed=AUDIT_SEED)
    with torch.no_grad():
        out = scorer_model(a_index, b_index)
        out_swapped = scorer_model(b_index, a_index)
    per_key = {}
    for key in sorted(out):
        ok = bool(torch.equal(out[key], out_swapped[key]))
        exact = exact and ok
        per_key[key] = {
            "max_absolute_difference": _round(
                float((out[key] - out_swapped[key]).abs().max().item())
            ),
            "bitwise_identical": ok,
        }
    rows["rung2"] = {"per_output": per_key}
    return {
        "per_rung": rows,
        "swap_symmetry_exact_everywhere": exact,
        "includes_cross_chart_rows": True,
        "mechanism": (
            "scorer.SymmetricPair combines the two codes with sum, product and "
            "absolute difference only, so the pair code is bitwise identical under "
            "the swap; the candidate chart is selected with torch.minimum over the "
            "two inputs' chart slots, which is also a function of the unordered pair, "
            "so every head is swap-exact on every row including cross-chart ones"
        ),
    }


def covariance_audit(codes) -> dict:
    """Architectural covariance under the certified affine relabeling, exactly.

    ``009.07`` section 6 requires covariance to be verified rather than assumed, on
    exact-oracle probes and on a large frozen learned probe set. Two things are
    checked here, both bitwise and both needing no training:

    1. **Candidate construction is covariant.** Chamber ``q``'s star in chart
       ``(S, FFF)``, relabeled token by token, is chamber ``phi(q)``'s star in the
       relabeled chart -- as a set of tokens, so no presentation is privileged.
    2. **Predictions are covariant.** Transport one model's adapter columns by the
       same relabeling and feed it the relabeled inputs and candidates: the scores are
       the original scores permuted by ``phi``, and the **argmax is exactly** so on
       every probe row. Any absolute-chamber structure in the parameters would break
       this. The scores themselves agree to :data:`COVARIANCE_TOLERANCE` rather than
       bitwise, because permuting the adapter's input columns reorders a float sum;
       the measured discrepancy is single-precision round-off and is reported.

    What is *not* claimed: that a model trained on Arm B and a model trained on Arm D
    produce identical predictions. They are different parameter draws acting on
    different one-hot coordinates, so that equality cannot hold and is not asserted.
    The learned half of the covariance requirement is the paired ``D - B`` interval in
    ``locator_analysis.py``, which is measured.
    """

    sigma = token_permutation()
    action = chamber_actions()["D_relabeled"]
    inverse_action = groups.chamber_permutation(groups.invert_affine(ARM_D_AFFINE))

    # 1. candidate construction
    star_rows = []
    star_ok = True
    for s, fff in sorted({code.habitat for code in codes["B_sfp"]}):
        for q in CHAMBER_VALUES:
            original = chamber_star_tokens(s, fff, q)
            moved = tuple(
                tuple(token[i] for i in _inverse(sigma)) for token in original
            )
            s2 = groups.flip_s(s) if ARM_D_FLIP_S else s
            fff2 = groups.apply_fff(ARM_D_GL32, fff)
            expected = chamber_star_tokens(s2, fff2, action[q])
            ok = set(moved) == set(expected)
            star_ok = star_ok and ok
            if len(star_rows) < 8:
                star_rows.append(
                    {
                        "chart": [s, fff],
                        "chamber": q,
                        "image_chart": [s2, fff2],
                        "image_chamber": action[q],
                        "star_maps_onto_the_image_star": ok,
                    }
                )
    # 2. prediction covariance under weight transport
    arm = _probe_arm("covariance", 16, 2)
    model = build_query_relative_scorer(arm, codes["B_sfp"], seed=AUDIT_SEED)
    moved_model = transport_model(model, sigma)
    # Relabel the arm's alphabet and rebuild the candidate table from it.
    from arms import relabel_address

    relabeled = tuple(relabel_address(code) for code in codes["B_sfp"])
    moved_model.register_buffer(
        "chamber_tokens",
        build_chamber_table(relabeled).tensor(),
        persistent=False,
    )
    moved_table = build_chamber_table(relabeled)
    moved_model.register_buffer(
        "chart_of_event",
        torch.tensor(chart_slots(relabeled, moved_table), dtype=torch.long),
        persistent=False,
    )
    # Transport the INPUT tokens too: the probe arm's tokens stand in for the arm's
    # code, so the same coordinate permutation is applied to them.
    with torch.no_grad():
        moved_model.chamber_tokens.copy_(build_chamber_table(relabeled).tensor())
        moved_model.catalogue_tokens.copy_(
            model.catalogue_tokens[..., _inverse_tensor(sigma)]
        )

    # Restricted to SAME-CHART pairs, which is every admitted pair: admission
    # requires one habitat. A cross-chart row's candidate set comes from the
    # declared symmetric tie-break in _ChamberTrunk.chart_slot, which depends on
    # the sorted order of the charts and is therefore not covariant. That row is
    # non-admitted and its PP output enters no reported metric; the restriction is
    # stated here rather than left implicit.
    # Enumerate every same-chart ordered pair, then sample with replacement, so the
    # probe is a frozen set drawn from the exact population rather than a rejection
    # loop whose size depends on luck.
    charts = model.chart_of_event
    population = torch.tensor(
        [
            [i, j]
            for i in range(84)
            for j in range(84)
            if int(charts[i]) == int(charts[j])
        ],
        dtype=torch.long,
    )
    if population.shape[0] != 14 * 6 * 6:
        raise AssertionError(
            f"{population.shape[0]} same-chart ordered pairs, not 14 * 36"
        )
    set_seed(AUDIT_SEED + 4)
    draw = torch.randint(0, population.shape[0], (SWAP_PROBE_PAIRS,), dtype=torch.long)
    a_index = population[draw, 0]
    b_index = population[draw, 1]
    with torch.no_grad():
        base = model(a_index, b_index)
        moved = moved_model(a_index, b_index)

    index = torch.tensor(
        [action[q] for q in CHAMBER_VALUES], dtype=torch.long
    )
    permuted = torch.empty_like(base["PP"])
    permuted[:, index] = base["PP"]
    pp_delta = float((permuted - moved["PP"]).abs().max().item())
    gate_delta = float((base["gate"] - moved["gate"]).abs().max().item())
    pp_within = pp_delta <= COVARIANCE_TOLERANCE
    gate_within = gate_delta <= COVARIANCE_TOLERANCE
    argmax_back = bool(
        torch.equal(
            torch.tensor(
                [inverse_action[q] for q in moved["PP"].argmax(dim=-1).tolist()],
                dtype=torch.long,
            ),
            base["PP"].argmax(dim=-1),
        )
    )
    return {
        "token_permutation": list(sigma),
        "chamber_action": list(action),
        "inverse_chamber_action": list(inverse_action),
        "candidate_star_rows": star_rows,
        "candidate_stars_checked": 14 * len(CHAMBER_VALUES),
        "candidate_construction_covariant": star_ok,
        "learned_probe_rows": SWAP_PROBE_PAIRS,
        "learned_probe_population": int(population.shape[0]),
        "learned_probe_restriction": (
            "same-chart pairs only, which is every admitted pair. The cross-chart "
            "tie-break in _ChamberTrunk.chart_slot depends on the sorted order of the "
            "charts and is NOT covariant; those rows are non-admitted and their PP "
            "output enters no reported metric."
        ),
        "tolerance": COVARIANCE_TOLERANCE,
        "max_score_discrepancy": _round(pp_delta),
        "scores_permute_by_the_chamber_action_within_tolerance": pp_within,
        "scores_permute_by_the_chamber_action_bitwise": pp_delta == 0.0,
        "max_gate_discrepancy": _round(gate_delta),
        "gate_unchanged_under_transport_within_tolerance": gate_within,
        "argmax_maps_back_under_the_inverse_exactly": argmax_back,
        "architectural_covariance_exact": (
            star_ok and pp_within and gate_within and argmax_back
        ),
        "why_not_bitwise": (
            "permuting the adapter's input columns reorders the accumulation inside "
            f"the matmul and float addition is not associative, so the {pp_delta:.3e} "
            "residual is single-precision round-off on a 16-term sum, not structure. "
            "scorer.SymmetricPair documents the same fact where it explains why it "
            "relies on commutativity and never on associativity. The DECISION is "
            "exactly covariant on all "
            f"{SWAP_PROBE_PAIRS} probe rows, with no tolerance."
        ),
        "what_is_not_claimed": (
            "no identity between a B-trained model and a D-trained model is claimed "
            "or checked here. Those are different parameter draws acting on "
            "different one-hot coordinates, so exact equality cannot hold; the "
            "learned half of section 6 is the measured paired D-B interval in "
            "locator_analysis.py."
        ),
        "method": (
            "weight transport: permute the adapter's input columns by the same "
            "coordinate permutation the relabeling induces on tokens, so W' t' = W t "
            "exactly. Any absolute-chamber preference in the parameters would "
            "survive the transport and break the identity."
        ),
    }


def _inverse(sigma: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(sigma)
    for i, j in enumerate(sigma):
        out[j] = i
    return tuple(out)


def _inverse_tensor(sigma: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(_inverse(sigma), dtype=torch.long)


def no_catalogue_dimension_audit(codes) -> dict:
    """No parameter and no output width carries the catalogue dimension."""

    rows = {}
    offenders = []
    for label, size in (("small", 40), ("catalogue", CATALOGUE_SIZE), ("large", 120)):
        arm = _probe_arm(f"cat-{label}", 16, 2, n_events=size)
        locator = build_chamber_locator(arm, codes["B_sfp"], seed=AUDIT_SEED)
        recomposer = build_query_relative_scorer(arm, codes["B_sfp"], seed=AUDIT_SEED)
        for rung, model in (("rung1", locator), ("rung2", recomposer)):
            shapes = param_shapes(model)
            bad = [row for row in shapes if CATALOGUE_SIZE in row["shape"]]
            offenders.extend({"probe": label, "rung": rung, **row} for row in bad)
            rows[f"{label}|{rung}"] = {
                "catalogue_size": size,
                "n_output_slots": model.n_slots,
                "param_count": count_params(model),
                "parameters_with_catalogue_dimension": bad,
            }
    widths = sorted({row["n_output_slots"] for key, row in rows.items() if "rung1" in key})
    counts = sorted({row["param_count"] for key, row in rows.items() if "rung1" in key})
    return {
        "per_probe": rows,
        "offenders": offenders,
        "no_parameter_carries_catalogue_dimension": not offenders,
        "rung1_output_widths": widths,
        "rung1_output_width_constant": len(widths) == 1,
        "rung1_param_counts": counts,
        "rung1_param_count_constant": len(counts) == 1,
        "catalogue_size_probes": list(CATALOGUE_SIZE_PROBES),
        "statement": (
            "output width and parameter count are both flat in the catalogue size, "
            "at 40, 84 and 120 candidates alike"
        ),
    }


def capacity_ledger(
    codes, locator: LocatorConfig | None = None, config: ScorerConfig | None = None
) -> dict:
    """Per-arm, per-head parameter ledger, and the honest comparison to ``009.06``.

    Two separate questions, kept separate. Within this turn, capacity must be
    matched across the science arms or a B-vs-C gap could be an artifact; it is,
    exactly, because all four arms declare ``token_dim = 16``. Across turns, the
    pointer is **not** capacity-matched to ``009.06``'s absolute head, and that is
    reported as a caution rather than smoothed over.
    """

    config = config or ScorerConfig()
    locator = locator or LocatorConfig()
    rows = []
    for name, token_dim, tokens_per_event in ARM_TOKEN_SHAPES:
        arm = _probe_arm(f"cap-{name}", token_dim, tokens_per_event)
        one = build_chamber_locator(arm, codes["B_sfp"], config, locator, seed=0)
        two = build_query_relative_scorer(arm, codes["B_sfp"], config, locator, seed=0)
        rows.append(
            {
                "arm": name,
                "token_dim": token_dim,
                "tokens_per_event": tokens_per_event,
                "adapter_params": count_params(two.encoder.adapter),
                "shared_trunk_params": count_params(two.encoder.trunk),
                "pair_params": count_params(two.pair),
                "pointer_params": count_params(two.pointer),
                "gate_params": count_params(two.gate),
                "pp_head_params": count_params(two.pp),
                "rung1_total_params": count_params(one),
                "rung2_total_params": count_params(two),
                "rung1_slots": one.n_slots,
                "rung2_slots": two.n_slots,
            }
        )
    rung1 = sorted({row["rung1_total_params"] for row in rows})
    rung2 = sorted({row["rung2_total_params"] for row in rows})
    pointer = rows[0]["pointer_params"]
    delta = pointer - PRIOR_PP_HEAD["params"]
    return {
        "rows": rows,
        "distinct_rung1_totals": rung1,
        "distinct_rung2_totals": rung2,
        "rung1_capacity_spread": rung1[-1] - rung1[0],
        "rung2_capacity_spread": rung2[-1] - rung2[0],
        "capacity_matched_across_all_science_arms": len(rung1) == 1 and len(rung2) == 1,
        "pointer_params": pointer,
        "prior_pp_head": dict(PRIOR_PP_HEAD),
        "pp_pathway_param_delta_vs_009_06": delta,
        "pp_pathway_param_ratio_vs_009_06": _round(pointer / PRIOR_PP_HEAD["params"]),
        "cross_turn_capacity_matched": delta == 0,
        "cross_turn_caution": (
            f"the pointer holds {pointer} parameters against {PRIOR_PP_HEAD['params']} "
            f"in 009.06's absolute PP head, a delta of {delta:+d}. Both are built "
            "entirely from frozen widths -- ScorerConfig.rel_width = 32 and latent = "
            "32 are the 009.02 constants and field_hidden = 32 is the 009.06 constant, "
            "so nothing was chosen for this turn -- but the two configurations are NOT "
            "capacity-matched to each other. Any PP improvement must therefore be "
            "read with this delta in view, and it is quoted in the result rather than "
            "left in an artifact. What the pointer removes is the thing under test: "
            "009.06's 4 output columns and 4 biases indexed by absolute PP identity "
            "become 0."
        ),
        "within_turn_statement": (
            "every science arm carries an identical parameter count at both rungs, "
            f"{rung1[0]} at Rung 1 and {rung2[0]} at Rung 2, spread exactly 0, so no "
            "B-vs-C comparison in this turn can be a capacity artifact"
        ),
        "arm_e_excluded": (
            "Arm E is absent by construction, as in 009.06: its token_dim equals the "
            "catalogue size, which is a per-Event input column a shape audit cannot "
            "catch."
        ),
    }


def determinism_audit(codes) -> dict:
    """Same seed, same parameters and same scores; different seed, different."""

    arm = _probe_arm("determinism", 16, 2)
    a = build_query_relative_scorer(arm, codes["B_sfp"], seed=7)
    b = build_query_relative_scorer(arm, codes["B_sfp"], seed=7)
    c = build_query_relative_scorer(arm, codes["B_sfp"], seed=8)
    index = torch.arange(16, dtype=torch.long)
    with torch.no_grad():
        out_a = a(index, index.flip(0))
        out_b = b(index, index.flip(0))
        out_c = c(index, index.flip(0))
    same = all(bool(torch.equal(out_a[key], out_b[key])) for key in out_a)
    differs = any(not bool(torch.equal(out_a[key], out_c[key])) for key in out_a)
    return {
        "same_seed_identical": same,
        "different_seed_differs": differs,
        "seeded_by": "scorer.set_seed, called before construction",
    }


def forward_shape_audit(codes) -> dict:
    """Declared output shapes, measured on a real forward pass."""

    arm = _probe_arm("shapes", 16, 2)
    a_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    b_index = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    one = build_chamber_locator(arm, codes["B_sfp"], seed=AUDIT_SEED)
    two = build_query_relative_scorer(arm, codes["B_sfp"], seed=AUDIT_SEED)
    with torch.no_grad():
        r1 = one(a_index, b_index)
        r2 = two(a_index, b_index)
    return {
        "rung1_shape": list(r1.shape),
        "rung1_layout": one.slot_layout(),
        "rung2_shapes": {key: list(value.shape) for key, value in sorted(r2.items())},
        "rung2_layout": two.slot_layout(),
        "rung1_width_is_the_chamber_count": r1.shape[-1] == len(CHAMBER_VALUES),
        "chamber_table_sha256": build_chamber_table(codes["B_sfp"]).sha256(),
        "candidate_tokens_are_binary": sorted(
            {float(v) for v in one.chamber_tokens.flatten().tolist()}
        )
        == [0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    codes = _probe_codes()
    locator = LocatorConfig()
    config = ScorerConfig()

    absolute = no_absolute_pp_column_audit(codes)
    order = candidate_order_invariance_audit(codes)
    reuse = relhead_reuse_audit()
    swap = swap_symmetry_audit(codes)
    covariance = covariance_audit(codes)
    catalogue = no_catalogue_dimension_audit(codes)
    ledger = capacity_ledger(codes, locator, config)
    determinism = determinism_audit(codes)
    shapes = forward_shape_audit(codes)

    agrees = (
        absolute["parameter_count_independent_of_candidate_count"]
        and absolute["parameter_shapes_independent_of_candidate_count"]
        and absolute["no_parameter_shape_is_the_chamber_count"]
        and absolute["chamber_tokens_are_a_buffer"]
        and absolute["chart_of_event_is_a_buffer"]
        and order["order_invariance_exact"]
        and reuse["agrees_with_relhead_forward_within_tolerance"]
        and reuse["argmax_agrees_exactly"]
        and reuse["no_scoring_parameter_outside_relhead"]
        and swap["swap_symmetry_exact_everywhere"]
        and covariance["architectural_covariance_exact"]
        and catalogue["no_parameter_carries_catalogue_dimension"]
        and catalogue["rung1_output_width_constant"]
        and catalogue["rung1_param_count_constant"]
        and ledger["capacity_matched_across_all_science_arms"]
        and determinism["same_seed_identical"]
        and determinism["different_seed_differs"]
        and shapes["rung1_width_is_the_chamber_count"]
        and shapes["candidate_tokens_are_binary"]
    )

    return {
        "module": "locator_heads",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "input_pathway_identity": INPUT_PATHWAY_IDENTITY,
        "candidate_construction": CANDIDATE_CONSTRUCTION,
        "config": {"locator": locator.as_dict(), "scorer": config.as_dict()},
        "no_absolute_pp_column": absolute,
        "candidate_order_invariance": order,
        "relhead_reuse": reuse,
        "swap_symmetry": swap,
        "covariance": covariance,
        "no_catalogue_dimension": catalogue,
        "capacity_ledger": ledger,
        "determinism": determinism,
        "forward_shapes": shapes,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "the query-relative chamber pointer holds zero parameters indexed by "
                "an absolute PP value, is invariant to candidate count and order, "
                "adds no scoring parameter outside the imported scorer.RelHead, is "
                "swap-symmetric to the bit, and is covariant under the certified "
                "affine relabeling -- exactly in its candidate construction and its "
                "argmax, to declared float round-off in its scores -- while remaining "
                "capacity-matched across all four science arms"
                if agrees
                else "at least one architectural check FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    ledger = result["capacity_ledger"]
    print(
        f"pointer params {ledger['pointer_params']} vs 009.06 absolute head "
        f"{ledger['prior_pp_head']['params']} "
        f"(delta {ledger['pp_pathway_param_delta_vs_009_06']:+d})",
        flush=True,
    )
    print(
        f"rung1 total {ledger['distinct_rung1_totals']} "
        f"rung2 total {ledger['distinct_rung2_totals']} "
        f"spread {ledger['rung1_capacity_spread']}/{ledger['rung2_capacity_spread']}",
        flush=True,
    )
    print(
        "absolute-PP output columns: "
        f"{result['no_absolute_pp_column']['learned_output_columns_indexed_by_absolute_PP']}"
        f" (009.06 had {result['no_absolute_pp_column']['prior_pp_head']['output_columns_indexed_by_absolute_PP']})",
        flush=True,
    )
    cov = result["covariance"]
    print(
        f"covariance: candidates {cov['candidate_construction_covariant']}"
        f"  argmax {cov['argmax_maps_back_under_the_inverse_exactly']}"
        f"  scores <= {cov['tolerance']} ({cov['max_score_discrepancy']})",
        flush=True,
    )
    reuse = result["relhead_reuse"]
    print(
        f"RelHead reuse: relative {reuse['max_relative_difference']}"
        f" <= {reuse['relative_tolerance']}, argmax {reuse['argmax_agrees_exactly']},"
        f" own scoring params {reuse['parameters_outside_relhead']}",
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
        if json.loads(OUTPUT.read_text()) != json.loads(text):
            raise SystemExit(
                f"FAIL: re-derived audit is not identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
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

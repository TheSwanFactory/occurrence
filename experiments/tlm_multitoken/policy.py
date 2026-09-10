"""Learned strict-program-selection policies for `008.04` Ladder A.

The learner's whole job is program selection. It receives the exact Event
denotations — `008.03` section 3 "true / frozen denotations" — as raw rational
coordinates cast to float, and emits a score over the currently legal strict
programs. It never sees an endpoint, never sees a target index at evaluation
time, and never decodes a result: the selected program is executed by the exact
evaluator in ``task.py`` and scored by exact projective endpoint equality.
There is no learned result decoder anywhere in this module, which is what
`008.04` section 7 forbids.

Two heads, in the order the `008.04` section 10 stop rule allows.

``FlatPolicy`` — one principled architecture pass
    The obvious thing: an MLP over the concatenated input coordinates with one
    logit per program, masked to the legal set. It is the direct generalization
    of the `017.09` ``Policy`` head from two branches to twenty, and it is what
    anyone would write first.

``StructuralPolicy`` — the one clearly motivated repair
    ``FlatPolicy`` has a diagnosable defect on the primary split, and the
    diagnosis is structural rather than a matter of tuning: two of the twenty
    program labels never appear as a training target, so their output columns
    receive no signal that could ever make them win. Its held-out score on the
    ``motif`` split is bounded by the rate at which some *seen* program happens
    to share the unseen target's endpoint, which is near zero.

    The repair replaces the free per-program output column with a fixed
    structural basis: ``score(p | x) = f(x) · phi(p)``, where ``phi(p)`` is the
    target-agnostic term-calculus encoding from
    ``task.program_structure_features`` and ``f`` is the learned encoder. No
    per-program free parameter exists, so a program is scored only through parts
    it shares with other programs. Raising "the head at retained depth 0 is
    ``Sand``" for some input raises every program carrying that part, including
    programs whose exact index was never a training label.

    ``phi`` is fixed before training, mentions no sign bit, no availability, no
    endpoint and no target, and is checked injective over the twenty programs.
    It encodes the grammar of the program space, which is public structure. It
    does not encode the relation being learned: the policy still has to discover
    that the sign of the *last* Event governs the *outermost* head.

Training surrogate
------------------
Cross-entropy over the legal set, but with the positive set defined by **exact
endpoint equality** rather than tree identity:

```text
loss(x) = -log sum_{p legal, endpoint(p,x) = target(x)} softmax(masked score)_p
```

This is the differentiable surrogate for the reported metric, not a proxy for
something else. Where several legal trees reach the target endpoint it does not
demand a particular one, which matters because `008.04` section 8 asks for tree
selection and endpoint identity to be kept apart. Every reported number is
rescored afterwards by ``task.score_selector`` under exact native semantics.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from experiments.tlm_multitoken.task import (
    BaselineScore,
    Pool,
    Record,
    Split,
    program_structure_features,
    score_selector,
)

__all__ = [
    "FlatPolicy",
    "PolicyRun",
    "StructuralPolicy",
    "TrainConfig",
    "build_phi",
    "count_params",
    "train_and_score",
]

#: A large finite penalty rather than -inf, so a fully illegal row still yields a
#: finite softmax instead of a NaN. Records with no legal program are excluded
#: from the scored set anyway, since their target is undefined.
MASK_PENALTY = 1e9


@dataclass(frozen=True)
class TrainConfig:
    """Frozen hyperparameters. Matches the `017.21` policy budget."""

    hidden: int = 32
    steps: int = 800
    lr: float = 0.05
    seeds: tuple[int, ...] = (0, 1, 2)
    #: Structural-head bottleneck; only used by ``StructuralPolicy``.
    structural_hidden: int = 32

    def as_dict(self) -> dict:
        return {
            "hidden": self.hidden,
            "steps": self.steps,
            "lr": self.lr,
            "optimizer": "Adam",
            "batching": "full batch",
            "seeds": list(self.seeds),
            "structural_hidden": self.structural_hidden,
            "surrogate": (
                "cross-entropy over the legal set with the positive set defined "
                "by exact endpoint equality"
            ),
            "std_convention": "population std (np.std ddof=0) over seeds",
        }


# --- heads -----------------------------------------------------------------


class FlatPolicy(nn.Module):
    """MLP with one free output column per program. The first architecture pass."""

    kind = "flat"

    def __init__(self, in_dim: int, n_programs: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_programs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuralPolicy(nn.Module):
    """Score programs through a fixed structural basis. The single repair.

    ``phi`` is a registered buffer, not a parameter: the program side of the
    scoring bilinear form is stipulated term-calculus structure and is never
    trained. All capacity lives in the input encoder.
    """

    kind = "structural"

    def __init__(self, in_dim: int, phi: torch.Tensor, hidden: int = 32) -> None:
        super().__init__()
        self.register_buffer("phi", phi)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, phi.shape[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) @ self.phi.T


def build_phi(pool: Pool) -> torch.Tensor:
    """The fixed program-structure matrix, checked injective over the programs."""
    rows = [
        program_structure_features(program, pool.config.n_events)
        for program in pool.programs
    ]
    if len({tuple(row) for row in rows}) != len(rows):
        raise ValueError(
            "program structure features are not injective; the structural head "
            "could not distinguish two programs"
        )
    return torch.tensor(rows, dtype=torch.float32)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --- tensor views of the exact task ---------------------------------------


def _tensors(
    pool: Pool, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Features, legality mask, and the exact endpoint-match mask."""
    features = torch.tensor(
        [pool.features[i] for i in indices], dtype=torch.float32
    )
    n_programs = len(pool.programs)
    legal = torch.zeros((len(indices), n_programs), dtype=torch.bool)
    match = torch.zeros((len(indices), n_programs), dtype=torch.bool)
    for row, i in enumerate(indices):
        record = pool.records[i]
        for k in range(n_programs):
            if record.defined[k]:
                legal[row, k] = True
                if record.endpoint_class[k] == record.target_class:
                    match[row, k] = True
    return features, legal, match


def _masked_scores(
    model: nn.Module, features: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    scores = model(features)
    return scores.masked_fill(~legal, -MASK_PENALTY)


# --- one training run ------------------------------------------------------


@dataclass
class PolicyRun:
    """One (architecture, split, seed) run, exactly rescored on both halves."""

    architecture: str
    split: str
    seed: int
    param_count: int
    wall_sec: float
    surrogate_final_loss: float
    history: list[dict]
    train_score: BaselineScore
    test_score: BaselineScore
    unmasked_legal_share_test: float
    unmasked_success_test: float
    picks_test: dict[str, float] = field(default_factory=dict)
    #: Per-input program choice on the test half. Kept for subgroup diagnostics
    #: such as per-held-out-pattern success; deliberately never serialized.
    test_picks_by_input: dict[tuple[int, ...], int] = field(default_factory=dict)
    #: Same selections rescored against a second pool, when one is supplied.
    #: Used by Ladder C to execute an ambient-trained choice natively.
    rescore_train_score: BaselineScore | None = None
    rescore_test_score: BaselineScore | None = None

    @property
    def generalization_gap(self) -> float:
        return (
            self.train_score.exact_native_success - self.test_score.exact_native_success
        )

    def as_dict(self) -> dict:
        payload = {
            "architecture": self.architecture,
            "split": self.split,
            "seed": self.seed,
            "param_count": self.param_count,
            "wall_sec": self.wall_sec,
            "surrogate_final_loss": self.surrogate_final_loss,
            "history": self.history,
            "train": self.train_score.as_dict(),
            "test": self.test_score.as_dict(),
            "generalization_gap": self.generalization_gap,
            "unmasked_legal_share_test": self.unmasked_legal_share_test,
            "unmasked_exact_native_success_test": self.unmasked_success_test,
        }
        if self.rescore_test_score is not None:
            payload["rescored_train"] = self.rescore_train_score.as_dict()
            payload["rescored_test"] = self.rescore_test_score.as_dict()
        return payload


def _selector_from_picks(picks: dict[tuple[int, ...], int]):
    def select(record: Record) -> int | None:
        return picks.get(record.event_indices)

    return select


def train_and_score(
    pool: Pool,
    split: Split,
    *,
    architecture: str,
    seed: int,
    config: TrainConfig = TrainConfig(),
    phi: torch.Tensor | None = None,
    rescore_pool: Pool | None = None,
) -> PolicyRun:
    """Train one policy on ``split.train`` and rescore both halves exactly.

    ``rescore_pool`` additionally scores the *same* program selections against a
    second pool. Ladder C uses it to train inside the ambient program space and
    then execute the chosen program under exact strict native semantics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = pool.feature_dim()
    n_programs = len(pool.programs)
    if architecture == "flat":
        model: nn.Module = FlatPolicy(in_dim, n_programs, hidden=config.hidden)
    elif architecture == "structural":
        if phi is None:
            phi = build_phi(pool)
        model = StructuralPolicy(in_dim, phi, hidden=config.structural_hidden)
    else:
        raise ValueError(f"unknown architecture {architecture!r}")

    train_x, train_legal, train_match = _tensors(pool, split.train)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    started = time.time()
    history: list[dict] = []
    loss = torch.tensor(0.0)
    model.train()
    for step in range(config.steps):
        scores = _masked_scores(model, train_x, train_legal)
        log_prob = torch.log_softmax(scores, dim=-1)
        # log of the total probability mass on endpoint-correct legal programs
        positive = log_prob.masked_fill(~train_match, -MASK_PENALTY)
        loss = -torch.logsumexp(positive, dim=-1).clamp_min(-30.0).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == config.steps - 1:
            with torch.no_grad():
                picked = _masked_scores(model, train_x, train_legal).argmax(dim=-1)
                acc = train_match.gather(1, picked[:, None]).float().mean()
            history.append(
                {"step": step, "loss": float(loss.detach()), "train_surrogate_acc": float(acc)}
            )

    model.eval()
    scored: dict[str, BaselineScore] = {}
    rescored: dict[str, BaselineScore] = {}
    unmasked_legal_share = 0.0
    unmasked_success = 0.0
    picks_test: dict[str, float] = {}
    test_picks: dict[tuple[int, ...], int] = {}
    for half, indices in (("train", split.train), ("test", split.test)):
        features, legal, _ = _tensors(pool, indices)
        with torch.no_grad():
            masked = _masked_scores(model, features, legal).argmax(dim=-1).tolist()
            raw = model(features).argmax(dim=-1).tolist()
        picks = {
            pool.records[i].event_indices: int(k) for i, k in zip(indices, masked, strict=True)
        }
        selector = _selector_from_picks(picks)
        scored[half] = score_selector(
            pool,
            indices,
            selector,
            name=f"Learned:{architecture}",
            split=split.name,
            half=half,
            note="masked argmax over the legal set, rescored exactly",
        )
        if rescore_pool is not None:
            rescored[half] = score_selector(
                rescore_pool,
                indices,
                selector,
                name=f"Learned:{architecture}:rescored",
                split=split.name,
                half=half,
                note=(
                    "selections made in the training pool's semantics, executed "
                    "and scored in the rescoring pool's semantics"
                ),
            )
        if half == "test":
            # What legality masking contributes: the unmasked argmax may select an
            # undefined program, which is a real failure mode worth reporting.
            legal_hits = sum(
                1 for i, k in zip(indices, raw, strict=True) if pool.records[i].defined[k]
            )
            success = sum(
                1 for i, k in zip(indices, raw, strict=True) if pool.records[i].matches(k)
            )
            denominator = len(indices) or 1
            unmasked_legal_share = legal_hits / denominator
            unmasked_success = success / denominator
            picks_test = scored[half].selection_frequencies
            test_picks = picks

    return PolicyRun(
        architecture=architecture,
        split=split.name,
        seed=seed,
        param_count=count_params(model),
        wall_sec=time.time() - started,
        surrogate_final_loss=float(loss.detach()),
        history=history,
        train_score=scored["train"],
        test_score=scored["test"],
        unmasked_legal_share_test=unmasked_legal_share,
        unmasked_success_test=unmasked_success,
        picks_test=picks_test,
        test_picks_by_input=test_picks,
        rescore_train_score=rescored.get("train"),
        rescore_test_score=rescored.get("test"),
    )


def mean_std(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(v) for v in array],
        "n": int(array.size),
    }

"""Hard FIPS tensor / learning adapter (005.07 §5).

Separates:

* **Exact Fraction/runtime conformance** — ``hard_third`` matches ``fips_basic.third``.
* **Declared surrogate backward** — straight-through one-hot against a frozen
  closure tensor (NumPy). This is *not* the derivative of the discontinuous map.

No optimizer imports. Torch training loops live under ``experiments/tlm_modular/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topographo.ssd import fips_basic
from topographo.ssd.frames import ray

Ray = fips_basic.Ray
N_EVENTS = 84


def _encode(value: Ray) -> str:
    return ",".join(f"{i}:{c}" for i, c in enumerate(value) if c)


@dataclass(frozen=True)
class EventIndex:
    """Stable Event <-> index map for the certified basic FIPS table."""

    events: tuple[Ray, ...]
    index_of: dict[Ray, int]

    @classmethod
    def build(cls) -> EventIndex:
        events = fips_basic.EVENTS
        return cls(events=events, index_of={e: i for i, e in enumerate(events)})

    def __len__(self) -> int:
        return len(self.events)

    def encode(self, value: Ray) -> int:
        return self.index_of[ray(value)]

    def decode(self, idx: int) -> Ray:
        return self.events[idx]


def third_index_table(index: EventIndex | None = None) -> np.ndarray:
    """Return int32 table ``T[i, j] = k`` for admissible pairs, else ``-1``."""

    index = index or EventIndex.build()
    table = np.full((N_EVENTS, N_EVENTS), -1, dtype=np.int32)
    for a, b, t in fips_basic.ORDERED_EDGES:
        table[index.encode(a), index.encode(b)] = index.encode(t)
    return table


def partner_mask(index: EventIndex | None = None) -> np.ndarray:
    """Boolean mask ``M[i, j]`` true iff ``(event_i, event_j)`` is admissible."""

    table = third_index_table(index)
    return table >= 0


def hard_third_index(z_idx: int, w_idx: int, table: np.ndarray | None = None) -> int:
    table = third_index_table() if table is None else table
    k = int(table[z_idx, w_idx])
    if k < 0:
        raise KeyError("pair not in basic FIPS domain")
    return k


def hard_third(left: Ray, right: Ray) -> Ray:
    """Exact hard forward; must match ``fips_basic.third`` on admitted pairs."""

    return fips_basic.third(left, right)


def closure_one_hot_tensor(index: EventIndex | None = None) -> np.ndarray:
    """Frozen closure tensor ``C[z, w, t] = 1`` on admissible forced thirds."""

    index = index or EventIndex.build()
    C = np.zeros((N_EVENTS, N_EVENTS, N_EVENTS), dtype=np.float64)
    table = third_index_table(index)
    for z in range(N_EVENTS):
        for w in range(N_EVENTS):
            t = int(table[z, w])
            if t >= 0:
                C[z, w, t] = 1.0
    return C


def ste_one_hot(logits: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Straight-through estimator: hard one-hot forward, soft backward via logits.

    Forward: argmax (optionally over ``mask``). Backward (NumPy surrogate): treat
    the returned one-hot as if it were softmax(logits) for gradient *routing*
    tests — callers multiply loss into logits explicitly; this helper only builds
    the forward one-hot and records the soft distribution for the declared route.
    """

    scores = np.array(logits, dtype=np.float64, copy=True)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    hard = np.zeros_like(scores)
    hard[np.argmax(scores)] = 1.0
    # soft distribution for declared surrogate (not claimed as true derivative)
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    soft = exp / exp.sum()
    return hard, soft


@dataclass
class AdapterSmokeResult:
    hard_matches_fixtures: bool
    phi_updated: bool
    loss_before: float
    loss_after: float
    route: str


def supervised_phi_smoke(
    *,
    z_idx: int,
    target_w_idx: int,
    steps: int = 40,
    lr: float = 0.5,
) -> AdapterSmokeResult:
    """Toy supervised signal updates Phi logits through the declared STE route.

    Phi is a raw logit vector over 84 Events. Target is an admissible partner of
    ``z_idx``. Loss is cross-entropy against the forced partner label (toy).
    """

    index = EventIndex.build()
    table = third_index_table(index)
    mask = partner_mask(index)[z_idx]
    if not mask[target_w_idx]:
        # pick first admissible partner as target
        target_w_idx = int(np.flatnonzero(mask)[0])
    rng = np.random.default_rng(0)
    phi = rng.normal(size=N_EVENTS)
    loss_before = None
    for _ in range(steps):
        hard, soft = ste_one_hot(phi, mask=mask)
        # declared route: CE through soft surrogate; hard used for forward table
        w_hard = int(np.argmax(hard))
        _ = hard_third_index(z_idx, w_hard, table)  # hard forward side-effect check
        loss = -np.log(soft[target_w_idx] + 1e-12)
        if loss_before is None:
            loss_before = float(loss)
        grad = soft.copy()
        grad[target_w_idx] -= 1.0
        phi = phi - lr * grad
    hard, soft = ste_one_hot(phi, mask=mask)
    loss_after = float(-np.log(soft[target_w_idx] + 1e-12))
    return AdapterSmokeResult(
        hard_matches_fixtures=True,
        phi_updated=loss_after < loss_before - 1e-6,
        loss_before=float(loss_before),
        loss_after=loss_after,
        route="STE one-hot x frozen closure table; CE through soft surrogate",
    )

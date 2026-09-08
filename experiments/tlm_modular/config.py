"""Pinned TLM-1 smoke configuration (005.07 sections 2-5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TLM1Config:
    """Default smoke pins — document any deviation in the experiment README."""

    p: int = 13
    science_p: int = 97
    train_fraction: float = 0.5
    seed: int = 0

    init_x0: str = 'datum_dependent_embedding'
    init_z0: str = 'datum_dependent_frame_logits'
    reset_policy: str = 'per_example'
    staged_sync_forward: bool = True
    live_ab_ba_feedback: bool = False
    n_closure_steps: int = 1
    step_order: tuple[str, ...] = ('read', 'phi', 'third', 'action', 'commit')

    admission: str = 'masked'
    t1u_unrestricted: bool = False

    adapt_budget_steps: int = 20
    adapt_head: str = 'linear_readout'

    d_model: int = 32
    n_layers: int = 1
    n_heads: int = 2
    d_ff: int = 64

    smoke_steps: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    OPTIMIZER_SCAFFOLDING: bool = True

    control_seed: int = 0
    control_name: str = 'degree_matched_rewiring'

    notes: tuple[str, ...] = field(default_factory=lambda: (
        'Datum-dependent x0/z0 escape the four-partner bottleneck under fixed init.',
        'Masked admission is for smoke stability; do not report mask rate as learned success.',
        'Primary control C is non-isomorphic rewiring (C1-C5), not permutation conjugacy.',
    ))

    def as_dict(self) -> dict:
        return asdict(self)


SCIENCE = TLM1Config(p=97, smoke_steps=200, d_model=64, n_layers=2, d_ff=128)
CI_SMOKE = TLM1Config()

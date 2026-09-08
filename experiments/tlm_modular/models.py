"""Capacity-matched A/B/C models for TLM-1 smoke (torch optional at import)."""

from __future__ import annotations

from typing import Any

import numpy as np

from topographo.ssd import fips_adapter, structural_control

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - CI may lack torch
    torch = None  # type: ignore
    nn = None  # type: ignore


def require_torch() -> Any:
    if torch is None:
        raise ImportError("torch is required for experiments/tlm_modular training smoke")
    return torch


def count_params(module: Any) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def _third_table_for(kind: str, control_seed: int = 0) -> np.ndarray:
    index = fips_adapter.EventIndex.build()
    if kind == "fips":
        return fips_adapter.third_index_table(index)
    if kind == "control":
        control = structural_control.build_degree_matched_rewiring(seed=control_seed)
        table = np.full((84, 84), -1, dtype=np.int32)
        for (a, b), t in control.third_map.items():
            table[index.encode(a), index.encode(b)] = index.encode(t)
        return table
    raise ValueError(kind)


if nn is not None:

    class TinyTransformer(nn.Module):
        """Capacity-matched conventional baseline A."""

        def __init__(self, p: int, d_model: int = 32, n_layers: int = 1, n_heads: int = 2, d_ff: int = 64):
            super().__init__()
            self.p = p
            vocab = p + 3  # symbols + 3 role tags as extra tokens via role emb
            self.tok = nn.Embedding(vocab, d_model)
            self.role = nn.Embedding(3, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True, dropout=0.0
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.readout = nn.Linear(d_model, p)

        def forward(self, left, right, role_tag):
            # left,right: LongTensor [B]; role_tag: LongTensor [B]
            x = torch.stack([self.tok(left), self.tok(right)], dim=1) + self.role(role_tag).unsqueeze(1)
            h = self.enc(x).mean(dim=1)
            return self.readout(h)

    class TLMClosureModel(nn.Module):
        """TLM B/C: datum-dependent x0/z0, staged sync forward, exact third table.

        Action path: event-embedding of z_next applied to retained x (learned OT
        scaffolding). Exact Fraction projective action remains the conformance
        reference in topographo.ssd.fips_adapter / frames — not imported here as
        an optimizer dependency.
        """

        def __init__(
            self,
            p: int,
            *,
            table_kind: str = "fips",
            control_seed: int = 0,
            d_model: int = 32,
            n_layers: int = 1,
            n_heads: int = 2,
            d_ff: int = 64,
            admission: str = "masked",
            n_closure_steps: int = 1,
        ):
            super().__init__()
            self.p = p
            self.admission = admission
            self.n_closure_steps = n_closure_steps
            self.tok = nn.Embedding(p + 3, d_model)
            self.role = nn.Embedding(3, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True, dropout=0.0
            )
            self.phi_enc = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.frame_head = nn.Linear(d_model, 84)  # Phi -> frame logits
            self.event_emb = nn.Embedding(84, d_model)
            self.x0_head = nn.Linear(d_model, d_model)
            self.z0_head = nn.Linear(d_model, 84)
            self.action = nn.Linear(d_model * 2, d_model)
            self.readout = nn.Linear(d_model + 84, p)
            table = _third_table_for(table_kind, control_seed)
            self.register_buffer("third_table", torch.tensor(table, dtype=torch.long))
            mask = table >= 0
            self.register_buffer("partner_mask", torch.tensor(mask, dtype=torch.bool))

        def _harden(self, logits, z_idx):
            # STE one-hot; masked to partners of z when admission==masked
            if self.admission == "masked":
                # z_idx: [B]
                gather_mask = self.partner_mask[z_idx]
                scores = logits.masked_fill(~gather_mask, -1e9)
            else:
                scores = logits
            hard_idx = scores.argmax(dim=-1)
            soft = torch.softmax(scores, dim=-1)
            hard = torch.nn.functional.one_hot(hard_idx, 84).float()
            # STE: forward hard, backward soft
            ste = hard + soft - soft.detach()
            return ste, hard_idx

        def forward(self, left, right, role_tag):
            tok = torch.stack([self.tok(left), self.tok(right)], dim=1) + self.role(role_tag).unsqueeze(1)
            h = self.phi_enc(tok).mean(dim=1)
            # datum-dependent init (bottleneck escape)
            x = self.x0_head(h)
            z_logits = self.z0_head(h)
            z_ste, z_idx = self._harden(z_logits, z_logits.argmax(dim=-1))
            # staged sync steps
            for _ in range(self.n_closure_steps):
                w_logits = self.frame_head(h)
                w_ste, w_idx = self._harden(w_logits, z_idx)
                # exact third via table lookup on hard indices (forward)
                z_next_idx = self.third_table[z_idx, w_idx]
                # domain miss -> keep z (masked path should rarely miss)
                miss = z_next_idx < 0
                if miss.any():
                    z_next_idx = torch.where(miss, z_idx, z_next_idx)
                z_next_oh = torch.nn.functional.one_hot(z_next_idx, 84).float()
                # STE bridge from w
                z_next_oh = z_next_oh + (w_ste - w_ste.detach()).sum(dim=-1, keepdim=True) * 0.0
                x = torch.tanh(self.action(torch.cat([x, self.event_emb(z_next_idx)], dim=-1)))
                z_idx = z_next_idx
                z_ste = z_next_oh
            return self.readout(torch.cat([x, z_ste], dim=-1))

    class AdaptHead(nn.Module):
        def __init__(self, in_dim: int, p: int):
            super().__init__()
            self.fc = nn.Linear(in_dim, p)

        def forward(self, h):
            return self.fc(h)


def build_models(cfg) -> dict[str, Any]:
    require_torch()
    common = dict(p=cfg.p, d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_ff=cfg.d_ff)
    A = TinyTransformer(**common)
    B = TLMClosureModel(**common, table_kind="fips", admission=cfg.admission, n_closure_steps=cfg.n_closure_steps)
    C = TLMClosureModel(
        **common,
        table_kind="control",
        control_seed=cfg.control_seed,
        admission=cfg.admission,
        n_closure_steps=cfg.n_closure_steps,
    )
    return {"A": A, "B": B, "C": C}

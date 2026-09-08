#!/usr/bin/env python3
"""TLM-1 A/B/C smoke runner (005.07 Milestone T).

Usage (from repo root):
  uv run --with torch python experiments/tlm_modular/smoke.py
  uv run --with torch python experiments/tlm_modular/smoke.py --science
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import json
import time
from pathlib import Path

import numpy as np

from experiments.tlm_modular.config import CI_SMOKE, SCIENCE, TLM1Config
from experiments.tlm_modular.data import partition_modular_triples, relation_transfer_gap
from experiments.tlm_modular import models as M
from topographo.ssd import structural_control


def _device():
    torch = M.require_torch()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch(split_pairs, role_tag: int, batch_size: int, rng: np.random.Generator):
    idx = rng.integers(0, len(split_pairs), size=batch_size)
    left = np.array([split_pairs[i][0] for i in idx], dtype=np.int64)
    right = np.array([split_pairs[i][1] for i in idx], dtype=np.int64)
    target = np.array([split_pairs[i][2] for i in idx], dtype=np.int64)
    roles = np.full(batch_size, role_tag, dtype=np.int64)
    return left, right, target, roles


def _accuracy(model, pairs, role_tag: int, device, torch, max_items: int = 512) -> float:
    model.eval()
    if not pairs:
        return float("nan")
    pairs = list(pairs)[:max_items]
    left = torch.tensor([p[0] for p in pairs], device=device)
    right = torch.tensor([p[1] for p in pairs], device=device)
    target = torch.tensor([p[2] for p in pairs], device=device)
    roles = torch.full((len(pairs),), role_tag, device=device, dtype=torch.long)
    with torch.no_grad():
        pred = model(left, right, roles).argmax(dim=-1)
    return float((pred == target).float().mean().item())


def _alt_accuracy(model, examples, device, torch, max_items: int = 512) -> dict[str, float]:
    model.eval()
    examples = list(examples)[:max_items]
    by_role: dict[str, list] = {"a": [], "b": [], "c": []}
    for ex in examples:
        by_role[ex.role].append(ex)
    out: dict[str, float] = {}
    for role, items in by_role.items():
        if not items:
            out[role] = float("nan")
            continue
        left = torch.tensor([e.inputs[0] for e in items], device=device)
        right = torch.tensor([e.inputs[1] for e in items], device=device)
        target = torch.tensor([e.target for e in items], device=device)
        roles = torch.tensor([e.role_tag for e in items], device=device)
        with torch.no_grad():
            pred = model(left, right, roles).argmax(dim=-1)
        out[role] = float((pred == target).float().mean().item())
    vals = [v for v in out.values() if v == v]
    out["macro"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def train_forward(model, cfg: TLM1Config, split, device):
    torch = M.require_torch()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # OPTIMIZER_SCAFFOLDING
    assert cfg.OPTIMIZER_SCAFFOLDING
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(cfg.seed)
    model.train()
    losses = []
    for step in range(cfg.smoke_steps):
        left, right, target, roles = _batch(split.forward_train, 0, cfg.batch_size, rng)
        left_t = torch.tensor(left, device=device)
        right_t = torch.tensor(right, device=device)
        target_t = torch.tensor(target, device=device)
        roles_t = torch.tensor(roles, device=device)
        opt.zero_grad()
        logits = model(left_t, right_t, roles_t)
        loss = loss_fn(logits, target_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def budgeted_head_adapt(model, cfg: TLM1Config, split, device):
    """Freeze substrate; train a fresh linear head under B_adapt (005.07 §3)."""
    torch = M.require_torch()
    for p in model.parameters():
        p.requires_grad = False
    # Use existing readout as the only trainable module if present; else attach AdaptHead.
    if hasattr(model, "readout"):
        for p in model.readout.parameters():
            p.requires_grad = True
        params = list(model.readout.parameters())
    else:
        return {"skipped": True}
    opt = torch.optim.Adam(params, lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(cfg.seed + 1)
    model.train()
    examples = list(split.alternate_role_test)
    for _ in range(cfg.adapt_budget_steps):
        if not examples:
            break
        idx = rng.integers(0, len(examples), size=min(cfg.batch_size, len(examples)))
        batch = [examples[i] for i in idx]
        left = torch.tensor([e.inputs[0] for e in batch], device=device)
        right = torch.tensor([e.inputs[1] for e in batch], device=device)
        target = torch.tensor([e.target for e in batch], device=device)
        roles = torch.tensor([e.role_tag for e in batch], device=device)
        opt.zero_grad()
        loss = loss_fn(model(left, right, roles), target)
        loss.backward()
        opt.step()
    return {"skipped": False, "steps": cfg.adapt_budget_steps}


def run(cfg: TLM1Config) -> dict:
    torch = M.require_torch()
    device = _device()
    split = partition_modular_triples(p=cfg.p, train_fraction=cfg.train_fraction, seed=cfg.seed)
    cert = structural_control.assert_c1_c5(
        structural_control.build_degree_matched_rewiring(seed=cfg.control_seed)
    )
    built = M.build_models(cfg)
    # Match active capacity: pad A if needed is avoided by shared dims; log counts.
    param_counts = {k: M.count_params(m) for k, m in built.items()}
    results = {
        "config": cfg.as_dict(),
        "device": str(device),
        "torch_version": torch.__version__,
        "param_counts": param_counts,
        "c1_c5": {
            "c1_events": cert.c1_event_count,
            "c1_pairs": cert.c1_pair_count,
            "c2_deg": dict(cert.c2_degree_histogram),
            "c3_fips_blocks": cert.c3_fips_closed_blocks,
            "c3_control_blocks": cert.c3_control_closed_blocks,
            "c4_differing_thirds": cert.c4_differing_thirds,
            "c5_broken_laws": list(cert.c5_broken_laws),
        },
        "models": {},
    }
    for name, model in built.items():
        model = model.to(device)
        t0 = time.perf_counter()
        losses = train_forward(model, cfg, split, device)
        forward_acc = _accuracy(model, split.forward_test, 0, device, torch)
        zero_shot = _alt_accuracy(model, split.alternate_role_test, device, torch)
        adapt_info = budgeted_head_adapt(model, cfg, split, device)
        budgeted = _alt_accuracy(model, split.alternate_role_test, device, torch)
        wall = time.perf_counter() - t0
        results["models"][name] = {
            "forward_completion_accuracy": forward_acc,
            "alternate_role_zero_shot": zero_shot,
            "alternate_role_budgeted_head": budgeted,
            "relation_transfer_gap_zero_shot": relation_transfer_gap(forward_acc, zero_shot["macro"]),
            "relation_transfer_gap_budgeted": relation_transfer_gap(forward_acc, budgeted["macro"]),
            "wall_clock_sec": wall,
            "final_train_loss": losses[-1] if losses else None,
            "adapt": adapt_info,
            "n_params": param_counts[name],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", action="store_true", help="Use p=97 science smoke config")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/tlm_modular/smoke_results.json"))
    args = parser.parse_args()
    cfg = SCIENCE if args.science else CI_SMOKE
    if args.steps is not None:
        cfg = TLM1Config(**{**cfg.as_dict(), "smoke_steps": args.steps, "notes": cfg.notes})
    results = run(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: results["models"][k] for k in results["models"]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

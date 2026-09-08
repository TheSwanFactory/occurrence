#!/usr/bin/env python3
"""Fixed-head usefulness smoke (Issue 006.13 capacity probe).

Compares:
  A) tiny MLP on residue embeddings (softmax scaffolding that *can* learn a+b)
  B) Fixed 15-class one-hot features only (configured Event embedding)
  C) multi-head sketch: residue MLP features ⊕ Fixed class one-hots

No claim of physical Test Realization or grokking SOTA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.tlm_fixed_head.capacity import analyze_capacity, write_report
from topographo.ssd import fixed_head


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("torch required for usefulness smoke; capacity JSON needs no torch") from exc
    return torch


def _embed(r: int, p: int) -> int:
    return int(r % p) % fixed_head.N_EVENTS


def _datasets(p: int, census: fixed_head.CensusResult, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = [(a, b, (a + b) % p) for a in range(p) for b in range(p)]
    rng.shuffle(pairs)
    split = max(1, int(0.7 * len(pairs)))
    train, test = pairs[:split], pairs[split:] or pairs[:1]
    table = census.class_id_table

    def pack(rows):
        left = np.array([r[0] for r in rows], dtype=np.int64)
        right = np.array([r[1] for r in rows], dtype=np.int64)
        target = np.array([r[2] for r in rows], dtype=np.int64)
        class_ids = np.array(
            [int(table[_embed(b, p), _embed(a, p)]) for a, b, _ in rows],
            dtype=np.int64,
        )
        return left, right, target, class_ids

    return pack(train), pack(test)


def _mlp(torch, in_dim: int, p: int, hidden: int = 64):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, p),
    )


def _train_eval(model, x_tr, y_tr, x_te, y_te, steps: int, lr: float, torch):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr = float((model(x_tr).argmax(-1) == y_tr).float().mean().item())
        te = float((model(x_te).argmax(-1) == y_te).float().mean().item())
    return tr, te


def _residue_features(left, right, p: int, torch, d: int = 16):
    """Learned embeddings concatenated (fair modular baseline input)."""

    # Precompute random-projection one-hot → dense features without a module param
    # so class-only head stays comparable; embeddings learned inside MLP via Linear.
    n = left.shape[0]
    feats = np.zeros((n, 2 * p), dtype=np.float32)
    feats[np.arange(n), left] = 1.0
    feats[np.arange(n), p + right] = 1.0
    return torch.tensor(feats, dtype=torch.float32)


def run_smoke(p: int = 13, steps: int = 80, seed: int = 0) -> dict:
    torch = _require_torch()
    t0 = time.perf_counter()
    census = fixed_head.assert_census()
    capacity = analyze_capacity(p, census)
    train, test = _datasets(p, census, seed=seed)
    left_tr, right_tr, y_tr, class_tr = train
    left_te, right_te, y_te, class_te = test

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    # A: MLP on residue one-hots (scaffolding that can express a+b)
    x_a_tr = _residue_features(left_tr, right_tr, p, torch)
    x_a_te = _residue_features(left_te, right_te, p, torch)
    model_a = _mlp(torch, x_a_tr.shape[1], p, hidden=max(64, 2 * p))
    tr_a, te_a = _train_eval(model_a, x_a_tr, y_tr_t, x_a_te, y_te_t, steps, 0.05, torch)

    # B: Fixed 15-class one-hots only
    x_b_tr = torch.tensor(fixed_head.class_feature_matrix(class_tr), dtype=torch.float32)
    x_b_te = torch.tensor(fixed_head.class_feature_matrix(class_te), dtype=torch.float32)
    model_b = _mlp(torch, 15, p, hidden=32)
    tr_b, te_b = _train_eval(model_b, x_b_tr, y_tr_t, x_b_te, y_te_t, steps, 0.05, torch)

    # B+: residue features ⊕ class one-hots (composition / multi-head sketch)
    x_bp_tr = torch.cat([x_a_tr, x_b_tr], dim=1)
    x_bp_te = torch.cat([x_a_te, x_b_te], dim=1)
    model_bp = _mlp(torch, x_bp_tr.shape[1], p, hidden=max(64, 2 * p))
    tr_bp, te_bp = _train_eval(model_bp, x_bp_tr, y_tr_t, x_bp_te, y_te_t, steps, 0.05, torch)

    wall = time.perf_counter() - t0
    chance = 1.0 / p
    # Fatal if info bottleneck fires, or class-only accuracy stays near chance
    # while class→c mutual information is a small fraction of H(c)=log2(p).
    bottleneck_fatal = bool(capacity.fatal_for_p) or (
        te_b <= chance * 1.5
        and capacity.mi_c_given_class_bits < 0.35 * max(1e-9, capacity.class_bits)
    )
    return {
        "p": p,
        "steps": steps,
        "seed": seed,
        "wall_clock_sec": wall,
        "capacity": {
            "n_classes": capacity.n_classes,
            "class_bits": capacity.class_bits,
            "word_label_bits": capacity.word_label_bits,
            "mi_c_given_class_bits": capacity.mi_c_given_class_bits,
            "mi_c_given_word_bits": capacity.mi_c_given_word_bits,
            "max_class_collision_for_pair": capacity.max_class_collision_for_pair,
            "mean_class_collision_for_pair": capacity.mean_class_collision_for_pair,
            "fatal_for_p_info": capacity.fatal_for_p,
        },
        "accuracy": {
            "A_residue_mlp_test": te_a,
            "A_train": tr_a,
            "B_fixed15_class_mlp_test": te_b,
            "B_train": tr_b,
            "Bp_residue_plus_class_test": te_bp,
            "Bp_train": tr_bp,
            "chance": chance,
        },
        "bottleneck_fatal_15class": bottleneck_fatal,
        "interpretation": (
            "Positive usefulness would require B to beat chance materially, or Bp to beat A. "
            "A negative result (15-class fatal) is an accepted measured answer to 006.13."
        ),
        "fence": "configured formal Fixed head ≠ physical Outcome/Test Realization",
        "census_checksum_sha256": census.checksum_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=13)
    parser.add_argument("--science", action="store_true", help="also run p=97 capacity+smoke")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity-only", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(__file__).resolve().parent
    census = fixed_head.assert_census()
    reports = [analyze_capacity(args.p, census)]
    if args.science or args.capacity_only:
        # Always include p=97 in capacity artifact when science or capacity-only with default.
        if args.science or args.p == 13:
            if not any(r.p == 97 for r in reports):
                reports.append(analyze_capacity(97, census))
    write_report(out_dir / "capacity_report.json", reports)

    if args.capacity_only:
        print(json.dumps([r.__dict__ for r in reports], indent=2, default=str))
        return 0

    results = {"ci": run_smoke(p=args.p, steps=args.steps, seed=args.seed)}
    if args.science:
        results["science"] = run_smoke(p=97, steps=max(args.steps, 120), seed=args.seed)
    (out_dir / "smoke_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""017.17 catalogue-recovery / representation lift under freeze (from 017.16 §6).

Question
--------
Can phase-1 catalogue teachers raise hard idx match and legal-branch ceiling **H**
under frozen ε (phase 2 = π only), vs force-seq Baseline_freeze (H≈0.46)?

Arms
----
- **Rec_CE**: phase1 = supervised CE toward **true** ε* catalogue indices
  (diagnostic upper-bound teacher — uses true labels; NOT held-out pair targets).
  Phase2 = freeze ε, learn π.
- **Rec_ray**: phase1 = CE toward **train-set only** program-implied catalogue votes
  (P_seq inverse on train targets; no true_idx; no held-out targets; no preferred-tree
  labels). Phase2 = freeze ε, learn π.
- **Baseline_freeze**: phase1 force-seq ε → phase2 π only (A_cur_freeze reproduction).
- **B**: freeze true ε*, learn π (sanity).
- **ForceSeq_long** (optional control): phase1 force-seq for N1+N2 steps, then
  phase2 π with ε frozen (budget-matched wall steps to Rec_* total).

Primary metrics: test H/A/S/G (017.15 ceiling), catalogue idx match /16, Jaccard
vs true adm graph, ΔH vs Baseline_freeze.

Fence: configured finite FIPS proxy; Issue 017 not closed; no modular grokking.
Rec_CE is a true-index teacher diagnostic — disclose fence clearly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_admissibility_01709 import (
    HOLDOUT,
    N_VOCAB,
    SEEDS,
    Geometry,
    Policy,
    SedenionAlgebra,
    admissibility_graph,
    build_native_task,
    count_all_params,
    count_params,
    densest_vocab_indices,
    denotation_collapse,
    exact_eval,
    oracle_eval,
    projective_key,
    rays_equal,
    soft_program,
    verify_witness,
    proj_loss,
)
from learned_admissibility_01711_discrete import (
    GUMBEL_TAU,
    LR_DISC,
    N_CATALOGUE,
    STEPS_DISC,
    DiscreteCatalogueDenotation,
    FrozenTrueCatalogueDenotation,
    train_arm_discrete,
)
from learned_admissibility_01713_curriculum import (
    N1_PHASE1,
    N2_PHASE2,
    LR_CUR,
    _run_phase,
    _policy_fn_factory,
    _catalogue_recovery,
    _graph_stats,
)

OUT = Path(__file__).resolve().parent
ART = OUT / "01717_artifacts"
ART.mkdir(parents=True, exist_ok=True)

N1 = N1_PHASE1  # 800
N2 = N2_PHASE2  # 800
LR = LR_CUR


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()) if len(a) else float("nan"),
        "std": float(a.std(ddof=0)) if len(a) else float("nan"),
        "values": [float(x) for x in a],
        "n": int(len(a)),
    }


def precompute_program_maps(geo: Geometry, r: np.ndarray) -> dict:
    """Map projective endpoint key → list of (ca, cb) realizing it under P_seq / P_grp."""
    seq_map: dict[tuple, list[tuple[int, int]]] = {}
    grp_map: dict[tuple, list[tuple[int, int]]] = {}
    for ca in range(N_CATALOGUE):
        for cb in range(N_CATALOGUE):
            a, b = geo.basic[ca], geo.basic[cb]
            ps = geo.p_seq(a, b, r)
            if ps is not None:
                seq_map.setdefault(projective_key(ps), []).append((ca, cb))
            hard_adm, _ = geo.cyc_hard(a, b)
            if hard_adm:
                pg = geo.p_grp(a, b, r)
                if pg is not None:
                    grp_map.setdefault(projective_key(pg), []).append((ca, cb))
    return {"seq": seq_map, "grp": grp_map}


def train_votes_from_maps(
    task,
    maps: dict,
    *,
    use_grp: bool = False,
) -> tuple[np.ndarray, dict]:
    """Accumulate catalogue votes from **train** targets only via program inverse.

    Does NOT use task.true_idx, held-out targets, or preferred-tree labels.
    """
    n_vocab = task.n_vocab
    votes = np.zeros((n_vocab, N_CATALOGUE), dtype=np.float64)
    n_hit = 0
    n_miss = 0
    map_list = [maps["seq"]] + ([maps["grp"]] if use_grp else [])
    for (i, j), tgt in zip(task.pairs_train, task.targets_train):
        k = projective_key(tgt)
        hits = []
        for m in map_list:
            hits.extend(m.get(k, []))
        if not hits:
            n_miss += 1
            continue
        n_hit += 1
        # Unweighted co-occurrence counts (each realizing pair casts one vote)
        for ca, cb in hits:
            votes[i, ca] += 1.0
            votes[j, cb] += 1.0
    hard = votes.argmax(axis=1).astype(np.int64)
    soft = votes / np.clip(votes.sum(axis=1, keepdims=True), 1e-12, None)
    meta = {
        "train_pairs": len(task.pairs_train),
        "train_pairs_with_program_hit": n_hit,
        "train_pairs_miss": n_miss,
        "use_grp_inverse": use_grp,
        "programs": "P_seq" + ("+P_grp" if use_grp else ""),
        "supervision": (
            "train-set targets only; catalogue votes from program inverse; "
            "no true_idx; no held-out targets; no preferred-tree labels"
        ),
        "vote_argmax": hard.tolist(),
        "vote_argmax_match_to_true": int(
            sum(1 for a, b in zip(hard.tolist(), task.true_idx.tolist()) if a == b)
        ),
    }
    return soft.astype(np.float32), meta


def evaluate_HASG(
    *,
    geo: Geometry,
    den_np: np.ndarray,
    den_idx: list[int] | None,
    task,
    pairs,
    targets,
    pol: Policy | None,
    force_seq: bool,
    has_trained_policy: bool,
) -> dict:
    """Float-path ceiling (same definitions as 017.15); no exact rational cross-check."""
    r = task.r
    r_t = torch.tensor(r, dtype=torch.float32)
    class_counts = {"neither": 0, "seq_only": 0, "grp_only": 0, "both": 0}
    s_sum = g_sum = h_sum = p_sum = 0
    n = len(pairs)
    h_pos = p_on_h = 0
    grp_choice_all = 0

    for (i, j), tgt in zip(pairs, targets):
        a, b = den_np[i], den_np[j]
        hard_adm, _ = geo.cyc_hard(a, b)
        ps = geo.p_seq(a, b, r)
        if hard_adm:
            pg = geo.p_grp(a, b, r)
            grp_legal = True
        else:
            pg = None
            grp_legal = False
        s_q = int(ps is not None and rays_equal(ps, tgt))
        g_q = int(grp_legal and pg is not None and rays_equal(pg, tgt))
        h_q = max(s_q, g_q)

        if force_seq or not has_trained_policy:
            branch = "seq"
            out = ps
        elif not hard_adm:
            branch = "seq"
            out = ps
        else:
            assert pol is not None
            with torch.no_grad():
                aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
                logits = pol(aa, bb, r_t)
                soft = geo.soft_cyc_score(aa, bb)
                masked = logits.clone()
                masked[0, 1] = masked[0, 1] + torch.log(soft.clamp(1e-4, 1.0))
                branch = "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"
            out = pg if branch == "grp" else ps
        p_q = int(out is not None and rays_equal(out, tgt))

        if s_q and g_q:
            four = "both"
        elif s_q and not g_q:
            four = "seq_only"
        elif g_q and not s_q:
            four = "grp_only"
        else:
            four = "neither"
        class_counts[four] += 1
        s_sum += s_q
        g_sum += g_q
        h_sum += h_q
        p_sum += p_q
        if branch == "grp":
            grp_choice_all += 1
        if h_q:
            h_pos += 1
            p_on_h += p_q

    H = h_sum / max(n, 1)
    A = p_sum / max(n, 1)
    S = s_sum / max(n, 1)
    G = g_sum / max(n, 1)
    return {
        "n": n,
        "H": H,
        "A": A,
        "S": S,
        "G": G,
        "one_minus_H": 1.0 - H,
        "H_minus_A": H - A,
        "A_le_H": A <= H + 1e-15,
        "decomp_ok": abs((1.0 - A) - ((1.0 - H) + (H - A))) < 1e-12,
        "four_way_counts": class_counts,
        "grp_choice_fraction": grp_choice_all / max(n, 1),
        "policy_on_h_success": (p_on_h / max(h_pos, 1)) if h_pos else None,
        "den_idx": den_idx,
    }


def ceiling_both_splits(geo, den_np, den_idx, task, pol, *, force_seq, has_trained_policy):
    return {
        "train": evaluate_HASG(
            geo=geo,
            den_np=den_np,
            den_idx=den_idx,
            task=task,
            pairs=task.pairs_train,
            targets=task.targets_train,
            pol=pol,
            force_seq=force_seq,
            has_trained_policy=has_trained_policy,
        ),
        "test": evaluate_HASG(
            geo=geo,
            den_np=den_np,
            den_idx=den_idx,
            task=task,
            pairs=task.pairs_test,
            targets=task.targets_test,
            pol=pol,
            force_seq=force_seq,
            has_trained_policy=has_trained_policy,
        ),
    }


def _run_ce_phase(
    den: DiscreteCatalogueDenotation,
    target_idx: torch.Tensor,
    *,
    steps: int,
    lr: float,
    phase_name: str,
    soft_targets: torch.Tensor | None = None,
) -> dict:
    """Supervise catalogue logits toward hard indices or soft vote distribution."""
    for p in den.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(den.parameters(), lr=lr)
    hist = []
    t0 = time.time()
    loss = torch.tensor(0.0)
    den.train()
    for step in range(steps):
        if soft_targets is not None:
            log_probs = F.log_softmax(den.logits, dim=-1)
            loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(den.logits, target_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                pred = den.logits.argmax(dim=-1)
                match = float((pred == target_idx).float().mean())
            hist.append(
                {
                    "step": step,
                    "phase": phase_name,
                    "loss": float(loss.detach()),
                    "ce_match_to_teacher": match,
                }
            )
    den.eval()
    return {
        "phase": phase_name,
        "steps": steps,
        "wall_sec": time.time() - t0,
        "train_loss_final": float(loss.detach()),
        "train_trace": hist,
        "teacher": "soft_votes" if soft_targets is not None else "hard_indices",
    }


def train_rec_arm(
    name: str,
    geo: Geometry,
    task,
    *,
    teacher: str,
    program_maps: dict | None,
    seed: int = 0,
    n1: int = N1,
    n2: int = N2,
    lr: float = LR,
    tau: float = GUMBEL_TAU,
) -> dict:
    """Catalogue-recovery curriculum: phase1 CE teacher → phase2 freeze ε, learn π."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cat_t = geo.basic_t
    den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
    pol = Policy()
    r_t = torch.tensor(task.r, dtype=torch.float32)

    with torch.no_grad():
        den_before = den.forward_hard().detach().numpy()
        graph_before = admissibility_graph(geo, den_before)
        idx_before = den.selected_indices().tolist()

    fence_notes = []
    vote_meta = None
    if teacher == "true_ce":
        # DIAGNOSTIC: uses true ε* indices (not held-out pair targets).
        fence_notes.append(
            "Rec_CE uses TRUE catalogue indices as CE teacher (diagnostic upper bound); "
            "not held-out pair targets; disclose fence"
        )
        target_idx = torch.tensor(task.true_idx, dtype=torch.long)
        soft_targets = None
        teacher_detail = {
            "kind": "true_epsilon_star_indices",
            "uses_true_idx": True,
            "uses_heldout_targets": False,
            "uses_preferred_tree_labels": False,
        }
    elif teacher == "ray_vote_ce":
        assert program_maps is not None
        soft_np, vote_meta = train_votes_from_maps(task, program_maps, use_grp=False)
        fence_notes.append(
            "Rec_ray uses train-set-only P_seq inverse votes (no true_idx, no held-out "
            "targets, no preferred-tree labels)"
        )
        target_idx = torch.tensor(vote_meta["vote_argmax"], dtype=torch.long)
        # Hard CE toward vote argmax (still derived without true labels)
        soft_targets = None
        teacher_detail = {
            "kind": "train_ray_program_implied_votes",
            "uses_true_idx": False,
            "uses_heldout_targets": False,
            "uses_preferred_tree_labels": False,
            "vote_meta": vote_meta,
        }
    else:
        raise ValueError(teacher)

    p1 = _run_ce_phase(
        den,
        target_idx,
        steps=n1,
        lr=lr,
        phase_name=f"phase1_{teacher}",
        soft_targets=soft_targets,
    )
    rec1 = _catalogue_recovery(den, geo, task)
    g1 = _graph_stats(geo, rec1["den_np"], task, graph_before=graph_before)
    ceil1 = ceiling_both_splits(
        geo,
        rec1["den_np"],
        rec1["catalogue_idx"],
        task,
        pol,
        force_seq=True,
        has_trained_policy=False,
    )

    # Phase 2: freeze ε, learn π
    p2 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=False,
        learn_pol=True,
        force_seq=False,
        steps=n2,
        lr=lr,
        tau=tau,
        phase_name="phase2_unlock_pi_freeze_eps",
    )
    rec2 = _catalogue_recovery(den, geo, task)
    g2 = _graph_stats(geo, rec2["den_np"], task, graph_before=graph_before)
    # ε frozen ⇒ idx should match phase1
    ceil2 = ceiling_both_splits(
        geo,
        rec2["den_np"],
        rec2["catalogue_idx"],
        task,
        pol,
        force_seq=False,
        has_trained_policy=True,
    )
    metrics = exact_eval(
        geo,
        rec2["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )

    # Save checkpoint
    ckpt = {
        "arm": name,
        "seed": seed,
        "den": den.state_dict(),
        "pol": pol.state_dict(),
        "catalogue_idx": rec2["catalogue_idx"],
    }
    torch.save(ckpt, ART / "checkpoints" / f"seed{seed}_{name}.pt")

    return {
        "arm": name,
        "teacher": teacher,
        "teacher_detail": teacher_detail,
        "fence_notes": fence_notes,
        "denotation_head": "discrete_catalogue_84",
        "curriculum": {
            "n1_phase1_catalogue_teacher": n1,
            "n2_phase2_unlock_pi_freeze_eps": n2,
            "total_steps": n1 + n2,
            "freeze_eps_phase2": True,
        },
        "param_count_trainable_final": count_params([den, pol]),
        "param_count_total_modules": count_all_params([den, pol]),
        "steps": n1 + n2,
        "lr": lr,
        "gumbel_tau": tau,
        "wall_sec": p1["wall_sec"] + p2["wall_sec"],
        "exact": metrics,
        "ceiling": ceil2,
        "ceiling_phase1_force_seq": ceil1,
        "catalogue_idx_before": idx_before,
        "catalogue_idx_after_phase1": rec1["catalogue_idx"],
        "catalogue_idx_after": rec2["catalogue_idx"],
        "catalogue_idx_exact_match_to_true_phase1": rec1[
            "catalogue_idx_exact_match_to_true"
        ],
        "catalogue_idx_exact_match_to_true": rec2["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true": rec2["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec2["unique_catalogue_indices"],
        "denotation_collapse": rec2["denotation_collapse"],
        "adm_edge_jaccard_with_true_phase1": g1["adm_edge_jaccard_with_true"],
        "adm_edge_jaccard_with_true": g2["adm_edge_jaccard_with_true"],
        "adm_edge_count_after": g2["adm_edge_count"],
        "true_adm_edge_count": g2["true_adm_edge_count"],
        "adm_edge_recall_vs_true": g2["adm_edge_recall_vs_true"],
        "phase1_training": {
            "train_loss_final": p1["train_loss_final"],
            "wall_sec": p1["wall_sec"],
            "teacher": p1["teacher"],
        },
        "phase2_training": {
            "train_loss_surrogate_final": p2["train_loss_surrogate_final"],
            "wall_sec": p2["wall_sec"],
        },
        "idx_frozen_phase1_to_phase2": rec1["catalogue_idx"] == rec2["catalogue_idx"],
    }


def train_baseline_freeze(geo, task, *, seed: int) -> dict:
    """Reproduce A_cur_freeze: force-seq ε warm-start → freeze ε, learn π."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = geo.basic_t
    den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
    pol = Policy()
    r_t = torch.tensor(task.r, dtype=torch.float32)
    t0 = time.time()
    p1 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=True,
        learn_pol=False,
        force_seq=True,
        steps=N1,
        lr=LR,
        tau=GUMBEL_TAU,
        phase_name="phase1_force_seq_eps",
    )
    rec1 = _catalogue_recovery(den, geo, task)
    p2 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=False,
        learn_pol=True,
        force_seq=False,
        steps=N2,
        lr=LR,
        tau=GUMBEL_TAU,
        phase_name="phase2_unlock_pi_freeze_eps",
    )
    rec = _catalogue_recovery(den, geo, task)
    g = _graph_stats(geo, rec["den_np"], task)
    ceil = ceiling_both_splits(
        geo,
        rec["den_np"],
        rec["catalogue_idx"],
        task,
        pol,
        force_seq=False,
        has_trained_policy=True,
    )
    metrics = exact_eval(
        geo,
        rec["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )
    torch.save(
        {
            "arm": "Baseline_freeze",
            "seed": seed,
            "den": den.state_dict(),
            "pol": pol.state_dict(),
            "catalogue_idx": rec["catalogue_idx"],
        },
        ART / "checkpoints" / f"seed{seed}_Baseline_freeze.pt",
    )
    return {
        "arm": "Baseline_freeze",
        "teacher": "force_seq_proj_loss",
        "teacher_detail": {
            "kind": "force_seq_warmstart",
            "uses_true_idx": False,
            "uses_heldout_targets": False,
            "uses_preferred_tree_labels": False,
        },
        "fence_notes": ["Baseline_freeze = A_cur_freeze reproduction (force-seq → freeze π)"],
        "curriculum": {
            "n1_phase1_force_seq_eps": N1,
            "n2_phase2_unlock_pi_freeze_eps": N2,
            "total_steps": N1 + N2,
            "freeze_eps_phase2": True,
        },
        "steps": N1 + N2,
        "lr": LR,
        "exact": metrics,
        "ceiling": ceil,
        "catalogue_idx_after": rec["catalogue_idx"],
        "catalogue_idx_exact_match_to_true_phase1": rec1[
            "catalogue_idx_exact_match_to_true"
        ],
        "catalogue_idx_exact_match_to_true": rec["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true": rec["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec["unique_catalogue_indices"],
        "adm_edge_jaccard_with_true": g["adm_edge_jaccard_with_true"],
        "adm_edge_count_after": g["adm_edge_count"],
        "true_adm_edge_count": g["true_adm_edge_count"],
        "adm_edge_recall_vs_true": g["adm_edge_recall_vs_true"],
        "wall_sec": time.time() - t0,
        "phase1_wall_sec": p1["wall_sec"],
        "phase2_wall_sec": p2["wall_sec"],
    }


def train_force_seq_long(geo, task, *, seed: int) -> dict:
    """Optional control: longer force-seq phase1 (N1+N2) then freeze π (N2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = geo.basic_t
    den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
    pol = Policy()
    r_t = torch.tensor(task.r, dtype=torch.float32)
    n1_long = N1 + N2  # 1600 — matched to Rec total before π
    p1 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=True,
        learn_pol=False,
        force_seq=True,
        steps=n1_long,
        lr=LR,
        tau=GUMBEL_TAU,
        phase_name="phase1_force_seq_long",
    )
    rec1 = _catalogue_recovery(den, geo, task)
    p2 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=False,
        learn_pol=True,
        force_seq=False,
        steps=N2,
        lr=LR,
        tau=GUMBEL_TAU,
        phase_name="phase2_unlock_pi_freeze_eps",
    )
    rec2 = _catalogue_recovery(den, geo, task)
    g2 = _graph_stats(geo, rec2["den_np"], task)
    ceil = ceiling_both_splits(
        geo,
        rec2["den_np"],
        rec2["catalogue_idx"],
        task,
        pol,
        force_seq=False,
        has_trained_policy=True,
    )
    metrics = exact_eval(
        geo,
        rec2["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )
    torch.save(
        {
            "arm": "ForceSeq_long",
            "seed": seed,
            "den": den.state_dict(),
            "pol": pol.state_dict(),
            "catalogue_idx": rec2["catalogue_idx"],
        },
        ART / "checkpoints" / f"seed{seed}_ForceSeq_long.pt",
    )
    return {
        "arm": "ForceSeq_long",
        "teacher": "force_seq_proj_loss_long",
        "teacher_detail": {
            "kind": "force_seq_long_warmstart",
            "uses_true_idx": False,
            "n1_long": n1_long,
        },
        "fence_notes": [
            "ForceSeq_long: phase1 force-seq for N1+N2=1600 then freeze π @ N2 "
            "(budget control vs Rec_* 800+800)"
        ],
        "curriculum": {
            "n1_phase1_force_seq_eps": n1_long,
            "n2_phase2_unlock_pi_freeze_eps": N2,
            "total_steps": n1_long + N2,
            "freeze_eps_phase2": True,
        },
        "steps": n1_long + N2,
        "lr": LR,
        "exact": metrics,
        "ceiling": ceil,
        "catalogue_idx_after": rec2["catalogue_idx"],
        "catalogue_idx_exact_match_to_true": rec2["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true": rec2["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec2["unique_catalogue_indices"],
        "adm_edge_jaccard_with_true": g2["adm_edge_jaccard_with_true"],
        "adm_edge_count_after": g2["adm_edge_count"],
        "true_adm_edge_count": g2["true_adm_edge_count"],
        "adm_edge_recall_vs_true": g2["adm_edge_recall_vs_true"],
        "wall_sec": p1["wall_sec"] + p2["wall_sec"],
        "catalogue_idx_exact_match_to_true_phase1": rec1[
            "catalogue_idx_exact_match_to_true"
        ],
    }


def train_B(geo, task, *, seed: int) -> dict:
    arm = train_arm_discrete(
        "B",
        geo,
        task,
        learn_den=False,
        learn_pol=True,
        force_seq=False,
        destroy_adm=False,
        freeze_to_true=True,
        seed=seed,
        steps=STEPS_DISC,
        lr=LR,
    )
    # Rebuild frozen true den + policy for ceiling
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = geo.basic_t
    den = FrozenTrueCatalogueDenotation(task.true_idx, cat_t)
    pol = Policy()
    r_t = torch.tensor(task.r, dtype=torch.float32)
    # learn π only
    for p in den.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)
    M_t = geo.M_t
    t0 = time.time()
    pol.train()
    for step in range(STEPS_DISC):
        E = den.forward_hard()
        a, b = E[idx[:, 0]], E[idx[:, 1]]
        soft = geo.soft_cyc_score(a, b)
        logits = pol(a, b, r_t)
        pred, _, _ = soft_program(a, b, r_t, logits, soft, M_t, force_seq=False)
        loss = proj_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    wall = time.time() - t0
    rec = _catalogue_recovery(den, geo, task)
    g = _graph_stats(geo, rec["den_np"], task)
    ceil = ceiling_both_splits(
        geo,
        rec["den_np"],
        rec["catalogue_idx"],
        task,
        pol,
        force_seq=False,
        has_trained_policy=True,
    )
    metrics = exact_eval(
        geo,
        rec["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )
    torch.save(
        {
            "arm": "B",
            "seed": seed,
            "den": den.state_dict(),
            "pol": pol.state_dict(),
            "catalogue_idx": rec["catalogue_idx"],
        },
        ART / "checkpoints" / f"seed{seed}_B.pt",
    )
    return {
        "arm": "B",
        "teacher": "true_epsilon_star_frozen",
        "teacher_detail": {
            "kind": "freeze_true_den_learn_pol",
            "uses_true_idx": True,
            "uses_heldout_targets": False,
        },
        "fence_notes": ["B = true ε* frozen; learn π only (sanity / upper geometry)"],
        "curriculum": {
            "n1_phase1": 0,
            "n2_phase2_learn_pi": STEPS_DISC,
            "total_steps": STEPS_DISC,
            "freeze_eps_phase2": True,
        },
        "steps": STEPS_DISC,
        "lr": LR,
        "exact": metrics,
        "ceiling": ceil,
        "catalogue_idx_after": rec["catalogue_idx"],
        "catalogue_idx_exact_match_to_true": rec["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true": rec["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec["unique_catalogue_indices"],
        "adm_edge_jaccard_with_true": g["adm_edge_jaccard_with_true"],
        "adm_edge_count_after": g["adm_edge_count"],
        "true_adm_edge_count": g["true_adm_edge_count"],
        "adm_edge_recall_vs_true": g["adm_edge_recall_vs_true"],
        "wall_sec": wall,
        "legacy_arm_exact_test": arm["exact"]["test"]["exact_success"],
    }


def slim_arm(arm: dict) -> dict:
    keys = [
        "arm",
        "teacher",
        "teacher_detail",
        "fence_notes",
        "curriculum",
        "steps",
        "lr",
        "wall_sec",
        "catalogue_idx_exact_match_to_true",
        "catalogue_ray_match_to_true",
        "unique_catalogue_indices",
        "adm_edge_jaccard_with_true",
        "adm_edge_count_after",
        "true_adm_edge_count",
        "adm_edge_recall_vs_true",
        "catalogue_idx_exact_match_to_true_phase1",
        "idx_frozen_phase1_to_phase2",
    ]
    out = {k: arm[k] for k in keys if k in arm}
    if "ceiling" in arm:
        out["ceiling_test"] = {
            m: arm["ceiling"]["test"][m]
            for m in ("H", "A", "S", "G", "one_minus_H", "H_minus_A", "four_way_counts")
        }
        out["ceiling_train"] = {
            m: arm["ceiling"]["train"][m] for m in ("H", "A", "S", "G")
        }
    if "exact" in arm:
        out["exact_train"] = arm["exact"]["train"]["exact_success"]
        out["exact_test"] = arm["exact"]["test"]["exact_success"]
    # drop bulky vote_meta lists in slim teacher_detail
    if "teacher_detail" in out and isinstance(out["teacher_detail"], dict):
        td = dict(out["teacher_detail"])
        if "vote_meta" in td and isinstance(td["vote_meta"], dict):
            vm = {
                k: td["vote_meta"][k]
                for k in td["vote_meta"]
                if k != "vote_argmax"
            }
            td["vote_meta"] = vm
        out["teacher_detail"] = td
    return out


def main() -> int:
    (ART / "checkpoints").mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    alg = SedenionAlgebra()
    geo = Geometry(alg)
    wit = verify_witness(alg, geo.M)
    print("witness_disagree", wit["disagree"], "matches_01707", wit["matches_01707_expected"])
    assert wit["disagree"]

    true_idx = densest_vocab_indices(geo, N_VOCAB)
    print("vocab_idx", true_idx.tolist())
    print("within_vocab_fips_edges", int(geo.fips_adj[np.ix_(true_idx, true_idx)].sum()))

    # Program maps depend only on r=e4 (shared across seeds)
    task0 = build_native_task(geo, true_idx, seed=0)
    t_maps = time.time()
    program_maps = precompute_program_maps(geo, task0.r)
    print(
        f"program_maps sec={time.time()-t_maps:.2f} "
        f"seq_keys={len(program_maps['seq'])} grp_keys={len(program_maps['grp'])}"
    )

    all_seed_results = []
    for seed in SEEDS:
        task = build_native_task(geo, true_idx, seed=seed)
        print(
            f"seed={seed} train={len(task.pairs_train)} test={len(task.pairs_test)} "
            f"true_adm={task.vocab_stats['true_adm_pairs_all']}"
        )
        arms = []
        arms.append(
            train_rec_arm(
                "Rec_CE",
                geo,
                task,
                teacher="true_ce",
                program_maps=None,
                seed=seed,
            )
        )
        print(
            f"  Rec_CE idx={arms[-1]['catalogue_idx_exact_match_to_true']}/16 "
            f"H={arms[-1]['ceiling']['test']['H']:.3f} "
            f"A={arms[-1]['ceiling']['test']['A']:.3f}"
        )
        arms.append(
            train_rec_arm(
                "Rec_ray",
                geo,
                task,
                teacher="ray_vote_ce",
                program_maps=program_maps,
                seed=seed,
            )
        )
        print(
            f"  Rec_ray idx={arms[-1]['catalogue_idx_exact_match_to_true']}/16 "
            f"vote_match={arms[-1]['teacher_detail']['vote_meta']['vote_argmax_match_to_true']}/16 "
            f"H={arms[-1]['ceiling']['test']['H']:.3f} "
            f"A={arms[-1]['ceiling']['test']['A']:.3f}"
        )
        arms.append(train_baseline_freeze(geo, task, seed=seed))
        print(
            f"  Baseline_freeze idx={arms[-1]['catalogue_idx_exact_match_to_true']}/16 "
            f"H={arms[-1]['ceiling']['test']['H']:.3f} "
            f"A={arms[-1]['ceiling']['test']['A']:.3f}"
        )
        arms.append(train_B(geo, task, seed=seed))
        print(
            f"  B idx={arms[-1]['catalogue_idx_exact_match_to_true']}/16 "
            f"H={arms[-1]['ceiling']['test']['H']:.3f} "
            f"A={arms[-1]['ceiling']['test']['A']:.3f}"
        )
        arms.append(train_force_seq_long(geo, task, seed=seed))
        print(
            f"  ForceSeq_long idx={arms[-1]['catalogue_idx_exact_match_to_true']}/16 "
            f"H={arms[-1]['ceiling']['test']['H']:.3f} "
            f"A={arms[-1]['ceiling']['test']['A']:.3f}"
        )
        oracle = oracle_eval(geo, task)
        all_seed_results.append(
            {
                "seed": seed,
                "vocab_stats": task.vocab_stats,
                "oracle": oracle,
                "arms": arms,
            }
        )

    # Aggregate
    arm_names = ["Rec_CE", "Rec_ray", "Baseline_freeze", "B", "ForceSeq_long"]
    summary = {}
    for name in arm_names:
        H_te, A_te, S_te, G_te = [], [], [], []
        idx_m, jacc, exact_te = [], [], []
        for s in all_seed_results:
            a = next(x for x in s["arms"] if x["arm"] == name)
            ct = a["ceiling"]["test"]
            H_te.append(ct["H"])
            A_te.append(ct["A"])
            S_te.append(ct["S"])
            G_te.append(ct["G"])
            idx_m.append(a["catalogue_idx_exact_match_to_true"])
            jacc.append(a["adm_edge_jaccard_with_true"])
            exact_te.append(a["exact"]["test"]["exact_success"])
        summary[name] = {
            "test_H": mean_std(H_te),
            "test_A": mean_std(A_te),
            "test_S": mean_std(S_te),
            "test_G": mean_std(G_te),
            "catalogue_idx_match_over_16": mean_std([float(x) for x in idx_m]),
            "adm_edge_jaccard_with_true": mean_std(jacc),
            "exact_test": mean_std(exact_te),
        }

    # ΔH vs Baseline_freeze
    base_H = summary["Baseline_freeze"]["test_H"]["values"]
    delta_H = {}
    for name in arm_names:
        if name == "Baseline_freeze":
            continue
        vals = summary[name]["test_H"]["values"]
        d = [v - b for v, b in zip(vals, base_H)]
        delta_H[name] = mean_std(d)

    # Verdict
    rec_ray_H = summary["Rec_ray"]["test_H"]["mean"]
    rec_ce_H = summary["Rec_CE"]["test_H"]["mean"]
    base_Hm = summary["Baseline_freeze"]["test_H"]["mean"]
    b_H = summary["B"]["test_H"]["mean"]
    material = 0.10  # absolute H lift threshold vs baseline toward B
    raised = max(rec_ray_H, rec_ce_H) - base_Hm
    toward_B = (max(rec_ray_H, rec_ce_H) - base_Hm) / max(b_H - base_Hm, 1e-9)
    if raised >= material and max(rec_ray_H, rec_ce_H) >= 0.8 * b_H:
        verdict = (
            "YES — catalogue recovery raised H materially toward B "
            f"(ΔH≈{raised:.3f}; fraction of gap closed≈{toward_B:.2f})"
        )
    elif raised >= material:
        verdict = (
            "PARTIAL — H rose materially vs Baseline_freeze "
            f"(ΔH≈{raised:.3f}) but remains below near-B"
        )
    else:
        verdict = (
            "NO — catalogue recovery did not raise H materially vs Baseline_freeze "
            f"(ΔH≈{raised:.3f})"
        )

    report = {
        "source": "017.17 catalogue-recovery / representation lift under freeze",
        "in_reply_to": "017.16-GrokBot-recovered-geometry-ceiling-and-policy-regret-result.md §6",
        "implementation": {
            "mul": "structure constants from topographo.ssd.sedenion.SedenionAlgebra.mul",
            "evaluator_exact": "existing float projective_key / rays_equal (legacy exact_eval)",
            "evaluator_ceiling": "017.15-style float H/A/S/G (all examples in denominator)",
            "host": "thebeast",
            "device": "cpu",
            "N1": N1,
            "N2": N2,
            "lr": LR,
            "gumbel_tau": GUMBEL_TAU,
            "seeds": list(SEEDS),
            "std_convention": "population std (np.std ddof=0), seed-averaged means",
            "arms": arm_names,
            "Rec_CE_fence": (
                "uses TRUE ε* catalogue indices as phase-1 CE teacher (diagnostic); "
                "not held-out pair targets"
            ),
            "Rec_ray_supervision": (
                "train-set only P_seq program-inverse catalogue votes → hard CE; "
                "no true_idx; no held-out targets; no preferred-tree labels"
            ),
        },
        "witness_check": wit,
        "task": {
            "n_vocab": N_VOCAB,
            "holdout": HOLDOUT,
            "r": "e4",
            "seeds": list(SEEDS),
            "fixed_true_idx": true_idx.tolist(),
            "within_vocab_fips_edges": int(
                geo.fips_adj[np.ix_(true_idx, true_idx)].sum()
            ),
        },
        "summary": summary,
        "delta_H_vs_Baseline_freeze": delta_H,
        "verdict": verdict,
        "seed_results": all_seed_results,
        "fence": (
            "configured experiment on finite FIPS proxy; catalogue-recovery under freeze; "
            "Rec_CE is true-index teacher diagnostic; Issue 017 not closed; "
            "no modular grokking claim"
        ),
        "wall_sec_total": time.time() - t_all,
    }

    full_path = OUT / "learned_admissibility_01717_report.json"
    # Full report may be large due to traces — strip train_trace from arms
    for s in report["seed_results"]:
        for a in s["arms"]:
            a.pop("train_trace", None)
            if "phase1_training" in a and isinstance(a["phase1_training"], dict):
                a["phase1_training"].pop("train_trace", None)
            # vote_argmax list is small OK
    full_path.write_text(json.dumps(report, indent=2) + "\n")

    slim_seeds = []
    for s in all_seed_results:
        slim_seeds.append(
            {
                "seed": s["seed"],
                "oracle_test": s["oracle"]["exact"]["test"]["exact_success"],
                "arms": [slim_arm(a) for a in s["arms"]],
            }
        )
    slim = {
        "source": report["source"],
        "in_reply_to": report["in_reply_to"],
        "implementation": report["implementation"],
        "witness_check": {
            "disagree": wit["disagree"],
            "matches_01707_expected": wit["matches_01707_expected"],
        },
        "task": report["task"],
        "summary": summary,
        "delta_H_vs_Baseline_freeze": delta_H,
        "verdict": verdict,
        "seed_results_slim": slim_seeds,
        "fence": report["fence"],
        "wall_sec_total": report["wall_sec_total"],
    }
    slim_path = OUT / "learned_admissibility_01717_report_slim.json"
    slim_path.write_text(json.dumps(slim, indent=2) + "\n")

    tiny = {
        "source": report["source"],
        "verdict": verdict,
        "test_mean_std_HA_idx": {
            name: {
                "H": {"mean": summary[name]["test_H"]["mean"], "std": summary[name]["test_H"]["std"]},
                "A": {"mean": summary[name]["test_A"]["mean"], "std": summary[name]["test_A"]["std"]},
                "idx_match_over_16": {
                    "mean": summary[name]["catalogue_idx_match_over_16"]["mean"],
                    "std": summary[name]["catalogue_idx_match_over_16"]["std"],
                },
            }
            for name in arm_names
        },
        "delta_H_vs_Baseline_freeze": {
            k: {"mean": v["mean"], "std": v["std"]} for k, v in delta_H.items()
        },
        "fence": report["fence"],
    }
    tiny_path = OUT / "learned_admissibility_01717_report_tiny.json"
    tiny_path.write_text(json.dumps(tiny, indent=2) + "\n")

    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("DELTA_H", json.dumps(delta_H, indent=2))
    print("VERDICT", verdict)
    print("wrote", full_path)
    print("wrote", slim_path)
    print("wrote", tiny_path)
    print("wall_sec_total", report["wall_sec_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

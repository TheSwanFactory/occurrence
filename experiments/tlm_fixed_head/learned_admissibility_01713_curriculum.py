#!/usr/bin/env python3
"""017.13 discrete curriculum / two-phase training (responds to 017.12 §10.1).

Question
--------
Does warm-starting discrete ε under force_seq (like arm C), then unlocking π
under the soft adm-gated mixture, stabilize primary arm A so held-out exact
success rises materially above ~0.035?

Phase protocol
--------------
Budget choice: **800 + 800** (N1 phase-1 force-seq ε, N2 phase-2 unlock π)
of a 1600-step curriculum budget. Matched single-phase baselines use 800
steps (identical to 017.12), so curriculum gets more wall steps by design —
document and compare honestly against A_joint@800 and C@800.

Arms
----
- **A_cur**: phase1 force-seq ε → phase2 joint ε+π
- **A_cur_freeze**: phase1 force-seq ε → phase2 π only (ε frozen)
- **A_joint**: single-phase A (control; should ~match 017.12)
- **B**, **C**, **Oracle**: matched baselines

Reuses DiscreteCatalogueDenotation / train_arm_discrete from
``learned_admissibility_01711_discrete.py`` and Geometry/task/Policy helpers
from ``learned_admissibility_01709.py``.

Fence: configured experiment; not physical OT / modular grokking.
Issue 017 not closed.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from learned_admissibility_01709 import (
    HOLDOUT,
    N_VOCAB,
    SEEDS,
    TOL,
    Geometry,
    Policy,
    SedenionAlgebra,
    admissibility_graph,
    aggregate,
    build_native_task,
    count_all_params,
    count_params,
    densest_vocab_indices,
    denotation_collapse,
    exact_eval,
    oracle_eval,
    proj_loss,
    soft_program,
    true_denotations,
    verify_witness,
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

OUT = Path(__file__).resolve().parent
# Curriculum budget: 800 warm-start + 800 unlock (documented choice)
N1_PHASE1 = 800
N2_PHASE2 = 800
LR_CUR = LR_DISC


def _graph_stats(geo: Geometry, den_np: np.ndarray, task, graph_before=None) -> dict:
    graph_after = admissibility_graph(geo, den_np)
    true_graph = geo.fips_adj[np.ix_(task.true_idx, task.true_idx)]
    after_edges = set(zip(*np.where(graph_after > 0)))
    true_edges = set(zip(*np.where(true_graph > 0)))
    inter = after_edges & true_edges
    union = after_edges | true_edges
    jaccard = len(inter) / max(len(union), 1)
    precision = len(inter) / max(len(after_edges), 1)
    recall = len(inter) / max(len(true_edges), 1)
    disagree_pairs = {p for p, v in task.true_disagree.items() if v}
    out = {
        "adm_edge_count": int(graph_after.sum()),
        "true_adm_edge_count": int(true_graph.sum()),
        "adm_edge_jaccard_with_true": float(jaccard),
        "adm_edge_precision_vs_true": float(precision),
        "adm_edge_recall_vs_true": float(recall),
        "task_relevant_disagree_edges_in_learned_graph": int(
            len(after_edges & disagree_pairs)
        ),
        "hard_adm_frac": float(graph_after.mean()),
    }
    if graph_before is not None:
        before_edges = set(zip(*np.where(graph_before > 0)))
        out["adm_edges_created"] = int(len(after_edges - before_edges))
        out["adm_edges_destroyed"] = int(len(before_edges - after_edges))
        out["adm_edge_count_before"] = int(graph_before.sum())
    return out


def _catalogue_recovery(den: nn.Module, geo: Geometry, task) -> dict:
    den_np = den.forward_hard().detach().numpy()
    idx_after = den.selected_indices().tolist()
    true_idx_list = task.true_idx.tolist()
    n_exact_idx = sum(1 for a, b in zip(idx_after, true_idx_list) if a == b)
    true_den = true_denotations(geo, task)
    n_ray_match = sum(
        1
        for i in range(task.n_vocab)
        if abs(np.dot(den_np[i], true_den[i]))
        > 0.99 * (np.linalg.norm(den_np[i]) * np.linalg.norm(true_den[i]) + 1e-12)
    )
    return {
        "catalogue_idx": idx_after,
        "catalogue_idx_exact_match_to_true": int(n_exact_idx),
        "catalogue_ray_match_to_true": int(n_ray_match),
        "unique_catalogue_indices": len(set(idx_after)),
        "denotation_collapse": denotation_collapse(den_np),
        "den_np": den_np,
    }


def _run_phase(
    *,
    den: nn.Module,
    pol: Policy,
    geo: Geometry,
    task,
    learn_den: bool,
    learn_pol: bool,
    force_seq: bool,
    steps: int,
    lr: float,
    tau: float,
    phase_name: str,
) -> dict:
    """One training phase; mutates den/pol in place."""
    params = []
    if learn_den:
        for p in den.parameters():
            p.requires_grad_(True)
        params += [p for p in den.parameters() if p.requires_grad]
    else:
        for p in den.parameters():
            p.requires_grad_(False)
    if learn_pol and not force_seq:
        for p in pol.parameters():
            p.requires_grad_(True)
        params += [p for p in pol.parameters() if p.requires_grad]
    else:
        for p in pol.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(params, lr=lr) if params else None
    r_t = torch.tensor(task.r, dtype=torch.float32)
    M_t = geo.M_t
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)

    hist = []
    t0 = time.time()
    loss = torch.tensor(0.0)
    den.train()
    for step in range(steps):
        E = den(hard=True, tau=tau)
        a = E[idx[:, 0]]
        b = E[idx[:, 1]]
        soft = geo.soft_cyc_score(a, b)
        if learn_pol and not force_seq:
            logits = pol(a, b, r_t)
        else:
            logits = torch.zeros(a.shape[0], 2)
        pred, w_grp, adm = soft_program(
            a, b, r_t, logits, soft, M_t, force_seq=force_seq
        )
        loss = proj_loss(pred, tgt)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                hard_E = den.forward_hard()
                hard_adm_now = float(admissibility_graph(geo, hard_E.numpy()).mean())
            hist.append(
                {
                    "step": step,
                    "phase": phase_name,
                    "loss": float(loss.detach()),
                    "mean_soft_adm": float(adm.detach().mean()),
                    "mean_w_grp": float(w_grp.detach().mean()),
                    "hard_adm_frac_checkpoint": hard_adm_now,
                }
            )

    den.eval()
    with torch.no_grad():
        E_f = den.forward_hard()
        a_f, b_f = E_f[idx[:, 0]], E_f[idx[:, 1]]
        soft_f = geo.soft_cyc_score(a_f, b_f)
        logits_f = (
            pol(a_f, b_f, r_t)
            if (learn_pol and not force_seq)
            else torch.zeros(a_f.shape[0], 2)
        )
        pred_f, w_grp_f, _ = soft_program(
            a_f, b_f, r_t, logits_f, soft_f, M_t, force_seq=force_seq
        )
        pn = torch.linalg.vector_norm(pred_f, dim=-1, keepdim=True).clamp_min(1e-8)
        tn = torch.linalg.vector_norm(tgt, dim=-1, keepdim=True).clamp_min(1e-8)
        soft_cos = float(((pred_f / pn) * (tgt / tn)).sum(dim=-1).abs().mean())
        mean_w_grp_final = float(w_grp_f.mean())

    return {
        "phase": phase_name,
        "steps": steps,
        "learn_den": learn_den,
        "learn_pol": learn_pol and not force_seq,
        "force_seq": force_seq,
        "param_count_trainable": count_params([den, pol]),
        "train_loss_surrogate_final": float(loss.detach()),
        "train_mean_abs_cosine_surrogate": soft_cos,
        "mean_w_grp_final": mean_w_grp_final,
        "wall_sec": time.time() - t0,
        "train_trace": hist,
    }


def _policy_fn_factory(pol: Policy, geo: Geometry, r_t: torch.Tensor, force_seq: bool):
    def policy_fn(i, j, a, b, hard_adm):
        if force_seq or not hard_adm:
            return "seq"
        with torch.no_grad():
            aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            logits = pol(aa, bb, r_t)
            soft = geo.soft_cyc_score(aa, bb)
            masked = logits.clone()
            masked[0, 1] = masked[0, 1] + torch.log(soft.clamp(1e-4, 1.0))
            return "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"

    return policy_fn


def train_arm_curriculum(
    name: str,
    geo: Geometry,
    task,
    *,
    freeze_eps_phase2: bool,
    seed: int = 0,
    n1: int = N1_PHASE1,
    n2: int = N2_PHASE2,
    lr: float = LR_CUR,
    tau: float = GUMBEL_TAU,
) -> dict:
    """Two-phase curriculum: force-seq ε warm-start → unlock π (± freeze ε)."""
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

    # ---- Phase 1: force_seq, learn ε only ----
    p1 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=True,
        learn_pol=False,
        force_seq=True,
        steps=n1,
        lr=lr,
        tau=tau,
        phase_name="phase1_force_seq_eps",
    )
    rec1 = _catalogue_recovery(den, geo, task)
    g1 = _graph_stats(geo, rec1["den_np"], task, graph_before=graph_before)
    # Exact under force_seq after phase 1 (geometry teacher signal)
    metrics_p1_force = exact_eval(
        geo,
        rec1["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=True),
        task,
        force_seq=True,
    )
    # Also probe what soft-mixture policy would do with frozen random π (diagnostic)
    phase1_snapshot = {
        "exact_force_seq": metrics_p1_force,
        "adm": {k: v for k, v in g1.items()},
        "catalogue_idx_exact_match_to_true": rec1["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true": rec1["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec1["unique_catalogue_indices"],
        "catalogue_idx": rec1["catalogue_idx"],
        "train_loss_surrogate_final": p1["train_loss_surrogate_final"],
        "train_mean_abs_cosine_surrogate": p1["train_mean_abs_cosine_surrogate"],
        "wall_sec": p1["wall_sec"],
        "param_count_trainable": p1["param_count_trainable"],
    }

    # ---- Phase 2: unlock π; optionally freeze ε ----
    p2 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=not freeze_eps_phase2,
        learn_pol=True,
        force_seq=False,
        steps=n2,
        lr=lr,
        tau=tau,
        phase_name="phase2_unlock_pi"
        + ("_freeze_eps" if freeze_eps_phase2 else "_joint"),
    )
    rec2 = _catalogue_recovery(den, geo, task)
    g2 = _graph_stats(geo, rec2["den_np"], task, graph_before=graph_before)
    # Graph change phase1 → phase2
    g_p1_to_p2 = _graph_stats(
        geo, rec2["den_np"], task, graph_before=admissibility_graph(geo, rec1["den_np"])
    )

    metrics = exact_eval(
        geo,
        rec2["den_np"],
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )

    # Mid-phase-2 branch frequency from train_trace mean_w_grp
    w_grp_trace = [h["mean_w_grp"] for h in p2["train_trace"]]
    branch_diag = {
        "phase2_mean_w_grp_final": p2["mean_w_grp_final"],
        "phase2_mean_w_grp_trace_mean": float(np.mean(w_grp_trace)) if w_grp_trace else 0.0,
        "phase2_exact_grp_choice_fraction_train": metrics["train"]["grp_choice_fraction"],
        "phase2_exact_grp_choice_fraction_test": metrics["test"]["grp_choice_fraction"],
        "phase2_exact_seq_choice_fraction_train": metrics["train"]["seq_choice_fraction"],
        "phase2_exact_seq_choice_fraction_test": metrics["test"]["seq_choice_fraction"],
    }

    return {
        "arm": name,
        "denotation_head": "discrete_catalogue_84",
        "curriculum": {
            "n1_phase1_force_seq_eps": n1,
            "n2_phase2_unlock_pi": n2,
            "total_steps": n1 + n2,
            "freeze_eps_phase2": freeze_eps_phase2,
            "budget_note": (
                "800+800 curriculum budget; single-phase baselines remain at 800 "
                "(matched to 017.12). Curriculum has more wall steps by design."
            ),
        },
        "relaxation": (
            "train: Gumbel-Softmax hard=True (STE) over 84 catalogue; "
            "eval: argmax catalogue vector"
        ),
        "param_count_trainable_final": count_params([den, pol]),
        "param_count_total_modules": count_all_params([den, pol]),
        "learn_den_final": not freeze_eps_phase2,
        "learn_pol_final": True,
        "force_seq_final": False,
        "freeze_eps_phase2": freeze_eps_phase2,
        "steps": n1 + n2,
        "lr": lr,
        "gumbel_tau": tau,
        "train_loss_surrogate_final": p2["train_loss_surrogate_final"],
        "train_mean_abs_cosine_surrogate": p2["train_mean_abs_cosine_surrogate"],
        "hard_adm_frac_learned_denotations": g2["hard_adm_frac"],
        "wall_sec": p1["wall_sec"] + p2["wall_sec"],
        "exact": metrics,
        "generalization_gap": metrics["train"]["exact_success"]
        - metrics["test"]["exact_success"],
        "admissibility_overlap_with_true_elementwise": float(
            (
                admissibility_graph(geo, rec2["den_np"])
                == geo.fips_adj[np.ix_(task.true_idx, task.true_idx)]
            ).mean()
        ),
        "adm_edge_count_before": int(graph_before.sum()),
        "adm_edge_count_after_phase1": g1["adm_edge_count"],
        "adm_edge_count_after": g2["adm_edge_count"],
        "true_adm_edge_count": g2["true_adm_edge_count"],
        "adm_edge_jaccard_with_true_phase1": g1["adm_edge_jaccard_with_true"],
        "adm_edge_jaccard_with_true": g2["adm_edge_jaccard_with_true"],
        "adm_edge_precision_vs_true": g2["adm_edge_precision_vs_true"],
        "adm_edge_recall_vs_true": g2["adm_edge_recall_vs_true"],
        "adm_edge_recall_vs_true_phase1": g1["adm_edge_recall_vs_true"],
        "task_relevant_disagree_edges_in_learned_graph": g2[
            "task_relevant_disagree_edges_in_learned_graph"
        ],
        "task_relevant_disagree_edges_phase1": g1[
            "task_relevant_disagree_edges_in_learned_graph"
        ],
        "adm_edges_created": g2.get("adm_edges_created", 0),
        "adm_edges_destroyed": g2.get("adm_edges_destroyed", 0),
        "adm_edges_changed_phase1_to_phase2_created": g_p1_to_p2.get(
            "adm_edges_created", 0
        ),
        "adm_edges_changed_phase1_to_phase2_destroyed": g_p1_to_p2.get(
            "adm_edges_destroyed", 0
        ),
        "catalogue_idx_before": idx_before,
        "catalogue_idx_after_phase1": rec1["catalogue_idx"],
        "catalogue_idx_after": rec2["catalogue_idx"],
        "catalogue_idx_exact_match_to_true_phase1": rec1[
            "catalogue_idx_exact_match_to_true"
        ],
        "catalogue_idx_exact_match_to_true": rec2["catalogue_idx_exact_match_to_true"],
        "catalogue_ray_match_to_true_phase1": rec1["catalogue_ray_match_to_true"],
        "catalogue_ray_match_to_true": rec2["catalogue_ray_match_to_true"],
        "unique_catalogue_indices": rec2["unique_catalogue_indices"],
        "denotation_collapse": rec2["denotation_collapse"],
        "phase1": phase1_snapshot,
        "phase2_training": {
            "train_loss_surrogate_final": p2["train_loss_surrogate_final"],
            "train_mean_abs_cosine_surrogate": p2["train_mean_abs_cosine_surrogate"],
            "param_count_trainable": p2["param_count_trainable"],
            "wall_sec": p2["wall_sec"],
        },
        "branch_frequencies_phase2": branch_diag,
        "train_trace": p1["train_trace"] + p2["train_trace"],
    }


def slim_arm(arm: dict) -> dict:
    """Drop bulky traces / full idx lists for slim report."""
    keys_keep = [
        "arm",
        "param_count_trainable",
        "param_count_trainable_final",
        "steps",
        "lr",
        "freeze_eps_phase2",
        "curriculum",
        "exact",
        "generalization_gap",
        "train_mean_abs_cosine_surrogate",
        "hard_adm_frac_learned_denotations",
        "adm_edge_count_before",
        "adm_edge_count_after_phase1",
        "adm_edge_count_after",
        "true_adm_edge_count",
        "adm_edge_jaccard_with_true_phase1",
        "adm_edge_jaccard_with_true",
        "adm_edge_recall_vs_true",
        "adm_edge_recall_vs_true_phase1",
        "task_relevant_disagree_edges_in_learned_graph",
        "task_relevant_disagree_edges_phase1",
        "catalogue_idx_exact_match_to_true_phase1",
        "catalogue_idx_exact_match_to_true",
        "catalogue_ray_match_to_true",
        "unique_catalogue_indices",
        "branch_frequencies_phase2",
        "wall_sec",
        "learn_den",
        "learn_pol",
        "force_seq",
        "freeze_to_true",
    ]
    out = {k: arm[k] for k in keys_keep if k in arm}
    # Flatten exact for readability in seed_results_slim
    if "exact" in arm:
        out["exact_train"] = arm["exact"]["train"]["exact_success"]
        out["exact_test"] = arm["exact"]["test"]["exact_success"]
        if "grp_choice_fraction" in arm["exact"]["train"]:
            out["grp_choice_fraction_train"] = arm["exact"]["train"][
                "grp_choice_fraction"
            ]
            out["grp_choice_fraction_test"] = arm["exact"]["test"]["grp_choice_fraction"]
    return out


def main() -> int:
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    wit = verify_witness(alg, geo.M)
    print("witness_disagree", wit["disagree"], "matches_01707", wit["matches_01707_expected"])
    assert wit["disagree"], "017.07 witness must disagree under SedenionAlgebra.mul"

    true_idx = densest_vocab_indices(geo, N_VOCAB)
    print("vocab_idx", true_idx.tolist())
    print("within_vocab_fips_edges", int(geo.fips_adj[np.ix_(true_idx, true_idx)].sum()))
    print(
        f"curriculum budget: N1={N1_PHASE1} force-seq ε + N2={N2_PHASE2} unlock π "
        f"(total {N1_PHASE1 + N2_PHASE2}); single-phase baselines @ {STEPS_DISC}"
    )

    all_seed_results = []
    for seed in SEEDS:
        task = build_native_task(geo, true_idx, seed=seed)
        print(
            f"seed={seed} train={len(task.pairs_train)} test={len(task.pairs_test)} "
            f"true_adm={task.vocab_stats['true_adm_pairs_all']} "
            f"disagree={task.vocab_stats['true_disagree_pairs_all']}"
        )
        assert task.vocab_stats["true_adm_pairs_all"] >= 20

        arms = []
        # A_cur: phase1 force-seq ε → phase2 joint ε+π
        arms.append(
            train_arm_curriculum(
                "A_cur",
                geo,
                task,
                freeze_eps_phase2=False,
                seed=seed,
            )
        )
        # A_cur_freeze: phase1 force-seq ε → phase2 π only (ε frozen)
        arms.append(
            train_arm_curriculum(
                "A_cur_freeze",
                geo,
                task,
                freeze_eps_phase2=True,
                seed=seed,
            )
        )
        # A_joint: single-phase A control (match 017.12)
        arms.append(
            train_arm_discrete(
                "A_joint",
                geo,
                task,
                learn_den=True,
                learn_pol=True,
                force_seq=False,
                destroy_adm=False,
                seed=seed,
            )
        )
        # B: freeze true ε*, learn π
        arms.append(
            train_arm_discrete(
                "B_freeze_true_den_learn_pol",
                geo,
                task,
                learn_den=False,
                learn_pol=True,
                force_seq=False,
                destroy_adm=False,
                freeze_to_true=True,
                seed=seed,
            )
        )
        # C: learn discrete ε, force seq
        arms.append(
            train_arm_discrete(
                "C_learn_discrete_den_force_seq",
                geo,
                task,
                learn_den=True,
                learn_pol=False,
                force_seq=True,
                destroy_adm=False,
                seed=seed,
            )
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
        print(
            f"  oracle train={oracle['exact']['train']['exact_success']:.3f} "
            f"test={oracle['exact']['test']['exact_success']:.3f}"
        )
        for a in arms:
            te = a["exact"]["test"]["exact_success"]
            tr = a["exact"]["train"]["exact_success"]
            j1 = a.get("adm_edge_jaccard_with_true_phase1")
            j2 = a.get("adm_edge_jaccard_with_true", a.get("adm_edge_jaccard_with_true"))
            e1 = a.get("adm_edge_count_after_phase1")
            e2 = a.get("adm_edge_count_after")
            extra = ""
            if j1 is not None:
                extra = (
                    f" jacc_p1={j1:.3f}→p2={j2:.3f} "
                    f"adm_p1={e1}→p2={e2} "
                    f"idx_p1={a.get('catalogue_idx_exact_match_to_true_phase1')}"
                    f"→p2={a.get('catalogue_idx_exact_match_to_true')}"
                )
            else:
                extra = (
                    f" adm {a.get('adm_edge_count_before')}→{e2} "
                    f"jacc={j2:.3f} idx={a.get('catalogue_idx_exact_match_to_true')}"
                )
            print(
                f"  {a['arm']}: train={tr:.3f} test={te:.3f} gap={a['generalization_gap']:.3f}"
                f"{extra}"
            )

    summary = aggregate(all_seed_results)
    ora_tr = [s["oracle"]["exact"]["train"]["exact_success"] for s in all_seed_results]
    ora_te = [s["oracle"]["exact"]["test"]["exact_success"] for s in all_seed_results]
    summary["oracle_true_eps_correct_branch"] = {
        "train_mean": float(np.mean(ora_tr)),
        "train_std": float(np.std(ora_tr)),
        "test_mean": float(np.mean(ora_te)),
        "test_std": float(np.std(ora_te)),
        "gap_mean": float(np.mean(np.asarray(ora_tr) - np.asarray(ora_te))),
        "n_seeds": len(ora_tr),
        "note": "diagnostic upper bound",
    }

    # Aggregate adm-graph + curriculum phase diagnostics
    adm_diag = {}
    for name in [a["arm"] for a in all_seed_results[0]["arms"]]:
        befores, afters, jacs, recalls, task_rel = [], [], [], [], []
        jacs_p1, afters_p1, idx_p1, idx_p2 = [], [], [], []
        grp_te = []
        for s in all_seed_results:
            arm = next(x for x in s["arms"] if x["arm"] == name)
            befores.append(arm["adm_edge_count_before"])
            afters.append(arm["adm_edge_count_after"])
            jacs.append(arm["adm_edge_jaccard_with_true"])
            recalls.append(arm["adm_edge_recall_vs_true"])
            task_rel.append(arm["task_relevant_disagree_edges_in_learned_graph"])
            if "adm_edge_jaccard_with_true_phase1" in arm:
                jacs_p1.append(arm["adm_edge_jaccard_with_true_phase1"])
                afters_p1.append(arm["adm_edge_count_after_phase1"])
                idx_p1.append(arm["catalogue_idx_exact_match_to_true_phase1"])
                idx_p2.append(arm["catalogue_idx_exact_match_to_true"])
            if "branch_frequencies_phase2" in arm:
                grp_te.append(
                    arm["branch_frequencies_phase2"][
                        "phase2_exact_grp_choice_fraction_test"
                    ]
                )
        entry = {
            "adm_edge_count_before_mean": float(np.mean(befores)),
            "adm_edge_count_after_mean": float(np.mean(afters)),
            "adm_edge_count_after_std": float(np.std(afters)),
            "jaccard_mean": float(np.mean(jacs)),
            "recall_mean": float(np.mean(recalls)),
            "task_relevant_disagree_edges_mean": float(np.mean(task_rel)),
        }
        if jacs_p1:
            entry["jaccard_phase1_mean"] = float(np.mean(jacs_p1))
            entry["adm_edge_count_after_phase1_mean"] = float(np.mean(afters_p1))
            entry["catalogue_idx_match_phase1_mean"] = float(np.mean(idx_p1))
            entry["catalogue_idx_match_phase2_mean"] = float(np.mean(idx_p2))
        if grp_te:
            entry["phase2_grp_choice_fraction_test_mean"] = float(np.mean(grp_te))
        adm_diag[name] = entry

    report = {
        "source": "017.13 discrete curriculum / two-phase training — responds to 017.12 §10.1",
        "in_reply_to": "017.12-GrokBot-discrete-catalogue-denotation-result.md",
        "implementation": {
            "mul": "structure constants from topographo.ssd.sedenion.SedenionAlgebra.mul",
            "helpers_reused": (
                "learned_admissibility_01709.py (Geometry, task, Policy, soft_program, exact_eval); "
                "learned_admissibility_01711_discrete.py (DiscreteCatalogueDenotation, train_arm_discrete)"
            ),
            "denotation_head": (
                "DiscreteCatalogueDenotation: logits (n_vocab × 84); "
                "train Gumbel-Softmax hard=True STE; eval argmax catalogue vector"
            ),
            "curriculum": {
                "N1_phase1": N1_PHASE1,
                "N2_phase2": N2_PHASE2,
                "budget_choice": "800+800 of 1600 curriculum budget (documented)",
                "A_cur": "phase1 force-seq ε → phase2 joint ε+π",
                "A_cur_freeze": "phase1 force-seq ε → phase2 π only (ε frozen)",
                "A_joint": "single-phase A @ 800 (017.12 control)",
            },
            "cyc_proxy": (
                "finite FIPS-style: zw!=0 and [zw] a basic-Event ray; "
                "NOT full continuous C"
            ),
            "soft_train_exact_eval": (
                "soft Cyc score during training; exact hard proxy + projective "
                "ray equality at evaluation"
            ),
            "vocab_selection": (
                f"same greedy densest FIPS subgraph as 017.10/017.12, n_vocab={N_VOCAB}"
            ),
            "arms": "A_cur / A_cur_freeze / A_joint / B / C + Oracle",
            "steps_curriculum": N1_PHASE1 + N2_PHASE2,
            "steps_baselines": STEPS_DISC,
            "lr": LR_CUR,
            "gumbel_tau": GUMBEL_TAU,
        },
        "witness_check": wit,
        "task": {
            "kind": "native-structure from fixed hidden OT-compatible denotations",
            "n_vocab": N_VOCAB,
            "holdout": HOLDOUT,
            "split": "over pair combinations, not repeats",
            "target_rule": (
                "grp when seq/grp disagree and grp defined; else seq "
                "(else grp if seq undefined)"
            ),
            "r_primary": "e4",
            "steps_curriculum": N1_PHASE1 + N2_PHASE2,
            "steps_baselines": STEPS_DISC,
            "lr": LR_CUR,
            "seeds": list(SEEDS),
            "fixed_true_idx": true_idx.tolist(),
            "within_vocab_fips_edges": int(
                geo.fips_adj[np.ix_(true_idx, true_idx)].sum()
            ),
        },
        "summary": summary,
        "admissibility_graph_diagnostics": adm_diag,
        "seed_results": all_seed_results,
        "fence": (
            "configured experiment; discrete curriculum over finite FIPS proxy; "
            "not physical program selection / modular grokking; Issue 017 not closed"
        ),
    }

    path = OUT / "learned_admissibility_01713_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    # Slim: drop full seed dumps / traces / catalogue idx lists
    slim_seeds = []
    for s in all_seed_results:
        slim_seeds.append(
            {
                "seed": s["seed"],
                "n_train": s["vocab_stats"].get(
                    "n_train", len(s.get("vocab_stats", {}))
                ),
                "vocab_stats_brief": {
                    "true_adm_pairs_all": s["vocab_stats"]["true_adm_pairs_all"],
                    "true_disagree_pairs_all": s["vocab_stats"][
                        "true_disagree_pairs_all"
                    ],
                    "n_train_pairs": s["vocab_stats"].get("n_train_pairs"),
                    "n_test_pairs": s["vocab_stats"].get("n_test_pairs"),
                },
                "oracle_train": s["oracle"]["exact"]["train"]["exact_success"],
                "oracle_test": s["oracle"]["exact"]["test"]["exact_success"],
                "arms": [slim_arm(a) for a in s["arms"]],
            }
        )
    # Prefer explicit pair counts from first arm's exact if available
    for i, s in enumerate(all_seed_results):
        # task sizes from oracle / exact n
        tr_n = s["arms"][0]["exact"]["train"]["n"]
        te_n = s["arms"][0]["exact"]["test"]["n"]
        slim_seeds[i]["n_train"] = tr_n
        slim_seeds[i]["n_test"] = te_n

    slim = {
        "source": report["source"],
        "in_reply_to": report["in_reply_to"],
        "implementation": report["implementation"],
        "witness_check": {
            "disagree": wit["disagree"],
            "matches_01707_expected": wit["matches_01707_expected"],
            "mul_source": wit.get("mul_source", "topographo.ssd.sedenion.SedenionAlgebra.mul"),
        },
        "task": report["task"],
        "summary": summary,
        "admissibility_graph_diagnostics": adm_diag,
        "seed_results_slim": slim_seeds,
        "fence": report["fence"],
    }
    slim_path = OUT / "learned_admissibility_01713_report_slim.json"
    slim_path.write_text(json.dumps(slim, indent=2) + "\n")

    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("ADM_DIAG")
    print(json.dumps(adm_diag, indent=2))
    print("wrote", path)
    print("wrote", slim_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

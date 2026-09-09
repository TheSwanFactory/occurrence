#!/usr/bin/env python3
"""017.11 discrete / catalogue denotation head (responds to 017.10 §10.1).

Question
--------
Can a denotation head that selects from the **84 basic-Event catalogue**
(rather than continuous ℝ¹⁶ Event-like cracks) unlock hard finite-FIPS
Cyc edges, so that learning ε_θ actually changes the admissibility graph
and held-out exact native success becomes possible for arm A?

Relaxation (stated clearly)
---------------------------
ε_θ: for each vocab token, learn logits over the 84 catalogue indices.
  - **Train:** Gumbel-Softmax (hard=True, straight-through) mixture of
    catalogue vectors — differentiable relaxation of categorical selection.
  - **Eval:** argmax catalogue vector (hard discrete denotation).
Soft training surrogate (1−|cos|) is OK; reported success is **exact**
projective ray equality under hard FIPS proxy + hard predicates.

Reuses Geometry / task / Policy / soft_program / exact_eval helpers from
``learned_admissibility_01709.py``. Multiplication = topographo
SedenionAlgebra.mul structure constants. Same native-structure task,
holdout, r=e4, seeds 0–2.

Fence: configured experiment; not physical OT / modular grokking.
Issue 017 not closed.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_admissibility_01709 import (
    DIM,
    HOLDOUT,
    LR,
    N_VOCAB,
    SEEDS,
    STEPS,
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
    make_destroy_mask,
    oracle_eval,
    proj_loss,
    soft_program,
    true_denotations,
    verify_witness,
)

OUT = Path(__file__).resolve().parent
N_CATALOGUE = 84
GUMBEL_TAU = 1.0
# Slightly longer than 017.10 continuous — discrete STE can need more steps
STEPS_DISC = 800
LR_DISC = 0.05


class DiscreteCatalogueDenotation(nn.Module):
    """Logits over 84 basic Events per vocab token.

    Train: Gumbel-Softmax hard=True (straight-through).
    Eval (forward_hard): argmax one-hot → catalogue vector.
    """

    def __init__(self, n_vocab: int, catalogue: torch.Tensor, *, noise: float = 0.02):
        super().__init__()
        assert catalogue.shape == (N_CATALOGUE, DIM)
        self.register_buffer("catalogue", catalogue.clone())
        # Small noise init — no leakage of ε*
        self.logits = nn.Parameter(torch.randn(n_vocab, N_CATALOGUE) * noise)

    def forward(self, *, hard: bool | None = None, tau: float = GUMBEL_TAU) -> torch.Tensor:
        """Differentiable denotations (Gumbel-Softmax STE when training)."""
        if hard is None:
            hard = self.training
        if self.training or hard is False:
            # hard=True → STE: forward is argmax-like discrete, backward soft
            y = F.gumbel_softmax(self.logits, tau=tau, hard=True, dim=-1)
        else:
            idx = self.logits.argmax(dim=-1)
            y = F.one_hot(idx, num_classes=N_CATALOGUE).to(dtype=self.logits.dtype)
        return y @ self.catalogue

    def forward_hard(self) -> torch.Tensor:
        """Eval denotations: pure argmax catalogue vectors (no Gumbel noise)."""
        idx = self.logits.argmax(dim=-1)
        return self.catalogue[idx]

    def selected_indices(self) -> np.ndarray:
        return self.logits.detach().argmax(dim=-1).cpu().numpy()


class FrozenTrueCatalogueDenotation(nn.Module):
    """Arm B: freeze to true ε* catalogue indices (no learnable denotation)."""

    def __init__(self, true_idx: np.ndarray, catalogue: torch.Tensor):
        super().__init__()
        self.register_buffer("catalogue", catalogue.clone())
        self.register_buffer(
            "true_idx", torch.tensor(true_idx, dtype=torch.long)
        )
        # Dummy param so count_params / modules stay uniform (frozen)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, *, hard: bool | None = None, tau: float = GUMBEL_TAU) -> torch.Tensor:
        return self.catalogue[self.true_idx]

    def forward_hard(self) -> torch.Tensor:
        return self.catalogue[self.true_idx]

    def selected_indices(self) -> np.ndarray:
        return self.true_idx.cpu().numpy()


def train_arm_discrete(
    name: str,
    geo: Geometry,
    task,
    *,
    learn_den: bool,
    learn_pol: bool,
    force_seq: bool,
    destroy_adm: bool,
    freeze_to_true: bool = False,
    steps: int = STEPS_DISC,
    seed: int = 0,
    lr: float = LR_DISC,
    tau: float = GUMBEL_TAU,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cat_t = geo.basic_t  # (84, 16)
    if freeze_to_true:
        den = FrozenTrueCatalogueDenotation(task.true_idx, cat_t)
        learn_den = False
    else:
        den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
    pol = Policy()

    destroy_mask = make_destroy_mask(task, seed) if destroy_adm else None
    if destroy_adm:
        rng = np.random.default_rng(seed + 123)
        destroy_gate_np = 0.1 + 0.9 * rng.random(len(task.pairs_train)).astype(np.float32)

    with torch.no_grad():
        den_before = den.forward_hard().detach().numpy()
        graph_before = admissibility_graph(geo, den_before)
        idx_before = den.selected_indices().tolist()

    params = []
    if learn_den and not freeze_to_true:
        params += list(den.parameters())
    else:
        for p in den.parameters():
            p.requires_grad_(False)
    if learn_pol and not force_seq:
        params += list(pol.parameters())
    else:
        for p in pol.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(params, lr=lr) if params else None
    r_t = torch.tensor(task.r, dtype=torch.float32)
    M_t = geo.M_t
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)
    if destroy_adm:
        destroy_gate_t = torch.tensor(destroy_gate_np, dtype=torch.float32)

    hist = []
    t0 = time.time()
    loss = torch.tensor(0.0)
    den.train()
    for step in range(steps):
        E = den(hard=True, tau=tau)  # STE Gumbel-Softmax
        a = E[idx[:, 0]]
        b = E[idx[:, 1]]
        soft = geo.soft_cyc_score(a, b)
        if learn_pol and not force_seq:
            logits = pol(a, b, r_t)
        else:
            logits = torch.zeros(a.shape[0], 2)
        pred, w_grp, adm = soft_program(
            a, b, r_t, logits, soft, M_t,
            force_seq=force_seq,
            destroy_adm=destroy_adm,
            destroy_gate=destroy_gate_t if destroy_adm else None,
        )
        loss = proj_loss(pred, tgt)
        # Mild entropy bonus early? Keep simple — match 017.10 style.
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
                    "loss": float(loss.detach()),
                    "mean_soft_adm": float(adm.detach().mean()),
                    "mean_w_grp": float(w_grp.detach().mean()),
                    "hard_adm_frac_checkpoint": hard_adm_now,
                }
            )

    den.eval()
    den_np = den.forward_hard().detach().numpy()
    idx_after = den.selected_indices().tolist()
    with torch.no_grad():
        E_f = den.forward_hard()
        a_f, b_f = E_f[idx[:, 0]], E_f[idx[:, 1]]
        soft_f = geo.soft_cyc_score(a_f, b_f)
        logits_f = (
            pol(a_f, b_f, r_t) if (learn_pol and not force_seq) else torch.zeros(a_f.shape[0], 2)
        )
        pred_f, _, _ = soft_program(
            a_f, b_f, r_t, logits_f, soft_f, M_t,
            force_seq=force_seq,
            destroy_adm=destroy_adm,
            destroy_gate=destroy_gate_t if destroy_adm else None,
        )
        pn = torch.linalg.vector_norm(pred_f, dim=-1, keepdim=True).clamp_min(1e-8)
        tn = torch.linalg.vector_norm(tgt, dim=-1, keepdim=True).clamp_min(1e-8)
        soft_cos = float(((pred_f / pn) * (tgt / tn)).sum(dim=-1).abs().mean())
        hard_adm_frac_learned = float(admissibility_graph(geo, E_f.numpy()).mean())

    graph_after = admissibility_graph(geo, den_np)
    true_graph = geo.fips_adj[np.ix_(task.true_idx, task.true_idx)]

    # Overlap diagnostics: edge set Jaccard / precision-recall vs true
    after_edges = set(zip(*np.where(graph_after > 0)))
    true_edges = set(zip(*np.where(true_graph > 0)))
    before_edges = set(zip(*np.where(graph_before > 0)))
    inter = after_edges & true_edges
    union = after_edges | true_edges
    jaccard = len(inter) / max(len(union), 1)
    precision = len(inter) / max(len(after_edges), 1)
    recall = len(inter) / max(len(true_edges), 1)
    # Task-relevant: edges that are true-adm AND preferred grp (disagree)
    disagree_pairs = {p for p, v in task.true_disagree.items() if v}
    task_relevant_created = len(after_edges & disagree_pairs)

    # Catalogue-index recovery vs true ε*
    true_idx_list = task.true_idx.tolist()
    n_exact_idx = sum(1 for a, b in zip(idx_after, true_idx_list) if a == b)
    # Projective match of selected catalogue vectors to true
    true_den = true_denotations(geo, task)
    n_ray_match = sum(
        1
        for i in range(task.n_vocab)
        if abs(np.dot(den_np[i], true_den[i]))
        > 0.99 * (np.linalg.norm(den_np[i]) * np.linalg.norm(true_den[i]) + 1e-12)
    )

    def policy_fn(i, j, a, b, hard_adm):
        if force_seq or not hard_adm:
            return "seq"
        with torch.no_grad():
            aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            logits = pol(aa, bb, r_t)
            soft = geo.soft_cyc_score(aa, bb)
            if destroy_adm:
                soft = torch.tensor([0.5], dtype=torch.float32)
            masked = logits.clone()
            masked[0, 1] = masked[0, 1] + torch.log(soft.clamp(1e-4, 1.0))
            return "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"

    metrics = exact_eval(
        geo,
        den_np,
        policy_fn,
        task,
        force_seq=force_seq,
        destroy_adm=destroy_adm,
        destroy_mask=destroy_mask,
    )
    collapse = denotation_collapse(den_np)
    # Also report unique catalogue indices selected
    unique_cat = len(set(idx_after))

    return {
        "arm": name,
        "denotation_head": "discrete_catalogue_84",
        "relaxation": (
            "train: Gumbel-Softmax hard=True (STE) over 84 catalogue; "
            "eval: argmax catalogue vector"
        ),
        "param_count_trainable": count_params([den, pol]),
        "param_count_total_modules": count_all_params([den, pol]),
        "learn_den": learn_den and not freeze_to_true,
        "learn_pol": learn_pol and not force_seq,
        "force_seq": force_seq,
        "destroy_adm": destroy_adm,
        "freeze_to_true": freeze_to_true,
        "steps": steps,
        "lr": lr,
        "gumbel_tau": tau,
        "train_loss_surrogate_final": float(loss.detach()),
        "train_mean_abs_cosine_surrogate": soft_cos,
        "hard_adm_frac_learned_denotations": hard_adm_frac_learned,
        "wall_sec": time.time() - t0,
        "exact": metrics,
        "generalization_gap": metrics["train"]["exact_success"] - metrics["test"]["exact_success"],
        "admissibility_overlap_with_true_elementwise": float((graph_after == true_graph).mean()),
        "adm_edge_count_before": int(graph_before.sum()),
        "adm_edge_count_after": int(graph_after.sum()),
        "true_adm_edge_count": int(true_graph.sum()),
        "adm_edge_jaccard_with_true": float(jaccard),
        "adm_edge_precision_vs_true": float(precision),
        "adm_edge_recall_vs_true": float(recall),
        "task_relevant_disagree_edges_in_learned_graph": int(task_relevant_created),
        "adm_edges_created": int(len(after_edges - before_edges)),
        "adm_edges_destroyed": int(len(before_edges - after_edges)),
        "catalogue_idx_before": idx_before,
        "catalogue_idx_after": idx_after,
        "catalogue_idx_exact_match_to_true": int(n_exact_idx),
        "catalogue_ray_match_to_true": int(n_ray_match),
        "unique_catalogue_indices": unique_cat,
        "denotation_collapse": collapse,
        "train_trace": hist,
        # Omit full graphs from primary report bulk; keep counts + jaccard
    }


def main() -> int:
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    wit = verify_witness(alg, geo.M)
    print("witness_disagree", wit["disagree"], "matches_01707", wit["matches_01707_expected"])
    assert wit["disagree"], "017.07 witness must disagree under SedenionAlgebra.mul"

    true_idx = densest_vocab_indices(geo, N_VOCAB)
    print("vocab_idx", true_idx.tolist())
    print("within_vocab_fips_edges", int(geo.fips_adj[np.ix_(true_idx, true_idx)].sum()))

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
        # A: learn discrete ε + π
        arms.append(
            train_arm_discrete(
                "A_learn_discrete_den_learn_pol",
                geo,
                task,
                learn_den=True,
                learn_pol=True,
                force_seq=False,
                destroy_adm=False,
                seed=seed,
            )
        )
        # B: freeze true ε*, learn π (sanity baseline; same as 017.10 B)
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
        # D: shuffled/destroyed adm with discrete ε
        arms.append(
            train_arm_discrete(
                "D_shuffled_admissibility_discrete",
                geo,
                task,
                learn_den=True,
                learn_pol=True,
                force_seq=False,
                destroy_adm=True,
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
            print(
                f"  {a['arm']}: train={tr:.3f} test={te:.3f} gap={a['generalization_gap']:.3f} "
                f"adm {a['adm_edge_count_before']}→{a['adm_edge_count_after']} "
                f"(true={a['true_adm_edge_count']}) "
                f"jacc={a['adm_edge_jaccard_with_true']:.3f} "
                f"params={a['param_count_trainable']}"
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

    # Aggregate adm-graph diagnostics for A
    adm_diag = {}
    for name in [a["arm"] for a in all_seed_results[0]["arms"]]:
        befores, afters, jacs, recalls, task_rel = [], [], [], [], []
        for s in all_seed_results:
            arm = next(x for x in s["arms"] if x["arm"] == name)
            befores.append(arm["adm_edge_count_before"])
            afters.append(arm["adm_edge_count_after"])
            jacs.append(arm["adm_edge_jaccard_with_true"])
            recalls.append(arm["adm_edge_recall_vs_true"])
            task_rel.append(arm["task_relevant_disagree_edges_in_learned_graph"])
        adm_diag[name] = {
            "adm_edge_count_before_mean": float(np.mean(befores)),
            "adm_edge_count_after_mean": float(np.mean(afters)),
            "adm_edge_count_after_std": float(np.std(afters)),
            "jaccard_mean": float(np.mean(jacs)),
            "recall_mean": float(np.mean(recalls)),
            "task_relevant_disagree_edges_mean": float(np.mean(task_rel)),
        }

    report = {
        "source": "017.11 discrete catalogue denotation Futurator — responds to 017.10 §10.1",
        "in_reply_to": "017.10-GrokBot-learned-geometric-admissibility-result.md",
        "implementation": {
            "mul": "structure constants from topographo.ssd.sedenion.SedenionAlgebra.mul",
            "helpers_reused": "learned_admissibility_01709.py (Geometry, task, Policy, soft_program, exact_eval)",
            "denotation_head": (
                "DiscreteCatalogueDenotation: logits (n_vocab × 84); "
                "train Gumbel-Softmax hard=True STE; eval argmax catalogue vector"
            ),
            "cyc_proxy": (
                "finite FIPS-style: zw!=0 and [zw] a basic-Event ray; "
                "NOT full continuous C"
            ),
            "soft_train_exact_eval": (
                "soft Cyc score during training; exact hard proxy + projective "
                "ray equality at evaluation"
            ),
            "vocab_selection": (
                f"same greedy densest FIPS subgraph as 017.10, n_vocab={N_VOCAB}"
            ),
            "arms": "A/B/C/D + Oracle (E/F omitted — not required for discrete head question)",
            "steps": STEPS_DISC,
            "lr": LR_DISC,
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
            "steps": STEPS_DISC,
            "lr": LR_DISC,
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
            "configured experiment; discrete catalogue head over finite FIPS proxy; "
            "not physical program selection / modular grokking; Issue 017 not closed"
        ),
    }

    path = OUT / "learned_admissibility_01711_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("ADM_DIAG")
    print(json.dumps(adm_diag, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

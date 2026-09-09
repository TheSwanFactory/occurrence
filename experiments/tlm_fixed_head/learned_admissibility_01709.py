#!/usr/bin/env python3
"""017.09 learned geometric-admissibility Futurator (native-structure milestone).

Question
--------
Can learning arrange denotations so useful native compositions become
geometrically Cyc-admissible, then exploit legal OT trees to predict held-out
relational structure?

Strict programs (017.07/017.08)
-------------------------------
  P_seq(a,b,r) = Occ(a, Occ(b,r))
  P_grp(a,b,r) = Occ(Cyc(a,b), r)   only when Cyc-admissible

Cyc finite proxy (EXPLICIT QUALIFICATION)
-----------------------------------------
Same proxy as ``probe_futurator_program_space.py`` / 017.08:
  zw = a*b nonzero AND projective [zw] coincides with a basic-Event ray
  from the 84-catalogue (equivalently: an ordered FIPS edge).
This is a **finite basic-Event / FIPS-style proxy**, NOT the full continuous
certified cyclic locus C from Theory 065/40.

Multiplication
--------------
All exact predicates and the differentiable surrogate use structure constants
extracted from ``topographo.ssd.sedenion.SedenionAlgebra.mul``.  No custom
Cayley–Dickson ``sed_mul`` (the box prototype’s CD path failed the 017.07
witness).

Native comparison: projective ray equality (sign-insensitive).  No learned
classifier as success criterion.

Fence: configured experiment; not physical OT enactment / modular grokking.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from topographo.ssd.sedenion import SedenionAlgebra

DIM = 16
TOL = 1e-8
OUT = Path(__file__).resolve().parent
N_VOCAB = 16
HOLDOUT = 0.35
STEPS = 500
LR = 0.05
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Algebra: structure constants from SedenionAlgebra.mul
# ---------------------------------------------------------------------------

def structure_constants(alg: SedenionAlgebra) -> np.ndarray:
    """M[i,j,k] with e_i * e_j = Σ_k M[i,j,k] e_k, from SedenionAlgebra.mul."""
    M = np.zeros((DIM, DIM, DIM), dtype=np.float64)
    for i in range(DIM):
        for j in range(DIM):
            a = np.zeros(DIM)
            a[i] = 1.0
            b = np.zeros(DIM)
            b[j] = 1.0
            M[i, j] = alg.mul(a, b)
    return M


def mul_np(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...j,ijk->...k", a, b, M)


def mul_t(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,...j,ijk->...k", a, b, M)


def projective_key(v: np.ndarray, tol: float = TOL) -> tuple:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < tol:
        return ("zero",)
    v = v / n
    for x in v:
        if abs(x) > 1e-12:
            if x < 0:
                v = -v
            break
    return tuple(np.round(v, 10).tolist())


def rays_equal(u: np.ndarray, v: np.ndarray) -> bool:
    return projective_key(u) == projective_key(v)


def mean_abs_cosine(pred: np.ndarray, tgt: np.ndarray) -> float:
    pn = np.linalg.norm(pred); tn = np.linalg.norm(tgt)
    if pn < TOL or tn < TOL:
        return 0.0
    return float(abs(np.dot(pred / pn, tgt / tn)))



def basis_vec(i: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def verify_witness(alg: SedenionAlgebra, M: np.ndarray) -> dict:
    """017.07 witness must disagree under topographo SedenionAlgebra.mul."""
    z = basis_vec(1) + basis_vec(10)
    w = basis_vec(4) + basis_vec(15)
    x = basis_vec(4)
    wx = alg.mul(w, x)
    zwx = alg.mul(z, wx)
    zw = alg.mul(z, w)
    zw_x = alg.mul(zw, x)
    # structure-constant path must match
    assert np.allclose(mul_np(z, w, M), zw)
    assert np.allclose(mul_np(zw, x, M), zw_x)
    return {
        "mul_source": "topographo.ssd.sedenion.SedenionAlgebra.mul",
        "w_x": wx.tolist(),
        "z_wx": zwx.tolist(),
        "z_w": zw.tolist(),
        "zw_x": zw_x.tolist(),
        "P_seq_key": list(projective_key(zwx)),
        "P_grp_key": list(projective_key(zw_x)),
        "disagree": not rays_equal(zwx, zw_x),
        "matches_01707_expected": (
            projective_key(zwx) == projective_key(basis_vec(1))
            and projective_key(zw_x) == projective_key(z)
        ),
    }


# ---------------------------------------------------------------------------
# Geometry / Cyc proxy
# ---------------------------------------------------------------------------

class Geometry:
    def __init__(self, alg: SedenionAlgebra):
        self.alg = alg
        self.M = structure_constants(alg)
        self.M_t = torch.tensor(self.M, dtype=torch.float32)
        self.basic = np.asarray(alg.basis_zero_divisors(), dtype=np.float64)
        assert self.basic.shape == (84, DIM)
        self.basic_keys = {projective_key(e) for e in self.basic}
        self.basic_t = torch.tensor(self.basic, dtype=torch.float32)
        # FIPS / probe proxy adjacency on the 84 catalogue
        self.fips_adj = np.zeros((84, 84), dtype=np.int8)
        for i in range(84):
            for j in range(84):
                if self.cyc_hard(self.basic[i], self.basic[j])[0]:
                    self.fips_adj[i, j] = 1

    def cyc_hard(self, a: np.ndarray, b: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Finite FIPS-style proxy: zw!=0 and [zw] is a basic-Event ray."""
        zw = mul_np(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), self.M)
        if float(np.linalg.norm(zw)) < TOL:
            return False, None
        if projective_key(zw) not in self.basic_keys:
            return False, None
        return True, zw

    def soft_cyc_score(self, a: torch.Tensor, b: torch.Tensor, temp: float = 20.0) -> torch.Tensor:
        """Train-time soft proxy: max |cos|(zw, basic) gated by ||zw||."""
        zw = mul_t(a, b, self.M_t.to(a.device))
        zn = torch.linalg.vector_norm(zw, dim=-1, keepdim=True).clamp_min(1e-8)
        u = zw / zn
        basics = self.basic_t.to(a.device)
        best = torch.matmul(u, basics.T).abs().max(dim=-1).values
        gate = torch.sigmoid(temp * (zn.squeeze(-1) - 0.05))
        return best * gate

    def occ(self, e: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        prod = mul_np(e, r, self.M)
        if float(np.linalg.norm(prod)) < TOL:
            return None
        return prod

    def p_seq(self, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        inner = self.occ(b, r)
        if inner is None:
            return None
        return self.occ(a, inner)

    def p_grp(self, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        ok, zw = self.cyc_hard(a, b)
        if not ok:
            return None
        return self.occ(zw, r)


def densest_vocab_indices(geo: Geometry, k: int = N_VOCAB) -> np.ndarray:
    """Greedy subset of the 84 catalogue maximizing within-vocab FIPS edges.

    Critical fix vs box prototype: random/top-degree sampling yielded true_adm
    0–6 (unusable). Prefer edges that also exhibit seq≠grp disagreement.
    """
    n = 84
    r = basis_vec(4)
    disagree = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if not geo.fips_adj[i, j]:
                continue
            ps = geo.p_seq(geo.basic[i], geo.basic[j], r)
            pg = geo.p_grp(geo.basic[i], geo.basic[j], r)
            if ps is not None and pg is not None and not rays_equal(ps, pg):
                disagree[i, j] = 1
    score = geo.fips_adj.astype(np.int32) + 2 * disagree.astype(np.int32)
    remaining = set(range(n))
    start = int(np.argmax(score.sum(1)))
    S = [start]
    remaining.remove(start)
    while len(S) < k and remaining:
        best, best_sc = None, -1
        for v in remaining:
            sc = int(score[v, S].sum() + score[S, v].sum())
            if sc > best_sc:
                best_sc, best = sc, v
        assert best is not None
        S.append(best)
        remaining.remove(best)
    return np.asarray(S, dtype=np.int64)


def project_event_like(raw: torch.Tensor) -> torch.Tensor:
    """Differentiable map R^16 → unit pure-pair Event-like crack."""
    p = raw[..., :8].clone()
    q = raw[..., 8:].clone()
    p[..., 0] = 0
    q[..., 0] = 0
    pn = torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(1e-8)
    p = p / pn
    q = q - (q * p).sum(dim=-1, keepdim=True) * p
    qn = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    q = q / qn
    return torch.cat([p, q], dim=-1) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    n_vocab: int
    true_idx: np.ndarray
    r: np.ndarray
    pairs_train: list[tuple[int, int]]
    pairs_test: list[tuple[int, int]]
    targets_train: list[np.ndarray]
    targets_test: list[np.ndarray]
    preferred_branch_train: list[str]
    preferred_branch_test: list[str]
    true_admissible: dict[tuple[int, int], bool]
    true_disagree: dict[tuple[int, int], bool]
    vocab_stats: dict


def preferred_target(geo: Geometry, a: np.ndarray, b: np.ndarray, r: np.ndarray):
    """Preferred legal tree: grp when both defined and disagree; else seq; else grp."""
    ps = geo.p_seq(a, b, r)
    ok, _ = geo.cyc_hard(a, b)
    pg = geo.p_grp(a, b, r) if ok else None
    if ps is not None and pg is not None:
        if not rays_equal(ps, pg):
            return pg, "grp", True, True
        return ps, "seq", True, False
    if ps is not None:
        return ps, "seq", False, False
    if pg is not None:
        return pg, "grp", True, False
    return None, None, False, False


def build_native_task(
    geo: Geometry,
    true_idx: np.ndarray,
    *,
    seed: int,
    holdout: float = HOLDOUT,
    r: np.ndarray | None = None,
) -> Task:
    """Build task from FIXED hidden denotations. Split over pair combinations.

    Held-out targets never enter init or masks.
    """
    rng = np.random.default_rng(seed)
    if r is None:
        r = basis_vec(4)
    n_vocab = int(true_idx.shape[0])
    pairs = [(i, j) for i in range(n_vocab) for j in range(n_vocab)]
    rng.shuffle(pairs)
    n_test = max(1, int(len(pairs) * holdout))
    raw_test = pairs[:n_test]
    raw_train = pairs[n_test:]

    true_adm: dict[tuple[int, int], bool] = {}
    true_dis: dict[tuple[int, int], bool] = {}

    def materialize(pair_list):
        keep, tgts, branches = [], [], []
        for i, j in pair_list:
            a = geo.basic[true_idx[i]]
            b = geo.basic[true_idx[j]]
            t, br, adm, dis = preferred_target(geo, a, b, r)
            true_adm[(i, j)] = adm
            true_dis[(i, j)] = dis
            if t is None:
                continue
            keep.append((i, j))
            tgts.append(np.asarray(t, dtype=np.float64))
            branches.append(br)
        return keep, tgts, branches

    pairs_train, targets_train, br_train = materialize(raw_train)
    pairs_test, targets_test, br_test = materialize(raw_test)

    sub = geo.fips_adj[np.ix_(true_idx, true_idx)]
    vocab_stats = {
        "n_vocab": n_vocab,
        "true_idx": true_idx.tolist(),
        "fips_edges_within_vocab": int(sub.sum()),
        "density": float(sub.sum()) / (n_vocab * n_vocab),
        "true_adm_pairs_all": int(sum(1 for v in true_adm.values() if v)),
        "true_disagree_pairs_all": int(sum(1 for v in true_dis.values() if v)),
        "n_train": len(pairs_train),
        "n_test": len(pairs_test),
        "train_grp_preferred": sum(1 for b in br_train if b == "grp"),
        "test_grp_preferred": sum(1 for b in br_test if b == "grp"),
        "r": "e4",
        "note": (
            "Vocab = densest FIPS subgraph among 84 basic Events (greedy). "
            "Finite FIPS proxy only; not full continuous C."
        ),
    }
    return Task(
        n_vocab=n_vocab,
        true_idx=true_idx,
        r=r,
        pairs_train=pairs_train,
        pairs_test=pairs_test,
        targets_train=targets_train,
        targets_test=targets_test,
        preferred_branch_train=br_train,
        preferred_branch_test=br_test,
        true_admissible=true_adm,
        true_disagree=true_dis,
        vocab_stats=vocab_stats,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Denotation(nn.Module):
    def __init__(self, n_vocab: int, init: np.ndarray | None = None, noise: float = 0.3):
        super().__init__()
        if init is not None:
            # Store raw so project_event_like ≈ init for true basic Events.
            # True cracks already have zero real parts / equal norms; invert softly.
            raw = torch.tensor(init, dtype=torch.float32) * math.sqrt(2.0)
            self.raw = nn.Parameter(raw.clone())
        else:
            self.raw = nn.Parameter(torch.randn(n_vocab, DIM) * noise)

    def forward(self) -> torch.Tensor:
        return project_event_like(self.raw)


class Policy(nn.Module):
    def __init__(self, d: int = DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b, r.expand_as(a)], dim=-1))


def count_params(modules) -> int:
    return int(sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad))


def count_all_params(modules) -> int:
    return int(sum(p.numel() for m in modules for p in m.parameters()))


def proj_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pn = torch.linalg.vector_norm(pred, dim=-1, keepdim=True).clamp_min(1e-8)
    tn = torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1e-8)
    cos = ((pred / pn) * (target / tn)).sum(dim=-1).abs()
    return (1.0 - cos).mean()


def soft_program(
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    logits: torch.Tensor,
    soft_cyc: torch.Tensor,
    M_t: torch.Tensor,
    *,
    force_seq: bool = False,
    destroy_adm: bool = False,
    destroy_gate: torch.Tensor | None = None,
):
    br = mul_t(b, r.expand_as(a), M_t)
    seq = mul_t(a, br, M_t)
    zw = mul_t(a, b, M_t)
    grp = mul_t(zw, r.expand_as(a), M_t)
    if force_seq:
        return seq, torch.zeros(a.shape[0], device=a.device), soft_cyc
    if destroy_adm:
        assert destroy_gate is not None
        adm = destroy_gate
    else:
        adm = soft_cyc
    masked = logits.clone()
    masked[:, 1] = masked[:, 1] + torch.log(adm.clamp(1e-4, 1.0))
    w = F.softmax(masked, dim=-1)
    w_seq = w[:, 0:1]
    w_grp = w[:, 1:2] * adm.unsqueeze(-1)
    s = (w_seq + w_grp).clamp_min(1e-8)
    w_seq, w_grp = w_seq / s, w_grp / s
    return w_seq * seq + w_grp * grp, w_grp.squeeze(-1), adm


def admissibility_graph(geo: Geometry, den_np: np.ndarray) -> np.ndarray:
    n = den_np.shape[0]
    g = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            g[i, j] = int(geo.cyc_hard(den_np[i], den_np[j])[0])
    return g


def denotation_collapse(den_np: np.ndarray) -> dict:
    keys = [projective_key(den_np[i]) for i in range(den_np.shape[0])]
    return {
        "unique_rays": len(set(keys)),
        "n_vocab": den_np.shape[0],
        "collapsed": len(set(keys)) < den_np.shape[0],
    }


def exact_eval(
    geo: Geometry,
    den_np: np.ndarray,
    policy_choose,
    task: Task,
    *,
    force_seq: bool = False,
    destroy_adm: bool = False,
    destroy_mask: dict[tuple[int, int], bool] | None = None,
) -> dict:
    def run_split(pairs, targets):
        ok = adm_count = disagree = grp_chosen = n = 0
        seq_chosen = 0
        for (i, j), tgt in zip(pairs, targets):
            a, b = den_np[i], den_np[j]
            hard_adm, _ = geo.cyc_hard(a, b)
            if destroy_adm:
                assert destroy_mask is not None
                hard_adm_use = bool(destroy_mask.get((i, j), False))
            else:
                hard_adm_use = hard_adm
            if hard_adm:
                adm_count += 1
                ps = geo.p_seq(a, b, task.r)
                pg = geo.p_grp(a, b, task.r)
                if ps is not None and pg is not None and not rays_equal(ps, pg):
                    disagree += 1
            if force_seq or not hard_adm_use:
                branch = "seq"
                out = geo.p_seq(a, b, task.r)
            else:
                branch = policy_choose(i, j, a, b, hard_adm_use)
                if branch == "grp":
                    out = geo.p_grp(a, b, task.r)
                else:
                    out = geo.p_seq(a, b, task.r)
            if out is None:
                continue
            n += 1
            if branch == "grp":
                grp_chosen += 1
            else:
                seq_chosen += 1
            if rays_equal(out, tgt):
                ok += 1
        return {
            "n": n,
            "exact_success": ok / max(n, 1),
            "adm_fraction": adm_count / max(len(pairs), 1),
            "disagree_among_adm": disagree / max(adm_count, 1),
            "grp_choice_fraction": grp_chosen / max(n, 1),
            "seq_choice_fraction": seq_chosen / max(n, 1),
        }

    return {
        "train": run_split(task.pairs_train, task.targets_train),
        "test": run_split(task.pairs_test, task.targets_test),
    }


def true_denotations(geo: Geometry, task: Task) -> np.ndarray:
    return np.stack([geo.basic[i] for i in task.true_idx], axis=0)


def make_destroy_mask(task: Task, seed: int) -> dict[tuple[int, int], bool]:
    """Preserve marginal adm frequency; break geometry↔Cyc link via pair shuffle."""
    rng = np.random.default_rng(seed + 9973)
    all_pairs = task.pairs_train + task.pairs_test
    # Use true graph marginal for frequency matching
    true_flags = [bool(task.true_admissible.get(p, False)) for p in all_pairs]
    n_pos = sum(true_flags)
    shuffled = np.array(true_flags, dtype=bool)
    rng.shuffle(shuffled)
    # If somehow empty, fall back to Bernoulli with empirical rate
    if n_pos == 0:
        rate = 0.25
        shuffled = rng.random(len(all_pairs)) < rate
    return {p: bool(shuffled[k]) for k, p in enumerate(all_pairs)}


def train_arm(
    name: str,
    geo: Geometry,
    task: Task,
    *,
    learn_den: bool,
    learn_pol: bool,
    force_seq: bool,
    destroy_adm: bool,
    freeze_to_true: bool = False,
    steps: int = STEPS,
    seed: int = 0,
    lr: float = LR,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    true_den = true_denotations(geo, task)
    if freeze_to_true:
        den = Denotation(task.n_vocab, init=true_den)
    else:
        den = Denotation(task.n_vocab)  # noise init — no leakage of ε*
    pol = Policy()

    destroy_mask = make_destroy_mask(task, seed) if destroy_adm else None
    # Fixed destroy gates for train pairs (capacity-matched soft scores)
    if destroy_adm:
        rng = np.random.default_rng(seed + 123)
        destroy_gate_np = rng.random(len(task.pairs_train)).astype(np.float32)
        # match soft-score marginal roughly toward high/low
        destroy_gate_np = 0.1 + 0.9 * destroy_gate_np

    with torch.no_grad():
        graph_before = admissibility_graph(geo, den().detach().numpy())

    params = []
    if learn_den:
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
    for step in range(steps):
        E = den()
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
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mean_soft_adm": float(adm.detach().mean()),
                    "mean_w_grp": float(w_grp.detach().mean()),
                }
            )

    den_np = den().detach().numpy()
    with torch.no_grad():
        E_f = den()
        a_f, b_f = E_f[idx[:, 0]], E_f[idx[:, 1]]
        soft_f = geo.soft_cyc_score(a_f, b_f)
        logits_f = pol(a_f, b_f, r_t) if (learn_pol and not force_seq) else torch.zeros(a_f.shape[0], 2)
        pred_f, _, _ = soft_program(
            a_f, b_f, r_t, logits_f, soft_f, M_t,
            force_seq=force_seq, destroy_adm=destroy_adm,
            destroy_gate=destroy_gate_t if destroy_adm else None,
        )
        soft_cos = float((1.0 - proj_loss(pred_f, tgt)).clamp(0, 1))  # mean |cos| approx via 1-loss
        # more precise:
        pn = torch.linalg.vector_norm(pred_f, dim=-1, keepdim=True).clamp_min(1e-8)
        tn = torch.linalg.vector_norm(tgt, dim=-1, keepdim=True).clamp_min(1e-8)
        soft_cos = float(((pred_f / pn) * (tgt / tn)).sum(dim=-1).abs().mean())
        hard_adm_frac_learned = float(admissibility_graph(geo, E_f.numpy()).mean())
    graph_after = admissibility_graph(geo, den_np)
    true_graph = geo.fips_adj[np.ix_(task.true_idx, task.true_idx)]

    def policy_fn(i, j, a, b, hard_adm):
        if force_seq or not hard_adm:
            return "seq"
        with torch.no_grad():
            aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            logits = pol(aa, bb, r_t)
            soft = geo.soft_cyc_score(aa, bb)
            if destroy_adm:
                # use destroy mask as hard gate already applied; soft for logits
                soft = torch.tensor([0.5], dtype=torch.float32)
            masked = logits.clone()
            masked[0, 1] = masked[0, 1] + torch.log(soft.clamp(1e-4, 1.0))
            return "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"

    metrics = exact_eval(
        geo, den_np, policy_fn, task,
        force_seq=force_seq,
        destroy_adm=destroy_adm,
        destroy_mask=destroy_mask,
    )
    overlap = float((graph_after == true_graph).mean())
    collapse = denotation_collapse(den_np)

    return {
        "arm": name,
        "param_count_trainable": count_params([den, pol]),
        "param_count_total_modules": count_all_params([den, pol]),
        "learn_den": learn_den,
        "learn_pol": learn_pol and not force_seq,
        "force_seq": force_seq,
        "destroy_adm": destroy_adm,
        "freeze_to_true": freeze_to_true,
        "train_loss_surrogate_final": float(loss.detach()),
        "train_mean_abs_cosine_surrogate": soft_cos,
        "hard_adm_frac_learned_denotations": hard_adm_frac_learned,
        "wall_sec": time.time() - t0,
        "exact": metrics,
        "generalization_gap": metrics["train"]["exact_success"] - metrics["test"]["exact_success"],
        "admissibility_overlap_with_true": overlap,
        "adm_edge_count_before": int(graph_before.sum()),
        "adm_edge_count_after": int(graph_after.sum()),
        "true_adm_edge_count": int(true_graph.sum()),
        "denotation_collapse": collapse,
        "train_trace": hist,
        "graph_before": graph_before.tolist(),
        "graph_after": graph_after.tolist(),
    }


def oracle_eval(geo: Geometry, task: Task) -> dict:
    """Upper bound: true ε* + correct preferred branch (not a cheating train arm)."""
    den_np = true_denotations(geo, task)
    preferred = {}
    for pairs, branches in (
        (task.pairs_train, task.preferred_branch_train),
        (task.pairs_test, task.preferred_branch_test),
    ):
        for (i, j), br in zip(pairs, branches):
            preferred[(i, j)] = br

    def policy_fn(i, j, a, b, hard_adm):
        return preferred.get((i, j), "grp" if hard_adm else "seq")

    metrics = exact_eval(geo, den_np, policy_fn, task)
    g = admissibility_graph(geo, den_np)
    return {
        "arm": "oracle_true_eps_correct_branch",
        "note": "diagnostic upper bound; not a trained arm",
        "param_count_trainable": 0,
        "exact": metrics,
        "generalization_gap": metrics["train"]["exact_success"] - metrics["test"]["exact_success"],
        "adm_edge_count": int(g.sum()),
        "denotation_collapse": denotation_collapse(den_np),
    }


def arm_E_mul(geo: Geometry, task: Task, steps: int = STEPS, seed: int = 0, lr: float = LR) -> dict:
    """Ambient Mul bracket control (fenced; not OT-native)."""
    torch.manual_seed(seed)
    den = Denotation(task.n_vocab)
    pol = Policy()
    opt = torch.optim.Adam(list(den.parameters()) + list(pol.parameters()), lr=lr)
    r_t = torch.tensor(task.r, dtype=torch.float32)
    M_t = geo.M_t
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)
    t0 = time.time()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        E = den()
        a, b = E[idx[:, 0]], E[idx[:, 1]]
        left = mul_t(mul_t(a, b, M_t), r_t.expand_as(a), M_t)  # (ab)r
        right = mul_t(a, mul_t(b, r_t.expand_as(a), M_t), M_t)  # a(br)
        logits = pol(a, b, r_t)
        w = F.softmax(logits, dim=-1)
        pred = w[:, 0:1] * right + w[:, 1:2] * left
        loss = proj_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    den_np = den().detach().numpy()

    def eval_split(pairs, targets):
        ok = n = 0
        for (i, j), tgt_np in zip(pairs, targets):
            a, b = den_np[i], den_np[j]
            aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = pol(aa, bb, r_t)[0]
                use_left = bool(logits[1] > logits[0])
            if use_left:
                out = mul_np(mul_np(a, b, geo.M), task.r, geo.M)
            else:
                out = mul_np(a, mul_np(b, task.r, geo.M), geo.M)
            if float(np.linalg.norm(out)) < TOL:
                continue
            n += 1
            if rays_equal(out, tgt_np):
                ok += 1
        return {"n": n, "exact_success": ok / max(n, 1)}

    tr = eval_split(task.pairs_train, task.targets_train)
    te = eval_split(task.pairs_test, task.targets_test)
    return {
        "arm": "E_mul_ambient",
        "fence": "ambient Mul bracketings; not OT-native",
        "param_count_trainable": count_params([den, pol]),
        "exact": {"train": tr, "test": te},
        "generalization_gap": tr["exact_success"] - te["exact_success"],
        "wall_sec": time.time() - t0,
        "train_loss_surrogate_final": float(loss.detach()),
        "denotation_collapse": denotation_collapse(den_np),
        "adm_edge_count_after": int(admissibility_graph(geo, den_np).sum()),
    }


def arm_F_mlp(task: Task, steps: int = STEPS, seed: int = 0, lr: float = LR, hidden: int = 48) -> dict:
    """Capacity-matched generic MLP control (not native consequence)."""
    torch.manual_seed(seed)
    n = task.n_vocab
    emb = nn.Embedding(n, DIM)
    net = nn.Sequential(nn.Linear(2 * DIM, hidden), nn.ReLU(), nn.Linear(hidden, DIM))
    opt = torch.optim.Adam(list(emb.parameters()) + list(net.parameters()), lr=lr)
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)
    t0 = time.time()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        x = torch.cat([emb(idx[:, 0]), emb(idx[:, 1])], dim=-1)
        pred = net(x)
        loss = proj_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def eval_split(pairs, targets):
        ok = n_ = 0
        with torch.no_grad():
            for (i, j), tgt_np in zip(pairs, targets):
                x = torch.cat([emb.weight[i], emb.weight[j]], dim=-1).unsqueeze(0)
                pred = net(x)[0].numpy()
                n_ += 1
                if rays_equal(pred, tgt_np):
                    ok += 1
        return {"n": n_, "exact_success": ok / max(n_, 1)}

    tr = eval_split(task.pairs_train, task.targets_train)
    te = eval_split(task.pairs_test, task.targets_test)
    return {
        "arm": "F_generic_mlp",
        "fence": "generic control; not native consequence",
        "param_count_trainable": count_params([emb, net]),
        "exact": {"train": tr, "test": te},
        "generalization_gap": tr["exact_success"] - te["exact_success"],
        "wall_sec": time.time() - t0,
        "train_loss_surrogate_final": float(loss.detach()),
        "note": "exact success = projective match of MLP output to native target ray",
    }


def r_sensitivity(geo: Geometry, true_idx: np.ndarray, seed: int = 0) -> dict:
    """Brief note: rebuild task under alternate retained r and report oracle ceilings."""
    out = {}
    for name, ri in (("e4", 4), ("e3", 3), ("e5", 5)):
        r = basis_vec(ri)
        task = build_native_task(geo, true_idx, seed=seed, r=r)
        ora = oracle_eval(geo, task)
        out[name] = {
            "n_train": len(task.pairs_train),
            "n_test": len(task.pairs_test),
            "true_adm": task.vocab_stats["true_adm_pairs_all"],
            "true_disagree": task.vocab_stats["true_disagree_pairs_all"],
            "oracle_train": ora["exact"]["train"]["exact_success"],
            "oracle_test": ora["exact"]["test"]["exact_success"],
        }
    return out


def aggregate(seed_results: list[dict]) -> dict:
    names = [a["arm"] for a in seed_results[0]["arms"]]
    summary = {}
    for name in names:
        trs, tes, gaps = [], [], []
        for s in seed_results:
            arm = next(x for x in s["arms"] if x["arm"] == name)
            trs.append(arm["exact"]["train"]["exact_success"])
            tes.append(arm["exact"]["test"]["exact_success"])
            gaps.append(arm["generalization_gap"])
        summary[name] = {
            "train_mean": float(np.mean(trs)),
            "train_std": float(np.std(trs)),
            "test_mean": float(np.mean(tes)),
            "test_std": float(np.std(tes)),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "n_seeds": len(trs),
        }
    return summary


def main() -> int:
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    wit = verify_witness(alg, geo.M)
    print("witness_disagree", wit["disagree"], "matches_01707", wit["matches_01707_expected"])
    assert wit["disagree"], "017.07 witness must disagree under SedenionAlgebra.mul"

    # Fixed vocab for all seeds (structure fixed; only split/init vary by seed)
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
        assert task.vocab_stats["true_adm_pairs_all"] >= 20, (
            "CRITICAL: true_adm too sparse — densest-subgraph selection failed"
        )

        arms = []
        # A: learn ε + π
        arms.append(train_arm(
            "A_learn_den_learn_pol", geo, task,
            learn_den=True, learn_pol=True, force_seq=False, destroy_adm=False, seed=seed,
        ))
        # B: freeze ε to true native assignment; learn π only
        arms.append(train_arm(
            "B_freeze_true_den_learn_pol", geo, task,
            learn_den=False, learn_pol=True, force_seq=False, destroy_adm=False,
            freeze_to_true=True, seed=seed,
        ))
        # C: learn ε, force P_seq
        arms.append(train_arm(
            "C_learn_den_force_seq", geo, task,
            learn_den=True, learn_pol=False, force_seq=True, destroy_adm=False, seed=seed,
        ))
        # D: shuffled admissibility
        arms.append(train_arm(
            "D_shuffled_admissibility", geo, task,
            learn_den=True, learn_pol=True, force_seq=False, destroy_adm=True, seed=seed,
        ))
        arms.append(arm_E_mul(geo, task, seed=seed))
        arms.append(arm_F_mlp(task, seed=seed))
        oracle = oracle_eval(geo, task)

        all_seed_results.append({
            "seed": seed,
            "vocab_stats": task.vocab_stats,
            "oracle": oracle,
            "arms": arms,
        })
        print(f"  oracle train={oracle['exact']['train']['exact_success']:.3f} "
              f"test={oracle['exact']['test']['exact_success']:.3f}")
        for a in arms:
            te = a["exact"]["test"]["exact_success"]
            tr = a["exact"]["train"]["exact_success"]
            print(
                f"  {a['arm']}: train={tr:.3f} test={te:.3f} gap={a['generalization_gap']:.3f} "
                f"params={a.get('param_count_trainable', a.get('param_count', '?'))}"
            )

    summary = aggregate(all_seed_results)
    # include oracle in summary
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

    r_sens = r_sensitivity(geo, true_idx, seed=0)

    report = {
        "source": "017.09 learned geometric-admissibility Futurator — native-structure milestone",
        "implementation_revision": {
            "mul": "structure constants from topographo.ssd.sedenion.SedenionAlgebra.mul",
            "rejected": "box prototype custom CD sed_mul (failed 017.07 witness: zw→0)",
            "helpers_reused": "occ/ray_key/witness patterns from probe_futurator_program_space.py",
            "cyc_proxy": (
                "finite FIPS-style: zw!=0 and [zw] a basic-Event ray "
                "(=336 ordered FIPS edges on 84 catalogue); NOT full continuous C"
            ),
            "soft_train_exact_eval": (
                "soft Cyc score (max |cos| to basic catalogue) during training; "
                "exact hard proxy + projective ray equality at evaluation"
            ),
            "arm_B_freeze": "true native basic-Event assignment ε* (not a wrong fixed assignment)",
            "vocab_selection": (
                f"greedy densest FIPS subgraph, n_vocab={N_VOCAB}, "
                "score=edges+2*disagree — critical fix for prior true_adm 0–6 failure"
            ),
            "box_report_appendix": "learned_admissibility_report_box_v1.json",
        },
        "witness_check": wit,
        "task": {
            "kind": "native-structure from fixed hidden OT-compatible denotations",
            "n_vocab": N_VOCAB,
            "holdout": HOLDOUT,
            "split": "over pair combinations, not repeats",
            "target_rule": "grp when seq/grp disagree and grp defined; else seq (else grp if seq undefined)",
            "r_primary": "e4",
            "steps": STEPS,
            "lr": LR,
            "seeds": list(SEEDS),
            "fixed_true_idx": true_idx.tolist(),
            "within_vocab_fips_edges": int(geo.fips_adj[np.ix_(true_idx, true_idx)].sum()),
        },
        "r_sensitivity": r_sens,
        "summary": summary,
        "seed_results": all_seed_results,
        "fence": (
            "configured experiment; not physical program selection / modular grokking; "
            "stage-1 native-structure only"
        ),
    }

    path = OUT / "learned_admissibility_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

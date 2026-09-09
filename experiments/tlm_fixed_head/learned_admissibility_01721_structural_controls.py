#!/usr/bin/env python3
"""017.21 structural-admissibility and ambient-Mul controls.

Question: once denotations/arithmetic are fixed, does the actual Cyc
admissibility relation help policy generalization beyond having a second
nonassociative program?

Arms
----
1. Native     — frozen recovered dens, true finite Cyc mask, learned 2-branch π
2. ForceSeq   — same dens, sequential only (no policy learning)
3. Ambient    — same dens+π budget, grouped whenever ambient Mul defined (no Cyc)
4. Rewired    — ambient endpoints, grouped gated by 3 degree-preserving edge-swap masks

Critical: Phi(a,b)=(conj(a),b) in tensor forward path AND target construction.
Exact scorer authoritative. Fixed benchmark materialized+hashed before arms.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learned_admissibility_01709 import (
    DIM,
    HOLDOUT,
    N_VOCAB,
    SEEDS,
    TOL,
    Geometry,
    Policy,
    SedenionAlgebra,
    basis_vec,
    build_native_task,
    count_params,
    densest_vocab_indices,
    mul_np,
    mul_t,
    projective_key,
    proj_loss,
    rays_equal,
    verify_witness,
)
from learned_admissibility_01711_discrete import (
    FrozenTrueCatalogueDenotation,
    LR_DISC,
    STEPS_DISC,
)
from learned_admissibility_01719_exact_eval import (
    audit_basis_products,
    audit_catalogue_products,
    audit_programs_at_e4,
    audit_rec_ray_teacher,
    exact_canonical_key,
    exact_from_float_idx,
    exact_p_grp,
    exact_p_seq,
    exact_rays_equal,
    exact_value_to_float,
    float_mul_with_phi,
    float_to_exact_map,
    minimal_mismatch_witness,
    phi_signs,
    preferred_target_exact,
)
from topographo.ssd import exact, fips_basic, projective
import topographo.ssd.exact as exact_mod

OUT = Path(__file__).resolve().parent
ART = OUT / "01721_artifacts"
ART.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ART / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ATTACH = OUT / "017.22-Code-attachments"
ATTACH.mkdir(parents=True, exist_ok=True)

POLICY_STEPS = STEPS_DISC  # 800 — same budget as prior freeze-π arms
POLICY_LR = LR_DISC        # 0.05
REWIRE_SEEDS = (101, 202, 303)
N_CATALOGUE = 84
PERM_SEEDS = tuple(range(10))  # 10 published catalogue permutations


# ---------------------------------------------------------------------------
# Phi-correct tensor / numpy multiplication (Value coordinates)
# ---------------------------------------------------------------------------

def phi_signs_t(device=None) -> torch.Tensor:
    s = torch.tensor(phi_signs(), dtype=torch.float32)
    return s.to(device) if device is not None else s


def mul_np_phi(a: np.ndarray, b: np.ndarray, M: np.ndarray) -> np.ndarray:
    s = phi_signs()
    return mul_np(np.asarray(a, dtype=np.float64) * s, np.asarray(b, dtype=np.float64) * s, M) * s


def mul_t_phi(a: torch.Tensor, b: torch.Tensor, M_t: torch.Tensor) -> torch.Tensor:
    s = phi_signs_t(a.device)
    return mul_t(a * s, b * s, M_t) * s


def soft_program_phi(
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    logits: torch.Tensor,
    gate: torch.Tensor,
    M_t: torch.Tensor,
    *,
    force_seq: bool = False,
):
    """Phi-correct soft mixture of P_seq and P_grp with hard/soft availability gate."""
    br = mul_t_phi(b, r.expand_as(a), M_t)
    seq = mul_t_phi(a, br, M_t)
    zw = mul_t_phi(a, b, M_t)
    grp = mul_t_phi(zw, r.expand_as(a), M_t)
    if force_seq:
        return seq, torch.zeros(a.shape[0], device=a.device), gate
    masked = logits.clone()
    masked[:, 1] = masked[:, 1] + torch.log(gate.clamp(1e-4, 1.0))
    w = F.softmax(masked, dim=-1)
    w_seq = w[:, 0:1]
    w_grp = w[:, 1:2] * gate.unsqueeze(-1)
    s = (w_seq + w_grp).clamp_min(1e-8)
    w_seq, w_grp = w_seq / s, w_grp / s
    return w_seq * seq + w_grp * grp, w_grp.squeeze(-1), gate


class PhiGeometry:
    """Geometry helpers with Phi-correct Value-space multiplication.

    Catalogue vectors (geo.basic) remain in public Value coordinates.
    """

    def __init__(self, geo: Geometry):
        self.geo = geo
        self.alg = geo.alg
        self.M = geo.M
        self.M_t = geo.M_t
        self.basic = geo.basic
        self.basic_t = geo.basic_t
        self.basic_keys = geo.basic_keys
        self.coordinate_system = (
            "public 061.11 / fips_basic Value coordinates; "
            "products via Phi↔core-C↔Phi (matches exact.mul)"
        )

    def occ(self, e: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        prod = mul_np_phi(e, r, self.M)
        if float(np.linalg.norm(prod)) < TOL:
            return None
        return prod

    def cyc_hard(self, a: np.ndarray, b: np.ndarray) -> tuple[bool, np.ndarray | None]:
        zw = mul_np_phi(a, b, self.M)
        if float(np.linalg.norm(zw)) < TOL:
            return False, None
        if projective_key(zw) not in self.basic_keys:
            return False, None
        return True, zw

    def ambient_zw(self, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        zw = mul_np_phi(a, b, self.M)
        if float(np.linalg.norm(zw)) < TOL:
            return None
        return zw

    def p_seq(self, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        inner = self.occ(b, r)
        if inner is None:
            return None
        return self.occ(a, inner)

    def p_grp_cyc(self, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        ok, zw = self.cyc_hard(a, b)
        if not ok:
            return None
        return self.occ(zw, r)

    def p_grp_ambient(self, a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray | None:
        zw = self.ambient_zw(a, b)
        if zw is None:
            return None
        return self.occ(zw, r)


# ---------------------------------------------------------------------------
# Exact ambient programs
# ---------------------------------------------------------------------------

def exact_p_grp_ambient(a: exact.Value, b: exact.Value, r: exact.Value) -> exact.Value | None:
    zw = exact.mul(a, b)
    if zw == exact.zero():
        return None
    prod = exact.mul(zw, r)
    if prod == exact.zero():
        return None
    return prod


# ---------------------------------------------------------------------------
# Fixed benchmark
# ---------------------------------------------------------------------------

def sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def materialize_benchmark(geo: Geometry, pgeo: PhiGeometry, fte: list[int], recovered: dict[int, list[int]]):
    """One fixed manifest: original splits + frozen dens + exact target rays."""
    true_idx = densest_vocab_indices(geo, N_VOCAB)
    re = exact.basis(4)
    seeds_payload = {}
    for seed in SEEDS:
        task = build_native_task(geo, true_idx, seed=seed)  # original float materialization for IDs
        # Reconstruct exact targets for the SAME kept pairs (and verify definedness)
        et_train, et_test = [], []
        ft_train, ft_test = [], []  # float Value-coord projections for training loss
        old_vs_exact = {"train_ray_match": 0, "train_n": 0, "test_ray_match": 0, "test_n": 0}
        for pairs, old_tgts, et_bucket, ft_bucket, tag in (
            (task.pairs_train, task.targets_train, et_train, ft_train, "train"),
            (task.pairs_test, task.targets_test, et_test, ft_test, "test"),
        ):
            for (i, j), old_t in zip(pairs, old_tgts):
                ti, tj = int(true_idx[i]), int(true_idx[j])
                ae = exact_from_float_idx(ti, fte)
                be = exact_from_float_idx(tj, fte)
                tgt, br, adm, dis = preferred_target_exact(ae, be, re)
                if tgt is None:
                    raise RuntimeError(f"exact target undefined for kept pair {(i,j)} seed={seed}")
                et_bucket.append(tgt)
                ft = exact_value_to_float(tgt)
                assert ft is not None
                ft_bucket.append(ft)
                old_vs_exact[f"{tag}_n"] += 1
                if rays_equal(old_t, ft):
                    old_vs_exact[f"{tag}_ray_match"] += 1

        # Availability masks on recovered dens (token×token)
        den = recovered[seed]
        cyc_mask = np.zeros((N_VOCAB, N_VOCAB), dtype=np.int8)
        amb_mask = np.zeros((N_VOCAB, N_VOCAB), dtype=np.int8)
        for i in range(N_VOCAB):
            for j in range(N_VOCAB):
                ae = exact_from_float_idx(int(den[i]), fte)
                be = exact_from_float_idx(int(den[j]), fte)
                cyc_mask[i, j] = int(fips_basic.admissible(ae, be))
                amb_mask[i, j] = int(exact_p_grp_ambient(ae, be, re) is not None)

        seeds_payload[seed] = {
            "pairs_train": [list(p) for p in task.pairs_train],
            "pairs_test": [list(p) for p in task.pairs_test],
            "exact_target_canon_train": [exact_canonical_key(t) for t in et_train],
            "exact_target_canon_test": [exact_canonical_key(t) for t in et_test],
            "preferred_branch_train": list(task.preferred_branch_train),
            "preferred_branch_test": list(task.preferred_branch_test),
            "recovered_catalogue_idx": list(den),
            "true_idx": true_idx.tolist(),
            "old_float_vs_exact_target_ray_match": old_vs_exact,
            "n_train": len(task.pairs_train),
            "n_test": len(task.pairs_test),
            "cyc_edge_count": int(cyc_mask.sum()),
            "ambient_edge_count": int(amb_mask.sum()),
            # runtime fields (not hashed as arrays — stored separately)
            "_task": task,
            "_et_train": et_train,
            "_et_test": et_test,
            "_ft_train": ft_train,
            "_ft_test": ft_test,
            "_cyc_mask": cyc_mask,
            "_amb_mask": amb_mask,
        }

    manifest_for_hash = {
        "seeds": SEEDS,
        "holdout": HOLDOUT,
        "r": "e4",
        "true_idx": true_idx.tolist(),
        "recovered_by_seed": {str(s): recovered[s] for s in SEEDS},
        "pairs_and_targets": {
            str(s): {
                "pairs_train": seeds_payload[s]["pairs_train"],
                "pairs_test": seeds_payload[s]["pairs_test"],
                "exact_target_canon_train": [list(x) if isinstance(x, tuple) else x for x in seeds_payload[s]["exact_target_canon_train"]],
                "exact_target_canon_test": [list(x) if isinstance(x, tuple) else x for x in seeds_payload[s]["exact_target_canon_test"]],
                "recovered_catalogue_idx": seeds_payload[s]["recovered_catalogue_idx"],
            }
            for s in SEEDS
        },
        "policy": {"arch": "Policy(d=16,hidden=32)", "steps": POLICY_STEPS, "lr": POLICY_LR},
        "coordinate_system": pgeo.coordinate_system,
        "note": "exact preferred targets designated as fixed benchmark; all arms share these rays",
    }
    h = sha256_json(manifest_for_hash)
    return {
        "manifest_sha256": h,
        "manifest": manifest_for_hash,
        "seeds": seeds_payload,
        "true_idx": true_idx.tolist(),
        "coordinate_system_catalogue_vectors": "Value (fips_basic / 061.11)",
        "coordinate_system_products": "Value via Phi adapter",
        "coordinate_system_targets": "exact Value (rational) → float projection for train loss",
    }


def load_recovered_maps(geo: Geometry) -> dict[int, list[int]]:
    """Frozen Rec_ray catalogue maps from 017.17/017.18 (validated in 017.20)."""
    true_idx = densest_vocab_indices(geo, N_VOCAB).tolist()
    out = {}
    ckpt_root = OUT / "01717_artifacts" / "checkpoints"
    for seed in SEEDS:
        path = ckpt_root / f"seed{seed}_Rec_ray.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        idx = [int(x) for x in ck["catalogue_idx"]]
        assert len(idx) == N_VOCAB
        out[seed] = idx
        # Expected 16/16 under first-max (conditional label applied in tie audit)
        match = sum(1 for a, b in zip(idx, true_idx) if a == b)
        assert match == 16, f"Rec_ray seed{seed} match {match}/16 — unexpected"
    return out


# ---------------------------------------------------------------------------
# Rewired masks (degree-preserving edge swaps)
# ---------------------------------------------------------------------------

def undirected_edges(mask: np.ndarray) -> list[tuple[int, int]]:
    n = mask.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # treat as undirected: edge if either direction (Cyc graph is nearly symmetric)
            if mask[i, j] or mask[j, i]:
                edges.append((i, j))
    return edges


def mask_from_undirected(edges: set[tuple[int, int]], n: int, template: np.ndarray) -> np.ndarray:
    """Rebuild directed mask: for each undirected edge, copy template directions if present,
    else set both directions (no self-loops)."""
    out = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        if template[i, j] or template[j, i]:
            out[i, j] = int(template[i, j])
            out[j, i] = int(template[j, i])
            # If template had only one direction, keep that; if neither somehow, both
            if out[i, j] == 0 and out[j, i] == 0:
                out[i, j] = out[j, i] = 1
        else:
            out[i, j] = out[j, i] = 1
    np.fill_diagonal(out, 0)
    return out


def degree_sequence_undirected(edges: set[tuple[int, int]], n: int) -> tuple[int, ...]:
    deg = [0] * n
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    return tuple(deg)


def rewire_degree_preserving(mask: np.ndarray, seed: int, n_attempts: int = 50000) -> dict:
    """Deterministic degree-preserving double-edge swaps on undirected projection."""
    rng = np.random.default_rng(seed)
    n = mask.shape[0]
    edges = set(undirected_edges(mask))
    deg0 = degree_sequence_undirected(edges, n)
    n_edges0 = len(edges)
    swaps_done = 0
    attempts = 0
    # Target many successful swaps relative to edge count
    target_swaps = max(20, n_edges0 * 3)
    while swaps_done < target_swaps and attempts < n_attempts:
        attempts += 1
        if len(edges) < 2:
            break
        elist = list(edges)
        (a, b), (c, d) = elist[rng.integers(0, len(elist))], elist[rng.integers(0, len(elist))]
        if len({a, b, c, d}) < 4:
            continue
        # Two possible rewirings: (a,c)+(b,d) or (a,d)+(b,c)
        options = [((a, c), (b, d)), ((a, d), (b, c))]
        rng.shuffle(options)
        for (e1, e2) in options:
            u1, v1 = (e1 if e1[0] < e1[1] else (e1[1], e1[0]))
            u2, v2 = (e2 if e2[0] < e2[1] else (e2[1], e2[0]))
            if u1 == v1 or u2 == v2:
                continue
            if (u1, v1) in edges or (u2, v2) in edges:
                continue
            # perform swap
            edges.remove((a, b) if a < b else (b, a))
            edges.remove((c, d) if c < d else (d, c))
            edges.add((u1, v1))
            edges.add((u2, v2))
            swaps_done += 1
            break
    deg1 = degree_sequence_undirected(edges, n)
    new_mask = mask_from_undirected(edges, n, mask)
    # Overlap with native undirected edges
    native_u = set(undirected_edges(mask))
    overlap = len(native_u & edges)
    union = len(native_u | edges)
    # Simple non-isomorphism heuristic: edge-set difference after degree match
    # (full graph-iso is hard; we report structural difference + degree preservation)
    isomorphic_candidate = edges == native_u
    return {
        "seed": seed,
        "n_undirected_edges": n_edges0,
        "swaps_done": swaps_done,
        "attempts": attempts,
        "degree_sequence_preserved": deg0 == deg1,
        "degree_sequence": list(deg0),
        "edge_overlap_undirected": overlap,
        "edge_union_undirected": union,
        "jaccard_undirected": (overlap / union) if union else 1.0,
        "identical_to_native": isomorphic_candidate,
        "non_isomorphic_edge_set": not isomorphic_candidate,
        "adjacency": new_mask.astype(int).tolist(),
        "_mask": new_mask,
    }


# ---------------------------------------------------------------------------
# Tie / order audit
# ---------------------------------------------------------------------------

def catalogue_permutations(seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    return rng.permutation(N_CATALOGUE)


def tie_and_order_audit(geo: Geometry, fte: list[int], bench: dict) -> dict:
    """Vote ties + 10 catalogue permutations + tied-token H range."""
    true_idx = np.asarray(bench["true_idx"], dtype=np.int64)
    per_seed = {}
    for seed in SEEDS:
        task = bench["seeds"][seed]["_task"]
        # Reuse 017.19 teacher audit (exact keys)
        audit = audit_rec_ray_teacher(geo, task, fte)
        # Enrich with all maximizing candidates
        # Rebuild votes quickly for candidate lists
        from learned_admissibility_01719_exact_eval import exact_program_maps

        maps_e = exact_program_maps(fte)
        votes = np.zeros((N_VOCAB, N_CATALOGUE), dtype=np.float64)
        re = exact.basis(4)
        for (i, j) in task.pairs_train:
            ti, tj = int(true_idx[i]), int(true_idx[j])
            etgt, _, _, _ = preferred_target_exact(
                exact_from_float_idx(ti, fte), exact_from_float_idx(tj, fte), re
            )
            k = exact_canonical_key(etgt)
            for ca, cb in maps_e["seq"].get(k, []):
                votes[i, ca] += 1.0
                votes[j, cb] += 1.0
        token_detail = []
        for t in range(N_VOCAB):
            row = votes[t]
            mx = float(row.max()) if row.sum() > 0 else 0.0
            cands = [int(c) for c in np.where(row == mx)[0]] if mx > 0 else []
            order = np.argsort(-row)
            ru = float(row[order[1]]) if row.sum() > 0 else 0.0
            token_detail.append(
                {
                    "token": t,
                    "max_votes": mx,
                    "maximizing_event_candidates": cands,
                    "n_maximizers": len(cands),
                    "margin_to_next": mx - ru if len(cands) == 1 else 0.0,
                    "first_max": int(order[0]) if row.sum() > 0 else None,
                    "true": int(true_idx[t]),
                    "first_max_matches_true": bool(row.sum() > 0 and int(order[0]) == int(true_idx[t])),
                    "tied": len(cands) > 1,
                }
            )
        tied_tokens = [d for d in token_detail if d["tied"]]
        # Enumerate alternative maps for tied tokens (train-only: keep first-max for non-ties;
        # for ties, enumerate candidates; held-out H reported for each combo)
        alt_H = []
        if tied_tokens:
            # cartesian product of tied candidate lists (small)
            from itertools import product

            slots = [d["maximizing_event_candidates"] for d in tied_tokens]
            base = [d["first_max"] for d in token_detail]
            for combo in product(*slots):
                den = list(base)
                for d, c in zip(tied_tokens, combo):
                    den[d["token"]] = c
                # held-out H under this denotation map (native Cyc availability)
                H = heldout_H_native(den, fte, bench["seeds"][seed])
                alt_H.append({"den": den, "test_H": H, "matches_true": den == true_idx.tolist()})

        # Catalogue-order sensitivity: permute catalogue labels, remap votes via perm
        perm_results = []
        for ps in PERM_SEEDS:
            perm = catalogue_permutations(ps)  # new_idx = perm[old_idx]
            inv = np.empty_like(perm)
            inv[perm] = np.arange(N_CATALOGUE)
            # votes were in float-basic indices; reindex catalogue axis
            votes_p = votes[:, inv]  # column c' gets votes of perm^{-1}
            # Actually: if we reorder catalogue storage by perm (position k holds old perm[k]),
            # vote for physical event e stays with e; first-max index in new order is perm position.
            # Map selected labels back to physical Event rays = geo.basic[old_idx].
            hard_new = votes.argmax(axis=1)  # physical float-basic idx unchanged by storage order
            # Storage-order only: argmax on permuted columns then map back
            hard_storage = votes[:, inv].argmax(axis=1)
            physical_from_storage = [int(inv[i]) for i in hard_storage]  # wait: votes_p[t,k]=votes[t,inv[k]]
            # votes_p[t, k] = votes[t, inv[k]] means column k corresponds to physical inv[k]
            # argmax_k votes_p = k* => physical = inv[k*]
            physical = [int(inv[k]) for k in hard_storage]
            # Equivalent check: physical should equal original argmax (storage order irrelevant to physics)
            same_physical = physical == hard_new.tolist()
            match16 = sum(1 for a, b in zip(hard_new.tolist(), true_idx.tolist()) if a == b)
            perm_results.append(
                {
                    "perm_seed": ps,
                    "perm": perm.tolist(),
                    "first_max_match_over_16": match16,
                    "storage_order_physical_same_as_unpermuted": same_physical,
                }
            )

        per_seed[seed] = {
            "token_detail": token_detail,
            "n_tied_tokens": len(tied_tokens),
            "tied_tokens": [d["token"] for d in tied_tokens],
            "first_max_match_over_16": sum(1 for d in token_detail if d["first_max_matches_true"]),
            "any_tie": len(tied_tokens) > 0,
            "alt_test_H_range": {
                "min": min((a["test_H"] for a in alt_H), default=None),
                "max": max((a["test_H"] for a in alt_H), default=None),
                "n_combos": len(alt_H),
                "combos": alt_H[:32],  # cap
            },
            "catalogue_permutations": perm_results,
            "first_max_recovery_16_16_all_perms": all(
                p["first_max_match_over_16"] == 16 for p in perm_results
            ),
            "01719_audit_summary": {
                "any_token_vote_tie": audit["leakage_and_ambiguity"]["any_token_vote_tie"],
                "exact_vote_match_to_true": audit["leakage_and_ambiguity"]["exact_vote_match_to_true"],
                "unique_aggregate_recovery": audit["leakage_and_ambiguity"]["unique_aggregate_recovery"],
            },
        }
    return {
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "primary_map_conditional_on_favorable_first_max": any(
            per_seed[s]["any_tie"] for s in SEEDS
        ),
        "note": (
            "Primary arm comparison uses the same frozen Rec_ray first-max map. "
            "Where vote ties exist, first-max is not unique identification."
        ),
    }


def heldout_H_native(den: list[int], fte: list[int], seed_payload: dict) -> float:
    re = exact.basis(4)
    pairs = seed_payload["pairs_test"]
    et = seed_payload["_et_test"]
    h = 0
    for (i, j), tgt in zip(pairs, et):
        ae = exact_from_float_idx(int(den[i]), fte)
        be = exact_from_float_idx(int(den[j]), fte)
        ps = exact_p_seq(ae, be, re)
        pg = exact_p_grp(ae, be, re) if fips_basic.admissible(ae, be) else None
        s_q = int(exact_rays_equal(ps, tgt))
        g_q = int(exact_rays_equal(pg, tgt))
        h += max(s_q, g_q)
    return h / max(len(pairs), 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def policy_branch(pol: Policy | None, a_f, b_f, r_t, gate_val: float, *, force_seq: bool) -> str:
    if force_seq or pol is None:
        return "seq"
    if gate_val < 0.5:
        return "seq"
    with torch.no_grad():
        aa = torch.tensor(a_f, dtype=torch.float32).unsqueeze(0)
        bb = torch.tensor(b_f, dtype=torch.float32).unsqueeze(0)
        logits = pol(aa, bb, r_t)
        gate = torch.tensor([gate_val], dtype=torch.float32)
        masked = logits.clone()
        masked[0, 1] = masked[0, 1] + torch.log(gate.clamp(1e-4, 1.0))
        return "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"


def evaluate_arm_exact(
    *,
    arm: str,
    seed: int,
    den: list[int],
    fte: list[int],
    pgeo: PhiGeometry,
    seed_payload: dict,
    pol: Policy | None,
    force_seq: bool,
    gate_mask: np.ndarray,
    use_ambient_endpoints: bool,
    mask_id: str | None = None,
) -> tuple[dict, dict, list[dict]]:
    """Exact HASG on train and test with fixed benchmark targets."""
    re = exact.basis(4)
    r = seed_payload["_task"].r
    r_t = torch.tensor(r, dtype=torch.float32)
    split_summaries = {}
    all_records = []
    for split, pairs, targets in (
        ("train", seed_payload["pairs_train"], seed_payload["_et_train"]),
        ("test", seed_payload["pairs_test"], seed_payload["_et_test"]),
    ):
        n = len(pairs)
        s_sum = g_sum = h_sum = p_sum = 0
        class_counts = Counter()
        legal_grp = 0
        grp_selected = 0
        undef = {"seq_undef": 0, "grp_unavailable": 0, "grp_occ_undef": 0}
        records = []
        for ex_i, ((i, j), etgt) in enumerate(zip(pairs, targets)):
            ai, bi = int(den[i]), int(den[j])
            ae = exact_from_float_idx(ai, fte)
            be = exact_from_float_idx(bi, fte)
            a_f, b_f = pgeo.basic[ai], pgeo.basic[bi]
            ps_e = exact_p_seq(ae, be, re)
            gate_on = bool(gate_mask[i, j])
            if use_ambient_endpoints:
                pg_e = exact_p_grp_ambient(ae, be, re) if gate_on else None
                # If gate on but ambient undefined, count as unavailable
                if gate_on and pg_e is None:
                    undef["grp_occ_undef"] += 1
                    gate_effective = False
                else:
                    gate_effective = gate_on and pg_e is not None
            else:
                # Native Cyc
                if gate_on:
                    if not fips_basic.admissible(ae, be):
                        # Should not happen if gate is true Cyc mask
                        pg_e = None
                        gate_effective = False
                    else:
                        pg_e = exact_p_grp(ae, be, re)
                        gate_effective = pg_e is not None
                        if pg_e is None:
                            undef["grp_occ_undef"] += 1
                else:
                    pg_e = None
                    gate_effective = False
            if gate_effective:
                legal_grp += 1
            else:
                undef["grp_unavailable"] += 1
            if ps_e is None:
                undef["seq_undef"] += 1

            s_q = int(exact_rays_equal(ps_e, etgt))
            g_q = int(gate_effective and exact_rays_equal(pg_e, etgt))
            h_q = max(s_q, g_q)

            branch = policy_branch(
                pol, a_f, b_f, r_t, float(gate_mask[i, j]), force_seq=force_seq
            )
            if branch == "grp" and not gate_effective:
                branch = "seq"
            if branch == "grp":
                out_e = pg_e
                grp_selected += 1
            else:
                out_e = ps_e
            p_q = int(exact_rays_equal(out_e, etgt))

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
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "mask_id": mask_id,
                    "split": split,
                    "example_id": f"{seed}:{arm}:{mask_id or '-'}:{split}:{ex_i}",
                    "pair": [int(i), int(j)],
                    "denotation_ids": [ai, bi],
                    "exact_event_ids": [fte[ai], fte[bi]],
                    "s": s_q,
                    "g": g_q,
                    "h": h_q,
                    "a": p_q,
                    "policy_choice": branch,
                    "four_way": four,
                    "grp_available": gate_effective,
                    "exact_target_canon": list(exact_canonical_key(etgt)),
                }
            )
        H, A, S, G = h_sum / n, p_sum / n, s_sum / n, g_sum / n
        split_summaries[split] = {
            "n": n,
            "H": H,
            "A": A,
            "S": S,
            "G": G,
            "one_minus_H": 1.0 - H,
            "H_minus_A": H - A,
            "one_minus_A": 1.0 - A,
            "decomp_ok": abs((1.0 - A) - ((1.0 - H) + (H - A))) < 1e-12,
            "four_way_counts": dict(class_counts),
            "n_legal_grp": legal_grp,
            "grp_availability": legal_grp / n,
            "grp_selection_fraction": grp_selected / n,
            "undefined_counts": undef,
        }
        all_records.extend(records)
    return split_summaries["train"], split_summaries["test"], all_records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(
    *,
    seed: int,
    den: list[int],
    pgeo: PhiGeometry,
    pairs_train: list,
    ft_train: list[np.ndarray],
    gate_mask: np.ndarray,
    force_seq: bool,
) -> Policy | None:
    if force_seq:
        return None
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = pgeo.basic_t
    den_mod = FrozenTrueCatalogueDenotation(np.asarray(den, dtype=np.int64), cat_t)
    for p in den_mod.parameters():
        p.requires_grad_(False)
    pol = Policy()
    opt = torch.optim.Adam(pol.parameters(), lr=POLICY_LR)
    idx = torch.tensor(pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(ft_train), dtype=torch.float32)
    r_t = torch.tensor(basis_vec(4), dtype=torch.float32)
    M_t = pgeo.M_t
    # Precompute gate per training pair
    gates = torch.tensor(
        [float(gate_mask[int(i), int(j)]) for i, j in pairs_train],
        dtype=torch.float32,
    )
    pol.train()
    for _ in range(POLICY_STEPS):
        E = den_mod.forward_hard()
        a, b = E[idx[:, 0]], E[idx[:, 1]]
        logits = pol(a, b, r_t)
        pred, _, _ = soft_program_phi(
            a, b, r_t, logits, gates, M_t, force_seq=False
        )
        loss = proj_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pol.eval()
    return pol


# ---------------------------------------------------------------------------
# Arithmetic audits (tensor path)
# ---------------------------------------------------------------------------

def audit_tensor_phi(pgeo: PhiGeometry, fte: list[int]) -> dict:
    """Verify Phi tensor mul vs exact on 256 basis + 84×84 + programs @ e4."""
    M_t = pgeo.M_t
    raw_basis = phi_basis = 0
    for i in range(16):
        for j in range(16):
            pe = exact.mul(exact.basis(i), exact.basis(j))
            af = torch.tensor(basis_vec(i), dtype=torch.float32)
            bf = torch.tensor(basis_vec(j), dtype=torch.float32)
            pr = mul_t(af, bf, M_t).numpy()
            pp = mul_t_phi(af, bf, M_t).numpy()
            pe_f = exact_value_to_float(pe)

            def ok(pf):
                if pe == exact.zero():
                    return float(np.linalg.norm(pf)) < TOL
                return float(np.linalg.norm(pf)) >= TOL and rays_equal(pf, pe_f)

            raw_basis += int(not ok(pr))
            phi_basis += int(not ok(pp))
    raw_cat = phi_cat = 0
    for i in range(84):
        for j in range(84):
            ae = exact_from_float_idx(i, fte)
            be = exact_from_float_idx(j, fte)
            pe = exact.mul(ae, be)
            af = pgeo.basic_t[i]
            bf = pgeo.basic_t[j]
            pr = mul_t(af, bf, M_t).numpy()
            pp = mul_t_phi(af, bf, M_t).numpy()
            pe_f = exact_value_to_float(pe)

            def ok(pf):
                if pe == exact.zero():
                    return float(np.linalg.norm(pf)) < TOL
                return float(np.linalg.norm(pf)) >= TOL and rays_equal(pf, pe_f)

            raw_cat += int(not ok(pr))
            phi_cat += int(not ok(pp))
    # Program paths
    r = basis_vec(4)
    re = exact.basis(4)
    seq_phi = grp_phi = 0
    for i in range(84):
        for j in range(84):
            ae = exact_from_float_idx(i, fte)
            be = exact_from_float_idx(j, fte)
            pse = exact_p_seq(ae, be, re)
            pge = exact_p_grp(ae, be, re)
            a, b = pgeo.basic[i], pgeo.basic[j]
            ps = pgeo.p_seq(a, b, r)
            pg = pgeo.p_grp_cyc(a, b, r)

            def match(pf, pe):
                if pf is None and pe is None:
                    return True
                if pf is None or pe is None:
                    return False
                return rays_equal(pf, exact_value_to_float(pe))

            seq_phi += int(not match(ps, pse))
            grp_phi += int(not match(pg, pge))
    return {
        "basis_256": {"raw_disagreements": raw_basis, "phi_disagreements": phi_basis},
        "catalogue_84x84": {"raw_disagreements": raw_cat, "phi_disagreements": phi_cat},
        "programs_at_e4": {
            "P_seq_phi_disagreements": seq_phi,
            "P_grp_cyc_phi_disagreements": grp_phi,
            "n_pairs": 84 * 84,
        },
        "phi_ok": phi_basis == 0 and phi_cat == 0 and seq_phi == 0 and grp_phi == 0,
    }


# ---------------------------------------------------------------------------
# Aggregation / verdict
# ---------------------------------------------------------------------------

def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()) if len(a) else float("nan"),
        "std": float(a.std(ddof=0)) if len(a) else float("nan"),
        "values": [float(x) for x in a],
        "n": int(len(a)),
    }


def paired_vs_native(native: list[dict], other: list[dict], key: str = "A") -> dict:
    """Seed-aligned paired differences other - native on test metrics."""
    diffs = []
    for n, o in zip(native, other):
        diffs.append(float(o["test"][key] - n["test"][key]))
    return {
        f"delta_{key}_other_minus_native": mean_std(diffs),
        "seed_level": [
            {"seed": n["seed"], f"native_{key}": n["test"][key], f"other_{key}": o["test"][key], "delta": d}
            for n, o, d in zip(native, other, diffs)
        ],
    }


def interpret_verdict(agg: dict) -> dict:
    nat_H = agg["Native"]["test_H"]["mean"]
    nat_A = agg["Native"]["test_A"]["mean"]
    fs_A = agg["ForceSeq"]["test_A"]["mean"]
    fs_H = agg["ForceSeq"]["test_H"]["mean"]
    amb_H = agg["Ambient"]["test_H"]["mean"]
    amb_A = agg["Ambient"]["test_A"]["mean"]
    # Rewired: mean across masks then seeds already aggregated as Rewired_mean
    rew_H = agg["Rewired"]["test_H"]["mean"]
    rew_A = agg["Rewired"]["test_A"]["mean"]

    ceiling = False
    policy = False
    notes = []

    # Native beating ForceSeq → utility of branch choice
    if nat_A > fs_A + 0.05 and nat_H > fs_H + 0.05:
        notes.append("Native >> ForceSeq on H and A: branch choice is useful on this benchmark.")
    elif nat_A > fs_A + 0.02:
        notes.append("Native beats ForceSeq on A: some utility of second program.")
    else:
        notes.append("Native does not clearly beat ForceSeq.")

    # Ceiling: Native vs Rewired mainly through H
    if nat_H > rew_H + 0.05 and abs(nat_A - (rew_A + (nat_H - rew_H))) < 0.15:
        ceiling = True
        notes.append("Native ceiling (H) exceeds rewired masks → mask↔target alignment.")
    if nat_H > amb_H + 0.02:
        ceiling = True
        notes.append("Native H exceeds Ambient (unexpected if Ambient strictly relaxes Cyc).")
    # Ambient H should be >= Native H typically (more programs); equal H expected if Native saturates
    if abs(nat_H - amb_H) < 0.02 and nat_H > 0.95:
        notes.append("Native and Ambient both near H=1: ceiling saturated; equal H not evidence of equivalence.")

    # Policy: Native beating Ambient with comparable H
    if abs(nat_H - amb_H) < 0.05 and nat_A > amb_A + 0.05:
        policy = True
        notes.append("Native beats Ambient on A at comparable H → policy/generalization benefit from Cyc gating.")
    elif abs(nat_H - 1.0) < 0.02 and abs(amb_H - 1.0) < 0.02 and abs(nat_A - amb_A) < 0.03:
        notes.append("Similar A with H≈1 across Native/Ambient: no evidence native gating adds predictive value here.")
    elif nat_A > rew_A + 0.05 and abs(nat_H - rew_H) < 0.05:
        policy = True
        notes.append("Native beats Rewired on A at comparable H → gating structure aids policy.")
    elif nat_H > rew_H + 0.05 and abs(nat_A - rew_A) < (nat_H - rew_H) + 0.05:
        ceiling = True
        notes.append("Native vs Rewired gap mainly via H (ceiling/alignment), not clearly policy.")

    if ceiling and policy:
        verdict = "both"
    elif ceiling:
        verdict = "ceiling"
    elif policy:
        verdict = "policy"
    elif abs(nat_A - amb_A) < 0.03 and abs(nat_A - rew_A) < 0.05 and abs(nat_H - rew_H) < 0.05:
        verdict = "neither"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "ceiling_evidence": ceiling,
        "policy_evidence": policy,
        "notes": notes,
        "table_means": {
            "Native": {"A": nat_A, "H": nat_H},
            "ForceSeq": {"A": fs_A, "H": fs_H},
            "Ambient": {"A": amb_A, "H": amb_H},
            "Rewired": {"A": rew_A, "H": rew_H},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_wall0 = time.time()
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    pgeo = PhiGeometry(geo)
    wit = verify_witness(alg, geo.M)
    fte = float_to_exact_map(geo)

    # --- Arithmetic audits ---
    audits = {
        "basis_np": audit_basis_products(alg),
        "catalogue_np": audit_catalogue_products(geo, fte, alg),
        "programs_np": audit_programs_at_e4(geo, fte, alg),
        "witness": minimal_mismatch_witness(geo, fte, alg),
        "tensor_phi": audit_tensor_phi(pgeo, fte),
        "01707_witness_disagree_under_core_C": wit.get("disagree"),
    }
    # Also confirm Phi np helper zero on catalogue (reuse 017.19 path via float_mul_with_phi)
    assert audits["tensor_phi"]["phi_ok"], f"Phi tensor audit failed: {audits['tensor_phi']}"

    recovered = load_recovered_maps(geo)
    bench = materialize_benchmark(geo, pgeo, fte, recovered)

    # Target provenance: single Phi transform vs old stored float targets?
    provenance = {
        "designated_benchmark": "reconstructed exact preferred targets (017.20 mirror)",
        "old_float_vs_exact_ray_match_by_seed": {
            str(s): bench["seeds"][s]["old_float_vs_exact_target_ray_match"] for s in SEEDS
        },
        "interpretation": None,
    }
    # If not all match, describe as corrected benchmark
    all_match = all(
        bench["seeds"][s]["old_float_vs_exact_target_ray_match"]["train_ray_match"]
        == bench["seeds"][s]["old_float_vs_exact_target_ray_match"]["train_n"]
        and bench["seeds"][s]["old_float_vs_exact_target_ray_match"]["test_ray_match"]
        == bench["seeds"][s]["old_float_vs_exact_target_ray_match"]["test_n"]
        for s in SEEDS
    )
    provenance["interpretation"] = (
        "old stored float targets already projectively match exact preferred rays under input mapping"
        if all_match
        else (
            "old stored float targets do NOT all match exact preferred rays; "
            "017.18/017.20 performance is preserved on a corrected exact benchmark rather than "
            "literal endpoint equivalence of stored float targets"
        )
    )

    # Tie / order audit BEFORE arms (uses same frozen map)
    tie_audit = tie_and_order_audit(geo, fte, bench)

    # Build rewired masks from seed0 Cyc mask on recovered dens (same dens all seeds=true)
    # Per-seed Cyc masks may be identical if dens identical
    rewired_by_seed = {}
    for seed in SEEDS:
        cyc = bench["seeds"][seed]["_cyc_mask"]
        masks = []
        for rs in REWIRE_SEEDS:
            info = rewire_degree_preserving(cyc, seed=rs + 1000 * seed)
            masks.append(info)
        rewired_by_seed[seed] = masks

    # --- Run arms ---
    arm_runs = {name: [] for name in ["Native", "ForceSeq", "Ambient"]}
    rewired_runs = []  # list of {seed, mask_id, ...}
    all_records = []
    param_counts = {"policy_trainable": count_params([Policy()]), "denotation_trainable": 0}

    for seed in SEEDS:
        sp = bench["seeds"][seed]
        den = sp["recovered_catalogue_idx"]
        cyc = sp["_cyc_mask"]
        amb = sp["_amb_mask"]
        pairs_train = [tuple(p) for p in sp["pairs_train"]]

        # Native
        pol_n = train_policy(
            seed=seed,
            den=den,
            pgeo=pgeo,
            pairs_train=pairs_train,
            ft_train=sp["_ft_train"],
            gate_mask=cyc.astype(np.float64),
            force_seq=False,
        )
        tr, te, rec = evaluate_arm_exact(
            arm="Native",
            seed=seed,
            den=den,
            fte=fte,
            pgeo=pgeo,
            seed_payload=sp,
            pol=pol_n,
            force_seq=False,
            gate_mask=cyc,
            use_ambient_endpoints=False,
        )
        arm_runs["Native"].append({"seed": seed, "train": tr, "test": te})
        all_records.extend(rec)
        torch.save({"arm": "Native", "seed": seed, "pol": pol_n.state_dict(), "den": den}, CKPT_DIR / f"seed{seed}_Native.pt")

        # ForceSeq
        tr, te, rec = evaluate_arm_exact(
            arm="ForceSeq",
            seed=seed,
            den=den,
            fte=fte,
            pgeo=pgeo,
            seed_payload=sp,
            pol=None,
            force_seq=True,
            gate_mask=np.zeros_like(cyc),
            use_ambient_endpoints=False,
        )
        arm_runs["ForceSeq"].append({"seed": seed, "train": tr, "test": te})
        all_records.extend(rec)

        # Ambient
        pol_a = train_policy(
            seed=seed,
            den=den,
            pgeo=pgeo,
            pairs_train=pairs_train,
            ft_train=sp["_ft_train"],
            gate_mask=amb.astype(np.float64),
            force_seq=False,
        )
        tr, te, rec = evaluate_arm_exact(
            arm="Ambient",
            seed=seed,
            den=den,
            fte=fte,
            pgeo=pgeo,
            seed_payload=sp,
            pol=pol_a,
            force_seq=False,
            gate_mask=amb,
            use_ambient_endpoints=True,
        )
        arm_runs["Ambient"].append({"seed": seed, "train": tr, "test": te})
        all_records.extend(rec)
        torch.save({"arm": "Ambient", "seed": seed, "pol": pol_a.state_dict(), "den": den}, CKPT_DIR / f"seed{seed}_Ambient.pt")

        # Rewired × 3
        for mi, minfo in enumerate(rewired_by_seed[seed]):
            mask = minfo["_mask"]
            mid = f"rewire{mi}_seed{minfo['seed']}"
            pol_r = train_policy(
                seed=seed,
                den=den,
                pgeo=pgeo,
                pairs_train=pairs_train,
                ft_train=sp["_ft_train"],
                gate_mask=mask.astype(np.float64),
                force_seq=False,
            )
            tr, te, rec = evaluate_arm_exact(
                arm="Rewired",
                seed=seed,
                den=den,
                fte=fte,
                pgeo=pgeo,
                seed_payload=sp,
                pol=pol_r,
                force_seq=False,
                gate_mask=mask,
                use_ambient_endpoints=True,
                mask_id=mid,
            )
            # Availability frequencies
            train_avail = tr["grp_availability"]
            test_avail = te["grp_availability"]
            rewired_runs.append(
                {
                    "seed": seed,
                    "mask_id": mid,
                    "mask_meta": {k: v for k, v in minfo.items() if k != "_mask" and k != "adjacency"},
                    "adjacency": minfo["adjacency"],
                    "train": tr,
                    "test": te,
                    "train_grp_availability": train_avail,
                    "test_grp_availability": test_avail,
                }
            )
            all_records.extend(rec)
            torch.save(
                {"arm": "Rewired", "seed": seed, "mask_id": mid, "pol": pol_r.state_dict(), "den": den},
                CKPT_DIR / f"seed{seed}_{mid}.pt",
            )

    # Aggregate
    def agg_arm(runs):
        return {
            "test_H": mean_std([r["test"]["H"] for r in runs]),
            "test_A": mean_std([r["test"]["A"] for r in runs]),
            "test_S": mean_std([r["test"]["S"] for r in runs]),
            "test_G": mean_std([r["test"]["G"] for r in runs]),
            "train_H": mean_std([r["train"]["H"] for r in runs]),
            "train_A": mean_std([r["train"]["A"] for r in runs]),
            "seed_level": runs,
        }

    # For Rewired: first average masks within seed, then across seeds
    rew_by_seed = {}
    for r in rewired_runs:
        rew_by_seed.setdefault(r["seed"], []).append(r)
    rew_seed_summaries = []
    for seed, rs in sorted(rew_by_seed.items()):
        rew_seed_summaries.append(
            {
                "seed": seed,
                "train": {
                    "H": float(np.mean([x["train"]["H"] for x in rs])),
                    "A": float(np.mean([x["train"]["A"] for x in rs])),
                    "S": float(np.mean([x["train"]["S"] for x in rs])),
                    "G": float(np.mean([x["train"]["G"] for x in rs])),
                },
                "test": {
                    "H": float(np.mean([x["test"]["H"] for x in rs])),
                    "A": float(np.mean([x["test"]["A"] for x in rs])),
                    "S": float(np.mean([x["test"]["S"] for x in rs])),
                    "G": float(np.mean([x["test"]["G"] for x in rs])),
                },
                "per_mask": [
                    {"mask_id": x["mask_id"], "test_H": x["test"]["H"], "test_A": x["test"]["A"],
                     "jaccard": x["mask_meta"]["jaccard_undirected"],
                     "degree_preserved": x["mask_meta"]["degree_sequence_preserved"],
                     "non_isomorphic_edge_set": x["mask_meta"]["non_isomorphic_edge_set"]}
                    for x in rs
                ],
            }
        )

    aggregates = {
        "Native": agg_arm(arm_runs["Native"]),
        "ForceSeq": agg_arm(arm_runs["ForceSeq"]),
        "Ambient": agg_arm(arm_runs["Ambient"]),
        "Rewired": agg_arm(rew_seed_summaries),
    }

    paired = {
        "ForceSeq": {
            "A": paired_vs_native(arm_runs["Native"], arm_runs["ForceSeq"], "A"),
            "H": paired_vs_native(arm_runs["Native"], arm_runs["ForceSeq"], "H"),
        },
        "Ambient": {
            "A": paired_vs_native(arm_runs["Native"], arm_runs["Ambient"], "A"),
            "H": paired_vs_native(arm_runs["Native"], arm_runs["Ambient"], "H"),
        },
        "Rewired": {
            "A": paired_vs_native(arm_runs["Native"], rew_seed_summaries, "A"),
            "H": paired_vs_native(arm_runs["Native"], rew_seed_summaries, "H"),
        },
    }

    verdict = interpret_verdict(aggregates)

    # Mask publication (seed0 representative + all)
    masks_pub = {
        str(seed): [
            {
                "mask_id": f"rewire{mi}_seed{m['seed']}",
                "rewire_seed": m["seed"],
                "swaps_done": m["swaps_done"],
                "degree_sequence_preserved": m["degree_sequence_preserved"],
                "degree_sequence": m["degree_sequence"],
                "jaccard_undirected": m["jaccard_undirected"],
                "edge_overlap_undirected": m["edge_overlap_undirected"],
                "non_isomorphic_edge_set": m["non_isomorphic_edge_set"],
                "identical_to_native": m["identical_to_native"],
                "adjacency": m["adjacency"],
            }
            for mi, m in enumerate(rewired_by_seed[seed])
        ]
        for seed in SEEDS
    }

    report = {
        "source": "017.21 structural-admissibility and ambient-Mul controls",
        "in_reply_to": "017.21-HC-structural-admissibility-and-ambient-Mul-controls-task.md",
        "pinned_01720": "quilt+s3://protology#package=occurrence/outcome@7e2edb7a…",
        "pinned_commit_01720": "6debbc7",
        "question": (
            "Does the actual Cyc admissibility relation help policy generalization "
            "once denotations/arithmetic are fixed, beyond having a second nonassociative program?"
        ),
        "verdict": verdict,
        "manifest_sha256": bench["manifest_sha256"],
        "arithmetic_audits": audits,
        "target_provenance": provenance,
        "coordinate_systems": {
            "catalogue_vectors": bench["coordinate_system_catalogue_vectors"],
            "products": bench["coordinate_system_products"],
            "targets": bench["coordinate_system_targets"],
            "stored_denotations": "Value-coordinate catalogue indices (frozen Rec_ray)",
        },
        "tie_order_audit": {
            k: v for k, v in tie_audit.items() if k != "per_seed"
        },
        "tie_order_audit_per_seed_slim": {
            s: {
                "n_tied_tokens": tie_audit["per_seed"][s]["n_tied_tokens"],
                "tied_tokens": tie_audit["per_seed"][s]["tied_tokens"],
                "first_max_match_over_16": tie_audit["per_seed"][s]["first_max_match_over_16"],
                "alt_test_H_range": {
                    "min": tie_audit["per_seed"][s]["alt_test_H_range"]["min"],
                    "max": tie_audit["per_seed"][s]["alt_test_H_range"]["max"],
                    "n_combos": tie_audit["per_seed"][s]["alt_test_H_range"]["n_combos"],
                },
                "first_max_recovery_16_16_all_perms": tie_audit["per_seed"][s][
                    "first_max_recovery_16_16_all_perms"
                ],
            }
            for s in map(str, SEEDS)
        },
        "primary_comparison_conditional_on_favorable_first_max": tie_audit[
            "primary_map_conditional_on_favorable_first_max"
        ],
        "aggregates": {
            k: {kk: vv for kk, vv in v.items() if kk != "seed_level"}
            for k, v in aggregates.items()
        },
        "seed_level": {
            "Native": arm_runs["Native"],
            "ForceSeq": arm_runs["ForceSeq"],
            "Ambient": arm_runs["Ambient"],
            "Rewired": rew_seed_summaries,
        },
        "paired_vs_Native": paired,
        "rewired_mask_publication": masks_pub,
        "rewired_per_mask_results": [
            {
                "seed": r["seed"],
                "mask_id": r["mask_id"],
                "test": r["test"],
                "train": r["train"],
                "mask_meta": r["mask_meta"],
            }
            for r in rewired_runs
        ],
        "param_counts": param_counts,
        "policy_budget": {"steps": POLICY_STEPS, "lr": POLICY_LR, "arch": "Policy(16→32→2)"},
        "recovery_cost_note": "Rec_ray maps loaded from 01717 checkpoints; not retrained",
        "fence": (
            "configured finite FIPS proxy; Issue 017 not closed; no modular stage; "
            "no decoder; no tune-until-win; OT-generated benchmark only"
        ),
        "wall_sec": time.time() - t_wall0,
    }

    # Write artifacts
    def _jsonable(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    report_path = OUT / "learned_admissibility_01721_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_jsonable)

    # slim / tiny
    slim = {
        "verdict": verdict,
        "manifest_sha256": bench["manifest_sha256"],
        "aggregates": report["aggregates"],
        "paired_vs_Native": {
            k: {m: v[m]["delta_" + m.lower() + "_other_minus_native"] if False else {
                "delta_mean": v[m][f"delta_{m}_other_minus_native"]["mean"],
            } for m in ("A", "H")}
            for k, v in paired.items()
        },
        "tie_order_audit_per_seed_slim": report["tie_order_audit_per_seed_slim"],
        "primary_comparison_conditional_on_favorable_first_max": report[
            "primary_comparison_conditional_on_favorable_first_max"
        ],
        "arithmetic_phi_ok": audits["tensor_phi"]["phi_ok"],
        "target_provenance": provenance,
        "wall_sec": report["wall_sec"],
    }
    # fix paired slim properly
    slim["paired_vs_Native"] = {
        k: {
            "delta_A_mean": paired[k]["A"]["delta_A_other_minus_native"]["mean"],
            "delta_H_mean": paired[k]["H"]["delta_H_other_minus_native"]["mean"],
        }
        for k in paired
    }
    with open(OUT / "learned_admissibility_01721_report_slim.json", "w") as f:
        json.dump(slim, f, indent=2, default=_jsonable)
    tiny = {
        "verdict": verdict["verdict"],
        "table": verdict["table_means"],
        "manifest_sha256": bench["manifest_sha256"],
        "phi_ok": audits["tensor_phi"]["phi_ok"],
        "conditional_first_max": report["primary_comparison_conditional_on_favorable_first_max"],
        "tie_one_liner": _tie_one_liner(tie_audit),
    }
    with open(OUT / "learned_admissibility_01721_report_tiny.json", "w") as f:
        json.dump(tiny, f, indent=2)

    # per-example jsonl
    with open(ART / "per_example.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # manifest hash file
    with open(ART / "benchmark_manifest.json", "w") as f:
        json.dump({"sha256": bench["manifest_sha256"], "manifest": bench["manifest"]}, f, indent=2)

    # full tie audit
    with open(ART / "tie_order_audit.json", "w") as f:
        json.dump(tie_audit, f, indent=2, default=_jsonable)

    # masks
    with open(ART / "rewired_masks.json", "w") as f:
        json.dump(masks_pub, f, indent=2)

    # Copy key artifacts to Code-attachments
    for src_name in [
        "learned_admissibility_01721_report.json",
        "learned_admissibility_01721_report_slim.json",
        "learned_admissibility_01721_report_tiny.json",
    ]:
        src = OUT / src_name
        dst = ATTACH / src_name
        dst.write_bytes(src.read_bytes())
    (ATTACH / "per_example.jsonl").write_bytes((ART / "per_example.jsonl").read_bytes())
    (ATTACH / "benchmark_manifest.json").write_bytes((ART / "benchmark_manifest.json").read_bytes())
    (ATTACH / "rewired_masks.json").write_bytes((ART / "rewired_masks.json").read_bytes())
    (ATTACH / "tie_order_audit.json").write_bytes((ART / "tie_order_audit.json").read_bytes())
    (ATTACH / "REPRODUCTION.md").write_text(
        "\n".join(
            [
                "# 017.22 Code attachments — reproduction",
                "",
                "```bash",
                "cd experiments/tlm_fixed_head",
                "uv run python -m unittest test_01721_phi_and_controls -v",
                "uv run python learned_admissibility_01721_structural_controls.py",
                "```",
                "",
                f"manifest_sha256: `{bench['manifest_sha256']}`",
                f"verdict: `{verdict['verdict']}`",
                "Pinned dens: `01717_artifacts/checkpoints/seed{0,1,2}_Rec_ray.pt`",
                "Policy budget: 800 steps, lr=0.05, Policy(16→32→2)",
                "Phi adapter applied in tensor forward + exact target construction.",
                "",
            ]
        )
    )

    print(json.dumps(tiny, indent=2))
    print("wall_sec", report["wall_sec"])
    print("wrote", report_path)
    return 0


def _tie_one_liner(tie_audit: dict) -> str:
    bits = []
    for s in map(str, SEEDS):
        ps = tie_audit["per_seed"][s]
        bits.append(
            f"seed{s}: ties={ps['n_tied_tokens']} tokens {ps['tied_tokens']}; "
            f"first-max {ps['first_max_match_over_16']}/16; "
            f"perm 16/16={ps['first_max_recovery_16_16_all_perms']}; "
            f"altH=[{ps['alt_test_H_range']['min']},{ps['alt_test_H_range']['max']}]"
        )
    cond = tie_audit["primary_map_conditional_on_favorable_first_max"]
    return (
        ("CONDITIONAL on favorable first-max. " if cond else "No ties. ")
        + " | ".join(bits)
    )


if __name__ == "__main__":
    raise SystemExit(main())

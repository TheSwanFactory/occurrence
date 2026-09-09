#!/usr/bin/env python3
"""017.15 recovered-geometry ceiling and branch-selection shortfall.

Evaluation-only diagnostic on frozen hard denotations from reproduced 017.13/017.14
checkpoints (same settings; no hyperparameter changes). No saved checkpoints existed
in-repo after 017.14, so this script re-runs the original training deterministically,
saves state dicts, then audits:

  P_seq = Occ(a, Occ(b,r))
  P_grp = Occ(Cyc(a,b), r)   # only if hard finite FIPS Cyc passes
  s_q, g_q, h_q=max(s,g), p_q = actual hard policy success

Partition: neither / seq-only / grp-only / both. Off-domain grp never counted
attainable. All task examples remain in the denominator.

Fence: configured finite FIPS proxy; Issue 017 not closed; no modular grokking claim.
"""
from __future__ import annotations

import copy
import hashlib
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
    build_native_task,
    densest_vocab_indices,
    exact_eval,
    mul_np,
    oracle_eval,
    projective_key,
    rays_equal,
    soft_program,
    true_denotations,
    verify_witness,
    proj_loss,
    count_params,
)
from learned_admissibility_01711_discrete import (
    GUMBEL_TAU,
    LR_DISC,
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

from topographo.ssd import exact, fips_basic, projective

OUT = Path(__file__).resolve().parent
ART = OUT / "01715_artifacts"
ART.mkdir(parents=True, exist_ok=True)

# Reproduction targets from committed 017.13 report (exact_success rounded display).
REPRO_TARGETS = {
    (0, "A_cur"): (0.267123, 0.191781),
    (0, "A_cur_freeze"): (0.416107, 0.4),
    (0, "A_joint"): (0.020833, 0.0),
    (0, "B_freeze_true_den_learn_pol"): (1.0, 0.92),
    (0, "C_learn_discrete_den_force_seq"): (0.322148, 0.346667),
    (1, "A_cur"): (0.321168, 0.21519),
    (1, "A_cur_freeze"): (0.514085, 0.375),
    (1, "A_joint"): (0.08209, 0.075949),
    (1, "B_freeze_true_den_learn_pol"): (0.986014, 0.864198),
    (1, "C_learn_discrete_den_force_seq"): (0.450704, 0.3375),
    (2, "A_cur"): (0.165517, 0.151515),
    (2, "A_cur_freeze"): (0.546667, 0.450704),
    (2, "A_joint"): (0.006711, 0.028986),
    (2, "B_freeze_true_den_learn_pol"): (1.0, 0.944444),
    (2, "C_learn_discrete_den_force_seq"): (0.446667, 0.422535),
}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _state_hash(obj) -> str:
    """Stable hash of a torch state_dict / numpy array via torch save bytes."""
    if isinstance(obj, dict):
        # tensor state dict
        bio = torch.save if False else None
        import io

        buf = io.BytesIO()
        torch.save(obj, buf)
        return _sha256_bytes(buf.getvalue())
    if isinstance(obj, np.ndarray):
        return _sha256_bytes(obj.astype(np.float64).tobytes() + str(obj.shape).encode())
    raise TypeError(type(obj))


def _float_to_exact_map(geo: Geometry) -> list[int]:
    """Map float catalogue index -> exact fips_basic.EVENTS index (projective)."""
    events = fips_basic.EVENTS

    def float_from_exact(v):
        arr = np.zeros(16, dtype=np.float64)
        for i, c in enumerate(v):
            arr[i] = float(c)
        return arr

    exact_keys = {projective_key(float_from_exact(e)): i for i, e in enumerate(events)}
    out = []
    for b in geo.basic:
        k = projective_key(b)
        j = exact_keys[k]
        out.append(j)
    return out


def _exact_value_from_float_basic(v: np.ndarray, float_to_exact: list[int], geo: Geometry) -> exact.Value:
    """Recover exact catalogue Event for a hard float denotation that matches a basic ray."""
    k = projective_key(v)
    for i, b in enumerate(geo.basic):
        if projective_key(b) == k:
            return fips_basic.EVENTS[float_to_exact[i]]
    raise ValueError("denotation not a catalogue basic ray")


def _exact_occ(e: exact.Value, r: exact.Value) -> exact.Value | None:
    prod = exact.mul(e, r)
    if prod == exact.zero():
        return None
    return prod


def _exact_p_seq(a: exact.Value, b: exact.Value, r: exact.Value) -> exact.Value | None:
    inner = _exact_occ(b, r)
    if inner is None:
        return None
    return _exact_occ(a, inner)


def _exact_p_grp(a: exact.Value, b: exact.Value, r: exact.Value) -> exact.Value | None:
    if not fips_basic.admissible(a, b):
        return None
    # Cyc product via exact mul; must be nonzero basic Event ray under finite proxy
    zw = exact.mul(a, b)
    if zw == exact.zero():
        return None
    return _exact_occ(zw, r)


def _exact_rays_equal(u: exact.Value | None, v: exact.Value | None) -> bool:
    if u is None or v is None:
        return False
    if u == exact.zero() or v == exact.zero():
        return False
    return projective.equivalent(u, v)


def _policy_choice(
    pol: Policy | None,
    geo: Geometry,
    r_t: torch.Tensor,
    a: np.ndarray,
    b: np.ndarray,
    hard_adm: bool,
    *,
    force_seq: bool,
    has_trained_policy: bool,
) -> tuple[str, str | None]:
    """Return (branch, fallback_reason)."""
    if force_seq or not has_trained_policy:
        return "seq", ("force_seq" if force_seq else "no_trained_policy")
    if not hard_adm:
        return "seq", "fallback_not_adm"
    assert pol is not None
    with torch.no_grad():
        aa = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        bb = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        logits = pol(aa, bb, r_t)
        soft = geo.soft_cyc_score(aa, bb)
        masked = logits.clone()
        masked[0, 1] = masked[0, 1] + torch.log(soft.clamp(1e-4, 1.0))
        branch = "grp" if bool(masked[0, 1] > masked[0, 0]) else "seq"
    return branch, None


def evaluate_ceiling_split(
    *,
    geo: Geometry,
    den_np: np.ndarray,
    den_idx: list[int] | None,
    task,
    pairs,
    targets,
    split: str,
    seed: int,
    arm: str,
    phase: str,
    pol: Policy | None,
    force_seq: bool,
    has_trained_policy: bool,
    float_to_exact: list[int],
    example_id_offset: int = 0,
) -> tuple[dict, list[dict]]:
    r = task.r
    r_t = torch.tensor(r, dtype=torch.float32)
    r_exact = exact.basis(4)  # e4 unnormalized; projective equivalent to float e4

    records: list[dict] = []
    s_sum = g_sum = h_sum = p_sum = 0
    class_counts = {"neither": 0, "seq_only": 0, "grp_only": 0, "both": 0}
    n = len(pairs)
    # policy diagnostics
    grp_choice_all = 0
    legal_grp_n = 0
    grp_choice_on_legal = 0
    grp_only_n = 0
    grp_only_recall = 0
    seq_only_n = 0
    seq_only_avoid_grp = 0
    both_n = 0
    both_grp = 0
    h_pos = 0
    p_on_h = 0
    exact_disagree = 0
    exact_compared = 0

    for ex_i, ((i, j), tgt) in enumerate(zip(pairs, targets)):
        a, b = den_np[i], den_np[j]
        hard_adm, zw = geo.cyc_hard(a, b)
        ps = geo.p_seq(a, b, r)
        # Never evaluate off-domain grp as attainable
        if hard_adm:
            pg = geo.p_grp(a, b, r)
            grp_legal = True
            grp_undef_reason = None
        else:
            pg = None
            grp_legal = False
            grp_undef_reason = "cyc_hard_failed"

        seq_legal = True
        seq_undef_reason = None
        if ps is None:
            seq_legal = False
            # distinguish which Occ failed
            inner = geo.occ(b, r)
            if inner is None:
                seq_undef_reason = "occ_b_r_undefined"
            else:
                seq_undef_reason = "occ_a_inner_undefined"

        if grp_legal and pg is None:
            grp_undef_reason = "occ_cyc_r_undefined"
            # still "legal Cyc" but Occ failed — g_q=0; not attainable success
            pass

        s_q = int(ps is not None and rays_equal(ps, tgt))
        g_q = int(grp_legal and pg is not None and rays_equal(pg, tgt))
        h_q = max(s_q, g_q)

        branch, fallback = _policy_choice(
            pol,
            geo,
            r_t,
            a,
            b,
            hard_adm,
            force_seq=force_seq,
            has_trained_policy=has_trained_policy,
        )
        if branch == "grp":
            # only if legal; policy_fn mirrors exact_eval masking
            out = pg if hard_adm else None
            if not hard_adm:
                # should not happen due to fallback
                branch = "seq"
                fallback = "policy_grp_but_not_adm"
                out = ps
        else:
            out = ps
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

        if hard_adm:
            legal_grp_n += 1
        if branch == "grp":
            grp_choice_all += 1
            if hard_adm:
                grp_choice_on_legal += 1
        if four == "grp_only":
            grp_only_n += 1
            if branch == "grp":
                grp_only_recall += 1
        if four == "seq_only":
            seq_only_n += 1
            if branch != "grp":
                seq_only_avoid_grp += 1
        if four == "both":
            both_n += 1
            if branch == "grp":
                both_grp += 1
        if h_q:
            h_pos += 1
            p_on_h += p_q

        # Exact rational cross-check (catalogue indices → exact Events)
        exact_s = exact_g = None
        exact_match_s = exact_match_g = None
        try:
            if den_idx is not None:
                ai = int(den_idx[i])
                bi = int(den_idx[j])
                ae = fips_basic.EVENTS[float_to_exact[ai]]
                be = fips_basic.EVENTS[float_to_exact[bi]]
            else:
                ae = _exact_value_from_float_basic(a, float_to_exact, geo)
                be = _exact_value_from_float_basic(b, float_to_exact, geo)
            # target: from true denotations preferred path — recover via float key match
            # Build exact target by matching float tgt to an Occ product under true ε*
            # For disagreement counting we compare exact endpoint equality to float decision.
            ps_e = _exact_p_seq(ae, be, r_exact)
            if hard_adm and fips_basic.admissible(ae, be):
                pg_e = _exact_p_grp(ae, be, r_exact)
            else:
                pg_e = None
            # Compare exact vs float success against tgt by projecting exact→float
            def exact_to_float(v: exact.Value | None):
                if v is None:
                    return None
                arr = np.zeros(16, dtype=np.float64)
                for ii, c in enumerate(v):
                    arr[ii] = float(c)
                return arr

            ps_ef = exact_to_float(ps_e)
            pg_ef = exact_to_float(pg_e)
            exact_s = int(ps_ef is not None and rays_equal(ps_ef, tgt))
            exact_g = int(
                hard_adm
                and fips_basic.admissible(ae, be)
                and pg_ef is not None
                and rays_equal(pg_ef, tgt)
            )
            exact_compared += 1
            if exact_s != s_q or exact_g != g_q:
                exact_disagree += 1
            exact_match_s = exact_s == s_q
            exact_match_g = exact_g == g_q
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            exact_match_s = f"error:{type(exc).__name__}"
            exact_match_g = exact_match_s

        neither_reason = None
        if four == "neither":
            reasons = []
            if not seq_legal:
                reasons.append(f"seq_undef:{seq_undef_reason}")
            elif not s_q:
                reasons.append("seq_wrong_endpoint")
            if not grp_legal:
                reasons.append(f"grp_unavailable:{grp_undef_reason}")
            elif pg is None:
                reasons.append(f"grp_undef:{grp_undef_reason}")
            elif not g_q:
                reasons.append("grp_wrong_endpoint")
            neither_reason = "+".join(reasons)

        rec = {
            "seed": seed,
            "arm": arm,
            "phase": phase,
            "split": split,
            "example_id": f"{seed}:{arm}:{phase}:{split}:{ex_i}",
            "pair": [int(i), int(j)],
            "hard_denotation_ids": [
                int(den_idx[i]) if den_idx is not None else None,
                int(den_idx[j]) if den_idx is not None else None,
            ],
            "target_ray_key": list(projective_key(tgt)),
            "seq_legal": bool(seq_legal),
            "seq_undefined_reason": seq_undef_reason,
            "grp_legal": bool(grp_legal),
            "grp_undefined_reason": grp_undef_reason,
            "seq_endpoint_key": list(projective_key(ps)) if ps is not None else None,
            "grp_endpoint_key": list(projective_key(pg)) if pg is not None else None,
            "s_q": s_q,
            "g_q": g_q,
            "h_q": h_q,
            "policy_choice": branch,
            "policy_fallback": fallback,
            "p_q": p_q,
            "four_way_class": four,
            "neither_reason": neither_reason,
            "exact_rational_s_q": exact_s,
            "exact_rational_g_q": exact_g,
            "exact_vs_float_s_match": exact_match_s,
            "exact_vs_float_g_match": exact_match_g,
        }
        records.append(rec)

    H = h_sum / max(n, 1)
    A = p_sum / max(n, 1)
    S = s_sum / max(n, 1)
    G = g_sum / max(n, 1)
    summary = {
        "n": n,
        "H": H,
        "A": A,
        "S": S,
        "G": G,
        "one_minus_A": 1.0 - A,
        "one_minus_H": 1.0 - H,
        "H_minus_A": H - A,
        "H_minus_S": H - S,
        "A_minus_S": A - S,
        "A_le_H": A <= H + 1e-15,
        "decomp_ok": abs((1.0 - A) - ((1.0 - H) + (H - A))) < 1e-12,
        "four_way_counts": class_counts,
        "four_way_fracs": {k: v / max(n, 1) for k, v in class_counts.items()},
        "policy_on_h_success": (p_on_h / max(h_pos, 1)) if h_pos else None,
        "n_h_positive": h_pos,
        "grouped_choice_rate_all": grp_choice_all / max(n, 1),
        "grouped_choice_rate_on_legal_grp": (
            grp_choice_on_legal / max(legal_grp_n, 1) if legal_grp_n else None
        ),
        "n_legal_grp": legal_grp_n,
        "grp_only_recall": (grp_only_recall / max(grp_only_n, 1) if grp_only_n else None),
        "n_grp_only": grp_only_n,
        "seq_only_avoid_grp_rate": (
            seq_only_avoid_grp / max(seq_only_n, 1) if seq_only_n else None
        ),
        "n_seq_only": seq_only_n,
        "both_grp_choice_rate": (both_grp / max(both_n, 1) if both_n else None),
        "n_both": both_n,
        "exact_rational_compared": exact_compared,
        "exact_vs_float_disagreements": exact_disagree,
    }
    return summary, records


def paired_policy_vs_forceseq(records_policy: list[dict], records_forceseq: list[dict]) -> dict:
    """Paired improved/harmed/unchanged on identical examples."""
    by_id = {r["example_id"].rsplit(":", 2)[-1] + ":" + str(r["pair"]): r for r in records_forceseq}
    # match on pair+split+seed
    key = lambda r: (r["seed"], r["split"], tuple(r["pair"]))
    fs = {key(r): r for r in records_forceseq}
    improved = harmed = unc_ok = unc_wrong = 0
    for r in records_policy:
        f = fs[key(r)]
        p, s = r["p_q"], f["s_q"]  # force-seq success is s_q at same denotations
        # For force-seq checkpoint evaluation, p_q under force_seq == s_q
        fs_ok = f["p_q"]
        if p and not fs_ok:
            improved += 1
        elif fs_ok and not p:
            harmed += 1
        elif p and fs_ok:
            unc_ok += 1
        else:
            unc_wrong += 1
    n = len(records_policy)
    return {
        "n": n,
        "improved": improved,
        "harmed": harmed,
        "unchanged_correct": unc_ok,
        "unchanged_wrong": unc_wrong,
    }


def train_curriculum_with_checkpoints(
    name: str,
    geo: Geometry,
    task,
    *,
    freeze_eps_phase2: bool,
    seed: int,
) -> dict:
    """Same as 017.13 train_arm_curriculum but returns models + phase snapshots."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = geo.basic_t
    den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
    pol = Policy()
    r_t = torch.tensor(task.r, dtype=torch.float32)

    with torch.no_grad():
        den_before = den.forward_hard().detach().numpy()
        graph_before = admissibility_graph(geo, den_before)

    p1 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=True,
        learn_pol=False,
        force_seq=True,
        steps=N1_PHASE1,
        lr=LR_CUR,
        tau=GUMBEL_TAU,
        phase_name="phase1_force_seq_eps",
    )
    rec1 = _catalogue_recovery(den, geo, task)
    g1 = _graph_stats(geo, rec1["den_np"], task, graph_before=graph_before)
    den_state_p1 = copy.deepcopy(den.state_dict())
    pol_state_p1 = copy.deepcopy(pol.state_dict())
    idx_p1 = den.selected_indices().tolist()
    den_np_p1 = rec1["den_np"].copy()

    metrics_p1_force = exact_eval(
        geo,
        den_np_p1,
        _policy_fn_factory(pol, geo, r_t, force_seq=True),
        task,
        force_seq=True,
    )

    p2 = _run_phase(
        den=den,
        pol=pol,
        geo=geo,
        task=task,
        learn_den=not freeze_eps_phase2,
        learn_pol=True,
        force_seq=False,
        steps=N2_PHASE2,
        lr=LR_CUR,
        tau=GUMBEL_TAU,
        phase_name="phase2_unlock_pi"
        + ("_freeze_eps" if freeze_eps_phase2 else "_joint"),
    )
    rec2 = _catalogue_recovery(den, geo, task)
    g2 = _graph_stats(geo, rec2["den_np"], task, graph_before=graph_before)
    den_state_p2 = copy.deepcopy(den.state_dict())
    pol_state_p2 = copy.deepcopy(pol.state_dict())
    idx_p2 = den.selected_indices().tolist()
    den_np_p2 = rec2["den_np"].copy()

    metrics = exact_eval(
        geo,
        den_np_p2,
        _policy_fn_factory(pol, geo, r_t, force_seq=False),
        task,
        force_seq=False,
    )

    return {
        "arm": name,
        "freeze_eps_phase2": freeze_eps_phase2,
        "exact": metrics,
        "exact_phase1_force_seq": metrics_p1_force,
        "generalization_gap": metrics["train"]["exact_success"] - metrics["test"]["exact_success"],
        "adm_edge_count_after_phase1": g1["adm_edge_count"],
        "adm_edge_count_after": g2["adm_edge_count"],
        "adm_edge_jaccard_with_true_phase1": g1["adm_edge_jaccard_with_true"],
        "adm_edge_jaccard_with_true": g2["adm_edge_jaccard_with_true"],
        "catalogue_idx_exact_match_to_true_phase1": rec1["catalogue_idx_exact_match_to_true"],
        "catalogue_idx_exact_match_to_true": rec2["catalogue_idx_exact_match_to_true"],
        "catalogue_idx_phase1": idx_p1,
        "catalogue_idx_final": idx_p2,
        "den_np_phase1": den_np_p1,
        "den_np_final": den_np_p2,
        "den_state_phase1": den_state_p1,
        "pol_state_phase1": pol_state_p1,
        "den_state_final": den_state_p2,
        "pol_state_final": pol_state_p2,
        "den_hash_phase1": _state_hash(den_state_p1),
        "pol_hash_phase1": _state_hash(pol_state_p1),
        "den_hash_final": _state_hash(den_state_p2),
        "pol_hash_final": _state_hash(pol_state_p2),
        "wall_sec": p1["wall_sec"] + p2["wall_sec"],
        "phase1_wall_sec": p1["wall_sec"],
        "phase2_wall_sec": p2["wall_sec"],
    }


def train_baseline_with_checkpoints(
    name: str,
    geo: Geometry,
    task,
    *,
    learn_den: bool,
    learn_pol: bool,
    force_seq: bool,
    freeze_to_true: bool = False,
    seed: int = 0,
) -> dict:
    """Wrap train_arm_discrete then rebuild final den/pol for ceiling eval.

    Re-runs the same training path as 017.11/017.13 baselines and returns
    hard denotations + policy.
    """
    # Reimplement lightly to capture models (train_arm_discrete discards them).
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat_t = geo.basic_t
    if freeze_to_true:
        den: nn.Module = FrozenTrueCatalogueDenotation(task.true_idx, cat_t)
        learn_den_eff = False
    else:
        den = DiscreteCatalogueDenotation(task.n_vocab, cat_t)
        learn_den_eff = learn_den
    pol = Policy()

    params = []
    if learn_den_eff and not freeze_to_true:
        params += list(den.parameters())
    else:
        for p in den.parameters():
            p.requires_grad_(False)
    if learn_pol and not force_seq:
        params += list(pol.parameters())
    else:
        for p in pol.parameters():
            p.requires_grad_(False)

    opt = torch.optim.Adam(params, lr=LR_DISC) if params else None
    r_t = torch.tensor(task.r, dtype=torch.float32)
    M_t = geo.M_t
    idx = torch.tensor(task.pairs_train, dtype=torch.long)
    tgt = torch.tensor(np.stack(task.targets_train), dtype=torch.float32)

    t0 = time.time()
    den.train()
    loss = torch.tensor(0.0)
    for step in range(STEPS_DISC):
        E = den(hard=True, tau=GUMBEL_TAU)
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
    den.eval()
    with torch.no_grad():
        den_np = den.forward_hard().detach().numpy()
        den_idx = den.selected_indices().tolist()
    metrics = exact_eval(
        geo,
        den_np,
        _policy_fn_factory(pol, geo, r_t, force_seq=force_seq),
        task,
        force_seq=force_seq,
    )
    g = _graph_stats(geo, den_np, task)
    rec = _catalogue_recovery(den, geo, task)
    return {
        "arm": name,
        "exact": metrics,
        "generalization_gap": metrics["train"]["exact_success"] - metrics["test"]["exact_success"],
        "adm_edge_count_after": g["adm_edge_count"],
        "adm_edge_jaccard_with_true": g["adm_edge_jaccard_with_true"],
        "catalogue_idx_exact_match_to_true": rec["catalogue_idx_exact_match_to_true"],
        "catalogue_idx_final": den_idx,
        "den_np_final": den_np,
        "den_state_final": copy.deepcopy(den.state_dict()),
        "pol_state_final": copy.deepcopy(pol.state_dict()),
        "den_hash_final": _state_hash(den.state_dict()),
        "pol_hash_final": _state_hash(pol.state_dict()),
        "force_seq": force_seq,
        "freeze_to_true": freeze_to_true,
        "learn_pol": learn_pol and not force_seq,
        "wall_sec": time.time() - t0,
        "pol": pol,
        "den": den,
    }


def load_pol_from_state(state) -> Policy:
    pol = Policy()
    pol.load_state_dict(state)
    pol.eval()
    return pol


def mean_std(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()) if len(a) else float("nan"),
        "std": float(a.std()) if len(a) else float("nan"),  # population std (ddof=0) like 017.13
        "values": [float(x) for x in xs],
        "n": len(xs),
    }


def main() -> int:
    t_all = time.time()
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    wit = verify_witness(alg, geo.M)
    print("witness_disagree", wit["disagree"], "matches_01707", wit["matches_01707_expected"])
    assert wit["disagree"]

    float_to_exact = _float_to_exact_map(geo)
    true_idx = densest_vocab_indices(geo, N_VOCAB)
    print("vocab_idx", true_idx.tolist())
    print("float_to_exact_identity", float_to_exact == list(range(84)))

    all_records: list[dict] = []
    seed_audits: list[dict] = []
    repro_rows: list[dict] = []
    ckpt_dir = ART / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        task = build_native_task(geo, true_idx, seed=seed)
        print(
            f"seed={seed} train={len(task.pairs_train)} test={len(task.pairs_test)}"
        )

        # --- Curriculum arms ---
        cur = train_curriculum_with_checkpoints(
            "A_cur", geo, task, freeze_eps_phase2=False, seed=seed
        )
        curf = train_curriculum_with_checkpoints(
            "A_cur_freeze", geo, task, freeze_eps_phase2=True, seed=seed
        )
        aj = train_baseline_with_checkpoints(
            "A_joint",
            geo,
            task,
            learn_den=True,
            learn_pol=True,
            force_seq=False,
            seed=seed,
        )
        barm = train_baseline_with_checkpoints(
            "B_freeze_true_den_learn_pol",
            geo,
            task,
            learn_den=False,
            learn_pol=True,
            force_seq=False,
            freeze_to_true=True,
            seed=seed,
        )
        carm = train_baseline_with_checkpoints(
            "C_learn_discrete_den_force_seq",
            geo,
            task,
            learn_den=True,
            learn_pol=False,
            force_seq=True,
            seed=seed,
        )
        oracle = oracle_eval(geo, task)

        # Save checkpoints
        for label, blob in [
            ("A_cur", cur),
            ("A_cur_freeze", curf),
            ("A_joint", aj),
            ("B", barm),
            ("C", carm),
        ]:
            path = ckpt_dir / f"seed{seed}_{label}.pt"
            payload = {
                k: blob[k]
                for k in blob
                if k.endswith("_state_final")
                or k.endswith("_state_phase1")
                or k in ("catalogue_idx_final", "catalogue_idx_phase1", "arm")
            }
            # numpy arrays separately
            torch.save(
                {
                    "arm": blob["arm"],
                    "den_state_final": blob.get("den_state_final"),
                    "pol_state_final": blob.get("pol_state_final"),
                    "den_state_phase1": blob.get("den_state_phase1"),
                    "pol_state_phase1": blob.get("pol_state_phase1"),
                    "catalogue_idx_final": blob.get("catalogue_idx_final"),
                    "catalogue_idx_phase1": blob.get("catalogue_idx_phase1"),
                    "den_hash_final": blob.get("den_hash_final"),
                    "den_hash_phase1": blob.get("den_hash_phase1"),
                    "pol_hash_final": blob.get("pol_hash_final"),
                    "pol_hash_phase1": blob.get("pol_hash_phase1"),
                },
                path,
            )

        # Reproduction check vs committed 017.13 report
        for arm_blob in (cur, curf, aj, barm, carm):
            key = (seed, arm_blob["arm"])
            tr = arm_blob["exact"]["train"]["exact_success"]
            te = arm_blob["exact"]["test"]["exact_success"]
            tgt = REPRO_TARGETS[key]
            row = {
                "seed": seed,
                "arm": arm_blob["arm"],
                "train_exact": tr,
                "test_exact": te,
                "target_train": tgt[0],
                "target_test": tgt[1],
                "train_match": abs(tr - tgt[0]) < 1e-6,
                "test_match": abs(te - tgt[1]) < 1e-6,
            }
            repro_rows.append(row)
            print(
                f"  repro {arm_blob['arm']}: tr={tr:.6f} (tgt {tgt[0]}) "
                f"te={te:.6f} (tgt {tgt[1]}) "
                f"match={row['train_match'] and row['test_match']}"
            )

        # Phase-1 denotation identity across curriculum arms
        p1_den_equal = cur["den_hash_phase1"] == curf["den_hash_phase1"]
        p1_idx_equal = cur["catalogue_idx_phase1"] == curf["catalogue_idx_phase1"]
        p1_vs_c_idx = cur["catalogue_idx_phase1"] == carm["catalogue_idx_final"]

        audits = {}

        def run_audit(
            key_name: str,
            *,
            den_np,
            den_idx,
            pol_state,
            force_seq: bool,
            has_trained_policy: bool,
            phase: str,
            arm_name: str,
        ):
            pol = load_pol_from_state(pol_state) if pol_state is not None else None
            split_summaries = {}
            for split, pairs, tgts in [
                ("train", task.pairs_train, task.targets_train),
                ("test", task.pairs_test, task.targets_test),
            ]:
                summ, recs = evaluate_ceiling_split(
                    geo=geo,
                    den_np=den_np,
                    den_idx=den_idx,
                    task=task,
                    pairs=pairs,
                    targets=tgts,
                    split=split,
                    seed=seed,
                    arm=arm_name,
                    phase=phase,
                    pol=pol,
                    force_seq=force_seq,
                    has_trained_policy=has_trained_policy,
                    float_to_exact=float_to_exact,
                )
                split_summaries[split] = summ
                all_records.extend(recs)
            audits[key_name] = {
                "arm": arm_name,
                "phase": phase,
                "force_seq": force_seq,
                "has_trained_policy": has_trained_policy,
                "den_idx": den_idx,
                "splits": split_summaries,
            }
            return split_summaries

        # Required checkpoints
        run_audit(
            "A_cur_phase1",
            den_np=cur["den_np_phase1"],
            den_idx=cur["catalogue_idx_phase1"],
            pol_state=cur["pol_state_phase1"],
            force_seq=True,
            has_trained_policy=False,
            phase="phase1",
            arm_name="A_cur",
        )
        run_audit(
            "A_cur_freeze_phase1",
            den_np=curf["den_np_phase1"],
            den_idx=curf["catalogue_idx_phase1"],
            pol_state=curf["pol_state_phase1"],
            force_seq=True,
            has_trained_policy=False,
            phase="phase1",
            arm_name="A_cur_freeze",
        )
        run_audit(
            "A_cur_final",
            den_np=cur["den_np_final"],
            den_idx=cur["catalogue_idx_final"],
            pol_state=cur["pol_state_final"],
            force_seq=False,
            has_trained_policy=True,
            phase="final",
            arm_name="A_cur",
        )
        # Same denotations, force-seq for paired comparison
        run_audit(
            "A_cur_final_forceseq",
            den_np=cur["den_np_final"],
            den_idx=cur["catalogue_idx_final"],
            pol_state=cur["pol_state_final"],
            force_seq=True,
            has_trained_policy=True,
            phase="final_forceseq_diag",
            arm_name="A_cur",
        )
        run_audit(
            "A_cur_freeze_final",
            den_np=curf["den_np_final"],
            den_idx=curf["catalogue_idx_final"],
            pol_state=curf["pol_state_final"],
            force_seq=False,
            has_trained_policy=True,
            phase="final",
            arm_name="A_cur_freeze",
        )
        run_audit(
            "A_cur_freeze_final_forceseq",
            den_np=curf["den_np_final"],
            den_idx=curf["catalogue_idx_final"],
            pol_state=curf["pol_state_final"],
            force_seq=True,
            has_trained_policy=True,
            phase="final_forceseq_diag",
            arm_name="A_cur_freeze",
        )
        run_audit(
            "C_final",
            den_np=carm["den_np_final"],
            den_idx=carm["catalogue_idx_final"],
            pol_state=carm["pol_state_final"],
            force_seq=True,
            has_trained_policy=False,
            phase="final",
            arm_name="C",
        )
        run_audit(
            "B_final",
            den_np=barm["den_np_final"],
            den_idx=barm["catalogue_idx_final"],
            pol_state=barm["pol_state_final"],
            force_seq=False,
            has_trained_policy=True,
            phase="final",
            arm_name="B",
        )
        run_audit(
            "B_final_forceseq",
            den_np=barm["den_np_final"],
            den_idx=barm["catalogue_idx_final"],
            pol_state=barm["pol_state_final"],
            force_seq=True,
            has_trained_policy=True,
            phase="final_forceseq_diag",
            arm_name="B",
        )
        run_audit(
            "A_joint_final",
            den_np=aj["den_np_final"],
            den_idx=aj["catalogue_idx_final"],
            pol_state=aj["pol_state_final"],
            force_seq=False,
            has_trained_policy=True,
            phase="final",
            arm_name="A_joint",
        )

        # Paired policy vs force-seq at frozen denotations
        def paired_from_keys(pol_key, fs_key, split):
            rp = [r for r in all_records if r["seed"] == seed and r["arm"] == audits[pol_key]["arm"] and r["phase"] == audits[pol_key]["phase"] and r["split"] == split]
            rf = [r for r in all_records if r["seed"] == seed and r["arm"] == audits[fs_key]["arm"] and r["phase"] == audits[fs_key]["phase"] and r["split"] == split]
            # more reliable filter by audit phase names already unique per arm
            return paired_policy_vs_forceseq(rp, rf)

        paired = {
            "A_cur_freeze_test": paired_from_keys(
                "A_cur_freeze_final", "A_cur_freeze_final_forceseq", "test"
            ),
            "A_cur_freeze_train": paired_from_keys(
                "A_cur_freeze_final", "A_cur_freeze_final_forceseq", "train"
            ),
            "A_cur_test": paired_from_keys("A_cur_final", "A_cur_final_forceseq", "test"),
            "B_test": paired_from_keys("B_final", "B_final_forceseq", "test"),
        }

        # Denotation equality freeze vs C
        freeze_vs_c = {
            "phase1_idx_equal_to_C_final": curf["catalogue_idx_phase1"] == carm["catalogue_idx_final"],
            "final_idx_equal_to_C_final": curf["catalogue_idx_final"] == carm["catalogue_idx_final"],
            "phase1_den_hash_equal_across_A_cur_arms": p1_den_equal,
            "phase1_idx_equal_across_A_cur_arms": p1_idx_equal,
            "A_cur_phase1_idx_equal_C": p1_vs_c_idx,
        }

        seed_audits.append(
            {
                "seed": seed,
                "n_train": len(task.pairs_train),
                "n_test": len(task.pairs_test),
                "oracle_exact": oracle["exact"],
                "repro": [r for r in repro_rows if r["seed"] == seed],
                "denotation_identity": freeze_vs_c,
                "checkpoint_hashes": {
                    "A_cur_phase1_den": cur["den_hash_phase1"],
                    "A_cur_freeze_phase1_den": curf["den_hash_phase1"],
                    "A_cur_final_den": cur["den_hash_final"],
                    "A_cur_freeze_final_den": curf["den_hash_final"],
                    "C_final_den": carm["den_hash_final"],
                    "B_final_den": barm["den_hash_final"],
                    "A_joint_final_den": aj["den_hash_final"],
                },
                "legacy_exact": {
                    "A_cur": cur["exact"],
                    "A_cur_freeze": curf["exact"],
                    "A_joint": aj["exact"],
                    "B": barm["exact"],
                    "C": carm["exact"],
                },
                "audits": {
                    k: {
                        "arm": v["arm"],
                        "phase": v["phase"],
                        "force_seq": v["force_seq"],
                        "has_trained_policy": v["has_trained_policy"],
                        "splits": v["splits"],
                    }
                    for k, v in audits.items()
                },
                "paired_policy_vs_forceseq": paired,
            }
        )

        # Print quick test H/A/S/G
        for k in [
            "A_cur_final",
            "A_cur_freeze_final",
            "C_final",
            "B_final",
            "A_joint_final",
            "A_cur_phase1",
            "A_cur_freeze_phase1",
        ]:
            te = audits[k]["splits"]["test"]
            print(
                f"  ceiling {k}: H={te['H']:.3f} A={te['A']:.3f} S={te['S']:.3f} G={te['G']:.3f} "
                f"1-H={te['one_minus_H']:.3f} H-A={te['H_minus_A']:.3f} A<=H={te['A_le_H']}"
            )

    # Aggregate across seeds
    def collect(audit_key: str, split: str, metric: str) -> list[float]:
        return [s["audits"][audit_key]["splits"][split][metric] for s in seed_audits]

    aggregate = {}
    for audit_key in [
        "A_cur_phase1",
        "A_cur_freeze_phase1",
        "A_cur_final",
        "A_cur_freeze_final",
        "C_final",
        "B_final",
        "A_joint_final",
        "A_cur_final_forceseq",
        "A_cur_freeze_final_forceseq",
        "B_final_forceseq",
    ]:
        aggregate[audit_key] = {}
        for split in ("train", "test"):
            aggregate[audit_key][split] = {
                m: mean_std(collect(audit_key, split, m))
                for m in ("H", "A", "S", "G", "one_minus_H", "H_minus_A", "H_minus_S", "A_minus_S")
            }
            # four-way mean counts
            counts = [s["audits"][audit_key]["splits"][split]["four_way_counts"] for s in seed_audits]
            aggregate[audit_key][split]["four_way_counts_mean"] = {
                k: float(np.mean([c[k] for c in counts])) for k in counts[0]
            }
            aggregate[audit_key][split]["policy_diags"] = {
                "grouped_choice_rate_all": mean_std(
                    collect(audit_key, split, "grouped_choice_rate_all")
                ),
                "grp_only_recall": mean_std(
                    [
                        s["audits"][audit_key]["splits"][split]["grp_only_recall"]
                        for s in seed_audits
                        if s["audits"][audit_key]["splits"][split]["grp_only_recall"] is not None
                    ]
                ),
                "seq_only_avoid_grp_rate": mean_std(
                    [
                        s["audits"][audit_key]["splits"][split]["seq_only_avoid_grp_rate"]
                        for s in seed_audits
                        if s["audits"][audit_key]["splits"][split]["seq_only_avoid_grp_rate"]
                        is not None
                    ]
                ),
                "policy_on_h_success": mean_std(
                    [
                        s["audits"][audit_key]["splits"][split]["policy_on_h_success"]
                        for s in seed_audits
                        if s["audits"][audit_key]["splits"][split]["policy_on_h_success"]
                        is not None
                    ]
                ),
            }

    # Phase2 reachability change for A_cur / A_cur_freeze
    phase2_reachability = {}
    for arm, p1k, p2k in [
        ("A_cur", "A_cur_phase1", "A_cur_final"),
        ("A_cur_freeze", "A_cur_freeze_phase1", "A_cur_freeze_final"),
    ]:
        phase2_reachability[arm] = {
            "test_H_phase1": mean_std(collect(p1k, "test", "H")),
            "test_H_final": mean_std(collect(p2k, "test", "H")),
            "test_S_phase1": mean_std(collect(p1k, "test", "S")),
            "test_S_final": mean_std(collect(p2k, "test", "S")),
            "delta_H_values": [
                b - a
                for a, b in zip(collect(p1k, "test", "H"), collect(p2k, "test", "H"))
            ],
            "delta_S_values": [
                b - a
                for a, b in zip(collect(p1k, "test", "S"), collect(p2k, "test", "S"))
            ],
        }

    # True vs recovered H
    true_vs_recovered = {
        "B_H_test": mean_std(collect("B_final", "test", "H")),
        "A_cur_freeze_H_test": mean_std(collect("A_cur_freeze_final", "test", "H")),
        "A_cur_H_test": mean_std(collect("A_cur_final", "test", "H")),
        "C_H_test": mean_std(collect("C_final", "test", "H")),
        "A_joint_H_test": mean_std(collect("A_joint_final", "test", "H")),
    }

    # Exact vs float disagreement totals
    exact_vs_float = {
        "total_compared": sum(
            s["audits"][k]["splits"][sp]["exact_rational_compared"]
            for s in seed_audits
            for k in s["audits"]
            for sp in ("train", "test")
        ),
        "total_disagreements": sum(
            s["audits"][k]["splits"][sp]["exact_vs_float_disagreements"]
            for s in seed_audits
            for k in s["audits"]
            for sp in ("train", "test")
        ),
    }

    repro_ok = all(r["train_match"] and r["test_match"] for r in repro_rows)

    # Primary diagnosis (computed)
    # Prefer A_cur_freeze final test: compare mean(1-H) vs mean(H-A)
    freeze_test = aggregate["A_cur_freeze_final"]["test"]
    one_h = freeze_test["one_minus_H"]["mean"]
    h_a = freeze_test["H_minus_A"]["mean"]
    a_cur_dh = phase2_reachability["A_cur"]["delta_H_values"]
    joint_damages = all(d < -1e-6 for d in a_cur_dh) or (np.mean(a_cur_dh) < -0.05)

    if not repro_ok:
        primary = "premise/reproduction/evaluator failure"
    elif joint_damages and one_h >= h_a:
        # still choose among the four allowed; joint damage reinforces reachability
        primary = (
            "frozen-representation/program reachability is the larger observed deficit"
            if one_h >= h_a
            else "policy selection is the larger observed deficit"
        )
    elif abs(one_h - h_a) < 0.05:
        primary = "mixed or seed-dependent deficits"
    elif one_h > h_a:
        primary = "frozen-representation/program reachability is the larger observed deficit"
    else:
        primary = "policy selection is the larger observed deficit"

    # Seed-level 1-H vs H-A for freeze
    seed_deficits = []
    for s in seed_audits:
        te = s["audits"]["A_cur_freeze_final"]["splits"]["test"]
        seed_deficits.append(
            {
                "seed": s["seed"],
                "one_minus_H": te["one_minus_H"],
                "H_minus_A": te["H_minus_A"],
                "larger": (
                    "reachability"
                    if te["one_minus_H"] > te["H_minus_A"] + 1e-12
                    else (
                        "policy"
                        if te["H_minus_A"] > te["one_minus_H"] + 1e-12
                        else "tie"
                    )
                ),
            }
        )
    if len({d["larger"] for d in seed_deficits}) > 1 and primary != "premise/reproduction/evaluator failure":
        # if seeds disagree on which is larger, upgrade to mixed
        if not all(d["larger"] == seed_deficits[0]["larger"] for d in seed_deficits):
            primary = "mixed or seed-dependent deficits"

    report = {
        "source": "017.15 recovered-geometry ceiling and policy shortfall",
        "in_reply_to": "017.15-HC-recovered-geometry-ceiling-and-policy-regret-task.md",
        "reproduction_status": {
            "saved_checkpoints_found_in_repo": False,
            "action": "reproduced original 017.13/017.14 runs deterministically (same settings)",
            "baseline_commit": "c41f77d22b62b71200d142aa3bb5635ec5defe98",
            "all_arms_match_01713_report": repro_ok,
            "per_arm": repro_rows,
            "std_convention": "population std (np.std ddof=0), seed-averaged means",
        },
        "implementation": {
            "mul": "structure constants from topographo.ssd.sedenion.SedenionAlgebra.mul",
            "evaluator": "existing float projective_key / rays_equal (TOL=1e-8, round 10)",
            "exact_crosscheck": (
                "topographo.ssd.exact + fips_basic.EVENTS via projective float↔exact map; "
                "CONVENTION (a,b)(c,d)=(ac-d*conj(b), conj(a)*d + c*b)"
            ),
            "equality_status": (
                "Primary scores use existing float projective_key evaluator for "
                "comparability with 017.14. Exact rational cross-check recorded; "
                "disagreement count in exact_vs_float."
            ),
            "denominator_policy": (
                "All task examples retained; undefined program outputs count as failure "
                "(differs from legacy exact_eval which dropped undefined from n)."
            ),
            "host": "thebeast",
            "steps": {"N1": N1_PHASE1, "N2": N2_PHASE2, "baselines": STEPS_DISC},
            "lr": LR_CUR,
            "gumbel_tau": GUMBEL_TAU,
            "seeds": list(SEEDS),
        },
        "witness_check": wit,
        "exact_vs_float": exact_vs_float,
        "aggregate": aggregate,
        "phase2_reachability": phase2_reachability,
        "true_vs_recovered_H": true_vs_recovered,
        "seed_deficits_A_cur_freeze_test": seed_deficits,
        "primary_diagnosis": primary,
        "seed_audits": seed_audits,
        "fence": (
            "evaluation-only diagnostic on finite FIPS proxy; Issue 017 not closed; "
            "no modular grokking / OT-advantage claim"
        ),
        "wall_sec_total": time.time() - t_all,
    }

    # Write full aggregate (without bulky per-example)
    full_path = OUT / "learned_admissibility_01715_report.json"
    # seed_audits is fine; records separate
    full_path.write_text(json.dumps(report, indent=2) + "\n")

    # Per-example JSONL
    per_ex_path = ART / "learned_admissibility_01715_per_example.jsonl"
    with per_ex_path.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Also JSON array for convenience (may be large but ~few thousand rows)
    per_ex_json = ART / "learned_admissibility_01715_per_example.json"
    per_ex_json.write_text(json.dumps(all_records, indent=2) + "\n")

    # Slim aggregate for Quilt
    slim = {
        "source": report["source"],
        "in_reply_to": report["in_reply_to"],
        "reproduction_status": report["reproduction_status"],
        "implementation": report["implementation"],
        "witness_check": {
            "disagree": wit["disagree"],
            "matches_01707_expected": wit["matches_01707_expected"],
        },
        "exact_vs_float": exact_vs_float,
        "aggregate_test_HASG": {
            k: {
                "H": aggregate[k]["test"]["H"],
                "A": aggregate[k]["test"]["A"],
                "S": aggregate[k]["test"]["S"],
                "G": aggregate[k]["test"]["G"],
                "one_minus_H": aggregate[k]["test"]["one_minus_H"],
                "H_minus_A": aggregate[k]["test"]["H_minus_A"],
                "four_way_counts_mean": aggregate[k]["test"]["four_way_counts_mean"],
            }
            for k in [
                "A_cur_final",
                "A_cur_freeze_final",
                "C_final",
                "B_final",
                "A_joint_final",
                "A_cur_phase1",
                "A_cur_freeze_phase1",
            ]
        },
        "phase2_reachability": phase2_reachability,
        "true_vs_recovered_H": true_vs_recovered,
        "seed_deficits_A_cur_freeze_test": seed_deficits,
        "primary_diagnosis": primary,
        "per_example_path": str(per_ex_path.relative_to(OUT.parent.parent))
        if False
        else "experiments/tlm_fixed_head/01715_artifacts/learned_admissibility_01715_per_example.jsonl",
        "fence": report["fence"],
    }
    # Fix relative path
    slim["per_example_path"] = (
        "experiments/tlm_fixed_head/01715_artifacts/learned_admissibility_01715_per_example.jsonl"
    )
    slim_path = OUT / "learned_admissibility_01715_report_slim.json"
    slim_path.write_text(json.dumps(slim, indent=2) + "\n")

    tiny = {
        "source": slim["source"],
        "reproduction_status": {
            "saved_checkpoints_found_in_repo": False,
            "reproduced_01713": repro_ok,
            "baseline_commit": "c41f77d22b62b71200d142aa3bb5635ec5defe98",
        },
        "test_mean_std_HASG": {
            k: {
                m: {
                    "mean": slim["aggregate_test_HASG"][k][m]["mean"],
                    "std": slim["aggregate_test_HASG"][k][m]["std"],
                }
                for m in ("H", "A", "S", "G")
            }
            for k in slim["aggregate_test_HASG"]
        },
        "primary_diagnosis": primary,
        "exact_vs_float": exact_vs_float,
        "per_example_path": slim["per_example_path"],
    }
    tiny_path = OUT / "learned_admissibility_01715_report_tiny.json"
    tiny_path.write_text(json.dumps(tiny, indent=2) + "\n")

    print("PRIMARY_DIAGNOSIS", primary)
    print("REPRO_OK", repro_ok)
    print("EXACT_VS_FLOAT", exact_vs_float)
    print("wrote", full_path)
    print("wrote", slim_path)
    print("wrote", tiny_path)
    print("wrote", per_ex_path)
    print("wall_sec", report["wall_sec_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

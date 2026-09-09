#!/usr/bin/env python3
"""017.23 final audit: catalogue-order fix, graph-iso claim, GroupedFirst baseline.

Bounded postprocessing on the pinned 017.22 benchmark. No new training.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from learned_admissibility_01709 import (
    N_VOCAB,
    SEEDS,
    Geometry,
    Policy,
    SedenionAlgebra,
)
from learned_admissibility_01719_exact_eval import (
    audit_rec_ray_teacher,
    exact_from_float_idx,
    exact_p_grp,
    exact_p_seq,
    exact_rays_equal,
    float_to_exact_map,
    preferred_target_exact,
)
from learned_admissibility_01721_structural_controls import (
    N_CATALOGUE,
    PERM_SEEDS,
    PhiGeometry,
    catalogue_permutations,
    exact_p_grp_ambient,
    heldout_H_native,
    load_recovered_maps,
    materialize_benchmark,
    sha256_json,
)
from topographo.ssd import exact, fips_basic, projective

OUT = Path(__file__).resolve().parent
ART21 = OUT / "01721_artifacts"
CKPT_DIR = ART21 / "checkpoints"
ART = OUT / "01723_artifacts"
ART.mkdir(parents=True, exist_ok=True)
ATTACH = OUT / "017.24-Code-attachments"
ATTACH.mkdir(parents=True, exist_ok=True)

PINNED_MANIFEST_SHA = "81a4700e8cf1bac0f66e473d144c4ccb003fe26f8cb7175108fe8884effd107e"


def first_max_idx(row: np.ndarray) -> int | None:
    """Consistent first-max: leftmost index of the maximum (numpy argmax semantics)."""
    if row.size == 0 or float(row.sum()) <= 0.0:
        return None
    return int(np.argmax(row))


def build_votes(geo: Geometry, fte: list[int], bench: dict, seed: int) -> np.ndarray:
    """Train-only Rec_ray inverse votes (exact keys) — same construction as 017.21 audit."""
    from learned_admissibility_01719_exact_eval import exact_program_maps, exact_canonical_key

    true_idx = np.asarray(bench["true_idx"], dtype=np.int64)
    task = bench["seeds"][seed]["_task"]
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
    return votes


def corrected_tie_and_order_audit(geo: Geometry, fte: list[int], bench: dict) -> dict:
    """§B: score physical Event identities from permuted argmax; consistent first-max."""
    true_idx = np.asarray(bench["true_idx"], dtype=np.int64)
    per_seed = {}
    for seed in SEEDS:
        votes = build_votes(geo, fte, bench, seed)
        token_detail = []
        for t in range(N_VOCAB):
            row = votes[t]
            mx = float(row.max()) if row.sum() > 0 else 0.0
            cands = [int(c) for c in np.where(row == mx)[0]] if mx > 0 else []
            fm = first_max_idx(row)
            # margin: next-strictly-lower vote (or 0 if tied / empty)
            if mx > 0 and len(cands) == 1:
                others = row[row < mx]
                ru = float(others.max()) if others.size else 0.0
                margin = mx - ru
            else:
                margin = 0.0
            token_detail.append(
                {
                    "token": t,
                    "max_votes": mx,
                    "maximizing_event_candidates": cands,
                    "n_maximizers": len(cands),
                    "margin_to_next": margin,
                    "first_max": fm,
                    "true": int(true_idx[t]),
                    "first_max_matches_true": bool(fm is not None and fm == int(true_idx[t])),
                    "tied": len(cands) > 1,
                }
            )
        tied_tokens = [d for d in token_detail if d["tied"]]
        alt_H = []
        if tied_tokens:
            slots = [d["maximizing_event_candidates"] for d in tied_tokens]
            base = [d["first_max"] for d in token_detail]
            for combo in product(*slots):
                den = list(base)
                for d, c in zip(tied_tokens, combo):
                    den[d["token"]] = c
                H = heldout_H_native(den, fte, bench["seeds"][seed])
                alt_H.append({"den": den, "test_H": H, "matches_true": den == true_idx.tolist()})

        # Reference physical map via consistent first-max on unpermuted votes
        ref_physical = [first_max_idx(votes[t]) for t in range(N_VOCAB)]
        ref_match = sum(
            1 for a, b in zip(ref_physical, true_idx.tolist()) if a is not None and a == b
        )

        perm_results = []
        for ps in PERM_SEEDS:
            perm = catalogue_permutations(ps)
            # storage position k holds physical event perm[k]
            # votes_storage[t, k] = votes[t, perm[k]]
            votes_s = votes[:, perm]
            storage_argmax = np.array(
                [first_max_idx(votes_s[t]) if votes_s[t].sum() > 0 else -1 for t in range(N_VOCAB)],
                dtype=np.int64,
            )
            physical = [
                int(perm[k]) if k >= 0 else None for k in storage_argmax
            ]
            match16 = sum(
                1 for a, b in zip(physical, true_idx.tolist()) if a is not None and a == b
            )
            map_changed = physical != ref_physical
            perm_results.append(
                {
                    "perm_seed": ps,
                    "perm": perm.tolist(),
                    "selected_physical": physical,
                    "first_max_match_over_16": match16,
                    "physical_map_changed_vs_reference": map_changed,
                    "reference_match_over_16": ref_match,
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
                "combos": alt_H[:32],
            },
            "catalogue_permutations": perm_results,
            "reference_physical": ref_physical,
            "all_perms_preserve_physical_map": all(
                not p["physical_map_changed_vs_reference"] for p in perm_results
            ),
            "all_perms_match16": all(p["first_max_match_over_16"] == 16 for p in perm_results),
            "min_perm_match16": min(p["first_max_match_over_16"] for p in perm_results),
            "max_perm_match16": max(p["first_max_match_over_16"] for p in perm_results),
            # SUPERSEDED claim from 017.22:
            "superseded_01722_claim_first_max_recovery_16_16_all_perms": False,
        }

    return {
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "primary_map_conditional_on_favorable_first_max": any(
            per_seed[s]["any_tie"] for s in SEEDS
        ),
        "corrected_scoring_note": (
            "match16 is computed from physical Event identities obtained by "
            "first-max (argmax) on permuted storage columns mapped back via perm[k]. "
            "017.22 incorrectly scored unpermuted argmax for every permutation."
        ),
        "note": (
            "Primary arm comparison uses the same frozen Rec_ray first-max map. "
            "Where vote ties exist, first-max is not unique identification."
        ),
    }


def synthetic_order_regressions() -> dict:
    """§B synthetic: tied reorder can change physical selection; untied cannot."""
    # Untied: unique max at physical 3
    votes_untied = np.zeros(8, dtype=np.float64)
    votes_untied[3] = 5.0
    votes_untied[1] = 2.0
    votes_untied[5] = 1.0
    # Tied: physical 2 and 6 both max
    votes_tied = np.zeros(8, dtype=np.float64)
    votes_tied[2] = 4.0
    votes_tied[6] = 4.0
    votes_tied[0] = 1.0

    def physical_under_perm(votes: np.ndarray, perm: np.ndarray) -> int:
        storage = votes[perm]
        k = first_max_idx(storage)
        assert k is not None
        return int(perm[k])

    # Untied: many perms should preserve physical 3
    rng = np.random.default_rng(0)
    untied_physicals = []
    for i in range(20):
        perm = rng.permutation(8)
        untied_physicals.append(physical_under_perm(votes_untied, perm))
    untied_invariant = all(p == 3 for p in untied_physicals)

    # Tied: find a perm that selects 2 and one that selects 6
    tied_seen = set()
    for i in range(200):
        perm = rng.permutation(8)
        tied_seen.add(physical_under_perm(votes_tied, perm))
        if tied_seen >= {2, 6}:
            break
    tied_can_change = tied_seen >= {2, 6}

    return {
        "untied_invariant_under_reorder": untied_invariant,
        "untied_selected_physicals": untied_physicals[:5],
        "tied_can_change_selected_physical": tied_can_change,
        "tied_selected_physicals_observed": sorted(tied_seen),
        "ok": untied_invariant and tied_can_change,
    }


def graph_from_adj(adj: np.ndarray) -> nx.Graph:
    """Undirected simple graph from (possibly directed) adjacency; ignore direction."""
    n = adj.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] or adj[j, i]:
                g.add_edge(i, j)
    return g


def classify_rewired_mask(native_adj: np.ndarray, rewired_adj: np.ndarray) -> dict:
    """§C: verify structure + exact labeled / isomorphic / non-isomorphic."""
    n = native_adj.shape[0]
    # Symmetry / self-loops on directed arrays
    sym_n = bool(np.array_equal(native_adj, native_adj.T))
    sym_r = bool(np.array_equal(rewired_adj, rewired_adj.T))
    self_n = int(np.trace(native_adj))
    self_r = int(np.trace(rewired_adj))
    Gn = graph_from_adj(native_adj)
    Gr = graph_from_adj(rewired_adj)
    deg_n = sorted(d for _, d in Gn.degree())
    deg_r = sorted(d for _, d in Gr.degree())
    e_n, e_r = Gn.number_of_edges(), Gr.number_of_edges()
    identical_labeled = set(Gn.edges()) == set(Gr.edges())
    # Exact isomorphism (small n=16)
    isomorphic = nx.is_isomorphic(Gn, Gr)
    # Decisive invariants
    cc_n = sorted(len(c) for c in nx.connected_components(Gn))
    cc_r = sorted(len(c) for c in nx.connected_components(Gr))
    tri_n = sum(nx.triangles(Gn).values()) // 3
    tri_r = sum(nx.triangles(Gr).values()) // 3
    if identical_labeled:
        status = "identical_labeled"
    elif isomorphic:
        status = "isomorphic_under_relabeling"
    else:
        status = "non_isomorphic"
    return {
        "n_vertices": n,
        "native_symmetric_directed": sym_n,
        "rewired_symmetric_directed": sym_r,
        "native_self_loops": self_n,
        "rewired_self_loops": self_r,
        "native_undirected_edge_count": e_n,
        "rewired_undirected_edge_count": e_r,
        "degree_sequence_native": deg_n,
        "degree_sequence_rewired": deg_r,
        "degree_sequence_match": deg_n == deg_r,
        "connected_component_sizes_native": cc_n,
        "connected_component_sizes_rewired": cc_r,
        "triangle_count_native": tri_n,
        "triangle_count_rewired": tri_r,
        "identical_labeled": identical_labeled,
        "isomorphic_under_relabeling": isomorphic,
        "classification": status,
        # SUPERSEDED: 017.22 "non_isomorphic_edge_set" meant only edge-set differs
        "superseded_01722_non_isomorphic_edge_set_meant_only_different_labeled_edges": True,
    }


def audit_saved_rewired_masks() -> dict:
    """Load published 017.22 rewired masks; do not regenerate."""
    path = ART21 / "rewired_masks.json"
    data = json.loads(path.read_text())
    # Need native Cyc mask per seed — reconstruct from dens
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    pgeo = PhiGeometry(geo)
    fte = float_to_exact_map(geo)
    recovered = load_recovered_maps(geo)
    bench = materialize_benchmark(geo, pgeo, fte, recovered)
    assert bench["manifest_sha256"] == PINNED_MANIFEST_SHA, (
        f"manifest mismatch: {bench['manifest_sha256']} != {PINNED_MANIFEST_SHA}"
    )
    results = {}
    summary_counts = Counter()
    for seed_s, masks in data.items():
        seed = int(seed_s)
        native = bench["seeds"][seed]["_cyc_mask"]
        per_mask = []
        for m in masks:
            adj = np.asarray(m["adjacency"], dtype=np.int8)
            cls = classify_rewired_mask(native, adj)
            cls["mask_id"] = m["mask_id"]
            cls["rewire_seed"] = m.get("rewire_seed")
            cls["published_degree_sequence_preserved"] = m.get("degree_sequence_preserved")
            cls["published_jaccard_undirected"] = m.get("jaccard_undirected")
            cls["published_non_isomorphic_edge_set_flag"] = m.get("non_isomorphic_edge_set")
            per_mask.append(cls)
            summary_counts[cls["classification"]] += 1
        results[seed_s] = per_mask
    return {
        "per_seed": results,
        "classification_counts": dict(summary_counts),
        "n_masks": sum(summary_counts.values()),
        "note": (
            "Exact nx.is_isomorphic on undirected projections of saved masks. "
            "017.22 flag non_isomorphic_edge_set only meant different labeled edge sets."
        ),
    }


def grouped_first_choice(
    *,
    gate_on: bool,
    pg_e,
    ps_e,
) -> tuple[str, object | None, dict]:
    """Deterministic GroupedFirst: grp if allowed+defined, else seq; never peek target."""
    undef = {"seq_undef": 0, "grp_unavailable": 0, "grp_occ_undef": 0, "selected_undef": 0}
    if gate_on:
        if pg_e is not None:
            return "grp", pg_e, undef
        undef["grp_occ_undef"] += 1
        undef["grp_unavailable"] += 1
    else:
        undef["grp_unavailable"] += 1
    if ps_e is None:
        undef["seq_undef"] += 1
        undef["selected_undef"] += 1
        return "seq", None, undef
    return "seq", ps_e, undef


def evaluate_grouped_first(
    *,
    arm: str,
    seed: int,
    den: list[int],
    fte: list[int],
    seed_payload: dict,
    gate_mask: np.ndarray,
    use_ambient_endpoints: bool,
    mask_id: str | None = None,
) -> tuple[dict, dict, list[dict]]:
    re = exact.basis(4)
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
        undef_tot = Counter()
        records = []
        for ex_i, ((i, j), etgt) in enumerate(zip(pairs, targets)):
            ai, bi = int(den[i]), int(den[j])
            ae = exact_from_float_idx(ai, fte)
            be = exact_from_float_idx(bi, fte)
            ps_e = exact_p_seq(ae, be, re)
            gate_on = bool(gate_mask[i, j])
            if use_ambient_endpoints:
                pg_e = exact_p_grp_ambient(ae, be, re) if gate_on else None
                gate_effective = gate_on and pg_e is not None
            else:
                if gate_on and fips_basic.admissible(ae, be):
                    pg_e = exact_p_grp(ae, be, re)
                    gate_effective = pg_e is not None
                else:
                    pg_e = None
                    gate_effective = False
            if gate_effective:
                legal_grp += 1
            s_q = int(exact_rays_equal(ps_e, etgt))
            g_q = int(gate_effective and exact_rays_equal(pg_e, etgt))
            h_q = max(s_q, g_q)
            branch, out_e, undef = grouped_first_choice(
                gate_on=gate_effective, pg_e=pg_e if gate_effective else None, ps_e=ps_e
            )
            # Note: pass gate_effective so we don't treat gated-but-undefined as allowed
            if branch == "grp":
                grp_selected += 1
            for k, v in undef.items():
                undef_tot[k] += v
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
            # Exception analysis vs preferred generator
            preferred_would_be_grp = g_q and not s_q
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "mask_id": mask_id,
                    "split": split,
                    "example_id": f"{seed}:GroupedFirst:{arm}:{mask_id or '-'}:{split}:{ex_i}",
                    "pair": [int(i), int(j)],
                    "denotation_ids": [ai, bi],
                    "s": s_q,
                    "g": g_q,
                    "h": h_q,
                    "a": p_q,
                    "policy_choice": branch,
                    "four_way": four,
                    "grp_available": gate_effective,
                    "miss_despite_h": bool(h_q and not p_q),
                    "grp_only_missed": bool(preferred_would_be_grp and not p_q),
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
            "undefined_counts": dict(undef_tot),
            "n_miss_despite_h": sum(1 for r in records if r["miss_despite_h"]),
            "n_grp_only_missed": sum(1 for r in records if r["grp_only_missed"]),
        }
        all_records.extend(records)
    return split_summaries["train"], split_summaries["test"], all_records


def load_learned_test_A_from_report() -> dict:
    """Published 017.22 learned-policy test means (no retrain)."""
    tiny = json.loads((OUT / "learned_admissibility_01721_report_tiny.json").read_text())
    return tiny["table"]


def mean_std(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "values": xs}


def run_grouped_first_suite(bench: dict, fte: list[int]) -> dict:
    masks_pub = json.loads((ART21 / "rewired_masks.json").read_text())
    gf_runs = {name: [] for name in ["Native", "Ambient"]}
    rewired_runs = []
    all_records = []
    for seed in SEEDS:
        sp = bench["seeds"][seed]
        den = sp["recovered_catalogue_idx"]
        cyc = sp["_cyc_mask"]
        amb = sp["_amb_mask"]
        for arm, gate, ambient in (
            ("Native", cyc, False),
            ("Ambient", amb, True),
        ):
            tr, te, rec = evaluate_grouped_first(
                arm=arm,
                seed=seed,
                den=den,
                fte=fte,
                seed_payload=sp,
                gate_mask=gate,
                use_ambient_endpoints=ambient,
            )
            gf_runs[arm].append({"seed": seed, "train": tr, "test": te})
            all_records.extend(rec)
        for mi, m in enumerate(masks_pub[str(seed)]):
            adj = np.asarray(m["adjacency"], dtype=np.int8)
            mid = m["mask_id"]
            tr, te, rec = evaluate_grouped_first(
                arm="Rewired",
                seed=seed,
                den=den,
                fte=fte,
                seed_payload=sp,
                gate_mask=adj,
                use_ambient_endpoints=True,
                mask_id=mid,
            )
            rewired_runs.append(
                {"seed": seed, "mask_id": mid, "train": tr, "test": te}
            )
            all_records.extend(rec)

    def agg(runs):
        return {
            "test_H": mean_std([r["test"]["H"] for r in runs]),
            "test_A": mean_std([r["test"]["A"] for r in runs]),
            "test_S": mean_std([r["test"]["S"] for r in runs]),
            "test_G": mean_std([r["test"]["G"] for r in runs]),
            "test_grp_selection": mean_std([r["test"]["grp_selection_fraction"] for r in runs]),
            "train_A": mean_std([r["train"]["A"] for r in runs]),
            "seed_level": runs,
        }

    rew_by_seed: dict[int, list] = {}
    for r in rewired_runs:
        rew_by_seed.setdefault(r["seed"], []).append(r)
    rew_seed = []
    for seed, rs in sorted(rew_by_seed.items()):
        rew_seed.append(
            {
                "seed": seed,
                "test": {
                    "H": float(np.mean([x["test"]["H"] for x in rs])),
                    "A": float(np.mean([x["test"]["A"] for x in rs])),
                    "S": float(np.mean([x["test"]["S"] for x in rs])),
                    "G": float(np.mean([x["test"]["G"] for x in rs])),
                    "grp_selection_fraction": float(
                        np.mean([x["test"]["grp_selection_fraction"] for x in rs])
                    ),
                },
                "train": {
                    "A": float(np.mean([x["train"]["A"] for x in rs])),
                    "H": float(np.mean([x["train"]["H"] for x in rs])),
                },
                "per_mask": [
                    {
                        "mask_id": x["mask_id"],
                        "test_A": x["test"]["A"],
                        "test_H": x["test"]["H"],
                        "grp_selection": x["test"]["grp_selection_fraction"],
                    }
                    for x in rs
                ],
            }
        )

    learned = load_learned_test_A_from_report()
    gf_table = {
        "Native": agg(gf_runs["Native"]),
        "Ambient": agg(gf_runs["Ambient"]),
        "Rewired": {
            "test_H": mean_std([r["test"]["H"] for r in rew_seed]),
            "test_A": mean_std([r["test"]["A"] for r in rew_seed]),
            "test_S": mean_std([r["test"]["S"] for r in rew_seed]),
            "test_G": mean_std([r["test"]["G"] for r in rew_seed]),
            "test_grp_selection": mean_std(
                [r["test"]["grp_selection_fraction"] for r in rew_seed]
            ),
            "seed_level": rew_seed,
            "per_mask_runs": rewired_runs,
        },
    }
    paired = {}
    for arm in ("Native", "Ambient", "Rewired"):
        gf_A = gf_table[arm]["test_A"]["mean"]
        gf_H = gf_table[arm]["test_H"]["mean"]
        learn_A = learned[arm]["A"]
        learn_H = learned[arm]["H"]
        paired[arm] = {
            "GroupedFirst_test_A": gf_A,
            "GroupedFirst_test_H": gf_H,
            "learned_test_A": learn_A,
            "learned_test_H": learn_H,
            "delta_A_GF_minus_learned": gf_A - learn_A,
            "delta_H_GF_minus_learned": gf_H - learn_H,
        }

    # Inspect Native exceptions
    native_misses = [r for r in all_records if r["arm"] == "Native" and r["miss_despite_h"]]
    native_fails = [r for r in all_records if r["arm"] == "Native" and r["a"] == 0]
    return {
        "GroupedFirst": gf_table,
        "paired_vs_learned_01722": paired,
        "ForceSeq_learned_reference": learned.get("ForceSeq"),
        "native_n_fail_test": sum(
            1 for r in all_records if r["arm"] == "Native" and r["split"] == "test" and r["a"] == 0
        ),
        "native_n_miss_despite_h_test": sum(
            1
            for r in all_records
            if r["arm"] == "Native" and r["split"] == "test" and r["miss_despite_h"]
        ),
        "native_fail_examples_cap": native_fails[:20],
        "records": all_records,
    }


def one_liners(order_audit: dict, graph_audit: dict, gf: dict) -> dict:
    # Order
    bits = []
    for s in map(str, SEEDS):
        ps = order_audit["per_seed"][s]
        bits.append(
            f"seed{s}: ties={ps['n_tied_tokens']} tokens {ps['tied_tokens']}; "
            f"first-max {ps['first_max_match_over_16']}/16; "
            f"perm_match16=[{ps['min_perm_match16']},{ps['max_perm_match16']}] "
            f"preserve_map={ps['all_perms_preserve_physical_map']}; "
            f"altH=[{ps['alt_test_H_range']['min']},{ps['alt_test_H_range']['max']}]"
        )
    cond = order_audit["primary_map_conditional_on_favorable_first_max"]
    order_line = (
        ("CONDITIONAL on favorable first-max. " if cond else "No ties. ")
        + "CORRECTED perm scoring (physical from permuted argmax). "
        + " | ".join(bits)
    )
    # Graph
    cc = graph_audit["classification_counts"]
    graph_line = (
        f"Saved rewired masks (n={graph_audit['n_masks']}): "
        f"identical_labeled={cc.get('identical_labeled', 0)}, "
        f"isomorphic_under_relabeling={cc.get('isomorphic_under_relabeling', 0)}, "
        f"non_isomorphic={cc.get('non_isomorphic', 0)}. "
        "017.22 'non_isomorphic_edge_set' only meant different labeled edges — superseded."
    )
    # GF
    nat = gf["paired_vs_learned_01722"]["Native"]
    gf_line = (
        f"Native GroupedFirst test A={nat['GroupedFirst_test_A']:.6f} "
        f"(H={nat['GroupedFirst_test_H']:.6f}) vs learned Native A={nat['learned_test_A']:.6f}; "
        f"ΔA(GF−learned)={nat['delta_A_GF_minus_learned']:.6f}"
    )
    return {"order": order_line, "graph": graph_line, "grouped_first": gf_line}


def main() -> None:
    t0 = time.time()
    alg = SedenionAlgebra()
    geo = Geometry(alg)
    pgeo = PhiGeometry(geo)
    fte = float_to_exact_map(geo)
    recovered = load_recovered_maps(geo)
    bench = materialize_benchmark(geo, pgeo, fte, recovered)
    assert bench["manifest_sha256"] == PINNED_MANIFEST_SHA, (
        f"manifest {bench['manifest_sha256']} != pinned {PINNED_MANIFEST_SHA}"
    )

    synth = synthetic_order_regressions()
    assert synth["ok"], synth

    order_audit = corrected_tie_and_order_audit(geo, fte, bench)
    graph_audit = audit_saved_rewired_masks()
    gf = run_grouped_first_suite(bench, fte)
    lines = one_liners(order_audit, graph_audit, gf)

    # Drop bulky records from main report; write separately
    records = gf.pop("records")
    report = {
        "task": "017.23",
        "result": "017.24",
        "manifest_sha256": bench["manifest_sha256"],
        "pinned_source_commit": "0f9e4c9d8f3f129554927ef429d9a15344992831",
        "no_new_training": True,
        "synthetic_order_regressions": synth,
        "corrected_order_audit_summary": {
            "conditional": order_audit["primary_map_conditional_on_favorable_first_max"],
            "per_seed": {
                s: {
                    "n_tied": order_audit["per_seed"][s]["n_tied_tokens"],
                    "tied_tokens": order_audit["per_seed"][s]["tied_tokens"],
                    "first_max_match": order_audit["per_seed"][s]["first_max_match_over_16"],
                    "min_perm_match16": order_audit["per_seed"][s]["min_perm_match16"],
                    "max_perm_match16": order_audit["per_seed"][s]["max_perm_match16"],
                    "all_perms_preserve_physical_map": order_audit["per_seed"][s][
                        "all_perms_preserve_physical_map"
                    ],
                    "alt_H": order_audit["per_seed"][s]["alt_test_H_range"],
                }
                for s in map(str, SEEDS)
            },
            "corrected_scoring_note": order_audit["corrected_scoring_note"],
        },
        "graph_audit_summary": {
            "classification_counts": graph_audit["classification_counts"],
            "n_masks": graph_audit["n_masks"],
            "note": graph_audit["note"],
            "per_mask_classifications": {
                s: [
                    {
                        "mask_id": m["mask_id"],
                        "classification": m["classification"],
                        "degree_sequence_match": m["degree_sequence_match"],
                        "triangles": [m["triangle_count_native"], m["triangle_count_rewired"]],
                        "cc_sizes": [
                            m["connected_component_sizes_native"],
                            m["connected_component_sizes_rewired"],
                        ],
                        "self_loops": [m["native_self_loops"], m["rewired_self_loops"]],
                    }
                    for m in graph_audit["per_seed"][s]
                ]
                for s in graph_audit["per_seed"]
            },
        },
        "grouped_first": {
            k: (
                {
                    "test_A": v["test_A"],
                    "test_H": v["test_H"],
                    "test_S": v.get("test_S"),
                    "test_G": v.get("test_G"),
                    "test_grp_selection": v.get("test_grp_selection"),
                }
                if k != "records"
                else None
            )
            for k, v in gf["GroupedFirst"].items()
        },
        "paired_vs_learned_01722": gf["paired_vs_learned_01722"],
        "ForceSeq_learned_reference": gf["ForceSeq_learned_reference"],
        "native_n_fail_test": gf["native_n_fail_test"],
        "native_n_miss_despite_h_test": gf["native_n_miss_despite_h_test"],
        "one_liners": lines,
        "elapsed_sec": time.time() - t0,
        "deps": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
    }

    # Slim / tiny
    tiny = {
        "manifest_sha256": bench["manifest_sha256"],
        "GroupedFirst_Native_test_A": gf["paired_vs_learned_01722"]["Native"][
            "GroupedFirst_test_A"
        ],
        "learned_Native_test_A": gf["paired_vs_learned_01722"]["Native"]["learned_test_A"],
        "delta_A": gf["paired_vs_learned_01722"]["Native"]["delta_A_GF_minus_learned"],
        "order_one_liner": lines["order"],
        "graph_one_liner": lines["graph"],
        "graph_counts": graph_audit["classification_counts"],
        "synthetic_ok": synth["ok"],
    }

    (ART / "learned_admissibility_01723_report.json").write_text(json.dumps(report, indent=2))
    (ART / "learned_admissibility_01723_report_tiny.json").write_text(json.dumps(tiny, indent=2))
    (ART / "corrected_tie_order_audit.json").write_text(
        json.dumps(order_audit, indent=2, default=str)
    )
    (ART / "graph_iso_audit.json").write_text(json.dumps(graph_audit, indent=2))
    (ART / "grouped_first_per_example.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    # Full GF detail
    (ART / "grouped_first_detail.json").write_text(
        json.dumps(
            {
                "GroupedFirst": {
                    k: {kk: vv for kk, vv in v.items() if kk != "per_mask_runs"}
                    for k, v in gf["GroupedFirst"].items()
                },
                "paired_vs_learned_01722": gf["paired_vs_learned_01722"],
            },
            indent=2,
        )
    )

    # Attachments (lean)
    for name in (
        "learned_admissibility_01723_report_tiny.json",
        "learned_admissibility_01723_report.json",
        "corrected_tie_order_audit.json",
        "graph_iso_audit.json",
        "grouped_first_detail.json",
    ):
        (ATTACH / name).write_bytes((ART / name).read_bytes())
    # Cap per-example in attachments — store under artifacts; attach a small sample
    sample = records[:80]
    (ATTACH / "grouped_first_per_example_sample.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sample) + "\n"
    )

    print(json.dumps(tiny, indent=2))
    print("ORDER:", lines["order"])
    print("GRAPH:", lines["graph"])
    print("GF:", lines["grouped_first"])
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

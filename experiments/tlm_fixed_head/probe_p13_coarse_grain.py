#!/usr/bin/env python3
"""007 p=13 Theory-41 word-slot coarse-grain probe.

Conjecture (Outcome/GPT correction of 007.03):
  Words in one Fixed class share e_w, so Theory-27 cannot distinguish
  within-class slots. But q(w)=a+b mod 13 is well-defined for the
  injective residue→Event embed on {0..12}, and
    e_c = sum_{w: q(w)=c} e_w
  is Theory-41-legal in E. Ask whether the 13 e_c are distinct and
  whether preparations separate argmax regions.

Not a product head / Fork choice. Diagnostic only.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from topographo.ssd import fixed_head

P = 13
N_EVENTS = fixed_head.N_EVENTS
N_WORDS = fixed_head.N_WORDS
DIM = fixed_head.DIM


def embed(r: int) -> int:
    """Injective on {0..12} for N_EVENTS=84."""
    assert 0 <= r < P <= N_EVENTS
    return r  # r % 84 == r


@dataclass
class CoarseGrainReport:
    p: int
    n_pairs: int
    n_results: int
    embed_injective_residues: bool
    # effect geometry
    pairwise_hs_distance_min: float
    pairwise_hs_distance_max: float
    pairwise_hs_distance_mean: float
    n_near_duplicate_pairs: int  # HS dist < atol
    span_rank_of_e_c: int
    # class multiset view: e_c = (1/N_WORDS) sum_k n_ck Q_k
    class_count_matrix_shape: tuple[int, int]
    class_count_row_sums: list[int]
    # preparation separation
    n_random_preparations: int
    n_results_that_win_argmax: int
    results_with_argmax_win: list[int]
    soft_margin_mean_when_win: float
    soft_margin_min_when_win: float
    # optional: try Fixed subspace / generators as states
    generator_prep_argmax: dict[str, int]
    notes: list[str]
    fence: str


def _hs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.trace(a.T @ b))


def _hs_norm(a: np.ndarray) -> float:
    return math.sqrt(max(0.0, _hs(a, a)))


def class_representatives(census, basis) -> list[np.ndarray]:
    """One E_Fix(M_w) matrix per Fixed class (not /N_WORDS)."""
    kraus = fixed_head.basic_kraus_operators()
    reps: list[np.ndarray | None] = [None] * fixed_head.EXPECTED_N_CLASSES
    for rec in census.records:
        if reps[rec.class_id] is None:
            moment = fixed_head.two_step_moment(kraus[rec.b], kraus[rec.a])
            reps[rec.class_id] = fixed_head.e_fix(moment, basis)
    assert all(r is not None for r in reps)
    return reps  # type: ignore


def build_e_c(census, basis) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Return (e_c list length p, n_ck matrix shape (p, 15), notes)."""
    notes = []
    # verify injective embed
    embeds = [embed(r) for r in range(P)]
    injective = len(set(embeds)) == P
    notes.append(f"residue embed injective on 0..{P-1}: {injective}")

    reps = class_representatives(census, basis)
    table = census.class_id_table
    n_ck = np.zeros((P, fixed_head.EXPECTED_N_CLASSES), dtype=np.int64)
    # Also accumulate e_c directly from word effects for the 169 probe words
    e_c = [np.zeros((DIM, DIM), dtype=np.float64) for _ in range(P)]
    for a in range(P):
        for b in range(P):
            c = (a + b) % P
            ea, eb = embed(a), embed(b)
            cid = int(table[eb, ea])
            n_ck[c, cid] += 1
            # e_w = (1/N_WORDS) Q_{class(w)}
            e_c[c] += reps[cid] / N_WORDS

    # sanity: e_c == (1/N_WORDS) sum_k n_ck Q_k
    for c in range(P):
        recon = np.zeros((DIM, DIM), dtype=np.float64)
        for k in range(fixed_head.EXPECTED_N_CLASSES):
            recon += n_ck[c, k] * reps[k]
        recon /= N_WORDS
        if not np.allclose(e_c[c], recon, atol=1e-10):
            notes.append(f"WARN recon mismatch at c={c}")
    notes.append(
        "e_c sums over the p^2 embedded modular words only "
        f"(n={P*P}), each contributing e_w=Q_class/N_WORDS; "
        "Theory-41 packing of those slots."
    )
    return e_c, n_ck, notes


def pairwise_distances(e_c: list[np.ndarray], atol: float = 1e-10):
    p = len(e_c)
    dists = []
    near = 0
    for i in range(p):
        for j in range(i + 1, p):
            d = _hs_norm(e_c[i] - e_c[j])
            dists.append(d)
            if d < atol:
                near += 1
    arr = np.asarray(dists, dtype=np.float64)
    return {
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "near_duplicates": near,
        "all_dists": arr,
    }


def span_rank(e_c: list[np.ndarray]) -> int:
    stacked = np.column_stack([m.reshape(-1) for m in e_c])
    return int(np.linalg.matrix_rank(stacked, tol=1e-8))


def readout(x: np.ndarray, effect: np.ndarray) -> float:
    return fixed_head.readout_probability(x, effect)


def preparation_argmax_survey(e_c: list[np.ndarray], n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = len(e_c)
    wins = Counter()
    margins = []
    for _ in range(n):
        x = rng.normal(size=DIM)
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            continue
        x = x / nrm
        scores = np.array([readout(x, e) for e in e_c], dtype=np.float64)
        # numerical: if all nearly equal, skip margin
        order = np.argsort(scores)
        best, second = int(order[-1]), int(order[-2])
        wins[best] += 1
        margins.append(float(scores[best] - scores[second]))
    return wins, margins


def generator_preps(basis, e_c: list[np.ndarray]) -> dict[str, int]:
    """Argmax under a few structured preparations (not a training loop)."""
    eye = np.eye(DIM)
    cands = {
        "e0": eye[0],
        "e8": eye[8],
        "e0+e8": (eye[0] + eye[8]) / math.sqrt(2),
        "v0": basis.Wp[:, 0],
        "v8": basis.Wp[:, 1],
        "uniform": np.ones(DIM) / math.sqrt(DIM),
    }
    out = {}
    for name, x in cands.items():
        scores = [readout(x, e) for e in e_c]
        out[name] = int(np.argmax(scores))
    return out


def main() -> int:
    census = fixed_head.assert_census()
    basis = fixed_head.fixed_basis()
    e_c, n_ck, notes = build_e_c(census, basis)
    dist = pairwise_distances(e_c)
    rank = span_rank(e_c)
    wins, margins = preparation_argmax_survey(e_c, n=20000, seed=0)
    win_results = sorted(wins.keys())
    margins_arr = np.asarray(margins, dtype=np.float64) if margins else np.array([0.0])
    # margins only for wins that are unique max — already recorded
    gen = generator_preps(basis, e_c)

    # score matrix diagnostics: mean diagonal dominance under random x that prefer c
    # (already have win counts)

    report = CoarseGrainReport(
        p=P,
        n_pairs=P * P,
        n_results=P,
        embed_injective_residues=True,
        pairwise_hs_distance_min=dist["min"],
        pairwise_hs_distance_max=dist["max"],
        pairwise_hs_distance_mean=dist["mean"],
        n_near_duplicate_pairs=dist["near_duplicates"],
        span_rank_of_e_c=rank,
        class_count_matrix_shape=(int(n_ck.shape[0]), int(n_ck.shape[1])),
        class_count_row_sums=[int(x) for x in n_ck.sum(axis=1).tolist()],
        n_random_preparations=int(sum(wins.values())),
        n_results_that_win_argmax=len(wins),
        results_with_argmax_win=win_results,
        soft_margin_mean_when_win=float(margins_arr.mean()),
        soft_margin_min_when_win=float(margins_arr.min()),
        generator_prep_argmax=gen,
        notes=notes,
        fence="configured formal Fixed effects + Theory-41 slot sum; not physical Test Realization",
    )

    # richer JSON
    out = {
        "report": asdict(report),
        "win_counts": {str(k): int(v) for k, v in sorted(wins.items())},
        "n_ck": n_ck.tolist(),
        "e_c_hs_norms": [_hs_norm(e) for e in e_c],
        "pairwise_hs_distance_min_pair": None,
        "interpretation": {
            "if_near_duplicate_pairs_0_and_rank_gt_1": "e_c geometrically distinct as matrices",
            "if_n_results_that_win_argmax_equals_p": "every modular residue is an argmax for some preparation",
            "if_n_results_that_win_argmax_lt_p": "some c never win argmax under sampled preparations — weak/no separation",
            "theory27_within_class": "settled: equal e_w => identical P(w|x); this probe tests coarse-grained e_c instead",
        },
    }
    # find closest pair
    best = (1e9, -1, -1)
    for i in range(P):
        for j in range(i + 1, P):
            d = _hs_norm(e_c[i] - e_c[j])
            if d < best[0]:
                best = (d, i, j)
    out["pairwise_hs_distance_min_pair"] = {"dist": best[0], "i": best[1], "j": best[2]}

    path = Path(__file__).resolve().parent / "p13_coarse_grain_report.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["report"], indent=2))
    print("win_counts", out["win_counts"])
    print("min_pair", out["pairwise_hs_distance_min_pair"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

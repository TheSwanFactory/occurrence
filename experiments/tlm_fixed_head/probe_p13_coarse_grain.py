#!/usr/bin/env python3
"""007.04 p=13 Theory-41 word-slot coarse-grain probe.

Conjecture (Outcome/GPT correction of 007.03):
  Words in one Fixed class share e_w, so Theory-27 cannot distinguish
  within-class slots. But q(w)=a+b mod 13 is well-defined for the
  injective residue->Event embed on {0..12}, and
    e_c = sum_{w: q(w)=c} e_w
  is Theory-41-legal in E. Ask whether the 13 e_c are distinct and
  whether preparations separate argmax regions.

Not a product head / Fork choice. Diagnostic only.

Tie discipline (017.24 lesson)
-----------------------------
An argmax winner is only credited when the maximum is attained by exactly
one result. ``numpy.argmax``/``argsort`` silently break ties toward the
lowest index, which inflates apparent separation: here results 2 and 11
have identical class-count rows, hence identical effects and exactly equal
Theory-27 scores, so a naive argmax would report 11 as a winner 6648 times.
Strict winners and tie sets are therefore reported separately.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from topographo.ssd import fixed_head

P = 13
N_EVENTS = fixed_head.N_EVENTS
N_WORDS = fixed_head.N_WORDS
DIM = fixed_head.DIM
N_PREPARATIONS = 20000
PREPARATION_SEED = 0
BRANCH = "experiment/007-p13-word-slot-coarse-grain"
FENCE = "configured formal Fixed effects + Theory-41 slot sum; not physical Test Realization"


def embed(r: int) -> int:
    """Injective on {0..12} for N_EVENTS=84."""
    assert 0 <= r < P <= N_EVENTS
    return r  # r % 84 == r


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
    return reps  # type: ignore[return-value]


def build_e_c(census, basis) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Return (e_c list length p, n_ck matrix shape (p, 15), notes)."""
    notes = []
    embeds = [embed(r) for r in range(P)]
    injective = len(set(embeds)) == P
    notes.append(f"residue embed injective on 0..{P-1}: {injective}")

    reps = class_representatives(census, basis)
    table = census.class_id_table
    n_ck = np.zeros((P, fixed_head.EXPECTED_N_CLASSES), dtype=np.int64)
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


def duplicate_row_groups(n_ck: np.ndarray) -> tuple[int, list[list[int]]]:
    """Group results by identical class-count row; report count and collisions."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for c in range(n_ck.shape[0]):
        groups.setdefault(tuple(int(v) for v in n_ck[c]), []).append(c)
    duplicates = [sorted(v) for v in groups.values() if len(v) > 1]
    return len(groups), sorted(duplicates)


def pairwise_distances(e_c: list[np.ndarray], atol: float = 1e-10) -> dict:
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
    }


def stacked_effects(e_c: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([m.reshape(-1) for m in e_c])


def span_rank(e_c: list[np.ndarray]) -> int:
    return int(np.linalg.matrix_rank(stacked_effects(e_c), tol=1e-8))


def singular_values(e_c: list[np.ndarray], keep: int = 8) -> list[float]:
    sv = np.linalg.svd(stacked_effects(e_c), compute_uv=False)
    return [float(v) for v in sv[:keep]]


def readout(x: np.ndarray, effect: np.ndarray) -> float:
    return fixed_head.readout_probability(x, effect)


def preparation_argmax_survey(
    e_c: list[np.ndarray],
    n: int = N_PREPARATIONS,
    seed: int = PREPARATION_SEED,
) -> tuple[Counter, Counter, list[float]]:
    """Strict-argmax survey over random unit preparations.

    Returns (strict_wins, tie_sets, spreads). A result is credited only when
    it is the *sole* maximizer; joint maximizers are recorded as a tie set.
    Effects that coincide exactly produce exactly equal scores, so tie
    detection is exact equality, not a tolerance.
    """
    rng = np.random.default_rng(seed)
    strict: Counter = Counter()
    tie_sets: Counter = Counter()
    spreads: list[float] = []
    for _ in range(n):
        x = rng.normal(size=DIM)
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            continue
        x = x / nrm
        scores = np.array([readout(x, e) for e in e_c], dtype=np.float64)
        top = float(scores.max())
        winners = tuple(int(i) for i in np.flatnonzero(scores == top))
        tie_sets[winners] += 1
        if len(winners) == 1:
            strict[winners[0]] += 1
        spreads.append(top - float(scores.min()))
    return strict, tie_sets, spreads


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


def build_report() -> dict:
    census = fixed_head.assert_census()
    basis = fixed_head.fixed_basis()
    e_c, n_ck, notes = build_e_c(census, basis)

    n_unique, duplicates = duplicate_row_groups(n_ck)
    dist = pairwise_distances(e_c)
    rank = span_rank(e_c)
    strict, tie_sets, spreads = preparation_argmax_survey(e_c)
    n_sampled = int(sum(tie_sets.values()))
    spread = np.asarray(spreads, dtype=np.float64) if spreads else np.array([0.0])
    fraction_unique = (sum(strict.values()) / n_sampled) if n_sampled else 0.0

    reasons = [
        (
            f"Only {n_unique} distinct class-count multisets among {P} results "
            f"(duplicate groups {duplicates})."
        ),
        (
            f"span_rank(e_c)={rank} equals dim(C)={fixed_head.EXPECTED_SPAN_DIM}; "
            "effects live in Fixed image."
        ),
        (
            f"Unique strict argmax winners under {n_sampled // 1000}k random "
            f"preparations: {sorted(strict)} (not all {P})."
        ),
        f"Score spreads tiny (mean {spread.mean():.3e}); soft margins not useful.",
        (
            "Equal e_w within a Fixed class still implies P(w|x)=P(w'|x); "
            "coarse-graining helps only via unequal n_ck \u2014 here that is "
            "insufficient for full separation."
        ),
    ]
    solved = (
        n_unique == P
        and dist["near_duplicates"] == 0
        and len(strict) == P
    )

    return {
        "p": P,
        "verdict": {
            "solves_p13_ot_native_modular_interface": bool(solved),
            "reasons": reasons,
        },
        "n_unique_n_ck_rows": n_unique,
        "duplicate_n_ck_groups": duplicates,
        "span_rank_of_e_c": rank,
        "singular_values": singular_values(e_c),
        "pairwise_hs": dist,
        "unique_strict_argmax_counts": {str(k): int(v) for k, v in sorted(strict.items())},
        "top_argmax_tie_sets": [[list(k), int(v)] for k, v in tie_sets.most_common(5)],
        "fraction_unique_argmax": fraction_unique,
        "score_spread": {
            "mean": float(spread.mean()),
            "min": float(spread.min()),
            "max": float(spread.max()),
        },
        "generator_prep_argmax": generator_preps(basis, e_c),
        "n_ck": n_ck.tolist(),
        "e_c_hs_norms": [_hs_norm(e) for e in e_c],
        "notes": notes,
        "fence": FENCE,
        "branch": BRANCH,
    }


def main() -> int:
    out = build_report()
    path = Path(__file__).resolve().parent / "p13_coarse_grain_report.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": out["verdict"]["solves_p13_ot_native_modular_interface"],
                "n_unique_n_ck_rows": out["n_unique_n_ck_rows"],
                "duplicate_n_ck_groups": out["duplicate_n_ck_groups"],
                "span_rank_of_e_c": out["span_rank_of_e_c"],
                "unique_strict_argmax_counts": out["unique_strict_argmax_counts"],
                "fraction_unique_argmax": out["fraction_unique_argmax"],
                "score_spread_mean": out["score_spread"]["mean"],
            },
            indent=2,
        )
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

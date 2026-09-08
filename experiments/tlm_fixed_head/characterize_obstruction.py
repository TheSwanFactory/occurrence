#!/usr/bin/env python3
"""007.02 characterization probes — localize info loss; no new head / fork."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ernest/GitHub/occurrence")
sys.path.insert(0, str(ROOT))

from topographo.ssd import fixed_head  # noqa: E402


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h


def mi_from_joint(joint: Counter) -> float:
    """MI(X;Y) from joint counts keyed (x, y)."""
    x_counts: Counter = Counter()
    y_counts: Counter = Counter()
    for (x, y), n in joint.items():
        x_counts[x] += n
        y_counts[y] += n
    total = sum(joint.values())
    if total <= 0:
        return 0.0
    h_x, h_y = entropy(x_counts), entropy(y_counts)
    h_xy = entropy(joint)
    return h_x + h_y - h_xy


def embed(r: int, p: int) -> int:
    return int(r % p) % fixed_head.N_EVENTS


def quantize_coords(coords, bins_per_axis: int = 8):
    """Discretize float Fixed coords for MI (characterization only)."""
    # coords are Fraction-like or float 4-tuples in practice from records
    out = []
    for c in coords:
        v = float(c)
        # map roughly [-2, 3] into bins; clamp
        t = (v + 2.0) / 5.0
        t = min(max(t, 0.0), 0.999999)
        out.append(int(t * bins_per_axis))
    return tuple(out)


def analyze(p: int, census, basis, kraus, sample_residual: bool = True):
    table = census.class_id_table
    # index records by (b,a)
    by_ba = {(rec.b, rec.a): rec for rec in census.records}

    joint_class_c: Counter = Counter()
    joint_word_c: Counter = Counter()
    joint_coord_c: Counter = Counter()
    joint_rank_c: Counter = Counter()
    joint_g1_c: Counter = Counter()
    joint_g4_c: Counter = Counter()
    # within-class: accumulate (class, word) -> c distribution via conditional MI later
    class_word_c: dict[int, Counter] = defaultdict(Counter)
    class_counts: Counter = Counter()
    out_counts: Counter = Counter()

    residual_same_class_diff_c = []
    residual_same_class_same_c = []

    n_pairs = p * p
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            ea, eb = embed(a, p), embed(b, p)
            rec = by_ba[(eb, ea)]
            cid = int(rec.class_id)
            word = ea + 84 * eb
            qcoords = quantize_coords(rec.coords)
            joint_class_c[(cid, c)] += 1
            joint_word_c[(word, c)] += 1
            joint_coord_c[(qcoords, c)] += 1
            joint_rank_c[(rec.rank, c)] += 1
            joint_g1_c[(int(rec.coords[1] != 0), c)] += 1
            joint_g4_c[(int(rec.coords[3] != 0), c)] += 1
            class_word_c[cid][(word, c)] += 1
            class_counts[cid] += 1
            out_counts[c] += 1

    h_c = entropy(out_counts)
    # MI(c; ·) = H(c) - H(c|·)  equivalently from joint
    def mi_c_given_feature(joint: Counter) -> float:
        # joint keys (feat, c)
        feat_counts: Counter = Counter()
        for (f, c), n in joint.items():
            feat_counts[f] += n
        h_c_given = 0.0
        total = sum(joint.values())
        for f, n in feat_counts.items():
            sub = Counter({c: joint[(f, c)] for (ff, c) in joint if ff == f})
            # faster: rebuild per f
            h_c_given += (n / total) * entropy(sub)
        # rebuild properly
        by_f: dict = defaultdict(Counter)
        for (f, c), n in joint.items():
            by_f[f][c] += n
        h_c_given = sum((sum(cnt.values()) / total) * entropy(cnt) for cnt in by_f.values())
        return h_c - h_c_given

    mi_class = mi_c_given_feature(joint_class_c)
    mi_word = mi_c_given_feature(joint_word_c)
    mi_coord = mi_c_given_feature(joint_coord_c)
    mi_rank = mi_c_given_feature(joint_rank_c)
    mi_g1 = mi_c_given_feature(joint_g1_c)
    mi_g4 = mi_c_given_feature(joint_g4_c)

    # MI(c; word | class) = H(c|class) - H(c|class,word)
    # Since word determines class, H(c|class,word)=H(c|word)
    h_c_given_class = h_c - mi_class
    h_c_given_word = h_c - mi_word
    mi_word_given_class = h_c_given_class - h_c_given_word

    # Per-class: remaining H(c|class) and word diversity
    per_class = []
    for cid, n in sorted(class_counts.items()):
        by_w: dict = defaultdict(Counter)
        c_only: Counter = Counter()
        for (w, c), nn in class_word_c[cid].items():
            by_w[w][c] += nn
            c_only[c] += nn
        h_c_cls = entropy(c_only)
        # H(c|word, class=cid)
        total_c = sum(c_only.values())
        h_c_w = sum((sum(cnt.values()) / total_c) * entropy(cnt) for cnt in by_w.values()) if total_c else 0.0
        per_class.append({
            "class_id": cid,
            "n_pairs": n,
            "n_distinct_words": len(by_w),
            "n_distinct_c": len(c_only),
            "H_c_given_class": h_c_cls,
            "H_c_given_word_in_class": h_c_w,
            "MI_c_word_in_class": h_c_cls - h_c_w,
        })

    # Residual energy sample: for embedded pairs, compare same-class different-c vs same-c
    if sample_residual:
        # gather list of (a,b,c,cid,ea,eb)
        rows = []
        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                ea, eb = embed(a, p), embed(b, p)
                cid = int(table[eb, ea])
                rows.append((a, b, c, cid, ea, eb))
        # compute residual for unique words used
        words_needed = {(eb, ea) for *_, ea, eb in rows}
        resid = {}
        for eb, ea in words_needed:
            moment = fixed_head.two_step_moment(kraus[eb], kraus[ea])
            proj = fixed_head.e_fix(moment, basis)
            r = moment - proj
            resid[(eb, ea)] = float(np.trace(r.T @ r))
        # pair residuals by class
        by_cid = defaultdict(list)
        for a, b, c, cid, ea, eb in rows:
            by_cid[cid].append((c, resid[(eb, ea)], ea, eb))
        for cid, items in by_cid.items():
            for i in range(len(items)):
                for j in range(i + 1, min(i + 40, len(items))):  # cap pairs
                    c1, r1, *_ = items[i]
                    c2, r2, *_ = items[j]
                    mean_r = 0.5 * (r1 + r2)
                    if c1 != c2:
                        residual_same_class_diff_c.append(mean_r)
                    else:
                        residual_same_class_same_c.append(mean_r)

    def _stats(xs):
        if not xs:
            return None
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
        }

    return {
        "p": p,
        "H_c_bits": h_c,
        "MI": {
            "c_given_word": mi_word,
            "c_given_class": mi_class,
            "c_given_quantized_coords_8bins": mi_coord,
            "c_given_rank": mi_rank,
            "c_given_g1_nonzero": mi_g1,
            "c_given_g4_nonzero": mi_g4,
            "c_word_given_class": mi_word_given_class,
        },
        "interpretation_hints": {
            "if_coord_MI_near_class_MI": "class quantization is not the main choke; Fixed image already poor for c",
            "if_coord_MI_near_word_MI": "Fixed coords still carry task info; discrete 15-class quotient discards it",
            "if_MI_word_given_class_high": "within a Fixed class, word identity still predicts c — quotient is the choke",
            "if_MI_word_given_class_near_0": "once class is known, word adds little about c — collapse already in Fixed",
        },
        "per_class_summary": {
            "mean_MI_c_word_in_class": float(np.mean([x["MI_c_word_in_class"] for x in per_class])),
            "mean_H_c_given_class": float(np.mean([x["H_c_given_class"] for x in per_class])),
            "classes_with_single_c": sum(1 for x in per_class if x["n_distinct_c"] == 1),
        },
        "per_class": per_class,
        "residual_HS_energy": {
            "same_class_diff_c": _stats(residual_same_class_diff_c),
            "same_class_same_c": _stats(residual_same_class_same_c),
            "note": "mean HS energy of M-E_Fix(M) for embedded-pair comparisons; not a causal proof",
        },
        "fence": "configured probe + formal Fixed head; not physical Test Realization",
    }


def main():
    census = fixed_head.assert_census()
    basis = fixed_head.fixed_basis()
    kraus = fixed_head.basic_kraus_operators()
    out = {
        "census_checksum": census.checksum_sha256,
        "probes": "007.02 characterization: coord MI, within-class word MI, rank/g1/g4, residual energy",
        "p13": analyze(13, census, basis, kraus, sample_residual=True),
        "p97": analyze(97, census, basis, kraus, sample_residual=True),
    }
    path = Path(__file__).resolve().parent / "characterization_probe.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    # compact print
    for key in ("p13", "p97"):
        block = out[key]
        print(key, json.dumps({
            "H_c": block["H_c_bits"],
            "MI": block["MI"],
            "per_class_summary": block["per_class_summary"],
            "residual": block["residual_HS_energy"],
        }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()

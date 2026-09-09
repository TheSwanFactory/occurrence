#!/usr/bin/env python3
"""017.07 minimal Futurator program-space smoke (Outcome handoff).

Primary arm (strict OT):
  P_seq(a,b,r) = Occ(a, Occ(b,r))
  P_grp(a,b,r) = Occ(Cyc(a,b), r)

Control arm (generalized, NOT OT-native):
  Mul(Mul(u,v),w) vs Mul(u,Mul(v,w))
  and bilateral (z x)w vs z(x w) from 017.07 §10

Fence: configured exact arithmetic; not physical program selection / trainability.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from topographo.ssd.sedenion import SedenionAlgebra

DIM = 16
TOL = 1e-10


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _projective_equal(u: np.ndarray, v: np.ndarray, tol: float = TOL) -> bool:
    """Rays equal up to nonzero real scale."""
    nu, nv = _norm(u), _norm(v)
    if nu < tol or nv < tol:
        return nu < tol and nv < tol
    u = u / nu
    v = v / nv
    # same or opposite direction
    return abs(abs(float(u @ v)) - 1.0) < 1e-8


def _ray_key(v: np.ndarray, tol: float = TOL) -> tuple:
    """Deterministic projective representative key (sign-normalized)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = _norm(v)
    if n < tol:
        return ("zero",)
    v = v / n
    # flip so first nonzero coord is positive
    for x in v:
        if abs(x) > 1e-12:
            if x < 0:
                v = -v
            break
    return tuple(np.round(v, decimals=10).tolist())


@dataclass
class OccResult:
    defined: bool
    value: list[float] | None
    ray_key: tuple | None


def occ(alg: SedenionAlgebra, e: np.ndarray, r: np.ndarray) -> OccResult:
    """Occ(e,r)=[e*r] if e*r != 0."""
    prod = alg.mul(e, r)
    if _norm(prod) < TOL:
        return OccResult(False, None, None)
    return OccResult(True, prod.tolist(), _ray_key(prod))


def sand(alg: SedenionAlgebra, e: np.ndarray, r: np.ndarray) -> OccResult:
    """Sand(e,r)=[(e*r)*e] if nonzero."""
    er = alg.mul(e, r)
    if _norm(er) < TOL:
        return OccResult(False, None, None)
    out = alg.mul(er, e)
    if _norm(out) < TOL:
        return OccResult(False, None, None)
    return OccResult(True, out.tolist(), _ray_key(out))


def mul(alg: SedenionAlgebra, x: np.ndarray, y: np.ndarray) -> OccResult:
    prod = alg.mul(x, y)
    if _norm(prod) < TOL:
        return OccResult(False, None, None)
    return OccResult(True, prod.tolist(), _ray_key(prod))


def basis_vec(i: int) -> np.ndarray:
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def run_exact_witness(alg: SedenionAlgebra) -> dict:
    """017.07 §5 witness: z=e1+e10, w=e4+e15, x=e4."""
    z = basis_vec(1) + basis_vec(10)
    w = basis_vec(4) + basis_vec(15)
    x = basis_vec(4)

    # Cyc(z,w) on certified locus: μ = [z*w]
    zw = alg.mul(z, w)
    assert _norm(zw) > TOL, "Cyc domain failed: zw=0"

    # P_seq = Occ(z, Occ(w,x))
    inner = occ(alg, w, x)
    assert inner.defined
    p_seq = occ(alg, z, np.asarray(inner.value))
    assert p_seq.defined

    # P_grp = Occ(Cyc(z,w), x) = Occ(zw, x)
    p_grp = occ(alg, zw, x)
    assert p_grp.defined

    equal = p_seq.ray_key == p_grp.ray_key
    # Expected: [e1] vs [e1+e10]=[z]
    expected_seq = _ray_key(basis_vec(1))
    expected_grp = _ray_key(z)

    # Two-leaf Occ vs Sand (§3.2)
    one = basis_vec(0)  # 1 = e0
    # actually retained representative x=1 means e0
    occ_z1 = occ(alg, z, one)
    sand_z1 = sand(alg, z, one)

    # Generalized bilateral control §10: (z x)w vs z(x w)
    zx = alg.mul(z, x)
    left = mul(alg, zx, w)
    xw = alg.mul(x, w)
    right = mul(alg, z, xw)

    # Generalized three-leaf Mul trees on (z,w,x)
    mul_left = None
    mul_right = None
    zw_m = mul(alg, z, w)
    if zw_m.defined:
        mul_left = mul(alg, np.asarray(zw_m.value), x)  # (z w) x
    wx_m = mul(alg, w, x)
    if wx_m.defined:
        mul_right = mul(alg, z, np.asarray(wx_m.value))  # z (w x)

    return {
        "witness_inputs": {"z": "e1+e10", "w": "e4+e15", "x": "e4"},
        "strict": {
            "P_seq_ray": p_seq.ray_key,
            "P_grp_ray": p_grp.ray_key,
            "P_seq_equals_P_grp": equal,
            "matches_01707_expected": (
                p_seq.ray_key == expected_seq and p_grp.ray_key == expected_grp
            ),
            "expected_P_seq": expected_seq,
            "expected_P_grp": expected_grp,
            "zw_ray": _ray_key(zw),
        },
        "two_leaf_Occ_vs_Sand": {
            "Occ_z_1": occ_z1.ray_key,
            "Sand_z_1": sand_z1.ray_key,
            "distinct": occ_z1.ray_key != sand_z1.ray_key,
        },
        "generalized_control": {
            "bilateral_zx_w": left.ray_key if left.defined else None,
            "bilateral_z_xw": right.ray_key if right.defined else None,
            "bilateral_distinct": (
                left.defined and right.defined and left.ray_key != right.ray_key
            ),
            "Mul_zw_x": mul_left.ray_key if mul_left and mul_left.defined else None,
            "Mul_z_wx": mul_right.ray_key if mul_right and mul_right.defined else None,
            "Mul_trees_distinct": (
                mul_left is not None
                and mul_right is not None
                and mul_left.defined
                and mul_right.defined
                and mul_left.ray_key != mul_right.ray_key
            ),
            "fence": "generalized Mul is ambient sedenion algebra, not OT occurrence",
        },
    }


def scan_basic_event_pairs(alg: SedenionAlgebra, retained: np.ndarray) -> dict:
    """Scan all ordered pairs of 84 basic Events for P_seq vs P_grp.

    Cyc domain proxy: treat Cyc as [z*w] whenever z*w != 0 AND [z*w] is
    (approximately) parallel to some basic Event — a weak finite proxy, not
    full continuous C. Also report raw zw!=0 counts separately.
    """
    events = [np.asarray(z, dtype=np.float64) for z in alg.basis_zero_divisors()]
    assert len(events) == 84

    # Precompute basic event ray keys for membership test
    basic_keys = {_ray_key(e) for e in events}

    n = len(events)
    both_defined = 0
    disagree = 0
    agree = 0
    cyc_proxy_ok = 0
    examples_disagree = []

    for i in range(n):
        for j in range(n):
            a, b = events[i], events[j]
            zw = alg.mul(a, b)
            if _norm(zw) < TOL:
                continue
            # proxy: product ray lands on a basic Event ray
            if _ray_key(zw) not in basic_keys:
                continue
            cyc_proxy_ok += 1
            inner = occ(alg, b, retained)
            if not inner.defined:
                continue
            p_seq = occ(alg, a, np.asarray(inner.value))
            p_grp = occ(alg, zw, retained)
            if not (p_seq.defined and p_grp.defined):
                continue
            both_defined += 1
            if p_seq.ray_key == p_grp.ray_key:
                agree += 1
            else:
                disagree += 1
                if len(examples_disagree) < 5:
                    examples_disagree.append(
                        {"a_idx": i, "b_idx": j, "P_seq": p_seq.ray_key, "P_grp": p_grp.ray_key}
                    )

    return {
        "retained": "e4",
        "n_basic_events": n,
        "ordered_pairs": n * n,
        "cyc_proxy_basic_product_ray": cyc_proxy_ok,
        "both_programs_defined": both_defined,
        "agree": agree,
        "disagree": disagree,
        "examples_disagree": examples_disagree,
        "note": (
            "Cyc domain here is a finite proxy (zw nonzero and [zw] a basic Event ray), "
            "not the full continuous cyclic/FIPS locus C from Theory 065/40."
        ),
    }


def main() -> int:
    alg = SedenionAlgebra()
    # Sanity: confirm multiply convention roughly matches 017.07 numbers
    z = basis_vec(1) + basis_vec(10)
    w = basis_vec(4) + basis_vec(15)
    x = basis_vec(4)
    wx = alg.mul(w, x)
    zwx = alg.mul(z, wx)
    zw = alg.mul(z, w)
    zw_x = alg.mul(zw, x)

    out = {
        "source": "occurrence/outcome 017.07 minimal Futurator program-space smoke",
        "branch_intent": "experiment/017-futurator-program-space-smoke",
        "arithmetic_check": {
            "w_x": wx.tolist(),
            "z_wx": zwx.tolist(),
            "z_w": zw.tolist(),
            "zw_x": zw_x.tolist(),
            "note": "017.07 expects w*e4=-e0-e11, z(w e4)=-2 e1, zw=-2e5+2e14, (zw)e4=-2e1-2e10",
        },
        "exact_witness": run_exact_witness(alg),
        "fips_proxy_scan": scan_basic_event_pairs(alg, x),
        "fence": (
            "Strict Occ/Cyc arm is OT-shaped program comparison; Mul arm is ambient control. "
            "No claim of physical program selection, trainability, or modular usefulness."
        ),
    }
    path = Path(__file__).resolve().parent / "futurator_program_space_report.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "witness_ok": out["exact_witness"]["strict"]["matches_01707_expected"],
        "P_seq_eq_P_grp": out["exact_witness"]["strict"]["P_seq_equals_P_grp"],
        "Occ_vs_Sand_distinct": out["exact_witness"]["two_leaf_Occ_vs_Sand"]["distinct"],
        "Mul_bilateral_distinct": out["exact_witness"]["generalized_control"]["bilateral_distinct"],
        "Mul_trees_distinct": out["exact_witness"]["generalized_control"]["Mul_trees_distinct"],
        "scan": {
            k: out["fips_proxy_scan"][k]
            for k in ("cyc_proxy_basic_product_ray", "both_programs_defined", "agree", "disagree")
        },
    }, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

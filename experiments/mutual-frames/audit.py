"""Deterministic exact finite audit; --check verifies recorded results and traces."""
import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from runtime import (
    Cyclic,
    Native,
    OperationalFrame,
    action,
    advance,
    clear_caches,
    encode_frame,
    encode_ray,
    encode_round,
    exact,
    is_edge,
    is_event,
    ray,
    round_step,
    step,
)

ROOT = Path(__file__).resolve().parent
POLICIES = ("AB", "BA", "snapshot_AB", "snapshot_BA")


class Recorder:
    """Observer-only indices deduplicate traces; never passed to runtime."""
    def __init__(self):
        self.frames, self.frame_ids = [], {}
        self.decisions, self.decision_ids = [], {}
        self.cases = []

    def frame(self, f):
        if f is None:
            return None
        if f not in self.frame_ids:
            self.frame_ids[f] = len(self.frames)
            self.frames.append(encode_frame(f))
        return self.frame_ids[f]

    def decision(self, d):
        if d not in self.decision_ids:
            self.decision_ids[d] = len(self.decisions)
            self.decisions.append({"before": self.frame(d.before), "after": self.frame(d.after),
                "presented": None if d.presented is None else encode_ray(d.presented),
                "status": d.status, "guard": d.guard, "presentation_source": d.source,
                "law_source": "before.enacted_transition; Cyclic type enacts autonomous rotation",
                "admissibility_source": "certified predicates of before/presented",
                "post_state_source": "topographo exact left action / certified cyclic rotation",
                "global_selector": False})
        return self.decision_ids[d]

    def record(self, family, pair, rounds):
        self.cases.append([family, [self.frame(f) for f in pair],
                           [[self.decision(d) for d in r.decisions] for r in rounds]])


def semantic_signature(result):
    return tuple((d.after, d.status, d.guard) for d in result.decisions)


def basic_events():
    return tuple(ray(exact.add(exact.basis(i), exact.scale(s, exact.basis(8+j))))
                 for i in range(1, 8) for j in range(1, 8) if i != j for s in (-1, 1))


def label(e):
    i = next(i for i in range(1, 8) if e[i])
    j = next(j for j in range(1, 8) if e[8+j])
    p = i ^ j
    sigma = exact.mul(exact.basis(i), exact.basis(j))[p]
    return p, int(e[8+j]*sigma), tuple(sorted((i, j, p)))


def fips_signature(f):
    if f is None or isinstance(f.retained_state, Native):
        return None
    edge = f.retained_state
    t = action(edge.r, edge.s)
    # General rays need not be basic. This finite classifier is observer-only.
    return tuple(sorted((edge.r, edge.s, t)))


def evaluate_case(pair, family, counters, recorder, witnesses):
    rounds = tuple(round_step(pair, p) for p in POLICIES)
    ab, ba, staged, reverse = rounds
    assert semantic_signature(staged) == semantic_signature(reverse)
    c = counters[family]
    c["cases"] += 1
    for p, result in zip(POLICIES[:3], rounds):
        c[p+"_complete"] += result.complete
        for d in result.decisions:
            c[p+"_"+d.status] += 1
    c["AB_BA_same_signature"] += semantic_signature(ab) == semantic_signature(ba)
    c["AB_BA_both_complete_different_endpoint"] += ab.complete and ba.complete and ab.endpoint != ba.endpoint
    c["both_serial_equal_snapshot"] += semantic_signature(ab) == semantic_signature(ba) == semantic_signature(staged)
    c["seed_guard_order_difference"] += tuple(d.guard for d in ab.decisions) != tuple(d.guard for d in ba.decisions)
    c["status_order_difference"] += tuple(d.status for d in ab.decisions) != tuple(d.status for d in ba.decisions)
    c["fips_block_order_difference"] += tuple(fips_signature(d.after) for d in ab.decisions) != tuple(fips_signature(d.after) for d in ba.decisions)
    if ab.complete and ba.complete and ab.endpoint != ba.endpoint and family not in witnesses:
        witnesses[family] = {"initial": [encode_frame(f) for f in pair],
                             "policies": {p: encode_round(r) for p, r in zip(POLICIES[:3], rounds)}}
    recorder.record(family, pair, rounds[:3])
    return rounds


def run():
    events = basic_events()
    assert len(set(events)) == 84 and all(is_event(e) for e in events)
    edges = tuple(Cyclic(a, b) for a in events for b in events if is_edge(a, b))
    blocks = {tuple(sorted((e.r, e.s, action(e.r, e.s)))) for e in edges}
    assert len(edges) == 336 and len(blocks) == 56
    components = defaultdict(set)
    degrees = Counter()
    for block in blocks:
        labels = [label(e) for e in block]
        assert len({lab[:2] for lab in labels}) == 1
        assert len({lab[2] for lab in labels}) == 3
        components[labels[0][:2]].add(block)
        degrees.update(block)
    assert len(components) == 14 and set(degrees.values()) == {2}
    assert all(len(bs) == 4 and len(set().union(*map(set, bs))) == 6 for bs in components.values())
    print("Finite geometry: 84 Events, 336 ordered edges, 56 blocks, 14 Pasch components", flush=True)

    recorder, counters, witnesses = Recorder(), defaultdict(Counter), {}
    for la, lb in itertools.product(("founding", "sce"), repeat=2):
        family = f"NN:{la}/{lb}"
        for a, b in itertools.product(events, repeat=2):
            evaluate_case((OperationalFrame(Native(a), la), OperationalFrame(Native(b), lb)),
                          family, counters, recorder, witnesses)
        print(f"{family}: {dict(counters[family])}", flush=True)

    for law in ("founding", "sce"):
        family = f"NC:{law}"
        for a, edge in itertools.product(events, edges):
            pair = OperationalFrame(Native(a), law), OperationalFrame(edge)
            rounds = evaluate_case(pair, family, counters, recorder, witnesses)
            # Slot exchange tests CN without introducing an operational frame ID.
            for policy, reference in zip(("BA", "AB", "snapshot_BA"), rounds):
                swapped = round_step(pair[::-1], policy)
                assert semantic_signature(swapped)[::-1] == semantic_signature(reference)
        print(f"{family}: {dict(counters[family])}", flush=True)

    # Every basic cyclic ordered edge paired with every other: disjoint local writes.
    for ea, eb in itertools.product(edges, repeat=2):
        evaluate_case((OperationalFrame(ea), OperationalFrame(eb)), "CC", counters, recorder, witnesses)
    assert counters["CC"]["both_serial_equal_snapshot"] == 336**2
    print("CC: all 112896 ordered edge pairs commute", flush=True)

    # Exact canonical counterexample with fixed laws in both frames.
    a = ray(exact.add(exact.basis(1), exact.basis(10)))
    b = ray(exact.add(exact.basis(4), exact.basis(15)))
    pair = OperationalFrame(Native(a)), OperationalFrame(Native(b))
    w = {p: round_step(pair, p) for p in POLICIES}
    assert all(r.complete for r in w.values())
    assert w["AB"].endpoint != w["BA"].endpoint != w["snapshot_AB"].endpoint
    witnesses["canonical_sce"] = {"initial": [encode_frame(f) for f in pair],
        "policies": {p: encode_round(w[p]) for p in POLICIES[:3]}}

    # Freeze offered participants, then execute in either order: local execution
    # commutes when the encounter data, not only initial frames, is held fixed.
    fixed = pair[1].presentation_interface(), pair[0].presentation_interface()
    pending = [step(pair[i], fixed[i]) for i in (0, 1)]
    assert tuple(d.after for d in pending) == w["snapshot_AB"].endpoint

    # Explicit undefined outcomes and missing presentation; no physical exit added.
    zero_peer = ray(exact.sub(exact.basis(4), exact.basis(15)))
    assert step(OperationalFrame(Native(zero_peer)), a).status == "undefined_projective"
    assert OperationalFrame(Native(exact.one())).presentation_interface() is None
    assert step(OperationalFrame(Native(a)), None).status == "presentation_unavailable"
    witnesses["annihilation"] = encode_round(round_step((OperationalFrame(Native(a)), OperationalFrame(Native(zero_peer))), "AB"))

    # Long runs: no growing operative tape, phase index or history. Input state
    # suffices; observer traces are outside the runtime. All three cyclic phases.
    for edge in edges:
        f = OperationalFrame(edge)
        initial = f
        for _ in range(30):
            f = step(f, None).after
            assert len(fields(f)) == 2 and isinstance(f.retained_state, Cyclic)
        assert f == initial
    repeated = {}
    for policy in POLICIES[:3]:
        current = pair
        trajectory = []
        for _ in range(12):
            result = round_step(current, policy)
            trajectory.append(encode_round(result))
            assert result.complete
            current = result.endpoint
        repeated[policy] = trajectory

    # Sign-character G2 automorphisms of the chosen octonion basis; verify the
    # automorphisms on ALL 256 products before applying them to the witness.
    symmetry_checks = 0
    for mask in range(8):
        def g(x):
            return tuple(c * (-1 if (mask & (i % 8)).bit_count() % 2 else 1) for i, c in enumerate(x))
        for i, j in itertools.product(range(16), repeat=2):
            assert g(exact.mul(exact.basis(i), exact.basis(j))) == exact.mul(g(exact.basis(i)), g(exact.basis(j)))
            symmetry_checks += 1
        gp = tuple(OperationalFrame(Native(g(f.retained_state.x))) for f in pair)
        for p in POLICIES:
            got = round_step(gp, p)
            expected = tuple(OperationalFrame(Cyclic(g(f.retained_state.r), g(f.retained_state.s))) for f in w[p].endpoint)
            assert got.endpoint == expected
    # Every edge is already in the exhaustive census, avoiding a privileged
    # canonical coordinate choice; additionally verify independent ray scaling.
    for edge in edges:
        assert Cyclic(exact.scale(-3, edge.r), exact.scale(Fraction(2, 7), edge.s)) == edge

    # Rational non-basic partner directions: exact tests after the finite suite.
    general = 0
    partners = [e.s for e in edges if e.r == a]
    for coeffs in itertools.product((-1, 0, 1), repeat=4):
        if not any(coeffs):
            continue
        b_general = exact.zero()
        for coef, partner in zip(coeffs, partners):
            b_general = exact.add(b_general, exact.scale(coef, partner))
        e = Cyclic(a, b_general)
        assert advance(advance(advance(e))) == e
        for law in ("founding", "sce"):
            evaluate_case((OperationalFrame(Native(a), law), OperationalFrame(Native(e.s), law)),
                          "rational_partners:"+law, counters, recorder, witnesses)
        general += 1
    # Pure cache removal does not change the diagnostic witness.
    clear_caches()
    for p in POLICIES:
        assert round_step(pair, p) == w[p]
    summary = {
        "topographo": importlib.metadata.version("topographo"),
        "base_commit": "5395412c810b7edf2d3c1f0f2524bb8ab492b377",
        "primary_verdict": "D — NON-SERIALIZABLE / LAW-CONFLICT BOUNDARY",
        "finite_geometry": {"events": 84, "ordered_edges": 336, "blocks": 56, "pasch_components": 14},
        "census": {k: dict(sorted(v.items())) for k, v in sorted(counters.items())},
        "cyclic_repeated_steps": len(edges)*30, "symmetry_basis_checks": symmetry_checks,
        "rational_partner_vectors": general,
        "runtime_fields": [f.name for f in fields(OperationalFrame)],
        "witnesses": witnesses, "repeated_canonical_sce": repeated,
        "boundaries": ["Presentation wiring is explicit experimental input, not derived from OT.",
          "Snapshot equality holds conditional on a shared incoming pair; it does not implement a synchronization protocol.",
          "Cyclic frames ignore external offers; asymmetric execution is not reciprocal after entry.",
          "Undefined projective or unavailable presentation halts a valid trajectory; it is not a physical terminal state.",
          "No continuous universality or necessity of a new coordination primitive is inferred."]}
    trace = {"format": "068-v1", "policies": list(POLICIES[:3]),
        "schema": "case = [family, incoming frame indices, [A decision index,B decision index] per policy]; indices are observer-only",
        "snapshot_definition": "both reads use the recorded incoming pair; AB/BA use latest committed peer",
        "frames": recorder.frames, "decisions": recorder.decisions, "cases": recorder.cases}
    blob = gzip.compress(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    summary["trace_archive"] = {"cases": len(recorder.cases), "distinct_decisions": len(recorder.decisions),
        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    return summary, blob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, blob = run()
    result = json.dumps(summary, indent=2, sort_keys=True)+"\n"
    if args.check:
        assert (ROOT/"results.json").read_text() == result
        assert (ROOT/"traces.json.gz").read_bytes() == blob
        print("PASS: exact replay matches results.json and traces.json.gz", flush=True)
    else:
        (ROOT/"results.json").write_text(result)
        (ROOT/"traces.json.gz").write_bytes(blob)
        print("PASS: wrote results.json and traces.json.gz", flush=True)

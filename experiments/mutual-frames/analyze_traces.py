"""Derive complete-case ordering metrics from recorded traces, not counters."""
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze():
    t = json.loads(gzip.decompress((ROOT/"traces.json.gz").read_bytes()))
    metrics = defaultdict(Counter)
    for family, _, policies in t["cases"]:
        ab, ba, staged = [[t["decisions"][i] for i in ids] for ids in policies]
        c = metrics[family]
        c["cases"] += 1
        complete = all(d["after"] is not None for d in ab+ba)
        same = [d["after"] for d in ab] == [d["after"] for d in ba]
        c["both_serial_complete"] += complete
        c["both_serial_complete_same_endpoint"] += complete and same
        c["both_serial_complete_different_endpoint"] += complete and not same
        c["same_incomplete_endpoint"] += not complete and same
        c["complete_snapshot_matches_either_serial"] += all(d["after"] is not None for d in staged) and any(
            [d["after"] for d in staged] == [d["after"] for d in r] for r in (ab, ba))
        c["presented_input_order_difference"] += [d["presented"] for d in ab] != [d["presented"] for d in ba]
        for a, b in zip(ab, ba):
            if a["after"] is None or b["after"] is None:
                continue
            sa = t["frames"][a["after"]]["retained_state"]
            sb = t["frames"][b["after"]]["retained_state"]
            c["slot_post_type_difference"] += sa["type"] != sb["type"]
            c["slot_both_cyclic_edge_difference"] += sa["type"] == sb["type"] == "Cyclic" and sa != sb
    return {k: dict(sorted(v.items())) for k, v in sorted(metrics.items())}


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, sort_keys=True))

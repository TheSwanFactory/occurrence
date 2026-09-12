"""Issue 009 artifact driver for the abstract F_2 group machinery.

The reusable API lives in :mod:`topographo.core.f2_groups` (promoted in 0.8.3):
the Fano plane ``PG(2,2)``, ``GL(3,2)``, ``GL(2,2)``/``AGL(2,2)``, the ``K4``
edge action, and ``is_structure_preserving`` — the Arm C rejection predicate.
This file is the thin local wrapper that writes and replays
``009_artifacts/groups.json``, following the seam already used by
``experiments/mutual-frames/runtime.py`` against ``topographo.ssd.frames``.

Consumers (``arms.py``, ``locator_task.py``, ``locator_heads.py``,
``discovery_task.py``) import the package module directly as::

    from topographo.core import f2_groups as groups

so every ``groups.<name>`` call site is unchanged by the promotion.

THE CRITICAL FENCE (read this before building any scramble)
-----------------------------------------------------------
    A local permutation of the three nonzero pp values alone is NOT a valid
    destructive scramble, because every such permutation lies in
    GL(2,2) ~= S3 and therefore preserves the local XOR-third law
    d3 = d1 XOR d2.

Break the law somewhere it can actually be broken: see
``f2_groups.non_induced_edge_permutation_witnesses`` for the 696
non-structure-preserving edge permutations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topographo.core.f2_groups import certificate

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_artifacts"
OUTPUT = ARTIFACTS / "groups.json"


def audit() -> dict:
    """The Issue-009 group census, re-derived from the promoted package module."""

    return certificate()


def render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived groups audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches groups.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)

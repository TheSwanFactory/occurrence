"""014.03 — readable algebraic grokking trajectory (Coder execution).

Executes the frozen scientific task

    quilt+s3://protology#package=occurrence/gpt@a5c2c5a7fac78eae4ba2dacbf45c5951a99096dd5745d1bdc85e9593c5b2ce9f
        &path=issues/014-grokking-algebraic-representation-readout/
              014.02a-GPT-readable-algebraic-grokking-trajectory-Coder.md

which incorporates unchanged the full experiment specification frozen in
``014.02-Codex-EXECUTE-readable-algebraic-grokking-trajectory.md``.

Stages (each is a subcommand; every stage writes and reads back its bytes)::

    contracts        reconstruct + certify the relation, freeze the 48/96/24
                     benchmark, emit benchmark_contract.json / probe_contract.json
                     / environment.json / baselines
    probe-controls   validate the frozen representation instrument on the three
                     declared controls and apply the certified-SFP gate
    select-regime    TRAIN-only regime selection over the frozen 3x3 grid
    trajectory       one scored (seed, arm) long run at the frozen horizon
    capture          deterministic replay that dumps main-model states at
                     declared update counts (for the component transplant)
    transplant       the predeclared four-way component transplant

Scientific fences (from 014.02, restated so the code cannot drift):

* The main learner sees only 84 opaque Event token ids and the designated TRAIN
  relation examples. No S / FFF / PP / pp, no native ray coordinates, no habitat
  or block ids, no Fano labels, no SFP code words, no ADMIT matrices, no
  automorphism labels ever enter its input features.
* Certified structure is used only to build the frozen benchmark, to score, and
  to run post-hoc representation diagnostics on already-fixed model states.
* Nothing in the regime selector may compute, load, log or inspect ROLE_TEST,
  NOVEL_TEST, probe, SFP-factor or certified-equivalence quantities.
* This task does not test H_native.
"""

from __future__ import annotations

import os

# Deterministic CPU execution is a frozen requirement of 014.02 section 5. The
# thread caps must be set before torch (and its BLAS backend) is imported.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

ISSUE_DIR = Path(__file__).resolve().parent.parent
REPO = ISSUE_DIR.parents[1]
SFP_DIR = REPO / "experiments" / "sfp_representation"

# --- delivery bundles ------------------------------------------------------
#
# 014.03   the immutable literal execution. Its probe optimizer is the literal
#          reading of 014.02 section 4 ("Adam" + "weight decay 1e-3" = coupled
#          L2 inside torch.optim.Adam). It stopped at the predeclared BLOCKED
#          state because the certified-SFP positive control missed its gate.
#
# 014.03a  the superseding corrected execution, authorized by the Executive
#          after 014.03 isolated the cause. The ONLY change is the resolution of
#          that ambiguous phrase to decoupled weight decay, which preserves the
#          declared value 1e-3 and changes nothing else: benchmark, split, model,
#          probe form, hidden width, seed, steps, candidate grid, scored seeds,
#          horizon, schedules, thresholds and diagnostics are all unchanged.

BUNDLES = {
    "014.03": {
        "probe_optimizer": "Adam",
        "probe_optimizer_reading": "coupled_L2_inside_torch_optim_Adam",
        "character": "immutable literal execution; stopped at BLOCKED",
    },
    "014.03a": {
        "probe_optimizer": "AdamW",
        "probe_optimizer_reading": "decoupled_weight_decay_torch_optim_AdamW",
        "character": "superseding corrected execution",
    },
}

BUNDLE = "014.03"
HERE = ISSUE_DIR / "014.03-Code-attachments"
PINS_DIR = HERE / "pins"
CHECKPOINTS = HERE / "checkpoints"


def configure_bundle(bundle: str) -> None:
    """Point every output path and the probe optimizer at one delivery bundle."""

    global BUNDLE, HERE, PINS_DIR, CHECKPOINTS
    if bundle not in BUNDLES:
        raise SystemExit(f"BLOCKED: unknown bundle {bundle!r}")
    BUNDLE = bundle
    HERE = ISSUE_DIR / f"{bundle}-Code-attachments"
    PINS_DIR = HERE / "pins"
    CHECKPOINTS = HERE / "checkpoints"
    HERE.mkdir(parents=True, exist_ok=True)


def probe_optimizer_name() -> str:
    return BUNDLES[BUNDLE]["probe_optimizer"]

# ``experiments/sfp_representation`` uses flat intra-directory imports.
for _path in (str(REPO), str(SFP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TASK_URI = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "a5c2c5a7fac78eae4ba2dacbf45c5951a99096dd5745d1bdc85e9593c5b2ce9f"
    "&path=issues/014-grokking-algebraic-representation-readout/"
    "014.02a-GPT-readable-algebraic-grokking-trajectory-Coder.md"
)
SPEC_URI = (
    "quilt+s3://protology#package=occurrence/gpt@"
    "71e5cc70df2fb001d27790c33f8dbfa6317f1513384bfb8200561465705d8ade"
    "&path=issues/014-grokking-algebraic-representation-readout/"
    "014.02-Codex-EXECUTE-readable-algebraic-grokking-trajectory.md"
)

# --- pins declared by 014.02 -------------------------------------------------

PINNED_INPUTS = {
    "009.02-Code-attachments/task_dataset.json": "task_dataset.json",
    "009.02-Code-attachments/sfp_codec.json": "sfp_codec.json",
    "009.02-Code-attachments/arms.json": "arms.json",
    "014.01-GPT-Owner-future-grokking-representation-readout-frame.md": "014.01-frame.md",
    "007.05-Klein-fair-grokking-interface-conjecture.md": "007.05-fair-grokking.md",
    "009.02-Kiro-native-FIPS-vs-SFP-representation-ablation-result.md": "009.02-result.md",
    "009.08-Kiro-query-relative-PP-localization-result.md": "009.08-result.md",
    "13-query-relative-consequence-addressing.md": "13-surface.md",
    "outcome/021.05-Research-SFP-zero-completion-result.md": "021.05-outcome.md",
    "014.02-Codex-EXECUTE-readable-algebraic-grokking-trajectory.md": "014.02-task.md",
    "014.02a-GPT-readable-algebraic-grokking-trajectory-Coder.md": "014.02a-dispatch.md",
}

EXPECTED = {
    "events": 84,
    "blocks": 56,
    "ordered_edges": 336,
    "habitats": 14,
    "incidences": 168,
    "table_sha256": "eb31fba3dbc3a4bbccb0154bcb2c26bcbc5809e477078ddf3ef5839d8904cdea",
    "catalogue_sha256": "98f60ad174f452a08a3d79799b2d3f3ff2d61c098eb77dd10f16d016b1472097",
    "blocks_sha256": "477b57a373a2da9d202929c9770b5491b87f0d4e9709342077f30b7b5229bffc",
    "dataset_sha256": "7872755fb1c364b18c5ffff7dbd74d225166366d3d126b2e79c16e76624ef721",
    "chart_sha256": "38dbbfec0111c654778bbf873f0dbcfd2d2141ff9b8405b71d273bbfd8830e5e",
    "arm_b_digest": "767a26b2c1677a59d7b99f98512b5cf17539b3e3be0204456a86cb5fc4c76954",
    "arm_c_digest": "b2c0b35d83c037943254411c3af13976b389534d478b9445a12804bce9d585cd",
}

# --- frozen benchmark / model / probe constants ------------------------------

N_EVENTS = 84
N_BLOCKS = 56
N_NOVEL_BLOCKS = 8
ROLE_SEED_STRING = "014/role/v1"

EMB_DIM = 64
TRUNK_HIDDEN = 256
PAIR_FEATURES = 3 * EMB_DIM          # 192
REPR_DIM = 2 * EMB_DIM               # 128 = concat(input_emb, candidate_emb)
PROBE_FEATURES_FACTOR = 3
PROBE_HIDDEN = 64

PROBE_SEED = 314159
PROBE_STEPS = 5000
PROBE_LR = 0.01
PROBE_BETAS = (0.9, 0.999)
PROBE_EPS = 1e-8
PROBE_WEIGHT_DECAY = 1e-3
PROBE_GATE = 0.95

RANDOM_CONTROL_SEED = 271828
RANDOM_CONTROL_DIM = REPR_DIM

CANDIDATE_LRS = (0.0003, 0.001, 0.003)
CANDIDATE_WDS = (0.01, 0.1, 1.0)
CALIBRATION_SEEDS = (0, 1, 2, 3)
CALIBRATION_CAP = 4096
SELECT_BETAS = (0.9, 0.98)
SELECT_EPS = 1e-8

SCORED_SEEDS = (2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007)
HORIZON = 65536

T_GEN_THRESHOLD = 0.95
T_REPR_CONSISTENCY_THRESHOLD = 0.90
DELAYED_GAP = 2048
DELAYED_ROLE_AT_FIT = 0.50
PERSISTENCE_CHECKPOINTS = 2   # "and the next two scheduled checkpoints"


def behavioral_schedule() -> tuple[int, ...]:
    """0,1,2,4,...,1024, then every 512 updates through 65536."""

    points = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    points += list(range(1536, HORIZON + 1, 512))
    return tuple(points)


def probe_schedule() -> tuple[int, ...]:
    """0,64,128,256,512,1024,2048,4096, then every 4096 through 65536."""

    points = [0, 64, 128, 256, 512, 1024, 2048]
    points += list(range(4096, HORIZON + 1, 4096))
    return tuple(sorted(set(points)))


BEHAVIORAL_CHECKPOINTS = behavioral_schedule()
PROBE_CHECKPOINTS = probe_schedule()


# ---------------------------------------------------------------------------
# deterministic serialization helpers
# ---------------------------------------------------------------------------

def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> dict:
    """Write pretty JSON, then read the committed bytes back and verify them."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    raw = path.read_bytes()
    reloaded = json.loads(raw.decode("utf-8"))
    readback = {
        "path": str(path.relative_to(HERE)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload_digest": digest(payload),
        "readback_payload_digest": digest(reloaded),
        "readback_exact": digest(reloaded) == digest(payload),
    }
    if not readback["readback_exact"]:
        raise SystemExit(f"BLOCKED: readback of {path} did not match written payload")
    return readback


def role_index(triple: tuple[int, int, int]) -> int:
    """r = int(SHA256("014/role/v1|x0,x1,x2"), 16) mod 3 — frozen by 014.02."""

    key = f"{ROLE_SEED_STRING}|{triple[0]},{triple[1]},{triple[2]}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 3


# ---------------------------------------------------------------------------
# 0. reconstruct and certify the finite relation (two independent code routes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Relation:
    """The reconstructed certified finite relation, in opaque index space."""

    blocks: tuple[tuple[int, int, int], ...]        # 56, lex by index triple
    ordered_edges: tuple[tuple[int, int, int], ...]  # 336 (a, b, third)
    third_of_pair: dict[tuple[int, int], int]        # unordered pair -> third
    block_of_pair: dict[tuple[int, int], int]        # unordered pair -> block id
    habitat_of_event: tuple[int, ...]                # index -> habitat ordinal
    encodings: tuple[str, ...]
    certification: dict


def _route_native() -> tuple[
    tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...], tuple, tuple
]:
    """Route A: certified native projective ray algebra (fips_basic + task.py)."""

    import task as task_mod
    from topographo.ssd import fips_basic

    catalogue = task_mod.build_catalogue()
    index_of = {e: i for i, e in enumerate(catalogue.events)}

    blocks = sorted(tuple(sorted(triple)) for triple in catalogue.blocks)
    edges = sorted(
        (index_of[a], index_of[b], index_of[t])
        for a, b, t in fips_basic.ORDERED_EDGES
    )
    habitats = sorted(set(catalogue.habitats))
    habitat_of_event = tuple(habitats.index(h) for h in catalogue.habitats)
    return tuple(blocks), tuple(edges), habitat_of_event, catalogue


def _route_sfp() -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Route B: SFP K4/torsor incidence law (endpoint intersection + pp XOR).

    Structurally independent of Route A: admission is decided by intersecting K4
    endpoint block coordinates inside a habitat and the forced third is obtained
    by XOR of the two incident displacements. No pair lookup table participates.
    """

    from sfp import ExactSfpCircuit, SfpCodec
    from topographo.ssd import fips_basic

    codec = SfpCodec()
    circuit = ExactSfpCircuit()
    events = tuple(fips_basic.EVENTS)
    index_of = {e: i for i, e in enumerate(events)}
    addresses = {e: codec.encode(e) for e in events}

    edges: list[tuple[int, int, int]] = []
    blocks: set[tuple[int, int, int]] = set()
    for a, b in itertools.permutations(events, 2):
        forced = circuit.forced_third_event(addresses[a], addresses[b])
        if forced is None:
            continue
        t = codec.decode(forced)
        edges.append((index_of[a], index_of[b], index_of[t]))
        blocks.add(tuple(sorted((index_of[a], index_of[b], index_of[t]))))
    return tuple(sorted(blocks)), tuple(sorted(edges))


def canonical_certificate(blocks: tuple[tuple[int, int, int], ...]) -> dict:
    """Exact canonical certificate of a 3-uniform block relation.

    The certified relation decomposes into connected components under
    event-sharing. Each component is canonicalized by brute-force minimal
    lexicographic relabelling of its own events (components are small), so the
    certificate is label-independent and exact — no preferred SFP coordinates
    and no heuristic refinement are involved.
    """

    adjacency: dict[int, set[int]] = {}
    for triple in blocks:
        for x in triple:
            adjacency.setdefault(x, set()).update(t for t in triple if t != x)

    seen: set[int] = set()
    component_forms: list[str] = []
    component_sizes: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component))
        seen |= component
        members = sorted(component)
        local_blocks = [t for t in blocks if set(t) <= component]
        component_sizes.append([len(members), len(local_blocks)])
        if len(members) > 8:
            raise SystemExit(
                "BLOCKED: canonical certificate route assumes small components; "
                f"found one of size {len(members)}"
            )
        best: str | None = None
        for perm in itertools.permutations(range(len(members))):
            relabel = {members[i]: perm[i] for i in range(len(members))}
            form = canonical(
                sorted(tuple(sorted(relabel[x] for x in t)) for t in local_blocks)
            )
            if best is None or form < best:
                best = form
        component_forms.append(str(best))

    component_forms.sort()
    return {
        "method": (
            "connected components under event sharing; each component "
            "canonicalized by exhaustive minimal-lexicographic relabelling of "
            "its own events; component forms sorted and digested"
        ),
        "components": len(component_forms),
        "component_shapes": sorted(component_sizes),
        "distinct_component_forms": sorted(set(component_forms)),
        "certificate": digest(component_forms),
    }


def reconstruct() -> Relation:
    """Rebuild the frozen relation and certify it against every declared pin."""

    from topographo.ssd import fips_basic

    blocks_a, edges_a, habitat_of_event, catalogue = _route_native()
    blocks_b, edges_b = _route_sfp()

    counts_ok = (
        len(blocks_a) == EXPECTED["blocks"]
        and len(edges_a) == EXPECTED["ordered_edges"]
        and len(catalogue.events) == EXPECTED["events"]
    )
    routes_agree = blocks_a == blocks_b and edges_a == edges_b
    if not (counts_ok and routes_agree):
        raise SystemExit(
            "BLOCKED: relation identity mismatch — "
            f"counts_ok={counts_ok} routes_agree={routes_agree}"
        )

    third_of_pair: dict[tuple[int, int], int] = {}
    for a, b, t in edges_a:
        key = (min(a, b), max(a, b))
        if key in third_of_pair and third_of_pair[key] != t:
            raise SystemExit("BLOCKED: forced third is not swap invariant")
        third_of_pair[key] = t
    if len(third_of_pair) != 3 * EXPECTED["blocks"]:
        raise SystemExit(
            f"BLOCKED: {len(third_of_pair)} unordered admitted pairs, expected 168"
        )

    block_of_pair: dict[tuple[int, int], int] = {}
    for block_id, triple in enumerate(blocks_a):
        for x, y in itertools.combinations(triple, 2):
            block_of_pair[(x, y)] = block_id
    if set(block_of_pair) != set(third_of_pair):
        raise SystemExit("BLOCKED: block pair set differs from admitted pair set")

    # pinned digests
    dataset = __import__("task").build_dataset(catalogue)
    observed = {
        "table_sha256": fips_basic.TABLE_SHA256,
        "catalogue_sha256": catalogue.sha256(),
        "blocks_sha256": digest([list(t) for t in catalogue.blocks]),
        "dataset_sha256": dataset.sha256(),
        "chart_sha256": __import__("sfp").SfpCodec().chart()["sha256"],
    }
    pin_rows = {
        name: {
            "expected": EXPECTED[name],
            "observed": observed[name],
            "agrees": EXPECTED[name] == observed[name],
        }
        for name in observed
    }
    if not all(row["agrees"] for row in pin_rows.values()):
        raise SystemExit(f"BLOCKED: pinned digest mismatch: {canonical(pin_rows)}")

    degrees = sorted({sum(1 for t in blocks_a if x in t) for x in range(N_EVENTS)})
    habitat_pure = all(
        len({habitat_of_event[x] for x in triple}) == 1 for triple in blocks_a
    )
    certificate = canonical_certificate(blocks_a)

    certification = {
        "routes": {
            "A": (
                "topographo.ssd.fips_basic certified projective ray algebra "
                "(EVENTS / ORDERED_EDGES / BLOCKS) indexed through "
                "experiments/sfp_representation/task.py build_catalogue()"
            ),
            "B": (
                "experiments/sfp_representation/sfp.py SfpCodec + ExactSfpCircuit: "
                "habitat K4 endpoint intersection with forced third by pp XOR, "
                "no pair lookup table"
            ),
            "independent_agreement_exact": routes_agree,
        },
        "counts": {
            "events": len(catalogue.events),
            "blocks": len(blocks_a),
            "ordered_edges": len(edges_a),
            "unordered_admitted_pairs": len(third_of_pair),
            "habitats": len(set(habitat_of_event)),
            "block_degree_spectrum": degrees,
        },
        "pins": pin_rows,
        "every_block_lies_in_one_habitat": habitat_pure,
        "forced_third_swap_invariant": True,
        "canonical_certificate": certificate,
        "topographo_version": importlib.metadata.version("topographo"),
    }

    return Relation(
        blocks=blocks_a,
        ordered_edges=edges_a,
        third_of_pair=third_of_pair,
        block_of_pair=block_of_pair,
        habitat_of_event=habitat_of_event,
        encodings=tuple(catalogue.encodings),
        certification=certification,
    )


# ---------------------------------------------------------------------------
# 1. freeze the memorization-versus-consequence benchmark
# ---------------------------------------------------------------------------

def incidence_components(
    blocks: list[tuple[int, int, int]], n_events: int
) -> list[set[str]]:
    """Connected components of the block/Event incidence bipartite graph.

    Vertices are ``b<block position>`` and ``e<event index>``; isolated Events
    (Events lying in no retained block) are rejected before counting, exactly as
    014.02 section 1.1 directs.
    """

    adjacency: dict[str, set[str]] = {}
    for position, triple in enumerate(blocks):
        bnode = f"b{position}"
        adjacency.setdefault(bnode, set())
        for x in triple:
            enode = f"e{x}"
            adjacency[bnode].add(enode)
            adjacency.setdefault(enode, set()).add(bnode)
    for x in range(n_events):
        adjacency.setdefault(f"e{x}", set())

    components: list[set[str]] = []
    seen: set[str] = set()
    for node in sorted(adjacency):
        if node in seen or not adjacency[node]:
            continue                                  # isolated vertex: rejected
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component))
        seen |= component
        components.append(component)
    return components


def select_novel_blocks(relation: Relation) -> dict:
    """Lexicographically first eight pairwise Event-disjoint holdout blocks.

    Condition 2 of 014.02 section 1.1 asks that the retained incidence graph be
    "connected after isolated vertices, if any, are rejected". The certified
    relation is a disjoint union of 14 habitat components (every block lies in
    exactly one habitat), so a single globally connected incidence graph does not
    exist for ANY subset choice — including the empty one. The executable reading
    frozen here, decided before any model was built or trained, is therefore:

        * no retained-side fragmentation relative to the certified relation
          (the retained incidence graph has exactly the same component count as
          the full relation, i.e. no habitat is split), and
        * no isolated Event: every Event, including every Event of a held-out
          block, still lies in at least one retained training block.

    Both the literal global-connectivity fact and this reading are recorded in
    benchmark_contract.json for Owner adjudication.
    """

    blocks = list(relation.blocks)
    full_components = incidence_components(blocks, N_EVENTS)
    chosen: list[int] = []
    attempts = {"nodes_visited": 0, "disjointness_prunes": 0, "condition2_rejects": 0}

    def condition_two(candidate: list[int]) -> dict:
        held = {blocks[i] for i in candidate}
        retained = [t for t in blocks if t not in held]
        coverage = {x: 0 for x in range(N_EVENTS)}
        for triple in retained:
            for x in triple:
                coverage[x] += 1
        isolated = sorted(x for x, c in coverage.items() if c == 0)
        components = incidence_components(retained, N_EVENTS)
        return {
            "retained_blocks": len(retained),
            "isolated_events": isolated,
            "no_isolated_event": not isolated,
            "retained_components": len(components),
            "full_relation_components": len(full_components),
            "no_new_fragmentation": len(components) == len(full_components),
            "satisfied": (not isolated) and len(components) == len(full_components),
        }

    def search(start: int, used: set[int]) -> bool:
        if len(chosen) == N_NOVEL_BLOCKS:
            report = condition_two(chosen)
            if report["satisfied"]:
                return True
            attempts["condition2_rejects"] += 1
            return False
        for i in range(start, len(blocks)):
            attempts["nodes_visited"] += 1
            triple = set(blocks[i])
            if triple & used:
                attempts["disjointness_prunes"] += 1
                continue
            chosen.append(i)
            if search(i + 1, used | triple):
                return True
            chosen.pop()
        return False

    if not search(0, set()):
        raise SystemExit(
            "BLOCKED: no eight pairwise Event-disjoint blocks satisfy the declared "
            "retention condition; the number eight was not changed"
        )

    novel_ids = tuple(chosen)
    return {
        "novel_block_positions": list(novel_ids),
        "novel_block_triples": [list(blocks[i]) for i in novel_ids],
        "seen_block_positions": [i for i in range(len(blocks)) if i not in set(novel_ids)],
        "rule": (
            "enumerate the 56 unordered blocks in lexicographic order of their "
            "canonical Event-index triples; deterministic depth-first backtracking "
            "over increasing positions returns the lexicographically first set of "
            "eight pairwise Event-disjoint blocks satisfying the retention condition"
        ),
        "pairwise_event_disjoint": True,
        "search_accounting": attempts,
        "retention_report": condition_two(list(novel_ids)),
        "global_connectivity_note": {
            "certified_relation_components": len(full_components),
            "single_connected_incidence_graph_possible": len(full_components) == 1,
            "reading": (
                "every certified block lies in exactly one habitat, so the "
                "block/Event incidence graph of the full relation already has 14 "
                "components; literal global connectivity is unsatisfiable for any "
                "holdout choice including the empty one. The executable reading "
                "frozen before training is: no new fragmentation relative to the "
                "certified relation, and no isolated Event."
            ),
        },
    }


@dataclass(frozen=True)
class Benchmark:
    """The frozen 48/96/24 split in opaque index space."""

    novel_blocks: tuple[int, ...]
    train: tuple[tuple[int, int, int], ...]        # (a, b, target)
    role_test: tuple[tuple[int, int, int], ...]
    novel_test: tuple[tuple[int, int, int], ...]
    designated_role: dict[int, int]                # block position -> r
    holdout: dict


def build_benchmark(relation: Relation) -> Benchmark:
    holdout = select_novel_blocks(relation)
    novel = tuple(holdout["novel_block_positions"])
    novel_set = set(novel)

    train: list[tuple[int, int, int]] = []
    role_test: list[tuple[int, int, int]] = []
    novel_test: list[tuple[int, int, int]] = []
    designated: dict[int, int] = {}

    for position, triple in enumerate(relation.blocks):
        if position in novel_set:
            for target in triple:
                pair = tuple(sorted(x for x in triple if x != target))
                novel_test.append((pair[0], pair[1], target))
            continue
        r = role_index(triple)
        designated[position] = r
        target = triple[r]
        pair = tuple(sorted(x for x in triple if x != target))
        train.append((pair[0], pair[1], target))
        for other in triple:
            if other == target:
                continue
            alt = tuple(sorted(x for x in triple if x != other))
            role_test.append((alt[0], alt[1], other))

    train.sort()
    role_test.sort()
    novel_test.sort()

    sizes = (len(train), len(role_test), len(novel_test))
    if sizes != (48, 96, 24):
        raise SystemExit(f"BLOCKED: split sizes {sizes}, expected (48, 96, 24)")

    pairs = [tuple(row[:2]) for row in train + role_test + novel_test]
    if len(set(pairs)) != len(pairs):
        raise SystemExit("BLOCKED: query pairs are not disjoint across the three sets")

    for a, b, target in train + role_test + novel_test:
        if relation.third_of_pair[(a, b)] != target:
            raise SystemExit("BLOCKED: a split target disagrees with the certified third")

    return Benchmark(
        novel_blocks=novel,
        train=tuple(train),
        role_test=tuple(role_test),
        novel_test=tuple(novel_test),
        designated_role=designated,
        holdout=holdout,
    )


# ---------------------------------------------------------------------------
# 2. the three declared baselines
# ---------------------------------------------------------------------------

def baselines(relation: Relation, benchmark: Benchmark) -> dict:
    """Directed pair lookup, triad-set memorization, certified SFP relation."""

    sets = {
        "TRAIN": benchmark.train,
        "ROLE_TEST": benchmark.role_test,
        "NOVEL_TEST": benchmark.novel_test,
    }

    # 1. directed pair lookup: memorize only the observed query -> answer map.
    pair_lookup = {(a, b): t for a, b, t in benchmark.train}

    # 2. triad-set memorization: each TRAIN example reveals its whole triple.
    triads = {tuple(sorted((a, b, t))) for a, b, t in benchmark.train}
    triad_index: dict[tuple[int, int], int] = {}
    for triple in triads:
        for x, y in itertools.combinations(sorted(triple), 2):
            triad_index[(x, y)] = next(z for z in triple if z not in (x, y))

    def score(answer_fn) -> dict:
        rows = {}
        for name, examples in sets.items():
            covered = 0
            correct = 0
            for a, b, target in examples:
                answer = answer_fn(a, b)
                if answer is not None:
                    covered += 1
                    correct += int(answer == target)
            rows[name] = {
                "examples": len(examples),
                "coverage": covered / len(examples),
                "accuracy": correct / len(examples),
            }
        return rows

    return {
        "policy": (
            "an uncovered query counts as incorrect; coverage is reported "
            "separately so abstention is never scored as a hit"
        ),
        "directed_pair_lookup": {
            "description": (
                "memorizes the 48 observed unordered query pairs and their "
                "designated answers; nothing else"
            ),
            "results": score(lambda a, b: pair_lookup.get((a, b))),
        },
        "triad_set_memorization": {
            "description": (
                "stores the 48 whole triples revealed by TRAIN and answers any "
                "query pair contained in a stored triple"
            ),
            "stored_triads": len(triads),
            "results": score(lambda a, b: triad_index.get((a, b))),
        },
        "certified_sfp_relation": {
            "description": (
                "the nonlearned certified relation itself (reference oracle); "
                "solves all three sets exactly by construction"
            ),
            "results": score(lambda a, b: relation.third_of_pair.get((a, b))),
        },
        "declared_hierarchy_holds": True,
    }


# ---------------------------------------------------------------------------
# arm codes for the representation controls
# ---------------------------------------------------------------------------

def arm_control_codes() -> dict:
    """Load the frozen Arm B / Arm C codes and verify them against arms.json.

    The pinned artifact stores each Event's code as the two canonical incidence
    presentation strings ``S=..,FFF=..,PP=..,pp=..``. They are expanded here with
    the frozen 009 field layout (S:2, FFF:7, PP:4, pp:3 = 16 units per
    presentation) and pooled by SUM over the two presentations, which is
    order-free and lossless: habitat plus the unordered endpoint pair plus the
    displacement determines the address exactly. Sum pooling is used rather than
    concatenation because the upstream 009 pooling contract forbids privileging
    either presentation, and because no canonical presentation order exists.
    """

    import arms as arms_mod
    from sfp import SfpCodec
    from task import build_catalogue

    pinned = json.loads((PINS_DIR / "arms.json").read_text())
    catalogue = build_catalogue()
    codec = SfpCodec()

    layout = [(name, offset, width) for name, offset, width, _ in arms_mod.FIELD_LAYOUT]
    token_dim = arms_mod.SFP_TOKEN_DIM

    def parse(code: str) -> list[float]:
        vector = [0.0] * token_dim
        for field in code.split(","):
            key, value = field.split("=")
            name, offset, width = next(row for row in layout if row[0] == key)
            index = int(value, 2)
            if name == "FFF":
                slot = index - 1
            elif name == "pp":
                slot = index - 1
            else:
                slot = index
            if not 0 <= slot < width:
                raise SystemExit(f"BLOCKED: field {name}={value} outside its layout")
            vector[offset + slot] = 1.0
        return vector

    def expand(code_strings: list[str]) -> list[list[float]]:
        vectors = []
        for entry in code_strings:
            left, right = entry.split("|")
            a = parse(left)
            b = parse(right)
            vectors.append([a[i] + b[i] for i in range(token_dim)])
        return vectors

    # Regenerate Arm B locally from the certified codec and check it against the
    # pinned code strings, so the control is pinned rather than merely copied.
    regenerated_b = []
    for event in catalogue.events:
        address = codec.encode(event)
        parts = [
            f"S={p.s},FFF={p.fff:03b},PP={p.q:02b},pp={p.d:02b}"
            for p in address.incidences()
        ]
        regenerated_b.append("|".join(parts))

    pinned_b = pinned["arms"]["B_sfp"]["codes"]
    pinned_c = pinned["arms"]["C_scrambled"]["codes"]
    if len(pinned_b) != N_EVENTS or len(pinned_c) != N_EVENTS:
        raise SystemExit("BLOCKED: pinned arm codes do not cover 84 Events")

    return {
        "B_sfp": expand(pinned_b),
        "C_scrambled": expand(pinned_c),
        "audit": {
            "arm_b_regenerated_matches_pinned": regenerated_b == pinned_b,
            "arm_b_pinned_digest": {
                "expected": EXPECTED["arm_b_digest"],
                "observed": pinned["digests"]["arm_token_digests"]["B_sfp"],
                "agrees": pinned["digests"]["arm_token_digests"]["B_sfp"]
                == EXPECTED["arm_b_digest"],
            },
            "arm_c_pinned_digest": {
                "expected": EXPECTED["arm_c_digest"],
                "observed": pinned["digests"]["arm_token_digests"]["C_scrambled"],
                "agrees": pinned["digests"]["arm_token_digests"]["C_scrambled"]
                == EXPECTED["arm_c_digest"],
            },
            "arm_c_is_non_automorphic": pinned["relational_laws"][
                "arm_c_every_habitat_rejected"
            ],
            "arm_c_relation_destroyed": pinned["relational_laws"][
                "arm_c_relation_destroyed"
            ],
            "arm_c_marginals_match_arm_b": pinned["relational_laws"][
                "arm_c_marginals_match_arm_b"
            ],
            "token_dim_per_presentation": token_dim,
            "pooling": "sum over the two canonical incidence presentations",
            "pooled_dim": token_dim,
        },
    }


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def environment_record() -> dict:
    import torch

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance only
            return "unavailable"

    return {
        "delivery_bundle": {
            "bundle": BUNDLE,
            "character": BUNDLES[BUNDLE]["character"],
            "probe_optimizer": BUNDLES[BUNDLE]["probe_optimizer"],
            "probe_optimizer_reading": BUNDLES[BUNDLE]["probe_optimizer_reading"],
        },
        "execution_identity": "Kiro (Claude Opus 5) acting in the generic Coder role",
        "task_uri": TASK_URI,
        "incorporated_specification_uri": SPEC_URI,
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "topographo": importlib.metadata.version("topographo"),
        "repository": {
            "root": str(REPO),
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(run(["git", "status", "--porcelain"])),
        },
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "thread_env": {
                var: os.environ.get(var)
                for var in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "device": "cpu",
            "dtype": "float32",
        },
        "declared_environment_differences": [
            (
                "the pinned 009.02 artifacts were produced under topographo 0.8.2; "
                "this execution runs topographo "
                f"{importlib.metadata.version('topographo')}. The certified relation "
                "is unchanged: fips_basic.TABLE_SHA256 still equals the pinned "
                f"{EXPECTED['table_sha256']}, and the catalogue / blocks / dataset / "
                "chart digests all reproduce their pinned values exactly."
            ),
        ],
        "pinned_inputs": {
            label: {
                "local_copy": f"pins/{filename}",
                "sha256": file_digest(PINS_DIR / filename),
                "bytes": (PINS_DIR / filename).stat().st_size,
            }
            for label, filename in PINNED_INPUTS.items()
        },
    }


def set_determinism() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized in this process
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# contracts stage
# ---------------------------------------------------------------------------

def architecture_record() -> dict:
    return {
        "input_event_embedding": [N_EVENTS, EMB_DIM],
        "candidate_event_embedding": [N_EVENTS, EMB_DIM],
        "pair_features": (
            "p = concat(e_a + e_b, e_a * e_b, abs(e_a - e_b)) with e from the "
            "input Event embedding table; 192 units"
        ),
        "trunk": "q = Linear(192,256) -> GELU -> Linear(256,64)",
        "scoring": (
            "score(c|a,b) = dot(q, u_c)/sqrt(64) + bias_c over all 84 candidate "
            "Events, with u from the candidate Event embedding table"
        ),
        "loss": "full 84-way cross entropy on the designated TRAIN target",
        "initialization": {
            "embedding_and_linear_weights": "Xavier uniform, gain 1.0",
            "biases": "zero",
            "candidate_bias": "zero",
        },
        "batching": "full batch, 48 designated TRAIN examples, no augmentation",
        "shared_tables": (
            "one input table and one candidate table shared across all examples; "
            "no per-pair or block-indexed parameter exists"
        ),
        "swap_invariance": (
            "exact by construction: every pair feature is symmetric in (a,b) and "
            "float addition/multiplication are commutative, so logits are "
            "bit-identical under input swap; verified before training and at "
            "every scored checkpoint"
        ),
        "memorization_capacity_note": (
            "per-Event embeddings and candidate biases are deliberately allowed; "
            "this is a supplied-vocabulary memorization-capable benchmark, not the "
            "information-restricted 009/011 discovery setting"
        ),
    }


def probe_contract_record() -> dict:
    record: dict = {
        "purpose": (
            "fixed direct representation readout: a fresh shared decoder trained "
            "only on TRAIN, applied to frozen main-model token representations"
        ),
        "representation": (
            "z_x = concat(input_embedding[x], candidate_embedding[x]); 128 "
            "dimensions; the main model's pair trunk and candidate bias are "
            "discarded for this diagnostic"
        ),
        "form": {
            "features": "f(a,b) = concat(z_a + z_b, z_a * z_b, abs(z_a - z_b))",
            "hidden": "h(a,b) = Linear(3*dim, 64) -> GELU",
            "candidate": "v(c) = Linear(dim, 64, bias=False)",
            "score": "score_probe(c|a,b) = dot(h(a,b), v(c))/sqrt(64)",
            "candidate_set": "all 84 Events",
        },
        "rules": {
            "initialization_seed": PROBE_SEED,
            "initialization": "Xavier uniform gain 1.0 on weights, zero biases",
            "optimizer": probe_optimizer_name(),
            "lr": PROBE_LR,
            "betas": list(PROBE_BETAS),
            "eps": PROBE_EPS,
            "weight_decay": PROBE_WEIGHT_DECAY,
            "steps": PROBE_STEPS,
            "batching": "full batch TRAIN only",
            "main_model_gradients": "forbidden; representations are detached",
            "identity_indexed_probe_parameters": "forbidden",
            "trained_on": "the 48 TRAIN examples only",
        },
        "width_adaptation": (
            "only the input projection width and the candidate projection input "
            "width follow the control-vector dimension; hidden width stays 64 and "
            "the semantic form is unchanged"
        ),
        "controls": {
            "B_certified_sfp": {
                "vector": (
                    "frozen 009 Arm B code, 16 one-hot units per incidence "
                    "presentation (S:2, FFF:7, PP:4, pp:3), summed over the two "
                    "presentations; 16 dimensions"
                ),
                "role": "positive control",
                "gate": (
                    f"must reach >= {PROBE_GATE} on BOTH probe ROLE_TEST and probe "
                    "NOVEL_TEST accuracy, otherwise the representation instrument "
                    "is BLOCKED and the main trajectory is not run"
                ),
            },
            "C_scrambled": {
                "vector": (
                    "frozen 009 Arm C non-automorphic scramble, identical shape and "
                    "matched field marginals, pooled identically; 16 dimensions"
                ),
                "role": "diagnostic only, never a selection criterion",
            },
            "random_gaussian": {
                "vector": (
                    f"seeded standard Gaussian token vectors, {RANDOM_CONTROL_DIM} "
                    f"dimensions, torch.manual_seed({RANDOM_CONTROL_SEED})"
                ),
                "role": "diagnostic only, never a selection criterion",
            },
            "secondary_declared_variant": (
                "the canonical-concatenation form of the Arm B and Arm C codes "
                "(32 dimensions) is additionally reported for transparency; the "
                "gate binds only to the summed primary declared above"
            ),
        },
        "reported_metrics": [
            "probe_train_accuracy",
            "probe_role_accuracy",
            "probe_novel_accuracy",
            "seen_block_three_role_consistency",
            "all_block_three_role_consistency",
            "full_56_block_relation_exact",
        ],
        "metric_definitions": {
            "seen_block_three_role_consistency": (
                "fraction of the 48 seen blocks whose three completions are all "
                "answered correctly"
            ),
            "all_block_three_role_consistency": (
                "the same fraction over all 56 certified blocks"
            ),
            "full_56_block_relation_exact": (
                "true only when all three completions of all 56 blocks are correct"
            ),
            "canonical_certificate": (
                "when the decoded relation is exact, an exact canonical "
                "hypergraph certificate is computed; no preferred SFP coordinate "
                "labels are required"
            ),
        },
    }
    if BUNDLE == "014.03a":
        record["optimizer_reading"] = BUNDLES[BUNDLE]["probe_optimizer_reading"]
        record["correction"] = PROBE_CORRECTION
    return record


PROBE_CORRECTION = {
    "character": (
        "superseding corrected execution of the same frozen experiment; the "
        "scientific contract is otherwise unchanged"
    ),
    "supersedes": "014.03-Coder-readable-algebraic-grokking-trajectory-result-GPT.md",
    "defect_found_by_014_03": (
        "014.02 section 4 freezes the probe optimizer as 'Adam' with 'weight decay "
        "1e-3'. In PyTorch that phrase has two implementations: a coupled L2 term "
        "inside torch.optim.Adam, and decoupled weight decay as in "
        "torch.optim.AdamW. 014.02 section 5 names 'AdamW' explicitly for the "
        "main-model regime grid, so 014.03 executed the literal coupled reading. "
        "Under that reading the certified-SFP positive control reaches only 0.9167 "
        "ROLE and 0.6667 NOVEL and cannot clear its own 0.95 gate, so 014.03 "
        "stopped at BLOCKED."
    ),
    "correction": (
        "resolve the ambiguous phrase to decoupled weight decay. The declared "
        "value 1e-3 is preserved. Under this reading the same probe reaches 0.9896 "
        "ROLE and 1.0000 NOVEL on the certified code and clears the gate."
    ),
    "unchanged_by_this_correction": [
        "benchmark construction",
        "held-out blocks and role assignment",
        "model architecture",
        "information boundary",
        "probe architecture, hidden width, seed, learning rate, betas, eps, steps",
        "the declared weight-decay value 1e-3",
        "positive-control gate threshold 0.95 on both ROLE and NOVEL",
        "candidate optimizer / weight-decay grid",
        "TRAIN-only selector",
        "scored seeds",
        "training horizon",
        "checkpoint schedules",
        "thresholds and temporal definitions",
        "component-transplant diagnostic",
        "interpretation fences",
    ],
    "isolation_evidence": (
        "014.03-Code-attachments/probe_instrument_diagnostic.json: probe "
        "initialization scheme, control pooling and step count were each ruled out; "
        "the scramble and random controls stay near chance under every reading, so "
        "the instrument's discrimination is unaffected and only the positive "
        "control's absolute level crosses the gate"
    ),
    "no_main_model_state_observed_before_authorization": True,
    "authorized_by": (
        "Executive (repository owner), after 014.03 delivered the BLOCKED verdict "
        "and its isolation diagnostic; recorded here rather than adjudicated by the "
        "Coder"
    ),
}


def regime_manifest_record() -> dict:
    return {
        "optimizer": "AdamW",
        "betas": list(SELECT_BETAS),
        "eps": SELECT_EPS,
        "schedule": "constant",
        "full_batch": True,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "calibration_cap": CALIBRATION_CAP,
        "learning_rate_grid": list(CANDIDATE_LRS),
        "weight_decay_grid": list(CANDIDATE_WDS),
        "candidates": len(CANDIDATE_LRS) * len(CANDIDATE_WDS),
        "selection_key": [
            "number of the four calibration seeds reaching exact TRAIN accuracy "
            "1.0 by update 4096, descending",
            "weight decay, descending",
            "median first exact-fit update among converged seeds, ascending",
            "learning rate, ascending",
        ],
        "requirement": "4/4 calibration seeds must reach exact TRAIN fit",
        "failure_state": "NO-REGIME (stop; do not widen the grid)",
        "matched_control": "selected configuration with weight_decay = 0.0",
        "boundary": (
            "selector code may not compute, load, log or inspect ROLE_TEST, "
            "NOVEL_TEST, representation-probe, SFP-factor or certified-equivalence "
            "metrics; audited mechanically"
        ),
    }


def temporal_definitions_record() -> dict:
    return {
        "t_fit": "first behavioral checkpoint with TRAIN accuracy == 1.0",
        "t_gen": (
            f"first behavioral checkpoint with ROLE_TEST accuracy >= {T_GEN_THRESHOLD} "
            "at that checkpoint and the next two scheduled behavioral checkpoints"
        ),
        "t_global": (
            f"first behavioral checkpoint with NOVEL_TEST accuracy >= {T_GEN_THRESHOLD} "
            "at that checkpoint and the next two scheduled behavioral checkpoints"
        ),
        "t_repr": (
            f"first probe checkpoint with probe ROLE_TEST accuracy >= {T_GEN_THRESHOLD} "
            "and seen_block_three_role_consistency >= "
            f"{T_REPR_CONSISTENCY_THRESHOLD} at that checkpoint and the next two "
            "probe checkpoints"
        ),
        "t_repr_global": (
            f"first probe checkpoint with probe NOVEL_TEST accuracy >= {T_GEN_THRESHOLD} "
            "and all_block_three_role_consistency >= "
            f"{T_REPR_CONSISTENCY_THRESHOLD} at that checkpoint and the next two "
            "probe checkpoints"
        ),
        "t_repr_exact": (
            "first probe checkpoint with full_56_block_relation_exact == true"
        ),
        "never_reached": "null, never the final checkpoint",
        "delayed_generalization_flag": [
            "t_fit exists",
            "t_gen exists",
            f"t_gen - t_fit >= {DELAYED_GAP} updates",
            f"ROLE_TEST accuracy at t_fit < {DELAYED_ROLE_AT_FIT}",
        ],
        "readings": {
            "M1": "t_fit < t_repr <= t_gen",
            "M2": "t_gen < t_repr, or t_repr absent despite generalization",
            "M3": (
                "t_repr exists but main ROLE_TEST generalization remains absent, or "
                "component tests fall below it"
            ),
            "M4": (
                "neither a stable readable relation nor held-out consequence "
                "generalization emerges"
            ),
        },
        "smoothing": (
            "threshold crossings are determined on unsmoothed values; smoothing is "
            "only ever a secondary visualization"
        ),
    }


def stage_contracts() -> None:
    set_determinism()
    started = time.time()
    relation = reconstruct()
    benchmark = build_benchmark(relation)
    base = baselines(relation, benchmark)
    arm_codes = arm_control_codes()

    benchmark_contract = {
        "module": "run_014.py (stage: contracts)",
        "task_uri": TASK_URI,
        "incorporated_specification_uri": SPEC_URI,
        "purpose": (
            "frozen pre-run contract for the 014 readable-algebraic-grokking "
            "trajectory: benchmark, model, regime manifest, schedules, thresholds"
        ),
        "pinned_inputs": {
            label: {
                "local_copy": f"pins/{filename}",
                "sha256": file_digest(PINS_DIR / filename),
            }
            for label, filename in PINNED_INPUTS.items()
        },
        "relation": relation.certification,
        "relation_reconstruction": {
            "blocks_sha256": digest([list(t) for t in relation.blocks]),
            "ordered_edges_sha256": digest([list(t) for t in relation.ordered_edges]),
            "unordered_admitted_pairs": len(relation.third_of_pair),
            "blocks": [list(t) for t in relation.blocks],
        },
        "holdout": benchmark.holdout,
        "role_assignment": {
            "rule": 'r = int(SHA256("014/role/v1|x0,x1,x2"), 16) mod 3 on the '
                    "canonical sorted Event-index triple; the designated training "
                    "target is x_r and the other two Events are the unordered "
                    "input pair",
            "seed_string": ROLE_SEED_STRING,
            "designated_role_by_block_position": {
                str(k): v for k, v in sorted(benchmark.designated_role.items())
            },
            "role_histogram": {
                str(r): sum(1 for v in benchmark.designated_role.values() if v == r)
                for r in (0, 1, 2)
            },
        },
        "split": {
            "sizes": {
                "TRAIN": len(benchmark.train),
                "ROLE_TEST": len(benchmark.role_test),
                "NOVEL_TEST": len(benchmark.novel_test),
            },
            "query_pairs_disjoint_across_sets": True,
            "one_unordered_training_example_per_seen_block": True,
            "TRAIN": [list(row) for row in benchmark.train],
            "ROLE_TEST": [list(row) for row in benchmark.role_test],
            "NOVEL_TEST": [list(row) for row in benchmark.novel_test],
            "digests": {
                "TRAIN": digest([list(r) for r in benchmark.train]),
                "ROLE_TEST": digest([list(r) for r in benchmark.role_test]),
                "NOVEL_TEST": digest([list(r) for r in benchmark.novel_test]),
            },
        },
        "information_boundary": {
            "main_learner_sees": (
                "84 opaque Event token ids and the 48 designated TRAIN examples "
                "only"
            ),
            "withheld_from_the_main_learner": [
                "S / FFF / PP / pp",
                "native ray coordinates or signs",
                "habitat or block IDs",
                "Fano numeric labels",
                "SFP code words",
                "ADMIT matrices or support graphs",
                "XOR / Fano completion tables",
                "any certified automorphism label",
            ],
            "certified_structure_used_only_for": [
                "frozen benchmark construction",
                "scoring already-fixed model states",
                "post-hoc representation diagnostics and probe controls",
            ],
            "tests_H_native": False,
        },
        "architecture": architecture_record(),
        "training_regime_manifest": regime_manifest_record(),
        "scored_seeds": list(SCORED_SEEDS),
        "training_horizon": HORIZON,
        "checkpoint_schedules": {
            "behavioral": list(BEHAVIORAL_CHECKPOINTS),
            "behavioral_count": len(BEHAVIORAL_CHECKPOINTS),
            "representation_probe": list(PROBE_CHECKPOINTS),
            "representation_probe_count": len(PROBE_CHECKPOINTS),
            "rule": (
                "no early stopping; runs are never terminated for generalizing or "
                "failing to generalize; seeds and horizon are fixed in advance"
            ),
        },
        "behavioral_metrics": [
            "TRAIN cross-entropy and accuracy",
            "ROLE_TEST accuracy",
            "NOVEL_TEST accuracy",
            "input-swap invariance",
            "parameter norms, embedding norms, effective ranks",
        ],
        "effective_rank_definition": (
            "entropy effective rank exp(-sum p_i log p_i) with p_i = s_i / sum(s) "
            "over the singular values of the matrix"
        ),
        "temporal_definitions": temporal_definitions_record(),
        "baselines": base,
        "arm_control_audit": arm_codes["audit"],
        "normalization_and_hash_rules": {
            "json": "indent=2, sort_keys=True, trailing newline",
            "digest": (
                "sha256 of json.dumps(payload, sort_keys=True, "
                "separators=(',',':')) encoded utf-8"
            ),
            "floats": (
                "metrics are rounded only for display; every stored trajectory "
                "value is the exact float repr produced by Python"
            ),
            "readback": "every artifact is re-read and re-digested after writing",
        },
        "fences": [
            "grokking does not universally mean algebra discovery",
            "no requirement that preferred coordinates appear neuron-by-neuron",
            "low dimensionality is not required",
            "decodability alone does not prove mechanistic use",
            "role generalization does not imply unseen-block algebra",
            "009 or 011 do not already constitute this experiment",
            "OT structure need not be the only representation solving the benchmark",
        ],
        "timing_note": (
            "this contract deliberately carries no wall-clock field, so its bytes "
            "and digest are reproducible; stage timing lives in "
            "contract_readback.json"
        ),
    }

    readbacks = {
        "benchmark_contract.json": write_json(
            HERE / "benchmark_contract.json", benchmark_contract
        ),
        "probe_contract.json": write_json(
            HERE / "probe_contract.json", probe_contract_record()
        ),
        "environment.json": write_json(HERE / "environment.json", environment_record()),
    }
    write_json(
        HERE / "contract_readback.json",
        {
            "delivery_bundle": BUNDLE,
            "readbacks": readbacks,
            "stage_elapsed_seconds": round(time.time() - started, 3),
        },
    )

    print("contracts committed and read back")
    for name, row in readbacks.items():
        print(f"  {name}: {row['bytes']} bytes sha256={row['sha256'][:16]}...")
    print(f"  novel blocks: {[list(relation.blocks[i]) for i in benchmark.novel_blocks]}")
    print(
        "  split sizes:",
        len(benchmark.train),
        len(benchmark.role_test),
        len(benchmark.novel_test),
    )


# ---------------------------------------------------------------------------
# the frozen main model
# ---------------------------------------------------------------------------

def xavier_uniform_(tensor, gain: float, generator) -> None:
    """Xavier uniform with an explicit generator, so init is seed-deterministic."""

    import math

    if tensor.dim() != 2:
        raise SystemExit("BLOCKED: Xavier init expects a 2-D parameter")
    fan_out, fan_in = tensor.shape
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    with __import__("torch").no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def build_main_model(seed: int):
    """The frozen memorization-capable architecture, initialized from ``seed``.

    Two shared 84 x 64 tables (input and candidate), a symmetric pair feature
    map, a 192-256-64 trunk, and a per-candidate additive scalar. No per-pair or
    index-keyed structural parameter exists.
    """

    import math

    import torch

    class MainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_emb = torch.nn.Parameter(torch.empty(N_EVENTS, EMB_DIM))
            self.cand_emb = torch.nn.Parameter(torch.empty(N_EVENTS, EMB_DIM))
            self.lin1 = torch.nn.Linear(PAIR_FEATURES, TRUNK_HIDDEN)
            self.lin2 = torch.nn.Linear(TRUNK_HIDDEN, EMB_DIM)
            self.cand_bias = torch.nn.Parameter(torch.zeros(N_EVENTS))
            self.scale = math.sqrt(EMB_DIM)

        def initialize(self, seed_value: int) -> None:
            generator = torch.Generator().manual_seed(seed_value)
            xavier_uniform_(self.input_emb, 1.0, generator)
            xavier_uniform_(self.cand_emb, 1.0, generator)
            xavier_uniform_(self.lin1.weight, 1.0, generator)
            xavier_uniform_(self.lin2.weight, 1.0, generator)
            with torch.no_grad():
                self.lin1.bias.zero_()
                self.lin2.bias.zero_()
                self.cand_bias.zero_()

        def forward(self, left, right):
            e_left = self.input_emb[left]
            e_right = self.input_emb[right]
            features = torch.cat(
                [
                    e_left + e_right,
                    e_left * e_right,
                    (e_left - e_right).abs(),
                ],
                dim=-1,
            )
            hidden = self.lin2(torch.nn.functional.gelu(self.lin1(features)))
            return hidden @ self.cand_emb.t() / self.scale + self.cand_bias

        def representation(self):
            """z_x = concat(input table row, candidate table row); detached."""

            return torch.cat([self.input_emb, self.cand_emb], dim=1).detach().clone()

    model = MainModel()
    model.initialize(seed)
    return model


def exact_fit_accuracy(model, left, right, target) -> float:
    """Fraction of designated answers ranked first. Used by calibration."""

    import torch

    with torch.no_grad():
        scores = model(left, right)
        return float((scores.argmax(dim=1) == target).float().mean())


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------

def as_tensors(rows: tuple[tuple[int, int, int], ...]):
    import torch

    left = torch.tensor([r[0] for r in rows], dtype=torch.long)
    right = torch.tensor([r[1] for r in rows], dtype=torch.long)
    target = torch.tensor([r[2] for r in rows], dtype=torch.long)
    return left, right, target


def load_contract() -> dict:
    path = HERE / "benchmark_contract.json"
    if not path.exists():
        raise SystemExit("BLOCKED: benchmark_contract.json is missing; run contracts")
    return json.loads(path.read_text())


def contract_split(contract: dict) -> dict:
    return {
        name: tuple(tuple(row) for row in contract["split"][name])
        for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")
    }


# ---------------------------------------------------------------------------
# the frozen representation probe
# ---------------------------------------------------------------------------

def build_probe(dim: int, seed: int = PROBE_SEED):
    """The fixed fresh decoder of 014.02 section 4. Never index-keyed."""

    import math

    import torch

    class RepresentationProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = torch.nn.Linear(PROBE_FEATURES_FACTOR * dim, PROBE_HIDDEN)
            self.candidate = torch.nn.Linear(dim, PROBE_HIDDEN, bias=False)
            self.scale = math.sqrt(PROBE_HIDDEN)

        def initialize(self, seed_value: int) -> None:
            generator = torch.Generator().manual_seed(seed_value)
            xavier_uniform_(self.hidden.weight, 1.0, generator)
            xavier_uniform_(self.candidate.weight, 1.0, generator)
            with torch.no_grad():
                self.hidden.bias.zero_()

        def forward(self, z, left, right):
            z_left = z[left]
            z_right = z[right]
            features = torch.cat(
                [
                    z_left + z_right,
                    z_left * z_right,
                    (z_left - z_right).abs(),
                ],
                dim=-1,
            )
            h = torch.nn.functional.gelu(self.hidden(features))
            return h @ self.candidate(z).t() / self.scale

    probe = RepresentationProbe()
    probe.initialize(seed)
    return probe


def fit_probe(z, splits: dict, blocks: list[list[int]], seen_positions: list[int]) -> dict:
    """Train the frozen probe on TRAIN only and report the declared metrics."""

    import torch

    z = z.detach().clone()
    if z.requires_grad:
        raise SystemExit("BLOCKED: probe representation is not detached")
    dim = z.shape[1]
    probe = build_probe(dim)
    train_left, train_right, train_target = as_tensors(splits["TRAIN"])
    factory = (
        torch.optim.Adam if probe_optimizer_name() == "Adam" else torch.optim.AdamW
    )
    optimizer = factory(
        probe.parameters(),
        lr=PROBE_LR,
        betas=PROBE_BETAS,
        eps=PROBE_EPS,
        weight_decay=PROBE_WEIGHT_DECAY,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = float("nan")
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(z, train_left, train_right)
        loss = loss_fn(logits, train_target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    if z.grad is not None:
        raise SystemExit("BLOCKED: probe training leaked gradient into the representation")

    with torch.no_grad():
        predictions: dict[tuple[int, int], int] = {}
        accuracies = {}
        for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST"):
            left, right, target = as_tensors(splits[name])
            argmax = probe(z, left, right).argmax(dim=1)
            accuracies[name] = float((argmax == target).float().mean())
            for i, row in enumerate(splits[name]):
                predictions[(row[0], row[1])] = int(argmax[i])

    correct_pairs = {
        (row[0], row[1])
        for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")
        for row in splits[name]
        if predictions[(row[0], row[1])] == row[2]
    }

    def all_three(position: int) -> bool:
        triple = blocks[position]
        return all(
            (min(x, y), max(x, y)) in correct_pairs
            for x, y in itertools.combinations(triple, 2)
        )

    seen_consistency = sum(all_three(p) for p in seen_positions) / len(seen_positions)
    all_consistency = sum(all_three(p) for p in range(len(blocks))) / len(blocks)
    exact = all_consistency == 1.0

    decoded = sorted(
        {
            tuple(sorted((pair[0], pair[1], answer)))
            for pair, answer in predictions.items()
        }
    )
    row = {
        "probe_train_accuracy": accuracies["TRAIN"],
        "probe_role_accuracy": accuracies["ROLE_TEST"],
        "probe_novel_accuracy": accuracies["NOVEL_TEST"],
        "seen_block_three_role_consistency": seen_consistency,
        "all_block_three_role_consistency": all_consistency,
        "full_56_block_relation_exact": exact,
        "probe_final_train_loss": final_loss,
        "decoded_distinct_triples": len(decoded),
        "representation_dim": dim,
    }
    if exact:
        row["canonical_certificate"] = canonical_certificate(
            tuple(tuple(t) for t in decoded)
        )
    return row


# ---------------------------------------------------------------------------
# probe controls stage
# ---------------------------------------------------------------------------

def control_vectors(arm_codes: dict) -> dict:
    """The three declared controls plus the declared secondary concat variant."""

    import torch

    pinned = json.loads((PINS_DIR / "arms.json").read_text())

    def concat_variant(name: str) -> list[list[float]]:
        import arms as arms_mod

        layout = [
            (field, offset, width) for field, offset, width, _ in arms_mod.FIELD_LAYOUT
        ]

        def parse(code: str) -> list[float]:
            vector = [0.0] * arms_mod.SFP_TOKEN_DIM
            for field in code.split(","):
                key, value = field.split("=")
                _, offset, width = next(r for r in layout if r[0] == key)
                index = int(value, 2)
                slot = index - 1 if key in ("FFF", "pp") else index
                vector[offset + slot] = 1.0
            return vector

        rows = []
        for entry in pinned["arms"][name]["codes"]:
            left, right = entry.split("|")
            rows.append(parse(left) + parse(right))
        return rows

    generator = torch.Generator().manual_seed(RANDOM_CONTROL_SEED)
    random_control = torch.randn(
        N_EVENTS, RANDOM_CONTROL_DIM, generator=generator, dtype=torch.float32
    )

    return {
        "B_certified_sfp": torch.tensor(arm_codes["B_sfp"], dtype=torch.float32),
        "C_scrambled": torch.tensor(arm_codes["C_scrambled"], dtype=torch.float32),
        "random_gaussian": random_control,
        "B_certified_sfp_concat_secondary": torch.tensor(
            concat_variant("B_sfp"), dtype=torch.float32
        ),
        "C_scrambled_concat_secondary": torch.tensor(
            concat_variant("C_scrambled"), dtype=torch.float32
        ),
    }


def stage_probe_controls() -> None:
    set_determinism()
    contract = load_contract()
    splits = contract_split(contract)
    blocks = contract["relation_reconstruction"]["blocks"]
    seen_positions = contract["holdout"]["seen_block_positions"]

    arm_codes = arm_control_codes()
    controls = control_vectors(arm_codes)

    results = {}
    for name, z in controls.items():
        started = time.time()
        row = fit_probe(z, splits, blocks, seen_positions)
        row["elapsed_seconds"] = round(time.time() - started, 3)
        results[name] = row
        print(
            f"  {name:38s} dim={row['representation_dim']:4d} "
            f"train={row['probe_train_accuracy']:.4f} "
            f"role={row['probe_role_accuracy']:.4f} "
            f"novel={row['probe_novel_accuracy']:.4f} "
            f"exact={row['full_56_block_relation_exact']}"
        )

    positive = results["B_certified_sfp"]
    gate_pass = (
        positive["probe_role_accuracy"] >= PROBE_GATE
        and positive["probe_novel_accuracy"] >= PROBE_GATE
    )

    payload = {
        "module": "run_014.py (stage: probe-controls)",
        "task_uri": TASK_URI,
        "probe_contract_sha256": file_digest(HERE / "probe_contract.json"),
        "benchmark_contract_sha256": file_digest(HERE / "benchmark_contract.json"),
        "arm_control_audit": arm_codes["audit"],
        "controls": results,
        "gate": {
            "rule": (
                "the certified-SFP positive control must reach at least "
                f"{PROBE_GATE} on BOTH probe ROLE_TEST and probe NOVEL_TEST"
            ),
            "threshold": PROBE_GATE,
            "observed_role": positive["probe_role_accuracy"],
            "observed_novel": positive["probe_novel_accuracy"],
            "passed": gate_pass,
            "status": "PASS" if gate_pass else "BLOCKED",
            "no_stronger_probe_invented": (
                "the probe form, width, optimizer, steps and seed are exactly as "
                "frozen in probe_contract.json before any main-model checkpoint "
                "existed"
            ),
        },
        "diagnostic_note": (
            "the scramble and random controls are diagnostic, never selection "
            "criteria; they are reported regardless of outcome"
        ),
    }
    write_json(HERE / "probe_controls.json", payload)
    print(f"probe controls committed; gate={payload['gate']['status']}")
    if not gate_pass:
        raise SystemExit(
            "BLOCKED: certified-SFP positive control did not clear the declared "
            "gate; the expensive main trajectory is not run"
        )


# ---------------------------------------------------------------------------
# post-gate instrument diagnostic
#
# Runs only AFTER the frozen gate verdict is committed, and changes nothing
# about it. Its sole purpose is to isolate WHY the frozen instrument reached the
# verdict it reached, so the Owner can adjudicate. It is not a selection
# criterion and no result of it is used to alter probe_contract.json.
# ---------------------------------------------------------------------------

DIAGNOSTIC_READINGS = (
    ("coupled_Adam_wd_1e-3", "Adam", 1e-3),
    ("decoupled_AdamW_wd_1e-3", "AdamW", 1e-3),
    ("no_weight_decay", "Adam", 0.0),
)
DIAGNOSTIC_REPORT_STEPS = (1000, 2000, 5000)


def diagnostic_fit(z, splits: dict, optimizer_name: str, decay: float) -> dict:
    import torch

    probe = build_probe(z.shape[1])
    factory = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.AdamW
    optimizer = factory(
        probe.parameters(),
        lr=PROBE_LR,
        betas=PROBE_BETAS,
        eps=PROBE_EPS,
        weight_decay=decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    train_left, train_right, train_target = as_tensors(splits["TRAIN"])
    rows = {}
    for step in range(1, PROBE_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(probe(z, train_left, train_right), train_target)
        loss.backward()
        optimizer.step()
        if step in DIAGNOSTIC_REPORT_STEPS:
            with torch.no_grad():
                accuracies = {}
                for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST"):
                    left, right, target = as_tensors(splits[name])
                    accuracies[name] = float(
                        (probe(z, left, right).argmax(dim=1) == target).float().mean()
                    )
            rows[str(step)] = accuracies
    return rows


def stage_probe_diagnostics() -> None:
    set_determinism()
    contract = load_contract()
    splits = contract_split(contract)
    gate = json.loads((HERE / "probe_controls.json").read_text())["gate"]

    arm_codes = arm_control_codes()
    controls = control_vectors(arm_codes)
    matrix: dict = {}
    for label, z in controls.items():
        matrix[label] = {"dim": int(z.shape[1]), "readings": {}}
        for reading, optimizer_name, decay in DIAGNOSTIC_READINGS:
            matrix[label]["readings"][reading] = diagnostic_fit(
                z, splits, optimizer_name, decay
            )
            final = matrix[label]["readings"][reading][str(PROBE_STEPS)]
            print(
                f"  {label:38s} {reading:24s} "
                f"role={final['ROLE_TEST']:.4f} novel={final['NOVEL_TEST']:.4f}"
            )

    primary = matrix["B_certified_sfp"]["readings"]
    payload = {
        "module": "run_014.py (stage: probe-diagnostics)",
        "status": "DIAGNOSTIC ONLY — does not alter the committed gate verdict",
        "committed_gate_verdict": gate,
        "question": (
            "the frozen instrument failed its own certified-SFP positive control; "
            "this diagnostic isolates which frozen or implementation-resolved "
            "degree of freedom is responsible"
        ),
        "ambiguity_under_test": {
            "specification_text": (
                "014.02 section 4 freezes the probe optimizer as 'Adam' with "
                "'weight decay 1e-3'. In PyTorch that phrase has two "
                "implementations: torch.optim.Adam(weight_decay=1e-3), which adds a "
                "coupled L2 term to the gradient, and decoupled weight decay as in "
                "torch.optim.AdamW(weight_decay=1e-3). 014.02 names 'AdamW' "
                "explicitly for the main-model regime grid in section 5, so the "
                "literal reading of 'Adam' in section 4 is the coupled form, which "
                "is what probe_contract.json declared and what the gate ran."
            ),
            "resolved_in_code_as": "torch.optim.Adam(weight_decay=1e-3) (coupled L2)",
        },
        "matrix": matrix,
        "isolation": {
            "probe_initialization_is_not_the_cause": (
                "the declared Xavier-uniform/zero-bias init and the PyTorch default "
                "Linear init give the same failing levels under the coupled reading"
            ),
            "control_pooling_is_not_the_cause": (
                "both the declared summed 16-dimensional primary and the declared "
                "32-dimensional canonical-concat secondary fail the gate under the "
                "coupled reading"
            ),
            "step_count_is_not_the_cause": (
                "under the coupled reading the positive control is flat in NOVEL "
                "accuracy from 500 through 12000 steps"
            ),
            "decisive_knob": "coupled versus decoupled weight decay",
            "coupled_positive_control": primary["coupled_Adam_wd_1e-3"][
                str(PROBE_STEPS)
            ],
            "decoupled_positive_control": primary["decoupled_AdamW_wd_1e-3"][
                str(PROBE_STEPS)
            ],
            "zero_decay_positive_control": primary["no_weight_decay"][str(PROBE_STEPS)],
        },
        "instrument_discrimination_survives_both_readings": (
            "the non-automorphic scramble control stays near 0.16-0.20 ROLE and "
            "0.08-0.13 NOVEL and the random control stays at 0.0 under every "
            "reading, so the instrument separates the certified code from matched "
            "misalignment either way; only the absolute level of the positive "
            "control moves across the 0.95 gate"
        ),
        "what_this_does_not_do": [
            "it does not modify probe_contract.json",
            "it does not modify or reopen the committed gate verdict",
            "it does not authorize the main trajectory",
            "no main-model checkpoint has been produced or observed at this point",
        ],
        "owner_decision_required": (
            "whether to re-authorize execution under the decoupled reading of the "
            "frozen probe optimizer, or to accept BLOCKED as the delivered result"
        ),
    }
    write_json(HERE / "probe_instrument_diagnostic.json", payload)
    print("probe instrument diagnostic committed")


# ---------------------------------------------------------------------------
# TRAIN-only regime selection
#
# Everything between this banner and the audit helper below constitutes the
# selector. It receives only the designated-answer training arrays and the
# frozen candidate grid. It never computes, loads, logs or inspects any
# held-out or diagnostic quantity; audit_selector_boundary() proves that
# mechanically over the selector call graph.
# ---------------------------------------------------------------------------

SELECTION_ACTIVE = False


class SealedHoldout:
    """Wrapper that refuses to yield its payload while selection is running."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.touched_during_selection = False

    def unseal(self) -> object:
        if SELECTION_ACTIVE:
            self.touched_during_selection = True
            raise RuntimeError(
                "BLOCKED: selector touched sealed held-out material"
            )
        return self._payload


def calibrate_candidate(
    left, right, target, learning_rate: float, decay: float, seed: int, cap: int
) -> dict:
    """Train one candidate configuration from one seed up to the frozen cap."""

    import torch

    model = build_main_model(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=SELECT_BETAS,
        eps=SELECT_EPS,
        weight_decay=decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    first_exact = None
    losses: list[float] = []
    for update in range(1, cap + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model(left, right)
        loss = loss_fn(scores, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        # exactness is measured on the post-update state, every single update,
        # so the recorded update index needs no granularity caveat
        if exact_fit_accuracy(model, left, right, target) == 1.0:
            first_exact = update
            break
    return {
        "seed": seed,
        "first_exact_fit_update": first_exact,
        "reached_exact_fit": first_exact is not None,
        "final_designated_loss": losses[-1] if losses else float("nan"),
    }


def select_regime_core(left, right, target) -> dict:
    """Rank the frozen candidate grid on designated-answer fit only."""

    import statistics

    candidates = []
    for learning_rate in CANDIDATE_LRS:
        for decay in CANDIDATE_WDS:
            seeds = [
                calibrate_candidate(
                    left, right, target, learning_rate, decay, seed, CALIBRATION_CAP
                )
                for seed in CALIBRATION_SEEDS
            ]
            converged = [s["first_exact_fit_update"] for s in seeds if s["reached_exact_fit"]]
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": decay,
                    "seeds": seeds,
                    "converged_seeds": len(converged),
                    "median_first_exact_fit_update": (
                        statistics.median(converged) if converged else None
                    ),
                }
            )
            print(
                f"  lr={learning_rate:<7} wd={decay:<5} "
                f"converged={len(converged)}/4 "
                f"median_fit={candidates[-1]['median_first_exact_fit_update']}"
            )

    def key(candidate: dict):
        return (
            -candidate["converged_seeds"],
            -candidate["weight_decay"],
            candidate["median_first_exact_fit_update"]
            if candidate["median_first_exact_fit_update"] is not None
            else float("inf"),
            candidate["learning_rate"],
        )

    ranked = sorted(candidates, key=key)
    return {"candidates": candidates, "ranked": ranked, "winner": ranked[0]}


FORBIDDEN_IN_SELECTOR = (
    "role_test",
    "novel_test",
    "role_accuracy",
    "novel_accuracy",
    "probe",
    "sfp",
    "certif",
    "equival",
    "oracle",
    "third_of_pair",
    "block_of_pair",
    "relation",
    "arm_control",
    "holdout",
)

SELECTOR_ROOTS = (
    "select_regime_core",
    "calibrate_candidate",
    "build_main_model",
    "exact_fit_accuracy",
    "xavier_uniform_",
)


def audit_selector_boundary() -> dict:
    """Mechanically audit the selector call graph for held-out metric contact."""

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    reachable: set[str] = set()
    queue = list(SELECTOR_ROOTS)
    while queue:
        name = queue.pop()
        if name in reachable or name not in functions:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                queue.append(node.func.id)

    findings: list[dict] = []
    identifiers: set[str] = set()
    for name in sorted(reachable):
        for node in ast.walk(functions[name]):
            tokens: list[str] = []
            if isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            elif isinstance(node, ast.arg):
                tokens.append(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                tokens.append(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                tokens.append(node.value)
            for token in tokens:
                identifiers.add(token)
                lowered = token.lower()
                for needle in FORBIDDEN_IN_SELECTOR:
                    if needle in lowered:
                        findings.append(
                            {"function": name, "token": token, "forbidden": needle}
                        )

    return {
        "method": (
            "AST call-graph walk from the selector roots over module-level "
            "functions; every Name, Attribute, argument, keyword and string "
            "constant in the reachable set is matched against the forbidden list"
        ),
        "selector_roots": list(SELECTOR_ROOTS),
        "reachable_functions": sorted(reachable),
        "forbidden_substrings": list(FORBIDDEN_IN_SELECTOR),
        "identifiers_scanned": len(identifiers),
        "violations": findings,
        "clean": not findings,
    }


def stage_select_regime() -> None:
    global SELECTION_ACTIVE

    set_determinism()
    contract = load_contract()
    splits = contract_split(contract)

    audit = audit_selector_boundary()
    if not audit["clean"]:
        raise SystemExit(f"BLOCKED: selector boundary violation: {canonical(audit)}")

    sealed = SealedHoldout(
        {name: splits[name] for name in ("ROLE_TEST", "NOVEL_TEST")}
    )
    left, right, target = as_tensors(splits["TRAIN"])

    started = time.time()
    SELECTION_ACTIVE = True
    try:
        outcome = select_regime_core(left, right, target)
    finally:
        SELECTION_ACTIVE = False

    winner = outcome["winner"]
    qualified = winner["converged_seeds"] == len(CALIBRATION_SEEDS)

    payload = {
        "module": "run_014.py (stage: select-regime)",
        "task_uri": TASK_URI,
        "benchmark_contract_sha256": file_digest(HERE / "benchmark_contract.json"),
        "manifest": regime_manifest_record(),
        "boundary_audit": audit,
        "sealed_holdout_touched_during_selection": sealed.touched_during_selection,
        "calibration": outcome["candidates"],
        "ranking": [
            {
                "learning_rate": candidate["learning_rate"],
                "weight_decay": candidate["weight_decay"],
                "converged_seeds": candidate["converged_seeds"],
                "median_first_exact_fit_update": candidate[
                    "median_first_exact_fit_update"
                ],
            }
            for candidate in outcome["ranked"]
        ],
        "requirement_4_of_4_met": qualified,
        "status": "SELECTED" if qualified else "NO-REGIME",
        "selected_regime": {
            "optimizer": "AdamW",
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "betas": list(SELECT_BETAS),
            "eps": SELECT_EPS,
            "schedule": "constant",
            "full_batch": True,
        },
        "matched_memorization_control": {
            "optimizer": "AdamW",
            "learning_rate": winner["learning_rate"],
            "weight_decay": 0.0,
            "betas": list(SELECT_BETAS),
            "eps": SELECT_EPS,
            "schedule": "constant",
            "full_batch": True,
            "difference_from_selected": "weight_decay only",
        },
        "scored_seeds": list(SCORED_SEEDS),
        "training_horizon": HORIZON,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(HERE / "selected_regime.json", payload)
    print(f"selected_regime committed: {payload['status']} {payload['selected_regime']}")
    if not qualified:
        raise SystemExit(
            "BLOCKED/NO-REGIME: no candidate reached 4/4 exact designated fit; "
            "the grid was not widened"
        )


# ---------------------------------------------------------------------------
# scored trajectories
# ---------------------------------------------------------------------------

def parameter_digest(model) -> str:
    """Digest of every parameter tensor, keyed by name in sorted order."""

    state = model.state_dict()
    blob = b"".join(
        name.encode("utf-8") + state[name].detach().contiguous().numpy().tobytes()
        for name in sorted(state)
    )
    return hashlib.sha256(blob).hexdigest()


def effective_rank(matrix) -> float:
    import torch

    with torch.no_grad():
        singular = torch.linalg.svdvals(matrix.detach().float())
        total = float(singular.sum())
        if total <= 0.0:
            return 0.0
        probabilities = singular / total
        entropy = float(
            -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
        )
        return float(__import__("math").exp(entropy))


def behavioral_metrics(model, tensors: dict) -> dict:
    import torch

    loss_fn = torch.nn.CrossEntropyLoss()
    row: dict = {}
    with torch.no_grad():
        left, right, target = tensors["TRAIN"]
        scores = model(left, right)
        row["train_cross_entropy"] = float(loss_fn(scores, target))
        row["train_accuracy"] = float((scores.argmax(dim=1) == target).float().mean())
        swapped = model(right, left)
        row["input_swap_invariance_bit_exact"] = bool(torch.equal(scores, swapped))
        row["input_swap_max_abs_difference"] = float((scores - swapped).abs().max())
        for name in ("ROLE_TEST", "NOVEL_TEST"):
            left, right, target = tensors[name]
            argmax = model(left, right).argmax(dim=1)
            key = "role_test_accuracy" if name == "ROLE_TEST" else "novel_test_accuracy"
            row[key] = float((argmax == target).float().mean())
        row["parameter_norm"] = float(
            torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters()))
        )
        row["input_embedding_norm"] = float(model.input_emb.detach().norm())
        row["candidate_embedding_norm"] = float(model.cand_emb.detach().norm())
        row["candidate_bias_norm"] = float(model.cand_bias.detach().norm())
        row["trunk_norm"] = float(
            torch.sqrt(
                (model.lin1.weight.detach() ** 2).sum()
                + (model.lin2.weight.detach() ** 2).sum()
            )
        )
        row["input_embedding_effective_rank"] = effective_rank(model.input_emb)
        row["candidate_embedding_effective_rank"] = effective_rank(model.cand_emb)
        row["trunk_lin1_effective_rank"] = effective_rank(model.lin1.weight)
        row["trunk_lin2_effective_rank"] = effective_rank(model.lin2.weight)
    return row


def run_trajectory(
    seed: int,
    arm: str,
    horizon: int,
    probe_steps_note: str,
    capture_updates: tuple[int, ...] = (),
    skip_probe: bool = False,
) -> dict:
    """One scored (seed, arm) run at the frozen horizon and schedules."""

    import torch

    set_determinism()
    contract = load_contract()
    regime = json.loads((HERE / "selected_regime.json").read_text())
    if regime["status"] != "SELECTED":
        raise SystemExit("BLOCKED: no selected regime is available")
    config = (
        regime["selected_regime"]
        if arm == "regularized"
        else regime["matched_memorization_control"]
    )

    splits = contract_split(contract)
    blocks = contract["relation_reconstruction"]["blocks"]
    seen_positions = contract["holdout"]["seen_block_positions"]
    tensors = {name: as_tensors(rows) for name, rows in splits.items()}

    model = build_main_model(seed)
    initial_digest = parameter_digest(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    behavioral_points = set(BEHAVIORAL_CHECKPOINTS)
    probe_points = set(PROBE_CHECKPOINTS)
    capture_points = set(capture_updates)
    behavioral_rows: list[dict] = []
    probe_rows: list[dict] = []
    captures: dict[int, dict] = {}
    started = time.time()

    def observe(update: int) -> None:
        if update in behavioral_points:
            row = behavioral_metrics(model, tensors)
            row["update"] = update
            behavioral_rows.append(row)
        if update in probe_points and not skip_probe:
            row = fit_probe(model.representation(), splits, blocks, seen_positions)
            row["update"] = update
            probe_rows.append(row)
        if update in capture_points:
            captures[update] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    observe(0)
    train_left, train_right, train_target = tensors["TRAIN"]
    for update in range(1, horizon + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model(train_left, train_right)
        loss = loss_fn(scores, train_target)
        loss.backward()
        optimizer.step()
        observe(update)

    elapsed = time.time() - started
    return {
        "seed": seed,
        "arm": arm,
        "config": config,
        "horizon": horizon,
        "initial_parameter_digest": initial_digest,
        "behavioral": behavioral_rows,
        "probe": probe_rows,
        "captures": captures,
        "probe_steps": PROBE_STEPS,
        "probe_note": probe_steps_note,
        "elapsed_seconds": round(elapsed, 3),
        "early_stopping": False,
        "final_parameter_digest": parameter_digest(model),
    }


def stage_trajectory(seed: int, arm: str) -> None:
    result = run_trajectory(seed, arm, HORIZON, "frozen 5000-step probe fits")
    result.pop("captures")
    path = CHECKPOINTS / f"trajectory_{arm}_seed{seed}.json"
    write_json(path, result)
    final = result["behavioral"][-1]
    print(
        f"seed={seed} arm={arm} elapsed={result['elapsed_seconds']}s "
        f"final train={final['train_accuracy']:.4f} "
        f"role={final['role_test_accuracy']:.4f} "
        f"novel={final['novel_test_accuracy']:.4f}"
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def stage_aggregate() -> None:
    """Collect the per-run trajectory files into the declared attachments."""

    regularized: list[dict] = []
    control: list[dict] = []
    probe_rows: list[dict] = []
    for seed in SCORED_SEEDS:
        for arm, sink in (("regularized", regularized), ("weight_decay_zero", control)):
            path = CHECKPOINTS / f"trajectory_{arm}_seed{seed}.json"
            if not path.exists():
                raise SystemExit(f"BLOCKED: missing scored run {path.name}")
            run = json.loads(path.read_text())
            probe_rows.append(
                {
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "checkpoints": run["probe"],
                }
            )
            sink.append({k: v for k, v in run.items() if k != "probe"})

    shared = {
        "task_uri": TASK_URI,
        "delivery_bundle": BUNDLE,
        "benchmark_contract_sha256": file_digest(HERE / "benchmark_contract.json"),
        "selected_regime_sha256": file_digest(HERE / "selected_regime.json"),
        "scored_seeds": list(SCORED_SEEDS),
        "horizon": HORIZON,
        "behavioral_checkpoints": list(BEHAVIORAL_CHECKPOINTS),
        "no_early_stopping": True,
        "no_reruns_after_seeing_results": True,
    }
    write_json(
        HERE / "trajectory_regularized.json",
        {**shared, "arm": "regularized", "runs": regularized},
    )
    write_json(
        HERE / "trajectory_weight_decay_zero.json",
        {**shared, "arm": "weight_decay_zero", "runs": control},
    )
    write_json(
        HERE / "representation_probe_trajectory.json",
        {
            **shared,
            "probe_contract_sha256": file_digest(HERE / "probe_contract.json"),
            "probe_checkpoints": list(PROBE_CHECKPOINTS),
            "runs": probe_rows,
        },
    )

    # durable per-run record, under the governing long-running-computation
    # discipline: checksums, schema, provenance and exact resume instructions
    write_json(
        CHECKPOINTS / "manifest.json",
        {
            "purpose": (
                "raw per-(seed, arm) trajectory records, one file per scored run. "
                "Every value here also appears in the three aggregate attachments; "
                "these files are the unaggregated evidence."
            ),
            "delivery_bundle": BUNDLE,
            "benchmark_contract_sha256": file_digest(HERE / "benchmark_contract.json"),
            "selected_regime_sha256": file_digest(HERE / "selected_regime.json"),
            "probe_contract_sha256": file_digest(HERE / "probe_contract.json"),
            "schema": {
                "seed": "int, one of the eight scored seeds",
                "arm": "regularized | weight_decay_zero",
                "config": "the committed optimizer configuration actually used",
                "initial_parameter_digest": "sha256 over named parameter bytes at update 0",
                "final_parameter_digest": "the same digest at the final update",
                "behavioral": "one row per behavioral checkpoint",
                "probe": "one row per representation-probe checkpoint",
            },
            "files": {
                path.name: {
                    "sha256": file_digest(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(CHECKPOINTS.glob("trajectory_*.json"))
            },
            "resume_instructions": (
                "each run is independent and deterministic. To regenerate one file: "
                "python run_014.py --bundle "
                f"{BUNDLE} trajectory --seed <seed> --arm <arm>. To regenerate the "
                "aggregates from existing files: python run_014.py --bundle "
                f"{BUNDLE} aggregate. Determinism requires the single-thread CPU "
                "settings recorded in environment.json."
            ),
            "self_audit": (
                "run_014.py aggregate refuses to proceed if any scored run file is "
                "missing, and analyze_014.py recomputes every temporal quantity from "
                "these bytes rather than from any cached summary"
            ),
        },
    )
    print("aggregated trajectories committed")


def stage_transplant() -> None:
    """The predeclared four-way component transplant. No post-hoc redesign."""

    import torch

    analysis_path = HERE / "analysis.json"
    if not analysis_path.exists():
        raise SystemExit(
            "BLOCKED: run analyze_014.py first; the transplant consumes the "
            "predeclared temporal diagnostics"
        )
    temporal = json.loads(analysis_path.read_text())["temporal_diagnostics"]
    contract = load_contract()
    splits = contract_split(contract)
    tensors = {name: as_tensors(rows) for name, rows in splits.items()}

    plans = []
    for row in temporal:
        if row["arm"] != "regularized":
            continue
        t_fit = row["t_fit"]
        t_gen = row["t_gen"]
        if t_fit is None:
            plans.append({"seed": row["seed"], "skipped": "no t_fit", "pair": None})
            continue
        if t_gen is not None and t_gen != t_fit:
            plans.append(
                {
                    "seed": row["seed"],
                    "pair": [t_fit, t_gen],
                    "basis": "t_fit vs t_gen (distinct transition times)",
                }
            )
        else:
            plans.append(
                {
                    "seed": row["seed"],
                    "pair": [t_fit, HORIZON],
                    "basis": (
                        "no distinct delayed transition existed for this seed, so the "
                        "declared fallback t_fit vs the final checkpoint is used"
                    ),
                }
            )

    results = []
    for plan in plans:
        if plan["pair"] is None:
            results.append(plan)
            continue
        early_update, late_update = plan["pair"]
        run = run_trajectory(
            plan["seed"],
            "regularized",
            late_update,
            "transplant capture replay; probe fits skipped because they never "
            "touch global RNG or the main parameters, so the replayed main "
            "trajectory is bit-identical to the scored run",
            capture_updates=(early_update, late_update),
            skip_probe=True,
        )
        captures = run["captures"]
        if early_update not in captures or late_update not in captures:
            raise SystemExit("BLOCKED: transplant replay did not capture both states")

        replayed = {row["update"]: row for row in run["behavioral"]}
        scored_path = (
            CHECKPOINTS / f"trajectory_regularized_seed{plan['seed']}.json"
        )
        scored = {
            row["update"]: row
            for row in json.loads(scored_path.read_text())["behavioral"]
        }
        compared = [u for u in sorted(replayed) if u in scored]
        mismatches = [
            u
            for u in compared
            if any(
                replayed[u][field] != scored[u][field]
                for field in (
                    "train_cross_entropy",
                    "train_accuracy",
                    "role_test_accuracy",
                    "novel_test_accuracy",
                    "parameter_norm",
                    "input_embedding_norm",
                    "candidate_embedding_norm",
                )
            )
        ]
        replay_check = {
            "checkpoints_compared": len(compared),
            "bit_identical_to_scored_run": not mismatches,
            "mismatched_updates": mismatches,
            "captured_updates": [early_update, late_update],
            "train_accuracy_at_captures": {
                str(u): replayed[u]["train_accuracy"]
                for u in (early_update, late_update)
                if u in replayed
            },
        }
        if mismatches:
            raise SystemExit(
                "BLOCKED: transplant replay diverged from the scored run at "
                f"updates {mismatches}"
            )

        model = build_main_model(plan["seed"])
        representation_keys = ("input_emb", "cand_emb")
        downstream_keys = (
            "lin1.weight",
            "lin1.bias",
            "lin2.weight",
            "lin2.bias",
            "cand_bias",
        )

        def evaluate(repr_source: str, down_source: str) -> dict:
            state = {}
            for key in representation_keys:
                state[key] = captures[
                    early_update if repr_source == "early" else late_update
                ][key].clone()
            for key in downstream_keys:
                state[key] = captures[
                    early_update if down_source == "early" else late_update
                ][key].clone()
            model.load_state_dict(state)
            with torch.no_grad():
                row = {}
                for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST"):
                    left, right, target = tensors[name]
                    row[name] = float(
                        (model(left, right).argmax(dim=1) == target).float().mean()
                    )
            return row

        results.append(
            {
                **plan,
                "early_update": early_update,
                "late_update": late_update,
                "replay_check": replay_check,
                "hybrids": {
                    "early_representation_early_downstream": evaluate("early", "early"),
                    "late_representation_late_downstream": evaluate("late", "late"),
                    "late_representation_early_downstream": evaluate("late", "early"),
                    "early_representation_late_downstream": evaluate("early", "late"),
                },
            }
        )
        print(f"  seed={plan['seed']} transplant {early_update} -> {late_update} done")

    payload = {
        "module": "run_014.py (stage: transplant)",
        "task_uri": TASK_URI,
        "delivery_bundle": BUNDLE,
        "partition": {
            "REPRESENTATION": "input Event embedding table + candidate Event embedding table",
            "DOWNSTREAM": "pair MLP + candidate bias",
        },
        "method": (
            "deterministic replay of the scored regularized run to the declared "
            "update counts, then four evaluations without any retraining"
        ),
        "predeclared": True,
        "interpretation_fence": (
            "component-level evidence about where the functional change resides, "
            "not a claim of a uniquely identified causal circuit; coadaptation can "
            "make both cross-hybrids fail and that is a result, not permission to "
            "invent another intervention"
        ),
        "runs": results,
    }
    write_json(HERE / "component_transplants.json", payload)
    print("component transplants committed")


def stage_negative_diagnostic() -> None:
    """Post-hoc explanation of a clean negative, on already-fixed model states.

    This changes no frozen choice and produces no new scored quantity. It asks a
    single mechanistic question: given that every seed memorized TRAIN and never
    generalized, what does the fixed final model actually do on held-out queries,
    and what does the frozen output contract do to the answer vocabulary?
    """

    import torch

    contract = load_contract()
    splits = contract_split(contract)
    tensors = {name: as_tensors(rows) for name, rows in splits.items()}

    designated_targets = {row[2] for row in splits["TRAIN"]}
    never_designated = [x for x in range(N_EVENTS) if x not in designated_targets]

    def answer_profile(name: str) -> dict:
        rows = splits[name]
        return {
            "examples": len(rows),
            "answers_that_are_designated_targets_somewhere": sum(
                1 for row in rows if row[2] in designated_targets
            ),
            "answers_never_designated_anywhere": sum(
                1 for row in rows if row[2] not in designated_targets
            ),
        }

    static = {
        "distinct_designated_target_events": len(designated_targets),
        "events_never_a_designated_target": len(never_designated),
        "answer_vocabulary_profile": {
            name: answer_profile(name) for name in ("TRAIN", "ROLE_TEST", "NOVEL_TEST")
        },
        "why_this_matters": (
            "full 84-way cross entropy on one designated answer per seen block "
            "pushes every non-target logit down. Events that are never a designated "
            "target therefore receive only downward pressure across the whole run, "
            "and a large share of the held-out answers are exactly such events."
        ),
    }

    per_seed = []
    for seed in SCORED_SEEDS:
        run = run_trajectory(
            seed,
            "regularized",
            HORIZON,
            "negative-result diagnostic replay",
            capture_updates=(HORIZON,),
            skip_probe=True,
        )
        state = run["captures"][HORIZON]
        model = build_main_model(seed)
        model.load_state_dict(state)
        with torch.no_grad():
            bias = model.cand_bias.detach()
            designated_mask = torch.zeros(N_EVENTS, dtype=torch.bool)
            for x in designated_targets:
                designated_mask[x] = True
            row = {
                "seed": seed,
                "candidate_bias_mean_designated": float(bias[designated_mask].mean()),
                "candidate_bias_mean_never_designated": float(
                    bias[~designated_mask].mean()
                ),
                "predictions": {},
            }
            for name in ("ROLE_TEST", "NOVEL_TEST"):
                left, right, target = tensors[name]
                predicted = model(left, right).argmax(dim=1)
                categories = {
                    "correct": 0,
                    "one_of_the_two_query_inputs": 0,
                    "a_designated_target_elsewhere": 0,
                    "never_designated_event": 0,
                }
                for i, example in enumerate(splits[name]):
                    guess = int(predicted[i])
                    if guess == example[2]:
                        categories["correct"] += 1
                    elif guess in (example[0], example[1]):
                        categories["one_of_the_two_query_inputs"] += 1
                    elif guess in designated_targets:
                        categories["a_designated_target_elsewhere"] += 1
                    else:
                        categories["never_designated_event"] += 1
                row["predictions"][name] = {
                    "counts": categories,
                    "distinct_predicted_events": len(set(predicted.tolist())),
                    "share_landing_on_a_designated_target": (
                        categories["a_designated_target_elsewhere"]
                        + categories["correct"]
                    )
                    / len(splits[name]),
                }
        per_seed.append(row)
        print(f"  seed={seed} diagnostic done")

    payload = {
        "module": "run_014.py (stage: negative-diagnostic)",
        "status": (
            "POST-HOC EXPLANATORY DIAGNOSTIC on already-fixed model states; it "
            "changes no frozen choice, adds no scored quantity, and does not "
            "reinterpret any temporal reading"
        ),
        "delivery_bundle": BUNDLE,
        "question": (
            "every scored seed memorized TRAIN and never generalized; what does the "
            "fixed final model do on held-out queries, and what does the frozen "
            "output contract do to the held-out answer vocabulary?"
        ),
        "benchmark_static_facts": static,
        "per_seed": per_seed,
        "architectural_reading": {
            "untied_tables": (
                "the frozen architecture gives each Event two independent "
                "parameter rows: an input-table row used only when the Event "
                "appears in a query pair, and a candidate-table row used only when "
                "the Event is scored as an answer. Learning 'z is the answer to "
                "{x,y}' updates z's candidate row and leaves z's input row "
                "untouched, so a memorized answer carries no information about the "
                "same Event as a future query input."
            ),
            "contrast_with_the_probe": (
                "the representation probe scores candidates with the SAME vector it "
                "uses to build the query features, so structure learned in one role "
                "is automatically available in the other. On the certified code the "
                "probe reaches 0.9896 ROLE and 1.0000 NOVEL from the identical 48 "
                "TRAIN examples and the identical 84-way objective. The benchmark is "
                "therefore solvable under this output contract; what the main "
                "architecture lacks is not capacity but any pressure to make one "
                "Event's two rows describe the same object."
            ),
            "consequence_for_the_result": (
                "the clean M4 negative is consistent with the main learner never "
                "forming a role-transferable Event representation at all, rather "
                "than forming one that the probe failed to read. Both readings are "
                "reported; this diagnostic does not settle which, and no claim "
                "beyond the finite tested benchmark is made."
            ),
        },
        "fence": (
            "this diagnostic is explanatory only. It does not license changing the "
            "architecture, the benchmark, or any threshold, and it does not convert "
            "the delivered negative into a positive result."
        ),
    }
    write_json(HERE / "negative_result_diagnostic.json", payload)
    print("negative-result diagnostic committed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=tuple(BUNDLES), default="014.03")
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("contracts")
    sub.add_parser("probe-controls")
    sub.add_parser("probe-diagnostics")
    sub.add_parser("select-regime")
    trajectory = sub.add_parser("trajectory")
    trajectory.add_argument("--seed", type=int, required=True)
    trajectory.add_argument(
        "--arm", choices=("regularized", "weight_decay_zero"), required=True
    )
    sub.add_parser("aggregate")
    sub.add_parser("transplant")
    sub.add_parser("negative-diagnostic")
    args = parser.parse_args(argv)
    configure_bundle(args.bundle)
    if args.stage == "contracts":
        stage_contracts()
    elif args.stage == "probe-controls":
        stage_probe_controls()
    elif args.stage == "probe-diagnostics":
        stage_probe_diagnostics()
    elif args.stage == "select-regime":
        stage_select_regime()
    elif args.stage == "trajectory":
        stage_trajectory(args.seed, args.arm)
    elif args.stage == "aggregate":
        stage_aggregate()
    elif args.stage == "transplant":
        stage_transplant()
    elif args.stage == "negative-diagnostic":
        stage_negative_diagnostic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

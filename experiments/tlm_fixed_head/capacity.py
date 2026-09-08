"""Alphabet capacity + collision structure for the 15 Fixed effect classes.

Answers 006.13 experimentally for modular arithmetic (no torch required):
how much task-relevant output the verified two-step Fixed head can express,
and whether the 15-class quotient is injective / useful for (a,b)->c.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from topographo.ssd import fixed_head


@dataclass(frozen=True)
class CapacityReport:
    p: int
    n_words: int
    n_classes: int
    n_modular_pairs: int
    n_modular_outputs: int
    # Map modular symbols into Fixed labels / classes.
    word_label_bits: float
    class_bits: float
    # Collision structure for role (a,b)->c using class features of a synthetic
    # embedding of residues into Event indices (mod 84).
    injective_pair_to_class: bool
    max_class_collision_for_pair: int
    mean_class_collision_for_pair: float
    # Conditional mutual information proxy: H(c) - H(c | class(a,b)).
    mi_c_given_class_bits: float
    mi_c_given_word_bits: float
    fatal_for_p: bool
    notes: tuple[str, ...]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        ent -= p * math.log2(p)
    return ent


def _embed_residue(r: int, p: int) -> int:
    """Deterministic residue -> Event index in 0..83 (capacity probe convention)."""

    return int(r % p) % fixed_head.N_EVENTS


def analyze_capacity(p: int, census: fixed_head.CensusResult | None = None) -> CapacityReport:
    census = census or fixed_head.two_step_census()
    table = census.class_id_table
    n_pairs = p * p
    # For each modular pair (a,b), look up Fixed class of embedded word (b_emb, a_emb).
    pair_to_class: dict[tuple[int, int], int] = {}
    class_to_outputs: dict[int, set[int]] = defaultdict(set)
    class_pair_counts: Counter[int] = Counter()
    joint_class_c: Counter[tuple[int, int]] = Counter()
    joint_word_c: Counter[tuple[int, int]] = Counter()
    out_counts: Counter[int] = Counter()
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            ea, eb = _embed_residue(a, p), _embed_residue(b, p)
            cid = int(table[eb, ea])
            pair_to_class[(a, b)] = cid
            class_to_outputs[cid].add(c)
            class_pair_counts[cid] += 1
            joint_class_c[(cid, c)] += 1
            joint_word_c[(ea + 84 * eb, c)] += 1
            out_counts[c] += 1

    collisions = [len(v) for v in class_to_outputs.values()]
    # H(c), H(c|class), H(c|word)
    h_c = _entropy(out_counts)
    # H(c|class) = sum_class p(class) H(c|class)
    h_c_given_class = 0.0
    total = n_pairs
    for cid, n in class_pair_counts.items():
        sub = Counter({c: joint_class_c[(cid, c)] for c in range(p) if joint_class_c[(cid, c)]})
        h_c_given_class += (n / total) * _entropy(sub)
    h_c_given_word = 0.0
    word_counts: Counter[int] = Counter()
    for (w, c), n in joint_word_c.items():
        word_counts[w] += n
    for w, n in word_counts.items():
        sub = Counter({c: joint_word_c[(w, c)] for c in range(p) if joint_word_c[(w, c)]})
        h_c_given_word += (n / total) * _entropy(sub)

    mi_class = h_c - h_c_given_class
    mi_word = h_c - h_c_given_word
    # Fatal if 15 classes cannot separate p outputs on average (info-theoretic).
    fatal = fixed_head.EXPECTED_N_CLASSES < p and mi_class < 0.5 * h_c
    notes = (
        "Embedding residues into Event indices is a configured probe map, not a "
        "physical Event-space realization.",
        "injective_pair_to_class means each modular pair lands in a unique Fixed "
        "class under the probe embedding (almost never true for large p).",
        "fatal_for_p marks an information bottleneck: 15 classes vs p outputs with "
        "weak class→c mutual information under this embedding.",
    )
    return CapacityReport(
        p=p,
        n_words=fixed_head.N_WORDS,
        n_classes=fixed_head.EXPECTED_N_CLASSES,
        n_modular_pairs=n_pairs,
        n_modular_outputs=p,
        word_label_bits=math.log2(fixed_head.N_WORDS),
        class_bits=math.log2(fixed_head.EXPECTED_N_CLASSES),
        injective_pair_to_class=len(set(pair_to_class.values())) == n_pairs,
        max_class_collision_for_pair=max(collisions) if collisions else 0,
        mean_class_collision_for_pair=float(sum(collisions) / len(collisions)) if collisions else 0.0,
        mi_c_given_class_bits=float(mi_class),
        mi_c_given_word_bits=float(mi_word),
        fatal_for_p=bool(fatal),
        notes=notes,
    )


def write_report(path: Path, reports: list[CapacityReport]) -> None:
    path.write_text(json.dumps([asdict(r) for r in reports], indent=2, sort_keys=True) + "\n")

"""Frozen `008.04` learning task at ``E^3 R`` with all three constructors.

`008.03` section 3 fixes the cell: signature ``E^3 R``, constructors
``Occ + Cyc + Sand``, retained context ``r = e4``, true/frozen denotations,
exact native endpoint scoring, learning restricted to program selection. This
module is the **torch-free** half of that experiment: it materializes the exact
program-space records, applies the `008.02` sign-bit target grammar, constructs
the held-out splits, runs the `008.03` section 2.3 program-space viability gate,
and scores every deterministic baseline.

Nothing here trains. Nothing here is approximate. A learned policy consumes
``Pool.features`` and returns program indices; every number it earns is scored
by the same exact machinery the baselines are scored by.

Target relation
---------------
Unchanged from `008.02`, as `008.04` section 4 instructs. Each basic Event is
``e_i + s e_{8+j}``; the grammar reads the sign bits ``s`` of the exposed Events
and selects a program by index:

```text
bits  = 4*b(s1) + 2*b(s2) + b(s3),   b(s) = 1 if s > 0 else 0
index = bits mod 20
```

Because ``bits < 8 < 20`` the reachable pool is exactly the eight ``Cyc``-free
bracketings, and in the enumeration order of ``programs.py`` the selected
program is the right comb whose retained head at depth ``d`` is ``Sand`` iff bit
``d`` of ``bits`` is set. Bit ``0`` is the sign of the **last** Event and it
governs the **outermost** head, so the grammar is a reversal, not a per-slot
copy. It is a rigid relation over Event structure: not availability, not
derivable from which branches happen to be legal, and not a function of any
model's scores.

Held-out structure
------------------
Three splits, all constructed before training, all recorded exactly.

``motif`` (primary, compositional)
    Withholds the two sign-bit patterns ``011`` and ``100``, and with them the
    two target bracketing motifs ``Sand(a1,Sand(a2,Occ(a3,r)))`` and
    ``Occ(a1,Occ(a2,Sand(a3,r)))``. Six patterns remain in training, so every
    constituent ``(retained depth, head constructor)`` choice still occurs — and
    so does both parities — but neither withheld combination ever does. This is
    the `008.04` section 6 "constituent motifs occur in training, the
    combination is withheld" split. Note what it costs a flat classifier: two of
    the twenty program labels never appear as a training target, so a model that
    can only reproduce seen labels scores zero on the held-out motifs by
    construction.

    The withheld patterns are chosen so that **no** deterministic executor is
    privileged on the held-out half. `008.04` section 4 forbids a target that
    picks ForceSeq or GroupedFirst by construction, and withholding the
    homogeneous patterns ``000`` / ``111`` would have done exactly that: the
    ``000`` target is the pure right comb, which is ForceSeq's fixed answer. It
    was measured at 0.50 on that held-out half before the split was corrected.
    Neither program 3 nor program 4 is the right comb or a most-grouped term.

``motif_parity`` (harder compositional diagnostic)
    Withholds all four odd-parity patterns, so half the target motifs are
    unseen and the model must cross a parity boundary. Every ``(depth, bit)``
    constituent still occurs in training, but the surviving patterns
    ``{000, 011, 101, 110}`` form a linear subspace on which "bit d" and
    "bit d xor overall parity" are indistinguishable. Reported as a diagnostic,
    not as the primary claim, precisely because the hypothesis is only
    identifiable under a compositional inductive bias.

``random`` (robustness)
    A plain i.i.d. split over Event triples. Held-out **combinations** only, no
    held-out motif. It is included because a deterministic sign-pattern lookup
    table fitted on train solves it exactly, which is what makes it a control
    rather than a claim: no learned policy can materially beat a deterministic
    baseline that already scores 1.0.

``unseen_events`` (robustness)
    Withholds 21 of the 84 Events entirely; test triples contain at least one
    Event never seen in training. Tests generalization over denotations rather
    than over bracketings. The lookup table also saturates this split, since
    sign bits are readable off an unseen Event, so it functions as a check that
    the policy reads Event structure instead of memorizing Event identities.

Fence
-----
Strict constructors only. Ambient generalized multiplication lives in
``ambient.py`` and is fenced there as non-OT-native. No physical Event supply
claim, no language-model claim, no modular arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from experiments.tlm_multitoken import native
from experiments.tlm_multitoken.probe_program_space import (
    Evaluator,
    cyc_free_indices,
    cyc_nodes,
    force_seq_index,
    grouped_first_order,
    retained_ray,
    sign_bit,
)
from experiments.tlm_multitoken.programs import (
    CONSTRUCTORS,
    Cyc,
    EventLeaf,
    Occ,
    RetainedLeaf,
    Sand,
    Term,
    enumerate_programs,
    event_leaf_indices,
    render,
)
from topographo.ssd import fips_basic, projective

__all__ = [
    "BASELINES",
    "ONE_SIDED_PINS",
    "PINS_00802",
    "PIN_TOLERANCE",
    "BaselineScore",
    "Pool",
    "Record",
    "Selector",
    "Split",
    "TaskConfig",
    "ViabilityGate",
    "best_deterministic",
    "build_splits",
    "cyc_free_program_names",
    "legal_program_counts",
    "materialize_pool",
    "program_structure_features",
    "restrict_split",
    "score_baselines",
    "score_selector",
    "sign_pattern",
    "split_digest",
    "target_index",
    "viability_gate",
]

#: `008.02` / `008.03` program-space pins for ``E^3 R`` with all three
#: constructors, over all 84 Events at ``r = e4``. `008.04` section 3 says to
#: stop and explain if the implementation path moves them materially.
PINS_00802: dict[str, float | int] = {
    "n_programs": 20,
    "joint_domain_share": 0.9854,
    "endpoint_disagreement_given_two": 1.0,
    "n_availability_patterns": 36,
    "availability_ceiling": 0.1445,
    "learned_headroom": 0.8555,
    "max_distinct_endpoints_in_a_pattern": 12,
    "target_defined_share": 0.9546,
}

#: Sampling tolerance for the share-valued pins. The pool is a uniform sample of
#: 592704 inputs, so a 0.02 band is generous for shares near 1 and near 0.14.
PIN_TOLERANCE = 0.02

#: Two pins are checked one-sided, because their sample estimator is biased in a
#: known direction and pretending otherwise would mean widening a tolerance to
#: hide a bias.
#:
#: The availability ceiling is a sum of per-pattern *maxima* over noisy hit
#: counts, so on a finite sample it over-estimates the exhaustive value, and the
#: bias shrinks as the pool grows (about +0.021 at 1200 inputs, +0.014 at 3000).
#: Over-estimating is conservative for the `008.04` claim: it raises the bar a
#: learned policy has to clear. What would invalidate the cell is the ceiling
#: coming out materially *below* the pin, which would mean the sample had lost
#: the availability structure the pin describes. So the ceiling is checked from
#: below and the headroom, being ``1 - ceiling``, from above.
ONE_SIDED_PINS: dict[str, str] = {
    "availability_ceiling": "at_least",
    "learned_headroom": "at_most",
}


@dataclass(frozen=True)
class TaskConfig:
    """Everything frozen before training. Serialized verbatim into the report."""

    n_events: int = 3
    retained_label: str = "e4"
    constructors: frozenset[str] = CONSTRUCTORS
    #: Uniform sample of the 84^3 = 592704 exposed inputs.
    pool_size: int = 40000
    pool_seed: int = 8_04_2026
    #: Sign-bit patterns withheld by the primary compositional split. Chosen so
    #: that neither held-out target is the right comb (ForceSeq) or a
    #: most-grouped term (GroupedFirst); see the module docstring.
    heldout_patterns: tuple[int, ...] = (0b011, 0b100)
    #: Harder compositional diagnostic: withhold every odd-parity pattern.
    parity_heldout_patterns: tuple[int, ...] = (0b001, 0b010, 0b100, 0b111)
    #: i.i.d. robustness split.
    random_test_fraction: float = 0.25
    random_split_seed: int = 804_001
    #: Event-identity robustness split.
    n_heldout_events: int = 21
    event_split_seed: int = 804_002

    def as_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "signature": f"E^{self.n_events} R",
            "retained_label": self.retained_label,
            "constructors": "+".join(sorted(self.constructors)),
            "pool_size": self.pool_size,
            "pool_seed": self.pool_seed,
            "heldout_patterns": [f"{p:03b}" for p in self.heldout_patterns],
            "parity_heldout_patterns": [
                f"{p:03b}" for p in self.parity_heldout_patterns
            ],
            "random_test_fraction": self.random_test_fraction,
            "random_split_seed": self.random_split_seed,
            "n_heldout_events": self.n_heldout_events,
            "event_split_seed": self.event_split_seed,
        }


# --- the target relation ---------------------------------------------------


def target_index(events: Sequence[native.Ray], n_programs: int) -> int:
    """`008.02` sign-bit latent bracketing grammar. Unchanged by `008.04`."""
    bits = 0
    for event in events:
        bits = (bits << 1) | sign_bit(event)
    return bits % n_programs


def sign_pattern(events: Sequence[native.Ray]) -> int:
    """The raw sign-bit word, before the modulus. Used to define the splits."""
    bits = 0
    for event in events:
        bits = (bits << 1) | sign_bit(event)
    return bits


# --- program structure, available without any target knowledge -------------


def _retained_chain(program: Term) -> list[Term]:
    """The retained spine of a program, outermost head first."""
    chain: list[Term] = []
    node = program
    while isinstance(node, (Occ, Sand)):
        chain.append(node)
        node = node.retained
    return chain


def _leaf_groups(program: Term) -> list[tuple[int, ...]]:
    """Event-leaf index groups, one per retained head, outermost head first."""
    return [tuple(event_leaf_indices(node.event)) for node in _retained_chain(program)]


def program_structure_features(program: Term, n_events: int) -> list[float]:
    """A target-agnostic structural encoding of a strict program tree.

    Every feature is a property of the term calculus itself — which constructor
    heads which retained depth, which Event leaves a head consumes, how much
    ``Cyc`` grouping the term carries. No feature mentions the target relation,
    the sign bits, or any model output. The point of the encoding is that the
    twenty programs are described by *shared reusable parts*, so a program whose
    exact index never appears as a training label is still scorable from parts
    that do.

    Layout, for ``n_events = 3``:

    ```text
    [0:3]   head at retained depth d is Sand      (d = 0,1,2; 0 if no depth d)
    [3:6]   a head exists at retained depth d
    [6:9]   Event leaf k is consumed by a Sand head
    [9:12]  Event leaf k is consumed by the head at retained depth k
    [12:15] number of Cyc nodes, one-hot over 0,1,2
    [15]    leaves 0 and 1 are immediate siblings of one Cyc node
    [16]    leaves 1 and 2 are immediate siblings of one Cyc node
    [17]    some head consumes all Event leaves
    [18]    a Cyc node has a Cyc left child   (left-nested grouping)
    [19]    a Cyc node has a Cyc right child  (right-nested grouping)
    [20]    retained-chain length, normalized
    ```

    Features 18 and 19 are what separate ``Cyc(Cyc(a1,a2),a3)`` from
    ``Cyc(a1,Cyc(a2,a3))``; without them the encoding is not injective and the
    structural head cannot tell the two fully grouped terms apart.
    """
    chain = _retained_chain(program)
    groups = _leaf_groups(program)
    is_sand = [isinstance(node, Sand) for node in chain]

    head_is_sand = [0.0] * n_events
    head_exists = [0.0] * n_events
    for depth in range(min(len(chain), n_events)):
        head_exists[depth] = 1.0
        head_is_sand[depth] = 1.0 if is_sand[depth] else 0.0

    leaf_under_sand = [0.0] * n_events
    leaf_at_own_depth = [0.0] * n_events
    for depth, group in enumerate(groups):
        for leaf in group:
            if leaf < n_events:
                leaf_under_sand[leaf] = 1.0 if is_sand[depth] else 0.0
                leaf_at_own_depth[leaf] = 1.0 if leaf == depth else 0.0

    n_cyc = cyc_nodes(program)
    cyc_onehot = [1.0 if n_cyc == k else 0.0 for k in range(n_events)]

    sibling_pairs = [0.0] * (n_events - 1)
    left_nested = 0.0
    right_nested = 0.0
    for node in _cyc_nodes_of(program):
        left = event_leaf_indices(node.left)
        right = event_leaf_indices(node.right)
        if (
            len(left) == 1
            and len(right) == 1
            and right[0] - left[0] == 1
            and left[0] < n_events - 1
        ):
            sibling_pairs[left[0]] = 1.0
        if isinstance(node.left, Cyc):
            left_nested = 1.0
        if isinstance(node.right, Cyc):
            right_nested = 1.0
    all_grouped = 1.0 if any(len(group) == n_events for group in groups) else 0.0

    return [
        *head_is_sand,
        *head_exists,
        *leaf_under_sand,
        *leaf_at_own_depth,
        *cyc_onehot,
        *sibling_pairs,
        all_grouped,
        left_nested,
        right_nested,
        len(chain) / n_events,
    ]


def _cyc_nodes_of(term: Term) -> list[Cyc]:
    """Every ``Cyc`` node in a term, in no particular order."""
    match term:
        case EventLeaf() | RetainedLeaf():
            return []
        case Cyc(left=left, right=right):
            return [term, *_cyc_nodes_of(left), *_cyc_nodes_of(right)]
        case Occ(event=event, retained=retained) | Sand(event=event, retained=retained):
            return _cyc_nodes_of(event) + _cyc_nodes_of(retained)
    raise TypeError(f"not a term: {term!r}")


# --- exact records ---------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One exposed input, fully evaluated on the exact path.

    ``endpoint_class`` holds a pool-global integer identity per program, with
    ``-1`` for undefined, so projective endpoint equality is an integer compare
    after materialization and never a re-derived approximation.
    """

    event_indices: tuple[int, ...]
    sign_pattern: int
    target_program: int
    target_class: int
    defined: tuple[bool, ...]
    endpoint_class: tuple[int, ...]
    availability_pattern: int

    @property
    def target_defined(self) -> bool:
        return self.target_class >= 0

    @property
    def legal(self) -> tuple[int, ...]:
        return tuple(k for k, ok in enumerate(self.defined) if ok)

    def matches(self, program: int | None) -> bool:
        """Exact projective endpoint equality against the target endpoint."""
        if program is None or not self.defined[program]:
            return False
        return self.endpoint_class[program] == self.target_class

    def n_distinct_endpoints(self) -> int:
        return len({c for c in self.endpoint_class if c >= 0})


@dataclass
class Pool:
    """The materialized exact task: records plus the model-visible features."""

    config: TaskConfig
    programs: tuple[Term, ...]
    program_names: tuple[str, ...]
    records: tuple[Record, ...]
    #: Per record, the exact Event denotations flattened to float coordinates.
    #: This is the Ladder-A "true / frozen denotations" input and nothing else.
    features: tuple[tuple[float, ...], ...]
    n_endpoint_classes: int
    evaluator_calls: int

    @property
    def scored(self) -> tuple[int, ...]:
        """Record indices with a defined target, the primary scoring set."""
        return tuple(i for i, r in enumerate(self.records) if r.target_defined)

    def feature_dim(self) -> int:
        return len(self.features[0]) if self.features else 0


def _flatten(events: Sequence[native.Ray], retained: native.Ray) -> tuple[float, ...]:
    out: list[float] = []
    for value in (*events, retained):
        out.extend(float(coefficient) for coefficient in value)
    return tuple(out)


def materialize_pool(config: TaskConfig = TaskConfig(), *, events=None) -> Pool:
    """Sample exposed inputs and evaluate every program exactly."""
    all_events = tuple(fips_basic.EVENTS) if events is None else tuple(events)
    retained = retained_ray()
    programs = enumerate_programs(config.n_events, config.constructors)
    if not programs:
        raise ValueError("constructor set yields no programs")

    n_total = len(all_events) ** config.n_events
    rng = random.Random(config.pool_seed)
    if config.pool_size >= n_total:
        chosen = list(range(n_total))
    else:
        chosen = rng.sample(range(n_total), config.pool_size)
    chosen.sort()

    evaluate = Evaluator(retained)
    class_of: dict[tuple, int] = {}
    records: list[Record] = []
    features: list[tuple[float, ...]] = []

    for flat in chosen:
        indices: list[int] = []
        rest = flat
        for _ in range(config.n_events):
            indices.append(rest % len(all_events))
            rest //= len(all_events)
        indices.reverse()
        combo = tuple(all_events[k] for k in indices)

        endpoints = [evaluate(program, combo) for program in programs]
        classes: list[int] = []
        for value in endpoints:
            if value is None:
                classes.append(-1)
                continue
            key = tuple(str(c) for c in projective.canonicalize(value))
            classes.append(class_of.setdefault(key, len(class_of)))

        defined = tuple(value is not None for value in endpoints)
        pattern = 0
        for k, ok in enumerate(defined):
            if ok:
                pattern |= 1 << k
        chosen_program = target_index(combo, len(programs))
        records.append(
            Record(
                event_indices=tuple(indices),
                sign_pattern=sign_pattern(combo),
                target_program=chosen_program,
                target_class=classes[chosen_program],
                defined=defined,
                endpoint_class=tuple(classes),
                availability_pattern=pattern,
            )
        )
        features.append(_flatten(combo, retained))

    return Pool(
        config=config,
        programs=programs,
        program_names=tuple(render(program) for program in programs),
        records=tuple(records),
        features=tuple(features),
        n_endpoint_classes=len(class_of),
        evaluator_calls=evaluate.calls,
    )


# --- splits ----------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A frozen train/test partition of pool record indices."""

    name: str
    kind: str
    rationale: str
    train: tuple[int, ...]
    test: tuple[int, ...]
    metadata: dict = field(default_factory=dict)

    def check_disjoint(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"split {self.name!r} leaks {len(overlap)} records")


def split_digest(pool: Pool, split: Split) -> str:
    """Reproducible membership digest over Event-index triples, not row order."""
    payload = {
        "name": split.name,
        "train": sorted(pool.records[i].event_indices for i in split.train),
        "test": sorted(pool.records[i].event_indices for i in split.test),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_splits(pool: Pool) -> tuple[Split, ...]:
    """Construct all three splits. Called before any training, by contract."""
    config = pool.config
    scored = pool.scored

    # --- primary: held-out sign-bit patterns, hence held-out target motifs ---
    motif = _pattern_holdout_split(
        pool,
        name="motif",
        heldout=config.heldout_patterns,
        rationale=(
            "withholds the sign-bit patterns 011 and 100 and with them two "
            "mixed target bracketings; every constituent (retained depth, head "
            "constructor) choice and both parities still occur in training, and "
            "neither held-out target is the right comb or a most-grouped term"
        ),
    )
    parity = _pattern_holdout_split(
        pool,
        name="motif_parity",
        heldout=config.parity_heldout_patterns,
        rationale=(
            "withholds every odd-parity sign pattern; constituents survive but "
            "the training patterns form a linear subspace on which 'bit d' and "
            "'bit d xor parity' agree, so the hypothesis is identifiable only "
            "under a compositional inductive bias"
        ),
    )

    # --- robustness: i.i.d. over Event triples ------------------------------
    order = list(scored)
    random.Random(config.random_split_seed).shuffle(order)
    n_test = int(len(order) * config.random_test_fraction)
    rnd = Split(
        name="random",
        kind="iid_heldout_combination",
        rationale=(
            "held-out Event combinations only, no held-out motif; the split a "
            "sign-pattern lookup table can solve"
        ),
        train=tuple(sorted(order[n_test:])),
        test=tuple(sorted(order[:n_test])),
        metadata={"test_fraction": config.random_test_fraction},
    )

    # --- robustness: held-out Event identities ------------------------------
    n_all_events = len(fips_basic.EVENTS)
    event_order = list(range(n_all_events))
    random.Random(config.event_split_seed).shuffle(event_order)
    heldout_events = frozenset(event_order[: config.n_heldout_events])
    unseen_train = tuple(
        i
        for i in scored
        if not (set(pool.records[i].event_indices) & heldout_events)
    )
    unseen_test = tuple(
        i for i in scored if set(pool.records[i].event_indices) & heldout_events
    )
    heldout_signs = Counter(
        sign_bit(fips_basic.EVENTS[k]) for k in sorted(heldout_events)
    )
    unseen = Split(
        name="unseen_events",
        kind="heldout_event_identity",
        rationale=(
            "21 of the 84 Events never appear in training; every test triple "
            "contains at least one unseen Event"
        ),
        train=unseen_train,
        test=unseen_test,
        metadata={
            "heldout_event_indices": sorted(heldout_events),
            "heldout_event_sign_bits": {str(k): v for k, v in sorted(heldout_signs.items())},
        },
    )

    splits = (motif, parity, rnd, unseen)
    for split in splits:
        split.check_disjoint()
    return splits


def restrict_split(split: Split, keep: Sequence[int], *, suffix: str) -> Split:
    """Restrict a split to a subset of record indices, keeping its identity.

    Used by the ambient control: the ambient semantics leaves a slightly
    different set of inputs with a defined target, and comparing arms on
    different record sets would not be a comparison. Restricting rather than
    rebuilding keeps the held-out membership identical wherever both are defined.
    """
    allowed = set(keep)
    return Split(
        name=f"{split.name}{suffix}",
        kind=split.kind,
        rationale=split.rationale,
        train=tuple(i for i in split.train if i in allowed),
        test=tuple(i for i in split.test if i in allowed),
        metadata={
            **split.metadata,
            "restricted_from": split.name,
            "dropped_train": len([i for i in split.train if i not in allowed]),
            "dropped_test": len([i for i in split.test if i not in allowed]),
        },
    )


def _head_constituents(pool: Pool, programs: set[int]) -> set[tuple[int, bool]]:
    """The ``(retained depth, head is Sand)`` parts a set of programs is built from."""
    out: set[tuple[int, bool]] = set()
    for program in sorted(programs):
        for depth, node in enumerate(_retained_chain(pool.programs[program])):
            out.add((depth, isinstance(node, Sand)))
    return out


def _pattern_holdout_split(
    pool: Pool, *, name: str, heldout: Sequence[int], rationale: str
) -> Split:
    """Withhold whole sign-bit patterns, and with them whole target motifs."""
    withheld = set(heldout)
    scored = pool.scored
    train = tuple(i for i in scored if pool.records[i].sign_pattern not in withheld)
    test = tuple(i for i in scored if pool.records[i].sign_pattern in withheld)
    train_motifs = {pool.records[i].target_program for i in train}
    test_motifs = {pool.records[i].target_program for i in test}
    constituents_train = _head_constituents(pool, train_motifs)
    constituents_test = _head_constituents(pool, test_motifs)
    right_comb = force_seq_index(pool.programs, pool.config.n_events)
    max_cyc = max(cyc_nodes(program) for program in pool.programs)
    return Split(
        name=name,
        kind="compositional_heldout_motif",
        rationale=rationale,
        train=train,
        test=test,
        metadata={
            "heldout_sign_patterns": sorted(f"{p:03b}" for p in withheld),
            "train_sign_patterns": sorted(
                {f"{pool.records[i].sign_pattern:03b}" for i in train}
            ),
            "train_target_programs": sorted(
                pool.program_names[p] for p in train_motifs
            ),
            "test_target_programs": sorted(pool.program_names[p] for p in test_motifs),
            "test_target_programs_unseen_in_train": sorted(
                pool.program_names[p] for p in test_motifs - train_motifs
            ),
            "constituent_head_choices_in_train": sorted(
                f"depth{d}:{'Sand' if s else 'Occ'}" for d, s in constituents_train
            ),
            "constituent_head_choices_in_test": sorted(
                f"depth{d}:{'Sand' if s else 'Occ'}" for d, s in constituents_test
            ),
            "every_test_constituent_seen_in_train": constituents_test
            <= constituents_train,
            # 008.04 section 4: the held-out target must not be a fixed executor.
            "heldout_target_is_right_comb": right_comb in test_motifs,
            "heldout_target_is_most_grouped": any(
                cyc_nodes(pool.programs[p]) == max_cyc for p in test_motifs
            ),
            "train_parities": sorted(
                {p.bit_count() % 2 for p in range(8) if p not in withheld}
            ),
        },
    )


# --- the 008.03 section 2.3 program-space viability gate --------------------


@dataclass(frozen=True)
class ViabilityGate:
    """Mechanical pre-training check. `008.04` section 4 requires all of it."""

    n_records: int
    n_scored: int
    target_defined_share: float
    joint_domain_share: float
    endpoint_disagreement_given_two: float
    n_availability_patterns: int
    max_distinct_endpoints_in_a_pattern: int
    availability_ceiling: float
    learned_headroom: float
    target_program_pool: tuple[str, ...]
    conformance: dict[str, dict]
    passed: bool

    def as_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "n_scored": self.n_scored,
            "target_defined_share": self.target_defined_share,
            "joint_domain_share": self.joint_domain_share,
            "endpoint_disagreement_given_two": self.endpoint_disagreement_given_two,
            "n_availability_patterns": self.n_availability_patterns,
            "max_distinct_endpoints_in_a_pattern": self.max_distinct_endpoints_in_a_pattern,
            "availability_ceiling": self.availability_ceiling,
            "learned_headroom": self.learned_headroom,
            "target_program_pool": list(self.target_program_pool),
            "conformance_against_008_02_pins": self.conformance,
            "passed": self.passed,
        }


def viability_gate(pool: Pool, *, min_headroom: float = 0.10) -> ViabilityGate:
    """Recompute the `008.02` pins on the sampled pool and compare."""
    records = pool.records
    scored = [records[i] for i in pool.scored]

    two_plus = sum(1 for r in records if len(r.legal) >= 2)
    disagreeing = sum(
        1 for r in records if len(r.legal) >= 2 and r.n_distinct_endpoints() >= 2
    )
    pattern_total: Counter[int] = Counter()
    pattern_hits: dict[int, Counter[int]] = defaultdict(Counter)
    pattern_endpoints: dict[int, int] = {}
    for r in records:
        pattern_total[r.availability_pattern] += 1
        pattern_endpoints[r.availability_pattern] = max(
            pattern_endpoints.get(r.availability_pattern, 0), r.n_distinct_endpoints()
        )
    for r in scored:
        for k in r.legal:
            if r.endpoint_class[k] == r.target_class:
                pattern_hits[r.availability_pattern][k] += 1

    denominator = len(scored) or 1
    ceiling = sum(
        max(hits.values()) for hits in pattern_hits.values() if hits
    ) / denominator
    headroom = 1.0 - ceiling

    measured = {
        "n_programs": len(pool.programs),
        "joint_domain_share": two_plus / len(records),
        "endpoint_disagreement_given_two": (disagreeing / two_plus) if two_plus else 0.0,
        "n_availability_patterns": len(pattern_total),
        "availability_ceiling": ceiling,
        "learned_headroom": headroom,
        "max_distinct_endpoints_in_a_pattern": max(pattern_endpoints.values()),
        "target_defined_share": len(scored) / len(records),
    }
    exact_keys = {"n_programs", "max_distinct_endpoints_in_a_pattern"}
    conformance: dict[str, dict] = {}
    for key, pin in PINS_00802.items():
        value = measured[key]
        direction = "two_sided"
        if key in exact_keys:
            ok = value == pin
            direction = "exact"
        elif key == "n_availability_patterns":
            # A uniform sample cannot be required to hit patterns that occur on
            # ~0.1 % of inputs, so the sample may only recover a subset.
            ok = value <= pin
            direction = "at_most"
        elif ONE_SIDED_PINS.get(key) == "at_least":
            ok = value >= pin - PIN_TOLERANCE
            direction = "at_least"
        elif ONE_SIDED_PINS.get(key) == "at_most":
            ok = value <= pin + PIN_TOLERANCE
            direction = "at_most"
        else:
            ok = abs(value - pin) <= PIN_TOLERANCE
        conformance[key] = {
            "pin_008_02": pin,
            "measured": value,
            "conforms": ok,
            "check": direction,
            "deviation_from_pin": value - pin,
        }

    pool_programs = sorted(
        {pool.program_names[r.target_program] for r in records}
    )
    passed = (
        all(entry["conforms"] for entry in conformance.values())
        and headroom >= min_headroom
        and measured["endpoint_disagreement_given_two"] >= 0.10
    )
    return ViabilityGate(
        n_records=len(records),
        n_scored=len(scored),
        target_defined_share=measured["target_defined_share"],
        joint_domain_share=measured["joint_domain_share"],
        endpoint_disagreement_given_two=measured["endpoint_disagreement_given_two"],
        n_availability_patterns=measured["n_availability_patterns"],
        max_distinct_endpoints_in_a_pattern=measured[
            "max_distinct_endpoints_in_a_pattern"
        ],
        availability_ceiling=ceiling,
        learned_headroom=headroom,
        target_program_pool=tuple(pool_programs),
        conformance=conformance,
        passed=passed,
    )


# --- scoring ---------------------------------------------------------------

Selector = Callable[[Record], int | None]


@dataclass(frozen=True)
class BaselineScore:
    """Exact-native scoring of one selector on one split half."""

    name: str
    split: str
    half: str
    n: int
    exact_native_success: float
    tree_selection_accuracy: float
    selected_defined_share: float
    endpoint_accuracy_given_selected_defined: float
    endpoint_match_via_different_tree: float
    success_by_n_legal: dict[int, float]
    selection_frequencies: dict[str, float]
    per_program_regret: dict[str, int]
    deployable: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "split": self.split,
            "half": self.half,
            "n": self.n,
            "exact_native_success": self.exact_native_success,
            "tree_selection_accuracy": self.tree_selection_accuracy,
            "selected_defined_share": self.selected_defined_share,
            "endpoint_accuracy_given_selected_defined": (
                self.endpoint_accuracy_given_selected_defined
            ),
            "endpoint_match_via_different_tree": self.endpoint_match_via_different_tree,
            "success_by_n_legal_alternatives": {
                str(k): v for k, v in sorted(self.success_by_n_legal.items())
            },
            "selection_frequencies": self.selection_frequencies,
            "per_program_regret": self.per_program_regret,
            "deployable": self.deployable,
            "note": self.note,
        }


def score_selector(
    pool: Pool,
    indices: Sequence[int],
    selector: Selector,
    *,
    name: str,
    split: str,
    half: str,
    deployable: bool = True,
    note: str = "",
) -> BaselineScore:
    """Score any selector — deterministic or learned — by exact native semantics."""
    n = len(indices)
    hits = tree_hits = defined_hits = cross_tree = 0
    by_n_legal_total: Counter[int] = Counter()
    by_n_legal_hits: Counter[int] = Counter()
    picks: Counter[int] = Counter()
    regret: Counter[int] = Counter()
    for i in indices:
        record = pool.records[i]
        pick = selector(record)
        n_legal = len(record.legal)
        by_n_legal_total[n_legal] += 1
        if pick is not None:
            picks[pick] += 1
        ok = record.matches(pick)
        if ok:
            hits += 1
            by_n_legal_hits[n_legal] += 1
        elif pick is not None:
            regret[pick] += 1
        if pick is not None and record.defined[pick]:
            defined_hits += 1
        if pick == record.target_program:
            tree_hits += 1
        elif ok:
            cross_tree += 1
    denominator = n or 1
    return BaselineScore(
        name=name,
        split=split,
        half=half,
        n=n,
        exact_native_success=hits / denominator,
        tree_selection_accuracy=tree_hits / denominator,
        selected_defined_share=defined_hits / denominator,
        endpoint_accuracy_given_selected_defined=(
            hits / defined_hits if defined_hits else 0.0
        ),
        endpoint_match_via_different_tree=cross_tree / denominator,
        success_by_n_legal={
            k: by_n_legal_hits[k] / by_n_legal_total[k]
            for k in sorted(by_n_legal_total)
        },
        selection_frequencies={
            pool.program_names[k]: picks[k] / denominator for k in sorted(picks)
        },
        per_program_regret={
            pool.program_names[k]: regret[k] for k in sorted(regret)
        },
        deployable=deployable,
        note=note,
    )


# --- deterministic baselines ------------------------------------------------


def _grouped_first(pool: Pool) -> Selector:
    order = grouped_first_order(pool.programs)

    def select(record: Record) -> int | None:
        for k in order:
            if record.defined[k]:
                return k
        return None

    return select


def _force_seq(pool: Pool) -> Selector:
    index = force_seq_index(pool.programs, pool.config.n_events)

    def select(record: Record) -> int | None:
        return index

    return select


def _fitted_rank(pool: Pool, train: Sequence[int]) -> tuple[int, ...]:
    """Programs ranked by exact-native hit count on the training half."""
    hits: Counter[int] = Counter()
    for i in train:
        record = pool.records[i]
        for k in record.legal:
            if record.endpoint_class[k] == record.target_class:
                hits[k] += 1
    return tuple(
        sorted(range(len(pool.programs)), key=lambda k: (-hits[k], k))
    )


def _first_legal_in_rank(rank: Sequence[int]) -> Selector:
    def select(record: Record) -> int | None:
        for k in rank:
            if record.defined[k]:
                return k
        return None

    return select


def _availability_rule_fitted(pool: Pool, train: Sequence[int]) -> Selector:
    """Best program per availability pattern, fitted on train only.

    The strongest *deployable* availability rule there is: it is exactly the
    per-pattern argmax that the `008.02` ceiling upper-bounds, estimated without
    touching a held-out label.
    """
    per_pattern: dict[int, Counter[int]] = defaultdict(Counter)
    for i in train:
        record = pool.records[i]
        for k in record.legal:
            if record.endpoint_class[k] == record.target_class:
                per_pattern[record.availability_pattern][k] += 1
    choice = {
        pattern: max(sorted(hits), key=lambda k: hits[k])
        for pattern, hits in per_pattern.items()
    }
    fallback = _first_legal_in_rank(_fitted_rank(pool, train))

    def select(record: Record) -> int | None:
        pick = choice.get(record.availability_pattern)
        if pick is not None and record.defined[pick]:
            return pick
        return fallback(record)

    return select


def _availability_oracle(pool: Pool, indices: Sequence[int]) -> Selector:
    """Per-pattern argmax fitted on the half being scored. Not deployable."""
    per_pattern: dict[int, Counter[int]] = defaultdict(Counter)
    for i in indices:
        record = pool.records[i]
        for k in record.legal:
            if record.endpoint_class[k] == record.target_class:
                per_pattern[record.availability_pattern][k] += 1
    choice = {
        pattern: max(sorted(hits), key=lambda k: hits[k])
        for pattern, hits in per_pattern.items()
    }

    def select(record: Record) -> int | None:
        pick = choice.get(record.availability_pattern)
        if pick is not None and record.defined[pick]:
            return pick
        legal = record.legal
        return legal[0] if legal else None

    return select


def _sign_pattern_lookup(pool: Pool, train: Sequence[int]) -> Selector:
    """Per-sign-pattern argmax fitted on train: a deterministic lookup table.

    Not an availability rule — it reads an allowed *input* feature — so it is
    not covered by the `008.02` ceiling and has to be carried explicitly.
    `008.04` section 5.B requires adding any such rule rather than ignoring it.
    On the ``random`` split it is the rule to beat; on the ``motif`` split it has
    nothing to look up for the withheld patterns and must fall back.
    """
    per_pattern: dict[int, Counter[int]] = defaultdict(Counter)
    for i in train:
        record = pool.records[i]
        for k in record.legal:
            if record.endpoint_class[k] == record.target_class:
                per_pattern[record.sign_pattern][k] += 1
    choice = {
        pattern: max(sorted(hits), key=lambda k: hits[k])
        for pattern, hits in per_pattern.items()
    }
    fallback = _first_legal_in_rank(_fitted_rank(pool, train))

    def select(record: Record) -> int | None:
        pick = choice.get(record.sign_pattern)
        if pick is not None and record.defined[pick]:
            return pick
        return fallback(record)

    return select


def _supplied_bracketing() -> Selector:
    """The `H_weak` reference: the target tree is handed over and executed."""

    def select(record: Record) -> int | None:
        return record.target_program

    return select


#: Name -> (builder, deployable, note). ``RandomLegalTree`` is scored
#: analytically rather than by sampling, so it is handled separately.
BASELINES: tuple[str, ...] = (
    "AvailabilityOracleCeiling",
    "AvailabilityRuleTrainFitted",
    "GroupedFirst",
    "ForceSeq",
    "RandomLegalTree",
    "MajorityProgramTrainFitted",
    "SignPatternLookupTrainFitted",
    "SuppliedBracketingExecutor",
)


def _random_legal_score(
    pool: Pool, indices: Sequence[int], *, split: str, half: str
) -> BaselineScore:
    """Expected exact-native success of a uniform draw over legal programs."""
    n = len(indices)
    expected = 0.0
    tree_expected = 0.0
    by_n_legal_total: Counter[int] = Counter()
    by_n_legal_hits: dict[int, float] = defaultdict(float)
    for i in indices:
        record = pool.records[i]
        legal = record.legal
        by_n_legal_total[len(legal)] += 1
        if not legal:
            continue
        matching = sum(
            1 for k in legal if record.endpoint_class[k] == record.target_class
        )
        expected += matching / len(legal)
        by_n_legal_hits[len(legal)] += matching / len(legal)
        if record.defined[record.target_program]:
            tree_expected += 1 / len(legal)
    denominator = n or 1
    return BaselineScore(
        name="RandomLegalTree",
        split=split,
        half=half,
        n=n,
        exact_native_success=expected / denominator,
        tree_selection_accuracy=tree_expected / denominator,
        selected_defined_share=1.0,
        endpoint_accuracy_given_selected_defined=expected / denominator,
        endpoint_match_via_different_tree=max(
            0.0, (expected - tree_expected) / denominator
        ),
        success_by_n_legal={
            k: by_n_legal_hits[k] / by_n_legal_total[k] for k in sorted(by_n_legal_total)
        },
        selection_frequencies={},
        per_program_regret={},
        note="analytic expectation over a uniform draw from the legal set",
    )


def score_baselines(pool: Pool, split: Split) -> list[BaselineScore]:
    """Every deterministic baseline, on both halves of one split."""
    out: list[BaselineScore] = []
    rank = _fitted_rank(pool, split.train)
    for half, indices in (("train", split.train), ("test", split.test)):
        builders: list[tuple[str, Selector, bool, str]] = [
            (
                "AvailabilityOracleCeiling",
                _availability_oracle(pool, indices),
                False,
                (
                    "per-availability-pattern argmax fitted on the scored half; "
                    "diagnostic upper bound on every availability rule, not a "
                    "deployable learner"
                ),
            ),
            (
                "AvailabilityRuleTrainFitted",
                _availability_rule_fitted(pool, split.train),
                True,
                "strongest deployable availability-only rule",
            ),
            ("GroupedFirst", _grouped_first(pool), True, "017.24 rule, generalized"),
            ("ForceSeq", _force_seq(pool), True, "pure right comb"),
            (
                "MajorityProgramTrainFitted",
                _first_legal_in_rank(rank),
                True,
                "frequency baseline: first legal program in train-hit order",
            ),
            (
                "SignPatternLookupTrainFitted",
                _sign_pattern_lookup(pool, split.train),
                True,
                (
                    "deterministic lookup on an allowed input feature; not an "
                    "availability rule, so not covered by the 008.02 ceiling"
                ),
            ),
            (
                "SuppliedBracketingExecutor",
                _supplied_bracketing(),
                False,
                "H_weak reference: the target tree is stipulated, not selected",
            ),
        ]
        for name, selector, deployable, note in builders:
            out.append(
                score_selector(
                    pool,
                    indices,
                    selector,
                    name=name,
                    split=split.name,
                    half=half,
                    deployable=deployable,
                    note=note,
                )
            )
        out.append(_random_legal_score(pool, indices, split=split.name, half=half))
    return out


def best_deterministic(
    scores: Sequence[BaselineScore], *, half: str = "test"
) -> BaselineScore:
    """The strongest *deployable* deterministic baseline on a half.

    Excludes the availability oracle (fitted on the scored half) and the
    supplied-bracketing executor (which is handed the answer's tree). Those are
    reference bounds, not rules a learner competes against.
    """
    candidates = [
        s
        for s in scores
        if s.half == half and s.deployable and s.name != "SuppliedBracketingExecutor"
    ]
    if not candidates:
        raise ValueError("no deployable deterministic baseline scored")
    return max(candidates, key=lambda s: s.exact_native_success)


def legal_program_counts(pool: Pool, indices: Sequence[int]) -> dict[str, int]:
    """`008.04` section 8 legal-program-count histogram."""
    hist: Counter[int] = Counter()
    for i in indices:
        hist[len(pool.records[i].legal)] += 1
    return {str(k): hist[k] for k in sorted(hist)}


def cyc_free_program_names(pool: Pool) -> tuple[str, ...]:
    return tuple(pool.program_names[k] for k in cyc_free_indices(pool.programs))

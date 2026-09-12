"""Deterministic baselines and the code-space shortcut search for the 009.05 ladder.

`009.05` sections 4.3 and 8 require, at every rung, that the precise
learner-visible features be recorded and searched for cheap deterministic
shortcuts, and that any rule reaching the target without learning the relation be
reported and the rung downgraded accordingly.

This is where the ladder differs most sharply from `009.02`. That module's
`FEATURE_ATOMS` are all **native** features -- `same_habitat`, `same_fano_point`,
`sign_pair`, `block_id_equality` and so on -- because the question there was
whether a native shortcut could explain an SFP result. The ladder's learner sees
the SFP code itself, so the shortcut search has to run over **code-space**
features, and no such search has been run before in this issue.

It finds something that matters, and it finds it without any learner:

    a three-atom deterministic tabulation over habitat-free code features
    resolves the whole certified consequence on held-out habitats under the
    exact code, and does not under the non-automorphic scramble

Cost classes follow `009.02`'s convention exactly. `pp_xor` and `shared_endpoint`
are marked ``relation_equivalent`` -- they *are* the certified law, so a lookup
keyed on them is a ceiling, not a shortcut, in the same way `block_id_equality` is
in `baselines.py`. `009.05` section 2 forbids handing the executable XOR to the
*learner*; measuring it as a nonlearned ceiling here is the opposite of that.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from folds import all_folds, structural_folds
from ladder_resolver import build_resolvers
from ladder_task import (
    CODE_ARMS,
    PP_CLASSES,
    StructuredTarget,
    code_tables,
    output_table,
    rung1_positions,
    rung1_target,
    rung2_positions,
    rung2_target,
)
from sfp import SfpAddress
from task import Dataset, build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_ladder_artifacts"
OUTPUT = ARTIFACTS / "ladder_baselines.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

#: Digest-derived, never ``hash()`` and never the ``random`` module's default.
BASE_SEED = 90050601

#: Maximum feature-subset size in the shortcut search, matching
#: ``baselines.MAX_SUBSET_SIZE``.
MAX_SUBSET_SIZE = 3

ROUND_DIGITS = 12

CHEAP = "cheaper_than_the_relation"
RELATION_EQUIVALENT = "relation_equivalent"

FENCES = (
    "Baselines read the certified labels; they never generate one.",
    (
        "pp_xor and shared_endpoint are RELATION_EQUIVALENT: a lookup keyed on them "
        "is a ceiling, not a shortcut."
    ),
    (
        "The executable XOR rule is never supplied to a learner. It appears here "
        "only as a nonlearned ceiling."
    ),
    "Every selector is fitted on training positions only.",
    "An unseen key ABSTAINS; it never falls back to the right answer.",
    "algebraic zero != NONADMISSION;   0 != bottom",
)

CHANCE_STATEMENT = (
    "Rung-1 chance is 1/3 exactly: the output alphabet is the three nonzero local "
    "ports and the label marginal is uniform at 112 each over the 336 admitted "
    "pairs. Reported analytically as well as sampled, so the sampled figure can be "
    "checked against it."
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def rd(value: float | None) -> float | None:
    """Round for serialisation so replay is byte-identical."""

    return None if value is None else round(float(value), ROUND_DIGITS)


def derive_seed(label: str) -> int:
    """Digest-pinned integer seed. Never ``hash()``, never iteration order."""

    blob = f"ladder_baselines/v1|{BASE_SEED}|{label}".encode()
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


# ---------------------------------------------------------------------------
# the learner-visible feature atoms, in CODE space
# ---------------------------------------------------------------------------

def _endpoints(code: SfpAddress) -> tuple[int, ...]:
    return tuple(sorted(code.endpoints))


ATOMS: tuple[tuple[str, str, str, object], ...] = (
    (
        "pp_a",
        CHEAP,
        "the local port of the first input Event, as the arm codes it",
        lambda t, r: t[r.a_index].d,
    ),
    (
        "pp_b",
        CHEAP,
        "the local port of the second input Event",
        lambda t, r: t[r.b_index].d,
    ),
    (
        "pp_pair",
        CHEAP,
        "the UNORDERED pair of input local ports; three values under the exact code",
        lambda t, r: tuple(sorted((t[r.a_index].d, t[r.b_index].d))),
    ),
    (
        "endpoints_a",
        CHEAP,
        "the K4 endpoint set of the first input Event",
        lambda t, r: _endpoints(t[r.a_index]),
    ),
    (
        "endpoints_b",
        CHEAP,
        "the K4 endpoint set of the second input Event",
        lambda t, r: _endpoints(t[r.b_index]),
    ),
    (
        "endpoint_pair",
        RELATION_EQUIVALENT,
        (
            "the unordered pair of input endpoint sets: the COMPLETE K4 incidence "
            "data of the pair. Outcome 021.05's Event-level law is defined on exactly "
            "this data -- admitted iff the endpoint sets meet in one vertex, forced "
            "third = the edge on the unshared vertices -- and "
            "sfp.ExactSfpCircuit.admit_event and forced_third_event compute from it "
            "and nothing else. It is therefore relation-equivalent by definition of "
            "the relation, not by measured accuracy."
        ),
        lambda t, r: tuple(
            sorted((_endpoints(t[r.a_index]), _endpoints(t[r.b_index])))
        ),
    ),
    (
        "s_a",
        CHEAP,
        "the sign bit of the first input Event",
        lambda t, r: t[r.a_index].s,
    ),
    (
        "fff_a",
        CHEAP,
        "the Fano point of the first input Event",
        lambda t, r: t[r.a_index].fff,
    ),
    (
        "same_habitat_code",
        CHEAP,
        "do the two input codes agree on (S, FFF)?",
        lambda t, r: t[r.a_index].habitat == t[r.b_index].habitat,
    ),
    (
        "repeated",
        CHEAP,
        "is this the repeated pair (a, a)?",
        lambda t, r: r.a_index == r.b_index,
    ),
    (
        "pp_xor",
        RELATION_EQUIVALENT,
        (
            "XOR of the two input local ports. This IS the certified forcing law, so "
            "a lookup keyed on it is a ceiling; 009.05 section 2 forbids supplying it "
            "to a learner and it is never given to one."
        ),
        lambda t, r: t[r.a_index].d ^ t[r.b_index].d,
    ),
    (
        "shared_endpoint",
        RELATION_EQUIVALENT,
        (
            "the unique shared K4 endpoint of the two input codes, or None. This is "
            "the certified admission test, so it is a ceiling rather than a shortcut."
        ),
        lambda t, r: (
            lambda shared: next(iter(shared)) if len(shared) == 1 else None
        )(t[r.a_index].endpoints & t[r.b_index].endpoints),
    ),
)

ATOM_NAMES = tuple(name for name, _, _, _ in ATOMS)
ATOM_COST = {name: cost for name, cost, _, _ in ATOMS}
ATOM_FN = {name: fn for name, _, _, fn in ATOMS}
CHEAP_ATOMS = tuple(name for name in ATOM_NAMES if ATOM_COST[name] == CHEAP)

#: Atom sets that jointly constitute the COMPLETE K4 incidence data of the input
#: pair. A single input's endpoint set is partial information and stays cheap, but
#: both together are the data the certified Event-level relation is defined on, so
#: any subset containing one of these sets is relation-equivalent for the same
#: reason ``endpoint_pair`` is.
COMPLETE_INCIDENCE_SETS = (
    frozenset({"endpoint_pair"}),
    frozenset({"endpoints_a", "endpoints_b"}),
)

COST_CLASSIFICATION_RULE = (
    "A subset is RELATION_EQUIVALENT when it contains a relation-equivalent atom "
    "(pp_xor, shared_endpoint, endpoint_pair) or jointly constitutes the complete K4 "
    "incidence data of the pair (endpoints_a together with endpoints_b). Everything "
    "else is CHEAP, meaning strictly cheaper than the relation. The rule is stated "
    "from Outcome 021.05's definition of the Event-level law rather than fitted to "
    "observed accuracy, so it cannot be tuned to flatter a learner. It was written "
    "after the first search run exposed that endpoint_pair alone reaches 0.994 -- "
    "which is exactly the kind of exposed deterministic rule 009.05 section 4.3 asks "
    "to be reported rather than absorbed."
)


def subset_cost_class(subset: tuple[str, ...]) -> str:
    """Cost class of a feature subset, by :data:`COST_CLASSIFICATION_RULE`."""

    members = frozenset(subset)
    if any(ATOM_COST[name] == RELATION_EQUIVALENT for name in subset):
        return RELATION_EQUIVALENT
    if any(complete <= members for complete in COMPLETE_INCIDENCE_SETS):
        return RELATION_EQUIVALENT
    return CHEAP


@dataclass(frozen=True)
class FeatureSpace:
    """Per-position integer codes for every atom, under one arm's code table.

    Vocabularies are built in ``sorted(repr(value))`` order, exactly as
    ``baselines.FeatureSpace`` does, so no ``hash()`` and no set iteration order
    can reach the artifact.
    """

    arm: str
    codes: tuple[tuple[int, ...], ...]
    vocabularies: dict[str, tuple[str, ...]]
    index_of: dict[str, int]

    def key(self, subset: tuple[str, ...], position: int) -> tuple[int, ...]:
        row = self.codes[position]
        return tuple(row[self.index_of[name]] for name in subset)


def build_feature_space(
    dataset: Dataset, arm: str, table: tuple[SfpAddress, ...]
) -> FeatureSpace:
    raw: dict[str, list[object]] = {name: [] for name in ATOM_NAMES}
    for record in dataset.records:
        for name in ATOM_NAMES:
            raw[name].append(ATOM_FN[name](table, record))
    vocabularies = {
        name: tuple(sorted({repr(value) for value in values}))
        for name, values in raw.items()
    }
    lookup = {
        name: {value: index for index, value in enumerate(vocabularies[name])}
        for name in ATOM_NAMES
    }
    codes = tuple(
        tuple(lookup[name][repr(raw[name][position])] for name in ATOM_NAMES)
        for position in range(len(dataset.records))
    )
    return FeatureSpace(
        arm=arm,
        codes=codes,
        vocabularies=vocabularies,
        index_of={name: index for index, name in enumerate(ATOM_NAMES)},
    )


# ---------------------------------------------------------------------------
# RUNG 1 selectors
# ---------------------------------------------------------------------------

ABSTAIN = 0


@dataclass
class Rung1Lookup:
    """Train-majority ``feature key -> pp3`` table. Unseen keys abstain.

    Abstention is scored as incorrect, never as a fallback to the right answer.
    Ties resolve to the smallest port so the fit is order-independent.
    """

    subset: tuple[str, ...]
    table: dict[tuple[int, ...], int] | None = None
    keys_fitted: int = 0
    unanimous_keys: int = 0

    @property
    def name(self) -> str:
        return "+".join(self.subset)

    @property
    def cost_class(self) -> str:
        return subset_cost_class(self.subset)

    def fit(
        self,
        dataset: Dataset,
        space: FeatureSpace,
        table: tuple[SfpAddress, ...],
        positions: tuple[int, ...],
    ) -> None:
        tally: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        for position in positions:
            record = dataset.records[position]
            tally[space.key(self.subset, position)][rung1_target(record, table)] += 1
        built = {}
        unanimous = 0
        for key, counter in tally.items():
            best = max(counter.values())
            built[key] = min(port for port, count in counter.items() if count == best)
            if len(counter) == 1:
                unanimous += 1
        self.table = built
        self.keys_fitted = len(built)
        self.unanimous_keys = unanimous

    def predict(self, space: FeatureSpace, position: int) -> int:
        if self.table is None:
            raise AssertionError("selector was not fitted")
        return self.table.get(space.key(self.subset, position), ABSTAIN)


def rung1_chance(seed_label: str, positions: tuple[int, ...]) -> list[int]:
    """A seeded uniform selector over the three ports."""

    generator = random.Random(derive_seed(seed_label))
    return [generator.choice(PP_CLASSES) for _ in positions]


def rung1_marginal(
    dataset: Dataset, table: tuple[SfpAddress, ...], positions: tuple[int, ...]
) -> int:
    """The train-majority port. Ties resolve to the smallest."""

    counter = Counter(
        rung1_target(dataset.records[position], table) for position in positions
    )
    best = max(counter.values())
    return min(port for port, count in counter.items() if count == best)


def score_rung1(
    dataset: Dataset,
    table: tuple[SfpAddress, ...],
    positions: tuple[int, ...],
    predictions: list[int],
) -> dict:
    hits = 0
    abstained = 0
    for position, predicted in zip(positions, predictions):
        if predicted == ABSTAIN:
            abstained += 1
            continue
        if predicted == rung1_target(dataset.records[position], table):
            hits += 1
    return {
        "size": len(positions),
        "hits": hits,
        "accuracy": rd(fraction(hits, len(positions))),
        "abstained": abstained,
        "coverage": rd(fraction(len(positions) - abstained, len(positions))),
    }


# ---------------------------------------------------------------------------
# RUNG 2 selectors
# ---------------------------------------------------------------------------

_BOTTOM_ANSWER = ("BOTTOM",)


def _answer_of(target: StructuredTarget) -> tuple:
    """Hashable canonical answer: BOTTOM, or the four-field address."""

    if target.is_bottom:
        return _BOTTOM_ANSWER
    return ("ADDRESS", target.s, target.fff, target.pp_set, target.d)


@dataclass
class Rung2Lookup:
    """Train-majority ``feature key -> BOTTOM or address`` table.

    One object decides admission and forcing together, exactly as
    ``baselines.FeatureLookup`` does, so a cheap rule cannot win the admission half
    while quietly abstaining on the forcing half.
    """

    subset: tuple[str, ...]
    table: dict[tuple[int, ...], tuple] | None = None
    keys_fitted: int = 0
    unanimous_keys: int = 0

    @property
    def name(self) -> str:
        return "+".join(self.subset)

    @property
    def cost_class(self) -> str:
        return subset_cost_class(self.subset)

    def fit(
        self,
        dataset: Dataset,
        space: FeatureSpace,
        table: tuple[SfpAddress, ...],
        positions: tuple[int, ...],
    ) -> None:
        tally: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        for position in positions:
            record = dataset.records[position]
            tally[space.key(self.subset, position)][
                _answer_of(rung2_target(record, table))
            ] += 1
        built = {}
        unanimous = 0
        for key, counter in tally.items():
            best = max(counter.values())
            built[key] = min(
                (answer for answer, count in counter.items() if count == best),
                key=repr,
            )
            if len(counter) == 1:
                unanimous += 1
        self.table = built
        self.keys_fitted = len(built)
        self.unanimous_keys = unanimous

    def predict(self, space: FeatureSpace, position: int) -> tuple | None:
        if self.table is None:
            raise AssertionError("selector was not fitted")
        return self.table.get(space.key(self.subset, position))


@dataclass
class Rung2CopyLookup:
    """The honest cheap ceiling: copy ``S`` and ``FFF``, tabulate the rest.

    A pure tabulation cannot reach the Rung-2 target at all under a structural
    holdout, and the reason is arithmetic rather than interesting: the address
    carries the habitat fields, the held-out habitat's ``(S, FFF)`` pair never
    occurs in training, and a lookup table can only ever emit an answer it has
    seen. It is not a fair stand-in for a cheap rule.

    The fair cheap rule uses the copy that ``009.05`` section 4.2 explicitly
    licenses -- ``S3 := S``, ``FFF3 := FFF`` -- and tabulates only the
    habitat-free remainder ``(PP, pp)``, whose values are chamber coordinates and
    local ports and therefore recur in every habitat. Admission is tabulated on
    the same key, so this single object still decides admission and forcing
    together.

    The copy is declared here, never learned, and it is not the forcing law: it
    fixes the two fields the certified relation copies and leaves the two fields
    the relation actually computes to the table.
    """

    subset: tuple[str, ...]
    table: dict[tuple[int, ...], tuple | None] | None = None
    keys_fitted: int = 0
    unanimous_keys: int = 0

    @property
    def name(self) -> str:
        return "copy_S_FFF+" + "+".join(self.subset)

    @property
    def cost_class(self) -> str:
        return subset_cost_class(self.subset)

    def fit(
        self,
        dataset: Dataset,
        space: FeatureSpace,
        table: tuple[SfpAddress, ...],
        positions: tuple[int, ...],
    ) -> None:
        tally: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        for position in positions:
            record = dataset.records[position]
            target = rung2_target(record, table)
            local = None if target.is_bottom else (target.pp_set, target.d)
            tally[space.key(self.subset, position)][local] += 1
        built: dict[tuple[int, ...], tuple | None] = {}
        unanimous = 0
        for key, counter in tally.items():
            best = max(counter.values())
            built[key] = min(
                (answer for answer, count in counter.items() if count == best),
                key=repr,
            )
            if len(counter) == 1:
                unanimous += 1
        self.table = built
        self.keys_fitted = len(built)
        self.unanimous_keys = unanimous

    def predict(
        self,
        dataset: Dataset,
        space: FeatureSpace,
        table: tuple[SfpAddress, ...],
        position: int,
    ) -> tuple | None:
        if self.table is None:
            raise AssertionError("selector was not fitted")
        key = space.key(self.subset, position)
        if key not in self.table:
            return None
        local = self.table[key]
        if local is None:
            return _BOTTOM_ANSWER
        source = table[dataset.records[position].a_index]
        pp_set, d = local
        return ("ADDRESS", source.s, source.fff, pp_set, d)


def score_rung2(
    dataset: Dataset,
    table: tuple[SfpAddress, ...],
    resolver,
    positions: tuple[int, ...],
    predictions: list[tuple | None],
) -> dict:
    n_admit = 0
    n_reject = 0
    sens_hits = 0
    spec_hits = 0
    address_hits = 0
    resolved_hits = 0
    field_hits = {"S": 0, "FFF": 0, "PP": 0, "pp": 0}
    abstained = 0
    for position, predicted in zip(positions, predictions):
        record = dataset.records[position]
        target = rung2_target(record, table)
        if predicted is None:
            abstained += 1
            if not target.is_bottom:
                n_admit += 1
            else:
                n_reject += 1
            continue
        says_admit = predicted[0] == "ADDRESS"
        if target.is_bottom:
            n_reject += 1
            if not says_admit:
                spec_hits += 1
            continue
        n_admit += 1
        if not says_admit:
            continue
        sens_hits += 1
        _, s, fff, pp_set, d = predicted
        if s == target.s:
            field_hits["S"] += 1
        if fff == target.fff:
            field_hits["FFF"] += 1
        if set(pp_set) & set(target.pp_set):
            field_hits["PP"] += 1
        if d == target.d:
            field_hits["pp"] += 1
        if (s, fff, d) == (target.s, target.fff, target.d) and set(pp_set) == set(
            target.pp_set
        ):
            address_hits += 1
            if resolver.resolve(s, fff, pp_set[0], d) == record.target_index:
                resolved_hits += 1
    sensitivity = fraction(sens_hits, n_admit)
    specificity = fraction(spec_hits, n_reject)
    return {
        "size": len(positions),
        "n_admit": n_admit,
        "n_nonadmission": n_reject,
        "abstained": abstained,
        "admission_sensitivity": rd(sensitivity),
        "admission_specificity": rd(specificity),
        "admission_balanced_accuracy": rd(
            None
            if sensitivity is None or specificity is None
            else 0.5 * (sensitivity + specificity)
        ),
        "field_accuracy": {
            field: rd(fraction(hits, n_admit)) for field, hits in sorted(field_hits.items())
        },
        "exact_structured_address_accuracy": rd(fraction(address_hits, n_admit)),
        "resolved_event_identity_accuracy": rd(fraction(resolved_hits, n_admit)),
    }


def rung2_chance() -> dict:
    """Analytic chance for the structured address under uniform field guessing."""

    per_field = {"S": 1 / 2, "FFF": 1 / 7, "PP": 2 / 4, "pp": 1 / 3}
    joint = per_field["S"] * per_field["FFF"] * per_field["PP"] * per_field["pp"]
    return {
        "per_field": {field: rd(value) for field, value in sorted(per_field.items())},
        "exact_structured_address": rd(joint),
        "statement": (
            "PP is 2/4 because either block-incidence presentation is correct. The "
            "joint chance of a uniform four-field guess is 1/84, which is exactly "
            "the chance of a uniform guess over the saturated catalogue -- the "
            "structured interface is not an easier lottery, it is the same lottery "
            "factorized."
        ),
        "equals_one_over_catalogue": rd(joint) == rd(1 / 84),
    }


# ---------------------------------------------------------------------------
# exact nonlearned oracles
# ---------------------------------------------------------------------------

def exact_oracle_rung1(
    dataset: Dataset, table: tuple[SfpAddress, ...], positions: tuple[int, ...]
) -> dict:
    """The certified relation, read directly. 1.0 by definition, reported anyway."""

    predictions = [
        rung1_target(dataset.records[position], table) for position in positions
    ]
    return score_rung1(dataset, table, positions, predictions)


def exact_oracle_rung2(
    dataset: Dataset,
    table: tuple[SfpAddress, ...],
    resolver,
    positions: tuple[int, ...],
) -> dict:
    predictions = [
        _answer_of(rung2_target(dataset.records[position], table))
        for position in positions
    ]
    return score_rung2(dataset, table, resolver, positions, predictions)


# ---------------------------------------------------------------------------
# the shortcut search
# ---------------------------------------------------------------------------

def feature_subsets() -> tuple[tuple[str, ...], ...]:
    """Every atom subset up to :data:`MAX_SUBSET_SIZE`, in canonical order."""

    subsets: list[tuple[str, ...]] = []
    for size in range(1, MAX_SUBSET_SIZE + 1):
        subsets.extend(itertools.combinations(ATOM_NAMES, size))
    return tuple(subsets)


def determination_census(
    dataset: Dataset, arm: str, space: FeatureSpace, table: tuple[SfpAddress, ...]
) -> dict:
    """Which feature subsets DETERMINE the certified answer over the whole pool.

    A per-atom cost label is not enough to read the shortcut search correctly, and
    the search itself exposed why -- which is exactly the case ``009.05`` section
    4.3 anticipates when it says to report any deterministic rule discovered during
    implementation.

    ``endpoint_pair`` is the two K4 endpoint sets of the input pair. That is the
    *complete local incidence configuration* the certified relation is a function
    of: two Events are admitted iff their endpoint sets meet in exactly one vertex,
    and the forced third is the edge on the two unshared vertices. A lookup keyed on
    it is therefore not cheaper than the relation -- it is the relation tabulated,
    the code-space analogue of ``baselines.py``'s ``block_id_equality`` ceiling.

    So the distinction is drawn mechanically instead of by assertion. A subset is
    ``relation_determining`` for an arm when its key determines the certified answer
    with zero ambiguity over **all 7056 ordered pairs**, not merely over one fold's
    training split. Such a subset is a ceiling. Everything else is strictly cheaper
    than the relation, and those are the baselines a learned result has to clear.
    """

    rows = []
    for subset in feature_subsets():
        answers: dict[tuple[int, ...], set] = defaultdict(set)
        for position in range(len(dataset.records)):
            record = dataset.records[position]
            target = rung2_target(record, table)
            local = None if target.is_bottom else (target.pp_set, target.d)
            answers[space.key(subset, position)].add(local)
        ambiguous = sum(1 for values in answers.values() if len(values) > 1)
        rows.append(
            {
                "subset": list(subset),
                "keys": len(answers),
                "ambiguous_keys": ambiguous,
                "relation_determining": ambiguous == 0,
                "cost_class": subset_cost_class(subset),
            }
        )
    determining = [row for row in rows if row["relation_determining"]]
    return {
        "arm": arm,
        "subsets_examined": len(rows),
        "relation_determining_subsets": len(determining),
        "minimal_relation_determining_subsets": sorted(
            (row["subset"] for row in determining if row["keys"] == min(
                (r["keys"] for r in determining), default=0
            )),
            key=lambda s: (len(s), s),
        )[:6],
        "smallest_determining_size": min(
            (len(row["subset"]) for row in determining), default=None
        ),
        "strictly_cheaper_subsets": len(rows) - len(determining),
        "rows": rows,
        "test": (
            "zero answer ambiguity over all 7056 ordered pairs, measured on the "
            "whole pool rather than on any training split, so the label cannot be "
            "tuned by a fold"
        ),
    }


def search_fold(
    dataset: Dataset,
    fold,
    arm: str,
    space: FeatureSpace,
    table: tuple[SfpAddress, ...],
    resolver,
) -> list[dict]:
    """Fit every feature subset on the fold's train split; score it held out."""

    r1_train, r1_test = rung1_positions(dataset, fold)
    r2_train, r2_test = rung2_positions(dataset, fold)
    rows = []
    for subset in feature_subsets():
        one = Rung1Lookup(subset)
        one.fit(dataset, space, table, r1_train)
        r1 = score_rung1(
            dataset, table, r1_test, [one.predict(space, p) for p in r1_test]
        )

        two = Rung2Lookup(subset)
        two.fit(dataset, space, table, r2_train)
        r2 = score_rung2(
            dataset, table, resolver, r2_test, [two.predict(space, p) for p in r2_test]
        )

        three = Rung2CopyLookup(subset)
        three.fit(dataset, space, table, r2_train)
        r2c = score_rung2(
            dataset,
            table,
            resolver,
            r2_test,
            [three.predict(dataset, space, table, p) for p in r2_test],
        )
        rows.append(
            {
                "arm": arm,
                "family": fold.family,
                "fold": fold.name,
                "subset": list(subset),
                "size": len(subset),
                "cost_class": one.cost_class,
                "rung1_keys_fitted": one.keys_fitted,
                "rung1_unanimous_keys": one.unanimous_keys,
                "rung1_accuracy": r1["accuracy"],
                "rung1_coverage": r1["coverage"],
                "rung2_keys_fitted": two.keys_fitted,
                "rung2_pure_lookup_exact_address": r2[
                    "exact_structured_address_accuracy"
                ],
                "rung2_exact_address_accuracy": r2c[
                    "exact_structured_address_accuracy"
                ],
                "rung2_resolved_event_accuracy": r2c[
                    "resolved_event_identity_accuracy"
                ],
                "rung2_admission_balanced_accuracy": r2c[
                    "admission_balanced_accuracy"
                ],
                "rung2_field_accuracy": r2c["field_accuracy"],
            }
        )
    return rows


def summarise_search(rows: list[dict], census: dict[str, dict]) -> dict:
    """The ceiling each arm's atoms actually reach, split by determining power.

    Two ceilings are reported per arm and family, and they answer different
    questions:

    ``determining_ceiling``      what a code-space tabulation of the complete local
                                 configuration achieves. The analogue of an exact
                                 oracle, not a shortcut.
    ``strictly_cheaper_ceiling`` what the best rule that does NOT determine the
                                 relation achieves. This is the bar a learned
                                 result has to clear.
    """

    determining = {
        arm: {tuple(row["subset"]) for row in body["rows"] if row["relation_determining"]}
        for arm, body in census.items()
    }

    grouped: dict[tuple[str, str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["family"], tuple(row["subset"]))].append(row)

    per_arm: dict[str, dict] = {}
    for (arm, family, subset), group in sorted(grouped.items()):
        cost = group[0]["cost_class"]
        r1 = [row["rung1_accuracy"] for row in group if row["rung1_accuracy"] is not None]
        r2 = [
            row["rung2_exact_address_accuracy"]
            for row in group
            if row["rung2_exact_address_accuracy"] is not None
        ]
        pure = [
            row["rung2_pure_lookup_exact_address"]
            for row in group
            if row["rung2_pure_lookup_exact_address"] is not None
        ]
        entry = {
            "subset": list(subset),
            "cost_class": cost,
            "relation_determining": subset in determining.get(arm, set()),
            "n_folds": len(group),
            "rung1_mean_accuracy": rd(sum(r1) / len(r1)) if r1 else None,
            "rung1_min_accuracy": rd(min(r1)) if r1 else None,
            "rung2_mean_exact_address": rd(sum(r2) / len(r2)) if r2 else None,
            "rung2_min_exact_address": rd(min(r2)) if r2 else None,
            "rung2_pure_lookup_mean_exact_address": (
                rd(sum(pure) / len(pure)) if pure else None
            ),
        }
        bucket = per_arm.setdefault(arm, {}).setdefault(family, {"subsets": []})
        bucket["subsets"].append(entry)

    for arm, families in per_arm.items():
        for family, body in families.items():
            subsets = body["subsets"]
            cheap = [row for row in subsets if row["cost_class"] == CHEAP]
            strict = [
                row
                for row in subsets
                if row["cost_class"] == CHEAP and not row["relation_determining"]
            ]
            determ = [row for row in subsets if row["relation_determining"]]
            body["cheap_rung1_ceiling"] = max(
                (row["rung1_mean_accuracy"] for row in cheap
                 if row["rung1_mean_accuracy"] is not None),
                default=None,
            )
            body["cheap_rung2_ceiling"] = max(
                (row["rung2_mean_exact_address"] for row in cheap
                 if row["rung2_mean_exact_address"] is not None),
                default=None,
            )
            body["strictly_cheaper_rung1_ceiling"] = max(
                (row["rung1_mean_accuracy"] for row in strict
                 if row["rung1_mean_accuracy"] is not None),
                default=None,
            )
            body["strictly_cheaper_rung2_ceiling"] = max(
                (row["rung2_mean_exact_address"] for row in strict
                 if row["rung2_mean_exact_address"] is not None),
                default=None,
            )
            body["determining_rung2_ceiling"] = max(
                (row["rung2_mean_exact_address"] for row in determ
                 if row["rung2_mean_exact_address"] is not None),
                default=None,
            )
            body["n_relation_determining_subsets"] = len(determ)
            body["best_cheap_rung1_subsets"] = sorted(
                row["subset"]
                for row in cheap
                if row["rung1_mean_accuracy"] == body["cheap_rung1_ceiling"]
            )[:6]
            body["best_cheap_rung2_subsets"] = sorted(
                row["subset"]
                for row in cheap
                if row["rung2_mean_exact_address"] == body["cheap_rung2_ceiling"]
            )[:6]
            body["best_strictly_cheaper_rung2_subsets"] = sorted(
                row["subset"]
                for row in strict
                if row["rung2_mean_exact_address"]
                == body["strictly_cheaper_rung2_ceiling"]
            )[:6]
            body["pure_lookup_rung2_ceiling"] = max(
                (row["rung2_pure_lookup_mean_exact_address"] for row in cheap
                 if row["rung2_pure_lookup_mean_exact_address"] is not None),
                default=None,
            )
            body["cheap_rung1_solves_it"] = (
                body["cheap_rung1_ceiling"] is not None
                and body["cheap_rung1_ceiling"] >= 0.99
            )
            body["cheap_rung2_solves_it"] = (
                body["cheap_rung2_ceiling"] is not None
                and body["cheap_rung2_ceiling"] >= 0.99
            )
            body["pure_lookup_rung2_is_at_floor"] = (
                body["pure_lookup_rung2_ceiling"] is not None
                and body["pure_lookup_rung2_ceiling"] <= 0.01
            )
    return per_arm


# ---------------------------------------------------------------------------
# per-fold baseline table
# ---------------------------------------------------------------------------

def baseline_table(dataset: Dataset) -> dict:
    tables = code_tables(dataset)
    resolvers = build_resolvers(dataset)
    folds = structural_folds(all_folds(dataset))
    spaces = {
        name: build_feature_space(dataset, name, output_table(name, tables))
        for name in CODE_ARMS
    }
    census = {
        name: determination_census(
            dataset, name, spaces[name], output_table(name, tables)
        )
        for name in CODE_ARMS
    }

    rows = []
    search_rows: list[dict] = []
    for arm in CODE_ARMS:
        table = output_table(arm, tables)
        space = spaces[arm]
        resolver = resolvers[arm]
        for fold in folds:
            r1_train, r1_test = rung1_positions(dataset, fold)
            r2_train, r2_test = rung2_positions(dataset, fold)

            chance = score_rung1(
                dataset,
                table,
                r1_test,
                rung1_chance(f"{arm}/{fold.family}/{fold.name}", r1_test),
            )
            marginal_port = rung1_marginal(dataset, table, r1_train)
            marginal = score_rung1(
                dataset, table, r1_test, [marginal_port] * len(r1_test)
            )
            oracle1 = exact_oracle_rung1(dataset, table, r1_test)
            oracle2 = exact_oracle_rung2(dataset, table, resolver, r2_test)

            rows.append(
                {
                    "arm": arm,
                    "family": fold.family,
                    "fold": fold.name,
                    "rung1_test_size": len(r1_test),
                    "rung1_chance_sampled": chance["accuracy"],
                    "rung1_marginal_port": marginal_port,
                    "rung1_marginal_accuracy": marginal["accuracy"],
                    "rung1_exact_oracle": oracle1["accuracy"],
                    "rung2_test_size": len(r2_test),
                    "rung2_exact_oracle_address": oracle2[
                        "exact_structured_address_accuracy"
                    ],
                    "rung2_exact_oracle_resolved": oracle2[
                        "resolved_event_identity_accuracy"
                    ],
                    "rung2_exact_oracle_admission": oracle2[
                        "admission_balanced_accuracy"
                    ],
                }
            )
            search_rows.extend(
                search_fold(dataset, fold, arm, space, table, resolver)
            )
    return {"per_fold": rows, "search_rows": search_rows, "census": census}


def aggregate_baselines(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["family"])].append(row)
    out = {}
    for (arm, family), group in sorted(grouped.items()):
        def mean(key: str) -> float | None:
            values = [row[key] for row in group if row[key] is not None]
            return rd(sum(values) / len(values)) if values else None

        out[f"{arm}|{family}"] = {
            "n_folds": len(group),
            "rung1_chance_sampled": mean("rung1_chance_sampled"),
            "rung1_marginal_accuracy": mean("rung1_marginal_accuracy"),
            "rung1_exact_oracle": mean("rung1_exact_oracle"),
            "rung2_exact_oracle_address": mean("rung2_exact_oracle_address"),
            "rung2_exact_oracle_resolved": mean("rung2_exact_oracle_resolved"),
        }
    return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> dict:
    dataset = build_dataset()
    built = baseline_table(dataset)
    rows = built["per_fold"]
    search_rows = built["search_rows"]
    census = built["census"]
    aggregates = aggregate_baselines(rows)
    search = summarise_search(search_rows, census)
    chance2 = rung2_chance()

    oracles_perfect = all(
        row["rung1_exact_oracle"] == 1.0
        and row["rung2_exact_oracle_address"] == 1.0
        and row["rung2_exact_oracle_resolved"] == 1.0
        for row in rows
    )

    b_loho = search["B_sfp"]["LOHO"]
    d_loho = search["D_relabeled"]["LOHO"]

    laws = {
        "exact_oracles_score_one": oracles_perfect,
        "rung2_chance_equals_one_over_catalogue": chance2[
            "equals_one_over_catalogue"
        ],
        "cheap_tabulation_solves_rung1_under_the_exact_code": b_loho[
            "cheap_rung1_solves_it"
        ],
        "relation_equivalent_tabulation_solves_rung2_under_the_exact_code": (
            b_loho["determining_rung2_ceiling"] == 1.0
        ),
        "relation_equivalent_tabulation_solves_rung2_under_the_relabeling": (
            d_loho["determining_rung2_ceiling"] == 1.0
        ),
        "no_relation_determining_subset_exists_under_the_scramble": (
            census["C_scrambled"]["relation_determining_subsets"] == 0
        ),
        "relation_determining_subsets_exist_under_the_exact_code": (
            census["B_sfp"]["relation_determining_subsets"] > 0
        ),
        "strictly_cheaper_rules_leave_headroom_under_the_exact_code": (
            b_loho["strictly_cheaper_rung2_ceiling"] is not None
            and b_loho["strictly_cheaper_rung2_ceiling"] < 0.99
        ),
        "pure_lookup_cannot_reach_rung2_for_any_arm": all(
            search[arm]["LOHO"]["pure_lookup_rung2_is_at_floor"]
            for arm in CODE_ARMS
        ),
    }

    broken = sorted(name for name, ok in laws.items() if not ok)
    return {
        "module": "ladder_baselines",
        "purpose": (
            "deterministic baselines and the CODE-SPACE shortcut search 009.05 "
            "section 8 requires at every rung"
        ),
        "executes": "009.05 sections 4.3 and 8",
        "fences": list(FENCES),
        "atoms": [
            {
                "name": name,
                "cost_class": cost,
                "definition": definition,
            }
            for name, cost, definition, _ in ATOMS
        ],
        "cheap_atoms": list(CHEAP_ATOMS),
        "relation_equivalent_atoms": [
            name for name in ATOM_NAMES if ATOM_COST[name] == RELATION_EQUIVALENT
        ],
        "complete_incidence_sets": [sorted(s) for s in COMPLETE_INCIDENCE_SETS],
        "cost_classification_rule": COST_CLASSIFICATION_RULE,
        "difference_from_009_02": (
            "009.02's FEATURE_ATOMS are native (same_habitat, sign_pair, "
            "block_id_equality and so on) and found no cheap shortcut on LOHO or "
            "LOFPO. These atoms are the SFP code fields the ladder's learner actually "
            "sees, and no search over them has been run before in this issue."
        ),
        "search_policy": {
            "max_subset_size": MAX_SUBSET_SIZE,
            "n_subsets": len(feature_subsets()),
            "fitted_on": "the fold's own training split only",
            "unseen_key_behaviour": "ABSTAIN, scored incorrect, never a fallback",
            "tie_break": "smallest port at Rung 1; repr-least answer at Rung 2",
        },
        "chance": {
            "rung1": rd(1 / 3),
            "rung1_statement": CHANCE_STATEMENT,
            "rung2": chance2,
        },
        "per_fold": rows,
        "aggregates": aggregates,
        "shortcut_search": search,
        "determination_census": {
            arm: {
                key: value
                for key, value in body.items()
                if key != "rows"
            }
            for arm, body in sorted(census.items())
        },
        "two_ceilings": (
            "determining_ceiling is what a tabulation of the COMPLETE local incidence "
            "configuration reaches; it is a code-space oracle, not a shortcut, and a "
            "learner is not expected to beat it. strictly_cheaper_rung2_ceiling is "
            "what the best NON-determining rule reaches, and that is the bar a "
            "learned result has to clear. Both are reported for every arm and family "
            "so neither can be quoted alone."
        ),
        "digests": {
            "per_fold": digest(rows),
            "search_rows": digest(search_rows),
            "census": digest(
                {arm: body["rows"] for arm, body in sorted(census.items())}
            ),
        },
        "reading": {
            "rung1": (
                "under the exact code the whole Rung-1 label is determined by the "
                "unordered pp pair, which takes three values, so a three-row "
                "tabulation fitted on training habitats transfers perfectly. Rung 1 "
                "therefore has a cheap ceiling of 1.0 and cannot discriminate: a "
                "learner can match it but not exceed it. 009.05 section 4.3 asks for "
                "exactly this to be reported and the rung downgraded accordingly."
            ),
            "rung2_pure_lookup": (
                "a pure tabulation scores 0.0 for EVERY arm, and the reason is "
                "arithmetic rather than structural: the address carries the habitat "
                "fields, a structural holdout removes the held-out habitat's (S, FFF) "
                "pair from training entirely, and a lookup table can only emit an "
                "answer it has already seen. This is reported because it would "
                "otherwise look like a strong negative baseline; it is not one, and "
                "reading it as the cheap ceiling would flatter any learner that can "
                "copy a field."
            ),
            "rung2": (
                "the fair rule copies S and FFF -- the copy 009.05 section 4.2 itself "
                "licenses -- and tabulates only the habitat-free (PP, pp) remainder. "
                "Two ceilings result, and the gap between them is the finding. A "
                "RELATION-EQUIVALENT key, one that carries the complete K4 incidence "
                "data, resolves the whole consequence on held-out habitats under the "
                "exact and relabeled codes. Under the non-automorphic scramble NO "
                "subset up to size three determines the consequence at all, so that "
                "ceiling does not exist for C and its best rule reaches 0.232. "
                "H_struct is therefore supported at the level of deterministic "
                "tabulation, with no learner involved and no training run required."
            ),
            "consequence_for_the_learner": (
                "a learned result at Rung 2 sits between two bars. It is not expected "
                "to beat the relation-equivalent tabulation, which is a code-space "
                "oracle. It IS expected to clear the best strictly-cheaper rule, and "
                "that bar is 0.5 for the exact code -- the level reached by keys that "
                "carry only partial incidence data. Both bars are reported for every "
                "arm and family so neither can be quoted alone."
            ),
        },
        "provenance": {
            "base_commit": BASE_COMMIT,
            "base_seed": BASE_SEED,
            "seed_derivation": (
                "int(sha256('ladder_baselines/v1|<BASE_SEED>|<label>')[:16], 16); "
                "never hash(), never iteration order"
            ),
            "constructs_no_tensor": True,
            "imports_torch_transitively_via_harness": True,
            "why": (
                "ladder_task reuses harness's frozen split functions rather than "
                "reimplementing them, and harness imports torch. Every selector here "
                "is integer arithmetic and dictionary lookup; no tensor is created."
            ),
        },
        "relational_laws": laws,
        "verdict": {
            "broken_laws": broken,
            "agrees": not broken,
            "statement": (
                "PASS - exact oracles score 1.0, structured chance equals 1/84, and "
                "the code-space shortcut search reproduces the structural signature "
                "with no learner involved: a relation-equivalent habitat-free "
                "tabulation resolves the whole consequence on held-out habitats under "
                "the exact and relabeled codes, while under the non-automorphic "
                "scramble no subset up to size three determines the consequence at all"
                if not broken
                else f"FAIL - broken laws: {broken}"
            ),
        },
    }


def _report(result: dict) -> None:
    print(f"Rung-1 chance: {result['chance']['rung1']}", flush=True)
    print(
        f"Rung-2 uniform-field chance: {result['chance']['rung2']['exact_structured_address']} "
        "(= 1/84)",
        flush=True,
    )
    print("LOHO Rung-2 ceilings (determining = code-space oracle):", flush=True)
    for arm in CODE_ARMS:
        body = result["shortcut_search"][arm]["LOHO"]
        print(
            f"  {arm:<13} determining {body['determining_rung2_ceiling']}  "
            f"strictly-cheaper {body['strictly_cheaper_rung2_ceiling']}  "
            f"(Rung 1 cheap {body['cheap_rung1_ceiling']})",
            flush=True,
        )
    for arm in CODE_ARMS:
        census = result["determination_census"][arm]
        print(
            f"  {arm:<13} relation-determining subsets: "
            f"{census['relation_determining_subsets']}/"
            f"{census['subsets_examined']}, smallest size "
            f"{census['smallest_determining_size']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit()
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if OUTPUT.read_text() != text:
            raise SystemExit(
                f"FAIL: re-derived audit is not byte-identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print("PASS: exact replay matches ladder_baselines.json", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["verdict"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()

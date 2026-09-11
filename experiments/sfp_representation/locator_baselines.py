"""009.07 baselines and ceilings for query-relative chamber localization.

``009.07`` section 4.3 requires five reference points for the ``q*`` task, and section
4.3's last paragraph requires relation-equivalent keys to be distinguished from
strictly cheaper shortcuts *before* any accuracy is interpreted::

    chance chamber choice                       1/4
    009.06 frozen absolute-PP head              0.7158 LOHO, 0.7054 LOFPO
    exact native/FIPS shared-chamber oracle     1.000
    exact SFP endpoint-intersection circuit     1.000  (nonlearned, relation-equivalent)
    best strictly-cheaper code-space baseline   measured on train-only fits

The shortcut search is ``ladder_baselines``' search, re-aimed at ``q*``: the same 12
learner-visible code-space atoms, the same 298 subsets of size 1..3, the same
cost-classification rule, the same ``FeatureSpace``. No atom is added, so the frozen
cost rule still covers every subset and cannot be quietly widened by this turn.

Two things differ from ``ladder_baselines`` and both matter:

* ``PP = 00`` is the declared ORIGIN BLOCK, a legitimate chamber value. So the
  abstention sentinel cannot be ``0`` -- an unseen key would otherwise "abstain" onto
  a real answer and be scored correct a quarter of the time. :data:`ABSTAIN_CHAMBER`
  is ``-1`` and is never a legal chamber.
* the exact endpoint-intersection circuit is a **ceiling**, not a competing learned
  baseline. Outcome ``021.05`` already established the relation is executable; here it
  measures how much of ``q*`` each arm's representation even supports.

Builds no tensor and holds no parameter; inherits torch transitively through
``ladder_task`` -> ``harness`` and records that rather than claiming otherwise.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from folds import all_folds, structural_folds
from ladder_baselines import (
    ATOM_NAMES,
    CHEAP,
    COST_CLASSIFICATION_RULE,
    MAX_SUBSET_SIZE,
    RELATION_EQUIVALENT,
    FeatureSpace,
    build_feature_space,
    feature_subsets,
    subset_cost_class,
)
from ladder_task import CODE_ARMS, SCIENCE_ARMS, code_tables, output_table
from locator_task import (
    CHAMBER_VALUES,
    PINS,
    PRIOR_009_06,
    certified_chamber_of_block,
    chamber_actions,
    qstar_positions,
    qstar_target,
)
from sfp import ExactSfpCircuit, SfpAddress, SfpCodec, bits2
from task import Dataset, build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_locator_artifacts"
OUTPUT = ARTIFACTS / "locator_baselines.json"

BASE_COMMIT = "4520c80807b8480081f396180446880a3ff6fba1"

ROUND_DIGITS = 12

#: The abstention sentinel. It must NOT be ``0``: ``PP = 00`` is the declared origin
#: block and a real chamber value, so ``ladder_baselines.ABSTAIN = 0`` would make an
#: unseen key land on a legal answer and collect a quarter of the credit for free.
ABSTAIN_CHAMBER = -1

FENCES = (
    "Baselines read the certified q*; they never generate one.",
    "Every selector is fitted on training positions only.",
    (
        "An unseen key ABSTAINS to -1, which is not a chamber, so abstention is "
        "scored incorrect and never falls back onto a legal answer."
    ),
    (
        "The exact endpoint-intersection circuit is a nonlearned CEILING. Outcome "
        "021.05 already established the relation is executable; it does not compete "
        "with the learner and does not invalidate the task."
    ),
    (
        "shared_endpoint, pp_xor and endpoint_pair remain RELATION_EQUIVALENT under "
        "the frozen ladder_baselines cost rule. A lookup keyed on them is a ceiling, "
        "not a shortcut."
    ),
    "No atom is added, so the frozen cost-classification rule still covers every subset.",
    "algebraic zero != NONADMISSION;   0 != bottom",
)

CHANCE_STATEMENT = (
    "Chance is 1/4 exactly. The certified q* marginal is uniform at 84 of 336 per "
    "chamber, and a chamber permutation cannot change a uniform marginal, so every "
    "arm's label marginal is uniform too. Reported analytically as well as by the "
    "fitted majority-class rule, so the two can be checked against each other."
)

NATIVE_ORACLE_STATEMENT = (
    "The exact native/FIPS shared-chamber oracle scores 1.000 by construction, and "
    "saying so is not circular: q* IS task.PairRecord.shared_block, the certified "
    "block that task.label_pair obtained from Catalogue.shared_blocks, mapped to a "
    "chamber by the frozen chart. The oracle is the label source, so its accuracy is "
    "1.000 identically and it fixes the top of the scale rather than measuring "
    "anything. It is listed because 009.07 section 4.3 requires the ceiling to be "
    "named explicitly."
)


# ---------------------------------------------------------------------------
# deterministic encodings
# ---------------------------------------------------------------------------

def render(payload: dict) -> str:
    """The one serialization format used by every Issue 009 artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def rd(value: float | None) -> float | None:
    return None if value is None else round(float(value), ROUND_DIGITS)


def fraction(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

class QstarLookup:
    """Train-majority ``feature key -> q*`` table. Unseen keys abstain.

    Ties resolve to the smallest chamber so the fit is order-independent, matching
    ``ladder_baselines.Rung1Lookup``'s convention.
    """

    def __init__(self, subset: tuple[str, ...]) -> None:
        self.subset = subset
        self.table: dict[tuple[int, ...], int] | None = None
        self.keys_fitted = 0
        self.unanimous_keys = 0

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
        chamber_of_block: dict,
        action: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> "QstarLookup":
        tally: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        for position in positions:
            record = dataset.records[position]
            label = qstar_target(record, chamber_of_block, action)
            tally[space.key(self.subset, position)][label] += 1
        built = {}
        unanimous = 0
        for key, counter in tally.items():
            best = max(counter.values())
            built[key] = min(q for q, count in counter.items() if count == best)
            if len(counter) == 1:
                unanimous += 1
        self.table = built
        self.keys_fitted = len(built)
        self.unanimous_keys = unanimous
        return self

    def predict(self, space: FeatureSpace, position: int) -> int:
        if self.table is None:
            raise AssertionError("selector was not fitted")
        return self.table.get(space.key(self.subset, position), ABSTAIN_CHAMBER)


def score_qstar(
    dataset: Dataset,
    chamber_of_block: dict,
    action: tuple[int, ...],
    positions: tuple[int, ...],
    predictions: list[int],
) -> dict:
    """Exact chamber accuracy, plus how often the selector answered at all."""

    if len(positions) != len(predictions):
        raise ValueError("prediction count does not match position count")
    hits = covered = 0
    for position, predicted in zip(positions, predictions):
        record = dataset.records[position]
        if predicted != ABSTAIN_CHAMBER:
            covered += 1
        if predicted == qstar_target(record, chamber_of_block, action):
            hits += 1
    return {
        "n": len(positions),
        "accuracy": rd(fraction(hits, len(positions))),
        "coverage": rd(fraction(covered, len(positions))),
        "abstentions": len(positions) - covered,
    }


def chamber_marginal(
    dataset: Dataset,
    chamber_of_block: dict,
    action: tuple[int, ...],
    positions: tuple[int, ...],
) -> int:
    """The training-majority chamber. Ties resolve to the smallest."""

    counter = Counter(
        qstar_target(dataset.records[p], chamber_of_block, action) for p in positions
    )
    best = max(counter.values())
    return min(q for q, count in counter.items() if count == best)


def exact_intersection_oracle(
    dataset: Dataset,
    table: tuple[SfpAddress, ...],
    positions: tuple[int, ...],
) -> list[int]:
    """The nonlearned SFP endpoint-intersection circuit's answer, per position.

    ``ExactSfpCircuit.shared_block`` is Outcome ``021.05``'s Event-level admission
    test: two Events are cyclically admitted iff their endpoint block sets meet in
    exactly one vertex. Applied to the ARM's coded inputs, so for the exact and
    relabeled codes it reaches the certified ``q*`` and for the scramble it reaches
    whatever the scramble left behind -- which is the representational-support
    fraction. It abstains where the coded inputs meet in no single chamber; those
    examples are kept, per ``009.07`` section 7.
    """

    circuit = ExactSfpCircuit()
    out = []
    for position in positions:
        record = dataset.records[position]
        got = circuit.shared_block(table[record.a_index], table[record.b_index])
        out.append(ABSTAIN_CHAMBER if got is None else got)
    return out


# ---------------------------------------------------------------------------
# the determining census
# ---------------------------------------------------------------------------

def qstar_determination_census(
    dataset: Dataset,
    arm: str,
    space: FeatureSpace,
    chamber_of_block: dict,
    action: tuple[int, ...],
) -> dict:
    """Which learner-visible subsets DETERMINE ``q*`` over the whole 7056-pair pool.

    Same test ``ladder_baselines.determination_census`` applies to the local answer:
    zero ambiguity measured over every ordered pair rather than over a training
    split, so no fold can tune the verdict. A determining subset is a **ceiling**;
    everything else is strictly cheaper than the relation and is a bar the learned
    result has to clear.

    Non-admitted pairs carry ``None`` as their answer, exactly as
    ``ladder_baselines`` does for ``BOTTOM``, so a subset that cannot separate
    admission from non-admission cannot be counted as determining ``q*``.
    """

    rows = []
    for subset in feature_subsets():
        answers: dict[tuple[int, ...], set] = defaultdict(set)
        for position in range(len(dataset.records)):
            record = dataset.records[position]
            answer = (
                qstar_target(record, chamber_of_block, action)
                if record.admitted
                else None
            )
            answers[space.key(subset, position)].add(answer)
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
    cheap_determining = [
        row for row in determining if row["cost_class"] == CHEAP
    ]
    return {
        "arm": arm,
        "subsets_examined": len(rows),
        "relation_determining_subsets": len(determining),
        "strictly_cheaper_subsets": len(rows) - len(determining),
        "smallest_determining_size": min(
            (len(row["subset"]) for row in determining), default=None
        ),
        "minimal_determining_subsets": sorted(
            (row["subset"] for row in determining
             if len(row["subset"]) == min(
                 (len(r["subset"]) for r in determining), default=0
             )),
            key=lambda s: (len(s), s),
        )[:8],
        "cheap_subsets_that_determine_qstar": [
            row["subset"] for row in cheap_determining
        ],
        "any_strictly_cheaper_subset_determines_qstar": bool(cheap_determining),
        "rows": rows,
        "test": (
            "zero answer ambiguity over all 7056 ordered pairs, measured on the whole "
            "pool rather than on any training split, so the label cannot be tuned by "
            "a fold"
        ),
        "cost_rule": COST_CLASSIFICATION_RULE,
    }


# ---------------------------------------------------------------------------
# the fold search
# ---------------------------------------------------------------------------

def search_fold(
    dataset: Dataset,
    fold,
    arm: str,
    space: FeatureSpace,
    chamber_of_block: dict,
    action: tuple[int, ...],
) -> list[dict]:
    """Fit every feature subset on the fold's train split; score it held out."""

    train, test = qstar_positions(dataset, fold)
    rows = []
    for subset in feature_subsets():
        selector = QstarLookup(subset).fit(
            dataset, space, chamber_of_block, action, train
        )
        scored = score_qstar(
            dataset,
            chamber_of_block,
            action,
            test,
            [selector.predict(space, p) for p in test],
        )
        rows.append(
            {
                "arm": arm,
                "family": fold.family,
                "fold": fold.name,
                "subset": list(subset),
                "size": len(subset),
                "cost_class": selector.cost_class,
                "keys_fitted": selector.keys_fitted,
                "unanimous_keys": selector.unanimous_keys,
                "accuracy": scored["accuracy"],
                "coverage": scored["coverage"],
                "abstentions": scored["abstentions"],
            }
        )
    return rows


def summarise_search(rows: list[dict], census: dict[str, dict]) -> dict:
    """The two ceilings per arm and family, split by determining power.

    ``determining_ceiling``      what a code-space tabulation of the complete local
                                 configuration achieves. The analogue of an exact
                                 oracle, not a shortcut.
    ``strictly_cheaper_ceiling`` what the best rule that is neither determining nor
                                 relation-equivalent achieves. This is the bar a
                                 learned result must clear.

    **Two independent tests decide which bucket a subset falls in, and both are
    needed.** ``relation_determining`` is measured -- zero answer ambiguity over all
    7056 pairs. ``cost_class`` is declared from Outcome ``021.05``'s definition of the
    Event-level law, via the frozen ``ladder_baselines`` rule. They do not coincide,
    and the gap matters: ``endpoint_pair`` alone reaches ``1.000`` on the admitted-only
    Rung-1 test while *not* determining ``q*`` over the whole pool, because it cannot
    separate admission from non-admission. Bucketing on the measured test alone would
    file the relation's own data as a "strictly cheaper shortcut" and invent a ceiling
    the learner had supposedly beaten. A subset is therefore counted as a ceiling if
    **either** test flags it, and both flags are recorded per subset.
    """

    determining = {
        arm: {
            tuple(row["subset"])
            for row in body["rows"]
            if row["relation_determining"]
        }
        for arm, body in census.items()
    }
    grouped: dict[tuple[str, str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["family"], tuple(row["subset"]))].append(row)

    per_arm: dict[str, dict] = {}
    for (arm, family, subset), group in sorted(grouped.items()):
        values = [row["accuracy"] for row in group if row["accuracy"] is not None]
        cost = group[0]["cost_class"]
        is_determining = subset in determining.get(arm, set())
        entry = {
            "subset": list(subset),
            "cost_class": cost,
            "relation_determining": is_determining,
            "relation_equivalent_by_cost_rule": cost == RELATION_EQUIVALENT,
            "is_a_ceiling": is_determining or cost == RELATION_EQUIVALENT,
            "n_folds": len(group),
            "mean_accuracy": rd(sum(values) / len(values)) if values else None,
            "min_accuracy": rd(min(values)) if values else None,
            "mean_coverage": rd(
                sum(row["coverage"] for row in group) / len(group)
            ),
        }
        per_arm.setdefault(arm, {}).setdefault(family, {"subsets": []})[
            "subsets"
        ].append(entry)

    for arm, families in per_arm.items():
        for family, body in families.items():
            subsets = body["subsets"]
            scored = [row for row in subsets if row["mean_accuracy"] is not None]
            ceilings = [row for row in scored if row["is_a_ceiling"]]
            cheap = [row for row in scored if not row["is_a_ceiling"]]
            body["determining_ceiling"] = (
                max(ceilings, key=lambda row: row["mean_accuracy"])
                if ceilings
                else None
            )
            body["strictly_cheaper_ceiling"] = (
                max(cheap, key=lambda row: row["mean_accuracy"]) if cheap else None
            )
            body["n_ceiling_subsets"] = len(ceilings)
            body["n_determining_by_measurement"] = sum(
                1 for row in scored if row["relation_determining"]
            )
            body["n_relation_equivalent_by_cost_rule"] = sum(
                1 for row in scored if row["relation_equivalent_by_cost_rule"]
            )
            body["n_strictly_cheaper"] = len(cheap)
            body["best_cheap_subsets"] = [
                {
                    "subset": row["subset"],
                    "mean_accuracy": row["mean_accuracy"],
                    "min_accuracy": row["min_accuracy"],
                }
                for row in sorted(
                    cheap, key=lambda row: -row["mean_accuracy"]
                )[:5]
            ]
            body.pop("subsets")
    return {
        "per_arm": per_arm,
        "bucketing_rule": (
            "a subset is a CEILING if it determines q* with zero ambiguity over all "
            "7056 pairs OR the frozen ladder_baselines cost rule classes it "
            "relation-equivalent. Both flags are recorded per subset. The measured "
            "test alone is not enough: endpoint_pair reaches 1.000 on the "
            "admitted-only Rung-1 test without determining q* over the whole pool, "
            "because it cannot separate admission from non-admission -- filing it as "
            "a shortcut would invent a bar the learner had supposedly cleared."
        ),
        "reading": (
            "a learned result is only interesting between the strictly-cheaper "
            "ceiling and the determining ceiling. Beating the former is the minimum; "
            "the latter is what the representation itself already attains without any "
            "learner."
        ),
    }


# ---------------------------------------------------------------------------
# the reference table
# ---------------------------------------------------------------------------

def baseline_table(dataset: Dataset, tables: dict, chamber_of_block: dict) -> dict:
    """The five reference points ``009.07`` section 4.3 names, per arm and family."""

    folds = structural_folds(all_folds(dataset))
    actions = chamber_actions()
    rows = {}
    for name in SCIENCE_ARMS:
        table = output_table(name, tables)
        action = actions[name]
        per_family: dict[str, dict] = {}
        for family in ("LOHO", "LOFPO"):
            marginal_scores = []
            circuit_scores = []
            circuit_abstentions = []
            for fold in (f for f in folds if f.family == family):
                train, test = qstar_positions(dataset, fold)
                majority = chamber_marginal(dataset, chamber_of_block, action, train)
                marginal_scores.append(
                    score_qstar(
                        dataset,
                        chamber_of_block,
                        action,
                        test,
                        [majority] * len(test),
                    )["accuracy"]
                )
                circuit = score_qstar(
                    dataset,
                    chamber_of_block,
                    action,
                    test,
                    exact_intersection_oracle(dataset, table, test),
                )
                circuit_scores.append(circuit["accuracy"])
                circuit_abstentions.append(circuit["abstentions"])
            per_family[family] = {
                "chance_analytic": PINS["qstar_chance"],
                "train_majority_chamber": rd(
                    sum(marginal_scores) / len(marginal_scores)
                ),
                "exact_native_shared_chamber_oracle": 1.0,
                "exact_sfp_intersection_circuit": rd(
                    sum(circuit_scores) / len(circuit_scores)
                ),
                "exact_sfp_intersection_abstentions": sum(circuit_abstentions),
                "prior_009_06_absolute_pp_head": PRIOR_009_06[
                    "field_PP_accuracy"
                ].get(family),
                "n_folds": len(marginal_scores),
            }
        rows[name] = per_family
    return {
        "per_arm": rows,
        "chance_statement": CHANCE_STATEMENT,
        "native_oracle_statement": NATIVE_ORACLE_STATEMENT,
        "prior_009_06": dict(PRIOR_009_06),
        "circuit_role": (
            "nonlearned CEILING, not a competing learned baseline. Its value on Arm C "
            "is the representational-support fraction, which is why it is not 1.000 "
            "there."
        ),
    }


def label_marginal(dataset: Dataset, chamber_of_block: dict) -> dict:
    """The certified ``q*`` marginal, and the analytic chance it implies."""

    actions = chamber_actions()
    rows = {}
    for name in SCIENCE_ARMS:
        counter = Counter(
            qstar_target(dataset.records[p], chamber_of_block, actions[name])
            for p in dataset.admit_positions
        )
        rows[name] = {bits2(q): counter.get(q, 0) for q in CHAMBER_VALUES}
    uniform = all(len(set(row.values())) == 1 for row in rows.values())
    return {
        "per_arm": rows,
        "admitted_pairs": len(dataset.admit_positions),
        "uniform_under_every_arm": uniform,
        "chance": PINS["qstar_chance"],
        "chance_matches_uniform_marginal": uniform,
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(*, verbose: bool = True) -> dict:
    dataset = build_dataset()
    tables = code_tables(dataset)
    codec = SfpCodec()
    chamber_of_block = certified_chamber_of_block(dataset, codec)
    actions = chamber_actions()
    folds = structural_folds(all_folds(dataset))

    spaces = {
        name: build_feature_space(dataset, name, tables[name]) for name in CODE_ARMS
    }
    census = {
        name: qstar_determination_census(
            dataset, name, spaces[name], chamber_of_block, actions[name]
        )
        for name in CODE_ARMS
    }
    rows: list[dict] = []
    for name in CODE_ARMS:
        for fold in folds:
            rows.extend(
                search_fold(
                    dataset,
                    fold,
                    name,
                    spaces[name],
                    chamber_of_block,
                    actions[name],
                )
            )
        if verbose:
            print(f"  searched {name}: {len(rows)} rows", flush=True)

    summary = summarise_search(rows, census)
    table = baseline_table(dataset, tables, chamber_of_block)
    marginal = label_marginal(dataset, chamber_of_block)

    b_det = census["B_sfp"]
    c_det = census["C_scrambled"]
    d_det = census["D_relabeled"]

    laws = {
        "exact_code_has_a_determining_subset": b_det["relation_determining_subsets"] > 0,
        "relabeled_code_has_a_determining_subset": (
            d_det["relation_determining_subsets"] > 0
        ),
        "exact_and_relabeled_agree_on_determining_count": (
            b_det["relation_determining_subsets"]
            == d_det["relation_determining_subsets"]
        ),
        "scramble_determining_subsets": c_det["relation_determining_subsets"],
        "no_strictly_cheaper_subset_determines_qstar_under_the_exact_code": (
            not b_det["any_strictly_cheaper_subset_determines_qstar"]
        ),
        "exact_circuit_reaches_one_on_the_exact_code": all(
            table["per_arm"]["B_sfp"][family]["exact_sfp_intersection_circuit"] == 1.0
            for family in ("LOHO", "LOFPO")
        ),
        "exact_circuit_reaches_one_on_the_relabeled_code": all(
            table["per_arm"]["D_relabeled"][family]["exact_sfp_intersection_circuit"]
            == 1.0
            for family in ("LOHO", "LOFPO")
        ),
        "chance_is_one_quarter": marginal["chance_matches_uniform_marginal"],
        "abstention_is_not_a_chamber": ABSTAIN_CHAMBER not in CHAMBER_VALUES,
    }
    agrees = all(
        value
        for key, value in laws.items()
        if key != "scramble_determining_subsets"
    )

    return {
        "module": "locator_baselines",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "atoms": list(ATOM_NAMES),
        "n_atoms": len(ATOM_NAMES),
        "max_subset_size": MAX_SUBSET_SIZE,
        "subsets_per_arm": len(feature_subsets()),
        "atoms_reused_from": (
            "ladder_baselines.ATOMS, unchanged. No atom is added, so the frozen "
            "cost-classification rule still covers every subset and the 298-subset "
            "count is the same one 009.06 searched."
        ),
        "arms_searched": list(CODE_ARMS),
        "arm_a_excluded_from_the_search": (
            "A_native has no code of its own; ARM_OUTPUT_CONVENTION gives it the "
            "B_sfp alphabet, so its code-space atoms would be B's and the row would "
            "be a duplicate. It appears in the baseline table, not in the search."
        ),
        "abstain_sentinel": ABSTAIN_CHAMBER,
        "abstain_rationale": (
            "PP=00 is the declared origin block and a legal chamber, so "
            "ladder_baselines.ABSTAIN = 0 cannot be reused here: an unseen key would "
            "land on a real answer and collect a quarter of the credit for free"
        ),
        "cost_classes": {"cheap": CHEAP, "relation_equivalent": RELATION_EQUIVALENT},
        "label_marginal": marginal,
        "baseline_table": table,
        "determination_census": census,
        "search_summary": summary,
        "search_rows": len(rows),
        "laws": laws,
        "torch_status": (
            "no tensor is constructed and no parameter is held. torch is inherited "
            "transitively through ladder_task -> harness and recorded rather than "
            "denied."
        ),
        "digests": {
            "dataset": dataset.sha256(),
            "search_rows": digest(rows),
            "census": digest(
                {name: body["rows"] for name, body in sorted(census.items())}
            ),
        },
        "verdict": {
            "agrees": agrees,
            "statement": (
                "chance is 1/4, the exact endpoint-intersection circuit reaches 1.000 "
                "on the exact and relabeled codes, a determining code-space subset "
                "exists for both and no strictly cheaper subset determines q* under "
                "the exact code"
                if agrees
                else "at least one baseline or ceiling law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    print(
        f"{result['subsets_per_arm']} subsets x {len(result['arms_searched'])} arms; "
        f"{result['search_rows']} search rows",
        flush=True,
    )
    for name, families in sorted(result["baseline_table"]["per_arm"].items()):
        row = families["LOHO"]
        print(
            f"  {name:<14} LOHO chance {row['chance_analytic']}"
            f"  majority {row['train_majority_chamber']}"
            f"  exact-circuit {row['exact_sfp_intersection_circuit']}"
            f"  009.06-PP {row['prior_009_06_absolute_pp_head']}",
            flush=True,
        )
    for name, body in sorted(result["determination_census"].items()):
        print(
            f"  {name:<14} determining subsets {body['relation_determining_subsets']:>3}"
            f"  smallest size {body['smallest_determining_size']}"
            f"  cheap-determining {len(body['cheap_subsets_that_determine_qstar'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = audit(verbose=not args.check)
    text = render(result)
    _report(result)

    if args.check:
        if not OUTPUT.exists():
            raise SystemExit(f"FAIL: missing {OUTPUT}; run without --check first")
        if json.loads(OUTPUT.read_text()) != json.loads(text):
            raise SystemExit(
                f"FAIL: re-derived audit is not identical to {OUTPUT.name}"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
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

"""009.09 ceilings and cheap baselines for the opaque local-chart problem.

Four things are established here, all without training anything:

1. **the exact one-anchor completion ceiling** (section 7.1). The nonlearned rule
   from Gate 0, applied to every scored query of every frozen episode. If Gate 0
   passed it scores ``1.000``, and that is the number the learner is judged
   against. It is a CEILING, never a competing learned baseline.

2. **the admission-only information ceiling** (section 7.2). Two independent
   derivations of ``1/2``: the Bayes rate under a uniform prior over the two
   admission-compatible block systems, and the symmetry argument -- every admitted
   pair admits an automorphism of the admission graph that fixes both query tokens
   and swaps the two candidate thirds, so a permutation-equivariant scorer is
   *forced* to tie. The zero-anchor learner may not materially exceed this.

3. **the cheap admission-only selectors** (section 7.3). A declared family, scored
   on the same episodes: uniform over all six nodes, uniform over the four
   eligible nodes, uniform over the two admission-compatible common neighbours,
   the lowest-indexed common neighbour, the training-majority node, and the
   support-pair lookup whose scored-query coverage is zero by construction.

4. **the determination census.** Over a declared set of atoms computable from the
   supplied inputs alone, every subset is tested for whether it determines the
   certified third with zero ambiguity. Two populations are censused: the full
   atom set, and the atoms an **admission-only** observer has. The second must
   contain no determining subset -- that is the code-space form of the anti-cheating
   claim, and it is the same instrument ``009.08`` pointed at ``q*``.

The ``009.09a`` section 2 dual-anchor arm is also given its ceiling here, over the
combinatorial dual family, and labelled as a control throughout.

Constructs no tensor; inherits torch transitively through
``discovery_task -> locator_task -> ladder_task -> harness``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

from discovery_task import (
    N_TOKENS,
    SCORED_QUERIES_PER_EPISODE,
    Episode,
    adjacency,
    anchor_third,
    automorphisms,
    build_episodes,
    build_local_problems,
    compatible_block_systems,
    fraction,
    habitats_of_fold,
    pin,
    rd,
    render,
)
from folds import all_folds, structural_folds
from sfp import SfpCodec
from task import build_dataset, digest

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "009_discovery_artifacts"
OUTPUT = ARTIFACTS / "discovery_baselines.json"

BASE_COMMIT = "7a37f58444901c32a6e750e617d8d4d82b9f6202"

#: Chance rates ``009.09`` section 7.3 asks for, declared before any measurement.
UNIFORM_ALL_NODES = 1.0 / N_TOKENS
UNIFORM_ELIGIBLE_NODES = 1.0 / (N_TOKENS - 2)
UNIFORM_COMMON_NEIGHBOURS = 0.5

#: ``009.09`` section 7.4: the supplied-SFP reference, quoted for context only.
SUPPLIED_SFP_REFERENCE = {
    "turn": "009.08",
    "metric": "positive_forced_third_exact_accuracy",
    "arm": "B_sfp",
    "family": "LOHO",
    "value": 0.9688,
    "status": (
        "CONTEXTUAL REFERENCE ONLY. 009.08 was handed the exact finite SFP "
        "representation; 009.09 withdraws it and supplies relational evidence "
        "instead. The two arms observe different things and are not capacity "
        "matched. The discovery result is judged against its own exact 1.000 "
        "one-anchor ceiling, per 009.09 section 7.4."
    ),
}

#: The declared atoms. Every one is computable from what the learner is given --
#: the admission matrix, the anchor marks and the query marks -- so a determining
#: subset is a statement about the SUPPLIED information, not about a hidden field.
ATOMS = (
    "candidate_is_a_query_node",
    "candidate_adjacent_to_both_query_nodes",
    "candidate_adjacent_to_exactly_one_query_node",
    "candidate_is_an_anchor_node",
    "anchor_nodes_among_the_query_pair",
    "anchor_nodes_adjacent_to_the_candidate",
    "candidate_admitted_degree",
)

#: The subset of atoms an admission-only observer has. Anything mentioning the
#: anchor is removed.
ADMISSION_ONLY_ATOMS = (
    "candidate_is_a_query_node",
    "candidate_adjacent_to_both_query_nodes",
    "candidate_adjacent_to_exactly_one_query_node",
    "candidate_admitted_degree",
)

#: Declared abstention value. It is not a node, so it can never be scored correct.
ABSTAIN = -1

FENCES = (
    "The exact one-anchor completion is a NONLEARNED CEILING, not a competing "
    "learned baseline. No part of it appears on any learned path.",
    "The admission-only ceiling of 1/2 is derived twice -- Bayes mass and the "
    "candidate-swapping automorphism -- and the zero-anchor learner may not "
    "materially exceed it.",
    "The dual-family thirds are COMBINATORIAL, not certified FIPS labels. The "
    "dual-anchor arm is a control and is never quoted as a result.",
    "009.08's 0.9688 is a contextual reference on a different observation set, not "
    "a capacity-matched arm.",
    "Abstention is -1, which is not a node, so an abstaining selector scores zero "
    "rather than borrowing a chance hit.",
    "The lowest-indexed-common-neighbour selector deliberately uses the OPAQUE "
    "name. It is a leakage probe: under fresh relabelling it must sit at chance "
    "among the two candidates.",
)


# ---------------------------------------------------------------------------
# atoms
# ---------------------------------------------------------------------------

def atom_values(
    episode: Episode, query: tuple[int, int], candidate: int
) -> dict[str, int]:
    """The declared atoms for one ``(episode, query, candidate)`` row."""

    nbrs = adjacency(episode.admit_nodes)
    a, b = query
    anchor = set(episode.anchor_nodes)
    neighbours = nbrs[candidate]
    touching = len({a, b} & neighbours)
    return {
        "candidate_is_a_query_node": int(candidate in (a, b)),
        "candidate_adjacent_to_both_query_nodes": int(touching == 2),
        "candidate_adjacent_to_exactly_one_query_node": int(touching == 1),
        "candidate_is_an_anchor_node": int(candidate in anchor),
        "anchor_nodes_among_the_query_pair": len({a, b} & anchor),
        "anchor_nodes_adjacent_to_the_candidate": len(anchor & neighbours),
        "candidate_admitted_degree": len(neighbours),
    }


def atom_rows(
    episodes: tuple[Episode, ...], *, family: str = "certified"
) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    """One row per ``(episode, scored query, candidate node)``.

    Returned as ``((atom_tuple, label), ...)`` where ``label`` is 1 on the certified
    (or, for the dual control, the combinatorial) third.
    """

    rows = []
    for episode in episodes:
        thirds = dict(
            episode.certified_third_nodes
            if family == "certified"
            else episode.dual_third_nodes
        )
        for query in episode.scored_queries:
            target = thirds[query]
            for candidate in range(N_TOKENS):
                values = atom_values(episode, query, candidate)
                rows.append(
                    (
                        tuple(values[name] for name in ATOMS),
                        int(candidate == target),
                        episode.episode_id,
                        query,
                        candidate,
                    )
                )
    return tuple(rows)


def determination_census(rows, atoms: tuple[str, ...]) -> dict:
    """Every subset of ``atoms``, tested for zero-ambiguity determination.

    A subset determines the answer iff no atom tuple ever appears both on a
    certified third and on a non-third. When that holds, exactly one node per query
    carries a positive tuple, so the selector is well defined and exact. When it
    fails the selector abstains rather than guessing.
    """

    index = {name: ATOMS.index(name) for name in atoms}
    queries = {}
    for values, label, episode_id, query, candidate in rows:
        queries.setdefault((episode_id, query), []).append((values, label, candidate))

    results = []
    for size in range(len(atoms) + 1):
        for subset in itertools.combinations(sorted(atoms), size):
            keys = [index[name] for name in subset]
            positive: set[tuple[int, ...]] = set()
            negative: set[tuple[int, ...]] = set()
            for values, label, *_ in rows:
                key = tuple(values[k] for k in keys)
                (positive if label else negative).add(key)
            determines = not (positive & negative)

            hits = 0
            abstentions = 0
            for group in queries.values():
                selected = [
                    candidate
                    for values, _label, candidate in group
                    if tuple(values[k] for k in keys) in positive
                ]
                if len(selected) != 1:
                    abstentions += 1
                    chosen = ABSTAIN
                else:
                    chosen = selected[0]
                target = next(c for _v, label, c in group if label)
                hits += int(chosen == target)
            results.append(
                {
                    "atoms": list(subset),
                    "size": size,
                    "determines": determines,
                    "accuracy": rd(fraction(hits, len(queries))),
                    "abstentions": abstentions,
                    "abstention_rate": rd(fraction(abstentions, len(queries))),
                }
            )

    determining = [row for row in results if row["determines"]]
    smallest = min((row["size"] for row in determining), default=None)
    best = max(results, key=lambda row: (row["accuracy"] or 0.0, -row["size"]))
    return {
        "atom_pool": list(atoms),
        "subsets_examined": len(results),
        "queries_scored": len(queries),
        "determining_subsets": len(determining),
        "smallest_determining_size": smallest,
        "smallest_determining_subsets": [
            row["atoms"] for row in determining if row["size"] == smallest
        ],
        "best_accuracy": best["accuracy"],
        "best_subset": best["atoms"],
        "per_subset": results,
        "digest": digest(
            [[row["atoms"], row["determines"], row["accuracy"]] for row in results]
        ),
    }


# ---------------------------------------------------------------------------
# ceilings
# ---------------------------------------------------------------------------

def exact_one_anchor_ceiling(
    episodes: tuple[Episode, ...], *, family: str = "certified"
) -> dict:
    """Section 7.1: the nonlearned exact completion, over the frozen episodes."""

    hits = 0
    unresolved = 0
    total = 0
    all_pairs_hits = 0
    all_pairs_total = 0
    for episode in episodes:
        thirds = dict(
            episode.certified_third_nodes
            if family == "certified"
            else episode.dual_third_nodes
        )
        for query in episode.scored_queries:
            got = anchor_third(episode.admit_nodes, episode.anchor_nodes, query)
            total += 1
            if got is None:
                unresolved += 1
            hits += int(got == thirds[query])
        for query in episode.admit_nodes:
            got = anchor_third(episode.admit_nodes, episode.anchor_nodes, query)
            all_pairs_total += 1
            all_pairs_hits += int(got == thirds[query])
    return {
        "role": "NONLEARNED CEILING",
        "family": family,
        "episodes": len(episodes),
        "scored_queries": total,
        "forced_third_accuracy": rd(fraction(hits, total)),
        "unresolved": unresolved,
        "all_admitted_pairs": all_pairs_total,
        "all_admitted_pairs_accuracy": rd(
            fraction(all_pairs_hits, all_pairs_total)
        ),
        "rule": (
            "among the common admitted neighbours of the query pair, the one whose "
            "triple meets the anchor in an ODD number of tokens"
        ),
        "status": (
            "a ceiling, never a competing learned baseline; it appears on no learned "
            "path and is quoted only as the bound the learner is measured against"
        ),
    }


def admission_only_ceiling(episodes: tuple[Episode, ...]) -> dict:
    """Section 7.2: the exact zero-anchor information ceiling, derived twice."""

    dataset = build_dataset()
    codec = SfpCodec()
    problems = {p.key: p for p in build_local_problems(dataset, codec)}

    bayes_hits = 0
    bayes_total = 0
    swap_witness_total = 0
    swap_witness_found = 0
    candidate_counts = set()

    for episode in episodes:
        problem = problems[episode.habitat]
        inverse = episode.inverse()
        systems = compatible_block_systems(problem.admit)
        nbrs = adjacency(episode.admit_nodes)
        auts = automorphisms(episode.admit_nodes)
        for query in episode.scored_queries:
            a, b = query
            common = sorted(nbrs[a] & nbrs[b])
            candidate_counts.add(len(common))

            # Bayes mass: the best fixed answer under a uniform prior over the
            # admission-compatible systems.
            tally: dict[int, int] = {}
            for system in systems:
                wanted = {inverse[a], inverse[b]}
                triple = [t for t in system if wanted <= set(t)]
                if len(triple) != 1:
                    continue
                third = episode.perm[next(iter(set(triple[0]) - wanted))]
                tally[third] = tally.get(third, 0) + 1
            bayes_hits += max(tally.values(), default=0)
            bayes_total += sum(tally.values())

            # Symmetry: an automorphism fixing both query nodes and swapping the
            # two candidates forces any equivariant scorer to tie.
            swap_witness_total += 1
            if len(common) == 2:
                c1, c2 = common
                if any(
                    perm[a] == a and perm[b] == b and perm[c1] == c2 and perm[c2] == c1
                    for perm in auts
                ):
                    swap_witness_found += 1

    bayes = rd(fraction(bayes_hits, bayes_total))
    return {
        "role": "EXACT INFORMATION CEILING for the zero-anchor arm",
        "episodes": len(episodes),
        "scored_queries": swap_witness_total,
        "common_neighbour_counts_observed": sorted(candidate_counts),
        "bayes_forced_third_ceiling": bayes,
        "equivariance_witness_rate": rd(
            fraction(swap_witness_found, swap_witness_total)
        ),
        "every_query_has_a_candidate_swapping_automorphism": (
            swap_witness_found == swap_witness_total
        ),
        "ceiling": bayes,
        "two_derivations_agree": bayes == UNIFORM_COMMON_NEIGHBOURS,
        "derivation_1": (
            "uniform prior over the two admission-compatible block systems: they "
            "disagree on every admitted pair, so any fixed answer captures exactly "
            "half the posterior mass"
        ),
        "derivation_2": (
            "for every scored query there is an automorphism of the admission graph "
            "fixing both query nodes and exchanging the two candidate thirds. A "
            "permutation-equivariant scorer must therefore assign them equal scores, "
            "so its decision is a tie-break and cannot beat 1/2 under fresh "
            "relabelling"
        ),
        "consequence": (
            "a zero-anchor learner materially above this number is evidence of "
            "leakage or stable-token semantics, not of learning"
        ),
    }


# ---------------------------------------------------------------------------
# cheap selectors
# ---------------------------------------------------------------------------

def cheap_selectors(
    episodes: tuple[Episode, ...], train: tuple[Episode, ...]
) -> dict:
    """Section 7.3: the declared cheap family, scored on the same episodes."""

    majority_tally: dict[int, int] = {}
    for episode in train:
        thirds = dict(episode.certified_third_nodes)
        for query in episode.scored_queries:
            node = thirds[query]
            majority_tally[node] = majority_tally.get(node, 0) + 1
    majority = (
        max(sorted(majority_tally), key=lambda node: majority_tally[node])
        if majority_tally
        else ABSTAIN
    )

    totals = {
        "lowest_common_neighbour": 0,
        "highest_common_neighbour": 0,
        "train_majority_node": 0,
        "support_pair_lookup": 0,
        "lowest_eligible_node": 0,
    }
    coverage = {"support_pair_lookup": 0}
    scored = 0
    for episode in episodes:
        thirds = dict(episode.certified_third_nodes)
        nbrs = adjacency(episode.admit_nodes)
        support = {
            query: thirds[query] for query in episode.support_queries
        }
        for query in episode.scored_queries:
            a, b = query
            target = thirds[query]
            common = sorted(nbrs[a] & nbrs[b])
            eligible = sorted(set(range(N_TOKENS)) - {a, b})
            scored += 1
            totals["lowest_common_neighbour"] += int(common[0] == target)
            totals["highest_common_neighbour"] += int(common[-1] == target)
            totals["train_majority_node"] += int(majority == target)
            totals["lowest_eligible_node"] += int(eligible[0] == target)
            looked_up = support.get(query, ABSTAIN)
            if query in support:
                coverage["support_pair_lookup"] += 1
            totals["support_pair_lookup"] += int(looked_up == target)

    return {
        "scored_queries": scored,
        "declared_chance": {
            "uniform_all_nodes": rd(UNIFORM_ALL_NODES),
            "uniform_eligible_nodes": rd(UNIFORM_ELIGIBLE_NODES),
            "uniform_common_neighbours": rd(UNIFORM_COMMON_NEIGHBOURS),
        },
        "measured": {
            name: rd(fraction(hits, scored)) for name, hits in sorted(totals.items())
        },
        "train_majority_node": majority,
        "support_pair_lookup_coverage": rd(
            fraction(coverage["support_pair_lookup"], scored)
        ),
        "support_pair_coverage_is_zero_by_construction": coverage[
            "support_pair_lookup"
        ]
        == 0,
        "best_cheap_admission_only_rule": rd(UNIFORM_COMMON_NEIGHBOURS),
        "opaque_name_tie_break_note": (
            "lowest_common_neighbour and highest_common_neighbour break the tie by "
            "OPAQUE NODE NAME. Under the frozen fresh relabelling they sit at chance "
            "among the two candidates, which is the measurement that shows an opaque "
            "name carries no structure -- it is not a rule the learner could use."
        ),
        "symmetry_respecting_bound": (
            "009.09 section 7.3 expects a symmetry-respecting admission-only "
            "selector to be bounded at 1/2 because the query pair has two "
            "admission-compatible continuations. Verified, not assumed: the observed "
            "common-neighbour count is 2 for every scored query."
        ),
    }


# ---------------------------------------------------------------------------
# per-fold view
# ---------------------------------------------------------------------------

def per_fold_ceilings(
    dataset, folds, episodes: tuple[Episode, ...]
) -> dict:
    """The ceilings restricted to each fold's test episodes.

    Placed on the learner's own scale so the result turn can compare like with
    like rather than quoting a global number against a per-fold one.
    """

    rows = []
    for fold in folds:
        held = set(habitats_of_fold(dataset, fold))
        test = tuple(e for e in episodes if e.habitat in held)
        exact = exact_one_anchor_ceiling(test)
        rows.append(
            {
                "family": fold.family,
                "name": fold.name,
                "test_episodes": len(test),
                "scored_queries": exact["scored_queries"],
                "one_anchor_ceiling": exact["forced_third_accuracy"],
                "all_pairs_ceiling": exact["all_admitted_pairs_accuracy"],
            }
        )
    return {
        "per_fold": rows,
        "one_anchor_ceiling_is_one_on_every_fold": all(
            row["one_anchor_ceiling"] == 1.0 for row in rows
        ),
        "all_pairs_ceiling_is_one_on_every_fold": all(
            row["all_pairs_ceiling"] == 1.0 for row in rows
        ),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(*, verbose: bool = True) -> dict:
    dataset = build_dataset()
    codec = SfpCodec()
    problems = build_local_problems(dataset, codec)
    folds = structural_folds(all_folds(dataset))
    episodes = build_episodes(problems, namespace="primary")
    dual = build_episodes(
        problems, namespace="dual_control", family="combinatorial_dual"
    )

    if verbose:
        print("--- ceilings ---", flush=True)
    exact = exact_one_anchor_ceiling(episodes)
    admission = admission_only_ceiling(episodes)
    dual_ceiling = exact_one_anchor_ceiling(dual, family="combinatorial_dual")
    cheap = cheap_selectors(episodes, episodes)
    folded = per_fold_ceilings(dataset, folds, episodes)

    if verbose:
        print("--- determination census ---", flush=True)
    rows = atom_rows(episodes)
    full_census = determination_census(rows, ATOMS)
    admission_census = determination_census(rows, ADMISSION_ONLY_ATOMS)

    laws = {
        "one_anchor_ceiling_is_one": exact["forced_third_accuracy"] == 1.0,
        "one_anchor_all_pairs_ceiling_is_one": exact[
            "all_admitted_pairs_accuracy"
        ]
        == 1.0,
        "one_anchor_never_unresolved": exact["unresolved"] == 0,
        "admission_only_ceiling_is_one_half": admission["ceiling"]
        == UNIFORM_COMMON_NEIGHBOURS,
        "admission_only_two_derivations_agree": admission["two_derivations_agree"],
        "every_query_has_a_candidate_swapping_automorphism": admission[
            "every_query_has_a_candidate_swapping_automorphism"
        ],
        "every_query_has_exactly_two_common_neighbours": admission[
            "common_neighbour_counts_observed"
        ]
        == [2],
        "support_pair_lookup_covers_no_scored_query": cheap[
            "support_pair_coverage_is_zero_by_construction"
        ],
        "one_anchor_ceiling_is_one_on_every_fold": folded[
            "one_anchor_ceiling_is_one_on_every_fold"
        ],
        "some_supplied_atom_subset_determines_the_third": (
            full_census["determining_subsets"] > 0
        ),
        "no_admission_only_atom_subset_determines_the_third": (
            admission_census["determining_subsets"] == 0
        ),
        "admission_only_census_cannot_beat_one_half": (
            (admission_census["best_accuracy"] or 0.0) <= UNIFORM_COMMON_NEIGHBOURS
        ),
        "dual_anchor_control_ceiling_is_one": dual_ceiling[
            "forced_third_accuracy"
        ]
        == 1.0,
        "scored_queries_match_the_manifest": exact["scored_queries"]
        == len(episodes) * SCORED_QUERIES_PER_EPISODE,
    }

    agrees = all(laws.values())
    return {
        "module": "discovery_baselines",
        "base_commit": BASE_COMMIT,
        "fences": list(FENCES),
        "pins": {
            "uniform_all_nodes": pin(
                rd(UNIFORM_ALL_NODES),
                cheap["declared_chance"]["uniform_all_nodes"],
                "1/6",
            ),
            "uniform_eligible_nodes": pin(
                rd(UNIFORM_ELIGIBLE_NODES),
                cheap["declared_chance"]["uniform_eligible_nodes"],
                "1/4, the four non-query nodes",
            ),
            "admission_only_ceiling": pin(
                UNIFORM_COMMON_NEIGHBOURS,
                admission["ceiling"],
                "two independent derivations",
            ),
            "one_anchor_ceiling": pin(
                1.0, exact["forced_third_accuracy"], "exact nonlearned completion"
            ),
        },
        "one_anchor_ceiling": exact,
        "admission_only_ceiling": admission,
        "dual_anchor_control_ceiling": dual_ceiling,
        "cheap_selectors": cheap,
        "per_fold": folded,
        "determination_census": {
            "full_atom_pool": full_census,
            "admission_only_atom_pool": admission_census,
            "reading": (
                "a determining subset over the SUPPLIED atoms is expected and is not "
                "a leak: it is the statement that the information the task hands the "
                "learner is sufficient. The load-bearing line is the second census: "
                "with the anchor atoms removed, NO subset determines the certified "
                "third, and none exceeds 1/2."
            ),
        },
        "supplied_sfp_reference": dict(SUPPLIED_SFP_REFERENCE),
        "torch": (
            "no tensor is constructed here; torch is inherited transitively through "
            "discovery_task -> locator_task -> ladder_task -> harness"
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "laws": laws,
        "verdict": {
            "agrees": agrees,
            "statement": (
                "the exact one-anchor completion scores 1.000 on every fold, the "
                "admission-only ceiling is 1/2 under two independent derivations, "
                "every scored query has exactly two admission-compatible "
                "continuations and a candidate-swapping automorphism, the "
                "support-pair lookup covers no scored query, and no admission-only "
                "atom subset determines the certified third"
                if agrees
                else "at least one ceiling or baseline law FAILED"
            ),
        },
    }


def _report(result: dict) -> None:
    exact = result["one_anchor_ceiling"]
    admission = result["admission_only_ceiling"]
    cheap = result["cheap_selectors"]
    print(
        f"  one-anchor exact ceiling      {exact['forced_third_accuracy']} "
        f"({exact['scored_queries']} scored queries)",
        flush=True,
    )
    print(f"  admission-only ceiling        {admission['ceiling']}", flush=True)
    print("  cheap selectors:", flush=True)
    for name, value in sorted(cheap["measured"].items()):
        print(f"    {name:<34} {value}", flush=True)
    for name, value in sorted(cheap["declared_chance"].items()):
        print(f"    {name:<34} {value}  (declared)", flush=True)
    full = result["determination_census"]["full_atom_pool"]
    limited = result["determination_census"]["admission_only_atom_pool"]
    print(
        f"  census: supplied atoms -> {full['determining_subsets']} determining "
        f"subsets, smallest size {full['smallest_determining_size']} "
        f"{full['smallest_determining_subsets'][:1]}",
        flush=True,
    )
    print(
        f"  census: admission-only -> {limited['determining_subsets']} determining "
        f"subsets, best accuracy {limited['best_accuracy']}",
        flush=True,
    )
    for name, value in sorted(result["laws"].items()):
        if not value:
            print(f"  FAIL law {name}", flush=True)


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
        expected = json.loads(OUTPUT.read_text())
        observed = json.loads(text)
        for key in ("started_at", "host"):
            expected.pop(key, None)
            observed.pop(key, None)
        if expected != observed:
            raise SystemExit(
                f"FAIL: re-derived baselines are not identical to {OUTPUT.name} "
                "(started_at/host excluded)"
            )
        if not result["verdict"]["agrees"]:
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: exact replay matches {OUTPUT.name}", flush=True)
        print(result["verdict"]["statement"], flush=True)
    else:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)
        if not result["verdict"]["agrees"]:
            print(json.dumps(result["laws"], indent=2, sort_keys=True), flush=True)
            raise SystemExit("FAIL: " + result["verdict"]["statement"])
        print(f"PASS: wrote {OUTPUT.relative_to(ROOT.parent.parent)}", flush=True)
        print(result["verdict"]["statement"], flush=True)


if __name__ == "__main__":
    main()

"""Audit held-out consequence identifiability at the Issue-014 boundary.

This program reads one input only: the declared blind information-boundary
packet. It uses no hidden labels, project relation data, or stochastic search.
It writes the exact machine report and, optionally, its deterministic Markdown
rendering.
"""

# Long Markdown paragraphs are assembled from adjacent string literals.
# ruff: noqa: ISC004

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_PACKET_SCHEMA = "occurrence.gpt.015.identifiability-input.v1"
REPORT_SCHEMA = "occurrence.gpt.015.identifiability-audit.v1"
TASK_REVISION = "cdc3b67e9ca3c3082616ad0f5328587d926efb8872ff90dcd0e3939c448dfb98"
EXPECTED_COUNTS = {
    "universe": 84,
    "train": 48,
    "role_test": 96,
    "novel_test": 24,
}
SPLITS = ("role_test", "novel_test")

Pair = tuple[int, int]
Block = tuple[int, int, int]


class AuditError(ValueError):
    """Raised when the packet or a claimed audit witness is invalid."""


def require(condition: bool, message: str) -> None:
    """Raise an auditable validation error when a required fact is false."""
    if not condition:
        raise AuditError(message)


def canonical_pair(values: Sequence[int], *, context: str) -> Pair:
    """Validate and canonicalize a two-token unordered pair."""
    require(
        isinstance(values, (list, tuple)) and len(values) == 2,
        f"{context} must contain exactly two tokens",
    )
    a, b = values
    require(
        isinstance(a, int) and not isinstance(a, bool),
        f"{context}[0] must be an integer",
    )
    require(
        isinstance(b, int) and not isinstance(b, bool),
        f"{context}[1] must be an integer",
    )
    require(a != b, f"{context} must contain two distinct tokens")
    return (a, b) if a < b else (b, a)


def block_pairs(block: Block) -> tuple[Pair, Pair, Pair]:
    """Return the three canonical pairs of a sorted three-token block."""
    a, b, c = block
    return ((a, b), (a, c), (b, c))


def third_token(block: Block, pair: Pair) -> int:
    """Return the unique token in block that is absent from pair."""
    remainder = set(block).difference(pair)
    require(len(remainder) == 1, f"{pair} is not a pair of block {block}")
    return next(iter(remainder))


def semantic_packet_digest(
    packet_schema: str,
    universe: Sequence[int],
    train: Sequence[tuple[Pair, int]],
    heldout: Mapping[str, Sequence[Pair]],
) -> str:
    """Hash the normalized scientific packet content, not file formatting."""
    normalized = {
        "schema": packet_schema,
        "token_universe": list(universe),
        "train": [{"query": list(query), "target": target} for query, target in train],
        "heldout_queries": {
            split: [list(query) for query in heldout[split]] for split in SPLITS
        },
    }
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_packet(path: Path) -> dict[str, Any]:
    """Load only the blind packet and validate its finite boundary."""
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load blind packet {path}: {exc}") from exc

    require(isinstance(packet, dict), "packet root must be an object")
    require(
        packet.get("schema") == EXPECTED_PACKET_SCHEMA,
        f"packet schema must be {EXPECTED_PACKET_SCHEMA!r}",
    )

    universe_section = packet.get("token_universe")
    require(isinstance(universe_section, dict), "token_universe must be an object")
    universe = universe_section.get("ids")
    require(isinstance(universe, list), "token_universe.ids must be a list")
    require(
        universe_section.get("count") == EXPECTED_COUNTS["universe"],
        "declared universe count must be 84",
    )
    require(
        universe == list(range(EXPECTED_COUNTS["universe"])),
        "token universe must be exactly [0, ..., 83]",
    )
    universe_set = set(universe)

    raw_train = packet.get("train")
    require(isinstance(raw_train, list), "train must be a list")
    require(
        len(raw_train) == EXPECTED_COUNTS["train"],
        "packet must contain exactly 48 TRAIN observations",
    )
    train: list[tuple[Pair, int]] = []
    train_map: dict[Pair, int] = {}
    for index, record in enumerate(raw_train):
        require(isinstance(record, dict), f"train[{index}] must be an object")
        query = canonical_pair(record.get("query"), context=f"train[{index}].query")
        target = record.get("target")
        require(
            isinstance(target, int) and not isinstance(target, bool),
            f"train[{index}].target must be an integer",
        )
        require(query[0] in universe_set and query[1] in universe_set, "TRAIN query")
        require(target in universe_set, f"train[{index}].target is outside U")
        require(query not in train_map, f"duplicate TRAIN query {query}")
        train_map[query] = target
        train.append((query, target))

    raw_heldout = packet.get("heldout_queries")
    require(isinstance(raw_heldout, dict), "heldout_queries must be an object")
    heldout: dict[str, list[Pair]] = {}
    for split in SPLITS:
        raw_queries = raw_heldout.get(split)
        require(isinstance(raw_queries, list), f"{split} must be a list")
        require(
            len(raw_queries) == EXPECTED_COUNTS[split],
            f"{split} has the wrong query count",
        )
        queries: list[Pair] = []
        seen: set[Pair] = set()
        for index, raw_query in enumerate(raw_queries):
            query = canonical_pair(raw_query, context=f"{split}[{index}]")
            require(
                query[0] in universe_set and query[1] in universe_set,
                f"{split}[{index}] contains a token outside U",
            )
            require(query not in seen, f"duplicate {split} query {query}")
            seen.add(query)
            queries.append(query)
        heldout[split] = queries

    train_pairs = set(train_map)
    role_pairs = set(heldout["role_test"])
    novel_pairs = set(heldout["novel_test"])
    require(train_pairs.isdisjoint(role_pairs), "TRAIN and ROLE_TEST overlap")
    require(train_pairs.isdisjoint(novel_pairs), "TRAIN and NOVEL_TEST overlap")
    require(role_pairs.isdisjoint(novel_pairs), "ROLE_TEST and NOVEL_TEST overlap")

    declared_sizes = packet.get("sizes")
    require(isinstance(declared_sizes, dict), "sizes must be an object")
    require(declared_sizes.get("train") == len(train), "sizes.train mismatch")
    require(
        declared_sizes.get("role_test_queries") == len(heldout["role_test"]),
        "sizes.role_test_queries mismatch",
    )
    require(
        declared_sizes.get("novel_test_queries") == len(heldout["novel_test"]),
        "sizes.novel_test_queries mismatch",
    )

    mandatory_blocks: list[Block] = []
    for index, (query, target) in enumerate(train):
        block = tuple(sorted((*query, target)))
        require(
            len(set(block)) == 3,
            f"TRAIN observation {index} does not denote three distinct Events",
        )
        mandatory_blocks.append(block)

    require(
        len(set(mandatory_blocks)) == len(mandatory_blocks),
        "TRAIN observations must induce 48 distinct mandatory blocks",
    )
    mandatory_pair_to_completion: dict[Pair, int] = {}
    mandatory_pair_to_block: dict[Pair, Block] = {}
    for block in mandatory_blocks:
        for pair in block_pairs(block):
            completion = third_token(block, pair)
            require(
                pair not in mandatory_pair_to_completion,
                f"mandatory blocks reuse pair {pair}",
            )
            mandatory_pair_to_completion[pair] = completion
            mandatory_pair_to_block[pair] = block

    return {
        "packet_schema": packet["schema"],
        "universe": tuple(universe),
        "train": tuple(train),
        "train_map": train_map,
        "heldout": {split: tuple(heldout[split]) for split in SPLITS},
        "mandatory_blocks": tuple(mandatory_blocks),
        "mandatory_pair_to_completion": mandatory_pair_to_completion,
        "mandatory_pair_to_block": mandatory_pair_to_block,
        "packet_semantic_sha256": semantic_packet_digest(
            packet["schema"], universe, train, heldout
        ),
    }


def validate_literal_model(
    universe: Sequence[int], train_map: Mapping[Pair, int], default: int
) -> None:
    """Check the compact total-function countermodel over all U choose 2 pairs."""
    universe_set = set(universe)
    require(default in universe_set, "literal model default must be in U")
    checked = 0
    for pair in itertools.combinations(universe, 2):
        value = train_map.get(pair, default)
        require(value in universe_set, f"literal model leaves U at {pair}")
        if pair in train_map:
            require(value == train_map[pair], f"literal model violates TRAIN at {pair}")
        checked += 1
    require(
        checked == len(universe) * (len(universe) - 1) // 2,
        "literal model did not cover the complete unordered-pair domain",
    )


def validate_block_set(
    blocks: Iterable[Block],
    universe: Sequence[int],
    required_blocks: Sequence[Block],
) -> dict[Pair, int]:
    """Check three-Event blocks, mandatory inclusion, and pair uniqueness."""
    universe_set = set(universe)
    normalized: list[Block] = []
    for raw_block in blocks:
        block = tuple(sorted(raw_block))
        require(len(block) == 3, f"block {raw_block} must have three entries")
        require(len(set(block)) == 3, f"block {raw_block} repeats an Event")
        require(set(block) <= universe_set, f"block {raw_block} leaves U")
        normalized.append(block)

    require(
        len(normalized) == len(set(normalized)),
        "a block-set witness contains duplicate blocks",
    )
    normalized_set = set(normalized)
    require(
        set(required_blocks) <= normalized_set,
        "a block-set witness omits a mandatory TRAIN block",
    )

    completions: dict[Pair, int] = {}
    for block in normalized:
        for pair in block_pairs(block):
            require(pair not in completions, f"block-set witness reuses pair {pair}")
            completions[pair] = third_token(block, pair)
    return completions


def classify(candidate_set: Sequence[int]) -> str:
    """Apply the task's exact forced-versus-underdetermined definition."""
    return "forced" if len(candidate_set) == 1 else "underdetermined"


def summarize(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize classifications and candidate-set sizes deterministically."""
    histogram = Counter(entry["candidate_set_size"] for entry in entries)
    forced = sum(entry["classification"] == "forced" for entry in entries)
    underdetermined = len(entries) - forced
    return {
        "query_count": len(entries),
        "forced": forced,
        "underdetermined": underdetermined,
        "candidate_set_size_histogram": {
            str(size): histogram[size] for size in sorted(histogram)
        },
    }


def audit_a(boundary: Mapping[str, Any]) -> dict[str, Any]:
    """Compute literal learner-visible identifiability and check witnesses."""
    universe: tuple[int, ...] = boundary["universe"]
    train_map: Mapping[Pair, int] = boundary["train_map"]
    defaults = universe[:2]
    for default in defaults:
        validate_literal_model(universe, train_map, default)

    query_results: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        entries: list[dict[str, Any]] = []
        for query in boundary["heldout"][split]:
            candidates = [train_map[query]] if query in train_map else list(universe)
            entry: dict[str, Any] = {
                "query": list(query),
                "classification": classify(candidates),
                "candidate_set_size": len(candidates),
                "candidate_set": candidates,
            }
            if len(candidates) > 1:
                answers = [
                    train_map.get(query, defaults[0]),
                    train_map.get(query, defaults[1]),
                ]
                require(
                    answers[0] != answers[1],
                    f"Audit-A countermodels do not disagree at {query}",
                )
                require(
                    all(answer in candidates for answer in answers),
                    f"Audit-A witness answer is not in C_A({query})",
                )
                entry["witness"] = {
                    "countermodels": ["A_default_0", "A_default_1"],
                    "answers": answers,
                }
            entries.append(entry)
        query_results[split] = entries

    all_entries = query_results["role_test"] + query_results["novel_test"]
    require(
        all(entry["candidate_set"] == list(universe) for entry in all_entries),
        "disjoint held-out queries must all have C_A(q) = U",
    )

    summaries = {split: summarize(query_results[split]) for split in SPLITS}
    return {
        "name": "A — literal learner-visible identifiability",
        "hypothesis_class": {
            "domain": "all unordered two-element subsets of U",
            "codomain": "U",
            "input_swap_invariance": "built in by unordered-pair keys",
            "constraints": "the 48 TRAIN values only",
            "other_relational_axioms": "none",
        },
        "proof": {
            "candidate_characterization": (
                "For a TRAIN pair q, C_A(q) is its observed singleton. For any "
                "other pair q and any c in U, define f(q)=c, retain every TRAIN "
                "value, and assign arbitrary legal values to all remaining pairs. "
                "This is a total compatible function, so C_A(q)=U."
            ),
            "heldout_application": (
                "All ROLE_TEST and NOVEL_TEST pairs are disjoint from TRAIN; "
                "therefore every held-out candidate set is exactly U."
            ),
        },
        "explicit_countermodels": {
            "A_default_0": (
                "f_0(q) is the observed target when q is a TRAIN pair and 0 otherwise."
            ),
            "A_default_1": (
                "f_1(q) is the observed target when q is a TRAIN pair and 1 otherwise."
            ),
            "verification": {
                "total_domain_pairs_checked_per_model": len(universe)
                * (len(universe) - 1)
                // 2,
                "agree_on_all_train_observations": True,
                "disagree_on_every_heldout_query": True,
            },
        },
        "summary": summaries,
        "role_vs_novel_identifiability": {
            "genuinely_different": False,
            "reason": (
                "Both splits are wholly underdetermined with the identical "
                "candidate set U for every query; only their query counts differ."
            ),
        },
        "missing_information_boundary": (
            "For any currently free pair, forcing an answer requires additional "
            "information or axioms that eliminate 83 of its 84 legal outputs. "
            "The literal contract supplies no relation from held-out pairs to "
            "TRAIN values."
        ),
        "queries": query_results,
    }


def audit_b_candidates(
    query: Pair,
    universe: Sequence[int],
    mandatory_pair_to_completion: Mapping[Pair, int],
) -> list[int]:
    """Compute the exact role-neutral block candidate set from local pair use."""
    if query in mandatory_pair_to_completion:
        return [mandatory_pair_to_completion[query]]

    x, y = query
    candidates: list[int] = []
    for candidate in universe:
        if candidate in query:
            continue
        x_pair = canonical_pair((x, candidate), context="candidate companion pair")
        y_pair = canonical_pair((y, candidate), context="candidate companion pair")
        if (
            x_pair not in mandatory_pair_to_completion
            and y_pair not in mandatory_pair_to_completion
        ):
            candidates.append(candidate)
    return candidates


def audit_b(boundary: Mapping[str, Any]) -> dict[str, Any]:
    """Compute minimal role-neutral triad identifiability and check witnesses."""
    universe: tuple[int, ...] = boundary["universe"]
    mandatory_blocks: tuple[Block, ...] = boundary["mandatory_blocks"]
    pair_completion: Mapping[Pair, int] = boundary["mandatory_pair_to_completion"]
    pair_block: Mapping[Pair, Block] = boundary["mandatory_pair_to_block"]

    base_completions = validate_block_set(mandatory_blocks, universe, mandatory_blocks)
    require(
        base_completions == dict(pair_completion),
        "mandatory block completion map is internally inconsistent",
    )

    query_results: dict[str, list[dict[str, Any]]] = {}
    included_candidate_witnesses_checked = 0
    excluded_candidate_obstructions_checked = 0
    reported_witnesses_checked = 0

    for split in SPLITS:
        entries: list[dict[str, Any]] = []
        for query in boundary["heldout"][split]:
            candidates = audit_b_candidates(query, universe, pair_completion)
            candidate_set = set(candidates)

            if query in pair_completion:
                require(
                    candidates == [pair_completion[query]],
                    f"mandatory pair {query} must have one forced completion",
                )
                entry: dict[str, Any] = {
                    "query": list(query),
                    "classification": "forced",
                    "candidate_set_size": 1,
                    "candidate_set": candidates,
                    "witness": {
                        "mandatory_block": list(pair_block[query]),
                        "forced_answer": candidates[0],
                    },
                }
                require(
                    base_completions[query] == candidates[0],
                    f"forced witness fails at {query}",
                )
                reported_witnesses_checked += 1
            else:
                # Necessity and sufficiency are checked for every c in U, not only
                # for the two compact witnesses retained in the report.
                for candidate in universe:
                    added_block = tuple(sorted((*query, candidate)))
                    if candidate in candidate_set:
                        completions = validate_block_set(
                            (*mandatory_blocks, added_block),
                            universe,
                            mandatory_blocks,
                        )
                        require(
                            completions.get(query) == candidate,
                            f"included candidate {candidate} fails at {query}",
                        )
                        included_candidate_witnesses_checked += 1
                    else:
                        if candidate in query:
                            obstructed = True
                        else:
                            x_pair = canonical_pair(
                                (query[0], candidate), context="excluded x-pair"
                            )
                            y_pair = canonical_pair(
                                (query[1], candidate), context="excluded y-pair"
                            )
                            obstructed = (
                                x_pair in pair_completion or y_pair in pair_completion
                            )
                        require(
                            obstructed,
                            f"excluded candidate {candidate} lacks obstruction "
                            f"at {query}",
                        )
                        excluded_candidate_obstructions_checked += 1

                require(
                    len(candidates) >= 2,
                    f"underdetermined query {query} lacks two witness answers",
                )
                witness_answers = candidates[:2]
                witness_blocks = [
                    sorted((*query, answer)) for answer in witness_answers
                ]
                for answer, block in zip(witness_answers, witness_blocks, strict=True):
                    completions = validate_block_set(
                        (*mandatory_blocks, tuple(block)),
                        universe,
                        mandatory_blocks,
                    )
                    require(
                        completions.get(query) == answer,
                        f"reported block witness fails at {query} -> {answer}",
                    )
                    reported_witnesses_checked += 1

                entry = {
                    "query": list(query),
                    "classification": classify(candidates),
                    "candidate_set_size": len(candidates),
                    "candidate_set": candidates,
                    "witness": {
                        "construction": (
                            "mandatory TRAIN blocks plus the one added block; "
                            "all other optional blocks absent"
                        ),
                        "answers": witness_answers,
                        "added_blocks": witness_blocks,
                    },
                }
            entries.append(entry)
        query_results[split] = entries

    summaries = {split: summarize(query_results[split]) for split in SPLITS}
    return {
        "name": "B — minimal role-neutral triad semantics",
        "hypothesis_class": {
            "mandatory_block_axiom": (
                "Each TRAIN observation (a,b)->c asserts the unordered "
                "three-distinct-Event block {a,b,c}."
            ),
            "role_completion_axiom": (
                "Every block entails all three pair-to-third completions."
            ),
            "pair_uniqueness_axiom": ("A query pair belongs to at most one block."),
            "optional_block_axiom": (
                "Additional three-Event blocks may exist when they preserve the "
                "other axioms."
            ),
            "excluded_axioms": [
                "fixed total block count",
                "degree constraints",
                "habitat, Pasch, or Fano structure",
                "SFP coordinates",
                "global automorphisms",
                "unprovided closure or incidence facts",
            ],
        },
        "proof": {
            "candidate_characterization": (
                "Let P be the pairs used by mandatory TRAIN blocks. If q is in "
                "P, pair uniqueness and role completion force the unique third "
                "token of its mandatory block. If q={x,y} is not in P, then c is "
                "possible exactly when c is distinct from x and y and neither "
                "{x,c} nor {y,c} lies in P. Necessity follows from a block having "
                "three distinct Events and pair uniqueness. Sufficiency follows "
                "by adjoining only {x,y,c} to the mandatory blocks."
            ),
            "formula": (
                "C_B({x,y})={third_P(x,y)} when {x,y} is in P; otherwise "
                "C_B({x,y})={c in U\\{x,y}: {x,c} not in P and {y,c} not in P}."
            ),
        },
        "explicit_witness_family": {
            "forced": (
                "The mandatory TRAIN block containing q supplies its sole answer."
            ),
            "underdetermined": (
                "For each reported answer c, take exactly the 48 mandatory "
                "blocks and add {q[0],q[1],c}; the checked unused companion pairs "
                "make this a compatible block set."
            ),
            "verification": {
                "mandatory_blocks_checked": len(mandatory_blocks),
                "mandatory_pairs_checked": len(pair_completion),
                "included_candidate_block_sets_checked": (
                    included_candidate_witnesses_checked
                ),
                "excluded_candidate_obstructions_checked": (
                    excluded_candidate_obstructions_checked
                ),
                "reported_witnesses_checked": reported_witnesses_checked,
            },
        },
        "summary": summaries,
        "role_vs_novel_identifiability": {
            "genuinely_different": (
                summaries["role_test"]["forced"] != summaries["novel_test"]["forced"]
                or summaries["role_test"]["underdetermined"]
                != summaries["novel_test"]["underdetermined"]
            ),
            "reason": (
                "ROLE_TEST pairs are mandatory-block role completions, whereas "
                "NOVEL_TEST pairs are not mandatory-block pairs and retain "
                "multiple admissible added-block completions."
            ),
        },
        "missing_information_boundary": (
            "Each currently underdetermined pair would require an additional "
            "block assertion or other incidence constraints strong enough to "
            "eliminate every admissible third token but one. Axioms 1–4 do not "
            "select among the compatible optional blocks."
        ),
        "queries": query_results,
    }


def build_report(boundary: Mapping[str, Any]) -> dict[str, Any]:
    """Build and cross-check the complete deterministic machine report."""
    audit_a_report = audit_a(boundary)
    audit_b_report = audit_b(boundary)

    report = {
        "schema": REPORT_SCHEMA,
        "task_revision": TASK_REVISION,
        "information_exposure": {
            "project_information_beyond_task_and_packet": True,
            "non_material_exposure": (
                "The session-start context exposed generic workspace path names; "
                "git status exposed the current branch/worktree state; exact-name "
                "existence checks were made for the declared packet and result "
                "paths; and repository-pinned lint tooling was invoked without "
                "inspecting its configuration content. No other scientific "
                "artifact content was read."
            ),
            "material_blindness_breached": False,
            "material_withheld_information_seen": [],
            "declaration": (
                "No Issue-014 held-out targets, neural outcomes, representation "
                "diagnostics, Surface 14 conclusion, or GPT Owner expectation was "
                "known or consulted."
            ),
        },
        "input_validation": {
            "packet_schema": boundary["packet_schema"],
            "packet_semantic_sha256": boundary["packet_semantic_sha256"],
            "universe_size": len(boundary["universe"]),
            "train_observations": len(boundary["train"]),
            "role_test_queries": len(boundary["heldout"]["role_test"]),
            "novel_test_queries": len(boundary["heldout"]["novel_test"]),
            "train_role_novel_pairs_pairwise_disjoint": True,
            "mandatory_blocks": len(boundary["mandatory_blocks"]),
            "mandatory_block_pairs": len(boundary["mandatory_pair_to_completion"]),
        },
        "audits": {
            "A_literal_learner_visible": audit_a_report,
            "B_minimal_role_neutral_triads": audit_b_report,
        },
        "interpretation_fence": [
            "This report concerns only the frozen finite information boundary.",
            "It does not explain or predict any neural learner's behavior.",
            "It does not claim that an unforced target is unlearnable under bias.",
            "It does not claim that a forced target will be learned by optimization.",
            "It does not select a scientifically correct additional structure.",
            "It does not recover, guess, or score any hidden certified target.",
        ],
        "determinism": {
            "stochastic_search_used": False,
            "global_exhaustive_search_used": False,
            "method": "direct finite combinatorial characterization",
        },
    }

    a_summary = audit_a_report["summary"]
    b_summary = audit_b_report["summary"]
    require(a_summary["role_test"]["forced"] == 0, "unexpected Audit-A role result")
    require(a_summary["role_test"]["underdetermined"] == 96, "Audit-A role count")
    require(a_summary["novel_test"]["forced"] == 0, "unexpected Audit-A novel result")
    require(
        a_summary["novel_test"]["underdetermined"] == 24,
        "Audit-A novel count",
    )
    require(b_summary["role_test"]["forced"] == 96, "Audit-B role count")
    require(
        b_summary["role_test"]["underdetermined"] == 0,
        "unexpected Audit-B role result",
    )
    require(b_summary["novel_test"]["forced"] == 0, "unexpected Audit-B novel result")
    require(
        b_summary["novel_test"]["underdetermined"] == 24,
        "Audit-B novel count",
    )
    return report


def format_set(values: Sequence[int]) -> str:
    """Render a finite candidate set compactly and exactly."""
    return "{" + ", ".join(str(value) for value in values) + "}"


def summary_table(audit: Mapping[str, Any]) -> list[str]:
    """Render one audit's split summaries as Markdown."""
    lines = [
        "| Split | Forced | Underdetermined | Candidate-size histogram |",
        "|---|---:|---:|---|",
    ]
    for split in SPLITS:
        summary = audit["summary"][split]
        histogram = ", ".join(
            f"{size} → {count}"
            for size, count in summary["candidate_set_size_histogram"].items()
        )
        lines.append(
            f"| {split.upper()} | {summary['forced']} | "
            f"{summary['underdetermined']} | {histogram} |"
        )
    return lines


def exact_query_table(
    entries: Sequence[Mapping[str, Any]], *, alias_full_universe: bool = False
) -> list[str]:
    """Render every query and its exact candidate set."""
    lines = [
        "| Query | Status | Exact candidate set | Size |",
        "|---|---|---|---:|",
    ]
    for entry in entries:
        query = "{" + ", ".join(str(value) for value in entry["query"]) + "}"
        candidates = (
            "U (the explicitly enumerated set above)"
            if alias_full_universe
            else format_set(entry["candidate_set"])
        )
        lines.append(
            f"| `{query}` | {entry['classification']} | {candidates} | "
            f"{entry['candidate_set_size']} |"
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the machine report as the required human-auditable result."""
    validation = report["input_validation"]
    audit_a_report = report["audits"]["A_literal_learner_visible"]
    audit_b_report = report["audits"]["B_minimal_role_neutral_triads"]
    universe = audit_a_report["queries"]["role_test"][0]["candidate_set"]

    lines = [
        "# 015.02 — Coder blind identifiability audit result — GPT",
        "",
        "## Information exposure",
        "",
        "**Information exposure:** Yes, but only non-material project information "
        "beyond the handed task and declared packet: the session-start context "
        "showed generic workspace path names, git status showed branch/worktree "
        "state, exact-name existence checks covered only the declared packet and "
        "result paths, and repository-pinned lint tooling was invoked without "
        "inspecting its configuration content. I did not read any other scientific "
        "artifact content. Material blindness was **not** breached: no withheld "
        "Issue-014 held-out target, neural outcome, representation diagnostic, "
        "Surface 14 conclusion, or GPT Owner expectation was known or consulted. "
        "The blind audit therefore proceeds.",
        "",
        "## Frozen input and method",
        "",
        f"- Task revision: `{report['task_revision']}`",
        f"- Packet schema: `{validation['packet_schema']}`",
        f"- Normalized packet SHA-256: `{validation['packet_semantic_sha256']}`",
        f"- Universe: {validation['universe_size']} opaque IDs (`0` through `83`)",
        f"- TRAIN observations: {validation['train_observations']}",
        f"- ROLE_TEST queries: {validation['role_test_queries']}",
        f"- NOVEL_TEST queries: {validation['novel_test_queries']}",
        "- TRAIN, ROLE_TEST, and NOVEL_TEST pairs: pairwise disjoint (checked)",
        f"- Mandatory TRAIN blocks/pairs for Audit B: "
        f"{validation['mandatory_blocks']}/{validation['mandatory_block_pairs']}",
        "- Method: direct exact combinatorics; no stochastic or global exhaustive "
        "search",
        "",
        "## Audit A — literal learner-visible identifiability",
        "",
        "### Assumptions and exact proof",
        "",
        "The hypothesis class is exactly all functions from unordered two-element "
        "subsets of `U` to `U`, constrained only at the 48 TRAIN pairs. Input-swap "
        "invariance is built into the pair keys; every one of the 84 tokens is a "
        "legal output; no other relational axiom is used.",
        "",
        audit_a_report["proof"]["candidate_characterization"],
        " " + audit_a_report["proof"]["heldout_application"],
        "",
        "### Counts and histograms",
        "",
        *summary_table(audit_a_report),
        "",
        "### Explicit compatible countermodels",
        "",
        "Define `f₀(q)` to equal the observed target on each TRAIN pair and `0` "
        "on every other pair. Define `f₁` identically except that every non-TRAIN "
        "pair maps to `1`. Both are total functions on all 3,486 unordered pairs, "
        "both satisfy every TRAIN observation, and they disagree on every held-out "
        "query. The script checks both models over the complete domain. These two "
        "models are representative witnesses; the proof above constructs a model "
        "for every candidate `c ∈ U`.",
        "",
        "### Exact candidate sets",
        "",
        "The explicit universe used in every row below is:",
        "",
        f"`U = {format_set(universe)}`",
        "",
        "#### ROLE_TEST",
        "",
        *exact_query_table(
            audit_a_report["queries"]["role_test"], alias_full_universe=True
        ),
        "",
        "#### NOVEL_TEST",
        "",
        *exact_query_table(
            audit_a_report["queries"]["novel_test"], alias_full_universe=True
        ),
        "",
        "### Boundary statement",
        "",
        "ROLE_TEST and NOVEL_TEST do **not** have genuinely different "
        "identifiability status under Audit A: every query in each split is "
        "underdetermined with candidate set `U`. To force any currently free "
        "pair, added information or axioms would have to eliminate 83 of its 84 "
        "legal outputs by linking that pair to a unique answer; the literal "
        "contract contains no such link.",
        "",
        "## Audit B — minimal role-neutral triad semantics",
        "",
        "### Assumptions and exact proof",
        "",
        "Each TRAIN observation asserts one unordered block of three distinct "
        "Events; each such block entails all three pair-to-third completions; each "
        "pair belongs to at most one block; and compatible additional blocks may "
        "exist. No fixed block count, degree constraint, habitat/Pasch/Fano "
        "structure, SFP coordinate, global automorphism, or unprovided closure or "
        "incidence fact is used.",
        "",
        audit_b_report["proof"]["candidate_characterization"],
        "",
        "Equivalently, with `P` the 144 mandatory-block pairs:",
        "",
        f"`{audit_b_report['proof']['formula']}`",
        "",
        "### Counts and histograms",
        "",
        *summary_table(audit_b_report),
        "",
        "### Explicit compatible block-set witnesses",
        "",
        "For each forced row, the reported mandatory TRAIN block is the witness "
        "and pair uniqueness excludes every other completion. For every "
        "underdetermined row, the machine report gives two answers and two added "
        "blocks. Each witness block set consists of exactly the 48 mandatory "
        "blocks plus that one added block, with all other optional blocks absent. "
        "The script validates both reported witnesses and, more strongly, a "
        "separate block-set witness for every member of every computed candidate "
        "set; it also checks a concrete pair-use obstruction for every excluded "
        "token.",
        "",
        "### Exact candidate sets",
        "",
        "#### ROLE_TEST",
        "",
        *exact_query_table(audit_b_report["queries"]["role_test"]),
        "",
        "#### NOVEL_TEST",
        "",
        *exact_query_table(audit_b_report["queries"]["novel_test"]),
        "",
        "### Boundary statement",
        "",
        "ROLE_TEST and NOVEL_TEST have genuinely **different** identifiability "
        "status under Audit B: all ROLE_TEST queries are forced role completions "
        "of mandatory blocks, while every NOVEL_TEST query admits multiple "
        "compatible optional-block completions. For a currently underdetermined "
        "pair to become forced, additional information must uniquely assert its "
        "block completion, or additional incidence constraints must eliminate all "
        "but one admissible third token. Axioms 1–4 do not make that selection.",
        "",
        "## Interpretation fence",
        "",
        "This establishes only finite-boundary identifiability or "
        "underdetermination under the separately stated assumptions. It does not "
        "explain neural behavior, equate unforced with unlearnable under useful "
        "bias, guarantee optimization of forced consequences, select a physically "
        "correct extra structure, recommend a learning mechanism, or recover or "
        "score any hidden certified target.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python issues/015-learning-law-for-consequence-structure/\\",
        "015.02-Code-attachments/audit_identifiability.py",
        "```",
        "",
        "The script's defaults read only "
        "`015.01-Code-attachments/014-literal-information-boundary.json` and "
        "write the two declared 015.02 result paths. Explicit `--packet`, "
        "`--json-output`, and `--markdown-output` paths may also be supplied.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse paths without reading any project configuration."""
    script_path = Path(__file__).resolve()
    issue_dir = script_path.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=(
            issue_dir
            / "015.01-Code-attachments"
            / "014-literal-information-boundary.json"
        ),
        help="path to the declared blind information-boundary JSON packet",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=script_path.parent / "identifiability_audit.json",
        help="path for the deterministic machine report",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=issue_dir / "015.02-Coder-blind-identifiability-audit-result-GPT.md",
        help="path for the deterministic human-readable result",
    )
    return parser.parse_args()


def main() -> None:
    """Run both exact audits, self-check, and write deterministic outputs."""
    args = parse_args()
    boundary = load_and_validate_packet(args.packet)
    report = build_report(boundary)
    json_text = (
        json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    markdown_text = render_markdown(report)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json_text, encoding="utf-8")
    args.markdown_output.write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()

"""011.03 bounded TRAIN-only optimization search and exact selector.

The process loads one content-addressed capsule containing only the eight main-arm
TRAIN namespaces.  It has no data-construction or evaluation import.  Every one
of the at-most-24 preregistered candidates runs on seeds 0..7, and the selected
prescription is derived by the Owner-specified five-part lexicographic rule.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fano_heads import BANNED_IDENTIFIERS, answer_rule_audit
from fano_optimization_core import (
    capsule_digest,
    file_sha256,
    load_capsule,
    train_prescription,
)
from fano_optimization_manifest import (
    DEV_SEEDS,
    EXACT_LOSS_MINIMUM,
    FRESH_SEEDS,
    LOSS_TOLERANCE,
    object_sha256,
    validate_loaded,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "011_optimization_artifacts"
CANDIDATE_MANIFEST = ARTIFACTS / "candidate_manifest.json"
CAPSULE = ARTIFACTS / "train_capsule.pt"
CAPSULE_MANIFEST = ARTIFACTS / "train_capsule.json"
OUTPUT = ARTIFACTS / "development_search.json"
PARTIAL = ARTIFACTS / "development_search.partial.json"
SELECTED = ARTIFACTS / "selected_optimization.json"
SEARCH_WORKERS = 4

SELECTION_BANNED_IMPORTS = (
    "fano_analysis",
    "fano_baselines",
    "fano_sweep",
    "fano_task",
    "task",
    "topographo",
)
SELECTION_BANNED_CALLS = (
    "build_batch",
    "build_batches",
    "build_localized_observation",
    "build_namespaces",
    "build_observation",
    "left_right_swap_audit",
    "recovery_audit",
    "score_batch",
    "transport_audit",
)
AUDIT_VOCABULARY = (
    "SELECTION_BANNED_IMPORTS",
    "SELECTION_BANNED_CALLS",
    "AUDIT_VOCABULARY",
)
SELECTION_MODULES = ("fano_optimization_core", "fano_optimization_search")
LEARNED_MODULES = ("fano_heads", *SELECTION_MODULES)

_WORKER_BATCHES: list[dict] | None = None


def render(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _strip_wall(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_wall(item)
            for key, item in value.items()
            if key != "wall_sec"
        }
    if isinstance(value, list):
        return [_strip_wall(item) for item in value]
    return value


def _load_inputs() -> tuple[dict, dict, dict]:
    candidate = json.loads(CANDIDATE_MANIFEST.read_text())
    validate_loaded(candidate)
    capsule_row = json.loads(CAPSULE_MANIFEST.read_text())
    if capsule_row["candidate_manifest_sha256"] != candidate["manifest_sha256"]:
        raise ValueError("TRAIN capsule was not bound to this candidate manifest")
    if file_sha256(CAPSULE) != capsule_row["capsule_file_sha256"]:
        raise ValueError("TRAIN capsule file hash differs from its manifest")
    capsule = load_capsule(CAPSULE)
    if capsule_digest(capsule) != capsule_row["capsule_content_sha256"]:
        raise ValueError("TRAIN capsule content hash differs from its manifest")
    return candidate, capsule_row, capsule


class _ExecutableStripper(ast.NodeTransformer):
    def _body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        rows = list(body)
        if (
            rows
            and isinstance(rows[0], ast.Expr)
            and isinstance(rows[0].value, ast.Constant)
            and isinstance(rows[0].value.value, str)
        ):
            rows = rows[1:]
        return rows

    def visit_Module(self, node: ast.Module) -> Any:
        node.body = self._body(node.body)
        node.body = [self.visit(row) for row in node.body]
        node.body = [row for row in node.body if row is not None]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        node.body = self._body(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.body = self._body(node.body)
        node.returns = None
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            argument.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name) and node.target.id in AUDIT_VOCABULARY:
            return None
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=self.visit(node.value))

    def visit_Assign(self, node: ast.Assign) -> Any:
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if any(name in AUDIT_VOCABULARY for name in names):
            return None
        return self.generic_visit(node)


def _executable_tree(module_name: str) -> ast.Module:
    module = __import__(module_name)
    tree = ast.parse(inspect.getsource(module))
    return ast.fix_missing_locations(_ExecutableStripper().visit(tree))


def learned_path_audit() -> dict:
    rows = {"fano_heads": answer_rule_audit()["per_module"]["fano_heads"]}
    for module_name in SELECTION_MODULES:
        names: set[str] = set()
        xor_operators = 0
        for node in ast.walk(_executable_tree(module_name)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[-1])
                if node.asname:
                    names.add(node.asname)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
                xor_operators += 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitXor):
                xor_operators += 1
        found = sorted(name for name in BANNED_IDENTIFIERS if name in names)
        rows[module_name] = {
            "banned_identifiers_in_code": found,
            "xor_operators_in_code": xor_operators,
            "clean": not found and xor_operators == 0,
        }
    return {
        "modules": list(LEARNED_MODULES),
        "per_module": rows,
        "clean": all(row["clean"] for row in rows.values()),
    }


def selection_boundary_audit(capsule: dict) -> dict:
    imports: set[str] = set()
    calls: set[str] = set()
    for module_name in SELECTION_MODULES:
        for node in ast.walk(_executable_tree(module_name)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    forbidden_imports = sorted(set(SELECTION_BANNED_IMPORTS) & imports)
    forbidden_calls = sorted(set(SELECTION_BANNED_CALLS) & calls)
    labels = [batch["split"] for batch in capsule["batches"]]
    checks = {
        "capsule_arm_is_main": capsule["arm"] == "main",
        "capsule_split_is_train": capsule["split"] == "train",
        "every_batch_is_train": bool(labels) and set(labels) == {"train"},
        "eight_train_batches": len(labels) == 8,
        "no_direct_data_or_evaluation_import": not forbidden_imports,
        "no_data_constructor_or_evaluation_call": not forbidden_calls,
    }
    return {
        "selection_modules": list(SELECTION_MODULES),
        "direct_imports": sorted(imports),
        "calls": sorted(calls),
        "forbidden_imports_found": forbidden_imports,
        "forbidden_calls_found": forbidden_calls,
        "batch_splits": labels,
        "checks": checks,
        "clean": all(checks.values()),
        "statement": (
            "the selector can load only the committed main/TRAIN tensor capsule; "
            "it neither imports task/evaluation modules nor calls their constructors"
        ),
    }


def _worker_init(
    capsule_path: str,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> None:
    global _WORKER_BATCHES
    path = Path(capsule_path)
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("worker sees the wrong TRAIN capsule file")
    capsule = load_capsule(path)
    if capsule_digest(capsule) != expected_content_sha256:
        raise ValueError("worker sees the wrong TRAIN capsule content")
    _WORKER_BATCHES = capsule["batches"]


def _run_seed(candidate: dict, seed: int) -> dict:
    if seed not in DEV_SEEDS:
        raise ValueError(f"selection refuses non-development seed {seed}")
    if _WORKER_BATCHES is None:
        raise RuntimeError("worker TRAIN capsule was not initialized")
    _, row = train_prescription(
        _WORKER_BATCHES,
        candidate["prescription"],
        seed=seed,
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["sha256"],
        **row,
    }


def _candidate_summary(candidate: dict, rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: row["seed"])
    converged = [row for row in ordered if row["converged"]]
    steps = [int(row["steps_to_convergence"]) for row in converged]
    median_steps = statistics.median(steps) if steps else None
    worst_excess = max(row["final_excess_loss"] for row in ordered)
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["sha256"],
        "canonical_json": candidate["canonical_json"],
        "converged_count": len(converged),
        "converged_seeds": [row["seed"] for row in converged],
        "median_steps_to_convergence": median_steps,
        "worst_final_excess_loss": worst_excess,
        "changed_knob_count": candidate["changed_knob_count"],
        "changed_knobs": candidate["changed_knobs"],
        "eligible_8_of_8": len(converged) == len(DEV_SEEDS),
        "per_seed": ordered,
    }


def _selection_key(summary: dict) -> tuple:
    median = summary["median_steps_to_convergence"]
    return (
        -int(summary["converged_count"]),
        math.inf if median is None else float(median),
        float(summary["worst_final_excess_loss"]),
        int(summary["changed_knob_count"]),
        str(summary["canonical_json"]),
    )


def select(manifest: dict, runs: list[dict]) -> tuple[list[dict], dict]:
    by_candidate: dict[str, list[dict]] = {}
    for row in runs:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    summaries = []
    for candidate in manifest["candidates"]:
        rows = by_candidate.get(candidate["candidate_id"], [])
        if sorted(row["seed"] for row in rows) != list(DEV_SEEDS):
            raise ValueError(
                f"candidate {candidate['candidate_id']} does not have exactly seeds 0..7"
            )
        summaries.append(_candidate_summary(candidate, rows))
    ranked = sorted(summaries, key=_selection_key)
    selected = ranked[0]
    for rank, summary in enumerate(ranked, start=1):
        summary["rank"] = rank
        summary["selection_key"] = [
            -summary["converged_count"],
            summary["median_steps_to_convergence"],
            summary["worst_final_excess_loss"],
            summary["changed_knob_count"],
            summary["canonical_json"],
        ]
    return ranked, selected


def _search_payload(
    manifest: dict,
    capsule_row: dict,
    capsule: dict,
    runs: list[dict],
) -> dict:
    learned = learned_path_audit()
    boundary = selection_boundary_audit(capsule)
    if not learned["clean"] or not boundary["clean"]:
        raise ValueError("TRAIN-only learned-path or access-boundary audit failed")
    ranked, selected = select(manifest, runs)
    stripped_runs = _strip_wall(sorted(runs, key=lambda row: (row["candidate_id"], row["seed"])))
    body = {
        "schema": "occurrence.011.03.development-search.v1",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "train_capsule_manifest_sha256": capsule_row["manifest_sha256"],
        "train_capsule_content_sha256": capsule_row["capsule_content_sha256"],
        "development_seeds": list(DEV_SEEDS),
        "read_surface": "main-arm TRAIN fit only",
        "exact_loss_minimum": EXACT_LOSS_MINIMUM,
        "loss_tolerance": LOSS_TOLERANCE,
        "candidate_count": len(manifest["candidates"]),
        "run_count": len(runs),
        "workers": SEARCH_WORKERS,
        "learned_path_audit": learned,
        "selection_boundary_audit": boundary,
        "runs": sorted(runs, key=lambda row: (row["candidate_id"], row["seed"])),
        "ranking": ranked,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_sha256": selected["candidate_sha256"],
        "selected_converged_count": selected["converged_count"],
        "eligible_for_fresh_evaluation": selected["eligible_8_of_8"],
        "runs_digest": object_sha256(stripped_runs),
        "non_replayable_keys": ["wall_sec"],
    }
    return {**body, "search_sha256": object_sha256(_strip_wall(body))}


def _selected_payload(manifest: dict, search: dict) -> dict:
    selected_id = search["selected_candidate_id"]
    candidate = next(
        row for row in manifest["candidates"] if row["candidate_id"] == selected_id
    )
    summary = next(
        row for row in search["ranking"] if row["candidate_id"] == selected_id
    )
    if not summary["eligible_8_of_8"]:
        raise ValueError("011.03 stop rule: no candidate reached 8/8")
    compact_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"per_seed", "canonical_json"}
    }
    body = {
        "schema": "occurrence.011.03.selected-optimization.v1",
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "development_search_sha256": search["search_sha256"],
        "development_runs_digest": search["runs_digest"],
        "selected_candidate_id": candidate["candidate_id"],
        "selected_candidate_sha256": candidate["sha256"],
        "selected_canonical_json": candidate["canonical_json"],
        "selected_prescription": candidate["prescription"],
        "development_summary": compact_summary,
        "exact_loss_minimum": EXACT_LOSS_MINIMUM,
        "loss_tolerance": LOSS_TOLERANCE,
        "development_seeds": list(DEV_SEEDS),
        "fresh_evaluation_seeds": list(FRESH_SEEDS),
        "fresh_seed_status": "not executed by this TRAIN-only program",
        "evaluation_lock": (
            "the evaluator must read this committed artifact and receive its exact "
            "selected_optimization_sha256 as a command-line lock"
        ),
    }
    return {**body, "selected_optimization_sha256": object_sha256(body)}


def _partial_payload(manifest_hash: str, rows: list[dict]) -> dict:
    return {
        "schema": "occurrence.011.03.development-search-partial.v1",
        "candidate_manifest_sha256": manifest_hash,
        "completed_candidate_ids": sorted({row["candidate_id"] for row in rows}),
        "runs": sorted(rows, key=lambda row: (row["candidate_id"], row["seed"])),
    }


def run_search(*, resume: bool) -> dict:
    manifest, capsule_row, capsule = _load_inputs()
    boundary = selection_boundary_audit(capsule)
    learned = learned_path_audit()
    if not boundary["clean"] or not learned["clean"]:
        raise SystemExit("FAIL: search boundary or learned path is not clean")

    runs: list[dict] = []
    if resume:
        if not PARTIAL.exists():
            raise SystemExit(f"FAIL: cannot resume; {PARTIAL} does not exist")
        partial = json.loads(PARTIAL.read_text())
        if partial["candidate_manifest_sha256"] != manifest["manifest_sha256"]:
            raise SystemExit("FAIL: partial search belongs to another manifest")
        runs = partial["runs"]
    elif PARTIAL.exists() or OUTPUT.exists() or SELECTED.exists():
        raise SystemExit(
            "FAIL: search output already exists; use --resume only for a matching partial"
        )

    completed = {row["candidate_id"] for row in runs}
    with ProcessPoolExecutor(
        max_workers=SEARCH_WORKERS,
        initializer=_worker_init,
        initargs=(
            str(CAPSULE),
            capsule_row["capsule_file_sha256"],
            capsule_row["capsule_content_sha256"],
        ),
    ) as executor:
        for index, candidate in enumerate(manifest["candidates"], start=1):
            candidate_id = candidate["candidate_id"]
            if candidate_id in completed:
                print(f"[{index:02d}/{len(manifest['candidates']):02d}] resume {candidate_id}")
                continue
            futures = {
                executor.submit(_run_seed, candidate, seed): seed for seed in DEV_SEEDS
            }
            rows = []
            for future in as_completed(futures):
                rows.append(future.result())
            rows.sort(key=lambda row: row["seed"])
            runs.extend(rows)
            summary = _candidate_summary(candidate, rows)
            print(
                f"[{index:02d}/{len(manifest['candidates']):02d}] {candidate_id}: "
                f"{summary['converged_count']}/8 converged, median steps "
                f"{summary['median_steps_to_convergence']}"
            )
            PARTIAL.write_text(render(_partial_payload(manifest["manifest_sha256"], runs)))

    payload = _search_payload(manifest, capsule_row, capsule, runs)
    OUTPUT.write_text(render(payload))
    if payload["eligible_for_fresh_evaluation"]:
        selected = _selected_payload(manifest, payload)
        SELECTED.write_text(render(selected))
        print(
            f"SELECTED {payload['selected_candidate_id']} at 8/8; lock "
            f"{selected['selected_optimization_sha256']}"
        )
    else:
        print(
            "STOP E: no preregistered candidate reached 8/8; fresh evaluation remains sealed"
        )
    if PARTIAL.exists():
        PARTIAL.unlink()
    return payload


def check() -> dict:
    manifest, capsule_row, capsule = _load_inputs()
    if not OUTPUT.exists():
        raise SystemExit(f"FAIL: {OUTPUT} does not exist")
    stored = json.loads(OUTPUT.read_text())
    rebuilt = _search_payload(manifest, capsule_row, capsule, stored["runs"])
    if stored != rebuilt:
        raise SystemExit("FAIL: development search artifact does not re-derive")
    if stored["eligible_for_fresh_evaluation"]:
        if not SELECTED.exists():
            raise SystemExit(f"FAIL: {SELECTED} does not exist")
        expected = _selected_payload(manifest, stored)
        if json.loads(SELECTED.read_text()) != expected:
            raise SystemExit("FAIL: selected optimization artifact does not re-derive")
    elif SELECTED.exists():
        raise SystemExit("FAIL: selected artifact exists despite the 8/8 stop rule")
    print(
        f"PASS: {OUTPUT.name} re-derives; selected "
        f"{stored['selected_candidate_id']} at {stored['selected_converged_count']}/8"
    )
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check and args.resume:
        raise SystemExit("FAIL: --check and --resume are mutually exclusive")
    if args.check:
        check()
    else:
        run_search(resume=args.resume)


if __name__ == "__main__":
    main()

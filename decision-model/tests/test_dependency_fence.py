from __future__ import annotations

import ast
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DECISION_MODEL_SOURCE = PACKAGE_ROOT / "src" / "decision_model"
TOPOGRAPHO_SOURCE = REPOSITORY_ROOT / "topographo"


def _dependency_references(path: Path, dependency: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    references: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == dependency:
                    references.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] == dependency:
                references.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call) and node.args:
            target = node.func
            dynamic_import = (
                isinstance(target, ast.Name) and target.id == "__import__"
            ) or (
                isinstance(target, ast.Attribute)
                and target.attr == "import_module"
            )
            first_argument = node.args[0]
            if (
                dynamic_import
                and isinstance(first_argument, ast.Constant)
                and isinstance(first_argument.value, str)
                and first_argument.value.split(".", 1)[0] == dependency
            ):
                references.append(
                    f"line {node.lineno}: dynamic import {first_argument.value}"
                )
    return tuple(references)


def _assert_no_dependency(
    paths: Iterable[Path],
    dependency: str,
) -> None:
    violations = {
        str(path): references
        for path in paths
        if (references := _dependency_references(path, dependency))
    }
    assert violations == {}


def test_generic_decision_model_has_no_topographo_dependency() -> None:
    _assert_no_dependency(DECISION_MODEL_SOURCE.rglob("*.py"), "topographo")


def test_topographo_has_no_reverse_decision_model_dependency() -> None:
    assert TOPOGRAPHO_SOURCE.is_dir()
    _assert_no_dependency(TOPOGRAPHO_SOURCE.rglob("*.py"), "decision_model")


def test_generic_decision_model_imports_only_the_standard_library() -> None:
    imported_roots: set[str] = set()
    for path in DECISION_MODEL_SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= sys.stdlib_module_names


def test_distribution_metadata_is_independent_and_pinned() -> None:
    metadata = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]

    assert project["name"] == "decision-model"
    assert project["version"] == "1.1.0"
    assert project["scripts"] == {
        "decision-model-zero": "decision_model._zero:main"
    }
    assert project["requires-python"] == ">=3.11"
    assert project["license"] == "MIT"
    assert project["authors"] == [{"name": "The Swan Factory"}]
    assert project["dependencies"] == []
    assert metadata["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert not any(
        dependency.split(".", 1)[0] == "topographo"
        for dependency in project["dependencies"]
    )

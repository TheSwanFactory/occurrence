from __future__ import annotations

import inspect

import decision_model
from decision_model import _core, _errors

EXPECTED_PUBLIC_API = (
    "AnnihilationError",
    "BackendResolutionError",
    "Distribution",
    "Effect",
    "InvalidResolutionError",
    "Probability",
    "ResolutionError",
    "Resolver",
    "State",
    "Test",
    "resolve_effect",
    "resolve_test",
)


def test_public_api_is_explicit_and_exact() -> None:
    assert decision_model.__all__ == EXPECTED_PUBLIC_API
    assert not {
        "Prepare",
        "Interpret",
        "sample",
        "argmax",
        "Policy",
        "Action",
        "Event",
    } & set(decision_model.__all__)


def test_wildcard_import_exposes_only_the_declared_api() -> None:
    namespace: dict[str, object] = {}
    exec("from decision_model import *", namespace)
    namespace.pop("__builtins__")
    assert tuple(namespace) == EXPECTED_PUBLIC_API


def test_reexports_preserve_object_identity() -> None:
    core_names = {
        "Distribution",
        "Effect",
        "Probability",
        "Resolver",
        "State",
        "Test",
        "resolve_effect",
        "resolve_test",
    }
    for name in core_names:
        assert getattr(decision_model, name) is getattr(_core, name)

    error_names = {
        "AnnihilationError",
        "BackendResolutionError",
        "InvalidResolutionError",
        "ResolutionError",
    }
    for name in error_names:
        assert getattr(decision_model, name) is getattr(_errors, name)


def test_state_is_the_exported_generic_concept() -> None:
    assert decision_model.State.__name__ == "State"
    assert decision_model.State.__contravariant__ is True


def test_resolution_functions_have_two_positional_inputs() -> None:
    effect_parameters = tuple(inspect.signature(decision_model.resolve_effect).parameters.values())
    test_parameters = tuple(inspect.signature(decision_model.resolve_test).parameters.values())

    assert tuple(parameter.name for parameter in effect_parameters) == ("state", "effect")
    assert tuple(parameter.name for parameter in test_parameters) == ("state", "test")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in (*effect_parameters, *test_parameters)
    )


def test_resolution_error_hierarchy_is_typed() -> None:
    assert issubclass(decision_model.BackendResolutionError, decision_model.ResolutionError)
    assert issubclass(decision_model.AnnihilationError, decision_model.BackendResolutionError)
    assert issubclass(decision_model.InvalidResolutionError, decision_model.ResolutionError)

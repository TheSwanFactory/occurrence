"""Compatibility facade for the layered exact sedenion machine.

New code should import algebra, programs, execution, projective state policy,
observers, and codecs from their owning ``topographo.ssd`` modules. This facade
keeps the 0.4 exact-machine API available without duplicating implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from topographo.ssd.codec import (
    JsonObject,
    expression_from_json,
    expression_to_json,
    literal,
    replay_serialized,
    run_to_json,
    serialize_run,
    value_from_json,
    value_to_json,
)
from topographo.ssd.exact import (
    CONVENTION,
    DIMENSION,
    Value,
    add,
    basis,
    checked,
    conj,
    equal,
    from_core_coordinates,
    mul,
    norm2,
    one,
    scale,
    sub,
    to_core_coordinates,
    value,
    zero,
)
from topographo.ssd.machine import (
    Annihilated,
    Completed,
    RawVectorPolicy,
    StatePolicy,
    Transition,
    execute,
    run,
)
from topographo.ssd.observers import squared_norm
from topographo.ssd.program import (
    Add,
    Conjugate,
    Expression,
    Literal,
    Multiply,
    Program,
    Subtract,
)
from topographo.ssd.program import evaluate as evaluate_typed
from topographo.ssd.projective import (
    ProjectivePolicy,
    canonicalize,
    equivalent,
    run_projective,
)

TraceStep = Transition
RunResult = Completed
RayResult = Completed
run_ray = run_projective


def evaluate(expression: Expression | Mapping[str, Any]) -> Value:
    """Evaluate typed syntax or parse and evaluate a legacy JSON expression."""

    if isinstance(expression, Mapping):
        return evaluate_typed(expression_from_json(expression))
    return evaluate_typed(expression)


__all__ = [
    "Add",
    "Annihilated",
    "CONVENTION",
    "Completed",
    "Conjugate",
    "DIMENSION",
    "Expression",
    "JsonObject",
    "Literal",
    "Multiply",
    "Program",
    "ProjectivePolicy",
    "RawVectorPolicy",
    "RayResult",
    "RunResult",
    "StatePolicy",
    "Subtract",
    "TraceStep",
    "Transition",
    "Value",
    "add",
    "basis",
    "canonicalize",
    "checked",
    "conj",
    "equal",
    "equivalent",
    "evaluate",
    "execute",
    "expression_from_json",
    "expression_to_json",
    "from_core_coordinates",
    "literal",
    "mul",
    "norm2",
    "one",
    "replay_serialized",
    "run",
    "run_projective",
    "run_ray",
    "run_to_json",
    "scale",
    "serialize_run",
    "squared_norm",
    "sub",
    "to_core_coordinates",
    "value",
    "value_from_json",
    "value_to_json",
    "zero",
]

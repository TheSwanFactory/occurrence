"""JSON wire codecs for exact expressions and ordered machine runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, TypeAlias

from topographo.ssd import exact
from topographo.ssd.machine import Completed, run
from topographo.ssd.observers import squared_norm
from topographo.ssd.program import (
    Add,
    Conjugate,
    Expression,
    Literal,
    Multiply,
    Program,
    Subtract,
    as_program,
)

JsonObject: TypeAlias = dict[str, Any]


def _fraction_json(number: Fraction) -> str:
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def value_to_json(operand: exact.Value) -> list[str]:
    """Encode a value as canonical rational strings."""

    return [_fraction_json(number) for number in exact.checked(operand)]


def value_from_json(raw: Any, *, where: str = "value") -> exact.Value:
    """Decode a value from a JSON array of rational strings."""

    if not isinstance(raw, list):
        raise TypeError(f"{where} must be a JSON array of 16 rational strings")
    if any(not isinstance(coefficient, str) for coefficient in raw):
        raise TypeError(f"{where} must contain only rational strings")
    try:
        return exact.value(raw)
    except (TypeError, ValueError) as error:
        raise type(error)(f"invalid {where}: {error}") from error


def _node(raw: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{where} must be a JSON object")
    return raw


def expression_from_json(raw: Any) -> Expression:
    """Parse an external JSON-compatible document into typed expression nodes."""

    node = _node(raw, where="expression")
    operation = node.get("op")
    if operation == "literal":
        return Literal(value_from_json(node.get("value"), where="literal value"))
    if operation == "conj":
        return Conjugate(expression_from_json(_node(node.get("arg"), where="conj.arg")))
    if operation in {"add", "sub", "mul"}:
        left = expression_from_json(
            _node(node.get("left"), where=f"{operation}.left")
        )
        right = expression_from_json(
            _node(node.get("right"), where=f"{operation}.right")
        )
        node_type = {"add": Add, "sub": Subtract, "mul": Multiply}[operation]
        return node_type(left, right)
    raise ValueError(f"unsupported expression operation: {operation!r}")


def expression_to_json(expression: Expression) -> JsonObject:
    """Encode a typed expression as a JSON-compatible object."""

    if isinstance(expression, Literal):
        return {"op": "literal", "value": value_to_json(expression.value)}
    if isinstance(expression, Conjugate):
        return {"op": "conj", "arg": expression_to_json(expression.argument)}
    if isinstance(expression, (Add, Subtract, Multiply)):
        operation = {Add: "add", Subtract: "sub", Multiply: "mul"}[type(expression)]
        return {
            "op": operation,
            "left": expression_to_json(expression.left),
            "right": expression_to_json(expression.right),
        }
    raise TypeError(f"expression must be a typed node; got {type(expression).__name__}")


def literal(operand: exact.Value) -> JsonObject:
    """Build a legacy JSON-compatible exact literal node."""

    return expression_to_json(Literal(operand))


def run_to_json(
    initial: exact.Value,
    events: Program | Sequence[exact.Value],
    result: Completed,
) -> JsonObject:
    """Encode a completed exact run as a JSON-compatible object."""

    initial = exact.checked(initial, where="initial state")
    program = as_program(events)
    return {
        "initial": value_to_json(initial),
        "events": [value_to_json(event) for event in program.events],
        "state": value_to_json(result.state),
        "trace": [
            {
                "event": value_to_json(step.event),
                "before": value_to_json(step.before),
                "after": value_to_json(step.after),
                "norm2": _fraction_json(squared_norm(step)),
            }
            for step in result.trace
        ],
    }


def serialize_run(
    initial: exact.Value,
    events: Program | Sequence[exact.Value],
) -> str:
    """Serialize an ordered run, including its exact endpoint and trace."""

    program = as_program(events)
    result = run(initial, program)
    return json.dumps(run_to_json(initial, program, result), indent=2, sort_keys=True)


def replay_serialized(document: str) -> Completed:
    """Replay a serialized run and reject any altered endpoint or trace."""

    try:
        raw = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise TypeError("serialized run must contain a JSON object")
    initial = value_from_json(raw.get("initial"), where="initial")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("events must be a JSON array")
    program = Program(
        [
            value_from_json(event, where=f"events[{index}]")
            for index, event in enumerate(raw_events)
        ]
    )
    replayed = run(initial, program)
    expected = run_to_json(initial, program, replayed)
    if raw != expected:
        raise ValueError("serialized endpoint or trace does not match exact replay")
    return replayed

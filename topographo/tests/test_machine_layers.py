import dataclasses
from fractions import Fraction

import pytest

from topographo.core.exact import ExactCayleyDicksonAlgebra
from topographo.ssd import (
    codec,
    exact,
    exact_machine,
    machine,
    observers,
    program,
    projective,
)


def test_core_exact_backend_uses_shared_table() -> None:
    algebra = ExactCayleyDicksonAlgebra(4)
    e1, e2 = algebra.basis(1), algebra.basis(2)
    assert algebra.mul(e1, e1) == algebra.scale(-1, algebra.one())
    assert algebra.mul(e1, e2) == algebra.basis(3)


def test_typed_expression_parser_preserves_parentheses() -> None:
    x, y, z = exact.basis(1), exact.basis(2), exact.basis(4)
    left = program.Multiply(program.Multiply(program.Literal(x), program.Literal(y)), program.Literal(z))
    right_data = {
        "op": "mul",
        "left": codec.literal(x),
        "right": {
            "op": "mul",
            "left": codec.literal(y),
            "right": codec.literal(z),
        },
    }
    right = codec.expression_from_json(right_data)
    assert program.evaluate(left) == exact.mul(exact.mul(x, y), z)
    assert program.evaluate(right) == exact.mul(x, exact.mul(y, z))
    assert program.evaluate(left) != program.evaluate(right)


def test_ordered_program_is_repeated_left_action() -> None:
    initial, first, second = exact.basis(4), exact.basis(2), exact.basis(1)
    ordered = machine.run(initial, program.Program([first, second]))
    collapsed = exact.mul(exact.mul(second, first), initial)
    assert ordered.state == exact.mul(second, exact.mul(first, initial))
    assert ordered.state != collapsed


def test_raw_and_projective_runs_use_shared_engine() -> None:
    initial = exact.value([1, 2] + [0] * 14)
    events = program.Program([exact.value([0, 1, 1] + [0] * 13), exact.basis(4)])
    assert machine.run(initial, events) == machine.execute(
        initial, events, machine.RawVectorPolicy()
    )
    assert projective.run_projective(initial, events) == machine.execute(
        initial, events, projective.ProjectivePolicy()
    )


def test_projective_equivalence_includes_negative_scalars() -> None:
    value = exact.value([1, -2, 3] + [0] * 13)
    assert projective.equivalent(value, exact.scale(Fraction(5, 7), value))
    assert projective.equivalent(value, exact.scale(-3, value))
    with pytest.raises(ValueError, match="zero vector"):
        projective.canonicalize(exact.zero())


def test_projective_zero_and_zero_divisor_annihilation() -> None:
    initial = exact.value([1, 2] + [0] * 14)
    zero_event = projective.run_projective(initial, [exact.zero()])
    assert isinstance(zero_event, machine.Annihilated)
    assert zero_event.step == 1
    assert zero_event.trace[-1].after == exact.zero()

    divisor_x = exact.add(exact.basis(1), exact.basis(10))
    divisor_y = exact.sub(exact.basis(4), exact.basis(15))
    assert exact.mul(divisor_x, divisor_y) == exact.zero()
    zero_divisor = projective.run_projective(divisor_y, [divisor_x])
    assert isinstance(zero_divisor, machine.Annihilated)
    assert zero_divisor.step == 1


def test_raw_execution_continues_through_zero() -> None:
    result = machine.run(exact.basis(7), [exact.zero(), exact.basis(3)])
    assert isinstance(result, machine.Completed)
    assert len(result.trace) == 2
    assert result.trace[1].before == exact.zero()


def test_transition_observations_and_outcome_aliases_are_not_duplicated() -> None:
    result = machine.run(exact.basis(2), [exact.basis(1)])
    transition = result.trace[0]
    assert [field.name for field in dataclasses.fields(transition)] == [
        "event",
        "before",
        "after",
    ]
    assert transition.norm2 == exact.norm2(transition.after)
    assert observers.squared_norm(transition) == transition.norm2
    assert exact_machine.TraceStep(
        transition.event,
        transition.before,
        transition.after,
        transition.norm2,
    ) == transition
    with pytest.raises(ValueError, match="norm2 does not match"):
        exact_machine.TraceStep(
            transition.event,
            transition.before,
            transition.after,
            transition.norm2 + 1,
        )
    assert exact_machine.RunResult is machine.Completed
    assert exact_machine.RayResult is machine.Completed
    assert exact_machine.TraceStep is machine.Transition


def test_algebra_and_machine_layers_do_not_import_json() -> None:
    assert "json" not in vars(exact)
    assert "json" not in vars(machine)
    assert "json" not in vars(projective)

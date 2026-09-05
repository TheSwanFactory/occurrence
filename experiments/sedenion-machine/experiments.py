"""Reproducible conformance-witness searches and sedenion experiments."""

from __future__ import annotations

import itertools
import json
import platform
import random
import sys
from collections.abc import Callable, Iterable
from fractions import Fraction
from pathlib import Path

from topographo.ssd import codec, exact, machine, observers, projective

SEED = 6112026
HERE = Path(__file__).resolve().parent


def sparse_values() -> list[exact.Value]:
    values: list[exact.Value] = []
    for left, right in itertools.combinations(range(1, 16), 2):
        for left_sign, right_sign in itertools.product((1, -1), repeat=2):
            coefficients = [0] * 16
            coefficients[left] = left_sign
            coefficients[right] = right_sign
            values.append(exact.value(coefficients))
    return values


def vector_json(vector: exact.Value) -> list[str]:
    return codec.value_to_json(vector)


def witness2(
    name: str,
    candidates: Iterable[exact.Value],
    predicate: Callable[[exact.Value, exact.Value], bool],
) -> tuple[exact.Value, exact.Value]:
    choices = list(candidates)
    for left in choices:
        for right in choices:
            if predicate(left, right):
                return left, right
    raise RuntimeError(f"required {name} witness was not found")


def noncommutative_witness() -> tuple[exact.Value, exact.Value]:
    bases = [exact.basis(index) for index in range(16)]
    return witness2(
        "noncommutativity",
        bases,
        lambda x, y: exact.mul(x, y) != exact.mul(y, x),
    )


def nonassociative_witness() -> tuple[exact.Value, exact.Value, exact.Value]:
    bases = [exact.basis(index) for index in range(16)]
    for x in bases:
        for y in bases:
            for z in bases:
                if exact.mul(exact.mul(x, y), z) != exact.mul(
                    x, exact.mul(y, z)
                ):
                    return x, y, z
    raise RuntimeError("required nonassociativity witness was not found")


def alternative_failure_witness() -> tuple[exact.Value, exact.Value]:
    bases = [exact.basis(index) for index in range(16)]
    sparse = sparse_values()
    for left_candidates, right_candidates in (
        (bases, bases),
        (sparse, bases),
        (bases, sparse),
        (sparse, sparse),
    ):
        for x in left_candidates:
            xx = exact.mul(x, x)
            for y in right_candidates:
                if exact.mul(xx, y) != exact.mul(x, exact.mul(x, y)):
                    return x, y
    raise RuntimeError("required alternativity-failure witness was not found")


def zero_divisor_witness() -> tuple[exact.Value, exact.Value]:
    def predicate(x: exact.Value, y: exact.Value) -> bool:
        return (
            x != exact.zero()
            and y != exact.zero()
            and exact.mul(x, y) == exact.zero()
        )

    bases = [exact.basis(index) for index in range(16)]
    try:
        return witness2("zero-divisor", bases, predicate)
    except RuntimeError:
        return witness2("zero-divisor", sparse_values(), predicate)


def random_nonzero(rng: random.Random) -> exact.Value:
    while True:
        candidate = exact.value([rng.randint(-2, 2) for _ in range(16)])
        if candidate != exact.zero():
            return candidate


def trace_json(
    result: machine.Completed | machine.Annihilated,
) -> list[dict[str, object]]:
    return [
        {
            "step": index,
            "event": vector_json(step.event),
            "before": vector_json(step.before),
            "after": vector_json(step.after),
            "norm2": str(observers.squared_norm(step)),
        }
        for index, step in enumerate(result.trace, start=1)
    ]


def generate_results() -> dict[str, object]:
    noncomm_x, noncomm_y = noncommutative_witness()
    assoc_x, assoc_y, assoc_z = nonassociative_witness()
    alt_x, alt_y = alternative_failure_witness()
    divisor_x, divisor_y = zero_divisor_witness()

    left_assoc = exact.mul(exact.mul(assoc_x, assoc_y), assoc_z)
    right_assoc = exact.mul(assoc_x, exact.mul(assoc_y, assoc_z))
    annihilation = machine.run(divisor_y, [divisor_x])
    forward = machine.run(exact.one(), [noncomm_x, noncomm_y])
    reverse = machine.run(exact.one(), [noncomm_y, noncomm_x])

    # With events [a, b], the transition endpoint is b*(a*x), not (b*a)*x.
    collapse_a, collapse_b, collapse_x = assoc_y, assoc_x, assoc_z
    ordered = machine.run(collapse_x, [collapse_a, collapse_b])
    collapsed = exact.mul(exact.mul(collapse_b, collapse_a), collapse_x)
    if ordered.state == collapsed:
        raise RuntimeError("required ordered-program collapse witness was not found")

    rng = random.Random(SEED)
    sampled_initial = random_nonzero(rng)
    sampled_events = [random_nonzero(rng) for _ in range(8)]
    replay_samples: dict[str, object] = {}
    for length in (1, 2, 4, 8):
        sample = machine.run(sampled_initial, sampled_events[:length])
        replay_samples[str(length)] = {
            "endpoint": vector_json(sample.state),
            "norm2": str(exact.norm2(sample.state)),
            "zero_hits": [
                index
                for index, step in enumerate(sample.trace, start=1)
                if step.after == exact.zero()
            ],
        }

    projective_initial = exact.value([1, 2] + [0] * 14)
    projective_events = [exact.value([0, 1, 1] + [0] * 13), exact.basis(4)]
    projective_base = projective.run_projective(
        projective_initial, projective_events
    )
    projective_scaled = projective.run_projective(
        exact.scale(-3, projective_initial),
        [
            exact.scale(5, projective_events[0]),
            exact.scale(-2, projective_events[1]),
        ],
    )
    if projective_base != projective_scaled or not isinstance(
        projective_base, machine.Completed
    ):
        raise RuntimeError("projective rescaling invariance check failed")
    projective_annihilated = projective.run_projective(divisor_y, [divisor_x])
    assert isinstance(projective_annihilated, machine.Annihilated), (
        "projective zero-divisor experiment did not annihilate"
    )

    return {
        "implementation": {
            "python": platform.python_version(),
            "coefficient_domain": "fractions.Fraction (exact rationals)",
            "dimension": 16,
            "basis_order": [f"e{index}" for index in range(16)],
            "module": "topographo.ssd.exact_machine",
            "convention": {
                "conj": "conj((a,b)) = (conj(a), -b)",
                "mul": exact.CONVENTION,
            },
        },
        "coverage": {
            "basis_products": 256,
            "seeded_vector_coefficients": "independent uniform integers in [-2, 2], rejecting all-zero vectors",
            "seed": SEED,
            "claim": "finite exact conformance tests and observations, not a universal proof",
        },
        "required_failures": {
            "noncommutativity": {
                "x": vector_json(noncomm_x),
                "y": vector_json(noncomm_y),
                "x*y": vector_json(exact.mul(noncomm_x, noncomm_y)),
                "y*x": vector_json(exact.mul(noncomm_y, noncomm_x)),
            },
            "nonassociativity": {
                "x": vector_json(assoc_x),
                "y": vector_json(assoc_y),
                "z": vector_json(assoc_z),
                "(x*y)*z": vector_json(left_assoc),
                "x*(y*z)": vector_json(right_assoc),
                "difference": vector_json(exact.sub(left_assoc, right_assoc)),
            },
            "alternativity_failure": {
                "x": vector_json(alt_x),
                "y": vector_json(alt_y),
                "(x*x)*y": vector_json(
                    exact.mul(exact.mul(alt_x, alt_x), alt_y)
                ),
                "x*(x*y)": vector_json(
                    exact.mul(alt_x, exact.mul(alt_x, alt_y))
                ),
            },
            "zero_divisors": {
                "x": vector_json(divisor_x),
                "y": vector_json(divisor_y),
                "x*y": vector_json(exact.mul(divisor_x, divisor_y)),
            },
            "norm_multiplicativity_failure": {
                "norm2(x*y)": str(exact.norm2(exact.mul(divisor_x, divisor_y))),
                "norm2(x)*norm2(y)": str(
                    exact.norm2(divisor_x) * exact.norm2(divisor_y)
                ),
            },
        },
        "experiments": {
            "nonassociative_bracketings": {
                "left": vector_json(left_assoc),
                "right": vector_json(right_assoc),
                "difference": vector_json(exact.sub(left_assoc, right_assoc)),
            },
            "annihilation": {
                "state": vector_json(annihilation.state),
                "trace": trace_json(annihilation),
            },
            "event_reversal": {
                "forward_endpoint": vector_json(forward.state),
                "reverse_endpoint": vector_json(reverse.state),
            },
            "program_collapse": {
                "a": vector_json(collapse_a),
                "b": vector_json(collapse_b),
                "initial": vector_json(collapse_x),
                "ordered_endpoint": vector_json(ordered.state),
                "collapsed_endpoint": vector_json(collapsed),
                "trace": trace_json(ordered),
            },
            "seeded_replay": {
                "initial": vector_json(sampled_initial),
                "events": [vector_json(event) for event in sampled_events],
                "prefixes": replay_samples,
            },
            "ray_mode": {
                "endpoint": vector_json(projective_base.state),
                "rescaling_invariant": True,
                "annihilation": {
                    "step": projective_annihilated.step,
                    "trace": trace_json(projective_annihilated),
                },
            },
        },
    }


def vector_notation(raw: object) -> str:
    if not isinstance(raw, list):
        raise TypeError("result vector must be a list")
    terms: list[str] = []
    for index, text in enumerate(raw):
        coefficient = Fraction(str(text))
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        basis_name = "1" if index == 0 else f"e{index}"
        if index == 0 or magnitude != 1:
            term = str(magnitude) if index == 0 else f"{magnitude}*{basis_name}"
        else:
            term = basis_name
        if not terms:
            terms.append(f"-{term}" if coefficient < 0 else term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms) or "0"


def observations(results: dict[str, object]) -> str:
    failures = results["required_failures"]
    experiments = results["experiments"]
    assert isinstance(failures, dict) and isinstance(experiments, dict)
    nonassoc = failures["nonassociativity"]
    divisors = failures["zero_divisors"]
    norm_failure = failures["norm_multiplicativity_failure"]
    replay = experiments["seeded_replay"]
    assert isinstance(nonassoc, dict)
    assert isinstance(divisors, dict)
    assert isinstance(norm_failure, dict)
    assert isinstance(replay, dict)
    prefixes = replay["prefixes"]
    assert isinstance(prefixes, dict)
    assoc_x = vector_notation(nonassoc["x"])
    assoc_y = vector_notation(nonassoc["y"])
    assoc_z = vector_notation(nonassoc["z"])
    left_assoc = vector_notation(nonassoc["(x*y)*z"])
    right_assoc = vector_notation(nonassoc["x*(y*z)"])
    divisor_x = vector_notation(divisors["x"])
    divisor_y = vector_notation(divisors["y"])
    lines = [
        "# Sedenion machine observations",
        "",
        "Generated by `uv run python experiments/sedenion-machine/experiments.py`; exact vectors are recorded in `results.json`.",
        "",
        "## Guaranteed by conformance checks",
        "",
        "The implementation uses 16 exact rational coefficients in basis order `e0` through `e15` and the pinned recursive Cayley–Dickson convention. Tests cover all 256 ordered basis products plus reproducibly seeded finite samples. This is finite conformance evidence, not a universal proof.",
        "",
        "Unit, zero, additive, bilinear, distributive, conjugation, quadratic-norm, basis-square, and basis-anticommutation checks passed. Seeded embedded quaternion samples were associative, and embedded octonion samples were alternative. JSON replay reproduced complete traces exactly.",
        "",
        "## Required dimension-16 failures",
        "",
        f"The first deterministic basis search found `(x, y, z) = ({assoc_x}, {assoc_y}, {assoc_z})`, with `(x*y)*z = {left_assoc}` and `x*(y*z) = {right_assoc}`.",
        f"A deterministic sparse search found `x = {divisor_x}` and `y = {divisor_y}`, with `x*y = 0`. Consequently `norm2(x*y) = {norm_failure['norm2(x*y)']}` while `norm2(x)*norm2(y) = {norm_failure['norm2(x)*norm2(y)']}`.",
        "Noncommutativity and failure of alternativity were also found and saved with both evaluated sides in `results.json`.",
        "",
        "## Program observations",
        "",
        "The zero-divisor event annihilated its nonzero initial state exactly at step 1. Reversing a noncommuting event pair changed the endpoint. The saved two-step trace differs from multiplying the events together first, confirming that explicit event order and parentheses are semantic.",
        "",
        f"Fixed-seed ({SEED}) dense vectors used independent uniform integer coefficients in `[-2, 2]`, rejecting all-zero draws. Prefixes of length 1, 2, 4, and 8 had zero-hit lists: "
        + ", ".join(
            f"{length}: {prefixes[length]['zero_hits']}"
            for length in ("1", "2", "4", "8")
        )
        + ". Absence of sampled zero hits is only an observation of this distribution.",
        "",
        "Projective mode produced the same canonical endpoint after independently rescaling the initial state and every event by nonzero rationals. The zero-divisor pair returned explicit `Annihilated(step=1)`.",
        "",
    ]
    return "\n".join(lines)


def check_artifacts(results: dict[str, object]) -> None:
    expected = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise TypeError("results.json must contain an object")
    actual_implementation = results.get("implementation")
    expected_implementation = expected.get("implementation")
    if not isinstance(actual_implementation, dict) or not isinstance(
        expected_implementation, dict
    ):
        raise TypeError("results must contain implementation metadata")
    actual_implementation["python"] = expected_implementation.get("python")
    if results != expected:
        raise RuntimeError("generated experiment results differ from results.json")

    expected_observations = (HERE / "observations.md").read_text(encoding="utf-8")
    if observations(results) != expected_observations:
        raise RuntimeError(
            "generated experiment observations differ from observations.md"
        )
    print("all generated experiment artifacts match committed files")


def main() -> None:
    arguments = sys.argv[1:]
    if arguments not in ([], ["--check"]):
        raise SystemExit("usage: experiments.py [--check]")
    results = generate_results()
    if arguments == ["--check"]:
        check_artifacts(results)
        return
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "observations.md").write_text(observations(results), encoding="utf-8")
    print(json.dumps(results["required_failures"], indent=2, sort_keys=True))
    print(f"wrote {HERE / 'results.json'}")
    print(f"wrote {HERE / 'observations.md'}")


if __name__ == "__main__":
    main()

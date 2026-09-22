from __future__ import annotations

from fractions import Fraction

import pytest

from decision_model import Distribution, Test
from decision_model._zero import (
    ZERO_CASES,
    ZERO_TEST,
    build_zero_payload,
    canonical_zero_bytes,
    main,
    resolve_zero_cases,
)

EXPECTED_PAYLOAD = {
    "schema": "decision-model-zero/v1",
    "backend": "generic-exact",
    "test": {
        "state_domain": "p in Q, 0 <= p <= 1",
        "effects": [
            {"label": "yes", "id": "identity", "expression": "p"},
            {"label": "no", "id": "complement", "expression": "1 - p"},
        ],
    },
    "cases": [
        {
            "id": "sharp",
            "state": {"numerator": 1, "denominator": 1},
            "distribution": [
                {
                    "label": "yes",
                    "probability": {"numerator": 1, "denominator": 1},
                },
                {
                    "label": "no",
                    "probability": {"numerator": 0, "denominator": 1},
                },
            ],
            "total": {"numerator": 1, "denominator": 1},
        },
        {
            "id": "non-sharp",
            "state": {"numerator": 1, "denominator": 3},
            "distribution": [
                {
                    "label": "yes",
                    "probability": {"numerator": 1, "denominator": 3},
                },
                {
                    "label": "no",
                    "probability": {"numerator": 2, "denominator": 3},
                },
            ],
            "total": {"numerator": 1, "denominator": 1},
        },
    ],
}


def test_zero_resolves_required_cases_exactly_through_the_public_contract() -> None:
    resolved = resolve_zero_cases()

    assert isinstance(ZERO_TEST, Test)
    assert tuple(case.identifier for case in resolved) == ("sharp", "non-sharp")
    assert tuple(case.state for case in resolved) == (Fraction(1), Fraction(1, 3))
    assert all(isinstance(case.distribution, Distribution) for case in resolved)
    assert tuple(
        tuple((label, probability.value) for label, probability in case.distribution)
        for case in resolved
    ) == (
        (("yes", Fraction(1)), ("no", Fraction(0))),
        (("yes", Fraction(1, 3)), ("no", Fraction(2, 3))),
    )
    assert all(
        isinstance(probability.value, Fraction)
        for case in resolved
        for _, probability in case.distribution
    )
    assert all(case.distribution.total == Fraction(1) for case in resolved)


@pytest.mark.parametrize(
    "state",
    (Fraction(0), Fraction(1, 3), Fraction(1)),
)
def test_shared_zero_test_has_the_exact_complement_identity(state: Fraction) -> None:
    yes = ZERO_TEST["yes"](state)
    no = ZERO_TEST["no"](state)

    assert isinstance(yes, Fraction)
    assert isinstance(no, Fraction)
    assert yes + no == Fraction(1)


def test_zero_payload_is_the_complete_normative_document() -> None:
    assert build_zero_payload() == EXPECTED_PAYLOAD
    assert [case["id"] for case in build_zero_payload()["cases"]] == [
        "sharp",
        "non-sharp",
    ]
    assert all(case.state != Fraction(0) for case in ZERO_CASES)


def test_zero_serialization_is_canonical_and_repeatable() -> None:
    expected = (
        '{"schema":"decision-model-zero/v1","backend":"generic-exact",'
        '"test":{"state_domain":"p in Q, 0 <= p <= 1","effects":['
        '{"label":"yes","id":"identity","expression":"p"},'
        '{"label":"no","id":"complement","expression":"1 - p"}]},'
        '"cases":[{"id":"sharp","state":{"numerator":1,"denominator":1},'
        '"distribution":[{"label":"yes","probability":{"numerator":1,'
        '"denominator":1}},{"label":"no","probability":{"numerator":0,'
        '"denominator":1}}],"total":{"numerator":1,"denominator":1}},'
        '{"id":"non-sharp","state":{"numerator":1,"denominator":3},'
        '"distribution":[{"label":"yes","probability":{"numerator":1,'
        '"denominator":3}},{"label":"no","probability":{"numerator":2,'
        '"denominator":3}}],"total":{"numerator":1,"denominator":1}}]}\n'
    ).encode("ascii")

    assert canonical_zero_bytes() == expected
    assert canonical_zero_bytes() == canonical_zero_bytes()


def test_zero_main_writes_one_document_and_no_stderr(
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    assert main() == 0
    captured = capsysbinary.readouterr()
    assert captured.out == canonical_zero_bytes()
    assert captured.err == b""

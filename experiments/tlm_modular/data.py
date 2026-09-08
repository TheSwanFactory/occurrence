"""Modular triples + alternate-role protocol (005.07 section 3)."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Literal

Role = Literal['a', 'b', 'c']


def canonical_triple(a: int, b: int, c: int, *, p: int) -> tuple[int, int, int]:
    """Canonical key for a complete modular triple under a+b == c (mod p) (mod p)."""
    return (min(a % p, b % p), max(a % p, b % p), c % p)


def all_forward_equations(p: int) -> list[tuple[int, int, int]]:
    return [(a, b, (a + b) % p) for a, b in itertools.product(range(p), repeat=2)]


def complete_triples(p: int) -> list[tuple[int, int, int]]:
    keys: dict[tuple[int, int, int], None] = {}
    for a, b, c in all_forward_equations(p):
        keys[canonical_triple(a, b, c, p=p)] = None
    return sorted(keys.keys())


@dataclass(frozen=True)
class RoleExample:
    role: Role
    a: int
    b: int
    c: int
    inputs: tuple[int, int]
    target: int
    role_tag: int


@dataclass(frozen=True)
class ModularSplit:
    p: int
    all_triples: tuple[tuple[int, int, int], ...]
    train_triples: tuple[tuple[int, int, int], ...]
    test_triples: tuple[tuple[int, int, int], ...]
    forward_train: tuple[tuple[int, int, int], ...]
    forward_test: tuple[tuple[int, int, int], ...]
    alternate_role_test: tuple[RoleExample, ...]


def _forwards_for_triple_key(key: tuple[int, int, int], p: int) -> list[tuple[int, int, int]]:
    out = []
    for a, b, c in all_forward_equations(p):
        if canonical_triple(a, b, c, p=p) == key:
            out.append((a, b, c))
    return out


def _role_examples_for_forward(a: int, b: int, c: int) -> list[RoleExample]:
    return [
        RoleExample('c', a, b, c, (a, b), c, 0),
        RoleExample('b', a, b, c, (a, c), b, 1),
        RoleExample('a', a, b, c, (b, c), a, 2),
    ]


def partition_modular_triples(*, p: int, train_fraction: float = 0.5, seed: int = 0) -> ModularSplit:
    """Partition by complete modular triples before rendering role views."""
    triples = complete_triples(p)
    rng = random.Random(seed)
    shuffled = list(triples)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    train_keys = tuple(shuffled[:n_train])
    test_keys = tuple(shuffled[n_train:])
    forward_train = tuple(eq for key in train_keys for eq in _forwards_for_triple_key(key, p))
    forward_test = tuple(eq for key in test_keys for eq in _forwards_for_triple_key(key, p))
    alt: list[RoleExample] = []
    for key in test_keys:
        a, b, c = _forwards_for_triple_key(key, p)[0]
        alt.extend(_role_examples_for_forward(a, b, c))
    return ModularSplit(
        p=p,
        all_triples=tuple(triples),
        train_triples=train_keys,
        test_triples=test_keys,
        forward_train=forward_train,
        forward_test=forward_test,
        alternate_role_test=tuple(alt),
    )


def relation_transfer_gap(forward_acc: float, alternate_role_acc: float) -> float:
    return forward_acc - alternate_role_acc

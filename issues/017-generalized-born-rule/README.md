# Issue 017 — Reusable OT generalized Born rule

- **Opened:** 2026-09-21
- **Closed:** 2026-09-21
- **Status:** closed
- **Resolution:** the established single-step OT Born transport identity is a
  reusable `topographo` API; the separate backend-neutral Decision Model
  contract is implemented as the `decision-model` distribution
- **Namespace:** this repository issue is distinct from the **Outcome 017**
  Futurator experiment lineage under `experiments/tlm_fixed_head/`

## Question

Does the repository already implement the Occurrence Theory generalized Born
rule, and if so, what remains to make it a reusable and scientifically honest
part of `topographo`?

## Answer

**Yes, narrowly.** Release 1.0.0 promotes the already-certified
single-transition OT Born transport identity into the public
`topographo.ot_born_transport` API. It also introduces the independent
backend-neutral `decision_model` import contract in the `decision-model`
distribution.

Three claims remain separate:

```text
ESTABLISHED  exact single-step OT Born transport identity
IMPLEMENTED  backend-neutral software contract for typed effect/test resolution
OPEN         complete generalized OT measurement/probability law
```

The implementation does not derive outcome families, composition,
interference, conditioning, projection, or collapse.

## Canonical identity

OT-II defines, for a unit retained state `x` and sampled event `a`,

```text
K_a             settlement operator
z_a = K_a e0    event recovered from the operator
J               canonical orthogonal complex structure
x'              K_a x / ||K_a x||
tau_a(x)        ||K_a x||^2 - 1
s(u)            (u·e0)^2 + (u·J e0)^2
A(z_a, x)       (z_a·x)^2 + ((J z_a)·x)^2
```

and proves

```text
s(x') (1 + tau_a(x)) = A(z_a, x),
```

or equivalently

```text
s(x') = A(z_a, x) / ||K_a x||^2
```

whenever `K_a x` is nonzero. The canonical statement and its physical caveat
are in [OT-II §4.2](../../occurrence-theory-ii.md#42-the-born-quotient). The
exact mathematical result is the transport identity; interpreting its numerator
as physical Born probability remains a reading rather than a theorem.

## Implemented API

[`topographo.born_transport`](../../topographo/born_transport.py) now provides:

```text
ot_born_transport
BornTransportResult
BornTransportAnnihilation
```

The function validates a real finite unit state and compatible real finite
16-dimensional settlement and complex-structure matrices. Compatibility means
operator and complex-structure antisymmetry, `J^2 = -I`, unit event recovery,
and the axis relation that forces the transport identity. Arbitrary same-shaped
matrices are rejected. Successful results expose the transported state,
normalization cost, strain, post-transition spine share, Hermitian numerator,
and signed identity residual. Exact and numerical near-annihilation are typed
before division, with a default squared-cost threshold of `1e-12` and an
explicit exact-only mode.

The canonical [`occurrence_ii_audit.py`](../../verify/occurrence_ii_audit.py)
consumes the public API. The Codex and Solomon/Joseph rechecks remain independent.
Deterministic tests cover all 84 committed events and all contract boundaries.

## Adjacent code that remains distinct

[`topographo.ssd.fixed_head`](../../topographo/ssd/fixed_head.py) exposes a
configured Theory-27 two-step quadratic readout:

```text
P(w | [x]) = x^T B_w x / x^T x.
```

It is not the OT-II quotient: it does not use `J`, spine share, or event strain.
Neither API is an alias or theorem claim for the other.

## Decision Model contract

[`017.02`](017.02-decision-model-package-contract.md) implements an independent
package:

```text
pip distribution: decision-model
Python import:     decision_model

ResolveEffect : State x Effect -> Probability
ResolveTest   : State x Test   -> Distribution
```

The generic package has no runtime dependency on `topographo`. A future backend
may consume only public `topographo` APIs; reverse dependency is forbidden and
mechanically tested. Preparation, test interpretation, decision policy, and
Event enactment remain outside resolver semantics.

## Numbered execution sequence

1. [`017.01`](017.01-topographo-born-transport-api.md) — **complete**: reusable
   OT Born transport API under `topographo`.
2. [`017.02`](017.02-decision-model-package-contract.md) — **complete**:
   independent backend-neutral Decision Model 1.0.0 package contract.
3. [`017.03`](017.03-decision-model-zero-exact-nonlinguistic-demo.md) —
   **complete in 1.1.0**: exact, nonlinguistic, deterministic Decision Model
   Zero demonstration and installed command.
4. [`017.04`](017.04-decision-model-topographo-backend.md) — next: thin
   `topographo` backend without overstating the theorem.
5. [`017.05`](017.05-preparation-and-test-bridges.md) — later: preparation and
   test-interpretation bridges, characterized separately.

Steps 017.04–017.05 remain successor package and research work. They are not
conditions for this narrow issue's closure. The separate GPT 017.17 audit uses
Decision Model Zero as an exact fixed-test research control without claiming
that the planned backend or preparation/test bridges are complete.

## Acceptance record

- The reusable implementation lives under `topographo`.
- It reproduces Theorem 4.3 across the committed Kraus-84 family.
- Exact and near annihilation have explicit tested behavior.
- The canonical audit calls the package implementation and preserves its
  certificate.
- Independent audit implementations remain independent and pass.
- Documentation distinguishes the quotient from Fixed-head readout.
- The Decision Model contract is independently buildable and dependency-fenced.

## Scientific fences

This issue establishes:

```text
the forced single-step identity has a reusable, robust implementation
the implementation agrees with the committed OT event/operator family
the canonical audit and package no longer duplicate the arithmetic
the generic Decision Model software contract is explicit and tested
```

It does not establish:

```text
a normalized physical outcome family
multi-step composition or path-sum interference
Born-rule measurement, projection, or collapse
POVM or density-matrix semantics
physical probability, energy, or temperature interpretations
equivalence with the configured Theory-27 Fixed head
universality beyond the committed OT/Kraus-84 construction
intelligent evaluation or learned preparation/test interpretation
```

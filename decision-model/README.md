# Decision Model

**Decision Model** is a small, backend-neutral Python contract for evaluating
already-constituted states against explicit effects and finite exhaustive tests.

```text
ResolveEffect : State × Effect -> Probability
ResolveTest   : State × Test   -> Distribution
```

Install the `decision-model` distribution and import `decision_model`:

```console
python -m pip install decision-model
```

Version 1.1.0 supports Python 3.11 and later and is released under the MIT
License by The Swan Factory.

## Decision Model Zero

Version 1.1.0 adds the exact, nonlinguistic Decision Model Zero demonstration.
A new user can install and run it directly:

```console
python -m pip install decision-model
decision-model-zero
```

The command prints one deterministic `decision-model-zero/v1` JSON document.
It uses exact rational arithmetic and resolves these ordered fixtures:

```text
sharp:     p = 1   -> ((yes, 1),   (no, 0))
non-sharp: p = 1/3 -> ((yes, 1/3), (no, 2/3))
```

The admitted state is an exact rational `p` with `0 <= p <= 1`. The one shared
ordered test has the identity effect `e_yes(p) = p` and complement effect
`e_no(p) = 1 - p`. It is exhaustive throughout that declared domain because

```text
e_yes(p) + e_no(p) = p + (1 - p) = 1.
```

For each fixture, the state remains a `fractions.Fraction`; the fixed public
`Test` is resolved by public `resolve_test`; and the resulting ordered
`Distribution`, probabilities, and total remain exact. The JSON presentation
encodes every rational as reduced integer `numerator` and positive
`denominator` fields. It does not use floats, tolerance, renormalization, or
sampling.

Here, **state** is the exact value being evaluated; an **effect** returns one
probability from that state; the **test** is the ordered exhaustive `yes`/`no`
family; and the **distribution** is the exact unit-total result. Calling this
family a test is a domain assertion justified by the displayed complement
identity. It is not a proof that arbitrary effect families are tests.

### Decision Model Zero limitations

Decision Model Zero does not provide state preparation, test interpretation,
arbitrary test-admissibility proofs, backend physics, `topographo` transport, a
generalized OT measurement law, conditioning, collapse, sampling, decision
policy, or intelligent evaluation. It also contains no language model, learned
component, Event enactment, or claim that its elementary complement test is a
physical Born measurement.

## Contract

The stable top-level API is deliberately compact:

- `State` is a generic type parameter. Decision Model does not prescribe a
  state representation.
- `Effect[State]` is a runtime-checkable callable protocol. An effect receives
  an already-constituted state and returns a `Probability` or real value.
- `Test[State, Label]` is an immutable, nonempty, ordered family of uniquely
  labeled callable effects. Calling something a `Test` asserts that the family
  is exhaustive; backend or domain code remains responsible for establishing
  that assertion.
- `Probability` is an immutable finite real value in `[0, 1]`.
- `Distribution[Label]` is an immutable, nonempty, uniquely labeled family of
  probabilities whose total is one within its explicit `tolerance`.
- `Resolver[State, Label]` is a runtime-checkable structural protocol for
  backends exposing `resolve_effect` and `resolve_test` methods.
- `resolve_effect` and `resolve_test` provide backend-neutral callable-effect
  resolution with endpoint validation.

```python
from fractions import Fraction

from decision_model import Test, resolve_test

state = {"yes": Fraction(1, 3)}
test = Test(
    (
        ("yes", lambda value: value["yes"]),
        ("no", lambda value: 1 - value["yes"]),
    )
)

distribution = resolve_test(state, test)
assert distribution["yes"].value == Fraction(1, 3)
assert distribution.total == 1
```

`resolve_test` is strict: it does not infer exhaustiveness by normalizing a
collection of weights. `Distribution` never renormalizes. Its default tolerance
is exact (`0`); callers accepting numerical error must pass a visible
nonnegative tolerance. Values accepted within that tolerance remain unchanged.
Typed failures from an individual effect propagate unchanged, preserving
backend-specific identity and metadata; the generic resolver does not add the
outcome label to that exception.

## Failure is not probability zero

`ResolutionError` is the typed failure boundary. `BackendResolutionError`
reports effect/backend execution failure, `InvalidResolutionError` reports a
result that violates endpoint invariants, and `AnnihilationError` is reserved
for a backend that establishes annihilation. These exceptions are never
translated into `Probability(0)`. Zero remains an ordinary successful
probability and is distinguishable from failure.

## 017.17 research audit

The isolated private module `decision_model._audit_01717` and the tracked
`artifacts/017.17-born-first-cross-carrier-audit.json` implement the bounded
Born-first coding audit without changing the public API or Zero product schema.
They realize one fixed 16-dimensional diagonal projector/complement test under
multiple exact rational-ray preparations, derive the zero-support matrix, and
run the equality-complement control. The exact pinned Interact authority then
supplies a local `Q(i)` coordinate realization of `J_e^{-1}`, `rho_e`, the
`<sigma,kappa>` quotient, and `beta_e`. The finite support-incidence hypothesis
passes for fixed answer branches, but its source-to-Interact representatives
were chosen to fit the table and are classified explicitly as an arbitrary
plumbing witness. No certified relation connects those representatives to the
Born preparations or effects.

The audit does not encode `beta_e` as a lookup table, substitute surrogate
geometry, or equate zero probability with branch illegality.

This is a reproducible research control, not a new generic resolver contract or
an assertion that `decision-model` implements a generalized OT measurement
backend.

## Dependency direction

The generic 1.1.0 package uses only the Python standard library and does not
import or require `topographo`. A future backend may provide this one-way
integration:

```text
decision_model.backends.topographo  --->  topographo
```

Such an adapter may consume only public `topographo` APIs. `topographo` must
never import `decision_model`, and generic Decision Model modules must remain
independent of `topographo`. No optional backend dependency is declared until
that adapter exists. The top-level `decision-model/` directory is an
independently buildable project with the same dependency fence expected if it
is later moved to a separate repository.

## Scope and claims

### Implemented and tested

Decision Model 1.1.0 certifies its software contract: immutable validated value
objects, callable effect dispatch, normalized labeled endpoint validation,
typed failure propagation, dependency isolation, and the exact deterministic
Decision Model Zero command. These are API and software behavior claims.

The 017.17 audit separately certifies the exact fixed-test construction and its
bounded support result under its declared evidence fence. It does not enlarge
the generic package's physical claims.

### Research direction

Backend-specific mathematics—including the OT single-step Born transport
identity—must be implemented and justified by the backend that supplies it.
This generic package does not claim a generalized measurement law, arbitrary
POVMs, collapse, conditioning, path interference, or intelligent evaluation.

Preparation (`Datum -> State`), interpretation (`Question -> Effect/Test`),
sampling, argmax, thresholding, narration, policies, decisions, actions, and
Event enactment are intentionally outside resolver semantics.

## Stability

The symbols listed by `decision_model.__all__`, their documented call shapes,
and the validation/failure invariants above are the stable 1.x public API.
Private modules, demonstration internals, and undocumented implementation
details are not covered. Breaking public changes require a new major version.

## Development

From the repository root:

```console
uv lock --project decision-model --check
uv run --project decision-model --frozen ruff check \
  decision-model/src decision-model/tests
uv run --project decision-model --frozen mypy --strict \
  decision-model/src/decision_model
uv run --project decision-model --frozen pytest \
  decision-model/tests
uv build --project decision-model --out-dir decision-model-dist
```

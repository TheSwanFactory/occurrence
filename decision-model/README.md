# Decision Model

**Decision Model** is a small, backend-neutral Python contract for evaluating
already-constituted states against explicit effects and finite exhaustive tests.

```text
ResolveEffect : State × Effect -> Probability
ResolveTest   : State × Test   -> Distribution
```

Install the `decision-model` distribution and import `decision_model`:

```bash
pip install decision-model
```

Version 1.0.0 supports Python 3.11 and later and is released under the MIT
License by The Swan Factory.

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

## Dependency direction

The generic 1.0.0 package uses only the Python standard library and does not
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

Decision Model 1.0.0 certifies its software contract: immutable validated value
objects, callable effect dispatch, normalized labeled endpoint validation,
typed failure propagation, and dependency isolation. These are API and software
behavior claims.

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
Private modules and undocumented implementation details are not covered.
Breaking public changes require a new major version.

## Development

From the repository root:

```console
uv run --project decision-model --frozen ruff check \
  decision-model/src decision-model/tests
uv run --project decision-model --frozen pytest \
  decision-model/tests
uv build --project decision-model
```

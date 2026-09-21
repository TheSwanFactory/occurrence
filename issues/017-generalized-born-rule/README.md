# Issue 017 — Reusable OT generalized Born rule

- **Opened:** 2026-09-21
- **Status:** open
- **Scope:** promote the established single-step OT Born transport identity from
  verification-only code into a reusable library API, while preserving the
  boundary between that identity and a still-open generalized measurement law
- **Namespace:** this repository issue is distinct from the **Outcome 017**
  Futurator experiment lineage under `experiments/tlm_fixed_head/`

## Question

Does the repository already implement the Occurrence Theory generalized Born
rule, and if so, what remains to make it a reusable and scientifically honest
part of `topographo`?

## Current answer

**Partially.** The repository executes and numerically certifies the canonical
single-transition OT Born transport identity, but only inline in audit scripts.
It does not expose that identity as a reusable package function, measurement
object, or public API. It also does not yet derive the wider quantum probability
calculus—outcome families, composition, interference, conditioning, projection,
or collapse—that “generalized Born rule” can imply.

The implementation task must therefore keep two claims separate:

```text
ESTABLISHED  the exact single-step OT Born transport identity
OPEN         a complete generalized OT measurement/probability law
```

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
are in [OT-II §4.2](../../occurrence-theory-ii.md#42-the-born-quotient).
The text calls the numerator’s interpretation as Born probability coherent but
speculative; the exact mathematical result is the transport identity.

## Existing executable implementations

| Location | What it does | Limitation |
|---|---|---|
| [`verify/occurrence_ii_audit.py`](../../verify/occurrence_ii_audit.py), lines 203–221 | Canonical CI audit: constructs `x'`, spine share, Hermitian-overlap numerator, and certifies `s_new * nrm2 == A` over 2,000 random states | Formula is inline in `main()`; returns no reusable result; the Born loop has no explicit annihilation policy |
| [`verify/occurrence_ii_codex_sol.py`](../../verify/occurrence_ii_codex_sol.py), lines 52–58 | Independent numerical check of the same identity | Script-local audit code; skips near-zero costs |
| [`verify/occurrence_ii_solomonjoseph.py`](../../verify/occurrence_ii_solomonjoseph.py), lines 695–716 | Independent reconstruction and check; records the antisymmetry/anticommutation reduction behind the numerator | Top-level verification code, not a package API |

The canonical audit is already run as a CI gate by
[`.github/workflows/occurrence.yml`](../../.github/workflows/occurrence.yml).
That establishes executable coverage of the identity, not a reusable
implementation.

## Adjacent code that is not this rule

[`topographo.ssd.fixed_head`](../../topographo/ssd/fixed_head.py) exposes
`two_step_moment`, `word_effect`, and `readout_probability` (lines 154–175), with

```text
P(w | [x]) = x^T B_w x / x^T x.
```

This is the repository’s reusable Born-like quadratic readout, but it is a
separate configured Theory-27 two-step Fixed-head adapter. Its module-level
physical fence explicitly says that it is **not** a physical Test Realization or
Outcome family. It does not implement the OT-II quotient above: it uses neither
`J`, the spine share, nor event strain.

The Fixed-head API must not be silently renamed or cited as an implementation
of the canonical OT-II Born transport identity.

## Implementation gap

There is currently no reusable `topographo` symbol that:

1. accepts an OT state, settlement operator/event, and canonical complex
   structure;
2. computes the normalized transported state, normalization cost/strain,
   post-transition spine share, and Hermitian-overlap numerator;
3. applies an explicit policy when `K_a x = 0` or is numerically near zero;
4. exposes the exact identity in a deterministic, testable result;
5. can be used by the canonical audit without eliminating the independence of
   the two external rechecks.

There is also no package abstraction for a normalized family of outcomes,
density operators, POVMs, path amplitudes, interference, projection, or
collapse. OT-II states those extensions as open in
[Conjecture C2](../../occurrence-theory-ii.md#b7-conjectures) and
[Open Problems 6 and 9](../../occurrence-theory-ii.md#b8-open-problems-and-verification-tasks).

## Required work

### 1. Reusable single-step transport API

Add a small package-level API for the **established** theorem. Its exact module
and type names may follow the surrounding package architecture, but the public
contract must expose at least:

```text
transported state x'
normalization cost ||K_a x||^2
event strain tau = cost - 1
post-transition spine share s(x')
Hermitian numerator A(z_a, x)
identity residual s(x') * cost - A
```

The API must validate dimensions and finite inputs and must define annihilation
semantics explicitly. It must not divide by zero or rely on random continuous
sampling to avoid the kernel. A typed result object is preferred to an
unstructured tuple.

### 2. Canonical integration

Refactor only the canonical CI audit to consume the reusable implementation.
Keep the Codex and Solomon/Joseph scripts independent so they remain meaningful
rechecks rather than three callers of the same code.

### 3. Deterministic verification

Add focused tests for:

- the theorem across the committed 84-event family;
- scale handling or unit-state preconditions, whichever contract is chosen;
- the exact/near-annihilation policy;
- agreement with the current canonical audit formula;
- malformed dimensions and non-finite values.

Randomized coverage may supplement but must not replace deterministic cases.

### 4. Documentation and naming fence

Document the API as the **OT Born transport identity** or **single-step Born
quotient**. If the phrase “generalized Born rule” is used, immediately state
that it refers only to this established OT single-transition identity.

Do not claim a complete measurement law until composition, outcome
normalization, interference, and conditioning/collapse are separately defined
and verified.

## Acceptance criteria

- A reusable implementation lives under `topographo`, not under `verify/` or an
  experiment directory.
- It reproduces Theorem 4.3 for the committed Kraus-84 family within the
  repository’s declared numerical tolerance.
- Annihilating and near-annihilating transitions have explicit, tested behavior.
- The canonical OT-II audit calls the package implementation and preserves its
  current certificate.
- Independent audit implementations remain independent and still pass.
- Package documentation distinguishes the OT-II transport quotient from
  `topographo.ssd.fixed_head.readout_probability`.
- No density-matrix, POVM, collapse, interference, or full quantum-probability
  claim is made without new theory and evidence.

## Scientific fences

This issue may establish:

```text
the existing forced single-step identity has a reusable, robust implementation
the implementation agrees with the committed OT event/operator family
the audit and package no longer duplicate the canonical arithmetic
```

It may not by itself establish:

```text
the overlap numerator samples or normalizes a complete outcome family
multi-step composition or path-sum interference
Born-rule measurement, projection, or collapse
POVM or density-matrix semantics
physical probability, energy, or temperature interpretations
equivalence with the configured Theory-27 Fixed head
universality beyond the committed OT/Kraus-84 construction
```

The target is deliberately narrow: first turn the exact theorem already present
in executable audits into reliable library code; treat the generalized
measurement theory as a separate open research burden.

# Multi-token native composition (Issue 008)

Plan for Quilt `008.01`: does a learned policy over strict OT-admissible program
trees add held-out predictive value on short multi-token sequences **beyond**
deterministic availability and fixed-bracketing baselines?

Status: **plan + reporting contract only.** No learning code yet. The program
signature search is the prerequisite and lands first.

## Why the Outcome 017 benchmark cannot be reused

`017.24` section 3 settles it. On the pinned two-program family the deterministic
GroupedFirst rule scores test `A = 1.000`, `H = 1.000` on the Native arm, matching
the learned policy exactly (`ΔA = 0.000`). The native target generator prefers the
grouped endpoint whenever legal branches disagree, so an availability rule
coincides with the preferred tree by construction.

That is a negative result for *learned* composition on that benchmark, and it is
why `008.01` section 4 installs a firewall: the target relation must be specified
independently of the candidate policy and must not reduce to an availability rule.

## Files

| File | Role |
|---|---|
| `reporting.py` | interface layering + decoder-validity contract (`007.05`) |
| `native.py` | strict `Occ` / `Cyc` / `Sand` on the exact rational path |
| `programs.py` | planar term calculus and program-tree enumeration |
| `probe_program_space.py` | the `008.01` section 3 signature search |

## Prerequisite: find the signature

`008.01` section 3 asks for the smallest exposed-input signature, preferably
starting at `E^3 R`, satisfying all three of:

1. at least two strict legal program classes jointly defined on a useful common domain;
2. exact native endpoints that differ on a non-negligible set;
3. no single deterministic rule over local branch availability — GroupedFirst
   included — reproduces the target assignment by construction.

If `E^3 R` fails any of the three, move minimally upward and record why.

Strict constructors only, reusing what Outcome 017 already certified:

```text
Occ  : (E,R) ⇀ R
Cyc  : (E,E) ⇀ E
Sand : (E,R) ⇀ R
```

Ambient `Mul` is **not** promoted to OT occurrence. It appears only as the fenced
control in ladder arm C.

`native.py` supplies all three on one exact path, separating **ill-typed** (wrong
role in a slot, raises `TypeError`) from **undefined** (well-typed, but the exact
value annihilates or `Cyc` falls outside the 336 admissible pairs, returns
`None`). Undefinedness is reportable data; an ill-typed term is not a program.

### What the Sand census constrains

`Sand` only became scoreable on the exact path here, so its structure was worth a
full census over the 7056 ordered Event pairs before designing the program space:

| Quantity | Value |
|---|---|
| Ordered Event pairs | 7056 |
| Admissible Cyc edges | 336 |
| `Sand` defined | 6720 |
| `Sand` undefined | 336 |
| `Sand(e,r) = [r]` (degenerate) | 1008 |
| `Sand` non-trivial | 5712 |
| `Sand` projectively equal to `Occ` | **0** |

Two consequences for the signature search:

- **`Sand` is a genuine constructor.** It never coincides with `Occ` on
  `Event × Event`, and at the 017 retained context `r = e4` both are total on the
  84 Events and disagree on all 84. So adding `Sand` genuinely widens the program
  space rather than relabelling `Occ`.
- **`Sand` is dead on the Cyc domain.** The Theory-065 edge identity
  `(e*r)*e = 2||e||² r` is exactly what `is_edge` tests, so every one of the 336
  admissible ordered pairs forces `Sand(e,r) = [r]`. A `Sand` node placed on an
  admissible edge contributes no endpoint information, and the enumeration must
  not count those placements as legal alternatives.

Pinned in `topographo/tests/test_multitoken_native.py`.

## Interface layering (Quilt 007.05)

Klein's fair-grokking conjecture separates externally stipulated interface from
learned middle from native constitution, and argues that failing the native layer
does not falsify the ordinary learning claim. Issue 008 has the same three layers,
so every arm declares which it exercises:

| Key | Meaning | Claim it earns |
|---|---|---|
| `supplied_tree` | stipulated bracketing executed for the model | `H_weak` |
| `selected_tree` | the learned middle: which legal program to run | `H_rel` |
| `recovered_dens` | denotations the model itself recovered | `H_native` |

The three claims stay logically separate. Failure of `H_native` does not falsify
`H_weak` or `H_rel`; success on `H_weak` does not establish `H_native`.

`supplied_tree` and `selected_tree` are mutually exclusive within one arm. If the
bracketing is handed to the model it did not select it, and the two must be
reported as different arms rather than one blended number.

`reporting.py` enforces this. `ArmReport.failures()` rejects a claim whose layer
was never exercised, and `validate_suite` additionally requires the section 9
diagnostics.

## Decoder validity checklist

From `007.05` section 8, sharpened by `008.01` section 5.D. A decoder invalidates
the comparison if it:

- contains a lookup for held-out answers;
- computes the target relation itself;
- was fitted using held-out targets (only `train` or `none` is admissible);
- supplies free parameters that manufacture native result identity while the arm
  claims `H_native`.

Encoding and decoding must be fixed independently of the held-out answers.
Architecture-specific internal representations are allowed; semantic privilege is
not. `DecoderAudit` records each condition per decoder.

## Experimental ladder (008.01 section 5)

| Arm | Configuration | Layers |
|---|---|---|
| **A** | true / frozen denotations + learned tree policy — **primary first test** | `selected_tree` |
| **B** | true / frozen denotations + deterministic baselines | `supplied_tree` |
| **C** | ambient generalized-sedenion control, fenced non-OT-native | `supplied_tree` |
| **D** | train-only recovered denotations, **only** after A shows value | `selected_tree`, `recovered_dens` |

Joint denotation + policy training is not a starting point.

## Baselines (008.01 section 8)

`GroupedFirst`, `ForceSeq` / right-comb, a fixed supplied-bracketing executor, a
random legal-tree selector, and the ambient generalized-`Mul` control. A learned
policy counts as evidence only if it materially exceeds every relevant
deterministic baseline on held-out exact-native success.

`learned A <= GroupedFirst` is reported as a negative result for learned
composition even when absolute accuracy is high.

## Held-out structure (008.01 section 6)

Held-out **combinations and bracketings**, not repeated instances of one program.
Split construction is reported explicitly. Where possible include a compositional
generalization split whose component motifs appear in training but whose
combination does not.

Optional role-neutral arm, following `007.05` section 5: hold a bracketing motif
out in one position and test it in another. `experiments/tlm_modular/data.py`
already carries `alternate_role_test` and `relation_transfer_gap` for the modular
case; reuse that shape rather than inventing a split.

## Scoring (008.01 section 7)

Primary scoring is exact native result semantics: projective endpoint equality,
defined / undefined status, and Event-role or incidence predicates where
independently certified. Any differentiable surrogate used for training is
reported separately, and every result is rescored under the exact evaluator.

Note that `007.05` does **not** loosen this. Its objection is to demanding an
OT-native *external symbol vocabulary* as a prerequisite for a grokking claim.
Issue 008 has no external label space — the result is a native projective
endpoint — so exact native scoring here is not the asymmetry Klein identifies.

## Reused, certified upstream

| Component | Source |
|---|---|
| Exact Phi-correct evaluator, `exact_occ` / `exact_p_seq` / `exact_p_grp` | `experiments/tlm_fixed_head/learned_admissibility_01719_exact_eval.py` |
| Strict constructors on one exact path, `occ` / `cyc` / `sand` | `native.py` (this directory), cross-checked against the above |
| Cyc admissibility, forced third (84 / 56 / 336) | `topographo.ssd.fips_basic` |
| Exact rational Values, projective equivalence | `topographo.ssd.exact`, `topographo.ssd.projective` |
| GroupedFirst / ForceSeq reference numbers | `experiments/tlm_fixed_head/learned_admissibility_01723_final_audit.py` |
| Degree-matched non-isomorphic rewiring control | `topographo.ssd.structural_control` |
| Frozen Rec_ray denotations for arm D | `experiments/tlm_fixed_head/01717_artifacts/checkpoints/seed{0,1,2}_Rec_ray.pt` |

Outcome 017 typing results are not reopened.

## Stop rule (008.01 section 10)

One principled architecture pass plus one clearly motivated repair, then return
the result positive or negative. A negative result must state whether the limit is
task construction, strict program-space sparsity, policy learning, native endpoint
separability, or deterministic-baseline sufficiency.

## Out of scope

Language-model claims; physical Event supply; physical Test Realization; published
grokking SOTA; reopening Outcome 017 typing; promoting ambient binary trees to
OT-native; a free decoder used to obtain success.

## How to run

```bash
uv run --frozen pytest topographo/tests/test_multitoken_reporting.py \
  topographo/tests/test_multitoken_native.py -q
```

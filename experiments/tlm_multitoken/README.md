# Multi-token native composition (Issue 008)

Plan for Quilt `008.01`: does a learned policy over strict OT-admissible program
trees add held-out predictive value on short multi-token sequences **beyond**
deterministic availability and fixed-bracketing baselines?

Status: **signature search (`008.02`), Ladder A (`008.05`), Ladder D (`008.08`)
all run.** See
[`008.02-signature-search-result.md`](008.02-signature-search-result.md),
[`008.05-E3R-three-constructor-learning-result.md`](008.05-E3R-three-constructor-learning-result.md)
and
[`008.08-Kiro-recovered-denotation-composition-result.md`](008.08-Kiro-recovered-denotation-composition-result.md).

Ladder D is positive but its commissioned robustness measurement is **degenerate**:
endpoint-inverse voting recovers all 16 tokens exactly, so the recovered arm is
bit-identical to the true arm and the recovery penalty is `+0.0000` as an identity
rather than a measurement. An added probe locates the real boundary — the learned
advantage survives at 14/16 recovered tokens and is gone by 7/16.

Headline: the `017.24` obstruction is the **constructor set**, not the token count.
With `Occ` and `Cyc` only, 93 % of `E^2 R` inputs and 88 % of `E^3 R` inputs admit
exactly one legal program, so availability already fixes the endpoint and no target
construction can defeat an availability rule. Admitting `Sand` — the third
constructor `017.04` certified, previously float-only — lifts the joint domain to
98.6 % and opens 0.73 headroom at `E^2 R` already. Ladder A should run at `E^3 R`
with all three constructors.

## Why the Outcome 017 benchmark cannot be reused

`017.24` section 3 records the symptom. On the pinned two-program family the
deterministic GroupedFirst rule scores test `A = 1.000`, `H = 1.000` on the Native
arm, matching the learned policy exactly (`ΔA = 0.000`), because the native target
generator prefers the grouped endpoint whenever legal branches disagree.

`008.01` section 4 responds by installing a firewall: the target must be specified
independently of the candidate policy and must not reduce to an availability rule.

The signature search shows the firewall is **necessary but not sufficient**. Even
under a target built to defeat availability rules, the Occ/Cyc-only program space
leaves an availability ceiling of 0.973 at `E^2 R` and 0.936 at `E^3 R`, because
almost every input admits only one legal program. Full numbers in
[`008.02`](008.02-signature-search-result.md).

## Files

| File | Role | Torch |
|---|---|:--:|
| `reporting.py` | interface layering + decoder-validity contract (`007.05`) | no |
| `native.py` | strict `Occ` / `Cyc` / `Sand` on the exact rational path | no |
| `programs.py` | planar term calculus and program-tree enumeration | no |
| `probe_program_space.py` | the `008.01` section 3 signature search | no |
| `program_space_report.json` | exhaustive sweep artifact, all 84 Events | — |
| `008.02-signature-search-result.md` | the signature-search result and its handoff | — |
| `task.py` | the frozen `008.04` task: exact records, target, splits, viability gate, baselines | no |
| `ambient.py` | Ladder C ambient generalized-`Mul` control, fenced non-native | no |
| `policy.py` | the two learned program-selection heads, exact rescoring | **yes** |
| `run_learning_00804.py` | the `008.04` driver | **yes** |
| `00804_artifacts/` | report / tiny report / split metadata | — |
| `008.05-E3R-three-constructor-learning-result.md` | the Ladder A result | — |
| `recovery.py` | the `008.07` token layer + train-only endpoint-inverse denotation recovery | no |
| `run_ladder_d_00807.py` | the `008.07` Ladder D driver | **yes** |
| `probe_degraded_recovery.py` | recovery penalty vs recovery quality (added, not commissioned) | **yes** |
| `00807_artifacts/` | Ladder D report, recovery diagnostics, degraded-recovery probe | — |
| `008.08-Kiro-recovered-denotation-composition-result.md` | the Ladder D result | — |

Everything except `policy.py` and `run_learning_00804.py` is torch-free and
pinned by `topographo/tests/test_multitoken_*.py` in CI. The learned-policy
modules are a manual run, following the same split as Outcome 017.

## Prerequisite: find the signature

`008.01` section 3 asks for the smallest exposed-input signature, preferably
starting at `E^3 R`, satisfying all three of:

1. at least two strict legal program classes jointly defined on a useful common domain;
2. exact native endpoints that differ on a non-negligible set;
3. no single deterministic rule over local branch availability — GroupedFirst
   included — reproduces the target assignment by construction.

**Answered.** Formally the smallest qualifying signature is `E^2 R`, but only once
`Sand` is admitted; with `Occ` and `Cyc` alone nothing through `E^3 R` qualifies.
008 should still run at `E^3 R`, since `E^2 R` is the 017 setting and would not
answer the multi-token question, and `E^3 R` carries 20 programs, 0.856 headroom,
36 availability patterns, and eight Cyc-free bracketings to hold out.

Condition 3 is settled by a **ceiling**, not by beating one rule: any availability
rule is a function of the availability pattern alone, so the best achievable
accuracy over all such rules is computable and bounds GroupedFirst, ForceSeq, and
the random legal selector simultaneously.

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

Run at `E^3 R` with all three constructors, per `008.02`.

| Arm | Configuration | Layers | Status |
|---|---|---|---|
| **A** | true / frozen denotations + learned tree policy — **primary first test** | `selected_tree` | run (`008.05`) |
| **B** | true / frozen denotations + deterministic baselines | `supplied_tree` | run (`008.05`) |
| **C** | ambient generalized-sedenion control, fenced non-OT-native | `selected_tree` | run (`008.05`) |
| **D** | train-only recovered denotations, **only** after A shows value | `selected_tree`, `recovered_dens` | run (`008.08`) |

Ladder D needs one interface change, because `008.05` handed the policy the exact
Event rays and so had nothing to recover. `recovery.py` adds a frozen
sign-balanced 16-token vocabulary over the 84 certified Events; everything
`008.06` section 7 pins stays fixed. Recovery is the banked `017` `Rec_ray`
endpoint-inverse voting, **streamed** rather than materialized: at `E^2 R` the
inverse map has 14112 entries, at `E^3 R` with twenty programs it has 11854080,
which does not fit in memory as a dict of exact ray keys.

Joint denotation + policy training is not a starting point.

Arm C's layer declaration is `selected_tree`, not the `supplied_tree` this table
carried before the arm existed. The control as built lets a capacity-matched
policy *select* among ambient trees, which is the like-for-like comparison; it
claims no `H_` hypothesis, because an ambient result earns no OT claim.

## Baselines (008.01 section 8)

`task.score_baselines` carries eight, on both halves of every split:

| Baseline | Deployable | What it is |
|---|:--:|---|
| `AvailabilityOracleCeiling` | no | per-availability-pattern argmax fitted on the half being scored; upper bound on the whole availability family |
| `AvailabilityRuleTrainFitted` | yes | the same argmax fitted on train only — the strongest availability rule you could ship |
| `GroupedFirst` | yes | the `017.24` rule, generalized to twenty programs |
| `ForceSeq` | yes | the pure right comb |
| `RandomLegalTree` | yes | analytic expectation of a uniform draw from the legal set |
| `MajorityProgramTrainFitted` | yes | frequency baseline: first legal program in train-hit order |
| `SignPatternLookupTrainFitted` | yes | deterministic lookup on an allowed **input** feature |
| `SuppliedBracketingExecutor` | no | the `H_weak` reference: the target tree is handed over |

The bar a learned policy must clear is the best **deployable** one. The oracle is
fitted on the held-out labels and the supplied-bracketing executor is handed the
answer's tree, so neither is a rule a learner competes against; both are reported
as bounds.

`SignPatternLookupTrainFitted` is not in `008.01` section 8. It was added because
`008.04` section 5.B requires adding any deterministic rule exposed during
implementation that uses only allowed input or availability features — and this
one turns out to matter enormously: since the target is a function of the Event
sign pattern, a lookup over the eight patterns scores **1.000** on any split that
does not withhold whole patterns. That is what makes the held-out-motif split the
only one that can carry the claim.

`learned A <= GroupedFirst` is reported as a negative result for learned
composition even when absolute accuracy is high. So is a tie with any other
deterministic rule.

## Held-out structure (008.01 section 6)

Four splits, all built before training, all digest-pinned in
`00804_artifacts/split_metadata.json`:

| Split | Kind | Withholds |
|---|---|---|
| `motif` | compositional (**primary**) | sign patterns `011` and `100`, hence two whole target bracketings |
| `motif_parity` | compositional (diagnostic) | all four odd-parity patterns |
| `random` | i.i.d. (control) | Event combinations only |
| `unseen_events` | Event identity (control) | 21 of the 84 Events |

Two constraints on the primary split, both checked mechanically:

- every constituent `(retained depth, head constructor)` choice still occurs in
  training, so only the *combination* is new;
- neither held-out target is the right comb or a most-grouped term. Withholding
  the homogeneous patterns `000` / `111` would have violated this — the `000`
  target *is* ForceSeq's fixed answer, and it measured 0.50 on that held-out half
  before the split was corrected. `test_multitoken_task.py` pins against a
  regression.

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

## Model scope (008.04 section 7)

One principled architecture pass plus the one repair the stop rule allows.

| Head | Params | Output |
|---|---:|---|
| `FlatPolicy` | 2740 | `Linear(64,32) → ReLU → Linear(32,20)`, one free column per program |
| `StructuralPolicy` | 2773 | `Linear(64,32) → ReLU → Linear(32,21)` scored against a **fixed** 20×21 program-structure basis |

Input is 64 numbers: the exact rational coordinates of the three Event
denotations and the retained context, cast to float. Nothing else. The output is
masked to the legal set and the selected program is executed by the exact
evaluator, so there is no learned result decoder anywhere.

`StructuralPolicy` is the repair, and the motivation is a defect rather than a
tuning intuition: on a held-out-motif split two of the twenty program labels
never appear as a training target, so `FlatPolicy`'s output columns for them get
no signal that could ever make them win, and its held-out score is bounded near
zero by construction. Replacing the free per-program column with a fixed
structural basis means a program is scored only through parts it shares with
other programs. `task.program_structure_features` mentions no sign bit, no
availability, no endpoint and no target, and is checked injective over the
twenty programs.

Training surrogate: cross-entropy over the legal set with the positive set
defined by **exact endpoint equality** rather than tree identity, full batch,
Adam at `lr = 0.05`, 800 steps, seeds `(0, 1, 2)`. Every reported number is
rescored under the exact evaluator afterwards.

## How to run

CI-safe pins (seconds, no torch):

```bash
uv run --frozen pytest topographo/tests/test_multitoken_reporting.py \
  topographo/tests/test_multitoken_native.py \
  topographo/tests/test_multitoken_programs.py \
  topographo/tests/test_multitoken_program_space.py \
  topographo/tests/test_multitoken_task.py -q
```

Exhaustive signature sweep (about 25 CPU-minutes, manual, not CI):

```bash
uv run --frozen python experiments/tlm_multitoken/probe_program_space.py \
  --signatures 2 3 --out experiments/tlm_multitoken/program_space_report.json
```

The `008.04` Ladder A run (about 5 minutes, needs torch, manual, not CI):

```bash
PYTHONPATH=. python experiments/tlm_multitoken/run_learning_00804.py \
  --pool-size 40000 --steps 800 --seeds 0 1 2 \
  --out experiments/tlm_multitoken/00804_artifacts
```

The driver refuses to train if the `008.03` section 2.3 viability gate does not
conform to the `008.02` pins; it writes `viability_gate_failure.json` and exits 2.

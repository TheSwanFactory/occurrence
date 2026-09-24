# Issue 017.24 matched TDM training-economy experiment

The first turn in Issue 017 that trains anything. 017.23 earned a two-parameter
Fano successor mathematically; this measures whether compiling the input
structure *before* Training removes labeled-data burden that Training would
otherwise carry.

Everything is exact or closed form and uses the standard library only, with one
exception: NumPy appears solely in condition `U` and in the parity certificates.
No torch, no network, no optimizer anywhere in `T`, `O`, `S`, or `G`.

## Result

**Strong positive on labeled-data burden — and the mechanism is structural
compilation, not OT.**

- The compiled two-parameter TDM reaches mean excess held-out log loss ≤ 0.02 at
  **64 labeled examples** in both regimes and at all three ground truths. No
  preregistered information-parity uncompiled learner reaches that threshold
  anywhere on the 4–1024 grid, and neither does the saturated 49-parameter
  ablation. Label ratio ≥ **32×** under the preregistered factor-two criterion.
- Condition `O`, given the derived class identifier and invoking neither OT nor
  the compiler, and condition `G`, a generic union-find closure over the supplied
  automorphism group, both produce **bit-identical** predictions to `T` in all
  **21,600** replicate comparisons. The gain is therefore structural
  compilation, not the Born rule.
- Regime B is qualitative, not quantitative. `T` transfers to unseen pair
  identities with an exact floor of **0**; `S` is pinned at exactly **0.130812**
  with zero variance across 400 replicates at every sample size; `U` transfers
  *worse* than `S` and degrades with more data, because it extrapolates
  confidently where `S` abstains at ½.
- The exact binomial-lattice reference predicted the whole `T` curve in advance:
  `E[excess NLL] = d/(2n) = 1/n`. All **54** cells agree within 4σ, worst 3.47σ.

## Controls

| Control | Construction | Outcome |
|---|---|---|
| Wrong compiler | two cells, same 21/28 cardinalities, 7 least members of each class exchanged | violates 3,332 automorphism image pairs; plateaus at 0.1077 against an exact floor of 0.106784 and never improves |
| Scrambled structure | 20 deterministic relabelings preserving 7 points, 7 lines, 21 marked pairs, 49 admitted pairs | groups of order 1, 2, 4; class counts 35, 36, 42, 49; **0 of 20** reproduce the 21/28 quotient; no two-class result forced |
| Relation erased | both sorts, 49 admitted pairs, 21 rows kept as opaque payload, relational meaning removed | compiler **and** relational encoder both fail closed; no fallback recovers the quotient from label positions |

The wrong-compiler control is the load-bearing one: it refutes "two parameters
beat forty-nine". Two parameters do not. The *right* two parameters do.

## Fairness

The credibility of the result rests on `U` being neither starved nor spoon-fed,
so both are certified constructively rather than asserted.

- **Not starved.** Seven explicit hidden units `relu(rowₖ(point) +
  onehotₖ(line) − 1)` sum to the marked-pair indicator, so the compiled family
  lies exactly inside `U`'s hypothesis class. A second explicit setting isolates
  a single admitted pair, so `U` is not restricted to class-invariant functions
  and the two-class structure is nowhere a parameter tying rule.
- **Not spoon-fed.** Every one of `U`'s 28 feature columns is a function of the
  query point alone or of the query line alone, so the column span is exactly
  the additive functions `f(point) + g(line)`. The indicator is not additive:
  on `{P0,P1} × {L0,L1}` its corners are `1,1,1,0`, alternating sum `−1`.
  Least-squares residual `0.5714`.
- `U_quadratic` is the labeled exception — its degree-two span **does** contain
  the class structure — and is included on purpose so the headline comparison
  runs against the strongest information-parity baseline short of handing the
  quotient over. It is what `U_best` selects in 17 of 18 primary cells.

### Baseline amendment, disclosed

A pilot showed the originally preregistered `U` was worse than a one-parameter
constant predictor (floor 0.1283) — an optimization failure, not an information
limit. Because §14 of the task makes baseline competence decisive, the family was
strengthened before the final replicate bank: 2 → 5 variants, a nested
complexity ladder (constant 1, linear 29, quadratic 407, mlp 481), a ridge grid
`{1e-3 … 10}` selected on training labels only, intercepts brought inside the
penalty so high ridge shrinks to `p = ½` matching `T`'s beta(1,1) prior mean,
step budget 600 → 1200, replicates 30 → 100. Every change works against the
thesis; none was chosen by comparing evaluation outputs.

The **fairness frontier** in `training_economy_results.json` makes that auditable:
`U`'s whole regularization path with the ridge chosen by *hindsight* against the
evaluation metric. It is explicitly not a legitimate baseline and does not decide
the verdict. It shows the top of the grid converges to the constant-model floor —
so the grid reached the achievable frontier rather than being truncated — and
that even the hindsight oracle never reaches 0.02.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `successor_schema.json` | required artifact 1: frozen `Σ′`, no rule, no answer, no λ | `149778bf5b8cca166ab77d3f33e9368191a58b33af9d1c15075f9ff41312a81d` |
| `preregistration.json` | required artifact 6: models, hyperparameters, sizes, seeds, metrics, thresholds, controls, verdict rule, amendment disclosure | `32fa75c0856cf13b3bbecf1eaf0619d851f8db69edc199015e94750176a575f9` |
| `training_economy.py` | required artifacts 3, 4, 5: splits, hidden generator, and `T`/`O`/`U`/`S`/`G` | `821d263a6401f8bc3b929e0e855fe6533dbdd573ecc7f07bab669495a9167e79` |
| `test_training_economy.py` | required artifact 10: 53 adversarial tests, leakage and parity included | `0c1ce0545715b0c6adcbc54e4f9f779545ab6dd5c53bedca7ff26cd88b12b1db` |
| `orbit_certificate.json` | required artifact 2: compiler output, orbit certificate, leakage, parity, controls, `decision-model` bridge | `416e14ed444867fd7eacb8255c487635e68e0ac3057a5b819f8edb6306980eaf` |
| `split_certificate.json` | required artifact 3: deterministic split generator with witnesses | `95c1b4b2798e740c283a556fe62c3141f98639769a44fd4af28f1fea97d317ac` |
| `analytic_reference.json` | required artifact 9: exact lattice reference and asymptotic floors | `1bd262ab3bc51f5c57a9fd3cf9da6662670b8bc69eb106a9622bc915e93e8645` |
| `training_economy_results.json` | required artifact 8: aggregate cells, thresholds, frontier, verdict, 41 checks | `0030113d03a8c978651c75ee4c9e4c211f89575f31f5b29dc4ccc2cab3a7415d` |
| `cost_report.json` | §8 cost accounting | `e0a6644b3d454692e74f2a4f71136538af7603d757bf445b30ca27859fdb7de9` |
| `per_replicate.csv.gz` | required artifact 7: 127,800 raw per-replicate rows | `a745332cedbf2f3380f16594694394ab07b2c3b172fea00f59f9e14c7774621e` |

Required artifact 11 is
`../017.25-Kiro-first-matched-TDM-training-economy-experiment-result.md`.

Nothing is duplicated. The 017.20 source fixture is read byte-pinned from
`../017.20-Code-attachments/source_fixture.json`, and `Σ′` and the
preregistration are byte-pinned too. Any change to those three files makes the
loader refuse to run.

## What the experiment actually computes

- automorphism enumeration over all 5,040 point relabelings with **full
  injective matching** of line images, so a scrambled relation whose line
  point-sets repeat is handled correctly rather than flattered by a matcher that
  assumes a projective plane;
- the canonical class certificate, plus an independent generic union-find closure
  of the 49 inputs under the 168 supplied permutations;
- the compiled TDM resolved through the shipped `decision-model` contracts
  `Test`, `resolve_test`, `Distribution` at **zero tolerance** on a 25-point λ
  grid and on all 49 admitted inputs;
- exact expected held-out NLL, Brier, and 10-bin calibration error — the
  large-fresh-sample limit in closed form, so evaluation contributes no sampling
  noise and every condition is scored on identical ground;
- the exact binomial double sum for the two pooled Bernoulli estimators, over
  the class-count lattice and the success lattice, with no simulation;
- exact asymptotic floors for the saturated, wrong, scrambled, and typed
  partitions;
- 3 ground truths × 2 regimes × 9 sample sizes × 400 closed-form replicates, and
  100 replicates for every `U` variant on byte-identical observations;
- a second estimator convention, beta(½,½), which moves no threshold.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.24-Code-attachments/training_economy.py --check
uv run --frozen pytest issues/017-generalized-born-rule/017.24-Code-attachments/test_training_economy.py
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.24-Code-attachments
```

`--write` regenerates every artifact. `--quick` runs a reduced grid for
development and refuses to write. A full run is about 6 minutes; `--check`
recomputes everything and compares, byte-exact for the structural and analytic
artifacts and within a declared `1e-6` for the NumPy condition.

## Fences

No language-scale claim. No LLM or SLM comparison. No physical Test Realization
or Outcome constitution. No autonomous dynamics and no Interact. It is **not**
claimed that every useful Type Schema admits a compact compiler, and it is
**not** claimed that OT is responsible for the measured gain — conditions `O` and
`G` exist precisely to deny that inference, and they do. The fixture is a finite
controlled problem and establishes mechanism, not production economics. Issue 018
is not resolved here. Nothing was added to the stable `decision-model` public
API, which remains 12 symbols.

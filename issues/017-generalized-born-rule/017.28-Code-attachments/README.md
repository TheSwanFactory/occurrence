# Issue 017.28 — which structural compiler do we actually need?

Executed under `../017.28-Task-compiler-scope-refinement-versus-exact-automorphism.md`.
The freeze is [`../017.28a-Kiro-phase-S-compiler-scope-freeze.md`](../017.28a-Kiro-phase-S-compiler-scope-freeze.md);
the result is [`../017.29-Kiro-compiler-scope-refinement-versus-exact-automorphism-result.md`](../017.29-Kiro-compiler-scope-refinement-versus-exact-automorphism-result.md).

## The question

017.27 earned a narrow result — ordinary typed source structure was sufficient for
a generic exact compiler to derive a useful input quotient before labels — and then
disclosed the next problem in the same breath. Full automorphism enumeration was
not necessary on either fixture; an iterated pair refinement recovered the same
quotient in two rounds of polynomial work. So the question is no longer whether
compilation helps. It is which compiler is sufficient, canonical, and tractable.

Two compilers on the same footing:

```text
Q_ref(Σ)   a frozen fixed-point refinement over the admitted inputs
Q_aut(Σ)   X_Σ / Aut(A_Σ), the byte-pinned 017.26 exact compiler
```

## Result

**Separation and value.** They agree on **82 of 87** preregistered worlds,
including all 45 relation-only worlds and both 017.27 fixtures. They come apart on
**5**, every one declaring a total binary operation. On the first separation under
the frozen order the extra distinction is worth a lot — and one extra declared
element would have made it unnecessary.

| | value |
|---|---|
| worlds proposed / compiled / skipped | 87 / 87 / 0 |
| worlds where `Q_ref = Q_aut` | **82** |
| separations | **5** |
| primary separation fixture | `quasigroup:n=6:square=1` |
| `\|X_Σ\|` · `\|Aut\|` | 36 · 24 |
| `Q_ref` cells · `Q_aut` cells | **1** · **2** (12 / 24) |
| `θ*` by the inherited 017.24a rule | (1/4, 3/4) |
| `R` irreducible floor | **0.116858121372717375879907** |
| `A` irreducible floor | **0** exactly |
| `A` reaches 0.02 excess log loss | **n = 64** |
| `R` reaches 0.02 | **never** |
| estimation crossover | **none** — `A` wins from n = 1 |
| `A ≡ H` | **44,000 / 44,000** bit-identical |
| frozen predictions reproduced | **55 / 55** within 4σ, worst 2.14σ |

## The two-phase split, and why it is structural

017.28 §15 requires the structural phase banked before Phase T. Here that is
enforced by the import graph rather than by promise. `compiler_scope.py` imports
the standard library and the byte-pinned 017.26 module; the 017.26 module does not
import the 017.24 implementation, because that module contains
`generate_observations`. `compiler_value.py` is a separate file that pins
`compiler_scope.py` and `phase_s_certificate.json` by SHA-256, so the structure the
statistics are measured against is provably the structure banked before `θ*` was
evaluated.

| | Phase S | Phase T |
|---|---|---|
| module | `compiler_scope.py` | `compiler_value.py` |
| dependencies | standard library, plus the byte-pinned 017.26 compiler | pins Phase S, 017.24a, 017.24 |
| commit | `occurrence@b973940` | preregistration `occurrence@85915f2` |
| banked | `occurrence/gpt@3f0fcaf` | preregistration `occurrence/gpt@da3bdce` |

## The frozen generator family

Six families, enumerated in an order **computed** by a rule rather than written
down: families sorted by declared primitive count, then declared sort count, then
name; parameters sorted by total declared carrier elements, then by the parameter
tuple flattened to integers. No randomness anywhere — the quasigroup family
enumerates reduced Latin squares in lexicographic order.

| family | sorts | primitives | worlds | separations |
|---|---|---|---|---|
| `circulant_graph` | 1 | 1 relation | 22 | 0 |
| `bipartite_incidence` | 2 | 1 relation | 13 | 0 |
| `quasigroup` | 3 | 1 operation | 28 | **4** |
| `two_relation_graph` | 1 | 2 relations | 10 | 0 |
| `corridor` | 2 | relation + operation | 9 | 0 |
| `composite_product` | 4 | 3 primitives | 5 | **1** |

## Conditions

`A` exact automorphism compiler, 2 parameters · `H` handed exact oracle, reading
`Q_aut` membership out of the banked certificate by a path that never names a
compiler entry point · `R` refinement compiler, 1 parameter · `S` saturated, 36 ·
`W` cardinality-matched control respecting no derived map. The uncompiled generic
learners of 017.25 and 017.27 are deliberately absent: §9 makes them secondary and
the main comparison compiler-versus-compiler.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `compiler_scope.py` | Phase S: frozen `Q_ref`, pinned `Q_aut`, family `G`, search, witnesses, audits | `d8a57e810676b74fbd07395542afda8682c417082a371f2a138f02d3b13fd525` |
| `phase_s_certificate.json` | the structure freeze, 32 checks, no wall clock by design | `5dff9481109cb70b84b0669c37e4a4fc9a235201c6a4e926a6ec9cfcdfb56275` |
| `phase_s_cost_report.json` | compilation cost and per-family scaling | `57eb56c5504bca9e7a7d119e9455f76dd68b9b39812d12201dc81e8caa688cba` |
| `test_compiler_scope.py` | 43 Phase S tests, including an independent autotopism enumeration | `e21437fa037ace354cdf2d22555d4b3d7a7d6dfcfdb4376de6af062689af7826` |
| `compiler_value.py` | Phase T: conditions, exact floors, sweep, verdict | `a50a4eb43c6ea1258dcf5ce06894b5165e32a9913d2e88089da135e0351699f8` |
| `preregistration.json` | `θ*`, seeds, thresholds, every frozen prediction, the verdict rule | `cb3915f8daa687c1fe97f9d248f1402bf68de9790b269f3e0474b95b0a09b1cc` |
| `phase_t_reference.json` | exact predictions and approximation floors, no label in scope | `6bfaa63461dca7bc71101c20c5f943a32d03d5f3479c6e90209e15e8377907cf` |
| `phase_t_results.json` | measured cells, verdict, claim ledger, 19 checks | `ad855c867f1cc93566a4dd815e39285af0e9bba202194aa9a3cff72f18ff19e3` |
| `phase_t_cost_report.json` | cost accounting | `c2e69af716c39550d86b302b60f85bfb9649c772a1c0c769f9ec96a9dfba94c1` |
| `phase_t_per_replicate.csv.gz` | 220,000 raw per-replicate rows | `1e08ff1d5b07e697290a2f4be86b37586dfc58003acfcb89ce6cf6e155f5e0ef` |
| `test_compiler_value.py` | 28 Phase T tests, including a hand recomputation of the floor | `6b83761f17ac870486259441be7ef8c9ed8fc125c4ea3efa6ad0878334f2fed7` |

## What is weaker than the headline

**The separation is rare and the family was built to find it.** 82 of 87 agreed,
and 24 of 28 quasigroup worlds agreed too, so the a-priori hypothesis that
motivated the family was only partly right.

**One declared element closes the gap.** Adding a reference row to the separation
fixture takes both compilers to the same 10-cell partition. The cheapest route to
the exact quotient on that world is a more explicit schema, not a stronger
compiler.

**The exact compiler was usually faster.** Median time ratio 0.82 across the
family; the 627× worst case lives entirely in the typed-operation worlds.

**No estimation trade-off exists here.** The grid was extended to n = 1 to look for
a regime where the one-parameter refinement wins. There is none, because the merged
cells differ by 0.5 in probability.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.28-Code-attachments/compiler_scope.py --check
uv run --frozen python issues/017-generalized-born-rule/017.28-Code-attachments/compiler_scope.py --summary
uv run --frozen python issues/017-generalized-born-rule/017.28-Code-attachments/compiler_value.py --check
uv run --frozen python issues/017-generalized-born-rule/017.28-Code-attachments/compiler_value.py --summary
uv run --frozen pytest issues/017-generalized-born-rule/017.28-Code-attachments
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.28-Code-attachments
```

`--summary` prints the whole 87-world ledger for Phase S and the crossover table
for Phase T. `--preregister` regenerates the preregistration without generating a
single observation. `--write` regenerates the artifacts; `--quick` runs a reduced
grid and refuses to write or verify. Phase S is about 63 s; Phase T is about 14 s
for 220,000 fits.

## Fences

`Q_aut` is one preregistered exact structural compiler, not the universal TDM
quotient. `Q_ref` is canonical only relative to its frozen construction; agreement
with `Q_aut` on 82 of 87 worlds does not transfer `Q_aut`'s mathematical canonicity
to it. Exact automorphism computation is not shown to be generally necessary — on
82 of 87 worlds it was not. No claim is made that a refinement-based learner could
not recover the distinction from labels, and no learner-independent
sample-complexity bound is proved. Definability, congruence, bisimulation-like,
context, and task-relative quotients are untouched. No language-scale claim, no
comparison to any language model, no physical Test Realization, no Interact. OT is
not claimed to be responsible for any measured gain. Issue 018 is not resolved.
Nothing was added to the stable `decision-model` public API.

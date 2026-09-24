# Issue 017.24a / 017.25a — separating the compiled quotient from the declared relation

The follow-up that attacks the open item 017.25 left behind. Its preregistration
was committed and banked **before** the run, and the result is reported
separately in `../017.25a-Kiro-quotient-refinement-result.md`.

Everything is exact or closed form and uses the standard library only, with one
exception: NumPy appears solely in the uncompiled learners. The shared 017.24
implementation is imported **byte-pinned** at SHA-256 `821d263a…`, so the
estimator, hidden generator, split builder, exact evaluation, uncompiled
learner, and inherited controls are provably the same code that produced 017.25.

## The question

017.25 §B confessed that on the Fano successor `Aut(Σ′)` is transitive on marked
and on unmarked pairs, so the derived class partition **is** the declared
relation plus its complement. That leaves its mechanism claim underdetermined: a
learner indexed by the relation alone might account for the whole effect.

## Result

**Separation established.** All seven preregistered criteria hold and all 54
frozen predictions are reproduced within 4σ, worst 3.44σ, zero failures.

- On Fano, the relation-indexed learner `R` is **bit-identical** to the compiled
  `T` in **7,200 of 7,200** replicate comparisons, with `R − T = 0.0000` at every
  sample size. On the fixture 017.25 actually used, the cheaper account is
  exactly correct.
- On the circulant, the two coincide in only **20 of 7,200**. `R` plateaus at
  0.0415 against its exactly predicted floor of **0.040527** and never reaches
  0.02; `T` reaches it at **128** labels.
- The frozen prediction that cut *against* the thesis also held: `R`, with two
  parameters against `T`'s four, beats `T` at **n = 4 and only n = 4**.
- Declared-symmetry dose–response is monotone at every n from 16 upward on both
  fixtures, tracking `dof/(2n)`.

## The two fixtures, matched on every cardinality

| | `Σ′` Fano {0,1,3} | `Σ″` circulant {0,1,2} |
|---|---|---|
| points, lines, marked pairs, inputs | 7, 7, 21, 49 | 7, 7, 21, 49 |
| `\|Aut\|` | 168 | 14 |
| derived classes | 2 — 21, 28 | 4 — 7, 14, 14, 14 |
| relation splits into | 21 \| 28 | **7 + 14** \| **14 + 14** |
| quotient strictly refines the relation | no | **yes** |
| relation-indexed floor | **0** | **0.040527** |

## Conditions

`T` compiled quotient · `O` oracle class · **`R` relation-indexed, two
parameters** · `S` saturated 49 · `G` generic group closure · declared-symmetry
ladder rungs (49/7/3/2 on Fano, 49/7/4 on the circulant) · six uncompiled
variants including **`U_relation`**, handed the relation bit outright · plus the
wrong-compiler, coarse-compiler, scrambled, and relation-erased controls.

`O` and `G` remain bit-identical to `T` across all 14,400 comparisons, so nothing
here revives OT as the source of the advantage.

## Files

| File | Purpose | SHA-256 |
|---|---|---|
| `refinement_schema.json` | frozen `Σ″`, with its ladder generators declared extensionally | `31a6d0000582c6d40522111dfc0e1fd87244e4afc6e428bc845fefe25ed755bb` |
| `preregistration.json` | the freeze: every predicted curve, floor, threshold, the crossing, the verdict rule, the falsification conditions | `4a49500eed41dcd831f96fa5b6ef33467e179c2b25cde59f0ab7fb544f0c7aa4` |
| `quotient_refinement.py` | fixtures, condition `R`, the ladder, `U_relation`, the generalized exact reference | `9fcf033500b33cc9665a58ab04a24e6d7921417a5789cc7039550823eda7085f` |
| `test_quotient_refinement.py` | 39 tests, structural and adversarial | `e5c5593f5c7c62864da772fdf7a917de6a1c881ce2215ef532070f5fccc5e332` |
| `refinement_certificate.json` | compiler output for both fixtures, ladders, subgroup-choice independence | `fbdcada21832c1bbaa72b1d7ab51578f3408d6c33a5d5a12bde14cc121568bb5` |
| `generalized_reference.json` | the frozen exact predictions and the coarse-compiler control | `34d15924b60b209b16a26f381a850abe7ea1547ea5031ff4c03c7dbac3da2e58` |
| `refinement_results.json` | measured cells, verdict, prediction validation, 27 checks | `86c6da792e1ee1fbc0decfaa351327730603d12a6a83954ca83c2ed53191c269` |
| `refinement_cost_report.json` | cost accounting | `f5a010c7a06f5dcc472b7383754a6787a6d4e76ea65debcf544a6d712bcf7d61` |
| `refinement_per_replicate.csv.gz` | 172,800 raw per-replicate rows | `ee28d2fd2cf665de56be7590b03f0a3157cb08f9a7611cd379e3116b7b286d95` |

## The generalized exact reference

017.24's reference was specific to two classes with matched sampling weights.
This one is exact for any number of classes **and for partitions too coarse for
the truth**, which is what condition `R` and the coarse-compiler control need. It
is also simpler: excess log loss is additive across cells and a cell's estimator
depends only on its own observation count, which is marginally binomial, so the
expectation is a finite sum per cell with no joint lattice. It reproduces the
017.24 reference on the matched two-class case to better than 1e-9, and that
agreement is a test.

## Post-freeze corrections

Two, both disclosed in full in §K of the result and both auditable against the
freeze banked at `occurrence/gpt@adbc039c`.

1. The inherited wrong-compiler control assumed two classes and raised on a
   four-class fixture. Generalized here rather than in the byte-pinned shared
   module, so 017.25's artifacts remain valid. On a two-class fixture it
   reproduces the 017.24 construction member for member, asserted mechanically.
2. The dose–response monotonicity check compared adjacent rungs in the wrong
   direction relative to its own sort order, contradicting the banked
   preregistration's own text, and reported a perfectly monotone result as a
   violation. Corrected, with a regression test pinning the direction against
   that text. The per-replicate table is byte-identical across this correction,
   so **no measurement changed**.

## Reproduce

```bash
uv run --frozen python issues/017-generalized-born-rule/017.24a-Code-attachments/quotient_refinement.py --predict
uv run --frozen python issues/017-generalized-born-rule/017.24a-Code-attachments/quotient_refinement.py --check
uv run --frozen pytest issues/017-generalized-born-rule/017.24a-Code-attachments/test_quotient_refinement.py
uv run --frozen ruff check --select E4,E7,E9,F,I issues/017-generalized-born-rule/017.24a-Code-attachments
```

`--predict` emits the compiler certificate and every frozen prediction without
generating a single label. `--write` regenerates the artifacts; `--quick` runs a
reduced grid and refuses to write. A full run is about five and a half minutes.

## Fences

No language-scale claim. No LLM or SLM comparison. No physical Test Realization
and no autonomous dynamics. Neither fixture is representative of anything beyond
itself. OT is not claimed to be responsible for any measured gain. Issue 018 is
not resolved. Nothing was added to the stable `decision-model` public API.

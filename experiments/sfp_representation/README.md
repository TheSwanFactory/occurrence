# 009 — Native FIPS vs SFP consequence representation

Does the compact SFP code `S | FFF | PP | pp` carry the certified admit-and-force
relation better than the native FIPS realization, and is any advantage a property
of the *relational alignment* rather than of code capacity or coordinate choice?

Five arms over one shared learner, on 84 certified basic Events and all 7056
ordered pairs:

| Arm | Representation | Role |
|---|---|---|
| **A_native** | native FIPS realization | is native realization sufficient? |
| **B_sfp** | the exact certified SFP code | the primary arm |
| **C_scrambled** | marginal-matched, structure-destroying code permutation | is B's gain relational? |
| **D_relabeled** | B under `GL(3,2)` + affine `AGL(2,2)` + global `S` flip | is B a basis artifact? |
| **E_opaque** | one learnable column per Event | memorization ceiling, **not** a representation arm |

Result: **`009.02`, disposition E (narrowed)**. See
[`009.02-Kiro-native-FIPS-vs-SFP-representation-ablation-result.md`](009.02-Kiro-native-FIPS-vs-SFP-representation-ablation-result.md).

> **Superseded in part by `009.06`.** The `009.02` forcing negative turned out to
> be substantially an artifact of the 84-way output interface. Changing only the
> output object moved held-out forced-third identity from `0.0335` to `0.7016`.
> See [the consequence-address ladder](#the-009-06-consequence-address-ladder)
> below; the `009.02` *admission* findings and all its structural machinery stand
> unchanged and are reused byte-identically.

Forcing is unsolved by every representation — the best forced-third accuracy over
all arms is 0.0335 while both nonlearned oracles score 1.0 — so no arm
demonstrates held-out generalization of the certified relation as a whole. The
**admission half is cleanly resolved**: B beats the marginal-matched scramble C by
`+0.637` on same-habitat-disjoint accuracy, sign-consistent across all 14 primary
folds, and beats the native realization by `+0.377` on admission balanced
accuracy, which sits at chance (0.5048) for A.

## Read the metric caution first

A three-feature deterministic lookup (`same_fano_point + repeated + sign_pair`)
reaches **0.921** admission balanced accuracy on LOHO while scoring **0.0** on
forced third and **0.0** on same-habitat-disjoint accuracy. Admission balanced
accuracy is a weak discriminator; the science rests on forced-third and
same-habitat-disjoint accuracy.

And `A_native`'s admission sensitivity is 0.1332 while `E_opaque`'s is 0.0662 —
they answer BOTTOM to ~87% and ~93% of admitted pairs. Their high non-admission
accuracy is that artifact. Never read a non-admission number without its
admission sensitivity.

## Files

| File | Role | Torch |
|---|---|:--:|
| `conformance.py` | Gate 0: 103 Outcome-`021.05` pins re-derived, 14 named laws, `TABLE_SHA256` | no |
| `sfp.py` | habitat enumeration, the `S \| FFF \| PP \| pp` codec, exactness proof over the finite domain | no |
| `groups.py` | abstract `GL(3,2)` / `AGL(2,2)` / `GL(2,2)` / K4 machinery and `is_structure_preserving` | no |
| `task.py` | the frozen 7056-pair admit-and-force dataset, labels from the certified relation | no |
| `folds.py` | 22 frozen fold manifests (14 LOHO + 7 LOFPO + 1 random control) and the leakage check | no |
| `arms.py` | the five representation arms and the Arm-C rejection proof | no |
| `baselines.py` | deterministic selectors, exact metric definitions, the compact-feature leakage search | no |
| `scorer.py` | the one shared candidate scorer and its structural proofs (swap symmetry, no per-Event output column) | **yes** |
| `harness.py` | one training protocol and the whole metric suite | **yes** |
| `run_sweep.py` | the two declared configurations: `primary` and the one permitted `repair` | **yes** |
| `analysis.py` | paired A/B/C/D effects, bootstrap intervals, the disposition | **yes** |
| `009_artifacts/conformance_audit.json` | Gate 0 audit | — |
| `009_artifacts/sfp_codec.json` | codec and exactness record | — |
| `009_artifacts/groups.json` | group census and induced/non-induced classification | — |
| `009_artifacts/task_dataset.json` | the frozen pool, `dataset_sha256 7872755f…` | — |
| `009_artifacts/folds.json` | fold manifests, `fold_manifest_sha256 27b01b5e…` | — |
| `009_artifacts/arms.json` | per-arm token tensors and digests | — |
| `009_artifacts/scorer.json` | architecture record and shape proofs | — |
| `009_artifacts/baselines.json` | baseline scores and the 2838-fit leakage search | — |
| `009_artifacts/harness_smoke.json` | declared smoke subset for the training protocol | — |
| `009_artifacts/sweep.json.pointer.json` | pointer to all 1400 runs, `8a717a9a…`, [hosted in the package](#bulk-run-data-lives-in-the-quilt-package) | — |
| `009_artifacts/analysis.json` | paired effects and disposition, `a772f30c…` | — |
| `009.02-…-ablation-result.md` | the result turn | — |

Everything except `scorer.py`, `harness.py`, `run_sweep.py` and `analysis.py` is
torch-free. Following the repo convention established by Outcome 017 and Issue
008, the torch modules are a **manual run and are not in CI**; only the torch-free
modules are linted and import-gated.

## Bulk run data lives in the Quilt package

`sweep.json` is 9.7 MiB of raw per-run rows. It is bulk run data, not provenance,
so it is hosted in the public `protology` Quilt package instead of in git. What
git carries is `009_artifacts/sweep.json.pointer.json`, which records the package
revision, the S3 URI, the byte count and the SHA-256 of the real file.

```text
package   occurrence/gpt   bucket protology
revision  4087609bc5f329a01647e50b44f795bcf2a2fd9c15687b32ff0a1563d50a77cf
key       issues/009-sfp-consequence-representation/009.02-Code-attachments/sweep.json
sha256    613adaddb8f3554bf3f7a7535a98b214c0fbd9663443e7c4e6b3326b99611b04
```

Only `analysis.py` consumes it. Fetch it to the path the pointer names before
running `analysis.py --check`:

```bash
aws s3 cp s3://protology/occurrence/gpt/issues/009-sfp-consequence-representation/009.02-Code-attachments/sweep.json \
  experiments/sfp_representation/009_artifacts/sweep.json
shasum -a 256 experiments/sfp_representation/009_artifacts/sweep.json
```

Every other module's `--check` is self-contained and needs nothing fetched. The
same package revision also carries copies of all eleven artifacts and the result
turn, so the run is reproducible from the package alone; the ten small artifacts
stay in git as well because they are what make the torch-free `--check` chain work
in a fresh clone.

## How to reproduce

Each module writes exactly one artifact and re-checks it byte for byte with
`--check`. Run them in dependency order; each imports the previous ones' frozen
output and re-derives nothing.

Torch-free, fast, CI-safe:

```bash
uv run --frozen python experiments/sfp_representation/conformance.py --check
uv run --frozen python experiments/sfp_representation/sfp.py --check
uv run --frozen python experiments/sfp_representation/groups.py --check
uv run --frozen python experiments/sfp_representation/task.py --check
uv run --frozen python experiments/sfp_representation/folds.py --check
uv run --frozen python experiments/sfp_representation/arms.py --check
uv run --frozen python experiments/sfp_representation/baselines.py --check
```

Needs torch, manual, **not** CI. `run_sweep.py` is the long one — 1400 runs,
about 1055 s for the primary block and 788 s for the repair block:

```bash
uv run --frozen python experiments/sfp_representation/scorer.py --check
uv run --frozen python experiments/sfp_representation/harness.py --check
uv run --frozen python experiments/sfp_representation/run_sweep.py --check
uv run --frozen python experiments/sfp_representation/analysis.py --check
```

`analysis.py` reads `009_artifacts/sweep.json`, which git does not carry — either
run `run_sweep.py` first or fetch it from the package, see
[Bulk run data lives in the Quilt package](#bulk-run-data-lives-in-the-quilt-package).

Omit `--check` to regenerate the artifact instead of comparing it.
`conformance.py` is a **hard gate**: nothing downstream is licensed if it fails.

Lint, matching the CI allowlist:

```bash
uv run --frozen ruff check --select E4,E7,E9,F,I experiments/sfp_representation/
```

## Physical fence

```text
SFP does not replace Occ / Cyc / Sand or native OT evaluation
FIPS geometry is not shown unnecessary in general
TLM did not discover SFP; SFP was supplied
XOR alone does not explain GPT 12
FFF=000 and pp=00 are formal completion values only; 0 != bottom
algebraic zero != NONADMISSION
the finite SFP code does not reconstruct the continuous G2 sweep
no language-model advantage is claimed
no generic computational superiority is claimed
no physical Event supply or Outcome semantics is claimed
Arm E is a diagnostic memorization ceiling, not a representation arm
the random pair holdout is a regression control and was found non-dispositive
```

Two of those need their mechanism stated rather than just their label.

**Arm E carries a per-Event input column.** Its `token_dim == 84 ==` the
catalogue size, so its input adapter holds one learnable column per Event. That is
deliberate for a memorization ceiling, but `scorer.py`'s shape check refuses
84-wide *output* tensors and cannot catch an 84-wide *input*. Arm E therefore
never substitutes for Arm C.

**The random pair holdout was mechanically confirmed non-dispositive.** The
leakage search found all 50 cheaper-than-the-relation shortcuts on that fold and
none on LOHO or LOFPO; the control is won outright by "same habitat and not
repeated ⇒ ADMIT".

---

## The `009.06` consequence-address ladder

`009.02` left a split verdict: the SFP code carries the *admission* half of the
certified relation and no representation solved *forcing* at all. `009.03` proposed
that the learner might be losing a real advantage at the output interface — 84
opaque public Event identities — rather than failing to hold the relation. `009.04`
proposed the right three-rung test but imposed an invalid `B > C` condition at the
local `pp` rung; `009.05` superseded it before execution, because
`GL(2,2) ≅ S3` makes every relabeling of that quotient a symmetry.

**`009.06` executes `009.05`.** See
[`009.06-Kiro-corrected-structured-consequence-address-ladder-result.md`](009.06-Kiro-corrected-structured-consequence-address-ladder-result.md).

The experiment changes **only the output object**. `ladder_heads.py` *imports*
`scorer.TokenEncoder` and `scorer.SymmetricPair` rather than reimplementing them,
and reuses the frozen splits, arms, seeds and training protocol, so a difference in
result is attributable to the interface.

```text
009.02   RelHead over 84 candidate Events + BottomHead        85 slots
Rung 1   one 3-way local port head                             3 slots
Rung 2   BOTTOM gate + typed S / FFF / PP / pp heads           17 slots
Rung 3   a fixed exact nonlearned resolver, no parameters
```

### Result: disposition C, partial

`positive_forced_third_exact_accuracy` — the same metric `009.02` reported and
failed — on the primary LOHO family, 14 folds × 8 seeds:

| Arm | `009.02` 84-way | `009.06` structured |
|---|---|---|
| **B_sfp** | 0.0335 | **0.7016** |
| D_relabeled | 0.0283 | 0.7522 |
| A_native | 0.0000 | 0.4412 |
| C_scrambled | 0.0052 | 0.1935 |
| exact oracles | 1.0000 | 1.0000 |

Both structural signatures `009.05` section 5.6 requires hold: `B - C = +0.5082`
(CI `[0.4583, 0.5595]`, sign-consistent across all 14 folds) and `D - B = +0.0506`
(CI `[-0.0052, 0.1049]`, includes zero).

### Read these four cautions first

**Rung 1 cannot discriminate.** Its label is a function of the unordered input port
pair, which under the exact code takes exactly three values, so a three-row lookup
table fitted on training habitats reaches 1.0 on held-out ones. The rung confirms
the local law is acquired — B and D hit 1.0 at a zero generalization gap — and
carries no structural weight in either direction. `B - C = +0.6682` is recorded and
excluded from the disposition by construction.

**The result required one declared repair, and both blocks are reported.**
Learning all four address fields fails at `0.0097`, and it fails on the fields the
relation *copies*, not the ones it computes: `pp` reaches `0.9948` while `FFF`
collapses to `0.0320`. That is closed-set habitat re-identification, not
consequence forcing. The single repair `009.05` section 9 permits copies `S` and
`FFF` — the construction section 4.2 already licenses at Rung 1 — and is
arm-neutral, since the certified relation copies both on all 336 admitted pairs
under B, C and D alike. **The catalogue interface was a cause, not the only one.**

**The obstruction moved rather than vanished.** Within the repair block `pp` is at
`0.9762` and `PP` at `0.7158`, and the address metric is essentially the `PP`
metric (`0.9762 × 0.7158 = 0.6987` against an observed `0.7016`). Locating the
shared block, not forcing, is now the binding constraint.

**A nonlearned tabulation beats the learner.** The code-space shortcut search —
new to this issue, since `009.02`'s atoms are all native — finds that under the
exact code the certified consequence is determined by a **two-atom** key with zero
ambiguity over all 7056 pairs, reaching 1.0 on held-out habitats. Under the
non-automorphic scramble **no subset up to size three determines it at all**. So
`H_struct` holds without any learner, and the learner at `0.7016` sits between the
best strictly-cheaper rule (`0.500`) and a ceiling the representation itself
attains (`1.000`).

### Ladder files

| File | Role | Torch |
|---|---|:--:|
| `ladder_task.py` | Rung-1 port and Rung-2 address targets, splits, leakage, the pre-registered habitat-locality mechanism | via `harness` |
| `ladder_heads.py` | typed field heads over the **imported** `009.02` encoder; the declared repair | **yes** |
| `ladder_resolver.py` | the fixed exact nonlearned resolver and its saturation/no-repair proofs | no tensors |
| `ladder_baselines.py` | deterministic baselines and the code-space shortcut search | no tensors |
| `ladder_sweep.py` | both declared blocks, 2688 runs, about 2820 s | **yes** |
| `ladder_analysis.py` | paired effects via the **imported** `009.02` bootstrap, two-ceiling reading, disposition | **yes** |

`ladder_task` reuses `harness.training_positions` / `evaluation_positions` so the
Rung-2 splits are provably the objects that produced the `009.02` numbers; a local
copy could drift. `harness` imports torch, so the whole ladder chain needs it —
`ladder_resolver` and `ladder_baselines` construct no tensor and hold no
parameters, but they inherit that import transitively, and their artifacts record
this rather than claiming otherwise.

### Reproducing the ladder

```bash
uv run --frozen python experiments/sfp_representation/ladder_task.py --check
uv run --frozen python experiments/sfp_representation/ladder_heads.py --check
uv run --frozen python experiments/sfp_representation/ladder_resolver.py --check
uv run --frozen python experiments/sfp_representation/ladder_baselines.py --check
uv run --frozen python experiments/sfp_representation/ladder_sweep.py --check   # ~2820 s
uv run --frozen python experiments/sfp_representation/ladder_analysis.py --check
```

`ladder_sweep.json` is 9.2 MiB of bulk run data and lives in the package, not git,
matching the convention `009.02` established for `sweep.json`. Only
`ladder_analysis.py` needs it:

```bash
aws s3 cp s3://protology/occurrence/gpt/issues/009-sfp-consequence-representation/009.06-Code-attachments/ladder_sweep.json \
  experiments/sfp_representation/009_ladder_artifacts/ladder_sweep.json
```

### Ladder fence

```text
Rung-1 B > C is NOT structural evidence; GL(2,2) = S3
the consequence relation is NOT fully constituted: 0.7016 against a 1.000 ceiling
the 84-way interface was A cause of the 009.02 negative, not the only one
BOTTOM is a typed gate, never pp=00, never FFF=000, never Event 0, never UNRESOLVED
UNRESOLVED != BOTTOM: failing to denote is not deciding not to admit
Arm E is not built; a token_dim of 84 is a per-Event input column
admission balanced accuracy still carries no claim, per 009.02
opaque-token discovery is now warranted, but should target PP and pp only
```

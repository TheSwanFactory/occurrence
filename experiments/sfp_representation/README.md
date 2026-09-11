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

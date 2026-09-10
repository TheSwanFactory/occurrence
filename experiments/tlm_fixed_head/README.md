# Fixed-head capacity / usefulness probe (Issue 006 → Quilt 006.14)

Configured computational OT-native **two-step Fixed head** on topographo **0.8.2**.
Implements the Owner-accepted existence census from Quilt **006.12 / 006.13** and
answers the capacity question experimentally for modular arithmetic.

## Physical fence

This is a **configured formal test / probability head**, **not** a physical Test
Realization or Outcome family (Theory 041/29, Outcome 20). Softmax baselines are
scaffolding, not an OT law. No SOTA grokking claim.

## Pinned scientific contract

```text
horizon n=2
ordered settlement word w=(b,a) among 84 basic Events → 7056 words
equal product weights
explicit Fixed twirl/projection E_Fix onto C=span{g0,g1,g3,g4} (Fixed 05b)
B_w = (1/7056) E_Fix(M_w),  M_w=(K_b K_a)^T (K_b K_a)
P(w|[x]) = <x, B_w x>/<x,x>
```

Accepted census (pinned in `topographo.ssd.fixed_head` + package tests):

| Quantity | Value |
|---|---|
| Ordered two-step words | 7056 |
| Distinct Fixed projection classes | 15 |
| Span dimension | 4 |
| Scalar class `I = g0+g3` multiplicity | 672 |
| Non-scalar word labels | 6384 |
| One-step firewall | `E_Fix(M_a)=I` for all 84 Events |
| Direct singleton in `C` | 0 |
| Normalization | `(1/7056) Σ_w E_Fix(M_w)=I` |
| Rank hist of `M_w` | `{8:504, 10:3024, 12:3528}` |

## Library adapter

`topographo.ssd.fixed_head` — Fixed 05b generators, Kraus lifts, `E_Fix`, class
table / census checksum, Theory-27 readout. No optimizer imports.

## What this experiment measures

1. **Alphabet capacity** — 7056 word labels → 15 effect classes; mutual information
   / collision structure vs modular symbols for `p=13` (CI) and `p=97` (science).
2. **Usefulness smoke** — linear readout on (A) residue one-hots, (B) Fixed
   15-class one-hots, (B+) concatenation. Reports wall-clock, accuracy, and whether
   the **15-class bottleneck is fatal** for `p=97`.
3. JSON artifacts: `capacity_report.json`, `smoke_results.json`.

Residue→Event embedding used for the probe is a **configured map** (index mod 84),
not a derived Event-space semantics.

## How to run

### Unit / census (no torch; CI)

```bash
uv run --frozen pytest topographo/tests/test_fixed_head.py -q
uv run --frozen pytest topographo/tests -q
uv run --frozen python experiments/tlm_fixed_head/smoke.py --capacity-only
```

### Local usefulness smoke (torch)

```bash
uv run --with torch python experiments/tlm_fixed_head/smoke.py
uv run --with torch python experiments/tlm_fixed_head/smoke.py --science --steps 60
```

## Capacity conclusion (measured at 0.8.2)

See committed `capacity_report.json` / `smoke_results.json`.

**Verdict:** the verified two-step Fixed head’s **15 effect classes are a fatal
bottleneck** for modular arithmetic usefulness under the configured residue→Event
probe embedding:

| p | class bits | MI(c; class) | mean outputs colliding per class | class-only test acc | fatal? |
|---|---|---|---|---|---|
| 13 | 3.91 | 0.60 bits | 6.64 | ~chance | yes (near-chance B; weak MI) |
| 97 | 3.91 | 0.23 bits | 89.4 | ~chance | **yes** (info + measured) |

Raw word labels retain far more MI to `c` than the 15-class quotient (e.g. ~6.1
bits at p=97). That pins a **failure mode of the 15-class vocabulary under this
probe**, not yet the full obstruction locus — see `characterize_obstruction.py`
and Quilt 007.02/007.03 before choosing a result interface.

## Out of scope

Physical Test Realization; published grokking SOTA; softmax-as-OT-law; waiting for
another Quilt theory turn — this PR *is* the capacity answer.

## Obstruction characterization (Issue 007 — no fork)

Diagnostic-only probes (not a product head):

```bash
uv run --frozen python experiments/tlm_fixed_head/characterize_obstruction.py
```

Writes `characterization_probe.json`. See Quilt `007.02` (agenda) / `007.03` (results).
Does not choose a result-interface fork.

## p=13 Theory-41 word-slot coarse-grain

Diagnostic for Issue 007 correction (Theory-41 `e_c = ∑_{q(w)=c} e_w` on injective p=13 embed):

```bash
uv run --frozen python experiments/tlm_fixed_head/probe_p13_coarse_grain.py
```

Artifact: `p13_coarse_grain_report.json`. Branch: `experiment/007-p13-word-slot-coarse-grain`.
Does **not** choose a 007 fork. Quilt: `007.04`.

Verdict is negative: 11/13 distinct class-count rows (`{1,12}` and `{2,11}` collide),
`span_rank(e_c)=4=dim C`, and only `{3,6,7,10}` ever strictly maximize.

**Tie discipline (017.24 lesson).** A result is credited only when it is the *sole*
maximizer. Results 2 and 11 carry identical effects, so they always tie; the
convention alone decides which one a naive readout reports — `argsort(...)[-1]`
credits 11, plain `argmax` credits 2, each 6648 times out of 20000. Neither is a
winner, and `fraction_unique_argmax = 0.6676` is the honest number. Pinned in
`topographo/tests/test_p13_coarse_grain_00704.py`.

## Futurator / Outcome 017 (finite FIPS proxy)

**Fence:** configured finite FIPS-proxy experiments on OT two-program family
`P_seq` / `P_grp`. Not modular grokking; not a language-model result; not physical
Event supply. Issue 017 is **closure-ready / closed-pending-Owner** after 017.24.

### Main entrypoints (discoverable)

| Script | Role |
|---|---|
| `probe_futurator_program_space.py` | 017.07/017.08 exact witness + FIPS-proxy scan |
| `learned_admissibility_01709.py` | continuous geometric-admissibility arms |
| `learned_admissibility_01711_discrete.py` | discrete/catalogue denotation head |
| `learned_admissibility_01713_curriculum.py` | two-phase curriculum |
| `learned_admissibility_01715_ceiling.py` | recovered-geometry ceiling / policy shortfall |
| `learned_admissibility_01717_catalogue_recovery.py` | Rec_CE / Rec_ray catalogue recovery |
| `learned_admissibility_01719_exact_eval.py` | exact evaluator reconciliation |
| `learned_admissibility_01721_structural_controls.py` | Cyc vs Ambient vs Rewired controls |
| `learned_admissibility_01723_final_audit.py` | **final:** order/graph audit fixes + GroupedFirst |
| `test_01719_exact_evaluator.py` / `test_01721_phi_and_controls.py` / `test_01723_final_audit.py` | unit regressions |

### Banked arc (through 017.24)

- **017.08** smoke: exact §5 witness; Occ≠Sand; ambient Mul control; 192/288 FIPS disagree
- **017.10** continuous dens: primary A fails; geometry matters when given (B≈0.91)
- **017.12** discrete head unlocks hard FIPS edges; A still weak held-out
- **017.14** curriculum; freeze-ε stronger (~0.41)
- **017.16** reachability dominates (`1−H≫H−A`)
- **017.18** Rec_ray raises H→1.0 / idx 16/16 (train-only)
- **017.20** Phi root-cause; recovery survives exact scorer
- **017.22** Native/Ambient/Rewired controls (Phi forward); Native A=H=1.0 under training
- **017.24** corrected order audit (seed1 perm not robust); exact non-iso on 9 masks;
  **GroupedFirst Native A=1.0 = learned Native** (learned π adds no accuracy here)

```bash
uv run --frozen python experiments/tlm_fixed_head/probe_futurator_program_space.py
uv run python -m unittest discover -s experiments/tlm_fixed_head -p 'test_017*.py' -v
uv run python experiments/tlm_fixed_head/learned_admissibility_01723_final_audit.py
```

Pinned 017.22/017.24 manifest SHA256:
`81a4700e8cf1bac0f66e473d144c4ccb003fe26f8cb7175108fe8884effd107e`.

Also retained (Issue 006/007 capacity probes): `smoke.py`, `capacity.py`,
`characterize_obstruction.py`, `probe_p13_coarse_grain.py` — separate from the 017 Futurator ladder.

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


## Futurator program-space smoke (Outcome 017.07)

Compare strict OT programs vs ambient Mul control (no decoder):

```bash
uv run --frozen python experiments/tlm_fixed_head/probe_futurator_program_space.py
```

- Strict: `P_seq=Occ(a,Occ(b,r))` vs `P_grp=Occ(Cyc(a,b),r)` on exact §5 witness + finite FIPS-proxy scan
- Control: generalized Mul trees / bilateral brackets (ambient, not OT-native)
- Artifact: `futurator_program_space_report.json`


## 017 Futurator program-space / learned admissibility

- `probe_futurator_program_space.py` — 017.07/017.08 exact witness + FIPS-proxy scan
- `learned_admissibility_01709.py` — 017.09 native-structure learned geometric-admissibility arms A–F
- `learned_admissibility_01711_discrete.py` — 017.11 discrete/catalogue denotation head (Gumbel-STE); arms A/B/C/D + oracle
- `learned_admissibility_01713_curriculum.py` — 017.13 discrete curriculum / two-phase (force-seq ε warm-start → unlock π)
- `learned_admissibility_01715_ceiling.py` — 017.15 evaluation-only recovered-geometry ceiling / policy shortfall
- `learned_admissibility_01717_catalogue_recovery.py` — 017.17 catalogue-recovery under freeze (Rec_CE / Rec_ray)
- Draft experiment/result: `017.11`/`017.12` discrete head; `017.13`/`017.14` curriculum; `017.15`/`017.16` ceiling; `017.17`/`017.18` catalogue-recovery; prior `017.10`
- Reports: `futurator_program_space_report.json`, `learned_admissibility_report.json`, `learned_admissibility_01711_report.json`, `learned_admissibility_01713_report.json`, `learned_admissibility_01715_report.json`, `learned_admissibility_01717_report.json` (+ slim/tiny); per-example under `01715_artifacts/`; checkpoints under `01717_artifacts/`

Finite FIPS Cyc proxy only; not physical OT / modular grokking.

### 017.11 / 017.12 status (brief)

Discrete catalogue head **unlocks hard FIPS edges** (A ~17 edges; C ~53) vs continuous 017.10's hard-adm 0. Primary arm A still fails held-out exact (~0.035); B ~0.91; C ~0.37 with strong graph recovery.

### 017.13 / 017.14 status (brief)

Curriculum (800 force-seq ε → 800 unlock π): **A_cur** exact test ~0.19; **A_cur_freeze** (~ε frozen in phase2) ~0.41 — both rise materially above A_joint ~0.035. Joint phase2 degrades recovered geometry; freeze preserves C-like graph. Issue 017 **not** closed.

### 017.15 / 017.16 status (brief)

Ceiling audit at frozen hard denotations: recovered **H≈0.46** vs true-ε* **H=1.0**; freeze policy **A≈0.41** so **H−A≈0.05** while **1−H≈0.54**. Joint phase2 cuts H (~0.46→0.20). Primary diagnosis: **reachability** deficit dominates branch-selection. Issue 017 **not** closed.

### 017.17 / 017.18 status (brief)

Catalogue-recovery under freeze: **Rec_CE** (true-index teacher diagnostic) and **Rec_ray** (train-only P_seq inverse votes) both reach **H=1.0**, idx **16/16**, A≈0.92 — matching B; Baseline_freeze remains H≈0.46 / idx~10/16. Longer force-seq alone does not lift H. Issue 017 **not** closed.


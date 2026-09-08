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
bits at p=97), so any useful modular readout needs **word-level labels and/or a
result quotient / multi-head composition outside the 15 Fixed classes** — the
15-class Fixed projection alone does not injectively or usefully encode
`(a,b)→c` for science-scale `p`.

## Out of scope

Physical Test Realization; published grokking SOTA; softmax-as-OT-law; waiting for
another Quilt theory turn — this PR *is* the capacity answer.

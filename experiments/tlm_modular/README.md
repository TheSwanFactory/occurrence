# TLM-1 modular smoke harness (Issue 005 Milestone T)

Reproducible A/B/C smoke for modular-arithmetic completion with exact FIPS closure
on topographo **0.8.1**. Implements the pins in `005.07` §§2–5 (amending `005.04`
Milestone T). This is **not** a Futurator rewrite and **not** TLM-2.

## Required config pins

| Pin | Default smoke value |
|---|---|
| Init of `x0`, `z0` | **Datum-dependent** embeddings / frame logits (escapes 4-partner bottleneck) |
| Reset | **Per example** |
| Forward rule | **Staged sync**: read `x,z` → `w=Phi(d)` → `z_next=exact_third` → `x_next=action` → commit (not live AB/BA) |
| Steps / order | `n_closure_steps=1`; order `(read, phi, third, action, commit)` |
| Admission | **Masked** to admissible partners (default); unrestricted is named `T1-U` |
| Adaptation budget | `adapt_budget_steps=20` linear readout head only |

See `config.py` (`CI_SMOKE` for p=13 CI-length; `SCIENCE` for p=97).

## Models

- **A** — capacity-matched tiny transformer baseline (shared depth/width)
- **B** — TLM with exact FIPS third table + OT-scaffolded action (masked)
- **C** — matched **non-isomorphic** degree-matched rewiring control (`topographo.ssd.structural_control`, C1–C5)
- **D** — optional label-permutation equivariance helper only (not an ablation)

## Alternate-role protocol

1. Partition by **complete modular triples** before rendering roles (`data.partition_modular_triples`)
2. Train forward `(a,b)->c` only
3. Freeze relational substrate; evaluate `(a,b,?)`, `(a,?,c)`, `(?,b,c)` with role tags
4. Report **zero-shot** and **budgeted-head** separately
5. Metrics: `forward_completion_accuracy`, `alternate_role_*`, `relation_transfer_gap`, `wall_clock_sec`

## Library adapters (stable)

- `topographo.ssd.fips_adapter` — Event index, hard third = Fraction fixtures, STE surrogate route (no optimizer imports)
- `topographo.ssd.structural_control` — degree-matched rewiring + C1–C5 certificates

Heavy training code stays under `experiments/tlm_modular/`.

## How to run

### Unit tests (no torch required)

```bash
uv run --frozen pytest topographo/tests/test_structural_control.py \
  topographo/tests/test_fips_adapter.py topographo/tests/test_tlm_modular_data.py -q
uv run --frozen pytest topographo/tests -q
```

### Local torch smoke (thebeast)

```bash
# install once if needed
python3 -m pip install --user torch

uv run --with torch python experiments/tlm_modular/smoke.py
# science-sized smoke (still short; not full grokking):
uv run --with torch python experiments/tlm_modular/smoke.py --science --steps 50
```

CI does **not** require multi-minute GPU runs. Torch smoke is local / optional.

## Hardware notes (recorded at 0.8.1)

| Item | Value |
|---|---|
| Host | `thebeast.lan` (Apple M4 Pro, 48 GiB) |
| Torch | 2.14.0 (pip user / `uv run --with torch`) |
| Device | MPS available on thebeast (smoke used MPS) |
| Default smoke | `p=13`, 1 layer, `d_model=32`, 30 steps |

## Out of scope

TLM-2 probability head; Theory-41; residue UpdateLaw; published grokking SOTA;
soft-ranked μ; fitting `p` to 84; live AB/BA as default transition law.

## Capacity note

Encoder depth/width are matched across A/B/C. B/C additionally include frame
logits, event embeddings, and action/readout paths required by the TLM dataflow,
so active parameter counts are higher than A. Counts are logged in
`smoke_results.json` (`n_params`). Unused parameter padding is intentionally
avoided (005.07 section 5).

# 014.06 tied Event identity experiment bundle

This directory contains the completed authorized Issue 014.05a execution supporting
`../014.06-Coder-tied-Event-identity-grokking-ablation-result-GPT.md`. The accepted
014.03a relation reconstruction, exact 48/96/24 benchmark, probe, schedules,
scoring, serialization, temporal definitions, and replay semantics were reused
read-only. The only conceptual main-model change is one shared `event_emb[84,64]`
Parameter used as both query identity and candidate identity.

Both mandatory arms completed all eight seeds through 65,536 updates. Each arm fit
TRAIN on 8/8 seeds by update 8192, so the fallback was not invoked. Neither arm
reached behavioral or representation generalization: the result is T1, with all
16 fitted runs reading M4. The certified probe gate passed at 0.9896 ROLE / 1.0000
NOVEL.

## Immutable inputs and fail-closed setup

`bootstrap-pins` copies all eleven accepted pins from
`../014.03a-Code-attachments/pins/` and creates fail-closed placeholders for:

- `pins/014.05a-dispatch.md` — the exact 014.05a dispatch bytes;
- `pins/014.05-spec.md` — the exact incorporated 014.05 bytes;
- `pins/014.04-owner.md` — the exact Owner adjudication bytes.

Replace each placeholder with raw bytes fetched from the URI written inside it.
The harness authenticates those bytes against the versioned single-part S3 ETag
(MD5) observed while resolving each URI, then records SHA-256, byte count, and
resolution evidence in every contract; an arbitrary non-placeholder replacement
is rejected. `contracts` also hashes the revision-pinned accepted 014.03a result,
benchmark, analysis, negative diagnostic, probe controls, and original frame. It
refuses to write experiment state while any placeholder remains. It
SHA-256 hashes every pin, reads the accepted benchmark directly from
`../014.03a-Code-attachments/benchmark_contract.json`, requires raw digest
`26f816cc5ae514cd73ad8d3603da70ac08aee99bd8343b8a0fc6f3f730f7e2dd`,
locally reconstructs the predecessor split, and fails as
`BLOCKED-BENCHMARK-MISMATCH` on any count, block, array, split digest, held-out
position/triple, seen position, or designated-role difference. It never searches
for an alternative split.

The contracts stage writes and reads back exactly these pre-run contracts:

```text
tied_benchmark_contract.json
tied_probe_contract.json
environment.json
```

Every later scientific stage requires the three-file readback proof. The frozen
mechanical identity audit checks that there is exactly one `[84,64]` Event
Parameter, that it is named only `event_emb`, that query and candidate object IDs
are identical, and that there is no Parameter alias or copy. Each trajectory
carries a passing copy of that audit.

## Exact staged commands

Run from this directory with the repository environment active:

```bash
python run_014_tied.py bootstrap-pins
# Replace the three EXACT_RAW_PIN_REQUIRED files with exact immutable raw bytes.
python run_014_tied.py contracts
python run_014_tied.py probe-controls
python run_014_tied.py capacity-calibration
```

The probe stage reruns accepted summed Arm B, summed matched Arm C, and seeded
64-dimensional Gaussian controls with explicit
`torch.optim.AdamW(lr=0.01, betas=(0.9,0.999), eps=1e-8,
weight_decay=0.001)` for 5,000 full-batch steps. Arm B must reach at least 0.95 on
both ROLE and NOVEL or execution stops `BLOCKED-PROBE-REGRESSION`.

Run all sixteen mandatory primary records:

```bash
for seed in 2000 2001 2002 2003 2004 2005 2006 2007; do
  python run_014_tied.py trajectory --seed "$seed" --arm matched_regularized
done
for seed in 2000 2001 2002 2003 2004 2005 2006 2007; do
  python run_014_tied.py trajectory --seed "$seed" --arm matched_zero_wd
done
python run_014_tied.py capacity-decision
```

`capacity-decision` requires all 16 primary records and reads only whether exact
TRAIN fit occurred by update 8192. If either primary reaches at least 7/8, it
writes a small not-invoked `trajectory_capacity_fallback.json` and fallback runs
are forbidden. If neither qualifies but the pre-scored 5x5 calibration selected
a 4/4 candidate, it authorizes exactly these eight conditional runs:

```bash
# Run only when trajectory_capacity_fallback.json has status AUTHORIZED.
for seed in 2000 2001 2002 2003 2004 2005 2006 2007; do
  python run_014_tied.py trajectory --seed "$seed" --arm capacity_fallback
done
```

Then aggregate and run the two analysis passes:

```bash
python run_014_tied.py aggregate
python analyze_014_tied.py                 # pass 1: temporal + tied/untied comparison
python run_014_tied.py transplant
python analyze_014_tied.py                 # pass 2: folds in transplant + comparison
```

The aggregate names are exactly:

```text
trajectory_matched_regularized.json
trajectory_matched_zero_wd.json
trajectory_capacity_fallback.json
representation_probe_trajectory.json
```

`comparison_to_01403a.json` is emitted by both analyzer passes. It loads the
accepted predecessor `analysis.json` and `negative_result_diagnostic.json`, keeps
the untied/tied architectural distinction explicit, applies T0–T4 without
claiming necessity or unique causality, and preserves the task fences.

## Runtime and checkpoint schema

No long scored training is part of harness installation. On the predecessor's
single-CPU measurements, sixteen 65,536-update arms took roughly 13 minutes;
expect similar order of magnitude here, plus the TRAIN-only 5x5 calibration,
about half that again if an eight-seed fallback is authorized, and a few minutes
for fitted-seed transplant replays. The 23 fresh 5,000-step probes per run are a
substantial part of runtime. Actual elapsed seconds are recorded per raw run.

`checkpoints/trajectory_<arm>_seed<seed>.json` is the durable raw unit. Each file
contains seed, arm, exact optimizer config, horizon, initial/final parameter
digests, the passing identity audit, 138 behavioral rows, 23 probe rows, no-early-
stopping declaration, and elapsed time. Behavioral rows contain exact TRAIN
loss/accuracy, ROLE/NOVEL accuracy, swap invariance, total/trunk/bias norms,
shared Event norm/effective rank, and MLP effective ranks. Probe rows contain all
six frozen readout metrics and the canonical relation certificate whenever exact.
`checkpoints/manifest.json` records byte hashes, schema, resume instructions, and
self-audit. Existing scored records are never overwritten, preventing favorable
reruns.

The transplant is restricted to fitted `matched_regularized` seeds. It replays
`t_fit`/`t_gen` when distinct and otherwise `t_fit`/final, checks complete
behavioral-row equality (including shared Event norm), partitions
`REPRESENTATION = event_emb` and `DOWNSTREAM = MLP + candidate bias`, and evaluates
all four combinations without retraining.

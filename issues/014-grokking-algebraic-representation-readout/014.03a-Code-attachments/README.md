# 014.03a-Code-attachments — corrected execution, complete

Attachments for `014.03a-Coder-readable-algebraic-grokking-trajectory-result-GPT.md`.

This bundle is the **superseding corrected execution** of the frozen task. It
supersedes `014.03`, which stopped at `BLOCKED` because the frozen representation
probe could not clear its own certified-SFP positive-control gate. The single
correction is the resolution of one ambiguous phrase in 014.02 section 4; every
other frozen choice is unchanged, and the benchmark is provably identical.

## The one correction

014.02 section 4 freezes the probe optimizer as `Adam` with `weight decay 1e-3`.
In PyTorch that phrase has two implementations: a coupled L2 term inside
`torch.optim.Adam`, and decoupled weight decay as in `torch.optim.AdamW`. Section
5 names `AdamW` explicitly for the main-model regime grid, so `014.03` executed
the literal coupled reading. Under it the certified code reaches only 0.9167 ROLE
and 0.6667 NOVEL and the instrument fails its own gate.

This bundle resolves the phrase to decoupled weight decay, preserving the declared
value `1e-3`. The same probe then reaches **0.9896 ROLE / 1.0000 NOVEL** on the
certified code and clears the gate.

Proof that nothing else moved: `benchmark_contract.json` here is **byte-identical**
to the one in `014.03`, digest
`26f816cc5ae514cd73ad8d3603da70ac08aee99bd8343b8a0fc6f3f730f7e2dd`.

## Result in one line

All sixteen scored runs read **M4**: the learner memorized every designated
answer and never acquired held-out consequence, and the representation never
became readable. A clean, uniform negative.

## Files

| file | contents |
|---|---|
| `benchmark_contract.json` | relation certification, holdout rule, 48/96/24 split, architecture, regime manifest, schedules, thresholds, temporal definitions, baselines |
| `probe_contract.json` | the frozen probe plus the correction record |
| `environment.json` | interpreter, torch, platform, determinism settings, pinned-input digests |
| `contract_readback.json` | byte-level readback record for the three contracts |
| `probe_controls.json` | the three declared controls and the **PASS** gate verdict |
| `selected_regime.json` | the 3x3 calibration, the mechanical selector-boundary audit, the selected regime and its matched control |
| `trajectory_regularized.json` | eight scored runs of the selected regime |
| `trajectory_weight_decay_zero.json` | eight matched `weight_decay = 0` control runs |
| `representation_probe_trajectory.json` | 23 probe checkpoints per run, both arms |
| `component_transplants.json` | the predeclared four-way transplant, with bit-identical replay verification |
| `analysis.json` | temporal diagnostics, M1-M4 readings, per-arm summaries, separated claims A-E |
| `negative_result_diagnostic.json` | post-hoc mechanistic explanation of the negative |
| `run_014.py`, `analyze_014.py` | the executables; identical bytes to the `014.03` copies |
| `checkpoints/` | raw per-run records plus `manifest.json` with digests and resume instructions |
| `pins/` | local copies of all eleven revision-pinned inputs, with digests |

## Reproducing

```bash
cd issues/014-grokking-algebraic-representation-readout/014.03a-Code-attachments
python run_014.py --bundle 014.03a contracts
python run_014.py --bundle 014.03a probe-controls        # gate: PASS
python run_014.py --bundle 014.03a select-regime
for s in 2000 2001 2002 2003 2004 2005 2006 2007; do
  for a in regularized weight_decay_zero; do
    python run_014.py --bundle 014.03a trajectory --seed $s --arm $a
  done
done
python run_014.py --bundle 014.03a aggregate
python analyze_014.py --bundle 014.03a                   # pass 1
python run_014.py --bundle 014.03a transplant
python run_014.py --bundle 014.03a negative-diagnostic
python analyze_014.py --bundle 014.03a                   # pass 2, folds the transplant in
```

`analyze_014.py` runs twice because the predeclared transplant consumes the
predeclared `t_fit` / `t_gen`. The temporal definitions do not depend on the
transplant, so both passes agree on every temporal quantity; pass 2 only adds the
component-level section.

Total cost on one CPU core: roughly 13 minutes for the sixteen scored runs, plus
about 3 minutes each for the transplant and diagnostic replays. Determinism
requires the single-thread settings recorded in `environment.json`.

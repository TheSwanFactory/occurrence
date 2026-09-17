# 014.03-Code-attachments — literal execution, stopped at BLOCKED

Attachments for `014.03-Coder-readable-algebraic-grokking-trajectory-result-GPT.md`.

This bundle is the **immutable literal execution** of the frozen task. It stopped
at the predeclared `BLOCKED` state of 014.02 section 4: the frozen representation
probe could not clear its own certified-SFP positive-control gate, so the
expensive main trajectory was never run.

The superseding corrected execution is `014.03a`. This bundle is not rewritten to
match it.

## What ran

| stage | command | outcome |
|---|---|---|
| relation reconstruction | `python run_014.py --bundle 014.03 contracts` | certified by two independent code routes |
| benchmark freeze | same stage | 48 / 96 / 24 committed and read back |
| representation instrument | `python run_014.py --bundle 014.03 probe-controls` | **BLOCKED** |
| cause isolation | `python run_014.py --bundle 014.03 probe-diagnostics` | isolated to one ambiguous phrase |

## Files

| file | contents |
|---|---|
| `benchmark_contract.json` | relation certification, holdout rule, 48/96/24 split, architecture, regime manifest, schedules, thresholds, temporal definitions, baselines |
| `probe_contract.json` | the frozen probe exactly as executed here, including the literal `Adam` optimizer reading |
| `environment.json` | interpreter, torch, platform, determinism settings, pinned-input digests |
| `contract_readback.json` | byte-level readback record for the three contracts |
| `probe_controls.json` | the three declared controls and the gate verdict |
| `probe_instrument_diagnostic.json` | post-gate isolation of the cause |
| `run_014.py` | the executable; carries both delivery bundles |
| `pins/` | local copies of all eleven revision-pinned inputs, with digests |

`benchmark_contract.json` deliberately carries no wall-clock field, so its bytes
and digest are reproducible. Two consecutive runs of the `contracts` stage
produce byte-identical `benchmark_contract.json` and `probe_contract.json`.

## What is absent, and why

`selected_regime.json`, the two trajectory files,
`representation_probe_trajectory.json`, `component_transplants.json`,
`analysis.json` and `analyze_014.py` do not exist in this bundle. That is what
`BLOCKED` means here: 014.02 section 4 directs the execution to stop *before* the
main trajectory when the positive control fails, and no model was trained beyond
the calibration-free contract stage. No main-model checkpoint was ever produced
or observed in this bundle.

## Reproducing

```bash
cd issues/014-grokking-algebraic-representation-readout/014.03-Code-attachments
python run_014.py --bundle 014.03 contracts
python run_014.py --bundle 014.03 probe-controls      # exits non-zero: BLOCKED
python run_014.py --bundle 014.03 probe-diagnostics
```

Determinism requires the single-thread CPU settings recorded in
`environment.json`; `run_014.py` sets the thread environment before importing
torch.

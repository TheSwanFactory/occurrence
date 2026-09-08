# 068 — Minimal mutual Operational-Frame machine

This executable experiment uses topographo's `ssd.frames` module (Theory 068.02 /
Outcome 15/18 Native/SCE/cyclic rules). It compares live peer presentation with
explicitly staged incoming inputs. It does not specify physical encounter
selection.

The reusable API lives in `topographo.ssd.frames`. Local `runtime.py` is a thin
re-export for audit scripts.

## A. Run

From the repository root (Python 3.12+):

```bash
uv run --frozen pytest topographo/tests/test_frames.py -q
uv run --frozen python experiments/mutual-frames/audit.py --check
uv run --frozen python experiments/mutual-frames/analyze_traces.py
```

The package tests cover the seven behavioral witnesses. The full exact census
takes several minutes and is **not** part of default pytest CI. `audit.py
--check` reruns every case and compares the recorded JSON and deterministic
compressed trace archive byte for byte. Omit `--check` to regenerate both
files. `analyze_traces.py` independently derives `ordering_metrics.json` from
the recorded transitions. Compressed-byte reproducibility is pinned to the
recorded environment; another Python/zlib version may require comparing the
decompressed JSON instead.

## B. Files

- `../../topographo/ssd/frames.py`: local state types, exact guards, released-engine
  execution, and explicit serial/snapshot diagnostic policies.
- `runtime.py`: thin re-export of `topographo.ssd.frames` (+ `exact`) for scripts.
- `audit.py`: exhaustive finite and adversarial audit, including rational partner
  directions, phase recurrence, scaling and symmetry checks.
- `../../topographo/tests/test_frames.py`: seven regression/domain tests, including
  8,400 independent source-guard versus output-eigenspace comparisons.
- `results.json`: census, complete witnesses, provenance and repeated traces.
- `traces.json.gz`: all 197,728 census cases, referencing 18,848 distinct local
  decisions and their exact inputs, outputs, sources and law provenance.
- `analyze_traces.py` / `ordering_metrics.json`: complete-case ordering analysis.
- `068.02-HC-minimal-mutual-operational-frame-machine-result.md`: interpretation.
- `publication.json`: Quilt/Theory 068.02 provenance snapshot.
- `requirements.txt`: recorded dependency versions at publication time.

Trace indices and A/B labels are observer bookkeeping. They never enter a frame
or select an operation. Each case contains AB, BA and staged-input results. The
reverse staged evaluation order is checked separately for equality in every
case. Frame and decision tables deduplicate data without dropping any case.

## C. Main observation

With live presentation, all 3,024 basic Native pairs completing both serial
orders have different ordered endpoints, for each of the four fixed local-law
configurations. Staged Native encounters complete on 6,720 pairs; none of those
complete staged endpoints equals either complete live serial endpoint.

All 112,896 pairs of basic cyclic edges commute. This is independent autonomous
continuation, not ongoing reciprocal feedback. A cyclic frame can present its
current participant to a Native neighbor, but the certified cyclic transition
does not consume that neighbor's offer.

The primary verdict is **D — NON-SERIALIZABLE / LAW-CONFLICT BOUNDARY**, scoped to
the tested live presentation interface. Fixed-input local execution remains
deterministic. No new coordination primitive is proved necessary.

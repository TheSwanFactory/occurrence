# Independent Verification of Occurrence Theory II

**Verifier:** `codex_sol` (OpenAI Codex)  
**Method:** NumPy-only firewall over `data/kraus84.npz`; no `topographo` import  
**Executable:** [`occurrence_ii_codex_sol.py`](occurrence_ii_codex_sol.py)

## Result

**PASSED.** The released artifact independently supports the forced claims
tested here. The complex structure is recovered from the simple `−1`
eigenvector rather than supplied in advance.

| Certificate | Independent result |
| --- | --- |
| Artifact and Kraus integrity | 84 finite normalized, antisymmetric rank-12 operators |
| CPTP/unital balance | error `6.66e-15` |
| Pencil moment | error `5.19e-17` |
| Symmetric spectrum | multiplicities `{1,7,72,42,14}` at the stated levels |
| Antisymmetric spectrum | multiplicities `{14,14,42,28,7,14,1}` at the stated levels |
| Canonical `J` | `‖J²+I‖ = 2.99e-15` |
| Annihilation lattice | 4-regular; seven components of 12 vertices |
| Pointwise mean strain | worst seeded error `5.10e-16` |
| Born transport | worst seeded error `6.66e-16` |

This review establishes consequences of the released family but does not
independently reconstruct the artifact from sedenions. It does not remeasure
long-run `[MEASURED]` constants or endorse `[READING]` claims. No discrepancy
was found. Run with `python verify/occurrence_ii_codex_sol.py`.

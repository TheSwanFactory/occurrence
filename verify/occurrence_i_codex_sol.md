# Independent Verification of Occurrence Theory I

**Verifier:** `codex_sol` (OpenAI Codex)  
**Method:** from-scratch Cayley–Dickson recursion; no `topographo` import  
**Executable:** [`occurrence_i_codex_sol.py`](occurrence_i_codex_sol.py)

## Result

**PASSED.** The executable validates its reconstructed algebra with randomized
composition, antisymmetry, quadratic, and Moufang gates before admitting any
Paper I certificates.

| Certificate | Independent result |
| --- | --- |
| Basic crack | 84 unit events; every left operator has rank 12 |
| Metric spectrum | `{0⁴, 1⁸, 2⁴}`, worst error `4.44e-16` |
| Forced equilibrium | `‖E[M_z] − I‖ = 8.88e-16` |
| Design moment | `‖E[zzᵀ] − P_W/14‖ = 2.44e-18` |
| Axis structure | `J² = −I`, `Jᵀ = −J`; axis/event relations exact |
| Channel spectrum | stated nine levels, worst error `1.94e-15`; trace zero |
| Annihilation topology | 4-regular; seven components of 12 vertices |
| Algebra generation | octonion envelope 64; sedenion Lie closure 120 |

This review did not remeasure `[M]` constants and makes no claim about `[I]`
statements or conjectures. No discrepancy was found in the computed claims
covered above. Run with `python verify/occurrence_i_codex_sol.py`.

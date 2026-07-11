# Occurrence Theory II

## The Physics of the Kraus-84 Family

**E. N. Prabhakar** ([Radical Centrism](https://radicalcentrism.org) / [iHack.us](https://ihack.us))
with **Bench d'Claude** (Anthropic)

**Ledger notation.** Every substantive claim carries exactly one tag:
**[C]** computation (exact numerical certificate, threshold 10⁻¹²); **[M]** measurement (Monte Carlo, with error bars); **[READING]** physical interpretation (not forced). Claims without tags are definitions or conventions.

> **Status.** This is the Paper II landing slot introduced by the `-ii` work.
> The narrative document is the explainer PDF [`docs/Occurrence_Theory.pdf`](docs/Occurrence_Theory.pdf);
> the machine-checked ledger is [`verify/occurrence_ii_audit.py`](verify/occurrence_ii_audit.py),
> which runs on the shared ground-truth family [`data/kraus84.npz`](data/kraus84.npz).
> Prose paragraphs of the paper are filled in incrementally against that ledger.

---

## 1. The firewall

Paper I ([`occurrence-theory.md`](occurrence-theory.md)) derives the
**Kraus-84 family** — the 84 left-multiplication operators of the unit
zero-divisor variety of the sedenions, with the uniform Aut-invariant measure.
Paper II asks a narrower question: *treat those 84 real 16×16 matrices as the
only given, forget the algebra that produced them, and see how much physics is
forced.*

That discipline is enforced structurally. The family is generated **once**,
from the blessed generator `topographo.core.cayley_dickson_table`, and frozen
into [`data/kraus84.npz`](data/kraus84.npz) — the shared ground truth that every
downstream derivation and every independent reviewer loads and diffs. After that
single provenance step the algebra "leaves the stage": Parts 1–8 use pure numpy
on the 84 matrices and nothing else.

## 2. What is forced [C]

The audit certifies each of the following against a threshold; any failure
exits nonzero and invalidates the run:

1. **Equilibrium** — `E[KᵀK] = I`: the channel is doubly stochastic; trace is
   conserved for every state.
2. **The spectrum** — nine distinct levels, decay rates quantized in sevenths
   (except one), with G₂ multiplicities.
3. **The clock** — the unique −1 eigenmode is an orthogonal complex structure
   `J` with `J² = −I`: the channel manufactures the complex numbers.
4. **Born transport & energy** — the second moment's 2-dimensional kernel is the
   spine `span{e₀, Je₀}`; transported probability equals the Hermitian modulus
   over the normalization cost, exactly.
5. **The coherence constant** — the slowest coherence decay ratio locks to
   `2√3/7` in the antisymmetric sector.
6. **Asymptotics** — everything dies except `span{I, J} ≅ ℂ`, on which the
   evolution is the unitary ℤ₂ clock.
7. **Locality** — the annihilation graph is 7 disjoint 12-event cells,
   4-regular, diameter 3 — one cell per Fano point.

## 3. What is only measured [M] / read [READING]

The temperature `Var[τ] ≈ 1/18`, the stationary spine share `s* ≈ 0.132`, and
the quenched Lyapunov exponent are **measured** Monte-Carlo estimates, reported
with error bars but **not** gated. The physical identifications — Born rule,
energy, temperature, arrow of time, low-energy quantum mechanics as the
peripheral spectrum — are **readings**: interpretations of the forced
structure, not claims the audit tests.

## 4. Reproducing and reviewing

- Run the ledger: `python verify/occurrence_ii_audit.py` (exit 0 = every
  certificate passed). It also runs as an exit-code gate in the `occurrence`
  CI workflow.
- Independent reviewers: drop `verify/occurrence_ii_<yourhandle>.<ext>` into
  [`verify/`](verify/README.md). Reviewer cells **must not import `topographo`**
  — reproduce the family yourself from `data/kraus84.npz` (or your own
  regeneration) so the check is genuinely independent.

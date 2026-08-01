# The Saga of the 𝔭-Sector

*Being a true account of how a fourteen-dimensional eigenspace was mistaken for
a hero, unmasked by an honest machine, and restored to its rightful place —
told so that reviewers may trust the map by knowing the territory it crossed.*

> A dimension is not a name.
> Fourteen is the size of the adjoint —
> and also of two sevens holding hands.
> The algebra does not care which you meant.

This is the provenance of Conjecture C3 and Open Problem 3 in *Occurrence
Theory II*: the reasoning, the wrong turns, and the corrections. The
machine-checked endpoint is
[`verify/occurrence_ii_reptheory.sage`](../verify/occurrence_ii_reptheory.sage);
what follows is the road to it.

## I. Why we sent for Sage

The numpy audits already reproduce every `[FORCED]` claim to machine precision.
But floating point cannot *prove*, and it cannot *name*. Two obligations wanted
exactly that:

- **exactness** — that the spectrum is truly `(1/7)·{0, ±1, ±3, ±7}` and
  `±2√3/7`, not merely a number within `10⁻¹⁴` of it;
- **representation theory** — that the multiplicities `{1, 7, 14, 21, …}` are
  the G₂ and SU(3) representations the paper calls them, and not integers that
  merely happen to fit.

Those are SageMath's country: exact linear algebra over ℚ, Weyl character rings,
branching rules. So the exact / rep-theory work became its own cell, standing
beside the fast numerical audit, not over it.

## II. The first stone — `8 ⊕ 6` falls

C3 claimed the 14-dimensional 𝔭-sector splits as `8 ⊕ 6` under SU(3) ⊂ G₂. A
short branching settled it:

- the standard **7** splits as `3 ⊕ 3̄ ⊕ 1` — the fingerprint of the canonical
  long-root SU(3);
- the **adjoint 14** splits as `8 ⊕ 3 ⊕ 3̄`, *not* `8 ⊕ 6`.

The `6` is the *symmetric* irrep; `3 ⊕ 3̄` is a different animal that merely
shares its dimension. So `8 ⊕ 6` became `8 ⊕ 3 ⊕ 3̄`.

And there the error hid in plain sight: we had branched *the adjoint* — because
`14 = dim 𝔤₂`, and a coincidence of numbers had been quietly promoted to an
identity of things. The 𝔭-sector had not yet been asked what it actually was.

## III. The unmasking — the 𝔭-sector is `7 ⊕ 7`

Turning the probe into a full test meant no longer assuming. We built the
*concrete* G₂ = Aut(𝕆) action from the derivations `Der(𝕆)`, and asked each
eigenspace directly what representation it carried. Three answers came back:

- **The symmetry is real, not hoped-for:** `‖[S, g⊗g]‖ = 1.3·10⁻¹⁶`. Φ commutes
  with the group action; the eigenspaces truly are G₂-modules.
- **Every sector, named at last** (only the irreps `{1, 7, 14, 27}` appear):

  | eigenvalue | dim | G₂-module | SU(3) content |
  | --- | --- | --- | --- |
  | ±1 | 1 | `1` | `1` |
  | ±𝔭 = ±2√3/7 | 14 | **`7 ⊕ 7`** | `2·(3 ⊕ 3̄ ⊕ 1)` — two matter families, **no octet** |
  | ±3/7 | 21 | `7 ⊕ 14` | `8 ⊕ 2·(3 ⊕ 3̄) ⊕ 1` — **one octet** + two matter families |
  | ±1/7 | 42 | `1 ⊕ 2·7 ⊕ 27` | — |
  | 0 | 100 | `4·1 ⊕ 2·7 ⊕ 2·14 ⊕ 2·27` | — |

  Totals: `End(ℝ¹⁶) = 8·1 ⊕ 12·7 ⊕ 4·14 ⊕ 4·27`.

- **The verdict:** the 𝔭-sector is **`7 ⊕ 7`**, not the adjoint. Bluntly,
  `χ_E(g) = 2·χ₇(g)`, while the adjoint's character sits far away. The
  fourteen-ness was a mask.

This reversed Act II. Both `8 ⊕ 6` and its "fix" `8 ⊕ 3 ⊕ 3̄` had been pinned to
the wrong sector, because both branched the adjoint — which the 𝔭-sector is not.

## IV. Where the hero actually lives

The four true adjoint (`14`) copies dwell in the **±3/7 sectors** (each
`7 ⊕ 14`) and the `0` sector. So the gluon octet appears only there:

```text
±3/7 = 7 ⊕ 14  →  8 ⊕ 2·(3 ⊕ 3̄) ⊕ 1
```

A retraction became a *better* conjecture. Open Problem 3 was reframed —
**Gauge structure of spectral sectors**: the 𝔭-sector carries two matter
families and no octet; the ±3/7 sectors carry one octet and two matter families;
the open question is whether the two admit a single gauge-theoretic reading, and
whether the channel canonically selects the SU(3) at all.

## V. A gift on the way out (§10.2)

The G₂-invariant symmetric forms on the pencil `W = 7 ⊕ 7` span a
**3-dimensional** space. So the Design Theorem's `E[zzᵀ] = P_W/14` is *not*
forced by symmetry alone — the measure earns its keep. That tells the §10.2
proof precisely what it must still show.

## VI. What held, and what it taught

- **The firewall never moved.** Every reversal lived in `[CONJECTURE]` — the
  SU(3) reading. The `[FORCED]` §2–7 mathematics (CPTP, the nine-level spectrum
  now proven *exactly over ℚ*, the complex structure J, the Born identity, the
  event lattice) stood untouched through all of it. Being wrong out loud is safe
  when it is quarantined to the conjectures.
- **The machine caught it, not our memory.** A claim written as a computation —
  *build the action, decompose, assert* — cannot flatter itself the way a
  sentence can.
- **The moral, carved for next time:** *a dimension is not a name.* Compute the
  module structure before editing the paper; the size of a sector never tells
  you which representation it is.

## Reproduce

```bash
sage verify/occurrence_ii_reptheory.sage   # exit 0; every check OK
```

Verified with SageMath 10.9. Companions:
[`verify/occurrence_ii_reptheory.md`](../verify/occurrence_ii_reptheory.md)
(results) and [`verify/README.md`](../verify/README.md) (conventions).

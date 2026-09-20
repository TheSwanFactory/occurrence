# Issue 015 — Learning law for consequence structure

> **Status:** COMPLETE
> **Terminal interpretation:** **C3** — role-neutral competitive relation
> learning is sufficient, relative to designated-query cross-entropy, for robust
> seen-triad consequence on this frozen finite benchmark.

Within the Issue 015 evidence bundle, this closure preserves every accepted
015.02, 015.04 and 015.07 result and attachment unmodified and adds only this
status/index/reproduction guide and the restored revision-pinned blind packet.
No shared-code extraction or refactor was performed; the frozen experiment code
remains unchanged.

## Accepted turns

| Turn | Accepted result and implementation | Role in the chain | Exact outcome |
|---|---|---|---|
| 015.02 | [Blind identifiability audit result](015.02-Coder-blind-identifiability-audit-result-GPT.md) · [audit](015.02-Code-attachments/audit_identifiability.py) | Separates literal supervision from minimal role-neutral triad semantics | Literal: ROLE 0/96 forced, NOVEL 0/24 forced. Triad semantics: ROLE 96/96 forced, NOVEL 0/24 forced. |
| 015.04 | [Role-neutral triad learning result](015.04-Coder-role-neutral-triad-learning-result-GPT.md) · [runner](015.04-Code-attachments/run_015_role.py) | Tests exact triad symmetry with designated-query cross-entropy | Triad-symmetric arm: sustained ROLE >=95% in 0/8 seeds and exact ROLE in 0/8. Terminal bin R1. |
| 015.07 | [Competitive consequence result](015.07-Coder-role-neutral-competitive-consequence-result-GPT.md) · [runner](015.07-Code-attachments/run_015_competitive.py) | Changes only the TRAIN objective to exhaustive local relational competition | Relation arm: exact local relation in 8/8 and exact ROLE in 8/8; query-CE baseline replicated exactly. Terminal bin C3. |

## Result chain

1. **Identifiability is assumption-relative.** Under the literal learner-visible
   class of all input-swap-invariant pair-to-Event functions, TRAIN constrains
   only its 48 observed pairs. Every held-out query admits all 84 Events, so
   neither ROLE (0/96) nor NOVEL (0/24) is forced. Under the separately stated
   minimal role-neutral triad semantics—three distinct Events, every pair
   completes to the third, and no pair belongs to two blocks—the same TRAIN
   observations force all 96 ROLE completions. All 24 NOVEL queries remain
   underdetermined, each with 79 compatible candidates.
2. **Exact score symmetry does not make those consequences behaviorally
   available under designated-query supervision.** The exact triad-symmetric
   scorer trained by 82-way cross-entropy only on each designated TRAIN query
   fits TRAIN in 8/8 seeds but reaches neither sustained 95% ROLE nor exact ROLE
   in any seed (0/8).
3. **TRAIN-only local competition is sufficient relative to that baseline.**
   For each of the 48 observed positive triads, the relational objective makes
   it compete against all 243 pair-preserving one-vertex corruptions (three
   branches of 81). The resulting arm reaches exact local-relation domination
   and exact ROLE in 8/8 seeds. Its matched query-CE arm exactly replicates the
   accepted 015.04 triad-symmetric trajectory: zero shared-value deviations and
   zero environment differences.

The terminal conclusion is deliberately narrow: on this frozen observed-triad
substrate, exhaustive TRAIN-derived relational competition is sufficient
**relative to designated-query cross-entropy** for the already-identifiable ROLE
consequences. No necessity result follows.

## Immutable provenance

All package references below name `occurrence/gpt` in the `s3://protology`
registry.

### External source turns

| Source | Revision |
|---|---|
| 015.01 blind identifiability task and input packet | `cdc3b67e9ca3c3082616ad0f5328587d926efb8872ff90dcd0e3939c448dfb98` |
| 015.03 role-learning benchmark and task | `6ceb230dae9e069011062ca88b13198042cee20b0ba8dc85ec4f3cf1623f5731` |
| 015.05 Owner adjudication and 015.06 competitive-learning task | `522ff97ad90d2462595398ed0271ff6219d15ce68a0ca654c7cbd4490445aaac` |

### Accepted packages

| Bundle | Revision |
|---|---|
| Accepted 015.04 package used as the 015.07 predecessor | `27612778252e0858328b4325154d07f068eafe6cc961dfe36c0861cbf79b5bf4` |
| Final accepted Issue 015 package | `4768ae4cd3ffccf4f1d983e3154e4fec25e19b342b0d9de2484da8837929ea00` |

### Precommit readbacks

| Turn | Read-back package revision | `precommit.json` SHA-256 |
|---|---|---|
| 015.04 | `45f62ae4a246ffbbd7747f0c18c21ee1a72c66a31aad2b7e598837ec00d77321` | `718bd4c756eb95999422fd0f516517fa5ec227e3e2ddb77c273ab1809eccec8f` |
| 015.07 | `9cb58b90c299807b88a475eabb2af37ae1126a5e64a0bfe7e7207b9384f31a38` | `e14af6665f09b670cad4499585e9f39ade935ed7676e682ed0954bd68c3b5928` |

### Restored blind packet

The pinned packet is restored at
[`015.01-Code-attachments/014-literal-information-boundary.json`](015.01-Code-attachments/014-literal-information-boundary.json)
from the identical logical path in
`occurrence/gpt@cdc3b67e9ca3c3082616ad0f5328587d926efb8872ff90dcd0e3939c448dfb98`.
The local file preserves the package entry's raw bytes.

```text
raw SHA-256                  231b18f0e66894e197636eb09b876558ff33bb277e4b207539c56328bd116e87
normalized semantic SHA-256 364e9191ac66abd59ac580734c0146fa54d2b8a2fdfd4c6a467477ef6c990aa1
```

## Reproduction and no-training verification

Run from the repository root. These commands do no model training. The 015.04
and 015.07 `verify` subcommands read accepted artifacts only and print their
reports.

### 015.02 exact audit

This is the accepted default reproduction command:

```bash
python issues/015-learning-law-for-consequence-structure/\
015.02-Code-attachments/audit_identifiability.py
```

It performs exact deterministic combinatorics rather than training. Unlike the
later `verify` subcommands, its default mode rewrites the accepted 015.02 JSON
and Markdown outputs; run it only from a clean worktree and confirm that it
produces no diff.

### 015.04 accepted-artifact verification

```bash
python issues/015-learning-law-for-consequence-structure/\
015.04-Code-attachments/run_015_role.py verify \
  --out-dir issues/015-learning-law-for-consequence-structure/\
015.04-Code-attachments \
  --result issues/015-learning-law-for-consequence-structure/\
015.04-Coder-role-neutral-triad-learning-result-GPT.md
```

### 015.07 accepted-artifact verification

This command uses the accepted local 015.04 triad-symmetric trajectory as the
predecessor and therefore needs no package fetch:

```bash
python issues/015-learning-law-for-consequence-structure/\
015.07-Code-attachments/run_015_competitive.py verify \
  --predecessor-trajectory-source \
    issues/015-learning-law-for-consequence-structure/\
015.04-Code-attachments/trajectory_triad_symmetric.json \
  --out-dir issues/015-learning-law-for-consequence-structure/\
015.07-Code-attachments \
  --result issues/015-learning-law-for-consequence-structure/\
015.07-Coder-role-neutral-competitive-consequence-result-GPT.md
```

## Scientific fences

- The positive result concerns only the 48 observed finite triads and their 96
  already-identifiable alternate-role consequences.
- The neural experiments neither load nor score NOVEL_TEST. The exact audit
  leaves all 24 NOVEL queries underdetermined; this closure makes no NOVEL
  learning claim.
- The 56-block global relation is neither discovered nor evaluated. No global
  completion, SFP/Fano/habitat-coordinate, or continuous-algebra claim follows.
- C3 establishes sufficiency relative to the matched designated-query CE
  baseline. It does not establish necessity, optimality, uniqueness, or a
  preferred OT learning law.
- The 243-corruption neighborhoods are deterministic structure licensed from
  TRAIN positives; the learner is not claimed to discover those alternatives
  spontaneously.
- No unique readable algebra in Event embeddings, generic consequence-learning
  reduction, language-scale result, physical Event supply, or physical Outcome
  semantics is claimed.
- An unforced literal query is not thereby unlearnable under every useful
  inductive bias, and a negative result for query CE does not refute other
  architectures or objectives.
- Accepted result documents, trajectories, analyses, precommits and runners are
  frozen. This closure performs no shared-code extraction or experiment-code
  refactor.

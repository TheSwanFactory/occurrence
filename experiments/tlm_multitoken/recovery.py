"""Train-only Event-denotation recovery for `008.07` Ladder D.

`008.05` handed the policy the exact Event rays directly, so there was nothing to
recover: the denotations *were* the input. Ladder D asks whether the learned
compositional advantage survives when denotations are no longer supplied
perfectly, which only has content if the denotations are hidden behind something.
So this module adds the one interface change Ladder D requires, and nothing else.

```text
008.05   exposed input = three certified Event rays          (denotations given)
008.07   exposed input = three TOKENS over a fixed vocabulary (denotations hidden)
```

Everything `008.06` section 7 pins stays fixed: `E^3 R`, the strict
`Occ + Cyc + Sand` calculus, `r = e4`, the twenty planar programs, the `008.02`
sign-bit target, the `StructuralPolicy`, the `motif` / `motif_parity` splits, the
baseline set, and the absence of any result decoder. The token layer is the
minimal addition that makes "recovered denotation" well-typed, and it is
reported as an interface change rather than smuggled in as if the task were
untouched — see the `008.08` result turn.

Recovery method
---------------
The banked Outcome-017 `Rec_ray` path: **endpoint-inverse voting**, train-only.
At training time the machine observes pairs

```text
(token triple, exact native target endpoint)
```

and nothing else — not the true denotations, not which program the hidden grammar
selected. For every catalogue triple and every strict program it asks whether the
resulting exact endpoint coincides projectively with an observed *training*
endpoint, and if so casts one vote per slot:

```text
for (c1,c2,c3) in catalogue^3, for p in programs:
    key = canonical(evaluate(p, (E[c1],E[c2],E[c3])))
    for (t1,t2,t3) in observed_train.get(key, ()):
        votes[t1][c1] += 1;  votes[t2][c2] += 1;  votes[t3][c3] += 1
```

This is `017.23`'s `build_votes` generalized from two slots to three and from two
programs to twenty. `017` could materialize its inverse map because at `E^2 R`
with two programs it has 14112 entries; at `E^3 R` with twenty it has 11854080,
which does not fit in memory as a dict of ray keys. The transfer is therefore
**streamed** rather than materialized: identical arithmetic and identical votes,
constant memory beyond the observed-endpoint index. Measured at about nine
CPU-minutes for the full pass.

Recovery is deliberately **grammar-agnostic**. It votes over all twenty programs
rather than the eight `Cyc`-free bracketings the `008.02` grammar can actually
reach, even though that pool is public. Restricting to the reachable pool would
hand recovery a piece of the relation the policy is supposed to learn.

Leak boundary
-------------
`008.07` section 4 requires training-side observations only.
:func:`audit_recovery_leak` checks it mechanically rather than by assertion in
prose: the observed index is rebuilt from the split and compared against the
held-out records, and recovery is refused if any held-out token triple, endpoint
key, or target program reached it.

Ties
----
The catalogue argmax is not always unique, and hiding that would misreport
identifiability. The convention is `017.23`'s consistent **first max** — the
leftmost index attaining the maximum, i.e. plain ``argmax`` semantics — and every
tied token is recorded by identity, together with the recovered map's sensitivity
to catalogue order under that convention.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from experiments.tlm_multitoken.probe_program_space import (
    Evaluator,
    retained_ray,
    sign_bit,
)
from experiments.tlm_multitoken.programs import evaluate, render
from experiments.tlm_multitoken.task import (
    Pool,
    Record,
    Split,
    TaskConfig,
    materialize_pool,
)
from topographo.ssd import fips_basic, projective

__all__ = [
    "RECOVERY_FENCE",
    "RecoveredPools",
    "Recovery",
    "VocabConfig",
    "audit_recovery_leak",
    "build_vocabulary",
    "materialize_token_pool",
    "observed_train_index",
    "recover_by_endpoint_votes",
    "recover_observation_sweep",
    "recovered_pool",
    "recovered_reachable_ceiling",
    "recovery_diagnostics",
]

RECOVERY_FENCE = (
    "denotations are recovered from train-side (token triple, exact target "
    "endpoint) observations only; no held-out target label, program assignment "
    "or endpoint influences the recovered map, and the map is frozen before any "
    "held-out evaluation"
)


@dataclass(frozen=True)
class VocabConfig:
    """The frozen token layer. Serialized verbatim into the report."""

    n_tokens: int = 16
    #: Frozen selection seed. Separate from the recovery seed and the policy seed.
    vocab_seed: int = 807_001
    #: Recovery has no stochastic stage in the voting method, but `008.07`
    #: section 4 requires a slot for one, recorded separately from policy seeds.
    recovery_seed: int = 807_002

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "vocab_seed": self.vocab_seed,
            "recovery_seed": self.recovery_seed,
            "catalogue_size": len(fips_basic.EVENTS),
            "selection": (
                "sign-balanced uniform sample without replacement: n_tokens/2 "
                "Events with sign bit 1 and n_tokens/2 with sign bit 0"
            ),
            "why_not_017_densest_subgraph": (
                "017 picked the densest FIPS subgraph to maximize Cyc "
                "admissibility. Here the 008.02 target lives entirely on the "
                "Cyc-free bracketings, so maximizing Cyc density would only "
                "inflate the legal-program count with distractors and change the "
                "availability structure relative to 008.05. A sign-balanced "
                "uniform sample is the neutral choice and keeps all eight sign "
                "patterns equinumerous."
            ),
        }


def build_vocabulary(config: VocabConfig = VocabConfig()) -> tuple[int, ...]:
    """Frozen token -> catalogue-index map. Sign-balanced, so all 8 patterns occur."""
    if config.n_tokens % 2:
        raise ValueError("n_tokens must be even to balance the sign bit")
    positive = [k for k, e in enumerate(fips_basic.EVENTS) if sign_bit(e) == 1]
    negative = [k for k, e in enumerate(fips_basic.EVENTS) if sign_bit(e) == 0]
    half = config.n_tokens // 2
    if len(positive) < half or len(negative) < half:
        raise ValueError("catalogue cannot supply a balanced vocabulary this large")
    rng = random.Random(config.vocab_seed)
    chosen = sorted(rng.sample(positive, half) + rng.sample(negative, half))
    return tuple(chosen)


def materialize_token_pool(
    true_idx: Sequence[int], *, task_config: TaskConfig | None = None
) -> Pool:
    """The true-denotation arm: exhaustive over ``n_tokens^3`` token triples.

    Reuses ``task.materialize_pool`` unchanged, with the vocabulary passed as the
    Event list, so ``Record.event_indices`` are token indices and every number is
    produced by the same tested machinery `008.05` used.
    """
    events = [fips_basic.EVENTS[k] for k in true_idx]
    config = task_config or TaskConfig(pool_size=len(events) ** 3)
    return materialize_pool(config, events=events)


# --- recovery ---------------------------------------------------------------


@dataclass
class Recovery:
    """A frozen recovered denotation map plus its identifiability record."""

    true_idx: tuple[int, ...]
    recovered_idx: tuple[int, ...]
    votes: tuple[tuple[int, ...], ...]
    tied_tokens: tuple[int, ...]
    zero_vote_tokens: tuple[int, ...]
    n_observed_endpoints: int
    n_train_records: int
    stream_evaluations: int
    wall_sec: float
    tie_convention: str = "consistent first max (leftmost argmax), 017.23"
    fence: str = RECOVERY_FENCE

    @property
    def n_tokens(self) -> int:
        return len(self.true_idx)

    def match_count(self) -> int:
        return sum(1 for t, r in zip(self.true_idx, self.recovered_idx, strict=True) if t == r)

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "true_idx": list(self.true_idx),
            "recovered_idx": list(self.recovered_idx),
            "catalogue_match_count": self.match_count(),
            "catalogue_match_rate": self.match_count() / self.n_tokens,
            "tied_tokens": list(self.tied_tokens),
            "n_tied_tokens": len(self.tied_tokens),
            "zero_vote_tokens": list(self.zero_vote_tokens),
            "n_observed_train_endpoints": self.n_observed_endpoints,
            "n_train_records": self.n_train_records,
            "stream_evaluations": self.stream_evaluations,
            "tie_convention": self.tie_convention,
            "wall_sec": self.wall_sec,
            "fence": self.fence,
        }


def observed_train_index(
    pool: Pool, split: Split, true_idx: Sequence[int]
) -> dict[tuple, list[tuple[int, ...]]]:
    """Map observed exact target endpoint -> the training token triples behind it.

    This is the entire training-side observation: a token triple and the exact
    native endpoint of the answer. No program identity, no denotation. It is the
    only thing recovery is allowed to read, which is why the leak audit rebuilds
    it independently and compares.
    """
    events = tuple(fips_basic.EVENTS)
    retained = retained_ray()
    observed: dict[tuple, list[tuple[int, ...]]] = defaultdict(list)
    for i in split.train:
        record = pool.records[i]
        if not record.target_defined:
            continue
        combo = tuple(events[true_idx[k]] for k in record.event_indices)
        value = evaluate(pool.programs[record.target_program], combo, retained)
        observed[projective.canonicalize(value)].append(record.event_indices)
    return dict(observed)


def recover_by_endpoint_votes(
    pool: Pool,
    split: Split,
    true_idx: Sequence[int],
    *,
    config: VocabConfig = VocabConfig(),
    catalogue: Sequence[int] | None = None,
    programs=None,
    progress: bool = True,
) -> Recovery:
    """Stream the exact program space and vote per token slot. Train-only."""
    started = time.time()
    events = tuple(fips_basic.EVENTS)
    retained = retained_ray()
    all_programs = programs if programs is not None else pool.programs
    candidates = list(range(len(events)) if catalogue is None else catalogue)

    observed_map = observed_train_index(pool, split, true_idx)
    n_tokens = len(true_idx)
    votes = [[0] * len(events) for _ in range(n_tokens)]
    evaluator = Evaluator(retained)
    evaluations = 0
    total = len(candidates) ** 3
    for n, (c1, c2, c3) in enumerate(itertools.product(candidates, repeat=3)):
        combo = (events[c1], events[c2], events[c3])
        for program in all_programs:
            value = evaluator(program, combo)
            evaluations += 1
            if value is None:
                continue
            hits = observed_map.get(projective.canonicalize(value))
            if not hits:
                continue
            for t1, t2, t3 in hits:
                votes[t1][c1] += 1
                votes[t2][c2] += 1
                votes[t3][c3] += 1
        if progress and n % 100_000 == 0 and n:
            done = n / total
            elapsed = time.time() - started
            print(
                f"    recovery stream {done:6.1%}  {elapsed:6.1f}s  "
                f"eta {elapsed / done - elapsed:6.1f}s",
                flush=True,
            )

    recovered: list[int] = []
    tied: list[int] = []
    zero: list[int] = []
    for token in range(n_tokens):
        row = votes[token]
        best = max(row)
        if best == 0:
            zero.append(token)
            # Nothing to go on. Keep the convention honest: leftmost candidate.
            recovered.append(candidates[0])
            continue
        winners = [k for k, v in enumerate(row) if v == best]
        if len(winners) > 1:
            tied.append(token)
        recovered.append(winners[0])

    return Recovery(
        true_idx=tuple(true_idx),
        recovered_idx=tuple(recovered),
        votes=tuple(tuple(row) for row in votes),
        tied_tokens=tuple(tied),
        zero_vote_tokens=tuple(zero),
        n_observed_endpoints=len(observed_map),
        n_train_records=len(split.train),
        stream_evaluations=evaluations,
        wall_sec=time.time() - started,
    )


def recover_observation_sweep(
    pool: Pool,
    split: Split,
    true_idx: Sequence[int],
    *,
    fractions: Sequence[float] = (0.002, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0),
    config: VocabConfig = VocabConfig(),
    catalogue: Sequence[int] | None = None,
    progress: bool = True,
) -> list[dict]:
    """How much training observation does exact recovery actually need?

    Recovery at this cell is not merely successful, it is *over-determined*: the
    full training half recovers all sixteen tokens with vote margins around
    ``10^4``. A result that only reported "recovery was perfect" would leave the
    Ladder-D boundary untested, so this sweep replaces that single fact with a
    curve, by fitting the same voting recovery to nested subsamples of the
    training observations.

    It is a **diagnostic on the recovery interface**, not a repair and not a
    second arm: the task, the splits, the target and both policy arms are
    untouched. Nested subsamples share one streamed pass over the exact program
    space, so the whole curve costs about what one recovery costs.

    Returns one entry per fraction, smallest first, each with the recovered map
    and its identifiability record.
    """
    started = time.time()
    events = tuple(fips_basic.EVENTS)
    retained = retained_ray()
    programs = pool.programs
    candidates = list(range(len(events)) if catalogue is None else catalogue)
    n_tokens = len(true_idx)

    usable = [i for i in split.train if pool.records[i].target_defined]
    order = list(usable)
    random.Random(config.recovery_seed).shuffle(order)

    ladders: list[dict] = []
    for fraction in sorted(fractions):
        count = max(1, int(round(len(order) * fraction)))
        subset = order[:count]
        observed: dict[tuple, list[tuple[int, ...]]] = defaultdict(list)
        for i in subset:
            record = pool.records[i]
            combo = tuple(events[true_idx[k]] for k in record.event_indices)
            value = evaluate(programs[record.target_program], combo, retained)
            observed[projective.canonicalize(value)].append(record.event_indices)
        ladders.append(
            {
                "fraction": fraction,
                "n_train_observations": count,
                "n_distinct_endpoints": len(observed),
                "observed": dict(observed),
                "votes": [[0] * len(events) for _ in range(n_tokens)],
            }
        )

    evaluator = Evaluator(retained)
    total = len(candidates) ** 3
    for n, (c1, c2, c3) in enumerate(itertools.product(candidates, repeat=3)):
        combo = (events[c1], events[c2], events[c3])
        for program in programs:
            value = evaluator(program, combo)
            if value is None:
                continue
            key = projective.canonicalize(value)
            for rung in ladders:
                hits = rung["observed"].get(key)
                if not hits:
                    continue
                votes = rung["votes"]
                for t1, t2, t3 in hits:
                    votes[t1][c1] += 1
                    votes[t2][c2] += 1
                    votes[t3][c3] += 1
        if progress and n % 100_000 == 0 and n:
            done = n / total
            elapsed = time.time() - started
            print(
                f"    sweep stream {done:6.1%}  {elapsed:6.1f}s  "
                f"eta {elapsed / done - elapsed:6.1f}s",
                flush=True,
            )

    out: list[dict] = []
    for rung in ladders:
        votes = rung["votes"]
        recovered: list[int] = []
        tied: list[int] = []
        zero: list[int] = []
        for token in range(n_tokens):
            row = votes[token]
            best = max(row)
            if best == 0:
                zero.append(token)
                recovered.append(candidates[0])
                continue
            winners = [k for k, v in enumerate(row) if v == best]
            if len(winners) > 1:
                tied.append(token)
            recovered.append(winners[0])
        match = sum(
            1 for t, r in zip(true_idx, recovered, strict=True) if t == r
        )
        signs = sum(
            1
            for t, r in zip(true_idx, recovered, strict=True)
            if sign_bit(events[t]) == sign_bit(events[r])
        )
        margins = []
        for token in range(n_tokens):
            row = votes[token]
            best = max(row)
            runner_up = max((v for v in row if v < best), default=0)
            margins.append(best - runner_up)
        out.append(
            {
                "fraction": rung["fraction"],
                "n_train_observations": rung["n_train_observations"],
                "n_distinct_endpoints": rung["n_distinct_endpoints"],
                "catalogue_match_count": match,
                "catalogue_match_rate": match / n_tokens,
                "sign_bit_preserved_count": signs,
                "n_tied_tokens": len(tied),
                "tied_tokens": tied,
                "n_zero_vote_tokens": len(zero),
                "min_vote_margin": min(margins),
                "median_vote_margin": sorted(margins)[n_tokens // 2],
                "recovered_idx": recovered,
            }
        )
    return out


def audit_recovery_leak(
    pool: Pool, split: Split, true_idx: Sequence[int], recovery: Recovery
) -> dict:
    """Mechanically verify the `008.07` section 4 leak boundary.

    Rebuilds the observation index from the training half and checks that no
    held-out token triple, held-out target endpoint, or held-out target program
    could have reached recovery. The endpoint check is the sharp one: a held-out
    endpoint that also occurs in training is not a leak, but a held-out endpoint
    absent from training must never appear in the observation index.
    """
    events = tuple(fips_basic.EVENTS)
    retained = retained_ray()

    def key_of(record: Record) -> tuple:
        combo = tuple(events[true_idx[k]] for k in record.event_indices)
        return projective.canonicalize(
            evaluate(pool.programs[record.target_program], combo, retained)
        )

    train_triples = {pool.records[i].event_indices for i in split.train}
    test_triples = {pool.records[i].event_indices for i in split.test}
    train_keys = {
        key_of(pool.records[i]) for i in split.train if pool.records[i].target_defined
    }
    test_keys = {
        key_of(pool.records[i]) for i in split.test if pool.records[i].target_defined
    }
    train_programs = {pool.records[i].target_program for i in split.train}
    test_programs = {pool.records[i].target_program for i in split.test}

    observed = observed_train_index(pool, split, true_idx)
    observed_triples = {t for triples in observed.values() for t in triples}

    return {
        "n_train_records": len(split.train),
        "n_test_records": len(split.test),
        "observed_endpoint_keys": len(observed),
        "observed_triples_subset_of_train": observed_triples <= train_triples,
        "no_heldout_triple_observed": not (observed_triples & test_triples),
        "observed_keys_subset_of_train_keys": set(observed) <= train_keys,
        "heldout_only_endpoints_never_observed": not (
            (test_keys - train_keys) & set(observed)
        ),
        "heldout_only_target_programs": sorted(
            render(pool.programs[p]) for p in test_programs - train_programs
        ),
        "recovery_used_n_observed_endpoints": recovery.n_observed_endpoints,
        "recovery_observation_count_matches_audit": (
            recovery.n_observed_endpoints == len(observed)
        ),
        "fence": RECOVERY_FENCE,
        "leak_free": (
            observed_triples <= train_triples
            and not (observed_triples & test_triples)
            and set(observed) <= train_keys
            and not ((test_keys - train_keys) & set(observed))
            and recovery.n_observed_endpoints == len(observed)
        ),
    }


# --- the recovered-denotation arm ------------------------------------------


@dataclass
class RecoveredPools:
    """The recovered arm, sharing one endpoint-class namespace with the truth."""

    pool: Pool
    n_endpoint_classes: int
    #: Per record: did recovery change the legal set / target definedness /
    #: target endpoint. Aggregated by :func:`recovery_diagnostics`.
    legal_set_changed: tuple[bool, ...]
    target_became_undefined: tuple[bool, ...]
    target_endpoint_changed: tuple[bool, ...]


def recovered_pool(
    true_pool: Pool, true_idx: Sequence[int], recovery: Recovery
) -> RecoveredPools:
    """Rebuild the pool under recovered denotations, scored against true targets.

    The machine now holds the recovered rays, so the programs it can run and the
    endpoints it reaches are the recovered ones. Success is still measured
    against the **ground-truth** target endpoint, because recovery degrades what
    the learner has, not what is true. That is what makes the four `008.07`
    section 6 failure modes separable.
    """
    events = tuple(fips_basic.EVENTS)
    retained = retained_ray()
    programs = true_pool.programs
    class_of: dict[tuple, int] = {}
    records: list[Record] = []
    features: list[tuple[float, ...]] = []
    legal_changed: list[bool] = []
    became_undefined: list[bool] = []
    endpoint_changed: list[bool] = []

    for true_record in true_pool.records:
        tokens = true_record.event_indices
        true_combo = tuple(events[true_idx[k]] for k in tokens)
        rec_combo = tuple(events[recovery.recovered_idx[k]] for k in tokens)

        true_target = evaluate(
            programs[true_record.target_program], true_combo, retained
        )
        if true_target is None:
            target_class = -1
        else:
            key = projective.canonicalize(true_target)
            target_class = class_of.setdefault(key, len(class_of))

        classes: list[int] = []
        for program in programs:
            value = evaluate(program, rec_combo, retained)
            if value is None:
                classes.append(-1)
                continue
            key = projective.canonicalize(value)
            classes.append(class_of.setdefault(key, len(class_of)))

        defined = tuple(c >= 0 for c in classes)
        pattern = 0
        for k, ok in enumerate(defined):
            if ok:
                pattern |= 1 << k
        records.append(
            Record(
                event_indices=tokens,
                sign_pattern=true_record.sign_pattern,
                target_program=true_record.target_program,
                target_class=target_class,
                defined=defined,
                endpoint_class=tuple(classes),
                availability_pattern=pattern,
            )
        )
        out: list[float] = []
        for value in (*rec_combo, retained):
            out.extend(float(c) for c in value)
        features.append(tuple(out))

        legal_changed.append(set(true_record.legal) != set(records[-1].legal))
        became_undefined.append(
            true_record.target_defined and not defined[true_record.target_program]
        )
        endpoint_changed.append(
            true_record.target_defined
            and defined[true_record.target_program]
            and classes[true_record.target_program] != target_class
        )

    pool = Pool(
        config=true_pool.config,
        programs=programs,
        program_names=true_pool.program_names,
        records=tuple(records),
        features=tuple(features),
        n_endpoint_classes=len(class_of),
        evaluator_calls=0,
    )
    return RecoveredPools(
        pool=pool,
        n_endpoint_classes=len(class_of),
        legal_set_changed=tuple(legal_changed),
        target_became_undefined=tuple(became_undefined),
        target_endpoint_changed=tuple(endpoint_changed),
    )


def recovery_diagnostics(
    true_pool: Pool,
    recovered: RecoveredPools,
    recovery: Recovery,
    *,
    splits: Sequence[Split] = (),
) -> dict:
    """The `008.07` section 6 report, keeping the four failure modes apart."""
    events = tuple(fips_basic.EVENTS)
    n = len(true_pool.records)

    per_token = []
    for token in range(recovery.n_tokens):
        true_event = events[recovery.true_idx[token]]
        rec_event = events[recovery.recovered_idx[token]]
        row = recovery.votes[token]
        best = max(row) if row else 0
        runner_up = max((v for v in row if v < best), default=0)
        per_token.append(
            {
                "token": token,
                "true_catalogue_index": recovery.true_idx[token],
                "recovered_catalogue_index": recovery.recovered_idx[token],
                "exact_match": recovery.true_idx[token] == recovery.recovered_idx[token],
                "projectively_equal": projective.equivalent(true_event, rec_event),
                "true_sign_bit": sign_bit(true_event),
                "recovered_sign_bit": sign_bit(rec_event),
                "sign_bit_preserved": sign_bit(true_event) == sign_bit(rec_event),
                "max_votes": best,
                "runner_up_votes": runner_up,
                "vote_margin": best - runner_up,
                "tied": token in recovery.tied_tokens,
                "n_candidates_at_max": sum(1 for v in row if v == best and best > 0),
            }
        )

    sign_preserved = sum(1 for e in per_token if e["sign_bit_preserved"])
    order_sensitive = [e["token"] for e in per_token if e["n_candidates_at_max"] > 1]

    by_split = {}
    for split in splits:
        for half, indices in (("train", split.train), ("test", split.test)):
            by_split[f"{split.name}|{half}"] = {
                "n": len(indices),
                "legal_set_changed": _share(recovered.legal_set_changed, indices),
                "target_became_undefined": _share(
                    recovered.target_became_undefined, indices
                ),
                "target_endpoint_changed": _share(
                    recovered.target_endpoint_changed, indices
                ),
            }

    return {
        "representation_recovery": {
            "catalogue_match_count": recovery.match_count(),
            "catalogue_match_rate": recovery.match_count() / recovery.n_tokens,
            "sign_bit_preserved_count": sign_preserved,
            "sign_bit_preserved_rate": sign_preserved / recovery.n_tokens,
            "n_tied_tokens": len(recovery.tied_tokens),
            "tied_tokens": list(recovery.tied_tokens),
            "n_zero_vote_tokens": len(recovery.zero_vote_tokens),
            "order_sensitive_tokens": order_sensitive,
            "order_sensitivity_note": (
                "a token with more than one candidate at the vote maximum is "
                "resolved by the declared first-max convention, so its recovered "
                "identity depends on catalogue order; these are the tokens whose "
                "recovery is not identifiable from the training observations"
            ),
            "per_token": per_token,
            "tie_convention": recovery.tie_convention,
        },
        "program_domain_change": {
            "legal_set_changed_share_all": _share(
                recovered.legal_set_changed, range(n)
            ),
            "target_became_undefined_share_all": _share(
                recovered.target_became_undefined, range(n)
            ),
        },
        "endpoint_change": {
            "target_endpoint_changed_share_all": _share(
                recovered.target_endpoint_changed, range(n)
            ),
            "note": (
                "share of inputs where the target program, executed on recovered "
                "rays, no longer reaches the ground-truth target endpoint; this "
                "bounds the recovered arm's achievable exact-native success from "
                "above, independently of any policy decision"
            ),
        },
        "by_split": by_split,
        "achievable_ceiling_note": (
            "an upper bound on recovered-arm success is the share of inputs where "
            "SOME legal recovered program reaches the ground-truth endpoint; it is "
            "reported per split as recovered_oracle_ceiling in the run report"
        ),
    }


def _share(flags: Sequence[bool], indices) -> float:
    picked = [flags[i] for i in indices]
    return sum(1 for f in picked if f) / (len(picked) or 1)


def recovered_reachable_ceiling(pool: Pool, indices: Sequence[int]) -> float:
    """Share of scored inputs where some legal recovered program is correct.

    Not a baseline and not deployable: it is the ceiling recovery leaves for any
    selector whatsoever, so it separates "recovery destroyed the answer" from
    "the policy failed to find it".
    """
    total = correct = 0
    for i in indices:
        record = pool.records[i]
        if not record.target_defined:
            continue
        total += 1
        if any(record.endpoint_class[k] == record.target_class for k in record.legal):
            correct += 1
    return correct / (total or 1)

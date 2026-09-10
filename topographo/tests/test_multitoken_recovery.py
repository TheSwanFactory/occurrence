"""CI-safe tests for the `008.07` Ladder-D token layer and denotation recovery.

No torch. The authoritative recovery streams all ``84^3 x 20`` exact endpoints and
takes about nine CPU-minutes per split, so these tests drive the same code over a
small catalogue where the stream is a few tens of thousands of evaluations.

What matters here is not accuracy but the boundaries: that the token layer really
hides the denotations, that recovery reads training observations only, that the
tie convention is declared rather than silently resolved, and that the recovered
arm is scored against ground-truth targets so the `008.07` section 6 failure modes
stay separable.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from experiments.tlm_multitoken import recovery, task
from experiments.tlm_multitoken.probe_program_space import retained_ray, sign_bit
from experiments.tlm_multitoken.programs import evaluate
from topographo.ssd import fips_basic, projective

#: A small catalogue that still contains a sign-balanced vocabulary. Chosen by
#: sign so the target grammar reaches more than one bracketing.
_POSITIVE = [k for k, e in enumerate(fips_basic.EVENTS) if sign_bit(e) == 1][:6]
_NEGATIVE = [k for k, e in enumerate(fips_basic.EVENTS) if sign_bit(e) == 0][:6]
CATALOGUE = tuple(sorted(_POSITIVE + _NEGATIVE))
VOCAB = tuple(sorted(_POSITIVE[:2] + _NEGATIVE[:2]))


@lru_cache(maxsize=None)
def _pool() -> task.Pool:
    return recovery.materialize_token_pool(VOCAB)


@lru_cache(maxsize=None)
def _split() -> task.Split:
    return task.build_splits(_pool(), include=("motif",))[0]


@lru_cache(maxsize=None)
def _recovery() -> recovery.Recovery:
    return recovery.recover_by_endpoint_votes(
        _pool(), _split(), VOCAB, catalogue=CATALOGUE, progress=False
    )


# --- the token layer -------------------------------------------------------


def test_vocabulary_is_sign_balanced_and_deterministic():
    config = recovery.VocabConfig()
    first = recovery.build_vocabulary(config)
    assert first == recovery.build_vocabulary(config)
    assert len(first) == config.n_tokens == 16
    bits = [sign_bit(fips_basic.EVENTS[k]) for k in first]
    assert sum(bits) == config.n_tokens // 2, "both sign values must be present"
    assert len(set(first)) == len(first)


def test_vocabulary_refuses_an_unbalanceable_size():
    with pytest.raises(ValueError, match="even"):
        recovery.build_vocabulary(recovery.VocabConfig(n_tokens=15))
    with pytest.raises(ValueError, match="balanced vocabulary"):
        recovery.build_vocabulary(recovery.VocabConfig(n_tokens=100))


def test_token_pool_exposes_tokens_not_events():
    """The whole point of Ladder D: the input is a token, the ray is hidden."""
    pool = _pool()
    assert len(pool.records) == len(VOCAB) ** 3
    for record in pool.records:
        assert all(0 <= k < len(VOCAB) for k in record.event_indices)


def test_token_pool_sign_pattern_comes_from_the_true_denotations():
    pool = _pool()
    for record in pool.records:
        expected = 0
        for k in record.event_indices:
            expected = (expected << 1) | sign_bit(fips_basic.EVENTS[VOCAB[k]])
        assert record.sign_pattern == expected


def test_all_eight_sign_patterns_occur_so_the_motif_split_is_well_formed():
    assert len({r.sign_pattern for r in _pool().records}) == 8


# --- the training observation ---------------------------------------------


def test_observed_index_contains_only_training_triples():
    pool, split = _pool(), _split()
    observed = recovery.observed_train_index(pool, split, VOCAB)
    train = {pool.records[i].event_indices for i in split.train}
    test = {pool.records[i].event_indices for i in split.test}
    seen = {t for triples in observed.values() for t in triples}
    assert seen <= train
    assert not (seen & test)


def test_observed_index_keys_are_exact_target_endpoints():
    pool, split = _pool(), _split()
    retained = retained_ray()
    observed = recovery.observed_train_index(pool, split, VOCAB)
    for key, triples in list(observed.items())[:10]:
        for tokens in triples:
            record = next(r for r in pool.records if r.event_indices == tokens)
            combo = tuple(fips_basic.EVENTS[VOCAB[k]] for k in tokens)
            value = evaluate(pool.programs[record.target_program], combo, retained)
            assert projective.canonicalize(value) == key


# --- recovery --------------------------------------------------------------


def test_recovery_votes_have_catalogue_shape_and_only_score_candidates():
    rec = _recovery()
    assert len(rec.votes) == len(VOCAB)
    for row in rec.votes:
        assert len(row) == len(fips_basic.EVENTS)
        # Only the offered candidates may receive votes.
        assert all(v == 0 for k, v in enumerate(row) if k not in CATALOGUE)


def test_recovery_recovers_within_the_offered_catalogue():
    rec = _recovery()
    assert set(rec.recovered_idx) <= set(CATALOGUE)
    assert len(rec.recovered_idx) == len(VOCAB)


def test_recovery_declares_its_tie_convention_and_records_ties():
    rec = _recovery()
    assert "first max" in rec.tie_convention
    for token in range(rec.n_tokens):
        row = rec.votes[token]
        best = max(row)
        if best == 0:
            assert token in rec.zero_vote_tokens
            continue
        winners = [k for k, v in enumerate(row) if v == best]
        # A tie must be reported, and the resolution must be the leftmost index.
        assert (len(winners) > 1) == (token in rec.tied_tokens)
        assert rec.recovered_idx[token] == winners[0]


def test_recovery_is_grammar_agnostic_over_all_twenty_programs():
    """Restricting to the eight reachable bracketings would leak the grammar."""
    pool = _pool()
    assert len(pool.programs) == 20
    rec = _recovery()
    # 12^3 catalogue triples x 20 programs.
    assert rec.stream_evaluations == len(CATALOGUE) ** 3 * 20


def test_recovery_records_its_observation_count():
    rec, pool, split = _recovery(), _pool(), _split()
    observed = recovery.observed_train_index(pool, split, VOCAB)
    assert rec.n_observed_endpoints == len(observed)
    assert rec.n_train_records == len(split.train)


# --- the leak audit --------------------------------------------------------


def test_leak_audit_passes_for_a_properly_fitted_recovery():
    audit = recovery.audit_recovery_leak(_pool(), _split(), VOCAB, _recovery())
    assert audit["leak_free"] is True
    assert audit["no_heldout_triple_observed"] is True
    assert audit["observed_triples_subset_of_train"] is True
    assert audit["heldout_only_endpoints_never_observed"] is True


def test_leak_audit_catches_a_recovery_fitted_on_the_wrong_half():
    """Swap the halves and the audit must refuse it."""
    pool, split = _pool(), _split()
    swapped = task.Split(
        name=split.name,
        kind=split.kind,
        rationale=split.rationale,
        train=split.test,
        test=split.train,
        metadata=split.metadata,
    )
    leaky = recovery.recover_by_endpoint_votes(
        pool, swapped, VOCAB, catalogue=CATALOGUE, progress=False
    )
    audit = recovery.audit_recovery_leak(pool, split, VOCAB, leaky)
    assert audit["leak_free"] is False


def test_leak_audit_notices_a_mismatched_observation_count():
    rec = _recovery()
    tampered = recovery.Recovery(
        true_idx=rec.true_idx,
        recovered_idx=rec.recovered_idx,
        votes=rec.votes,
        tied_tokens=rec.tied_tokens,
        zero_vote_tokens=rec.zero_vote_tokens,
        n_observed_endpoints=rec.n_observed_endpoints + 1,
        n_train_records=rec.n_train_records,
        stream_evaluations=rec.stream_evaluations,
        wall_sec=rec.wall_sec,
    )
    audit = recovery.audit_recovery_leak(_pool(), _split(), VOCAB, tampered)
    assert audit["recovery_observation_count_matches_audit"] is False
    assert audit["leak_free"] is False


# --- the recovered arm ----------------------------------------------------


def _perfect(rec: recovery.Recovery) -> recovery.Recovery:
    return recovery.Recovery(
        true_idx=rec.true_idx,
        recovered_idx=rec.true_idx,
        votes=rec.votes,
        tied_tokens=(),
        zero_vote_tokens=(),
        n_observed_endpoints=rec.n_observed_endpoints,
        n_train_records=rec.n_train_records,
        stream_evaluations=rec.stream_evaluations,
        wall_sec=0.0,
    )


def test_perfect_recovery_reproduces_the_true_arm_exactly():
    """A sanity anchor: if recovery is right, nothing about the task changed."""
    pool = _pool()
    rebuilt = recovery.recovered_pool(pool, VOCAB, _perfect(_recovery()))
    for true_record, rec_record in zip(pool.records, rebuilt.pool.records, strict=True):
        assert true_record.defined == rec_record.defined
        assert true_record.target_defined == rec_record.target_defined
        assert true_record.matches(true_record.target_program) == rec_record.matches(
            rec_record.target_program
        )
    assert not any(rebuilt.legal_set_changed)
    assert not any(rebuilt.target_became_undefined)
    assert not any(rebuilt.target_endpoint_changed)


def test_perfect_recovery_leaves_a_reachable_ceiling_of_one():
    pool = _pool()
    rebuilt = recovery.recovered_pool(pool, VOCAB, _perfect(_recovery()))
    ceiling = recovery.recovered_reachable_ceiling(
        rebuilt.pool, rebuilt.pool.scored
    )
    assert ceiling == 1.0


def test_recovered_arm_is_scored_against_ground_truth_targets():
    """Recovery degrades what the learner has, not what is true."""
    pool = _pool()
    wrong = recovery.Recovery(
        true_idx=VOCAB,
        recovered_idx=tuple(reversed(VOCAB)),
        votes=_recovery().votes,
        tied_tokens=(),
        zero_vote_tokens=(),
        n_observed_endpoints=0,
        n_train_records=0,
        stream_evaluations=0,
        wall_sec=0.0,
    )
    rebuilt = recovery.recovered_pool(pool, VOCAB, wrong)
    # Target programs and sign patterns are ground truth and must not move.
    for true_record, rec_record in zip(pool.records, rebuilt.pool.records, strict=True):
        assert true_record.target_program == rec_record.target_program
        assert true_record.sign_pattern == rec_record.sign_pattern
    # A wrong denotation map must actually cost something.
    assert any(rebuilt.target_endpoint_changed)
    assert recovery.recovered_reachable_ceiling(
        rebuilt.pool, rebuilt.pool.scored
    ) < 1.0


def test_recovered_pool_features_are_the_recovered_rays():
    pool = _pool()
    rec = _recovery()
    rebuilt = recovery.recovered_pool(pool, VOCAB, rec)
    retained = retained_ray()
    record = rebuilt.pool.records[0]
    expected: list[float] = []
    for k in record.event_indices:
        expected.extend(float(c) for c in fips_basic.EVENTS[rec.recovered_idx[k]])
    expected.extend(float(c) for c in retained)
    assert list(rebuilt.pool.features[0]) == expected


# --- diagnostics ----------------------------------------------------------


def test_diagnostics_keep_the_four_failure_modes_apart():
    pool, rec = _pool(), _recovery()
    rebuilt = recovery.recovered_pool(pool, VOCAB, rec)
    diagnostics = recovery.recovery_diagnostics(
        pool, rebuilt, rec, splits=(_split(),)
    )
    assert set(diagnostics) >= {
        "representation_recovery",
        "program_domain_change",
        "endpoint_change",
        "by_split",
    }
    rep = diagnostics["representation_recovery"]
    assert set(rep) >= {
        "catalogue_match_count",
        "sign_bit_preserved_count",
        "n_tied_tokens",
        "order_sensitive_tokens",
        "per_token",
        "tie_convention",
    }
    assert len(rep["per_token"]) == len(VOCAB)
    for entry in rep["per_token"]:
        assert set(entry) >= {
            "true_catalogue_index",
            "recovered_catalogue_index",
            "exact_match",
            "sign_bit_preserved",
            "vote_margin",
            "n_candidates_at_max",
        }


def test_diagnostics_report_order_sensitivity_for_every_tied_token():
    rec = _recovery()
    pool = _pool()
    rebuilt = recovery.recovered_pool(pool, VOCAB, rec)
    rep = recovery.recovery_diagnostics(pool, rebuilt, rec)["representation_recovery"]
    assert set(rec.tied_tokens) <= set(rep["order_sensitive_tokens"])


def test_recovery_carries_its_fence():
    assert "train-side" in recovery.RECOVERY_FENCE
    assert "frozen before any" in recovery.RECOVERY_FENCE


# --- the split selector ---------------------------------------------------


def test_unseen_event_split_is_omitted_under_the_token_layer():
    """Holding out Event identities would hold out vocabulary entries."""
    names = {s.name for s in task.build_splits(_pool(), include=("motif", "random"))}
    assert names == {"motif", "random"}


def test_build_splits_refuses_an_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        task.build_splits(_pool(), include=("motif", "nonsense"))

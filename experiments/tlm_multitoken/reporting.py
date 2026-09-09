"""Reporting contract for Issue 008 multi-token native composition.

Two separate fences are enforced here, both *before* any learning code exists.

1. **Interface layering (Quilt 007.05).** Klein's fair-grokking conjecture
   separates externally stipulated interface from learned middle from native
   constitution. Issue 008 has the same three layers under different names, and
   008.01 section 4 requires that they never be conflated in a result:

   ```text
   supplied_tree   stipulated bracketing executed for the model
   selected_tree   the learned middle: which legal program to run
   recovered_dens  native constitution: denotations the model recovered
   ```

   An arm declares exactly which layers it exercises. A report that credits
   ``selected_tree`` while silently supplying the tree is rejected.

2. **Decoder validity (007.05 section 8).** A decoder invalidates the comparison
   if it contains a lookup for held-out answers, computes the target relation
   itself, or was fitted using held-out targets. 008.01 section 5.D adds that no
   free decoder may manufacture native result identity.

Nothing here trains, scores, or touches the algebra. It only refuses to let a
result claim more than its configuration earns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "AvailabilityRule",
    "DecoderAudit",
    "Hypothesis",
    "InterfaceLayer",
    "ArmReport",
    "REQUIRED_BASELINES",
    "REQUIRED_DIAGNOSTICS",
    "validate_arm",
    "validate_suite",
]


class InterfaceLayer(str, Enum):
    """The three things 008.01 section 4 forbids conflating."""

    SUPPLIED_TREE = "supplied_tree"
    SELECTED_TREE = "selected_tree"
    RECOVERED_DENS = "recovered_dens"


class Hypothesis(str, Enum):
    """007.05 section 6 claim separation, transposed to program composition.

    ``H_WEAK`` — with a stipulated bracketing, the model executes composition
    correctly on held-out inputs.
    ``H_REL`` — one learned structure selects the correct program across roles /
    motif positions, not just one privileged direction.
    ``H_NATIVE`` — the architecture constitutes the denotations and result
    identity itself, without a stipulated decoder.

    Failure of ``H_NATIVE`` does not falsify ``H_WEAK`` or ``H_REL``; success on
    ``H_WEAK`` does not establish ``H_NATIVE``.
    """

    H_WEAK = "H_weak"
    H_REL = "H_rel"
    H_NATIVE = "H_native"


#: The layer each claim must actually exercise to be earned. Claims are ordered
#: by strength but not nested: an arm may exercise more layers than a weak claim
#: needs, it may never credit a layer it did not exercise.
_HYPOTHESIS_REQUIRES: dict[Hypothesis, InterfaceLayer] = {
    Hypothesis.H_WEAK: InterfaceLayer.SUPPLIED_TREE,
    Hypothesis.H_REL: InterfaceLayer.SELECTED_TREE,
    Hypothesis.H_NATIVE: InterfaceLayer.RECOVERED_DENS,
}


class AvailabilityRule(str, Enum):
    """Deterministic baselines 008.01 section 8 requires a learned policy to beat."""

    GROUPED_FIRST = "GroupedFirst"
    FORCE_SEQ = "ForceSeq"
    SUPPLIED_BRACKETING = "SuppliedBracketingExecutor"
    RANDOM_LEGAL = "RandomLegalTree"


#: Every arm-level comparison must carry these. ``AMBIENT_MUL`` is the
#: non-native control from 008.01 section 5.C, fenced separately.
REQUIRED_BASELINES: frozenset[str] = frozenset(
    {rule.value for rule in AvailabilityRule} | {"AmbientMul"}
)

#: 008.01 section 9 minimum diagnostics.
REQUIRED_DIAGNOSTICS: tuple[str, ...] = (
    "legal_program_count_by_signature",
    "endpoint_disagreement_rate",
    "deterministic_baseline_accuracy",
    "learned_train_exact_success",
    "learned_test_exact_success",
    "generalization_gap",
    "tree_selection_frequencies",
    "per_program_confusion",
    "success_by_n_legal_alternatives",
    "undefinedness_handling",
    "parameter_counts",
    "seeds",
)


@dataclass(frozen=True)
class DecoderAudit:
    """007.05 section 8 invalidity conditions, plus the 008.01 section 5.D fence.

    ``fitted_on`` names the split a decoder was fitted on. Anything other than
    ``"train"`` or ``"none"`` invalidates the comparison.
    """

    name: str
    fitted_on: str = "none"
    contains_heldout_lookup: bool = False
    computes_target_relation: bool = False
    free_parameters: int = 0

    def failures(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.contains_heldout_lookup:
            out.append(f"decoder {self.name!r} contains a held-out answer lookup")
        if self.computes_target_relation:
            out.append(f"decoder {self.name!r} computes the target relation itself")
        if self.fitted_on not in {"none", "train"}:
            out.append(
                f"decoder {self.name!r} was fitted on {self.fitted_on!r}; "
                "only 'none' or 'train' preserves the comparison"
            )
        return tuple(out)

    @property
    def valid(self) -> bool:
        return not self.failures()


@dataclass(frozen=True)
class ArmReport:
    """One ladder arm. ``claims`` is what the arm is allowed to support."""

    arm: str
    signature: str
    layers: frozenset[InterfaceLayer]
    claims: frozenset[Hypothesis]
    baselines: frozenset[str] = field(default_factory=frozenset)
    decoders: tuple[DecoderAudit, ...] = ()
    target_is_availability_rule: bool = False
    native_exact_scoring: bool = True
    surrogate_rescored_exactly: bool = True

    def failures(self) -> tuple[str, ...]:
        out: list[str] = []

        # A bracketing is either handed to the model or chosen by it, never both.
        if {InterfaceLayer.SUPPLIED_TREE, InterfaceLayer.SELECTED_TREE} <= self.layers:
            out.append(
                f"arm {self.arm!r} declares both supplied_tree and selected_tree; "
                "008.01 section 4 requires following a bracketing and selecting one "
                "to be reported as different arms"
            )

        for claim in sorted(self.claims, key=lambda h: h.value):
            required = _HYPOTHESIS_REQUIRES[claim]
            if required not in self.layers:
                out.append(
                    f"arm {self.arm!r} claims {claim.value} without exercising "
                    f"{required.value}"
                )

        # 008.01 section 4: the target must not encode a single availability rule.
        if self.target_is_availability_rule and InterfaceLayer.SELECTED_TREE in self.layers:
            out.append(
                f"arm {self.arm!r} selects trees against a target that is itself an "
                "availability rule; 008.01 section 4 firewall"
            )

        # 008.01 section 7: exact native scoring is primary.
        if not self.native_exact_scoring:
            out.append(f"arm {self.arm!r} does not score exact native endpoints")
        elif not self.surrogate_rescored_exactly:
            out.append(
                f"arm {self.arm!r} used a surrogate without rescoring "
                "under the exact evaluator"
            )

        # 008.01 section 8: beating the deterministic baselines is the whole point.
        if InterfaceLayer.SELECTED_TREE in self.layers:
            missing_baselines = REQUIRED_BASELINES - self.baselines
            if missing_baselines:
                out.append(
                    f"arm {self.arm!r} reports a learned policy without baselines: "
                    f"{', '.join(sorted(missing_baselines))}"
                )

        for decoder in self.decoders:
            out.extend(decoder.failures())
            if decoder.free_parameters and Hypothesis.H_NATIVE in self.claims:
                out.append(
                    f"arm {self.arm!r} claims H_native with a free decoder "
                    f"({decoder.name!r}, {decoder.free_parameters} parameters); "
                    "008.01 section 5.D forbids manufacturing native result identity"
                )

        return tuple(out)

    @property
    def valid(self) -> bool:
        return not self.failures()


def validate_arm(arm: ArmReport) -> None:
    """Raise ``ValueError`` listing every fence the arm violates."""
    failures = arm.failures()
    if failures:
        raise ValueError("; ".join(failures))


def validate_suite(arms: list[ArmReport], diagnostics: dict[str, object]) -> None:
    """Validate every arm and require the 008.01 section 9 diagnostics."""
    failures: list[str] = []
    for arm in arms:
        failures.extend(arm.failures())
    missing = [key for key in REQUIRED_DIAGNOSTICS if key not in diagnostics]
    if missing:
        failures.append(f"missing required diagnostics: {', '.join(missing)}")
    if failures:
        raise ValueError("; ".join(failures))

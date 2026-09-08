"""Outcome-aware Futurator ask runtime (Issue 005 Milestone F).

Presentation, admissibility, and enacted law are explicit inputs. Success is a
typed ConstitutedOutcome; failure is a typed NonAdmission. Projective zero
products constitute AnnihilationBoundary — they are not NonAdmission.

Naming: sealed Event frames are EventDenotation. OperationalFrame remains the
Theory 068.02 runtime unit in topographo.ssd.frames and is intentionally not
reused here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from topographo.ssd import exact, machine, projective
from topographo.ssd.seal import (
    EventDenotation,
    MissingSealedDenotation,
    PublicEventView,
    resolve,
)


@dataclass(frozen=True)
class Presentation:
    """Staged participants supplied to ask; not an automatic history tape."""

    participants: tuple[object, ...]


@dataclass(frozen=True)
class Admissibility:
    """Caller-certified domain facts for the staged presentation."""

    certified: bool = True
    reason: str = ""


@dataclass(frozen=True)
class RetainedRay:
    """Policy post-state retained after a successful non-annihilating step."""

    state: exact.Value


@dataclass(frozen=True)
class AnnihilationBoundary:
    """Constituted projective encounter with undefined successor (zx = 0)."""

    step: int
    trace: tuple[machine.Transition, ...]


@dataclass(frozen=True)
class FipsThird:
    """Forced unique third Event from the certified finite FIPS table."""

    third: EventDenotation
    retained: RetainedRay | AnnihilationBoundary | None = None


@dataclass(frozen=True)
class OperationDomainInadmissible:
    """Pair or presentation outside the certified operation domain."""

    detail: str = ""


@dataclass(frozen=True)
class ConstitutionNotSupplied:
    """Missing presentation, law, or certified structure."""

    detail: str = ""


NonAdmissionReason = (
    OperationDomainInadmissible
    | MissingSealedDenotation
    | ConstitutionNotSupplied
)


@dataclass(frozen=True)
class NonAdmission:
    reason: NonAdmissionReason


Consequence = RetainedRay | AnnihilationBoundary | FipsThird


@dataclass(frozen=True)
class ConstitutedOutcome:
    presentation: Presentation
    admissibility: Admissibility
    enacted_law: object
    execution: machine.Completed | machine.Annihilated | None
    consequence: Consequence


class EnactedLaw(Protocol):
    def apply(
        self,
        presentation: Presentation,
        admissibility: Admissibility,
    ) -> ConstitutedOutcome | NonAdmission: ...


def _require_sealed_event(participant: object) -> EventDenotation | NonAdmission:
    if isinstance(participant, PublicEventView):
        return NonAdmission(MissingSealedDenotation("classify-only view cannot act"))
    if isinstance(participant, EventDenotation):
        return participant
    if isinstance(participant, tuple) and len(participant) == exact.DIMENSION:
        # Raw exact values are accepted only when already sealed by the caller
        # into EventDenotation; bare vectors are treated as missing seal.
        return NonAdmission(MissingSealedDenotation("bare value is not a sealed EventDenotation"))
    return NonAdmission(MissingSealedDenotation(f"unsupported event participant: {type(participant)!r}"))


def _as_state(participant: object) -> exact.Value:
    if isinstance(participant, RetainedRay):
        return participant.state
    return exact.checked(participant, where="retained state")  # type: ignore[arg-type]


@dataclass(frozen=True)
class NativeLeftAction:
    """Raw left action; continues through zero as RetainedRay."""

    def apply(
        self,
        presentation: Presentation,
        admissibility: Admissibility,
    ) -> ConstitutedOutcome | NonAdmission:
        if not admissibility.certified:
            return NonAdmission(OperationDomainInadmissible(admissibility.reason or "not certified"))
        if len(presentation.participants) != 2:
            return NonAdmission(ConstitutionNotSupplied("NativeLeftAction needs (event, state)"))
        event_part, state_part = presentation.participants
        sealed = _require_sealed_event(event_part)
        if isinstance(sealed, NonAdmission):
            return sealed
        event = resolve(sealed)
        state = _as_state(state_part)
        execution = machine.run(state, [event])
        return ConstitutedOutcome(
            presentation=presentation,
            admissibility=admissibility,
            enacted_law=self,
            execution=execution,
            consequence=RetainedRay(execution.state),
        )


@dataclass(frozen=True)
class ProjectiveNativeLeftAction:
    """Projective left action; zx=0 constitutes AnnihilationBoundary."""

    def apply(
        self,
        presentation: Presentation,
        admissibility: Admissibility,
    ) -> ConstitutedOutcome | NonAdmission:
        if not admissibility.certified:
            return NonAdmission(OperationDomainInadmissible(admissibility.reason or "not certified"))
        if len(presentation.participants) != 2:
            return NonAdmission(ConstitutionNotSupplied("ProjectiveNativeLeftAction needs (event, state)"))
        event_part, state_part = presentation.participants
        sealed = _require_sealed_event(event_part)
        if isinstance(sealed, NonAdmission):
            return sealed
        event = resolve(sealed)
        state = _as_state(state_part)
        execution = projective.run_projective(state, [event])
        if isinstance(execution, machine.Annihilated):
            consequence: Consequence = AnnihilationBoundary(execution.step, execution.trace)
        else:
            consequence = RetainedRay(execution.state)
        return ConstitutedOutcome(
            presentation=presentation,
            admissibility=admissibility,
            enacted_law=self,
            execution=execution,
            consequence=consequence,
        )


@dataclass(frozen=True)
class FipsClosure:
    """Exact finite FIPS forced third; optional retained-state action with that third."""

    def apply(
        self,
        presentation: Presentation,
        admissibility: Admissibility,
    ) -> ConstitutedOutcome | NonAdmission:
        from topographo.ssd import fips_basic

        if not admissibility.certified:
            return NonAdmission(OperationDomainInadmissible(admissibility.reason or "not certified"))
        n = len(presentation.participants)
        if n not in (2, 3):
            return NonAdmission(ConstitutionNotSupplied("FipsClosure needs (z, w) or (z, w, state)"))
        z_part, w_part = presentation.participants[0], presentation.participants[1]
        z_sealed = _require_sealed_event(z_part)
        if isinstance(z_sealed, NonAdmission):
            return z_sealed
        w_sealed = _require_sealed_event(w_part)
        if isinstance(w_sealed, NonAdmission):
            return w_sealed
        z_ray, w_ray = resolve(z_sealed), resolve(w_sealed)
        if not fips_basic.admissible(z_ray, w_ray):
            return NonAdmission(OperationDomainInadmissible("pair not in basic FIPS domain"))
        third_ray = fips_basic.third(z_ray, w_ray)
        third = EventDenotation(third_ray)
        retained: RetainedRay | AnnihilationBoundary | None = None
        execution: machine.Completed | machine.Annihilated | None = None
        if n == 3:
            state = _as_state(presentation.participants[2])
            execution = projective.run_projective(state, [third_ray])
            if isinstance(execution, machine.Annihilated):
                retained = AnnihilationBoundary(execution.step, execution.trace)
            else:
                retained = RetainedRay(execution.state)
        return ConstitutedOutcome(
            presentation=presentation,
            admissibility=admissibility,
            enacted_law=self,
            execution=execution,
            consequence=FipsThird(third, retained),
        )


def ask(
    presentation: Presentation,
    admissibility: Admissibility,
    enacted_law: EnactedLaw,
) -> ConstitutedOutcome | NonAdmission:
    """Constitute an Outcome under the supplied law, or return typed NonAdmission."""

    return enacted_law.apply(presentation, admissibility)

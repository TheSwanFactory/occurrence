"""Typed failures for resolution operations."""

from __future__ import annotations

__all__ = (
    "AnnihilationError",
    "BackendResolutionError",
    "InvalidResolutionError",
    "ResolutionError",
)


class ResolutionError(Exception):
    """Base class for failures that do not produce a probability endpoint."""


class BackendResolutionError(ResolutionError):
    """A backend or effect failed before producing a probability endpoint."""


class AnnihilationError(BackendResolutionError):
    """A backend determined that an operation annihilated the supplied state."""


class InvalidResolutionError(ResolutionError):
    """A purported backend result violated a probability endpoint invariant."""

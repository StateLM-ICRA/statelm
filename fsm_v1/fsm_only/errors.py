"""Exceptions raised by :mod:`fsm_only`."""


class FSMOnlyError(Exception):
    """Base class for package errors."""


class TrainingDataError(FSMOnlyError):
    """The labeled training data is incomplete or contradictory."""


class ModelFormatError(FSMOnlyError):
    """A compiled artifact is malformed or unsupported."""


class InventoryFormatError(FSMOnlyError):
    """The inventory data is malformed."""


class ControllerError(FSMOnlyError):
    """The learned controller has no valid deterministic transition."""

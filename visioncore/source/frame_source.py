"""FrameSource factory and registry for VisionCore.

This module provides a **registry-based factory** for
:class:`~visioncore.source.base.FrameSource` subclasses. Sources register
themselves under a string ``kind``; callers create sources by kind name
without importing the concrete class. This decouples source *selection*
(config-driven, often from a settings file) from source *construction*
(which may pull in heavy I/O dependencies).

Scope (Milestone C2)
--------------------
C2 registers a single built-in source kind:

    "dummy"  -> :class:`~visioncore.source.dummy_source.DummyFrameSource`

Future milestones register real sources as sibling modules:

    "camera" -> CameraFrameSource      (C3, wraps ``camera/`` backends)
    "file"   -> FileFrameSource        (C3, replays a video file)
    "rtsp"   -> RtspFrameSource        (C4, network stream)

Each real source is the **only** module in the source package that
imports its transport library (DirectShow / OpenCV / av / socket). The
factory keeps that coupling confined: callers that only use ``"dummy"``
never pay the import cost of the real transports.

Registry semantics
------------------
* Registration is **idempotent** for the same (name, class) pair, and
  **refuses** to overwrite a name with a *different* class (raises
  :class:`ValueError`) -- this prevents accidental shadowing.
* The registry is module-level and not thread-safe. Registration happens
  at import time (each source module calls :func:`register_frame_source`
  at the bottom); runtime mutation is discouraged but not prevented.
* :func:`create_frame_source` raises :class:`ValueError` for an unknown
  kind, listing the available kinds in the error message.

Example
-------
    >>> from visioncore.source.frame_source import create_frame_source
    >>> src = create_frame_source("dummy", source_id="cam0")
    >>> src.source_id
    'cam0'
    >>> with src:
    ...     f = src.read()
    ... # close() called automatically
"""

from __future__ import annotations

import logging
from typing import Any

from visioncore.source.base import FrameSource


__all__ = [
    "register_frame_source",
    "create_frame_source",
    "available_frame_sources",
    "get_frame_source_class",
]


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

_REGISTRY: dict[str, type[FrameSource]] = {}


def register_frame_source(
    name: str, cls: type[FrameSource], *, overwrite: bool = False,
) -> None:
    """Register a FrameSource subclass under a string kind name.

    Parameters:
        name: The kind string used by :func:`create_frame_source`
            (e.g. ``"dummy"``, ``"camera"``). Must be a non-empty string.
        cls: The :class:`FrameSource` subclass to register. Must be a
            concrete subclass (not :class:`FrameSource` itself).
        overwrite: If ``True``, silently replace any existing registration
            for ``name``. If ``False`` (default), refuse to overwrite a
            name that is already registered to a *different* class (raises
            :class:`ValueError`). Re-registering the same (name, class)
            pair is always a no-op.

    Raises:
        ValueError: If ``name`` is empty, ``cls`` is not a concrete
            :class:`FrameSource` subclass, or ``name`` is already
            registered to a different class and ``overwrite`` is ``False``.

    Example:
        >>> register_frame_source("dummy", DummyFrameSource)  # idempotent
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"name must be a non-empty string, got {name!r}"
        )
    if not (isinstance(cls, type) and issubclass(cls, FrameSource)):
        raise ValueError(
            f"cls must be a FrameSource subclass, got {cls!r}"
        )
    if cls is FrameSource:
        raise ValueError(
            "cannot register the abstract FrameSource base class itself; "
            "register a concrete subclass"
        )
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls and not overwrite:
        raise ValueError(
            f"frame source kind {name!r} is already registered to "
            f"{existing.__name__}; pass overwrite=True to replace, or "
            f"register under a different name"
        )
    _REGISTRY[name] = cls
    logger.debug("register_frame_source: name=%s cls=%s (overwrite=%s)",
                 name, cls.__name__, overwrite)


def get_frame_source_class(name: str) -> type[FrameSource] | None:
    """Return the registered class for ``name``, or ``None`` if unknown.

    Parameters:
        name: The kind string to look up.

    Returns:
        The registered :class:`FrameSource` subclass, or ``None`` if no
        source is registered under ``name``.
    """
    return _REGISTRY.get(name)


def create_frame_source(name: str, **kwargs: Any) -> FrameSource:
    """Create a FrameSource instance by registered kind name.

    Looks up ``name`` in the registry and instantiates the registered
    class with ``kwargs``. This is the config-driven entry point: a
    settings file specifies ``{"type": "dummy", "source_id": "cam0"}``
    and the caller does ``create_frame_source("dummy", source_id="cam0")``.

    Parameters:
        name: The registered kind string (e.g. ``"dummy"``).
        **kwargs: Forwarded to the registered class's ``__init__``.

    Returns:
        A new :class:`FrameSource` instance.

    Raises:
        ValueError: If ``name`` is not registered. The error message lists
            the available kinds for easy diagnosis.
        TypeError: If the registered class rejects ``kwargs``.

    Example:
        >>> src = create_frame_source("dummy", source_id="cam0")
        >>> src.source_id
        'cam0'
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown frame source kind: {name!r}; "
            f"available: {available_frame_sources()}"
        )
    logger.debug("create_frame_source: name=%s cls=%s kwargs=%s",
                 name, cls.__name__, list(kwargs))
    return cls(**kwargs)


def available_frame_sources() -> list[str]:
    """Return the sorted list of registered source kind names.

    Returns:
        A new list of registered kind strings, sorted alphabetically.
    """
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------
# Built-in source registration
# ----------------------------------------------------------------------
# Importing the concrete source module triggers its availability; we then
# register it under a stable kind name. Real sources (C3+) do the same in
# their own modules. This is the ONLY place dummy_source is imported by the
# factory, keeping the dependency direction one-way (factory -> source, not
# the reverse -- dummy_source does not import this module, so no cycle).
# ----------------------------------------------------------------------
from visioncore.source.dummy_source import DummyFrameSource  # noqa: E402

register_frame_source("dummy", DummyFrameSource)

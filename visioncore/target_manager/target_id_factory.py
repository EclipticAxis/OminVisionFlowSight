"""Unified Target ID factory for VisionCore.

Centralises the generation of ``target_id`` strings so that every code
path (stateless converters, stateful :class:`TargetManager`, future
EventBus emitters) produces IDs with the **same** format.

Format
------
::

    S{slot_id}-T{local_id:04d}

Examples::

    S0-T0001   # slot 0, local id 1
    S0-T0007   # slot 0, local id 7
    S1-T0002   # slot 1, local id 2
    S3-T0099   # slot 3, local id 99

The ``local_id`` is zero-padded to **at least** 4 digits. IDs larger than
9999 simply grow (e.g. ``S0-T10000``) -- the padding is a minimum, not a
fixed width.

Design rationale
----------------
Before this module existed, ID generation was duplicated in two places:

1. :func:`visioncore.target_manager.converters.track_to_target` --
   ``f"S{slot_id}-T{track.track_id}"`` (unpadded).
2. :meth:`TargetManager._generate_id` --
   ``f"S{slot_id}-T{self._next_id:04d}"`` (zero-padded).

This caused a visible inconsistency: the same logical target could be
labelled ``S0-T7`` (converter) or ``S0-T0007`` (manager). The factory
collapses both into a single formatting function with a guaranteed
uniform output.

Thread safety:
    The factory is a pure function with no shared state. Safe to call
    from any thread without synchronisation.
"""

from __future__ import annotations

__all__ = [
    "create_target_id",
    "TARGET_ID_SEPARATOR",
]


#: Separator between the slot segment and the local-id segment.
TARGET_ID_SEPARATOR: str = "-"


def create_target_id(slot_id: int, local_id: int) -> str:
    """Generate a slot-scoped, uniformly-formatted target identifier.

    The returned string has the form ``S{slot_id}-T{local_id:04d}``
    where ``local_id`` is zero-padded to a minimum width of 4.

    Parameters:
        slot_id: Camera slot identifier (typically 0–3). Embedded in the
            ``S`` prefix so that targets from different cameras never
            collide even when they share the same ``local_id``.
        local_id: The locally-unique identifier within the slot. For
            stateless converters this is the source ``track_id``; for the
            stateful :class:`TargetManager` it is the monotonic counter.

    Returns:
        A target ID string such as ``"S0-T0001"``.

    Example:
        >>> create_target_id(0, 1)
        'S0-T0001'
        >>> create_target_id(1, 1)
        'S1-T0001'
        >>> create_target_id(3, 99)
        'S3-T0099'
        >>> create_target_id(0, 10000)
        'S0-T10000'
    """
    return f"S{slot_id}{TARGET_ID_SEPARATOR}T{local_id:04d}"

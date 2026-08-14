"""Runtime state -- cache, statistics, and timing state container.

This module extracts the internal state management from
``ai/inference.py``'s InferWorker into a clean, testable container. It
contains:

* :class:`RuntimeState` -- state container for temporary caches,
  statistics counters, and timing state.

The state container does **not** contain detection logic (filtering,
NMS, box conversion) or model management code. It is a pure data
structure with helper methods for incrementing counters, managing
caches, and tracking timing.

Scope (Milestone D9.4)
----------------------
D9.4 extracts state from ``ai/inference.py``:

* ``_last_detections`` → ``RuntimeState.slot_caches``
* ``_last_submit_ts_by_slot`` → ``RuntimeState.slot_timestamps``
* ``_head_attr_cache`` → ``RuntimeState.head_attr_cache``
* ``_debug_infer/submit/emit_reports`` → ``RuntimeState.stats``
* ``_infer_cycle_count`` → ``RuntimeState.stats``
* ``_redetect_*_count`` → ``RuntimeState.stats``
* ``_sr_*_count`` → ``RuntimeState.stats``
* ``_warmup_done`` → ``RuntimeState.warmup_done``
* ``_startup_guard_until`` → ``RuntimeState.startup_guard_until``

D9.4 deliberately does **not** contain:

* any detection logic;
* any model management;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Example
-------
    >>> from visioncore.runtime.state import RuntimeState
    >>> state = RuntimeState()
    >>> state.stats.increment("infer_cycles")
    >>> state.stats.get("infer_cycles")
    1
    >>> state.warmup_done
    False
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RuntimeState"]


logger = logging.getLogger(__name__)


# ======================================================================
# Statistics counter
# ======================================================================

@dataclass
class Stats:
    """Named statistics counters with zero-default semantics.

    Counters are accessed by name via :meth:`increment` / :meth:`get` /
    :meth:`reset`. Unknown names are initialised to 0 on first access.

    Example:
        >>> s = Stats()
        >>> s.increment("infer_cycles")
        >>> s.increment("infer_cycles", 5)
        >>> s.get("infer_cycles")
        6
        >>> s.get("missing")
        0
    """

    _counters: dict[str, int | float] = field(default_factory=dict)

    def increment(self, name: str, amount: int | float = 1) -> None:
        """Increment a named counter.

        Parameters:
            name: Counter name.
            amount: Increment value (default 1).
        """
        self._counters[name] = self._counters.get(name, 0) + amount

    def get(self, name: str) -> int | float:
        """Return the current value of a named counter (0 if unset)."""
        return self._counters.get(name, 0)

    def set(self, name: str, value: int | float) -> None:
        """Set a named counter to a specific value."""
        self._counters[name] = value

    def reset(self, name: str | None = None) -> None:
        """Reset a named counter (or all counters if name is None)."""
        if name is None:
            self._counters.clear()
        else:
            self._counters.pop(name, None)

    def snapshot(self) -> dict[str, int | float]:
        """Return a copy of all counters."""
        return dict(self._counters)

    def __repr__(self) -> str:
        return f"Stats({self._counters!r})"


# ======================================================================
# Slot cache (per-slot last detections + timestamps)
# ======================================================================

@dataclass
class SlotCache:
    """Per-slot temporary cache.

    Stores the last detection results and submission timestamp for each
    camera slot.

    Example:
        >>> cache = SlotCache()
        >>> cache.set_detections(0, [{"label": "person"}])
        >>> cache.get_detections(0)
        [{'label': 'person'}]
        >>> cache.set_timestamp(0, 1.5)
        >>> cache.get_timestamp(0)
        1.5
    """

    _detections: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    _timestamps: dict[int, float] = field(default_factory=dict)

    def set_detections(self, slot_id: int, detections: list[dict[str, Any]]) -> None:
        """Store detection results for a slot."""
        self._detections[slot_id] = detections

    def get_detections(self, slot_id: int) -> list[dict[str, Any]]:
        """Return the last detection results for a slot (empty list if none)."""
        return self._detections.get(slot_id, [])

    def set_timestamp(self, slot_id: int, timestamp: float) -> None:
        """Store the last submission timestamp for a slot."""
        self._timestamps[slot_id] = timestamp

    def get_timestamp(self, slot_id: int) -> float | None:
        """Return the last submission timestamp for a slot, or None."""
        return self._timestamps.get(slot_id)

    def clear(self) -> None:
        """Clear all slot caches."""
        self._detections.clear()
        self._timestamps.clear()

    def clear_slot(self, slot_id: int) -> None:
        """Clear cache for a specific slot."""
        self._detections.pop(slot_id, None)
        self._timestamps.pop(slot_id, None)

    @property
    def slot_ids(self) -> list[int]:
        """Return all cached slot IDs."""
        return list(self._detections.keys())

    def __repr__(self) -> str:
        return f"SlotCache(slots={self.slot_ids})"


# ======================================================================
# Head attribute cache
# ======================================================================

class HeadAttrCache:
    """Per-track head attribute cache.

    Stores head classification results (hat, mask, etc.) per track ID
    with frame-based expiry.

    Example:
        >>> cache = HeadAttrCache(stride=2)
        >>> cache.set(1, 10, {"hat": True})
        >>> cache.get(1, 11)  # within stride
        {'hat': True}
        >>> cache.get(1, 13)  # expired
    """

    __slots__ = ("_stride", "_entries")

    def __init__(self, stride: int = 2) -> None:
        self._stride: int = stride
        self._entries: dict[int, tuple[int, dict[str, Any]]] = {}

    def get(self, track_id: int, frame_id: int) -> dict[str, Any] | None:
        """Return cached attributes if within stride, or None.

        Parameters:
            track_id: The track identifier.
            frame_id: The current frame number.

        Returns:
            The cached attributes dict, or None if expired/missing.
        """
        entry = self._entries.get(track_id)
        if entry is None:
            return None
        cached_frame, attrs = entry
        if (frame_id - cached_frame) < self._stride:
            return attrs
        return None

    def set(self, track_id: int, frame_id: int, attrs: dict[str, Any]) -> None:
        """Cache attributes for a track.

        Parameters:
            track_id: The track identifier.
            frame_id: The frame number when the attributes were computed.
            attrs: The attributes dict to cache.
        """
        self._entries[track_id] = (frame_id, attrs)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)

    def __repr__(self) -> str:
        return f"HeadAttrCache(size={self.size}, stride={self._stride})"


# ======================================================================
# RuntimeState -- the main state container
# ======================================================================

class RuntimeState:
    """State container for runtime caches, statistics, and timing.

    :class:`RuntimeState` holds all mutable runtime state that
    ``InferWorker`` manages but that is **not** business logic. It is a
    pure data structure with helper methods.

    Attributes:
        stats: Named statistics counters.
        slot_cache: Per-slot detection cache and timestamps.
        head_attr_cache: Per-track head attribute cache.
        warmup_done: Whether model warmup has been performed.
        startup_guard_until: Monotonic timestamp when startup guard
            expires (0.0 = no guard active).
        startup_guard_seconds: Default startup guard duration.

    Example:
        >>> state = RuntimeState()
        >>> state.stats.increment("infer_cycles")
        >>> state.stats.get("infer_cycles")
        1
        >>> state.warmup_done
        False
        >>> state.is_startup_guard_active()
        False
    """

    __slots__ = (
        "stats",
        "slot_cache",
        "head_attr_cache",
        "warmup_done",
        "startup_guard_until",
        "startup_guard_seconds",
    )

    def __init__(
        self,
        startup_guard_seconds: float = 3.0,
        head_attr_stride: int = 2,
    ) -> None:
        """Create a RuntimeState.

        Parameters:
            startup_guard_seconds: Default startup guard duration.
            head_attr_stride: Frame stride for head attribute cache
                expiry.
        """
        self.stats: Stats = Stats()
        self.slot_cache: SlotCache = SlotCache()
        self.head_attr_cache: HeadAttrCache = HeadAttrCache(stride=head_attr_stride)
        self.warmup_done: bool = False
        self.startup_guard_until: float = 0.0
        self.startup_guard_seconds: float = startup_guard_seconds
        logger.debug("RuntimeState created: guard=%.1fs stride=%d",
                     startup_guard_seconds, head_attr_stride)

    # ------------------------------------------------------------------
    # Startup guard
    # ------------------------------------------------------------------

    def enable_startup_guard(self, seconds: float | None = None) -> None:
        """Enable the startup guard for a duration.

        Parameters:
            seconds: Guard duration. ``None`` uses
                ``startup_guard_seconds``.
        """
        use_seconds = seconds if seconds is not None else self.startup_guard_seconds
        self.startup_guard_until = time.monotonic() + use_seconds
        logger.info("RuntimeState: startup guard enabled for %.2fs", use_seconds)

    def is_startup_guard_active(self) -> bool:
        """Return ``True`` iff the startup guard is currently active."""
        return time.monotonic() < self.startup_guard_until

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def mark_warmup_done(self) -> None:
        """Mark warmup as completed."""
        self.warmup_done = True

    def mark_warmup_needed(self) -> None:
        """Mark warmup as needed (e.g. after model switch)."""
        self.warmup_done = False

    # ------------------------------------------------------------------
    # Convenience: redetect stats
    # ------------------------------------------------------------------

    def record_redetect_attempt(self) -> None:
        """Increment the redetect attempt counter."""
        self.stats.increment("redetect_attempts")

    def record_redetect_success(self) -> None:
        """Increment the redetect success counter."""
        self.stats.increment("redetect_successes")

    def record_redetect_failure(self) -> None:
        """Increment the redetect failure counter."""
        self.stats.increment("redetect_failures")

    @property
    def redetect_stats(self) -> dict[str, int | float]:
        """Return redetect-related statistics."""
        return {
            "attempts": self.stats.get("redetect_attempts"),
            "successes": self.stats.get("redetect_successes"),
            "failures": self.stats.get("redetect_failures"),
        }

    # ------------------------------------------------------------------
    # Convenience: inference stats
    # ------------------------------------------------------------------

    def record_infer_cycle(self) -> None:
        """Increment the inference cycle counter."""
        self.stats.increment("infer_cycles")

    @property
    def infer_cycle_count(self) -> int | float:
        """The number of inference cycles completed."""
        return self.stats.get("infer_cycles")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.stats.reset()
        self.slot_cache.clear()
        self.head_attr_cache.clear()
        self.warmup_done = False
        self.startup_guard_until = 0.0
        logger.debug("RuntimeState: reset")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RuntimeState(warmup_done={self.warmup_done}, "
            f"guard_active={self.is_startup_guard_active()}, "
            f"stats={self.stats.snapshot()}, "
            f"slots={self.slot_cache.slot_ids})"
        )
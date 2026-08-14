"""TargetStore -- thread-safe in-memory CRUD container for Target entities.

The store is a low-level data structure with no lifecycle logic. It owns a
``dict[str, Target]`` keyed by ``target_id`` and protects all access with a
re-entrant lock so it can be safely shared between the inference thread,
the GUI thread, and any future worker threads.

Responsibilities:
    * Add / remove / look up Target objects by id.
    * Return snapshots of the current target set.
    * Report basic statistics (count, containment).

Non-responsibilities (delegated to :class:`TargetManager`):
    * Lifecycle state transitions.
    * ID generation.
    * Stale-target detection.
    * Event emission.

Thread safety:
    Every public method acquires ``self._lock`` (an :class:`~threading.RLock`).
    The lock is re-entrant so a thread that already holds the lock can call
    other locked methods without deadlocking. Returned Target references are
    **not** copied -- callers that need isolation should copy explicitly.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visioncore.core.target import Target


class TargetStore:
    """Thread-safe store for :class:`~visioncore.core.target.Target` entities.

    The store uses a plain ``dict`` internally and guards every operation
    with an :class:`~threading.RLock`. It does not validate Target state
    transitions -- that is the responsibility of :class:`TargetManager`.

    Attributes:
        _targets: Internal mapping ``target_id -> Target``.
        _lock: Re-entrant lock protecting ``_targets``.
        _logger: Module-level logger for operation traces.

    Example:
        >>> store = TargetStore()
        >>> from visioncore.core.target import Target, TargetState
        >>> t = Target(target_id="T-001", track=None)
        >>> store.add_target(t)
        >>> store.get_target("T-001") is t
        True
        >>> len(store)
        1
        >>> store.clear()
        1
        >>> len(store)
        0
    """

    __slots__ = ("_targets", "_lock", "_logger")

    def __init__(self) -> None:
        """Initialise an empty store."""
        self._targets: dict[str, Target] = {}
        self._lock: threading.RLock = threading.RLock()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.debug("TargetStore initialised (empty)")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_target(self, target: Target) -> None:
        """Register a Target in the store.

        Parameters:
            target: The Target to add. Its ``target_id`` must not already
                exist in the store.

        Raises:
            ValueError: If a Target with the same ``target_id`` is already
                present. The existing entry is left untouched.

        Example:
            >>> store = TargetStore()
            >>> from visioncore.core.target import Target
            >>> store.add_target(Target(target_id="T-1", track=None))
            >>> store.add_target(Target(target_id="T-1", track=None))
            Traceback (most recent call last):
                ...
            ValueError: Target already exists: T-1
        """
        tid: str = target.target_id
        with self._lock:
            if tid in self._targets:
                raise ValueError(f"Target already exists: {tid}")
            self._targets[tid] = target
            self._logger.debug(
                "add_target: id=%s state=%s priority=%d",
                tid, target.state.name, target.priority,
            )

    def remove_target(self, target_id: str) -> Target | None:
        """Remove and return a Target by id.

        Parameters:
            target_id: The id of the Target to remove.

        Returns:
            The removed Target, or ``None`` if no Target with the given id
            was found.

        Example:
            >>> store = TargetStore()
            >>> store.remove_target("nonexistent") is None
            True
        """
        with self._lock:
            removed: Target | None = self._targets.pop(target_id, None)
            if removed is not None:
                self._logger.debug(
                    "remove_target: id=%s (was state=%s)",
                    target_id, removed.state.name,
                )
            else:
                self._logger.debug(
                    "remove_target: id=%s not found (no-op)", target_id,
                )
            return removed

    def get_target(self, target_id: str) -> Target | None:
        """Look up a Target by id without removing it.

        Parameters:
            target_id: The id to look up.

        Returns:
            The Target with the given id, or ``None`` if not found.

        Note:
            The returned object is the **live** reference stored internally.
            Mutations to it are visible to all other callers. Use
            :func:`copy.deepcopy` if you need an isolated snapshot.
        """
        with self._lock:
            return self._targets.get(target_id)

    def get_all_targets(self) -> list[Target]:
        """Return a list of all Targets currently in the store.

        Returns:
            A new list containing all Target references, in insertion order
            (Python 3.7+ dict ordering). The Target objects themselves are
            **not** copied -- see :meth:`get_target` for the same caveat.

        Example:
            >>> store = TargetStore()
            >>> len(store.get_all_targets())
            0
        """
        with self._lock:
            return list(self._targets.values())

    def clear(self) -> int:
        """Remove all Targets from the store.

        Returns:
            The number of Targets that were removed.

        Example:
            >>> store = TargetStore()
            >>> store.clear()
            0
        """
        with self._lock:
            count: int = len(self._targets)
            self._targets.clear()
            self._logger.info("clear: removed %d target(s)", count)
            return count

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of Targets in the store."""
        with self._lock:
            return len(self._targets)

    def __contains__(self, target_id: object) -> bool:
        """Return ``True`` if a Target with the given id exists."""
        with self._lock:
            if isinstance(target_id, str):
                return target_id in self._targets
            return False

    def __repr__(self) -> str:
        """Return a concise representation showing the target count."""
        with self._lock:
            count: int = len(self._targets)
            states: dict[str, int] = {}
            for t in self._targets.values():
                name: str = t.state.name
                states[name] = states.get(name, 0) + 1
        return f"TargetStore(count={count}, states={states})"

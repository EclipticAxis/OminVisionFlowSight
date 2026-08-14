"""TargetLifecycleManager -- pure state machine for Target lifecycle.

This module implements **only** the state machine that governs Target
lifecycle transitions. It contains no store, no ID generation, no signal
emission, and no connection to any business logic. Callers pass in a
:class:`~visioncore.core.target.Target` and the manager validates and
executes the requested state change.

Transition table (exactly 6 transitions)
----------------------------------------

    ┌─────────┐  mark_lost    ┌─────────┐  mark_recovered  ┌────────────┐
    │ ACTIVE  │ ────────────▶ │  LOST   │ ───────────────▶ │ RECOVERED  │
    └─────────┘                └─────────┘                  └────────────┘
       │  ▲                       │  ▲                          │
       │  │ release               │  │ remove                   │ activate
       ▼  │                       ▼  │                          ▼
    ┌─────────┐                ┌─────────┐                   ┌─────────┐
    │ LOCKED  │                │ REMOVED │                   │ ACTIVE  │
    └─────────┘                └─────────┘                   └─────────┘
                               (terminal)

Rules:
    * REMOVED is **terminal** -- no outgoing transitions.
    * REMOVED can only be reached from LOST (not from ACTIVE / LOCKED).
    * RECOVERED is **transient** -- its only exit is back to ACTIVE.
    * LOCKED ↔ ACTIVE is a bidirectional toggle (lock / release).

Thread safety:
    The manager is stateless (no shared mutable state). Each
    :meth:`transition` call mutates only the Target passed in. Multiple
    threads can safely share a single manager instance, but Target objects
    themselves are not internally synchronised -- callers must ensure no
    two threads transition the same Target concurrently.
"""

from __future__ import annotations

import logging
from typing import final

from visioncore.core.target import Target, TargetState

_logger: logging.Logger = logging.getLogger(__name__)


class TargetLifecycleManager:
    """Pure lifecycle state machine for :class:`Target` entities.

    This class implements **only** state transitions. It does not store
    targets, generate IDs, emit signals, or connect to any external system.

    Usage:

        >>> from visioncore.core.detection import BBox, Detection
        >>> from visioncore.core.track import Track
        >>> from visioncore.core.target import Target
        >>> from visioncore.target_manager.lifecycle import TargetLifecycleManager
        >>> lm = TargetLifecycleManager()
        >>> t = Target(target_id="T-1",
        ...            track=Track(1, Detection(BBox(.5,.5,.2,.2), .9, 0, "p")))
        >>> lm.mark_lost(t)
        True
        >>> t.state.name
        'LOST'
        >>> lm.mark_recovered(t)
        True
        >>> lm.activate(t)
        True
        >>> t.state.name
        'ACTIVE'

    The :data:`TRANSITIONS` table defines exactly 6 allowed transitions.
    Any attempt to transition outside this table is rejected with a
    ``False`` return value and a WARNING log entry.
    """

    __slots__ = ("_logger",)

    # ------------------------------------------------------------------
    # Static transition table
    # ------------------------------------------------------------------

    TRANSITIONS: dict[TargetState, frozenset[TargetState]] = {
        TargetState.ACTIVE: frozenset({
            TargetState.LOST,       # mark_lost
            TargetState.LOCKED,     # lock
        }),
        TargetState.LOST: frozenset({
            TargetState.RECOVERED,  # mark_recovered
            TargetState.REMOVED,    # remove
        }),
        TargetState.RECOVERED: frozenset({
            TargetState.ACTIVE,     # activate
        }),
        TargetState.LOCKED: frozenset({
            TargetState.ACTIVE,     # release
        }),
        TargetState.REMOVED: frozenset(),  # terminal
    }
    """Immutable transition table.

    Maps each source state to the set of states reachable from it.
    ``REMOVED`` maps to an empty set (terminal -- no exits).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise the lifecycle manager with a module-level logger."""
        self._logger: logging.Logger = _logger

    # ------------------------------------------------------------------
    # Query methods (classmethods -- no instance state needed)
    # ------------------------------------------------------------------

    @classmethod
    def can_transition(
        cls,
        current: TargetState,
        intended: TargetState,
    ) -> bool:
        """Check whether ``current -> intended`` is a valid transition.

        Parameters:
            current: The Target's current state.
            intended: The desired new state.

        Returns:
            ``True`` if the transition is allowed by :data:`TRANSITIONS`,
            ``False`` otherwise.
        """
        return intended in cls.TRANSITIONS.get(current, frozenset())

    @classmethod
    def get_valid_transitions(cls, state: TargetState) -> frozenset[TargetState]:
        """Return all states reachable from ``state`` in one step.

        Parameters:
            state: The source state.

        Returns:
            A frozen set of reachable :class:`TargetState` values.
            Empty for terminal states (REMOVED).
        """
        return cls.TRANSITIONS.get(state, frozenset())

    @classmethod
    def is_terminal(cls, state: TargetState) -> bool:
        """Check whether ``state`` is terminal (no outgoing transitions).

        Parameters:
            state: The state to check.

        Returns:
            ``True`` if no transitions leave ``state`` (i.e. REMOVED).
        """
        return len(cls.TRANSITIONS.get(state, frozenset())) == 0

    # ------------------------------------------------------------------
    # Generic transition
    # ------------------------------------------------------------------

    @final
    def transition(
        self,
        target: Target,
        intended: TargetState,
    ) -> bool:
        """Execute a validated state transition on ``target``.

        Parameters:
            target: The Target whose state should change. Its ``state``
                field is mutated in place on success.
            intended: The desired new :class:`TargetState`.

        Returns:
            ``True`` if the transition was valid and applied.
            ``False`` if the transition is not allowed (the Target is
            left unchanged). Invalid attempts are logged at WARNING.

        Note:
            This method is ``@final`` -- subclasses should not override
            it. To customise transition behaviour, override the named
            methods (:meth:`mark_lost`, :meth:`lock`, etc.) instead.
        """
        current: TargetState = target.state

        if not self.can_transition(current, intended):
            allowed: frozenset[TargetState] = self.get_valid_transitions(current)
            allowed_str: str = (
                ", ".join(sorted(s.name for s in allowed))
                if allowed else "(none -- terminal)"
            )
            self._logger.warning(
                "transition REJECTED: target=%s  %s -> %s  "
                "(allowed from %s: %s)",
                target.target_id,
                current.name, intended.name,
                current.name, allowed_str,
            )
            return False

        target.state = intended
        self._logger.info(
            "transition OK: target=%s  %s -> %s",
            target.target_id,
            current.name, intended.name,
        )
        return True

    # ------------------------------------------------------------------
    # Named transition methods (one per allowed transition)
    # ------------------------------------------------------------------

    def mark_lost(self, target: Target) -> bool:
        """Transition: ``ACTIVE -> LOST``.

        Called when a target's track has missed too many consecutive
        detections. The target remains in the store and may be recovered
        if re-detected.

        Parameters:
            target: The Target to mark as lost.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.LOST)

    def mark_recovered(self, target: Target) -> bool:
        """Transition: ``LOST -> RECOVERED``.

        Called when a lost target is re-associated with a new detection.
        RECOVERED is a transient state -- the caller should immediately
        follow with :meth:`activate` to return to ACTIVE.

        Parameters:
            target: The Target to mark as recovered.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.RECOVERED)

    def activate(self, target: Target) -> bool:
        """Transition: ``RECOVERED -> ACTIVE`` or ``LOCKED -> ACTIVE``.

        Returns a target to the normal operational state. Valid from
        RECOVERED (after re-detection) or LOCKED (after operator release).

        Parameters:
            target: The Target to activate.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.ACTIVE)

    def lock(self, target: Target) -> bool:
        """Transition: ``ACTIVE -> LOCKED``.

        Marks a target for exclusive attention (e.g. operator lock,
        region guard). Locked targets typically receive priority
        processing.

        Parameters:
            target: The Target to lock.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.LOCKED)

    def release(self, target: Target) -> bool:
        """Transition: ``LOCKED -> ACTIVE``.

        Releases a previously locked target back to normal tracking.

        Parameters:
            target: The Target to release.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.ACTIVE)

    def remove(self, target: Target) -> bool:
        """Transition: ``LOST -> REMOVED``.

        Permanently removes a target from the lifecycle. REMOVED is
        terminal -- no further transitions are possible.

        .. note::
            This method only changes the Target's state. It does **not**
            delete the Target from any store. Store cleanup is the
            responsibility of :class:`~visioncore.target_manager.store.TargetStore`
            or :class:`~visioncore.target_manager.manager.TargetManager`.

        Parameters:
            target: The Target to remove.

        Returns:
            ``True`` if the transition succeeded.
        """
        return self.transition(target, TargetState.REMOVED)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation."""
        return "TargetLifecycleManager()"


__all__ = ["TargetLifecycleManager"]

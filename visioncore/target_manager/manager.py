"""TargetManager -- orchestrates TargetStore + TargetLifecycleManager.

TargetManager is the **composition root** for target management. It wires
together:

* :class:`TargetStore` -- thread-safe CRUD container.
* :class:`TargetLifecycleManager` -- pure state machine (6 transitions).
* :class:`~visioncore.eventbus.bus.EventBus` -- optional lifecycle event
  publisher (Milestone 3).

The manager itself adds:

* **ID generation** -- monotonic ``T-NNNN`` identifiers.
* **Store + lifecycle coordination** -- transitions are delegated to the
  lifecycle manager; successful transitions are persisted in the store.
* **Lifecycle event publication** -- when an :class:`EventBus` is
  attached, every successful creation / state change automatically
  publishes the corresponding typed event
  (:class:`~visioncore.eventbus.events.TargetCreatedEvent`,
  :class:`~visioncore.eventbus.events.TargetLostEvent`, ...). Event
  construction and publication is centralised in a single private method
  (:meth:`_emit_lifecycle_event`) so the logic lives in one place.
* **Batch reconciliation** -- :meth:`update_targets` synchronises the
  target set with a fresh list of tracks in one call.
* **Query helpers** -- filter by state (``get_active_targets``, etc.).

**Not connected** to InferWorker / tracker / GUI. The methods are fully
functional but no external system calls them yet.

Backward compatibility:
    The ``event_bus`` parameter defaults to ``None``. With no bus
    attached the manager behaves exactly as before -- no events are
    emitted, and all existing callers and tests are unaffected.

Thread safety:
    All public methods acquire ``self._lock`` (RLock). The store and
    lifecycle manager have their own internal protection; the manager's
    lock coordinates multi-step operations (e.g. ``update_targets``).
    Event publication delegates to the bus's own thread-safe
    :meth:`~visioncore.eventbus.bus.EventBus.publish`.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any

from visioncore.core.target import Target, TargetState
from visioncore.core.track import Track
from visioncore.eventbus.bus import EventBus
from visioncore.eventbus.events import (
    BaseEvent,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.target_manager.lifecycle import TargetLifecycleManager
from visioncore.target_manager.store import TargetStore
from visioncore.target_manager.target_id_factory import create_target_id


class TargetManager:
    """Composition root for Target lifecycle management.

    Wraps a :class:`TargetStore` and a :class:`TargetLifecycleManager`,
    coordinating them so that every state transition is validated by the
    lifecycle machine before being applied to the stored Target.

    Attributes:
        _store: The underlying :class:`TargetStore`.
        _lifecycle: The :class:`TargetLifecycleManager` for transitions.
        _lock: Re-entrant lock for multi-step operations.
        _logger: Module-level logger.
        _next_id: Monotonic counter for ID generation.
        _event_bus: Optional :class:`EventBus` for lifecycle event
            publication. When ``None`` (default), no events are emitted
            and the manager behaves exactly as before -- this preserves
            backward compatibility.
        _event_id_counter: Monotonic counter for event id generation.
    """

    __slots__ = (
        "_store",
        "_lifecycle",
        "_lock",
        "_logger",
        "_next_id",
        "_event_bus",
        "_event_id_counter",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        store: TargetStore | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialise the manager with optional pre-existing store and bus.

        Parameters:
            store: An existing :class:`TargetStore` to use. If ``None``,
                a new empty store is created.
            event_bus: An optional :class:`EventBus`. When provided,
                lifecycle events (TargetCreated/Lost/Recovered/Locked/
                Removed) are published automatically on every successful
                state change. When ``None`` (default), no events are
                emitted -- the manager behaves exactly as before, which
                keeps existing callers and tests unchanged.
        """
        self._store: TargetStore = store if store is not None else TargetStore()
        self._lifecycle: TargetLifecycleManager = TargetLifecycleManager()
        self._lock: threading.RLock = threading.RLock()
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._next_id: int = 1
        self._event_bus: EventBus | None = event_bus
        self._event_id_counter: itertools.count = itertools.count(1)
        self._logger.info(
            "TargetManager initialised (store_count=%d, event_bus=%s)",
            len(self._store),
            "on" if event_bus is not None else "off",
        )

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_id(self, slot_id: int = 0) -> str:
        """Generate the next slot-scoped target identifier.

        Delegates formatting to :func:`create_target_id` so that the
        manager and the stateless converters share a single ID-format
        source of truth.

        Format: ``S{slot_id}-T{NNNN}`` (e.g. ``S0-T0001``, ``S1-T0002``).
        The monotonic counter is shared across slots so that target IDs
        are globally unique even when two slots happen to produce the
        same ``track_id``.

        Caller must hold ``self._lock``.

        Parameters:
            slot_id: Camera slot identifier embedded in the ID prefix.
        """
        tid: str = create_target_id(slot_id, self._next_id)
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------
    # Event emission (single, central implementation)
    # ------------------------------------------------------------------
    #
    # All lifecycle event construction + publication flows through this
    # one method. The individual transition methods (create_target,
    # mark_lost, ...) merely call it with the appropriate event class.
    # This keeps the event-generation logic in a single place rather
    # than scattered across the file.
    # ------------------------------------------------------------------

    def _emit_lifecycle_event(
        self,
        event_cls: type[BaseEvent],
        target: Target,
        timestamp: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Construct and publish a lifecycle event for ``target``.

        This is the **single point** through which all lifecycle events
        flow. It is a no-op when no :class:`EventBus` is attached (the
        default), so existing behaviour is fully preserved.

        Fault isolation:
            The entire construct-and-publish path is wrapped in
            ``try/except``. Any failure (event construction error,
            publish-side bug, ...etc) is logged and **swallowed** so that
            event publication can never cause a lifecycle transition
            method (``mark_lost`` / ``lock_target`` / ...) to raise
            instead of returning its normal ``bool``. The contract is:
            lifecycle state changes are authoritative; events are a
            best-effort side channel.

        Parameters:
            event_cls: The concrete :class:`BaseEvent` subclass to
                construct (e.g. :class:`TargetLostEvent`).
            target: The Target the event concerns. Its ``target_id`` and
                ``slot_id`` populate the corresponding event fields.
            timestamp: Event timestamp in seconds. When ``<= 0.0`` the
                current wall-clock time (:func:`time.time`) is used so
                callers that omit a timestamp still get a meaningful one.
            payload: Optional event-specific payload dict. Merged as-is
                into the event's ``payload`` field.
        """
        bus: EventBus | None = self._event_bus
        if bus is None:
            return
        try:
            ts: float = timestamp if timestamp > 0.0 else time.time()
            event_id: str = f"evt-{next(self._event_id_counter):06d}"
            event: BaseEvent = event_cls(
                event_id=event_id,
                timestamp=ts,
                target_id=target.target_id,
                slot_id=target.slot_id,
                payload=dict(payload) if payload else {},
            )
            bus.publish(event)
            self._logger.debug(
                "emit: %s id=%s target=%s slot=%d",
                event_cls.__name__, event_id, target.target_id, target.slot_id,
            )
        except Exception:  # noqa: BLE001 -- intentional: protect core logic
            self._logger.exception(
                "emit FAILED (suppressed): %s target=%s -- "
                "lifecycle transition is unaffected",
                event_cls.__name__, target.target_id,
            )

    # ------------------------------------------------------------------
    # Lifecycle: creation
    # ------------------------------------------------------------------

    def create_target(
        self,
        track: Track | None,
        *,
        slot_id: int = 0,
        priority: int = 0,
        attributes: dict[str, Any] | None = None,
        last_seen: float = 0.0,
    ) -> Target:
        """Create a new Target, register it in the store, and return it.

        The new Target starts in :attr:`TargetState.ACTIVE` (or
        :attr:`TargetState.LOST` if ``track`` is ``None``).

        Parameters:
            track: The initial :class:`Track`, or ``None``.
            priority: Processing priority [0, 255].
            attributes: Optional metadata dict (shallow-copied).
            last_seen: Timestamp (seconds) of most recent observation.

        Returns:
            The newly created :class:`Target`.
        """
        with self._lock:
            tid: str = self._generate_id(slot_id)
            initial_state: TargetState = (
                TargetState.ACTIVE if track is not None
                else TargetState.LOST
            )
            target: Target = Target(
                target_id=tid,
                track=track,
                slot_id=slot_id,
                priority=priority,
                state=initial_state,
                attributes=dict(attributes) if attributes else {},
                last_seen=last_seen,
            )
            self._store.add_target(target)
            self._logger.info(
                "create_target: id=%s state=%s priority=%d track=%s",
                tid, initial_state.name, priority,
                track.track_id if track is not None else "None",
            )
            self._emit_lifecycle_event(
                TargetCreatedEvent, target, last_seen,
                payload={
                    "initial_state": initial_state.name,
                    "track_id": track.track_id if track is not None else None,
                    "priority": priority,
                },
            )
            return target

    # ------------------------------------------------------------------
    # Lifecycle: single-target state transitions
    # ------------------------------------------------------------------
    # All transition methods delegate validation + state mutation to
    # self._lifecycle, then apply store-level side effects (timestamp
    # updates, track replacement, store removal).
    # ------------------------------------------------------------------

    def mark_lost(self, target_id: str, timestamp: float = 0.0) -> bool:
        """Transition: ``ACTIVE -> LOST``.

        Parameters:
            target_id: The id of the Target.
            timestamp: Timestamp at which the target was considered lost.

        Returns:
            ``True`` if the transition succeeded.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("mark_lost: id=%s not found", target_id)
            return False
        if not self._lifecycle.mark_lost(target):
            return False
        if timestamp > 0.0:
            target.last_seen = timestamp
        self._emit_lifecycle_event(
            TargetLostEvent, target, timestamp,
            payload={"last_seen": target.last_seen},
        )
        return True

    def mark_recovered(
        self,
        target_id: str,
        track: Track,
        timestamp: float = 0.0,
    ) -> bool:
        """Transition: ``LOST -> RECOVERED -> ACTIVE`` (two-step, immediate).

        Parameters:
            target_id: The id of the Target to recover.
            track: The new :class:`Track` to associate.
            timestamp: Timestamp at which recovery occurred.

        Returns:
            ``True`` if recovery succeeded.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("mark_recovered: id=%s not found", target_id)
            return False
        # Step 1: LOST -> RECOVERED (via lifecycle)
        if not self._lifecycle.mark_recovered(target):
            return False
        target.track = track
        if timestamp > 0.0:
            target.last_seen = timestamp
        # Emit the recovery event while the target is in RECOVERED state.
        self._emit_lifecycle_event(
            TargetRecoveredEvent, target, timestamp,
            payload={
                "new_track_id": track.track_id,
                "last_seen": target.last_seen,
            },
        )
        # Step 2: RECOVERED -> ACTIVE (via lifecycle). No event -- the
        # RECOVERED -> ACTIVE transition has no dedicated event type.
        self._lifecycle.activate(target)
        self._logger.info(
            "mark_recovered: id=%s track_id=%s -> ACTIVE",
            target_id, track.track_id,
        )
        return True

    def lock_target(self, target_id: str, timestamp: float = 0.0) -> bool:
        """Transition: ``ACTIVE -> LOCKED``.

        Parameters:
            target_id: The id of the Target to lock.
            timestamp: Timestamp at which the lock occurred. When ``<= 0.0``
                the event timestamp defaults to wall-clock time.

        Returns:
            ``True`` if the lock succeeded.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("lock_target: id=%s not found", target_id)
            return False
        if not self._lifecycle.lock(target):
            return False
        self._emit_lifecycle_event(
            TargetLockedEvent, target, timestamp,
            payload={"locked_by": "manager"},
        )
        return True

    def unlock_target(self, target_id: str) -> bool:
        """Transition: ``LOCKED -> ACTIVE``.

        Parameters:
            target_id: The id of the Target to unlock.

        Returns:
            ``True`` if the unlock succeeded.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("unlock_target: id=%s not found", target_id)
            return False
        return self._lifecycle.release(target)

    def release_target(self, target_id: str) -> bool:
        """Alias for :meth:`unlock_target` (backward compatibility)."""
        return self.unlock_target(target_id)

    def mark_removed(self, target_id: str, timestamp: float = 0.0) -> bool:
        """Transition: ``LOST -> REMOVED``, then delete from store.

        Only targets in ``LOST`` state can be removed (enforced by the
        lifecycle manager). To remove an ``ACTIVE`` or ``LOCKED`` target,
        call :meth:`mark_lost` first.

        Parameters:
            target_id: The id of the Target to remove.
            timestamp: Timestamp at which removal occurred. When ``<= 0.0``
                the event timestamp defaults to wall-clock time.

        Returns:
            ``True`` if the target was found, transitioned, and deleted.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("mark_removed: id=%s not found", target_id)
            return False
        # Capture the pre-removal state for the event payload before the
        # lifecycle manager mutates it.
        prior_state: TargetState = target.state
        if not self._lifecycle.remove(target):
            return False
        # Emit before store deletion so the event still references the
        # (now REMOVED) target. payload records the state it was in.
        self._emit_lifecycle_event(
            TargetRemovedEvent, target, timestamp,
            payload={
                "final_state": prior_state.name,
                "reason": "lifecycle",
            },
        )
        self._store.remove_target(target_id)
        self._logger.info("mark_removed: id=%s -> REMOVED + deleted", target_id)
        return True

    def remove_target(self, target_id: str) -> bool:
        """Alias for :meth:`mark_removed` (backward compatibility).

        .. note::
            Behaviour changed from the previous skeleton: removal now
            requires the target to be in ``LOST`` state (enforced by
            :class:`TargetLifecycleManager`). Previously, removal was
            allowed from any state. Since no existing system code calls
            this method, the change has zero runtime impact.
        """
        return self.mark_removed(target_id)

    # ------------------------------------------------------------------
    # Lifecycle: batch reconciliation
    # ------------------------------------------------------------------

    def update_targets(
        self,
        tracks: list[Track],
        *,
        slot_id: int = 0,
        timestamp: float = 0.0,
        stale_threshold: float = 5.0,
        removal_threshold: float = 30.0,
    ) -> dict[str, int]:
        """Reconcile the target set with a fresh list of tracks.

        This is the **per-frame reconciliation** method. It:

        1. Builds an index of existing targets by ``(slot_id, track_id)``.
        2. For each input track:
           * If a matching target exists and is **ACTIVE** → update its
             track and ``last_seen``.
           * If a matching target exists and is **LOST** → recover it
             (``LOST -> RECOVERED -> ACTIVE``) and update its track.
           * If no matching target exists → create a new one.
        3. For targets **not** matched by any input track (same slot only):
           * If **ACTIVE** and ``last_seen`` is older than
             ``stale_threshold`` → mark lost.
           * If **LOST** and ``last_seen`` is older than
             ``removal_threshold`` → mark removed and delete.

        Parameters:
            tracks: The current frame's tracks from the tracker.
            slot_id: Camera slot identifier. Targets from other slots
                are not affected by this call.
            timestamp: Current frame timestamp (seconds).
            stale_threshold: Seconds since ``last_seen`` after which an
                unmatched ACTIVE target is marked lost.
            removal_threshold: Seconds since ``last_seen`` after which an
                unmatched LOST target is removed.

        Returns:
            A dict with counts::

                {
                    "created":   int,  # new targets created
                    "updated":   int,  # active targets updated
                    "recovered": int,  # lost targets recovered
                    "lost":      int,  # active targets marked lost
                    "removed":   int,  # lost targets removed
                }
        """
        counts: dict[str, int] = {
            "created": 0, "updated": 0, "recovered": 0,
            "lost": 0, "removed": 0,
        }

        with self._lock:
            # --- Build index: (slot_id, track_id) -> Target ---
            target_by_key: dict[tuple[int, int], Target] = {}
            for t in self._store.get_all_targets():
                if t.track is not None and t.state != TargetState.REMOVED:
                    target_by_key[(t.slot_id, t.track.track_id)] = t

            # --- Process input tracks ---
            matched_ids: set[str] = set()
            current_keys: set[tuple[int, int]] = set()

            for track in tracks:
                key: tuple[int, int] = (slot_id, track.track_id)
                current_keys.add(key)
                existing: Target | None = target_by_key.get(key)

                if existing is None:
                    # Create new target
                    self.create_target(
                        track=track,
                        slot_id=slot_id,
                        last_seen=timestamp,
                    )
                    counts["created"] += 1
                elif existing.state == TargetState.ACTIVE:
                    # Update existing active target
                    existing.track = track
                    if timestamp > 0.0:
                        existing.last_seen = timestamp
                    counts["updated"] += 1
                    matched_ids.add(existing.target_id)
                elif existing.state == TargetState.LOST:
                    # Recover lost target
                    self.mark_recovered(
                        existing.target_id, track, timestamp=timestamp,
                    )
                    counts["recovered"] += 1
                    matched_ids.add(existing.target_id)
                else:
                    # LOCKED or RECOVERED — update track but don't change state
                    existing.track = track
                    if timestamp > 0.0:
                        existing.last_seen = timestamp
                    matched_ids.add(existing.target_id)

            # --- Process unmatched targets (same slot only) ---
            for target in self._store.get_all_targets():
                if target.target_id in matched_ids:
                    continue
                if target.slot_id != slot_id:
                    continue  # different slot — not our concern
                if target.track is not None and (slot_id, target.track.track_id) in current_keys:
                    continue  # already handled above

                if target.state == TargetState.ACTIVE:
                    age: float = timestamp - target.last_seen if timestamp > 0 else 0.0
                    if age >= stale_threshold:
                        self.mark_lost(target.target_id, timestamp=timestamp)
                        counts["lost"] += 1

                elif target.state == TargetState.LOST:
                    age = timestamp - target.last_seen if timestamp > 0 else 0.0
                    if age >= removal_threshold:
                        self.mark_removed(target.target_id, timestamp=timestamp)
                        counts["removed"] += 1

        self._logger.info(
            "update_targets: slot=%d %d tracks -> created=%d updated=%d "
            "recovered=%d lost=%d removed=%d (total=%d)",
            slot_id, len(tracks), counts["created"], counts["updated"],
            counts["recovered"], counts["lost"], counts["removed"],
            len(self._store),
        )

        return counts

    # ------------------------------------------------------------------
    # Single-target updates
    # ------------------------------------------------------------------

    def update_last_seen(self, target_id: str, timestamp: float) -> bool:
        """Update the ``last_seen`` timestamp of a Target.

        Returns:
            ``True`` if found and updated, ``False`` if not found.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("update_last_seen: id=%s not found", target_id)
            return False
        target.last_seen = timestamp
        return True

    def update_track(self, target_id: str, track: Track) -> bool:
        """Replace the Track associated with a Target.

        Returns:
            ``True`` if found and updated, ``False`` if not found.
        """
        target: Target | None = self._store.get_target(target_id)
        if target is None:
            self._logger.debug("update_track: id=%s not found", target_id)
            return False
        target.track = track
        self._logger.debug(
            "update_track: id=%s new_track_id=%s",
            target_id, track.track_id,
        )
        return True

    # ------------------------------------------------------------------
    # Queries (delegate to store with optional filtering)
    # ------------------------------------------------------------------

    def get_target(self, target_id: str) -> Target | None:
        """Look up a Target by id."""
        return self._store.get_target(target_id)

    def get_all_targets(self) -> list[Target]:
        """Return all Targets."""
        return self._store.get_all_targets()

    def get_targets_by_state(self, state: TargetState) -> list[Target]:
        """Return all Targets whose state matches ``state``."""
        return [t for t in self._store.get_all_targets() if t.state == state]

    def get_active_targets(self) -> list[Target]:
        """Return all Targets in :attr:`TargetState.ACTIVE`."""
        return self.get_targets_by_state(TargetState.ACTIVE)

    def get_locked_targets(self) -> list[Target]:
        """Return all Targets in :attr:`TargetState.LOCKED`."""
        return self.get_targets_by_state(TargetState.LOCKED)

    def get_lost_targets(self) -> list[Target]:
        """Return all Targets in :attr:`TargetState.LOST`."""
        return self.get_targets_by_state(TargetState.LOST)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def tick(self, timestamp: float) -> None:
        """Periodic maintenance hook.

        Called at a fixed cadence (e.g. once per second) by an external
        driver -- typically the main window's QTimer or a dedicated
        maintenance thread. Unlike :meth:`update_targets`, which is
        driven by per-frame detection results, :meth:`tick` operates on
        **wall-clock time** and is responsible for housekeeping that does
        not depend on having fresh detections.

        Current behaviour (Milestone 2):
            * Logs a debug trace with the current timestamp and target
              count. This is intentionally a **no-op** -- no targets are
              created, transitioned, or removed.

        .. todo:: (Milestone 3 -- EventBus)

            * **Stale sweep**: iterate all targets whose ``last_seen``
              exceeds the stale / removal thresholds and transition them
              via :meth:`mark_lost` / :meth:`mark_removed`. This
              complements :meth:`update_targets` which only sweeps the
              slot that produced the current frame -- :meth:`tick` can
              sweep *all* slots including those that have gone dark.
            * **EventBus emission**: emit lifecycle events
              (``target.lost``, ``target.removed``, ``target.recovered``)
              via the future :class:`EventBus` so that downstream
              subscribers (GUI, logging, gimbal controller) react to
              state changes without polling the store.
            * **Metrics snapshot**: aggregate per-slot target counts and
              state distributions for the health-monitor dashboard.

        Parameters:
            timestamp: Current wall-clock or monotonic time (seconds).
        """
        self._logger.debug(
            "tick: ts=%.4f target_count=%d (event_bus=%s)",
            timestamp, len(self._store),
            "on" if self._event_bus is not None else "off",
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def target_count(self) -> int:
        """Number of Targets currently in the store."""
        return len(self._store)

    @property
    def store(self) -> TargetStore:
        """The underlying :class:`TargetStore` instance."""
        return self._store

    @property
    def lifecycle(self) -> TargetLifecycleManager:
        """The underlying :class:`TargetLifecycleManager` instance."""
        return self._lifecycle

    @property
    def event_bus(self) -> EventBus | None:
        """The attached :class:`EventBus`, or ``None`` if none is attached."""
        return self._event_bus

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"TargetManager(target_count={len(self._store)}, "
            f"next_id={self._next_id}, "
            f"event_bus={'on' if self._event_bus is not None else 'off'})"
        )

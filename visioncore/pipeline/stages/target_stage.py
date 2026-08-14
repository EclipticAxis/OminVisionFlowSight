"""TargetStage -- the target-management stage for the VisionCore pipeline.

This module defines four cohesive concerns, all in one file per the C5
specification:

* :class:`TargetUpdate` -- a frozen return envelope holding the two lists
  a target manager produces: ``targets`` and ``target_states``.
* :class:`TargetManager` -- the abstract target-manager **interface** that
  :class:`TargetStage` depends on. Concrete target managers (a future
  adapter wrapping ``visioncore.target_manager.TargetManager``, or
  :class:`DummyTargetManager` below) implement this interface.
* :class:`DummyTargetManager` -- a test double returning predetermined
  targets and snapshots.
* :class:`TargetStage` -- a :class:`~visioncore.pipeline.base.PipelineStage`
  that wraps a :class:`TargetManager`, reads ``context.tracks``, calls
  ``manager.update(tracks)``, and writes the results to **both**
  ``context.targets`` and ``context.target_states``.

Scope (Milestone C5)
--------------------
C5 delivers **only** the target stage + its interface + a test double.
It deliberately does **not** contain:

* any GUI / Qt code -- the stage produces data, never touches the UI;
* any InferWorker reference -- the stage is a pure data-transformation
  step, decoupled from the legacy inference thread;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Naming note: the ``TargetManager`` ABC defined here is the **pipeline-stage
interface** for a target manager. A *different*, concrete
``TargetManager`` already exists in :mod:`visioncore.target_manager` -- it
is the full-featured implementation (store + lifecycle + event bus). The
two share a name but are distinct: this ABC is the minimal contract the
stage depends on (dependency inversion); the concrete one is adapted to
it in a future milestone. Import with module-qualified paths to avoid
confusion::

    from visioncore.pipeline.stages.target_stage import TargetManager  # ABC
    from visioncore.target_manager import TargetManager as ConcreteTargetManager

Dual ``TargetState`` (mandatory reading)
-----------------------------------------
This module touches **both** types named ``TargetState``:

* :class:`visioncore.core.target.TargetState` -- the **lifecycle enum**
  (ACTIVE / LOST / LOCKED / RECOVERED / REMOVED). It is the ``state``
  field of a :class:`~visioncore.core.target.Target`.
* :class:`visioncore.state.target_state.TargetState` -- the **snapshot
  dataclass** (frozen, transport-ready). It is the element type of
  ``context.target_states`` and of :attr:`TargetUpdate.target_states`.

To avoid collision, this module aliases them at import time:

* ``TargetLifecycleState`` -> the enum.
* ``TargetSnapshot``        -> the snapshot dataclass.

This mirrors the documented pitfall in :mod:`visioncore.pipeline.context`
and :mod:`visioncore.protocol.base`.

Data flow
---------
::

    context.tracks ──> TargetStage.process ──> manager.update(tracks, ts)
                                                     │
                                          TargetUpdate(targets, target_states)
                                                     │
                              ┌──────────────────────┴───────────────────────┐
                              ▼                                                ▼
                    context.targets (REPLACE)              context.target_states (REPLACE)

The stage **always** calls ``manager.update`` -- even when ``tracks`` is
empty. An empty list is the legitimate "no new tracks, advance / stale-
sweep existing targets" signal, not a skip condition (same semantics as
:class:`~visioncore.pipeline.stages.tracker_stage.TrackerStage`).

Thread safety
-------------
:class:`TargetStage` is not thread-safe. It is designed for the C1
pipeline's serial execution model.

Example
-------
    >>> from visioncore.pipeline.stages.target_stage import (
    ...     TargetStage, DummyTargetManager,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> from visioncore.core.track import Track
    >>> from visioncore.core.detection import BBox, Detection
    >>> stage = TargetStage(DummyTargetManager())
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty(timestamp=1.0)
    >>> ctx.tracks.append(Track(1, Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, "person")))
    >>> stage.process(ctx)
    >>> len(ctx.targets), len(ctx.target_states)
    (1, 1)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from visioncore.core.detection import BBox, Detection
from visioncore.core.target import Target
from visioncore.core.target import TargetState as TargetLifecycleState
from visioncore.core.track import Track
from visioncore.pipeline.base import PipelineStage
from visioncore.state.target_state import TargetState as TargetSnapshot

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = [
    "TargetUpdate",
    "TargetManager",
    "TargetError",
    "DummyTargetManager",
    "TargetStage",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class TargetError(RuntimeError):
    """Structured exception signalling a recoverable target-manager failure.

    Target managers *may* raise ``TargetError`` (or any subclass) to
    indicate a failure specific to target management -- e.g. a store
    inconsistency, a lifecycle transition rejected unexpectedly, a
    projection that produced invalid snapshots.

    The enclosing :class:`~visioncore.pipeline.pipeline.Pipeline` does
    **not** catch ``TargetError`` specially -- it propagates out of
    ``run()`` exactly like any other exception, per the C1 "honest
    exceptions" contract.
    """


# ======================================================================
# TargetUpdate -- the return envelope
# ======================================================================

@dataclass(frozen=True, slots=True)
class TargetUpdate:
    """Result of a target-manager update: targets + projected snapshots.

    A :class:`TargetManager.update` call returns a :class:`TargetUpdate`
    holding the two parallel lists the stage writes to the context:

    * ``targets`` -- the complete current set of
      :class:`~visioncore.core.target.Target` entities.
    * ``target_states`` -- the projected
      :class:`~visioncore.state.target_state.TargetState` **snapshots**
      (the transport-ready, observable projection of those targets).

    Both lists are owned by the caller after return; the manager must not
    retain references it later mutates.

    Attributes:
        targets: The complete current target set, possibly empty.
        target_states: Snapshots projected from ``targets``, possibly
            empty. Length need not equal ``len(targets)`` (a manager may
            project only a subset, e.g. active targets only).
    """

    targets: list[Target]
    target_states: list[TargetSnapshot]

    def __repr__(self) -> str:
        """Return a concise representation with list lengths."""
        return (
            f"TargetUpdate(targets={len(self.targets)}, "
            f"target_states={len(self.target_states)})"
        )


# ======================================================================
# TargetManager interface (abstract)
# ======================================================================

class TargetManager(ABC):
    """Abstract target-manager interface -- the contract every manager obeys.

    A :class:`TargetManager` consumes a list of
    :class:`~visioncore.core.track.Track` instances (the current frame's
    tracks) and produces a :class:`TargetUpdate` containing the complete
    current target set plus their projected snapshots. A manager is
    **stateful**: it maintains target identities across calls, reconciling
    new tracks with existing targets, managing lifecycle transitions
    (active / lost / recovered / removed), and projecting observable
    snapshots.

    Naming note
    -----------
    This is the **pipeline-stage interface** (the dependency-inversion
    contract). A concrete, full-featured ``TargetManager`` already exists
    in :mod:`visioncore.target_manager` (store + lifecycle + event bus);
    it is adapted to this interface in a future milestone. The two share
    a name but live in different modules -- use module-qualified imports.

    Lifecycle contract
    ------------------
    * ``initialize()`` -- acquire / reset resources (clear target store,
      reset ID counters). Idempotent.
    * ``update(tracks, *, timestamp)`` -- reconcile targets with ``tracks``,
      return :class:`TargetUpdate` (targets + snapshots).
    * ``shutdown()`` -- release resources. Idempotent, must not raise.
    * ``health_check()`` -- side-effect-free probe; ``True`` iff ready.

    Calling ``update([])`` with an empty list is the legitimate "no new
    tracks this frame" signal: the manager advances / stale-sweeps
    existing targets (marking unmatched ones lost, removing expired ones)
    and returns the reduced target set. It is **not** a no-op.

    Subclassing
    -----------
    Subclasses **must** implement all four abstract methods.
    """

    __slots__ = ()

    @abstractmethod
    def initialize(self) -> None:
        """Acquire / reset resources before the first :meth:`update`.

        Typical work: clear any prior target state, reset ID counters,
        initialise the target store. Must be idempotent -- a second call
        resets the manager to a fresh state.

        Raises:
            TargetError: If initialisation fails. :meth:`shutdown` must
                still be safe to call.
        """

    @abstractmethod
    def update(
        self,
        tracks: list[Track],
        *,
        timestamp: float = 0.0,
    ) -> TargetUpdate:
        """Reconcile targets with ``tracks``; return targets + snapshots.

        Matches the supplied tracks to existing targets (using whatever
        strategy the implementation chooses), updates target states
        (track replacement, last_seen, lifecycle transitions), births
        new targets for unmatched tracks, and projects the observable
        snapshots. Returns a :class:`TargetUpdate` holding both lists.

        Parameters:
            tracks: The current frame's tracks. May be empty -- an empty
                list means "no new tracks, advance / stale-sweep existing
                targets". Must not be ``None``; pass an empty list for
                the no-new-tracks case.
            timestamp: Current frame timestamp (seconds). Used for
                time-based stale / removal decisions and stamped onto
                projected snapshots. Defaults to ``0.0``.

        Returns:
            A :class:`TargetUpdate` with the complete current target set
            and their projected snapshots. Both lists are owned by the
            caller.

        Raises:
            TargetError: For deliberate, recoverable manager failures.
            Exception: Any other exception propagates out of the stage's
                ``process()`` unchanged.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`initialize`.

        Idempotent and must not raise. Clears the target store, releases
        any held resources.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the manager is initialised and ready.

        Side-effect-free. Must not raise; on internal error return
        ``False``.
        """


# ======================================================================
# DummyTargetManager -- test double
# ======================================================================

def _make_default_target() -> Target:
    """Build a synthetic Target for DummyTargetManager defaults."""
    return Target(
        target_id="S0-T0001",
        track=Track(
            track_id=1,
            detection=Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, "person"),
        ),
        slot_id=0,
        state=TargetLifecycleState.ACTIVE,
    )


def _make_default_snapshot(timestamp: float = 0.0) -> TargetSnapshot:
    """Build a synthetic TargetState snapshot for DummyTargetManager defaults."""
    return TargetSnapshot(
        target_id=1,
        local_id=1,
        global_id=None,
        label="person",
        confidence=0.9,
        cx=0.5,
        cy=0.5,
        vx=0.0,
        vy=0.0,
        width=0.2,
        height=0.4,
        timestamp=timestamp,
        camera_id=0,
        metadata={},
    )


class DummyTargetManager(TargetManager):
    """Configurable test target-manager returning predetermined outputs.

    :class:`DummyTargetManager` does no real target management -- no
    store, no lifecycle, no matching. It returns a fixed
    :class:`TargetUpdate` (predetermined targets + snapshots) on every
    :meth:`update` call, making the manager's behaviour **observable and
    controllable** in tests.

    What it is for
    --------------
    * **Stage testing** -- exercise :class:`TargetStage`'s lifecycle
      delegation and dual-output data flow without a real manager.
    * **Pipeline integration** -- feed deterministic targets + snapshots
      to test downstream consumers (protocol adapters, GUI bridges).
    * **Benchmarks** -- measure stage overhead with zero management cost.

    Attributes:
        raise_on_update: An exception instance to raise on the next
            :meth:`update` call, or ``None`` (default) to update normally.
            Persists until cleared.
        initialize_count: Number of times :meth:`initialize` was called.
        update_count: Number of times :meth:`update` was called.
        shutdown_count: Number of times :meth:`shutdown` was called.
        last_tracks: The tracks list passed to the most recent
            :meth:`update` call (for test assertions).
        last_timestamp: The timestamp passed to the most recent
            :meth:`update` call.

    Example:
        >>> mgr = DummyTargetManager()
        >>> mgr.initialize()
        >>> result = mgr.update([])
        >>> len(result.targets), len(result.target_states)
        (1, 1)
    """

    __slots__ = (
        "_targets",
        "_snapshots",
        "_initialized",
        "raise_on_update",
        "initialize_count",
        "update_count",
        "shutdown_count",
        "last_tracks",
        "last_timestamp",
    )

    def __init__(
        self,
        targets: list[Target] | None = None,
        target_states: list[TargetSnapshot] | None = None,
        *,
        raise_on_update: BaseException | None = None,
    ) -> None:
        """Initialise with optional predetermined targets and snapshots.

        Parameters:
            targets: The list of :class:`~visioncore.core.target.Target`
                instances to return. When ``None`` (default), a single
                synthetic active target (``S0-T0001``, a centred person)
                is generated.
            target_states: The list of
                :class:`~visioncore.state.target_state.TargetState`
                snapshots to return. When ``None`` (default), a single
                synthetic snapshot (matching the default target) is
                generated with ``timestamp=0.0``.
            raise_on_update: An exception instance to raise on the next
                :meth:`update` call, or ``None`` (default) to update
                normally.
        """
        self._targets: list[Target] = (
            list(targets) if targets is not None else [_make_default_target()]
        )
        self._snapshots: list[TargetSnapshot] = (
            list(target_states) if target_states is not None
            else [_make_default_snapshot()]
        )
        self._initialized: bool = False
        self.raise_on_update: BaseException | None = raise_on_update
        self.initialize_count: int = 0
        self.update_count: int = 0
        self.shutdown_count: int = 0
        self.last_tracks: list[Track] | None = None
        self.last_timestamp: float | None = None
        logger.debug("DummyTargetManager created: targets=%d snapshots=%d",
                     len(self._targets), len(self._snapshots))

    # ------------------------------------------------------------------
    # Lifecycle (TargetManager contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the manager initialised. Idempotent."""
        self.initialize_count += 1
        self._initialized = True
        logger.debug("DummyTargetManager.initialize: count=%d",
                     self.initialize_count)

    def update(
        self,
        tracks: list[Track],
        *,
        timestamp: float = 0.0,
    ) -> TargetUpdate:
        """Return a :class:`TargetUpdate` with the predetermined outputs.

        Records ``tracks`` and ``timestamp`` for test assertions. Returns
        a :class:`TargetUpdate` holding **fresh lists** (so callers may
        mutate them) containing the same Target / snapshot instances.

        If :attr:`raise_on_update` is set, raise it instead (after
        recording the inputs and incrementing :attr:`update_count`).
        """
        self.update_count += 1
        self.last_tracks = tracks
        self.last_timestamp = timestamp
        if self.raise_on_update is not None:
            logger.debug("DummyTargetManager.update: RAISING %s",
                         type(self.raise_on_update).__name__)
            raise self.raise_on_update
        logger.debug(
            "DummyTargetManager.update: count=%d tracks=%d -> targets=%d snapshots=%d",
            self.update_count, len(tracks),
            len(self._targets), len(self._snapshots),
        )
        return TargetUpdate(
            targets=list(self._targets),
            target_states=list(self._snapshots),
        )

    def shutdown(self) -> None:
        """Mark the manager shut down. Idempotent, never raises."""
        self.shutdown_count += 1
        self._initialized = False
        logger.debug("DummyTargetManager.shutdown: count=%d",
                     self.shutdown_count)

    def health_check(self) -> bool:
        """Return the initialised flag. Never raises."""
        return self._initialized

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        """Whether :meth:`initialize` has been called and not undone."""
        return self._initialized

    @property
    def target_count(self) -> int:
        """Number of Targets returned per ``update`` call."""
        return len(self._targets)

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots returned per ``update`` call."""
        return len(self._snapshots)

    def reset(self) -> None:
        """Reset counters and state. Preserves the target / snapshot buffers."""
        self._initialized = False
        self.initialize_count = 0
        self.update_count = 0
        self.shutdown_count = 0
        self.last_tracks = None
        self.last_timestamp = None


# ======================================================================
# TargetStage -- the pipeline stage
# ======================================================================

class TargetStage(PipelineStage):
    """Pipeline stage that wraps a :class:`TargetManager` and manages targets.

    :class:`TargetStage` is a thin adapter from the
    :class:`TargetManager` interface to the
    :class:`~visioncore.pipeline.base.PipelineStage` interface. It owns no
    target-management logic of its own -- it delegates everything to the
    wrapped manager. This is the dependency-inversion that keeps the
    pipeline decoupled from any specific target-management implementation:
    the stage depends on the **interface**, concrete managers are injected
    at construction.

    Data flow
    ---------
    On :meth:`process`:

    1. Read ``context.tracks`` (may be empty).
    2. Call ``manager.update(tracks, timestamp=context.timestamp)`` --
       **always**, even when empty.
    3. **Replace** ``context.targets`` with the returned
       :attr:`TargetUpdate.targets`.
    4. **Replace** ``context.target_states`` with the returned
       :attr:`TargetUpdate.target_states`.

    Both outputs use REPLACE semantics (clear + extend in place): the
    manager owns the complete current target set and snapshot set at any
    moment. The clear happens **after** a successful update, so if
    ``update`` raises, the prior targets / snapshots are preserved
    (not cleared).

    Lifecycle delegation
    --------------------
    * :meth:`initialize` -> ``manager.initialize()``
    * :meth:`process` -> ``manager.update(context.tracks, ts)``
    * :meth:`shutdown` -> ``manager.shutdown()``
    * :meth:`health_check` -> ``manager.health_check()``

    Attributes:
        _manager: The wrapped :class:`TargetManager` instance.

    Example:
        >>> stage = TargetStage(DummyTargetManager(), name="targets")
        >>> stage.name
        'targets'
    """

    __slots__ = ("_manager",)

    def __init__(
        self, manager: TargetManager, *, name: str | None = None,
    ) -> None:
        """Construct a TargetStage wrapping ``manager``.

        Parameters:
            manager: The :class:`TargetManager` instance to wrap. Must
                not be ``None`` and must be a :class:`TargetManager` (the
                interface is enforced). This is the dependency-inversion
                point.
            name: Optional stage name (defaults to ``"TargetStage"``).

        Raises:
            TypeError: If ``manager`` is ``None`` or not a
                :class:`TargetManager`.
        """
        super().__init__(name=name if name is not None else "TargetStage")
        if manager is None or not isinstance(manager, TargetManager):
            raise TypeError(
                f"manager must be a TargetManager instance, got "
                f"{type(manager).__name__ if manager is not None else 'None'}"
            )
        self._manager: TargetManager = manager
        logger.debug("TargetStage created: name=%s manager=%s",
                     self._name, type(manager).__name__)

    # ------------------------------------------------------------------
    # PipelineStage lifecycle (delegated to the manager)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Delegate to ``manager.initialize()``."""
        self._manager.initialize()
        logger.debug("TargetStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Read ``context.tracks``, update, replace targets + snapshots.

        Always calls ``manager.update(context.tracks, timestamp=...)`` --
        even when tracks is empty (empty == "advance / stale-sweep").
        Then replaces both ``context.targets`` and
        ``context.target_states`` with the returned lists (clear + extend
        in place). If ``update`` raises, the prior targets / snapshots
        are preserved (clear happens only after a successful update).
        """
        tracks: list[Track] = context.tracks
        result: TargetUpdate = self._manager.update(
            tracks, timestamp=context.timestamp,
        )
        # Replace BOTH outputs in place. The clear happens AFTER update
        # succeeded, so an exception preserves prior targets/snapshots.
        context.targets.clear()
        context.targets.extend(result.targets)
        context.target_states.clear()
        context.target_states.extend(result.target_states)
        logger.debug(
            "TargetStage.process: name=%s tracks=%d -> targets=%d snapshots=%d",
            self._name, len(tracks),
            len(result.targets), len(result.target_states),
        )

    def shutdown(self) -> None:
        """Delegate to ``manager.shutdown()``."""
        self._manager.shutdown()
        logger.debug("TargetStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Delegate to ``manager.health_check()``."""
        return self._manager.health_check()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def manager(self) -> TargetManager:
        """The wrapped :class:`TargetManager` instance."""
        return self._manager

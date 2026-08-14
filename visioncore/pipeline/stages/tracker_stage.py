"""TrackerStage -- the tracking stage for the VisionCore pipeline.

This module defines three cohesive concerns, all in one file per the C4
specification:

* :class:`Tracker` -- the abstract tracker **interface** that
  :class:`TrackerStage` depends on. Concrete trackers (a future
  UKF/Kalman wrapper, or :class:`DummyTracker` below) implement this
  interface; the stage never depends on a concrete class.
* :class:`DummyTracker` -- a test double that returns predetermined
  :class:`~visioncore.core.track.Track` instances, with no model and no
  state.
* :class:`TrackerStage` -- a :class:`~visioncore.pipeline.base.PipelineStage`
  that wraps a :class:`Tracker`, reads ``context.detections``, calls
  ``tracker.update(detections)``, and writes the results to
  ``context.tracks``.

Scope (Milestone C4)
--------------------
C4 delivers **only** the tracker stage + its interface + a test double.
It deliberately does **not** contain:

* any ByteTrack / BoTSORT / OSTrack code or imports -- the stage depends
  on the abstract :class:`Tracker` interface only, never on a concrete
  tracking library;
* any Kalman filter, UKF, ReID, or appearance-feature code;
* any modification to ``ai/``, ``camera/``, or ``gui/`` -- the existing
  ``ai/tracker.py`` DetectionTracker continues to run unchanged.

Interface-first design
----------------------
:class:`TrackerStage` is constructed with a :class:`Tracker` (the
interface), not a concrete class. This is the same dependency-inversion
used by :class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`:
the stage knows *that* a tracker tracks, not *how* it tracks. Real tracker
backends (C5+) arrive as :class:`Tracker` subclasses; the stage never
changes.

Data flow
---------
::

    context.detections ──> TrackerStage.process ──> tracker.update(detections)
                                                            │
                                                list[Track]
                                                            │
                                                            ▼
                                                  context.tracks (REPLACE)

Unlike :class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`
(which **extends** ``context.detections`` to allow multi-detector chains),
:class:`TrackerStage` **replaces** ``context.tracks``. A tracker owns the
**complete current track set** at any moment -- its output is the full
state, not an additive contribution. Chaining two tracker stages would
have the second replace the first's output (the second tracker is then
the "current" one).

The stage **always** calls ``tracker.update(detections)``, even when
``detections`` is empty. An empty list is the legitimate "no new
detections this frame, just predict / advance existing tracks" signal --
it is **not** a skip condition. This contrasts with
:class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`, which
skips when ``context.frame`` is ``None``.

Thread safety
-------------
:class:`TrackerStage` is not thread-safe. It is designed for the C1
pipeline's serial execution model. Concrete trackers that share state
across stages must serialise access externally.

Example
-------
    >>> from visioncore.pipeline.stages.tracker_stage import (
    ...     TrackerStage, DummyTracker,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> from visioncore.core.detection import BBox, Detection
    >>> stage = TrackerStage(DummyTracker())
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> ctx.detections.append(Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, "person"))
    >>> stage.process(ctx)
    >>> len(ctx.tracks)
    1
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from visioncore.core.detection import BBox, Detection
from visioncore.core.track import Track
from visioncore.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["Tracker", "TrackerError", "DummyTracker", "TrackerStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class TrackerError(RuntimeError):
    """Structured exception signalling a recoverable tracker failure.

    Trackers *may* raise ``TrackerError`` (or any subclass) to indicate
    a failure specific to the tracker's domain -- e.g. a filter diverged
    (NaN covariance), a match produced an inconsistent state, an
    appearance model reported an OOM.

    Raising ``TrackerError`` instead of a bare ``RuntimeError`` lets
    upstream callers distinguish "the tracker deliberately reported a
    failure" from "an unexpected bug crashed the tracker". The enclosing
    :class:`~visioncore.pipeline.pipeline.Pipeline` does **not** catch
    ``TrackerError`` specially -- it propagates out of ``run()`` exactly
    like any other exception, per the C1 "honest exceptions" contract.
    """


# ======================================================================
# Tracker interface (abstract)
# ======================================================================

class Tracker(ABC):
    """Abstract tracker interface -- the contract every tracker obeys.

    A :class:`Tracker` consumes a list of
    :class:`~visioncore.core.detection.Detection` instances (the current
    frame's detections) and produces a list of
    :class:`~visioncore.core.track.Track` instances (the current set of
    tracked objects). A tracker is **stateful**: it maintains track
    identities across calls, associating new detections with existing
    tracks, predicting positions, and managing track birth / death.

    The interface is **algorithm-agnostic**: it says nothing about
    ByteTrack, BoTSORT, OSTrack, Kalman filters, UKF, or any other
    concrete tracking algorithm. Concrete trackers implement ``update``
    however they choose; the stage and the pipeline never inspect the
    mechanism.

    Lifecycle contract
    ------------------
    * ``initialize()`` -- acquire / reset resources (clear track state,
      init filters). Idempotent.
    * ``update(detections)`` -- associate ``detections`` with existing
      tracks, advance state, return the current complete track set.
    * ``shutdown()`` -- release resources. Idempotent, must not raise.
    * ``health_check()`` -- side-effect-free probe; ``True`` iff ready.
      Must not raise.

    Calling ``update([])`` with an empty list is the legitimate "no new
    detections this frame" signal: the tracker advances / predicts
    existing tracks and returns the (possibly reduced) current set. It is
    **not** a no-op and must not raise.

    Subclassing
    -----------
    Subclasses **must** implement all four abstract methods.
    """

    __slots__ = ()

    @abstractmethod
    def initialize(self) -> None:
        """Acquire / reset resources before the first :meth:`update`.

        Typical work: clear any prior track state, initialise filter
        parameters, reset ID counters. Must be idempotent -- a second
        call resets the tracker to a fresh state. Calling ``initialize``
        between two videos / sessions is the supported way to clear
        cross-session state.

        Raises:
            TrackerError: If initialisation fails and the tracker cannot
                proceed. :meth:`shutdown` must still be safe to call.
        """

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate ``detections`` with existing tracks; return track set.

        The tracker maintains state across calls: it matches the supplied
        detections to its existing tracks (using whatever algorithm it
        chooses -- IoU, Hungarian, cascade, appearance, ...), updates
        track states (position, velocity, age, lost_frames), births new
        tracks for unmatched detections, and removes tracks that have
        been lost too long.

        The returned list is the **complete current track set** -- every
        track the tracker currently holds, in tracker-defined order. The
        :class:`TrackerStage` replaces ``context.tracks`` with this list.

        Parameters:
            detections: The current frame's detections. May be empty --
                an empty list means "no new detections this frame, just
                predict / advance existing tracks". Must not be ``None``;
                pass an empty list explicitly for the predict-only case.

        Returns:
            A list of :class:`~visioncore.core.track.Track` instances --
            the complete current track set, possibly empty. The returned
            list is owned by the caller; the tracker must not retain a
            reference that it later mutates (return a fresh list or a
            copy).

        Raises:
            TrackerError: For deliberate, recoverable tracker failures.
            Exception: Any other exception propagates out of the stage's
                ``process()`` unchanged.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`initialize`.

        Idempotent and must not raise. Clears track state, releases
        filter / ReID resources. The enclosing stage guarantees this is
        called even if :meth:`update` raised.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the tracker is initialised and ready.

        Side-effect-free. Must not raise; on internal error return
        ``False``.
        """


# ======================================================================
# DummyTracker -- test double
# ======================================================================

class DummyTracker(Tracker):
    """Configurable test tracker that returns predetermined Tracks.

    :class:`DummyTracker` does no real tracking -- no Kalman filter, no
    matching, no state. It returns a fixed list of
    :class:`~visioncore.core.track.Track` instances on every
    :meth:`update` call, making the tracker's behaviour **observable and
    controllable** in tests.

    What it is for
    --------------
    * **Stage testing** -- exercise :class:`TrackerStage`'s lifecycle
      delegation and data flow without a real tracker.
    * **Pipeline integration** -- feed deterministic tracks into a
      pipeline to test downstream target-management / projection stages.
    * **Benchmarks** -- measure stage overhead with zero tracking cost.

    Attributes:
        raise_on_update: An exception instance to raise on the next
            :meth:`update` call, or ``None`` (default) to track normally.
            Persists until cleared -- set, update raises, clear, update
            works. Used by exception-propagation tests.
        initialize_count: Number of times :meth:`initialize` was called.
        update_count: Number of times :meth:`update` was called.
        shutdown_count: Number of times :meth:`shutdown` was called.
        last_detections: The detections list passed to the most recent
            :meth:`update` call (for test assertions on what the stage
            forwarded).

    Example:
        >>> trk = DummyTracker()
        >>> trk.initialize()
        >>> tracks = trk.update([])
        >>> len(tracks)
        1
        >>> tracks[0].track_id
        1
    """

    __slots__ = (
        "_tracks",
        "_initialized",
        "raise_on_update",
        "initialize_count",
        "update_count",
        "shutdown_count",
        "last_detections",
    )

    def __init__(
        self,
        tracks: list[Track] | None = None,
        *,
        raise_on_update: BaseException | None = None,
    ) -> None:
        """Initialise with an optional list of Tracks to return.

        Parameters:
            tracks: The list of :class:`~visioncore.core.track.Track`
                instances to return from :meth:`update`. When ``None``
                (default), a single synthetic Track (track_id=1, a
                centred person detection) is generated.
            raise_on_update: An exception instance to raise on the next
                :meth:`update` call, or ``None`` (default) to track
                normally.
        """
        if tracks is not None:
            self._tracks: list[Track] = list(tracks)
        else:
            self._tracks = [
                Track(
                    track_id=1,
                    detection=Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, "person"),
                ),
            ]
        self._initialized: bool = False
        self.raise_on_update: BaseException | None = raise_on_update
        self.initialize_count: int = 0
        self.update_count: int = 0
        self.shutdown_count: int = 0
        self.last_detections: list[Detection] | None = None
        logger.debug("DummyTracker created: tracks=%d", len(self._tracks))

    # ------------------------------------------------------------------
    # Lifecycle (Tracker contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the tracker initialised. Idempotent."""
        self.initialize_count += 1
        self._initialized = True
        logger.debug("DummyTracker.initialize: count=%d", self.initialize_count)

    def update(self, detections: list[Detection]) -> list[Track]:
        """Return a shallow copy of the predetermined Tracks.

        Records ``detections`` in :attr:`last_detections` so tests can
        assert on what the stage forwarded. Returns a fresh list (so
        callers may mutate it without affecting the tracker's internal
        buffer) containing the same Track instances (which are mutable
        dataclass instances -- callers that need isolation must copy).

        If :attr:`raise_on_update` is set, raise it instead (after
        incrementing :attr:`update_count` and recording ``detections``).
        """
        self.update_count += 1
        self.last_detections = detections
        if self.raise_on_update is not None:
            logger.debug("DummyTracker.update: RAISING %s",
                         type(self.raise_on_update).__name__)
            raise self.raise_on_update
        logger.debug("DummyTracker.update: count=%d detections=%d returning %d",
                     self.update_count, len(detections), len(self._tracks))
        return list(self._tracks)

    def shutdown(self) -> None:
        """Mark the tracker shut down. Idempotent, never raises."""
        self.shutdown_count += 1
        self._initialized = False
        logger.debug("DummyTracker.shutdown: count=%d", self.shutdown_count)

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
    def track_count(self) -> int:
        """Number of Tracks returned per ``update`` call."""
        return len(self._tracks)

    def reset(self) -> None:
        """Reset counters and state. Preserves the track buffer."""
        self._initialized = False
        self.initialize_count = 0
        self.update_count = 0
        self.shutdown_count = 0
        self.last_detections = None


# ======================================================================
# TrackerStage -- the pipeline stage
# ======================================================================

class TrackerStage(PipelineStage):
    """Pipeline stage that wraps a :class:`Tracker` and tracks detections.

    :class:`TrackerStage` is a thin adapter from the
    :class:`Tracker` interface to the
    :class:`~visioncore.pipeline.base.PipelineStage` interface. It owns no
    tracking logic of its own -- it delegates everything to the wrapped
    tracker. This is the dependency-inversion that keeps the pipeline
    decoupled from any specific tracking algorithm (ByteTrack / BoTSORT /
    OSTrack / Kalman / UKF / ...): the stage depends on the **interface**,
    concrete trackers are injected at construction.

    Data flow
    ---------
    On :meth:`process`:

    1. Read ``context.detections`` (may be empty).
    2. Call ``tracker.update(detections)`` -- **always**, even when empty.
       An empty list is the legitimate "no new detections, just predict"
       signal, not a skip condition.
    3. **Replace** ``context.tracks`` with the returned list (clear + extend
       in place, preserving the list object identity).

    Replace vs extend
    -----------------
    Unlike :class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`
    (which **extends** detections to allow multi-detector chains),
    :class:`TrackerStage` **replaces** tracks. A tracker owns the complete
    current track set at any moment -- its output is the full state, not
    an additive contribution. Chaining two tracker stages would have the
    second replace the first's output.

    Lifecycle delegation
    --------------------
    * :meth:`initialize` -> ``tracker.initialize()``
    * :meth:`process` -> ``tracker.update(context.detections)``
    * :meth:`shutdown` -> ``tracker.shutdown()``
    * :meth:`health_check` -> ``tracker.health_check()``

    Attributes:
        _tracker: The wrapped :class:`Tracker` instance.

    Example:
        >>> stage = TrackerStage(DummyTracker(), name="track")
        >>> stage.name
        'track'
    """

    __slots__ = ("_tracker",)

    def __init__(self, tracker: Tracker, *, name: str | None = None) -> None:
        """Construct a TrackerStage wrapping ``tracker``.

        Parameters:
            tracker: The :class:`Tracker` instance to wrap. Must not be
                ``None`` and must be a :class:`Tracker` (the interface is
                enforced, not a concrete algorithm class). This is the
                dependency-inversion point: the stage depends on the
                interface, the concrete tracker is injected.
            name: Optional stage name (defaults to ``"TrackerStage"``).
                Useful when a pipeline chains multiple tracker stages.

        Raises:
            TypeError: If ``tracker`` is ``None`` or not a
                :class:`Tracker`.

        Example:
            >>> stage = TrackerStage(DummyTracker())
            >>> stage.health_check()
            False
        """
        super().__init__(name=name if name is not None else "TrackerStage")
        if tracker is None or not isinstance(tracker, Tracker):
            raise TypeError(
                f"tracker must be a Tracker instance, got "
                f"{type(tracker).__name__ if tracker is not None else 'None'}"
            )
        self._tracker: Tracker = tracker
        logger.debug("TrackerStage created: name=%s tracker=%s",
                     self._name, type(tracker).__name__)

    # ------------------------------------------------------------------
    # PipelineStage lifecycle (delegated to the tracker)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Delegate to ``tracker.initialize()``."""
        self._tracker.initialize()
        logger.debug("TrackerStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Read ``context.detections``, track, replace ``context.tracks``.

        Always calls ``tracker.update(context.detections)`` -- even when
        the detection list is empty (empty == "no new detections, just
        predict"). Then replaces ``context.tracks`` with the returned
        track set (clear + extend in place).
        """
        detections: list[Detection] = context.detections
        tracks: list[Track] = self._tracker.update(detections)
        # Replace in place (clear + extend) so any external reference to
        # the list object sees the new contents. A tracker owns the
        # complete current track set -- its output is the full state,
        # not an additive contribution.
        context.tracks.clear()
        context.tracks.extend(tracks)
        logger.debug(
            "TrackerStage.process: name=%s detections=%d tracks=%d",
            self._name, len(detections), len(context.tracks),
        )

    def shutdown(self) -> None:
        """Delegate to ``tracker.shutdown()``."""
        self._tracker.shutdown()
        logger.debug("TrackerStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Delegate to ``tracker.health_check()``."""
        return self._tracker.health_check()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tracker(self) -> Tracker:
        """The wrapped :class:`Tracker` instance."""
        return self._tracker

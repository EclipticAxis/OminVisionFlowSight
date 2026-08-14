"""PipelineContext -- the mutable data envelope that flows through a Pipeline.

A :class:`PipelineContext` is the single object passed stage-by-stage through a
:class:`~visioncore.pipeline.pipeline.Pipeline`. Each
:class:`~visioncore.pipeline.base.PipelineStage` reads from and writes to the
same context instance, so the context acts as the **shared blackboard** between
stages: a capture stage populates :attr:`frame`, a detection stage appends to
:attr:`detections`, a tracking stage consumes detections and produces
:attr:`tracks`, a target-management stage promotes tracks to :attr:`targets`,
and a projection stage derives :attr:`target_states` snapshots from targets.

Design rationale
----------------
* **Mutable, not frozen**: unlike :class:`~visioncore.core.frame.Frame` or
  :class:`~visioncore.core.detection.Detection` (which are immutable value
  objects), the context is a *working surface* that stages mutate as data
  accumulates. Mutability is intentional and confined to a single pipeline
  run on a single thread.
* **Slotted**: ``slots=True`` keeps the per-frame footprint tight. A pipeline
  processes many frames per second; a fat ``__dict__`` per context would show
  up in allocation profiles.
* **Plain containers**: every flowing field is a ``list`` (or a single Frame /
  None). No bespoke collection types, no numpy in the schema -- the context is
  serialisable and framework-agnostic.
* **No behaviour**: the context holds data only. It does not detect, track,
  or publish. All logic lives in stages. This keeps the data path trivial to
  reason about and test.

The ``TargetState`` naming pitfall
----------------------------------
:attr:`target_states` holds **snapshot** instances from
:mod:`visioncore.state.target_state`, **not** the lifecycle enum re-exported at
the top-level ``visioncore`` namespace. These two types share a name but are
semantically distinct (see ``docs/targetstate-design.md`` and the warning in
:mod:`visioncore.protocol.base`):

    from visioncore.state.target_state import TargetState   # snapshot (here)
    from visioncore import TargetState                       # lifecycle enum (NOT here)

The snapshot is the *observable, transport-ready* projection of a target --
exactly what a pipeline's projection stage produces and what the protocol
layer (Milestone B2) consumes. The lifecycle enum is metadata *on* a
:class:`~visioncore.core.target.Target` and is already captured by
:attr:`targets`.

Thread safety
-------------
PipelineContext is **not** thread-safe. It is designed to flow through a
single pipeline on a single thread. Stages that need to fan out to worker
threads must copy the context (or the relevant fields) before handing it off,
and must serialise any write-back. The shared-blackboard contract assumes
serial stage execution.

Example
-------
    >>> from visioncore.pipeline.context import PipelineContext
    >>> ctx = PipelineContext.empty(timestamp=0.0)
    >>> ctx.frame is None
    True
    >>> ctx.detections
    []
    >>> ctx.metadata["stage_trace"] = []
    >>> ctx.metadata["stage_trace"].append("capture")
    >>> ctx.metadata["stage_trace"]
    ['capture']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from visioncore.core.detection import Detection
    from visioncore.core.frame import Frame
    from visioncore.core.target import Target
    from visioncore.core.track import Track
    from visioncore.state.target_state import TargetState


__all__ = ["PipelineContext"]


@dataclass(slots=True)
class PipelineContext:
    """Mutable per-frame data envelope that flows through a Pipeline.

    A single context instance is created per pipeline run and is passed,
    stage by stage, to each :class:`~visioncore.pipeline.base.PipelineStage`.
    Stages read upstream fields and write downstream fields, building up the
    detection -> track -> target -> snapshot chain as they go.

    Attributes:
        frame: The current captured :class:`~visioncore.core.frame.Frame`.
            ``None`` until a capture/ingest stage populates it. Downstream
            stages (detector, tracker) typically require this to be set.
        detections: Detector outputs for the current frame. Empty until a
            detection stage runs. Order is detector-defined (e.g. by score).
        tracks: Tracked objects produced by associating detections across
            frames. Empty until a tracking stage runs.
        targets: High-level :class:`~visioncore.core.target.Target` entities
            under sustained observation. Empty until a target-management
            stage runs. A target aggregates one or more tracks over time.
        target_states: Projected :class:`~visioncore.state.target_state.TargetState`
            **snapshots** (the observable, transport-ready projection of
            targets). Empty until a projection stage runs. These snapshots
            are what the protocol layer (Milestone B2) consumes. Note: this
            is the snapshot dataclass, **not** the lifecycle enum.
        timestamp: Processing timestamp in seconds for this pipeline run.
            Usually the frame's capture timestamp; may be wall-clock or
            monotonic. Used for temporal ordering and latency measurement.
            Defaults to ``0.0`` when the caller has no clock.
        metadata: Extensible key-value store for stage-to-stage communication
            that does not belong in the typed schema above. Common uses:
            per-stage trace logs (``metadata["order"]``), feature flags,
            debug counters, or stage-specific annotations. Defaults to an
            empty dict. Stages may freely read and write this dict.

    Example:
        >>> from visioncore.pipeline.context import PipelineContext
        >>> ctx = PipelineContext.empty(timestamp=1.5)
        >>> ctx.timestamp
        1.5
        >>> ctx.detections.append("pretend-detection")  # stages do this
        >>> len(ctx.detections)
        1
    """

    frame: "Frame | None"
    detections: "list[Detection]"
    tracks: "list[Track]"
    targets: "list[Target]"
    target_states: "list[TargetState]"
    timestamp: float
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, timestamp: float = 0.0) -> "PipelineContext":
        """Create a fresh, fully-empty context ready for a pipeline run.

        All collection fields are initialised to empty lists, ``frame`` is
        ``None``, and ``metadata`` is a fresh dict. The only set field is
        ``timestamp``.

        Parameters:
            timestamp: Processing timestamp in seconds. Defaults to ``0.0``
                when the caller has no clock.

        Returns:
            A new empty :class:`PipelineContext`.

        Example:
            >>> ctx = PipelineContext.empty()
            >>> ctx.frame is None and ctx.detections == []
            True
        """
        return cls(
            frame=None,
            detections=[],
            tracks=[],
            targets=[],
            target_states=[],
            timestamp=timestamp,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a compact one-line summary of the context's payload.

        Useful for logging and debugging: shows the counts of each
        collection and whether a frame is attached, without dumping the
        (potentially large) contents.

        Returns:
            A string like ``"frame=yes det=3 trk=2 tgt=2 ts=2 len=5"``.

        Example:
            >>> ctx = PipelineContext.empty(timestamp=2.0)
            >>> ctx.summary()
            'frame=no det=0 trk=0 tgt=0 ts=2 len=0'
        """
        frame_tag: str = "yes" if self.frame is not None else "no"
        return (
            f"frame={frame_tag} "
            f"det={len(self.detections)} "
            f"trk={len(self.tracks)} "
            f"tgt={len(self.targets)} "
            f"ts={len(self.target_states)} "
            f"len={len(self.metadata)}"
        )

    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation.

        Uses :meth:`summary` so the representation stays compact even when
        the collections hold thousands of detections.
        """
        return (
            f"PipelineContext({self.summary()}, "
            f"timestamp={self.timestamp:.4f})"
        )

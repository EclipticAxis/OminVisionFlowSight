"""Adapters wrapping legacy InferWorker backends as Pipeline stage interfaces.

This module bridges the existing ``ai/inference.py`` InferWorker detection /
tracking / target-management backends to the C3-C6 Pipeline stage interfaces
(:class:`~visioncore.pipeline.stages.detector_stage.Detector`,
:class:`~visioncore.pipeline.stages.tracker_stage.Tracker`,
:class:`~visioncore.pipeline.stages.target_stage.TargetManager`,
:class:`~visioncore.pipeline.stages.event_stage.EventBus`).

Each adapter delegates to the **same** InferWorker method that the legacy
inline code path uses, ensuring behavioural equivalence: when
``pipeline_enabled=True``, the Pipeline path calls the same YOLO model, the
same DetectionTracker, the same TargetManager -- just routed through the
Pipeline framework instead of inline calls.

Legacy dict ↔ core dataclass conversion
---------------------------------------
The InferWorker uses ``list[dict]`` for detections (the serialised form of
``ai.detection.Detection``). The Pipeline stages use core dataclasses
(``visioncore.core.detection.Detection``, ``visioncore.core.track.Track``).
The adapters convert at the boundary:

* :func:`_dict_to_core_detection` -- detection dict → core Detection.
* :func:`_core_detection_to_dict` -- core Detection → detection dict.

**Production schema (step-4 parity contract):** detection dicts use the
``x1/y1/x2/y2`` corner keys plus ``confidence`` / ``class_id`` / ``label``
-- the same schema as ``ai.detection.Detection.to_dict()`` and
``DetectionTracker``. The legacy ``bbox`` tuple key is still accepted on
input (backwards compatibility), but **never** appears on output.

**Lossless extras round-trip (step-4 parity contract):** every non-basic
field (``track_id``, ``track_state``, OBB ``polygon``, pose ``keypoints``,
``gesture`` / ``head_attrs``, ...) is deep-copied into the private
:class:`_AdaptedDetection` carrier and restored on the way back out, so
the Detector -> Tracker dict<->core round-trip loses nothing. Basic-field
changes made on the core side still map back to the output corners.

Scope (Milestone C8)
--------------------
C8 delivers adapters for the **core** flow (detect → track → target →
event). Advanced features (ROI re-detection, appearance histograms, UHD
parallel detection, CHC head classification) remain in the legacy inline
path and are applied as post-processing around the Pipeline call. They
will be migrated to dedicated Pipeline stages in subsequent milestones.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from visioncore.core.detection import BBox, Detection as CoreDetection
from visioncore.core.track import Track as CoreTrack
from visioncore.pipeline.stages.detector_stage import Detector
from visioncore.pipeline.stages.event_stage import EventBus
from visioncore.pipeline.stages.target_stage import TargetManager, TargetUpdate
from visioncore.pipeline.stages.tracker_stage import Tracker

if TYPE_CHECKING:
    from visioncore.core.frame import Frame as CoreFrame


__all__ = [
    "InferWorkerDetectorAdapter",
    "InferWorkerTrackerAdapter",
    "InferWorkerTargetManagerAdapter",
    "InferWorkerEventBusAdapter",
    "LegacyRedetectAdapter",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Dict ↔ core conversion helpers
# ======================================================================

#: Detection-dict keys consumed as "basic" fields (mapped onto the core
#: bbox/score/class fields). Everything else is an extra carried through
#: :attr:`_AdaptedDetection.extras`.
_BASIC_KEYS: frozenset[str] = frozenset(
    {"x1", "y1", "x2", "y2", "confidence", "label", "class_id"}
)

#: Private extras key holding the original corner tuple. Because
#: corner -> centre/size -> corner is not float-exact, the snapshot lets
#: :func:`_core_detection_to_dict` reproduce the *identical* corner
#: values when the core bbox is unchanged (see :data:`_SRC_BBOX_KEY`).
_CORNERS_KEY: str = "__corners__"

#: Private extras key holding the ``(x, y, w, h)`` centre-format bbox
#: at ingestion time. Comparing the *current* core bbox against this
#: tuple detects whether a pipeline stage modified the detection: equal
#: -> corners come from the :data:`_CORNERS_KEY` snapshot (bit-exact);
#: different -> corners are derived from the modified core bbox.
_SRC_BBOX_KEY: str = "__source_bbox__"


@dataclass(frozen=True, slots=True, eq=False)
class _AdaptedDetection(CoreDetection):
    """Core Detection plus a lossless extras carrier (internal).

    Deep-copied non-basic detection-dict fields travel in ``extras`` so
    the dict ↔ core round-trip preserves them (keypoints, polygon,
    track_id, track_state, gesture, head attributes, ...). ``extras``
    is excluded from equality/hash (``compare=False``) and from the
    repr, so value semantics stay exactly those of the four core fields
    and the base :class:`~visioncore.core.detection.Detection` contract
    (four fields) is untouched.

    This is a *local* extension used only by the conversion helpers in
    this module; no global id map, no ``ai`` import side effects.
    """

    extras: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __eq__(self, other: Any) -> bool:
        """Value equality on the four core fields (extras excluded).

        Accepts any :class:`~visioncore.core.detection.Detection`, so an
        adapted detection compares equal to a plain core detection with
        the same bbox/score/class identity.
        """
        if not isinstance(other, CoreDetection):
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.score == other.score
            and self.class_id == other.class_id
            and self.class_name == other.class_name
        )

    def __hash__(self) -> int:
        return hash((self.bbox, self.score, self.class_id, self.class_name))


def _extract_corners(d: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract ``(x1, y1, x2, y2)`` corners from a detection dict.

    Production schema (``x1/y1/x2/y2`` keys) takes precedence; the
    legacy ``bbox`` tuple key is accepted as a backwards-compatible
    fallback. Raises ``KeyError`` when neither is present.
    """
    if "x1" in d:
        return (float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]))
    x1, y1, x2, y2 = d["bbox"]
    return (float(x1), float(y1), float(x2), float(y2))


def _dict_to_core_detection(d: dict[str, Any]) -> CoreDetection:
    """Convert a legacy detection dict to a core Detection (lossless).

    Maps the ``(x1, y1, x2, y2)`` corner-format bbox to the core
    ``BBox(x, y, w, h)`` centre-format, ``confidence`` → ``score``,
    ``label`` → ``class_name``. Every non-basic field is deep-copied
    into :attr:`_AdaptedDetection.extras` (plus the private corner
    snapshot) so :func:`_core_detection_to_dict` can restore it.
    """
    corners = _extract_corners(d)
    x1, y1, x2, y2 = corners
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cw, ch = x2 - x1, y2 - y1
    extras: dict[str, Any] = {
        k: copy.deepcopy(v)
        for k, v in d.items()
        if k not in _BASIC_KEYS and k != "bbox"
    }
    extras[_CORNERS_KEY] = corners
    extras[_SRC_BBOX_KEY] = (cx, cy, cw, ch)
    return _AdaptedDetection(
        bbox=BBox(x=cx, y=cy, w=cw, h=ch),
        score=float(d.get("confidence", 0.0)),
        class_id=int(d.get("class_id", 0)),
        class_name=str(d.get("label", "")),
        extras=extras,
    )


def _core_detection_to_dict(core: CoreDetection) -> dict[str, Any]:
    """Convert a core Detection to a production-schema detection dict.

    Inverse of :func:`_dict_to_core_detection`. The output carries
    ``x1/y1/x2/y2`` (never a ``bbox`` key) plus ``confidence`` /
    ``class_id`` / ``label``, with the deep-copied extras restored on
    top. When the core bbox still matches the ingested corners, the
    original corner values are reproduced bit-exactly; when a pipeline
    stage changed the core bbox, the modified corners are derived from
    it instead (snapshot invalidated).
    """
    b: BBox = core.bbox
    hw: float = b.w * 0.5
    hh: float = b.h * 0.5
    derived: tuple[float, float, float, float] = (
        b.x - hw, b.y - hh, b.x + hw, b.y + hh,
    )
    extras: dict[str, Any] = getattr(core, "extras", None) or {}
    snapshot: Any = extras.get(_CORNERS_KEY)
    source_bbox: Any = extras.get(_SRC_BBOX_KEY)
    unchanged = (
        snapshot is not None
        and source_bbox is not None
        and (b.x, b.y, b.w, b.h) == tuple(source_bbox)
    )
    if unchanged:
        x1, y1, x2, y2 = snapshot
    else:
        x1, y1, x2, y2 = derived
    out: dict[str, Any] = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": core.score,
        "class_id": core.class_id,
        "label": core.class_name,
    }
    for key, value in extras.items():
        if key.startswith("__"):
            continue  # private round-trip bookkeeping, never leaked
        out[key] = copy.deepcopy(value)
    return out


def _dict_to_core_track(d: dict[str, Any]) -> CoreTrack:
    """Convert a legacy tracked-detection dict to a core Track."""
    core = _dict_to_core_detection(d)
    extras: dict[str, Any] = getattr(core, "extras", None) or {}
    track_id = extras.get("track_id", d.get("track_id", 0))
    return CoreTrack(
        track_id=int(track_id),
        detection=core,
    )


# ======================================================================
# Detector adapter
# ======================================================================

class InferWorkerDetectorAdapter(Detector):
    """Wraps ``InferWorker._run_inference`` as a :class:`Detector`.

    Calls the exact same inference method as the legacy inline path,
    ensuring the same YOLO/ONNX model is invoked with the same parameters.
    The raw detection dicts are stored in :attr:`last_raw_detections` for
    later recovery by ``_run_pipeline_mode`` (the tracked dicts, not the
    raw dicts, are what get emitted -- but having the raw dicts available
    is useful for diagnostics).

    Attributes:
        _worker: The InferWorker instance (weak reference by identity).
        _frame_image: The numpy frame to detect on (set before each run).
        last_raw_detections: The raw detection dicts from the most recent
            :meth:`detect` call.
    """

    __slots__ = ("_worker", "_frame_image", "last_raw_detections")

    def __init__(self, worker: Any) -> None:
        super().__init__()
        self._worker = worker
        self._frame_image: Any = None
        self.last_raw_detections: list[dict] = []

    def set_frame_image(self, image: Any) -> None:
        """Set the numpy frame image for the next :meth:`detect` call."""
        self._frame_image = image

    def initialize(self) -> None:
        logger.debug("InferWorkerDetectorAdapter.initialize (no-op, worker already init)")

    def detect(self, frame: "CoreFrame") -> list[CoreDetection]:
        raw: list[dict] = self._worker._run_inference(self._frame_image)
        self.last_raw_detections = raw
        return [_dict_to_core_detection(d) for d in raw]

    def shutdown(self) -> None:
        logger.debug("InferWorkerDetectorAdapter.shutdown (no-op)")

    def health_check(self) -> bool:
        return self._worker._model is not None


# ======================================================================
# Tracker adapter
# ======================================================================

class InferWorkerTrackerAdapter(Tracker):
    """Wraps a per-slot ``DetectionTracker.update`` as a :class:`Tracker`.

    Calls the same tracker as the legacy inline path, ensuring the same
    Kalman/UKF filtering and matching strategy. The tracked detection
    dicts (the tracker's actual output, with track_ids and predicted
    positions) are stored in :attr:`last_tracked_dicts` for recovery by
    ``_run_pipeline_mode`` -- these are the dicts that ultimately get
    emitted via ``detection_ready``.

    .. note:: ROI re-detection parity (step 4): when a redetect callback
       is set via :meth:`set_redetect_callback`, it is forwarded to
       ``tracker.update`` exactly like the legacy inline path; when no
       callback is set, ``tracker.update`` receives only ``frame`` --
       the legacy disabled-path calling convention. The callback is the
       *same* factory product (``InferWorker._make_redetect_callback``)
       that the legacy path uses, so attempt/success/failed counting and
       ROI behaviour are identical in both modes.

    Attributes:
        _worker: The InferWorker instance.
        _slot_id: The camera slot ID (for per-slot tracker lookup).
        _frame_image: The numpy frame (set before each run, for the
            tracker's frame-dependent operations).
        _redetect_callback: Optional ROI-redetect callback forwarded to
            ``tracker.update`` (``None`` = disabled path, frame only).
        last_tracked_dicts: The tracked detection dicts from the most
            recent :meth:`update` call.
    """

    __slots__ = (
        "_worker", "_slot_id", "_frame_image",
        "_redetect_callback", "last_tracked_dicts",
    )

    def __init__(self, worker: Any, slot_id: int) -> None:
        super().__init__()
        self._worker = worker
        self._slot_id = slot_id
        self._frame_image: Any = None
        self._redetect_callback: Any = None
        self.last_tracked_dicts: list[dict] = []

    def set_frame_image(self, image: Any) -> None:
        """Set the numpy frame for the next :meth:`update` call."""
        self._frame_image = image

    def set_redetect_callback(self, callback: Any) -> None:
        """Set (or clear, with ``None``) the ROI-redetect callback.

        Mirrors the legacy ``redetect_enabled`` branching: ``None``
        keeps the ``tracker.update(dicts, frame=...)`` convention.
        """
        self._redetect_callback = callback

    def initialize(self) -> None:
        logger.debug("InferWorkerTrackerAdapter.initialize slot=%d (no-op)", self._slot_id)

    def update(self, detections: list[CoreDetection]) -> list[CoreTrack]:
        # Convert core Detections back to dicts for the legacy tracker.
        dicts: list[dict] = [_core_detection_to_dict(d) for d in detections]
        tracker = self._worker._trackers.get(self._slot_id)
        if tracker is None:
            self.last_tracked_dicts = dicts
            return [_dict_to_core_track(d) for d in dicts]
        redetect_callback = self._redetect_callback
        if redetect_callback is not None:
            # Enabled path: same signature as the legacy inline call.
            tracked: list[dict] = tracker.update(
                dicts, redetect_callback=redetect_callback,
                frame=self._frame_image,
            )
        else:
            # Disabled path: frame only, exactly like the legacy call.
            tracked = tracker.update(dicts, frame=self._frame_image)
        # Update appearance histograms (same as legacy path).
        tracker.update_appearance(self._frame_image)
        self.last_tracked_dicts = tracked
        return [_dict_to_core_track(d) for d in tracked]

    def shutdown(self) -> None:
        logger.debug("InferWorkerTrackerAdapter.shutdown slot=%d (no-op)", self._slot_id)

    def health_check(self) -> bool:
        return self._slot_id in self._worker._trackers


# ======================================================================
# TargetManager adapter
# ======================================================================

class InferWorkerTargetManagerAdapter(TargetManager):
    """Wraps the InferWorker's TargetManager as a Pipeline :class:`TargetManager`.

    Delegates to the same ``TargetManager.update_targets`` call that
    ``_bypass_update_targets`` uses, plus projects snapshots. The
    TargetManager instance is the **same** one the InferWorker already
    holds (``self._target_mgr``), so target state is shared between
    Pipeline mode and the existing shadow integration.

    Attributes:
        _worker: The InferWorker instance.
        _slot_id: The camera slot ID.
        updated: ``True`` once :meth:`update` has completed a
            ``TargetManager.update_targets`` call for the current run.
            ``InferWorker`` uses this to skip its run()-tail bypass in
            Pipeline mode (single-write contract): the bypass only runs
            when the TargetStage did **not** complete the update.
        last_targets: The targets list from the most recent update.
        last_snapshots: The snapshots list from the most recent update.
    """

    __slots__ = ("_worker", "_slot_id", "updated", "last_targets", "last_snapshots")

    def __init__(self, worker: Any, slot_id: int) -> None:
        super().__init__()
        self._worker = worker
        self._slot_id = slot_id
        self.updated: bool = False
        self.last_targets: list = []
        self.last_snapshots: list = []

    def initialize(self) -> None:
        logger.debug("InferWorkerTargetManagerAdapter.initialize slot=%d (no-op)", self._slot_id)

    def update(self, tracks: list[CoreTrack], *, timestamp: float = 0.0) -> TargetUpdate:
        # Convert core Tracks to the format detections_to_tracks produces
        # (which is what _bypass_update_targets feeds to update_targets).
        # The existing TargetManager.update_targets takes list[Track].
        from visioncore.core.track import Track as CoreTrack
        mgr = self._worker._target_mgr
        # update_targets already handles slot_id and timestamp.
        # `updated` is set only AFTER the write completes, so a raising
        # update leaves it False and the InferWorker run()-tail bypass
        # still performs the stale-cleanup write (single-write contract).
        mgr.update_targets(tracks, slot_id=self._slot_id, timestamp=timestamp)
        self.updated = True
        targets = mgr.get_all_targets()
        # Project snapshots for active targets.
        from visioncore.state.target_state import TargetState as Snapshot
        snapshots: list[Snapshot] = []
        for t in targets:
            if t.track is not None and t.state.name == "ACTIVE":
                d = t.track.detection
                snapshots.append(Snapshot(
                    target_id=0,  # simplified; real projection would use t.target_id
                    local_id=t.track.track_id,
                    global_id=None,
                    label=d.class_name,
                    confidence=d.score,
                    cx=d.bbox.x, cy=d.bbox.y,
                    vx=t.track.velocity[0], vy=t.track.velocity[1],
                    width=d.bbox.w, height=d.bbox.h,
                    timestamp=timestamp,
                    camera_id=t.slot_id,
                    metadata={},
                ))
        self.last_targets = targets
        self.last_snapshots = snapshots
        return TargetUpdate(targets=targets, target_states=snapshots)

    def shutdown(self) -> None:
        logger.debug("InferWorkerTargetManagerAdapter.shutdown slot=%d (no-op)", self._slot_id)

    def health_check(self) -> bool:
        return self._worker._target_mgr is not None


# ======================================================================
# EventBus adapter
# ======================================================================

class InferWorkerEventBusAdapter(EventBus):
    """Wraps the InferWorker's existing EventBus as a Pipeline :class:`EventBus`.

    Delegates :meth:`publish` to the InferWorker's ``self._event_bus`` --
    the **same** EventBus that the shadow integration (Milestone 3) already
    created. Events published through the Pipeline's EventStage thus reach
    the same bus (and any subscribers) as the existing shadow events.

    Attributes:
        _worker: The InferWorker instance.
    """

    __slots__ = ("_worker",)

    def __init__(self, worker: Any) -> None:
        super().__init__()
        self._worker = worker

    def publish(self, event: Any) -> int:
        return self._worker._event_bus.publish(event)

    def health_check(self) -> bool:
        return self._worker._event_bus is not None


# ======================================================================
# LegacyRedetectAdapter (D9.1)
# ======================================================================

class LegacyRedetectAdapter:
    """Wraps ``InferWorker._redetect_person_in_roi`` as a :class:`Redetector`.

    Delegates to the exact same method that the legacy inline code path
    uses, ensuring behavioural equivalence: when
    ``pipeline_enabled=True``, the Pipeline's RedetectStage calls the
    same YOLO model on the same ROI crop with the same confidence
    threshold.

    Attributes:
        _worker: The InferWorker instance.
    """

    __slots__ = ("_worker",)

    def __init__(self, worker: Any) -> None:
        """Construct a LegacyRedetectAdapter.

        Parameters:
            worker: The InferWorker instance (or any object with a
                ``_redetect_person_in_roi(frame, predicted_box,
                original_confidence)`` method).
        """
        self._worker = worker

    def redetect(
        self,
        frame: Any,
        predicted_box: tuple[float, float, float, float],
        original_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Delegate to ``worker._redetect_person_in_roi``.

        Parameters:
            frame: The frame image (numpy array).
            predicted_box: Normalised ``(x1, y1, x2, y2)`` predicted box.
            original_confidence: The original track's confidence.

        Returns:
            A detection dict, or ``None`` if redetection failed.
        """
        return self._worker._redetect_person_in_roi(
            frame, predicted_box, original_confidence,
        )

    def health_check(self) -> bool:
        """Return ``True`` iff the worker's model is available."""
        return getattr(self._worker, '_model', None) is not None

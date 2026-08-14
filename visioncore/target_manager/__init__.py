"""TargetManager package for VisionCore.

Provides:
    * :class:`TargetStore` -- thread-safe CRUD container for Target entities.
    * :class:`TargetManager` -- lifecycle state-machine manager.
    * :class:`TargetLifecycleManager` -- pure state machine (6 transitions).
    * :func:`track_to_target` / :func:`tracks_to_targets` -- Track→Target
      conversion functions.
    * :func:`detection_to_track` / :func:`detections_to_tracks` --
      detection-dict→Track conversion (used by InferWorker bypass).
    * :func:`create_target_id` -- unified Target ID factory.

Integration status (Milestone 2):
    The package is wired into the live pipeline via a **shadow bypass**
    in :class:`~ai.inference.InferWorker`. After each frame's
    ``detection_ready`` signal, :meth:`InferWorker._bypass_update_targets`
    converts the detection dicts to :class:`Track` objects (via
    :func:`detections_to_tracks`) and feeds them to
    :meth:`TargetManager.update_targets` with the originating ``slot_id``.
    This keeps the TargetManager in sync with the tracker without
    affecting the existing detection output.

    Multi-slot isolation is enforced by the ``(slot_id, track_id)``
    tuple key used inside :meth:`update_targets`, so targets from
    different cameras never collide even when they share the same
    ``track_id``.

    The bypass is **read-only** from the GUI's perspective: the
    :attr:`InferWorker.target_manager` property exposes the manager for
    inspection, but no GUI component currently consumes Target state.

Integration status (Milestone 3 -- EventBus):
    :class:`TargetManager` now accepts an optional
    :class:`~visioncore.eventbus.bus.EventBus`. When attached, every
    successful lifecycle transition (create / mark_lost / mark_recovered
    / lock_target / mark_removed) automatically publishes the matching
    typed event (:class:`~visioncore.eventbus.events.TargetCreatedEvent`
    etc.). Event construction is centralised in
    :meth:`TargetManager._emit_lifecycle_event`. With no bus attached
    (the default) the manager behaves exactly as before -- existing
    callers and tests are unaffected.

Quick start::

    from visioncore.target_manager import (
        TargetManager, TargetLifecycleManager,
        track_to_target, detections_to_tracks, create_target_id,
    )

    mgr = TargetManager()
    target = mgr.create_target(track=my_track, slot_id=0, priority=5)
    mgr.lock_target(target.target_id)
    mgr.remove_target(target.target_id)

    # Or convert detection dicts directly (as InferWorker does):
    tracks = detections_to_tracks(raw_detection_dicts)
    mgr.update_targets(tracks, slot_id=0, timestamp=12.5)

    # Or use the pure state machine directly:
    lm = TargetLifecycleManager()
    lm.mark_lost(target)
    lm.mark_recovered(target)
    lm.activate(target)

    # Or generate an ID without creating a Target:
    tid = create_target_id(slot_id=1, local_id=42)  # "S1-T0042"
"""

from __future__ import annotations

from visioncore.target_manager.converters import (
    detection_to_track,
    detections_to_tracks,
    track_to_target,
    tracks_to_targets,
)
from visioncore.target_manager.lifecycle import TargetLifecycleManager
from visioncore.target_manager.manager import TargetManager
from visioncore.target_manager.store import TargetStore
from visioncore.target_manager.target_id_factory import create_target_id

__all__ = [
    "TargetStore",
    "TargetManager",
    "TargetLifecycleManager",
    "track_to_target",
    "tracks_to_targets",
    "detection_to_track",
    "detections_to_tracks",
    "create_target_id",
]

"""VisionCore pipeline stages -- concrete processing stages.

This subpackage hosts concrete :class:`~visioncore.pipeline.base.PipelineStage`
implementations that turn the C1 pipeline framework into a working
vision-processing chain. Each stage wraps one of the VisionCore input /
processing abstractions and adapts it to the pipeline's
``initialize / process / shutdown / health_check`` contract.

Scope (Milestones C3 + C4 + C5)
-------------------------------
C3 delivered the detector stage:

    DetectorStage   -- wraps a Detector, reads ``context.frame``,
                       writes ``context.detections``.
    Detector        -- the abstract detector interface (model-agnostic).
    DummyDetector   -- a test double returning fixed Detections.
    DetectorError   -- structured exception for recoverable detector failures.

C4 added the tracker stage:

    TrackerStage    -- wraps a Tracker, reads ``context.detections``,
                       writes ``context.tracks``.
    Tracker         -- the abstract tracker interface (algorithm-agnostic).
    DummyTracker    -- a test double returning fixed Tracks.
    TrackerError    -- structured exception for recoverable tracker failures.

C5 adds the target stage:

    TargetStage     -- wraps a TargetManager, reads ``context.tracks``,
                       writes ``context.targets`` AND ``context.target_states``.
    TargetManager   -- the abstract target-manager interface (the pipeline-stage
                       contract; distinct from the concrete visioncore.target_manager.TargetManager).
    TargetUpdate    -- frozen return envelope (targets + target_states).
    DummyTargetManager -- a test double returning fixed targets + snapshots.
    TargetError     -- structured exception for recoverable manager failures.

All stages depend on their respective **interfaces** only, never on a
concrete backend. The detector bans YOLO / RT-DETR / GroundingDINO / SAM;
the tracker bans ByteTrack / BoTSORT / OSTrack; the target stage bans
GUI / InferWorker. Real backends arrive in C6+ as interface subclasses;
the stages never change.

Future milestones will add sibling stages here:

    CaptureStage      (C6) -- wraps a FrameSource, fills context.frame

Each will be a sibling module in this package.
"""

from __future__ import annotations

from visioncore.pipeline.stages.detector_stage import (
    Detector,
    DetectorError,
    DetectorStage,
    DummyDetector,
)
from visioncore.pipeline.stages.event_stage import (
    DummyEventBus,
    EventBus,
    EventStage,
)
from visioncore.pipeline.stages.target_stage import (
    DummyTargetManager,
    TargetError,
    TargetManager,
    TargetStage,
    TargetUpdate,
)
from visioncore.pipeline.stages.tracker_stage import (
    DummyTracker,
    Tracker,
    TrackerError,
    TrackerStage,
)

__all__ = [
    # ---- Detector (C3) ----
    "Detector",
    "DetectorError",
    "DetectorStage",
    "DummyDetector",
    # ---- Tracker (C4) ----
    "Tracker",
    "TrackerError",
    "TrackerStage",
    "DummyTracker",
    # ---- Target (C5) ----
    "TargetManager",
    "TargetError",
    "TargetStage",
    "TargetUpdate",
    "DummyTargetManager",
    # ---- Event (C6) ----
    "EventBus",
    "EventStage",
    "DummyEventBus",
]

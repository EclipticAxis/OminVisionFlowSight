"""Unit tests for visioncore.core data structures.

Verifies creation, field values, defaults, immutability, and __repr__
for the six primary model types: Frame, Detection, Track, Target, Event,
and the TargetState enum.

Run::

    python -m pytest tests/test_core_models.py -v
    # or
    python tests/test_core_models.py
"""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import numpy as np

from visioncore.core import (
    BBox,
    Detection,
    Event,
    Frame,
    Target,
    TargetState,
    Track,
)


# ---------------------------------------------------------------------------
# Star-import sanity: the six primary types must be importable via ``*``
# ---------------------------------------------------------------------------

def test_star_import_six_primary_types():
    """``from visioncore.core import *`` must expose the six model types."""
    import visioncore.core as vc
    required = {"Frame", "Detection", "Track", "Target", "Event", "TargetState"}
    exported = set(vc.__all__)
    assert required.issubset(exported), (
        f"Missing primary types in __all__: {required - exported}"
    )


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def test_frame_creation():
    """Frame stores all four fields and reports correct values."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    f = Frame(frame_id=42, timestamp=1.5, source_id="cam0", image=img)
    assert f.frame_id == 42
    assert f.timestamp == 1.5
    assert f.source_id == "cam0"
    assert f.image is img


def test_frame_is_frozen():
    """Frame must be immutable -- attribute assignment raises."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    f = Frame(frame_id=0, timestamp=0.0, source_id="s", image=img)
    try:
        f.frame_id = 99  # type: ignore[misc]
        raise AssertionError("Frame should be frozen")
    except AttributeError:
        pass


def test_frame_repr_compact():
    """Frame __repr__ summarises image by shape/dtype, not pixel values."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    f = Frame(frame_id=1, timestamp=0.0, source_id="cam", image=img)
    r = repr(f)
    assert "Frame(" in r
    assert "shape=(480, 640, 3)" in r
    assert "dtype=uint8" in r
    # Should not dump raw pixel data
    assert "[0" not in r


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detection_creation():
    """Detection stores bbox, score, class_id, class_name."""
    bbox = BBox(x=0.5, y=0.5, w=0.2, h=0.4)
    d = Detection(bbox=bbox, score=0.92, class_id=0, class_name="person")
    assert d.bbox is bbox
    assert d.score == 0.92
    assert d.class_id == 0
    assert d.class_name == "person"


def test_detection_is_frozen():
    """Detection must be immutable."""
    d = Detection(bbox=BBox(0.5, 0.5, 0.1, 0.1), score=0.5,
                  class_id=0, class_name="x")
    try:
        d.score = 0.0  # type: ignore[misc]
        raise AssertionError("Detection should be frozen")
    except AttributeError:
        pass


def test_detection_has_complete_type_hints():
    """All Detection fields must have type annotations."""
    hints = get_type_hints(Detection)
    for field_name in ("bbox", "score", "class_id", "class_name"):
        assert field_name in hints, f"Missing annotation: {field_name}"


def test_detection_repr():
    """Detection __repr__ includes score with 4 decimal places."""
    d = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.92,
                  class_id=0, class_name="person")
    r = repr(d)
    assert "Detection(" in r
    assert "0.9200" in r
    assert "'person'" in r


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

def test_track_creation_with_defaults():
    """Track with only required fields uses sensible defaults."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    t = Track(track_id=1, detection=det)
    assert t.track_id == 1
    assert t.detection is det
    assert t.velocity == (0.0, 0.0)
    assert t.age == 0
    assert t.lost_frames == 0


def test_track_creation_full():
    """Track accepts all fields explicitly."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    t = Track(track_id=5, detection=det, velocity=(0.1, -0.05),
              age=10, lost_frames=2)
    assert t.track_id == 5
    assert t.velocity == (0.1, -0.05)
    assert t.age == 10
    assert t.lost_frames == 2


def test_track_is_mutable():
    """Track must be mutable (unlike Detection)."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    t = Track(track_id=1, detection=det)
    t.age = 5
    t.lost_frames = 3
    t.velocity = (0.2, 0.1)
    assert t.age == 5
    assert t.lost_frames == 3
    assert t.velocity == (0.2, 0.1)


def test_track_repr():
    """Track __repr__ includes key state fields."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    t = Track(track_id=7, detection=det, velocity=(0.1, 0.2),
              age=3, lost_frames=1)
    r = repr(t)
    assert "Track(" in r
    assert "track_id=7" in r
    assert "age=3" in r
    assert "lost_frames=1" in r


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def test_target_creation_with_defaults():
    """Target with only required fields defaults to ACTIVE state."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    track = Track(track_id=1, detection=det)
    tg = Target(target_id="T-001", track=track)
    assert tg.target_id == "T-001"
    assert tg.track is track
    assert tg.priority == 0
    assert tg.state == TargetState.ACTIVE
    assert tg.attributes == {}
    assert tg.last_seen == 0.0


def test_target_creation_full():
    """Target accepts all fields including custom attributes."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    track = Track(track_id=1, detection=det)
    tg = Target(
        target_id="T-002",
        track=track,
        priority=10,
        state=TargetState.LOCKED,
        attributes={"identity": "alice", "threat": "low"},
        last_seen=12.5,
    )
    assert tg.target_id == "T-002"
    assert tg.priority == 10
    assert tg.state == TargetState.LOCKED
    assert tg.attributes["identity"] == "alice"
    assert tg.last_seen == 12.5


def test_target_track_can_be_none():
    """Target must accept track=None for LOST / REMOVED states."""
    tg = Target(target_id="T-003", track=None,
                state=TargetState.REMOVED)
    assert tg.track is None
    assert tg.state == TargetState.REMOVED


def test_target_is_mutable():
    """Target must be mutable."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    track = Track(track_id=1, detection=det)
    tg = Target(target_id="T-001", track=track)
    tg.state = TargetState.LOST
    tg.priority = 5
    tg.attributes["note"] = "test"
    assert tg.state == TargetState.LOST
    assert tg.priority == 5
    assert tg.attributes["note"] == "test"


def test_target_repr():
    """Target __repr__ includes state name and track id."""
    det = Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9,
                    class_id=0, class_name="person")
    track = Track(track_id=9, detection=det)
    tg = Target(target_id="T-009", track=track, state=TargetState.LOCKED)
    r = repr(tg)
    assert "Target(" in r
    assert "state=LOCKED" in r
    assert "id=9" in r  # track id summary


# ---------------------------------------------------------------------------
# TargetState enum
# ---------------------------------------------------------------------------

def test_target_state_has_five_members():
    """TargetState must define exactly the five lifecycle states."""
    names = [s.name for s in TargetState]
    assert names == ["ACTIVE", "LOST", "LOCKED", "RECOVERED", "REMOVED"]


def test_target_states_are_distinct():
    """Each TargetState member must have a unique value."""
    values = [s.value for s in TargetState]
    assert len(values) == len(set(values)), "Duplicate enum values"


def test_target_state_recovered_is_not_removed():
    """RECOVERED and REMOVED are different states."""
    assert TargetState.RECOVERED != TargetState.REMOVED
    assert TargetState.ACTIVE != TargetState.LOST


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

def test_event_creation_with_payload():
    """Event stores event_type, timestamp, and payload."""
    e = Event("target.lost", 12.5, {"target_id": "T-001"})
    assert e.event_type == "target.lost"
    assert e.timestamp == 12.5
    assert e.payload["target_id"] == "T-001"


def test_event_creation_default_payload():
    """Event without payload defaults to an empty dict."""
    e = Event("frame.dropped", 3.0)
    assert e.event_type == "frame.dropped"
    assert e.timestamp == 3.0
    assert e.payload == {}


def test_event_default_payload_is_unique():
    """Each Event with default payload must get its own dict (no shared aliasing)."""
    e1 = Event("a", 0.0)
    e2 = Event("b", 1.0)
    e1.payload["x"] = 1
    assert "x" not in e2.payload, "Default dict is shared between instances"


def test_event_is_frozen():
    """Event must be immutable."""
    e = Event("test", 0.0, {"k": "v"})
    try:
        e.event_type = "other"  # type: ignore[misc]
        raise AssertionError("Event should be frozen")
    except AttributeError:
        pass


def test_event_repr():
    """Event __repr__ summarises payload by keys."""
    e = Event("target.lost", 12.5, {"target_id": "T-001", "reason": "occlusion"})
    r = repr(e)
    assert "Event(" in r
    assert "'target.lost'" in r
    assert "'target_id'" in r
    assert "'reason'" in r


# ---------------------------------------------------------------------------
# Slots verification -- all dataclasses should use __slots__
# ---------------------------------------------------------------------------

def test_all_dataclasses_use_slots():
    """Frame, Detection, Track, Target, Event must all have __slots__."""
    for cls in (Frame, BBox, Detection, Track, Target, Event):
        assert "__slots__" in cls.__dict__, f"{cls.__name__} missing __slots__"
        assert "__dict__" not in cls.__dict__, f"{cls.__name__} has __dict__"


# ---------------------------------------------------------------------------
# Dataclass field count verification
# ---------------------------------------------------------------------------

def test_frame_has_four_fields():
    """Frame must have exactly: frame_id, timestamp, source_id, image."""
    fields = {f.name for f in dataclasses.fields(Frame)}
    assert fields == {"frame_id", "timestamp", "source_id", "image"}


def test_detection_has_four_fields():
    """Detection must have exactly: bbox, score, class_id, class_name."""
    fields = {f.name for f in dataclasses.fields(Detection)}
    assert fields == {"bbox", "score", "class_id", "class_name"}


def test_track_has_five_fields():
    """Track must have exactly: track_id, detection, velocity, age, lost_frames."""
    fields = {f.name for f in dataclasses.fields(Track)}
    assert fields == {"track_id", "detection", "velocity", "age", "lost_frames"}


def test_target_has_seven_fields():
    """Target must have exactly: target_id, track, slot_id, priority, state, attributes, last_seen."""
    fields = {f.name for f in dataclasses.fields(Target)}
    assert fields == {"target_id", "track", "slot_id", "priority", "state",
                      "attributes", "last_seen"}


def test_event_has_three_fields():
    """Event must have exactly: event_type, timestamp, payload."""
    fields = {f.name for f in dataclasses.fields(Event)}
    assert fields == {"event_type", "timestamp", "payload"}


# ---------------------------------------------------------------------------
# Main entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run all test_* functions in this module
    import inspect
    import sys

    failures: list[str] = []
    passed = 0
    for name, obj in sorted(inspect.getmembers(sys.modules[__name__])):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  [PASS] {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  [FAIL] {name}: {exc}")
                failures.append(name)
            except Exception as exc:
                print(f"  [ERR ] {name}: {type(exc).__name__}: {exc}")
                failures.append(name)

    print()
    total = passed + len(failures)
    print(f"=== {passed}/{total} passed, {len(failures)} failed ===")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)

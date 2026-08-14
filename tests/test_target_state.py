"""Unit tests for visioncore.state.TargetState.

Verifies construction (with and without metadata), frozen-immutability,
serialisation (``to_dict``), deserialisation (``from_dict``), and the
``copy_with`` derivation helper. Also covers edge cases: unknown keys in
``from_dict`` are ignored, ``copy_with`` rejects unknown field names,
metadata is defensively copied (caller mutation does not leak in),
and the two ``TargetState`` types (this snapshot dataclass vs. the
lifecycle enum in ``visioncore.core``) coexist without import collision.

Run::

    python -m pytest tests/test_target_state.py -v
    # or
    python tests/test_target_state.py
"""

from __future__ import annotations

import dataclasses

from visioncore.state import TargetState
from visioncore.state.target_state import TargetState as TargetStateDirect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> TargetState:
    """Build a TargetState with sensible defaults + per-test overrides."""
    defaults: dict = dict(
        target_id=42,
        local_id=1,
        global_id=None,
        label="person",
        confidence=0.92,
        cx=0.5,
        cy=0.4,
        vx=0.01,
        vy=-0.002,
        width=0.12,
        height=0.30,
        timestamp=12.5,
        camera_id=0,
        metadata={},
    )
    defaults.update(overrides)
    return TargetState(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_all_fields():
    """A TargetState stores every field with the value it was given."""
    ts = _make_state()
    assert ts.target_id == 42
    assert ts.local_id == 1
    assert ts.global_id is None
    assert ts.label == "person"
    assert ts.confidence == 0.92
    assert ts.cx == 0.5
    assert ts.cy == 0.4
    assert ts.vx == 0.01
    assert ts.vy == -0.002
    assert ts.width == 0.12
    assert ts.height == 0.30
    assert ts.timestamp == 12.5
    assert ts.camera_id == 0
    assert ts.metadata == {}


def test_construction_default_metadata_empty_dict():
    """Omitting metadata yields a fresh empty dict (not shared)."""
    ts = TargetState(
        target_id=1, local_id=1, global_id=None,
        label="person", confidence=0.5,
        cx=0.0, cy=0.0, vx=0.0, vy=0.0,
        width=0.1, height=0.1, timestamp=0.0, camera_id=0,
    )
    assert ts.metadata == {}
    assert isinstance(ts.metadata, dict)


def test_construction_default_metadata_independent_instances():
    """Two instances built without metadata get distinct empty dicts."""
    a = TargetState(
        target_id=1, local_id=1, global_id=None,
        label="x", confidence=0.5,
        cx=0.0, cy=0.0, vx=0.0, vy=0.0,
        width=0.1, height=0.1, timestamp=0.0, camera_id=0,
    )
    b = TargetState(
        target_id=2, local_id=2, global_id=None,
        label="y", confidence=0.5,
        cx=0.0, cy=0.0, vx=0.0, vy=0.0,
        width=0.1, height=0.1, timestamp=0.0, camera_id=0,
    )
    assert a.metadata is not b.metadata


def test_construction_with_metadata():
    """Metadata passed at construction is stored verbatim."""
    md = {"source": "yolo26", "roi": "entry"}
    ts = _make_state(metadata=md)
    assert ts.metadata is md  # construction stores the ref; from_dict copies
    assert ts.metadata["source"] == "yolo26"


def test_construction_none_optional_fields():
    """local_id, global_id, camera_id all accept None."""
    ts = TargetState(
        target_id=7, local_id=None, global_id=None,
        label="head", confidence=0.8,
        cx=0.2, cy=0.3, vx=0.0, vy=0.0,
        width=0.05, height=0.05, timestamp=1.0, camera_id=None,
    )
    assert ts.local_id is None
    assert ts.global_id is None
    assert ts.camera_id is None


def test_construction_default_rect_health_none():
    """rect and health (D5 extension fields) default to None until a stage runs."""
    ts = _make_state()
    assert ts.rect is None
    assert ts.health is None


def test_rect_and_health_round_trip():
    """rect/health survive to_dict -> from_dict and copy_with (tuple + float)."""
    ts = _make_state(rect=(0.1, 0.2, 0.3, 0.4), health=1.0)
    ts2 = TargetState.from_dict(ts.to_dict())
    assert isinstance(ts2.rect, tuple)
    assert ts2.rect == (0.1, 0.2, 0.3, 0.4)
    assert ts2.health == 1.0
    moved = ts.copy_with(rect=(0.2, 0.3, 0.4, 0.5), health=0.5)
    assert moved.rect == (0.2, 0.3, 0.4, 0.5)
    assert moved.health == 0.5
    # Original untouched.
    assert ts.rect == (0.1, 0.2, 0.3, 0.4)
    assert ts.health == 1.0


def test_package_and_direct_import_are_same():
    """visioncore.state.TargetState == visioncore.state.target_state.TargetState."""
    assert TargetState is TargetStateDirect


def test_dataclass_is_frozen_and_slotted():
    """The dataclass decorator configuration is correct."""
    # frozen=True raises on attribute assignment; slots=True removes __dict__.
    fields = dataclasses.fields(TargetState)
    assert len(fields) == 18, f"expected 18 fields, got {len(fields)}"
    # __slots__ presence check: instances must not have a __dict__.
    ts = _make_state()
    assert not hasattr(ts, "__dict__"), "slots=True should remove __dict__"


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------

def test_frozen_attribute_assignment_raises():
    """Direct assignment to any field raises FrozenInstanceError."""
    ts = _make_state()
    try:
        ts.target_id = 99  # type: ignore[misc]
        raise AssertionError("TargetState should be frozen (target_id)")
    except dataclasses.FrozenInstanceError:
        pass


def test_frozen_kinematic_field_assignment_raises():
    """Assignment to cx (a kinematic field) also raises."""
    ts = _make_state()
    try:
        ts.cx = 0.99  # type: ignore[misc]
        raise AssertionError("TargetState should be frozen (cx)")
    except dataclasses.FrozenInstanceError:
        pass


def test_frozen_metadata_assignment_raises():
    """Replacing the metadata attribute raises (dict itself is mutable, by design)."""
    ts = _make_state(metadata={"k": "v"})
    try:
        ts.metadata = {}  # type: ignore[misc]
        raise AssertionError("TargetState should be frozen (metadata attr)")
    except dataclasses.FrozenInstanceError:
        pass


def test_hash_method_present_but_dict_field_blocks_hashing():
    """frozen=True provides __hash__, but the dict field makes hashing raise.

    This documents the standard dataclass behaviour: a frozen dataclass
    gets a generated ``__hash__`` (so the attribute exists and the type
    is nominally hashable), but if any field holds an unhashable value
    -- and ``dict`` is always unhashable, empty or not -- calling
    ``hash()`` raises ``TypeError`` at runtime. Callers that need to use
    TargetState as a dict key or set member must omit metadata or
    convert it to a frozenset/tuple first.
    """
    ts = _make_state(metadata={})
    # The hash method is present (frozen=True provides it).
    assert hasattr(type(ts), "__hash__")
    assert type(ts).__hash__ is not None
    # But actually hashing raises because dict is unhashable.
    try:
        hash(ts)
        raise AssertionError("hash() should raise TypeError on a dict-bearing dataclass")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# to_dict (serialisation)
# ---------------------------------------------------------------------------

def test_to_dict_contains_all_fields():
    """to_dict produces a dict with one entry per declared field."""
    ts = _make_state(metadata={"k": "v"})
    d = ts.to_dict()
    expected_keys = {
        "target_id", "local_id", "global_id", "label", "confidence",
        "cx", "cy", "vx", "vy", "width", "height", "timestamp",
        "camera_id", "metadata", "attributes", "gesture",
        "rect", "health",
    }
    assert set(d.keys()) == expected_keys


def test_to_dict_values_match():
    """to_dict values equal the corresponding attributes."""
    ts = _make_state(metadata={"source": "yolo26"})
    d = ts.to_dict()
    assert d["target_id"] == 42
    assert d["local_id"] == 1
    assert d["global_id"] is None
    assert d["label"] == "person"
    assert d["confidence"] == 0.92
    assert d["cx"] == 0.5
    assert d["cy"] == 0.4
    assert d["vx"] == 0.01
    assert d["vy"] == -0.002
    assert d["width"] == 0.12
    assert d["height"] == 0.30
    assert d["timestamp"] == 12.5
    assert d["camera_id"] == 0
    assert d["metadata"] == {"source": "yolo26"}
    assert d["rect"] is None
    assert d["health"] is None


def test_to_dict_metadata_is_deep_copy():
    """to_dict returns an independent copy of metadata (caller cannot mutate the snapshot)."""
    nested = {"a": [1, 2, 3]}
    ts = _make_state(metadata=nested)
    d = ts.to_dict()
    # Mutate the returned dict's metadata deeply.
    d["metadata"]["a"].append(99)
    # Original snapshot's metadata must be unaffected.
    assert ts.metadata["a"] == [1, 2, 3]


def test_to_dict_does_not_mutate_snapshot():
    """Calling to_dict has no side effect on the snapshot."""
    ts = _make_state(metadata={"k": "v"})
    _ = ts.to_dict()
    _ = ts.to_dict()
    assert ts.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# from_dict (deserialisation)
# ---------------------------------------------------------------------------

def test_from_dict_round_trip():
    """from_dict(to_dict(x)) reproduces x field-for-field."""
    ts = _make_state(metadata={"source": "yolo26", "n": 7})
    d = ts.to_dict()
    ts2 = TargetState.from_dict(d)
    assert ts2 == ts


def test_from_dict_ignores_unknown_keys():
    """Unknown keys in the input dict are silently dropped, not raised."""
    d = {
        "target_id": 1, "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
        "metadata": {},
        "unknown_field_a": "ignored",
        "unknown_field_b": 123,
    }
    ts = TargetState.from_dict(d)
    assert ts.target_id == 1
    assert ts.label == "person"


def test_from_dict_missing_required_field_raises():
    """Missing a required field propagates TypeError from the constructor."""
    d = {
        "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
        # target_id omitted
    }
    try:
        TargetState.from_dict(d)
        raise AssertionError("from_dict should raise TypeError on missing field")
    except TypeError:
        pass


def test_from_dict_metadata_defensively_copied():
    """from_dict copies metadata so the caller's dict cannot leak in."""
    md = {"k": "v"}
    d = {
        "target_id": 1, "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
        "metadata": md,
    }
    ts = TargetState.from_dict(d)
    # Caller mutates their original dict after construction.
    md["k"] = "MUTATED"
    # Snapshot's metadata must be unaffected.
    assert ts.metadata == {"k": "v"}


def test_from_dict_without_metadata_uses_default():
    """from_dict with no metadata key yields an empty dict."""
    d = {
        "target_id": 1, "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
    }
    ts = TargetState.from_dict(d)
    assert ts.metadata == {}


def test_from_dict_preserves_none_optional_fields():
    """None values for local_id/global_id/camera_id round-trip correctly."""
    d = {
        "target_id": 7, "local_id": None, "global_id": None,
        "label": "head", "confidence": 0.8,
        "cx": 0.2, "cy": 0.3, "vx": 0.0, "vy": 0.0,
        "width": 0.05, "height": 0.05, "timestamp": 1.0, "camera_id": None,
    }
    ts = TargetState.from_dict(d)
    assert ts.local_id is None
    assert ts.global_id is None
    assert ts.camera_id is None


def test_from_dict_reject_none_metadata():
    """from_dict rejects metadata=None with TypeError (B1.1 contract tightening).

    Previously, passing ``metadata=None`` was silently accepted and the
    snapshot's metadata field became ``None``, violating the ``dict[str, Any]``
    type annotation and breaking downstream callers that invoke
    ``.keys()`` / ``.items()``. The B1.1 hotfix makes this fail fast.
    """
    d = {
        "target_id": 1, "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
        "metadata": None,
    }
    try:
        TargetState.from_dict(d)
        raise AssertionError(
            "from_dict should raise TypeError when metadata is None"
        )
    except TypeError as exc:
        # Error message must mention 'metadata' and 'None' for debuggability.
        msg = str(exc)
        assert "metadata" in msg
        assert "None" in msg


def test_from_dict_reject_non_dict_metadata():
    """from_dict rejects non-dict metadata (str, list, int, tuple) with TypeError.

    Any metadata value that is not a ``dict`` and not absent is rejected.
    This catches producer bugs where metadata is serialised as a JSON string
    or array by mistake, instead of being decoded into a dict first.
    """
    base = {
        "target_id": 1, "local_id": 1, "global_id": None,
        "label": "person", "confidence": 0.5,
        "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
        "width": 0.1, "height": 0.2, "timestamp": 0.0, "camera_id": 0,
    }
    bad_values = [
        "not a dict",          # str
        ["a", "b"],            # list
        42,                    # int
        3.14,                  # float
        ("a", "b"),            # tuple
        {"a", "b"},            # set
        object(),              # arbitrary object
    ]
    for bad in bad_values:
        d = dict(base, metadata=bad)
        try:
            TargetState.from_dict(d)
            raise AssertionError(
                f"from_dict should raise TypeError for metadata={bad!r} "
                f"(type {type(bad).__name__})"
            )
        except TypeError as exc:
            msg = str(exc)
            assert "metadata" in msg
            assert "dict" in msg
            # Error message should name the actual type received.
            assert type(bad).__name__ in msg


# ---------------------------------------------------------------------------
# copy_with (derivation)
# ---------------------------------------------------------------------------

def test_copy_with_overrides_single_field():
    """copy_with replaces the named field and preserves the rest."""
    ts = _make_state()
    moved = ts.copy_with(cx=0.55)
    assert moved.cx == 0.55
    # Untouched fields carry over.
    assert moved.target_id == ts.target_id
    assert moved.label == ts.label
    assert moved.cy == ts.cy
    assert moved.timestamp == ts.timestamp


def test_copy_with_overrides_multiple_fields():
    """copy_with accepts multiple overrides at once."""
    ts = _make_state()
    moved = ts.copy_with(cx=0.6, cy=0.7, timestamp=13.0, confidence=0.85)
    assert moved.cx == 0.6
    assert moved.cy == 0.7
    assert moved.timestamp == 13.0
    assert moved.confidence == 0.85
    # Untouched.
    assert moved.target_id == ts.target_id
    assert moved.label == ts.label


def test_copy_with_does_not_mutate_original():
    """copy_with leaves the source snapshot untouched."""
    ts = _make_state(cx=0.5)
    _ = ts.copy_with(cx=0.99)
    assert ts.cx == 0.5


def test_copy_with_unknown_field_raises():
    """Unknown field names in overrides raise TypeError (typo guard)."""
    ts = _make_state()
    try:
        ts.copy_with(targe_id=99)  # typo: targe_id
        raise AssertionError("copy_with should reject unknown field 'targe_id'")
    except TypeError as exc:
        assert "targe_id" in str(exc)


def test_copy_with_metadata_independent_when_not_overridden():
    """copy_with shallow-copies metadata so derived snapshot is independent."""
    ts = _make_state(metadata={"k": "v"})
    moved = ts.copy_with(cx=0.6)
    # The derived snapshot's metadata is a different dict object.
    assert moved.metadata is not ts.metadata
    # But contents match (shallow copy).
    assert moved.metadata == ts.metadata


def test_copy_with_metadata_override_replaces_whole_dict():
    """When metadata is overridden, the new dict replaces the old entirely."""
    ts = _make_state(metadata={"old": 1})
    moved = ts.copy_with(metadata={"new": 2})
    assert moved.metadata == {"new": 2}
    # Original unaffected.
    assert ts.metadata == {"old": 1}


def test_copy_with_returns_same_type():
    """copy_with returns a TargetState, not a generic object."""
    ts = _make_state()
    moved = ts.copy_with(cx=0.6)
    assert isinstance(moved, TargetState)


def test_copy_with_no_overrides_returns_equal_snapshot():
    """copy_with() with no overrides yields an equal but distinct snapshot."""
    ts = _make_state(metadata={"k": "v"})
    twin = ts.copy_with()
    assert twin == ts
    assert twin is not ts
    # Metadata is independent.
    assert twin.metadata is not ts.metadata


def test_copy_with_chain():
    """Chained copy_with calls accumulate overrides without leaking state.

    Each copy_with returns a fresh frozen snapshot, so chaining is safe:
    intermediate snapshots are immutable and cannot be corrupted by later
    links. This test simulates a 4-step kinematic propagation (cx/cy
    advanced by velocity*dt, timestamp advanced by dt, confidence decay)
    and verifies the final state matches expectations while every
    intermediate snapshot remains pristine.
    """
    import math
    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

    ts0 = _make_state(cx=0.5, cy=0.5, vx=0.1, vy=-0.05,
                      timestamp=10.0, confidence=0.9, metadata={"src": "yolo26"})
    dt = 0.1

    # Step 1: advance x by vx*dt
    ts1 = ts0.copy_with(cx=ts0.cx + ts0.vx * dt)
    # Step 2: advance y by vy*dt
    ts2 = ts1.copy_with(cy=ts1.cy + ts1.vy * dt)
    # Step 3: advance timestamp
    ts3 = ts2.copy_with(timestamp=ts2.timestamp + dt)
    # Step 4: decay confidence
    ts4 = ts3.copy_with(confidence=ts3.confidence * 0.95)

    # Final state has all four overrides applied.
    assert close(ts4.cx, 0.5 + 0.1 * 0.1)           # 0.51
    assert close(ts4.cy, 0.5 + (-0.05) * 0.1)       # 0.495
    assert close(ts4.timestamp, 10.0 + 0.1)         # 10.1
    assert close(ts4.confidence, 0.9 * 0.95)        # 0.855

    # Every intermediate snapshot is untouched by later links.
    assert ts0.cx == 0.5 and ts0.cy == 0.5 and ts0.timestamp == 10.0
    assert close(ts1.cx, 0.51) and ts1.cy == 0.5 and ts1.timestamp == 10.0
    assert close(ts2.cx, 0.51) and close(ts2.cy, 0.495) and ts2.timestamp == 10.0
    assert close(ts3.cx, 0.51) and close(ts3.cy, 0.495) and close(ts3.timestamp, 10.1)

    # Metadata is propagated through the chain and remains independent
    # at every link (copy_with shallow-copies metadata when not overridden).
    assert ts4.metadata == {"src": "yolo26"}
    assert ts4.metadata is not ts3.metadata
    assert ts3.metadata is not ts2.metadata
    assert ts2.metadata is not ts1.metadata
    assert ts1.metadata is not ts0.metadata

    # Mutating the final snapshot's metadata does NOT propagate back up
    # the chain -- this is the core guarantee that makes chaining safe.
    ts4.metadata["new_key"] = "leak?"
    assert "new_key" not in ts3.metadata
    assert "new_key" not in ts0.metadata


# ---------------------------------------------------------------------------
# Equality semantics (dataclass-generated __eq__)
# ---------------------------------------------------------------------------

def test_dataclass_equality():
    """Two TargetState instances are == iff all fields are equal.

    frozen dataclasses get an auto-generated __eq__ that compares the
    tuple of field values. This test pins down the semantics so downstream
    code can rely on ``==`` for snapshot comparison, deduplication, and
    set/dict membership (note: dict membership also requires __hash__,
    which is blocked by the dict field -- see
    test_hash_method_present_but_dict_field_blocks_hashing).
    """
    # Two instances built with identical arguments are equal.
    a = _make_state(metadata={"k": "v"})
    b = _make_state(metadata={"k": "v"})
    assert a == b
    assert not (a != b)
    # Equal but distinct objects (value equality, not identity).
    assert a is not b

    # Changing any single field breaks equality.
    fields_to_vary = [
        ("target_id",   999),
        ("local_id",    999),
        ("global_id",   999),
        ("label",       "vehicle"),
        ("confidence",  0.01),
        ("cx",          0.99),
        ("cy",          0.99),
        ("vx",          0.99),
        ("vy",          0.99),
        ("width",       0.99),
        ("height",      0.99),
        ("timestamp",   999.0),
        ("camera_id",   999),
    ]
    for field_name, new_val in fields_to_vary:
        c = a.copy_with(**{field_name: new_val})
        assert a != c, f"Equality should break when {field_name} changes"

    # Metadata contents matter, not identity.
    d1 = _make_state(metadata={"k": "v"})
    d2 = _make_state(metadata={"k": "v"})
    assert d1.metadata is not d2.metadata  # distinct dicts
    assert d1 == d2                         # but equal contents -> equal snapshots

    # Different metadata contents -> not equal.
    e1 = _make_state(metadata={"k": "v1"})
    e2 = _make_state(metadata={"k": "v2"})
    assert e1 != e2

    # Empty vs non-empty metadata -> not equal.
    f1 = _make_state(metadata={})
    f2 = _make_state(metadata={"k": "v"})
    assert f1 != f2

    # None optional fields compare equal across instances.
    g1 = _make_state(local_id=None, global_id=None, camera_id=None)
    g2 = _make_state(local_id=None, global_id=None, camera_id=None)
    assert g1 == g2

    # None vs non-None optional field -> not equal.
    h1 = _make_state(local_id=None)
    h2 = _make_state(local_id=1)
    assert h1 != h2

    # Equality is reflexive and symmetric (sanity).
    assert a == a
    assert (a == b) == (b == a)


# ---------------------------------------------------------------------------
# Coexistence with visioncore.core.TargetState (enum)
# ---------------------------------------------------------------------------

def test_coexistence_with_core_target_state_enum():
    """The snapshot dataclass and the lifecycle enum coexist without collision."""
    from visioncore.core import TargetState as TargetLifecycleEnum
    from visioncore.state import TargetState as TargetStateSnapshot

    assert TargetLifecycleEnum is not TargetStateSnapshot
    # The enum has ACTIVE member; the snapshot does not.
    assert hasattr(TargetLifecycleEnum, "ACTIVE")
    assert not hasattr(TargetStateSnapshot, "ACTIVE")
    # The snapshot is a dataclass; the enum is not.
    assert dataclasses.is_dataclass(TargetStateSnapshot)
    assert not dataclasses.is_dataclass(TargetLifecycleEnum)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_is_compact_and_includes_keys():
    """__repr__ includes the class name, key fields, and metadata keys (not values)."""
    ts = _make_state(metadata={"alpha": 1, "beta": 2})
    r = repr(ts)
    assert r.startswith("TargetState(")
    assert "target_id=42" in r
    assert "label='person'" in r
    assert "metadata_keys={'alpha', 'beta'}" in r or "metadata_keys={" in r
    # Should not dump metadata values.
    assert "alpha=1" not in r


def test_repr_handles_empty_metadata():
    """__repr__ renders empty metadata as {}."""
    ts = _make_state(metadata={})
    r = repr(ts)
    assert "metadata_keys={}" in r


# ---------------------------------------------------------------------------
# Module entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

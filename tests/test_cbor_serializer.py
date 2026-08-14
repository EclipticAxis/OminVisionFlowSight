"""Unit tests for visioncore.protocol.serialization.CBORSerializer.

Verifies roundtrip (multiple variants), invalid CBOR, missing field,
type enforcement, bytes/bytearray/memoryview input acceptance, and the
mandatory type-annotation convention. Mirrors the JSON serializer test
structure for easy comparison.

Run::

    python -m pytest tests/test_cbor_serializer.py -v
    # or
    PYTHONPATH=F:/VisionBata python tests/test_cbor_serializer.py
"""

from __future__ import annotations

import ast
import inspect

import cbor2

from visioncore.protocol.serialization import CBORSerializer
from visioncore.protocol.serialization.cbor_serializer import (
    CBORSerializer as CBORSerializerDirect,
)
from visioncore.state.target_state import TargetState


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
# Type annotation convention (mandatory per B2 spec, extended to B4)
# ---------------------------------------------------------------------------

def test_type_annotation_convention_no_top_level_import():
    """The CBOR serializer must not use 'from visioncore import TargetState'.

    Uses AST parsing to check actual ImportFrom nodes -- docstrings that
    mention the forbidden pattern are not flagged.
    """
    mod = __import__(CBORSerializer.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "visioncore":
            for alias in node.names:
                assert alias.name != "TargetState", (
                    f"{CBORSerializer.__module__} line {node.lineno}: "
                    f"forbidden 'from visioncore import TargetState'."
                )


def test_cbor_serializer_uses_snapshot_not_enum():
    """serialize accepts the snapshot dataclass, rejects the lifecycle enum."""
    from visioncore.core import TargetState as TargetLifecycleEnum

    s = CBORSerializer()
    # Snapshot is accepted.
    data = s.serialize(_make_state())
    assert isinstance(data, bytes)

    # Enum is rejected.
    try:
        s.serialize(TargetLifecycleEnum.ACTIVE)  # type: ignore[arg-type]
        raise AssertionError("serialize should reject the enum")
    except TypeError:
        pass


def test_no_forbidden_transport_imports():
    """CBOR serializer must not import socket/udp/ros2/mavlink."""
    mod = __import__(CBORSerializer.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    forbidden_modules = {"socket", "udp", "ros2", "ros", "mavlink", "mav"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"{CBORSerializer.__module__}: forbidden import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_modules, (
                f"{CBORSerializer.__module__}: forbidden from-import {node.module}"
            )


# ---------------------------------------------------------------------------
# Package / import sanity
# ---------------------------------------------------------------------------

def test_direct_and_package_imports_are_same():
    """visioncore.protocol.serialization.CBORSerializer == direct import."""
    assert CBORSerializer is CBORSerializerDirect


def test_package_exports_cbor_serializer():
    """visioncore.protocol.serialization.__all__ contains CBORSerializer."""
    import visioncore.protocol.serialization as ser
    assert "CBORSerializer" in ser.__all__


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_basic():
    """serialize -> deserialize yields an equal TargetState."""
    s = CBORSerializer()
    ts = _make_state()
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2


def test_roundtrip_returns_bytes():
    """serialize returns bytes, not str."""
    s = CBORSerializer()
    data = s.serialize(_make_state())
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_roundtrip_with_metadata():
    """Roundtrip preserves populated metadata."""
    s = CBORSerializer()
    ts = _make_state(metadata={"src": "yolo26", "roi": "entry", "n": 7})
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2
    assert ts2.metadata == {"src": "yolo26", "roi": "entry", "n": 7}


def test_roundtrip_with_nested_metadata():
    """Roundtrip preserves nested dict / list structures in metadata."""
    s = CBORSerializer()
    ts = _make_state(metadata={
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "tags": ["a", "b", "c"],
        "nested": {"deep": {"value": 42}},
    })
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2


def test_roundtrip_with_none_optionals():
    """Roundtrip preserves None for local_id / global_id / camera_id."""
    s = CBORSerializer()
    ts = _make_state(local_id=None, global_id=None, camera_id=None)
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2
    assert ts2.local_id is None
    assert ts2.global_id is None
    assert ts2.camera_id is None


def test_roundtrip_empty_metadata():
    """Roundtrip works with empty metadata."""
    s = CBORSerializer()
    ts = _make_state(metadata={})
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2
    assert ts2.metadata == {}


def test_roundtrip_extreme_floats():
    """Roundtrip preserves extreme float values."""
    s = CBORSerializer()
    ts = _make_state(
        cx=0.999999, cy=-0.000001, vx=123.456, vy=-987.654,
        timestamp=1e10, confidence=0.0, width=1.0, height=0.0,
    )
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2


def test_roundtrip_negative_and_zero_ids():
    """Roundtrip preserves zero/negative IDs."""
    s = CBORSerializer()
    ts = _make_state(target_id=0, local_id=-1, camera_id=-99)
    data = s.serialize(ts)
    ts2 = s.deserialize(data)
    assert ts == ts2


def test_roundtrip_unknown_keys_ignored():
    """deserialize tolerates extra unknown keys (forward-compat)."""
    s = CBORSerializer()
    ts = _make_state()
    data = s.serialize(ts)
    # Inject unknown keys via raw cbor2.
    d = cbor2.loads(data)
    d["unknown_future_field"] = "ignored"
    data2 = cbor2.dumps(d)
    ts2 = s.deserialize(data2)
    assert ts == ts2


def test_roundtrip_cbor_bytes_decode_with_cbor2_module():
    """serialize output is parseable by the cbor2 module directly."""
    s = CBORSerializer()
    data = s.serialize(_make_state(metadata={"k": "v"}))
    d = cbor2.loads(data)
    assert isinstance(d, dict)
    assert d["target_id"] == 42
    assert d["label"] == "person"


# ---------------------------------------------------------------------------
# Invalid CBOR
# ---------------------------------------------------------------------------

def test_invalid_cbor_raises_value_error():
    """Malformed bytes raise ValueError, not cbor2.CBORDecodeError."""
    s = CBORSerializer()
    bad_inputs = [
        b"\xff\xff\xff\xff invalid cbor",
        b"",
        b"\x00",
        b"not cbor at all",
        b"\x9f",  # unterminated indefinite-length array
        b"\xbf",  # unterminated indefinite-length map
    ]
    for bad in bad_inputs:
        try:
            s.deserialize(bad)
            raise AssertionError(
                f"deserialize should raise ValueError for {bad!r}"
            )
        except ValueError:
            pass


def test_invalid_cbor_non_dict_raises_value_error():
    """Valid CBOR that is not a dict raises ValueError."""
    s = CBORSerializer()
    non_dict_inputs = [
        cbor2.dumps([1, 2, 3]),       # list
        cbor2.dumps("a string"),      # string
        cbor2.dumps(42),              # int
        cbor2.dumps(3.14),            # float
        cbor2.dumps(True),            # bool
    ]
    for bad in non_dict_inputs:
        try:
            s.deserialize(bad)
            raise AssertionError(
                f"deserialize should raise ValueError for non-dict CBOR"
            )
        except ValueError as exc:
            assert "dict" in str(exc)


def test_invalid_cbor_str_input_rejected():
    """deserialize rejects str input -- CBOR is binary, not text."""
    s = CBORSerializer()
    ts = _make_state()
    data = s.serialize(ts)
    # str is not accepted, even if it contains the right bytes.
    try:
        s.deserialize(data.decode("latin-1"))  # type: ignore[arg-type]
        raise AssertionError("deserialize should reject str input")
    except TypeError as exc:
        assert "bytes" in str(exc) or "bytearray" in str(exc)


def test_invalid_cbor_wrong_type_raises_type_error():
    """deserialize rejects non-binary input with TypeError."""
    s = CBORSerializer()
    bad_inputs = [None, 42, 3.14, [1, 2], {"a": 1}, object()]
    for bad in bad_inputs:
        try:
            s.deserialize(bad)  # type: ignore[arg-type]
            raise AssertionError(
                f"deserialize should reject {bad!r} ({type(bad).__name__})"
            )
        except TypeError:
            pass


def test_invalid_cbor_truncated_raises_value_error():
    """Truncated CBOR (valid prefix, missing tail) raises ValueError."""
    s = CBORSerializer()
    data = s.serialize(_make_state())
    # Truncate to half length.
    truncated = data[:len(data) // 2]
    try:
        s.deserialize(truncated)
        raise AssertionError("deserialize should raise ValueError for truncated CBOR")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Binary buffer type acceptance
# ---------------------------------------------------------------------------

def test_deserialize_accepts_bytearray():
    """deserialize accepts bytearray input."""
    s = CBORSerializer()
    ts = _make_state()
    data = s.serialize(ts)
    ts2 = s.deserialize(bytearray(data))
    assert ts == ts2


def test_deserialize_accepts_memoryview():
    """deserialize accepts memoryview input."""
    s = CBORSerializer()
    ts = _make_state()
    data = s.serialize(ts)
    ts2 = s.deserialize(memoryview(data))
    assert ts == ts2


# ---------------------------------------------------------------------------
# Missing field
# ---------------------------------------------------------------------------

def test_missing_required_field_raises_type_error():
    """deserialize raises TypeError when a required field is missing."""
    s = CBORSerializer()
    full = _make_state().to_dict()

    required_fields = [
        "target_id", "local_id", "global_id", "label", "confidence",
        "cx", "cy", "vx", "vy", "width", "height", "timestamp", "camera_id",
    ]
    for field_name in required_fields:
        d = dict(full)
        del d[field_name]
        data = cbor2.dumps(d)
        try:
            s.deserialize(data)
            raise AssertionError(
                f"deserialize should raise TypeError when {field_name!r} is missing"
            )
        except TypeError:
            pass


def test_missing_metadata_uses_default():
    """Missing metadata key is allowed -- default_factory supplies {}."""
    s = CBORSerializer()
    d = _make_state().to_dict()
    del d["metadata"]
    data = cbor2.dumps(d)
    ts = s.deserialize(data)
    assert ts.metadata == {}


def test_metadata_none_rejected():
    """deserialize rejects metadata=None per the B1.1 contract."""
    s = CBORSerializer()
    d = _make_state().to_dict()
    d["metadata"] = None
    data = cbor2.dumps(d)
    try:
        s.deserialize(data)
        raise AssertionError("deserialize should reject metadata=None")
    except TypeError:
        pass


def test_metadata_non_dict_rejected():
    """deserialize rejects non-dict metadata per the B1.1 contract."""
    s = CBORSerializer()
    for bad_meta in ["string", ["list"], 42, 3.14]:
        d = _make_state().to_dict()
        d["metadata"] = bad_meta
        data = cbor2.dumps(d)
        try:
            s.deserialize(data)
            raise AssertionError(
                f"deserialize should reject metadata={bad_meta!r}"
            )
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# serialize type enforcement
# ---------------------------------------------------------------------------

def test_serialize_rejects_non_target_state():
    """serialize rejects None and non-TargetState values with TypeError."""
    s = CBORSerializer()
    bad_values = [None, "string", 42, 3.14, [1, 2], {"a": 1}, object()]
    for bad in bad_values:
        try:
            s.serialize(bad)  # type: ignore[arg-type]
            raise AssertionError(
                f"serialize should reject {bad!r} ({type(bad).__name__})"
            )
        except TypeError as exc:
            assert "TargetState" in str(exc)


def test_serialize_preserves_state():
    """serialize has no side effect on the input state."""
    s = CBORSerializer()
    ts = _make_state(metadata={"k": "v"})
    _ = s.serialize(ts)
    _ = s.serialize(ts)
    assert ts.metadata == {"k": "v"}


def test_serialize_is_reusable():
    """A single serializer instance can serialize many different states."""
    s = CBORSerializer()
    states = [_make_state(target_id=i) for i in range(10)]
    for ts in states:
        data = s.serialize(ts)
        ts2 = s.deserialize(data)
        assert ts == ts2


# ---------------------------------------------------------------------------
# Cross-format compatibility (JSON <-> CBOR via dict)
# ---------------------------------------------------------------------------

def test_cbor_and_json_produce_equivalent_dicts():
    """CBOR and JSON serialize the same TargetState to equivalent dicts.

    This verifies that switching from JSON to CBOR does not lose or
    alter any data -- the two formats are interchangeable for roundtrip.
    """
    import json
    from visioncore.protocol.serialization import TargetStateSerializer

    json_s = TargetStateSerializer()
    cbor_s = CBORSerializer()

    ts = _make_state(metadata={"src": "yolo26", "n": 42, "nested": {"a": [1, 2]}})

    # Serialize via both formats.
    json_str = json_s.serialize(ts)
    cbor_bytes = cbor_s.serialize(ts)

    # Decode the raw payloads (bypassing TargetState.from_dict) and
    # compare the dicts directly.
    json_dict = json.loads(json_str)
    cbor_dict = cbor2.loads(cbor_bytes)

    assert json_dict == cbor_dict, (
        f"JSON dict != CBOR dict:\n  JSON: {json_dict}\n  CBOR: {cbor_dict}"
    )


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    """__repr__ includes the class name."""
    s = CBORSerializer()
    r = repr(s)
    assert "CBORSerializer" in r


# ---------------------------------------------------------------------------
# Module entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures: list[str] = []
    passed = 0
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in names:
        try:
            g[name]()
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

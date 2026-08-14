"""Unit tests for visioncore.protocol.serialization.TargetStateSerializer.

Verifies the three categories required by the B3 spec -- roundtrip,
invalid JSON, missing field -- plus edge cases: non-dict JSON, bytes
input, non-TargetState rejection, non-JSON-serializable metadata,
formatting options, and the mandatory type-annotation convention.

Run::

    python -m pytest tests/test_json_serializer.py -v
    # or
    PYTHONPATH=F:/VisionBata python tests/test_json_serializer.py
"""

from __future__ import annotations

import ast
import inspect
import json

from visioncore.protocol.serialization import TargetStateSerializer
from visioncore.protocol.serialization.json_serializer import (
    TargetStateSerializer as TargetStateSerializerDirect,
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
# Type annotation convention (mandatory per B2 spec, extended to B3)
# ---------------------------------------------------------------------------

def test_type_annotation_convention_no_top_level_import():
    """The serializer module must not use 'from visioncore import TargetState'.

    Uses AST parsing to check actual ImportFrom nodes -- docstrings that
    mention the forbidden pattern (while explaining why it is forbidden)
    are not flagged.
    """
    mod = __import__(TargetStateSerializer.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "visioncore":
            for alias in node.names:
                assert alias.name != "TargetState", (
                    f"{TargetStateSerializer.__module__} line {node.lineno}: "
                    f"forbidden 'from visioncore import TargetState'. Use "
                    f"'from visioncore.state.target_state import TargetState'."
                )


def test_serializer_uses_snapshot_not_enum():
    """serialize accepts the snapshot dataclass, rejects the lifecycle enum."""
    from visioncore.core import TargetState as TargetLifecycleEnum

    s = TargetStateSerializer()
    # Snapshot is accepted.
    js = s.serialize(_make_state())
    assert isinstance(js, str)

    # Enum is rejected.
    try:
        s.serialize(TargetLifecycleEnum.ACTIVE)  # type: ignore[arg-type]
        raise AssertionError("serialize should reject the enum")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# Package / import sanity
# ---------------------------------------------------------------------------

def test_direct_and_package_imports_are_same():
    """visioncore.protocol.serialization.TargetStateSerializer == direct import."""
    assert TargetStateSerializer is TargetStateSerializerDirect


def test_package_exports_serializer():
    """visioncore.protocol.serialization.__all__ contains TargetStateSerializer."""
    import visioncore.protocol.serialization as ser
    assert "TargetStateSerializer" in ser.__all__


# ---------------------------------------------------------------------------
# Roundtrip (B3 spec: "roundtrip")
# ---------------------------------------------------------------------------

def test_roundtrip_basic():
    """serialize -> deserialize yields an equal TargetState."""
    s = TargetStateSerializer()
    ts = _make_state()
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2


def test_roundtrip_with_metadata():
    """Roundtrip preserves populated metadata."""
    s = TargetStateSerializer()
    ts = _make_state(metadata={"src": "yolo26", "roi": "entry", "n": 7})
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2
    assert ts2.metadata == {"src": "yolo26", "roi": "entry", "n": 7}


def test_roundtrip_with_nested_metadata():
    """Roundtrip preserves nested dict / list structures in metadata."""
    s = TargetStateSerializer()
    ts = _make_state(metadata={
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "tags": ["a", "b", "c"],
        "nested": {"deep": {"value": 42}},
    })
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2
    assert ts2.metadata["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    assert ts2.metadata["tags"] == ["a", "b", "c"]
    assert ts2.metadata["nested"]["deep"]["value"] == 42


def test_roundtrip_with_none_optionals():
    """Roundtrip preserves None for local_id / global_id / camera_id."""
    s = TargetStateSerializer()
    ts = _make_state(local_id=None, global_id=None, camera_id=None)
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2
    assert ts2.local_id is None
    assert ts2.global_id is None
    assert ts2.camera_id is None


def test_roundtrip_empty_metadata():
    """Roundtrip works with empty metadata (default)."""
    s = TargetStateSerializer()
    ts = _make_state(metadata={})
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2
    assert ts2.metadata == {}


def test_roundtrip_extreme_floats():
    """Roundtrip preserves extreme float values (precision check)."""
    s = TargetStateSerializer()
    ts = _make_state(
        cx=0.999999, cy=-0.000001, vx=123.456, vy=-987.654,
        timestamp=1e10, confidence=0.0, width=1.0, height=0.0,
    )
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2


def test_roundtrip_negative_and_zero_ids():
    """Roundtrip preserves zero/negative IDs."""
    s = TargetStateSerializer()
    ts = _make_state(target_id=0, local_id=-1, camera_id=-99)
    js = s.serialize(ts)
    ts2 = s.deserialize(js)
    assert ts == ts2


def test_roundtrip_produces_valid_json():
    """serialize output is parseable by the standard json module."""
    s = TargetStateSerializer()
    ts = _make_state(metadata={"k": "v"})
    js = s.serialize(ts)
    # Standard json.loads must accept the output.
    d = json.loads(js)
    assert isinstance(d, dict)
    assert d["target_id"] == 42
    assert d["label"] == "person"


def test_roundtrip_unknown_keys_ignored():
    """deserialize tolerates extra unknown keys in the JSON (forward-compat)."""
    s = TargetStateSerializer()
    ts = _make_state()
    js = s.serialize(ts)
    # Inject unknown keys.
    d = json.loads(js)
    d["unknown_future_field"] = "ignored"
    d["another_unknown"] = 12345
    js2 = json.dumps(d)
    ts2 = s.deserialize(js2)
    assert ts == ts2


# ---------------------------------------------------------------------------
# Invalid JSON (B3 spec: "invalid json")
# ---------------------------------------------------------------------------

def test_invalid_json_syntax_raises_value_error():
    """Malformed JSON raises ValueError, not json.JSONDecodeError."""
    s = TargetStateSerializer()
    bad_inputs = [
        "{not valid json",
        "}{",
        '{"target_id": }',
        "[1, 2, ",  # truncated
        "",
        "   ",
        "null",
    ]
    for bad in bad_inputs:
        try:
            s.deserialize(bad)
            raise AssertionError(
                f"deserialize should raise ValueError for {bad!r}"
            )
        except ValueError as exc:
            assert "not valid JSON" in str(exc) or "JSON" in str(exc), (
                f"Error message should mention JSON for {bad!r}, got: {exc}"
            )


def test_invalid_json_non_dict_raises_value_error():
    """Valid JSON that is not a dict raises ValueError."""
    s = TargetStateSerializer()
    non_dict_inputs = [
        "[1, 2, 3]",           # list
        '"a string"',          # string
        "42",                  # int
        "3.14",                # float
        "true",                # bool
    ]
    for bad in non_dict_inputs:
        try:
            s.deserialize(bad)
            raise AssertionError(
                f"deserialize should raise ValueError for {bad!r}"
            )
        except ValueError as exc:
            assert "dict" in str(exc), (
                f"Error message should mention 'dict' for {bad!r}, got: {exc}"
            )


def test_invalid_json_bytes_input():
    """deserialize accepts bytes and decodes UTF-8 before parsing."""
    s = TargetStateSerializer()
    ts = _make_state()
    js = s.serialize(ts)
    # Encode to bytes and round-trip.
    ts2 = s.deserialize(js.encode("utf-8"))
    assert ts == ts2

    # Malformed bytes also raise ValueError.
    try:
        s.deserialize(b"\xff\xfe not valid utf-8 json")
        raise AssertionError("deserialize should raise ValueError for bad bytes")
    except (ValueError, UnicodeDecodeError):
        pass


def test_invalid_json_wrong_type_raises_type_error():
    """deserialize rejects non-str/non-bytes input with TypeError."""
    s = TargetStateSerializer()
    bad_inputs = [None, 42, 3.14, [1, 2], {"a": 1}, object()]
    for bad in bad_inputs:
        try:
            s.deserialize(bad)  # type: ignore[arg-type]
            raise AssertionError(
                f"deserialize should reject {bad!r} ({type(bad).__name__})"
            )
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# Missing field (B3 spec: "missing field")
# ---------------------------------------------------------------------------

def test_missing_required_field_raises_type_error():
    """deserialize raises TypeError when a required field is missing."""
    s = TargetStateSerializer()
    # Build a valid JSON dict then strip one required field at a time.
    full = _make_state().to_dict()

    required_fields = [
        "target_id", "local_id", "global_id", "label", "confidence",
        "cx", "cy", "vx", "vy", "width", "height", "timestamp", "camera_id",
    ]
    for field_name in required_fields:
        d = dict(full)
        del d[field_name]
        js = json.dumps(d)
        try:
            s.deserialize(js)
            raise AssertionError(
                f"deserialize should raise TypeError when {field_name!r} is missing"
            )
        except TypeError:
            pass


def test_missing_metadata_uses_default():
    """Missing metadata key is allowed -- default_factory supplies {}."""
    s = TargetStateSerializer()
    d = _make_state().to_dict()
    del d["metadata"]
    js = json.dumps(d)
    ts = s.deserialize(js)
    assert ts.metadata == {}


def test_missing_optional_field_none_preserved():
    """Missing None-optional fields (local_id etc.) still raise -- they are required.

    local_id / global_id / camera_id accept None as a VALUE but the KEY
    is still required. This test documents that distinction.
    """
    s = TargetStateSerializer()
    d = _make_state().to_dict()
    del d["local_id"]  # remove the key entirely
    js = json.dumps(d)
    try:
        s.deserialize(js)
        raise AssertionError(
            "deserialize should raise TypeError when local_id key is missing "
            "(even though None is a valid value)"
        )
    except TypeError:
        pass


def test_metadata_none_rejected():
    """deserialize rejects metadata=None per the B1.1 contract tightening."""
    s = TargetStateSerializer()
    d = _make_state().to_dict()
    d["metadata"] = None
    js = json.dumps(d)
    try:
        s.deserialize(js)
        raise AssertionError(
            "deserialize should reject metadata=None (B1.1 contract)"
        )
    except TypeError as exc:
        assert "metadata" in str(exc).lower() or "None" in str(exc)


def test_metadata_non_dict_rejected():
    """deserialize rejects non-dict metadata per the B1.1 contract."""
    s = TargetStateSerializer()
    for bad_meta in ["string", ["list"], 42, 3.14]:
        d = _make_state().to_dict()
        d["metadata"] = bad_meta
        js = json.dumps(d)
        try:
            s.deserialize(js)
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
    s = TargetStateSerializer()
    bad_values = [None, "string", 42, 3.14, [1, 2], {"a": 1}, object()]
    for bad in bad_values:
        try:
            s.serialize(bad)  # type: ignore[arg-type]
            raise AssertionError(
                f"serialize should reject {bad!r} ({type(bad).__name__})"
            )
        except TypeError as exc:
            assert "TargetState" in str(exc), (
                f"Error message should mention TargetState for {bad!r}"
            )


def test_serialize_rejects_non_json_serializable_metadata():
    """serialize raises TypeError when metadata has non-JSON-serializable values."""
    s = TargetStateSerializer()

    class Custom:
        pass

    bad_metadata_values = [
        {"obj": Custom()},            # custom object
        {"set": {1, 2, 3}},           # set (not JSON-serializable)
        {"tuple": (1, 2, 3)},         # tuple -- json.dumps actually accepts
    ]
    # Note: json.dumps accepts tuples (converts to list), so we only test
    # truly non-serializable types: custom objects and sets.
    truly_bad = [
        {"obj": Custom()},
        {"set": {1, 2, 3}},
    ]
    for bad_md in truly_bad:
        ts = _make_state(metadata=bad_md)
        try:
            s.serialize(ts)
            raise AssertionError(
                f"serialize should reject metadata={bad_md!r}"
            )
        except TypeError as exc:
            assert "non-JSON-serializable" in str(exc) or "serializable" in str(exc), (
                f"Error should mention non-JSON-serializable for {bad_md!r}, got: {exc}"
            )


# ---------------------------------------------------------------------------
# Formatting options
# ---------------------------------------------------------------------------

def test_indent_produces_pretty_printed_json():
    """indent=2 produces multi-line JSON with indentation."""
    s = TargetStateSerializer(indent=2)
    js = s.serialize(_make_state())
    assert "\n" in js  # multi-line
    assert "  " in js  # has indentation


def test_no_indent_produces_compact_json():
    """Default (indent=None) produces single-line JSON."""
    s = TargetStateSerializer()
    js = s.serialize(_make_state())
    assert "\n" not in js


def test_sort_keys_produces_deterministic_output():
    """sort_keys=True produces deterministic key ordering."""
    s1 = TargetStateSerializer(sort_keys=True)
    s2 = TargetStateSerializer(sort_keys=True)
    ts = _make_state()
    js1 = s1.serialize(ts)
    js2 = s2.serialize(ts)
    assert js1 == js2
    # Verify keys are sorted: target_id should come after confidence
    # (alphabetically 'c' < 't').
    d = json.loads(js1)
    keys = list(d.keys())
    assert keys == sorted(keys)


def test_repr_includes_options():
    """__repr__ includes indent and sort_keys settings."""
    s = TargetStateSerializer(indent=4, sort_keys=True)
    r = repr(s)
    assert "TargetStateSerializer" in r
    assert "indent=4" in r
    assert "sort_keys=True" in r


# ---------------------------------------------------------------------------
# Idempotence and independence
# ---------------------------------------------------------------------------

def test_serialize_does_not_mutate_state():
    """serialize has no side effect on the input state."""
    s = TargetStateSerializer()
    ts = _make_state(metadata={"k": "v"})
    _ = s.serialize(ts)
    _ = s.serialize(ts)
    assert ts.metadata == {"k": "v"}


def test_serialize_returns_independent_string():
    """Each serialize call returns a new string object."""
    s = TargetStateSerializer()
    ts = _make_state()
    js1 = s.serialize(ts)
    js2 = s.serialize(ts)
    assert js1 == js2
    assert js1 is not js2  # distinct string objects (not interned in general)


def test_serializer_is_reusable_across_calls():
    """A single serializer instance can serialize many different states."""
    s = TargetStateSerializer()
    states = [_make_state(target_id=i) for i in range(10)]
    for ts in states:
        js = s.serialize(ts)
        ts2 = s.deserialize(js)
        assert ts == ts2


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

"""Unit tests for visioncore.protocol.adapters.UDPAdapter.

Verifies the three scenarios required by the B5 spec -- loopback,
multiple states, large payload -- plus contract enforcement (connect
before publish, type checks), serializer injection (both JSON and CBOR),
context manager, and the type-annotation convention.

The loopback tests use real UDP sockets on 127.0.0.1 with ephemeral
ports (bound to port 0 and reading the assigned port). This makes the
tests self-contained -- no external server or fixed port needed.

Run::

    python -m pytest tests/test_udp_adapter.py -v
    # or
    PYTHONPATH=F:/VisionBata python tests/test_udp_adapter.py
"""

from __future__ import annotations

import ast
import inspect
import socket
import threading
import time

from visioncore.protocol.adapters import UDPAdapter
from visioncore.protocol.adapters.udp_adapter import UDPAdapter as UDPAdapterDirect
from visioncore.protocol.base import ProtocolAdapter
from visioncore.protocol.serialization import CBORSerializer, TargetStateSerializer
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


def _make_large_state() -> TargetState:
    """Build a TargetState with 100-key nested metadata (~7KB CBOR)."""
    metadata: dict = {}
    for i in range(100):
        metadata[f"key_{i:03d}"] = {
            "index": i,
            "value": float(i) * 0.01,
            "label": f"label_{i}",
            "nested": {"deep": [i, i + 1, i + 2]},
        }
    return _make_state(
        target_id=999, metadata=metadata,
        timestamp=99999.0, label="large_state",
    )


def _bind_receiver() -> socket.socket:
    """Create a UDP socket bound to 127.0.0.1 on an ephemeral port.

    Returns the socket; the caller can read ``sock.getsockname()[1]``
    to learn the assigned port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind to port 0 -> OS assigns an ephemeral port.
    sock.bind(("127.0.0.1", 0))
    # Set a receive timeout so tests don't hang forever on failure.
    sock.settimeout(5.0)
    return sock


# ---------------------------------------------------------------------------
# Type annotation convention + forbidden imports
# ---------------------------------------------------------------------------

def test_type_annotation_convention_no_top_level_import():
    """UDPAdapter module must not use 'from visioncore import TargetState'."""
    mod = __import__(UDPAdapter.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "visioncore":
            for alias in node.names:
                assert alias.name != "TargetState", (
                    f"{UDPAdapter.__module__} line {node.lineno}: "
                    f"forbidden 'from visioncore import TargetState'."
                )


def test_no_hardcoded_json():
    """The adapter module must NOT import json or hardcode JSON format.

    The adapter is serializer-agnostic -- it delegates serialization
    to the injected serializer and never calls json.dumps/loads itself.
    """
    mod = __import__(UDPAdapter.__module__, fromlist=["_"])
    src = inspect.getsource(mod)

    # The adapter must not import the json module.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "json", (
                    f"UDPAdapter must not import json -- "
                    f"it should use the injected serializer."
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "json", (
                f"UDPAdapter must not import from json -- "
                f"it should use the injected serializer."
            )

    # Source must not contain json.dumps or json.loads calls.
    assert "json.dumps" not in src, "UDPAdapter must not call json.dumps"
    assert "json.loads" not in src, "UDPAdapter must not call json.loads"


# ---------------------------------------------------------------------------
# Package / import sanity
# ---------------------------------------------------------------------------

def test_direct_and_package_imports_are_same():
    """visioncore.protocol.adapters.UDPAdapter == direct import."""
    assert UDPAdapter is UDPAdapterDirect


def test_package_exports_udp_adapter():
    """visioncore.protocol.adapters.__all__ contains UDPAdapter."""
    import visioncore.protocol.adapters as adapters
    assert "UDPAdapter" in adapters.__all__


def test_udp_adapter_is_protocol_adapter():
    """UDPAdapter is a subclass of ProtocolAdapter."""
    assert issubclass(UDPAdapter, ProtocolAdapter)


# ---------------------------------------------------------------------------
# Constructor and repr
# ---------------------------------------------------------------------------

def test_constructor_stores_params():
    """Constructor stores serializer, host, port."""
    ser = CBORSerializer()
    a = UDPAdapter(ser, "127.0.0.1", 9999)
    assert a._serializer is ser
    assert a._host == "127.0.0.1"
    assert a._port == 9999


def test_repr_includes_params():
    """__repr__ includes serializer type, host, port, connection state."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    r = repr(a)
    assert "UDPAdapter" in r
    assert "CBORSerializer" in r
    assert "127.0.0.1" in r
    assert "9999" in r
    assert "connected=False" in r


# ---------------------------------------------------------------------------
# Lifecycle contract
# ---------------------------------------------------------------------------

def test_starts_disconnected():
    """A new UDPAdapter is not connected."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    assert a.health_check() is False


def test_connect_disconnect():
    """connect/disconnect toggle health_check. Both are idempotent."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    assert a.health_check() is False

    a.connect()
    assert a.health_check() is True

    # Idempotent connect.
    a.connect()
    assert a.health_check() is True

    a.disconnect()
    assert a.health_check() is False

    # Idempotent disconnect.
    a.disconnect()
    assert a.health_check() is False


def test_publish_when_disconnected_raises():
    """publish on a disconnected adapter raises RuntimeError."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    try:
        a.publish(_make_state())
        raise AssertionError("publish should raise RuntimeError when disconnected")
    except RuntimeError as exc:
        assert "not connected" in str(exc).lower()


def test_context_manager():
    """UDPAdapter supports 'with' -- connect on enter, disconnect on exit."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    assert a.health_check() is False
    with a:
        assert a.health_check() is True
    assert a.health_check() is False


def test_context_manager_disconnects_on_exception():
    """__exit__ calls disconnect even when an exception propagates."""
    a = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
    try:
        with a:
            raise ValueError("test error")
    except ValueError:
        pass
    assert a.health_check() is False


# ---------------------------------------------------------------------------
# B5 spec: loopback
# ---------------------------------------------------------------------------

def test_loopback_cbor():
    """Send a CBOR-serialized state via UDP loopback, receive and verify.

    Binds a receiver socket on 127.0.0.1, creates a UDPAdapter pointing
    to that port, publishes a state, receives the datagram, deserializes
    with the same CBOR serializer, and verifies equality.
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = CBORSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        state = _make_state(target_id=77, label="vehicle", metadata={"src": "test"})

        with adapter:
            adapter.publish(state)

        # Receive the datagram.
        data, addr = receiver.recvfrom(65535)
        assert len(data) > 0
        assert addr[0] == "127.0.0.1"

        # Deserialize and verify.
        received_state = ser.deserialize(data)
        assert received_state == state
        assert received_state.target_id == 77
        assert received_state.label == "vehicle"
    finally:
        receiver.close()


def test_loopback_json():
    """Send a JSON-serialized state via UDP loopback, receive and verify.

    Proves the adapter is serializer-agnostic -- it works with JSON
    (str output) as well as CBOR (bytes output). The adapter must
    encode the str to UTF-8 before sending.
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = TargetStateSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        state = _make_state(target_id=88, label="person", metadata={"k": "v"})

        with adapter:
            adapter.publish(state)

        data, addr = receiver.recvfrom(65535)
        assert len(data) > 0

        # The adapter should have encoded the JSON string to UTF-8 bytes.
        # The JSON serializer can deserialize from bytes (B3 feature).
        received_state = ser.deserialize(data)
        assert received_state == state
        assert received_state.target_id == 88
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# B5 spec: multiple states
# ---------------------------------------------------------------------------

def test_multiple_states_cbor():
    """Send multiple states in sequence, receive all, verify order.

    Publishes 5 states with distinct target_ids, receives each datagram,
    deserializes, and verifies the order is preserved (UDP on loopback
    preserves order for a single sender).
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = CBORSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        states = [
            _make_state(target_id=i, cx=0.1 * i, timestamp=float(i))
            for i in range(5)
        ]

        with adapter:
            for s in states:
                adapter.publish(s)

        # Receive and verify each datagram in order.
        received: list[TargetState] = []
        for _ in range(5):
            data, _ = receiver.recvfrom(65535)
            received.append(ser.deserialize(data))

        assert len(received) == 5
        for i, (sent, got) in enumerate(zip(states, received)):
            assert got == sent, f"State {i} mismatch: sent={sent}, got={got}"
            assert got.target_id == i
    finally:
        receiver.close()


def test_multiple_states_json():
    """Same as test_multiple_states_cbor but with JSON serializer."""
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = TargetStateSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        states = [
            _make_state(target_id=100 + i, label=f"label_{i}")
            for i in range(3)
        ]

        with adapter:
            for s in states:
                adapter.publish(s)

        received: list[TargetState] = []
        for _ in range(3):
            data, _ = receiver.recvfrom(65535)
            received.append(ser.deserialize(data))

        assert len(received) == 3
        for i, (sent, got) in enumerate(zip(states, received)):
            assert got == sent
            assert got.target_id == 100 + i
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# B5 spec: large payload
# ---------------------------------------------------------------------------

def test_large_payload_cbor():
    """Send a large TargetState (100-key metadata, ~7KB CBOR) via loopback.

    This exceeds the typical 1472-byte Ethernet MTU but is within the
    65507-byte UDP limit. On localhost (loopback MTU ~65535) this works
    without fragmentation issues.
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = CBORSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        state = _make_large_state()

        # Verify the payload is actually "large" (>1472 bytes).
        payload = ser.serialize(state)
        assert len(payload) > 1472, (
            f"Test payload too small ({len(payload)} bytes) -- "
            f"need > 1472 to exceed typical MTU"
        )

        with adapter:
            adapter.publish(state)

        # Receive with a large buffer.
        data, _ = receiver.recvfrom(65535)
        assert len(data) == len(payload)

        received_state = ser.deserialize(data)
        assert received_state == state
        assert received_state.target_id == 999
        assert len(received_state.metadata) == 100
    finally:
        receiver.close()


def test_large_payload_json():
    """Same as test_large_payload_cbor but with JSON serializer.

    JSON payload for the same state is ~9.9KB (vs CBOR's ~6.8KB),
    still within UDP limits on loopback.
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = TargetStateSerializer()
        adapter = UDPAdapter(ser, "127.0.0.1", port)
        state = _make_large_state()

        payload = ser.serialize(state)
        payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        assert len(payload_bytes) > 1472

        with adapter:
            adapter.publish(state)

        data, _ = receiver.recvfrom(65535)
        received_state = ser.deserialize(data)
        assert received_state == state
        assert len(received_state.metadata) == 100
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# Type enforcement
# ---------------------------------------------------------------------------

def test_publish_rejects_non_target_state():
    """publish rejects non-TargetState values with TypeError."""
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        a = UDPAdapter(CBORSerializer(), "127.0.0.1", port)
        a.connect()
        for bad in [None, "string", 42, [1, 2], {"a": 1}, object()]:
            try:
                a.publish(bad)  # type: ignore[arg-type]
                raise AssertionError(f"publish should reject {bad!r}")
            except TypeError:
                pass
        a.disconnect()
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------

def test_reconnect_after_disconnect():
    """A new connect() after disconnect() creates a fresh working socket."""
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = CBORSerializer()
        a = UDPAdapter(ser, "127.0.0.1", port)

        # First connection.
        a.connect()
        a.publish(_make_state(target_id=1))
        # Drain the first datagram before disconnecting.
        data1, _ = receiver.recvfrom(65535)
        assert ser.deserialize(data1).target_id == 1
        a.disconnect()

        # Second connection -- must work just as well.
        a.connect()
        a.publish(_make_state(target_id=2))

        data2, _ = receiver.recvfrom(65535)
        state = ser.deserialize(data2)
        assert state.target_id == 2
        a.disconnect()
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# publish_many (inherited default)
# ---------------------------------------------------------------------------

def test_publish_many_inherited_default():
    """publish_many loops publish() -- each state sent as separate datagram."""
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        ser = CBORSerializer()
        a = UDPAdapter(ser, "127.0.0.1", port)
        states = [_make_state(target_id=i) for i in range(3)]

        with a:
            a.publish_many(states)

        received: list[TargetState] = []
        for _ in range(3):
            data, _ = receiver.recvfrom(65535)
            received.append(ser.deserialize(data))

        for i, (sent, got) in enumerate(zip(states, received)):
            assert got == sent
    finally:
        receiver.close()


def test_publish_many_empty_sequence():
    """publish_many with empty sequence is a no-op (no datagrams sent)."""
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        a = UDPAdapter(CBORSerializer(), "127.0.0.1", port)
        # Empty publish_many works even when disconnected (inherited default).
        a.publish_many([])
        assert a.health_check() is False
    finally:
        receiver.close()


# ---------------------------------------------------------------------------
# Serializer injection: adapter does not hardcode format
# ---------------------------------------------------------------------------

def test_adapter_works_with_both_serializers():
    """The same adapter code path works with JSON and CBOR serializers.

    This is the core B5 requirement: the adapter must NOT hardcode JSON.
    It delegates to serializer.serialize() and handles both str and bytes
    return types.
    """
    receiver = _bind_receiver()
    port = receiver.getsockname()[1]
    try:
        state = _make_state(target_id=55, metadata={"k": "v"})

        # Test with JSON serializer (returns str).
        json_ser = TargetStateSerializer()
        a_json = UDPAdapter(json_ser, "127.0.0.1", port)
        with a_json:
            a_json.publish(state)
        data_json, _ = receiver.recvfrom(65535)
        state_json = json_ser.deserialize(data_json)
        assert state_json == state

        # Test with CBOR serializer (returns bytes).
        cbor_ser = CBORSerializer()
        a_cbor = UDPAdapter(cbor_ser, "127.0.0.1", port)
        with a_cbor:
            a_cbor.publish(state)
        data_cbor, _ = receiver.recvfrom(65535)
        state_cbor = cbor_ser.deserialize(data_cbor)
        assert state_cbor == state

        # The two payloads are different formats (JSON text vs CBOR binary).
        assert data_json != data_cbor
    finally:
        receiver.close()


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

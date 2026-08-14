"""CBOR serializer for TargetState snapshots.

Defines :class:`CBORSerializer`, which converts
:class:`~visioncore.state.target_state.TargetState` instances to and from
CBOR (Concise Binary Object Representation, RFC 8949) bytes. CBOR is the
binary counterpart to the JSON text format provided by
:class:`~visioncore.protocol.serialization.json_serializer.TargetStateSerializer`.

CBOR advantages over JSON for snapshot transport
------------------------------------------------
* **Smaller payloads** -- binary encoding of ints/floats/strings is
  typically 20-40% smaller than the equivalent JSON text. Critical
  when snapshots flow over bandwidth-limited links (UDP, serial).
* **Faster encode/decode** -- no text parsing or float-to-string
  conversion. cbor2's C extension is significantly faster than
  ``json`` for large payloads.
* **Native bytes output** -- ``serialize()`` returns ``bytes`` directly,
  no ``.encode("utf-8")`` step needed before writing to a socket.
* **Type fidelity** -- CBOR preserves int vs float distinction, supports
  byte strings, and has a richer type system than JSON.

CBOR disadvantages
------------------
* **Not human-readable** -- debugging requires a hex dump or decoder.
  Use :class:`TargetStateSerializer` (JSON) for logging and console
  output; use CBOR for wire transport.
* **External dependency** -- requires the ``cbor2`` package. JSON uses
  only the standard library.

Dependency
----------
This module imports :mod:`cbor2` (PyPI package, version 6.1.0+).
Install into the project venv::

    pip install cbor2

The ``cbor2`` package ships a C extension for CPython; if the extension
is unavailable it falls back to a pure-Python implementation (slower
but functional).

Type annotation convention
--------------------------
This module imports ``TargetState`` via its fully-qualified module path::

    from visioncore.state.target_state import TargetState

This avoids collision with the ``visioncore.core.TargetState`` enum
re-exported from the top-level ``visioncore`` package.

Transport agnosticism
---------------------
This module contains **no** socket, **no** UDP, **no** ROS2, **no**
MAVLink code. It is a pure data-to-bytes converter. Wiring it onto a
transport is the responsibility of a protocol adapter.

Example
-------
    >>> from visioncore.state.target_state import TargetState
    >>> from visioncore.protocol.serialization import CBORSerializer
    >>> s = CBORSerializer()
    >>> ts = TargetState(1, 1, None, "person", 0.9, 0.5, 0.5,
    ...                  0.0, 0.0, 0.1, 0.2, 0.0, 0, {"src": "yolo"})
    >>> data = s.serialize(ts)
    >>> isinstance(data, bytes)
    True
    >>> ts2 = s.deserialize(data)
    >>> ts == ts2
    True
"""

from __future__ import annotations

from typing import Any

import cbor2

from visioncore.state.target_state import TargetState


__all__ = ["CBORSerializer"]


# Type alias for accepted binary input types. ``bytearray`` and
# ``memoryview`` are common in network code (e.g. recv buffers, zero-copy
# slices). cbor2.loads accepts all three natively.
_BinaryInput = bytes | bytearray | memoryview


class CBORSerializer:
    """Serializes :class:`TargetState` snapshots to and from CBOR bytes.

    The binary counterpart to
    :class:`~visioncore.protocol.serialization.json_serializer.TargetStateSerializer`.
    Use CBOR when payload size or encode/decode speed matters more than
    human readability -- e.g. for network transport over UDP, for
    high-volume snapshot logging to binary files, or for inter-process
    shared-memory queues.

    The serializer leverages TargetState's existing ``to_dict()`` /
    ``from_dict()`` methods, adding CBOR encoding/decoding and consistent
    error handling on top. No custom CBOR encoder hook is needed --
    TargetState fields are all CBOR-native types (int, float, str, dict,
    None).

    Output format
    -------------
    The CBOR output is a single CBOR map with one entry per TargetState
    field, mirroring the JSON output structure but encoded as binary::

        A1                                    # map(1) -- outer map
          6A 7461726765745F6964               # text "target_id"
          18 2A                               # unsigned 42
          ...

    The binary encoding is deterministic for the same input on the same
    cbor2 version -- int/float/str/dict encodings have no ambiguity.
    Unlike JSON, there is no ``sort_keys`` option because CBOR maps are
    unordered by spec and key ordering does not affect human readability.

    Thread safety
    -------------
    CBORSerializer is stateless (no instance variables beyond
    construction-time configuration). Instances are safe to share
    between threads without locking.

    Example:
        >>> s = CBORSerializer()
        >>> data = s.serialize(ts)
        >>> ts2 = s.deserialize(data)
        >>> ts == ts2
        True
    """

    def __init__(self) -> None:
        """Construct a CBORSerializer.

        CBORSerializer takes no configuration options. Unlike the JSON
        serializer, there is no ``indent`` (CBOR is binary, no
        pretty-printing) and no ``sort_keys`` (CBOR maps are unordered
        by spec, and binary output is not meant for human comparison).
        """
        # No state -- the serializer is a pure function wrapper around
        # cbor2.dumps/loads. Holding an explicit __init__ makes the
        # constructor signature explicit and leaves room for future
        # options (e.g. a `timezone` parameter for datetime encoding)
        # without breaking callers.

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self, state: TargetState) -> bytes:
        """Serialize a TargetState to CBOR bytes.

        Args:
            state: The snapshot to serialize. Must be a TargetState
                instance.

        Returns:
            A ``bytes`` object containing the CBOR-encoded snapshot.
            The bytes are ready to write to a socket, file, or shared
            memory buffer without further encoding.

        Raises:
            TypeError: If ``state`` is not a TargetState, or if the
                snapshot's ``metadata`` contains values that cbor2
                cannot encode (rare -- cbor2 handles most Python types
                including bytes, datetime, and sets natively, unlike
                JSON).
        """
        if not isinstance(state, TargetState):
            raise TypeError(
                f"serialize: state must be a TargetState, got "
                f"{type(state).__name__}"
            )
        # TargetState.to_dict() returns a dict of CBOR-native primitives.
        # cbor2.dumps handles int/float/str/dict/None/list natively, and
        # also supports bytes/datetime/set (which JSON does not). The
        # only failure mode is a truly exotic type (e.g. a custom class
        # with no __cbor2__ hook) -- surface that with a clear error.
        try:
            return cbor2.dumps(state.to_dict())
        except (cbor2.CBOREncodeTypeError, cbor2.CBOREncodeValueError) as exc:
            raise TypeError(
                f"serialize: TargetState contains data that cbor2 cannot "
                f"encode: {exc}"
            ) from exc
        except cbor2.CBOREncodeError as exc:
            raise TypeError(
                f"serialize: cbor2 encoding failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    def deserialize(self, data: _BinaryInput) -> TargetState:
        """Deserialize CBOR bytes to a TargetState.

        Accepts ``bytes``, ``bytearray``, and ``memoryview`` -- the
        common binary buffer types in network code. ``str`` is NOT
        accepted (CBOR is binary; pass ``data.encode("utf-8")`` if you
        have a text buffer, though that would be unusual for CBOR).

        Args:
            data: A CBOR-encoded byte buffer containing a dict with
                TargetState fields. Unknown keys are silently ignored
                (forward-compatibility, same as JSON deserialize).
                Missing required fields raise ``TypeError``.

        Returns:
            A new frozen TargetState reconstructed from the CBOR data.

        Raises:
            TypeError: If ``data`` is not a binary buffer type, or if
                the decoded dict is missing required TargetState fields,
                or if ``metadata`` is ``None`` or not a ``dict`` (per
                the B1.1 contract tightening).
            ValueError: If ``data`` is not valid CBOR, or if the CBOR
                decodes to a non-dict value (e.g. a list or a bare
                integer).
        """
        # Accept bytes, bytearray, memoryview -- reject str and others.
        # str is rejected because CBOR is binary; accepting str would
        # imply text decoding which is not meaningful for CBOR.
        if isinstance(data, (bytes, bytearray, memoryview)):
            # cbor2.loads accepts all three natively. Convert memoryview
            # to bytes for consistency (cbor2 handles it, but being
            # explicit avoids edge cases with zero-length views).
            if isinstance(data, memoryview):
                data = bytes(data)
        else:
            raise TypeError(
                f"deserialize: data must be bytes, bytearray, or "
                f"memoryview, got {type(data).__name__}"
            )

        # Parse CBOR -- surface decode errors with a clear message.
        # cbor2.CBORDecodeError covers: truncated input, invalid major
        # type, bad UTF-8 in text strings, truncated nested structures.
        try:
            obj: Any = cbor2.loads(data)
        except cbor2.CBORDecodeError as exc:
            raise ValueError(
                f"deserialize: input is not valid CBOR: {exc}"
            ) from exc
        except cbor2.CBORDecodeEOF as exc:
            raise ValueError(
                f"deserialize: CBOR input is truncated (unexpected end "
                f"of data): {exc}"
            ) from exc

        # CBOR can decode to any type (list, int, str, bytes, ...).
        # TargetState.from_dict requires a dict -- reject anything else.
        if not isinstance(obj, dict):
            raise ValueError(
                f"deserialize: CBOR must decode to a dict, got "
                f"{type(obj).__name__}"
            )

        # Delegate to TargetState.from_dict, which enforces the B1.1
        # metadata type contract (rejects None / non-dict metadata).
        return TargetState.from_dict(obj)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return "CBORSerializer()"

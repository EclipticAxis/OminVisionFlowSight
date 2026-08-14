"""JSON serializer for TargetState snapshots.

Defines :class:`TargetStateSerializer`, which converts
:class:`~visioncore.state.target_state.TargetState` instances to and from
JSON strings. This is the canonical text-based serialization format for
the VisionCore protocol layer.

The serializer leverages TargetState's existing ``to_dict()`` /
``from_dict()`` methods, adding JSON encoding/decoding and consistent
error handling on top. No custom JSON encoder is needed -- TargetState
fields are all JSON-native primitives (int, float, str, dict).

Type annotation convention
--------------------------
This module imports ``TargetState`` via its fully-qualified module path::

    from visioncore.state.target_state import TargetState

This avoids collision with the ``visioncore.core.TargetState`` enum
re-exported from the top-level ``visioncore`` package. See
``docs/protocol-foundation.md`` for the full rationale.

Transport agnosticism
---------------------
This module contains **no** socket, **no** UDP, **no** ROS2, **no**
MAVLink code. It is a pure data-to-text converter. Wiring it onto a
transport is the responsibility of a future adapter (e.g. a
``UdpAdapter`` that uses ``TargetStateSerializer`` internally to
prepare payloads).

Error handling
--------------
* ``serialize()`` raises ``TypeError`` if the input is not a TargetState,
  or if TargetState's metadata contains non-JSON-serializable values
  (e.g. numpy arrays, custom objects).
* ``deserialize()`` raises ``ValueError`` if the input is not valid
  JSON or does not decode to a dict. It raises ``TypeError`` if the
  decoded dict is missing required TargetState fields or has an
  invalid ``metadata`` type (None / non-dict -- per the B1.1 contract).

Example
-------
    >>> from visioncore.state.target_state import TargetState
    >>> from visioncore.protocol.serialization import TargetStateSerializer
    >>> s = TargetStateSerializer()
    >>> ts = TargetState(1, 1, None, "person", 0.9, 0.5, 0.5,
    ...                  0.0, 0.0, 0.1, 0.2, 0.0, 0, {"src": "yolo"})
    >>> js = s.serialize(ts)
    >>> ts2 = s.deserialize(js)
    >>> ts == ts2
    True
"""

from __future__ import annotations

import json
from typing import Any

from visioncore.state.target_state import TargetState


__all__ = ["TargetStateSerializer"]


class TargetStateSerializer:
    """Serializes :class:`TargetState` snapshots to and from JSON strings.

    The serializer is the canonical text-based serialization format for
    the VisionCore protocol layer. It wraps TargetState's existing
    ``to_dict()`` / ``from_dict()`` methods with JSON encoding/decoding
    and consistent error handling.

    The serializer is stateless aside from formatting options (indent,
    sort_keys) -- it does not hold references to serialized data, open
    files, or maintain buffers. Instances are safe to reuse across
    calls and to share between threads (the internal state is set once
    at construction and never mutated).

    Output format
    -------------
    The JSON output is a flat dict with one key per TargetState field::

        {
            "target_id": 42,
            "local_id": 1,
            "global_id": null,
            "label": "person",
            "confidence": 0.92,
            "cx": 0.5,
            "cy": 0.4,
            "vx": 0.01,
            "vy": -0.002,
            "width": 0.12,
            "height": 0.3,
            "timestamp": 12.5,
            "camera_id": 0,
            "metadata": {"src": "yolo"}
        }

    No format-version wrapper is applied in B3. A future versioned
    format (``{"version": 1, "state": {...}}``) can be added without
    breaking ``deserialize`` -- unknown top-level keys are ignored by
    ``TargetState.from_dict``, and a versioned wrapper would simply
    extract the ``"state"`` sub-dict before delegating.

    Args:
        indent: If not ``None``, pretty-print JSON with this many spaces
            per indent level. Useful for human-readable output (logging,
            debugging, file dumps). ``None`` (default) produces compact
            single-line JSON -- most efficient for transport.
        sort_keys: If ``True``, sort dict keys in the output. Useful for
            deterministic output in tests and content-addressed caching.
            ``False`` (default) preserves TargetState's field declaration
            order, which matches ``to_dict()`` output.

    Example:
        >>> s = TargetStateSerializer(indent=2, sort_keys=True)
        >>> js = s.serialize(ts)
        >>> # js is now a pretty-printed, key-sorted JSON string
    """

    def __init__(
        self,
        *,
        indent: int | None = None,
        sort_keys: bool = False,
    ) -> None:
        self._indent: int | None = indent
        self._sort_keys: bool = sort_keys

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self, state: TargetState) -> str:
        """Serialize a TargetState to a JSON string.

        Args:
            state: The snapshot to serialize. Must be a TargetState
                instance.

        Returns:
            A JSON string representation of the snapshot. The string is
            UTF-8 encoded text; callers that need bytes should call
            ``.encode("utf-8")`` on the result.

        Raises:
            TypeError: If ``state`` is not a TargetState, or if the
                snapshot's ``metadata`` contains values that are not
                JSON-serializable (e.g. numpy arrays, custom objects,
                sets). The error message identifies the offending value.
        """
        if not isinstance(state, TargetState):
            raise TypeError(
                f"serialize: state must be a TargetState, got "
                f"{type(state).__name__}"
            )
        # TargetState.to_dict() returns a dict of JSON-native primitives
        # (int/float/str/dict). json.dumps will only fail if metadata
        # contains non-serializable values -- surface that with a clear
        # error message rather than letting json's opaque TypeError
        # propagate.
        try:
            return json.dumps(
                state.to_dict(),
                indent=self._indent,
                sort_keys=self._sort_keys,
            )
        except TypeError as exc:
            raise TypeError(
                f"serialize: TargetState contains non-JSON-serializable "
                f"data in metadata (or elsewhere): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    def deserialize(self, data: str | bytes) -> TargetState:
        """Deserialize a JSON string (or bytes) to a TargetState.

        Accepts both ``str`` and ``bytes`` input. ``bytes`` are decoded
        as UTF-8 before parsing -- this is a convenience for network
        code that naturally works with byte buffers.

        Args:
            data: A JSON string or UTF-8 bytes encoding a dict with
                TargetState fields. Unknown keys are silently ignored
                (forward-compatibility). Missing required fields raise
                ``TypeError``.

        Returns:
            A new frozen TargetState reconstructed from the JSON data.

        Raises:
            TypeError: If ``data`` is not a ``str`` or ``bytes``.
            ValueError: If ``data`` is not valid JSON, or if the JSON
                decodes to a non-dict value (e.g. a list or a bare
                string).
            TypeError: If the decoded dict is missing required TargetState
                fields, or if ``metadata`` is present but is ``None`` or
                not a ``dict`` (per the B1.1 contract tightening).
        """
        # Accept both str and bytes for network-code convenience.
        if isinstance(data, bytes):
            text: str = data.decode("utf-8")
        elif isinstance(data, str):
            text = data
        else:
            raise TypeError(
                f"deserialize: data must be str or bytes, got "
                f"{type(data).__name__}"
            )

        # Parse JSON -- surface decode errors with a clear message.
        try:
            obj: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"deserialize: input is not valid JSON: {exc}"
            ) from exc

        # JSON can decode to any primitive type (list, str, int, ...).
        # TargetState.from_dict requires a dict -- reject anything else
        # with a clear error rather than letting from_dict fail with a
        # confusing AttributeError.
        if not isinstance(obj, dict):
            raise ValueError(
                f"deserialize: JSON must decode to a dict, got "
                f"{type(obj).__name__} ({obj!r:.60})"
            )

        # Delegate to TargetState.from_dict, which enforces the B1.1
        # metadata type contract (rejects None / non-dict metadata).
        return TargetState.from_dict(obj)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TargetStateSerializer(indent={self._indent!r}, "
            f"sort_keys={self._sort_keys!r})"
        )

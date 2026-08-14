"""TargetState -- immutable kinematic + identity snapshot for a tracked target.

This module defines :class:`TargetState`, a frozen (hashable) dataclass that
captures the *observable* state of a single target at one instant in time:
where it is, how fast it is moving, what it is, and which camera saw it.

TargetState is the fundamental unit that flows between vision pipeline
stages once a track has been promoted to a "target of interest". It is
deliberately decoupled from:

    * the legacy mutable :class:`visioncore.core.Track` (which holds a
      Detection + velocity and is updated in place each frame);
    * the legacy :class:`visioncore.core.Target` (which aggregates a Track
      + lifecycle state + attributes as a mutable container);
    * the legacy :class:`visioncore.core.TargetState` **enum** (ACTIVE /
      LOST / LOCKED / RECOVERED / REMOVED) -- the enum names *lifecycle*,
      this dataclass names *snapshot*. The two coexist intentionally.

Design rationale
----------------
* **Immutable + slotted**: snapshots are safe to share across threads,
  cache, log, and replay without defensive copying. ``slots=True`` keeps
  memory footprint tight when millions of snapshots are retained.
* **Plain primitives**: every field is a primitive (int / float / str /
  dict). No numpy, no Qt, no framework coupling. Serialisation is a
  single ``dataclasses.asdict`` away.
* **Three-ID model** (``target_id`` / ``local_id`` / ``global_id``):
  supports both single-camera operation (``local_id`` only) and future
  multi-camera fusion (``global_id`` from a cross-camera ReID stage).
  See ``docs/targetstate-design.md`` for the extension path.
* **No protocol code**: this module contains *no* transport, *no* ROS2,
  *no* MAVLink, *no* UDP. It is a pure data structure. Wiring it onto a
  transport is the responsibility of a future adapter layer.

Thread safety
-------------
TargetState is frozen; instances may be freely shared between threads
without locking. The ``metadata`` dict is technically mutable (Python has
no frozen dict) -- by convention it must be treated as read-only after
construction. Producers that need to defend against downstream mutation
should pass ``copy_with(metadata={...})`` or a ``dict.copy()``.

Example
-------
    >>> ts = TargetState(
    ...     target_id=42,
    ...     local_id=1,
    ...     global_id=None,
    ...     label="person",
    ...     confidence=0.92,
    ...     cx=0.5, cy=0.4,
    ...     vx=0.01, vy=-0.002,
    ...     width=0.12, height=0.30,
    ...     timestamp=12.5,
    ...     camera_id=0,
    ...     metadata={"source": "yolo26"},
    ... )
    >>> ts.target_id
    42
    >>> moved = ts.copy_with(cx=0.55, timestamp=12.6)
    >>> moved.cx, ts.cx
    (0.55, 0.5)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


__all__ = ["TargetState"]


@dataclass(slots=True, frozen=True)
class TargetState:
    """Immutable snapshot of a target's observable state.

    A TargetState is a point-in-time observation: identity (who), label
    (what), kinematics (where + how fast), size, and provenance (which
    camera, when). It carries no behaviour and no lifecycle -- it is the
    raw material that downstream stages (state machines, fusion, decision
    modules, transport adapters) consume.

    Attributes:
        target_id: Stable identifier for this target within the current
            pipeline scope. Typically the per-camera track id assigned by
            :class:`visioncore.target_manager.TargetManager`. Must be
            present -- a snapshot without an identity is meaningless.
        local_id: Identifier local to the producing camera/slot. ``None``
            when the producer has no notion of a separate local id (e.g.
            single-camera systems where ``local_id == target_id``). Used
            by multi-camera fusion to disambiguate overlapping target_ids
            across slots.
        global_id: Cross-camera identifier assigned by a global fusion /
            ReID stage. ``None`` until fusion has run. Stays ``None`` in
            single-camera deployments -- absence is the signal that no
            global identity has been established yet.
        label: Semantic class label for the target (e.g. ``"person"``,
            ``"vehicle"``, ``"head"``). Authoritative identity is the
            string itself; consumers must not assume a fixed vocabulary.
        confidence: Producer confidence in the range ``[0.0, 1.0]``.
            Combines detector score and (optionally) tracker stability.
            Higher is more certain. Not validated on construction -- the
            pipeline is responsible for clamping at the boundary.
        cx: Centre x-coordinate of the target's bounding box, in
            normalised image coordinates ``[0.0, 1.0]``.
        cy: Centre y-coordinate of the target's bounding box, in
            normalised image coordinates ``[0.0, 1.0]``.
        vx: Estimated instantaneous x-velocity in normalised units per
            second. ``0.0`` when no motion estimate is available.
        vy: Estimated instantaneous y-velocity in normalised units per
            second. ``0.0`` when no motion estimate is available.
        width: Bounding box width, normalised ``[0.0, 1.0]``.
        height: Bounding box height, normalised ``[0.0, 1.0]``.
        timestamp: Wall-clock or monotonic timestamp in seconds when the
            snapshot was taken. Used for temporal ordering and latency
            measurement. The time base is producer-defined; consumers
            must not assume a specific epoch.
        camera_id: Identifier of the source camera/slot. ``None`` when
            the snapshot originates from a non-camera source (e.g. a
            fused/global target produced by cross-camera ReID).
        metadata: Extensible key-value store for producer-specific data
            that does not belong in the core schema (e.g. detection
            backend name, ReID embedding hash, region-of-interest tag).
            Defaults to an empty dict. Treat as read-only after
            construction -- mutate via :meth:`copy_with` instead.
        attributes: Key-value store for target-level attributes estimated
            by an attribute-estimation stage (e.g. ``{"age": "adult",
            "color": "red"}``). Defaults to an empty dict. Populated by
            :class:`~visioncore.pipeline.stages.attribute_gesture_stage.AttributeStage`.
        gesture: Gesture label detected for this target (e.g.
            ``"waving"``, ``"pointing"``). ``None`` until a gesture
            stage has run. Populated by
            :class:`~visioncore.pipeline.stages.attribute_gesture_stage.GestureStage`.
        rect: Axis-aligned bounding box in corner form ``(x1, y1, x2,
            y2)``, derived from the centre/size representation
            (``cx``/``cy``/``width``/``height``) by
            :class:`~visioncore.pipeline.stages.rectangle_filter_stage.RectangleStage`.
            ``None`` until that stage has run.
        health: Target health score in ``[0.0, 1.0]`` (``1.0`` = perfect
            track, lower values signal degraded / noisy tracks). ``None``
            until
            :class:`~visioncore.pipeline.stages.rectangle_filter_stage.HealthStage`
            has run.
    """

    target_id: int
    local_id: int | None
    global_id: int | None
    label: str
    confidence: float
    cx: float
    cy: float
    vx: float
    vy: float
    width: float
    height: float
    timestamp: float
    camera_id: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    gesture: str | None = None
    rect: tuple[float, float, float, float] | None = None
    health: float | None = None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise this snapshot to a plain dict.

        The returned dict is a deep-ish copy: the ``metadata`` dict is
        copied so that downstream mutation does not leak back into this
        snapshot. All other fields are primitives and copied by value.

        Returns:
            A new dict with one entry per field, suitable for JSON
            serialisation (provided ``metadata`` values are themselves
            JSON-serialisable -- the core schema does not enforce this).

        Example:
            >>> ts = TargetState(1, 1, None, "person", 0.9, 0.5, 0.5,
            ...                  0.0, 0.0, 0.1, 0.2, 0.0, 0, {})
            >>> d = ts.to_dict()
            >>> d["target_id"], d["label"]
            (1, 'person')
        """
        # dataclasses.asdict performs a recursive copy of nested dicts /
        # lists, which is exactly what we want for metadata isolation.
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetState":
        """Construct a TargetState from a plain dict.

        Inverse of :meth:`to_dict`. Unknown keys are silently ignored
        (forward-compatibility: a producer may add fields that this
        version of the schema does not yet know about). Missing required
        fields raise ``TypeError`` from the dataclass constructor.

        Metadata type contract (B1.1 tightening):

            +--------------------------+-----------------------------+
            | ``metadata`` in ``data`` | Behaviour                   |
            +==========================+=============================+
            | key absent               | allowed -- default_factory  |
            |                          | supplies a fresh ``{}``     |
            +--------------------------+-----------------------------+
            | ``None``                 | rejected -- ``TypeError``   |
            +--------------------------+-----------------------------+
            | non-dict value           | rejected -- ``TypeError``   |
            +--------------------------+-----------------------------+
            | ``dict``                 | accepted -- defensively     |
            |                          | shallow-copied              |
            +--------------------------+-----------------------------+

        Args:
            data: Dict with keys matching the field names. ``metadata``
                is copied defensively so the caller's dict cannot leak
                into the constructed instance.

        Returns:
            A new frozen TargetState.

        Raises:
            TypeError: If a required field is missing, or if ``metadata``
                is present but is ``None`` or not a ``dict``.

        Example:
            >>> d = {"target_id": 1, "local_id": 1, "global_id": None,
            ...      "label": "person", "confidence": 0.9,
            ...      "cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.0,
            ...      "width": 0.1, "height": 0.2, "timestamp": 0.0,
            ...      "camera_id": 0, "metadata": {"k": "v"},
            ...      "extra_unknown_field": "ignored"}
            >>> ts = TargetState.from_dict(d)
            >>> ts.target_id, ts.metadata["k"]
            (1, 'v')
        """
        # Build a kwargs dict containing only the fields this dataclass
        # knows about. This makes from_dict tolerant of schema additions
        # produced by newer code -- unknown keys are dropped, not raised.
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in field_names:
                kwargs[key] = value

        # Metadata type contract enforcement (B1.1).
        #   - absent  : leave kwargs untouched -> dataclass default_factory
        #               supplies a fresh empty dict.
        #   - None    : reject. None is not a dict and would silently
        #               violate the field's type annotation, breaking
        #               downstream callers that expect .keys() / .items().
        #   - non-dict: reject for the same reason.
        #   - dict    : defensive shallow copy so the caller's dict
        #               cannot leak into the constructed snapshot.
        if "metadata" in kwargs:
            md_value = kwargs["metadata"]
            if md_value is None:
                raise TypeError(
                    "from_dict: 'metadata' must be a dict, got None. "
                    "Omit the key to use the default empty dict, or pass "
                    "an empty dict explicitly."
                )
            if not isinstance(md_value, dict):
                raise TypeError(
                    f"from_dict: 'metadata' must be a dict, got "
                    f"{type(md_value).__name__}."
                )
            kwargs["metadata"] = dict(md_value)
        # else: metadata absent -> default_factory handles it.

        return cls(**kwargs)

    def copy_with(self, **overrides: Any) -> "TargetState":
        """Return a new TargetState with the given fields replaced.

        Because TargetState is frozen, in-place mutation is impossible.
        ``copy_with`` is the supported way to produce a *derived* snapshot
        -- e.g. advancing a target's position by one frame::

            next_state = state.copy_with(
                cx=state.cx + state.vx * dt,
                cy=state.cy + state.vy * dt,
                timestamp=state.timestamp + dt,
            )

        Args:
            **overrides: Field names mapped to their new values. Keys
                that are not TargetState fields raise ``TypeError``.

        Returns:
            A new frozen TargetState with the overridden fields replaced
            and all other fields carried over by value. The ``metadata``
            dict is shallow-copied so the new snapshot's metadata is
            independent of the original's.

        Raises:
            TypeError: If a key in ``overrides`` is not a TargetState
                field. This is a programming error and fails loudly
                rather than silently dropping the override.

        Example:
            >>> ts = TargetState(1, 1, None, "person", 0.9, 0.5, 0.5,
            ...                  0.0, 0.0, 0.1, 0.2, 0.0, 0, {})
            >>> moved = ts.copy_with(cx=0.6, timestamp=1.0)
            >>> moved.cx, moved.timestamp, ts.cx
            (0.6, 1.0, 0.5)
        """
        # Reject unknown field names loudly -- a typo'd override that
        # silently no-ops is a nasty class of bug.
        field_names = {f.name for f in dataclasses.fields(self)}
        bad_keys = set(overrides) - field_names
        if bad_keys:
            raise TypeError(
                f"copy_with received unknown field(s): "
                f"{sorted(bad_keys)}. Valid fields: {sorted(field_names)}"
            )

        # Start from the current field values, apply overrides, and
        # shallow-copy metadata to preserve snapshot independence.
        new_values: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self)
        }
        new_values.update(overrides)

        # If metadata was not overridden, copy the original so that the
        # new snapshot has its own independent dict (caller may later
        # build a further-derived snapshot via copy_with and mutate the
        # metadata they pass in -- we must not let that leak back).
        if "metadata" not in overrides:
            new_values["metadata"] = dict(new_values["metadata"])

        return type(self)(**new_values)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation.

        Floats are formatted to four decimals to keep the output compact
        for kinematic values; metadata is summarised by its keys (not its
        values) to avoid dumping large payloads into logs.
        """
        if self.metadata:
            meta_keys: str = (
                "{" + ", ".join(repr(k) for k in self.metadata) + "}"
            )
        else:
            meta_keys = "{}"

        if self.attributes:
            attr_keys: str = (
                "{" + ", ".join(repr(k) for k in self.attributes) + "}"
            )
        else:
            attr_keys = "{}"

        if self.rect is not None:
            rect_str: str = (
                f"({self.rect[0]:.4f}, {self.rect[1]:.4f}, "
                f"{self.rect[2]:.4f}, {self.rect[3]:.4f})"
            )
        else:
            rect_str = "None"

        health_str: str = (
            f"{self.health:.4f}" if self.health is not None else "None"
        )

        return (
            f"TargetState(target_id={self.target_id!r}, "
            f"local_id={self.local_id!r}, "
            f"global_id={self.global_id!r}, "
            f"label={self.label!r}, "
            f"confidence={self.confidence:.4f}, "
            f"cx={self.cx:.4f}, cy={self.cy:.4f}, "
            f"vx={self.vx:.4f}, vy={self.vy:.4f}, "
            f"width={self.width:.4f}, height={self.height:.4f}, "
            f"timestamp={self.timestamp:.4f}, "
            f"camera_id={self.camera_id!r}, "
            f"metadata_keys={meta_keys}, "
            f"attributes_keys={attr_keys}, "
            f"gesture={self.gesture!r}, "
            f"rect={rect_str}, health={health_str})"
        )

"""Event data structure for VisionCore.

Defines Event, an immutable record of a significant occurrence in the vision
system. Events are the system's audit and reaction log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable record of a significant occurrence in the vision system.

    Events are the system's audit and reaction log. Producers (detectors,
    trackers, decision modules) emit Events; consumers (loggers, alert
    systems, UI) subscribe to them. Because Events are frozen, they can be
    safely shared across threads and stored in replay buffers.

    The ``payload`` dict is technically mutable (Python does not offer a
    built-in frozen dict), but by convention it should be treated as
    read-only after construction. Producers that need to protect against
    accidental mutation should pass a copy.

    Attributes:
        event_type: A short, namespaced string identifying the event kind,
            e.g. "track.created", "target.lost", "zone.breach". Convention:
            dot-separated ``producer.category``.
        timestamp: Wall-clock or monotonic timestamp in seconds when the
            event was generated. Used for temporal ordering and latency
            measurement.
        payload: Structured, event-specific data. The schema is determined
            by event_type; consumers must negotiate the expected keys.
            Using a dict keeps the core model decoupled from any specific
            event schema. Defaults to an empty dict.

    Example:
        >>> e = Event("target.lost", 12.5, {"target_id": "T-001"})
        >>> e.event_type
        'target.lost'
    """

    event_type: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise representation.

        The payload is summarised by its keys to keep the representation
        compact for large payloads.
        """
        if self.payload:
            payload_keys: str = (
                "{" + ", ".join(repr(k) for k in self.payload) + "}"
            )
        else:
            payload_keys = "{}"

        return (
            f"Event(event_type={self.event_type!r}, "
            f"timestamp={self.timestamp:.4f}, "
            f"payload_keys={payload_keys})"
        )

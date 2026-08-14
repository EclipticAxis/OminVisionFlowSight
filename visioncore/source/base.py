"""FrameSource -- the abstract input-source contract for VisionCore.

Defines :class:`FrameSource`, the abstract base class that every input
source (camera, video file, RTSP stream, synthetic test source) must
implement. A FrameSource is the **origin** of the data that flows through
a :class:`~visioncore.pipeline.pipeline.Pipeline`: its :meth:`read` method
produces :class:`~visioncore.core.frame.Frame` instances that a future
capture stage will push into a
:class:`~visioncore.pipeline.context.PipelineContext`.

Scope of this milestone (C2)
---------------------------
This module defines the **contract only**. It contains:

* :class:`FrameSource` -- the ABC with a five-method lifecycle.
* :class:`FrameSourceError` -- a structured exception that sources *may*
  raise to signal a recoverable source-specific failure.

It deliberately does **not** contain:

* any camera, file, or network I/O (that is C3+, as sibling modules);
* any numpy or OpenCV capture code -- concrete sources implement capture,
  the base class only defines the contract;
* any modification to ``ai/``, ``camera/``, or ``gui/`` -- the existing
  VisionDataPlatform capture stack (``camera/``) continues to run unchanged.

Return-type policy (mandatory)
------------------------------
:meth:`read` returns ``Frame | None`` -- **never** a bare ``numpy.ndarray``,
**never** an OpenCV ``Mat``. The :class:`~visioncore.core.frame.Frame`
dataclass is the unified envelope: it wraps the raw pixel array (which may
be a numpy ndarray internally) together with the frame's identity
(``frame_id``), provenance (``source_id``), and timing (``timestamp``).
Returning the envelope rather than the raw array lets every downstream
stage rely on a single, stable type and keeps provenance attached to the
pixels at all times.

Lifecycle contract
------------------
Every source obeys a five-method lifecycle:

1. ``open()`` -- acquire the underlying resource (open device, open file,
   connect socket). Idempotent -- a second call is a no-op. Must be called
   before :meth:`read`.
2. ``is_open()`` -- cheap state probe returning ``True`` iff the source is
   currently open. Side-effect-free.
3. ``read()`` -- produce the next :class:`~visioncore.core.frame.Frame`,
   or ``None`` when the source is temporarily empty or permanently
   exhausted. Raises :class:`RuntimeError` if called on a closed source
   (contract violation, distinct from the legitimate ``None`` return).
4. ``close()`` -- release the resource acquired in :meth:`open`. Idempotent
   and **must not raise** -- a failing ``close`` would mask the real error
   that triggered teardown.
5. ``health_check()`` -- side-effect-free probe returning ``True`` iff the
   source is open and operationally ready to deliver frames. Must not
   raise; on internal error return ``False``.

The recommended pattern is the context manager, which guarantees
:meth:`close` regardless of what :meth:`read` does::

    with source:
        frame = source.read()
    # close() called automatically on exit, even on exception

Thread safety
-------------
FrameSource itself does not enforce thread safety. Sources are not
required to be safe for concurrent ``read()`` calls from multiple threads
-- producers that share a source across threads must serialise access.
However, :class:`~visioncore.core.frame.Frame` instances are frozen and
safe to share between threads without copying.

Example
-------
    >>> from visioncore.source.dummy_source import DummyFrameSource
    >>> src = DummyFrameSource(source_id="cam0")
    >>> with src:
    ...     f = src.read()
    ...     f.source_id
    ... # close() called automatically
    'cam0'
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visioncore.core.frame import Frame


__all__ = ["FrameSource", "FrameSourceError"]


logger = logging.getLogger(__name__)


class FrameSourceError(RuntimeError):
    """Structured exception signalling a recoverable source failure.

    Sources *may* raise ``FrameSourceError`` (or any subclass) to indicate
    a failure specific to the source's domain -- e.g. a device dropped, a
    file truncated, a stream timed out.

    Raising ``FrameSourceError`` instead of a bare ``RuntimeError`` lets
    upstream callers distinguish "the source deliberately reported a
    failure" from "an unexpected bug crashed the source". Callers that
    want retry / fallback / re-open semantics catch ``FrameSourceError``
    around :meth:`FrameSource.read` and drive a ``close()`` / ``open()``
    recovery cycle.

    Example:
        >>> raise FrameSourceError("device /dev/video0 disconnected")
        Traceback (most recent call last):
            ...
        visioncore.source.base.FrameSourceError: device /dev/video0 disconnected
    """


class FrameSource(ABC):
    """Abstract base class for all vision input sources.

    A FrameSource is the origin of the data flowing through a pipeline.
    Concrete subclasses implement :meth:`read` to produce
    :class:`~visioncore.core.frame.Frame` instances from some underlying
    medium (camera device, video file, network stream, synthetic
    generator). The five-method lifecycle (open / is_open / read / close /
    health_check) is the contract every source obeys.

    Subclassing
    -----------
    Subclasses **must** implement all five abstract methods:
    :meth:`open`, :meth:`is_open`, :meth:`read`, :meth:`close`,
    :meth:`health_check`.

    Subclasses **may** pass a ``source_id`` to ``super().__init__()`` to
    stamp a stable provenance identifier onto the frames they produce.

    Attributes:
        _source_id: Stable identifier for this source (e.g. camera index,
            RTSP URL, file path). Stamped onto generated frames' ``source_id``
            field. Defaults to the class name when ``None``.
    """

    __slots__ = ("_source_id",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, source_id: str | None = None) -> None:
        """Initialise the source with an optional provenance identifier.

        Parameters:
            source_id: Stable identifier stamped onto produced frames'
                ``source_id`` field (e.g. ``"cam0"``, an RTSP URL, a file
                path). When ``None`` (default), the class name is used.
                Two sources of the same class may carry different ids --
                this is how a pipeline distinguishes multi-camera inputs.

        Example:
            >>> class MySrc(FrameSource):
            ...     def open(self): pass
            ...     def is_open(self): return True
            ...     def read(self): return None
            ...     def close(self): pass
            ...     def health_check(self): return True
            >>> MySrc().source_id
            'MySrc'
            >>> MySrc(source_id="cam0").source_id
            'cam0'
        """
        self._source_id: str = (
            source_id if source_id is not None else type(self).__name__
        )
        logger.debug("FrameSource created: source_id=%s cls=%s",
                     self._source_id, type(self).__name__)

    # ------------------------------------------------------------------
    # Abstract lifecycle -- subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource before the first :meth:`read`.

        Called **once** before reading begins. Typical work: open a device
        handle, open a file, connect a socket, allocate buffers. Must be
        idempotent -- a second call is a no-op.

        Raises:
            FrameSourceError: If the resource cannot be acquired and the
                source cannot proceed. The source must leave itself in a
                state where :meth:`close` is still safe to call.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Return ``True`` iff the source is currently open.

        Cheap state probe -- reads an internal flag, no I/O. Distinct from
        :meth:`health_check` (which probes operational readiness, not just
        the open/closed flag).

        Returns:
            ``True`` if :meth:`open` has been called and :meth:`close` has
            not since been called, ``False`` otherwise.
        """

    @abstractmethod
    def read(self) -> "Frame | None":
        """Produce the next :class:`~visioncore.core.frame.Frame`, or ``None``.

        Return-type policy (mandatory): this method returns a
        :class:`~visioncore.core.frame.Frame` instance, **never** a bare
        ``numpy.ndarray`` and **never** an OpenCV ``Mat``. ``None`` signals
        "no frame available right now" -- either a transient empty state
        (camera warming up, brief drop) or permanent exhaustion (file EOF).

        Calling :meth:`read` on a closed source is a contract violation
        and raises :class:`RuntimeError` (distinct from the legitimate
        ``None`` return, which only happens on an open source).

        Returns:
            The next Frame, or ``None`` if the source is open but has no
            frame to deliver right now.

        Raises:
            RuntimeError: If called on a closed source.
            FrameSourceError: For deliberate, recoverable source failures
                (device drop, stream timeout). Callers may recover via
                ``close()`` / ``open()``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the resource acquired in :meth:`open`.

        Called **once** at teardown. Must be idempotent and **must not
        raise** -- a failing ``close`` would mask the real error that
        triggered teardown. Log internally instead.

        The context manager guarantees this is called even if :meth:`read`
        raised. Safe to call on an already-closed or never-opened source.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the source is open and operationally ready.

        Side-effect-free probe. Must not raise; on internal error return
        ``False``. Distinct from :meth:`is_open` (which reports only the
        open/closed flag): ``health_check`` additionally verifies that the
        source can actually deliver frames (e.g. device still connected,
        file still readable, stream still alive).

        Returns:
            ``True`` if the source is open and ready to deliver frames,
            ``False`` otherwise.
        """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        """The stable provenance identifier stamped onto produced frames."""
        return self._source_id

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with source_id and open state.

        Calls :meth:`is_open` to report the lifecycle state. Because
        ``is_open`` is contractually cheap and side-effect-free, this is
        safe to call at any time (e.g. from a debugger).
        """
        try:
            opened: bool = self.is_open()
        except Exception:  # noqa: BLE001 -- defensive: is_open must not raise
            opened = False
        return (
            f"{type(self).__name__}(source_id={self._source_id!r}, "
            f"open={opened})"
        )

    # ------------------------------------------------------------------
    # Context manager -- guaranteed teardown
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameSource":
        """Enter context: open the source and return self.

        Calling ``__enter__`` on an already-open source is a no-op
        (``open`` is idempotent). This lets a source be reused across
        multiple ``with`` blocks.
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit context: close the source. Does not suppress exceptions.

        ``close`` is called regardless of whether the ``with`` body raised.
        The exception (if any) is **not** suppressed -- it propagates after
        teardown completes. This is the same contract as
        :class:`~visioncore.protocol.base.ProtocolAdapter.__exit__`.
        """
        self.close()
        # Returning None (falsy) means exceptions are not suppressed.

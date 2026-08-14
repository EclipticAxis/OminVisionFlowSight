"""DummyFrameSource -- a configurable test source for the pipeline.

Defines :class:`DummyFrameSource`, a concrete
:class:`~visioncore.source.base.FrameSource` that emits predetermined
:class:`~visioncore.core.frame.Frame` instances. It does no real I/O --
no camera, no file, no network. Its sole purpose is to make the source
layer's behaviour **observable and controllable** in tests and demos,
without depending on hardware or media files.

Scope (Milestone C2)
--------------------
C2 delivers **only** :class:`DummyFrameSource`. Real sources
(``CameraFrameSource``, ``FileFrameSource``, ``RtspFrameSource``) arrive
in C3+ as sibling modules under :mod:`visioncore.source`. They will all
subclass :class:`~visioncore.source.base.FrameSource` and obey the same
five-method lifecycle.

What it is for
--------------
* **Pipeline testing** -- feeds a fixed Frame into a Pipeline so the
  downstream stages (detection, tracking) can be tested without a live
  camera. The strategic C3 plan is a ``CaptureStage`` that wraps a
  ``FrameSource`` and calls :meth:`read` to populate
  ``context.frame``; :class:`DummyFrameSource` is the test double for
  that integration.
* **Source-layer testing** -- exercises the :class:`FrameSource` contract
  (open/close/read/health_check, exception recovery) with a deterministic,
  side-effect-free source.
* **Benchmarks / demos** -- a reproducible frame stream with no external
  dependencies.

Return-type policy
------------------
:meth:`read` always returns a :class:`~visioncore.core.frame.Frame`
instance (or ``None`` when exhausted with ``loop=False``) -- never a bare
``numpy.ndarray``. The Frame's ``image`` field does hold an ndarray
internally (that is the data model's design), but the ndarray is always
wrapped in the unified Frame envelope before leaving the source.

numpy usage
-----------
``DummyFrameSource`` lazy-imports numpy **only** inside its synthetic-frame
generator (:meth:`_make_default_frame`) -- i.e. when the caller does not
supply explicit ``frames``. The rest of the module (lifecycle, read loop,
counters) never touches numpy directly. Note that the
:class:`~visioncore.core.frame.Frame` dataclass itself holds an
``np.ndarray`` as its ``image`` field, so importing ``Frame`` (which this
module does at runtime) transitively loads numpy; that is a property of
the core data model (B1), not of this source. The C2 return-type policy
is satisfied regardless: :meth:`read` yields a :class:`Frame` envelope,
never a bare ``ndarray``.

Thread safety
-------------
:class:`DummyFrameSource` is not thread-safe. It is designed for serial
use on a single thread, matching the C1 pipeline's serial execution model.

Example
-------
    >>> from visioncore.source.dummy_source import DummyFrameSource
    >>> src = DummyFrameSource(source_id="cam0")
    >>> with src:
    ...     f = src.read()
    ...     f.source_id, f.frame_id
    ... # close() called automatically
    ('cam0', 0)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from visioncore.core.frame import Frame
from visioncore.source.base import FrameSource

if TYPE_CHECKING:
    pass


__all__ = ["DummyFrameSource"]


logger = logging.getLogger(__name__)


class DummyFrameSource(FrameSource):
    """Configurable FrameSource that emits predetermined Frames.

    :class:`DummyFrameSource` does no real I/O. It either replays a
    caller-supplied list of :class:`~visioncore.core.frame.Frame`
    instances, or auto-generates a single synthetic Frame (a small zeros
    array) when no list is given. It is the standard test double for the
    source layer and for pipeline-level integration tests.

    What it does on each lifecycle method
    --------------------------------------
    * :meth:`open` -- increments :attr:`open_count`; sets :attr:`_open` to
      ``True`` and health to ``True``.
    * :meth:`is_open` -- returns :attr:`_open`.
    * :meth:`read` -- increments :attr:`read_count`; if
      :attr:`raise_on_read` is set, raises it (after incrementing);
      otherwise returns the next Frame from the replay buffer, cycling if
      ``loop=True`` or returning ``None`` when exhausted (``loop=False``).
      Raises :class:`RuntimeError` if called on a closed source.
    * :meth:`close` -- increments :attr:`close_count`; sets :attr:`_open`
      to ``False`` and health to ``False``. Never raises.
    * :meth:`health_check` -- returns the current health flag.

    Exception recovery
    ------------------
    :attr:`raise_on_read` simulates a transient source failure. Set it to
    an exception instance; the next :meth:`read` raises it. Clear it
    (``src.raise_on_read = None``) and call :meth:`close` + :meth:`open`
    to simulate a recovery cycle. This is the pattern tested by the
    "exception recovery" suite.

    Attributes:
        raise_on_read: An exception instance to raise on the next
            :meth:`read` call, or ``None`` (default) to read normally.
            Persists until cleared -- set, read raises, clear, read works.
        open_count: Number of times :meth:`open` was called.
        read_count: Number of times :meth:`read` was called (including
            calls that raised).
        close_count: Number of times :meth:`close` was called.

    Example:
        >>> src = DummyFrameSource(source_id="cam0")
        >>> src.open()
        >>> f = src.read()
        >>> f.source_id
        'cam0'
        >>> (src.open_count, src.read_count, src.close_count)
        (1, 1, 0)
    """

    __slots__ = (
        "_frames",
        "_index",
        "_open",
        "_healthy",
        "raise_on_read",
        "loop",
        "open_count",
        "read_count",
        "close_count",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        frames: list[Frame] | None = None,
        *,
        source_id: str | None = None,
        loop: bool = True,
        raise_on_read: BaseException | None = None,
    ) -> None:
        """Initialise a DummyFrameSource with optional replay frames.

        Parameters:
            frames: Optional list of :class:`~visioncore.core.frame.Frame`
                instances to replay. When ``None`` (default), a single
                synthetic Frame (a 64x64x3 uint8 zeros array) is
                auto-generated via a lazy numpy import. Supplying explicit
                frames avoids the numpy dependency entirely.
            source_id: Provenance identifier stamped onto auto-generated
                frames' ``source_id`` field. When ``None`` (default),
                ``"dummy"`` is used. Ignored when ``frames`` is supplied
                (those frames already carry their own ``source_id``).
            loop: If ``True`` (default), :meth:`read` cycles through the
                frame buffer forever. If ``False``, :meth:`read` returns
                ``None`` after the last frame is consumed (exhaustion).
            raise_on_read: An exception instance to raise on the next
                :meth:`read` call, or ``None`` (default) to read normally.

        Example:
            >>> src = DummyFrameSource(source_id="cam0")
            >>> src.source_id
            'cam0'
            >>> src.loop
            True
        """
        super().__init__(source_id=source_id if source_id is not None else "dummy")
        if frames is not None:
            self._frames: list[Frame] = list(frames)
        else:
            self._frames = [self._make_default_frame(self._source_id)]
        self._index: int = 0
        self._open: bool = False
        self._healthy: bool = False
        self.raise_on_read: BaseException | None = raise_on_read
        self.loop: bool = loop
        self.open_count: int = 0
        self.read_count: int = 0
        self.close_count: int = 0
        logger.debug("DummyFrameSource created: source_id=%s frames=%d loop=%s",
                     self._source_id, len(self._frames), self.loop)

    # ------------------------------------------------------------------
    # Lifecycle (FrameSource contract)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Mark the source open and healthy. Idempotent."""
        self.open_count += 1
        self._open = True
        self._healthy = True
        logger.debug("DummyFrameSource.open: source_id=%s count=%d",
                     self._source_id, self.open_count)

    def is_open(self) -> bool:
        """Return the open/closed flag. Cheap, side-effect-free."""
        return self._open

    def read(self) -> Frame | None:
        """Return the next Frame, or ``None`` when exhausted (``loop=False``).

        If :attr:`raise_on_read` is set, raise it (after incrementing
        :attr:`read_count`). Raises :class:`RuntimeError` if called on a
        closed source.
        """
        if not self._open:
            raise RuntimeError(
                f"read() on a closed source (source_id={self._source_id!r}); "
                f"call open() first"
            )
        self.read_count += 1
        if self.raise_on_read is not None:
            logger.debug("DummyFrameSource.read: source_id=%s RAISING %s",
                         self._source_id, type(self.raise_on_read).__name__)
            raise self.raise_on_read
        if not self._frames:
            return None
        if self._index >= len(self._frames):
            if self.loop:
                self._index = 0
            else:
                return None  # exhausted
        frame: Frame = self._frames[self._index]
        self._index += 1
        logger.debug("DummyFrameSource.read: source_id=%s count=%d frame_id=%s",
                     self._source_id, self.read_count, frame.frame_id)
        return frame

    def close(self) -> None:
        """Mark the source closed and unhealthy. Idempotent, never raises."""
        self.close_count += 1
        self._open = False
        self._healthy = False
        logger.debug("DummyFrameSource.close: source_id=%s count=%d",
                     self._source_id, self.close_count)

    def health_check(self) -> bool:
        """Return the current health flag. Never raises."""
        return self._healthy

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """Current health flag (backed by ``_healthy``)."""
        return self._healthy

    @property
    def frame_buffer_size(self) -> int:
        """Number of frames in the replay buffer."""
        return len(self._frames)

    def reset(self) -> None:
        """Reset counters, read index, and lifecycle state.

        Convenience for tests that reuse a source across multiple open/close
        cycles without re-instantiating. Sets the source back to the
        closed/unhealthy state with zeroed counters and a rewound read
        index. Does **not** alter :attr:`raise_on_read`, :attr:`loop`, or
        the frame buffer.
        """
        self._open = False
        self._healthy = False
        self._index = 0
        self.open_count = 0
        self.read_count = 0
        self.close_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_default_frame(source_id: str) -> Frame:
        """Build a small synthetic Frame for auto-generation mode.

        Lazy-imports numpy so that this module does not hard-require numpy
        at import time -- only when auto-generation is actually needed
        (i.e. the caller did not supply explicit ``frames``).

        Parameters:
            source_id: Stamped onto the Frame's ``source_id`` field.

        Returns:
            A frozen :class:`~visioncore.core.frame.Frame` wrapping a
            64x64x3 uint8 zeros array (a minimal, cheap placeholder image
            suitable for pipeline plumbing tests).
        """
        import numpy as np  # lazy: only needed for synthetic generation
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        return Frame(
            frame_id=0,
            timestamp=0.0,
            source_id=source_id,
            image=image,
        )

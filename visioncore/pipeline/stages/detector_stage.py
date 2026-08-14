"""DetectorStage -- the detection stage for the VisionCore pipeline.

This module defines three cohesive concerns, all in one file per the C3
specification:

* :class:`Detector` -- the abstract detector **interface** that
  :class:`DetectorStage` depends on. Concrete detectors (a future
  YOLO/RT-DETR/... wrapper, or :class:`DummyDetector` below) implement
  this interface; the stage never depends on a concrete class.
* :class:`DummyDetector` -- a test double that returns predetermined
  :class:`~visioncore.core.detection.Detection` instances, with no model
  and no I/O.
* :class:`DetectorStage` -- a :class:`~visioncore.pipeline.base.PipelineStage`
  that wraps a :class:`Detector`, reads ``context.frame``, calls
  ``detector.detect(frame)``, and writes the results to
  ``context.detections``.

Scope (Milestone C3)
--------------------
C3 delivers **only** the detector stage + its interface + a test double.
It deliberately does **not** contain:

* any YOLO / RT-DETR / GroundingDINO / SAM code or imports -- the stage
  depends on the abstract :class:`Detector` interface only, never on a
  concrete model family;
* any model loading, GPU allocation, or ONNX/cv2/ultralytics code;
* any modification to ``ai/``, ``camera/``, or ``gui/`` -- the existing
  ``ai/inference.py`` detection path continues to run unchanged.

Interface-first design
----------------------
:class:`DetectorStage` is constructed with a :class:`Detector` (the
interface), not a concrete class. This is the dependency-inversion that
keeps the stage decoupled from any specific detection backend: the stage
knows *that* a detector detects, not *how* it detects. Real detector
backends (C4+) arrive as :class:`Detector` subclasses; the stage never
changes.

Data flow
---------
::

    context.frame ──> DetectorStage.process ──> detector.detect(frame)
                                                      │
                                          list[Detection]
                                                      │
                                                      ▼
                                            context.detections (extend)

If ``context.frame`` is ``None`` (e.g. a source produced no frame this
round), the stage skips detection and leaves ``context.detections``
untouched -- a ``None`` frame is a legitimate "nothing to do" signal,
not an error.

Thread safety
-------------
:class:`DetectorStage` is not thread-safe. It is designed for the C1
pipeline's serial execution model. Concrete detectors that share GPU
resources across stages must serialise access externally.

Example
-------
    >>> from visioncore.pipeline.stages.detector_stage import (
    ...     DetectorStage, DummyDetector,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> from visioncore.core.frame import Frame
    >>> import numpy as np
    >>> stage = DetectorStage(DummyDetector())
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> ctx.frame = Frame(0, 0.0, "cam0", np.zeros((4, 4, 3), dtype=np.uint8))
    >>> stage.process(ctx)
    >>> len(ctx.detections)
    1
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from visioncore.core.detection import BBox, Detection
from visioncore.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from visioncore.core.frame import Frame
    from visioncore.pipeline.context import PipelineContext


__all__ = ["Detector", "DetectorError", "DummyDetector", "DetectorStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class DetectorError(RuntimeError):
    """Structured exception signalling a recoverable detector failure.

    Detectors *may* raise ``DetectorError`` (or any subclass) to indicate
    a failure specific to the detector's domain -- e.g. a model produced
    NaN scores, an inference call timed out, a backend reported an OOM.

    Raising ``DetectorError`` instead of a bare ``RuntimeError`` lets
    upstream callers distinguish "the detector deliberately reported a
    failure" from "an unexpected bug crashed the detector". The enclosing
    :class:`~visioncore.pipeline.pipeline.Pipeline` does **not** catch
    ``DetectorError`` specially -- it propagates out of ``run()`` exactly
    like any other exception, per the C1 "honest exceptions" contract.
    """


# ======================================================================
# Detector interface (abstract)
# ======================================================================

class Detector(ABC):
    """Abstract detector interface -- the contract every detector obeys.

    A :class:`Detector` consumes a :class:`~visioncore.core.frame.Frame`
    and produces a list of :class:`~visioncore.core.detection.Detection`
    instances. It owns its own resources (model weights, GPU sessions,
    ONNX sessions) and exposes a four-method lifecycle that
    :class:`DetectorStage` delegates to.

    The interface is **model-agnostic**: it says nothing about YOLO,
    RT-DETR, GroundingDINO, SAM, or any other concrete architecture.
    Concrete detectors implement ``detect`` however they choose; the
    stage and the pipeline never inspect the mechanism.

    Lifecycle contract
    ------------------
    * ``initialize()`` -- acquire resources (load model weights, warm up
      sessions). Idempotent.
    * ``detect(frame)`` -- run detection on one frame, return a list of
      :class:`~visioncore.core.detection.Detection`.
    * ``shutdown()`` -- release resources. Idempotent, must not raise.
    * ``health_check()`` -- side-effect-free probe; ``True`` iff ready.
      Must not raise.

    Subclassing
    -----------
    Subclasses **must** implement all four abstract methods.
    """

    __slots__ = ()

    @abstractmethod
    def initialize(self) -> None:
        """Acquire resources before the first :meth:`detect` call.

        Typical work: load model weights, create an ONNX/TensorRT session,
        warm up allocations. Must be idempotent -- a second call is a
        no-op.

        Raises:
            DetectorError: If initialisation fails and the detector cannot
                proceed. :meth:`shutdown` must still be safe to call.
        """

    @abstractmethod
    def detect(self, frame: "Frame") -> list[Detection]:
        """Run detection on ``frame``, returning a list of Detections.

        The returned list may be empty (no objects found). Detections are
        immutable value objects; callers may freely retain references.

        Parameters:
            frame: The :class:`~visioncore.core.frame.Frame` to detect on.
                Must not be ``None`` -- a ``None`` frame never reaches
                :meth:`detect` because :class:`DetectorStage` skips when
                ``context.frame`` is ``None``.

        Returns:
            A list of :class:`~visioncore.core.detection.Detection`
            instances, possibly empty. Order is detector-defined (e.g. by
            score).

        Raises:
            DetectorError: For deliberate, recoverable detector failures.
            Exception: Any other exception propagates out of the stage's
                ``process()`` unchanged.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`initialize`.

        Idempotent and must not raise. Releases model sessions, GPU
        memory, file handles. The enclosing stage guarantees this is
        called even if :meth:`detect` raised.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the detector is initialised and ready.

        Side-effect-free. Must not raise; on internal error return
        ``False``.
        """


# ======================================================================
# DummyDetector -- test double
# ======================================================================

class DummyDetector(Detector):
    """Configurable test detector that returns predetermined Detections.

    :class:`DummyDetector` does no real inference -- no model, no GPU, no
    I/O. It returns a fixed list of
    :class:`~visioncore.core.detection.Detection` instances on every
    :meth:`detect` call, making the detector's behaviour **observable and
    controllable** in tests.

    What it is for
    --------------
    * **Stage testing** -- exercise :class:`DetectorStage`'s lifecycle
      delegation and data flow without a real model.
    * **Pipeline integration** -- feed deterministic detections into a
      pipeline to test downstream tracking / projection stages.
    * **Benchmarks** -- measure stage overhead with zero inference cost.

    Attributes:
        raise_on_detect: An exception instance to raise on the next
            :meth:`detect` call, or ``None`` (default) to detect normally.
            Persists until cleared -- set, detect raises, clear, detect
            works. Used by exception-propagation tests.
        initialize_count: Number of times :meth:`initialize` was called.
        detect_count: Number of times :meth:`detect` was called.
        shutdown_count: Number of times :meth:`shutdown` was called.

    Example:
        >>> det = DummyDetector()
        >>> det.initialize()
        >>> import numpy as np
        >>> from visioncore.core.frame import Frame
        >>> dets = det.detect(Frame(0, 0.0, "s", np.zeros((4, 4, 3), dtype=np.uint8)))
        >>> len(dets)
        1
        >>> dets[0].class_name
        'person'
    """

    __slots__ = (
        "_detections",
        "_initialized",
        "raise_on_detect",
        "initialize_count",
        "detect_count",
        "shutdown_count",
    )

    def __init__(
        self,
        detections: list[Detection] | None = None,
        *,
        raise_on_detect: BaseException | None = None,
    ) -> None:
        """Initialise with an optional list of Detections to replay.

        Parameters:
            detections: The list of
                :class:`~visioncore.core.detection.Detection` instances to
                return from :meth:`detect`. When ``None`` (default), a
                single synthetic Detection (a centred person at score
                0.9) is generated.
            raise_on_detect: An exception instance to raise on the next
                :meth:`detect` call, or ``None`` (default) to detect
                normally.
        """
        if detections is not None:
            self._detections: list[Detection] = list(detections)
        else:
            self._detections = [
                Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, "person"),
            ]
        self._initialized: bool = False
        self.raise_on_detect: BaseException | None = raise_on_detect
        self.initialize_count: int = 0
        self.detect_count: int = 0
        self.shutdown_count: int = 0
        logger.debug("DummyDetector created: detections=%d",
                     len(self._detections))

    # ------------------------------------------------------------------
    # Lifecycle (Detector contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the detector initialised. Idempotent."""
        self.initialize_count += 1
        self._initialized = True
        logger.debug("DummyDetector.initialize: count=%d",
                     self.initialize_count)

    def detect(self, frame: "Frame") -> list[Detection]:
        """Return a shallow copy of the predetermined Detections.

        Returns a fresh list (so callers may mutate it without affecting
        the detector's internal buffer) containing the same Detection
        instances (which are frozen/immutable, so sharing is safe).

        If :attr:`raise_on_detect` is set, raise it instead (after
        incrementing :attr:`detect_count`).
        """
        self.detect_count += 1
        if self.raise_on_detect is not None:
            logger.debug("DummyDetector.detect: RAISING %s",
                         type(self.raise_on_detect).__name__)
            raise self.raise_on_detect
        logger.debug("DummyDetector.detect: count=%d returning %d",
                     self.detect_count, len(self._detections))
        return list(self._detections)

    def shutdown(self) -> None:
        """Mark the detector shut down. Idempotent, never raises."""
        self.shutdown_count += 1
        self._initialized = False
        logger.debug("DummyDetector.shutdown: count=%d",
                     self.shutdown_count)

    def health_check(self) -> bool:
        """Return the initialised flag. Never raises."""
        return self._initialized

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        """Whether :meth:`initialize` has been called and not undone."""
        return self._initialized

    @property
    def detection_count(self) -> int:
        """Number of Detections returned per ``detect`` call."""
        return len(self._detections)

    def reset(self) -> None:
        """Reset counters and state. Preserves the detection buffer."""
        self._initialized = False
        self.initialize_count = 0
        self.detect_count = 0
        self.shutdown_count = 0


# ======================================================================
# DetectorStage -- the pipeline stage
# ======================================================================

class DetectorStage(PipelineStage):
    """Pipeline stage that wraps a :class:`Detector` and detects on frames.

    :class:`DetectorStage` is a thin adapter from the
    :class:`Detector` interface to the
    :class:`~visioncore.pipeline.base.PipelineStage` interface. It owns no
    detection logic of its own -- it delegates everything to the wrapped
    detector. This is the dependency-inversion that keeps the pipeline
    decoupled from any specific detection backend (YOLO / RT-DETR /
    GroundingDINO / SAM / ...): the stage depends on the **interface**,
    concrete detectors are injected at construction.

    Data flow
    ---------
    On :meth:`process`:

    1. Read ``context.frame``.
    2. If ``None``, skip (log debug) -- a missing frame is a legitimate
       "nothing to do" signal, not an error.
    3. Otherwise call ``detector.detect(frame)``.
    4. Extend ``context.detections`` with the returned list.

    The stage **extends** (not replaces) ``context.detections``, so a
    pipeline may chain multiple detector stages (e.g. a person detector
    followed by a head detector) and their outputs accumulate. For a
    single-detector pipeline on a fresh context, extend is equivalent to
    replace (the list starts empty).

    Lifecycle delegation
    --------------------
    * :meth:`initialize` -> ``detector.initialize()``
    * :meth:`process` -> ``detector.detect(context.frame)``
    * :meth:`shutdown` -> ``detector.shutdown()``
    * :meth:`health_check` -> ``detector.health_check()``

    Attributes:
        _detector: The wrapped :class:`Detector` instance.

    Example:
        >>> stage = DetectorStage(DummyDetector(), name="detect")
        >>> stage.name
        'detect'
    """

    __slots__ = ("_detector",)

    def __init__(self, detector: Detector, *, name: str | None = None) -> None:
        """Construct a DetectorStage wrapping ``detector``.

        Parameters:
            detector: The :class:`Detector` instance to wrap. Must not be
                ``None`` and must be a :class:`Detector` (the interface is
                enforced, not a concrete model class). This is the
                dependency-inversion point: the stage depends on the
                interface, the concrete detector is injected.
            name: Optional stage name (defaults to ``"DetectorStage"``).
                Useful when a pipeline chains multiple detector stages.

        Raises:
            TypeError: If ``detector`` is ``None`` or not a
                :class:`Detector`.

        Example:
            >>> stage = DetectorStage(DummyDetector())
            >>> stage.health_check()
            False
        """
        super().__init__(name=name if name is not None else "DetectorStage")
        if detector is None or not isinstance(detector, Detector):
            raise TypeError(
                f"detector must be a Detector instance, got "
                f"{type(detector).__name__ if detector is not None else 'None'}"
            )
        self._detector: Detector = detector
        logger.debug("DetectorStage created: name=%s detector=%s",
                     self._name, type(detector).__name__)

    # ------------------------------------------------------------------
    # PipelineStage lifecycle (delegated to the detector)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Delegate to ``detector.initialize()``."""
        self._detector.initialize()
        logger.debug("DetectorStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Read ``context.frame``, detect, extend ``context.detections``.

        If ``context.frame`` is ``None``, skip detection (the source
        produced no frame this round). Otherwise call
        ``detector.detect(frame)`` and extend ``context.detections`` with
        the result.
        """
        frame = context.frame
        if frame is None:
            logger.debug(
                "DetectorStage.process: name=%s frame is None, skipping",
                self._name,
            )
            return
        detections: list[Detection] = self._detector.detect(frame)
        context.detections.extend(detections)
        logger.debug(
            "DetectorStage.process: name=%s detected %d (total=%d)",
            self._name, len(detections), len(context.detections),
        )

    def shutdown(self) -> None:
        """Delegate to ``detector.shutdown()``."""
        self._detector.shutdown()
        logger.debug("DetectorStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Delegate to ``detector.health_check()``."""
        return self._detector.health_check()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def detector(self) -> Detector:
        """The wrapped :class:`Detector` instance."""
        return self._detector

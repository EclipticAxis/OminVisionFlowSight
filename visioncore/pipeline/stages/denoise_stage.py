"""DenoiseStage -- the denoising stage for the VisionCore pipeline.

This module defines :class:`DenoiseStage`, the first concrete
:class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
plugin. It represents the **architectural placeholder** for frame
denoising in the pipeline: a stage that sits before detection and
refines the raw frame to improve downstream detection quality.

Scope (Milestone D2)
--------------------
D2 delivers **only** the stage shell -- no actual denoising algorithm.
The stage passes ``context.frame`` through unchanged. The constructor
accepts an optional ``denoiser`` instance (the future denoising backend),
but it is held and not invoked. Real denoising backends (bilateral
filter, non-local means, learned denoiser, ...) arrive in later
milestones as the ``denoiser`` object's concrete implementation; the
stage itself never changes.

D2 deliberately does **not** contain:

* any OpenCV denoising call (``cv2.fastNlMeansDenoising``,
  ``cv2.bilateralFilter``, ...);
* any model loading, GPU allocation, or ONNX/torch code;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Dependency-inversion
--------------------
Like :class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`
(which wraps a :class:`~visioncore.pipeline.stages.detector_stage.Detector`),
:class:`DenoiseStage` wraps a denoiser object. The stage knows *that* a
denoiser denoises, not *how* it denoises. The denoiser parameter is
typed as ``Any`` -- a future milestone will define a ``Denoiser`` ABC
and tighten the annotation. For now, ``None`` is a valid "no denoiser"
value (the stage becomes a pure passthrough).

Data flow
---------
::

    context.frame ──> DenoiseStage.process ──> context.frame (unchanged)

The stage **always** runs -- even when ``context.frame`` is ``None``
(a ``None`` frame is a legitimate "nothing to do" signal, not an error).
When a real denoiser is injected, the data flow becomes::

    context.frame ──> denoiser.denoise(frame) ──> context.frame (replaced)

Thread safety
-------------
:class:`DenoiseStage` is not thread-safe. It is designed for the
pipeline's serial execution model.

Example
-------
    >>> from visioncore.pipeline.stages.denoise_stage import DenoiseStage
    >>> from visioncore.pipeline.context import PipelineContext
    >>> from visioncore.core.frame import Frame
    >>> import numpy as np
    >>> stage = DenoiseStage(name="pre-detect-denoise")
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> ctx.frame = Frame(0, 0.0, "cam0", np.zeros((4, 4, 3), dtype=np.uint8))
    >>> stage.process(ctx)
    >>> ctx.frame.frame_id
    0
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from visioncore.pipeline.stages.advanced import (
    AdvancedStage,
    StageCapability,
)

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["DenoiseStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Module-level capability (single source of truth)
# ======================================================================

_CAPABILITY = StageCapability(
    name="denoise",
    version="0.1.0",
    description=(
        "Denoising stage placeholder: passes context.frame through "
        "unchanged until a real denoiser backend is injected."
    ),
    required_context=["frame"],
    provided_context=[],
)


# ======================================================================
# DenoiseStage
# ======================================================================

class DenoiseStage(AdvancedStage):
    """Pipeline stage that wraps a denoiser and refines ``context.frame``.

    :class:`DenoiseStage` is a thin adapter from a denoiser backend to
    the :class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
    interface. In D2 it is a **pure passthrough**: ``process()`` calls
    :meth:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage.check_context`
    and leaves ``context.frame`` untouched. When a real denoiser is
    injected in a future milestone, the stage will call
    ``denoiser.denoise(frame)`` and replace ``context.frame`` with the
    result.

    Attributes:
        _denoiser: The wrapped denoiser instance, or ``None`` for a
            pure passthrough.

    Example:
        >>> stage = DenoiseStage(name="denoise")
        >>> stage.name
        'denoise'
        >>> stage.capability.name
        'denoise'
    """

    __slots__ = ("_denoiser", "_ready")

    def __init__(
        self,
        denoiser: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a DenoiseStage with an optional denoiser backend.

        Parameters:
            denoiser: The denoiser instance to wrap. ``None`` (default)
                means no denoiser is available -- the stage acts as a
                pure passthrough. Any object is accepted; a future
                milestone will tighten the type to a ``Denoiser`` ABC.
            name: Optional stage name (defaults to ``"DenoiseStage"``).

        Example:
            >>> s = DenoiseStage(name="pre-detect")
            >>> s.name
            'pre-detect'
            >>> s.denoiser is None
            True
        """
        super().__init__(name=name if name is not None else "DenoiseStage")
        self._denoiser: Any | None = denoiser
        self._ready: bool = False
        logger.debug("DenoiseStage created: name=%s denoiser=%s",
                     self._name, type(denoiser).__name__ if denoiser is not None else "None")

    # ------------------------------------------------------------------
    # Capability declaration
    # ------------------------------------------------------------------

    @property
    def capability(self) -> StageCapability:
        """The stage's declared :class:`StageCapability`.

        ``required_context=["frame"]``, ``provided_context=[]`` (D2
        passthrough writes nothing).
        """
        return _CAPABILITY

    # ------------------------------------------------------------------
    # Lifecycle (AdvancedStage contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("DenoiseStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Pass ``context.frame`` through unchanged (D2 passthrough).

        Calls :meth:`AdvancedStage.check_context` first to validate the
        declared ``required_context=["frame"]`` contract, then returns
        without mutating ``context.frame``. When a real denoiser is
        injected, this method will call ``denoiser.denoise(frame)`` and
        assign the result back to ``context.frame``.

        Parameters:
            context: The shared
                :class:`~visioncore.pipeline.context.PipelineContext`.
                ``context.frame`` is read and left unchanged.
        """
        self.check_context(context)
        logger.debug("DenoiseStage.process: name=%s frame=%s (passthrough)",
                     self._name,
                     context.frame.frame_id if context.frame is not None else None)

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("DenoiseStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def denoiser(self) -> Any | None:
        """The wrapped denoiser instance, or ``None``."""
        return self._denoiser

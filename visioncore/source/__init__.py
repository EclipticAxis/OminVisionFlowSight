"""VisionCore source layer -- the unified input-source abstraction.

This package hosts the *source layer* of VisionCore: an abstraction over
where :class:`~visioncore.core.frame.Frame` instances come from. A
:class:`FrameSource` is the origin of the data that flows through a
:class:`~visioncore.pipeline.pipeline.Pipeline`; a future capture stage
(C3) will wrap a source and call ``read()`` to populate
``context.frame``.

Scope (Milestone C2 -- FrameSource Abstraction)
-----------------------------------------------
C2 delivers the **contract + one test double + a factory**:

    FrameSource        -- abstract base class (the five-method lifecycle).
    FrameSourceError   -- structured exception for recoverable failures.
    DummyFrameSource   -- configurable test source that emits fixed Frames.
    create_frame_source / register_frame_source -- config-driven factory.

It deliberately does **not** contain:

* any camera, file, or network I/O (that is C3+, as sibling modules);
* any numpy or OpenCV capture code in the base class -- concrete sources
  implement capture, the base only defines the contract;
* any modification to ``ai/``, ``camera/``, or ``gui/`` -- the existing
  VisionDataPlatform capture stack continues to run unchanged.

Return-type policy (mandatory)
------------------------------
:meth:`FrameSource.read` returns ``Frame | None`` -- **never** a bare
``numpy.ndarray`` and **never** an OpenCV ``Mat``. The
:class:`~visioncore.core.frame.Frame` dataclass is the unified envelope:
it wraps the raw pixel array together with identity, provenance, and
timing. Returning the envelope keeps provenance attached to the pixels
and lets every downstream stage rely on a single, stable type.

Design summary
--------------
* **Contract first**: :class:`FrameSource` is an ABC with five abstract
  methods (open / is_open / read / close / health_check); concrete
  sources implement, the base does not extend.
* **Unified envelope**: ``read()`` yields :class:`~visioncore.core.Frame`,
  never a raw array -- provenance and timing travel with the pixels.
* **Context manager**: ``__enter__`` opens, ``__exit__`` closes, guaranteeing
  teardown even on exception (same contract as
  :class:`~visioncore.protocol.base.ProtocolAdapter`).
* **Factory + registry**: sources register under a string kind; callers
  create by name without importing the concrete class, decoupling config
  from transport-specific imports.
* **Honest exceptions**: ``read()`` raises :class:`RuntimeError` on a
  closed source (contract violation) and :class:`FrameSourceError` on a
  recoverable failure; ``None`` is reserved for "open but empty/exhausted".

Quick start::

    from visioncore.source import create_frame_source

    src = create_frame_source("dummy", source_id="cam0")
    with src:
        frame = src.read()
    # close() called automatically on exit

Future milestones (C3+)
-----------------------
C3 adds real sources (``CameraFrameSource``, ``FileFrameSource``) as
sibling modules, plus a ``CaptureStage`` that wraps a source and feeds a
:class:`~visioncore.pipeline.context.PipelineContext`. Each real source
is the only module that imports its transport library; the factory keeps
that coupling confined.
"""

from __future__ import annotations

from visioncore.source.base import FrameSource, FrameSourceError
from visioncore.source.dummy_source import DummyFrameSource
from visioncore.source.frame_source import (
    available_frame_sources,
    create_frame_source,
    get_frame_source_class,
    register_frame_source,
)

__all__ = [
    # ---- Core contract ----
    "FrameSource",
    "FrameSourceError",
    # ---- Built-in sources ----
    "DummyFrameSource",
    # ---- Factory / registry ----
    "create_frame_source",
    "register_frame_source",
    "available_frame_sources",
    "get_frame_source_class",
]

"""Vision processors -- post-processing plugins (Milestone D9.3).

This package hosts *processor plugins*: ``PluginInterface`` subclasses
that transform detection results after the core detect → track → target
pipeline. Processors are registered in a
:class:`~visioncore.plugin.registry.PluginRegistry` and composed
configurably.

Example::

    from visioncore.plugin.processors import FilterProcessor, BBoxProcessor
    fp = FilterProcessor()
    bp = BBoxProcessor()
"""

from __future__ import annotations

from visioncore.plugin.processors.processor import (
    BBoxProcessor,
    FilterProcessor,
    ProcessorPlugin,
)

__all__ = [
    "ProcessorPlugin",
    "FilterProcessor",
    "BBoxProcessor",
]

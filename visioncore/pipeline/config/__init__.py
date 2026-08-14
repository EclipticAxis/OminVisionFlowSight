"""Pipeline configuration -- config-driven stage loading (Milestone D8).

This package provides :class:`PipelineConfig`, which reads a
JSON / YAML configuration file and builds a fully-configured
:class:`~visioncore.pipeline.pipeline.Pipeline` with stages registered in
the specified order.

Example::

    from visioncore.pipeline.config import PipelineConfig
    pipeline = PipelineConfig.build_from_file("config/visioncore_pipeline.json")
"""

from __future__ import annotations

from visioncore.pipeline.config.pipeline_config import (
    BUILTIN_STAGES,
    PipelineConfig,
)

__all__ = [
    "PipelineConfig",
    "BUILTIN_STAGES",
]

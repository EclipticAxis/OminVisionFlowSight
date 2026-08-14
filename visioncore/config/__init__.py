"""VisionCore project configuration (Milestone D8).

Simple project-level settings for the VisionCore pipeline. Provides
``pipeline_enabled`` (flag) and ``pipeline_config_path`` (default
config path) so that ``PipelineConfig`` and application code can
query the project's pipeline configuration without hardcoding paths.

Example
-------
    >>> from visioncore.config import settings
    >>> settings.pipeline_enabled
    True
    >>> settings.pipeline_config_path is None or isinstance(
    ...     settings.pipeline_config_path, str)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PipelineSettings", "settings"]


@dataclass
class PipelineSettings:
    """Project-level pipeline settings.

    Attributes:
        pipeline_enabled: When ``False``, pipeline execution is
            skipped and ``PipelineConfig.build()`` returns an empty
            pipeline. Defaults to ``True``.
        pipeline_config_path: Default path to the pipeline config file.
            ``None`` means no default (caller must provide the path
            explicitly).
    """

    pipeline_enabled: bool = True
    pipeline_config_path: str | None = None


settings: PipelineSettings = PipelineSettings()

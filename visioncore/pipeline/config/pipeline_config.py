"""Pipeline configuration loader (Milestone D8).

Reads a JSON or YAML config file that specifies pipeline stages and their
parameters, resolves each stage name to a :class:`PipelineStage` class via
a built-in registry and/or an external
:class:`~visioncore.plugin.registry.PluginRegistry`, and returns a
fully-configured :class:`~visioncore.pipeline.pipeline.Pipeline`.

Scope (Milestone D8)
--------------------
D8 delivers **config-driven pipeline assembly**. It deliberately does
**not** contain:

* any concrete stage implementation (those live in their own modules);
* any networking, ROS2, or MAVLink transport code;
* any modification to ``ai/``, ``gui/``, or ``camera/``.

Example config (JSON)
---------------------
::

    {
      "stages": [
        {"type": "DummyStage", "params": {"name": "capture"}},
        {"type": "DenoiseStage", "params": {"name": "denoise"}},
        {"type": "DummyStage", "params": {"name": "detect"}}
      ]
    }

Example
-------
    >>> import json, tempfile, pathlib
    >>> cfg = {"stages": [{"type": "DummyStage", "params": {"name": "s1"}}]}
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "p.json"
    >>> _ = p.write_text(json.dumps(cfg))
    >>> pipeline = PipelineConfig.build_from_file(str(p))
    >>> [s.name for s in pipeline.stages]
    ['s1']
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from visioncore.pipeline.base import PipelineStage

__all__ = ["PipelineConfig", "BUILTIN_STAGES", "PipelineConfigError"]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class PipelineConfigError(RuntimeError):
    """Structured exception signalling a pipeline-configuration failure.

    Raised by :meth:`PipelineConfig.build_from_file` when the config
    file cannot be read, parsed, or resolved.

    Example:
        >>> raise PipelineConfigError("file not found")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        visioncore.pipeline.config.pipeline_config.PipelineConfigError: file not found
    """


# ======================================================================
# Built-in stage registry
# ======================================================================

def _get_builtin_stages() -> dict[str, type[PipelineStage]]:
    """Build the built-in stage registry on first call.

    Returns a dict mapping stage type names to their classes. Stages
    are imported lazily so the registry only loads what it needs.
    """
    from visioncore.pipeline.stage import DummyStage

    builtin: dict[str, type[PipelineStage]] = {
        "DummyStage": DummyStage,
    }

    # Advanced-stage plugins (D2-D5)
    try:
        from visioncore.pipeline.stages.denoise_stage import DenoiseStage
        builtin["DenoiseStage"] = DenoiseStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.redetect_stage import RedetectStage
        builtin["RedetectStage"] = RedetectStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.attribute_gesture_stage import (
            AttributeStage, GestureStage,
        )
        builtin["AttributeStage"] = AttributeStage
        builtin["GestureStage"] = GestureStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.rectangle_filter_stage import (
            RectangleStage, FilterStage, HealthStage,
        )
        builtin["RectangleStage"] = RectangleStage
        builtin["FilterStage"] = FilterStage
        builtin["HealthStage"] = HealthStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.detector_stage import DetectorStage
        builtin["DetectorStage"] = DetectorStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.tracker_stage import TrackerStage
        builtin["TrackerStage"] = TrackerStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.target_stage import TargetStage
        builtin["TargetStage"] = TargetStage
    except ImportError:
        pass

    try:
        from visioncore.pipeline.stages.event_stage import EventStage
        builtin["EventStage"] = EventStage
    except ImportError:
        pass

    return builtin


# Module-level cache for the built-in registry.
_BUILTIN_STAGES: dict[str, type[PipelineStage]] | None = None


def _builtin_stages() -> dict[str, type[PipelineStage]]:
    """Return the cached built-in stage registry (lazy init)."""
    global _BUILTIN_STAGES  # noqa: PLW0603
    if _BUILTIN_STAGES is None:
        _BUILTIN_STAGES = _get_builtin_stages()
    return _BUILTIN_STAGES


# Public constant for inspection / tests.
BUILTIN_STAGES: dict[str, type[PipelineStage]] = _builtin_stages()


# ======================================================================
# PipelineConfig
# ======================================================================

class PipelineConfig:
    """Config-driven pipeline builder.

    :class:`PipelineConfig` reads a JSON / YAML config file that
    specifies an ordered list of stages, resolves each stage's ``type``
    name to a :class:`PipelineStage` class via the built-in registry
    and / or an external
    :class:`~visioncore.plugin.registry.PluginRegistry`, and assembles
    a fully-configured :class:`~visioncore.pipeline.pipeline.Pipeline`.

    Example:
        >>> import json, tempfile, pathlib
        >>> cfg = {"stages": [{"type": "DummyStage", "params": {"name": "s1"}}]}
        >>> p = pathlib.Path(tempfile.mkdtemp()) / "p.json"
        >>> _ = p.write_text(json.dumps(cfg))
        >>> pipeline = PipelineConfig.build_from_file(str(p))
        >>> [s.name for s in pipeline.stages]
        ['s1']
    """

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_file(path: str) -> dict[str, Any]:
        """Read and parse a JSON or YAML config file.

        Parameters:
            path: Path to the config file (``.json`` or ``.yaml`` /
                ``.yml``).

        Returns:
            The parsed config dict.

        Raises:
            PipelineConfigError: If the file is missing, unreadable,
                or contains invalid JSON / YAML.
        """
        p = Path(path)
        if not p.exists():
            raise PipelineConfigError(f"config file not found: {path}")
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineConfigError(
                f"config file unreadable: {path}: {type(exc).__name__}: {exc}"
            ) from exc

        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise PipelineConfigError(
                    "YAML config requires the 'pyyaml' package: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                data: dict[str, Any] = yaml.safe_load(text)
            except Exception as exc:  # noqa: BLE001
                raise PipelineConfigError(
                    f"invalid YAML in {path}: {type(exc).__name__}: {exc}"
                ) from exc
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PipelineConfigError(
                    f"invalid JSON in {path}: {exc}"
                ) from exc
        else:
            raise PipelineConfigError(
                f"unsupported config format: {suffix!r} "
                f"(expected .json, .yaml, or .yml)"
            )

        if not isinstance(data, dict):
            raise PipelineConfigError(
                f"config root must be a dict, got {type(data).__name__}"
            )
        logger.debug("PipelineConfig.load_file: loaded %s (%d keys)",
                     path, len(data))
        return data

    # ------------------------------------------------------------------
    # Stage resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_stage(
        stage_type: str,
        extra_stages: dict[str, type[PipelineStage]] | None = None,
        plugin_registry: "PluginRegistry | None" = None,
    ) -> type[PipelineStage]:
        """Resolve a stage type name to a PipelineStage class.

        Resolution order:
        1. ``extra_stages`` (if provided)
        2. ``BUILTIN_STAGES``
        3. ``plugin_registry`` (if provided) — returns
           ``type(plugin_instance)``, but only if the plugin IS a
           PipelineStage subclass.

        Parameters:
            stage_type: The stage type name (e.g. ``"DenoiseStage"``).
            extra_stages: Additional stage classes keyed by name.
            plugin_registry: A :class:`visioncore.plugin.registry.PluginRegistry`
                to search for plugin-backed stages.

        Returns:
            The resolved :class:`PipelineStage` class.

        Raises:
            PipelineConfigError: If the stage type cannot be resolved.
        """
        if extra_stages and stage_type in extra_stages:
            return extra_stages[stage_type]
        if stage_type in BUILTIN_STAGES:
            return BUILTIN_STAGES[stage_type]
        if plugin_registry is not None:
            plugin = plugin_registry.get_plugin(stage_type)
            if plugin is not None and isinstance(plugin, PipelineStage):
                return type(plugin)
        raise PipelineConfigError(
            f"unknown stage type: {stage_type!r}"
        )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_config(
        cls,
        config: dict[str, Any],
        extra_stages: dict[str, type[PipelineStage]] | None = None,
        plugin_registry: "PluginRegistry | None" = None,
    ) -> "Pipeline":
        """Build a Pipeline from a parsed config dict.

        Parameters:
            config: Parsed config (must contain ``"stages"`` list).
            extra_stages: Additional stage classes keyed by name.
            plugin_registry: Optional plugin registry for resolution.

        Returns:
            A configured :class:`~visioncore.pipeline.pipeline.Pipeline`
            with stages added in config order.

        Raises:
            PipelineConfigError: On invalid config structure or unknown
                stage types.
        """
        from visioncore.pipeline.pipeline import Pipeline

        # Check pipeline_enabled flag.
        try:
            from visioncore.config import settings
            if not settings.pipeline_enabled:
                logger.info("PipelineConfig: pipeline_enabled=False, "
                            "returning empty pipeline")
                return Pipeline()
        except ImportError:
            pass

        if not isinstance(config, dict):
            raise PipelineConfigError(
                f"config must be a dict, got {type(config).__name__}"
            )
        stages_cfg: Any = config.get("stages")
        if stages_cfg is None:
            raise PipelineConfigError(
                "config missing required 'stages' key"
            )
        if not isinstance(stages_cfg, list):
            raise PipelineConfigError(
                f"'stages' must be a list, got {type(stages_cfg).__name__}"
            )

        pipeline = Pipeline()
        for i, entry in enumerate(stages_cfg):
            if not isinstance(entry, dict):
                raise PipelineConfigError(
                    f"stages[{i}]: must be a dict, got "
                    f"{type(entry).__name__}"
                )
            stage_type: Any = entry.get("type")
            if stage_type is None:
                raise PipelineConfigError(
                    f"stages[{i}]: missing required 'type' key"
                )
            if not isinstance(stage_type, str):
                raise PipelineConfigError(
                    f"stages[{i}].type: must be a string, got "
                    f"{type(stage_type).__name__}"
                )

            params: dict[str, Any] = entry.get("params", {})
            if not isinstance(params, dict):
                raise PipelineConfigError(
                    f"stages[{i}].params: must be a dict, got "
                    f"{type(params).__name__}"
                )

            stage_cls: type[PipelineStage] = cls.resolve_stage(
                stage_type,
                extra_stages=extra_stages,
                plugin_registry=plugin_registry,
            )
            try:
                stage_instance: PipelineStage = stage_cls(**params)
            except TypeError as exc:
                raise PipelineConfigError(
                    f"stages[{i}] ({stage_type}): invalid params {params!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            pipeline.add_stage(stage_instance)
            logger.debug("PipelineConfig: stages[%d] = %s(name=%r)",
                         i, stage_type, stage_instance.name)

        logger.info("PipelineConfig: built pipeline with %d stage(s)",
                    len(pipeline))
        return pipeline

    @classmethod
    def build_from_file(
        cls,
        path: str,
        extra_stages: dict[str, type[PipelineStage]] | None = None,
        plugin_registry: "PluginRegistry | None" = None,
    ) -> "Pipeline":
        """Load a config file and build a Pipeline.

        Convenience wrapper combining :meth:`load_file` and
        :meth:`build_from_config`.

        Parameters:
            path: Path to the config file (``.json`` / ``.yaml`` /
                ``.yml``).
            extra_stages: Additional stage classes keyed by name.
            plugin_registry: Optional plugin registry for resolution.

        Returns:
            A configured Pipeline.
        """
        config = cls.load_file(path)
        return cls.build_from_config(
            config,
            extra_stages=extra_stages,
            plugin_registry=plugin_registry,
        )
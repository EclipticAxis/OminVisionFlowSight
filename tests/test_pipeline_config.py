"""Pipeline configuration unit tests (Milestone D8).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - PipelineConfig.build_from_file with JSON configs
    - Stage resolution (builtin, extra_stages, unknown → error)
    - Pipeline.load_config convenience classmethod
    - pipeline_enabled flag (empty pipeline when False)
    - Config validation (missing type, bad params, missing file, bad JSON)
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline import Pipeline, PipelineStage
from visioncore.pipeline.config.pipeline_config import (
    BUILTIN_STAGES,
    PipelineConfig,
    PipelineConfigError,
)
from visioncore.pipeline.context import PipelineContext


# ======================================================================
# Test helpers
# ======================================================================

class _Raises:
    """Context manager asserting that a block raises a matching exception."""

    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected: type[BaseException] = expected
        self.match: str | None = match
        self.caught: BaseException | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__}, but no exception was raised"
            )
        if not isinstance(exc_val, BaseException):
            raise AssertionError(f"expected {self.expected.__name__}, got {exc_val!r}")
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}"
            )
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(
                f"expected {self.match!r} in str, got {str(exc_val)!r}"
            )
        self.caught = exc_val
        return True


def raises(expected: type[BaseException], match: str | None = None) -> _Raises:
    return _Raises(expected, match=match)


def _write_config(data: dict[str, Any], suffix: str = ".json") -> str:
    """Write a config dict to a temp file and return its path."""
    d = tempfile.mkdtemp()
    p = Path(d) / f"config{suffix}"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ======================================================================
# 0. Module surface
# ======================================================================

def test_pipeline_config_error_is_runtime_error():
    assert issubclass(PipelineConfigError, RuntimeError)


def test_builtin_stages_contains_dummy():
    assert "DummyStage" in BUILTIN_STAGES
    assert issubclass(BUILTIN_STAGES["DummyStage"], PipelineStage)


def test_builtin_stages_contains_advanced():
    """BUILTIN_STAGES includes at least one advanced stage."""
    advanced = [k for k in BUILTIN_STAGES if k != "DummyStage"]
    assert len(advanced) >= 1


# ======================================================================
# 1. PipelineConfig.build_from_file (JSON)
# ======================================================================

def test_build_from_file_simple():
    """Config with two DummyStages produces a pipeline with two stages."""
    path = _write_config({
        "stages": [
            {"type": "DummyStage", "params": {"name": "a"}},
            {"type": "DummyStage", "params": {"name": "b"}},
        ]
    })
    pipeline = PipelineConfig.build_from_file(path)
    assert len(pipeline) == 2
    names = [s.name for s in pipeline.stages]
    assert names == ["a", "b"]


def test_build_from_file_no_params():
    """Omitting params uses stage class defaults."""
    path = _write_config({
        "stages": [{"type": "DummyStage"}]
    })
    pipeline = PipelineConfig.build_from_file(path)
    assert len(pipeline) == 1
    assert pipeline.stages[0].name == "DummyStage"


def test_build_from_file_empty_stages():
    """Config with empty stages list produces empty pipeline."""
    path = _write_config({"stages": []})
    pipeline = PipelineConfig.build_from_file(path)
    assert len(pipeline) == 0


def test_build_from_file_pipeline_is_new_state():
    """Returned pipeline is in NEW state (not yet initialised)."""
    path = _write_config({
        "stages": [{"type": "DummyStage", "params": {"name": "s1"}}]
    })
    pipeline = PipelineConfig.build_from_file(path)
    assert pipeline.initialized is False
    assert pipeline.shutdown_done is False


def test_pipeline_is_runnable_after_config_load():
    """Config-loaded pipeline can be initialised and run."""
    path = _write_config({
        "stages": [
            {"type": "DummyStage", "params": {"name": "capture"}},
            {"type": "DummyStage", "params": {"name": "detect"}},
        ]
    })
    pipeline = PipelineConfig.build_from_file(path)
    ctx = PipelineContext.empty()
    with pipeline:
        pipeline.run(ctx)
    assert ctx.metadata.get("order") == ["capture", "detect"]
    assert pipeline.shutdown_done is True


# ======================================================================
# 2. Stage resolution
# ======================================================================

def test_resolve_stage_builtin():
    cls = PipelineConfig.resolve_stage("DummyStage")
    assert cls.__name__ == "DummyStage"


def test_resolve_stage_extra():
    """extra_stages override takes precedence."""
    class MyStage(PipelineStage):
        def initialize(self): pass
        def process(self, context): pass
        def shutdown(self): pass
        def health_check(self): return True

    cls = PipelineConfig.resolve_stage("MyStage", extra_stages={"MyStage": MyStage})
    assert cls is MyStage


def test_resolve_stage_extra_overrides_builtin():
    """extra_stages can shadow a builtin."""
    class DummyStage(PipelineStage):
        def initialize(self): pass
        def process(self, context): pass
        def shutdown(self): pass
        def health_check(self): return True

    cls = PipelineConfig.resolve_stage(
        "DummyStage",
        extra_stages={"DummyStage": DummyStage},
    )
    assert cls is DummyStage  # shadowed


def test_resolve_stage_unknown_raises():
    with raises(PipelineConfigError, match="unknown stage type"):
        PipelineConfig.resolve_stage("NonExistentStage")


# ======================================================================
# 3. Pipeline.load_config convenience
# ======================================================================

def test_pipeline_load_config():
    """Pipeline.load_config classmethod works identically."""
    path = _write_config({
        "stages": [{"type": "DummyStage", "params": {"name": "x"}}]
    })
    pipeline = Pipeline.load_config(path)
    assert len(pipeline) == 1
    assert pipeline.stages[0].name == "x"


def test_pipeline_load_config_returns_new_pipeline():
    path = _write_config({"stages": []})
    p1 = Pipeline.load_config(path)
    p2 = Pipeline.load_config(path)
    assert p1 is not p2


# ======================================================================
# 4. pipeline_enabled flag
# ======================================================================

def test_pipeline_enabled_false_returns_empty():
    """When pipeline_enabled=False, build returns an empty pipeline."""
    from visioncore.config import settings
    old = settings.pipeline_enabled
    try:
        settings.pipeline_enabled = False
        path = _write_config({
            "stages": [{"type": "DummyStage", "params": {"name": "s1"}}]
        })
        pipeline = PipelineConfig.build_from_file(path)
        assert len(pipeline) == 0
    finally:
        settings.pipeline_enabled = old


# ======================================================================
# 5. Config validation errors
# ======================================================================

def test_missing_file_raises():
    with raises(PipelineConfigError, match="not found"):
        PipelineConfig.build_from_file("/nonexistent/path/config.json")


def test_bad_json_raises():
    d = tempfile.mkdtemp()
    p = Path(d) / "bad.json"
    p.write_text("{invalid json", encoding="utf-8")
    with raises(PipelineConfigError, match="invalid JSON"):
        PipelineConfig.build_from_file(str(p))


def test_unsupported_format_raises():
    d = tempfile.mkdtemp()
    p = Path(d) / "config.txt"
    p.write_text("{}", encoding="utf-8")
    with raises(PipelineConfigError, match="unsupported config format"):
        PipelineConfig.build_from_file(str(p))


def test_missing_stages_key_raises():
    path = _write_config({"other_key": []})
    with raises(PipelineConfigError, match="missing required 'stages' key"):
        PipelineConfig.build_from_file(path)


def test_stages_not_a_list_raises():
    path = _write_config({"stages": "not a list"})
    with raises(PipelineConfigError, match="'stages' must be a list"):
        PipelineConfig.build_from_file(path)


def test_stage_entry_not_a_dict_raises():
    path = _write_config({"stages": ["not a dict"]})
    with raises(PipelineConfigError, match="stages[0]: must be a dict"):
        PipelineConfig.build_from_file(path)


def test_stage_missing_type_raises():
    path = _write_config({"stages": [{"params": {}}]})
    with raises(PipelineConfigError, match="missing required 'type' key"):
        PipelineConfig.build_from_file(path)


def test_stage_invalid_params_raises():
    path = _write_config({
        "stages": [{"type": "DummyStage", "params": "not a dict"}]
    })
    with raises(PipelineConfigError, match="must be a dict"):
        PipelineConfig.build_from_file(path)


def test_stage_unknown_type_raises():
    path = _write_config({
        "stages": [{"type": "NonExistentStage", "params": {}}]
    })
    with raises(PipelineConfigError, match="unknown stage type"):
        PipelineConfig.build_from_file(path)


def test_stage_bad_params_raises():
    """Passing invalid constructor params raises PipelineConfigError."""
    path = _write_config({
        "stages": [{"type": "DummyStage", "params": {"bad_kwarg": True}}]
    })
    with raises(PipelineConfigError, match="invalid params"):
        PipelineConfig.build_from_file(path)


# ======================================================================
# 6. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """pipeline_config.py contains no network / model / GUI / ai imports."""
    import visioncore.pipeline.config.pipeline_config as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "torch", "onnx", "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for b in banned:
                    if b in alias.name.lower():
                        violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = (node.module or "").lower()
            for b in banned:
                if b in mod_name:
                    violations.append(f"from {node.module}")
    assert not violations, f"pipeline_config.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing pipeline config does not load torch."""
    import visioncore.pipeline.config.pipeline_config  # noqa: F401
    assert "torch" not in sys.modules, "torch loaded as side effect"


# ======================================================================
# Runner
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} pipeline config tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}")
    print(f"Result: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        print(f"\n{'=' * 60}\nFailure details:\n")
        for name, tb in failures:
            print(f"---- {name} ----")
            print(tb)
        raise SystemExit(1)
    print("All pipeline config tests passed.")

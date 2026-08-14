"""Model runtime -- lifecycle management for inference models.

This module extracts the model lifecycle logic from ``ai/inference.py``
into a testable, config-driven abstraction. It contains:

* :class:`ModelInfo` -- model metadata (path, backend, device, loaded).
* :class:`ModelRuntime` -- abstract base class for a single model's
  lifecycle (load / unload / warmup / predict / health).
* :class:`ModelManager` -- manages multiple runtimes: device detection,
  model switching, warmup sequencing, and aggregate health.

The runtime layer does **not** contain detection logic (NMS, box
conversion, label filtering). The ``predict()`` method returns the raw
model output; post-processing is the detector's responsibility.

Scope (Milestone D9.2)
----------------------
D9.2 extracts model lifecycle from ``ai/inference.py``:

* ``_initialize_backend`` → ``ModelManager.load()``
* ``_warmup`` → ``ModelManager.warmup_all()``
* ``_switch_model_if_needed`` → ``ModelManager.switch_model()``
* ``_detect_device`` → ``ModelManager.detect_device()``
* ``_load_detect_model`` / ``_load_onnx_detect_model`` → concrete
  ``ModelRuntime`` subclasses (injected, not defined here)
* ``_predict_model`` → ``ModelRuntime.predict()``
* ``_model`` / ``_detect_model`` / ``_pose_model`` → ``ModelManager``
  named runtime slots

D9.2 deliberately does **not** contain:

* any YOLO / ONNX / torch code (concrete runtimes are injected);
* any detection post-processing (NMS, box conversion);
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Example
-------
    >>> from visioncore.runtime.model import ModelManager, ModelRuntime
    >>> manager = ModelManager()
    >>> manager.detect_device()
    'cpu'
    >>> manager.warmup_all()
    >>> manager.health_check()
    True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ModelInfo",
    "ModelManager",
    "ModelManagerError",
    "ModelRuntime",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class ModelManagerError(RuntimeError):
    """Structured exception signalling a model manager failure.

    Example:
        >>> raise ModelManagerError("model load failed")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        visioncore.runtime.model.model_runtime.ModelManagerError: model load failed
    """


# ======================================================================
# Model metadata
# ======================================================================

@dataclass
class ModelInfo:
    """Model metadata.

    Attributes:
        path: Model file path (or empty for auto/default).
        backend: Backend type (``"pytorch"``, ``"onnx"``, ``"cpu"``).
        device: Device string (``"cpu"``, ``"cuda"``, ``"directml"``).
        loaded: Whether the model is currently loaded.
        warmup_done: Whether warmup has been performed.
    """

    path: str = ""
    backend: str = "pytorch"
    device: str = "cpu"
    loaded: bool = False
    warmup_done: bool = False


# ======================================================================
# ModelRuntime -- abstract base class for a single model
# ======================================================================

class ModelRuntime(ABC):
    """Abstract base class for a single model's lifecycle.

    A runtime manages one model: loading, unloading, warmup, prediction,
    and health checking. The model itself is opaque — the runtime knows
    *that* it predicts, not *how* it predicts.

    Example:
        >>> class NoopRuntime(ModelRuntime):
        ...     def load(self, path, device="cpu"):
        ...         pass
        ...     def unload(self):
        ...         pass
        ...     def warmup(self, imgsz=640):
        ...         pass
        ...     def predict(self, frame, conf=0.25, imgsz=640):
        ...         return []
        ...     def health_check(self):
        ...         return True
        >>> r = NoopRuntime()
        >>> r.health_check()
        True
    """

    @abstractmethod
    def load(self, path: str | None = None, device: str = "cpu") -> None:
        """Load the model from ``path`` onto ``device``.

        Parameters:
            path: Model file path. ``None`` uses the runtime's default.
            device: Device string (``"cpu"``, ``"cuda"``, etc.).
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and release resources."""

    @abstractmethod
    def warmup(self, imgsz: int = 640) -> None:
        """Run warmup inference on a dummy input.

        Parameters:
            imgsz: Input image size for warmup.
        """

    @abstractmethod
    def predict(
        self,
        frame: Any,
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> Any:
        """Run inference and return raw model output.

        Parameters:
            frame: Input frame (numpy array).
            conf: Confidence threshold.
            imgsz: Input image size.

        Returns:
            Raw model output (type depends on backend).
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the model is loaded and ready."""


# ======================================================================
# ModelManager -- multi-model lifecycle manager
# ======================================================================

class ModelManager:
    """Manages multiple model runtimes with device detection and switching.

    :class:`ModelManager` coordinates named runtime slots (e.g.
    ``"detect"``, ``"pose"``), handles model switching with rollback on
    failure, and provides aggregate health checking.

    Example:
        >>> manager = ModelManager()
        >>> manager.detect_device()
        'cpu'
        >>> manager.warmup_all()
        >>> manager.health_check()
        True
    """

    __slots__ = ("_device", "_runtimes", "_infos", "_default_imgsz")

    def __init__(self, device: str = "cpu", default_imgsz: int = 640) -> None:
        """Create a ModelManager.

        Parameters:
            device: Device string for all runtimes.
            default_imgsz: Default image size for warmup/predict.
        """
        self._device: str = device
        self._runtimes: dict[str, ModelRuntime] = {}
        self._infos: dict[str, ModelInfo] = {}
        self._default_imgsz: int = default_imgsz
        logger.debug("ModelManager created: device=%s imgsz=%d",
                     device, default_imgsz)

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_device() -> str:
        """Detect the best available device.

        Checks for CUDA availability (via ``torch`` if installed),
        falls back to ``"cpu"``.

        Returns:
            Device string: ``"cuda"`` if available, otherwise ``"cpu"``.
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info("ModelManager.detect_device: CUDA available: %s",
                            device_name)
                return "cuda"
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("ModelManager.detect_device: CUDA check failed: %s",
                           exc)
        logger.info("ModelManager.detect_device: using CPU")
        return "cpu"

    # ------------------------------------------------------------------
    # Runtime registry
    # ------------------------------------------------------------------

    def register_runtime(self, name: str, runtime: ModelRuntime) -> None:
        """Register a named runtime slot.

        Parameters:
            name: Slot name (e.g. ``"detect"``, ``"pose"``).
            runtime: The ModelRuntime instance.
        """
        self._runtimes[name] = runtime
        self._infos[name] = ModelInfo(device=self._device)
        logger.debug("ModelManager.register_runtime: name=%s", name)

    def get_runtime(self, name: str) -> ModelRuntime | None:
        """Return the runtime registered under ``name``, or ``None``."""
        return self._runtimes.get(name)

    def list_runtimes(self) -> list[str]:
        """Return the names of all registered runtimes."""
        return list(self._runtimes.keys())

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        path: str | None = None,
        device: str | None = None,
    ) -> None:
        """Load the model in the named runtime slot.

        Parameters:
            name: Runtime slot name.
            path: Model file path. ``None`` uses default.
            device: Device string. ``None`` uses the manager's device.

        Raises:
            ModelManagerError: If the runtime slot is not registered.
        """
        runtime = self._runtimes.get(name)
        if runtime is None:
            raise ModelManagerError(f"load: unknown runtime {name!r}")
        use_device = device or self._device
        info = self._infos.get(name)
        try:
            runtime.load(path, use_device)
            if info is not None:
                info.path = path or ""
                info.device = use_device
                info.loaded = True
                info.warmup_done = False
            logger.info("ModelManager.load: name=%s path=%s device=%s",
                        name, path or "<default>", use_device)
        except Exception as exc:
            if info is not None:
                info.loaded = False
            raise ModelManagerError(
                f"load: {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def unload(self, name: str) -> None:
        """Unload the model in the named runtime slot.

        Parameters:
            name: Runtime slot name.
        """
        runtime = self._runtimes.get(name)
        if runtime is None:
            return
        try:
            runtime.unload()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ModelManager.unload: %s failed: %s", name, exc)
        info = self._infos.get(name)
        if info is not None:
            info.loaded = False
            info.warmup_done = False
        logger.debug("ModelManager.unload: name=%s", name)

    def warmup_all(self, imgsz: int | None = None) -> None:
        """Run warmup on all loaded runtimes.

        Parameters:
            imgsz: Image size for warmup. ``None`` uses default.
        """
        use_imgsz = imgsz or self._default_imgsz
        for name, runtime in self._runtimes.items():
            try:
                runtime.warmup(use_imgsz)
                info = self._infos.get(name)
                if info is not None:
                    info.warmup_done = True
                logger.debug("ModelManager.warmup_all: name=%s done", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ModelManager.warmup_all: %s failed: %s",
                               name, exc)

    def switch_model(
        self,
        name: str,
        new_path: str,
        device: str | None = None,
    ) -> bool:
        """Hot-swap the model in the named slot with rollback on failure.

        Parameters:
            name: Runtime slot name.
            new_path: New model file path.
            device: Device string. ``None`` uses the manager's device.

        Returns:
            ``True`` on success, ``False`` on failure (old model restored).
        """
        runtime = self._runtimes.get(name)
        if runtime is None:
            logger.warning("ModelManager.switch_model: unknown runtime %r", name)
            return False
        info = self._infos.get(name)
        old_path = info.path if info is not None else ""
        use_device = device or self._device

        try:
            logger.info("ModelManager.switch_model: %s -> %s", name, new_path)
            runtime.unload()
            runtime.load(new_path, use_device)
            runtime.warmup(self._default_imgsz)
            if info is not None:
                info.path = new_path
                info.device = use_device
                info.loaded = True
                info.warmup_done = True
            logger.info("ModelManager.switch_model: %s switched successfully", name)
            return True
        except Exception as exc:
            logger.error("ModelManager.switch_model: %s failed, rolling back: %s",
                         name, exc)
            try:
                runtime.unload()
                runtime.load(old_path or None, use_device)
                if info is not None:
                    info.path = old_path
                    info.loaded = True
            except Exception as rollback_exc:  # noqa: BLE001
                logger.error("ModelManager.switch_model: rollback also failed: %s",
                             rollback_exc)
                if info is not None:
                    info.loaded = False
            return False

    # ------------------------------------------------------------------
    # Aggregate health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return ``True`` iff all registered runtimes are healthy.

        A runtime is healthy if its ``health_check()`` returns ``True``.
        An empty manager (no runtimes) is vacuously healthy.
        """
        for name, runtime in self._runtimes.items():
            try:
                if not runtime.health_check():
                    logger.debug("ModelManager.health_check: %s unhealthy", name)
                    return False
            except Exception:  # noqa: BLE001
                logger.warning("ModelManager.health_check: %s raised", name)
                return False
        return True

    def get_info(self, name: str) -> ModelInfo | None:
        """Return the ModelInfo for the named slot, or ``None``."""
        return self._infos.get(name)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names = list(self._runtimes.keys())
        return (
            f"ModelManager(device={self._device!r}, "
            f"runtimes={names})"
        )
"""Plugin registry -- dynamic registration and discovery (Milestone D7).

This module extends the D6 plugin interface layer with a **discovery-
oriented** plugin registry. :class:`PluginRegistry` adds:

* :meth:`PluginRegistry.get_plugin` -- an alias for ``get()`` per the
  D7 spec (``get_plugin`` is the external-facing query name);
* :meth:`PluginRegistry.load_entrypoints` -- discovers and registers
  plugins published as `setuptools entry points
  <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_
  under a given group name, using ``importlib.metadata``
  (available on Python >= 3.9);
* :meth:`PluginRegistry.load_package` -- loads a plugin module by
  dotted path and registers any top-level ``PluginInterface`` subclass
  instances it exposes (useful for in-process manual discovery and for
  tests that cannot rely on installed entry points).

Scope (Milestone D7)
--------------------
D7 extends the plugin layer with **discovery**. It deliberately does
**not** contain:

* any concrete detector / tracker / algorithm plugin;
* any networking, ROS2, or MAVLink transport code;
* any modification to ``ai/``, ``gui/``, or ``camera/``;
* any heavyweight third-party dependency (``pluggy``, etc.). The
  entry-point mechanism uses only the standard library.

Example
-------
    >>> from visioncore.plugin.registry import PluginRegistry
    >>> from visioncore.plugin.base import NullPlugin
    >>> registry = PluginRegistry()
    >>> registry.register(NullPlugin("null"))
    >>> registry.get_plugin("null").name
    'null'
    >>> registry.get_plugin("missing") is None
    True
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from visioncore.plugin.base import (
    PluginError,
    PluginInterface,
    PluginManager,
)

__all__ = ["PluginRegistry"]


logger = logging.getLogger(__name__)


# ======================================================================
# PluginRegistry
# ======================================================================

class PluginRegistry(PluginManager):
    """Dynamic plugin registry with entry-point discovery.

    :class:`PluginRegistry` is the recommended plugin registry for
    VisionCore applications. It extends :class:`PluginManager` with:

    * a concrete in-memory store (``dict[str, PluginInterface]``);
    * :meth:`get_plugin` -- the external query name (alias for
      ``get``);
    * :meth:`load_entrypoints` -- setuptools entry-point scanner;
    * :meth:`load_package` -- dotted-path module loader.

    The registry enforces the same rules as
    :class:`~visioncore.plugin.base.MemoryPluginManager`:

    * only ``PluginInterface`` instances can be registered;
    * duplicate names are rejected with :class:`PluginError`;
    * unregistration is idempotent (unknown names are silently ignored).

    Attributes:
        _plugins: Internal ordered dict mapping plugin name to instance.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.names()
        []
        >>> len(registry)
        0
    """

    __slots__ = ("_plugins",)

    def __init__(self) -> None:
        """Create an empty plugin registry."""
        self._plugins: dict[str, "visioncore.plugin.base.PluginInterface"] = {}
        logger.debug("PluginRegistry created")

    # ------------------------------------------------------------------
    # PluginManager abstract contract (concrete implementation)
    # ------------------------------------------------------------------

    def register(
        self,
        plugin: "visioncore.plugin.base.PluginInterface",
    ) -> None:
        """Register a plugin under its ``name``.

        Raises:
            PluginError: If ``plugin`` is not a PluginInterface, or a
                plugin with the same name is already registered.
        """
        if not isinstance(plugin, PluginInterface):
            raise PluginError(
                f"register: expected a PluginInterface, got "
                f"{type(plugin).__name__}"
            )
        if plugin.name in self._plugins:
            raise PluginError(
                f"register: plugin {plugin.name!r} is already registered"
            )
        self._plugins[plugin.name] = plugin
        logger.debug("PluginRegistry.register: name=%s version=%s",
                     plugin.name, plugin.version)

    def unregister(self, name: str) -> None:
        """Remove the plugin registered under ``name`` (idempotent)."""
        removed: "visioncore.plugin.base.PluginInterface" | None = (
            self._plugins.pop(name, None)
        )
        logger.debug("PluginRegistry.unregister: name=%s removed=%s",
                     name, removed is not None)

    def get(
        self,
        name: str,
    ) -> "visioncore.plugin.base.PluginInterface" | None:
        """Return the plugin registered under ``name``, or ``None``."""
        return self._plugins.get(name)

    def list(self) -> list["visioncore.plugin.base.PluginInterface"]:
        """Return all registered plugins, in registration order."""
        return list(self._plugins.values())

    # ------------------------------------------------------------------
    # D7: query and discovery
    # ------------------------------------------------------------------

    def get_plugin(
        self,
        name: str,
    ) -> "visioncore.plugin.base.PluginInterface" | None:
        """Return the plugin registered under ``name``, or ``None``.

        This is the external-facing query name mandated by the D7 spec.
        It delegates to :meth:`get`.

        Parameters:
            name: The plugin name to look up.

        Returns:
            The registered plugin, or ``None`` when no plugin with that
            name is registered.
        """
        return self.get(name)

    def load_entrypoints(self, group_name: str) -> list["visioncore.plugin.base.PluginInterface"]:
        """Discover and register plugins published as entry points.

        Uses the standard library ``importlib.metadata`` (Python >= 3.9)
        to scan for entry points registered under ``group_name`` in
        installed packages. Each entry point's ``load()`` return value
        is inspected: if it is a ``PluginInterface`` **instance**, it is
        registered directly; if it is a ``PluginInterface`` **subclass**,
        it is instantiated first (no arguments).

        Packages with entry points that fail to load are logged as
        warnings and skipped -- the registry remains usable.

        Parameters:
            group_name: The setuptools entry-point group to scan
                (e.g. ``"visioncore.plugin"``).

        Returns:
            The list of plugins that were successfully discovered and
            registered (may be empty).
        """
        discovered: list["visioncore.plugin.base.PluginInterface"] = []
        try:
            from importlib.metadata import entry_points
        except ImportError:
            logger.warning(
                "PluginRegistry.load_entrypoints: importlib.metadata "
                "not available -- entry-point discovery disabled"
            )
            return discovered

        # Python 3.12+: entry_points(group=...) returns list[EntryPoint].
        # Python 3.9-3.11: entry_points(group=...) returns dict-like
        #                  {group: [EntryPoint, ...]} -- still iterable.
        try:
            eps = entry_points(group=group_name)
        except TypeError:
            # Fallback: pre-3.12 API without keyword filter.
            try:
                all_eps = entry_points()  # type: ignore[call-overload]
                eps = all_eps.get(group_name, [])  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 -- pragma: no cover
                logger.warning(
                    "PluginRegistry.load_entrypoints: failed to query "
                    "entry_points for group %r", group_name
                )
                return discovered

        for ep in eps:
            try:
                loaded = ep.load()
            except Exception as exc:  # noqa: BLE001 -- skip broken entry points
                logger.warning(
                    "PluginRegistry.load_entrypoints: %r failed to "
                    "load: %s: %s", ep.name, type(exc).__name__, exc,
                )
                continue

            plugin: "visioncore.plugin.base.PluginInterface" | None = None
            if isinstance(loaded, PluginInterface):
                plugin = loaded
            elif isinstance(loaded, type) and issubclass(loaded, PluginInterface):
                try:
                    plugin = loaded()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "PluginRegistry.load_entrypoints: %r "
                        "instantiation failed: %s: %s",
                        ep.name, type(exc).__name__, exc,
                    )
                    continue

            if plugin is None:
                logger.warning(
                    "PluginRegistry.load_entrypoints: %r is not a "
                    "PluginInterface instance or subclass -- skipped",
                    ep.name,
                )
                continue

            try:
                self.register(plugin)
                discovered.append(plugin)
                logger.info(
                    "PluginRegistry.load_entrypoints: registered %r "
                    "from entry point %r (group=%r)",
                    plugin.name, ep.name, group_name,
                )
            except PluginError as exc:
                logger.warning(
                    "PluginRegistry.load_entrypoints: %r skipped: %s",
                    ep.name, exc,
                )

        logger.debug(
            "PluginRegistry.load_entrypoints: group=%r discovered=%d",
            group_name, len(discovered),
        )
        return discovered

    def load_package(self, dotted_path: str) -> list["visioncore.plugin.base.PluginInterface"]:
        """Load a Python module by dotted path and register plugin instances.

        Imports the module identified by ``dotted_path`` and scans its
        top-level attributes for ``PluginInterface`` instances. Each
        instance found is registered in this registry (duplicate names
        are rejected via :class:`PluginError`).

        This is useful for in-process manual discovery (config-driven
        plugin loading) and for tests that cannot rely on installed
        setuptools entry points.

        Parameters:
            dotted_path: A fully-qualified Python module path
                (e.g. ``"mypackage.plugins.my_plugin"``).

        Returns:
            The list of plugins that were successfully discovered and
            registered (may be empty).

        Raises:
            PluginError: If the module cannot be imported.
        """
        try:
            module = importlib.import_module(dotted_path)
        except ImportError as exc:
            raise PluginError(
                f"load_package: failed to import {dotted_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        discovered: list["visioncore.plugin.base.PluginInterface"] = []
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
            except Exception:  # noqa: BLE001 -- skip properties that raise
                continue
            if isinstance(attr, PluginInterface):
                try:
                    self.register(attr)
                    discovered.append(attr)
                except PluginError as exc:
                    logger.warning(
                        "PluginRegistry.load_package: %s.%s skipped: %s",
                        dotted_path, attr_name, exc,
                    )

        logger.debug("PluginRegistry.load_package: path=%r discovered=%d",
                     dotted_path, len(discovered))
        return discovered

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered plugins."""
        return len(self._plugins)

    def __repr__(self) -> str:
        """Return a concise representation with registered names."""
        return (
            f"PluginRegistry(plugins={self.names()!r}, "
            f"count={len(self._plugins)})"
        )
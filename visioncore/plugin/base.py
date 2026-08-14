"""Plugin interface layer -- PluginInterface, PluginManager, example plugins.

This module defines the *plugin abstraction* (Milestone D6): a minimal
contract that future plugins (detectors, trackers, algorithms) implement
so they can be loaded, queried, and managed uniformly. It contains three
cohesive concerns:

* :class:`PluginError` -- structured exception for plugin registration /
  management failures.
* :class:`PluginInterface` -- the abstract base class for **all**
  plugins. It defines the lifecycle contract ``load() -> run() ->
  shutdown()`` with empty default implementations, so a concrete plugin
  only overrides what it needs -- but the ABC still forces every
  subclass to provide the full contract before it can be instantiated.
* :class:`PluginManager` -- the abstract base class for plugin
  registries (register / unregister / get / list), plus
  :class:`MemoryPluginManager`, a concrete in-memory reference
  implementation. Managers **load** (register) plugins, **query** them
  by interface, and can drive their lifecycle via
  :meth:`PluginManager.load_all` / :meth:`PluginManager.shutdown_all`.

Two example plugins ship with the layer:

* :class:`NullPlugin` -- a no-op passthrough: ``run()`` returns its
  input snapshot unchanged.
* :class:`ConsolePlugin` -- logs lifecycle events and each processed
  snapshot to ``logging``. It performs **no** network communication.

Scope (Milestone D6)
--------------------
D6 delivers the **interfaces and examples only**. It deliberately does
**not** contain:

* any concrete detector / tracker / algorithm plugin;
* any networking, ROS2, or MAVLink transport code;
* any modification to ``ai/``, ``gui/``, or ``camera/``.

Type annotation convention
--------------------------
Every interface signature that mentions a type uses its **full path**
in the annotation. In particular, plugin ``run()`` declares its data
type as ``"visioncore.state.target_state.TargetState"`` (never a bare
``TargetState``). With ``from __future__ import annotations`` these are
strings at runtime; the ``import visioncore.state.target_state`` below
binds the top-level ``visioncore`` name so the full-path strings remain
resolvable by type checkers / introspection tools.

Example
-------
    >>> from visioncore.plugin.base import (
    ...     ConsolePlugin, MemoryPluginManager, NullPlugin,
    ... )
    >>> manager = MemoryPluginManager()
    >>> manager.register(NullPlugin())
    >>> manager.register(ConsolePlugin("console"))
    >>> manager.names()
    ['null', 'console']
    >>> manager.get("null") is not None
    True
    >>> manager.unregister("console")
    >>> manager.names()
    ['null']
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import visioncore.state.target_state  # noqa: F401 -- binds 'visioncore' so
# full-path string annotations (D6 spec) resolve for type checkers.

__all__ = [
    "PluginError",
    "PluginInterface",
    "PluginManager",
    "MemoryPluginManager",
    "NullPlugin",
    "ConsolePlugin",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class PluginError(RuntimeError):
    """Structured exception signalling a plugin-layer failure.

    Raised by plugin managers when a plugin cannot be registered: the
    value is not a :class:`PluginInterface`, or a plugin with the same
    name is already registered. Concrete plugins may also raise it for
    their own recoverable failures.

    Example:
        >>> raise PluginError("duplicate plugin 'console'")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        visioncore.plugin.base.PluginError: duplicate plugin 'console'
    """


# ======================================================================
# PluginInterface -- the base class for all plugins
# ======================================================================

class PluginInterface(ABC):
    """Abstract base class for all VisionCore plugins.

    A plugin is a swappable unit of processing that follows the lifecycle
    contract ``load() -> run(...) -> shutdown()``:

    * :meth:`load` -- acquire resources once, before the first run;
    * :meth:`run` -- process one target snapshot and return the result;
    * :meth:`shutdown` -- release resources; must not raise.

    The base class provides **empty default implementations** for all
    three lifecycle methods (no-op load, identity passthrough run,
    no-op shutdown), so a plugin can be written by overriding only the
    methods it cares about. The ABC machinery still requires every
    subclass to *declare* the full contract (``name``, ``version``,
    ``load``, ``run``, ``shutdown``) before it can be instantiated.

    Type annotations in the contract use full paths -- e.g.
    ``run()`` declares ``"visioncore.state.target_state.TargetState"``
    -- per the D6 spec, so the interface is self-documenting.

    Attributes:
        name: Canonical plugin name; also the key used by
            :class:`PluginManager` registries.
        version: Plugin version string (semantic versioning encouraged).

    Example:
        >>> class AddOnePlugin(PluginInterface):
        ...     @property
        ...     def name(self) -> str:
        ...         return "add-one"
        ...     @property
        ...     def version(self) -> str:
        ...         return "0.1.0"
        ...     def load(self) -> None:
        ...         pass
        ...     def run(self, target_state):
        ...         return target_state.copy_with(health=2.0)
        ...     def shutdown(self) -> None:
        ...         pass
        >>> AddOnePlugin().name
        'add-one'
        >>> AddOnePlugin().version
        '0.1.0'
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Identity (abstract -- subclasses MUST implement)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical plugin name (the registry key)."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version string."""

    # ------------------------------------------------------------------
    # Lifecycle (abstract -- default empty implementations)
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Acquire resources before the first :meth:`run` call.

        Called once, idempotent. The default implementation is a no-op;
        concrete plugins override it to allocate models, open handles,
        or prepare state.
        """
        # D6: the interface ships an empty implementation -- concrete
        # plugins override with real resource acquisition.
        pass

    @abstractmethod
    def run(
        self,
        target_state: "visioncore.state.target_state.TargetState",
    ) -> "visioncore.state.target_state.TargetState":
        """Process one target snapshot and return the result.

        The default implementation is an **identity passthrough** (the
        input snapshot is returned unchanged). Concrete plugins override
        it to transform the snapshot -- e.g. a detector plugin fills
        box/confidence fields, an algorithm plugin enriches metadata.

        Parameters:
            target_state: A frozen
                ``visioncore.state.target_state.TargetState`` snapshot.

        Returns:
            A ``visioncore.state.target_state.TargetState`` -- the
            processed snapshot (usually a derived copy; snapshots are
            immutable).
        """
        return target_state

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`load`.

        Idempotent, must not raise. The default implementation is a
        no-op.
        """
        # D6: the interface ships an empty implementation -- concrete
        # plugins override with real teardown.
        pass

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with name and version."""
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"version={self.version!r})"
        )


# ======================================================================
# PluginManager -- abstract base class for plugin registries
# ======================================================================

class PluginManager(ABC):
    """Abstract base class for plugin registries.

    A plugin manager **loads** (registers) plugins, **queries** them by
    name / interface, and can drive the whole set through its lifecycle.
    The core registry operations -- :meth:`register`, :meth:`unregister`,
    :meth:`get`, :meth:`list` -- are abstract; concrete registries (see
    :class:`MemoryPluginManager`) supply the storage. Convenience
    operations (:meth:`names`, :meth:`load_all`,
    :meth:`shutdown_all`) are implemented once here on top of the
    abstract contract.

    Example:
        >>> class TinyManager(PluginManager):
        ...     def __init__(self):
        ...         self._store = {}
        ...     def register(self, plugin):
        ...         self._store[plugin.name] = plugin
        ...     def unregister(self, name):
        ...         self._store.pop(name, None)
        ...     def get(self, name):
        ...         return self._store.get(name)
        ...     def list(self):
        ...         return list(self._store.values())
        >>> manager = TinyManager()
        >>> manager.register(NullPlugin("a"))
        >>> manager.names()
        ['a']
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Registry contract (abstract -- subclasses MUST implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def register(
        self,
        plugin: "visioncore.plugin.base.PluginInterface",
    ) -> None:
        """Register a plugin, making it queryable by name.

        Parameters:
            plugin: The plugin to register. It must be a
                :class:`PluginInterface` whose :attr:`~PluginInterface.name`
                is not already registered.
        """

    @abstractmethod
    def unregister(self, name: str) -> None:
        """Remove the plugin registered under ``name``.

        Parameters:
            name: The plugin name to remove. Unknown names are ignored
                (idempotent).
        """

    @abstractmethod
    def get(
        self,
        name: str,
    ) -> "visioncore.plugin.base.PluginInterface" | None:
        """Return the plugin registered under ``name``, or ``None``.

        Parameters:
            name: The plugin name to look up.

        Returns:
            The registered plugin, or ``None`` when no plugin with that
            name is registered.
        """

    @abstractmethod
    def list(self) -> list["visioncore.plugin.base.PluginInterface"]:
        """Return all registered plugins, in registration order.

        Returns:
            A list of the registered
            :class:`visioncore.plugin.base.PluginInterface` instances.
        """

    # ------------------------------------------------------------------
    # Convenience operations (concrete -- composed from the contract)
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return the names of all registered plugins, in registration order."""
        return [plugin.name for plugin in self.list()]

    def __contains__(self, name: object) -> bool:
        """Return ``True`` iff a plugin named ``name`` is registered."""
        return isinstance(name, str) and self.get(name) is not None

    def load_all(self) -> None:
        """Call :meth:`~PluginInterface.load` on every registered plugin.

        Safe for managers with zero plugins (no-op).
        """
        for plugin in self.list():
            plugin.load()

    def shutdown_all(self) -> None:
        """Call :meth:`~PluginInterface.shutdown` on every registered plugin.

        Safe for managers with zero plugins (no-op).
        """
        for plugin in self.list():
            plugin.shutdown()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with the registered names."""
        try:
            names: list[str] = self.names()
        except Exception:  # noqa: BLE001 -- defensive: repr must not raise
            names = []
        return f"{type(self).__name__}(plugins={names!r})"


# ======================================================================
# MemoryPluginManager -- concrete in-memory reference implementation
# ======================================================================

class MemoryPluginManager(PluginManager):
    """In-memory plugin registry (reference implementation).

    Stores registered plugins in an insertion-ordered dict keyed by
    :attr:`~PluginInterface.name`. Enforces the registration rules:

    * the value must be a :class:`PluginInterface` (else
      :class:`PluginError`);
    * the name must not already be registered (else
      :class:`PluginError`).

    Example:
        >>> manager = MemoryPluginManager()
        >>> manager.register(NullPlugin("null"))
        >>> manager.get("null").name
        'null'
        >>> manager.register(NullPlugin("null"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        visioncore.plugin.base.PluginError: ...
    """

    __slots__ = ("_plugins",)

    def __init__(self) -> None:
        """Create an empty plugin registry."""
        self._plugins: dict[str, "visioncore.plugin.base.PluginInterface"] = {}
        logger.debug("MemoryPluginManager created")

    def register(
        self,
        plugin: "visioncore.plugin.base.PluginInterface",
    ) -> None:
        """Register ``plugin`` under its ``name``.

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
        logger.debug("MemoryPluginManager.register: name=%s version=%s",
                     plugin.name, plugin.version)

    def unregister(self, name: str) -> None:
        """Remove the plugin registered under ``name`` (idempotent)."""
        removed: "visioncore.plugin.base.PluginInterface" | None = (
            self._plugins.pop(name, None)
        )
        logger.debug("MemoryPluginManager.unregister: name=%s removed=%s",
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


# ======================================================================
# NullPlugin -- example no-op plugin
# ======================================================================

class NullPlugin(PluginInterface):
    """Example plugin: a no-op passthrough.

    :class:`NullPlugin` is the canonical minimal plugin. It performs no
    work and holds no resources:

    * :meth:`load` -- marks the plugin loaded;
    * :meth:`run` -- returns the input snapshot **unchanged**;
    * :meth:`shutdown` -- marks the plugin unloaded.

    It is also a convenient test double for exercising
    :class:`PluginManager` registries.

    Example:
        >>> plugin = NullPlugin("null")
        >>> plugin.name, plugin.version
        ('null', '0.1.0')
        >>> plugin.load()
        >>> plugin.health_check()
        True
        >>> plugin.shutdown()
        >>> plugin.health_check()
        False
    """

    __slots__ = ("_name", "_loaded")

    def __init__(self, name: str = "null") -> None:
        """Construct a NullPlugin with an optional name.

        Parameters:
            name: Plugin name (default ``"null"``), used as the registry
                key.
        """
        self._name: str = name
        self._loaded: bool = False
        logger.debug("NullPlugin created: name=%s", self._name)

    @property
    def name(self) -> str:
        """The plugin name (default ``"null"``)."""
        return self._name

    @property
    def version(self) -> str:
        """The plugin version."""
        return "0.1.0"

    def load(self) -> None:
        """Mark the plugin loaded. Idempotent, never raises."""
        self._loaded = True
        logger.debug("NullPlugin.load: name=%s", self._name)

    def run(
        self,
        target_state: "visioncore.state.target_state.TargetState",
    ) -> "visioncore.state.target_state.TargetState":
        """Return the input snapshot unchanged (identity passthrough).

        Parameters:
            target_state: A
                ``visioncore.state.target_state.TargetState`` snapshot.

        Returns:
            The exact same snapshot instance.
        """
        logger.debug("NullPlugin.run: name=%s target_id=%s",
                     self._name, target_state.target_id)
        return target_state

    def shutdown(self) -> None:
        """Mark the plugin unloaded. Idempotent, never raises."""
        self._loaded = False
        logger.debug("NullPlugin.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the plugin is loaded."""
        return self._loaded


# ======================================================================
# ConsolePlugin -- example logging plugin
# ======================================================================

class ConsolePlugin(PluginInterface):
    """Example plugin: logs its lifecycle and each processed snapshot.

    :class:`ConsolePlugin` demonstrates a plugin that *does* something
    observable: it logs to the standard ``logging`` module on
    :meth:`load`, :meth:`run`, and :meth:`shutdown`. It performs
    **no** network communication and holds no resources. ``run()``
    passes the snapshot through unchanged.

    Example:
        >>> plugin = ConsolePlugin("console")
        >>> plugin.name, plugin.version
        ('console', '0.1.0')
        >>> plugin.load()
        >>> plugin.run.__doc__ is not None
        True
    """

    __slots__ = ("_name", "_loaded")

    def __init__(self, name: str = "console") -> None:
        """Construct a ConsolePlugin with an optional name.

        Parameters:
            name: Plugin name (default ``"console"``), used as the
                registry key.
        """
        self._name: str = name
        self._loaded: bool = False
        logger.debug("ConsolePlugin created: name=%s", self._name)

    @property
    def name(self) -> str:
        """The plugin name (default ``"console"``)."""
        return self._name

    @property
    def version(self) -> str:
        """The plugin version."""
        return "0.1.0"

    def load(self) -> None:
        """Log the load event and mark the plugin ready. Idempotent."""
        self._loaded = True
        logger.info("ConsolePlugin(%s).load: plugin ready", self._name)

    def run(
        self,
        target_state: "visioncore.state.target_state.TargetState",
    ) -> "visioncore.state.target_state.TargetState":
        """Log the snapshot and pass it through unchanged.

        Parameters:
            target_state: A
                ``visioncore.state.target_state.TargetState`` snapshot.

        Returns:
            The exact same snapshot instance (no network, no I/O).
        """
        logger.info(
            "ConsolePlugin(%s).run: target_id=%s label=%s",
            self._name, target_state.target_id, target_state.label,
        )
        return target_state

    def shutdown(self) -> None:
        """Log the shutdown event and mark the plugin unloaded. Never raises."""
        self._loaded = False
        logger.info("ConsolePlugin(%s).shutdown: resources released", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the plugin is loaded."""
        return self._loaded
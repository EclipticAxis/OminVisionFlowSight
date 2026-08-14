"""Helper module exposing plugin instances for load_package tests (D7)."""
from visioncore.plugin.base import NullPlugin, ConsolePlugin

_alpha = NullPlugin("alpha")
_beta = ConsolePlugin("beta")
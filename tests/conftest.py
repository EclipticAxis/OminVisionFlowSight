"""Pytest options for the snapshot test suite."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden snapshots in tests/golden/tracker/ instead of comparing.",
    )

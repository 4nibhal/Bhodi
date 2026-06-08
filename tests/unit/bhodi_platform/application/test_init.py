"""Smoke tests for `bhodi_platform.application`'s public surface.

Catches the class of bug where a name is advertised in `__all__`
but cannot be imported (because the underlying module is broken
or has been removed).
"""

from __future__ import annotations

import pytest


def test_all_advertised_names_resolve() -> None:
    """Every name in `bhodi_platform.application.__all__` must be importable."""
    import bhodi_platform.application as app

    for name in app.__all__:
        # Touching the attribute triggers the lazy `__getattr__` import.
        # If the underlying module is broken or missing, this raises.
        assert hasattr(app, name), f"`bhodi_platform.application.{name}` is advertised but not present"


def test_indexing_entry_point_is_the_facade() -> None:
    """`BhodiApplication.index_document` is the canonical indexing entry point.

    After the removal of `IndexDocumentsUseCase` (a broken unused use
    case), indexing flows through the facade, not through a separate
    use case class.
    """
    from bhodi_platform.application.facade import BhodiApplication

    assert hasattr(BhodiApplication, "index_document")

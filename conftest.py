"""Shared by every test: no connection held by one test reaches the next."""
import pytest


@pytest.fixture(autouse=True)
def _no_held_live_connection():
    """stems2live keeps one `sideman.client.Live` for a run in a module global. A test that reached a
    loopback server through it would leave that connection, to a server that is gone, for the next test."""
    yield
    import sys
    S = sys.modules.get("stems2live")
    if S is not None and getattr(S, "_LIVE", None) is not None:
        S._LIVE[1].close()
        S._LIVE = None

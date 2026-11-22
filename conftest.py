"""Shared by every test: no connection held by one test reaches the next."""
import os
import tempfile

import pytest

# Under pytest-xdist each worker gets its own numba cache. librosa compiles several functions with
# cache=True; with a cold cache, workers (and the subprocesses the tests start) compiling and writing the
# same cache files at once crashed with SIGSEGV inside numba (seen in CI, then reproduced locally: a
# subprocess of test_analysis_key.py exiting -11). Numba's docs promise atomic cache writes, but the crash
# is real, so no two workers share a directory. This must be set before numba is first imported, and
# subprocesses inherit it. A serial run keeps numba's default location.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        tempfile.gettempdir(), "roadie-numba-cache", os.environ["PYTEST_XDIST_WORKER"])


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes over 6 s (real audio through hit_groups with regrouping); "
                                       "deselect with -m \"not slow\", runs by default")


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


@pytest.fixture(autouse=True, scope="session")
def _own_heavy_job_lock(tmp_path_factory):
    """No test, whatever its fixtures' scope, takes the real machine-wide lock (~/Library/Caches/roadie),
    which another job may hold. One temp lock file for the session; test_jobqueue.py sets its own per test."""
    mp = pytest.MonkeyPatch()
    mp.setenv("ROADIE_QUEUE_LOCK", str(tmp_path_factory.mktemp("queue") / "heavy-job.lock"))
    mp.delenv("ROADIE_NO_QUEUE", raising=False)
    yield
    mp.undo()

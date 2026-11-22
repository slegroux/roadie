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
    config.addinivalue_line("markers", "slow: CI runs it in the slow suite, not on every pull request. Marked on whole files "
                                       "where a module fixture or the analysis dominates, otherwise on single tests that took "
                                       "about 7 s or more in CI. Plain pytest runs it; CI selects with -m slow / -m \"not slow\"")


# test_drums.py and test_analysis_drumroles.py are being edited by other changes, so their slow tests are marked
# here, by file name and by test function name (every parametrization), not in the files. Move the marks into the
# files later (`pytestmark = pytest.mark.slow`, or `@pytest.mark.slow` on each test) and delete these two sets.
# test_drums.py: its module fixture `made` alone takes about 40 s in CI, and the file is 244 s of the 877 s serial run
# (CI run 45002429822). test_analysis_drumroles.py marks some of its own slow tests in the file; the names below
# are the ones that took 7 s or more in the fast job of CI run 45004487359.
_SLOW_FILES = {"test_drums.py"}
_SLOW_TESTS = {
    "test_analysis_drumroles.py": {
        "test_sections_are_the_same_groups_restricted_and_their_order_does_not_matter",
        "test_a_hit_8_ms_early_and_one_12_ms_late_keep_their_16th_and_say_how_far_off_they_are",
        "test_one_repeated_tom_is_one_group",
        "test_a_stem_read_from_a_file_and_the_inputs_that_are_refused",
        "test_16th_hats_under_a_real_open_hat_are_placed_in_the_hat_group_or_reported",
        "test_a_bed_40_db_under_the_hits_does_not_change_the_groups",
        "test_a_bed_30_db_under_the_hits_gives_fewer_groups_never_wrong_ones",
        "test_clap_and_snare_on_beats_2_and_4_are_two_groups",
        "test_a_tentative_satellite_is_not_the_neighbour_separation_is_measured_against",
    },
}


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.path.name
        if name in _SLOW_FILES or getattr(item, "originalname", item.name) in _SLOW_TESTS.get(name, ()):
            item.add_marker(pytest.mark.slow)


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


@pytest.fixture(autouse=True, scope="session")
def _own_hit_cache_for_the_session(tmp_path_factory):
    """Module- and session-scoped fixtures run before any test's own folder exists: they get one temp cache for the
    session, never ~/Library/Caches/roadie/drum-hits. `ROADIE_NO_CACHE` is cleared so a developer's shell cannot turn the cache off under the tests."""
    mp = pytest.MonkeyPatch()
    mp.setenv("ROADIE_CACHE_DIR", str(tmp_path_factory.mktemp("hitcache")))
    mp.delenv("ROADIE_NO_CACHE", raising=False)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _own_hit_cache(tmp_path, monkeypatch):
    """Each test gets its own cache of detected hits, so no test reads another's entries or the real cache (workers the test
    spawns inherit the variable)."""
    monkeypatch.setenv("ROADIE_CACHE_DIR", str(tmp_path / "hit-cache"))

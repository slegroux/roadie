#!/usr/bin/env python3
"""Tests for roadie_analysis/jobqueue.py: the machine-wide lock for heavy jobs. No audio; every test
points ROADIE_QUEUE_LOCK into tmp_path and uses short polling intervals."""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from roadie_analysis import jobqueue  # noqa: E402

# A job in its own process: takes the lock, says so, then holds it until its stdin closes.
HOLD = """
import sys
sys.path.insert(0, %r)
from roadie_analysis.jobqueue import heavy_job
with heavy_job(sys.argv[1], repeat=0.3, poll=0.05):
    print("held", flush=True)
    sys.stdin.read()
""" % HERE

# The same for a job that only takes the lock and goes.
TAKE = """
import sys
sys.path.insert(0, %r)
from roadie_analysis.jobqueue import heavy_job
with heavy_job(sys.argv[1], repeat=0.3, poll=0.05):
    print("done", flush=True)
""" % HERE


@pytest.fixture(autouse=True)
def lock(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADIE_QUEUE_LOCK", str(tmp_path / "cache" / "heavy-job.lock"))
    monkeypatch.delenv("ROADIE_NO_QUEUE", raising=False)
    monkeypatch.delenv(jobqueue.HELD_ENV, raising=False)
    return tmp_path / "cache" / "heavy-job.lock"


def spawn(code, *args, **kw):
    return subprocess.Popen([sys.executable, "-c", code, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **kw)


def holder(label):
    p = spawn(HOLD, label)
    assert p.stdout.readline().strip() == "held"
    return p


def release(p):
    p.stdin.close()
    p.wait(timeout=10)


def test_a_second_process_waits_for_the_first_and_names_it():
    h = holder("roadie drums Glacier 3000")
    w = spawn(TAKE, "second")
    time.sleep(0.8)
    assert w.poll() is None, "the second job did not wait"
    release(h)
    out, err = w.communicate(timeout=10)
    assert out.strip() == "done"
    lines = err.splitlines()
    assert "roadie drums Glacier 3000" in lines[0] and "pid %d" % h.pid in lines[0] and "since" in lines[0]
    assert any(l.startswith("roadie: still waiting for roadie drums Glacier 3000") for l in lines[1:-1]), err
    assert lines[-1].startswith("roadie: lock acquired after waiting")


def test_a_bounded_wait_gives_up_and_names_the_holder(monkeypatch):
    h = holder("roadie drums Glacier 3000")
    try:
        t = time.monotonic()
        with pytest.raises(jobqueue.QueueTimeout, match=r"machine busy: roadie drums Glacier 3000 \(pid %d, since .*\); retry later" % h.pid):
            with jobqueue.heavy_job("late", poll=0.05, max_wait_s=0.5, stream=open(os.devnull, "w")):
                pytest.fail("took the lock")
        assert 0.4 < time.monotonic() - t < 3
        assert jobqueue.HELD_ENV not in os.environ
    finally:
        release(h)
    with jobqueue.heavy_job("after", poll=0.05, max_wait_s=0.5):     # nothing was left held by the failed wait
        pass


def test_a_failure_after_the_lock_is_taken_does_not_keep_it(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(jobqueue.os, "pwrite", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        with jobqueue.heavy_job("note cannot be written", poll=0.05):      # the note is best effort
            pass
    real, calls = time.monotonic, []

    def breaks_after_the_flock():                  # 1st call: the wait's start; 2nd: just after the flock
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real()
    with monkeypatch.context() as m:
        m.setattr(jobqueue.time, "monotonic", breaks_after_the_flock)
        with pytest.raises(RuntimeError, match="boom"):
            jobqueue.acquire("broken", poll=0.05)
    out, err = spawn(TAKE, "next").communicate(timeout=10)             # the descriptor was closed: the lock is free
    assert out.strip() == "done" and err == "", err
    assert jobqueue.HELD_ENV not in os.environ


def test_a_bare_file_name_is_a_lock_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROADIE_QUEUE_LOCK", "heavy.lock")
    with jobqueue.heavy_job("here", poll=0.05):
        pass
    assert (tmp_path / "heavy.lock").exists()


def test_a_killed_holder_frees_the_lock():
    h = holder("doomed")
    h.send_signal(signal.SIGKILL)
    h.wait(timeout=10)
    w = spawn(TAKE, "next")
    out, err = w.communicate(timeout=10)
    assert out.strip() == "done" and err == "", err      # no waiting line: the lock was free at once


def test_no_queue_does_not_wait(monkeypatch):
    h = holder("busy")
    try:
        monkeypatch.setenv("ROADIE_NO_QUEUE", "1")
        t = time.monotonic()
        with jobqueue.heavy_job("skipped", poll=0.05):
            pass
        assert time.monotonic() - t < 1
    finally:
        release(h)


def test_a_nested_call_in_the_same_thread_does_not_wait(lock):
    with jobqueue.heavy_job("outer", poll=0.05):
        with jobqueue.heavy_job("inner", poll=0.05):
            assert os.environ[jobqueue.HELD_ENV] == str(lock)
        with jobqueue.heavy_job("inner again", poll=0.05):
            pass
        assert b"outer" in lock.read_bytes()
    assert jobqueue.HELD_ENV not in os.environ and lock.read_bytes() == b""


def test_another_thread_of_the_holder_is_another_job():
    got = []
    with jobqueue.heavy_job("first", poll=0.05):
        def other():
            with jobqueue.heavy_job("second", poll=0.05, stream=open(os.devnull, "w")):
                got.append(time.monotonic())
        t = threading.Thread(target=other)
        t.start()
        time.sleep(0.5)
        assert got == []
        released = time.monotonic()
    t.join(timeout=10)
    assert got and got[0] >= released


CHILD = """
import sys
sys.path.insert(0, %r)
from roadie_analysis.jobqueue import heavy_job
with heavy_job("child", poll=0.05):
    print("child ran")
""" % HERE


def test_a_child_of_the_holder_does_not_wait_for_it():
    with jobqueue.heavy_job("parent", poll=0.05):
        r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=10)
    assert r.stdout.strip() == "child ran" and r.stderr == "", r.stderr


def test_a_marker_whose_holder_is_gone_is_not_taken_for_ownership(lock):
    """An environment copy that outlived its holder must not let a job past the queue."""
    h = holder("other job")                     # the lock is held by someone else, and the marker says it is ours
    try:
        env = dict(os.environ, **{jobqueue.HELD_ENV: str(lock) + "x"})
        w = spawn(TAKE, "late", env=env)
        time.sleep(0.6)
        assert w.poll() is None                  # a marker for another lock file means nothing here
    finally:
        release(h)
    w.communicate(timeout=10)
    # and a marker for this lock with nobody holding it: the job takes the lock itself
    env = dict(os.environ, **{jobqueue.HELD_ENV: str(lock)})
    out, err = spawn(TAKE, "stale", env=env).communicate(timeout=10)
    assert out.strip() == "done" and err == ""


def test_run_hands_the_lock_to_the_command_until_it_exits():
    p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "roadie_analysis", "jobqueue.py"), "run", "roadie split x.wav",
                          "--", "sleep", "30"], stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):                     # the lock file holds the label once the lock is taken
            if os.path.exists(jobqueue.lock_path()) and b"roadie split x.wav" in open(jobqueue.lock_path(), "rb").read():
                break
            time.sleep(0.05)
        w = spawn(TAKE, "queued")
        time.sleep(0.6)
        assert w.poll() is None                  # `sleep` (the same pid, after the exec) holds it
        p.kill()
        p.wait(timeout=10)
        out, err = w.communicate(timeout=10)
        assert out.strip() == "done" and "roadie split x.wav" in err and "pid %d" % p.pid in err, err
    finally:
        p.kill()


def test_run_without_a_command_is_a_usage_error():
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "roadie_analysis", "jobqueue.py"), "run", "label"],
                       capture_output=True, text=True, timeout=10)
    assert r.returncode == 2 and "usage" in r.stderr


def test_the_drum_stability_check_runs_all_its_analyses_under_one_lock(tmp_path, monkeypatch):
    from roadie_analysis import drumstability
    (tmp_path / "drums_hh.flac").write_bytes(b"")
    seen = []

    def fake_run(*a, **kw):
        seen.append(os.environ.get(jobqueue.HELD_ENV))
        raise ValueError("stop here")                   # no audio is analysed
    monkeypatch.setattr(drumstability, "run", fake_run)
    with pytest.raises(SystemExit, match="stop here"):
        drumstability.main([str(tmp_path), "--stems", "hh"])
    assert seen == [os.environ["ROADIE_QUEUE_LOCK"]]
    assert jobqueue.HELD_ENV not in os.environ          # and it let go

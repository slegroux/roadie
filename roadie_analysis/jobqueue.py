"""One machine-wide lock for Roadie's heavy jobs (`roadie split`, `roadie drums`), so jobs started from
several terminals wait their turn instead of fighting for the cores and the graphics memory.

    with heavy_job("roadie drums Glacier 3000"):
        ...                                  # the heavy part

The lock is `fcntl.flock` on a file (default `~/Library/Caches/roadie/heavy-job.lock`, `ROADIE_QUEUE_LOCK`
moves it): the kernel frees it when the holder dies, however it dies. The file's JSON (pid, label, start
time) only tells a waiter who holds it. `ROADIE_NO_QUEUE=1` skips the queue.

Nesting never waits on itself. In the same thread a second `heavy_job` is a no-op. A child of the holder
(a pool worker, a subprocess) finds `ROADIE_QUEUE_HELD=<lock path>` in its environment, tries the lock
once, finds it held, and passes through; if nobody holds it (the marker outlived its holder) it takes it
like anyone else. Another thread of the holder's process is a different job and does wait.

Shell use: `python -I jobqueue.py run <label> -- <command...>` takes the lock, then replaces itself with
the command, which keeps the lock (the descriptor is inherited) until it and its children are gone.
Standard library only: the launcher runs it before any venv is checked.
"""

import contextlib
import fcntl
import json
import os
import sys
import threading
import time

HELD_ENV = "ROADIE_QUEUE_HELD"
_local = threading.local()        # .depth: this thread's open heavy_job blocks
_process_holds = 0                # heavy_jobs of this interpreter that own the lock (0 or 1)


def lock_path() -> str:
    return os.environ.get("ROADIE_QUEUE_LOCK") or os.path.join(os.path.expanduser("~"), "Library", "Caches", "roadie", "heavy-job.lock")


def bypassed() -> bool:
    return os.environ.get("ROADIE_NO_QUEUE", "") not in ("", "0")


def _holder(fd: int) -> dict:
    try:
        return json.loads(os.pread(fd, 4096, 0).decode("utf-8", "replace") or "{}")
    except (OSError, ValueError):
        return {}


def _since(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if time.strftime("%Y-%m-%d", time.localtime(t)) == time.strftime("%Y-%m-%d") \
        else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _span(s: float) -> str:
    s = int(s)
    return "%d min %02d s" % (s // 60, s % 60) if s >= 60 else "%d s" % s


def _holder_text(info: dict) -> str:
    if not info.get("pid"):
        return "another heavy job"
    return "%s (pid %s, since %s)" % (info.get("label") or "another heavy job", info["pid"], _since(info["started"]) if info.get("started") else "?")


class _Lock:
    """An acquired (or passed-through) heavy_job. `fd` is None when this block does not own the lock."""

    def __init__(self, fd, previous_marker=None, owns_depth=False):
        self.fd, self.previous_marker, self.owns_depth = fd, previous_marker, owns_depth

    def release(self):
        global _process_holds
        if self.owns_depth:
            _local.depth -= 1
        if self.fd is None:
            return
        _process_holds -= 1
        if self.previous_marker is None:
            os.environ.pop(HELD_ENV, None)
        else:
            os.environ[HELD_ENV] = self.previous_marker
        try:
            os.ftruncate(self.fd, 0)     # nobody reads a dead holder's note
        except OSError:
            pass
        os.close(self.fd)                # closing drops the flock
        self.fd = None


def acquire(label: str, repeat: float = 60.0, poll: float = 0.25, stream=None) -> _Lock:
    """Take the lock, waiting for it. Prints who holds it to `stream` (stderr) once, then every `repeat` s."""
    global _process_holds
    stream = stream or sys.stderr
    if bypassed():
        return _Lock(None)
    if getattr(_local, "depth", 0) > 0:
        _local.depth += 1
        return _Lock(None, owns_depth=True)
    path = lock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.monotonic()
    told = None                       # when the last waiting line went out
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
            if os.environ.get(HELD_ENV) == path and _process_holds == 0:
                os.close(fd)          # a descendant of the holder: it already has the lock
                return _Lock(None)
            now = time.monotonic()
            info = _holder(fd)
            if told is None and (info.get("pid") or now - t0 > 2):   # a new holder writes its note a moment after locking
                print("roadie: waiting for %s; ROADIE_NO_QUEUE=1 skips the queue" % _holder_text(info), file=stream, flush=True)
                told = now
            elif told is not None and now - told >= repeat:
                print("roadie: still waiting for %s (%s so far)" % (_holder_text(info), _span(now - t0)), file=stream, flush=True)
                told = now
            time.sleep(poll)
    except BaseException:
        os.close(fd)
        raise
    waited = time.monotonic() - t0
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps({"pid": os.getpid(), "label": label, "started": time.time()}).encode(), 0)
    previous = os.environ.get(HELD_ENV)
    os.environ[HELD_ENV] = path
    _process_holds += 1
    _local.depth = 1
    if told is not None:
        print("roadie: lock acquired after waiting %s" % _span(waited), file=stream, flush=True)
    return _Lock(fd, previous, owns_depth=True)


@contextlib.contextmanager
def heavy_job(label: str, repeat: float = 60.0, poll: float = 0.25, stream=None):
    lock = acquire(label, repeat, poll, stream)
    try:
        yield
    finally:
        lock.release()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4 or argv[0] != "run" or argv[2] != "--":
        print("usage: jobqueue.py run <label> -- <command> [args...]", file=sys.stderr)
        return 2
    lock = acquire(argv[1])
    if lock.fd is not None:
        os.set_inheritable(lock.fd, True)     # the command keeps the lock: its descriptor outlives the exec
    try:
        os.execvp(argv[3], argv[3:])
    except OSError as e:
        print("roadie: cannot run %s: %s" % (argv[3], e), file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())

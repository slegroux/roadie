"""A small content-addressed store on disk, for results that depend only on some audio samples and fixed settings.

    k = key(samples, sr, {"threshold": 1.5, ...})       # sha256 over the sample bytes, the rate, the channels, the settings
    hit = get(k)                                        # the stored JSON value, or None
    put(k, value)

Used by `drumroles.hit_groups` for the detected hits and their groups (a drum stem's slice is the same samples in every
analysis of the same audio, whatever the tempo or t0 says), so a second analysis does not detect again.

- Where: `ROADIE_CACHE_DIR` if set, else `~/Library/Caches/roadie/drum-hits/`. One file per entry, `<key>.json.gz`
  (gzip-compressed JSON, never pickle).
- Size: at most `ROADIE_CACHE_MAX_MB` megabytes (default 300). After a write, when the folder is over the limit, the
  entries read or written longest ago are removed first (a read touches its file).
- `ROADIE_NO_CACHE=1`: no reads and no writes.
- Safe with several processes: an entry is written to a temp file in the same folder, then `os.replace`d. An entry that
  cannot be read (truncated, not JSON, not gzip) is a miss, and the next `put` overwrites it. A failing disk
  (full, read-only) loses the cache, never the analysis.
Standard library and numpy only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time

import numpy as np

DEFAULT_MAX_MB = 300
SUFFIX = ".json.gz"
_STALE_TMP_S = 3600


def enabled() -> bool:
    return os.environ.get("ROADIE_NO_CACHE", "") in ("", "0")


def cache_dir() -> str:
    return os.environ.get("ROADIE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), "Library", "Caches", "roadie", "drum-hits")


def max_bytes() -> int:
    try:
        return int(float(os.environ.get("ROADIE_CACHE_MAX_MB", DEFAULT_MAX_MB)) * 1024 * 1024)
    except ValueError:
        return DEFAULT_MAX_MB * 1024 * 1024


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("cannot cache a %s" % type(o).__name__)


def key(samples, sr: int, settings: dict) -> str:
    """sha256 over the raw sample bytes (with their dtype and shape, so the channel count), the sample rate and the settings.
    `samples` is an array, or a list of arrays (their order counts)."""
    arrays = [np.ascontiguousarray(a) for a in (samples if isinstance(samples, (list, tuple)) else [samples])]
    h = hashlib.sha256()
    h.update(json.dumps({"arrays": [[a.dtype.str, a.shape] for a in arrays], "sr": int(sr), "settings": settings}, sort_keys=True,
                        default=_jsonable).encode())
    for a in arrays:
        h.update(memoryview(a).cast("B"))                 # no copy
    return h.hexdigest()


def _path(k: str) -> str:
    return os.path.join(cache_dir(), k + SUFFIX)


def get(k: str):
    """The value stored under `k`, or None (a miss, and also an entry that cannot be read)."""
    if not enabled():
        return None
    p = _path(k)
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            value = json.load(f)
    except Exception:                                 # missing, truncated, garbled (zlib.error), not JSON: a miss, never an error
        return None
    try:
        os.utime(p)                                   # read: the entry is the newest (a read-only folder still serves its entries)
    except OSError:
        pass
    return value


def put(k: str, value) -> None:
    """Store `value` (JSON, numpy scalars and arrays allowed) under `k`, then evict down to the size limit."""
    if not enabled():
        return
    d = cache_dir()
    tmp = None
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=SUFFIX, dir=d)
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as z:
            z.write(json.dumps(value, default=_jsonable, separators=(",", ":")).encode())
        os.replace(tmp, _path(k))
        tmp = None
        evict(d)
    except (OSError, TypeError, ValueError):          # a full disk, or a value JSON cannot hold: the run goes on without the entry
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def evict(d: str | None = None, limit: int | None = None) -> int:
    """Remove the least recently used entries until the folder holds at most `limit` bytes (default: the size limit). Returns
    the number of entries removed. Temp files of a writer that died are removed after an hour."""
    d = d or cache_dir()
    limit = max_bytes() if limit is None else limit
    now, entries = time.time(), []
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    for n in names:
        p = os.path.join(d, n)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if n.startswith(".tmp-"):
            if now - st.st_mtime > _STALE_TMP_S:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        elif n.endswith(SUFFIX):
            entries.append((st.st_mtime, n, st.st_size, p))
    total, removed = sum(e[2] for e in entries), 0
    for _, _, size, p in sorted(entries):
        if total <= limit:
            break
        try:
            os.unlink(p)
        except FileNotFoundError:                     # another worker removed it: it no longer counts
            total -= size
            continue
        except OSError:
            continue
        total -= size
        removed += 1
    return removed


def size_bytes(d: str | None = None) -> int:
    """The entries' total size, for reports."""
    d = d or cache_dir()
    try:
        return sum(os.stat(os.path.join(d, n)).st_size for n in os.listdir(d) if n.endswith(SUFFIX))
    except OSError:
        return 0

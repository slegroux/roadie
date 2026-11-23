#!/usr/bin/env python3
"""The store of detected hits (roadie_analysis/cache.py) and its use by hit_groups and section_job. Fast set: a few
seconds of synthetic noise, one short stem. conftest.py gives every test its own ROADIE_CACHE_DIR."""

import gzip
import os
import time

import numpy as np
import pytest
import soundfile as sf

from roadie_analysis import cache, drumroles, drums, drumsections

SR = 44100


def samples(seed=1, n=SR // 2):
    return np.random.default_rng(seed).standard_normal((n, 2)).astype(np.float32)


# ---------------------------------------------------------------- the store

def test_a_stored_value_is_found_again():
    k = cache.key(samples(), SR, {"threshold": 1.5})
    assert cache.get(k) is None
    cache.put(k, {"hits": [1, 2.5, None], "name": "hh"})
    assert cache.get(k) == {"hits": [1, 2.5, None], "name": "hh"}


def test_a_changed_sample_rate_channel_count_setting_or_version_is_another_entry():
    x = samples()
    base = cache.key(x, SR, {"threshold": 1.5, "version": 1})
    y = x.copy()
    y[1234, 0] = np.nextafter(y[1234, 0], np.float32(1.0))             # one sample, one bit
    assert cache.key(y, SR, {"threshold": 1.5, "version": 1}) != base
    assert cache.key(x, SR + 1, {"threshold": 1.5, "version": 1}) != base
    assert cache.key(x[:, :1], SR, {"threshold": 1.5, "version": 1}) != base
    assert cache.key(x, SR, {"threshold": 1.6, "version": 1}) != base
    assert cache.key(x, SR, {"threshold": 1.5, "version": 2}) != base
    assert cache.key(x.copy(), SR, {"version": 1, "threshold": 1.5}) == base      # same content, same key
    cache.put(base, {"a": 1})
    assert cache.get(cache.key(y, SR, {"threshold": 1.5, "version": 1})) is None


def test_a_truncated_or_garbled_entry_is_a_miss_and_is_overwritten():
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": list(range(500))})
    path = os.path.join(cache.cache_dir(), k + cache.SUFFIX)
    data = open(path, "rb").read()
    open(path, "wb").write(data[: len(data) // 2])
    assert cache.get(k) is None
    open(path, "wb").write(b"not gzip at all")
    assert cache.get(k) is None
    open(path, "wb").write(gzip.compress(b"{not json"))
    assert cache.get(k) is None
    cache.put(k, {"a": 1})
    assert cache.get(k) == {"a": 1}


def test_a_garbled_deflate_stream_is_a_miss_not_an_error():
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": list(range(2000))})
    path = os.path.join(cache.cache_dir(), k + cache.SUFFIX)
    data = bytearray(open(path, "rb").read())
    for i in range(20, 60):                                                # flip bytes inside the stream, header intact
        data[i] ^= 0xFF
    open(path, "wb").write(bytes(data))
    assert cache.get(k) is None
    cache.put(k, {"a": 1})
    assert cache.get(k) == {"a": 1}


def test_a_value_json_cannot_hold_loses_the_entry_not_the_run():
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": {1, 2}})                                            # a set
    cache.put(k, {"a": {(1, 2): 3}})                                       # a tuple key
    assert cache.get(k) is None
    assert not [n for n in os.listdir(cache.cache_dir()) if n.startswith(".tmp-")]


def test_a_read_still_serves_the_entry_when_its_time_cannot_be_touched(monkeypatch):
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": 1})
    monkeypatch.setattr(os, "utime", lambda *a, **kw: (_ for _ in ()).throw(PermissionError()))
    assert cache.get(k) == {"a": 1}


def test_an_entry_another_worker_removed_still_counts_off_the_total(monkeypatch):
    d = cache.cache_dir()
    keys = [cache.key(samples(i), SR, {}) for i in range(3)]
    rng = np.random.default_rng(1)
    for i, k in enumerate(keys):
        cache.put(k, {"pad": rng.bytes(5000).hex()})
        os.utime(os.path.join(d, k + cache.SUFFIX), (1000 + i, 1000 + i))
    paths = [os.path.join(d, k + cache.SUFFIX) for k in keys]
    size = os.stat(paths[2]).st_size
    real = os.unlink
    def gone(p):                                                           # the oldest is removed by someone else just before
        real(p)
        raise FileNotFoundError(p)
    monkeypatch.setattr(os, "unlink", lambda p: gone(p) if p == paths[0] else real(p))
    cache.evict(d, limit=2 * size + 100)                                   # room for two: only the oldest has to go
    assert not os.path.exists(paths[0]) and os.path.exists(paths[1]) and os.path.exists(paths[2])


def test_over_the_limit_the_entry_used_longest_ago_goes_first(monkeypatch):
    keys = [cache.key(samples(i), SR, {}) for i in range(5)]
    rng = np.random.default_rng(0)
    values = [{"pad": rng.bytes(10000).hex(), "i": i} for i in range(5)]       # random bytes do not compress: about 10 KB each
    for k, v in zip(keys, values):
        cache.put(k, v)
    d = cache.cache_dir()
    sizes = [os.stat(os.path.join(d, k + cache.SUFFIX)).st_size for k in keys]
    now = time.time()
    for i, k in enumerate(keys):                                           # ages: 0 the oldest, 4 the newest
        os.utime(os.path.join(d, k + cache.SUFFIX), (now - 100 + i, now - 100 + i))
    assert cache.get(keys[0]) is not None                                  # a read makes the oldest the newest
    monkeypatch.setenv("ROADIE_CACHE_MAX_MB", str((sum(sizes) - sizes[1] + 1) / 1024 / 1024))
    cache.put(keys[4], values[4])                                          # a write over the limit evicts
    left = {k for k in keys if cache.get(k) is not None}
    assert left == {keys[0], keys[2], keys[3], keys[4]}                    # keys[1] was the least recently used
    assert cache.size_bytes() <= cache.max_bytes()


def test_no_cache_env_reads_and_writes_nothing(monkeypatch):
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": 1})
    monkeypatch.setenv("ROADIE_NO_CACHE", "1")
    assert cache.get(k) is None                                            # no read, though the entry is there
    k2 = cache.key(samples(2), SR, {})
    cache.put(k2, {"a": 2})                                                # no write
    monkeypatch.delenv("ROADIE_NO_CACHE")
    assert cache.get(k) == {"a": 1} and cache.get(k2) is None


def test_a_folder_that_cannot_be_written_loses_the_cache_not_the_run(tmp_path, monkeypatch):
    blocked = tmp_path / "file-not-folder"
    blocked.write_text("x")
    monkeypatch.setenv("ROADIE_CACHE_DIR", str(blocked / "inside"))
    k = cache.key(samples(), SR, {})
    cache.put(k, {"a": 1})
    assert cache.get(k) is None


# ---------------------------------------------------------------- the analysis


def test_an_entry_missing_a_part_is_a_miss():
    assert drumroles._unpack({"groups": []}) is None
    assert drumroles._unpack({"hat_like": 0.1, "grid_hits": 0, "hits": [], "notes": [], "groups": []}) is not None

def hat_stem(bars=2, bpm=120.0, seed=3):
    """Mono stem of closed hats on every 16th and one tom a bar: short, quick to analyse."""
    step = 60.0 / bpm / 4
    x = np.zeros(int((bars * 16 * step + 0.7) * SR))
    rng = np.random.default_rng(seed)
    for i in range(bars * 16):
        n = int(0.25 * SR)
        t = np.arange(n) / SR
        h = np.diff(rng.standard_normal(n + 1)) * np.exp(-t / 0.01) * 0.4
        o = int(i * step * SR)
        x[o:o + n] += h
    return x.astype(np.float32)


def count_detections(monkeypatch):
    calls = []
    real = drumroles._detect
    monkeypatch.setattr(drumroles, "_detect", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def strip(res):
    return repr(sorted(res.items()))


@pytest.mark.slow
def test_a_second_analysis_of_the_same_audio_reuses_the_hits_whatever_the_grid_says(monkeypatch):
    x = hat_stem()
    calls = count_detections(monkeypatch)
    plain = drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0)
    assert len(calls) == 1
    cold = drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    warm = drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 2                                                 # the cold cached run detected, the warm one did not
    assert strip(cold) == strip(plain) == strip(warm)
    moved = drumroles.hit_groups(x, 120.004, 0.002, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 2                                                 # another tempo and t0: the same hits
    assert strip(moved) == strip(drumroles.hit_groups(x, 120.004, 0.002, sr=SR, stability_runs=0))      # (this one detects: 3)
    assert moved["hits"] != warm["hits"]                                   # ...placed on the new grid
    drumroles.hit_groups(x[:-1], 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 4                                                 # one sample less: detected again


@pytest.mark.slow
def test_a_changed_setting_or_a_bumped_version_detects_again(monkeypatch):
    x = hat_stem()
    calls = count_detections(monkeypatch)
    for _ in range(2):
        drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 1
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, threshold=1.4, cache=True)
    assert len(calls) == 2
    monkeypatch.setattr(drumroles, "CACHE_VERSION", drumroles.CACHE_VERSION + 1)
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 3
    monkeypatch.setattr(drumroles, "_source_hash", lambda: "another source")   # an edit to the detector's source changes the key by itself
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert len(calls) == 4


@pytest.mark.slow
def test_no_cache_and_grid_onsets_never_touch_the_folder(monkeypatch):
    x = hat_stem()
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=False)
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True, grid_onsets=True)    # its hits depend on the grid
    monkeypatch.setenv("ROADIE_NO_CACHE", "1")
    drumroles.hit_groups(x, 120.0, 0.0, sr=SR, stability_runs=0, cache=True)
    assert cache.size_bytes() == 0


@pytest.mark.slow
def test_section_job_takes_no_cache_down_to_the_detector(tmp_path, monkeypatch):
    f = tmp_path / "drums_hh.wav"
    sf.write(str(f), hat_stem(), SR)
    sec = {"name": "A", "start_bar": 0, "bars": 2, "first": True, "to_end": True}
    drumsections.section_job(f, 120.0, 0.0, sec, None, False)
    assert cache.size_bytes() == 0
    a = drumsections.section_job(f, 120.0, 0.0, sec, None, True)
    assert cache.size_bytes() > 0
    calls = count_detections(monkeypatch)
    b = drumsections.section_job(f, 120.004, 0.002, sec, None, True)
    assert not calls
    assert len(a["res"]["hits"]) == len(b["res"]["hits"]) > 0


def test_a_group_reading_is_read_once_per_set_of_cuts(monkeypatch):
    rng = np.random.default_rng(4)
    cuts = [rng.standard_normal((int(0.2 * SR), 1)) * np.exp(-np.arange(int(0.2 * SR)) / (0.01 * SR))[:, None] for _ in range(5)]
    reads = []
    real = drumsections._group_reading
    monkeypatch.setattr(drumsections, "_group_reading", lambda *a: reads.append(1) or real(*a))
    first = drumsections.group_reading(cuts, SR, cache=True)
    assert drumsections.group_reading(cuts, SR, cache=True) == first == real(cuts, SR)
    assert len(reads) == 1                                                 # the cold one; the warm one and the direct call above are not counted
    assert drumsections.group_reading(cuts[:4], SR, cache=True) != first
    assert len(reads) == 2


def test_roadie_drums_no_cache_is_passed_to_the_analysis(monkeypatch, tmp_path, capsys):
    import roadie_drums
    seen = {}

    def fake(sd, **kw):
        seen.update(kw)
        raise ValueError("stop here")
    monkeypatch.setattr(roadie_drums, "stems_dir_of", lambda a: tmp_path)
    monkeypatch.setattr(drums, "drum_roles", fake)
    assert roadie_drums.main([str(tmp_path), "--no-cache"]) == 1
    assert seen["cache"] is False
    roadie_drums.main([str(tmp_path)])
    assert seen["cache"] is True


@pytest.mark.parametrize("flag,expect", [([], True), (["--no-cache"], False)])
def test_drumstability_no_cache_is_passed_to_every_run(monkeypatch, tmp_path, flag, expect):
    from roadie_analysis import drumstability
    seen = []
    monkeypatch.setattr(drumstability, "run", lambda *a, **k: seen.append(k.get("cache")) or (_ for _ in ()).throw(ValueError("stop")))
    monkeypatch.setattr(drums, "find_stems", lambda d: {"hh": tmp_path})
    with pytest.raises(SystemExit):
        drumstability.main([str(tmp_path)] + flag)
    assert seen == [expect]


@pytest.mark.slow
def test_drums_json_hit_times_are_the_same_with_the_cache_cold_warm_or_off(tmp_path, monkeypatch):
    """drums.json keeps each role's hit_times by default (a reference profile reads them): the cache must not move one."""
    sf.write(str(tmp_path / "drums_hh.wav"), hat_stem(), SR)
    secs = [{"name": "A", "start_bar": 0, "bars": 2}]

    def times():
        rec = drums.drum_roles(tmp_path, tempo=120.0, t0=0.0, sections=secs, write=False, jobs=1)
        return {r["name"]: r["hit_times"] for r in rec["stems"]["hh"]["roles"]}
    cold, warm = times(), times()
    assert cache.size_bytes() > 0
    monkeypatch.setenv("ROADIE_NO_CACHE", "1")
    off = times()
    assert cold == warm == off and sum(len(t) for t in cold.values()) > 0, (cold, warm, off)

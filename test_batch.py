#!/usr/bin/env python3
"""Tests for batch.py — tiny dummy files, fake `roadie` and `rclone` shell scripts."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

import batch

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_ROADIE = r"""#!/bin/sh
echo "$@" >> "$FAKE_LOG"
case "$1" in
split)
  src="$3"; out="$6"; base=$(basename "$src"); name="${base%.*}"
  if [ -n "$FAKE_SLOW" ]; then case "$base" in *"$FAKE_SLOW"*) sleep 30;; esac; fi
  mkdir -p "$out/${name}_120bpm_stems"
  echo wav > "$out/${name}_120bpm.wav"
  echo f > "$out/${name}_120bpm_stems/vocals.flac" ;;
sections) echo '{}' > "$2/sections.json" ;;
als) p="${2%_stems}"; echo als > "$p.als" ;;
esac
"""

FAKE_RCLONE = r"""#!/bin/sh
echo "$@" >> "$FAKE_RCLONE_LOG"
case "$1" in
lsf) if [ -d "$2" ]; then ls -p "$2"; else exit 3; fi ;;
copy) [ -n "$FAKE_RCLONE_FAIL" ] && exit 1; mkdir -p "$3"; cp -R "$2"/. "$3"/ ;;
check) diff -r "$2" "$3" ;;
esac
"""


def touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    for name, body in (("roadie", FAKE_ROADIE), ("rclone", FAKE_RCLONE)):
        f = tmp_path / ("fake_" + name)
        f.write_text(body)
        f.chmod(0o755)
    monkeypatch.setenv("ROADIE_BATCH_ROADIE", str(tmp_path / "fake_roadie"))
    monkeypatch.setenv("ROADIE_BATCH_RCLONE", str(tmp_path / "fake_rclone"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "roadie.log"))
    monkeypatch.setenv("FAKE_RCLONE_LOG", str(tmp_path / "rclone.log"))
    monkeypatch.setattr(batch, "probe_duration", lambda p: 0.0)
    monkeypatch.setattr(batch, "KILL_GRACE", 2)
    monkeypatch.setattr(batch, "SPLIT_NODUR", 2)
    monkeypatch.setattr(batch, "disk_free", lambda p: 100 * 1024 ** 3)
    return tmp_path


def plan(tmp, *roots, dest=None):
    out = tmp / "manifest.tsv"
    argv = ["plan"] + [str(r) for r in roots] + ["--out", str(out)]
    argv += ["--dest", str(dest or tmp / "dest")]
    assert batch.main(argv) == 0
    return out


def excluded(tmp):
    rows = (tmp / "manifest.excluded.tsv").read_text().splitlines()[1:]
    return dict(r.split("\t") for r in rows)


def run(tmp, manifest, *extra):
    return batch.main(["run", str(manifest), "--work", str(tmp / "cache" / "work"),
                       "--state", str(tmp / "cache" / "state.jsonl"),
                       "--logs", str(tmp / "cache" / "logs"),
                       "--report-dir", str(tmp / "reports")] + list(extra))


def states(tmp):
    p = tmp / "cache" / "state.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def calls(tmp, name="roadie.log"):
    p = tmp / name
    return p.read_text().splitlines() if p.exists() else []


# ---------------------------------------------------------------- plan

def test_plan_excludes_conflicts_packs_processed_and_duplicates(env):
    tmp = env
    st, ref = tmp / "Studies", tmp / "References"
    touch(st / "Carry On.wav")
    touch(st / "Carry On.mp3")
    touch(ref / "Carry On.flac")                       # lossless, but Studies wins the tie
    touch(ref / "Kolsch_Full Circle Moment.mp3")
    touch(ref / "Full Circle Moment.wav")
    touch(ref / "The Space In Between.mp3")
    touch(ref / "The Space In Between (Ben Bohmer Remix).mp3")
    touch(ref / "10A - 91 - Some Artist - Night Drive [abcdefghijk].m4a")
    touch(ref / "Night Drive_124bpm.mp3")
    touch(ref / "Breathing (conflicted).mp3")
    touch(ref / "Done Track.mp3")
    (st / "Done Track_120bpm_stems").mkdir()
    touch(st / "Done Track.wav")
    for i in range(5):
        touch(ref / "Joachim FaderPro STEMS" / ("part%d.wav" % i))
        touch(ref / "Synth Pack" / ("riff%d.aif" % i))
    for i in range(4):                                   # under 5 files, but stem-named
        touch(ref / "Pastor X Faderpro - Stems (bpm120)" / ("Space STEMS %d.wav" % i))
    for i in range(5):                                   # "Systems" is not "stems"
        touch(ref / "Systems" / ("Track %d.mp3" % i))
    touch(ref / "Samples" / "hit.wav")
    touch(ref / "Live Project" / "x.wav")
    touch(ref / "._junk.mp3")

    manifest = plan(tmp, st, ref)
    ex = excluded(tmp)
    rows = [dict(zip(batch.COLUMNS, ln.split("\t")))
            for ln in manifest.read_text().splitlines()[1:]]
    new = {os.path.relpath(r["src"], tmp) for r in rows if r["mode"] == "new"}

    assert new == {"Studies/Carry On.wav",
                   "References/Full Circle Moment.wav",
                   "References/The Space In Between.mp3",
                   "References/The Space In Between (Ben Bohmer Remix).mp3",
                   "References/10A - 91 - Some Artist - Night Drive [abcdefghijk].m4a"} | \
                  {"References/Systems/Track %d.mp3" % i for i in range(5)}
    assert ex[str(ref / "Pastor X Faderpro - Stems (bpm120)" / "Space STEMS 0.wav")] == "stem pack"
    assert ex[str(st / "Carry On.mp3")].startswith("duplicate of")
    assert ex[str(ref / "Carry On.flac")] == "duplicate of %s" % (st / "Carry On.wav")
    assert ex[str(ref / "Kolsch_Full Circle Moment.mp3")] == \
        "duplicate of %s" % (ref / "Full Circle Moment.wav")
    assert ex[str(ref / "Night Drive_124bpm.mp3")].startswith("duplicate of")
    assert ex[str(ref / "Breathing (conflicted).mp3")] == "pCloud conflict copy"
    assert ex[str(ref / "Joachim FaderPro STEMS" / "part0.wav")] == "stem pack"
    assert ex[str(ref / "Synth Pack" / "riff3.aif")] == "stem pack"
    stems_dir = str(st / "Done Track_120bpm_stems")
    assert ex[str(st / "Done Track.wav")] == "already processed: " + stems_dir
    assert ex[str(ref / "Done Track.mp3")] == "already processed: " + stems_dir  # whole group
    assert not any("Samples" in p or "Live Project" in p or "._junk" in p for p in ex)
    assert not any("Samples" in r["src"] or "Live Project" in r["src"] for r in rows)

    topup = [r for r in rows if r["mode"] == "topup"]
    assert [r["src"] for r in topup] == [stems_dir] and topup[0]["dest"] == ""
    assert rows[0]["mode"] == "topup"
    carry = next(r for r in rows if r["name"] == "Carry On")
    assert carry["dest"] == "%s/Studies/Carry On" % (tmp / "dest")
    assert len({r["id"] for r in rows}) == len(rows)


def test_norm_key_examples():
    assert batch.norm_key("6A - 126 - 43 - Artist - Title (Extended Mix)") == "title extended mix"
    assert batch.norm_key("8A or 9A - 122 - Foo Bar") == "foo bar"
    assert batch.norm_key("Café_Del-Mar 124BPM F#Maj") == "cafe del mar"
    assert batch.related("kolsch full circle moment", "full circle moment")
    assert not batch.related("the space in between ben bohmer remix", "the space in between")
    assert not batch.related("big night", "night")           # shorter has one word


def test_verify_sources_flags_a_changed_file(env):
    tmp = env
    src = touch(tmp / "Studies" / "a.wav")
    manifest = plan(tmp, tmp / "Studies")
    assert batch.main(["verify-sources", str(manifest)]) == 0
    src.write_bytes(b"changed!")
    assert batch.main(["verify-sources", str(manifest)]) == 1


# ---------------------------------------------------------------- safe_rmtree

def test_safe_rmtree_refuses_outside_and_root(env):
    tmp = env
    root = tmp / "cache" / "work"
    inside = root / "t" / "f"
    touch(inside)
    outside = touch(tmp / "other" / "keep.txt")
    with pytest.raises(RuntimeError):
        batch.safe_rmtree(str(outside.parent), str(root))
    with pytest.raises(RuntimeError):
        batch.safe_rmtree(str(root), str(root))
    with pytest.raises(RuntimeError):
        batch.safe_rmtree(str(root / ".." / "other"), str(root))
    link = root / "t" / "link"
    os.symlink(str(outside.parent), str(link))
    batch.safe_rmtree(str(link), str(root))                  # removes the link only
    assert outside.exists() and not os.path.lexists(link)
    batch.safe_rmtree(str(root / "t"), str(root))
    assert not (root / "t").exists()


# ---------------------------------------------------------------- run

def test_run_processes_uploads_and_cleans_up(env):
    tmp = env
    touch(tmp / "Studies" / "sub" / "Song.mp3")
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest) == 0
    d = tmp / "dest" / "Studies" / "sub" / "Song"
    assert (d / "Song_120bpm_stems" / "vocals.flac").exists()
    assert (d / "Song_120bpm.als").exists()
    assert not list(d.glob("*.wav"))                      # 24-bit working wav not uploaded
    assert os.listdir(tmp / "cache" / "work") == []
    assert [s["status"] for s in states(tmp)] == ["done"]
    assert list((tmp / "reports").glob("batch-report-*.md"))
    assert any(" --create-empty-src-dirs" in c for c in calls(tmp, "rclone.log"))


def test_run_skips_done_rows(env):
    tmp = env
    touch(tmp / "Studies" / "Song.mp3")
    manifest = plan(tmp, tmp / "Studies")
    row_id = manifest.read_text().splitlines()[1].split("\t")[0]
    (tmp / "cache").mkdir()
    (tmp / "cache" / "state.jsonl").write_text(json.dumps({"id": row_id, "status": "done"}) + "\n")
    assert run(tmp, manifest) == 0
    assert calls(tmp) == [] and calls(tmp, "rclone.log") == []


def test_run_skips_a_track_already_on_pcloud(env):
    tmp = env
    touch(tmp / "Studies" / "Song.mp3")
    manifest = plan(tmp, tmp / "Studies")
    touch(tmp / "dest" / "Studies" / "Song" / "Song_120bpm_stems" / "v.flac")
    touch(tmp / "dest" / "Studies" / "Song" / "Song_120bpm.als")
    assert run(tmp, manifest) == 0
    assert calls(tmp) == [] and states(tmp)[0]["status"] == "done"


def test_timeout_records_failed_and_continues(env, monkeypatch):
    tmp = env
    touch(tmp / "Studies" / "a_slow.mp3")
    touch(tmp / "Studies" / "b_fast.mp3")
    monkeypatch.setenv("FAKE_SLOW", "a_slow")
    manifest = plan(tmp, tmp / "Studies")
    t0 = time.monotonic()
    assert run(tmp, manifest) == 0
    assert time.monotonic() - t0 < 5
    got = {s["name"]: s for s in states(tmp)}
    assert got["a_slow"]["status"] == "failed" and "timeout" in got["a_slow"]["error"]
    assert got["b_fast"]["status"] == "done"
    assert os.listdir(tmp / "cache" / "work") == []       # failed row's work dir removed
    assert not (tmp / "dest" / "Studies" / "a_slow").exists()
    assert (tmp / "dest" / "Studies" / "b_fast").exists()


def test_upload_failure_exits_3_keeps_work_and_stops(env, monkeypatch):
    tmp = env
    touch(tmp / "Studies" / "a.mp3")
    touch(tmp / "Studies" / "b.mp3")
    monkeypatch.setenv("FAKE_RCLONE_FAIL", "1")
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest) == 3
    work = tmp / "cache" / "work"
    kept = os.listdir(work)
    assert len(kept) == 1 and kept[0].startswith("Studies-a-")
    assert [s["status"] for s in states(tmp)] == ["upload_failed"]
    assert len([c for c in calls(tmp) if c.startswith("split")]) == 1
    assert "upload_failed" in next((tmp / "reports").glob("*.md")).read_text()


def test_floor_stop(env, monkeypatch):
    tmp = env
    touch(tmp / "Studies" / "a.mp3")
    monkeypatch.setattr(batch, "disk_free", lambda p: 1 * 1024 ** 3)
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest) == 0
    assert [s["status"] for s in states(tmp)] == ["stopped_floor"]
    assert calls(tmp) == []
    assert "Remaining" in next((tmp / "reports").glob("*.md")).read_text()


def test_topup_never_runs_als_when_the_als_exists(env):
    tmp = env
    stems = tmp / "Studies" / "Song_120bpm_stems"
    touch(stems / "vocals.flac")
    touch(tmp / "Studies" / "Song_120bpm.als", b"mine")
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest) == 0
    assert calls(tmp) == ["sections %s" % stems]          # no als, no split
    assert (tmp / "Studies" / "Song_120bpm.als").read_bytes() == b"mine"
    assert (stems / "sections.json").exists()
    # second run: everything present, nothing is called
    (tmp / "cache" / "state.jsonl").unlink()
    (tmp / "roadie.log").unlink()
    assert run(tmp, manifest) == 0
    assert calls(tmp) == []


def test_topup_creates_a_missing_als_without_force(env):
    tmp = env
    stems = tmp / "Studies" / "Song_120bpm_stems"
    touch(stems / "vocals.flac")
    touch(stems / "sections.json")
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest) == 0
    assert calls(tmp) == ["als %s" % stems]


def test_dry_run_touches_nothing(env):
    tmp = env
    touch(tmp / "Studies" / "a.mp3")
    manifest = plan(tmp, tmp / "Studies")
    assert run(tmp, manifest, "--dry-run") == 0
    assert not (tmp / "cache").exists() and not (tmp / "reports").exists()
    assert calls(tmp) == [] and calls(tmp, "rclone.log") == []


def test_sigterm_kills_the_child_group_and_writes_the_report(env, monkeypatch):
    tmp = env
    touch(tmp / "Studies" / "a_slow.mp3")
    manifest = plan(tmp, tmp / "Studies")
    e = dict(os.environ, FAKE_SLOW="a_slow", PYTHONPATH=HERE)
    cmd = [sys.executable, os.path.join(HERE, "batch.py"), "run", str(manifest),
           "--work", str(tmp / "cache" / "work"), "--state", str(tmp / "cache" / "state.jsonl"),
           "--logs", str(tmp / "cache" / "logs"), "--report-dir", str(tmp / "reports")]
    p = subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if calls(tmp):
                break
            time.sleep(0.05)
        assert calls(tmp), "fake split never started"
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=15) == 128 + signal.SIGTERM
    finally:
        if p.poll() is None:
            p.kill()
    assert [s["status"] for s in states(tmp)] == ["interrupted"]
    assert list((tmp / "reports").glob("batch-report-*.md"))
    assert subprocess.run(["pgrep", "-f", "a_slow.mp3"], capture_output=True).returncode != 0


def test_second_run_is_refused_while_the_lock_is_held(env):
    import fcntl
    tmp = env
    touch(tmp / "Studies" / "a.mp3")
    manifest = plan(tmp, tmp / "Studies")
    (tmp / "cache").mkdir()
    with open(tmp / "cache" / "run.pid", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert run(tmp, manifest) == 2
    assert calls(tmp) == []

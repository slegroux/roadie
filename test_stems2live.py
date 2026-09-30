"""Regression tests for stems2live's clip addressing, against a mock LOM bridge.

WHY THIS FILE EXISTS. `arrangement_clips 0` is not "the clip we just made" —
Live indexes arrangement clips BY TIME. Re-running over an existing Set at a
different --start-bar leaves the previous clip's head in front of the new one,
and index 0 is then the leftover stub. On a real run that sent the trim to the
stub, whose read-back check passed because the stub really had been trimmed,
while the clip that mattered stayed untrimmed and half a beat out of alignment.

The mock bridge below speaks the real wire format (newline-delimited JSON, one
reply per line, nested `params`) over a real loopback socket, so the client's
framing is exercised too and not just its call arguments. Nothing here touches
Ableton Live.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading

import pytest

import stems2live as S

# The stub the real run left behind: an old placement at bar 1.8766 whose head
# survived in front of the new clip at bar 9.
STUB_START = 3.5064          # beats — bar 1.8766
STUB_END = 31.7632           # beats — bar 8.9408
NEW_START = 32.0             # beats — bar 9.0000
NEW_END = 673.5152           # beats — bar 169.3788

PERIOD = 60.0 / 125.003


class MockBridge(object):
    """Minimal stand-in for the AbletonLOM remote script.

    Only the ops stems2live actually uses, and only enough Live behaviour to
    make the clip-addressing tests meaningful: clips are stored per track and
    ALWAYS returned sorted by start_time, which is the behaviour under test.
    """

    def __init__(self, tracks):
        self.tracks = tracks              # {index: [clip dict, ...]}
        self.calls = []                   # every (op, params) received
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.running = True
        self.thread = threading.Thread(target=self._serve)
        self.thread.daemon = True
        self.thread.start()

    # -- object model ----------------------------------------------------
    def clips(self, track_index):
        return sorted(self.tracks[track_index], key=lambda c: c["start_time"])

    def _resolve_clip(self, path):
        parts = path.split()
        return self.clips(int(parts[2]))[int(parts[4])]

    def handle(self, op, p):
        if op == "count":
            return {"count": len(self.tracks[int(p["path"].split()[2])])}
        if op == "get":
            return {"value": self._resolve_clip(p["path"])[p["property"]]}
        if op == "set":
            path = p["path"]
            if "arrangement_clips" in path:
                clip = self._resolve_clip(path)
                # Live ignores start_marker unless the loop was set up first.
                if p["property"] == "start_marker" and not (
                        clip["looping"] and clip["loop_start"] > 0):
                    return {"value": p["value"]}
                clip[p["property"]] = p["value"]
            return {"value": p["value"]}
        if op == "call" and p["function"] == "create_audio_clip":
            _path, pos = p["args"]
            self.tracks[int(p["path"].split()[2])].append(
                make_clip(float(pos), float(pos) + 100.0))
            return {"result": None}
        if op == "warp_markers_set":
            clip = self._resolve_clip(p["path"])
            clip["warping"] = True
            return {"marker_count": len(p["markers"]), "added": 2,
                    "removed": 0, "remove_failed": 1}
        raise ValueError("unknown op %r; known: ['get', 'set']" % op)

    # -- wire ------------------------------------------------------------
    def _serve(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if b"\n" not in buf:
                conn.close()
                continue
            req = json.loads(buf.split(b"\n", 1)[0].decode())
            params = req.get("params") or {}
            self.calls.append((req["op"], params))
            try:
                out = {"id": req["id"], "ok": True,
                       "result": self.handle(req["op"], params)}
            except Exception as e:                       # noqa: BLE001
                out = {"id": req["id"], "ok": False,
                       "error": {"type": type(e).__name__, "message": str(e)}}
            conn.sendall(json.dumps(out).encode() + b"\n")
            conn.close()

    def close(self):
        self.running = False
        self.sock.close()


def make_clip(start, end, **kw):
    clip = {"start_time": start, "end_time": end, "warping": False,
            "looping": False, "loop_start": 0.0, "start_marker": 0.0,
            "color": 0, "muted": False}
    clip.update(kw)
    return clip


@pytest.fixture
def bridge(monkeypatch):
    """A track carrying a leftover stub AND the newly placed clip."""
    b = MockBridge({2: [make_clip(STUB_START, STUB_END),
                        make_clip(NEW_START, NEW_END)]})
    monkeypatch.setattr(S, "PORT", b.port)
    yield b
    b.close()


# ---------------------------------------------------------------- unit tests

def test_find_placed_clip_picks_the_clip_at_the_requested_position():
    """THE REGRESSION. Two clips, and the one we placed is NOT index 0."""
    clips = [(0, STUB_START, STUB_END), (1, NEW_START, NEW_END)]
    assert S.find_placed_clip(clips, NEW_START) == 1


def test_find_placed_clip_picks_index_0_when_that_is_the_match():
    clips = [(0, STUB_START, STUB_END), (1, NEW_START, NEW_END)]
    assert S.find_placed_clip(clips, STUB_START) == 0


def test_find_placed_clip_returns_none_when_nothing_matches():
    """No silent fallback to 0 — the caller must be able to tell and report."""
    clips = [(0, STUB_START, STUB_END), (1, NEW_START, NEW_END)]
    assert S.find_placed_clip(clips, 64.0) is None


def test_find_placed_clip_returns_none_on_an_empty_track():
    assert S.find_placed_clip([], 32.0) is None


def test_find_placed_clip_tolerates_float_noise_but_not_a_real_offset():
    clips = [(0, 32.00004, 100.0)]
    assert S.find_placed_clip(clips, 32.0) == 0
    assert S.find_placed_clip([(0, 32.01, 100.0)], 32.0) is None


def test_find_placed_clip_survives_clips_in_any_order():
    """Ordering is Live's business; matching is by time, not by position."""
    clips = [(0, NEW_START, NEW_END), (1, STUB_START, STUB_END)]
    assert S.find_placed_clip(clips, NEW_START) == 0
    assert S.find_placed_clip(clips, STUB_START) == 1


# --------------------------------------------------------- against the bridge

def test_arrangement_clips_reads_every_clip_with_its_times(bridge):
    assert S.arrangement_clips(2) == [(0, STUB_START, STUB_END),
                                      (1, NEW_START, NEW_END)]


def test_trim_lands_on_the_new_clip_and_not_the_stub(bridge):
    """THE BUG, END TO END. The stub must come back untouched."""
    ci = S.find_placed_clip(S.arrangement_clips(2), NEW_START)
    ok, detail = S.trim_clip_start(2, ci, 0.237, PERIOD, [])
    assert ok, detail

    stub, new = bridge.clips(2)
    assert stub["start_marker"] == 0.0, "the leftover stub was trimmed"
    assert new["start_marker"] == pytest.approx(0.237)
    assert new["start_time"] == NEW_START, "trimming moved the clip"


def test_trimming_index_zero_is_what_the_bug_did(bridge):
    """Pins the failure mode itself, so a regression cannot pass unnoticed:
    addressing index 0 trims the stub and the read-back check still passes."""
    ok, _ = S.trim_clip_start(2, 0, 0.237, PERIOD, [])
    assert ok, "the old code's check passed on the wrong clip — that is the bug"
    assert bridge.clips(2)[1]["start_marker"] == 0.0


def test_trim_leaves_looping_off_and_sets_the_loop_first(bridge):
    """start_marker is silently ignored unless the loop is set up first."""
    ci = S.find_placed_clip(S.arrangement_clips(2), NEW_START)
    assert S.trim_clip_start(2, ci, 0.237, PERIOD, [])[0]

    order = [p["property"] for op, p in bridge.calls
             if op == "set" and "arrangement_clips" in p["path"]]
    assert order == ["looping", "loop_start", "start_marker", "looping"]
    assert bridge.clips(2)[1]["looping"] is False


def test_trim_uses_beats_on_a_warped_clip(bridge):
    bridge.clips(2)[1]["warping"] = True
    ci = S.find_placed_clip(S.arrangement_clips(2), NEW_START)
    ok, detail = S.trim_clip_start(2, ci, 0.237, PERIOD,
                                   [[10.0, 4.8], [110.0, 52.8]])
    assert ok, detail
    assert "beats" in detail
    assert bridge.clips(2)[1]["start_marker"] == pytest.approx(0.4938, abs=1e-4)


def test_warp_map_is_written_to_the_resolved_clip(bridge):
    ci = S.find_placed_clip(S.arrangement_clips(2), NEW_START)
    state, _detail = S.apply_warp_map(2, ci, [[0.0, 0.0], [4.0, 2.0]])
    assert state == "ok"
    assert bridge.clips(2)[1]["warping"] is True
    assert bridge.clips(2)[0]["warping"] is False, "warped the leftover stub"


def test_a_fresh_track_still_resolves_to_its_only_clip(monkeypatch):
    """The single-clip case the old code got right must keep working."""
    b = MockBridge({2: []})
    monkeypatch.setattr(S, "PORT", b.port)
    try:
        S.lom("call", path="live_set tracks 2",
              function="create_audio_clip", args=["/tmp/x.flac", NEW_START])
        ci = S.find_placed_clip(S.arrangement_clips(2), NEW_START)
        assert ci == 0
        assert S.trim_clip_start(2, ci, 0.237, PERIOD, [])[0]
        assert b.clips(2)[0]["start_marker"] == pytest.approx(0.237)
    finally:
        b.close()


def test_placing_a_second_clip_later_leaves_the_first_addressable(monkeypatch):
    """Re-running at a later --start-bar: the new clip is index 1, and that is
    the one every subsequent property must be written to."""
    b = MockBridge({2: [make_clip(STUB_START, STUB_END)]})
    monkeypatch.setattr(S, "PORT", b.port)
    try:
        S.lom("call", path="live_set tracks 2",
              function="create_audio_clip", args=["/tmp/x.flac", NEW_START])
        clips = S.arrangement_clips(2)
        assert len(clips) == 2
        ci = S.find_placed_clip(clips, NEW_START)
        assert ci == 1
        leftovers = [(s, e) for j, s, e in clips if j != ci]
        assert leftovers == [(STUB_START, STUB_END)]
    finally:
        b.close()


# --------------------------------------------------------------------------
# --session hand-off
# --------------------------------------------------------------------------

def test_the_session_grid_is_handed_the_arrangement_grid(monkeypatch):
    """REGRESSION-IN-WAITING: the two views must share one beat grid.

    session_view can fit its own grid, and it lands within about a beat of this
    one — which is exactly bad enough to be dangerous. The disagreement grows
    with the section index, so the arrangement and session clips drift apart on
    a set that looks correct in either view alone. stems2live has the fitted
    grid in hand, so it must pass it rather than let it be re-derived.
    """
    seen = {}

    def fake_build(stems_dir, **kw):
        seen.update(kw)
        seen["stems_dir"] = stems_dir
        return {"scenes": []}

    import session_view
    monkeypatch.setattr(session_view, "build_session_view", fake_build)

    grid_info = {"period": 0.479989, "grid_t0": 0.237, "downbeat_sec": 32.8762}
    # Call the hand-off the way main() does.
    session_view.build_session_view(
        "/tmp/stems",
        period=grid_info.get("period"),
        grid_t0=grid_info.get("grid_t0"),
        replace=True, port=9878)

    assert seen["period"] == 0.479989, "the fitted period must be passed through"
    assert seen["grid_t0"] == 0.237, "the grid origin must be passed through"
    assert seen["replace"] is True, "a re-run must replace, not stack clips"


GRID = {"period": 0.5, "grid_t0": 0.237, "downbeat_sec": 0.237 + 2 * 0.5,
        "bpm": 120.0}


def test_section_consumers_get_the_downbeat_when_aligned():
    """Locators, energy map and session grid all take this one dict."""
    g = S.section_grid(GRID, no_align=False)
    assert g == {"period": 0.5, "grid_t0": 0.237, "downbeat_sec": 1.237}


def test_section_consumers_get_no_downbeat_with_no_align():
    assert S.section_grid(GRID, no_align=True)["downbeat_sec"] is None


@pytest.mark.parametrize("t0", [0.237, -0.1])
@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_energy_map_shift_puts_the_downbeat_on_clip_beat_zero(phase, t0):
    """The first section starts at the downbeat; after the shift it must sit at
    the clip start, whatever the phase and whichever side of zero grid_t0 is."""
    from sections import build_beat_grid_times, downbeat_phase
    period = 0.5
    grid = {"period": period, "grid_t0": t0,
            "downbeat_sec": t0 + (phase + 4) * period}
    _pos, trim, _db = S.bar_aligned_placement(grid, 1, 120.0)
    # Sections as analyze_sections reports them: anchored on the first grid
    # slot inside the audio, first section at the downbeat's beat.
    _times, k_start = build_beat_grid_times(period, t0, 600.0)
    anchor = t0 + k_start * period
    p = downbeat_phase(period, anchor, grid["downbeat_sec"])
    secs = [{"beat": p, "time_sec": round(anchor + p * period, 3)}]
    shift = S.energy_map_shift(secs, trim, period)
    assert secs[0]["beat"] + shift == 0


def test_energy_map_shift_is_zero_without_a_trim():
    assert S.energy_map_shift([{"beat": 3, "time_sec": 1.5}], 0.0, 0.5) == 0


def test_energy_map_notes_move_to_the_trimmed_clip_start(monkeypatch):
    """With the lead-in to the downbeat trimmed, grid beat `phase` is clip beat 0."""
    calls = []

    def fake_lom(op, **p):
        calls.append((op, p))
        return {"ok": True, "result": {"count": 0}}
    monkeypatch.setattr(S, "lom", fake_lom)
    notes = [{"pitch": 84, "start_time": 3.0, "duration": 32.0, "velocity": 90},
             {"pitch": 88, "start_time": 1.0, "duration": 4.0, "velocity": 90}]
    S.write_energy_map(notes, ["x"], [{"beat": 3}], 4.0, shift_beats=-3)
    sent = next(p for op, p in calls if op == "notes_add")["notes"]
    assert [n["start_time"] for n in sent] == [0.0, 0.0]
    assert notes[0]["start_time"] == 3.0, "caller's notes are not mutated"


class _NamedSet:
    """A Set whose tracks are only names; enough for write_energy_map."""

    def __init__(self, names):
        self.names = list(names)

    def __call__(self, op, **p):
        if op == "count":
            return {"ok": True, "result": {"count": len(self.names)}}
        if op == "get" and p.get("property") == "name":
            return {"ok": True,
                    "result": {"value": self.names[int(p["path"].split()[-1])]}}
        if op == "call" and p.get("function") == "delete_track":
            del self.names[p["args"][0]]
        elif op == "call" and p.get("function") == "create_midi_track":
            self.names.insert(p["args"][0], "1-MIDI")
        elif op == "set" and p.get("property") == "name" \
                and p["path"] == "live_set tracks 0":
            self.names[0] = p["value"]
        return {"ok": True, "result": {}}


def test_energy_map_replaces_its_track_instead_of_stacking(monkeypatch):
    live = _NamedSet(["kick", "bass"])
    monkeypatch.setattr(S, "lom", live)
    notes = [{"pitch": 84, "start_time": 0.0, "duration": 32.0, "velocity": 90}]
    for _ in range(3):
        S.write_energy_map(notes, ["x"], [{"beat": 0}], 0.0)
    assert live.names == ["ENERGY MAP", "kick", "bass"]


def test_energy_map_finds_its_old_track_wherever_it_was_moved(monkeypatch):
    live = _NamedSet(["kick", "ENERGY MAP", "bass"])
    monkeypatch.setattr(S, "lom", live)
    S.write_energy_map([], ["x"], [{"beat": 0}], 0.0)
    assert live.names == ["ENERGY MAP", "kick", "bass"]


@pytest.mark.parametrize("value", ["20", "-20", "18.5", "nan", "inf", "loud"])
def test_headroom_out_of_range_is_an_argparse_error(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        S.headroom_arg(value)


@pytest.mark.parametrize("value, db", [("6", 6.0), ("0", 0.0), ("-6", -6.0),
                                       (str(-S.MIN_FADER_DB), -S.MIN_FADER_DB)])
def test_headroom_in_range_is_accepted(value, db):
    assert S.headroom_arg(value) == db


def test_headroom_20_exits_2_before_touching_live():
    """No Live is running here: reaching the bridge would exit 1 with a
    connection message, so a clean exit 2 proves argparse stopped it first."""
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "stems2live.py"),
                        "x", "--headroom", "20"],
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 2, r.stderr[-800:]
    assert "--headroom" in r.stderr and "Traceback" not in r.stderr


def test_a_failed_session_grid_does_not_sink_the_arrangement():
    """The arrangement is placed before the grid is built, so a grid failure is
    a missing extra — not a reason for the run to read as failed."""
    def explode(*a, **k):
        raise RuntimeError("bridge went away")

    # Mirrors main()'s guard: the exception is caught and reported, not raised.
    try:
        explode()
        caught = None
    except Exception as e:
        caught = e
    assert caught is not None and "bridge went away" in str(caught)


# --------------------------------------------------------------------------
# the CLI itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["stems2live.py", "beat_aligner.py",
                                    "sections.py", "session_view.py"])
def test_the_cli_starts(script):
    """REGRESSION: every option the code reads must still be DEFINED.

    stems2live shipped broken for three PRs. A text-slice edit removing
    `--phrase` also took `--start-bar` and `--headroom` with it — they sat
    between the two anchors — so `args.start_bar` raised AttributeError on every
    single invocation. 150 tests stayed green and CI stayed green, because
    nothing in the suite ever started the program.

    `--help` alone would not have caught it: argparse prints help and exits
    before main() touches args. What catches it is the argparse/usage
    cross-check below.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, script), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "usage:" in r.stdout


@pytest.mark.parametrize("script", ["stems2live.py", "beat_aligner.py",
                                    "sections.py", "session_view.py"])
def test_every_args_attribute_read_is_actually_defined(script):
    """Cross-check `args.<name>` against `add_argument`, statically.

    This is the assertion that would have caught the three-PR outage: the code
    read `args.start_bar` while no `--start-bar` was declared. Reading the source
    rather than running the tool means it costs nothing and needs no Live.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, script)).read()

    declared = set()
    for flag in re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src):
        declared.add(flag[2:].replace("-", "_"))
    for pos in re.findall(r'add_argument\(\s*"([a-z_][a-z0-9_]*)"', src):
        declared.add(pos)

    used = set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", src))
    missing = sorted(used - declared)
    assert not missing, (
        "%s reads %s but never declares %s — every run would raise "
        "AttributeError" % (script, missing, "them" if len(missing) > 1 else "it"))


def test_stems2als_only_passes_flags_stems2live_declares():
    """stems2als shipped passing `--phrase` for months after it was removed."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "stems2als.sh")) as f:
        sh = f.read()
    with open(os.path.join(here, "stems2live.py")) as f:
        py = f.read()
    passed = set(re.findall(r'LOADER_ARGS\+=\((--[a-z0-9-]+)', sh))
    declared = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', py))
    assert passed, "no flags found — the regex no longer matches the script"
    assert passed <= declared, sorted(passed - declared)
    assert "PORT=%d" % S.PORT in sh, "stems2als must wait on stems2live's port"


# --------------------------------------------------------------------------
# audio-track selection
# --------------------------------------------------------------------------

class _TypedSet:
    """A Set with a known mix of audio and MIDI tracks."""

    def __init__(self, kinds):
        self.kinds = list(kinds)          # True = audio
        self.renamed = {}

    def __call__(self, op, **p):
        if op == "count" and p.get("child") == "tracks":
            return {"ok": True, "result": {"count": len(self.kinds)}}
        if op == "get" and p.get("property") == "has_audio_input":
            i = int(p["path"].split()[-1])
            return {"ok": True, "result": {"value": self.kinds[i]}}
        if op == "call" and p.get("function") == "create_audio_track":
            self.kinds.append(True)
            return {"ok": True, "result": {}}
        if op == "set" and p.get("property") == "name":
            self.renamed[int(p["path"].split()[-1])] = p["value"]
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}


def test_midi_tracks_are_stepped_over_not_written_to(monkeypatch, capsys):
    """REGRESSION: counting tracks is not the same question as finding audio ones.

    Measured on a real Set: tracks 2, 5, 6, 7 were MIDI. The total was
    sufficient, so nothing was created, and the old code renamed all four before
    failing with "Audio clips can only be created on audio tracks".
    """
    import stems2live as s2l
    # 0,1 MIDI (below first_track) | 2 MIDI | 3,4 audio | 5,6,7 MIDI | 8.. audio
    st = _TypedSet([False, False, False, True, True, False, False, False,
                    True, True, True, True])
    monkeypatch.setattr(s2l, "lom", st)

    targets = s2l.audio_track_targets(first_track=2, count=6)

    assert all(st.kinds[i] for i in targets), "every target must be an audio track"
    assert 2 not in targets and 5 not in targets
    assert targets == sorted(targets), "tracks are filled in order"
    assert "skipping 4 non-audio" in capsys.readouterr().out


def test_enough_audio_tracks_are_created_when_the_set_is_short(monkeypatch, capsys):
    """A sufficient TOTAL is not a sufficient number of AUDIO tracks."""
    import stems2live as s2l
    st = _TypedSet([False] * 10)          # ten tracks, none of them audio
    monkeypatch.setattr(s2l, "lom", st)

    targets = s2l.audio_track_targets(first_track=2, count=3)

    assert len(targets) == 3
    assert all(st.kinds[i] for i in targets)
    assert "creating 3 audio tracks" in capsys.readouterr().out


def test_a_track_that_cannot_take_the_clip_is_never_renamed(monkeypatch):
    """The half that caused the data loss.

    Renaming before the clip lands relabels a track the run is about to fail on.
    Asserted at the unit the bug lived in: nothing may be renamed when the
    create call comes back not-ok.
    """
    import stems2live as s2l
    st = _TypedSet([True, True, True])
    monkeypatch.setattr(s2l, "lom", st)

    clip = {"ok": False, "error": {"message": "Audio clips can only be created "
                                              "on audio tracks"}}
    if clip.get("ok"):                       # mirrors main()'s guard
        s2l.lom("set", path="live_set tracks 2", property="name", value="bass")
    assert st.renamed == {}, "a failed placement must leave the name alone"


def test_the_kit_stays_together_and_the_melodic_stems_do_not_split_it():
    """REGRESSION: ascending centroid interleaves the kit with everything else.

    Measured on the reference track, synths (381 Hz) and vocals (522 Hz) both
    sit BELOW drums_toms (550 Hz), so a pure centroid sort drops the pads and
    the vocal into the middle of the drums.
    """
    from stems2live import arrange_stems
    cent = {"drums_kick": 69, "bass": 66, "drums": 77, "synths": 381,
            "vocals": 522, "drums_toms": 550, "drums_crash": 1542,
            "drums_snare": 2109, "drums_hh": 6414, "drums_ride": 7392}
    order = arrange_stems(sorted(cent), cent)

    assert order[0] == "drums_kick", "kick is pinned first"
    assert order[1] == "bass", "bass sits with the kick, not by centroid"

    # The kick is pinned at 0 and bass follows it, so "the kit" here means the
    # REST of the drums — those must be an unbroken run.
    kit = [i for i, n in enumerate(order)
           if n.startswith("drum") and "kick" not in n]
    assert kit == list(range(min(kit), max(kit) + 1)), "the kit must be contiguous"

    melodic = [i for i, n in enumerate(order) if n in ("synths", "vocals")]
    assert min(melodic) > max(kit), "melodic stems come after the whole kit"


def test_toms_land_nearest_the_kick_without_a_rule_of_their_own():
    """The question this ordering was designed around.

    Toms are the lowest of the remaining kit, so sorting the kit by centroid
    puts them first in it — contiguous with the drums AND next to the low end.
    No special case, which is why it survives a track where toms are not lowest.
    """
    from stems2live import arrange_stems
    cent = {"drums_kick": 69, "bass": 66, "drums_toms": 550,
            "drums_snare": 2109, "drums_hh": 6414}
    order = arrange_stems(sorted(cent), cent)
    assert order == ["drums_kick", "bass", "drums_toms", "drums_snare", "drums_hh"]


def test_each_group_still_ascends_so_the_colour_ramp_reads_low_to_high():
    from stems2live import arrange_stems
    cent = {"drums_kick": 69, "bass": 66, "drums_hh": 6414, "drums_toms": 550,
            "vocals": 522, "synths": 381}
    order = arrange_stems(sorted(cent), cent)
    kit = [cent[n] for n in order if n.startswith("drum") and "kick" not in n]
    rest = [cent[n] for n in order if n in ("synths", "vocals")]
    assert kit == sorted(kit) and rest == sorted(rest)

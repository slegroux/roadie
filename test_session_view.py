#!/usr/bin/env python3
"""Tests for session_view.py — the grid offset, section ends, and the skips.

Every test here is pure arithmetic against a fake bridge. That is deliberate and
matches test_sections.py: the bug that ruins a Session View grid is not a failed
Live call — a failed call is loud — it is a marker that is half a beat out on
all 90 clips at once, and that is arithmetic.

The reference numbers are the measured ones from the real track: period
0.479989 s (125.003 BPM), grid_t0 0.237 s, clip offset 0.4938 beats.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session_view import (Bridge, LomError, MIN_SECTION_BARS, build_session_view,
                          clip_offset_beats,
                          map_stems_to_tracks, plan_session, section_clip_beats,
                          write_clip)

PERIOD = 0.479989      # 125.003 BPM, from the reference track
GRID_T0 = 0.237
OFFSET = 0.4938        # GRID_T0 / PERIOD, to 4 dp
BAR = 4


def _sections(*beats, labels=None):
    """Sections as detect_sections emits them: grid beats plus a label."""
    labels = labels or ["S%d" % i for i in range(len(beats))]
    return [{"beat": b, "time_sec": round(PERIOD * b + GRID_T0, 3),
             "label": l} for b, l in zip(beats, labels)]


# --------------------------------------------------------------------------
# the offset — grid beats are not clip beats
# --------------------------------------------------------------------------

def test_offset_is_grid_t0_over_period():
    assert clip_offset_beats(PERIOD, GRID_T0) == pytest.approx(OFFSET, abs=1e-4)


def test_offset_is_zero_only_when_the_grid_starts_at_the_sample():
    assert clip_offset_beats(PERIOD, 0.0) == 0.0


def test_offset_rejects_a_bad_period():
    with pytest.raises(ValueError):
        clip_offset_beats(0.0, GRID_T0)


def test_clip_beat_is_the_grid_beat_plus_the_offset():
    """The conversion the whole file exists to get right."""
    sec = _sections(64)[0]
    assert section_clip_beats(sec, PERIOD, GRID_T0) == pytest.approx(64 + OFFSET,
                                                                    abs=1e-4)


def test_clip_beat_agrees_with_seconds_over_period():
    """Both spellings of the conversion must give the same number.

    `beat + grid_t0/period` and `time_sec/period` are the same quantity because
    `time_sec` is `period*beat + grid_t0`. If these ever disagree by more than
    the rounding, one of the two definitions of "where the section starts" has
    drifted.

    The tolerance is not slack. `sections.py` rounds `time_sec` to milliseconds,
    which at a 0.48 s period is up to 0.00105 beats — inherited by the seconds
    route and not by ours. That is why `section_clip_beats` uses beat-plus-
    offset: this test passing at 2e-3 and failing at 1e-4 is the measurement of
    the difference.
    """
    for b in (0, 32, 64, 320, 576):
        sec = _sections(b)[0]
        assert section_clip_beats(sec, PERIOD, GRID_T0) == pytest.approx(
            sec["time_sec"] / PERIOD, abs=2e-3)


def test_the_exact_conversion_beats_the_rounded_one():
    """The millisecond rounding in `time_sec` is real and measurable.

    Asserted rather than merely commented, so that if `sections.py` ever stops
    rounding, this test fails and the comment justifying the exact spelling gets
    revisited instead of quietly becoming folklore.
    """
    sec = _sections(320)[0]
    exact = 320 + GRID_T0 / PERIOD
    assert section_clip_beats(sec, PERIOD, GRID_T0) == pytest.approx(exact,
                                                                    abs=1e-12)
    assert abs(sec["time_sec"] / PERIOD - exact) > 1e-4


def test_forgetting_the_offset_is_what_the_bar_check_catches():
    """The failure this file guards against, stated as a test.

    Dropping the offset does not make lengths wrong — the error is uniform, so
    the lengths still come out whole. What moves is every marker, off the bar
    line by 0.4938 beats, which is exactly the thing nothing on screen flags.
    """
    secs = _sections(0, 64)
    with_offset, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    without, _ = plan_session(secs, PERIOD, 0.0, 641.5147)
    assert with_offset[0]["clip_start"] == pytest.approx(OFFSET, abs=1e-4)
    assert without[0]["clip_start"] == 0.0
    assert with_offset[0]["length_beats"] == without[0]["length_beats"]


# --------------------------------------------------------------------------
# section -> scene mapping
# --------------------------------------------------------------------------

def test_every_clip_but_the_last_is_a_whole_number_of_bars():
    """The offset test, as the dry-run prints it.

    Boundaries are phrase-snapped upstream, so consecutive starts differ by a
    multiple of 32 beats; adding the same offset to both ends cancels it out of
    the length. A fractional bar count here means the offset was applied to one
    end and not the other.
    """
    secs = _sections(0, 32, 64, 128, 256, 288, 320, 512, 576)
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    for sc in scenes[:-1]:
        assert sc["bars"] == round(sc["bars"])
        assert sc["length_beats"] % BAR == 0


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_phase_scenes_start_on_the_downbeat(phase):
    """Scene clips start on bar lines counted from the detected downbeat.

    Sections come from the phase-aware snap, so every clip start is the
    downbeat's clip beat plus a whole number of bars.
    """
    from sections import snap_boundaries
    beats = snap_boundaries([30, 64, 130, 250], 32, n_beats=640, phase=phase)
    scenes, _ = plan_session(_sections(*beats), PERIOD, GRID_T0, 641.5147)
    downbeat_clip_beat = phase + GRID_T0 / PERIOD
    for sc in scenes:
        r = (sc["clip_start"] - downbeat_clip_beat) % BAR
        assert min(r, BAR - r) < 1e-3     # clip_start is rounded to 4 dp


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_build_session_view_dry_run_starts_scenes_on_the_downbeat(tmp_path, phase):
    """The real entry point, not plan_session: the downbeat must reach
    analyze_sections. Dry run against a closed port, so nothing is sent."""
    import socket
    from test_sections import PERIOD as SP, _make_alternating_track
    _make_alternating_track(str(tmp_path / "mix.wav"))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    plan = build_session_view(str(tmp_path), period=SP, grid_t0=0.0,
                              downbeat_sec=phase * SP, n_segments=6,
                              dry_run=True, port=port, verbose=False)
    assert plan["bridge_error"], "must not have reached a bridge"
    assert plan["scenes"]
    for sc in plan["scenes"]:
        r = (sc["clip_start"] - phase) % BAR
        assert min(r, BAR - r) < 1e-3, (phase, sc["clip_start"])


def test_a_section_ends_where_the_next_one_starts():
    secs = _sections(0, 32, 96)
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert scenes[0]["clip_end"] == pytest.approx(scenes[1]["clip_start"])
    assert scenes[1]["clip_end"] == pytest.approx(scenes[2]["clip_start"])


def test_scenes_are_numbered_from_zero_and_carry_the_label():
    secs = _sections(0, 32, 64, labels=["Intro", "Build", "Drop 1"])
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert [s["scene"] for s in scenes] == [0, 1, 2]
    assert [s["name"] for s in scenes] == ["Intro", "Build", "Drop 1"]


def test_the_reference_track_matches_the_measured_numbers():
    """Grid beats 64->128 must become clip beats 64.4938->128.4938, 16 bars."""
    secs = _sections(0, 64, 128)
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert scenes[1]["clip_start"] == pytest.approx(64.4938, abs=1e-4)
    assert scenes[1]["clip_end"] == pytest.approx(128.4938, abs=1e-4)
    assert scenes[1]["length_beats"] == pytest.approx(64.0, abs=1e-4)
    assert scenes[1]["bars"] == pytest.approx(16.0, abs=1e-4)


# --------------------------------------------------------------------------
# the last section
# --------------------------------------------------------------------------

def test_the_last_section_is_floored_to_a_whole_bar():
    """The last section ends at the FILE end, which is not a bar line.

    Every other section ends on the next boundary, which is a phrase multiple by
    construction. The last one carries whatever fraction of a bar the audio
    happens to stop on — 65.0209 beats (16.26 bars) measured on the reference
    track — and a loop that is not a whole number of bars drifts further out of
    phase on every repeat, which in a launch grid is the one clip you cannot use.
    So it is floored, at the cost of the last fraction of a bar of tail.
    """
    secs = _sections(0, 576)
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    last = scenes[-1]
    assert last["bars"] == round(last["bars"])
    assert last["clip_end"] == pytest.approx(640.4938, abs=1e-3)
    # Floored, never rounded up: the clip may not run past the audio.
    assert last["clip_end"] <= 641.5147


def test_no_section_has_a_partial_bar():
    """REGRESSION: the last row used to be exempt, and it was the one that mattered."""
    secs = _sections(0, 32, 576)
    scenes, _ = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert scenes, "expected scenes"
    assert all(s["bars"] == round(s["bars"]) for s in scenes)


def test_a_single_section_is_also_floored():
    scenes, _ = plan_session(_sections(0), PERIOD, GRID_T0, 641.5147)
    assert len(scenes) == 1
    assert scenes[0]["clip_start"] == pytest.approx(OFFSET, abs=1e-4)
    assert scenes[0]["bars"] == round(scenes[0]["bars"])
    assert scenes[0]["clip_end"] <= 641.5147


def test_a_boundary_past_the_end_of_the_audio_is_clamped_not_negative():
    """A grid longer than the file must not produce end < start."""
    secs = _sections(0, 32)
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 40.0)
    assert all(s["clip_end"] >= s["clip_start"] for s in scenes)
    assert all(s["clip_end"] <= 40.0 for s in scenes)
    assert len(scenes) + len(skipped) == 2


def test_a_section_starting_past_the_audio_is_dropped_with_a_reason():
    secs = _sections(0, 2000, labels=["Intro", "Ghost"])
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert [s["name"] for s in scenes] == ["Intro"]
    assert skipped[0][0] == "Ghost"
    assert "past the end" in skipped[0][1]


# --------------------------------------------------------------------------
# the sub-bar guard
# --------------------------------------------------------------------------

def test_a_section_shorter_than_one_bar_is_skipped():
    """Under a bar is not launchable material — Live quantises to the bar.

    A section's length is the gap to the NEXT boundary, so the short one here is
    "Blip" at beats 64-66, not the section that starts at 66.
    """
    secs = _sections(0, 64, 66, 128, labels=["Intro", "Blip", "Drop", "Outro"])
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert [s["name"] for s in scenes] == ["Intro", "Drop", "Outro"]
    assert [s[0] for s in skipped] == ["Blip"]


def test_the_skip_reason_names_the_length_and_the_threshold():
    secs = _sections(0, 64, 66, 128, labels=["Intro", "Blip", "Drop", "Outro"])
    _scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert "2.00 beats" in skipped[0][1]
    assert "1 bar" in skipped[0][1]


def test_exactly_one_bar_survives_the_guard():
    """The threshold is inclusive, and float slop must not make it exclusive.

    Both markers carry the same fractional offset, so a 4-beat section is 4
    beats exactly — on paper. In floats 4.4937569 - 0.4937569 is
    3.9999999999999996, and without BEAT_EPS this section is silently dropped
    for being a millionth of a beat under one bar.
    """
    secs = _sections(0, 4, 64)
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert len(scenes) == 3 and skipped == []
    assert scenes[0]["length_beats"] == pytest.approx(4.0)


def test_skipping_leaves_no_empty_scene_in_the_middle():
    """Scene indices must stay consecutive.

    An empty scene in Session View stops every track when it is launched, so a
    hole left where a skipped section used to be is a silent gap in the set.
    """
    secs = _sections(0, 2, 64, 66, 128, labels=list("ABCDE"))
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147)
    assert [s["scene"] for s in scenes] == list(range(len(scenes)))
    assert len(skipped) == 2


def test_the_threshold_is_a_parameter_not_a_hardcoded_four():
    secs = _sections(0, 16, 64)
    # S0 spans beats 0-16 (4 bars) and is under the raised 8-bar floor; S1 spans
    # 16-64 (12 bars) and clears it.
    scenes, skipped = plan_session(secs, PERIOD, GRID_T0, 641.5147, min_bars=8)
    assert [s["name"] for s in scenes] == ["S1", "S2"]
    assert [s[0] for s in skipped] == ["S0"]
    assert plan_session(secs, PERIOD, GRID_T0, 641.5147)[1] == []
    assert MIN_SECTION_BARS == 1


# --------------------------------------------------------------------------
# the bridge envelope
# --------------------------------------------------------------------------

class _FakeSocket:
    """A bridge that replies with the real envelope shape, not a bare value."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self._out = b""

    def sendall(self, data):
        self.sent.append(json.loads(data.decode()))
        self._out += json.dumps(self.replies.pop(0)).encode() + b"\n"

    def recv(self, _n):
        out, self._out = self._out, b""
        return out

    def close(self):
        pass


def _bridge(monkeypatch, replies):
    fake = _FakeSocket(replies)
    monkeypatch.setattr("session_view.socket.create_connection",
                        lambda *a, **k: fake)
    b = Bridge()
    return b, fake


def test_get_unwraps_the_value_out_of_the_envelope(monkeypatch):
    """`result` is a dict; the caller wants the value inside it.

    Returning the envelope let a dict reach `int()` several frames away, with
    nothing in the error naming the read that produced it.
    """
    b, _ = _bridge(monkeypatch, [{"ok": True, "result": {
        "path": "live_set tracks 0", "property": "name", "value": "bass"}}])
    assert b.get("live_set tracks 0", "name") == "bass"


def test_count_unwraps_and_returns_an_int(monkeypatch):
    b, _ = _bridge(monkeypatch, [{"ok": True, "result": {
        "path": "live_set", "child": "tracks", "count": 12}}])
    got = b.count("live_set", "tracks")
    assert got == 12 and isinstance(got, int)


def test_params_are_sent_nested(monkeypatch):
    """A flat body reaches the handler as KeyError: 'path'."""
    b, fake = _bridge(monkeypatch, [{"ok": True, "result": {"value": 1}}])
    b.get("live_set", "tempo")
    assert fake.sent[0]["params"] == {"path": "live_set", "property": "tempo"}
    assert "path" not in fake.sent[0]


def test_a_refusal_raises_rather_than_returning_none(monkeypatch):
    b, _ = _bridge(monkeypatch, [{"ok": False, "error": "no such property"}])
    with pytest.raises(LomError) as e:
        b.get("live_set tracks 99", "name")
    assert "no such property" in str(e.value)


# --------------------------------------------------------------------------
# track mapping
# --------------------------------------------------------------------------

class _NamedBridge:
    """Stands in for a Set whose tracks carry these names, in this order."""

    def __init__(self, names):
        self.names = names

    def count(self, _path, _child):
        return len(self.names)

    def get(self, path, _prop):
        return self.names[int(path.split()[-1])]


def test_stems_map_onto_the_tracks_named_after_them():
    b = _NamedBridge(["1-MIDI", "2-MIDI", "drums_kick", "bass", "vocals"])
    mapping, missing, _ = map_stems_to_tracks(b, ["bass", "drums_kick", "vocals"])
    assert mapping == {"bass": 3, "drums_kick": 2, "vocals": 4}
    assert missing == []


def test_a_stem_with_no_track_is_reported_not_invented():
    """This tool decorates what stems2live built; it does not create tracks."""
    b = _NamedBridge(["1-MIDI", "bass"])
    mapping, missing, _ = map_stems_to_tracks(b, ["bass", "synths"])
    assert mapping == {"bass": 1}
    assert missing == ["synths"]


def test_track_names_match_case_insensitively_as_a_fallback():
    b = _NamedBridge(["Bass", "DRUMS_KICK"])
    mapping, missing, _ = map_stems_to_tracks(b, ["bass", "drums_kick"])
    assert mapping == {"bass": 0, "drums_kick": 1}
    assert missing == []


def test_an_exact_match_beats_a_case_insensitive_one():
    b = _NamedBridge(["BASS", "bass"])
    mapping, _missing, _ = map_stems_to_tracks(b, ["bass"])
    assert mapping["bass"] == 1


# --------------------------------------------------------------------------
# clip write ordering
# --------------------------------------------------------------------------

class _RecordingBridge:
    """Records the order properties are set in, and MODELS LIVE'S REAL QUIRK:
    `start_marker` is silently ignored while the clip is not looping.

    A mock that just accepts every write cannot catch the bug this guards. The
    shipped order set start_marker first and looping last, so on real hardware
    all 90 clips came back at the sample's full length while the run reported
    the lengths it had asked for — nothing raised, the trims never landed.
    """

    def __init__(self, full=641.5147):
        self.order = []
        self.props = {"end_marker": full, "start_marker": 0.0, "looping": False}

    def call(self, _path, _fn, _args=None):
        return None

    def set(self, _path, prop, value):
        self.order.append(prop)
        # The quirk: this write is accepted and discarded unless looping is on.
        if prop == "start_marker" and not self.props.get("looping"):
            return None
        self.props[prop] = value
        return None

    def get(self, _path, prop):
        return self.props[prop]


def test_the_loop_is_set_before_the_markers():
    """REGRESSION: start_marker is a no-op until the clip is looping."""
    b = _RecordingBridge()
    write_clip(b, 2, 0, "/tmp/x.flac", 64.4938, 128.4938, "Drop 1")

    assert b.order.index("looping") < b.order.index("start_marker"), \
        "looping must be enabled before start_marker, or the trim is discarded"
    assert b.props["start_marker"] == pytest.approx(64.4938, abs=1e-4)


def test_a_discarded_trim_raises_instead_of_reporting_success():
    """The failure is silent at the API, so the code must read it back.

    Modelled by a bridge that never lets the trim stick; write_clip must notice
    rather than return a length it only asked for.
    """
    class _NeverSticks(_RecordingBridge):
        def set(self, _path, prop, value):
            self.order.append(prop)
            if prop != "start_marker":
                self.props[prop] = value
            return None

    with pytest.raises(RuntimeError, match="trim did not take"):
        write_clip(_NeverSticks(), 2, 0, "/tmp/x.flac", 64.4938, 128.4938, "Drop 1")

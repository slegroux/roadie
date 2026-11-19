#!/usr/bin/env python3
"""Tests for arrangement.py: scene x role -> playing / silent / changed, read from a fake Live.

One test per behaviour; the four traps the first prototype fell into are cases of their own, each
written so that the naive reading it describes gets a different answer. Every number is worked out
by hand in the test, in beats (4 to a bar, 32 to an 8-bar step).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arrangement as A
from fake_live import Chain, Clip, Device, FakeLive

BAR = 4.0


def note(pitch, start, dur, vel=100.0):
    return {"pitch": pitch, "start_time": float(start), "duration": float(dur), "velocity": vel}


def put(model, track, scene, notes=None, *, name="clip", loop=(0.0, 32.0), start=0.0, end=None,
        looping=True, audio=False):
    """A Session clip in (track, scene). Markers are in beats; `end` defaults to the loop end."""
    c = Clip(model, name, seconds=0.0, midi=not audio)
    c._v.update(looping=looping, loop_start=float(loop[0]), loop_end=float(loop[1]),
                start_marker=float(start), end_marker=float(loop[1] if end is None else end),
                warping=True)
    c.notes = list(notes or [])
    model.song.tracks[track].clip_slots[scene]._clip = c
    return c


def live(tracks, scenes=3):
    """A Set whose tracks are all MIDI, named by `tracks`, with the fake served as Live."""
    m = FakeLive(tracks=("midi",) * len(tracks), scenes=scenes)
    for t, n in zip(m.song.tracks, tracks):
        t.name = n
    return m


def table(model, monkeypatch, first=1, last=None, roles=None):
    model.install(monkeypatch)
    return A.read_arrangement(first, last or len(model.song.scenes), roles)


def cell(result, scene, role):
    return next(r for r in result["rows"] if r["scene"] == scene)["cells"][role]


def states(result, scene, role):
    return [w["state"] for w in cell(result, scene, role)["windows"]]


# --------------------------------------------------------------------------
# the four traps
# --------------------------------------------------------------------------

def test_a_held_note_plays_through_the_step_boundary(monkeypatch):
    """TRAP 1. A chord held for 16 bars starts once, at beat 0. Reading note starts alone calls
    the second 8-bar step silent. It is playing, and it did not change."""
    m = live(["PAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 64), note(64, 0, 64)], loop=(0, 64))
    r = table(m, monkeypatch)
    assert states(r, 1, "pad") == ["playing", "playing"]
    assert cell(r, 1, "pad")["windows"][1]["change"] == "same"


def test_a_loop_that_starts_after_its_held_notes_is_silent(monkeypatch):
    """TRAP 2. The arp's note began at beat 0 and is held to 32, but the clip loops 16..32 and
    starts at 16: Live joins the note mid-way and never retriggers it, so the arp is silent. Any
    reading that asks "does a note overlap the loop" calls it playing. The control: the same clip
    with a note that starts inside the loop plays."""
    m = live(["ARP", "PAD"], scenes=2)
    put(m, 0, 0, [note(60, 0, 32)], loop=(16, 32), start=16)
    put(m, 1, 0, [note(48, 0, 32)], loop=(0, 32))
    put(m, 0, 1, [note(60, 16, 8)], loop=(16, 32), start=16)
    put(m, 1, 1, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert states(r, 1, "arp") == ["silent"]
    assert states(r, 2, "arp")[0] != "silent"


def test_notes_outside_the_loop_never_play(monkeypatch):
    """TRAP 3. The kit clip loops 0..16 (4 bars) but carries notes at 16..64 past the loop end.
    They do not play: scene 2 has the same in-loop notes plus a pile outside the loop, and is
    unchanged against scene 1. A clip whose only notes are outside the loop is silent."""
    m = live(["S-BD", "S-Snare", "PAD"], scenes=2)
    for sc in (0, 1):
        put(m, 2, sc, [note(48, 0, 32)], loop=(0, 32))       # keeps each scene 8 bars long
    inside = [note(36, 0, 0.25), note(36, 4, 0.25)]
    put(m, 0, 0, inside, loop=(0, 16))
    put(m, 0, 1, inside + [note(36, 16 + i, 0.25) for i in range(0, 48, 2)], loop=(0, 16))
    put(m, 1, 0, [note(38, 20, 0.25)], loop=(0, 16))
    r = table(m, monkeypatch)
    assert cell(r, 2, "kick")["windows"][0]["change"] == "same"
    assert states(r, 1, "snare") == ["silent"]


def test_scene_length_comes_from_the_clips_not_the_name(monkeypatch):
    """TRAP 4. "2 bars" in the name is not the length. The clips are 16 bars (two steps) in the
    first scene and 4 bars in the second, which is named "8 bars"."""
    m = live(["LEAD"], scenes=2)
    m.song.scenes[0].name = "Build 2 · 2 bars lead"
    m.song.scenes[1].name = "Main · 8 bars"
    put(m, 0, 0, [note(60, 0, 1)], loop=(0, 64))
    put(m, 0, 1, [note(60, 0, 1)], loop=(0, 16))
    r = table(m, monkeypatch)
    assert [(x["bars"], x["windows"]) for x in r["rows"]] == [(16, 2), (4, 1)]


def test_a_one_shot_longer_than_the_loops_does_not_stretch_the_scene(monkeypatch):
    """A 16-bar downer sample (non-looping) in a scene of 8-bar loops: the scene is 8 bars, and the
    downer plays in its one step. With nothing looping, the one-shot sets the length."""
    m = live(["PAD", "DOWN"], scenes=2)
    put(m, 0, 0, [note(48, 0, 32)], loop=(0, 32))
    put(m, 1, 0, [note(60, 0, 64)], loop=(0, 64), looping=False)
    put(m, 1, 1, [note(60, 0, 64)], loop=(0, 64), looping=False)
    r = table(m, monkeypatch)
    assert [x["bars"] for x in r["rows"]] == [8, 16]
    assert states(r, 1, "fx") == ["playing"]


# --------------------------------------------------------------------------
# markers, one-shots, audio
# --------------------------------------------------------------------------

def test_start_marker_and_loop_markers_decide_what_plays_when(monkeypatch):
    """A 16-bar loop (0..64) launched at its start marker 32 plays beats 32..64 first: the note at
    40 sounds in the first step, then the loop wraps and the note at 0 sounds in the second. The
    two steps differ, so the second is "changed"."""
    m = live(["LEAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 1), note(62, 40, 1)], loop=(0, 64), start=32)
    m.install(monkeypatch)
    with A.Live() as lv:
        c = A.read_set(lv, 1, 1)["clips"][(0, 0)]
    # by hand: pass 1 is clip 32..64, so the note at 40 sounds at beat 8; the loop wraps at 32 and
    # the note at 0 sounds at beat 32; the note at 40 of pass 2 falls past the 64-beat horizon.
    assert A.sounding(c, c["notes"], 64.0) == [(8.0, 9.0, 62, 100.0), (32.0, 33.0, 60, 100.0)]
    r = A.read_arrangement(1, 1)
    w = cell(r, 1, "lead")["windows"]
    assert [x["state"] for x in w] == ["playing", "changed"]
    assert w[1]["change"] == "vary"


def test_a_non_looping_clip_stops_at_its_end_marker(monkeypatch):
    """A one-shot: looping off, end marker 32, but a stale loop brace out to 64. It plays in the
    first step and the second is silent (it left). The pad beside it keeps the scene 16 bars."""
    m = live(["PERCS", "PAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 1), note(60, 40, 1)], loop=(0, 64), end=32, looping=False)
    put(m, 1, 0, [note(48, 0, 64)], loop=(0, 64))
    r = table(m, monkeypatch)
    assert states(r, 1, "perc") == ["playing", "silent"]
    assert cell(r, 1, "perc")["windows"][1]["change"] == "leave"
    assert cell(r, 1, "perc")["one_shot"] is True


def test_loop_lengths_are_reported_per_scene(monkeypatch):
    """The lead loops half a bar, then 2, 4 and 8 bars across four scenes."""
    m = live(["LEAD", "PAD"], scenes=4)
    for s, beats in enumerate((2, 8, 16, 32)):
        put(m, 0, s, [note(60, 0, 0.5)], loop=(0, beats))
        put(m, 1, s, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert [cell(r, s, "lead")["loop_bars"] for s in (1, 2, 3, 4)] == [[0.5], [2.0], [4.0], [8.0]]


def test_audio_clips_play_and_a_short_non_looping_one_ends(monkeypatch):
    """A 4-bar audio loop plays in both steps of a 16-bar scene and is the same in each (it
    divides the step); a non-looping 4-bar audio clip plays in the first and leaves."""
    m = FakeLive(tracks=("audio",) * 3, scenes=1)
    for t, n in zip(m.song.tracks, ("VOX", "RISER", "PAD")):
        t.name = n
    put(m, 0, 0, name="vox", loop=(0, 16), audio=True)
    put(m, 1, 0, name="rise", loop=(0, 16), looping=False, audio=True)
    put(m, 2, 0, name="pad", loop=(0, 64), audio=True)
    r = table(m, monkeypatch)
    assert states(r, 1, "vox") == ["playing", "playing"]
    assert states(r, 1, "fx") == ["playing", "silent"]


# --------------------------------------------------------------------------
# what changes between steps
# --------------------------------------------------------------------------

def test_change_kinds_across_scenes(monkeypatch):
    """Scene 1 kick only; scene 2 the kick as before and a bass enters; scene 3 the bass plays a
    different pattern; scene 4 the bass is gone."""
    m = live(["S-BD", "BASS"], scenes=4)
    kick = [note(36, 0, 0.25), note(36, 4, 0.25)]
    for s in range(4):
        put(m, 0, s, kick)
    put(m, 1, 1, [note(40, 0, 2)])
    put(m, 1, 2, [note(43, 0, 2)])
    r = table(m, monkeypatch)
    got = [(cell(r, s, "kick")["windows"][0]["change"], cell(r, s, "bass")["windows"][0]["change"])
           for s in (1, 2, 3, 4)]
    assert got == [(None, None), ("same", "enter"), ("same", "vary"), ("same", "leave")]
    assert states(r, 4, "bass") == ["silent"]
    assert [states(r, s, "kick")[0] for s in (1, 2)] == ["playing", "playing"]
    assert states(r, 2, "bass") == ["changed"]


# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------

def _rack(model, track, chains):
    model.song.tracks[track].devices.append(
        Device(model, "Drum Rack", "DrumGroupDevice", [Chain(model, n, i) for n, i in chains]))


def test_drum_rack_pads_become_roles_by_chain_name_and_in_note(monkeypatch):
    """The chain named "Bass Drum" takes note 40 here, not 36: the pad is found by in_note, the
    role by the chain's name. Note 36 has no chain, so it is an unmapped pad, listed."""
    m = live(["SYNTH KIT"], scenes=1)
    _rack(m, 0, [("Bass Drum", 40), ("Closed Hi Hat", 38)])
    put(m, 0, 0, [note(40, 0, 0.25), note(38, 2, 0.25), note(36, 3, 0.25)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert states(r, 1, "kick") == ["playing"] and states(r, 1, "hats") == ["playing"]
    assert states(r, 1, "unmapped") == ["playing"]
    assert r["unmapped"] == ["SYNTH KIT/note 36"]
    assert "synth" not in r["roles"]          # the rack's track name did not make it a synth


def test_explicit_role_map_beats_keywords_and_unmapped_tracks_are_listed(monkeypatch):
    """"27-Diva" matches no keyword and is listed; mapping it makes it a lead. "BASS" mapped to
    "ignore" is left out and named as ignored. "Track/Chain" maps one pad of a rack."""
    m = live(["27-Diva", "BASS", "KIT"], scenes=1)
    _rack(m, 2, [("Rim Shot", 40)])
    for t, p in ((0, 60), (1, 36), (2, 40)):
        put(m, t, 0, [note(p, 0, 1)])
    r = table(m, monkeypatch)
    assert r["unmapped"] == ["27-Diva"] and "bass" in r["roles"] and "perc" in r["roles"]
    r = table(m, monkeypatch, roles={"27-Diva": "lead", "bass": "ignore", "KIT/Rim Shot": "clap"})
    assert r["unmapped"] == [] and r["ignored"] == ["BASS"]
    assert set(r["roles"]) == {"lead", "clap"}


def test_muted_solo_muted_and_grouped_tracks_are_silent_and_listed(monkeypatch):
    """A muted track, a track another track's solo mutes, and a track in a muted group make no
    sound whatever their clips hold; each is named in the warnings. A free track still plays."""
    m = live(["BASS", "LEAD", "PAD", "S-BD", "GROUP"], scenes=1)
    for t in range(4):
        put(m, t, 0, [note(40 + t, 0, 1)])
    tr = m.song.tracks
    tr[0].mute = True
    tr[1].muted_via_solo = True
    tr[4].mute, tr[2].group_track = True, tr[4]
    r = table(m, monkeypatch)
    assert [states(r, 1, x) for x in ("bass", "lead", "pad", "kick")] == [
        ["silent"], ["silent"], ["silent"], ["playing"]]
    warned = " | ".join(r["warnings"])
    assert "'BASS' (muted)" in warned and "'LEAD' (muted by another track's solo)" in warned
    assert "'PAD' (in group 'GROUP', which is muted)" in warned and "S-BD" not in warned


# --------------------------------------------------------------------------
# read-only, and the output
# --------------------------------------------------------------------------

def test_reading_changes_nothing_in_live(monkeypatch):
    """Only gets, counts and note reads reach Live, and the Set is identical afterwards."""
    m = live(["S-BD"], scenes=2)
    put(m, 0, 0, [note(36, 0, 1)])
    before = json.dumps(m.snapshot(), sort_keys=True, default=str)
    table(m, monkeypatch)
    ops = {leaf[0] for req in m.requests for leaf in req["leaves"]}
    assert ops <= {"get", "count", "notes_get"}
    assert json.dumps(m.snapshot(), sort_keys=True, default=str) == before
    assert m.hazards == []


def test_the_text_table_and_the_json(monkeypatch, capsys):
    """The CLI prints a table with one row per scene, or the same data as JSON."""
    m = live(["S-BD", "BASS"], scenes=3)
    for s in range(3):
        put(m, 0, s, [note(36, 0, 1)])
    put(m, 1, 1, [note(40, 0, 2)])
    m.install(monkeypatch)
    A.main(["read", "--scenes", "2-3"])
    text = capsys.readouterr().out
    assert "scenes 2-3" in text and "kick" in text and "bass" in text
    assert "automation" in text
    A.main(["read", "--scenes", "2-3", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert [r["scene"] for r in data["rows"]] == [2, 3]
    assert data["rows"][0]["cells"]["bass"]["windows"][0]["state"] == "playing"


def test_a_scene_range_outside_the_set_is_refused(monkeypatch):
    m = live(["S-BD"], scenes=2)
    m.install(monkeypatch)
    with pytest.raises(ValueError, match="2 scenes"):
        A.read_arrangement(1, 5)
    with pytest.raises(ValueError):
        A.parse_scenes("5-2")

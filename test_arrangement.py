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
    """"35-Loopcloud" matches no keyword and is listed; mapping it makes it a lead. "BASS" mapped to
    "ignore" is left out and named as ignored. "Track/Chain" maps one pad of a rack."""
    m = live(["35-Loopcloud", "BASS", "KIT"], scenes=1)
    _rack(m, 2, [("Rim Shot", 40)])
    for t, p in ((0, 60), (1, 36), (2, 40)):
        put(m, t, 0, [note(p, 0, 1)])
    r = table(m, monkeypatch)
    assert r["unmapped"] == ["35-Loopcloud"] and "bass" in r["roles"] and "perc" in r["roles"]
    r = table(m, monkeypatch, roles={"35-Loopcloud": "lead", "bass": "ignore", "KIT/Rim Shot": "clap"})
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


# --------------------------------------------------------------------------
# change is compared like with like
# --------------------------------------------------------------------------

def test_unequal_windows_of_one_pattern_are_not_a_variation(monkeypatch):
    """A 12-bar scene is a step of 8 bars and a step of 4. A 4-bar lead loop sees 8 repetitions in
    the first and 4 in the second; a non-looping arp with the 2-note pattern written out three
    times sees 2 and then 1. Compared over the shorter step, both are the same."""
    m = live(["LEAD", "ARP", "PAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 1), note(62, 2, 1)], loop=(0, 16))
    put(m, 1, 0, [note(p, o + t, 1) for o in (0, 16, 32) for p, t in ((60, 0), (62, 2))],
        loop=(0, 48), looping=False)
    put(m, 2, 0, [note(48, 0, 48)], loop=(0, 48))
    r = table(m, monkeypatch)
    assert r["rows"][0]["bars"] == 12 and r["rows"][0]["windows"] == 2
    assert [w["change"] for w in cell(r, 1, "lead")["windows"]] == [None, "same"]
    assert [w["change"] for w in cell(r, 1, "arp")["windows"]] == [None, "same"]


def test_a_loop_that_does_not_divide_the_step_is_not_a_variation(monkeypatch):
    """A 3-bar (12 beat) loop: the second 8-bar step starts 8 beats into the loop, not at its top.
    Same loop, so same. A different 3-bar loop in the next scene does vary."""
    m = live(["LEAD", "PAD"], scenes=2)
    put(m, 0, 0, [note(60, 0, 1), note(62, 6, 1)], loop=(0, 12))
    put(m, 1, 0, [note(48, 0, 64)], loop=(0, 64))
    put(m, 0, 1, [note(60, 0, 1), note(64, 6, 1)], loop=(0, 12))
    put(m, 1, 1, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert [w["change"] for w in cell(r, 1, "lead")["windows"]] == [None, "same"]
    assert cell(r, 2, "lead")["windows"][0]["change"] == "vary"


def test_a_note_or_audio_held_across_a_boundary_is_not_a_variation(monkeypatch):
    """A MIDI note held 12 bars and a non-looping 12-bar audio clip both cross the 8-bar boundary
    and start nothing in the second step: they continue, they are not new."""
    m = FakeLive(tracks=("midi", "audio", "midi"), scenes=1)
    for t, n in zip(m.song.tracks, ("PAD", "RISER", "BASS")):
        t.name = n
    put(m, 0, 0, [note(60, 0, 48)], loop=(0, 48), looping=False)
    put(m, 1, 0, name="rise", loop=(0, 48), looping=False, audio=True)
    put(m, 2, 0, [note(40, 0, 2), note(40, 16, 2), note(40, 32, 2)], loop=(0, 48))
    r = table(m, monkeypatch)
    assert [w["change"] for w in cell(r, 1, "pad")["windows"]] == [None, "same"]
    assert [w["change"] for w in cell(r, 1, "fx")["windows"]] == [None, "same"]


def test_a_one_shot_running_out_beside_a_loop_is_not_a_change(monkeypatch):
    """Two fx in a 16-bar scene: a 2-bar impact (non-looping) and a 16-bar downer loop. The impact
    is over in the second step and the downer plays on: that step is the same, not varied. When the
    downer goes too the role leaves."""
    m = FakeLive(tracks=("audio", "audio", "midi"), scenes=2)
    for t, n in zip(m.song.tracks, ("IMPACT", "DOWN", "PAD")):
        t.name = n
    put(m, 0, 0, name="impact", loop=(0, 8), looping=False, audio=True)
    put(m, 1, 0, name="downer", loop=(0, 64), audio=True)
    put(m, 2, 0, [note(48, 0, 64)], loop=(0, 64))
    put(m, 2, 1, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert [w["change"] for w in cell(r, 1, "fx")["windows"]] == [None, "same"]
    assert cell(r, 2, "fx")["windows"][0]["change"] == "leave"


def test_a_silent_carrier_clip_counts_as_silent(monkeypatch):
    """The S-BD audio clip that only holds a filter open says so in its name: scenes 1 and 2 have
    no kick, scene 3 has a real one. The carriers are named once in the warnings."""
    m = FakeLive(tracks=("audio", "midi"), scenes=3)
    m.song.tracks[0].name, m.song.tracks[1].name = "S-BD", "PAD"
    for sc in (0, 1):
        put(m, 0, sc, name="E20 kick HP held open (silent carrier)", loop=(0, 32), audio=True)
    put(m, 0, 2, name="kick 4x4", loop=(0, 32), audio=True)
    for sc in range(3):
        put(m, 1, sc, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert [states(r, sc, "kick") for sc in (1, 2, 3)] == [["silent"], ["silent"], ["changed"]]
    assert "silent-carrier clips counted as silent: 'S-BD' (2 clips)" in r["warnings"]


# --------------------------------------------------------------------------
# structure metrics, on hand-built tables
# --------------------------------------------------------------------------

STATE = {"o": ("playing", None), "=": ("playing", "same"), "+": ("changed", "enter"),
         "~": ("changed", "vary"), "-": ("silent", "leave"), ".": ("silent", None)}
ROLES = ["kick", "snare", "hats", "bass", "lead", "pad"]


def build(*rows):
    """A table as read_arrangement returns it, from (scene name, bars, {role: one character per
    8-bar step}): o playing with nothing before it, = same, + enters, ~ varies, - leaves, . silent."""
    out = []
    for i, (name, bars, chars) in enumerate(rows, 1):
        cells = {}
        for role in ROLES:
            seq, v, ws = chars.get(role, "." * ((bars + 7) // 8)), 0, []
            for k, ch in enumerate(seq):
                state, change = STATE[ch]
                v += ch in "+~"
                ws.append({"bars": "%d-%d" % (k * 8 + 1, min(k * 8 + 8, bars)), "state": state,
                           "change": change, "sig": "v%d" % v if state != "silent" else None})
            cells[role] = {"windows": ws, "loop_bars": [], "one_shot": False, "clips": []}
        out.append({"scene": i, "name": name, "bars": bars, "windows": (bars + 7) // 8,
                    "cells": cells})
    return {"set": "x", "scenes": [1, len(out)], "window_bars": 8, "roles": list(ROLES),
            "rows": out, "unmapped": [], "ignored": [], "warnings": [], "notes": []}


def _all(ch):
    return {r: ch for r in ROLES}


SET = build(
    ("Intro · drone", 8, dict(pad="o")),
    ("Main 1 · + kick", 8, dict(kick="+", hats="+", pad="=")),
    ("Drop 1A · full", 16, dict(kick="==", snare="+=", hats="==", bass="+=", lead="+~", pad="==")),
    ("Drop 1B · same", 8, _all("=")),
    ("Breakdown · pads", 16, dict(kick="-.", snare="-.", hats="==", bass="==", lead="==", pad="==")),
    ("Drop 2A · full", 16, dict(kick="+=", snare="+=", hats="==", bass="==", lead="==", pad="==")),
    ("Drop 2B · rolling", 16, dict(kick="==", snare="==", hats="==", bass="==", lead="~=", pad="==")),
    ("Outro A", 8, dict(kick="=", snare="-", hats="=", bass="-", lead="-", pad="=")),
    ("Mystery", 8, dict(kick="=", hats="=", pad="=")))


def test_section_labels_come_from_the_scene_name():
    assert A.section_of("Drop 1A · impact") == ("Drop 1", "A")
    assert A.section_of("Main 1 · + kick") == ("Main 1", "")
    assert A.section_of("Outro B · drone + vox") == ("Outro", "B")
    assert A.section_of("Breakdown · lead 2-bar") == ("Breakdown", "")
    assert A.section_of("Build 2") == ("Build 2", "")
    assert A.section_of("Main 1 and bass") == ("Main 1", "")
    assert A.section_of("Dropout") is None and A.section_of("Verse 1") is None
    assert A.section_of("") is None


def test_the_section_map_and_the_drop_ratio():
    """Hand-counted: 13 steps, 104 bars. Drop 1 is scenes 3-4 (16 + 8 bars), Drop 2 scenes 6-7
    (16 + 16), so the ratio is 32 / 24. "Mystery" matches no label and is listed."""
    m = A.structure_metrics(SET)
    assert (m["bars"], m["steps"]) == (104, 13)
    assert [(s["label"], s["bar"], s["bars"], s["scenes"]) for s in m["sections"]] == [
        ("Intro", 1, 8, [1, 1]), ("Main 1", 9, 8, [2, 2]), ("Drop 1", 17, 24, [3, 4]),
        ("Breakdown", 41, 16, [5, 5]), ("Drop 2", 57, 32, [6, 7]), ("Outro", 89, 8, [8, 8]),
        (None, 97, 8, [9, 9])]
    assert m["unlabelled_scenes"] == [{"scene": 9, "name": "Mystery"}]
    assert m["drop_ratio"] == {"drop1_bars": 24, "drop2_bars": 32, "ratio": 1.333}


def test_steps_that_add_nothing_or_only_remove():
    """By hand, from the third step on: bar 33 (Drop 1B, all the same), 49 and 65 and 81 (a second
    step of a section that changes nothing) and 97 add nothing; bars 41 (kick and snare leave) and
    89 (snare, bass and lead leave) only remove. Steps that enter or vary are not flagged."""
    m = A.structure_metrics(SET)
    assert [(f["bar"], f["flag"]) for f in m["flags"]] == [
        (33, "nothing new"), (41, "only a removal"), (49, "nothing new"), (65, "nothing new"),
        (81, "nothing new"), (89, "only a removal"), (97, "nothing new")]
    step = next(c for c in m["changes"] if c["bar"] == 17)
    assert (step["enter"], step["vary"], step["flag"]) == (["bass", "lead", "snare"], [], None)
    assert next(c for c in m["changes"] if c["bar"] == 25)["vary"] == ["lead"]


def test_pitched_parts_per_section():
    """Pitched = bass, lead, arp, pad, drone, pluck, synth. Here only bass, lead and pad exist: the
    pad alone in the intro, main and outro, all three from the first Drop 1 step on."""
    got = {s["label"]: s["pitched_parts"] for s in A.structure_metrics(SET)["sections"]}
    assert got["Intro"] == {"max": 1, "per_step": [1]}
    assert got["Drop 1"] == {"max": 3, "per_step": [3, 3, 3]}
    assert got["Breakdown"]["max"] == 3 and got["Outro"]["max"] == 1


def test_hook_run_length_and_variation():
    """The lead plays steps 3-11 (bars 17-88, 9 steps of 8). It varies at bars 25 and 73; its
    signature takes 3 values (enter, the bar 25 variant, the bar 73 variant)."""
    h = A.structure_metrics(SET)["hook"]
    assert (h["first_bar"], h["steps_playing"], h["bars_playing"]) == (17, 9, 72)
    assert (h["longest_run"]["bar"], h["longest_run"]["bars"]) == (17, 72)
    assert (h["varies"], h["distinct_variants"]) == (2, 3)


def test_bass_entry_bar_and_whether_it_is_before_the_first_drop():
    """The bass enters in the first Drop 1 step, bar 17, which is the drop itself. Moved a scene
    earlier, into a Build, it arrives before the drop."""
    b = A.structure_metrics(SET)["bass"]
    assert (b["entry_bar"], b["first_drop_bar"], b["before_first_drop"]) == (17, 17, False)
    early = build(("Intro", 8, dict(pad="o")), ("Build 1", 8, dict(bass="+", pad="=")),
                  ("Drop 1", 8, dict(bass="=", kick="+", pad="=")))
    b = A.structure_metrics(early)["bass"]
    assert (b["entry_bar"], b["first_drop_bar"], b["before_first_drop"]) == (9, 17, True)


def test_kick_out_length_counts_only_between_two_stretches_of_kick():
    """The kick is in from bar 9; it is out for the two steps of the Breakdown, 16 bars from bar
    41, and plays again. The silent intro before bar 9 is not "kick out"."""
    k = A.structure_metrics(SET)["kick"]
    assert k["first_bar"] == 9 and k["longest_out_bars"] == 16
    assert [(g["bar"], g["bars"], g["section"]) for g in k["out"]] == [(41, 16, "Breakdown")]


def test_hats_constancy():
    """The hats enter at bar 9 and play every step after: 12 of 13 steps, constant. A table where
    they drop out for a step is not."""
    h = A.structure_metrics(SET)["hats"]
    assert (h["first_bar"], h["steps_playing"], h["of_steps"], h["constant"]) == (9, 12, 13, True)
    gap = build(("Main 1", 8, dict(hats="o")), ("Drop 1", 8, dict(hats="=")),
                ("Breakdown", 8, dict(hats="-")), ("Drop 2", 8, dict(hats="+")))
    h = A.structure_metrics(gap)["hats"]
    assert not h["constant"] and [(g["bar"], g["bars"]) for g in h["gaps"]] == [(17, 8)]


def test_a_role_the_set_lacks_is_none_in_its_metrics():
    bare = build(("Intro", 8, dict(pad="o")))
    bare["roles"] = ["pad"]
    bare["rows"][0]["cells"] = {"pad": bare["rows"][0]["cells"]["pad"]}
    m = A.structure_metrics(bare)
    assert (m["bass"], m["kick"], m["hats"], m["hook"], m["drop_ratio"]) == (None,) * 5


def test_diva_is_the_lead_through_the_sets_role_map_not_by_default(monkeypatch, tmp_path, capsys):
    """Without a map "27-Diva" matches no keyword and is listed as unmapped (so "Diva Pad" is not
    quietly a lead). With the Set's map file, {"27-Diva": "lead"}, it is the lead and the hook runs
    16 bars from bar 1."""
    m = live(["27-Diva"], scenes=2)
    for sc in (0, 1):
        put(m, 0, sc, [note(60, 0, 1)], loop=(0, 32))
    r = table(m, monkeypatch)
    assert r["roles"] == ["unmapped"] and r["unmapped"] == ["27-Diva"]
    rolemap = tmp_path / "roles.json"
    rolemap.write_text('{"27-Diva": "lead"}')
    A.main(["read", "--scenes", "1-2", "--metrics", "--json", "--roles", str(rolemap)])
    data = json.loads(capsys.readouterr().out)
    assert data["roles"] == ["lead"] and data["unmapped"] == []
    h = data["metrics"]["hook"]
    assert (h["first_bar"], h["bars_playing"], h["longest_run"]["bars"]) == (1, 16, 16)


def test_metrics_on_the_command_line(monkeypatch, capsys):
    """--metrics adds a "Structure" block to the text and a "metrics" key to the JSON."""
    m = live(["S-BD", "BASS"], scenes=2)
    m.song.scenes[0].name, m.song.scenes[1].name = "Intro", "Drop 1A"
    for sc in range(2):
        put(m, 0, sc, [note(36, 0, 1)])
    put(m, 1, 1, [note(40, 0, 2)])
    m.install(monkeypatch)
    A.main(["read", "--scenes", "1-2", "--metrics"])
    assert "Structure: 16 bars in 2 steps" in capsys.readouterr().out
    A.main(["read", "--scenes", "1-2", "--metrics", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["metrics"]["bass"]["entry_bar"] == 9
    assert [s["label"] for s in data["metrics"]["sections"]] == ["Intro", "Drop 1"]


# --------------------------------------------------------------------------
# review fixes
# --------------------------------------------------------------------------

def test_drop_ratio_sums_every_section_with_the_label():
    """Drop 1 is split by an unlabelled "Fill": 16 + 16 bars, not the last 16. Drop 2 is 32, so
    the ratio is 32 / 32 = 1."""
    t = build(("Drop 1A", 16, dict(pad="o=")), ("Fill", 8, dict(pad="=")),
              ("Drop 1B", 16, dict(pad="==")), ("Breakdown", 8, dict(pad="=")),
              ("Drop 2", 32, dict(pad="====")))
    m = A.structure_metrics(t)
    assert m["drop_ratio"] == {"drop1_bars": 32, "drop2_bars": 32, "ratio": 1.0}
    assert [(s["label"], s["bars"]) for s in m["sections"]] == [
        ("Drop 1", 16), (None, 8), ("Drop 1", 16), ("Breakdown", 8), ("Drop 2", 32)]


def test_a_part_of_a_role_stopping_is_a_removal_unless_a_one_shot_ran_out(monkeypatch):
    """Two pads in scene 1, one in scene 2: the role is `thin` and the step is "only a removal",
    not "nothing new". A one-shot that was still sounding when the scene ended is the same."""
    m = live(["PAD A", "PAD B"], scenes=2)
    for t, p in ((0, 48), (1, 55)):
        put(m, t, 0, [note(p, 0, 32)])
    put(m, 0, 1, [note(48, 0, 32)])
    r = table(m, monkeypatch)
    assert cell(r, 2, "pad")["windows"][0]["change"] == "thin"
    assert [(f["bar"], f["flag"]) for f in A.structure_metrics(r)["flags"]] == [
        (9, "only a removal")]
    m = FakeLive(tracks=("audio", "audio"), scenes=2)
    for t, n in zip(m.song.tracks, ("IMPACT", "DOWN")):
        t.name = n
    put(m, 0, 0, name="impact", loop=(0, 32), looping=False, audio=True)
    put(m, 1, 0, name="downer", loop=(0, 32), audio=True)
    put(m, 1, 1, name="downer", loop=(0, 32), audio=True)
    r = table(m, monkeypatch)
    assert cell(r, 2, "fx")["windows"][0]["change"] == "thin"


def test_two_tracks_with_one_name_are_two_voices(monkeypatch):
    """Both pads are called PAD. One changes its note between the scenes, the other does not: the
    role varies. Keyed by name the second would hide the first."""
    m = live(["PAD", "PAD"], scenes=2)
    put(m, 0, 0, [note(48, 0, 32)])
    put(m, 1, 0, [note(50, 0, 32)])
    put(m, 0, 1, [note(52, 0, 32)])
    put(m, 1, 1, [note(50, 0, 32)])
    r = table(m, monkeypatch)
    assert cell(r, 2, "pad")["windows"][0]["change"] == "vary"


def test_a_start_marker_inside_the_loop_is_not_a_variation(monkeypatch):
    """A 4-bar loop (0..16) started 4 beats in plays the note at 6 first and its cycle from beat 12
    on: over 16 bars it is one pattern, so the second step is the same as the first. By hand,
    both steps start the note at 6 at beats 2, 18 and the note at 0 at 12, 28 (rel. to the step).
    A lead-in before the loop (loop 8..16, start 0, a note at 2) is not the loop: it varies."""
    m = live(["LEAD", "ARP", "PAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 1), note(62, 6, 1)], loop=(0, 16), start=4)
    put(m, 1, 0, [note(60, 2, 1), note(62, 9, 1)], loop=(8, 16), start=0)
    put(m, 2, 0, [note(48, 0, 64)], loop=(0, 64))
    r = table(m, monkeypatch)
    assert [w["change"] for w in cell(r, 1, "lead")["windows"]] == [None, "same"]
    assert [w["change"] for w in cell(r, 1, "arp")["windows"]] == [None, "vary"]


def test_a_loop_longer_than_8_bars_is_named_in_the_warnings(monkeypatch):
    """A 12-bar loop in a 24-bar scene: its steps are slices of the loop and can read varied on a
    repeat, so the clip is listed, never silent about it."""
    m = live(["LEAD"], scenes=1)
    put(m, 0, 0, [note(60, 0, 1), note(62, 40, 1)], loop=(0, 48))
    put(m, 0, 0, [note(60, 0, 1), note(62, 40, 1)], loop=(0, 96))
    r = table(m, monkeypatch)
    assert any("loop longer than 8 bars on 'LEAD' in scenes 1" in w for w in r["warnings"])


def test_hats_that_the_outro_drops_are_still_constant():
    """Same rule as the kick-out: only a gap between two stretches counts. The hats play in the
    first three steps and are out for the last two (8 bars each): constant, 16 bars out at the end."""
    t = build(("Main 1", 8, dict(hats="o")), ("Drop 1", 8, dict(hats="=")),
              ("Drop 2", 8, dict(hats="=")), ("Outro", 16, dict(hats="-.")))
    h = A.structure_metrics(t)["hats"]
    assert h["constant"] and h["gaps"] == [] and h["trailing_out_bars"] == 16
    assert h["steps_playing"] == 3 and h["of_steps"] == 5


def test_the_metrics_run_on_one_bar_steps_without_a_live_table():
    """metrics_from_steps takes steps and sections only: 30 one-bar steps, kick in bars 1-10 and
    14-30, bass from bar 22, a Drop starting at bar 22. Entry and out-stretch are to the bar: bass
    22 (not 17 or 25), kick out 3 bars from bar 11. No step carries a change, so none is flagged."""
    play = lambda on: {"state": "playing" if on else "silent"}
    steps = [{"bar": b, "bars": 1, "cells": {"kick": play(b <= 10 or b >= 14),
                                              "bass": play(b >= 22)}} for b in range(1, 31)]
    secs = [{"label": "Intro", "bar": 1, "bars": 21}, {"label": "Drop 1", "bar": 22, "bars": 9}]
    m = A.metrics_from_steps(steps, secs, step_bars=1)
    assert m["step_bars"] == 1 and m["bars"] == 30
    assert (m["bass"]["entry_bar"], m["bass"]["first_drop_bar"], m["bass"]["before_first_drop"]) == (
        22, 22, False)
    assert [(g["bar"], g["bars"]) for g in m["kick"]["out"]] == [(11, 3)]
    assert m["kick"]["first_bar"] == 1 and m["flags"] == [] and m["changes"] == []
    assert (m["hook"], m["hats"], m["drop_ratio"]) == (None, None, None)


def test_the_section_labels_sections_py_writes_map_to_the_base_labels():
    """sections.py names sections Intro, Main, Build, Drop, Breakdown, Outro (numbered when they
    repeat, then a stem comment and a length), and Break from its loudness method. Those map; its
    other loudness names (Quiet, Mid, "C full") are listed as unlabelled, not guessed."""
    got = {n: A.section_of(n) for n in ("Drop 1 (bass in) \u00b7 64", "Main", "Build 2 (kick in)",
                                         "Breakdown", "Break", "Break 2", "Outro", "Quiet",
                                         "Mid", "C full")}
    assert got["Drop 1 (bass in) \u00b7 64"] == ("Drop 1", "")
    assert got["Build 2 (kick in)"] == ("Build 2", "") and got["Main"] == ("Main", "")
    assert got["Break"] == ("Breakdown", "") and got["Break 2"] == ("Breakdown 2", "")
    assert got["Breakdown"] == ("Breakdown", "") and got["Outro"] == ("Outro", "")
    assert got["Quiet"] is None and got["Mid"] is None and got["C full"] is None
    t = build(("Mid", 8, dict(pad="o")), ("Drop 1", 8, dict(pad="=")))
    assert A.structure_metrics(t)["unlabelled"] == [{"bar": 1, "bars": 8, "name": "Mid"}]


def test_silent_carrier_is_a_marker_at_the_end_of_the_clip_name(monkeypatch):
    """"(silent carrier)" at the end of the name is the marker. The words inside another name are
    not, and a role-map entry {"<clip name>": "play"} makes a marked clip count again (scene 3 is
    a different clip from scene 2, so it is "changed", where silent it would read silent)."""
    m = FakeLive(tracks=("audio", "midi"), scenes=3)
    m.song.tracks[0].name, m.song.tracks[1].name = "S-BD", "PAD"
    for sc, nm in enumerate(("E20 kick HP held open (silent carrier)",
                             "not a silent carrier here", "E21 kick (silent carrier)")):
        put(m, 0, sc, name=nm, loop=(0, 32), audio=True)
        put(m, 1, sc, [note(48, 0, 32)], loop=(0, 32))
    r = table(m, monkeypatch, roles={"E21 kick (silent carrier)": "play"})
    assert [states(r, sc, "kick") for sc in (1, 2, 3)] == [["silent"], ["changed"], ["changed"]]
    assert r["warnings"].count("silent-carrier clips counted as silent: 'S-BD' (1 clips)") == 1

#!/usr/bin/env python3
"""Tests for roadie_analysis/micro.py: synthetic stems at a known tempo in a temp study, nothing from the library."""

import json
import os
import shutil
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roadie_analysis import micro

SR = 22050
BPM = 120.0
BAR = 2.0            # seconds at 120 bpm
T0 = 0.5             # the first bar line in the file
BARS = 16
S16 = BAR / 16
A1, C2 = 33, 36      # MIDI


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(f, dur, decay, harm=(1, .4, .2)):
    t = np.arange(int(dur * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harm))
    return x * np.minimum(1, t / 0.004) * np.exp(-decay * t) * np.minimum(1, (dur - t) / 0.02)


BASS_STEPS = ((0, A1), (3, A1), (4, A1), (8, C2), (11, C2), (12, C2))       # o..o o... o..o o...
AM, F = (57, 60, 64), (53, 57, 60)
VOCAL_PHRASES = ((0, 6), (4 * 4 + 3, 8), (8 * 4, 6))                         # (first beat, beats): bar 1.1, bar 5.4, bar 9.1


def make_study(root, verdict="confident", stems=("bass", "synths", "vocals"), drums=True):
    total = int((T0 + BARS * BAR + 1.0) * SR)
    bass, syn, voc = np.zeros(total), np.zeros(total), np.zeros(total)
    for b in range(BARS):
        for st, m in BASS_STEPS:
            i = int((T0 + b * BAR + st * S16) * SR)
            x = 0.5 * tone(hz(m), 0.1, 12)
            bass[i:i + len(x)] += x
        i = int((T0 + b * BAR) * SR)
        x = sum(0.15 * tone(hz(m), BAR, 0.5, (1, .1)) for m in (AM if (b // 4) % 2 == 0 else F))
        syn[i:i + len(x)] += x
    for first, n in VOCAL_PHRASES:
        i, k = int((T0 + first * BAR / 4) * SR), int(n * BAR / 4 * SR)
        t = np.arange(k) / SR
        voc[i:i + k] += 0.3 * np.sin(2 * np.pi * 440 * t + 2 * np.sin(2 * np.pi * 5 * t)) * np.minimum(1, t / 0.02)
    sd = root / "Artist" / "Track" / "Track_120bpm_stems"
    sd.mkdir(parents=True)
    audio = {"bass": ("bass.flac", bass), "synths": ("synths.wav", syn), "vocals": ("vocals.flac", voc)}
    for s in stems:
        sf.write(str(sd / audio[s][0]), audio[s][1], SR)
    (sd / "alignment.json").write_text(json.dumps({"result": {
        "bpm": BPM, "downbeat_sec": T0 + 2 * BAR, "downbeat_verdict": verdict, "downbeat_confidence": 0.9,
        "warp_map": [], "detector": "heuristic"}}))
    (sd / "sections.json").write_text(json.dumps({"bpm": BPM, "method": "stems", "sections": [
        {"beat": 0, "bar": 1, "time_sec": T0, "duration_beats": 32, "label": "Intro", "stems": ["bass", "synths", "vocals"]},
        {"beat": 32, "bar": 9, "time_sec": T0 + 8 * BAR, "duration_beats": 32, "label": "Drop 1", "stems": ["bass", "synths"]}]}))
    (sd.parent / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "A minor", "camelot": "8A",
                                                      "key_source": "tagged"}))
    if drums:
        role = {"name": "hh offbeat", "named": True, "measured": 30, "line": "..x...x...x...x.",
                "per_section": [{"index": 0, "hits": 16, "bars_with": 4, "line": "..x...x...x...x."},
                                {"index": 1, "hits": 14, "bars_with": 4, "line": "..x...x...x...o."}]}
        kick = {"name": "kick", "named": True, "measured": 64, "line": "x...x...x...x...",
                "per_section": [{"index": 1, "hits": 32, "bars_with": 8, "line": "x...x...x...x..."}]}
        tiny = {"name": "hh 3", "named": False, "measured": 2, "line": "x...............",
                "per_section": [{"index": 0, "hits": 2, "bars_with": 1, "line": "x..............."}]}
        (sd / "drums.json").write_text(json.dumps({
            "kind": "drum_roles", "version": 1, "made": "2026-11-21T10:00:00+0100",
            "grid": {"tempo": BPM, "t0": T0, "beats_per_bar": 4,
                     "sections": [{"name": "Intro", "start_bar": 0, "bars": 8}, {"name": "Drop 1", "start_bar": 8, "bars": 8}]},
            "stems": {"hh": {"roles": [role, tiny]}, "kick": {"roles": [kick]}}}))
    return sd


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    sd = make_study(tmp_path_factory.mktemp("micro"))
    return sd, micro.micro_of(sd)


def test_bass_rhythm_and_degrees_are_read_back(measured):
    _sd, rec = measured
    for s in rec["sections"]:
        b = s["bass"]
        assert b["rhythm"] == "o..o o... o..o o..." and b["rhythm_share"] == 1.0
        assert b["rhythm_steps"] == [1, 4, 5, 9, 12, 13]
        assert b["onsets_per_bar"] == 6.0 and b["off_grid_share"] == 0.0
        assert b["on_step1_share"] == pytest.approx(1 / 6, abs=0.01) and b["on_offbeat_8ths_share"] == 0.0
        assert sorted(d["degree"] for d in b["degrees"]) == ["1", "b3"]
        assert b["pitch_classes"] == 2 and b["movement"] == "moving"
        assert b["register"] in ("A1", "C2") and b["register_range"] == "A1-C2"
        assert b["style"] == "stabby" and b["duty"] < 0.5
        assert b["repeat_bars"] == 1 and b["changes_at_bars"] == []
        assert b["bars_with_onsets"] == 8 and len(b["level_by_16th_db"]) == 16 and max(b["level_by_16th_db"]) == 0.0
        assert s["reliable"] is True


def test_two_chord_synth_loop_gives_two_chords_and_a_change_every_4_bars(measured):
    _sd, rec = measured
    y = rec["sections"][0]["synths"]
    assert y["chords"] == ["i"] * 4 + ["VI"] * 4
    assert y["distinct_chords"] == ["i", "VI"] and y["progression"] == "i VI"
    assert y["bars_per_chord"] == 4.0
    assert y["rhythm"] == "o... .... .... ...." and y["onsets_per_bar"] == 1.0
    assert y["drone"]["pitch_classes"] == ["A", "C"] or y["drone"]["pitch_classes"] == ["C", "A"]   # common tones held
    assert y["repeat_bars"] == 1 and y["changes_at_bars"] == [5]
    assert any("leads and pads" in n for n in y["notes"])


def test_vocal_bars_and_phrase_starts(measured):
    _sd, rec = measured
    intro, drop = (s["vocals"] for s in rec["sections"])
    assert intro["bars_with_vocal"] == "1-2, 5-7" and intro["share"] == 0.62
    assert intro["phrase_starts"] == [{"bar": 1, "beat": 1}, {"bar": 5, "beat": 4}]
    assert drop["bars_with_vocal"] == "1-2" and drop["phrase_starts"] == [{"bar": 1, "beat": 1}]


def test_drums_come_from_drums_json_and_hat_accent_is_missing_with_a_reason(measured):
    _sd, rec = measured
    intro, drop = (s["drums"] for s in rec["sections"])
    assert [r["role"] for r in intro["roles"]] == ["hh offbeat"]          # 2-hit roles are left out, as study show does
    hat = intro["roles"][0]
    assert hat["pattern"] == "..x. ..x. ..x. ..x." and hat["accent_16th"] is None and "per-hit levels" in hat["accent_note"]
    assert {r["role"] for r in drop["roles"]} == {"hh offbeat", "kick"}
    assert "accent_16th" not in next(r for r in drop["roles"] if r["role"] == "kick")
    assert rec["drums_made"] == "2026-11-21T10:00:00+0100"


def test_written_record_is_the_returned_one(measured):
    sd, rec = measured
    on_disk = json.loads((sd / "micro.json").read_text())
    assert on_disk["kind"] == "micro" and on_disk["sections"] == json.loads(json.dumps(rec["sections"]))
    assert rec["key"]["name"] == "A minor" and rec["grid"]["t0"] == T0


def test_chance_level_grid_flags_every_section_and_missing_stems_are_null_with_a_reason(tmp_path):
    sd = make_study(tmp_path, verdict="chance-level", stems=("vocals",), drums=False)
    rec = micro.micro_of(sd, write=False)
    assert rec["grid"]["reliable"] is False and "chance-level" in rec["grid"]["unreliable_because"][0]
    assert all(s["reliable"] is False for s in rec["sections"])
    assert all(s["bass"] is None and s["synths"] is None and s["drums"] is None for s in rec["sections"])
    assert any("no bass stem" in n for n in rec["notes"]) and any("no drums.json" in n for n in rec["notes"])
    assert not (sd / "micro.json").exists()


def test_cli_prints_the_summary(tmp_path, capsys):
    sd = make_study(tmp_path, stems=("vocals",), drums=False)
    assert micro.main([str(sd), "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "vocals  bars 1-2, 5-7" in out and "nothing written" in out
    shutil.rmtree(sd)
    assert micro.main([str(sd), "--no-write"]) == 1


@pytest.mark.parametrize("root,quality,want", [(9, "min", "i"), (5, "maj", "VI"), (0, "maj", "III"), (7, "maj", "VII"),
                                               (9, "5", "i5"), (4, "sus4", "vsus4"), (10, "maj", "bII"), (4, "maj", "V")])
def test_roman_numerals_in_a_minor(root, quality, want):
    assert micro.roman(root, quality, 9, "minor") == want


def test_match_chord_breaks_a_sus_tie_with_the_bass():
    v = np.zeros(12)
    v[[0, 2, 7]] = 1.0                       # C D G: Csus2 or Gsus4
    assert (micro.match_chord(v, bass_pc=7)["root"], micro.match_chord(v, bass_pc=7)["quality"]) == (7, "sus4")
    assert (micro.match_chord(v, bass_pc=0)["root"], micro.match_chord(v, bass_pc=0)["quality"]) == (0, "sus2")
    v = np.zeros(12)
    v[[9, 4]] = 1.0                          # A E: root and fifth
    assert micro.match_chord(v)["quality"] == "5" and micro.match_chord(v)["root"] == 9
    assert micro.match_chord(np.zeros(12)) is None


def test_repeat_period_is_the_smallest_that_holds_in_75_percent():
    assert micro.repeat(["a"] * 8) == (1, [])
    assert micro.repeat(["a", "b"] * 4) == (2, [])
    assert micro.repeat(["a", "b", "c", "d"] * 2 + ["a", "b", "c", "x"]) == (4, [12])
    assert micro.repeat(list("abcdefgh")) == (None, list(range(2, 9)))
    assert micro.repeat(["a", None, "a", "a"]) == (1, [])


def test_progression_is_the_shortest_cycle_that_fits():
    assert micro.cycle(["iv", "VI", "i", "VII"] * 3 + ["iv"]) == (4, 1.0, ["iv", "VI", "i", "VII"])
    assert micro.cycle(list("abcabcabd")) == (3, 0.83, list("abc"))
    assert micro.cycle("VI i VII iv VI i VII iv VI i VII i iv".split())[2] == ["VI", "i", "VII", "iv"]  # one extra i
    assert micro.cycle(["i", "VI"]) == (2, None, ["i", "VI"])          # once through: no cycle to speak of
    seq = "i VII iv VI i VII iv i III i VII iv VI i VII iv VI i III i".split()      # Destination Lost, Drop 2
    assert micro.cycle(seq) == (4, 0.8, ["i", "VII", "iv", "VI"])

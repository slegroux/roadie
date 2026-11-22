#!/usr/bin/env python3
"""Tests for studyreport.py: temp studies with hand-made records (no audio is read), nothing from the library."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
import studyreport

# 32 bars: Drop 1 (1-8), Breakdown (9-16, kick out, bass back at 13), Drop 2 (17-32)
KICK = "1" * 8 + "0" * 8 + "1" * 16
BASS = "1" * 8 + "0" * 4 + "1" * 20
HATS = "1" * 32
SYNTHS = "1" * 32
VOCALS = "0" * 8 + "1" * 8 + "0" * 16


def make_study(root, verdict="confident", micro=None, mix=True, key=True):
    folder = root / "Artist" / "Track [124 Am 8A]"
    sd = folder / "Track_124bpm_stems"
    sd.mkdir(parents=True)
    for n in ("drums_kick", "drums_hh", "bass", "synths", "vocals"):
        (sd / (n + ".flac")).write_bytes(b"")
    (sd / "alignment.json").write_text(json.dumps({"fingerprint": {}, "result": {
        "bpm": 124.0, "downbeat_sec": 0.5, "downbeat_verdict": verdict, "downbeat_confidence": 0.31 if verdict == "chance-level" else 0.8,
        "detector": "heuristic", "warp_map": [], "has_drift": False}}))
    (sd / "sections.json").write_text(json.dumps({"bpm": 124.0, "method": "stems", "sections": [
        {"beat": 0, "bar": 1, "time_sec": 0.5, "duration_beats": 32, "label": "Drop 1", "stems": ["kick", "bass", "hats", "synths"], "live_bar": 9.0},
        {"beat": 32, "bar": 9, "time_sec": 16.0, "duration_beats": 32, "label": "Breakdown", "stems": ["hats", "synths", "vocals"], "live_bar": 17.0},
        {"beat": 64, "bar": 17, "time_sec": 31.5, "duration_beats": 64, "label": "Drop 2", "stems": ["kick", "bass", "hats", "synths"], "live_bar": 25.0}],
        "bar_activity": {"first_bar": 1, "bars": 32, "stems": {"drums_kick": KICK, "bass": BASS, "drums_hh": HATS,
                                                                "synths": SYNTHS, "vocals": VOCALS}}}))
    if key:
        (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "A minor", "camelot": "8A",
                                                       "key_source": "online", "key_url": "https://example.org/track"}))
    if mix:
        (folder / "analysis").mkdir()
        roles = {"kick": {"absent": False, "trace": False, "level_db": -14.0}, "vocals": {"absent": True}}
        (folder / "analysis" / "Track.profile.json").write_text(json.dumps({
            "kind": "reference_profile", "version": 6, "inputs": {"mix": [], "stems": {}}, "sections": [
                {"name": "Drop 1", "start_bar": 0, "bars": 8, "roles": roles,
                 "mix": {"level_db": -9.5, "integrated_lufs": -8.9, "bands_db": {"sub": -12.0, "bass": -13.0, "low_mid": -20.0,
                                                                                  "mid": -22.0, "high_mid": -28.0, "high": -33.0}}}]}))
    if micro is not None:
        (sd / "micro.json").write_text(json.dumps(micro))
    return folder


def micro_record(reliable=True):
    bass = {"bars_playing": 8, "rhythm": "o..o o... o..o o...", "rhythm_steps": [1, 4, 5, 9, 12, 13], "rhythm_share": 0.75,
            "onsets_per_bar": 6.0, "off_grid_share": 0.0, "on_step1_share": 0.17, "on_offbeat_8ths_share": 0.0,
            "degrees": [{"degree": "1", "note": "A", "share": 0.5}, {"degree": "b3", "note": "C", "share": 0.5}],
            "pitch_names": None, "pitched_share": 1.0, "pitch_classes": 2, "top_pitch_class": "A", "top_pitch_class_share": 0.5,
            "movement": "moving", "register": "A1", "register_range": "A1-C2", "duty": 0.3, "style": "stabby",
            "repeat_bars": 1, "changes_at_bars": [8], "notes": []}
    synths = {"bars_playing": 8, "rhythm": "o... .... .... ....", "rhythm_steps": [1], "rhythm_share": 1.0, "onsets_per_bar": 1.0,
              "off_grid_share": 0.0, "chords": ["i"] * 4 + ["VI"] * 4, "progression": "i VI", "bars_per_chord": 4.0,
              "loop_bars": None, "distinct_chords": ["i", "VI"], "drone": {"pitch_classes": [], "bars_share": [], "degrees": []},
              "repeat_bars": 1, "changes_at_bars": [5], "runs": [{"chord": "i", "bars": 4}, {"chord": "VI", "bars": 4}],
              "notes": ["the synths stem holds leads and pads together"]}
    silent = {"bars_playing": 0, "share": 0.0, "bars_with_vocal": None, "phrase_starts": None, "phrase_start_beats": None,
              "notes": ["silent in this section"]}
    drums = {"drums_section": "Drop 1", "notes": [], "roles": [
        {"stem": "hh", "role": "hh offbeat", "named": True, "hits": 30, "bars_with": 8, "pattern": "..x. ..x. ..x. ..x.",
         "accent_16th": None, "accent_note": "drums.json records no per-hit levels"}]}
    sec = {"index": 0, "name": "Drop 1", "file_bar": 1, "start_bar": 0, "bars": 8, "reliable": reliable,
           "bass": bass, "synths": synths, "vocals": silent, "drums": drums}
    return {"kind": "micro", "version": 1, "made": "2026-11-21T12:00:00+0100",
            "grid": {"reliable": reliable, "unreliable_because": [] if reliable else ["the downbeat is chance-level"]},
            "key": {"name": "A minor"}, "drums_made": None, "notes": [], "sections": [sec]}


def report(folder, capsys, *extra):
    assert study.main(["report", str(folder), *extra]) == 0
    return capsys.readouterr().out


def test_known_grid_gives_the_section_map_kick_out_and_groove_blocks(tmp_path):
    st = study.load_study(str(make_study(tmp_path, micro=micro_record())), tmp_path)
    mc, why = studyreport.read_micro(st)
    d = studyreport.report_data(st, mc, why)
    assert d["grid"]["reliable"] is True and d["grid"]["warp_map"] is False
    assert [(s["label"], s["file_bar"], s["live_bar"], s["bars"]) for s in d["sections"]] == [
        ("Drop 1", 1, 9.0, 8), ("Breakdown", 9, 17.0, 8), ("Drop 2", 17, 25.0, 16)]
    kick = d["structure"]["kick"]
    assert [(o["bar"], o["bars"], o["section"]) for o in kick["out"]] == [(9, 8, "Breakdown")]
    blocks = d["structure"]["blocks"]
    assert [(b["bar"], b["bars"]) for b in blocks] == [(1, 8), (17, 16)]
    assert blocks[1]["kick_out_before"] == 8 and blocks[1]["bass_lead"] == 4
    assert d["entries"]["vocals"]["enters"] == 9 and d["entries"]["vocals"]["last_bar"] == 16
    assert d["entries"]["bass"]["out"][0]["bar"] == 9 and d["entries"]["bass"]["out"][0]["bars"] == 4
    md = studyreport.to_markdown(d)
    for heading in ("## Section map", "## Entries and exits", "## Kick-outs", "## Groove blocks", "## Mix profile per section",
                    "## Micro", "### Bass", "### Synths", "### Vocals", "### Drums", "## Missing layers"):
        assert heading in md
    assert "| Drop 1 | 1 | 9 | 8 | kick bass hats synths |" in md and "| 9 | 8 | Breakdown |" in md
    assert "A minor (Am, Camelot 8A), source online, https://example.org/track" in md
    assert studyreport.FLAG not in md
    assert "`o..o o... o..o o...` (75 %)" in md and "1 50 %, b3 50 %" in md and "A1 (A1-C2)" in md
    assert "| Drop 1 | 8 of 8 | `o... .... .... ....` (100 %) | 1 | i VI | 4 | i, VI |" in md
    assert "missing (no per-hit levels in drums.json)" in md and "- (silent in this section)" in md
    assert "missing: drums" in md and "missing: micro" not in md


def test_chance_level_grid_is_flagged_on_every_bar_and_pattern(tmp_path, capsys):
    folder = make_study(tmp_path, verdict="chance-level", micro=micro_record(reliable=False))
    md = report(folder, capsys)
    f = studyreport.FLAG
    assert "grid unreliable" in md and "chance-level (confidence 0.31)" in md
    assert "| Drop 1 | 1%s | 9%s | 8 |" % (f, f) in md
    assert "| 9%s | 8 | Breakdown |" % f in md
    assert "`o..o o... o..o o...`%s" % f in md and "i VI%s" % f in md and "`..x. ..x. ..x. ..x.`%s" % f in md


def test_missing_layers_are_listed_with_their_command(tmp_path, capsys):
    folder = make_study(tmp_path, mix=False, key=False)
    md = report(folder, capsys)
    assert "missing: key: `roadie key \"Track\"" in md
    assert "missing: mix: `reference_profile" in md and "missing: drums: `roadie drums" in md
    assert "missing: micro: `roadie micro \"Artist/Track\"" in md
    assert "## Micro\n\nmissing: micro" in md and "- **key**: missing" in md


def test_write_puts_report_md_in_the_study_and_json_is_the_record(tmp_path, capsys):
    folder = make_study(tmp_path, micro=micro_record())
    out = report(folder, capsys, "--write")
    assert out.strip() == "wrote %s" % (folder / "report.md")
    text = (folder / "report.md").read_text()
    assert text.startswith("# Artist - Track\n") and "## Micro" in text
    data = json.loads(report(folder, capsys, "--json"))
    assert data["kind"] == "study_report" and data["micro"]["sections"][0]["bass"]["rhythm"] == "o..o o... o..o o..."
    assert report(folder, capsys) == text                     # stdout is the same Markdown as the file


def test_a_report_on_a_study_that_is_not_there_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(study, "STUDIES_ROOTS", [])
    assert study.main(["report", str(tmp_path / "nothing")]) == 1
    assert "study: report:" in capsys.readouterr().err


def test_micro_from_an_older_drums_json_gets_a_note(tmp_path):
    folder = make_study(tmp_path, micro=dict(micro_record(), drums_made="2026-11-20T09:00:00+0100"))
    sd = folder / "Track_124bpm_stems"
    (sd / "drums.json").write_text(json.dumps({"kind": "drum_roles", "version": 1, "made": "2026-11-21T10:00:00+0100",
                                               "inputs": [], "stems": {}}))
    st = study.load_study(str(folder), tmp_path)
    d = studyreport.report_data(st, *studyreport.read_micro(st))
    assert any("older drums.json" in c for c in d["caveats"])


@pytest.mark.parametrize("bad", ["{not json", json.dumps({"kind": "other"})])
def test_unreadable_micro_json_is_said_so(tmp_path, bad):
    folder = make_study(tmp_path)
    (folder / "Track_124bpm_stems" / "micro.json").write_text(bad)
    mc, why = studyreport.read_micro(study.load_study(str(folder), tmp_path))
    assert mc is None and ("could not be read" in why or "not a micro record" in why)

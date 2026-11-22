#!/usr/bin/env python3
"""Tests for studyreport.py: temp studies with hand-made records (no audio is read), nothing from the library."""

import json
import os
import sys

import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
import studyreport

# 32 bars: Drop 1 (1-8), Breakdown (9-16, kick out, bass back at 13), Drop 2 (17-32)
KICK = "1" * 8 + "0" * 8 + "1" * 16
BASS = "1" * 8 + "0" * 4 + "1" * 20
HATS = "1" * 32
SYNTHS = "1" * 32
VOCALS = "0" * 8 + "1" * 8 + "0" * 16


def make_study(root, verdict="confident", micro=None, mix=True, key=True, title="Track"):
    folder = root / "Artist" / ("%s [124 Am 8A]" % title)
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
        if micro.get("inputs") == "as on disk":        # what roadie micro records: the three stems and the grid files
            micro = dict(micro, inputs=[{"name": n, "size": (sd / n).stat().st_size} for n in (
                "bass.flac", "synths.flac", "vocals.flac", "alignment.json", "sections.json")])
        (sd / "micro.json").write_text(json.dumps(micro))
    return folder


def micro_record(reliable=True):
    bass = {"bars_playing": 8, "bars_with_onsets": 8, "level_by_16th_db": [-12.0, -3.0] + [0.0] * 14, "rhythm": "o..o o... o..o o...", "rhythm_steps": [1, 4, 5, 9, 12, 13], "rhythm_share": 0.75,
            "onsets_per_bar": 6.0, "off_grid_share": 0.0, "on_step1_share": 0.17, "on_offbeat_8ths_share": 0.0,
            "degrees": [{"degree": "1", "note": "A", "share": 0.5}, {"degree": "b3", "note": "C", "share": 0.5}],
            "pitch_names": None, "pitched_share": 1.0, "pitch_classes": 2, "top_pitch_class": "A", "top_pitch_class_share": 0.5,
            "movement": "moving", "register": "A1", "register_range": "A1-C2", "duty": 0.3, "style": "stabby",
            "repeat_bars": 1, "changes_at_bars": [8], "notes": []}
    synths = {"bars_playing": 8, "bars_with_onsets": 8, "level_by_16th_db": [0.0] * 16, "progression_share": None, "rhythm": "o... .... .... ....", "rhythm_steps": [1], "rhythm_share": 1.0, "onsets_per_bar": 1.0,
              "off_grid_share": 0.0, "chords": ["i"] * 4 + ["VI"] * 4, "progression": "i VI", "bars_per_chord": 4.0,
              "loop_bars": None, "distinct_chords": ["i", "VI"], "drone": {"pitch_classes": [], "bars_share": [], "degrees": []},
              "repeat_bars": 1, "changes_at_bars": [5], "runs": [{"chord": "i", "bars": 4}, {"chord": "VI", "bars": 4}],
              "notes": ["the synths stem holds leads and pads together"]}
    silent = {"bars_playing": 0, "share": 0.0, "bars_with_vocal": None, "phrase_starts": None, "phrase_start_beats": None,
              "notes": ["silent in this section"]}
    drums = {"drums_section": "Drop 1", "notes": [], "roles": [
        {"stem": "hh", "role": "hh offbeat", "named": True, "hits": 30, "bars_with": 8, "pattern": "..x. ..x. ..x. ..x.",
         "accent_16th": None, "accent_note": "drums.json records no per-hit levels"}]}
    secs = [{"index": i, "name": name, "file_bar": bar, "start_bar": bar - 1, "bars": bars, "reliable": reliable,
             "bass": bass, "synths": synths, "vocals": silent, "drums": dict(drums, drums_section=name)}
            for i, (name, bar, bars) in enumerate((("Drop 1", 1, 8), ("Breakdown", 9, 8), ("Drop 2", 17, 16)))]
    return {"kind": "micro", "version": 1, "made": "2026-11-21T12:00:00+0100",
            "grid": {"tempo": 124.0, "t0": 0.5, "reliable": reliable,
                     "unreliable_because": [] if reliable else ["the downbeat is chance-level"]},
            "key": {"name": "A minor"}, "drums_made": None, "notes": [], "inputs": "as on disk", "sections": secs}


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
    assert "`o..o o... o..o o...` (75 % of 8)" in md and "| 16ths 1 |" in md and "1 50 %, b3 50 %" in md and "A1 (A1-C2)" in md
    assert "| Drop 1 | 8 of 8 | `o... .... .... ....` (100 % of 8) | 1 | none | i VI | 4 | i, VI |" in md
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
    assert "| 1, 4, 5, 9, 12, 13%s |" % f in md and "| 17 %% / 0 %%%s |" % f in md and "| i, VI%s |" % f in md


def test_a_fresh_micro_json_has_no_caveat_and_a_changed_input_makes_it_stale(tmp_path):
    folder = make_study(tmp_path, micro=micro_record())
    st = study.load_study(str(folder), tmp_path)
    d = studyreport.report_data(st, *studyreport.read_micro(st))
    assert d["micro_stale"] == [] and not any(c.startswith("micro:") for c in d["caveats"])
    (folder / "Track_124bpm_stems" / "bass.flac").write_bytes(b"new")
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "C minor"}))
    st = study.load_study(str(folder), tmp_path)
    d = studyreport.report_data(st, *studyreport.read_micro(st))
    assert any("changed since: bass.flac" in w for w in d["micro_stale"]) and any("key A minor, now C minor" in w
                                                                                    for w in d["micro_stale"])
    assert any("micro.json is stale" in c and "re-run roadie micro" in c for c in d["caveats"])


def test_micro_stale_reads_drums_json_through_read_text_not_the_study_record(tmp_path, monkeypatch):
    folder = make_study(tmp_path, micro=dict(micro_record(), drums_made="2026-11-20T09:00:00+0100"))
    st = study.load_study(str(folder), tmp_path)
    st.layers["drums"].data = {"made": "2026-11-20T09:00:00+0100"}      # what a lagging mount would show: no change
    read = []
    real = studyreport.read_text

    def spy(path):
        read.append(Path(path).name)
        return json.dumps({"made": "2026-11-21T10:00:00+0100"}) if Path(path).name == "drums.json" else real(path)

    monkeypatch.setattr(studyreport, "read_text", spy)
    why = studyreport.micro_stale(st, json.loads((folder / "Track_124bpm_stems" / "micro.json").read_text()))
    assert "drums.json" in read and any("older drums.json" in w and "2026-11-21T10:00:00" in w for w in why)


def test_an_empty_cell_says_why(tmp_path, capsys):
    mc = micro_record()
    mc["sections"][0]["bass"] = dict(mc["sections"][0]["bass"], rhythm=None, rhythm_steps=None, rhythm_share=None,
                                     bars_with_onsets=0, on_step1_share=None, on_offbeat_8ths_share=None, repeat_bars=None,
                                     changes_at_bars=None, notes=["no onset on the grid in the bars it plays",
                                                                  "repeat: too few playing bars"])
    mc["sections"][1]["synths"] = dict(mc["sections"][1]["synths"], rhythm=None, rhythm_share=None, bars_with_onsets=0,
                                       notes=["the synths stem holds leads and pads together", "no onset on the grid in "
                                              "the bars it plays"])
    md = report(make_study(tmp_path, micro=mc), capsys)
    assert "| - (no onset on the grid in the bars it plays) | - |" in md and "- (too few playing bars)" in md
    assert "| Breakdown | 8 of 8 | - (no onset on the grid in the bars it plays) |" in md     # not the stem-wide synths note
    assert "- Drop 1: no onset on the grid in the bars it plays; repeat: too few playing bars" in md


def test_study_show_does_not_list_micro_json_as_an_extra(tmp_path):
    st = study.load_study(str(make_study(tmp_path, micro=micro_record())), tmp_path)
    assert not any(p.name == "micro.json" for p in st.extras)


def test_pcloud_report_write_reads_back_and_a_mismatch_raises(tmp_path, monkeypatch):
    folder = make_study(tmp_path)
    st = study.load_study(str(folder), tmp_path)
    store, temps = {}, []

    def fake_rclone(*args):
        assert args[0] == "copyto" and args[2] == "pcloud:S/report.md"
        temps.append(args[1])
        store["text"] = open(args[1]).read()
        return ""

    monkeypatch.setattr(studyreport.keymod, "pcloud_remote", lambda p: "pcloud:S/report.md" if p == folder / "report.md" else None)
    monkeypatch.setattr(studyreport.studymod, "_rclone", fake_rclone)
    monkeypatch.setattr(studyreport.studymod, "_remote_read", lambda r: store["text"])
    assert studyreport.write_report(st, "# report\n") == "pcloud:S/report.md" and store["text"] == "# report\n"
    monkeypatch.setattr(studyreport.studymod, "_remote_read", lambda r: "something else")
    with pytest.raises(OSError, match="read back"):
        studyreport.write_report(st, "# report\n")
    assert temps and not any(os.path.exists(t) for t in temps) and not (folder / "report.md").exists()


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


def artist_micro(duty, on1, loop, reliable=True):
    mc = micro_record(reliable)
    for sec in mc["sections"]:
        sec["bass"] = dict(sec["bass"], duty=duty, on_step1_share=on1)
        sec["synths"] = dict(sec["synths"], progression=loop, progression_share=1.0)
    return mc


@pytest.fixture
def artist_root(tmp_path, monkeypatch):
    make_study(tmp_path, micro=artist_micro(0.3, 0.17, "i VI"), title="One")
    make_study(tmp_path, micro=artist_micro(0.5, 0.25, "VI i"), title="Two")
    make_study(tmp_path, verdict="chance-level", micro=artist_micro(0.9, 0.9, "i iv", reliable=False), title="Three")
    make_study(tmp_path, mix=False, title="Four")                             # no micro, no mix
    monkeypatch.setattr(study, "STUDIES_ROOTS", [study.Root("t", tmp_path)])
    return tmp_path


def test_artist_medians_leave_out_unreliable_grids_only_for_bar_1_dependent_fields(artist_root):
    a = studyreport.artist_data("Artist")
    assert sorted(x["title"] for x in a["rows"]) == ["Four", "One", "Three", "Two"]
    agg = {g["field"]: g for g in a["aggregates"]}
    on1, duty = agg["bass_on1"], agg["bass_duty"]
    assert on1["bar1"] and (on1["median"], on1["used"], on1["of"]) == (0.21, 2, 4) and on1["left_out_grid"] == ["Three"]
    assert on1["no_value"] == ["Four"]
    assert not duty["bar1"] and (duty["median"], duty["used"]) == (0.5, 3) and duty["left_out_grid"] == []
    assert (agg["kickout_bars"]["median"], agg["kickout_bars"]["used"]) == (8, 4)          # every track: a length
    md = studyreport.artist_markdown(a)
    assert "| bass onsets on step 1 | bar-1-dependent | 0.21 | 0.17-0.25 | 2 of 4 | unreliable grid: Three; no value: Four |" in md
    assert "| bass duty | shift-invariant | 0.50 | 0.30-0.90 | 3 of 4 | no value: Four |" in md
    assert "Bar-1-dependent ones" in md and "| Three (old stems) | 124 | Am 8A (online) | chance-level † |" in md


def _kinds(a):
    return {g["kind"]: g for g in a["patterns"]}


def test_artist_patterns_give_every_class_out_of_the_tracks_that_qualify(artist_root):
    k = _kinds(studyreport.artist_data("Artist"))
    loops = k["chord loop (numerals, any rotation)"]
    assert loops["of"] == 3 and [(v["pattern"] in ("i VI", "VI i"), v["tracks"]) for v in loops["values"]] == [
        (True, ["One", "Two"]), (False, ["Three"])]                          # the 1-track class is listed too
    assert loops["left_out"] == {"no micro": ["Four"]}
    assert k["longest kick-out"]["of"] == 4 and k["longest kick-out"]["values"] == [
        {"pattern": "8-15 bars", "tracks": ["Four", "One", "Three", "Two"]}]
    assert k["drum role"]["of"] == 3 and k["drum role"]["left_out"] == {"no drums": ["Four"]}
    line = k["drum role pattern (16 steps)"]
    assert line["of"] == 2 and line["left_out"]["unreliable grid"] == ["Three"] and line["values"][0]["tracks"] == ["One", "Two"]


def test_a_detected_key_leaves_a_track_out_of_the_chord_loops(artist_root):
    marker = artist_root / "Artist" / "Two [124 Am 8A]" / "study.json"
    marker.write_text(json.dumps({"kind": "study_marker", "key": "A minor", "key_source": "detected", "key_confidence": 0.07}))
    loops = _kinds(studyreport.artist_data("Artist"))["chord loop (numerals, any rotation)"]
    assert loops["of"] == 2 and loops["left_out"]["detected or no key"] == ["Two"]
    assert [v["tracks"] for v in loops["values"]] == [["One"], ["Three"]]


def test_artist_markdown_states_denominators_and_too_few_to_compare(artist_root):
    md = studyreport.artist_markdown(studyreport.artist_data("Artist"))
    assert "| macro | longest kick-out | 4 of 4 (tracks with activity) | all 4: 8-15 bars | - |" in md
    assert "| micro | drum role pattern (16 steps) | 2 of 4 (tracks with drums, a reliable grid) |" in md
    assert "unreliable grid: Three; no drums: Four" in md or "no drums: Four; unreliable grid: Three" in md
    assert ": 2 of 3 (One, Two)" in md and "by construction" in md
    (artist_root / "Artist" / "One [124 Am 8A]" / "Track_124bpm_stems" / "alignment.json").write_text(json.dumps({"result": {
        "bpm": 124.0, "downbeat_sec": 0.5, "downbeat_verdict": "chance-level", "downbeat_confidence": 0.3}}))
    md = studyreport.artist_markdown(studyreport.artist_data("Artist"))
    assert "fewer than 2 tracks qualify (1): not compared" in md


def test_an_rclone_failure_in_one_study_does_not_stop_the_artist_report(artist_root, monkeypatch):
    import subprocess
    real = studyreport.read_micro

    def flaky(st):
        if st.title == "Two":
            raise subprocess.CalledProcessError(1, ["rclone", "cat"], stderr="pcloud: timeout")
        return real(st)

    monkeypatch.setattr(studyreport, "read_micro", flaky)
    a = studyreport.artist_data("Artist")
    assert [u["title"] for u in a["unreadable"]] == ["Two"] and len(a["rows"]) == 3
    md = studyreport.artist_markdown(a)
    assert "3 of 3, 1 unreadable" in md and "Two: could not be read" in md


def test_artist_report_lists_missing_layers_and_writes_artist_md(artist_root, capsys):
    assert study.main(["report", "--artist", "Artist", "--write"]) == 0
    assert capsys.readouterr().out.strip() == "wrote %s" % (artist_root / "Artist" / "_artist.md")
    md = (artist_root / "Artist" / "_artist.md").read_text()
    assert "| Four | micro | `roadie micro \"Artist/Four\" (writes micro.json beside drums.json)` |" in md
    assert "| Four | mix | `reference_profile" in md and "missing: micro" in md
    data = json.loads((lambda: (study.main(["report", "--artist", "t:Artist", "--json"]), capsys.readouterr().out)[1])())
    assert data["kind"] == "artist_report" and len(data["rows"]) == 4
    assert study.main(["report"]) == 2 and study.main(["report", "One", "--artist", "Artist"]) == 2
    assert study.main(["report", "--artist", "Nobody"]) == 1


def test_the_card_parts_show_the_kick_out_and_the_bar_before_it(tmp_path):
    mc = micro_record()
    mc["transitions"] = [{"bar": 8, "beat": 28.0, "kind": "dropout", "ratio": 4.0, "into": "Breakdown"}]
    st = study.load_study(str(make_study(tmp_path, micro=mc)), tmp_path)
    md = studyreport.to_markdown(studyreport.report_data(st, *studyreport.read_micro(st)))
    em = md.split("## Element map")[1].split("```")[1].splitlines()
    assert em[1].split() == ["bar", "9"] and em[2].count("|") == 3 and "|Dr" in em[2]          # 2, 2 and 4 columns
    assert next(l for l in em if l.startswith("kick")).split()[1] == "██··████"       # out in Live bars 17-24
    assert next(l for l in em if l.startswith("bass")).split()[1] == "██·█████"
    assert "- Bar 16, the bar before **Breakdown**: a dropout: ≥ 4× fewer drum hits" in md          # Live bar 16, capped
    assert "Longest stretch with no element coming in or going out: bars 25–40 (16 bars)." in md


def test_the_card_parts_say_what_is_missing_and_flag_an_unreliable_grid(tmp_path):
    st = study.load_study(str(make_study(tmp_path, verdict="chance-level")), tmp_path)
    md = studyreport.to_markdown(studyreport.report_data(st, *studyreport.read_micro(st)))
    assert "missing: micro (the fill or dropout scan is part of `roadie micro`)" in md
    assert md.split("## Element map")[1].split("```")[1].splitlines()[1].split() == ["bar", "9†"]


def test_bass_duty_is_weighted_by_the_bars_it_plays(tmp_path):
    mc = micro_record()
    for sec, (duty, bars) in zip(mc["sections"], ((0.2, 8), (0.2, 8), (0.8, 16))):
        sec["bass"] = dict(sec["bass"], duty=duty, bars_playing=bars)
    st = study.load_study(str(make_study(tmp_path, micro=mc)), tmp_path)
    row = studyreport.track_row(st, studyreport.report_data(st, *studyreport.read_micro(st)))
    assert row["bass_duty"] == 0.5                                           # the median of the sections would be 0.2


def test_bass_entry_says_whether_the_activity_or_the_drop_label_is_missing(artist_root):
    sj = artist_root / "Artist" / "Two [124 Am 8A]" / "Track_124bpm_stems" / "sections.json"
    data = json.loads(sj.read_text())
    for s in data["sections"]:
        s["label"] = s["label"].replace("Drop", "Main")
    sj.write_text(json.dumps(data))
    left = _kinds(studyreport.artist_data("Artist"))["bass entry"]["left_out"]
    assert left == {"no Drop label": ["Two"]}


def test_the_transitions_say_hits_after_a_drumless_section_and_warn_on_a_shaky_grid(tmp_path):
    mc = micro_record(reliable=False)
    mc["transitions"] = [{"bar": 8, "beat": 28.0, "kind": "fill", "ratio": 40.0, "into": "Breakdown",
                          "median_hits": 0.0, "last_hits": 20}]
    st = study.load_study(str(make_study(tmp_path, verdict="chance-level", micro=mc)), tmp_path)
    md = studyreport.to_markdown(studyreport.report_data(st, *studyreport.read_micro(st)))
    assert "- Bar 16†, the bar before **Breakdown**: drums come in (20 hits; the section had none in its usual bar)." in md
    assert md.count("may be the line itself, a bar late") == 1

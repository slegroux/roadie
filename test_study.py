#!/usr/bin/env python3
"""Tests for study.py: temp study trees, nothing from the real Studies folder."""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
from study import load_study, to_json

HERE = os.path.dirname(os.path.abspath(__file__))


def make_study(root, artist, title, sections=True, alignment=True):
    stems = root / artist / title / ("%s_124bpm_stems" % title)
    stems.mkdir(parents=True)
    for n in ("drums_kick", "drums_hh", "bass", "synths", "vocals"):
        (stems / (n + ".flac")).write_bytes(b"")
    (stems / "bass.flac.asd").write_bytes(b"")
    if alignment:
        (stems / "alignment.json").write_text(json.dumps({"fingerprint": {}, "result": {
            "bpm": 124.0, "downbeat_verdict": "chance-level", "downbeat_confidence": 0.3,
            "detector": "heuristic"}}))
    if sections:
        (stems / "sections.json").write_text(json.dumps({"bpm": 124.0, "method": "stems", "sections": [
            {"beat": 0, "bar": 1, "duration_beats": 96, "stems": ["kick", "hats"], "label": "Intro", "live_bar": 9.0},
            {"beat": 96, "bar": 25, "duration_beats": 160, "stems": ["kick", "bass"], "label": "Drop", "live_bar": 33.0},
        ]}))
    return root / artist / title


def test_load_study_reads_grid_and_sections_and_reports_the_rest_missing(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    s = load_study("flare", tmp_path)
    assert (s.artist, s.title, s.bpm, s.bar_offset) == ("Einmusik", "Flare", 124.0, 8)
    assert sorted(s.stems) == ["bass", "hats", "kick", "synths", "vocals"]   # the .asd is not a stem
    secs = s.sections()
    assert (secs[0].label, secs[0].bar, secs[0].bars, secs[0].live_bar) == ("Intro", 1, 24, 9.0)
    assert secs[1].roles == ["kick", "bass"]
    assert s.layers["sections"].source.name == "sections.json"
    assert "labels by rule, not checked by ear" in s.layers["sections"].caveats
    assert any("chance-level" in c for c in s.layers["grid"].caveats)
    for name in ("mix", "activity", "drums"):
        assert s.layers[name].data is None and s.layers[name].missing
    assert s.mix() is None and s.activity() is None


def test_study_without_sections_json_has_empty_sections_and_a_command(tmp_path):
    make_study(tmp_path, "Bodzin", "Earth", sections=False)
    s = load_study("Earth", tmp_path)
    assert s.sections() == [] and s.layers["sections"].missing == "roadie sections <stems>"
    assert s.layers["grid"].data is not None


def test_ambiguous_title_lists_the_matches_and_unknown_is_an_error(tmp_path):
    make_study(tmp_path, "A", "Earth")
    make_study(tmp_path, "B", "Earth")
    with pytest.raises(ValueError, match=r"A/Earth.*B/Earth"):
        load_study("Earth", tmp_path)
    assert load_study("B/Earth", tmp_path).artist == "B"
    with pytest.raises(ValueError, match="no study named"):
        load_study("Nope", tmp_path)


def test_a_path_resolves_without_the_root(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare")
    assert load_study(str(folder), tmp_path / "elsewhere").title == "Flare"


def test_to_json_is_a_versioned_record_with_paths_relative_to_the_study(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    d = json.loads(json.dumps(to_json(load_study("Flare", tmp_path))))
    assert (d["kind"], d["version"]) == ("study", 1)
    assert sorted(d["layers"]) == ["activity", "drums", "grid", "mix", "sections"]
    assert d["layers"]["sections"]["source"] == "Flare_124bpm_stems/sections.json"
    assert d["layers"]["mix"]["missing"]


def test_cli_show_prints_the_record_and_json(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    env = dict(os.environ, ROADIE_STUDIES=str(tmp_path))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Flare", *a],
                                    capture_output=True, text=True, env=env)
    text = run()
    assert text.returncode == 0 and "124 bpm" in text.stdout and "Live bar 9" in text.stdout
    assert "mix       missing:" in text.stdout
    assert json.loads(run("--json").stdout)["kind"] == "study"
    bad = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Nope"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 1 and "no study named" in bad.stderr

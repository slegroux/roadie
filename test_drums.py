"""roadie_analysis.drums, `roadie drums` and the study reader's drums layer, on synthetic stems at 124 BPM.

Four-on-the-floor kick, open-ish hats on the offbeats and closed hats on every other 16th, built the way
test_analysis_drumroles.py builds its stems, with the downbeat two bars into the file so that the default grid has to be
moved back by whole bars. Each test is one behaviour; the stems are made and analysed once.
"""
import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import roadie_drums  # noqa: E402
import study  # noqa: E402
from roadie_analysis import drums  # noqa: E402
from test_analysis_drumroles import BPM, SR, STEP, T0, at, hat_events, stem  # noqa: E402

BARS = 8
BAR_S = 16 * STEP


def kick_hit(db=0.0, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.05) * 10 ** (db / 20)


def make_stems(root):
    """<root>/Test/Song/Song_124bpm_stems with drums_kick and drums_hh, alignment.json, sections.json."""
    d = root / "Test" / "Song" / "Song_124bpm_stems"
    d.mkdir(parents=True)
    sf.write(str(d / "drums_kick.flac"), stem([(at(i * 4), kick_hit(i % 3 - 1)) for i in range(BARS * 4)], BARS).astype(np.float32), SR)
    sf.write(str(d / "drums_hh.flac"), stem(hat_events(BARS), BARS).astype(np.float32) * 0.3, SR)
    (d / "alignment.json").write_text(json.dumps({"result": {"bpm": BPM, "downbeat_sec": T0 + 2 * BAR_S, "downbeat_verdict": "ok"}}))
    (d / "sections.json").write_text(json.dumps({"bpm": BPM, "method": "stems", "sections": [
        {"beat": 0, "bar": 1, "time_sec": T0, "duration_beats": 16, "stems": [], "label": "A", "live_bar": 1.0},
        {"beat": 16, "bar": 5, "time_sec": T0 + 4 * BAR_S, "duration_beats": 16, "stems": [], "label": "B", "live_bar": 5.0}]}))
    return d


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = make_stems(tmp_path_factory.mktemp("drums"))
    return d, drums.drum_roles(d)


def role(rec, stem_name, name):
    return next(r for r in rec["stems"][stem_name]["roles"] if r["name"] == name)


def test_each_role_gets_its_name_and_its_16_step_pattern_counted_from_the_downbeat(made):
    d, rec = made
    assert rec["grid"]["t0"] == pytest.approx(T0, abs=1e-3)           # the downbeat two bars in, moved back to the first bar line
    assert role(rec, "kick", "kick")["line"] == "x...x...x...x..."
    assert role(rec, "hh", "hh offbeat")["line"] == "..x...x...x...x."
    assert role(rec, "hh", "hh 16th")["line"] == "xx.xxx.xxx.xxx.x"
    assert role(rec, "kick", "kick")["hits"] == BARS * 4
    assert role(rec, "hh", "hh offbeat")["hits"] == BARS * 4


def test_drums_json_and_one_clip_per_role_are_written_with_the_inputs_for_staleness(made):
    d, rec = made
    saved = json.loads((d / "drums.json").read_text())
    assert (saved["kind"], saved["version"]) == ("drum_roles", 1)
    assert {i["name"]: i["size"] for i in saved["inputs"]} == {
        n: (d / n).stat().st_size for n in ("drums_kick.flac", "drums_hh.flac", "alignment.json", "sections.json")}
    for stem_name, s in saved["stems"].items():
        for r in s["roles"]:
            assert r["clip"] and (d / r["clip"]).is_file()
            audio, sr = sf.read(str(d / r["clip"]))
            assert sr == SR and len(audio) > 0
    assert (d / "drum_roles" / "kick.wav").is_file()
    assert saved["stems"]["kick"]["per_section"][0]["roles"] == {"kick": 16}


def test_no_write_prints_the_table_and_writes_nothing(tmp_path, capsys):
    d = make_stems(tmp_path)
    before = sorted(p.name for p in d.iterdir())
    assert roadie_drums.main([str(d), "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "kick" in out and "x... x... x... x..." in out
    assert sorted(p.name for p in d.iterdir()) == before


def test_the_study_reader_shows_the_drums_layer_and_flags_it(made, capsys):
    d, _ = made
    s = study.load_study(str(d.parent))
    layer = s.layers["drums"]
    assert layer.missing is None and layer.data["kind"] == "drum_roles"
    assert any(c.startswith("fresh") for c in layer.caveats)
    assert study.to_json(s)["layers"]["drums"]["data"]["stems"]["kick"]["roles"][0]["name"] == "kick"
    assert d / "drums.json" not in s.extras
    study._show(s)
    assert "x... x... x... x..." in capsys.readouterr().out

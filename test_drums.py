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
from test_analysis_drumroles import BPM, SR, STEP, T0, at, noise_hit, stem, tom_hit  # noqa: E402

BARS = 24
BAR_S = 16 * STEP


def kick_hit(db=0.0, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.05) * 10 ** (db / 20)


def hat_pattern(bar):
    """Offbeats in sections A and C (bars 0-7, 16-23), the odd 16ths in B (bars 8-15): one hat sound, two patterns."""
    return [2, 6, 10, 14] if not 8 <= bar < 16 else [1, 3, 5, 7, 9, 11, 13, 15]


def make_stems(root):
    """<root>/Test/Song/Song_124bpm_stems: drums_kick (four on the floor), drums_hh (one closed hat whose pattern changes
    in section B), drums_toms (one tom per section), alignment.json with the downbeat two bars in, sections.json (3 x 8 bars)."""
    d = root / "Test" / "Song" / "Song_124bpm_stems"
    d.mkdir(parents=True)
    wr = lambda name, ev: sf.write(str(d / name), stem(ev, BARS).astype(np.float32) * (0.3 if "hh" in name else 1.0), SR)  # noqa: E731
    wr("drums_kick.flac", [(at(i * 4), kick_hit(i % 3 - 1)) for i in range(BARS * 4)])
    wr("drums_hh.flac", [(at(b * 16 + s), noise_hit(b * 16 + s, 0.005, (s % 3 - 1))) for b in range(BARS) for s in hat_pattern(b)])
    wr("drums_toms.flac", [(at(b * 16), tom_hit()) for b in (1, 9, 17)])
    (d / "alignment.json").write_text(json.dumps({"result": {"bpm": BPM, "downbeat_sec": T0 + 2 * BAR_S, "downbeat_verdict": "ok"}}))
    (d / "sections.json").write_text(json.dumps({"bpm": BPM, "method": "stems", "sections": [
        {"beat": 32 * k, "bar": 8 * k + 1, "time_sec": T0 + 8 * k * BAR_S, "duration_beats": 32, "stems": [], "label": "ABC"[k],
         "live_bar": 8.0 * k + 1} for k in range(3)]}))
    return d


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = make_stems(tmp_path_factory.mktemp("drums"))
    return d, drums.drum_roles(d, jobs=1)


def role(rec, stem_name, name):
    return next(r for r in rec["stems"][stem_name]["roles"] if r["name"] == name)


def test_a_hat_whose_pattern_changes_between_sections_is_one_role_with_a_pattern_per_section(made):
    d, rec = made
    hh = rec["stems"]["hh"]["roles"]
    assert len(hh) == 1 and hh[0]["hits"] == 8 * 4 * 2 + 8 * 8
    lines = {v["section"]: v["line"] for v in hh[0]["per_section"]}
    assert lines == {"A": "..x...x...x...x.", "B": ".x.x.x.x.x.x.x.x", "C": "..x...x...x...x."}


def test_a_sound_heard_once_in_each_section_is_one_role(made):
    d, rec = made
    toms = rec["stems"]["toms"]["roles"]
    assert len(toms) == 1 and toms[0]["hits"] == 3 and [v["hits"] for v in toms[0]["per_section"]] == [1, 1, 1]
    assert toms[0]["line"] == "x..............."


def test_the_kick_is_named_over_all_sections_and_its_grid_is_counted_from_the_downbeat(made):
    d, rec = made
    assert rec["grid"]["t0"] == pytest.approx(T0, abs=1e-3)           # the downbeat two bars in, moved back to the first bar line
    k = role(rec, "kick", "kick")
    assert (k["line"], k["hits"]) == ("x...x...x...x...", BARS * 4)
    assert [v["hits"] for v in k["per_section"]] == [32, 32, 32]


def test_drums_json_and_one_clip_per_role_are_written_with_the_inputs_for_staleness(made):
    d, rec = made
    saved = json.loads((d / "drums.json").read_text())
    assert (saved["kind"], saved["version"]) == ("drum_roles", 1)
    assert {i["name"]: i["size"] for i in saved["inputs"]} == {
        n: (d / n).stat().st_size for n in ("drums_kick.flac", "drums_hh.flac", "drums_toms.flac", "alignment.json", "sections.json")}
    for stem_name, s in saved["stems"].items():
        for r in s["roles"]:
            assert r["clip"] is None or (d / r["clip"]).is_file()
            if r["clip"]:
                audio, sr = sf.read(str(d / r["clip"]))
                assert sr == SR and len(audio) > 0
    assert saved["stems"]["kick"]["roles"][0]["clip"] == "drum_roles/kick.wav"
    assert (d / "drum_roles" / "kick.wav").is_file()
    assert saved["stems"]["kick"]["per_section"][0]["roles"] == {"kick": 32}


def test_a_process_pool_gives_the_same_record_as_one_process(made):
    d, serial = made
    pooled = drums.drum_roles(d, write=False, jobs=2)
    drop = ("made", "seconds", "jobs", "stage_seconds", "written")
    clean = lambda r: json.loads(json.dumps({k: v for k, v in r.items() if k not in drop}))  # noqa: E731
    for r in (serial, pooled):                                     # the serial record was written, the pooled one was not
        for s in r["stems"].values():
            for x in s["roles"]:
                x["clip"], x["clip_note"] = None, None
    assert clean(pooled) == clean(serial) and pooled["jobs"] == 2


def test_no_write_prints_the_table_and_writes_nothing(tmp_path, capsys):
    d = make_stems(tmp_path)
    before = sorted(p.name for p in d.iterdir())
    assert roadie_drums.main([str(d), "--no-write", "--jobs", "1"]) == 0
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

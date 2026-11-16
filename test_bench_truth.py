"""bench_truth.py and benchmark.py's truth_main on a tiny synthetic 24-bit pack. No pack audio."""
import argparse
import json
import os

import numpy as np
import pytest
import soundfile as sf

import bench_truth as T
import benchmark as B

SR = 8000
BPM = 60          # 4 beats of 1 s: a bar is 4 s
BARS = 8


def _noise(seed, bars_on=None, level=0.05):
    """24-bit-grid noise over BARS bars, in `bars_on` only."""
    n = SR * 4
    x = np.zeros(n * BARS)
    r = np.random.default_rng(seed)
    for b in (range(BARS) if bars_on is None else bars_on):
        x[b * n:(b + 1) * n] = level * r.standard_normal(n)
    return np.round(x * 2 ** 23) / 2 ** 23


def _pack(tmp_path, extra_sources=()):
    """A pack of five .aif files and its manifest; returns (manifest dict, manifest path)."""
    pack = tmp_path / "pack"
    pack.mkdir()
    files = {"KICK.aif": (_noise(1, [1, 2, 3, 4]), "drums_kick", "kick"),
             "SNARE.aif": (_noise(2, [2, 3, 4, 5]), "drums_snare", "snare"),
             "CLAP.aif": (_noise(3, [3, 4]), "drums_snare", "clap"),
             "BASS.aif": (_noise(4, range(2, 8)), "bass", None),
             "PAD.aif": (_noise(5, range(0, 8), 0.03), "synths", None),
             "RTRN.aif": (_noise(6, [4, 5, 6], 0.01), "returns", None)}
    for name, (x, _r, _d) in files.items():
        sf.write(str(pack / name), np.stack([x, x * 0.5], axis=1), SR, subtype="PCM_24", format="AIFF")
    m = {"name": "tiny", "pack_dir": str(pack), "out_dir": str(tmp_path / "out"), "bpm": BPM, "bars": BARS,
         "beats_per_bar": 4, "bar1_s": 0.0,
         "stems": ["drums_kick", "drums_snare", "bass", "synths", "vocals"],
         "readings": {"snare family": {"stem": "drums_snare", "truth": ["snare", "clap"]},
                      "hits": {"stem": ["drums_kick", "drums_snare"], "truth": ["kick", "snare", "clap"]},
                      "synths+returns": {"stem": "synths", "truth": ["synths", "returns"]},
                      "typo": {"stem": "bass", "truth": ["bass", "no_such_file"]}},
         "excluded": [],
         "sources": [{"file": f, "source": f[:-4], "role": r, **({"drum_source": d} if d else {}), "note": ""}
                     for f, (_x, r, d) in files.items()]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(m))
    return m, str(path)


def _separate(out_dir, noise_db=-20.0, name="mix_stems"):
    """A fake separator: each true stem plus noise orthogonal to it at noise_db."""
    d = os.path.join(out_dir, "separated", name)
    os.makedirs(d)
    for stem in ("drums_kick", "drums_snare", "bass", "synths", "vocals"):
        p = os.path.join(out_dir, "truth", stem + ".flac")
        if not os.path.exists(p):
            x = np.random.default_rng(9).standard_normal((SR * 4 * BARS, 2)) * 0.01
        else:
            x = sf.read(p, dtype="float64", always_2d=True)[0]
            n = np.random.default_rng(10).standard_normal(x.shape)
            n -= (n * x).sum() / (x * x).sum() * x
            x = x + n * np.sqrt((x * x).sum() / (n * n).sum() * 10 ** (noise_db / 10))
        sf.write(os.path.join(d, stem + ".flac"), x, SR, subtype="PCM_24")
    return d


def test_build_writes_the_mix_the_role_stems_and_the_drum_sources(tmp_path):
    m, _ = _pack(tmp_path)
    T.check_manifest(m)
    roles, frames, sr, subtype, bars, peak = T.build(m, m["out_dir"])
    assert roles == ["drums_kick", "drums_snare", "bass", "synths", "returns"]   # vocals: no source, no file
    assert (frames, sr, subtype, bars) == (SR * 4 * BARS, SR, "PCM_24", 8.0)
    assert sorted(os.listdir(os.path.join(m["out_dir"], "truth"))) == sorted(r + ".flac" for r in roles)
    assert sorted(os.listdir(os.path.join(m["out_dir"], "truth_sources"))) == ["clap.flac", "kick.flac", "snare.flac"]
    res, _ = T.verify(m, m["out_dir"], roles, frames)
    assert res["roles"]["peak_dbfs"] < T.PASS_DB and res["pack"]["peak_dbfs"] < T.PASS_DB
    mix = sf.read(os.path.join(m["out_dir"], "mix.wav"), dtype="float64")[0]
    snare = sf.read(os.path.join(m["out_dir"], "truth", "drums_snare.flac"), dtype="float64")[0]
    one = sf.read(os.path.join(m["out_dir"], "truth_sources", "snare.flac"), dtype="float64")[0]
    clap = sf.read(os.path.join(m["out_dir"], "truth_sources", "clap.flac"), dtype="float64")[0]
    assert np.array_equal(snare, one + clap)             # the role is the sum of its sources, bit for bit
    assert np.abs(mix).max() == pytest.approx(peak["mix"])


def test_build_deletes_truth_files_for_roles_the_manifest_no_longer_maps(tmp_path):
    m, _ = _pack(tmp_path)
    T.build(m, m["out_dir"])
    truth = os.path.join(m["out_dir"], "truth")
    for stale in ("vocals.flac", "old_role.flac"):
        sf.write(os.path.join(truth, stale), np.zeros((100, 2)), SR, subtype="PCM_24")
    sf.write(os.path.join(m["out_dir"], "truth_sources", "gone.flac"), np.zeros((100, 2)), SR, subtype="PCM_24")
    roles, *_ = T.build(m, m["out_dir"])
    assert sorted(os.listdir(truth)) == sorted(r + ".flac" for r in roles)
    assert "gone.flac" not in os.listdir(os.path.join(m["out_dir"], "truth_sources"))


def test_sources_only_writes_just_the_drum_sources(tmp_path, monkeypatch, capsys):
    _, path = _pack(tmp_path)
    monkeypatch.setattr("sys.argv", ["bench_truth.py", path, "--sources-only"])
    T.main()
    out = tmp_path / "out"
    assert sorted(os.listdir(out)) == ["truth_sources"]
    assert "truth_sources: kick, snare, clap" in capsys.readouterr().out


def test_main_exits_zero_when_the_residuals_pass_and_says_so(tmp_path, monkeypatch, capsys):
    _, path = _pack(tmp_path)
    monkeypatch.setattr("sys.argv", ["bench_truth.py", path])
    with pytest.raises(SystemExit) as e:
        T.main()
    assert e.value.code == 0
    text = capsys.readouterr().out
    assert "6 pack files, 0 excluded" in text and "8.00 bars at 60 BPM" in text
    assert "no source, no stem written: vocals" in text and text.count("ok") == 2


def test_a_pack_of_another_length_is_refused(tmp_path):
    m, _ = _pack(tmp_path)
    m["bars"] = 9
    with pytest.raises(SystemExit):
        T.pack_format(m)


def test_a_file_the_manifest_does_not_list_is_refused(tmp_path):
    m, _ = _pack(tmp_path)
    sf.write(os.path.join(m["pack_dir"], "EXTRA.aif"), np.zeros((100, 2)), SR, subtype="PCM_24", format="AIFF")
    with pytest.raises(SystemExit):
        T.check_manifest(m)


def test_truth_main_end_to_end_scores_the_stems_and_the_readings(tmp_path, capsys):
    m, path = _pack(tmp_path)
    T.build(m, m["out_dir"])
    stems = _separate(m["out_dir"])
    B.truth_main(argparse.Namespace(truth=path, stems=None, json=None))
    out = capsys.readouterr().out
    data = json.load(open(os.path.join(m["out_dir"], "scores_separated_mix_stems.json")))
    rows = data["rows"]
    assert data["stems_dir"] == stems
    assert rows["drums_kick"]["si_sdr_db"] == pytest.approx(20.0, abs=0.5)
    assert rows["bass"]["si_sdr_db"] == pytest.approx(20.0, abs=0.5)
    assert rows["bass"]["level_gap_db"] == pytest.approx(0.0, abs=0.1)
    assert rows["vocals"]["skipped"] == "no true role"                       # a role without a source is not scored
    assert rows["synths+returns"]["truth"] == ["synths", "returns"]
    assert rows["snare family"]["truth"] == ["snare", "clap"]                # a reading resolves a drum source by name
    assert rows["hits"]["stem"] == "drums_kick+drums_snare"                  # two estimated stems, summed
    assert rows["hits"]["si_sdr_db"] > 15
    assert rows["typo"]["skipped"] == "no true file for no_such_file"        # no silent drop of a name
    k = data["bleed"]["drums_kick"]
    assert k["shares"]["drums_kick"] > 0.9 and 0.0 <= k["r2"] <= 1.0
    assert "SD act" in out and "net rem" in out and "wrote" in out


def test_truth_main_names_the_json_after_the_stems_and_keeps_two_runs_apart(tmp_path):
    m, path = _pack(tmp_path)
    T.build(m, m["out_dir"])
    _separate(m["out_dir"], name="a_stems")
    B.truth_main(argparse.Namespace(truth=path, stems=None, json=None))
    other = os.path.join(m["out_dir"], "elsewhere", "b_stems")
    os.makedirs(os.path.dirname(other))
    os.rename(os.path.join(m["out_dir"], "separated", "a_stems"), other)
    B.truth_main(argparse.Namespace(truth=path, stems=other, json=None))
    assert sorted(f for f in os.listdir(m["out_dir"]) if f.endswith(".json")) == [
        "scores_elsewhere_b_stems.json", "scores_separated_a_stems.json"]


def test_truth_main_refuses_a_role_whose_truth_file_is_missing(tmp_path):
    m, path = _pack(tmp_path)
    T.build(m, m["out_dir"])
    _separate(m["out_dir"])
    os.remove(os.path.join(m["out_dir"], "truth", "bass.flac"))
    with pytest.raises(SystemExit, match="no truth file for role bass"):
        B.truth_main(argparse.Namespace(truth=path, stems=None, json=None))


def test_truth_main_ignores_a_stale_truth_file_for_a_role_without_a_source(tmp_path):
    m, path = _pack(tmp_path)
    T.build(m, m["out_dir"])
    _separate(m["out_dir"])
    sf.write(os.path.join(m["out_dir"], "truth", "vocals.flac"), np.ones((SR * 4 * BARS, 2)) * 0.1, SR, subtype="PCM_24")
    B.truth_main(argparse.Namespace(truth=path, stems=None, json=None))
    data = json.load(open(os.path.join(m["out_dir"], "scores_separated_mix_stems.json")))
    assert data["rows"]["vocals"]["skipped"] == "no true role" and "vocals" not in data["bleed"]["bass"]["shares"]


def test_a_grid_that_starts_late_scores_the_same_stems_on_shifted_bars(tmp_path):
    """bar1_s: the same files on a grid 1 s late put the bars' edges 1 s later, and the readings change."""
    m, path = _pack(tmp_path)
    T.build(m, m["out_dir"])
    _separate(m["out_dir"])
    B.truth_main(argparse.Namespace(truth=path, stems=None, json=str(tmp_path / "a.json")))
    m["bar1_s"] = 1.0
    late = tmp_path / "late.json"
    late.write_text(json.dumps(m))
    B.truth_main(argparse.Namespace(truth=str(late), stems=None, json=str(tmp_path / "b.json")))
    a, b = (json.load(open(tmp_path / f))["rows"]["drums_kick"] for f in ("a.json", "b.json"))
    assert a["si_sdr_db"] == pytest.approx(20.0, abs=0.5) and b["si_sdr_db"] == pytest.approx(20.0, abs=0.5)
    assert a["truth_sd_active_db"] != pytest.approx(b["truth_sd_active_db"], abs=0.01)   # the per-bar levels moved with the edges

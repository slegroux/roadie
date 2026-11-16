"""bench_mitigations.py on synthetic signals. No pack audio."""
import numpy as np
import pytest
import soundfile as sf

import bench_mitigations as M

SR = 16000


def _noise(seed, n=SR * 6, level=0.1):
    return np.random.default_rng(seed).standard_normal((n, 2)) * level


def _w(path, x):
    sf.write(str(path), x, SR, subtype="PCM_24", format="FLAC")


def test_residual_of_stems_that_sum_to_the_mix_is_silent_and_a_missing_stem_is_its_level(tmp_path):
    a, b = _noise(1), _noise(2, level=0.05)
    _w(tmp_path / "mix.flac", a + b)
    _w(tmp_path / "a.flac", a)
    _w(tmp_path / "b.flac", b)
    r = M.residual(str(tmp_path / "mix.flac"), [str(tmp_path / "a.flac"), str(tmp_path / "b.flac")], str(tmp_path / "r.flac"))
    assert r["peak_dbfs"] < -130 and r["rms_db_re_mix"] < -110 and r["frame_gaps"] == [0, 0]
    r = M.residual(str(tmp_path / "mix.flac"), [str(tmp_path / "a.flac")], str(tmp_path / "r2.flac"))
    want = 10 * np.log10((b ** 2).mean() / ((a + b) ** 2).mean())
    assert r["rms_db_re_mix"] == pytest.approx(want, abs=0.05)
    got = sf.read(str(tmp_path / "r2.flac"), dtype="float64")[0]
    assert np.abs(got - b).max() < 1e-6                                          # the residual file is mix - a


def test_residual_reports_a_stem_of_another_length(tmp_path):
    a = _noise(1)
    _w(tmp_path / "mix.flac", a)
    _w(tmp_path / "a.flac", a[:-100])
    r = M.residual(str(tmp_path / "mix.flac"), [str(tmp_path / "a.flac")], str(tmp_path / "r.flac"))
    assert r["frame_gaps"] == [-100] and r["frames"] == len(a) - 100


def test_re_masked_stems_sum_to_the_mix():
    k = np.random.default_rng(3)
    stems = [_noise(i)[:SR * 2] for i in (4, 5, 6)]
    mix = sum(stems) + k.standard_normal(stems[0].shape) * 0.02                  # the mix holds something the stems do not
    out = M.wiener_segment(mix, stems)
    assert np.abs(sum(out) - mix).max() < 1e-4                                   # the shares sum to 1 in every bin
    assert all(o.shape == mix.shape for o in out)


def test_re_mask_of_perfect_stems_gives_them_back_and_cleans_a_stem_with_bleed():
    t = np.arange(SR * 2) / SR
    a = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1) * 0.3
    b = np.stack([np.sin(2 * np.pi * 3000 * t)] * 2, 1) * 0.3
    mix = a + b
    ra, rb = M.wiener_segment(mix, [a, b])
    mid = slice(SR // 4, -SR // 4)
    assert np.abs(ra - a)[mid].max() < 0.01 and np.abs(rb - b)[mid].max() < 0.01
    bleedy = a + 0.3 * b                                                        # stem a holds a third of b
    ra2, _ = M.wiener_segment(mix, [bleedy, b])
    err_before = np.abs(bleedy - a)[mid].mean()
    err_after = np.abs(ra2 - a)[mid].mean()
    assert err_after < err_before * 0.5                                          # the mask takes the bleed back out


def test_wiener_writes_stems_that_sum_to_the_mix_across_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "SEGMENT", SR)                                       # several segments in a 6 s file
    s1, s2 = _noise(1), _noise(2, level=0.05)
    mix = s1 + s2 + _noise(3, level=0.01)
    _w(tmp_path / "mix.flac", mix)
    _w(tmp_path / "bass.flac", s1)
    _w(tmp_path / "synths.flac", s2)
    r = M.wiener(str(tmp_path / "mix.flac"), {"bass": str(tmp_path / "bass.flac"), "synths": str(tmp_path / "synths.flac")}, str(tmp_path / "out"))
    out = sum(sf.read(str(tmp_path / "out" / (k + ".flac")), dtype="float64")[0] for k in ("bass", "synths"))
    assert r["frames"] == len(mix) and np.abs(out - mix).max() < 1e-4            # also across the segment joins


def test_gate_silences_the_marked_groups_hits_with_fades_and_leaves_the_rest():
    x = np.ones((1000, 1))
    res = {"hits": [{"id": 0, "group": 1, "status": "grouped", "start": 100, "end": 300},
                    {"id": 1, "group": 2, "status": "grouped", "start": 500, "end": 700},
                    {"id": 2, "group": 1, "status": "assigned", "start": 800, "end": 900},
                    {"id": 3, "group": 1, "status": "faint", "start": 950, "end": 990}]}
    y, n = M.gate_stem(x, 1000, res, [1], fade_s=0.004)                           # 4-sample fades
    assert n == 2 and y.shape == x.shape
    assert (y[104:296] == 0).all() and (y[804:896] == 0).all()                   # silent over the cut
    assert 0 < y[97, 0] < 1 and 0 < y[302, 0] < 1                                # faded in and out
    assert (y[500:700] == 1).all() and (y[950:990] == 1).all() and (y[:90] == 1).all()
    assert M.gate_stem(x, 1000, res, [])[1] == 0


def test_assemble_names_the_models_files_as_stems(tmp_path):
    m = {"out_dir": str(tmp_path)}
    (tmp_path / "kuielab_b_drums").mkdir()
    _w(tmp_path / "kuielab_b_drums" / "mix_(Drums)_kuielab_b_drums.flac", _noise(1))
    spec = M.ASSEMBLE["kuielab_b_drums"]
    import json
    (tmp_path / "m.json").write_text(json.dumps(m))
    import argparse
    M.cmd_assemble(argparse.Namespace(manifest=str(tmp_path / "m.json"), name="kuielab_b_drums"))
    assert (tmp_path / spec["into"] / "drums.flac").exists() and not (tmp_path / spec["into"] / "no_drums.flac").exists()


def test_a_sum_that_reaches_full_scale_is_refused_rather_than_clipped(tmp_path):
    a, b = _noise(1, level=0.1), _noise(2, level=0.1)
    _w(tmp_path / "a.flac", a)
    _w(tmp_path / "b.flac", b)
    sr, n = M.write_sum([str(tmp_path / "a.flac"), str(tmp_path / "b.flac")], str(tmp_path / "ok.flac"))
    assert n == len(a) and np.abs(sf.read(str(tmp_path / "ok.flac"), dtype="float64")[0] - (a + b)).max() < 1e-6
    loud = np.full((SR, 2), 0.6)
    _w(tmp_path / "l1.flac", loud)
    _w(tmp_path / "l2.flac", loud)
    with pytest.raises(ValueError, match="clip"):
        M.write_sum([str(tmp_path / "l1.flac"), str(tmp_path / "l2.flac")], str(tmp_path / "bad.flac"))
    assert not (tmp_path / "bad.flac").exists()                                  # and no clipped file is left behind


def test_gate_hits_silences_the_cut_of_each_hit_given():
    x = np.ones((1000, 1))
    hits = [{"start": 100, "end": 300}, {"start": 800, "end": 900}, {"start": 998, "end": 999}]
    y, n = M.gate_hits(x, 1000, hits, fade_s=0.004)
    assert n == 2                                                                # the 1-sample hit at the end has no room to fade
    assert (y[104:296] == 0).all() and (y[804:896] == 0).all() and (y[500:700] == 1).all()


def test_the_oracle_gate_takes_the_hits_with_no_true_onset_of_the_stems_own_sources():
    res = {"hits": [{"id": i, "onset_s": t, "status": s} for i, (t, s) in enumerate(
        [(1.00, "grouped"), (1.02, "grouped"), (1.04, "grouped"), (2.50, "assigned"), (3.00, "faint"), (3.01, "overlapped")])]}
    gone = M.hits_without_true_onset(res, [1.0, 3.0], tol=0.025)
    assert [h["id"] for h in gone] == [2, 3]                                     # 1.04 is 40 ms off, 2.50 is nowhere; faint and overlapped hits count
    assert [h["id"] for h in M.hits_without_true_onset(res, [], tol=0.025)] == list(range(6))     # a stem with no onset: all


def test_coincidence_groups_are_the_ones_that_pass_the_coincidence_test_alone():
    res = {"groups": [{"id": 0, "count": 10}, {"id": 1, "count": 3}, {"id": 2, "count": 10}, {"id": 3, "count": 10}, {"id": 4, "count": 10}]}
    ev = lambda rate, p: {"rate": rate, "p": p}                                          # noqa: E731
    bleed = {0: {"evidence": ev(0.9, 0.001)}, 1: {"evidence": ev(0.9, 0.001)}, 2: {"evidence": ev(0.5, 0.001)},
             3: {"evidence": ev(0.9, 0.2)}, 4: {"evidence": None}}
    assert M.coincidence_groups(res, bleed) == {0}              # not group 1 (3 hits), 2 (50 %), 3 (p 0.2) or 4 (nothing to coincide with)


def _manifest(tmp_path, bars=4):
    import json
    out = tmp_path / "pack"
    (out / "truth").mkdir(parents=True)
    m = {"out_dir": str(out), "bpm": 120, "bars": bars, "beats_per_bar": 4, "stems": ["bass", "synths"],
         "sources": [{"role": "bass", "file": "B.aif", "source": "B"}, {"role": "synths", "file": "S.aif", "source": "S"}]}
    (tmp_path / "m.json").write_text(json.dumps(m))
    return m, str(tmp_path / "m.json"), out


def test_the_oracle_mask_built_from_the_true_stems_gives_them_back_and_sums_to_the_mix(tmp_path):
    import argparse
    m, path, out = _manifest(tmp_path)
    t = np.arange(SR * 8) / SR
    bass = np.stack([np.sin(2 * np.pi * 110 * t)] * 2, 1) * 0.3
    synths = np.stack([np.sin(2 * np.pi * 2500 * t)] * 2, 1) * 0.2
    ret = np.stack([np.sin(2 * np.pi * 6000 * t)] * 2, 1) * 0.1                  # a role with no stem of its own: the mask holds it too
    for k, v in (("bass", bass), ("synths", synths), ("returns", ret)):
        _w(out / "truth" / (k + ".flac"), v)
    sf.write(str(out / "mix.wav"), bass + synths + ret, SR, subtype="PCM_24")
    M.cmd_oracle(argparse.Namespace(manifest=path, out=str(tmp_path / "o"), power=2.0))
    got = {k: sf.read(str(tmp_path / "o" / (k + ".flac")), dtype="float64")[0] for k in ("bass", "synths", "returns")}
    mid = slice(SR, -SR)
    assert np.abs(got["bass"] - bass)[mid].max() < 0.01 and np.abs(got["synths"] - synths)[mid].max() < 0.01
    assert np.abs(sum(got.values()) - (bass + synths + ret))[mid].max() < 1e-3


def test_bars_counts_the_active_bars_where_the_second_set_has_the_higher_sdr(tmp_path, capsys):
    import argparse
    m, path, out = _manifest(tmp_path)
    t = np.arange(SR * 8) / SR
    true = np.stack([np.sin(2 * np.pi * 220 * t)] * 2, 1) * 0.3
    _w(out / "truth" / "bass.flac", true)
    _w(out / "truth" / "synths.flac", true * 0.5)
    sf.write(str(out / "mix.wav"), true, SR, subtype="PCM_24")
    for name, noise in (("a", 0.2), ("b", 0.02)):
        (tmp_path / name).mkdir()
        _w(tmp_path / name / "bass.flac", true + _noise(7, n=SR * 8, level=noise))
    M.cmd_bars(argparse.Namespace(manifest=path, a=str(tmp_path / "a"), b=str(tmp_path / "b"), json=str(tmp_path / "bars.json")))
    import json
    got = json.load(open(tmp_path / "bars.json"))
    assert list(got) == ["bass"] and got["bass"]["active"] == 4 and got["bass"]["share"] == 1.0 and got["bass"]["median_db"] > 10
    assert "bass" in capsys.readouterr().out

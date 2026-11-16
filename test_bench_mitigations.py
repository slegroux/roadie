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

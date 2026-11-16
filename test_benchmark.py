"""Metric functions of benchmark.py's --truth mode, on synthetic signals. No pack audio."""
import numpy as np
import pytest
import soundfile as sf

import benchmark as B

SR = 8000
BAR_S = 0.5
N_BARS = 8


def _rng():
    return np.random.default_rng(7)


def _orthogonal_noise(x, rel_db):
    """Noise orthogonal to x at rel_db against it, so the SI-SDR is exact."""
    n = np.random.default_rng(99).standard_normal(len(x))
    n -= (n @ x) / (x @ x) * x
    return n * np.sqrt((x @ x) / (n @ n) * 10 ** (rel_db / 10))


def test_si_sdr_of_signal_against_itself_is_large():
    x = _rng().standard_normal(20000)
    assert B.si_sdr(x, x) > 100


def test_si_sdr_with_noise_at_minus_20_db_is_20():
    x = _rng().standard_normal(20000)
    assert B.si_sdr(x + _orthogonal_noise(x, -20), x) == pytest.approx(20.0, abs=0.01)


def test_si_sdr_ignores_gain_but_not_noise():
    x = _rng().standard_normal(20000)
    assert B.si_sdr(0.1 * x, x) >= 100
    assert B.si_sdr(x + _orthogonal_noise(x, 0), x) == pytest.approx(0.0, abs=0.01)


def test_si_sdr_of_silent_reference_is_nan():
    assert np.isnan(B.si_sdr(np.ones(100), np.zeros(100)))


def test_bleed_shares_recover_known_mix_weights():
    r = _rng()
    roles = {name: r.random((40, 5)) * scale for name, scale in
             (("a", 1.0), ("b", 10.0), ("c", 0.1))}
    stem = 0.7 * roles["a"] + 0.2 * roles["b"]
    shares, rest, fit = B.bleed_shares(stem, roles)
    total = stem.sum()
    assert shares["a"] == pytest.approx(0.7 * roles["a"].sum() / total, abs=1e-6)
    assert shares["b"] == pytest.approx(0.2 * roles["b"].sum() / total, abs=1e-6)
    assert shares["c"] == pytest.approx(0.0, abs=1e-6)
    assert rest == pytest.approx(0.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9) and fit["at_bound"] == 0


def test_bleed_shares_leave_unexplained_energy_in_the_remainder():
    r = _rng()
    a, other = r.random((40, 5)), r.random((40, 5))
    a[20:] = 0.0                                       # a plays in the first half only
    other[:20] = 0.0                                   # the unknown source in the second
    shares, rest, fit = B.bleed_shares(a + other, {"a": a})
    assert shares["a"] == pytest.approx(a.sum() / (a + other).sum(), abs=1e-6)
    assert rest == pytest.approx(other.sum() / (a + other).sum(), abs=1e-6)
    assert fit["r2"] < 0.9                             # the fit does not follow the stem: R2 says so, the remainder alone does not


def test_bleed_weights_are_bounded_and_the_bound_is_counted():
    r = _rng()
    roles = {"a": r.random((40, 3)) + 0.1}
    stem = 50.0 * roles["a"]                           # a stem 17 dB louder than its only source
    shares, rest, fit = B.bleed_shares(stem, roles)
    assert shares["a"] == pytest.approx(B.BLEED_W_MAX * roles["a"].sum() / stem.sum(), abs=1e-6)
    assert fit["at_bound"] == 3                        # one per band
    assert rest == pytest.approx(1 - B.BLEED_W_MAX / 50.0, abs=1e-6)
    loose = B.bleed_shares(stem, roles, w_max=100.0)[0]["a"]
    assert loose == pytest.approx(1.0, abs=1e-6)       # unbounded, the fit explains it all


def test_bleed_shares_of_a_silent_stem_is_nan():
    shares, rest, fit = B.bleed_shares(np.zeros((10, 2)), {"a": np.ones((10, 2))})
    assert np.isnan(shares["a"]) and np.isnan(rest) and np.isnan(fit["r2"])


def test_pearson_and_active_bars():
    assert B.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(B.pearson([1, 1, 1], [1, 2, 3]))
    db = np.array([-120, -120, -10, -12, -10, -45, -10.0])
    assert B.active_bars(db).tolist() == [False, False, True, True, True, False, True]


def _write(path, x):
    sf.write(path, np.stack([x, x], axis=1), SR, subtype="PCM_24")


def _true_role(bars_on):
    """Noise in the listed bars, silence elsewhere."""
    x = np.zeros(int(SR * BAR_S * N_BARS))
    n = int(SR * BAR_S)
    r = _rng()
    for b in bars_on:
        x[b * n:(b + 1) * n] = 0.08 * r.standard_normal(n)
    return x


def test_score_stem_on_a_stem_with_known_noise_and_bleed(tmp_path):
    truth = _true_role([2, 3, 4, 5])
    on = truth != 0                                      # noise only where the role plays
    est = truth.copy()
    est[on] += _orthogonal_noise(truth[on], -20)
    _write(tmp_path / "t.wav", truth)
    _write(tmp_path / "e.wav", est)
    edges = B.bar_edges(N_BARS, BAR_S, SR)
    row, est_db, ref_db = B.score_stem([str(tmp_path / "e.wav")], [str(tmp_path / "t.wav")], edges)
    assert row["si_sdr_db"] == pytest.approx(20.0, abs=0.5)
    assert row["si_sdr_active_median_db"] == pytest.approx(20.0, abs=1.5)
    assert row["level_gap_db"] == pytest.approx(0.0, abs=0.1)
    assert row["corr_db"] > 0.99
    assert row["bars_active_true"] == 4 and row["bars_active_agree"] == N_BARS


def test_score_stem_level_gap_and_a_summed_truth(tmp_path):
    a, b = _true_role([0, 1, 2]), _true_role([4, 5])
    b = b * 0.5
    _write(tmp_path / "a.wav", a)
    _write(tmp_path / "b.wav", b)
    _write(tmp_path / "e.wav", 0.5 * (a + b))          # 6 dB down on the summed truth
    edges = B.bar_edges(N_BARS, BAR_S, SR)
    row, _, _ = B.score_stem([str(tmp_path / "e.wav")], [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")], edges)
    assert row["level_gap_db"] == pytest.approx(-6.02, abs=0.05)
    assert row["level_gap_active_median_db"] == pytest.approx(-6.02, abs=0.05)
    assert row["si_sdr_db"] > 100


def test_band_energy_files_a_tone_under_its_bar_and_band(tmp_path):
    n = int(SR * BAR_S)
    x = np.zeros(n * N_BARS)
    t = np.arange(n) / SR
    x[2 * n:3 * n] = np.sin(2 * np.pi * 1000 * t)       # bar 2, 1 kHz: the 500-2000 Hz band
    _write(tmp_path / "x.wav", x)
    bands = (20, 120, 500, 2000, 3900)                  # SR/2 = 4 kHz
    e = B.band_energy([str(tmp_path / "x.wav")], B.bar_edges(N_BARS, BAR_S, SR), SR, bands)
    assert e.shape == (N_BARS, 4)
    assert np.unravel_index(e.argmax(), e.shape) == (2, 2)
    assert e[2, 2] > 100 * np.delete(e, 2, axis=0).max()


def test_score_stem_reports_rho_and_a_level_that_wanders(tmp_path):
    """Per-bar levels: the truth steps through 6 dB changes, the stem follows with a known error."""
    levels = np.array([0, -6, -12, -3, -9, 0, -6, -12.0])        # the truth's per-bar level, dB
    wander = np.array([0, 2, -2, 0, 2, -2, 0, 0.0])              # the stem's own error, dB
    n = int(SR * BAR_S)
    base = _rng().standard_normal(n * N_BARS)
    ref = np.concatenate([base[b * n:(b + 1) * n] * 10 ** (levels[b] / 20) * 0.05 for b in range(N_BARS)])
    est = np.concatenate([base[b * n:(b + 1) * n] * 10 ** ((levels[b] + wander[b] - 3) / 20) * 0.05
                          for b in range(N_BARS)])
    _write(tmp_path / "r.wav", ref)
    _write(tmp_path / "e.wav", est)
    row, _, _ = B.score_stem([str(tmp_path / "e.wav")], [str(tmp_path / "r.wav")], B.bar_edges(N_BARS, BAR_S, SR))
    assert row["level_gap_active_median_db"] == pytest.approx(-3.0 + np.median(wander), abs=0.1)
    assert row["truth_sd_active_db"] == pytest.approx(levels.std(), abs=0.05)
    assert row["track_err_active_db"] == pytest.approx(np.sqrt(np.mean((wander - np.median(wander)) ** 2)), abs=0.1)
    assert row["corr_db_active"] == pytest.approx(np.corrcoef(levels, levels + wander)[0, 1], abs=0.01)
    assert 0.9 < row["rho"] <= 1 and row["si_sdr_db"] > 5      # the same noise with a gain per bar


def test_a_flat_truth_blanks_the_active_bar_correlation(tmp_path):
    n = int(SR * BAR_S)
    rng = _rng()
    ref = np.zeros(n * N_BARS)
    for b in range(N_BARS):
        ref[b * n:(b + 1) * n] = rng.standard_normal(n) * 0.05 * 10 ** (rng.uniform(-0.5, 0.5) / 20)   # +-0.5 dB
    est = ref * 10 ** (rng.uniform(-3, 3, N_BARS).repeat(n) / 20)                                     # +-3 dB wander
    _write(tmp_path / "r.wav", ref)
    _write(tmp_path / "e.wav", est)
    row, _, _ = B.score_stem([str(tmp_path / "e.wav")], [str(tmp_path / "r.wav")], B.bar_edges(N_BARS, BAR_S, SR))
    assert row["truth_sd_active_db"] < B.FLAT_SD_DB
    assert np.isnan(row["corr_db_active"]) and not np.isnan(row["corr_db_active_raw"])
    assert row["track_err_active_db"] > row["truth_sd_active_db"]      # the stem wanders more than the truth does


def test_uncorrelated_signals_read_below_the_floor_and_rho_near_zero(tmp_path):
    a = np.random.default_rng(1).standard_normal(SR * 8) * 0.1
    b = np.random.default_rng(2).standard_normal(SR * 8) * 0.1
    _write(tmp_path / "a.wav", a)
    _write(tmp_path / "b.wav", b)
    row, _, _ = B.score_stem([str(tmp_path / "a.wav")], [str(tmp_path / "b.wav")], B.bar_edges(N_BARS, BAR_S, SR))
    assert abs(row["rho"]) < 0.02 and row["si_sdr_db"] < B.UNCORRELATED_DB
    assert B.fmt_sdr(row["si_sdr_db"]) == "<-30" and B.fmt_sdr(6.37) == "6.4" and B.fmt_sdr(float("nan")) == "-"
    assert row["si_sdr_db"] == pytest.approx(10 * np.log10(row["rho"] ** 2 / (1 - row["rho"] ** 2)), abs=0.01)


def test_a_bar_grid_that_starts_late_moves_the_bars(tmp_path):
    n = int(SR * BAR_S)
    x = np.zeros(n * N_BARS + n // 2)
    x[n // 2 + 2 * n:n // 2 + 3 * n] = 0.1 * _rng().standard_normal(n)      # the 3rd bar of a grid that starts at 0.25 s
    _write(tmp_path / "x.wav", x)
    edges = B.bar_edges(N_BARS, BAR_S, SR, bar1_s=0.25)
    assert edges[0] == int(0.25 * SR) and np.diff(edges).tolist() == [n] * N_BARS
    _, _, ref_db = B.score_stem([str(tmp_path / "x.wav")], [str(tmp_path / "x.wav")], edges)
    assert ref_db.argmax() == 2 and ref_db[2] > -40 and ref_db[[0, 1, 3]].max() < -100
    _, _, wrong = B.score_stem([str(tmp_path / "x.wav")], [str(tmp_path / "x.wav")], B.bar_edges(N_BARS, BAR_S, SR))
    assert wrong[2] < ref_db[2] - 3                                        # the same audio read on the wrong grid splits across two bars


def test_pick_stems_dir_takes_one_and_refuses_none_or_several(tmp_path):
    sep = tmp_path / "separated"
    sep.mkdir()
    with pytest.raises(SystemExit):
        B.pick_stems_dir(str(tmp_path), None)                 # none
    (sep / "a_stems").mkdir()
    assert B.pick_stems_dir(str(tmp_path), None) == str(sep / "a_stems")
    (sep / "b_stems").mkdir()
    with pytest.raises(SystemExit):
        B.pick_stems_dir(str(tmp_path), None)                 # several: no silent first
    assert B.pick_stems_dir(str(tmp_path), "x") == "x"


def test_corpus_mode_functions_run_without_numpy():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.modules['numpy'] = None\nimport benchmark\nprint(benchmark.sample_offsets(100, 3, 10))"],
                       capture_output=True, text=True, cwd=__import__("os").path.dirname(__file__) or ".")
    assert r.returncode == 0, r.stderr


def test_resolve_truth_takes_a_role_or_a_source_and_refuses_a_name_that_is_both():
    roles, sources = {"drums_kick": "k.flac", "bass": "b.flac"}, {"kick": "ks.flac", "bass": "bs.flac"}
    assert B.resolve_truth("drums_kick", roles, sources) == "k.flac"
    assert B.resolve_truth("kick", roles, sources) == "ks.flac"
    assert B.resolve_truth("nothing", roles, sources) is None
    with pytest.raises(ValueError, match="bass"):
        B.resolve_truth("bass", roles, sources)                                  # the role would have won without a word


def test_scored_readings_lists_each_stem_against_its_role_then_the_readings():
    m = {"stems": ["bass", "vocals"], "readings": {"mix": {"stem": ["bass", "vocals"], "truth": ["bass"]}, "one": {"stem": "bass", "truth": ["bass"]}}}
    assert B.scored_readings(m, {"bass": "b.flac"}) == [("bass", ["bass"], ["bass"]), ("vocals", ["vocals"], []),
                                                         ("mix", ["bass", "vocals"], ["bass"]), ("one", ["bass"], ["bass"])]

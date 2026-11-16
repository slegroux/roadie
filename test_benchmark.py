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
    shares, rest = B.bleed_shares(stem, roles)
    total = stem.sum()
    assert shares["a"] == pytest.approx(0.7 * roles["a"].sum() / total, abs=1e-6)
    assert shares["b"] == pytest.approx(0.2 * roles["b"].sum() / total, abs=1e-6)
    assert shares["c"] == pytest.approx(0.0, abs=1e-6)
    assert rest == pytest.approx(0.0, abs=1e-6)


def test_bleed_shares_leave_unexplained_energy_in_the_remainder():
    r = _rng()
    a, other = r.random((40, 5)), r.random((40, 5))
    a[20:] = 0.0                                       # a plays in the first half only
    other[:20] = 0.0                                   # the unknown source in the second
    shares, rest = B.bleed_shares(a + other, {"a": a})
    assert shares["a"] == pytest.approx(a.sum() / (a + other).sum(), abs=1e-6)
    assert rest == pytest.approx(other.sum() / (a + other).sum(), abs=1e-6)


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

"""roadie_analysis.oneshot on synthetic hits whose answers are known without the code.

Copied from Sideman's tests/test_sample.py (main bc9ba08); only the imports differ, and the two tests that went
through Sideman's MCP server (`mcp_server.server`: out-of-range hit, clean hit through the tools) call
`oneshot.profile` and `oneshot.compare` directly instead: the tool wrapper stays in Sideman until its item 5,
and Roadie's own server (item 3) gets those assertions back.
"""
import json
import math
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from roadie_analysis import oneshot  # noqa: E402

SR = 48000


def hit(freq, tau, seconds=0.6, partials=None, phase=0.0, attack=0.0, sr=SR):
    """A decaying tone, mono [n, 1]. `freq` is Hz or a function of time (s) -> Hz;
    `partials` maps harmonic number -> amplitude; `tau` is the amplitude time
    constant in s; `attack` a linear ramp in s."""
    t = np.arange(int(seconds * sr)) / sr
    f = freq(t) if callable(freq) else np.full(len(t), float(freq))
    ph = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f[:-1]) / sr])
    x = sum(a * np.sin(k * ph + phase) for k, a in (partials or {1: 1.0}).items())
    x = x * np.exp(-t / tau)
    if attack:
        x = x * np.minimum(1.0, t / attack)
    return x[:, None]


def st(a, b):
    return 12 * math.log2(a / b)


# 200 Hz held for its first 10 ms, then a raised-cosine glide down to 100 Hz by
# 40 ms: the shared test hit for the invariance and compare tests. The plan's own
# sweep case (no hold) is test_pitch_sweep_from_200_to_100_hz.
def sweep(t):
    return np.where(t < 0.010, 200.0,
                    np.where(t < 0.040, 150.0 + 50.0 * np.cos(np.pi * (t - 0.010) / 0.030), 100.0))


def glide(t):
    """200 Hz falling linearly to 100 Hz over the first 30 ms, from the onset on."""
    return np.where(t < 0.030, 200.0 - 100.0 * t / 0.030, 100.0)


# ---------------------------------------------------------------- envelope

def test_envelope_falls_of_a_50ms_time_constant():
    # amplitude e^(-t/tau): it is X dB down after tau * ln(10^(X/20))
    r = oneshot.profile_array(hit(100, 0.050), SR)
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)          # 115.1
    assert r["fall_10_ms"] == pytest.approx(50 * math.log(10 ** 0.5), abs=6)   # 57.6
    assert r["fall_40_ms"] == pytest.approx(50 * math.log(100), abs=6)         # 230.3
    assert r["duration_ms"] == pytest.approx(50 * math.log(1000), abs=10)      # 345.4


def test_envelope_attack_and_peak():
    # a cosine starting at its crest: the peak sample is the amplitude at t = 0
    r = oneshot.profile_array(0.5 * hit(100, 0.1, phase=math.pi / 2), SR)
    assert r["peak_db"] == pytest.approx(20 * math.log10(0.5), abs=0.01)
    assert r["onset_ms"] == 0.0
    assert r["attack_ms"] <= 5.0                       # within half a period of 100 Hz
    # a 10 ms linear attack on a slow decay peaks at 10 ms; 200 Hz -> within 2.5 ms
    r = oneshot.profile_array(hit(200, 0.15, attack=0.010), SR)
    assert r["attack_ms"] == pytest.approx(10.0, abs=2.5)


def test_envelope_and_pitch_ignore_leading_silence():
    x = hit(sweep, 0.15)
    a = oneshot.profile_array(x, SR)
    b = oneshot.profile_array(np.concatenate([np.zeros((int(0.030 * SR), 1)), x]), SR)
    assert b["onset_ms"] == 30.0 and a["onset_ms"] == 0.0
    for k in ("onset_ms", "onsets_ms"):
        a.pop(k), b.pop(k)
    assert a == b


def test_envelope_and_pitch_ignore_level():
    x = hit(sweep, 0.15, partials={1: 1.0, 2: 0.3, 3: 0.2})
    a, b = oneshot.profile_array(x, SR), oneshot.profile_array(x * 10 ** (-6 / 20), SR)
    assert b["peak_db"] == pytest.approx(a["peak_db"] - 6.0, abs=0.01)
    for k in ("attack_ms", "fall_10_ms", "fall_20_ms", "fall_40_ms", "pitch_start_hz",
              "pitch_body_hz", "pitch_drop_st", "pitch_settle_ms", "click_db"):
        assert b[k] == pytest.approx(a[k], abs=0.15), k
    for k in a["harmonics_db"]:
        assert b["harmonics_db"][k] == pytest.approx(a["harmonics_db"][k], abs=0.15)


def test_envelope_and_pitch_ignore_sample_rate():
    # the same hit synthesised at 44.1 and 48 kHz
    a = oneshot.profile_array(hit(sweep, 0.05, partials={1: 1.0, 3: 0.3}, sr=48000), 48000)
    b = oneshot.profile_array(hit(sweep, 0.05, partials={1: 1.0, 3: 0.3}, sr=44100), 44100)
    for k in ("attack_ms", "fall_10_ms", "fall_20_ms", "fall_40_ms", "duration_ms", "pitch_settle_ms"):
        assert b[k] == pytest.approx(a[k], abs=2), k
    assert b["pitch_body_hz"] == pytest.approx(a["pitch_body_hz"], rel=0.002)
    assert b["pitch_start_hz"] == pytest.approx(a["pitch_start_hz"], rel=0.01)
    assert b["pitch_drop_st"] == pytest.approx(a["pitch_drop_st"], abs=0.1)
    assert b["harmonics_db"][3] == pytest.approx(a["harmonics_db"][3], abs=0.5)
    assert a["harmonics_db"][2] < -40 and b["harmonics_db"][2] < -40      # none there: both read the floor
    for part in ("attack", "body"):
        assert b["noise_db"][part] == pytest.approx(a["noise_db"][part], abs=1.5), part
    assert b["click_db"] == pytest.approx(a["click_db"], abs=1.5)
    for band in ("sub", "bass", "low_mid", "mid"):
        assert b["bands_db"][band] == pytest.approx(a["bands_db"][band], abs=0.5), band
    # a point the track cannot read is None on both sides; the others agree. (pytest.approx on the whole dict
    # relied on older versions letting None through; 9.2 raises TypeError for None with a relative tolerance.)
    assert {k for k, v in b["pitch_track_hz"].items() if v is None} == {k for k, v in a["pitch_track_hz"].items() if v is None}
    assert {k: v for k, v in b["pitch_track_hz"].items() if v is not None} == pytest.approx(
        {k: v for k, v in a["pitch_track_hz"].items() if v is not None}, rel=0.02)


def test_envelope_cut_short_is_none_with_a_reason():
    # 100 ms of a 50 ms time constant only falls 17.4 dB before the file ends
    r = oneshot.profile_array(hit(100, 0.050, seconds=0.1), SR)
    assert r["fall_10_ms"] is not None
    assert r["fall_20_ms"] is None and r["fall_40_ms"] is None
    assert any("20 dB" in n for n in r["notes"]) and any("40 dB" in n for n in r["notes"])


def test_envelope_silent_or_too_short_is_refused():
    with pytest.raises(ValueError, match="silent"):
        oneshot.profile_array(np.zeros((SR, 1)), SR)
    with pytest.raises(ValueError, match="shorter"):
        oneshot.profile_array(hit(100, 0.05, seconds=0.002), SR)


# ---------------------------------------------------------------- pitch

def test_pitch_sweep_from_200_to_100_hz():
    r = oneshot.profile_array(hit(glide, 0.15), SR)
    assert r["pitched"] is True
    assert r["pitch_start_hz"] == pytest.approx(200, abs=6)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=2)
    assert r["pitch_drop_st"] == pytest.approx(12, abs=0.5)
    # within 0.5 st of 100 Hz once 200 - 100 t / 30 ms <= 100 * 2^(0.5/12); a 4-period
    # (40 ms) frame centred on t sees up to 20 ms later
    settle = 30 * (2 - 2 ** (0.5 / 12))                                    # 29.1 ms
    assert settle <= r["pitch_settle_ms"] <= settle + 20
    track = r["pitch_track_hz"]
    for ms in (0, 5, 10, 20):                                               # 200, 183.3, 166.7, 133.3 Hz
        assert track[ms] == pytest.approx(200 - 100 * ms / 30, abs=6), ms
    assert track[80] == pytest.approx(100, abs=1) and track[150] == pytest.approx(100, abs=1)


def test_pitch_kick_glide_from_160_to_45_hz():
    # 45 + 115 e^(-t / 12 ms): 160 Hz at the onset, 120.8 at 5 ms, 95.0 at 10 ms,
    # a drop of 12 log2(160 / 45) = 21.96 st to its 45 Hz body
    def f(t):
        return 45 + 115 * np.exp(-np.asarray(t) / 0.012)
    r = oneshot.profile_array(hit(f, 0.12), SR)
    assert r["pitch_start_hz"] == pytest.approx(160, rel=0.10)
    assert r["pitch_drop_st"] == pytest.approx(12 * math.log2(160 / 45), abs=2)
    assert r["pitch_track_hz"][5] == pytest.approx(float(f(0.005)), rel=0.05)
    assert r["pitch_track_hz"][10] == pytest.approx(float(f(0.010)), rel=0.05)


def test_pitch_start_of_a_steady_sawtooth():
    # harmonics at 1/k: zero crossings still come twice a period, and none of them
    # read anything but 80 Hz
    r = oneshot.profile_array(hit(80, 0.2, partials={k: 1.0 / k for k in range(1, 60)}), SR)
    assert r["pitch_start_hz"] == pytest.approx(80, rel=0.02)
    assert all(v == pytest.approx(80, rel=0.02) for v in r["pitch_track_hz"].values() if v is not None)


def test_pitch_track_before_the_first_frame_is_none_not_a_copy():
    # sin + 2 sin(3x) crosses zero six times a period, so the crossings give no pitch:
    # the points before the first frame (4 periods long, centred 2 periods = 20 ms in)
    # are None with a note, not its value
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 1.0, 3: 2.0}), SR)
    assert r["pitch_start_at_ms"] == pytest.approx(20, abs=1)
    assert [r["pitch_track_hz"][ms] for ms in (0, 5, 10)] == [None, None, None]
    assert any("no value before 20 ms" in n for n in r["notes"])


def test_pitch_that_never_holds_still_has_no_settle_time():
    # 300 e^(-t / 100 ms) falls 0.5 st in 2.9 ms: it never stays within 0.5 st of its
    # body pitch for 10 ms
    r = oneshot.profile_array(hit(lambda t: 300 * np.exp(-t / 0.1), 0.15), SR)
    assert r["pitch_settle_ms"] is None
    assert any("never settles" in n or "not followed" in n for n in r["notes"])


def test_pitch_fast_glide_reads_the_pitch_where_it_was_measured():
    # 100 + 100 e^(-t / 15 ms): its first cycle is shorter than any pitch frame; the
    # zero crossings read it, and the start is the pitch at pitch_start_at_ms
    def f(t):
        return 100 + 100 * np.exp(-np.asarray(t) / 0.015)
    r = oneshot.profile_array(hit(f, 0.15), SR)
    assert r["pitch_start_at_ms"] <= 0.5 * 1000 / r["pitch_start_hz"]      # within the first half-period
    assert r["pitch_start_hz"] == pytest.approx(float(f(r["pitch_start_at_ms"] / 1000)), rel=0.03)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=2)
    # within 0.5 st of 100 Hz once 100 e^(-t/15) <= 100 (2^(0.5/12) - 1)
    assert r["pitch_settle_ms"] == pytest.approx(-15 * math.log(2 ** (0.5 / 12) - 1), abs=5)   # 52.9


@pytest.mark.parametrize("f0", [55.0, 100.0, 300.0])
def test_pitch_is_the_fundamental_not_the_loudest_partial(f0):
    # 3rd harmonic at twice the fundamental's amplitude: 6 dB above it
    r = oneshot.profile_array(hit(f0, 0.2, partials={1: 1.0, 3: 2.0}), SR)
    assert r["pitched"] is True
    assert r["pitch_body_hz"] == pytest.approx(f0, rel=0.02)
    assert r["pitch_start_hz"] == pytest.approx(f0, rel=0.02)
    assert all(v == pytest.approx(f0, rel=0.02) for v in r["pitch_track_hz"].values() if v is not None)
    assert r["harmonics_db"][3] == pytest.approx(20 * math.log10(2), abs=1)


def test_pitch_steady_tone_has_no_drop():
    r = oneshot.profile_array(hit(100, 0.1), SR)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=0.5)
    assert r["pitch_drop_st"] == pytest.approx(0, abs=0.1)
    assert r["pitch_settle_ms"] == 0.0


# ---------------------------------------------------------------- tone

def test_harmonic_levels_of_odd_partials_at_1_over_n():
    # a square-like tone: odd harmonics at 1/n, so the nth is 20 log10(1/n) dB
    r = oneshot.profile_array(hit(100, 0.15, partials={n: 1.0 / n for n in range(1, 16, 2)}), SR)
    h = r["harmonics_db"]
    assert h[3] == pytest.approx(20 * math.log10(1 / 3), abs=1)      # -9.5
    assert h[5] == pytest.approx(20 * math.log10(1 / 5), abs=1)      # -14.0
    assert h[7] == pytest.approx(20 * math.log10(1 / 7), abs=1)      # -16.9
    assert h[2] < -40 and h[4] < -40
    assert r["partials"] == []


def test_harmonic_series_leaves_out_an_inharmonic_partial():
    t = np.arange(int(0.6 * SR)) / SR
    x = (np.sin(2 * np.pi * 100 * t) + 10 ** (-10 / 20) * np.sin(2 * np.pi * 237 * t)) * np.exp(-t / 0.15)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.01)
    assert len(r["partials"]) == 1
    assert r["partials"][0]["hz"] == pytest.approx(237, abs=1)
    assert r["partials"][0]["db"] == pytest.approx(-10, abs=1)
    assert r["harmonics_db"][2] < -40


def test_noise_share_of_a_sine_with_white_noise_20db_down():
    # both share one envelope, so the noise is 1/101 of the power: -20.04 dB
    t = np.arange(int(0.6 * SR)) / SR
    noise = np.random.default_rng(3).standard_normal(len(t)) * math.sqrt(0.5 / 100)
    x = (np.sin(2 * np.pi * 150 * t) + noise) * np.exp(-t / 0.1)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is True
    assert r["noise_db"]["body"] == pytest.approx(-20.04, abs=1.5)
    assert r["noise_db"]["attack"] == pytest.approx(-20.04, abs=1.5)


def test_noise_alone_is_unpitched_with_no_harmonic_values():
    t = np.arange(int(0.6 * SR)) / SR
    x = np.random.default_rng(4).standard_normal(len(t)) * np.exp(-t / 0.1) * 0.3
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is False
    assert r["harmonics_db"] is None
    for k in ("pitch_start_hz", "pitch_body_hz", "pitch_drop_st", "pitch_settle_ms", "pitch_track_hz"):
        assert r[k] is None, k
    assert r["noise_db"]["body"] > -1.0
    # a white-noise periodogram is exponentially distributed: geometric / arithmetic
    # mean = e^(-Euler's gamma) = 0.561
    assert r["flatness"]["body"] == pytest.approx(math.exp(-0.5772156649), abs=0.05)
    assert any("no tonal peak in the body" in n for n in r["notes"])


def test_noise_flatness_of_a_pure_tone_is_near_zero():
    r = oneshot.profile_array(hit(200, 0.1), SR)
    assert r["flatness"]["attack"] < 0.01 and r["flatness"]["body"] < 0.01
    assert r["noise_db"]["body"] < -25


def test_click_level_of_a_01ms_and_a_2ms_attack():
    # a tone starting at its crest: 0.1 ms to full level is a step, 2 ms is a ramp
    sharp = oneshot.profile_array(hit(100, 0.15, phase=math.pi / 2, attack=0.0001), SR)
    soft = oneshot.profile_array(hit(100, 0.15, phase=math.pi / 2, attack=0.002), SR)
    assert sharp["click_db"] >= soft["click_db"] + 10


def test_bands_of_a_1khz_hit_are_all_mid():
    r = oneshot.profile_array(hit(1000, 0.1), SR)
    assert r["bands_db"]["mid"] == pytest.approx(0.0, abs=0.1)
    assert all(v < -30 for k, v in r["bands_db"].items() if k != "mid")


def test_width_of_mono_identical_and_antiphase_channels():
    x = hit(200, 0.1)
    assert oneshot.profile_array(x, SR)["width"] == 0.0
    assert oneshot.profile_array(np.hstack([x, x]), SR)["width"] == pytest.approx(0.0, abs=1e-3)
    assert oneshot.profile_array(np.hstack([x, -x]), SR)["width"] == pytest.approx(1.0, abs=1e-3)


def test_profile_reads_a_file(tmp_path):
    p = tmp_path / "tom.wav"
    sf.write(str(p), hit(100, 0.05), SR, subtype="FLOAT")
    r = oneshot.profile(p)
    assert r["file"] == str(p) and r["sample_rate"] == SR and r["channels"] == 1
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)


# ---------------------------------------------------------------- compare
# The base hit uses the sweep above (200 Hz held 10 ms, then a glide to 100 Hz),
# so pitch drop and settle are compared too.

BASE = {1: 1.0, 2: 0.3, 3: 0.2}


def up2(t):
    return sweep(t) * 2 ** (2 / 12)


def lowpassed_noise(n, hz, seed):
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / SR) >= hz] = 0
    v = np.fft.irfft(spec, n)
    return v / np.sqrt((v ** 2).mean())


def with_noise(x, noise):
    # the same envelope as the tone; the tone's power is 0.5 * (1 + 0.09 + 0.04),
    # so noise at 1/100 of it sits 20 dB down: a noise share of 1/101 = -20.04 dB
    t = np.arange(len(x)) / SR
    return x + (noise * np.exp(-t / 0.1) * math.sqrt(0.5 * 1.13 / 100))[:, None]


def test_compare_level_silence_and_sample_rate_give_no_reading():
    x = hit(sweep, 0.1, partials=BASE)
    ref = oneshot.profile_array(x, SR)
    quieter = oneshot.compare(oneshot.profile_array(x * 10 ** (-6 / 20), SR), ref)
    assert quieter["readings"] == []
    assert quieter["diff"]["peak_db"] == pytest.approx(-6.0, abs=0.01)
    late = oneshot.profile_array(np.concatenate([np.zeros((int(0.030 * SR), 1)), x]), SR)
    assert oneshot.compare(late, ref)["readings"] == []
    # the same hit synthesised at 44.1 kHz, not resampled
    at44 = oneshot.profile_array(hit(sweep, 0.1, partials=BASE, sr=44100), 44100)
    assert oneshot.compare(at44, ref)["readings"] == []


def test_compare_two_semitones_up_reads_only_the_pitch():
    a = oneshot.profile_array(hit(sweep, 0.1, partials=BASE), SR)
    b = oneshot.profile_array(hit(up2, 0.1, partials=BASE), SR)
    c = oneshot.compare(a, b)
    assert c["diff"]["pitch_body_st"] == pytest.approx(-2.0, abs=0.1)
    assert len(c["readings"]) == 1 and c["readings"][0].startswith("body pitch 2.0 st below the reference")
    # a profile read back from JSON (string keys) compares the same
    assert oneshot.compare(json.loads(json.dumps(a)), json.loads(json.dumps(b)))["readings"] == c["readings"]


def test_compare_half_the_decay_reads_only_decay():
    a = oneshot.profile_array(hit(sweep, 0.05, partials=BASE), SR)
    b = oneshot.profile_array(hit(sweep, 0.1, partials=BASE), SR)
    c = oneshot.compare(a, b)
    assert c["readings"] and all(r.startswith(("falls", "lasts")) for r in c["readings"]), c["readings"]
    assert any(r.startswith("falls 20 dB") and "shorter" in r for r in c["readings"])
    # an exponential's fall times scale with its time constant: 50 / 100 ms
    for k in ("fall_10_ms", "fall_20_ms", "fall_40_ms"):
        assert c["diff"][k]["ratio"] == pytest.approx(0.5, abs=0.05), k


def test_compare_noise_added_reads_only_the_noise():
    # noise kept under 2 kHz: white noise also brightens the attack, which
    # click_db rightly reads (next test), so this isolates the noise share
    x = hit(sweep, 0.1, partials=BASE)
    noisy = oneshot.profile_array(with_noise(x, lowpassed_noise(len(x), 2000, 7)), SR)
    c = oneshot.compare(noisy, oneshot.profile_array(x, SR))
    assert noisy["noise_db"]["body"] == pytest.approx(-20.04, abs=1.5)
    assert len(c["readings"]) == 1, c["readings"]
    assert c["readings"][0].startswith("noise") and "more than the reference in the body" in c["readings"][0]


def test_compare_white_noise_reads_noise_and_a_brighter_attack():
    x = hit(sweep, 0.1, partials=BASE)
    white = np.random.default_rng(8).standard_normal(len(x))
    c = oneshot.compare(oneshot.profile_array(with_noise(x, white), SR), oneshot.profile_array(x, SR))
    kinds = sorted(r.split()[0] for r in c["readings"])
    assert kinds == ["click", "noise"], c["readings"]       # no harmonic reading from the noise floor


def test_compare_pitched_against_unpitched_is_a_reading():
    t = np.arange(int(0.6 * SR)) / SR
    noise = (np.random.default_rng(9).standard_normal(len(t)) * np.exp(-t / 0.1) * 0.3)[:, None]
    c = oneshot.compare(oneshot.profile_array(hit(100, 0.1), SR), oneshot.profile_array(noise, SR))
    assert c["readings"][0].startswith("pitched (100 Hz), the reference is not")
    assert c["diff"]["pitch_body_st"] is None and c["diff"]["harmonics_db"] is None


def test_compare_floors_none_and_order():
    a = {"pitched": True, "pitch_body_hz": 100.0 * 2 ** (0.6 / 12), "attack_ms": 0.4, "fall_20_ms": 100.0,
         "fall_40_ms": None, "pitch_settle_ms": 0.0, "harmonics_db": {2: -55.0, 3: -10.0},
         "harmonics_floor_db": {2: -90.0, 3: -90.0}, "noise_db": {"attack": -30.0, "body": None},
         "click_db": -45.0, "width": 0.30}
    b = {"pitched": True, "pitch_body_hz": 100.0, "attack_ms": 0.3, "fall_20_ms": 200.0,
         "fall_40_ms": 300.0, "pitch_settle_ms": 1.5, "harmonics_db": {2: -60.0, 3: -10.0},
         "harmonics_floor_db": {2: -90.0, 3: -90.0}, "noise_db": {"attack": -40.0, "body": -20.0},
         "click_db": -60.0, "width": 0.15}
    c = oneshot.compare(a, b)
    # body pitch 0.6 st = 2x its threshold, fall-20 50 % shorter = 3.3x, width 0.15 = 1.5x;
    # attack 0.1 ms and settle 1.5 ms are under the 2 ms floor, the 2nd harmonic is under
    # -50 dB, attack noise under -25 dB and click under -40 dB on both sides
    assert [r.split()[0] for r in c["readings"]] == ["falls", "body", "width"], c["readings"]
    assert c["diff"]["fall_40_ms"] is None and c["diff"]["noise_db"]["body"] is None
    assert c["diff"]["attack_ms"] == {"ms": pytest.approx(0.1), "ratio": pytest.approx(4 / 3, abs=1e-3)}


# ---------------------------------------------------------------- hits

def test_hits_four_at_500ms_and_the_third_alone():
    one = hit(100, 0.050, seconds=0.5, partials={n: 1.0 / n for n in range(1, 16, 2)})
    take = np.concatenate([one] * 4)
    r = oneshot.profile_array(take, SR, hit=2)
    assert r["hits"] == 4 and r["hit"] == 2 and r["truncated"] is False
    assert r["onsets_ms"] == pytest.approx([0, 500, 1000, 1500], abs=0.5)
    assert r["onset_ms"] == pytest.approx(1000, abs=0.5)
    # the single hit's numbers, as items 1 and 2 derive them
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.02)
    assert r["harmonics_db"][3] == pytest.approx(20 * math.log10(1 / 3), abs=1)
    assert r["harmonics_db"][5] == pytest.approx(20 * math.log10(1 / 5), abs=1)
    assert r["harmonics_db"][2] < -40 and r["harmonics_db"][4] < -40
    assert oneshot.profile_array(take, SR)["hit"] == 0


def test_hits_file_ending_60ms_into_a_300ms_decay_is_truncated():
    # fall-40 of e^(-t/tau) is tau ln(100): 300 ms for tau = 65.1 ms
    r = oneshot.profile_array(hit(100, 0.300 / math.log(100), seconds=0.060), SR)
    assert r["hits"] == 1 and r["truncated"] is True and r["fall_40_ms"] is None
    assert any("file ends before the hit falls 40 dB" in n for n in r["notes"])


def test_hits_next_hit_cuts_the_one_before():
    # two hits 200 ms apart, each falling 40 dB only after 100 ln(100) = 460 ms
    take = np.concatenate([hit(100, 0.1, seconds=0.2), hit(100, 0.1, seconds=0.8)])
    first = oneshot.profile_array(take, SR, hit=0)
    assert first["hits"] == 2 and first["truncated"] is True and first["fall_40_ms"] is None
    assert any("next hit starts before the hit falls 40 dB" in n for n in first["notes"])
    assert first["fall_10_ms"] == pytest.approx(100 * math.log(10 ** 0.5), abs=6)    # 115 ms, before the cut
    default = oneshot.profile_array(take, SR)
    assert default["hit"] == 1 and default["truncated"] is False


def test_hits_fast_roll_is_split_at_120ms():
    # each hit is 21 dB down (8.686 * 120 / 50) when the next starts
    take = np.concatenate([hit(100, 0.05, seconds=0.12)] * 5 + [hit(100, 0.05, seconds=0.5)])
    r = oneshot.profile_array(take, SR)
    assert r["hits"] == 6
    assert r["onsets_ms"] == pytest.approx([0, 120, 240, 360, 480, 600], abs=1)


@pytest.mark.parametrize("kind", ["tremolo 6 Hz", "tremolo 20 Hz", "beating 4 Hz", "beating 15 Hz"])
def test_hits_tremolo_or_beating_is_not_split(kind):
    t = np.arange(int(1.0 * SR)) / SR
    rate = float(kind.split()[1])
    if kind.startswith("tremolo"):         # 99 % depth: 40 dB troughs
        x = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.3) * (1 - 0.99 * (0.5 - 0.5 * np.cos(2 * np.pi * rate * t)))
    else:                                  # two equal modes: true nulls `rate` times a second
        x = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * (100 + rate) * t)) * np.exp(-t / 0.4)
    assert oneshot.profile_array(x[:, None], SR)["hits"] == 1


def test_hits_out_of_range_is_refused(tmp_path):
    take = np.concatenate([hit(100, 0.05, seconds=0.5)] * 2)
    with pytest.raises(ValueError, match="hit 2 is out of range: the take has 2 hits"):
        oneshot.profile_array(take, SR, hit=2)
    p = tmp_path / "take.wav"
    sf.write(str(p), take, SR, subtype="FLOAT")
    with pytest.raises(ValueError, match="out of range"):
        oneshot.profile(str(p), hit=5)
    assert oneshot.profile(str(p), hit=1)["onset_ms"] == pytest.approx(500, abs=0.5)
    c = oneshot.compare(oneshot.profile(str(p), hit=0), oneshot.profile(str(p), hit=1))
    assert c["readings"] == []


# ---------------------------------------------------------------- review fixes

def test_envelope_dc_offset_does_not_change_the_falls():
    # a DC offset is no part of the decay: + 0.05 (-26 dB) must read as the plain hit
    plain = oneshot.profile_array(hit(100, 0.050), SR)
    for dc in (0.05, -0.2, 0.005):
        r = oneshot.profile_array(hit(100, 0.050) + dc, SR)
        assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6), dc
        assert r["fall_40_ms"] == pytest.approx(50 * math.log(100), abs=6), dc
        assert r["bands_db"] == pytest.approx(plain["bands_db"], abs=0.2), dc     # DC adds no sub


def test_envelope_onset_ignores_a_lone_spike_before_the_hit():
    # a single sample at -38 dB, 8 ms before a hit that starts at 10 ms: its 1 ms frame
    # is 38 + 10 log10(48) = 55 dB down, so the onset is the hit's
    x = np.concatenate([np.zeros((480, 1)), hit(100, 0.050)])
    x[96] = 10 ** (-38 / 20)
    assert oneshot.profile_array(x, SR)["onset_ms"] == pytest.approx(10.0, abs=0.05)


def test_envelope_nan_or_inf_is_refused():
    x = hit(100, 0.05, seconds=1.0)
    for bad in (np.nan, np.inf):
        y = x.copy()
        y[100] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            oneshot.profile_array(y, SR)


def test_envelope_5ms_file_is_none_with_reasons_not_numbers():
    r = oneshot.profile_array(hit(200, 0.05, seconds=0.005), SR)
    assert r["fall_20_ms"] is None and r["fall_40_ms"] is None and r["truncated"] is True
    assert r["pitch_body_hz"] is None and r["harmonics_db"] is None
    assert any("too short" in n for n in r["notes"])


def test_envelope_attack_ignores_start_phase_and_clipping():
    # both start phases of an instant 40 Hz hit, and a hit clipped flat at full scale,
    # peak at the onset; a 10 ms linear ramp first comes within 1 dB of its top at
    # 10 * 10^(-1/20) = 8.9 ms
    for x in (hit(40, 0.2), hit(40, 0.2, phase=math.pi / 2), np.clip(4 * hit(100, 0.1), -1, 1)):
        assert oneshot.profile_array(x, SR)["attack_ms"] <= 1.0
    r = oneshot.profile_array(hit(200, 0.15, attack=0.010), SR)
    assert r["attack_ms"] == pytest.approx(10 * 10 ** (-1 / 20), abs=1.0)


def test_pitch_weak_fundamental_under_strong_overtones():
    # 100 Hz at -26 dB under 200, 300, 400 Hz: 300 Hz belongs to 100's series and
    # not to 200's, so 100 Hz is the fundamental
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 0.05, 2: 1.0, 3: 0.5, 4: 0.3}), SR)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.02)
    assert r["harmonics_db"][2] == pytest.approx(20 * math.log10(1 / 0.05), abs=1)    # +26 dB


def test_pitch_lone_peak_under_the_pitch_is_named_not_taken():
    # a 150 Hz tom (and its 2nd harmonic) over a 50 Hz rumble 14 dB down: nothing but the
    # rumble itself is in 50 Hz's series and not 150 Hz's, so the pitch stays 150 Hz
    t = np.arange(int(0.6 * SR)) / SR
    x = (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
         + 0.2 * np.sin(2 * np.pi * 50 * t)) * np.exp(-t / 0.15)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitch_body_hz"] == pytest.approx(150, rel=0.01)
    assert any(n.startswith("50 Hz") and "1/3 of the pitch" in n for n in r["notes"])
    # the same shape with the low peak further down (100 Hz at -22 dB under 300 and 600 Hz)
    # cannot be told from a rumble either: it is named, not dropped
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 0.08, 3: 1.0, 6: 0.3}), SR)
    assert any(n.startswith("100 Hz") and "1/3 of the pitch" in n for n in r["notes"])


@pytest.mark.parametrize("snr", [0, -3, -6])
def test_noise_tone_buried_in_noise_stays_pitched(snr):
    # a 200 Hz tone with white noise at the tone's power + |snr| dB: per bin the tone
    # still stands far above the noise, so it is pitched at 200 Hz
    t = np.arange(int(0.5 * SR)) / SR
    noise = np.random.default_rng(1).standard_normal(len(t)) * math.sqrt(0.5) * 10 ** (-snr / 20)
    r = oneshot.profile_array(((np.sin(2 * np.pi * 200 * t) + noise) * np.exp(-t / 0.08))[:, None], SR)
    assert r["pitched"] is True and r["pitch_body_hz"] == pytest.approx(200, rel=0.01)


@pytest.mark.parametrize("f0, tau, too_short", [(100, 0.012, True), (300, 0.003, True), (60, 0.040, False),
                                                (1000, 0.004, False), (1000, 0.002, False)])
def test_harmonic_short_decay_shows_no_fake_harmonic_or_noise(f0, tau, too_short):
    # a pure tone has no 2nd harmonic and no noise, however fast it decays: either the
    # body is too short to say (None, with a note) or the reading says so. The body runs
    # to the -40 dB point, tau ln(100), and needs 4 periods: 100 Hz tau 12 ms holds
    # (55.3 - 27.6) ms = 2.8 periods, 300 Hz tau 3 ms 2.1; 60 Hz tau 40 ms (40-150 ms)
    # holds 6.6, 1 kHz tau 4 ms 9.2, tau 2 ms 4.6
    r = oneshot.profile_array(hit(f0, tau, seconds=0.3), SR)
    assert r["pitch_body_hz"] == pytest.approx(f0, rel=0.02)
    assert (r["harmonics_db"] is None) is too_short
    if r["harmonics_db"] is None:
        assert r["noise_db"]["body"] is None
        assert any("under 4 periods" in n for n in r["notes"])
    else:
        assert r["harmonics_db"][2] < -40
        assert r["noise_db"]["body"] < -30


def test_compare_unpitched_hits_read_the_band_balance():
    # white noise against the same noise with everything above 2 kHz removed: with no
    # pitch to go by, the tone is the band balance, and only the two bands above 2 kHz
    # (high_mid 2-6 kHz, high 6-20 kHz) may read low. The four bands below keep their
    # balance, so none of them reads high.
    rng = np.random.default_rng(7)
    n = int(SR * 0.4)
    env = np.exp(-np.arange(n) / SR / 0.05)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    spec[np.fft.rfftfreq(n, 1 / SR) > 2000.0] = 0.0
    dark = np.fft.irfft(spec, n)
    a = oneshot.profile_array((0.3 * dark * env)[:, None], SR)
    b = oneshot.profile_array((0.3 * white * env)[:, None], SR)
    assert a["pitched"] is False and b["pitched"] is False
    bands = [r for r in oneshot.compare(a, b)["readings"] if "against the other bands" in r]
    assert sorted(r.split(" (")[0] for r in bands) == ["high mids", "highs"], bands
    assert all(" low " in r for r in bands), bands
    # the same hit against itself reads nothing
    assert oneshot.compare(b, b)["readings"] == []


def test_compare_pitch_drop_line_gives_the_time_and_direction():
    base = {"pitched": True, "pitch_body_hz": 100.0}
    steady = dict(base, pitch_drop_st=0.0, pitch_settle_ms=0.0)
    # 12 st down over 40 ms is a glide; 17 st over 3 ms is a click; -5 st is a rise
    glide_, click, rise = (dict(base, pitch_drop_st=d, pitch_settle_ms=t)
                           for d, t in ((12.0, 40.0), (17.0, 3.0), (-5.0, 20.0)))
    assert oneshot.compare(glide_, steady)["readings"] == ["pitch drops 12.0 st over 40 ms, reference drops 0.0 st"]
    assert oneshot.compare(click, steady)["readings"] == [
        "pitch drops 17.0 st over 3.0 ms (a click), reference drops 0.0 st"]
    assert oneshot.compare(rise, steady)["readings"] == ["pitch rises 5.0 st over 20 ms, reference drops 0.0 st"]


@pytest.mark.parametrize("seed", [2, 7])
def test_noise_cut_off_at_2khz_is_unpitched_at_its_edge(seed):
    # band-limited noise has no pitch. Its brick-wall edge has nothing above it, so a
    # floor taken from both sides at once made the last bins read as a 2 kHz tone
    # (these two seeds did, before the floor took the higher side).
    n = int(SR * 0.4)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / SR) > 2000.0] = 0.0
    x = 0.3 * np.fft.irfft(spec, n) * np.exp(-np.arange(n) / SR / 0.05)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is False and r["pitch_body_hz"] is None and r["harmonics_db"] is None


# ---------------------------------------------------------------- second review

def kick_glide(t):
    """45 + 115 e^(-t / 12 ms): 160 Hz at the onset down to a 45 Hz body."""
    return 45 + 115 * np.exp(-np.asarray(t) / 0.012)


def pad(x, ms=30):
    return np.concatenate([np.zeros((int(ms * SR / 1000), x.shape[1])), x])


@pytest.mark.parametrize("name", ["808 45 Hz", "clipped kick", "kick cut at 150 ms"])
def test_envelope_low_tone_still_sounding_is_not_dc(name):
    # none of these files has a quiet block at its end, so no DC is invented: the hit
    # behind 30 ms of zeros reads exactly as the hit
    x = {"808 45 Hz": hit(45, 0.3), "clipped kick": np.clip(3 * hit(kick_glide, 0.2), -1, 1),
         "kick cut at 150 ms": hit(kick_glide, 0.12, seconds=0.15)}[name]
    a, b = oneshot.profile_array(x, SR), oneshot.profile_array(pad(x), SR)
    assert oneshot.compare(b, a)["readings"] == []
    assert a["attack_ms"] <= 1.0 and b["attack_ms"] <= 1.0
    if name == "kick cut at 150 ms":
        assert a["fall_10_ms"] == pytest.approx(120 * math.log(10 ** 0.5), abs=6)     # 138.2 ms
        assert a["pitch_drop_st"] == pytest.approx(12 * math.log2(160 / 45), abs=2)


def test_envelope_cut_45hz_take_keeps_its_fall():
    # tau 120 ms falls 10 dB at 120 ln(10^0.5) = 138.2 ms, inside the 150 ms the file holds
    r = oneshot.profile_array(hit(45, 0.12, seconds=0.15), SR)
    assert r["fall_10_ms"] == pytest.approx(120 * math.log(10 ** 0.5), abs=6)
    assert not any("every hit is cut off" in n for n in r["notes"])      # one hit: no such note


def test_hits_take_starting_inside_a_hit_defaults_to_the_next():
    # the file starts 26.5 ms into a hit (-4.6 dB down), then a full hit at 300 ms
    full = hit(100, 0.05, seconds=0.5, partials={1: 1, 3: 0.3})
    take = np.concatenate([full[1272:1272 + 14400], full])
    r = oneshot.profile_array(take, SR)
    assert r["hits"] == 2 and r["hit"] == 1 and r["onset_ms"] == pytest.approx(300, abs=0.5)
    assert any("starts inside a hit" in n for n in r["notes"])
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)
    assert oneshot.profile_array(take, SR, hit=0)["hit"] == 0


def test_hits_too_short_last_one_keeps_the_hit_picked():
    take = np.concatenate([hit(100, 0.1, seconds=0.2), hit(100, 0.1, seconds=0.2), hit(100, 0.1, seconds=0.003)])
    r = oneshot.profile_array(take, SR)
    assert r["hits"] == 3 and r["hit"] == 0 and r["truncated"] is True
    assert any("hit 2 is too short" in n for n in r["notes"])
    assert any("every hit is cut off" in n for n in r["notes"])


def test_envelope_capped_file_blames_the_limit():
    r = oneshot.profile_array(hit(100, 20.0, seconds=10.5), SR)
    assert any("10 s limit is reached before the hit falls 20 dB" in n for n in r["notes"])
    assert not any("file ends" in n for n in r["notes"])


@pytest.mark.parametrize("f0", [40, 45, 60, 300, 557, 600, 750, 1000])
def test_pitch_steady_tone_has_no_drop_at_any_phase_or_rate(f0):
    # a steady sine has no pitch drop, whatever its start phase or sample rate, and two
    # such hits of one pitch compare with no pitch or drop reading
    profiles = []
    for phase in (0.0, 1.0, math.pi / 2, math.pi):
        for sr in (44100, 48000):
            r = oneshot.profile_array(hit(f0, 0.1, seconds=0.3, phase=phase, sr=sr), sr)
            assert abs(r["pitch_drop_st"]) <= 0.2, (phase, sr, r["pitch_drop_st"])
            profiles.append(r)
    for a in profiles:
        for b in profiles:
            assert not [t for t in oneshot.compare(a, b)["readings"] if "pitch" in t], (a["pitch_start_hz"], b["pitch_start_hz"])


@pytest.mark.parametrize("partials", [{1: 0.33, 2: 1.0}, {1: 1.0, 2: 1.0}, {1: 0.5, 2: 1.0, 3: 0.5}])
def test_pitch_steady_tone_with_a_loud_2nd_harmonic_has_no_drop(partials):
    # a louder 2nd harmonic adds zero crossings in long-short turns: they must not read
    # as a pitch, and the frames must not be pulled by its lobe (4 % with 3-period frames)
    r = oneshot.profile_array(hit(100, 0.2, partials=partials), SR)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.01)
    assert abs(r["pitch_drop_st"]) <= 0.2
    assert all(v == pytest.approx(100, rel=0.012) for v in r["pitch_track_hz"].values() if v is not None)


def test_pitch_weak_fundamental_is_tracked_like_any_tone():
    # 100 Hz at -26 dB under 200/300/400 Hz: the track follows the 2nd harmonic and halves it
    weak = {1: 0.05, 2: 1.0, 3: 0.5, 4: 0.3}
    r = oneshot.profile_array(hit(100, 0.2, partials=weak), SR)
    assert r["pitch_start_hz"] == pytest.approx(100, rel=0.02)
    assert abs(r["pitch_drop_st"]) <= 0.2
    assert [v for v in r["pitch_track_hz"].values() if v is not None] == pytest.approx([100] * 5, rel=0.02)
    # the same tone gliding 160 -> 100 Hz over 30 ms reads a drop against it
    glide160 = oneshot.profile_array(hit(lambda t: np.where(t < 0.03, 160 - 60 * t / 0.03, 100.0), 0.2,
                                         partials=weak), SR)
    assert any(t.startswith("pitch drops") for t in oneshot.compare(glide160, r)["readings"])


@pytest.mark.parametrize("rumble", [40.0, 50.0, 63.0, 75.0])
def test_pitch_tom_over_any_rumble_stays_at_150(rumble):
    # a 150 Hz tom (with its 2nd harmonic) over a rumble 14 dB down, at 1/k of the pitch
    # or not: the tom's series holds nearly all the energy, so the pitch is 150 Hz and
    # the rumble is listed as a partial
    t = np.arange(int(0.6 * SR)) / SR
    x = (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
         + 0.2 * np.sin(2 * np.pi * rumble * t)) * np.exp(-t / 0.15)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is True and r["pitch_body_hz"] == pytest.approx(150, rel=0.01)
    assert r["partials"][0]["hz"] == pytest.approx(rumble, rel=0.01)
    assert r["partials"][0]["db"] == pytest.approx(20 * math.log10(0.2), abs=1)          # -14 dB


@pytest.mark.parametrize("f0, tau", [(1000, 0.002), (300, 0.003), (200, 0.012), (60, 0.020)])
def test_compare_half_the_decay_of_a_fast_hit_reads_only_decay(f0, tau):
    # the same pure tone at tau and 2 tau: only the falls and the length differ, each
    # by the ratio of the time constants
    a = oneshot.profile_array(hit(f0, tau, seconds=0.3), SR)
    b = oneshot.profile_array(hit(f0, 2 * tau, seconds=0.3), SR)
    c = oneshot.compare(a, b)
    assert c["readings"] and all(r.startswith(("falls", "lasts")) for r in c["readings"]), c["readings"]
    assert c["diff"]["fall_20_ms"]["ratio"] == pytest.approx(0.5, abs=0.07)


def test_compare_drops_read_at_different_times_are_compared_from_the_later():
    # the same kick glide, one with a 3rd harmonic that keeps its zero crossings from
    # reading a pitch: its start is the first frame's (20 ms), the other's the onset's.
    # From 20 ms on both glides are the same, so there is no drop reading
    a = oneshot.profile_array(hit(kick_glide, 0.12, partials={1: 1.0, 3: 1.5}), SR)
    b = oneshot.profile_array(hit(kick_glide, 0.12), SR)
    assert abs(a["pitch_start_at_ms"] - b["pitch_start_at_ms"]) > 2
    c = oneshot.compare(a, b)
    assert not [t for t in c["readings"] if "pitch" in t], c["readings"]
    assert c["diff"]["pitch_drop_from_ms"] == pytest.approx(max(a["pitch_start_at_ms"], b["pitch_start_at_ms"]))
    assert abs(c["diff"]["pitch_drop_st"]) < 1


def test_compare_drop_not_comparable_says_so_only_when_it_matters():
    base = {"pitched": True, "pitch_body_hz": 100.0, "pitch_track_hz": {}}
    early = dict(base, pitch_start_hz=200.0, pitch_start_at_ms=0.0, pitch_drop_st=12.0)
    late = dict(base, pitch_start_hz=110.0, pitch_start_at_ms=30.0, pitch_drop_st=1.65)
    lines = oneshot.compare(early, late)["readings"]
    assert lines == ["pitch drop not compared: the start is read at 0.0 ms on one side and 30 ms on the other"]
    small = dict(early, pitch_start_hz=105.0, pitch_drop_st=0.84)
    still = dict(late, pitch_start_hz=104.0, pitch_drop_st=0.68)
    assert oneshot.compare(small, still)["readings"] == []


def test_compare_octave_or_twelfth_flip_is_one_line():
    def tone(partials):
        return oneshot.profile_array(hit(100, 0.2, partials=partials), SR)
    # 100 Hz 10.5 dB under 200 Hz reads 200 Hz (a lone peak at 1/2, LONE_DB down);
    # 9.6 dB under reads 100 Hz: the same partials, read an octave apart
    c = oneshot.compare(tone({1: 0.30, 2: 1.0}), tone({1: 0.33, 2: 1.0}))
    assert len(c["readings"]) == 1 and c["readings"][0].startswith("pitch read an octave above the reference")
    c = oneshot.compare(tone({1: 0.08, 3: 1.0, 6: 0.3}), tone({1: 0.5, 3: 1.0, 6: 0.3}))
    assert c["readings"][0].startswith("pitch read a twelfth above the reference")
    assert not [t for t in c["readings"][1:] if "harmonic" in t or "drop" in t], c["readings"]


def test_compare_times_longer_and_shorter_weigh_the_same():
    def fall(ms):
        return {"pitched": False, "fall_20_ms": ms}
    for ms, read in ((116.0, True), (86.0, True), (115.0, False), (87.0, False)):
        # |ln(ratio)| against ln(1.15): 1.16 and 0.86 are past it, 1.15 and 0.87 are not
        assert bool(oneshot.compare(fall(ms), fall(100.0))["readings"]) is read, ms


def fft_resample(x, sr0, sr1):
    """Band-limited resampling by zero-padding or truncating the spectrum."""
    n1 = int(round(len(x) * sr1 / sr0))
    spec = np.fft.rfft(x, axis=0)
    out = np.zeros((n1 // 2 + 1, x.shape[1]), complex)
    k = min(len(out), len(spec))
    out[:k] = spec[:k]
    return np.fft.irfft(out, n1, axis=0) * n1 / len(x)


@pytest.mark.parametrize("name", ["sweep", "kick", "tom with h2"])
def test_compare_hit_resampled_from_48_to_44k_gives_no_reading(name):
    x = {"sweep": hit(sweep, 0.1, partials=BASE), "kick": hit(kick_glide, 0.12),
         "tom with h2": hit(150, 0.15, partials={1: 1.0, 2: 0.3})}[name]
    a = oneshot.profile_array(x, SR)
    b = oneshot.profile_array(fft_resample(x, 48000, 44100), 44100)
    assert oneshot.compare(b, a)["readings"] == []


def test_noise_fast_hit_recorded_quietly_reads_no_attack_noise():
    # 1 kHz, tau 2 ms, at -30 dBFS in 16 bits: the attack window stops 30 dB down, before
    # flattening the decay would lift the quantisation floor into "noise"
    x = hit(1000, 0.002, seconds=0.3)
    quiet = np.round(x * 10 ** (-30 / 20) * 32767) / 32767
    c = oneshot.compare(oneshot.profile_array(quiet, SR), oneshot.profile_array(x, SR))
    assert not [t for t in c["readings"] if "attack" in t], c["readings"]


def test_pitch_no_start_when_the_track_cannot_reach_the_onset():
    # a hit that is noise for its first 60 ms and a 300 Hz tone after: the track from the
    # body cannot be followed back to the onset, so there is no start pitch or drop
    t = np.arange(int(0.5 * SR)) / SR
    noise = np.random.default_rng(12).standard_normal(len(t)) * (t < 0.06)
    tone = np.sin(2 * np.pi * 300 * t) * (t >= 0.06) * np.exp(-(t - 0.06) / 0.1)
    r = oneshot.profile_array((noise * 0.8 + tone)[:, None], SR)
    assert r["pitched"] is True and r["pitch_body_hz"] == pytest.approx(300, rel=0.01)
    assert r["pitch_start_hz"] is None and r["pitch_drop_st"] is None
    assert any("cannot be followed back to the onset" in n for n in r["notes"])


# ---------------------------------------------------------------- third review

def test_hits_trimmed_take_followed_by_a_louder_cut_hit_keeps_hit_0():
    # a complete hit trimmed at its onset, then a louder hit cut after 100 ms: hit 0
    # starts at full level, as a tail would, but it is the only complete hit
    full = hit(100, 0.05, seconds=0.5, partials={1: 1, 3: 0.3})
    take = np.concatenate([full, 1.6 * full[:4800]])
    r = oneshot.profile_array(take, SR)
    assert r["hits"] == 2 and r["hit"] == 0 and r["truncated"] is False
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)        # 115.1 ms
    assert not any("starts inside a hit" in n for n in r["notes"])
    # the same take behind 30 ms of silence is the same hit: no readings
    late = np.concatenate([np.zeros((int(0.03 * SR), 1)), take])
    assert oneshot.compare(oneshot.profile_array(late, SR), r)["readings"] == []


def test_hits_trimmed_take_followed_by_a_3ms_fragment_does_not_raise():
    # the louder fragment is too short to measure (150 samples = 3.1 ms): hit 0 is kept
    full = hit(100, 0.05, seconds=0.5, partials={1: 1, 3: 0.3})
    r = oneshot.profile_array(np.concatenate([full, 1.6 * full[:150]]), SR)
    assert r["hit"] == 0 and r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)


def test_pitch_gliding_kick_with_a_loud_3rd_harmonic_is_not_read_3x_high():
    # the body pitch at 40-150 ms lies between 55 + 180 e^(-150/30.9) = 56.4 Hz and
    # 55 + 180 e^(-40/30.9) = 104.3 Hz. Its 3rd harmonic is 3.5 dB louder; taking the
    # fundamental for a rumble under it read 188 Hz. No pitch is acceptable, 3x is not.
    r = oneshot.profile_array(hit(lambda t: 55 + 180 * np.exp(-t / 0.0309), 0.2, partials={1: 1, 3: 1.5}), SR)
    assert r["pitch_body_hz"] is None or 56.4 * 0.97 <= r["pitch_body_hz"] <= 104.3 * 1.03, r["pitch_body_hz"]
    assert not any("rumble" in n for n in r["notes"])


def test_envelope_dc_offset_behind_silence_or_a_constant_lead_in():
    # the offset must go whether the lead-in is digital zeros or the offset itself
    tom = hit(150, 0.08, seconds=0.5)
    ref = oneshot.profile_array(tom, SR)
    lead = int(0.03 * SR)
    for pad in (0.0, 0.02):
        x = np.concatenate([np.full((lead, 1), pad), tom + 0.02])
        assert oneshot.compare(oneshot.profile_array(x, SR), ref)["readings"] == [], pad


def test_compare_two_octave_flip_is_one_line():
    # the same tone, fundamental just under and just over the level where it stops
    # being taken for a rumble under its 4th harmonic: one line, not a 24 st pitch
    # difference and a 4th harmonic 100 dB apart
    a = oneshot.profile_array(hit(100, 0.1, partials={1: 0.30, 4: 1}), SR)
    b = oneshot.profile_array(hit(100, 0.1, partials={1: 0.33, 4: 1}), SR)
    lines = oneshot.compare(a, b)["readings"]
    if a["pitch_body_hz"] != pytest.approx(b["pitch_body_hz"], rel=0.02):
        assert len(lines) == 1 and "two octaves" in lines[0], lines
    else:
        assert lines == []


def test_envelope_cut_hit_has_no_duration():
    # a 300 ms time constant cut 60 ms in never falls 60 dB inside the file
    r = oneshot.profile_array(hit(100, 0.3, seconds=0.06), SR)
    assert r["truncated"] is True and r["duration_ms"] is None and r["fall_40_ms"] is None
    assert any("no duration" in n for n in r["notes"])


def test_compare_settle_time_alone_is_not_a_reading():
    # two steady tones: neither drops, so when each "settles" says nothing
    base = {"pitched": True, "pitch_body_hz": 230.0}
    a = dict(base, pitch_drop_st=0.27, pitch_settle_ms=0.0)
    b = dict(base, pitch_drop_st=0.27, pitch_settle_ms=7.5)
    assert oneshot.compare(a, b)["readings"] == []
    # with a real drop on one side the settle time still reads
    c = dict(base, pitch_drop_st=3.0, pitch_settle_ms=30.0)
    d = dict(base, pitch_drop_st=3.0, pitch_settle_ms=7.5)
    assert [r.split()[1] for r in oneshot.compare(c, d)["readings"]] == ["settles"]


# ---------------------------------------------------------------- clean hit

def take(onsets_s, total_s, freq=100, tau=0.03):
    """Decaying 100 Hz hits starting at `onsets_s`, in a file `total_s` long."""
    x = np.zeros((int(total_s * SR), 1))
    for t in onsets_s:
        h = hit(freq, tau, seconds=total_s - t)
        x[int(t * SR):int(t * SR) + len(h)] += h
    return x


# With tau = 30 ms a hit is 40 dB down after 30 * ln(100) = 138 ms, so a hit with
# less than that before the next onset (or the end) is cut off.
def test_clean_hit_is_the_third_when_the_gaps_are_120_120_and_500_ms():
    r = oneshot.profile_array(take([0, 0.12, 0.24], 0.74), SR, hit="clean")
    assert r["hits"] == 3 and r["hit"] == 2 and not r["truncated"]
    assert any("clean hit" in n and "500 ms" in n for n in r["notes"]), r["notes"]


def test_clean_hit_is_the_earliest_typical_one_not_the_one_with_most_room():
    # rooms 300, 600, 300 ms, all complete: the median is 300 ms and all are typical -> the earliest
    r = oneshot.profile_array(take([0, 0.3, 0.9], 1.2), SR, hit="clean")
    assert r["hits"] == 3 and r["hit"] == 0
    # the same, the odd long room first: 800, 300, 300, 300 ms -> still the earliest
    assert oneshot.profile_array(take([0, 0.8, 1.1, 1.4], 1.7), SR, hit="clean")["hit"] == 0
    # two equal rooms: the earliest
    assert oneshot.profile_array(take([0, 0.4], 0.8), SR, hit="clean")["hit"] == 0


def test_clean_hit_skips_a_complete_hit_with_less_than_90_percent_of_the_median_room():
    # tau 20 ms: 40 dB down after 92 ms. Rooms 150, 500, 500 ms: all complete, the median is 500, and
    # 150 < 0.9 * 500 -> not typical (a fill note), so the pick is hit 1
    x = take([0, 0.15, 0.65], 1.15, tau=0.02)
    r = oneshot.profile_array(x, SR, hit="clean")
    assert r["hits"] == 3 and r["hit"] == 1


def test_clean_hit_leaves_out_a_ghost_note_and_hits_before_the_span():
        # three ghosts first (the median hit is a ghost), then two full-level hits: only the 6 dB band picks a loud one
    loud, ghost = take([1.4, 1.9], 2.4), 0.2 * take([0.2, 0.7, 1.0], 2.4)
    r = oneshot.profile_array(loud + ghost, SR, hit="clean")
    assert r["hits"] == 5 and r["hit"] == 3 and r["onset_ms"] == pytest.approx(1400, abs=3)
    # within= keeps hits that start before the span out (the tail of a hit from the section before)
    r = oneshot.profile_array(take([0.0, 0.7, 1.4], 2.1, tau=0.05), SR, hit="clean", within=(0.5, 2.1))
    assert r["hit"] == 1 and r["onset_ms"] == pytest.approx(700, abs=3)
    # near= prefers the hit at that position, within 30 ms
    r = oneshot.profile_array(take([0.0, 0.7, 1.4], 2.1, tau=0.05), SR, hit="clean", near=1.41)
    assert r["hit"] == 2


def test_clean_hit_when_only_the_last_is_complete():
    r = oneshot.profile_array(take([0, 0.15, 0.30], 0.6, tau=0.05), SR, hit="clean")   # 40 dB down after 230 ms
    assert r["hits"] == 3 and r["hit"] == 2


def test_clean_hit_of_two_complete_hits_the_one_with_far_less_room_is_not_typical():
    # tau 50 ms: hit 1 has 250 ms of room (just complete), hit 2 has 400; the median is 325 and 250 < 0.9 * 325
    r = oneshot.profile_array(take([0, 0.20, 0.45], 0.85, tau=0.05), SR, hit="clean")
    assert r["hits"] == 3 and r["hit"] == 2


def test_clean_hit_a_quiet_lead_in_is_preferred_to_an_earlier_hit_riding_on_a_tail():
    # tau 50 ms. Hit 0 is cut off by hit 1 100 ms later; hit 1 is complete (600 ms of room) but starts on
    # hit 0's tail, which is 16 dB down in the 20 ms before it. Hit 2 (500 ms of room) starts from quiet.
    # Of the two complete hits the one with the quiet lead-in is the pick, though it is later.
    r = oneshot.profile_array(take([0, 0.10, 0.70], 1.2, tau=0.05), SR, hit="clean")
    assert r["hits"] == 3 and r["hit"] == 2
    # with nothing else complete, the hit on the tail is still better than nothing (the rule is a preference)
    assert oneshot.profile_array(take([0, 0.10], 0.9, tau=0.05), SR, hit="clean")["hit"] == 1


def test_clean_hit_a_peak_more_than_3_db_from_the_median_is_not_typical():
    # four hits with 500 ms of room each; the first is 4 dB louder than the other three (the median), all
    # within the 6 dB band: the pick is the earliest typical one, hit 1, not hit 0
    x = take([0, 0.5, 1.0, 1.5], 2.0, tau=0.03)
    x[int(0.5 * SR):] *= 10 ** (-4 / 20)
    r = oneshot.profile_array(x, SR, hit="clean")
    assert r["hits"] == 4 and r["hit"] == 1
    # 2 dB louder is typical: the earliest, hit 0
    y = take([0, 0.5, 1.0, 1.5], 2.0, tau=0.03)
    y[int(0.5 * SR):] *= 10 ** (-2 / 20)
    assert oneshot.profile_array(y, SR, hit="clean")["hit"] == 0


def test_clean_hit_none_complete_is_refused_with_no_clean_hit():
    with pytest.raises(ValueError, match="no clean hit"):
        oneshot.profile_array(take([0, 0.06, 0.12], 0.2, tau=0.1), SR, hit="clean")
    for bad in ("best", "2", 1.0, True, 2.5):
        with pytest.raises(ValueError, match="hit must be"):
            oneshot.profile_array(take([0], 0.5), SR, hit=bad)
    # the default and a number are unchanged by "clean" existing
    assert oneshot.profile_array(take([0, 0.3], 0.9), SR)["hit"] == 0


def test_clean_hit_through_the_file_entry_point(tmp_path):
    p = tmp_path / "take.wav"
    sf.write(str(p), take([0, 0.12, 0.24], 0.74), SR, subtype="FLOAT")
    assert oneshot.profile(str(p), hit="clean")["hit"] == 2
    a, b = oneshot.profile(str(p), hit="clean"), oneshot.profile(str(p), hit="clean")
    assert oneshot.compare(a, b)["readings"] == []
    with pytest.raises(ValueError, match="no clean hit"):
        q = tmp_path / "roll.wav"
        sf.write(str(q), take([0, 0.06, 0.12], 0.2, tau=0.1), SR, subtype="FLOAT")
        oneshot.profile(str(q), hit="clean")


def test_clean_hit_near_a_position_is_judged_against_the_span_not_a_louder_hit_at_its_edge():
    # a full-level hit starts 20 ms before the span (inside the 30 ms slack of near=), the hit in the span is
    # 8 dB down: it is the only candidate inside, so near= must still find it. (A louder edge hit that set the
    # 6 dB band would have dropped it.)
    x = take([0.48], 2.4) + 0.4 * take([1.4], 2.4)
    inside = oneshot.profile_array(x, SR, hit="clean", within=(0.5, 2.4))
    assert inside["hit"] == 1 and inside["onset_ms"] == pytest.approx(1400, abs=3)
    near = oneshot.profile_array(x, SR, hit="clean", within=(0.5, 2.4), near=1.40)
    assert near["hit"] == 1 and "requested position" in near["notes"][0]
    nothing_inside = oneshot.profile_array(x[:int(1.0 * SR)], SR, hit="clean", within=(0.5, 1.0), near=0.49)
    assert nothing_inside["hit"] == 0                  # only an edge hit exists, and it is the one asked for


def test_hit_cut_is_the_same_whatever_the_offset_it_was_read_at():
    x = take([0.30, 0.90], 1.6, tau=0.04)
    a, at_a = oneshot.hit_cut(x, SR, 0.30, 0.55)
    shifted = np.concatenate([np.zeros((int(0.0123 * SR), 1)), x])                  # the same audio read 12.3 ms late
    b, at_b = oneshot.hit_cut(shifted, SR, 0.30 + 0.0123 + 0.004, 0.55)                # and asked for 4 ms off
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) <= 2 and np.allclose(a[:n], b[:n], atol=1e-9)
    assert abs(len(a) / SR - (0.55 + 0.002)) < 0.002          # the lead before the onset, then the length asked for
    assert abs((at_b - at_a) - 0.0123) < 1.5 / SR             # and the onset moved by exactly the 12.3 ms
    ra = oneshot.profile_array(a, SR, hit=0, single=True)
    rb = oneshot.profile_array(b, SR, hit=0, single=True)
    assert oneshot.compare(ra, rb)["readings"] == []


def test_single_takes_the_whole_array_as_one_hit_without_looking_for_hits_inside():
    x = take([0.0, 0.40], 0.8, tau=0.05)                       # two hits
    assert oneshot.profile_array(x, SR)["hits"] == 2
    one = oneshot.profile_array(x, SR, hit=0, single=True)
    assert one["hits"] == 1 and one["hit"] == 0 and one["onsets_ms"] == [one["onset_ms"]]


def test_hit_cut_finds_the_same_onset_wherever_it_is_asked_for_near_the_hit():
    """A noise hat (tau 30 ms) with a soft lead-in, 4 ms of noise at a fifth of its level before its rise, 125 ms after
    another hat. Where `at` falls (8 ms either side of the main rise) decides what the 20 ms tail window holds, and so
    which of the rises a single search takes; searching again from what it found, until it finds the same, gives one
    onset for every `at`, inside the 4 ms lead-in."""
    rng = np.random.default_rng(3)

    def hat(n, tau):
        return np.diff(rng.standard_normal(n + 1)) * 0.5 * np.exp(-np.arange(n) / SR / tau)

    x = np.zeros(SR)
    on = int(0.5 * SR)
    for t0 in (on - int(0.125 * SR), on):
        x[t0:t0 + int(0.3 * SR)] += hat(int(0.3 * SR), 0.03)
    x[on - int(0.004 * SR):on] += 0.2 * hat(int(0.004 * SR), 1.0)
    found = {oneshot.hit_cut(np.stack([x, x], 1), SR, on / SR + d / 1000, 0.1)[1] for d in np.arange(-8.0, 8.01, 0.5)}
    assert len(found) == 1, sorted((f - on / SR) * 1000 for f in found)
    assert -4.0 <= (found.pop() - on / SR) * 1000 <= 0.0


@pytest.mark.parametrize("tau,gap", [(0.08, 0.20), (0.12, 0.15)])
def test_hit_cut_reads_a_low_tail_over_20_ms_not_one_frame(tau, gap):
    """A kick gliding 24 st down to 55 Hz (decay `tau`) and the same kick `gap` s later, on its tail. One 1 ms frame
    of a 55 Hz tail is loud or quiet with the phase it falls on, so a tail level read from the window's first frame
    puts the onset in the tail (tens of ms early) for some `at`. Read as the loudest 1 ms over 20 ms, every `at` within
    6 ms of the second kick finds one onset, at the kick's start (within 2 ms after it: the sine starts at 0)."""
    t = np.arange(int(0.6 * SR)) / SR
    h = np.sin(2 * np.pi * np.cumsum(55.0 * 2 ** (2.0 * np.exp(-t / 0.02))) / SR) * np.exp(-t / tau)
    x = np.zeros(2 * SR)
    for s0 in (int(0.5 * SR), int((0.5 + gap) * SR)):
        x[s0:s0 + len(h)] += h[:len(x) - s0]
    on = int((0.5 + gap) * SR) / SR
    found = {oneshot.hit_cut(np.stack([x, x], 1), SR, on + d / 1000, 0.2)[1] for d in np.arange(-6.0, 6.01, 0.25)}
    assert len(found) == 1, sorted((f - on) * 1000 for f in found if f is not None)
    assert 0.0 <= (found.pop() - on) * 1000 <= 2.0

#!/usr/bin/env python3
"""Tests for beat_aligner, on synthetic signals with known ground truth.

Two levels, deliberately:

- The grid/drift/offset functions are tested on synthetic BEAT ARRAYS. Ground
  truth is exact there, so the assertions can be exact — a tempo assertion good
  to 0.01 BPM would be meaningless if it had to absorb a beat tracker's error
  first.
- One end-to-end test renders actual audio and runs the whole pipeline through
  aubiotrack, which is what catches wiring mistakes the unit tests cannot see.

The two cases that guard real, previously-shipped bugs are marked REGRESSION:
dropped-beat resilience (the `i % 4` bug, where one missing beat rotated the bar
phase for the rest of the track) and fixed-tempo drift (the old detector
compared the std of all intervals to the period, so any track with a break
reported drift).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import beat_aligner as ba

# A yt2stems folder of the reference track (125.000 BPM, no drift). The audio is
# not redistributable, so it is never committed: point ROADIE_REF_STEMS at your
# own copy, or put it at ./reference/. Absent, the real-stem tests skip.
# YT2LIVE_REF_STEMS, the pre-rename name, is still honoured.
STEMS_DIR = (os.environ.get("ROADIE_REF_STEMS")
             or os.environ.get("YT2LIVE_REF_STEMS")) or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reference")

HAVE_AUBIO = subprocess.run(["which", "aubiotrack"],
                            capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def real_stems_result():
    """analyze_alignment on the real stems, once for the whole module.

    Module-scoped because it reads nine stems and runs aubio — a few seconds,
    which is worth paying once rather than per assertion.

    use_cache=False on purpose. These are the tests that exercise the real
    signal path end to end; letting them read alignment.json would turn every
    assertion below into a check that JSON round-trips, and a stale cache would
    keep them green through a genuine regression. The cache has its own tests.
    """
    if not os.path.isdir(STEMS_DIR):
        pytest.skip("test stems not present")
    if not HAVE_AUBIO:
        pytest.skip("aubiotrack not installed")
    return ba.analyze_alignment(STEMS_DIR, use_cache=False)


def beat_times(bpm=124.0, n_beats=600, pre_roll=1.7, drop=0.0, gap=None,
               jitter=0.004, seed=1):
    """Detections for a fixed-tempo track. Returns (times, period, pre_roll).

    `drop` removes that fraction at random (aubio misses beats); `gap` is a
    (start_k, end_k) run removed wholesale, standing in for a breakdown. Jitter
    is the per-beat placement error — 4 ms is roughly what aubio manages on a
    clean kick.

    Jitter is drawn for ALL beats and only then subsetted, so that for a given
    seed a thinned track is an exact subset of the full one. The dropped-beat
    tests compare the two beat for beat, which is only meaningful if the
    surviving timestamps are bit-identical rather than merely similar.
    """
    rng = np.random.RandomState(seed)
    period = 60.0 / bpm
    k = np.arange(n_beats)
    t = pre_roll + period * k + rng.normal(0, jitter, n_beats)

    keep = rng.rand(n_beats) > drop
    if gap is not None:
        keep &= ~((k >= gap[0]) & (k < gap[1]))

    return t[keep], period, pre_roll


class TestFitBeatGrid:
    def test_clean_track_recovers_tempo_and_origin(self):
        times, period, pre_roll = beat_times()
        p, t0, k, inliers = ba.fit_beat_grid(times)

        assert 60.0 / p == pytest.approx(124.0, abs=0.01)
        # t0 is the grid origin nearest zero, so it is the pre-roll modulo one
        # beat — the grid extends backwards past the first detection.
        assert t0 == pytest.approx(pre_roll % period, abs=0.005)
        assert inliers.all()
        # Indices must be strictly increasing: one duplicate or inversion means
        # two detections were assigned the same slot.
        assert np.all(np.diff(k) > 0)

    def test_dropped_beats_and_break_do_not_move_the_grid(self):
        """REGRESSION: the `i % 4` bug. 15% dropped + an 8-bar silent break."""
        clean, period, _ = beat_times()
        holed, _, _ = beat_times(drop=0.15, gap=(200, 232))

        p_clean, t0_clean, _k_clean, _ = ba.fit_beat_grid(clean)
        p_holed, t0_holed, k_holed, inliers = ba.fit_beat_grid(holed)

        assert 60.0 / p_holed == pytest.approx(124.0, abs=0.01)
        assert p_holed == pytest.approx(p_clean, rel=1e-4)
        assert t0_holed == pytest.approx(t0_clean, abs=0.005)
        assert inliers.all()

        # The real point: every surviving beat must keep the exact grid index it
        # had before its neighbours were deleted. `holed` is a subset of
        # `clean`, so this is a beat-for-beat comparison.
        k_clean = ba.fit_beat_grid(clean)[2]
        survivor = np.isin(clean, holed)
        assert survivor.sum() > 400
        assert np.array_equal(k_clean[survivor], k_holed)

    def test_bar_phase_is_unchanged_by_dropped_beats(self):
        """REGRESSION: one dropped beat used to rotate every later bar.

        Asserting `k0 % 4 == phase` would prove nothing — the selector only ever
        returns such indices. What has to hold is that the anchor picks the same
        MUSICAL beat with and without the drops, so the comparison is against
        the clean track's anchor time, not against the selector's own invariant.
        """
        clean, period, _ = beat_times()
        holed, _, _ = beat_times(drop=0.15, gap=(200, 232))

        p_c, t0_c, k_c, inl_c = ba.fit_beat_grid(clean)
        p_h, t0_h, k_h, inl_h = ba.fit_beat_grid(holed)
        anchor_c, k0_c = ba.select_first_stable_downbeat(k_c, inl_c, p_c, t0_c, 0)
        anchor_h, k0_h = ba.select_first_stable_downbeat(k_h, inl_h, p_h, t0_h, 0)

        # Same bar line of the same grid: the two anchors differ by a whole
        # number of BARS, not by 1-3 beats.
        bars_apart = (anchor_h - anchor_c) / period / 4.0
        assert abs(bars_apart - round(bars_apart)) < 1e-3
        assert (k0_h - k0_c) % 4 == 0

    def test_array_index_bar_counting_would_break(self):
        """Pins the old bug: counting bars by array position diverges after a drop.

        Guards against a future "simplification" back to `i % 4` by asserting
        the two indexings genuinely disagree on this signal — without which the
        test above could pass for a track where nothing was dropped at all.
        """
        clean, _, _ = beat_times()
        holed, _, _ = beat_times(drop=0.15, gap=(200, 232))
        p, t0, k_holed, _ = ba.fit_beat_grid(holed)

        survivor = np.isin(clean, holed)
        grid_bar_pos = k_holed % 4                     # correct
        array_bar_pos = np.arange(len(holed)) % 4      # the old, broken way
        # After 15% drops plus a 32-beat gap they agree only by coincidence.
        agreement = float(np.mean(grid_bar_pos == array_bar_pos))
        assert agreement < 0.5
        assert survivor.sum() == len(holed)

    def test_half_time_detection_resolves_up(self):
        times, _, _ = beat_times(bpm=62.0)
        p, _, _, inliers = ba.fit_beat_grid(times)

        assert 60.0 / p == pytest.approx(124.0, abs=0.01)
        # Every 62 BPM detection is a slot of the 124 BPM grid, so none is lost.
        assert inliers.all()

    def test_double_time_detection_resolves_down(self):
        times, _, _ = beat_times(bpm=248.0)
        p, t0, k, inliers = ba.fit_beat_grid(times)

        assert 60.0 / p == pytest.approx(124.0, abs=0.01)
        # Half the detections sit on half-grid slots after the correction and
        # must be REJECTED, not rounded onto the nearest whole slot.
        assert inliers.sum() == pytest.approx(len(times) / 2, rel=0.05)
        residual = np.abs(times[inliers] - (p * k[inliers] + t0))
        assert (residual < ba.OUTLIER_TOL * p).all()

    def test_survives_heavy_placement_jitter(self):
        times, _, _ = beat_times(jitter=0.030)
        p, _, _, _ = ba.fit_beat_grid(times)
        assert 60.0 / p == pytest.approx(124.0, abs=0.02)

    def test_forced_period_is_used_verbatim(self):
        times, _, _ = beat_times()
        forced = 60.0 / 130.0
        p, t0, k, inliers = ba.fit_beat_grid(times, force_period=forced)

        assert p == forced
        # The anchor still has to be fitted to the audio rather than assumed.
        assert 0.0 <= t0 < p

    @pytest.mark.parametrize("bad,match", [
        (np.arange(4) * 0.48, "at least 8 beats"),
        (np.sort(np.random.RandomState(0).uniform(0, 280, 40)), "steady pulse"),
        (np.sort(np.random.RandomState(1).uniform(0, 280, 470)), "steady pulse"),
    ])
    def test_degenerate_input_raises_rather_than_inventing_a_tempo(self, bad, match):
        with pytest.raises(ValueError, match=match):
            ba.fit_beat_grid(bad)


class TestAnchorSelection:
    def test_anchor_lands_on_a_grid_downbeat(self):
        times, _, _ = beat_times()
        p, t0, k, inliers = ba.fit_beat_grid(times)

        for phase in range(4):
            anchor, k0 = ba.select_first_stable_downbeat(k, inliers, p, t0, phase)
            assert k0 % 4 == phase
            assert anchor == pytest.approx(p * k0 + t0)

    def test_anchor_is_not_placed_inside_a_gap(self):
        """The 8 slots after the anchor must all carry a detection."""
        times, _, _ = beat_times(drop=0.15, gap=(0, 120))
        p, t0, k, inliers = ba.fit_beat_grid(times)
        anchor, k0 = ba.select_first_stable_downbeat(k, inliers, p, t0, 0)

        present = set(int(x) for x in k[inliers])
        assert k0 >= 120
        assert all((k0 + n) in present for n in range(1, 9))

    def test_anchor_uses_the_fitted_grid_not_the_raw_detection(self):
        """Sub-10 ms accuracy: the fitted time beats any single noisy detection."""
        times, period, _ = beat_times(jitter=0.020, seed=7)
        p, t0, k, inliers = ba.fit_beat_grid(times)
        anchor, k0 = ba.select_first_stable_downbeat(k, inliers, p, t0, 0)

        # Grid indices are counted from t0, the origin nearest zero, not from
        # the first detection — so the true time of slot k0 is (pre-roll modulo
        # one beat) plus k0 beats.
        truth = (1.7 % period) + period * k0
        nearest_detection = times[np.argmin(np.abs(times - anchor))]
        assert abs(anchor - truth) < abs(nearest_detection - truth) + 1e-9
        assert abs(anchor - truth) < 0.010


class TestOffsetFormula:
    @pytest.mark.parametrize("bpm", [118.0, 124.0, 128.0, 140.0])
    @pytest.mark.parametrize("pre_roll", [0.0, 0.3, 1.7, 5.25])
    def test_anchor_plus_offset_is_a_whole_number_of_bars(self, bpm, pre_roll):
        times, period, _ = beat_times(bpm=bpm, pre_roll=pre_roll, n_beats=200)
        p, t0, k, inliers = ba.fit_beat_grid(times)
        anchor, _k0 = ba.select_first_stable_downbeat(k, inliers, p, t0, 0)

        offset = round((-(anchor / p)) % 4.0, 4)
        landed = (anchor / p + offset) % 4.0
        assert min(landed, 4.0 - landed) < 1e-3

    def test_offset_is_never_negative(self):
        """START must be >= 0 or Live would clip the pre-roll audio."""
        for pre_roll in (0.0, 0.1, 1.7, 3.9):
            times, _, _ = beat_times(pre_roll=pre_roll, n_beats=200)
            p, t0, k, inliers = ba.fit_beat_grid(times)
            anchor, _k0 = ba.select_first_stable_downbeat(k, inliers, p, t0, 0)
            offset = round((-(anchor / p)) % 4.0, 4)
            assert 0.0 <= offset < 4.0


def accelerating_beats(bpm_a=124.0, bpm_b=132.0, dur=180.0, drop=0.0,
                       jitter=0.004, seed=3):
    """Detections for a track whose tempo ramps linearly from bpm_a to bpm_b."""
    rng = np.random.RandomState(seed)
    t, times = 0.0, []
    while t < dur:
        times.append(t)
        t += 60.0 / (bpm_a + (bpm_b - bpm_a) * min(t / dur, 1.0))
    times = np.array(times) + rng.normal(0, jitter, len(times))
    if drop:
        times = times[rng.rand(len(times)) > drop]
    return np.sort(times)


class TestTempoDrift:
    def test_fixed_tempo_reports_no_drift(self):
        """REGRESSION: the old std-of-intervals test fired on any steady track."""
        times, _, _ = beat_times()
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        has_drift, pct = ba.analyze_tempo_drift(times, p)

        assert has_drift is False
        assert pct < ba.DRIFT_THRESHOLD * 100

    def test_a_break_in_the_beats_is_not_drift(self):
        """REGRESSION: a 34 s gap used to blow the spread up to 3.364."""
        times, _, _ = beat_times(drop=0.15, gap=(200, 272))
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        has_drift, _pct = ba.analyze_tempo_drift(times, p)

        assert has_drift is False

    def test_genuine_acceleration_is_detected(self):
        times = accelerating_beats(124.0, 130.0, dur=290.0)
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        has_drift, pct = ba.analyze_tempo_drift(times, p)

        assert has_drift is True
        assert pct > ba.DRIFT_THRESHOLD * 100

    def test_drift_is_not_measured_through_the_global_grid(self):
        """REGRESSION for the bug that made drift undetectable on real material.

        `fit_beat_grid` rejects, as outliers, exactly the beats that drifted away
        from its single uniform grid. Measuring drift on the SURVIVORS therefore
        reads a ramping track as nearly steady — they are near-uniform by
        construction. On a real 124 -> 132 set that understatement was enough to
        fall under the threshold and write no warp map at all.

        This pins the gap: the grid-filtered view must understate the true spread
        by a wide margin, while the grid-free measurement on all detections must
        recover it.
        """
        times = accelerating_beats(124.0, 132.0, dur=180.0, drop=0.25,
                                   jitter=0.018, seed=5)
        p, t0, k, inliers = ba.fit_beat_grid(times)

        # The grid really does throw a large share of a drifting track away.
        assert inliers.sum() < 0.85 * len(times)

        true_spread = abs(60.0 / 124.0 - 60.0 / 132.0) / (60.0 / 128.0) * 100.0
        _has, full_pct = ba.analyze_tempo_drift(times, p)
        _has_in, inlier_pct = ba.analyze_tempo_drift(times[inliers], p)

        # Grid-free recovers most of the real ramp; the survivors-only view,
        # measured the old way through grid indices, sees a fraction of it.
        assert full_pct > 0.5 * true_spread
        assert _has is True

        ideal = p * k[inliers] + t0
        old_style_pct = _drift_through_grid(ideal, k[inliers])
        assert old_style_pct < 0.1 * true_spread

    def test_ideal_grid_times_measure_nothing(self):
        """Ideal grid times are uniform by construction, so they carry no drift.

        Kept as a wiring guard: if a future edit feeds the fitted grid back in
        instead of the raw detections, drift silently becomes undetectable.
        """
        times = accelerating_beats(124.0, 130.0, dur=290.0)
        p, t0, k, inliers = ba.fit_beat_grid(times)

        ideal = p * k[inliers] + t0
        assert ba.analyze_tempo_drift(ideal, p) == (False, 0.0)
        assert ba.analyze_tempo_drift(times, p)[0] is True


def _drift_through_grid(times, grid_k, segment_sec=30.0, min_segment_beats=16):
    """The superseded grid-index drift measurement, kept only for the regression.

    Reproduces what analyze_tempo_drift used to do so the test can show how much
    signal that approach discards. Not used in production.
    """
    times = np.asarray(times, dtype=float)
    ks = np.asarray(grid_k, dtype=float)
    seg_ids = ((times - times[0]) / segment_sec).astype(int)
    periods = []
    for seg in np.unique(seg_ids):
        sel = seg_ids == seg
        if int(sel.sum()) < min_segment_beats:
            continue
        if ks[sel].max() - ks[sel].min() < min_segment_beats:
            continue
        periods.append(ba._fit_line(ks[sel], times[sel])[0])
    if len(periods) < 2:
        return 0.0
    periods = np.array(periods)
    return float((periods.max() - periods.min()) / np.median(periods) * 100.0)


class TestWarpMap:
    def test_steady_tempo_produces_no_map(self):
        """A steady track must stay unwarped — warping it can only add error."""
        times, _, _ = beat_times()
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        assert ba.build_warp_map(times, p) == []

    def test_drifting_track_produces_a_monotonic_map(self):
        times = accelerating_beats(124.0, 132.0, dur=180.0, drop=0.25,
                                   jitter=0.018, seed=5)
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        markers = ba.build_warp_map(times, p)

        assert len(markers) >= 2
        beats = [m[0] for m in markers]
        samples = [m[1] for m in markers]
        assert all(b >= 0 for b in beats)
        assert all(s >= 0 for s in samples)
        assert all(beats[i] > beats[i - 1] for i in range(1, len(beats)))
        assert all(samples[i] > samples[i - 1] for i in range(1, len(samples)))

    def test_map_tempo_follows_the_ramp(self):
        """The implied tempo between markers must rise across an accelerando."""
        times = accelerating_beats(124.0, 132.0, dur=180.0, seed=5)
        p, _t0, _k, _inliers = ba.fit_beat_grid(times)
        markers = ba.build_warp_map(times, p)

        implied = [60.0 * (markers[i][0] - markers[i - 1][0])
                   / (markers[i][1] - markers[i - 1][1])
                   for i in range(1, len(markers))]
        assert implied[-1] > implied[0] + 3.0
        assert 122.0 < implied[0] < 127.0
        assert 129.0 < implied[-1] < 134.0


class TestOctaveResolution:
    @pytest.mark.parametrize("raw,expected", [
        (62.0, 124.0), (124.0, 124.0), (248.0, 124.0),
        (85.0, 85.0), (170.0, 170.0), (171.0, 85.5), (84.0, 168.0),
    ])
    def test_resolves_into_range(self, raw, expected):
        assert ba.resolve_bpm_octave(raw) == pytest.approx(expected)

    def test_non_positive_bpm_falls_back(self):
        assert ba.resolve_bpm_octave(0.0) == 120.0
        assert ba.resolve_bpm_octave(-5.0) == 120.0


class TestPhaseVerdict:
    def test_chance_is_chance_level(self):
        assert ba.phase_verdict(ba.PHASE_CHANCE, "heuristic")[0] == "chance-level"

    def test_each_family_is_judged_against_its_own_ceiling(self):
        # The same raw number is a strong heuristic result and a weak trained
        # one: 0.45 is 44% of the way to the heuristic's 0.70 ceiling but only
        # 27% of the way to a trained detector's 1.0.
        assert ba.phase_verdict(0.45, "heuristic")[0] == "ok"
        assert ba.phase_verdict(0.45, "beat_this")[0] == "weak"
        assert ba.phase_verdict(0.99, "beat_this")[0] == "ok"


def render_stems(directory, bpm=124.0, bars=24, pre_roll=1.7, sr=22050):
    """Write a small synthetic stem set. Returns (period, pre_roll, true_phase).

    Kick on every beat, snare on 2 and 4, and a bass note that changes on each
    bar line — the harmonic cue is the only one that can tell beat 1 from beat
    3, so without a chord change there is nothing for phase detection to find.

    `true_phase` is the grid phase of a real downbeat. Grid indices are counted
    from the origin nearest zero, so beat j sits at index j + floor(pre_roll /
    period), and the bar lines are that offset modulo 4.
    """
    import soundfile as sf

    period = 60.0 / bpm
    n_beats = bars * 4
    n = int((pre_roll + period * (n_beats + 1)) * sr)
    rng = np.random.RandomState(0)

    def blank():
        return np.zeros(n, dtype=np.float32)

    def place(buf, sig, t, gain=1.0):
        start = int(t * sr)
        end = min(n, start + len(sig))
        if end > start:
            buf[start:end] += sig[:end - start] * gain

    kick_len = int(0.10 * sr)
    freq = np.linspace(120.0, 45.0, kick_len)
    kick = (np.sin(2 * np.pi * np.cumsum(freq) / sr)
            * np.exp(-np.linspace(0, 12, kick_len))).astype(np.float32)

    snare_len = int(0.12 * sr)
    snare = ((rng.uniform(-1, 1, snare_len) * 0.8
              + np.sin(2 * np.pi * 190 * np.arange(snare_len) / sr) * 0.5)
             * np.exp(-np.linspace(0, 11, snare_len))).astype(np.float32)

    kick_buf, snare_buf, bass_buf = blank(), blank(), blank()
    bass_len = int(period * 0.9 * sr)
    for i in range(n_beats):
        t = pre_roll + i * period
        place(kick_buf, kick, t, 0.95 + 0.05 * rng.rand())
        if i % 4 in (1, 3):
            place(snare_buf, snare, t, 0.7)
        if i % 4 == 0:
            note = [55.0, 73.4, 61.7, 82.4][(i // 4) % 4]
            tone = (np.sin(2 * np.pi * note * np.arange(bass_len) / sr)
                    * np.exp(-np.linspace(0, 2.5, bass_len))).astype(np.float32)
            place(bass_buf, tone, t, 0.6)

    for name, buf in (("drums_kick", kick_buf), ("drums_snare", snare_buf),
                      ("bass", bass_buf)):
        peak = max(1e-9, float(np.abs(buf).max()))
        # PCM_16, not the float32 soundfile would infer from the array: several
        # readers in this chain handle float WAV poorly.
        sf.write(os.path.join(directory, name + ".wav"),
                 buf / peak * 0.9, sr, subtype="PCM_16")

    return period, pre_roll, int(pre_roll // period) % 4


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Rendered stems plus injected ground-truth detections.

    The beat tracker is bypassed deliberately. aubio locks onto three times the
    period on a synthetic loop — it needs the timing irregularity of real
    playing — so leaving it in would test aubio's weakness rather than this
    module's wiring. Real aubio output is covered by TestRealStems; everything
    downstream of detection is covered here, on audio that is really on disk and
    really read by the phase detector.
    """
    def build(bpm=124.0, bars=24, drop=0.0, gap=None, jitter=0.004):
        stems = tmp_path / "synthetic_stems"
        stems.mkdir(exist_ok=True)
        period, pre_roll, true_phase = render_stems(str(stems), bpm=bpm, bars=bars)
        times, _, _ = beat_times(bpm=bpm, n_beats=bars * 4, pre_roll=pre_roll,
                                 drop=drop, gap=gap, jitter=jitter)
        monkeypatch.setattr(ba, "estimate_aubio_beats", lambda path: times)
        return str(stems), times, period, pre_roll, true_phase
    return build


class TestSyntheticAudioEndToEnd:
    def test_pipeline_recovers_tempo_and_lands_on_a_bar_line(self, synthetic):
        stems, _times, _period, pre_roll, _phase = synthetic()

        res = ba.analyze_alignment(stems)

        assert res["bpm"] == pytest.approx(124.0, abs=0.01)
        assert res["primary_stem"] == "drums_kick.wav"
        assert res["has_drift"] is False
        landed = (res["downbeat_sec"] / res["period"]
                  + res["start_offset_beats"]) % 4.0
        assert min(landed, 4.0 - landed) < 1e-3
        # The anchor must sit on an actual beat, not in the pre-roll silence.
        assert res["downbeat_sec"] >= pre_roll - 0.05

    def test_phase_detection_finds_the_real_bar_line(self, synthetic):
        """The bass changes on the bar line, so the harmonic cue has an answer.

        Pins `detector="heuristic"` because this exercises the HEURISTIC's
        harmonic cue specifically. It relied on the default before, so changing
        that default to madmom broke it — a test of one backend should never be
        at the mercy of which backend is currently default.
        """
        stems, _times, _period, _pre_roll, true_phase = synthetic(bars=32)

        res = ba.analyze_alignment(stems, detector="heuristic")

        assert res["detector"] == "heuristic"
        assert "bass" in res["harmonic_source"]
        assert res["downbeat_phase"] == true_phase
        assert res["downbeat_confidence"] > ba.PHASE_CHANCE

    def test_the_default_detector_is_the_heuristic_and_actually_answers(self, synthetic):
        """Pins the default, and that it ENGAGES rather than silently degrading.

        `detect_downbeat_phase` falls back to the heuristic on any backend
        failure, so asserting only the phase would pass even if the named
        backend had never run. Assert the backend that ANSWERED.

        The default is the heuristic because it is right and fast here, not
        because it scores well: on the reference stems it reads 0.302 against
        madmom's 0.541 yet picks the phase the structural check agrees with,
        while madmom picks one the structural check has to override. Same final
        answer, 8.31 s against 31.19 s.
        """
        stems, _times, _period, _pre_roll, true_phase = synthetic(bars=32)

        res = ba.analyze_alignment(stems)

        assert ba.DEFAULT_DETECTOR == "heuristic"
        assert res["detector"].startswith("heuristic")
        assert res["downbeat_phase"] == true_phase

    def test_madmom_is_still_reachable_and_answers_when_asked(self, synthetic):
        """Not the default any more, but it must not rot into a dead option.

        Without this, madmom could break and the suite would stay green — the
        fallback would quietly serve heuristic results under its name.
        """
        pytest.importorskip("madmom")
        stems, _times, _period, _pre_roll, true_phase = synthetic(bars=32)

        res = ba.analyze_alignment(stems, detector="madmom")

        assert res["detector"].startswith("madmom")
        assert res["downbeat_phase"] == true_phase

    def test_structural_tiebreaker_declines_on_a_tie(self, synthetic):
        """REGRESSION: a dead heat must NOT override the detector.

        The synthetic returns boundary votes [28, 0, 0, 28] — phases 0 and 3
        exactly level. Against a uniform-over-four null both score p = 4.9e-05,
        so a single significance gate passes and `argmax` hands back the lower
        index, overriding a correct phase-3 answer with a wrong phase-0 one.
        Comparing winner against runner-up reads the same votes as p = 0.55.

        Asserts the OUTCOME (no override, detector's answer survives) rather than
        the p-value, so it still guards if the gate is reimplemented.
        """
        stems, _t, _p, _pre, true_phase = synthetic(
            bars=32, drop=0.15, gap=(60, 92))

        res = ba.analyze_alignment(stems)

        assert res["downbeat_phase"] == true_phase
        votes = res["structural_votes"]
        top2 = sorted(votes, reverse=True)[:2]
        if top2[0] == top2[1]:            # the tie this test exists for
            assert res["phase_override"] is None or "say phase" not in \
                res["phase_override"]

    def test_structural_tiebreaker_can_be_disabled(self, synthetic):
        stems, _t, _p, _pre, _phase = synthetic(bars=32)

        res = ba.analyze_alignment(stems, use_structural=False)

        assert res["structural_phase"] is None
        assert res["phase_override"] is None
        assert "structural" not in res["detector"]

    def test_dropped_beats_do_not_move_the_bar_line(self, synthetic):
        """REGRESSION, end to end: 15% dropped plus an 8-bar break."""
        stems, _times, _period, _pre, true_phase = synthetic(
            bars=32, drop=0.15, gap=(60, 92))

        res = ba.analyze_alignment(stems)

        assert res["bpm"] == pytest.approx(124.0, abs=0.02)
        assert res["downbeat_phase"] == true_phase
        landed = (res["downbeat_sec"] / res["period"]
                  + res["start_offset_beats"]) % 4.0
        assert min(landed, 4.0 - landed) < 1e-3

    def test_drift_is_given_every_raw_detection(self, synthetic, monkeypatch):
        """REGRESSION, wiring half: guards the call site, not just the function.

        Two ways the call site can silently destroy the drift signal, and this
        catches both. Passing the IDEAL grid times measures nothing, because they
        are uniform by construction. Passing only the grid INLIERS understates
        the drift, because the grid has already rejected the beats that drifted —
        that one shipped, and made a 124 -> 132 BPM set report as steady.

        So the assertion is exact on both counts: every value handed to the drift
        measurement must be a real detection, and ALL of them must be there.
        """
        stems, times, _period, _pre, _phase = synthetic()

        captured = {}
        real_fn = ba.analyze_tempo_drift

        def spy(beat_times, seed_period, **kwargs):
            captured["times"] = np.asarray(beat_times, dtype=float)
            return real_fn(beat_times, seed_period, **kwargs)

        monkeypatch.setattr(ba, "analyze_tempo_drift", spy)
        ba.analyze_alignment(stems)

        assert len(captured.get("times", [])) > 0
        assert np.isin(captured["times"], times).all()
        assert len(captured["times"]) == len(times)

    def test_tempo_override_is_honoured_end_to_end(self, synthetic):
        stems, _times, _period, _pre, _phase = synthetic()

        res = ba.analyze_alignment(stems, override_tempo=126.0)

        assert res["bpm"] == 126.0
        assert res["period"] == pytest.approx(60.0 / 126.0, abs=1e-6)
        landed = (res["downbeat_sec"] / res["period"]
                  + res["start_offset_beats"]) % 4.0
        assert min(landed, 4.0 - landed) < 1e-3

    def test_unknown_detector_is_rejected(self, synthetic):
        stems, _times, _period, _pre, _phase = synthetic()
        with pytest.raises(ValueError, match="unknown detector"):
            ba.analyze_alignment(stems, detector="nope")


@pytest.mark.skipif(not os.path.isdir(STEMS_DIR), reason="test stems not present")
@pytest.mark.skipif(not HAVE_AUBIO, reason="aubiotrack not installed")
class TestRealStems:
    """Integration check against the reference stems (see STEMS_DIR).

    The track is 125.000 BPM. Do NOT relax this to 126.71 or 127 — see
    `_refine_period`'s docstring: the filename, the cached .asd and any median
    of aubio's intervals all carry the same 1.4% low bias, and none of them is
    an independent measurement.
    """

    @pytest.fixture
    def result(self, real_stems_result):
        return real_stems_result

    def test_tempo_is_125(self, result):
        assert result["bpm"] == pytest.approx(125.0, abs=0.05)

    def test_no_drift(self, result):
        assert result["has_drift"] is False
        assert result["drift_pct"] < ba.DRIFT_THRESHOLD * 100

    def test_almost_every_detection_fits_the_grid(self, result):
        assert result["inlier_count"] / result["beat_count"] > 0.95

    def test_anchor_is_a_grid_index_carrying_the_detected_phase(self, result):
        """The anchor sits on a grid BEAT, and that beat carries the phase.

        This used to assert a whole number of BARS from `grid_t0`, which is a
        stronger claim than the code makes and only held by luck: `grid_t0` is
        the grid's phase anchor, an arbitrary beat, NOT a downbeat. Bars come out
        integral only when `downbeat_phase == 0`, which is what the heuristic
        happened to return. Switching the default detector to madmom moved the
        phase to 2 and the assertion broke on correct output — the anchor was
        beat 66, an exact grid index, just not a multiple of four.

        The real invariants are below: integral in BEATS, and the index's bar
        position equals the phase that was detected.
        """
        beats = (result["downbeat_sec"] - result["grid_t0"]) / result["period"]
        assert abs(beats - round(beats)) < 1e-3
        assert round(beats) % 4 == result["downbeat_phase"]

    def test_offset_lands_on_a_bar_line(self, result):
        landed = (result["downbeat_sec"] / result["period"]
                  + result["start_offset_beats"]) % 4.0
        assert min(landed, 4.0 - landed) < 1e-3

    def test_contract_with_stems2live(self, result):
        # stems2live reads exactly these; renaming one breaks alignment silently.
        for key in ("bpm", "downbeat_sec", "start_offset_beats", "primary_stem",
                    "has_drift", "downbeat_confidence", "detector",
                    "phase_source"):
            assert key in result


class TestAlignmentCache:
    """The cache's invalidation contract.

    A cache that never invalidates is indistinguishable from a correct one right
    up until it serves a stale answer, so every one of these is a test that it
    STOPS being used, not that it gets used.
    """

    def _paths(self, tmp_path):
        stems = tmp_path / "stems"
        stems.mkdir()
        for name in ("drums_kick", "bass", "synths"):
            (stems / (name + ".flac")).write_bytes(b"x" * 64)
        return {n: str(stems / (n + ".flac")) for n in ("drums_kick", "bass", "synths")}

    def test_roundtrip_returns_what_was_written(self, tmp_path):
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 125.003})
        assert ba.read_cache(d, fp) == {"bpm": 125.003}

    def test_missing_cache_is_a_miss_not_an_error(self, tmp_path):
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        assert ba.read_cache(d, ba._cache_fingerprint(paths, None, "heuristic")) is None

    def test_corrupt_cache_is_a_miss_not_a_crash(self, tmp_path):
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 1.0})
        with open(os.path.join(d, ba.CACHE_NAME), "w") as fh:
            fh.write("{not json")
        assert ba.read_cache(d, fp) is None

    def test_changed_stem_mtime_invalidates(self, tmp_path):
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 125.003})
        os.utime(paths["bass"], (1_600_000_000, 1_600_000_000))
        after = ba._cache_fingerprint(paths, None, "heuristic")
        assert ba.read_cache(d, after) is None

    def test_changed_stem_size_invalidates(self, tmp_path):
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 125.003})
        with open(paths["bass"], "ab") as fh:
            fh.write(b"more")
        assert ba.read_cache(d, ba._cache_fingerprint(paths, None, "heuristic")) is None

    @pytest.mark.parametrize("kw", [
        {"override_tempo": 126.0},
        {"detector": "beat_this"},
            ])
    def test_every_result_changing_parameter_is_in_the_key(self, tmp_path, kw):
        """A parameter that changes the answer but not the key is a stale-result bug."""
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        base = dict(override_tempo=None, detector="heuristic")
        fp = ba._cache_fingerprint(paths, **base)
        ba.write_cache(d, fp, {"bpm": 125.003})
        assert ba.read_cache(d, ba._cache_fingerprint(paths, **{**base, **kw})) is None

    def test_version_bump_invalidates(self, tmp_path):
        """Bumping CACHE_VERSION must retire every cache written before it."""
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 125.003})
        assert ba.read_cache(d, {**fp, "version": fp["version"] + 1}) is None

    def test_fingerprint_survives_a_directory_rename(self, tmp_path):
        """yt2stems renames the stems dir right after tagging the BPM.

        The cache is written before that rename and read after it, so a
        fingerprint carrying absolute paths would be invalidated by the very
        step that produced it.
        """
        paths = self._paths(tmp_path)
        d = os.path.dirname(next(iter(paths.values())))
        fp = ba._cache_fingerprint(paths, None, "heuristic")
        ba.write_cache(d, fp, {"bpm": 125.003})

        renamed = os.path.join(os.path.dirname(d), "renamed_125bpm_stems")
        os.rename(d, renamed)
        moved = {n: os.path.join(renamed, os.path.basename(p)) for n, p in paths.items()}
        assert ba.read_cache(renamed, ba._cache_fingerprint(moved, None, "heuristic")) \
            == {"bpm": 125.003}

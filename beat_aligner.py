#!/usr/bin/env python3
"""Joint Beat & Downbeat Detector and Alignment Engine for yt2live.

Fits an explicit beat GRID to the raw detections, then does downbeat phase
detection, stability-gated anchor selection and tempo drift analysis against
that grid rather than against the detection array.

The grid is the whole point. Aubio drops beats — on the one real track here, 21
of 469 intervals are multi-beat gaps, including a 72-beat gap across a
breakdown. Any code that counts bar position with `i % 4` over the detection
array is therefore wrong: a single dropped beat rotates the bar phase for the
entire remainder of the track. Every step below indexes by the fitted grid
index `k`, which survives gaps.

Two signals, two jobs: TEMPO is fitted to the kick stem, where the pulse is
cleanest; PHASE is detected on the summed mix, where the harmonic information
that distinguishes bar 1 from bar 3 actually lives.
"""

import os
import sys

# Re-exec with local .venv python if present to ensure librosa/numpy environment
_real_script = os.path.realpath(__file__)
_venv_py = os.path.join(os.path.dirname(_real_script), ".venv", "bin", "python")
if os.path.exists(_venv_py) and sys.executable != _venv_py and "VIRTUAL_ENV" not in os.environ:
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import json
import math
import subprocess
import numpy as np

# NO blanket warnings filter. There used to be a `warnings.filterwarnings("ignore")`
# here, and it silently swallowed librosa's "Empty filters detected in mel
# frequency basis" — which was true, and is why _band_onset_envelope exists.

# PyPI imports installed in .venv (librosa, soundfile, scipy)
try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

# A residual larger than this fraction of the period means the detection does
# not belong to this grid — most often a beat sitting on a half-grid slot after
# a downward octave correction, which must be rejected rather than rounded.
OUTLIER_TOL = 0.25

# Downbeat backends. `heuristic` needs nothing but librosa and is the default;
# the other two are trained models and are opt-in because of what they drag in.
# beat_this is deliberately absent. It was listed as a backend but never once
# run here: it needs torch, torch came in only while evaluating allin1 (which
# does not work — see the README), and removing that 614 MB took beat_this's
# only dependency with it. Advertising a backend nobody has exercised is worse
# than not offering it, since the dispatch degrades to the heuristic silently
# and you would never know the model had not run. madmom already covers the
# trained-model case. To bring it back: install torch plus
# `beat-this tqdm einops soxr rotary-embedding-torch`, restore a
# _beat_this_downbeats that returns downbeat times in seconds, and add it here.
DETECTORS = ("heuristic", "madmom")

# heuristic by default, and this reverses an earlier call worth recording.
#
# madmom was made the default on the strength of its confidence: 0.541 against
# the heuristic's 0.302 on the reference track (0.25 is chance at four phases).
# That number is a VOTE SHARE, and it was attached to the wrong answer — madmom
# chose phase 2, the heuristic chose phase 0, and the structural check agreed
# with the heuristic at p=6.9e-03 and overrode madmom. Confidently wrong is
# worse than uncertain and right.
#
# Measured end to end on the reference stems, both with the structural
# tiebreaker active:
#
#     heuristic   phase 0, offset 3.5062   8.31 s   structural AGREES
#     madmom      phase 0, offset 3.5062  31.19 s   structural had to OVERRIDE
#
# Same answer, 3.75x the wall clock, and it needs a git-main install that the
# PyPI release cannot provide. madmom stays available for material the heuristic
# is not tuned for — it weighs kick and snare energy, which is a four-on-the-
# floor assumption — but it is not what most tracks through here should pay for.
DEFAULT_DETECTOR = "heuristic"

# Where a completed analysis is parked, inside the stems directory so it travels
# with the stems (including through yt2stems's BPM rename). Bump CACHE_VERSION
# whenever a change to this module would alter the numbers — an unversioned
# cache would serve stale answers with no way to tell.
CACHE_NAME = "alignment.json"
CACHE_VERSION = 3   # 3: phrase alignment removed; no phrase_bars in the key

# Analysis sample rate for the summed mix. madmom resamples internally anyway,
# internally anyway, and the heuristic's top band is 1 kHz, so nothing above
# this is used by anything downstream.
ANALYSIS_SR = 22050

# STFT geometry for the onset envelopes and chroma.
HOP_LENGTH = 512
N_FFT = 2048

# Band edges for the heuristic's onset envelopes: kick fundamental, and
# snare/clap body well clear of it.
KICK_MAX_HZ = 150.0
SNARE_MIN_HZ = 1000.0

# Chance level for a 4-phase decision. Confidence is reported on a comparable
# scale by every backend (share of evidence for the winning phase), so 0.25 means
# "no idea" regardless of which detector produced it.
PHASE_CHANCE = 0.25


# Heuristic cue weights. Each cue is normalised to sum 1 across the four phases
# before weighting, so these are directly comparable.
#
# W_HARM is the largest deliberately. The other three cues are all NEARLY
# SYMMETRIC under phase -> phase+2, which is why the old scorer picked beat 3 as
# often as beat 1: with `1.5*low[p] + low[p+2]`, phases p and p+2 differ only by
# `0.5*(low[p] - low[p+2])`, and on four-on-the-floor those two are equal by
# construction. The snare term is worse — a standard backbeat puts equal energy
# on 2 and 4, so it cancels exactly. Harmonic change is the only cue here that
# actually distinguishes the start of a bar from its middle, so it has to carry
# the decision.
W_KICK_1 = 1.5
W_KICK_3 = 1.0
W_SNARE = 0.8
W_HARM = 2.0

# Relative spread between segment tempos above which drift is real. The test
# track measures 0.17% — that is aubio's per-segment timing noise, not tempo
# movement — so 0.5% clears the noise floor 3x while still catching a live set
# that speeds up (a 124 -> 130 BPM ramp spreads ~2.4%).
DRIFT_THRESHOLD = 0.005

# Minimum gap, as a fraction of a beat, for a detection to count as a NEW beat
# when local indices are accumulated. Aubio interleaves half-grid detections
# ~0.5 beat from the real ones; anything under this is one of those and is
# dropped. Real beats survive with margin: 35 ms of jitter at each end still
# leaves a true one-beat gap above 0.85 of a beat.
SUB_BEAT_MIN = 0.75

# Residual tolerance for trimming a per-segment tempo fit, as a fraction of the
# segment's own period. TIGHTER than OUTLIER_TOL on purpose: across ~30 s the
# tempo barely moves, so a real beat cannot sit far off the local line and
# anything that does is a mistracked detection. At 0.25 the test track's intro —
# a 6.6 s run where aubio tracks a different pulse — stays inside the fit and
# drags that segment to 125.98 BPM against 125.00 everywhere else, reading as
# 0.81% drift on a steady track. At 0.15 the run is rejected and it reads 0.11%.
SEGMENT_TOL = 0.15

# A pulse means consecutive intervals cluster near one value. Measured on the
# raw intervals, so it stays true of a track that speeds up — where no single
# grid fits, but the local spacing is still regular. Whole-track coherence
# cannot be used for this: it reads 0.23 on a 124->130 BPM ramp, below the 0.40
# that randomly placed onsets reach, so the two are not separable that way.
PULSE_TOL = 0.15
MIN_PULSE_FRACTION = 0.4

# Half-width of the window used to read onset strength at a grid time. The
# ideal grid time falls between onset-strength frames (hop 512 ~ 11.6 ms) and a
# transient can sit a frame or two off it, so point sampling reads the gaps
# between peaks: on the test stem it returns 0.02 where the peak is 2.15. At
# 60 ms this still cannot reach a neighbouring beat, which is ~480 ms away.
ONSET_WINDOW_SEC = 0.06


def estimate_aubio_beats(path):
    """Extract raw beat timestamps from aubiotrack CLI."""
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            ["aubiotrack", "-i", path],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600
        ).stdout
        beats = [float(x) for x in out.split() if x.strip()]
        return np.array(beats) if len(beats) >= 8 else None
    except Exception:
        return None


def resolve_bpm_octave(bpm, min_bpm=85.0, max_bpm=170.0):
    """Disambiguate half-time / double-time BPM into standard electronic range."""
    if bpm <= 0:
        return 120.0
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    return bpm


def _fit_line(k, t):
    """Least-squares (period, t0) for t ~ period * k + t0."""
    A = np.vstack([np.asarray(k, dtype=float), np.ones(len(k))]).T
    period, t0 = np.linalg.lstsq(A, np.asarray(t, dtype=float), rcond=None)[0]
    return float(period), float(t0)


def _comb_phase(beats, period):
    """Phase coherence of every detection against a candidate period.

    Each beat becomes a unit vector at angle 2*pi*b/period. If the period is
    right, all the vectors point the same way and the mean has length ~1; if it
    is wrong they fan out and cancel. The angle of that mean is the grid's
    phase, so this recovers period AND anchor without trusting any single beat.
    """
    z = np.exp(2j * np.pi * np.asarray(beats, dtype=float) / period).mean()
    return float(np.abs(z)), float((np.angle(z) / (2.0 * np.pi) * period) % period)


def _refine_period(beats, seed, tol=0.06):
    """Sharpen a seeded period by maximising comb phase coherence.

    The median beat interval is a BIASED estimator here and must not be used as
    the answer. On the test stem the median interval is 0.473511 s (126.713 BPM)
    while the audio is 0.480000 s (125.000 BPM). Aubio emits predicted beats at
    its own internal tempo hypothesis and only periodically resyncs to a real
    transient, so its interval distribution spikes at that hypothesis (the
    median) with a tail of longer correction intervals — the MEAN, 0.4792, sits
    near the truth and the median does not. Any estimator that takes a median or
    a mode of aubio's intervals inherits aubio's error, however robustly it is
    computed. Over 591 beats the 1.4% bias accumulates 3.8 s, about eight whole
    beats.

    DO NOT "correct" this back to 126.71 or 127. Three sources look like
    independent confirmation of the wrong value and none of them is:
      - the `_127bpm` in the filename is yt2stems' own rounded detection, which
        comes from this same script (see yt2stems.sh detect_bpm);
      - a cached `drums_kick.flac.asd` sits next to the stem, and Live's warp
        marker in it reads 126.713 — matching the aubio median to six decimals,
        i.e. a stored echo of an earlier run, not a measurement. Live's own Set
        tempo disagreed with that marker, which is what exposed it as stale;
      - any "robust" re-derivation from aubio's intervals returns the median
        again, because 96% of the intervals it accepts are 1x the period.

    Coherence instead uses the whole span: the correct period is the only one in
    the neighbourhood that keeps 470 beats in phase over 280 s (coherence 0.951
    against 0.060 for the median; residual RMS 34.6 ms against 131.5 ms). Two
    checks that touch neither aubio nor Live agree — onset-envelope
    autocorrelation gives 0.480004 s at the 64-beat lag and 0.480002 s at the
    256-beat lag, and 414 kick transients peak-picked straight off the waveform
    give 0.47999-0.48004 s. Predicted beats land a mean 4.2 ms from a real
    transient, against 119.3 ms — pure chance — for the median.

    The search stays within +/-tol of the seed so it cannot slide onto the
    half-time or double-time peak; that is `resolve_bpm_octave`'s job.
    """
    beats = np.asarray(beats, dtype=float)
    span = float(beats[-1] - beats[0])
    if span <= 0:
        return seed, 0.0

    # A coherence peak is ~seed^2/span wide, so step at an eighth of that to be
    # sure of landing on it; cap the candidate count for very long recordings.
    step = max(0.125 * seed * seed / span, 2.0 * tol * seed / 20000.0)
    best = (-1.0, seed, 0.0)
    for period in np.arange(seed * (1.0 - tol), seed * (1.0 + tol), step):
        r, phase = _comb_phase(beats, period)
        if r > best[0]:
            best = (r, float(period), phase)
    return best[1], best[2]


def fit_beat_grid(beats, force_period=None, max_iter=8):
    """Fit beats to a uniform grid `period * k + t0`; returns (period, t0, k, inliers).

    Why fit at all: the detections are neither dense nor uniform, so their array
    positions carry no musical meaning. Assigning each detection an integer grid
    index makes bar position (`k % 4`) survive dropped beats and breakdowns.

    The period must be pinned by `_refine_period` BEFORE any of this. Left to
    discover the period itself from a median seed, this loop is a runaway with
    local minima: off-grid detections drag the period, which re-assigns more
    indices, which drags it further, and on the test stem it walks away shedding
    inliers on every pass.

    Least squares is then run on INLIERS ONLY and re-trimmed each pass. That
    trimming is what makes a downward octave correction safe — on a double-time
    detection half the beats land on half-grid slots, and fitting them too pulls
    the result to 123.69 BPM where trimming holds it at 124.00. They have to be
    dropped, not rounded onto the nearest whole slot.

    `force_period` pins the spacing (the --tempo override) and fits only the
    anchor, so anchor / stability / drift stay consistent with the forced tempo.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        raise ValueError("need at least 8 beats to fit a grid, got %d" % len(beats))

    if force_period is not None:
        period = float(force_period)
        if period <= 0:
            raise ValueError("forced period must be positive, got %r" % force_period)
        _, t0 = _comb_phase(beats, period)
    else:
        intervals = np.diff(beats)
        # Median over plausible single-beat intervals only: gaps are whole
        # multiples of the period and would drag a plain median upward. This is
        # a SEED for the coherence search, never the reported period.
        valid = intervals[(intervals > 0.3) & (intervals < 0.85)]
        seed = float(np.median(valid)) if len(valid) else float(np.median(intervals))
        if not np.isfinite(seed) or seed <= 0:
            raise ValueError("could not seed a beat period from %d beats" % len(beats))

        # Reject input that has no pulse in it at all, before it can be dressed
        # up as a tempo. Gaps are fine (they just miss the band), so this asks
        # only that a decent share of the spacings agree with each other.
        pulse = float(np.mean(np.abs(intervals - seed) <= PULSE_TOL * seed))
        if pulse < MIN_PULSE_FRACTION:
            raise ValueError(
                "detections do not describe a steady pulse (only %.0f%% of "
                "intervals cluster near %.3f s); no usable beat grid"
                % (pulse * 100.0, seed))

        # Sharpen the period at the octave the detections are actually AT, then
        # octave-correct. Coherence has to be measured where the data lives: on
        # a double-time detection every second beat sits on a half slot of the
        # corrected period, so the unit vectors cancel in pairs and coherence
        # reads ~0.01 on perfectly good beats. Correcting afterwards is a pure
        # multiplication that leaves the phase intact, so the grid is still
        # BUILT at the final resolution — indices and least squares below all
        # run at the corrected period.
        period, t0 = _refine_period(beats, seed)
        period = 60.0 / resolve_bpm_octave(60.0 / period)
        t0 = t0 % period

    k = np.round((beats - t0) / period).astype(int)
    inliers = np.abs(beats - (period * k + t0)) <= OUTLIER_TOL * period
    for _ in range(max_iter):
        if int(inliers.sum()) < 8:
            raise ValueError("only %d of %d beats fit a uniform grid"
                             % (inliers.sum(), len(beats)))
        if force_period is None:
            period, t0 = _fit_line(k[inliers], beats[inliers])
        else:
            t0 = float(np.mean(beats[inliers] - period * k[inliers]))
        k_new = np.round((beats - t0) / period).astype(int)
        inliers_new = np.abs(beats - (period * k_new + t0)) <= OUTLIER_TOL * period
        stable = np.array_equal(k_new, k) and np.array_equal(inliers_new, inliers)
        k, inliers = k_new, inliers_new
        if stable:
            break

    if int(inliers.sum()) < 8:
        raise ValueError("grid refit left only %d usable beats" % inliers.sum())

    return period, t0, k, inliers


def _band_onset_envelope(y, sr, fmin=None, fmax=None):
    """Half-wave-rectified spectral flux over one frequency band.

    Replaces `librosa.onset.onset_strength(fmax=150)`, which routed through a
    mel filterbank: 128 mel bands below 150 Hz leaves 116 of them empty at any
    normal sample rate, so the kick envelope was built from 12 usable bands and
    librosa said so in a warning that the old module-level filter swallowed.

    A plain STFT band sum has no such failure mode — the band edges land on FFT
    bins directly, and 150 Hz at n_fft=2048/22.05k is bin 14, so there is real
    resolution down there.
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    sel = np.ones(len(freqs), dtype=bool)
    if fmin is not None:
        sel &= freqs >= fmin
    if fmax is not None:
        sel &= freqs <= fmax
    if not sel.any():
        return np.zeros(S.shape[1])
    band = S[sel].sum(axis=0)
    return np.maximum(np.diff(band, prepend=band[0]), 0.0)


def bar_chroma_novelty(y, sr, grid_times, grid_k, period):
    """Per-phase harmonic novelty across bar boundaries. Returns a 4-vector summing to 1.

    THIS is the cue that breaks the beat-1 / beat-3 tie, and it has to be
    measured at BAR scale to work. Instantaneous chroma flux sampled at each
    beat does not: measured on this repo's test set it spreads only 0.048 across
    the four phases and points at a backbeat, because at beat resolution it is
    reading bass-note rhythm rather than chord change. Comparing the mean chroma
    of the bar BEFORE a candidate boundary against the bar AFTER it spreads
    0.184 on the same audio and agrees with the snare cue.

    Chroma is averaged per beat first, keyed by grid index `k`, so bars are
    assembled from grid positions and a dropped beat cannot shift a bar window.
    Boundaries missing more than one beat on either side are skipped.

    Expect the phase one step BEFORE the true downbeat to score highly too: its
    windows straddle the real bar line, so a genuine change one beat away still
    registers. The percussive parity cue is what separates those two.
    """
    per_beat = _beat_chroma(y, sr, grid_times, grid_k, period)
    if len(per_beat) < 16:
        return np.full(4, 0.25)

    k_min, k_max = min(per_beat), max(per_beat)
    scores = np.zeros(4)
    for phase in range(4):
        dists = []
        for k0 in range(k_min + 4, k_max - 4):
            if k0 % 4 != phase:
                continue
            before = _window_mean(per_beat, k0 - 4, 4, 3)
            after = _window_mean(per_beat, k0, 4, 3)
            if before is not None and after is not None:
                dists.append(_cosine_distance(before, after))
        scores[phase] = float(np.mean(dists)) if dists else 0.0

    total = scores.sum()
    return scores / total if total > 0 else np.full(4, 0.25)


def _beat_chroma(y, sr, grid_times, grid_k, period):
    """Mean chroma per beat, keyed by grid index `k`.

    Keying by `k` rather than by array position is what lets bar and phrase
    windows be assembled arithmetically: a dropped beat leaves a hole in the
    dict instead of shifting every later beat into the wrong bar.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH)
    frame_times = np.arange(chroma.shape[1]) * HOP_LENGTH / float(sr)

    per_beat = {}
    for k_i, t_i in zip(np.asarray(grid_k), np.asarray(grid_times, dtype=float)):
        a = np.searchsorted(frame_times, t_i)
        b = np.searchsorted(frame_times, t_i + period)
        if b > a:
            per_beat[int(k_i)] = chroma[:, a:b].mean(axis=1)
    return per_beat


def _window_mean(per_beat, k_start, n_beats, min_present):
    """Mean chroma over `n_beats` grid slots from k_start, or None if too sparse."""
    vals = [per_beat[k_start + i] for i in range(n_beats)
            if (k_start + i) in per_beat]
    return np.mean(vals, axis=0) if len(vals) >= min_present else None


def _cosine_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float(a @ b / (na * nb)) if na and nb else 0.0


def _sample_peaks(env, frame_times, grid_times):
    """Peak of `env` in a +/-ONSET_WINDOW_SEC window around each grid time.

    A grid time is a prediction, not a frame boundary, so reading the envelope
    exactly there mostly samples the troughs between peaks — on the test stem
    that is 0.02 against a true peak of 2.15, i.e. scoring rounding noise. The
    window is far too narrow to reach a neighbouring beat (~480 ms away).
    """
    grid_times = np.asarray(grid_times, dtype=float)
    lo = np.searchsorted(frame_times, grid_times - ONSET_WINDOW_SEC)
    hi = np.searchsorted(frame_times, grid_times + ONSET_WINDOW_SEC)
    return np.array([env[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi)])


def _phase_means(values, grid_k):
    """Mean of `values` grouped by bar position, normalised to sum 1."""
    bar_pos = np.asarray(grid_k) % 4
    means = np.array([
        float(np.mean(values[bar_pos == p])) if np.any(bar_pos == p) else 0.0
        for p in range(4)
    ])
    total = means.sum()
    return means / total if total > 0 else np.full(4, 0.25)


def heuristic_downbeat_phase(y, harm_y, sr, grid_times, grid_k, period):
    """Band-energy + bar-novelty phase estimate. Returns (phase, confidence, scores).

    `y` is the full summed mix (percussive cues); `harm_y` is the drums-free
    submix (harmonic cue) and may be None.

    Three cues, measured on this repo's test set:
    - Kick on beats 1 and 3 (low-band onset). Spread 0.039 — nearly useless on
      four-on-the-floor, where every beat carries a kick by construction.
    - Snare/clap on beats 2 and 4 (high-band onset). Spread 0.154 — resolves
      PARITY only: it says which pair of phases holds the backbeat, so the
      downbeat is one of the other two, but it cannot say which.
    - Harmonic novelty across bar boundaries. Spread 0.184 — the only cue that
      separates beat 1 from beat 3, which is why the old scorer landed a
      half-bar out roughly half the time.

    Everything is sampled at the IDEAL grid times and grouped by `k % 4`, so a
    dropped beat cannot rotate the phase of every bar that follows it.
    """
    low_env = _band_onset_envelope(y, sr, fmax=KICK_MAX_HZ)
    high_env = _band_onset_envelope(y, sr, fmin=SNARE_MIN_HZ)

    # Both share the STFT hop, so one frame-time axis covers them; take its
    # length from an envelope rather than from len(y) so it cannot drift out of
    # step with what librosa actually produced.
    frame_times = librosa.frames_to_time(
        np.arange(len(low_env)), sr=sr, hop_length=HOP_LENGTH)

    n_low = _phase_means(_sample_peaks(low_env, frame_times, grid_times), grid_k)
    n_high = _phase_means(_sample_peaks(high_env, frame_times, grid_times), grid_k)

    # Harmonic novelty needs the drums OUT: on the full mix the chroma is
    # dominated by broadband kit transients that fire on every beat, which flattens
    # the cue to a 0.048 spread and points it at a backbeat. Given only the
    # harmonic stems the same measurement spreads 0.184 and is correct.
    if harm_y is not None and len(harm_y):
        n_harm = bar_chroma_novelty(harm_y, sr, grid_times, grid_k, period)
        w_harm = W_HARM
    else:
        n_harm = np.full(4, 0.25)
        w_harm = 0.0

    scores = np.array([
        W_KICK_1 * n_low[p]
        + W_KICK_3 * n_low[(p + 2) % 4]
        + W_SNARE * (n_high[(p + 1) % 4] + n_high[(p + 3) % 4])
        + w_harm * n_harm[p]
        for p in range(4)
    ])

    total = scores.sum()
    phase = int(np.argmax(scores))
    conf = float(scores[phase] / total) if total > 0 else PHASE_CHANCE
    return phase, conf, scores


def _madmom_downbeats(y, sr):
    """Downbeat times (seconds) from madmom's RNN + DBN bar tracker.

    This DOES work on the repo's Python 3.13 venv, contrary to what this
    docstring used to claim. The PyPI release (0.16.1, 2018) does not -- it does
    `from collections import MutableSequence`, removed in 3.10. Three things are
    needed, and all three are non-obvious:

        uv pip install "setuptools<81"        # 81+ dropped pkg_resources
        uv pip install cython
        uv pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom"

    git main (0.17.dev0) has the 3.10+ fixes; --no-build-isolation is required
    because madmom's setup.py imports Cython and numpy directly.

    Worth the trouble: on the reference track this backend reports phase
    confidence 0.541 against the heuristic's 0.302 (0.25 is chance), and it is
    the difference between "bar 1 is a guess" and a usable answer.

    Goes via a temp wav rather than an array: madmom's signal handling wants its
    own Signal type at its own rate, and a file path is the one input every
    madmom version accepts identically.
    """
    import tempfile

    from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                           RNNDownBeatProcessor)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, np.asarray(y, dtype=np.float32), sr)
        activations = RNNDownBeatProcessor()(tmp_path)
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
        tracked = tracker(activations)
        # rows are (time, beat_number); beat 1 is the downbeat
        return np.asarray([t for t, b in tracked if int(b) == 1], dtype=float)
    finally:
        os.unlink(tmp_path)


def phase_from_downbeats(downbeat_times, period, t0):
    """Vote detected downbeats onto grid phases. Returns (phase, confidence, votes).

    The trained detectors produce their own beat grid, which we deliberately do
    NOT use for tempo — `_refine_period`'s comb fit over the kick is the better
    estimate and is already consistent with everything downstream. Only the
    PHASE is taken: each detected downbeat is snapped to the nearest fitted grid
    index and votes for that index's `k % 4`. Downbeats that do not land near a
    grid slot are discarded rather than rounded, on the same reasoning as
    OUTLIER_TOL everywhere else.

    Confidence is the winning phase's share of the votes, so 0.25 is chance and
    1.0 means every detected downbeat agreed.
    """
    votes = np.zeros(4)
    for d in np.asarray(downbeat_times, dtype=float):
        k_float = (d - t0) / period
        k_round = round(k_float)
        if abs(k_float - k_round) <= OUTLIER_TOL:
            votes[int(k_round) % 4] += 1

    total = votes.sum()
    if total == 0:
        return None, 0.0, votes
    phase = int(np.argmax(votes))
    return phase, float(votes[phase] / total), votes


def detect_downbeat_phase(detector, y, harm_y, sr, grid_times, grid_k, period, t0):
    """Dispatch to a downbeat backend. Returns (phase, confidence, detector_used, scores).

    Any failure in a trained backend falls back to the heuristic rather than
    aborting the run — a missing optional dependency should cost you accuracy,
    not the whole alignment. The name of the backend that actually produced the
    answer is returned so the caller can report it honestly.

    Confidence is on a common scale — the winning phase's share of the evidence,
    with 0.25 meaning chance — but the two families cannot reach the same
    ceiling. A trained detector's vote share can hit 1.0 when every downbeat it
    finds agrees. The heuristic's score share tops out near 0.70, because its
    cues are partly symmetric by construction and always feed the losing phases
    something. Compare a heuristic number against 0.70, not against 1.0.
    """
    if len(grid_times) < 16:
        return 0, PHASE_CHANCE, "insufficient-beats", np.full(4, 0.25)

    if detector == "madmom":
        try:
            downbeats = _madmom_downbeats(y, sr)
            phase, conf, votes = phase_from_downbeats(downbeats, period, t0)
            if phase is not None:
                return phase, conf, detector, votes
            sys.stderr.write(
                "%s: no downbeat landed on the fitted grid; using heuristic\n"
                % detector)
        except ImportError as e:
            sys.stderr.write(
                "%s unavailable (%s).\n"
                "  madmom:    uv pip install 'setuptools<81' cython, then\n"
                "             uv pip install --no-build-isolation "
                "'madmom @ git+https://github.com/CPJKU/madmom'\n"
                "             (the PyPI release is from 2018 and needs "
                "Python < 3.10; git main does not)\n"
                "Falling back to the heuristic detector.\n" % (detector, e))
        except Exception as e:
            sys.stderr.write("%s failed (%s: %s); using heuristic\n"
                             % (detector, type(e).__name__, e))

    phase, conf, scores = heuristic_downbeat_phase(
        y, harm_y, sr, grid_times, grid_k, period)
    return phase, conf, "heuristic", scores


# Boundary counts are pooled over several segment counts rather than trusting one.
# The phase histogram was stable across k = 8/12/16/20 on the reference track
# (13/16 on the winner at k=16), and pooling costs one clustering call each.
STRUCTURAL_K = (8, 12, 16, 20)

# Significance required before structure is allowed to OVERRIDE the detector.
# This is a one-sided binomial against "boundaries fall uniformly across the four
# bar positions". 0.01 is deliberately strict: overriding a trained detector on
# weak evidence is worse than leaving a coin flip alone, and on the reference
# track the real signal cleared it by four orders of magnitude (p = 3.8e-6).
STRUCTURAL_ALPHA = 0.01


def structural_downbeat_phase(y, sr, grid_times, grid_k, k_values=STRUCTURAL_K):
    """Downbeat phase from where the ARRANGEMENT changes. Returns (phase, conf, p, votes).

    THE POINT IS INDEPENDENCE. Every other cue here is percussive or trained on
    percussion: the heuristic weighs kick and snare energy, madmom's RNN was
    trained on beat/downbeat annotations. On four-on-the-floor both are looking at
    a signal that is symmetric under phase -> phase+2 — the kick is identical on
    beats 1 and 3, the backbeat identical on 2 and 4 — so both can be confidently
    half a bar wrong, and when they disagree neither has the evidence to settle it.

    Measured on the reference track: the heuristic scored 0.303 across four phases
    and madmom split its 159 downbeat votes 86/73 between phases 2 and 0. That
    0.541 "confidence" IS the vote share, not certainty. They picked differently
    and both were guessing.

    Arrangement change is a different signal entirely. Sections start on downbeats
    — and in dance music on 8-bar phrase starts — so clustering beat-synchronous
    timbre and harmony (MFCC + chroma + RMS) and asking which bar position the
    boundaries land on is evidence neither percussive cue can see. On the
    reference track it put 13 of 16 boundaries on phase 0 (p = 3.8e-6) against 1
    on phase 2, and 6 boundaries landed exactly on an 8-bar phrase start versus 0
    under the alternative.

    Returns the winning phase, its share of boundaries (0.25 is chance, on the
    same scale as every other confidence here), the one-sided binomial p-value,
    and the raw per-phase counts.

    Caveat worth keeping: validated on ONE track. The p-value is what gates the
    override, so weak or ambiguous structure leaves the detector's answer alone.
    """
    # Lazy: sections.py imports THIS module at import time, so a module-level
    # import here would be circular. By call time both are in sys.modules.
    from sections import beat_sync_features

    F, _rms = beat_sync_features(y, sr, grid_times)
    n_beats = min(F.shape[1], len(grid_k))
    if n_beats < 32:
        return None, 0.0, 1.0, np.zeros(4)

    votes = np.zeros(4)
    for k in k_values:
        # Need enough beats per segment for a boundary to mean anything.
        if k < 2 or n_beats // k < 4:
            continue
        try:
            bounds = librosa.segment.agglomerative(F[:, :n_beats], k)
        except Exception:
            continue
        for idx in np.asarray(bounds, dtype=int):
            if 0 <= idx < n_beats:
                votes[int(grid_k[idx]) % 4] += 1

    total = float(votes.sum())
    if total < 8:
        return None, 0.0, 1.0, votes

    order = np.argsort(votes)[::-1]
    phase, runner = int(order[0]), int(order[1])
    conf = float(votes[phase] / total)

    def _binom(k, n, p0):
        if n <= 0:
            return 1.0
        try:
            from scipy.stats import binomtest
            return float(binomtest(int(k), int(n), p0,
                                   alternative="greater").pvalue)
        except Exception:
            # Normal approximation if scipy is unavailable; the gate still works.
            mu, sd = p0 * n, np.sqrt(n * p0 * (1.0 - p0))
            z = (k - mu) / sd if sd > 0 else 0.0
            return float(0.5 * math.erfc(z / np.sqrt(2.0)))

    # TWO gates, and the second is the one that matters. The first asks whether
    # boundaries care about bar position at all; the second asks whether we can
    # actually tell WHICH position — and only the second catches a tie.
    #
    # Found by a synthetic that returned votes [28, 0, 0, 28]: a dead heat
    # between phases 0 and 3. Against a uniform null both score p = 4.9e-05, so
    # the first gate passes and `argmax` silently returns the lower index. That
    # override replaced a CORRECT detector answer (phase 3) with a wrong one,
    # which is strictly worse than not having the tiebreaker at all. Comparing
    # the winner with the runner-up reads that tie as p = 0.55 and declines.
    p_uniform = _binom(votes[phase], total, 0.25)
    p_margin = _binom(votes[phase], votes[phase] + votes[runner], 0.5)
    p = max(p_uniform, p_margin)

    return phase, conf, p, votes


def phase_verdict(confidence, detector):
    """(verdict, ceiling) for a phase confidence. Chance is PHASE_CHANCE, not 0.

    The ceiling is not 1.0 for the heuristic — its cues are partly symmetric and
    always feed the losing phases something, so ~0.70 is as certain as it can
    express. Judge each family against its own ceiling, or a heuristic answer
    reads as far worse than it is. Shared by the CLI and by stems2live so the
    two cannot disagree about what counts as a usable downbeat.
    """
    ceiling = 0.70 if detector == "heuristic" else 1.0
    rel = (confidence - PHASE_CHANCE) / (ceiling - PHASE_CHANCE)
    verdict = "chance-level" if rel < 0.10 else "weak" if rel < 0.35 else "ok"
    return verdict, ceiling


def build_analysis_mix(paths, sr=ANALYSIS_SR, exclude_drums=False, cache=None):
    """Sum the stems back to a mono mix for PHASE detection. Returns (y, sr, names).

    `exclude_drums=True` drops every percussion stem, for the harmonic-novelty
    cue — see bar_chroma_novelty for why that cue cannot see past the kit.

    `cache` is an optional dict shared across calls. Both mixes are built from
    the same folder and the harmonic stems belong to both, so without it those
    stems are decoded twice — measured at 1.05 s of the 7.55 s alignment on the
    test track. It costs nothing in peak memory: the summing loop below already
    holds every decoded stem at once, so the cache keeps alive only what the
    full-mix build had resident anyway.

    Keyed on (name, sr) rather than name, so a caller that mixes sample rates
    cannot silently get an array at the wrong rate.

    Why not just reuse the kick stem the grid was fitted on: a kick stem is the
    most beat-informative and the LEAST downbeat-informative signal in the set.
    Four-on-the-floor is periodic at beat level by construction, so every bar
    position looks identical and no detector — heuristic or trained — can tell
    beat 1 from beat 3 out of it. Downbeat cues live in bass movement, chord
    changes and phrase boundaries, i.e. in the harmonic stems. Tempo comes from
    the kick; phase comes from here.

    yt2stems folders contain no `mix` file, only parts, so the mix is
    reconstructed. The `drums` composite is dropped whenever `drums_*` parts are
    present — they sum back to it, and keeping both plays the kit twice (~+6 dB)
    and skews every percussive cue. This is the same rule stems2live applies
    when it mutes the composite clip.
    """
    names = sorted(paths)
    parts = [n for n in names if n.startswith("drums_")]
    if parts and "drums" in names:
        names = [n for n in names if n != "drums"]
    if exclude_drums:
        names = [n for n in names
                 if not n.startswith("drums") and "perc" not in n.lower()]
    if not names:
        return None, sr, []

    tracks = []
    loaded = []
    for name in names:
        key = (name, sr)
        if cache is not None and key in cache:
            y = cache[key]
        else:
            try:
                y, _ = librosa.load(paths[name], sr=sr, mono=True)
            except Exception as e:
                sys.stderr.write("skipping %s in analysis mix (%s)\n" % (name, e))
                continue
            if cache is not None:
                cache[key] = y
        if len(y):
            tracks.append(y)
            loaded.append(name)

    if not tracks:
        raise RuntimeError("no stem could be loaded for the analysis mix")

    # Stems are nominally the same length, but a truncated or padded export
    # should not silently shorten the mix.
    n = max(len(t) for t in tracks)
    mix = np.zeros(n, dtype=np.float64)
    for t in tracks:
        mix[:len(t)] += t

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix *= 0.9 / peak
    # `loaded`, not `names`: a stem that failed to decode was still being
    # reported in phase_source, claiming it contributed to a mix it is not in.
    return mix.astype(np.float32), sr, loaded


def select_first_stable_downbeat(grid_k, inliers, period, t0, downbeat_phase,
                                 min_stable_beats=16):
    """Earliest grid downbeat followed by min_stable_beats consecutive detections.

    Continuity is tested on the GRID, not on the interval list: every one of the
    next `min_stable_beats` slots must carry a detected inlier. A breakdown or a
    run of dropped beats fails that test, so the anchor cannot land inside a gap
    — which the old interval-tolerance test allowed, because a doubled interval
    only shows up as one bad gap and the beats around it still look regular.

    16 beats, not 8. Two bars is a thin evidence window for a decision that
    positions every clip in the set; four bars costs nothing on real material
    (the test track's anchor is unchanged) and rejects a stray two-bar run in an
    intro that happens to precede a gap.

    Returns (anchor_sec, k0) with the anchor read off the FITTED grid rather
    than the raw detection, which is worth ~10 ms of accuracy.
    """
    present = set(int(x) for x in np.asarray(grid_k)[np.asarray(inliers)])
    if not present:
        return t0, 0

    candidates = sorted(k for k in present if k % 4 == downbeat_phase)
    if not candidates:
        # Phase detection disagrees with everything detected; anchor on the
        # earliest real beat rather than inventing a slot.
        k0 = min(present)
        return period * k0 + t0, k0

    def stable(k0):
        return all((k0 + n) in present for n in range(1, min_stable_beats + 1))

    for k0 in candidates:
        if stable(k0):
            return period * k0 + t0, k0

    # No continuous run anywhere (very sparse detection): first downbeat slot.
    k0 = candidates[0]
    return period * k0 + t0, k0


def fit_tempo_segments(beat_times, seed_period, segment_sec=30.0,
                       min_segment_beats=16, tol=SEGMENT_TOL, max_iter=8):
    """Fit one tempo per ~segment_sec of audio, WITHOUT a global grid.

    `beat_times` must be ALL raw detections, not the grid inliers. This is the
    whole point of the function and getting it wrong is silent:

    THE GLOBAL GRID CANNOT MEASURE DRIFT. `fit_beat_grid` rejects, as outliers,
    exactly the beats that drifted away from its single uniform grid, so the
    survivors are near-uniform BY CONSTRUCTION. Measured on a track accelerating
    124 -> 132 BPM (a true 6.25% spread), grid-based segments read 0.68-2.27%
    depending on how many detections survived — understating the drift 3-6x, and
    on real material enough to fall under the threshold entirely and report a
    ramping set as steady. The information is in the detections; the grid throws
    it away. So nothing here may touch `grid_k`.

    Local indices instead. Within a segment each interval is divided by its
    rounded integer ratio to a seed period, and the indices ACCUMULATE those
    rounded steps. A dropped beat simply makes one interval a 2x or 3x multiple
    and still lands on the right index, so gaps stay harmless without a grid.

    Accumulating is deliberate and is not interchangeable with the direct form
    `round((t - t0)/period)`. Direct indexing needs the period accurate to better
    than 1/(2*n_beats) — about 0.8% over a 60-beat segment — and a period taken
    from noisy intervals is not that good: with 35 ms jitter the misassigned
    indices corrupt the line fit into a 4.2% reading on a STEADY track. Each
    accumulated step only needs its own interval to round correctly, which at
    480 +/- 50 ms is a 5-sigma margin.

    Each segment is then least-squares fitted and re-trimmed, which averages the
    jitter down across ~16+ beats rather than trusting any single interval.

    Returns dicts of `period`, `t_lo`/`t_hi` (SMOOTHED segment boundary times,
    read off the fit rather than off a raw detection) and `beats`. Segments too
    sparse to fit are skipped, not guessed at.
    """
    times = np.asarray(beat_times, dtype=float)
    seed = float(seed_period)
    if len(times) < 2 * min_segment_beats or seed <= 0:
        return []

    seg_ids = ((times - times[0]) / segment_sec).astype(int)
    segments = []
    for seg in np.unique(seg_ids):
        st = times[seg_ids == seg]
        if len(st) < min_segment_beats:
            continue

        # Greedy walk, because indices accumulate and one bad step poisons every
        # index after it. Aubio emits runs of half-grid detections interleaved
        # with the real beats — on the steady test track, 10 of them in a single
        # burst — and each is ~0.5 beat from its neighbour. Folding those in as
        # steps shifts the rest of the segment and reads as tempo change: that
        # alone turned the track's 0.17% into a 0.81% false positive. Anything
        # closer than SUB_BEAT_MIN of a beat is not a beat, so it is DROPPED.
        idx, kept_t, acc, last = [0.0], [st[0]], 0.0, st[0]
        for t_det in st[1:]:
            gap = (t_det - last) / seed
            if gap < SUB_BEAT_MIN:
                continue
            acc += max(1.0, float(np.round(gap)))
            idx.append(acc)
            kept_t.append(t_det)
            last = t_det
        m = np.asarray(idx, dtype=float)
        st = np.asarray(kept_t, dtype=float)
        if len(st) < min_segment_beats or m.max() - m.min() < min_segment_beats:
            continue

        # Trim within the segment only. Safe here in a way global trimming is
        # not: across ~30 s the tempo barely moves, so an outlier is a real
        # mis-detection rather than a beat that drifted.
        keep = np.ones(len(st), dtype=bool)
        seg_period = seg_t0 = None
        for _ in range(max_iter):
            if int(keep.sum()) < 8:
                break
            seg_period, seg_t0 = _fit_line(m[keep], st[keep])
            keep_new = np.abs(st - (seg_period * m + seg_t0)) <= tol * seg_period
            if np.array_equal(keep_new, keep):
                break
            keep = keep_new

        if seg_period is None or seg_period <= 0 or int(keep.sum()) < 8:
            continue
        mk = m[keep]
        segments.append({
            "period": float(seg_period),
            "t_lo": float(seg_period * mk.min() + seg_t0),
            "t_hi": float(seg_period * mk.max() + seg_t0),
            "beats": int(keep.sum()),
        })
    return segments


def analyze_tempo_drift(beat_times, seed_period, segment_sec=30.0,
                        min_segment_beats=16, threshold=DRIFT_THRESHOLD):
    """Compare per-segment fitted tempos; returns (has_drift, drift_pct).

    `beat_times` must be ALL raw detections and `seed_period` only sets the scale
    for rounding interval ratios — see fit_tempo_segments for why no grid index
    may enter this measurement.

    An earlier version compared the standard deviation of ALL intervals to the
    period, which is a guaranteed false positive: one breakdown contributes a
    34 s interval and the spread explodes (3.364 against a 0.03 threshold on the
    test track). Gaps say nothing about tempo. Per-segment fits ignore gaps and
    measure whether the tempo itself moves.

    Fewer than two usable segments is reported as no drift, not as an error:
    with one segment there is nothing to compare against.
    """
    segments = fit_tempo_segments(beat_times, seed_period, segment_sec,
                                  min_segment_beats)
    if len(segments) < 2:
        return False, 0.0

    periods = np.array([s["period"] for s in segments])
    spread = float((periods.max() - periods.min()) / np.median(periods))
    return bool(spread > threshold), float(round(spread * 100.0, 4))


def build_warp_map(beat_times, seed_period, segment_sec=30.0,
                   min_segment_beats=16, threshold=DRIFT_THRESHOLD):
    """Warp markers for a drifting track. Returns [[beat_time, sample_time], ...].

    Empty when the tempo does not drift, and the caller MUST NOT write a map in
    that case. An unwarped clip whose grid already matches is sample-accurate;
    warping it can only add error.

    ONE MARKER PER SEGMENT BOUNDARY, NOT ONE PER DETECTED BEAT. Residual jitter
    of the detections runs ~35 ms RMS — that is aubio's placement noise, not the
    track moving — so a marker at every detected beat would pin the audio to that
    noise and stretch every beat by up to ±35 ms to chase it. Markers go at the
    boundaries of the per-segment fits, where each `sample_time` is a
    least-squares value over ~16+ beats and the noise has averaged down.

    NO GLOBAL GRID ENTERS THIS. If the tempo really drifts then the single
    uniform grid is wrong across most of the track, so a marker derived from it
    would place audio at a time the audio never had. Every `sample_time` here is
    read off the LOCAL segment fit, and every `beat_time` is the musical position
    ACCUMULATED through the preceding segments at their own tempos.

    Units, both easy to get wrong and both silent when wrong:
      beat_time   — beats from the CLIP's sample start. The first marker sits at
                    `t_lo / period` beats, i.e. the beats that fit in the audio
                    ahead of it at that segment's tempo; each later marker adds
                    the beats elapsed across the span before it.
      sample_time — SECONDS from the sample start. NOT sample frames: do not
                    scale by the sample rate.

    Markers landing before the sample start are dropped rather than clamped —
    they refer to audio that does not exist. Live extrapolates outside the outer
    markers at the tempo of the nearest pair, which is the local segment's
    tempo, so an intro ahead of the first marker still lands right.

    Raises ValueError if the result is not strictly increasing on both axes. The
    bridge enforces the beat_time half, but a non-monotonic map corrupts the clip
    and it is far better to fail here, with the segment fits still in hand, than
    inside Live.
    """
    segments = fit_tempo_segments(beat_times, seed_period, segment_sec,
                                  min_segment_beats)
    if len(segments) < 2:
        return []

    periods = np.array([s["period"] for s in segments])
    spread = float((periods.max() - periods.min()) / np.median(periods))
    if spread <= threshold:
        return []

    # Walk the segments in time, accumulating musical position. Each segment
    # contributes its own two endpoints so its fitted tempo is encoded exactly;
    # the span BETWEEN two segments is carried at the earlier one's tempo, which
    # is the only tempo actually measured over that stretch.
    markers = []
    beat = segments[0]["t_lo"] / segments[0]["period"]
    markers.append([beat, segments[0]["t_lo"]])
    for i, seg in enumerate(segments):
        beat += (seg["t_hi"] - seg["t_lo"]) / seg["period"]
        markers.append([beat, seg["t_hi"]])
        if i + 1 < len(segments) and segments[i + 1]["t_lo"] > seg["t_hi"]:
            beat += (segments[i + 1]["t_lo"] - seg["t_hi"]) / seg["period"]
            markers.append([beat, segments[i + 1]["t_lo"]])

    markers = [[float(b), float(s)] for b, s in markers if b >= 0 and s >= 0]

    # Collapse anything that did not advance. Two segments can meet at the same
    # smoothed time, and a duplicate would trip the strict-increase check below
    # over what is really a zero-length span.
    deduped = markers[:1]
    for m in markers[1:]:
        if m[0] > deduped[-1][0] + 1e-9 and m[1] > deduped[-1][1] + 1e-9:
            deduped.append(m)
    markers = deduped

    # The bridge rejects a map with fewer than 2 markers; say so here instead of
    # letting it fail one socket round-trip later.
    if len(markers) < 2:
        return []

    for i in range(1, len(markers)):
        if markers[i][0] <= markers[i - 1][0]:
            raise ValueError(
                "warp map beat_times not strictly increasing at marker %d "
                "(%.4f after %.4f)" % (i, markers[i][0], markers[i - 1][0]))
        if markers[i][1] <= markers[i - 1][1]:
            raise ValueError(
                "warp map sample_times not strictly increasing at marker %d "
                "(%.4f s after %.4f s) — segment fits disagree about time order"
                % (i, markers[i][1], markers[i - 1][1]))

    return markers


def _cache_fingerprint(paths, override_tempo, detector,
                       use_structural=True):
    """Identity of an analysis: the inputs plus every parameter that changes it.

    Sizes and mtimes rather than content hashes — the stems are hundreds of MB
    and hashing them would cost more than the analysis being cached. Basenames
    only, never absolute paths: yt2stems RENAMES the stems directory once it has
    a BPM to tag with, so a cache keyed on absolute paths would be invalidated
    by the very step that wrote it.
    """
    return {
        "version": CACHE_VERSION,
        "stems": sorted([os.path.basename(p), os.path.getsize(p),
                         int(os.stat(p).st_mtime)] for p in paths.values()),
        "override_tempo": override_tempo,
        "detector": detector,
        # Changes the ANSWER, not just the reporting — structure can override the
        # detector's phase — so it has to be part of the identity.
        "use_structural": bool(use_structural),
    }


def read_cache(stems_dir, fingerprint):
    """Cached result for this exact fingerprint, or None."""
    path = os.path.join(stems_dir, CACHE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    if blob.get("fingerprint") != fingerprint:
        return None
    return blob.get("result")


def cached_grid(stems_dir):
    """The grid last cached for these exact stems, or None.

    For the standalone section tools, which should snap to the grid the
    arrangement was built on. Matched on cache version and the stem files
    (names, sizes, mtimes) only — NOT on detector or tempo override, because
    the question here is "which grid did stems2live last use on this audio",
    whatever options it ran with. Stale or unreadable caches give None.
    """
    if not os.path.isdir(stems_dir):
        return None
    paths = {os.path.splitext(f)[0]: os.path.join(stems_dir, f)
             for f in os.listdir(stems_dir) if f.endswith((".flac", ".wav"))}
    if not paths:
        return None
    try:
        with open(os.path.join(stems_dir, CACHE_NAME)) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    want = _cache_fingerprint(paths, None, DEFAULT_DETECTOR, True)
    have = blob.get("fingerprint") or {}
    if have.get("version") != want["version"] or have.get("stems") != want["stems"]:
        return None
    res = blob.get("result") or {}
    if not all(k in res for k in ("period", "grid_t0", "downbeat_sec")):
        return None
    return res


def write_cache(stems_dir, fingerprint, result):
    """Persist a result. Best-effort: a read-only stems dir is not an error."""
    path = os.path.join(stems_dir, CACHE_NAME)
    try:
        with open(path, "w") as fh:
            json.dump({"fingerprint": fingerprint, "result": result}, fh, indent=2)
    except OSError as e:
        sys.stderr.write("could not write %s (%s)\n" % (path, e))


def analyze_alignment(stems_dir_or_audio, override_tempo=None, detector=DEFAULT_DETECTOR,
                      use_cache=True, use_structural=True):
    """Main analysis pipeline for tempo, downbeat, offset, and drift.

    Two signals, two jobs. TEMPO is fitted to the kick stem, where the pulse is
    cleanest; PHASE is detected on the summed mix, where the harmonic
    information that distinguishes bar 1 from bar 3 actually lives. Mixing those
    up — running phase detection on the kick — is what made downbeats land a
    half-bar out.

    The result is cached in the stems directory as `alignment.json`. This whole
    analysis used to run TWICE per track: once inside yt2stems, where all 20
    fields were computed and then `grep`ed down to `round(bpm)` for a filename
    tag, and again in stems2live to recover the precision that rounding threw
    away. The cache is what lets the first run's answer survive to the second,
    and it is why stems2als no longer needs to warn against reading the BPM back
    out of the filename — the filename tag is now only ever a label.

    `use_cache=False` forces recomputation, which is what --no-cache is for when
    the aligner itself has changed.
    """
    if detector not in DETECTORS:
        raise ValueError("unknown detector %r; choose from %s"
                         % (detector, ", ".join(DETECTORS)))

    paths = {}
    if os.path.isdir(stems_dir_or_audio):
        stems_dir = stems_dir_or_audio
        for f in os.listdir(stems_dir):
            if f.endswith((".flac", ".wav")):
                name = os.path.splitext(f)[0]
                paths[name] = os.path.join(stems_dir, f)
    else:
        paths["mix"] = stems_dir_or_audio

    # Cache lookup before any decoding. Only a stems DIRECTORY is cacheable: a
    # single-file target has nowhere of its own to park the result, and a
    # single-file analysis is not interchangeable with a full-stems one anyway
    # (phase off a lone kick stem is near chance — see heuristic_downbeat_phase).
    cacheable = os.path.isdir(stems_dir_or_audio) and bool(paths)
    fingerprint = None
    if cacheable:
        fingerprint = _cache_fingerprint(paths, override_tempo, detector,
                                         use_structural)
        if use_cache:
            cached = read_cache(stems_dir_or_audio, fingerprint)
            if cached is not None:
                return cached

    # Preferred audio for beat tracking: kick stem > drums stem > bass > full mix
    pref_order = ["drums_kick", "drums", "bass", "mix"]
    primary_stem = None
    for p in pref_order:
        if p in paths:
            primary_stem = paths[p]
            break

    if not primary_stem and paths:
        primary_stem = list(paths.values())[0]

    if not primary_stem:
        raise ValueError(f"No audio files found in {stems_dir_or_audio}")

    # 1. Raw beat detection
    beats = estimate_aubio_beats(primary_stem)
    if beats is None and librosa and sf:
        # Fallback to librosa beat tracking
        y, sr = librosa.load(primary_stem, sr=22050, mono=True)
        tempo, beats_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beats_frames, sr=sr)

    if beats is None or len(beats) < 8:
        raise RuntimeError(f"Could not extract steady beats from {primary_stem}")

    # 2. Fit the grid. A --tempo override pins the period here rather than being
    #    patched over the result, so every later step sees the tempo Live gets.
    force_period = 60.0 / float(override_tempo) if override_tempo else None
    period, t0, k, inliers = fit_beat_grid(beats, force_period=force_period)
    bpm = 60.0 / period

    # Beats that actually fit the grid: their indices and their ideal times.
    # Drift deliberately uses NEITHER — see step 7.
    grid_k = k[inliers]
    grid_times = period * grid_k + t0

    # A low inlier fraction is itself a drift hint: a track holding one tempo
    # keeps nearly all its detections on a single uniform grid, so a large
    # rejection usually means the grid is chasing a tempo that moved.
    inlier_fraction = float(inliers.sum()) / float(len(beats))

    # 3. Downbeat phase, on grid positions, from the summed mix rather than from
    #    the kick stem the grid was fitted on.
    phase_source = "unavailable"
    harmonic_source = "none"
    detector_used = "none"
    downbeat_phase, phase_conf = 0, PHASE_CHANCE
    phase_scores = np.full(4, 0.25)
    structural_phase, structural_conf, structural_p = None, 0.0, 1.0
    structural_votes = [0, 0, 0, 0]
    phase_override = None
    # Bound before the try: the steps below read both, and if phase
    # detection raises they must still be defined rather than NameError-ing on
    # the way out.
    harm_y, mix_sr = None, ANALYSIS_SR
    if librosa and sf:
        try:
            # One decode per stem across both mixes. The harmonic stems belong
            # to both, and re-reading them cost 1.05 s of a 7.55 s run.
            decoded = {}
            mix_y, mix_sr, mix_names = build_analysis_mix(paths, cache=decoded)
            harm_y, _sr, harm_names = build_analysis_mix(
                paths, exclude_drums=True, cache=decoded)
            # Both mixes are summed by now; the per-stem arrays are dead weight
            # for the rest of the run (~270 MB on a 5-minute 10-stem folder).
            decoded.clear()
            downbeat_phase, phase_conf, detector_used, phase_scores = \
                detect_downbeat_phase(detector, mix_y, harm_y, mix_sr,
                                      grid_times, grid_k, period, t0)
            phase_source = "+".join(mix_names)
            harmonic_source = "+".join(harm_names) if harm_names else "none"

            # 3b. Structural tiebreaker. Runs ALWAYS (it is cheap and its
            #     agreement is worth reporting), but only OVERRIDES when the
            #     boundary clustering is significant — see STRUCTURAL_ALPHA.
            #     Every percussive cue is symmetric under phase -> phase+2 on
            #     four-on-the-floor, so this is the only evidence here that can
            #     break a half-bar tie.
            if use_structural:
                try:
                    s_phase, s_conf, s_p, s_votes = structural_downbeat_phase(
                        mix_y, mix_sr, grid_times, grid_k)
                    structural_phase = s_phase
                    structural_conf = s_conf
                    structural_p = s_p
                    structural_votes = [int(v) for v in s_votes]
                    if s_phase is not None and s_p < STRUCTURAL_ALPHA:
                        if s_phase != downbeat_phase:
                            phase_override = (
                                "%s said phase %d; arrangement boundaries say "
                                "phase %d (p=%.1e)"
                                % (detector_used, downbeat_phase, s_phase, s_p))
                            downbeat_phase = s_phase
                            # Report the evidence that actually decided it.
                            phase_conf = s_conf
                            detector_used = "%s+structural" % detector_used
                        else:
                            phase_override = "structural agrees (p=%.1e)" % s_p
                            detector_used = "%s+structural" % detector_used
                except Exception as e:
                    sys.stderr.write(
                        "structural phase check failed (%s: %s); keeping %s\n"
                        % (type(e).__name__, e, detector_used))
        except Exception as e:
            sys.stderr.write("phase detection failed (%s: %s); assuming phase 0\n"
                             % (type(e).__name__, e))

    # 4. First grid downbeat with a continuous run behind it.
    downbeat_sec, _k0 = select_first_stable_downbeat(
        k, inliers, period, t0, downbeat_phase)

    # 5. Alignment offset in beats. The clip is placed at arrangement beat
    #    START, the anchor lands at START + downbeat_sec/period, and this offset
    #    makes that a multiple of a bar. Pre-roll audio survives because
    #    START >= 0.
    #
    #    stems2live no longer uses this to POSITION clips — it places them on a
    #    whole bar and trims the lead-in instead — but it is still the honest
    #    answer to "how far off a bar line is the anchor", and the fallback path
    #    in stems2live reads it.
    phase_beats = (downbeat_sec / period) % 4.0
    start_offset_beats = float(round((-phase_beats) % 4.0, 4))

    # 7. Tempo drift, and the warp map that corrects it. Both read the same
    #    segment fits, so the map can never disagree with the flag that decides
    #    whether to write it. A steady track gets an empty map and stays
    #    unwarped — see build_warp_map.
    #
    #    ALL detections go in, not `raw_times` (the grid inliers). The grid has
    #    already rejected the beats that drifted away from it, so measuring drift
    #    on the survivors reads a ramping track as steady — that is the exact bug
    #    this call site used to have. `period` is a scale for rounding interval
    #    ratios here, not a grid.
    has_drift, drift_pct = analyze_tempo_drift(beats, period)
    warp_map = []
    if has_drift:
        try:
            warp_map = build_warp_map(beats, period)
        except ValueError as e:
            # A map that will not validate is worth losing; the clips are still
            # placed correctly without one.
            sys.stderr.write("warp map rejected (%s); leaving the clip unwarped\n" % e)

    # Reported as a flag as well as a number so a caller does not have to know
    # that each detector family has its own ceiling to interpret the number.
    verdict, _ceiling = phase_verdict(phase_conf, detector_used)
    result = {
        "bpm": float(round(bpm, 3)),
        "period": float(round(period, 6)),
        "downbeat_sec": float(round(downbeat_sec, 4)),
        "downbeat_phase": int(downbeat_phase),
        "downbeat_confidence": float(round(phase_conf, 4)),
        "downbeat_verdict": str(verdict),
        "detector": str(detector_used),
        "phase_source": str(phase_source),
        "harmonic_source": str(harmonic_source),
        "phase_scores": [float(round(s, 4)) for s in np.asarray(phase_scores)],
        "structural_phase": (int(structural_phase)
                             if structural_phase is not None else None),
        "structural_confidence": float(round(structural_conf, 4)),
        "structural_p": float(structural_p),
        "structural_votes": list(structural_votes),
        "phase_override": (str(phase_override)
                           if phase_override is not None else None),
        "start_offset_beats": float(start_offset_beats),
        "has_drift": bool(has_drift),
        "warp_map": warp_map,
        "primary_stem": os.path.basename(primary_stem),
        "beat_count": int(len(beats)),
        "grid_t0": float(round(t0, 4)),
        "inlier_count": int(inliers.sum()),
        "inlier_fraction": float(round(inlier_fraction, 4)),
        "drift_pct": float(drift_pct)
    }

    if cacheable:
        write_cache(stems_dir_or_audio, fingerprint, result)
    return result


def main():
    ap = argparse.ArgumentParser(description="Downbeat detector & stem aligner")
    ap.add_argument("target", help="Path to stems directory or single audio file")
    ap.add_argument("--tempo", type=float, default=None, help="Override detected tempo")
    ap.add_argument("--detector", choices=DETECTORS, default=DEFAULT_DETECTOR,
                    help="downbeat backend. heuristic (default) needs no extra "
                         "dependency and is ~4x faster; it reads near the 0.25 "
                         "chance level on four-on-the-floor, but the "
                         "arrangement-boundary tiebreaker is what settles bar 1 "
                         "either way. madmom is the trained tracker, worth "
                         "trying on material the heuristic is not tuned for. A "
                         "missing backend degrades to heuristic rather than "
                         "aborting")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute even if %s holds a matching result. The "
                         "cache already invalidates on stem mtime/size and on "
                         "every parameter that changes the answer, so this is "
                         "for when the ALIGNER changed" % CACHE_NAME)
    ap.add_argument("--no-structural", action="store_true",
                    help="skip the arrangement-boundary tiebreaker. That check "
                         "is the only cue here that is not percussive, so it is "
                         "what breaks a half-bar tie when the detector is split "
                         "— turn it off only to see the raw detector answer")
    ap.add_argument("--json", action="store_true", help="Output JSON results")
    args = ap.parse_args()

    res = analyze_alignment(args.target, override_tempo=args.tempo,
                            detector=args.detector,
                            use_cache=not args.no_cache,
                            use_structural=not args.no_structural)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        conf = res["downbeat_confidence"]
        verdict, ceiling = phase_verdict(conf, res["detector"])
        print(f"Primary Stem:        {res['primary_stem']}  (tempo)")
        print(f"Phase Source:        {res['phase_source']}")
        print(f"Harmonic Source:     {res['harmonic_source']}")
        print(f"Detector:            {res['detector']}")
        print(f"Detected Tempo:      {res['bpm']:.3f} BPM")
        print(f"Grid Beats:          {res['inlier_count']} of {res['beat_count']} detections")
        print(f"First Downbeat:      {res['downbeat_sec']:.4f}s (phase beat {res['downbeat_phase'] + 1})")
        print(f"Phase Confidence:    {conf:.3f}  ({verdict}; {PHASE_CHANCE:.2f} = chance, "
              f"{ceiling:.2f} = this detector's ceiling)")
        print(f"Phase Scores:        {['%.3f' % s for s in res['phase_scores']]}")
        if res.get("structural_phase") is not None:
            sv, sp = res["structural_votes"], res["structural_p"]
            print(f"Structural Phase:    beat {res['structural_phase'] + 1}  "
                  f"votes {sv}  p={sp:.1e}")
            if res.get("phase_override"):
                mark = "OVERRODE" if "say phase" in res["phase_override"] else "agrees"
                print(f"  ^^ arrangement boundaries {mark}: {res['phase_override']}")
                if mark == "OVERRODE":
                    print("     Boundaries come from timbre/harmony change, which is "
                          "the only cue here that is NOT percussive — kick and snare "
                          "are both symmetric under a half-bar shift, so they cannot "
                          "settle this and the detector was guessing.")
        # Tempo can be trustworthy while the bar phase is not, and the two fail
        # differently: a wrong phase is not subtly wrong, it puts every clip a
        # whole number of beats out. Say that in words rather than leaving it to
        # be inferred from a number against a ceiling most readers do not know.
        if verdict != "ok":
            print("  ^^ BAR 1 IS UNRELIABLE (%s). The tempo and beat grid above "
                  "are sound, but WHICH beat is bar 1 is close to a guess here, "
                  "so every clip may sit 1-3 beats out. Check bar 1 by ear "
                  "before editing; --detector madmom is worth a try."
                  % verdict)
        print(f"Nudge Offset:        +{res['start_offset_beats']:.4f} beats (onto a bar line)")
        print(f"Tempo Drift:         {'YES' if res['has_drift'] else 'NO'} (spread {res['drift_pct']:.4f}%)")
        # Say explicitly that no map is being written, rather than printing
        # nothing — silence here reads as "the warp step failed".
        if res["warp_map"]:
            print(f"Warp Map:            {len(res['warp_map'])} markers at segment "
                  f"boundaries (needs patch 0002 + a Live restart to apply)")
        else:
            print("Warp Map:            none — steady tempo, clip stays unwarped")


if __name__ == "__main__":
    main()

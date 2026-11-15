"""Profile a drum one-shot in the terms a synth is programmed in: envelope,
pitch and pitch drop, harmonics, noise, click, band balance and width.

Every measure is relative to the hit itself (its onset, its peak, its own
fundamental), so two files at different levels, sample rates or with
different leading silence profile alike. Local only; needs the `listen` extra.

Choices that are not obvious from the code:

- The onset is the first 1 ms frame within 40 dB of the hit's loudest 1 ms
  (and in it, the first sample at that level); everything before it is
  dropped and every time is counted from it. A DC offset is removed first
  when the file shows one: the quietest 10 ms blocks are quiet and share a
  mean above -60 dB of the peak (see _dc; a file with digital silence, or a
  low tone still sounding at its end, has none).
- Decay uses a power envelope smoothed over whole periods of the body pitch
  (5 ms for an unpitched hit), so partials beating against each other do not
  read as decay. A fall time is the last moment the envelope is still above
  the threshold, measured from the envelope's peak. The attack time is where
  that same envelope first comes within 1 dB of its peak: whole periods, so
  the start phase and a clipped plateau do not move it, and a linear ramp
  reads at 10^(-1/20) = 89 % of its length.
- Pitch frames are Hann windows on the analytic signal (no negative-frequency
  image), four periods long, zero-padded 8x, with parabolic interpolation on
  the dB spectrum. A spectral peak must be the maximum within one bin of the
  unpadded FFT, so window sidelobes never count.
- The track is chained: frames 1 ms apart, from the body window's centre
  back to the onset and on to 300 ms, each sized by and held within 2 st of
  its neighbour's pitch, so it cannot jump to a harmonic or to a thump below.
  A frame is centred on its time, so the first one sits 2 periods in. (Three
  periods let a 2nd harmonic louder than the fundamental pull a frame 4 %.)
- Before that, the pitch comes from half-periods between zero crossings
  (hysteresis a tenth of the local amplitude, after a 1/8-period centred
  boxcar), when they follow one pitch: each within 1.8x of the last, and
  meeting the first frame within 1.5 st. Leading crossings out of step (a
  click) are skipped; otherwise the first two half-periods are extended to
  the onset when that step stands out of the crossings' jitter.
  pitch_start_at_ms says where the start was read. Where neither the
  crossings nor a frame cover a track point, it is None. A stick transient
  whose crossings run smoothly into the tone reads as part of the glide.
- The body pitch comes from one long window, 40-150 ms (shortened for a hit
  that is over sooner), since a 25 ms window resolves only about 40 Hz.
- The fundamental is the lowest peak within 20 dB of the strongest, with two
  corrections: a lone low peak (no harmonic of its own beyond the series of
  a strong peak 2-4x above it) more than 10 dB down is a rumble, and a weaker
  peak at 1/2-1/4 of the pitch whose series has a peak of its own beyond the
  pitch's series is a weak fundamental under strong overtones. A lone peak at
  1/k of the pitch cannot be told from a rumble, so a note names it.
- A hit is pitched when the body has a peak at least 15 dB above its local
  spectral floor, the fundamental's harmonic series holds at least half of
  the energy of all such peaks, and the fundamental stands 25 dB above the
  noise around it, or 15 dB in a body whose noise share is under -3 dB.
  (Prominence alone cannot separate a ride, 20 dB, from a clave, 20 dB;
  the noise share alone flipped a tone 30 dB clear of its noise.) Otherwise
  (noise, a cymbal, an inharmonic bell) it reports its partials and no pitch.
- Noise share is the energy outside the main lobes of the tonal peaks (every
  peak 15 dB above its local floor, harmonic or not; a lobe runs down to its
  first minimum, at most 4 bins out), against the total, from 20 Hz up. The
  attack window is a plain Hann over 0-30 ms, so it weighs the middle of the
  attack most; the click right at the onset is click_db's job.
  A glide smears its fundamental, so a sweeping attack reads noisier than
  the same tone held still.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sideman.listen import BANDS, FLOOR_DB, band_power, check_channels, load, to_db

ONSET_DB = 40.0          # the onset is the first 1 ms frame within this of the loudest
FALLS_DB = (10, 20, 40)
END_DB = 60.0            # duration runs to the last moment within this of the peak
GATE_DB = 50.0           # pitch frames must end before the hit falls this far
TRACK_MS = (0, 5, 10, 20, 40, 80, 150, 300)
ATTACK_S = 0.030
ATTACK_RANGE_DB = 30.0   # the attack noise window ends this far under the envelope peak
ATTACK_DB = 1.0          # the attack ends within this of the envelope peak
BODY_S = (0.040, 0.150)
PERIODS = 4.0            # pitch frame length, in periods of the frame's pitch (3 let a louder 2nd harmonic pull it 4 %)
FRAME_S = (0.004, 0.040)
STRONG_DB = 20.0         # a partial this far under the strongest still counts as the fundamental
PROMINENT_DB = 15.0      # a peak must stand this far above its local floor to be tonal
HARM_TOL = 0.03          # a partial within 3 % of k * f0 is harmonic k
SETTLE_ST = 0.5
SETTLE_HOLD_S = 0.010    # settled = within SETTLE_ST and still there 10 ms later
EARLY_SPAN_S = 0.030     # zero crossings are read up to the first frame's centre + this
EARLY_RATIO = 1.8        # one half-period may differ from the last by this factor (a fast kick: 1.5)
EARLY_ZIGZAG = 0.04      # half-periods that step up and down by more than this are not one pitch
EARLY_STEP_ST = 0.3      # the first half-periods are extended to the onset only for a step this big
EARLY_ST = 1.5           # and they must meet the first frame within this
STEP_ST = 2.0            # a track frame may move this far from the one 1 ms before it
PITCHED_DB = 25.0        # a fundamental this far above the noise around it is pitched...
NOISY_DB = -3.0          # ...and so is a less prominent one in a body that is not mostly noise
LONE_DB = 10.0           # a lone peak at 1/k of a strong one, further down than this, is a rumble
PARTIALS_DB = 40.0       # inharmonic partials are listed down to this under the strongest peak
MAX_PARTIALS = 5
MIN_PERIODS = 4          # a body window shorter than this many periods gives no harmonics
CLICK_HZ = 2000.0
DC_REL = 1e-3            # a DC offset under -60 dB of the peak is left alone
MAX_S = 10.0             # a one-shot longer than this is cut, with a note
RISE_S = 0.010           # hits: a new hit rises RISE_DB within this...
RISE_DB = 15.0
FAST_DB = 6.0            # ...reaching within this of its peak (the max over PEAK_S)...
PEAK_S = 0.050
JUMP_DB = 6.0            # ...and tops the CEILING_S before it by this
CEILING_S = 0.050
MIN_GAP_S = 0.050
# compare(): a difference past one of these is a reading
PITCH_ST = 0.3           # body pitch
DROP_ST = 1.0            # pitch drop
TIME_RATIO = 0.15        # attack, falls, duration, settle: 15 %...
TIME_FLOOR_MS = 2.0      # ...and at least this, so 0.3 vs 0.4 ms is not a reading
LEVEL_DB = 3.0           # harmonics, noise share
CLICK_STEP_DB = 6.0
WIDTH_STEP = 0.1
# compare(): below these on both sides a difference is the method's floor, not the sound
HARM_FLOOR_DB = -50.0    # a harmonic must be above this...
HARM_CLEAR_DB = PROMINENT_DB  # ...and stand clear of its own spectral floor like a tonal peak
NOISE_FLOOR_DB = -25.0   # a pure decaying tone reads -26 to -42 from window leakage
CLICK_FLOOR_DB = -40.0
BAND_FLOOR_DB = -30.0    # a band this far under the hit's total on both sides is not read
START_GAP_MS = 2.0       # starts read further apart than this are compared from the later one
FAST_SETTLE_MS = 5.0     # a pitch drop over in less than this is a click, not a glide
BAND_NAMES = {"sub": "sub (20-60 Hz)", "bass": "bass (60-250 Hz)", "low_mid": "low mids (250-500 Hz)",
              "mid": "mids (500 Hz-2 kHz)", "high_mid": "high mids (2-6 kHz)", "high": "highs (6-20 kHz)"}


def _boxcar(p: np.ndarray, length: int) -> np.ndarray:
    """Centred moving mean of p over `length` samples, shrinking at the ends."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(len(p))
    lo = np.clip(i - length // 2, 0, len(p))
    hi = np.clip(i - length // 2 + length, 0, len(p))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _analytic(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    h = np.zeros(nfft)
    h[0] = h[nfft // 2] = 1.0
    h[1:nfft // 2] = 2.0
    return np.fft.ifft(np.fft.fft(y, nfft, axis=0) * h[:, None], axis=0)[:n]


def _spectrum(z: np.ndarray, a: int, b: int, sr: int, level: np.ndarray | None = None) -> dict:
    """Channel-mean power spectrum (dB, positive frequencies) of z over [a, b),
    Hann-windowed, zero-padded at least 8x. z is the analytic signal for the
    short pitch frames (no negative-frequency image) and the plain signal for
    the attack and body windows: the analytic signal of a fast decay carries
    a 1/t error tail from the onset that is louder than the decay itself 40 dB
    down. With `level` (a power envelope),
    the frame is first flattened by an exponential fitted to it, so a fast
    decay does not widen every peak into its neighbours."""
    m = b - a
    win = np.hanning(m)
    if level is not None:
        t = np.arange(m) / sr
        slope = np.polyfit(t, 10 * np.log10(level[a:b] + 1e-30), 1)[0]      # dB/s
        win = win * 10 ** (-slope * (t - t.mean()) / 20)
    nfft = 1 << int(math.ceil(math.log2(8 * m)))
    p = (np.abs(np.fft.fft(z[a:b] * win[:, None], nfft, axis=0)[:nfft // 2]) ** 2).mean(axis=1)
    return {"p": p, "d": 10.0 * np.log10(p + 1e-30), "hz": np.arange(nfft // 2) * sr / nfft,
            "bin": max(1, int(round(nfft / m))), "sr": sr, "nfft": nfft}


def _peaks(s: dict, hi_hz: float) -> np.ndarray:
    """Bins that are the maximum within one unpadded bin either side (so a
    window sidelobe never qualifies), from 20 Hz to hi_hz, ascending."""
    d, h = s["d"], s["bin"]
    top = sliding_window_view(np.pad(d, h, constant_values=-np.inf), 2 * h + 1).max(axis=1)
    return np.nonzero((d >= top) & (s["hz"] >= 20.0) & (s["hz"] <= hi_hz))[0]


def _interp(s: dict, i: int) -> tuple[float, float]:
    """(Hz, dB) of the peak at bin i, parabolic on the dB spectrum."""
    d = s["d"]
    off, level = 0.0, float(d[i])
    if 0 < i < len(d) - 1:
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            off = 0.5 * (y0 - y2) / den
            level = float(y1 - 0.25 * (y0 - y2) * off)
    return float((i + off) * s["sr"] / s["nfft"]), level


def _floor(s: dict, i: int, both: bool = False) -> float | None:
    """Local spectral floor at bin i: the median (dB) of a ring 2-10 bins away,
    outside the main lobe of a peak at i. With `both`, the higher of the two
    sides' medians: the edge of band-limited noise has nothing above it, and
    against the whole ring it reads as a peak standing clear of its floor."""
    d, h = s["d"], s["bin"]
    sides = [r for r in (d[max(0, i - 10 * h):max(0, i - 2 * h)], d[i + 2 * h + 1:i + 10 * h + 1]) if len(r)]
    if not sides:
        return None
    return max(float(np.median(r)) for r in sides) if both else float(np.median(np.concatenate(sides)))


def _tonal(s: dict, hi_hz: float) -> list[tuple[float, float, int]]:
    """Peaks standing PROMINENT_DB above their local floor, as (Hz, dB, bin),
    strongest first."""
    out = []
    for i in _peaks(s, hi_hz):
        floor = _floor(s, i)
        if floor is not None and s["d"][i] - floor >= PROMINENT_DB:
            out.append(_interp(s, i) + (int(i),))
    return sorted(out, key=lambda t: -t[1])


def _frame_pitch(z: np.ndarray, t: float, sr: int, guess: float, end: int, k: int = 1) -> tuple:
    """(Hz, centre in s, clamped) of a frame centred on t, PERIODS periods of
    guess / k long (k: which harmonic is followed, so neighbours stay out of
    its lobe). A frame that would start before the onset starts there
    (clamped); one that would run past `end` is tried at 3 periods (a short
    hit), and not read if it still does not fit. The pitch is the
    lowest peak within STRONG_DB of the frame's strongest that is also within
    STEP_ST of `guess`, so the track cannot jump to a harmonic or to rumble."""
    for periods in (PERIODS, PERIODS - 1):
        w = int(round(min(max(periods * k / guess, FRAME_S[0]), FRAME_S[1]) * sr))
        a = int(round(t * sr)) - w // 2
        clamped, a = a <= 0, max(0, a)
        if a + w <= end:
            break
    else:
        return None, None, clamped
    s = _spectrum(z, a, a + w, sr)
    idx = _peaks(s, min(20000.0, 0.45 * sr))
    if len(idx):
        d = s["d"][idx]
        idx = idx[(d >= d.max() - STRONG_DB) & (np.abs(12 * np.log2(s["hz"][idx] / guess)) <= STEP_ST)]
    if not len(idx):
        return None, None, clamped
    return _interp(s, int(idx[0]))[0], (a + w / 2) / sr, clamped


def _track(z: np.ndarray, sr: int, f0: float, mid: float, end: int, k: int = 1) -> tuple[list, bool]:
    """Dense pitch track, frames 1 ms apart, chained outward from `mid` (the
    body window's centre, where f0 holds) back to the onset and on to TRACK_MS's
    last point or `end`. ([(centre s, Hz)] in time order, whether it reached the onset)."""
    back, guess, reached = [], f0, False
    for t in np.arange(mid, -1e-9, -0.001):
        f, c, clamped = _frame_pitch(z, float(t), sr, guess, end, k)
        if f is None:
            break
        back.append((c, f))
        guess = f
        if clamped:
            reached = True
            break
    fwd, guess = [], f0
    for t in np.arange(mid + 0.001, TRACK_MS[-1] / 1000.0 + 0.001, 0.001):
        f, c, _ = _frame_pitch(z, float(t), sr, guess, end, k)
        if f is None:
            break
        fwd.append((c, f))
        guess = f
    return back[::-1] + fwd, reached


def _crossings(v: np.ndarray, sr: int, period: float, from_zero: float | None) -> np.ndarray:
    """Zero-crossing times (s, interpolated) of v, counted with a hysteresis of
    a tenth of the local amplitude (the max over one period either side), so
    noise riding on a slow crossing does not add crossings. The onset counts
    as one, at `from_zero` (in samples, possibly before it), when the hit
    starts near zero."""
    w = max(1, int(period * sr))
    a = np.abs(v)
    loc = sliding_window_view(np.pad(a, (w, w)), 2 * w + 1).max(axis=1)[:len(v)]
    out, state, last = [], 0, None
    if from_zero is not None:
        out.append(from_zero / sr)
    for i in range(1, len(v)):
        if (v[i - 1] <= 0 < v[i]) or (v[i - 1] >= 0 > v[i]):
            last = i - 1 + v[i - 1] / (v[i - 1] - v[i])
        side = 1 if v[i] > 0.1 * loc[i] else (-1 if v[i] < -0.1 * loc[i] else 0)
        if side and side != state:
            if state and last is not None:
                out.append(last / sr)
            state = side
    return np.array(out)


def _early_pitch(y: np.ndarray, sr: int, f1: float, t1: float) -> tuple | None:
    """The pitch before the first pitch frame's centre t1, from half-periods
    between zero crossings (a frame needs four periods; a half-period
    needs half of one). ([(midpoint s, Hz)], the time they cover from (0
    when the onset is a crossing or they are extended to it), a click skipped) or None when the crossings do not follow one pitch: each
    half-period within EARLY_RATIO of the last, reaching t1 within EARLY_ST
    of the frame there. Leading crossings out of step (a click) are skipped;
    otherwise the first two half-periods are extended linearly to the onset
    when that step is at least EARLY_STEP_ST, the third half-period moves the
    same way, and it stands out of the crossings' jitter."""
    m = y.mean(axis=1)[:int((t1 + EARLY_SPAN_S) * sr)]
    from_zero = None           # a hit starting near zero crosses it at the onset: where, exactly,
    if len(m) > 1 and abs(m[0]) < 0.1 * np.abs(m[:max(1, int(sr / f1))]).max() and m[1] != m[0]:
        from_zero = min(0.0, -m[0] / (m[1] - m[0]))   # often a sample before the trimmed start
    smooth = max(1, int(round(sr / (8 * f1)))) | 1   # an odd, centred boxcar: zero phase (an even one
    # would delay every crossing but the onset's by half a sample), takes a click off
    m = np.convolve(m, np.ones(smooth) / smooth, mode="same")
    tz = _crossings(m, sr, 1.0 / f1, from_zero)
    if len(tz) < 4:
        return None
    mid, f = (tz[1:] + tz[:-1]) / 2, 0.5 / np.diff(tz)
    ok = np.r_[(f[1:] / f[:-1] > 1 / EARLY_RATIO) & (f[1:] / f[:-1] < EARLY_RATIO), True]
    first = 0
    while first < len(ok) - 2 and not ok[first]:
        first += 1
    run = first
    while run < len(ok) and ok[run]:
        run += 1
    mid, f = mid[first:run + 1], f[first:run + 1]
    if len(f) < 3 or mid[-1] < t1 or abs(12 * math.log2(np.interp(t1, mid, f) / f1)) > EARLY_ST:
        return None
    before = f[mid < t1 + 0.5 / f1]     # the points used, and one more: long and short half-periods
    step = np.diff(before) / before[:-1]  # in turn mean a strong overtone adds crossings, so they
    turns = np.sign(step[np.abs(step) > EARLY_ZIGZAG])        # are not the half-periods of one
    if np.any(turns[1:] != turns[:-1]):                       # pitch
        return None
    points = [(float(t), float(v)) for t, v in zip(mid, f)]
    since = float(tz[first])               # the first half-period starts here (0 or just before)
    if first:
        return points, since, True
    # extend the first two half-periods to the onset, but only when the step is
    # larger than twice what jitter alone could make of it (noise on the
    # crossings, from the scatter of the second differences)
    f0 = f[0] - (f[1] - f[0]) / (mid[1] - mid[0]) * mid[0]
    jitter = np.std(np.diff(f[:8], 2)) / math.sqrt(6) if len(f) >= 4 else 0.0
    spread = jitter * math.sqrt(1 + 2 * (mid[0] / (mid[1] - mid[0])) ** 2)
    real = (abs(12 * math.log2(f0 / f[0])) >= EARLY_STEP_ST and (f[2] - f[1]) * (f[1] - f[0]) > 0
            and abs(f0 - f[0]) > 2 * spread)          # a step of its own, and the third half-period agrees
    if not real:
        return points, max(0.0, since), False
    return [(0.0, float(f0))] + points, 0.0, False


def _last_above(env: np.ndarray, start: int, level: float) -> float | None:
    """Fractional index where env falls below `level` for the last time after
    `start`, or None if it is still above at the end of the file."""
    above = np.nonzero(env[start:] >= level)[0]
    if not len(above):
        return float(start)
    k = start + int(above[-1])
    if k >= len(env) - 1:
        return None
    a, b = 10 * math.log10(env[k]), 10 * math.log10(max(env[k + 1], 1e-30))
    lv = 10 * math.log10(level)
    return k + ((a - lv) / (a - b) if a > b else 0.0)


def _noise_flatness(s: dict, peaks: list, hi_hz: float) -> tuple[float, float]:
    """(noise share in dB, spectral flatness 0-1) over 20 Hz-hi_hz."""
    band = (s["hz"] >= 20.0) & (s["hz"] <= hi_hz)
    d, reach = s["d"], 4 * s["bin"]
    tonal = np.zeros(len(d), bool)
    for _, _, i in peaks:            # the main lobe: down to its first minimum, at most 4 bins out
        lo = i
        while lo > max(0, i - reach) and d[lo - 1] < d[lo]:
            lo -= 1
        hi = i
        while hi < min(len(d) - 1, i + reach) and d[hi + 1] < d[hi]:
            hi += 1
        tonal[lo:hi + 1] = True
    p = s["p"][band]
    total = float(p.sum())
    noise = to_db(float(s["p"][band & ~tonal].sum()) / total) if total > 0 else None
    flat = float(np.exp(np.mean(np.log(p + 1e-30))) / np.mean(p)) if total > 0 else None
    return noise, flat


def _fundamental(tonal: list) -> tuple[float, float, int, list]:
    """(Hz, dB, bin, notes) of the fundamental among the tonal peaks: the
    lowest within STRONG_DB of the strongest, then
    - a lone low peak (no harmonic of its own beyond the series of a strong
      peak 2-4x above it) more than LONE_DB under the strongest is a rumble,
      and that peak is the fundamental instead;
    - a weaker peak at 1/2-1/4 of it (down to PARTIALS_DB) is the fundamental
      when its series has a peak of its own beyond that series: a weak
      fundamental under strong overtones.
    A lone peak at 1/k of the result is named in a note."""
    top = tonal[0][1]
    cand = sorted((t for t in tonal if t[1] >= top - PARTIALS_DB), key=lambda t: t[0])

    def sub(g, h):             # g at h/k, k = 2-4
        k = round(h[0] / g[0])
        return 2 <= k <= 4 and abs(h[0] - k * g[0]) <= HARM_TOL * h[0]

    def backed(g, h):          # a peak besides g in g's series and not in h's
        return any(t is not g and _harmonic(t[0], g[0]) and not _harmonic(t[0], h[0]) for t in cand)
    p = min((t for t in cand if t[1] >= top - STRONG_DB), key=lambda t: t[0])
    for _ in range(len(cand)):
        up = [h for h in cand if h[1] >= top - STRONG_DB and sub(p, h) and not backed(p, h)]
        if up and p[1] < top - LONE_DB:
            p = up[0]
            continue
        down = [g for g in cand if sub(g, p) and backed(g, p)]
        if down:
            p = down[0]
            continue
        break
    notes = ["%s Hz at %.0f dB sits at 1/%d of the pitch with no harmonics of its own: "
             "a weak fundamental or a rumble" % (_num(g[0]), g[1] - p[1], round(p[0] / g[0]))
             for g in cand if sub(g, p)]
    return p[0], p[1], p[2], notes


def _harmonic(hz: float, f0: float) -> bool:
    """Whether hz is within HARM_TOL of a whole multiple (1, 2, ...) of f0."""
    k = round(hz / f0)
    return k >= 1 and abs(hz - k * f0) <= HARM_TOL * k * f0


def _ms(samples: float, sr: int) -> float:
    return round(1000.0 * samples / sr, 1)


def _segments(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """[start, end) sample ranges of the hits in a take. A new hit is a rise of
    RISE_DB that comes within FAST_DB of its peak inside RISE_S and tops the
    CEILING_S before it by JUMP_DB, at least MIN_GAP_S after the last one, so a
    tremolo, a tom's beating or a slow swell does not split a hit. A hit that
    starts before the last one has fallen 15 dB (a fast roll) is not split
    off either."""
    hop = max(1, int(round(0.001 * sr)))                     # the envelope on a 1 ms grid
    env = _boxcar((x ** 2).mean(axis=1), max(1, int(0.005 * sr)))[::hop]
    e = 10.0 * np.log10(env / env.max() + 1e-30)
    rise, gap, ahead, ceil = (int(round(v * 1000)) for v in (RISE_S, MIN_GAP_S, PEAK_S, CEILING_S))
    first = int(np.argmax(e >= -ONSET_DB))
    starts, last = [0], first
    for j in range(first + 1, len(e)):
        if j - last < gap or e[j] < -ONSET_DB:
            continue
        jm = j - rise + int(np.argmin(e[j - rise:j])) if j >= rise else int(np.argmin(e[:j]))
        if e[j] - e[jm] < RISE_DB:
            continue
        top = e[jm:jm + ahead].max()
        if e[jm:jm + rise + 1].max() < top - FAST_DB or top < e[max(0, jm - ceil):jm].max() + JUMP_DB:
            continue
        starts.append(jm * hop)
        last = jm
    return list(zip(starts, starts[1:] + [x.shape[0]]))


def _dc(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-channel DC offset, read where the hit has decayed or not yet
    begun: the mean of the quietest tenth of the 10 ms blocks (lowest
    variance, which DC does not add to). It counts only when those blocks
    really are quiet (each block's spread under the offset itself, or under
    -40 dB of the peak) and agree with each other (spread of their means under
    a fifth of the offset; at least three blocks): a low tone still sounding
    at the end of the file has no quiet block, and a slice of its lobe is
    not DC. A file with any digitally silent block, or under 30 ms, has no
    offset."""
    n = max(1, int(round(0.010 * sr)))
    m = x.shape[0] // n
    zero = np.zeros(x.shape[1])
    if m < 3:
        return zero
    blocks = x[:m * n].reshape(m, n, x.shape[1])
    live = np.nonzero(np.abs(blocks).max(axis=(1, 2)) > 0)[0]   # digital zeros lie outside the signal
    if len(live) < 3:
        return zero
    blocks = blocks[live]
    var = blocks.var(axis=1)                                  # [m, channels]
    quiet = np.argsort(var.sum(axis=1))[:max(3, len(live) // 10)]      # at least three, to see them agree
    means = blocks[quiet].mean(axis=1)                        # [quiet, channels]
    dc = means.mean(axis=0)
    spread = np.sqrt(var[quiet]).max(axis=0)
    peak = np.abs(x).max()
    real = ((spread < np.abs(dc)) | (spread < 0.01 * peak)) & (means.std(axis=0) < 0.2 * np.abs(dc))
    return np.where(real, dc, 0.0)


def _onset(x: np.ndarray, mag: np.ndarray, sr: int, after_hit: bool, tail_s: float = 0.0) -> int:
    """Index of a hit's onset in x: the first 1 ms frame whose mean power is
    within ONSET_DB of the hit's loudest 1 ms, and in that frame the first
    sample at that level, so a lone click or dither spike does not count.
    After an earlier hit, the level must also be twice its tail's, the tail
    read from x's first 1 ms or, with `tail_s`, from its loudest 1 ms over the
    first `tail_s` seconds (one 1 ms frame of a low tone's tail is loud or quiet
    with the phase it falls on). 0 when nothing in x rises that far (or x
    starts with the hit)."""
    frame = max(1, int(round(0.001 * sr)))
    c = np.concatenate([[0.0], np.cumsum((x ** 2).mean(axis=1))])
    fwd = (c[frame:] - c[:-frame]) / frame if len(x) > frame else c[-1:] / max(1, len(x))
    floor = fwd.max() * 10 ** (-ONSET_DB / 10)
    level = math.sqrt(floor)
    if after_hit:              # rise out of the tail: four times its power, twice its peak
        span = max(frame, int(round(tail_s * sr)))
        floor = max(floor, 4.0 * float(fwd[:span - frame + 1].max()))
        level = max(level, 2.0 * float(mag[:span].max()))
    first = int(np.argmax(fwd >= floor))
    inside = np.nonzero(mag[first:first + frame] >= level)[0]
    return first + (int(inside[0]) if len(inside) else 0)


def _starts_inside(x: np.ndarray, mag: np.ndarray, sr: int, segs: list) -> bool:
    """Whether the take begins inside a hit that started before the file did:
    its first 1 ms is already within FAST_DB of hit 0's loudest 1 ms, and a
    later hit's peak is more than 3 dB above hit 0's."""
    s, e = segs[0]
    frame = max(1, int(round(0.001 * sr)))
    c = np.concatenate([[0.0], np.cumsum((x[s:e] ** 2).mean(axis=1))])
    fwd = (c[frame:] - c[:-frame]) / frame if e - s > frame else c[-1:]
    later = max(float(mag[a:b].max()) for a, b in segs[1:])
    return fwd[0] >= fwd.max() * 10 ** (-FAST_DB / 10) and later > float(mag[s:e].max()) * 10 ** (3 / 20)


CLEAN_BAND_DB = 6.0      # a clean candidate is within this of the loudest complete hit (a ghost note is not one)
LEAD_S = 0.020           # preferred: the 20 ms before its onset...
LEAD_DB = 20.0           # ...are at least this far under its peak (not the tail of an earlier hit)
ROOM_FRAC = 0.9          # "typical": room at least this share of the candidates' median...
TYPICAL_DB = 3.0         # ...and, preferably, a peak within this of their median peak
NEAR_S = 0.030           # a hit asked for "near" a position is taken within this
HIT_LEAD_S = 0.002       # hit_cut starts this long before the onset it finds
HIT_TAIL_S = 0.020       # hit_cut reads the tail's level over this (a whole period down to 50 Hz)


def _clean_hit(x: np.ndarray, sr: int, segs: list, onsets: list, mag: np.ndarray,
               within: tuple | None = None, near: float | None = None) -> tuple[int, dict, int, str]:
    """(index, measured profile, ms of room, note) of the clean hit.

    The candidates are the complete hits (not cut off before they fall 40 dB, long enough to
    measure), except that a take whose first hit begins at full level while a later one is more
    than 3 dB louder is read as starting inside a hit, and that first hit is not offered. With
    `within` = (start s, end s) only hits whose onset lies inside count, so the tail of a hit from
    before the span is never one. The pick: the candidates within CLEAN_BAND_DB of the loudest
    (a ghost note is not clean); of those the ones with a quiet lead-in (the 20 ms before the onset
    LEAD_DB under the peak), if any; of those the ones whose room (onset to the next onset, or the
    end of the take) is at least ROOM_FRAC of the median; of those the ones whose peak is within
    TYPICAL_DB of the median peak, if any; the earliest. With `near` (seconds), the complete hit
    whose onset is within NEAR_S of it wins instead, when there is one, whatever its level.
    Raises ValueError("no clean hit ...") when there is no candidate."""
    skip0 = len(segs) > 1 and _starts_inside(x, mag, sr, segs)
    complete = []
    for k, (_, end) in enumerate(segs):
        if k == 0 and skip0:
            continue
        if within is not None and not within[0] - NEAR_S <= onsets[k] / sr < within[1] + NEAR_S:
            continue                           # never a candidate (below): not worth measuring
        cut = "the file ends" if k == len(segs) - 1 else "the next hit starts"
        try:
            got = _measure(x[onsets[k]:end], sr, cut, quick=True)
        except ValueError:
            continue
        if got["truncated"]:
            continue
        lead = mag[max(0, onsets[k] - int(LEAD_S * sr)):onsets[k]]
        quiet = not len(lead) or 20 * math.log10(max(float(lead.max()), 1e-12)) <= got["peak_db"] - LEAD_DB
        room = (onsets[k + 1] if k + 1 < len(segs) else x.shape[0]) - onsets[k]
        complete.append({"k": k, "got": got, "room": room, "at": onsets[k] / sr, "quiet": quiet, "cut": cut})
    inside = complete
    if within is not None:
        inside = [c for c in complete if within[0] <= c["at"] < within[1]]
        slack = NEAR_S if near is not None else 0.0
        complete = [c for c in complete if within[0] - slack <= c["at"] < within[1] + slack]
    if not inside and near is not None:
        inside = complete                      # nothing inside the span, but a hit at the edge may be the one asked for
    if not inside:
        raise ValueError("no clean hit: none of the %d hit%s is complete (not cut off before falling 40 dB) "
                         "and inside the span asked for" % (len(segs), "" if len(segs) == 1 else "s"))
    c, why = None, None
    if near is not None:                       # like with like: the hit at that place, whatever the span's own band
        close = [c for c in complete if abs(c["at"] - near) <= NEAR_S]
        if close:
            c, why = min(close, key=lambda c: abs(c["at"] - near)), "the hit at the requested position"
    if c is None:
        loud = max(c["got"]["peak_db"] for c in inside)
        cands = [c for c in inside if c["got"]["peak_db"] >= loud - CLEAN_BAND_DB]
        cands = [c for c in cands if c["quiet"]] or cands
        med_room = float(np.median([c["room"] for c in cands]))
        roomy = [c for c in cands if c["room"] >= ROOM_FRAC * med_room]
        med_peak = float(np.median([c["got"]["peak_db"] for c in roomy]))
        typical = [c for c in roomy if abs(c["got"]["peak_db"] - med_peak) <= TYPICAL_DB]
        c = min(typical or roomy, key=lambda c: c["k"])
        why = "typical of %d candidate%s" % (len(roomy), "" if len(roomy) == 1 else "s")
    got = _measure(x[onsets[c["k"]]:segs[c["k"]][1]], sr, c["cut"])      # only the pick is measured in full
    return c["k"], got, int(round(_ms(c["room"], sr))), why


def hit_cut(x: np.ndarray, sr: int, at: float, length: float | None = None) -> tuple[np.ndarray, float | None]:
    """(cut, onset s) for the hit near `at` (seconds) in x [n, channels]. The onset is found here from the
    audio alone, not from where an earlier slice began: the first rise out of the tail within 100 ms after
    `at`, the tail's level read as its loudest 1 ms over HIT_TAIL_S from `at` - 30 ms, so one search never
    goes earlier than `at` - 10 ms. The search is then made again from the onset it found, until it finds the same
    one (at most four times, so up to about 40 ms earlier in all). That makes the onset of one rise out of a tail
    independent of where `at` fell; with grace notes or a swell a few ms apart it can still be any of the rises. The cut runs from HIT_LEAD_S before the onset to `length` seconds after it (the end of x if None).
    Two cuts of the same audio with the same `length` are the same samples whatever the offset they were read
    at. When nothing rises there the onset is None and the cut starts HIT_LEAD_S before `at`. Profile a cut
    with profile_array(cut, hit=0, single=True): the whole cut is the one hit."""
    x = np.asarray(x, dtype=np.float64)
    back, ahead = int(round(0.030 * sr)), int(round(0.100 * sr))

    def rise(i: int) -> int | None:
        a, b = max(0, i - back), min(len(x), i + ahead)
        win = x[a:b]
        k = _onset(win, np.abs(win).max(axis=1), sr, True, HIT_TAIL_S) if len(win) else 0
        return a + k if k > 0 else None            # with the tail read over 20 ms, a rise is never at the window's start

    onset = rise(int(round(at * sr)))
    for _ in range(4):
        again = None if onset is None else rise(onset)
        if again is None or again == onset:
            break
        onset = again
    start = int(round(at * sr)) if onset is None else onset
    end = len(x) if length is None else min(len(x), start + int(length * sr))
    return x[max(0, start - int(HIT_LEAD_S * sr)):max(end, start + 1)], None if onset is None else onset / sr


def profile_array(x: np.ndarray, sr: int, hit: int | str | None = None, within: tuple | None = None,
                  near: float | None = None, single: bool = False) -> dict:
    """The one-shot profile of one signal [n, channels]: of hit `hit` (from 0)
    when it holds several, by default the first that is not cut off; "clean"
    picks a typical complete hit (see profile()). `within` (start s, end s) and
    `near` (s) only apply to "clean": hits must start inside the span, and the
    hit starting within 30 ms of `near` is preferred. Raises ValueError("no clean
    hit ...") when there is no candidate. `single` takes the whole of x as one hit (hit 0), without
    looking for hits inside it: for a cut made with hit_cut."""
    if isinstance(hit, bool) or not (hit is None or hit == "clean" or isinstance(hit, (int, np.integer))):
        raise ValueError("hit must be a whole number, \"clean\" or None, got %r" % (hit,))
    check_channels(x)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("the file holds NaN or infinite samples")
    dc = _dc(x, sr)            # a DC offset would read as level that never decays
    if len(x) and np.abs(dc).max() > DC_REL * np.abs(x).max():
        nz = np.nonzero(np.abs(x).max(axis=1) > 0)[0]        # digital zeros around the take keep their 0
        x = x.copy()
        x[nz[0]:nz[-1] + 1] -= dc
    mag = np.abs(x).max(axis=1)
    if not len(mag) or mag.max() <= 0:
        raise ValueError("the file is silent")
    segs = [(0, x.shape[0])] if single else _segments(x, sr)
    if isinstance(hit, (int, np.integer)) and not 0 <= hit < len(segs):
        raise ValueError("hit %r is out of range: the take has %d hit%s (0-%d)"
                         % (hit, len(segs), "" if len(segs) == 1 else "s", len(segs) - 1))
    onsets = [_onset(x[s:e], mag[s:e], sr, s > 0) + s for s, e in segs]
    order, extra, tail = list(range(len(segs))), [], False
    picked = None
    if hit == "clean":
        k, got, room, why = _clean_hit(x, sr, segs, onsets, mag, within, near)
        picked, order = (k, got), []
        extra.append("hit %d is the clean hit (%s): complete, with %d ms from its onset to the next onset "
                     "or the end of the file" % (k, why, room))
    elif hit is not None:
        order = [int(hit)]
    elif len(segs) > 1 and _starts_inside(x, mag, sr, segs):
        # hit 0 looks like the tail of a hit that began before the file: try the
        # others first. A take trimmed at the onset of a fast hit looks the same,
        # so hit 0 stays the fallback when no later hit is complete.
        order, tail = order[1:] + [0], True
    for k in order:
        cut = "the file ends" if k == len(segs) - 1 else "the next hit starts"
        try:
            got = _measure(x[onsets[k]:segs[k][1]], sr, cut)
        except ValueError:
            if hit is not None or (picked is None and k == order[-1]):
                raise
            extra.append("hit %d is too short to measure" % k)
            continue
        if picked is None:
            picked = (k, got)
        if hit is not None or not got["truncated"]:
            picked = (k, got)
            break
    else:
        if len(order) > 1:
            extra.append("every hit is cut off before it falls 40 dB: this is the first")
    k, got = picked
    if tail and k != 0:
        extra.insert(0, "the take probably starts inside a hit: hit 0 starts at full level and a louder "
                        "hit follows, so the default is hit %d (pass hit=0 to measure hit 0)" % k)
    out = {"sample_rate": sr, "channels": x.shape[1], "hits": len(segs),
           "onsets_ms": [_ms(o, sr) for o in onsets], "hit": k, "onset_ms": _ms(onsets[k], sr)}
    out.update(got)
    out["notes"] = extra + out["notes"]
    return out


def _measure(y: np.ndarray, sr: int, cut: str, quick: bool = False) -> dict:
    """The profile of one hit y [n, channels], which starts at its onset and
    ends where `cut` says ("the file ends" or "the next hit starts"). `quick`
    returns only {"peak_db", "truncated"}, as the full profile has them."""
    notes = []
    peak = float(np.abs(y).max())
    if len(y) < FRAME_S[0] * sr:
        raise ValueError("the hit is shorter than %g ms" % (1000 * FRAME_S[0]))
    if len(y) > MAX_S * sr:
        y = y[:int(MAX_S * sr)]
        cut = "the %g s limit is reached" % MAX_S
        notes.append("only the first %g s after the onset are measured" % MAX_S)
    n, nyq_hi = len(y), min(20000.0, 0.45 * sr)
    power = (y ** 2).mean(axis=1)
    out = {"peak_db": round(to_db(peak * peak), 2)}

    # a first, pitch-blind envelope places the body window and the pitch gate
    rough = _boxcar(power, max(1, int(0.005 * sr)))
    rp = int(np.argmax(rough))
    t40 = _last_above(rough, rp, rough[rp] * 10 ** (-40 / 10))
    gate = _last_above(rough, rp, rough[rp] * 10 ** (-GATE_DB / 10))
    gate = n if gate is None else int(gate)
    t1 = min(BODY_S[1] * sr, n if t40 is None else t40)
    t0 = min(BODY_S[0] * sr, t1 / 2)
    b0, b1 = int(t0), int(t1)
    out["body_window_ms"] = [_ms(b0, sr), _ms(b1, sr)]

    f0, tonal, body = None, [], None
    if b1 - b0 >= int(FRAME_S[0] * sr):
        body = _spectrum(y, b0, b1, sr)
        tonal = _tonal(body, nyq_hi)
    if tonal:
        total, pool = sum(10 ** (t[1] / 10) for t in tonal), tonal
        while True:            # a low peak whose series holds too little is a rumble: try the next up
            f0, l1, i0, said = _fundamental(pool)
            series = sum(10 ** (t[1] / 10) for t in tonal if _harmonic(t[0], f0))
            higher = [t for t in pool if t[0] > f0 * (1 + HARM_TOL)]
            if series >= 0.5 * total or l1 >= tonal[0][1] - LONE_DB or not any(t[1] >= tonal[0][1] - STRONG_DB for t in higher):
                break
            notes.append("%s Hz at %.0f dB holds too little of the energy in its series: "
                         "taken as a rumble" % (_num(f0), l1 - tonal[0][1]))
            pool = higher
        notes += said
        stands = body["d"][i0] - (_floor(body, i0, both=True) or -np.inf)
        if series < 0.5 * total:
            f0 = None
            notes.append("no harmonic series: the partials are listed, no pitch is given")
        elif stands < PITCHED_DB and _noise_flatness(body, tonal, nyq_hi)[0] > NOISY_DB:
            f0 = None
            notes.append("the fundamental stands only %.0f dB above the noise around it and the body "
                         "is mostly noise: the partials are listed, no pitch is given" % stands)
    elif body is None:
        notes.append("the hit is too short for a body window")
    else:
        notes.append("no tonal peak in the body: unpitched")
    pitched = f0 is not None
    out["pitched"] = pitched

    # envelope over whole periods of the body pitch: the attack ends where it
    # first comes within ATTACK_DB of its peak (whole periods, so the start
    # phase does not move it, and a clipped plateau does not either)
    if pitched:
        period = sr / f0
        length = int(round(period * math.ceil(0.0025 * f0)))
    else:
        length = int(0.005 * sr)
    env = _boxcar(power, max(1, length))
    if quick:                  # what the clean-hit pick needs, read as below
        ip = int(np.argmax(env))
        return {"peak_db": out["peak_db"], "truncated": _last_above(env, ip, env[ip] * 10 ** (-FALLS_DB[-1] / 10)) is None}
    if pitched:                # harmonics on the body flattened by the whole-period envelope, so a
        flat = _spectrum(y, b0, b1, sr, level=env)      # fast decay does not smear f0 into 2 f0
        near = [t for t in _tonal(flat, nyq_hi) if abs(t[0] - f0) <= HARM_TOL * f0]
        lf = max(near, key=lambda t: t[1])[1] if near else _interp(flat, int(np.argmin(np.abs(flat["hz"] - f0))))[1]
    out["attack_ms"] = _ms(int(np.argmax(env >= env.max() * 10 ** (-ATTACK_DB / 10))), sr)
    ip = int(np.argmax(env))
    for drop in FALLS_DB:
        k = _last_above(env, ip, env[ip] * 10 ** (-drop / 10))
        out["fall_%d_ms" % drop] = None if k is None else _ms(k - ip, sr)
        if k is None:
            notes.append("%s before the hit falls %d dB" % (cut, drop))
    out["truncated"] = out["fall_40_ms"] is None
    k = _last_above(env, ip, env[ip] * 10 ** (-END_DB / 10))
    out["duration_ms"] = None if k is None else _ms(k, sr)
    if k is None:
        notes.append("%s before the hit falls %d dB: no duration" % (cut, END_DB))
    end = n if k is None else max(1, int(k))

    # pitch: start, body, drop, settle, track
    for key in ("pitch_start_hz", "pitch_start_at_ms", "pitch_body_hz", "pitch_drop_st",
                "pitch_settle_ms", "pitch_track_hz"):
        out[key] = None
    if pitched:
        out["pitch_body_hz"] = round(f0, 2)
        # a fundamental more than STRONG_DB under its strongest harmonic is lost in that
        # harmonic's skirt in a short frame: follow the harmonic and divide by its number
        k = round(tonal[0][0] / f0) if tonal[0][1] - l1 > STRONG_DB and _harmonic(tonal[0][0], f0) else 1
        z = _analytic(y)
        dense, reached = _track(z, sr, k * f0, round((b0 + b1) / 2 / sr, 3), gate, k)
        dense = [(c, f / k) for c, f in dense]
        if k > 1:
            notes.append("the fundamental is weak: the pitch is followed on harmonic %d" % k)
        early, onset_known = [], False
        if dense:
            c, fs = dense[0]
            zc = _early_pitch(y, sr, fs, c) if reached else None
            if zc is not None and not any(t < c for t, _ in zc[0]):
                zc = None              # after a skipped click, nothing is left before the first frame
            if zc is not None:
                early, since, skipped = zc
                early = [(t, f) for t, f in early if t < c]
                onset_known = since == 0.0      # the crossings cover the hit from its onset
                start_t, fs = early[0]
                if skipped:
                    notes.append("the first %.1f ms do not read as a pitch (a click or transient): the start "
                                 "is read after them" % (1000 * start_t))
            else:
                start_t = c
                if reached:
                    notes.append("the start pitch is the first pitch frame's, %.0f ms in: the zero crossings "
                                 "before it do not follow one pitch" % (1000 * c))
            if reached:
                out["pitch_start_hz"], out["pitch_start_at_ms"] = round(fs, 1), round(1000 * start_t, 1)
                out["pitch_drop_st"] = round(12 * math.log2(fs / f0), 2)
            else:              # the chain broke on its way back: its last frame is no start
                notes.append("the pitch cannot be followed back to the onset (the track starts %.0f ms in): "
                             "no start pitch or drop" % (1000 * c))
            seq = early + dense
            near = [abs(12 * math.log2(f / f0)) <= SETTLE_ST for _, f in seq]
            for i, (c0, _) in enumerate(seq):
                if seq[-1][0] - c0 < SETTLE_HOLD_S:
                    notes.append("the pitch is not followed for %.0f ms after it comes within %.1f st of the "
                                 "body pitch" % (1000 * SETTLE_HOLD_S, SETTLE_ST) if any(near[i:]) else
                                 "the pitch never settles within %.1f st of the body pitch" % SETTLE_ST)
                    break
                if all(ok for (cc, _), ok in zip(seq[i:], near[i:]) if cc - c0 <= SETTLE_HOLD_S):
                    if i > 0:
                        out["pitch_settle_ms"] = round(1000 * c0, 1)
                    elif onset_known:
                        out["pitch_settle_ms"] = 0.0
                    else:
                        notes.append("the pitch is already within %.1f st of the body pitch where it is "
                                     "first read, %.0f ms in" % (SETTLE_ST, 1000 * c0))
                    break
            else:
                notes.append("the pitch never settles within %.1f st of the body pitch" % SETTLE_ST)
        else:
            notes.append("no pitch track: the body pitch cannot be followed frame by frame")
        track = {}
        for ms in TRACK_MS:
            t = ms / 1000.0
            if dense and t < dense[0][0] - 0.0006:     # before the first frame: zero crossings or nothing
                if early and t >= since:     # read over the half-period that holds t
                    track[ms] = round(float(np.interp(t, [e[0] for e in early], [e[1] for e in early])), 1)
                else:
                    track[ms] = None
            else:
                got = [f for c, f in dense if abs(c - t) < 0.0006]   # frames sit on whole ms
                track[ms] = round(got[0], 1) if got else None
        if dense and any(track[ms] is None for ms in TRACK_MS if ms / 1000.0 < dense[0][0] - 0.0006):
            notes.append("pitch_track_hz has no value before %.0f ms, the first pitch frame's centre"
                         % (1000 * dense[0][0]))
        out["pitch_track_hz"] = track
    else:
        notes.append("unpitched: no pitch track and no harmonics")

    # tone: harmonics, inharmonic partials, noise, flatness, click
    out["harmonics_db"] = out["harmonics_floor_db"] = None
    few = pitched and (b1 - b0) * f0 / sr < MIN_PERIODS
    if few:
        notes.append("the body window holds under %d periods of the pitch: no harmonics, no body "
                     "noise or flatness" % MIN_PERIODS)
    if pitched and not few:
        h, fl = {}, {}
        peaks = _peaks(flat, nyq_hi)
        for k in range(2, 9):
            target = k * f0
            if target > nyq_hi:
                h[k] = fl[k] = None
                continue
            near = np.abs(flat["hz"] - target) <= max(HARM_TOL * target, flat["hz"][1])
            sel = np.intersect1d(np.nonzero(near)[0], peaks)
            i = int(sel[np.argmax(flat["d"][sel])]) if len(sel) else int(np.argmin(np.abs(flat["hz"] - target)))
            floor = _floor(flat, i)
            fl[k] = None if floor is None else round(max(FLOOR_DB, floor - lf), 1)
            if len(sel):
                h[k] = round(max(FLOOR_DB, _interp(flat, i)[1] - lf), 1)
            else:              # no peak of its own: the floor there, an upper bound
                h[k] = fl[k] if fl[k] is not None else FLOOR_DB
        out["harmonics_db"], out["harmonics_floor_db"] = h, fl
    ref = l1 if pitched else (tonal[0][1] if tonal else None)
    partials = [t for t in tonal if t[1] >= tonal[0][1] - PARTIALS_DB and not (pitched and _harmonic(t[0], f0))]
    out["partials"] = [{"hz": round(t[0], 1), "db": round(t[1] - ref, 1)} for t in partials[:MAX_PARTIALS]]
    a1 = min(n, int(ATTACK_S * sr))
    # the attack window, flattened by the envelope so a fast decay is not noise, ends
    # where the envelope is ATTACK_RANGE_DB down: past that, flattening would lift the
    # file's own noise floor (a fast hit recorded quietly)
    low = np.nonzero(env[ip:a1] < env[ip] * 10 ** (-ATTACK_RANGE_DB / 10))[0]
    a2 = max(int(FRAME_S[0] * sr), ip + int(low[0])) if len(low) else a1
    att = _spectrum(y, 0, min(a1, a2), sr, level=env)
    out["noise_db"], out["flatness"] = {}, {}
    for name, s in (("attack", att), ("body", None if few else body)):
        if s is None:
            out["noise_db"][name] = out["flatness"][name] = None
            continue
        nz, fl = _noise_flatness(s, _tonal(s, nyq_hi) if s is att else tonal, nyq_hi)
        out["noise_db"][name] = None if nz is None else round(nz, 1)
        out["flatness"][name] = None if fl is None else round(fl, 3)
    # click: the attack's power above 2 kHz against the attack's own power, so
    # the decay and the body's level do not move it
    nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    spec = np.fft.rfft(y, nfft, axis=0)
    spec[np.fft.rfftfreq(nfft, 1.0 / sr) < CLICK_HZ] = 0
    hf = np.fft.irfft(spec, nfft, axis=0)[:a1]
    pa = float(power[:a1].mean())
    out["click_db"] = round(to_db(float((hf ** 2).mean()) / pa), 1) if pa > 0 else None

    # balance and width over the whole hit
    hit = y[:end]
    spec = (np.abs(np.fft.rfft(hit, axis=0)) ** 2).mean(axis=1)
    hz = np.fft.rfftfreq(len(hit), 1.0 / sr)
    bands = {name: band_power(spec, hz, lo, hi) for name, lo, hi in BANDS}
    total = sum(bands.values())
    out["bands_db"] = {k: round(to_db(v / total), 1) for k, v in bands.items()} if total > 0 else None
    if hit.shape[1] == 2:
        pm = float((((hit[:, 0] + hit[:, 1]) / 2) ** 2).mean())
        ps = float((((hit[:, 0] - hit[:, 1]) / 2) ** 2).mean())
        out["width"] = round(math.sqrt(ps) / (math.sqrt(pm) + math.sqrt(ps)), 3) if pm + ps > 0 else None
    else:
        out["width"] = 0.0
    out["notes"] = notes
    return out


def profile(path: str | pathlib.Path, hit: int | str | None = None) -> dict:
    """Envelope, pitch and tone of a one-shot file, relative to the hit itself.

    hits, onsets_ms the hits found in the file and where each starts
    hit             which one this profile is (from 0): `hit`, by default the first
                    that is not cut off; hit="clean" asks for a typical complete hit. Of the
                    hits that are not cut off, those within 6 dB of the loudest; of those the
                    ones that start from quiet (the 20 ms before the onset 20 dB under the
                    peak), if any; of those the ones with room (onset to the next onset or the
                    end of the file) at least 90 % of the median; of those the ones with a
                    peak within 3 dB of the median peak, if any; the earliest. A take whose
                    first hit begins at full level while a later one is more than 3 dB louder
                    is read as starting inside a hit, and that first hit is not offered.
                    Raises "no clean hit" when no hit is complete
    onset_ms        where the hit starts in the file (first 1 ms within 40 dB of its loudest)
    truncated       the next hit or the end of the file cuts it before it falls 40 dB
    peak_db         sample peak, dBFS (the only absolute level here)
    attack_ms       onset to the envelope peak (first within 1 dB of it, over whole periods)
    fall_10/20/40_ms  envelope peak to the last moment above peak - 10/20/40 dB
    duration_ms     onset to the last moment above peak - 60 dB (None on a cut hit)
    pitched         false for noise, a cymbal or an inharmonic sound: no pitch, no
                    harmonics (the fundamental must stand 25 dB above the noise around
                    it, or 15 dB in a body that is not mostly noise)
    pitch_start_hz  pitch at the onset, from zero crossings (or where pitch_start_at_ms
                    says: after a click, or at the first frame when the crossings
                    do not follow one pitch); None when the track cannot be followed
                    back to the onset
    pitch_body_hz   pitch over body_window_ms (40-150 ms, shorter for a short hit)
    pitch_drop_st   start minus body, semitones
    pitch_settle_ms when the pitch comes within 0.5 st of the body pitch and stays 10 ms
                    (None when it never does, or is already there where first read)
    pitch_track_hz  pitch at 0, 5, 10, 20, 40, 80, 150, 300 ms (None where nothing reads
                    it: before the first crossing or frame, once the hit is 50 dB down)
    harmonics_db    harmonics 2-8 against the fundamental, on the body (None above
                    min(20 kHz, 0.45 sr); a harmonic with no peak of its own reads the
                    floor there; all None when the body holds under 4 periods)
    harmonics_floor_db  the spectral floor around each of those, so a value near it
                    is the floor, not a harmonic
    partials        strongest peaks that are not harmonics: Hz and dB against the
                    fundamental (or the strongest peak when unpitched)
    noise_db        energy outside the tonal peaks against the total, attack (0-30 ms) and body
    flatness        spectral flatness, attack and body: 0 pure tones .. ~0.56 white noise
    click_db        attack-window (0-30 ms) power above 2 kHz against the attack's
                    total: a click, or any bright attack (a broadband noise layer too)
    bands_db        six-band balance against the hit's 20 Hz-20 kHz total
    width           0 = mono, 0.5 = equal mid and side, 1 = all side
    notes           why a value is None
    """
    x, sr = load(path)
    out = {"file": str(path)}
    out.update(profile_array(x, sr, hit))
    return out


TIMES = (("attack_ms", "attack %s ms, reference %s ms"),
         ("fall_10_ms", "falls 10 dB in %s ms, reference %s ms"),
         ("fall_20_ms", "falls 20 dB in %s ms, reference %s ms"),
         ("fall_40_ms", "falls 40 dB in %s ms, reference %s ms"),
         ("duration_ms", "lasts %s ms, reference %s ms"),
         ("pitch_settle_ms", "pitch settles in %s ms, reference %s ms"))


def _drop(p: dict) -> str:
    """A profile's pitch drop in words, with how long it takes: a 17 st drop
    over 3 ms is a click, the same drop over 40 ms is a kick's glide."""
    d, t = p["pitch_drop_st"], p.get("pitch_settle_ms")
    text = "%s %.1f st" % ("drops" if d > -0.05 else "rises", abs(d))
    if t is not None and abs(d) > DROP_ST:
        text += " over %s ms" % _num(t)
        if t < FAST_SETTLE_MS:
            text += " (a click)"
    return text


def _key(d: dict, k: int):
    """d[k] for a profile fresh from profile() (int keys) or read back from JSON (str keys)."""
    return d.get(k, d.get(str(k)))


def _num(v: float) -> str:
    return "%.1f" % v if abs(v) < 10 else "%.0f" % v


def _ordinal(k: int) -> str:
    return {2: "2nd", 3: "3rd"}.get(k, "%dth" % k)


def _pitch_at(p: dict, ms: float) -> float | None:
    """A profile's pitch at `ms`, from its start and track points (log-linear
    between the two around it), or None when no reading brackets that time."""
    pts = {}
    if p.get("pitch_start_hz") and p.get("pitch_start_at_ms") is not None:
        pts[float(p["pitch_start_at_ms"])] = p["pitch_start_hz"]
    for k, v in (p.get("pitch_track_hz") or {}).items():
        if v:
            pts.setdefault(float(k), v)
    t = sorted(pts)
    below, above = [x for x in t if x <= ms], [x for x in t if x >= ms]
    if not below or not above:
        return None
    t0, t1 = below[-1], above[0]
    if t1 == t0:
        return pts[t0]
    return 2 ** np.interp(ms, [t0, t1], [math.log2(pts[t0]), math.log2(pts[t1])])


def _flip(a: dict, b: dict) -> str | None:
    """A line when the two body pitches sit 2x or 3x apart and the higher one
    lists the lower pitch among its partials: the same tone read on two
    different partials, not a pitch difference."""
    fa, fb = a.get("pitch_body_hz"), b.get("pitch_body_hz")
    if not (fa and fb):
        return None
    hi, lo, hi_f, lo_f = (a, b, fa, fb) if fa > fb else (b, a, fb, fa)
    for k, interval in ((2, "an octave"), (3, "a twelfth"), (4, "two octaves")):
        if abs(12 * math.log2(hi_f / lo_f) - 12 * math.log2(k)) <= PITCH_ST:
            low = [q for q in hi.get("partials") or [] if abs(q["hz"] - lo_f) <= HARM_TOL * lo_f]
            if low:
                side = "this side" if hi is a else "the reference"
                return ("pitch read %s %s the reference (%s Hz, reference %s Hz) on the same partials: %s "
                        "takes its %s Hz partial as the fundamental and its %s Hz one, %.0f dB down, as a "
                        "rumble; harmonic and drop readings are left out"
                        % (interval, "above" if hi is a else "below", _num(fa), _num(fb), side,
                           _num(hi_f), _num(low[0]["hz"]), -low[0]["db"]))
    return None


def compare(a: dict, b: dict) -> dict:
    """How profile `a` (the candidate, e.g. a synth take) differs from profile
    `b` (the reference). `diff` holds a - b for every measure both sides have:
    pitch in semitones, times as {"ms", "ratio"} (a / b), the rest in their
    own units. `readings` is one line per difference past its threshold,
    largest first relative to that threshold. A measure that is None on
    either side gives no reading, and levels below the method's floor on both
    sides (harmonics under -50 dB or near the noise, noise share under -25 dB,
    click under -40 dB) are not read. The six-band balance is read only when
    a side is unpitched: between two pitched hits it mostly restates the
    pitch. A pitch-drop line gives the time each drop takes."""
    diff, found = {}, []

    def add(score, text):
        found.append((score, text))

    if a.get("pitched") != b.get("pitched"):
        add(math.inf, "pitched (%s Hz), the reference is not" % _num(a["pitch_body_hz"]) if a.get("pitched")
            else "unpitched, the reference is pitched (%s Hz)" % _num(b["pitch_body_hz"]))
    for key, name in (("pitch_body_hz", "pitch_body_st"), ("pitch_start_hz", "pitch_start_st")):
        va, vb = a.get(key), b.get(key)
        diff[name] = round(12 * math.log2(va / vb), 2) if va and vb else None
    def sub(key, nd):
        va, vb = a.get(key), b.get(key)
        return (round(va - vb, nd) if va is not None and vb is not None else None), va, vb

    flip = _flip(a, b)
    if flip:
        add(1e6, flip)
    d = diff["pitch_body_st"]
    if d is not None and abs(d) > PITCH_ST and not flip:
        add(abs(d) / PITCH_ST, "body pitch %.1f st %s the reference (%s Hz, reference %s Hz)"
            % (abs(d), "above" if d > 0 else "below", _num(a["pitch_body_hz"]), _num(b["pitch_body_hz"])))
    d, va, vb = sub("pitch_drop_st", 2)
    ta, tb = a.get("pitch_start_at_ms"), b.get("pitch_start_at_ms")
    drop_read = False
    if d is not None and ta is not None and tb is not None and abs(ta - tb) > START_GAP_MS:
        # the starts were read at different times: a drop from 0 ms and one from 20 ms
        # are not the same measure, so both are taken from the later time
        late = max(ta, tb)
        pa, pb = _pitch_at(a, late), _pitch_at(b, late)
        diff["pitch_drop_from_ms"] = late
        if pa and pb:
            da, db_ = 12 * math.log2(pa / a["pitch_body_hz"]), 12 * math.log2(pb / b["pitch_body_hz"])
            d = diff["pitch_drop_st"] = round(da - db_, 2)
            if abs(d) > DROP_ST and not flip:
                drop_read = True
                add(abs(d) / DROP_ST, "pitch %s %.1f st from %s ms on, reference %.1f st (the starts are read "
                    "at %s and %s ms)" % ("drops" if da > -0.05 else "rises", abs(da), _num(late),
                                          db_, _num(ta), _num(tb)))
        else:
            diff["pitch_drop_st"] = None
            if max(abs(va), abs(vb)) > DROP_ST and not flip:
                add(1.0, "pitch drop not compared: the start is read at %s ms on one side and %s ms on "
                    "the other" % (_num(ta), _num(tb)))
    else:
        diff["pitch_drop_st"] = d
        if d is not None and abs(d) > DROP_ST and not flip:
            drop_read = True
            add(abs(d) / DROP_ST, "pitch %s, reference %s" % (_drop(a), _drop(b)))

    for key, text in TIMES:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            diff[key] = None
            continue
        d, ratio = va - vb, (va / vb if vb > 0 else None)
        diff[key] = {"ms": round(d, 1), "ratio": None if ratio is None else round(ratio, 3)}
        if key == "pitch_settle_ms" and (drop_read or all(
                abs(p.get("pitch_drop_st") or 0.0) <= DROP_ST for p in (a, b))):
            continue                # the pitch-drop line gives both times; with no drop on
                                    # either side there is nothing to settle from
        # read on |ln ratio|, so 15 % longer and 15 % shorter (ratio 1/1.15) weigh the same
        far = None if not ratio else abs(math.log(ratio)) / math.log(1 + TIME_RATIO)
        if abs(d) > TIME_FLOOR_MS and (far is None or far > 1):
            line = text % (_num(va), _num(vb))
            if ratio is not None:
                line += " (%d %% %s)" % (round(abs(ratio - 1) * 100), "longer" if d > 0 else "shorter")
            add(far if far is not None else abs(d) / TIME_FLOOR_MS, line)

    ha, hb = a.get("harmonics_db"), b.get("harmonics_db")
    diff["harmonics_db"] = None
    if ha and hb:
        diff["harmonics_db"] = {}

        def heard(p, h, k):
            v, floor = _key(h, k), _key(p.get("harmonics_floor_db") or {}, k)
            return v is not None and v > HARM_FLOOR_DB and (floor is None or v - floor >= HARM_CLEAR_DB)
        for k in range(2, 9):
            va, vb = _key(ha, k), _key(hb, k)
            diff["harmonics_db"][k] = None if va is None or vb is None else round(va - vb, 1)
            d = diff["harmonics_db"][k]
            if d is not None and abs(d) > LEVEL_DB and (heard(a, ha, k) or heard(b, hb, k)) and not flip:
                add(abs(d) / LEVEL_DB, "%s harmonic %.1f dB %s (%s dB, reference %s dB)"
                    % (_ordinal(k), abs(d), "high" if d > 0 else "low", _num(va), _num(vb)))

    for key, floor, step, word in (("noise_db", NOISE_FLOOR_DB, LEVEL_DB, "noise"),
                                   ("flatness", None, None, None)):
        diff[key] = {}
        for part in ("attack", "body"):
            va, vb = (a.get(key) or {}).get(part), (b.get(key) or {}).get(part)
            diff[key][part] = None if va is None or vb is None else round(va - vb, 3 if word is None else 1)
            d = diff[key][part]
            if word and d is not None and abs(d) > step and max(va, vb) > floor:
                add(abs(d) / step, "%s %.1f dB %s than the reference in the %s (%s dB of the total, reference %s dB)"
                    % (word, abs(d), "more" if d > 0 else "less", part, _num(va), _num(vb)))
    d, va, vb = sub("click_db", 1)
    diff["click_db"] = d
    if d is not None and abs(d) > CLICK_STEP_DB and max(va, vb) > CLICK_FLOOR_DB:
        add(abs(d) / CLICK_STEP_DB, "click %.1f dB %s than the reference" % (abs(d), "stronger" if d > 0 else "weaker"))
    d, va, vb = sub("width", 3)
    diff["width"] = d
    if d is not None and abs(d) > WIDTH_STEP:
        add(abs(d) / WIDTH_STEP, "width %.2f, reference %.2f (%s)" % (va, vb, "wider" if d > 0 else "narrower"))
    ba, bb = a.get("bands_db"), b.get("bands_db")
    diff["bands_db"] = {k: round(ba[k] - bb[k], 1) for k in ba if k in bb} if ba and bb else None
    if diff["bands_db"] and not (a.get("pitched") and b.get("pitched")):
        # with no pitch or harmonics to go by, the band balance is the tone. Read it
        # against the median band difference, as listen.compare does: losing the
        # highs must not read as every other band being louder
        offset = float(np.median(list(diff["bands_db"].values())))
        for k, d in diff["bands_db"].items():
            d -= offset
            if abs(d) > LEVEL_DB and max(ba[k], bb[k]) > BAND_FLOOR_DB:
                add(abs(d) / LEVEL_DB, "%s %.1f dB %s against the other bands"
                    % (BAND_NAMES.get(k, k), abs(d), "high" if d > 0 else "low"))
    diff["peak_db"] = sub("peak_db", 2)[0]
    return {"a": a, "b": b, "diff": diff, "readings": [t for _, t in sorted(found, key=lambda f: -f[0])]}

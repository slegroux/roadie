#!/usr/bin/env python3
"""Song-section detection for yt2live, for placing Ableton LOCATORS.

Finds where the arrangement CHANGES, snaps every change to an 8-bar phrase
boundary, and describes each resulting section by how loud it is relative to the
track. The output is a list of boundaries suitable for `create_cue_point`.

WHAT THE LABELS ARE. Two levels, and the distinction matters. Every section
carries `level` — a raw loudness bucket ("A quiet", "B full") from one number,
the section's mean beat-synchronous RMS over the track's — and `label`, an
arrangement reading built on top of it: Intro, Build, Drop, Main, Break, Outro.

`label` is RULES, not recognition. There is no model here; nothing in this file
knows what a drop sounds like. What makes the names better than chance is that
loudness alone was never the only signal available — POSITION and CONTRAST are
too. The first section is an intro and the last an outro by arrangement
convention; loud straight after quiet is the drop gesture; quiet after loud is a
breakdown; and a mid-energy section is a BUILD only because a full one follows
it, which is why labelling needs a second pass over the whole list rather than
one pass per section. An earlier version reported only the loudness bucket, on
the grounds that RMS cannot see structure. That was true and still under-sold
what order and contrast carry.

Expect odd names where the conventions do not hold: a track that opens loud, or
drops without a preceding break. The BOUNDARIES remain the trustworthy part —
they come from timbre and harmony (MFCC + chroma) and land where the arrangement
really turns over. Read `label` as a reading of them, not as ground truth.

WHY THE SNAP MATTERS. `librosa.segment.agglomerative` returns boundaries at beat
resolution, and on electronic material they land one or two beats off the phrase
line — musically wrong, and visibly wrong in Live's arrangement, where a locator
a beat before the bar reads as a mistake. Electronic tracks change on 8-bar
phrases, so every boundary is rounded to the nearest phrase line — 32 beats
apart, counted from the detected DOWNBEAT rather than from grid beat 0. Several
raw boundaries routinely round onto the SAME phrase, which is why the result is
deduplicated rather than assumed distinct.

UNITS, and the one that bites. Sections are measured in SOURCE-FILE time,
counted from the start of the stem audio. Locators live in ARRANGEMENT time. The
two differ by `start_offset_beats`, the nudge `stems2live` applies to every clip
— up to 31 beats with `--phrase 8`. `sections_to_locators` does that conversion
and takes the offset explicitly; nothing here defaults it to zero silently.

This module never talks to Live. It returns data; the caller writes cue points.
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
import numpy as np

from beat_aligner import (ANALYSIS_SR, HOP_LENGTH, N_FFT, build_analysis_mix,
                          cached_grid, estimate_aubio_beats, fit_beat_grid)

try:
    import librosa
except ImportError:
    librosa = None

# Beats per bar. Everything downstream counts bars, so this is not a free
# parameter — 3/4 material would need the phrase logic revisited, not just this.
BEATS_PER_BAR = 4

# Phrase length in bars. Sections in house/techno/trance change on 8-bar
# boundaries; 4 is too fine (it snaps to the wrong side of a real 8-bar change
# as often as the right one) and 16 merges genuinely separate 8-bar sections.
PHRASE_BARS = 8

# Number of MFCCs. 13 is the usual timbral summary — enough to separate a
# filtered breakdown from a full arrangement, few enough that the top
# coefficients (which are mostly noise at beat resolution) do not dominate.
N_MFCC = 13

# How many 8-bar phrases one requested segment should cover, before snapping.
# Measured section lengths on real electronic material run 2-4 phrases, so 1.6
# DELIBERATELY OVER-SEGMENTS. That asymmetry is the point: snapping and
# deduplication can only ever remove boundaries, never invent one, so asking for
# too many is recoverable and asking for too few is not. On the 307 s test track
# this yields k=12 raw boundaries which collapse to 8 distinct phrases.
PHRASES_PER_SEGMENT = 1.6

# Floor and ceiling on the requested segment count. The floor keeps a two-minute
# edit from being described as a single block; the ceiling stops a 20-minute DJ
# set from asking for 60 boundaries that all snap together anyway.
MIN_SEGMENTS = 4
MAX_SEGMENTS = 32

# Energy thresholds, as a ratio of the section's mean beat RMS to the track's.
# Not symmetric around 1.0 on purpose: RMS is dominated by the loud sections, so
# the track mean sits above the midpoint and a symmetric pair would call almost
# everything "quiet". Measured on the test track, these split 0.10-0.59 (breaks)
# from 0.89 (mid) from 1.13-1.31 (full).
QUIET_REL = 0.75
FULL_REL = 1.12

# Stems preferred for beat tracking, cleanest pulse first. Same order as
# beat_aligner's, so a locator grid cannot disagree with the clip grid.
TEMPO_STEM_ORDER = ("drums_kick", "drums", "bass", "mix")


def stem_paths(stems_dir):
    """Map stem name -> path for the audio files in a yt2stems folder.

    The `endswith` filter is not decoration. Live drops a `drums_kick.flac.asd`
    analysis file next to the stems, and a bare glob picks it up and hands
    libsndfile a binary blob it cannot open.
    """
    paths = {}
    for f in sorted(os.listdir(stems_dir)):
        if f.endswith((".flac", ".wav")):
            paths[os.path.splitext(f)[0]] = os.path.join(stems_dir, f)
    return paths


def _tempo_stem(paths):
    """Path of the stem to fit the beat grid on, or None."""
    for name in TEMPO_STEM_ORDER:
        if name in paths:
            return paths[name]
    return next(iter(paths.values()), None)


def choose_segment_count(n_beats, phrase_beats=PHRASE_BARS * BEATS_PER_BAR):
    """How many raw segments to ask agglomerative clustering for.

    Scaled by track LENGTH IN PHRASES rather than fixed, because a fixed count
    means opposite things at opposite lengths: 12 segments over three minutes is
    one boundary every 1.6 phrases, and 12 over ten minutes is one every 5 — the
    first over-segments usefully, the second misses whole sections. Measured in
    phrases rather than seconds so the answer does not change with tempo.

    See PHRASES_PER_SEGMENT for why the request deliberately over-segments, and
    MIN_SEGMENTS / MAX_SEGMENTS for the clamps.
    """
    n_phrases = float(n_beats) / float(phrase_beats)
    k = int(round(n_phrases / PHRASES_PER_SEGMENT))
    return int(max(MIN_SEGMENTS, min(MAX_SEGMENTS, k)))


def build_beat_grid_times(period, t0, duration_sec):
    """Ideal grid times from the sample start, as (times, k_start).

    Every beat slot the audio actually contains, not just the ones a detector
    found. Boundaries are reported in these grid beats, so a beat aubio dropped
    inside a breakdown still has an index and a phrase position — the same reason
    beat_aligner fits a grid instead of indexing the detection array.
    """
    if period <= 0:
        raise ValueError("beat period must be positive, got %r" % period)
    k_start = int(np.ceil(-t0 / period)) if t0 < 0 else 0
    n = int(np.floor((duration_sec - (period * k_start + t0)) / period))
    if n < 1:
        raise ValueError("no beat slots fit in %.2f s at period %.4f s"
                         % (duration_sec, period))
    k = np.arange(k_start, k_start + n + 1)
    return period * k + t0, k_start


def beat_sync_features(y, sr, grid_times):
    """Beat-synchronous (MFCC + chroma + RMS) matrix and the per-beat RMS row.

    Returns (F, rms_per_beat) where F has one column per grid beat, z-scored per
    row, and rms_per_beat is the RAW (un-normalised) beat RMS the labels use.

    Aggregating to beats rather than to frames is what makes the segmentation
    musical: at hop 512 a 5-minute track is 13k frames of mostly transient
    detail, and clustering that finds bar-level texture changes rather than
    section changes. One column per beat puts the clustering at the scale
    arrangement decisions are actually made on.

    Rows are z-scored so an MFCC coefficient with a large native range cannot
    outweigh chroma. RMS is deliberately left as a single row against 25 rows of
    timbre and harmony — it contributes to the boundaries but does not decide
    them, which is what keeps a mid-section volume dip from reading as a new
    section.
    """
    if librosa is None:
        raise RuntimeError("librosa is required for section detection")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    n_frames = min(mfcc.shape[1], chroma.shape[1], rms.shape[1])
    mfcc, chroma, rms = mfcc[:, :n_frames], chroma[:, :n_frames], rms[:, :n_frames]

    frames = librosa.time_to_frames(np.asarray(grid_times, dtype=float),
                                    sr=sr, hop_length=HOP_LENGTH)
    frames = frames[(frames >= 0) & (frames < n_frames)]
    if len(frames) < 2:
        raise ValueError("only %d beat slots land inside the audio" % len(frames))
    if np.any(np.diff(frames) < 1):
        raise ValueError(
            "beat period is shorter than the analysis hop — two beats share a "
            "frame, so they cannot be summarised separately")

    # librosa.util.sync inserts a leading interval for the frames BEFORE the
    # first index whenever that index is non-zero, which it is here (t0 > 0 on
    # any real track). Dropping it realigns column i onto beat i; without this
    # every reported boundary sits one beat late.
    def _sync(x):
        s = librosa.util.sync(x, frames, aggregate=np.mean)
        return s[:, 1:] if frames[0] > 0 else s

    parts = np.vstack([_sync(mfcc), _sync(chroma), _sync(rms)])
    rms_per_beat = _sync(rms)[0]

    mean = parts.mean(axis=1, keepdims=True)
    std = parts.std(axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    return (parts - mean) / std, rms_per_beat


def snap_boundaries(raw_beats, phrase_beats, n_beats, phase=0):
    """Round beat indices onto phrase lines, deduplicate, keep the first one.

    Phrase lines are `phase + n * phrase_beats`: counted from the DOWNBEAT, not
    from grid beat 0. `phase` is the grid beat the first downbeat falls on
    (0-3). Snapping to plain multiples of 32 is only right when the file happens
    to start on a downbeat; with the downbeat on grid beat 2, every boundary
    would land two beats before the bar line it marks. The first section starts
    AT the downbeat, so the 1-3 beat pickup before it belongs to no section —
    stems2live trims that lead-in off the clips anyway.

    Rounds to the NEAREST phrase rather than flooring: a change detected one or
    two beats early — which is normal, since a riser or a filter sweep starts
    before the phrase it announces — floors to the phrase before it, putting the
    locator a full 8 bars early. Nearest puts it on the phrase the section
    actually starts on.

    The `set` is load-bearing. Adjacent raw boundaries routinely round onto the
    same phrase, and a duplicate cue point in Live is a real one that silently
    replaces its neighbour.
    """
    if phrase_beats <= 0:
        raise ValueError("phrase length must be positive, got %r" % phrase_beats)
    phase = int(phase)
    snapped = {phase}
    for b in raw_beats:
        s = int(round((float(b) - phase) / phrase_beats)) * phrase_beats + phase
        if phase <= s < n_beats:
            snapped.add(s)
    return sorted(snapped)


def label_for(rel_energy, index):
    """Bucketed loudness only, e.g. "C full". See `arrangement_labels` for form.

    Kept as the raw measurement, and used as the fallback when the structural
    pass cannot say anything more specific.
    """
    level = ("quiet" if rel_energy < QUIET_REL else
             "full" if rel_energy > FULL_REL else "mid")
    return "%s %s" % (_letter(index), level)


def arrangement_labels(rel_energies):
    """Name sections in dance-music terms: intro, build, drop, break, outro.

    Loudness ALONE cannot do this, which is why the first version of this module
    only reported "quiet"/"mid"/"full". What carries the extra information is
    position and CONTRAST — where a section sits in the track, and how its energy
    compares with the section before it:

      - the first section is the intro, and the last is the outro; in dance
        arrangements those are near-fixed conventions rather than inferences
      - a loud section entering straight after a quiet one is a DROP; that
        contrast is the defining gesture of the form
      - a quiet section after a loud one is a BREAKDOWN
      - a mid-energy section immediately BEFORE a drop is a build; a build is
        only a build because of what follows it, so this has to look ahead
      - a loud section following another loud one is a continuation, not a
        second drop

    Still inference, not ground truth: it reads energy and order, and it has no
    notion of melody, vocals or arrangement. A track that opens loud, or drops
    without a preceding break, will be labelled oddly. The BOUNDARIES remain the
    measured part — these names are a reading of them.
    """
    n = len(rel_energies)
    if n == 0:
        return []
    if n == 1:
        return ["Intro"]

    def level(x):
        return "quiet" if x < QUIET_REL else "full" if x > FULL_REL else "mid"

    lv = [level(r) for r in rel_energies]
    out = [None] * n
    out[0] = "Intro"
    out[-1] = "Outro"

    for i in range(1, n - 1):
        prev, cur = lv[i - 1], lv[i]
        nxt = lv[i + 1] if i + 1 < n else None
        if cur == "full":
            out[i] = "Drop" if prev in ("quiet", "mid") else "Main"
        elif cur == "quiet":
            out[i] = "Break" if prev == "full" else "Quiet"
        else:                                    # mid
            # A build is defined by what comes next, not by its own level.
            out[i] = "Build" if nxt == "full" else "Mid"

    # Number repeats so two drops are distinguishable in the arrangement.
    counts = {}
    for i, name in enumerate(out):
        counts[name] = counts.get(name, 0) + 1
    seen = {}
    total = counts
    for i, name in enumerate(out):
        if total[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out[i] = "%s %d" % (name, seen[name])
    return out


def _letter(i):
    """A, B, ... Z, AA, AB, ... for section index i."""
    name = ""
    i = int(i)
    while True:
        name = chr(ord("A") + i % 26) + name
        i = i // 26 - 1
        if i < 0:
            return name


def downbeat_phase(period, grid_t0, downbeat_sec):
    """Grid beat (0-3) the bar line falls on, or 0 when no downbeat is known.

    Section beats are counted from `grid_t0`, so this is the beat index of the
    detected downbeat, folded into one bar.
    """
    if downbeat_sec is None:
        return 0
    k = round((float(downbeat_sec) - float(grid_t0)) / float(period))
    return k % BEATS_PER_BAR


def analyze_sections(stems_dir, phrase_bars=PHRASE_BARS, n_segments=None,
                     period=None, grid_t0=None, sr=ANALYSIS_SR,
                     downbeat_sec=None):
    """Full section analysis. Returns a dict with `sections` plus the grid it used.

    `downbeat_sec` is beat_aligner's detected downbeat, in source-file seconds.
    Boundaries snap to phrases counted from it (see `snap_boundaries`). Without
    it they snap from grid beat 0, which is only right on a track whose first
    grid beat is a downbeat.

    `period` and `grid_t0` skip beat tracking entirely. Pass beat_aligner's own
    `period` / `grid_t0` whenever they are already in hand: locators derived from
    a second, independently fitted grid can sit a beat off the clips they are
    meant to mark, and re-running aubio to rediscover a number the caller already
    has is a minute of wasted decoding.
    """
    if librosa is None:
        raise RuntimeError("librosa is required for section detection")

    paths = stem_paths(stems_dir) if os.path.isdir(stems_dir) else \
        {"mix": stems_dir}
    if not paths:
        raise ValueError("no .flac or .wav stems found in %s" % stems_dir)

    if period is None or grid_t0 is None:
        primary = _tempo_stem(paths)
        beats = estimate_aubio_beats(primary)
        if beats is None:
            raise RuntimeError("could not extract beats from %s" % primary)
        period, grid_t0, _k, _inliers = fit_beat_grid(beats)
    period, grid_t0 = float(period), float(grid_t0)

    # The same summed mono mix beat_aligner uses for phase, including its rule
    # that the `drums` composite is dropped when `drums_*` parts are present —
    # keeping both plays the kit twice and inflates the RMS the labels read.
    y, sr, mix_names = build_analysis_mix(paths, sr=sr)
    if y is None or not len(y):
        raise RuntimeError("analysis mix is empty")
    duration = len(y) / float(sr)

    grid_times, k_start = build_beat_grid_times(period, grid_t0, duration)
    # A grid_t0 a hair under zero puts its first slot before the audio, so the
    # grid starts k_start slots in. Re-anchor on that first slot: feature column
    # b is then grid beat b of the grid_t0 reported below, and time_sec, the
    # downbeat phase and every consumer's `grid_t0 + beat * period` agree.
    # Without this every section read one beat early. k_start is 0 whenever
    # grid_t0 >= 0, so that case is untouched.
    grid_t0 += k_start * period
    F, rms_per_beat = beat_sync_features(y, sr, grid_times)
    n_beats = F.shape[1]

    phrase_beats = int(phrase_bars) * BEATS_PER_BAR
    if n_beats < 2 * phrase_beats:
        raise ValueError("track is only %d beats, shorter than two %d-bar "
                         "phrases; nothing to segment" % (n_beats, phrase_bars))

    k = int(n_segments) if n_segments else choose_segment_count(n_beats, phrase_beats)
    k = max(2, min(k, n_beats - 1))
    raw = librosa.segment.agglomerative(F, k)
    phase = downbeat_phase(period, grid_t0, downbeat_sec)
    bounds = snap_boundaries(raw, phrase_beats, n_beats, phase=phase)

    track_rms = float(np.mean(rms_per_beat))
    if track_rms <= 0:
        raise RuntimeError("analysis mix is silent; no energy to label sections by")

    sections = []
    for i, b in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else n_beats
        rel = float(np.mean(rms_per_beat[b:end])) / track_rms
        sections.append({
            "beat": int(b),
            "bar": int(b // BEATS_PER_BAR) + 1,
            "time_sec": float(round(period * b + grid_t0, 3)),
            "duration_beats": int(end - b),
            "rel_energy": float(round(rel, 3)),
            "level": label_for(rel, i),
        })

    # Structural names need the WHOLE list — a build is only a build because of
    # the section after it — so they are assigned in a second pass, not inline.
    for sec, name in zip(sections, arrangement_labels([s["rel_energy"]
                                                       for s in sections])):
        sec["label"] = name

    return {
        "bpm": float(round(60.0 / period, 3)),
        "period": float(round(period, 6)),
        "grid_t0": float(round(grid_t0, 4)),
        "duration_sec": float(round(duration, 2)),
        "beat_count": int(n_beats),
        "phrase_bars": int(phrase_bars),
        "segments_requested": int(k),
        "mix_stems": list(mix_names),
        "sections": sections,
    }


def detect_sections(stems_dir, phrase_bars=PHRASE_BARS, n_segments=None,
                    period=None, grid_t0=None, sr=ANALYSIS_SR, downbeat_sec=None):
    """Section boundaries for a stems folder, as a list of dicts.

    Each dict carries `beat` (grid beats from the sample start), `bar` (counted
    from the downbeat), `time_sec` (source-file seconds), `duration_beats`,
    `rel_energy` and `label`. Boundaries are strictly increasing and every one
    sits `phrase_bars` multiples after the downbeat. See `analyze_sections` for the grid metadata.
    """
    return analyze_sections(stems_dir, phrase_bars=phrase_bars,
                            n_segments=n_segments, period=period,
                            grid_t0=grid_t0, sr=sr,
                            downbeat_sec=downbeat_sec)["sections"]


def section_energy(stems_dir, sections, sr=ANALYSIS_SR, exclude=("drums",)):
    """Per-stem RMS in each section. Returns {stem: [level, ...]}, 0..1.

    Each stem is normalised against ITS OWN loudest section, not against the
    mix. The question this answers is "is this stem playing here, and how hard"
    — arrangement, not balance. Normalising across stems instead would just
    restate the mix: a shaker would read near-silent everywhere and tell you
    nothing about whether it enters at the drop.

    `drums` is excluded by default for the same reason stems2live mutes it: it
    is the composite of the drums_* parts, so including it double-counts the
    kit and flattens exactly the contrast this is measuring.

    Analysis only — no bridge, no Live. The caller decides what to do with it.
    """
    if librosa is None:
        raise RuntimeError("librosa is required for section energy")
    paths = {n: p for n, p in stem_paths(stems_dir).items() if n not in exclude}
    if not paths or not sections:
        return {}

    bounds = [float(s["time_sec"]) for s in sections]
    out = {}
    for name in sorted(paths):
        y, _sr = librosa.load(paths[name], sr=sr, mono=True)
        dur = len(y) / float(sr)
        levels = []
        for i, t0 in enumerate(bounds):
            t1 = bounds[i + 1] if i + 1 < len(bounds) else dur
            seg = y[int(t0 * sr):int(t1 * sr)]
            levels.append(float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0)
        peak = max(levels)
        out[name] = [(v / peak if peak > 0 else 0.0) for v in levels]
    return out


# A transition gesture is a deviation from the outgoing section's OWN typical
# bar, in either direction. Both thresholds were set by looking at the reference
# track, so they are calibrated on one piece of music — see detect_transitions.
FILL_RATIO = 1.5      # last bar this many times busier than the section median
DROPOUT_RATIO = 2.0   # ...or this many times sparser


def detect_transitions(stems_dir, sections, period, sr=ANALYSIS_SR,
                       fill_ratio=FILL_RATIO, dropout_ratio=DROPOUT_RATIO):
    """Percussion gestures in the bar before each boundary. Returns a list of
    {"bar", "beat", "kind", "ratio", "into"} with `kind` in {"fill", "dropout"}.

    A fill is more hits than usual right before the change. That was the whole
    hypothesis, and on the reference track it fired on ONE boundary out of
    eight. What the measurement actually found is that the opposite gesture is
    commoner here: the bar before Main 1 carries 3 hits against a section median
    of 44.5 — everything drops out for a bar — and Quiet and Main 2 do the same
    thing less dramatically. So both directions are reported, because both are
    the same musical move (mark the seam by breaking the pattern) and only one
    of them is called a fill.

    PERCUSSION ONLY. A fill is a drum gesture; pads swelling into a drop is a
    different thing and would drown the signal. `drums` is excluded with the
    other composites for the usual double-count reason.

    Density is measured against the OUTGOING section's own median bar, not a
    global average, so a busy section and a sparse one are judged on their own
    terms.

    CALIBRATED ON ONE TRACK, and the thresholds were chosen after seeing its
    numbers, which is the textbook way to overfit. It marked 4 of 8 boundaries
    there. Treat a miss as "no gesture found", not as "no gesture".
    """
    if librosa is None:
        raise RuntimeError("librosa is required for transition detection")
    paths = {n: p for n, p in stem_paths(stems_dir).items()
             if n.startswith("drums") and n != "drums"}
    if not paths or len(sections) < 2:
        return []

    onsets = []
    for name in sorted(paths):
        y, _sr = librosa.load(paths[name], sr=sr, mono=True)
        onsets.append(librosa.onset.onset_detect(y=y, sr=sr, units="time",
                                                 backtrack=False))

    def hits(t0, t1):
        return sum(int(((o >= t0) & (o < t1)).sum()) for o in onsets)

    bar_sec = BEATS_PER_BAR * float(period)
    out = []
    for i in range(1, len(sections)):
        a = float(sections[i - 1]["time_sec"])
        b = float(sections[i]["time_sec"])
        n_bars = max(1, int(round((b - a) / bar_sec)))
        counts = [hits(a + k * bar_sec, a + (k + 1) * bar_sec)
                  for k in range(n_bars)]
        # 0.5 rather than 0: a section with a silent median bar would otherwise
        # divide by zero, and "busier than silence" is not a fill.
        median = float(np.median(counts)) or 0.5
        last = counts[-1]

        if last / median >= fill_ratio:
            kind, ratio = "fill", last / median
        elif median / max(last, 0.5) >= dropout_ratio:
            kind, ratio = "dropout", median / max(last, 0.5)
        else:
            continue
        out.append({
            "bar": int(sections[i]["bar"]) - BEATS_PER_BAR // BEATS_PER_BAR,
            "beat": float(sections[i]["beat"]) - BEATS_PER_BAR,
            "kind": kind,
            "ratio": float(round(ratio, 2)),
            "into": sections[i].get("label", "?"),
        })
    return out


def transition_notes(transitions, fill_pitch=88, dropout_pitch=86,
                     length_beats=BEATS_PER_BAR):
    """Transitions as MIDI notes, above the macro curve. Two pitches, so the
    two gestures are distinguishable at a glance rather than by velocity alone.

    Velocity carries how pronounced the gesture is, capped at ratio 4 — past
    that the difference between "very sparse" and "silent" is not musically
    interesting and would flatten everything else against the ceiling.
    """
    notes = []
    for t in transitions:
        pitch = fill_pitch if t["kind"] == "fill" else dropout_pitch
        vel = int(max(1, min(127, round(min(t["ratio"], 4.0) / 4.0 * 127))))
        notes.append({"pitch": int(pitch),
                      "start_time": float(max(0.0, t["beat"])),
                      "duration": float(length_beats),
                      "velocity": vel})
    return notes


def macro_energy_notes(energy, sections, pitch=84, tail_beats=64):
    """The MACRO energy curve: one note per section, velocity = overall level.

    A single line, deliberately. The per-stem matrix answers "what is playing";
    this answers "how hard is the track working here", which is the macro
    arrangement question — where the tension builds and releases across the
    whole song rather than which element carries it.

    Level is the mean across stems, each already normalised against itself, so
    a section is loud here because MANY elements are near their own peak — not
    because one loud stem dominates the sum. Sat well above the per-stem rows
    (pitch 84 against 72 and down) so it reads as a separate lane.
    """
    if not sections:
        return []
    names = sorted(energy)
    notes = []
    for i, sec in enumerate(sections):
        vals = [energy[n][i] for n in names if i < len(energy[n])]
        level = float(np.mean(vals)) if vals else 0.0
        b0 = float(sec["beat"])
        b1 = (float(sections[i + 1]["beat"]) if i + 1 < len(sections)
              else b0 + tail_beats)
        notes.append({"pitch": int(pitch), "start_time": b0,
                      "duration": float(b1 - b0),
                      "velocity": int(max(1, min(127, round(level * 127))))})
    return notes


def energy_notes(energy, sections, order=None, top_pitch=72, floor=0.10,
                 tail_beats=64):
    """The energy matrix as MIDI notes. Returns [{pitch, start_time, duration, velocity}].

    One ROW PER STEM (pitch descending from `top_pitch`, so the first stem sits
    at the top of the piano roll and the rows read in the same order as the
    tracks), one NOTE PER SECTION the stem plays in, and VELOCITY carrying the
    level — which is what Live colours notes by, so the arrangement's shape is
    legible at a glance.

    Times are in GRID BEATS from the sample start; the caller adds whatever
    offset it placed the clips at. Cells below `floor` are omitted rather than
    written at velocity 1: an absent stem should be a gap you can see, not a
    faint note you have to squint at.
    """
    names = [n for n in (order or sorted(energy)) if n in energy]
    notes = []
    for row, name in enumerate(names):
        pitch = top_pitch - row
        for i, level in enumerate(energy[name]):
            if level < floor:
                continue
            b0 = float(sections[i]["beat"])
            b1 = (float(sections[i + 1]["beat"]) if i + 1 < len(sections)
                  else b0 + tail_beats)
            notes.append({"pitch": int(pitch), "start_time": b0,
                          "duration": float(b1 - b0),
                          "velocity": int(max(1, min(127, round(level * 127))))})
    return notes, names


def sections_to_locators(sections, period, start_offset_beats, name_prefix=""):
    """Convert sections to `create_cue_point` arguments. Returns [{time, name}].

    `time` is Live song time in BEATS, which is what the bridge's
    `create_cue_point(time, name)` expects.

    THE OFFSET IS NOT OPTIONAL and is not defaulted. Sections are measured from
    the start of the SAMPLE; clips are placed at arrangement beat
    `start_offset_beats`, so the sample's beat 0 is at that arrangement beat, not
    at 0. Pass the same `start_offset_beats` that stems2live gave every clip.
    Forget it and every locator sits early by up to a full bar — up to 31 beats
    when `--phrase 8` is in play, which is a quarter of a minute at 125 BPM.

    `period` must be the period the SET tempo was taken from. Live song time is
    beats, so seconds are divided by that period; using a different one makes the
    error grow with time instead of staying constant.
    """
    if period <= 0:
        raise ValueError("beat period must be positive, got %r" % period)
    offset = float(start_offset_beats)
    return [{
        "time": float(round(offset + s["time_sec"] / period, 4)),
        "name": "%s%s" % (name_prefix, s["label"]),
    } for s in sections]


def cli_grid(stems_dir, tempo=None, grid_t0=None, downbeat_sec=None,
             out=sys.stderr):
    """(period, grid_t0, downbeat_sec) for the standalone CLIs.

    With no grid on the command line, reuse the one beat_aligner cached in the
    stems folder (alignment.json, written by stems2live / yt2stems), so a
    standalone run snaps to the same bar lines as the arrangement. With no
    cache — or a grid given without --downbeat — say once that bar 1 is
    assumed at the first grid beat, since that is exactly the case where
    sections can sit 1-3 beats off the bar line.
    """
    period = (60.0 / tempo) if tempo else None
    if period is None and grid_t0 is None:
        cached = cached_grid(stems_dir)
        if cached is not None:
            print("grid: using %s from %s (%.3f BPM, downbeat %.3fs)"
                  % ("alignment.json", os.path.basename(os.path.normpath(stems_dir)),
                     60.0 / cached["period"], cached["downbeat_sec"]), file=out)
            return (float(cached["period"]), float(cached["grid_t0"]),
                    downbeat_sec if downbeat_sec is not None
                    else float(cached["downbeat_sec"]))
    if downbeat_sec is None:
        print("grid: no alignment.json with these stems and no --downbeat — bar 1 "
              "is assumed at the first grid beat. Run stems2live (or pass "
              "--downbeat) to snap to the detected downbeat.", file=out)
    return period, grid_t0, downbeat_sec


def _mmss(t):
    return "%d:%04.1f" % (int(t) // 60, t % 60)


def main():
    ap = argparse.ArgumentParser(
        description="Detect song sections and emit Ableton locator positions")
    ap.add_argument("stems_dir", help="Path to a yt2stems folder (or one audio file)")
    ap.add_argument("--phrase-bars", type=int, default=PHRASE_BARS,
                    help="snap boundaries to this phrase length in bars "
                         "(default: %d)" % PHRASE_BARS)
    ap.add_argument("--segments", type=int, default=None,
                    help="raw segment count before snapping; default scales with "
                         "track length (see choose_segment_count)")
    ap.add_argument("--tempo", type=float, default=None,
                    help="override the detected tempo (BPM); needs --grid-t0")
    ap.add_argument("--grid-t0", type=float, default=None,
                    help="grid anchor in seconds, from beat_aligner's grid_t0 "
                         "(default: the stems folder's alignment.json if any)")
    ap.add_argument("--downbeat", type=float, default=None,
                    help="first downbeat in seconds, from beat_aligner's "
                         "downbeat_sec; boundaries snap to phrases counted from "
                         "it (default: alignment.json, else the first grid beat)")
    ap.add_argument("--start-offset", type=float, default=0.0,
                    help="stems2live's start_offset_beats, for the locator times. "
                         "Leaving this at 0 is correct ONLY for an unaligned run "
                         "(default: 0)")
    ap.add_argument("--json", action="store_true", help="Output JSON results")
    args = ap.parse_args()

    period, grid_t0, downbeat = cli_grid(args.stems_dir, args.tempo,
                                         args.grid_t0, args.downbeat)
    res = analyze_sections(
        args.stems_dir, phrase_bars=args.phrase_bars, n_segments=args.segments,
        period=period, grid_t0=grid_t0, downbeat_sec=downbeat)
    locators = sections_to_locators(res["sections"], res["period"],
                                    args.start_offset)

    if args.json:
        res["start_offset_beats"] = float(args.start_offset)
        res["locators"] = locators
        print(json.dumps(res, indent=2))
        return

    print("Mix Stems:           %s" % "+".join(res["mix_stems"]))
    print("Tempo:               %.3f BPM (period %.6f s, t0 %.4f s)"
          % (res["bpm"], res["period"], res["grid_t0"]))
    print("Duration:            %s  (%d beats, %d bars)"
          % (_mmss(res["duration_sec"]), res["beat_count"],
             res["beat_count"] // BEATS_PER_BAR))
    print("Segmentation:        %d raw -> %d sections after %d-bar snap"
          % (res["segments_requested"], len(res["sections"]), res["phrase_bars"]))
    print("Locator Offset:      +%.4f beats (arrangement time = offset + sec/period)"
          % args.start_offset)
    print()
    print("  bar      time     locator     rel   label")
    for s, loc in zip(res["sections"], locators):
        print("  %-6d %8s  %9.3f  %6.2f   %s"
              % (s["bar"], _mmss(s["time_sec"]), loc["time"],
                 s["rel_energy"], s["label"]))
    print()
    print("Boundaries are MEASURED (spectral change: MFCC + chroma + RMS, "
          "agglomerative\nclustering, snapped to 8-bar phrases). Labels are "
          "RULES over energy and order —\nno model recognises a 'drop'; it is "
          "loud-after-quiet. Expect odd names on tracks\nthat open loud or drop "
          "without a preceding break.")


if __name__ == "__main__":
    main()

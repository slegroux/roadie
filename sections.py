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


# --- stem-activity method ----------------------------------------------------
#
# WHY A SECOND METHOD. The spectral method above reads the SUMMED mix. In
# melodic techno a section change is mostly a change in who is playing: the kick
# leaves for a breakdown, the bass and hats come back as a build, the kick
# returns as the drop. A sum blurs exactly that. When the separated stems exist
# their per-bar levels carry it directly, and they are free: the stems are
# already on disk and reading them is a few seconds, with no model to run.
#
# Checked by ear on ONE track so far (Gerlach "Off", 125 BPM): the quiet section
# starts at Live bar 69 and the drop at Live bar 101; the spectral method placed
# them 4 bars late. A second track (Sainte Vie "Cenote") was looked at without
# ground truth. Treat the thresholds below as starting values, not tuned constants.

STEM_FLOOR_DB = 30.0      # activity 0 = this far under a stem's own loud level
STEM_SILENT_DB = -50.0    # a stem whose loud level is under this is treated as absent
                          # (an empty ride, bleed-only vocals): relative activity
                          # would otherwise read its noise as "playing" in every bar
STEM_ON = 0.5             # activity above which a stem counts as playing...
STEM_OFF = 0.3            # ...and below which it has stopped (hysteresis)
STEM_MIN_RUN = 4          # bars a presence state must hold to count
STEM_SNAP_BARS = 1        # snap a boundary to the phrase grid only this close
STEM_NOVELTY_BARS = 4     # window either side when scoring the phrase offset

# Stem files that make up each role, in preference order within a role.
STEM_ROLES = {
    "kick": ("drums_kick",),
    "bass": ("bass",),
    "hats": ("drums_hh", "drums_ride"),
    "synths": ("synths",),
}
# Presence of these drives the boundaries; vocals are reported, not cut on —
# they chop in and out too often to mark a section.
STEM_CUT_ROLES = ("kick", "bass", "hats", "synths")


def _bar_db(path, edge_times):
    """Level of each bar of one stem, in dB. `edge_times` has n_bars + 1 entries.

    Streamed in blocks so a long set does not have to fit in memory: each block's
    squared samples are added to the bar they fall in.
    """
    import soundfile as sf
    with sf.SoundFile(path) as f:
        idx = np.clip(np.round(np.asarray(edge_times) * f.samplerate).astype(np.int64),
                      0, f.frames)
        n_bars = len(idx) - 1
        sums = np.zeros(n_bars)
        off = 0
        for block in f.blocks(blocksize=1 << 20, dtype="float32", always_2d=True):
            x = block.mean(axis=1).astype(np.float64)
            bar = np.searchsorted(idx, np.arange(off, off + len(x)), side="right") - 1
            ok = (bar >= 0) & (bar < n_bars)
            sums += np.bincount(bar[ok], weights=x[ok] ** 2, minlength=n_bars)
            off += len(x)
    n = np.maximum(np.diff(idx), 1)
    return 10.0 * np.log10(sums / n + 1e-12)


def _activity(db):
    """0..1: 1 at the stem's own loud level, 0 at STEM_FLOOR_DB under it.

    Returns None for a stem that is effectively silent: relative activity has no
    meaning when the stem's loud level is itself near silence, and would read the
    noise floor as "playing" in every bar. The loud level is taken over the bars
    above STEM_SILENT_DB, so a stem that plays in only a few bars still has one.
    """
    loud = db[db > STEM_SILENT_DB]
    if not len(loud):
        return None
    top = float(np.percentile(loud, 95))
    return np.clip((db - top + STEM_FLOOR_DB) / STEM_FLOOR_DB, 0.0, 1.0)


def _hysteresis(x, on=STEM_ON, off=STEM_OFF):
    state, out = bool(x[0] > on), np.zeros(len(x), bool)
    for i, v in enumerate(x):
        if state and v < off:
            state = False
        elif not state and v > on:
            state = True
        out[i] = state
    return out


def _despeckle(b, n=STEM_MIN_RUN):
    """Absorb runs shorter than `n` bars into a neighbour: a short run at the start
    joins the run after it, any other short run joins the one before it."""
    b = b.copy()
    i = 0
    while i < len(b):
        j = i
        while j < len(b) and b[j] == b[i]:
            j += 1
        if j - i < n and j < len(b) and i == 0:
            b[i:j] = b[j]
        elif j - i < n and i > 0:
            b[i:j] = b[i - 1]
        i = j
    return b


def _novelty(F, w=STEM_NOVELTY_BARS):
    """How much the stem mix changes at each bar: |mean(next w) - mean(prev w)|."""
    out = np.zeros(len(F))
    for i in range(1, len(F)):
        out[i] = np.linalg.norm(F[i:i + w].mean(axis=0) - F[max(0, i - w):i].mean(axis=0))
    return out


def estimate_phrase_offset(novelty, phrase_bars=PHRASE_BARS):
    """Which bar (0..phrase_bars-1) of the 8-bar cycle the changes land on.

    Phrase lines are NOT assumed to start at bar 0 of the detected grid: on the
    test track the structure sat 4 bars off the downbeat's own 8-bar cycle. Every
    bar votes for its position in the cycle with its novelty, a neighbour votes
    half. Ties go to 0 (the downbeat's own grid) unless another offset clearly
    wins, so a track whose structure does start on the downbeat is unchanged.
    """
    score = np.zeros(phrase_bars)
    for i, v in enumerate(novelty):
        r = i % phrase_bars
        score[r] += v
        score[(r + 1) % phrase_bars] += 0.5 * v
        score[(r - 1) % phrase_bars] += 0.5 * v
    best = int(np.argmax(score))
    if score[0] >= 0.9 * score[best]:
        best = 0
    return best


def _snap_bars(cuts, offset, phrase_bars, n_bars):
    """Snap bar indices to the phrase grid when close; keep them when not."""
    out = []
    for c in sorted(set(cuts)):
        line = offset + phrase_bars * int(round((c - offset) / float(phrase_bars)))
        cand = line if (abs(c - line) <= STEM_SNAP_BARS and 0 < line < n_bars) else c
        if (cand >= STEM_MIN_RUN and n_bars - cand >= STEM_MIN_RUN
                and (not out or cand - out[-1] >= STEM_MIN_RUN)):
            out.append(cand)
    return out


def stem_labels(rows):
    """Name sections from which stems play. Rules, like `arrangement_labels`.

      - first section Intro; last section Outro, unless the kick and bass are both
        still playing in it (a track that ends on the full groove has no outro)
      - no kick: Breakdown, except that a kickless section straight before the
        kick returns, once hats are back, is a Build (never the section right
        after the Intro: a kickless start is a Breakdown, not a build to nothing)
      - kick back after a kickless stretch: Drop if the bass is with it, else Build
      - kick throughout, bass enters: Drop; bass already there: Main
      - kick without bass ahead of a section that has it: Build
    A muffled breakdown kick (well under the stem's loud level) counts as no kick.
    """
    n = len(rows)
    if n == 0:
        return []
    if n == 1:
        return ["Intro"]
    out = [None] * n
    out[0] = "Intro"
    if not (rows[-1]["kick"] and rows[-1]["bass"]):
        out[-1] = "Outro"
    nothing = {"kick": False, "bass": False, "hats": False, "synths": False}
    for i in range(1, n):
        if out[i] is not None:
            continue
        r, prev = rows[i], rows[i - 1]
        nxt = rows[i + 1] if i + 1 < n else nothing
        if not r["kick"]:
            out[i] = "Build" if (nxt["kick"] and r["hats"] and i > 1) else "Breakdown"
        elif not prev["kick"]:
            out[i] = "Drop" if r["bass"] else "Build"
        elif r["bass"]:
            out[i] = "Main" if prev["bass"] else "Drop"
        else:
            out[i] = "Build" if nxt["bass"] else "Main"
    counts = {}
    for name in out:
        counts[name] = counts.get(name, 0) + 1
    seen = {}
    for i, name in enumerate(out):
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            out[i] = "%s %d" % (name, seen[name])
    return out


def analyze_sections_by_stems(stems_dir, period, grid_t0, downbeat_sec=None,
                              phrase_bars=PHRASE_BARS):
    """Sections from per-bar stem activity. Same result shape as `analyze_sections`.

    Returns None when the folder lacks what this needs (a kick source and a bass
    stem), so `method="auto"` can fall back to the spectral method.
    """
    import soundfile as sf
    paths = stem_paths(stems_dir) if os.path.isdir(stems_dir) else {}
    parts = [n for n in paths if n.startswith("drums_")]
    # The kick comes from drums_kick; the whole-kit composite stands in for it only
    # when there are no drum parts at all. With parts but no kick the composite would
    # read any hats or snare as a kick, so that folder is left to the spectral method.
    if "drums_kick" in paths:
        kick_path = paths["drums_kick"]
    elif "drums" in paths and not parts:
        kick_path = paths["drums"]
    else:
        return None
    if "bass" not in paths:
        return None
    period, grid_t0 = float(period), float(grid_t0)
    if grid_t0 < 0:                       # same re-anchor as build_beat_grid_times
        grid_t0 += int(np.ceil(-grid_t0 / period)) * period
    used = [n for n in paths if not (n == "drums" and parts)]
    lengths = [sf.info(paths[n]) for n in used]
    duration = min(i.frames / float(i.samplerate) for i in lengths)
    n_beats = int(np.floor((duration - grid_t0) / period))
    phase = downbeat_phase(period, grid_t0, downbeat_sec)
    n_bars = (n_beats - phase) // BEATS_PER_BAR
    if n_bars < 2 * phrase_bars:
        raise ValueError("track is only %d bars, shorter than two %d-bar phrases; "
                         "nothing to segment" % (n_bars, phrase_bars))
    edges = grid_t0 + period * (phase + BEATS_PER_BAR * np.arange(n_bars + 1))

    db = {n: _bar_db(paths[n], edges) for n in used}
    act = {n: a for n, a in ((n, _activity(v)) for n, v in db.items()) if a is not None}
    if kick_path == paths.get("drums"):   # composite-only folder: it stands in for the kick
        if "drums" not in act:
            return None
        act["drums_kick"] = act["drums"]
    if "drums_kick" not in act or "bass" not in act:
        return None                       # the kick or the bass stem is silent

    def role(name):
        have = [act[n] for n in STEM_ROLES[name] if n in act]
        return np.max(np.stack(have), axis=0) if have else np.zeros(n_bars)

    A = {r: role(r) for r in STEM_CUT_ROLES}
    present = {r: _despeckle(_hysteresis(A[r])) for r in STEM_CUT_ROLES}
    vocals = _despeckle(_hysteresis(act["vocals"])) if "vocals" in act else np.zeros(n_bars, bool)

    state = np.stack([present[r] for r in STEM_CUT_ROLES], axis=1)
    change = [i for i in range(1, n_bars) if (state[i] != state[i - 1]).any()]
    nov = _novelty(np.stack([A[r] for r in STEM_CUT_ROLES], axis=1))
    offset = estimate_phrase_offset(nov, phrase_bars)
    cuts = _snap_bars(change, offset, phrase_bars, n_bars)
    bounds = [0] + cuts + [n_bars]

    power = sum(10.0 ** (db[n] / 10.0) for n in used if n in act)
    rms = np.sqrt(power)
    mean_rms = float(np.mean(rms))
    if mean_rms < 1e-5:
        raise RuntimeError("stems are silent; no energy to label sections by")

    rows, sections = [], []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        mean_state = state[a:b].mean(axis=0) > 0.5
        flags = dict(zip(STEM_CUT_ROLES, (bool(x) for x in mean_state)))
        rows.append(flags)
        beat = phase + BEATS_PER_BAR * a
        rel = float(np.mean(rms[a:b])) / mean_rms
        names = [r for r in STEM_CUT_ROLES if flags[r]]
        if vocals[a:b].mean() > 0.5:
            names.append("vocals")
        sections.append({
            "beat": int(beat),
            "bar": int(beat // BEATS_PER_BAR) + 1,
            "time_sec": float(round(period * beat + grid_t0, 3)),
            "duration_beats": int(BEATS_PER_BAR * (b - a)) if b < n_bars
            else int(n_beats - beat),
            "rel_energy": float(round(rel, 3)),
            "level": label_for(rel, i),
            "stems": names,
        })
    for sec, name in zip(sections, stem_labels(rows)):
        sec["label"] = name

    return {
        "bpm": float(round(60.0 / period, 3)),
        "period": float(round(period, 6)),
        "grid_t0": float(round(grid_t0, 4)),
        "duration_sec": float(round(duration, 2)),
        "beat_count": int(n_beats),
        "phrase_bars": int(phrase_bars),
        "segments_requested": len(sections),
        "mix_stems": [n for n in used if n in act],
        "sections": sections,
        "method": "stems",
        "phrase_offset_bars": int(offset),
    }


def analyze_sections(stems_dir, phrase_bars=PHRASE_BARS, n_segments=None,
                     period=None, grid_t0=None, sr=ANALYSIS_SR,
                     downbeat_sec=None, method="auto"):
    """Full section analysis. Returns a dict with `sections` plus the grid it used.

    `method` picks the detector: "stems" reads per-bar stem activity (needs a
    kick source and a bass stem), "spectral" clusters the summed mix, "auto" (the
    default) uses stems when the folder has them and spectral otherwise.

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

    if method not in ("auto", "stems", "spectral"):
        raise ValueError("method must be auto, stems or spectral, got %r" % (method,))
    if method in ("auto", "stems"):
        try:
            res = analyze_sections_by_stems(stems_dir, period, grid_t0,
                                            downbeat_sec=downbeat_sec,
                                            phrase_bars=phrase_bars)
        except ValueError:                # too short for the stems method
            if method == "stems":
                raise
            res = None                    # auto: the spectral method states its own limit
        if res is not None:
            if n_segments:
                import warnings
                warnings.warn("--segments applies only to the spectral method; "
                              "ignored by the stems method", stacklevel=2)
            return res
        if method == "stems":
            raise ValueError("the stems method needs a kick source (drums_kick, or "
                             "drums when there are no drum parts) and a bass stem "
                             "that are not silent in %s" % stems_dir)
    if librosa is None:
        raise RuntimeError("librosa is required for the spectral section method")

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
                    period=None, grid_t0=None, sr=ANALYSIS_SR, downbeat_sec=None,
                    method="auto"):
    """Section boundaries for a stems folder, as a list of dicts.

    Each dict carries `beat` (grid beats from the sample start), `bar` (counted
    from the downbeat), `time_sec` (source-file seconds), `duration_beats`,
    `rel_energy` and `label`. Boundaries are strictly increasing. With the spectral
    method every one sits `phrase_bars` multiples after the downbeat; the stems
    method snaps to an estimated phrase cycle only when within a bar of it and
    allows sections as short as 4 bars. See `analyze_sections` for the grid metadata.
    """
    return analyze_sections(stems_dir, phrase_bars=phrase_bars,
                            n_segments=n_segments, period=period,
                            grid_t0=grid_t0, sr=sr,
                            downbeat_sec=downbeat_sec, method=method)["sections"]


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


def live_bars(res, downbeat_sec, start_bar):
    """Live arrangement bar of each section, or None when no downbeat is known.

    The `bar` field counts from the start of the audio file. Clips go into Live
    on a bar line with the lead-in trimmed (`stems2live.bar_aligned_placement`),
    so the file's own beat grid sits at `start - trim / period` in the arrangement
    and a section at `time_sec` lands at Live bar 1 + that / 4. This is the number
    to look for in Live's ruler.
    """
    if downbeat_sec is None:
        return None
    from stems2live import bar_aligned_placement
    period = res["period"]
    start, trim, _ = bar_aligned_placement(
        {"period": period, "grid_t0": res["grid_t0"], "downbeat_sec": downbeat_sec},
        start_bar, res["bpm"])
    shift = start - trim / period
    return [round(1.0 + (shift + sec["time_sec"] / period) / BEATS_PER_BAR, 2)
            for sec in res["sections"]]


def _mmss(t):
    return "%d:%04.1f" % (int(t) // 60, t % 60)


def write_sections_json(stems_dir, res, start_offset, locators):
    """Save the analysis as <stems_dir>/sections.json so it outlives the run.

    Only for a stems DIRECTORY (the other accepted input is one audio file, which
    has nowhere of its own to put it). Best effort: an unwritable folder costs the
    file, not the analysis. Returns the path written, or None."""
    if not os.path.isdir(stems_dir):
        return None
    path = os.path.join(stems_dir, "sections.json")
    out = dict(res, start_offset_beats=float(start_offset), locators=locators)
    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    except OSError as e:
        print("sections: could not write %s: %s" % (path, e), file=sys.stderr)
        return None
    return path


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
    ap.add_argument("--start-bar", type=int, default=9,
                    help="Live bar the clips start on, as `roadie load` / `roadie als` "
                         "place them (default 9); sets the Live bar column")
    ap.add_argument("--method", choices=("auto", "stems", "spectral"), default="auto",
                    help="stems: per-bar stem activity (needs kick + bass stems); "
                         "spectral: cluster the summed mix; auto (default): stems "
                         "when the folder has them")
    ap.add_argument("--no-write", action="store_true",
                    help="do not save sections.json into the stems folder")
    args = ap.parse_args()

    period, grid_t0, downbeat = cli_grid(args.stems_dir, args.tempo,
                                         args.grid_t0, args.downbeat)
    res = analyze_sections(
        args.stems_dir, phrase_bars=args.phrase_bars, n_segments=args.segments,
        period=period, grid_t0=grid_t0, downbeat_sec=downbeat, method=args.method)
    locators = sections_to_locators(res["sections"], res["period"],
                                    args.start_offset)
    live = live_bars(res, downbeat, args.start_bar)
    for sec, lb in zip(res["sections"], live or []):
        sec["live_bar"] = lb
    saved = None if args.no_write else write_sections_json(
        args.stems_dir, res, args.start_offset, locators)

    if args.json:
        res["start_offset_beats"] = float(args.start_offset)
        res["locators"] = locators
        print(json.dumps(res, indent=2))
        return

    print("Method:              %s" % ("per-bar stem activity" if res.get("method") == "stems"
                                         else "spectral clustering of the summed mix"))
    if res.get("method") == "stems":
        print("Phrase offset:       %d bar(s) into the %d-bar cycle"
              % (res["phrase_offset_bars"], res["phrase_bars"]))
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
    stems_view = res.get("method") == "stems"
    print("  Live   file     time     locator     rel   label"
          + ("          playing" if stems_view else ""))
    for i, (s, loc) in enumerate(zip(res["sections"], locators)):
        lb = ("%g" % live[i]) if live else "-"
        print("  %-6s %-5d %8s  %9.3f  %6.2f   %-13s%s"
              % (lb, s["bar"], _mmss(s["time_sec"]), loc["time"], s["rel_energy"],
                 s["label"], " ".join(s["stems"]) if stems_view else ""))
    if live:
        print("\n  'Live' is the bar in Live's ruler with clips starting at bar %d "
              "(--start-bar); 'file' counts from the start of the audio."
              % args.start_bar)
    print()
    if saved:
        print("Saved:               %s" % saved)
        print()
    if stems_view:
        print("Boundaries are MEASURED (which stems play, bar by bar; a muffled kick "
              "counts as\nno kick), snapped to the 8-bar cycle when within a bar of it. "
              "Labels are RULES over\nwhich stems are present — no model recognises a "
              "'drop'. Check the first one or two\nboundaries by ear.")
    else:
        print("Boundaries are MEASURED (spectral change: MFCC + chroma + RMS, "
              "agglomerative\nclustering, snapped to 8-bar phrases). Labels are "
              "RULES over energy and order —\nno model recognises a 'drop'; it is "
              "loud-after-quiet. Expect odd names on tracks\nthat open loud or drop "
              "without a preceding break.")


if __name__ == "__main__":
    main()

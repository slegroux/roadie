#!/usr/bin/env python3
"""The micro layer of a study: what the bass, the synths, the vocal and the drums play inside each section, at the
scale of a bar. `roadie micro <study | stems_dir> [--json] [--no-write]` writes `micro.json` beside `drums.json`.

Macro is the structure (sections, entries, groove blocks: `roadie study report`); micro is each instrument's pattern
at a small scale. The fields follow what the user's notes measure by hand ("16th steps 2, 5, 8, 11, 15", "2 bars per
chord", "pedal", "stabby vs continuous", "accent on the third 16th").

Grid: the tempo and downbeat of alignment.json, with bar 0 the downbeat moved back whole bars to the first bar line in
the file, exactly as drums.json counts (`roadie_analysis/drums.py`); a section starts at the bar its sections.json time
falls on. Every 16-step pattern, chord per bar and phrase start depends on that bar line, so each section carries
`reliable: false` when the downbeat is chance-level, the grid has a warp map (the tempo drifts, bars are placed at a
constant tempo here) or there is no alignment.json.

Choices:

- A stem plays in a bar by sections.py's rule (level within 15 dB of its own loud level, with hysteresis), so
  "playing" means what it means in sections.json. Onsets (librosa's spectral flux) count only in bars where the stem
  plays and when they rise within ONSET_FLOOR_DB of the stem's loud frame level; an onset within GRID_TOL of a 16th
  line is on that 16th, else off-grid (counted, never quantised).
- A one-bar rhythm is the set of 16ths with an onset in that bar (`o..o o... o..o o...`); the section's rhythm is the
  most common one over the bars with an onset, with its share of those bars.
- Bass pitch: pyin per onset (the median f0 of the voiced frames until the next onset; half-semitone bins from A0, so
  the notes are bin centres and the run is about 6 times faster than pyin's default 0.1), named as a scale degree of the
  study's key (study.json). Duty is the sounding time per playing bar (frames within DUTY_DB of the loud frame level):
  under STABBY it is "stabby", from CONTINUOUS "continuous", between "between".
- Synth chords: chroma (CQT) summed over the bar, matched to major, minor, sus2, sus4 and root+5 templates by cosine;
  near-ties go to the root of the bass's first note in that bar (sus2 on C and sus4 on G are the same notes). Named
  relative to the key: diatonic degrees of its scale plain (`i`, `VI`, `III`, `VII` in a minor key), others with b/#;
  upper case for major, lower for minor, a sus or root+5 chord in the case of the scale's own triad (`i5`, `Vsus4`).
  A chord run lasts while the root holds; the progression is a loop of 2 to 4 roots (`cycle`) or, when none holds,
  every run in order; one loop is written on the same phase in every section of the track (`phase_loops`).
  The synths stem is leads and pads together: chords and rhythm are of the mix of both.
- Drone: a pitch class at DRONE_LEVEL of the bar's strongest in at least 90 % of the bars the synths play. A common
  tone of the chords counts too: it is "held", not necessarily one sustained note.
- Variation: the repeat period is the smallest of 1, 2, 4, 8 bars at which a bar equals the bar that many before in at
  least 75 % of the pairs (bass: rhythm and the set of pitch classes it plays; synths: chord and rhythm); `changes_at_bars`
  are the bars that do not repeat at that period (with no period, the bars that differ from the bar before).
- Drums are not measured here: the roles and patterns are drums.json's, per section. The accent position of a hat
  role needs per-hit levels, which drums.json does not record: it is null with that reason.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from roadie_analysis import drums as drummod  # noqa: E402
from roadie_analysis import drumsections  # noqa: E402
from roadie_analysis import key as keymod  # noqa: E402

KIND, VERSION = "micro", 1
SR = 22050
HOP = 256                  # 11.6 ms frames
PITCH_SR = 11025           # pyin on the bass at a lower rate: its fundamentals are under 300 Hz
STEPS = 16
GRID_TOL = 0.3             # an onset within this share of a 16th of the line is on it (37 ms at 124 bpm)
ONSET_FLOOR_DB = 30.0      # an onset quieter than this under the stem's loud frame level is bleed or a tail
DUTY_DB = 20.0             # a frame "sounds" within this of the stem's loud frame level
STABBY, CONTINUOUS = 0.5, 0.8
PEDAL_SHARE = 0.75         # the top pitch class carries at least this share of the pitched onsets: pedal
MIN_PC_SHARE = 0.05        # a pitch class under this share of the onsets is a pitch-tracking slip, not counted
DRONE_LEVEL, DRONE_BARS = 0.5, 0.9
REPEAT_SHARE = 0.75
LOOP_MIN, LOOP_MAX = 2, 4  # a chord loop is 2 to 4 chords: longer is a sequence of changes, not a loop
FOLLOW_SHARE = 0.75        # a loop with chords put in names the progression when it is followed in this share of the runs
SHUFFLES, SHUFFLE_BEAT = 200, 0.95  # ...and scores above this share of this many no-repeat shuffles of the same runs
PERIODS = (1, 2, 4, 8)
MIN_PAIRS = 3              # a repeat period read from fewer pairs of playing bars than this is not one
TIE = 0.02                 # chord scores this close are a tie, broken by the bass
OFFBEAT_8THS = (2, 6, 10, 14)  # 0-based 16ths: the "and" of each beat
AUDIO = (".flac", ".wav")
STEMS = ("bass", "synths", "vocals")
HAT_STEMS = ("hh", "ride")
BASS_FMAX = 1000.0         # the bass's onsets are read below this: above it a separated bass is mostly artefacts
FEW_ONSETS = 2.0           # a continuous bass with fewer onsets a bar than this moves in level, not in note starts

PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
SCALES = {keymod.MINOR: (0, 2, 3, 5, 7, 8, 10), keymod.MAJOR: (0, 2, 4, 5, 7, 9, 11)}
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")
TEMPLATES = (("maj", (0, 4, 7)), ("min", (0, 3, 7)), ("sus2", (0, 2, 7)), ("sus4", (0, 5, 7)), ("5", (0, 7)))

SYNTHS_NOTE = "the synths stem holds leads and pads together: rhythm and chords are of both at once"
ACCENT_MISSING = ("drums.json records no per-hit levels (a role has one median level_db): the loudest 16th of the beat "
                  "cannot be read from it")


# ---- names

def pattern_text(steps) -> str:
    """`o..o o... o..o o...` for the 0-based 16ths in `steps`, four groups of four (the form `study show` prints)."""
    s = "".join("o" if i in steps else "." for i in range(STEPS))
    return " ".join(s[i:i + 4] for i in range(0, STEPS, 4))


def note_name(midi: float) -> str:
    m = int(round(midi))
    return "%s%d" % (PC_NAMES[m % 12], m // 12 - 1)


def degree(pc: int, tonic: int) -> str:
    return DEGREES[(pc - tonic) % 12]


def roman(root: int, quality: str, tonic: int, mode: str) -> str:
    """A chord relative to the key: the scale's degrees plain, the others flat of the degree above, except that in a
    minor key the degrees of the parallel major are sharps of the minor ones (A minor: C# is #III, F# #VI, G# #VII)."""
    iv = (root - tonic) % 12
    scale = SCALES[mode]
    if iv in scale:
        n, acc = scale.index(iv), ""
    elif mode == keymod.MINOR and iv in SCALES[keymod.MAJOR]:
        n, acc = scale.index((iv - 1) % 12), "#"
    elif (iv + 1) % 12 in scale:
        n, acc = scale.index((iv + 1) % 12), "b"
    else:
        n, acc = scale.index((iv - 1) % 12), "#"
    name = ROMAN[n]
    if quality == "maj":
        return acc + name
    if quality == "min":
        return acc + name.lower()
    # sus and root+5: the case of the scale's own triad on that degree (minor third above it in the scale: lower)
    third = (scale[(n + 2) % 7] - scale[n]) % 12
    base = name.lower() if not acc and third == 3 else name
    return acc + base + quality


def chord_name(root: int, quality: str) -> str:
    return PC_NAMES[root] + {"maj": "", "min": "m"}.get(quality, quality)


def bar_ranges(bars: list[int]) -> str:
    """`1-4, 9-12, 15` for sorted bar numbers."""
    out, i = [], 0
    while i < len(bars):
        j = i
        while j + 1 < len(bars) and bars[j + 1] == bars[j] + 1:
            j += 1
        out.append(str(bars[i]) if i == j else "%d-%d" % (bars[i], bars[j]))
        i = j + 1
    return ", ".join(out)


# ---- the grid

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def grid_of(stems_dir) -> dict:
    """{tempo, bar_s, t0, downbeat_s, verdict, confidence, warp_map, reliable, why, sections, notes} for a stems folder,
    from alignment.json and sections.json (`drums.grid_from_folder`), bar 0 placed as drums.json places it. Sections are
    sections.json's, each {index, name, file_bar, start_bar (bars from t0), bars}. Raises ValueError with no tempo or
    downbeat, or no sections."""
    d = Path(stems_dir)
    g = drummod.grid_from_folder(d)
    tempo, down = g["tempo"], g["downbeat_s"]
    if not tempo or not (tempo > 0 and math.isfinite(tempo)) or down is None:
        raise ValueError("no grid in %s: make alignment.json (roadie load) and sections.json (roadie sections)" % d)
    bar_s = 60.0 / tempo * drummod.BEATS_PER_BAR
    t0 = float(down) - math.floor(float(down) / bar_s) * bar_s
    align = _read_json(d / "alignment.json")
    res = align.get("result") if isinstance(align, dict) else None
    res = res if isinstance(res, dict) else {}
    why = []
    if not res:
        why.append("no alignment.json: bar 0 is the start of the first section, not a measured downbeat")
    if res.get("downbeat_verdict") == "chance-level":
        why.append("the downbeat is chance-level (confidence %s): bar 1 may start on the wrong beat"
                   % res.get("downbeat_confidence"))
    warp = bool(res.get("warp_map"))
    if warp:
        why.append("the grid has a warp map (the tempo drifts): bars are placed at a constant tempo here")
    data = _read_json(d / "sections.json")
    secs = []
    for s in (data or {}).get("sections") or []:
        if isinstance(s.get("time_sec"), (int, float)) and isinstance(s.get("duration_beats"), (int, float)):
            secs.append({"index": len(secs), "name": s.get("label"), "file_bar": s.get("bar"),
                         "start_bar": max(0, int(round((float(s["time_sec"]) - t0) / bar_s))),
                         "bars": max(1, int(s["duration_beats"]) // drummod.BEATS_PER_BAR)})
    if not secs:
        raise ValueError("no sections in %s/sections.json: run roadie sections first" % d)
    return {"tempo": float(tempo), "bar_s": bar_s, "t0": t0, "downbeat_s": down,
            "verdict": res.get("downbeat_verdict"), "confidence": res.get("downbeat_confidence"),
            "warp_map": warp, "reliable": not why, "why": why, "sections": secs, "notes": list(g["notes"])}


# ---- audio measures

def _load(path, sr=SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def bar_activity(y, sr, t0: float, bar_s: float, n_bars: int) -> np.ndarray:
    """Which bars (0..n_bars-1 from t0) the stem plays in, by sections.py's rule (`_activity`, `_hysteresis`)."""
    import sections as sectionsmod
    c = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)])
    edges = np.clip(np.round((t0 + np.arange(n_bars + 1) * bar_s) * sr).astype(np.int64), 0, len(y))
    n = np.maximum(np.diff(edges), 1)
    db = 10.0 * np.log10((c[edges[1:]] - c[edges[:-1]]) / n + 1e-12)
    act = sectionsmod._activity(db)
    if act is None:
        return np.zeros(n_bars, dtype=bool)
    return sectionsmod._hysteresis(act)


def frame_db(y) -> np.ndarray:
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    return 20.0 * np.log10(rms + 1e-9)


def loud_level(fdb: np.ndarray) -> float | None:
    """The stem's loud frame level: the 95th percentile of its frames above -60 dB; None for a silent stem."""
    audible = fdb[fdb > -60.0]
    return float(np.percentile(audible, 95)) if len(audible) >= 10 else None


def onsets(y, sr, fdb: np.ndarray, loud: float, fmax: float | None = None) -> np.ndarray:
    """Onset times (s) of the stem that rise within ONSET_FLOOR_DB of its loud frame level: spectral flux on a log mel
    spectrogram (up to `fmax`), each peak moved back to the energy minimum before it (librosa's backtrack), which is
    where the note starts: the flux peak itself comes late on a slow attack (on a separated bass, 70 % of the peaks
    fell off the 16ths, 18 % once backtracked)."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64 if fmax else 128, fmax=fmax)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=HOP)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="frames", backtrack=True)
    keep = []
    span = max(1, int(round(0.1 * sr / HOP)))
    for f in fr:
        if f < len(fdb) and float(fdb[f:f + span].max()) >= loud - ONSET_FLOOR_DB:
            keep.append(f)
    return librosa.frames_to_time(np.asarray(keep, dtype=int), sr=sr, hop_length=HOP)


def place(times, t0: float, bar_s: float) -> list[dict]:
    """Each onset on the grid: {t, bar (0-based from t0), step (0..15), on_grid}."""
    step_s = bar_s / STEPS
    out = []
    for t in times:
        q = (float(t) - t0) / step_s
        k = int(round(q))
        out.append({"t": float(t), "bar": k // STEPS, "step": k % STEPS, "on_grid": abs(q - k) <= GRID_TOL})
    return out


def level_by_16th(fdb: np.ndarray, t0: float, bar_s: float, bars) -> list[float] | None:
    """The stem's mean level at each 16th of the bar over `bars`, in dB under the loudest 16th: where a part sounds
    in the bar even when it has no note starts (a sidechained or filtered continuous part). None with no frames."""
    ft = np.arange(len(fdb)) * HOP / SR
    pos = (ft - t0) / bar_s
    sel = np.isin(np.floor(pos).astype(int), list(bars))
    if not sel.any():
        return None
    step = np.floor((pos[sel] % 1.0) * STEPS).astype(int)
    p = 10 ** (fdb[sel] / 10.0)
    m = np.array([p[step == k].mean() if (step == k).any() else 0.0 for k in range(STEPS)])
    if not m.max() > 0:
        return None
    return [round(float(10 * np.log10(v / m.max())), 1) if v > 0 else None for v in m]


def bass_pitches(y_low, sr, times) -> list[float | None]:
    """The MIDI pitch of each onset: the median pyin f0 of the voiced frames from 25 ms after it to the next onset (at
    most 300 ms); None when fewer than 2 frames are voiced. The frames are centred and 186 ms long, so the first ones
    also hear up to 68 ms before the onset (the note before): the median over the frames damps that, it does not
    remove it."""
    import librosa
    if not len(times):
        return []
    f0, voiced, _ = librosa.pyin(y_low, fmin=float(librosa.note_to_hz("A0")), fmax=float(librosa.note_to_hz("C4")),
                                 sr=sr, frame_length=2048, hop_length=HOP, resolution=0.5)
    ft = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
    out = []
    for i, t in enumerate(times):
        end = min(times[i + 1] - 0.01 if i + 1 < len(times) else t + 0.3, t + 0.3)
        sel = (ft >= t + 0.025) & (ft <= end) & voiced & np.isfinite(f0)
        out.append(float(69 + 12 * np.log2(np.median(f0[sel]) / 440.0)) if sel.sum() >= 2 else None)
    return out


def chroma_bars(y, sr, t0: float, bar_s: float, n_bars: int) -> np.ndarray:
    """(n_bars, 12) chroma summed over each bar."""
    import librosa
    hop = 1024
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(c.shape[1]), sr=sr, hop_length=hop)
    idx = np.floor((ft - t0) / bar_s).astype(int)
    out = np.zeros((n_bars, 12))
    ok = (idx >= 0) & (idx < n_bars)
    np.add.at(out, idx[ok], c[:, ok].T)
    return out


def match_chord(vec, bass_pc: int | None = None) -> dict | None:
    """{root, quality, score, margin} for one bar's chroma; None for an empty one."""
    v = np.asarray(vec, dtype=float)
    if not v.sum() > 0:
        return None
    v = v / np.linalg.norm(v)
    scores = []
    for q, ivs in TEMPLATES:
        t = np.zeros(12)
        t[list(ivs)] = 1.0
        t /= np.linalg.norm(t)
        for r in range(12):
            scores.append((float(np.dot(v, np.roll(t, r))), r, q))
    scores.sort(key=lambda s: -s[0])
    best = scores[0]
    if bass_pc is not None:
        tied = [s for s in scores if s[0] >= best[0] - TIE and s[1] == bass_pc]
        if tied:
            best = tied[0]
    other = next(s for s in scores if (s[1], s[2]) != (best[1], best[2]))
    return {"root": best[1], "quality": best[2], "score": round(best[0], 3), "margin": round(best[0] - other[0], 3)}


def cycle(seq: list, confirmed=frozenset()) -> tuple[int, float | None, list]:
    """(k, share, loop) for a section's chord runs: a loop of LOOP_MIN to LOOP_MAX chords, or the whole sequence with
    share None when no loop is there. Two ways to find one, both limited to 2-4 chords so a long run of changes is never
    called a loop:

    - lag: at k, a run equals the run k before in at least REPEAT_SHARE of the pairs, with the sequence at least twice
      the loop's length. A loop longer than half the sequence (less than two passes: `iv VI i VII iv VI`) only when
      every pair matches, there are at least 2 pairs, and its rotation class is in `confirmed` (another section of the
      track plays it twice through): alone, `a b c b a b` would read as `a b c b`. The loop is the most common window
      at the loop's own phase.
    - a loop played in order with now and then a chord put in or left out (`_follow`): at least 2 loops' worth of runs
      matched in order, a score (matches less left-out chords) of FOLLOW_SHARE of the runs, ties to the longer loop.
      `i VII iv VI i VII i iv VI i`: 9 of 10.

    Anything short of an exact repeat (a lag under 100 %, or an alignment) is a loop only when the sequence is more
    loop-like (`_evidence`) than SHUFFLE_BEAT of SHUFFLES orderings of the same runs with no chord twice in a row
    (fixed seed, so a run is repeatable): three or four chords in any order often reach 75 % on one reading or
    another, and only what few orderings reach is a loop. The lag reading comes first (the smallest k), then the
    alignment.

    The loop starts where the section first plays it (`phase_loops` puts every section of a track on one phase)."""
    n = len(seq)
    if n < 2 * LOOP_MIN:
        return n, None, list(seq)
    found = []                                       # (k, share, loop): the inexact ones, tested together below
    for k in range(LOOP_MIN, LOOP_MAX + 1):
        pairs = n - k
        if pairs < 2:
            break
        share = _lag(seq, k)
        if (2 * k <= n and share >= REPEAT_SHARE) or share == 1.0:
            wins = Counter(tuple(seq[i:i + k]) for i in range(0, n - k + 1, k))
            loop = list(max(wins, key=lambda w: (wins[w], -next(i for i in range(0, n, k) if tuple(seq[i:i + k]) == w))))
            if share == 1.0 and (2 * k <= n or _rotation(loop) in confirmed):
                return k, 1.0, loop
            if 2 * k <= n:
                found.append((k, round(share, 2), loop))
    best = _best_alignment(seq)
    if best and best[0] / n >= FOLLOW_SHARE:
        found.append((best[1], round(best[0] / n, 2), best[2]))
    if found and _beats_shuffles(_evidence(seq), [_evidence(x) for x in _shuffles(seq)]):
        return found[0]
    return n, None, list(seq)


def _evidence(seq) -> float:
    """How loop-like a sequence is, one number for all the inexact readings at once (so testing several does not add up
    their chances): the best lag share at 2-4 with two passes, or the best alignment score over the runs."""
    n = len(seq)
    lags = [_lag(seq, k) for k in range(LOOP_MIN, LOOP_MAX + 1) if 2 * k <= n]
    al = _best_alignment(seq)
    return max(lags + [al[0] / n if al else 0.0])


def _lag(seq, k: int) -> float:
    return sum(seq[i] == seq[i - k] for i in range(k, len(seq))) / (len(seq) - k)


def _beats_shuffles(value, null: list) -> bool:
    """Does `value` beat SHUFFLE_BEAT of the shuffles' values (a tie is not beaten)?"""
    return bool(null) and sum(x >= value for x in null) <= (1 - SHUFFLE_BEAT) * len(null)


def _best_alignment(seq) -> tuple[int, int, list] | None:
    """(score, k, loop) of the best `_follow` over the windows of 2-4 runs, at least 2 loops' worth matched."""
    best = None
    for k in range(LOOP_MIN, LOOP_MAX + 1):
        for w in sorted({tuple(seq[i:i + k]) for i in range(len(seq) - k + 1)}):
            score, m, first = _follow(seq, w)
            if m >= 2 * k and (best is None or (score, k) > best[:2]):
                best = (score, k, list(w[first:] + w[:first]))
    return best


def _shuffles(seq) -> list[list]:
    """Up to SHUFFLES orderings of `seq` with no chord twice in a row (chord runs never repeat), from a fixed seed."""
    rng, out = random.Random(0), []
    for _ in range(SHUFFLES * 50):
        if len(out) == SHUFFLES:
            break
        x = list(seq)
        rng.shuffle(x)
        if all(a != b for a, b in zip(x, x[1:])):
            out.append(x)
    return out


def _follow(seq, w) -> tuple[int, int, int | None]:
    """(score, matched runs, loop index of the first match) for the best in-order alignment of `seq` to loop `w` played
    over and over: a run that is the loop's next chord matches (+1), a run put in is passed over (0), a loop chord that
    is not played is skipped (-1, so a reordering of the loop's chords does not score as the loop). Free start and end."""
    k = len(w)
    cur = [(0, 0, None)] * k

    def skip(row):
        for _ in range(k - 1):
            for p in range(k):
                sc, m, f = row[p]
                if m and (sc - 1, m) > row[(p + 1) % k][:2]:
                    row[(p + 1) % k] = (sc - 1, m, f)
        return row

    for x in seq:
        nxt = list(cur)                                      # the run passed over
        for p in range(k):
            sc, m, f = cur[p]
            if x == w[p] and (sc + 1, m + 1) > nxt[(p + 1) % k][:2]:
                nxt[(p + 1) % k] = (sc + 1, m + 1, p if f is None else f)
        cur = skip(nxt)
    return max(cur, key=lambda c: (c[0], c[1]))


def _rotation(w) -> tuple:
    w = tuple(w)
    return min(w[i:] + w[:i] for i in range(len(w)))


def confirm_long_loops(sections: list[dict]) -> None:
    """A second look at the sections with no loop: one longer than half the section is accepted when another section of
    the track plays the same loop (any rotation) twice through (`cycle`'s `confirmed`)."""
    passes = {}
    for i, s in enumerate(sections):
        y = s.get("synths") or {}
        if y.get("progression_share") is not None and y.get("loop_runs") and len(y.get("runs") or []) >= 2 * y["loop_runs"]:
            passes.setdefault(_rotation(y["section_progression"].split()), set()).add(i)
    for i, s in enumerate(sections):
        y = s.get("synths") or {}
        if y.get("runs") and y.get("progression_share") is None:
            set_progression(y, frozenset(c for c, where in passes.items() if where - {i}))


def phase_loops(sections: list[dict]) -> None:
    """Every section's synth loop on one phase for the track: a loop heard as `i VII iv VI` in one section and as
    `iv VI i VII` in three is written `iv VI i VII` in all four (the rotation most sections start it on; ties to the
    first section's). Changes `progression` in place."""
    loops = [s["synths"] for s in sections if s.get("synths") and s["synths"].get("progression_share") is not None
             and s["synths"].get("progression")]
    by_class = {}
    for y in loops:
        w = tuple(y["progression"].split())
        by_class.setdefault(_rotation(w), []).append(w)
    for y in loops:
        w = tuple(y["progression"].split())
        seen = Counter(by_class[_rotation(w)])
        top = max(seen, key=lambda r: (seen[r], -by_class[_rotation(w)].index(r)))
        y["progression"] = " ".join(top)


def repeat(sigs: list) -> tuple[int | None, list[int] | None, str | None]:
    """(period, changes, why none) for per-bar signatures (None = the part is silent that bar); see the module docstring.
    A period needs at least MIN_PAIRS pairs of playing bars at it; with fewer at every period, (None, None, why). Bars in
    `changes` are 1-based within the section."""
    n = len(sigs)
    tried = False
    for p in PERIODS:
        if n < 2 * p:
            break
        pairs = [i for i in range(p, n) if sigs[i] is not None and sigs[i - p] is not None]
        if len(pairs) < MIN_PAIRS:
            continue
        tried = True
        if sum(sigs[i] == sigs[i - p] for i in pairs) / len(pairs) >= REPEAT_SHARE:
            return p, [i + 1 for i in pairs if sigs[i] != sigs[i - p]], None
    if not tried:
        return None, None, "repeat: too few playing bars (fewer than %d bar pairs at any period)" % MIN_PAIRS
    return None, [i + 1 for i in range(1, n) if sigs[i] is not None and sigs[i - 1] is not None and sigs[i] != sigs[i - 1]], \
        "repeat: no period of 1, 2, 4 or 8 bars repeats in 75 % of the bar pairs"


# ---- per section

def _bars_of(sec) -> range:
    return range(sec["start_bar"], sec["start_bar"] + sec["bars"])


def rhythm_fields(placed: list[dict], playing: np.ndarray, sec) -> dict:
    """The rhythm part shared by bass and synths for one section."""
    bars = [b for b in _bars_of(sec) if b < len(playing) and playing[b]]
    keep = set(bars)
    mine = [o for o in placed if o["bar"] in keep]
    on = [o for o in mine if o["on_grid"]]
    per_bar = {b: frozenset(o["step"] for o in on if o["bar"] == b) for b in bars}
    with_onsets = [s for s in per_bar.values() if s]
    out = {"bars_playing": len(bars), "bars_with_onsets": len(with_onsets), "rhythm": None, "rhythm_steps": None,
           "rhythm_share": None,
           "onsets_per_bar": round(len(mine) / len(bars), 2) if bars else None,
           "off_grid_share": round(1 - len(on) / len(mine), 2) if mine else None, "notes": []}
    if with_onsets:
        top, count = Counter(with_onsets).most_common(1)[0]
        out.update(rhythm=pattern_text(top), rhythm_steps=[s + 1 for s in sorted(top)],
                   rhythm_share=round(count / len(with_onsets), 2))
    elif bars:
        out["notes"].append("no onset on the grid in the bars it plays")
    else:
        out["notes"].append("silent in this section")
    out["_per_bar"] = per_bar
    out["_on"] = on
    return out


def bass_section(sec, placed, pitches, playing, sounding, tonic) -> dict:
    out = rhythm_fields(placed, playing, sec)
    on, per_bar = out.pop("_on"), out.pop("_per_bar")
    out.update(on_step1_share=None, on_offbeat_8ths_share=None, degrees=None, pitch_names=None, pitched_share=None,
               pitch_classes=None, top_pitch_class=None, top_pitch_class_share=None, movement=None, register=None,
               register_range=None,
               duty=None, style=None, repeat_bars=None, changes_at_bars=None)
    if on:
        out["on_step1_share"] = round(sum(o["step"] == 0 for o in on) / len(on), 2)
        out["on_offbeat_8ths_share"] = round(sum(o["step"] in OFFBEAT_8THS for o in on) / len(on), 2)
    bars = set(per_bar)
    pitched = [(o, p) for o, p in zip(placed, pitches) if o["bar"] in bars and p is not None]
    total = sum(1 for o in placed if o["bar"] in bars)
    root_of, pcs_of = {}, {}
    if pitched:
        notes = [int(round(p)) for _, p in pitched]
        pcs = Counter(m % 12 for m in notes)
        n = sum(pcs.values())
        kept = [(pc, c) for pc, c in pcs.most_common() if c / n >= MIN_PC_SHARE]
        top_pc, top_c = kept[0] if kept else pcs.most_common(1)[0]
        counted = {pc for pc, _ in kept}
        reg = [m for m in notes if m % 12 in counted] or notes                   # the register of the notes counted
        out.update(pitch_classes=len(kept), top_pitch_class=PC_NAMES[top_pc], top_pitch_class_share=round(top_c / n, 2),
                   movement="pedal" if top_c / n >= PEDAL_SHARE else "moving",
                   register=note_name(sorted(reg)[(len(reg) - 1) // 2]),
                   register_range="%s-%s" % (note_name(min(reg)), note_name(max(reg))),
                   pitched_share=round(n / total, 2) if total else None)
        out["pitch_names"] = [{"note": PC_NAMES[pc], "share": round(c / n, 2)} for pc, c in kept]
        if tonic is not None:
            out["degrees"] = [{"degree": degree(pc, tonic), "note": PC_NAMES[pc], "share": round(c / n, 2)} for pc, c in kept]
        else:
            out["notes"].append("degrees: the study has no key (roadie key, then roadie study tag)")
        for b in bars:
            here = [int(round(p)) % 12 for o, p in pitched if o["bar"] == b]
            root_of[b] = here[0] if here else None              # the bar's first pitched note: the root it sits on
            pcs_of[b] = frozenset(pc for pc in here if pc in counted)     # a pitch-tracking slip does not break a repeat
    elif bars:
        out["notes"].append("pitch: no onset had a pitch pyin could follow")
    if bars:
        d = [float(sounding[b]) for b in sorted(bars) if b < len(sounding)]
        out["duty"] = round(float(np.mean(d)), 2)
        out["style"] = "stabby" if out["duty"] < STABBY else "continuous" if out["duty"] >= CONTINUOUS else "between"
        sigs = [(per_bar[b], pcs_of.get(b)) if b in per_bar else None for b in _bars_of(sec)]
        out["repeat_bars"], out["changes_at_bars"], why = repeat(sigs)
        out["_root_of"] = root_of
        if why:
            out["notes"].append(why)
    return out


def set_progression(y: dict, confirmed=frozenset()) -> None:
    """The progression fields of a synths section from its runs (`cycle` on the roots: VII and VIIsus2 are one), with
    `confirmed` the loops (rotation classes) other sections of the track play twice through. Replaces earlier ones."""
    seq = [re.sub(r"(5|sus2|sus4)$", "", r["chord"]) for r in y["runs"]]
    k, share, cyc = cycle(seq, confirmed)
    y["notes"] = [n for n in y["notes"] if not n.startswith("progression:")]
    y.update(progression=" ".join(cyc), progression_share=share, section_progression=" ".join(cyc), loop_bars=None,
             loop_runs=k if share is not None else None)
    if share is not None and y["bars_per_chord"]:
        y["loop_bars"] = round(k * y["bars_per_chord"], 1)     # k chords at the median run: a section that starts
        #                                                        mid-chord does not shorten it
    if share is not None and share < 1.0:
        y["notes"].append("progression: the loop that fits %d %% of the chord changes; every change is in runs"
                          % round(100 * share))
    elif share is None and len(seq) >= 4:
        y["notes"].append("progression: no loop of 2 to 4 chords, every change listed")


def synth_section(sec, placed, playing, chroma, bass_roots, tonic, mode) -> dict:
    out = rhythm_fields(placed, playing, sec)
    _on, per_bar = out.pop("_on"), out.pop("_per_bar")
    out.update(chords=None, progression=None, bars_per_chord=None, loop_bars=None, distinct_chords=None, drone=None,
               repeat_bars=None, changes_at_bars=None)
    bars = list(_bars_of(sec))
    if not per_bar:
        return out
    chords = []
    for b in bars:
        if b not in per_bar or b >= len(chroma):
            chords.append(None)
            continue
        m = match_chord(chroma[b], bass_roots.get(b))
        if m is None:
            chords.append(None)
            continue
        m["name"] = chord_name(m["root"], m["quality"])
        m["roman"] = roman(m["root"], m["quality"], tonic, mode) if tonic is not None else None
        chords.append(m)
    label = [(c["roman"] or c["name"]) if c else None for c in chords]
    out["chords"] = label
    out["chord_margin"] = round(float(np.median([c["margin"] for c in chords if c])), 3) if any(chords) else None
    if tonic is None:
        out["notes"].append("chords are named absolutely: the study has no key (roadie key, then roadie study tag)")
    runs = []                                   # [label, bars, labels]: a run is a stretch on one root
    for c, lab in zip(chords, label):
        if lab is None:
            continue
        if runs and runs[-1][3] == c["root"]:
            runs[-1][1] += 1
            runs[-1][2].append(lab)
        else:
            runs.append([lab, 1, [lab], c["root"]])
    for r in runs:                              # a run is named by its most common chord (VII, VII5, VIIsus2: VII)
        r[0] = Counter(r[2]).most_common(1)[0][0]
    if runs:
        seq = [r[0] for r in runs]
        out["runs"] = [{"chord": r[0], "bars": r[1]} for r in runs]
        lengths = [r[1] for r in runs[1:-1]] if len(runs) >= 3 else [r[1] for r in runs]
        out["bars_per_chord"] = float(np.median(lengths)) if len(runs) > 1 else None
        if len(runs) == 1:
            out["notes"].append("one chord through the section: no chord change")
        set_progression(out)
        out["distinct_chords"] = sorted(set(seq), key=seq.index)
    played = [chroma[b] for b in bars if b in per_bar and b < len(chroma) and chroma[b].max() > 0]
    if played:
        present = np.array([v / v.max() >= DRONE_LEVEL for v in played])
        share = present.mean(axis=0)
        held = [i for i in np.argsort(-share) if share[i] >= DRONE_BARS]
        out["drone"] = {"pitch_classes": [PC_NAMES[i] for i in held], "bars_share": [round(float(share[i]), 2) for i in held]}
        if tonic is not None:
            out["drone"]["degrees"] = [degree(int(i), tonic) for i in held]
    sigs = [(label[i], per_bar.get(b)) if b in per_bar else None for i, b in enumerate(bars)]
    out["repeat_bars"], out["changes_at_bars"], why = repeat(sigs)
    if why:
        out["notes"].append(why)
    return out


def vocal_phrases(y, sr, t0: float, bar_s: float, n_bars: int, loud_bar_db: float) -> list[int]:
    """The beats (from bar 0, 0-based) on which a vocal phrase starts, over the whole track, so a phrase that runs on
    over a section line is not started again there: a phrase is 2 or more beats within 15 dB of the vocal's loud bar
    level (the level sections.py calls playing), a one-beat gap inside it a breath."""
    beat_s = bar_s / drummod.BEATS_PER_BAR
    nb = n_bars * drummod.BEATS_PER_BAR
    c = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)])
    edges = np.clip(np.round((t0 + np.arange(nb + 1) * beat_s) * sr).astype(np.int64), 0, len(y))
    db = 10.0 * np.log10((c[edges[1:]] - c[edges[:-1]]) / np.maximum(np.diff(edges), 1) + 1e-12)
    on = list(db >= loud_bar_db - 15.0)
    for i in range(1, nb - 1):
        if not on[i] and on[i - 1] and on[i + 1]:
            on[i] = True
    starts, i = [], 0
    while i < nb:
        if on[i]:
            j = i
            while j < nb and on[j]:
                j += 1
            if j - i >= 2:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts


def vocal_section(sec, playing, phrase_beats: list[int]) -> dict:
    """Bars with the vocal, the share of the section, and the phrases that start in it (bar and beat within the
    section; a phrase begun in the section before is not counted again)."""
    bars = [b for b in _bars_of(sec) if b < len(playing) and playing[b]]
    out = {"bars_playing": len(bars), "share": round(len(bars) / sec["bars"], 2),
           "bars_with_vocal": bar_ranges([b - sec["start_bar"] + 1 for b in bars]) or None,
           "phrase_starts": None, "phrase_start_beats": None, "notes": []}
    if not bars:
        out["notes"].append("silent in this section")
        return out
    first, nb = sec["start_bar"] * drummod.BEATS_PER_BAR, sec["bars"] * drummod.BEATS_PER_BAR
    starts = [{"bar": (b - first) // 4 + 1, "beat": (b - first) % 4 + 1} for b in phrase_beats if first <= b < first + nb]
    out["phrase_starts"] = starts
    out["phrase_start_beats"] = dict(sorted(Counter(s["beat"] for s in starts).items())) if starts else None
    if not starts:
        out["notes"].append("no phrase starts in this section (a phrase of 2 beats or more within 15 dB of the vocal's "
                            "loud level; one begun in the section before is counted there)")
    return out


def drum_section(sec, drums: dict | None) -> dict | None:
    """drums.json's roles of 4+ measured hits that play in this section, with their pattern there; hat roles get an
    accent field (null: drums.json has no per-hit levels). None when there is no drums.json."""
    if not drums:
        return None
    plan = (drums.get("grid") or {}).get("sections") or []
    if not plan:
        return {"drums_section": None, "roles": [], "notes": ["drums.json has no grid sections: its roles cannot be placed "
                                                              "in a section (re-run roadie drums)"]}
    idx = next((i for i, p in enumerate(plan) if p["start_bar"] <= sec["start_bar"] < p["start_bar"] + p["bars"]), None)
    if idx is None:
        idx = len(plan) - 1 if sec["start_bar"] >= plan[-1]["start_bar"] else 0
    out = {"drums_section": plan[idx]["name"], "roles": [], "notes": []}
    if plan[idx]["name"] != sec["name"]:
        out["notes"].append("drums.json joins sections under %d bars: these roles are those of %r"
                            % (drumsections.MIN_SECTION_BARS, plan[idx]["name"]))
    for kind, st in (drums.get("stems") or {}).items():
        for r in st.get("roles", []):
            if r.get("measured", 0) < drummod.MIN_SHOWN:
                continue
            v = next((v for v in r.get("per_section", []) if v.get("index") == idx), None)
            if v is None or v["hits"] < drummod.MIN_SHOWN:          # a role with a hit or two here is not its pattern
                continue
            row = {"stem": kind, "role": r["name"], "named": r.get("named"), "hits": v["hits"], "bars_with": v["bars_with"],
                   "pattern": " ".join(v["line"][i:i + 4] for i in range(0, STEPS, 4))}
            if r.get("name_steady") is False:
                row["name_steady"] = False
            if kind in HAT_STEMS:
                row["accent_16th"] = None
                row["accent_note"] = ACCENT_MISSING
            out["roles"].append(row)
    if not out["roles"]:
        out["notes"].append("no drum role of %d+ hits plays in this section" % drummod.MIN_SHOWN)
    return out


# ---- the whole study

def _stem(folder: Path, name: str) -> Path | None:
    found = [f for f in sorted(folder.glob(name + ".*")) if f.suffix in AUDIO and not f.name.startswith(".")]
    return found[0] if found else None


def study_key(marker: dict | None) -> dict | None:
    """{name, tonic, mode, source, camelot} from a study.json marker; None without a readable key."""
    if not isinstance(marker, dict) or not isinstance(marker.get("key"), str):
        return None
    try:
        pc, mode = keymod.parse_key(marker["key"])
    except ValueError:
        return None
    return {"name": keymod.key_name(pc, mode), "tonic": pc, "mode": mode, "camelot": keymod.camelot(pc, mode),
            "source": marker.get("key_source"), "confidence": marker.get("key_confidence")}


def _level(m: dict, sec: dict, t0: float, bar_s: float) -> list[float] | None:
    bars = [b for b in _bars_of(sec) if b < len(m["playing"]) and m["playing"][b]]
    return level_by_16th(m["fdb"], t0, bar_s, bars) if bars else None


def micro(stems_dir, key: dict | None = None) -> dict:
    """The micro.json record of a local stems folder (see the module docstring). `key` is `study_key(...)`; None reads
    the study.json beside the stems folder."""
    stems_dir = Path(stems_dir)
    t_start = time.monotonic()
    grid = grid_of(stems_dir)
    if key is None:
        key = study_key(_read_json(stems_dir.parent / "study.json"))
    tonic, mode = (key["tonic"], key["mode"]) if key else (None, None)
    t0, bar_s = grid["t0"], grid["bar_s"]
    notes = [n for n in grid["notes"] if "chance-level" not in n] + list(grid["why"])     # `why` says it once
    if not key:
        notes.append("no key in study.json: bass notes and chords are named absolutely, not as degrees")
    files = {n: _stem(stems_dir, n) for n in STEMS}
    for n, f in files.items():
        if f is None:
            notes.append("no %s stem: %s is null in every section" % (n, n))
    drums = _read_json(stems_dir / "drums.json")
    if not (isinstance(drums, dict) and drums.get("kind") == drummod.KIND):
        drums = None
        notes.append("no drums.json: drums is null in every section (roadie drums <study>)")
    else:
        dg = drums.get("grid") or {}
        if abs(float(dg.get("t0", t0)) - t0) > 0.005 or abs(float(dg.get("tempo", grid["tempo"])) - grid["tempo"]) > 0.001:
            notes.append("drums.json was made on another grid (t0 %s, tempo %s): re-run roadie drums" % (dg.get("t0"), dg.get("tempo")))
    secs = grid["sections"]
    n_bars = max(s["start_bar"] + s["bars"] for s in secs)
    loaded = {n: _load(f) for n, f in files.items() if f is not None}
    if loaded:
        n_bars = max(n_bars, int(math.ceil((max(len(y) for y in loaded.values()) / SR - t0) / bar_s)))
    meas = {}
    for n, y in loaded.items():
        fdb = frame_db(y)
        loud = loud_level(fdb)
        playing = bar_activity(y, SR, t0, bar_s, n_bars)
        meas[n] = {"y": y, "fdb": fdb, "loud": loud, "playing": playing}
        if loud is None:
            notes.append("%s stem is silent" % n)
            meas[n]["placed"] = []
            continue
        meas[n]["placed"] = [o for o in place(onsets(y, SR, fdb, loud, BASS_FMAX if n == "bass" else None), t0, bar_s)
                             if 0 <= o["bar"] < n_bars and playing[o["bar"]]]
    bass_roots = {}
    pitches = []
    sounding = np.zeros(n_bars)
    if "bass" in meas and meas["bass"]["loud"] is not None:
        import librosa
        m = meas["bass"]
        pitches = bass_pitches(librosa.resample(m["y"], orig_sr=SR, target_sr=PITCH_SR), PITCH_SR,
                               [o["t"] for o in m["placed"]])
        frames_bar = np.floor((np.arange(len(m["fdb"])) * HOP / SR - t0) / bar_s).astype(int)
        snd = m["fdb"] >= m["loud"] - DUTY_DB
        for b in range(n_bars):
            sel = frames_bar == b
            sounding[b] = float(snd[sel].mean()) if sel.any() else 0.0
    chroma = None
    if "synths" in meas and meas["synths"]["loud"] is not None:
        chroma = chroma_bars(meas["synths"]["y"], SR, t0, bar_s, n_bars)
    phrase_beats = None
    if "vocals" in meas and meas["vocals"]["loud"] is not None:
        v = meas["vocals"]
        c = np.concatenate([[0.0], np.cumsum(np.asarray(v["y"], dtype=np.float64) ** 2)])
        edges = np.clip(np.round((t0 + np.arange(n_bars + 1) * bar_s) * SR).astype(np.int64), 0, len(v["y"]))
        bdb = 10.0 * np.log10((c[edges[1:]] - c[edges[:-1]]) / np.maximum(np.diff(edges), 1) + 1e-12)
        loud_bar = float(np.percentile(bdb[bdb > -50.0], 95)) if (bdb > -50.0).sum() >= 4 else float(bdb.max())
        phrase_beats = vocal_phrases(v["y"], SR, t0, bar_s, n_bars, loud_bar)
    out_secs = []
    for sec in secs:
        rec = {k: sec[k] for k in ("index", "name", "file_bar", "start_bar", "bars")}
        rec["reliable"] = grid["reliable"]
        if "bass" in meas and meas["bass"]["loud"] is not None:
            b = bass_section(sec, meas["bass"]["placed"], pitches, meas["bass"]["playing"], sounding, tonic)
            bass_roots.update(b.pop("_root_of", {}))
            b["level_by_16th_db"] = _level(meas["bass"], sec, t0, bar_s)
            if b["style"] == "continuous" and (b["onsets_per_bar"] or 0) < FEW_ONSETS:
                b["notes"].append("continuous with under %g onsets a bar: it moves in level more than in note starts "
                                  "(level_by_16th_db says where it is quieter, not why)" % FEW_ONSETS)
            rec["bass"] = b
        else:
            rec["bass"] = None
        if chroma is not None:
            rec["synths"] = synth_section(sec, meas["synths"]["placed"], meas["synths"]["playing"], chroma, bass_roots,
                                          tonic, mode)
            rec["synths"]["notes"].insert(0, SYNTHS_NOTE)
            rec["synths"]["level_by_16th_db"] = _level(meas["synths"], sec, t0, bar_s)
        else:
            rec["synths"] = None
        if phrase_beats is not None:
            rec["vocals"] = vocal_section(sec, meas["vocals"]["playing"], phrase_beats)
        else:
            rec["vocals"] = None
        rec["drums"] = drum_section(sec, drums)
        out_secs.append(rec)
    confirm_long_loops(out_secs)
    phase_loops(out_secs)
    return {"kind": KIND, "version": VERSION, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "grid": {"tempo": grid["tempo"], "t0": round(t0, 4), "beats_per_bar": drummod.BEATS_PER_BAR,
                     "downbeat_s": grid["downbeat_s"], "downbeat_verdict": grid["verdict"],
                     "downbeat_confidence": grid["confidence"], "warp_map": grid["warp_map"],
                     "reliable": grid["reliable"], "unreliable_because": grid["why"],
                     "bar_0": "the downbeat moved back whole bars to the first bar line in the file, as drums.json"},
            "key": {k: key[k] for k in ("name", "camelot", "source", "confidence")} if key else None,
            "drums_made": drums.get("made") if drums else None,
            "inputs": [{"name": f.name, "size": f.stat().st_size} for f in list(files.values()) if f is not None]
                      + [{"name": n, "size": (stems_dir / n).stat().st_size}
                         for n in ("alignment.json", "sections.json", "drums.json") if (stems_dir / n).exists()],
            "legend": {"rhythm": "16 steps from the bar line, one per 16th: o = an onset there; rhythm_steps are 1-based",
                       "degrees": "scale degrees of the study's key (1 = tonic, b3 = minor third)",
                       "chords": "per bar from the section's first, relative to the key: upper case major, lower minor, "
                                 "5 = root and fifth, sus2/sus4; null = the synths do not play that bar",
                       "bars": "bar numbers inside a section are 1-based from its first bar"},
            "notes": notes, "seconds": round(time.monotonic() - t_start, 1), "sections": out_secs}


# ---- reading and writing a study (local or pCloud)

def _rclone(*args) -> str:
    try:
        return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise OSError("rclone %s failed: %s" % (args[0], (e.stderr or "").strip() or "exit %s" % e.returncode))


def micro_of(stems_dir, write: bool = True) -> dict:
    """`micro` for a stems folder, written to `<stems_dir>/micro.json` with `write`. A study on the pCloud mount is never
    read through the FUSE mount: its three stems, alignment.json, sections.json, drums.json and study.json are copied to a
    temp folder with rclone (deleted after), and micro.json goes back with `rclone copyto`, then is read back and compared."""
    stems_dir = Path(stems_dir)
    remote = keymod.pcloud_remote(stems_dir)
    if remote is None:
        rec = micro(stems_dir)
        if write:
            tmp = stems_dir / ".micro.json.tmp"
            try:
                tmp.write_text(json.dumps(rec, indent=2) + "\n")
                os.replace(tmp, stems_dir / "micro.json")
            finally:
                tmp.unlink(missing_ok=True)
            rec["written"] = str(stems_dir / "micro.json")
        return rec
    names = _rclone("lsf", "--files-only", remote).splitlines()
    wanted = [n for n in names if n in ("alignment.json", "sections.json", "drums.json")]
    for s in STEMS:
        hit = sorted(n for n in names if Path(n).stem == s and Path(n).suffix in AUDIO)
        wanted += hit[:1]
    parent = remote.rpartition("/")[0]
    with tempfile.TemporaryDirectory(prefix="roadie-micro-") as tmp:
        tmp = Path(tmp)
        local = tmp / Path(remote).name
        local.mkdir()
        for n in wanted:
            _rclone("copyto", "%s/%s" % (remote, n), str(local / n))
        marker = None
        if "study.json" in _rclone("lsf", "--files-only", parent).splitlines():
            try:
                marker = json.loads(_rclone("cat", parent + "/study.json"))
            except ValueError:
                marker = None
        rec = micro(local, study_key(marker))
        if write:
            text = json.dumps(rec, indent=2) + "\n"
            (local / "micro.json").write_text(text)
            _rclone("copyto", str(local / "micro.json"), remote + "/micro.json")
            if _rclone("cat", remote + "/micro.json") != text:
                raise OSError("%s/micro.json: read back after writing and it is not what was written" % remote)
            rec["written"] = remote + "/micro.json"
    return rec


def format_summary(rec: dict) -> str:
    """A few lines per section, for the terminal (the report renders the full record)."""
    g = rec["grid"]
    lines = ["grid: %.3f bpm, bar 0 at %.3f s, downbeat %s%s" % (
        g["tempo"], g["t0"], g["downbeat_verdict"] or "not measured", "" if g["reliable"] else "  UNRELIABLE: " + "; ".join(g["unreliable_because"])),
             "key: %s" % (rec["key"]["name"] if rec["key"] else "unknown")]
    lines += ["note: " + n for n in rec["notes"]]
    for s in rec["sections"]:
        lines.append("")
        lines.append("%s  (file bar %s, %d bars)" % (s["name"], s["file_bar"], s["bars"]))
        b = s["bass"]
        if b:
            lines.append("  bass    %s  %s  %s/bar  degrees %s  %s  %s %s" % (
                b["rhythm"] or "-", "%d %%" % (100 * b["rhythm_share"]) if b["rhythm_share"] is not None else "",
                b["onsets_per_bar"] if b["onsets_per_bar"] is not None else "-",
                ", ".join(d["degree"] for d in b["degrees"]) if b["degrees"] else "-", b["register"] or "-",
                b["movement"] or "-", b["style"] or ""))
        sy = s["synths"]
        if sy:
            lines.append("  synths  %s  %s/bar  %s  %s bars per chord" % (
                sy["rhythm"] or "-", sy["onsets_per_bar"] if sy["onsets_per_bar"] is not None else "-",
                sy["progression"] or "-", "%g" % sy["bars_per_chord"] if sy["bars_per_chord"] else "-"))
        v = s["vocals"]
        if v:
            lines.append("  vocals  bars %s (%d %%), %d phrases" % (v["bars_with_vocal"] or "-", 100 * v["share"],
                                                                 len(v["phrase_starts"] or [])))
        d = s["drums"]
        if d:
            for r in d["roles"]:
                lines.append("  %-7s %-14s %s" % (r["stem"], r["role"], r["pattern"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie micro", description="Each instrument's patterns per section of a study")
    ap.add_argument("target", help="a study (title or path) or a stems folder")
    ap.add_argument("--json", action="store_true", help="print the micro.json record")
    ap.add_argument("--no-write", action="store_true", help="write no micro.json")
    args = ap.parse_args(argv)
    try:
        import roadie_drums
        p = Path(args.target).expanduser()
        sd = p if (p / "sections.json").is_file() else roadie_drums.stems_dir_of(args.target)
        rec = micro_of(sd, write=not args.no_write)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print("micro: %s" % e, file=sys.stderr)
        return 1
    print(json.dumps(rec, indent=2) if args.json else format_summary(rec))
    if not args.json:
        print("\n%s (%.0f s)" % ("written to %s" % rec["written"] if "written" in rec else "nothing written (--no-write)",
                                 rec["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

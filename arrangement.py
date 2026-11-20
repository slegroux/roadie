#!/usr/bin/env python3
"""Read a range of Session View scenes from Live as an arrangement: which element plays when.

    arrangement.py read --scenes A-B [--roles MAP.json] [--json] [--metrics]
    arrangement.py ref <study> [--json]
    arrangement.py compare --scenes A-B --ref <study>... | --ref all [--roles MAP] [--set-json FILE] [--out REPORT.md]

The Set's scenes, played top to bottom, ARE the arrangement. This reads scenes A..B (1-based, as
Live's scene numbers) and builds, per 8-bar step, one cell per scene x role: playing, silent, or
changed against the step before. It is READ-ONLY: it only gets and counts through
`sideman.client.Live`, never sets, calls or plays anything.

WHAT "PLAYING" MEANS, and each rule is a bug the first prototype had:
  - A note SOUNDS from its start to its end. A chord held 16 bars plays through the 8-bar boundary
    in the middle of it: reading only note starts calls the second step silent.
  - A clip plays from its start marker to its loop end, then loops loop_start..loop_end. A note
    that began before the point where the clip joins it is NOT retriggered (Live does not play a
    note it joins mid-way): a loop that starts after its held notes began is silent.
  - Notes outside the loop (before loop_start, past loop_end) never play. A non-looping clip
    plays start_marker..end_marker once and is silent after (a one-shot).
  - A scene's length is the length of its longest looping clip (a one-shot longer than that is a
    tail running past the scene), in whole bars. The scene NAME is never parsed: "Build 2 · 2 bars lead" says nothing about how long the scene runs.

ROLES. An explicit map (--roles, JSON {"track name": "role", "track name/chain name": "role"},
case-insensitive; the role "ignore" leaves a track out) wins; otherwise name keywords decide
(KEYWORDS below). A Drum Rack with no explicit entry of its own is split into pads: a note belongs
to the chain whose `in_note` is its pitch, and the chain's name picks the role. A track, or a pad,
no rule maps goes to the role "unmapped" and is listed by name in the output: never dropped.

CHANGE is compared like with like: a loop that fits an 8-bar step is its cycle, so a 3-bar loop does
not "vary" because the next step starts mid-loop; steps of unequal length are compared over the
shorter; a note or audio clip held in from the step before, starting nothing new, is the same; a
voice that runs out while others of its role play on (a one-shot) is not a change of the role.

--metrics adds the structure of the range (structure_metrics): sections by the base labels in the
scene names (Intro, Main N, Build N, Drop N, Breakdown, Outro) and their lengths, the 8-bar steps
that add nothing ("nothing new") or only remove, pitched parts per section, the hook (the lead,
any track mapped to it: LD, LEAD, 27-Diva), the bass entry bar, how long the kick is out between
two stretches of kick, Drop 2 / Drop 1, and whether the hats stay on. A clip named "silent
carrier" counts as silent.

ref <study> runs the same metrics on a reference study: its sections and its per-bar stem activity
(`bar_activity` in sections.json) as one-bar steps, in file bars. No notes, so no "varied", no
"nothing new" and no hook: the output says so.

NOT READ: clip automation (envelopes). A filter ramp on a lead counts as "no change"; the output
says so in its notes.
"""

import os
import sys

# Re-exec with local .venv python if present: `sideman` (the Live client) lives there.
_real_script = os.path.realpath(__file__)
_venv = os.path.join(os.path.dirname(_real_script), ".venv")
_venv_py = os.path.join(_venv, "bin", "python")
if os.path.realpath(sys.prefix) == os.path.realpath(_venv):
    os.environ.pop("ROADIE_REEXEC", None)    # arrived; later scripts may hop again
elif os.path.exists(_venv_py) and not os.environ.get("ROADIE_REEXEC"):
    os.environ["ROADIE_REEXEC"] = "1"        # one hop, whatever the venv looks like
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import hashlib
import json
import math
import re
import time

try:
    from sideman.client import Live, LiveError
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] != "sideman":
        raise
    sys.exit("sideman is not installed in .venv: run ./install.sh")

import live_batch

BEATS_PER_BAR = 4
WINDOW_BARS = 8
WINDOW_BEATS = WINDOW_BARS * BEATS_PER_BAR
EPS = 1e-6
MIN_LOOP_BEATS = 1 / 64          # a loop shorter than this would unroll into millions of passes
DRUM_RACK = "DrumGroupDevice"

# Role keywords, first match wins, on the lowercased track or chain name. The order is load-bearing:
# "Bass Drum" must be a kick before "bass" makes it a bass, "Hi Tom" a tom before "hi" ever matters.
KEYWORDS = (
    ("kick", r"kick|(?<![a-z])bd(?![a-z])|bass drum"),
    ("snare", r"snare|(?<![a-z])sn(?![a-z])"),
    ("clap", r"clap"),
    ("ride", r"ride"),
    ("shaker", r"shk|shaker"),
    ("tom", r"(?<![a-z])[lmh]?toms?(?![a-z])"),
    ("hats", r"hat|(?<![a-z])hh"),
    ("perc", r"perc|conga|(?<![a-z])rim(?![a-z])|bongo|tambourine|cowbell"),
    ("bass", r"bass|(?<![a-z])sub(?![a-z])"),
    ("lead", r"(?<![a-z])lead(?![a-z])|(?<![a-z])ld(?![a-z])"),
    ("arp", r"(?<![a-z])arp"),
    ("pad", r"(?<![a-z])pad(?![a-z])"),
    ("drone", r"drone"),
    ("pluck", r"pluck"),
    ("vox", r"vox|vocal|voice"),
    ("fx", r"(?<![a-z])fx(?![a-z])|riser|impact|(?<![a-z])down|(?<![a-z])rev(?![a-z])|sweep|noise"),
    ("synth", r"synth"),
)
ROLE_ORDER = [r for r, _ in KEYWORDS]
# A clip whose name ends in "(silent carrier)" holds a device open (a filter, a sidechain) and makes
# no sound of its own: counted as silent, whatever the track maps to and whatever it holds (audio
# content is not read, so a name is the only way). A role-map entry {"<clip name>": "play"} makes
# that one clip count again.
SILENT_CLIP = re.compile(r"\(silent carrier\)\s*$", re.I)
PLAY = "play"
UNMAPPED = "unmapped"
IGNORE = "ignore"

NOTES = [
    "Clip automation (envelopes) is not read: a filter ramp or a volume ride on a clip counts as "
    "no change. Silence here is silence of notes and clips, not of the sound.",
    "Audio clips count as playing wherever they are placed: their content is not read, so a clip "
    "that only carries a filter or a sidechain shows as playing, unless its name says "
    "\"silent carrier\" (then it counts as silent).",
    "Bars are 4 beats; the Set's time signature is not read.",
]


# --------------------------------------------------------------------------
# roles
# --------------------------------------------------------------------------

def load_role_map(path):
    """{lowercased key: role} from a JSON file of {"track name": "role", "track/chain": "role"}."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not all(isinstance(v, str) for v in raw.values()):
        raise ValueError("%s: expected a JSON object of name -> role" % path)
    return {str(k).lower(): v for k, v in raw.items()}


def keyword_role(name):
    low = (name or "").lower()
    for role, pattern in KEYWORDS:
        if re.search(pattern, low):
            return role
    return None


def _lower_map(roles):
    return {str(k).lower(): v for k, v in (roles or {}).items()}


# --------------------------------------------------------------------------
# what a clip plays: the pure part
# --------------------------------------------------------------------------

def extent_beats(clip):
    """How long the clip runs once launched, to loop around: its loop for a looping clip, its
    start to end marker for a non-looping one."""
    if clip["looping"]:
        return max(clip["loop_end"] - clip["loop_start"], 0.0)
    return max(clip["end_marker"] - clip["start_marker"], 0.0)


def scene_beats(clips):
    """How long a scene runs, from its clips: its longest looping clip. A non-looping clip is a
    one-shot that may well outlast the scene (a 16-bar downer sample in an 8-bar scene), so it
    sets the length only when nothing in the scene loops."""
    loops = [extent_beats(c) for c in clips if c["looping"]]
    return max(loops or [extent_beats(c) for c in clips] or [0.0])


def passes(clip, horizon):
    """[(scene_time, clip_from, clip_to)]: stretches of the clip played back to back from the
    launch. A looping clip plays start_marker..loop_end once, then loop_start..loop_end again and
    again; a non-looping clip plays start_marker..end_marker once."""
    m = clip["start_marker"]
    if not clip["looping"]:
        return [(0.0, m, clip["end_marker"])]
    ls, le = clip["loop_start"], clip["loop_end"]
    if m >= le - EPS:
        m = ls
    out = [(0.0, m, le)]
    length = le - ls
    t = le - m
    if length >= MIN_LOOP_BEATS:
        while t < horizon - EPS:
            out.append((t, ls, le))
            t += length
    return out


def sounding(clip, notes, horizon):
    """[(start, end, pitch, velocity)] in scene beats: every note as it sounds, from its start to
    its end, cut at the end of the pass it plays in. A note counts only in the pass that reaches
    its start: one that began before the clip joined it does not sound (Live does not retrigger
    it), and one outside the loop never does."""
    out = []
    for t0, a, b in passes(clip, horizon):
        for n in notes:
            s = n["start_time"]
            if not (a - EPS <= s < b - EPS):
                continue
            start = t0 + (s - a)
            end = t0 + (min(s + n["duration"], b) - a)
            if start < horizon - EPS and end > start + EPS:
                out.append((start, min(end, horizon), n["pitch"], n.get("velocity", 100.0)))
    return out


def _digest(obj):
    return hashlib.md5(repr(obj).encode()).hexdigest()[:6]


def steady_from(clip):
    """Scene beat from which a looping clip is in its steady cycle: 0 when it starts at its loop
    start, else when the first, partial pass (from the start marker) ends."""
    m, ls, le = clip["start_marker"], clip["loop_start"], clip["loop_end"]
    return 0.0 if abs(m - ls) < EPS or m >= le - EPS else le - m


def midi_record(clip, notes, intervals, w0, w1):
    """What a MIDI voice plays in scene beats [w0, w1), or None if it is silent. A note sounding
    at all in the window counts, one held in from an earlier step included.

    A clip in its steady loop, with a cycle no longer than an 8-bar step, is its cycle ("loop"):
    the same loop is the same wherever the step starts in it, so a 3-bar loop does not "vary"
    from one step to the next. It carries its cycle and the phase the step starts at, so a step
    before the steady loop (a start marker inside the loop) can still be compared with it.
    Anything else is its events ("ev"): those that start in the window, relative to its start, the
    pitches held in from before it, and those still sounding at its end.
    """
    ivs = [i for i in intervals if i[0] < w1 - EPS and i[1] > w0 + EPS]
    if not ivs:
        return None
    ls, le = clip["loop_start"], clip["loop_end"]
    if (clip["looping"] and MIN_LOOP_BEATS <= le - ls <= WINDOW_BEATS + EPS
            and w0 >= steady_from(clip) - EPS):
        cycle = tuple(sorted((n["pitch"], round(float(n["start_time"] - ls), 3),
                              round(float(min(n["start_time"] + n["duration"], le)
                                          - n["start_time"]), 3),
                              int(n.get("velocity", 100.0) // 8))
                             for n in notes if ls - EPS <= n["start_time"] < le - EPS))
        length = float(le - ls)
        phase = (w0 - steady_from(clip)) % length
        return ("loop", round(length, 3), _digest(cycle), cycle, round(phase, 3),
                round(float(w1 - w0), 3))
    return ("ev", round(float(w1 - w0), 3),
            tuple(sorted((p, round(float(s - w0), 3), round(float(e - s), 3), int(v // 8))
                         for s, e, p, v in ivs if s >= w0 - EPS)),
            tuple(sorted({p for s, e, p, v in ivs if s < w0 - EPS})),
            tuple(sorted({p for s, e, p, v in ivs if e >= w1 - EPS})))


def audio_record(clip, horizon, w0, w1):
    """What an audio clip plays in [w0, w1): it plays wherever it is placed, and is the same clip
    from one step to the next however far into it the step starts."""
    for t0, a, b in passes(clip, horizon):
        if min(w1, t0 + (b - a)) > max(w0, t0) + EPS:
            length = clip["loop_end"] - clip["loop_start"]
            return ("aud", clip["name"], round(float(length), 3) if clip["looping"] else None)
    return None


def _canon(rec):
    """The record without what depends on where in the loop a step starts: what a signature, and
    so "how many variants", is made of."""
    return rec[:3] if rec[0] == "loop" else rec


def _unrolled(rec, n):
    """The events a "loop" record starts in its first n beats, the way an "ev" record lists them."""
    _kind, length, _d, cycle, phase, _n = rec
    out = []
    for p, off, dur, vb in cycle:
        t = off - phase
        if t < -EPS:
            t += length * math.ceil((-t - EPS) / length)
        while t < n - EPS:
            out.append((p, round(t, 3), dur, vb))
            t += length
    return sorted(out)


def same_content(a, b):
    """Is record `b` (a step) what record `a` (the step before) was, compared like with like?
    Two loops are the same if their cycles are. Events are compared over the length of the shorter
    step; a step with nothing new starting in it, only notes held in from the step before, is the
    same as that step. A loop against events (a start marker inside the loop makes the first step
    events) is compared by what starts: the loop unrolled from its phase against the events."""
    if a[0] == b[0]:
        if a[0] == "loop":
            return a[1:3] == b[1:3]
        if a[0] == "aud":
            return a == b
        n = min(a[1], b[1])
        sa = [e for e in a[2] if e[1] < n - EPS]
        sb = [e for e in b[2] if e[1] < n - EPS]
        return sa == sb or (not sb and bool(b[3]) and set(b[3]) <= set(a[4]))
    if {a[0], b[0]} == {"loop", "ev"}:
        n = min(a[5] if a[0] == "loop" else a[1], b[5] if b[0] == "loop" else b[1])
        starts = lambda r: (_unrolled(r, n) if r[0] == "loop"
                            else [e for e in r[2] if e[1] < n - EPS])
        return starts(a) == starts(b)
    return False


# --------------------------------------------------------------------------
# reading Live
# --------------------------------------------------------------------------

def _vec_len(v):
    """Length of a list property as `get` returns it: the items, or {"__vector__", "count"}."""
    if isinstance(v, list):
        return len(v)
    if isinstance(v, dict) and v.get("__vector__"):
        return int(v["count"])
    return 0


def _reads(live, specs):
    """The value of every {path, property}, None where Live would not give it. A get_batch never
    fails as a whole, so an unreadable property costs only its own entry."""
    out = []
    for i in range(0, len(specs), live_batch.OPS_PER_REQUEST):
        res = live.get_batch(specs[i:i + live_batch.OPS_PER_REQUEST])["results"]
        out.extend(r.get("value") if r.get("ok") else None for r in res)
    return out


CLIP_PROPS = ("name", "looping", "loop_start", "loop_end", "start_marker", "end_marker",
              "is_midi_clip", "muted", "warping")


class ReadTooSlow(ValueError):
    """The read of the Set went past its deadline: Live answers, but slowly."""


class _Timed:
    """`live` with a deadline for the whole read: each request checks the clock first. A full read
    is about 130 requests (counts, batched gets, one notes_get per MIDI clip); the client's own
    limit is per request, so a Live that answers slowly would otherwise keep a caller waiting."""

    def __init__(self, live, seconds):
        self._live, self._seconds = live, seconds
        self._deadline = time.monotonic() + seconds

    def __getattr__(self, name):
        fn = getattr(self._live, name)

        def call(*a, **k):
            if time.monotonic() > self._deadline:
                raise ReadTooSlow("reading the Set from Live took over %g s: save a read with `roadie "
                                  "arrangement read --json` once Live answers and pass it as the saved "
                                  "read" % self._seconds)
            return fn(*a, **k)
        return call


def read_set(live, first, last, timeout_s=None):
    """Everything the table needs, read once: scene names, tracks, clips with their notes, Drum
    Rack chains. `first`/`last` are 1-based scene numbers. `timeout_s` is a deadline for the whole
    read (ReadTooSlow), checked before each request."""
    if timeout_s is not None:
        live = _Timed(live, timeout_s)
    n_scenes = live.count("live_set", "scenes")["count"]
    if not 1 <= first <= last <= n_scenes:
        raise ValueError("scenes %d-%d: the Set has %d scenes" % (first, last, n_scenes))
    n_tracks = live.count("live_set", "tracks")["count"]
    tempo = live.get("live_set", "tempo")["value"]
    set_path = live.get("live_set", "file_path").get("value") or ""
    scene_ids = list(range(first - 1, last))

    names = _reads(live, [{"path": "live_set scenes %d" % s, "property": "name"} for s in scene_ids])
    tp = ("name", "is_foldable", "has_midi_input", "mute", "muted_via_solo", "is_grouped")
    tr = _reads(live, [{"path": "live_set tracks %d" % t, "property": p}
                       for t in range(n_tracks) for p in tp])
    tracks = []
    for t in range(n_tracks):
        v = dict(zip(tp, tr[t * len(tp):(t + 1) * len(tp)]))
        tracks.append({"index": t, "name": v["name"], "foldable": bool(v["is_foldable"]),
                       "midi": bool(v["has_midi_input"]), "grouped": bool(v["is_grouped"]),
                       "muted": bool(v["mute"] or v["muted_via_solo"]),
                       "muted_why": "muted" if v["mute"] else
                       "muted by another track's solo" if v["muted_via_solo"] else ""})
    # A track in a muted (or solo-muted) group is silent too: read up to three group levels.
    gp = [(t["index"], d) for t in tracks if t["grouped"] for d in (1, 2, 3)]
    gv = _reads(live, [{"path": "live_set tracks %d %s" % (t, " ".join(["group_track"] * d)),
                        "property": p} for t, d in gp for p in ("name", "mute", "muted_via_solo")])
    for k, (t, _d) in enumerate(gp):
        gname, gmute, gsolo = gv[k * 3:(k + 1) * 3]
        if (gmute or gsolo) and not tracks[t]["muted"]:
            tracks[t]["muted"] = True
            tracks[t]["muted_why"] = "in group %r, which is %s" % (
                gname, "muted" if gmute else "muted by another track's solo")

    # Group tracks hold no clips of their own.
    slots = [(t["index"], s) for t in tracks if not t["foldable"] for s in scene_ids]
    has = _reads(live, [{"path": "live_set tracks %d clip_slots %d" % ts, "property": "has_clip"}
                        for ts in slots])
    present = [ts for ts, h in zip(slots, has) if h]

    props = _reads(live, [{"path": "live_set tracks %d clip_slots %d clip" % ts, "property": p}
                          for ts in present for p in CLIP_PROPS])
    clips, warnings = {}, []
    for k, (t, s) in enumerate(present):
        c = dict(zip(CLIP_PROPS, props[k * len(CLIP_PROPS):(k + 1) * len(CLIP_PROPS)]))
        where = "track %r scene %d" % (tracks[t]["name"], s + 1)
        if any(c[p] is None for p in ("looping", "loop_start", "loop_end", "start_marker",
                                      "end_marker", "is_midi_clip")):
            warnings.append("clip not readable, left out: " + where)
            continue
        if not c["is_midi_clip"] and not c["warping"]:
            for p in ("loop_start", "loop_end", "start_marker", "end_marker"):
                c[p] = c[p] * tempo / 60.0          # an unwarped clip's markers are seconds
        c["notes"] = []
        clips[(t, s)] = c

    for (t, s), c in clips.items():
        if c["is_midi_clip"]:
            r = live.notes_get("live_set tracks %d clip_slots %d clip" % (t, s),
                               from_time=-1000.0, time_span=100000.0)
            c["notes"] = [n for n in r["notes"]
                          if not n.get("mute") and n.get("velocity", 100.0) > 0
                          and n["duration"] > 0]

    # Drum Racks, on the MIDI tracks that have a clip in range: chain name and in_note.
    midi_tracks = sorted({t for (t, _s), c in clips.items() if c["is_midi_clip"]})
    dev_n = _reads(live, [{"path": "live_set tracks %d" % t, "property": "devices"}
                          for t in midi_tracks])
    devs = [(t, d) for t, v in zip(midi_tracks, dev_n) for d in range(_vec_len(v))]
    cls = _reads(live, [{"path": "live_set tracks %d devices %d" % td, "property": "class_name"}
                        for td in devs])
    racks = [td for td, c in zip(devs, cls) if c == DRUM_RACK]
    chain_n = _reads(live, [{"path": "live_set tracks %d devices %d" % td, "property": "chains"}
                            for td in racks])
    chains = [(t, d, i) for (t, d), v in zip(racks, chain_n) for i in range(_vec_len(v))]
    cv = _reads(live, [{"path": "live_set tracks %d devices %d chains %d" % tdi, "property": p}
                       for tdi in chains for p in ("name", "in_note")])
    pads = {}                                      # track -> {in_note: [chain names]}
    for k, (t, _d, _i) in enumerate(chains):
        name, note = cv[k * 2], cv[k * 2 + 1]
        if note is not None:
            pads.setdefault(t, {}).setdefault(int(note), []).append(name or "")

    return {"set": os.path.basename(set_path), "tempo": tempo, "scene_ids": scene_ids,
            "scene_names": names, "tracks": tracks, "clips": clips, "pads": pads,
            "warnings": warnings}


# --------------------------------------------------------------------------
# voices: what carries a role
# --------------------------------------------------------------------------

def build_voices(data, role_map):
    """One voice per track, or per pad of a Drum Rack that has no entry of its own in the map.
    A voice is {key, label, track, pitch or None, role}; unmapped ones keep the role "unmapped"
    and are listed by the caller."""
    voices, ignored = [], []
    for t in data["tracks"]:
        have = [(s, c) for (ti, s), c in data["clips"].items() if ti == t["index"]]
        if not have:
            continue
        name = t["name"] or "track %d" % t["index"]
        explicit = role_map.get(name.lower())
        if explicit == IGNORE:
            ignored.append(name)
        elif explicit or t["index"] not in data["pads"]:
            role = explicit or keyword_role(name) or UNMAPPED
            voices.append({"key": (t["index"], None), "label": name, "track": t["index"],
                           "pitch": None, "role": role})
        else:
            rack = data["pads"][t["index"]]
            pitches = sorted({n["pitch"] for _s, c in have for n in c["notes"]})
            for p in pitches:
                chain_names = rack.get(p)
                if not chain_names:
                    voices.append({"key": (t["index"], p), "label": "%s/note %d" % (name, p),
                                   "track": t["index"], "pitch": p, "role": UNMAPPED})
                    continue
                for cn in chain_names:
                    sub = role_map.get(("%s/%s" % (name, cn)).lower())
                    if sub == IGNORE:
                        ignored.append("%s/%s" % (name, cn))
                        continue
                    voices.append({"key": (t["index"], p, cn), "label": "%s/%s" % (name, cn),
                                   "track": t["index"], "pitch": p,
                                   "role": sub or keyword_role(cn) or UNMAPPED})
    return voices, ignored


def role_order(roles):
    known = [r for r in ROLE_ORDER if r in roles]
    custom = [r for r in roles if r not in ROLE_ORDER and r != UNMAPPED]
    return known + custom + ([UNMAPPED] if UNMAPPED in roles else [])


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

def _bars_text(x):
    return ("%g" % round(x, 3))


def read_arrangement(first, last, role_map=None, timeout_s=None):
    """The table for scenes first..last (1-based, inclusive), as plain data (see format_table)."""
    role_map = _lower_map(role_map)
    with Live() as live:
        data = read_set(live, first, last, timeout_s)
    for c in data["clips"].values():
        name = (c["name"] or "").lower()
        c["silent"] = bool(SILENT_CLIP.search(name)) and role_map.get(name) != PLAY
    voices, ignored = build_voices(data, role_map)
    roles = role_order({v["role"] for v in voices})

    # prev: role -> {voice key: (record, scene, non-looping, clip beats)} of the step before
    rows, prev, started, long_loops = [], {}, False, {}
    for s, scene_name in zip(data["scene_ids"], data["scene_names"]):
        here = {t: c for (t, sc), c in data["clips"].items()
                if sc == s and not c["muted"] and not c["silent"]
                and not data["tracks"][t]["muted"]}
        beats = scene_beats(here.values())
        bars = int(math.ceil(beats / BEATS_PER_BAR - EPS)) if beats > EPS else 0
        horizon = bars * BEATS_PER_BAR
        n_windows = int(math.ceil(bars / WINDOW_BARS))
        row = {"scene": s + 1, "name": scene_name or "", "bars": bars, "windows": n_windows,
               "cells": {}}
        # per voice: its sounding intervals in this scene
        played = {}
        for v in voices:
            c = here.get(v["track"])
            if c is None:
                continue
            if c["is_midi_clip"]:
                notes = [n for n in c["notes"] if v["pitch"] is None or n["pitch"] == v["pitch"]]
                played[v["key"]] = (notes, sounding(c, notes, horizon))
            else:
                played[v["key"]] = None
        v_of = {v["key"]: v for v in voices}
        for v in voices:                    # loops past 8 bars: a step is a slice of the loop
            c = here.get(v["track"])
            if (c is not None and c["is_midi_clip"] and c["looping"] and bars > WINDOW_BARS
                    and c["loop_end"] - c["loop_start"] > WINDOW_BEATS + EPS):
                long_loops.setdefault(data["tracks"][v["track"]]["name"], set()).add(s + 1)
        for role in roles:
            mine = [v for v in voices if v["role"] == role and v["track"] in here
                    and (v["pitch"] is None or v["key"] in played)]
            clip_info, cells, voice_plays = [], [], set()
            for w in range(n_windows):
                w0, w1 = w * WINDOW_BEATS, min((w + 1) * WINDOW_BEATS, horizon)
                recs = {}
                for v in mine:
                    c = here[v["track"]]
                    rec = (midi_record(c, played[v["key"]][0], played[v["key"]][1], w0, w1)
                           if c["is_midi_clip"] else audio_record(c, horizon, w0, w1))
                    if rec is not None:
                        recs[v["key"]] = rec
                        voice_plays.add(v["track"])
                playing = bool(recs)
                sig = (_digest(sorted((repr(k), repr(_canon(r))) for k, r in recs.items()))
                       if playing else None)
                p_recs = prev.get(role, {})
                if not started and w == 0:
                    change = None
                elif playing:
                    if not p_recs:
                        change = "enter"
                    else:
                        # What is playing now can be new; a voice that stopped is a removal,
                        # unless it is a non-looping clip whose length ended inside this scene
                        # (a one-shot that ran out).
                        moved = any(k not in p_recs or not same_content(p_recs[k][0], recs[k])
                                    for k in recs)
                        removed = any(k not in recs and not (
                            p_recs[k][1] == s and p_recs[k][2] and p_recs[k][3] <= w0 + EPS)
                            for k in p_recs)
                        change = "vary" if moved else "thin" if removed else "same"
                else:
                    change = "leave" if p_recs else None
                state = ("silent" if not playing
                         else "changed" if change in ("enter", "vary") else "playing")
                cells.append({"bars": "%d-%d" % (w * WINDOW_BARS + 1,
                                                 min((w + 1) * WINDOW_BARS, bars)),
                              "state": state, "change": change, "sig": sig})
                prev[role] = {k: (r, s, not here[v_of[k]["track"]]["looping"],
                                  extent_beats(here[v_of[k]["track"]]))
                              for k, r in recs.items()}
            seen = set()
            for v in mine:
                if v["track"] in seen:
                    continue
                seen.add(v["track"])
                c = here[v["track"]]
                clip_info.append({"track": data["tracks"][v["track"]]["name"], "clip": c["name"],
                                  "looping": bool(c["looping"]),
                                  "one_shot": not c["looping"],
                                  "loop_bars": round(extent_beats(c) / BEATS_PER_BAR, 4),
                                  "playing": v["track"] in voice_plays})
            loops = sorted({ci["loop_bars"] for ci in clip_info if ci["playing"] and ci["looping"]})
            row["cells"][role] = {"windows": cells, "loop_bars": loops,
                                  "one_shot": any(ci["playing"] and ci["one_shot"]
                                                  for ci in clip_info),
                                  "clips": clip_info}
        started = started or n_windows > 0
        rows.append(row)

    unmapped = sorted({v["label"] for v in voices if v["role"] == UNMAPPED})
    warnings = list(data["warnings"])
    for t in sorted({t for (t, _s) in data["clips"]}):
        if data["tracks"][t]["muted"]:
            warnings.append("muted track, counted as silent: %r (%s)"
                            % (data["tracks"][t]["name"], data["tracks"][t]["muted_why"]))
    for name, scenes in sorted(long_loops.items()):
        warnings.append("loop longer than 8 bars on %r in scenes %s: its 8-bar steps are different "
                        "slices of the loop, so a step can read varied on a pure repeat"
                        % (name, ", ".join(str(x) for x in sorted(scenes))))
    carriers = {}
    for (t, _s), c in data["clips"].items():
        if c["silent"]:
            carriers[data["tracks"][t]["name"]] = carriers.get(data["tracks"][t]["name"], 0) + 1
    for name, n in sorted(carriers.items()):
        warnings.append("silent-carrier clips counted as silent: %r (%d clips)" % (name, n))
    for (t, s), c in sorted(data["clips"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if c["muted"] and not data["tracks"][t]["muted"]:
            warnings.append("muted clip, counted as silent: track %r scene %d"
                            % (data["tracks"][t]["name"], s + 1))
    return {"set": data["set"], "scenes": [first, last], "window_bars": WINDOW_BARS,
            "roles": roles, "rows": rows, "unmapped": unmapped, "ignored": sorted(set(ignored)),
            "warnings": warnings, "notes": NOTES}


# --------------------------------------------------------------------------
# structure metrics: from the table, no Live
# --------------------------------------------------------------------------

PITCHED = ("bass", "lead", "arp", "pad", "drone", "pluck", "synth")   # "pitched parts"
HOOK_ROLE = "lead"            # the hook is the lead line, whatever track plays it (LD, LEAD, 27-Diva)

# The base section labels, from the start of a name: "Drop 1A · impact" is Drop 1, part A; "Outro B"
# is Outro, part B. `sections.py` names reference sections with the same labels (Intro, Main, Build,
# Drop, Breakdown, Outro, numbered when repeated: "Drop 1 (bass in) · 64"), and `Break` from its
# loudness method, which is read as Breakdown. Its other loudness labels ("Quiet", "Mid" and the
# fallback "C full") say nothing about form: they stay unlabelled and are listed, never guessed.
SECTION = re.compile(
    r"\s*(Intro|Main|Build|Drop|Breakdown|Break|Outro)\b\s*(\d+)?\s*((?-i:[A-Z])\b)?", re.I)


def section_of(scene_name):
    """(label, part) from a scene or section name by the base labels, or None:
    "Drop 1A" -> ("Drop 1", "A"), "Break 2" -> ("Breakdown 2", "")."""
    m = SECTION.match(scene_name or "")
    if not m:
        return None
    kind = m.group(1).capitalize()
    kind = "Breakdown" if kind == "Break" else kind
    return ("%s %s" % (kind, m.group(2)) if m.group(2) else kind), (m.group(3) or "")


def _steps(result):
    """The 8-bar steps of the range, in order, with the section each is in."""
    steps, bar = [], 1
    for row in result["rows"]:
        sec = section_of(row["name"])
        for w in range(row["windows"]):
            n = min(WINDOW_BARS, row["bars"] - w * WINDOW_BARS)
            cells = {role: c["windows"][w] for role, c in row["cells"].items()}
            steps.append({"scene": row["scene"], "name": row["name"], "bar": bar, "bars": n,
                          "in_scene": "%d-%d" % (w * WINDOW_BARS + 1, w * WINDOW_BARS + n),
                          "section": sec[0] if sec else None, "part": sec[1] if sec else "",
                          "cells": cells})
            bar += n
    return steps


def _playing(step, role):
    c = step["cells"].get(role)
    return bool(c) and c["state"] != "silent"


def _runs(steps, flags):
    """[(first step index, count)] of consecutive steps where flags[i] is true."""
    out, i = [], 0
    while i < len(steps):
        if flags[i]:
            j = i
            while j < len(steps) and flags[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _span(steps, i, n):
    return {"bar": steps[i]["bar"], "bars": sum(x["bars"] for x in steps[i:i + n]),
            "section": steps[i].get("section"), "scene": steps[i].get("scene")}


MIN_ENTRY_BARS = 4        # a stretch shorter than this is bleed or a hit, not a part coming in


def _stretch(steps, play, min_bars=MIN_ENTRY_BARS):
    """(first, last) step index of the first and the last stretch of at least `min_bars` bars in
    which a role plays, or None. A one-bar blip of bleed before the real entry is not an entry."""
    runs = [(i, i + k - 1) for i, k in _runs(steps, play)
            if sum(x["bars"] for x in steps[i:i + k]) >= min_bars]
    return (runs[0][0], runs[-1][1]) if runs else None


def _internal_gaps(steps, play):
    """(gaps, trailing bars): the stretches a role is out between its first and its last stretch of
    MIN_ENTRY_BARS or more, read bar by bar, and the bars it is out after the last. What comes
    before its first stretch is not "out"."""
    st = _stretch(steps, play)
    if st is None:
        return [], 0
    first, last = st
    gaps = [_span(steps, i, k) for i, k in _runs(steps, [not p for p in play])
            if first < i and i + k - 1 < last]
    return gaps, sum(x["bars"] for x in steps[last + 1:])


def groove_blocks(steps, min_gap=MIN_ENTRY_BARS, min_block=8):
    """The groove blocks of a track, the same way for a Set's 8-bar steps and a reference's 1-bar
    steps: a block is a maximal stretch in which kick and bass both play, an interruption shorter
    than `min_gap` bars (a fill) ignored, and it counts from `min_block` bars. Blocks are separated by
    4 or more bars without the groove. On 8-bar steps a gap of 4 to 7 bars cannot be seen and a
    fill is never bridged. Each block: its first bar, its length, `bass_lead` (the bars the bass
    plays, kick out, just before the kick joins; 1 bar or less reads as 0, section snapping against
    unsnapped activity), and for the later ones `gap_before` and `kick_out_before` (bars of the gap in
    which the kick is out). None if the kick or the bass is missing."""
    if not steps or any(r not in steps[0]["cells"] for r in ("kick", "bass")):
        return None
    kick = [_playing(x, "kick") for x in steps]
    bass = [_playing(x, "bass") for x in steps]
    on = [k and b for k, b in zip(kick, bass)]
    n = len(steps)
    for i, k in _runs(steps, [not v for v in on]):
        if 0 < i and i + k < n and sum(x["bars"] for x in steps[i:i + k]) < min_gap:
            on[i:i + k] = [True] * k
    runs = [(i, k) for i, k in _runs(steps, on) if sum(x["bars"] for x in steps[i:i + k]) >= min_block]
    out = []
    for idx, (i, k) in enumerate(runs):
        lead, j = 0, i - 1
        while j >= 0 and bass[j] and not kick[j]:
            lead += steps[j]["bars"]
            j -= 1
        rec = {"bar": steps[i]["bar"], "bars": sum(x["bars"] for x in steps[i:i + k]),
               "bass_lead": 0 if lead <= 1 else lead, "bass_lead_raw": lead}
        if idx:
            end = runs[idx - 1][0] + runs[idx - 1][1]
            rec["gap_before"] = sum(x["bars"] for x in steps[end:i])
            rec["kick_out_before"] = sum(x["bars"] for x, kk in zip(steps[end:i], kick[end:i]) if not kk)
        out.append(rec)
    return out


def metrics_from_steps(steps, sections, step_bars=WINDOW_BARS):
    """The structure of a track from its steps and sections, with nothing in it that is Live's.

    `steps`: in order, each {"bar": first bar (1-based), "bars": length, "cells": {role:
    {"state": "silent" or not, "change": enter / vary / same / leave / thin / None, "sig": any}}},
    optionally "scene", "name", "in_scene" (kept on the records that name a step). `change` may be
    left out of every cell, then no step is flagged. `sections`: {"label": a base label or None,
    "bar", "bars"}, optionally "name"/"scenes". `step_bars` is the resolution of `steps`: every bar
    this reports (entry, kick-out, hat gaps) is a step boundary, so a reference's per-bar activity
    gives bars and a Set's 8-bar steps give multiples of 8 plus 1. Roles the steps lack are None.
    An entry (bass, kick, hats) is the first stretch of MIN_ENTRY_BARS or more, so a bar of bleed
    before the real entry is not one; gaps are read bar by bar between the first and last such
    stretch.
    """
    n = len(steps)
    roles = {r for x in steps for r in x["cells"]}
    out = {"bars": sum(x["bars"] for x in steps), "steps": n, "step_bars": step_bars}

    secs = []
    for sec in sections:
        mine = [x for x in steps if sec["bar"] <= x["bar"] < sec["bar"] + sec["bars"]]
        counts = [sum(1 for r in PITCHED if _playing(x, r)) for x in mine]
        secs.append(dict(sec, pitched_parts={"max": max(counts or [0]), "per_step": counts}))
    out["sections"] = secs
    out["unlabelled"] = [{"bar": x["bar"], "bars": x["bars"], "name": x.get("name", "")}
                         for x in sections if x["label"] is None]

    # what each step adds, against the step before; the first step has no before
    changes, flags = [], []
    if any("change" in c for x in steps for c in x["cells"].values()):
        for i, x in enumerate(steps):
            if i == 0:
                continue
            kinds = {k: sorted(r for r, c in x["cells"].items() if c.get("change") == k)
                     for k in ("enter", "vary", "leave", "thin")}
            flag = (None if kinds["enter"] or kinds["vary"]
                    else "only a removal" if kinds["leave"] or kinds["thin"] else "nothing new")
            rec = {"scene": x.get("scene"), "name": x.get("name", ""), "bar": x["bar"],
                   "bars": x["bars"], "in_scene": x.get("in_scene"), "flag": flag, **kinds}
            changes.append(rec)
            if flag:
                flags.append(rec)
    out["changes"], out["flags"] = changes, flags
    out["changes_available"] = any("change" in c for x in steps for c in x["cells"].values())
    out["first_bar"] = steps[0]["bar"] if steps else None
    out["blocks"] = groove_blocks(steps)

    def playing(role):
        return [_playing(x, role) for x in steps]

    # hook: how long the lead runs, and whether it varies while it does
    if HOOK_ROLE in roles and n:
        play = playing(HOOK_ROLE)
        runs = _runs(steps, play)
        sigs = [x["cells"][HOOK_ROLE].get("sig") for x, p in zip(steps, play) if p]
        out["hook"] = {
            "role": HOOK_ROLE, "steps_playing": sum(play),
            "bars_playing": sum(x["bars"] for x, p in zip(steps, play) if p),
            "first_bar": next((x["bar"] for x, p in zip(steps, play) if p), None),
            "longest_run": max([_span(steps, i, k) for i, k in runs],
                               key=lambda r: r["bars"], default=None),
            "runs": [_span(steps, i, k) for i, k in runs],
            "varies": sum(1 for x, p in zip(steps[1:], play[1:])
                          if p and x["cells"][HOOK_ROLE].get("change") == "vary"),
            "distinct_variants": len(set(sigs))}
    else:
        out["hook"] = None

    # bass: the bar it enters, and whether that is before the first drop
    if "bass" in roles and n:
        play = playing("bass")
        st = _stretch(steps, play)
        i = st[0] if st else None
        raw = play.index(True) if True in play else None
        drop = next((x["bar"] for x in sections if (x["label"] or "").startswith("Drop")), None)
        out["bass"] = {"entry_bar": steps[i]["bar"] if i is not None else None,
                       "entry_bar_raw": steps[raw]["bar"] if raw is not None else None,
                       "entry_scene": steps[i].get("scene") if i is not None else None,
                       "first_drop_bar": drop,
                       "before_first_drop": (None if i is None or drop is None
                                             else steps[i]["bar"] < drop)}
    else:
        out["bass"] = None

    # kick: first bar, and every stretch it is out between two stretches it plays
    if "kick" in roles and n:
        play = playing("kick")
        gaps, trailing = _internal_gaps(steps, play)
        st = _stretch(steps, play)
        if st:
            first, last = st
            out["kick"] = {"first_bar": steps[first]["bar"], "out": gaps,
                           "longest_out_bars": max([g["bars"] for g in gaps], default=0),
                           "last_bar": steps[last]["bar"] + steps[last]["bars"] - 1,
                           "trailing_out_bars": trailing}
        else:
            out["kick"] = {"first_bar": None, "out": [], "longest_out_bars": 0, "last_bar": None,
                           "trailing_out_bars": 0}
    else:
        out["kick"] = None

    # Drop 2 against Drop 1: every section that carries the label, summed
    d = {}
    for x in sections:
        if x["label"] in ("Drop 1", "Drop 2"):
            d[x["label"]] = d.get(x["label"], 0) + x["bars"]
    out["drop_ratio"] = ({"drop1_bars": d["Drop 1"], "drop2_bars": d["Drop 2"],
                          "ratio": round(d["Drop 2"] / d["Drop 1"], 3)}
                         if d.get("Drop 1") and "Drop 2" in d else None)

    # hats: do they stay on between their first and last stretch (the outro may drop them)
    if "hats" in roles and n:
        play = playing("hats")
        gaps, trailing = _internal_gaps(steps, play)
        st = _stretch(steps, play)
        out["hats"] = {"first_bar": steps[st[0]]["bar"] if st else None,
                       "steps_playing": sum(play), "of_steps": n,
                       "constant": st is not None and not gaps, "gaps": gaps,
                       "trailing_out_bars": trailing}
    else:
        out["hats"] = None
    return out


def structure_metrics(result):
    """Structure of the range read from Live: sections from the scene names, 8-bar steps from the
    table, the numbers from `metrics_from_steps`. Bars count from the range's first scene (bar 1)."""
    steps = _steps(result)
    sections, unlabelled = [], []
    bar = 1
    for row in result["rows"]:
        sec = section_of(row["name"])
        if sec is None:
            unlabelled.append({"scene": row["scene"], "name": row["name"]})
        label = sec[0] if sec else None
        if sections and label is not None and sections[-1]["label"] == label:
            sections[-1]["scenes"][1] = row["scene"]
            sections[-1]["bars"] += row["bars"]
        else:
            sections.append({"label": label, "scenes": [row["scene"], row["scene"]],
                             "bars": row["bars"], "bar": bar, "name": row["name"]})
        bar += row["bars"]
    out = metrics_from_steps(steps, sections, WINDOW_BARS)
    out["unlabelled_scenes"] = unlabelled
    return out


# --------------------------------------------------------------------------
# compare: the Set against reference studies
# --------------------------------------------------------------------------

GAP_BARS = 4          # a kick or hats gap shorter than this is a fill, counted apart from the gaps
SHORT_BLOCK = 16      # a groove block shorter than this moves the block numbers: flagged
EARLY_BARS = 8        # a groove already on in the first bars: the first block is not an arrival
DUPLICATE_AGREEMENT = 0.9

# The main table is made of groove blocks (kick and bass together), measured the same way on the
# Set and on a reference from activity. Labels are a second, separate table.
BLOCK_COLUMNS = (("n_blocks", "blocks"), ("bars_before_block1", "bars before block 1"),
                 ("block1_bars", "block 1"), ("block2_bars", "block 2"),
                 ("block_ratio", "block 2 / block 1"),
                 ("major1_bars", "1st block of 16+ bars"), ("major2_bars", "2nd block of 16+ bars"),
                 ("major_ratio", "2nd / 1st of 16+ bars"),
                 ("kick_out_before_block2", "kick out before block 2"),
                 ("bass_lead1", "bass lead at block 1"), ("bass_lead2", "bass lead at block 2"))
GAP_COLUMNS = (("longest_kick_out_bars", "longest kick-out"), ("kick_gaps", "kick gaps of 4+ bars"),
               ("kick_fills", "kick fills (under 4)"), ("hats_gaps", "hats gaps of 4+ bars"),
               ("hats_fills", "hats fills (under 4)"))
NUMERIC = [k for k, _ in BLOCK_COLUMNS + GAP_COLUMNS]


def profile_of(m):
    """The numbers of one metric block (`metrics_from_steps`), measured the same way on a Set range
    and on a reference, none an absolute bar number: the groove blocks (kick and bass together),
    and the kick and hats gaps. None where it does not exist (no kick or bass stem, one block only).
    `flags` are what moves the block numbers."""
    blocks = m.get("blocks")
    b1 = blocks[0] if blocks else None
    b2 = blocks[1] if blocks and len(blocks) > 1 else None
    first = m.get("first_bar") or 1
    big = [x for x in blocks or [] if x["bars"] >= SHORT_BLOCK]    # the blocks that are not a teaser
    out = {"n_blocks": len(blocks) if blocks is not None else None,
           "bars_before_block1": b1["bar"] - first if b1 else None,
           "block1_bars": b1["bars"] if b1 else None, "block2_bars": b2["bars"] if b2 else None,
           "block_ratio": round(b2["bars"] / b1["bars"], 3) if b1 and b2 else None,
           "major1_bars": big[0]["bars"] if big else None,
           "major2_bars": big[1]["bars"] if len(big) > 1 else None,
           "major_ratio": round(big[1]["bars"] / big[0]["bars"], 3) if len(big) > 1 else None,
           "kick_out_before_block2": b2["kick_out_before"] if b2 else None,
           "bass_lead1": b1["bass_lead"] if b1 else None,
           "bass_lead2": b2["bass_lead"] if b2 else None}
    flags = []
    if blocks is None:
        flags.append("no kick or bass stem")
    elif not blocks:
        flags.append("no groove block")
    else:
        if b1["bar"] - first < EARLY_BARS:
            flags.append("groove already on in the first %d bars" % EARLY_BARS)
        if any(x["bars"] < SHORT_BLOCK for x in blocks[:2]):
            flags.append("a block under %d bars" % SHORT_BLOCK)
        if len(blocks) > 2:
            flags.append("block 3 and later not compared (%d blocks)" % len(blocks))
    out["flags"] = flags
    kick, hats = m["kick"], m["hats"]
    if kick and kick["first_bar"]:
        gaps = kick["out"]
        longest = max(gaps, key=lambda g: g["bars"], default=None)
        out.update(longest_kick_out_bars=longest["bars"] if longest else 0,
                   longest_kick_out_in=(longest["section"] or "no label") if longest else None,
                   kick_gaps=sum(1 for g in gaps if g["bars"] >= GAP_BARS),
                   kick_fills=sum(1 for g in gaps if g["bars"] < GAP_BARS))
    else:
        out.update(longest_kick_out_bars=None, longest_kick_out_in=None, kick_gaps=None, kick_fills=None)
    if hats and hats["first_bar"]:
        out.update(hats_gaps=sum(1 for g in hats["gaps"] if g["bars"] >= GAP_BARS),
                   hats_fills=sum(1 for g in hats["gaps"] if g["bars"] < GAP_BARS))
    else:
        out.update(hats_gaps=None, hats_fills=None)
    return out


def labelled_drops(m):
    """The drops as the sections are labelled (the Set's scene names, a reference's rules): each
    `Drop N` section with the `Main` sections that follow it, which the rules use for the rest of the
    same drop. Not comparable across the two labelling systems."""
    secs, out = m["sections"], []
    for i, x in enumerate(secs):
        if (x["label"] or "").startswith("Drop"):
            mains, j = 0, i + 1
            while j < len(secs) and (secs[j]["label"] or "").startswith("Main"):
                mains += secs[j]["bars"]
                j += 1
            out.append({"label": x["label"], "bar": x["bar"], "bars": x["bars"], "mains_after": mains})
    return out


def resolve_refs(names):
    """([Study], [{"name", "root", "reason"}]) for `names` (study names, or "all": every analysed
    study under the roots). A study without the activity layer, or one that cannot be read, is
    left out with its reason, never silently."""
    import study as study_reader
    if not names:
        raise ValueError("give at least one reference (a study name, or all)")
    if any(n.strip().lower() == "all" for n in names):
        found = [r for r in study_reader.find_studies() if r.analysed]
        entries = [(str(r.root), "%s/%s" % (r.artist, r.title), r.source) for r in found]
    else:
        entries = [(n, n, "") for n in names]
    studies, left, seen = [], [], set()
    for target, label, root in entries:
        try:
            st = study_reader.load_study(target)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
            left.append({"name": label, "root": root, "reason": "cannot be read: %s" % e})
            continue
        key = (st.source, st.artist, st.title)
        if key in seen:
            continue
        seen.add(key)
        if st.activity() is None:
            left.append({"name": "%s/%s" % (st.artist, st.title), "root": st.source,
                         "reason": st.layers["activity"].missing
                         or "its bar_activity could not be read"})
        else:
            studies.append(st)
    return studies, left


def _agreement(a, b):
    """Share of bars on which two studies' activity agrees, over the main roles both have (percussion
    and vocals differ between separations of one track and say little)."""
    roles = [r for r in ("kick", "bass", "hats", "synths") if r in a and r in b]
    if len(roles) < 2:
        return 0.0
    n = min(len(next(iter(a.values()))), len(next(iter(b.values()))))
    same = sum(1 for r in roles for i in range(n) if a[r][i] == b[r][i])
    return same / float(n * len(roles))


def drop_duplicates(studies):
    """(kept, [{"name", "root", "reason"}]): the same audio counted twice (an Extended Mix with the
    same length and activity, true stems and separated stems of one track) weighs double in a
    range and a median. Two studies are one track when their lengths are within a bar and their
    activity agrees on 90% of bars; the local one (true stems) is kept, else the shorter title."""
    acts = {id(s): s.activity() for s in studies}
    order = sorted(studies, key=lambda s: (s.source != "local", len(s.title), s.artist.casefold(), s.title.casefold()))
    kept, dropped = [], []
    for st in order:
        a = acts[id(st)]
        twin = next((k for k in kept if abs(len(next(iter(acts[id(k)].values()))) -
                                           len(next(iter(a.values())))) <= 1
                     and _agreement(acts[id(k)], a) >= DUPLICATE_AGREEMENT), None)
        if twin is None:
            kept.append(st)
        else:
            dropped.append({"name": "%s/%s" % (st.artist, st.title), "root": st.source,
                            "reason": "duplicate of %s/%s [%s]: same length and %d%%+ of the same "
                                      "activity, counted once" % (twin.artist, twin.title, twin.source,
                                                                  int(DUPLICATE_AGREEMENT * 100))})
    return kept, dropped


def _stats(profiles):
    """{key: {min, max, median, n}} over the references that have the number."""
    import statistics
    out = {}
    for k in NUMERIC:
        vals = [p[k] for p in profiles if p.get(k) is not None]
        out[k] = ({"min": min(vals), "max": max(vals), "median": statistics.median(vals),
                   "n": len(vals)} if vals else {"min": None, "max": None, "median": None, "n": 0})
    return out


def _g(x):
    return "n/a" if x is None else "%g" % x


def readings(sp, stats, refs):
    """Where the Set sits against the references, in plain numbers, from the block table only. The
    headline is the length ratio of the first two blocks of 16+ bars, which a short teaser block
    does not distort; medians are given over all references and over the unflagged ones, except
    where the Set is itself flagged (its raw block numbers are not alike to the unflagged group)."""
    N = len(refs)
    all_, clean = stats["all"], stats["unflagged"]
    out = []
    flagged = bool(sp["flags"])

    def span(k, both=True):
        a, c = all_[k], clean[k]
        return "median %s over %d of %d references (range %s to %s)%s" % (
            _g(a["median"]), a["n"], N, _g(a["min"]), _g(a["max"]),
            "; %s over the %d unflagged" % (_g(c["median"]), c["n"]) if both and c["n"] != a["n"] else "")

    def count(key, pred):
        vals = [r["profile"][key] for r in refs if r["profile"][key] is not None]
        return sum(1 for v in vals if pred(v)), len(vals)

    if sp["major_ratio"] is not None and all_["major_ratio"]["n"]:
        shorter, n = count("major_ratio", lambda v: v < 1)
        longer, _ = count("major_ratio", lambda v: v > 1)
        out.append("Length ratio (the first two blocks of 16+ bars, so a short teaser block does not count): "
                   "%d / %d bars = %s in the Set; %s; the 2nd is shorter than the 1st in %d of %d, longer in %d." % (
                       sp["major2_bars"], sp["major1_bars"], _g(sp["major_ratio"]), span("major_ratio"),
                       shorter, n, longer))
    if flagged:
        out.append("The Set is itself flagged (%s): its raw block 1 and block 2 / block 1 below are not alike "
                   "to the unflagged references, so they are given over all references only." % "; ".join(sp["flags"]))
    if sp["bars_before_block1"] is not None and all_["bars_before_block1"]["n"]:
        early, n = count("bars_before_block1", lambda v: v < EARLY_BARS)
        out.append("Groove start: kick and bass first play together %d bars in, in the Set; %s; in %d of %d "
                   "references they are together within the first %d bars." % (
                       sp["bars_before_block1"], span("bars_before_block1"), early, n, EARLY_BARS))
    if sp["block1_bars"] is not None and all_["block1_bars"]["n"]:
        out.append("Block 1: %d bars in the Set; %s." % (sp["block1_bars"], span("block1_bars", not flagged)))
    if sp["n_blocks"] is not None and all_["n_blocks"]["n"]:
        out.append("Blocks: %d in the Set; %s." % (sp["n_blocks"], span("n_blocks", False)))
    if sp["block_ratio"] is not None and all_["block_ratio"]["n"]:
        shorter, n = count("block_ratio", lambda v: v < 1)
        longer, _ = count("block_ratio", lambda v: v > 1)
        out.append("Block 2 / block 1: %s in the Set (%d / %d bars); %s; block 2 is shorter than block 1 in "
                   "%d of %d, longer in %d." % (_g(sp["block_ratio"]), sp["block2_bars"], sp["block1_bars"],
                                                span("block_ratio", not flagged), shorter, n, longer))
    if sp["bass_lead1"] is not None and all_["bass_lead1"]["n"]:
        zero1, n1 = count("bass_lead1", lambda v: v == 0)
        text = ("Bass lead (bars the bass plays before the kick joins, 1 bar or less read as 0): in the Set "
                "%d at block 1" % sp["bass_lead1"])
        text += ", %d at block 2" % sp["bass_lead2"] if sp["bass_lead2"] is not None else ""
        text += "; references at block 1: no lead in %d of %d, %s" % (zero1, n1, span("bass_lead1"))
        if sp["bass_lead2"] is not None and all_["bass_lead2"]["n"]:
            zero2, n2 = count("bass_lead2", lambda v: v == 0)
            text += "; at block 2: no lead in %d of %d, %s" % (zero2, n2, span("bass_lead2"))
        out.append(text + ".")
    if sp["kick_out_before_block2"] is not None and all_["kick_out_before_block2"]["n"]:
        out.append("Kick out before block 2: %d bars in the Set; %s." % (
            sp["kick_out_before_block2"], span("kick_out_before_block2")))
    if sp["kick_gaps"] is not None and all_["kick_gaps"]["n"]:
        out.append("Kick gaps of 4+ bars: %d in the Set (whole 8-bar steps, so it cannot show fills or a "
                   "4-7 bar gap); %s; the references also have %s fills of under 4 bars (median)." % (
                       sp["kick_gaps"], span("kick_gaps"), _g(all_["kick_fills"]["median"])))
    return out


def compare_report(set_result, studies, left_out=()):
    """The Set's metrics against each reference's: one report, as plain data. `set_result` is a read
    (`read_arrangement`, or the JSON of `roadie arrangement read --json`); its stored metrics are
    never trusted, they are recomputed here with the current code."""
    try:
        sm = structure_metrics(set_result)
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        raise ValueError("the saved read is malformed (%s: %s): make one with `roadie arrangement "
                         "read --scenes A-B --json`" % (type(e).__name__, e))
    sp = profile_of(sm)
    studies, dups = drop_duplicates(list(studies))
    refs, chance = [], 0
    for st in studies:
        r = reference_metrics(st)
        prof = profile_of(r["metrics"])
        notes = list(prof["flags"])
        if st.layers["grid"].source is None:
            notes.append("no grid")
        if any("chance-level" in c for c in st.layers["grid"].caveats):
            chance += 1
        drops = labelled_drops(r["metrics"])
        refs.append({"name": "%s - %s" % (st.artist, st.title), "root": st.source, "artist": st.artist,
                     "title": st.title, "profile": prof, "notes": notes,
                     "drops_as_labelled": drops,
                     "sections_as_labelled": [x["name"] for x in r["metrics"]["sections"]]})
    refs.sort(key=lambda r: (r["artist"].casefold(), r["title"].casefold(), r["root"]))
    stats = {"all": _stats([r["profile"] for r in refs]),
             "unflagged": _stats([r["profile"] for r in refs if not r["profile"]["flags"]])}
    pitched = [{"label": x["label"] or "(no label)", "bar": x["bar"], "bars": x["bars"],
                "roles_at_once": x["pitched_parts"]["max"]} for x in sm["sections"]]
    return {
        "kind": "arrangement_compare", "version": 2,
        "set": {"name": set_result.get("set") or "", "scenes": set_result.get("scenes"),
                "profile": sp, "blocks": sm.get("blocks"), "drops_as_labelled": labelled_drops(sm)},
        "references": refs, "left_out": list(left_out) + dups,
        "chance_level_downbeat": chance,
        "range": {w: {k: {"min": v["min"], "max": v["max"], "n": v["n"]} for k, v in st_.items()}
                  for w, st_ in stats.items()},
        "median": {w: {k: {"value": v["median"], "n": v["n"]} for k, v in st_.items()}
                   for w, st_ in stats.items()},
        "readings": readings(sp, stats, refs),
        "set_only": {"flags": sm["flags"], "hook": sm["hook"], "pitched_parts": pitched,
                     "changes_available": sm.get("changes_available", True)},
        "notes": [
            "A groove block is a maximal stretch in which kick and bass both play, an interruption "
            "shorter than 4 bars ignored, from 8 bars; blocks are separated by 4 or more bars without it. "
            "It is computed the same way from the Set's role table and from a reference's activity. On the "
            "Set the steps are 8 bars, so a gap of 4 to 7 bars cannot be seen there and a fill is never "
            "bridged.",
            "Bass lead is the bars the bass plays, kick out, just before the kick joins a block; 1 bar or "
            "less reads as 0 (a section boundary snapped to the phrase grid against unsnapped activity).",
            "Positions are bars before the first block, never absolute bar numbers: the Set counts from the "
            "first scene read, a reference from the start of its file (a lead-in included).",
            "Flagged rows: the groove is already on in the first %d bars, a block under %d bars, or a third "
            "or later block that is not compared. Ranges and medians are given over all references and over "
            "the unflagged ones; n is in each cell." % (EARLY_BARS, SHORT_BLOCK),
            "Kick and hats gaps are read bar by bar on a reference (fills counted apart) and in 8-bar steps "
            "on the Set.",
            "The labelled table is not comparable across its two sources: the Set's labels are the scene "
            "names the user wrote (a drop is every scene of it), a reference's are rules over which stems "
            "play (a drop is the first segment labelled Drop; the rest of it is often labelled Main, shown "
            "after it). No ratio is computed from it.",
            "Pitched parts is not compared: a reference counts stems (at most 2), the Set counts roles.",
            "Automation in the Set's clips is not read.",
            "A study whose sections.json predates bar_activity and could not be re-made without changing "
            "its other fields is left out, not re-run."],
    }


def _drops_text(drops):
    return "; ".join("%s %d%s" % (d["label"], d["bars"], " + Main %d" % d["mains_after"] if d["mains_after"] else "")
                     for d in drops) or "no Drop label"


def format_compare(rep):
    """The report as Markdown: references in rows, Set first, range and median last."""
    sp, refs, sc = rep["set"]["profile"], rep["references"], rep["set"]["scenes"]
    lines = ["# Arrangement compare: %s%s against %d reference%s" % (
        rep["set"]["name"] or "the Set", ", scenes %d-%d" % tuple(sc) if sc else "", len(refs),
        "" if len(refs) == 1 else "s"), "",
        "> Reference sections are labelled by rules over the separated stems, not checked by ear; the "
        "groove blocks are a reading of separated audio, not of notes. %d of %d references have a "
        "chance-level downbeat confidence (bar lines and phrase snapping may be off; they are not "
        "excluded)." % (rep["chance_level_downbeat"], len(refs)), "",
        "## Readings", ""]
    lines += ["- " + r for r in rep["readings"]] + [""]
    stat_labels = (("Range, all", "range", "all"), ("Median, all", "median", "all"),
                   ("Range, unflagged", "range", "unflagged"), ("Median, unflagged", "median", "unflagged"))

    def stat_rows(cols, tail):
        out = []
        for label, kind, which in stat_labels:
            cells = []
            for k, _ in cols:
                v = rep[kind][which][k]
                if not v["n"] if kind == "range" else not v["n"]:
                    cells.append("n/a")
                elif kind == "range":
                    cells.append("%s to %s (n=%d)" % (_g(v["min"]), _g(v["max"]), v["n"]))
                else:
                    cells.append("%s (n=%d)" % (_g(v["value"]), v["n"]))
            out.append("| **%s** | | %s |%s" % (label, " | ".join(cells), tail))
        return out

    for title, cols, notes_col in (("Groove blocks (kick and bass together), bars", BLOCK_COLUMNS, True),
                                   ("Kick and hats gaps, bars", GAP_COLUMNS, False)):
        head = "| Source | Root | %s |%s" % (" | ".join(h for _, h in cols), " Notes |" if notes_col else "")
        lines += ["## " + title, "", head, "|" + "---|" * (2 + len(cols) + (1 if notes_col else 0))]
        lines.append("| **The Set** | live | %s |%s" % (" | ".join(_g(sp.get(k)) for k, _ in cols),
                                                       " %s |" % "; ".join(sp["flags"]) if notes_col else ""))
        for r in refs:
            lines.append("| %s | %s | %s |%s" % (r["name"], r["root"], " | ".join(_g(r["profile"].get(k)) for k, _ in cols),
                                                 " %s |" % "; ".join(r["notes"]) if notes_col else ""))
        lines += stat_rows(cols, " |" if notes_col else "")
        if not notes_col:
            lines += ["", "Where the longest kick-out falls: Set: %s%s." % (
                sp.get("longest_kick_out_in") or "n/a", "".join(
                    "; %s: %s" % (r["name"], r["profile"]["longest_kick_out_in"])
                    for r in refs if r["profile"].get("longest_kick_out_in")))]
        lines.append("")
    lines += ["## As labelled (not comparable across the two sources)", "",
              "The Set's labels are the scene names the user wrote; a reference's are rules over the stems, which "
              "split one drop into `Drop N` and the `Main` sections after it. Shown as `Drop N bars + Main bars`; "
              "no ratio is computed.", "",
              "| Source | Root | Drops as labelled |", "|---|---|---|",
              "| **The Set** | live | %s |" % _drops_text(rep["set"]["drops_as_labelled"])]
    lines += ["| %s | %s | %s |" % (r["name"], r["root"], _drops_text(r["drops_as_labelled"])) for r in refs]
    so = rep["set_only"]
    lines += ["", "## The Set only (a reference is audio, so none of this exists for it)", ""]
    if so["flags"]:
        for f in so["flags"]:
            lines.append("- %s, bars %s of the scene: %s" % ((f["name"] or "scene %s" % f["scene"])[:40],
                                                             f["in_scene"], f["flag"]))
    else:
        lines.append("- no 8-bar step adds nothing or only removes")
    h = so["hook"]
    lines.append("- hook (lead): " + ("not in this range" if not h else "never plays" if not h["first_bar"] else
                                     "runs %d bars from bar %d, varies at %d steps, %d variants" % (
                                         h["longest_run"]["bars"], h["longest_run"]["bar"], h["varies"],
                                         h["distinct_variants"])))
    lines.append("- pitched roles at once, by section: " + ", ".join(
        "%s %d" % (p["label"], p["roles_at_once"]) for p in so["pitched_parts"]))
    lines += ["", "## Left out", ""]
    lines += (["- %s [%s]: %s" % (x["name"], x["root"] or "-", x["reason"]) for x in rep["left_out"]]
              or ["- none"])
    lines += ["", "## Notes", ""] + ["- " + n for n in rep["notes"]]
    return "\n".join(lines) + "\n"


def build_compare(scenes=None, refs=("all",), roles=None, read=None, timeout_s=None):
    """The compare report. `read` is a saved read (a dict or the path of `roadie arrangement read
    --json`), recomputed with the current code, and then `roles` is refused (the roles were applied
    when it was made); else the scenes are read from Live (read-only), within `timeout_s` for the
    whole read, and `roles` is a role map (dict or JSON path)."""
    if isinstance(read, (str, os.PathLike)):
        with open(read) as f:
            try:
                read = json.load(f)
            except ValueError as e:
                raise ValueError("the saved read %s is not JSON: %s" % (read, e))
    if read is None:
        if not scenes:
            raise ValueError("give the scene range (A-B) or a saved read")
        first, last = parse_scenes(scenes)
        rm = load_role_map(roles) if isinstance(roles, (str, os.PathLike)) else roles
        read = read_arrangement(first, last, rm, timeout_s)
    else:
        if not isinstance(read, dict) or not isinstance(read.get("rows"), list):
            raise ValueError("the saved read is malformed: it has no rows; make one with `roadie "
                             "arrangement read --scenes A-B --json`")
        if roles:
            raise ValueError("roles apply when the Set is read from Live; a saved read was made with its "
                             "roles already (make a new one with `roadie arrangement read --roles ... --json`)")
        if scenes and read.get("scenes") and list(parse_scenes(scenes)) != list(read["scenes"]):
            raise ValueError("the saved read is of scenes %d-%d, not %s" % (read["scenes"][0], read["scenes"][1], scenes))
    studies, left = resolve_refs(list(refs))
    return compare_report(read, studies, left)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

MARK = {"enter": "+", "vary": "~", "same": "=", "leave": "-", "thin": "-", None: "."}


def _cell_text(cell):
    chars = ""
    for w in cell["windows"]:
        if w["state"] == "silent":
            chars += MARK[w["change"]]            # "-" the step it leaves, "." after
        else:
            chars += MARK[w["change"]] if w["change"] else "="
    extra = []
    if cell["loop_bars"]:
        extra.append("/".join(_bars_text(b) for b in cell["loop_bars"]))
    if cell["one_shot"]:
        extra.append("1x")
    return chars + (" " + " ".join(extra) if extra else "")


def format_table(result):
    roles = result["roles"]
    body = [[str(r["scene"]), str(r["bars"]), (r["name"] or "")[:26]]
            + [_cell_text(r["cells"][role]) if role in r["cells"] else "" for role in roles]
            for r in result["rows"]]
    head = ["scene", "bars", "name"] + roles
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = lambda cols: "  ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip()
    lines = ["%s, scenes %d-%d, one character per %d bars"
             % (result["set"] or "Set", result["scenes"][0], result["scenes"][1],
                result["window_bars"]),
             "+ enters  ~ varies  = same  - leaves (or one part of the role stops)  . silent;  then loop bars, 1x = plays once",
             "", fmt(head)] + [fmt(b) for b in body]
    if result["unmapped"]:
        lines += ["", "unmapped (no role; map them with --roles): " + ", ".join(result["unmapped"])]
    if result["ignored"]:
        lines += ["", "ignored by the role map: " + ", ".join(result["ignored"])]
    for w in result["warnings"]:
        lines.append("warning: " + w)
    lines += [""] + ["note: " + n for n in result["notes"]]
    return "\n".join(lines)


def format_metrics(m, unavailable=None, title=None, by_labels=True):
    """The metric block as text. `unavailable` says, per metric, why a source cannot give it
    ("changes", "hook"): printed in its place, never a silent zero or "none". `by_labels` False (a
    reference) keeps the labels as the section map only: no Drop 2 / Drop 1 and no bass-against-drop,
    which depend on how the sections were labelled; the groove blocks stand in their place."""
    unavailable = unavailable or {}
    unit = ("%d-bar steps" % m["step_bars"]) if m["step_bars"] != 1 else "1-bar steps"
    lines = [title or "Structure: %d bars in %d steps of %d (bars count from the first scene of "
             "the range)" % (m["bars"], m["steps"], m["step_bars"]), "", "sections"]
    for sec in m["sections"]:
        sc = sec.get("scenes")
        where = ("" if not sc else "scene %d" % sc[0] if sc[0] == sc[1] else "scenes %d-%d" % tuple(sc))
        lines.append("  %-12s %-13s bar %-3d %3d bars   pitched parts, most at once: %d"
                     % (sec["label"] or "(no label)", where, sec["bar"], sec["bars"],
                        sec["pitched_parts"]["max"]))
    for u in m.get("unlabelled_scenes", []):
        lines.append("  scene %d (%r) matches no section label" % (u["scene"], u["name"]))
    if not m.get("unlabelled_scenes"):
        for u in m["unlabelled"]:
            lines.append("  section at bar %d (%r) matches no base label: listed, not guessed"
                         % (u["bar"], u["name"]))
    lines += ["", "%s that add nothing or only remove" % ("8-bar steps" if m["step_bars"] == 8 else unit)]
    if not m["changes_available"]:
        lines.append("  not available: " + unavailable.get("changes", "the steps carry no change"))
    elif m["flags"]:
        for f in m["flags"]:
            lines.append("  %s, bars %s of the scene (bar %d): %s"
                         % ((f["name"] or "scene %s" % f["scene"])[:30], f["in_scene"], f["bar"],
                            f["flag"]))
    else:
        lines.append("  none")
    h = m["hook"]
    lines += ["", "hook (%s)" % HOOK_ROLE]
    if h and h["first_bar"]:
        lines.append("  in at bar %d, plays %d of %d steps (%d bars); longest run %d bars from bar %d"
                     % (h["first_bar"], h["steps_playing"], m["steps"], h["bars_playing"],
                        h["longest_run"]["bars"], h["longest_run"]["bar"]))
        lines.append("  varies at %d steps, %d distinct variants" % (h["varies"],
                                                                      h["distinct_variants"]))
    elif h:
        lines.append("  never plays")
    else:
        lines.append("  not available: " + unavailable.get(
            "hook", "no %s role in this range" % HOOK_ROLE))
    b = m["bass"]
    lines += ["", "bass: " + ("no bass role" if b is None else
                              "never plays" if b["entry_bar"] is None else
                              "in at bar %d%s%s" % (
                                  b["entry_bar"],
                                  " (scene %d)" % b["entry_scene"] if b["entry_scene"] else "",
                                  "" if b["before_first_drop"] is None or not by_labels else
                                  ", %s the first drop (bar %d)" % (
                                      "BEFORE" if b["before_first_drop"] else "not before",
                                      b["first_drop_bar"])))]
    k = m["kick"]
    if k is None:
        lines.append("kick: no kick role")
    elif k["first_bar"] is None:
        lines.append("kick: never plays")
    else:
        lines.append("kick: in at bar %d; out %s" % (k["first_bar"], "; ".join(
            "%d bars from bar %d (%s)" % (g["bars"], g["bar"], g["section"] or "no label")
            for g in k["out"]) or "never between two stretches it plays"))
    dr = m["drop_ratio"]
    if by_labels:
        lines.append("drops: " + ("Drop 2 %d bars / Drop 1 %d bars = %g" % (
            dr["drop2_bars"], dr["drop1_bars"], dr["ratio"]) if dr else "no Drop 1 and Drop 2"))
    else:
        lines.append("drops as labelled (rules; a drop is often Drop + the Main after it; no ratio): " + (
            "; ".join("%s %d%s" % (d["label"], d["bars"], " + Main %d" % d["mains_after"] if d["mains_after"] else "")
                      for d in labelled_drops(m)) or "no Drop label"))
    ht = m["hats"]
    lines.append("hats: " + ("no hats role" if ht is None else
                             "never play" if ht["first_bar"] is None else
                             "in at bar %d, play %d of %d steps, %s%s" % (
                                 ht["first_bar"], ht["steps_playing"], ht["of_steps"],
                                 "constant until the last one" if ht["constant"] else
                                 "out at " + ", ".join("bar %d (%d bars)" % (g["bar"], g["bars"])
                                                       for g in ht["gaps"]),
                                 "; out for the last %d bars" % ht["trailing_out_bars"]
                                 if ht["trailing_out_bars"] else "")))
    blocks = m.get("blocks")
    lines += ["", "groove blocks (kick and bass together; a fill under 4 bars ignored, from 8 bars; split by 4+ bars "
                  "without it%s)" % ("; 8-bar steps: a gap of 4 to 7 bars cannot be seen" if m["step_bars"] == 8 else "")]
    if blocks is None:
        lines.append("  not available: no kick or no bass role")
    elif not blocks:
        lines.append("  none")
    for x in blocks or []:
        lines.append("  bar %d, %d bars, bass lead %d%s" % (
            x["bar"], x["bars"], x["bass_lead"],
            ", kick out %d bars before it" % x["kick_out_before"] if "kick_out_before" in x else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# a reference: the same metrics from a study's per-bar stem activity
# --------------------------------------------------------------------------

# The study roles (study.STEM_ROLES) as the Set's roles: what a metric looks for by name.
STUDY_ROLE = {"kick": "kick", "bass": "bass", "hats": "hats", "snare": "snare", "perc": "perc",
              "synths": "synth", "vocals": "vox"}
REF_UNAVAILABLE = {
    "flags": "a reference is audio, not notes: nothing can be called 'varied', so no step is flagged "
             "as adding nothing or only removing",
    "changes": "a reference is audio, not notes: what each step adds, varies or removes is not read",
    "hook": "the stems have no lead part (the synths stem is leads and pads together), so the hook "
            "run and its variation are not read",
}
# A reference's numbers that look like the Set's but are not the same measure.
REF_NOT_COMPARABLE = {
    "pitched_parts": "counts stems (bass, synths) and is at most 2 for a reference; the Set's counts "
                     "pitched roles among 7: not comparable, never one row",
    "sections_vs_bars": "the section map (and the drop lengths, in the JSON only) comes from despeckled activity "
                        "(a stretch must last 4 bars); entries and kick/hats gaps are read bar by "
                        "bar (an entry is the first stretch of 4 bars or more). A one-bar fill can "
                        "make a kick-out 15 bars inside a 16-bar breakdown: that is the fill, not "
                        "a disagreement",
    "bar_numbers": "file bars from the audio's grid start (a lead-in shifts them); the Set counts "
                   "from its first scene: compare positions as bars before the first groove block, "
                   "never bare bar numbers",
}


def reference_metrics(study):
    """The structure metrics of a study (`study.Study`): one-bar steps from its `bar_activity` and
    its sections, through `metrics_from_steps(step_bars=1)`. Bars are the study's FILE bars (the
    Live bar is the file bar plus `bar_offset` when it is known). Needs sections.json with a
    bar_activity (`roadie sections <stems>`); raises ValueError without one."""
    act = study.activity()
    if act is None:
        raise ValueError("%s - %s has no bar_activity: %s" % (
            study.artist, study.title, study.layers["activity"].missing
            or "its sections.json carries one that could not be read"))
    first = study.activity_first_bar()
    n = len(next(iter(act.values())))
    sections = []
    for sec in study.sections():
        base = section_of(sec.label)
        sections.append({"label": base[0] if base else None, "bar": sec.bar, "bars": sec.bars,
                         "name": sec.label})

    def section_at(bar):
        return next((x["name"] for x in sections if x["bar"] <= bar < x["bar"] + x["bars"]), None)

    steps = [{"bar": first + i, "bars": 1, "section": section_at(first + i),
              "cells": {STUDY_ROLE.get(r, r): {"state": "playing" if flags[i] else "silent"}
                        for r, flags in act.items()}} for i in range(n)]
    caveats = []
    for name in ("sections", "grid", "activity"):
        layer = study.layers[name]
        caveats += layer.caveats
        if layer.source is None and name == "grid":
            caveats.append("no grid: alignment.json is missing, the bar lines are the tempo and "
                           "anchor sections.json was made with")
    caveats.append("bars are file bars" + (", Live bar = file bar + %d" % study.bar_offset
                                           if study.bar_offset is not None else
                                           "; no Live bar is known"))
    m = metrics_from_steps(steps, sections, 1)
    # What a reference cannot give is null with a reason, never an empty list that reads as "none".
    m["changes"] = m["flags"] = m["hook"] = None
    m["not_available"] = {k: REF_UNAVAILABLE[k] for k in ("flags", "changes", "hook")}
    return {"kind": "arrangement_reference", "version": 1,
            "study": {"artist": study.artist, "title": study.title, "root": study.source,
                      "bpm": study.bpm, "bar_offset": study.bar_offset},
            "caveats": caveats, "not_comparable": REF_NOT_COMPARABLE, "metrics": m}


def format_reference(ref):
    st = ref["study"]
    title = "Structure of %s - %s (reference, %s bpm), %d bars in 1-bar steps, file bars" % (
        st["artist"], st["title"], "%g" % st["bpm"] if st["bpm"] else "unknown",
        ref["metrics"]["bars"])
    return (format_metrics(ref["metrics"], ref["metrics"]["not_available"], title, by_labels=False)
            + "\n\n" + "\n".join("note: " + c for c in ref["caveats"])
            + "\n" + "\n".join("not comparable with the Set (%s): %s" % (k, v)
                                for k, v in ref["not_comparable"].items()))


def parse_scenes(text):
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", text or "")
    if not m:
        raise ValueError("--scenes expects A-B or A (1-based scene numbers), got %r" % text)
    first = int(m.group(1))
    last = int(m.group(2) or first)
    if first < 1 or last < first:
        raise ValueError("--scenes %r: need 1 <= A <= B" % text)
    return first, last


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roadie arrangement", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("read", help="read a scene range from the open Live Set (read-only)")
    rd.add_argument("--scenes", required=True, metavar="A-B", help="1-based scene range, e.g. 33-47")
    rd.add_argument("--roles", metavar="MAP.json", help="track name -> role map (JSON)")
    rd.add_argument("--json", action="store_true", help="JSON instead of the text table")
    rd.add_argument("--metrics", action="store_true",
                    help="add the structure metrics: sections, steps that add nothing, hook, bass, kick, drops, hats")
    rf = sub.add_parser("ref", help="the same metrics for a reference study, from its bar_activity")
    rf.add_argument("study", help="a study title under the studies roots, or a path (see `roadie study list`)")
    rf.add_argument("--json", action="store_true", help="JSON instead of the text block")
    cp = sub.add_parser("compare", help="the Set's metrics next to those of reference studies")
    cp.add_argument("--scenes", metavar="A-B", help="1-based scene range (not needed with --set-json)")
    cp.add_argument("--ref", action="append", required=True, metavar="STUDY",
                    help="a study name (see `roadie study list`); repeat it, or --ref all")
    cp.add_argument("--roles", metavar="MAP.json", help="track name -> role map for the Live read")
    cp.add_argument("--set-json", metavar="FILE", help="a saved `roadie arrangement read --json` instead of Live")
    cp.add_argument("--out", metavar="REPORT.md", help="write the Markdown here")
    cp.add_argument("--json", action="store_true", help="print the report as JSON instead of Markdown")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        try:
            rep = build_compare(args.scenes, args.ref, args.roles, args.set_json)
        except (ValueError, OSError) as e:
            sys.exit("arrangement: %s" % e)
        except LiveError as e:
            sys.exit("arrangement: Live refused: %s" % e)
        md = format_compare(rep)
        if args.out:
            with open(os.path.expanduser(args.out), "w") as f:
                f.write(md)
            print("wrote %s: %d references, %d left out" % (args.out, len(rep["references"]),
                                                            len(rep["left_out"])))
        print(json.dumps(rep, indent=1, ensure_ascii=False) if args.json
              else (md if not args.out else "\n".join("- " + r for r in rep["readings"])))
        return
    if args.cmd == "ref":
        try:
            import study as study_reader
            ref = reference_metrics(study_reader.load_study(args.study))
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
            sys.exit("arrangement: %s" % e)
        print(json.dumps(ref, indent=1, ensure_ascii=False) if args.json else format_reference(ref))
        return
    try:
        first, last = parse_scenes(args.scenes)
        role_map = load_role_map(args.roles) if args.roles else None
        result = read_arrangement(first, last, role_map)
    except (ValueError, OSError) as e:
        sys.exit("arrangement: %s" % e)
    except LiveError as e:
        sys.exit("arrangement: Live refused: %s" % e)
    if args.metrics:
        result["metrics"] = structure_metrics(result)
    if args.json:
        print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        print(format_table(result) + ("\n\n" + format_metrics(result["metrics"])
                                      if args.metrics else ""))


if __name__ == "__main__":
    main()

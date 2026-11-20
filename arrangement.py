#!/usr/bin/env python3
"""Read a range of Session View scenes from Live as an arrangement: which element plays when.

    arrangement.py read --scenes A-B [--roles MAP.json] [--json] [--metrics]

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
    ("lead", r"(?<![a-z])lead(?![a-z])|(?<![a-z])ld(?![a-z])|diva"),
    ("arp", r"(?<![a-z])arp"),
    ("pad", r"(?<![a-z])pad(?![a-z])"),
    ("drone", r"drone"),
    ("pluck", r"pluck"),
    ("vox", r"vox|vocal|voice"),
    ("fx", r"(?<![a-z])fx(?![a-z])|riser|impact|(?<![a-z])down|(?<![a-z])rev(?![a-z])|sweep|noise"),
    ("synth", r"synth"),
)
ROLE_ORDER = [r for r, _ in KEYWORDS]
# A clip named like this holds a device open (a filter, a sidechain) and makes no sound of its own:
# counted as silent, whatever the track maps to. Audio content is not read, so a name is the only way.
SILENT_CLIP = re.compile(r"silent carrier", re.I)
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
    from one step to the next. Anything else is its events ("ev"): those that start in the window,
    relative to its start, the pitches held in from before it, and those still sounding at its end.
    """
    ivs = [i for i in intervals if i[0] < w1 - EPS and i[1] > w0 + EPS]
    if not ivs:
        return None
    ls, le = clip["loop_start"], clip["loop_end"]
    if (clip["looping"] and MIN_LOOP_BEATS <= le - ls <= WINDOW_BEATS + EPS
            and w0 >= steady_from(clip) - EPS):
        cycle = sorted((n["pitch"], round(float(n["start_time"] - ls), 3),
                        round(float(min(n["start_time"] + n["duration"], le) - n["start_time"]), 3),
                        int(n.get("velocity", 100.0) // 8))
                       for n in notes if ls - EPS <= n["start_time"] < le - EPS)
        return ("loop", round(float(le - ls), 3), _digest(cycle))
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


def same_content(a, b):
    """Is record `b` (a step) what record `a` (the step before) was, compared like with like?
    Events are compared over the length of the shorter step; a step with nothing new starting in
    it, only notes held in from the step before, is the same as that step."""
    if a[0] != b[0]:
        return False
    if a[0] != "ev":
        return a == b
    n = min(a[1], b[1])
    sa = [e for e in a[2] if e[1] < n - EPS]
    sb = [e for e in b[2] if e[1] < n - EPS]
    return sa == sb or (not sb and bool(b[3]) and set(b[3]) <= set(a[4]))


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


def read_set(live, first, last):
    """Everything the table needs, read once: scene names, tracks, clips with their notes, Drum
    Rack chains. `first`/`last` are 1-based scene numbers."""
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
        c["silent"] = bool(SILENT_CLIP.search(c["name"] or ""))
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


def read_arrangement(first, last, role_map=None):
    """The table for scenes first..last (1-based, inclusive), as plain data (see format_table)."""
    role_map = _lower_map(role_map)
    with Live() as live:
        data = read_set(live, first, last)
    voices, ignored = build_voices(data, role_map)
    roles = role_order({v["role"] for v in voices})

    rows, prev, started = [], {}, False   # prev: role -> {voice label: record} of the step before
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
                        recs[v["label"]] = rec
                        voice_plays.add(v["track"])
                playing = bool(recs)
                sig = _digest(sorted((k, repr(r)) for k, r in recs.items())) if playing else None
                p_recs = prev.get(role, {})
                if not started and w == 0:
                    change = None
                elif playing:
                    if not p_recs:
                        change = "enter"
                    else:
                        # Only what is playing now can be new. A voice that stopped while
                        # others play on (a one-shot that ran out) is not a change of the role.
                        moved = any(k not in p_recs or not same_content(p_recs[k], recs[k])
                                    for k in recs)
                        change = "vary" if moved else "same"
                else:
                    change = "leave" if p_recs else None
                state = ("silent" if not playing
                         else "changed" if change in ("enter", "vary") else "playing")
                cells.append({"bars": "%d-%d" % (w * WINDOW_BARS + 1,
                                                 min((w + 1) * WINDOW_BARS, bars)),
                              "state": state, "change": change, "sig": sig})
                prev[role] = recs
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

# The base section labels, from the start of a scene name: "Drop 1A · impact" is Drop 1, part A;
# "Outro B" is Outro, part B.
SECTION = re.compile(r"\s*(Intro|Main|Build|Drop|Breakdown|Outro)\b\s*(\d+)?\s*((?-i:[A-Z])\b)?", re.I)


def section_of(scene_name):
    """(label, part) from a scene name by the base labels, or None: "Drop 1A" -> ("Drop 1", "A")."""
    m = SECTION.match(scene_name or "")
    if not m:
        return None
    kind = m.group(1).capitalize()
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
            "section": steps[i]["section"], "scene": steps[i]["scene"]}


def structure_metrics(result):
    """Structure of the range, from its table: sections and lengths, what each 8-bar step adds,
    and the numbers the reference studies are measured in. Bars count from the range's first
    scene (bar 1). A role the Set does not have is None in every metric about it."""
    steps = _steps(result)
    n = len(steps)
    roles = set(result["roles"])
    out = {"bars": sum(x["bars"] for x in steps), "steps": n}

    # sections: consecutive scenes under one label; an unlabelled scene is a section of its own
    sections, unlabelled = [], []
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
                             "bars": row["bars"]})
    bar = 1
    for sec in sections:
        sec["bar"] = bar
        bar += sec["bars"]
        mine = [x for x in steps if sec["scenes"][0] <= x["scene"] <= sec["scenes"][1]]
        counts = [sum(1 for r in PITCHED if _playing(x, r)) for x in mine]
        sec["pitched_parts"] = {"max": max(counts or [0]), "per_step": counts}
    out["sections"], out["unlabelled_scenes"] = sections, unlabelled

    # what each step adds, against the step before; the first step of the range has no before
    changes, flags = [], []
    for i, x in enumerate(steps):
        if i == 0:
            continue
        kinds = {k: sorted(r for r, c in x["cells"].items() if c["change"] == k)
                 for k in ("enter", "vary", "leave")}
        flag = (None if kinds["enter"] or kinds["vary"]
                else "only a removal" if kinds["leave"] else "nothing new")
        rec = {"scene": x["scene"], "name": x["name"], "bar": x["bar"], "bars": x["bars"],
               "in_scene": x["in_scene"], "enter": kinds["enter"], "vary": kinds["vary"],
               "leave": kinds["leave"], "flag": flag}
        changes.append(rec)
        if flag:
            flags.append(rec)
    out["changes"], out["flags"] = changes, flags

    def role_runs(role):
        play = [_playing(x, role) for x in steps]
        return play, _runs(steps, play)

    # hook: how long the lead runs, and whether it varies while it does
    if HOOK_ROLE in roles and n:
        play, runs = role_runs(HOOK_ROLE)
        sigs = [x["cells"][HOOK_ROLE]["sig"] for x, p in zip(steps, play) if p]
        out["hook"] = {
            "role": HOOK_ROLE, "steps_playing": sum(play), "bars_playing": sum(
                x["bars"] for x, p in zip(steps, play) if p),
            "first_bar": next((x["bar"] for x, p in zip(steps, play) if p), None),
            "longest_run": max([_span(steps, i, k) for i, k in runs],
                               key=lambda r: r["bars"], default=None),
            "runs": [_span(steps, i, k) for i, k in runs],
            "varies": sum(1 for x, p in zip(steps[1:], play[1:])
                          if p and x["cells"][HOOK_ROLE]["change"] == "vary"),
            "distinct_variants": len(set(sigs))}
    else:
        out["hook"] = None

    # bass: the bar it enters, and whether that is before the first drop
    if "bass" in roles and n:
        play, _ = role_runs("bass")
        i = play.index(True) if True in play else None
        drop = next((s["bar"] for s in sections if (s["label"] or "").startswith("Drop")), None)
        out["bass"] = {"entry_bar": steps[i]["bar"] if i is not None else None,
                       "entry_scene": steps[i]["scene"] if i is not None else None,
                       "first_drop_bar": drop,
                       "before_first_drop": (None if i is None or drop is None
                                             else steps[i]["bar"] < drop)}
    else:
        out["bass"] = None

    # kick: first bar, and every stretch it is out between two stretches it plays
    if "kick" in roles and n:
        play, _ = role_runs("kick")
        if True in play:
            first, last = play.index(True), n - 1 - play[::-1].index(True)
            gaps = [_span(steps, i, k) for i, k in _runs(steps, [not p for p in play])
                    if first < i and i + k - 1 < last]
            out["kick"] = {"first_bar": steps[first]["bar"], "out": gaps,
                           "longest_out_bars": max([g["bars"] for g in gaps], default=0),
                           "last_bar": steps[last]["bar"] + steps[last]["bars"] - 1}
        else:
            out["kick"] = {"first_bar": None, "out": [], "longest_out_bars": 0, "last_bar": None}
    else:
        out["kick"] = None

    # Drop 2 against Drop 1
    d = {s["label"]: s["bars"] for s in sections if s["label"] in ("Drop 1", "Drop 2")}
    out["drop_ratio"] = ({"drop1_bars": d["Drop 1"], "drop2_bars": d["Drop 2"],
                          "ratio": round(d["Drop 2"] / d["Drop 1"], 3)}
                         if d.get("Drop 1") and "Drop 2" in d else None)

    # hats: do they stay on from the step they enter
    if "hats" in roles and n:
        play, _ = role_runs("hats")
        if True in play:
            first = play.index(True)
            gaps = [_span(steps, i, k) for i, k in _runs(steps, [not p for p in play])
                    if i > first]
            out["hats"] = {"first_bar": steps[first]["bar"], "steps_playing": sum(play),
                           "of_steps": n, "constant": not gaps, "gaps": gaps}
        else:
            out["hats"] = {"first_bar": None, "steps_playing": 0, "of_steps": n,
                           "constant": False, "gaps": []}
    else:
        out["hats"] = None
    return out


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

MARK = {"enter": "+", "vary": "~", "same": "=", "leave": "-", None: "."}


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
             "+ enters  ~ varies  = same  - leaves  . silent;  then loop bars, 1x = plays once",
             "", fmt(head)] + [fmt(b) for b in body]
    if result["unmapped"]:
        lines += ["", "unmapped (no role; map them with --roles): " + ", ".join(result["unmapped"])]
    if result["ignored"]:
        lines += ["", "ignored by the role map: " + ", ".join(result["ignored"])]
    for w in result["warnings"]:
        lines.append("warning: " + w)
    lines += [""] + ["note: " + n for n in result["notes"]]
    return "\n".join(lines)


def format_metrics(m):
    def where(r):
        return "bar %d, %d bars, scene %d" % (r["bar"], r["bars"], r["scene"])
    lines = ["Structure: %d bars in %d steps of 8 (bars count from the first scene of the range)"
             % (m["bars"], m["steps"]), "", "sections"]
    for s in m["sections"]:
        sc = ("scene %d" % s["scenes"][0] if s["scenes"][0] == s["scenes"][1]
              else "scenes %d-%d" % tuple(s["scenes"]))
        lines.append("  %-10s %-14s bar %-3d %3d bars   pitched parts, most at once: %d"
                     % (s["label"] or "(no label)", sc, s["bar"], s["bars"],
                        s["pitched_parts"]["max"]))
    if m["unlabelled_scenes"]:
        lines.append("  scenes whose name matches no section label: " + ", ".join(
            "%d (%r)" % (u["scene"], u["name"]) for u in m["unlabelled_scenes"]))
    lines += ["", "8-bar steps that add nothing"]
    if m["flags"]:
        for f in m["flags"]:
            lines.append("  %s, bars %s of the scene (bar %d): %s"
                         % ((f["name"] or "scene %d" % f["scene"])[:30], f["in_scene"], f["bar"],
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
    else:
        lines.append("  never plays" if h else "  no %s role in this range" % HOOK_ROLE)
    b = m["bass"]
    lines += ["", "bass: " + ("no bass role in this range" if b is None else
                              "never plays" if b["entry_bar"] is None else
                              "in at bar %d (scene %d)%s" % (
                                  b["entry_bar"], b["entry_scene"],
                                  "" if b["before_first_drop"] is None else
                                  ", %s the first drop (bar %d)" % (
                                      "BEFORE" if b["before_first_drop"] else "not before",
                                      b["first_drop_bar"])))]
    k = m["kick"]
    if k is None:
        lines.append("kick: no kick role in this range")
    elif k["first_bar"] is None:
        lines.append("kick: never plays")
    else:
        lines.append("kick: in at bar %d; out %s" % (k["first_bar"], "; ".join(
            "%d bars from bar %d (%s)" % (g["bars"], g["bar"], g["section"] or "no label")
            for g in k["out"]) or "never between two stretches it plays"))
    dr = m["drop_ratio"]
    lines.append("drops: " + ("Drop 2 %d bars / Drop 1 %d bars = %g" % (
        dr["drop2_bars"], dr["drop1_bars"], dr["ratio"]) if dr else "no Drop 1 and Drop 2 in this range"))
    ht = m["hats"]
    lines.append("hats: " + ("no hats role in this range" if ht is None else
                             "never play" if ht["first_bar"] is None else
                             "in at bar %d, play %d of %d steps, %s" % (
                                 ht["first_bar"], ht["steps_playing"], ht["of_steps"],
                                 "constant" if ht["constant"] else "out at " + ", ".join(
                                     "bar %d (%d bars)" % (g["bar"], g["bars"])
                                     for g in ht["gaps"]))))
    return "\n".join(lines)


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
    args = ap.parse_args(argv)
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

#!/usr/bin/env python3
"""Read a range of Session View scenes from Live as an arrangement: which element plays when.

    arrangement.py read --scenes A-B [--roles MAP.json] [--json]

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
UNMAPPED = "unmapped"
IGNORE = "ignore"

NOTES = [
    "Clip automation (envelopes) is not read: a filter ramp or a volume ride on a clip counts as "
    "no change. Silence here is silence of notes and clips, not of the sound.",
    "Audio clips count as playing wherever they are placed: their content is not read, so a clip "
    "that only carries a filter or a sidechain shows as playing.",
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


def midi_window(intervals, w0, w1):
    """(playing, signature) for scene beats [w0, w1): a note sounding at all in the window counts,
    one that started in an earlier window and is still held included. The signature is the
    window's events relative to its start, so a held chord is the same in both steps it spans."""
    ev = sorted((p, round(float(max(s, w0) - w0), 3), round(float(min(e, w1) - w0), 3),
                 int(v // 8))
                for s, e, p, v in intervals if s < w1 - EPS and e > w0 + EPS)
    return (True, _digest(ev)) if ev else (False, None)


def audio_window(clip, horizon, w0, w1):
    """(playing, signature) for an audio clip: it plays wherever it is placed. The signature is
    the clip and where in it the window starts, so a loop that divides the step repeats."""
    cover, pos = 0.0, None
    for t0, a, b in passes(clip, horizon):
        lo, hi = max(w0, t0), min(w1, t0 + (b - a))
        if hi > lo + EPS:
            cover += hi - lo
            if pos is None:
                pos = a + (lo - t0)
    if pos is None:
        return False, None
    if clip["looping"]:
        length = clip["loop_end"] - clip["loop_start"]
        if length >= MIN_LOOP_BEATS:
            pos = (pos - clip["loop_start"]) % length
    return True, _digest((clip["name"], round(float(pos), 2), round(float(cover), 2)))


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
    tr = _reads(live, [{"path": "live_set tracks %d" % t, "property": p}
                       for t in range(n_tracks) for p in ("name", "is_foldable", "has_midi_input")])
    tracks = [{"index": t, "name": tr[t * 3], "foldable": bool(tr[t * 3 + 1]),
               "midi": bool(tr[t * 3 + 2])} for t in range(n_tracks)]

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


def read_arrangement(first, last, role_map=None):
    """The table for scenes first..last (1-based, inclusive), as plain data (see format_table)."""
    role_map = _lower_map(role_map)
    with Live() as live:
        data = read_set(live, first, last)
    voices, ignored = build_voices(data, role_map)
    roles = role_order({v["role"] for v in voices})

    rows, prev, started = [], {}, False
    for s, scene_name in zip(data["scene_ids"], data["scene_names"]):
        here = {t: c for (t, sc), c in data["clips"].items() if sc == s and not c["muted"]}
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
                played[v["key"]] = sounding(c, notes, horizon)
            else:
                played[v["key"]] = None
        for role in roles:
            mine = [v for v in voices if v["role"] == role and v["track"] in here
                    and (v["pitch"] is None or v["key"] in played)]
            clip_info, cells, voice_plays = [], [], set()
            for w in range(n_windows):
                w0, w1 = w * WINDOW_BEATS, min((w + 1) * WINDOW_BEATS, horizon)
                sigs = []
                for v in mine:
                    c = here[v["track"]]
                    ok, sig = (midi_window(played[v["key"]], w0, w1) if c["is_midi_clip"]
                               else audio_window(c, horizon, w0, w1))
                    if ok:
                        sigs.append((v["label"], sig))
                        voice_plays.add(v["track"])
                playing = bool(sigs)
                sig = _digest(sorted(sigs)) if playing else None
                p_play, p_sig = prev.get(role, (False, None))
                if not started and w == 0:
                    change = None
                elif playing:
                    change = "enter" if not p_play else ("vary" if sig != p_sig else "same")
                else:
                    change = "leave" if p_play else None
                state = ("silent" if not playing
                         else "changed" if change in ("enter", "vary") else "playing")
                cells.append({"bars": "%d-%d" % (w * WINDOW_BARS + 1,
                                                 min((w + 1) * WINDOW_BARS, bars)),
                              "state": state, "change": change, "sig": sig})
                prev[role] = (playing, sig)
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
    for (t, s), c in sorted(data["clips"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if c["muted"]:
            warnings.append("muted clip, counted as silent: track %r scene %d"
                            % (data["tracks"][t]["name"], s + 1))
    return {"set": data["set"], "scenes": [first, last], "window_bars": WINDOW_BARS,
            "roles": roles, "rows": rows, "unmapped": unmapped, "ignored": sorted(set(ignored)),
            "warnings": warnings, "notes": NOTES}


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
    args = ap.parse_args(argv)
    try:
        first, last = parse_scenes(args.scenes)
        role_map = load_role_map(args.roles) if args.roles else None
        result = read_arrangement(first, last, role_map)
    except (ValueError, OSError) as e:
        sys.exit("arrangement: %s" % e)
    except LiveError as e:
        sys.exit("arrangement: Live refused: %s" % e)
    print(json.dumps(result, indent=1, ensure_ascii=False) if args.json else format_table(result))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn detected song sections into a launchable Ableton SESSION VIEW grid.

    session_view.py <stems_dir> [--dry-run] [--json]

`stems2live` gives you the track flat: one long clip per stem, laid end to end
in the arrangement. That is a mixdown you can mute — it is not an instrument.
This takes the SAME stems and the SAME section boundaries and rebuilds them as a
grid: one scene per section, one looping audio clip per stem per section. The
arrangement becomes launchable, so the intro can run twice, the vocal can sit
out of the second drop, and a breakdown can loop for as long as the room wants
it. Nothing new is analysed here; `sections.detect_sections` already found the
boundaries and `beat_aligner` already fitted the grid.

THIS TOOL DECORATES, IT DOES NOT BUILD. Tracks must already exist and be named
after their stems — that is `stems2live`'s job. A stem with no matching track is
skipped with a warning rather than given a new track, because inventing tracks
here would silently produce a second, differently ordered copy of the set.

THE OFFSET, which is the one thing that will silently ruin the whole grid.
Sections are reported in GRID beats, counted from the fitted grid origin `t0`.
Clip markers are measured from the SAMPLE START. The two origins differ by `t0`,
so every marker needs

    clip_beat = section_beat + grid_t0 / period

— 0.4938 beats on the reference track. Get it wrong and every clip in the set
sits half a beat out, every loop is half a beat long or short, and there is
nothing on screen to flag it: the grid looks right, plays wrong, and the error
is uniform enough to read as "the stems are a bit loose". The test for it is
that every clip length comes out a whole number of bars; `--dry-run` prints
that column so you can check before 90 clips are written.

WARP FIRST, THEN TRIM, and the ordering is not cosmetic. On a fresh audio clip
`end_marker` is in SECONDS; setting `warping = true` re-expresses every marker
in BEATS (measured: 307.92 -> 641.515 on the same clip). Trimming before the
warp writes second-values into beat-fields and lands the clip roughly 480 times
too short.

Talks to the AbletonLOM remote script on 127.0.0.1:9878 directly, so it runs
from a plain shell without the MCP server in the way.
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
import socket

from sections import (BEATS_PER_BAR, PHRASE_BARS, analyze_sections, cli_grid,
                      stem_paths)

LOM_HOST = "127.0.0.1"
LOM_PORT = 9878

# Shortest section worth a scene, in bars. A sub-bar clip is not launchable
# material: Live quantises the launch to a bar by default, so a half-bar loop
# either never gets heard or fires as a stutter. Sections this short only appear
# when the phrase snap has collapsed two boundaries onto near-neighbours, which
# is a detection artefact rather than a section anyone wants to play.
MIN_SECTION_BARS = 1

# Float slop tolerance, in beats. `start` and `end` both carry the same
# fractional grid offset, so their difference is exact arithmetically — and only
# arithmetically: 4.4937569 - 0.4937569 evaluates to 3.9999999999999996, and a
# bare `<` against one bar then drops a section that is exactly one bar long.
# A millionth of a beat is 2 microseconds; nothing musical lives below it.
BEAT_EPS = 1e-6


class LomError(RuntimeError):
    """The bridge answered, and the answer was a refusal."""


class Bridge:
    """One persistent connection to the AbletonLOM remote script.

    The connection is reused across requests on purpose. A full grid is ~7 calls
    per clip and ~90 clips; a socket per call is 600-odd connects, and the remote
    script spawns a thread for each one. The wire is newline-delimited JSON in
    both directions, so several requests can share a connection safely.

    `params` IS NESTED. A flat body — `{"op": "get", "path": ...}` — reaches the
    handler with no `path` key and comes back as `KeyError: 'path'`, which reads
    like a missing object rather than a malformed request.
    """

    def __init__(self, host=LOM_HOST, port=LOM_PORT, timeout=120.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._sock = None
        self._buf = b""
        self._next_id = 0

    def connect(self):
        if self._sock is not None:
            return
        try:
            self._sock = socket.create_connection((self.host, self.port),
                                                  timeout=self.timeout)
        except (ConnectionRefusedError, OSError) as e:
            raise LomError(
                "nothing listening on %s:%d (%s).\nOpen Ableton Live and set "
                "Preferences -> Link, Tempo & MIDI -> Control Surface = "
                "AbletonLOM." % (self.host, self.port, e))

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._buf = b""

    def request(self, op, **params):
        self.connect()
        self._next_id += 1
        line = json.dumps({"id": self._next_id, "op": op,
                           "params": params}).encode() + b"\n"
        self._sock.sendall(line)
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise LomError("Live closed the connection during %r" % op)
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\n", 1)
        reply = json.loads(raw.decode())
        if not reply.get("ok"):
            raise LomError("%s %s: %s" % (op, params.get("path", ""),
                                          reply.get("error") or "refused"))
        return reply.get("result")

    def get(self, path, prop):
        """The property's VALUE, not the bridge's envelope around it.

        `result` comes back as `{"path", "property", "value", "type"}`. Handing
        that dict to a caller expecting a number fails somewhere far away from
        here — `int(...)` on a dict, with no mention of which read produced it.
        """
        return self.request("get", path=path, property=prop)["value"]

    def set(self, path, prop, value):
        return self.request("set", path=path, property=prop, value=value)

    def call(self, path, function, args=None, confirm=False):
        return self.request("call", path=path, function=function,
                            args=list(args or []), confirm=confirm)

    def count(self, path, child):
        """The integer count, unwrapped from `{"path", "child", "count"}`."""
        return int(self.request("count", path=path, child=child)["count"])


def clip_offset_beats(period, grid_t0):
    """Beats between the sample start and grid beat 0. See the module docstring.

    Positive on every real track: the fitted grid origin sits a fraction of a
    beat INTO the file, so grid beat 0 is already `grid_t0` seconds of audio in,
    and a clip marker counting from the sample start has to add that back.
    """
    if period <= 0:
        raise ValueError("beat period must be positive, got %r" % period)
    return float(grid_t0) / float(period)


def section_clip_beats(section, period, grid_t0):
    """Clip-relative start marker, in beats, for one section.

    Beat-plus-offset rather than `section["time_sec"] / period`, which is the
    same quantity on paper and NOT the same number in practice: `sections.py`
    rounds `time_sec` to milliseconds, and at a 0.48 s period that half-
    millisecond is ~0.001 beats of error which the seconds route inherits and
    this one does not. The difference is inaudible on one clip and is still the
    wrong default — this is the exact spelling. It also puts the offset at the
    call site instead of hiding it inside a division that reads like a plain
    unit conversion.
    """
    return float(section["beat"]) + clip_offset_beats(period, grid_t0)


def plan_session(sections, period, grid_t0, clip_length_beats,
                 beats_per_bar=BEATS_PER_BAR, min_bars=MIN_SECTION_BARS):
    """Work out the whole grid without touching Live. Returns (scenes, skipped).

    Each scene dict carries `scene` (its index in the Session View, assigned
    only to sections that survive), `name`, `grid_beat`, `clip_start`,
    `clip_end`, `length_beats` and `bars`. `skipped` is a list of
    (label, reason) for everything dropped, so the caller can say WHICH sections
    are missing rather than just printing a smaller number of scenes.

    A section ENDS where the next one begins; the last one runs to the end of
    the audio. That last clip is therefore the only one not guaranteed to be a
    whole number of bars — the audio stops where it stops. Every other length
    landing on a bar line is the offset check.

    Scene indices are assigned AFTER skipping, not before. Leaving a hole for a
    dropped section would put an empty row in the middle of the grid, and an
    empty scene in Session View stops everything when you launch it.
    """
    offset = clip_offset_beats(period, grid_t0)
    total = float(clip_length_beats)
    bar = float(beats_per_bar)

    scenes, skipped = [], []
    for i, sec in enumerate(sections):
        start = float(sec["beat"]) + offset
        if i + 1 < len(sections):
            end = float(sections[i + 1]["beat"]) + offset
        else:
            # LAST SECTION: floor to a whole bar. Its end is the end of the FILE,
            # not a section boundary, so it carries whatever fraction of a bar the
            # audio happens to end on — measured 65.0209 beats (16.26 bars) on the
            # reference track. Every other scene came out at exactly 32/64/128/192
            # beats. A clip whose loop is not a whole number of bars drifts a
            # little further out of phase on every repeat, so in a launch grid it
            # is the one clip you cannot use. Costs the last ~0.5 s of audio,
            # which is the tail of an outro.
            end = start + math.floor((total - start) / bar + BEAT_EPS) * bar
        end = min(end, total)
        label = sec.get("label") or sec.get("level") or "Section %d" % (i + 1)

        if start >= total:
            skipped.append((label, "starts past the end of the audio"))
            continue
        length = end - start
        if length < bar * min_bars - BEAT_EPS:
            skipped.append((label, "%.2f beats is under %d bar%s"
                            % (length, min_bars, "" if min_bars == 1 else "s")))
            continue

        scenes.append({
            "scene": len(scenes),
            "name": label,
            "grid_beat": int(sec["beat"]),
            "clip_start": round(start, 4),
            "clip_end": round(end, 4),
            "length_beats": round(length, 4),
            "bars": round(length / bar, 4),
        })
    return scenes, skipped


def map_stems_to_tracks(bridge, stem_names):
    """Stem name -> track index, by reading every track's name out of the Set.

    Matching is on the name `stems2live` wrote, exact first and case-insensitive
    second. Index-based mapping was the obvious alternative and is wrong: the
    stems are ordered by measured spectral centroid, so the mapping changes with
    the material, and any track the user has added or moved since silently
    shifts every clip onto the wrong instrument.
    """
    n = bridge.count("live_set", "tracks")
    names = [bridge.get("live_set tracks %d" % i, "name") for i in range(n)]
    lowered = {}
    for i, nm in enumerate(names):
        lowered.setdefault(str(nm).strip().lower(), i)

    mapping, missing = {}, []
    for stem in stem_names:
        if stem in names:
            mapping[stem] = names.index(stem)
        elif stem.strip().lower() in lowered:
            mapping[stem] = lowered[stem.strip().lower()]
        else:
            missing.append(stem)
    return mapping, missing, names


def track_clip_color(bridge, track_index):
    """Colour of the track's first arrangement clip, or None if it has none.

    Reused rather than recomputed so a column of session clips matches the
    arrangement clip `stems2live` already coloured by spectral centroid. Reading
    it back is what keeps the two views in step without this file having to know
    anything about how that ramp is derived.
    """
    try:
        if bridge.count("live_set tracks %d" % track_index,
                        "arrangement_clips") < 1:
            return None
        return int(bridge.get("live_set tracks %d arrangement_clips 0"
                              % track_index, "color"))
    except (LomError, TypeError, ValueError):
        return None


def ensure_scenes(bridge, scenes, verbose=True):
    """Rename the Set's existing scenes, creating only the shortfall.

    A new Live Set ships with 8 empty scenes. Appending one scene per section on
    top of those leaves 8 dead rows above the grid, and the user has to delete
    them by hand before anything is launchable. Reusing what is there first
    keeps scene 0 as the first section, which is what "launch the top row"
    should mean.
    """
    have = bridge.count("live_set", "scenes")
    need = len(scenes)
    if need > have:
        if verbose:
            print("scenes: %d present, creating %d more" % (have, need - have))
        for _ in range(need - have):
            bridge.call("live_set", "create_scene", [-1])
    elif verbose:
        print("scenes: reusing %d of %d existing" % (need, have))
    for sc in scenes:
        bridge.set("live_set scenes %d" % sc["scene"], "name", sc["name"])


def write_clip(bridge, track_index, scene_index, path, start, end, name,
               color=None):
    """Create, warp, trim and loop one session clip. Returns its full length.

    The full length is read back AFTER warping and BEFORE trimming, because that
    is the only moment it means "how long the sample is, in beats" — and it is
    what the last scene's end has to be clamped against. Markers past the end of
    the sample are rejected by Live one clip at a time, which turns one bad
    number into 10 identical failures with no obvious cause.
    """
    slot = "live_set tracks %d clip_slots %d" % (track_index, scene_index)
    bridge.call(slot, "create_audio_clip", [path])
    clip = slot + " clip"

    # Warping switches every marker on this clip from seconds to beats. Read the
    # length only after it, or the clamp below compares beats against seconds.
    bridge.set(clip, "warping", True)
    full = float(bridge.get(clip, "end_marker"))

    start = max(0.0, min(float(start), full))
    end = max(start, min(float(end), full))

    # LOOP FIRST, THEN THE MARKERS. `start_marker` is SILENTLY IGNORED while the
    # clip is not looping: the write answers ok and the property reads back 0.0.
    # This shipped the wrong way round — start_marker first, looping last — and
    # every one of 90 clips came back at the sample's full 641.515 beats
    # (160.38 bars) instead of its section length, while the run reported the
    # lengths it had ASKED for. Nothing failed; the trims simply never landed.
    bridge.set(clip, "looping", True)
    bridge.set(clip, "loop_start", start)
    bridge.set(clip, "loop_end", end)
    bridge.set(clip, "start_marker", start)
    bridge.set(clip, "end_marker", end)
    bridge.set(clip, "name", name)
    if color is not None:
        bridge.set(clip, "color", int(color))

    # Read the trim back. The failure this guards is silent by construction, so
    # believing the write is exactly how it got shipped the first time.
    got = float(bridge.get(clip, "start_marker"))
    if abs(got - start) > 1e-3:
        raise RuntimeError(
            "trim did not take on track %d scene %d: asked start_marker %.4f, "
            "reads %.4f — the clip is untrimmed and will play the whole stem"
            % (track_index, scene_index, start, got))
    return full


def clear_slot(bridge, track_index, scene_index):
    """Delete an existing clip in the slot. True if there was one to delete.

    `create_audio_clip` refuses a slot that is already full, so without this a
    second run fails on every clip it succeeded at the first time — the failure
    mode of a tool you are meant to re-run while tuning the section detection.
    Gated behind --replace because deleting clips is the user's call.
    """
    slot = "live_set tracks %d clip_slots %d" % (track_index, scene_index)
    if not bridge.get(slot, "has_clip"):
        return False
    bridge.call(slot, "delete_clip", confirm=True)
    return True


def build_session_view(stems_dir, period=None, grid_t0=None,
                       phrase_bars=PHRASE_BARS, n_segments=None,
                       dry_run=False, replace=False, host=LOM_HOST,
                       port=LOM_PORT, verbose=True, downbeat_sec=None):
    """Detect sections and build the Session View grid. Returns the plan dict.

    `period` and `grid_t0` MUST be the ones `stems2live` used when it placed the
    arrangement clips, whenever they are in hand. A second, independently fitted
    grid agrees to within a beat or so, which is exactly bad enough: the session
    clips and the arrangement clips then disagree by a fraction of a beat that
    grows with the section index, and the two views drift apart on a set that
    looks fine in both. `downbeat_sec` likewise: without it scenes start on
    phrases counted from grid beat 0, 1-3 beats off the bar line whenever the
    track's first downbeat is not on grid beat 0.

    With `dry_run` nothing is written. Live is still read — track names have to
    come from somewhere — but the run degrades to an unmapped plan if the bridge
    is not there, so the arithmetic can be checked without Live open at all.
    """
    stems_dir = os.path.abspath(stems_dir)
    paths = {name: os.path.abspath(p)
             for name, p in stem_paths(stems_dir).items()}
    if not paths:
        raise ValueError("no .flac or .wav stems found in %s" % stems_dir)

    if verbose:
        print("analysing sections in %s ..." % os.path.basename(stems_dir))
    res = analyze_sections(stems_dir, phrase_bars=phrase_bars,
                           n_segments=n_segments, period=period,
                           grid_t0=grid_t0, downbeat_sec=downbeat_sec)
    period, grid_t0 = res["period"], res["grid_t0"]
    total_beats = res["duration_sec"] / period

    scenes, skipped = plan_session(res["sections"], period, grid_t0, total_beats)
    if not scenes:
        raise ValueError("no section survived the %d-bar minimum; nothing to build"
                         % MIN_SECTION_BARS)

    bridge = Bridge(host=host, port=port)
    stem_names = sorted(paths)
    mapping, missing, bridge_error = {}, list(stem_names), None
    try:
        mapping, missing, _names = map_stems_to_tracks(bridge, stem_names)
    except LomError as e:
        bridge_error = str(e)
        if not dry_run:
            bridge.close()
            raise

    # Top to bottom in track order, so the printed progress reads the same way
    # the grid does.
    ordered = sorted(mapping, key=mapping.get)

    plan = {
        "stems_dir": stems_dir,
        "bpm": res["bpm"],
        "period": period,
        "grid_t0": grid_t0,
        "clip_offset_beats": round(clip_offset_beats(period, grid_t0), 4),
        "duration_sec": res["duration_sec"],
        "clip_length_beats": round(total_beats, 4),
        "scenes": scenes,
        "skipped_sections": [{"label": l, "reason": r} for l, r in skipped],
        "tracks": {s: mapping[s] for s in ordered},
        "unmapped_stems": missing,
        "bridge_error": bridge_error,
        "dry_run": bool(dry_run),
        "clips_written": 0,
        "clips_failed": 0,
    }

    if dry_run:
        bridge.close()
        return plan

    colors = {s: track_clip_color(bridge, mapping[s]) for s in ordered}
    ensure_scenes(bridge, scenes, verbose=verbose)

    written = failed = 0
    try:
        for sc in scenes:
            done = []
            for stem in ordered:
                ti = mapping[stem]
                try:
                    if replace:
                        clear_slot(bridge, ti, sc["scene"])
                    write_clip(bridge, ti, sc["scene"], paths[stem],
                               sc["clip_start"], sc["clip_end"],
                               "%s %s" % (sc["name"], stem), colors.get(stem))
                    written += 1
                    done.append(stem)
                except LomError as e:
                    failed += 1
                    if verbose:
                        print("  clip failed: scene %d track %d (%s): %s"
                              % (sc["scene"], ti, stem, e))
            # Per scene rather than per clip: 90 lines of output is as unreadable
            # as none, but a silent 90-call run looks hung.
            if verbose:
                print("scene %2d  %-12s  beats %9.4f -> %9.4f  (%.4g bars)  "
                      "%d/%d clips"
                      % (sc["scene"], sc["name"], sc["clip_start"],
                         sc["clip_end"], sc["bars"], len(done), len(ordered)))
    finally:
        bridge.close()

    plan["clips_written"] = written
    plan["clips_failed"] = failed
    return plan


def format_plan(plan):
    """The human-readable plan. This is what --dry-run is FOR.

    The `bars` column is the offset check: every clip except the last must be a
    whole number of bars, because every boundary was snapped to a phrase before
    it got here. A column of x.5 values means the grid offset was dropped and
    the whole grid is half a beat out.
    """
    out = []
    add = out.append
    add("Stems:               %s" % plan["stems_dir"])
    add("Tempo:               %.3f BPM (period %.6f s, t0 %.4f s)"
        % (plan["bpm"], plan["period"], plan["grid_t0"]))
    add("Clip offset:         +%.4f beats  (grid_t0 / period: grid beat 0 sits "
        "this far into the sample)" % plan["clip_offset_beats"])
    add("Clip length:         %.4f beats  (%.2f s, %.4g bars)"
        % (plan["clip_length_beats"], plan["duration_sec"],
           plan["clip_length_beats"] / BEATS_PER_BAR))
    add("Scenes:              %d" % len(plan["scenes"]))
    add("Tracks:              %d of %d stems mapped"
        % (len(plan["tracks"]), len(plan["tracks"]) + len(plan["unmapped_stems"])))
    if plan.get("bridge_error"):
        add("Live:                UNREACHABLE - %s"
            % plan["bridge_error"].splitlines()[0])
        add("                     plan below is arithmetic only; no track mapping")
    add("")

    add("  scene  name          grid beat   clip start     clip end      length"
        "     bars")
    for sc in plan["scenes"]:
        whole = abs(sc["bars"] - round(sc["bars"])) < 1e-6
        add("  %5d  %-12s  %9d  %11.4f  %11.4f  %10.4f  %7.4g%s"
            % (sc["scene"], sc["name"], sc["grid_beat"], sc["clip_start"],
               sc["clip_end"], sc["length_beats"], sc["bars"],
               "" if whole else "  <- partial (track end)"))
    add("")

    if plan["tracks"]:
        add("  tracks: %s" % ", ".join("%s->%d" % (s, t)
                                       for s, t in plan["tracks"].items()))
    if plan["unmapped_stems"]:
        add("  WARNING: no track named %s — those stems get no clips. Run "
            "stems2live first." % ", ".join(plan["unmapped_stems"]))
    for skip in plan["skipped_sections"]:
        add("  skipped section %r: %s" % (skip["label"], skip["reason"]))

    total = len(plan["scenes"]) * len(plan["tracks"])
    if plan["dry_run"]:
        add("")
        add("DRY RUN — nothing written. This plan is %d clips across %d scenes."
            % (total, len(plan["scenes"])))
        add("Check the bars column: EVERY row must be a whole number of bars, "
            "the last one included —")
        add("it is floored to one. A column of half-bars means the clip offset "
            "was lost and the")
        add("whole grid is off by t0/period.")
    else:
        add("")
        add("wrote %d clips, %d failed" % (plan["clips_written"],
                                           plan["clips_failed"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description="Build an Ableton Session View grid from detected sections")
    ap.add_argument("stems_dir", help="Path to a yt2stems folder")
    ap.add_argument("--phrase-bars", type=int, default=PHRASE_BARS,
                    help="snap section boundaries to this phrase length in bars "
                         "(default: %d)" % PHRASE_BARS)
    ap.add_argument("--segments", type=int, default=None,
                    help="raw segment count before snapping; default scales "
                         "with track length")
    ap.add_argument("--tempo", type=float, default=None,
                    help="override the detected tempo (BPM); needs --grid-t0. "
                         "Pass the tempo stems2live used, or the session clips "
                         "will drift against the arrangement ones")
    ap.add_argument("--grid-t0", type=float, default=None,
                    help="grid anchor in seconds, from beat_aligner's grid_t0 "
                         "(default: the stems folder's alignment.json if any)")
    ap.add_argument("--downbeat", type=float, default=None,
                    help="first downbeat in seconds, from beat_aligner's "
                         "downbeat_sec; scenes start on phrases counted from it "
                         "(default: alignment.json, else the first grid beat)")
    ap.add_argument("--replace", action="store_true",
                    help="delete any clip already in a target slot instead of "
                         "failing on it (needed to re-run over an existing grid)")
    ap.add_argument("--port", type=int, default=LOM_PORT,
                    help="AbletonLOM bridge port (default: %d)" % LOM_PORT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full scene/clip plan and write nothing")
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args()

    period, grid_t0, downbeat = cli_grid(args.stems_dir, args.tempo,
                                         args.grid_t0, args.downbeat)
    try:
        plan = build_session_view(
            args.stems_dir, phrase_bars=args.phrase_bars,
            n_segments=args.segments,
            period=period, grid_t0=grid_t0, downbeat_sec=downbeat,
            dry_run=args.dry_run, replace=args.replace,
            port=args.port, verbose=not args.json)
    except (LomError, ValueError) as e:
        sys.exit(str(e))

    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print()
    print(format_plan(plan))


if __name__ == "__main__":
    main()

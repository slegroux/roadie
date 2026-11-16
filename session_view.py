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

Talks to the AbletonLOM remote script on 127.0.0.1:9878 through
`sideman.client.Live`, so it runs from a plain shell without the MCP server in
the way.
"""

import os
import sys

# Re-exec with local .venv python if present to ensure librosa/numpy environment
_real_script = os.path.realpath(__file__)
_venv = os.path.join(os.path.dirname(_real_script), ".venv")
_venv_py = os.path.join(_venv, "bin", "python")
if os.path.realpath(sys.prefix) == os.path.realpath(_venv):
    os.environ.pop("ROADIE_REEXEC", None)    # arrived; later scripts may hop again
elif os.path.exists(_venv_py) and not os.environ.get("ROADIE_REEXEC"):
    os.environ["ROADIE_REEXEC"] = "1"        # one hop, whatever the venv looks like
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import json
import math

try:
    from sideman.client import Live, LiveError
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] != "sideman":
        raise
    sys.exit("sideman is not installed in .venv: run ./install.sh")

import live_batch
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
    """The AbletonLOM remote script, through `sideman.client.Live`.

    One connection is held for the life of the Bridge, through the client's
    `with` protocol, which saves Live a tick on every request (measured on a
    ten-stem Set: 389 ms a request against 603 ms connecting every time).
    `close()` closes it. A full grid is ~90 clips of ~10 operations each; `reads`,
    `ops` and `steps` send them a step at a time for every clip, in a few
    requests instead of one each.

    `Live.request` takes `params` as a dict and nests it for the wire; a flat
    body reaches the handler with no `path` key and comes back as
    `KeyError: 'path'`, which reads like a missing object rather than a
    malformed request. Every method below goes through `request` rather than
    `Live`'s typed ones so that what is sent stays exactly what it was.
    """

    def __init__(self, host=LOM_HOST, port=LOM_PORT, timeout=120.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._live = Live(host, port, timeout).__enter__()

    def close(self):
        """Close the held connection. A request after this connects on its own."""
        self._live.close()

    def request(self, op, **params):
        try:
            return self._live.request(op, params)
        except LiveError as e:
            raise self._failure(e, op, params.get("path", ""))
        except (ConnectionResetError, BrokenPipeError):
            # Mid-request, with Live there a moment ago: not "nothing listening".
            raise
        except OSError as e:
            raise self._unreachable(e)

    def _failure(self, e, op, path):
        """The exception a LiveError on `op` at `path` is raised as."""
        if e.type == "ConnectionRefused":
            return LomError(
                "nothing listening on %s:%d.\nOpen Ableton Live and set "
                "Preferences -> Link, Tempo & MIDI -> Control Surface = "
                "AbletonLOM." % (self.host, self.port))
        if e.type == "ConnectionClosed":
            return LomError("Live closed the connection during %r" % op)
        # Only a timeout waiting on the connection; Live's own "Timeout" reply is a
        # refusal of this one request and falls through to LomError below.
        if e.type == "Timeout" and isinstance(e.__context__, TimeoutError):
            return TimeoutError("no reply from Live within %gs" % self.timeout)
        return LomError("%s %s: %s" % (op, path, e))

    def _unreachable(self, e):
        # Unreachable host, a connect that timed out: the client lets these
        # through unwrapped, and the run has always reported them as "no
        # bridge" rather than crashing.
        return LomError(
            "nothing listening on %s:%d (%s).\nOpen Ableton Live and set "
            "Preferences -> Link, Tempo & MIDI -> Control Surface = "
            "AbletonLOM." % (self.host, self.port, e))

    # Many operations in few requests: see live_batch for what may share a request and why.
    # Each method gives back, per op, what that op sent alone would have: its result, or the
    # LomError `request` would have raised for it (not raised here). A read request that fails
    # as a whole fails every read it carried; only a timeout on the connection is raised, as
    # always. A write request that goes unanswered (Live's Timeout reply, a closed connection,
    # no reply) ends the run with live_batch.StillRunning: Live keeps running a task it gave up
    # waiting on, so its changes may still land and cannot be reported as not made.

    def _batch(self, op, ops, **params):
        """The results list of one get_batch or transaction, or one error per op."""
        writes = op != "get_batch"
        try:
            return self._live.request(op, dict(params, **{
                "ops" if writes else "specs": ops}))["results"], None
        except LiveError as e:
            if writes and e.type in ("Timeout", "ConnectionClosed"):
                raise live_batch.StillRunning(str(e))
            errors = [self._failure(e, o.get("op", "get"), o.get("path", "")) for o in ops]
            if isinstance(errors[0], TimeoutError):
                raise errors[0]
            return None, errors
        except (ConnectionResetError, BrokenPipeError) as e:
            if writes:
                raise live_batch.StillRunning(str(e) or type(e).__name__)
            raise
        except OSError as e:
            return None, [self._unreachable(e)] * len(ops)

    def _entry(self, spec, entry):
        """(result, error) for one op's entry in a batch's results."""
        if not entry.get("ok"):
            return None, LomError("%s %s: %s" % (spec.get("op", "get"), spec.get("path", ""),
                                                  entry.get("error")))
        if spec.get("op") == "transaction":
            # A nested live_batch.seq: its first failed op is its error.
            for sub, res in zip(spec["ops"], entry["result"]["results"]):
                if not res.get("ok"):
                    return entry["result"], self._entry(sub, res)[1]
        return entry.get("result"), None

    def reads(self, specs):
        """Read every {path, property}: (value, error) each, in order. `get_batch` never fails
        as a whole, so an unreadable property costs only its own entry."""
        out = []
        for chunk in live_batch.pack([dict(s, op="get") for s in specs]):
            res, errors = self._batch("get_batch", [{"path": s["path"],
                                                     "property": s["property"]}
                                                    for s in chunk])
            if errors:
                out.extend((None, e) for e in errors)
                continue
            for s, entry in zip(chunk, res):
                if entry.get("ok"):
                    out.append((entry["value"], None))
                else:
                    out.append(self._entry(s, entry))
        return out

    def ops(self, ops):
        """Run sets and calls in order, none stopping another: (result, error) each. Each request
        is one `transaction`, so ONE undo step."""
        out = []
        for chunk in live_batch.pack(ops):
            res, errors = self._batch("transaction", chunk, stop_on_error=False,
                                      rollback_on_error=False)
            if errors:
                out.extend((None, e) for e in errors)
            else:
                out.extend(self._entry(s, entry) for s, entry in zip(chunk, res))
        return out

    def steps(self, ops):
        """Run ops in order; raise the LomError of the first that fails, as one request each did."""
        for chunk in live_batch.pack(ops):
            res, errors = self._batch("transaction", chunk, stop_on_error=True,
                                      rollback_on_error=False)
            if errors:
                raise errors[0]
            for s, entry in zip(chunk, res):
                err = self._entry(s, entry)[1]
                if err is not None:
                    raise err

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

    One read: Live lists up to 64 tracks with their names. A longer list comes
    back as a count, and the names are then read in a second request.
    """
    ((tracks, err),) = bridge.reads([{"path": "live_set", "property": "tracks"}])
    if err is not None:
        raise err
    if isinstance(tracks, list) and all("name" in t for t in tracks):
        names = [t["name"] for t in tracks]
    else:
        names = []
        for value, err in bridge.reads([{"path": "live_set tracks %d" % i, "property": "name"}
                                        for i in range(live_batch.vector_len(tracks))]):
            if err is not None:
                raise err
            names.append(value)
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


def track_clip_colors(bridge, track_indices, *specs):
    """{track index: colour of its first arrangement clip, or None if it has none}.

    Reused rather than recomputed so a column of session clips matches the
    arrangement clip `stems2live` already coloured by spectral centroid. Reading
    it back is what keeps the two views in step without this file having to know
    anything about how that ramp is derived.

    One request for every track: the clip list (its length is the count) and
    clip 0's colour, read together. A track with no clip fails the colour read
    on its own, and that read is not used. With `specs` those reads go in the
    same request, and ({track: colour}, their (value, error)s) comes back.
    """
    reads = list(specs)
    for ti in track_indices:
        reads += [{"path": "live_set tracks %d" % ti, "property": "arrangement_clips"},
                  {"path": "live_set tracks %d arrangement_clips 0" % ti, "property": "color"}]
    got = bridge.reads(reads)
    extra, got = got[:len(specs)], got[len(specs):]
    out = {}
    for k, ti in enumerate(track_indices):
        (clips, err), (color, color_err) = got[2 * k], got[2 * k + 1]
        try:
            if err is not None or live_batch.vector_len(clips) < 1 or color_err is not None:
                out[ti] = None
                continue
            out[ti] = int(color)
        except (TypeError, ValueError):
            out[ti] = None
    return (out, extra) if specs else out


def ensure_scenes(bridge, scenes, verbose=True, have=None):
    """Rename the Set's existing scenes, creating only the shortfall.

    A new Live Set ships with 8 empty scenes. Appending one scene per section on
    top of those leaves 8 dead rows above the grid, and the user has to delete
    them by hand before anything is launchable. Reusing what is there first
    keeps scene 0 as the first section, which is what "launch the top row"
    should mean.

    The creates go in one request and the names in the next, so no scene is
    named by index in the request that made it. Both stop at the first failure
    and raise it, as one request each did. `have` is the scene count when the
    caller has read it.
    """
    if have is None:
        have = bridge.count("live_set", "scenes")
    need = len(scenes)
    if need > have:
        if verbose:
            print("scenes: %d present, creating %d more" % (have, need - have))
        bridge.steps([{"op": "call", "path": "live_set", "function": "create_scene",
                       "args": [-1], "confirm": False}] * (need - have))
    elif verbose:
        print("scenes: reusing %d of %d existing" % (need, have))
    bridge.steps([{"op": "set", "path": "live_set scenes %d" % sc["scene"],
                   "property": "name", "value": sc["name"]} for sc in scenes])


def write_clips(bridge, jobs, replace=False):
    """Clear (with `replace`), create, warp, trim and loop session clips.

    `jobs` are dicts of track, scene, path, start, end, name and color, in the
    order the clips used to be written one at a time. One outcome per job, in
    that order: the clip's full length, the LomError that stopped it (as
    `request` would have raised it), or an exception that ended the run at that
    clip — a trim that did not take — which the caller raises when it reaches
    it. Once a job has ended the run, no job after it is sent anything more.

    THE FIRST CLIP GOES ALONE. It runs through every step on its own before the
    rest are batched, so a trim that does not take on every clip (the bug below
    that once shipped: 90 clips at full length) ends the run after ONE clip, as
    when clips were written one at a time. A failure that hits a later clip only
    ends the run at that clip too, but the clips batched with it in the same
    requests have by then been written the same way.

    Every loop and marker set is checked against the value Live answers with (a
    set replies with the property as it reads after the write, in the same task),
    and a mismatch ends the run like the read-back does, at no extra request.
    Name and colour are not checked: Live keeps its own palette and names.

    The full length is read back AFTER warping and BEFORE trimming, because that
    is the only moment it means "how long the sample is, in beats" — and it is
    what the last scene's end has to be clamped against. Markers past the end of
    the sample are rejected by Live one clip at a time, which turns one bad
    number into 10 identical failures with no obvious cause.

    Each step is one request (or a few, see live_batch) for every clip still
    standing, and every step that depended on the one before it in the
    one-clip-at-a-time code is still a request after it: the delete before the
    create into that slot, the create before anything addresses the new clip,
    `warping` before the length is read, `looping` before the markers, the
    markers before the read-back. Two stems on one track write the same slots:
    the second is a round of its own after the first, so it sees the slot as
    the first left it.
    """
    out = [None] * len(jobs)
    stop = [len(jobs)]                        # the first job that ended the run

    def slot(j):
        return "live_set tracks %d clip_slots %d" % (jobs[j]["track"], jobs[j]["scene"])

    def live(ks):
        return [k for k in ks if out[k] is None and k < stop[0]]

    def end_run(k, exc):
        out[k] = exc
        stop[0] = min(stop[0], k)

    def run(ks, make, checked=()):
        """One op per job, all sent; a failure ends that job. A set of a property in `checked`
        whose reply reads other than asked ends the run at that job."""
        ks = live(ks)
        sent = [make(k) for k in ks]
        for k, op, (res, err) in zip(ks, sent, bridge.ops(sent)):
            if err is not None:
                out[k] = err
                continue
            sets = zip(op["ops"], res["results"]) if op["op"] == "transaction" else [(op, {
                "result": res})]
            for spec, entry in sets:
                prop = spec.get("property")
                if prop in checked and not _took(spec["value"], entry["result"]["value"]):
                    end_run(k, RuntimeError(_did_not_take(jobs[k], prop, spec["value"],
                                                          entry["result"]["value"])))
                    break
        return live(ks)

    seen, rounds = {}, []
    for k, job in enumerate(jobs):
        n = seen[slot(k)] = seen.get(slot(k), -1) + 1
        if n == len(rounds):
            rounds.append([])
        rounds[n].append(k)
    if rounds and len(rounds[0]) > 1:
        rounds[0:1] = [rounds[0][:1], rounds[0][1:]]          # the first clip goes alone

    bounds = {}
    for ks in rounds:
        # `create_audio_clip` refuses a slot that is already full, so without the
        # clear a second run fails on every clip it succeeded at the first time —
        # the failure mode of a tool you are meant to re-run while tuning the
        # section detection. Gated behind --replace because deleting clips is the
        # user's call.
        if replace:
            ks = live(ks)
            full = []
            for k, (has, err) in zip(ks, bridge.reads([{"path": slot(k), "property": "has_clip"}
                                                      for k in ks])):
                if err is not None:
                    out[k] = err
                elif has:
                    full.append(k)
            run(full, lambda k: {"op": "call", "path": slot(k), "function": "delete_clip",
                                 "args": [], "confirm": True})
        ks = run(ks, lambda k: {"op": "call", "path": slot(k), "function": "create_audio_clip",
                                "args": [jobs[k]["path"]], "confirm": False})

        # Warping switches every marker on this clip from seconds to beats. Read the
        # length only after it, or the clamp below compares beats against seconds.
        ks = run(ks, lambda k: {"op": "set", "path": slot(k) + " clip",
                                "property": "warping", "value": True}, ("warping",))
        for k, (value, err) in zip(ks, bridge.reads([{"path": slot(k) + " clip",
                                                      "property": "end_marker"} for k in ks])):
            if err is not None:
                out[k] = err
                continue
            try:
                full = float(value)
            except Exception as e:
                end_run(k, e)
                continue
            start = max(0.0, min(float(jobs[k]["start"]), full))
            bounds[k] = (full, start, max(start, min(float(jobs[k]["end"]), full)))

        # LOOP FIRST, THEN THE MARKERS. `start_marker` is SILENTLY IGNORED while the
        # clip is not looping: the write answers ok and the property reads back 0.0.
        # This shipped the wrong way round — start_marker first, looping last — and
        # every one of 90 clips came back at the sample's full 641.515 beats
        # (160.38 bars) instead of its section length, while the run reported the
        # lengths it had ASKED for. Nothing failed; the trims simply never landed.
        ks = run(ks, lambda k: {"op": "set", "path": slot(k) + " clip",
                                "property": "looping", "value": True}, ("looping",))

        def marks(k):
            clip, (_full, start, end) = slot(k) + " clip", bounds[k]
            sets = [("loop_start", start), ("loop_end", end), ("start_marker", start),
                    ("end_marker", end), ("name", jobs[k]["name"])]
            if jobs[k]["color"] is not None:
                sets.append(("color", int(jobs[k]["color"])))
            return live_batch.seq([{"op": "set", "path": clip, "property": p, "value": v}
                                   for p, v in sets])
        ks = run(ks, marks, live_batch.MARKERS)

        # Read the trim back. The failure this guards is silent by construction, so
        # believing the write is exactly how it got shipped the first time.
        for k, (value, err) in zip(ks, bridge.reads([{"path": slot(k) + " clip",
                                                      "property": "start_marker"} for k in ks])):
            if err is not None:
                out[k] = err
                continue
            try:
                got = float(value)
            except Exception as e:
                end_run(k, e)
                continue
            full, start, _end = bounds[k]
            if not _took(start, got):
                end_run(k, RuntimeError(_did_not_take(jobs[k], "start_marker", start, got)))
            else:
                out[k] = full
    return out


def _took(asked, got):
    """Does a property read `got` after `asked` was written to it?"""
    if isinstance(asked, bool) or not isinstance(asked, (int, float)):
        return got == asked
    try:
        return abs(float(got) - asked) <= 1e-3
    except (TypeError, ValueError):
        return False


def _did_not_take(job, prop, asked, got):
    """The message a loop or marker write that did not stick ends the run with."""
    if prop == "start_marker" and isinstance(got, (int, float)):
        return ("trim did not take on track %d scene %d: asked start_marker %.4f, "
                "reads %.4f — the clip is untrimmed and will play the whole stem"
                % (job["track"], job["scene"], asked, float(got)))
    return ("trim did not take on track %d scene %d: asked %s %s, reads %s — the clip "
            "does not loop the section" % (job["track"], job["scene"], prop, asked, got))


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
    except BaseException:            # a timeout, Ctrl-C: the Bridge holds a connection now, so it is closed
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

    written = failed = 0
    try:
        # The scene count goes with the colours: both are reads, nothing is written before.
        by_track, ((have, err),) = track_clip_colors(
            bridge, [mapping[s] for s in ordered], {"path": "live_set", "property": "scenes"})
        if err is not None:
            raise err
        colors = {s: by_track[mapping[s]] for s in ordered}
        ensure_scenes(bridge, scenes, verbose=verbose, have=live_batch.vector_len(have))
        jobs = [{"track": mapping[stem], "scene": sc["scene"], "path": paths[stem],
                 "start": sc["clip_start"], "end": sc["clip_end"],
                 "name": "%s %s" % (sc["name"], stem), "color": colors.get(stem)}
                for sc in scenes for stem in ordered]
        outcomes = iter(write_clips(bridge, jobs, replace=replace))
        for sc in scenes:
            done = []
            for stem in ordered:
                ti = mapping[stem]
                outcome = next(outcomes)
                if isinstance(outcome, LomError):
                    failed += 1
                    if verbose:
                        print("  clip failed: scene %d track %d (%s): %s"
                              % (sc["scene"], ti, stem, outcome))
                elif isinstance(outcome, BaseException):
                    raise outcome
                else:
                    written += 1
                    done.append(stem)
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

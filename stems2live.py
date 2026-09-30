#!/usr/bin/env python3
"""Load a yt2stems output folder into the running Ableton Live set.

    stems2live.py <stems_dir> [--tempo 126] [--first-track 2] [--start-bar 1]

Places every stem as an arrangement clip ON A BAR LINE — the lead-in ahead of
the first grid beat is trimmed off the front of each clip rather than pushing
the clip off the bar, so the clip starts at bar `--start-bar` exactly and the
audio still lands where the beat grid says it should. Names each track after its
stem, and colours the clips on a red->blue ramp ordered by measured spectral
centroid, so the arrangement reads low-frequency at the top to high at the
bottom.

Requires Live running with the AbletonLOM control surface active (it listens on
127.0.0.1:9878). Talks to that socket directly rather than going through the MCP
server, so it works from a plain shell.

The bridge is generic: instead of one command per feature it exposes the Live
Object Model itself, so every call here is a `get`/`set`/`call`/`count` against
a space-separated LOM path such as `live_set tracks 3 arrangement_clips 0`.

One constraint comes from Live itself, not from this script: its Object Model
exposes no track reordering, so "order" here means which stem is assigned to
which existing track index, counting down from --first-track. Missing tracks are
created at the end of the set. (Track COLOUR is settable — the old AbletonMCP
bridge simply had no handler for it.)

KICK IS PINNED FIRST, ahead of the centroid sort. On the set this was built
against the kick measured 66 Hz against the bass at 68 Hz — a 2 Hz margin, i.e.
luck. A sub-heavy track flips that ordering, and the kick belongs at the top
regardless of what the measurement says.
"""

import argparse, colorsys, itertools, json, math, os, socket, subprocess, sys
from concurrent import futures

PORT = 9878
# geometric centres of the bands sampled for the centroid
BANDS = [(30, 80), (80, 200), (200, 500), (500, 1200),
         (1200, 3000), (3000, 7000), (7000, 16000)]
CENTRES = [math.sqrt(lo * hi) for lo, hi in BANDS]

_REQ_ID = itertools.count(1)


def lom(op, **params):
    """One round trip to the AbletonLOM bridge. Returns the reply envelope.

    Wire format is newline-delimited JSON: a request is
    `{"id": n, "op": ..., "params": {...}}` — note `params` is NESTED, a flat
    body makes the handler raise `KeyError: 'path'` — and the reply is exactly
    one line, `{"id": n, "ok": bool, "result": ..., "error": {...}}`.

    Read to the first NEWLINE, not to EOF: the bridge keeps the connection open
    for further requests, so waiting for the socket to close would hang.
    """
    try:
        s = socket.create_connection(("localhost", PORT), timeout=120)
    except ConnectionRefusedError:
        sys.exit("Nothing listening on 127.0.0.1:%d.\n"
                 "Open Ableton Live and set Preferences -> Link, Tempo & MIDI ->\n"
                 "Control Surface = AbletonLOM (Input/Output = None)." % PORT)
    try:
        s.sendall((json.dumps({"id": next(_REQ_ID), "op": op, "params": params})
                   + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                raise RuntimeError("no reply from Live on port %d" % PORT)
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode())
    finally:
        s.close()


def lom_err(reply):
    """The error message from a reply envelope; "" when it succeeded."""
    if reply.get("ok"):
        return ""
    return str((reply.get("error") or {}).get("message") or "unknown error")


def status(reply):
    """The word the old bridge put in its `status` field, so the table reads
    the same as it always did."""
    return "success" if reply.get("ok") else "error"


def lom_get(path, prop):
    r = lom("get", path=path, property=prop)
    if not r.get("ok"):
        raise RuntimeError("get %s.%s: %s" % (path, prop, lom_err(r)))
    return r["result"]["value"]


def lom_count(path, child):
    r = lom("count", path=path, child=child)
    if not r.get("ok"):
        raise RuntimeError("count %s.%s: %s" % (path, child, lom_err(r)))
    return r["result"]["count"]


def probe_bridge():
    """Confirm the bridge answers and knows `warp_markers_set`. Returns whether
    the warp op is available.

    The op is probed by CALLING it with deliberately empty params: a bridge that
    has it fails inside the handler (`KeyError: 'path'`), one that does not fails
    at dispatch with "unknown op". Nothing is touched either way. Testing for the
    op name in the reply would not work — the "unknown op" message lists the
    known ops and quotes the one asked for, so the name is present regardless.
    """
    r = lom("ping")
    if not r.get("ok"):
        sys.exit("AbletonLOM did not answer ping: %s" % lom_err(r))
    res = r.get("result") or {}
    if not res.get("handlers"):
        sys.exit("AbletonLOM is listening but its handlers failed to load:\n%s\n"
                 "Fix handlers.py and send {\"op\": \"reload\"} — no Live restart "
                 "needed." % res.get("handler_error"))
    if "unknown op" in lom_err(lom("warp_markers_set")).lower():
        print("NOTE: this bridge has no `warp_markers_set` op, so a drifting set "
              "cannot be warped.\n"
              "      Send {\"op\": \"reload\"} to port %d to pick up an updated "
              "handlers.py —\n"
              "      the LOM handlers hot-reload, so this needs NO Live restart "
              "(unlike the MCP path)." % PORT)
        return False
    return True


def arrangement_clips(track_index):
    """[(index, start_time, end_time)] for a track's arrangement clips.

    ARRANGEMENT CLIPS ARE INDEXED BY TIME, NOT BY CREATION ORDER. `end_time` is
    read best-effort and comes back None if Live will not give it — it is only
    used to describe leftovers in a message, never to decide anything.
    """
    out = []
    for i in range(lom_count("live_set tracks %d" % track_index,
                             "arrangement_clips")):
        p = "live_set tracks %d arrangement_clips %d" % (track_index, i)
        try:
            end = float(lom_get(p, "end_time"))
        except RuntimeError:
            end = None
        out.append((i, float(lom_get(p, "start_time")), end))
    return out


# Live's mixer fader is a 0..1 taper, NOT dB, and the mapping is not documented.
# Measured against Live 12.2.7 by setting values and reading display_value back:
#
#     1.00 -> +6.0    0.80 -> -2.0    0.60 -> -10.0
#     0.85 ->  0.0    0.75 -> -4.0    0.50 -> -14.0
#     0.92 -> +2.8    0.70 -> -6.0    0.40 -> -18.0
#
# Dead linear at 40 dB per unit across that span, so value = 0.85 + dB/40 and
# unity is 0.85 (which is also Live's default). It stops being linear below
# ~0.4: 0.30 reads -24.2 where the formula predicts -22, so the conversion is
# refused past MIN_FADER_DB rather than quietly returning a wrong fader.
UNITY_FADER = 0.85
DB_PER_FADER_UNIT = 40.0
MIN_FADER_DB = -18.0


def db_to_fader(db):
    """Live mixer fader value for a dB offset from unity. See the table above."""
    if db > 6.0 or db < MIN_FADER_DB:
        raise ValueError("headroom %.1f dB is outside the measured linear range "
                         "(+6 .. %.0f dB)" % (db, MIN_FADER_DB))
    return UNITY_FADER + db / DB_PER_FADER_UNIT


def headroom_arg(text):
    """argparse type for --headroom: a dB amount db_to_fader can take.

    Checked at parse time so an out-of-range value is a usage error before
    anything is sent to Live — not a ValueError after the tempo is written.
    The sign is dropped the same way main() drops it.
    """
    try:
        db = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("not a number: %r" % text) from None
    if not math.isfinite(db):
        raise argparse.ArgumentTypeError("not a finite number: %r" % text)
    try:
        db_to_fader(-abs(db))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return db


def set_track_headroom(track_index, db):
    """Pull one track's fader down by `db`. Returns (ok, detail).

    Applied per stem track rather than to the master on purpose: the point is to
    leave YOUR faders reading near unity while the SUM has room, so subsequent
    mix moves are relative to a sane starting point. Pulling the master down
    instead would leave every stem fader at unity and hide the headroom.
    """
    try:
        value = db_to_fader(db)
    except ValueError as e:
        return False, str(e)
    path = "live_set tracks %d mixer_device volume" % track_index
    r = lom("set", path=path, property="value", value=value)
    if not r.get("ok"):
        return False, lom_err(r)
    return True, "%.1f dB" % db


def write_energy_map(notes, rows, sections, start_beat, name="ENERGY MAP",
                     shift_beats=0.0):
    """Put the energy matrix on a silent MIDI track at the top of the set.

    A MIDI track with no instrument makes no sound but IS visible, so the piano
    roll becomes a legend for the arrangement: one row per stem in the same
    order as the tracks below it, blocks where that stem plays, velocity for how
    hard. Read it against the locators.

    Inserted at index 0 so it sits above the stems — `create_midi_track` takes a
    position, so this is not an append-then-move (Live exposes no track
    reordering, so an append could not be moved afterwards).

    Any existing track called `name` is deleted first, so a re-run replaces the
    map instead of stacking another one on top of it.

    `shift_beats` moves every note from sample grid beats to beats from the
    clip start: when the lead-in up to the downbeat is trimmed off the audio,
    grid beat `-shift_beats` is where the audio clips start. Notes that would
    land before the clip are pinned to its start.
    """
    if shift_beats:
        notes = [dict(n, start_time=max(0.0, float(n["start_time"]) + shift_beats))
                 for n in notes]
    # Highest index first, so deleting one does not shift the ones still to check.
    for i in reversed(range(lom_count("live_set", "tracks"))):
        r = lom("get", path="live_set tracks %d" % i, property="name")
        if r.get("ok") and r["result"]["value"] == name:
            d = lom("call", path="live_set", function="delete_track", args=[i])
            if not d.get("ok"):
                print("energy map: could not replace the existing track %d: %s"
                      % (i, lom_err(d)))
                return
    r = lom("call", path="live_set", function="create_midi_track", args=[0])
    if not r.get("ok"):
        print("energy map: could not create the track: %s" % lom_err(r))
        return
    lom("set", path="live_set tracks 0", property="name", value=name)

    # Span the whole arrangement so every section has somewhere to live.
    last = sections[-1]
    length = float(last["beat"]) + 64.0
    r = lom("call", path="live_set tracks 0", function="create_midi_clip",
            args=[float(start_beat), length])
    if not r.get("ok"):
        print("energy map: could not create the clip: %s" % lom_err(r))
        return
    clip = "live_set tracks 0 arrangement_clips 0"
    lom("set", path=clip, property="name", value="arrangement energy")

    r = lom("notes_add", path=clip, notes=notes)
    if not r.get("ok"):
        print("energy map: notes rejected: %s" % lom_err(r))
        return
    print("energy map: %d notes on a silent MIDI track at the top — rows top "
          "to bottom: %s" % (len(notes), ", ".join(rows)))


def arrange_stems(stems, cent):
    """Order stems: kick, bass, the rest of the kit, then everything else.

    Pure centroid order was the old rule and it interleaves the kit with the
    melodic stems, because a kit piece is not where its centroid says it is.
    Measured on the reference track: synths 381 Hz and vocals 522 Hz both sit
    BELOW drums_toms at 550 Hz, so ascending centroid put the pads and the
    vocal in the middle of the drums.

    Grouped instead:

      1. kick   — pinned, see the module docstring. It measured 69 Hz against
                  the bass at 66 Hz, so centroid alone would put bass on top.
      2. bass   — the other half of the low end, next to the kick where you mix
                  it.
      3. kit    — every other `drums*` stem, ascending by centroid. Toms are the
                  lowest of them (550 Hz), so they land nearest the kick without
                  needing a rule of their own: the kit stays contiguous AND the
                  low percussion stays near the low end.
      4. rest   — melodic and vocal stems, ascending by centroid.

    Ascending within each group, so the colour ramp still reads low-to-high
    down the arrangement — it just no longer crosses between families.
    """
    kick = [n for n in stems if "kick" in n.lower()]
    bass = [n for n in stems if n not in kick and "bass" in n.lower()]
    used = set(kick) | set(bass)
    kit = sorted((n for n in stems
                  if n not in used and n.lower().startswith("drum")),
                 key=cent.get)
    rest = sorted((n for n in stems if n not in used and n not in kit),
                  key=cent.get)
    return kick + bass + kit + rest


def is_audio_track(track_index):
    """Can this track hold an audio clip?

    `has_audio_input` is the property Live exposes for it; a MIDI track answers
    False and raises nothing, so a missing/failed read is treated as "not safe
    to use" rather than assumed audio.
    """
    r = lom("get", path="live_set tracks %d" % track_index,
            property="has_audio_input")
    return bool(r.get("ok")) and bool(r["result"]["value"])


def audio_track_targets(first_track, count):
    """`count` audio-track indices at or after `first_track`, creating as needed.

    Existing AUDIO tracks are reused in order; MIDI tracks in the range are
    stepped over, not renamed and not written to. If that leaves too few, audio
    tracks are appended until there are enough.

    The old rule counted tracks and appended when the TOTAL was short, which is
    a different question and gives the wrong answer whenever a Set mixes types:
    a Set with plenty of tracks, four of them MIDI, satisfied the count and then
    failed on every MIDI one — after renaming it.
    """
    total = lom_count("live_set", "tracks")
    targets, skipped = [], []
    for i in range(first_track, total):
        if len(targets) == count:
            break
        (targets if is_audio_track(i) else skipped).append(i)

    if skipped:
        print("skipping %d non-audio track%s at %s — they cannot hold audio "
              "clips and are left untouched"
              % (len(skipped), "" if len(skipped) == 1 else "s",
                 ", ".join(str(i) for i in skipped)))

    missing = count - len(targets)
    if missing > 0:
        print("creating %d audio track%s (set has %d)"
              % (missing, "" if missing == 1 else "s", total))
        for _ in range(missing):
            r = lom("call", path="live_set", function="create_audio_track",
                    args=[-1])
            if not r.get("ok"):
                sys.exit("create_audio_track failed: %s" % lom_err(r))
            targets.append(lom_count("live_set", "tracks") - 1)
    return targets


def find_placed_clip(clips, position, tol=1e-4):
    """Index of the clip starting at `position`, or None. See the warning below.

    `arrangement_clips 0` IS NOT "THE CLIP WE JUST MADE". Live orders arrangement
    clips by time, so index 0 is the EARLIEST clip on the track — which is only
    the new one when the track was empty. Re-run over an existing Set at a
    different --start-bar and the previous clip's head survives as a stub in
    front of the new clip (creating a clip replaces only the span it covers), so
    index 0 is the STUB.

    Measured on a real run at --start-bar 9: every track came back with clip 0 at
    bar 1.8766 (a 7-bar leftover) and clip 1 at bar 9.0000. The trim was applied
    to clip 0, its read-back check PASSED because the stub genuinely had been
    trimmed, and the clip that mattered stayed untrimmed — leaving the downbeat
    at bar 26.12 instead of 26.00. Colour, muted and the warp map were landing on
    the same wrong clip; they only ever looked right because the track had one
    clip.
    """
    for i, start, _end in clips:
        if abs(start - position) <= tol:
            return i
    return None


def apply_warp_map(track_index, clip_index, markers):
    """Write a warp map to a clip. Returns (state, detail) for the caller to print.

    `state` is "ok", "unsupported", or "failed", and the distinction matters
    because only one of the three is worth aborting the warp step over:

      unsupported — the bridge's handlers.py predates the `warp_markers_set` op,
        so dispatch answers "unknown op". Unlike the AbletonMCP path, this does
        NOT need a Live restart: handlers.py is reimported on {"op": "reload"}.
      failed — the map was rejected (bad geometry, missing clip). Reported per
        clip; the clips themselves are already placed and useful.

    This has to be its own op rather than a generic `call`, because
    Clip.add_warp_marker() takes a C++ TWarpMarker that no JSON argument can
    express — the object has to be built by Python running inside Live.

    A nonzero `remove_failed` in the reply is NORMAL and is not an error. A fresh
    audio clip carries a shadow marker encoding its detected tempo, and Live
    refuses to remove it; the bridge counts that refusal instead of aborting.
    """
    r = lom("warp_markers_set",
            path="live_set tracks %d arrangement_clips %d" % (track_index,
                                                              clip_index),
            markers=markers)
    if r.get("ok"):
        res = r.get("result") or {}
        return "ok", ("warp %d markers (+%d -%d, %d refused)"
                      % (res.get("marker_count", 0), res.get("added", 0),
                         res.get("removed", 0), res.get("remove_failed", 0)))
    msg = lom_err(r) or "no message"
    if "unknown op" in msg.lower():
        return "unsupported", msg
    return "failed", msg


def bar_aligned_placement(grid_info, start_bar, tempo):
    """Where the clips go and how much lead-in to trim. Returns
    (position_beats, trim_sec, downbeat_beat).

    THE PROBLEM. The file's own beat 0 sits `grid_t0` seconds before the first
    grid beat — 0.237 s here, 0.4938 of a beat — and that fraction has to go
    somewhere. Offsetting the clip position by it (what `start_offset_beats`
    does) puts the audio right but starts the clip at bar 1.88, mid-bar. Taking
    it off the FRONT OF THE CLIP instead lets the clip start on a bar line with
    the audio landing in exactly the same place.

    THE PHASE. Trimming `grid_t0` alone only aligns the clip to the grid, not to
    a BAR: the anchor sits at grid index `k0`, and `k0 % 4` is the beat of the
    bar it falls on. Trimming `(k0 % 4)` further beats absorbs that, so the
    anchor lands on a multiple of 4 for ANY phase — with phase 0 it is already a
    bar line and the extra trim is zero, with phase 2 it would otherwise be two
    beats out. `k0` is exact by construction: the anchor IS a grid beat, so
    `(downbeat_sec - grid_t0) / period` is an integer up to float noise.

    The trim never exceeds one bar, so the clip lands at `(start_bar - 1) * 4`
    and the anchor falls on a bar line of Live's grid.
    """
    period = float(grid_info.get("period") or (60.0 / tempo))
    # Normalise the anchor into [0, period) and re-derive k0 against it, so the
    # trim below cannot come out negative when a refit leaves grid_t0 a hair
    # under zero. Any (t0, k0) pair with downbeat_sec == t0 + k0 * period gives
    # the same answer, so this is free.
    t0 = float(grid_info.get("grid_t0", grid_info["downbeat_sec"])) % period
    k0 = int(round((float(grid_info["downbeat_sec"]) - t0) / period))

    trim_sec = t0 + (k0 % 4) * period
    anchor = k0 - (k0 % 4)                 # beats from the trimmed clip start
    pos = (start_bar - 1) * 4.0
    return pos, trim_sec, pos + anchor


def section_grid(grid_info, no_align):
    """The grid every section consumer gets: locators, energy map, session view.

    One dict so the three cannot disagree. The downbeat is what makes their
    phrases start on a bar line; with --no-align the file start is left where
    it is and there is no bar line to snap to, so it is dropped.
    """
    return {"period": grid_info["period"],
            "grid_t0": grid_info["grid_t0"],
            "downbeat_sec": None if no_align else grid_info["downbeat_sec"]}


def energy_map_shift(sections, trim_sec, period):
    """Beats to move energy-map notes from sample grid beats to clip beats.

    Sections count beats from their own anchor (re-anchored inside the audio
    when grid_t0 < 0, see sections.analyze_sections); the clip starts `trim_sec`
    into the sample. When aligned that difference is a whole number of beats:
    minus the lead-in up to the downbeat. Unaligned (no trim) there is no shift.
    """
    if not trim_sec:
        return 0
    anchor = sections[0]["time_sec"] - sections[0]["beat"] * period
    return round((anchor - trim_sec) / period)


def sample_time_to_beats(sec, warp_map, period):
    """Where `sec` seconds from the sample start sits in a WARPED clip's own
    beat time.

    Live reads a warped clip's positions off its warp map, not off the set
    tempo, so `sec / period` is only right when the two agree. Markers are
    [beat_time, sample_time]; this interpolates between the bracketing pair and
    extrapolates off the nearest pair outside them, which is what Live does with
    audio past the outer markers.

    Falls back to the uniform grid when there is no map to read — a clip Live
    auto-warped on import, where our own map is empty.
    """
    if len(warp_map or []) < 2:
        return sec / period
    lo, hi = warp_map[0], warp_map[1]
    for i in range(len(warp_map) - 1):
        if warp_map[i][1] <= sec <= warp_map[i + 1][1]:
            lo, hi = warp_map[i], warp_map[i + 1]
            break
    else:
        if sec > warp_map[-1][1]:
            lo, hi = warp_map[-2], warp_map[-1]
    span = hi[1] - lo[1]
    if span <= 0:
        return sec / period
    return lo[0] + (sec - lo[1]) * (hi[0] - lo[0]) / span


def trim_clip_start(track_index, clip_index, trim_sec, period, warp_map):
    """Drop `trim_sec` of lead-in off the front of a clip. (ok, detail).

    `start_marker` IS SILENTLY IGNORED UNLESS THE LOOP IS SET FIRST. Setting it
    on its own answers ok=True and reads back 0.0. The sequence that sticks is
    looping=True -> loop_start -> start_marker -> looping=False; the trim then
    persists with looping back off, `start_time` stays where it was put, and
    `length` has shrunk by exactly the trim.

    UNITS DEPEND ON `warping`: an unwarped clip takes SECONDS, a warped one takes
    BEATS, and the same trim reads 0.237 or 0.4938 accordingly. `warping` is read
    off the clip rather than assumed — Live's own auto-warp-on-import can have
    turned it on behind us, and guessing wrong here is silent.
    """
    cpath = "live_set tracks %d arrangement_clips %d" % (track_index, clip_index)
    try:
        warping = bool(lom_get(cpath, "warping"))
    except RuntimeError as e:
        return False, str(e)
    value = (sample_time_to_beats(trim_sec, warp_map, period) if warping
             else trim_sec)
    unit = "beats" if warping else "s"

    for prop, val in (("looping", True), ("loop_start", value),
                      ("start_marker", value), ("looping", False)):
        r = lom("set", path=cpath, property=prop, value=val)
        if not r.get("ok"):
            return False, "%s: %s" % (prop, lom_err(r))

    # Read back, because the failure this guards against is SILENT: without the
    # loop dance above the write is accepted and the marker stays at 0.0.
    try:
        got = float(lom_get(cpath, "start_marker"))
    except RuntimeError as e:
        return False, str(e)
    if abs(got - value) > 1e-3:
        return False, ("start_marker did not take — asked %.4f %s, reads %.4f"
                       % (value, unit, got))
    return True, "trim %.4f %s" % (value, unit)


def settle(time):
    """Move the play head and let Live actually get there. NOT redundant.

    Locators can only be created or deleted at the play head — Live's only API
    for it is `set_or_delete_cue()`, which takes no position. Setting
    `current_song_time` in the same main-thread task as the toggle is not enough:
    Live has not finished moving by the time the toggle runs, so it fires at the
    PREVIOUS position. Measured: nine locators produced cues at the wrong times,
    names attached one section behind, a stray cue at the play head left over
    from clip placement, and — because the toggle deletes when it lands on an
    existing cue — four of the nine silently missing.

    Issuing the move as its OWN bridge round trip is what fixes it: each request
    is a separate main-thread task, so Live has settled before the next one runs.
    With this, all nine land exactly and clearing goes 10 -> 0 instead of leaving
    stragglers behind.

    This is also why the transport must be STOPPED first (see stop_transport):
    while Live is playing the play head keeps moving after the set, and asking
    for 600.0 reads back 601.48.

    Returns the reply so the caller can check it. A move that FAILED — asking for
    a time past song_length raises "Cannot set the Songtime behind the
    Songlength" — must not be followed by a toggle, which would then fire at the
    previous position and delete whatever locator is sitting there.
    """
    return lom("set", path="live_set", property="current_song_time", value=time)


def stop_transport():
    """Stop playback if it is running, so the play head stays where put.
    Returns whether it had to stop anything."""
    if not lom_get("live_set", "is_playing"):
        return False
    lom("call", path="live_set", function="stop_playing")
    return True


def cue_point_list():
    """Current locators, in Live's own order, as {index, time, name} dicts."""
    try:
        n = lom_count("live_set", "cue_points")
    except RuntimeError:
        return []
    out = []
    for i in range(n):
        p = "live_set cue_points %d" % i
        try:
            out.append({"index": i, "time": float(lom_get(p, "time")),
                        "name": lom_get(p, "name")})
        except RuntimeError:
            continue
    return out


def toggle_cue(time):
    """Fire `set_or_delete_cue` at `time`, in two round trips. See settle().

    Returns the reply envelope. The op is a TOGGLE with no position argument:
    at a time that already holds a locator it DELETES, anywhere else it CREATES.
    Both the placing and the clearing paths below go through here.
    """
    moved = settle(time)
    if not moved.get("ok"):
        return moved
    return lom("call", path="live_set", function="set_or_delete_cue")


def create_cue_point(time, name):
    """Place a named locator, or rename the one already there. (ok, message).

    Three things this has to get right, all of them measured against Live 12.2.7
    while inserting nine section locators. The old AbletonMCP path did them
    inside a patched remote-script handler; the LOM bridge is generic, so they
    live here now.

    1. NEVER TOGGLE ONTO AN EXISTING LOCATOR. `set_or_delete_cue` is a toggle, so
       asking for a locator where one already sits would silently DELETE it.
       Look first and rename in place instead.
    2. LIVE SNAPS `current_song_time` to its own grid, so the cue does not land
       on the requested time. Everything below therefore works off the ACTUAL
       position read back from Live, and the new cue is identified by DIFFING the
       locator list — which does not care where Live actually put it. Matching on
       the requested time instead is what left all nine locators named with
       Live's defaults "1".."9".
    3. A CUE CANNOT BE CREATED PAST song_length — Live raises "Cannot set the
       Songtime behind the Songlength" — and the move failing is worse than
       useless, because the toggle would then fire wherever the play head still
       is. Five of the nine failed this way against an empty arrangement, hence
       the guard and its message: place the clips FIRST, they are what extends
       the song.
    """
    try:
        song_length = float(lom_get("live_set", "song_length"))
    except RuntimeError:
        song_length = None
    if song_length is not None and time > song_length:
        return False, ("cue point at %.3f is past song_length %.3f — place the "
                       "arrangement clips FIRST, they are what extends the song"
                       % (time, song_length))

    moved = settle(time)
    if not moved.get("ok"):
        return False, lom_err(moved)
    # Its own round trip, so Live has finished moving before anything reads the
    # position or toggles at it.
    actual = round(float(lom_get("live_set", "current_song_time")), 6)

    here = [c for c in cue_point_list() if round(c["time"], 6) == actual]
    if here:
        idx = here[0]["index"]                 # rename; do NOT toggle it away
    else:
        before = {round(c["time"], 6) for c in cue_point_list()}
        r = lom("call", path="live_set", function="set_or_delete_cue")
        if not r.get("ok"):
            return False, lom_err(r)
        new = [c for c in cue_point_list() if round(c["time"], 6) not in before]
        if not new:
            return False, ("set_or_delete_cue produced no new locator at %.3f"
                           % actual)
        idx = new[0]["index"]

    r = lom("set", path="live_set cue_points %d" % idx, property="name",
            value=name)
    if not r.get("ok"):
        return False, lom_err(r)
    return True, ""


def band_db(path, lo, hi):
    """Mean level in a band, via ffmpeg's volumedetect. ONE band, ONE full decode.

    Superseded by band_levels(), which gets every band from a single ffmpeg
    process. Kept as the path that needs nothing but a plain `ffmpeg -af`, for
    an ffmpeg too old for the filter graph band_levels() builds.
    """
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path,
         "-af", "highpass=f=%d,lowpass=f=%d,volumedetect" % (lo, hi),
         "-f", "null", "/dev/null"],
        capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    return -120.0


def band_levels(path):
    """Every band's mean level in dB from ONE decode of the file.

    band_db() costs a full decode PER BAND — seven ffmpeg processes per stem, 70
    for a ten-stem folder, 17.0 s on the test set, and the largest avoidable
    cost left in the pipeline. This splits the decoded stream into one chain per
    band inside a single ffmpeg process instead.

    Same `highpass,lowpass,volumedetect` chain per band as before, so the
    numbers are IDENTICAL rather than merely close. That matters: the centroid
    drives the clip colour ramp and the track ordering, and a "cleaner"
    reimplementation would silently renumber both. An in-process scipy version
    was tried and is 2x SLOWER than the seven ffmpeg calls it replaces —
    `sosfilt` is a serial IIR over 54M samples and cannot touch ffmpeg's C.

    Chains are read back by their `Parsed_volumedetect_N` id, sorted ascending.
    ffmpeg numbers filters in graph order, so the Nth volumedetect is the Nth
    band; the ids are parsed rather than computed so a change to how ffmpeg
    bases its numbering cannot silently transpose the bands.
    """
    n = len(BANDS)
    chains = ["[0:a]asplit=%d%s" % (n, "".join("[a%d]" % i for i in range(n)))]
    for i, (lo, hi) in enumerate(BANDS):
        chains.append("[a%d]highpass=f=%d,lowpass=f=%d,volumedetect[o%d]"
                      % (i, lo, hi, i))
    cmd = ["ffmpeg", "-nostdin", "-i", path, "-filter_complex", ";".join(chains)]
    for i in range(n):
        cmd += ["-map", "[o%d]" % i, "-f", "null", "/dev/null"]

    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    found = {}
    for line in err.splitlines():
        if "mean_volume:" in line and "Parsed_volumedetect_" in line:
            idx = int(line.split("Parsed_volumedetect_")[1].split(" ")[0].split("@")[0])
            found[idx] = float(line.split("mean_volume:")[1].split("dB")[0])
    if len(found) != n:
        raise RuntimeError("expected %d volumedetect readings, got %d"
                           % (n, len(found)))
    return [found[k] for k in sorted(found)]


def centroid(path):
    """Log-frequency centroid weighted by linear band energy."""
    try:
        levels = band_levels(path)
    except Exception as e:
        # An ffmpeg too old for this filter graph, or a stream it will not
        # asplit: fall back to the one-decode-per-band route, which is slow but
        # has no filter_complex dependency.
        sys.stderr.write("one-pass band levels failed for %s (%s); "
                         "using per-band decode\n" % (os.path.basename(path), e))
        levels = [band_db(path, lo, hi) for lo, hi in BANDS]
    energy = [10 ** (db / 10) for db in levels]
    total = sum(energy)
    if total <= 0:
        return CENTRES[0]
    return math.exp(sum(math.log(f) * e for f, e in zip(CENTRES, energy)) / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems_dir")
    ap.add_argument("--tempo", type=float, default=None,
                    help="override the measured tempo (a rounded integer here "
                         "will drift against the audio)")
    ap.add_argument("--no-align", action="store_true",
                    help="place clips at the raw file start instead of nudging "
                         "them onto a bar line")
    ap.add_argument("--detector", default="heuristic",
                    choices=("heuristic", "madmom"),
                    help="downbeat backend passed to beat_aligner. heuristic "
                         "(default) needs nothing extra and is ~4x faster; the "
                         "arrangement-boundary tiebreaker is what actually "
                         "settles bar 1. madmom is the trained tracker and is "
                         "worth trying on material the heuristic is not tuned "
                         "for")
    ap.add_argument("--start-bar", type=int, default=9,
                    help="bar the clips start on (default 9, leaving eight empty "
                         "bars of room ahead of the audio to mix into). The "
                         "lead-in ahead of the first grid beat is trimmed off the "
                         "front of each clip instead of offsetting its position, "
                         "so the clip begins exactly on the bar line. Use 1 to "
                         "butt the audio against the start of the arrangement")
    ap.add_argument("--headroom", type=headroom_arg, default=6.0,
                    help="dB to pull every stem track down by, so the mix has "
                         "somewhere to go (default 6). A YouTube master is "
                         "brickwalled at 0 dBFS and the separated stems sum back "
                         "to roughly that, so at unity any EQ or compression you "
                         "add clips immediately. Use 0 to leave faders alone")
    ap.add_argument("--first-track", type=int, default=2,
                    help="index of the first track to write to (default 2, "
                         "leaving Live's two default MIDI tracks alone)")
    ap.add_argument("--locators", action="store_true",
                    help="drop a Live locator at each detected section start "
                         "(intro / build / drop / breakdown), snapped to 8-bar "
                         "phrase boundaries")
    ap.add_argument("--energy-map", action="store_true",
                    help="add a silent MIDI track at the TOP of the set whose "
                         "piano roll is the arrangement: one row per stem, one "
                         "note per section it plays in, velocity carrying how "
                         "hard. No instrument, so it makes no sound — it is a "
                         "map you read against the locators")
    ap.add_argument("--session", action="store_true",
                    help="also build a Session View grid: one scene per detected "
                         "section, one looping clip per stem per section. Runs "
                         "after the arrangement clips, and reuses the SAME beat "
                         "grid, so the two views cannot drift apart")
    ap.add_argument("--keep-locators", action="store_true",
                    help="with --locators, add to the existing locators instead "
                         "of clearing them first")
    args = ap.parse_args()
    if args.start_bar < 1:
        sys.exit("--start-bar is a bar number and Live counts from 1, got %d"
                 % args.start_bar)

    # Stems are FLAC except the merged synths (float32 WAV), so match both or
    # a FLAC run finds nothing at all. Keep a name->path map; extensions differ.
    # ABSOLUTE paths, always. Live resolves file_path against ITS OWN working
    # directory, not the shell's, so a relative stems_dir makes the bridge reject
    # every clip with "Please provide an absolute path" — and the run still gets
    # as far as creating the tracks and setting the tempo, so it looks like a
    # partial success rather than a path problem.
    paths = {}
    for f in sorted(os.listdir(args.stems_dir)):
        if f.endswith((".flac", ".wav")):
            paths.setdefault(os.path.splitext(f)[0],
                             os.path.abspath(os.path.join(args.stems_dir, f)))
    stems = sorted(paths)
    if not stems:
        sys.exit("no .wav/.flac files in %s" % args.stems_dir)

    warp_ok = probe_bridge()

    # PICK AUDIO TRACKS BY TYPE, NEVER BY COUNTING. Counting was the old rule —
    # "if there are fewer than first_track + len(stems) tracks, append some" —
    # and it is wrong on any Set that is not audio tracks all the way down.
    #
    # Measured on a real Set: tracks 2, 5, 6 and 7 were the user's MIDI tracks.
    # The count was sufficient, so nothing was created; the loop then RENAMED
    # those four to stem names and only afterwards discovered that
    # `Audio clips can only be created on audio tracks`. Four MIDI tracks lost
    # their names for nothing. Renaming before knowing the track is usable is
    # what turned a clean failure into data loss.
    targets = audio_track_targets(args.first_track, len(stems))

    # Joint beat & downbeat alignment engine. analyze_alignment returns a dict
    # or raises, so past this block grid_info is always a fitted grid.
    try:
        from beat_aligner import analyze_alignment, phase_verdict
        grid_info = analyze_alignment(
            args.stems_dir, override_tempo=args.tempo, detector=args.detector,
)
    except ImportError as e:
        # ONLY a genuinely absent beat_aligner reaches this. Catching every
        # Exception here once hid a real bug: a signature mismatch raised
        # TypeError, was swallowed, and every run quietly used a cruder
        # aubiotrack fallback — losing the comb-phase tempo refit, outlier
        # rejection and drift analysis, with nothing in the output to say so.
        # That fallback is now deleted and this is fatal instead: beat_aligner
        # is a sibling module of this script, so its absence is a broken
        # install, not a condition worth working around.
        sys.exit("beat_aligner unavailable (%s); alignment cannot run" % e)

    tempo = args.tempo if args.tempo else grid_info["bpm"]
    if tempo:
        lom("set", path="live_set", property="tempo", value=round(tempo, 3))
        print("tempo -> %.3f BPM" % round(tempo, 3))

    # The period the SET tempo runs at. Every seconds<->beats conversion below
    # goes through this one value, so the clip position, the trim and the locator
    # times cannot end up disagreeing about how long a beat is.
    period = float(grid_info["period"])

    # One position and one trim for every clip, so relative sync between the
    # stems is untouched — they are cut from one source and must stay
    # sample-locked to each other.
    START = 0.0          # arrangement beat the clips are placed at
    TRIM = 0.0           # seconds of lead-in taken off the front of each clip
    if args.no_align and args.start_bar != 1:
        print("--no-align places clips at the raw file start, so --start-bar %d "
              "is ignored" % args.start_bar)
    if not args.no_align and tempo:
        START, TRIM, db_beat = bar_aligned_placement(
            grid_info, args.start_bar, tempo)
        trim_beats = TRIM / period
        ref = grid_info["downbeat_sec"]
        stem_name = grid_info["primary_stem"]
        target = "onto a bar line"
        print("aligning: clips at bar %.4f (beat %.4f), lead-in trimmed "
              "%.4fs (%.4f beats)"
              % (START / 4.0 + 1.0, START, TRIM, trim_beats))
        print("  downbeat at %.3fs lands on bar %.4f (%s)"
              % (ref, db_beat / 4.0 + 1.0, target))
        # THRESHOLD IS ONE BEAT, NOT ONE BAR. The trim can never reach a bar:
        # grid_t0 is folded into [0, period) and at most 3 further beats of bar
        # phase are added, so it is bounded by 4 beats by construction. The
        # fraction below one beat is the lead-in ahead of the file's first grid
        # beat and is silence by definition. Anything ABOVE one beat is whole
        # beats of real audio being deleted to bring the anchor onto a bar line
        # — which is exactly where a track with a pickup or an upbeat loses it.
        if trim_beats > 1.0:
            print("  WARNING: the trim is %.2f beats, so %d whole beat(s) of "
                  "AUDIO are dropped\n"
                  "           from the front of every clip, not just the "
                  "sub-beat lead-in. A\n"
                  "           pickup or upbeat there would be lost. Check the "
                  "start by ear;\n"
                  "           --no-align keeps the raw file start."
                  % (trim_beats, int(trim_beats)))
        print("  tempo from %s; phase from %s via %s"
              % (stem_name, grid_info.get("phase_source", "?"),
                 grid_info.get("detector", "?")))
        # Surfaced because an unreliable phase means the bar line is a guess and
        # every clip may sit a whole number of beats out — worth knowing BEFORE
        # you start editing rather than after. The verdict comes from
        # beat_aligner so the two cannot disagree about what counts as usable;
        # each detector family has its own ceiling, which is not obvious from
        # the raw number.
        conf = grid_info["downbeat_confidence"]
        verdict, ceiling = phase_verdict(conf, grid_info.get("detector"))
        if verdict == "ok":
            print("  phase confidence %.3f (%s; 0.25 = chance, %.2f = ceiling)"
                  % (conf, verdict, ceiling))
        else:
            print("  WARNING: phase confidence %.3f is %s — BAR 1 IS "
                  "UNRELIABLE. The tempo is sound, but clips may sit 1-3 "
                  "beats off the bar line. Check bar 1 by ear, or try "
                  "--detector madmom" % (conf, verdict))
        if grid_info.get("has_drift"):
            print("tempo drift detected (spread %.4f%%) — a warp map will be "
                  "written after the clips are placed"
                  % grid_info.get("drift_pct", 0.0))
        else:
            print("no tempo drift — clips stay unwarped on Live's fixed grid")

    # Sections, locators, scenes and the energy map snap to phrases counted
    # from the downbeat in this grid.
    sgrid = section_grid(grid_info, args.no_align)

    if args.headroom:
        print("headroom: every stem track pulled to %.1f dB (fader %.3f) — the "
              "source is brickwalled at 0 dBFS,\n"
              "          so at unity the stems sum back to it with nothing left "
              "for your own processing"
              % (-abs(args.headroom), db_to_fader(-abs(args.headroom))))

    print("measuring spectral centroids...")
    # Stems are independent measurements and each one is an ffmpeg process that
    # spends its life outside the GIL, so threads get real parallelism here.
    # Capped at 6 — measured on a machine with 6 performance cores, where the
    # extra efficiency cores just added scheduling noise to a CPU-bound filter
    # graph. Sequentially this step was the largest cost left in the pipeline.
    with futures.ThreadPoolExecutor(max_workers=min(6, len(stems))) as pool:
        cent = dict(zip(stems, pool.map(lambda n: centroid(paths[n]), stems)))

    order = arrange_stems(stems, cent)

    # `drums` is the composite of the drums_* parts — the parts sum back to it.
    # Loading both plays the kit twice (~+6 dB), so mute the composite whenever
    # the parts are present. The clip still loads, so you can unmute to A/B.
    parts = [n for n in order if n.startswith("drums_")]
    muted = {"drums"} if parts and "drums" in order else set()

    lo = math.log(min(cent[n] for n in order))
    hi = math.log(max(cent[n] for n in order))
    span = (hi - lo) or 1.0

    # The same map goes on every clip: the stems are sample-identical in length
    # and cut from one source, so they drift together or not at all.
    warp_map = grid_info.get("warp_map") or []
    warp_state = None if warp_ok else "unsupported"
    stale = []           # tracks left carrying clips from an earlier run

    print("%3s  %-14s %9s   colour" % ("trk", "stem", "centroid"))
    for i, name in enumerate(order):
        ti = targets[i]
        t = (math.log(cent[name]) - lo) / span
        r, g, b = colorsys.hsv_to_rgb(t * (240 / 360), 0.85, 1.0)
        rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)

        # NO delete step. Track.delete_clip() takes a Clip OBJECT, which no JSON
        # argument can express, so the bridge cannot reach it. Creating a clip at
        # the same position REPLACES the one there, but only across the span the
        # new clip covers — place at a DIFFERENT position and the old clip's head
        # survives in front of the new one. Hence find_placed_clip below.
        clip = lom("call", path="live_set tracks %d" % ti,
                   function="create_audio_clip", args=[paths[name], START])

        # RENAME ONLY ONCE THE CLIP IS ON THE TRACK. The rename used to come
        # first, so a track that could not take the audio was relabelled anyway
        # — four of a user's MIDI tracks lost their names to stems that then
        # failed to land. A failed placement must leave the Set as it found it.
        if not clip.get("ok"):
            print("%3d  %-14s  ERROR: %s — track left untouched"
                  % (ti, name, lom_err(clip)))
            continue
        lom("set", path="live_set tracks %d" % ti, property="name", value=name)
        if args.headroom:
            ok_hr, hr_detail = set_track_headroom(ti, -abs(args.headroom))
            if not ok_hr:
                print("%3d  %-14s  headroom not applied: %s" % (ti, name, hr_detail))

        # Resolve the clip we just made BY ITS START TIME. Never index 0 — see
        # find_placed_clip. Everything after this point addresses `ci`.
        try:
            clips = arrangement_clips(ti)
        except RuntimeError as e:
            print("%3d  %-14s  ERROR: could not list arrangement clips (%s) — "
                  "colour, mute, warp and trim SKIPPED for this track"
                  % (ti, name, e))
            continue
        ci = find_placed_clip(clips, START)
        if ci is None:
            # Not a silent fallback to 0: acting on the wrong clip is what this
            # whole path exists to prevent.
            print("%3d  %-14s  ERROR: no clip starts at beat %.4f (found %s) — "
                  "colour, mute, warp and trim SKIPPED for this track"
                  % (ti, name, START,
                     ", ".join("%.4f" % s for _i, s, _e in clips) or "none"))
            continue

        cpath = "live_set tracks %d arrangement_clips %d" % (ti, ci)
        col = lom("set", path=cpath, property="color", value=rgb)
        # Colour the TRACK to match its clip. The module docstring used to say
        # only clip colour was settable — true of the AbletonMCP bridge, which
        # had no handler for it, and false here: LOM reaches Track.color
        # directly (verified: 11958214 -> 16725558). Without it the track
        # headers stay Live's default palette while the clips carry the ramp,
        # so the frequency ordering is invisible until you look at the clips.
        lom("set", path="live_set tracks %d" % ti, property="color", value=rgb)
        if name in muted:
            lom("set", path=cpath, property="muted", value=True)

        # Leftovers are AUDIBLE DUPLICATE AUDIO, not a cosmetic wart: the stale
        # clip still plays. Collected and reported loudly after the table.
        leftovers = [(s, e) for j, s, e in clips if j != ci]
        if leftovers:
            stale.append((ti, name, leftovers))

        # Warp only a drifting set, and stop trying after the bridge says it has
        # never heard of the command — that answer will not change mid-run.
        warp_note = ""
        if warp_map and warp_state != "unsupported":
            warp_state, detail = apply_warp_map(ti, ci, warp_map)
            if warp_state == "ok":
                warp_note = "  " + detail
            elif warp_state == "unsupported":
                warp_note = "  [no warp]"
            else:
                warp_note = "  [warp failed: %s]" % detail

        # AFTER the warp map, never before: the trim's unit is seconds on an
        # unwarped clip and beats on a warped one, so `warping` has to have
        # settled before trim_clip_start reads it.
        trim_note = ""
        if TRIM > 0:
            ok, detail = trim_clip_start(ti, ci, TRIM, period, warp_map)
            trim_note = "  " + detail if ok else "  [trim failed: %s]" % detail

        print("%3d  %-14s %8.0fHz   #%06X  clip=%s col=%s%s%s%s%s"
              % (ti, name, cent[name], rgb,
                 status(clip), status(col),
                 "  [muted: parts present]" if name in muted else "",
                 warp_note, trim_note,
                 "  [+%d STALE]" % len(leftovers) if leftovers else ""))

    # STALE CLIPS ARE AUDIBLE, so this is a hard warning rather than a note. It
    # happens when a Set is re-run at a different --start-bar: the new clip
    # replaces only the span it covers, and the old clip's head plays on in front
    # of it as duplicate audio.
    if stale:
        print("\n" + "=" * 72)
        print("STALE AUDIO ON %d TRACK(S) — THESE STILL PLAY. MANUAL CLEANUP "
              "NEEDED." % len(stale))
        print("=" * 72)
        for ti, name, leftovers in stale:
            for start, end in leftovers:
                where = ("bar %.4f -> %.4f" % (start / 4.0 + 1.0,
                                               end / 4.0 + 1.0)
                         if end is not None else "bar %.4f" % (start / 4.0 + 1.0))
                print("  track %-3d %-14s leftover clip at %s" % (ti, name, where))
        print("These are clips from an EARLIER run at a different position. The "
              "new clips are\n"
              "correct — coloured, trimmed and aligned — but the leftovers "
              "overlap them and\n"
              "will sound as doubled audio. Delete them by hand in the "
              "arrangement.\n"
              "This script cannot: Track.delete_clip() takes a Clip object, "
              "which no JSON\n"
              "argument can express, so the LOM bridge has no way to reach it.")

    # LOCATORS LAST, AND THIS ORDERING IS LOAD-BEARING. A cue point cannot be
    # created past song_length — Live raises "Cannot set the Songtime behind the
    # Songlength" — and song_length is whatever the arrangement currently
    # reaches. Running this before the clips are placed silently loses every
    # locator past the end of an empty arrangement; measured on a real run, five
    # of nine went missing that way.
    if args.locators:
        print()
        # BEFORE ANYTHING TOUCHES THE PLAY HEAD. Locators are placed at the play
        # head, and a running transport keeps moving it after settle() sets it —
        # measured: asking for 600.0 read back 601.48, and every locator landed
        # wrong.
        if stop_transport():
            print("locators: transport was running — stopped playback so the "
                  "play head stays where it is put")
        try:
            from sections import detect_sections, sections_to_locators
            # Reuse the grid we already fitted rather than re-deriving it, so the
            # locators cannot disagree with the tempo the clips were placed at.
            secs = detect_sections(args.stems_dir, **sgrid)
        except Exception as e:
            secs = None
            print("locators: section detection failed (%s: %s)"
                  % (type(e).__name__, e))

        if secs:
            if not args.keep_locators:
                removed = 0
                # Bounded: a Set with hundreds of locators is not ours to churn
                # through, and an unbounded loop here would hang on any cue that
                # refuses to delete.
                #
                # There is no delete-by-time op — Live 12 has no such method at
                # all. Deleting means parking the play head ON the locator and
                # letting `set_or_delete_cue` toggle it away. The count is
                # rechecked every pass, because the same toggle CREATES a locator
                # wherever it lands on empty ground: if it did not go down, stop
                # rather than sprinkle stray cues through the Set.
                existing = cue_point_list()
                for _ in range(64):
                    if not existing:
                        break
                    if not toggle_cue(float(existing[0]["time"])).get("ok"):
                        break
                    after = cue_point_list()
                    if len(after) >= len(existing):
                        break
                    removed += 1
                    existing = after
                print("locators: cleared %d existing" % removed)

            placed = failed = 0
            # Sections are measured from the start of the SAMPLE, and the sample
            # no longer starts where the clip does: the first TRIM seconds were
            # cut off the front. So the sample's time 0 sits `TRIM / period`
            # beats BEFORE the clip, and the locator offset has to say so or
            # every marker lands late by the trim.
            offset = START - TRIM / period
            clamped = 0
            for loc in sections_to_locators(secs, period, offset):
                # The offset can be NEGATIVE now — the sample's time 0 sits just
                # before the clip, in the audio that was trimmed away — and Live
                # has no song time before bar 1. A section boundary inside the
                # trimmed lead-in is pinned to bar 1 rather than lost.
                t = loc["time"]
                if t < 0.0:
                    t, clamped = 0.0, clamped + 1
                ok, msg = create_cue_point(t, loc["name"])
                if ok:
                    placed += 1
                else:
                    failed += 1
                    if failed == 1:      # first failure only; the rest rhyme
                        print("locators: %s" % msg)
            print("locators: placed %d of %d section markers"
                  % (placed, placed + failed))
            if clamped:
                print("           %d fell inside the trimmed lead-in and were "
                      "pinned to bar 1" % clamped)
            # Boundaries are measured; the names are rules over energy and
            # order. Saying so is cheaper than a user trusting "Drop" too far.
            print("           boundaries measured from spectral change; names "
                  "are energy/order rules")

    # ENERGY MAP before the session grid, because it inserts a track at index 0
    # and every index after it shifts by one. Doing it here means the session
    # step re-reads track names afterwards and maps onto the new indices; doing
    # it last would be fine too, but doing it BETWEEN the arrangement loop and
    # anything holding a track index would not.
    if args.energy_map:
        print()
        try:
            from sections import (detect_sections, detect_transitions,
                                  energy_notes, macro_energy_notes,
                                  section_energy, transition_notes)
            secs = detect_sections(args.stems_dir, **sgrid)
            energy = section_energy(args.stems_dir, secs)
            notes, rows = energy_notes(energy, secs,
                                       order=[n for n in order if n in energy])
            notes = macro_energy_notes(energy, secs) + notes
            # Transitions sit above the macro curve: two more pitches, and the
            # gesture is the thing you scan for when deciding where to cut.
            try:
                trans = detect_transitions(
                    args.stems_dir, secs,
                    grid_info["period"])
                notes = transition_notes(trans) + notes
                if trans:
                    print("transitions: %s"
                          % ", ".join("%s at bar %d (x%.1f into %s)"
                                      % (t["kind"], t["bar"], t["ratio"], t["into"])
                                      for t in trans))
            except Exception as e:
                print("transition detection skipped (%s: %s)" % (type(e).__name__, e))
            write_energy_map(notes, rows, secs, START,
                             shift_beats=energy_map_shift(secs, TRIM, period))
        except Exception as e:
            print("energy map failed (%s: %s) — nothing else is affected"
                  % (type(e).__name__, e))

    # SESSION VIEW LAST, and that ordering is not cosmetic. It maps stems onto
    # tracks BY NAME, so the tracks have to exist and be named first — which is
    # what the loop above just did.
    #
    # `period` and `grid_t0` are handed over rather than re-derived. Left to fit
    # its own grid, session_view lands within a beat or so of this one, which is
    # exactly bad enough: the arrangement and session clips then disagree by a
    # fraction of a beat that grows with the section index, and a set that looks
    # right in both views is quietly out of phase between them.
    if args.session:
        print()
        try:
            from session_view import build_session_view
            build_session_view(args.stems_dir, replace=True, port=PORT,
                               **sgrid)
        except Exception as e:
            # The arrangement is already placed and usable; a failed grid is a
            # missing extra, not a reason to look like the whole run failed.
            print("session view failed (%s: %s) — the arrangement is unaffected"
                  % (type(e).__name__, e))

    # After the table, so it is the last thing on screen rather than buried in
    # it. The clips are placed and usable — only the drift correction is missing.
    if warp_map and warp_state == "unsupported":
        print("\nWARNING: this bridge's handlers.py has no `warp_markers_set` "
              "op, so no warp map\n"
              "was written and the set will drift against the grid.\n"
              "Copy an updated handlers.py into the AbletonLOM remote script and "
              "send\n"
              "{\"op\": \"reload\"} to port %d. NO Live restart is needed — the "
              "shell reimports\n"
              "handlers.py on demand, which is exactly what the old AbletonMCP "
              "path could not do." % PORT)


if __name__ == "__main__":
    main()

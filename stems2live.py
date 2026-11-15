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
127.0.0.1:9878). Talks to it through `sideman.client.Live`, Sideman's own Python
client, rather than going through the MCP server, so it works from a plain shell.

The bridge is generic: instead of one command per feature it exposes the Live
Object Model itself, so every call here is a `get`/`set`/`call`/`count` against
a space-separated LOM path such as `live_set tracks 3 arrangement_clips 0`.
They go out grouped, many to a `get_batch` or `transaction` request, because a
request costs Live the same 400 ms whatever it carries (see live_batch).

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

import os
import sys

# Re-exec with local .venv python if present: `sideman` (the Live client) and
# librosa are installed there, not in the system python the shebang finds.
_real_script = os.path.realpath(__file__)
_venv = os.path.join(os.path.dirname(_real_script), ".venv")
_venv_py = os.path.join(_venv, "bin", "python")
if os.path.realpath(sys.prefix) == os.path.realpath(_venv):
    os.environ.pop("ROADIE_REEXEC", None)    # arrived; later scripts may hop again
elif os.path.exists(_venv_py) and not os.environ.get("ROADIE_REEXEC"):
    os.environ["ROADIE_REEXEC"] = "1"        # one hop, whatever the venv looks like
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse, colorsys, math, subprocess
from concurrent import futures

try:
    from sideman.client import Live, LiveError
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] != "sideman":
        raise
    sys.exit("sideman is not installed in .venv: run ./install.sh")

import live_batch

PORT = 9878
LOM_TIMEOUT = 120.0
# geometric centres of the bands sampled for the centroid
BANDS = [(30, 80), (80, 200), (200, 500), (500, 1200),
         (1200, 3000), (3000, 7000), (7000, 16000)]
CENTRES = [math.sqrt(lo * hi) for lo, hi in BANDS]


_LIVE = None      # the connection this run holds: (PORT, LOM_TIMEOUT) it was opened for, and the Live


def _live():
    """The `sideman.client.Live` every request of this run goes through, holding ONE connection.

    A load is a few hundred requests and a held connection saves Live a tick on each (measured on a
    ten-stem Set: 389 ms a request against 603 ms connecting every time). `Live.__enter__` is what
    holds it; the connection closes with the process. A changed PORT or LOM_TIMEOUT gets a new one.
    """
    global _LIVE
    if _LIVE is None or _LIVE[0] != (PORT, LOM_TIMEOUT):
        if _LIVE is not None:
            _LIVE[1].close()
        _LIVE = ((PORT, LOM_TIMEOUT), Live("127.0.0.1", PORT, LOM_TIMEOUT).__enter__())
    return _LIVE[1]


def lom(op, **params):
    """One round trip to the AbletonLOM bridge. Returns the reply envelope.

    The wire is `sideman.client.Live.request`, on the one connection `_live()` holds. This
    keeps the envelope the callers below were written against, `{"ok": True,
    "result": ...}` or `{"ok": False, "error": {"type", "message"}}`, because
    they branch on `ok` and print the error instead of catching exceptions:
    Live's refusals come back as that envelope rather than being raised.

    `Live.request(op, params)` rather than its typed methods, on purpose:
    `Live.call` adds `"confirm": false` to every call and `args: []` to ones
    that sent none, which would change what goes out on the wire.

    Failures of the connection itself are raised, as they always were: nothing
    listening exits with the set-up instructions, a connection closed before a
    reply is a RuntimeError, and a reply that never comes is a TimeoutError.
    """
    try:
        return {"ok": True,
                "result": _live().request(op, params)}
    except LiveError as e:
        if e.type == "ConnectionRefused":
            sys.exit("Nothing listening on 127.0.0.1:%d.\n"
                     "Open Ableton Live and set Preferences -> Link, Tempo & MIDI ->\n"
                     "Control Surface = AbletonLOM (Input/Output = None)." % PORT)
        if e.type == "ConnectionClosed":
            raise RuntimeError("no reply from Live on port %d" % PORT)
        # Only a timeout waiting on the connection is raised; Live's own "Timeout" reply (its
        # main thread was blocked) is an error envelope like any other refusal,
        # which fails one operation, not the run.
        if e.type == "Timeout" and isinstance(e.__context__, TimeoutError):
            raise TimeoutError("no reply from Live on port %d within %gs"
                               % (PORT, LOM_TIMEOUT))
        # `str(e)` is "<type>: <message>": give back the message as Live sent it.
        return {"ok": False,
                "error": {"type": e.type, "message": str(e)[len(e.type) + 2:]}}


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


# Many operations in few requests: see live_batch for what may share a request and why. Each
# helper hands back the envelope `lom` would have returned for every op sent alone, so the
# callers branch and print exactly as they did. A read request that fails as a whole fails
# every read it carried, with its envelope; a write request that goes unanswered ends the run.

def _envelope(entry):
    """One op's entry in a transaction's results, as the envelope of that op sent alone."""
    if entry.get("ok"):
        return {"ok": True, "result": entry.get("result")}
    t, m = live_batch.split_error(entry.get("error"))
    return {"ok": False, "error": {"type": t, "message": m}}


def lom_reads(specs):
    """Read every {path, property} in `specs`: one envelope each, in order, as `lom("get")` gives.

    `get_batch` never fails as a whole, so an unreadable property costs only its own entry.
    """
    out = []
    for chunk in live_batch.pack([dict(s, op="get") for s in specs]):
        r = lom("get_batch", specs=[{"path": s["path"], "property": s["property"]}
                                    for s in chunk])
        if not r.get("ok"):
            out.extend([r] * len(chunk))
            continue
        for e in r["result"]["results"]:
            if e.get("ok"):
                out.append({"ok": True, "result": {k: v for k, v in e.items() if k != "ok"}})
            else:
                out.append(_envelope(e))
    return out


def _write_batch(ops, stop_on_error):
    """One `transaction`. A reply that never came, a closed connection, or Live's own Timeout
    reply ends the run (live_batch.StillRunning): Live keeps running a task it gave up waiting
    on, so these changes may still land, and none of them can be reported as not made."""
    try:
        r = lom("transaction", ops=ops, stop_on_error=stop_on_error, rollback_on_error=False)
    except (RuntimeError, TimeoutError) as e:
        raise live_batch.StillRunning(str(e))
    if not r.get("ok") and r["error"].get("type") == "Timeout":
        raise live_batch.StillRunning(lom_err(r))
    return r


def lom_ops(ops):
    """Run sets and calls in order, none stopping another: one envelope each.

    Every request is one `transaction`, so ONE undo step. `live_batch.seq` nests ops that stop at
    their first failure; `seq_results` unpacks its envelope.
    """
    out = []
    for chunk in live_batch.pack(ops):
        r = _write_batch(chunk, False)
        if not r.get("ok"):
            out.extend([r] * len(chunk))
            continue
        out.extend(_envelope(e) for e in r["result"]["results"])
    return out


def lom_steps(ops):
    """Run ops in order and stop at the first that fails. Envelopes of the ops that ran; when one
    failed it is the last."""
    out = []
    for chunk in live_batch.pack(ops):
        r = _write_batch(chunk, True)
        out.extend([r] if not r.get("ok")
                   else [_envelope(e) for e in r["result"]["results"]])
        if not out[-1].get("ok"):
            break
    return out


def seq_results(env):
    """The envelopes of the ops of a `live_batch.seq` that ran, in order."""
    if not env.get("ok"):
        return [env]
    return [_envelope(e) for e in env["result"]["results"]]


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


# Clips a track is read for before its count is known. A fresh track has one, a re-run at
# another --start-bar two; reads past the end fail on their own and change nothing.
CLIPS_READ_AHEAD = 4


def arrangement_clips(track_indices):
    """{track index: [(index, start_time, end_time)]} for each track's arrangement clips, or
    the RuntimeError that stopped reading that track.

    ARRANGEMENT CLIPS ARE INDEXED BY TIME, NOT BY CREATION ORDER. `end_time` is
    read best-effort and comes back None if Live will not give it — it is only
    used to describe leftovers in a message, never to decide anything.

    One request for every track: the clip list (its length is the count) and the times of the
    first CLIPS_READ_AHEAD clips; a second only for a track with more clips than that.
    """
    def times(ti, i):
        p = "live_set tracks %d arrangement_clips %d" % (ti, i)
        return [{"path": p, "property": "end_time"}, {"path": p, "property": "start_time"}]

    specs = []
    for ti in track_indices:
        specs.append({"path": "live_set tracks %d" % ti, "property": "arrangement_clips"})
        for i in range(CLIPS_READ_AHEAD):
            specs += times(ti, i)
    envs = iter(lom_reads(specs))
    counts, got = {}, {}
    for ti in track_indices:
        vec = next(envs)
        got[ti] = [(next(envs), next(envs)) for _ in range(CLIPS_READ_AHEAD)]
        if not vec.get("ok"):
            counts[ti] = RuntimeError("count live_set tracks %d.arrangement_clips: %s"
                                      % (ti, lom_err(vec)))
        else:
            counts[ti] = live_batch.vector_len(vec["result"]["value"])
    more = [(ti, i) for ti in track_indices if not isinstance(counts[ti], Exception)
            for i in range(CLIPS_READ_AHEAD, counts[ti])]
    envs = iter(lom_reads([s for ti, i in more for s in times(ti, i)]))
    for ti, _i in more:
        got[ti].append((next(envs), next(envs)))

    out = {}
    for ti in track_indices:
        if isinstance(counts[ti], Exception):
            out[ti] = counts[ti]
            continue
        clips = []
        for i in range(counts[ti]):
            p = "live_set tracks %d arrangement_clips %d" % (ti, i)
            end_r, start_r = got[ti][i]
            end = float(end_r["result"]["value"]) if end_r.get("ok") else None
            if not start_r.get("ok"):
                clips = RuntimeError("get %s.start_time: %s" % (p, lom_err(start_r)))
                break
            clips.append((i, float(start_r["result"]["value"]), end))
        out[ti] = clips
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


def track_headroom_op(track_index, db):
    """The set that pulls one track's fader down by `db`. Raises ValueError past the
    measured range (see db_to_fader).

    Applied per stem track rather than to the master on purpose: the point is to
    leave YOUR faders reading near unity while the SUM has room, so subsequent
    mix moves are relative to a sane starting point. Pulling the master down
    instead would leave every stem fader at unity and hide the headroom.
    """
    return {"op": "set", "path": "live_set tracks %d mixer_device volume" % track_index,
            "property": "value", "value": db_to_fader(db)}


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
    # Every name in one read; the deletes highest index first, so deleting one does not shift
    # the ones still to delete. A track's name does not change when a track above it goes, so
    # reading them all before any delete finds the same tracks the one-at-a-time loop did.
    n = lom_count("live_set", "tracks")
    names = lom_reads([{"path": "live_set tracks %d" % i, "property": "name"}
                       for i in range(n)])
    doomed = [i for i in reversed(range(n))
              if names[i].get("ok") and names[i]["result"]["value"] == name]
    deleted = lom_steps([{"op": "call", "path": "live_set", "function": "delete_track",
                          "args": [i]} for i in doomed])
    if deleted and not deleted[-1].get("ok"):
        print("energy map: could not replace the existing track %d: %s"
              % (doomed[len(deleted) - 1], lom_err(deleted[-1])))
        return
    r = lom("call", path="live_set", function="create_midi_track", args=[0])
    if not r.get("ok"):
        print("energy map: could not create the track: %s" % lom_err(r))
        return

    named = {"op": "set", "path": "live_set tracks 0", "property": "name", "value": name}
    # Span the whole arrangement so every section has somewhere to live.
    try:
        last = sections[-1]
        length = float(last["beat"]) + 64.0
    except Exception:
        lom_ops([named])         # a bad section list raises after the naming, as it always did
        raise
    # The new track is named and given its clip in the request AFTER the one that made it, and
    # the clip is named and filled in the one after that: neither is addressed by index in the
    # request that created it.
    _named, r = lom_ops([
        named,
        {"op": "call", "path": "live_set tracks 0", "function": "create_midi_clip",
         "args": [float(start_beat), length]}])
    if not r.get("ok"):
        print("energy map: could not create the clip: %s" % lom_err(r))
        return
    clip = "live_set tracks 0 arrangement_clips 0"
    _named, r = lom_ops([
        {"op": "set", "path": clip, "property": "name", "value": "arrangement energy"},
        {"op": "notes_add", "path": clip, "notes": notes}])
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


def is_audio_track(reply):
    """Can this track hold an audio clip? `reply` is its `has_audio_input` read.

    `has_audio_input` is the property Live exposes for it; a MIDI track answers
    False and raises nothing, so a missing/failed read is treated as "not safe
    to use" rather than assumed audio.
    """
    return bool(reply.get("ok")) and bool(reply["result"]["value"])


# Tracks read for their type before the count is known, past the ones a load needs: a few MIDI
# tracks in the way cost no second request.
TRACKS_READ_AHEAD = 8


def audio_track_targets(first_track, count):
    """`count` audio-track indices at or after `first_track`, creating as needed.

    Existing AUDIO tracks are reused in order; MIDI tracks in the range are
    stepped over, not renamed and not written to. If that leaves too few, audio
    tracks are appended until there are enough.

    The old rule counted tracks and appended when the TOTAL was short, which is
    a different question and gives the wrong answer whenever a Set mixes types:
    a Set with plenty of tracks, four of them MIDI, satisfied the count and then
    failed on every MIDI one — after renaming it.

    One request reads the track list (its length is the count) and the type of
    the first `count + TRACKS_READ_AHEAD` candidates; a second reads the rest
    only if those were not enough. Reading past the point where enough are found,
    or past the last track, changes nothing.
    """
    ahead = first_track + count + TRACKS_READ_AHEAD
    envs = lom_reads([{"path": "live_set", "property": "tracks"}]
                     + [{"path": "live_set tracks %d" % i, "property": "has_audio_input"}
                        for i in range(first_track, ahead)])
    if not envs[0].get("ok"):
        raise RuntimeError("count live_set.tracks: %s" % lom_err(envs[0]))
    total = live_batch.vector_len(envs[0]["result"]["value"])
    kinds = envs[1:1 + max(0, min(total, ahead) - first_track)]
    if total > ahead and sum(map(is_audio_track, kinds)) < count:
        kinds += lom_reads([{"path": "live_set tracks %d" % i, "property": "has_audio_input"}
                            for i in range(ahead, total)])
    targets, skipped = [], []
    for i, kind in zip(range(first_track, total), kinds):
        if len(targets) == count:
            break
        (targets if is_audio_track(kind) else skipped).append(i)

    if skipped:
        print("skipping %d non-audio track%s at %s — they cannot hold audio "
              "clips and are left untouched"
              % (len(skipped), "" if len(skipped) == 1 else "s",
                 ", ".join(str(i) for i in skipped)))

    missing = count - len(targets)
    if missing > 0:
        print("creating %d audio track%s (set has %d)"
              % (missing, "" if missing == 1 else "s", total))
        # All of them in one request, stopping at the first failure as the one-at-a-time loop
        # did. Index -1 appends, so they land at `total` onwards: the indices the loop's
        # count-minus-one after each create gave.
        made = lom_steps([{"op": "call", "path": "live_set",
                           "function": "create_audio_track", "args": [-1]}] * missing)
        if not made[-1].get("ok"):
            sys.exit("create_audio_track failed: %s" % lom_err(made[-1]))
        targets.extend(range(total, total + missing))
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


def warp_map_op(track_index, clip_index, markers):
    """The op that writes a warp map to a clip; warp_outcome reads its reply."""
    return {"op": "warp_markers_set",
            "path": "live_set tracks %d arrangement_clips %d" % (track_index, clip_index),
            "markers": markers}


def warp_outcome(r):
    """A warp map write's reply as (state, detail) for the caller to print.

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


def trim_clip_starts(clips, trim_sec, period, warp_map):
    """Drop `trim_sec` of lead-in off the front of each (track, clip index). One
    (ok, detail) per clip, in order.

    `start_marker` IS SILENTLY IGNORED UNLESS THE LOOP IS SET FIRST. Setting it
    on its own answers ok=True and reads back 0.0. The sequence that sticks is
    looping=True -> loop_start -> start_marker -> looping=False; the trim then
    persists with looping back off, `start_time` stays where it was put, and
    `length` has shrunk by exactly the trim.

    UNITS DEPEND ON `warping`: an unwarped clip takes SECONDS, a warped one takes
    BEATS, and the same trim reads 0.237 or 0.4938 accordingly. `warping` is read
    off the clip rather than assumed — Live's own auto-warp-on-import can have
    turned it on behind us, and guessing wrong here is silent.

    Each step below is one request for every clip, and the steps are separate
    requests, as they were when every set was a request of its own: Live has
    applied `looping` before any marker is written, the markers before
    `looping` goes off, and all of it before the read-back. A clip stops at its
    first failed step and is left out of the steps after it.
    """
    paths = ["live_set tracks %d arrangement_clips %d" % c for c in clips]
    out = [None] * len(paths)
    value, unwritten = {}, {}

    def in_unit(k, warping):
        value[k] = (sample_time_to_beats(trim_sec, warp_map, period) if warping
                    else trim_sec, "beats" if warping else "s", warping)

    for k, r in enumerate(lom_reads([{"path": p, "property": "warping"} for p in paths])):
        if not r.get("ok"):
            out[k] = (False, "get %s.warping: %s" % (paths[k], lom_err(r)))
            continue
        in_unit(k, bool(r["result"]["value"]))

    # LIVE CAN TURN `warping` ON BEHIND US after it was read: with no warp map, a fresh clip
    # reads unwarped, and Live's auto-warp lands once it has analysed the file, without
    # converting the markers. Measured 2026-11-15 on a 92 s stem: the trim went in as
    # 0.5662 s and the clip ended warped with start_marker 0.5662 beats, 0.3 s of lead-in
    # left (the code before batching did the same and reported success). So the read-back
    # re-reads `warping`, and a clip whose warping changed is trimmed again in the new unit;
    # it fails only if warping changes a second time.
    pending = sorted(value)
    for attempt in (1, 2):
        # (property, value) per step; None is the clip's own trim value. A set answers with
        # the property as it reads after the write, in the same task: one that reads other
        # than asked fails the trim at the end, but does not stop the steps, so the clip
        # ends up as the one-set-a-request code left it.
        for step in ((("looping", True),), (("loop_start", None), ("start_marker", None)),
                     (("looping", False),)):
            live = [k for k in pending if out[k] is None]
            ops = [live_batch.seq([{"op": "set", "path": paths[k], "property": prop,
                                    "value": value[k][0] if val is None else val}
                                   for prop, val in step]) for k in live]
            for k, env in zip(live, lom_ops(ops)):
                for (prop, val), r in zip(step, seq_results(env)):
                    if not r.get("ok"):
                        out[k] = (False, "%s: %s" % (prop, lom_err(r)))
                        break
                    asked, got = value[k][0] if val is None else val, r["result"].get("value")
                    if k not in unwritten and not _took(asked, got):
                        unwritten[k] = ("%s did not take — asked %s, reads %s"
                                        % (prop, _shown(asked, value[k][1]),
                                           _shown(got, value[k][1])))

        # Read back, because the failure this guards against is SILENT: without the
        # loop dance above the write is accepted and the marker stays at 0.0.
        live = [k for k in pending if out[k] is None]
        backs = iter(lom_reads([{"path": paths[k], "property": prop} for k in live
                                for prop in ("start_marker", "warping")]))
        pending = []
        for k in live:
            r, w = next(backs), next(backs)
            if not r.get("ok"):
                out[k] = (False, "get %s.start_marker: %s" % (paths[k], lom_err(r)))
                continue
            got = float(r["result"]["value"])
            v, unit, warping = value[k]
            if w.get("ok") and bool(w["result"]["value"]) != warping:
                if attempt == 1:
                    in_unit(k, not warping)
                    unwritten.pop(k, None)
                    pending.append(k)
                else:
                    out[k] = (False, "warping changed again, to %s, after the trim was "
                                     "redone in %s" % (not warping,
                                                       "beats" if warping else "seconds"))
            elif abs(got - v) > 1e-3:
                out[k] = (False, "start_marker did not take — asked %.4f %s, reads %.4f"
                          % (v, unit, got))
            elif k in unwritten:
                out[k] = (False, unwritten[k])
            else:
                out[k] = (True, "trim %.4f %s" % (v, unit))
        if not pending:
            break
    return out


def _took(asked, got):
    """Does a property read `got` after `asked` was written to it?"""
    if isinstance(asked, bool) or not isinstance(asked, (int, float)):
        return got == asked
    try:
        return abs(float(got) - asked) <= 1e-3
    except (TypeError, ValueError):
        return False


def _shown(v, unit):
    return "%.4f %s" % (v, unit) if isinstance(v, float) else str(v)


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


# Locators read before their count is known: a Set with a locator per section has about ten.
CUES_READ_AHEAD = 16


def cue_point_list(*specs):
    """Current locators, in Live's own order, as {index, time, name} dicts.

    One request, the list and the first CUES_READ_AHEAD locators' time and name
    together (a second only past that). It is only ever sent in a request of its
    own after a play-head move or a toggle, as every read of it always was.
    With `specs` those reads go in the same request, ahead of the list, and
    (their envelopes, the locators) comes back.
    """
    def fields(i):
        p = "live_set cue_points %d" % i
        return [{"path": p, "property": "time"}, {"path": p, "property": "name"}]

    head = list(specs) + [{"path": "live_set", "property": "cue_points"}]
    for i in range(CUES_READ_AHEAD):
        head += fields(i)
    envs = lom_reads(head)
    extra, vec, got = envs[:len(specs)], envs[len(specs)], envs[len(specs) + 1:]
    out = []
    if vec.get("ok"):
        n = live_batch.vector_len(vec["result"]["value"])
        got += lom_reads([s for i in range(CUES_READ_AHEAD, n) for s in fields(i)])
        for i in range(n):
            t, nm = got[2 * i], got[2 * i + 1]
            if t.get("ok") and nm.get("ok"):
                out.append({"index": i, "time": float(t["result"]["value"]),
                            "name": nm["result"]["value"]})
    return (extra, out) if specs else out


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
    # position or toggles at it. The locator list is read in that same request:
    # both are reads, and nothing is sent between them.
    (now,), cues = cue_point_list({"path": "live_set", "property": "current_song_time"})
    if not now.get("ok"):
        raise RuntimeError("get live_set.current_song_time: %s" % lom_err(now))
    actual = round(float(now["result"]["value"]), 6)

    here = [c for c in cues if round(c["time"], 6) == actual]
    if here:
        idx = here[0]["index"]                 # rename; do NOT toggle it away
    else:
        # The same list: it was read once, and nothing has changed it since.
        before = {round(c["time"], 6) for c in cues}
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
    ap.add_argument("--no-tempo-check", action="store_true",
                    help="keep the beat detector's tempo without asking the kick "
                         "stem whether it is right")
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
            tempo_check=not args.no_tempo_check)
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
    # THE TABLE IS BUILT IN STEPS, EACH ONE REQUEST (OR A FEW) FOR EVERY TRACK,
    # and printed once a row has them all. A step runs for the tracks the one
    # before it left standing, in table order, so a track that fails is reported
    # and skipped at the same step as when each track ran start to finish. No
    # track's ops touch another track's objects, so step N on every track before
    # step N+1 on any leaves the Set as the track-by-track loop did.
    rows = []
    for i, name in enumerate(order):
        t = (math.log(cent[name]) - lo) / span
        r, g, b = colorsys.hsv_to_rgb(t * (240 / 360), 0.85, 1.0)
        rows.append({"ti": targets[i], "name": name, "lines": [],
                     "rgb": (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)})

    # NO delete step. Track.delete_clip() takes a Clip OBJECT, which no JSON
    # argument can express, so the bridge cannot reach it. Creating a clip at
    # the same position REPLACES the one there, but only across the span the
    # new clip covers — place at a DIFFERENT position and the old clip's head
    # survives in front of the new one. Hence find_placed_clip below.
    #
    # RENAME ONLY ONCE THE CLIP IS ON THE TRACK. The rename used to come
    # first, so a track that could not take the audio was relabelled anyway
    # — four of a user's MIDI tracks lost their names to stems that then
    # failed to land. A failed placement must leave the Set as it found it.
    # So the name and the fader are nested after the create, and Live runs them
    # only once it has succeeded.
    # Rows of the table print from a finally, so whatever the steps so far found still prints
    # when the run ends early (a refused write batch, a lost connection).
    try:
        ops = []
        for row in rows:
            ti = row["ti"]
            after = [{"op": "set", "path": "live_set tracks %d" % ti, "property": "name",
                      "value": row["name"]}]
            if args.headroom:
                try:
                    after.append(track_headroom_op(ti, -abs(args.headroom)))
                except ValueError as e:
                    row["headroom_error"] = str(e)
            ops.append(live_batch.seq([
                {"op": "call", "path": "live_set tracks %d" % ti,
                 "function": "create_audio_clip", "args": [paths[row["name"]], START]},
                live_batch.seq(after, stop=False)]))
        for row, env in zip(rows, lom_ops(ops)):
            placed = seq_results(env)
            row["clip"] = placed[0]
            if not placed[0].get("ok"):
                row["lines"].append("%3d  %-14s  ERROR: %s — track left untouched"
                                    % (row["ti"], row["name"], lom_err(placed[0])))
                continue
            row["placed"] = True
            if args.headroom:
                hr = seq_results(placed[1])[1:]
                detail = row.get("headroom_error") or (
                    lom_err(hr[0]) if hr and not hr[0].get("ok") else None)
                if detail:
                    row["lines"].append("%3d  %-14s  headroom not applied: %s"
                                        % (row["ti"], row["name"], detail))

        # Resolve the clip we just made BY ITS START TIME. Never index 0 — see
        # find_placed_clip. Everything after this point addresses `ci`. Read in a
        # request after the one that placed the clips.
        placed = [row for row in rows if row.get("placed")]
        lists = arrangement_clips([row["ti"] for row in placed])
        for row in placed:
            ti, name, clips = row["ti"], row["name"], lists[row["ti"]]
            if isinstance(clips, RuntimeError):
                row["lines"].append("%3d  %-14s  ERROR: could not list arrangement clips (%s) — "
                                    "colour, mute, warp and trim SKIPPED for this track"
                                    % (ti, name, clips))
                continue
            ci = find_placed_clip(clips, START)
            if ci is None:
                # Not a silent fallback to 0: acting on the wrong clip is what this
                # whole path exists to prevent.
                row["lines"].append("%3d  %-14s  ERROR: no clip starts at beat %.4f (found %s) — "
                                    "colour, mute, warp and trim SKIPPED for this track"
                                    % (ti, name, START,
                                       ", ".join("%.4f" % s for _i, s, _e in clips) or "none"))
                continue
            row["ci"], row["clips"] = ci, clips

        found = [row for row in placed if "ci" in row]
        ops = []
        for row in found:
            ti, cpath = row["ti"], "live_set tracks %d arrangement_clips %d" % (row["ti"], row["ci"])
            # Colour the TRACK to match its clip. The module docstring used to say
            # only clip colour was settable — true of the AbletonMCP bridge, which
            # had no handler for it, and false here: LOM reaches Track.color
            # directly (verified: 11958214 -> 16725558). Without it the track
            # headers stay Live's default palette while the clips carry the ramp,
            # so the frequency ordering is invisible until you look at the clips.
            mine = [{"op": "set", "path": cpath, "property": "color", "value": row["rgb"]},
                    {"op": "set", "path": "live_set tracks %d" % ti, "property": "color",
                     "value": row["rgb"]}]
            if row["name"] in muted:
                mine.append({"op": "set", "path": cpath, "property": "muted", "value": True})
            # Warp only a drifting set, and stop trying after the bridge says it has
            # never heard of the command — that answer will not change mid-run.
            if warp_map and warp_state != "unsupported":
                mine.append(warp_map_op(ti, row["ci"], warp_map))
            ops.append(live_batch.seq(mine, stop=False))
        for row, env in zip(found, lom_ops(ops)):
            done = seq_results(env)
            row["col"] = done[0]
            # Leftovers are AUDIBLE DUPLICATE AUDIO, not a cosmetic wart: the stale
            # clip still plays. Collected and reported loudly after the table.
            row["leftovers"] = [(s, e) for j, s, e in row["clips"] if j != row["ci"]]
            if row["leftovers"]:
                stale.append((row["ti"], row["name"], row["leftovers"]))
            # A track after the one that said "unknown op" was sent the map too, and
            # Live changed nothing for it; it reads as if it was never sent.
            row["warp_note"] = ""
            if warp_map and warp_state != "unsupported":
                warp_state, detail = warp_outcome(done[-1])
                if warp_state == "ok":
                    row["warp_note"] = "  " + detail
                elif warp_state == "unsupported":
                    row["warp_note"] = "  [no warp]"
                else:
                    row["warp_note"] = "  [warp failed: %s]" % detail

        # AFTER the warp map, never before: the trim's unit is seconds on an
        # unwarped clip and beats on a warped one, so `warping` has to have
        # settled before trim_clip_starts reads it, a request later.
        for row in found:
            row["trim_note"] = ""
        if TRIM > 0:
            trims = trim_clip_starts([(row["ti"], row["ci"]) for row in found],
                                     TRIM, period, warp_map)
            for row, (ok, detail) in zip(found, trims):
                row["trim_note"] = "  " + detail if ok else "  [trim failed: %s]" % detail
        for row in found:
            row["finished"] = True
    finally:
        for row in rows:
            for line in row["lines"]:
                print(line)
            if row.get("finished"):
                print("%3d  %-14s %8.0fHz   #%06X  clip=%s col=%s%s%s%s%s"
                      % (row["ti"], row["name"], cent[row["name"]], row["rgb"],
                         status(row["clip"]), status(row["col"]),
                         "  [muted: parts present]" if row["name"] in muted else "",
                         row["warp_note"], row["trim_note"],
                         "  [+%d STALE]" % len(row["leftovers"]) if row["leftovers"] else ""))

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
            print("           boundaries are measured (stem activity, or spectral "
                  "change without stems); names are rules")

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

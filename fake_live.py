"""An Ableton Live for tests: a state model of what stems2live and session_view touch, served
the way the AbletonLOM Remote Script serves it, batch ops included.

WHAT IT IS FOR. Batching changes which requests reach Live and in what order, so a diff of the
wire cannot say whether a load still leaves the Set as it did. This model can: run the old code
and the new against two copies of the same Set and compare the Sets afterwards (snapshot()). It
holds tracks (audio and MIDI, name, colour, fader), arrangement clips (time, markers, loop,
warping, warp map, colour, mute, name, notes), session clip slots, scenes, locators, tempo, the
play head and the transport.

WHAT IT CHECKS ON TOP: the dependencies batching could break. Live runs one request as one task
on its main thread; the one-op-per-request code put every dependent step in a later task. Within
one request the model logs a HAZARD for
  - an index into a list that a create or delete earlier in the same request changed
    (a new track, clip, scene or locator addressed by index in the request that made it), and a
    count or read of such a list;
  - a read of a property a write earlier in the same request changed (a read-back in the same
    task as its write), including song_length after a clip was placed;
  - a marker write on a clip after its `looping` or `warping` was written in the same request,
    and a `looping` write after a marker write;
  - a toggle (set_or_delete_cue) or a read of the play head after a play-head move in the same
    request. The move itself only lands when the request ends, which is how Live behaves: the
    toggle fires at the PREVIOUS position (see stems2live.settle).
The model still answers a hazardous op (as if Live had settled), so a run completes; the test
asserts there were none.

A request can also be answered with Live's Timeout reply or a closed connection AFTER it ran
(`cuts`), which is what Live does with a task it gave up waiting on: it keeps running it.

WHAT IT DOES NOT MODEL, and so cannot vouch for:
  - time: the 15 s main-thread limit is only an error the test injects (`cuts`), never a
    consequence of how much a request carries; no op has a cost;
  - Live changing a clip's `warping` on its own after the create (auto-warp landing later,
    measured on Live 2026-11-15, markers not converted): only the scripts change it here,
    unless a test patches `dispatch` to flip it (test_batch_live's `_auto_warp`);
  - order inside one task between writes to DIFFERENT properties of one clip (loop_start, then
    loop_end, then start_marker, ...): setters apply at once and never constrain each other,
    except that start_marker is ignored while the clip is not looping (the scripts check every
    marker set's reply against the value asked);
  - several heavy creates in one task interacting (they are independent here);
  - Live's palette and name rules (a colour or name is stored as sent);
  - audio and Live's own warp analysis (warping=True rescales the markers by the Set tempo);
  - undo/redo themselves (only how many undo steps a run would leave: one per request that
    changed something, on Live 12.4+ where parameter changes are synced into a batch's step),
    and Live before 12.4, where parameter changes split off into steps of their own;
  - default colours (every new object starts at 0), devices, automation.

Server semantics are copied from Sideman's remote_script/AbletonLOM/handlers.py at the commit
requirements-sideman.txt pins: get_batch never fails as a whole; transaction and set_batch run
ops in order, each failure is an entry "<Type>: <message>", stop_on_error halts at the first,
nested transactions run as ops of their own; delete_track, delete_clip and a set_or_delete_cue
that would delete need confirm (or expect).
"""

import json
import os
import sys

from sideman.client import Live, LiveError

try:
    import soundfile
except ImportError:                       # pragma: no cover - pinned in requirements.txt
    soundfile = None

MARKERS = ("loop_start", "loop_end", "start_marker", "end_marker")


class _Obj(object):
    PROPS = ()            # readable through op "get"
    WRITABLE = ()         # settable through op "set"
    CALLS = ()            # callable through op "call"

    def __init__(self, model):
        self._m = model


class Param(_Obj):
    PROPS = WRITABLE = ("value",)

    def __init__(self, model, value):
        super().__init__(model)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._m._wrote(self, "value")
        self._value = float(v)


class Mixer(_Obj):
    PROPS = ("volume",)

    def __init__(self, model):
        super().__init__(model)
        self.volume = Param(model, 0.85)


class Clip(_Obj):
    PROPS = ("name", "color", "muted", "warping", "looping", "start_time", "end_time",
             "is_midi_clip", "length", "warp_markers") + MARKERS
    WRITABLE = ("name", "color", "muted", "warping", "looping") + MARKERS
    _DIRTIES = {"warping": ("warping",) + MARKERS + ("end_time", "length"),
                "looping": ("looping",) + MARKERS + ("end_time", "length")}

    def __init__(self, model, name, start_time=None, seconds=0.0, midi=False, path=None):
        super().__init__(model)
        self._v = {"name": name, "color": 0, "muted": False, "warping": False,
                   "looping": False, "loop_start": 0.0, "loop_end": seconds,
                   "start_marker": 0.0, "end_marker": seconds}
        self.is_midi_clip = midi
        self.file_path = path
        self.start_time = start_time
        self.end_time = None if start_time is None else start_time + model.beats(seconds)
        self.warp_markers = []
        self.notes = []

    @property
    def length(self):
        return None if self.start_time is None else self.end_time - self.start_time

    def __getattr__(self, name):
        v = self.__dict__.get("_v")
        if v is not None and name in v:
            return v[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in Clip.WRITABLE and "_v" in self.__dict__:
            self._set(name, value)
        else:
            object.__setattr__(self, name, value)

    def _set(self, name, value):
        m = self._m
        if name in MARKERS and (m._written(self, "looping") or m._written(self, "warping")):
            m.hazard("%s written after looping/warping in the same request" % name)
        if name == "looping" and any(m._written(self, p) for p in MARKERS):
            m.hazard("looping written after a marker in the same request")
        m._wrote(self, *self._DIRTIES.get(name, (name,) if name not in MARKERS
                                          else MARKERS + ("end_time", "length")))
        if name == "start_marker" and not self._v["looping"]:
            return                    # Live's quirk: ignored unless the clip is looping
        if name == "warping" and value and not self._v["warping"]:
            for p in MARKERS:         # seconds -> beats, at the Set tempo
                self._v[p] = m.beats(self._v[p])
        self._v[name] = value


class ClipSlot(_Obj):
    PROPS = ("has_clip", "clip")
    CALLS = ("create_audio_clip", "delete_clip")

    def __init__(self, model, track):
        super().__init__(model)
        self.track, self._clip = track, None

    @property
    def clip(self):
        self._m._read(self, "clip")
        return self._clip

    @property
    def has_clip(self):
        self._m._read(self, "clip")
        return self._clip is not None

    def create_audio_clip(self, path):
        self._m._read(self, "clip")           # is the slot empty?
        if self._clip is not None:
            raise RuntimeError("Clip slot is not empty")
        seconds = self._m.audio_seconds(path, self.track)
        self._m._wrote(self, "clip")
        self._clip = Clip(self._m, os.path.splitext(os.path.basename(path))[0],
                          seconds=seconds, path=path)
        return self._clip

    def delete_clip(self):
        self._m._read(self, "clip")
        if self._clip is None:
            raise RuntimeError("There is no clip in this slot")
        self._m._wrote(self, "clip")
        self._clip = None


class Track(_Obj):
    PROPS = ("name", "color", "has_audio_input", "mixer_device", "arrangement_clips",
             "clip_slots", "devices", "is_foldable", "has_midi_input", "is_grouped",
             "group_track", "mute", "muted_via_solo")
    WRITABLE = ("name", "color")
    CALLS = ("create_audio_clip", "create_midi_clip")

    def __init__(self, model, audio, name, scenes):
        super().__init__(model)
        self.audio, self._name, self._color = audio, name, 0
        self.mixer_device = Mixer(model)
        self.clips = []
        self.clip_slots = [ClipSlot(model, self) for _ in range(scenes)]
        self.devices = []
        self.is_foldable = False
        self.group_track = None
        self.mute = self.muted_via_solo = False

    has_midi_input = property(lambda self: not self.audio)
    is_grouped = property(lambda self: self.group_track is not None)

    name = property(lambda self: self._name,
                    lambda self, v: (self._m._wrote(self, "name"),
                                     object.__setattr__(self, "_name", v)))
    color = property(lambda self: self._color,
                     lambda self, v: (self._m._wrote(self, "color"),
                                      object.__setattr__(self, "_color", int(v))))

    @property
    def has_audio_input(self):
        return self.audio

    @property
    def arrangement_clips(self):
        return sorted(self.clips, key=lambda c: c.start_time)

    def create_audio_clip(self, path, position):
        seconds = self._m.audio_seconds(path, self)
        pos = float(position)
        clip = Clip(self._m, os.path.splitext(os.path.basename(path))[0], pos, seconds,
                    path=path)
        self._place(clip)
        return clip

    def create_midi_clip(self, start, length):
        if self.audio:
            raise RuntimeError("MIDI clips can only be created on MIDI tracks")
        clip = Clip(self._m, "", float(start), 0.0, midi=True)
        clip.end_time = float(start) + float(length)
        self._place(clip)
        return clip

    def _place(self, clip):
        """Live replaces only the span the new clip covers: a clip it overlaps loses that part,
        and a head in front of it survives as a stub."""
        lo, hi = clip.start_time, clip.end_time
        keep = []
        for c in self.clips:
            if c.end_time <= lo or c.start_time >= hi:
                keep.append(c)
            elif c.start_time < lo:
                c.end_time = lo
                keep.append(c)
            elif c.end_time > hi:
                c.start_time = hi
                keep.append(c)
        self.clips = keep + [clip]
        self._m._moved(self, "arrangement_clips", 0)
        self._m._wrote(self._m.song, "song_length")
        self._m.changed()


class Chain(_Obj):
    PROPS = ("name", "in_note")

    def __init__(self, model, name, in_note):
        super().__init__(model)
        self.name, self.in_note = name, in_note


class Device(_Obj):
    """A device a test puts on a track; a Drum Rack (class_name DrumGroupDevice) has chains."""
    PROPS = ("name", "class_name", "chains")

    def __init__(self, model, name, class_name="Device", chains=()):
        super().__init__(model)
        self.name, self.class_name, self.chains = name, class_name, list(chains)


class Scene(_Obj):
    PROPS = WRITABLE = ("name",)

    def __init__(self, model, name=""):
        super().__init__(model)
        self._name = name

    name = property(lambda self: self._name,
                    lambda self, v: (self._m._wrote(self, "name"),
                                     object.__setattr__(self, "_name", v)))


class CuePoint(_Obj):
    PROPS = ("time", "name")
    WRITABLE = ("name",)

    def __init__(self, model, time, name):
        super().__init__(model)
        self.time, self._name = time, name

    name = property(lambda self: self._name,
                    lambda self, v: (self._m._wrote(self, "name"),
                                     object.__setattr__(self, "_name", v)))


class Song(_Obj):
    PROPS = ("tempo", "current_song_time", "is_playing", "song_length", "tracks", "scenes",
             "cue_points", "file_path")
    file_path = ""                        # an unsaved Set
    WRITABLE = ("tempo", "current_song_time")
    CALLS = ("create_audio_track", "create_midi_track", "delete_track", "create_scene",
             "set_or_delete_cue", "stop_playing")

    def __init__(self, model, tempo):
        super().__init__(model)
        self._tempo, self._time, self.is_playing = tempo, 0.0, False
        self.tracks, self.scenes, self.cue_points = [], [], []

    @property
    def tempo(self):
        return self._tempo

    @tempo.setter
    def tempo(self, v):
        self._m._wrote(self, "tempo")
        self._tempo = float(v)

    @property
    def song_length(self):
        ends = [c.end_time for t in self.tracks for c in t.clips] + [c.time for c in self.cue_points]
        return max(ends + [0.0])

    @property
    def current_song_time(self):
        self._m._read(self, "current_song_time")
        return self._time

    @current_song_time.setter
    def current_song_time(self, v):
        v = float(v)
        if v > self.song_length:
            raise RuntimeError("Cannot set the Songtime behind the Songlength")
        self._m._wrote(self, "current_song_time")
        # Live snaps the play head to its grid, and lands only once the task has ended.
        self._m.pending_time = round(v * 64.0) / 64.0

    def _new_track(self, audio, index):
        i = len(self.tracks) if index == -1 else int(index)
        name = "%d-%s" % (i + 1, "Audio" if audio else "MIDI")
        t = Track(self._m, audio, name, len(self.scenes))
        self.tracks.insert(i, t)
        self._m._moved(self, "tracks", i)
        self._m.changed()
        return t

    def create_audio_track(self, index):
        return self._new_track(True, index)

    def create_midi_track(self, index):
        return self._new_track(False, index)

    def delete_track(self, index):
        del self.tracks[int(index)]
        self._m._moved(self, "tracks", int(index))
        self._m.changed()

    def create_scene(self, index):
        if int(index) != -1:
            raise ValueError("the model appends scenes only")
        self.scenes.append(Scene(self._m))
        self._m._moved(self, "scenes", len(self.scenes) - 1)
        for t in self.tracks:
            t.clip_slots.append(ClipSlot(self._m, t))
            self._m._moved(t, "clip_slots", len(t.clip_slots) - 1)
        self._m.changed()
        return self.scenes[-1]

    def cue_at_playhead(self):
        for c in self.cue_points:
            if abs(c.time - self._time) <= 0.01:
                return c
        return None

    def set_or_delete_cue(self):
        if self._m._written(self, "current_song_time"):
            self._m.hazard("set_or_delete_cue in the request that moved the play head")
        here = self.cue_at_playhead()
        if here is not None:
            self.cue_points.remove(here)
        else:
            self.cue_points.append(CuePoint(self._m, self._time, str(len(self.cue_points) + 1)))
            self.cue_points.sort(key=lambda c: c.time)
        self._m._moved(self, "cue_points", 0)
        self._m.changed()

    def stop_playing(self):
        self.is_playing = False


def _type_name(obj):
    return type(obj).__name__


def _is_vector(v):
    return isinstance(v, list)


def jsonify(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if _is_vector(v):
        if depth == 0 and len(v) <= 64:
            return [jsonify(x, depth + 1) for x in v]
        return {"__vector__": True, "count": len(v)}
    if isinstance(v, _Obj):
        out = {"__lom__": _type_name(v)}
        name = getattr(v, "name", None)
        if isinstance(name, str):
            out["name"] = name
        return out
    return {"__repr__": str(v), "type": _type_name(v)}


class FakeLive(object):
    """The Set, and the Remote Script serving it. `request(op, params)` is one request."""

    def __init__(self, tracks=("midi", "midi", "audio", "audio"), scenes=8, tempo=120.0):
        self.song = Song(self, tempo)
        self.pending_time = None
        self.requests = []                # one dict per request: op, leaves, module
        self.hazards = []
        self.undo_steps = 0
        self.faults = []                  # callables (op, params) -> exception or None
        # callables (op, params) -> "Timeout" / "ConnectionClosed" / None: the request is run to
        # the end, as Live runs a task it gave up waiting on, and then answered with that error
        self.cuts = []
        self._undo_depth = 0
        self._step_changed = False
        self._dirty, self._moves = set(), {}
        for _ in range(scenes):
            self.song.scenes.append(Scene(self))
        for kind in tracks:
            self.song._new_track(kind == "audio", -1)
        self._dirty, self._moves, self._step_changed = set(), {}, False

    # ------------------------------------------------------------ the Set
    def beats(self, seconds):
        return seconds * self.song.tempo / 60.0

    def audio_seconds(self, path, track):
        if not track.audio:
            raise RuntimeError("Audio clips can only be created on audio tracks")
        if not os.path.isabs(path):
            raise RuntimeError("Please provide an absolute path")
        if not os.path.exists(path):
            raise RuntimeError("File not found: %s" % path)
        return float(soundfile.info(path).duration)

    def add_cue(self, time, name):
        self.song.cue_points.append(CuePoint(self, float(time), name))
        self.song.cue_points.sort(key=lambda c: c.time)

    def snapshot(self):
        """Everything the scripts can change, as plain data."""
        def clip(c):
            if c is None:
                return None
            return {"name": c.name, "color": c.color, "muted": c.muted, "warping": c.warping,
                    "looping": c.looping, "start_time": c.start_time, "end_time": c.end_time,
                    "markers": [c._v[p] for p in MARKERS], "file": c.file_path,
                    "warp_markers": c.warp_markers, "notes": c.notes, "midi": c.is_midi_clip}
        s = self.song
        return {"tempo": s.tempo, "time": s._time, "playing": s.is_playing,
                "scenes": [sc.name for sc in s.scenes],
                "cues": [(c.time, c.name) for c in s.cue_points],
                "tracks": [{"audio": t.audio, "name": t.name, "color": t.color,
                            "volume": t.mixer_device.volume.value,
                            "clips": [clip(c) for c in t.arrangement_clips],
                            "slots": [clip(sl._clip) for sl in t.clip_slots]}
                           for t in s.tracks]}

    # --------------------------------------------------------- the checks
    def hazard(self, what):
        self.hazards.append(what)

    def changed(self):
        self._step_changed = True

    def _wrote(self, obj, *props):
        for p in props:
            self._dirty.add((id(obj), p))
        if not (isinstance(obj, Song) and props == ("current_song_time",)):
            self.changed()

    def _written(self, obj, prop):
        return (id(obj), prop) in self._dirty

    def _read(self, obj, prop):
        if self._written(obj, prop):
            self.hazard("%s.%s read in the request that wrote it" % (_type_name(obj), prop))

    def _moved(self, owner, name, index):
        key = (id(owner), name)
        self._moves[key] = min(self._moves.get(key, index), index)

    def _index(self, owner, name, index):
        key = (id(owner), name)
        if key in self._moves and index >= self._moves[key]:
            self.hazard("%s %d addressed in the request that created or deleted %s"
                        % (name, index, name))

    # ------------------------------------------------- the Remote Script
    def resolve(self, path):
        if not path or not str(path).strip():
            raise ValueError("empty path")
        comps = str(path).split()
        if comps[0] != "live_set":
            raise ValueError("unknown root %r; expected one of ['live_set']" % comps[0])
        cur, owner, name, walked = self.song, None, None, ["live_set"]
        for comp in comps[1:]:
            if cur is None:
                raise ValueError("path %r resolved to None at %r" % (path, " ".join(walked)))
            if comp.lstrip("-").isdigit():
                self._index(owner, name, int(comp))
                try:
                    cur = cur[int(comp)]
                except IndexError:
                    raise IndexError("index %s out of range at %r" % (comp, " ".join(walked)))
            else:
                if isinstance(cur, list) or comp not in type(cur).PROPS:
                    raise AttributeError("no member %r at %r" % (comp, " ".join(walked)))
                owner, name, cur = cur, comp, getattr(cur, comp)
            walked.append(comp)
        return cur

    def _undo(self):
        model = self

        class _Step(object):
            def __enter__(self):
                model._undo_depth += 1

            def __exit__(self, *exc):
                model._undo_depth -= 1
                if model._undo_depth == 0:
                    if model._step_changed:
                        model.undo_steps += 1
                    model._step_changed = False
                return False
        return _Step()

    def _fault(self, op, params):
        for f in self.faults:
            e = f(op, params)
            if e is not None:
                raise e

    def op_get(self, p):
        obj = self.resolve(p["path"])
        prop = p["property"]
        if prop not in type(obj).PROPS and prop not in type(obj).CALLS:
            raise RuntimeError("%r unavailable on this %s: %s" % (prop, _type_name(obj), prop))
        if prop in type(obj).CALLS:
            raise TypeError("%r is a function; use op 'call'" % prop)
        self._read(obj, prop)
        if _is_vector(getattr(obj, prop)):
            self._index(obj, prop, 0)
        v = getattr(obj, prop)
        return {"path": p["path"], "property": prop, "value": jsonify(v), "type": _type_name(v)}

    def op_set(self, p):
        path, prop = p["path"], p["property"]
        obj = self.resolve(path)
        if obj is None or (prop not in type(obj).PROPS and prop not in type(obj).CALLS):
            raise AttributeError("%s has no member %r" % (_type_name(obj), prop))
        if prop in type(obj).CALLS:
            raise TypeError("%r is a function; use op 'call'" % prop)
        if prop not in type(obj).WRITABLE:
            raise AttributeError("can't set attribute %r" % prop)
        with self._undo():
            setattr(obj, prop, p["value"])
        dirty, self._dirty = self._dirty, set()      # the reply's own read-back is no hazard
        try:
            value = jsonify(getattr(obj, prop))
        finally:
            self._dirty = dirty
        return {"path": path, "property": prop, "value": value}

    def op_call(self, p):
        path, fn = p["path"], p["function"]
        args = p.get("args") or []
        obj = self.resolve(path)
        if obj is None or fn not in type(obj).CALLS:
            raise AttributeError("%s has no member %r" % (_type_name(obj), fn))
        gated = fn in ("delete_track", "delete_clip") or (
            fn == "set_or_delete_cue" and obj.cue_at_playhead() is not None)
        if gated and p.get("expect") is None and not p.get("confirm"):
            raise PermissionError("%r is destructive; re-send with expect=<the target's name, "
                                  "as you just read it> (or confirm=true)" % fn)
        with self._undo():
            result = getattr(obj, fn)(*args)
        return {"path": path, "function": fn, "result": jsonify(result)}

    def op_count(self, p):
        obj = self.resolve(p["path"])
        val = getattr(obj, p["child"])
        if not _is_vector(val):
            raise TypeError("%r is not a list" % p["child"])
        self._index(obj, p["child"], 0)
        return {"path": p["path"], "child": p["child"], "count": len(val)}

    def op_warp_markers_set(self, p):
        obj = self.resolve(p["path"])
        markers = p.get("markers") or []
        if len(markers) < 2:
            raise ValueError("at least 2 warp markers required, got %d" % len(markers))
        parsed = sorted((float(a), float(b)) for a, b in markers)
        with self._undo():
            obj.warping = True
            old = len(obj.warp_markers)
            obj.warp_markers = [list(m) for m in parsed]
            self._wrote(obj, "warp_markers")
        return {"path": p["path"], "marker_count": len(parsed) + (0 if old else 1),
                "added": len(parsed), "removed": old, "remove_failed": 0 if old else 1,
                "warping": True, "marker_shape": "kwargs"}

    def op_notes_add(self, p):
        clip = self.resolve(p["path"])
        if not getattr(clip, "is_midi_clip", False):
            raise TypeError("%s is not a MIDI clip" % p["path"])
        for i, n in enumerate(p["notes"]):
            missing = [k for k in ("pitch", "start_time", "duration") if k not in n]
            if missing:
                raise KeyError("note %d is missing %s (required: pitch, start_time, "
                               "duration)" % (i, ", ".join(missing)))
        with self._undo():
            clip.notes = clip.notes + [dict(n) for n in p["notes"]]
            self.changed()
        return {"path": p["path"], "added": len(p["notes"])}

    def op_notes_get(self, p):
        clip = self.resolve(p["path"])
        if not getattr(clip, "is_midi_clip", False):
            raise TypeError("%s is not a MIDI clip" % p["path"])
        lo = float(p.get("from_time", 0.0))
        hi = lo + float(p.get("time_span", float("inf")))
        notes = [dict({"velocity": 100.0, "mute": False}, note_id=i, **n)
                 for i, n in enumerate(clip.notes) if lo <= n["start_time"] < hi]
        return {"path": p["path"], "count": len(notes), "total": len(clip.notes), "notes": notes}

    def op_ping(self, p):
        return {"handlers": True}

    def op_get_batch(self, p):
        return get_batch(self.dispatch, p)

    def op_set_batch(self, p):
        return run_ops(self.dispatch, [dict(s, op="set") for s in p["specs"]],
                       p.get("stop_on_error", True), self._undo(), self.OPS)

    def op_transaction(self, p):
        if p.get("rollback_on_error"):
            raise AssertionError("rollback_on_error is never to be used")
        return run_ops(self.dispatch, p["ops"], p.get("stop_on_error", True), self._undo(),
                       self.OPS)

    OPS = ("ping", "get", "set", "call", "count", "warp_markers_set", "notes_add", "notes_get",
           "get_batch", "set_batch", "transaction")

    def dispatch(self, op, params):
        if op not in self.OPS:
            raise ValueError("unknown op %r; known: %s" % (op, sorted(self.OPS)))
        if op not in ("get_batch", "set_batch", "transaction"):
            self._fault(op, params)
        return getattr(self, "op_" + op)(params)

    def request(self, op, params, module=None):
        """One request: one task on Live's main thread."""
        self._dirty, self._moves = set(), {}
        self.requests.append({"op": op, "leaves": leaves(op, params), "module": module})
        try:
            result = self.dispatch(op, params)
            for cut in self.cuts:
                kind = cut(op, params)
                if kind:
                    raise type(kind, (Exception,), {})(
                        "main thread did not respond in 15.0s" if kind == "Timeout"
                        else "Live closed the connection before replying")
            return result
        finally:
            if self.pending_time is not None:
                self.song._time = self.pending_time + (1.48 if self.song.is_playing else 0.0)
                self.pending_time = None
            self._dirty, self._moves = set(), {}

    def install(self, monkeypatch):
        """Serve `sideman.client.Live.request` from this model, as the client would see it:
        params through JSON, a refusal as LiveError(type, message)."""
        model = self

        def request(_live, op, params=None):
            params = json.loads(json.dumps(params or {}))
            try:
                return model.request(op, params, module=_caller())
            except Exception as e:                          # noqa: BLE001
                raise LiveError(type(e).__name__, str(e))
        monkeypatch.setattr(Live, "request", request)
        return self


class _NoStep(object):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def get_batch(dispatch, p):
    """op_get_batch: every spec read through `dispatch("get", spec)`; never fails as a whole."""
    out = []
    for spec in p["specs"]:
        try:
            out.append(dict(dispatch("get", spec), ok=True))
        except Exception as e:
            out.append({"ok": False, "path": spec.get("path"),
                        "property": spec.get("property"),
                        "error": "%s: %s" % (type(e).__name__, e)})
    return {"count": len(out), "ok_count": sum(1 for r in out if r["ok"]), "results": out}


def run_ops(dispatch, ops, stop_on_error, step=None, known=None):
    """_run_ops: ops in order inside one undo `step`, a failure an entry of its own."""
    results, failed = [], False
    with step or _NoStep():
        for i, spec in enumerate(ops):
            name = spec.get("op", "set")
            if known is not None and name not in known:
                results.append({"index": i, "op": name, "ok": False,
                                "error": "unknown op %r" % name})
                failed = True
                if stop_on_error:
                    break
                continue
            try:
                results.append({"index": i, "op": name, "ok": True,
                                "result": dispatch(name, spec)})
            except Exception as e:
                results.append({"index": i, "op": name, "ok": False,
                                "error": "%s: %s" % (type(e).__name__, e)})
                failed = True
                if stop_on_error:
                    break
    applied = sum(1 for r in results if r["ok"])
    return {"count": len(results), "applied": applied, "failed": failed,
            "results": results, "undo": "ONE Cmd-Z reverts the batch"}


BATCH_OPS = ("get_batch", "set_batch", "transaction")


def with_batches(handle):
    """A fake of one op at a time, `handle(op, params)` -> result (raising to refuse), served
    with the Remote Script's batch ops on top: each op inside a batch goes to `handle`."""
    def dispatch(op, params):
        if op == "get_batch":
            return get_batch(dispatch, params)
        if op == "set_batch":
            return run_ops(dispatch, [dict(s, op="set") for s in params["specs"]],
                           params.get("stop_on_error", True))
        if op == "transaction":
            return run_ops(dispatch, params["ops"], params.get("stop_on_error", True))
        return handle(op, params)
    return dispatch


def lom_with_batches(lom):
    """The same for a fake of `stems2live.lom`, (op, **params) -> envelope."""
    def handle(op, params):
        r = lom(op, **{k: v for k, v in params.items() if k != "op"})
        if not r.get("ok"):
            err = r.get("error") or {}
            raise type(str(err.get("type") or "Error"), (Exception,), {})(err.get("message"))
        return r.get("result")
    served = with_batches(handle)

    def fake(op, **params):
        if op not in BATCH_OPS:
            return lom(op, **params)
        return {"ok": True, "result": served(op, params)}
    return fake


def leaves(op, params):
    """The single operations one request carries, as (op, params), nested batches flattened."""
    if op == "get_batch":
        return [("get", s) for s in params.get("specs", [])]
    if op == "set_batch":
        return [("set", s) for s in params.get("specs", [])]
    if op == "transaction":
        return [leaf for s in params.get("ops", []) for leaf in leaves(s.get("op", "set"), s)]
    return [(op, params)]


def _caller():
    """Which script sent the request: "scenes" from session_view, else "load"."""
    frame = sys._getframe(2)
    while frame is not None:
        if "session_view" in os.path.basename(frame.f_code.co_filename):
            return "scenes"
        frame = frame.f_back
    return "load"


def phase(module, op, p):
    """The step of a run a single operation belongs to, by what it does."""
    path, prop = p.get("path", ""), p.get("property", p.get("child", ""))
    fn = p.get("function", "")
    if op == "ping" or (op == "warp_markers_set" and "path" not in p):
        return "probe"
    if "cue_points" in path or prop in ("cue_points", "current_song_time", "song_length",
                                        "is_playing") or fn in ("set_or_delete_cue",
                                                                "stop_playing"):
        return "locators"
    if module == "scenes":
        if "scenes" in path or prop == "scenes" or fn == "create_scene":
            return "scenes: scenes"
        if "clip_slots" in path:
            if prop == "has_clip" or fn == "delete_clip":
                return "scenes: clear slots"
            if fn == "create_audio_clip":
                return "scenes: create clips"
            if prop in ("warping", "end_marker") and op != "set" or prop == "warping":
                return "scenes: warp + length"
            if op == "get":
                return "scenes: read-back"
            return "scenes: loop + markers"
        if "arrangement_clips" in path or prop == "arrangement_clips":
            return "scenes: clip colours"
        return "scenes: map tracks"
    if prop == "tempo":
        return "tempo"
    if fn in ("create_midi_track", "delete_track", "create_midi_clip") or op == "notes_add" \
            or "tracks 0 arrangement_clips 0" in path and prop == "name":
        return "energy map"
    if prop == "has_audio_input" or fn == "create_audio_track" or (
            path == "live_set" and prop == "tracks"):
        return "audio tracks"
    if fn == "create_audio_clip" or (prop == "name" and "arrangement" not in path) \
            or "mixer_device" in path:
        return "place clips"
    if prop in ("arrangement_clips", "start_time", "end_time"):
        return "find clips"
    if prop in ("color", "muted") or op == "warp_markers_set":
        return "colour, mute, warp"
    return "trim"

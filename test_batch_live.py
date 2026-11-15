"""`roadie load` and `roadie scenes` batch their Live operations; the Set they leave is the one
the one-request-per-operation code left.

The proof is a state model, not a wire diff (batching changes the wire by design): the code as it
was at Roadie 65a3754 (baseline_65a3754/, byte for byte, see the first test) and the code now each
run a whole `stems2live` load against their own copy of the same in-memory Live (fake_live.py),
from the same stems folder and analysis, and the tests compare:

  - the Set afterwards, every property either code can touch (fake_live.FakeLive.snapshot);
  - everything printed, which carries every per-item error;
  - the exit, if the run ended in one;
  - that the new code never leaned, inside one request, on an op earlier in that request (the
    model's hazard log: a new object addressed by index, a read-back in the request that wrote,
    a marker after `looping`, a toggle after a play-head move);
  - how many requests each sent (pinned, so a change that adds requests shows here).

The scenario `measured` is the run measured against Live on 2026-11-15: a ten-stem folder with
`--start-bar 1 --session` into Live's default Set. The baseline sends exactly the 439 requests
that run sent, op for op, which is what says the model asks what Live was asked.

Analysis is fixed (beat grid, centroids, sections) so both runs place the same thing; it is not
what is under test here.
"""

import collections
import copy
import hashlib
import importlib.util
import os
import sys

import numpy as np
import pytest
import soundfile

import beat_aligner
import fake_live
import sections as SEC
import session_view as V
import stems2live as S

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "baseline_65a3754")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD_S = _load("baseline_stems2live", os.path.join(BASE, "stems2live.py"))
OLD_V = _load("baseline_session_view", os.path.join(BASE, "session_view.py"))

PERIOD = 0.5                          # 120 BPM
DURATION = 32.0                       # seconds of audio per stem: 64 beats
# The ten stems of the measured folder's shape: a drums composite with its parts (the composite
# is muted), bass, and melodic stems. Centroids in Hz drive the order and colours.
TEN = {"bass": 66.0, "drums": 900.0, "drums_kick": 60.0, "drums_snare": 1500.0,
       "drums_hats": 6000.0, "drums_toms": 550.0, "other": 700.0, "piano": 450.0,
       "synths": 381.0, "vocals": 522.0}


def grid(drift=False):
    """What beat_aligner would report: the downbeat two beats in, so the trim is 1.2 s."""
    return {"bpm": 120.0, "period": PERIOD, "grid_t0": 0.2, "downbeat_sec": 1.2,
            "primary_stem": "drums_kick.wav", "phase_source": "kick", "detector": "heuristic",
            "downbeat_confidence": 0.6, "has_drift": drift, "drift_pct": 0.3 if drift else 0.0,
            "warp_map": [[0.4, 0.2], [32.4, 16.3], [64.0, 32.0]] if drift else []}


def secs(n):
    """`n` sections evenly over the 64 beats."""
    step = 64 // n
    return [{"beat": i * step, "time_sec": round(0.2 + i * step * PERIOD, 3),
             "label": "S%d" % i, "duration_beats": step} for i in range(n)]


def make_stems(path, names):
    os.makedirs(path)
    sr = 1000
    for k, name in enumerate(sorted(names)):
        y = 0.01 * np.sin(np.arange(int(DURATION * sr)) * (k + 1) * 0.01).astype("float32")
        soundfile.write(os.path.join(path, name + ".wav"), y, sr)
    return path


@pytest.fixture
def analysis(monkeypatch):
    """Fix the analysis both codes read, and hand back a setter for it."""
    state = {"grid": grid(), "sections": secs(2), "cent": dict(TEN)}
    monkeypatch.setattr(beat_aligner, "analyze_alignment",
                        lambda d, **kw: copy.deepcopy(state["grid"]))
    for mod in (S, OLD_S):
        monkeypatch.setattr(mod, "centroid",
                            lambda p: state["cent"][os.path.splitext(os.path.basename(p))[0]])
    monkeypatch.setattr(SEC, "detect_sections", lambda d, **kw: copy.deepcopy(state["sections"]))
    monkeypatch.setattr(SEC, "section_energy",
                        lambda d, s: {n: [0.5] * len(s) for n in state["cent"]})
    monkeypatch.setattr(SEC, "energy_notes",
                        lambda e, s, order=None: ([{"pitch": 60 + i, "start_time": 0.0,
                                                    "duration": 4.0, "velocity": 90}
                                                   for i, _n in enumerate(order)], list(order)))
    monkeypatch.setattr(SEC, "macro_energy_notes", lambda e, s: [])
    monkeypatch.setattr(SEC, "detect_transitions", lambda d, s, p: [])
    monkeypatch.setattr(SEC, "transition_notes", lambda t: [])

    def analyze_sections(d, period=None, grid_t0=None, **kw):
        return {"period": period or PERIOD, "grid_t0": 0.2 if grid_t0 is None else grid_t0,
                "duration_sec": DURATION, "bpm": 60.0 / (period or PERIOD),
                "sections": copy.deepcopy(state["sections"])}
    for mod in (V, OLD_V):
        monkeypatch.setattr(mod, "analyze_sections", analyze_sections)
    return state


def load(code, model, stems, argv, monkeypatch, capsys):
    """One `roadie load` by `code` ("old" or "new") against `model`. (printed, exit or None)."""
    mod, view = (OLD_S, OLD_V) if code == "old" else (S, V)
    model.install(monkeypatch)
    monkeypatch.setitem(sys.modules, "session_view", view)
    monkeypatch.setattr(sys, "argv", ["stems2live.py", stems] + argv)
    capsys.readouterr()
    exit_ = None
    try:
        mod.main()
    except SystemExit as e:
        exit_ = str(e.code)
    finally:
        if mod._LIVE is not None:
            mod._LIVE[1].close()
            mod._LIVE = None
    return capsys.readouterr().out, exit_


def both(make_model, stems, passes, monkeypatch, capsys):
    """Run the same passes (argv lists) with each code on its own fresh model. Per code:
    the model and, per pass, (printed, exit)."""
    out = {}
    for code in ("old", "new"):
        model = make_model()
        runs = [load(code, model, stems, argv, monkeypatch, capsys) for argv in passes]
        out[code] = (model, runs)
    return out


def assert_same(res):
    old, new = res["old"], res["new"]
    for (o_out, o_exit), (n_out, n_exit) in zip(old[1], new[1]):
        assert n_out == o_out
        assert n_exit == o_exit
    assert new[0].snapshot() == old[0].snapshot()
    assert old[0].hazards == [], "the model flags the baseline: the checks are wrong"
    assert new[0].hazards == []


def counts(model):
    """{op: n} over every request a model served."""
    return dict(collections.Counter(r["op"] for r in model.requests))


def leaf_counts(model):
    return dict(collections.Counter(op for r in model.requests for op, _p in r["leaves"]))


def phases(model):
    """{phase: requests}, a request filed under the step its first operation belongs to."""
    c = collections.Counter()
    for r in model.requests:
        op, p = r["leaves"][0] if r["leaves"] else (r["op"], {})
        c[fake_live.phase(r["module"], op, p)] += 1
    return dict(c)


# ------------------------------------------------------------------ the baseline

def test_the_baseline_is_roadie_65a3754_byte_for_byte():
    """The blob ids `git ls-tree 65a3754 stems2live.py session_view.py` prints."""
    want = {"stems2live.py": "59b569352771730432348371eaa581732ae847a2",
            "session_view.py": "a400b607b5813213d6e01bb74a793e625791dc3d"}
    for name, blob in want.items():
        with open(os.path.join(BASE, name), "rb") as f:
            data = f.read()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == blob


# ---------------------------------------------------------------- the equivalence

def test_measured_run_same_set_far_fewer_requests(tmp_path, analysis, monkeypatch, capsys):
    """The 2026-11-15 run: ten stems, --start-bar 1 --session, Live's default Set."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    res = both(fake_live.FakeLive, stems, [["--start-bar", "1", "--session"]],
               monkeypatch, capsys)
    assert_same(res)
    old, new = res["old"][0], res["new"][0]
    # What Live counted on the real run, op for op.
    assert counts(old) == {"set": 244, "get": 124, "call": 38, "count": 31, "ping": 1,
                           "warp_markers_set": 1}
    assert len(old.requests) == 439
    assert counts(new) == {"ping": 1, "warp_markers_set": 1, "get_batch": 12,
                           "transaction": 22, "set": 1}
    assert len(new.requests) == 37            # 7 of them the first session clip, on its own
    # The same operations reached Live; only how they were grouped changed. The new code reads
    # a few more (whole lists at once instead of stopping early) and nests its per-item steps.
    lo, ln = leaf_counts(old), leaf_counts(new)
    assert ln["call"] == lo["call"] and ln["set"] == lo["set"]
    # One undo step per request that changed something.
    assert old.undo_steps == 282 and new.undo_steps == 23


def test_rerun_with_drift_locators_energy_map_and_a_session_grid(tmp_path, analysis,
                                                                 monkeypatch, capsys):
    """Everything a load can do, twice over the same Set: the second pass at another bar finds
    stale clips, existing locators, its own energy map and a full Session grid."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    analysis["grid"] = grid(drift=True)
    analysis["sections"] = secs(9)            # nine scenes: one more than the Set has

    def make():
        m = fake_live.FakeLive(tracks=("midi", "midi", "audio", "midi", "audio", "audio"))
        m.song.is_playing = True
        m.add_cue(8.0, "old marker")
        return m
    every = ["--locators", "--energy-map", "--session", "--headroom", "3"]
    res = both(make, stems, [["--start-bar", "1"] + every, ["--start-bar", "3"] + every],
               monkeypatch, capsys)
    assert_same(res)
    first, second = res["new"][1][0][0], res["new"][1][1][0]
    assert "skipping 1 non-audio track at 3" in first and "creating 7 audio tracks" in first
    assert "locators: placed 9 of 9" in first and "energy map: 10 notes" in first
    assert "STALE AUDIO ON 10 TRACK(S)" in second and "warp 4 markers" in second
    assert "skipping 2 non-audio tracks at 2, 4" in second
    assert first.count("10/10 clips") == second.count("10/10 clips") == 9
    old, new = res["old"][0], res["new"][0]
    assert len(new.requests) * 8 < len(old.requests), (len(new.requests), len(old.requests))


def test_failed_items_inside_a_batch_are_reported_as_before(tmp_path, analysis,
                                                            monkeypatch, capsys):
    """One op of a batch fails for one track or clip: that item prints what it printed when
    each op was a request of its own, and every other item is written as before."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)

    def fault(op, p):
        path, fn, prop = p.get("path", ""), p.get("function"), p.get("property")
        args = p.get("args") or [""]
        if fn == "create_audio_clip" and "clip_slots" not in path and "piano" in str(args[0]):
            return RuntimeError("injected: no room")
        if fn == "create_audio_clip" and "clip_slots 1" in path and "other" in str(args[0]):
            return RuntimeError("injected: slot refused")
        if op == "set" and prop == "value" and path.startswith("live_set tracks 2 "):
            return RuntimeError("injected: fader stuck")
        if op == "set" and prop == "color" and path == "live_set tracks 3 arrangement_clips 0":
            return RuntimeError("injected: no colour")
        if op == "set" and prop == "loop_start" and path == "live_set tracks 5 arrangement_clips 0":
            return RuntimeError("injected: loop refused")
        if op == "set" and prop == "looping" and path == "live_set tracks 6 clip_slots 0 clip":
            return RuntimeError("injected: session loop refused")
        if op == "get" and prop == "end_time" and path == "live_set tracks 7 arrangement_clips 0":
            return RuntimeError("injected: no end")
        return None

    def make():
        m = fake_live.FakeLive()
        m.faults.append(fault)
        return m
    res = both(make, stems, [["--start-bar", "1", "--session"]], monkeypatch, capsys)
    assert_same(res)
    out = res["new"][1][0][0]
    assert "piano           ERROR: RuntimeError: injected: no room — track left untouched" \
        not in out                       # the message is Live's, without its type
    assert "ERROR: injected: no room — track left untouched" in out
    assert "headroom not applied: injected: fader stuck" in out
    assert "col=error" in out
    assert "[trim failed: loop_start: injected: loop refused]" in out
    assert "injected: slot refused" in out and "injected: session loop refused" in out
    assert out.count("8/9 clips") == 2


def _discarding(prop, path, value=0.0):
    """A model on which writing `prop` at `path` is accepted and does nothing (answers `value`)."""
    def make():
        m = fake_live.FakeLive()
        orig = m.dispatch

        def dispatch(op, params):
            if op == "set" and params.get("property") == prop and params.get("path") == path:
                return {"path": path, "property": prop, "value": value}
            return orig(op, params)
        m.dispatch = dispatch
        return m
    return make


def test_a_trim_that_takes_on_no_clip_stops_the_grid_after_one_clip_as_before(
        tmp_path, analysis, monkeypatch, capsys):
    """The first session clip goes through every step alone, so when the trim does not take
    on it (as it would on every clip, the bug that once shipped), the run ends after that one
    clip and the Set is exactly what the one-clip-at-a-time code left."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    res = both(_discarding("start_marker", "live_set tracks 2 clip_slots 0 clip"), stems,
               [["--start-bar", "1", "--session"]], monkeypatch, capsys)
    assert_same(res)
    out = res["new"][1][0][0]
    assert "session view failed (RuntimeError: trim did not take on track 2 scene 0" in out
    slots = [s for t in res["new"][0].snapshot()["tracks"] for s in t["slots"] if s]
    assert len(slots) == 1


def test_a_marker_set_that_reads_back_otherwise_ends_the_grid(tmp_path, analysis,
                                                             monkeypatch, capsys):
    """Every loop and marker write is checked against Live's reply, not only start_marker:
    a loop_end that did not take ends the run at that clip (the old code did not look)."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    model = _discarding("loop_end", "live_set tracks 4 clip_slots 1 clip", 99.0)()
    out, _ = load("new", model, stems, ["--start-bar", "1", "--session"], monkeypatch, capsys)
    assert "trim did not take on track 4 scene 1: asked loop_end " in out
    assert "reads 99.0 — the clip does not loop the section" in out
    assert model.hazards == []


def test_an_arrangement_set_that_reads_back_otherwise_fails_that_trim(tmp_path, analysis,
                                                                     monkeypatch, capsys):
    """In the arrangement the reply check fails the one trim, and the clip is left as the
    one-set-a-request code left it: the steps after the odd set still run."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    res = both(_discarding("loop_start", "live_set tracks 3 arrangement_clips 0", 7.0), stems,
               [["--start-bar", "1"]], monkeypatch, capsys)
    (o_out, _), (n_out, _) = res["old"][1][0], res["new"][1][0]
    assert res["new"][0].snapshot() == res["old"][0].snapshot()
    assert "[trim failed: loop_start did not take — asked 1.2000 s, reads 7.0000 s]" in n_out
    assert o_out.replace("[trim failed: loop_start did not take — asked 1.2000 s, reads "
                         "7.0000 s]", "trim 1.2000 s") == n_out.replace(
        "[trim failed: loop_start did not take — asked 1.2000 s, reads 7.0000 s]",
        "trim 1.2000 s")


def _auto_warp(model, path, times=1):
    """Live turning `warping` on by itself, without converting the markers, right after the
    trim's `looping=False` on `path` (as measured on 2026-11-15); `times` such flips."""
    orig, left = model.dispatch, [times]

    def dispatch(op, params):
        res = orig(op, params)
        if op == "set" and params.get("property") == "looping" and params.get("value") is False \
                and params["path"] == path and left[0]:
            left[0] -= 1
            clip = model.resolve(path)
            clip._v["warping"] = not clip._v["warping"]
        return res
    model.dispatch = dispatch


def test_warping_that_changes_after_it_was_read_redoes_the_trim_in_beats(tmp_path, analysis,
                                                                         monkeypatch, capsys):
    """The trim is written in the unit `warping` gave when it was read. If Live turned warping
    on afterwards (auto-warp landing late), the read-back sees it and the trim is written
    again in beats, so the marker means the same 1.2 s it was asked to."""
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick"])
    model = fake_live.FakeLive()
    path = "live_set tracks 3 arrangement_clips 0"
    _auto_warp(model, path)
    out, _ = load("new", model, stems, ["--start-bar", "1"], monkeypatch, capsys)
    beats = model.beats(1.2)
    assert "trim %.4f beats" % beats in out
    assert out.count("trim 1.2000 s") == 1
    assert "trim failed" not in out
    clip = model.resolve(path)
    assert clip.warping is True and clip.looping is False
    assert abs(clip.start_marker - beats) < 1e-6 and abs(clip.loop_start - beats) < 1e-6
    assert model.hazards == []


def test_warping_that_changes_twice_fails_the_trim(tmp_path, analysis, monkeypatch, capsys):
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick"])
    model = fake_live.FakeLive()
    _auto_warp(model, "live_set tracks 3 arrangement_clips 0", times=2)
    out, _ = load("new", model, stems, ["--start-bar", "1"], monkeypatch, capsys)
    assert ("[trim failed: warping changed again, to False, after the trim was redone in "
            "beats]") in out
    assert out.count("trim 1.2000 s") == 1


def test_rows_found_before_the_run_ends_still_print(tmp_path, analysis, monkeypatch, capsys):
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick", "vocals"])
    model = fake_live.FakeLive()
    model.faults.append(lambda op, p: RuntimeError("injected: no room")
                        if p.get("function") == "create_audio_clip"
                        and "vocals" in str((p.get("args") or [""])[0]) else None)
    model.cuts.append(lambda op, p: "Timeout" if op == "transaction" and any(
        q.get("property") == "color" for _o, q in fake_live.leaves(op, p)) else None)
    out, exit_ = load("new", model, stems, ["--start-bar", "1"], monkeypatch, capsys)
    assert "  4  vocals          ERROR: injected: no room — track left untouched" in out
    assert exit_.startswith("Live did not answer the last batch of changes")


def test_a_trim_that_does_not_take_still_ends_the_grid_at_that_clip(tmp_path, analysis,
                                                                    monkeypatch, capsys):
    """The one place the end state can differ, and only on this failure: a session trim that
    does not take on a LATER clip (the first one took) raises at that clip, as before, but the
    clips sent with it in the same requests were already written. Everything up to it is the
    same; the rest is only ever clips of that same batch, written as they would have been."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)

    def fault(op, p):
        if op == "set" and p.get("property") == "start_marker" \
                and p.get("path") == "live_set tracks 5 clip_slots 0 clip":
            return RuntimeError("never raised: the model just ignores it")
        return None

    def make():
        m = fake_live.FakeLive()
        orig = m.dispatch

        def dispatch(op, params):               # the write is accepted and discarded
            if fault(op, params) is not None:
                return {"path": params["path"], "property": "start_marker", "value": 0.0}
            return orig(op, params)
        m.dispatch = dispatch
        return m
    res = both(make, stems, [["--start-bar", "1", "--session"]], monkeypatch, capsys)
    (o_out, _), (n_out, _) = res["old"][1][0], res["new"][1][0]
    assert n_out == o_out
    assert "session view failed (RuntimeError: trim did not take on track 5 scene 0" in n_out
    old, new = res["old"][0].snapshot(), res["new"][0].snapshot()
    assert res["new"][0].hazards == []
    # Scene 0 in track order, as the clips were written: everything before track 5 is the same.
    for t, (o, n) in enumerate(zip(old["tracks"], new["tracks"])):
        assert o["clips"] == n["clips"] and o["name"] == n["name"]
        for s, (os_, ns) in enumerate(zip(o["slots"], n["slots"])):
            if s == 0 and t < 5:
                assert ns == os_
            if os_ is not None and not (s == 0 and t == 5):
                assert ns == os_
    assert new["tracks"][5]["slots"][0] == old["tracks"][5]["slots"][0]


def _placing(op, params):
    return op == "transaction" and any(
        p.get("function") == "create_audio_clip" and "clip_slots" not in p.get("path", "")
        for _o, p in fake_live.leaves(op, params))


def _filling_slots(op, params):
    return op == "transaction" and any(
        p.get("function") == "create_audio_clip" and "clip_slots" in p.get("path", "")
        for _o, p in fake_live.leaves(op, params))


@pytest.mark.parametrize("kind", ["Timeout", "ConnectionClosed"])
def test_an_unanswered_write_batch_ends_the_run_and_its_items_are_not_called_untouched(
        kind, tmp_path, analysis, monkeypatch, capsys):
    """Live's own Timeout reply (its main thread gave up waiting) or a closed connection after
    a batch of writes was sent: Live keeps running that task, so its changes may still land.
    The run stops there with that said, and no item of the batch is reported as untouched."""
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick", "vocals"])
    model = fake_live.FakeLive()
    model.cuts.append(lambda op, p: kind if _placing(op, p) else None)
    out, exit_ = load("new", model, stems, ["--start-bar", "1"], monkeypatch, capsys)
    assert exit_.startswith("Live did not answer the last batch of changes")
    assert "may still land" in exit_ and "left untouched" not in out
    assert "trk  stem" in out                            # the table header, and no rows
    # The batch did land: every clip is on its track, and the tracks are named.
    tracks = model.snapshot()["tracks"]
    assert [t["name"] for t in tracks][2:] == ["drums_kick", "bass", "vocals"]
    assert all(len(t["clips"]) == 1 for t in tracks[2:])
    assert len([r for r in model.requests if r["op"] == "transaction"]) == 2   # nothing after


@pytest.mark.parametrize("kind", ["Timeout", "ConnectionClosed"])
def test_an_unanswered_session_batch_ends_the_whole_run(kind, tmp_path, analysis, monkeypatch,
                                                         capsys):
    """The same in the Session grid: not "session view failed — the arrangement is unaffected"
    and on to the next step, but the end of the run."""
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick", "vocals"])
    model = fake_live.FakeLive()
    model.cuts.append(lambda op, p: kind if _filling_slots(op, p) else None)
    out, exit_ = load("new", model, stems, ["--start-bar", "1", "--session"], monkeypatch,
                      capsys)
    assert exit_.startswith("Live did not answer the last batch of changes")
    assert "session view failed" not in out
    assert model.snapshot()["tracks"][2]["slots"][0] is not None      # it landed


def test_a_write_batch_refused_outright_still_fails_only_its_items(tmp_path, analysis,
                                                                   monkeypatch, capsys):
    """A batch Live answered with a refusal (it did not run it) fails each of its items with
    that refusal, as each request did alone, and the run goes on."""
    stems = make_stems(str(tmp_path / "song_stems"), ["bass", "drums_kick", "vocals"])
    model = fake_live.FakeLive()
    orig = model.request

    def request(op, params, module=None):
        if _placing(op, params):
            model.requests.append({"op": op, "leaves": [], "module": module})
            raise type("HandlerLoadError", (Exception,), {})("handlers not loaded")
        return orig(op, params, module)
    model.request = request
    out, exit_ = load("new", model, stems, ["--start-bar", "1"], monkeypatch, capsys)
    assert exit_ is None
    for t, stem in ((2, "drums_kick"), (3, "bass"), (4, "vocals")):
        assert ("%3d  %-14s  ERROR: handlers not loaded — track left untouched"
                % (t, stem)) in out
    assert [t["name"] for t in model.snapshot()["tracks"]][2:] == ["3-Audio", "4-Audio",
                                                                  "5-Audio"]


def test_no_request_carries_more_than_the_limits(tmp_path, analysis, monkeypatch, capsys):
    """Live gives up waiting on a request after 15 s and still runs it, so a request carries at
    most HEAVY_PER_REQUEST audio loads or warps and OPS_PER_REQUEST operations."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    analysis["grid"] = grid(drift=True)
    analysis["sections"] = secs(9)
    model = fake_live.FakeLive()
    load("new", model, stems, ["--start-bar", "1", "--session"], monkeypatch, capsys)
    import live_batch
    for r in model.requests:
        heavy = sum(live_batch.weight(dict(p, op=op))[1] for op, p in r["leaves"])
        assert heavy <= live_batch.HEAVY_PER_REQUEST
        assert len(r["leaves"]) <= live_batch.OPS_PER_REQUEST


def test_trace(tmp_path, analysis, monkeypatch, capsys):
    """Not an assertion of numbers (the tests above pin them): the per-phase trace the report
    quotes, printed with -s. Kept as a test so it cannot rot."""
    stems = make_stems(str(tmp_path / "song_stems"), TEN)
    res = both(fake_live.FakeLive, stems, [["--start-bar", "1", "--session"]],
               monkeypatch, capsys)
    rows = []
    for code in ("old", "new"):
        m = res[code][0]
        rows.append((code, len(m.requests), counts(m), leaf_counts(m), phases(m), m.undo_steps))
    with capsys.disabled():
        for row in rows:
            print("\n%s: %d requests, undo steps %d\n  per op: %s\n  operations: %s\n"
                  "  per phase: %s" % (row[0], row[1], row[5], row[2], row[3], row[4]))
    assert all(r[1] > 0 for r in rows)


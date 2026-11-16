"""The model test_batch_live.py proves the batching against (fake_live.py) catches what batching
could break: each dependency the batching keeps in separate requests, put in one request, is
logged as a hazard, and the same ops sent one request each are not."""

import numpy as np
import pytest
import soundfile

import fake_live



def _bad_batches(wav):
    slot, clip = "live_set tracks 2 clip_slots 0", "live_set tracks 2 clip_slots 0 clip"
    arr = "live_set tracks 2 arrangement_clips 0"
    create = {"op": "call", "path": slot, "function": "create_audio_clip", "args": [wav]}
    place = {"op": "call", "path": "live_set tracks 2", "function": "create_audio_clip",
             "args": [wav, 0.0]}
    return {
        "a new clip addressed in the request that made it": (
            [], [create, {"op": "set", "path": clip, "property": "warping", "value": True}]),
        "a slot refilled in the request that emptied it": (
            [[create]], [{"op": "call", "path": slot, "function": "delete_clip", "confirm": True},
                         create]),
        "the length read in the request that warped": (
            [[create]], [{"op": "set", "path": clip, "property": "warping", "value": True},
                         {"op": "get", "path": clip, "property": "end_marker"}]),
        "a marker after looping": (
            [[create]], [{"op": "set", "path": clip, "property": "looping", "value": True},
                         {"op": "set", "path": clip, "property": "start_marker", "value": 1.0}]),
        "looping off after a marker": (
            [[place]], [{"op": "set", "path": arr, "property": "start_marker", "value": 1.0},
                        {"op": "set", "path": arr, "property": "looping", "value": False}]),
        "an arrangement clip listed in the request that placed it": (
            [], [place, {"op": "count", "path": "live_set tracks 2",
                         "child": "arrangement_clips"}]),
        "a new track named by index in the request that made it": (
            [], [{"op": "call", "path": "live_set", "function": "create_audio_track",
                  "args": [-1]},
                 {"op": "set", "path": "live_set tracks 4", "property": "name", "value": "x"}]),
        "a toggle in the request that moved the play head": (
            [[place]], [{"op": "set", "path": "live_set", "property": "current_song_time",
                         "value": 4.0},
                        {"op": "call", "path": "live_set", "function": "set_or_delete_cue"}]),
        "song_length read in the request that placed a clip": (
            [], [place, {"op": "get", "path": "live_set", "property": "song_length"}]),
    }


@pytest.mark.parametrize("what", sorted(_bad_batches("x")))
def test_the_model_flags_what_must_not_share_a_request(what, tmp_path):
    """Each dependency the batching keeps in separate requests, put in one: the model logs it.
    Sent as separate requests, the same ops log nothing."""
    wav = str(tmp_path / "a.wav")
    soundfile.write(wav, np.zeros(8000, dtype="float32"), 1000)
    before, ops = _bad_batches(wav)[what]
    for split in (False, True):
        m = fake_live.FakeLive()
        for setup in before:
            m.request("transaction", {"ops": setup})
        assert m.hazards == []
        if split:
            for op in ops:
                m.request("transaction", {"ops": [op]})
        else:
            m.request("transaction", {"ops": ops, "stop_on_error": False})
        assert bool(m.hazards) is not split, (what, split, m.hazards)


def test_a_toggle_sharing_the_play_head_move_fires_at_the_previous_position(tmp_path):
    """Why settle() is its own request, as the model has it: in one request the locator lands
    where the play head was, not where it was sent."""
    wav = str(tmp_path / "a.wav")
    soundfile.write(wav, np.zeros(8000, dtype="float32"), 1000)
    m = fake_live.FakeLive()
    m.request("call", {"path": "live_set tracks 2", "function": "create_audio_clip",
                       "args": [wav, 0.0]})
    m.request("transaction", {"ops": [
        {"op": "set", "path": "live_set", "property": "current_song_time", "value": 8.0},
        {"op": "call", "path": "live_set", "function": "set_or_delete_cue"}]})
    assert [c[0] for c in m.snapshot()["cues"]] == [0.0]
    m.request("set", {"path": "live_set", "property": "current_song_time", "value": 12.0})
    m.request("call", {"path": "live_set", "function": "set_or_delete_cue"})
    assert [c[0] for c in m.snapshot()["cues"]] == [0.0, 12.0]

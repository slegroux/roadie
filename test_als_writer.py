#!/usr/bin/env python3
"""Tests for als_writer — a synthetic template and one-second WAVs, no Live.

The synthetic template has only the paths als_writer reads and writes. The last
test runs the real template through the same code when it is on this machine.
"""

import gzip
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import als_writer as aw

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <NextPointeeId Value="100" />
    <Tracks>
      <MidiTrack Id="12"><Name><EffectiveName Value="1-MIDI" /></Name></MidiTrack>
      <AudioTrack Id="8">
        <Name>
          <EffectiveName Value="3-Audio" />
          <UserName Value="" />
          <MemorizedFirstClipName Value="" />
        </Name>
        <Color Value="20" />
        <DeviceChain>
          <Mixer>
            <Volume>
              <Manual Value="1" />
              <AutomationTarget Id="40"><LockEnvelope Value="0" /></AutomationTarget>
              <ModulationTarget Id="41" />
            </Volume>
            <Sends><TrackSendHolder Id="0" /></Sends>
          </Mixer>
          <MainSequencer>
            <Sample><ArrangerAutomation><Events /></ArrangerAutomation></Sample>
            <ClipSlotList><ClipSlot Id="0" /></ClipSlotList>
          </MainSequencer>
        </DeviceChain>
        <Pointee Id="42" />
      </AudioTrack>
      <AudioTrack Id="9"><Name><EffectiveName Value="4-Audio" /></Name></AudioTrack>
      <ReturnTrack Id="2" />
    </Tracks>
    <MainTrack Id="3">
      <DeviceChain><Mixer><Tempo><Manual Value="120" /></Tempo></Mixer></DeviceChain>
    </MainTrack>
    <Locators><Locators><Locator Id="0"><Time Value="9" /></Locator></Locators></Locators>
  </LiveSet>
</Ableton>
"""

PERIOD = 0.48                    # 125 BPM
RATE = 8000


@pytest.fixture
def stems(tmp_path):
    sf = pytest.importorskip("soundfile")
    d = tmp_path / "song_125bpm_stems"
    d.mkdir()
    out = []
    for name in ("drums_kick", "drums", "vocals"):
        p = d / (name + ".wav")
        sf.write(str(p), np.zeros(RATE * 2, dtype="float32"), RATE)   # 2 s
        out.append((name, str(p)))
    return out


@pytest.fixture
def template(tmp_path):
    p = tmp_path / "empty.als"
    with gzip.open(p, "wb") as f:
        f.write(TEMPLATE.encode())
    return str(p)


def build(template, stems, **kw):
    root = aw.read_set(template)
    args = dict(start_beat=32.0, trim_sec=0.0, period=PERIOD, warp_map=[], bpm=125.0)
    args.update(kw)
    aw.build_set(root, stems, **args)
    return root


def audio_tracks(root):
    return [t for t in root.find("LiveSet/Tracks") if t.tag == "AudioTrack"]


def v(el, path):
    return el.find(path).get("Value")


def test_one_track_per_stem_in_order_replacing_the_templates_audio_tracks(template, stems):
    root = build(template, stems)
    names = [v(t, "Name/EffectiveName") for t in audio_tracks(root)]
    assert names == ["drums_kick", "drums", "vocals"]
    tags = [t.tag for t in root.find("LiveSet/Tracks")]
    assert tags == ["MidiTrack"] + ["AudioTrack"] * 3 + ["ReturnTrack"]


def test_ids_are_unique_and_next_pointee_id_moves_past_them(template, stems):
    root = build(template, stems)
    ids = [int(e.get("Id")) for e in root.iter()
           if "Id" in e.attrib and (e.tag.endswith("Target") or e.tag == "Pointee")]
    assert len(ids) == len(set(ids)) == 3 * 3
    assert int(v(root, "LiveSet/NextPointeeId")) > max(ids)
    track_ids = [int(t.get("Id")) for t in root.find("LiveSet/Tracks")]
    assert len(track_ids) == len(set(track_ids))
    assert all(i not in track_ids for i in (3,))      # main track's id is not reused


def test_headroom_becomes_linear_gain_on_the_fader(template, stems):
    root = build(template, stems, headroom_db=6.0)
    gain = float(v(audio_tracks(root)[0], "DeviceChain/Mixer/Volume/Manual"))
    assert gain == pytest.approx(0.501187, abs=1e-5)


def test_clip_is_placed_on_the_bar_line_with_the_trim_taken_off_its_front(template, stems):
    trim = 0.2                                  # seconds -> 0.2/0.48 beats
    root = build(template, stems, trim_sec=trim)
    clip = audio_tracks(root)[0].find(".//AudioClip")
    trim_b, total_b = trim / PERIOD, 2.0 / PERIOD
    assert clip.get("Time") == "32"
    assert float(v(clip, "CurrentStart")) == 32
    assert float(v(clip, "CurrentEnd")) == pytest.approx(32 + total_b - trim_b, abs=1e-6)
    assert float(v(clip, "Loop/LoopStart")) == pytest.approx(trim_b, abs=1e-6)
    assert float(v(clip, "Loop/LoopEnd")) == pytest.approx(total_b, abs=1e-6)
    assert float(v(clip, "Loop/HiddenLoopEnd")) == pytest.approx(total_b, abs=1e-6)
    assert v(clip, "Loop/LoopOn") == "false"


def test_no_drift_clip_gets_the_two_marker_tempo_map(template, stems):
    clip = audio_tracks(build(template, stems))[0].find(".//AudioClip")
    marks = [(float(m.get("SecTime")), float(m.get("BeatTime")))
             for m in clip.iter("WarpMarker")]
    assert marks[0] == (0.0, 0.0)
    assert marks[1][1] == pytest.approx(1 / 32)
    assert marks[1][0] == pytest.approx(PERIOD / 32)
    assert [m.get("Id") for m in clip.iter("WarpMarker")] == ["2", "3"]


def test_a_measured_warp_map_is_written_as_is(template, stems):
    warp_map = [[0.0, 0.0], [4.0, 1.9], [8.0, 3.9]]           # [beat, sec]
    clip = audio_tracks(build(template, stems, warp_map=warp_map))[0].find(".//AudioClip")
    marks = [(float(m.get("SecTime")), float(m.get("BeatTime")))
             for m in clip.iter("WarpMarker")]
    assert marks == [(0.0, 0.0), (1.9, 4.0), (3.9, 8.0)]


def test_sample_reference_points_at_the_file(template, stems):
    clip = audio_tracks(build(template, stems, rel_dir="song_125bpm_stems"))[0].find(".//AudioClip")
    name, path = stems[0]
    assert v(clip, "SampleRef/FileRef/Path") == path
    assert v(clip, "SampleRef/FileRef/RelativePath") == "song_125bpm_stems/drums_kick.wav"
    assert v(clip, "SampleRef/FileRef/OriginalFileSize") == str(os.path.getsize(path))
    assert v(clip, "SampleRef/DefaultDuration") == str(RATE * 2)
    assert v(clip, "SampleRef/DefaultSampleRate") == str(RATE)


def test_a_muted_stem_gets_a_disabled_clip(template, stems):
    root = build(template, stems, mute={"drums"})
    got = {v(t, "Name/EffectiveName"): v(t.find(".//AudioClip"), "Disabled")
           for t in audio_tracks(root)}
    assert got == {"drums_kick": "false", "drums": "true", "vocals": "false"}


def test_tempo_and_locators_are_set(template, stems):
    locs = [{"time": 32.0, "name": "Intro"}, {"time": 96.0, "name": "Drop 1"}]
    root = build(template, stems, bpm=125.0042, locators=locs)
    assert v(root, "LiveSet/MainTrack/DeviceChain/Mixer/Tempo/Manual") == "125.004"
    got = [(v(l, "Time"), v(l, "Name")) for l in root.find("LiveSet/Locators/Locators")]
    assert got == [("32", "Intro"), ("96", "Drop 1")]        # the template's own is gone


def test_names_with_xml_characters_survive(template, tmp_path):
    sf = pytest.importorskip("soundfile")
    p = tmp_path / 'a"b&c.wav'
    sf.write(str(p), np.zeros(RATE, dtype="float32"), RATE)
    root = build(template, [('a"b&c', str(p))])
    out = tmp_path / "x.als"
    aw.write_set(root, str(out))
    back = aw.read_set(str(out))
    assert v(audio_tracks(back)[0].find(".//AudioClip"), "Name") == 'a"b&c'


def test_write_set_makes_gzipped_xml_that_reads_back(template, stems, tmp_path):
    out = tmp_path / "o.als"
    aw.write_set(build(template, stems), str(out))
    raw = gzip.open(out).read()
    assert raw.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    assert len(audio_tracks(aw.read_set(str(out)))) == 3


def test_a_template_without_an_audio_track_is_refused(tmp_path, stems):
    p = tmp_path / "bad.als"
    with gzip.open(p, "wb") as f:
        f.write(b"<Ableton><LiveSet><NextPointeeId Value='1'/><Tracks/></LiveSet></Ableton>")
    with pytest.raises(ValueError, match="AudioTrack"):
        aw.build_set(aw.read_set(str(p)), stems, 0.0, 0.0, PERIOD, [], 125.0)


def test_default_out_sits_next_to_the_stems_folder_named_after_it():
    assert aw.default_out("/a/b/Song_125bpm_stems/") == "/a/b/Song_125bpm.als"


def test_num_has_no_exponent_or_trailing_zeros():
    assert aw.num(32.0) == "32"
    assert aw.num(0.5) == "0.5"
    assert aw.num(0.0000123) == "0.0000123"
    assert aw.num(-0.0) == "0"


REAL = os.path.expanduser("~/Music/Ableton/User Library/Templates/empty.als")


@pytest.mark.skipif(not os.path.exists(REAL), reason="no real Live template here")
def test_the_real_template_builds_with_unique_ids(stems):
    root = aw.read_set(REAL)
    aw.build_set(root, stems, 32.0, 0.0, PERIOD, [], 125.0)
    ids = [e.get("Id") for e in root.iter()
           if "Id" in e.attrib and (e.tag.endswith("Target") or e.tag == "Pointee")]
    assert len(ids) == len(set(ids))
    assert int(root.find("LiveSet/NextPointeeId").get("Value")) > max(map(int, ids))
    assert len(audio_tracks(root)) == 3

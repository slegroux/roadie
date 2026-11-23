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
            <Speaker><Manual Value="true" /></Speaker>
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
      <AutomationEnvelopes><Envelopes>
        <AutomationEnvelope Id="0">
          <EnvelopeTarget><PointeeId Value="7" /></EnvelopeTarget>
          <Automation><Events><EnumEvent Id="0" Time="-63072000" Value="201" /></Events></Automation>
        </AutomationEnvelope>
        <AutomationEnvelope Id="1">
          <EnvelopeTarget><PointeeId Value="8" /></EnvelopeTarget>
          <Automation><Events><FloatEvent Id="0" Time="-63072000" Value="120" /></Events></Automation>
        </AutomationEnvelope>
      </Envelopes></AutomationEnvelopes>
      <DeviceChain><Mixer><Tempo>
        <Manual Value="120" />
        <AutomationTarget Id="8" />
      </Tempo></Mixer></DeviceChain>
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
    assert len(ids) == len(set(ids)) == 3 * 3 + 1      # +1: the main track's tempo target
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
    assert v(clip, "SampleRef/FileRef/RelativePathType") == "1"      # relative to the .als
    assert v(clip, "SampleRef/FileRef/OriginalFileSize") == str(os.path.getsize(path))
    assert v(clip, "SampleRef/DefaultDuration") == str(RATE * 2)
    assert v(clip, "SampleRef/DefaultSampleRate") == str(RATE)


def test_crc16_matches_the_catalog_check_value():
    assert aw.crc16(b"123456789") == 0xFEE8          # CRC-16/UMTS


def test_live_crc_reads_only_the_first_16_kib(tmp_path):
    head = bytes(range(256)) * 64                     # exactly 16384 bytes
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(head + b"tail one")
    b.write_bytes(head + b"a different tail")
    assert aw.live_crc(str(a)) == aw.live_crc(str(b)) == aw.crc16(head)


def test_the_clip_carries_the_files_size_and_crc(template, stems):
    clip = audio_tracks(build(template, stems))[0].find(".//AudioClip")
    assert v(clip, "SampleRef/FileRef/OriginalCrc") == str(aw.live_crc(stems[0][1]))


def test_a_muted_stem_gets_a_disabled_clip(template, stems):
    root = build(template, stems, mute={"drums"})
    got = {v(t, "Name/EffectiveName"): v(t.find(".//AudioClip"), "Disabled")
           for t in audio_tracks(root)}
    assert got == {"drums_kick": "false", "drums": "true", "vocals": "false"}


def test_tempo_and_locators_are_set(template, stems):
    locs = [{"time": 32.0, "name": "Intro"}, {"time": 96.0, "name": "Drop 1"}]
    root = build(template, stems, bpm=125.0042, locators=locs)
    assert v(root, "LiveSet/MainTrack/DeviceChain/Mixer/Tempo/Manual") == "125.004"
    # Live plays the tempo automation lane, not the manual value alone.
    events = [e.get("Value") for e in root.iter("FloatEvent")]
    assert events == ["125.004"]
    assert [e.get("Value") for e in root.iter("EnumEvent")] == ["201"]    # other lanes untouched
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
    aw.set_tempo(root, 125.004)
    main = root.find("LiveSet/MainTrack")
    target = main.find("DeviceChain/Mixer/Tempo/AutomationTarget").get("Id")
    lane = [e for e in main.iter("AutomationEnvelope")
            if e.find("EnvelopeTarget/PointeeId").get("Value") == target]
    assert lane and all(ev.get("Value") == "125.004"
                        for e in lane for ev in e.iter("FloatEvent"))


# --- energy lanes --------------------------------------------------------------

def test_energy_lane_puts_a_rack_on_the_track_and_automates_its_macro(template, stems):
    lanes = {"drums_kick": [(32.0, 127.0), (36.0, 40.0)], "vocals": [(32.0, 0.0)]}
    root = build(template, stems, energy=lanes, main_energy=[(32.0, 100.0)])
    for track in audio_tracks(root) + [root.find("LiveSet/MainTrack")]:
        name = track.find("Name/EffectiveName")
        name = name.get("Value") if name is not None else "Main"
        racks = track.findall("DeviceChain/DeviceChain/Devices/AudioEffectGroupDevice")
        if name == "drums":                       # no lane asked for: untouched
            assert not racks
            continue
        assert len(racks) == 1 and v(racks[0], "UserName") == "Energy"
        assert v(racks[0], "On/Manual") == "true"
        assert racks[0].find("Branches") is not None and not len(racks[0].find("Branches"))
        target = racks[0].find("MacroControls.0/AutomationTarget").get("Id")
        env = [e for e in track.iter("AutomationEnvelope")
               if v(e, "EnvelopeTarget/PointeeId") == target]
        assert len(env) == 1
        lane = track.find("DeviceChain/AutomationLanes/AutomationLanes/AutomationLane")
        assert (v(lane, "SelectedDevice"), v(lane, "SelectedEnvelope")) == ("1", "1")
        times = [float(ev.get("Time")) for ev in env[0].iter("FloatEvent")]
        assert times[0] < 0 and times[1:] == sorted(times[1:])   # default, then by time
    assert v(root, "LiveSet/AutomationMode") == "true"     # lanes show on open
    kick = audio_tracks(root)[0]
    vals = [float(ev.get("Value")) for ev in kick.iter("FloatEvent")]
    assert vals == [127.0, 127.0, 40.0]
    ids = [int(e.get("Id")) for e in root.iter()
           if "Id" in e.attrib and (e.tag.endswith("Target") or e.tag == "Pointee")]
    assert len(ids) == len(set(ids))
    assert int(v(root, "LiveSet/NextPointeeId")) > max(ids)


def test_bar_power_and_points_follow_loudness_on_the_bar_grid(tmp_path):
    sf = pytest.importorskip("soundfile")
    bar = int(RATE * PERIOD * 4)
    x = np.concatenate([0.5 * np.ones(bar), 0.05 * np.ones(bar), np.zeros(bar)])
    p = tmp_path / "s.wav"
    sf.write(str(p), x.astype("float32"), RATE)
    power = aw.bar_power(str(p), lambda sec: 32.0 + sec / PERIOD, first_beat=32.0, hop=256)
    assert sorted(power) == [8, 9, 10]                     # clips start on bar 9
    pts = aw.energy_points(power, aw.peak_db(power))
    assert [b for b, _ in pts] == [32.0, 36.0, 40.0]
    assert pts[0][1] == 127.0 and 0 < pts[1][1] < 127.0 and pts[2][1] == 0.0
    assert pts[1][1] == pytest.approx(127 * (1 - 20 / aw.ENERGY_RANGE_DB), abs=1)


# --- REF track: the original mix, muted ------------------------------------------

@pytest.fixture
def mix(stems, tmp_path):
    """The original mix beside the stems folder, as roadie split lays it out."""
    sf = pytest.importorskip("soundfile")
    p = tmp_path / "song_125bpm.wav"
    sf.write(str(p), np.zeros(RATE * 2, dtype="float32"), RATE)
    return str(p)


def clip_shape(track):
    clip = track.find(".//AudioClip")
    return (clip.get("Time"), v(clip, "CurrentStart"), v(clip, "CurrentEnd"),
            v(clip, "Loop/LoopStart"), v(clip, "Loop/LoopEnd"),
            [(m.get("SecTime"), m.get("BeatTime")) for m in clip.iter("WarpMarker")])


def test_ref_is_a_muted_last_track_aligned_like_the_stems(template, stems, mix, tmp_path):
    warp_map = [[0.0, 0.0], [4.0, 1.9], [8.0, 3.9]]
    out = tmp_path / "sets" / "song.als"
    out.parent.mkdir()
    rel = os.path.relpath(mix, out.parent)
    root = build(template, stems, trim_sec=0.2, warp_map=warp_map, ref=(mix, rel))
    tracks = audio_tracks(root)
    assert [v(t, "Name/EffectiveName") for t in tracks] == \
        ["drums_kick", "drums", "vocals", "REF"]
    ref = tracks[-1]
    assert v(ref, "DeviceChain/Mixer/Speaker/Manual") == "false"       # track muted
    assert all(v(t, "DeviceChain/Mixer/Speaker/Manual") == "true" for t in tracks[:-1])
    assert v(ref.find(".//AudioClip"), "Disabled") == "false"           # clip active
    assert clip_shape(ref) == clip_shape(tracks[0])
    assert v(ref, "Color") == "69"
    assert v(ref, "DeviceChain/Mixer/Volume/Manual") == v(tracks[0], "DeviceChain/Mixer/Volume/Manual")
    fref = ref.find(".//AudioClip/SampleRef/FileRef")
    assert v(fref, "RelativePathType") == "1"
    assert os.path.normpath(os.path.join(out.parent, v(fref, "RelativePath"))) == mix
    assert v(fref, "OriginalCrc") == str(aw.live_crc(mix))
    ids = [int(e.get("Id")) for e in root.iter()
           if "Id" in e.attrib and (e.tag.endswith("Target") or e.tag == "Pointee")]
    assert len(ids) == len(set(ids)) and int(v(root, "LiveSet/NextPointeeId")) > max(ids)


def test_ref_changes_nothing_computed_for_the_stems(template, stems, mix):
    lanes = {"drums_kick": [(32.0, 127.0)], "drums": [(32.0, 5.0)], "vocals": [(32.0, 0.0)]}
    plain = build(template, stems, energy=lanes, main_energy=[(32.0, 100.0)])
    withref = build(template, stems, energy=lanes, main_energy=[(32.0, 100.0)],
                    ref=(mix, "song_125bpm.wav"))
    assert not withref.findall(".//AudioTrack")[-1].findall(".//AudioEffectGroupDevice")  # no lane on REF
    withref.find("LiveSet/Tracks").remove(audio_tracks(withref)[-1])
    # Everything else is identical, apart from NextPointeeId, which moved past REF's ids.
    for r in (plain, withref):
        r.find("LiveSet/NextPointeeId").set("Value", "0")
    assert ET.tostring(plain) == ET.tostring(withref)


def test_find_ref_takes_the_mix_beside_the_stems_and_any_audio_extension(stems, tmp_path):
    folder = os.path.dirname(stems[0][1])
    assert aw.find_ref(folder) is None
    (tmp_path / "other.wav").write_bytes(b"x")
    assert aw.find_ref(folder) is None
    mp3 = tmp_path / "song_125bpm.mp3"
    mp3.write_bytes(b"x")
    assert aw.find_ref(folder) == str(mp3)


def run_main(tmp_path, monkeypatch, stems, *argv):
    """main() on the synthetic stems; the beat analysis is stubbed (silence has none)."""
    import beat_aligner
    import stems2live
    monkeypatch.setattr(beat_aligner, "analyze_alignment",
                        lambda *a, **k: {"bpm": 125.0, "period": PERIOD, "warp_map": []})
    monkeypatch.setattr(stems2live, "centroid", lambda p: 0.0)
    out = tmp_path / ("out%d.als" % len(list(tmp_path.glob("out*.als"))))
    aw.main([os.path.dirname(stems[0][1]), "-o", str(out), "--template",
             str(tmp_path / "empty.als"), "--no-align", "--no-locators",
             *argv])
    return aw.read_set(str(out)), out


def test_main_adds_ref_by_default_and_not_with_no_ref_or_without_a_mix(
        template, stems, mix, tmp_path, monkeypatch, capsys):
    with_ref, _ = run_main(tmp_path, monkeypatch, stems)
    assert [v(t, "Name/EffectiveName") for t in audio_tracks(with_ref)][-1] == "REF"
    skipped, out_a = run_main(tmp_path, monkeypatch, stems, "--no-ref")
    os.rename(mix, mix + ".away")
    capsys.readouterr()
    nomix, out_b = run_main(tmp_path, monkeypatch, stems)
    assert "no original mix" in capsys.readouterr().out      # one line, not an error
    # --no-ref and no mix write the same Set (the file that is not there changes nothing).
    assert gzip.open(out_a).read() == gzip.open(out_b).read()
    assert len(audio_tracks(skipped)) == 3


def test_main_ref_option_uses_that_file(template, stems, mix, tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    other = tmp_path / "elsewhere.flac"
    sf.write(str(other), np.zeros(RATE * 2, dtype="float32"), RATE)
    os.rename(mix, mix + ".away")
    root, out = run_main(tmp_path, monkeypatch, stems, "--ref", str(other))
    ref = audio_tracks(root)[-1]
    assert v(ref, "Name/EffectiveName") == "REF"
    assert v(ref.find(".//AudioClip"), "SampleRef/FileRef/Path") == str(other)


def test_main_warns_when_the_mix_length_differs(template, stems, mix, tmp_path, monkeypatch, capsys):
    sf = pytest.importorskip("soundfile")
    sf.write(mix, np.zeros(RATE * 3, dtype="float32"), RATE)      # 3 s against the stems' 2 s
    root, _ = run_main(tmp_path, monkeypatch, stems)
    assert "warning" in capsys.readouterr().out
    assert v(audio_tracks(root)[-1], "Name/EffectiveName") == "REF"


# --- the Live version a Set is written for -----------------------------------

HEADER_12_2 = ('<Ableton MajorVersion="5" MinorVersion="12.0_12203" SchemaChangeCount="3" '
               'Creator="Ableton Live 12.2.7" Revision="x">')
HEADER_12_4 = ('<Ableton MajorVersion="5" MinorVersion="12.0_12402" SchemaChangeCount="5" '
               'Creator="Ableton Live 12.4.6" Revision="y">')


def template_with_header(tmp_path, header, name):
    p = tmp_path / name
    with gzip.open(p, "wb") as f:
        f.write(TEMPLATE.replace("<Ableton>", header).encode())
    return str(p)


def written(template, stems, tmp_path, name):
    out = tmp_path / name
    aw.write_set(build(template, stems), str(out))
    return gzip.open(out).read()


def test_a_12_2_template_gets_the_clip_layout_unchanged(template, stems, tmp_path):
    t = template_with_header(tmp_path, HEADER_12_2, "t122.als")
    clip = audio_tracks(build(t, stems))[0].find(".//AudioClip")
    assert clip.find("AutomationEnvelopesListWrapper") is None
    assert clip.find("SampleRef/FileRef/SourceHint") is None
    # same clip bytes as a template with no header: the layout is chosen, not the text
    assert (written(t, stems, tmp_path, "a.als").split(b"<LiveSet>")[1]
            == written(template, stems, tmp_path, "b.als").split(b"<LiveSet>")[1])


def test_a_12_4_template_gets_the_two_elements_where_live_writes_them(template, stems, tmp_path):
    t = template_with_header(tmp_path, HEADER_12_4, "t124.als")
    for track in audio_tracks(build(t, stems)):
        clip = track.find(".//AudioClip")
        kids = [c.tag for c in clip]
        assert kids[kids.index("ScaleInformation") + 1:kids.index("SampleRef") + 1] \
            == ["AutomationEnvelopesListWrapper", "SampleRef"]
        assert clip.find("AutomationEnvelopesListWrapper").attrib == {"LomId": "0"}
        fileref = [c.tag for c in clip.find("SampleRef/FileRef")]
        assert fileref[-2:] == ["OriginalCrc", "SourceHint"]
        assert v(clip, "SampleRef/FileRef/SourceHint") == ""


def test_set_live_version_reads_creator_then_minor_version():
    mk = lambda **a: ET.Element("Ableton", a)
    assert aw.set_live_version(mk(Creator="Ableton Live 12.4.6")) == (12, 4, 6)
    assert aw.set_live_version(mk(MinorVersion="12.0_12203")) == (12, 2, 3)
    assert aw.set_live_version(mk()) is None


def make_app(tmp_path, name, version):
    import plistlib
    app = tmp_path / "apps" / name / "Contents"
    app.mkdir(parents=True)
    with open(app / "Info.plist", "wb") as f:
        plistlib.dump({"CFBundleShortVersionString": version + " (2026-09-10_0de5c8fa9a)"}, f)


def test_installed_live_version_is_the_newest_app_or_none(tmp_path):
    pattern = str(tmp_path / "apps" / "Ableton Live*.app")
    assert aw.installed_live_version(pattern) is None
    make_app(tmp_path, "Ableton Live 11 Suite.app", "11.3.2")
    make_app(tmp_path, "Ableton Live 12 Suite.app", "12.4.6")
    assert aw.installed_live_version(pattern) == (12, 4, 6)


@pytest.mark.parametrize("installed, warns", [((12, 4, 6), True), ((12, 2, 7), False),
                                              ((12, 1, 0), False), (None, False)])
def test_main_warns_once_when_the_installed_live_is_newer_than_the_template(
        template, stems, mix, tmp_path, monkeypatch, capsys, installed, warns):
    monkeypatch.setattr(aw, "installed_live_version", lambda: installed)
    template_with_header(tmp_path, HEADER_12_2, "empty.als")      # main() reads tmp_path/empty.als
    root, out = run_main(tmp_path, monkeypatch, stems)
    err = capsys.readouterr().err
    assert len(err.splitlines()) == (1 if warns else 0)
    assert (not warns) or ("12.2.7" in err and "12.4.6" in err and "save it once" in err)
    assert root.get("Creator") == "Ableton Live 12.2.7"           # the header is never rewritten

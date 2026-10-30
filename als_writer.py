#!/usr/bin/env python3
"""Write a Live project (.als) from a yt2stems folder, without Live running.

    als_writer.py <stems_dir> [-o OUT.als] [--template T.als] [--start-bar 9]

An .als is gzipped XML. This takes your empty template, replaces its audio
tracks with one per stem, puts one warped arrangement clip on each, sets the
tempo and the track gain, and adds a locator per detected section — the same
result `roadie load` builds through the Live bridge, minus the bridge.

WHAT IS COPIED, NOT INVENTED. The structure of a track comes from the template's
own AudioTrack, so everything Live puts in a track that this does not know about
(routing, mixer, sends) is carried over as Live wrote it. The clip layout was read
off a Live 12.2.7 project that already held stems. Nothing here has been opened
in Live by this code: the first file you make needs one open to confirm.

IDS. Elements that Live addresses by id (automation and modulation targets) must
be unique across the whole set, so every cloned track gets fresh ones counted up
from the set's NextPointeeId, which is then moved past them. Track ids are a
separate pool, taken as max + 1.

FILE CHECKSUM. Live stores each sample's size and a 16-bit CRC and calls the
file missing when they disagree, even if the Path resolves. The CRC is CRC-16
(poly 0x8005, init 0, not reflected) of the first 16384 bytes; this reproduces
all 122 references, FLAC and WAV, in a project Live saved. Written as 0, the
first version of this file opened as "media files are missing".

SAMPLE PATHS. RelativePathType 3 means "inside the project folder" (Live writes
it for Samples/...); a set that is not in a Live project has no such base, and
Live reported every stem missing. Type 1, a path relative to the .als file, is
what Live 12 writes for a sample beside or above the set (e.g. a glacier.als
referencing ../Teho - Glacier 3000.mp3), so that is what this writes.

KNOWN GAPS, each a guess to be corrected the first time Live says otherwise:
  * Clip colours are palette INDICES (not RGB like the bridge's), taken per stem
    name from a project Live produced. Unknown names get 0.
  * Clips use warp mode 0 (Beats) with the two-marker tempo map Live writes on
    import, or the measured warp map when the track drifts.
"""

import os
import sys

# Re-exec with the local .venv python if present, like the other scripts, so
# numpy/soundfile/librosa resolve when this is run through the roadie command.
_real_script = os.path.realpath(__file__)
_venv_py = os.path.join(os.path.dirname(_real_script), ".venv", "bin", "python")
if os.path.exists(_venv_py) and sys.executable != _venv_py and "VIRTUAL_ENV" not in os.environ:
    os.execv(_venv_py, [_venv_py] + sys.argv)

import argparse
import copy
import gzip
import re
import string
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

DEFAULT_TEMPLATE = os.path.expanduser(
    "~/Music/Ableton/User Library/Templates/empty.als")

# Palette indices per stem, copied from a project Live itself saved.
COLOURS = {"drums_kick": 14, "bass": 14, "drums": 14, "drums_toms": 5,
           "drums_crash": 6, "drums_snare": 21, "drums_hh": 9, "drums_ride": 65,
           "synths": 4, "vocals": 5}

# Track ids and pointee ids are different pools; see the module docstring.
TRACK_TAGS = ("MidiTrack", "AudioTrack", "ReturnTrack", "MainTrack",
              "PreHearTrack")

CLIP = string.Template("""<AudioClip Id="${clip_id}" Time="${start}">
<LomId Value="0" />
<LomIdView Value="0" />
<CurrentStart Value="${start}" />
<CurrentEnd Value="${end}" />
<Loop>
<LoopStart Value="${trim}" />
<LoopEnd Value="${total}" />
<StartRelative Value="0" />
<LoopOn Value="false" />
<OutMarker Value="${total}" />
<HiddenLoopStart Value="0" />
<HiddenLoopEnd Value="${total}" />
</Loop>
<Name Value="${name}" />
<Annotation Value="" />
<Color Value="${colour}" />
<LaunchMode Value="0" />
<LaunchQuantisation Value="0" />
<TimeSignature>
<TimeSignatures>
<RemoteableTimeSignature Id="0">
<Numerator Value="4" />
<Denominator Value="4" />
<Time Value="0" />
</RemoteableTimeSignature>
</TimeSignatures>
</TimeSignature>
<Envelopes>
<Envelopes />
</Envelopes>
<ScrollerTimePreserver>
<LeftTime Value="0" />
<RightTime Value="0" />
</ScrollerTimePreserver>
<TimeSelection>
<AnchorTime Value="0" />
<OtherTime Value="0" />
</TimeSelection>
<Legato Value="false" />
<Ram Value="false" />
<GrooveSettings>
<GrooveId Value="-1" />
</GrooveSettings>
<Disabled Value="${disabled}" />
<VelocityAmount Value="0" />
<FollowAction>
<FollowTime Value="4" />
<IsLinked Value="true" />
<LoopIterations Value="1" />
<FollowActionA Value="4" />
<FollowActionB Value="0" />
<FollowChanceA Value="100" />
<FollowChanceB Value="0" />
<JumpIndexA Value="1" />
<JumpIndexB Value="1" />
<FollowActionEnabled Value="false" />
</FollowAction>
<Grid>
<FixedNumerator Value="1" />
<FixedDenominator Value="16" />
<GridIntervalPixel Value="20" />
<Ntoles Value="2" />
<SnapToGrid Value="true" />
<Fixed Value="false" />
</Grid>
<FreezeStart Value="0" />
<FreezeEnd Value="0" />
<IsWarped Value="true" />
<TakeId Value="0" />
<IsInKey Value="true" />
<ScaleInformation>
<Root Value="0" />
<Name Value="0" />
</ScaleInformation>
<SampleRef>
<FileRef>
<RelativePathType Value="1" />
<RelativePath Value="${relpath}" />
<Path Value="${path}" />
<Type Value="2" />
<LivePackName Value="" />
<LivePackId Value="" />
<OriginalFileSize Value="${size}" />
<OriginalCrc Value="${crc}" />
</FileRef>
<LastModDate Value="${mtime}" />
<SourceContext />
<SampleUsageHint Value="0" />
<DefaultDuration Value="${frames}" />
<DefaultSampleRate Value="${rate}" />
<SamplesToAutoWarp Value="0" />
</SampleRef>
<Onsets>
<UserOnsets />
<HasUserOnsets Value="false" />
</Onsets>
<WarpMode Value="0" />
<GranularityTones Value="30" />
<GranularityTexture Value="65" />
<FluctuationTexture Value="25" />
<TransientResolution Value="6" />
<TransientLoopMode Value="2" />
<TransientEnvelope Value="100" />
<ComplexProFormants Value="100" />
<ComplexProEnvelope Value="128" />
<Sync Value="true" />
<HiQ Value="true" />
<Fade Value="false" />
<Fades>
<FadeInLength Value="0" />
<FadeOutLength Value="0" />
<ClipFadesAreInitialized Value="true" />
<CrossfadeInState Value="0" />
<FadeInCurveSkew Value="0" />
<FadeInCurveSlope Value="0" />
<FadeOutCurveSkew Value="0" />
<FadeOutCurveSlope Value="0" />
<IsDefaultFadeIn Value="false" />
<IsDefaultFadeOut Value="false" />
</Fades>
<PitchCoarse Value="0" />
<PitchFine Value="0" />
<SampleVolume Value="1" />
<WarpMarkers>
${markers}
</WarpMarkers>
<SavedWarpMarkersForStretched />
<MarkersGenerated Value="true" />
<IsSongTempoLeader Value="false" />
</AudioClip>""")


def num(x):
    """A number the way Live writes it: no trailing zeros, no exponent."""
    t = ("%.12f" % float(x)).rstrip("0").rstrip(".")
    return "0" if t in ("", "-0") else t


def crc16(data):
    """CRC-16/UMTS (poly 0x8005, init 0, no reflection) of `data`."""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def live_crc(path):
    """The OriginalCrc Live stores for a sample: crc16 of its first 16 KiB."""
    with open(path, "rb") as f:
        return crc16(f.read(16384))


def attr(s):
    return escape(str(s), {'"': "&quot;"})


# --- warp map and geometry ---------------------------------------------------

def warp_markers(period, warp_map):
    """[(sec, beat)] for a clip. No drift: the two-marker map Live writes on
    import, which pins the clip to the set tempo. Drift: the measured map, with
    a leading (0, 0) added if it does not start there."""
    if not warp_map or len(warp_map) < 2:
        return [(0.0, 0.0), (period / 32.0, 1.0 / 32.0)]
    out = [(float(sec), float(beat)) for beat, sec in warp_map]
    if out[0][0] > 0:
        out.insert(0, (0.0, 0.0))
    return out


def sample_beats(seconds, period, warp_map):
    """Beat time of `seconds` into the sample, through the warp map if any."""
    from stems2live import sample_time_to_beats
    return sample_time_to_beats(seconds, warp_map, period)


def audio_info(path):
    """(frames, sample_rate) of an audio file."""
    try:
        import soundfile as sf
        i = sf.info(path)
        return int(i.frames), int(i.samplerate)
    except ImportError:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True).stdout.strip().split(",")
        rate = int(out[0])
        return int(round(float(out[1]) * rate)), rate


def make_clip(name, path, relpath, start, trim_beats, period, warp_map, colour,
              disabled=False, clip_id=0, info=None):
    """One arrangement AudioClip element for `path`, placed at beat `start`."""
    frames, rate = info or audio_info(path)
    total = sample_beats(frames / float(rate), period, warp_map)
    length = total - trim_beats
    markers = "\n".join(
        '<WarpMarker Id="%d" SecTime="%s" BeatTime="%s" />'
        % (2 + i, num(sec), num(beat))
        for i, (sec, beat) in enumerate(warp_markers(period, warp_map)))
    xml = CLIP.substitute(
        clip_id=clip_id, start=num(start), end=num(start + length),
        trim=num(trim_beats), total=num(total), name=attr(name),
        colour=colour, disabled="true" if disabled else "false",
        relpath=attr(relpath), path=attr(path), size=os.path.getsize(path), crc=live_crc(path),
        mtime=int(os.path.getmtime(path)), frames=frames, rate=rate,
        markers=markers)
    return ET.fromstring(xml)


# --- the set -----------------------------------------------------------------

def read_set(path):
    with gzip.open(path, "rb") as f:
        return ET.fromstring(f.read())


def _max_id(root, tags):
    return max([int(e.get("Id")) for t in tags for e in root.iter(t)
                if e.get("Id", "").lstrip("-").isdigit()] or [0])


def _is_pointee(el):
    return "Id" in el.attrib and (el.tag.endswith("Target") or el.tag == "Pointee")


def clone_track(proto, track_id, ids, name, colour, gain, clips):
    """A copy of `proto` renamed `name`, with fresh pointee ids from `ids` (an
    iterator), `gain` (linear) on its fader, and `clips` as its arrangement."""
    t = copy.deepcopy(proto)
    t.set("Id", str(track_id))
    for el in t.iter():
        if _is_pointee(el):
            el.set("Id", str(next(ids)))
    nm = t.find("Name")
    nm.find("EffectiveName").set("Value", name)
    nm.find("UserName").set("Value", name)
    nm.find("MemorizedFirstClipName").set("Value", name)
    t.find("Color").set("Value", str(colour))
    t.find("DeviceChain/Mixer/Volume/Manual").set("Value", num(gain))
    events = t.find("DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events")
    for c in list(events):
        events.remove(c)
    for c in clips:
        events.append(c)
    return t


def set_tempo(root, bpm):
    """Set the song tempo in both places Live keeps it: the fader's manual value
    and the tempo automation lane's starting event. Live plays the automation
    lane, so setting only the manual value leaves the song at the template's 120."""
    bpm = num(round(bpm, 3))
    main = root.find("LiveSet/MainTrack")
    tempo = main.find("DeviceChain/Mixer/Tempo")
    tempo.find("Manual").set("Value", bpm)
    target = tempo.find("AutomationTarget").get("Id")
    for env in main.iter("AutomationEnvelope"):
        if env.find("EnvelopeTarget/PointeeId").get("Value") == target:
            for ev in env.iter("FloatEvent"):
                ev.set("Value", bpm)


def set_locators(root, locators):
    """Replace the set's locators with [{time, name}]."""
    outer = root.find("LiveSet/Locators")
    inner = outer.find("Locators")
    for c in list(inner):
        inner.remove(c)
    for i, loc in enumerate(locators):
        e = ET.SubElement(inner, "Locator", Id=str(i))
        for tag, val in (("LomId", "0"), ("Time", num(loc["time"])),
                         ("Name", loc["name"]), ("Annotation", ""),
                         ("IsSongStart", "false")):
            ET.SubElement(e, tag, Value=val)


def build_set(root, stems, start_beat, trim_sec, period, warp_map, bpm,
              headroom_db=6.0, locators=None, rel_dir=None, mute=()):
    """Edit `root` (a parsed template) in place. `stems` is the ordered
    [(name, absolute_path)]. Returns the list of track names written."""
    tracks_el = root.find("LiveSet/Tracks")
    audio = [t for t in tracks_el if t.tag == "AudioTrack"]
    if not audio:
        raise ValueError("the template has no AudioTrack to copy")
    proto, slot = audio[0], list(tracks_el).index(audio[0])
    for t in audio:
        tracks_el.remove(t)

    pointee = root.find("LiveSet/NextPointeeId")
    ids = iter(range(max(int(pointee.get("Value")), _max_id(root, ["Pointee"]) + 1),
                     10 ** 9))
    next_track = _max_id(root, TRACK_TAGS) + 1
    gain = 10 ** (-abs(headroom_db) / 20.0)
    trim_beats = sample_beats(trim_sec, period, warp_map)

    for i, (name, path) in enumerate(stems):
        rel = os.path.join(rel_dir, os.path.basename(path)) if rel_dir \
            else os.path.basename(path)
        clip = make_clip(name, path, rel, start_beat, trim_beats, period,
                         warp_map, COLOURS.get(name, 0), disabled=name in mute)
        tracks_el.insert(slot + i, clone_track(
            proto, next_track + i, ids, name, COLOURS.get(name, 0), gain, [clip]))

    pointee.set("Value", str(next(ids)))
    set_tempo(root, bpm)
    if locators is not None:
        set_locators(root, locators)
    return [n for n, _ in stems]


def write_set(root, out):
    ET.indent(root, space="\t")
    data = ('<?xml version="1.0" encoding="UTF-8"?>\n'.encode()
            + ET.tostring(root, encoding="utf-8", short_empty_elements=True) + b"\n")
    with gzip.GzipFile(out, "wb", mtime=0) as f:
        f.write(data)


# --- command line ------------------------------------------------------------

def default_out(stems_dir):
    stems_dir = os.path.abspath(stems_dir).rstrip("/")
    base = re.sub(r"_stems$", "", os.path.basename(stems_dir))
    return os.path.join(os.path.dirname(stems_dir), base + ".als")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write a Live project from a stems folder, without Live")
    ap.add_argument("stems_dir")
    ap.add_argument("-o", "--out", help="output .als (default: next to the "
                    "stems folder, named after it)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="empty Live set to copy (default: %(default)s)")
    ap.add_argument("--tempo", type=float, default=None,
                    help="override the measured tempo")
    ap.add_argument("--start-bar", type=int, default=9,
                    help="bar the clips start on (default 9)")
    ap.add_argument("--headroom", type=float, default=6.0,
                    help="dB to pull every stem track down by (default 6)")
    ap.add_argument("--no-align", action="store_true",
                    help="place clips at the raw file start")
    ap.add_argument("--no-locators", action="store_true",
                    help="skip section detection")
    ap.add_argument("-f", "--force", action="store_true",
                    help="overwrite the output if it exists")
    args = ap.parse_args(argv)

    stems_dir = os.path.abspath(args.stems_dir)
    out = os.path.abspath(args.out) if args.out else default_out(stems_dir)
    if not os.path.isdir(stems_dir):
        sys.exit("no such stems folder: %s" % args.stems_dir)
    if not os.path.isfile(args.template):
        sys.exit("template not found: %s (see --template)" % args.template)
    if os.path.exists(out) and not args.force:
        sys.exit("%s exists; --force to overwrite" % out)

    paths = {}
    for f in sorted(os.listdir(stems_dir)):
        if f.endswith((".flac", ".wav")):
            paths.setdefault(os.path.splitext(f)[0], os.path.join(stems_dir, f))
    if not paths:
        sys.exit("no .wav/.flac files in %s" % stems_dir)

    from beat_aligner import analyze_alignment
    from stems2live import (arrange_stems, bar_aligned_placement, centroid,
                            section_grid)
    grid = analyze_alignment(stems_dir, override_tempo=args.tempo)
    bpm = args.tempo or grid["bpm"]
    period = float(grid["period"])
    warp_map = grid.get("warp_map") or []

    start, trim = 0.0, 0.0
    if not args.no_align:
        start, trim, _ = bar_aligned_placement(grid, args.start_bar, bpm)

    cent = {n: centroid(p) for n, p in paths.items()}
    order = arrange_stems(sorted(paths), cent)
    mute = ({"drums"} if "drums" in order
            and any(n.startswith("drums_") for n in order) else set())

    locators = None
    if not args.no_locators:
        from sections import detect_sections, sections_to_locators
        secs = detect_sections(stems_dir, **section_grid(grid, args.no_align))
        locators = [dict(l, time=max(0.0, l["time"]))
                    for l in sections_to_locators(secs, period, start - trim / period)]

    root = read_set(args.template)
    rel_dir = os.path.relpath(stems_dir, os.path.dirname(out))
    build_set(root, [(n, paths[n]) for n in order], start, trim, period, warp_map,
              bpm, args.headroom, locators, rel_dir, mute)
    write_set(root, out)

    print("wrote %s" % out)
    print("  %d tracks, %.3f BPM, clips at bar %.2f, %s locators"
          % (len(order), bpm, start / 4.0 + 1.0,
             "no" if locators is None else len(locators)))
    if mute:
        print("  'drums' clip is muted: the drum parts sum back to it")


if __name__ == "__main__":
    main()

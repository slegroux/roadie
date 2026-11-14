#!/usr/bin/env python3
"""Roadie's MCP server: the analysis tools that need a reference track, its stems or samples.

    roadie mcp                      run it (stdio); an MCP client starts it, not you
    claude mcp add roadie -- roadie mcp

Four tools: sample_analyze, sample_compare, reference_profile and session_compare. The first three
have the names, arguments, defaults and output of Sideman's tools of the same name; session_compare
is Sideman's mix_report with a reference profile. Measuring the Set on its own (audio_analyze,
audio_compare, mix_report with no reference or an audio file) stays in Sideman's server: register
both. Everything runs locally on audio files; the one message sent to Live is a `get` of a clip's
file_path, when a source is given as a LOM path, through `sideman.client.Live`.

Needs the `mcp` package, which install.sh puts in .venv from requirements-mcp.txt.
"""

import os
import sys

# Re-exec with local .venv python if present: `mcp`, `sideman` and numpy are installed there,
# not in the system python the shebang finds.
_real_script = os.path.realpath(__file__)
_venv = os.path.join(os.path.dirname(_real_script), ".venv")
_venv_py = os.path.join(_venv, "bin", "python")
if os.path.realpath(sys.prefix) == os.path.realpath(_venv):
    os.environ.pop("ROADIE_REEXEC", None)    # arrived; later scripts may hop again
elif os.path.exists(_venv_py) and not os.environ.get("ROADIE_REEXEC"):
    os.environ["ROADIE_REEXEC"] = "1"        # one hop, whatever the venv looks like
    os.execv(_venv_py, [_venv_py] + sys.argv)

import json
from typing import Any

MCP_HINT = ("roadie mcp needs the `mcp` package in .venv: run ./install.sh, or "
            "uv pip install --python .venv/bin/python -r requirements-mcp.txt")

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] != "mcp":
        raise
    sys.exit(MCP_HINT)

try:
    import sideman.client as client
except ModuleNotFoundError as _e:
    if (_e.name or "").split(".")[0] != "sideman":
        raise
    sys.exit("sideman is not installed in .venv: run ./install.sh")

from roadie_analysis import oneshot, profile

mcp = MCPServer(
    "roadie",
    instructions=(
        "Analysis that needs a reference track, its stems or samples: drum one-shots (sample_analyze, "
        "sample_compare), a reference track measured once per section (reference_profile), and a capture "
        "of the Set compared with that profile (session_compare). Measuring the Set on its own "
        "(audio_analyze, audio_compare, mix_report) is Sideman's server. A source is a file path or an "
        "audio clip's LOM path such as \"live_set tracks 3 clip_slots 0 clip\"."
    ),
)

PROFILE_ONLY = ("session_compare needs a reference profile (from reference_profile: the path of its "
                ".profile.json, its JSON text or the profile itself); a report against an audio file, or with "
                "no reference, is Sideman's mix_report")


class AnalysisError(ToolError):
    """Every failure a tool reports. It must be a ToolError: the MCP SDK treats any other exception
    as a crash and sends the client only "Error executing tool <name>", dropping the diagnosis."""


def _audio_file(source: str) -> str:
    """An absolute file path, from a path or an audio clip's LOM path."""
    source = source.strip()
    if not source:
        raise AnalysisError("no audio source given: pass a file path or an audio clip's LOM path")
    if source.split(" ", 1)[0] in ("live_set", "live_app", "app_view"):
        try:
            got = client.Live().get(source, "file_path")
        except (client.LiveError, OSError, ValueError) as e:
            raise AnalysisError("cannot read the file_path of %s from Live: %s" % (source, e))
        path = got.get("value") if isinstance(got, dict) else got
        if not path:
            raise AnalysisError("%s has no file_path; only audio clips do" % source)
    else:
        path = os.path.abspath(os.path.expanduser(source))
    if not os.path.isfile(path):
        raise AnalysisError("no such audio file: %s" % path)
    return path


def _file(path: str, what: str) -> str:
    """An absolute path of an existing plain file."""
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isfile(path):
        raise AnalysisError("no such %s file: %s" % (what, path))
    return path


def _profile(source: str, hit: int | str | None) -> dict:
    path = _audio_file(source)
    try:
        return oneshot.profile(path, hit)
    except MemoryError:
        raise AnalysisError("not enough memory to profile %s; trim it to the hits" % path)
    except (OSError, RuntimeError, ValueError) as e:   # LibsndfileError is a RuntimeError
        raise AnalysisError("cannot measure %s: %s" % (path, e))


@mcp.tool()
def sample_analyze(source: str, hit: int | str | None = None) -> str:
    """Profile a drum one-shot, locally, in the terms a synth is programmed in.

    `source` is a mono or stereo file path or an audio clip's LOM path (the
    clip's whole source file). Every measure is relative to the hit itself:
    times from its onset (the first 1 ms within 40 dB of its loudest), levels
    against its own peak or fundamental, so level, leading silence and sample
    rate do not change them. For a mix or a whole track use Sideman's audio_analyze.

    A take may hold several hits: `hits` counts them, `onsets_ms` places
    them, and `hit` (from 0) picks one; by default the first that is not
    `truncated` (cut by the next hit or the file's end before it falls
    40 dB; its falls past that point are null). A hit is split off only
    when it rises 15 dB within 10 ms over a tail that has fallen that far,
    so a tremolo or a tom's beating stays one hit, and so does a fast roll.
    `hit="clean"` picks a typical complete hit. Of the hits that are not cut off,
    those within 6 dB of the loudest (a ghost note is not clean); of those the
    ones starting from quiet (the 20 ms before the onset 20 dB under the peak),
    if any; of those the ones with room to the next onset (or the end of the
    file) of at least 90 % of the median; of those the ones with a peak within
    3 dB of the median peak, if any; the earliest. A take whose first hit begins
    at full level while a later one is more than 3 dB louder is read as starting
    inside a hit, and that first hit is not offered. The profile's `hit` and a
    note say which; a take with no complete hit is refused with "no clean hit".

    Envelope: peak_db (dBFS), attack_ms (onset to the first moment within
    1 dB of the envelope peak, over whole periods), fall_10_ms / fall_20_ms / fall_40_ms (from the
    peak to the last moment above peak - 10/20/40 dB), duration_ms (to -60 dB; None on a cut hit).
    Pitch: `pitched` (false for noise, a cymbal or an inharmonic bell: then no
    pitch and no harmonics, only `partials`), pitch_body_hz (40-150 ms, the
    fundamental: the lowest strong partial, not the loudest, and a weak one
    under strong overtones when its own series shows it), pitch_start_hz (at
    the onset, from zero crossings; pitch_start_at_ms says where it was read
    when a click or a messy start moves it later), pitch_drop_st (start minus
    body), pitch_settle_ms (when it comes within 0.5 st of the body pitch and
    stays 10 ms), pitch_track_hz at 0, 5, 10, 20, 40, 80, 150, 300 ms (null
    where nothing reads it).
    Tone, on the body: harmonics_db (2-8, dB against the fundamental; null
    when the body holds under 4 periods; a harmonic with no peak reads the
    floor) with harmonics_floor_db (the spectral floor around each), `partials` (strongest
    non-harmonic peaks, Hz and dB), noise_db (energy outside the tonal peaks,
    dB against the total) and flatness (0 tone .. 0.56 white noise) for attack
    (0-30 ms) and body, click_db (attack power above 2 kHz against the
    attack's total: a click or any bright attack), bands_db (six bands as
    audio_analyze, against the hit's total) and width (0 mono .. 1 all side).
    A value that cannot be measured is null and `notes` says why.
    """
    return json.dumps(_profile(source, hit), indent=2)


@mcp.tool()
def sample_compare(a: str, b: str, hit_a: int | str | None = None, hit_b: int | str | None = None) -> str:
    """Compare two drum hits in synth terms: `a` the candidate (usually a synth
    take), `b` the reference sample. Each is a file path or an audio clip's
    LOM path; hit_a / hit_b pick a hit from a take holding several (see
    sample_analyze, also for "clean"; by default the first that is not cut off).

    Returns both sample_analyze profiles, `diff` (a - b: pitch in semitones,
    times as {"ms", "ratio" a/b}, levels in dB, width) and `readings`: one
    plain line per difference past its threshold, largest first, such as
    "body pitch 2.1 st below the reference" or "falls 20 dB in 80 ms,
    reference 140 ms (43 % shorter)". Thresholds: body pitch 0.3 st, pitch
    drop 1 st, attack / falls / duration / settle 15 % and 2 ms, harmonics
    and noise share 3 dB, click 6 dB, width 0.1; one side pitched and the
    other not is a reading of its own. Levels at the method's floor on both
    sides are not read (harmonics under -50 dB or near the noise, noise share
    under -25 dB, click under -40 dB). A pitch-drop line gives the time each
    drop takes: "drops 17.0 st over 3.0 ms (a click)" is a stick transient,
    the same drop over 40 ms is a kick's glide. When either hit is unpitched
    (hats, shakers, claps) the six-band balance is read too, 3 dB against
    the other bands. Starts read at different times are compared from the
    later one; two pitches an octave or a twelfth apart on the same partials
    are one line. Level, leading silence and a 44.1/48/96 kHz rate give no
    reading (at 22.05 kHz the highs are empty and read low). Change the patch,
    record again, compare again until `readings` is empty. Local only.
    """
    return json.dumps(oneshot.compare(_profile(a, hit_a), _profile(b, hit_b)), indent=2)


@mcp.tool()
def session_compare(manifest: dict[str, Any] | str, reference: dict[str, Any] | str,
                    out_dir: str | None = None) -> str:
    """Per-section comparison of a capture of the Set with a reference track that has stems: Sideman's
    mix report, plus how each element and drum sound sits against the reference's.

    `manifest` (a dict, or the path of a JSON file) places time-aligned files
    on the Set's bar grid:
        {"tempo": 122, "beats_per_bar": 4, "offset_s": 0.0,
         "sources": {"master": "/abs/master.wav", "kick": "...", "bass": "..."},
         "sections": [{"name": "INTRO A", "start_bar": 0, "bars": 8}, ...]}
    start_bar counts from the files' bar 0 (at offset_s). "master" is required;
    a "kick" (or "drums") plus a "bass" source adds masking. A source may be a list
    of files, summed into one role (a bass recorded on two tracks).

    The mix layer, per section: integrated / short-term max LUFS, true peak, crest, width,
    low-end correlation, bands and side-minus-mid; with kick and bass, their
    40-120 Hz overlap (0 separate or sidechained .. 1 fully stacked; flagged
    above 0.5 unless the quieter part is more than 20 dB down). Over the whole capture, the spectral change between 4-bar
    blocks, flagging runs of more than 8 bars where nothing changes.

    `reference` is a reference profile, and is required (from reference_profile:
    the path of its .profile.json, JSON text, or the profile itself; one made by
    another version is refused with "rebuild it with reference_profile"). A report
    against an audio file, or with no reference, needs only the Set: that is Sideman's
    mix_report, and it is refused here. Sections pair
    BY NAME with the profile's; a manifest section may carry
    "reference": {"section": <profile section name>, "from_bar": N} (both
    optional) to pair with another section or start N bars into it. A shorter
    section pairs with the first bars (noted: the reference's bands, width and
    loudness are for the whole section); one with no match, or longer than the
    profile's, gets the mix layer only, with a note. Each paired section gets
    the mix compared with the reference's section (bands, width, loudness) and:
    `elements`: every source in both the manifest and the profile as a ratio to
    the kick in dB (reference, ours, difference; negative = quieter than the
    kick), roles in the reference only / in ours only, the ratio per 4 bars, and
    `gaps` (ratios more than 1 dB apart, largest first); each row also has the
    element's width on both sides (0 mono .. 1 all side; the reference's is the
    whole section's) and `width_gaps` lists those more than 0.1 apart, separate
    from the level gaps. Presence is judged against our element sources together
    and the profile's stems together (never the master capture, which carries the
    master chain's gain): present when a quarter of the bars are within 40 dB of
    them, a `trace` when a quarter are within 60 dB, read on each bar's core (20 ms
    inside both lines, so a note ending on a line or a hit just before it never
    makes a role present in the bar or section before). A trace on both sides, or a
    trace against an absent role, is listed apart and never a gap; a present role
    against a trace is compared and can be a gap (`trace_side` says which side).
    Where the kick is out on either side (absent, or 30 dB under the loudest other
    role) the `yardstick` is the loudest role in both ("Elements against the bass
    (kick out)"); with none the table is left out. A kick out on one side only is
    also listed in the one-side lists, with how far down it is. The `master` source is never
    ratioed against the elements, since it sits after the master chain. If the
    master capture is the reference mix (per-bar curve and bands within 0.3 dB
    after removing the level offset, width within 0.01), the section says
    "the capture is the reference: check solos". With no `kick` source the table
    is left out with a note and the mix layer still reports.
    `drums`: for each of kick, toms, snare, hh, clap, rim in both the manifest
    and the profile and present on both sides in the paired bars, sample_compare's
    readings between a typical clean hit of ours and of the reference stem over the
    section. Each side is cut with a beat of pre-roll and post-roll and only hits
    that start inside the section count; the reference's hit is picked first and
    ours is the one at the same place in the section (within 30 ms) when there is
    one, else a hit rising within 30 ms of that place (noted), else our own with a
    note that the hits are not at the same place. A hit's earliness plus the
    capture's read offset must stay within 20 ms. No clean hit on a side gives a note and no readings. `reference_stale` names profile
    inputs that changed since it was built (a drum role whose stem changed is
    skipped). The numbers are measurements, not moves: the ear decides which
    gaps matter.

    Writes mix_report.json and mix_report.md (default: next to the master
    file) and returns the report. Local only.
    """
    if isinstance(reference, dict) or (isinstance(reference, str) and reference.lstrip().startswith("{")):
        ref = reference                                   # the profile, or its JSON text
    elif isinstance(reference, str) and reference.strip().lower().endswith(".json"):
        ref = _file(reference, "profile")
    else:
        raise AnalysisError(PROFILE_ONLY)
    try:
        return json.dumps(profile.session_compare(manifest, ref, out_dir), indent=2)
    except MemoryError:
        raise AnalysisError("not enough memory for this capture; report fewer sections at a time")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, IndexError,
            ArithmeticError) as e:
        raise AnalysisError("cannot build the session comparison: %s: %s" % (type(e).__name__, e))


@mcp.tool()
def reference_profile(mix: str, stems: dict[str, str | list[str]], tempo: float, t0: float,
                      sections: list[dict[str, Any]], beats_per_bar: int = 4,
                      out: str | None = None) -> str:
    """Measure a reference track and its stems once, per section, so a capture of
    the Set can be compared with it: pass the result to session_compare as `reference`.

    `mix` is the reference mix; `stems` maps a role to a file path or a list of
    paths summed into that role: {"kick": "k.wav", "bass": "b.wav", "drums":
    ["hh.wav", "snare.wav", "toms.wav"]}. Roles are free-form; `kick` is the
    yardstick for every level against the kick. `tempo` (BPM) and `t0` (seconds
    from the start of the files to bar 0) place the bar grid; `sections` are
    {"name", "start_bar", "bars"} as in a capture manifest, with unique names
    (session_compare pairs sections by name). All files must share one sample rate;
    of different lengths they are trimmed to the shortest, and `notes` says so.

    Per section: the mix's level (mean power, dBFS), integrated LUFS, six
    bands, width and per-bar levels; for each role its level, `vs_kick_db`,
    per-bar levels, bands and width. A role is present in a section when at least
    a quarter of its bars (and one) are within 40 dB of the stems' power sum, a
    trace when a quarter are within 60 dB (listed, not a gap against an absent
    role in session_compare), else absent there (not -120 dB); the mix is not the
    reference for this, so a limiter's make-up gain does not move it. Presence is
    read on each bar's core, 20 ms inside both lines (`core_bar_db`). A section or
    bar's level window starts at the onset of a hit that rises within 20 ms before
    its line, so a hit that starts a few ms early belongs to its own bar, and a
    note ending on the line stays in the bar before (the same rules on the capture
    side); every role keeps its per-bar levels, absent ones too.

    Writes <mix name>.profile.json next to the mix (`out`: a .json path or a
    folder), which holds everything above plus each input's path, mtime and
    size (session_compare reports in `reference_stale` any that changed since). This
    tool returns a summary: `json_path` (pass it to session_compare as `reference`),
    tempo, t0, `notes`, and per section the roles present, trace and absent.
    Local only."""
    try:
        files = {role: ([_audio_file(p) for p in v] if isinstance(v, list) else _audio_file(v))
                 for role, v in stems.items()}
        prof = profile.reference_profile(
            _audio_file(mix), files, tempo, t0, sections, beats_per_bar,
            os.path.abspath(os.path.expanduser(out)) if out else None)
    except MemoryError:
        raise AnalysisError("not enough memory for these files; profile fewer stems at a time")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, IndexError,
            ArithmeticError) as e:
        raise AnalysisError("cannot build the reference profile: %s: %s" % (type(e).__name__, e))
    rows = []
    for sec in prof["sections"]:
        roles = sec["roles"]
        rows.append({"name": sec["name"], "start_bar": sec["start_bar"], "bars": sec["bars"],
                     "mix_level_db": sec["mix"]["level_db"],
                     "present": [r for r, e in roles.items() if not e["absent"] and not e["trace"]],
                     "trace": [r for r, e in roles.items() if not e["absent"] and e["trace"]],
                     "absent": [r for r, e in roles.items() if e["absent"]]})
    return json.dumps({"json_path": prof["json_path"], "tempo": prof["tempo"], "t0": prof["t0"],
                       "beats_per_bar": prof["beats_per_bar"], "sample_rate": prof["sample_rate"],
                       "roles": list(prof["inputs"]["stems"]), "notes": prof["notes"], "sections": rows},
                      indent=2)


def main() -> None:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__.strip())
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

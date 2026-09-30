#!/usr/bin/env python3
"""Tests for sections.py — phrase snapping, dedup, and arrangement conversion.

The synthetic-audio test at the bottom is the only one that runs the real
pipeline; everything above it is pure arithmetic and runs instantly. That split
is deliberate — the bugs that actually put a locator in the wrong place are
off-by-one and unit-conversion bugs, and those are testable without audio.
"""

import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sections import (BEATS_PER_BAR, FULL_REL, QUIET_REL, _letter,
                      analyze_sections, build_beat_grid_times,
                      choose_segment_count, detect_sections, label_for,
                      sections_to_locators, snap_boundaries, stem_paths)

PHRASE_BEATS = 32  # 8 bars


# --------------------------------------------------------------------------
# snapping
# --------------------------------------------------------------------------

def test_snap_rounds_to_nearest_phrase_multiple():
    got = snap_boundaries([0, 30, 65, 127], PHRASE_BEATS, n_beats=512)
    assert got == [0, 32, 64, 128]
    assert all(b % PHRASE_BEATS == 0 for b in got)


def test_snap_rounds_up_not_down():
    """A boundary detected early must not fall back a whole phrase.

    Risers and filter sweeps start before the phrase they announce, so raw
    boundaries sit a beat or two EARLY. Flooring would put the locator 8 bars
    before the section it marks.
    """
    assert snap_boundaries([62, 63], PHRASE_BEATS, n_beats=512) == [0, 64]


def test_snap_deduplicates_collisions():
    """Several raw boundaries land in one phrase; only one locator may result."""
    got = snap_boundaries([60, 62, 64, 66, 70], PHRASE_BEATS, n_beats=512)
    assert got == [0, 64]
    assert len(got) == len(set(got))


def test_snap_always_includes_beat_zero():
    assert snap_boundaries([100], PHRASE_BEATS, n_beats=512)[0] == 0
    assert snap_boundaries([], PHRASE_BEATS, n_beats=512) == [0]


def test_snap_output_is_strictly_increasing():
    got = snap_boundaries([300, 40, 41, 200, 199, 12], PHRASE_BEATS, n_beats=512)
    assert all(got[i] < got[i + 1] for i in range(len(got) - 1))


def test_snap_drops_boundaries_past_the_end():
    """Rounding UP off the end of the track must not invent a locator."""
    assert snap_boundaries([250], PHRASE_BEATS, n_beats=256) == [0]


def test_snap_honours_other_phrase_lengths():
    assert snap_boundaries([30, 65], 16, n_beats=512) == [0, 32, 64]


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_snap_counts_phrases_from_the_downbeat_phase(phase):
    """Phrase lines sit at `phase + 32n`, not at multiples of 32 from beat 0."""
    got = snap_boundaries([30 + phase, 65 + phase, 127 + phase], PHRASE_BEATS,
                          n_beats=512, phase=phase)
    assert got == [phase, 32 + phase, 64 + phase, 128 + phase]


def test_snap_phase_zero_is_the_old_behaviour():
    raw = [300, 40, 41, 200, 199, 12, 250, 3]
    assert snap_boundaries(raw, PHRASE_BEATS, 512, phase=0) == \
        snap_boundaries(raw, PHRASE_BEATS, 512)


def test_snap_folds_a_pickup_boundary_into_the_first_section():
    """A raw boundary rounding onto the downbeat itself must not add a 1-3 beat
    section before it."""
    assert snap_boundaries([2, 5], PHRASE_BEATS, n_beats=512, phase=3) == [3]


def test_snap_rejects_nonpositive_phrase():
    with pytest.raises(ValueError):
        snap_boundaries([10], 0, n_beats=512)


# --------------------------------------------------------------------------
# arrangement-time conversion
# --------------------------------------------------------------------------

def _sections(*times):
    return [{"time_sec": t, "label": "A quiet"} for t in times]


def test_locators_zero_offset_is_seconds_over_period():
    locs = sections_to_locators(_sections(0.0, 15.36), period=0.48,
                                start_offset_beats=0.0)
    assert [l["time"] for l in locs] == [0.0, 32.0]


def test_locators_apply_a_nonzero_offset():
    """The offset is what puts a locator on Live's bar line rather than near it.

    0.237 s at a 0.479989 s period is 0.4938 beats into the sample; stems2live's
    phrase nudge of 27.5062 beats is chosen precisely so that lands on beat 28.
    """
    locs = sections_to_locators(_sections(0.237, 15.597), period=0.479989,
                                start_offset_beats=27.5062)
    assert locs[0]["time"] == pytest.approx(28.0, abs=1e-3)
    assert locs[1]["time"] == pytest.approx(60.0, abs=1e-3)


def test_locator_offset_shifts_every_locator_by_the_same_amount():
    secs = _sections(0.0, 15.36, 61.44)
    a = sections_to_locators(secs, 0.48, 0.0)
    b = sections_to_locators(secs, 0.48, 31.5)
    deltas = [y["time"] - x["time"] for x, y in zip(a, b)]
    assert deltas == pytest.approx([31.5, 31.5, 31.5])


def test_locator_offset_is_required_not_defaulted():
    """Forgetting the offset must be impossible to do by accident."""
    with pytest.raises(TypeError):
        sections_to_locators(_sections(0.0), 0.48)


def test_locators_carry_the_label_as_the_name():
    secs = [{"time_sec": 0.0, "label": "C full"}]
    assert sections_to_locators(secs, 0.48, 0.0)[0]["name"] == "C full"
    assert sections_to_locators(secs, 0.48, 0.0, name_prefix="s: ")[0]["name"] \
        == "s: C full"


def test_locators_reject_a_bad_period():
    with pytest.raises(ValueError):
        sections_to_locators(_sections(0.0), 0.0, 0.0)


# --------------------------------------------------------------------------
# adaptive segment count
# --------------------------------------------------------------------------

def test_segment_count_scales_with_track_length():
    short = choose_segment_count(3 * 60 / 0.48, PHRASE_BEATS)    # ~3 min
    long_ = choose_segment_count(10 * 60 / 0.48, PHRASE_BEATS)   # ~10 min
    assert long_ > short


def test_segment_count_is_clamped_at_both_ends():
    assert choose_segment_count(64, PHRASE_BEATS) >= 4
    assert choose_segment_count(60 * 60 / 0.48, PHRASE_BEATS) <= 32


def test_segment_count_counts_phrases_not_seconds():
    """Same number of bars at half the tempo must ask for the same k."""
    assert (choose_segment_count(640, PHRASE_BEATS)
            == choose_segment_count(640, PHRASE_BEATS))


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def test_labels_bucket_relative_energy():
    assert label_for(QUIET_REL - 0.01, 0) == "A quiet"
    assert label_for(1.0, 1) == "B mid"
    assert label_for(FULL_REL + 0.01, 2) == "C full"


def test_letters_wrap_past_z():
    assert _letter(0) == "A"
    assert _letter(25) == "Z"
    assert _letter(26) == "AA"


# --------------------------------------------------------------------------
# beat grid
# --------------------------------------------------------------------------

def test_grid_times_start_at_t0_and_stay_inside_the_audio():
    times, k_start = build_beat_grid_times(0.5, 0.25, duration_sec=10.0)
    assert k_start == 0
    assert times[0] == pytest.approx(0.25)
    assert times[-1] <= 10.0
    assert np.allclose(np.diff(times), 0.5)


def test_grid_skips_slots_before_the_sample_start():
    times, k_start = build_beat_grid_times(0.5, -0.75, duration_sec=10.0)
    assert k_start == 2
    assert times[0] == pytest.approx(0.25)
    assert times.min() >= 0


# --------------------------------------------------------------------------
# end to end on synthetic audio
# --------------------------------------------------------------------------

PERIOD = 0.5           # 120 BPM
BLOCK_BEATS = 64       # 16 bars
N_BLOCKS = 6
SYNTH_SR = 22050


def _make_alternating_track(path):
    """Loud/quiet alternation every 16 bars, at an exactly known grid.

    Both the loudness AND the timbre alternate: a section change that only
    the RMS row could see would test one row of 26 and prove little about the
    MFCC/chroma rows that actually decide the boundaries.
    """
    rng = np.random.default_rng(0)
    beat_n = int(round(PERIOD * SYNTH_SR))
    blocks = []
    for b in range(N_BLOCKS):
        loud = (b % 2 == 0)
        beats = []
        for i in range(BLOCK_BEATS):
            t = np.arange(beat_n) / float(SYNTH_SR)
            if loud:
                env = np.exp(-t * 25.0)
                kick = 0.9 * np.sin(2 * np.pi * 55.0 * t) * env
                hat = 0.25 * rng.standard_normal(beat_n) * np.exp(-t * 60.0)
                chord = 0.20 * sum(np.sin(2 * np.pi * f * t)
                                   for f in (220.0, 277.2, 330.0))
                beats.append(kick + hat + chord)
            else:
                beats.append(0.06 * np.sin(2 * np.pi * 110.0
                                           * (np.arange(beat_n) + i * beat_n)
                                           / float(SYNTH_SR)))
        blocks.append(np.concatenate(beats))
    sf.write(path, np.concatenate(blocks).astype(np.float32), SYNTH_SR)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_stems")
    _make_alternating_track(str(d / "mix.wav"))
    # Live's cached analysis file: the listing must skip it, not hand it to
    # libsndfile. Byte content is deliberately not valid audio.
    (d / "mix.wav.asd").write_bytes(b"\x00\x01not audio")
    return str(d)


def test_stem_listing_ignores_live_analysis_files(synth_dir):
    assert sorted(stem_paths(synth_dir)) == ["mix"]


def test_detects_the_16_bar_alternation(synth_dir):
    """Every loud/quiet transition, and nothing between them."""
    sections = detect_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                               period=PERIOD, grid_t0=0.0)
    beats = [s["beat"] for s in sections]
    assert beats == [b * BLOCK_BEATS for b in range(N_BLOCKS)]


def test_synthetic_boundaries_are_valid(synth_dir):
    sections = detect_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                               period=PERIOD, grid_t0=0.0)
    beats = [s["beat"] for s in sections]
    bars = [s["bar"] for s in sections]
    assert len(set(beats)) == len(beats)
    assert all(beats[i] < beats[i + 1] for i in range(len(beats) - 1))
    assert all((bar - 1) % 8 == 0 for bar in bars)
    assert bars == [b // BEATS_PER_BAR + 1 for b in beats]


def test_synthetic_labels_follow_the_alternation(synth_dir):
    """`level` is the raw loudness bucket and must track the alternation."""
    sections = detect_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                               period=PERIOD, grid_t0=0.0)
    levels = [s["level"].split()[1] for s in sections]
    assert levels == ["full", "quiet"] * (N_BLOCKS // 2)
    rel = [s["rel_energy"] for s in sections]
    assert min(rel[0::2]) > max(rel[1::2])


def test_structural_labels_name_the_form(synth_dir):
    """`label` is the arrangement reading, distinct from the loudness bucket.

    On a loud/quiet alternation the interesting assertion is not which words
    come out but that the two ends are pinned by convention and that every
    quiet-after-loud reads as a break — that is the rule doing the work.
    """
    sections = detect_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                               period=PERIOD, grid_t0=0.0)
    labels = [s["label"] for s in sections]
    assert labels[0] == "Intro"
    assert labels[-1] == "Outro"
    # No label is the bare "X quiet" loudness form any more.
    assert not any(l.split()[0] in ("A", "B", "C", "D") and l.split()[-1]
                   in ("quiet", "mid", "full") for l in labels)
    for i in range(1, len(sections) - 1):
        if sections[i]["level"].endswith("quiet") and \
           sections[i - 1]["level"].endswith("full"):
            assert labels[i].startswith("Break")


def test_arrangement_labels_are_pure_and_handle_degenerate_input():
    from sections import arrangement_labels
    assert arrangement_labels([]) == []
    assert arrangement_labels([1.0]) == ["Intro"]
    # Loud straight after quiet is the drop gesture.
    assert arrangement_labels([0.2, 1.5, 1.6])[1] == "Drop"
    # A mid section is a BUILD only because a full one follows it; the same
    # energy reads as plain "Mid" when it does not. This is the look-ahead rule,
    # and it is why labelling cannot be done per-section in one pass.
    assert arrangement_labels([0.2, 1.0, 1.5])[1] == "Build"
    assert arrangement_labels([0.2, 1.0, 0.3, 1.5])[1] == "Mid"
    # Repeats are numbered so two drops are distinguishable in the arrangement.
    assert arrangement_labels([0.2, 1.5, 0.3, 1.5, 0.2]).count("Drop 1") == 1


def test_section_times_match_the_grid(synth_dir):
    sections = detect_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                               period=PERIOD, grid_t0=0.0)
    for s in sections:
        assert s["time_sec"] == pytest.approx(s["beat"] * PERIOD, abs=1e-3)


def test_end_to_end_locators_land_on_bar_lines(synth_dir):
    """With grid_t0 = 0 and no nudge, every locator is a whole number of beats."""
    res = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                           period=PERIOD, grid_t0=0.0)
    locs = sections_to_locators(res["sections"], res["period"], 0.0)
    assert [l["time"] for l in locs] == pytest.approx(
        [float(b * BLOCK_BEATS) for b in range(N_BLOCKS)], abs=1e-3)


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_phase_locators_land_on_bar_lines_for_every_downbeat(synth_dir, phase):
    """REGRESSION: locators snapped from grid beat 0 sat `phase` beats before the
    bar line whenever the downbeat was not on grid beat 0.

    Runs the real chain stems2live uses: bar_aligned_placement for the clip
    position and trim, sections for the boundaries, sections_to_locators for
    arrangement time. --start-bar 2 so a locator before the clip is visible
    rather than clamped to bar 1.
    """
    from stems2live import bar_aligned_placement
    grid = {"period": PERIOD, "grid_t0": 0.0, "downbeat_sec": phase * PERIOD}
    start, trim, _db = bar_aligned_placement(grid, start_bar=2, tempo=120.0)
    res = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                           period=PERIOD, grid_t0=0.0,
                           downbeat_sec=grid["downbeat_sec"])
    locs = [loc["time"] for loc in sections_to_locators(res["sections"], PERIOD,
                                                        start - trim / PERIOD)]
    assert locs[0] == pytest.approx(start, abs=1e-3), "first section at the clip"
    for t in locs:
        assert t % BEATS_PER_BAR == pytest.approx(0.0, abs=1e-3), locs
    assert res["sections"][0]["bar"] == 1


def test_phase_without_a_downbeat_is_grid_beat_zero(synth_dir):
    """No downbeat in hand keeps the old snap exactly."""
    a = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                         period=PERIOD, grid_t0=0.0)
    b = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                         period=PERIOD, grid_t0=0.0, downbeat_sec=0.0)
    assert a == b


@pytest.mark.parametrize("t0", [-PERIOD, -0.2])
def test_negative_t0_reports_times_where_the_features_start(synth_dir, t0):
    """A refit can leave grid_t0 just under zero. The grid then starts at the
    first slot inside the audio (k_start > 0); each section's time_sec must be
    the time of the beat its features were taken from."""
    res = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                           period=PERIOD, grid_t0=t0)
    grid_times, _k = build_beat_grid_times(PERIOD, t0, res["duration_sec"])
    for sec in res["sections"]:
        assert sec["time_sec"] == pytest.approx(grid_times[sec["beat"]],
                                                abs=1e-3), sec
    # And `beat` is counted from the grid_t0 it reports, as consumers assume.
    for sec in res["sections"]:
        assert sec["time_sec"] == pytest.approx(
            res["grid_t0"] + sec["beat"] * PERIOD, abs=1e-3)


def test_negative_t0_of_one_whole_beat_is_the_zero_grid(synth_dir):
    """grid_t0 = -period is the same physical grid as grid_t0 = 0."""
    a = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                         period=PERIOD, grid_t0=0.0)
    b = analyze_sections(synth_dir, phrase_bars=8, n_segments=N_BLOCKS,
                         period=PERIOD, grid_t0=-PERIOD)
    assert [s["time_sec"] for s in b["sections"]] == \
        [s["time_sec"] for s in a["sections"]]


# --------------------------------------------------------------------------
# standalone CLIs reuse the cached grid
# --------------------------------------------------------------------------

def _cached_stems(tmp_path, phase, detector="madmom"):
    """A synth stems folder with an alignment.json as stems2live leaves it."""
    import beat_aligner as ba
    d = tmp_path / "stems"
    d.mkdir()
    _make_alternating_track(str(d / "mix.wav"))
    paths = {"mix": str(d / "mix.wav")}
    grid = {"period": PERIOD, "grid_t0": 0.0, "downbeat_sec": phase * PERIOD,
            "bpm": 120.0}
    ba.write_cache(str(d), ba._cache_fingerprint(paths, None, detector, True), grid)
    return str(d), grid


def test_cli_grid_reuses_the_cached_grid_whatever_detector_wrote_it(tmp_path):
    import io
    from sections import cli_grid
    d, grid = _cached_stems(tmp_path, phase=2)
    out = io.StringIO()
    assert cli_grid(d, out=out) == (PERIOD, 0.0, 2 * PERIOD)
    assert "alignment.json" in out.getvalue()


def test_cli_grid_without_a_cache_says_bar_1_is_assumed(tmp_path):
    import io
    from sections import cli_grid
    out = io.StringIO()
    assert cli_grid(str(tmp_path), out=out) == (None, None, None)
    assert out.getvalue().count("bar 1 is assumed") == 1


def test_cli_grid_ignores_a_cache_for_different_audio(tmp_path):
    import io
    from sections import cli_grid
    d, _grid = _cached_stems(tmp_path, phase=2)
    with open(os.path.join(d, "mix.wav"), "ab") as fh:     # the stems changed
        fh.write(b"\0" * 64)
    out = io.StringIO()
    assert cli_grid(d, out=out) == (None, None, None)
    assert "bar 1 is assumed" in out.getvalue()


def test_cli_grid_explicit_values_win(tmp_path):
    import io
    from sections import cli_grid
    d, _grid = _cached_stems(tmp_path, phase=2)
    assert cli_grid(d, tempo=120.0, grid_t0=0.0, downbeat_sec=0.5,
                    out=io.StringIO()) == (0.5, 0.0, 0.5)
    assert cli_grid(d, downbeat_sec=1.5, out=io.StringIO())[2] == 1.5


@pytest.mark.parametrize("script", ["sections.py", "session_view.py"])
def test_standalone_cli_snaps_to_the_cached_downbeat(tmp_path, script):
    """`roadie sections` / `roadie scenes` with no grid flags."""
    import json
    import socket
    import subprocess
    d, _grid = _cached_stems(tmp_path, phase=3)
    argv = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(
        __file__)), script), d, "--json", "--segments", str(N_BLOCKS)]
    if script == "session_view.py":
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        argv += ["--dry-run", "--port", str(port)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, VIRTUAL_ENV="1"), check=False)
    assert r.returncode == 0, r.stderr[-800:]
    assert "alignment.json" in r.stderr
    res = json.loads(r.stdout)
    if script == "sections.py":
        starts = [sec["beat"] for sec in res["sections"]]
    else:
        starts = [sc["grid_beat"] for sc in res["scenes"]]
    assert starts and all(b % BEATS_PER_BAR == 3 for b in starts), starts


def test_short_track_is_rejected_not_guessed_at(tmp_path):
    """One phrase of audio has no structure to report; say so rather than invent."""
    sf.write(str(tmp_path / "mix.wav"),
             np.zeros(int(SYNTH_SR * 8.0), dtype=np.float32), SYNTH_SR)
    with pytest.raises(ValueError):
        detect_sections(str(tmp_path), phrase_bars=8, period=PERIOD, grid_t0=0.0)


def test_empty_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        detect_sections(str(tmp_path), period=PERIOD, grid_t0=0.0)


# --------------------------------------------------------------------------
# energy map
# --------------------------------------------------------------------------

def _secs_for_energy():
    return [{"beat": 0, "time_sec": 0.0, "label": "Intro"},
            {"beat": 32, "time_sec": 15.4, "label": "Drop"},
            {"beat": 64, "time_sec": 30.7, "label": "Outro"}]


def test_energy_notes_are_one_row_per_stem_descending_from_the_top():
    """Rows must read in the same order as the tracks, so the map is a legend.

    Pitch DESCENDS from top_pitch because a piano roll draws higher pitches
    higher: the first stem has to be the highest note to appear at the top.
    """
    from sections import energy_notes
    energy = {"kick": [1.0, 1.0, 0.0], "bass": [0.0, 1.0, 1.0]}
    notes, rows = energy_notes(energy, _secs_for_energy(),
                               order=["kick", "bass"], top_pitch=72)

    assert rows == ["kick", "bass"]
    assert {n["pitch"] for n in notes if n["pitch"] == 72}, "first stem at top_pitch"
    kick = [n for n in notes if n["pitch"] == 72]
    bass = [n for n in notes if n["pitch"] == 71]
    assert len(kick) == 2 and len(bass) == 2


def test_an_absent_stem_leaves_a_GAP_not_a_faint_note():
    """Below the floor there is no note at all.

    Writing velocity-1 notes for silent cells would make the map unreadable —
    the whole point is that a gap is visible as a gap.
    """
    from sections import energy_notes
    energy = {"vocals": [0.0, 0.05, 0.9]}
    notes, _ = energy_notes(energy, _secs_for_energy(), floor=0.10)
    assert len(notes) == 1
    assert notes[0]["start_time"] == 64.0


def test_note_spans_match_the_section_boundaries():
    from sections import energy_notes
    energy = {"kick": [1.0, 1.0, 1.0]}
    notes, _ = energy_notes(energy, _secs_for_energy(), tail_beats=64)
    spans = sorted((n["start_time"], n["duration"]) for n in notes)
    assert spans == [(0.0, 32.0), (32.0, 32.0), (64.0, 64.0)]


def test_velocity_carries_the_level_and_stays_playable():
    """Velocity is what Live colours notes by, so it is the readable channel.

    Clamped to 1..127: a 0 velocity note is not a quiet note, it is a note Live
    may drop entirely.
    """
    from sections import energy_notes
    energy = {"a": [1.0, 0.5, 0.11]}
    notes, _ = energy_notes(energy, _secs_for_energy(), floor=0.10)
    vels = [n["velocity"] for n in sorted(notes, key=lambda n: n["start_time"])]
    assert vels[0] == 127
    assert 60 <= vels[1] <= 68
    assert all(1 <= v <= 127 for v in vels)
    assert vels == sorted(vels, reverse=True), "louder section -> higher velocity"


def test_section_energy_normalises_each_stem_against_itself():
    """Arrangement, not balance: 'is this stem playing here' beats 'how loud
    is it in the mix'. Against the mix a shaker reads silent everywhere."""
    from sections import energy_notes
    loud = {"kick": [0.5, 1.0]}
    quiet = {"shaker": [0.5, 1.0]}
    secs = _secs_for_energy()[:2]
    n_loud, _ = energy_notes(loud, secs)
    n_quiet, _ = energy_notes(quiet, secs)
    assert [n["velocity"] for n in n_loud] == [n["velocity"] for n in n_quiet]


def test_the_macro_curve_is_one_note_per_section_above_the_stem_rows():
    """Two lanes in one clip: the mean across stems, and the per-stem matrix.

    Pitch 84 against 72-and-down leaves an octave of space, so the curve reads
    as its own lane rather than as a tenth stem.
    """
    from sections import energy_notes, macro_energy_notes
    energy = {"kick": [0.2, 1.0, 0.4], "bass": [0.0, 1.0, 0.6]}
    secs = _secs_for_energy()

    macro = macro_energy_notes(energy, secs)
    rows, _ = energy_notes(energy, secs)

    assert len(macro) == len(secs), "one note per section, gaps included"
    assert {n["pitch"] for n in macro} == {84}
    assert max(n["pitch"] for n in rows) < 84 - 6, "clear of the stem rows"
    # mean of 0.2 and 0.0 -> quiet; mean of two 1.0s -> full
    assert macro[0]["velocity"] < macro[1]["velocity"]
    assert macro[1]["velocity"] == 127


def test_the_macro_curve_keeps_a_note_where_every_stem_is_silent():
    """Unlike the matrix, the curve has no floor.

    A silent section is part of the shape — a break reading 1 is the trough
    you want to see. Dropping it would leave a hole in the line.
    """
    from sections import macro_energy_notes
    macro = macro_energy_notes({"kick": [0.0, 1.0, 0.0]}, _secs_for_energy())
    assert len(macro) == 3
    assert all(n["velocity"] >= 1 for n in macro)


def test_the_macro_curve_averages_rather_than_following_the_loudest_stem():
    """'How hard is the track working' — many elements near their own peak,
    not one loud stem carrying it."""
    from sections import macro_energy_notes
    one_loud = {"a": [1.0, 1.0], "b": [0.0, 0.0], "c": [0.0, 0.0]}
    all_mid = {"a": [0.4, 0.4], "b": [0.4, 0.4], "c": [0.4, 0.4]}
    secs = _secs_for_energy()[:2]
    assert (macro_energy_notes(all_mid, secs)[0]["velocity"]
            > macro_energy_notes(one_loud, secs)[0]["velocity"])


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------

def test_transition_notes_separate_the_two_gestures_by_pitch():
    """Fill and dropout are different moves; velocity alone would conflate them."""
    from sections import transition_notes
    n = transition_notes([{"bar": 32, "beat": 124.0, "kind": "dropout",
                           "ratio": 14.8, "into": "Main 1"},
                          {"bar": 80, "beat": 316.0, "kind": "fill",
                           "ratio": 6.0, "into": "Drop 2"}])
    assert [x["pitch"] for x in n] == [86, 88]
    assert all(x["duration"] == 4 for x in n)


def test_transition_velocity_is_capped_so_one_extreme_does_not_flatten_the_rest():
    """A 14.8x dropout and a 4x dropout should not both read as 'the maximum'
    while a 2x one is invisible — the cap is what keeps the middle legible."""
    from sections import transition_notes
    vels = [x["velocity"] for x in transition_notes(
        [{"bar": 1, "beat": 0.0, "kind": "fill", "ratio": r, "into": "X"}
         for r in (1.5, 2.0, 4.0, 14.8)])]
    assert vels == sorted(vels)
    assert vels[-1] == vels[-2] == 127, "ratios past the cap share the ceiling"
    assert vels[0] < 60, "a marginal gesture stays visibly marginal"


def test_a_transition_never_starts_before_the_arrangement():
    """A boundary in the first bar would put the marker at a negative beat,
    which Live rejects for the whole clip rather than for that note."""
    from sections import transition_notes
    n = transition_notes([{"bar": 1, "beat": -4.0, "kind": "fill",
                           "ratio": 2.0, "into": "Intro"}])
    assert n[0]["start_time"] >= 0.0

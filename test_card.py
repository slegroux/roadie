"""Tests for card.py, the track card's parts the study report uses (from PR #6; the tests of its CLI, its arrangement
table and its own section analysis went with them, the report has its own).

The layout is PR #6's synthetic one (a 2 s bar at 120 BPM): kick bars 0-47 and 80-111, hats 0-47 and 64-127, bass 16-47
and 64-111, synths 16-127. With the first bar at Live bar 9, bar index i is Live bar 9 + i.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from card import despeckled, element_map, longest_static_stretch, observations, runs, transition_lines


def _bits(n, *spans):
    return "".join("1" if any(a <= i <= b for a, b in spans) else "0" for i in range(n))


PRESENCE = {"drums_kick": _bits(128, (0, 47), (80, 111)), "drums_hh": _bits(128, (0, 47), (64, 127)),
            "bass": _bits(128, (16, 47), (64, 111)), "synths": _bits(128, (16, 127))}
STARTS = [0, 16, 48, 64, 80, 112]
LABELS = ["Intro", "Drop 1", "Breakdown", "Build", "Drop 2", "Outro"]


def _sections(**changes):
    bars = [16, 32, 16, 16, 32, 16]
    rel = [1.0, 2.0, 0.5, 1.0, 2.0, 1.0]
    out = [{"label": l, "start": s, "bars": b, "stems": [], "rel_energy": r} for l, s, b, r in zip(LABELS, STARTS, bars, rel)]
    for i, kw in changes.items():
        out[int(i[1:])].update(kw)
    return out


def test_runs_lists_inclusive_spans():
    assert runs("0011100011") == [(2, 4), (8, 9)]
    assert runs("") == [] and runs("000") == []


def test_element_map_rows_follow_presence():
    lines = element_map(PRESENCE, STARTS, LABELS, 9)
    kick = next(l for l in lines if l.startswith("kick"))
    assert kick.split()[1] == "█" * 12 + "·" * 8 + "█" * 8 + "·" * 4
    assert lines[0].split() == ["bar", "9", "41", "73", "105"]
    assert "|Drop 1" in lines[1]
    assert [l.split()[0] for l in lines[2:]] == ["kick", "bass", "hh", "synths"]        # low to high


def test_a_flag_marks_the_ruler():
    assert element_map(PRESENCE, STARTS, LABELS, 9, flag="†")[0].split() == ["bar", "9†", "41†", "73†", "105†"]


def test_a_partly_filled_cell_is_shaded():
    lines = element_map({"bass": "11000000"}, [0], ["Intro"], 1)
    assert lines[-1].split()[1] == "▒·"


def test_a_ruler_step_the_columns_cannot_hit_is_refused():
    with pytest.raises(ValueError):
        element_map({"bass": "1" * 64}, [0], ["Intro"], 1, bars_per_char=3)


def test_longest_static_stretch_takes_the_first_of_equal_runs():
    assert longest_static_stretch(PRESENCE) == (16, 47)
    assert "bars 25–56 (32 bars)" in observations(_sections(), PRESENCE, lambda i: 9 + i)[0]


def test_transitions_are_reported_in_bars():
    tr = [{"bar": 80, "beat": 316.0, "kind": "dropout", "ratio": 3.0, "into": "Drop 2"}]
    assert transition_lines(tr, lambda b: b + 8)[0].startswith("- Bar 88, the bar before **Drop 2**: a dropout: 3.0× fewer")
    assert transition_lines([], lambda b: b) == ["No fill or dropout in the bar before any section change."]


def test_a_short_middle_section_is_a_mini_break_but_a_short_intro_is_not():
    secs = _sections(s0={"bars": 4})
    secs.insert(3, dict(secs[2], label="Breakdown", bars=4, stems=["synths"]))
    notes = observations(secs, None, lambda i: 9 + i)
    assert sum(n.startswith("Mini-break at bar") for n in notes) == 1 and not any("bar 9:" in n for n in notes)


def test_breakdown_depth_is_deepest_breakdown_against_loudest_drop():
    secs = _sections(s2={"label": "Breakdown 1", "rel_energy": 0.5}, s3={"label": "Breakdown 2", "rel_energy": 0.25},
                     s4={"rel_energy": 1.0})
    # loudest drop +6.02 dB, deepest breakdown -12.04 dB
    assert "The deepest breakdown sits 18.1 dB under the loudest drop." in observations(secs, None, lambda i: i)


def test_breakdown_depth_reads_the_spectral_names_too():
    secs = [dict(s, label=l, rel_energy=e) for s, (l, e) in zip(_sections(), (
        ("Intro", 1.0), ("Drop", 1.0), ("Quiet", 0.5), ("Break", 1.0), ("Main", 1.0), ("Outro", 1.0)))]
    assert any("sits 6.0 dB under the loudest drop" in n for n in observations(secs, None, lambda i: i))


def test_sections_off_the_8_bar_grid_are_named():
    notes = observations(_sections(s1={"bars": 30}), None, lambda i: i)
    assert "Sections whose length is not a multiple of 8 bars: Drop 1 (30)." in notes


def test_a_one_bar_gap_does_not_cut_the_longest_stretch_once_despeckled():
    presence = {"bass": "1" * 16 + "0" * 8, "vocals": "1" * 7 + "0" + "1" * 8 + "0" * 8}
    assert longest_static_stretch(presence) == (8, 15)                     # bar by bar: the gap at 7 cuts it
    assert despeckled(presence)["vocals"] == "1" * 16 + "0" * 8
    assert longest_static_stretch(despeckled(presence)) == (0, 15)


def test_a_fill_after_a_drumless_section_is_said_in_hits_and_big_ratios_are_capped():
    tr = [{"bar": 44, "kind": "fill", "ratio": 40.0, "median_hits": 0.0, "last_hits": 20, "into": "Drop 2"},
          {"bar": 60, "kind": "fill", "ratio": 6.0, "median_hits": 2.0, "last_hits": 12, "into": "Drop 3"},
          {"bar": 70, "kind": "dropout", "ratio": 3.6, "median_hits": 18.0, "last_hits": 5, "into": "Outro"}]
    lines = transition_lines(tr, lambda b: b)
    assert lines[0] == "- Bar 44, the bar before **Drop 2**: drums come in (20 hits; the section had none in its usual bar)."
    assert "a fill: ≥ 4× the drum hits" in lines[1] and "a dropout: 3.6× fewer" in lines[2]

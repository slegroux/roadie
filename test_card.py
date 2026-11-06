"""Tests for card.py, the track card.

The synthetic stems from test_sections (120 BPM, a 2 s bar) have a known layout:
kick bars 0-47 and 80-111 (48-55 muffled, which counts as no kick), hats 0-47 and
64-127, bass 16-47 and 64-111, synths 16-127. With the first bar at Live bar 9,
bar index i is Live bar 9 + i.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import card
from card import element_map, longest_static_stretch, render_card, runs, track_name
from test_sections import _by_stems, _layout, _write_stems


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    return _by_stems(_write_stems(tmp_path_factory.mktemp("card") / "s", _layout(), 128))


def test_runs_lists_inclusive_spans():
    assert runs("0011100011") == [(2, 4), (8, 9)]
    assert runs("") == [] and runs("000") == []


def test_track_name_drops_the_tempo_and_stems_suffix():
    assert track_name("/x/Off_Gerlach_125bpm_stems/") == "Off Gerlach"
    assert track_name("a_b_122.5bpm_stems") == "a b"


def test_presence_marks_each_stem_bar_by_bar(res):
    p = res["presence"]
    assert runs(p["drums_kick"]) == [(0, 47), (80, 111)]       # the muffled 48-55 is out
    assert runs(p["bass"]) == [(16, 47), (64, 111)]
    assert set(len(b) for b in p.values()) == {128}


def test_entries_and_exits_are_in_live_bars(res):
    text = render_card("t", res, 9)
    assert "- **kick**: 9–56, 89–120 (80 bars)" in text
    assert "- **synths**: 25–136 (112 bars)" in text
    assert "First entries, in order: kick, hh at 9; bass, synths at 25." in text


def test_arrangement_rows_name_every_change(res):
    text = render_card("t", res, 9)
    assert "| 57 | Breakdown | 16 | synths | kick out, bass out, hats out |" in text
    assert "| 89 | Drop 2 | 32 | kick, bass, hats, synths | kick in |" in text


def test_element_map_rows_follow_presence(res):
    lines = element_map(res["presence"], [0, 16, 48, 64, 80, 112],
                        ["Intro", "Drop 1", "Breakdown", "Build", "Drop 2", "Outro"], 9)
    kick = next(l for l in lines if l.startswith("kick"))
    assert kick.split()[1] == "█" * 12 + "·" * 8 + "█" * 8 + "·" * 4
    assert lines[0].split() == ["bar", "9", "41", "73", "105"]
    assert "|Drop 1" in lines[1]


def test_a_partly_filled_cell_is_shaded():
    lines = element_map({"bass": "11000000"}, [0], ["Intro"], 1)
    assert lines[-1].split()[1] == "▒·"


def test_longest_static_stretch_takes_the_first_of_equal_runs(res):
    assert longest_static_stretch(res["presence"]) == (16, 47)
    assert "bars 25–56 (32 bars)" in render_card("t", res, 9)


def test_transitions_are_reported_in_bars(res):
    tr = [{"bar": 80, "beat": 316.0, "kind": "dropout", "ratio": 3.0, "into": "Drop 2"}]
    text = render_card("t", res, 9, transitions=tr)
    assert "- Bar 88, the bar before **Drop 2**: a dropout: 3.0× fewer" in text
    assert "## Into each section" not in render_card("t", res, 9, transitions=None)
    assert "No fill or dropout" in render_card("t", res, 9, transitions=[])


def test_a_short_middle_section_is_a_mini_break_but_a_short_intro_is_not(res):
    secs = [dict(s) for s in res["sections"]]
    secs[0]["duration_beats"] = 16                   # a 4-bar intro
    secs.insert(3, dict(secs[2], label="Breakdown", duration_beats=16, stems=["synths"]))
    text = render_card("t", dict(res, sections=secs), 9)
    assert "Mini-break at bar 9" not in text
    assert text.count("Mini-break at bar") == 1


def test_without_presence_the_card_says_why_there_is_no_map(res):
    spectral = {k: v for k, v in res.items() if k != "presence"}
    spectral["sections"] = [{k: v for k, v in s.items() if k != "stems"}
                            for s in res["sections"]]
    text = render_card("t", spectral, 1, bar_kind="file")
    assert "No element map" in text and "## Entries and exits" not in text
    assert "| 1 | Intro | 16 | — | — |" in text
    assert "not Live bars" in text


def test_main_writes_card_md_and_honours_out(tmp_path, monkeypatch, capsys):
    d = _write_stems(tmp_path / "x_120bpm_stems", _layout(), 128)
    monkeypatch.setattr(card, "cli_grid", lambda *a, **k: (0.5, 0.0, 0.0))
    monkeypatch.setattr(sys, "argv", ["card.py", d, "--no-transitions"])
    card.main()
    printed = capsys.readouterr().out
    with open(os.path.join(d, "card.md")) as f:
        assert f.read() == printed
    assert printed.startswith("# x\n") and "| 9 | Intro |" in printed

    other = tmp_path / "note.md"
    monkeypatch.setattr(sys, "argv", ["card.py", d, "--no-transitions", "-o", str(other)])
    card.main()
    assert other.read_text() == capsys.readouterr().out


def test_no_write_saves_nothing(tmp_path, monkeypatch):
    d = _write_stems(tmp_path / "s", _layout(), 128)
    monkeypatch.setattr(card, "cli_grid", lambda *a, **k: (0.5, 0.0, 0.0))
    monkeypatch.setattr(sys, "argv", ["card.py", d, "--no-transitions", "--no-write"])
    card.main()
    assert not os.path.exists(os.path.join(d, "card.md"))


def test_a_drifting_track_gets_a_warning_and_a_steady_one_does_not(res):
    assert "The tempo drifts 2.1% across this track." in render_card("t", res, 9, drift_pct=2.146)
    assert "drifts" not in render_card("t", res, 9)


def _levels(res, *pairs):
    """`res` with its sections relabelled and given these (label, rel_energy) pairs."""
    secs = [dict(s, label=l, rel_energy=e) for s, (l, e) in zip(res["sections"], pairs)]
    return dict(res, sections=secs)


def test_breakdown_depth_is_deepest_breakdown_against_loudest_drop(res):
    r = _levels(res, ("Intro", 1.0), ("Drop 1", 2.0), ("Breakdown 1", 0.5),
                ("Breakdown 2", 0.25), ("Drop 2", 1.0), ("Outro", 1.0))
    # loudest drop +6.02 dB, deepest breakdown -12.04 dB
    assert "The deepest breakdown sits 18.1 dB under the loudest drop." in render_card("t", r, 9)


def test_breakdown_depth_reads_the_spectral_names_too(res):
    r = _levels(res, ("Intro", 1.0), ("Drop", 1.0), ("Quiet", 0.5),
                ("Break", 1.0), ("Main", 1.0), ("Outro", 1.0))
    assert "sits 6.0 dB under the loudest drop" in render_card("t", r, 9)


def test_a_real_spectral_run_renders_without_a_map(tmp_path):
    from sections import analyze_sections
    d = _write_stems(tmp_path / "s", _layout(), 128)
    r = analyze_sections(d, period=0.5, grid_t0=0.0, downbeat_sec=0.0, method="spectral")
    assert r.get("method") != "stems" and "presence" not in r
    text = render_card("t", r, 1, bar_kind="file")
    assert "No element map" in text
    assert text.count("\n| ") == len(r["sections"]) + 1        # header + one row each


def test_mmss_carries_rounded_seconds_into_the_minute():
    from card import _mmss
    assert _mmss(299.6) == "5:00" and _mmss(59.4) == "0:59" and _mmss(61) == "1:01"


def test_a_ruler_step_the_columns_cannot_hit_is_refused():
    with pytest.raises(ValueError):
        element_map({"bass": "1" * 64}, [0], ["Intro"], 1, bars_per_char=3)


def _run_main(monkeypatch, d, *extra, cached=None):
    monkeypatch.setattr(card, "cli_grid", lambda *a, **k: (0.5, 0.0, 0.0))
    monkeypatch.setattr(card, "cached_grid", lambda *a, **k: cached)
    monkeypatch.setattr(sys, "argv", ["card.py", d, "--no-transitions"] + list(extra))
    card.main()


def test_main_reads_drift_from_the_cached_grid(tmp_path, monkeypatch, capsys):
    d = _write_stems(tmp_path / "s", _layout(), 128)
    _run_main(monkeypatch, d, "--no-write", cached={"has_drift": True, "drift_pct": 2.1464})
    assert "The tempo drifts 2.1% across this track." in capsys.readouterr().out
    _run_main(monkeypatch, d, "--no-write", cached={"has_drift": False, "drift_pct": 0.4})
    assert "drifts" not in capsys.readouterr().out


def test_out_to_a_folder_names_the_file_after_the_track(tmp_path, monkeypatch, capsys):
    d = _write_stems(tmp_path / "Off_Gerlach_125bpm_stems", _layout(), 128)
    vault = tmp_path / "vault"
    vault.mkdir()
    _run_main(monkeypatch, d, "-o", str(vault))
    assert (vault / "Off Gerlach.md").read_text(encoding="utf-8") == capsys.readouterr().out


def test_no_write_wins_over_out(tmp_path, monkeypatch):
    d = _write_stems(tmp_path / "s", _layout(), 128)
    note = tmp_path / "note.md"
    _run_main(monkeypatch, d, "--no-write", "-o", str(note))
    assert not note.exists() and not os.path.exists(os.path.join(d, "card.md"))

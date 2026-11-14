"""The reference-profile half of the analysis (roadie_analysis.profile) on synthetic signals whose answers
are known without the code: reference_profile, and the element and drum layers of a mix_report made
against a profile.

Copied from Sideman's tests/test_listen.py (main bc9ba08): the tests that build or read a profile, with
`listen` renamed `profile` and nothing else changed except what is named below. The tests of the
measurement core and of a report without a profile (and its golden) stay in Sideman, whose `listen` this
builds on. The assertions that go through Sideman's MCP server (`mcp_server.server`: the tool's summary,
its LiveError messages) are left out of the two tests that carried them, which call the library directly
(test_profile_mix_report_takes_the_path_or_the_json, and the end of
test_profile_loading_refuses_a_wrong_shape_or_version_and_says_to_rebuild) and expect the library's own
errors: the tool wrapper stays in Sideman until its item 5, and Roadie's own server (item 3) gets those
assertions back.
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from roadie_analysis import profile  # noqa: E402

SR = 48000


def sine(freq, seconds, amp=1.0, phase=0.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def stereo(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


def write(tmp_path, name, x, sr=SR):
    p = tmp_path / name
    sf.write(str(p), x, sr, subtype="FLOAT")
    return p


TEMPO = 120.0          # 1 bar = 2 s


def bars(n):
    return int(n * 2 * SR)


T0 = 0.5


KICK_AMP = 0.4


DRUM_FILE_DB = -16.0      # each of hh and snare, in dB against the kick's amplitude


DRUMS_VS_KICK = DRUM_FILE_DB + 10 * math.log10(2)    # two equal files: twice the power of one


ROLE_HZ = {"kick": 50, "bass": 110, "synth": 1500, "hh": 8000, "snare": 3000}


def gain(db):
    return 10 ** (db / 20)


def level_sine(freq, db_by_bar):
    """A sine at KICK_AMP * gain(db) per bar (None = silent), T0 of silence in front."""
    amp = np.repeat([0.0 if d is None else KICK_AMP * gain(d) for d in db_by_bar], bars(1))
    t = np.arange(len(amp)) / SR
    return np.concatenate([np.zeros(int(T0 * SR)), amp * np.sin(2 * np.pi * freq * t)])


def stem_set(tmp_path, folder, bass_db=-6.0, bars_n=24, kick_db=0.0, drum_db=DRUM_FILE_DB):
    """kick, bass (`bass_db` against the kick, a number or one per bar), synth (12 dB under the
    kick, silent in bars 8-15), hh + snare (DRUMS_VS_KICK together) and their mix."""
    d = tmp_path / folder
    d.mkdir()
    per_bar = {"kick": list(kick_db) if isinstance(kick_db, (list, tuple)) else [kick_db] * bars_n,
               "bass": list(bass_db) if isinstance(bass_db, (list, tuple)) else [bass_db] * bars_n,
               "synth": [None if 8 <= b < 16 else -12.0 for b in range(bars_n)],
               "hh": [drum_db] * bars_n, "snare": [drum_db] * bars_n}
    sig = {k: level_sine(ROLE_HZ[k], v) for k, v in per_bar.items()}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    paths["drums"] = write(d, "drums.wav", stereo(sig["hh"] + sig["snare"]))     # for a capture: one file
    paths["mix"] = write(d, "mix.wav", stereo(sum(sig.values())))
    return paths


SECTIONS = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8},
            {"name": "C", "start_bar": 16, "bars": 8}]


def profile_of(tmp_path, ref, **kw):
    return profile.reference_profile(
        ref["mix"], {"kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"],
                     "drums": [ref["hh"], ref["snare"]]},
        TEMPO, T0, SECTIONS, out=tmp_path / "ref.profile.json", **kw)


def test_profile_ratios_to_the_kick_match_the_gains_in_every_section(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    assert [s["name"] for s in p["sections"]] == ["A", "B", "C"]
    for s in p["sections"]:
        r = s["roles"]
        assert r["kick"]["vs_kick_db"] == 0.0
        assert abs(r["bass"]["vs_kick_db"] - (-6.0)) <= 0.1, s["name"]
        assert abs(r["drums"]["vs_kick_db"] - DRUMS_VS_KICK) <= 0.1, s["name"]
        assert abs(r["kick"]["level_db"] - 20 * math.log10(KICK_AMP / math.sqrt(2))) <= 0.05
    for name in ("A", "C"):
        assert abs(p["sections"][0 if name == "A" else 2]["roles"]["synth"]["vs_kick_db"] - (-12.0)) <= 0.1


def test_profile_a_silent_role_is_absent_not_minus_120(tmp_path):
    p = profile_of(tmp_path, stem_set(tmp_path, "ref"))
    b = p["sections"][1]["roles"]["synth"]
    assert b["absent"] and set(b) == {"absent", "per_bar_db", "core_bar_db"} and b["per_bar_db"] == [None] * 8   # no level, bars kept
    assert b["core_bar_db"] == [None] * 8
    assert all(not s["roles"][r]["absent"] for s in p["sections"] for r in ("kick", "bass", "drums"))
    assert not p["sections"][0]["roles"]["synth"]["absent"]


def test_profile_per_bar_curve_follows_t0_and_the_bar_grid(tmp_path):
    ref = stem_set(tmp_path, "ref", bass_db=[-6.0] * 4 + [-2.0] * 4 + [-6.0] * 16)
    p = profile_of(tmp_path, ref)
    curve = p["sections"][0]["roles"]["bass"]["per_bar_db"]
    want = [20 * math.log10(KICK_AMP * gain(d) / math.sqrt(2)) for d in [-6.0] * 4 + [-2.0] * 4]
    assert len(curve) == 8 and all(abs(c - w) <= 0.05 for c, w in zip(curve, want)), (curve, want)


def test_profile_bands_width_and_mix_are_measured(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    s = p["sections"][0]
    assert s["roles"]["bass"]["bands_db"]["bass"] > s["roles"]["bass"]["bands_db"]["mid"] + 40   # 110 Hz is in 60-250
    assert s["roles"]["synth"]["bands_db"]["mid"] > s["roles"]["synth"]["bands_db"]["bass"] + 40   # 1.5 kHz is in 500-2k
    assert s["roles"]["kick"]["width"] == 0.0 and s["mix"]["width"] == 0.0
    want = 10 * math.log10(sum((KICK_AMP * gain(d)) ** 2 / 2 for d in (0, -6, -12, DRUM_FILE_DB, DRUM_FILE_DB)))
    assert abs(s["mix"]["level_db"] - want) <= 0.1
    assert s["mix"]["integrated_lufs"] is not None and len(s["mix"]["per_bar_db"]) == 8
    wide = write(tmp_path, "wide.wav", np.stack([sine(1000, 4, 0.3), -sine(1000, 4, 0.3)], axis=1))
    q = profile.reference_profile(wide, {"kick": wide}, TEMPO, 0.0, [{"name": "W", "start_bar": 0, "bars": 2}])
    assert abs(q["sections"][0]["roles"]["kick"]["width"] - 1.0) <= 0.01      # opposite polarity is all side


def test_profile_is_written_with_its_inputs_and_goes_stale_when_one_changes(tmp_path):
    import json
    import os
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    dest = pathlib.Path(p["json_path"])
    assert dest == (tmp_path / "ref.profile.json").resolve()
    on_disk = json.loads(dest.read_text())
    assert on_disk["tempo"] == TEMPO and on_disk["t0"] == T0 and on_disk["sections"][1]["name"] == "B"
    assert [i["path"] for i in on_disk["inputs"]["stems"]["drums"]] == [str(ref["hh"].resolve()), str(ref["snare"].resolve())]
    assert on_disk["inputs"]["mix"][0]["mtime"] == ref["mix"].stat().st_mtime
    assert profile.profile_stale(p) == [] and profile.profile_stale(on_disk) == []
    st = ref["bass"].stat()
    os.utime(ref["bass"], (st.st_atime, st.st_mtime + 5))
    assert [(s["role"], s["reason"]) for s in profile.profile_stale(on_disk)] == [("bass", "changed")]
    ref["snare"].unlink()
    assert {(s["role"], s["reason"]) for s in profile.profile_stale(on_disk)} == {("bass", "changed"), ("drums", "missing")}


def test_profile_default_path_is_next_to_the_mix(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = profile.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, SECTIONS[:1])
    assert pathlib.Path(p["json_path"]) == ref["mix"].resolve().with_name("mix.profile.json")
    assert pathlib.Path(p["json_path"]).is_file()


def test_profile_refuses_mixed_sample_rates_and_trims_different_lengths(tmp_path):
    ref = stem_set(tmp_path, "ref")
    odd = write(tmp_path, "odd.wav", stereo(sine(110, 2, 0.1, sr=44100, ), None), sr=44100)
    with pytest.raises(ValueError, match="sample rate"):
        profile.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": odd}, TEMPO, T0, SECTIONS[:1])
    short = write(tmp_path, "short.wav", stereo(level_sine(50, [0.0] * 10)))     # 10 bars + T0, the others 24
    p = profile.reference_profile(ref["mix"], {"kick": short, "bass": ref["bass"]}, TEMPO, T0,
                                 [{"name": "A", "start_bar": 0, "bars": 8}])
    assert any("trimmed" in n and "short.wav" in n for n in p["notes"]), p["notes"]
    with pytest.raises(ValueError, match="sections run to"):
        profile.reference_profile(ref["mix"], {"kick": short}, TEMPO, T0, SECTIONS)


def test_profile_refuses_bad_input(tmp_path):
    ref = stem_set(tmp_path, "ref")
    with pytest.raises(ValueError, match="unique"):
        profile.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, [SECTIONS[0], SECTIONS[0]])
    with pytest.raises(ValueError, match="no such audio file"):
        profile.reference_profile(ref["mix"], {"kick": tmp_path / "nope.wav"}, TEMPO, T0, SECTIONS[:1])
    with pytest.raises(ValueError, match="role"):
        profile.reference_profile(ref["mix"], {"mix": ref["kick"]}, TEMPO, T0, SECTIONS[:1])
    with pytest.raises(ValueError, match="tempo"):
        profile.reference_profile(ref["mix"], {"kick": ref["kick"]}, 0, T0, SECTIONS[:1])


def capture_of(ref, **over):
    """A manifest whose sources are the files of a stem_set (drums as the one summed file)."""
    src = {"master": ref["mix"], "kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"],
           "drums": ref["drums"]}
    src.update(over)
    return {"tempo": TEMPO, "beats_per_bar": 4, "offset_s": T0,
            "sources": {k: str(v) for k, v in src.items() if v is not None},
            "sections": [dict(s) for s in SECTIONS]}


def test_elements_bass_raised_4_db_is_the_only_gap(tmp_path):
    prof = profile_of(tmp_path, stem_set(tmp_path, "ref"))
    ours = stem_set(tmp_path, "ours", bass_db=-6.0 + 4.0)
    r = profile.mix_report(capture_of(ours), reference=prof, out_dir=tmp_path / "out")
    assert r["reference_kind"] == "profile" and r["reference_stale"] == []
    for s in r["sections"]:
        e = s["elements"]
        assert e["paired"] and not e["capture_is_reference"], s["name"]
        assert [g["role"] for g in e["gaps"]] == ["bass"], (s["name"], e["gaps"])
        assert abs(e["gaps"][0]["difference_db"] - 4.0) <= 0.2
        row = {t["role"]: t for t in e["table"]}
        assert abs(row["bass"]["reference_db"] - (-6.0)) <= 0.1 and abs(row["bass"]["ours_db"] - (-2.0)) <= 0.1
        assert abs(row["drums"]["difference_db"]) <= 0.1
    b = r["sections"][1]["elements"]
    assert "synth" not in {t["role"] for t in b["table"]} and b["silent_in_both"] == ["synth"]
    assert "master" not in {t["role"] for s in r["sections"] for t in s["elements"]["table"]}
    md = pathlib.Path(r["md_path"]).read_text()
    assert "## Elements against the kick" in md and "bass +4.0 dB" in md
    assert "bass" in pathlib.Path(r["json_path"]).read_text()


def test_elements_the_capture_that_is_the_reference_is_flagged(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    r = profile.mix_report(capture_of(ref), reference=prof["json_path"], out_dir=tmp_path / "out")   # a profile path
    for s in r["sections"]:
        e = s["elements"]
        assert e["capture_is_reference"] and "the capture is the reference: check solos" in e["notes"]
        assert e["gaps"] == []
        assert s["vs_reference"]["flagged_bands"] == []
    assert "the capture is the reference: check solos" in pathlib.Path(r["md_path"]).read_text()
    ours = stem_set(tmp_path, "ours", bass_db=-6.0 + 4.0)           # a different master: no flag
    assert not any(s["elements"]["capture_is_reference"] for s in profile.mix_report(capture_of(ours), reference=prof)["sections"])


def test_elements_without_a_kick_source_leaves_the_table_out_but_keeps_the_mix_layer(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    r = profile.mix_report(capture_of(ref, kick=None), reference=prof, out_dir=tmp_path / "out")
    for s in r["sections"]:
        e = s["elements"]
        assert e["table"] is None and e["gaps"] == [] and e["arc"] == {}
        assert any("no kick source" in n for n in e["notes"]), e["notes"]
        assert s["integrated_lufs"] is not None and "bands_db" in s and "vs_reference" in s
    assert "no kick source" in pathlib.Path(r["md_path"]).read_text()


def test_elements_roles_in_one_side_only_and_sections_that_do_not_pair(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    ours = stem_set(tmp_path, "ours")
    m = capture_of(ours, drums=None, vox=ours["hh"])
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "Z", "start_bar": 8, "bars": 8},
                     {"name": "C", "start_bar": 16, "bars": 4}]
    r = profile.mix_report(m, reference=prof)
    a, z, c = (s["elements"] for s in r["sections"])
    assert [x["role"] for x in a["reference_only"]] == ["drums"] and a["reference_only"][0]["reason"] == "not in the manifest"
    assert abs(a["reference_only"][0]["vs_yardstick_db"] - DRUMS_VS_KICK) <= 0.1
    assert [x["role"] for x in a["ours_only"]] == ["vox"] and a["ours_only"][0]["reason"] == "not in the profile"
    assert z == {"paired": False, "notes": ["no section named 'Z' in the profile (it has: A, B, C): mix layer only"]}
    assert c["paired"] and c["paired_with"] == {"section": "C", "from_bar": 0, "bars": 4}
    assert "paired with bars 1-4 of the profile's 'C' (8 bars)" in c["notes"][0]
    assert "vs_reference" not in r["sections"][1] and r["sections"][1]["integrated_lufs"] is not None


def test_elements_arc_follows_the_ratio_per_four_bars(tmp_path):
    one = [{"name": "A", "start_bar": 0, "bars": 8}]
    ref = stem_set(tmp_path, "ref", bass_db=-6.0)
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": ref["bass"]}, TEMPO, T0, one)
    ours = stem_set(tmp_path, "ours", bass_db=[-6.0] * 4 + [-2.0] * 4 + [-6.0] * 16)
    m = capture_of(ours)
    m["sections"] = one
    arc = profile.mix_report(m, reference=prof)["sections"][0]["elements"]["arc"]["bass"]
    assert len(arc["reference"]) == len(arc["ours"]) == 2, arc
    assert [round(v) for v in arc["reference"]] == [-6, -6] and [round(v) for v in arc["ours"]] == [-6, -2], arc
    assert all(abs(v - w) <= 0.2 for v, w in zip(arc["reference"], [-6.0, -6.0])), arc
    assert all(abs(v - w) <= 0.2 for v, w in zip(arc["ours"], [-6.0, -2.0])), arc


def test_elements_a_stale_profile_is_reported_and_a_plain_audio_reference_is_unchanged(tmp_path):
    import os
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    st = ref["kick"].stat()
    os.utime(ref["kick"], (st.st_atime, st.st_mtime + 5))
    r = profile.mix_report(capture_of(stem_set(tmp_path, "ours")), reference=prof, out_dir=tmp_path / "o1")
    assert [s["role"] for s in r["reference_stale"]] == ["kick"]
    assert "changed since the profile was built" in pathlib.Path(r["md_path"]).read_text()
    r = profile.mix_report(capture_of(ref), reference=str(ref["mix"]), out_dir=tmp_path / "o2")
    assert "reference_kind" not in r and r["reference"] == str(ref["mix"])
    for s in r["sections"]:
        assert "elements" not in s and set(s["vs_reference"]) == {"band_balance_db", "flagged_bands", "lufs_diff"}
    assert "Elements against the kick" not in pathlib.Path(r["md_path"]).read_text()
    with pytest.raises(ValueError, match="not a reference profile"):
        profile.mix_report(capture_of(ref), reference={"sections": []})


def test_profile_mix_report_takes_the_path_or_the_json(tmp_path):
    import json

    ref = stem_set(tmp_path, "ref")
    out = tmp_path / "tool.profile.json"
    prof = profile.reference_profile(
        str(ref["mix"]), {"kick": str(ref["kick"]), "bass": str(ref["bass"]), "synth": str(ref["synth"]),
                          "drums": [str(ref["hh"]), str(ref["snare"])]},
        TEMPO, T0, SECTIONS, out=str(out))
    assert out.is_file() and prof["json_path"] == str(out.resolve())
    for reference in (str(out), out.read_text(), json.loads(out.read_text())):      # path, JSON text, dict
        r = profile.mix_report(capture_of(ref), reference=reference, out_dir=str(tmp_path / "out"))
        assert r["reference_kind"] == "profile" and r["sections"][0]["elements"]["capture_is_reference"]
    with pytest.raises(ValueError, match="section names must be unique"):
        profile.reference_profile(str(ref["mix"]), {"kick": str(ref["kick"])}, TEMPO, T0, SECTIONS + SECTIONS[:1])
    with pytest.raises(ValueError, match="no such profile"):
        profile.mix_report(capture_of(ref), reference=str(tmp_path / "nope.json"))


def test_elements_a_shorter_section_pairs_with_the_first_bars_and_from_bar_picks_others(tmp_path):
    ref = stem_set(tmp_path, "ref", bass_db=[-6.0] * 4 + [-2.0] * 4 + [-6.0] * 16)
    prof = profile_of(tmp_path, ref)
    ours = stem_set(tmp_path, "ours", bass_db=-2.0)
    m = capture_of(ours)

    def run(**ref_key):
        m["sections"] = [{"name": "X", "start_bar": 0, "bars": 4, **ref_key}]
        return profile.mix_report(m, reference=prof)["sections"][0]["elements"]

    first = run(reference={"section": "A"})            # A's bars 1-4: bass -6 against our -2
    assert first["paired_with"] == {"section": "A", "from_bar": 0, "bars": 4}
    assert [g["role"] for g in first["gaps"]] == ["bass"] and abs(first["gaps"][0]["difference_db"] - 4.0) <= 0.2
    assert "paired with bars 1-4 of the profile's 'A' (8 bars)" in first["notes"][0]
    assert "whole section" in first["notes"][0] and not first["capture_is_reference"]
    later = run(reference={"section": "A", "from_bar": 4})   # A's bars 5-8: bass -2, as ours
    assert later["paired_with"]["from_bar"] == 4 and later["gaps"] == []
    row = {t["role"]: t for t in later["table"]}
    assert abs(row["bass"]["reference_db"] - (-2.0)) <= 0.1 and abs(row["bass"]["difference_db"]) <= 0.2
    assert run(reference={"from_bar": 4}) == {"paired": False, "notes": ["no section named 'X' in the profile (it has: A, B, C): mix layer only"]}
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 12}]       # longer than the profile's 8
    longer = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
    assert not longer["paired"] and "only 8 fit" in longer["notes"][0]
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 4, "reference": {"from_bar": 6}}]
    assert not profile.mix_report(m, reference=prof)["sections"][0]["elements"]["paired"]
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 4, "reference": {"bar": 6}}]
    with pytest.raises(ValueError, match="reference"):
        profile.mix_report(m, reference=prof)


def test_elements_where_the_kick_is_out_the_loudest_shared_role_is_the_yardstick(tmp_path):
    out_in_b = [0.0] * 8 + [-62.0] * 8 + [0.0] * 8              # the kick stem 62 dB down in bars 9-16
    ref = stem_set(tmp_path, "ref", kick_db=out_in_b)
    prof = profile_of(tmp_path, ref)
    ours = stem_set(tmp_path, "ours", kick_db=out_in_b, drum_db=DRUM_FILE_DB + 3.0)   # drums +3 dB
    r = profile.mix_report(capture_of(ours), reference=prof, out_dir=tmp_path / "out")
    a, b, c = (s["elements"] for s in r["sections"])
    for e in (a, c):                                              # kick present: unchanged
        assert e["yardstick"] == "kick" and not e["kick_out"]
        assert [g["role"] for g in e["gaps"]] == ["drums"] and abs(e["gaps"][0]["difference_db"] - 3.0) <= 0.2
    assert b["yardstick"] == "bass" and b["kick_out"]               # bass: -6 dB vs the kick's level, the loudest left
    assert any("elements against the bass" in n for n in b["notes"])
    row = {t["role"]: t for t in b["table"]}
    assert "bass" not in row                                          # the yardstick is not a row of its own
    assert abs(row["drums"]["reference_db"] - (DRUMS_VS_KICK + 6.0)) <= 0.1      # drums against the bass, finite
    assert abs(row["drums"]["ours_db"] - (DRUMS_VS_KICK + 6.0 + 3.0)) <= 0.1
    assert [g["role"] for g in b["gaps"]] == ["drums"] and abs(b["gaps"][0]["difference_db"] - 3.0) <= 0.2
    md = pathlib.Path(r["md_path"]).read_text()
    assert "Elements against the bass (kick out)" in md


def test_elements_kick_out_with_no_other_shared_role_leaves_the_table_out(tmp_path):
    out_in_b = [0.0] * 8 + [-62.0] * 8 + [0.0] * 8
    ref = stem_set(tmp_path, "ref", kick_db=out_in_b)
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": ref["bass"]}, TEMPO, T0, SECTIONS)
    m = capture_of(ref, bass=None, synth=ref["synth"], drums=None)         # ours: kick + synth only
    b = profile.mix_report(m, reference=prof)["sections"][1]["elements"]
    assert b["table"] is None and b["yardstick"] is None and b["kick_out"]
    assert any("no other element sounds in both" in n for n in b["notes"])


def decaying(freq, tau, seconds):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)


def beat_loop(freq, bars_n=4, lead=0.0, tau=0.05):
    """Four on the floor at 120 BPM (a hit every 0.5 s, 40 dB down after 0.23 s), `lead` s of silence first."""
    x = np.zeros(int((lead + bars_n * 2.0) * SR))
    for b in range(bars_n * 4):
        s0 = int((lead + b * 0.5) * SR)
        h = decaying(freq, tau, 0.5)
        x[s0:s0 + len(h)] += h
    return x


def roll(freq, bars_n=4, lead=0.0, every=0.06, tau=0.15):
    """Hits 60 ms apart that never fall 40 dB before the next: no clean hit anywhere."""
    x = np.zeros(int((lead + bars_n * 2.0) * SR))
    for k in range(int(bars_n * 2.0 / every)):
        s0 = int((lead + k * every) * SR)
        h = decaying(freq, tau, 0.5)[:len(x) - s0]
        x[s0:s0 + len(h)] += h
    return x


KICK_HZ = 80.0


KICK_LOWER_HZ = KICK_HZ * 2 ** (-2 / 12)       # two semitones down


def drum_files(tmp_path, folder, kick_hz, lead=0.25):
    d = tmp_path / folder
    d.mkdir()
    kick, hh = beat_loop(kick_hz, lead=lead), roll(8000, lead=lead)
    return {"kick": write(d, "kick.wav", stereo(0.5 * kick)), "hh": write(d, "hh.wav", stereo(0.3 * hh)),
            "mix": write(d, "mix.wav", stereo(0.5 * kick + 0.3 * hh))}


def drum_setup(tmp_path, ours_hz=KICK_LOWER_HZ):
    ref = drum_files(tmp_path, "dref", KICK_HZ)
    ours = drum_files(tmp_path, "dours", ours_hz)
    sec = [{"name": "K", "start_bar": 0, "bars": 4}]
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "hh": ref["hh"]}, TEMPO, 0.25, sec,
                                    out=tmp_path / "d.profile.json")
    m = {"tempo": TEMPO, "offset_s": 0.25, "sections": sec,
         "sources": {"master": str(ours["mix"]), "kick": str(ours["kick"]), "hh": str(ours["hh"])}}
    return ref, ours, prof, m


def test_drum_layer_kick_tuned_two_semitones_down_gives_one_pitch_line(tmp_path):
    ref, ours, prof, m = drum_setup(tmp_path)
    r = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")
    kick = next(d for d in r["sections"][0]["drums"]["roles"] if d["role"] == "kick")
    assert len(kick["readings"]) == 1, kick["readings"]
    assert "body pitch" in kick["readings"][0] and "below the reference" in kick["readings"][0]
    import json
    import re
    st_down = float(re.search(r"body pitch ([0-9.]+) st below", kick["readings"][0]).group(1))
    assert abs(st_down - 2.0) <= 0.1, kick["readings"]                   # we tuned the synth 2 st down
    assert kick["ours_hit"] is not None and kick["reference_hit"] is not None and kick["notes"] == []
    md = pathlib.Path(r["md_path"]).read_text()
    assert "## Drum sounds" in md and "- kick: body pitch" in md
    assert json.loads(pathlib.Path(r["json_path"]).read_text())["sections"][0]["drums"]["roles"][0]["role"] == "kick"


def test_drum_layer_a_role_with_only_overlapping_hits_has_no_clean_hit_and_no_readings(tmp_path):
    ref, ours, prof, m = drum_setup(tmp_path)
    hh = next(d for d in profile.mix_report(m, reference=prof)["sections"][0]["drums"]["roles"] if d["role"] == "hh")
    assert hh["readings"] == [] and hh["ours_hit"] is None
    assert any("ours: no clean hit" in n for n in hh["notes"]) and any("the reference: no clean hit" in n for n in hh["notes"])


def test_drum_layer_skips_a_role_whose_stem_changed_and_needs_a_profile(tmp_path):
    import os
    ref, ours, prof, m = drum_setup(tmp_path)
    st = ref["kick"].stat()
    os.utime(ref["kick"], (st.st_atime, st.st_mtime + 5))
    kick = next(d for d in profile.mix_report(m, reference=prof)["sections"][0]["drums"]["roles"] if d["role"] == "kick")
    assert kick["readings"] == [] and "changed or is missing" in kick["notes"][0]
    os.utime(ref["kick"], (st.st_atime, st.st_mtime))
    for reference in (None, str(ref["mix"])):                       # no profile: no drums key, old output
        r = profile.mix_report(m, reference=reference, out_dir=tmp_path / "plain")
        assert all("drums" not in s and "elements" not in s for s in r["sections"])
        assert "Drum sounds" not in pathlib.Path(r["md_path"]).read_text()


def test_elements_kick_out_is_decided_for_both_sides_together(tmp_path):
    """The reviewer's probe: in the break the reference kick is 50 dB down and ours 35 dB down (29 dB
    under the bass: 'in' by the 30 dB rule alone), everything else identical. Both sides must use the
    same fallback yardstick, so there is no gap and the notes name it."""
    ref = stem_set(tmp_path, "ref", kick_db=[0.0] * 8 + [-50.0] * 8 + [0.0] * 8)
    ours = stem_set(tmp_path, "ours", kick_db=[0.0] * 8 + [-35.0] * 8 + [0.0] * 8)
    r = profile.mix_report(capture_of(ours), reference=profile_of(tmp_path, ref), out_dir=tmp_path / "out")
    a, b, c = (s["elements"] for s in r["sections"])
    assert a["yardstick"] == c["yardstick"] == "kick" and a["gaps"] == c["gaps"] == []
    assert b["yardstick"] == "bass" and b["kick_out"] and b["gaps"] == []
    assert any("elements against the bass" in n and "the reference" in n for n in b["notes"]), b["notes"]
    assert "kick" not in {t["role"] for t in b["table"]}
    for t in b["table"]:
        assert abs(t["difference_db"]) <= 0.2, t


def edge_leak_files(tmp_path, early=0.005):
    """kick all through; a bass that starts `early` s before the line between bar 8 and bar 9 (A | B) and
    plays on at -6 dB against the kick, so section A holds only the last few ms of it."""
    d = tmp_path / ("leak%g" % early)
    d.mkdir()
    kick = level_sine(ROLE_HZ["kick"], [0.0] * 24)
    t = np.arange(len(kick)) / SR
    start = int((T0 + 16.0 - early) * SR)
    bass = np.where(np.arange(len(kick)) >= start, KICK_AMP * gain(-6.0) * np.sin(2 * np.pi * 110 * t), 0.0)
    return {"kick": write(d, "kick.wav", stereo(kick)), "bass": write(d, "bass.wav", stereo(bass)),
            "mix": write(d, "mix.wav", stereo(kick + bass))}


@pytest.mark.parametrize("early", [0.0013, 0.005, 0.019])
def test_profile_a_note_starting_just_before_the_section_edge_is_not_in_the_section(tmp_path, early):
    f = edge_leak_files(tmp_path, early)
    secs = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8}]
    p = profile.reference_profile(f["mix"], {"kick": f["kick"], "bass": f["bass"]}, TEMPO, T0, secs)
    a = p["sections"][0]["roles"]["bass"]
    assert a["absent"] and a["per_bar_db"] == [None] * 8, a      # it belongs to the next section's window (20 ms early)
    b = p["sections"][1]["roles"]["bass"]
    assert not b["absent"] and not b["trace"] and abs(b["vs_kick_db"] - (-6.0)) <= 0.1


def test_profile_presence_has_three_levels_and_the_60_db_branch(tmp_path):
    d = tmp_path / "lv"
    d.mkdir()
    sig = {"kick": level_sine(50, [0.0] * 8), "loud": level_sine(110, [-6.0] * 8),
           "ghost": level_sine(1500, [-50.0] * 8), "faint": level_sine(3000, [-75.0] * 8)}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    mix = write(d, "mix.wav", stereo(sum(sig.values())))
    p = profile.reference_profile(mix, paths, TEMPO, T0, SECTIONS[:1])
    r = p["sections"][0]["roles"]
    # mix = kick + loud: 10*log10(1 + 0.25) = 0.97 dB over the kick; ghost -50 and faint -75 dB under the kick's amplitude
    assert not r["loud"]["absent"] and not r["loud"]["trace"]
    assert not r["ghost"]["absent"] and r["ghost"]["trace"] and abs(r["ghost"]["vs_kick_db"] - (-50.0)) <= 0.1
    assert r["faint"]["absent"] and "level_db" not in r["faint"]            # 76 dB under the mix: past ABSENT_DB, and not reported as a level


def test_elements_a_trace_is_listed_and_never_a_gap_or_a_one_side_role(tmp_path):
    d = tmp_path / "tr"
    d.mkdir()

    def files(sub, ghost_db):
        dd = d / sub
        dd.mkdir()
        sig = {"kick": level_sine(50, [0.0] * 8), "bass": level_sine(110, [-6.0] * 8)}
        if ghost_db is not None:
            sig["ghost"] = level_sine(1500, [ghost_db] * 8)
        paths = {k: write(dd, k + ".wav", stereo(v)) for k, v in sig.items()}
        paths["mix"] = write(dd, "mix.wav", stereo(sum(sig.values())))
        return paths

    ref, ours, plain = files("ref", -50.0), files("ours", -44.0), files("plain", None)
    prof = profile.reference_profile(ref["mix"], {k: v for k, v in ref.items() if k != "mix"}, TEMPO, T0, SECTIONS[:1])

    def run(f):
        m = {"tempo": TEMPO, "offset_s": T0, "sections": [dict(SECTIONS[0])],
             "sources": {"master": str(f["mix"]), **{k: str(v) for k, v in f.items() if k != "mix"}}}
        return profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")

    e = run(ours)["sections"][0]["elements"]               # 6 dB apart, both traces
    assert e["gaps"] == [] and [t["role"] for t in e["table"]] == ["bass"]
    assert [(t["role"], t["reference"], t["ours"]) for t in e["traces"]] == [("ghost", "trace", "trace")]
    assert abs(e["traces"][0]["ours_db"] - e["traces"][0]["reference_db"] - 6.0) <= 0.3
    e = run(plain)["sections"][0]["elements"]              # ours has no ghost at all
    assert e["reference_only"] == [] and e["gaps"] == [] and e["traces"][0]["ours"] == "not in the manifest"
    assert "Traces" in pathlib.Path(run(plain)["md_path"]).read_text()


def stereo_set(tmp_path, folder, kick_gain=1.0, pan=0.3, bars_n=16, synth_lr=None):
    """kick (centre), bass (centre) and a synth panned `pan` (right = pan * left): real width.
    `synth_lr` = (left gain, right gain) sets the synth's two channels directly."""
    d = tmp_path / folder
    d.mkdir()
    k = level_sine(50, [0.0] * bars_n) * kick_gain
    b = level_sine(110, [-6.0] * bars_n)
    sy = level_sine(1500, [-12.0] * bars_n)
    gl, gr = synth_lr or (1.0, pan)
    mix = np.stack([k + b + gl * sy, k + b + gr * sy], axis=1)
    return {"kick": write(d, "kick.wav", stereo(k)), "bass": write(d, "bass.wav", stereo(b)),
            "synth": write(d, "synth.wav", np.stack([gl * sy, gr * sy], axis=1)), "mix": write(d, "mix.wav", mix),
            "mix_array": mix}


def test_elements_the_capture_that_is_the_reference_is_found_by_shape_not_level(tmp_path):
    secs = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8}]
    ref = stereo_set(tmp_path, "ref")
    prof = profile.reference_profile(ref["mix"], {k: ref[k] for k in ("kick", "bass", "synth")}, TEMPO, T0, secs)
    assert prof["sections"][0]["mix"]["width"] > 0.05                       # real stereo, not the mono degenerate case

    def flags(master_array, name):
        m = {"tempo": TEMPO, "offset_s": T0, "sections": secs,
             "sources": {"master": str(write(tmp_path, name + ".wav", master_array)), "kick": str(ref["kick"]),
                         "bass": str(ref["bass"]), "synth": str(ref["synth"])}}
        return [s["elements"]["capture_is_reference"] for s in profile.mix_report(m, reference=prof)["sections"]]

    mix = ref["mix_array"]
    assert flags(mix, "same") == [True, True]
    assert flags(mix * gain(-0.5), "quieter") == [True, True]                # the reference played 0.5 dB lower
    assert flags(mix * gain(-3.0), "quieter3") == [True, True]               # the offset is removed, whatever it is
    louder_kick = stereo_set(tmp_path, "kick15", kick_gain=gain(15.0))["mix_array"]    # our kick 15 dB louder
    assert flags(louder_kick, "kick15") == [False, False]
    assert flags(stereo_set(tmp_path, "pan", pan=0.8)["mix_array"], "pan") == [False, False]   # width 0.35 -> 0.10


def test_profile_loading_refuses_a_wrong_shape_or_version_and_says_to_rebuild(tmp_path):
    import copy
    import json

    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    m = capture_of(ref)
    damage = {
        "version 99": lambda p: p.update(version=99),
        "version 1": lambda p: p.update(version=1),
        "no bands": lambda p: p["sections"][0]["mix"].pop("bands_db"),
        "roles as a list": lambda p: p["sections"][0].update(roles=[]),
        "no per_bar": lambda p: p["sections"][0]["roles"]["bass"].pop("per_bar_db"),
        "no trace flag": lambda p: p["sections"][0]["roles"]["bass"].pop("trace"),
        "short per_bar": lambda p: p["sections"][0]["mix"].update(per_bar_db=[0.0]),
        "no inputs": lambda p: p.update(inputs={}),
        "sections not a list": lambda p: p.update(sections=3),
        "name missing": lambda p: p["sections"][0].pop("name"),
    }
    for label, f in damage.items():
        bad = copy.deepcopy(prof)
        f(bad)
        with pytest.raises(ValueError, match="rebuild it with reference_profile"):
            profile.mix_report(m, reference=bad)
    assert profile.mix_report(m, reference=prof, out_dir=tmp_path / "ok")["reference_kind"] == "profile"


def test_manifest_source_may_be_a_list_of_files_summed_into_one_role(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    ours = stem_set(tmp_path, "ours")
    x, _ = sf.read(str(ours["bass"]), dtype="float32")
    halves = [write(tmp_path, "bass_a.wav", x * 0.5), write(tmp_path, "bass_b.wav", x * 0.5)]    # 0.5 + 0.5 = the bass
    m = capture_of(ours, bass=None)
    m["sources"]["bass"] = [str(h) for h in halves]
    r = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")
    for s in r["sections"]:
        row = {t["role"]: t for t in s["elements"]["table"]}
        assert abs(row["bass"]["ours_db"] - (-6.0)) <= 0.1 and s["elements"]["gaps"] == []
    one = tmp_path / "just_one"
    with pytest.raises(ValueError, match="no such audio file"):
        m["sources"]["bass"] = [str(halves[0]), str(one)]
        profile.mix_report(m, reference=prof)
    m["sources"]["bass"] = []
    with pytest.raises(ValueError, match="no files"):
        profile.mix_report(m, reference=prof)


def test_profile_small_things_json_extension_tempo_note_markdown_and_stale_tolerance(tmp_path):
    import os
    ref = stem_set(tmp_path, "ref")
    p = profile.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, SECTIONS[:1], out=tmp_path / "X.JSON")
    assert pathlib.Path(p["json_path"]) == (tmp_path / "X.JSON").resolve() and (tmp_path / "X.JSON").is_file()
    prof = profile_of(tmp_path, ref)
    m = capture_of(stem_set(tmp_path, "ours"))
    m["tempo"] = 121.0
    r = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")
    assert any("tempo differs" in n and "121" in n and "120" in n for n in r["sections"][0]["elements"]["notes"])
    md = pathlib.Path(r["md_path"]).read_text()
    assert "Reference profile:" in md and "Reference: `" not in md                    # one reference sentence
    st = ref["bass"].stat()
    os.utime(ref["bass"], (st.st_atime, st.st_mtime + 1.5))                         # a cloud re-sync touches mtimes
    assert profile.profile_stale(prof) == []
    os.utime(ref["bass"], (st.st_atime, st.st_mtime + 3.0))
    assert [s["role"] for s in profile.profile_stale(prof)] == ["bass"]
    os.utime(ref["bass"], (st.st_atime, st.st_mtime))
    with open(ref["bass"], "ab") as f:                                              # same mtime window, different size
        f.write(b"\0" * 8)
    os.utime(ref["bass"], (st.st_atime, st.st_mtime))
    assert [s["role"] for s in profile.profile_stale(prof)] == ["bass"]
    with pytest.raises(ValueError, match="beats_per_bar"):
        profile.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, SECTIONS[:1], beats_per_bar=float("nan"))


def test_profile_a_phrase_in_four_of_eight_bars_is_present_and_one_bar_in_eight_is_not_enough(tmp_path):
    d = tmp_path / "ph"
    d.mkdir()
    sig = {"kick": level_sine(50, [0.0] * 8),
           "phrase": level_sine(1500, [-6.0 if i in (0, 1, 5, 6) else None for i in range(8)]),     # 4 of 8 bars
           "pair": level_sine(3000, [-6.0 if i in (2, 5) else None for i in range(8)]),             # 2 of 8: a quarter
           "stray": level_sine(110, [-6.0 if i == 3 else None for i in range(8)]),                  # 1 of 8
           "quiet_phrase": level_sine(200, [-50.0 if i < 4 else None for i in range(8)])}           # 4 bars, trace zone
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    mix = write(d, "mix.wav", stereo(sum(sig.values())))
    r = profile.reference_profile(mix, paths, TEMPO, T0, SECTIONS[:1])["sections"][0]["roles"]
    assert not r["phrase"]["absent"] and not r["phrase"]["trace"]
    assert not r["pair"]["absent"] and not r["pair"]["trace"]                 # exactly a quarter of the bars
    assert r["stray"]["absent"] and "level_db" not in r["stray"]                                     # one bar in eight is under a quarter
    assert not r["quiet_phrase"]["absent"] and r["quiet_phrase"]["trace"]      # -50 dB: a quarter within 60, none within 40


def test_elements_a_role_panned_wider_on_our_side_is_one_width_gap_and_no_level_gap(tmp_path):
    secs = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8}]
    ref = stereo_set(tmp_path, "ref")                                  # synth (1, 0.3): width 0.35
    prof = profile.reference_profile(ref["mix"], {k: ref[k] for k in ("kick", "bass", "synth")}, TEMPO, T0, secs)
    # the same power, 1 + 0.3^2 = 1.09, spread as (0.934, -0.467): 0.934^2 + 0.467^2 = 1.09, width 0.75
    ours = stereo_set(tmp_path, "ours", synth_lr=(0.934, -0.467))
    m = {"tempo": TEMPO, "offset_s": T0, "sections": secs,
         "sources": {"master": str(ours["mix"]), "kick": str(ours["kick"]), "bass": str(ours["bass"]),
                     "synth": str(ours["synth"])}}
    r = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")
    for s in r["sections"]:
        e = s["elements"]
        assert e["gaps"] == [], e["gaps"]                              # the level is the same
        assert [g["role"] for g in e["width_gaps"]] == ["synth"], e["width_gaps"]
        g = e["width_gaps"][0]
        assert abs(g["reference"] - 0.35) <= 0.02 and abs(g["ours"] - 0.75) <= 0.02 and abs(g["difference"] - 0.40) <= 0.03
        row = {t["role"]: t for t in e["table"]}
        assert row["bass"]["reference_width"] == row["bass"]["ours_width"] == 0.0         # centred on both sides
    md = pathlib.Path(r["md_path"]).read_text()
    assert "Width (ref / ours)" in md and "Width gaps (more than 0.1 apart): synth 0.35 -> 0.75." in md
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 4}]           # a partial section: the note says whole section
    e = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
    assert "widths and loudness are for the whole section" in e["notes"][0] and e["width_gaps"] == []


def test_elements_mono_on_both_sides_has_no_width_gap(tmp_path):
    ref = stem_set(tmp_path, "ref")
    r = profile.mix_report(capture_of(stem_set(tmp_path, "ours")), reference=profile_of(tmp_path, ref))
    for s in r["sections"]:
        assert s["elements"]["width_gaps"] == []
        assert all(t["reference_width"] == t["ours_width"] == 0.0 for t in s["elements"]["table"])


def hit_train(freq, tau, times, total_s, amp=1.0, sr=SR):
    """Decaying sines `amp * exp(-t/tau)` starting at each time in `times` (s), in a file `total_s` long."""
    x = np.zeros(int(total_s * sr))
    h = decaying(freq, tau, 0.5)
    for t in times:
        s0 = int(round(t * sr))
        if 0 <= s0 < len(x):
            x[s0:s0 + len(h)] += amp * h[:len(x) - s0]
    return x


def identical_material(tmp_path, folder, shift_s, tau, bars_n=8):
    """A kick train (every beat of 124 BPM, `shift_s` early/late of the grid), a snare on beats 2 and 4 and
    a steady bass: the same material for the reference and for us."""
    d = tmp_path / folder
    d.mkdir()
    bar = 4 * 60 / 124
    total = 0.5 + bars_n * bar + 1.0
    k = 0.8 * hit_train(55, tau, [0.5 + b * bar / 4 + shift_s for b in range(bars_n * 4)], total)
    sn = 0.3 * hit_train(220, tau, [0.5 + (b * 2 + 1) * bar / 4 + shift_s for b in range(bars_n * 2)], total)
    bs = np.zeros_like(k)
    bs[int(0.5 * SR):int((0.5 + bars_n * bar) * SR)] = 0.1 * np.sin(2 * np.pi * 110 * np.arange(int(bars_n * bar * SR)) / SR)
    paths = {"kick": write(d, "kick.wav", stereo(k)), "snare": write(d, "snare.wav", stereo(sn)),
             "bass": write(d, "bass.wav", stereo(bs)), "mix": write(d, "mix.wav", stereo(k + sn + bs))}
    return paths


@pytest.mark.parametrize("tau,shift", [(0.04, -0.003), (0.015, 0.004), (0.008, -0.003)])
def test_elements_partial_pairing_of_identical_material_has_no_bias_and_the_arc_agrees(tmp_path, tau, shift):
    """4 and 6 of 8 bars of the same material (the same timing on both sides): every row and every arc
    block within 0.05 dB. Every window starts 20 ms early on both sides, so nothing is cut at a section or bar line."""
    ref = identical_material(tmp_path, "ref", shift, tau)
    ours = identical_material(tmp_path, "ours", shift, tau)
    secs = [{"name": "A", "start_bar": 0, "bars": 8}]
    prof = profile.reference_profile(ref["mix"], {k: ref[k] for k in ("kick", "snare", "bass")}, 124.0, 0.5, secs)
    for label, msecs in (("first 4", [{"name": "X", "start_bar": 0, "bars": 4, "reference": {"section": "A"}}]),
                         ("last 4", [{"name": "X", "start_bar": 4, "bars": 4, "reference": {"section": "A", "from_bar": 4}}]),
                         ("first 6", [{"name": "X", "start_bar": 0, "bars": 6, "reference": {"section": "A"}}]),
                         ("last 6", [{"name": "X", "start_bar": 2, "bars": 6, "reference": {"section": "A", "from_bar": 2}}]),
                         ("whole", [{"name": "A", "start_bar": 0, "bars": 8}])):
        m = {"tempo": 124.0, "offset_s": 0.5, "sections": msecs,
             "sources": {"master": ours["mix"], "kick": ours["kick"], "snare": ours["snare"], "bass": ours["bass"]}}
        e = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
        assert {t["role"] for t in e["table"]} == {"snare", "bass"}, label
        for t in e["table"]:
            assert abs(t["difference_db"]) <= 0.05, (label, t)
        if msecs[0]["bars"] > 4:
            assert set(e["arc"]) == {"snare", "bass"} and all(len(a["ours"]) == len(a["reference"]) == 2
                                                               for a in e["arc"].values()), label
        else:
            assert e["arc"] == {}, label
        for role, a in e["arc"].items():
            for r, o in zip(a["reference"], a["ours"]):
                assert abs(r - o) <= 0.05, (label, role, a)


def drum_pattern(tmp_path, folder, shift_s, early_s=0.003, extra=None, ghosts=False, bars_n=8, skip_first=False):
    """kick (60 Hz, every beat, `early_s` early like Flare's), snare (200 Hz, beats 2 and 4), hh (8 kHz, every
    8th), 120 BPM, 0.5 s lead; all `shift_s` late. `extra` = (time, Hz) adds one more kick hit; `ghosts` plays
    three beats of each bar 14 dB down."""
    d = tmp_path / folder
    d.mkdir()
    total = 0.5 + bars_n * 2.0 + 1.0
    beats = [0.5 + b * 0.5 - early_s + shift_s for b in range(1 if skip_first else 0, bars_n * 4)]
    if ghosts:        # an accent on beat 2 of each bar, ghosts (-14 dB) on the other three: most hits are ghosts
        loud = [t for i, t in enumerate(beats) if i % 4 == 1]
        k = 0.8 * hit_train(60, 0.05, loud, total) + 0.8 * 0.2 * hit_train(60, 0.05, [t for i, t in enumerate(beats) if i % 4 != 1], total)
    else:
        k = 0.8 * hit_train(60, 0.05, beats, total)
    if extra:
        k = k + 0.8 * hit_train(extra[1], 0.05, [extra[0] + shift_s], total)
    sn = 0.5 * hit_train(200, 0.04, [0.5 + (2 * b + 1) * 0.5 + shift_s for b in range(bars_n * 2)], total)
    hh = 0.3 * hit_train(8000, 0.01, [0.5 + b * 0.25 + shift_s for b in range(bars_n * 8)], total)
    paths = {"kick": write(d, "kick.wav", stereo(k)), "snare": write(d, "snare.wav", stereo(sn)),
             "hh": write(d, "hh.wav", stereo(hh)), "mix": write(d, "mix.wav", stereo(k + sn + hh))}
    return paths


def drum_report(tmp_path, ref, ours, offset, sec=None, roles=("kick", "snare", "hh")):
    prof = profile.reference_profile(ref["mix"], {r: ref[r] for r in roles}, 120.0, 0.5, [{"name": "K", "start_bar": 0, "bars": 4}])
    m = {"tempo": 120.0, "offset_s": offset, "sections": [sec or {"name": "K", "start_bar": 0, "bars": 4}],
         "sources": {"master": ours["mix"], **{r: ours[r] for r in roles}}}
    return profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"]


@pytest.mark.parametrize("shift", [0.005, -0.005])
def test_drum_layer_identical_audio_shifted_5_ms_gives_no_readings(tmp_path, shift):
    """The kick is 3 ms early (a hit starts before the section line, the next section's first hit before the
    end line). The same audio is read 5 ms off by the manifest: no readings on any role."""
    ref = drum_pattern(tmp_path, "ref", 0.0)
    ours = drum_pattern(tmp_path, "ours", 0.0)
    roles = drum_report(tmp_path, ref, ours, 0.5 + shift)
    assert [d["role"] for d in roles] == ["kick", "snare", "hh"]
    for d in roles:
        assert d["readings"] == [] and d["ours_hit"] is not None and d["reference_hit"] is not None, d
        assert d["notes"] == [], d["notes"]
        assert abs(d["ours_at_ms"] - d["reference_at_ms"] - 1000 * shift * -1) <= 12, d    # the same hit, 5 ms apart


def test_drum_layer_a_hit_before_the_section_line_is_never_the_clean_hit(tmp_path):
    """The reference kick has an odd hit (90 Hz) 150 ms before the line, with the most room; the pattern in the
    section is 60 Hz. Picking the odd hit would read a pitch difference against ours."""
    # the odd hit has 650 ms of room: the beat at the line is dropped from the reference, so nothing sits on its tail
    ref = drum_pattern(tmp_path, "ref", 0.0, extra=(0.5 - 0.150, 90), skip_first=True)
    ours = drum_pattern(tmp_path, "ours", 0.0)
    roles = drum_report(tmp_path, ref, ours, 0.5)
    kick = roles[0]
    assert kick["readings"] == [] and kick["reference_at_ms"] >= -5, kick
    assert not any("pitch" in r for r in kick["readings"])


def test_drum_layer_ghost_notes_are_not_the_clean_hit(tmp_path):
    ref = drum_pattern(tmp_path, "ref", 0.0, ghosts=True)
    ours = drum_pattern(tmp_path, "ours", 0.0, ghosts=True)
    kick = drum_report(tmp_path, ref, ours, 0.5)[0]
    # the accent is beat 2 (0.5 s into the bar); the median hit is a ghost, so only the 6 dB band keeps it out
    assert kick["readings"] == [] and abs(kick["reference_at_ms"] - 497) <= 8, kick


def test_drum_layer_runs_only_for_roles_present_on_both_sides_and_one_failure_is_one_note(tmp_path, monkeypatch):
    ref = drum_pattern(tmp_path, "ref", 0.0)
    ours = drum_pattern(tmp_path, "ours", 0.0)
    quiet = tmp_path / "quiet_kick.wav"
    x, _ = sf.read(str(ours["kick"]), dtype="float32")
    sf.write(str(quiet), x * gain(-52.0), SR, subtype="FLOAT")                    # our kick 52 dB down: a trace
    ours2 = dict(ours, kick=quiet)
    roles = {d["role"]: d for d in drum_report(tmp_path, ref, ours2, 0.5)}
    assert roles["kick"]["readings"] == [] and "not present on both sides" in roles["kick"]["notes"][0]
    assert roles["snare"]["readings"] == [] and roles["snare"]["ours_hit"] is not None       # the others still run
    assert roles["snare"]["notes"] == [], roles["snare"]
    from roadie_analysis import oneshot

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(oneshot, "compare", boom)
    roles = drum_report(tmp_path, ref, ours, 0.5)                                  # the report survives
    assert all(any("failed: RuntimeError: synthetic failure" in n for n in d["notes"]) for d in roles)


def test_drum_layer_reads_only_the_span_it_needs(tmp_path):
    x = np.arange(48000 * 3, dtype=np.float32) / 1e6
    p = write(tmp_path, "long.wav", np.stack([x, -x], axis=1))
    got, start = profile._read_span([{"path": str(p)}, {"path": str(p)}], SR, 1.0, 1.5)
    assert start == 1.0 and got.shape == (24000, 2)
    assert np.allclose(got[:, 0], 2 * x[SR:SR + 24000], atol=1e-6)               # two files summed
    got, start = profile._read_span([{"path": str(p)}], SR, 2.9, 4.0)             # clamped to the file
    assert got.shape[0] == int(0.1 * SR) and start == pytest.approx(2.9, abs=1e-4)
    assert profile._read_span([{"path": str(p)}], SR, 5.0, 6.0) is None


def test_elements_a_present_role_against_a_trace_is_compared_and_can_be_a_gap(tmp_path):
    def files(sub, hat_db):
        dd = tmp_path / sub
        dd.mkdir()
        sig = {"kick": level_sine(50, [0.0] * 8), "bass": level_sine(110, [-6.0] * 8),
               "hat": level_sine(8000, [hat_db] * 8)}
        paths = {k: write(dd, k + ".wav", stereo(v)) for k, v in sig.items()}
        paths["mix"] = write(dd, "mix.wav", stereo(sum(sig.values())))
        return paths

    ref, ours = files("ref", -35.0), files("ours", -45.0)          # elements together = kick + bass = +0.97 dB: 36 / 46 dB under
    prof = profile.reference_profile(ref["mix"], {k: v for k, v in ref.items() if k != "mix"}, TEMPO, T0, SECTIONS[:1])
    assert not prof["sections"][0]["roles"]["hat"]["trace"]
    m = {"tempo": TEMPO, "offset_s": T0, "sections": [dict(SECTIONS[0])],
         "sources": {"master": str(ours["mix"]), **{k: str(v) for k, v in ours.items() if k != "mix"}}}
    r = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")
    e = r["sections"][0]["elements"]
    assert e["presence"]["hat"] == {"reference": "present", "ours": "trace"}
    assert [(g["role"]) for g in e["gaps"]] == ["hat"] and abs(e["gaps"][0]["difference_db"] - (-10.0)) <= 0.3
    assert [t["trace_side"] for t in e["table"] if t["role"] == "hat"] == ["ours"] and e["traces"] == []
    assert "hat (trace in ours)" in pathlib.Path(r["md_path"]).read_text()
    both = profile.mix_report(dict(m, sources=dict(m["sources"], **{k: str(v) for k, v in ref.items() if k != "mix"},
                                                   master=str(ref["mix"]))), reference=prof)["sections"][0]["elements"]
    assert both["gaps"] == [] and both["traces"] == []


def test_elements_presence_does_not_depend_on_the_master_makeup_gain(tmp_path):
    def files(sub, pad_db):
        dd = tmp_path / sub
        dd.mkdir()
        sig = {"kick": level_sine(50, [0.0] * 8), "bass": level_sine(110, [-6.0] * 8),
               "pad": level_sine(700, [pad_db] * 8)}
        paths = {k: write(dd, k + ".wav", stereo(v)) for k, v in sig.items()}
        paths["mix"] = write(dd, "mix.wav", stereo(sum(sig.values())))
        return paths, sig

    ref, _ = files("ref", -35.0)                                    # 36 dB under the elements together
    ours, sig = files("ours", -29.0)                                # 6 dB louder: 30 dB under
    prof = profile.reference_profile(ref["mix"], {k: v for k, v in ref.items() if k != "mix"}, TEMPO, T0, SECTIONS[:1])
    for makeup in (0.0, 12.0, 24.0, 36.0):          # at 36 dB the pad is 66 dB under the master: absent if judged by it
        master = write(tmp_path, "master_%d.wav" % makeup, stereo(sum(sig.values()) * gain(makeup)))
        m = {"tempo": TEMPO, "offset_s": T0, "sections": [dict(SECTIONS[0])],
             "sources": {"master": str(master), **{k: str(v) for k, v in ours.items() if k != "mix"}}}
        e = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
        assert [g["role"] for g in e["gaps"]] == ["pad"] and abs(e["gaps"][0]["difference_db"] - 6.0) <= 0.3, makeup


def test_elements_a_role_absent_over_the_profile_section_is_present_in_the_paired_bars(tmp_path):
    d = tmp_path / "riser"
    d.mkdir()
    sig = {"kick": level_sine(50, [0.0] * 16), "riser": level_sine(900, [-6.0 if 12 <= i < 15 else None for i in range(16)])}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    mix = write(d, "mix.wav", stereo(sum(sig.values())))
    secs = [{"name": "A", "start_bar": 0, "bars": 16}]
    prof = profile.reference_profile(mix, paths, TEMPO, T0, secs)
    assert prof["sections"][0]["roles"]["riser"]["absent"] and len(prof["sections"][0]["roles"]["riser"]["per_bar_db"]) == 16
    m = {"tempo": TEMPO, "offset_s": T0, "sources": {"master": str(mix), **{k: str(v) for k, v in paths.items()}},
         "sections": [{"name": "X", "start_bar": 12, "bars": 4, "reference": {"section": "A", "from_bar": 12}}]}
    e = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
    assert e["presence"]["riser"] == {"reference": "present", "ours": "present"}
    assert [t["role"] for t in e["table"]] == ["riser"] and abs(e["table"][0]["difference_db"]) <= 0.1
    assert e["reference_only"] == [] and e["ours_only"] == [] and e["silent_in_both"] == []


def test_elements_a_kick_out_on_one_side_only_is_listed_as_a_difference(tmp_path):
    ref = stem_set(tmp_path, "ref")
    ours = stem_set(tmp_path, "ours", kick_db=-41.0)              # 35 dB under the bass: out
    r = profile.mix_report(capture_of(ours), reference=profile_of(tmp_path, ref), out_dir=tmp_path / "out")
    e = r["sections"][0]["elements"]
    assert e["yardstick"] == "bass" and e["kick_out"]
    kick = [x for x in e["reference_only"] if x["role"] == "kick"]
    assert len(kick) == 1 and "kick out in ours: 35 dB under the loudest other role" in kick[0]["reason"], e["reference_only"]
    assert "kick" not in {t["role"] for t in e["table"]}
    ours_in = profile.mix_report(capture_of(ref), reference=profile_of(tmp_path, ours := stem_set(tmp_path, "ours2", kick_db=-41.0)))
    k2 = [x for x in ours_in["sections"][0]["elements"]["ours_only"] if x["role"] == "kick"]
    assert len(k2) == 1 and "kick out in the reference" in k2[0]["reason"]


def test_elements_the_kick_out_line_is_30_db(tmp_path):
    """Kick 25 dB under the loudest other role on both sides: the kick is the yardstick; 35 dB under: bass is."""
    for kick_db, want in ((-31.0, "kick"), (-41.0, "bass")):               # the bass sits at -6 dB
        ref, ours = stem_set(tmp_path, "r%d" % -kick_db, kick_db=kick_db), stem_set(tmp_path, "o%d" % -kick_db, kick_db=kick_db)
        e = profile.mix_report(capture_of(ours), reference=profile_of(tmp_path, ref))["sections"][0]["elements"]
        assert e["yardstick"] == want, (kick_db, e["yardstick"])


def test_elements_the_capture_is_the_reference_tolerance_is_a_third_of_a_db(tmp_path):
    """Mono material (width cannot decide it): the bass alone is a little louder in the master. Its band moves
    by that much against the others, so 0.1 dB is inside the 0.3 dB tolerance and 1.5 dB is outside."""
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    bass, _ = sf.read(str(ref["bass"]), dtype="float32")
    mix, _ = sf.read(str(ref["mix"]), dtype="float32")

    def flagged(extra_db, name):
        master = write(tmp_path, name + ".wav", mix + (gain(extra_db) - 1.0) * bass)
        r = profile.mix_report(capture_of(ref, master=master), reference=prof)
        return [s["elements"]["capture_is_reference"] for s in r["sections"]]

    assert flagged(0.1, "a") == [True, True, True] and flagged(1.5, "b") == [False, False, False]


def test_profile_presence_is_against_the_stems_and_not_a_hotter_mix_file(tmp_path):
    """The reference mix file is 20 dB hotter than its stems (a limited master). A pad 35 dB under the stems'
    sum is present in the profile; judged against the mix it would be 55 dB under and a trace."""
    d = tmp_path / "hot"
    d.mkdir()
    sig = {"kick": level_sine(50, [0.0] * 8), "bass": level_sine(110, [-6.0] * 8), "pad": level_sine(700, [-34.0] * 8)}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    mix = write(d, "mix.wav", stereo(sum(sig.values()) * gain(20.0)))
    pad = profile.reference_profile(mix, paths, TEMPO, T0, SECTIONS[:1])["sections"][0]["roles"]["pad"]
    assert not pad["absent"] and not pad["trace"]          # 34 + 0.97 = 35 dB under the stems' sum


def test_elements_no_width_gaps_on_a_partial_pairing_and_a_note_says_so(tmp_path):
    """Identical stereo material whose pad is mono in bars 1-4 and wide in bars 5-8. Paired as 4 of 8 bars the
    reference's width is the whole section's, which is not comparable with the paired bars: no width gaps."""
    def files(folder):
        d = tmp_path / folder
        d.mkdir()
        k = level_sine(50, [0.0] * 8)
        pad = level_sine(900, [-6.0] * 8)
        right = pad.copy()
        right[int((T0 + 8.0) * SR):] *= -1.0                       # the second half in antiphase: all side
        return {"kick": write(d, "kick.wav", stereo(k)), "pad": write(d, "pad.wav", np.stack([pad, right], axis=1)),
                "mix": write(d, "mix.wav", np.stack([k + pad, k + right], axis=1))}

    ref, ours = files("ref"), files("ours")
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "pad": ref["pad"]}, TEMPO, T0, SECTIONS[:1])
    m = {"tempo": TEMPO, "offset_s": T0, "sources": {"master": str(ours["mix"]), "kick": str(ours["kick"]), "pad": str(ours["pad"])},
         "sections": [{"name": "A", "start_bar": 0, "bars": 4}]}
    e = profile.mix_report(m, reference=prof)["sections"][0]["elements"]
    assert e["width_gaps"] == [] and "width gaps are not ranked" in e["notes"][0]
    assert e["table"][0]["reference_width"] > 0.2 and e["table"][0]["ours_width"] == 0.0       # they do differ
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 8}]                              # the whole section: all fine
    assert profile.mix_report(m, reference=prof)["sections"][0]["elements"]["width_gaps"] == []


def lead_pattern(tmp_path, folder, bars_n=4):
    """120 BPM, 0.5 s lead-in, two 2-bar sections. A kick on every beat 3 ms early; a sparse tom (100 Hz) with a hit
    10 ms before the line of each section and one more 1.75 s into it; hats (8 kHz, short decay) on every 8th."""
    d = tmp_path / folder
    d.mkdir()
    total = 0.5 + bars_n * 2.0 + 1.0
    kick = 0.8 * hit_train(60, 0.05, [0.5 + b * 0.5 - 0.003 for b in range(bars_n * 4)], total)
    toms = []
    for sec in range(bars_n // 2):
        toms += [0.5 + sec * 4.0 - 0.010, 0.5 + sec * 4.0 + 1.75]
    tom = 0.5 * hit_train(100, 0.15, toms, total)
    hh = 0.25 * hit_train(8000, 0.01, [0.5 + b * 0.25 for b in range(bars_n * 8)], total)
    paths = {"kick": write(d, "kick.wav", stereo(kick)), "toms": write(d, "toms.wav", stereo(tom)),
             "hh": write(d, "hh.wav", stereo(hh)), "mix": write(d, "mix.wav", stereo(kick + tom + hh))}
    return paths


@pytest.mark.parametrize("lines_moved", [0.0, -0.020])
def test_drum_layer_and_levels_are_the_same_for_identical_audio_read_at_any_offset(tmp_path, lines_moved):
    """The same audio on both sides, read -20 .. +20 ms apart (odd offsets among them, off the 1 ms grid), with every
    section line also moved 20 ms against the pattern: no gaps, no width gaps, no one-side roles, no drum readings,
    and our hit is the one our own segmentation offers at the reference's place (no note at all)."""
    ref = lead_pattern(tmp_path, "ref")
    ours = lead_pattern(tmp_path, "ours")
    secs = [{"name": "K", "start_bar": 0, "bars": 2}, {"name": "L", "start_bar": 2, "bars": 2}]
    t0 = 0.5 + lines_moved
    prof = profile.reference_profile(ref["mix"], {r: ref[r] for r in ("kick", "toms", "hh")}, 120.0, t0, secs)
    for delta in (-0.020, -0.013, -0.007, 0.0047, 0.013, 0.020):
        m = {"tempo": 120.0, "offset_s": t0 + delta, "sections": secs,
             "sources": {"master": ours["mix"], **{r: ours[r] for r in ("kick", "toms", "hh")}}}
        for sec in profile.mix_report(m, reference=prof)["sections"]:
            e = sec["elements"]
            label = (lines_moved, delta, sec["name"])
            assert e["gaps"] == [] and e["width_gaps"] == [] and e["reference_only"] == [] and e["ours_only"] == [], (label, e)
            assert {t["role"] for t in e["table"]} == {"toms", "hh"}, label
            for d in sec["drums"]["roles"]:
                assert d["readings"] == [] and d["ours_hit"] is not None and d["notes"] == [], (label, d)


def glide_kick(times, total_s, amp=0.8, freq=55.0, tau=0.08, seconds=0.5, sr=SR):
    """Kicks that glide 24 st down to `freq` (time constant 20 ms) and decay with `tau`, `seconds` long, starting at
    each of `times`."""
    t = np.arange(int(seconds * sr)) / sr
    h = np.sin(2 * np.pi * np.cumsum(freq * 2 ** (2.0 * np.exp(-t / 0.02))) / sr) * np.exp(-t / tau)
    x = np.zeros(int(total_s * sr))
    for s in times:
        s0 = int(round(s * sr))
        x[s0:s0 + len(h)] += amp * h[:len(x) - s0]
    return x


@pytest.mark.parametrize("delta", [0.0005, 0.0015, -0.0035])
def test_drum_layer_identical_kick_with_an_early_soft_hit_read_half_a_ms_off_gives_no_readings(tmp_path, delta):
    """140 BPM gliding 55 Hz kick on every beat, plus one soft hit (-8 dB) 232 ms before the section line; the same
    audio on both sides. Read a few ms off, our slice splits that soft hit's tail differently, so the first pass places
    the line's hit at a different onset than the reference's (8.4 against 7.2 ms early): readings from the hits as the
    first pass cut them would differ. Both hits measured from cuts of their own, the tail's level read over 20 ms (not
    one 1 ms frame of a 55 Hz tail, which is loud or quiet with its phase), must read the same."""
    beat = 60.0 / 140.0
    total = 0.6 + 16 * beat + 1.5
    k = glide_kick([0.6 + b * beat for b in range(16)], total) + glide_kick([0.6 + 4 * beat - 0.232], total, amp=0.32)
    d = {}
    for side in ("ref", "ours"):
        f = tmp_path / side
        f.mkdir()
        d[side] = {"kick": write(f, "kick.wav", stereo(k)), "mix": write(f, "mix.wav", stereo(k))}
    secs = [{"name": "K", "start_bar": 1, "bars": 2}]
    prof = profile.reference_profile(d["ref"]["mix"], {"kick": d["ref"]["kick"]}, 140.0, 0.6, secs)
    m = {"tempo": 140.0, "offset_s": 0.6 + delta, "sections": secs,
         "sources": {"master": d["ours"]["mix"], "kick": d["ours"]["kick"]}}
    kick = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"][0]
    assert kick["readings"] == [] and kick["ours_hit"] is not None and kick["notes"] == [], kick


def one_role_report(tmp_path, role, ref_x, our_x, total_s=None):
    """The drum layer's entry for `role` alone, 120 BPM, one 4-bar section from 0.5 s, read at the same offset."""
    d = {}
    for side, x in (("ref", ref_x), ("ours", our_x)):
        f = tmp_path / side
        f.mkdir()
        d[side] = {role: write(f, role + ".wav", stereo(x)), "mix": write(f, "mix.wav", stereo(x))}
    secs = [{"name": "K", "start_bar": 0, "bars": 4}]
    prof = profile.reference_profile(d["ref"]["mix"], {role: d["ref"][role]}, 120.0, 0.5, secs)
    m = {"tempo": 120.0, "offset_s": 0.5, "sections": secs, "sources": {"master": d["ours"]["mix"], role: d["ours"][role]}}
    return profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"][0]


def long_toms(times, total_s, freq=100.0, tau=0.4):
    """Toms 1.2 s long (they end 26 dB down), so each one's tail is still sounding at the next hit 1 s later."""
    t = np.arange(int(1.2 * SR)) / SR
    h = 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)
    x = np.zeros(int(total_s * SR))
    for s0 in (int(round(v * SR)) for v in times):
        x[s0:s0 + len(h)] += h[:len(x) - s0]
    return x


@pytest.mark.parametrize("late", [0.050, 0.120])
def test_drum_layer_toms_late_everywhere_are_not_at_the_same_place_and_read_nothing_else(tmp_path, late):
    """The same toms (beats 1 and 3, each still sounding when the next starts) with ours 50 or 120 ms late everywhere.
    The tail of the hit before sits at the reference's place on our side; it is not a hit. The layer must say the hits
    are not at the same place, with ours where it really is, and read no timbre difference (there is none)."""
    beats = [0.5 + b * 0.5 for b in range(16) if b % 4 in (0, 2)]
    total = 0.5 + 8.0 + 2.0
    toms = one_role_report(tmp_path, "toms", long_toms(beats, total), long_toms([b + late for b in beats], total))
    assert any("not at the same place" in n for n in toms["notes"]), toms
    assert not any("reference's place" in n for n in toms["notes"]), toms
    assert abs(toms["ours_at_ms"] - toms["reference_at_ms"] - 1000 * late) <= 2, toms
    assert toms["readings"] == [], toms


def test_drum_layer_an_808_late_and_lower_is_not_at_the_same_place_and_reads_only_its_pitch(tmp_path):
    """An 808 (55 Hz, gliding, tau 350 ms, 1.2 s long: each still sounding at the next hit 1 s later); ours is 3 st
    lower and 70 ms late everywhere. Expected: the hits are not at the same place (ours 70 ms later), and the one timbre
    reading is the body pitch, 3 st below; no attack, noise or click reading from a cut that starts in a tail."""
    beats = [0.5 + b * 0.5 for b in range(16) if b % 4 in (0, 2)]
    total = 0.5 + 8.0 + 2.0
    ref = glide_kick(beats, total, tau=0.35, seconds=1.2)
    ours = glide_kick([b + 0.070 for b in beats], total, tau=0.35, seconds=1.2, freq=55.0 * 2 ** (-3 / 12))
    kick = one_role_report(tmp_path, "kick", ref, ours)
    assert any("not at the same place" in n for n in kick["notes"]), kick
    assert abs(kick["ours_at_ms"] - kick["reference_at_ms"] - 70.0) <= 2, kick
    assert len(kick["readings"]) == 1 and kick["readings"][0].startswith("body pitch 3.0 st below"), kick


@pytest.mark.parametrize("delta", [0.006, 0.010])
def test_drum_layer_the_reference_hit_last_in_its_slice_is_compared_over_the_same_length(tmp_path, delta):
    """One tom (100 Hz, tau 150 ms, 0.5 s long) on beat 4 of a one-bar section at 120 BPM, and an accent twice as
    loud 2 ms after the reference's slice ends (a beat past the section, 50 ms short of the beat line). The tom is
    the last hit in the reference's slice. Our slice runs 30 ms longer, and read 6 or 10 ms late it holds the
    accent's first 34 or 38 ms: our cut must stop where the reference's does, or it holds the accent and reads an
    attack about a second long."""
    total = 5.0
    x = 0.5 * hit_train(100, 0.15, [2.0], total) + 1.0 * hit_train(100, 0.15, [2.952], total)
    d = {}
    for side in ("ref", "ours"):
        f = tmp_path / side
        f.mkdir()
        d[side] = {"toms": write(f, "toms.wav", stereo(x)), "mix": write(f, "mix.wav", stereo(x))}
    secs = [{"name": "K", "start_bar": 0, "bars": 1}]
    prof = profile.reference_profile(d["ref"]["mix"], {"toms": d["ref"]["toms"]}, 120.0, 0.5, secs)
    m = {"tempo": 120.0, "offset_s": 0.5 + delta, "sections": secs,
         "sources": {"master": d["ours"]["mix"], "toms": d["ours"]["toms"]}}
    toms = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"][0]
    assert toms["readings"] == [] and toms["ours_hit"] is not None and toms["notes"] == [], toms
    assert abs(toms["reference_at_ms"] - 1500.0) <= 1 and abs(toms["ours_at_ms"] - (1500.0 - 1000 * delta)) <= 1, toms


@pytest.mark.parametrize("fade_ms", [0.0, 4.0])
def test_profile_a_note_in_one_bar_of_eight_ending_on_the_bar_line_is_one_bar_and_absent(tmp_path, fade_ms):
    """A note at -6 dB against the kick in bar 4 of 8 only, ending exactly on the line into bar 5 (and with Live's 4 ms
    declick fade before it). One bar in eight is under a quarter: absent. No bar but bar 4 holds any of it, in the
    level windows or the cores."""
    d = tmp_path / "one"
    d.mkdir()
    note = level_sine(110, [-6.0 if i == 3 else None for i in range(8)])
    end, f = int((T0 + 8.0) * SR), int(fade_ms / 1000 * SR)
    if f:
        note[end - f:end] *= np.linspace(1.0, 0.0, f)
    sig = {"kick": level_sine(50, [0.0] * 8), "note": note}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    mix = write(d, "mix.wav", stereo(sum(sig.values())))
    r = profile.reference_profile(mix, paths, TEMPO, T0, SECTIONS[:1])["sections"][0]["roles"]["note"]
    assert r["absent"], r
    assert [i for i, v in enumerate(r["per_bar_db"]) if v is not None] == [3], r
    assert [i for i, v in enumerate(r["core_bar_db"]) if v is not None] == [3], r


@pytest.mark.parametrize("ref_stop,ours_stop", [(0.0, 0.005), (0.0, 0.015), (0.030, 0.0), (-0.010, 0.0)])
def test_elements_a_bass_that_stops_on_the_section_line_is_not_in_the_next_section(tmp_path, ref_stop, ours_stop):
    """The kick plays through A and B (4 bars each, 120 BPM); the bass (-6 dB against the kick) plays through A and stops
    on the A|B line in one side and 5, 15 or 30 ms before it (or 10 ms after it) in the other. B has no bass on either
    side: no gap, no one-side role, no trace; A's bass rows agree within 0.1 dB (30 ms of 8 s is 0.02 dB)."""
    def files(folder, stop):
        d = tmp_path / folder
        d.mkdir()
        kick = level_sine(50, [0.0] * 8)
        t = np.arange(len(kick)) / SR
        bass = np.where((t >= T0) & (t < T0 + 8.0 - stop), KICK_AMP * gain(-6.0) * np.sin(2 * np.pi * 110 * t), 0.0)
        return {"kick": write(d, "kick.wav", stereo(kick)), "bass": write(d, "bass.wav", stereo(bass)),
                "mix": write(d, "mix.wav", stereo(kick + bass))}

    ref, ours = files("ref", ref_stop), files("ours", ours_stop)
    secs = [{"name": "A", "start_bar": 0, "bars": 4}, {"name": "B", "start_bar": 4, "bars": 4}]
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": ref["bass"]}, TEMPO, T0, secs)
    assert prof["sections"][1]["roles"]["bass"]["absent"]
    m = {"tempo": TEMPO, "offset_s": T0, "sections": secs,
         "sources": {"master": str(ours["mix"]), "kick": str(ours["kick"]), "bass": str(ours["bass"])}}
    a, b = (s["elements"] for s in profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"])
    assert b["presence"]["bass"] == {"reference": "absent", "ours": "absent"}, b["presence"]
    assert b["gaps"] == [] and b["reference_only"] == [] and b["ours_only"] == [] and b["traces"] == [], b
    assert a["gaps"] == [] and abs(a["table"][0]["difference_db"]) <= 0.1, a["table"]


def house_files(tmp_path, folder, early):
    """124 BPM house, 0.6 s lead-in: a kick on every beat 4 ms early; toms only from bar 5 on: one long tom (tau 400 ms)
    on the downbeat of bar 5, `early` s before the line, and a four-tom fill on the last beat of bar 8."""
    d = tmp_path / folder
    d.mkdir()
    beat = 60.0 / 124.0
    total = 0.6 + 32 * beat + 1.5
    kick = 0.8 * hit_train(55, 0.08, [0.6 + b * beat - 0.004 for b in range(32)], total)
    toms = 0.5 * hit_train(100, 0.4, [0.6 + 16 * beat - early], total)
    toms = toms + 0.5 * sum(hit_train(140 - 10 * k, 0.2, [0.6 + 31 * beat + k * beat / 4], total) for k in range(4))
    return {"kick": write(d, "kick.wav", stereo(kick)), "toms": write(d, "toms.wav", stereo(toms)),
            "mix": write(d, "mix.wav", stereo(kick + toms))}


@pytest.mark.parametrize("early,deltas", [(0.007, (0.0, 0.007, 0.013)), (0.016, (0.004,))])
def test_elements_a_tom_just_before_the_line_into_the_section_with_toms_is_not_in_the_section_before(tmp_path, early, deltas):
    """The same house audio on both sides; toms are silent through section A (bars 1-4) and start on the A|B downbeat,
    7 ms (16 ms) early. Read up to 13 ms (4 ms) late, the tom's earliness plus the read offset is at most 20 ms: no
    side has toms in A, so nothing is "in ours only" or "in the reference only"; B and C have toms on both sides."""
    ref, ours = house_files(tmp_path, "ref", early), house_files(tmp_path, "ours", early)
    secs = [{"name": "A", "start_bar": 0, "bars": 4}, {"name": "B", "start_bar": 4, "bars": 2},
            {"name": "C", "start_bar": 6, "bars": 2}]
    prof = profile.reference_profile(ref["mix"], {"kick": ref["kick"], "toms": ref["toms"]}, 124.0, 0.6, secs)
    for delta in deltas:
        m = {"tempo": 124.0, "offset_s": 0.6 + delta, "sections": secs,
             "sources": {"master": str(ours["mix"]), "kick": str(ours["kick"]), "toms": str(ours["toms"])}}
        a, b, c = (s["elements"] for s in profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"])
        assert a["presence"]["toms"] == {"reference": "absent", "ours": "absent"}, (delta, a["presence"])
        for e in (a, b, c):
            assert e["ours_only"] == [] and e["reference_only"] == [] and e["gaps"] == [], (delta, e)
        assert b["presence"]["toms"] == c["presence"]["toms"] == {"reference": "present", "ours": "present"}, delta


@pytest.mark.parametrize("delta", [0.0047, 0.007, 0.013])
def test_drum_layer_identical_flams_where_the_post_roll_ends_give_our_own_hit_and_no_note(tmp_path, delta):
    """House snare flams (a hit at half level 12 ms before each full one) on beats 2 and 4 of every bar at 124 BPM, the
    same audio on both sides. The beat after the section holds a flam, 12 ms before its line: a post-roll that ended on
    that line cut the reference 12 ms into it and ours (read 4.7 to 13 ms late) further in, so the two takes split it
    differently and our own segmentation offered no complete hit. Read up to 13 ms late, ours is its own hit at the
    reference's place, with no note and no reading."""
    beat = 60.0 / 124.0
    total = 0.6 + 24 * beat + 1.5
    full = [0.6 + b * beat for b in range(24) if b % 4 in (1, 3)]
    x = 0.5 * hit_train(190, 0.08, full, total) + 0.25 * hit_train(190, 0.08, [t - 0.012 for t in full], total)
    d = {}
    for side in ("ref", "ours"):
        f = tmp_path / side
        f.mkdir()
        d[side] = {"snare": write(f, "snare.wav", stereo(x)), "mix": write(f, "mix.wav", stereo(x))}
    secs = [{"name": "A", "start_bar": 0, "bars": 4}]
    prof = profile.reference_profile(d["ref"]["mix"], {"snare": d["ref"]["snare"]}, 124.0, 0.6, secs)
    m = {"tempo": 124.0, "offset_s": 0.6 + delta, "sections": secs,
         "sources": {"master": d["ours"]["mix"], "snare": d["ours"]["snare"]}}
    sn = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"][0]
    assert sn["readings"] == [] and sn["ours_hit"] is not None and sn["notes"] == [], sn


def test_drum_layer_our_post_roll_is_longer_so_identical_audio_read_early_keeps_its_complete_hit(tmp_path):
    """One tom 1.2 s long (it falls to nothing where it ends) 1.242 s into a one-bar section at 120 BPM; it ends 8 ms
    before the reference's post-roll does (a beat after the section, 50 ms short of the beat line), just enough for it
    to read as falling 40 dB there. The same audio read 10 ms early: were our post-roll as long as the reference's, our
    take would end 2 ms before the tom does, cut off, and our own segmentation would offer no complete hit. Ours runs
    30 ms longer: our own hit, no note, no reading."""
    x = long_toms([0.5 + 1.242], 5.0)
    d = {}
    for side in ("ref", "ours"):
        f = tmp_path / side
        f.mkdir()
        d[side] = {"toms": write(f, "toms.wav", stereo(x)), "mix": write(f, "mix.wav", stereo(x))}
    secs = [{"name": "A", "start_bar": 0, "bars": 1}]
    prof = profile.reference_profile(d["ref"]["mix"], {"toms": d["ref"]["toms"]}, 120.0, 0.5, secs)
    m = {"tempo": 120.0, "offset_s": 0.5 - 0.010, "sections": secs,
         "sources": {"master": d["ours"]["mix"], "toms": d["ours"]["toms"]}}
    toms = profile.mix_report(m, reference=prof, out_dir=tmp_path / "out")["sections"][0]["drums"]["roles"][0]
    assert toms["readings"] == [] and toms["ours_hit"] is not None and toms["notes"] == [], toms

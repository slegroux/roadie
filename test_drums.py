"""roadie_analysis.drums, `roadie drums` and the study reader's drums layer, on synthetic stems at 124 BPM.

Four-on-the-floor kick, open-ish hats on the offbeats and closed hats on every other 16th, built the way
test_analysis_drumroles.py builds its stems, with the downbeat two bars into the file so that the default grid has to be
moved back by whole bars. Each test is one behaviour; the stems are made and analysed once.
"""
import json
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import roadie_drums  # noqa: E402
import study  # noqa: E402
from roadie_analysis import drums  # noqa: E402
from test_analysis_drumroles import BPM, SR, STEP, T0, at, clap_hit, noise_hit, stem, tom_hit  # noqa: E402

BARS = 24
BAR_S = 16 * STEP


def kick_hit(db=0.0, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.05) * 10 ** (db / 20)


def hat_pattern(bar):
    """Offbeats in sections A and C (bars 0-7, 16-23), the odd 16ths in B (bars 8-15): one hat sound, two patterns."""
    return [2, 6, 10, 14] if not 8 <= bar < 16 else [1, 3, 5, 7, 9, 11, 13, 15]


def make_stems(root):
    """<root>/Test/Song/Song_124bpm_stems: drums_kick (four on the floor), drums_hh (one closed hat whose pattern changes
    in section B), drums_toms (one tom per section), alignment.json with the downbeat two bars in, sections.json (3 x 8 bars)."""
    d = root / "Test" / "Song" / "Song_124bpm_stems"
    d.mkdir(parents=True)
    wr = lambda name, ev: sf.write(str(d / name), stem(ev, BARS).astype(np.float32) * (0.3 if "hh" in name else 1.0), SR)  # noqa: E731
    wr("drums_kick.flac", [(at(i * 4), kick_hit(i % 3 - 1)) for i in range(BARS * 4)])
    wr("drums_hh.flac", [(at(b * 16 + s), noise_hit(b * 16 + s, 0.005, (s % 3 - 1))) for b in range(BARS) for s in hat_pattern(b)])
    wr("drums_toms.flac", [(at(b * 16), tom_hit()) for b in (1, 9, 17)])
    (d / "alignment.json").write_text(json.dumps({"result": {"bpm": BPM, "downbeat_sec": T0 + 2 * BAR_S, "downbeat_verdict": "ok"}}))
    (d / "sections.json").write_text(json.dumps({"bpm": BPM, "method": "stems", "sections": [
        {"beat": 32 * k, "bar": 8 * k + 1, "time_sec": T0 + 8 * k * BAR_S, "duration_beats": 32, "stems": [], "label": "ABC"[k],
         "live_bar": 8.0 * k + 1} for k in range(3)]}))
    return d


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = make_stems(tmp_path_factory.mktemp("drums"))
    return d, drums.drum_roles(d, jobs=1)


def role(rec, stem_name, name):
    return next(r for r in rec["stems"][stem_name]["roles"] if r["name"] == name)


def test_a_hat_whose_pattern_changes_between_sections_is_one_role_with_a_pattern_per_section(made):
    d, rec = made
    hh = rec["stems"]["hh"]["roles"]
    assert len(hh) == 1 and hh[0]["hits"] == 8 * 4 * 2 + 8 * 8
    lines = {v["section"]: v["line"] for v in hh[0]["per_section"]}
    assert lines == {"A": "..x...x...x...x.", "B": ".x.x.x.x.x.x.x.x", "C": "..x...x...x...x."}


def test_a_sound_heard_once_in_each_section_is_one_role(made):
    d, rec = made
    toms = rec["stems"]["toms"]["roles"]
    assert len(toms) == 1 and toms[0]["hits"] == 3 and [v["hits"] for v in toms[0]["per_section"]] == [1, 1, 1]
    assert toms[0]["line"] == "x..............."


def test_the_kick_is_named_over_all_sections_and_its_grid_is_counted_from_the_downbeat(made):
    d, rec = made
    assert rec["grid"]["t0"] == pytest.approx(T0, abs=1e-3)           # the downbeat two bars in, moved back to the first bar line
    k = role(rec, "kick", "kick")
    assert (k["line"], k["hits"]) == ("x...x...x...x...", BARS * 4)
    assert [v["hits"] for v in k["per_section"]] == [32, 32, 32]


def test_drums_json_and_one_clip_per_role_are_written_with_the_inputs_for_staleness(made):
    d, rec = made
    saved = json.loads((d / "drums.json").read_text())
    assert (saved["kind"], saved["version"]) == ("drum_roles", 1)
    assert {i["name"]: i["size"] for i in saved["inputs"]} == {
        n: (d / n).stat().st_size for n in ("drums_kick.flac", "drums_hh.flac", "drums_toms.flac", "alignment.json", "sections.json")}
    for stem_name, s in saved["stems"].items():
        for r in s["roles"]:
            assert r["clip"] is None or (d / r["clip"]).is_file()
            if r["clip"]:
                audio, sr = sf.read(str(d / r["clip"]))
                assert sr == SR and len(audio) > 0
    assert saved["stems"]["kick"]["roles"][0]["clip"] == "drum_roles/kick.wav"
    assert (d / "drum_roles" / "kick.wav").is_file()
    assert saved["stems"]["kick"]["per_section"][0]["roles"] == {"kick": 32}


def test_a_process_pool_gives_the_same_record_as_one_process(made):
    d, serial = made
    pooled = drums.drum_roles(d, write=False, jobs=2)
    drop = ("made", "seconds", "jobs", "stage_seconds", "written")
    clean = lambda r: json.loads(json.dumps({k: v for k, v in r.items() if k not in drop}))  # noqa: E731
    for r in (serial, pooled):                                     # the serial record was written, the pooled one was not
        for s in r["stems"].values():
            for x in s["roles"]:
                x["clip"], x["clip_note"] = None, None
    assert clean(pooled) == clean(serial) and pooled["jobs"] == 2


def test_no_write_prints_the_table_and_writes_nothing(tmp_path, capsys):
    d = make_stems(tmp_path)
    before = sorted(p.name for p in d.iterdir())
    assert roadie_drums.main([str(d), "--no-write", "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    assert "kick" in out and "x... x... x... x..." in out
    assert sorted(p.name for p in d.iterdir()) == before


def test_the_study_reader_shows_the_drums_layer_and_flags_it(made, capsys):
    d, _ = made
    s = study.load_study(str(d.parent))
    layer = s.layers["drums"]
    assert layer.missing is None and layer.data["kind"] == "drum_roles"
    assert any(c.startswith("fresh") for c in layer.caveats)
    assert study.to_json(s)["layers"]["drums"]["data"]["stems"]["kick"]["roles"][0]["name"] == "kick"
    assert d / "drums.json" not in s.extras
    study._show(s)
    assert "x... x... x... x..." in capsys.readouterr().out


def test_a_section_where_the_stem_is_near_silent_has_no_hits_it_is_judged_against_the_whole_stem(tmp_path):
    """A kick in bars 0-7, then only a noise bed 50 dB under it: against its own loudest moment the bed would be hits."""
    d = tmp_path / "Song_stems"
    d.mkdir()
    x = stem([(at(i * 4), kick_hit()) for i in range(32)], 16)
    bed = np.random.default_rng(3).standard_normal(len(x)) * 10 ** (-50 / 20) * 0.3
    x[int((T0 + 8 * BAR_S) * SR):] += bed[int((T0 + 8 * BAR_S) * SR):]
    sf.write(str(d / "drums_kick.flac"), x.astype(np.float32), SR)
    secs = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8}]
    rec = drums.drum_roles(d, tempo=BPM, t0=T0, sections=secs, write=False, jobs=1)
    kick = rec["stems"]["kick"]
    assert kick["hits"] == 32 and [v["section"] for r in kick["roles"] for v in r["per_section"]] == ["A"]


def test_two_sections_with_the_same_label_keep_their_own_hit_counts(tmp_path):
    """sections.json labels repeat (Drop, Drop): the counts are by section place, not by name."""
    d = tmp_path / "Song_stems"
    d.mkdir()
    x = stem([(at(i * 4), kick_hit()) for i in range(16)] + [(at(128 + i * 16), kick_hit()) for i in range(8)], 16)
    sf.write(str(d / "drums_kick.flac"), x.astype(np.float32), SR)
    secs = [{"name": "Drop", "start_bar": 0, "bars": 8}, {"name": "Drop", "start_bar": 8, "bars": 8}]
    rec = drums.drum_roles(d, tempo=BPM, t0=T0, sections=secs, write=False, jobs=1)
    kick = rec["stems"]["kick"]
    assert [s["roles"] for s in kick["per_section"]] == [{"kick": 16}, {"kick": 8}]
    assert [(v["index"], v["hits"]) for v in kick["roles"][0]["per_section"]] == [(0, 16), (1, 8)]


def test_the_stem_peak_read_in_blocks_is_the_whole_stem_envelope_peak(tmp_path):
    """Same number as oneshot's on the whole stem, with a DC offset and digital silence around the audio, block sizes that split hits."""
    from roadie_analysis import drumsections, oneshot
    rng = np.random.default_rng(2)
    x = stem([(at(i * 4), kick_hit(i % 3 - 1)) for i in range(8)] + [(at(40), noise_hit(1, 0.01, 6.0))], 4)[:, None]
    x = np.hstack([x, 0.5 * x + 0.01 * rng.standard_normal(x.shape)]) * 0.5 + 0.02       # stereo, DC offset
    x[: SR // 4] = 0.0                                                                      # digital silence before the audio
    x[-SR // 2:] = 0.0
    sf.write(str(tmp_path / "s.wav"), x.astype(np.float32), SR, subtype="FLOAT")
    y, sr = sf.read(str(tmp_path / "s.wav"), dtype="float32", always_2d=True)
    want = oneshot.envelope_peak(oneshot._undc(np.asarray(y, dtype=np.float64), sr), sr)
    for block_s in (10.0, 0.37):
        assert drumsections.stem_peak(tmp_path / "s.wav", block_s) == pytest.approx(want, rel=1e-6)


class FakeRclone:
    """rclone against a local folder standing in for the pCloud library (`pcloud:<path>` is <cloud>/<path>), in place of
    `subprocess.run`: lsf, copyto, cat and deletefile, each call logged."""

    def __init__(self, cloud):
        self.cloud, self.calls = cloud, []

    def path(self, arg):
        return self.cloud / arg[len("pcloud:"):] if arg.startswith("pcloud:") else pathlib.Path(arg)

    def __call__(self, cmd, **kw):
        import shutil
        import types
        assert cmd[0] == "rclone"
        self.calls.append(cmd[1:])
        out = ""
        if cmd[1] == "lsf":
            out = "".join(p.name + "\n" for p in sorted(self.path(cmd[-1]).iterdir()) if p.is_file())
        elif cmd[1] == "copyto":
            dst = self.path(cmd[3])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(cmd[2]), dst)
        elif cmd[1] == "cat":
            out = self.path(cmd[2]).read_text()
        elif cmd[1] == "deletefile":
            self.path(cmd[2]).unlink()
        else:
            raise AssertionError(cmd)
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_a_pcloud_study_is_analysed_in_a_temp_folder_and_uploaded_by_rclone_never_through_the_mount(tmp_path, monkeypatch):
    cloud = tmp_path / "cloud"
    d = cloud / "Music" / "Studies" / "A" / "Song" / "Song_stems"
    d.mkdir(parents=True)
    sf.write(str(d / "drums_kick.flac"), stem([(at(i * 4), kick_hit()) for i in range(32)], 8).astype(np.float32), SR)
    (d / "alignment.json").write_text(json.dumps({"result": {"bpm": BPM, "downbeat_sec": T0, "downbeat_verdict": "ok"}}))
    (d / "drum_roles").mkdir()
    for name in ("old-gone.wav", "kick.wav", "mine.wav"):
        (d / "drum_roles" / name).write_bytes(b"old")
    (d / "drums.json").write_text(json.dumps({"kind": "drum_roles", "version": 1, "stems": {"kick": {"roles": [
        {"clip": "drum_roles/kick.wav"}, {"clip": "drum_roles/old-gone.wav"}]}}}))
    fake = FakeRclone(cloud)
    monkeypatch.setattr(drums.subprocess, "run", fake)
    monkeypatch.setattr(drums.keymod, "pcloud_remote", lambda p: (
        "pcloud:" + pathlib.Path(p).resolve().relative_to(cloud.resolve()).as_posix() if str(p).startswith(str(cloud)) else None))
    rec = drums.drum_roles(d, jobs=1)
    uses = [c for c in fake.calls]
    assert rec["written"].endswith("Song_stems/drums.json") and rec["written"].startswith("pcloud:")
    new = json.loads((d / "drums.json").read_text())
    assert new["stems"]["kick"]["roles"][0]["name"] == "kick" and new["version"] == 1
    lib = sorted(p.name for p in (d / "drum_roles").iterdir())
    assert lib == ["kick.wav", "mine.wav"]                         # old-gone (listed before, not now) went; mine.wav was never listed
    assert (d / "drum_roles" / "kick.wav").read_bytes() != b"old"
    verbs = [(c[0], pathlib.Path(c[-1]).name) for c in uses]
    assert verbs.index(("copyto", "kick.wav")) < verbs.index(("copyto", "drums.json")) < verbs.index(("deletefile", "old-gone.wav"))
    stems_copied = [c for c in uses if c[0] == "copyto" and c[1].endswith("drums_kick.flac")]
    assert stems_copied and not stems_copied[0][2].startswith(str(cloud))      # the audio went to a temp folder
    tmp_used = pathlib.Path(stems_copied[0][2]).parent
    assert not tmp_used.exists()                                                # and the temp folder is gone
    assert sorted(p.name for p in d.iterdir()) == ["alignment.json", "drum_roles", "drums.json", "drums_kick.flac"]


def test_roles_do_not_move_when_the_grid_moves_by_a_few_thousandths_of_a_bpm_or_a_millisecond(tmp_path):
    """The same audio under tempo +-0.005 bpm and t0 +-1 ms: the same roles, and every hit in the same role."""
    from roadie_analysis import drumstability
    d = make_stems(tmp_path)
    for f in ("drums_kick.flac", "drums_toms.flac"):
        (d / f).unlink()                                                    # one stem is enough: the hats, with three sections
    runs = drumstability.run(d, ["hh"], jobs=1, only=["tempo+0.005", "tempo-0.005", "t0+0.001", "t0-0.001"])
    assert len(runs) == 5
    base = runs.pop("base")
    for label, rec in runs.items():
        c = drumstability.compare(base, rec, "hh")
        assert c["agreement"] == 1.0 and c["named_same"], (label, c)
        assert all(r["kept"] == 1.0 for r in c["roles"]), (label, c)


def test_a_name_that_changes_with_the_on_grid_tolerance_is_flagged_unsteady():
    named = lambda n: {"roles": [{"groups": [0, 2], "name": n, "named": not n.startswith("hh 3")}]}  # noqa: E731
    assert drums._steadiness([2, 0], "hh 16th", True, (named("hh 16th"), named("hh 16th"))) is None
    assert drums._steadiness([0], "hh 3", False, (named("hh 3"), named("hh 3"))) is None
    why = drums._steadiness([0, 2], "hh 16th", True, (named("hh 3"), named("hh 16th")))
    assert "tighter tolerance (-3 ms): not named" in why and "looser" not in why
    assert "named 'hh 16th'" in drums._steadiness([0, 2], "hh 3", False, (named("hh 3"), named("hh 16th")))


def _record(*roles):
    """A hand-made record with one stem `hh`: roles are (name, named, [hit times])."""
    return {"stems": {"hh": {"roles": [{"name": n, "named": nm, "hit_times": t, "measured": len(t)} for n, nm, t in roles]}}}


def test_the_stability_check_sees_a_merge_not_only_a_split():
    from roadie_analysis import drumstability
    a, b = [1.0 + 0.1 * i for i in range(10)], [5.0 + 0.1 * i for i in range(6)]
    base = _record(("hh 1", False, a), ("hh 2", False, b))
    same = drumstability.compare(base, _record(("hh 1", False, a), ("hh 2", False, b)), "hh")
    assert same["agreement"] == 1.0 and not any(r["absorbed"] for r in same["roles"])
    merged = drumstability.compare(base, _record(("hh 1", False, a + b)), "hh")           # two roles became one
    rows = {r["name"]: r for r in merged["roles"]}
    assert rows["hh 1"]["kept"] == rows["hh 2"]["kept"] == 1.0                            # no hit left its role's match...
    assert rows["hh 2"]["absorbed"] and rows["hh 2"]["purity"] == 0.375 and not rows["hh 1"]["absorbed"]
    assert merged["agreement"] == 0.625                                                   # ...but the six hits of hh 2 do not count
    split = drumstability.compare(base, _record(("hh 1", False, a[:5]), ("hh 3", False, a[5:]), ("hh 2", False, b)), "hh")
    assert split["agreement"] == 0.688 and split["roles"][0]["split"] == 2


def test_the_stability_check_refuses_stems_that_are_not_there_and_perturbations_that_do_not_exist(tmp_path, capsys):
    from roadie_analysis import drumstability
    (tmp_path / "drums_hh.flac").write_bytes(b"")
    with pytest.raises(SystemExit, match="none of the stems snare"):
        drumstability.main([str(tmp_path), "--stems", "snare"])
    with pytest.raises(SystemExit, match="matches no perturbation"):
        drumstability.main([str(tmp_path), "--stems", "hh,snare", "--only", "tempo+0.5"])


@pytest.mark.parametrize("sr", [44100, 48000])
def test_a_section_slice_starts_on_the_same_sample_when_its_bar_line_moves_a_millisecond(tmp_path, sr):
    """The slice is anchored to a multiple of lcm(1 ms, 10 ms) of samples, so t0 and t0 + 1 ms give the same slice."""
    import math
    from roadie_analysis import drumsections
    x = np.zeros(40 * sr, dtype=np.float32)
    for k in range(60):
        x[int((0.3 + 0.5 * k) * sr):][:200] = 0.5
    sf.write(str(tmp_path / "s.wav"), x, sr, subtype="FLOAT")
    sec = {"name": "B", "start_bar": 4, "bars": 6, "first": False, "to_end": False}
    a = drumsections.section_job(tmp_path / "s.wav", 120.0, 0.5, sec)
    b = drumsections.section_job(tmp_path / "s.wav", 120.0, 0.501, sec)
    quantum = drumsections.slice_quantum(sr)
    assert 0.4 <= quantum / sr <= 0.5 and quantum % math.lcm(round(0.001 * sr), round(0.010 * sr)) == 0
    assert a["offset_s"] == b["offset_s"] and round(a["offset_s"] * sr) % quantum == 0 and a["offset_s"] > 0


def test_a_hat_detected_late_keeps_its_name_when_the_grid_moves_a_few_milliseconds(tmp_path):
    """16th hats all 19 ms late of the grid, right at the 20 ms edge: against the global grid a 3 ms move in either direction
    puts every hit off the grid; against the section's own phase the role is named in all three runs."""
    d = tmp_path / "Song_stems"
    d.mkdir()
    ev = [(at(b * 16 + s, 19.0), noise_hit(b * 16 + s, 0.005, (s % 3 - 1))) for b in range(24) for s in range(1, 16, 2)]
    sf.write(str(d / "drums_hh.flac"), (stem(ev, 24) * 0.3).astype(np.float32), SR)
    secs = [{"name": "ABC"[k], "start_bar": 8 * k, "bars": 8} for k in range(3)]
    names = {}
    for label, dt0 in (("base", 0.0), ("early", -0.003), ("late", 0.003)):
        rec = drums.drum_roles(d, tempo=BPM, t0=T0 + dt0, sections=secs, write=False, jobs=1)
        names[label] = [(r["name"], r["hits"]) for r in rec["stems"]["hh"]["roles"]]
        assert all(abs(s["grid_shift_ms"] - (19.0 - dt0 * 1000)) < 1.0 for s in rec["stems"]["hh"]["per_section"]), rec["stems"]["hh"]["per_section"]
    assert names["base"] == names["early"] == names["late"] == [("hh 16th", 24 * 8)], names



def test_a_section_phase_needs_a_cluster_and_scattered_hits_leave_the_grid_alone():
    from roadie_analysis import drumsections
    assert drumsections._phase([0.2, 0.1, -0.3, 0.4, 1.0, 0.0] + [30.0, 31.0]) == pytest.approx(0.15, abs=0.5)   # six on the grid, two strays
    assert drumsections._phase([-14, -9, -4, 0, 5, 8, 12, 38.2, 38.4]) is None                                  # no cluster of 40 %: the two strays are not a phase
    assert drumsections._phase(list(np.linspace(-55, 55, 24))) is None                                           # spread evenly: no phase
    assert drumsections._phase([0.0] * 7) is None                                                                # under 8 hits: no phase
    hits = [{"owned": True, "level_db": -20.0, "section": 0, "offset_ms": o, "onset_s": 0.5 + 0.125 * k + o / 1000.0}
            for k, o in enumerate([-14, -9, -4, 0, 5, 8, 12, 38.2, 38.4])]
    assert drumsections._rephase(hits, [{"name": "A"}], 120.0, 0.5) == [0.0]                                      # falls back to the grid itself
    assert [h["on_grid"] for h in hits] == [True] * 7 + [False] * 2                                              # the six stay on it, the two strays off


def test_a_sound_that_plays_late_stays_off_the_grid_in_every_section_whichever_is_the_majority(tmp_path):
    """A on the grid and B 30 ms late, in one stem; in the middle section B has most of the hits. B is never read as on the grid,
    and the record says how late it is."""
    d = tmp_path / "Song_stems"
    d.mkdir()
    ev = []
    for b in range(24):
        n_a, n_b = (2, 8) if 8 <= b < 16 else (8, 1)                     # hits of A and of B in the bar
        ev += [(at(b * 16 + s), noise_hit(b * 16 + s, 0.005)) for s in range(0, 16, 2)[:n_a]]
        ev += [(at(b * 16 + s, 30.0), tom_hit(f=150.0 + (s % 3))) for s in range(1, 16, 2)[:n_b]]
    sf.write(str(d / "drums_hh.flac"), (stem(ev, 24) * 0.3).astype(np.float32), SR)
    secs = [{"name": "ABC"[k], "start_bar": 8 * k, "bars": 8} for k in range(3)]
    rec = drums.drum_roles(d, tempo=BPM, t0=T0, sections=secs, write=False, jobs=1)
    roles = [r for r in rec["stems"]["hh"]["roles"] if r["measured"] >= 4]
    late = [r for r in roles if r["late_ms"] is not None and r["late_ms"] > 20]
    on = [r for r in roles if r["late_ms"] is not None and abs(r["late_ms"]) < 5]
    assert len(late) == 1 and on and 25 <= late[0]["late_ms"] <= 35, [(r["name"], r["hits"], r["late_ms"], r["off_grid"]) for r in roles]
    assert late[0]["off_grid"] == late[0]["hits"] and all(r["off_grid"] == 0 for r in on)
    assert [v["late_ms"] for v in late[0]["per_section"]] and all(25 <= v["late_ms"] <= 35 for v in late[0]["per_section"])


def test_a_hit_whose_cut_is_empty_is_silent_not_a_crash(monkeypatch):
    from roadie_analysis import drumroles
    sr = 44100
    x = np.random.default_rng(1).standard_normal(2 * sr) * 0.1
    segs = [(10000, 12000), (10040, 14000), (20000, len(x))]                 # the second hit starts 40 samples after the first: inside the 2 ms guard
    monkeypatch.setattr(drumroles.oneshot, "_segments", lambda x, sr, ref=None: segs)
    monkeypatch.setattr(drumroles, "_onset_after_tail", lambda x, sr, start: start)
    res = drumroles.hit_groups(x, 120.0, 0.0, None, sr=sr)
    assert [h["status"] for h in res["hits"]][:1] == ["silent"]


def _fake_groups(*specs):
    """hits and `built` for `_merge_alike` / `_echoes`: each spec is a list of (bar, step, extra_ms) hits of one group."""
    hits, built = [], []
    for k, spec in enumerate(specs):
        ids = []
        for bar, step, extra in spec:
            ids.append(len(hits))
            hits.append({"step": step, "bar": bar, "on_grid": True, "onset_s": 0.5 + (bar * 16 + step) * STEP + extra / 1000.0})
        built.append((ids, [], [k]))
    return hits, built


def test_groups_on_the_same_steps_that_take_turns_and_sound_alike_become_one_role():
    from roadie_analysis import drumsections
    a = [(b, s, 0.0) for b in range(0, 8) for s in (4, 12)]                  # a backbeat on steps 4 and 12 in bars 0-7...
    b = [(b, s, 0.0) for b in range(10, 18) for s in (4, 12)]                # ...and the same steps in bars 10-17: never at the same time
    hits, built = _fake_groups(a, b)
    close, far = np.array([[0, 1.6], [1.6, 0]]), np.array([[0, 1.9], [1.9, 0]])
    got, stats, info = drumsections._merge_alike(built, hits, close, [True, True])
    assert len(got) == 1 and sorted(got[0][0]) == list(range(32)) and [(m["hits"], m["bars"]) for m in info[0]] == [(16, [[0, 7]]), (16, [[10, 17]])]
    got, _, info = drumsections._merge_alike(built, hits, far, [True, True])                 # sounds different: two roles
    assert len(got) == 2 and info == [None, None]
    got, _, _ = drumsections._merge_alike(built, hits, close, [True, False])                 # one has no clean hit: the sound cannot be compared
    assert len(got) == 2
    got, _, _ = drumsections._merge_alike(built, hits, close, [True, True], [True, False])   # one is not a steady group (a piece of a continuum)
    assert len(got) == 2


def test_groups_that_play_together_or_on_other_steps_do_not_merge():
    from roadie_analysis import drumsections
    close = np.array([[0, 1.6], [1.6, 0]])
    same_time, other_steps = [(b, s, 0.0) for b in range(8) for s in (4, 12)], [(b, s, 0.0) for b in range(8) for s in (2, 10)]
    hits, built = _fake_groups(same_time, [(b, s, 5.0) for b in range(8) for s in (4, 12)])   # 5 ms apart: one event, two sounds
    assert len(drumsections._merge_alike(built, hits, close, [True, True])[0]) == 2
    hits, built = _fake_groups(same_time, [(b, s, 0.0) for b in range(10, 18) for s in (2, 10)])
    assert len(drumsections._merge_alike(built, hits, close, [True, True])[0]) == 2          # other steps


def test_a_group_that_trails_another_by_a_constant_gap_is_its_echo_and_other_gaps_are_not():
    from roadie_analysis import drumsections
    main = [(b, s, 0.0) for b in range(12) for s in (4, 12)]

    def run(gap_ms, spread=0.0):
        echo = [(b, s, gap_ms + (spread if (b + s) % 2 else -spread)) for b in range(12) for s in (4, 12)]
        hits, built = _fake_groups(main, echo)
        groups = [{"id": k, "count": len(m), "assigned": 0, "members": m, "assigned_members": []} for k, (m, _, _) in enumerate(built)]
        drumsections._echoes(groups, hits, BPM)
        return groups[1]
    e = run(152.0, 4.0)
    assert e["echo_of"] == 0 and 148 <= e["echo_ms"] <= 156 and e["echo_bars"] == [[0, 11]]
    assert run(10.0)["echo_of"] is None                       # a second hit 10 ms later is one hit detected twice
    assert run(2 * STEP * 1000, 0.0)["echo_of"] is None      # two 16ths later: the grid explains it, two parts on the grid
    assert run(152.0, 25.0)["echo_of"] is None               # the gap is not constant


def test_an_echo_in_a_stem_is_labelled_with_its_gap_and_takes_no_name_of_its_own(tmp_path):
    """A hat-like noise hit on steps 5 and 13 and a duller clap 152 ms after each: the clap group is the first one's echo."""
    d = tmp_path / "Song_stems"
    d.mkdir()
    ev = []
    for b in range(16):
        for s in (4, 12):
            ev.append((at(b * 16 + s), noise_hit(b * 16 + s, 0.02, 0.0)))
            ev.append((at(b * 16 + s, 152.0), clap_hit(b * 16 + s, -6.0)))
    sf.write(str(d / "drums_snare.flac"), (stem(ev, 16) * 0.3).astype(np.float32), SR)
    secs = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8}]
    rec = drums.drum_roles(d, tempo=BPM, t0=T0, sections=secs, write=False, jobs=1)
    roles = rec["stems"]["snare"]["roles"]
    echo = [r for r in roles if r.get("echo_of")]
    assert len(echo) == 1 and not echo[0]["named"] and 148 <= echo[0]["echo_ms"] <= 156 and echo[0]["echo_bars"] == [[0, 15]], [(r["name"], r["hits"]) for r in roles]
    assert echo[0]["name"] == "echo of " + echo[0]["echo_of"] and echo[0]["hits"] == 32
    parent = [r for r in roles if r["name"] == echo[0]["echo_of"]]
    assert len(parent) == 1 and parent[0]["hits"] == 32

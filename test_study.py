#!/usr/bin/env python3
"""Tests for study.py: temp study trees, nothing from the real Studies folder."""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
from study import load_study, to_json

HERE = os.path.dirname(os.path.abspath(__file__))


def make_study(root, artist, title, sections=True, alignment=True):
    stems = root / artist / title / ("%s_124bpm_stems" % title)
    stems.mkdir(parents=True)
    for n in ("drums_kick", "drums_hh", "bass", "synths", "vocals"):
        (stems / (n + ".flac")).write_bytes(b"")
    (stems / "bass.flac.asd").write_bytes(b"")
    if alignment:
        (stems / "alignment.json").write_text(json.dumps({"fingerprint": {}, "result": {
            "bpm": 124.0, "downbeat_verdict": "chance-level", "downbeat_confidence": 0.3,
            "detector": "heuristic"}}))
    if sections:
        (stems / "sections.json").write_text(json.dumps({"bpm": 124.0, "method": "stems", "sections": [
            {"beat": 0, "bar": 1, "duration_beats": 96, "stems": ["kick", "hats"], "label": "Intro", "live_bar": 9.0},
            {"beat": 96, "bar": 25, "duration_beats": 160, "stems": ["kick", "bass"], "label": "Drop", "live_bar": 33.0},
        ]}))
    return root / artist / title


def test_load_study_reads_grid_and_sections_and_reports_the_rest_missing(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    s = load_study("flare", tmp_path)
    assert (s.artist, s.title, s.bpm, s.bar_offset) == ("Einmusik", "Flare", 124.0, 8)
    assert sorted(s.stems) == ["bass", "hats", "kick", "synths", "vocals"]   # the .asd is not a stem
    secs = s.sections()
    assert (secs[0].label, secs[0].bar, secs[0].bars, secs[0].live_bar) == ("Intro", 1, 24, 9.0)
    assert secs[1].roles == ["kick", "bass"]
    assert s.layers["sections"].source.name == "sections.json"
    assert "labels by rule, not checked by ear" in s.layers["sections"].caveats
    assert any("chance-level" in c for c in s.layers["grid"].caveats)
    for name in ("mix", "activity", "drums"):
        assert s.layers[name].data is None and s.layers[name].missing
    assert s.mix() is None and s.activity() is None


def test_study_without_sections_json_has_empty_sections_and_a_command(tmp_path):
    make_study(tmp_path, "Bodzin", "Earth", sections=False)
    s = load_study("Earth", tmp_path)
    assert s.sections() == [] and s.layers["sections"].missing == "roadie sections <stems>"
    assert s.layers["grid"].data is not None


def test_ambiguous_title_lists_the_matches_and_unknown_is_an_error(tmp_path):
    make_study(tmp_path, "A", "Earth")
    make_study(tmp_path, "B", "Earth")
    with pytest.raises(ValueError, match=r"A/Earth.*B/Earth"):
        load_study("Earth", tmp_path)
    assert load_study("B/Earth", tmp_path).artist == "B"
    with pytest.raises(ValueError, match="no study named"):
        load_study("Nope", tmp_path)


def test_a_path_resolves_without_the_root(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare")
    assert load_study(str(folder), tmp_path / "elsewhere").title == "Flare"


def test_to_json_is_a_versioned_record_with_paths_relative_to_the_study(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    d = json.loads(json.dumps(to_json(load_study("Flare", tmp_path))))
    assert (d["kind"], d["version"]) == ("study", 1)
    assert sorted(d["layers"]) == ["activity", "drums", "grid", "mix", "sections"]
    assert d["layers"]["sections"]["source"] == "Flare_124bpm_stems/sections.json"
    assert d["layers"]["mix"]["missing"]


def test_cli_show_prints_the_record_and_json(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    env = dict(os.environ, ROADIE_STUDIES=str(tmp_path))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Flare", *a],
                                    capture_output=True, text=True, env=env)
    text = run()
    assert text.returncode == 0 and "124 bpm" in text.stdout and "Live bar 9" in text.stdout
    assert "mix       missing:" in text.stdout
    assert json.loads(run("--json").stdout)["kind"] == "study"
    bad = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Nope"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 1 and "no study named" in bad.stderr


def test_spectral_sections_json_without_stems_or_method_loads(tmp_path):
    folder = make_study(tmp_path, "A", "Spec", sections=False)
    (next(folder.glob("*_stems")) / "sections.json").write_text(json.dumps({"bpm": 124.0, "sections": [
        {"beat": 0, "bar": 1, "duration_beats": 64, "label": "Intro", "level": "A quiet"}]}))
    s = load_study("Spec", tmp_path)
    assert s.sections()[0].roles == [] and s.sections()[0].bars == 16
    assert s.layers["sections"].made_by == "sections (spectral method)"
    assert s.bar_offset is None and any("no live_bar" in c for c in s.layers["sections"].caveats)


def test_offset_and_bpm_are_unknown_not_zero_when_nothing_gives_them(tmp_path):
    folder = make_study(tmp_path, "Bodzin", "Earth", sections=False, alignment=False)
    stems = next(folder.glob("*_stems"))
    (stems / "alignment.json").write_text(json.dumps({"result": {"downbeat_verdict": "ok"}}))
    s = load_study("Earth", tmp_path)
    assert s.bar_offset is None and s.bpm is None
    d = to_json(s)
    assert d["bar_offset"] is None and d["bpm"] is None
    env = dict(os.environ, ROADIE_STUDIES=str(tmp_path))
    out = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Earth"],
                         capture_output=True, text=True, env=env).stdout
    assert "unknown bpm, 4/4, Live bar = unknown" in out


def test_disagreeing_live_bar_offsets_are_unknown_with_a_caveat(tmp_path):
    folder = make_study(tmp_path, "A", "Odd")
    path = next(folder.glob("*_stems")) / "sections.json"
    d = json.loads(path.read_text())
    d["sections"][1]["live_bar"] = 40.0
    path.write_text(json.dumps(d))
    s = load_study("Odd", tmp_path)
    assert s.bar_offset is None and any("disagree" in c for c in s.layers["sections"].caveats)


def test_unreadable_file_is_shown_with_its_source_never_as_missing(tmp_path):
    folder = make_study(tmp_path, "A", "Bad")
    stems = next(folder.glob("*_stems"))
    (stems / "sections.json").write_text("{not json")
    (stems / "alignment.json").write_text("[]")
    s = load_study("Bad", tmp_path)
    for name in ("sections", "grid"):
        layer = s.layers[name]
        assert layer.source and layer.missing is None and "could not be read" in layer.caveats[0]
    env = dict(os.environ, ROADIE_STUDIES=str(tmp_path))
    r = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Bad"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "sections.json" in r.stdout and "roadie sections" not in r.stdout


def test_find_studies_lists_every_folder_with_analysed_only_when_it_has_stems(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    (tmp_path / "Anyma" / "Anyma Project").mkdir(parents=True)             # an Ableton project, no stems
    (tmp_path / ".omc" / "state").mkdir(parents=True)                       # tool folder, not a study
    marked = tmp_path / "Rodriguez" / "Synth 1"
    (marked / "truth").mkdir(parents=True)
    (marked / "truth" / "bass.flac").write_bytes(b"")
    (marked / "study.json").write_text(json.dumps({"kind": "study_marker", "stems": "truth", "bpm": 124, "t0": 0.0}))
    refs = {(r.artist, r.title): r.analysed for r in study.find_studies(tmp_path)}
    assert refs == {("Einmusik", "Flare"): True, ("Anyma", "Anyma Project"): False,
                    ("Rodriguez", "Synth 1"): True}
    s = load_study("Synth 1", tmp_path)
    assert s.bpm == 124.0 and list(s.stems) == ["bass"] and s.layers["sections"].missing
    # a marker naming a folder that is not there falls back to *_stems, or to not analysed
    (marked / "study.json").write_text(json.dumps({"kind": "study_marker", "stems": "../x"}))
    assert not [r for r in study.find_studies(tmp_path) if r.title == "Synth 1"][0].analysed


def test_cli_list_shows_layers_and_the_not_analysed(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare")
    make_study(tmp_path, "Bodzin", "Earth", sections=False)
    (tmp_path / "Anyma" / "Anyma Project").mkdir(parents=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "list"], capture_output=True,
                       text=True, env=dict(os.environ, ROADIE_STUDIES=str(tmp_path)))
    out = r.stdout
    assert r.returncode == 0 and "analysed (2)" in out and "not analysed (1)" in out
    earth = [l for l in out.splitlines() if "Bodzin/Earth" in l][0]
    assert earth.split("missing:")[1].split()[0] == "sections"
    assert "Anyma/Anyma Project" in out


def add_profile(folder, version=6, name="x.profile.json"):
    """A reference profile in <folder>/analysis whose inputs are the study's own stems."""
    stems = next(folder.glob("*_stems"))
    for f in stems.glob("*.flac"):
        f.write_bytes(b"x" * 10)
    ins = [{"path": str(f), "mtime": f.stat().st_mtime, "size": f.stat().st_size}
           for f in sorted(stems.glob("*.flac"))]
    prof = {"kind": "reference_profile", "version": version, "inputs": {"mix": ins[:1], "stems": {"all": ins[1:]}},
            "sections": [{"name": "s%d" % i} for i in range(3)]}
    (folder / "analysis").mkdir(exist_ok=True)
    path = folder / "analysis" / name
    path.write_text(json.dumps(prof))
    return path, prof


def test_mix_layer_fresh_then_stale_names_the_changed_stem(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare")
    add_profile(folder)
    s = load_study("Flare", tmp_path)
    assert s.layers["mix"].made_by == "reference_profile v6" and len(s.mix()) == 3
    assert s.layers["mix"].caveats[-1].startswith("fresh: 5 inputs")
    (next(folder.glob("*_stems")) / "bass.flac").write_bytes(b"longer than before")
    stale = load_study("Flare", tmp_path).layers["mix"].caveats[-1]
    assert stale.startswith("stale: bass.flac (size") and "path, mtime and size" in stale


def test_mix_layer_checks_by_name_and_size_when_the_recorded_paths_are_gone(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare")
    path, prof = add_profile(folder)
    for items in [prof["inputs"]["mix"], *prof["inputs"]["stems"].values()]:
        for i in items:
            i["path"] = "/elsewhere/" + os.path.basename(i["path"])
    path.write_text(json.dumps(prof))
    assert "fresh" in load_study("Flare", tmp_path).layers["mix"].caveats[-1]
    assert "file name and size" in load_study("Flare", tmp_path).layers["mix"].caveats[-1]


def test_newer_profile_version_is_not_interpreted(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare")
    add_profile(folder, version=7)
    s = load_study("Flare", tmp_path)
    assert s.mix() is None and s.layers["mix"].source and "newer than this reader" in s.layers["mix"].caveats[0]


def test_other_json_files_are_extras_but_the_study_own_files_are_not(tmp_path):
    folder = make_study(tmp_path, "Bodzin", "Earth")
    (folder / "rebuild").mkdir()
    (folder / "rebuild" / "score.json").write_text("{}")
    (folder / "rebuild" / "arrangement.json").write_text("not even json")
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "x.json").write_text("{}")
    s = load_study("Earth", tmp_path)
    assert [str(p.relative_to(folder)) for p in s.extras] == ["rebuild/arrangement.json", "rebuild/score.json"]
    assert s.layers["mix"].missing


def test_profile_with_no_inputs_is_not_checked_never_fresh(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")
    path, prof = add_profile(folder)
    prof["inputs"] = {"mix": [], "stems": {}}
    path.write_text(json.dumps(prof))
    assert load_study("Flare", tmp_path).layers["mix"].caveats[-1].startswith("not checked")


def test_a_fresh_profile_beats_a_newer_stale_one_and_the_caveat_names_the_other(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")
    fresh, _ = add_profile(folder)
    stale, prof = add_profile(folder, name="stale.profile.json")
    prof["inputs"]["mix"][0]["size"] += 5
    stale.write_text(json.dumps(prof))
    os.utime(stale, (fresh.stat().st_mtime + 100,) * 2)                   # the stale one is newer
    s = load_study("Flare", tmp_path)
    assert s.layers["mix"].source.name == "x.profile.json"
    assert "analysis/stale.profile.json" in s.layers["mix"].caveats[0]
    assert stale in s.extras


def test_fallback_by_name_with_two_files_of_that_name_is_ambiguous(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")
    path, prof = add_profile(folder)
    for i in prof["inputs"]["mix"]:
        i["path"] = "/elsewhere/" + os.path.basename(i["path"])
    path.write_text(json.dumps(prof))
    name = os.path.basename(prof["inputs"]["mix"][0]["path"])
    (folder / "copy").mkdir()
    (folder / "copy" / name).write_bytes(b"x" * 10)
    line = load_study("Flare", tmp_path).layers["mix"].caveats[-1]
    assert line.startswith("ambiguous") and name in line


def test_malformed_profile_inputs_do_not_take_the_study_down(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")
    path, prof = add_profile(folder)
    prof["inputs"]["stems"] = None
    path.write_text(json.dumps(prof))
    s = load_study("Flare", tmp_path)
    assert s.mix() is None and "could not read" in s.layers["mix"].caveats[-1]
    assert s.sections()


def test_a_unique_title_substring_resolves_when_there_is_no_exact_match(tmp_path):
    make_study(tmp_path, "Rodriguez", "Synthwave 1 benchmark")
    make_study(tmp_path, "Bodzin", "Earth")
    make_study(tmp_path, "Other", "Earthbound")
    assert load_study("synthwave", tmp_path).title == "Synthwave 1 benchmark"
    assert load_study("Earth", tmp_path).artist == "Bodzin"               # exact still wins
    with pytest.raises(ValueError, match="ambiguous"):
        load_study("th", tmp_path)


def two_roots(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_study(a, "Einmusik", "Flare")
    make_study(b, "Einmusik", "Flare")
    make_study(b, "Anyma", "Eternity")
    (b / "Anyma" / "Eternity" / "Eternity_124bpm_stems" / "alignment.json").write_text(
        json.dumps({"result": {"bpm": 125.0}}))
    return [study.Root("local", a), study.Root("pcloud", b), study.Root("gone", tmp_path / "not-mounted")]


def test_parse_roots_reads_labels_paths_and_defaults(monkeypatch, tmp_path):
    got = study.parse_roots(os.pathsep.join(["pc=/x/y", "/plain"]))
    assert [(r.label, str(r.path)) for r in got] == [("pc", "/x/y"), ("root2", "/plain")]
    monkeypatch.setenv("HOME", str(tmp_path))                     # no local folder: the library alone
    assert [r.label for r in study.parse_roots(None)] == ["pcloud"]
    (tmp_path / "Projects" / "Music" / "slgMusProd" / "Studies").mkdir(parents=True)
    assert [r.label for r in study.parse_roots(None)] == ["local", "pcloud"]
    assert [r.label for r in study.parse_roots("a=/x")] == ["a"]    # the env override is taken as given


def test_a_title_in_two_roots_is_an_error_naming_both_unless_the_root_is_given(tmp_path):
    roots = two_roots(tmp_path)
    with pytest.raises(ValueError, match=r"local:Einmusik/Flare.*pcloud:Einmusik/Flare.*local:Flare"):
        load_study("Flare", roots)
    s = load_study("pcloud:Flare", roots)
    assert s.source == "pcloud" and str(s.root).startswith(str(tmp_path / "b"))
    assert to_json(s)["root"] == "pcloud" and load_study("LOCAL:flare", roots).source == "local"
    assert load_study("Eternity", roots).bpm == 125.0 and load_study("Eternity", roots).source == "pcloud"
    with pytest.raises(ValueError, match="no study named"):
        load_study("local:Eternity", roots)


def test_list_groups_by_root_and_a_missing_root_is_a_line_not_an_error(tmp_path):
    roots = two_roots(tmp_path)
    data = study.list_to_json(roots)
    by_label = {r["label"]: r for r in data["roots"]}
    assert (by_label["local"]["analysed"], by_label["pcloud"]["analysed"]) == (1, 2)
    assert by_label["gone"]["available"] is False
    assert {(s["root"], s["title"]) for s in data["studies"]} == {
        ("local", "Flare"), ("pcloud", "Flare"), ("pcloud", "Eternity")}
    spec = os.pathsep.join("%s=%s" % (r.label, r.path) for r in roots)
    out = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "list"], capture_output=True,
                         text=True, env=dict(os.environ, ROADIE_STUDIES=spec))
    assert out.returncode == 0 and "gone  " in out.stdout and "not mounted or missing, skipped" in out.stdout
    assert out.stdout.index("local  ") < out.stdout.index("pcloud  ")


def test_list_only_lists_directories_it_never_opens_the_json(tmp_path):
    root = tmp_path / "r"
    folder = make_study(root, "A", "Flare")
    stems = next(folder.glob("*_stems"))
    (stems / "sections.json").write_text("{broken")                      # listing does not parse it
    (folder / "analysis").mkdir()
    (folder / "analysis" / "x.profile.json").write_text("{broken")
    flags = study.list_to_json([study.Root("l", root)])["studies"][0]["layers"]
    assert flags["sections"] and flags["mix"] and flags["grid"] and not flags["activity"]
    (stems / "sections.json").write_text('{"method": "stems", "bar_activity": {broken')   # the key, not the JSON
    flags = study.list_to_json([study.Root("l", root)])["studies"][0]["layers"]
    assert flags["activity"] and flags["sections"]


def test_names_match_across_unicode_forms(tmp_path):
    import unicodedata
    nfd = unicodedata.normalize("NFD", "La clé des champs")
    assert nfd != "La clé des champs"
    make_study(tmp_path, unicodedata.normalize("NFD", "Ben Böhmer"), nfd)
    for typed in ("La clé des champs", "LA CLÉ", "Ben Böhmer/La clé des champs"):
        assert load_study(typed, tmp_path).title == nfd
    two = [study.Root("a", tmp_path)]
    assert load_study("a:la clé des champs", two).source == "a"


def test_a_recorded_path_outside_the_study_is_not_trusted(tmp_path):
    other = make_study(tmp_path, "B", "Other")
    mine = make_study(tmp_path, "A", "Flare")
    _, prof = add_profile(other, name="y.profile.json")            # inputs: the OTHER study's files
    path, _ = add_profile(mine)
    (next(mine.glob("*_stems")) / "bass.flac").write_bytes(b"x" * 99)  # mine differs in size
    path.write_text(json.dumps(prof))
    line = load_study("Flare", tmp_path).layers["mix"].caveats[-1]
    assert not line.startswith("fresh") and "bass.flac" in line and "not files of this study" in line


def test_an_unlistable_root_or_artist_does_not_break_the_other_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_study(a, "X", "Good")
    make_study(b, "Locked", "Hidden")
    make_study(b, "Open", "Fine")
    os.chmod(b / "Locked", 0)
    os.chmod(a, 0o755)
    roots = [study.Root("a", a), study.Root("b", b)]
    try:
        assert load_study("Good", roots).source == "a" and load_study("Fine", roots).source == "b"
        data = study.list_to_json(roots)
        assert any("Locked" in e for e in data["roots"][1]["errors"]) and data["roots"][1]["available"]
        os.chmod(b / "Locked", 0o755)
        os.chmod(b, 0)
        data = study.list_to_json(roots)
        assert data["roots"][1]["available"] is False and data["roots"][0]["available"]
        assert load_study("Good", roots).title == "Good"
    finally:
        os.chmod(b, 0o755)
        os.chmod(b / "Locked", 0o755)


def test_the_record_carries_the_path_and_empty_names_are_rejected(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")
    s = load_study("Flare", tmp_path)
    assert to_json(s)["path"] == str(folder.absolute())
    data = study.list_to_json(tmp_path)
    assert data["studies"][0]["path"] == str(folder.absolute()) and "file names" in data["note"]
    for bad in ("", "  ", "local:"):
        with pytest.raises(ValueError):
            load_study(bad, [study.Root("local", tmp_path)])


# --- the activity layer: bar_activity in sections.json ------------------------

def _with_activity(tmp_path, activity):
    folder = make_study(tmp_path, "Rodriguez Jr.", "Synthwave")
    stems = next(folder.glob("*_stems"))
    secs = json.loads((stems / "sections.json").read_text())
    secs["bar_activity"] = activity
    (stems / "sections.json").write_text(json.dumps(secs))
    return load_study("Synthwave", tmp_path)


def test_activity_comes_back_by_role_from_bar_activity(tmp_path):
    """Per stem in the file, per role out: the hats role is drums_hh or drums_ride (either plays),
    `returns` is in no role and is named in the layer's caveats. First bar and flags by hand."""
    s = _with_activity(tmp_path, {"first_bar": 1, "bars": 6, "stems": {
        "drums_kick": "001111", "bass": "000011", "drums_hh": "011000", "drums_ride": "000100",
        "returns": "111111"}})
    layer = s.layers["activity"]
    assert layer.source.name == "sections.json" and layer.missing is None
    assert any("returns" in c for c in layer.caveats)
    assert s.activity_first_bar() == 1
    assert s.activity() == {"kick": [False, False, True, True, True, True],
                            "bass": [False, False, False, False, True, True],
                            "hats": [False, True, True, True, False, False]}


def test_activity_stays_absent_without_bar_activity_and_unreadable_when_malformed(tmp_path):
    assert load_study("Flare", make_study(tmp_path / "a", "E", "Flare").parent.parent).activity() is None
    s = _with_activity(tmp_path / "b", {"first_bar": 1, "bars": 6, "stems": {"bass": "0101"}})
    assert s.activity() is None and s.layers["activity"].caveats == [
        "bar_activity exists but could not be read"]
    assert s.layers["activity"].missing is None        # the file is there: no command offered over it


def test_a_bracket_suffix_is_not_part_of_the_title(tmp_path):
    folder = make_study(tmp_path, "Einmusik", "Flare [124 Am 8A]")
    assert [(r.artist, r.title) for r in study.find_studies(tmp_path)] == [("Einmusik", "Flare")]
    for name in ("Flare", "flar", "Einmusik/Flare", "Flare [124 Am 8A]"):
        s = load_study(name, tmp_path)
        assert (s.title, s.root) == ("Flare", folder)
    assert load_study("local:Flare", [study.Root("local", tmp_path)]).title == "Flare"
    assert load_study(str(folder), tmp_path).title == "Flare"


def tag_tree(tmp_path):
    return make_study(tmp_path, "Einmusik", "Flare")


def test_a_marker_without_the_new_fields_still_loads_and_the_record_carries_them(tmp_path):
    folder = tag_tree(tmp_path)
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "t0": 0.0}))
    s = load_study("Flare", tmp_path)
    assert (s.key, s.camelot, s.key_source, s.key_confidence, s.key_url) == (None, None, None, None, None)
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "listed_bpm": 124, "key": "Am",
                                                   "key_source": "online", "key_url": "https://x/y"}))
    d = to_json(load_study("Flare", tmp_path))
    assert (d["listed_bpm"], d["key"], d["camelot"], d["key_source"], d["key_url"]) == (124.0, "Am", "8A", "online", "https://x/y")


def test_tag_writes_only_its_fields_and_keeps_the_others(tmp_path):
    folder = tag_tree(tmp_path)
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "stems": "Flare_124bpm_stems", "t0": 0.5}))
    study.tag_study(folder, key="Am", bpm=124)
    on_disk = json.loads((folder / "study.json").read_text())
    assert on_disk == {"kind": "study_marker", "stems": "Flare_124bpm_stems", "t0": 0.5, "listed_bpm": 124,
                       "key": "A minor", "camelot": "8A", "key_source": "tagged"}
    study.tag_study(folder, bpm=124.5)                            # a bpm alone leaves the key
    s = load_study("Flare", tmp_path)
    assert (s.listed_bpm, s.key, s.camelot, s.key_source) == (124.5, "A minor", "8A", "tagged")
    assert s.bpm == 124.0                                         # the grid bpm is the measured one, untouched
    assert json.loads((folder / "study.json").read_text())["t0"] == 0.5


def test_a_stronger_key_source_wins_and_by_ear_beats_all(tmp_path):
    folder = tag_tree(tmp_path)
    study.tag_study(folder, key="E minor", source="tagged")
    with pytest.raises(ValueError, match="outranks"):
        study.tag_study(folder, key="Am", source="detected", confidence=0.2)
    with pytest.raises(ValueError, match="outranks"):
        study.tag_study(folder, key="Am", source="online", url="https://x/y")
    assert load_study("Flare", tmp_path).key == "E minor"
    study.tag_study(folder, key="Bm", source="by ear")
    with pytest.raises(ValueError, match="outranks"):
        study.tag_study(folder, key="Em", source="tagged")
    s = load_study("Flare", tmp_path)
    assert (s.key, s.camelot, s.key_source) == ("B minor", "10A", "by ear")
    # a weaker key can still be written when nothing stronger is on record, and carries its confidence
    other = make_study(tmp_path, "Einmusik", "Limba")
    study.tag_study(other, key="C", source="detected", confidence=0.12)
    assert json.loads((other / "study.json").read_text())["key_confidence"] == 0.12
    study.tag_study(other, key="G", source="online", url="https://x/y")      # stronger replaces, stale confidence goes
    assert "key_confidence" not in json.loads((other / "study.json").read_text())


def test_tag_refuses_to_overwrite_a_file_that_is_not_a_marker_or_a_bad_key(tmp_path):
    folder = tag_tree(tmp_path)
    (folder / "study.json").write_text('{"something": "else"}')
    with pytest.raises(ValueError, match="not a study marker"):
        study.tag_study(folder, key="Am")
    (folder / "study.json").unlink()
    with pytest.raises(ValueError, match="not a key"):
        study.tag_study(folder, key="Hm")
    assert not (folder / "study.json").exists()


def test_cli_tag_then_list_and_show_print_bpm_key_and_camelot(tmp_path):
    make_study(tmp_path, "Einmusik", "Flare [124 Am 8A]")
    make_study(tmp_path, "Bodzin", "Earth")
    env = dict(os.environ, ROADIE_STUDIES=str(tmp_path))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "study.py"), *a], capture_output=True,
                                    text=True, env=env)
    assert run("tag", "Flare", "--key", "Am", "--bpm", "124", "--by-ear").returncode == 0
    assert run("tag", "Earth", "--key", "Em", "--source", "detected", "--confidence", "0.1").returncode == 0
    out = run("list").stdout
    flare = [l for l in out.splitlines() if "Einmusik/Flare" in l][0].split("has:")[0].split()
    assert flare[1:4] == ["124", "Am", "8A"] and flare[4:] == ["by", "ear"]
    earth = [l for l in out.splitlines() if "Bodzin/Earth" in l][0].split("has:")[0].split()
    assert earth[1:] == ["124", "Em?", "9A?", "detected"]                      # a detected key is marked as an estimate
    assert "key: A minor (Am, 8A), source by ear" in run("show", "Flare").stdout
    assert "key: E minor? (Em?, 9A?), source detected, confidence 0.1" in run("show", "Earth").stdout
    weaker = run("tag", "Flare", "--key", "C", "--source", "online")
    assert weaker.returncode == 1 and "outranks" in weaker.stderr
    assert run("tag", "Flare").returncode == 1                    # nothing to tag
    assert run("tag", "Flare", "--key", "Am", "--by-ear", "--source", "online").returncode == 1


class FakePCloud:
    """rclone against a dict of remote files, standing in for `subprocess.run`: lsf, cat and copyto."""

    def __init__(self, files):
        self.files, self.calls, self.drop_writes = dict(files), [], False

    def __call__(self, cmd, **kw):
        import types
        assert cmd[0] == "rclone"
        self.calls.append(cmd[1:])
        if cmd[1] == "lsf":
            parent = cmd[-1].rstrip("/") + "/"
            out = "".join(k[len(parent):] + "\n" for k in self.files if k.startswith(parent) and "/" not in k[len(parent):])
        elif cmd[1] == "cat":
            out = self.files[cmd[2]]
        else:
            if not self.drop_writes:
                self.files[cmd[3]] = open(cmd[2]).read()
            out = ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_a_pcloud_study_is_read_and_written_by_rclone_never_through_the_mount(tmp_path, monkeypatch):
    folder = tag_tree(tmp_path)                     # the mount view of the study: no study.json at all (stale)
    remote = "pcloud:Music/Studies/Einmusik/Flare/study.json"
    held = json.dumps({"kind": "study_marker", "stems": "truth", "t0": 0.5, "listed_bpm": 124}) + "\n"
    fake = FakePCloud({remote: held})
    monkeypatch.setattr(study.keymod, "pcloud_remote", lambda p: remote)
    monkeypatch.setattr(study.subprocess, "run", fake)
    study.tag_study(folder, key="Am")
    kept = json.loads(fake.files[remote])
    assert kept["stems"] == "truth" and kept["t0"] == 0.5 and kept["listed_bpm"] == 124 and kept["key"] == "A minor"
    assert not (folder / "study.json").exists()                 # nothing written or read through the mount
    # a second tag straight after sees the first (read through the API), and a write that does not stick fails
    study.tag_study(folder, bpm=125)
    assert json.loads(fake.files[remote])["key"] == "A minor"
    fake.files[remote] = "{}"
    with pytest.raises(ValueError, match="not a study marker"):
        study.tag_study(folder, bpm=125)
    fake.files[remote], fake.drop_writes = held, True
    with pytest.raises(OSError, match="read back"):
        study.tag_study(folder, bpm=126)


def test_only_the_bpm_key_camelot_suffix_is_stripped_from_a_folder_name(tmp_path):
    for name, title in (("Flare [124 Am 8A]", "Flare"), ("Track [Live]", "Track [Live]"),
                        ("Track [Live] [124 F#m 11A?]", "Track [Live]"), ("Mix [123.5 Bb 6B]", "Mix"),
                        ("Cut [2024]", "Cut [2024]"), ("Set [122]", "Set"), ("Set [121.5] ", "Set"), ("Op [7]", "Op [7]"), ("Breathing (Extended Mix)", "Breathing (Extended Mix)")):
        assert study._strip_suffix(name) == title
    make_study(tmp_path, "A", "Track [Live]")
    make_study(tmp_path, "A", "Track")
    assert load_study("Track [Live]", tmp_path).root.name == "Track [Live]"
    assert load_study("Track", tmp_path).root.name == "Track"


def test_the_listed_bpm_shows_beside_the_measured_one_and_never_replaces_it(tmp_path):
    folder = make_study(tmp_path, "Matador", "Eternity")             # measured 124.0 (alignment.json)
    study.tag_study(folder, bpm=92, bpm_url="https://x/y")
    s = load_study("Eternity", tmp_path)
    assert (s.bpm, s.listed_bpm, s.bpm_url) == (124.0, 92, "https://x/y")
    r = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Eternity"], capture_output=True,
                       text=True, env=dict(os.environ, ROADIE_STUDIES=str(tmp_path)))
    assert "124 bpm (listed 92)" in r.stdout
    row = study.list_to_json(tmp_path)["studies"][0]
    assert (row["bpm"], row["listed_bpm"]) == (124.0, 92)
    out = subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "list"], capture_output=True, text=True,
                         env=dict(os.environ, ROADIE_STUDIES=str(tmp_path))).stdout
    assert "124 (92)" in out
    study.tag_study(folder, bpm=124.4)                               # within 1 bpm: nothing extra to show
    assert "(listed" not in subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "show", "Eternity"],
                                           capture_output=True, text=True,
                                           env=dict(os.environ, ROADIE_STUDIES=str(tmp_path))).stdout


def test_list_and_show_take_the_bpm_from_the_same_file(tmp_path):
    folder = make_study(tmp_path, "A", "Flare")                      # alignment.json 124.0, sections.json 124.0
    stems = next(folder.glob("*_stems"))
    secs = json.loads((stems / "sections.json").read_text())
    (stems / "sections.json").write_text(json.dumps(dict(secs, bpm=124.003)))
    assert study.list_to_json(tmp_path)["studies"][0]["bpm"] == load_study("Flare", tmp_path).bpm == 124.0
    (stems / "alignment.json").unlink()                              # no grid: both fall back to sections.json
    assert study.list_to_json(tmp_path)["studies"][0]["bpm"] == load_study("Flare", tmp_path).bpm == 124.003


def test_a_key_with_no_recorded_source_counts_as_tagged_and_bad_numbers_are_refused(tmp_path):
    folder = tag_tree(tmp_path)
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "C"}))
    with pytest.raises(ValueError, match="outranks"):
        study.tag_study(folder, key="Am", source="detected")
    assert json.loads((folder / "study.json").read_text()) == {"kind": "study_marker", "key": "C"}
    study.tag_study(folder, key="Am", source="by ear")               # a stronger one still replaces it
    for bad in (float("nan"), float("inf"), -1, 0, True, "124"):
        with pytest.raises(ValueError, match="bpm"):
            study.tag_study(folder, bpm=bad)
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="confidence"):
            study.tag_study(folder, key="Bm", source="by ear", confidence=bad)
    assert "NaN" not in (folder / "study.json").read_text()


def test_a_nested_benchmark_with_a_marker_is_neither_a_study_nor_a_second_stems_folder(tmp_path):
    folder = make_study(tmp_path, "Rodriguez Jr.", "Synthwave #1 [124 Cm 5A]")
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "C minor", "key_source": "online"}))
    bench = folder / "benchmark"
    (bench / "truth").mkdir(parents=True)
    (bench / "truth" / "bass.flac").write_bytes(b"")
    (bench / "bsroformer_stems").mkdir()                       # a *_stems folder one level down: not the study's
    (bench / "bsroformer_stems" / "bass.flac").write_bytes(b"")
    (bench / "study.json").write_text(json.dumps({"kind": "study_marker", "stems": "truth", "bpm": 124}))
    (bench / "scores.json").write_text("{}")
    assert [(r.artist, r.title, r.analysed) for r in study.find_studies(tmp_path)] == [("Rodriguez Jr.", "Synthwave #1", True)]
    s = load_study("Synthwave #1", tmp_path)
    assert next(iter(s.stems["bass"])).parent.name == "Synthwave #1 [124 Cm 5A]_124bpm_stems"
    assert s.key == "C minor" and s.bpm == 124.0
    assert s.extras == []                                        # the nested study's files are not this study's extras
    (bench / "analysis").mkdir()                                 # a profile cut on the benchmark's own audio and sections
    (bench / "analysis" / "mix.profile.json").write_text(json.dumps(
        {"kind": "reference_profile", "version": 6, "inputs": {"mix": [], "stems": {}}, "sections": [{"name": "x"}]}))
    s = load_study("Synthwave #1", tmp_path)
    assert s.layers["mix"].source is None and s.layers["mix"].missing and s.extras == []
    assert study.list_to_json(tmp_path)["studies"][0]["layers"]["mix"] is False          # list and show agree
    nested = load_study(str(bench), tmp_path)                                             # still readable by its own path
    assert nested.layers["mix"].source.name == "mix.profile.json" and list(nested.stems) == ["bass"]


def test_tagging_the_same_key_again_keeps_its_note_url_and_confidence(tmp_path):
    folder = tag_tree(tmp_path)
    study.tag_study(folder, key="Am", source="online", url="https://x/y", note="old tag: Em", bpm=124)
    study.tag_study(folder, key="A minor", source="online", url="https://x/other", note="another", confidence=0.3)
    kept = json.loads((folder / "study.json").read_text())
    assert (kept["key_url"], kept["key_note"], kept["listed_bpm"]) == ("https://x/y", "old tag: Em", 124)
    assert kept["key_confidence"] == 0.3                        # a field that was not there yet is filled in
    study.tag_study(folder, key="Bm", source="online", url="https://x/new")                  # another key: the old note goes
    assert "key_note" not in json.loads((folder / "study.json").read_text())

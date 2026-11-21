#!/usr/bin/env python3
"""Tests for studyrename.py (`roadie study rename --dry-run`): temp trees, nothing is renamed."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
import studyrename
from test_study import add_profile, load_study, make_study

HERE = os.path.dirname(os.path.abspath(__file__))


def snapshot(root):
    return sorted((str(p), p.stat().st_size) for p in root.rglob("*"))


def tree(tmp_path):
    lib = tmp_path / "lib"
    flare = make_study(lib, "Einmusik", "Flare")
    study.tag_study(flare, key="Am", bpm=124, source="online", url="https://x/y")
    study.tag_study(make_study(lib, "Einmusik", "Limba"), key="C", source="detected", confidence=0.1)
    make_study(lib, "Einmusik", "Nosara")                                  # no key
    study.tag_study(make_study(lib, "Einmusik", "Back"), key="G", bpm=92)    # listed far from measured 124
    make_study(lib, "Einmusik", "Back [125 Cm 5A]")                          # the new name of "Back" already exists
    (lib / "Anyma" / "Anyma Project").mkdir(parents=True)
    return lib


def test_the_dry_run_names_every_study_from_its_record_and_changes_nothing(tmp_path):
    lib = tree(tmp_path)
    add_profile(lib / "Einmusik" / "Flare")                                 # inputs recorded under this folder
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "x.json").write_text(json.dumps({"out_dir": str(lib / "Einmusik" / "Flare")}))
    before = snapshot(tmp_path)
    roots = [study.Root("pcloud", lib)]
    plan = studyrename.plan_rename([r for r in study.find_studies(roots) if r.analysed], roots, tmp_path / "bench")
    new = {e["old"]: e["new"] for e in plan["entries"]}
    assert new["Flare"] == "Flare [124 Am 8A]" and new["Limba"] == "Limba [124 C? 8B?]"
    assert new["Nosara"] == "Nosara [124]"
    assert new["Back"] == "Back [124 G 9B]"                                   # measured 124 beats a listing of 92
    flare = [e for e in plan["entries"] if e["old"] == "Flare"][0]["recorded"]
    assert any(l.startswith("analysis/x.profile.json: 5 recorded input paths") for l in flare)
    assert any(l.startswith("bench/x.json") for l in flare)
    assert plan["collisions"] == []                                           # "Back [125 Cm 5A]" is not "Back [124 G 9B]"
    assert plan["merge"] is None
    assert snapshot(tmp_path) == before


def test_collisions_are_listed_and_cli_refuses_write_and_a_missing_dry_run(tmp_path):
    lib = tmp_path / "lib"
    for t in ("Flare", "Flare [124 Am 8A]"):
        study.tag_study(make_study(lib, "A", t), key="Am", bpm=124, source="online")
    roots = [study.Root("pcloud", lib)]
    plan = studyrename.plan_rename([r for r in study.find_studies(roots) if r.analysed], roots, tmp_path / "none")
    assert any("a folder of that name exists (Flare [124 Am 8A])" in c for c in plan["collisions"])
    assert any("all become Flare [124 Am 8A]" in c for c in plan["collisions"])
    env = dict(os.environ, ROADIE_STUDIES="pcloud=%s" % lib)
    run = lambda *a: subprocess.run([sys.executable, os.path.join(HERE, "study.py"), "rename", *a], capture_output=True, text=True, env=env)
    before = snapshot(lib)
    assert run("--write").returncode == 1 and run().returncode == 1
    r = run("--dry-run", "--out", str(tmp_path / "t.md"))
    assert r.returncode == 0 and "| A | Flare | Flare [124 Am 8A]" in (tmp_path / "t.md").read_text()
    assert run("--dry-run", "--out", str(lib / "t.md")).returncode == 1
    assert snapshot(lib) == before


def test_the_synthwave_merge_is_proposed_with_exact_files_and_not_done(tmp_path):
    lib = tmp_path / "lib"
    pack = make_study(lib, "Rodriguez Jr.", "Synthwave #1 (remix pack)")
    (pack / "KICK.aif").write_bytes(b"")
    (pack / "KICK.aif.asd").write_bytes(b"")
    (pack / "Synthwave 1 multitrack.als").write_bytes(b"")
    (pack / "mix_124bpm.wav").write_bytes(b"")
    bench = make_study(lib, "Rodriguez Jr.", "Synthwave 1 benchmark")
    (bench / "truth").mkdir()
    study.tag_study(pack, key="Cm", bpm=124, source="online")
    study.tag_study(bench, key="Cm", bpm=124, source="online")
    before = snapshot(lib)
    roots = [study.Root("pcloud", lib)]
    plan = studyrename.plan_rename([r for r in study.find_studies(roots) if r.analysed], roots, tmp_path / "none")
    m = plan["merge"]
    assert m["name"] == "Synthwave #1 [124 Cm 5A]"
    assert m["stay"] == ["Synthwave #1 (remix pack)_124bpm_stems", "mix_124bpm.wav", "study.json"]
    assert m["into_pack"] == ["KICK.aif", "KICK.aif.asd", "Synthwave 1 multitrack.als"]
    text = studyrename.render(plan, "pcloud")
    assert "Rodriguez Jr./Synthwave #1 [124 Cm 5A]/`" in text and "benchmark/study.json" in text and "a `*_stems` folder" in text
    assert snapshot(lib) == before


def files(root):
    """(relative path, size) of every file under root: the tree as the renames must preserve it."""
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file())


def write_tree(tmp_path):
    lib = tmp_path / "lib"
    flare = make_study(lib, "Einmusik", "Flare")
    (flare / "analysis").mkdir()
    (flare / "analysis" / "x.txt").write_text("hello")
    study.tag_study(flare, key="Am", bpm=124, source="online", url="https://x/y")
    study.tag_study(make_study(lib, "Einmusik", "Nosara"), key="C", source="detected", confidence=0.1)
    (lib / "Anyma" / "Anyma Project").mkdir(parents=True)
    (lib / "Anyma" / "Anyma Project" / "a.als").write_text("project")
    return lib


def plan_for(lib):
    roots = [study.Root("pcloud", lib)]
    return studyrename.plan_rename([r for r in study.find_studies(roots) if r.analysed], roots, lib.parent / "none")


def test_renames_move_server_side_with_checks_and_the_undo_restores_the_tree(tmp_path):
    lib = write_tree(tmp_path)
    before = files(lib)
    moves = studyrename.build_moves(plan_for(lib), str(lib))
    assert [(m["old"], m["new"]) for m in moves] == [("Einmusik/Flare", "Einmusik/Flare [124 Am 8A]"),
                                                     ("Einmusik/Nosara", "Einmusik/Nosara [124 C? 8B?]")]
    lines = []
    res = studyrename.run_moves(str(lib), moves, lines.append)
    assert res["failed"] is None and len(res["ok"]) == 2 and lines[0].startswith("OK Einmusik/Flare -> Einmusik/Flare [124 Am 8A] (")
    assert sorted(p.name for p in (lib / "Einmusik").iterdir()) == ["Flare [124 Am 8A]", "Nosara [124 C? 8B?]"]
    assert load_study("Flare", lib).title == "Flare" and load_study("Nosara", lib).key == "C major"
    assert (lib / "Anyma" / "Anyma Project" / "a.als").exists()              # untouched
    after = studyrename.r_files(str(lib))
    beforel = [(size, rel) for rel, size in before]
    assert studyrename.compare_listings(beforel, after, res["ok"]) == []
    assert studyrename.compare_listings(beforel, after, []) != []              # without the moves it is a difference
    undo = studyrename.undo_moves({"root": str(lib), "moves": res["ok"]}, lines.append)
    assert undo["failed"] is None and files(lib) == before
    assert sorted(p.name for p in (lib / "Einmusik").iterdir()) == ["Flare", "Nosara"]


def test_an_existing_target_is_skipped_never_merged_and_a_failed_check_stops_the_run(tmp_path):
    lib = write_tree(tmp_path)
    (lib / "Einmusik" / "Flare [124 Am 8A]").mkdir()
    (lib / "Einmusik" / "Flare [124 Am 8A]" / "mine.txt").write_text("keep")
    before = files(lib)
    moves = studyrename.build_moves(plan_for(lib), str(lib))
    lines = []
    res = studyrename.run_moves(str(lib), moves, lines.append)
    assert [m["old"] for m in res["skipped"]] == ["Einmusik/Flare"] and [m["old"] for m in res["ok"]] == ["Einmusik/Nosara"]
    assert any(l.startswith("SKIPPED Einmusik/Flare") and "target exists" in l for l in lines)
    assert (lib / "Einmusik" / "Flare" / "analysis" / "x.txt").exists() and files(lib / "Einmusik" / "Flare [124 Am 8A]") == [("mine.txt", 4)]
    # a source that is not there stops everything after it
    stop = studyrename.run_moves(str(lib), [{"old": "Einmusik/Gone", "new": "Einmusik/Gone2", "group": None},
                                            {"old": "Einmusik/Flare", "new": "Einmusik/Elsewhere", "group": None}], lines.append)
    assert stop["failed"] and "not there" in stop["failed"]["why"] and stop["ok"] == []
    assert (lib / "Einmusik" / "Flare").exists() and not (lib / "Einmusik" / "Elsewhere").exists()


def synthwave_tree(tmp_path):
    lib = tmp_path / "lib"
    pack = make_study(lib, "Rodriguez Jr.", "Synthwave #1 (remix pack)")
    for n in ("KICK.aif", "KICK.aif.asd", "Synthwave 1 multitrack.als", "mix_124bpm.wav"):
        (pack / n).write_bytes(b"x" * 7)
    (pack / "Backup").mkdir()
    (pack / "Backup" / "old.als").write_text("b")
    bench = make_study(lib, "Rodriguez Jr.", "Synthwave 1 benchmark")
    (bench / "truth").mkdir()
    (bench / "truth" / "bass.flac").write_bytes(b"t")
    (bench / "study.json").write_text(json.dumps({"kind": "study_marker", "stems": "truth", "bpm": 124}))
    study.tag_study(pack, key="Cm", bpm=124, source="online", url="https://x/s")
    return lib


def test_the_synthwave_merge_runs_in_three_checked_steps_and_the_undo_reverses_it(tmp_path):
    lib = synthwave_tree(tmp_path)
    before = files(lib)
    plan = plan_for(lib)
    moves = studyrename.build_moves(plan, str(lib))
    merged = "Rodriguez Jr./Synthwave #1 [124 Cm 5A]"
    assert moves[0] == {"old": "Rodriguez Jr./Synthwave #1 (remix pack)", "new": merged, "group": "synthwave"}
    assert [m["old"].split("/")[-1] for m in moves[1:-1]] == ["Backup", "KICK.aif", "KICK.aif.asd", "Synthwave 1 multitrack.als"]
    assert moves[-1] == {"old": "Rodriguez Jr./Synthwave 1 benchmark", "new": merged + "/benchmark", "group": "synthwave"}
    check = (["Rodriguez Jr./Synthwave #1 (remix pack)", "Rodriguez Jr./Synthwave 1 benchmark"], merged)
    lines = []
    res = studyrename.run_moves(str(lib), moves, lines.append, check)
    assert res["failed"] is None and lines[-1].startswith("OK merge total: %d files" % len(before))
    top = lib / "Rodriguez Jr." / "Synthwave #1 [124 Cm 5A]"
    assert sorted(p.name for p in (lib / "Rodriguez Jr.").iterdir()) == ["Synthwave #1 [124 Cm 5A]"]
    assert sorted(p.name for p in top.iterdir()) == ["Synthwave #1 (remix pack)_124bpm_stems", "benchmark", "mix_124bpm.wav", "pack", "study.json"]
    assert (top / "pack" / "KICK.aif").exists() and (top / "pack" / "Backup" / "old.als").exists()
    assert json.loads((top / "benchmark" / "study.json").read_text())["stems"] == "truth"
    assert json.loads((top / "study.json").read_text())["key"] == "C minor"
    assert [r.title for r in study.find_studies(lib)] == ["Synthwave #1"]
    assert load_study("Synthwave #1", lib).key == "C minor"
    assert studyrename.compare_listings([(s, r) for r, s in before], studyrename.r_files(str(lib)), res["ok"]) == []
    undo = studyrename.undo_moves({"root": str(lib), "moves": res["ok"]}, lines.append)
    assert undo["failed"] is None and files(lib) == before
    # a merged folder that already exists stops the merge before anything moves
    (lib / "Rodriguez Jr." / "Synthwave #1 [124 Cm 5A]").mkdir()
    again = studyrename.run_moves(str(lib), moves, lines.append, check)
    assert again["ok"] == [] and len(again["skipped"]) == len(moves)
    assert (lib / "Rodriguez Jr." / "Synthwave #1 (remix pack)" / "KICK.aif").exists()

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
    moves = studyrename.build_moves(plan_for(lib))
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
    moves = studyrename.build_moves(plan_for(lib))
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


import types

import pytest


def write_args(lib, out, stamp=None):
    return types.SimpleNamespace(root="pcloud", out_dir=str(out), stamp=stamp)


def as_library(monkeypatch, lib):
    """Treat the temp folder as the pCloud library: rclone takes the local path as its remote."""
    monkeypatch.setattr(study, "STUDIES_ROOTS", [study.Root("pcloud", lib)])
    monkeypatch.setattr(studyrename.keymod, "pcloud_remote", lambda p: str(p))


def test_a_second_run_never_replaces_the_first_runs_map_and_the_log_is_written_as_it_goes(tmp_path, monkeypatch):
    lib = write_tree(tmp_path)
    as_library(monkeypatch, lib)
    out = tmp_path / "out"
    assert studyrename.main_write(write_args(lib, out, "first")) == 0
    first = (out / "rename-map-first.json").read_text()
    assert len(json.loads(first)["moves"]) == 2 and "OK Einmusik/Flare" in (out / "rename-log-first.txt").read_text()
    assert studyrename.main_write(write_args(lib, out, "first")) == 1        # same stamp: refused before anything moves
    assert (out / "rename-map-first.json").read_text() == first
    assert studyrename.main_write(write_args(lib, out)) == 0                 # the default stamp is a timestamp: a new set of files
    stamped = sorted(p.name for p in out.glob("rename-map-*.json") if p.name != "rename-map-first.json")
    assert len(stamped) == 1 and stamped[0] != "rename-map-run.json" and json.loads((out / stamped[0]).read_text())["moves"] == []
    # a run that dies part-way still has its log
    (lib / "Einmusik" / "Nosara [124 C? 8B?]").rename(lib / "Einmusik" / "Nosara")
    calls = []
    real = studyrename.r_files

    def dies(path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return real(path)

    monkeypatch.setattr(studyrename, "r_files", dies)
    with pytest.raises(KeyboardInterrupt):
        studyrename.main_write(write_args(lib, out, "dies"))
    assert (out / "rename-log-dies.txt").read_text().startswith("map: 1 moves saved to rename-map-dies.json")


def test_write_reads_the_records_through_rclone_and_refuses_collisions(tmp_path, monkeypatch):
    lib = write_tree(tmp_path)
    as_library(monkeypatch, lib)
    monkeypatch.setattr(study, "_marker", lambda folder: None)               # the mount serves a stale view: no records
    refs = [r for r in study.find_studies([study.Root("pcloud", lib)]) if r.analysed]
    roots = [study.Root("pcloud", lib)]
    stale = {e["old"]: e["new"] for e in studyrename.plan_rename(refs, roots, tmp_path / "none")["entries"]}
    assert stale["Flare"] == "Flare [124]"
    fresh = {e["old"]: e["new"] for e in studyrename.plan_rename(refs, roots, tmp_path / "none", remote_records=True)["entries"]}
    assert fresh["Flare"] == "Flare [124 Am 8A]" and fresh["Nosara"] == "Nosara [124 C? 8B?]"
    (lib / "Einmusik" / "Flare [124 Am 8A]").mkdir()
    out = tmp_path / "out"
    assert studyrename.main_write(write_args(lib, out, "c")) == 1
    assert not out.exists() or not list(out.iterdir())                       # refused before any file or move
    assert (lib / "Einmusik" / "Flare").exists()


def test_a_target_differing_only_by_case_or_unicode_form_counts_as_existing_and_no_name_is_refused_silently(tmp_path):
    import unicodedata
    lib = write_tree(tmp_path)
    (lib / "Einmusik" / "FLARE [124 AM 8A]").mkdir()
    (lib / "Einmusik" / unicodedata.normalize("NFD", "Café")).mkdir()
    assert studyrename.r_exists(str(lib / "Einmusik" / "flare [124 am 8a]"), fold=True)
    assert not studyrename.r_exists(str(lib / "Einmusik" / "flare [124 am 8a]"))
    assert studyrename.r_exists(str(lib / "Einmusik" / unicodedata.normalize("NFC", "Café")), fold=True)
    res = studyrename.run_moves(str(lib), studyrename.build_moves(plan_for(lib)), lambda l: None)
    assert [m["old"] for m in res["skipped"]] == ["Einmusik/Flare"] and (lib / "Einmusik" / "Flare" / "analysis" / "x.txt").exists()
    with pytest.raises(ValueError, match="has no bpm"):
        studyrename.build_moves({"entries": [{"artist": "A", "old": "X", "new": None}]})

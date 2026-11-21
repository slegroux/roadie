#!/usr/bin/env python3
"""Tests for studyrename.py (`roadie study rename --dry-run`): temp trees, nothing is renamed."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import study
import studyrename
from test_study import add_profile, make_study

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

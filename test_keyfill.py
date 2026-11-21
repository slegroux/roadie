#!/usr/bin/env python3
"""Tests for keyfill.py (`roadie study fill-keys --dry-run`): temp library, a fake detector, nothing written."""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import keyfill
import study
from test_study import make_study

HERE = os.path.dirname(os.path.abspath(__file__))


def lookup():
    row = lambda study_, bpm, key, cam, **kw: dict({"row": "x", "study": study_, "bpm": bpm, "key": key, "camelot": cam,
                                                     "url": "https://x/" + study_.split("/")[1], "use_published": True}, **kw)
    return {"decisions": {"Bodzin/Earth": {"key": "B minor", "source": "online", "key_note": "user's old tag: E minor"},
                          "Vitalic/Alone": {"key": "E minor", "source": "tagged"}},
            "rows": [row("Bodzin/Earth", 125, "B minor", "10A"), row("Einmusik/Flare", 124, "C minor", "5A"),
                     row("Matador/Baby", 127, "G major", "9B"), row("Vitalic/Alone", 128, "E minor", "9A"),
                     row("PML/Under Water", 122, "Eb minor", "2A", use_published=False, unused_reason="identity not established"),
                     row("Gone/Missing", 120, "C", "8B")]}


DETECTED = {"Bodzin/Earth": ("B minor", "10A", 0.18), "Einmusik/Flare": ("C minor", "5A", 0.05),
            "Matador/Baby": ("G minor", "6A", 0.1), "Vitalic/Alone": ("E minor", "9A", 0.2),
            "PML/Under Water": ("F major", "7B", 0.05), "Sylvain/Live": ("A minor", "8A", 0.12)}


def detector(path):
    p = path.split(os.sep)
    name = "%s/%s" % (p[-2], p[-1])
    if name == "Rodriguez/Bench":
        raise ValueError("%s has no synths stem" % name)
    key, cam, conf = DETECTED[name]
    return {"key": key, "camelot": cam, "confidence": conf}


def library(tmp_path):
    for artist, title in (("Bodzin", "Earth"), ("Einmusik", "Flare"), ("Matador", "Baby"), ("Vitalic", "Alone"),
                          ("PML", "Under Water"), ("Sylvain", "Live"), ("Rodriguez", "Bench")):
        make_study(tmp_path, artist, title)
    (tmp_path / "Anyma" / "Anyma Project").mkdir(parents=True)             # not analysed: not in the plan
    return [r for r in study.find_studies(tmp_path) if r.analysed]


def snapshot(root):
    return sorted((str(p), p.stat().st_size) for p in root.rglob("*"))


def test_the_plan_applies_the_rules_and_writes_nothing(tmp_path):
    refs = library(tmp_path)
    before = snapshot(tmp_path)
    plan = keyfill.run_plan(lookup(), refs, detector=detector, min_free_gb=0)
    by = {e["study"]: e for e in plan["entries"]}
    earth = by["Bodzin/Earth"]                  # user decision: published key as online, the old tag kept
    assert (earth["key"], earth["source"], earth["url"], earth["key_note"], earth["verdict"]) == (
        "B minor", "online", "https://x/Earth", "user's old tag: E minor", "agree")
    assert (earth["measured_bpm"], earth["listed_bpm"], earth["folder_name"]) == (124.0, 125, "Earth [125 Bm 10A]")
    baby = by["Matador/Baby"]                   # published, detection disagrees: published still written
    assert (baby["key"], baby["source"], baby["verdict"]) == ("G major", "online", "disagree (parallel)")
    assert baby["key_note"] == "detection disagrees (parallel): G minor 6A, confidence 0.100"
    assert by["Einmusik/Flare"]["verdict"] == "agree"
    vit = by["Vitalic/Alone"]
    assert (vit["source"], vit["url"], vit["verdict"]) == ("tagged", None, "agree")
    pml = by["PML/Under Water"]                 # identity unconfirmed: the row is not used, detected only, marked
    assert (pml["source"], pml["key"], pml["listed_bpm"], pml["folder_name"]) == ("detected", "F major", None, "Under Water [124 F? 7B?]")
    assert "identity not established" in pml["notes"][0]
    assert by["Sylvain/Live"]["source"] == "detected"           # no row: detection only
    bench = by["Rodriguez/Bench"]               # one stem: no key at all, and not renamed
    assert (bench["source"], bench["key"], bench["folder_name"]) == (None, None, "Bench [124]")
    assert "no synths stem" in bench["verdict"]
    assert plan["unmatched_rows"] == ["Gone/Missing"] and plan["unmatched_decisions"] == []
    text = keyfill.render(plan, "lib")
    assert "studies with no lookup row: Rodriguez/Bench, Sylvain/Live" in text
    assert snapshot(tmp_path) == before


def test_cli_needs_dry_run_writes_only_the_table_and_refuses_a_path_in_the_library(tmp_path):
    lib, out = tmp_path / "lib", tmp_path / "table.md"
    lib.mkdir()
    make_study(lib, "Einmusik", "Flare")
    (tmp_path / "lookup.json").write_text(json.dumps(lookup()))
    env = dict(os.environ, ROADIE_STUDIES="pcloud=%s" % lib)
    base = [sys.executable, os.path.join(HERE, "study.py"), "fill-keys", "--from", str(tmp_path / "lookup.json")]
    before = snapshot(lib)
    r = subprocess.run(base, capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "--dry-run" in r.stderr
    r = subprocess.run(base + ["--dry-run", "--out", str(lib / "t.md")], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "inside a Studies folder" in r.stderr
    assert snapshot(lib) == before
    # the real detector needs audio; the studies here have empty stems, so detection is recorded as "none"
    r = subprocess.run(base + ["--dry-run", "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "Einmusik/Flare | 124 | 124 | C minor | 5A | online" in text and "Flare [124 Cm 5A]" in text
    assert "**Counts:** online 1, tagged 0, detected 0, no key 0; disagreements 0." in text
    assert snapshot(lib) == before


def test_whole_sets_get_a_note_and_no_key_and_the_measured_bpm_names_the_folder_when_the_listing_is_far_off(tmp_path):
    refs = library(tmp_path)
    lk = lookup()
    lk["decisions"]["Sylvain/Live"] = {"no_key": True, "key_note": "whole-set recording: no single key"}
    lk["decisions"]["Rodriguez/Bench"] = {"key": "C minor", "source": "online", "url": "https://x/orig", "key_note": "listing is for 'Original Mix'"}
    lk["rows"][1]["bpm"] = 92                                       # Flare: listed 92, measured 124
    by = {e["study"]: e for e in keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0)["entries"]}
    live = by["Sylvain/Live"]
    assert (live["key"], live["source"], live["key_note"], live["verdict"]) == (
        None, None, "whole-set recording: no single key", "no key (whole-set recording: no single key)")
    assert by["Rodriguez/Bench"]["url"] == "https://x/orig" and by["Rodriguez/Bench"]["source"] == "online"
    assert (by["Einmusik/Flare"]["listed_bpm"], by["Einmusik/Flare"]["folder_name"]) == (92, "Flare [124 Cm 5A]")
    assert keyfill.name_bpm(123.999, 123) == 123 and keyfill.name_bpm(120.0, 113) == 120.0
    assert keyfill.name_bpm(None, 125) == 125 and keyfill.name_bpm(124.0, None) == 124.0


def test_write_records_through_tag_study_keeps_the_marker_and_saves_what_it_replaces(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    refs = library(lib)
    marker = {"kind": "study_marker", "stems": "Bench_124bpm_stems", "bpm": 124, "t0": 0.0}
    (lib / "Rodriguez" / "Bench" / "study.json").write_text(json.dumps(marker))
    lk = lookup()
    lk["decisions"]["Sylvain/Live"] = {"no_key": True, "key_note": "whole-set recording: no single key"}
    lk["rows"].append({"row": "x", "study": "Rodriguez/Bench", "bpm": 124, "key": "C minor", "camelot": "5A",
                       "url": "https://x/b", "use_published": True})
    plan = keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0, rewrite_all=True)
    before = snapshot(lib)
    lines = []
    ok, failed = keyfill.write_records(plan, tmp_path / "backup", lines.append, rewrite_all=True)
    assert (ok, failed) == (7, 0) and lines[-1] == "done: 7 OK, 0 FAILED"
    after = {p for p, _ in snapshot(lib)} - {p for p, _ in before}
    assert {os.path.basename(p) for p in after} == {"study.json"} and len(after) == 6     # only new study.json files
    kept = json.loads((lib / "Rodriguez" / "Bench" / "study.json").read_text())
    assert (kept["stems"], kept["bpm"], kept["t0"], kept["key"], kept["key_source"], kept["listed_bpm"]) == (
        "Bench_124bpm_stems", 124, 0.0, "C minor", "online", 124)
    assert json.loads((tmp_path / "backup" / "Rodriguez__Bench.study.json").read_text()) == marker
    baby = json.loads((lib / "Matador" / "Baby" / "study.json").read_text())
    assert baby["key"] == "G major" and baby["key_note"].startswith("detection disagrees (parallel)")
    live = json.loads((lib / "Sylvain" / "Live" / "study.json").read_text())
    assert "key" not in live and live["key_note"] == "whole-set recording: no single key"
    assert study.load_study("Live", lib).key is None and study.load_study("Baby", lib).key == "G major"
    # a key from a weaker source than the record's is a logged failure, and the run goes on
    lk["rows"][0]["bpm"] = 125
    (lib / "Einmusik" / "Flare" / "study.json").write_text(json.dumps({"kind": "study_marker", "key": "A minor", "key_source": "by ear"}))
    lines.clear()
    ok, failed = keyfill.write_records(keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0, rewrite_all=True),
                                       tmp_path / "backup", lines.append, rewrite_all=True)
    assert failed == 1 and any(l.startswith("FAILED Einmusik/Flare") and "outranks" in l for l in lines) and ok == 6


def test_cli_write_is_for_one_non_local_root_with_a_backup_dir(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_study(lib, "Einmusik", "Flare")
    (tmp_path / "lookup.json").write_text(json.dumps(lookup()))
    base = [sys.executable, os.path.join(HERE, "study.py"), "fill-keys", "--from", str(tmp_path / "lookup.json"), "--write"]
    env = dict(os.environ, ROADIE_STUDIES="local=%s" % lib)
    r = subprocess.run(base + ["--root", "local", "--backup-dir", str(tmp_path / "b")], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "not local" in r.stderr
    r = subprocess.run(base + ["--root", "local"], capture_output=True, text=True, env=env)
    assert r.returncode == 1
    assert not (lib / "Einmusik" / "Flare" / "study.json").exists()


def test_a_rerun_writes_only_studies_without_a_key_and_keeps_what_the_user_set(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    refs = library(lib)
    lk = lookup()
    lk["decisions"]["Sylvain/Live"] = {"no_key": True, "key_note": "whole-set recording: no single key"}
    lk["rows"].append({"row": "x", "study": "Rodriguez/Bench", "bpm": 124, "key": "C minor", "camelot": "5A",
                       "url": "https://x/b", "use_published": True})
    plan = keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0)
    assert all(e["write"] for e in plan["entries"])
    lines = []
    assert keyfill.write_records(plan, tmp_path / "backup", lines.append) == (7, 0)
    again = keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0)       # nothing is left to do
    assert not any(e["write"] for e in again["entries"]) and "**Records to write: 0**" in keyfill.render(again, "lib")
    lines.clear()
    assert keyfill.write_records(again, tmp_path / "backup", lines.append) == (0, 0)
    assert all(l.startswith("SKIPPED") for l in lines[:-1])
    # the user's own bpm and note survive a run on a study that still has no key; --all rewrites the study
    folder = lib / "Sylvain" / "Live"
    study.tag_study(folder, bpm=99)
    (folder / "study.json").write_text(json.dumps({"kind": "study_marker", "listed_bpm": 99}))
    plan = keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0)
    assert [e["study"] for e in plan["entries"] if e["write"]] == ["Sylvain/Live"]
    keyfill.write_records(plan, tmp_path / "backup", lines.append)
    assert json.loads((folder / "study.json").read_text()) == {"kind": "study_marker", "listed_bpm": 99, "key_note": "whole-set recording: no single key"}
    assert sorted(p.name for p in (tmp_path / "backup").iterdir()) == ["Sylvain__Live.study.json"]
    flare = lib / "Einmusik" / "Flare" / "study.json"                           # a tempo the user set, no key yet
    flare.write_text(json.dumps({"kind": "study_marker", "listed_bpm": 99}))
    keyfill.write_records(keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0), tmp_path / "backup", lines.append)
    assert (json.loads(flare.read_text())["listed_bpm"], json.loads(flare.read_text())["key"]) == (99, "C minor")
    mine = lib / "Vitalic" / "Alone" / "study.json"
    mine.write_text(json.dumps({"kind": "study_marker", "key": "E minor", "key_source": "tagged", "key_note": "mine"}))
    for _ in range(2):                                                          # --all twice: the same key keeps its note
        keyfill.write_records(keyfill.run_plan(lk, refs, detector=detector, min_free_gb=0, rewrite_all=True),
                              tmp_path / "backup", lines.append, rewrite_all=True)
    assert json.loads(mine.read_text())["key_note"] == "mine"
    assert sorted(p.name for p in (tmp_path / "backup").glob("Vitalic__Alone*")) == ["Vitalic__Alone.study.json", "Vitalic__Alone.study.json.1"]

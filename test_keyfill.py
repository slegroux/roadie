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
    assert by["Einmusik/Flare"]["verdict"] == "agree"
    vit = by["Vitalic/Alone"]
    assert (vit["source"], vit["url"], vit["verdict"]) == ("tagged", None, "agree")
    pml = by["PML/Under Water"]                 # identity unconfirmed: the row is not used, detected only, marked
    assert (pml["source"], pml["key"], pml["listed_bpm"], pml["folder_name"]) == ("detected", "F major", None, "Under Water [124 F? 7B?]")
    assert "identity not established" in pml["notes"][0]
    assert by["Sylvain/Live"]["source"] == "detected"           # no row: detection only
    bench = by["Rodriguez/Bench"]               # one stem: no key at all, and not renamed
    assert (bench["source"], bench["key"], bench["folder_name"]) == (None, None, None)
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

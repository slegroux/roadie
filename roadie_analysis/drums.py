"""Drum roles for a folder of drum stems: `drum_roles` runs hit_groups, mark_bleed and name_roles over every
`drums_<kind>` stem, and writes `drums.json` and one clip per role. The rules are drumroles.py's (docs/DRUMROLES.md).

The grid comes from the folder's `alignment.json` (tempo `bpm`, bar line `downbeat_sec`) and `sections.json` unless
given. A default bar line is moved back by whole bars until it is the first one in the file (so no hit has a negative
bar): the 16ths are still counted from the downbeat. A `t0` that is passed is used as it is. Needs numpy and
soundfile (requirements.txt).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from roadie_analysis import drumroles

KIND, VERSION = "drum_roles", 1
AUDIO = (".flac", ".wav")
BEATS_PER_BAR = 4
X_SHARE, O_SHARE = 0.5, 0.10   # pattern line: x = the role plays at that 16th in at least half of its bars, o = at least a tenth
LEGEND = "x = most bars (50 % or more), o = some bars (10 to 50 %), . = fewer"


def pattern_line(pattern: list) -> str:
    """16 characters, one per 16th from the downbeat, in groups of four: `x...x...x...x...`."""
    return "".join("x" if v >= X_SHARE else "o" if v >= O_SHARE else "." for v in pattern)


def find_stems(stems_dir) -> dict[str, Path]:
    """{kind: file} for each `drums_<kind>` stem of the folder (`drums.flac`, the sum, is not one)."""
    found = {}
    for f in sorted(Path(stems_dir).iterdir()):
        if f.suffix in AUDIO and f.stem.startswith("drums_") and not f.name.startswith("."):
            found.setdefault(f.stem[len("drums_"):], f)
    return found


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def grid_from_folder(stems_dir) -> dict:
    """{"tempo", "downbeat_s", "sections": [{name, time_s, bars}], "files": [Path], "notes"} from alignment.json and
    sections.json; a missing value is None. The sections' times are in seconds so they can be placed on any t0."""
    d = Path(stems_dir)
    tempo = downbeat = None
    files, notes, secs = [], [], []
    align = _read_json(d / "alignment.json")
    res = align.get("result") if isinstance(align, dict) else None
    if isinstance(res, dict):
        files.append(d / "alignment.json")
        tempo, downbeat = res.get("bpm"), res.get("downbeat_sec")
        if res.get("downbeat_verdict") == "chance-level":
            notes.append("the downbeat is chance-level (confidence %s): the 16ths may be counted from the wrong beat of the "
                         "bar" % res.get("downbeat_confidence"))
    data = _read_json(d / "sections.json")
    if isinstance(data, dict) and isinstance(data.get("sections"), list):
        files.append(d / "sections.json")
        tempo = tempo if tempo else data.get("bpm")
        for s in data["sections"]:
            if isinstance(s.get("time_sec"), (int, float)) and isinstance(s.get("duration_beats"), (int, float)):
                secs.append({"name": s.get("label"), "time_s": float(s["time_sec"]),
                             "bars": max(1, int(s["duration_beats"]) // BEATS_PER_BAR)})
        if downbeat is None and secs:
            downbeat = secs[0]["time_s"]                       # sections start on a bar line
            notes.append("no alignment.json: bar 0 is the start of the first section")
    return {"tempo": tempo, "downbeat_s": downbeat, "sections": secs, "files": files, "notes": notes}


def _file_record(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime": round(st.st_mtime, 3)}


def drum_roles(stems_dir, tempo: float | None = None, t0: float | None = None, sections: list | None = None,
               out=None, write: bool = True, deadline: float | None = None) -> dict:
    """The roles of every drum stem in `stems_dir`, as the `drums.json` record. `tempo` (BPM), `t0` (seconds from the start of the
    files to bar 0) and `sections` ([{name, start_bar, bars}], bars counted from t0) default to the folder's alignment.json
    and sections.json. With `write` the record goes to `<out or stems_dir>/drums.json` and each role's clean hit to
    `<out or stems_dir>/drum_roles/<stem>-<role>.wav`. `deadline` is a time.monotonic() value checked between stems; passing it
    raises TimeoutError (nothing is written). Raises ValueError when there are no drum stems or no grid.

    Per stem: {"file", "sample_rate", "hits", "hat_like", "roles", "bleed", "per_section", "notes"}. A role is {"name", "named",
    "rule", "reason" (why neutral), "hits" (measured + assigned), "measured", "pattern" (share of bars at each 16th), "line",
    "level_db", "stability", "separation", "groups", "note", "variants", "clip" (path relative to the folder, or None),
    "clip_note" (why none), "per_section" {section: hits}}. "bleed" lists the groups marked bleed (left out of the roles)."""
    stems_dir = Path(stems_dir)
    stems = find_stems(stems_dir)
    if not stems:
        raise ValueError("no drum stems (drums_<kind>.flac or .wav) in %s" % stems_dir)
    g = grid_from_folder(stems_dir)
    notes = list(g["notes"])
    tempo = tempo if tempo is not None else g["tempo"]
    if tempo is None or not (tempo > 0 and math.isfinite(tempo)):
        raise ValueError("no tempo: pass tempo=, or make alignment.json (roadie load) or sections.json (roadie sections)")
    bar_s = 60.0 / tempo * BEATS_PER_BAR
    if t0 is None:
        if g["downbeat_s"] is None:
            raise ValueError("no downbeat: pass t0=, or make alignment.json (roadie load) or sections.json (roadie sections)")
        down = float(g["downbeat_s"])
        t0 = down - math.floor(down / bar_s) * bar_s
        shifted = round((down - t0) / bar_s)
        if shifted:
            notes.append("bar 0 is the downbeat %.3f s moved back %d whole bars to %.3f s, so that no hit has a negative bar; "
                         "the 16ths are counted from the same bar line" % (down, shifted, t0))
        default_t0 = True
    else:
        default_t0 = False
    if sections is None:
        sections = []
        for s in g["sections"]:
            a = max(0, int(round((s["time_s"] - t0) / bar_s)))
            sections.append({"name": s["name"], "start_bar": a, "bars": s["bars"]})
        sections = sections or None
    elif not sections:
        sections = None
    results = {}
    for kind, f in stems.items():
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("passed the deadline before %s (done: %s)" % (kind, ", ".join(results) or "none"))
        results[kind] = drumroles.hit_groups(f, tempo, t0, sections)
    bleed = drumroles.mark_bleed(results)
    record = {"kind": KIND, "version": VERSION, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "grid": {"tempo": tempo, "t0": round(t0, 4), "beats_per_bar": BEATS_PER_BAR,
                       "from": "alignment.json and sections.json" if default_t0 else "given",
                       "downbeat_s": g["downbeat_s"] if default_t0 else None, "sections": sections},
              "inputs": [_file_record(p) for p in list(stems.values()) + g["files"]],
              "legend": LEGEND, "notes": notes, "stems": {}}
    clips = []
    for kind, res in results.items():
        named = drumroles.name_roles(res, kind, bleed[kind])
        record["stems"][kind], more = _stem_record(kind, stems[kind], res, named, bleed[kind], sections)
        clips += more
    if write:
        base = Path(out) if out else stems_dir
        (base / "drum_roles").mkdir(parents=True, exist_ok=True)
        for kind, role, audio, sr in clips:
            rel = "drum_roles/%s.wav" % _slug(kind, role["name"])
            sf.write(str(base / rel), audio, sr, subtype="PCM_24")
            role["clip"] = rel
        (base / "drums.json").write_text(json.dumps(record, indent=2) + "\n")
        record["written"] = str(base / "drums.json")
    else:
        for _kind, role, _a, _s in clips:
            role["clip_note"] = "not written (write=False)"
    return record


def _slug(kind: str, name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s if s.startswith(kind) else "%s-%s" % (kind, s)


def _stem_record(kind: str, path: Path, res: dict, named: dict, bleed: dict, sections: list | None):
    by_id = {g["id"]: g for g in res["groups"]}
    info = {g["id"]: g for g in named["groups"]}
    roles, clips = [], []
    for r in named["roles"]:
        big = by_id[max(r["groups"], key=lambda i: by_id[i]["count"])]
        e = info[big["id"]]
        clip = big["clean_hit"]
        rec = {"name": r["name"], "named": r["named"], "rule": r["rule"], "reason": None if r["named"] else e["reason"],
               "hits": r["count"] + r["assigned"], "measured": r["count"], "assigned": r["assigned"],
               "pattern": [round(float(v), 3) for v in r["pattern"]], "line": pattern_line(r["pattern"]),
               "level_db": None if r["level_db"] is None else round(float(r["level_db"]), 1),
               "stability": big["stability"], "stability_status": big["stability_status"],
               "separation": None if big["separation"] is None else round(float(big["separation"]), 2),
               "groups": r["groups"], "note": r["note"], "variants": r["variants"],
               "possibly_also": r["possibly_also"], "clip": None,
               "clip_note": None if clip else big["clean_note"], "per_section": {}}
        for sec in res["sections"]:
            n = sum(g["count"] + g["assigned"] for g in sec["groups"] if g["id"] in r["groups"])
            if n:
                rec["per_section"][sec["name"]] = n
        roles.append(rec)
        if clip:
            clips.append((kind, rec, np.asarray(clip["audio"], dtype=np.float32), res["sample_rate"]))
    marked = []
    for g in res["groups"]:
        if bleed.get(g["id"], {}).get("bleed"):
            marked.append({"name": "%s %d" % (kind, g["id"] + 1), "hits": g["count"] + g["assigned"],
                           "line": pattern_line(g["pattern"]), "level_db": None if g["level_db"] is None else round(float(g["level_db"]), 1),
                           "why": bleed[g["id"]]["why"]})
    per_section = [{"name": s["name"], "start_bar": s["start_bar"], "bars": s["bars"],
                    "roles": {r["name"]: r["per_section"][s["name"]] for r in roles if s["name"] in r["per_section"]}}
                   for s in res["sections"]]
    rec = {"file": path.name, "sample_rate": res["sample_rate"], "hits": len(res["hits"]), "hat_like": res["hat_like"],
           "bars": res["bars"], "first_bar": res["first_bar"], "roles": roles, "bleed": marked, "per_section": per_section,
           "notes": res["notes"] + named["notes"]}
    return rec, clips


def format_table(record: dict) -> str:
    """The record as one readable table per stem."""
    g = record["grid"]
    lines = ["grid: %.3f bpm, bar 0 at %.3f s (%s)" % (g["tempo"], g["t0"], g["from"]),
             "pattern: 16 steps from the downbeat, one per 16th; %s" % record["legend"]]
    lines += ["note: " + n for n in record["notes"]]
    for kind, s in record["stems"].items():
        lines += ["", "%s  (%s, %d hits found, %d bars)" % (kind, s["file"], s["hits"], s["bars"]),
                  "  %-14s %5s  %-19s %7s  %s" % ("role", "hits", "pattern", "level", "flags")]
        for r in s["roles"]:
            flags = [] if r["named"] else ["neutral"]
            if r["clip"] is None and not str(r["clip_note"]).startswith("not written"):
                flags.append("no clip")
            if r["stability"] is not None:
                flags.append("stability %.2f" % r["stability"])
            if r["separation"] is not None:
                flags.append("separation %.2f" % r["separation"])
            if r["variants"]:
                flags.append("%d variants" % len(r["variants"]))
            lvl = "%.1f dB" % r["level_db"] if r["level_db"] is not None else "-"
            lines.append("  %-14s %5d  %-19s %7s  %s" % (r["name"], r["hits"], " ".join(r["line"][i:i + 4] for i in range(0, 16, 4)),
                                                          lvl, ", ".join(flags)))
            if r["reason"]:
                lines.append("      why neutral: %s" % r["reason"])
            if r["note"]:
                lines.append("      note: %s" % r["note"])
            if r["clip"]:
                lines.append("      clip: %s" % r["clip"])
            elif r["clip_note"]:
                lines.append("      no clip: %s" % r["clip_note"])
        for b in s["bleed"]:
            lines.append("  BLEED %-8s %5d  %-19s %7s  %s" % (b["name"], b["hits"], " ".join(b["line"][i:i + 4] for i in range(0, 16, 4)),
                                                              "%.1f dB" % b["level_db"] if b["level_db"] is not None else "-", b["why"]))
        lines += ["  " + n for n in s["notes"]]
    return "\n".join(lines)

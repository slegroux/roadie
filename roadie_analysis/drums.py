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
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import soundfile as sf

from roadie_analysis import drumroles, drumsections

KIND, VERSION = "drum_roles", 1
AUDIO = (".flac", ".wav")
BEATS_PER_BAR = 4
X_SHARE, O_SHARE = 0.5, 0.10   # pattern line: x = the role plays at that 16th in at least half of its bars, o = at least a tenth
LEGEND = "x = most of the bars the role plays in (50 % or more), o = some of them (10 to 50 %), . = fewer"
MIN_SHOWN = 4              # the table and `study show` list a role of fewer measured hits as one line (all stay in drums.json)


def pattern_line(pattern: list) -> str:
    """16 characters, one per 16th from the downbeat: `x...x...x...x...`. `pattern` is the share of the role's own bars."""
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
               out=None, write: bool = True, deadline: float | None = None, jobs: int | None = None) -> dict:
    """The roles of every drum stem in `stems_dir`, as the `drums.json` record. Each stem is analysed section by section
    (drumsections.py: sections shorter than 8 bars join their neighbour, no sections means the whole track as one) and the
    groups of the sections are matched by sound, so one role keeps one identity and name over the track; `mark_bleed` and
    `name_roles` then run on the merged roles' hits. `tempo` (BPM), `t0` (seconds from the start of the files to bar 0) and
    `sections` ([{name, start_bar, bars}], bars counted from t0) default to the folder's alignment.json and sections.json.
    `jobs` processes run the (stem, section) jobs, the matching and the naming (default: the cores; 1: in this process).
    With `write` the record goes to `<out or stems_dir>/drums.json` and each role's clean hit to
    `<out or stems_dir>/drum_roles/<stem>-<role>.wav` (the clips the folder's earlier drums.json listed are removed first).
    `deadline` is a time.monotonic() value checked as results come in; passing it raises TimeoutError (nothing is written; jobs
    already started finish in the background). Raises ValueError when there are no drum stems or no grid.

    Per stem: {"file", "sample_rate", "hits", "hat_like", "roles", "bleed", "per_section", "notes"}. A role is {"name", "named",
    "rule", "reason" (why neutral), "hits" (measured + assigned), "measured", "pattern" (share of all the track's bars at each
    16th), "own_pattern" (share of the bars the role plays in), "line", "level_db", "stability", "separation", "groups", "note",
    "variants", "clip" (path relative to the folder, or None), "clip_note" (why none), "per_section" [{section, start_bar,
    bars, hits, bars_with, pattern, line}] (the sections the role plays in)}. "bleed" lists the roles marked bleed (left out)."""
    t_start = time.monotonic()
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
    plan = drumsections.plan_sections(sections or None)
    if len(plan) != len(sections or [None]):
        notes.append("sections shorter than %d bars were joined to their neighbour: %s" % (
            drumsections.MIN_SECTION_BARS, ", ".join(p["name"] for p in plan if "+" in str(p["name"]))))
    workers = min(max(1, jobs if jobs else (os.cpu_count() or 1)), len(stems) * len(plan))
    pool = None if workers == 1 else ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    stages = {}
    try:
        t = time.monotonic()
        peaks = dict(zip(stems, _run(drumsections.stem_peak, [(f,) for f in stems.values()], pool, deadline)))
        stages["levels"] = time.monotonic() - t
        t = time.monotonic()
        keys = [(k, si) for k in stems for si in range(len(plan))]
        outs = _run(drumsections.section_job, [(stems[k], tempo, t0, plan[si], peaks[k]) for k, si in keys], pool, deadline)
        stages["sections"] = time.monotonic() - t
        t = time.monotonic()
        merged = _run(drumsections.merge_stem, [([outs[i] for i, (k2, _) in enumerate(keys) if k2 == k], plan, tempo, t0) for k in stems],
                      pool, deadline)
        results = dict(zip(stems, merged))
        stages["matching"] = time.monotonic() - t
        t = time.monotonic()
        bleed = drumroles.mark_bleed(results)
        stages["bleed"] = time.monotonic() - t
        t = time.monotonic()
        named = _run(drumroles.name_roles, [(results[k], k, bleed[k]) for k in stems], pool, deadline)
        stages["naming"] = time.monotonic() - t
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    record = {"kind": KIND, "version": VERSION, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "grid": {"tempo": tempo, "t0": round(t0, 4), "beats_per_bar": BEATS_PER_BAR,
                       "from": "alignment.json and sections.json" if default_t0 else "given",
                       "downbeat_s": g["downbeat_s"] if default_t0 else None,
                       "sections": [{k: p[k] for k in ("name", "start_bar", "bars")} for p in plan]},
              "by": "section", "inputs": [_file_record(p) for p in list(stems.values()) + g["files"]],
              "legend": LEGEND, "notes": notes, "stems": {}}
    clips = []
    for kind, nm in zip(stems, named):
        record["stems"][kind], more = _stem_record(kind, stems[kind], results[kind], nm, bleed[kind], plan, t0, bar_s)
        clips += more
    if write:
        base = Path(out) if out else stems_dir
        _remove_old_clips(base)
        (base / "drum_roles").mkdir(parents=True, exist_ok=True)
        for kind, role, audio, sr in clips:
            rel = "drum_roles/%s.wav" % _slug(kind, role["name"])
            sf.write(str(base / rel), audio, sr, subtype="PCM_24")
            role["clip"] = rel
        record["seconds"], record["jobs"], record["stage_seconds"] = round(time.monotonic() - t_start, 1), workers, {
            k: round(v, 1) for k, v in stages.items()}
        (base / "drums.json").write_text(json.dumps(record, indent=2) + "\n")
        record["written"] = str(base / "drums.json")
    else:
        for _kind, role, _a, _s in clips:
            role["clip_note"] = "not written (write=False)"
        record["seconds"], record["jobs"], record["stage_seconds"] = round(time.monotonic() - t_start, 1), workers, {
            k: round(v, 1) for k, v in stages.items()}
    return record


def _run(fn, argsets: list, pool, deadline: float | None) -> list:
    """fn(*args) for each argument set, in order, in the pool (or here when there is none); TimeoutError past `deadline`."""
    if pool is None:
        out = []
        for a in argsets:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("passed the deadline after %d of %d jobs" % (len(out), len(argsets)))
            out.append(fn(*a))
        return out
    futs = [pool.submit(fn, *a) for a in argsets]
    out = []
    try:
        for f in futs:
            out.append(f.result(timeout=None if deadline is None else max(0.0, deadline - time.monotonic())))
    except FutureTimeout:
        raise TimeoutError("passed the deadline after %d of %d jobs" % (len(out), len(argsets)))
    return out


def _remove_old_clips(base: Path) -> None:
    """The clips the folder's earlier drums.json lists, which this run replaces."""
    old = _read_json(base / "drums.json")
    if not isinstance(old, dict) or old.get("kind") != KIND:
        return
    for s in (old.get("stems") or {}).values():
        for r in s.get("roles", []):
            clip = r.get("clip")
            if isinstance(clip, str) and clip.startswith("drum_roles/") and ".." not in clip:
                (base / clip).unlink(missing_ok=True)


def _slug(kind: str, name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s if s.startswith(kind) else "%s-%s" % (kind, s)


def _own_line(hits: list, bars: int | None = None) -> tuple[list, int]:
    """(share of the bars the hits are in that have an on-grid hit at each 16th, number of those bars)."""
    own = {h["bar"] for h in hits}
    on = [h for h in hits if h["on_grid"]]
    return [len({h["bar"] for h in on if h["step"] == s}) / len(own) if own else 0.0 for s in range(16)], len(own)


def _stem_record(kind: str, path: Path, res: dict, named: dict, bleed: dict, plan: list, t0: float, bar_s: float):
    by_id = {g["id"]: g for g in res["groups"]}
    info = {g["id"]: g for g in named["groups"]}
    hits = res["hits"]

    def in_section(h, p):
        return (p["first"] or h["bar"] >= p["start_bar"]) and (p["to_end"] or h["bar"] < p["start_bar"] + p["bars"])

    def section_view(ids):
        mine = [hits[i] for i in ids]
        rows = []
        for p in plan:
            hs = [h for h in mine if in_section(h, p)]
            if hs:
                pat, nb = _own_line(hs)
                rows.append({"section": p["name"], "start_bar": p["start_bar"], "bars": p["bars"], "hits": len(hs), "bars_with": nb,
                             "pattern": [round(float(v), 3) for v in pat], "line": pattern_line(pat)})
        return rows

    roles, clips = [], []
    for r in named["roles"]:
        big = by_id[max(r["groups"], key=lambda i: by_id[i]["count"])]
        e = info[big["id"]]
        clip = big["clean_hit"]
        ids = [i for gid in r["groups"] for i in by_id[gid]["members"] + by_id[gid]["assigned_members"]]
        own, nb = _own_line([hits[i] for i in ids])
        rec = {"name": r["name"], "named": r["named"], "rule": r["rule"], "reason": None if r["named"] else e["reason"],
               "hits": r["count"] + r["assigned"], "measured": r["count"], "assigned": r["assigned"],
               "bars_with": nb, "pattern": [round(float(v), 3) for v in r["pattern"]],
               "own_pattern": [round(float(v), 3) for v in own], "line": pattern_line(own),
               "off_grid": sum(by_id[i]["off_grid"] for i in r["groups"]),
               "level_db": None if r["level_db"] is None else round(float(r["level_db"]), 1),
               "stability": None if big["stability"] is None else round(float(big["stability"]), 2),
               "stability_status": big["stability_status"],
               "separation": None if big["separation"] is None else round(float(big["separation"]), 2),
               "groups": r["groups"], "note": r["note"], "variants": r["variants"],
               "possibly_also": r["possibly_also"], "clip": None,
               "clip_note": None if clip else big["clean_note"], "per_section": section_view(ids)}
        roles.append(rec)
        if clip:
            clips.append((kind, rec, np.asarray(clip["audio"], dtype=np.float32), res["sample_rate"]))
    marked = []
    for g in res["groups"]:
        if bleed.get(g["id"], {}).get("bleed"):
            pat, _ = _own_line([hits[i] for i in g["members"] + g["assigned_members"]])
            marked.append({"name": "%s %d" % (kind, g["id"] + 1), "hits": g["count"] + g["assigned"], "line": pattern_line(pat),
                           "level_db": None if g["level_db"] is None else round(float(g["level_db"]), 1),
                           "why": bleed[g["id"]]["why"]})
    per_section = [{"name": p["name"], "start_bar": p["start_bar"], "bars": p["bars"],
                    "roles": {r["name"]: v["hits"] for r in roles for v in r["per_section"] if v["section"] == p["name"]}}
                   for p in plan]
    rec = {"file": path.name, "sample_rate": res["sample_rate"], "hits": sum(h["owned"] for h in hits),
           "hat_like": res["hat_like"], "bars": res["bars"], "first_bar": res["first_bar"], "roles": roles, "bleed": marked,
           "per_section": per_section, "notes": res["notes"] + named["notes"]}
    return rec, clips


def format_table(record: dict) -> str:
    """The record as one readable table per stem."""
    g = record["grid"]
    lines = ["grid: %.3f bpm, bar 0 at %.3f s (%s)" % (g["tempo"], g["t0"], g["from"]),
             "pattern: 16 steps from the downbeat, one per 16th; %s" % record["legend"],
             "each role has its pattern over the whole track, then one line per section it plays in (hits, pattern)",
             "no clip: no member of the group is complete (cut off by the next hit, or by 0.5 s, before it falls 40 dB)"]
    lines += ["note: " + n for n in record["notes"]]
    for kind, s in record["stems"].items():
        lines += ["", "%s  (%s, %d hits found, %d bars)" % (kind, s["file"], s["hits"], s["bars"]),
                  "  %-14s %5s  %-19s %7s  %s" % ("role", "hits", "pattern", "level", "flags")]
        shown = [r for r in s["roles"] if r["measured"] >= MIN_SHOWN]
        small = [r for r in s["roles"] if r["measured"] < MIN_SHOWN]
        for r in shown:
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
            for v in r["per_section"] if len(r["per_section"]) > 1 else []:
                lines.append("      %-22s %4d  %s" % (v["section"][:22], v["hits"], " ".join(v["line"][i:i + 4] for i in range(0, 16, 4))))
            if r["clip"]:
                lines.append("      clip: %s" % r["clip"])
        if small:
            lines.append("  + %d groups of fewer than %d measured hits (%d hits in all): in drums.json" % (
                len(small), MIN_SHOWN, sum(r["hits"] for r in small)))
        for b in s["bleed"]:
            lines.append("  BLEED %-8s %5d  %-19s %7s  %s" % (b["name"], b["hits"], " ".join(b["line"][i:i + 4] for i in range(0, 16, 4)),
                                                              "%.1f dB" % b["level_db"] if b["level_db"] is not None else "-", b["why"]))
        lines += ["  " + n for n in s["notes"]]
    return "\n".join(lines)

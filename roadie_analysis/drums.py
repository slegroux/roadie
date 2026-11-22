"""Drum roles for a folder of drum stems: `drum_roles` runs hit_groups, mark_bleed and name_roles over every
`drums_<kind>` stem, and writes `drums.json` and one clip per role. The rules are drumroles.py's (docs/DRUMROLES.md).

The grid comes from the folder's `alignment.json` (tempo `bpm`, bar line `downbeat_sec`) and `sections.json` unless
given. A default bar line is moved back by whole bars to the first one at or after the start of the file; the 16ths are
still counted from the downbeat. A hit before that line (a pickup, a hit in the first partial bar) has bar -1. A `t0` that is passed is used as it is. Needs numpy and
soundfile (requirements.txt).
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import soundfile as sf

from roadie_analysis import drumroles, drumsections
from roadie_analysis import key as keymod

KIND, VERSION = "drum_roles", 1
AUDIO = (".flac", ".wav")
BEATS_PER_BAR = 4
X_SHARE, O_SHARE = 0.5, 0.10   # pattern line: x = the role plays at that 16th in at least half of its bars, o = at least a tenth
STEADY_MARGIN_S = 0.003    # a name is steady when it holds with the 20 ms on-grid tolerance 3 ms tighter and 3 ms looser (about what
                           # 0.001 bpm moves the last section's bar line by: 160 bars of 1.97 s)
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
               out=None, write: bool = True, deadline: float | None = None, jobs: int | None = None,
               hit_times: bool = False) -> dict:
    """The roles of every drum stem in `stems_dir`, as the `drums.json` record. Each stem is analysed section by section
    (drumsections.py: sections shorter than 8 bars join their neighbour, no sections means the whole track as one) and the
    groups of the sections are matched by sound, so one role keeps one identity and name over the track; `mark_bleed` and
    `name_roles` then run on the merged roles' hits. `tempo` (BPM), `t0` (seconds from the start of the files to bar 0) and
    `sections` ([{name, start_bar, bars}], bars counted from t0) default to the folder's alignment.json and sections.json.
    `jobs` processes run the (stem, section) jobs, the matching and the naming (default: the cores; 1: in this process).
    With `write` the record goes to `<out or stems_dir>/drums.json` and each role's clean hit to
    `<out or stems_dir>/drum_roles/<stem>-<role>.wav` (new clips first, then drums.json by temp file and rename, then the clips the earlier drums.json listed and this one does not).
    A study on the pCloud mount is analysed in a temp folder and its results uploaded with rclone, never read or written through the mount (`_library_roles`). `hit_times` adds each role's hits' onset times (`hit_times`) to the record. `deadline` is a time.monotonic() value checked as results come in; passing it raises TimeoutError (nothing is written; jobs
    already started finish in the background). Raises ValueError when there are no drum stems or no grid.

    Per stem: {"file", "sample_rate", "hits", "hat_like", "roles", "bleed", "per_section", "notes"}. A role is {"name", "named",
    "rule", "reason" (why neutral), "hits" (measured + assigned), "measured", "pattern" (share of all the track's bars at each
    16th), "own_pattern" (share of the bars the role plays in), "line", "level_db", "stability", "separation", "groups", "note",
    "variants", "clip" (path relative to the folder, or None), "clip_note" (why none), "per_section" [{index, section, start_bar,
    bars, hits, bars_with, pattern, line}] (the sections the role plays in; "index" is the section's place in the grid's list)}. "bleed" lists the roles marked bleed (left out)."""
    t_start = time.monotonic()
    stems_dir = Path(stems_dir)
    remote = keymod.pcloud_remote(stems_dir)
    if remote is not None:
        return _library_roles(stems_dir, remote, tempo, t0, sections, out, write, deadline, jobs, hit_times)
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
            notes.append("bar 0 is the downbeat %.3f s moved back %d whole bars to %.3f s, the first bar line in the file (a hit before "
                         "it, a pickup, has bar -1); the 16ths are counted from the same bar line" % (down, shifted, t0))
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
        named = _run(drumroles.name_roles, [(_for_naming(results[k]), k, bleed[k]) for k in stems], pool, deadline)
        stages["naming"] = time.monotonic() - t
        t = time.monotonic()
        tight = _run(drumroles.name_roles, [(_with_grid_tol(_for_naming(results[k]), drumroles.GRID_LEAD_S - STEADY_MARGIN_S), k, bleed[k]) for k in stems],
                     pool, deadline)
        loose = _run(drumroles.name_roles, [(_with_grid_tol(_for_naming(results[k]), drumroles.GRID_LEAD_S + STEADY_MARGIN_S), k, bleed[k]) for k in stems],
                     pool, deadline)
        stages["steadiness"] = time.monotonic() - t
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
    for kind, nm, tg, lo in zip(stems, named, tight, loose):
        record["stems"][kind], more = _stem_record(kind, stems[kind], results[kind], nm, bleed[kind], plan, t0, bar_s, hit_times, (tg, lo))
        clips += more
    for kind, st in record["stems"].items():
        far = ["%s (%+.0f ms)" % (ps["name"], ps["grid_shift_ms"]) for ps in st["per_section"] if abs(ps["grid_shift_ms"]) > drumroles.GRID_LEAD_S * 1000]
        if far:
            record["notes"].append("%s: in %s most hits sit more than %g ms from the aligner's grid lines: the grid's phase may be wrong "
                                   "here, or this stem's dominant sound plays late" % (kind, ", ".join(far), drumroles.GRID_LEAD_S * 1000))
    if write:
        base = Path(out) if out else stems_dir
        old = _read_json(base / "drums.json")
        (base / "drum_roles").mkdir(parents=True, exist_ok=True)
        for kind, role, audio, sr in clips:                       # the new clips first, the new record next, the old clips last
            rel = "drum_roles/%s.wav" % _slug(kind, role["name"])
            sf.write(str(base / rel), audio, sr, subtype="PCM_24")
            role["clip"] = rel
        record["seconds"], record["jobs"], record["stage_seconds"] = round(time.monotonic() - t_start, 1), workers, {
            k: round(v, 1) for k, v in stages.items()}
        tmp = base / ".drums.json.tmp"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, base / "drums.json")                      # a reader never sees half a drums.json
        _remove_old_clips(base, old, {r["clip"] for s in record["stems"].values() for r in s["roles"] if r["clip"]})
        record["written"] = str(base / "drums.json")
    else:
        for _kind, role, _a, _s in clips:
            role["clip_note"] = "not written (write=False)"
        record["seconds"], record["jobs"], record["stage_seconds"] = round(time.monotonic() - t_start, 1), workers, {
            k: round(v, 1) for k, v in stages.items()}
    return record


def _rclone(*args) -> str:
    try:
        return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        raise OSError("rclone %s failed: %s" % (args[0], (e.stderr or "").strip() or "exit %s" % e.returncode))


def _library_roles(stems_dir: Path, remote: str, tempo, t0, sections, out, write: bool, deadline, jobs, hit_times=False) -> dict:
    """`drum_roles` for a study on the pCloud mount, which is never read or written through the FUSE mount (that fills a local
    cache with audio and writes into the archive through a mount): the drum stems, alignment.json and sections.json are copied
    to a temp folder with rclone, analysed there, and the results go back with rclone: the new clips, then drums.json, then
    the deletion of the clips the previous drums.json listed and the new one does not (nothing else is ever removed). The temp
    folder is deleted. With `out` the results go to that local folder and nothing is uploaded; with `write=False`, nowhere."""
    names = _rclone("lsf", "--files-only", remote).splitlines()
    wanted = [n for n in names if (n.startswith("drums_") and Path(n).suffix in AUDIO and not n.startswith("."))
              or n in ("alignment.json", "sections.json")]
    with tempfile.TemporaryDirectory(prefix="roadie-drums-") as tmp:
        tmp = Path(tmp)
        for n in wanted:
            _rclone("copyto", "%s/%s" % (remote, n), str(tmp / n))
        old = None
        if "drums.json" in names:
            try:
                old = json.loads(_rclone("cat", "%s/drums.json" % remote))
            except ValueError:
                old = None
        rec = drum_roles(tmp, tempo, t0, sections, out, write, deadline, jobs, hit_times)
        if write and not out:
            clips = sorted({r["clip"] for s in rec["stems"].values() for r in s["roles"] if r["clip"]})
            for c in clips:
                _rclone("copyto", str(tmp / c), "%s/%s" % (remote, c))
            _rclone("copyto", str(tmp / "drums.json"), "%s/drums.json" % remote)
            for c in sorted(_old_clips(old) - set(clips)):
                _rclone("deletefile", "%s/%s" % (remote, c))
            rec["written"] = "%s/drums.json" % remote
    return rec


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


def _old_clips(old) -> set[str]:
    """The clips an earlier drums.json record lists (paths under drum_roles/)."""
    if not isinstance(old, dict) or old.get("kind") != KIND:
        return set()
    return {r["clip"] for s in (old.get("stems") or {}).values() for r in s.get("roles", [])
            if isinstance(r.get("clip"), str) and r["clip"].startswith("drum_roles/") and ".." not in r["clip"]}


def _remove_old_clips(base: Path, old, keep: set) -> None:
    """The clips the earlier drums.json listed and the new one does not."""
    for clip in _old_clips(old) - keep:
        (base / clip).unlink(missing_ok=True)


def _slug(kind: str, name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s if s.startswith(kind) else "%s-%s" % (kind, s)


def _late_ms(hits: list) -> float | None:
    """Where these hits sit against the ALIGNER's grid: the median of their raw offsets (ms, positive: late, negative: early),
    whatever the section phase or the 20 ms tolerance say. None without hits."""
    off = [h["offset_raw_ms"] for h in hits if "offset_raw_ms" in h]
    return round(float(np.median(off)), 1) if off else None


def _own_line(hits: list, bars: int | None = None) -> tuple[list, int]:
    """(share of the bars the hits are in that have an on-grid hit at each 16th, number of those bars)."""
    own = {h["bar"] for h in hits}
    on = [h for h in hits if h["on_grid"]]
    return [len({h["bar"] for h in on if h["step"] == s}) / len(own) if own else 0.0 for s in range(16)], len(own)


def _for_naming(res: dict) -> dict:
    """`res` without the groups that are an echo of another: they take no name and take no part in the naming of the others."""
    out = dict(res)
    out["groups"] = [g for g in res["groups"] if g.get("echo_of") is None]
    return out


def _with_grid_tol(res: dict, tol: float) -> dict:
    """`res` with each hit's `on_grid` judged against `tol` seconds instead of the core's 20 ms (groups are unchanged: only the
    patterns the naming rules read move)."""
    out = dict(res)
    out["hits"] = [dict(h, on_grid=abs(h["offset_ms"]) <= tol * 1000.0) for h in res["hits"]]
    return out


def _steadiness(role_groups: list, name: str, named: bool, variants: tuple) -> str | None:
    """None when the role keeps its name (or stays neutral) under both on-grid tolerances, else a sentence saying what changes."""
    seen = []
    for label, v in zip(("tighter", "looser"), variants):
        other = next((x for x in v["roles"] if sorted(x["groups"]) == sorted(role_groups)), None)
        got = other["name"] if other and other["named"] else None
        if (got is not None) != named or (named and got != name):
            seen.append("%s tolerance (%+d ms): %s" % (label, -3 if label == "tighter" else 3, "named %r" % got if got else "not named"))
    return None if not seen else "the name depends on hits near the 20 ms on-grid edge: with a " + "; with a ".join(seen)


def _stem_record(kind: str, path: Path, res: dict, named: dict, bleed: dict, plan: list, t0: float, bar_s: float,
                 hit_times: bool = False, variants: tuple | None = None):
    by_id = {g["id"]: g for g in res["groups"]}
    info = {g["id"]: g for g in named["groups"]}
    hits = res["hits"]

    def in_section(h, p):
        return (p["first"] or h["bar"] >= p["start_bar"]) and (p["to_end"] or h["bar"] < p["start_bar"] + p["bars"])

    def section_view(ids):
        mine = [hits[i] for i in ids]
        rows = []
        for i, p in enumerate(plan):
            hs = [h for h in mine if in_section(h, p)]
            if hs:
                pat, nb = _own_line(hs)
                rows.append({"index": i, "section": p["name"], "start_bar": p["start_bar"], "bars": p["bars"], "hits": len(hs), "bars_with": nb,
                             "pattern": [round(float(v), 3) for v in pat], "line": pattern_line(pat), "late_ms": _late_ms(hs)})
        return rows

    group_role = {gid: r["name"] for r in named["roles"] for gid in r["groups"]}
    echoes = []
    for g in res["groups"]:
        if g.get("echo_of") is not None:
            parent = group_role.get(g["echo_of"], "%s %d" % (kind, g["echo_of"] + 1))
            echoes.append({"name": "echo of %s" % parent, "named": False, "rule": None, "groups": [g["id"]], "count": g["count"],
                           "assigned": g["assigned"], "level_db": g["level_db"], "pattern": g["pattern"], "note": None, "variants": None,
                           "possibly_also": [], "echo_of": parent, "echo_ms": g["echo_ms"], "echo_bars": g["echo_bars"]})
    roles, clips = [], []
    for r in named["roles"] + echoes:
        big = by_id[max(r["groups"], key=lambda i: by_id[i]["count"])]
        e = info.get(big["id"]) or {"reason": "an echo of %s: its hits follow that role's by %.0f ms" % (r.get("echo_of"), r.get("echo_ms") or 0)}
        clip = big["clean_hit"]
        ids = [i for gid in r["groups"] for i in by_id[gid]["members"] + by_id[gid]["assigned_members"]]
        own, nb = _own_line([hits[i] for i in ids])
        rec = {"name": r["name"], "named": r["named"], "rule": r["rule"], "reason": None if r["named"] else e["reason"],
               "hits": r["count"] + r["assigned"], "measured": r["count"], "assigned": r["assigned"],
               "bars_with": nb, "pattern": [round(float(v), 3) for v in r["pattern"]],
               "own_pattern": [round(float(v), 3) for v in own], "line": pattern_line(own),
               "off_grid": sum(by_id[i]["off_grid"] for i in r["groups"]),
               "late_ms": _late_ms([hits[i] for i in ids]),
               "level_db": None if r["level_db"] is None else round(float(r["level_db"]), 1),
               "stability": None if big["stability"] is None else round(float(big["stability"]), 2),
               "stability_status": big["stability_status"],
               "separation": None if big["separation"] is None else round(float(big["separation"]), 2),
               "groups": r["groups"], "note": r["note"], "variants": r["variants"],
               "possibly_also": r["possibly_also"], "clip": None,
               "clip_note": None if clip else big["clean_note"], "per_section": section_view(ids)}
        merged = [m for gid in r["groups"] for m in (by_id[gid].get("merged_from") or [])]
        if merged:                                     # one sound the grouping had split: where each part plays is arrangement information
            rec["merged_from"] = merged
        if r.get("echo_of") is not None:
            rec["echo_of"], rec["echo_ms"], rec["echo_bars"] = r["echo_of"], r["echo_ms"], r["echo_bars"]
        maybe = sorted({group_role[og] for gid in r["groups"] for og in by_id[gid].get("may_be_one_sound_with", [])
                        if og in group_role and group_role[og] != r["name"]})
        if maybe and r.get("echo_of") is None:
            rec["may_be_one_sound_with"], rec["may_be_note"] = maybe, "not compared: no clean hit on one side"
        if variants and r.get("echo_of") is None:
            why = _steadiness(r["groups"], r["name"], r["named"], variants)
            rec["name_steady"], rec["name_steady_note"] = why is None, why
        if hit_times:                                  # onset seconds of the role's hits, for roadie_analysis/drumstability.py
            rec["hit_times"] = sorted(round(hits[i]["onset_s"], 4) for i in ids)
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
    shifts = res.get("section_shift_ms") or [0.0] * len(plan)
    per_section = [{"name": p["name"], "start_bar": p["start_bar"], "bars": p["bars"],      # by index: labels repeat (two "Drop")
                    "grid_shift_ms": shifts[i],      # where this stem's hits sit against the grid in this section (median)
                    "roles": {r["name"]: v["hits"] for r in roles for v in r["per_section"] if v["index"] == i}}
                   for i, p in enumerate(plan)]
    rec = {"file": path.name, "sample_rate": res["sample_rate"], "hits": sum(h["owned"] for h in hits),
           "hat_like": res["hat_like"], "bars": res["bars"], "first_bar": res["first_bar"], "roles": roles, "bleed": marked,
           "per_section": per_section, "notes": res["notes"] + named["notes"]}
    return rec, clips


def _bars_text(runs: list) -> str:
    """[[3, 9], [12, 12]] as "3-9, 12" (file bars counted from bar 0)."""
    return ", ".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in runs)


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
            if r.get("late_ms") is not None:
                flags.append("late %+.0f ms" % r["late_ms"])
            if r["variants"]:
                flags.append("%d variants" % len(r["variants"]))
            if r.get("name_steady") is False:
                flags.append("name unsteady")
            if r.get("echo_of"):
                flags.append("echo of %s, %+.0f ms later" % (r["echo_of"], r["echo_ms"]))
            if r.get("merged_from"):
                flags.append("%d groups merged" % len(r["merged_from"]))
            if r.get("may_be_one_sound_with"):
                flags.append("may be one sound with %s (%s)" % (", ".join(r["may_be_one_sound_with"]), r["may_be_note"]))
            lvl = "%.1f dB" % r["level_db"] if r["level_db"] is not None else "-"
            lines.append("  %-14s %5d  %-19s %7s  %s" % (r["name"], r["hits"], " ".join(r["line"][i:i + 4] for i in range(0, 16, 4)),
                                                          lvl, ", ".join(flags)))
            if r["reason"]:
                lines.append("      why neutral: %s" % r["reason"])
            if r["note"]:
                lines.append("      note: %s" % r["note"])
            if r.get("name_steady") is False:
                lines.append("      unsteady: %s" % r["name_steady_note"])
            for m in r.get("merged_from") or []:
                lines.append("      merged part: %d hits in bars %s" % (m["hits"], _bars_text(m["bars"])))
            if r.get("echo_bars"):
                lines.append("      echo is on in bars %s" % _bars_text(r["echo_bars"]))
            for v in r["per_section"] if len(r["per_section"]) > 1 else []:
                lines.append("      %-22s %4d  %s" % (v["section"][:22], v["hits"], " ".join(v["line"][i:i + 4] for i in range(0, 16, 4))))
            if r["clip"]:
                lines.append("      clip: %s" % r["clip"])
        shifts = ["%s %+.1f" % (ps["name"][:14], ps["grid_shift_ms"]) for ps in s["per_section"] if ps.get("grid_shift_ms")]
        if shifts:
            lines.append("  section phase against the aligner's grid (ms): " + ", ".join(shifts))
        if small:
            lines.append("  + %d groups of fewer than %d measured hits (%d hits in all): in drums.json" % (
                len(small), MIN_SHOWN, sum(r["hits"] for r in small)))
        for b in s["bleed"]:
            lines.append("  BLEED %-8s %5d  %-19s %7s  %s" % (b["name"], b["hits"], " ".join(b["line"][i:i + 4] for i in range(0, 16, 4)),
                                                              "%.1f dB" % b["level_db"] if b["level_db"] is not None else "-", b["why"]))
        lines += ["  " + n for n in s["notes"]]
    return "\n".join(lines)

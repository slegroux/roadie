"""A drum stem analysed section by section, then merged into one set of roles.

`hit_groups` on a whole track groups every hit of a stem at once and gets slower and less stable the longer the stem is
(1222 hats of Glacier 3000 gave eleven groups, none stable). Here each section of the track is cut out of the stem and
grouped on its own (`section_job`: few hits, so the grouping is exact and the jobs can run in parallel), then
`merge_stem` matches the groups of all sections by sound and builds ONE result shaped like `hit_groups`'s, with the
matched groups as its groups, so `mark_bleed` and `name_roles` run on the merged role's hits, not on a section's.

Choices:

- A section shorter than MIN_SECTION_BARS joins the next one (the last joins the one before): too few hits to group.
  The first section starts at bar 0 and the last runs to the end of the file, so no hit is outside every section.
- A section's slice runs from PRE_S before its first bar line to POST_S after its last, so a hit on the line has a lead
  and a hit just before the line is whole. A hit belongs to the section whose bar lines enclose it after moving them OWN_S
  (the grid tolerance, 20 ms) earlier: a hit 8 ms early on the first beat of a section is that section's. Hits in the
  overlap of two slices are kept from one only.
- Matching: the clean hit of each group (else the cut of the member nearest its medoid) is compared with every other
  group's with `spectral_distance` over the shorter cut, as the core does, and the groups are merged by average linkage
  at the core's THRESHOLD, with one rule added: two groups of one section never merge (a pitch split inside a section must
  survive). `separation` of a merged role is the mean of those distances to the nearest other role of at least 4 hits.
- `stability` of a merged role is the hit-weighted mean of its groups' measured stabilities; with none measured it is
  "stable by construction" when a group says so (a section of at most 32 hits is grouped exactly), else not measured.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from roadie_analysis import drumroles, oneshot

MIN_SECTION_BARS = 8
PRE_S = 0.040
POST_S = 0.060
OWN_S = 0.020
STEPS = 16


def plan_sections(sections: list | None) -> list[dict]:
    """[{name, start_bar, bars}] with short sections joined (see the module docstring); `bars` of the last is its own and
    the plan marks it `to_end`. None or empty: one section, the whole track."""
    if not sections:
        return [{"name": "whole track", "start_bar": 0, "bars": None, "first": True, "to_end": True}]
    secs = [dict(name=s.get("name"), start_bar=s["start_bar"], bars=s["bars"]) for s in sorted(sections, key=lambda s: s["start_bar"])]
    i = 0
    while i < len(secs) and len(secs) > 1:
        if secs[i]["bars"] >= MIN_SECTION_BARS:
            i += 1
            continue
        j = i + 1 if i + 1 < len(secs) else i - 1
        a, b = (secs[i], secs[j]) if j > i else (secs[j], secs[i])
        merged = {"name": "%s + %s" % (a["name"], b["name"]), "start_bar": a["start_bar"],
                  "bars": b["start_bar"] + b["bars"] - a["start_bar"]}
        secs[min(i, j):max(i, j) + 1] = [merged]
        i = min(i, j)
    secs[0]["bars"] += secs[0]["start_bar"]
    secs[0]["start_bar"] = 0
    for k, s in enumerate(secs):
        s["first"], s["to_end"] = k == 0, k == len(secs) - 1
    return secs


def section_job(path, tempo: float, t0: float, sec: dict) -> dict:
    """hit_groups on one section of one stem file. Returns {"res", "offset_s" (where the slice starts in the file), "sr",
    "reps" (group id -> the audio that stands for the group when sections are matched)}, or {"res": None} when the
    section holds no audio."""
    info = sf.info(str(path))
    sr, bar_s = info.samplerate, 60.0 / tempo * 4
    start_t = t0 + sec["start_bar"] * bar_s
    a = 0 if sec["first"] else max(0, int(round((start_t - PRE_S) * sr)))
    b = info.frames if sec["to_end"] else min(info.frames, int(round((start_t + sec["bars"] * bar_s + POST_S) * sr)))
    if b <= a:
        return {"res": None, "sr": sr}
    x, _ = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
    res = drumroles.hit_groups(x, tempo, start_t - a / sr, None, sr=sr)
    xu = oneshot._undc(np.asarray(x, dtype=np.float64), sr)
    reps = {}
    for g in res["groups"]:
        if g["clean_hit"] is not None:
            reps[g["id"]] = np.asarray(g["clean_hit"]["audio"])
        else:
            h = res["hits"][min(g["members"], key=lambda i: res["hits"][i]["distance"])]
            reps[g["id"]] = xu[h["start"]:h["end"]]
    return {"res": res, "offset_s": a / sr, "sr": sr, "reps": reps}


def _owned(h_time: float, plan: dict, bar_s: float, t0: float) -> bool:
    lo = -np.inf if plan["first"] else t0 + plan["start_bar"] * bar_s - OWN_S
    hi = np.inf if plan["to_end"] else t0 + (plan["start_bar"] + plan["bars"]) * bar_s - OWN_S
    return lo <= h_time < hi


def merge_stem(outs: list, plan: list, tempo: float, t0: float, threshold: float = drumroles.THRESHOLD) -> dict:
    """One `hit_groups`-shaped result for a stem from its sections' `section_job` outputs (aligned with `plan`). Bars are
    the track's (bar 0 at `t0`); every hit of every slice is in `hits` (`owned` says whether the section keeps it), and
    only owned hits are in a group. `sections_of` gives, per group, its section index and local id."""
    bar_s = 60.0 / tempo * 4
    live = [(si, o) for si, o in enumerate(outs) if o["res"] is not None]
    sr = live[0][1]["sr"] if live else 44100
    out = {"sample_rate": sr, "tempo": tempo, "t0": t0, "threshold": threshold, "linkage": "average", "max_cut_s": drumroles.MAX_CUT_S,
           "grid_onsets": False, "hat_like": None, "grid_hits": 0, "hits": [], "groups": [], "unmeasured_pattern": [],
           "hidden_pattern": [], "unmeasured": [], "sections": [], "notes": [], "bars": 0, "first_bar": 0, "hidden_cells": [],
           "unmeasured_cells": [], "hidden_tail_db": [], "unmeasured_tail_db": []}
    if not live:
        out["notes"].append("the stem holds no audio in any section")
        return out
    hits, idmap = [], {}
    for si, o in live:
        res, p = o["res"], plan[si]
        for h in res["hits"]:
            g = dict(h)
            g["id"] = idmap[(si, h["id"])] = len(hits)
            g["bar"] = h["bar"] + p["start_bar"]
            g["onset_s"] = h["onset_s"] + o["offset_s"]
            g["owned"] = _owned(g["onset_s"], p, bar_s, t0)
            g["group"], g["section"] = None, si
            hits.append(g)
        if res["hat_like"] is not None:
            out["hat_like"] = max(out["hat_like"] or 0.0, res["hat_like"])
        out["notes"] += [n for n in res["notes"] if n not in out["notes"]]
    out["hits"] = hits
    entries = [(si, g) for si, o in live for g in o["res"]["groups"]]
    n = len(entries)
    raw = np.zeros((n, n))
    reps = [outs[si]["reps"][g["id"]] for si, g in entries]
    for i in range(n):
        for j in range(i + 1, n):
            raw[i, j] = raw[j, i] = drumroles._pair_distance(reps[i], reps[j], sr)
    cut = raw.copy()
    for i in range(n):
        for j in range(n):
            if i != j and entries[i][0] == entries[j][0]:
                cut[i, j] = 1e6                                   # two groups of one section are two sounds
    clusters = drumroles._agglomerate(cut, threshold) if n else []

    def owned_ids(si, ids):
        return [idmap[(si, i)] for i in ids if hits[idmap[(si, i)]]["owned"]]

    built = []
    for cl in clusters:
        mem, asg = [], []
        for k in cl:
            si, g = entries[k]
            mem += owned_ids(si, g["members"])
            asg += owned_ids(si, g["assigned_members"])
        if mem or asg:
            built.append((sorted(mem), sorted(asg), cl))
    built.sort(key=lambda b: (-len(b[0]), (b[0] + b[1])[0]))
    real = [h["bar"] for h in hits if h["owned"] and h["status"] not in ("faint", "silent", "no_lead")]
    first, last = (min(real), max(real)) if real else (0, 0)
    nbars = last - first + 1
    for gid, (mem, asg, cl) in enumerate(built):
        for i in mem + asg:
            hits[i]["group"] = gid
        on = [hits[i] for i in mem + asg if hits[i]["on_grid"]]
        lv = [hits[i]["level_db"] for i in mem if hits[i]["level_db"] is not None]
        parts = sorted(cl, key=lambda k: -len(owned_ids(entries[k][0], entries[k][1]["members"])))
        stabs = [(len(owned_ids(entries[k][0], entries[k][1]["members"])), entries[k][1]["stability"]) for k in parts
                 if entries[k][1]["stability"] is not None]
        if stabs:
            stab, status = round(float(sum(c * s for c, s in stabs) / max(1, sum(c for c, _ in stabs))), 3), "measured"
        else:
            stab = None
            status = "stable by construction" if any(entries[k][1]["stability_status"] == "stable by construction" for k in parts) else "not measured"
        clean, note = None, None
        for k in parts:
            si, g = entries[k]
            if g["clean_hit"] is not None:
                clean = dict(g["clean_hit"], hit=idmap[(si, g["clean_hit"]["hit"])])
                break
        if clean is None:
            note = entries[parts[0]][1]["clean_note"] or "no clean hit"
        pools = [[dict(m, hit=idmap[(entries[k][0], m["hit"])]) for m in entries[k][1].get("read_members", [])] for k in parts]
        read = []
        for r in range(drumroles.AGREE_MEMBERS):                   # round robin over the sections, so one section does not decide
            for pool in pools:
                if r < len(pool) and len(read) < drumroles.AGREE_MEMBERS:
                    read.append(pool[r])
        out["groups"].append({
            "id": gid, "count": len(mem), "assigned": len(asg), "tentative": len(mem) < drumroles.TENTATIVE_N,
            "stability": stab, "stability_status": status, "separation": None, "nearest_group": None, "separation_any": None,
            "nearest_any": None, "bars_with": len({hits[i]["bar"] for i in mem + asg}),
            "pattern": [len({h["bar"] for h in on if h["step"] == s}) / nbars if nbars else 0.0 for s in range(STEPS)],
            "level_db": float(np.median(lv)) if lv else None, "off_grid": len(mem) + len(asg) - len(on),
            "members": mem, "assigned_members": asg, "clean_hit": clean, "clean_note": note, "read_members": read,
            "sections_of": [(entries[k][0], entries[k][1]["id"]) for k in cl]})
    for a, ga in enumerate(out["groups"]):
        rows = []
        for b, gb in enumerate(out["groups"]):
            if b != a:
                d = float(np.mean([raw[i, j] for i in built[a][2] for j in built[b][2]]))
                rows.append((d, b, gb["count"] >= drumroles.TENTATIVE_N))
        if rows:
            d, b, _ = min(rows)
            ga["separation_any"], ga["nearest_any"] = d, b
        big = [r for r in rows if r[2]]
        if big:
            d, b, _ = min(big)
            ga["separation"], ga["nearest_group"] = d, b
    # hidden and unmeasured cells, on the track's bars, from the bars each section owns
    for si, o in live:
        res, p = o["res"], plan[si]
        for name, tails in (("hidden_cells", "hidden_tail_db"), ("unmeasured_cells", "unmeasured_tail_db")):
            for (b, s), db in zip(res[name], res[tails]):
                if (p["first"] or b >= 0) and (p["to_end"] or b < p["bars"]):
                    out[name].append([b + p["start_bar"], s])
                    out[tails].append(db)
    out.update({"bars": nbars, "first_bar": first})
    return out

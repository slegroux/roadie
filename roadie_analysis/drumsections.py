"""A drum stem analysed section by section, then merged into one set of roles.

`hit_groups` on a whole track groups every hit of a stem at once and gets slower and less stable the longer the stem is
(1222 hats of Glacier 3000 gave eleven groups, none stable). Here each section of the track is cut out of the stem and
grouped on its own (`section_job`: few hits, so the grouping is exact and the jobs can run in parallel), then
`merge_stem` matches the groups of all sections by sound and builds ONE result shaped like `hit_groups`'s, with the
matched groups as its groups, so `mark_bleed` and `name_roles` run on the merged role's hits, not on a section's.

Choices:

- A section shorter than MIN_SECTION_BARS joins the next one (the last joins the one before): too few hits to group.
  The first section starts at bar 0 and the last runs to the end of the file, so no hit is outside every section.
- A section is judged against the whole stem's loudest 5 ms (`stem_peak`, passed to `hit_groups` as `ref_power`), with the
  core's dB figures unchanged: a section where the stem is near silent has no hits, as in the whole-track run.
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

import math

import numpy as np
import soundfile as sf

from roadie_analysis import drumroles, oneshot

MIN_SECTION_BARS = 8
PRE_S = 0.040
POST_S = 0.060
OWN_S = 0.020
STEPS = 16
QUANTUM_S = 0.45         # a slice starts and ends on a multiple of about this many seconds (see section_job)
SHIFT_WINDOW_MS = 3.0     # ...from the densest cluster of offsets, this wide either side
SHIFT_SUPPORT = 0.4      # ...and the densest window must hold at least this share of them (40 %: a cluster that is not a minority of two)
SHIFT_DRIFT_MS = 15.0    # a section's phase may differ from the track-wide one by this much (the drift of 0.005 bpm over a track is 13 ms)
SHIFT_MIN_HITS = 8       # a section's grid phase is read from its hits' densest offsets when it has at least this many


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


def stem_peak(path, block_s: float = 10.0) -> float:
    """The peak power of the whole stem's 5 ms envelope (what hit detection measures levels against), so a section is judged
    against the stem's loudest moment and not its own. The same number as `oneshot.envelope_peak(oneshot._undc(x, sr), sr)`
    on the whole stem, read in blocks (two passes: the DC offset from the 10 ms blocks' statistics, then the envelope with a
    5 ms overlap), so a stem is never held whole in memory."""
    with sf.SoundFile(str(path)) as f:
        sr, ch, total = f.samplerate, f.channels, f.frames
        n10 = max(1, int(round(0.010 * sr)))
        step = n10 * max(1, int(block_s / 0.010))
        peak, nz_lo, nz_hi, live, var, mean = 0.0, None, None, [], [], []
        for start in range(0, total, step):                  # pass 1: what oneshot._dc reads
            f.seek(start)
            x = f.read(min(step, total - start), dtype="float64", always_2d=True)
            if not len(x):
                break
            mag = np.abs(x)
            peak = max(peak, float(mag.max()))
            nz = np.nonzero(mag.max(axis=1) > 0)[0]
            if len(nz):
                nz_lo = start + int(nz[0]) if nz_lo is None else nz_lo
                nz_hi = start + int(nz[-1])
            m = len(x) // n10
            if m:
                blocks = x[:m * n10].reshape(m, n10, ch)
                live.append(np.abs(blocks).max(axis=(1, 2)) > 0)
                var.append(blocks.var(axis=1))
                mean.append(blocks.mean(axis=1))
        dc = np.zeros(ch)
        live = np.concatenate(live) if live else np.zeros(0, dtype=bool)
        if len(live) >= 3 and live.sum() >= 3:
            var, mean = np.concatenate(var)[live], np.concatenate(mean)[live]
            quiet = np.argsort(var.sum(axis=1))[:max(3, len(var) // 10)]
            q = mean[quiet]
            dcm = q.mean(axis=0)
            spread = np.sqrt(var[quiet]).max(axis=0)
            real = ((spread < np.abs(dcm)) | (spread < 0.01 * peak)) & (q.std(axis=0) < 0.2 * np.abs(dcm))
            dc = np.where(real, dcm, 0.0)
        use_dc = nz_lo is not None and np.abs(dc).max() > oneshot.DC_REL * peak
        L, hop = max(1, int(0.005 * sr)), max(1, int(round(0.001 * sr)))
        out, own = 0.0, hop * 1000
        for start in range(0, total, own):                   # pass 2: the 5 ms envelope on a 1 ms grid, with L samples of context
            lo, hi = max(0, start - L), min(total, start + own + L)
            f.seek(lo)
            x = f.read(hi - lo, dtype="float64", always_2d=True)
            if use_dc:
                a, b = max(nz_lo, lo), min(nz_hi + 1, hi)
                if b > a:
                    x[a - lo:b - lo] -= dc
            c = np.concatenate([[0.0], np.cumsum((x ** 2).mean(axis=1))])
            i = np.arange(start, min(total, start + own), 1)
            i = i[i % hop == 0]
            a = np.clip(i - L // 2, 0, total)
            b = np.clip(i - L // 2 + L, 0, total)
            out = max(out, float(((c[b - lo] - c[a - lo]) / np.maximum(b - a, 1)).max()))
    return out


def slice_quantum(sr: int) -> int:
    """Samples a slice start and end are rounded to: a multiple of lcm(1 ms, 10 ms in samples), the grids the detector and the DC
    offset are read on, about QUANTUM_S long at every sample rate (19,404 at 44.1 kHz, 21,600 at 48 kHz)."""
    base = math.lcm(max(1, int(round(0.001 * sr))), max(1, int(round(0.010 * sr))))
    return base * max(1, int(round(QUANTUM_S * sr / base)))


def section_job(path, tempo: float, t0: float, sec: dict, ref_power: float | None = None) -> dict:
    """hit_groups on one section of one stem file. Returns {"res", "offset_s" (where the slice starts in the file), "sr",
    "reps" (group id -> the audio that stands for the group when sections are matched)}, or {"res": None} when the
    section holds no audio."""
    info = sf.info(str(path))
    sr, bar_s = info.samplerate, 60.0 / tempo * 4
    start_t = t0 + sec["start_bar"] * bar_s
    # The slice starts and ends on a multiple of `quantum` samples: the detector reads the audio on a 1 ms grid and the DC
    # offset on 10 ms blocks, both anchored at the slice start, so a slice that begins a few samples later (a grid that moved
    # by 0.001 bpm moves a section line by about a millisecond) finds hits a few samples apart and groups them differently.
    # With a fixed anchor the same audio gives the same hits and groups whatever the grid says.
    quantum = slice_quantum(sr)
    a = 0 if sec["first"] else max(0, int((start_t - PRE_S) * sr) // quantum * quantum)
    b = info.frames if sec["to_end"] else min(info.frames, -(-int((start_t + sec["bars"] * bar_s + POST_S) * sr) // quantum) * quantum)
    if b <= a:
        return {"res": None, "sr": sr}
    x, _ = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
    res = drumroles.hit_groups(x, tempo, start_t - a / sr, None, sr=sr, ref_power=ref_power)
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


def _dense_offset(off: np.ndarray, width_ms: float = SHIFT_WINDOW_MS) -> tuple[float, int]:
    """(where most of the hits sit, how many hits are there): the offset (ms) with the most hits within `width_ms` either side
    and the median of those hits, with their number. Not the median of all of them: a stem with two sounds that are detected
    at different latencies has a median between them that is neither. It moves with the grid exactly (every offset shifts by
    the same amount), which is what makes the counts that follow steady."""
    count = np.array([(np.abs(off - x) <= width_ms).sum() for x in off])
    centre = off[int(np.argmax(count))]
    near = off[np.abs(off - centre) <= width_ms]
    return float(np.median(near)), int(len(near))


def _phase(off: list) -> float | None:
    """The phase of a set of hit offsets (ms), or None when there is no cluster to read it from: at least SHIFT_MIN_HITS hits,
    and the densest window must hold at least SHIFT_SUPPORT of them (scattered hits have no phase)."""
    if len(off) < SHIFT_MIN_HITS:
        return None
    centre, n = _dense_offset(np.array(off))
    return centre if n >= SHIFT_SUPPORT * len(off) else None


def _rephase(hits: list, plan: list, tempo: float, t0: float) -> list:
    """Measure every hit against its section's phase of the grid, and say what the phase was. The stem's track-wide phase (where
    most of all its owned hits sit; SHIFT_SUPPORT of them must sit there, else 0) is the default. A section takes its own phase
    (the same reading over its own hits) only when that is within SHIFT_DRIFT_MS of the track-wide one: the drift of a grid that
    is a few thousandths of a bpm off is about 13 ms over a track, and a section may follow it. Where a section's own phase is
    further away, or has no cluster, it falls back to the track-wide one: a part that plays deliberately late stays late
    however many of its section's hits it makes. `bar`, `step`, `offset_ms` and `on_grid` (within drumroles.GRID_LEAD_S, the
    core's 20 ms, unchanged) are recomputed against the lines moved by the phase; what a hit is measured from is the only thing
    that changes, so counts at the edge of the tolerance no longer depend on where the grid happens to lie. The unshifted offset
    stays in `offset_raw_ms` (against the aligner's grid). Returns the shift of each section (ms, by plan index)."""
    step_s = 60.0 / tempo / 4
    usable = [h for h in hits if h["owned"] and h["level_db"] is not None]
    default = _phase([h["offset_ms"] for h in usable])
    default = 0.0 if default is None else default
    shifts = []
    for si in range(len(plan)):
        own = _phase([h["offset_ms"] for h in usable if h["section"] == si])
        shifts.append(own if own is not None and abs(own - default) <= SHIFT_DRIFT_MS else default)
    for h in hits:
        h["offset_raw_ms"] = h["offset_ms"]
        rel = h["onset_s"] - (t0 + shifts[h["section"]] / 1000.0)
        line = int(round(rel / step_s))
        h["bar"], h["step"] = divmod(line, STEPS)
        off = rel - line * step_s
        h["offset_ms"] = round(1000.0 * off, 2)
        h["on_grid"] = abs(off) <= drumroles.GRID_LEAD_S
    return [round(x, 2) for x in shifts]


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
    out["section_shift_ms"] = _rephase(hits, plan, tempo, t0)
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

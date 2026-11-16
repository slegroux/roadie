#!/usr/bin/env python3
"""Score Roadie's drum roles against the true drum sources of a remix pack.

    bench_roles.py bench/synthwave1.json [--set separated] [--set true] [--set name=DIR]
                                         [--jobs 4] [--average] [--rerun] [--out FILE.json]

`hit_groups`, `mark_bleed` and `name_roles` (roadie_analysis.drumroles) run on each drum
stem of a set; this scores the roles they find against the sources bench_truth.py wrote to
`truth_sources/` (one file per source the manifest gives a `drum_source`). Sets:

    separated   the stems yt2stems.sh -d wrote for the mix (the manifest's out_dir/separated, the one
                *_stems folder in it)
    true        the true role stems (out_dir/truth/drums_*.flac): the ceiling, what the same rules
                give when the separation is perfect (several sources still share a role stem:
                SNARE, SNARE2 and CLAP are one drums_snare, four sources are one drums_hh)
    NAME=DIR    any other folder of drums_*.flac

Per true source (kick, snare, snare2, clap, chh1, chh2, chh3, shaker, perc1, metal_hit), per set:

    best role   among the roles with a group that sounds like the source (a hit within SOUND_MAX = 2.0
                of the reference, drumroles' own "sound alike" distance), the one whose hits match the
                source's onsets best: F1 of recall (share of the source's onsets with a role hit within
                TOL_S) and precision (share of the role's hits within TOL_S of one of the source's onsets).
                A hit and an onset pair one to one (a greedy two-pointer matching, optimal in one
                dimension). Onsets alone cannot tell sources that play on the same steps apart (a closed hat
                on every quarter note lands on the kick), hence the sound condition. When no role sounds
                like the source the plain best F1 is shown, marked `?`, and `timing_best` in the JSON says
                what timing alone would have picked. A role of sources that play together (SNARE with CLAP)
                scores full precision for each, since its hit is at each one's onset.
    detected    recall over every hit the role's stem found, grouped or not: what the grouping and the
                measurement rules lose between "the onset was seen" and "it is in a role"
    name        the role's name (a neutral `<stem> N` when no rule fits; the reason is in the JSON)
    dist        `oneshot.spectral_distance` from the nearest of the role's groups' hits to the true sample: one
                clean hit read from the source file. `d main`: the same for the role's biggest group. 0 is the
                same spectrum; two takes of one noise hat read 0.55-0.82; the grouping threshold is 1.5. A
                group with no complete member (every tail cut by the next hit) is shown by its medoid's cut,
                marked `*`.
    closest     the group of any stem whose hit is nearest the true sample: where the sound is, when the best
                role is not it

True onsets are the source file's own, found by the same detector the roles use
(oneshot._segments and the onset reading hit_groups makes), at a tolerance of TOL_S (25 ms,
drumroles.COINCIDE_S, the tolerance its bleed test counts a coincidence at).

--average (mitigation (a)): for each source's best role, every measured hit of its biggest
group is cut from the stem at its onset and the cuts are averaged (mean and median, sample
by sample over the hits that cover it, at least 3 and a quarter of them), once as detected
and once with each hit moved up to AVG_LAG_S to line up with the clean hit. Distance of the
mean, the median and the single clean hit to the true sample is the verdict's number.

Results are cached per (set, stem) under out_dir/roles/ (the grouping of a whole hat stem
takes minutes); --rerun redoes them. No pack audio is read or written here beyond
truth_sources/, which bench_truth.py wrote. Local only.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

DRUM_STEMS = ("drums_kick", "drums_snare", "drums_toms", "drums_hh", "drums_ride", "drums_crash")
KIND = {"drums_kick": "kick", "drums_snare": "snare", "drums_toms": "toms", "drums_hh": "hh"}   # name_roles' stem kinds
TOL_S = 0.025
SOUND_MAX = 2.0         # a role "sounds like" a source when one of its groups' hits is within this spectral distance of the source's reference hit (drumroles.BLEED_DISTANCE: the module's own "sound alike")
REF_BARS = 32           # the reference clean hit is read from this many bars after a source's first hit (a source with fewer hits: all)
REF_MAX_HITS = 200
AVG_MAX_HITS = 200      # a group's hits are averaged from at most this many, evenly spaced in time
AVG_MAX_S = 0.3         # the average is at most this long from the onset
AVG_MIN_HITS = 3
AVG_LAG_S = 0.001       # a hit may move this far to line up with the clean hit before it is averaged


# ---------------------------------------------------------------- pure scoring

def match_onsets(true_s, est_s, tol=TOL_S):
    """Number of onsets paired one to one within `tol` seconds. Both lists sorted by this function.
    Greedy from the earliest: in one dimension that is a maximum matching."""
    a, b = sorted(true_s), sorted(est_s)
    i = j = n = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            n, i, j = n + 1, i + 1, j + 1
        elif b[j] < a[i]:
            j += 1
        else:
            i += 1
    return n


def f1(recall, precision):
    return 0.0 if recall + precision <= 0 else 2 * recall * precision / (recall + precision)


def role_onsets(result, role):
    """Onsets (seconds) of the hits that sit in a role's groups: measured ones and the ones assigned under a tail."""
    ids = set(role["groups"])
    return sorted(h["onset_s"] for h in result["hits"] if h["group"] in ids and h["status"] in ("grouped", "assigned"))


def main_group(result, role, any_group=False):
    """The role's biggest group that has a clean hit (its measured members, the most), else None; with any_group, the
    biggest group of the role whether or not it has one."""
    by_id = {g["id"]: g for g in result["groups"]}
    order = sorted(role["groups"], key=lambda i: -by_id[i]["count"])
    for gid in order:
        if by_id[gid]["clean_hit"] is not None:
            return by_id[gid]
    return by_id[order[0]] if any_group and order else None


def candidates(results, named):
    """Every role of every stem of a set: (stem, role, hit onsets)."""
    return [(stem, r, role_onsets(results[stem], r)) for stem in results for r in named[stem]["roles"]]


def best_role(true_s, cands, dist=None, sound_max=SOUND_MAX):
    """The candidate with the best F1 against the source's onsets (ties: the higher recall), among the roles that sound like
    the source when `dist(stem, role)` (the nearest of the role's groups to the source's reference hit, or None) is given and
    some role is within `sound_max`; the plain best F1 otherwise. Onsets alone cannot tell sources that play on the same
    steps apart: a closed hat on every quarter note lands on the kick's hits, which then match it with a precision of 0.8.
    Returns {"stem", "role", "tp", "hits", "recall", "precision", "f1", "sound": True (chosen among the roles that sound
    like it), False (none does: the plain best) or None (no `dist`), "timing_best": the plain best's (stem, name, recall,
    precision) when it differs}, or None when there are no candidates."""
    def score(c):
        stem, role, ons = c
        tp = match_onsets(true_s, ons)
        rec = tp / len(true_s) if len(true_s) else 0.0
        pre = tp / len(ons) if ons else 0.0
        return (f1(rec, pre), rec), {"stem": stem, "role": role, "tp": tp, "hits": len(ons), "recall": rec,
                                      "precision": pre, "f1": f1(rec, pre)}
    scored = [score(c) for c in cands]
    if not scored:
        return None
    plain = max(scored, key=lambda t: t[0])[1]
    pool, sound = scored, None
    if dist is not None:
        alike = [t for t, c in zip(scored, cands) if (d := dist(c[0], c[1])) is not None and d <= sound_max]
        pool, sound = (alike, True) if alike else (scored, False)
    best = max(pool, key=lambda t: t[0])[1]
    best["sound"] = sound
    best["timing_best"] = None if best is plain else {"stem": plain["stem"], "role": plain["role"]["name"],
                                                       "recall": plain["recall"], "precision": plain["precision"]}
    return best


# ---------------------------------------------------------------- audio side (needs the drum roles module)

def _undc(x, sr):
    """The DC removal hit_groups applies to a stem before it looks at it (a [n, channels] array)."""
    import numpy as np
    from roadie_analysis import oneshot
    x = np.asarray(x, dtype=np.float64)
    return oneshot._undc(x[:, None] if x.ndim == 1 else x, sr)


def true_onsets(x, sr):
    """The onsets of a clean source, found as hit_groups finds them (seconds)."""
    import numpy as np
    from roadie_analysis import drumroles, oneshot
    x = _undc(x, sr)
    if not len(x) or float(np.abs(x).max()) <= 0:
        return []
    out = []
    for s, e in oneshot._segments(x, sr):
        o = oneshot._hit_onset(x[s:e], sr) if s == 0 else drumroles._onset_after_tail(x, sr, s)
        out.append(o / sr)
    return sorted(out)


def medoid_cut(x, result, g):
    """The cut of a group's medoid hit (the measured member nearest the group's medoid, which is at distance 0 from
    itself): onset to 2 ms before the next hit (at most 0.5 s). `x` is the stem as hit_groups read it. It is what a
    group with no complete member (every tail cut off by the next hit) can be shown by."""
    by_id = {h["id"]: h for h in result["hits"]}
    h = by_id[min(g["members"], key=lambda i: (by_id[i]["distance"], i))]
    return x[h["start"]:h["end"]]


def group_audio(result, g, load_stem=None, stem=None):
    """(audio, clean) for a group: its clean hit, else, with `load_stem(stem)` to read the stem, its medoid's cut
    (clean False: the tail is cut by the next hit, so a distance to it also counts that). (None, False) when neither."""
    if g["clean_hit"] is not None:
        return g["clean_hit"]["audio"], True
    if load_stem is not None and g["members"]:
        return medoid_cut(load_stem(stem), result, g), False
    return None, False


def reference_hit(x, sr, onsets, tempo, t0):
    """One hit of a true source to measure the roles against: the clean hit of the biggest group hit_groups finds in the
    source's first REF_BARS bars of playing (the whole file for a source with REF_MAX_HITS hits or fewer). A source
    whose tails ring into the next hit has no complete member: its biggest group's medoid cut stands in, and the info
    says so. Returns (audio, info), or (None, why)."""
    import numpy as np
    from roadie_analysis import drumroles
    if not onsets:
        return None, "no onsets"
    bar_s = 4 * 60.0 / tempo
    a = max(0, int((onsets[0] - 0.05) * sr))
    b = len(x) if len(onsets) <= REF_MAX_HITS else min(len(x), int((onsets[0] + REF_BARS * bar_s) * sr))
    xs = np.asarray(x[a:b], dtype=np.float64)
    res = drumroles.hit_groups(xs, tempo, t0 - a / sr, sr=sr, stability_runs=0)
    xu = _undc(xs, sr)
    for g in sorted(res["groups"], key=lambda g: -g["count"]):
        audio, clean = group_audio(res, g, lambda _s: xu)
        if audio is not None:
            return audio, {"hits_in_excerpt": len(res["hits"]), "groups": len(res["groups"]), "group_hits": g["count"],
                           "clean": clean}
    return None, "no group"


def _run_stem(args):
    path, tempo, t0, cache = args
    from roadie_analysis import drumroles
    t = time.time()
    res = drumroles.hit_groups(path, tempo, t0)
    res["_seconds"] = time.time() - t
    with open(cache, "wb") as fh:
        pickle.dump(res, fh)
    return cache, res["_seconds"], len(res["hits"])


def run_set(name, stem_paths, tempo, t0, cache_dir, jobs, rerun):
    """hit_groups on each stem of a set (cached, in parallel); then mark_bleed and name_roles over the set."""
    from roadie_analysis import drumroles
    os.makedirs(os.path.join(cache_dir, name), exist_ok=True)
    todo, cache = [], {}
    for stem, p in stem_paths.items():
        cache[stem] = os.path.join(cache_dir, name, stem + ".pkl")
        if rerun or not os.path.exists(cache[stem]) or os.path.getmtime(cache[stem]) < os.path.getmtime(p):
            todo.append((p, tempo, t0, cache[stem]))
    if todo:
        print("  %s: hit_groups on %d stems (%d jobs)" % (name, len(todo), min(jobs, len(todo))), flush=True)
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            for c, secs, n in ex.map(_run_stem, todo):
                print("    %-40s %6.0f s  %5d hits" % (os.path.basename(c), secs, n), flush=True)
    results = {s: pickle.load(open(c, "rb")) for s, c in cache.items()}
    bleed = drumroles.mark_bleed(results) if len(results) > 1 else {s: {} for s in results}
    named = {s: drumroles.name_roles(r, KIND.get(s), bleed.get(s)) for s, r in results.items()}
    return results, named, bleed


def distance(a, b, sr):
    from roadie_analysis import oneshot
    return float(oneshot.spectral_distance(a, b, sr))


def average_hits(x, sr, result, group, clean, ref, lag_s=AVG_LAG_S):
    """Mean and median of a group's hits, cut at their onsets, and their distances to the true sample `ref`.
    x is the stem as hit_groups read it (after DC removal). Hits are the group's measured members, at most
    AVG_MAX_HITS evenly spaced in time; each cut ends at its own `end` (before the next hit or MAX_CUT_S), and a
    sample is averaged over the hits whose cut reaches it, as long as there are at least AVG_MIN_HITS of them
    and a quarter of the hits. With lag_s > 0 each hit first moves up to lag_s either way to the place its
    first 20 ms correlate best with the clean hit's. Returns {"hits", "mean", "median", "single", "length_ms"}."""
    import numpy as np
    by_id = {h["id"]: h for h in result["hits"]}
    ids = sorted(i for i in group["members"] if by_id[i]["status"] == "grouped")
    if len(ids) > AVG_MAX_HITS:
        ids = [ids[int(round(k))] for k in np.linspace(0, len(ids) - 1, AVG_MAX_HITS)]
    L = int(AVG_MAX_S * sr)
    ch = x.shape[1]
    stack = np.full((len(ids), L, ch), np.nan)
    c = np.asarray(clean, dtype=np.float64)
    c = c[:, None] if c.ndim == 1 else c
    win = min(int(0.020 * sr), len(c))
    lag = int(round(lag_s * sr))
    for k, i in enumerate(ids):
        h = by_id[i]
        s = h["start"]
        if lag:
            seg = x[max(0, s - lag):s + lag + win]
            if len(seg) >= win + 1:
                tgt = c[:win].mean(1)
                scores = [float(np.dot(seg[o:o + win].mean(1), tgt)) if o + win <= len(seg) else -np.inf
                          for o in range(len(seg) - win + 1)]
                s = max(0, s - lag) + int(np.argmax(scores))
        cut = x[s:s + (h["end"] - h["start"])][:L]
        stack[k, :len(cut)] = cut
    count = np.isfinite(stack[:, :, 0]).sum(0)
    keep = count >= max(AVG_MIN_HITS, 0.25 * len(ids))
    n = int(np.argmin(keep)) if not keep.all() else L
    if n == 0 or len(ids) < AVG_MIN_HITS:
        return None
    with np.errstate(all="ignore"):
        mean, med = np.nanmean(stack[:, :n], axis=0), np.nanmedian(stack[:, :n], axis=0)
    return {"hits": len(ids), "length_ms": 1000.0 * n / sr, "mean": distance(mean, ref, sr),
            "median": distance(med, ref, sr), "single": distance(clean, ref, sr)}


# ---------------------------------------------------------------- driver

def group_distances(results, ref, sr, load_stem=None):
    """{(stem, group id): (distance of the group's hit to `ref`, clean)} for every group that has a hit to show (group_audio)."""
    out = {}
    for stem, res in results.items():
        for g in res["groups"]:
            audio, clean = group_audio(res, g, load_stem, stem)
            if audio is not None:
                out[(stem, g["id"])] = (distance(audio, ref, sr), clean)
    return out


def score_sources(results, named, sources, refs, sr, load_stem=None):
    """One row per true source for one set. `load_stem(stem)` reads a stem as hit_groups did (only called for a group
    that has no complete member to show). The best role is chosen among the roles with a group that sounds like the
    source (best_role); `dist` is the nearest of that role's groups, `dist_main` the role's biggest group's."""
    cands = candidates(results, named)
    rows = {}
    for src, info in sources.items():
        true_s = info["onsets"]
        ref = refs.get(src)
        gd = group_distances(results, ref, sr, load_stem) if ref is not None else {}

        def role_dist(stem, role, gd=gd):
            ds = [gd[(stem, g)][0] for g in role["groups"] if (stem, g) in gd]
            return min(ds) if ds else None

        b = best_role(true_s, cands, role_dist if gd else None)
        row = {"true_hits": len(true_s), "best": None, "ref": info.get("reference")}
        if b:
            res = results[b["stem"]]
            by_id = {g["id"]: g for g in res["groups"]}
            near = min((g for g in b["role"]["groups"] if (b["stem"], g) in gd), key=lambda g: gd[(b["stem"], g)][0], default=None)
            main = max(b["role"]["groups"], key=lambda g: by_id[g]["count"])
            det = match_onsets(true_s, [h["onset_s"] for h in res["hits"]])
            row["best"] = {"stem": b["stem"], "role": b["role"]["name"], "named": b["role"]["named"],
                           "why_neutral": None if b["role"]["named"] else named[b["stem"]]["groups"][main]["reason"],
                           "groups": b["role"]["groups"], "hits": b["hits"], "recall": b["recall"],
                           "precision": b["precision"], "f1": b["f1"], "sound": b["sound"], "timing_best": b["timing_best"],
                           "detected": det / len(true_s) if true_s else 0.0,
                           "near_group": near, "dist": gd[(b["stem"], near)][0] if near is not None else None,
                           "clean": gd[(b["stem"], near)][1] if near is not None else None,
                           "dist_main": gd[(b["stem"], main)][0] if (b["stem"], main) in gd else None}
        alike = [c for c in cands if (d := role_dist(c[0], c[1])) is not None and d <= SOUND_MAX] if gd else []
        union = sorted(t for _, _, ons in alike for t in ons)
        tp = match_onsets(true_s, union)
        row["alike"] = {"roles": [(c[0], c[1]["name"]) for c in alike], "hits": len(union),
                        "recall": tp / len(true_s) if true_s else 0.0, "precision": tp / len(union) if union else 0.0}
        if gd:
            (stem, gid), (d, clean) = min(gd.items(), key=lambda kv: kv[1][0])
            row["closest"] = {"stem": stem, "group": gid, "dist": d, "clean": clean,
                              "count": next(g["count"] for g in results[stem]["groups"] if g["id"] == gid)}
        else:
            row["closest"] = None
        rows[src] = row
    return rows


def sources_stage(manifest, out_dir, cache_dir):
    """Onsets and a reference clean hit of every true drum source (cached)."""
    import soundfile as sf
    p = os.path.join(cache_dir, "sources.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    out = {"info": {}, "refs": {}, "sr": None}
    for s in manifest["sources"]:
        d = s.get("drum_source")
        if not d:
            continue
        x, sr = sf.read(os.path.join(out_dir, "truth_sources", d + ".flac"), dtype="float64", always_2d=True)
        ons = true_onsets(x, sr)
        ref, why = reference_hit(_undc(x, sr), sr, ons, manifest["bpm"], manifest.get("bar1_s", 0.0))
        out["info"][d] = {"onsets": ons, "reference": why, "file": s["file"], "role": s["role"]}
        out["refs"][d] = ref
        out["sr"] = sr
        print("  source %-10s %5d onsets, reference %s" % (d, len(ons), "ok" if ref is not None else why), flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(out, fh)
    return out


def table(name, rows):
    lines = ["%s" % name,
             "%-10s %6s  %-24s %5s %6s %6s %6s %6s %6s  %-12s  %-24s" % (
                 "source", "true", "best role (stem: name)", "hits", "recall", "prec", "detect", "dist", "d main", "all alike r/p", "closest group (stem g: dist)")]
    for src, r in rows.items():
        b, c = r["best"], r["closest"]
        if not b:
            lines.append("%-10s %6d  (no role)" % (src, r["true_hits"]))
            continue
        star = lambda d, clean: "-" if d is None else "%.2f%s" % (d, "" if clean else "*")   # noqa: E731
        lines.append("%-10s %6d  %-24s %5d %6.2f %6.2f %6.2f %6s %6s  %-12s  %-24s" % (
            src, r["true_hits"], ("%s: %s%s" % (b["stem"].replace("drums_", ""), b["role"], "" if b["sound"] else " ?"))[:24], b["hits"],
            b["recall"], b["precision"], b["detected"], star(b["dist"], b["clean"]), star(b["dist_main"], True),
            "%.2f/%.2f" % (r["alike"]["recall"], r["alike"]["precision"]) if r["alike"]["roles"] else "-",
            "-" if not c else "%s g%d: %s" % (c["stem"].replace("drums_", ""), c["group"], star(c["dist"], c["clean"]))))
    lines.append("  all alike: the union of the hits of every role of the set that has a group within %.1f of the source's hit." % SOUND_MAX)
    lines.append("  ? no role has a group within %.1f of the source's hit: the plain best F1 is shown. * no complete member to "
                 "show: the medoid's cut stands in (its tail is cut by the next hit)." % SOUND_MAX)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--set", action="append", dest="sets", default=[],
                    help="separated | true | NAME=DIR (repeatable; default: separated and true)")
    ap.add_argument("--stems", help="the separated stems folder (default: the only *_stems in out_dir/separated)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--average", action="store_true", help="mitigation (a): averaged hits against the single clean hit")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--out", help="JSON output (default: out_dir/roles_<sets>.json)")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    out_dir = os.path.expanduser(manifest["out_dir"])
    cache_dir = os.path.join(out_dir, "roles")
    tempo, t0 = manifest["bpm"], manifest.get("bar1_s", 0.0)

    from benchmark import pick_stems_dir
    wanted = args.sets or ["separated", "true"]
    sets = {}
    for w in wanted:
        if w == "separated":
            d = pick_stems_dir(out_dir, args.stems)
        elif w == "true":
            d = os.path.join(out_dir, "truth")
        elif "=" in w:
            w, d = w.split("=", 1)
            d = os.path.expanduser(d)
        else:
            sys.exit("unknown set %r: separated, true or NAME=DIR" % w)
        paths = {s: os.path.join(d, s + ext) for s in DRUM_STEMS for ext in (".flac", ".wav")
                 if os.path.exists(os.path.join(d, s + ext))}
        if not paths:
            sys.exit("no drums_*.flac in %s" % d)
        sets[w] = paths

    print("true sources")
    src = sources_stage(manifest, out_dir, cache_dir)
    sr = src["sr"]
    report = {"tolerance_s": TOL_S, "sources": {k: {"true_hits": len(v["onsets"]), "file": v["file"], "role": v["role"]}
                                              for k, v in src["info"].items()}, "sets": {}}
    for name, paths in sets.items():
        print("set %s: %s" % (name, ", ".join(sorted(paths))))
        results, named, bleed = run_set(name, paths, tempo, t0, cache_dir, args.jobs, args.rerun)
        loaded = {}

        def load_stem(stem, paths=paths, loaded=loaded):
            if stem not in loaded:
                from roadie_analysis.drumroles import load
                x, xsr = load(paths[stem])
                loaded[stem] = _undc(x, xsr)
            return loaded[stem]

        rows = score_sources(results, named, src["info"], src["refs"], sr, load_stem)
        roles = {s: [{"name": r["name"], "named": r["named"], "groups": r["groups"], "count": r["count"],
                      "assigned": r["assigned"]} for r in n["roles"]] for s, n in named.items()}
        marked = {s: sorted(g for g, b in bl.items() if b["bleed"]) for s, bl in bleed.items()}
        entry = {"rows": rows, "roles": roles, "bleed_groups": marked,
                 "seconds": {s: r.get("_seconds") for s, r in results.items()}}
        if args.average:
            avg = {}
            for srcname, r in rows.items():
                b = r["best"]
                if not b or b["near_group"] is None or src["refs"].get(srcname) is None:
                    continue
                res = results[b["stem"]]
                g = next(g for g in res["groups"] if g["id"] == b["near_group"])
                x = load_stem(b["stem"])
                clean, _ = group_audio(res, g, load_stem, b["stem"])
                avg[srcname] = {"detector_onset": average_hits(x, sr, res, g, clean, src["refs"][srcname], 0.0),
                                "aligned": average_hits(x, sr, res, g, clean, src["refs"][srcname])}
            entry["average"] = avg
        report["sets"][name] = entry
        print()
        print(table("set %s  (tolerance %d ms)" % (name, 1000 * TOL_S), rows))
        if args.average:
            print("\n  averaged hits: distance to the true sample (single clean hit | mean | median), hits averaged, ms")
            for srcname, a in entry["average"].items():
                for how, v in a.items():
                    if v:
                        print("  %-10s %-15s %5.2f | %5.2f | %5.2f   %4d hits, %4.0f ms" % (
                            srcname, how, v["single"], v["mean"], v["median"], v["hits"], v["length_ms"]))
        print("  bleed-marked groups: " + (", ".join("%s %s" % (s, g) for s, g in marked.items() if g) or "none"))
        print()

    out = args.out or os.path.join(out_dir, "roles_%s.json" % "_".join(sets))
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()

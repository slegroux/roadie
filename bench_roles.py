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

Per true source (kick, snare, snare2, clap, chh1, chh2, chh3, shaker, perc1, metal_hit), per set, all scored on EXCLUSIVE
positions: the source's onsets that no other source of its role stem (the manifest's `role`) plays within TOL_S. Sources that
play on the same steps (a closed hat on every 8th, another on every 16th, which includes them) cannot be told apart by timing
where they coincide, so a position another source plays at counts for neither recall nor precision. A source with fewer than
EXCL_MIN exclusive positions is scored on all of them (`excl` shows an `a`).

    role        one per source, assigned one to one: the assignment of roles (of any stem of the set) to sources that
                maximises the sum of F1 (Hungarian, scipy.optimize.linear_sum_assignment) among the roles that sound
                like the source (a group's hit within SOUND_MAX = 2.0 of its reference, drumroles' own "sound alike"
                distance) and find some of its onsets; a source with none left reads `(no role)`. Recall is the share
                of the exclusive onsets with a role hit within TOL_S, precision the share of the role's hits away from every
                other source's onsets that pair with one (a hit and an onset pair one to one: a greedy two-pointer matching,
                optimal in one dimension). A role shared by sources that play together (SNARE and CLAP in `snare 1`) goes to
                the one it matches best; the other reads `(no role)` or its next role and shows its own best as `alone`.
    timing best the role with the best F1 by timing alone, whatever it sounds like, shown under the row when it is not the
                pick, its precision is at least TIMING_P (0.9) and its F1 beats the pick's: what the sound condition hides
    detected    recall over every hit the role's stem found, grouped or not: what the grouping and the measurement rules lose
                between "the onset was seen" and "it is in a role"
    dist        `oneshot.spectral_distance` from the nearest of the role's groups' hits to the true sample: one clean hit read
                from the source file. `d main`: the same for the role's biggest group. 0 is the same spectrum; two takes of
                one noise hat read 0.55-0.82; the grouping threshold is 1.5. A group with no complete member (every tail cut
                by the next hit) is shown by its medoid's cut, marked `*`.
    all alike   the union of every role of the set that sounds like the source, recall and precision on the same positions
    closest     the group of any stem whose hit is nearest the true sample: where the sound is, when the role is not it

True onsets are the source file's own, found by the same detector the roles use
(oneshot._segments and the onset reading hit_groups makes), at a tolerance of TOL_S (25 ms,
drumroles.COINCIDE_S, the tolerance its bleed test counts a coincidence at). One table lists every set (the sets side by side).

--average (mitigation (a)): for each source's role (its `alone` role when it shares one), every measured COMPLETE hit of its
nearest group (its cut falls 40 dB before the next hit cuts it) is cut from the stem at its onset and the cuts are averaged
(mean and median, sample by sample over the hits that cover it, at least 3 and a quarter of them), once as detected and once
with each hit moved up to AVG_LAG_S (10 ms) either way to the place where the normalized cross-correlation of its first
AVG_ALIGN_S with the clean hit's is greatest. The shift every hit took is printed (median, share at the +-10 ms edge) and kept in
the JSON. Distance of the mean, the median and the single clean hit to the true sample is the verdict's number.

Results are cached per (set, stem) under out_dir/roles/ (the grouping of a whole hat stem
takes minutes), each keyed on the stem file's mtime and size, a hash of drumroles.py and oneshot.py, the tempo and bar 1; a
change to any of them redoes that result, --rerun redoes them all. The sources' cache (sources.pkl) is keyed the same way on the
source files, the functions that read them and the reference's size. No pack audio is read or written here beyond
truth_sources/, which bench_truth.py wrote. Local only.
"""
import argparse
import hashlib
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
AVG_LAG_S = 0.010       # a hit may move this far either way to line up with the clean hit before it is averaged
AVG_ALIGN_S = 0.030     # ...by the normalized cross-correlation of this much of its start with the clean hit's
TIMING_P = 0.9          # a role by timing alone is shown when its precision is at least this and its F1 beats the pick's
EXCL_MIN = 10           # a source is scored on its exclusive positions only when it has at least this many


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


def near_any(t, zone, tol=TOL_S):
    """Boolean array: which of the times `t` have one of `zone` within `tol`. Both are lists of seconds."""
    import numpy as np
    t, z = np.asarray(sorted(t), dtype=float), np.asarray(sorted(zone), dtype=float)
    if not len(t) or not len(z):
        return np.zeros(len(t), dtype=bool)
    i = np.searchsorted(z, t)
    left, right = z[np.maximum(i - 1, 0)], z[np.minimum(i, len(z) - 1)]
    return np.minimum(np.abs(t - left), np.abs(t - right)) <= tol


def exclusive_onsets(true_s, others, tol=TOL_S, minimum=EXCL_MIN):
    """(the onsets of `true_s` that no onset of `others` (the other sources of the same role stem) is within `tol` of,
    True), or (all of `true_s`, False) when fewer than `minimum` are exclusive: a source that always plays with another
    cannot be told from it by timing, and its scores then carry every position."""
    ts = sorted(true_s)
    keep = [t for t, n in zip(ts, near_any(ts, others, tol)) if not n]
    return (keep, True) if len(keep) >= minimum else (ts, False)


def role_scores(true_s, others, cands, dist=None, sound_max=SOUND_MAX):
    """One source against every candidate role, scored on exclusive positions: the source's onsets with no other source
    of its role stem within TOL_S (all of them when it has fewer than EXCL_MIN, `exclusive` False). Recall is the share of
    those onsets a role hit pairs with; precision the share of the role's hits away from every other source's onsets
    (the ones left to this source or to nothing) that pair with one. A hit on a position another source of the stem
    plays at counts for neither: a closed hat on every 8th reads precision 1.00 against a hat on every 16th by
    timing alone, and the exclusive positions are where it cannot. `dist(stem, role)` (the nearest of the role's groups to
    the source's reference hit, or None) sets `alike`: within `sound_max`. Returns (one dict per candidate in order:
    {"k", "stem", "role", "tp", "hits", "role_hits", "recall", "precision", "f1", "dist", "alike"}, {"onsets", "exclusive"
    (how many positions are scored), "only" (True when they are the exclusive ones)})."""
    ons, only = exclusive_onsets(true_s, others)
    zone = others if only else []
    out = []
    for k, (stem, role, hits) in enumerate(cands):
        free = [t for t, n in zip(sorted(hits), near_any(hits, zone)) if not n]
        tp = match_onsets(ons, free)
        rec = tp / len(ons) if ons else 0.0
        pre = tp / len(free) if free else 0.0
        d = dist(stem, role) if dist else None
        out.append({"k": k, "stem": stem, "role": role, "tp": tp, "hits": len(free), "role_hits": len(hits), "recall": rec,
                    "precision": pre, "f1": f1(rec, pre), "dist": d, "alike": d is not None and d <= sound_max})
    return out, {"onsets": len(true_s), "exclusive": len(ons), "only": only}


def sounds_right(score):
    """A role is the source's own when it sounds like it AND finds at least one of its onsets: a role that sounds like the
    source and never plays where it does is not where the source is."""
    return score["alike"] and score["f1"] > 0


def best_alone(scores):
    """The best of one source's role scores taken alone, among the roles that sound right (sounds_right: ties go to the higher
    recall); None when none does."""
    return max((s for s in scores if sounds_right(s)), key=lambda s: (s["f1"], s["recall"]), default=None)


def timing_best(scores):
    """The best F1 by timing alone (ties: the higher recall), whatever it sounds like."""
    return max(scores, key=lambda s: (s["f1"], s["recall"]), default=None)


def assign_roles(scored):
    """One role per source and each role to one source, among the roles that sound right for it (sounds_right): the
    assignment that maximises the sum of F1 over the sources (Hungarian, scipy.optimize.linear_sum_assignment).
    {source: [its score dicts, one per candidate role]} in, {source: the assigned score dict, or None when no role that
    sounds right is left for it}. A role that does not sound like the source is never assigned: on timing alone a source
    takes a role that merely plays on its steps (a closed hat on every 8th lands on the kick), and by the sum it would take
    that role from the source that sounds like it; `timing_best` shows such a role instead. Sources that play one sound
    together (SNARE and CLAP in one `snare 1` role) share a role: it goes to the one it matches best, and the other reads
    None here and shows its own best role in `alone`."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    names = [n for n, sc in scored.items() if sc]
    out = {n: None for n in scored}
    if not names:
        return out
    w = np.array([[s["f1"] + 1.0 if sounds_right(s) else -1e3 for s in scored[n]] for n in names])
    for i, j in zip(*linear_sum_assignment(w, maximize=True)):
        out[names[i]] = scored[names[i]][j] if sounds_right(scored[names[i]][j]) else None
    return out


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


def code_hash(*funcs):
    """Hash of the code whose output the caches hold: roadie_analysis' drumroles.py and oneshot.py, and the source text of
    any `funcs` (this file's own functions the cached result also depends on)."""
    import inspect
    from roadie_analysis import drumroles, oneshot
    h = hashlib.sha256()
    for path in (drumroles.__file__, oneshot.__file__):
        with open(path, "rb") as fh:
            h.update(fh.read())
    for f in funcs:
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()[:16]


def file_key(path):
    """What a cached result depends on in an input file: its mtime and its size."""
    st = os.stat(path)
    return [st.st_mtime_ns, st.st_size]


def read_cache(path, key):
    """The result cached at `path` when it was made with this `key` (inputs, code, parameters), else None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            c = pickle.load(fh)
    except Exception:
        return None
    return c["result"] if isinstance(c, dict) and c.get("key") == key and "result" in c else None


def write_cache(path, key, result):
    with open(path, "wb") as fh:
        pickle.dump({"key": key, "result": result}, fh)


def _run_stem(args):
    path, tempo, t0, cache, key = args
    from roadie_analysis import drumroles
    t = time.time()
    res = drumroles.hit_groups(path, tempo, t0)
    res["_seconds"] = time.time() - t
    write_cache(cache, key, res)
    return cache, res["_seconds"], len(res["hits"])


def run_set(name, stem_paths, tempo, t0, cache_dir, jobs, rerun):
    """hit_groups on each stem of a set (cached, in parallel); then mark_bleed and name_roles over the set. A cached
    result is used only when the stem file (mtime, size), drumroles.py and oneshot.py (hash), tempo and t0 are the ones
    it was made with."""
    from roadie_analysis import drumroles
    os.makedirs(os.path.join(cache_dir, name), exist_ok=True)
    code = code_hash()
    todo, cache, results = [], {}, {}
    for stem, p in stem_paths.items():
        cache[stem] = os.path.join(cache_dir, name, stem + ".pkl")
        key = {"input": file_key(p), "code": code, "tempo": tempo, "t0": t0}
        results[stem] = None if rerun else read_cache(cache[stem], key)
        if results[stem] is None:
            todo.append((p, tempo, t0, cache[stem], key))
    if todo:
        print("  %s: hit_groups on %d stems (%d jobs)" % (name, len(todo), min(jobs, len(todo))), flush=True)
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            for c, secs, n in ex.map(_run_stem, todo):
                print("    %-40s %6.0f s  %5d hits" % (os.path.basename(c), secs, n), flush=True)
        for t in todo:
            results[next(s for s, c in cache.items() if c == t[3])] = read_cache(t[3], t[4])
    bleed = drumroles.mark_bleed(results) if len(results) > 1 else {s: {} for s in results}
    named = {s: drumroles.name_roles(r, KIND.get(s), bleed.get(s)) for s, r in results.items()}
    return results, named, bleed


def distance(a, b, sr):
    from roadie_analysis import oneshot
    return float(oneshot.spectral_distance(a, b, sr))


def align_lag(x, start, clean, lag, win):
    """How many samples hit `start` of the stem `x` ([n, ch]) must move to line up with `clean`: the shift within +-`lag`
    at which the normalized cross-correlation of the next `win` samples (channels averaged) with the clean hit's first
    `win` is greatest. 0 when the window does not fit. Normalized, so a louder or quieter hit scores the same."""
    import numpy as np
    c = np.asarray(clean, dtype=np.float64)
    tgt = (c[:, None] if c.ndim == 1 else c)[:win].mean(1)
    lo = max(0, start - lag)
    seg = x[lo:start + lag + len(tgt)].mean(1)
    if len(seg) < len(tgt) + 1 or not float(np.dot(tgt, tgt)) > 0:
        return 0
    cross = np.correlate(seg, tgt, "valid")
    csum = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    energy = csum[len(tgt):len(tgt) + len(cross)] - csum[:len(cross)]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(energy > 0, cross / np.sqrt(energy * float(np.dot(tgt, tgt))), -np.inf)
    return lo + int(np.argmax(score)) - start


def average_hits(x, sr, result, group, clean, ref, lag_s=AVG_LAG_S):
    """Mean and median of a group's hits, cut at their onsets, and their distances to the true sample `ref`.
    x is the stem as hit_groups read it (after DC removal). Hits are the group's measured members whose cut is complete
    (it falls 40 dB before the next hit or MAX_CUT_S cuts it: oneshot._measure's `truncated`, the test the clean hit's own
    members pass), at most AVG_MAX_HITS evenly spaced in time; each cut ends at its own `end`, and a sample is averaged
    over the hits whose cut reaches it, as long as there are at least AVG_MIN_HITS of them and a quarter of the hits.
    With lag_s > 0 each hit first moves, up to lag_s either way, to the place its first AVG_ALIGN_S have the greatest
    normalized cross-correlation with the clean hit's (align_lag). Returns {"hits", "complete" (of the group's
    members), "mean", "median", "single", "length_ms", "lags_ms" (the shift of every hit averaged), "lag_median_ms",
    "lag_edge" (the share of hits at the search edge: a search that wants more room)}, or None when fewer than
    AVG_MIN_HITS complete members remain."""
    import numpy as np
    from roadie_analysis import oneshot
    by_id = {h["id"]: h for h in result["hits"]}
    members = sorted(i for i in group["members"] if by_id[i]["status"] == "grouped")

    def complete(i):
        h = by_id[i]
        try:
            return not oneshot._measure(x[h["start"]:h["end"]], sr, "the cut ends", quick=True)["truncated"]
        except ValueError:
            return False
    ids = [i for i in members if complete(i)]
    n_complete = len(ids)
    if len(ids) > AVG_MAX_HITS:
        ids = [ids[int(round(k))] for k in np.linspace(0, len(ids) - 1, AVG_MAX_HITS)]
    if len(ids) < AVG_MIN_HITS:
        return None
    L = int(AVG_MAX_S * sr)
    ch = x.shape[1]
    stack = np.full((len(ids), L, ch), np.nan)
    c = np.asarray(clean, dtype=np.float64)
    c = c[:, None] if c.ndim == 1 else c
    win = min(int(AVG_ALIGN_S * sr), len(c))
    lag = int(round(lag_s * sr))
    lags = []
    for k, i in enumerate(ids):
        h = by_id[i]
        s = h["start"]
        if lag:
            s += align_lag(x, s, c, lag, win)
            lags.append(s - h["start"])
        cut = x[max(0, s):s + (h["end"] - h["start"])][:L]
        stack[k, :len(cut)] = cut
    count = np.isfinite(stack[:, :, 0]).sum(0)
    keep = count >= max(AVG_MIN_HITS, 0.25 * len(ids))
    n = int(np.argmin(keep)) if not keep.all() else L
    if n == 0:
        return None
    with np.errstate(all="ignore"):
        mean, med = np.nanmean(stack[:, :n], axis=0), np.nanmedian(stack[:, :n], axis=0)
    lags_ms = [1000.0 * v / sr for v in lags]
    return {"hits": len(ids), "complete": n_complete, "members": len(members), "length_ms": 1000.0 * n / sr,
            "mean": distance(mean, ref, sr), "median": distance(med, ref, sr), "single": distance(clean, ref, sr),
            "lags_ms": lags_ms, "lag_median_ms": float(np.median(lags_ms)) if lags_ms else 0.0,
            "lag_edge": float(np.mean([abs(v) >= lag - 1 for v in lags])) if lags else 0.0}


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
    that has no complete member to show). Every source is scored against every role on its exclusive positions
    (role_scores; the other sources of its role stem are the ones whose positions it leaves out), then the roles are
    assigned to the sources one to one (assign_roles), a role that sounds like the source being preferred. `best` is the
    assigned role (None: none matches a single onset), `alone` the role the source would take alone when that is not
    `best`, `timing_best` the best F1 by timing whatever it sounds like, set only when it is not `best`, its precision
    is at least TIMING_P and its F1 is higher than `best`'s (what a sound condition hides); `dist` is the nearest of
    the role's groups to the source's reference hit, `dist_main` the role's biggest group's."""
    cands = candidates(results, named)
    others_of = {k: sorted(t for j, v in sources.items() if j != k and info.get("role") and v.get("role") == info["role"] for t in v["onsets"])
                 for k, info in sources.items()}
    gds, scored, infos = {}, {}, {}
    for src, info in sources.items():
        ref = refs.get(src)
        gd = gds[src] = group_distances(results, ref, sr, load_stem) if ref is not None else {}

        def role_dist(stem, role, gd=gd):
            ds = [gd[(stem, g)][0] for g in role["groups"] if (stem, g) in gd]
            return min(ds) if ds else None
        scored[src], infos[src] = role_scores(info["onsets"], others_of[src], cands, role_dist if gd else None)
    assigned = assign_roles(scored)
    rows = {}
    for src, info in sources.items():
        true_s, gd = info["onsets"], gds[src]
        b = assigned[src]
        alone, tb = best_alone(scored[src]), timing_best(scored[src])
        row = {"true_hits": len(true_s), "exclusive": infos[src]["exclusive"], "exclusive_only": infos[src]["only"],
               "best": None, "alone": None, "timing_best": None, "ref": info.get("reference")}

        def brief(c):
            groups = cands[c["k"]][1]["groups"]
            return {"stem": c["stem"], "role": cands[c["k"]][1]["name"], "groups": groups, "recall": c["recall"],
                    "precision": c["precision"], "f1": c["f1"], "dist": c["dist"], "alike": c["alike"],
                    "own_stem": c["stem"] == info.get("role"),
                    "near_group": min((g for g in groups if (c["stem"], g) in gd), key=lambda g: gd[(c["stem"], g)][0], default=None)}
        if alone and (b is None or alone["k"] != b["k"]):
            row["alone"] = brief(alone)
        if tb and tb["f1"] > 0 and (b is None or tb["k"] != b["k"]) and tb["precision"] >= TIMING_P and (b is None or tb["f1"] > b["f1"]):
            row["timing_best"] = brief(tb)
        if b:
            stem, role = cands[b["k"]][0], cands[b["k"]][1]
            res = results[stem]
            by_id = {g["id"]: g for g in res["groups"]}
            near = min((g for g in role["groups"] if (stem, g) in gd), key=lambda g: gd[(stem, g)][0], default=None)
            main = max(role["groups"], key=lambda g: by_id[g]["count"])
            det = match_onsets(exclusive_onsets(true_s, others_of[src])[0], [h["onset_s"] for h in res["hits"]])
            row["best"] = {"stem": stem, "role": role["name"], "named": role["named"],
                           "why_neutral": None if role["named"] else named[stem]["groups"][main]["reason"],
                           "groups": role["groups"], "hits": b["hits"], "role_hits": b["role_hits"], "recall": b["recall"],
                           "precision": b["precision"], "f1": b["f1"], "sound": b["alike"],
                           "detected": det / infos[src]["exclusive"] if infos[src]["exclusive"] else 0.0,
                           "near_group": near, "dist": gd[(stem, near)][0] if near is not None else None,
                           "clean": gd[(stem, near)][1] if near is not None else None,
                           "dist_main": gd[(stem, main)][0] if (stem, main) in gd else None}
        alike = [c for c, sc in zip(cands, scored[src]) if sc["alike"]]
        zone = others_of[src] if infos[src]["only"] else []
        union = sorted(t for _, _, ons in alike for t, n in zip(sorted(ons), near_any(ons, zone)) if not n)
        ex = exclusive_onsets(true_s, others_of[src])[0]
        tp = match_onsets(ex, union)
        row["alike"] = {"roles": [(c[0], c[1]["name"]) for c in alike], "hits": len(union),
                        "recall": tp / len(ex) if ex else 0.0, "precision": tp / len(union) if union else 0.0}
        if gd:
            (stem, gid), (d, clean) = min(gd.items(), key=lambda kv: kv[1][0])
            row["closest"] = {"stem": stem, "group": gid, "dist": d, "clean": clean,
                              "count": next(g["count"] for g in results[stem]["groups"] if g["id"] == gid)}
        else:
            row["closest"] = None
        rows[src] = row
    return rows


def sources_stage(manifest, out_dir, cache_dir):
    """Onsets and a reference clean hit of every true drum source (cached; the cache holds for the source files (mtime,
    size), true_onsets and reference_hit, drumroles.py and oneshot.py, the tempo, bar 1 and the reference's size parameters)."""
    import soundfile as sf
    files = {s["drum_source"]: os.path.join(out_dir, "truth_sources", s["drum_source"] + ".flac")
             for s in manifest["sources"] if s.get("drum_source")}
    key = {"inputs": {d: file_key(p) for d, p in files.items()}, "code": code_hash(true_onsets, reference_hit, _undc),
           "bpm": manifest["bpm"], "bar1_s": manifest.get("bar1_s", 0.0), "ref": [REF_BARS, REF_MAX_HITS]}
    p = os.path.join(cache_dir, "sources.pkl")
    cached = read_cache(p, key)
    if cached is not None:
        return cached
    out = {"info": {}, "refs": {}, "sr": None}
    for s in manifest["sources"]:
        d = s.get("drum_source")
        if not d:
            continue
        x, sr = sf.read(files[d], dtype="float64", always_2d=True)
        ons = true_onsets(x, sr)
        ref, why = reference_hit(_undc(x, sr), sr, ons, manifest["bpm"], manifest.get("bar1_s", 0.0))
        out["info"][d] = {"onsets": ons, "reference": why, "file": s["file"], "role": s["role"]}
        out["refs"][d] = ref
        out["sr"] = sr
        print("  source %-10s %5d onsets, reference %s" % (d, len(ons), "ok" if ref is not None else why), flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    write_cache(p, key, out)
    return out


def table(sets):
    """One table for every set, a row per source and set (the sets side by side, as the plan asks): `sets` is
    {set name: rows of score_sources}. Under a row, what `alone` and `timing_best` say when they differ from `best`."""
    lines = ["%-10s %-9s %6s %5s  %-24s %5s %6s %6s %6s %6s %6s %6s  %-12s  %-24s" % (
        "source", "set", "true", "excl", "role (stem: name)", "hits", "recall", "prec", "F1", "detect", "dist", "d main",
        "all alike r/p", "closest group (stem g: dist)")]
    star = lambda d, clean: "-" if d is None else "%.2f%s" % (d, "" if clean else "*")   # noqa: E731
    role_name = lambda c: "%s: %s" % (c["stem"].replace("drums_", ""), c["role"])         # noqa: E731
    names = list(next(iter(sets.values())))
    for src in names:
        for k, (name, rows) in enumerate(sets.items()):
            r = rows[src]
            b, c = r["best"], r["closest"]
            head = "%-10s %-9s %6d %5s  " % (src if k == 0 else "", name, r["true_hits"], "%d%s" % (r["exclusive"], "" if r["exclusive_only"] else "a"))
            cl = "-" if not c else "%s g%d: %s" % (c["stem"].replace("drums_", ""), c["group"], star(c["dist"], c["clean"]))
            al = "%.2f/%.2f" % (r["alike"]["recall"], r["alike"]["precision"]) if r["alike"]["roles"] else "-"
            if not b:
                lines.append(head + "%-24s %5s %6s %6s %6s %6s %6s %6s  %-12s  %-24s" % ("(no role)", "-", "-", "-", "-", "-", "-", "-", al, cl))
            else:
                lines.append(head + "%-24s %5d %6.2f %6.2f %6.2f %6.2f %6s %6s  %-12s  %-24s" % (
                    ("%s%s" % (role_name(b), "" if b["sound"] else " ?"))[:24], b["hits"], b["recall"], b["precision"], b["f1"],
                    b["detected"], star(b["dist"], b["clean"]), star(b["dist_main"], True), al, cl))
            for tag, c2 in (("alone", r["alone"]), ("timing best", r["timing_best"])):
                if c2:
                    lines.append("%27s %s: %s (group%s %s)  R %.2f  P %.2f  F1 %.2f%s%s" % (
                        "", tag, role_name(c2), "" if len(c2["groups"]) == 1 else "s", ",".join(str(g) for g in c2["groups"]),
                        c2["recall"], c2["precision"], c2["f1"], "  dist %.2f" % c2["dist"] if c2["dist"] is not None else "",
                        "  [own stem]" if c2["own_stem"] else "  [another stem]"))
    lines.append("  excl: the source's onsets scored: those with no other source of its role stem within %d ms; an `a` is all of them "
                 "(under %d are exclusive). hits: the role's hits away from the other sources' onsets. recall, precision, F1 and detect are over "
                 "those positions." % (round(1000 * TOL_S), EXCL_MIN))
    lines.append("  roles are assigned to sources one to one (Hungarian on F1) among the roles with a group within %.1f of the source's hit that "
                 "find some of its onsets: a source that shares its role with another (SNARE and CLAP) reads `alone` for the role it would "
                 "have taken, and `timing best` is the role with the best F1 by timing whatever it sounds like (dist is how far it is), "
                 "shown when its precision is at least %.1f and its F1 beats the pick's." % (SOUND_MAX, TIMING_P))
    lines.append("  all alike: the union of the hits of every role of the set with a group within %.1f of the source's hit. * no complete "
                 "member to show: the medoid's cut stands in." % SOUND_MAX)
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
                b = r["alone"] or r["best"]          # a source that shares its role (CLAP with SNARE) is averaged on the role it would take alone
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
        if args.average:
            print("\n  set %s, averaged hits: distance to the true sample (single clean hit | mean | median), hits averaged "
                  "of the complete members, ms; the shift each hit took to line up (median ms, share at the +-%d ms edge)"
                  % (name, round(1000 * AVG_LAG_S)))
            for srcname, a in entry["average"].items():
                for how, v in a.items():
                    if v:
                        print("  %-10s %-15s %5.2f | %5.2f | %5.2f   %4d hits of %4d complete (%d members), %4.0f ms%s" % (
                            srcname, how, v["single"], v["mean"], v["median"], v["hits"], v["complete"], v["members"], v["length_ms"],
                            "   shift %+.2f ms, %.0f %% at the edge" % (v["lag_median_ms"], 100 * v["lag_edge"]) if how == "aligned" else ""))
                    else:
                        print("  %-10s %-15s  no average: fewer than %d complete members" % (srcname, how, AVG_MIN_HITS))
        print("  set %s bleed-marked groups: %s\n" % (name, ", ".join("%s %s" % (s, g) for s, g in marked.items() if g) or "none"))

    print(table({n: e["rows"] for n, e in report["sets"].items()}))
    print()

    out = args.out or os.path.join(out_dir, "roles_%s.json" % "_".join(sets))
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()

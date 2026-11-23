"""How steady are the drum roles when the grid moves by an amount no ear would notice?

    python -m roadie_analysis.drumstability <stems_dir> [--stems hh,snare] [--jobs N] [--only tempo+0.001,t0-0.001] [--no-cache]

Runs `drums.drum_roles` (nothing written) at the folder's grid and at perturbed grids (tempo +-0.001 and +-0.005 bpm, t0
+-1 ms), then compares, hit by hit (hits matched by onset time within MATCH_S), which role each hit lands in. Role names
move around when groups reorder, so roles are compared as sets of hits: for every base role of at least MIN_HITS hits, the
perturbed role that holds most of its hits is its match, and `kept` is the share of its hits that sit there (a hit that is
no longer in any role counts as lost). The reverse is measured too, so that two base roles collapsing into one perturbed role
cannot pass: `purity` is the share of the matched perturbed role's hits that belong to the base role, and a base role whose
match is dominated by another base role is `absorbed` (its hits count as not kept in `agreement`). `agreement` is the share of
the hits of all such base roles that are kept in a perturbed role that they dominate. A named role is `named_ok` when its match
has the same name. Needs numpy and soundfile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from roadie_analysis.jobqueue import heavy_job

MATCH_S = 0.003
MIN_HITS = 4
PERTURBATIONS = (("tempo", 0.001), ("tempo", -0.001), ("tempo", 0.005), ("tempo", -0.005), ("t0", 0.001), ("t0", -0.001))


def hit_roles(rec: dict, stem: str) -> tuple[np.ndarray, np.ndarray, list]:
    """(onset times sorted, role index per hit, [role names]) of one stem of a record made with `hit_times=True`."""
    roles = rec["stems"][stem]["roles"]
    pairs = sorted((t, k) for k, r in enumerate(roles) for t in r["hit_times"])
    return (np.array([t for t, _ in pairs]), np.array([k for _, k in pairs], dtype=int), [r["name"] for r in roles])


def compare(base: dict, other: dict, stem: str, match_s: float = MATCH_S, min_hits: int = MIN_HITS) -> dict:
    """{"agreement", "roles": [{"name", "named", "hits", "kept", "purity", "absorbed", "match", "match_name", "named_ok",
    "split"}], "named_same", "named_base", "named_other"}."""
    tb, rb, nb = hit_roles(base, stem)
    to, ro, no = hit_roles(other, stem)
    named_b = {r["name"] for r in base["stems"][stem]["roles"] if r["named"]}
    named_o = {r["name"] for r in other["stems"][stem]["roles"] if r["named"]}

    def nearest(ta, tb_):                                   # index in tb_ of the hit nearest each time of ta within match_s, else -1
        out = []
        for t in ta:
            j = int(np.searchsorted(tb_, t))
            near = [m for m in (j - 1, j) if 0 <= m < len(tb_) and abs(tb_[m] - t) <= match_s]
            out.append(min(near, key=lambda m: abs(tb_[m] - t)) if near else -1)
        return np.array(out, dtype=int)

    back = nearest(to, tb)                                  # each perturbed hit's base hit
    base_role_of = np.where(back >= 0, rb[np.maximum(back, 0)], -1) if len(rb) else np.full(len(to), -1)
    rows, kept_all, hits_all = [], 0, 0
    for k, role in enumerate(base["stems"][stem]["roles"]):
        idx = np.nonzero(rb == k)[0]
        if len(idx) < min_hits:
            continue
        fwd = nearest(tb[idx], to)
        where = np.where(fwd >= 0, ro[np.maximum(fwd, 0)], -1) if len(ro) else np.full(len(idx), -1)   # the role each hit has in `other`, -1 none
        found = where[where >= 0]
        match = int(np.bincount(found).argmax()) if len(found) else -1
        kept = int((where == match).sum()) if match >= 0 else 0
        split = int(sum(1 for m in set(found.tolist()) if (where == m).sum() >= 0.1 * len(idx)))
        purity, absorbed = None, False
        if match >= 0:
            mine = base_role_of[ro == match]                # the base roles of every hit of the matched perturbed role
            purity = float((mine == k).sum() / len(mine)) if len(mine) else None
            known = mine[mine >= 0]
            absorbed = bool(len(known) and int(np.bincount(known).argmax()) != k)
        mrole = other["stems"][stem]["roles"][match] if match >= 0 else {}
        n_parts = lambda r: len(r.get("merged_from") or [1])  # noqa: E731
        rows.append({"echo": role.get("echo_of") is not None, "echo_same": (mrole.get("echo_of") is not None) == (role.get("echo_of") is not None),
                     "merged_parts": n_parts(role), "merged_same": n_parts(role) == n_parts(mrole) if mrole else False,
                     "name": role["name"], "named": role["named"], "hits": int(len(idx)), "kept": round(kept / len(idx), 3),
                     "purity": None if purity is None else round(purity, 3), "absorbed": absorbed,
                     "match": match, "match_name": no[match] if match >= 0 else None, "split": split,
                     "named_ok": (role["name"] == no[match]) if role["named"] and match >= 0 else (False if role["named"] else None)})
        kept_all += 0 if absorbed else kept
        hits_all += len(idx)
    return {"agreement": round(kept_all / hits_all, 3) if hits_all else None, "roles": rows, "named_same": named_b == named_o,
            "named_base": sorted(named_b), "named_other": sorted(named_o)}


def run(stems_dir, stems=None, jobs=None, only=None, tempo=None, t0=None, cache=True) -> dict:
    """The base run and the perturbed runs of `stems` ({name: record}); `stems` limits the drum stems analysed (a scratch
    folder of links is made, so nothing is written into `stems_dir`)."""
    from roadie_analysis import drums
    src = Path(stems_dir)
    g = drums.grid_from_folder(src)
    tempo = tempo or g["tempo"]
    if not tempo or (t0 is None and g["downbeat_s"] is None):
        raise ValueError("%s has no tempo or downbeat: it needs alignment.json (roadie load) or sections.json (roadie sections)" % src)
    bar_s = 60.0 / tempo * drums.BEATS_PER_BAR
    down = float(g["downbeat_s"] if g["downbeat_s"] is not None else t0)
    t0 = t0 if t0 is not None else down - int(down // bar_s) * bar_s
    with tempfile.TemporaryDirectory(prefix="roadie-stab-") as tmp:
        for f in src.iterdir():
            keep = f.name in ("alignment.json", "sections.json") or (
                f.name.startswith("drums_") and f.suffix in drums.AUDIO and (not stems or f.stem[len("drums_"):] in stems))
            if keep:
                os.symlink(f, Path(tmp) / f.name)
        out = {}
        for label, kw in (("base", {}),) + tuple(("%s%+g" % (k, d), {k: d}) for k, d in PERTURBATIONS):
            if only and label != "base" and label not in only:
                continue
            tt = tempo + kw.get("tempo", 0.0)
            t = t0 + kw.get("t0", 0.0)
            out[label] = drums.drum_roles(tmp, tempo=tt, t0=t, write=False, jobs=jobs, hit_times=True, cache=cache)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drumstability", description=__doc__.split("\n\n")[0])
    ap.add_argument("stems_dir")
    ap.add_argument("--stems", default="hh,snare", help="drum stems to analyse, comma separated (default hh,snare)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma separated perturbations to run, e.g. tempo+0.001,t0-0.001")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="detect every hit in every run: do not use the cache of detected hits")
    args = ap.parse_args(argv)
    from roadie_analysis import drums
    try:
        present = drums.find_stems(args.stems_dir)
    except OSError as e:
        sys.exit("drumstability: %s" % e)
    asked = [s for s in args.stems.split(",") if s]
    stems = [s for s in asked if s in present]
    if not stems:
        sys.exit("drumstability: none of the stems %s is in %s (it has: %s)" % (", ".join(asked), args.stems_dir, ", ".join(present) or "no drum stems"))
    if len(stems) < len(asked):
        print("drumstability: skipping stems not in the folder: %s" % ", ".join(s for s in asked if s not in present), file=sys.stderr)
    labels = ["%s%+g" % (k, d) for k, d in PERTURBATIONS]
    only = args.only.split(",") if args.only else None
    if only and not [o for o in only if o in labels]:
        sys.exit("drumstability: --only %s matches no perturbation (they are: %s)" % (args.only, ", ".join(labels)))
    try:
        with heavy_job("roadie drumstability %s" % args.stems_dir):     # the seven runs are one job in the machine's queue
            runs = run(args.stems_dir, stems, args.jobs, only, cache=not args.no_cache)
    except ValueError as e:
        sys.exit("drumstability: %s" % e)
    report = {}
    for stem in stems:
        report[stem] = {label: compare(runs["base"], rec, stem) for label, rec in runs.items() if label != "base"}
    if args.json:
        print(json.dumps(report, indent=1))
        return 0
    for stem, per in report.items():
        base_roles = [(r["name"], r["named"], r["hits"]) for r in runs["base"]["stems"][stem]["roles"] if r["measured"] >= MIN_HITS]
        print("%s base roles: %s" % (stem, ", ".join("%s%s %d" % (n, "*" if nm else "", h) for n, nm, h in base_roles)))
        for label, c in per.items():
            worst = min((r["kept"] for r in c["roles"]), default=None)
            pure = min((r["purity"] for r in c["roles"] if r["purity"] is not None), default=None)
            named = [r for r in c["roles"] if r["named"]]
            merged = [r["name"] for r in c["roles"] if r["absorbed"]]
            odd = [r["name"] for r in c["roles"] if not (r["echo_same"] and r["merged_same"])]
            print("  %-14s agreement %s  worst role kept %s  worst purity %s%s  named same %s  %s" % (
                label, c["agreement"], worst, pure, "  ABSORBED: " + ", ".join(merged) if merged else "", c["named_same"],
                ", ".join("%s %.2f%s" % (r["name"], r["kept"], "" if r["named_ok"] else " (renamed)") for r in named)
                + ("  MERGE/ECHO CHANGED: " + ", ".join(odd) if odd else "")))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())

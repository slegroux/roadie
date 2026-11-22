"""How steady are the drum roles when the grid moves by an amount no ear would notice?

    python -m roadie_analysis.drumstability <stems_dir> [--stems hh,snare] [--jobs N] [--only tempo+0.001,t0-0.001]

Runs `drums.drum_roles` (nothing written) at the folder's grid and at perturbed grids (tempo +-0.001 and +-0.005 bpm, t0
+-1 ms), then compares, hit by hit (hits matched by onset time within MATCH_S), which role each hit lands in. Role names
move around when groups reorder, so roles are compared as sets of hits: for every base role of at least MIN_HITS hits, the
perturbed role that holds most of its hits is its match, and `kept` is the share of its hits that sit there (a hit that is
no longer in any role counts as lost). `agreement` is the same over all such base roles together. A named role is
`named_ok` when its match has the same name. Needs numpy and soundfile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MATCH_S = 0.003
MIN_HITS = 4
PERTURBATIONS = (("tempo", 0.001), ("tempo", -0.001), ("tempo", 0.005), ("tempo", -0.005), ("t0", 0.001), ("t0", -0.001))


def hit_roles(rec: dict, stem: str) -> tuple[np.ndarray, np.ndarray, list]:
    """(onset times sorted, role index per hit, [role names]) of one stem of a record made with `hit_times=True`."""
    roles = rec["stems"][stem]["roles"]
    pairs = sorted((t, k) for k, r in enumerate(roles) for t in r["hit_times"])
    return (np.array([t for t, _ in pairs]), np.array([k for _, k in pairs], dtype=int), [r["name"] for r in roles])


def compare(base: dict, other: dict, stem: str, match_s: float = MATCH_S, min_hits: int = MIN_HITS) -> dict:
    """{"agreement", "roles": [{"name", "named", "hits", "kept", "match", "match_name", "named_ok", "split"}], "named_same"}."""
    tb, rb, nb = hit_roles(base, stem)
    to, ro, no = hit_roles(other, stem)
    named_b = {r["name"] for r in base["stems"][stem]["roles"] if r["named"]}
    named_o = {r["name"] for r in other["stems"][stem]["roles"] if r["named"]}
    rows, kept_all, hits_all = [], 0, 0
    for k, role in enumerate(base["stems"][stem]["roles"]):
        idx = np.nonzero(rb == k)[0]
        if len(idx) < min_hits:
            continue
        where = []                                          # the role each of its hits has in `other`, -1 when it has none
        for t in tb[idx]:
            j = int(np.searchsorted(to, t))
            near = [m for m in (j - 1, j) if 0 <= m < len(to) and abs(to[m] - t) <= match_s]
            where.append(int(ro[min(near, key=lambda m: abs(to[m] - t))]) if near else -1)
        where = np.array(where)
        found = where[where >= 0]
        match = int(np.bincount(found).argmax()) if len(found) else -1
        kept = int((where == match).sum()) if match >= 0 else 0
        split = int(sum(1 for m in set(found.tolist()) if (where == m).sum() >= 0.1 * len(idx)))
        rows.append({"name": role["name"], "named": role["named"], "hits": int(len(idx)), "kept": round(kept / len(idx), 3),
                     "match": match, "match_name": no[match] if match >= 0 else None, "split": split,
                     "named_ok": (role["name"] == no[match]) if role["named"] and match >= 0 else (False if role["named"] else None)})
        kept_all += kept
        hits_all += len(idx)
    return {"agreement": round(kept_all / hits_all, 3) if hits_all else None, "roles": rows, "named_same": named_b == named_o,
            "named_base": sorted(named_b), "named_other": sorted(named_o)}


def run(stems_dir, stems=None, jobs=None, only=None, tempo=None, t0=None) -> dict:
    """The base run and the perturbed runs of `stems` ({name: record}); `stems` limits the drum stems analysed (a scratch
    folder of links is made, so nothing is written into `stems_dir`)."""
    from roadie_analysis import drums
    src = Path(stems_dir)
    g = drums.grid_from_folder(src)
    tempo = tempo or g["tempo"]
    bar_s = 60.0 / tempo * drums.BEATS_PER_BAR
    down = float(g["downbeat_s"])
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
            out[label] = drums.drum_roles(tmp, tempo=tt, t0=t, write=False, jobs=jobs, hit_times=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drumstability", description=__doc__.split("\n\n")[0])
    ap.add_argument("stems_dir")
    ap.add_argument("--stems", default="hh,snare", help="drum stems to analyse, comma separated (default hh,snare)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma separated perturbations to run, e.g. tempo+0.001,t0-0.001")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    stems = [s for s in args.stems.split(",") if s]
    runs = run(args.stems_dir, stems, args.jobs, args.only.split(",") if args.only else None)
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
            named = [r for r in c["roles"] if r["named"]]
            print("  %-14s agreement %s  worst role kept %s  named same %s  %s" % (
                label, c["agreement"], worst, c["named_same"],
                ", ".join("%s %.2f%s" % (r["name"], r["kept"], "" if r["named_ok"] else " (renamed)") for r in named)))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())

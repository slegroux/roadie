#!/usr/bin/env python3
"""`roadie study rename --dry-run [--root pcloud]`: the folder name each analysed study would get,
`<Title> [<bpm> <key> <camelot>]`, from its record, as one table. Nothing is renamed: `--write` does not
exist yet (the real rename is built after the user has seen this table).

The new name takes the key's short form (`Cm`, `Bbm`, `F#`, `Ab`) and a `?` after a detected key and its code;
a study with no key is `[<bpm>]`. The bpm is the listed one when within 1 of the measured one, else the
measured one (keyfill.name_bpm). The table also lists name collisions, the paths recorded elsewhere that a
rename would invalidate (a profile's input paths, the benchmark configs under `bench/`), and the special
cases that are proposed here and not executed: the Synthwave merge.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import keyfill
import study

MERGE_PACK, MERGE_BENCH = ("Rodriguez Jr.", "Synthwave #1 (remix pack)"), ("Rodriguez Jr.", "Synthwave 1 benchmark")
HERE = Path(__file__).resolve().parent


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def recorded_paths(ref: study.StudyRef, bench_dir: Path) -> list[str]:
    """What names this study's folder outside its own name: profile inputs under `/<artist>/<folder>/` and
    bench configs mentioning `/<artist>/<folder>`. One line each, with a count for profile inputs."""
    here = "/%s/%s" % (_nfc(ref.root.parent.name), _nfc(ref.root.name))
    out = []
    for p in study._json_files(ref.root):
        if not p.name.endswith(".profile.json"):
            continue
        prof = study._read_json(p)
        try:
            inputs = list(prof["inputs"]["mix"]) + [i for items in prof["inputs"]["stems"].values() for i in items]
            n = sum(1 for i in inputs if here + "/" in _nfc(i["path"]))
        except (KeyError, TypeError, AttributeError):
            continue
        if n:
            out.append("%s: %d recorded input paths contain %s/ (checked by name and size afterwards: stale, not lost)"
                       % (p.relative_to(ref.root), n, here))
    for f in sorted(bench_dir.glob("*.json")) if bench_dir.is_dir() else []:
        text = _nfc(f.read_text())
        if here + '"' in text or here + "/" in text:
            out.append("bench/%s: a path names %s" % (f.name, here))
    return out


def plan_rename(refs: list[study.StudyRef], roots, bench_dir: Path = HERE / "bench") -> dict:
    entries = []
    for ref in refs:
        s = study.load_study(str(ref.root))
        new = keyfill.folder_name(s.title, keyfill.name_bpm(s.bpm, s.listed_bpm), s.key, s.key_source == "detected")
        entries.append({"artist": ref.artist, "old": ref.root.name, "new": new, "title": s.title, "ref": ref,
                        "recorded": recorded_paths(ref, bench_dir)})
    existing = {}
    for ref in study.find_studies(roots):          # every folder, analysed or not
        existing.setdefault((ref.artist, _nfc(ref.root.name).casefold()), []).append(ref.root.name)
    by_new, collisions = {}, []
    for e in entries:
        if e["new"] is None:
            continue
        key = (e["artist"], _nfc(e["new"]).casefold())
        by_new.setdefault(key, []).append(e)
        other = [n for n in existing.get(key, []) if n != e["old"]]
        if other:
            collisions.append("%s/%s -> %s: a folder of that name exists (%s)" % (e["artist"], e["old"], e["new"], other[0]))
    collisions += ["%s: %s all become %s" % (k[0], " and ".join(x["old"] for x in v), v[0]["new"])
                   for k, v in by_new.items() if len(v) > 1]
    return {"entries": entries, "collisions": collisions, "merge": synthwave_merge(entries)}


def _top(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def synthwave_merge(entries: list[dict]) -> dict | None:
    """The proposed merge of the Synthwave benchmark into the remix pack: which top-level entries go where."""
    by = {(e["artist"], e["old"]): e for e in entries}
    pack, bench = by.get(MERGE_PACK), by.get(MERGE_BENCH)
    if not pack or not bench:
        return None
    pack_dir, bench_dir = pack["ref"].root, bench["ref"].root
    stay = [n for n in _top(pack_dir) if n.endswith("_stems") or n.lower().endswith(".wav") or n == "study.json"]
    into_pack = [n for n in _top(pack_dir) if n not in stay]
    return {"pack_dir": pack_dir, "bench_dir": bench_dir, "stay": stay, "into_pack": into_pack,
            "bench_items": _top(bench_dir), "name": pack["new"].replace("Synthwave #1 (remix pack)", "Synthwave #1")
            if pack["new"] else None}


def render(plan: dict, root_label: str) -> str:
    es, merge = plan["entries"], plan["merge"]
    changed = [e for e in es if e["new"] and e["new"] != e["old"]]
    L = ["# Rename dry run, root `%s`: %d analysed studies, %d would change\n" % (root_label, len(es), len(changed)),
         "Nothing was renamed (`study rename` has no `--write` yet). New name = `<Title> [<bpm> <key> <camelot>]` from the "
         "record: short key form, `?` after a detected key and its code, `[<bpm>]` with no key; the bpm is the listed "
         "one when within 1 of the measured one, else the measured one.\n",
         "| Artist | Old folder | New folder | Recorded paths it invalidates |", "|---|---|---|---|"]
    for e in es:
        tag = " (proposed merge, see below)" if (e["artist"], e["old"]) in (MERGE_PACK, MERGE_BENCH) else ""
        L.append("| %s | %s | %s%s | %s |" % (e["artist"], e["old"], e["new"] or "(no bpm: unchanged)", tag,
                                              "; ".join(e["recorded"]) or "-"))
    L += ["", "## Collisions (%d)\n" % len(plan["collisions"])]
    L += ["- " + c for c in plan["collisions"]] or ["none"]
    L += ["", "## Recorded paths elsewhere\n",
          "Reads of a study go by folder, so the reader finds a renamed study again; what would go stale is anything "
          "that stored the old path: a `.profile.json`'s input paths (its freshness check then falls back to file name "
          "and size, so a profile stays valid but is described as checked by name), the `bench/*.json` configs "
          "(`out_dir`), Live's recent-files list and `.als` sample references (not scanned: Live asks for the samples "
          "again). The per-study column above lists the profile and bench hits found.\n"]
    L += ["## Version and pack suffixes\n",
          "Titles with a version suffix (Extended Mix, a Remix, Cercle) name different audio from the plain track and keep "
          "their title as is; the new name only appends the bracket. The one pack suffix among the analysed studies is "
          "`(remix pack)`, handled by the merge below.\n"]
    if merge:
        L += ["## Proposed, not executed: merge the Synthwave studies\n",
              "`%s/%s` and `%s/%s` are one track. Proposed: one folder `%s/%s/`, built by server-side moves "
              "(`rclone moveto`), one folder at a time, each followed by a listing check.\n" % (
                  MERGE_PACK + MERGE_BENCH + (MERGE_PACK[0], merge["name"])),
              "- Stays at the study root (the reader needs a `*_stems` folder here): %s" % ", ".join("`%s`" % n for n in merge["stay"]),
              "- Moves into `pack/` (%d entries; the remix pack's samples and Ableton project files): %s" % (
                  len(merge["into_pack"]), ", ".join("`%s`" % n for n in merge["into_pack"] if not n.lower().endswith((".aif", ".asd")))
                  + ", and %d `.aif`/`.asd` sample files" % sum(1 for n in merge["into_pack"] if n.lower().endswith((".aif", ".asd")))),
              "- The whole benchmark folder moves into `benchmark/` (%d entries: %s)." % (
                  len(merge["bench_items"]), ", ".join("`%s`" % n for n in merge["bench_items"])),
              "- The merged root gets the remix pack's record as `study.json` with no `stems` field, so the reader takes the "
              "separated stems (`*_stems/`) at the root; the benchmark's own marker (`stems: truth`) moves with it to "
              "`benchmark/study.json` where nothing reads it as a study.",
              "- What the reader then sees: grid, sections and activity from the separated stems; the `mix` layer from "
              "`benchmark/analysis/mix.profile.json`, because the reader walks every folder under a study for profiles and "
              "lists the benchmark's other JSON (scores, roles, baselines) as `extras`. The benchmark's truth stems stay "
              "readable at `benchmark/truth/`.",
              "- Invalidated by the merge: `bench/synthwave1.json` `out_dir` (now `.../%s/benchmark`), the benchmark profile's "
              "input paths, and the `.als` files' sample paths under `pack/`." % merge["name"], ""]
    return "\n".join(L) + "\n"


def main(args) -> int:
    if args.write:
        print("study: rename: --write is not built yet; this is the dry run the real rename waits on", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("study: rename: give --dry-run", file=sys.stderr)
        return 1
    roots = [r for r in study.STUDIES_ROOTS if r.label == args.root]
    if not roots:
        print("study: rename: no root labelled %r" % args.root, file=sys.stderr)
        return 1
    if args.out and any(Path(r.path).resolve() in Path(args.out).resolve().parents for r in study.STUDIES_ROOTS):
        print("study: rename: %s is inside a Studies folder: nothing is written there" % args.out, file=sys.stderr)
        return 1
    try:
        plan = plan_rename([r for r in study.find_studies(roots) if r.analysed], roots)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print("study: rename: %s" % e, file=sys.stderr)
        return 1
    text = render(plan, args.root)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0

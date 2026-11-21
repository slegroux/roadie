#!/usr/bin/env python3
"""`roadie study rename --dry-run | --write | --undo MAP.json`: the folder name each analysed study gets,
`<Title> [<bpm> <key> <camelot>]`, from its record. `--dry-run` prints the table and renames nothing.
`--write` renames on the pCloud library with server-side `rclone moveto` only (see `run_moves`), after saving
the plan as a map (old -> new, the undo) and a recursive listing; `--undo` plays a map backwards.

The new name takes the key's short form (`Cm`, `Bbm`, `F#`, `Ab`) and a `?` after a detected key and its code;
a study with no key is `[<bpm>]`. The bpm is the listed one when within 1 of the measured one, else the
measured one (keyfill.name_bpm). The table also lists name collisions, the paths recorded elsewhere that a
rename would invalidate (a profile's input paths, the benchmark configs under `bench/`), and the special
cases that are proposed here and not executed: the Synthwave merge.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import keyfill
import study
from roadie_analysis import key as keymod

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


# ---- the real thing: server-side moves on rclone, checked one at a time

def _rclone(*args) -> str:
    return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout


def r_files(path: str) -> list[tuple[int, str]]:
    """(size, relative path) of every file under a folder (or the file itself), by rclone."""
    out = []
    for line in _rclone("lsf", "-R", "--files-only", "--format", "sp", path).splitlines():
        size, _, rel = line.partition(";")
        out.append((int(size), rel))
    return out


def r_exists(path: str) -> bool:
    parent, _, name = path.rpartition("/")
    try:
        return name in {n.rstrip("/") for n in _rclone("lsf", parent).splitlines()}
    except subprocess.CalledProcessError as e:
        if "directory not found" in (e.stderr or ""):
            return False
        raise


def r_names(path: str) -> list[str]:
    """The names directly inside a folder (a folder's name without its trailing slash)."""
    return sorted(n.rstrip("/") for n in _rclone("lsf", path).splitlines())


def build_moves(plan: dict, root: str) -> list[dict]:
    """The ordered moves of a rename plan on the remote `root`: each study's folder to its new name, then
    the Synthwave merge (the pack folder renamed, its entries moved into `pack/`, the benchmark moved into
    `benchmark/`). Each move is {old, new, group}, paths relative to `root`; names are checked against the
    remote when the move runs."""
    merge = plan["merge"]
    in_merge = {MERGE_PACK, MERGE_BENCH} if merge else set()
    moves = []
    for e in plan["entries"]:
        if (e["artist"], e["old"]) in in_merge or not e["new"] or e["new"] == e["old"]:
            continue
        moves.append({"old": "%s/%s" % (e["ref"].root.parent.name, e["old"]),
                      "new": "%s/%s" % (e["ref"].root.parent.name, e["new"]), "group": None})
    if merge:
        artist = MERGE_PACK[0]
        old_pack = "%s/%s" % (artist, merge["pack_dir"].name)
        merged = "%s/%s" % (artist, merge["name"])
        entries = [n for n in r_names("%s/%s" % (root, old_pack))
                   if not (n.endswith("_stems") or n.lower().endswith(".wav") or n == "study.json")]
        moves.append({"old": old_pack, "new": merged, "group": "synthwave"})
        moves += [{"old": "%s/%s" % (merged, n), "new": "%s/pack/%s" % (merged, n), "group": "synthwave"} for n in entries]
        moves.append({"old": "%s/%s" % (artist, merge["bench_dir"].name), "new": merged + "/benchmark", "group": "synthwave"})
    return moves


def run_moves(root: str, moves: list[dict], log=print, merge_check: tuple[list[str], str] | None = None) -> dict:
    """Run moves one at a time with `rclone moveto` (server-side: no copy, no delete, never the FUSE mount).
    Per move: the target must not exist (else it is SKIPPED, and the rest of its group with it: nothing is ever
    merged into an existing folder), the source must; files are counted before at the old path and after at
    the new one, and the old path must be gone. Any failed check stops the whole run. `merge_check` is
    (old paths, new path): after the last move of the group their total file count must equal the new one's.
    Returns {ok, skipped, failed, counts}."""
    done, skipped, skipped_groups, failed = [], [], set(), None
    total_before = {}
    last = {m.get("group"): i for i, m in enumerate(moves) if m.get("group")}
    for i, m in enumerate(moves):
        old, new, group = m["old"], m["new"], m.get("group")
        src, dst = "%s/%s" % (root, old), "%s/%s" % (root, new)
        if group and group in skipped_groups:
            skipped.append(m)
            log("SKIPPED %s -> %s: an earlier step of the group was skipped" % (old, new))
            continue
        try:
            if r_exists(dst):
                skipped.append(m)
                if group:
                    skipped_groups.add(group)
                log("SKIPPED %s -> %s: the target exists (nothing is merged into an existing folder)" % (old, new))
                continue
            if not r_exists(src):
                raise RuntimeError("the source is not there")
            before = len(r_files(src))
            if group and merge_check and group not in total_before:
                total_before[group] = sum(len(r_files("%s/%s" % (root, p))) for p in merge_check[0])
            _rclone("moveto", src, dst)
            after = len(r_files(dst))
            if before != after:
                raise RuntimeError("%d files before, %d after" % (before, after))
            if r_exists(src):
                raise RuntimeError("the old path is still there")
        except (RuntimeError, subprocess.CalledProcessError, ValueError) as e:
            why = (e.stderr or "").strip() if isinstance(e, subprocess.CalledProcessError) else e
            failed = {"move": m, "why": str(why)}
            log("FAILED %s -> %s: %s. STOPPED: nothing else was moved." % (old, new, why))
            break
        done.append(m)
        log("OK %s -> %s (%d files)" % (old, new, before))
        if group and merge_check and last[group] == i:
            n_after = len(r_files("%s/%s" % (root, merge_check[1])))
            if n_after != total_before.get(group):
                failed = {"move": m, "why": "merged folder holds %d files, the two old folders held %d" % (n_after, total_before.get(group))}
                log("FAILED merge total: %s" % failed["why"])
                break
            log("OK merge total: %d files under %s, the same as in the old folders" % (n_after, merge_check[1]))
    return {"ok": done, "skipped": skipped, "failed": failed}


def undo_moves(map_: dict, log=print) -> dict:
    """The saved moves reversed, last first, checked the same way."""
    back = [{"old": m["new"], "new": m["old"], "group": None} for m in reversed(map_["moves"])]
    return run_moves(map_["root"], back, log)


def compare_listings(before: list[tuple[int, str]], after: list[tuple[int, str]], moves: list[dict]) -> list[str]:
    """Differences between two library listings across the moves: file counts, the multiset of
    (base name, size), and each file's size at its expected new path. An empty result means identical."""
    from collections import Counter
    out = []
    if len(before) != len(after):
        out.append("file count %d before, %d after" % (len(before), len(after)))
    base = lambda lst: Counter((rel.rpartition("/")[2], size) for size, rel in lst)
    for k, v in ((base(before) - base(after)).items()):
        out.append("missing after: %s (%d bytes) x%d" % (k[0], k[1], v))
    for k, v in ((base(after) - base(before)).items()):
        out.append("new after: %s (%d bytes) x%d" % (k[0], k[1], v))

    def mapped(rel):
        # apply the moves in order: a file inside a renamed folder, then inside an entry moved into pack/
        for m in moves:
            if rel == m["old"] or rel.startswith(m["old"] + "/"):
                rel = m["new"] + rel[len(m["old"]):]
        return rel
    expect = Counter((mapped(rel), size) for size, rel in before)
    got = Counter((rel, size) for size, rel in after)
    for (rel, size), n in (expect - got).items():
        out.append("expected at %s (%d bytes) but not there" % (rel, size))
    for (rel, size), n in (got - expect).items():
        out.append("unexpected at %s (%d bytes)" % (rel, size))
    return out


def main(args) -> int:
    if args.undo:
        map_ = json.loads(Path(args.undo).read_text())
        res = undo_moves(map_)
        return 1 if res["failed"] or res["skipped"] else 0
    if args.write:
        return main_write(args)
    if not args.dry_run:
        print("study: rename: give --dry-run, --write or --undo MAP.json", file=sys.stderr)
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


def main_write(args) -> int:
    roots = [r for r in study.STUDIES_ROOTS if r.label == args.root]
    root = keymod.pcloud_remote(roots[0].path) if roots else None
    if not root or not args.out_dir:
        print("study: rename: --write needs --root to be a pCloud library root, and --out-dir for the map, "
              "listings and log", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    if any(Path(r.path).resolve() in out.resolve().parents or Path(r.path).resolve() == out.resolve() for r in study.STUDIES_ROOTS):
        print("study: rename: --out-dir is inside a Studies folder", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp
    lines = []

    def log(line):
        lines.append(line)
        print(line, flush=True)

    plan = plan_rename([r for r in study.find_studies(roots) if r.analysed], roots)
    moves = build_moves(plan, root)
    (out / ("rename-map-%s.json" % stamp)).write_text(json.dumps({"kind": "rename_map", "version": 1, "root": root, "moves": moves}, indent=1, ensure_ascii=False))
    log("map: %d moves saved to rename-map-%s.json" % (len(moves), stamp))
    before = r_files(root)
    (out / ("listing-before-%s.txt" % stamp)).write_text("".join("%d;%s\n" % f for f in before))
    log("listing before: %d files" % len(before))
    merge = plan["merge"]
    check = (["%s/%s" % (MERGE_PACK[0], merge["pack_dir"].name), "%s/%s" % (MERGE_BENCH[0], merge["bench_dir"].name)],
             "%s/%s" % (MERGE_PACK[0], merge["name"])) if merge else None
    res = run_moves(root, moves, log, check)
    after = r_files(root)
    (out / ("listing-after-%s.txt" % stamp)).write_text("".join("%d;%s\n" % f for f in after))
    diffs = compare_listings(before, after, res["ok"])
    log("listing after: %d files; before/after: %s" % (len(after), "identical (counts, names and sizes, and each file at its expected path)" if not diffs else "%d DIFFERENCES" % len(diffs)))
    for d in diffs[:50]:
        log("  " + d)
    log("done: %d moved, %d skipped, %s" % (len(res["ok"]), len(res["skipped"]), "FAILED, stopped" if res["failed"] else "no failure"))
    (out / ("rename-log-%s.txt" % stamp)).write_text("\n".join(lines) + "\n")
    return 1 if res["failed"] else 0

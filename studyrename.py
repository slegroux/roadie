#!/usr/bin/env python3
"""`roadie study rename --dry-run | --write | --undo MAP.json`: the folder name each analysed study gets,
`<Title> [<bpm> <key> <camelot>]`, from its record. `--dry-run` prints the table and renames nothing.
`--write` renames on the pCloud library with server-side `rclone moveto` only (see `run_moves`), after saving
the plan as a map (old -> new, the undo) and a recursive listing; `--undo` plays a map backwards.

The new name takes the key's short form (`Cm`, `Bbm`, `F#`, `Ab`) and a `?` after a detected key and its code;
a study with no key is `[<bpm>]`. The bpm is the listed one when within 1 of the measured one, else the
measured one (keyfill.name_bpm). The table also lists name collisions, the paths recorded elsewhere that a
rename would invalidate (a profile's input paths, the benchmark configs under `bench/`). The one-off merge of
the Synthwave benchmark into its remix pack was done by hand-built moves on 2026-11-21 and is no longer here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import keyfill
import study
from roadie_analysis import key as keymod

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


def plan_rename(refs: list[study.StudyRef], roots, bench_dir: Path = HERE / "bench", remote_records: bool = False) -> dict:
    """The names for `refs`. With `remote_records`, each study's key, key source and listed bpm come from its
    `study.json` read through rclone (the FUSE mount can lag a write): what `--write` uses."""
    entries = []
    for ref in refs:
        s = study.load_study(str(ref.root))
        remote = keymod.pcloud_remote(ref.root / "study.json") if remote_records else None
        if remote:
            text = study._remote_read(remote)
            f = study._key_fields(study._parse_marker(text) if text else None)
            s.key, s.key_source, s.listed_bpm = f["key"], f["key_source"], f["listed_bpm"]
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
    return {"entries": entries, "collisions": collisions}


def render(plan: dict, root_label: str) -> str:
    es = plan["entries"]
    changed = [e for e in es if e["new"] and e["new"] != e["old"]]
    L = ["# Rename dry run, root `%s`: %d analysed studies, %d would change\n" % (root_label, len(es), len(changed)),
         "This table renames nothing. New name = `<Title> [<bpm> <key> <camelot>]` from the "
         "record: short key form, `?` after a detected key and its code, `[<bpm>]` with no key; the bpm is the listed "
         "one when within 1 of the measured one, else the measured one.\n",
         "| Artist | Old folder | New folder | Recorded paths it invalidates |", "|---|---|---|---|"]
    for e in es:
        L.append("| %s | %s | %s | %s |" % (e["artist"], e["old"], e["new"] or "(no bpm: refused)",
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
          "their title as is; the new name only appends the bracket.\n"]
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


def r_exists(path: str, fold: bool = False) -> bool:
    """Is there a file or folder of this name? With `fold`, names that differ only by case or Unicode form
    count as the same (the check for a move's target: nothing may be merged into a near-identical folder)."""
    parent, _, name = path.rpartition("/")
    key = (lambda n: _nfc(n).casefold()) if fold else (lambda n: n)
    try:
        return key(name) in {key(n.rstrip("/")) for n in _rclone("lsf", parent).splitlines()}
    except subprocess.CalledProcessError as e:
        if "directory not found" in (e.stderr or ""):
            return False
        raise


def build_moves(plan: dict) -> list[dict]:
    """The moves of a rename plan: each study's folder to its new name, paths relative to the library root.
    A plan entry that has no new name (no bpm) is refused rather than skipped."""
    moves = []
    for e in plan["entries"]:
        if not e["new"]:
            raise ValueError("%s/%s has no bpm: it cannot be named" % (e["artist"], e["old"]))
        if e["new"] != e["old"]:
            artist = e["ref"].root.parent.name
            moves.append({"old": "%s/%s" % (artist, e["old"]), "new": "%s/%s" % (artist, e["new"])})
    return moves


def run_moves(root: str, moves: list[dict], log=print) -> dict:
    """Run moves one at a time with `rclone moveto` (server-side: no copy, no delete, never the FUSE mount).
    Per move: the target must not exist, by exact name or by NFC and case-folded name (else it is SKIPPED:
    nothing is ever merged into an existing folder), the source must; files are counted before at the old
    path and after at the new one, and the old path must be gone. Any failed check stops the whole run.
    Returns {ok, skipped, failed}."""
    done, skipped, failed = [], [], None
    for m in moves:
        old, new = m["old"], m["new"]
        src, dst = "%s/%s" % (root, old), "%s/%s" % (root, new)
        try:
            if r_exists(dst, fold=True):
                skipped.append(m)
                log("SKIPPED %s -> %s: the target exists (nothing is merged into an existing folder)" % (old, new))
                continue
            if not r_exists(src):
                raise RuntimeError("the source is not there")
            before = len(r_files(src))
            _rclone("moveto", src, dst)
            after = len(r_files(dst))
            if before != after:
                raise RuntimeError("%d files before, %d after" % (before, after))
            if r_exists(src):
                raise RuntimeError("the old path is still there")
        except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError) as e:
            why = (e.stderr or "").strip() if isinstance(e, subprocess.CalledProcessError) else e
            failed = {"move": m, "why": str(why)}
            log("FAILED %s -> %s: %s. STOPPED: nothing else was moved." % (old, new, why))
            break
        done.append(m)
        log("OK %s -> %s (%d files)" % (old, new, before))
    return {"ok": done, "skipped": skipped, "failed": failed}


def undo_moves(map_: dict, log=print) -> dict:
    """The saved moves reversed, last first, checked the same way."""
    back = [{"old": m["new"], "new": m["old"]} for m in reversed(map_["moves"])]
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
    stamp = args.stamp or time.strftime("%Y%m%d-%H%M%S")
    names = {k: out / ("%s-%s.%s" % (k, stamp, ext)) for k, ext in (("rename-map", "json"), ("listing-before", "txt"),
                                                                   ("listing-after", "txt"), ("rename-log", "txt"))}
    try:
        plan = plan_rename([r for r in study.find_studies(roots) if r.analysed], roots, remote_records=True)
        if plan["collisions"]:
            raise ValueError("the plan has collisions, nothing is renamed:\n  " + "\n  ".join(plan["collisions"]))
        moves = build_moves(plan)
        out.mkdir(parents=True, exist_ok=True)
        for f in names.values():                      # exclusive: an earlier run's map and listings are never replaced
            f.open("x").close()
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print("study: rename: %s" % e, file=sys.stderr)
        return 1
    logf = names["rename-log"].open("a")

    def log(line):                                    # on disk as it goes: an interrupted run still has its log
        logf.write(line + "\n")
        logf.flush()
        print(line, flush=True)

    try:
        names["rename-map"].write_text(json.dumps({"kind": "rename_map", "version": 1, "root": root, "moves": moves},
                                                  indent=1, ensure_ascii=False))
        log("map: %d moves saved to %s" % (len(moves), names["rename-map"].name))
        before = r_files(root)
        names["listing-before"].write_text("".join("%d;%s\n" % f for f in before))
        log("listing before: %d files" % len(before))
        res = run_moves(root, moves, log)
        after = r_files(root)
        names["listing-after"].write_text("".join("%d;%s\n" % f for f in after))
        diffs = compare_listings(before, after, res["ok"])
        log("listing after: %d files; before/after: %s" % (len(after), "identical (counts, names and sizes, and each file "
            "at its expected path)" if not diffs else "%d DIFFERENCES" % len(diffs)))
        for d in diffs[:50]:
            log("  " + d)
        log("done: %d moved, %d skipped, %s" % (len(res["ok"]), len(res["skipped"]), "FAILED, stopped" if res["failed"] else "no failure"))
    finally:
        logf.close()
    return 1 if res["failed"] or diffs else 0

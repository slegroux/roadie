#!/usr/bin/env python3
"""`roadie study fill-keys --from LOOKUP.json --dry-run | --write`: decide, for every analysed study of a library
root, what its `study.json` gets (key, source, listed bpm) and print one table. `--dry-run` writes nothing
under the Studies folders (only the table and an optional detection cache are files). `--write` records
each study through `study.tag_study`, saving any study.json it replaces first; it touches no other file.

The rules, from the user's decisions on plan roadie/03 item 3:
- a published key (a lookup row matched to the study by its `study` field, exactly) is recorded as `online`
  with its URL and its listed bpm; `decisions` in the lookup file override a study's key, source and note
  (the user's old tag kept as `key_note`);
- a `decisions` entry with `no_key` records a note and no key (a whole-set recording);
- a row marked `use_published: false` (identity not established) is not used: that study gets a detected key only;
- a study with no published key gets the detected key as `detected`, only when both stems were analysed;
  otherwise no key;
- detection (`roadie key`) runs on every study as the second opinion; a published key it does not agree
  with exactly is a disagreement (the published one is still what would be written).
A row matching no study, a decision matching none, and a study matching no row are listed, never guessed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import study
from roadie_analysis import key as keymod

NOTES_NO_KEY = "no bpm: not renamed until one is known"


def relation(a: str, b: str) -> str:
    """How two keys sit on the Camelot wheel: exact, relative, fifth, parallel or different."""
    (pa, ma), (pb, mb) = keymod.parse_key(a), keymod.parse_key(b)
    ca, cb = keymod.camelot(pa, ma), keymod.camelot(pb, mb)
    if ca == cb:
        return "exact"
    na, nb = int(ca[:-1]), int(cb[:-1])
    if na == nb:
        return "relative"
    if ca[-1] == cb[-1] and (na - nb) % 12 in (1, 11):
        return "fifth"
    return "parallel" if pa == pb else "different"


def name_bpm(measured: float | None, listed: float | None) -> float | None:
    """The tempo a folder name carries: the listed one when it is within 1 bpm of the measured one (or nothing
    was measured), else the measured one: the user's choice, 2026-11-20."""
    if listed is not None and (measured is None or abs(listed - measured) <= 1):
        return listed
    return measured if measured is not None else None


def folder_name(title: str, bpm: float | None, key: str | None, detected: bool) -> str | None:
    """`Title [124 Am 8A]` (a `?` after an unconfirmed key and code); `Title [124]` with no key; None with no bpm."""
    if bpm is None:
        return None
    if not key:
        return "%s [%d]" % (title, int(bpm + 0.5))
    pc, mode = keymod.parse_key(key)
    q = "?" if detected else ""
    return "%s [%d %s%s %s%s]" % (title, int(bpm + 0.5), keymod.short_name(pc, mode), q, keymod.camelot(pc, mode), q)


def _detect(ref: study.StudyRef, detector) -> dict:
    try:
        r = detector(str(ref.root))
    except (ValueError, OSError) as e:
        return {"ok": False, "reason": str(e).splitlines()[0][:120]}
    except Exception as e:      # rclone failures and the like: recorded for this study, the run goes on
        return {"ok": False, "reason": "%s: %s" % (type(e).__name__, str(e)[:100])}
    return {"ok": True, "key": r["key"], "camelot": r["camelot"], "confidence": r["confidence"]}


def decide(name: str, measured, row: dict | None, decision: dict | None, det: dict) -> dict:
    """One study's entry: what would be written, and how it sits against detection."""
    pub = row if row and row.get("use_published", True) else None
    out = {"study": name, "path": None, "measured_bpm": measured, "listed_bpm": pub["bpm"] if pub else None,
           "bpm_url": pub["url"] if pub else None, "key": None, "camelot": None, "source": None, "url": None,
           "key_confidence": None, "key_note": None, "detected": det, "notes": []}
    if row and not pub:
        out["notes"].append("lookup row not used: " + row.get("unused_reason", "marked not to be used"))
    if decision and decision.get("no_key"):
        out["key_note"] = decision["key_note"]
    elif decision:
        out.update(key=decision["key"], source=decision["source"], key_note=decision.get("key_note"))
        out["url"] = decision.get("url") or (row["url"] if row and decision["source"] == "online" else None)
    elif pub:
        out.update(key=pub["key"], source="online", url=pub["url"])
    elif det["ok"]:
        out.update(key=det["key"], source="detected", key_confidence=det["confidence"])
    if out["key"]:
        pc, mode = keymod.parse_key(out["key"])
        out["key"], out["camelot"] = keymod.key_name(pc, mode), keymod.camelot(pc, mode)
    if out["source"] in ("online", "tagged"):
        if not det["ok"]:
            out["verdict"] = "published only (no detection: %s)" % det["reason"]
        else:
            rel = relation(det["key"], out["key"])
            out["verdict"] = "agree" if rel == "exact" else "disagree (%s)" % rel
    elif out["source"] == "detected":
        out["verdict"] = "detected only"
    elif decision and decision.get("no_key"):
        out["verdict"] = "no key (%s)" % decision["key_note"]
    else:
        out["verdict"] = "no listing, no key (%s)" % det.get("reason", "?")
    if out["verdict"].startswith("disagree"):      # the disagreement stays visible in the record
        why = "detection disagrees (%s): %s %s, confidence %.3f" % (
            out["verdict"][len("disagree ("):-1], det["key"], det["camelot"], det["confidence"])
        out["key_note"] = "%s; %s" % (out["key_note"], why) if out["key_note"] else why
    if not row:
        out["notes"].append("no lookup row")
    return out


def existing_marker(folder: Path) -> dict | None:
    """The study.json marker on record for a study, read through rclone for a pCloud one (never the mount)."""
    path = folder / "study.json"
    remote = keymod.pcloud_remote(path)
    text = study._remote_read(remote) if remote else (path.read_text() if path.exists() else None)
    return study._parse_marker(text) if text else None


def needs_write(entry: dict, existing: dict | None, rewrite_all: bool) -> bool:
    """A study is written when its record has no key yet (or, for a whole set, not this note yet); a study that
    has a key is left alone unless `rewrite_all`."""
    if rewrite_all:
        return True
    if existing and existing.get("key"):
        return False
    if entry["key"] is None and entry["listed_bpm"] is None:
        return (existing or {}).get("key_note") != entry["key_note"]
    return True


def run_plan(lookup: dict, refs, detector=keymod.key_of, cache: dict | None = None, save_cache=None,
             min_free_gb: float = 3.0, rewrite_all: bool = False) -> dict:
    """Entries for every study in `refs` (analysed StudyRefs), plus the unmatched rows and decisions."""
    rows = {study._norm(r["study"]): r for r in lookup["rows"]}
    decisions = {study._norm(k): v for k, v in lookup.get("decisions", {}).items()}
    cache = {} if cache is None else cache
    entries, seen = [], set()
    for ref in refs:
        name = "%s/%s" % (ref.artist, ref.title)
        k = study._norm(name)
        seen.add(k)
        if name not in cache:
            if shutil.disk_usage(tempfile.gettempdir()).free < min_free_gb * 2 ** 30:
                raise RuntimeError("stopped before %s: under %g GB free" % (name, min_free_gb))
            cache[name] = _detect(ref, detector)
            if save_cache:
                save_cache(cache)
        try:
            measured = study.load_study(str(ref.root)).bpm
        except (ValueError, OSError, KeyError, TypeError, AttributeError):
            measured = None
        e = decide(name, measured, rows.get(k), decisions.get(k), cache[name])
        e["path"] = str(ref.root)
        e["existing"] = existing_marker(ref.root)
        e["write"] = needs_write(e, e["existing"], rewrite_all)
        e["folder_name"] = folder_name(ref.title, name_bpm(measured, e["listed_bpm"]), e["key"], e["source"] == "detected")
        if e["folder_name"] is None:
            e["notes"].append(NOTES_NO_KEY)
        entries.append(e)
    return {"entries": entries,
            "unmatched_rows": [r["study"] for r in lookup["rows"] if study._norm(r["study"]) not in seen],
            "unmatched_decisions": [k for k in lookup.get("decisions", {}) if study._norm(k) not in seen]}


def render(plan: dict, root_label: str) -> str:
    es = plan["entries"]
    count = lambda src: sum(1 for e in es if e["source"] == src)
    dis = [e for e in es if e["verdict"].startswith("disagree")]
    L = ["# Keys table, root `%s`, %d analysed studies\n" % (root_label, len(es)),
         "`key to write` is what `study tag` records (`listed bpm` with it); `folder name` is what the rename pass would "
         "give (the listed bpm rounded when within 1 of the measured one, else the measured one; `?` after a detected "
         "key). The key note carries the old tag, or the detection that disagreed.\n",
         "**Counts:** online %d, tagged %d, detected %d, no key %d; disagreements %d.\n" % (
             count("online"), count("tagged"), count("detected"), sum(1 for e in es if e["source"] is None), len(dis)),
         "**Records to write: %d** (a study whose record already has a key is left alone unless `--all`).\n" % sum(
             1 for e in es if e["write"]),
         "| Study | Measured bpm | Listed bpm | Key to write | Camelot | Source | Detected (confidence) | Verdict | Folder name | Notes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for e in es:
        d = e["detected"]
        det = "%s %s (%.3f)" % (d["key"], d["camelot"], d["confidence"]) if d["ok"] else "none: %s" % d["reason"]
        note = "; ".join(e["notes"] + (["key_note: %s" % e["key_note"]] if e["key_note"] else []))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e["study"], "%g" % e["measured_bpm"] if e["measured_bpm"] is not None else "-",
            "%g" % e["listed_bpm"] if e["listed_bpm"] is not None else "-", e["key"] or "-", e["camelot"] or "-",
            (e["source"] + ("?" if e["source"] == "detected" else "")) if e["source"] else "-", det, e["verdict"],
            e["folder_name"] or "-", note or "-"))
    L += ["", "## Disagreements (%d)\n" % len(dis)]
    L += ["- %s: would write %s (%s, %s); detection says %s (%.3f), %s" % (
        e["study"], e["key"], e["camelot"], e["source"], e["detected"]["key"], e["detected"]["confidence"],
        e["verdict"]) for e in dis] or ["none"]
    L += ["", "## Unmatched\n",
          "- lookup rows with no study in this root: %s" % (", ".join(plan["unmatched_rows"]) or "none"),
          "- decisions with no study in this root: %s" % (", ".join(plan["unmatched_decisions"]) or "none"),
          "- studies with no lookup row: %s" % (", ".join(e["study"] for e in es if "no lookup row" in e["notes"]) or "none")]
    return "\n".join(L) + "\n"


def _free_name(directory: Path, name: str) -> Path:
    """`name`, or `name.1`, `name.2`...: a backup that exists is never replaced."""
    path, n = directory / name, 0
    while path.exists():
        n += 1
        path = directory / ("%s.%d" % (name, n))
    return path


def write_records(plan: dict, backup_dir: Path, log=print, rewrite_all: bool = False) -> tuple[int, int]:
    """Record each entry marked `write` in its study's `study.json` through `study.tag_study` (rclone read, write,
    read-back for a pCloud study), one study at a time, saving the study.json it replaces first (under a free
    name). Without `rewrite_all` an existing `listed_bpm` is kept. One log line per study. A study that fails is
    logged and the run goes on. Returns (ok, failed)."""
    ok = failed = 0
    backup_dir.mkdir(parents=True, exist_ok=True)
    for e in plan["entries"]:
        if not e["write"]:
            log("SKIPPED %s: nothing to record (its record has a key; --all rewrites)" % e["study"])
            continue
        folder = Path(e["path"])
        try:
            if e["key"] is None and e["listed_bpm"] is None and not e["key_note"]:
                raise ValueError("nothing to record")
            path = folder / "study.json"
            remote = keymod.pcloud_remote(path)
            old = study._remote_read(remote) if remote else (path.read_text() if path.exists() else None)
            if old is not None:
                _free_name(backup_dir, "%s__%s.study.json" % (folder.parent.name, folder.name)).write_text(old)
            kept_bpm = not rewrite_all and (e["existing"] or {}).get("listed_bpm") is not None
            study.tag_study(folder, key=e["key"], bpm=None if kept_bpm else e["listed_bpm"], source=e["source"] or "tagged",
                            confidence=e["key_confidence"], url=e["url"], bpm_url=None if kept_bpm else e["bpm_url"],
                            note=e["key_note"])
            log("OK %s: %s%s%s%s" % (
                e["study"], "key %s (%s)" % (e["key"], e["source"]) if e["key"] else "no key",
                ", listed bpm %g" % e["listed_bpm"] if e["listed_bpm"] is not None and not kept_bpm else "",
                ", note: %s" % e["key_note"] if e["key_note"] else "",
                " (replaced a study.json: saved)" if old is not None else ""))
            ok += 1
        except Exception as ex:
            err = ex.stderr.strip() if getattr(ex, "stderr", None) else ex
            log("FAILED %s: %s" % (e["study"], err))
            failed += 1
    log("done: %d OK, %d FAILED" % (ok, failed))
    return ok, failed


def main(args) -> int:
    if not args.dry_run and not args.write:
        print("study: fill-keys: give --dry-run (the table only) or --write (record the keys)", file=sys.stderr)
        return 1
    if args.write and (args.dry_run or args.root == "local" or not args.backup_dir):
        print("study: fill-keys: --write is for one library root (not local), without --dry-run, with --backup-dir",
              file=sys.stderr)
        return 1
    roots = [r for r in study.STUDIES_ROOTS if r.label == args.root]
    if not roots:
        print("study: fill-keys: no root labelled %r" % args.root, file=sys.stderr)
        return 1
    for f in (args.out, args.cache, args.log):
        if f and any(Path(r.path).resolve() in Path(f).resolve().parents for r in study.STUDIES_ROOTS):
            print("study: fill-keys: %s is inside a Studies folder: nothing is written there" % f, file=sys.stderr)
            return 1
    try:
        lookup = json.loads(Path(args.lookup).read_text())
        cache = json.loads(Path(args.cache).read_text()) if args.cache and Path(args.cache).exists() else {}
        refs = [r for r in study.find_studies(roots) if r.analysed]
        save = (lambda c: Path(args.cache).write_text(json.dumps(c, indent=1))) if args.cache else None
        plan = run_plan(lookup, refs, cache=cache, save_cache=save, rewrite_all=args.all)
    except (ValueError, OSError, RuntimeError, KeyError) as e:
        print("study: fill-keys: %s" % e, file=sys.stderr)
        return 1
    text = render(plan, args.root)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if args.write:
        lines = []
        def log(line):
            lines.append(line)
            print(line, flush=True)
        _, failed = write_records(plan, Path(args.backup_dir), log, args.all)
        if args.log:
            Path(args.log).write_text("\n".join(lines) + "\n")
        return 1 if failed else 0
    return 0

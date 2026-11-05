#!/usr/bin/env python3
"""roadie batch — unattended, one-track-at-a-time Roadie runs with pCloud storage.

  batch.py plan ROOT... --out MANIFEST.tsv     inventory -> manifest + exclusions
  batch.py run MANIFEST.tsv [options]          process the manifest sequentially
  batch.py verify-sources MANIFEST.tsv         were any sources touched?

Each `new` row is processed in a scratch dir (outside the sources), uploaded
with rclone, verified, and only then deleted. Sources are only ever read. Every
delete goes through safe_rmtree(), which refuses anything not strictly inside
the work root. Standard library only.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from collections import namedtuple

REPO = os.path.dirname(os.path.realpath(__file__))
CACHE = os.path.expanduser("~/.cache/roadie-batch")
DEFAULT_DEST = "pcloud:Music/Production/Roadie"
DEFAULT_REPORT_DIR = os.path.expanduser("~/.omc/reports/2026-11-04-yt2live-overnight-batch")

AUDIO_EXTS = {"mp3", "wav", "flac", "aif", "aiff", "m4a", "ogg"}
AIF_EXTS = {"aif", "aiff"}
FORMAT_RANK = {"wav": 0, "flac": 0, "aif": 0, "aiff": 0, "m4a": 1, "ogg": 2, "mp3": 3}
COLUMNS = ["id", "mode", "tree", "rel_dir", "src", "name", "duration_sec", "size", "mtime", "dest"]

# Timeouts (seconds). Module-level so tests can shrink them.
SPLIT_FACTOR = 3
SPLIT_PAD = 600
SPLIT_NODUR = 3600
POST_TIMEOUT = 900
LSF_TIMEOUT = 300
UPLOAD_TIMEOUT = 7200
CHECK_TIMEOUT = 1800
KILL_GRACE = 10


# ---------------------------------------------------------------- helpers

def roadie_cmd():
    return os.environ.get("ROADIE_BATCH_ROADIE") or os.path.join(REPO, "roadie")


def rclone_cmd():
    return os.environ.get("ROADIE_BATCH_RCLONE") or "rclone"


def disk_free(path):
    return shutil.disk_usage(path).free


def probe_duration(path):
    """Seconds of audio via ffprobe; 0.0 when it cannot be read."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True,
                           timeout=60, check=False)
        return float(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def safe_rmtree(path, work_root):
    """Delete a file or tree, but only strictly inside work_root. The ONLY delete in this module."""
    root = os.path.realpath(work_root)
    if len(root.split(os.sep)) < 3:
        raise RuntimeError("refusing: work root %r is too shallow" % root)
    name = os.path.basename(os.path.normpath(path))
    if name in ("", ".", ".."):
        raise RuntimeError("refusing to delete %r" % path)
    if os.path.islink(path):
        real = os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(path))), name)
    else:
        real = os.path.realpath(path)
    if os.path.dirname(real) != root and not real.startswith(root + os.sep):
        raise RuntimeError("refusing to delete %r: not inside work root %r" % (path, root))
    if os.path.islink(real) or os.path.isfile(real):
        os.remove(real)
    elif os.path.isdir(real):
        shutil.rmtree(real)


# ---------------------------------------------------------------- plan

_CAMELOT = re.compile(r"^\s*\d{1,2}[AaBb](?:\s+or\s+\d{1,2}[AaBb])*\s*-\s*(?:\d{1,3}\s*-\s*)*")
_YTID = re.compile(r"\[[A-Za-z0-9_-]{11}\]")
_BPM = re.compile(r"(?<![A-Za-z0-9])\d{2,3}\s?bpm(?![A-Za-z])", re.I)
_MUSKEY = re.compile(r"(?<![A-Za-z0-9])(?!Am(?![A-Za-z0-9]))[A-G][#b]?(?:Maj|Min|maj|min|m)(?![A-Za-z0-9])")


def norm_key(name):
    """Normalised artist-less title used for duplicate detection (name minus extension)."""
    s = _CAMELOT.sub("", name)
    s = _YTID.sub(" ", s)
    s = _BPM.sub(" ", s)
    s = _MUSKEY.sub(" ", s)
    if " - " in s:
        s = s.split(" - ", 1)[1]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = "".join(c for c in s if c.isalnum() or c == " ")
    s = " ".join(s.split())
    return s or " ".join(name.lower().split())


def related(a, b):
    """Equal keys, or one ends with the other and the shorter has >= 2 words."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short.split()) >= 2 and long_.endswith(" " + short)


def make_id(tree, rel_dir, name):
    full = "/".join(p for p in (tree, rel_dir, name) if p)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", full).strip("-")[:60]
    return "%s-%s" % (slug, hashlib.sha1(full.encode("utf-8")).hexdigest()[:6])


def make_dest(base, tree, rel_dir, name):
    return "/".join([base.rstrip("/")] + [p for p in (tree, rel_dir.replace(os.sep, "/"), name) if p])


Cand = namedtuple("Cand", "path tree tree_idx rel_dir name ext key processed")


def collect(roots):
    """Walk the roots. Returns (candidates, stems_dirs, excluded[(path, reason)], n_audio)."""
    cands, stems, excluded = [], [], []
    n_audio = 0
    for ti, root in enumerate(roots):
        root = os.path.abspath(root)
        tree = os.path.basename(root)
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            rel = "" if rel == "." else rel
            sdirs = sorted(d for d in dirnames if d.endswith("_stems"))
            good_sdirs = []
            for d in sdirs:
                full = os.path.join(dirpath, d)
                if "(conflicted)" in d:
                    excluded.append((full, "pCloud conflict copy"))
                elif "\t" in full or "\n" in full:
                    excluded.append((full, "unsafe path (tab or newline)"))
                else:
                    good_sdirs.append(d)
                    stems.append((tree, rel, full))
            dirnames[:] = sorted(d for d in dirnames if not d.endswith("_stems")
                                 and not d.endswith(" Project")
                                 and d.lower() not in ("samples", "backup"))
            audio = sorted(f for f in filenames if not f.startswith(".")
                           and os.path.splitext(f)[1].lower().lstrip(".") in AUDIO_EXTS)
            if not audio:
                continue
            n_aif = sum(os.path.splitext(f)[1].lower().lstrip(".") in AIF_EXTS for f in audio)
            # A folder named for stems needs only 2 files (the Pastor FaderPro pack
            # has 4); the aif-majority rule keeps its 5-file floor. \bstems?\b, not
            # "stem", so a folder called "Systems" is not a pack.
            stem_named = re.search(r"\bstems?\b|remix.?pack", tree + "/" + rel, re.I)
            pack = ((len(audio) >= 2 and stem_named)
                    or (len(audio) >= 5 and n_aif * 2 >= len(audio)))
            skeys = {d: norm_key(d[:-len("_stems")]) for d in good_sdirs}
            for f in audio:
                n_audio += 1
                path = os.path.join(dirpath, f)
                name, ext = os.path.splitext(f)
                ext = ext.lower().lstrip(".")
                if "\t" in path or "\n" in path:
                    excluded.append((path, "unsafe path (tab or newline)"))
                elif "(conflicted)" in f:
                    excluded.append((path, "pCloud conflict copy"))
                elif pack:
                    excluded.append((path, "stem pack"))
                else:
                    key = norm_key(name)
                    processed = next((os.path.join(dirpath, d) for d in good_sdirs
                                      if d == name + "_stems" or skeys[d] == key), None)
                    cands.append(Cand(path, tree, ti, rel, name, ext, key, processed))
    return cands, stems, excluded, n_audio


def dedupe(cands):
    """Returns (keepers, excluded[(path, reason)])."""
    ordered = sorted(cands, key=lambda c: (FORMAT_RANK[c.ext], c.tree_idx, len(c.path), c.path))
    groups = []
    for c in ordered:
        for g in groups:
            if related(g[0].key, c.key):
                g.append(c)
                break
        else:
            groups.append([c])
    keepers, excluded = [], []
    for g in groups:
        done = next((c.processed for c in g if c.processed), None)
        if done:
            excluded += [(c.path, "already processed: %s" % done) for c in g]
        else:
            keepers.append(g[0])
            excluded += [(c.path, "duplicate of %s" % g[0].path) for c in g[1:]]
    return keepers, excluded


def tsv_write(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    header = lines[0].split("\t")
    if header != COLUMNS:
        raise SystemExit("batch: %s is not a manifest (header %r)" % (path, header))
    return [dict(zip(COLUMNS, ln.split("\t"))) for ln in lines[1:] if ln]


def cmd_plan(args):
    for r in args.roots:
        if not os.path.isdir(r):
            print("batch: not a directory: %s" % r, file=sys.stderr)
            return 2
    cands, stems, excluded, n_audio = collect(args.roots)
    keepers, dup_excluded = dedupe(cands)
    excluded += dup_excluded
    trees = [os.path.basename(os.path.abspath(r)) for r in args.roots]

    rows = []
    for t, rel, full in stems:
        st = os.stat(full)
        name = os.path.basename(full)
        rows.append((0, trees.index(t), full, [make_id(t, rel, name), "topup", t, rel, full, name,
                                               0, 0, st.st_mtime_ns, ""]))
    for c in keepers:
        st = os.stat(c.path)
        rows.append((1, c.tree_idx, c.path, [
            make_id(c.tree, c.rel_dir, c.name), "new", c.tree, c.rel_dir, c.path, c.name,
            "%.1f" % probe_duration(c.path), st.st_size, st.st_mtime_ns,
            make_dest(args.dest, c.tree, c.rel_dir, c.name)]))
    rows.sort(key=lambda r: r[:3])
    tsv_write(args.out, COLUMNS, [r[3] for r in rows])
    ex_path = os.path.splitext(args.out)[0] + ".excluded.tsv"
    tsv_write(ex_path, ["path", "reason"], sorted(excluded))

    reasons = {}
    for _, reason in excluded:
        k = "duplicate" if reason.startswith("duplicate of") else reason.split(":")[0]
        reasons[k] = reasons.get(k, 0) + 1
    print("candidates: %d audio files" % n_audio)
    print("new: %d   topup: %d   excluded: %d" % (len(keepers), len(stems), n_audio - len(keepers)))
    for k in sorted(reasons):
        print("  excluded %-24s %d" % (k, reasons[k]))
    print("manifest: %s\nexcluded: %s" % (args.out, ex_path))
    return 0


# ---------------------------------------------------------------- run

class Interrupted(Exception):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


class StepFailed(Exception):
    pass


DONE_MARKER = ".roadie-batch-ok"   # on pCloud, written after a passing rclone check
READY_SUFFIX = ".ready"            # beside a work dir whose analysis finished
EXIT_FLOOR = 4


class UploadFailed(Exception):
    pass


class StopFloor(Exception):
    pass


Result = namedtuple("Result", "rc timed_out out")


def kill_group(proc):
    """SIGTERM the child's whole process group, SIGKILL whatever is left after KILL_GRACE."""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + KILL_GRACE
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=KILL_GRACE)
    except subprocess.TimeoutExpired:
        pass


def run_cmd(argv, timeout, log, capture=False, env=None):
    """Run argv in its own session, output to the log file; kill the whole group on timeout."""
    with open(log, "ab") as lf:
        lf.write(("\n$ %s\n" % shlex.join(argv)).encode("utf-8"))
        lf.flush()
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE if capture else lf,
                                    stderr=lf, start_new_session=True, env=env)
        except OSError as e:
            lf.write(("cannot run: %s\n" % e).encode("utf-8"))
            return Result(127, False, b"")
        try:
            try:
                out, _ = proc.communicate(timeout=timeout)
                return Result(proc.returncode, False, out or b"")
            except subprocess.TimeoutExpired:
                kill_group(proc)
                out, _ = proc.communicate()
                lf.write(("timed out after %ss, process group killed\n" % timeout).encode("utf-8"))
                return Result(-1, True, out or b"")
        except BaseException:
            kill_group(proc)
            raise


def read_state(path):
    """Last status per id from the JSONL state file."""
    last = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    ev = json.loads(ln)
                    last[ev["id"]] = ev["status"]
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return last


class Runner:
    def __init__(self, rows, a):
        self.rows = rows
        self.work_root = os.path.abspath(a.work)
        self.state_path = os.path.abspath(a.state)
        self.logs = os.path.abspath(a.logs)
        self.floor = a.floor_gb * 1024 ** 3
        self.limit = a.limit
        self.start = datetime.datetime.now()
        self.results = []      # dicts: name, id, status, secs, detail, log
        self.attempted = set()
        self.min_free = None
        self.current = None

    def log_path(self, row):
        return os.path.join(self.logs, row["id"] + ".log")

    def record(self, row, status, secs, error="", step="", note=""):
        ev = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "id": row["id"],
              "name": row["name"], "mode": row["mode"], "status": status, "secs": round(secs, 1),
              "step": step, "error": error, "note": note, "dest": row["dest"]}
        with open(self.state_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
        if status in ("done", "failed"):     # the others still need a re-run: they stay "remaining"
            self.attempted.add(row["id"])
        self.results.append({"name": row["name"], "id": row["id"], "status": status,
                             "secs": round(secs, 1), "detail": error or note or row["dest"],
                             "log": self.log_path(row)})
        print("%-16s %s (%.0fs)%s" % (status, row["name"], secs, " " + error if error else ""),
              flush=True)

    def step(self, row, name, argv, timeout, env=None):
        r = run_cmd(argv, timeout, self.log_path(row), env=env)
        if r.timed_out:
            raise StepFailed("%s: timeout after %ds" % (name, timeout))
        if r.rc != 0:
            raise StepFailed("%s: exit %d" % (name, r.rc))

    def note_free(self):
        try:
            free = disk_free(self.work_root)
        except OSError:
            return
        self.min_free = free if self.min_free is None else min(self.min_free, free)

    # -- row handlers

    def do_topup(self, row):
        stems = row["src"]
        if not os.path.isdir(stems):
            raise StepFailed("stems dir missing")
        base = os.path.basename(stems)[:-len("_stems")]
        als = os.path.join(os.path.dirname(stems), base + ".als")
        did = []
        if not os.path.exists(os.path.join(stems, "sections.json")):
            self.step(row, "sections", [roadie_cmd(), "sections", stems], POST_TIMEOUT)
            did.append("sections")
        if not os.path.exists(als):
            self.step(row, "als", [roadie_cmd(), "als", stems], POST_TIMEOUT)
            did.append("als")
        return "topped up: " + (", ".join(did) if did else "nothing missing")

    def do_new(self, row):
        # Done on pCloud means the marker is there: it is written only after `rclone
        # check` passed, so a half-finished upload is never mistaken for a finished one.
        dest = row["dest"]
        r = run_cmd([rclone_cmd(), "lsf", dest + "/"], LSF_TIMEOUT, self.log_path(row), capture=True)
        if r.rc == 0 and DONE_MARKER in r.out.decode("utf-8", "replace").splitlines():
            return "already on pCloud"
        os.makedirs(self.work_root, exist_ok=True)
        work = os.path.join(self.work_root, row["id"])
        ready = work + READY_SUFFIX          # local analysis finished; only the upload is left
        if not (os.path.isfile(ready) and os.path.isdir(work)):
            if disk_free(self.work_root) < self.floor:
                raise StopFloor()
            for p in (work, ready):
                if os.path.lexists(p):
                    safe_rmtree(p, self.work_root)
            os.makedirs(work)
            tmp = os.path.join(work, "tmp")  # demucs temp inside the work dir, so cleanup reaches it
            os.makedirs(tmp)
            try:
                dur = float(row["duration_sec"] or 0)
                tmo = int(SPLIT_FACTOR * dur + SPLIT_PAD) if dur > 0 else SPLIT_NODUR
                self.step(row, "split", [roadie_cmd(), "split", "--file", row["src"], "-d", "-o", work],
                          tmo, env=dict(os.environ, TMPDIR=tmp))
                safe_rmtree(tmp, self.work_root)
                stems = [d for d in os.listdir(work)
                         if d.endswith("_stems") and os.path.isdir(os.path.join(work, d))]
                if len(stems) != 1:
                    raise StepFailed("split: expected one *_stems dir, found %d" % len(stems))
                stems = os.path.join(work, stems[0])
                self.step(row, "sections", [roadie_cmd(), "sections", stems], POST_TIMEOUT)
                self.step(row, "als", [roadie_cmd(), "als", stems], POST_TIMEOUT)
                for f in sorted(os.listdir(work)):
                    p = os.path.join(work, f)
                    if f.lower().endswith(".wav") and os.path.isfile(p) and not os.path.islink(p):
                        safe_rmtree(p, self.work_root)
            except StepFailed:
                safe_rmtree(work, self.work_root)
                raise
            with open(ready, "w") as f:
                f.write(row["src"] + "\n")
        try:
            self.step(row, "upload", [rclone_cmd(), "copy", work, dest, "--transfers", "4",
                                      "--create-empty-src-dirs"], UPLOAD_TIMEOUT)
            self.step(row, "verify", [rclone_cmd(), "check", work, dest, "--one-way"], CHECK_TIMEOUT)
            self.step(row, "mark", [rclone_cmd(), "touch", dest + "/" + DONE_MARKER], LSF_TIMEOUT)
        except StepFailed as e:
            raise UploadFailed(str(e))
        safe_rmtree(work, self.work_root)
        safe_rmtree(ready, self.work_root)
        return dest

    # -- main loop

    def loop(self):
        last = read_state(self.state_path)
        n = 0
        for row in self.rows:
            if last.get(row["id"]) == "done":
                print("skip (done)      %s" % row["name"], flush=True)
                continue
            if self.limit is not None and n >= self.limit:
                break
            n += 1
            self.current = row
            t0 = time.monotonic()
            try:
                note = self.do_topup(row) if row["mode"] == "topup" else self.do_new(row)
                self.record(row, "done", time.monotonic() - t0, note=note if note != row["dest"] else "")
            except StepFailed as e:
                self.record(row, "failed", time.monotonic() - t0, error=str(e), step=str(e).split(":")[0])
            except UploadFailed as e:
                self.record(row, "upload_failed", time.monotonic() - t0, error=str(e), step="upload")
                self.note_free()
                print("upload failed; work dir kept; stopping", file=sys.stderr)
                return 3
            except StopFloor:
                self.record(row, "stopped_floor", time.monotonic() - t0,
                            error="free disk below %.1f GB" % (self.floor / 1024 ** 3))
                self.note_free()
                return EXIT_FLOOR
            except Interrupted:
                raise
            except Exception as e:  # a bug must not end the night
                self.record(row, "failed", time.monotonic() - t0, error="internal error: %r" % (e,))
            self.current = None
            self.note_free()
        return 0

    def interrupted(self, signum):
        if self.current is not None:
            self.record(self.current, "interrupted", 0.0, error="signal %d" % signum)
            self.current = None

    def write_report(self, report_dir, rc):
        end = datetime.datetime.now()
        done = {i for i, s in read_state(self.state_path).items() if s == "done"}
        remaining = [r for r in self.rows if r["id"] not in done and r["id"] not in self.attempted]
        counts = {}
        for r in self.results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        out = ["# Roadie batch report %s" % end.date().isoformat(), "",
               "- start: %s" % self.start.isoformat(timespec="seconds"),
               "- end: %s" % end.isoformat(timespec="seconds"),
               "- exit code: %d" % rc,
               "- counts: %s" % (", ".join("%s %d" % kv for kv in sorted(counts.items())) or "none"),
               "- min free disk: %s" % ("%.1f GB" % (self.min_free / 1024 ** 3)
                                        if self.min_free is not None else "n/a"),
               "- remaining rows: %d" % len(remaining), "",
               "| name | status | secs | dest / error | log |", "|---|---|---|---|---|"]
        out += ["| %s | %s | %s | %s | %s |" % (esc(r["name"]), r["status"], r["secs"],
                                                esc(r["detail"]), esc(r["log"])) for r in self.results]
        if remaining:
            out += ["", "## Remaining", ""] + ["- %s (%s)" % (r["name"], r["mode"]) for r in remaining]
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, "batch-report-%s.md" % end.date().isoformat())
        sep = "\n\n---\n\n" if os.path.exists(path) else ""
        with open(path, "a", encoding="utf-8") as f:
            f.write(sep + "\n".join(out) + "\n")
        print("report: %s" % path)
        return path


def dry_run(rows, a):
    last = read_state(os.path.abspath(a.state))
    n = 0
    for row in rows:
        if last.get(row["id"]) == "done":
            print("skip (done)  %s" % row["name"])
            continue
        if a.limit is not None and n >= a.limit:
            print("beyond --limit  %s" % row["name"])
            continue
        n += 1
        if row["mode"] == "topup":
            print("topup  %s: sections (if sections.json missing), als (if .als missing, never -f)"
                  % row["src"])
        else:
            print("new    %s -> %s: skip if stems+.als already there; else split -d, sections, als, "
                  "rm 24-bit wav, rclone copy + check, rm work/%s" % (row["src"], row["dest"], row["id"]))
    return 0


def cmd_run(a):
    rows = read_manifest(a.manifest)
    if a.dest:
        for r in rows:
            if r["mode"] == "new":
                r["dest"] = make_dest(a.dest, r["tree"], r["rel_dir"], r["name"])
    rows = [r for r in rows if not a.only_mode or r["mode"] == a.only_mode]
    rows.sort(key=lambda r: r["mode"] != "topup")   # stable: topup first, manifest order kept
    if a.dry_run:
        return dry_run(rows, a)

    work_root = os.path.abspath(a.work)
    os.makedirs(work_root, exist_ok=True)
    os.makedirs(os.path.abspath(a.logs), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.state)), exist_ok=True)
    lock = open(os.path.join(os.path.dirname(work_root), "run.pid"), "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("batch: another run holds %s" % lock.name, file=sys.stderr)
        return 2
    lock.seek(0)
    lock.truncate()
    lock.write("%d\n" % os.getpid())
    lock.flush()

    sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def on_signal(signum, _frame):
        for s in sigs:                       # a second signal must not cut short kill_group
            signal.signal(s, signal.SIG_IGN)
        raise Interrupted(signum)

    old = {s: signal.signal(s, on_signal) for s in sigs}
    runner = Runner(rows, a)
    rc = 1
    try:
        try:
            rc = runner.loop()
        except Interrupted as e:
            for s in old:
                signal.signal(s, signal.SIG_IGN)
            runner.interrupted(e.signum)
            rc = 128 + e.signum
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        runner.note_free()
        runner.write_report(os.path.abspath(a.report_dir), rc)
        lock.close()
    return rc


# ---------------------------------------------------------------- verify-sources

def cmd_verify(a):
    changed = 0
    for r in read_manifest(a.manifest):
        if r["mode"] != "new":
            continue
        try:
            st = os.stat(r["src"])
        except OSError:
            print("MISSING  %s" % r["src"])
            changed += 1
            continue
        if str(st.st_size) != r["size"] or str(st.st_mtime_ns) != r["mtime"]:
            print("CHANGED  %s" % r["src"])
            changed += 1
    print("%d source(s) changed" % changed)
    return 1 if changed else 0


# ---------------------------------------------------------------- cli

def main(argv=None):
    p = argparse.ArgumentParser(prog="roadie batch", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="inventory ROOTs into a manifest")
    pp.add_argument("roots", nargs="+", metavar="ROOT")
    pp.add_argument("--out", required=True, metavar="MANIFEST.tsv")
    pp.add_argument("--dest", default=DEFAULT_DEST)

    pr = sub.add_parser("run", help="process a manifest, one row at a time")
    pr.add_argument("manifest")
    pr.add_argument("--dest", default=None,
                    help="override the manifest's destination base (default: use the manifest)")
    pr.add_argument("--work", default=os.path.join(CACHE, "work"))
    pr.add_argument("--state", default=os.path.join(CACHE, "state.jsonl"))
    pr.add_argument("--logs", default=os.path.join(CACHE, "logs"))
    pr.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    pr.add_argument("--floor-gb", type=float, default=4)
    pr.add_argument("--limit", type=int, default=None, help="stop after N rows (skipped ones do not count)")
    pr.add_argument("--only-mode", choices=("new", "topup"))
    pr.add_argument("--dry-run", action="store_true")

    pv = sub.add_parser("verify-sources", help="exit 1 if any source changed since the plan")
    pv.add_argument("manifest")

    a = p.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "verify-sources": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

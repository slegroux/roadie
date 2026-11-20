#!/usr/bin/env python3
"""Read one study (a track's stems and the analyses saved beside them) as a single record.

Read-only: nothing here measures anything or writes to the study. Each part of the record
(a `Layer`) names the file it came from, or, when absent, the command that would make it.
Tools that compare studies call `load_study` rather than opening `sections.json` themselves.

A study is a folder holding a `*_stems/` directory (a yt2stems result), found under a Studies
root as <Artist>/<Title>/. There can be several roots (ROADIE_STUDIES, a list separated by
os.pathsep, each entry optionally `label=path`); a root that is not there is skipped, never an
error. Discovery only lists directories and reads small JSON files: it never opens audio, and the
roots may be a slow FUSE mount.
"""

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

DEFAULT_ROOTS = "local=~/Projects/Music/slgMusProd/Studies%spcloud=~/pCloud Drive/Music/Production/Roadie/Studies" % os.pathsep


class Root(NamedTuple):
    label: str          # the prefix that picks this root: `label:Title`
    path: Path


def parse_roots(spec: str | None) -> list[Root]:
    """`label=path` entries separated by os.pathsep; an entry without a label is `root<N>`."""
    roots = []
    for i, entry in enumerate(e for e in (spec or DEFAULT_ROOTS).split(os.pathsep) if e.strip()):
        label, sep, path = entry.partition("=")
        if not sep or "/" in label or not label.strip():
            label, path = "root%d" % (i + 1), entry
        roots.append(Root(label.strip(), Path(path.strip()).expanduser()))
    return roots


STUDIES_ROOTS = parse_roots(os.environ.get("ROADIE_STUDIES"))


def _norm(s: str) -> str:
    """Names compare equal across Unicode forms: pCloud keeps NFD file names, a typed name is NFC."""
    return unicodedata.normalize("NFC", s).casefold()


def _roots(roots) -> list[Root]:
    """The roots to search: None means STUDIES_ROOTS (read at call time); a bare path is one root."""
    if roots is None:
        return list(STUDIES_ROOTS)
    if isinstance(roots, (str, Path)):
        return [Root("root1", Path(roots))]
    return list(roots)

BEATS_PER_BAR = 4
PROFILE_KIND, PROFILE_VERSION = "reference_profile", 6    # the profile this reader knows (roadie_analysis/profile.py)

# Stem files that make up each role, in preference order within a role. The names are the
# ones sections.json uses for the roles playing in a section.
STEM_ROLES = {
    "kick": ("drums_kick",),
    "bass": ("bass",),
    "hats": ("drums_hh", "drums_ride"),
    "snare": ("drums_snare",),
    "perc": ("drums_toms", "drums_crash"),
    "synths": ("synths",),
    "vocals": ("vocals",),
}

# What to run for a layer that is not on disk.
MISSING = {
    "grid": "roadie load <stems> (writes alignment.json)",
    "sections": "roadie sections <stems>",
    "activity": "bar_activity: re-run the sections command on the stems folder (a sections.json made before it has none)",
    "mix": "reference_profile (roadie MCP tool) on the mix and stems, per section",
    "drums": "not made yet: drum roles per stem (studies/04 item 4)",
}
LAYER_NAMES = ("grid", "sections", "activity", "mix", "drums")


@dataclass
class Layer:
    """One analysis result, as found on disk."""
    name: str
    source: Path | None = None
    made_by: str | None = None
    data: dict | None = None
    missing: str | None = None
    caveats: list[str] = field(default_factory=list)


@dataclass
class Section:
    label: str
    bar: int            # file bar, 1-based
    bars: int
    live_bar: float | None
    roles: list[str]


@dataclass
class Study:
    artist: str
    title: str
    root: Path
    bpm: float | None                    # None when no layer knows it
    beats_per_bar: int
    bar_offset: int | None               # Live bar = file bar + bar_offset; None when unknown
    stems: dict[str, list[Path]]         # role -> files
    layers: dict[str, Layer]
    extras: list[Path] = field(default_factory=list)
    source: str = ""                      # label of the root it was found under

    def sections(self) -> list[Section]:
        data = self.layers["sections"].data
        if not data:
            return []
        return [Section(s["label"], s["bar"], s["duration_beats"] // self.beats_per_bar,
                        s.get("live_bar"), s.get("stems", [])) for s in data["sections"]]

    def activity(self) -> dict[str, list[bool]] | None:
        """Which bars each role plays, from file bar `activity_first_bar()`: role -> one flag per
        bar, a role playing when any of its stems does. None when sections.json has no bar_activity."""
        data = self.layers["activity"].data
        if not data:
            return None
        out = {}
        for role, names in STEM_ROLES.items():
            flags = [[c == "1" for c in data["stems"][n]] for n in names if n in data["stems"]]
            if flags:
                out[role] = [any(bar) for bar in zip(*flags)]
        return out

    def activity_first_bar(self) -> int | None:
        data = self.layers["activity"].data
        return data["first_bar"] if data else None

    def mix(self) -> list[dict] | None:
        data = self.layers["mix"].data
        return data["sections"] if data else None


def _number(x, default=None):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else default


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _marker(folder: Path) -> dict | None:
    m = _read_json(folder / "study.json")
    return m if isinstance(m, dict) and m.get("kind") == "study_marker" else None


def _stems_dir(folder: Path) -> Path | None:
    """The folder named by the study.json marker, else the first `*_stems/` folder."""
    named = (_marker(folder) or {}).get("stems")
    if isinstance(named, str) and named and "/" not in named and named != "..":
        if (folder / named).is_dir():
            return folder / named
    found = sorted(p for p in folder.glob("*_stems") if p.is_dir())
    return found[0] if found else None


@dataclass
class StudyRef:
    artist: str
    title: str
    root: Path              # the study folder
    analysed: bool          # it has a stems folder
    source: str = ""        # label of the root it is under
    stems_dir: Path | None = None


def _scan(roots) -> tuple[list[StudyRef], dict[str, list[str]], set[str]]:
    """(studies, errors by root label, labels of roots that could not be listed at all). A folder that
    cannot be listed is recorded and skipped, so one bad root or artist never breaks lookups elsewhere."""
    refs, errors, down = [], {}, set()
    for r in _roots(roots):
        errors[r.label] = []
        if not r.path.is_dir():
            continue
        try:
            artists = sorted(p for p in r.path.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError as e:
            errors[r.label].append("%s: %s" % (r.path, e.strerror or type(e).__name__))
            down.add(r.label)
            continue
        for artist in artists:
            try:
                titles = sorted(p for p in artist.iterdir() if p.is_dir() and not p.name.startswith("."))
                for title in titles:
                    stems = _stems_dir(title)
                    refs.append(StudyRef(artist.name, title.name, title, stems is not None, r.label, stems))
            except OSError as e:
                errors[r.label].append("%s: %s" % (artist, e.strerror or type(e).__name__))
    return refs, errors, down


def find_studies(roots=None) -> list[StudyRef]:
    """Every <Artist>/<Title>/ folder under each root, analysed or not; hidden folders are skipped.
    A root that is not there, or a folder that cannot be listed, contributes nothing."""
    return _scan(roots)[0]


def _resolve(name_or_path: str, roots: list[Root]) -> tuple[Path, str]:
    """(study folder, root label) for an existing path, or a title among the analysed studies.
    `label:Title` limits the search to that root."""
    if not name_or_path.strip():
        raise ValueError("no study name given")
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        if p.name.endswith("_stems"):
            p = p.parent
        if _stems_dir(p):
            real = p.resolve()
            for r in roots:
                try:
                    real.relative_to(r.path.resolve())
                    return p, r.label
                except (ValueError, OSError):
                    pass
            return p, "path"
    name = name_or_path.strip()
    label, _, rest = name.partition(":")
    if rest.strip() and _norm(label) in {_norm(r.label) for r in roots}:
        roots = [r for r in roots if _norm(r.label) == _norm(label)]
        name = rest.strip()
    want = _norm(name)
    if not want:
        raise ValueError("no study name given")
    refs = [r for r in find_studies(roots) if r.analysed]
    hits = [r for r in refs if want in (_norm(r.title), _norm("%s/%s" % (r.artist, r.title)))]
    if not hits:      # no exact title: a title that contains the name, if that is one study or several
        hits = [r for r in refs if want in _norm(r.title)]
    if not hits:
        raise ValueError("no study named %r under %s" % (name_or_path, ", ".join(
            "%s=%s%s" % (r.label, r.path, "" if r.path.is_dir() else " (not there)") for r in roots)))
    if len(hits) > 1:
        raise ValueError("%r is ambiguous, one of: %s (give the root to pick one, as %s:%s)" % (
            name_or_path, ", ".join("%s:%s/%s" % (h.source, h.artist, h.title) for h in hits),
            hits[0].source, name))
    return hits[0].root, hits[0].source


def _json_files(folder: Path) -> list[Path]:
    out = []
    for dirpath, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [Path(dirpath) / f for f in sorted(files) if f.endswith(".json") and not f.startswith(".")]
    return out


def _check(folder: Path, prof: dict) -> tuple[str, str]:
    """(state, one line) for a profile's inputs against the disk. State is fresh, stale,
    ambiguous or unchecked; a wrong shape raises KeyError/TypeError/AttributeError."""
    by_name = {}
    for dirpath, _dirs, files in os.walk(folder):
        for f in files:
            by_name.setdefault(f, []).append(os.path.join(dirpath, f))
    inputs = list(prof["inputs"]["mix"]) + [i for items in prof["inputs"]["stems"].values() for i in items]
    if not inputs:
        return "unchecked", "not checked: the profile records no inputs"
    changed, ambiguous, by_path, by_size = [], [], 0, 0
    here = Path(folder).resolve()

    def inside(p):          # a recorded path is trusted only when it is a file of this study
        try:
            return os.path.exists(p) and Path(p).resolve().is_relative_to(here)
        except (OSError, ValueError):
            return False

    for item in inputs:
        name = os.path.basename(item["path"])
        what = []
        if inside(item["path"]):
            by_path += 1
            st = os.stat(item["path"])
            what = [w for w, bad in (("size", st.st_size != item["size"]),
                                     ("mtime", abs(st.st_mtime - item["mtime"]) > 2.0)) if bad]
        elif len(by_name.get(name, [])) > 1:
            ambiguous.append(name)
        elif name in by_name:
            by_size += 1
            what = ["size"] if os.path.getsize(by_name[name][0]) != item["size"] else []
        else:
            what = ["missing"]
        if what:
            changed.append("%s (%s)" % (name, ", ".join(what)))
    check = ("path, mtime and size" if by_path and not by_size else
             "file name and size, the recorded paths are not files of this study" if by_size and not by_path else
             "path, mtime and size where the recorded path is in this study, else file name and size")
    amb = "; several files share the name of: %s" % ", ".join(sorted(set(ambiguous))) if ambiguous else ""
    if changed:
        return "stale", "stale: %s changed since the profile was made (checked by %s)%s" % (
            "; ".join(changed), check, amb)
    if ambiguous:
        return "ambiguous", "ambiguous: could not tell if inputs changed%s (checked by %s)" % (amb, check)
    return "fresh", "fresh: %d inputs unchanged (checked by %s)" % (len(inputs), check)


def _mix_layer(folder: Path, path: Path, prof: dict) -> tuple[Layer, str]:
    """The mix layer for one profile, and its state (fresh, stale, ambiguous, unchecked, newer, unreadable)."""
    version = prof.get("version")
    made_by = "reference_profile v%s" % version
    if not isinstance(version, int) or version > PROFILE_VERSION:
        return Layer("mix", path, made_by, None, None, [
            "profile version %s is newer than this reader knows (%d): not interpreted"
            % (version, PROFILE_VERSION)]), "newer"
    caveats = []
    if version < PROFILE_VERSION:
        caveats.append("profile version %d is older than %d: rebuild it before comparing"
                       % (version, PROFILE_VERSION))
    try:
        state, line = _check(folder, prof)
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as e:
        return Layer("mix", path, made_by, None, None, caveats + [
            "could not read: the profile's inputs are malformed (%s: %s)" % (type(e).__name__, e)]), "unreadable"
    return Layer("mix", path, made_by, prof, None, caveats + [line]), state


def _activity_layer(path: Path, act) -> Layer:
    """The activity layer from sections.json's `bar_activity`, or an unreadable layer when it is not
    {"first_bar": int, "bars": n, "stems": {stem: n characters of 0/1}}."""
    try:
        ok = (isinstance(act["first_bar"], int) and isinstance(act["bars"], int)
              and isinstance(act["stems"], dict) and act["stems"]
              and all(isinstance(v, str) and len(v) == act["bars"] and set(v) <= {"0", "1"}
                      for v in act["stems"].values()))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        return Layer("activity", path, None, None, None, ["bar_activity exists but could not be read"])
    mapped = {n for names in STEM_ROLES.values() for n in names}
    caveats = ["playing = the stem's level within 30 dB of its own loud level, per bar: a reading "
               "of the audio, not of notes"]
    other = sorted(n for n in act["stems"] if n not in mapped)
    if other:
        caveats.append("stems that are in no role, so not in activity(): %s" % ", ".join(other))
    return Layer("activity", path, "sections (stems method, bar_activity)", act, None, caveats)


def load_study(name_or_path: str, roots=None) -> Study:
    folder, source = _resolve(name_or_path, _roots(roots))
    stems_dir = _stems_dir(folder)
    files = sorted(f for f in stems_dir.iterdir() if f.suffix in (".flac", ".wav"))
    stems = {role: [f for n in names for f in files if f.stem == n]
             for role, names in STEM_ROLES.items()}
    stems = {role: fs for role, fs in stems.items() if fs}

    layers = {n: Layer(n, missing=MISSING[n]) for n in LAYER_NAMES}
    bpm, bar_offset = _number((_marker(folder) or {}).get("bpm")), None

    def unreadable(name, path):
        # The file is there: never offer the command that would overwrite it.
        return Layer(name, path, None, None, None, ["file exists but could not be read"])

    path = stems_dir / "alignment.json"
    if path.exists():
        align = _read_json(path)
        res = align.get("result") if isinstance(align, dict) else None
        if isinstance(res, dict):
            bpm = _number(res.get("bpm"), bpm)
            caveats = []
            if res.get("downbeat_verdict") == "chance-level":
                caveats.append("downbeat confidence is chance-level (%s)" % res.get("downbeat_confidence"))
            layers["grid"] = Layer("grid", path, "beat_aligner (%s)" % res.get("detector", "?"),
                                   res, None, caveats)
        else:
            layers["grid"] = unreadable("grid", path)

    layers["activity"].missing = ("roadie sections <stems> first: it makes sections.json, and "
                                  "bar_activity in it when the stems have a kick and a bass")
    path = stems_dir / "sections.json"
    if path.exists():
        layers["activity"].missing = MISSING["activity"]
        secs = _read_json(path)
        if isinstance(secs, dict) and isinstance(secs.get("sections"), list):
            bpm = _number(secs.get("bpm")) if bpm is None else bpm
            caveats = ["labels by rule, not checked by ear"]
            offsets = {int(round(x["live_bar"] - x["bar"])) for x in secs["sections"]
                       if isinstance(x.get("live_bar"), (int, float)) and isinstance(x.get("bar"), (int, float))}
            if len(offsets) == 1:
                bar_offset = offsets.pop()
            elif offsets:
                caveats.append("live_bar offsets disagree between sections: Live bar unknown")
            else:
                caveats.append("no live_bar: no downbeat was known when it was made")
            layers["sections"] = Layer("sections", path,
                                       "sections (%s method)" % secs.get("method", "spectral"),
                                       secs, None, caveats)
            act = secs.get("bar_activity")
            if act is not None:
                layers["activity"] = _activity_layer(path, act)
            elif secs.get("method", "spectral") != "stems":
                layers["activity"].missing = (
                    "none, and none possible: these sections came from the spectral method (the "
                    "stems have no usable kick and bass), which makes no per-bar stem activity")
        else:
            layers["sections"] = unreadable("sections", path)
            layers["activity"].missing = "sections.json exists but could not be read"

    known = {stems_dir / "alignment.json", stems_dir / "sections.json", folder / "study.json"}
    profiles, extras = [], []
    for p in _json_files(folder):
        if p in known:
            continue
        d = _read_json(p) if p.name.endswith(".profile.json") else None
        if isinstance(d, dict) and d.get("kind") == PROFILE_KIND:
            profiles.append((p, d))
        else:
            extras.append(p)
    if profiles:
        # A fresh profile of a known version first, then the newest: a pCloud re-sync touches
        # mtimes, so mtime alone could pick a stale profile.
        cands = []
        for p, d in profiles:
            layer, state = _mix_layer(folder, p, d)
            cands.append((state != "fresh", -p.stat().st_mtime, str(p), layer))
        cands.sort(key=lambda c: c[:3])
        layers["mix"] = cands[0][3]
        rest = [c[3].source for c in cands[1:]]
        if rest:
            layers["mix"].caveats.insert(0, "chosen among %d profiles (a fresh one first, then the newest); "
                                         "also found, listed under extras: %s" % (
                                             len(cands), ", ".join(str(p.relative_to(folder)) for p in rest)))
        extras += rest
    else:
        bad = [p for p in extras if p.name.endswith(".profile.json")]
        if bad:
            layers["mix"] = unreadable("mix", bad[0])
            extras.remove(bad[0])

    return Study(folder.parent.name, folder.name, folder, bpm, BEATS_PER_BAR, bar_offset,
                 stems, layers, sorted(extras), source)


def to_json(study: Study, layers: list[str] | None = None) -> dict:
    """The record as JSON. `layers` keeps only those layers (a name not in LAYER_NAMES is an error)."""
    unknown = [n for n in layers or [] if n not in LAYER_NAMES]
    if unknown:
        raise ValueError("unknown layer %s; the layers are %s" % (", ".join(unknown), ", ".join(LAYER_NAMES)))

    def rel(p):
        return str(Path(p).relative_to(study.root))

    return {
        "kind": "study", "version": 1,
        "root": study.source, "path": str(study.root.absolute()), "artist": study.artist, "title": study.title,
        "bpm": study.bpm, "beats_per_bar": study.beats_per_bar, "bar_offset": study.bar_offset,
        "stems": {r: [rel(f) for f in fs] for r, fs in study.stems.items()},
        "layers": {n: {"source": rel(l.source) if l.source else None, "made_by": l.made_by,
                       "missing": l.missing, "caveats": l.caveats, "data": l.data}
                   for n, l in study.layers.items() if layers is None or n in layers},
        "extras": [rel(p) for p in study.extras],
    }


def _show(study: Study) -> None:
    print("%s - %s  [%s]" % (study.artist, study.title, study.source))
    print("  %s bpm, %d/4, Live bar = %s" % (
        "%g" % study.bpm if study.bpm is not None else "unknown", study.beats_per_bar,
        "file bar + %d" % study.bar_offset if study.bar_offset is not None else "unknown"))
    print("  stems: %s" % ", ".join(study.stems))
    for name, layer in study.layers.items():
        if layer.source is None:
            print("  %-9s missing: %s" % (name, layer.missing))
            continue
        print("  %-9s %s  [%s]" % (name, layer.source.relative_to(study.root), layer.made_by or "unreadable"))
        for c in layer.caveats:
            print("            note: %s" % c)
        if name == "mix" and layer.data:
            print("            %d sections" % len(study.mix()))
        if name == "sections":
            for s in study.sections():
                live = "Live bar %g" % s.live_bar if s.live_bar is not None else "-"
                print("    %-12s file bar %-4d %3d bars  %-12s %s" % (
                    s.label, s.bar, s.bars, live, " ".join(s.roles)))
    if study.extras:
        print("  extras (%d, not interpreted)" % len(study.extras))
        for p in study.extras:
            print("    %s" % p.relative_to(study.root))


def _layer_flags(ref: StudyRef) -> dict[str, bool]:
    """Which layers a study has, by file names (listing, no JSON parsed): the files in the stems
    folder, and `*.profile.json` in the study folder, its analysis/ and the stems folder. The one
    file opened is sections.json (a few KB), searched for the text "bar_activity"."""
    flags = {n: False for n in LAYER_NAMES}
    flags["grid"] = (ref.stems_dir / "alignment.json").exists()
    flags["sections"] = (ref.stems_dir / "sections.json").exists()
    if flags["sections"]:      # the one file opened here, only to look for the key
        try:
            flags["activity"] = '"bar_activity"' in (ref.stems_dir / "sections.json").read_text()
        except (OSError, ValueError):
            pass
    for d in (ref.root, ref.root / "analysis", ref.stems_dir):
        try:
            if any(f.endswith(".profile.json") and not f.startswith(".") for f in os.listdir(d)):
                flags["mix"] = True
        except OSError:
            pass
    return flags


LAYERS_NOTE = ("layer flags come from file names only, except that sections.json is opened to look for "
               "its bar_activity key (the activity layer); `roadie study show <name>` reads the files "
               "and is the check")


def list_to_json(roots=None) -> dict:
    """Every study under the roots: {root, path, artist, title, analysed, layers: {name: has it}}, and
    each root with whether it is there and what could not be listed. Shallow: directory listings
    only, so a slow mount is cheap."""
    roots = _roots(roots)
    refs, errors, down = _scan(roots)
    studies = [{"root": r.source, "path": str(r.root.absolute()), "artist": r.artist, "title": r.title,
                "analysed": r.analysed, "layers": _layer_flags(r) if r.analysed else None} for r in refs]
    return {"kind": "study_list", "version": 1, "note": LAYERS_NOTE,
            "roots": [{"label": r.label, "path": str(r.path),
                       "available": r.path.is_dir() and r.label not in down,
                       "errors": errors[r.label],
                       "analysed": sum(1 for s in studies if s["root"] == r.label and s["analysed"]),
                       "not_analysed": sum(1 for s in studies if s["root"] == r.label and not s["analysed"])}
                      for r in roots],
            "studies": studies}


def _list(as_json: bool) -> int:
    try:
        data = list_to_json()
    except (OSError, ValueError) as e:
        print("study: could not list: %s" % e, file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    for root in data["roots"]:
        rows = [s for s in data["studies"] if s["root"] == root["label"]]
        if not root["available"]:
            why = "; ".join(root["errors"]) or "not mounted or missing"
            print("%s  %s: %s, skipped" % (root["label"], root["path"], why))
            continue
        print("%s  %s" % (root["label"], root["path"]))
        for e in root["errors"]:
            print("  could not list %s" % e)
        print("  analysed (%d)" % root["analysed"])
        for r in (s for s in rows if s["analysed"]):
            have = [n for n, ok in r["layers"].items() if ok]
            print("    %-40s has: %-22s missing: %s" % ("%s/%s" % (r["artist"], r["title"]), " ".join(have) or "-",
                                                      " ".join(n for n in r["layers"] if n not in have) or "-"))
        print("  not analysed (%d)" % root["not_analysed"])
        for r in (s for s in rows if not s["analysed"]):
            print("    %s/%s" % (r["artist"], r["title"]))
    print("note: %s" % data["note"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie study", description="Read a study as one record")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="every study under the studies root")
    ls.add_argument("--json", action="store_true", help="print the list as JSON")
    show = sub.add_parser("show", help="print one study")
    show.add_argument("name", help="a title under the studies root, or a path")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        return _list(args.json)
    try:
        study = load_study(args.name)
        if args.json:
            print(json.dumps(to_json(study), indent=2))
        else:
            _show(study)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
        print("study: %s: %s" % ("could not read" if not isinstance(e, ValueError) else "error", e),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

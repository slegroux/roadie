#!/usr/bin/env python3
"""Read one study (a track's stems and the analyses saved beside them) as a single record.

Read-only: nothing here measures anything or writes to the study. Each part of the record
(a `Layer`) names the file it came from, or, when absent, the command that would make it.
Tools that compare studies call `load_study` rather than opening `sections.json` themselves.

A study is a folder holding a `*_stems/` directory (a yt2stems result), found under the
Studies root as <Artist>/<Title>/.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

STUDIES_ROOT = Path(os.environ.get("ROADIE_STUDIES")
                    or "~/Projects/Music/slgMusProd/Studies").expanduser()

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
    "activity": "not made yet: bar_activity in sections.json (roadie/01 item 3)",
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

    def sections(self) -> list[Section]:
        data = self.layers["sections"].data
        if not data:
            return []
        return [Section(s["label"], s["bar"], s["duration_beats"] // self.beats_per_bar,
                        s.get("live_bar"), s.get("stems", [])) for s in data["sections"]]

    def activity(self) -> dict[str, list[bool]] | None:
        return self.layers["activity"].data

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
    root: Path
    analysed: bool          # it has a stems folder


def find_studies(root: Path = STUDIES_ROOT) -> list[StudyRef]:
    """Every <Artist>/<Title>/ folder under root, analysed or not; hidden folders are skipped."""
    root = Path(root)
    refs = []
    if root.is_dir():
        for artist in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for title in sorted(p for p in artist.iterdir() if p.is_dir() and not p.name.startswith(".")):
                refs.append(StudyRef(artist.name, title.name, title, _stems_dir(title) is not None))
    return refs


def _resolve(name_or_path: str, root: Path) -> Path:
    """The study folder for an existing path, or a title among the analysed studies."""
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        if p.name.endswith("_stems"):
            p = p.parent
        if _stems_dir(p):
            return p
    want = name_or_path.strip().lower()
    hits = [r.root for r in find_studies(root) if r.analysed and want in (
        r.title.lower(), ("%s/%s" % (r.artist, r.title)).lower())]
    if not hits:      # no exact title: a title that contains the name, if that is one study or several
        hits = [r.root for r in find_studies(root) if r.analysed and want in r.title.lower()]
    if not hits:
        raise ValueError("no study named %r under %s" % (name_or_path, root))
    if len(hits) > 1:
        raise ValueError("%r is ambiguous, one of: %s" % (
            name_or_path, ", ".join("%s/%s" % (h.parent.name, h.name) for h in hits)))
    return hits[0]


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
    for item in inputs:
        name = os.path.basename(item["path"])
        what = []
        if os.path.exists(item["path"]):
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
             "file name and size, the recorded paths do not exist here" if by_size and not by_path else
             "path, mtime and size, else file name and size")
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


def load_study(name_or_path: str, root: Path = STUDIES_ROOT) -> Study:
    folder = _resolve(name_or_path, Path(root))
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

    path = stems_dir / "sections.json"
    if path.exists():
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
        else:
            layers["sections"] = unreadable("sections", path)

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
                 stems, layers, sorted(extras))


def to_json(study: Study) -> dict:
    def rel(p):
        return str(Path(p).relative_to(study.root))

    return {
        "kind": "study", "version": 1,
        "artist": study.artist, "title": study.title,
        "bpm": study.bpm, "beats_per_bar": study.beats_per_bar, "bar_offset": study.bar_offset,
        "stems": {r: [rel(f) for f in fs] for r, fs in study.stems.items()},
        "layers": {n: {"source": rel(l.source) if l.source else None, "made_by": l.made_by,
                       "missing": l.missing, "caveats": l.caveats, "data": l.data}
                   for n, l in study.layers.items()},
        "extras": [rel(p) for p in study.extras],
    }


def _show(study: Study) -> None:
    print("%s - %s" % (study.artist, study.title))
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


def _list() -> int:
    refs = find_studies()
    done = [r for r in refs if r.analysed]
    print("analysed (%d)" % len(done))
    for r in done:
        try:
            st = load_study(str(r.root))
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as e:
            print("  %-37s could not read: %s" % ("%s/%s" % (r.artist, r.title), e))
            continue
        have = [n for n, l in st.layers.items() if l.source]
        print("  %-37s has: %-22s missing: %s" % ("%s/%s" % (r.artist, r.title), " ".join(have) or "-",
                                                  " ".join(n for n in st.layers if n not in have) or "-"))
    rest = [r for r in refs if not r.analysed]
    print("not analysed (%d)" % len(rest))
    for r in rest:
        print("  %s/%s" % (r.artist, r.title))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie study", description="Read a study as one record")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every study under the studies root")
    show = sub.add_parser("show", help="print one study")
    show.add_argument("name", help="a title under the studies root, or a path")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        return _list()
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

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
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roadie_analysis import key as keymod

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


# `[124 Am 8A]`: bpm (decimals allowed), key, Camelot code, each of key and code optionally marked `?` (an
# estimate); a study with no key is `[124]` (two or three digits, so `[2024]` is not one)
_SUFFIX = re.compile(r"\s*\[\s*\d{2,3}(?:\.\d+)?(?:\s+[A-G][#b]?m?\??\s+\d{1,2}[AB]\??)?\s*\]\s*$")


def _strip_suffix(name: str) -> str:
    """A folder's title: `Flare [124 Am 8A]` is `Flare`. Only the bpm-key-Camelot suffix goes: `Track [Live]` stays whole."""
    return _SUFFIX.sub("", name).strip() or name


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
    bpm: float | None                    # the grid bpm (what the sections were measured with); None when no layer knows it
    beats_per_bar: int
    bar_offset: int | None               # Live bar = file bar + bar_offset; None when unknown
    stems: dict[str, list[Path]]         # role -> files
    layers: dict[str, Layer]
    extras: list[Path] = field(default_factory=list)
    source: str = ""                      # label of the root it was found under
    key: str | None = None                # "A minor", from study.json
    camelot: str | None = None
    key_source: str | None = None         # one of KEY_SOURCES
    key_confidence: float | None = None   # a detected key's margin over the runner-up
    key_url: str | None = None            # the listing an online key came from
    key_note: str | None = None
    listed_bpm: float | None = None       # the published or tagged tempo, from study.json: for display, never computed with
    bpm_url: str | None = None

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


# Where a key came from, weakest first: `tag` never replaces a key with one from a weaker source.
KEY_SOURCES = ("detected", "online", "tagged", "by ear")


def _text(x) -> str | None:
    return x.strip() if isinstance(x, str) and x.strip() else None


def _key_fields(marker: dict | None) -> dict:
    """The key part of a study.json marker, each field checked; absent or malformed ones are None."""
    m = marker or {}
    key, cam = _text(m.get("key")), _text(m.get("camelot"))
    if key and not cam:
        try:
            cam = keymod.camelot(*keymod.parse_key(key))
        except ValueError:
            pass
    src = m.get("key_source")
    return {"key": key, "camelot": cam, "key_source": src if key and src in KEY_SOURCES else None,
            "key_confidence": _number(m.get("key_confidence")), "key_url": _text(m.get("key_url")),
            "key_note": _text(m.get("key_note")), "listed_bpm": _number(m.get("listed_bpm")),
            "bpm_url": _text(m.get("bpm_url"))}


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
                    refs.append(StudyRef(artist.name, _strip_suffix(title.name), title, stems is not None, r.label, stems))
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
    want = _norm(_strip_suffix(name))
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
    marker = _marker(folder)
    bpm, bar_offset = _number((marker or {}).get("bpm")), None

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

    return Study(folder.parent.name, _strip_suffix(folder.name), folder,
                 bpm, BEATS_PER_BAR, bar_offset,
                 stems, layers, sorted(extras), source, **_key_fields(marker))


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
        "bpm": study.bpm, "key": study.key, "camelot": study.camelot, "key_source": study.key_source,
        "key_confidence": study.key_confidence, "key_url": study.key_url, "key_note": study.key_note,
        "listed_bpm": study.listed_bpm, "bpm_url": study.bpm_url, "beats_per_bar": study.beats_per_bar, "bar_offset": study.bar_offset,
        "stems": {r: [rel(f) for f in fs] for r, fs in study.stems.items()},
        "layers": {n: {"source": rel(l.source) if l.source else None, "made_by": l.made_by,
                       "missing": l.missing, "caveats": l.caveats, "data": l.data}
                   for n, l in study.layers.items() if layers is None or n in layers},
        "extras": [rel(p) for p in study.extras],
    }


def _key_cells(f) -> tuple[str, str]:
    """(short key, Camelot code) for a list column; a detected key is an estimate and prints with `?`."""
    if not f["key"]:
        return "-", "-"
    try:
        short = keymod.short_name(*keymod.parse_key(f["key"]))
    except ValueError:
        short = f["key"]
    mark = "?" if f["key_source"] == "detected" else ""
    return short + mark, (f["camelot"] + mark) if f["camelot"] else "-"


def _listed_note(bpm, listed) -> str:
    """` (listed 92)` when the tempo on record differs from the measured one by more than 1."""
    return " (listed %g)" % listed if listed is not None and (bpm is None or abs(listed - bpm) > 1) else ""


def _bpm_cell(bpm, listed) -> str:
    return ("%g" % bpm if bpm is not None else "-") + _listed_note(bpm, listed).replace("listed ", "")


def _key_line(study: Study) -> str:
    if not study.key:
        return "unknown%s" % (" (%s)" % study.key_note if study.key_note else
                              " (`roadie key %s` estimates it, `roadie study tag` records it)" % study.title)
    short, cam = _key_cells(vars(study))
    why = "source %s" % (study.key_source or "not recorded")
    if study.key_confidence is not None:
        why += ", confidence %g" % study.key_confidence
    if study.key_url:
        why += ", %s" % study.key_url
    if study.key_note:
        why += ", note: %s" % study.key_note
    return "%s%s (%s, %s), %s" % (study.key, "?" if study.key_source == "detected" else "", short, cam, why)


def _show(study: Study) -> None:
    print("%s - %s  [%s]" % (study.artist, study.title, study.source))
    print("  %s bpm%s, %d/4, Live bar = %s" % (
        "%g" % study.bpm if study.bpm is not None else "unknown", _listed_note(study.bpm, study.listed_bpm),
        study.beats_per_bar, "file bar + %d" % study.bar_offset if study.bar_offset is not None else "unknown"))
    print("  key: %s" % _key_line(study))
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


def _layer_flags(ref: StudyRef) -> tuple[dict[str, bool], float | None]:
    """Which layers a study has, by file names (listing, no JSON parsed): the files in the stems
    folder, and `*.profile.json` in the study folder, its analysis/ and the stems folder. The one
    file opened is sections.json (a few KB), searched for the text "bar_activity" and read for its bpm
    when it parses (None otherwise)."""
    flags = {n: False for n in LAYER_NAMES}
    flags["grid"] = (ref.stems_dir / "alignment.json").exists()
    flags["sections"] = (ref.stems_dir / "sections.json").exists()
    bpm = None
    if flags["sections"]:      # the one file opened here, to look for the key and the bpm
        try:
            text = (ref.stems_dir / "sections.json").read_text()
            flags["activity"] = '"bar_activity"' in text
            bpm = _number((json.loads(text) or {}).get("bpm"))
        except (OSError, ValueError, AttributeError):
            pass
    for d in (ref.root, ref.root / "analysis", ref.stems_dir):
        try:
            if any(f.endswith(".profile.json") and not f.startswith(".") for f in os.listdir(d)):
                flags["mix"] = True
        except OSError:
            pass
    return flags, bpm


LAYERS_NOTE = ("layer flags come from file names only, except that sections.json is opened to look for "
               "its bar_activity key (the activity layer); `roadie study show <name>` reads the files "
               "and is the check")


def _list_row(r: StudyRef) -> dict:
    row = {"root": r.source, "path": str(r.root.absolute()), "artist": r.artist, "title": r.title,
           "analysed": r.analysed, "layers": None, "bpm": None, **_key_fields(None)}
    if r.analysed:
        row["layers"], sections_bpm = _layer_flags(r)
        marker = _marker(r.root) or {}
        align = _read_json(r.stems_dir / "alignment.json") if row["layers"]["grid"] else None
        res = align.get("result") if isinstance(align, dict) else None
        # the same order load_study uses: alignment.json, else the marker, else sections.json
        row.update(_key_fields(marker), bpm=_number(res.get("bpm") if isinstance(res, dict) else None,
                                                    _number(marker.get("bpm"), sections_bpm)))
    return row


def list_to_json(roots=None) -> dict:
    """Every study under the roots: {root, path, artist, title, analysed, bpm, key, camelot, key_source,
    key_confidence, key_url, layers: {name: has it}}, and
    each root with whether it is there and what could not be listed. Shallow: directory listings
    only, so a slow mount is cheap."""
    roots = _roots(roots)
    refs, errors, down = _scan(roots)
    studies = [_list_row(r) for r in refs]
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
        if root["analysed"]:
            print("    %-40s %-13s %-6s %-5s" % ("", "bpm (listed)", "key", "camelot"))
        for r in (s for s in rows if s["analysed"]):
            have = [n for n, ok in r["layers"].items() if ok]
            short, cam = _key_cells(r)
            print("    %-40s %-13s %-6s %-5s has: %-22s missing: %s" % (
                "%s/%s" % (r["artist"], r["title"]),
                _bpm_cell(r["bpm"], r["listed_bpm"]), short, cam,
                " ".join(have) or "-", " ".join(n for n in r["layers"] if n not in have) or "-"))
        print("  not analysed (%d)" % root["not_analysed"])
        for r in (s for s in rows if not s["analysed"]):
            print("    %s/%s" % (r["artist"], r["title"]))
    print("note: %s" % data["note"])
    return 0


def _rclone(*args) -> str:
    return subprocess.run(["rclone", *args], check=True, capture_output=True, text=True).stdout


def _remote_read(remote: str) -> str | None:
    """The text of a file on pCloud read through the API, not the FUSE mount (whose view can lag); None
    when the folder lists without it."""
    parent, _, name = remote.rpartition("/")
    return _rclone("cat", remote) if name in _rclone("lsf", "--files-only", parent).splitlines() else None


def _finite(x, what: str, positive: bool = True) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or (positive and x <= 0):
        raise ValueError("%s must be a %sfinite number, not %r" % (what, "positive " if positive else "", x))
    return x


def tag_study(folder: Path, key: str | None = None, bpm: float | None = None, source: str = "tagged",
              confidence: float | None = None, url: str | None = None, bpm_url: str | None = None,
              note: str | None = None) -> dict:
    """Write a study's own `listed_bpm` (the published or tagged tempo: the grid bpm the sections were
    measured with is never touched) and/or key into its `study.json`, changing only those fields (`stems`,
    `t0` and anything else stay). A key from a weaker source than the one on record is refused (KEY_SOURCES:
    by ear beats tagged beats online beats detected; a key with no recorded source counts as tagged). A study
    on the pCloud mount is read and written through rclone (never the mount), and read back to check the
    write; a local one directly. Returns the marker as written."""
    if key is None and bpm is None and note is None:
        raise ValueError("nothing to tag: give a key, a bpm and/or a note")
    if source not in KEY_SOURCES:
        raise ValueError("key source must be one of %s" % ", ".join(KEY_SOURCES))
    path = folder / "study.json"
    remote = keymod.pcloud_remote(path)
    if remote:
        old_text = _remote_read(remote)
        existing = old_text is not None
        marker = _parse_marker(old_text) if existing else {"kind": "study_marker"}
    else:
        existing = path.exists()
        marker = _parse_marker(path.read_text()) if existing else {"kind": "study_marker"}
    if marker is None:
        raise ValueError("%s is not a study marker (kind study_marker): not overwriting it" % path)
    if bpm is not None:
        marker["listed_bpm"] = _finite(bpm, "bpm")
        if bpm_url:
            marker["bpm_url"] = bpm_url
    if key is None and note is not None:
        marker["key_note"] = note
    if key is not None:
        if confidence is not None:
            _finite(confidence, "confidence", positive=False)
        pc, mode = keymod.parse_key(key)
        old = _key_fields(marker)
        old_source = old["key_source"] or "tagged"
        if old["key"] and KEY_SOURCES.index(source) < KEY_SOURCES.index(old_source):
            raise ValueError("%s already has a key from source %r (%s), which outranks %r: not changed"
                             % (folder.name, old_source, old["key"], source))
        for stale in ("key_confidence", "key_url", "key_note"):     # they described the old key
            marker.pop(stale, None)
        marker.update(key=keymod.key_name(pc, mode), camelot=keymod.camelot(pc, mode), key_source=source)
        for name, value in (("key_confidence", confidence), ("key_url", url), ("key_note", note)):
            if value is not None:
                marker[name] = value
    text = json.dumps(marker, indent=2, ensure_ascii=False) + "\n"
    if remote:
        with tempfile.TemporaryDirectory(prefix="roadie-tag-") as tmp:
            Path(tmp, "study.json").write_text(text)
            _rclone("copyto", str(Path(tmp, "study.json")), remote)
        if _remote_read(remote) != text:
            raise OSError("%s: read back after writing and it is not what was written" % remote)
    else:
        tmp_path = path.with_name(".study.json.tmp")
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    return marker


def _parse_marker(text: str) -> dict | None:
    """A study.json marker from its text, None when it is not one (unparseable, or another kind of file)."""
    try:
        m = json.loads(text)
    except ValueError:
        return None
    return m if isinstance(m, dict) and m.get("kind") == "study_marker" else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie study", description="Read a study as one record")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="every study under the studies root")
    ls.add_argument("--json", action="store_true", help="print the list as JSON")
    show = sub.add_parser("show", help="print one study")
    show.add_argument("name", help="a title under the studies root, or a path")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    tag = sub.add_parser("tag", help="write a study's bpm and/or key into its study.json (the only command that writes)")
    tag.add_argument("name", help="a title under the studies root, or a path")
    tag.add_argument("--key", help="the key: Am, A minor, F#Maj, 8A, ...")
    tag.add_argument("--bpm", type=float, help="the published or tagged tempo (written as listed_bpm; the measured bpm is not changed)")
    tag.add_argument("--bpm-url", help="the listing the bpm came from")
    tag.add_argument("--key-note", help="a note kept with the key, e.g. an older tag it replaced")
    tag.add_argument("--by-ear", action="store_true", help="the key was settled by ear (beats every other source)")
    tag.add_argument("--source", choices=KEY_SOURCES[:3], help="where the key came from (default tagged: the user's own)")
    tag.add_argument("--url", help="the listing an online key came from")
    tag.add_argument("--confidence", type=float, help="a detected key's margin over the runner-up")
    fill = sub.add_parser("fill-keys", help="dry run: the keys and listed bpm the records would get, as one table (writes nothing here)")
    fill.add_argument("--from", dest="lookup", required=True, help="the published-lookup JSON")
    fill.add_argument("--dry-run", action="store_true", help="make the table only")
    fill.add_argument("--write", action="store_true", help="record the keys (needs --backup-dir; not for the local root)")
    fill.add_argument("--backup-dir", help="where --write saves each study.json it replaces")
    fill.add_argument("--log", help="write the --write log to this file")
    fill.add_argument("--root", default="pcloud", help="the root label to plan for (default pcloud)")
    fill.add_argument("--out", help="write the table (markdown) to this file")
    fill.add_argument("--cache", help="keep detection results in this JSON file, so a rerun does not repeat them")
    args = ap.parse_args(argv)
    if args.cmd == "fill-keys":
        import keyfill
        return keyfill.main(args)
    if args.cmd == "list":
        return _list(args.json)
    if args.cmd == "tag":
        try:
            if args.by_ear and args.source:
                raise ValueError("--by-ear and --source are two answers to the same question")
            folder, _ = _resolve(args.name, _roots(None))
            m = tag_study(folder, args.key, args.bpm, "by ear" if args.by_ear else args.source or "tagged",
                          args.confidence, args.url, args.bpm_url, args.key_note)
        except (ValueError, OSError, subprocess.CalledProcessError) as e:
            print("study: tag: %s" % (e.stderr.decode(errors="replace").strip() if isinstance(e, subprocess.CalledProcessError)
                                      and e.stderr else e), file=sys.stderr)
            return 1
        print("tagged %s/%s: listed bpm %s, key %s" % (folder.parent.name, _strip_suffix(folder.name),
              "%g" % m["listed_bpm"] if "listed_bpm" in m else "-",
              "%s (%s), %s" % (m["key"], m["camelot"], m["key_source"]) if m.get("key") else "-"))
        return 0
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

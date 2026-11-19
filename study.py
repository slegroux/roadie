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
    bpm: float
    beats_per_bar: int
    bar_offset: int                      # Live bar = file bar + bar_offset
    stems: dict[str, list[Path]]         # role -> files
    layers: dict[str, Layer]
    extras: list[Path] = field(default_factory=list)

    def sections(self) -> list[Section]:
        data = self.layers["sections"].data
        if not data:
            return []
        return [Section(s["label"], s["bar"], s["duration_beats"] // self.beats_per_bar,
                        s.get("live_bar"), s["stems"]) for s in data["sections"]]

    def activity(self) -> dict[str, list[bool]] | None:
        return self.layers["activity"].data

    def mix(self) -> list[dict] | None:
        data = self.layers["mix"].data
        return data["sections"] if data else None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _stems_dir(folder: Path) -> Path | None:
    found = sorted(p for p in folder.glob("*_stems") if p.is_dir())
    return found[0] if found else None


def _resolve(name_or_path: str, root: Path) -> Path:
    """The study folder for an existing path, or a title among the folders that have stems."""
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        if p.name.endswith("_stems"):
            p = p.parent
        if _stems_dir(p):
            return p
    want = name_or_path.strip().lower()
    hits = []
    for stems in sorted(root.glob("*/*/*_stems")) if root.is_dir() else []:
        folder = stems.parent
        if want in (folder.name.lower(), ("%s/%s" % (folder.parent.name, folder.name)).lower()):
            hits.append(folder)
    if not hits:
        raise ValueError("no study named %r under %s" % (name_or_path, root))
    if len(hits) > 1:
        raise ValueError("%r is ambiguous, one of: %s" % (
            name_or_path, ", ".join("%s/%s" % (h.parent.name, h.name) for h in hits)))
    return hits[0]


def load_study(name_or_path: str, root: Path = STUDIES_ROOT) -> Study:
    folder = _resolve(name_or_path, Path(root))
    stems_dir = _stems_dir(folder)
    files = sorted(f for f in stems_dir.iterdir() if f.suffix in (".flac", ".wav"))
    stems = {role: [f for n in names for f in files if f.stem == n]
             for role, names in STEM_ROLES.items()}
    stems = {role: fs for role, fs in stems.items() if fs}

    layers = {n: Layer(n, missing=MISSING[n]) for n in LAYER_NAMES}
    bpm, bar_offset = 0.0, 0

    align = _read_json(stems_dir / "alignment.json")
    if align and "result" in align:
        res = align["result"]
        bpm = float(res["bpm"])
        caveats = []
        if res.get("downbeat_verdict") == "chance-level":
            caveats.append("downbeat confidence is chance-level (%s)" % res.get("downbeat_confidence"))
        layers["grid"] = Layer("grid", stems_dir / "alignment.json",
                               "beat_aligner (%s)" % res.get("detector", "?"), res, None, caveats)

    secs = _read_json(stems_dir / "sections.json")
    if secs and secs.get("sections"):
        bpm = bpm or float(secs["bpm"])
        first = secs["sections"][0]
        if first.get("live_bar") is not None:
            bar_offset = int(round(first["live_bar"] - first["bar"]))
        caveats = ["labels by rule, not checked by ear"]
        if first.get("live_bar") is None:
            caveats.append("no live_bar: no downbeat was known when it was made")
        layers["sections"] = Layer("sections", stems_dir / "sections.json",
                                   "sections (%s method)" % secs.get("method", "?"), secs,
                                   None, caveats)

    return Study(folder.parent.name, folder.name, folder, bpm, BEATS_PER_BAR, bar_offset,
                 stems, layers)


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
    print("  %g bpm, %d/4, Live bar = file bar + %d" % (study.bpm, study.beats_per_bar, study.bar_offset))
    print("  stems: %s" % ", ".join(study.stems))
    for name, layer in study.layers.items():
        if layer.data is None:
            print("  %-9s missing: %s" % (name, layer.missing))
            continue
        print("  %-9s %s  [%s]" % (name, layer.source.relative_to(study.root), layer.made_by))
        for c in layer.caveats:
            print("            caveat: %s" % c)
        if name == "sections":
            for s in study.sections():
                live = "Live bar %g" % s.live_bar if s.live_bar is not None else "-"
                print("    %-12s file bar %-4d %3d bars  %-12s %s" % (
                    s.label, s.bar, s.bars, live, " ".join(s.roles)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie study", description="Read a study as one record")
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="print one study")
    show.add_argument("name", help="a title under the studies root, or a path")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    args = ap.parse_args(argv)
    try:
        study = load_study(args.name)
    except ValueError as e:
        print("study: %s" % e, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(to_json(study), indent=2))
    else:
        _show(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""roadie drums: the roles inside each drum stem of a study (roadie_analysis/drums.py, docs/DRUMROLES.md).

    roadie drums <study name | stems_dir> [--jobs N] [--json] [--no-write]

Reads the grid from alignment.json and sections.json beside the stems, writes drums.json and
drum_roles/<stem>-<role>.wav there (nothing with --no-write), prints a table per stem.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import study  # noqa: E402
from roadie_analysis import drums  # noqa: E402
from roadie_analysis.jobqueue import heavy_job  # noqa: E402


def stems_dir_of(arg: str) -> Path:
    """A folder with drum stems as it is, else a study (by name or path) and its stems folder."""
    p = Path(arg).expanduser()
    if p.is_dir() and drums.find_stems(p):
        return p
    folder, _src = study._resolve(arg, study._roots(None))
    sd = study._stems_dir(folder)
    if sd is None:
        raise ValueError("no stems folder in %s" % folder)
    return sd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie drums", description="Drum roles of a study's drum stems")
    ap.add_argument("target", help="a study (title or path) or a stems folder")
    ap.add_argument("--json", action="store_true", help="print the drums.json record instead of the table")
    ap.add_argument("--jobs", type=int, default=None, metavar="N", help="processes for the (stem, section) jobs (default: the fast cores; 1: one process)")
    ap.add_argument("--no-write", action="store_true", help="write no drums.json and no clips")
    args = ap.parse_args(argv)
    try:
        sd = stems_dir_of(args.target)
        with heavy_job("roadie drums %s" % args.target):
            t = time.monotonic()
            rec = drums.drum_roles(sd, write=not args.no_write, jobs=args.jobs)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print("drums: %s" % e, file=sys.stderr)
        return 1
    secs = time.monotonic() - t
    note = None if args.json else study.stem_model_note(study._stem_model(sd))
    print(json.dumps(rec, indent=2) if args.json else (note + "\n" if note else "") + drums.format_table(rec))
    if not args.json:
        print("\n%s: %.0f s with %d jobs (%s)" % ("written to %s" % rec["written"] if "written" in rec else "nothing written (--no-write)",
                                              secs, rec["jobs"], ", ".join("%s %.0f s" % kv for kv in rec["stage_seconds"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

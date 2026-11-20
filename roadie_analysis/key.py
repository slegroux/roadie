#!/usr/bin/env python3
"""The musical key of a study, from its bass and synths stems: `roadie key <study or stems folder> [--json]`.

An estimate, not a fact: chroma (CQT) summed over each stem's whole length, the two stems weighted
equally, correlated with the 24 rotated Krumhansl-Kessler major and minor profiles. The confidence is the
margin of the best correlation over the runner-up's, so a relative major/minor pair or a fifth apart
(the usual near misses) shows as a small margin. A published or tagged key beats this (see
`study tag`); it is for cross-checking and for tracks with no listing.

The top half of this file is the pure part: the key <-> Camelot table, the short and long spellings, and
`parse_key`, which reads the forms the user's tags use ("Em", "EMaj", "F#Maj", "GMaj", "C", "6A", "11B").
A study on the pCloud mount is not read through the FUSE mount (that fills a local cache with audio):
its bass and synths stems are copied to a temp folder with `rclone copyto`, analysed, and deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAJOR, MINOR = "major", "minor"

# Pitch-class spellings as DJ software prints them: sharps for minor keys (C#m, G#m), flats for the
# others, F# for the tritone.
MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
# Camelot wheel: the tonic's pitch class (C = 0) at number 1..12, B ring = major, A ring = minor.
CAMELOT_TONICS = {MAJOR: (11, 6, 1, 8, 3, 10, 5, 0, 7, 2, 9, 4),
                  MINOR: (8, 3, 10, 5, 0, 7, 2, 9, 4, 11, 6, 1)}
CAMELOT_RING = {MAJOR: "B", MINOR: "A"}

_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def key_name(pc: int, mode: str) -> str:
    return "%s %s" % ((MAJOR_NAMES if mode == MAJOR else MINOR_NAMES)[pc % 12], mode)


def short_name(pc: int, mode: str) -> str:
    return (MAJOR_NAMES if mode == MAJOR else MINOR_NAMES)[pc % 12] + ("m" if mode == MINOR else "")


def camelot(pc: int, mode: str) -> str:
    return "%d%s" % (CAMELOT_TONICS[mode].index(pc % 12) + 1, CAMELOT_RING[mode])


def parse_key(text: str) -> tuple[int, str]:
    """(tonic pitch class, mode) from "A minor", "Am", "EMaj", "F#Maj", "C" (major), "Abm", or a Camelot
    code ("6A", "11B"). A bare note is major; a lower-case `m` or `min`/`minor` is minor, `M`/`maj`/`major`
    major. Raises ValueError for anything else."""
    s = text.strip()
    m = re.fullmatch(r"(\d{1,2})\s*([ABab])", s)
    if m:
        n, ring = int(m.group(1)), m.group(2).upper()
        if not 1 <= n <= 12:
            raise ValueError("no such Camelot code: %r" % text)
        mode = MINOR if ring == "A" else MAJOR
        return CAMELOT_TONICS[mode][n - 1], mode
    m = re.fullmatch(r"([A-Ga-g])\s*([#b♯♭]?)\s*([A-Za-z]*)", s)
    if not m:
        raise ValueError("not a key: %r" % text)
    pc = _LETTERS[m.group(1).upper()] + {"#": 1, "♯": 1, "b": -1, "♭": -1, "": 0}[m.group(2)]
    suffix = m.group(3)
    if suffix in ("m",) or suffix.lower() in ("min", "minor"):
        return pc % 12, MINOR
    if suffix in ("", "M") or suffix.lower() in ("maj", "major"):
        return pc % 12, MAJOR
    raise ValueError("not a key: %r" % text)


# Krumhansl-Kessler probe-tone profiles, tonic first.
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _pearson(a, b) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def key_from_chroma(chroma) -> dict:
    """Rank the 24 keys for a 12-value chroma (index 0 = C). Returns the best key, its correlation,
    the runner-up and `confidence` = best - runner-up."""
    scores = []
    for mode, prof in ((MAJOR, KK_MAJOR), (MINOR, KK_MINOR)):
        for pc in range(12):
            rotated = [prof[(i - pc) % 12] for i in range(12)]
            scores.append((_pearson(list(chroma), rotated), pc, mode))
    scores.sort(reverse=True)

    def entry(s):
        return {"key": key_name(s[1], s[2]), "short": short_name(s[1], s[2]),
                "camelot": camelot(s[1], s[2]), "score": round(s[0], 4)}

    best, second = entry(scores[0]), entry(scores[1])
    return dict(best, confidence=round(scores[0][0] - scores[1][0], 4), runner_up=second)


def stem_chroma(path, sr: int = 22050) -> list[float]:
    """One stem's chroma summed over its length and scaled to sum 1, so a loud stem does not drown a quiet one."""
    import librosa
    import numpy as np
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048).sum(axis=1)
    total = float(np.sum(c))
    return [float(x) / total for x in c] if total > 0 else [0.0] * 12


def detect_key(paths) -> dict:
    """The key of these audio files taken together (the stems' chroma added up), from `key_from_chroma`."""
    paths = list(paths)
    if not paths:
        raise ValueError("no stems to read the key from")
    total = [0.0] * 12
    for p in paths:
        total = [a + b for a, b in zip(total, stem_chroma(p))]
    return dict(key_from_chroma(total), stems=[Path(p).name for p in paths])


# ---- reading a study

def pcloud_remote(path) -> str | None:
    """`pcloud:<path under the pCloud Drive mount>` for a file on the FUSE mount, else None."""
    try:
        rel = Path(path).resolve().relative_to(Path("~/pCloud Drive").expanduser().resolve())
    except ValueError:
        return None
    return "pcloud:" + rel.as_posix()


def _stem_files(arg: str) -> list[Path]:
    """The bass and synths stems of a study (a title or path) or of a stems folder with no study around it."""
    here = str(Path(__file__).resolve().parent.parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import study
    try:
        files = [f for role in ("bass", "synths") for f in study.load_study(arg).stems.get(role, [])]
    except ValueError:
        folder = Path(arg).expanduser()
        if not folder.is_dir():
            raise
        files = [f for n in ("bass", "synths") for f in sorted(folder.glob(n + ".*"))
                 if f.suffix in (".flac", ".wav")]
    if not files:
        raise ValueError("%s has no bass or synths stem" % arg)
    return files


def key_of(arg: str) -> dict:
    files = _stem_files(arg)
    remotes = [pcloud_remote(f) for f in files]
    if not any(remotes):
        return detect_key(files)
    with tempfile.TemporaryDirectory(prefix="roadie-key-") as tmp:
        local = []
        for f, remote in zip(files, remotes):
            dest = Path(tmp) / f.name
            subprocess.run(["rclone", "copyto", remote, str(dest)], check=True, capture_output=True)
            local.append(dest)
        return detect_key(local)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadie key", description="Estimate a study's key from its bass and synths stems")
    ap.add_argument("study", help="a study title or path, or a stems folder")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    try:
        r = key_of(args.study)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print("key: %s" % (e.stderr.decode(errors="replace").strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e),
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        ru = r["runner_up"]
        print("%s (%s, %s)  confidence %.3f  runner-up %s (%s)  [%s]" % (
            r["key"], r["short"], r["camelot"], r["confidence"], ru["key"], ru["camelot"], ", ".join(r["stems"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

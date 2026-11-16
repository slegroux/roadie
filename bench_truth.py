#!/usr/bin/env python3
"""Build the ground truth for the separation benchmark from a remix pack.

    bench_truth.py bench/synthwave1.json [--out DIR] [--verify-only]

The manifest maps every file of a multitrack pack to one of Roadie's stem names
(drums_kick, drums_snare, drums_toms, drums_hh, drums_ride, drums_crash, bass,
synths, vocals) or to `returns` (printed effect returns, which have no true role).
This writes, into the manifest's out_dir (or --out):

    mix.wav               every pack file summed at unity, the pack's rate and bit depth
    truth/<role>.flac     one true stem per role that has at least one source

then checks the files it wrote: the pack's frame count is 177 bars at 124 BPM, and
the mix minus the sum of the role stems (and minus a fresh sum of the pack) is below
-90 dB. Audio is read from the pack and written to out_dir only; nothing here, and
nothing in the manifest, is audio. The pack is commercial material.

Memory: the role sums are held as float64 while building, about 2.5 GB.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

BLOCK = 1 << 19      # frames per block when verifying
PASS_DB = -90.0


def load_manifest(path):
    with open(path) as fh:
        m = json.load(fh)
    for k in ("pack_dir", "out_dir"):
        m[k] = os.path.expanduser(m[k])
    return m


def check_manifest(m):
    """Every .aif in pack_dir is either a source or excluded, and every source exists."""
    files = [s["file"] for s in m["sources"]]
    excluded = [e["file"] for e in m["excluded"]]
    if len(set(files)) != len(files):
        raise SystemExit("manifest lists a file twice")
    on_disk = {f for f in os.listdir(m["pack_dir"]) if f.lower().endswith((".aif", ".aiff"))}
    missing = [f for f in files if f not in on_disk]
    unlisted = sorted(on_disk - set(files) - set(excluded))
    if missing or unlisted:
        raise SystemExit("manifest and %s disagree: missing %s, unlisted %s"
                         % (m["pack_dir"], missing, unlisted))
    roles = set(m["stems"]) | {"returns"}
    bad = [s["file"] for s in m["sources"] if s["role"] not in roles]
    if bad:
        raise SystemExit("unknown role for %s" % bad)


def pack_format(m):
    """Frame count, rate and subtype shared by all sources; refuses a pack that disagrees."""
    infos = {s["file"]: sf.info(os.path.join(m["pack_dir"], s["file"])) for s in m["sources"]}
    keys = {(i.frames, i.samplerate, i.channels, i.subtype) for i in infos.values()}
    if len(keys) != 1:
        raise SystemExit("pack files differ in length or format: %s" % sorted(keys))
    frames, sr, ch, subtype = keys.pop()
    bar_s = m["beats_per_bar"] * 60.0 / m["bpm"]
    bars = frames / sr / bar_s
    if abs(bars - m["bars"]) > 0.01:
        raise SystemExit("pack is %.3f bars at %s BPM, manifest says %d" % (bars, m["bpm"], m["bars"]))
    return frames, sr, ch, subtype, bars


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def build(m, out_dir):
    frames, sr, ch, subtype, bars = pack_format(m)
    roles = [r for r in m["stems"] + ["returns"] if any(s["role"] == r for s in m["sources"])]
    os.makedirs(os.path.join(out_dir, "truth"), exist_ok=True)

    # One whole file at a time, front to back: the pack sits on a cloud drive, where reading
    # 30 files in interleaved blocks was minutes slower than reading each in one go.
    acc = {r: np.zeros((frames, ch)) for r in roles}
    for s in m["sources"]:
        acc[s["role"]] += sf.read(os.path.join(m["pack_dir"], s["file"]), dtype="float64",
                                  always_2d=True)[0]
    mix = sum(acc.values())
    peak = {"mix": float(np.abs(mix).max()), **{r: float(np.abs(acc[r]).max()) for r in roles}}
    if any(v >= 1.0 for v in peak.values()):
        raise SystemExit("a sum reaches full scale; a fixed-point file would clip")
    sf.write(os.path.join(out_dir, "mix.wav"), mix, sr, subtype=subtype, format="WAV")
    for r in roles:
        sf.write(os.path.join(out_dir, "truth", r + ".flac"), acc[r], sr, subtype=subtype, format="FLAC")
    return roles, frames, sr, subtype, bars, peak


def verify(m, out_dir, roles, frames):
    """Residuals, read back from the files on disk: written mix vs the sum of the written
    role stems, and written mix vs a fresh sum of the pack."""
    fresh = None
    for s in m["sources"]:
        x = sf.read(os.path.join(m["pack_dir"], s["file"]), dtype="float64", always_2d=True)[0]
        fresh = x if fresh is None else fresh + x
    mix = sf.SoundFile(os.path.join(out_dir, "mix.wav"))
    rf = {r: sf.SoundFile(os.path.join(out_dir, "truth", r + ".flac")) for r in roles}
    res = {"roles": [0.0, 0.0], "pack": [0.0, 0.0]}      # [peak, sum of squares]
    mix_ss, count = 0.0, 0
    try:
        done = 0
        while done < frames:
            n = min(BLOCK, frames - done)
            x = mix.read(n, dtype="float64", always_2d=True)
            r_sum = sum(f.read(n, dtype="float64", always_2d=True) for f in rf.values())
            for key, other in (("roles", r_sum), ("pack", fresh[done:done + n])):
                d = x - other
                res[key][0] = max(res[key][0], float(np.abs(d).max()))
                res[key][1] += float((d ** 2).sum())
            mix_ss += float((x ** 2).sum())
            count += x.size
            done += n
    finally:
        mix.close()
        for f in rf.values():
            f.close()
    mix_rms = (mix_ss / count) ** 0.5
    out = {}
    for key, (pk, ss) in res.items():
        out[key] = {"peak_dbfs": db(pk), "rms_rel_mix_db": db((ss / count) ** 0.5 / mix_rms)}
    return out, db(mix_rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", help="output folder (default: the manifest's out_dir)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the files already written; do not rebuild")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    out_dir = os.path.expanduser(args.out) if args.out else m["out_dir"]
    check_manifest(m)
    print("%d pack files, %d excluded (%s)"
          % (len(m["sources"]), len(m["excluded"]), ", ".join(e["file"] for e in m["excluded"])))

    if args.verify_only:
        frames, sr, _ch, subtype, bars = pack_format(m)
        roles = [r for r in m["stems"] + ["returns"] if any(s["role"] == r for s in m["sources"])]
        peak = None
    else:
        roles, frames, sr, subtype, bars, peak = build(m, out_dir)

    print("%d files, %.2f bars at %s BPM (%d frames, %d Hz, %s)"
          % (len(m["sources"]), bars, m["bpm"], frames, sr, subtype))
    print("role stems: " + ", ".join("%s (%d)" % (r, sum(s["role"] == r for s in m["sources"]))
                                     for r in roles))
    empty = [r for r in m["stems"] if r not in roles]
    if empty:
        print("no source, no stem written: " + ", ".join(empty))
    if peak:
        print("mix peak %.2f dBFS, no gain" % db(peak["mix"]))
        for r in roles:
            print("  %-12s peak %7.2f dBFS" % (r, db(peak[r])))

    res, mix_rms_db = verify(m, out_dir, roles, frames)
    print("mix RMS %.2f dBFS" % mix_rms_db)
    ok = True
    for key, label in (("roles", "mix - sum of role stems"), ("pack", "mix - fresh sum of the pack")):
        r = res[key]
        good = r["peak_dbfs"] < PASS_DB and r["rms_rel_mix_db"] < PASS_DB
        ok &= good
        print("%-28s peak %8.2f dBFS, RMS %8.2f dB re mix  %s"
              % (label, r["peak_dbfs"], r["rms_rel_mix_db"], "ok" if good else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

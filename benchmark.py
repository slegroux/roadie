#!/usr/bin/env python3
"""Benchmark stem-separation models across a corpus of tracks.

    benchmark.py <corpus_dir> [--models a,b,c] [--samples 5] [--excerpt 60]
                              [--overlap 0.5] [--out results.csv]

Answers one question per track: **how much of the bass does each model actually
recover?** Published SDR is measured on MUSDB18 (live bands, electric bass), which
does not represent 4x4 material where a kick and a sub share an octave. On one
house set the model ranked WORST on published bass SDR was the only one that
produced a bass stem at all. This script exists to find out whether that holds
across a corpus, or was a fluke.

Primary metric — bass deficit, in dB:

    deficit = (source 20-120 Hz level) - (bass stem 20-120 Hz level)

Level-independent, so it compares across tracks. Lower is better. A deficit near
zero means the bass landed in the bass stem; a large deficit means it went
somewhere else (usually drums, since the kick shares that band).

WHY --samples EXISTS. Measuring one 60s window is not enough. On the first track
tested, htdemucs_ft moved 29 dB between two runs of the SAME track — second-best
to worst. Two things differed between those runs (excerpt length and --overlap),
so the cause was not isolated. Sampling several windows per track exposes that
volatility directly: a model whose deficit swings 30 dB within one track cannot
be ranked from a single window, and that variance is itself a result worth
reporting.

Caveat this script cannot fix: without reference stems there is no ground truth,
so this measures *where energy went*, not perceptual quality. It will not tell you
a hi-hat bled into the snare. Treat it as a strong signal about gross
misallocation, not a quality score.

Requires: ffmpeg, demucs (uv tool install demucs --with numpy).
Results are cached per (track, model, offset, overlap); re-runs resume.
"""

import argparse, csv, os, re, shutil, subprocess, sys, tempfile

SUB_LO, SUB_HI = 20, 120          # where a 4x4 kick and a sub-bass collide
DEFAULT_MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]
AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a")
STEM_COLS = ("bass", "drums", "other", "vocals", "piano", "guitar")


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def ff_level(path, lo=None, hi=None):
    """Mean level in dB, optionally band-limited. -120.0 if silent/unreadable."""
    af = "volumedetect"
    if lo is not None:
        af = "highpass=f=%d,lowpass=f=%d,%s" % (lo, hi, af)
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-af", af, "-f", "null", "/dev/null"],
        capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", out)
    return float(m.group(1)) if m else -120.0


def sample_offsets(dur, n, length):
    """N evenly spaced windows, skipping the first and last 10% so intros and
    outros (often sparse or silent) do not dominate the measurement."""
    lo, hi = dur * 0.10, dur * 0.90 - length
    if hi <= lo:
        return [max(0.0, (dur - length) / 2)]
    if n == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 1) for i in range(n)]


def make_excerpt(src, dst, offset, length):
    """Excerpt at 44.1k/24-bit so every model sees identical input."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-ss", str(offset), "-t", str(length), "-i", src,
                    "-ar", "44100", "-c:a", "pcm_s24le", dst], check=True)


def separate(clip, model, outdir, overlap):
    """Run demucs; return {stem_name: path}. Empty dict on failure."""
    r = subprocess.run(
        ["demucs", "-n", model, "--int24", "--overlap", str(overlap),
         "-o", outdir, clip],
        capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or [""])[-1]
        sys.stderr.write("  demucs failed (%s): %s\n" % (model, tail))
        return {}
    stems = {}
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if f.endswith(".wav"):
                stems[f[:-4]] = os.path.join(root, f)
    return stems


def bench_track(track, models, offsets, length, overlap, cache):
    rows, name = [], os.path.basename(track)
    for si, off in enumerate(offsets):
        print("  sample %d/%d @ %.0fs" % (si + 1, len(offsets), off))
        tmp = tempfile.mkdtemp(prefix="bench_")
        try:
            clip = os.path.join(tmp, "clip.wav")
            make_excerpt(track, clip, off, length)
            src_sub, src_all = ff_level(clip, SUB_LO, SUB_HI), ff_level(clip)

            for model in models:
                key = "%s::%s::%s::%s" % (name, model, off, overlap)
                if key in cache:
                    rows.append(cache[key])
                    print("    %-13s (cached)" % model)
                    continue
                stems = separate(clip, model, os.path.join(tmp, model), overlap)
                if "bass" not in stems:
                    print("    %-13s FAILED" % model)
                    continue
                bass_sub = ff_level(stems["bass"], SUB_LO, SUB_HI)
                row = {
                    "track": name, "model": model,
                    "offset_s": off, "excerpt_s": length, "overlap": overlap,
                    "src_sub_db": round(src_sub, 1),
                    "src_mean_db": round(src_all, 1),
                    "bass_sub_db": round(bass_sub, 1),
                    "bass_deficit_db": round(src_sub - bass_sub, 1),
                    "n_stems": len(stems),
                }
                for s in STEM_COLS:
                    row[s + "_db"] = round(ff_level(stems[s]), 1) if s in stems else ""
                rows.append(row)
                cache[key] = row
                print("    %-13s deficit %6.1f dB" % (model, row["bass_deficit_db"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rows


def _stats(vals):
    v = sorted(vals)
    return v[len(v) // 2], sum(v) / len(v), v[0], v[-1]


def summarise(rows, models):
    if not rows:
        return
    print("\n" + "=" * 74)
    print("BASS DEFICIT BY MODEL  (dB below source sub-band; lower is better)")
    print("=" * 74)
    print("%-14s %4s %8s %8s %8s %8s %8s"
          % ("model", "n", "median", "mean", "best", "worst", "range"))
    for m in models:
        vals = [r["bass_deficit_db"] for r in rows if r["model"] == m]
        if not vals:
            continue
        med, mean, best, worst = _stats(vals)
        print("%-14s %4d %8.1f %8.1f %8.1f %8.1f %8.1f"
              % (m, len(vals), med, mean, best, worst, worst - best))

    # Within-track spread is the stability question: a model that swings widely
    # across windows of the SAME track cannot be ranked from one measurement.
    print("\nWITHIN-TRACK SPREAD ACROSS SAMPLES (max range on any single track)")
    unstable = []
    for m in models:
        worst_range, worst_track = 0.0, ""
        for t in sorted({r["track"] for r in rows}):
            vals = [r["bass_deficit_db"] for r in rows
                    if r["model"] == m and r["track"] == t]
            if len(vals) > 1 and (max(vals) - min(vals)) > worst_range:
                worst_range, worst_track = max(vals) - min(vals), t
        if worst_track:
            flag = "  <-- UNSTABLE" if worst_range >= 10 else ""
            print("  %-14s %6.1f dB   (%s)%s" % (m, worst_range, worst_track[:34], flag))
            if worst_range >= 10:
                unstable.append(m)

    print("\nPER-TRACK WINNER (dissent is the interesting part, not the average)")
    wins = {}
    for t in sorted({r["track"] for r in rows}):
        best_m, best_v = None, None
        for m in models:
            vals = [r["bass_deficit_db"] for r in rows
                    if r["model"] == m and r["track"] == t]
            if not vals:
                continue
            med = sorted(vals)[len(vals) // 2]        # median across samples
            if best_v is None or med < best_v:
                best_m, best_v = m, med
        if best_m:
            wins[best_m] = wins.get(best_m, 0) + 1
            print("  %-40s %-13s %6.1f dB (median)" % (t[:40], best_m, best_v))

    print("\nWINS: " + ", ".join("%s=%d" % (m, wins.get(m, 0)) for m in models))
    n_tracks = len({r["track"] for r in rows})
    if unstable:
        print("CAUTION: %s vary >=10 dB within a single track. Single-window "
              "rankings are not trustworthy for these." % ", ".join(unstable))
    if len(wins) > 1:
        print("Models disagree across the corpus — report the split, not one winner.")
    if n_tracks < 5:
        print("NOTE: only %d track(s). Too few to generalise — plan calls for 5-10."
              % n_tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--samples", type=int, default=5,
                    help="windows per track (default 5)")
    ap.add_argument("--excerpt", type=int, default=60, help="seconds (default 60)")
    ap.add_argument("--overlap", default="0.5",
                    help="demucs --overlap (default 0.5; demucs's own default is 0.25)")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tracks = sorted(os.path.join(args.corpus_dir, f)
                    for f in os.listdir(args.corpus_dir)
                    if f.lower().endswith(AUDIO_EXT))
    if not tracks:
        sys.exit("no audio files in %s" % args.corpus_dir)

    cache = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for r in csv.DictReader(fh):
                for k in ("src_sub_db", "src_mean_db", "bass_sub_db",
                          "bass_deficit_db", "offset_s"):
                    r[k] = float(r[k])
                r["n_stems"] = int(r["n_stems"])
                r["excerpt_s"] = int(r["excerpt_s"])
                cache["%s::%s::%s::%s" % (r["track"], r["model"],
                                          r["offset_s"], r["overlap"])] = r
        print("resuming: %d cached results\n" % len(cache))

    all_rows = []
    for i, t in enumerate(tracks, 1):
        dur = duration(t)
        offs = sample_offsets(dur, args.samples, args.excerpt)
        print("[%d/%d] %s  (%.0fs, %d samples)"
              % (i, len(tracks), os.path.basename(t), dur, len(offs)))
        all_rows += bench_track(t, models, offs, args.excerpt, args.overlap, cache)

    if all_rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print("\nwrote %s (%d rows)" % (args.out, len(all_rows)))
        summarise(all_rows, models)


if __name__ == "__main__":
    main()

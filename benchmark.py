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

TRUTH MODE. With true stems the caveat above goes away:

    benchmark.py --truth bench/synthwave1.json [--stems DIR] [--json OUT]

scores the stems `yt2stems.sh -d` wrote for a mix against the true role stems that
bench_truth.py wrote for it. Per separated stem: SI-SDR over the whole track and the
median per-bar SI-SDR over the bars where the true role plays, the level gap, the
per-bar energy correlation, and a bleed matrix. See the functions below for the
exact definitions.
"""

import argparse, csv, json, os, re, shutil, subprocess, sys, tempfile

import numpy as np

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


# ---------------------------------------------------------------------------
# Truth mode
# ---------------------------------------------------------------------------

BLEED_BANDS = (20, 120, 500, 2000, 6000, 20000)   # Hz edges: sub, low, mid, high-mid, air
FRAME = 4096                                      # band-energy frame; divides TRUTH_BLOCK
TRUTH_BLOCK = 1 << 19
ACTIVE_FLOOR_DB = -80.0     # a bar quieter than this is silent
ACTIVE_RANGE_DB = 30.0      # a bar is active within this of the role's loud level (sections.STEM_FLOOR_DB)


def si_sdr_from_stats(ss, se, ee):
    """SI-SDR in dB from the sums <ref,ref>, <est,ref>, <est,est>. NaN if the reference
    is silent. The scaled reference is alpha*ref with alpha = <est,ref>/<ref,ref>; the
    noise is est minus it; its energy is <est,est> - <est,ref>^2/<ref,ref>. Capped at
    ~120 dB when the noise vanishes (est == ref)."""
    ss, se, ee = np.asarray(ss, float), np.asarray(se, float), np.asarray(ee, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = se * se / ss
        noise = np.maximum(ee - target, ss * 1e-12)
        out = 10.0 * np.log10(target / noise + 1e-30)
    return np.where(ss > 0, out, np.nan)


def si_sdr(est, ref):
    """Scale-invariant SDR of `est` against `ref`, over all samples of both."""
    est, ref = np.asarray(est, float).ravel(), np.asarray(ref, float).ravel()
    return float(si_sdr_from_stats((ref * ref).sum(), (est * ref).sum(), (est * est).sum()))


def bar_edges(n_bars, bar_s, sr, bar1_s=0.0):
    """Sample index of each bar line, n_bars + 1 of them."""
    return np.round((bar1_s + np.arange(n_bars + 1) * bar_s) * sr).astype(np.int64)


def energy_db(ms):
    """Mean-square per bar to dB, with the floor sections._bar_db uses (silence is -120)."""
    return 10.0 * np.log10(np.asarray(ms, float) + 1e-12)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def active_bars(db):
    """Bars where a role plays: above the silence floor and within ACTIVE_RANGE_DB of the
    role's own loud level (95th percentile of its non-silent bars)."""
    db = np.asarray(db, float)
    audible = db > ACTIVE_FLOOR_DB
    if not audible.any():
        return audible
    top = np.percentile(db[audible], 95)
    return audible & (db > top - ACTIVE_RANGE_DB)


def bar_stats(est_paths, ref_paths, edges):
    """One streamed pass over a separated stem and a true role (a sum of ref_paths).
    Per bar: <ref,ref>, <est,ref>, <est,est> over all channels (for SI-SDR) and the
    mean-square of each side's mono mix (for energy), so memory stays at one block.
    Samples past the shorter side are ignored."""
    import soundfile as sf
    est_f = [sf.SoundFile(p) for p in est_paths]
    ref_f = [sf.SoundFile(p) for p in ref_paths]
    try:
        n_bars = len(edges) - 1
        total = min(f.frames for f in est_f + ref_f)
        acc = {k: np.zeros(n_bars) for k in ("ss", "se", "ee", "ms_ref", "ms_est")}
        off = 0
        while off < total:
            n = min(TRUTH_BLOCK, total - off)
            e = sum(f.read(n, dtype="float64", always_2d=True) for f in est_f)
            r = sum(f.read(n, dtype="float64", always_2d=True) for f in ref_f)
            bar = np.searchsorted(edges, np.arange(off, off + n), side="right") - 1
            ok = (bar >= 0) & (bar < n_bars)
            for key, v in (("ss", (r * r).sum(1)), ("se", (e * r).sum(1)), ("ee", (e * e).sum(1)),
                           ("ms_ref", r.mean(1) ** 2), ("ms_est", e.mean(1) ** 2)):
                acc[key] += np.bincount(bar[ok], weights=v[ok], minlength=n_bars)
            off += n
        n_samp = np.maximum(np.diff(edges), 1)
        acc["ms_ref"] /= n_samp
        acc["ms_est"] /= n_samp
        acc["frames_compared"] = total
        acc["frames"] = [f.frames for f in est_f + ref_f]
        return acc
    finally:
        for f in est_f + ref_f:
            f.close()


def band_energy(paths, edges, sr, bands=BLEED_BANDS):
    """Per bar and band, mean energy of the mono mix of `paths` (summed), (n_bars, n_bands).
    Non-overlapping Hann frames of FRAME samples, |rfft|^2 summed inside each band, the
    frame filed under the bar holding its centre. Scale is arbitrary but identical for
    every signal, which is all the bleed fit needs."""
    import soundfile as sf
    files = [sf.SoundFile(p) for p in paths]
    try:
        n_bars = len(edges) - 1
        total = min(f.frames for f in files)
        win = np.hanning(FRAME)
        bins = np.fft.rfftfreq(FRAME, 1.0 / sr)
        masks = [(bins >= lo) & (bins < hi) for lo, hi in zip(bands[:-1], bands[1:])]
        out, count = np.zeros((n_bars, len(masks))), np.zeros(n_bars)
        off = 0
        while off < total:
            n = min(TRUTH_BLOCK, total - off)
            x = sum(f.read(n, dtype="float64", always_2d=True) for f in files).mean(1)
            x = np.pad(x, (0, -n % FRAME))
            spec = np.abs(np.fft.rfft(x.reshape(-1, FRAME) * win, axis=1)) ** 2
            centre = off + np.arange(spec.shape[0]) * FRAME + FRAME // 2
            bar = np.searchsorted(edges, centre, side="right") - 1
            ok = (bar >= 0) & (bar < n_bars)
            for j, m in enumerate(masks):
                out[:, j] += np.bincount(bar[ok], weights=spec[ok][:, m].sum(1), minlength=n_bars)
            count += np.bincount(bar[ok], minlength=n_bars)
            off += n
        return out / np.maximum(count, 1)[:, None]
    finally:
        for f in files:
            f.close()


def bleed_shares(stem_e, role_e):
    """Where a separated stem's energy came from, as shares by true role.

    stem_e is the stem's (segments, bands) energy, one row per beat; role_e maps role
    -> the same for the true role. In each band separately, non-negative least squares
    of the stem's energy on the true roles' energy, one equation per beat: weight
    w[r, band] says how much of role r's energy in that band the stem kept (1.0 = all
    of it, 0 = none). A weight per band, not one per role, because a stem keeps a
    role's low end and drops its highs, or the reverse. Each band's equations are
    scaled by that band's mean stem energy so every band fits equally well.
    share[r] = sum over bands of w * energy of role r, divided by the stem's energy.
    Energies add only for uncorrelated signals, so this is an estimate, not a
    decomposition. remainder = 1 - sum of shares: energy the true roles do not explain
    (negative: the fit over-explains).
    Returns ({role: share}, remainder)."""
    from scipy.optimize import nnls
    names = list(role_e)
    stem_e = np.asarray(stem_e, float)
    total = stem_e.sum()
    if total <= 0:
        return {r: float("nan") for r in names}, float("nan")
    got = np.zeros(len(names))
    for j in range(stem_e.shape[1]):
        A = np.stack([np.asarray(role_e[r], float)[:, j] for r in names], axis=1)
        y = stem_e[:, j]
        scale = 1.0 / y.mean() if y.mean() > 0 else 0.0
        w, _ = nnls(A * scale, y * scale)
        got += w * A.sum(0)
    shares = {r: float(got[i] / total) for i, r in enumerate(names)}
    return shares, 1.0 - sum(shares.values())


def score_stem(est_paths, ref_paths, edges):
    """Every per-stem number the truth mode reports, for one stem against one truth."""
    st = bar_stats(est_paths, ref_paths, edges)
    ref_db, est_db = energy_db(st["ms_ref"]), energy_db(st["ms_est"])
    act = active_bars(ref_db)
    bar_sdr = si_sdr_from_stats(st["ss"], st["se"], st["ee"])
    ss, se, ee = (st[k].sum() for k in ("ss", "se", "ee"))
    row = {
        "si_sdr_db": float(si_sdr_from_stats(ss, se, ee)),
        "si_sdr_active_median_db": float(np.nanmedian(bar_sdr[act])) if act.any() else float("nan"),
        "level_gap_db": float(10 * np.log10((ee + 1e-30) / (ss + 1e-30))) if ss > 0 else float("nan"),
        "level_gap_active_median_db": float(np.median((est_db - ref_db)[act])) if act.any() else float("nan"),
        "corr_db": pearson(est_db, ref_db),
        "corr_linear": pearson(st["ms_est"], st["ms_ref"]),
        "corr_db_active": pearson(est_db[act], ref_db[act]) if act.sum() > 1 else float("nan"),
        "bars_active_true": int(act.sum()),
        "bars_active_agree": int((active_bars(est_db) == act).sum()),
        "bars": len(act),
        "frames": st["frames"],
    }
    return row, est_db, ref_db


def find_audio(folder, name):
    for ext in (".flac", ".wav"):
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    return None


def fmt(v, spec="%.1f"):
    return "-" if v is None or v != v else spec % v


def truth_main(args):
    manifest = json.load(open(args.truth))
    out_dir = os.path.expanduser(manifest["out_dir"])
    truth_dir = os.path.join(out_dir, "truth")
    stems_dir = args.stems
    if not stems_dir:
        found = sorted(d for d in (os.path.join(out_dir, "separated", n) for n in
                                   os.listdir(os.path.join(out_dir, "separated")))
                       if d.endswith("_stems") and os.path.isdir(d))
        if not found:
            sys.exit("no *_stems folder in %s; pass --stems" % os.path.join(out_dir, "separated"))
        stems_dir = found[0]
    import soundfile as sf
    sr = sf.info(os.path.join(out_dir, "mix.wav")).samplerate
    bar_s = manifest["beats_per_bar"] * 60.0 / manifest["bpm"]
    edges = bar_edges(manifest["bars"], bar_s, sr, manifest.get("bar1_s", 0.0))

    true_paths = {r: find_audio(truth_dir, r) for r in manifest["stems"] + ["returns"]}
    true_paths = {r: p for r, p in true_paths.items() if p}
    print("truth: %s\nstems: %s" % (truth_dir, stems_dir))

    scored = []   # (label, stem, truth roles)
    for stem in manifest["stems"]:
        scored.append((stem, stem, [stem] if stem in true_paths else []))
    for label, rd in manifest.get("readings", {}).items():
        scored.append((label, rd["stem"], [r for r in rd["truth"] if r in true_paths]))

    rows = {}
    for label, stem, roles in scored:
        est = find_audio(stems_dir, stem)
        if not est or not roles:
            why = "no separated stem" if not est else "no true role"
            rows[label] = {"stem": stem, "truth": roles, "skipped": why}
            continue
        print("  scoring %s vs %s" % (stem, "+".join(roles)))
        row, _, _ = score_stem([est], [true_paths[r] for r in roles], edges)
        row.update(stem=stem, truth=roles)
        rows[label] = row

    # Bleed: each separated stem's band energies against every true role's, per beat.
    per_bar = manifest["beats_per_bar"]
    beat_edges = bar_edges(manifest["bars"] * per_bar, bar_s / per_bar, sr, manifest.get("bar1_s", 0.0))
    role_e = {r: band_energy([p], beat_edges, sr) for r, p in true_paths.items()}
    bleed = {}
    for stem in manifest["stems"]:
        est = find_audio(stems_dir, stem)
        if not est:
            continue
        shares, rest = bleed_shares(band_energy([est], beat_edges, sr), role_e)
        bleed[stem] = {"shares": shares, "remainder": rest}

    print()
    print("%-16s %-14s %8s %9s %8s %8s %7s %7s %7s %9s"
          % ("stem", "true role", "SI-SDR", "SDR act", "gap", "gap act", "corr dB", "corr lin",
             "corr act", "act/agree"))
    for label, r in rows.items():
        if "skipped" in r:
            print("%-16s %-14s  (%s)" % (label, "+".join(r["truth"]) or "-", r["skipped"]))
            continue
        print("%-16s %-14s %8s %9s %8s %8s %7s %7s %7s %9s"
              % (label, "+".join(r["truth"])[:14], fmt(r["si_sdr_db"]), fmt(r["si_sdr_active_median_db"]),
                 fmt(r["level_gap_db"]), fmt(r["level_gap_active_median_db"]), fmt(r["corr_db"], "%.2f"),
                 fmt(r["corr_linear"], "%.2f"), fmt(r["corr_db_active"], "%.2f"),
                 "%d/%d" % (r["bars_active_true"], r["bars_active_agree"])))
    roles = list(true_paths)
    print("\nBLEED  (share of each separated stem's energy fitted to each true role, %%; per beat, bands %s Hz)"
          % "/".join(str(b) for b in BLEED_BANDS))
    print("%-14s" % "stem" + "".join("%9s" % r.replace("drums_", "")[:8] for r in roles) + "%13s" % "unexplained")
    for stem, b in bleed.items():
        print("%-14s" % stem + "".join("%9s" % fmt(100 * b["shares"][r], "%.1f") for r in roles)
              + "%13s" % fmt(100 * b["remainder"], "%.1f"))

    out = args.json or os.path.join(out_dir, "baseline.json")
    with open(out, "w") as fh:
        json.dump({"stems_dir": stems_dir, "rows": rows, "bleed": bleed}, fh, indent=1)
    print("\nwrote %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", nargs="?")
    ap.add_argument("--truth", metavar="MANIFEST",
                    help="score separated stems against true stems (see the docstring)")
    ap.add_argument("--stems", help="--truth: the <name>_stems folder (default: the first "
                                    "*_stems in the manifest's out_dir/separated)")
    ap.add_argument("--json", help="--truth: write the numbers here (default: out_dir/baseline.json)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--samples", type=int, default=5,
                    help="windows per track (default 5)")
    ap.add_argument("--excerpt", type=int, default=60, help="seconds (default 60)")
    ap.add_argument("--overlap", default="0.5",
                    help="demucs --overlap (default 0.5; demucs's own default is 0.25)")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()
    if args.truth:
        return truth_main(args)
    if not args.corpus_dir:
        ap.error("corpus_dir is required without --truth")

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

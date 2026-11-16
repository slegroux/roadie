#!/usr/bin/env python3
"""The separation benchmark's mitigations (studies/06 item 4), each a way to make a set of separated stems
better, scored by benchmark.py --truth on the folder it writes.

    bench_mitigations.py residual MANIFEST STEMS [--drums full|subs]   (b) mix minus the stems: level, what it holds
    bench_mitigations.py wiener   MANIFEST STEMS OUT                    (c) soft-mask re-mask of the stems over the mix
    bench_mitigations.py gate     MANIFEST STEMS OUT                    (d) silence the hits mark_bleed marks as bleed
    bench_mitigations.py assemble MANIFEST NAME                         (e) name another model's outputs as stems
    bench_mitigations.py compare  SCORES.json [SCORES.json ...]         SI-SDR per stem, side by side

(a), averaged hits, is `bench_roles.py --average`. STEMS is a folder of the names benchmark.py reads
(bass, drums, synths, vocals, drums_kick ... drums_crash). Everything written goes to a new folder
under the manifest's out_dir; the mix and the truth are read only.

(c) is a plain power soft mask in numpy (norbert is not installed): per channel and STFT bin, stem k gets
|S_k|^p / sum_j |S_j|^p (p = 2; --power 1 is the magnitude mask) of the mix's STFT (Hann 4096, hop 1024), so the re-masked stems sum to the mix. It is the
first, non-iterative step of a Wiener filter with the stems themselves as the variance estimates.
(d) zeroes, in each drum stem, the cut of every hit that belongs to a group mark_bleed marked, with 2 ms fades;
--coincidence-only also zeroes the groups that pass its coincidence test alone (a counterfactual, not the module's rule).
"""
import argparse
import json
import os
import shutil
import sys

BLOCK = 1 << 19
LEAVES = ("drums_kick", "drums_snare", "drums_toms", "drums_hh", "drums_ride", "drums_crash", "bass", "synths", "vocals")
DRUM_SUBS = LEAVES[:6]
NFFT = 4096
HOP = NFFT // 4
SEGMENT = 1 << 20          # samples per re-mask segment
POWER = 2.0
FADE_S = 0.002

# (e): how another model's output files become benchmark.py stems. {source file glob (inside the folder): stem}.
# A stem built from several files is their sum (as yt2stems.sh sums other+piano+guitar into synths).
ASSEMBLE = {
    "bsroformer_sw": {"into": "bsroformer_sw/stems", "files": {
        "bass": ["mix_(bass)_BS-Roformer-SW.flac"], "drums": ["mix_(drums)_BS-Roformer-SW.flac"],
        "synths": ["mix_(other)_BS-Roformer-SW.flac", "mix_(piano)_BS-Roformer-SW.flac", "mix_(guitar)_BS-Roformer-SW.flac"],
        "vocals": ["mix_(vocals)_BS-Roformer-SW.flac"],
        **{"drums_" + k: ["drumsep/mix_(drums)_BS-Roformer-SW_(%s)_MDX23C-DrumSep-aufr33-jarredou.flac" % k]
           for k in ("kick", "snare", "toms", "hh", "ride", "crash")}}, "base": "bsroformer_sw"},
    "melband_inst_v2": {"into": "melband_inst_v2/stems", "files": {
        "vocals": ["mix_(Vocals)_melband_roformer_inst_v2.flac"],
        "instrumental": ["mix_(Instrumental)_melband_roformer_inst_v2.flac"]}, "base": "melband_inst_v2"},
    "kuielab_b_drums": {"into": "kuielab_b_drums/stems", "files": {
        "drums": ["mix_(Drums)_kuielab_b_drums.flac"], "no_drums": ["mix_(No Drums)_kuielab_b_drums.flac"]},
        "base": "kuielab_b_drums"},
    "bsroformer_1053": {"into": "bsroformer_1053/stems", "files": {
        "drum_bass": ["mix_(Drum-Bass)_model_bs_roformer_ep_937_sdr_10.flac"],
        "no_drum_bass": ["mix_(No Drum-Bass)_model_bs_roformer_ep_937_sdr_10.flac"]}, "base": "bsroformer_1053"},
}


def db(x):
    import numpy as np
    return 20.0 * np.log10(max(float(x), 1e-12))


def audio(folder, name):
    from benchmark import find_audio
    return find_audio(folder, name)


def write_sum(paths, out, sr_expected=None, subtype="PCM_24"):
    """Sum of files (as float64), written to out, one block at a time; the shortest file sets the length."""
    import soundfile as sf
    files = [sf.SoundFile(p) for p in paths]
    try:
        sr, ch, total = files[0].samplerate, files[0].channels, min(f.frames for f in files)
        with sf.SoundFile(out, "w", samplerate=sr, channels=ch, subtype=subtype, format="FLAC") as w:
            done = 0
            while done < total:
                n = min(BLOCK, total - done)
                w.write(sum(f.read(n, dtype="float64", always_2d=True) for f in files))
                done += n
        return sr, total
    finally:
        for f in files:
            f.close()


def residual(mix_path, stem_paths, out_path):
    """mix minus the sum of the stems, written to out_path. Returns {"peak_dbfs", "rms_db_re_mix", "frames", "frame_gaps"}:
    the residual's peak (dBFS) and RMS relative to the mix's RMS, over the shortest file; frame_gaps is how far each
    stem's length is from the mix's (frames), since a separator may add or drop a few samples."""
    import numpy as np
    import soundfile as sf
    mix = sf.SoundFile(mix_path)
    files = [sf.SoundFile(p) for p in stem_paths]
    try:
        total = min([mix.frames] + [f.frames for f in files])
        peak, ss, ms, count = 0.0, 0.0, 0.0, 0
        with sf.SoundFile(out_path, "w", samplerate=mix.samplerate, channels=mix.channels, subtype="PCM_24", format="FLAC") as w:
            done = 0
            while done < total:
                n = min(BLOCK, total - done)
                m = mix.read(n, dtype="float64", always_2d=True)
                r = m - sum(f.read(n, dtype="float64", always_2d=True) for f in files)
                w.write(np.clip(r, -1.0, 1.0))
                peak = max(peak, float(np.abs(r).max()))
                ss += float((r ** 2).sum())
                ms += float((m ** 2).sum())
                count += r.size
                done += n
        return {"peak_dbfs": db(peak), "rms_db_re_mix": db(np.sqrt(ss / count) / np.sqrt(ms / count)),
                "frames": total, "frame_gaps": [f.frames - mix.frames for f in files]}
    finally:
        mix.close()
        for f in files:
            f.close()


def wiener_segment(x, stems, power=POWER, nfft=NFFT, eps=1e-12):
    """Re-mask: x [samples, ch] is the mix, stems a list of [samples, ch] arrays of one length. Returns a list of
    arrays: stem k's share |S_k|^p / sum_j |S_j|^p of the mix's STFT, per channel and bin, back to samples. The
    shares sum to 1 in every bin, so the outputs sum to x (to float rounding)."""
    import numpy as np
    from scipy.signal import istft, stft
    n = len(x)

    def spec(a):
        return stft(a.T, nperseg=nfft, noverlap=nfft - nfft // 4, window="hann", boundary="zeros", padded=True)[2]

    X = spec(x)
    mags = [np.abs(spec(s)) ** power for s in stems]
    total = sum(mags) + eps * len(stems)
    out = []
    for m in mags:
        y = istft(X * ((m + eps) / total), nperseg=nfft, noverlap=nfft - nfft // 4, window="hann", boundary=True)[1]
        out.append(y.T[:n])
    return out


def wiener(mix_path, stem_paths, out_dir, power=POWER):
    """Write the re-masked stems into out_dir (one FLAC each, the same names). Segments of SEGMENT samples with a
    margin of two FFT frames each side that is cut off after, so no segment edge reaches the output."""
    import numpy as np
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    mix = sf.SoundFile(mix_path)
    files = {k: sf.SoundFile(p) for k, p in stem_paths.items()}
    writers = {k: sf.SoundFile(os.path.join(out_dir, k + ".flac"), "w", samplerate=mix.samplerate, channels=mix.channels,
                               subtype="PCM_24", format="FLAC") for k in files}
    try:
        total = min([mix.frames] + [f.frames for f in files.values()])
        margin = 2 * NFFT
        peak = 0.0
        for a in range(0, total, SEGMENT):
            b = min(total, a + SEGMENT)
            lo, hi = max(0, a - margin), min(total, b + margin)
            for f in [mix] + list(files.values()):
                f.seek(lo)
            x = mix.read(hi - lo, dtype="float64", always_2d=True)
            segs = wiener_segment(x, [f.read(hi - lo, dtype="float64", always_2d=True) for f in files.values()], power)
            for (k, w), y in zip(writers.items(), segs):
                y = y[a - lo:b - lo]
                peak = max(peak, float(np.abs(y).max()))
                w.write(np.clip(y, -1.0, 1.0))
        return {"frames": total, "peak_dbfs": db(peak)}
    finally:
        mix.close()
        for f in files.values():
            f.close()
        for w in writers.values():
            w.close()


def gate_stem(x, sr, result, groups, fade_s=FADE_S):
    """x [samples, ch] of a drum stem with the cut of every hit of `groups` (ids; measured and assigned hits) silenced:
    from 2 ms before its onset to its cut's end, with `fade_s` fades at both edges. `result` is the stem's hit_groups
    result (its hits carry the stem's sample positions). Returns (gated copy, number of hits gated)."""
    import numpy as np
    y = np.array(x, dtype=np.float64)
    gain = np.ones(len(y))
    f = max(1, int(round(fade_s * sr)))
    n = 0
    ids = set(groups)
    for h in result["hits"]:
        if h["group"] in ids and h["status"] in ("grouped", "assigned"):
            a, b = max(0, h["start"] - f), min(len(y), h["end"] + f)
            if b - a <= 2 * f:
                continue
            g = np.zeros(b - a)
            g[:f] = np.linspace(1.0, 0.0, f, endpoint=False)         # 1 -> 0 over f samples, silent, 0 -> 1 over f
            g[-f:] = np.linspace(0.0, 1.0, f, endpoint=False)
            gain[a:b] = np.minimum(gain[a:b], g)
            n += 1
    return y * gain[:, None], n


# ---------------------------------------------------------------- commands

def load_manifest(path):
    m = json.load(open(path))
    m["out_dir"] = os.path.expanduser(m["out_dir"])
    return m


def cmd_residual(a):
    import soundfile as sf
    from benchmark import band_energy, bleed_shares, role_energies, true_role_paths
    m = load_manifest(a.manifest)
    mix = os.path.join(m["out_dir"], "mix.wav")
    names = ["bass", "synths", "vocals"] + (["drums"] if a.drums == "full" else list(DRUM_SUBS))
    paths = {}
    for n in names:
        p = audio(a.stems, n)
        if p:
            paths[n] = p
    out_dir = os.path.join(m["out_dir"], "residual")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "%s_%s.flac" % (os.path.basename(os.path.normpath(os.path.dirname(a.stems) or a.stems)), a.drums))
    r = residual(mix, list(paths.values()), out)
    print("residual = mix - (%s)" % " + ".join(paths))
    print("  peak %.2f dBFS, RMS %.2f dB re the mix, %d frames; stem lengths differ from the mix by %s" % (
        r["peak_dbfs"], r["rms_db_re_mix"], r["frames"], r["frame_gaps"]))
    sr = sf.info(mix).samplerate
    true_paths = true_role_paths(m, m["out_dir"])
    beat_edges, role_e = role_energies(m, true_paths, sr)
    shares, rest, fit = bleed_shares(band_energy([out], beat_edges, sr), role_e)
    print("  what it holds (share of its energy fitted to each true role, %%; R2 %.2f, net remainder %.1f%%):" % (fit["r2"], 100 * rest))
    print("  " + "  ".join("%s %.1f" % (k.replace("drums_", ""), 100 * v) for k, v in shares.items()))
    res = {"stems": list(paths), "file": out, **r, "shares": shares, "remainder": rest, "r2": fit["r2"], "at_bound": fit["at_bound"]}
    with open(out[:-5] + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote %s" % out)


def cmd_wiener(a):
    m = load_manifest(a.manifest)
    mix = os.path.join(m["out_dir"], "mix.wav")
    paths = {n: audio(a.stems, n) for n in LEAVES}
    paths = {k: v for k, v in paths.items() if v}
    print("re-masking %d stems over the mix: %s" % (len(paths), ", ".join(paths)))
    out = os.path.expanduser(a.out)
    r = wiener(mix, paths, out, a.power)
    # drums = the sum of the six drum parts, so the full-drums reading can be scored too
    subs = [os.path.join(out, k + ".flac") for k in DRUM_SUBS if k in paths]
    if subs:
        write_sum(subs, os.path.join(out, "drums.flac"))
    print("wrote %s (%d frames, peak %.2f dBFS)" % (out, r["frames"], r["peak_dbfs"]))


def cmd_gate(a):
    import soundfile as sf
    import bench_roles
    from roadie_analysis.drumroles import load
    m = load_manifest(a.manifest)
    cache = os.path.join(m["out_dir"], "roles")
    paths = {k: audio(a.stems, k) for k in DRUM_SUBS}
    paths = {k: v for k, v in paths.items() if v}
    results, named, bleed = bench_roles.run_set(a.set, paths, m["bpm"], m.get("bar1_s", 0.0), cache, 1, False)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    report = {}
    for stem, p in paths.items():
        marked = sorted(g for g, b in bleed.get(stem, {}).items() if b["bleed"])
        if a.coincidence_only:       # not the module's rule: every group of 4+ hits that coincides 70 %+, at p <= 1 %, with another stem's group
            marked = sorted(set(marked) | {g["id"] for g in results[stem]["groups"] if g["count"] >= 4 and bleed[stem][g["id"]]["evidence"]
                                           and bleed[stem][g["id"]]["evidence"]["rate"] >= 0.7 and bleed[stem][g["id"]]["evidence"]["p"] <= 0.01})
        x, sr = sf.read(p, dtype="float64", always_2d=True)
        # hit_groups ran on the stem as load() read it, DC removed; gate the stem as read, so its sample positions agree
        y, n = gate_stem(x, sr, results[stem], marked) if marked else (x, 0)
        sf.write(os.path.join(out, stem + ".flac"), y, sr, subtype="PCM_24", format="FLAC")
        near = []        # unmarked groups that coincide with another stem's group at 70 % or more: which condition they failed
        for g in results[stem]["groups"]:
            ev = bleed[stem][g["id"]]["evidence"]
            if g["id"] not in marked and ev and g["count"] >= 4 and ev["rate"] >= 0.7:
                near.append({"group": g["id"], "hits": g["count"], **{k: ev.get(k) for k in (
                    "source_stem", "source_group", "rate", "expected", "p", "louder_db", "louder_share", "distance", "failed")}})
        report[stem] = {"groups": marked, "hits_gated": n, "hits": len(results[stem]["hits"]),
                        "why": {g: bleed[stem][g]["why"] for g in marked}, "coincident_but_unmarked": near}
        print("  %-12s groups marked %s, %d hits gated of %d; %d groups of 4+ hits coincide 70%%+ with another stem's and failed: %s" % (
            stem, marked or "-", n, len(results[stem]["hits"]), len(near), sorted({x["failed"] for x in near}) or "-"))
    with open(os.path.join(out, "gating.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("wrote %s" % out)


def cmd_assemble(a):
    m = load_manifest(a.manifest)
    spec = ASSEMBLE[a.name]
    base = os.path.join(m["out_dir"], spec["base"])
    into = os.path.join(m["out_dir"], spec["into"])
    os.makedirs(into, exist_ok=True)
    for stem, files in spec["files"].items():
        paths = [os.path.join(base, f) for f in files]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print("  %-12s missing %s" % (stem, missing))
            continue
        if len(paths) == 1:
            shutil.copyfile(paths[0], os.path.join(into, stem + ".flac"))
        else:
            write_sum(paths, os.path.join(into, stem + ".flac"))
        print("  %-12s from %d file(s)" % (stem, len(paths)))
    print("wrote %s" % into)


def cmd_compare(a):
    rows = [json.load(open(p))["rows"] for p in a.scores]
    names = [os.path.basename(p)[:-5].replace("scores_", "") for p in a.scores]
    labels = []
    for r in rows:
        labels += [k for k in r if k not in labels]
    print("%-18s" % "SI-SDR dB" + "".join("%22s" % n[-21:] for n in names))
    for k in labels:
        cells = []
        for r in rows:
            v = r.get(k)
            cells.append("-" if not v or "skipped" in v else "<-30" if v["si_sdr_db"] < -30 else "%.1f (rho %.2f)" % (v["si_sdr_db"], v["rho"]))
        print("%-18s" % k[:18] + "".join("%22s" % c for c in cells))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("residual")
    p.add_argument("manifest"), p.add_argument("stems")
    p.add_argument("--drums", choices=("full", "subs"), default="full")
    p.set_defaults(fn=cmd_residual)
    p = sub.add_parser("wiener")
    p.add_argument("manifest"), p.add_argument("stems"), p.add_argument("out")
    p.add_argument("--power", type=float, default=POWER, help="the mask's exponent: 2 is the power (Wiener) mask, 1 the magnitude mask")
    p.set_defaults(fn=cmd_wiener)
    p = sub.add_parser("gate")
    p.add_argument("manifest"), p.add_argument("stems"), p.add_argument("out")
    p.add_argument("--set", default="separated", help="the bench_roles cache name of this stems folder (default separated)")
    p.add_argument("--coincidence-only", action="store_true",
                   help="not mark_bleed's rule: also gate every group of 4+ hits that coincides 70 %%+ (p <= 1 %%) with another stem's group, "
                        "level and likeness unchecked; a counterfactual for what mark_bleed's two other conditions are worth")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("assemble")
    p.add_argument("manifest"), p.add_argument("name", choices=sorted(ASSEMBLE))
    p.set_defaults(fn=cmd_assemble)
    p = sub.add_parser("compare")
    p.add_argument("scores", nargs="+")
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

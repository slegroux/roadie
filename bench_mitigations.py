#!/usr/bin/env python3
"""The separation benchmark's mitigations (studies/06 item 4), each a way to make a set of separated stems
better, scored by benchmark.py --truth on the folder it writes.

    bench_mitigations.py residual MANIFEST STEMS [--drums full|subs]   (b) mix minus the stems: level, what it holds
    bench_mitigations.py wiener   MANIFEST STEMS OUT                    (c) soft-mask re-mask of the stems over the mix
    bench_mitigations.py oracle   MANIFEST OUT                          (c) the same mask built from the TRUE stems: its ceiling
    bench_mitigations.py gate     MANIFEST STEMS OUT                    (d) silence the hits mark_bleed marks as bleed
    bench_mitigations.py gate     MANIFEST STEMS OUT --oracle           (d) hit-level oracle (the detector's cuts, perfect labels): silence every hit with no true onset of the stem's own sources
    bench_mitigations.py oracle-score MANIFEST [--power P]              (c) the oracle mask scored in memory, with and without `returns`: no audio written
    bench_mitigations.py bars     MANIFEST STEMS_A STEMS_B              (e) the share of active bars where B's SI-SDR beats A's, per stem
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
--oracle is a hit-level oracle: the detector's cuts with perfect labels. It zeroes every cut of a hit that has no true onset of its
stem's own sources (the manifest's drum sources with that role) within the roles' tolerance (25 ms). It is a heuristic, not an
upper bound: a cut that holds a true hit and a false one (or a true hit with a late tail) goes whole, and the kick loses from that.
It never touches a stem with no own source (the ride).
`oracle` is `wiener` with the true role stems (truth/, `returns` included, so the masks sum to the mix) as the variance estimates,
all in one mask over the mix: what the mask gives when it is told the right answer, with the mix's phase. `oracle-score` scores
the same mask in memory, and again without `returns` among the variance estimates (the real re-mask has no returns stem: the
returns' energy is then shared out among the stems), so the two rows are like for like with the stems of the re-mask.
"""
import argparse
import json
import os
import shutil

BLOCK = 1 << 19
LEAVES = ("drums_kick", "drums_snare", "drums_toms", "drums_hh", "drums_ride", "drums_crash", "bass", "synths", "vocals", "returns")
DRUM_SUBS = LEAVES[:6]
NFFT = 4096
HOP = NFFT // 4
SEGMENT = 1 << 20          # samples per re-mask segment
POWER = 2.0
FADE_S = 0.002
TOL_S = 0.025            # a hit has a true onset when one of its stem's own sources has one this near (bench_roles.TOL_S)

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
    """Sum of files (as float64), written to out, one block at a time; the shortest file sets the length. FLAC cannot hold
    a sample at or over full scale, so a sum that reaches 1.0 raises ValueError (out is removed): nothing is clipped."""
    import numpy as np
    import soundfile as sf
    files = [sf.SoundFile(p) for p in paths]
    try:
        sr, ch, total = files[0].samplerate, files[0].channels, min(f.frames for f in files)
        peak = 0.0
        with sf.SoundFile(out, "w", samplerate=sr, channels=ch, subtype=subtype, format="FLAC") as w:
            done = 0
            while done < total:
                n = min(BLOCK, total - done)
                block = sum(f.read(n, dtype="float64", always_2d=True) for f in files)
                peak = max(peak, float(np.abs(block).max()))
                w.write(block)
                done += n
        if peak >= 1.0:
            os.remove(out)
            raise ValueError("the sum of %d files peaks at %.3f (full scale 1.0): writing it as %s would clip" % (len(paths), peak, subtype))
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


def wiener_segments(mix, files, power=POWER):
    """The re-masked stems, one segment at a time: yields (a, b, {name: array [b - a, ch]}) for the samples a..b, over the
    open SoundFiles `mix` and `files` ({name: SoundFile}). Segments of SEGMENT samples with a margin of two FFT frames each side
    that is cut off after, so no segment edge reaches the output."""
    total = min([mix.frames] + [f.frames for f in files.values()])
    margin = 2 * NFFT
    for a in range(0, total, SEGMENT):
        b = min(total, a + SEGMENT)
        lo, hi = max(0, a - margin), min(total, b + margin)
        for f in [mix] + list(files.values()):
            f.seek(lo)
        x = mix.read(hi - lo, dtype="float64", always_2d=True)
        segs = wiener_segment(x, [f.read(hi - lo, dtype="float64", always_2d=True) for f in files.values()], power)
        yield a, b, {k: y[a - lo:b - lo] for k, y in zip(files, segs)}


def wiener(mix_path, stem_paths, out_dir, power=POWER):
    """Write the re-masked stems into out_dir (one FLAC each, the same names)."""
    import numpy as np
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    mix = sf.SoundFile(mix_path)
    files = {k: sf.SoundFile(p) for k, p in stem_paths.items()}
    writers = {k: sf.SoundFile(os.path.join(out_dir, k + ".flac"), "w", samplerate=mix.samplerate, channels=mix.channels,
                               subtype="PCM_24", format="FLAC") for k in files}
    try:
        total = min([mix.frames] + [f.frames for f in files.values()])
        peak = 0.0
        for a, b, ys in wiener_segments(mix, files, power):
            for k, y in ys.items():
                peak = max(peak, float(np.abs(y).max()))
                writers[k].write(np.clip(y, -1.0, 1.0))
        return {"frames": total, "peak_dbfs": db(peak)}
    finally:
        mix.close()
        for f in files.values():
            f.close()
        for w in writers.values():
            w.close()


def gate_hits(x, sr, hits, fade_s=FADE_S):
    """x [samples, ch] of a drum stem with the cut of every hit in `hits` (hit dicts with "start" and "end", samples of the
    stem) silenced: from `fade_s` before its onset to its cut's end, with fades of `fade_s` at both edges. Returns (gated
    copy, number of hits gated)."""
    import numpy as np
    y = np.array(x, dtype=np.float64)
    gain = np.ones(len(y))
    f = max(1, int(round(fade_s * sr)))
    n = 0
    for h in hits:
        a, b = max(0, h["start"] - f), min(len(y), h["end"] + f)
        if b - a <= 2 * f:
            continue
        g = np.zeros(b - a)
        g[:f] = np.linspace(1.0, 0.0, f, endpoint=False)         # 1 -> 0 over f samples, silent, 0 -> 1 over f
        g[-f:] = np.linspace(0.0, 1.0, f, endpoint=False)
        gain[a:b] = np.minimum(gain[a:b], g)
        n += 1
    return y * gain[:, None], n


def gate_stem(x, sr, result, groups, fade_s=FADE_S):
    """gate_hits on the measured and assigned hits of `groups` (ids) of a stem's hit_groups `result` (its hits carry the
    stem's sample positions)."""
    ids = set(groups)
    return gate_hits(x, sr, [h for h in result["hits"] if h["group"] in ids and h["status"] in ("grouped", "assigned")], fade_s)


def hits_without_true_onset(result, onsets, tol=TOL_S):
    """The hits of a stem's hit_groups `result`, whatever their status, with no true onset (seconds, `onsets`) within `tol`:
    the ones a gate that knew the truth would silence."""
    import numpy as np
    ons = np.asarray(sorted(onsets), dtype=float)
    if not len(ons):
        return list(result["hits"])
    out = []
    for h in result["hits"]:
        i = np.searchsorted(ons, h["onset_s"])
        near = min(abs(h["onset_s"] - ons[max(i - 1, 0)]), abs(h["onset_s"] - ons[min(i, len(ons) - 1)]))
        if near > tol:
            out.append(h)
    return out


def coincidence_groups(result, bleed, minimum=4, rate=0.7, p=0.01):
    """The ids of a stem's groups of at least `minimum` hits that coincide with another stem's group at `rate` or more, at
    a chance of `p` or less: the coincidence test of mark_bleed alone, its level and likeness conditions left out. `bleed`
    is the stem's mark_bleed entry ({group id: {"evidence": ... or None}})."""
    return {g["id"] for g in result["groups"]
            if g["count"] >= minimum and bleed[g["id"]]["evidence"]
            and bleed[g["id"]]["evidence"]["rate"] >= rate and bleed[g["id"]]["evidence"]["p"] <= p}


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
    if a.add_back:       # the residual added back to one stem, to score it: benchmark.py --truth MANIFEST --stems <this folder>
        back = os.path.join(out_dir, "plus_%s_%s" % (a.add_back, a.drums))
        os.makedirs(back, exist_ok=True)
        write_sum([paths[a.add_back], out], os.path.join(back, a.add_back + ".flac"))
        print("wrote %s: %s + the residual (score it with benchmark.py --truth %s --stems %s)" % (back, a.add_back, a.manifest, back))


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


def cmd_oracle(a):
    """The soft-mask re-mask with the true role stems as the variance estimates (what the mask gives when told the answer)."""
    m = load_manifest(a.manifest)
    a.stems = os.path.join(m["out_dir"], "truth")
    cmd_wiener(a)


def oracle_scores(mix_path, stem_paths, readings, power=POWER):
    """SI-SDR (dB, over the whole file, all channels pooled) of the soft mask built from `stem_paths` ({name: true stem file}),
    in memory: nothing is written, nor clipped or rounded to 24 bits as a file would be. `readings` is [(label, [mask stems
    summed], [truth files summed])]; a reading is the sum of those stems of the mask against the sum of those files."""
    import numpy as np
    import soundfile as sf
    from benchmark import si_sdr_from_stats
    mix = sf.SoundFile(mix_path)
    files = {k: sf.SoundFile(p) for k, p in stem_paths.items()}
    refs = [[sf.SoundFile(p) for p in paths] for _, _, paths in readings]
    acc = [np.zeros(3) for _ in readings]
    try:
        for a, b, ys in wiener_segments(mix, files, power):
            for k, (_, names, _) in enumerate(readings):
                e = sum(ys[n] for n in names)
                r = 0
                for f in refs[k]:
                    f.seek(a)
                    r = r + f.read(b - a, dtype="float64", always_2d=True)[:len(e)]
                n = min(len(e), len(r))
                e, r = e[:n], r[:n]
                acc[k] += ((r * r).sum(), (e * r).sum(), (e * e).sum())
        return {label: float(si_sdr_from_stats(*acc[k])) for k, (label, _, _) in enumerate(readings)}
    finally:
        mix.close()
        for f in list(files.values()) + [f for fs in refs for f in fs]:
            f.close()


def cmd_oracle_score(a):
    """The oracle mask, scored in memory with `returns` among the variance estimates and without it, per stem and reading the
    mask can give (a drum stem or role, bass, synths, `drums` = the sum of the drum parts)."""
    import benchmark as B
    m = load_manifest(a.manifest)
    truth = os.path.join(m["out_dir"], "truth")
    mix = os.path.join(m["out_dir"], "mix.wav")
    true_paths = B.true_role_paths(m, m["out_dir"])
    source_paths = {s["drum_source"]: B.find_audio(os.path.join(m["out_dir"], "truth_sources"), s["drum_source"])
                    for s in m["sources"] if s.get("drum_source")}
    avail = {k: audio(truth, k) for k in LEAVES}
    avail = {k: v for k, v in avail.items() if v}
    drums = [k for k in DRUM_SUBS if k in avail]
    readings = []
    for label, stems, names in B.scored_readings(m, true_paths):
        parts = [p for s in stems for p in ([s] if s in avail else drums if s == "drums" else [None])]
        refs = [B.resolve_truth(n, true_paths, source_paths) for n in names]
        if names and None not in parts and all(refs):
            readings.append((label, parts, refs))
    out = {}
    for tag, keep in (("with_returns", list(avail)), ("no_returns", [k for k in avail if k != "returns"])):
        out[tag] = oracle_scores(mix, {k: avail[k] for k in keep}, readings, a.power)
    print("%-18s %14s %14s" % ("SI-SDR dB, power %g" % a.power, "with returns", "no returns"))
    for label, _, _ in readings:
        print("%-18s %14.1f %14.1f" % (label[:18], out["with_returns"][label], out["no_returns"][label]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"power": a.power, **out}, fh, indent=1)
        print("wrote %s" % a.json)


def cmd_bars(a):
    """Per stem and reading, the share of its active bars (the true role's, benchmark.active_bars) where stem set B has the
    higher SI-SDR than stem set A, and the median difference (dB) over them."""
    import numpy as np
    import soundfile as sf
    import benchmark as B
    m = load_manifest(a.manifest)
    sr = sf.info(os.path.join(m["out_dir"], "mix.wav")).samplerate
    edges = B.bar_edges(m["bars"], m["beats_per_bar"] * 60.0 / m["bpm"], sr, m.get("bar1_s", 0.0))
    true_paths = B.true_role_paths(m, m["out_dir"])
    source_paths = {s["drum_source"]: B.find_audio(os.path.join(m["out_dir"], "truth_sources"), s["drum_source"])
                    for s in m["sources"] if s.get("drum_source")}
    print("%-18s %8s %8s %8s %8s" % ("stem", "active", "B > A", "share", "median B-A"))
    out = {}
    for label, stems, names in B.scored_readings(m, true_paths):
        est = [[B.find_audio(d, s) for s in stems] for d in (a.a, a.b)]
        refs = [B.resolve_truth(n, true_paths, source_paths) for n in names]
        if not names or not all(refs) or not all(all(e) for e in est):
            continue
        st = [B.bar_stats(e, refs, edges) for e in est]
        sdr = [B.si_sdr_from_stats(v["ss"], v["se"], v["ee"]) for v in st]
        act = B.active_bars(B.energy_db(st[0]["ms_ref"]))
        d = (sdr[1] - sdr[0])[act]
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        out[label] = {"active": int(len(d)), "b_better": int((d > 0).sum()), "share": float((d > 0).mean()), "median_db": float(np.median(d))}
        print("%-18s %8d %8d %7.0f%% %+8.1f" % (label[:18], len(d), (d > 0).sum(), 100 * (d > 0).mean(), np.median(d)))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


def cmd_gate(a):
    import soundfile as sf
    import bench_roles
    m = load_manifest(a.manifest)
    cache = os.path.join(m["out_dir"], "roles")
    paths = {k: audio(a.stems, k) for k in DRUM_SUBS}
    paths = {k: v for k, v in paths.items() if v}
    results, named, bleed = bench_roles.run_set(a.set, paths, m["bpm"], m.get("bar1_s", 0.0), cache, 1, False)
    own = {}                  # --oracle: the true onsets of each stem's own sources
    if a.oracle:
        src = bench_roles.sources_stage(m, m["out_dir"], cache)
        for k in paths:
            own[k] = sorted(t for v in src["info"].values() if v["role"] == k for t in v["onsets"])
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    report = {}
    for stem, p in paths.items():
        marked = sorted(g for g, b in bleed.get(stem, {}).items() if b["bleed"])
        if a.coincidence_only:       # not the module's rule: every group of 4+ hits that coincides 70 %+, at p <= 1 %, with another stem's group
            marked = sorted(set(marked) | coincidence_groups(results[stem], bleed[stem]))
        x, sr = sf.read(p, dtype="float64", always_2d=True)
        # hit_groups removes the stem's DC before it looks at it, which moves no sample: its hit positions are the file's
        if a.oracle:
            gated = hits_without_true_onset(results[stem], own[stem]) if own[stem] else []
            y, n = gate_hits(x, sr, gated) if gated else (x, 0)
            report[stem] = {"own_onsets": len(own[stem]), "hits_gated": n, "hits": len(results[stem]["hits"])}
            print("  %-12s %5d true onsets of its own sources, %d hits gated of %d" % (stem, len(own[stem]), n, len(results[stem]["hits"])))
            sf.write(os.path.join(out, stem + ".flac"), y, sr, subtype="PCM_24", format="FLAC")
            continue
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
    p.add_argument("--add-back", metavar="STEM", help="also write plus_STEM_DRUMS/STEM.flac: that stem with the residual added back")
    p.set_defaults(fn=cmd_residual)
    p = sub.add_parser("wiener")
    p.add_argument("manifest"), p.add_argument("stems"), p.add_argument("out")
    p.add_argument("--power", type=float, default=POWER, help="the mask's exponent: 2 is the power (Wiener) mask, 1 the magnitude mask")
    p.set_defaults(fn=cmd_wiener)
    p = sub.add_parser("oracle")
    p.add_argument("manifest"), p.add_argument("out")
    p.add_argument("--power", type=float, default=POWER, help="the mask's exponent: 2 is the power (Wiener) mask, 1 the magnitude mask")
    p.set_defaults(fn=cmd_oracle)
    p = sub.add_parser("oracle-score")
    p.add_argument("manifest")
    p.add_argument("--power", type=float, default=POWER)
    p.add_argument("--json")
    p.set_defaults(fn=cmd_oracle_score)
    p = sub.add_parser("bars")
    p.add_argument("manifest"), p.add_argument("a"), p.add_argument("b")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_bars)
    p = sub.add_parser("gate")
    p.add_argument("manifest"), p.add_argument("stems"), p.add_argument("out")
    p.add_argument("--set", default="separated", help="the bench_roles cache name of this stems folder (default separated)")
    p.add_argument("--coincidence-only", action="store_true",
                   help="not mark_bleed's rule: also gate every group of 4+ hits that coincides 70 %%+ (p <= 1 %%) with another stem's group, "
                        "level and likeness unchecked; a counterfactual for what mark_bleed's two other conditions are worth")
    p.add_argument("--oracle", action="store_true",
                   help="hit-level oracle (the detector's cuts, perfect labels): silence every hit with no true onset of its stem's own sources "
                        "within 25 ms instead of mark_bleed's groups")
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

"""Sideman patch-match spectral distance (multi-resolution mel STFT) on whole 4-bar loops.
usage: mrstft_score.py <sideman worktree> ref.wav take...  (takes aligned to the ref's loop with score.align)"""
import sys, os, numpy as np, soundfile as sf
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.dirname(__file__))
from sideman import oneshot
import score as sc
ref = sys.argv[2]; r, srr = sf.read(ref, always_2d=True); n = lambda s: int(sc.BEATS*60/sc.BPM*s)
rm = r.mean(1); on = int(np.argmax(np.convolve(rm**2, np.ones(64)/64, 'same') > (rm**2).max()*1e-4)); r = r[on:on+n(srr)]
L = n(srr)/srr
for p in sys.argv[3:]:
    t, sr = sf.read(p, always_2d=True)
    o = sc.align(t.mean(1), sr, r.mean(1), srr); t = t[o:o+n(sr)]
    if sr != srr: t = oneshot._resample(t, sr, srr)
    m = min(len(t), len(r))
    d = oneshot.distance_terms(t[:m], r[:m], srr, max_s=L)
    print(f"{os.path.basename(p)[:44]:46s} distance {d['distance']:.3f}  " + '  '.join(f'{k}={v:.3f}' for k, v in d.items() if k != 'distance' and isinstance(v, float)))
# yardstick: Bodzin against his own variant loop (same bars, tweaked patch)
v = ref.replace('Loop_Sub37_3', 'Loop_Sub37_5')
if os.path.exists(v):
    t, sr = sf.read(v, always_2d=True); tm = t.mean(1); on = int(np.argmax(np.convolve(tm**2, np.ones(64)/64, 'same') > (tm**2).max()*1e-4)); t = t[on:on+n(sr)]
    m = min(len(t), len(r)); print(f"{'YARDSTICK Loop_Sub37_5 vs _3':46s} distance {oneshot.distance_terms(t[:m], r[:m], srr, max_s=L)['distance']:.3f}")

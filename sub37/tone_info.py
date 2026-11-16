"""Per tone: f0 -> MIDI note, length, gate (where the level falls toward release), peak/RMS. usage: tone_info.py dir..."""
import sys, os, glob, json, numpy as np, soundfile as sf
def f0(m, sr):
    s = m[int(0.05*sr):int(0.05*sr)+int(0.5*sr)]
    if len(s) < 2048: s = m
    X = np.abs(np.fft.rfft(s*np.hanning(len(s)), 8*len(s))); f = np.fft.rfftfreq(8*len(s), 1/sr)
    sel = (f > 25) & (f < 1200); Xs, fs = X[sel], f[sel]
    # harmonic sum over candidates
    cands = fs[np.argsort(Xs)[::-1][:12]]; best, bf = -1, None
    for c in cands:
        for d in (1, 2, 3):
            f1 = c/d
            if f1 < 25: continue
            sc = sum(np.interp(f1*k, fs, Xs) for k in range(1, 7))
            if sc > best: best, bf = sc, f1
    return bf
out = {}
for d in sys.argv[1:]:
    rows = []
    for p in sorted(glob.glob(os.path.join(d, '*.wav')), key=lambda s: int(''.join(c for c in os.path.basename(s).split('_')[0] if c.isdigit()) or 0)):
        x, sr = sf.read(p); m = x.mean(1) if x.ndim > 1 else x
        fr = f0(m, sr); midi = 69 + 12*np.log2(fr/440)
        env = np.sqrt(np.convolve(m**2, np.ones(int(0.01*sr))/int(0.01*sr), 'same'))
        db = 20*np.log10(env + 1e-9); pk = db.max()
        end = np.where(db > pk - 50)[0][-1]/sr
        rows.append(dict(file=os.path.basename(p), f0=round(fr, 2), midi=round(float(midi), 2), len=round(len(m)/sr, 2), audible=round(end, 2), sr=sr, ch=x.ndim))
    out[os.path.basename(d.rstrip('/'))] = rows
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1].rstrip('/')), '..', 'tones.json'), 'w'), indent=1)
for k, rows in out.items():
    print(k, len(rows), ' '.join(f"{r['midi']:.1f}/{r['audible']}s" for r in rows))

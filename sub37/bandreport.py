"""Per-band report against a reference loop: no single score.
usage: bandreport.py ref.wav label=take.aif [label=take.aif ...]"""
import sys, os, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__)); import score as sc
ZONES = [('sub 25-40', 25, 45), ('bass 50-80', 45, 90), ('low-mid 100-250', 90, 280), ('mid 315-1k', 280, 1120),
         ('high-mid 1.25-4k', 1120, 4500), ('high 5-16k', 4500, 18000)]
beat = 60/124; N = lambda s: int(16*beat*s)
def load(p, ref=None):
    x, sr = sf.read(p); m = x.mean(1)
    if ref is None:
        on = int(np.argmax(np.convolve(m**2, np.ones(64)/64, 'same') > (m**2).max()*1e-4)); return x[on:on+N(sr)], sr
    o = sc.align(m, sr, ref[0].mean(1), ref[1]); return x[o:o+N(sr)], sr
def spec(m, sr):
    X = np.abs(np.fft.rfft(m*np.hanning(len(m))))**2; f = np.fft.rfftfreq(len(m), 1/sr); return X, f
def zones(X, f):
    tot = X[(f > 20) & (f < 20000)].sum(); return [10*np.log10(X[(f >= a) & (f < b)].sum()/tot+1e-12) for _, a, b in ZONES]
def partials(m, sr, a, b, f0, n=6):
    s = m[int(a*sr):int(b*sr)]; X = np.abs(np.fft.rfft(s*np.hanning(len(s)))); f = np.fft.rfftfreq(len(s), 1/sr)
    v = [20*np.log10(X[(f > f0*k*0.97) & (f < f0*k*1.03)].max()+1e-12) for k in range(1, n+1)]; return [x-v[0] for x in v]
def feats(x, sr):
    m = x.mean(1); sd = (x[:, 0]-x[:, 1])/2; X, f = spec(m, sr); M, C, _, _ = sc.feats(m, sr)
    return dict(z=zones(X, f), crest=20*np.log10(abs(m).max()/np.sqrt((m**2).mean())), width=np.sqrt((sd**2).mean()/(m**2).mean()),
                ped=partials(m, sr, 2*beat+0.05, 3*beat-0.02, 65.41), lfo=sc.lfo_rate(M), rms=20*np.log10(np.sqrt((m**2).mean())))
ref = load(sys.argv[1]); R = feats(*ref)
takes = [(a.split('=', 1)[0], feats(*load(a.split('=', 1)[1], ref))) for a in sys.argv[2:]]
w = max(len(l) for l, _ in takes) + 2
print('Band balance vs Bodzin (dB, share of energy; + = more than Bodzin)')
print(' '*w + ''.join(f'{z[0]:>18s}' for z in ZONES))
for l, T in takes: print(f'{l:{w}s}' + ''.join(f'{t-r:+18.1f}' for t, r in zip(T['z'], R['z'])))
print('\nCharacter (Bodzin first)')
print(f"{'':{w}s}{'crest dB':>10s}{'width':>8s}{'LFO Hz':>12s}{'wobble dB':>12s}   pedal C h2..h6 vs h1 (dB)")
for l, T in [('Bodzin', R)] + takes:
    lf = T['lfo']; print(f"{l:{w}s}{T['crest']:10.1f}{T['width']:8.2f}{lf[0][0]:6.2f}/{lf[1][0]:<5.2f}{lf[0][1]:6.1f}/{lf[1][1]:<5.1f}   " + ' '.join(f'{v:+6.1f}' for v in T['ped'][1:]))

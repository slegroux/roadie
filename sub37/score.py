"""Score synth takes against a reference loop (same tempo, loop of `beats` beats).
usage: score.py ref.wav take1.aif [take2.aif ...]"""
import sys, numpy as np, soundfile as sf
BPM, BEATS, HOP, WIN = 124.0, 16, 0.01, 0.04
def mono(p):
    x, sr = sf.read(p); return (x.mean(1) if x.ndim > 1 else x), sr
def stft(m, sr):
    H, W = int(HOP*sr), int(WIN*sr); w = np.hanning(W); f = np.fft.rfftfreq(W, 1/sr)
    fr = np.array([np.abs(np.fft.rfft(m[i:i+W]*w))**2 for i in range(0, len(m)-W, H)]); return fr, f
def mel_fb(f, n=48, lo=30, hi=12000):
    mel = lambda h: 2595*np.log10(1+h/700); inv = lambda m: 700*(10**(m/2595)-1)
    pts = inv(np.linspace(mel(lo), mel(hi), n+2)); fb = np.zeros((n, len(f)))
    for i in range(n):
        a, b, c = pts[i:i+3]; fb[i] = np.clip(np.minimum((f-a)/(b-a), (c-f)/(c-b)), 0, None)
    return fb
def feats(m, sr):
    S, f = stft(m, sr); M = 10*np.log10(S @ mel_fb(f).T + 1e-10)
    C = (S*f).sum(1)/(S.sum(1)+1e-12); return M, C, S, f
def align(t, sr, r, srr):
    """offset of take (within one loop cycle) whose centroid contour best matches the ref"""
    _, Cr, _, _ = feats(r, srr); _, Ct, _, _ = feats(t, sr)
    a = np.log2(Cr+1); a = (a-a.mean())/a.std(); b = np.log2(Ct+1); n = len(a)
    k = max(range(0, max(1, len(b)-n)), key=lambda k: np.dot((b[k:k+n]-b[k:k+n].mean())/(b[k:k+n].std()+1e-9), a))
    return int(k*HOP*sr)
def bands(S, f):
    e = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20000]; X = S.sum(0); tot = X.sum()
    return np.array([10*np.log10(X[(f >= a) & (f < b)].sum()/tot+1e-12) for a, b in zip(e, e[1:])]), e
def lfo_rate(M, f_lo=0.5, f_hi=8):
    # wobble of the 400-4k mel bands during the two sustained notes (beats 8-11.2, 12-15.7)
    beat = 60/BPM; out = []
    for a, b in ((8*beat+.15, 11.2*beat), (12*beat+.15, 15.7*beat)):
        s = M[int(a/HOP):int(b/HOP), 18:40].mean(1); s = s-np.convolve(s, np.ones(41)/41, 'same'); s = s[25:-25]
        F = np.abs(np.fft.rfft(s*np.hanning(len(s)), 16*len(s))); fq = np.fft.rfftfreq(16*len(s), HOP)
        k = (fq > f_lo) & (fq < f_hi); out.append((fq[k][np.argmax(F[k])], s.std()))
    return out
def echo(m, sr):
    # energy-envelope autocorrelation at the 1/8T lag (1/3 beat) relative to lag 0
    H = int(0.005*sr); e = np.array([np.sqrt((m[i:i+H]**2).mean()) for i in range(0, len(m)-H, H)]); e = e-e.mean()
    lag = int(round(60/BPM/3/0.005)); return float(np.dot(e[:-lag], e[lag:])/np.dot(e, e))
if __name__ == '__main__':
    r, srr = mono(sys.argv[1]); n = lambda s: int(BEATS*60/BPM*s)
    on = int(np.argmax(np.convolve(r**2, np.ones(64)/64, 'same') > (r**2).max()*1e-4)); r = r[on:on+n(srr)]
    Mr, Cr, Sr, fr = feats(r, srr); br, edges = bands(Sr, fr); lr = lfo_rate(Mr); er = echo(r, srr)
    lv = lambda m: 20*np.log10(np.sqrt((m**2).mean()))
    print(f'REF  rms {lv(r):6.1f} dB | LFO {lr[0][0]:.2f}/{lr[1][0]:.2f} Hz ±{lr[0][1]:.1f}/{lr[1][1]:.1f} dB | 1/8T echo corr {er:+.2f}')
    rows = []
    for p in sys.argv[2:]:
        t, sr = mono(p); o = align(t, sr, r, srr); t = t[o:o+n(sr)]
        Mt, Ct, St, ft = feats(t, sr); L = min(len(Mt), len(Mr))
        a, b = Mt[:L]-Mt[:L].mean(1, keepdims=True), Mr[:L]-Mr[:L].mean(1, keepdims=True)
        cos = float(np.mean((a*b).sum(1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)+1e-9)))
        bt, _ = bands(St, ft); d = bt-br; mae = float(np.abs(d[1:10]).mean())
        cerr = float(np.abs(np.log2((Ct[:L]+1)/(Cr[:L]+1))).mean())
        lt = lfo_rate(Mt); et = echo(t, sr)
        score = 100*np.exp(-mae/10)*np.exp(-cerr)*max(cos, 0)**0.5
        rows.append((p.split('/')[-1][:44], score, cos, mae, cerr, d, lt, et, lv(t)))
    for name, score, cos, mae, cerr, d, lt, et, lvl in rows:
        print(f'\n{name}\n  SCORE {score:5.1f}/100 | spectral-shape corr {cos:.3f} | band error {mae:4.1f} dB | brightness error {cerr:.2f} oct | rms {lvl:6.1f} dB')
        print('  bands (take-ref dB): ' + ' '.join(f'{edges[i]}:{d[i]:+.1f}' for i in range(len(d))))
        print(f'  LFO {lt[0][0]:.2f}/{lt[1][0]:.2f} Hz ±{lt[0][1]:.1f}/{lt[1][1]:.1f} dB | 1/8T echo corr {et:+.2f}')

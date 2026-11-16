"""f0 over time (lowest strong spectral peak) in 0.25 s frames: python f0track.py file.aif"""
import sys, numpy as np, soundfile as sf, math
x, sr = sf.read(sys.argv[1]); m = x.mean(1) if x.ndim > 1 else x
W = int(0.25*sr); names = 'C C# D D# E F F# G G# A A# B'.split()
for i in range(0, len(m)-W, W):
    seg = m[i:i+W]
    if np.sqrt((seg**2).mean()) < 1e-4: print(f'{i/sr:5.2f}s  silent'); continue
    X = np.abs(np.fft.rfft(seg*np.hanning(W), 8*W)); f = np.fft.rfftfreq(8*W, 1/sr)
    k = (f > 15) & (f < 2000); Xk, fk = X[k], f[k]; thr = Xk.max()*0.25
    pk = [j for j in range(1, len(Xk)-1) if Xk[j] > thr and Xk[j] >= Xk[j-1] and Xk[j] >= Xk[j+1]]
    f0 = fk[pk[0]] if pk else fk[np.argmax(Xk)]; mid = 69+12*math.log2(f0/440); n = round(mid)
    top = sorted(pk, key=lambda j: -Xk[j])[:3]
    print(f'{i/sr:5.2f}s  f0 {f0:7.2f} Hz  {names[n%12]}{n//12-2} {100*(mid-n):+4.0f}c   loudest {[round(fk[j],1) for j in top]}')

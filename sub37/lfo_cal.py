import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt
x, sr = sf.read(sys.argv[1]); m = x.mean(1)
on = int(np.argmax(np.abs(m) > np.abs(m).max()*0.1)); m = m[on+int(0.5*sr): on+int(7*sr)]
b = sosfiltfilt(butter(4, [300, 3000], 'band', fs=sr, output='sos'), m)
env = np.sqrt(sosfiltfilt(butter(2, 20, fs=sr, output='sos'), b**2).clip(1e-12))
e = 20*np.log10(env[::48]); e -= e.mean(); fs = sr/48
E = np.abs(np.fft.rfft(e*np.hanning(len(e)), 16*len(e))); f = np.fft.rfftfreq(16*len(e), 1/fs)
s = (f > 0.5) & (f < 12); print(f"LFO {f[s][np.argmax(E[s])]:.2f} Hz  depth {np.percentile(e,95)-np.percentile(e,5):.1f} dB")

"""Per-note envelope analysis of a 4-bar take against a reference loop (Boavista bars 6-9).
Amp: attack 10->90 %, time to peak, level at 80 % of the note vs peak. Brightness (>400 Hz band, the filter
envelope's footprint): level at onset+20 ms vs the whole-loop median, decay from its peak to -6 dB, level at 80 % vs peak.
usage: envelopes.py ref.wav label=take ..."""
import sys, os, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt
sys.path.insert(0, os.path.dirname(__file__)); import score as sc
BPM = 124; beat = 60/BPM
NOTES = [(0,.25,'ans'),(.25,.25,'ans'),(.5,1,'ped'),(1.5,.25,'ans'),(1.75,.25,'ans'),(2,1,'ped'),(3,.25,'ans'),(3.25,.25,'ans'),
         (3.5,1,'ped'),(4.5,.25,'ans'),(4.75,.25,'ans'),(5,1,'ped'),(6,.25,'ans'),(6.25,.25,'ans'),(6.5,1,'ped'),(7.5,.25,'ans'),
         (7.75,.25,'ans'),(8,3.25,'long'),(11.25,.25,'ans'),(11.5,.25,'ans'),(11.75,.25,'ans'),(12,3.75,'long')]
HOP = 0.002
def env(x, sr, win):
    W = int(win*sr); H = int(HOP*sr); e = np.sqrt(np.convolve(x**2, np.ones(W)/W, 'same'))[::H]; return 20*np.log10(e+1e-9)
def analyse(m, sr):
    hp = sosfiltfilt(butter(4, 400, 'hp', fs=sr, output='sos'), m)
    A = env(m, sr, 0.02); B = env(hp, sr, 0.01); bmed = np.median(B)
    out = {'ans': [], 'ped': [], 'long': []}
    for st, du, g in NOTES:
        i0 = int(st*beat/HOP); i1 = int((st+du)*beat/HOP); seg = A[i0:i1]; bseg = B[i0:i1]
        if len(seg) < 10: continue
        lin = 10**(seg/20); pk = lin[:min(len(lin), int(0.15/HOP))].max()
        k10 = np.argmax(lin >= 0.1*pk); k90 = np.argmax(lin >= 0.9*pk); kp = np.argmax(lin[:min(len(lin), int(0.15/HOP))])
        sus = seg[int(0.8*len(seg))] - 20*np.log10(pk)
        bpk_i = np.argmax(bseg[:max(5, int(0.3*len(bseg)))]); bpk = bseg[bpk_i]
        after = bseg[bpk_i:]; dk = np.argmax(after <= bpk-6) if (after <= bpk-6).any() else None
        out[g].append(dict(att=(k90-k10)*HOP*1000, tpk=kp*HOP*1000, sus=sus, b20=bseg[min(len(bseg)-1, int(0.02/HOP))]-bmed,
                           bdec=None if dk is None else dk*HOP*1000, bend=bseg[int(0.8*len(bseg))]-bpk))
    return out
def med(v): v = [x for x in v if x is not None]; return (np.median(v), len(v)) if v else (float('nan'), 0)
ref = sys.argv[1]; r, srr = sf.read(ref); rm = r.mean(1); on = int(np.argmax(np.convolve(rm**2, np.ones(64)/64, 'same') > (rm**2).max()*1e-4))
rm = rm[on:on+int(16*beat*srr)]
rows = [('Bodzin', analyse(rm, srr))]
for a in sys.argv[2:]:
    lab, p = a.split('=', 1); x, sr = sf.read(p); m = x.mean(1); o = sc.align(m, sr, rm, srr); rows.append((lab, analyse(m[o:o+int(16*beat*sr)], sr)))
cols = [('att', 'amp attack ms'), ('tpk', 'amp peak ms'), ('sus', 'amp end dB'), ('b20', 'bright@20ms dB'), ('bdec', 'bright -6dB ms'), ('bend', 'bright end dB')]
for g, gname in (('ans', '16th answers (x15)'), ('ped', 'pedal C, 1 beat (x5)'), ('long', 'long notes G# F (x2)')):
    print(f'\n{gname}\n' + f"{'':16s}" + ''.join(f'{c[1]:>16s}' for c in cols))
    for lab, res in rows:
        print(f'{lab:16s}' + ''.join(f"{med([n[c[0]] for n in res[g]])[0]:16.1f}" for c in cols))

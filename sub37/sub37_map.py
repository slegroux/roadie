"""Sub 37 (decoded fractions) -> UADx Moog Minimoog Synth starting values. Pure: no Live, no I/O."""
import math

# ---- Sub37 (decoded fractions) -> UADx Minimoog (Live 0..1 / indices). Anchored on the Boavista pass of
# 2026-11-15 (take 8): osc2 tune 0.497 + st/20.1, levels osc1 0.97 / osc2 0.84x / sub 0.8x, cutoff 0.176 -> 0.36,
# EG +26.7 % -> contour 0.35, filter A 0.31 -> 0.21, D 0.77 -> 0.35, amp release 0.56 -> decay 0.45 (DECAY on),
# LFO 2.7 Hz -> 0.49, mod depth +38 % -> MOD AMOUNT 0.6, multidrive 0.85 -> feedback 0.08.
LFO_CAL = [(0.1, 0.51), (0.49, 2.74), (0.5, 3.16), (0.6, 7.25)]          # UAD LFO RATE -> Hz, measured
def lfo_norm(hz):
    pts = LFO_CAL; lh = math.log(max(hz, 0.05))
    for (a, fa), (b, fb) in zip(pts, pts[1:]):
        if lh <= math.log(fb) or (b, fb) == pts[-1]:
            return min(1.0, max(0.0, a + (lh - math.log(fa)) * (b - a) / (math.log(fb) - math.log(fa))))
def wave_idx(frac):          # Sub37 wave: tri 0 -> saw 1/3 -> square 2/3 -> pulse 1
    p = frac * 3
    return 0 if p < 0.25 else 1 if p < 0.45 else 2 if p < 1.5 else 3 if p < 2.2 else 4 if p < 2.6 else 5
def oct_idx(v):              # Sub37 octave value 0..3 = 16' 8' 4' 2' -> UAD 0 Lo,1 32',2 16',3 8',4 4',5 2'
    return 2 + int(round(v))
def map_patch(x):
    f = lambda k: x[k]["frac"]; v = lambda k: x[k]["value"]
    o1 = oct_idx(v("osc1oct")); o2 = oct_idx(v("osc2oct"))
    st2 = (f("osc2freq") - 0.5) * 14 + (f("osc2beat") - 0.5) * 7.4
    egamt = (f("fltegamt") - 0.5) * 2
    kb = f("fltkbamt")
    lfo_hz = {2: 2.7 * (f("lfo1rate") / 0.1476) ** 1.0, 1: 0.6 + 4 * f("lfo1rate"), 0: 0.1 + 1 * f("lfo1rate")}[int(v("lfo1range"))]
    depth = max(0.0, (f("mod1fltamt") - 0.5) * 2) * f("modwheel") * f("mod1mwamt")
    sus_amp = f("ampegsus"); rel = f("ampegrls")
    drive = f("fltdrive"); fb = f("fbextlvl") if v("fbexton") else 0.0
    p = {
        "OSC-1 Octave": o1, "OSC-2 Octave": o2, "OSC-3 Octave": max(0, o1 - 1),
        "OSC-1 WAVE": wave_idx(f("osc1wav")), "OSC-2 WAVE": wave_idx(f("osc2wav")), "OSC-3 WAVE": 3,
        "OSC-1 On": int(v("osc1on") and f("osc1lvl") > 0.01), "OSC-2 On": int(v("osc2on") and f("osc2lvl") > 0.01),
        "OSC-3 On": int(v("suboscon") and f("subosclvl") > 0.01), "NOISE On": int(v("noiseon") and f("noiselvl") > 0.01),
        "OSC-1 Mix": 0.97 * f("osc1lvl"), "OSC-2 Mix": 0.84 * f("osc2lvl"), "OSC-3 Mix": 0.8 * f("subosclvl"),
        "NOISE Mix": f("noiselvl"), "NOISE Pink": 1,
        "OSC-2 Tune": min(1, max(0, 0.497 + st2 / 20.1)), "OSC-3 Tune": 0.5, "OSC-3 Ctrl. On": 1, "OSC Mod": 0,
        "EXT. On": 1, "Input Source": 1, "EXT. Mix": min(0.3, 0.08 * drive / 0.85 + 0.3 * fb),
        "FILTER Cutoff": min(1, max(0, f("fltcutoff") + 0.184)), "FILTER Res": 0.9 * f("fltres"),
        "FILTER Contour": min(1, max(0, egamt * 1.31)),
        "FILTER Kbd 1/3": int(0.2 <= kb < 0.5 or kb >= 0.8), "FILTER Kbd 2/3": int(kb >= 0.5),
        "FILT Env Attack": 0.68 * f("fltegatk"), "FILT Env Decay": 0.45 * f("fltegdcy"), "FILT Env Sustain": f("fltegsus"),
        "AMP Env Attack": min(1, 0.15 + 0.7 * f("ampegatk")), "AMP Env Sustain": sus_amp,
        "AMP Env Decay": 0.8 * rel if rel > 0.02 else 0.45 * f("ampegdcy"), "DECAY On": int(rel > 0.02),
        "Trigger Mode": 0 if v("ampegmtrig") else 1, "Note Priority": 0,
        "MOD SOURCE A": 0, "MOD SOURCE B": 0, "Modulation Mix": 1.0, "FILTER Mod": 1,
        "MOD AMOUNT": min(1.0, 0.6 * math.sqrt(depth / 0.38)) if depth > 0 else 0.0,
        "LFO RATE": lfo_norm(lfo_hz), "LFO WAVE": 0 if v("mod1src") == 1 else 1, "TEMPO SYNC": int(v("lfo1sync")),
        "GLIDE On": int(f("glidetime") > 0.02), "GLIDE": min(1, 0.9 * f("glidetime")),
        "VOLUME": 0.7, "TUNE": 0.5, "POWER": 1,
    }
    return p, dict(lfo_hz=round(lfo_hz, 2), depth=round(depth, 3), slope_db=6 * (1 + int(v("fltslope"))), drive=round(drive, 2), fb=round(fb, 2))

"""Offline checks for the Sub 37 tools: SysEx field unpacking and the Sub37 -> UADx starting map."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sub37_decode as dec
import sub37_map as sm


def test_field_unpacking():
    """High bits at encOff/encBitOff, then 6 extension bits per byte, left-aligned from bit 5."""
    f = dict(id="x", caption="X", min=0, max=16383, off=2, bit=1, nb=2, ext=12, zero=None)
    v = 0b10_101010_010101                       # hi 0b10, ext 0b101010 then 0b010101
    d = bytearray(8); d[2] = 0b10 << 1; d[3] = 0b101010; d[4] = 0b010101
    assert dec.decode([f], bytes(d))["x"]["value"] == v
    g = dict(f, ext=3)                           # a partial extension byte keeps its top bits
    d[3] = 0b101000
    assert dec.decode([g], bytes(d))["x"]["value"] == (0b10 << 3) | 0b101


def test_boavista_anchor():
    """The Boavista pass (2026-11-15, take 8) is the map's anchor: its decoded values give its tuned knobs."""
    fr = dict(osc1oct=(1, 1 / 3), osc2oct=(0, 0), osc1wav=(0, .2175), osc2wav=(0, .3759), osc2freq=(16383, 1.0), osc2beat=(0, .5274),
              osc1on=(1, 1), osc1lvl=(0, 1.0), osc2on=(1, 1), osc2lvl=(0, .5025), suboscon=(1, 1), subosclvl=(0, 1.0),
              noiseon=(1, 1), noiselvl=(0, 0.0), fltcutoff=(0, .1763), fltres=(0, .0007), fltegamt=(0, .6336), fltkbamt=(0, .4275),
              fltegatk=(0, .3122), fltegdcy=(0, .7736), fltegsus=(0, .0002), ampegatk=(0, .0021), ampegdcy=(0, 0.0),
              ampegsus=(0, 1.0), ampegrls=(0, .5611), ampegmtrig=(1, 1), lfo1rate=(0, .1476), lfo1range=(2, 1.0),
              lfo1sync=(0, 0), mod1fltamt=(0, .69), modwheel=(0, 1.0), mod1mwamt=(0, 1.0), mod1src=(0, 0),
              fltslope=(1, 1 / 3), fltdrive=(0, .85), fbextlvl=(0, 0.0), fbexton=(0, 0), glidetime=(0, 0.0))
    x = {k: {"value": v, "frac": f} for k, (v, f) in fr.items()}
    p, info = sm.map_patch(x)
    assert (p["OSC-1 Octave"], p["OSC-2 Octave"], p["OSC-3 Octave"]) == (3, 2, 2)     # 8', 16', sub 16'
    assert abs(p["FILTER Cutoff"] - 0.36) < 0.01 and abs(p["FILT Env Decay"] - 0.35) < 0.01
    assert abs(p["LFO RATE"] - 0.49) < 0.01 and abs(info["lfo_hz"] - 2.7) < 0.05
    assert p["Trigger Mode"] == 0 and p["GLIDE On"] == 0 and p["OSC-3 Ctrl. On"] == 1


def test_lfo_and_wave_maps():
    rates = [sm.lfo_norm(h) for h in (0.3, 1, 2.7, 5, 10)]
    assert rates == sorted(rates) and all(0 <= r <= 1 for r in rates)
    assert [sm.wave_idx(f) for f in (0, .22, .5, .7, 1)] == [0, 2, 3, 3, 5]

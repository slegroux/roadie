"""Decode a Moog Sub 37 single-patch .syx with bsp2/tks moog_sub37.msp (hardware-derived field map).
Offsets count from the byte after F0. A field's high bits sit in byte encOff at encBitOff (encBitNum
bits); encBitNumExt extension bits follow, 6 per byte, left-aligned from bit 5.
usage: sub37_decode.py moog_sub37.msp patch.syx [...]  -> JSON {patch: {id: {caption, value, min, max, frac}}}"""
import re, sys, json
def fields(msp):
    t = open(msp).read(); out = []
    for m in re.finditer(r'<nrpn\s+id=(\w+)\s+nr=(\d+)\s+caption="([^"]*)"(.*?)/>', t, re.S):
        a = dict(re.findall(r'(\w+)=(-?\d+)', m.group(4)))
        if 'encOff' not in a: continue
        out.append(dict(id=m.group(1), caption=m.group(3), min=int(a.get('min', 0)), max=int(a.get('max', 127)),
                        off=int(a['encOff']), bit=int(a.get('encBitOff', 0)), nb=int(a.get('encBitNum', 0)),
                        ext=int(a.get('encBitNumExt', 0)), zero=int(a['uiZero']) if 'uiZero' in a else None))
    return out
def decode(fs, d):
    res = {}
    for f in fs:
        v = (d[f['off']] >> f['bit']) & ((1 << f['nb']) - 1) if f['nb'] else 0
        left, k = f['ext'], 1
        while left > 0:
            take = min(6, left); b = d[f['off'] + k] & 0x3F
            v = (v << take) | (b >> (6 - take)); left -= take; k += 1
        rng = f['max'] - f['min']
        res[f['id']] = dict(caption=f['caption'], value=v, min=f['min'], max=f['max'], frac=round((v - f['min']) / rng, 4) if rng else 0,
                            zero=f['zero'], ok=f['min'] <= v <= f['max'])
    return res
if __name__ == '__main__':
    fs = fields(sys.argv[1]); out = {}
    for p in sys.argv[2:]:
        raw = open(p, 'rb').read(); d = raw[1:]
        name = ''.join(chr(c) for c in raw[0x17:0x17+40] if 32 <= c < 127)[:0]
        out[p.split('/')[-1].replace('.syx', '').replace('000_', '')] = decode(fs, d)
    json.dump(out, sys.stdout, indent=1)

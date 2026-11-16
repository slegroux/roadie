import struct, sys, glob
def up(b, i):
    t = b[i]
    if t <= 0x7f: return t, i+1
    if 0x80 <= t <= 0x8f: n = t & 15; i += 1; d = {}
    elif 0x90 <= t <= 0x9f: n = t & 15; i += 1; return lst(b, i, n)
    elif 0xa0 <= t <= 0xbf: n = t & 31; return b[i+1:i+1+n].decode('utf8', 'replace'), i+1+n
    elif t == 0xc0: return None, i+1
    elif t in (0xc2, 0xc3): return t == 0xc3, i+1
    elif t == 0xcc: return b[i+1], i+2
    elif t == 0xcd: return struct.unpack('>H', b[i+1:i+3])[0], i+3
    elif t == 0xce: return struct.unpack('>I', b[i+1:i+5])[0], i+5
    elif t == 0xd9: n = b[i+1]; return b[i+2:i+2+n].decode('utf8', 'replace'), i+2+n
    elif t == 0xdc: n = struct.unpack('>H', b[i+1:i+3])[0]; return lst(b, i+3, n)
    elif t == 0xde: n = struct.unpack('>H', b[i+1:i+3])[0]; i += 3; d = {}
    elif t == 0xca: return struct.unpack('>f', b[i+1:i+5])[0], i+5
    elif t == 0xcb: return struct.unpack('>d', b[i+1:i+9])[0], i+9
    elif t >= 0xe0: return t-256, i+1
    else: raise ValueError(hex(t))
    for _ in range(n):
        k, i = up(b, i); v, i = up(b, i); d[k] = v
    return d, i
def lst(b, i, n):
    out = []
    for _ in range(n):
        v, i = up(b, i); out.append(v)
    return out, i
ids = {}
for f in glob.glob("/Library/Application Support/Universal Audio/Plug-Ins/uaudio_minimoog.lunacomponent/extras/nks/*.nksf"):
    d = open(f, 'rb').read(); i = 12
    while i < len(d):
        cid = d[i:i+4]; n = struct.unpack('<I', d[i+4:i+8])[0]
        if cid == b'NICA': o, _ = up(d[i+12:i+8+n], 0)
        i += 8+n+(n&1)
    for page in o['ni8']:
        for c in page:
            if c.get('id') is not None: ids.setdefault(c['id'], set()).add(c.get('name'))
for k in sorted(ids): print(k, sorted(ids[k]))

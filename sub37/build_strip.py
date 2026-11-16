"""Build an Instrument Rack .adg (and an Arturia Mini V3 preset) from a track's Instrument Rack in a saved .als.
usage: build_strip.py set.xml "<track name>" template.adg.xml out.adg branch_name [arturia_out preset_name]"""
import re, sys, gzip, binascii, plistlib
def top_children(xml):
    """(tag, start, end) of top-level elements in an XML fragment"""
    out, depth, cur = [], 0, None
    for m in re.finditer(r'<(/?)([A-Za-z0-9_.]+)([^>]*?)(/?)>', xml):
        close, tag, _, selfc = m.groups()
        if close:
            depth -= 1
            if depth == 0: out.append((cur[0], cur[1], m.end()))
            if depth < 0: break
        elif selfc:
            if depth == 0: out.append((tag, m.start(), m.end()))
        else:
            if depth == 0: cur = (tag, m.start())
            depth += 1
    return out
def element(xml, start_tag_pos):
    tag = re.match(r'<([A-Za-z0-9_.]+)', xml[start_tag_pos:]).group(1)
    sub = xml[start_tag_pos:]; ch = top_children(sub)
    return sub[:ch[0][2]]
setx, track, tmpl, out = sys.argv[1:5]; branch = sys.argv[5]
t = open(setx, encoding='utf-8').read()
i = t.find(f'<EffectiveName Value="{track}"'); tr = t[i:t.find('</MidiTrack>', i)]
dev = tr[tr.find('<Devices>') + len('<Devices>'):]
rack = top_children(dev)[0]; assert rack[0] == 'InstrumentGroupDevice', rack
rk = dev[rack[1]:rack[2]]
chain = rk[rk.find('<InstrumentBranch '):]; d0 = chain.find('<Devices>') + len('<Devices>')
devs = top_children(chain[d0:])
presets = []; mini_state = None
nb = open(tmpl, encoding='utf-8').read()
native_tail = '''
							<PresetRef>
								<AbletonDefaultPresetRef Id="0">
									<FileRef>
										<RelativePathType Value="0" />
										<RelativePath Value="" />
										<Path Value="" />
										<Type Value="2" />
										<LivePackName Value="" />
										<LivePackId Value="" />
										<OriginalFileSize Value="0" />
										<OriginalCrc Value="0" />
									</FileRef>
									<DeviceId Name="%s" />
								</AbletonDefaultPresetRef>
							</PresetRef>'''
for n, (tag, a, b) in enumerate(devs):
    x = chain[d0:][a:b]
    if tag == 'AuPluginDevice':
        p = x.find('<AuPreset '); au = element(x, p)
        au = re.sub(r'^<AuPreset Id="\d+"', f'<AuPreset Id="{n}"', au)
        pl = x[x.find('<ParameterList>'):x.find('</ParameterList>')]
        # Live stores each slot as a Float/Enum pair; quantized params (switches) are the Enum ones.
        allents = re.findall(r'<(PluginFloatParameter|PluginEnumParameter) Id="\d+">.*?<ParameterId Value="(-?\d+)" />.*?<VisualIndex Value="(-?\d+)" />', pl, re.S)
        ents = [(pos, typ, pid, vi) for pos, (typ, pid, vi) in enumerate(allents) if pid != '-1' or typ == 'PluginFloatParameter']
        ps = ''.join(f'''<PluginParameterSettings Id="{k}"><Index Value="{pos}" /><VisualIndex Value="{vi}" /><ParameterId Value="{pid}" /><Type Value="{typ}" /><MacroControlIndex Value="-1" /><MidiControllerRange /><LomId Value="0" /></PluginParameterSettings>''' for k, (pos, typ, pid, vi) in enumerate(ents))
        ents = [e for e in ents if e[2] != '-1']
        au = re.sub(r'<ParameterSettings\s*/>|<ParameterSettings>.*?</ParameterSettings>', '<ParameterSettings>' + ps + '</ParameterSettings>', au, count=1, flags=re.S)
        print('   exposed', len(ents))
        presets.append(au)
        nm = re.search(r'<AuPluginInfo[^>]*>.*?<Name Value="([^"]*)"', x, re.S) or re.search(r'<PlugName Value="([^"]*)"', x)
        if nm and 'Mini V3' in nm.group(1):
            # keep the SFC rack's own Mini preset (its 53-param exposed layout); swap in only the sound state
            ta = nb.find('<AuPreset Id="0">'); tpl_au = nb[ta:nb.find('</AuPreset>', ta) + len('</AuPreset>')]
            ourbuf = re.search(r'<Buffer>.*?</Buffer>', au, re.S).group(0)
            au = re.sub(r'<Buffer>.*?</Buffer>', lambda _m: ourbuf, tpl_au, count=1, flags=re.S)
            au = re.sub(r'^<AuPreset Id="\d+"', f'<AuPreset Id="{n}"', au)
            presets[-1] = au
            print('   Mini: SFC exposed layout kept,', au.count('<PluginParameterSettings'), 'params')
            buf = re.search(r'<Buffer>(.*?)</Buffer>', au, re.S).group(1)
            mini_state = plistlib.loads(binascii.unhexlify(re.sub(r'\s', '', buf)))['Controller State']
        print(n, 'plugin', nm.group(1) if nm else '?', len(au))
    else:
        x2 = re.sub(r'^<(\w+) Id="\d+"', r'<\1 Id="0"', x)
        presets.append(f'<AbletonDevicePreset Id="{n}">\n<OverwriteProtectionNumber Value="3076" />\n<Device>\n{x2}\n</Device>' + native_tail % tag + '\n</AbletonDevicePreset>')
        print(n, 'native', tag, len(x))
# swap the template's DevicePresets (inside the first InstrumentBranchPreset) for ours
bp = nb.find('<InstrumentBranchPreset Id="0">'); ds = nb.find('<DevicePresets>', bp); de = nb.find('</DevicePresets>', ds)
nb = nb[:ds] + '<DevicePresets>\n' + '\n'.join(presets) + '\n</DevicePresets>' + nb[de + len('</DevicePresets>'):]
nb = re.sub(r'(<InstrumentBranchPreset Id="0">\s*<Name Value=")[^"]*(")', r'\g<1>' + branch + r'\2', nb, count=1)
with gzip.open(out, 'wb') as f: f.write(nb.encode('utf-8'))
print('wrote', out, len(nb))
if len(sys.argv) > 7 and mini_state:
    s = mini_state.decode('latin1'); name = sys.argv[7]
    hdr = re.match(r'(22 serialization::archive 10 0 7 0 7 )(\d+) ', s)
    pos = hdr.end(1); L = int(hdr.group(2)); rest = s[pos + len(hdr.group(2)) + 1 + L:]       # drop old name
    m = re.match(r' (\d+) ', rest); bank_len = int(m.group(1)); rest = rest[m.end() + bank_len:]  # drop old bank
    m = re.match(r' (\d+) (\d+) ', rest); flag = m.group(1); alen = int(m.group(2)); rest = rest[m.end() + alen:]  # drop author
    s2 = s[:pos] + f'{len(name)} {name} 4 User {flag} 7 Sylvain' + rest
    open(sys.argv[6], 'wb').write(s2.encode('latin1'))
    print('wrote arturia', sys.argv[6], len(s2), '| header:', s2[:110])

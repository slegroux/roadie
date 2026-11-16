"""Build one UAD track per Bodzin Sub37 patch in the open Set: Sub37 Strip rack (BZ Boavista (UAD).adg) with the
patch's decoded values, slot 0 = Bodzin's MIDI file, slot 1 = a test clip of reference tone pitches, plus a capture
track per song. Writes session.json (track names, tone layout) for passes.py.  usage: build_session.py"""
import json, os, sys, time, math, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uadlib import L, set_params, map_patch, names, HERE
import mido
SONGS = [  # patch key, track label, MIDI file, tones dir
    ("Earth", "Earth", "Earth.mid", "01_Earth"), ("LLL", "LLL", "LLL.mid", "02_LLL"),
    ("InfiniteMonkey", "Infinite Monkey", "InfiniteMonkey.mid", "05_Infinite Monkey"), ("Dune", "Dune", "Dune.mid", "06_Dune"),
    ("CooperStation", "Cooper Station", "CooperStation.mid", "08_Cooper_Station"),
    ("NothingLikeYou", "Nothing Like You", "NothingLikeYou.mid", "09_Nothing_Like_You_feat._Luna_Semara"),
    ("Isaac_1", "Isaac 1", "Isaac.mid", "10_Isaac_1"), ("Isaac_2", "Isaac 2", "Isaac_2.mid", "10_Isaac_2"),
    ("Collider", "Collider", "Collider.mid", "12_Collider"), ("Trancoso", "Trancoso", "Transcoso.mid", "13_Trancoso"),
    ("SpacetimeHero", "Spacetime Hero", "SpacetimeHero.mid", "14_Spactime_Hero"), ("Ataraxia", "Ataraxia", "Ataraxia.mid", "15_Ataraxia"),
]
RACK = "user_library/_PRODUCTION/Instruments/Sub37 Strip/BZ Boavista (UAD).adg"
BEAT = 60 / 124.0
dec = json.load(open(os.path.join(HERE, "decoded.json"))); tones = json.load(open(os.path.join(HERE, "tones.json")))
def midi_notes(path):
    m = mido.MidiFile(path); tpb = m.ticks_per_beat; notes = []
    for tr in m.tracks:
        t = 0; on = {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0: on[msg.note] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on") and msg.note in on:
                t0, vel = on.pop(msg.note); notes.append({"pitch": msg.note, "start_time": t0 / tpb, "duration": max(1, t - t0) / tpb, "velocity": float(vel)})
    s0 = math.floor(min(n["start_time"] for n in notes) / 4) * 4
    for n in notes: n["start_time"] -= s0
    return notes, math.ceil(max(n["start_time"] + n["duration"] for n in notes) / 4) * 4
def footage_offset(x):
    f = lambda k: x[k]; ft = []
    o1 = 16 / 2 ** f("osc1oct")["value"]
    if f("osc1on")["value"] and f("osc1lvl")["frac"] > 0.01: ft.append(o1)
    if f("suboscon")["value"] and f("subosclvl")["frac"] > 0.01: ft.append(2 * o1)
    if f("osc2on")["value"] and f("osc2lvl")["frac"] > 0.05: ft.append(16 / 2 ** f("osc2oct")["value"])
    return round(12 * math.log2(max(ft) / 8))
def pick_tones(rows):
    longs = sorted([r for r in rows if r["audible"] >= 1.0], key=lambda r: r["midi"]); shorts = sorted([r for r in rows if r["audible"] < 1.0], key=lambda r: r["midi"])
    sel = lambda lst, k: [lst[int(round(i * (len(lst) - 1) / max(1, k - 1)))] for i in range(min(k, len(lst)))] if lst else []
    out, seen = [], set()
    for r in sel(longs, 3) + sel(shorts, 3):
        if r["file"] not in seen: seen.add(r["file"]); out.append(r)
    return out
def wait_devices(tp, n=6, tmax=15):
    t0 = time.time()
    while time.time() - t0 < tmax:
        try:
            if L.count(f"{tp} devices 0 chains 0", "devices")["count"] >= n: return
        except Exception: pass
        time.sleep(0.5)
    raise SystemExit("rack did not load on " + tp)
session = {"songs": []}
for key, label, midf, tdir in SONGS:
    x = dec[key]; params, info = map_patch(x)
    tp = L.call("live_set", "create_midi_track", [-1])["result_path"]; ti = int(tp.split()[-1])
    L.set(tp, "name", f"UAD {label}")
    L.browser_load(RACK, track_index=ti); wait_devices(tp)
    ch = f"{tp} devices 0 chains 0"; uad = f"{ch} devices 0"
    L.set(f"{tp} devices 0", "name", f"BZ {label} (UAD)")
    set_params(uad, params); set_params(f"{ch} devices 4", {"Device On": 0})      # EchoBoy off: the references are dry one-shots
    notes, length = midi_notes(os.path.join(HERE, "mid", midf))
    L.call(f"{tp} clip_slots 0", "create_clip", [float(length)]); L.set(f"{tp} clip_slots 0 clip", "name", f"{label} - Bodzin MIDI")
    for i in range(0, len(notes), 200): L.notes_add(f"{tp} clip_slots 0 clip", notes[i:i+200])
    off = footage_offset(x); sel = pick_tones(tones[tdir]); layout, b = [], 0.0; tn = []
    for r in sel:
        win = min(r["audible"], 2.5) + 0.15; gate = 2.5 if r["audible"] >= 2.5 else max(0.05, r["audible"] - (0.05 if r["audible"] < 0.6 else 0.35))
        pitch = int(round(r["midi"])) + off
        tn.append({"pitch": pitch, "start_time": b, "duration": gate / BEAT, "velocity": 110.0})
        layout.append(dict(file=r["file"], start_beat=b, win=win, ref_midi=r["midi"], pitch=pitch)); b += math.ceil((win + 0.5) / BEAT)
    tlen = math.ceil(b / 4) * 4
    L.call(f"{tp} clip_slots 1", "create_clip", [float(tlen)]); L.set(f"{tp} clip_slots 1 clip", "name", f"{label} - tone test ({len(tn)} ref pitches)")
    L.notes_add(f"{tp} clip_slots 1 clip", tn)
    cp = L.call("live_set", "create_audio_track", [-1])["result_path"]; L.set(cp, "name", f"CAP {label}")
    for i in range(40):
        try:
            if L.get(f"{cp} available_input_routing_types {i}", "display_name")["value"] == f"UAD {label}":
                L.set(cp, "input_routing_type", {"__path__": f"{cp} available_input_routing_types {i}"}); break
        except Exception: break
    L.set(cp, "input_routing_channel", {"__path__": f"{cp} available_input_routing_channels 1"})   # Post FX
    L.set(cp, "current_monitoring_state", 2)                                                         # Off
    session["songs"].append(dict(key=key, label=label, track=f"UAD {label}", cap=f"CAP {label}", tones_dir=tdir, layout=layout,
                                 tone_len_beats=tlen, midi_len=length, notes=len(notes), offset=off, info=info, start=params))
    print(label, "| MIDI", len(notes), "notes,", length, "beats | tones", [(l["ref_midi"], l["pitch"]) for l in layout], "| offset", off, flush=True)
json.dump(session, open(os.path.join(HERE, "session.json"), "w"), indent=1)

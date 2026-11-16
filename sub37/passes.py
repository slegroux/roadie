"""Auto-tune every UAD song track against its Bodzin one-shots, all songs per pass (one capture track each).
Per pass: set one candidate move per song, record every tone-test clip at once, score each tone with Sideman's
one-shot distance (RMS-matched), keep moves that improve. State in tune_state.json (resumable), log in tune_log.txt.
usage: passes.py [minutes]"""
import json, os, sys, time, shutil, math, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uadlib import L, set_params, get_params, HERE
from sideman import oneshot  # on sys.path via uadlib (SIDEMAN_SRC)
BEAT = 60 / 124.0; SLOT = 2
MOVES = [("u", "FILTER Cutoff", 0.06), ("u", "FILTER Contour", 0.08), ("u", "FILT Env Decay", 0.06), ("u", "OSC-2 Mix", 0.12),
         ("u", "OSC-3 Mix", 0.12), ("u", "EXT. Mix", 0.04), ("d", "Drive", 0.12), ("u", "AMP Env Decay", 0.08),
         ("u", "FILT Env Attack", 0.05), ("u", "FILT Env Sustain", 0.1), ("u", "FILTER Res", 0.1), ("u", "AMP Env Attack", 0.06),
         ("u", "MOD AMOUNT", 0.1)]
sess = json.load(open(os.path.join(HERE, "session.json")))
STATE = os.path.join(HERE, "tune_state.json"); LOG = open(os.path.join(HERE, "tune_log.txt"), "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
_tracks = None
def track_index(name):      # Live answers ~0.5 s per request with 12 UAD strips loaded: read the names once
    global _tracks
    if _tracks is None:
        n = L.count("live_set", "tracks")["count"]
        _tracks = [x["value"] for x in L.get_batch([{"path": f"live_set tracks {i}", "property": "name"} for i in range(n)])["results"]]
    return _tracks.index(name)
def devs(song):
    t = track_index(song["track"]); ch = f"live_set tracks {t} devices 0 chains 0"
    return t, {"u": f"{ch} devices 0", "d": f"{ch} devices 3"}
def pspec(dev, name, v):
    from uadlib import names
    return {"path": f"{dev} parameters {names(dev)[name]}", "property": "value", "value": float(v)}
def onset(m, sr, rel_db=-40):
    env = np.convolve(m**2, np.ones(64) / 64, "same"); thr = env.max() * 10 ** (rel_db / 10)
    return int(np.argmax(env > thr))
_refs = {}
def ref_seg(song, lay, sr):
    k = (song["tones_dir"], lay["file"], sr)
    if k not in _refs:
        x, rs = sf.read(os.path.join(HERE, "tones", song["tones_dir"], lay["file"]), always_2d=True)
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        if rs != sr: x = oneshot._resample(x, rs, sr)
        o = onset(x.mean(1), sr); _refs[k] = x[o:o + int(lay["win"] * sr)]
    return _refs[k]
def score(song, path):
    x, sr = sf.read(path, always_2d=True)
    m = x.mean(1); first = onset(m, sr); ds = []
    for lay in song["layout"]:
        a = first + int((lay["start_beat"] - song["layout"][0]["start_beat"]) * BEAT * sr) - int(0.01 * sr)
        r = ref_seg(song, lay, sr); seg = x[max(0, a):max(0, a) + len(r) + int(0.05 * sr)]
        if len(seg) < len(r) // 2: ds.append(2.0); continue
        o = onset(seg.mean(1), sr); seg = seg[o:o + len(r)]; n = min(len(seg), len(r))
        if n < 0.03 * sr: ds.append(2.0); continue
        s1, s2 = seg[:n], r[:n]; g = np.sqrt((s2**2).mean() / max((s1**2).mean(), 1e-12)); s1 = s1 * g
        ds.append(oneshot.distance_terms(s1, s2, sr, max_s=lay["win"])["distance"])
    return float(np.mean(ds)), ds
def record(songs):
    """Record every song's tone clip into a new scene's slot on its CAP track at once; return {label: wav copy}."""
    slot = L.call("live_set", "create_scene", [-1])["result_path"].split()[-1]
    caps = {s["label"]: track_index(s["cap"]) for s in songs}; srcs = {s["label"]: track_index(s["track"]) for s in songs}
    L.set_batch([{"path": f"live_set tracks {caps[s['label']]}", "property": "arm", "value": 1} for s in songs])
    ops = [{"op": "call", "path": f"live_set tracks {caps[s['label']]} clip_slots {slot}", "function": "fire", "args": []} for s in songs]
    ops += [{"op": "call", "path": f"live_set tracks {srcs[s['label']]} clip_slots 1", "function": "fire", "args": []} for s in songs]
    # Live starts recording late when a fire also has to start the transport: keep it running, fire on the next
    # bar (clip_trigger_quantization = 1 bar), and stop by song position, not by a timer.
    if not L.get("live_set", "is_playing")["value"]:
        L.call("live_set", "start_playing"); time.sleep(1.0)
    now = L.get("live_set", "current_song_time")["value"]
    L.transaction(ops)
    start = (math.floor(now / 4) + 1) * 4; end = start + max(s["tone_len_beats"] for s in songs) + 2
    while L.get("live_set", "current_song_time")["value"] < end:
        time.sleep(0.5)
    ops = [{"op": "call", "path": f"live_set tracks {caps[s['label']]} clip_slots {slot}", "function": "stop", "args": []} for s in songs]
    ops += [{"op": "call", "path": f"live_set tracks {srcs[s['label']]}", "function": "stop_all_clips", "args": []} for s in songs]
    L.transaction(ops)
    L.set_batch([{"path": f"live_set tracks {caps[s['label']]}", "property": "arm", "value": 0} for s in songs])
    time.sleep(1.5); out = {}
    r = L.get_batch([{"path": f"live_set tracks {caps[s['label']]} clip_slots {slot} clip", "property": "file_path"} for s in songs])["results"]
    for s, x in zip(songs, r):
        if not x.get("ok"): log("no take for", s["label"], x.get("error")); continue
        dst = os.path.join(HERE, "takes", f"{s['key']}_{int(time.time())}.aif"); shutil.copy(x["value"], dst); out[s["label"]] = dst
    return out
def main(minutes):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    songs = sess["songs"]
    for s in songs:
        if s["label"] not in st:
            t, d = devs(s)
            cur = {f"u:{k}": v for k, v in get_params(d["u"], [m[1] for m in MOVES if m[0] == "u"]).items()}
            cur["d:Drive"] = get_params(d["d"], ["Drive"])["Drive"]
            st[s["label"]] = dict(cur=cur, best=None, mi=0, dir=1, scale=1.0, fails=0, hist=[], cand=None)
    t_end = time.time() + minutes * 60; npass = 0
    while time.time() < t_end:
        npass += 1; active = [s for s in songs if st[s["label"]]["scale"] > 0.12]
        if not active: log("all songs converged"); break
        stage = []
        for s in active:                                   # stage one candidate per song (baseline first)
            S = st[s["label"]]; _, d = devs(s)
            if S["best"] is None: S["cand"] = None; continue
            dev, name, step = MOVES[S["mi"]]; key = f"{dev}:{name}"
            nv = min(1.0, max(0.0, S["cur"][key] + S["dir"] * step * S["scale"]))
            if abs(nv - S["cur"][key]) < 1e-4: S["cand"] = ("skip", key, nv); continue
            S["cand"] = (dev, key, nv); stage.append(pspec(d[dev], name, nv))
        if stage: L.set_batch(stage, stop_on_error=False)
        time.sleep(0.3)
        try: takes = record(active)
        except Exception as e:
            log("record failed:", e); time.sleep(3); continue
        revert = []
        for s in active:
            S = st[s["label"]]; _, d = devs(s)
            if s["label"] not in takes: continue
            sc, per = score(s, takes[s["label"]]); c = S["cand"]
            if S["best"] is None:
                S["best"] = sc; S["best_take"] = takes[s["label"]]; S["hist"].append(["baseline", round(sc, 4)]); log(s["label"], "baseline", round(sc, 4), [round(v, 3) for v in per]); continue
            dev, key, nv = c if c else (None, None, None)
            if dev != "skip" and sc < S["best"] - 0.002:
                S["hist"].append([key, round(nv, 3), round(sc, 4)]); log(s["label"], "keep", key, round(S["cur"][key], 3), "->", round(nv, 3), round(S["best"], 4), "->", round(sc, 4))
                S["cur"][key] = nv; S["best"] = sc; S["best_take"] = takes[s["label"]]; S["fails"] = 0; continue
            if dev and dev != "skip": revert.append(pspec(d[dev], key.split(":", 1)[1], S["cur"][key]))
            if S["dir"] == 1: S["dir"] = -1
            else:
                S["dir"] = 1; S["mi"] = (S["mi"] + 1) % len(MOVES); S["fails"] += 1
                if S["mi"] == 0: S["scale"] *= 0.5; log(s["label"], "step scale ->", S["scale"])
        if revert: L.set_batch(revert, stop_on_error=False)
        json.dump(st, open(STATE, "w"), indent=1)
        log(f"pass {npass} done:", ", ".join(f"{k} {v['best']:.3f}" for k, v in st.items() if v["best"] is not None))
    for s in songs:                                      # leave each track on its best values
        _, d = devs(s); S = st[s["label"]]
        set_params(d["u"], {k[2:]: v for k, v in S["cur"].items() if k.startswith("u:")}); set_params(d["d"], {"Drive": S["cur"]["d:Drive"]})
    json.dump(st, open(STATE, "w"), indent=1); log("finished")
if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 40)

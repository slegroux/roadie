"""Shared helpers for the Sub37 -> UADx Minimoog batch (Sideman client, param names, Sub37 -> UAD map)."""
import sys, json, math, os
SIDEMAN = os.environ.get("SIDEMAN_SRC", os.path.expanduser("~/Workspace/omc-worktrees/feat/sideman-sideman-patch-match"))  # oneshot.distance_terms: feat/sideman-patch-match (unmerged)
sys.path.insert(0, SIDEMAN)
from sideman.client import Live
HERE = os.environ.get("SUB37_WORK", os.path.expanduser("~/.omc/reports/sub37-resynth/all"))  # data: decoded.json, tones/, mid/, takes/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sub37_map import map_patch, lfo_norm, wave_idx, oct_idx, LFO_CAL  # noqa: F401
L = Live()
_names = {}
def names(dev):
    if dev not in _names:
        n = L.count(dev, "parameters")["count"]
        r = L.get_batch([{"path": f"{dev} parameters {i}", "property": "name"} for i in range(n)])["results"]
        _names[dev] = {x["value"]: i for i, x in enumerate(r)}
    return _names[dev]
def set_params(dev, vals):
    ix = names(dev)
    specs = [{"path": f"{dev} parameters {ix[k]}", "property": "value", "value": float(v)} for k, v in vals.items()]
    for i in range(0, len(specs), 50):
        L.set_batch(specs[i:i+50], stop_on_error=False)
def get_params(dev, keys=None):
    ix = names(dev); keys = keys or list(ix)
    r = L.get_batch([{"path": f"{dev} parameters {ix[k]}", "property": "value"} for k in keys])["results"]
    return {k: x.get("value") for k, x in zip(keys, r)}

if __name__ == "__main__":
    d = json.load(open(os.path.join(HERE, "decoded.json")))
    for k in d:
        p, info = map_patch(d[k]); print(k, info, {kk: round(vv, 2) for kk, vv in p.items() if kk in ("FILTER Cutoff", "FILTER Contour", "EXT. Mix", "LFO RATE", "MOD AMOUNT", "GLIDE", "OSC-2 Tune")})

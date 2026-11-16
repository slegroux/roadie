import sys, json
import os; sys.path.insert(0, os.environ.get("SIDEMAN_SRC", os.path.expanduser("~/Workspace/omc-worktrees/feat/sideman-sideman-patch-match")))
from sideman.client import Live
L = Live()
dev = sys.argv[1]
n = L.count(dev, "parameters")["count"]
specs=[]
for i in range(n):
    for prop in ("name","value","min","max","is_quantized"):
        specs.append({"path":f"{dev} parameters {i}","property":prop})
r = L.get_batch(specs)
res = r["results"] if isinstance(r,dict) else r
vals = {}
for x in res:
    i = int(x["path"].split()[-1]); vals.setdefault(i,{})[x["property"]] = x.get("value", x.get("error"))
for i in range(n):
    v = vals[i]
    try:
        s = L.call(f"{dev} parameters {i}", "str_for_value", [v["value"]])
        s = s["result"]
    except Exception as e:
        s = "?"
    q = "Q" if v["is_quantized"] else " "
    print(f"{i:3d} {q} {v['name']:<22} {v['value']:.4f}  [{v['min']}-{v['max']}]  {s}")

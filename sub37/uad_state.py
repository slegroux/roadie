import sys, json, struct
def payload(nksf):
    d=open(nksf,'rb').read(); i=12
    while i<len(d):
        cid=d[i:i+4]; n=struct.unpack('<I',d[i+4:i+8])[0]
        if cid==b'PCHK': b=d[i+8:i+8+n]; break
        i+=8+n+(n&1)
    j=b.index(b'{\n'); depth=0
    for k in range(j,len(b)):
        c=b[k:k+1]
        if c==b'{': depth+=1
        elif c==b'}':
            depth-=1
            if depth==0: return json.loads(b[j:k+1])
if __name__=="__main__":
    p=payload(sys.argv[1])
    for k,v in p["controls"].items(): print(f"{k:<18}{v.get('real_value')}")
    print([k for k in p if k!="controls"])

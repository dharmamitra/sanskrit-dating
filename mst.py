import json, collections, math, sys, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
cache={}
def W_(t):
    if t in cache: return cache[t]
    v=m2w(t) or t; cache[t]=v; return v
we=collections.defaultdict(float)
with open('edges_raw.tsv') as fh:
    next(fh)
    for line in fh:
        r,p,c,_,_=line.rstrip('\n').split('\t'); a,b=W_(r),W_(p)
        if a==b: continue
        a,b=sorted((a,b)); we[(a,b)]+=float(c)
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w
# load final dates for annotation
date={}
with open('dated_final2.tsv') as fh:
    next(fh)
    for line in fh:
        p=line.rstrip('\n').split('\t'); date[p[1]]=p[0]
def title(w): return meta.get(w,{}).get('title','')[:34]

bycoll=collections.defaultdict(list)
for (x,y) in we:
    pass
works=sorted({w for e in we for w in e})
for w in works: bycoll[coll(w)].append(w)

OUT=open('mst_edges.tsv','w')
OUT.write("collection\ttextA\tdateA\ttitleA\ttextB\tdateB\ttitleB\tshared_parallels\n")
summary=[]
for c,ws in sorted(bycoll.items()):
    ws=sorted(ws); n=len(ws)
    if n<5: continue
    si={w:i for i,w in enumerate(ws)}
    # max spanning tree = MST on distance = 1/log1p(count)
    R=[];C=[];V=[]
    for (x,y),cnt in we.items():
        if x in si and y in si:
            d=1.0/math.log1p(cnt)
            R.append(si[x]);C.append(si[y]);V.append(d)
    if not V: continue
    M=sp.csr_matrix((V,(R,C)),(n,n))
    mst=minimum_spanning_tree(M).tocoo()
    ncomp,_=connected_components(M,directed=False)
    edges=[]
    for i,j,d in zip(mst.row,mst.col,mst.data):
        cnt=round(math.expm1(1.0/d))
        edges.append((cnt,ws[i],ws[j]))
    edges.sort(reverse=True)
    for cnt,a,b in edges:
        OUT.write(f"{c}\t{a}\t{date.get(a,'')}\t{title(a)}\t{b}\t{date.get(b,'')}\t{title(b)}\t{cnt}\n")
    summary.append((c,n,len(edges),ncomp))
OUT.close()
print(f"{'coll':8s} {'texts':>5} {'MSTedges':>8} {'components':>10}",file=sys.stderr)
for c,n,e,nc in sorted(summary,key=lambda s:-s[1]):
    print(f"{c:8s} {n:5d} {e:8d} {nc:10d}",file=sys.stderr)
print(f"\nwrote mst_edges.tsv ({len(summary)} collections)",file=sys.stderr)

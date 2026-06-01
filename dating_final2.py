import json, collections, math, sys, statistics, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve

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
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w
def coarse(w): m=re.match(r'([A-Za-z]+)',coll(w)); return m.group(1) if m else coll(w)
works=sorted({w for e in we for w in e})
dated={w:mid(w) for w in works if mid(w) is not None}
bycoll=collections.defaultdict(list)
for w in works: bycoll[coll(w)].append(w)
# coarse medians + global
crs=collections.defaultdict(list); allv=[]
for w,v in dated.items(): crs[coarse(w)].append(v); allv.append(v)
gmed=statistics.median(allv)

def within_cat_dates(ws, lam=0.05):
    """propagate dates within a collection subgraph; returns dict w->est for undated, or None."""
    anc=[w for w in ws if w in dated]
    if len(anc)<2: return None
    cmed=statistics.median([dated[w] for w in anc])
    sub=sorted(ws); si={w:i for i,w in enumerate(sub)}; n=len(sub)
    R=[];C=[];V=[]
    for (x,y),c in we.items():
        if x in si and y in si:
            i,j=si[x],si[y]; w=math.log1p(c); R+=[i,j];C+=[j,i];V+=[w,w]
    if not V: return {w:cmed for w in ws if w not in dated}
    Wm=sp.csr_matrix((V,(R,C)),(n,n)); deg=np.asarray(Wm.sum(1)).ravel()
    lab=np.zeros(n,bool);y=np.zeros(n)
    for w in anc: lab[si[w]]=True; y[si[w]]=dated[w]
    U=np.where(~lab)[0];L=np.where(lab)[0]
    dinv=np.zeros(n);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+lam*sp.identity(len(U))
    xU=spsolve(A.tocsc(),P[U][:,L]@y[L]+lam*cmed);x=y.copy();x[U]=xU
    return {sub[i]:x[i] for i in U}

# global orphan harmonic (for collections with <2 anchors)
def global_harmonic(lam=0.02):
    idx={w:i for i,w in enumerate(works)}; N=len(works)
    R=[];C=[];V=[]
    for (x,y),c in we.items():
        i,j=idx[x],idx[y]; w=math.log1p(c); R+=[i,j];C+=[j,i];V+=[w,w]
    Wm=sp.csr_matrix((V,(R,C)),(N,N)); deg=np.asarray(Wm.sum(1)).ravel()
    lab=np.zeros(N,bool);y=np.zeros(N)
    for w,v in dated.items(): lab[idx[w]]=True;y[idx[w]]=v
    U=np.where(~lab)[0];L=np.where(lab)[0]
    dinv=np.zeros(N);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+lam*sp.identity(len(U))
    xU=spsolve(A.tocsc(),P[U][:,L]@y[L]+lam*gmed);x=y.copy();x[U]=xU
    return {works[i]:x[i] for i in range(N)}, idx
ghx,idx=global_harmonic()

est={}; method={}
for c,ws in bycoll.items():
    wd=within_cat_dates(ws)
    if wd is not None:
        cmed=statistics.median([dated[w] for w in ws if w in dated])
        for w in ws:
            if w in dated: continue
            est[w]=wd[w]; method[w]=f'within-cat:{c}'
    else:
        cw=coarse(ws[0])
        if cw in crs and len(crs[cw])>=2:
            cm=statistics.median(crs[cw])
            for w in ws:
                if w in dated: continue
                est[w]=cm; method[w]=f'coarse:{cw}'
        else:
            for w in ws:
                if w in dated: continue
                est[w]=ghx[w]; method[w]='graph-orphan'

rows=[]
deg_all=collections.Counter()
for (x,y),c in we.items(): deg_all[x]+=1; deg_all[y]+=1
for w in works:
    m=meta.get(w,{})
    if w in dated:
        rows.append((dated[w],w,m.get('title',''),coll(w),'anchor',m.get('nb',''),m.get('na',''),deg_all[w]))
    else:
        rows.append((est[w],w,m.get('title',''),coll(w),method[w],'','',deg_all[w]))
rows.sort()
with open('dated_final2.tsv','w') as f:
    f.write("est_date\twork\ttitle\tcollection\tmethod\tnb\tna\tdeg\n")
    for r in rows:
        f.write(f"{r[0]:.0f}\t"+"\t".join(str(x) for x in r[1:])+"\n")
mc=collections.Counter(v.split(':')[0] for v in method.values())
print("methods:",dict(mc),file=sys.stderr)
print(f"wrote dated_final2.tsv ({len(works)} works)",file=sys.stderr)

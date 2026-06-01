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
works=sorted({w for e in we for w in e}); idx={w:i for i,w in enumerate(works)}; N=len(works)
dated={w:mid(w) for w in works if mid(w) is not None}
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w
def coarse(w): m=re.match(r'([A-Za-z]+)',coll(w)); return m.group(1) if m else coll(w)

# collection stats (median + spread)
full=collections.defaultdict(list); crs=collections.defaultdict(list); allv=[]
for w,v in dated.items(): full[coll(w)].append(v); crs[coarse(w)].append(v); allv.append(v)
gm=statistics.median(allv)
def stats(vs): return statistics.median(vs), (min(vs),max(vs)), len(vs)

# graph harmonic (for orphan fallback only), regularized to global median
R=[];C=[];V=[]
for (x,y),c in we.items():
    i,j=idx[x],idx[y]; w=math.log1p(c); R+=[i,j];C+=[j,i];V+=[w,w]
Wm=sp.csr_matrix((V,(R,C)),(N,N)); deg=np.asarray(Wm.sum(1)).ravel()
def harmonic(dl,lam=0.02,target=gm):
    lab=np.zeros(N,bool);y=np.zeros(N)
    for w,v in dl.items(): lab[idx[w]]=True;y[idx[w]]=v
    U=np.where(~lab)[0];L=np.where(lab)[0]
    dinv=np.zeros(N);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+lam*sp.identity(len(U))
    xU=spsolve(A.tocsc(),P[U][:,L]@y[L]+lam*target);x=y.copy();x[U]=xU;return x
hx=harmonic(dated)

rows=[]
for i,w in enumerate(works):
    m=meta.get(w,{})
    if w in dated:
        rows.append((w,m.get('title',''),coll(w),'anchor',mid(w),'',m.get('nb',''),m.get('na',''),'',int(deg[i])))
        continue
    fw=coll(w); cw=coarse(w)
    if fw in full and len(full[fw])>=2:
        md,(lo,hi),n=stats(full[fw]); method=f'collection:{fw}(n={n})'; est=md; rng=f'{lo:.0f}..{hi:.0f}'
    elif cw in crs and len(crs[cw])>=2:
        md,(lo,hi),n=stats(crs[cw]); method=f'coarse:{cw}(n={n})'; est=md; rng=f'{lo:.0f}..{hi:.0f}'
    else:
        est=hx[i]; method='graph(no collection anchor)'; rng=''
    rows.append((w,m.get('title',''),coll(w),'inferred',est,method,'','',rng,int(deg[i])))

rows.sort(key=lambda r:r[4])
with open('dated_final.tsv','w') as f:
    f.write("work\ttitle\tcollection\tsource\test_date\tmethod\tnb\tna\tcoll_range\tdeg\n")
    for r in rows:
        r=list(r); r[4]=f"{r[4]:.0f}"
        f.write("\t".join(str(x) for x in r)+"\n")

n_coll=sum(1 for r in rows if r[5].startswith('collection'))
n_coarse=sum(1 for r in rows if r[5].startswith('coarse'))
n_graph=sum(1 for r in rows if r[5].startswith('graph'))
print(f"works {N}: anchors {len(dated)}, inferred {N-len(dated)}",file=sys.stderr)
print(f"  inferred via full-collection: {n_coll}",file=sys.stderr)
print(f"  inferred via coarse-collection: {n_coarse}",file=sys.stderr)
print(f"  inferred via graph (orphans): {n_graph}",file=sys.stderr)
print("wrote dated_final.tsv",file=sys.stderr)

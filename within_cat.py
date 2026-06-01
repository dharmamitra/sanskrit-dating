import json, collections, math, sys, statistics, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr

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
works=sorted({w for e in we for w in e})
dated={w:mid(w) for w in works if mid(w) is not None}

# group works by collection
bycoll=collections.defaultdict(list)
for w in works: bycoll[coll(w)].append(w)

def harmonic_sub(sub_works, dl, lam=0.05):
    """harmonic propagation on the subgraph induced by sub_works."""
    sub=sorted(sub_works); si={w:i for i,w in enumerate(sub)}; n=len(sub)
    R=[];C=[];V=[]
    for (x,y),c in we.items():
        if x in si and y in si:
            i,j=si[x],si[y]; w=math.log1p(c); R+=[i,j];C+=[j,i];V+=[w,w]
    if not V: return None,si,sub
    Wm=sp.csr_matrix((V,(R,C)),(n,n)); deg=np.asarray(Wm.sum(1)).ravel()
    tgt=statistics.median([v for w,v in dl.items() if w in si]) if any(w in si for w in dl) else statistics.median(list(dl.values()))
    lab=np.zeros(n,bool);y=np.zeros(n)
    for w,v in dl.items():
        if w in si: lab[si[w]]=True; y[si[w]]=v
    U=np.where(~lab)[0];L=np.where(lab)[0]
    if len(L)==0 or len(U)==0: return None,si,sub
    dinv=np.zeros(n);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+lam*sp.identity(len(U))
    xU=spsolve(A.tocsc(),P[U][:,L]@y[L]+lam*tgt);x=y.copy();x[U]=xU
    return x,si,sub

print(f"{'coll':8s} {'nanc':>4} {'flatMAE':>7} {'graphMAE':>8} {'flatSpr':>7} {'graphSpr':>8}  title-ish",file=sys.stderr)
agg_flat=[];agg_graph=[];allpairs=[]
for c,ws in sorted(bycoll.items()):
    anc=[w for w in ws if w in dated]
    if len(anc)<6: continue
    # within-collection LOO
    truth=[];flatp=[];graphp=[]
    for held in anc:
        dl={w:dated[w] for w in anc if w!=held}
        flat=statistics.median(list(dl.values()))
        x,si,sub=harmonic_sub(ws, dl)
        if x is not None and held in si and not np.isnan(x[si[held]]):
            g=x[si[held]]
        else:
            g=flat
        truth.append(dated[held]); flatp.append(flat); graphp.append(g)
    truth=np.array(truth);flatp=np.array(flatp);graphp=np.array(graphp)
    fmae=np.abs(flatp-truth).mean(); gmae=np.abs(graphp-truth).mean()
    fspr=spearmanr(flatp,truth).statistic if len(set(flatp))>1 else float('nan')
    gspr=spearmanr(graphp,truth).statistic if len(set(graphp))>1 else float('nan')
    print(f"{c:8s} {len(anc):4d} {fmae:7.0f} {gmae:8.0f} {fspr:7.2f} {gspr:8.2f}  {meta.get(anc[0],{}).get('title','')[:20]}",file=sys.stderr)
    agg_flat += list(np.abs(flatp-truth)); agg_graph += list(np.abs(graphp-truth))
    for t,g in zip(truth,graphp): allpairs.append((t,g))
print(f"\npooled (collections w/ >=6 anchors): flat MAE={np.mean(agg_flat):.0f}  within-cat-graph MAE={np.mean(agg_graph):.0f}",file=sys.stderr)
ts=[p[0] for p in allpairs]; gs=[p[1] for p in allpairs]
print(f"pooled within-cat-graph Spearman vs truth: {spearmanr(gs,ts).statistic:.3f}",file=sys.stderr)

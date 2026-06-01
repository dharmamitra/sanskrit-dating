import json, collections, math, sys, statistics
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingRegressor

meta = json.load(open('meta.json'))
meta_keys = sorted(meta.keys(), key=len, reverse=True)
def map_to_work(t):
    for k in meta_keys:
        if t==k or t.startswith(k): return k
    return None
nodes=[]
with open('nodes_raw.tsv') as fh:
    next(fh)
    for line in fh: nodes.append(line.split('\t',1)[0])
n2w={t:(map_to_work(t) or t) for t in nodes}
wedges=collections.defaultdict(float)
with open('edges_raw.tsv') as fh:
    next(fh)
    for line in fh:
        r,p,c,_,_=line.rstrip('\n').split('\t')
        a,b=n2w.get(r,r),n2w.get(p,p)
        if a==b: continue
        a,b=sorted((a,b)); wedges[(a,b)]+=float(c)
def mid(w):
    m=meta.get(w);
    return None if (not m or m['nb'] is None or m['na'] is None) else 0.5*(m['nb']+m['na'])
works=sorted({w for e in wedges for w in e})
idx={w:i for i,w in enumerate(works)}; N=len(works)
dated={w:mid(w) for w in works if mid(w) is not None}

# adjacency with weights
adj=collections.defaultdict(list)
for (a,b),c in wedges.items():
    w=math.log1p(c); adj[a].append((b,w)); adj[b].append((a,w))

def feats(node, dated_lookup):
    # distribution of dated-neighbor dates, weighted
    ds=[]; ws=[]
    for (nb,w) in adj[node]:
        if nb in dated_lookup:
            ds.append(dated_lookup[nb]); ws.append(w)
    deg=len(adj[node]); nd=len(ds)
    if nd==0:
        return [np.nan]*9+[math.log1p(deg),0]
    ds=np.array(ds); ws=np.array(ws); ws/=ws.sum()
    order=np.argsort(ds); dss=ds[order]; wss=ws[order]; cw=np.cumsum(wss)
    def wq(q): return float(dss[np.searchsorted(cw,q)] if q<cw[-1] else dss[-1])
    wmean=float((ds*ws).sum())
    return [wmean, ds.min(), ds.max(), wq(.1), wq(.25), wq(.5), wq(.75), wq(.9),
            float(np.sqrt(((ds-wmean)**2*ws).sum())), math.log1p(deg), nd]

anchors=[w for w in dated if adj[w]]
# LOO: for each anchor, build features from OTHER anchors only, train GBM on rest
# (train once per fold is expensive; instead do honest CV: train on fold's train anchors)
from sklearn.model_selection import KFold
A=np.array(anchors)
y=np.array([dated[w] for w in anchors])
kf=KFold(n_splits=10, shuffle=False)
preds=np.zeros(len(A))
for tr,te in kf.split(A):
    train_set=set(A[tr])
    dl={w:dated[w] for w in train_set}          # only training anchors are "known"
    Xtr=np.array([feats(w,dl) for w in A[tr]])
    Xte=np.array([feats(w,dl) for w in A[te]])
    m=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8)
    # impute nan with col median
    med=np.nanmedian(Xtr,0);
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xte=np.where(np.isnan(Xte),med,Xte)
    m.fit(Xtr,y[tr]); preds[te]=m.predict(Xte)
err=np.abs(preds-y)
print(f"FEATURE GBM 10-fold CV (n={len(A)})",file=sys.stderr)
print(f"  MAE            = {err.mean():6.1f} y",file=sys.stderr)
print(f"  median abs err = {np.median(err):6.1f} y",file=sys.stderr)
for t in (50,100,200,300,500):
    print(f"  within {t:4d}y    : {100*np.mean(err<=t):4.1f}%",file=sys.stderr)
gm=y.mean(); print(f"  baseline(mean {gm:.0f}) MAE={np.abs(y-gm).mean():.1f}",file=sys.stderr)
# feature importances on full fit
dl=dict(dated); X=np.array([feats(w,dl) for w in A]); med=np.nanmedian(X,0); X=np.where(np.isnan(X),med,X)
m=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8).fit(X,y)
names=['wmean','min','max','q10','q25','q50','q75','q90','wstd','logdeg','n_dated']
imp=sorted(zip(names,m.feature_importances_),key=lambda z:-z[1])
print("  feat importance:", ", ".join(f"{n}={v:.2f}" for n,v in imp),file=sys.stderr)
det=sorted(zip(A,y,preds),key=lambda z:-abs(z[2]-z[1]))
print("\n  worst 12:",file=sys.stderr)
for w,t,p in det[:12]:
    print(f"   {w:26s} true={t:6.0f} pred={p:6.0f} err={abs(p-t):5.0f}  {meta.get(w,{}).get('title','')[:24]}",file=sys.stderr)

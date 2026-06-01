import pickle, json, collections, statistics, sys
import numpy as np
from scipy.stats import spearmanr

D=pickle.load(open('vocab2.pkl','rb')); vocab=D['vocab']; glob=D['glob']
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
wv=collections.defaultdict(collections.Counter)
for t,c in vocab.items(): wv[m2w(t)].update(c)
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w

MFW=[w for w,_ in glob.most_common(300)]
def vec(c):
    tot=sum(c.values()) or 1; return np.array([c[w]/tot for w in MFW])
works=[w for w in wv if sum(wv[w].values())>=1000]
dated={w:mid(w) for w in works if mid(w) is not None}
bycoll=collections.defaultdict(list)
for w in works: bycoll[coll(w)].append(w)

# global z-space
Xg=np.array([vec(wv[w]) for w in works]); mug=Xg.mean(0); sdg=Xg.std(0)+1e-9
Zg={w:(vec(wv[w])-mug)/sdg for w in works}
# per-collection z-space
Zc={}
for c,ws in bycoll.items():
    X=np.array([vec(wv[w]) for w in ws]); mu=X.mean(0); sd=X.std(0)+1e-9
    for w in ws: Zc[w]=(vec(wv[w])-mu)/sd

def knn_predict(target, candidates, zspace, k=5):
    if not candidates: return None
    zt=zspace[target]
    d=[(np.abs(zspace[c]-zt).mean(),c) for c in candidates]
    d.sort(); d=d[:k]
    wt=np.array([1/(x+1e-6) for x,_ in d]); vals=np.array([dated[c] for _,c in d])
    return float(np.sum(wt*vals)/wt.sum())

def predict(w, hold=None):
    c=coll(w); co=[a for a in bycoll[c] if a in dated and a!=w and a!=hold]
    if len(co)>=2:
        return knn_predict(w, co, Zc, k=min(5,len(co))), f'within-cat:{c}'
    allanc=[a for a in dated if a!=w and a!=hold]
    return knn_predict(w, allanc, Zg, k=5), 'cross-cat-vocab'

# headline LOO
errs=[];P=[];T=[]
for h in dated:
    p,_=predict(h, hold=h)
    if p is None: continue
    errs.append(abs(p-dated[h]));P.append(p);T.append(dated[h])
errs=np.array(errs)
print(f"HEADLINE LOO (vocab two-level) n={len(errs)}",file=sys.stderr)
print(f"  MAE={errs.mean():.1f}  median={np.median(errs):.1f}  Spearman={spearmanr(P,T).statistic:.3f}",file=sys.stderr)
for t in (50,100,200,300): print(f"  within {t}y: {100*np.mean(errs<=t):.1f}%",file=sys.stderr)
print(f"  [compare] within-cat GRAPH MAE 157 | collection-prior 232 | undirected graph 296",file=sys.stderr)

# full output
rows=[]; method={}
for w in works:
    if w in dated:
        rows.append((dated[w],w,meta.get(w,{}).get('title',''),coll(w),'anchor')); continue
    p,m=predict(w)
    if p is None: p=statistics.median(list(dated.values())); m='global-fallback'
    rows.append((p,w,meta.get(w,{}).get('title',''),coll(w),m)); method[w]=m
rows.sort()
with open('dated_final3.tsv','w') as f:
    f.write("est_date\twork\ttitle\tcollection\tmethod\n")
    for r in rows: f.write(f"{r[0]:.0f}\t"+"\t".join(map(str,r[1:]))+"\n")
mc=collections.Counter(method.values())
print(f"\nmethods for inferred: {dict(mc)}",file=sys.stderr)
print(f"wrote dated_final3.tsv ({len(works)} works with >=1000 tokens)",file=sys.stderr)

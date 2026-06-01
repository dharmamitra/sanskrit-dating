import json, collections, pickle, sys
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])

# ---- morph (file->work, token-weighted) ----
with open('morph_features.tsv') as fh:
    hdr=fh.readline().rstrip('\n').split('\t')[1:]
    agg=collections.defaultdict(lambda:collections.defaultdict(float)); ntok=collections.defaultdict(float)
    for line in fh:
        p=line.rstrip('\n').split('\t'); d=dict(zip(hdr,map(float,p[1:]))); w=m2w(p[0]); n=d['n_tokens']
        ntok[w]+=n
        for k,v in d.items():
            if k!='n_tokens': agg[w][k]+=v*n
mfeats=[c for c in hdr if c!='n_tokens']
works=[w for w in agg if ntok[w]>=1000]
idx={w:i for i,w in enumerate(works)}; N=len(works)
M=np.array([[agg[w][c]/ntok[w] for c in mfeats] for w in works])

# ---- fw-grams (file->work) -> tfidf -> svd ----
fg=pickle.load(open('fwgrams.pkl','rb'))
wg=collections.defaultdict(collections.Counter)
for t,c in fg.items():
    if m2w(t) in idx: wg[m2w(t)].update(c)
gdf=collections.Counter()
for w in wg:
    for g in wg[w]: gdf[g]+=1
gvoc=[g for g,d in gdf.items() if d>=10]; gi={g:i for i,g in enumerate(gvoc)}
R=[];C=[];V=[]
for w in wg:
    r=idx[w]; tot=sum(wg[w].values()) or 1
    for g,n in wg[w].items():
        if g in gi: R.append(r);C.append(gi[g]);V.append(n/tot)
G=sp.csr_matrix((V,(R,C)),shape=(N,len(gvoc)))
gidf=np.log((N+1)/(np.array([gdf[g] for g in gvoc])+1))
G=normalize(G.multiply(gidf).tocsr())
GS=TruncatedSVD(50,random_state=0).fit_transform(G) if len(gvoc)>50 else G.toarray()
print(f"works={N}  morph feats={len(mfeats)}  fw-grams={len(gvoc)}->SVD{GS.shape[1]}",file=sys.stderr)

dates=np.array([mid(w) if mid(w) is not None else np.nan for w in works])
anc=np.where(~np.isnan(dates))[0]; y=dates[anc]
print(f"anchors: {len(anc)}",file=sys.stderr)

def cv(model_f,Xm,name):
    pred=cross_val_predict(model_f(),Xm[anc],y,cv=KFold(10,shuffle=True,random_state=0))
    err=np.abs(pred-y)
    print(f"   {name:30s} MAE={err.mean():6.1f} med={np.median(err):6.1f} Spearman={spearmanr(pred,y).statistic:.3f} <=300y:{100*np.mean(err<=300):.0f}%",file=sys.stderr)
    return pred
gb=lambda: GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0)
ridge=lambda: RidgeCV(alphas=np.logspace(-3,3,25))
Mz=StandardScaler().fit_transform(M)
COMB=np.hstack([Mz,GS])
print(f"\nanchored regression (10-fold CV, NO category):",file=sys.stderr)
cv(gb,M,"GBM / morph only")
cv(ridge,GS,"ridge / fw-grams only")
cv(gb,GS,"GBM / fw-grams only")
cv(gb,COMB,"GBM / morph + fw-grams")
print(f"   baseline(mean) MAE={np.abs(y-y.mean()).mean():.1f}",file=sys.stderr)

# ---- final: train best (GBM combined) on all anchors, date everything ----
model=gb().fit(COMB[anc],y)
pred=model.predict(COMB)
rows=sorted(zip(pred,works),key=lambda z:z[0])
with open('dated_morph.tsv','w') as f:
    f.write("est_date\twork\tsource\ttitle\n")
    for p,w in rows:
        src='anchor' if not np.isnan(dates[idx[w]]) else 'inferred'
        f.write(f"{p:.0f}\t{w}\t{src}\t{meta.get(w,{}).get('title','')}\n")
print(f"\nwrote dated_morph.tsv ({N} works, morphology+fw-grams, category-free)",file=sys.stderr)

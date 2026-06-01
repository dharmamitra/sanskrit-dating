import json, collections, pickle, sys, re
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def meta_mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])

# vedic anchors (longest-prefix match)
van=[]
with open('vedic_anchors.tsv') as fh:
    next(fh)
    for line in fh:
        p=line.rstrip('\n').split('\t'); van.append((p[0],float(p[1]),float(p[2])))
van.sort(key=lambda x:-len(x[0]))
def vedic_mid(w):
    for pre,nb,na in van:
        if w.startswith(pre): return .5*(nb+na)
    return None

# group key (collapse duplicated multi-file works to prevent CV leakage)
COLLAPSE=['SA_GV01_rvpp','SA_GV01_rv_hn','SA_GV01_rv','SA_GV03_sb','SA_GV05_brup','SA_GV05_chup',
          'SA_GV05_aitup','SA_GV05_prasup','SA_GV05_chupsb','SA_GV02_gop']
def group(w):
    for c in COLLAPSE:
        if w.startswith(c): return c
    return w

# ---- features (morph + fw-grams), reuse build ----
with open('morph_features.tsv') as fh:
    hdr=fh.readline().rstrip('\n').split('\t')[1:]
    agg=collections.defaultdict(lambda:collections.defaultdict(float)); ntok=collections.defaultdict(float)
    for line in fh:
        p=line.rstrip('\n').split('\t'); d=dict(zip(hdr,map(float,p[1:]))); w=m2w(p[0]); n=d['n_tokens']
        ntok[w]+=n
        for k,v in d.items():
            if k!='n_tokens': agg[w][k]+=v*n
mfeats=[c for c in hdr if c!='n_tokens']
works=[w for w in agg if ntok[w]>=1000]; idx={w:i for i,w in enumerate(works)}; N=len(works)
M=np.array([[agg[w][c]/ntok[w] for c in mfeats] for w in works])
fg=pickle.load(open('fwgrams.pkl','rb')); wg=collections.defaultdict(collections.Counter)
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
G=sp.csr_matrix((V,(R,C)),(N,len(gvoc)))
gidf=np.log((N+1)/(np.array([gdf[g] for g in gvoc])+1)); G=normalize(G.multiply(gidf).tocsr())
GS=TruncatedSVD(50,random_state=0).fit_transform(G)
COMB=np.hstack([StandardScaler().fit_transform(M),GS])

# ---- anchor sets ----
mdate=np.array([meta_mid(w) if meta_mid(w) is not None else np.nan for w in works])
vdate=np.array([vedic_mid(w) if vedic_mid(w) is not None else np.nan for w in works])
groups=np.array([group(w) for w in works])

def run(label, use_vedic):
    y=mdate.copy()
    if use_vedic:
        for i in range(N):
            if np.isnan(y[i]) and not np.isnan(vdate[i]): y[i]=vdate[i]
    anc=np.where(~np.isnan(y))[0]
    pred=cross_val_predict(GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0),
                           COMB[anc], y[anc], cv=GroupKFold(10), groups=groups[anc])
    err=np.abs(pred-y[anc])
    print(f"\n=== {label} (n_anchors={len(anc)}, GroupKFold CV) ===",file=sys.stderr)
    print(f"  MAE={err.mean():.1f} med={np.median(err):.1f} Spearman={spearmanr(pred,y[anc]).statistic:.3f} <=300y:{100*np.mean(err<=300):.0f}%",file=sys.stderr)
    # report recovery of held-out Vedic groups
    if use_vedic:
        vg=[(works[i],y[i],pred[j]) for j,i in enumerate(anc) if not np.isnan(vdate[i])]
        # collapse to group means
        bygrp=collections.defaultdict(list)
        for w,t,p in vg: bygrp[group(w)].append((t,p))
        print("  held-out Vedic-group predictions (true-range-mid vs CV-pred):",file=sys.stderr)
        for g,lst in sorted(bygrp.items(),key=lambda kv:kv[1][0][0]):
            t=lst[0][0]; pm=np.mean([p for _,p in lst])
            print(f"    {g:22s} true~{t:6.0f}  pred~{pm:6.0f}  (n={len(lst)})",file=sys.stderr)
    return y

run("BASELINE (metadata anchors only)", False)
y=run("WITH Vedic anchors", True)

# ---- final model trained on all anchors -> redate everything ----
anc=np.where(~np.isnan(y))[0]
model=GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0).fit(COMB[anc],y[anc])
pred=model.predict(COMB)
with open('dated_morph_vedic.tsv','w') as f:
    f.write("est_date\twork\tsource\ttitle\n")
    for p,w in sorted(zip(pred,works)):
        src='meta-anchor' if not np.isnan(mdate[idx[w]]) else ('vedic-anchor' if not np.isnan(vdate[idx[w]]) else 'inferred')
        f.write(f"{p:.0f}\t{w}\t{src}\t{meta.get(w,{}).get('title','')}\n")
print(f"\nwrote dated_morph_vedic.tsv",file=sys.stderr)

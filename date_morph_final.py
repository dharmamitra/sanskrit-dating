import json, collections, pickle, numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def meta_iv(w):
    m=meta.get(w)
    if m and m['nb'] is not None and m['na'] is not None:
        a,b=float(m['nb']),float(m['na']); return (min(a,b),max(a,b))
    return None
van=[]
for line in open('vedic_anchors.tsv').read().splitlines()[1:]:
    p=line.split('\t'); van.append((p[0],float(p[1]),float(p[2])))
van.sort(key=lambda x:-len(x[0]))
def vedic_iv(w):
    for pre,nb,na in van:
        if w.startswith(pre): return (nb,na)
    return None
COLLAPSE=['SA_GV01_rvpp','SA_GV01_rv_hn','SA_GV01_rv','SA_GV03_sb','SA_GV05_brup','SA_GV05_chup','SA_GV05_aitup','SA_GV05_prasup','SA_GV05_chupsb','SA_GV02_gop']
def group(w):
    for c in COLLAPSE:
        if w.startswith(c): return c
    return w

# ---- features (morph + fw-grams) ----
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
gvoc=[g for g,dd in gdf.items() if dd>=10]; gi={g:i for i,g in enumerate(gvoc)}
R=[];C=[];V=[]
for w in wg:
    r=idx[w]; tot=sum(wg[w].values()) or 1
    for g,n in wg[w].items():
        if g in gi: R.append(r);C.append(gi[g]);V.append(n/tot)
G=sp.csr_matrix((V,(R,C)),(N,len(gvoc))); gidf=np.log((N+1)/(np.array([gdf[g] for g in gvoc])+1))
G=normalize(G.multiply(gidf).tocsr()); GS=TruncatedSVD(50,random_state=0).fit_transform(G)
COMB=np.hstack([StandardScaler().fit_transform(M),GS])

# ---- anchors: intervals, midpoint targets, width weights ----
lo=np.full(N,np.nan); hi=np.full(N,np.nan); src=['inferred']*N
for i,w in enumerate(works):
    iv=meta_iv(w)
    if iv: lo[i],hi[i]=iv; src[i]='meta-anchor'
    else:
        iv=vedic_iv(w)
        if iv: lo[i],hi[i]=iv; src[i]='vedic-anchor'
anc=np.where(~np.isnan(lo))[0]
mid=0.5*(lo+hi); width=hi-lo
# width weight: inverse width, floored so a 10y anchor can't dominate
wt=1.0/np.clip(width,50,None)
groups=np.array([group(w) for w in works])
gb=lambda: GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0)

# ---- CV: weighted vs unweighted, honest (no clamp on held-out) ----
def cv(weighted):
    pred=np.zeros(len(anc))
    gk=GroupKFold(10)
    A=anc; g=groups[A]
    for tr,te in gk.split(COMB[A],mid[A],g):
        m=gb()
        sw=wt[A][tr] if weighted else None
        m.fit(COMB[A][tr],mid[A][tr],sample_weight=sw)
        pred[te]=m.predict(COMB[A][te])
    L,H,MID=lo[anc],hi[anc],mid[anc]
    inside=(pred>=L)&(pred<=H)
    edge=np.where(inside,0.0,np.minimum(np.abs(pred-L),np.abs(pred-H)))
    return dict(mae_mid=np.abs(pred-MID).mean(), inrange=inside.mean()*100,
                edge_mean=edge.mean(), edge_med=np.median(edge),
                spr=spearmanr(pred,MID).statistic)
for wtd in (False,True):
    r=cv(wtd)
    print(f"{'WIDTH-WEIGHTED' if wtd else 'unweighted   '}: MAE_mid={r['mae_mid']:.0f}  in-range={r['inrange']:.0f}%  "
          f"interval-err mean={r['edge_mean']:.0f}/med={r['edge_med']:.0f}  Spearman={r['spr']:.3f}")

# ---- final model: width-weighted on all anchors; predict; clamp anchors into range ----
model=gb().fit(COMB[anc],mid[anc],sample_weight=wt[anc])
raw=model.predict(COMB)
est=raw.copy()
nclamped=0
for i in anc:                       # constrain range-texts into their range
    c=min(max(raw[i],lo[i]),hi[i])
    if c!=raw[i]: nclamped+=1
    est[i]=c
print(f"\nanchored works clamped into range: {nclamped}/{len(anc)} (rest already in-range)")
order=np.argsort(est)
with open('dated_morph_final.tsv','w') as f:
    f.write("est_date\twork\tsource\tnb\tna\tmodel_raw\ttitle\n")
    for i in order:
        nb='' if np.isnan(lo[i]) else f"{lo[i]:.0f}"; na='' if np.isnan(hi[i]) else f"{hi[i]:.0f}"
        f.write(f"{est[i]:.0f}\t{works[i]}\t{src[i]}\t{nb}\t{na}\t{raw[i]:.0f}\t{meta.get(works[i],{}).get('title','')}\n")
print(f"wrote dated_morph_final.tsv ({N} works)")

import json, collections, pickle, numpy as np, scipy.sparse as sp
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
# features
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
G=sp.csr_matrix((V,(R,C)),(N,len(gvoc))); gidf=np.log((N+1)/(np.array([gdf[g] for g in gvoc])+1))
G=normalize(G.multiply(gidf).tocsr()); GS=TruncatedSVD(50,random_state=0).fit_transform(G)
COMB=np.hstack([StandardScaler().fit_transform(M),GS])
# anchors with intervals
lo=np.full(N,np.nan); hi=np.full(N,np.nan)
for i,w in enumerate(works):
    iv=meta_iv(w) or vedic_iv(w)
    if iv: lo[i],hi[i]=iv
anc=np.where(~np.isnan(lo))[0]
mid=0.5*(lo+hi); width=hi-lo
print(f"anchors: {len(anc)}")
print(f"range WIDTH (na-nb) distribution: min={np.nanmin(width[anc]):.0f} median={np.nanmedian(width[anc]):.0f} mean={np.nanmean(width[anc]):.0f} max={np.nanmax(width[anc]):.0f}")
import numpy as _n
for q in (50,100,200,300,500):
    print(f"  anchors with width>={q}y: {100*np.mean(width[anc]>=q):.0f}%")
groups=np.array([group(w) for w in works])
pred=cross_val_predict(GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0),
                       COMB[anc],mid[anc],cv=GroupKFold(10),groups=groups[anc])
mp=pred; L=lo[anc]; H=hi[anc]; MID=mid[anc]
mae_mid=np.abs(mp-MID).mean()
# interval-aware error: 0 if inside [lo,hi], else distance to nearest edge
inside=(mp>=L)&(mp<=H)
edge_err=np.where(inside,0.0,np.minimum(np.abs(mp-L),np.abs(mp-H)))
print(f"\nCV predictions vs anchors:")
print(f"  MAE to midpoint           = {mae_mid:.0f} y   (what we've been reporting)")
print(f"  fraction landing IN-range = {100*inside.mean():.0f}%")
print(f"  interval error (0 if in-range, else dist to edge): mean={edge_err.mean():.0f} y  median={np.median(edge_err):.0f} y")

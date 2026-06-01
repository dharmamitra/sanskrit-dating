import pickle, json, collections, statistics, sys, re
import numpy as np
from scipy.stats import spearmanr, pearsonr

D=pickle.load(open('vocab.pkl','rb'))
vocab=D['vocab']; glob=D['glob']; tlen=D['len']
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
# aggregate text-id vocab -> work
wvocab=collections.defaultdict(collections.Counter)
for t,c in vocab.items():
    wvocab[m2w(t)].update(c)
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])

# RV work id
RVKEYS=[w for w in wvocab if w.startswith('SA_GV01_rv')]
# merge all RV sections into one reference profile
rv=collections.Counter()
for w in RVKEYS: rv.update(wvocab[w])
print(f"RV reference: {len(RVKEYS)} sections, {sum(rv.values())} tokens, {len(rv)} word types",file=sys.stderr)

# choose works with enough tokens
MINTOK=2000
works=[w for w in wvocab if sum(wvocab[w].values())>=MINTOK]
print(f"works with >={MINTOK} tokens: {len(works)}",file=sys.stderr)

# MFW = top-K most frequent global words (function-word-ish)
K=int(sys.argv[1]) if len(sys.argv)>1 else 250
mfw=[w for w,_ in glob.most_common(K)]
print(f"top {K} MFW sample:", mfw[:25],file=sys.stderr)

def vec(counter):
    tot=sum(counter.values()) or 1
    return np.array([counter[w]/tot for w in mfw])
X=np.array([vec(wvocab[w]) for w in works])
# z-score columns (Burrows Delta space)
mu=X.mean(0); sd=X.std(0)+1e-9
Z=(X-mu)/sd
rvz=((vec(rv)-mu)/sd)
idx={w:i for i,w in enumerate(works)}

dates=np.array([mid(w) if mid(w) is not None else np.nan for w in works])
anc=[i for i in range(len(works)) if not np.isnan(dates[i])]
print(f"anchors among these works: {len(anc)}",file=sys.stderr)

# 1) distance from RV vs date
dist_rv=np.abs(Z-rvz).mean(1)   # Burrows delta to RV
da=dates[anc]; dr=dist_rv[anc]
print(f"\n[1] Burrows-delta-from-RV vs date: Spearman={spearmanr(dr,da).statistic:.3f} Pearson={pearsonr(dr,da)[0]:.3f}",file=sys.stderr)

# 2) kNN dating in MFW space, LOO over anchors
def delta(i,j): return np.abs(Z[i]-Z[j]).mean()
ancarr=np.array(anc)
Za=Z[ancarr]
def loo_knn(kk):
    err=[]; preds=[];trues=[]
    for a in range(len(ancarr)):
        d=np.abs(Za-Za[a]).mean(1); d[a]=1e9
        nn=np.argsort(d)[:kk]
        wts=1/(d[nn]+1e-6)
        pred=np.sum(wts*dates[ancarr[nn]])/wts.sum()
        err.append(abs(pred-dates[ancarr[a]])); preds.append(pred);trues.append(dates[ancarr[a]])
    err=np.array(err)
    return err.mean(), np.median(err), spearmanr(preds,trues).statistic
print(f"\n[2] vocab-kNN dating (LOO over {len(anc)} anchors):",file=sys.stderr)
for kk in (1,3,5,10,15):
    mae,med,spr=loo_knn(kk)
    print(f"   k={kk:2d}  MAE={mae:6.1f}  median={med:6.1f}  Spearman={spr:.3f}",file=sys.stderr)
gm=np.mean(dates[anc]); print(f"   baseline(mean) MAE={np.mean(np.abs(dates[anc]-gm)):.1f}",file=sys.stderr)

# 3) PCA, correlate components with date
Zc=Z-Z.mean(0)
U,S,Vt=np.linalg.svd(Zc,full_matrices=False)
pc=U[:,:5]*S[:5]
print(f"\n[3] PCA components vs date (anchors):",file=sys.stderr)
for k in range(5):
    r=spearmanr(pc[anc,k],dates[anc]).statistic
    print(f"   PC{k+1}: Spearman={r:.3f}  (var {100*S[k]**2/np.sum(S**2):.1f}%)",file=sys.stderr)

# ---- independence test: kNN using ONLY cross-collection neighbors ----
def coll(w):
    p=w.split('_'); return p[1] if len(p)>1 else w
ac_coll=[coll(works[i]) for i in anc]
def loo_knn_xcoll(kk, cross_only):
    err=[];preds=[];trues=[]
    for a in range(len(ancarr)):
        d=np.abs(Za-Za[a]).mean(1); d[a]=1e9
        if cross_only:
            for b in range(len(ancarr)):
                if b!=a and ac_coll[b]==ac_coll[a]: d[b]=1e9
        nn=np.argsort(d)[:kk]
        if d[nn[0]]>1e8: continue
        wts=1/(d[nn]+1e-6); pred=np.sum(wts*dates[ancarr[nn]])/wts.sum()
        err.append(abs(pred-dates[ancarr[a]]));preds.append(pred);trues.append(dates[ancarr[a]])
    err=np.array(err)
    return err.mean(),np.median(err),spearmanr(preds,trues).statistic,len(err)
print("\n[4] within-collection vs cross-collection-only kNN (k=5):",file=sys.stderr)
for cross in (False,True):
    mae,med,spr,n=loo_knn_xcoll(5,cross)
    tag="CROSS-collection only" if cross else "any neighbor"
    print(f"   {tag:24s}: MAE={mae:6.1f} med={med:6.1f} Spearman={spr:.3f} (n={n})",file=sys.stderr)

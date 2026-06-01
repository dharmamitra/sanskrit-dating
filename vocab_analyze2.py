import pickle, json, collections, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr

D=pickle.load(open('vocab2.pkl','rb'))
vocab=D['vocab']; glob=D['glob']; ocl=D['orig_complen']
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
wv=collections.defaultdict(collections.Counter); woc=collections.defaultdict(lambda:[0,0])
for t,c in vocab.items():
    w=m2w(t); wv[w].update(c)
    a,b=ocl.get(t,(0,0)); woc[w][0]+=a; woc[w][1]+=b
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w

rv=collections.Counter()
for w in wv:
    if w.startswith('SA_GV01_rv'): rv.update(wv[w])
print(f"RV ref tokens={sum(rv.values())}",file=sys.stderr)

MINTOK=2000
works=[w for w in wv if sum(wv[w].values())>=MINTOK]
K=int(sys.argv[1]) if len(sys.argv)>1 else 250
mfw=[w for w,_ in glob.most_common(K)]
print(f"works>={MINTOK}tok: {len(works)};  top MFW: {mfw[:22]}",file=sys.stderr)
def vec(c):
    tot=sum(c.values()) or 1; return np.array([c[w]/tot for w in mfw])
X=np.array([vec(wv[w]) for w in works]); mu=X.mean(0); sd=X.std(0)+1e-9
Z=(X-mu)/sd; rvz=(vec(rv)-mu)/sd
dates=np.array([mid(w) if mid(w) is not None else np.nan for w in works])
anc=[i for i in range(len(works)) if not np.isnan(dates[i])]
ancarr=np.array(anc); Za=Z[ancarr]; colls=[coll(works[i]) for i in anc]
print(f"anchors: {len(anc)}",file=sys.stderr)

dr=np.abs(Z[ancarr]-rvz).mean(1); da=dates[ancarr]
print(f"\n[1] delta-from-RV vs date: Spearman={spearmanr(dr,da).statistic:.3f}",file=sys.stderr)

def knn(kk, cross=False):
    err=[];P=[];T=[]
    for a in range(len(ancarr)):
        d=np.abs(Za-Za[a]).mean(1); d[a]=1e9
        if cross:
            for b in range(len(ancarr)):
                if b!=a and colls[b]==colls[a]: d[b]=1e9
        nn=np.argsort(d)[:kk]
        if d[nn[0]]>1e8: continue
        wt=1/(d[nn]+1e-6); pred=np.sum(wt*dates[ancarr[nn]])/wt.sum()
        err.append(abs(pred-dates[ancarr[a]]));P.append(pred);T.append(dates[ancarr[a]])
    return np.mean(err),np.median(err),spearmanr(P,T).statistic,len(err)
print(f"\n[2] vocab-kNN LOO:",file=sys.stderr)
for kk in (1,3,5,10):
    m_,md,sp,n=knn(kk); print(f"   k={kk:2d} MAE={m_:6.1f} med={md:6.1f} Spearman={sp:.3f}",file=sys.stderr)
print(f"   baseline(mean) MAE={np.mean(np.abs(da-da.mean())):.1f}",file=sys.stderr)
print(f"\n[3] independence (k=5):",file=sys.stderr)
for cr in (False,True):
    m_,md,sp,n=knn(5,cr); print(f"   {'CROSS-coll only' if cr else 'any nbr'}: MAE={m_:6.1f} Spearman={sp:.3f} (n={n})",file=sys.stderr)

# drift markers, raw + within-collection residual
def residualize(vals):
    vals=np.array(vals,float); cm=collections.defaultdict(list)
    for v,c in zip(vals,colls): cm[c].append(v)
    means={c:np.mean(v) for c,v in cm.items()}
    return np.array([v-means[c] for v,c in zip(vals,colls)])
def freq(c,ws): tot=sum(c.values()) or 1; return sum(c.get(w,0) for w in ws)/tot
def meanlen_orig(w): a,b=woc[w]; return a/b if b else 0
VEDIC=['ha','vai','sma','u','kila','aṅga','nu','īm','sīm','cid','iva']
feats={
 'orig_mean_word_len(compounding)':[meanlen_orig(works[i]) for i in anc],
 'vedic_particles':[freq(wv[works[i]],VEDIC) for i in anc],
 'iti_freq':[freq(wv[works[i]],['iti']) for i in anc],
 'ca_freq':[freq(wv[works[i]],['ca']) for i in anc],
}
dres=residualize(da)
print(f"\n[4] drift markers (Spearman raw | within-collection):",file=sys.stderr)
for nm,v in feats.items():
    raw=spearmanr(v,da).statistic
    fr=residualize(v); msk=np.abs(dres)>1e-9
    wic=spearmanr(fr[msk],dres[msk]).statistic if msk.sum()>5 else float('nan')
    print(f"   {nm:34s} {raw:6.3f} | {wic:6.3f}",file=sys.stderr)

Zc=Z-Z.mean(0); U,S,Vt=np.linalg.svd(Zc,full_matrices=False); pc=U[:,:6]*S[:6]
print(f"\n[5] PCA vs date:",file=sys.stderr)
for k in range(6):
    print(f"   PC{k+1}: Spearman={spearmanr(pc[ancarr,k],da).statistic:.3f} (var{100*S[k]**2/np.sum(S**2):.1f}%)",file=sys.stderr)

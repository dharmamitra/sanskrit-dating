import json, collections, pickle, sys
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

rng=np.random.default_rng(0)
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def meta_iv(w):
    m=meta.get(w)
    return (min(float(m['nb']),float(m['na'])),max(float(m['nb']),float(m['na']))) if m and m['nb'] is not None and m['na'] is not None else None
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

# ---- chunk features ----
chunks=[];works=[];D=[]
with open('chunks_dense.tsv') as fh:
    fh.readline()
    for line in fh:
        p=line.rstrip('\n').split('\t');chunks.append(p[0]);works.append(p[1]);D.append([float(x) for x in p[3:]])
D=np.array(D);chunks=np.array(chunks);cwork=np.array([m2w(w) for w in works])
fw=pickle.load(open('chunks_fw.pkl','rb'));cdf=collections.Counter()
for cid in chunks:
    for g in fw.get(cid,{}): cdf[g]+=1
gvoc=[g for g,dd in cdf.items() if dd>=20];gi={g:i for i,g in enumerate(gvoc)}
R=[];C=[];V=[]
for r,cid in enumerate(chunks):
    cc=fw.get(cid,{});tot=sum(cc.values()) or 1
    for g,n in cc.items():
        if g in gi: R.append(r);C.append(gi[g]);V.append(n/tot)
G=sp.csr_matrix((V,(R,C)),(len(chunks),len(gvoc)));gidf=np.log((len(chunks)+1)/(np.array([cdf[g] for g in gvoc])+1))
G=normalize(G.multiply(gidf).tocsr());FW=TruncatedSVD(50,random_state=0).fit_transform(G);X=np.hstack([D,FW])
def wdate(w):
    iv=meta_iv(w) or vedic_iv(w);return .5*(iv[0]+iv[1]) if iv else None
ywork={w:wdate(w) for w in set(cwork)}
ychunk=np.array([ywork[w] if ywork[w] is not None else np.nan for w in cwork])
anc=np.where(~np.isnan(ychunk))[0]; grp=np.array([group(w) for w in cwork])

# ---- linguistic estimate m_i per work: OOF for anchors, final-model for undated ----
print("CV (out-of-fold) chunk predictions...",file=sys.stderr)
oof=np.full(len(chunks),np.nan)
oof[anc]=cross_val_predict(HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,max_leaf_nodes=63,random_state=0),
                           X[anc],ychunk[anc],cv=GroupKFold(10),groups=grp[anc])
final=HistGradientBoostingRegressor(max_iter=700,learning_rate=0.05,max_leaf_nodes=63,random_state=0).fit(X[anc],ychunk[anc])
allpred=final.predict(X)
byw=collections.defaultdict(list); byw_oof=collections.defaultdict(list)
for i,w in enumerate(cwork):
    byw[w].append(allpred[i])
    if not np.isnan(oof[i]): byw_oof[w].append(oof[i])
WORKS=sorted(byw); widx={w:i for i,w in enumerate(WORKS)}; Nw=len(WORKS)
m=np.array([np.median(byw_oof[w]) if w in byw_oof else np.median(byw[w]) for w in WORKS])  # linguistic estimate
nch=np.array([len(byw[w]) for w in WORKS])
iv=[meta_iv(w) or vedic_iv(w) for w in WORKS]
isanc=np.array([x is not None for x in iv])
lo=np.array([x[0] if x else np.nan for x in iv]); hi=np.array([x[1] if x else np.nan for x in iv])
mid=0.5*(lo+hi); width=hi-lo
print(f"works={Nw}  anchored={isanc.sum()}",file=sys.stderr)

# ---- Gibbs ----
finf=np.sqrt(1+4.0/nch)                 # n-inflation of linguistic noise (few chunks -> noisier)
s_anc=np.where(isanc, np.clip(width/3.3,25,None), np.inf)  # SOFT anchor sd; wide interval=weak
p_anc=np.where(isanc, 1.0/s_anc**2, 0.0)
t=m.copy()                              # init at linguistic estimate
mu=np.nanmean(m); sig2=np.nanvar(m); sl2=150.0**2
NIT=6000; BURN=1500
samps=np.zeros((NIT-BURN,Nw)); mu_s=[]; sl2_s=[]; sig2_s=[]
for it in range(NIT):
    tau2=sl2*finf**2                     # linguistic variance per work
    p_ling=1.0/tau2; p_hier=1.0/sig2
    post_prec=p_ling+p_anc+p_hier
    anc_term=np.where(isanc, mid*p_anc, 0.0)
    post_mean=(m*p_ling + anc_term + mu*p_hier)/post_prec
    t=post_mean + rng.standard_normal(Nw)/np.sqrt(post_prec)
    # hierarchical mu, sig2 (conjugate, flat-ish priors)
    mu=rng.normal(t.mean(), np.sqrt(sig2/Nw))
    sig2=1.0/rng.gamma(2+Nw/2, 1.0/(1000+0.5*np.sum((t-mu)**2)))
    # linguistic noise sl2 from residuals m-t (account for finf)
    rr=(m-t)/finf
    sl2=1.0/rng.gamma(2+Nw/2, 1.0/(1000+0.5*np.sum(rr**2)))
    if it>=BURN:
        samps[it-BURN]=t; mu_s.append(mu); sl2_s.append(sl2); sig2_s.append(sig2)
med=np.median(samps,0); lo95=np.percentile(samps,2.5,0); hi95=np.percentile(samps,97.5,0)
mu_h=np.mean(mu_s); sl2_h=np.mean(sl2_s); sig2_h=np.mean(sig2_s)
print(f"posterior sigma_ling≈{np.sqrt(sl2_h):.0f}  hier sd≈{np.sqrt(sig2_h):.0f}",file=sys.stderr)
ciw=hi95-lo95
print(f"95% credible-interval width: median={np.median(ciw):.0f}y  (anchored {np.median(ciw[isanc]):.0f}y, undated {np.median(ciw[~isanc]):.0f}y)",file=sys.stderr)

# ---- HONEST evaluation: leave-each-anchor's-own-interval-out posterior mean ----
A=np.where(isanc)[0]
pL=1.0/(sl2_h*finf**2); pH=1.0/sig2_h
ho=(m*pL+mu_h*pH)/(pL+pH)               # posterior dropping own anchor term (~ linguistic, shrunk to hier mean)
ho_sd=1.0/np.sqrt(pL+pH)
inside=np.mean([lo[i]<=ho[i]<=hi[i] for i in A])*100
edge=np.array([0.0 if lo[i]<=ho[i]<=hi[i] else min(abs(ho[i]-lo[i]),abs(ho[i]-hi[i])) for i in A])
print(f"\nHELD-OUT (drop own anchor) — the real test:",file=sys.stderr)
print(f"  in-range={inside:.0f}%  interval-edge-err median={np.median(edge):.0f}y  Spearman(held-out, mid)={spearmanr(ho[A],mid[A]).statistic:.3f}",file=sys.stderr)
# disputes: held-out posterior puts <10% mass in scholarly interval (normal approx)
from scipy.stats import norm
disp=[]
for i in A:
    fin=norm.cdf(hi[i],ho[i],ho_sd[i])-norm.cdf(lo[i],ho[i],ho_sd[i])
    if fin<0.10: disp.append((WORKS[i],lo[i],hi[i],ho[i],fin))
disp.sort(key=lambda d:d[4])
print(f"\nANCHORS THE MODEL DISPUTES (held-out posterior <10% mass in scholarly interval): {len(disp)}",file=sys.stderr)
for w,l,h,me,f in disp[:18]:
    print(f"   {w:26s} scholarly[{l:.0f},{h:.0f}] held-out={me:.0f} (mass {100*f:.0f}%)  {meta.get(w,{}).get('title','')[:26]}",file=sys.stderr)

# ---- output ----
with open('dated_gibbs.tsv','w') as f:
    f.write("work\tsource\tn_chunks\tling_est\tnb\tna\tpost_median\tcrI_lo95\tcrI_hi95\ttitle\n")
    for i in np.argsort(med):
        w=WORKS[i]; src='anchor' if isanc[i] else 'inferred'
        f.write(f"{w}\t{src}\t{nch[i]}\t{m[i]:.0f}\t{lo[i] if isanc[i] else '':}\t{hi[i] if isanc[i] else '':}\t{med[i]:.0f}\t{lo95[i]:.0f}\t{hi95[i]:.0f}\t{meta.get(w,{}).get('title','')}\n")
print("\nwrote dated_gibbs.tsv",file=sys.stderr)

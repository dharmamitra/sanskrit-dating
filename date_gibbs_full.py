import os, json, collections, pickle, sys, re
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from scipy.special import ndtr, ndtri
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
# ---- closed-interval anchors from the curated constraint files ----
# These externally-researched dates (researched_anchors / manual_constraints / ...) were
# previously fed ONLY to the Gibbs sampler, never to the linguistic clock's training set.
# As a result whole genres (Epic, Purana) carried 0-3 clock-training anchors, fell below the
# per-category threshold, and were dated by the GLOBAL clock (trained on Shastra/Vedic/Kavya).
# Loading them here lets them ALSO supervise the stylometric clock, so Epic/Purana qualify
# for their own category clocks. Only closed-interval `anchor` rows are usable as a regression
# target (one-sided not_before/not_after bounds have no midpoint and stay Gibbs-only).
constraint_iv={}
for _cf in ['dcs_anchors.tsv','researched_anchors.tsv','chronbmm_priors.tsv','manual_constraints.tsv','new_texts_anchors.tsv']:
    if not os.path.exists(_cf): continue
    for line in open(_cf).read().splitlines()[1:]:
        if not line.strip(): continue
        p=(line.split('\t')+['','','',''])[:5]
        if p[0]=='anchor':
            try: a,b=float(p[2]),float(p[3])
            except ValueError: continue
            constraint_iv[p[1]]=(min(a,b),max(a,b))
COLLAPSE=['SA_GV01_rvpp','SA_GV01_rv_hn','SA_GV01_rv','SA_GV03_sb','SA_GV05_brup','SA_GV05_chup','SA_GV05_aitup','SA_GV05_prasup','SA_GV05_chupsb','SA_GV02_gop']
def group(w):
    for c in COLLAPSE:
        if w.startswith(c): return c
    return w

# ---- category labels (the unit we "stay within" to avoid cross-domain pollution) ----
# fine category = the GRETIL/DSBC shelf code (e.g. K12 tantric scriptures, K10 canonical
# sutras, T02 tantric commentary, GK19 kavya, MB Muktabodha tantra, GV01 Rigveda...).
# coarse domain = a broad tradition/genre bucket used as fallback when a fine code is
# too thin to support its own stylometric clock / hierarchical mean.
def fine_cat(w):
    m=re.match(r'SA_([A-Za-z]+\d*)_',w)
    return m.group(1) if m else 'NA'
def coarse_dom(w):
    c=fine_cat(w)
    if c.startswith('GV'): return 'Vedic'
    if c.startswith('GE'): return 'Epic'
    if c.startswith('GP'): return 'Purana'
    if c.startswith('GK'): return 'Kavya'
    if c.startswith('GS'): return 'Shastra'
    if c.startswith('GR'): return 'AgamaShaiva'
    if c[:1]=='G':         return 'GMisc'
    if c[:1]=='K':         return 'BuddhCanon'
    if c[:1]=='T':         return 'BuddhTreatise'
    if c.startswith('MB'): return 'TantricMB'
    return 'GLOBAL'

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
    iv=meta_iv(w) or vedic_iv(w) or constraint_iv.get(w);return .5*(iv[0]+iv[1]) if iv else None
ywork={w:wdate(w) for w in set(cwork)}
ychunk=np.array([ywork[w] if ywork[w] is not None else np.nan for w in cwork])
anc=np.where(~np.isnan(ychunk))[0]; grp=np.array([group(w) for w in cwork])

# ---- linguistic estimate m_i: WITHIN-CATEGORY stylometric clock ----
# Instead of one global style->date regressor (which lets Vedic/epic anchors pull a
# tantric work's estimate, and vice-versa), fit a separate regressor per category and
# date each work with its OWN category's clock.  A fine category must clear thresholds
# (enough anchored works + chunks) to get its own model; else fall back to its coarse
# domain; else the global model.  Honest OOF (GroupKFold by work) is preserved per model.
MIN_W=12; MIN_C=60   # min anchored works / anchor chunks to fit a category-specific model
def _hgb(it): return HistGradientBoostingRegressor(max_iter=it,learning_rate=0.05,max_leaf_nodes=63,random_state=0)
fine_c =np.array([fine_cat(w)   for w in cwork])
coarse_c=np.array([coarse_dom(w) for w in cwork])
def _qualset(labels):
    works_by=collections.defaultdict(set); chunks_by=collections.Counter()
    for i in anc:
        works_by[labels[i]].add(cwork[i]); chunks_by[labels[i]]+=1
    return {k for k in works_by if len(works_by[k])>=MIN_W and chunks_by[k]>=MIN_C}
qual_fine=_qualset(fine_c); qual_coarse=_qualset(coarse_c)
def model_key(w):
    f=fine_cat(w)
    if f in qual_fine: return ('F',f)
    c=coarse_dom(w)
    if c in qual_coarse: return ('C',c)
    return ('G','GLOBAL')
keychunk=np.array([str(model_key(w)) for w in cwork])
print(f"category clocks: {len(qual_fine)} fine + {len(qual_coarse)} coarse (+global fallback)",file=sys.stderr)

print("CV (out-of-fold) chunk predictions, per category...",file=sys.stderr)
# global base (covers everything; also used for any leftover works)
oof=np.full(len(chunks),np.nan)
oof[anc]=cross_val_predict(_hgb(500),X[anc],ychunk[anc],cv=GroupKFold(10),groups=grp[anc])
allpred=_hgb(700).fit(X[anc],ychunk[anc]).predict(X)
# per-category override (finest qualifying level for each work)
for key in sorted(set(keychunk)):
    if key=="('G', 'GLOBAL')": continue
    idx=np.where(keychunk==key)[0]; aidx=np.intersect1d(idx,anc)
    ng=len(set(grp[aidx]))
    if ng<3 or len(aidx)<MIN_C: continue
    ns=min(10,ng)
    oof[aidx]=cross_val_predict(_hgb(500),X[aidx],ychunk[aidx],cv=GroupKFold(ns),groups=grp[aidx])
    allpred[idx]=_hgb(700).fit(X[aidx],ychunk[aidx]).predict(X[idx])
byw=collections.defaultdict(list); byw_oof=collections.defaultdict(list)
for i,w in enumerate(cwork):
    byw[w].append(allpred[i])
    if not np.isnan(oof[i]): byw_oof[w].append(oof[i])
WORKS=sorted(byw); widx={w:i for i,w in enumerate(WORKS)}; Nw=len(WORKS)
m=np.array([np.median(byw_oof[w]) if w in byw_oof else np.median(byw[w]) for w in WORKS])  # linguistic estimate
nch=np.array([len(byw[w]) for w in WORKS])
# ---- ordering constraints (root <= commentary) — kept CROSS-domain (borrowing is real) ----
REL=[]
for line in open('relations_work.tsv').read().splitlines()[1:]:
    p=line.split('\t');
    if p[0] in widx and p[1] in widx and p[0]!=p[1]: REL.append((p[0],p[1]))
preds=collections.defaultdict(list); succs=collections.defaultdict(list)
for w1,w2 in REL:
    i,j=widx[w1],widx[w2]; preds[j].append(i); succs[i].append(j)
CNODES=np.array(sorted(set([widx[w] for w,_ in REL]+[widx[w] for _,w in REL])))
print(f"ordering constraints: {len(REL)} pairs over {len(CNODES)} works",file=sys.stderr)
iv=[meta_iv(w) or vedic_iv(w) for w in WORKS]
isanc=np.array([x is not None for x in iv])
lo=np.array([x[0] if x else np.nan for x in iv]); hi=np.array([x[1] if x else np.nan for x in iv])
mid=0.5*(lo+hi); width=hi-lo
print(f"works={Nw}  anchored={isanc.sum()}",file=sys.stderr)
# ---- translation terminus-ante-quem (Chinese translation date - lag) ----
taq=np.full(Nw, np.inf); TAQINFO={}
for line in open('translation_taq.tsv').read().splitlines()[1:]:
    p=line.split('\t'); w=p[0]
    if w in widx: taq[widx[w]]=float(p[1]); TAQINFO[w]=(float(p[1]),p[3],p[5])
nconf=0
for i in np.where(np.isfinite(taq))[0]:
    if isanc[i] and taq[i] < lo[i]-1:
        print(f"  CONFLICT: {WORKS[i]} taq={taq[i]:.0f} < scholarly nb={lo[i]:.0f}",file=sys.stderr); nconf+=1
CNODES=np.array(sorted(set(list(CNODES)+list(np.where(np.isfinite(taq))[0]))))
print(f"translation termini: {int(np.isfinite(taq).sum())} works ({nconf} conflict w/ scholarly nb)",file=sys.stderr)
# ---- manual curated constraints ----
lb=np.full(Nw, -np.inf); nman=0
for _cf in ['dcs_anchors.tsv','researched_anchors.tsv','chronbmm_priors.tsv','manual_constraints.tsv','new_texts_anchors.tsv']:
  if os.path.exists(_cf):
    for line in open(_cf).read().splitlines()[1:]:
        if not line.strip(): continue
        p=(line.split('\t')+['','','',''])[:5]; typ=p[0]; w=p[1]
        if typ=='anchor' and w in widx:
            i=widx[w]; lo[i]=float(p[2]); hi[i]=float(p[3]); isanc[i]=True; nman+=1
        elif typ=='not_before' and w in widx:
            lb[widx[w]]=max(lb[widx[w]], float(p[2])); nman+=1
        elif typ=='not_after' and w in widx:
            taq[widx[w]]=min(taq[widx[w]], float(p[2])); nman+=1
        elif typ=='order' and w in widx and p[2] in widx:
            preds[widx[p[2]]].append(widx[w]); succs[widx[w]].append(widx[p[2]]); nman+=1
mid=0.5*(lo+hi); width=hi-lo
CNODES=np.array(sorted(set([i for i in preds]+[i for i in succs]+list(np.where(np.isfinite(taq))[0])+list(np.where(np.isfinite(lb))[0]))))
print(f"manual constraints applied: {nman}",file=sys.stderr)

# ---- per-work category index for the hierarchical prior (fine category, partial-pooled) ----
wcat=[fine_cat(w) for w in WORKS]
catset=sorted(set(wcat)); cidx={c:k for k,c in enumerate(catset)}; cof=np.array([cidx[c] for c in wcat])
Kc=len(catset)
print(f"hierarchical categories: {Kc} (each work shrinks to its own category mean)",file=sys.stderr)

# ---- Gibbs ----
finf=np.sqrt(1+4.0/nch)                 # n-inflation of linguistic noise (few chunks -> noisier)
s_anc=np.where(isanc, np.clip(width/3.3,25,None), np.inf)  # SOFT anchor sd; wide interval=weak
p_anc=np.where(isanc, 1.0/s_anc**2, 0.0)
t=m.copy()                              # init at linguistic estimate
mu0=np.nanmean(m); tau0_2=np.nanvar(m)  # GLOBAL hyperprior over category means
mu_c=np.array([np.nanmedian(m[cof==k]) if np.any(cof==k) else mu0 for k in range(Kc)])  # category means
sig2=np.nanvar(m); sl2=150.0**2
NIT=6000; BURN=1500
samps=np.zeros((NIT-BURN,Nw)); muc_s=np.zeros((NIT-BURN,Kc)); sl2_s=[]; sig2_s=[]
for it in range(NIT):
    tau2=sl2*finf**2                     # linguistic variance per work
    p_ling=1.0/tau2; p_hier=1.0/sig2
    muvec=mu_c[cof]                      # each work's CATEGORY mean (was a single global mean)
    post_prec=p_ling+p_anc+p_hier
    anc_term=np.where(isanc, mid*p_anc, 0.0)
    post_mean=(m*p_ling + anc_term + muvec*p_hier)/post_prec
    t=post_mean + rng.standard_normal(Nw)/np.sqrt(post_prec)
    for i in rng.permutation(CNODES):
        L=max(max([t[j] for j in preds[i]], default=-1e9), lb[i]); U=min(min([t[j] for j in succs[i]], default=1e9), taq[i])
        if L>=U: t[i]=0.5*(L+U); continue
        mn=post_mean[i]; sd=1.0/np.sqrt(post_prec[i])
        a=ndtr((L-mn)/sd); b=ndtr((U-mn)/sd)
        if b-a<1e-12: t[i]=min(max(mn,L),U); continue
        u=a+(b-a)*rng.random(); t[i]=mn+sd*ndtri(min(max(u,1e-9),1-1e-9))
    # per-category means mu_c (conjugate Normal, partial-pooled toward global mu0)
    for k in range(Kc):
        mem=np.where(cof==k)[0]; nk=len(mem)
        prec=nk*p_hier + 1.0/tau0_2
        mean=(p_hier*t[mem].sum() + mu0/tau0_2)/prec
        mu_c[k]=mean + rng.standard_normal()/np.sqrt(prec)
    # global hyperprior over the category means
    mu0=rng.normal(mu_c.mean(), np.sqrt(tau0_2/Kc))
    tau0_2=1.0/rng.gamma(2+Kc/2, 1.0/(1000+0.5*np.sum((mu_c-mu0)**2)))
    # within-category spread sig2 (shared; residuals around each work's category mean)
    sig2=1.0/rng.gamma(2+Nw/2, 1.0/(1000+0.5*np.sum((t-mu_c[cof])**2)))
    # linguistic noise sl2 from residuals m-t (account for finf)
    rr=(m-t)/finf
    sl2=1.0/rng.gamma(2+Nw/2, 1.0/(1000+0.5*np.sum(rr**2)))
    if it>=BURN:
        samps[it-BURN]=t; muc_s[it-BURN]=mu_c; sl2_s.append(sl2); sig2_s.append(sig2)
med=np.median(samps,0); lo95=np.percentile(samps,2.5,0); hi95=np.percentile(samps,97.5,0)
sl2_h=np.mean(sl2_s); sig2_h=np.mean(sig2_s); muc_h=muc_s.mean(0)
print(f"posterior sigma_ling≈{np.sqrt(sl2_h):.0f}  within-category sd≈{np.sqrt(sig2_h):.0f}",file=sys.stderr)
ciw=hi95-lo95
print(f"95% credible-interval width: median={np.median(ciw):.0f}y  (anchored {np.median(ciw[isanc]):.0f}y, undated {np.median(ciw[~isanc]):.0f}y)",file=sys.stderr)

# ---- HONEST evaluation: drop each anchor's own interval, shrink to its CATEGORY mean ----
A=np.where(isanc)[0]
pL=1.0/(sl2_h*finf**2); pH=1.0/sig2_h
ho=(m*pL+muc_h[cof]*pH)/(pL+pH)         # posterior dropping own anchor term (linguistic + category mean)
ho_sd=1.0/np.sqrt(pL+pH)
inside=np.mean([lo[i]<=ho[i]<=hi[i] for i in A])*100
edge=np.array([0.0 if lo[i]<=ho[i]<=hi[i] else min(abs(ho[i]-lo[i]),abs(ho[i]-hi[i])) for i in A])
print(f"\nHELD-OUT (drop own anchor, shrink to category) — the real test:",file=sys.stderr)
print(f"  in-range={inside:.0f}%  interval-edge-err median={np.median(edge):.0f}y  Spearman(held-out, mid)={spearmanr(ho[A],mid[A]).statistic:.3f}",file=sys.stderr)
from scipy.stats import norm
disp=[]
for i in A:
    fin=norm.cdf(hi[i],ho[i],ho_sd[i])-norm.cdf(lo[i],ho[i],ho_sd[i])
    if fin<0.10: disp.append((WORKS[i],lo[i],hi[i],ho[i],fin))
disp.sort(key=lambda d:d[4])
print(f"\nANCHORS THE MODEL DISPUTES (held-out posterior <10% mass in scholarly interval): {len(disp)}",file=sys.stderr)
for w,l,h,me,f in disp[:18]:
    print(f"   {w:26s} scholarly[{l:.0f},{h:.0f}] held-out={me:.0f} (mass {100*f:.0f}%)  {meta.get(w,{}).get('title','')[:26]}",file=sys.stderr)

# ---- per-domain held-out summary (shows the cross-domain-pollution fix) ----
print("\nHELD-OUT in-range by coarse domain:",file=sys.stderr)
dlabel=np.array([coarse_dom(WORKS[i]) for i in range(Nw)])
for d in sorted(set(dlabel[A])):
    Ad=[i for i in A if dlabel[i]==d]
    ir=np.mean([lo[i]<=ho[i]<=hi[i] for i in Ad])*100
    print(f"   {d:16s} n={len(Ad):4d}  in-range={ir:.0f}%  med-edge-err={np.median([0.0 if lo[i]<=ho[i]<=hi[i] else min(abs(ho[i]-lo[i]),abs(ho[i]-hi[i])) for i in Ad]):.0f}y",file=sys.stderr)

# ---- output ----
with open('dated_gibbs_full.tsv','w') as f:
    # NOTE: title stays at index 9 (visualizers read it positionally); category appended at 10.
    f.write("work\tsource\tn_chunks\tling_est\tnb\tna\tpost_median\tcrI_lo95\tcrI_hi95\ttitle\tcategory\n")
    for i in np.argsort(med):
        w=WORKS[i]; src='anchor' if isanc[i] else 'inferred'
        f.write(f"{w}\t{src}\t{nch[i]}\t{m[i]:.0f}\t{lo[i] if isanc[i] else '':}\t{hi[i] if isanc[i] else '':}\t{med[i]:.0f}\t{lo95[i]:.0f}\t{hi95[i]:.0f}\t{meta.get(w,{}).get('title','')}\t{wcat[i]}\n")
print("\nwrote dated_gibbs_full.tsv",file=sys.stderr)

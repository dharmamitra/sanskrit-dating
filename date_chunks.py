import json, collections, pickle, sys, re, os
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.decomposition import TruncatedSVD

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

# ---- load chunk dense features ----
chunks=[]; works=[]; ntok=[]; D=[]
with open('chunks_dense.tsv') as fh:
    cols=fh.readline().rstrip('\n').split('\t')[3:]
    for line in fh:
        p=line.rstrip('\n').split('\t')
        chunks.append(p[0]); works.append(p[1]); ntok.append(float(p[2])); D.append([float(x) for x in p[3:]])
D=np.array(D); chunks=np.array(chunks); cwork=np.array([m2w(w) for w in works])
print(f"chunks={len(chunks)}  dense feats={D.shape[1]}",file=sys.stderr)

# ---- fw-grams -> svd (chunk level) ----
fw=pickle.load(open('chunks_fw.pkl','rb'))
cdf=collections.Counter()
for cid in chunks:
    for g in fw.get(cid,{}): cdf[g]+=1
gvoc=[g for g,dd in cdf.items() if dd>=20]; gi={g:i for i,g in enumerate(gvoc)}
R=[];C=[];V=[]
for r,cid in enumerate(chunks):
    cc=fw.get(cid,{}); tot=sum(cc.values()) or 1
    for g,n in cc.items():
        if g in gi: R.append(r);C.append(gi[g]);V.append(n/tot)
G=sp.csr_matrix((V,(R,C)),(len(chunks),len(gvoc)))
gidf=np.log((len(chunks)+1)/(np.array([cdf[g] for g in gvoc])+1)); G=normalize(G.multiply(gidf).tocsr())
FW=TruncatedSVD(50,random_state=0).fit_transform(G)
X=np.hstack([D,FW])
print(f"fw-grams {len(gvoc)}->50; total feat dim={X.shape[1]}",file=sys.stderr)

# ---- labels ----
def wdate(w):
    iv=meta_iv(w) or vedic_iv(w); return .5*(iv[0]+iv[1]) if iv else None
y=np.array([wdate(w) if wdate(w) is not None else np.nan for w in cwork])
grp=np.array([group(w) for w in cwork])
anc=np.where(~np.isnan(y))[0]
print(f"anchor chunks={len(anc)} from {len(set(cwork[anc]))} works",file=sys.stderr)

def workscore(chunk_pred, A):
    # aggregate chunk predictions to work (median), score vs work midpoint
    byw=collections.defaultdict(list)
    for j,i in enumerate(A): byw[cwork[i]].append(chunk_pred[j])
    ws=sorted(byw); wp=np.array([np.median(byw[w]) for w in ws]); wt=np.array([wdate(w) for w in ws])
    iv=[meta_iv(w) or vedic_iv(w) for w in ws]
    inside=np.array([1.0 if (iv[k][0]<=wp[k]<=iv[k][1]) else 0.0 for k in range(len(ws))])
    edge=np.array([0.0 if inside[k] else min(abs(wp[k]-iv[k][0]),abs(wp[k]-iv[k][1])) for k in range(len(ws))])
    return np.abs(wp-wt).mean(), spearmanr(wp,wt).statistic, inside.mean()*100, np.median(edge), len(ws)

print("\nCHUNK-LEVEL CV (GroupKFold by work):",file=sys.stderr)
for name,est in [("HistGBM",HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,max_leaf_nodes=63,random_state=0)),
                 ("MLP(256,64)",make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(256,64),alpha=1.0,max_iter=800,random_state=0,early_stopping=True)))]:
    pred=cross_val_predict(est,X[anc],y[anc],cv=GroupKFold(10),groups=grp[anc])
    csp=spearmanr(pred,y[anc]).statistic; cmae=np.abs(pred-y[anc]).mean()
    wmae,wsp,winr,wmed,nw=workscore(pred,anc)
    print(f"  {name:12s} chunk: MAE={cmae:.0f} Spearman={csp:.3f} | WORK-agg: MAE={wmae:.0f} Spearman={wsp:.3f} in-range={winr:.0f}% interval-med={wmed:.0f} (works={nw})",file=sys.stderr)

# ---- final model on all anchor chunks; predict all chunks ----
final=HistGradientBoostingRegressor(max_iter=700,learning_rate=0.05,max_leaf_nodes=63,random_state=0).fit(X[anc],y[anc])
allpred=final.predict(X)

# ---- Mahabharata stratification ----
print("\nMAHABHARATA per-book chunk-date distribution (held out of training):",file=sys.stderr)
bybook=collections.defaultdict(list)
for i,t in enumerate(works):
    if t.startswith('SA_GE07_mbh_') and t.endswith('_u'):
        bk=t.replace('SA_GE07_mbh_','').replace('_u','')
        bybook[bk].append(allpred[i])
for bk in sorted(bybook):
    v=np.array(bybook[bk])
    print(f"  Book {bk:>3s}: n={len(v):3d}  median={np.median(v):5.0f}  IQR=[{np.percentile(v,25):.0f},{np.percentile(v,75):.0f}]  range=[{v.min():.0f},{v.max():.0f}]",file=sys.stderr)

# ---- Book 6 (Bhismaparvan) by adhyaya, flag Bhagavadgita (6.23-6.40) ----
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
SPEAK={'vac','brū','ah','vad','abhidhā'}
d=json.load(open(SEG+'/SA_GE07_mbh_06_u.json'))
kept=[]; prev=None
for s in d:
    fa=s.get('full_analysis')
    if not fa: continue
    if len(fa)<=3 and fa[-1].get('lemma') in SPEAK: continue
    sig=(s.get('analyzed') or '').strip()
    if sig and sig==prev: continue
    prev=sig; kept.append(s)
def adhyaya(s):
    o=s.get('original') or ''
    m=re.match(r'0?6[,.](\d+)',o)
    return int(m.group(1)) if m else None
cidpred={c:allpred[i] for i,c in enumerate(chunks)}
print("\nBHISMAPARVAN (Book 6) chunk trajectory  [* = Bhagavadgita region, adhy 23-40]:",file=sys.stderr)
k=0
for st in range(0,len(kept),100):
    segs=kept[st:st+100]
    cid=f"SA_GE07_mbh_06_u#{k}"; k+=1
    if cid not in cidpred: continue
    ads=[adhyaya(s) for s in segs if adhyaya(s)]
    lo,hi=(min(ads),max(ads)) if ads else (None,None)
    gita = lo is not None and not(hi<23 or lo>40)
    bar='#'*max(1,int((cidpred[cid]+500)/60))
    print(f"  chunk {k-1:2d} adhy {str(lo):>3s}-{str(hi):<3s} pred={cidpred[cid]:6.0f} {'*GITA' if gita else '     '} {bar}",file=sys.stderr)

# ---- save outputs + Gita-vs-rest stat ----
import numpy as _np
with open('dated_chunks.tsv','w') as f:
    f.write("chunk\twork\tpred\n")
    for i,c in enumerate(chunks): f.write(f"{c}\t{works[i]}\t{allpred[i]:.0f}\n")
byw=collections.defaultdict(list)
for i,w in enumerate(cwork): byw[w].append(allpred[i])
with open('dated_chunks_works.tsv','w') as f:
    f.write("work\tn_chunks\tmedian\tp25\tp75\ttitle\n")
    for w in sorted(byw,key=lambda w:_np.median(byw[w])):
        v=_np.array(byw[w]); f.write(f"{w}\t{len(v)}\t{_np.median(v):.0f}\t{_np.percentile(v,25):.0f}\t{_np.percentile(v,75):.0f}\t{meta.get(w,{}).get('title','')}\n")
# Gita vs rest in Book 6
gita=[];rest=[]
k=0
for st in range(0,len(kept),100):
    segs=kept[st:st+100]; cid=f"SA_GE07_mbh_06_u#{k}"; k+=1
    if cid not in cidpred: continue
    ads=[adhyaya(s) for s in segs if adhyaya(s)]
    if not ads: continue
    lo,hi=min(ads),max(ads)
    (gita if not(hi<23 or lo>40) else rest).append(cidpred[cid])
import numpy as _np
print(f"\nBOOK 6 Gita-region (adhy23-40) vs rest: Gita median={_np.median(gita):.0f} (n={len(gita)}) | rest median={_np.median(rest):.0f} (n={len(rest)})",file=sys.stderr)
from scipy.stats import mannwhitneyu
try:
    u,p=mannwhitneyu(gita,rest); print(f"  Mann-Whitney p={p:.3f} (is Gita region distinguishable?)",file=sys.stderr)
except Exception as e: print(e,file=sys.stderr)
print("wrote dated_chunks.tsv, dated_chunks_works.tsv",file=sys.stderr)

import json, collections, pickle, sys
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import normalize
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

# load chunk features
chunks=[]; works=[]; D=[]
with open('chunks_dense.tsv') as fh:
    fh.readline()
    for line in fh:
        p=line.rstrip('\n').split('\t'); chunks.append(p[0]); works.append(p[1]); D.append([float(x) for x in p[3:]])
D=np.array(D); chunks=np.array(chunks); cwork=np.array([m2w(w) for w in works])
fw=pickle.load(open('chunks_fw.pkl','rb')); cdf=collections.Counter()
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
FW=TruncatedSVD(50,random_state=0).fit_transform(G); X=np.hstack([D,FW])

def wdate(w):
    iv=meta_iv(w) or vedic_iv(w); return .5*(iv[0]+iv[1]) if iv else None
y=np.array([wdate(w) if wdate(w) is not None else np.nan for w in cwork])

# HOLD OUT the entire Yogacara cluster (T06 + srabhu) so none of its dates leak
def is_yog(w): return w.startswith('SA_T06')
holdout=np.array([is_yog(w) for w in cwork])
train=(~np.isnan(y)) & (~holdout)
print(f"train anchor chunks={train.sum()} (Yogacara cluster held out); predicting Yogacara chunks out-of-sample",file=sys.stderr)
model=HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_leaf_nodes=63,random_state=0).fit(X[train],y[train])
pred=model.predict(X)

# Schmithausen strata mapping
STRATA={
 'oldest (Sravaka/Bodhisattva/verse core)':['SA_T06_srabhu_u','SA_T06_srabhusu','SA_T06_bsa034','SA_T06_asycsaru'],
 'intermediate (rest of Basic Section)':['SA_T06_sambhu','SA_T06_ybh-laukikamarga','SA_T06_-ybh-klesa','SA_T06_n1394u'],
 'developed treatises (vijnanavada systematization)':['SA_T06_n2994u','SA_T06_asabhs_u','SA_T06_asmahsuu','SA_T06_asmahsbu','SA_T06_bsa030_u','SA_T06_vmvkbh_u','SA_T06_bsa010_u','SA_T06_vasvvmsu','SA_T06_bsa022_u','SA_T06_bsa018','SA_T06_bsa019_u','SA_T06_bsa065'],
 'latest (commentaries: Sthiramati etc.)':['SA_T06_sthmavt','SA_T06_sthmavt1','SA_T06_sthmavtyg','SA_T06_sthtvbh','SA_T06_pskvbhu','SA_T06_trimsikatika','SA_T06_abhsubhu'],
}
byw=collections.defaultdict(list)
for i,w in enumerate(cwork):
    if holdout[i]: byw[w].append(pred[i])
print("\n=== OUT-OF-SAMPLE chunk-date predictions for Yogacara cluster (model never saw their dates) ===",file=sys.stderr)
ranks=[];meds=[]
for si,(stratum,wl) in enumerate(STRATA.items(),1):
    print(f"\n[Stratum {si}] {stratum}",file=sys.stderr)
    smeds=[]
    for w in wl:
        if w not in byw: continue
        v=np.array(byw[w]); md=np.median(v)
        iv=meta_iv(w); tag=f"(meta {iv[0]:.0f}-{iv[1]:.0f})" if iv else "(undated)"
        print(f"   {w:26s} n={len(v):3d} pred_median={md:5.0f} [{np.percentile(v,25):.0f},{np.percentile(v,75):.0f}] {tag}  {meta.get(w,{}).get('title','')[:28]}",file=sys.stderr)
        smeds.append(md); ranks.append(si); meds.append(md)
    if smeds: print(f"   -> stratum median = {np.median(smeds):.0f}",file=sys.stderr)
if len(set(ranks))>1:
    rho=spearmanr(ranks,meds).statistic
    print(f"\nSpearman(Schmithausen stratum rank, predicted date) = {rho:.3f}  (n={len(meds)} works)",file=sys.stderr)
    print("(positive => linguistic dating independently reproduces Schmithausen's oldest->latest ordering)",file=sys.stderr)

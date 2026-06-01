import json, collections, pickle, sys, os
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
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
y=np.array([wdate(w) if wdate(w) is not None else np.nan for w in cwork])
# HOLD OUT the AKBh cluster (kosa + Yasomitra vyakhya)
hold=np.array([w.startswith('SA_T07_vako') or w.startswith('SA_T07_yabhk') for w in cwork])
train=(~np.isnan(y))&(~hold)
print(f"train chunks={train.sum()} (AKBh held out)",file=sys.stderr)
model=HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_leaf_nodes=63,random_state=0).fit(X[train],y[train])
pred=model.predict(X); cidpred={c:pred[i] for i,c in enumerate(chunks)}

# chapter map for vakobhau (start_seg -> label)
CH=[(0,'1 Dhātu'),(1360,'2 Indriya'),(4129,'3 Loka'),(6985,'4 Karma'),(9878,'5 Anuśaya'),
    (11380,'6-8 Mārga/Jñāna/Samāpatti'),(15376,'9 Pudgalaviniścaya')]
def chap(oi):
    lab=CH[0][1]
    for st,l in CH:
        if oi>=st: lab=l
        else: break
    return lab
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
SPEAK={'vac','brū','ah','vad','abhidhā'}
fid='SA_T07_vakobhau'; d=json.load(open(f"{SEG}/{fid}.json"))
kept=[];prev=None
for i,s in enumerate(d):
    fa=s.get('full_analysis')
    if not fa: continue
    if len(fa)<=3 and fa[-1].get('lemma') in SPEAK: continue
    sig=(s.get('analyzed') or '').strip()
    if sig and sig==prev: continue
    prev=sig; kept.append(i)
rows=[];k=0
for st in range(0,len(kept),100):
    idxs=kept[st:st+100]; cid=f"{fid}#{k}"; k+=1
    if cid not in cidpred: continue
    rows.append((k-1, chap(idxs[len(idxs)//2]), cidpred[cid]))
order=[r[0] for r in rows]; pr=np.array([r[2] for r in rows])
med=np.median(pr); mad=np.median(np.abs(pr-med))*1.4826+1e-9
print(f"\n===== ABHIDHARMAKOŚABHĀṢYA control ({fid}, held out) =====",file=sys.stderr)
print(f"chunks={len(rows)} median={med:.0f} MAD-sd={mad:.0f} range=[{pr.min():.0f},{pr.max():.0f}]",file=sys.stderr)
print(f"WITHIN-WORK Spearman(position, predicted date) = {spearmanr(order,pr).statistic:+.3f}",file=sys.stderr)
print(f"  [compare: Bodhisattvabhūmi +0.34, Śrāvakabhūmi +0.18 -- single-author AKBh should be ~0 if controlled]",file=sys.stderr)
bych=collections.defaultdict(list)
for k0,c,p in rows: bych[c].append(p)
print("\n  per-chapter medians (textual order):",file=sys.stderr)
for st,l in CH:
    if l in bych:
        v=np.array(bych[l]); print(f"   ch {l:28s} n={len(v):3d} median={np.median(v):5.0f} [{np.percentile(v,25):.0f},{np.percentile(v,75):.0f}]",file=sys.stderr)
# Ch9 vs rest
ch9=bych.get('9 Pudgalaviniścaya',[]); rest=[p for c,ps in bych.items() if c!='9 Pudgalaviniścaya' for p in ps]
if ch9 and len(ch9)>=3:
    u,pv=mannwhitneyu(ch9,rest)
    print(f"\n  Ch9 Pudgalaviniścaya (self-refutation) median={np.median(ch9):.0f} (n={len(ch9)}) vs rest {np.median(rest):.0f}: Mann-Whitney p={pv:.3f}",file=sys.stderr)
# outliers
print("\n  outlier chunks (|z|>=1.8):",file=sys.stderr)
for k0,c,p in rows:
    z=(p-med)/mad
    if abs(z)>=1.8: print(f"   chunk {k0:3d} ch {c:26s} pred={p:5.0f} z={z:+.1f}",file=sys.stderr)

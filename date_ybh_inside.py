import json, collections, pickle, sys, re, os
import numpy as np, scipy.sparse as sp
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
# features
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
hold=np.array([w.startswith('SA_T06') for w in cwork]);train=(~np.isnan(y))&(~hold)
model=HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_leaf_nodes=63,random_state=0).fit(X[train],y[train])
pred=model.predict(X)
cidpred={c:pred[i] for i,c in enumerate(chunks)}

SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
SPEAK={'vac','brū','ah','vad','abhidhā'}
HDR=re.compile(r'paṭala|yogasthāna|\(Chapter|\(Book|bhūmiḥ',re.I)
def analyze(fileid, label):
    d=json.load(open(f"{SEG}/{fileid}.json"))
    # section headers: original-seg-index -> label
    headers=[]
    for i,s in enumerate(d):
        o=(s.get('original') or '')
        if HDR.search(o) and len(o)<60: headers.append((i,o.strip()))
    def sec_at(oi):
        lab='(start)'
        for hi,hl in headers:
            if hi<=oi: lab=hl
            else: break
        return lab
    # replicate preprocessing/chunking, track original indices
    kept=[]; prev=None
    for i,s in enumerate(d):
        fa=s.get('full_analysis')
        if not fa: continue
        if len(fa)<=3 and fa[-1].get('lemma') in SPEAK: continue
        sig=(s.get('analyzed') or '').strip()
        if sig and sig==prev: continue
        prev=sig; kept.append(i)
    preds=[]; rows=[]
    k=0
    for st in range(0,len(kept),100):
        idxs=kept[st:st+100]; cid=f"{fileid}#{k}"; k+=1
        if cid not in cidpred: continue
        mid_oi=idxs[len(idxs)//2]
        rows.append((k-1, idxs[0], idxs[-1], sec_at(mid_oi), cidpred[cid])); preds.append(cidpred[cid])
    preds=np.array(preds); med=np.median(preds); mad=np.median(np.abs(preds-med))*1.4826+1e-9
    print(f"\n===== {label} ({fileid}) =====",file=sys.stderr)
    print(f"chunks={len(preds)}  median={med:.0f}  MAD-based sd={mad:.0f}  range=[{preds.min():.0f},{preds.max():.0f}]",file=sys.stderr)
    print(f"  {'ck':>3s} {'segs':>11s} {'pred':>6s} {'z':>5s}  section",file=sys.stderr)
    for (k0,a,b,sec,p) in rows:
        z=(p-med)/mad; flag=' <<OUTLIER' if abs(z)>=1.8 else ''
        print(f"  {k0:3d} {a:5d}-{b:<5d} {p:6.0f} {z:+5.1f}  {sec[:40]}{flag}",file=sys.stderr)
    # per-section summary
    bysec=collections.defaultdict(list)
    for (k0,a,b,sec,p) in rows: bysec[sec].append(p)
    print("  -- section medians (n>=1) --",file=sys.stderr)
    for sec,ps in sorted(bysec.items(),key=lambda kv:np.median(kv[1])):
        print(f"     {np.median(ps):6.0f}  (n={len(ps):2d})  {sec[:46]}",file=sys.stderr)

analyze('SA_T06_bsa034','Bodhisattvabhūmi')
analyze('SA_T06_srabhusu','Śrāvakabhūmi (Shukla, clean Skt)')

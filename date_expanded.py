import json, collections, pickle, numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
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

# ---- ntok per file (from morph) + aggregate any per-file rate-table to work ----
ntok_file={}
def load_table(path, has_ntok):
    global ntok_file
    rows={}
    with open(path) as fh:
        cols=fh.readline().rstrip('\n').split('\t')[1:]
        for line in fh:
            p=line.rstrip('\n').split('\t'); rows[p[0]]=dict(zip(cols,map(float,p[1:])))
    if has_ntok:
        for t,d in rows.items(): ntok_file[t]=d['n_tokens']
    return rows, [c for c in cols if c!='n_tokens']
morph_rows,mfeats=load_table('morph_features.tsv',True)
pos_rows,pfeats=load_table('pos_features.tsv',False)

def aggregate(rows,feats):
    agg=collections.defaultdict(lambda:collections.defaultdict(float)); wn=collections.defaultdict(float)
    for t,d in rows.items():
        w=m2w(t); n=ntok_file.get(t,0)
        if n<=0: continue
        wn[w]+=n
        for c in feats: agg[w][c]+=d.get(c,0)*n
    return agg,wn
maggn,wn=aggregate(morph_rows,mfeats)
pagg,_=aggregate(pos_rows,pfeats)
works=[w for w in maggn if wn[w]>=1000]; idx={w:i for i,w in enumerate(works)}; N=len(works)
M=np.array([[maggn[w][c]/wn[w] for c in mfeats] for w in works])
P=np.array([[(pagg[w][c]/wn[w] if wn[w]>0 else 0) for c in pfeats] for w in works])

# fw-grams -> svd
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
G=normalize(G.multiply(gidf).tocsr()); FW=TruncatedSVD(50,random_state=0).fit_transform(G)
print(f"works={N}  morph={M.shape[1]}  POS={P.shape[1]}  fw-gram->{FW.shape[1]}",file=__import__('sys').stderr)

# anchors
lo=np.full(N,np.nan); hi=np.full(N,np.nan)
for i,w in enumerate(works):
    iv=meta_iv(w) or vedic_iv(w)
    if iv: lo[i],hi[i]=iv
anc=np.where(~np.isnan(lo))[0]; mid=0.5*(lo+hi); groups=np.array([group(w) for w in works])

def score(pred,A):
    L,H=lo[A],hi[A]; MID=mid[A]
    inside=(pred>=L)&(pred<=H); edge=np.where(inside,0.0,np.minimum(np.abs(pred-L),np.abs(pred-H)))
    return np.abs(pred-MID).mean(), inside.mean()*100, np.median(edge), spearmanr(pred,MID).statistic
def run(Xblocks,name,model='gbm'):
    X=np.hstack(Xblocks)
    if model=='gbm':
        est=GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0)
    else:
        est=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,32),alpha=1.0,max_iter=1500,random_state=0,early_stopping=True))
    pred=cross_val_predict(est,X[anc],mid[anc],cv=GroupKFold(10),groups=groups[anc])
    mae,inr,med,spr=score(pred,anc)
    print(f"  {name:34s} MAE_mid={mae:6.0f} in-range={inr:4.0f}% interval-med={med:5.0f} Spearman={spr:.3f}",file=__import__('sys').stderr)
print("\nGBM:",file=__import__('sys').stderr)
run([M],"morph (old)")
run([FW],"fw-grams (old)")
run([P],"POS uni+bigrams (NEW)")
run([M,P],"morph + POS")
run([M,P,FW],"morph + POS + fw-grams (ALL)")
print("\nshallow MLP (128,32):",file=__import__('sys').stderr)
run([M,P,FW],"ALL (MLP)",model='mlp')

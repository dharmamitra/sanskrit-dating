import pickle, json, collections, sys
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

D=pickle.load(open('vocab2.pkl','rb')); vocab=D['vocab']; glob=D['glob']
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
wv=collections.defaultdict(collections.Counter)
for t,c in vocab.items(): wv[m2w(t)].update(c)
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def coll(w): p=w.split('_'); return p[1] if len(p)>1 else w

works=[w for w in wv if sum(wv[w].values())>=2000]
N=len(works)
df=collections.Counter()
for w in works:
    for t in wv[w]: df[t]+=1
MODE=sys.argv[1] if len(sys.argv)>1 else 'content'   # 'content'=full vocab, 'stop'=top-300 only
if MODE=='stop':
    voc=[w for w,_ in glob.most_common(300)]
else:
    voc=[t for t,d in df.items() if 10<=d<=0.5*N]
vi={t:i for i,t in enumerate(voc)}
print(f"MODE={MODE}  works={N}  vocab={len(voc)}",file=sys.stderr)

# sparse TF matrix
R=[];C=[];V=[]
for r,w in enumerate(works):
    tot=sum(wv[w].values())
    for t,n in wv[w].items():
        if t in vi: R.append(r);C.append(vi[t]);V.append(n/tot)
X=sp.csr_matrix((V,(R,C)),shape=(N,len(voc)))
# tf-idf: sublinear tf already rel-freq; apply idf
idf=np.log(N/np.array([df[t] for t in voc]))
X=X.multiply(idf).tocsr()
Xn=normalize(X)   # L2 rows

dates=np.array([mid(w) if mid(w) is not None else np.nan for w in works])
anc=np.where(~np.isnan(dates))[0]
colls=np.array([coll(w) for w in works])
def eta2(axis):
    # fraction of variance between collections (genre-ness)
    tot=np.var(axis)*len(axis)
    bs=0
    for c in set(colls):
        m=axis[colls==c]; bs+=len(m)*(m.mean()-axis.mean())**2
    return bs/tot if tot>0 else 0
def eval_axis(axis,name):
    sp_=spearmanr(axis[anc],dates[anc]).statistic
    print(f"  {name:24s} |rho_date|={abs(sp_):.3f}  genre_eta2={eta2(axis):.2f}",file=sys.stderr)
    return abs(sp_)

# ---- LSA / SVD ----
svd=TruncatedSVD(n_components=30, random_state=0)
S=svd.fit_transform(Xn)   # works x 30
print(f"\n[A] SVD/LSA components (unsupervised), |corr with date| & genre-eta2:",file=sys.stderr)
best=0;bestk=-1
for k in range(10):
    r=eval_axis(S[:,k],f"PC{k+1} (var{100*svd.explained_variance_ratio_[k]:.1f}%)")
    if r>best: best=r;bestk=k
print(f"  -> best single PC for date: PC{bestk+1} |rho|={best:.3f}",file=sys.stderr)

# ---- spectral seriation (Fiedler vector of cosine-sim kNN graph) ----
from sklearn.neighbors import kneighbors_graph
A=kneighbors_graph(S[:,:30], n_neighbors=15, mode='distance', metric='cosine')
A=0.5*(A+A.T); A.data=np.exp(-A.data/ (A.data.mean()+1e-9))  # affinity
deg=np.asarray(A.sum(1)).ravel(); Dm=sp.diags(deg)
L=Dm-A
from scipy.sparse.linalg import eigsh
dinv=sp.diags(1/np.sqrt(deg+1e-9)); Ln=dinv@L@dinv
vals,vecs=eigsh(Ln,k=3,which='SM')
fied=vecs[:,1]
print(f"\n[B] spectral seriation (Fiedler vector):",file=sys.stderr)
eval_axis(fied,"Fiedler")

# ---- distance from RV in this space ----
rv=collections.Counter()
for w in wv:
    if w.startswith('SA_GV01_rv'): rv.update(wv[w])
rr=np.zeros(len(voc))
tot=sum(rv.values())
for t,n in rv.items():
    if t in vi: rr[vi[t]]=n/tot
rr=rr*idf; rr=rr/ (np.linalg.norm(rr)+1e-9)
dfromrv=1-Xn.dot(rr)
print(f"\n[C] cosine-distance from RV:",file=sys.stderr)
eval_axis(dfromrv,"dist_from_RV")

print(f"\n(baseline: random axis |rho|~0; collection-genre axes have high eta2)",file=sys.stderr)

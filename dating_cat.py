import json, collections, math, sys, statistics, re
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
cache={}
def W_(t):
    if t in cache: return cache[t]
    v=m2w(t) or t; cache[t]=v; return v

# undirected edges (work-level counts)
we=collections.defaultdict(float)
with open('edges_raw.tsv') as fh:
    next(fh)
    for line in fh:
        r,p,c,_,_=line.rstrip('\n').split('\t')
        a,b=W_(r),W_(p)
        if a==b: continue
        a,b=sorted((a,b)); we[(a,b)]+=float(c)
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
works=sorted({w for e in we for w in e}); idx={w:i for i,w in enumerate(works)}; N=len(works)
dated={w:mid(w) for w in works if mid(w) is not None}

def coll(w):
    p=w.split('_'); return p[1] if len(p)>1 else w
def coarse(w):
    m=re.match(r'([A-Za-z]+)',coll(w)); return m.group(1) if m else coll(w)
COLL={w:coll(w) for w in works}; COARSE={w:coarse(w) for w in works}

def build_prior(dsub, min_full=2, min_coarse=2):
    """leak-safe collection prior from a given dated subset."""
    full=collections.defaultdict(list); crs=collections.defaultdict(list); allv=[]
    for w,v in dsub.items():
        full[COLL[w]].append(v); crs[COARSE[w]].append(v); allv.append(v)
    gm=statistics.mean(allv)
    fm={k:statistics.mean(v) for k,v in full.items()}
    cm={k:statistics.mean(v) for k,v in crs.items()}
    fc={k:len(v) for k,v in full.items()}; cc={k:len(v) for k,v in crs.items()}
    prior=np.empty(N); src=[]
    for i,w in enumerate(works):
        if COLL[w] in fm and fc[COLL[w]]>=min_full: prior[i]=fm[COLL[w]]; src.append('full')
        elif COARSE[w] in cm and cc[COARSE[w]]>=min_coarse: prior[i]=cm[COARSE[w]]; src.append('coarse')
        else: prior[i]=gm; src.append('global')
    return prior, src

def build_prior_loo(held):
    dsub={w:v for w,v in dated.items() if w!=held}
    return build_prior(dsub)

# graph matrix
R=[];C=[];V=[]
for (x,y),c in we.items():
    i,j=idx[x],idx[y]; w=math.log1p(c); R+=[i,j];C+=[j,i];V+=[w,w]
Wm=sp.csr_matrix((V,(R,C)),(N,N)); deg=np.asarray(Wm.sum(1)).ravel()

def harmonic_prior(dl, prior, lam):
    lab=np.zeros(N,bool);y=np.zeros(N)
    for w,v in dl.items(): lab[idx[w]]=True;y[idx[w]]=v
    U=np.where(~lab)[0];L=np.where(lab)[0]
    if len(U)==0: return y
    dinv=np.zeros(N);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+lam*sp.identity(len(U))
    b=P[U][:,L]@y[L]+lam*prior[U]
    xU=spsolve(A.tocsc(),b);x=y.copy();x[U]=xU;return x

anchors=[w for w in dated if deg[idx[w]]>0]
y=np.array([dated[w] for w in anchors])

def loo_eval(name, predfn):
    pred=np.array([predfn(h) for h in anchors])
    err=np.abs(pred-y)
    print(f"  {name:30s} MAE={err.mean():6.1f} med={np.median(err):6.1f} "
          f"<=200y:{100*np.mean(err<=200):4.1f}% <=300:{100*np.mean(err<=300):4.1f}% <=500:{100*np.mean(err<=500):4.1f}%",
          file=sys.stderr)
    return pred,err

print(f"works {N}  anchors(with edges) {len(anchors)}",file=sys.stderr)
print("\nLEAVE-ONE-OUT:",file=sys.stderr)
# (a) collection prior alone
def pred_prior(h):
    pr,_=build_prior_loo(h); return pr[idx[h]]
loo_eval("collection-prior ALONE", pred_prior)
# (c) graph + prior, sweep lambda
for lam in (0.05,0.1,0.2,0.4,0.8):
    def predfn(h, lam=lam):
        dl={w:v for w,v in dated.items() if w!=h}
        pr,_=build_prior_loo(h)
        return harmonic_prior(dl,pr,lam)[idx[h]]
    loo_eval(f"graph + prior (lam={lam})", predfn)
# reference: graph + GLOBAL prior (==old undirected harmonic)
def pred_glob(h):
    dl={w:v for w,v in dated.items() if w!=h}
    gm=np.full(N,statistics.mean(dl.values()))
    return harmonic_prior(dl,gm,0.02)[idx[h]]
loo_eval("graph + GLOBAL prior (old)", pred_glob)

# ---- final full model: best lambda, full output ----
BEST=0.2
prior,src=build_prior(dated)
x=harmonic_prior(dated, prior, BEST)
# also keep prior-only for comparison column
with open('dated_cat.tsv','w') as f:
    f.write("work\ttitle\tcollection\tsource\tprior_src\tnb\tna\tprior\test_date\tdeg\n")
    for i in np.argsort(x):
        w=works[i]; m=meta.get(w,{})
        f.write(f"{w}\t{m.get('title','')}\t{COLL[w]}\t{'anchor' if w in dated else 'inferred'}\t{src[i]}\t"
                f"{m.get('nb','')}\t{m.get('na','')}\t{prior[i]:.0f}\t{x[i]:.0f}\t{int(deg[i])}\n")
print(f"\nwrote dated_cat.tsv (lambda={BEST})",file=sys.stderr)

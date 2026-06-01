import json, collections, math, sys, statistics
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# ---------- load ----------
meta = json.load(open('meta.json'))
meta_keys = sorted(meta.keys(), key=len, reverse=True)

def map_to_work(text_id):
    for k in meta_keys:
        if text_id == k or text_id.startswith(k):
            return k
    return None

nodes=[]
with open('nodes_raw.tsv') as fh:
    next(fh)
    for line in fh:
        t=line.split('\t',1)[0]; nodes.append(t)
node2work={t:(map_to_work(t) or t) for t in nodes}

wedges=collections.defaultdict(float)
with open('edges_raw.tsv') as fh:
    next(fh)
    for line in fh:
        r,p,c,_,_=line.rstrip('\n').split('\t')
        wr=node2work.get(r,r); wp=node2work.get(p,p)
        if wr==wp: continue
        a,b=sorted((wr,wp)); wedges[(a,b)]+=float(c)

def mid(w):
    m=meta.get(w)
    if not m or m['nb'] is None or m['na'] is None: return None
    return 0.5*(m['nb']+m['na'])

works=set()
for (a,b) in wedges: works.update((a,b))
works=sorted(works)
idx={w:i for i,w in enumerate(works)}
N=len(works)
dated={w:mid(w) for w in works if mid(w) is not None}
print(f"works in graph: {N}  dated(anchors): {len(dated)}  work-edges: {len(wedges)}",file=sys.stderr)

rows=[];cols=[];vals=[]
for (a,b),c in wedges.items():
    i,j=idx[a],idx[b]; w=math.log1p(c)
    rows+=[i,j];cols+=[j,i];vals+=[w,w]
W=sp.csr_matrix((vals,(rows,cols)),shape=(N,N))
deg=np.asarray(W.sum(1)).ravel()
gm=statistics.mean(dated.values())
EPS=0.02

def harmonic(labeled_vals):
    lab=np.zeros(N,bool); y=np.zeros(N)
    for w,v in labeled_vals.items():
        lab[idx[w]]=True; y[idx[w]]=v
    U=np.where(~lab)[0]; Lk=np.where(lab)[0]
    if len(U)==0: return y
    dinv=np.zeros(N); nz=deg>0; dinv[nz]=1.0/deg[nz]
    P=sp.diags(dinv)@W
    P_UU=P[U][:,U]; P_UL=P[U][:,Lk]
    A=sp.identity(len(U))-P_UU+EPS*sp.identity(len(U))
    b=P_UL@y[Lk]+EPS*gm
    xU=spsolve(A.tocsc(),b)
    x=y.copy(); x[U]=xU
    return x

anchors=[w for w in dated if deg[idx[w]]>0]
errs=[]; det=[]
for held in anchors:
    av={w:v for w,v in dated.items() if w!=held}
    x=harmonic(av); pred=x[idx[held]]
    errs.append(abs(pred-dated[held]))
    det.append((held,dated[held],pred,int(deg[idx[held]])))
errs=np.array(errs)
print(f"\nLEAVE-ONE-OUT CV  (n={len(errs)} anchors with edges)",file=sys.stderr)
print(f"  MAE            = {errs.mean():6.1f} y",file=sys.stderr)
print(f"  median abs err = {np.median(errs):6.1f} y",file=sys.stderr)
for t in (50,100,200,300,500):
    print(f"  within {t:4d}y    : {100*np.mean(errs<=t):4.1f}%",file=sys.stderr)
bmae=statistics.mean(abs(dated[w]-gm) for w in anchors)
print(f"  baseline(global mean {gm:.0f}) MAE = {bmae:.1f}",file=sys.stderr)
det.sort(key=lambda d:abs(d[2]-d[1]),reverse=True)
print("\n  worst 12 (work | true | pred | err | deg | title):",file=sys.stderr)
for w,t,p,dg in det[:12]:
    print(f"   {w:26s} {t:6.0f} {p:6.0f} {abs(p-t):5.0f} {dg:4d}  {meta.get(w,{}).get('title','')[:24]}",file=sys.stderr)

x=harmonic(dated)
with open('dated_works.tsv','w') as out:
    out.write("work\ttitle\tsource\tnb\tna\test_date\tdeg\tn_dated_nbrs\n")
    Wbin=W.copy(); Wbin.data[:]=1
    datedmask=np.array([w in dated for w in works])
    ndn=(Wbin @ datedmask.astype(int))
    for i in np.argsort(x):
        w=works[i]; m=meta.get(w,{})
        src='anchor' if w in dated else 'inferred'
        out.write(f"{w}\t{m.get('title','')}\t{src}\t{m.get('nb','')}\t{m.get('na','')}\t{x[i]:.0f}\t{int(deg[i])}\t{int(ndn[i])}\n")
print(f"\nwrote dated_works.tsv ({N} works)",file=sys.stderr)

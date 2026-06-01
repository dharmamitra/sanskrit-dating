import json, collections, math, sys, statistics
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

cnt=collections.defaultdict(int); rl=collections.defaultdict(int); pl=collections.defaultdict(int)
with open('edges_dir.tsv') as fh:
    next(fh)
    for line in fh:
        r,p,c,r_l,p_l,_=line.rstrip('\n').split('\t')
        a,b=W_(r),W_(p)
        if a==b: continue
        cnt[(a,b)]+=int(c); rl[(a,b)]+=int(r_l); pl[(a,b)]+=int(p_l)

pairs={}
for (a,b) in list(cnt):
    x,y=sorted((a,b))
    if (x,y) in pairs: continue
    pairs[(x,y)]=dict(count=cnt.get((x,y),0)+cnt.get((y,x),0),
                      lenX=rl.get((x,y),0)+pl.get((y,x),0),
                      lenY=pl.get((x,y),0)+rl.get((y,x),0))
breadth=collections.Counter()
for (x,y) in pairs: breadth[x]+=1; breadth[y]+=1
works=sorted({w for e in pairs for w in e}); idx={w:i for i,w in enumerate(works)}; N=len(works)
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def iv(w):
    m=meta.get(w); return (float(m['nb']),float(m['na'])) if m and m['nb'] is not None and m['na'] is not None else None
dated={w:mid(w) for w in works if mid(w) is not None}

# undirected harmonic (baseline, reused)
R=[];C=[];V=[]
for (x,y),p in pairs.items():
    i,j=idx[x],idx[y]; w=math.log1p(p['count']); R+=[i,j];C+=[j,i];V+=[w,w]
Wm=sp.csr_matrix((V,(R,C)),(N,N)); deg=np.asarray(Wm.sum(1)).ravel()
gm=statistics.mean(dated.values()); EPS=0.02
def harmonic(dl):
    lab=np.zeros(N,bool);y=np.zeros(N)
    for w,v in dl.items(): lab[idx[w]]=True;y[idx[w]]=v
    U=np.where(~lab)[0];L=np.where(lab)[0]
    if len(U)==0: return y
    dinv=np.zeros(N);dinv[deg>0]=1/deg[deg>0];P=sp.diags(dinv)@Wm
    A=sp.identity(len(U))-P[U][:,U]+EPS*sp.identity(len(U))
    xU=spsolve(A.tocsc(),P[U][:,L]@y[L]+EPS*gm);x=y.copy();x[U]=xU;return x

# orientation: positive => x later. source=earlier, borrower=later
THR=1.0
def orient():
    src=collections.defaultdict(list); bor=collections.defaultdict(list)  # src[B]=[S...], bor[S]=[B...]
    for (x,y),p in pairs.items():
        e=math.log((p['lenY']+1)/(p['lenX']+1)); b=math.log((breadth[x]+1)/(breadth[y]+1))
        s=1.0*e+1.5*b
        if abs(s)<THR: continue
        if s>0: later,earlier=x,y
        else:   later,earlier=y,x
        src[later].append(earlier); bor[earlier].append(later)
    return src,bor
src,bor=orient()
print(f"works {N}  oriented constraints {sum(len(v) for v in src.values())}",file=sys.stderr)

LOW,HIGH=-500.,1900.
def propagate_intervals(dl):
    lo=np.full(N,LOW); hi=np.full(N,HIGH)
    for w,v in dl.items():
        i=iv(w)
        if i: lo[idx[w]],hi[idx[w]]=i
        else: lo[idx[w]]=hi[idx[w]]=dl[w]
    for _ in range(40):
        changed=False
        for B,Ss in src.items():
            iB=idx[B]; m=max((lo[idx[S]] for S in Ss),default=LOW)
            if m>lo[iB]+1e-6: lo[iB]=min(m,hi[iB]); changed=True
        for S,Bs in bor.items():
            iS=idx[S]; m=min((hi[idx[B]] for B in Bs),default=HIGH)
            if m<hi[iS]-1e-6: hi[iS]=max(m,lo[iS]); changed=True
        if not changed: break
    return lo,hi

def predict_all(dl):
    x=harmonic(dl)
    lo,hi=propagate_intervals(dl)
    out=x.copy()
    for i in range(N):
        l,h=lo[i],hi[i]
        if l>h: l,h=min(l,h),max(l,h)
        out[i]=min(max(x[i],l),h)   # clamp harmonic into feasible interval
    return out,lo,hi

# LOO
anchors=[w for w in dated if deg[idx[w]]>0]
errs=[];det=[]
for held in anchors:
    dl={w:v for w,v in dated.items() if w!=held}
    out,lo,hi=predict_all(dl)
    pred=out[idx[held]]; errs.append(abs(pred-dated[held]))
    det.append((held,dated[held],pred,lo[idx[held]],hi[idx[held]]))
errs=np.array(errs)
print(f"\nDIRECTIONAL (interval-clamped harmonic) LOO  n={len(errs)}",file=sys.stderr)
print(f"  MAE={errs.mean():.1f}  median={np.median(errs):.1f}",file=sys.stderr)
for t in (50,100,200,300,500): print(f"  within {t}y: {100*np.mean(errs<=t):.1f}%",file=sys.stderr)
print(f"  (undirected harmonic baseline was MAE 296)",file=sys.stderr)
det.sort(key=lambda d:abs(d[2]-d[1]),reverse=True)
print("\n  worst 10 (true|pred|[lo,hi]):",file=sys.stderr)
for w,t,p,l,h in det[:10]:
    print(f"   {w:24s} true={t:6.0f} pred={p:6.0f} [{l:6.0f},{h:6.0f}] {meta.get(w,{}).get('title','')[:22]}",file=sys.stderr)

# full output
out,lo,hi=predict_all(dated)
with open('dated_dir.tsv','w') as f:
    f.write("work\ttitle\tsource\tnb\tna\test_date\tlo\thi\tdeg\n")
    for i in np.argsort(out):
        w=works[i]; m=meta.get(w,{})
        f.write(f"{w}\t{m.get('title','')}\t{'anchor' if w in dated else 'inferred'}\t{m.get('nb','')}\t{m.get('na','')}\t{out[i]:.0f}\t{lo[i]:.0f}\t{hi[i]:.0f}\t{int(deg[i])}\n")
print("\nwrote dated_dir.tsv",file=sys.stderr)

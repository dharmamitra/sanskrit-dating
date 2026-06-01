import json, collections, math, sys, statistics
import numpy as np

meta = json.load(open('meta.json'))
mk = sorted(meta, key=len, reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return None

# node->work map (cache over distinct nodes appearing in edges)
def load_dir():
    cnt=collections.defaultdict(int); rl=collections.defaultdict(int); pl=collections.defaultdict(int)
    cache={}
    def w(t):
        if t in cache: return cache[t]
        v=m2w(t) or t; cache[t]=v; return v
    with open('edges_dir.tsv') as fh:
        next(fh)
        for line in fh:
            r,p,c,r_l,p_l,_=line.rstrip('\n').split('\t')
            a=w(r); b=w(p)
            if a==b: continue
            cnt[(a,b)]+=int(c); rl[(a,b)]+=int(r_l); pl[(a,b)]+=int(p_l)
    return cnt,rl,pl
cnt,rl,pl=load_dir()

def mid(x):
    m=meta.get(x); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])
def iv(x):
    m=meta.get(x); return (m['nb'],m['na']) if m and m['nb'] is not None and m['na'] is not None else None

# unordered pairs
pairs={}
works=set()
for (a,b) in list(cnt):
    works.add(a);works.add(b)
    x,y=sorted((a,b))
    if (x,y) in pairs: continue
    cAB=cnt.get((x,y),0); cBA=cnt.get((y,x),0)
    lenX=rl.get((x,y),0)+pl.get((y,x),0)
    lenY=pl.get((x,y),0)+rl.get((y,x),0)
    pairs[(x,y)]=dict(count=cAB+cBA, lenX=lenX, lenY=lenY)
# breadth = distinct partners
breadth=collections.Counter()
for (x,y) in pairs: breadth[x]+=1; breadth[y]+=1

def signals(x,y,p):
    e = math.log((p['lenX']+1)/(p['lenY']+1))        # >0 => X longer => X later
    b = math.log((breadth[x]+1)/(breadth[y]+1))       # >0 => X broader => X later
    return e,b

# ---------- VALIDATE against anchor pairs with disjoint intervals ----------
def disjoint_dir(x,y):
    ix,iy=iv(x),iv(y)
    if not ix or not iy: return None
    if ix[1] < iy[0]: return 'x_older'   # x strictly before y -> y is later
    if iy[1] < ix[0]: return 'y_older'
    return None
def report(name, pred_fn, gold):
    ok=tot=0
    for (x,y,truth) in gold:
        d=pred_fn(x,y,pairs[(x,y)])
        if d is None: continue
        tot+=1
        # truth: 'x_older' means y later; predict 'later'=='x' or 'y'
        later_true = 'y' if truth=='x_older' else 'x'
        if d==later_true: ok+=1
    print(f"  {name:22s}: {ok}/{tot} = {100*ok/max(tot,1):4.1f}%  (decided {tot})",file=sys.stderr)
    return ok,tot

gold=[]
for (x,y) in pairs:
    d=disjoint_dir(x,y)
    if d: gold.append((x,y,d))
print(f"anchor pairs with disjoint dates (known order): {len(gold)}",file=sys.stderr)
print(f"works in graph: {len(works)}  pairs: {len(pairs)}",file=sys.stderr)

def by_expansion(x,y,p):
    e,_=signals(x,y,p)
    if abs(e)<0.05: return None
    return 'x' if e>0 else 'y'
def by_breadth(x,y,p):
    _,b=signals(x,y,p)
    if abs(b)<0.05: return None
    return 'x' if b>0 else 'y'
def by_combo(x,y,p):
    e,b=signals(x,y,p); s=1.0*e+1.0*b
    if abs(s)<0.05: return None
    return 'x' if s>0 else 'y'
def by_breadth_then_exp(x,y,p):
    e,b=signals(x,y,p)
    if abs(b)>=0.4: return 'x' if b>0 else 'y'
    if abs(e)>=0.2: return 'x' if e>0 else 'y'
    return None

print("\nRULE ACCURACY on known-order anchor pairs (higher=better, 50%=coin):",file=sys.stderr)
report("expansion (longer=later)", by_expansion, gold)
report("breadth (broader=later)", by_breadth, gold)
report("combo e+b", by_combo, gold)
report("breadth-then-expansion", by_breadth_then_exp, gold)
# also: which sign? maybe SHORTER is later (extraction/anthology). test inverse expansion
def by_expansion_inv(x,y,p):
    e,_=signals(x,y,p)
    if abs(e)<0.05: return None
    return 'y' if e>0 else 'x'
report("INVERSE expansion", by_expansion_inv, gold)

# ---------- confidence sweep: positive score => X later ----------
def score_xlater(x,y,p):
    e = math.log((p['lenY']+1)/(p['lenX']+1))   # shorter X => later (inverse expansion)
    b = math.log((breadth[x]+1)/(breadth[y]+1)) # broader X => later
    return e, b, 1.0*e+1.5*b

print("\nCONFIDENCE SWEEP (orient only |combined|>=thr); acc on known pairs:",file=sys.stderr)
print(f"  {'thr':>5} {'decided':>8} {'cover%':>7} {'acc%':>6}",file=sys.stderr)
tot_gold=len(gold)
for thr in (0.0,0.2,0.5,1.0,1.5,2.0,3.0):
    ok=dec=0
    for (x,y,truth) in gold:
        _,_,s=score_xlater(x,y,pairs[(x,y)])
        if abs(s)<thr: continue
        dec+=1
        later_pred='x' if s>0 else 'y'
        later_true='y' if truth=='x_older' else 'x'
        if later_pred==later_true: ok+=1
    print(f"  {thr:5.1f} {dec:8d} {100*dec/tot_gold:6.1f} {100*ok/max(dec,1):6.1f}",file=sys.stderr)

# also weight gold by 'how far apart' (larger gap = easier/more reliable). acc on big-gap pairs:
def gap(x,y):
    ix,iy=iv(x),iv(y); return abs(0.5*(ix[0]+ix[1])-0.5*(iy[0]+iy[1]))
big=[(x,y,t) for (x,y,t) in gold if gap(x,y)>=300]
ok=dec=0
for (x,y,truth) in big:
    _,_,s=score_xlater(x,y,pairs[(x,y)])
    if abs(s)<0.5: continue
    dec+=1
    if ('x' if s>0 else 'y')==('y' if truth=='x_older' else 'x'): ok+=1
print(f"\n  big-gap(>=300y) pairs, thr0.5: {ok}/{dec}={100*ok/max(dec,1):.1f}% of {len(big)}",file=sys.stderr)

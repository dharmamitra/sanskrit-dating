import json, collections, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def mid(w):
    m=meta.get(w); return None if not m or m['nb'] is None or m['na'] is None else .5*(m['nb']+m['na'])

# load morph features, aggregate file->work (token-weighted; rates aggregate exactly)
hdr=None; agg=collections.defaultdict(lambda:collections.defaultdict(float)); ntok=collections.defaultdict(float)
with open('morph_features.tsv') as fh:
    hdr=fh.readline().rstrip('\n').split('\t')[1:]
    for line in fh:
        p=line.rstrip('\n').split('\t'); t=p[0]; vals=list(map(float,p[1:]))
        d=dict(zip(hdr,vals)); w=m2w(t); n=d['n_tokens']
        ntok[w]+=n
        for k,v in d.items():
            if k=='n_tokens': continue
            agg[w][k]+=v*n   # rate*tokens = count
feats=[c for c in hdr if c!='n_tokens']
works=[w for w in agg if ntok[w]>=1000]
W={w:np.array([agg[w][c]/ntok[w] for c in feats]) for w in works}
X=np.array([W[w] for w in works])
dates=np.array([mid(w) if mid(w) is not None else np.nan for w in works])
anc=np.where(~np.isnan(dates))[0]
print(f"works with morph feats (>=1000 tok): {len(works)};  anchors: {len(anc)}",file=sys.stderr)

# [1] individual diachronic markers vs date
print("\n[1] marker vs date (Spearman; expected sign in parens):",file=sys.stderr)
exp={'Mood_Sub':'-(Vedic)','Mood_Jus':'-(Vedic)','Formation_aor':'-(declines)',
     'VerbForm_Conv':'+(rises)','mean_cpd_len':'+(rises)','cpd_member_rate':'+',
     'Formation_peri':'+(later)','VerbForm_Part':'+','Voice_Passive':'+',
     'Tense_Present':'?','Case_Vocative':'-(hymnic)','finite_rate':'-'}
order=sorted(feats,key=lambda c:-abs(spearmanr(X[anc,feats.index(c)],dates[anc]).statistic))
for c in order:
    r=spearmanr(X[anc,feats.index(c)],dates[anc]).statistic
    print(f"   {c:22s} rho={r:+.3f}   {exp.get(c,'')}",file=sys.stderr)

# [2] anchored regression, NO category, LOO/CV
y=dates[anc]; Xa=X[anc]
def cv(model,Xm,name):
    pred=cross_val_predict(model,Xm,y,cv=KFold(10,shuffle=True,random_state=0))
    err=np.abs(pred-y)
    print(f"   {name:28s} MAE={err.mean():6.1f} med={np.median(err):6.1f} "
          f"Spearman={spearmanr(pred,y).statistic:.3f} <=200y:{100*np.mean(err<=200):.0f}% <=300y:{100*np.mean(err<=300):.0f}%",file=sys.stderr)
    return pred
print(f"\n[2] anchored regression (10-fold CV, NO category), n={len(anc)}:",file=sys.stderr)
ridge=lambda: RidgeCV(alphas=np.logspace(-3,3,25))
gb=lambda: GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0)
# feature groups
grp={
 'compound only':[i for i,c in enumerate(feats) if 'cpd' in c],
 'verbal only':[i for i,c in enumerate(feats) if any(k in c for k in('Tense','Mood','VerbForm','Voice','Formation','finite'))],
 'case only':[i for i,c in enumerate(feats) if 'Case' in c],
 'ALL morph':list(range(len(feats))),
}
for gname,idxs in grp.items():
    cv(ridge(), Xa[:,idxs], f"ridge / {gname}")
cv(gb(), Xa, "GBM / ALL morph")
print(f"   baseline(mean) MAE={np.abs(y-y.mean()).mean():.1f}",file=sys.stderr)

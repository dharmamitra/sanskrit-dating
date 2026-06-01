import pandas as pd, json, collections, unicodedata, re, sys
def clean(s):
    s=re.sub(r'\[[^\]]*\]','',str(s)); s=re.sub(r'\([^\)]*\)','',s); s=re.sub(r'\b\d+\b','',s)
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn').lower()
def norm(s): return re.sub(r'[^a-z]','',clean(s))
xl=pd.ExcelFile('/qnap/code/chronbmm/data/sanskrit/priors/Dates.ods', engine='odf')
PS=xl.parse('Primary sources', header=0)
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
inset=set(m2w(l.split('\t')[1]) for l in open('chunks_dense.tsv').read().splitlines()[1:])
t2w=collections.defaultdict(list)
for w in inset:
    if not w: continue
    nt=norm(meta.get(w,{}).get('title',''))
    if len(nt)>=6: t2w[nt].append(w)
def mp(full):
    nf=norm(full); return t2w[nf][0] if nf in t2w and len(nf)>=6 else None
van=[l.split('\t')[0] for l in open('vedic_anchors.tsv').read().splitlines()[1:]]
def anchored(w):
    m=meta.get(w,{})
    return (m.get('nb') is not None and m.get('na') is not None) or any(w.startswith(p) for p in van)
# stratum -> (nb,na) standard Vedic/Sutra-period chronology (chronBMM Group classification)
STRAT={'Ā':(-700,-500),'ŚS':(-700,-300),'GS':(-600,-300),'DhS':(-600,-200)}
SPECIAL={  # known stratum-outliers / per-text scholarly dates
 'Vaikhānasadharmasūtra':(200,500),     # late Dharmasutra (Olivelle ~3rd-4th c CE)
 'Māṇḍūkya-Upaniṣad':(-200,100),        # late prose Upanisad (Olivelle ~start of CE)
}
rows=[]
for _,r in PS.dropna(subset=['Abbreviation','FullName']).iterrows():
    full=str(r['FullName']).strip(); grp=str(r['Group']).strip()
    w=mp(full)
    if not w or anchored(w): continue
    if full in SPECIAL: nb,na=SPECIAL[full]; src='chronBMM-class + scholarly outlier date'
    elif grp in STRAT: nb,na=STRAT[grp]; src=f'chronBMM stratum {grp}'
    else: continue
    rows.append((w,nb,na,full,grp,src))
with open('chronbmm_priors.tsv','w') as f:
    f.write("type\targ1\targ2\targ3\tnote\n")
    for w,nb,na,full,grp,src in rows:
        f.write(f"anchor\t{w}\t{nb}\t{na}\t{src}: {full}\n")
print(f"chronbmm_priors.tsv: {len(rows)} stratum-based anchors for previously-unanchored late-Vedic texts",file=sys.stderr)
for w,nb,na,full,grp,src in sorted(rows,key=lambda x:x[1]):
    print(f"  [{nb:5d},{na:5d}] {grp:4s} {full[:34]:35s} -> {w}",file=sys.stderr)

import json, re, unicodedata, collections, sys
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
def deacc(s): return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn').lower()
# commentary levels (higher = later). High-precision keywords only.
KW=[(2,'anutika'),(2,'tika'),(2,'vyakhya'),(2,'panjika'),(2,'vivarana'),(2,'vivrti'),(2,'dipika'),(2,'kaumudi'),
    (1,'bhasya'),(1,'varttika'),(1,'vartika'),(1,'vrtti'),
    (0,'karika'),(0,'sutra')]
def detect(title):
    t=deacc(title)
    for lv,k in KW:
        if k in t: return lv, re.sub(r'[^a-z]','',t.replace(k,' '))
    return None, re.sub(r'[^a-z]','',t)
# which works are in our dataset (have chunks)?
inset=set()
for line in open('chunks_dense.tsv').read().splitlines()[1:]:
    inset.add(m2w(line.split('\t')[1]))
rows=[]
for fn,m in meta.items():
    ti=(m.get('title') or '').strip()
    if not ti: continue
    lv,base=detect(ti); rows.append((m2w(fn),ti,lv,base))
bygroup=collections.defaultdict(list)
for w,ti,lv,base in rows:
    if len(base)>=6 and w in inset: bygroup[base].append((w,ti,lv))
relset=set(); detail=[]
for base,members in bygroup.items():
    norm=[(w,ti,(0 if l is None else l)) for w,ti,l in members]
    if len(set(l for _,_,l in norm))<2: continue          # need >=2 distinct levels
    for w1,t1,l1 in norm:
        for w2,t2,l2 in norm:
            if l1<l2 and w1!=w2:
                key=(w1,w2)
                if key not in relset:
                    relset.add(key); detail.append((w1,t1,l1,w2,t2,l2,base))
with open('relations_work.tsv','w') as f:
    f.write("earlier_work\tlater_work\tearlier_title\tlater_title\tbase\n")
    for w1,t1,l1,w2,t2,l2,base in detail:
        f.write(f"{w1}\t{w2}\t{t1}\t{t2}\t{base}\n")
print(f"WORK-LEVEL root<commentary constraints: {len(detail)} (deduped, both in dataset)\n",file=sys.stderr)
for w1,t1,l1,w2,t2,l2,base in sorted(detail,key=lambda d:d[6]):
    print(f"  {t1[:32]:32s}(L{l1}) <= {t2[:32]:32s}(L{l2})",file=sys.stderr)

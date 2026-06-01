import json, glob, os, collections, sys, pickle
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
lf=pickle.load(open('lemma_freq.pkl','rb'))
STOP=set([l for l,_ in lf.most_common(100)])
print(f"stoplist (top-100 lemmas) sample: {[l for l,_ in lf.most_common(40)]}",file=sys.stderr)
files=[f for f in sorted(glob.glob(SEG+'/*.json')) if not f.endswith('_analyzed.json')]
out={}; nfa=0
for fp in files:
    try: d=json.load(open(fp))
    except: continue
    if not any(s.get('full_analysis') for s in d[:80]): continue
    nfa+=1
    t=os.path.basename(fp)[:-5]
    seq=[]
    for s in d:
        for x in s.get('full_analysis') or []:
            lem=x.get('lemma') or ''
            seq.append(lem if lem in STOP else None)
    bg=collections.Counter()
    for i in range(len(seq)-1):
        a,b=seq[i],seq[i+1]
        if a and b: bg[f'{a}_{b}']+=1           # adjacent function-word bigram
    for i in range(len(seq)-2):
        a,b,c=seq[i],seq[i+1],seq[i+2]
        if a and b and c: bg[f'{a}_{b}_{c}']+=1  # trigram
    if sum(bg.values())>=50: out[t]=bg
    if nfa%200==0: print(f"...{nfa} files",file=sys.stderr)
pickle.dump(out, open('fwgrams.pkl','wb'))
allg=collections.Counter()
for c in out.values(): allg.update(c)
print(f"DONE: {len(out)} texts, distinct fw-grams={len(allg)}; top: {[g for g,_ in allg.most_common(15)]}",file=sys.stderr)

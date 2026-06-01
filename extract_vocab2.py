import json, glob, collections, sys, os, pickle
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
files=[f for f in sorted(glob.glob(SEG+'/*.json')) if not f.endswith('_analyzed.json')]
print(f"{len(files)} text files",file=sys.stderr)
def tid(fn): return os.path.basename(fn)[:-5]   # strip .json
vocab={}            # textid -> Counter
orig_complen={}     # textid -> (sum original-word-chars, sum original-word-count) for compounding metric
glob_freq=collections.Counter()
for i,fp in enumerate(files):
    t=tid(fp)
    try: d=json.load(open(fp))
    except Exception as e:
        print("ERR",fp,e,file=sys.stderr); continue
    c=collections.Counter()
    och=ocn=0
    for s in d:
        a=s.get('analyzed') or ''
        for w in a.split():
            w=w.strip("-/|.,;:()[]{}\"'").lower()
            if w: c[w]+=1
        o=s.get('original') or ''
        for w in o.split():
            w=w.strip("/|.,;:()[]{}\"")
            if w and not any(ch.isdigit() for ch in w):
                och+=len(w); ocn+=1
    if sum(c.values())==0: continue
    vocab[t]=c; glob_freq.update(c); orig_complen[t]=(och,ocn)
    if (i+1)%400==0: print(f"...{i+1}/{len(files)} texts={len(vocab)}",file=sys.stderr)
pickle.dump({'vocab':vocab,'glob':glob_freq,'orig_complen':orig_complen},open('vocab2.pkl','wb'))
print(f"DONE texts={len(vocab)} distinct={len(glob_freq)} tokens={sum(glob_freq.values())}",file=sys.stderr)

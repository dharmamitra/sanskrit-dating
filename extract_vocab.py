import gzip, json, glob, collections, sys, re, pickle
files=sorted(glob.glob('matches/*.ndjson.gz'))
ALLOWED=set("abcdefghijklmnopqrstuvwxyzāīūṛṝḷḹṅñṭḍṇśṣḥṃ'")
def toks(s):
    out=[]
    for t in s.split():
        t=t.strip("/|.,;:()[]{}\"").lower()
        if not t: continue
        if all(c in ALLOWED for c in t) and any(c.isalpha() for c in t):
            out.append(t)
    return out
def tid(seg): return seg.rsplit(':',1)[0]
def lineno(seg):
    try: return int(seg.rsplit(':',1)[1])
    except: return None
vocab=collections.defaultdict(collections.Counter)   # text -> Counter(word->count)
seen=collections.defaultdict(set)                     # text -> set(lineno)
def add(segnrs, segtexts):
    for sn,st in zip(segnrs,segtexts):
        t=tid(sn); ln=lineno(sn)
        if ln is None or ln in seen[t]: continue
        seen[t].add(ln)
        vocab[t].update(toks(st))
for i,fp in enumerate(files):
    try:
        with gzip.open(fp,'rt') as fh:
            for line in fh:
                if not line.strip(): continue
                try: d=json.loads(line)
                except: continue
                rs=d.get('root_segnr');rt=d.get('root_segtext')
                ps=d.get('par_segnr');pt=d.get('par_segtext')
                if rs and rt: add(rs,rt)
                if ps and pt: add(ps,pt)
    except Exception as ex: print("ERR",fp,ex,file=sys.stderr)
    if (i+1)%300==0:
        nt=sum(len(c) for c in vocab.values())
        print(f"...{i+1}/{len(files)} texts={len(vocab)} total_word_entries={nt}",file=sys.stderr)
# save: per-text total tokens + counters; and global freq
glob_freq=collections.Counter()
text_len={}
for t,c in vocab.items():
    glob_freq.update(c); text_len[t]=sum(c.values())
pickle.dump({'vocab':dict(vocab),'glob':glob_freq,'len':text_len}, open('vocab.pkl','wb'))
print(f"DONE texts={len(vocab)} distinct_words={len(glob_freq)} total_tokens={sum(glob_freq.values())}",file=sys.stderr)

import json, glob, os, collections, sys, pickle
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
files=[f for f in sorted(glob.glob(SEG+'/*.json')) if not f.endswith('_analyzed.json')]
AOR={'s','is','red','root'}
rows={}; lemma_freq=collections.Counter()
def has(tag,key,val=None):
    return False
nfa=0
for fi,fp in enumerate(files):
    try: d=json.load(open(fp))
    except: continue
    if not any(s.get('full_analysis') for s in d[:80]): continue
    nfa+=1
    t=os.path.basename(fp)[:-5]
    c=collections.Counter(); tot=0
    cpd_lens=[]   # member counts of compounds (len>=2)
    cpd_member_tokens=0
    for s in d:
        fa=s.get('full_analysis')
        if not fa: continue
        run=0
        for x in fa:
            tot+=1
            lem=x.get('lemma') or ''
            if lem: lemma_freq[lem]+=1
            tag=x.get('tag') or ''
            isc='Compound' in tag
            if isc: run+=1; cpd_member_tokens+=1
            else:
                if run>0: cpd_lens.append(run+1)
                run=0
            # feature counts
            for part in tag.split(','):
                part=part.strip()
                if not part: continue
                if '=' in part:
                    k,v=part.split('=',1)
                    if k in ('Tense','Mood','VerbForm','Voice','Case'): c[f'{k}={v}']+=1
                    elif k=='Formation':
                        c['Formation=peri' if v=='peri' else ('Formation=aor' if v in AOR else f'Formation={v}')]+=1
                    elif k=='Person': c['Finite']+=1
        if run>0: cpd_lens.append(run+1)
    if tot<500: continue
    ncpd=len(cpd_lens)
    feat={
      'n_tokens':tot,
      'cpd_member_rate':cpd_member_tokens/tot,
      'mean_cpd_len':(sum(cpd_lens)/ncpd) if ncpd else 0,
      'cpd_ge3_rate':(sum(1 for L in cpd_lens if L>=3)/ncpd) if ncpd else 0,
      'finite_rate':c['Finite']/tot,
    }
    for key in ['Tense=Present','Tense=Past','Tense=Future','Tense=Imperativef',
                'Mood=Optative','Mood=Imperative','Mood=Sub','Mood=Jus',
                'VerbForm=Part','VerbForm=Conv','VerbForm=Gdv','VerbForm=Inf',
                'Voice=Passive','Formation=peri','Formation=aor',
                'Case=Nominative','Case=Accusative','Case=Genitive','Case=Instrumental',
                'Case=Locative','Case=Vocative','Case=Ablative','Case=Dative']:
        feat[key.replace('=','_')]=c[key]/tot
    rows[t]=feat
    if nfa%150==0: print(f"...{nfa} analyzed files, {fi+1}/{len(files)} scanned",file=sys.stderr)
# write
cols=list(next(iter(rows.values())).keys())
with open('morph_features.tsv','w') as out:
    out.write("text\t"+"\t".join(cols)+"\n")
    for t,feat in rows.items():
        out.write(t+"\t"+"\t".join(f"{feat[c]:.6g}" for c in cols)+"\n")
pickle.dump(lemma_freq, open('lemma_freq.pkl','wb'))
print(f"DONE: {len(rows)} texts with morph features, {len(cols)} features; lemmas={len(lemma_freq)}",file=sys.stderr)

import json, glob, os, collections, sys
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
files=[f for f in sorted(glob.glob(SEG+'/*.json')) if not f.endswith('_analyzed.json')]
PRON={'tad','idam','etad','adas','kim','yad','asmad','yuṣmad','mad','tvad','sva','bhavat','ena','ka','ya','sa','enad'}
SPEAK={'vac','brū','ah','vad','abhidhā'}
SYM=['BOL','EOL','NOUN','PRON','VFIN','PART','CONV','INF','GDV','CPD','INDC','X']
def pos(lem,tag):
    if 'Compound' in tag: return 'CPD'
    if 'Person=' in tag: return 'VFIN'
    if 'VerbForm=Part' in tag: return 'PART'
    if 'VerbForm=Conv' in tag: return 'CONV'
    if 'VerbForm=Inf' in tag: return 'INF'
    if 'VerbForm=Gdv' in tag: return 'GDV'
    if 'Case=' in tag: return 'PRON' if lem in PRON else 'NOUN'
    if tag.strip() in ('','_'): return 'INDC'
    return 'X'
uni_cols=[f'P_{s}' for s in SYM if s not in('BOL','EOL')]
bg_cols=[f'B_{a}_{b}' for a in SYM for b in SYM]
combo_cols=['c_ppp','c_pres_opt','c_pres_subj','c_adverb_indc']
hdr=uni_cols+bg_cols+combo_cols
rows={}; nfa=0
for fp in files:
    try: d=json.load(open(fp))
    except: continue
    if not any(s.get('full_analysis') for s in d[:80]): continue
    nfa+=1; t=os.path.basename(fp)[:-5]
    uni=collections.Counter(); bg=collections.Counter(); combo=collections.Counter()
    tot=0; ntbg=0; prev=None
    for s in d:
        fa=s.get('full_analysis')
        if not fa: continue
        # preprocessing: skip "X uvāca" dialogue openings
        if len(fa)<=3 and (fa[-1].get('lemma') in SPEAK): continue
        # dedup exact-repeat consecutive lines
        sig=(s.get('analyzed') or '').strip()
        if sig and sig==prev: continue
        prev=sig
        seq=['BOL']
        for x in fa:
            lem=x.get('lemma') or ''; tag=x.get('tag') or ''
            p=pos(lem,tag); seq.append(p); tot+=1
            uni[p]+=1
            if 'VerbForm=Part' in tag and 'Tense=Past' in tag: combo['c_ppp']+=1
            if 'Tense=Present' in tag and 'Mood=Optative' in tag: combo['c_pres_opt']+=1
            if 'Tense=Present' in tag and 'Mood=Sub' in tag: combo['c_pres_subj']+=1
            if p=='INDC': combo['c_adverb_indc']+=1
        seq.append('EOL')
        for i in range(len(seq)-1):
            bg[(seq[i],seq[i+1])]+=1; ntbg+=1
    if tot<500: continue
    feat={}
    for s in SYM:
        if s in('BOL','EOL'): continue
        feat[f'P_{s}']=uni[s]/tot
    for a in SYM:
        for b in SYM:
            feat[f'B_{a}_{b}']=bg[(a,b)]/ntbg if ntbg else 0
    for c in combo_cols: feat[c]=combo[c]/tot
    rows[t]=feat
    if nfa%300==0: print(f"...{nfa} files",file=sys.stderr)
with open('pos_features.tsv','w') as out:
    out.write("text\t"+"\t".join(hdr)+"\n")
    for t,f in rows.items():
        out.write(t+"\t"+"\t".join(f"{f[c]:.6g}" for c in hdr)+"\n")
print(f"DONE: {len(rows)} texts, {len(hdr)} POS features",file=sys.stderr)

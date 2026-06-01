import json, glob, os, collections, sys, pickle
SEG=os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
files=[f for f in sorted(glob.glob(SEG+'/*.json')) if not f.endswith('_analyzed.json')]
lf=pickle.load(open('lemma_freq.pkl','rb')); STOP=set(l for l,_ in lf.most_common(100))
PRON={'tad','idam','etad','adas','kim','yad','asmad','yuṣmad','mad','tvad','sva','bhavat','ena','ka','ya','sa','enad'}
SPEAK={'vac','brū','ah','vad','abhidhā'}; AOR={'s','is','red','root'}
SYM=['BOL','EOL','NOUN','PRON','VFIN','PART','CONV','INF','GDV','CPD','INDC','X']
def POS(lem,tag):
    if 'Compound' in tag: return 'CPD'
    if 'Person=' in tag: return 'VFIN'
    if 'VerbForm=Part' in tag: return 'PART'
    if 'VerbForm=Conv' in tag: return 'CONV'
    if 'VerbForm=Inf' in tag: return 'INF'
    if 'VerbForm=Gdv' in tag: return 'GDV'
    if 'Case=' in tag: return 'PRON' if lem in PRON else 'NOUN'
    if tag.strip() in ('','_'): return 'INDC'
    return 'X'
MORPHK=['Tense=Present','Tense=Past','Tense=Future','Tense=Imperativef','Mood=Optative','Mood=Imperative',
        'Mood=Sub','Mood=Jus','VerbForm=Part','VerbForm=Conv','VerbForm=Gdv','VerbForm=Inf','Voice=Passive',
        'Formation=peri','Formation=aor','Case=Nominative','Case=Accusative','Case=Genitive','Case=Instrumental',
        'Case=Locative','Case=Vocative','Case=Ablative','Case=Dative']
uni_cols=[f'P_{s}' for s in SYM if s not in('BOL','EOL')]
bg_cols=[f'B_{a}_{b}' for a in SYM for b in SYM]
dense_cols=(['cpd_member_rate','mean_cpd_len','cpd_ge3_rate','finite_rate']
            +[k.replace('=','_') for k in MORPHK]+uni_cols+bg_cols+['c_ppp','c_pres_opt','c_pres_subj'])

def chunk_feats(segs):
    uni=collections.Counter(); bg=collections.Counter(); mc=collections.Counter(); cb=collections.Counter()
    tot=0; ntbg=0; cpd_lens=[]; cpd_mem=0; lemseq=[]
    for fa in segs:
        seq=['BOL']; run=0
        for x in fa:
            tot+=1; lem=x.get('lemma') or ''; tag=x.get('tag') or ''
            p=POS(lem,tag); seq.append(p); uni[p]+=1
            lemseq.append(lem if lem in STOP else None)
            if 'Compound' in tag: run+=1; cpd_mem+=1
            else:
                if run>0: cpd_lens.append(run+1)
                run=0
            for part in tag.split(','):
                part=part.strip()
                if '=' in part:
                    k,v=part.split('=',1)
                    if k in('Tense','Mood','VerbForm','Voice','Case'): mc[f'{k}={v}']+=1
                    elif k=='Formation': mc['Formation=peri' if v=='peri' else('Formation=aor' if v in AOR else '')]+=1
                    elif k=='Person': mc['Finite']+=1
            if 'VerbForm=Part' in tag and 'Tense=Past' in tag: cb['c_ppp']+=1
            if 'Tense=Present' in tag and 'Mood=Optative' in tag: cb['c_pres_opt']+=1
            if 'Tense=Present' in tag and 'Mood=Sub' in tag: cb['c_pres_subj']+=1
        if run>0: cpd_lens.append(run+1)
        seq.append('EOL')
        for i in range(len(seq)-1): bg[(seq[i],seq[i+1])]+=1; ntbg+=1
    if tot<200: return None,None,tot
    nc=len(cpd_lens)
    f={'cpd_member_rate':cpd_mem/tot,'mean_cpd_len':(sum(cpd_lens)/nc if nc else 0),
       'cpd_ge3_rate':(sum(1 for L in cpd_lens if L>=3)/nc if nc else 0),'finite_rate':mc['Finite']/tot}
    for k in MORPHK: f[k.replace('=','_')]=mc[k]/tot
    for s in SYM:
        if s not in('BOL','EOL'): f[f'P_{s}']=uni[s]/tot
    for a in SYM:
        for b in SYM: f[f'B_{a}_{b}']=bg[(a,b)]/ntbg if ntbg else 0
    for c in ['c_ppp','c_pres_opt','c_pres_subj']: f[c]=cb[c]/tot
    # fw-grams
    fw=collections.Counter()
    for i in range(len(lemseq)-1):
        a,b=lemseq[i],lemseq[i+1]
        if a and b: fw[f'{a}_{b}']+=1
    for i in range(len(lemseq)-2):
        a,b,c=lemseq[i],lemseq[i+1],lemseq[i+2]
        if a and b and c: fw[f'{a}_{b}_{c}']+=1
    return f,fw,tot

CHUNK=100
dense_out=open('chunks_dense.tsv','w'); dense_out.write("chunk\twork\tn_tokens\t"+"\t".join(dense_cols)+"\n")
fwmap={}; nfa=0; nchunks=0
for fp in files:
    try: d=json.load(open(fp))
    except: continue
    if not any(s.get('full_analysis') for s in d[:80]): continue
    nfa+=1; t=os.path.basename(fp)[:-5]
    # preprocessing -> list of kept segments (token-lists)
    kept=[]; prev=None
    for s in d:
        fa=s.get('full_analysis')
        if not fa: continue
        if len(fa)<=3 and fa[-1].get('lemma') in SPEAK: continue
        sig=(s.get('analyzed') or '').strip()
        if sig and sig==prev: continue
        prev=sig; kept.append(fa)
    # chunk into groups of CHUNK segments
    k=0
    for st in range(0,len(kept),CHUNK):
        segs=kept[st:st+CHUNK]
        f,fw,tot=chunk_feats(segs)
        if f is None: continue
        cid=f"{t}#{k}"; k+=1; nchunks+=1
        dense_out.write(f"{cid}\t{t}\t{tot}\t"+"\t".join(f"{f[c]:.6g}" for c in dense_cols)+"\n")
        fwmap[cid]=fw
    if nfa%300==0: print(f"...{nfa} files, {nchunks} chunks",file=sys.stderr)
dense_out.close()
pickle.dump(fwmap,open('chunks_fw.pkl','wb'))
print(f"DONE: {nfa} files -> {nchunks} chunks; dense cols={len(dense_cols)}",file=sys.stderr)

"""Rebuild text-information.json (titles + descriptions + edition) from the current
DharmaNexus metadata (~/data/dharmanexus-sanskrit/SA_files.json), merging in the
critical-edition / archive.org scan data from sanskrit-editions.json.

Refreshes metadata-derived fields (title, author, tradition, genre, date_estimate,
summary, history, edition, archive_url) from the latest SA_files.json; preserves any
curated-only fields already present (related_works, confidence, sources, file_name).
"""
import json, re, os

SA_PATH=os.path.expanduser('~/data/dharmanexus-sanskrit/SA_files.json')
SA={d['textname']:d for d in json.load(open(SA_PATH))}
ED={e['id']:e for e in json.load(open('sanskrit-editions.json'))['texts']}
TI=json.load(open('text-information.json')) if os.path.exists('text-information.json') else {}

def section(md,name):
    m=re.search(r'^## '+re.escape(name)+r'\s*\n(.*?)(?=^## |\Z)', md, re.S|re.M)
    return m.group(1).strip() if m else ''
def field(md,name):
    m=re.search(r'\*\*'+re.escape(name)+r':\*\*\s*(.+?)(?=\n\n|\n\*\*|\n## |\Z)', md, re.S)
    return ' '.join(m.group(1).split()) if m else ''
def summary_of(md):
    ws=section(md,'Web Summary')
    if not ws: return ''
    m=re.search(r'\*\*Estimated date:\*\*.*?\n(.+?)(?=\*\*History:\*\*|\Z)', ws, re.S)
    if m: return ' '.join(m.group(1).split())
    body=re.sub(r'_[^_]*_','',ws,count=1)                 # drop AI disclaimer
    body=re.sub(r'\*\*[^*]+:\*\*[^\n]*','',body)           # drop any **field:** lines
    return ' '.join(body.split())
def history_of(md):
    ws=section(md,'Web Summary')
    m=re.search(r'\*\*History:\*\*\s*(.+)\Z', ws, re.S)
    return ' '.join(m.group(1).split()) if m else ''
def edition_for(tn,md):
    """(edition string, archive_url) — editions.json first (has scan links), then the
    GRETIL **Edition:** header, then the new-corpus ## Edition / **Source:** prose."""
    e=ED.get(tn)
    if e and e.get('edition'):
        au=e.get('archive_url','') if e.get('archive_confidence') in ('high','medium') else ''
        return e['edition'], au
    ge=field(md,'Edition')
    if ge: return ge, ''
    # new-corpus texts: pull the bibliographic sentence from the ## Edition prose
    # (names the editio princeps / editor / year), not the descriptive opening.
    sec=re.sub(r'\*','',section(md,'Edition'))
    if sec:
        sents=re.split(r'(?<=[.])\s', sec)
        for s in sents:
            if re.search(r'[Ee]ditio princeps|[Ee]dited|critical edition|edition (?:by|of|and)|\b(?:1[89]\d\d|20\d\d)\b', s):
                return ' '.join(s.split()), ''
    src=field(md,'Source')                              # e.g. "Edition by Shinichi Tsuda"
    if src: return src, ''
    return (' '.join(sents[0].split()) if sec else ''), ''

# NOTE: SA_files.json uniformtitle/author are ASCII-folded and occasionally garbled
# (Bhagavadgītā->Bhagavadgīta, Karṇaparvan->"7"), i.e. LOWER quality than the already-
# curated text-information.json, whose descriptive fields were themselves built from this
# same Web Summary. So for EXISTING entries we PRESERVE the curated fields and only (a) add
# edition / archive scan links and (b) backfill any field that is missing/empty. NEW works
# (no prior entry) are built in full from SA_files — best available source for them.
out=dict(TI); added=0; edadded=0; backfilled=0
for tn,d in SA.items():
    md=d.get('raw_metadata','') or ''
    title=(d.get('uniformtitle') or '').strip() or (d.get('displayName') or '').split(':')[-1].strip() or tn
    derived={'tradition':field(md,'Tradition'),'genre':field(md,'Genre'),
             'date_estimate':field(md,'Estimated date'),'summary':summary_of(md),
             'history':history_of(md),'title':title,'author':d.get('author','')}
    ed,au=edition_for(tn,md)
    if tn not in TI:                                   # NEW work: build in full
        e={'work_id':d.get('filenr',''), **{k:derived.get(k,'') for k in
            ['title','author','tradition','genre','date_estimate','summary','history']}}
        if ed: e['edition']=ed
        if au: e['archive_url']=au
        out[tn]=e; added+=1
    else:                                              # EXISTING: preserve, augment only
        e=out[tn]
        if ed and not e.get('edition'): e['edition']=ed; edadded+=1
        if au and not e.get('archive_url'): e['archive_url']=au
        for k,v in derived.items():                    # backfill only genuinely empty fields
            if v and not (e.get(k) or '').strip(): e[k]=v; backfilled+=1

json.dump(out,open('text-information.json','w'),ensure_ascii=False,indent=1)
ned=sum(1 for v in out.values() if v.get('edition'))
nau=sum(1 for v in out.values() if v.get('archive_url'))
print(f"text-information.json: {len(out)} entries  (+{added} new works; "
      f"existing: +{edadded} editions, {backfilled} empty fields backfilled)")
print(f"  with edition: {ned}   with archive.org scan link: {nau}")

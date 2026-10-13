import csv
def num(x):
    x=(x or '').strip().replace(',','')
    if x=='' : return None
    try: return float(x)
    except: return None
rows=list(csv.DictReader(open('metadata.tsv'),delimiter='\t'))
meta={}
for r in rows:
    fn=(r['File name'] or '').strip()
    if not fn: continue
    nb=num(r.get('not before')); na=num(r.get('not after'))
    # fix reversed intervals (e.g. lagrvvju nb=499 na=300)
    if nb is not None and na is not None and nb>na: nb,na=na,nb
    meta[fn]={'title':(r.get('Full Sanskrit title') or '').strip(),
              'author':(r.get('Author(s)') or '').strip(),
              'nb':nb,'na':na}
# Every other DharmaNexus SA text with segments becomes a work of its own (undated
# unless an anchor file dates it), except files that an existing grouping key
# (a meta key without its own segment file, e.g. SA_GP12_skp) deliberately
# collects. Keys ending in '_' are malformed and would act as catch-all prefixes
# (SA_GV06_ swallowed every GV06 text), so they are dropped.
import json, os
for k in [k for k in meta if k.endswith('_')]:
    del meta[k]
SEG = os.path.expanduser('~/data/dharmanexus-sanskrit/segments')
segs = {f[:-5] for f in os.listdir(SEG) if f.endswith('.json') and not f.endswith('_analyzed.json')}
groups = [k for k in meta if k not in segs and any(g.startswith(k) for g in segs)]
cat = json.load(open(os.path.expanduser('~/data/dharmanexus-sanskrit/SA_files.json'), encoding='utf-8'))
added = 0
for e in cat:
    fn = e['filename']
    if fn in meta or fn not in segs or fn.endswith('_') or any(fn.startswith(g) for g in groups):
        continue
    meta[fn] = {'title': e.get('uniformtitle') or e.get('displayName') or '', 'author': e.get('author', ''),
                'nb': None, 'na': None}
    added += 1
print("added from SA_files.json:", added)
# works whose text is no longer in DharmaNexus (e.g. the removed Sanskrit reconstructions from
# Tibetan/Chinese) are dropped: no segment file and no segment file under the key as a prefix
gone = [k for k in meta if k not in segs and not any(g.startswith(k) for g in segs)]
for k in gone:
    del meta[k]
print("dropped (no text in DharmaNexus):", len(gone))
# save
json.dump(meta,open('meta.json','w'),ensure_ascii=False)
print("meta works:",len(meta))
print("with both dates:",sum(1 for m in meta.values() if m['nb'] is not None and m['na'] is not None))

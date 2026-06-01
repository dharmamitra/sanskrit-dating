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
# save
import json
json.dump(meta,open('meta.json','w'),ensure_ascii=False)
print("meta works:",len(meta))
print("with both dates:",sum(1 for m in meta.values() if m['nb'] is not None and m['na'] is not None))

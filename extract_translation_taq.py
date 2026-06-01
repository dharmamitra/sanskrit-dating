import json, os, glob, collections, sys
LAG=20
T=json.load(open(os.path.expanduser('~/data/zh/cbeta-metadata/work-info/T.json')))
def zh_info(num):
    for key in (f"T{num}", f"T{num[:4]}"):
        if key in T: return key, T[key].get('time_from'), T[key].get('title','')
    return None,None,None
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return None
PA=os.path.expanduser('~/code/mitra-multilingual-matching/pair_alignments')
best={}
for f in glob.glob(PA+'/SA_*__ZH_*.json'):
    sa,zh=os.path.basename(f)[:-5].split('__'); num=zh.split('_',2)[2]
    zid,tf,title=zh_info(num)
    if tf is None: continue
    w=m2w(sa)
    if w is None: continue
    if w not in best or tf<best[w][0]:
        best[w]=(tf, tf-LAG, zid, title, sa)
with open('translation_taq.tsv','w') as out:
    out.write("work\tnot_after\ttransl_date\ttaisho\tsanskrit_id\ttitle\n")
    for w,(tf,taq,zid,title,sa) in sorted(best.items(),key=lambda kv:kv[1][1]):
        out.write(f"{w}\t{taq}\t{tf}\t{zid}\t{sa}\t{title}\n")
print(f"wrote translation_taq.tsv: {len(best)} works with a Chinese-translation terminus ante quem (lag={LAG}y)",file=sys.stderr)

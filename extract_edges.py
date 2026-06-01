import gzip, json, os, glob, collections, sys
files = sorted(glob.glob('matches/*.ndjson.gz'))
edges = collections.defaultdict(lambda:[0,0.0,0])  # (root,par)->[count,sumscore,sum_parlen]
nodes = collections.Counter()
def tid(seg): return seg.rsplit(':',1)[0]
for i,fp in enumerate(files):
    try:
        with gzip.open(fp,'rt') as fh:
            for line in fh:
                if not line.strip(): continue
                try: d=json.loads(line)
                except: continue
                rs=d.get('root_segnr') or []
                ps=d.get('par_segnr') or []
                if not rs or not ps: continue
                r=tid(rs[0]); p=tid(ps[0])
                nodes[r]+=1; nodes[p]+=1
                if r==p: continue
                key=(r,p)
                e=edges[key]; e[0]+=1; e[1]+=float(d.get('score') or 0); e[2]+=int(d.get('par_length') or 0)
    except Exception as ex:
        print("ERR",fp,ex,file=sys.stderr)
    if (i+1)%200==0: print(f"...{i+1}/{len(files)} files, {len(edges)} edges, {len(nodes)} nodes",file=sys.stderr)
with open('edges_raw.tsv','w') as out:
    out.write("root\tpar\tcount\tsum_score\tsum_parlen\n")
    for (r,p),(c,s,pl) in edges.items():
        out.write(f"{r}\t{p}\t{c}\t{s:.4f}\t{pl}\n")
with open('nodes_raw.tsv','w') as out:
    out.write("text_id\tn_segments_seen\n")
    for n,c in nodes.most_common():
        out.write(f"{n}\t{c}\n")
print(f"DONE files={len(files)} edges={len(edges)} nodes={len(nodes)}",file=sys.stderr)

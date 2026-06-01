import gzip, json, glob, collections, sys
files = sorted(glob.glob('matches/*.ndjson.gz'))
edges = collections.defaultdict(lambda:[0,0,0,0.0])  # (root,par)->[count,sum_root_len,sum_par_len,sum_score]
def tid(s): return s.rsplit(':',1)[0]
for i,fp in enumerate(files):
    try:
        with gzip.open(fp,'rt') as fh:
            for line in fh:
                if not line.strip(): continue
                try: d=json.loads(line)
                except: continue
                rs=d.get('root_segnr') or []; ps=d.get('par_segnr') or []
                if not rs or not ps: continue
                r=tid(rs[0]); p=tid(ps[0])
                if r==p: continue
                e=edges[(r,p)]
                e[0]+=1
                e[1]+=int(d.get('root_length') or 0)
                e[2]+=int(d.get('par_length') or 0)
                e[3]+=float(d.get('score') or 0)
    except Exception as ex: print("ERR",fp,ex,file=sys.stderr)
    if (i+1)%300==0: print(f"...{i+1}/{len(files)} {len(edges)} edges",file=sys.stderr)
with open('edges_dir.tsv','w') as out:
    out.write("root\tpar\tcount\tsum_root_len\tsum_par_len\tsum_score\n")
    for (r,p),(c,rl,pl,s) in edges.items():
        out.write(f"{r}\t{p}\t{c}\t{rl}\t{pl}\t{s:.3f}\n")
print(f"DONE {len(edges)} directed edges",file=sys.stderr)

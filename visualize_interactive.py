import numpy as np, collections as C, json, os
import plotly.graph_objects as go
rng=np.random.default_rng(0)
meta=json.load(open('meta.json')); mk=sorted(meta,key=len,reverse=True)
def m2w(t):
    for k in mk:
        if t==k or t.startswith(k): return k
    return t
rows=[r.split('\t') for r in open('dated_gibbs_full.tsv').read().splitlines()[1:]]
def coll(w):
    p=w.split('_'); return p[1] if len(p)>1 else ''

# ---- harvest author / note text per work-id for search ----
notes=C.defaultdict(list)
for f in ['manual_constraints.tsv','researched_anchors.tsv','chronbmm_priors.tsv','dcs_anchors.tsv']:
    if not os.path.exists(f): continue
    for l in open(f).read().splitlines()[1:]:
        p=l.split('\t')
        if len(p)<2: continue
        typ=p[0]; note=p[-1]
        # only notes that describe the work ITSELF; SKIP 'order' (they name the OTHER work, e.g. "X after Vasubandhu")
        if typ in ('anchor','not_before','not_after') and len(p)>=2:
            notes[p[1]].append(note)
# vedic anchor labels by prefix
ved=[]
if os.path.exists('vedic_anchors.tsv'):
    for l in open('vedic_anchors.tsv').read().splitlines()[1:]:
        p=l.split('\t'); ved.append((p[0],p[3] if len(p)>3 else ''))
def searchstr(w,title):
    parts=[title, w, meta.get(m2w(w),{}).get('author','')]
    parts+=notes.get(w,[])
    for pre,lab in ved:
        if w.startswith(pre): parts.append(lab)
    return ' '.join(x for x in parts if x).lower()

CAT={}
for c in ['GV','GV00','GV01','GV02','GV03','GV04']: CAT[c]='Veda (Saṃhitā/Brāhmaṇa)'
CAT['GV05']='Upaniṣad'
for c in ['GV06','GSD36','GSD37']: CAT[c]='Vedic Sūtra & Dharmaśāstra'
for c in ['GE07','GE09']: CAT[c]='Epic'
for c in ['GP10','GP11','GP12']: CAT[c]='Purāṇa'
for c in ['GR13','GR14','GSP30','MB']: CAT[c]='Tantra & Āgama'
for c in ['GK16','GK17','GK18','T12']: CAT[c]='Poetics & Dramaturgy'
for c in ['GK19','GK20','GK23','T13']: CAT[c]='Kāvya & Drama'
for c in ['GK21','GK22','GS38','GS39','T15']: CAT[c]='Story, Fable & Nīti'
for c in ['GS24','GS25']: CAT[c]='Grammar & Lexicon'
for c in ['GS26','GR12','GSP27','GSP28','GSP29','GSP31','GSP32','GSP33','GSP34','GSP35','GSP36']: CAT[c]='Philosophy (Brahmanical/Jain)'
for c in ['GS40','T14']: CAT[c]='Medicine (Āyurveda)'
CAT['GS41']='Astronomy & Mathematics'
for c in ['K01','K02','K03','K05','K06','K07','K08','K09','K10','K12','K12X','K14','T08','T09','T17']: CAT[c]='Buddhist Sūtra & Vinaya'
for c in ['T03','T04','T05','T06','T07','T10','T11']: CAT[c]='Buddhist Philosophy'
for c in ['T01','T02','T16']: CAT[c]='Buddhist Tantra & Hymn'
LANES=['Veda (Saṃhitā/Brāhmaṇa)','Upaniṣad','Vedic Sūtra & Dharmaśāstra','Epic','Purāṇa','Tantra & Āgama',
       'Poetics & Dramaturgy','Kāvya & Drama','Story, Fable & Nīti','Grammar & Lexicon',
       'Philosophy (Brahmanical/Jain)','Medicine (Āyurveda)','Astronomy & Mathematics',
       'Buddhist Sūtra & Vinaya','Buddhist Philosophy','Buddhist Tantra & Hymn']
COL={'Veda (Saṃhitā/Brāhmaṇa)':'#D35400','Upaniṣad':'#E67E22','Vedic Sūtra & Dharmaśāstra':'#F0A93B',
     'Epic':'#8E44AD','Purāṇa':'#B7950B','Tantra & Āgama':'#C2185B','Poetics & Dramaturgy':'#117A65',
     'Kāvya & Drama':'#229954','Story, Fable & Nīti':'#52BE80','Grammar & Lexicon':'#1A5276',
     'Philosophy (Brahmanical/Jain)':'#2471A3','Medicine (Āyurveda)':'#5499C7','Astronomy & Mathematics':'#2E86C1',
     'Buddhist Sūtra & Vinaya':'#A93226','Buddhist Philosophy':'#CB4335','Buddhist Tantra & Hymn':'#E6796B'}
li={t:i for i,t in enumerate(LANES)}
def yr(v):
    v=int(round(v)); return f"{-v} BCE" if v<0 else f"{v} CE"
G=C.defaultdict(lambda: dict(x=[],y=[],s=[],op=[],cd=[]))
for r in rows:
    cat=CAT.get(coll(r[0]))
    if cat is None: continue
    try: md,lo,hi=float(r[6]),float(r[7]),float(r[8])
    except: continue
    nch=float(r[2]); title=r[9].strip() or r[0]
    disp=title if not title.lower().startswith('unknown') else title.split(':',1)[-1].strip()+f" ({r[0]})"
    base=float(np.interp(hi-lo,[100,900],[0.85,0.10]))
    g=G[cat]
    g['x'].append(md); g['y'].append(li[cat]+rng.uniform(-0.34,0.34))
    g['s'].append(float(np.clip(5+2.4*np.sqrt(nch),5,38))); g['op'].append(base)
    g['cd'].append([disp,yr(md),f"{yr(lo)} – {yr(hi)}",cat,('anchored' if r[1]=='anchor' else 'inferred'),
                    int(nch*100), searchstr(r[0],title), base, r[0]])
fig=go.Figure()
eras=[(-1700,-500,'Vedic','#FFF3E0'),(-500,200,'Epic & Sūtra','#F3E5F5'),(200,650,'Classical','#E8F5E9'),
      (650,1200,'Early Medieval','#E3F2FD'),(1200,1900,'Late Medieval','#FFF8E1')]
for x0,x1,lbl,c in eras:
    fig.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=0.5,line_width=0,layer='below',
                  annotation_text=lbl,annotation_position='top',annotation_font_size=11,annotation_font_color='#999')
for cat in LANES:
    g=G[cat]
    fig.add_trace(go.Scattergl(x=g['x'],y=g['y'],mode='markers',name=f"{cat} ({len(g['x'])})",
        marker=dict(size=g['s'],color=COL[cat],opacity=g['op'],line=dict(width=0.3,color='white')),
        customdata=g['cd'],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}  ·  95%% CI: %{customdata[2]}"
                      "<br>%{customdata[3]} — %{customdata[4]}  ·  ~%{customdata[5]:,} lines"
                      "<br><i>click dot → open on DharmaNexus ↗</i><extra></extra>"))
fig.update_layout(template='plotly_white',height=1000,hovermode='closest',
    xaxis=dict(title='Year',range=[-1700,1900],tickmode='array',
               tickvals=[-1500,-1000,-500,0,500,1000,1500],
               ticktext=['1500 BCE','1000 BCE','500 BCE','1 CE','500 CE','1000 CE','1500 CE'],
               zeroline=True,zerolinecolor='#bbb',gridcolor='#eee'),
    yaxis=dict(tickmode='array',tickvals=list(range(len(LANES))),ticktext=LANES,
               autorange='reversed',showgrid=True,gridcolor='#f0f0f0',range=[-0.6,len(LANES)-0.4]),
    legend=dict(title='Genre',font=dict(size=10),itemsizing='constant'),
    margin=dict(l=240,r=20,t=20,b=50))
chart=fig.to_html(include_plotlyjs=True,full_html=False,div_id='chart')
N=sum(len(G[c]['x']) for c in LANES)
HTML=f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Sanskrit Chronology</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa}}
 #bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
 h1{{margin:0 0 2px;font-size:19px}} .sub{{color:#777;font-size:12.5px}}
 #search{{font-size:15px;padding:8px 12px;width:340px;border:2px solid #2471A3;border-radius:7px;margin-top:8px}}
 #count{{margin-left:12px;color:#2471A3;font-weight:600}} #clr{{margin-left:6px;cursor:pointer;color:#888;font-size:13px}}
 .ex{{color:#aaa;font-size:12px;margin-left:6px}} .ex b{{color:#2471A3;cursor:pointer;font-weight:600}}
</style></head><body>
<div id="bar">
 <h1>A Computed Chronology of Sanskrit Literature</h1>
 <div class="sub">{N} texts · hover for details · <b>click a dot to open it on DharmaNexus</b> · drag to zoom · click legend to toggle genres · dot size = text length</div>
 <input id="search" placeholder="Search author or title…" autocomplete="off">
 <span id="count"></span><span id="clr">✕ clear</span>
 <span class="ex">try: <b onclick="setq('vasubandhu')">Vasubandhu</b> · <b onclick="setq('kālidāsa')">Kālidāsa</b> · <b onclick="setq('nāgārjuna')">Nāgārjuna</b> · <b onclick="setq('abhinavagupta')">Abhinavagupta</b> · <b onclick="setq('purāṇa')">Purāṇa</b></span>
</div>
{chart}
<script>
const gd=document.getElementById('chart'), inp=document.getElementById('search'), cnt=document.getElementById('count');
function apply(){{
 const term=inp.value.trim().toLowerCase();
 const ops=[]; let m=0,xmin=1e9,xmax=-1e9;
 for(let i=0;i<gd.data.length;i++){{
   const cd=gd.data[i].customdata, xs=gd.data[i].x;
   ops.push(cd.map((row,j)=>{{
     if(term==="") return row[7];
     const hit=row[6].indexOf(term)>=0;
     if(hit){{m++; if(xs[j]<xmin)xmin=xs[j]; if(xs[j]>xmax)xmax=xs[j];}}
     return hit?0.95:0.025;
   }}));
 }}
 Plotly.restyle(gd,{{'marker.opacity':ops}});
 if(term===""){{cnt.textContent=""; Plotly.relayout(gd,{{'xaxis.range':[-1700,1900]}});}}
 else{{cnt.textContent=m+" match"+(m==1?"":"es");
   if(m>0){{const pad=Math.max(60,(xmax-xmin)*0.12); Plotly.relayout(gd,{{'xaxis.range':[xmin-pad,xmax+pad]}});}}}}
}}
function setq(q){{inp.value=q; apply(); inp.focus();}}
gd.on('plotly_click',function(e){{var w=e.points[0].customdata[8];
  if(w) window.open('https://dharmamitra.org/nexus/db/sa/'+w+'/text','_blank');}});
inp.addEventListener('input',apply);
document.getElementById('clr').addEventListener('click',()=>{{inp.value="";apply();}});
</script></body></html>"""
open('sanskrit_chronology_interactive.html','w').write(HTML)
print("wrote sanskrit_chronology_interactive.html  (%.1f MB, %d texts, searchable)"%(os.path.getsize('sanskrit_chronology_interactive.html')/1e6,N))

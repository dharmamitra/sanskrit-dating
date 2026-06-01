import numpy as np, collections as C, json, os, re, unicodedata, html as _html
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

# ---- authoritative per-work metadata + descriptions (text-information.json) ----
# Keyed by the same SA_<coll>_<slug> work-id as the dated TSV. Entries carry
# title / author / genre / tradition / date_estimate / summary / history.
TI=json.load(open('text-information.json'))
def _root(w):
    p=w.split('_'); return '_'.join(p[:2])
_ti_by_root=C.defaultdict(list)
for k in TI: _ti_by_root[_root(k)].append(k)
def tinfo(w):
    """Exact match, else longest work-id within the same collection that is a prefix
    of (or prefixed by) this work-id — recovers split/sub-part works. {} if none."""
    if w in TI: return TI[w]
    cands=[k for k in _ti_by_root.get(_root(w),[]) if w.startswith(k) or k.startswith(w)]
    if cands: return TI[max(cands,key=len)]
    return {}

def author_for_search(a):
    # The author field is free text whose SPURIOUS names almost always sit inside
    # parenthetical qualifiers — aliases, role tags, and cross-references such as
    #   "anonymous/traditional (sometimes attributed to Vasubandhu or his school)"  (Tarkaśāstra)
    #   "Maitreyanātha (attributed; commentator on Nāgārjuna)"                       (Bhavasaṅkrāntiṭīkā)
    #   "Śaṅkarasvāmin (... sometimes ascribed to Dignāga)"                          (Nyāyapraveśaka)
    #   "attributed to Āryaśūra (also given as ... Aśvaghoṣa)"                        (Subhāṣitaratnakaraṇḍaka)
    # so an author query was surfacing works that merely reference that name. Drop the
    # parentheticals and index only the lead attribution; co-authors separated by ';'
    # (e.g. "Maitreya (kārikās); Vasubandhu (bhāṣya)") survive — each is a real author.
    return re.sub(r'\([^)]*\)', ' ', a or '')

def deaccent(s):
    # Strip IAST diacritics via NFKD decomposition (ā→a, ū→u, ṛ→r, ś/ṣ→s, ṅ/ñ/ṇ→n,
    # ṭ/ḍ→t/d, ḥ→h, ṃ→m, …) so a plain-ASCII query like "bhumi" matches "bhūmi".
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def searchstr(w,title):
    # Search index = STRUCTURED fields only (title, work-id, author, genre).
    # Deliberately excludes free-text descriptions / anchor notes — those mention
    # OTHER authors (e.g. "cited by Vasubandhu") and caused cross-pollution where an
    # author query surfaced every work that merely referenced that author.
    e=tinfo(w)
    parts=[e.get('title') or title, w,
           author_for_search(e.get('author') or meta.get(m2w(w),{}).get('author','')),
           e.get('genre','')]
    base=' '.join(x for x in parts if x).lower()
    # Append a diacritic-free duplicate so diacritic-insensitive queries also hit.
    folded=deaccent(base)
    return base if folded==base else base+' '+folded

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
def likely_range(lo,hi):
    # An informed, deliberately imprecise band: round the 95% credible interval
    # OUTWARD to the nearest 25 years so the tooltip never implies false precision.
    lo=int(np.floor(lo/25.0)*25); hi=int(np.ceil(hi/25.0)*25)
    return f"c. {yr(lo)} – {yr(hi)}"
G=C.defaultdict(lambda: dict(x=[],y=[],s=[],op=[],cd=[]))
for r in rows:
    cat=CAT.get(coll(r[0]))
    if cat is None: continue
    try: md,lo,hi=float(r[6]),float(r[7]),float(r[8])
    except: continue
    e=tinfo(r[0])
    nch=float(r[2]); title=(e.get('title') or '').strip() or r[9].strip() or r[0]
    disp=title if not title.lower().startswith('unknown') else title.split(':',1)[-1].strip()+f" ({r[0]})"
    base=float(np.interp(hi-lo,[100,900],[0.85,0.10]))
    def esc(s): return _html.escape((s or '').strip())
    g=G[cat]
    g['x'].append(md); g['y'].append(li[cat]+rng.uniform(-0.34,0.34))
    g['s'].append(float(np.clip(5+2.4*np.sqrt(nch),5,38))); g['op'].append(base)
    g['cd'].append([disp,likely_range(lo,hi),f"{yr(lo)} – {yr(hi)}",cat,('anchored' if r[1]=='anchor' else 'inferred'),
                    int(nch*100), searchstr(r[0],title), base, r[0],
                    esc(e.get('author') or meta.get(m2w(r[0]),{}).get('author','')),  # 9
                    esc(e.get('genre','')),        # 10
                    esc(e.get('tradition','')),    # 11
                    esc(e.get('date_estimate','')),# 12
                    esc(e.get('summary','')),      # 13
                    esc(e.get('history','')),      # 14
                    ("  ·  "+esc(e.get('author') or meta.get(m2w(r[0]),{}).get('author','')))
                        if (e.get('author') or meta.get(m2w(r[0]),{}).get('author','')) else ""])  # 15 hover author
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
        hovertemplate="<b>%{customdata[0]}</b>%{customdata[15]}"
                      "<br>likely range: <b>%{customdata[1]}</b>  <i>(approximate)</i>"
                      "<br>%{customdata[3]} — %{customdata[4]}  ·  ~%{customdata[5]:,} lines"
                      "<br><i>uncertain estimate, not a precise date · click → DharmaNexus ↗</i><extra></extra>"))
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
 #detail{{position:fixed;right:16px;bottom:16px;width:370px;max-height:60vh;overflow-y:auto;
   background:#fff;border:1px solid #ddd;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.13);
   padding:14px 16px;font-size:13px;line-height:1.5;z-index:20;display:none}}
 #detail .dt{{font-weight:700;font-size:15px;margin-bottom:2px}}
 #detail .dm{{color:#2471A3;font-weight:600;margin-bottom:1px}}
 #detail .dg{{color:#777;font-size:12px;margin-bottom:8px}}
 #detail p{{margin:0 0 8px}} #detail .hist{{color:#555;font-size:12.5px;border-top:1px solid #eee;padding-top:8px}}
 #detail .dl{{color:#aaa;font-size:11.5px}} #detail .dx{{float:right;cursor:pointer;color:#bbb;font-size:14px;margin:-4px -4px 0 0}}
 footer{{margin:8px 18px 40px;padding:18px 22px;background:#fff;border:1px solid #e5e5e5;border-radius:10px;
   color:#555;font-size:13px;line-height:1.6;max-width:1100px}}
 footer h2{{margin:0 0 8px;font-size:15px;color:#1d1d1f}}
 footer b{{color:#2471A3}} footer ol{{margin:8px 0 0;padding-left:20px}} footer li{{margin-bottom:6px}}
 footer .cav{{margin-top:12px;padding-top:10px;border-top:1px solid #eee;color:#888;font-size:12px}}
 footer a{{color:#2471A3}}
</style></head><body>
<div id="bar">
 <h1>A Computed Chronology of Sanskrit Literature</h1>
 <div class="sub">{N} texts · hover a dot for its summary · <b>click a dot to open it on DharmaNexus</b> · drag to zoom · click legend to toggle genres · dot size = text length</div>
 <input id="search" placeholder="Search author, title or genre (diacritics optional)…" autocomplete="off">
 <span id="count"></span><span id="clr">✕ clear</span>
 <span class="ex">try: <b onclick="setq('vasubandhu')">Vasubandhu</b> · <b onclick="setq('kālidāsa')">Kālidāsa</b> · <b onclick="setq('nāgārjuna')">Nāgārjuna</b> · <b onclick="setq('abhinavagupta')">Abhinavagupta</b> · <b onclick="setq('purāṇa')">Purāṇa</b></span>
</div>
{chart}
<div id="detail"></div>
<script>
const gd=document.getElementById('chart'), inp=document.getElementById('search'), cnt=document.getElementById('count');
// cache original coordinates once; filtering nulls out non-matches so they are
// neither drawn NOR hoverable (opacity/hoverinfo can't suppress Plotly's tooltip
// while a hovertemplate is set, but a null-coordinate point doesn't exist to hover).
const X0=gd.data.map(t=>t.x.slice()), Y0=gd.data.map(t=>t.y.slice());
function apply(){{
 const term=inp.value.trim().toLowerCase();
 const ops=[], XS=[], YS=[]; let m=0,xmin=1e9,xmax=-1e9;
 for(let i=0;i<gd.data.length;i++){{
   const cd=gd.data[i].customdata, x0=X0[i], y0=Y0[i];
   const op=[], xx=[], yy=[];
   for(let j=0;j<cd.length;j++){{
     if(term===""){{ op.push(cd[j][7]); xx.push(x0[j]); yy.push(y0[j]); continue; }}
     if(cd[j][6].indexOf(term)>=0){{
       m++; if(x0[j]<xmin)xmin=x0[j]; if(x0[j]>xmax)xmax=x0[j];
       op.push(0.95); xx.push(x0[j]); yy.push(y0[j]);
     }} else {{ op.push(0); xx.push(null); yy.push(null); }}  // non-match: remove from plot
   }}
   ops.push(op); XS.push(xx); YS.push(yy);
 }}
 Plotly.restyle(gd,{{'x':XS,'y':YS,'marker.opacity':ops}});
 if(term===""){{cnt.textContent=""; Plotly.relayout(gd,{{'xaxis.range':[-1700,1900]}});}}
 else{{cnt.textContent=m+" match"+(m==1?"":"es");
   if(m>0){{const pad=Math.max(60,(xmax-xmin)*0.12); Plotly.relayout(gd,{{'xaxis.range':[xmin-pad,xmax+pad]}});}}}}
}}
function setq(q){{inp.value=q; apply(); inp.focus();}}
const det=document.getElementById('detail');
gd.on('plotly_hover',function(e){{
  const d=e.points[0].customdata;
  const term=inp.value.trim().toLowerCase();          // while searching, ignore hidden non-matches
  if(term && d[6].indexOf(term)<0) return;
  let h='<span class="dx" id="detx">✕</span>';
  h+='<div class="dt">'+d[0]+'</div>';
  h+='<div class="dm">'+(d[9]?d[9]+'  ·  ':'')+'likely range: '+d[1]+'</div>';
  h+='<div class="dg">'+(d[10]||d[3])+(d[11]?'  ·  '+d[11]:'')+'</div>';
  if(d[13]) h+='<p>'+d[13]+'</p>';
  if(d[14]) h+='<p class="hist">'+d[14]+'</p>';
  h+='<div class="dl">click the dot → open on DharmaNexus ↗</div>';
  det.innerHTML=h; det.style.display='block'; det.scrollTop=0;
  document.getElementById('detx').onclick=function(){{det.style.display='none';}};
}});
gd.on('plotly_click',function(e){{var w=e.points[0].customdata[8];
  if(w) window.open('https://dharmamitra.org/nexus/db/sa/'+w+'/text','_blank');}});
inp.addEventListener('input',apply);
document.getElementById('clr').addEventListener('click',()=>{{inp.value="";apply();}});
</script>
<footer>
 <h2>How these dates were computed</h2>
 Each text's position combines two independent sources of evidence, fused in a single Bayesian model:
 <ol>
  <li><b>Scholarly priors.</b> For works with an established date in the secondary literature, that
      consensus enters as an informative prior — either a point anchor or a soft window — together with
      hard <i>before / after</i> ordering constraints drawn from known author relationships and commentaries.
      Dated <b>Chinese translations</b> are used as hard priors too: a securely dated translation fixes a
      <i>terminus ante quem</i> (the Sanskrit original must predate it), anchoring many Buddhist texts that
      have no other external date.</li>
  <li><b>Linguistic dating.</b> Every text is dated from its own language independently of any anchor,
      using a category-free stylometric clock built on morphology and function-word <i>n</i>-grams
      (rather than vocabulary, which in normatively-frozen Sanskrit is too weak a signal). This yields a
      composition-style estimate that is largely genre-independent.</li>
 </ol>
 A <b>Gibbs sampler</b> fuses these signals — sampling each text's date conditioned on its linguistic
 estimate and its priors, subject to the <i>before / after</i> ordering constraints — and runs to
 convergence. Each dot is the posterior median; the 95% credible interval (shown on hover) widens
 where evidence is sparse and tightens around well-anchored texts.
 <div class="cav">
  Posterior estimates, not established facts — especially for inferred (non-anchored) works and texts of
  uncertain or composite authorship. Per-work summaries are AI-assisted research notes.
  Method &amp; code: <a href="https://github.com/dharmamitra/sanskrit-dating">github.com/dharmamitra/sanskrit-dating</a>.
 </div>
</footer>
</body></html>"""
open('sanskrit_chronology_interactive.html','w').write(HTML)
print("wrote sanskrit_chronology_interactive.html  (%.1f MB, %d texts, searchable)"%(os.path.getsize('sanskrit_chronology_interactive.html')/1e6,N))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, collections as C
rng=np.random.default_rng(0)
rows=[r.split('\t') for r in open('dated_gibbs_full.tsv').read().splitlines()[1:]]
def coll(w):
    p=w.split('_'); return p[1] if len(p)>1 else ''
# refined collection-code -> genre
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

# lane order (grouped) + colors by family
LANES=['Veda (Saṃhitā/Brāhmaṇa)','Upaniṣad','Vedic Sūtra & Dharmaśāstra',
       'Epic','Purāṇa','Tantra & Āgama',
       'Poetics & Dramaturgy','Kāvya & Drama','Story, Fable & Nīti',
       'Grammar & Lexicon','Philosophy (Brahmanical/Jain)','Medicine (Āyurveda)','Astronomy & Mathematics',
       'Buddhist Sūtra & Vinaya','Buddhist Philosophy','Buddhist Tantra & Hymn']
COL={'Veda (Saṃhitā/Brāhmaṇa)':'#D35400','Upaniṣad':'#E67E22','Vedic Sūtra & Dharmaśāstra':'#F0A93B',
     'Epic':'#8E44AD','Purāṇa':'#B7950B','Tantra & Āgama':'#C2185B',
     'Poetics & Dramaturgy':'#117A65','Kāvya & Drama':'#229954','Story, Fable & Nīti':'#52BE80',
     'Grammar & Lexicon':'#1A5276','Philosophy (Brahmanical/Jain)':'#2471A3','Medicine (Āyurveda)':'#5499C7','Astronomy & Mathematics':'#2E86C1',
     'Buddhist Sūtra & Vinaya':'#A93226','Buddhist Philosophy':'#CB4335','Buddhist Tantra & Hymn':'#E6796B'}
li={t:i for i,t in enumerate(LANES)}
pts=[]; alld=[]; cnt=C.Counter()
for r in rows:
    try: md,lo,hi=float(r[6]),float(r[7]),float(r[8])
    except: continue
    cat=CAT.get(coll(r[0]))
    if cat is None: continue
    cnt[cat]+=1; alld.append(md)
    a=float(np.interp(hi-lo,[100,900],[0.85,0.06]))
    nch=float(r[2])                       # n_chunks ≈ text length (each chunk ~100 segments)
    pts.append((md,li[cat],a,r[1]=='anchor',nch))

fig=plt.figure(figsize=(16,13))
gs=fig.add_gridspec(2,1,height_ratios=[1,9],hspace=0.04)
axT=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1],sharex=axT)
XMIN,XMAX=-1600,1900
eras=[(-1600,-500,'Vedic'),(-500,200,'Epic & Sūtra'),(200,650,'Classical'),(650,1200,'Early Medieval'),(1200,1900,'Late Medieval')]
ecol=['#FFF3E0','#F3E5F5','#E8F5E9','#E3F2FD','#FFF8E1']
for a in (axT,axB):
    for (x0,x1,_),c in zip(eras,ecol): a.axvspan(x0,x1,color=c,zorder=0)
axT.hist(alld,bins=np.arange(XMIN,XMAX+1,50),color='#34495E',alpha=.85)
axT.set_ylabel('texts /50yr',fontsize=9); axT.tick_params(labelbottom=False)
axT.set_title('Sanskrit Literature by Genre: Computed Dates of All '+str(len(pts))+' Texts',fontsize=18,weight='bold',pad=40)
axT.text(0.5,1.16,'each dot = one text at its posterior date · opacity = confidence (solid = tight 95% interval, faint = uncertain)',
         transform=axT.transAxes,ha='center',fontsize=10.5,color='#666')
for (x0,x1,lbl) in eras:
    axB.text((max(x0,XMIN)+min(x1,XMAX))/2,0.994,lbl,ha='center',va='top',fontsize=9.5,style='italic',color='#888',
             transform=axB.get_xaxis_transform(),zorder=5)
for md,l,a,anc,nch in pts:
    y=l+rng.uniform(-0.34,0.34)
    s=float(np.clip(6+15*np.sqrt(nch),8,260))           # dot AREA scales with text length
    axB.scatter(md,y,s=s,color=COL[LANES[l]],alpha=a,edgecolors=('black' if anc else 'none'),linewidths=.3,zorder=3)
for i in range(1,len(LANES)): axB.axhline(i-0.5,color='white',lw=1.4,zorder=1)
axB.set_yticks(range(len(LANES))); axB.set_yticklabels([f'{t}  (n={cnt[t]})' for t in LANES],fontsize=10)
for tick,ln in zip(axB.get_yticklabels(),LANES): tick.set_color(COL[ln])
axB.set_ylim(-0.6,len(LANES)-0.4); axB.invert_yaxis(); axB.set_xlim(XMIN,XMAX)
ticks=[-1500,-1000,-500,1,500,1000,1500]
axB.set_xticks(ticks); axB.set_xticklabels([f'{-t} BCE' if t<0 else ('1 CE' if t==1 else f'{t} CE') for t in ticks],fontsize=10)
axB.axvline(0,color='#888',lw=.8,ls='--',zorder=1); axB.set_xlabel('Year',fontsize=11)
for s in ('top','right'): axB.spines[s].set_visible(False)
# size legend (dot area = text length)
from matplotlib.lines import Line2D
sl=[(3,'~300 lines'),(30,'~3,000'),(120,'~12,000')]
hs=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#555',markeredgecolor='none',
           markersize=np.sqrt(6+15*np.sqrt(n))/1.1,label=lb) for n,lb in sl]
leg2=axB.legend(handles=hs,loc='lower left',fontsize=9,title='Text length',framealpha=.92,labelspacing=1.1,handletextpad=1.2,borderpad=1)
axB.add_artist(leg2)
REF=[('SA_GV01_rv_01_u','Ṛgveda'),('SA_GS24_panini_u','Pāṇini'),('SA_T04_nagmmk_u','Nāgārjuna'),
     ('SA_GK19_ragh','Kālidāsa'),('SA_GS41_aryabh_u','Āryabhaṭa'),('SA_MB_Tantraloka-1-14-HK','Abhinavagupta')]
post={r[0]:r for r in rows}
for w,nm in REF:
    if w in post:
        md=float(post[w][6]); l=li[CAT[coll(w)]]
        axB.annotate(nm,(md,l-0.44),fontsize=8,ha='center',va='bottom',color='#222',style='italic',zorder=6)
plt.savefig('sanskrit_chronology_all.png',dpi=140,bbox_inches='tight',facecolor='white')
print('wrote; genres:',len(LANES),'total',len(pts))

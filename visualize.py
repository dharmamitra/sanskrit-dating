import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

post={r.split('\t')[0]:r.split('\t') for r in open('dated_gibbs_full.tsv').read().splitlines()[1:]}
def D(w):
    r=post.get(w);
    return (float(r[6]),float(r[7]),float(r[8])) if r else None

# curated landmarks: (work_id, display, tradition)
LM=[
 ('SA_GV01_rv_01_u','Ṛgveda','Vedic'),
 ('SA_GV00_avs','Atharvaveda','Vedic'),
 ('SA_GV03_sb','Śatapatha-Brāhmaṇa','Vedic'),
 ('SA_GV05_brup___u','Bṛhadāraṇyaka-Upaniṣad','Vedic'),
 ('SA_GV05_chup___u','Chāndogya-Upaniṣad','Vedic'),
 ('SA_GV05_kathop_u','Kaṭha-Upaniṣad','Vedic'),
 ('SA_GS24_panini_u','Pāṇini · Aṣṭādhyāyī','Śāstra'),
 ('SA_GS38_kautil','Kauṭilya · Arthaśāstra','Śāstra'),
 ('SA_GSP34_yogasutu','Patañjali · Yogasūtra','Śāstra'),
 ('SA_GE07_mbh_06_u','Mahābhārata','Epic'),
 ('SA_GE07_bhgce__u','Bhagavadgītā','Epic'),
 ('SA_GK19_asvbc_1u','Aśvaghoṣa · Buddhacarita','Buddhist'),
 ('SA_T04_nagmmk_u','Nāgārjuna · Mūlamadhyamakakārikā','Buddhist'),
 ('SA_T06_srabhusu','Śrāvakabhūmi','Buddhist'),
 ('SA_T06_bsa034','Bodhisattvabhūmi','Buddhist'),
 ('SA_T07_vakobhau','Vasubandhu · Abhidharmakośa','Buddhist'),
 ('SA_GK19_kakumspu','Kālidāsa · Kumārasambhava','Kāvya'),
 ('SA_GK19_ragh','Kālidāsa · Raghuvaṃśa','Kāvya'),
 ('SA_GK20_ksakunpu','Kālidāsa · Abhijñānaśākuntala','Kāvya'),
 ('SA_GK19_amaru_u','Amaruśataka','Kāvya'),
 ('SA_GS41_aryabh_u','Āryabhaṭa · Āryabhaṭīya','Śāstra'),
 ('SA_GS41_brsphutu','Brahmagupta · Brāhmasphuṭasiddhānta','Śāstra'),
 ('SA_T04_n1003u','Bhāviveka · Madhyamakahṛdaya','Buddhist'),
 ('SA_T04_canprasu','Candrakīrti · Prasannapadā','Buddhist'),
 ('SA_T06_sthmavt','Sthiramati · Madhyāntavibhāgaṭīkā','Buddhist'),
 ('SA_GK16_andhvy_u','Ānandavardhana · Dhvanyāloka','Kāvya'),
 ('SA_T04_bsa003_u','Śāntideva · Bodhicaryāvatāra','Buddhist'),
 ('SA_GS40_vagaah_u','Vāgbhaṭa · Aṣṭāṅgahṛdaya','Śāstra'),
 ('SA_GP10_bhp_02u','Bhāgavata-Purāṇa','Purāṇa/Tantra'),
 ('SA_GP12_skp','Skandapurāṇa','Purāṇa/Tantra'),
 ('SA_GP12_agp_bi_u','Agni-Purāṇa','Purāṇa/Tantra'),
 ('SA_GSP29_nyamanu','Jayanta · Nyāyamañjarī','Śāstra'),
 ('SA_GK19_jaygit1u','Jayadeva · Gītagovinda','Kāvya'),
 ('SA_GK21_sokss','Somadeva · Kathāsaritsāgara','Kāvya'),
 ('SA_MB_Tantraloka-1-14-HK','Abhinavagupta · Tantrāloka','Purāṇa/Tantra'),
 ('SA_GSP30_pratyabu','Kṣemarāja · Pratyabhijñāhṛdaya','Purāṇa/Tantra'),
 ('SA_T04_bsa004_u','Atiśa · Bodhipathapradīpa','Buddhist'),
 ('SA_GK19_naicau','Śrīharṣa · Naiṣadhīyacarita','Kāvya'),
 ('SA_GSP33_pada','Vidyāraṇya · Pañcadaśī','Śāstra'),
 ('SA_GSP36_sadha','Mādhava · Sarvadarśanasaṃgraha','Śāstra'),
 ('SA_GSP34_hathyopu','Haṭhayogapradīpikā','Purāṇa/Tantra'),
 ('SA_GS40_bhavpr_u','Bhāvamiśra · Bhāvaprakāśa','Śāstra'),
]
COL={'Vedic':'#E8820C','Epic':'#8E44AD','Buddhist':'#C0392B','Kāvya':'#27AE60','Śāstra':'#2471A3','Purāṇa/Tantra':'#B7791F'}
items=[(D(w),nm,tr) for w,nm,tr in LM if D(w)]
items.sort(key=lambda x:x[0][0])

fig=plt.figure(figsize=(15,15))
gs=fig.add_gridspec(2,1,height_ratios=[1,7],hspace=0.04)
axT=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1],sharex=axT)
XMIN,XMAX=-1600,1800

# era bands
eras=[(-1600,-500,'Vedic',  '#FFF3E0'),(-500,200,'Epic & Sūtra','#F3E5F5'),
      (200,650,'Classical','#E8F5E9'),(650,1200,'Early Medieval','#E3F2FD'),(1200,1800,'Late Medieval','#FFF8E1')]
for a in (axT,axB):
    for x0,x1,_,c in eras: a.axvspan(x0,x1,color=c,zorder=0)
for x0,x1,lbl,_ in eras:
    axB.text(max(x0,XMIN)/2+min(x1,XMAX)/2,0.995,lbl,ha='center',va='top',fontsize=10.5,
             style='italic',color='#777',transform=axB.get_xaxis_transform(),zorder=4)

# top: corpus density (all reliable works = CrI width < 600)
dates=[];
for r in post.values():
    try:
        lo,hi=float(r[7]),float(r[8])
        if hi-lo<600: dates.append(float(r[6]))
    except: pass
axT.hist(dates,bins=np.arange(XMIN,XMAX+1,50),color='#34495E',alpha=.8)
axT.set_ylabel('texts\n(per 50 yr)',fontsize=9); axT.tick_params(labelbottom=False)
axT.set_title('A Computed Chronology of Sanskrit Literature',fontsize=18,weight='bold',pad=42)
axT.text(0.5,1.18,'landmark texts: dot = posterior date · bar = 95% credible interval   |   top: density of '+str(len(dates))+' confidently-dated works',
         transform=axT.transAxes,ha='center',fontsize=10.5,color='#666')

# bottom: landmark timeline
ys=range(len(items))
for y,((md,lo,hi),nm,tr) in zip(ys,items):
    c=COL[tr]
    axB.plot([lo,hi],[y,y],color=c,lw=2.2,alpha=.55,solid_capstyle='round',zorder=2)
    axB.plot(md,y,'o',color=c,ms=7,zorder=3,markeredgecolor='white',markeredgewidth=.8)
    axB.text(XMAX+30,y,nm,va='center',ha='left',fontsize=9.5,color='#222')
axB.set_yticks([]); axB.set_ylim(-1,len(items)); axB.invert_yaxis()
axB.set_xlim(XMIN,XMAX)
# x ticks as BCE/CE
ticks=[-1500,-1000,-500,1,500,1000,1500]
axB.set_xticks(ticks)
axB.set_xticklabels([f'{-t} BCE' if t<0 else ('1 CE' if t==1 else f'{t} CE') for t in ticks],fontsize=10)
axB.axvline(0,color='#999',lw=.8,ls='--',zorder=1)
axB.set_xlabel('Year',fontsize=11)
for s in ('top','right','left'): axB.spines[s].set_visible(False)
axB.grid(axis='x',color='white',lw=1,zorder=1)
leg=[Patch(facecolor=COL[k],label=k) for k in ['Vedic','Epic','Buddhist','Kāvya','Śāstra','Purāṇa/Tantra']]
axB.legend(handles=leg,loc='lower right',fontsize=9.5,framealpha=.9,title='Tradition')
plt.savefig('sanskrit_chronology.png',dpi=130,bbox_inches='tight',facecolor='white')
print('wrote sanskrit_chronology.png with',len(items),'landmarks')

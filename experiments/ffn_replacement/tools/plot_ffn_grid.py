# LEGACY plot: produces the 3-panel "FFN-LUT paper grid" overview. This is NOT the
# paper's Figure 2 -- that is the 2-panel figure committed at paper/FFN_GRID_plots.png,
# whose generator does not live in this repo. Kept for the grid overview only.
import json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as ml
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
def bpb(d): return json.load(open(f"{ROOT}/{d}/summary.json"))["final_val_bpb"]
VANILLA=bpb("exp_n_0135_untied_vanilla_baseline_16k")   # UNTIED, the new zero-line
TIED=1.19665                                            # tied vanilla, reference only
VLABEL="vanilla 4× MLP FFN (untied, 7.08M)"
# (id, dir, H, d, nap, tph, params_M, flop_M, vbw_M)
P=[
 ("0126","exp_n_0126_grid_H4d48_nap7_tph64",   4,48, 7, 64, 39.04,1.886,1.933),
 ("0127","exp_n_0127_grid_H4d48_nap7_tph128",  4,48, 7,128, 48.48,2.003,2.081),
 ("0128","exp_n_0128_grid_H4d48_nap8_tph64",   4,48, 8, 64, 48.48,1.892,1.933),
 ("0129","exp_n_0129_grid_H4d48_nap8_tph256",  4,48, 8,256,105.10,2.261,2.375),
 ("0130","exp_n_0130_grid_H4d48_nap10_tph64",  4,48,10, 64,105.10,1.905,1.933),
 ("0131","exp_n_0131_grid_H2d96_nap8_tph128",  2,96, 8,128, 67.35,1.966,2.081),
 ("0132","exp_n_0132_grid_H8d24_nap8_tph128",  8,24, 8,128, 67.35,2.114,2.081),
 ("0133","exp_n_0133_grid_H4d48_nap10_tph128", 4,48,10,128,180.60,2.040,2.081),
 ("0118","exp_n_0118_ffnsw_S2a_nap9_FULL16k",  4,48, 9,256,180.60,2.286,2.375),
 ("0119","exp_n_0119_ffnsw_S2a_nap9_tph128_16k",4,48,9,128,105.10,2.028,2.081),
 ("0121","exp_n_0121_ffnsw_nap8_tph128_16k",   4,48, 8,128, 67.35,2.015,2.081),
 ("0084","exp_n_0084_untied_nheads4",          4,48, 7,256, 67.35,2.236,2.375),
 # exp_n_0120 and exp_n_0125 rows removed: not cited by the paper, so their dirs are
 # not part of this folder and loading them would raise.
]
rows=[dict(id=i,H=H,d=dd,nap=nap,tph=tph,pm=pm,fl=fl,vb=vb,bpb=bpb(dr)) for i,dr,H,dd,nap,tph,pm,fl,vb in P]
byid={r["id"]:r for r in rows}
def cells(r): return 2**r["nap"]
def lab(r): return f"2^{r['nap']}×{r['tph']}t"
HCOL={2:'#7048e8',4:'#1c7ed6',8:'#e8590c'}; ANCHOR='#ffd43b'

# ---------- markdown table (untied vanilla = zero line; tied vanilla = reference row) ----------
tabrows=[dict(**r,kind=("NEW" if r["id"] in ("0126","0127","0128","0129","0130","0131","0132","0133") else "reuse")) for r in rows]
tabrows.append(dict(id="exp_n_0135",kind="vanilla (untied)",H="—",d="—",nap=None,tph="—",pm=35.79,fl=14.16,vb=14.16,bpb=VANILLA,ffn=7.08))
tabrows.append(dict(id="exp073",kind="reference (tied)",H="—",d="—",nap=None,tph="—",pm=23.21,fl=14.16,vb=14.16,bpb=TIED,ffn=7.08))
def ffn_of(r):
    return r.get("ffn", 6*192*r["tph"]*cells(r)/1e6 + 0.888) if r["nap"] is not None else r["ffn"]
tabrows.sort(key=lambda r:r["bpb"])
print(f"UNTIED vanilla zero-line (exp_n_0135) = {VANILLA:.5f}\n")
print("| id | kind | H | d | nap(cells) | tph | total params | FFN params | ×Van | FFN-FLOP | vBW | val_bpb | Δ vs untied vanilla |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in tabrows:
    napc = "dense 4×MLP" if r["nap"] is None else f"{r['nap']}({2**r['nap']})"
    xv = 1.00 if r["nap"] is None else round(ffn_of(r)/7.08,2)
    d0 = "**0 (zero-line)**" if r["id"]=="exp_n_0135" else f"{r['bpb']-VANILLA:+.5f}"
    bb = f"**{r['bpb']:.5f}**" if r["id"] in ("exp_n_0135","exp073") else f"{r['bpb']:.5f}"
    idc = f"**{r['id']}**" if r["id"] in ("exp_n_0135","exp073") else r["id"]
    print(f"| {idc} | {r['kind']} | {r['H']} | {r['d']} | {napc} | {r['tph']} | {r['pm']:.2f}M | {ffn_of(r):.1f}M | {xv}× | {r['fl']:.3f}M | {r['vb']:.3f}M | {bb} | {d0} |")
print(f"\ntied vanilla (exp073) = {TIED} = {TIED-VANILLA:+.5f} vs the untied zero-line (tied is better than untied for the vanilla FFN)")

# ---------- 3 plots ----------
def VL(ax,xtext):
    ax.axhline(VANILLA,ls='--',c='#c92a2a',lw=1.3,zorder=2)
    ax.text(xtext,VANILLA,' '+VLABEL,fontsize=7.1,c='#c92a2a',va='bottom',ha='right')
fig=plt.figure(figsize=(16.5,5.6))
# (a) params <-> bpb  (better=up via set_ylim(worse,better); no invert_yaxis)
axA=fig.add_subplot(1,3,1)
LEAD={"0120":(-42,-14),"0084":(16,9),"0128":(-40,-7),"0125":(16,7)}
OFF ={"0126":(7,-3),"0127":(8,1),"0131":(2,11),"0132":(12,-13),
      "0129":(6,5),"0119":(7,-9),"0130":(9,2),"0118":(9,4),"0133":(9,-11)}
for r in rows:
    if r["id"]=="0121":
        axA.scatter(r["pm"],r["bpb"],s=320,c=ANCHOR,marker='*',edgecolors='k',linewidths=1.1,zorder=6)
        axA.annotate("anchor · H4·d48 2^8×128t",(r["pm"],r["bpb"]),fontsize=6.4,fontweight='bold',
                     xytext=(13,-1),textcoords='offset points');continue
    axA.scatter(r["pm"],r["bpb"],s=58,c=HCOL[r["H"]],marker='o',edgecolors='k',linewidths=.4,zorder=3)
    if r["id"] in LEAD:
        dx,dy=LEAD[r["id"]]
        axA.annotate(lab(r),(r["pm"],r["bpb"]),fontsize=5.6,xytext=(dx,dy),textcoords='offset points',
                     arrowprops=dict(arrowstyle='-',lw=.45,color='#888',shrinkA=0,shrinkB=2))
    else:
        dx,dy=OFF.get(r["id"],(5,2));axA.annotate(lab(r),(r["pm"],r["bpb"]),fontsize=5.6,xytext=(dx,dy),textcoords='offset points')
best=9;fx=[];fy=[]
for r in sorted(rows,key=lambda r:r["pm"]):
    if r["bpb"]<best-1e-9: best=r["bpb"];fx.append(r["pm"]);fy.append(r["bpb"])
axA.plot(fx,fy,'-',color='#adb5bd',lw=1.2,zorder=1)
VL(axA,182)
axA.set_ylim(1.2085,1.1725);axA.set_xlabel('total params (M)');axA.set_ylabel('val_bpb ↓');axA.grid(alpha=.25)
axA.set_title('(a) params ↔ bpb  (label = cells×tables; color = heads)')
axA.legend(handles=[ml.Line2D([],[],marker='o',ls='',mfc=HCOL[2],mec='k',label='H2 · d96'),
                    ml.Line2D([],[],marker='o',ls='',mfc=HCOL[4],mec='k',label='H4 · d48'),
                    ml.Line2D([],[],marker='o',ls='',mfc=HCOL[8],mec='k',label='H8 · d24'),
                    ml.Line2D([],[],marker='*',ls='',mfc=ANCHOR,mec='k',ms=13,label='anchor')],
           fontsize=6.6,loc='upper right',title='H·d=192 (all)',title_fontsize=6.6)
# (b) iso-param diagonals
axB=fig.add_subplot(1,3,2)
for grp,col,fam in [(["0129","0119","0130"],'#f03e3e','105.1M · 10.8×'),(["0133","0118"],'#9c36b5','180.6M · 21.5×')]:
    xs=[byid[i]["vb"] for i in grp];ys=[byid[i]["bpb"] for i in grp]
    axB.plot(xs,ys,'o-',color=col,lw=1.6,ms=7,label=fam)
    for i in grp:
        r=byid[i];axB.annotate(f"H4·d48\n2^{r['nap']}×{r['tph']}t",(r['vb'],r['bpb']),fontsize=6.2,xytext=(4,3),textcoords='offset points')
VL(axB,2.37)
axB.set_ylim(1.205,1.172);axB.set_xlabel('vBW (M) — more tables → higher');axB.set_ylabel('val_bpb ↓');axB.grid(alpha=.25)
axB.legend(fontsize=7,title='iso-param family',title_fontsize=7);axB.set_title('(b) iso-param diagonals: more tables win')
# (c) head line
axC=fig.add_subplot(1,3,3)
head=[("0131",2),("0121",4),("0132",8)]
axC.plot([h for _,h in head],[byid[i]["bpb"] for i,_ in head],'-',color='#12874f',lw=1.8,zorder=2)
for i,h in head:
    r=byid[i];axC.scatter([h],[r["bpb"]],s=70,c=HCOL[h],marker='o',edgecolors='k',linewidths=.5,zorder=4)
    axC.annotate(f"H{h}·d{r['d']}\n2^{r['nap']}×{r['tph']}t\n{r['bpb']:.4f}",(h,r['bpb']),fontsize=6.6,xytext=(6,3),textcoords='offset points')
axC.scatter([4],[byid["0121"]["bpb"]],s=360,c=ANCHOR,marker='*',edgecolors='k',linewidths=1.1,zorder=5)
axC.annotate("anchor",(4,byid["0121"]["bpb"]),fontsize=7,fontweight='bold',xytext=(-2,-18),textcoords='offset points',ha='center')
VL(axC,8)
axC.set_ylim(1.2035,1.1876);axC.set_xscale('log',base=2);axC.set_xticks([2,4,8]);axC.set_xticklabels(['H2·d96','H4·d48','H8·d24'])
axC.set_xlabel('routing heads (H·d=192 fixed, 2^8 × 128 tables)');axC.set_ylabel('val_bpb ↓');axC.grid(alpha=.25)
axC.set_title('(c) head line @67.35M: fewer/wider heads win')
fig.suptitle('FFN-LUT paper grid — routed CompressionMHL FFN vs VANILLA 4× MLP (untied, 1.20144) — 16k, eff batch 24,576',fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95])
# Deliberately NOT named FFN_GRID_plots.png: that name belongs to the paper's Figure 2
# (paper/FFN_GRID_plots.png) and a second file by that name would shadow it.
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"ffn_grid_overview_3panel.png")
fig.savefig(out,dpi=125);print("saved",out)

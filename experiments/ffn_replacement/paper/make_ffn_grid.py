import os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as ml
BASE = os.path.dirname(os.path.abspath(__file__))
# committed final_val_bpb from research/hyperplane_ffn_next, hardcoded so the figure
# regenerates standalone (no dependency on a live checkout).
BPB={"0135":1.20144,"0126":1.20694,"0127":1.19471,"0128":1.20228,"0129":1.18148,
     "0130":1.19405,"0131":1.18883,"0132":1.19263,"0133":1.17961,"0118":1.17460,
     "0119":1.18386,"0120":1.19859,"0121":1.19146,"0084":1.19866,"0125":1.20332,
     "0137":1.19448,"0153":1.20772}
VANILLA=BPB["0135"]                                     # UNTIED, the zero-line
TIED=1.19665                                            # tied vanilla, reference only
VLABEL="vanilla 4× MLP FFN (untied, 7.08M)"
# (id, dir, H, d, nap, tph, params_M, flop_M, vbw_M)
P=[
 # flop_M column uses the conservative convention (matmul 2 FLOP/MAC + 1/anchor-compare + 1/gather-add)
 ("0126","exp_n_0126_grid_H4d48_nap7_tph64",   4,48, 7, 64, 39.04,1.854,1.933),
 ("0127","exp_n_0127_grid_H4d48_nap7_tph128",  4,48, 7,128, 48.48,1.938,2.081),
 ("0128","exp_n_0128_grid_H4d48_nap8_tph64",   4,48, 8, 64, 48.48,1.855,1.933),
 ("0129","exp_n_0129_grid_H4d48_nap8_tph256",  4,48, 8,256,105.10,2.114,2.375),
 ("0130","exp_n_0130_grid_H4d48_nap10_tph64",  4,48,10, 64,105.10,1.859,1.933),
 ("0131","exp_n_0131_grid_H2d96_nap8_tph128",  2,96, 8,128, 67.35,1.929,2.081),
 ("0132","exp_n_0132_grid_H8d24_nap8_tph128",  8,24, 8,128, 67.35,1.966,2.081),
 ("0133","exp_n_0133_grid_H4d48_nap10_tph128", 4,48,10,128,180.60,1.948,2.081),
 ("0118","exp_n_0118_ffnsw_S2a_nap9_FULL16k",  4,48, 9,256,180.60,2.120,2.375),
 ("0119","exp_n_0119_ffnsw_S2a_nap9_tph128_16k",4,48,9,128,105.10,1.945,2.081),
 ("0120","exp_n_0120_ffnsw_S2a_nap9_tph64_16k",4,48, 9, 64, 67.35,1.857,1.933),
 ("0121","exp_n_0121_ffnsw_nap8_tph128_16k",   4,48, 8,128, 67.35,1.942,2.081),
 ("0084","exp_n_0084_untied_nheads4",          4,48, 7,256, 67.35,2.107,2.375),
 ("0125","exp_n_0125_ffnsw_H8_in24out24_tph64_16k",8,24,8,64,48.48,1.868,1.933),
 ("0137","exp_n_0137_grid_H1d192_nap8_tph128", 1,192, 8,128, 67.35,1.923,2.070),
 ("0153","exp_n_0153_H2d96_nap7_tph64",        2,96, 7, 64, 39.04,1.849,1.933),
]
rows=[dict(id=i,H=H,d=dd,nap=nap,tph=tph,pm=pm,fl=fl,vb=vb,bpb=BPB[i]) for i,dr,H,dd,nap,tph,pm,fl,vb in P]
byid={r["id"]:r for r in rows}
def cells(r): return 2**r["nap"]
def lab(r): return ("H2·2^7×64t" if r['id']=="0153" else f"2^{r['nap']}×{r['tph']}t")
HCOL={1:'#2f9e44',2:'#7048e8',4:'#1c7ed6',8:'#e8590c'}; ANCHOR='#ffd43b'

# ---------- markdown table (untied vanilla = zero line; tied vanilla = reference row) ----------
tabrows=[dict(**r,kind=("NEW" if r["id"] in ("0126","0127","0128","0129","0130","0131","0132","0133","0137") else "reuse")) for r in rows]
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
fig=plt.figure(figsize=(11,5.2))
# (a) iso-param diagonals: bpb vs FFN virtual bandwidth
axB=fig.add_subplot(1,2,1)
for grp,col,fam in [(["0129","0119","0130"],'#f03e3e','105.1M · 10.8×'),(["0133","0118"],'#9c36b5','180.6M · 21.5×')]:
    xs=[byid[i]["vb"] for i in grp];ys=[byid[i]["bpb"] for i in grp]
    axB.plot(xs,ys,'o-',color=col,lw=1.6,ms=7,label=fam)
    for i in grp:
        r=byid[i];axB.annotate(f"H4·d48\n2^{r['nap']}×{r['tph']}t",(r['vb'],r['bpb']),fontsize=6.2,xytext=(4,3),textcoords='offset points')
VL(axB,2.37)
axB.set_ylim(1.172,1.205);axB.set_xlabel('vBW (M) — more tables → higher');axB.set_ylabel('val_bpb ↓');axB.grid(alpha=.25)
axB.legend(fontsize=7,title='iso-param family',title_fontsize=7);axB.set_title('(a) iso-param diagonals: more tables win')
# (b) head line
axC=fig.add_subplot(1,2,2)
head=[("0137",1),("0131",2),("0121",4),("0132",8)]
axC.plot([h for _,h in head],[byid[i]["bpb"] for i,_ in head],'-',color='#12874f',lw=1.8,zorder=2)
for i,h in head:
    r=byid[i];axC.scatter([h],[r["bpb"]],s=70,c=HCOL[h],marker='o',edgecolors='k',linewidths=.5,zorder=4)
    axC.annotate(f"H{h}·d{r['d']}\n2^{r['nap']}×{r['tph']}t\n{r['bpb']:.4f}",(h,r['bpb']),fontsize=6.6,xytext=(6,3),textcoords='offset points')
axC.scatter([4],[byid["0121"]["bpb"]],s=360,c=ANCHOR,marker='*',edgecolors='k',linewidths=1.1,zorder=5)
axC.annotate("anchor",(4,byid["0121"]["bpb"]),fontsize=7,fontweight='bold',xytext=(-2,-18),textcoords='offset points',ha='center')
VL(axC,8)
axC.set_ylim(1.1876,1.2035);axC.set_xscale('log',base=2);axC.set_xticks([1,2,4,8]);axC.set_xticklabels(['H1·d192','H2·d96','H4·d48','H8·d24'])
axC.set_xlabel('routing heads (H·d=192 fixed, 2^8 × 128 tables)');axC.set_ylabel('val_bpb ↓');axC.grid(alpha=.25)
axC.set_title('(b) head line @67.35M: fewer/wider heads win')
fig.suptitle('Input-reprojection sweep — FFN virtual bandwidth and head-line views (16k; vanilla 4× MLP zero-line 1.20144)',fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.94])
out=os.path.join(BASE, "FFN_GRID_plots.png");fig.savefig(out,dpi=125);print("saved",out)

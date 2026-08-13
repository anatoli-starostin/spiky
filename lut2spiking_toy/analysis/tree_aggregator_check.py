#!/usr/bin/env python3
# ============================================================================
# MODEL 4 verification — HIERARCHICAL / TREE aggregation of delay-coded tables.
#
# Model 3 dumps all N tables' output spikes onto ONE aggregator: fan-in grows
# linearly with N. Model 4 builds a binary aggregation TREE so every aggregator
# has fan-in EXACTLY 2 (bounded, independent of N) and total depth is O(log2 N).
#
# Every emitter (a table OR an aggregator) emits Dout latency spikes:
#       t_{c,j} = GZ_c + ALPHA + BETA * value_c[j]        (bigger value = later)
# An aggregator over its children uses the SAME sustained-current mechanism as
# model 3: each incoming spike switches on constant current I from its arrival
# until the aggregator's readout clock T = max(child spikes) + SETTLE, so
#       V_j(T) = I * sum_c (T - t_{c,j}) = I*(n*T - C0 - BETA*P_j)
# with P_j = sum_c value_c[j] the partial sum and C0 = sum_c (GZ_c + ALPHA).
# Reading V_j recovers P_j exactly; the aggregator then BECOMES an emitter with
# GZ = T and value = P, so the next level treats it as an ordinary table.
#
# 4 tables (seeds 0..3): L1a=agg(T0,T1), L1b=agg(T2,T3), L2=agg(L1a,L1b).
# Final S[j] = sum_t O_t[r_t][j], emitted at t_out = GZ_L2 + ALPHA + BETA*S[j].
# ============================================================================
import math, random

def mulberry32(seed):
    s=[seed & 0xffffffff]
    def rng():
        s[0]=(s[0]+0x6d2b79f5)&0xffffffff
        t=s[0]; t=(t ^ (t>>15)); t=(t*(t|1))&0xffffffff
        t^=(t+((t ^ (t>>7))*(t|61)&0xffffffff))&0xffffffff; t&=0xffffffff
        return ((t ^ (t>>14))&0xffffffff)/4294967296
    return rng
def make_normal(rng):
    spare=[None]
    def n():
        if spare[0] is not None:
            v=spare[0]; spare[0]=None; return v
        while True:
            u=2*rng()-1; v=2*rng()-1; s=u*u+v*v
            if 0<s<1: break
        m=math.sqrt(-2*math.log(s)/s); spare[0]=v*m; return u*m
    return n
def build(seed):
    rng=mulberry32(seed); N=make_normal(rng)
    D,K,Dout=6,3,4
    W=[[N() for _ in range(D)] for _ in range(K)]
    b=[N() for _ in range(K)]
    V=[[N() for _ in range(Dout)] for _ in range(1<<K)]
    return W,b,V
def bits(W,b,x): return [1 if (sum(W[k][i]*x[i] for i in range(len(x)))+b[k])>0 else 0 for k in range(len(W))]
def row(bt):
    r=0
    for bb in bt: r=(r<<1)|bb
    return r

ALPHA,BETA,SETTLE,MIN_TRAVEL,I_CUR = 5.0,1.0,0.6,0.6,1.0
Dout=4
MODELS=[build(s) for s in range(4)]

def table_emitter(m,x):
    W,b,V=m
    bt=bits(W,b,x); r=row(bt); O=V[r]
    fv=(6.5 if 0 in bt else 5.0)+MIN_TRAVEL
    GZ=fv+SETTLE
    tout=[GZ+ALPHA+BETA*O[j] for j in range(Dout)]
    return dict(GZ=GZ, val=list(O), tout=tout)

def aggregate(children):
    n=len(children)
    T=max(c['tout'][j] for c in children for j in range(Dout))+SETTLE   # readout clock
    C0=sum(c['GZ']+ALPHA for c in children)
    val=[]; V=[]
    for j in range(Dout):
        Vj=I_CUR*sum(T-c['tout'][j] for c in children)                 # sustained-current membrane
        V.append(Vj)
        Pj=(I_CUR*(n*T-C0)-Vj)/(I_CUR*BETA)                           # recover the partial sum
        val.append(Pj)
    tout=[T+ALPHA+BETA*val[j] for j in range(Dout)]
    return dict(GZ=T, val=val, tout=tout, V=V, T=T)

def run(x):
    tabs=[table_emitter(MODELS[s],x) for s in range(4)]
    L1a=aggregate([tabs[0],tabs[1]])
    L1b=aggregate([tabs[2],tabs[3]])
    L2 =aggregate([L1a,L1b])
    return tabs,L1a,L1b,L2

if __name__=='__main__':
    NRUN=50000
    rng=random.Random(20260730)
    e_l1=e_l2=e_dec=0.0
    match=0; total=0
    for _ in range(NRUN):
        x=[rng.uniform(-1,1) for _ in range(6)]
        tabs,L1a,L1b,L2=run(x)
        Pa_true=[tabs[0]['val'][j]+tabs[1]['val'][j] for j in range(Dout)]
        Pb_true=[tabs[2]['val'][j]+tabs[3]['val'][j] for j in range(Dout)]
        S_true =[sum(tabs[t]['val'][j] for t in range(4)) for j in range(Dout)]
        for j in range(Dout):
            e_l1=max(e_l1, abs(L1a['val'][j]-Pa_true[j]), abs(L1b['val'][j]-Pb_true[j]))
            e_l2=max(e_l2, abs(L2['val'][j]-S_true[j]))
            sdec=(L2['tout'][j]-L2['GZ']-ALPHA)/BETA                   # decode from FINAL emitted spike
            e_dec=max(e_dec, abs(sdec-S_true[j]))
            total+=1
            if abs(L1a['val'][j]-Pa_true[j])<1e-9 and abs(L1b['val'][j]-Pb_true[j])<1e-9 \
               and abs(L2['val'][j]-S_true[j])<1e-9 and abs(sdec-S_true[j])<1e-9:
                match+=1
    print("="*68)
    print("MODEL 4 — hierarchical TREE aggregation (4 tables, fan-in 2, depth 2)")
    print(f"  {NRUN} random inputs x in [-1,1]^6, tables seed0..3, Dout={Dout}")
    print("-"*68)
    print(f"  (a) L1 aggregator membranes  == true partial sums : max abs err {e_l1:.2e}")
    print(f"  (b) L2 aggregator membrane   == true total sum     : max abs err {e_l2:.2e}")
    print(f"  (c) FINAL emitted-spike decode == true total sum   : max abs err {e_dec:.2e}")
    print(f"  (d) match rate (all < 1e-9)  : {100.0*match/total:.4f}%  ({match}/{total})")
    ok = e_l1<1e-9 and e_l2<1e-9 and e_dec<1e-9 and match==total
    print(f"  VERDICT: {'PASS (fp exact)' if ok else 'FAIL'}")
    print("-"*68)
    print("  SCALING — binary tree over N tables (fan-in bounded at 2):")
    print(f"    {'N':>6} {'depth=log2 N':>14} {'max fan-in':>11} {'#aggregators':>13} {'agg-neurons':>12}")
    for N in [4,16,64,256,1024]:
        depth=int(math.ceil(math.log2(N)))
        aggs=N-1                      # internal nodes of a binary tree with N leaves
        print(f"    {N:>6} {depth:>14} {2:>11} {aggs:>13} {aggs*Dout:>12}")
    print("  Flat model 3 by contrast: depth 1 but ONE aggregator with fan-in = N (unbounded).")
    print("  Tree trades O(log N) depth for O(1) fan-in per neuron — the scale-to-hundreds story.")
    print("="*68)
    tabs,L1a,L1b,L2=run([0.0]*6)
    print(f"  x=0: partials Pa={[round(v,3) for v in L1a['val']]} Pb={[round(v,3) for v in L1b['val']]}")
    print(f"        total S={[round(v,3) for v in L2['val']]}  final t_out={[round(t,2) for t in L2['tout']]}  T2={L2['T']:.2f}")

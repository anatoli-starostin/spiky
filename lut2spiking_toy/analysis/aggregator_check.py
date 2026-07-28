#!/usr/bin/env python3
# ============================================================================
# MODEL 3 verification — "delay-coded tables + a listening aggregator population"
#
# Each table emits its output vector as latency-coded SPIKES (value in the
# conduction delay, bigger value = LATER):
#       t_{t,j} = GZ_t + ALPHA + BETA * O_t[r_t][j]
# A separate population of Dout aggregator neurons listens to ALL tables' output
# spikes. Each incoming spike switches on a SUSTAINED constant current I from its
# arrival until a fixed readout clock T_agg. The aggregator membrane then holds
#       V_j(T_agg) = I * sum_t (T_agg - t_{t,j})
#                  = I * (N*T_agg - sum_t t_{t,j})
#                  = I * (N*T_agg - C0 - BETA * S_j),   C0 = sum_t(GZ_t+ALPHA)
# with S_j = sum_t O_t[r_t][j] the TRUE sum. V_j is AFFINE in S_j, so reading it
# at T_agg recovers the sum exactly and re-emits it as a latency spike at
#       t_out_j = T_agg + ALPHA_A + BETA_A * S_j     (bigger sum = later).
#
# Contrast: a plain integrate-to-threshold aggregator fires on the k-th arrival —
# an ORDER STATISTIC of the incoming spike times, NOT their sum. This script
# verifies the sustained-current readout reproduces sum_t O_t[r_t] to fp, while
# the order-statistic does not.
# ============================================================================
import math, random
from statistics import mean

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

# ---- constants (mirror animate3.js) ----
ALPHA,BETA,SETTLE = 5.0,1.0,0.6          # per-table output latency code
MIN_TRAVEL = 0.6
I_CUR = 1.0                              # aggregator sustained current
SETTLE_A = 0.6                          # T_agg margin past last table spike
ALPHA_A,BETA_A = 5.0,1.0                # aggregator emission latency code
Dout,NT = 4,2

M=[build(0),build(1)]
def emit(x):
    """per-table winning rows, GZ_t, and the Dout output-spike times t_{t,j}."""
    tabs=[]
    for (W,b,V) in M:
        bt=bits(W,b,x); r=row(bt); O=V[r]
        fv=(6.5 if 0 in bt else 5.0)+MIN_TRAVEL
        GZ=fv+SETTLE
        tj=[GZ+ALPHA+BETA*O[j] for j in range(Dout)]
        tabs.append(dict(bt=bt,r=r,O=O,GZ=GZ,tout=tj))
    return tabs

def aggregate(x):
    tabs=emit(x)
    Strue=[sum(tabs[t]['O'][j] for t in range(NT)) for j in range(Dout)]
    allt=[tabs[t]['tout'][j] for t in range(NT) for j in range(Dout)]
    T_agg=max(allt)+SETTLE_A
    C0=sum(tabs[t]['GZ']+ALPHA for t in range(NT))
    Shat=[]; Vj=[]
    for j in range(Dout):
        V=I_CUR*sum(T_agg-tabs[t]['tout'][j] for t in range(NT))   # sustained-current membrane
        Vj.append(V)
        sh=(I_CUR*(NT*T_agg-C0)-V)/(I_CUR*BETA)                    # recover the sum from V_j
        Shat.append(sh)
    outT=[T_agg+ALPHA_A+BETA_A*Shat[j] for j in range(Dout)]
    Sdec=[(outT[j]-T_agg-ALPHA_A)/BETA_A for j in range(Dout)]     # decode from emitted spike time
    return Strue,Shat,Sdec,T_agg,outT,tabs

def order_stat(x):
    """naive integrate-to-threshold: fires on the 2nd (=N-th) arrival -> max time, NOT sum."""
    tabs=emit(x)
    return [max(tabs[t]['tout'][j] for t in range(NT)) for j in range(Dout)]

if __name__=='__main__':
    NRUN=50000
    rng=random.Random(20260729)
    e_mem=e_dec=0.0
    gap_order=0.0
    for _ in range(NRUN):
        x=[rng.uniform(-1,1) for _ in range(6)]
        Strue,Shat,Sdec,T_agg,outT,tabs=aggregate(x)
        for j in range(Dout):
            e_mem=max(e_mem,abs(Shat[j]-Strue[j]))
            e_dec=max(e_dec,abs(Sdec[j]-Strue[j]))
        os=order_stat(x)
        # order-statistic "decoded" as if it were the sum vs the true sum -> large gap
        for j in range(Dout):
            gap_order=max(gap_order,abs((os[j]-tabs[0]['GZ']-ALPHA)-Strue[j]))
    print("="*66)
    print("MODEL 3 — sustained-current aggregator over N=2 delay-coded tables")
    print(f"  {NRUN} random inputs x in [-1,1]^6, tables seed0+seed1, Dout={Dout}")
    print("-"*66)
    print(f"  aggregator membrane V_j readout  == true sum_t O_t[r_t][j] : max abs err {e_mem:.2e}")
    print(f"  emitted output-spike-time decode == true sum               : max abs err {e_dec:.2e}")
    ok = e_mem<1e-9 and e_dec<1e-9
    print(f"  match (both < 1e-9) : {'PASS (fp exact)' if ok else 'FAIL'}")
    print("-"*66)
    print(f"  CONTRAST — naive integrate-to-threshold (order statistic, k=N arrival):")
    print(f"    treating that time as the sum is off by up to {gap_order:.3f}  (order stat != sum)")
    print("="*66)
    for name,x in [("x=0",[0.0]*6)]:
        Strue,Shat,Sdec,T_agg,outT,tabs=aggregate(x)
        print(f"  {name}: S_true={[round(s,3) for s in Strue]}  T_agg={T_agg:.2f}")
        print(f"        emitted t_out={[round(t,2) for t in outT]}  decoded={[round(s,3) for s in Sdec]}")
    print("  VERDICT: sustained-current-until-T_agg reads the SUM exactly; order-statistic does not.")

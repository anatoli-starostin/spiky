import re, math, sys

W = "/home/astarostin/projects/lut_nanochat_wt/experiments/lut_nanochat/nanochat/wandb"
RUNS = {
    "v2":      f"{W}/run-20261014_210912-d24-lutffn-confn1-1xh100-s0-full-v2-9811cced/files/output.log",
    "aborted": f"{W}/run-20261016_022851-d24-lutffn-confn1-h32-tph32-1xh100-s0-3f8b4134/files/output.log",
    "live":    f"{W}/run-20261016_024335-d24-lutffn-confn1-h32-tph32-1xh100-s0-7f4cc747/files/output.log",
}
STEP_RE = re.compile(r"^step (\d+)/(\d+) .*?\|\s*loss:\s*([\d.]+).*?\|\s*lrm:\s*([\d.]+).*?\|\s*dt:\s*([\d.]+)ms")
VBPB_RE = re.compile(r"(?:Step\s+(\d+)\s*\|\s*)?Validation bpb:\s*([\d.]+)")

def parse(path):
    loss, lrm, dt, vbpb = {}, {}, {}, {}
    last_step = -1
    try:
        with open(path) as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    s = int(m.group(1)); last_step = s
                    loss[s] = float(m.group(3)); lrm[s] = float(m.group(4)); dt[s] = float(m.group(5))
                    continue
                v = VBPB_RE.search(line)
                if v:
                    s = int(v.group(1)) if v.group(1) is not None else last_step + 1
                    vbpb[s] = float(v.group(2))
    except FileNotFoundError:
        pass
    return loss, lrm, dt, vbpb

D = {k: parse(p) for k, p in RUNS.items()}
for k in RUNS:
    loss, lrm, dt, vbpb = D[k]
    print(f"[{k}] loss pts={len(loss)} (steps {min(loss) if loss else '-'}..{max(loss) if loss else '-'}) | vbpb pts={len(vbpb)}")

v2l, v2lrm, v2dt, v2b = D["v2"]
abl, ablrm, abdt, abb = D["aborted"]
lvl, lvlrm, lvdt, lvb = D["live"]

def f(x): return f"{x:.6f}"

print("\n=== ABORTED(5280) vs LIVE(5568) — same geometry+seed, differ only in horizon ===")
print("expectation: bit-identical until LR decay begins (~step 4224 for 5280); aborted died at step 18")
print(f"{'step':>5} {'aborted':>12} {'live':>12} {'|loss diff|':>12} {'ab_lrm':>8} {'lv_lrm':>8} {'lrm diff':>9}")
for s in sorted(set(abl) & set(lvl)):
    print(f"{s:>5} {f(abl[s]):>12} {f(lvl[s]):>12} {abs(abl[s]-lvl[s]):>12.2e} {ablrm[s]:>8.4f} {lvlrm[s]:>8.4f} {abs(ablrm[s]-lvlrm[s]):>9.2e}")

print("\n=== v2(r768) vs LIVE(r1536) — DIFFERENT geometry; the suspicious comparison ===")
common = sorted(set(v2l) & set(lvl))
print(f"overlap: {len(common)} steps ({min(common)}..{max(common)})")
cps = [0,1,2,3,4,5,6,8,10,15,20,30,50,75,100,150,200,250,300,400,500,600,750,900,1000,1100,1200,1250]
print(f"{'step':>5} {'v2_loss':>12} {'live_loss':>12} {'live-v2':>12} {'rel%':>8} {'v2_lrm':>8} {'lv_lrm':>8}")
for s in cps:
    if s in v2l and s in lvl:
        a, b = v2l[s], lvl[s]
        print(f"{s:>5} {f(a):>12} {f(b):>12} {b-a:>+12.6f} {(b-a)/a*100:>+8.3f} {v2lrm.get(s,float('nan')):>8.4f} {lvlrm.get(s,float('nan')):>8.4f}")

def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
g = {s: lvl[s]-v2l[s] for s in common}
buckets = [("0-3",0,3),("4-10",4,10),("11-50",11,50),("51-200",51,200),("201-500",201,500),
           ("501-900",501,900),("901-1250",901,1250)]
print("\nmean(live-v2) loss gap by step bucket:")
for name, lo, hi in buckets:
    xs = [g[s] for s in common if lo <= s <= hi]
    print(f"  {name:>9}: n={len(xs):>4}  mean gap = {mean(xs):+.5f}")

# identical-count check: how many overlap steps are bit-identical v2==live?
ident = sum(1 for s in common if v2l[s] == lvl[s])
print(f"\nbit-identical v2==live steps: {ident}/{len(common)}  (step0 both start at ln(32768)={math.log(32768):.6f})")

# aborted vs live identical count
cab = sorted(set(abl) & set(lvl))
identab = sum(1 for s in cab if abl[s] == lvl[s])
print(f"bit-identical aborted==live steps: {identab}/{len(cab)}")

print("\n=== val bpb: v2 vs live (matched steps) ===")
print(f"{'step':>5} {'v2_bpb':>12} {'live_bpb':>12} {'live-v2':>12}")
for s in sorted(set(v2b) & set(lvb)):
    print(f"{s:>5} {f(v2b[s]):>12} {f(lvb[s]):>12} {lvb[s]-v2b[s]:>+12.6f}")
print("live bpb all:", {s: round(v,4) for s,v in sorted(lvb.items())})
print("v2 bpb (first few):", {s: round(v,4) for s,v in sorted(v2b.items())[:8]})

# mean dt (s/step) corroboration
def mdt(dt, lo, hi):
    xs=[dt[s]/1000 for s in dt if lo<=s<=hi]
    return mean(xs)
print(f"\n=== cost corroboration (mean dt, s/step, steps 50-1200 to skip warmup/eval spikes) ===")
print(f"  v2   : {mdt(v2dt,50,1200):.3f} s/step")
print(f"  live : {mdt(lvdt,50,1200):.3f} s/step")
print("PARSE_DONE")

"""spnet-NATIVE encoding of the seed-0 toy LUT + a genuine SPIKING output stage.

Logic layers (verified 64/64, leak-free neurons, impulse/coincidence):
INPUT -> DETECT (per-edge-weighted sum + clock-gated readout Θ=50) -> COMPL (NOT
det, late clock) -> ROWS (coincidence AND, thr 2.5) -> OUTPUT membrane = LUT[r][d].

Spiking output stage: after the value LUT[r][d] sits on the (leak-free) output
membrane, a CLOCK COMB (START -> o_d at increasing delays, small weight delta each
tick) ramps every output up until it crosses threshold. Since v_d(t) = value_d +
delta*(ticks since comb start), the crossing time is t_out_d = base + (Θo -
value_d)/delta — LARGER value fires EARLIER, so the 4 output spikes come out in
descending value order. We read both the exact membrane value (before the comb)
and the emitted spike time, and check the spike ORDER matches the value ranking.

Build: uses spnet_harness.build_packed (weight palette). The clean single-shared-
meta per-edge-weights path (#78) currently hits a SEPARATE native _grow_explicit
segfault (few metas + this topology; distinct from the buffer bug PR #79 fixed) —
documented in the report; the palette gives the same map to <1e-4.
"""
import torch
import spiky_cuda  # noqa
from lutmodel import build_model, lut_spec
from spnet_harness import build_packed, leakfree_meta
from spiky.spnet.spnet import NeuronDataType

m = build_model(0)
W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]
sumW = [sum(W[k]) for k in range(K)]
det_theta = [(sumW[k] - b[k]) / 2.0 for k in range(K)]
COMP_THR, ROW_THR, LOW_THR, THETA_BIG = 5.0, 2.5, 0.5, 50.0
CLK_W, INH_W, CLK_C_DELAY = 10.0, -20.0, 8
# spiking output stage
OUT_THR = 6.0             # > max |value|, so the value alone never fires the output
COMB_DELTA = 0.5          # comb ramp increment per tick
COMB_BASE = 16            # comb first arrives at START_tick(1)+COMB_BASE
COMB_M = 26               # number of comb impulses (ticks of ramp)
READ_VALUE_TICK = 15      # read the exact membrane value here (row deposited, comb not yet)
N_TICKS = 46

nodes = ["START"] + ["x%d" % i for i in range(D)] + ["H%d" % k for k in range(K)] \
        + ["C%d" % k for k in range(K)] + ["r%d" % a for a in range(1 << K)] \
        + ["o%d" % j for j in range(Dout)]
local = {n: i for i, n in enumerate(nodes)}
thr = {"START": LOW_THR}
for i in range(D):
    thr["x%d" % i] = LOW_THR
for k in range(K):
    thr["H%d" % k] = THETA_BIG
for k in range(K):
    thr["C%d" % k] = COMP_THR
for a in range(1 << K):
    thr["r%d" % a] = ROW_THR
for j in range(Dout):
    thr["o%d" % j] = OUT_THR
uthr = sorted(set(thr.values()))
neuron_metas = [leakfree_meta(t) for t in uthr]
node_meta = [uthr.index(thr[n]) for n in nodes]

syn = []
for k in range(K):
    for i in range(D):
        syn.append((local["x%d" % i], local["H%d" % k], W[k][i], 1))
    syn.append((local["START"], local["H%d" % k], THETA_BIG - det_theta[k], 1))
    syn.append((local["START"], local["C%d" % k], CLK_W, CLK_C_DELAY))
    syn.append((local["H%d" % k], local["C%d" % k], INH_W, 1))
for a in range(1 << K):
    for k in range(K):
        bit = (a >> (K - 1 - k)) & 1
        syn.append((local[("H%d" % k) if bit else ("C%d" % k)], local["r%d" % a], 1.0, 1))
    for j in range(Dout):
        syn.append((local["r%d" % a], local["o%d" % j], V[a][j], 1))
# clock comb -> spiking output latency stage
for j in range(Dout):
    for i in range(COMB_M):
        syn.append((local["START"], local["o%d" % j], COMB_DELTA, COMB_BASE + i))

packed = build_packed([{"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn}], neuron_metas)
spnet, gid = packed["spnet"], packed["gid"]
print("built: %d synapses, %d syn-metas" % (packed["n_synapses"], packed["n_syn_metas"]), flush=True)

Hids = torch.tensor([gid(0, local["H%d" % k]) for k in range(K)], dtype=torch.int32)
Rids = torch.tensor([gid(0, local["r%d" % a]) for a in range(1 << K)], dtype=torch.int32)
Oids = torch.tensor([gid(0, local["o%d" % j]) for j in range(Dout)], dtype=torch.int32)

spec = lut_spec(m)
row_ok = out_ok = order_ok = 0
errs = []
sample = None
for si, s in enumerate(spec):
    x = s["x"]
    ev = [gid(0, local["START"])] + [gid(0, local["x%d" % i]) for i in range(D) if x[i] > 0]
    S = torch.zeros((1, N_TICKS, len(ev)), dtype=torch.int32)
    Vv = torch.zeros((1, N_TICKS, len(ev)), dtype=torch.float32)
    for j, nid in enumerate(ev):
        S[0, 1, j] = nid; Vv[0, 1, j] = 50.0
    spnet.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                        input_values=Vv, do_train=False, sparse_input=S,
                        do_record_voltage=True, do_reset_context=True)
    rr = spnet.export_neuron_data(Rids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(1 << K, N_TICKS)
    ovlt = spnet.export_neuron_data(Oids, 1, NeuronDataType.Voltage, 0, N_TICKS - 1).view(Dout, N_TICKS)
    ospk = spnet.export_neuron_data(Oids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(Dout, N_TICKS)
    rows = [a for a in range(1 << K) if (rr[a] > 0.5).any().item()]
    row = rows[0] if len(rows) == 1 else -1
    val = [float(ovlt[j, READ_VALUE_TICK].item()) for j in range(Dout)]     # exact membrane value
    t_out = []
    for j in range(Dout):
        nz = torch.nonzero(ospk[j] > 0.5).flatten()
        t_out.append(int(nz[0].item()) if nz.numel() else None)
    if row == s["row"]:
        row_ok += 1
    for j in range(Dout):
        errs.append(abs(val[j] - s["out"][j]))
    if row == s["row"] and all(abs(val[j] - s["out"][j]) < 1e-4 for j in range(Dout)):
        out_ok += 1
    # order check: sort dims by spike time ascending vs by true value descending
    if all(t is not None for t in t_out):
        by_spike = sorted(range(Dout), key=lambda j: (t_out[j], -s["out"][j]))
        by_value = sorted(range(Dout), key=lambda j: -s["out"][j])
        # allow ties in spike time to match either order of tied values
        ok = True
        for a2 in range(Dout):
            for b2 in range(a2 + 1, Dout):
                ja, jb = by_value[a2], by_value[b2]      # ja has higher value than jb
                if t_out[ja] > t_out[jb]:                # higher value must not fire later
                    ok = False
        if ok:
            order_ok += 1
        if si == 5:
            sample = (x, s["out"], val, t_out, by_spike, by_value)

n = len(spec)
print("=" * 64)
print("SPNET-NATIVE ENCODING + SPIKING OUTPUT — real spnet CPU, 64 inputs")
print("-" * 64)
print("  row-selection accuracy : %d/%d = %.1f%%" % (row_ok, n, 100 * row_ok / n))
print("  output VALUE match (membrane, tol 1e-4): %d/%d = %.1f%%" % (out_ok, n, 100 * out_ok / n))
print("  output value error: mean=%.3e max=%.3e" % (sum(errs) / len(errs), max(errs)))
print("  SPIKE-ORDER matches value ranking: %d/%d = %.1f%%" % (order_ok, n, 100 * order_ok / n))
if sample:
    x, tv, mv, to, bs, bv = sample
    print("-" * 64)
    print("  sample input x=%s" % x)
    print("    true LUT out : %s" % [round(v, 3) for v in tv])
    print("    membrane val : %s" % [round(v, 3) for v in mv])
    print("    output spike ticks (o0..o3): %s" % to)
    print("    order by spike (earliest first): %s ; by value (largest first): %s" % (bs, bv))

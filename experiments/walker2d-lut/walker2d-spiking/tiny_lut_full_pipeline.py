"""exp012: the complete three-stage handcrafted spiking LUT, end to end on the real engine.

  STAGE 1  latency inputs -> 136 dual-rail WTA comparators -> 192 order bits
           + TIE-BREAK: a per-pair detector fed by BOTH rails (threshold needs 2) vetoes the
             GT memory cell, so a tie emits only the LT rail = the LUT's strict-'>' tie->bit-0
  STAGE 2  2048 cell neurons, 6-way coincidence -> one-hot cell per table
           + value emission: fixed per-(cell,dim) delay encoding weights[t,k,o]
  STAGE 3  6 anti-leaky output neurons, arrival-time logsumexp -> affine decode

Timing. Rails fire at decision time (~tick 50-135), the gate at 200, memory emits at 203.
The tie-detector therefore has ~150 ticks of slack: it cancels the GT memory cell's stored
w_mem long before the gate arrives, and because both the stored charge and the veto decay at
the same rate the cancellation holds.
"""
import argparse
import json
import os

import numpy as np
import torch

from tiny_lut_order_detect import encode
from tiny_lut_order_full import pair_list

TAU, TPH = 0.09036568, 32
T_IN = 128
DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M_RAIL, TAU_MEM, W_MEM, W_GATE = 20.0, 1200.0, 0.6, 0.6
W_TIE, TH_TIE = 0.6, 1.0
W_AND = 0.18
TAU_M_OUT = 10.0
GATE_TICK, EMIT = 200, 203


def calib(tau_m, n_euler=2, dt=0.5):
    return 1.0 / np.log((1.0 + dt / tau_m) ** n_euler)


def build(Z, dims, tie_break, device="cuda", settle=None):
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    pairs, slot = pair_list(Z)
    P = len(pairs)
    W = Z["weights"].astype(np.float64)
    tau_eff = calib(TAU_M_OUT)
    scale = tau_eff / TAU
    dly, amps, c0s = {}, {}, {}
    for o in dims:
        Wd = W[:, :, o]
        c0 = float(np.ceil(scale * Wd.max() + 2))
        arr = np.rint(-scale * Wd + c0)
        dly[o] = arr.astype(np.int64)
        amps[o] = 1.0 / (2.0 * np.exp((arr.max() - arr) / tau_eff).sum())
        c0s[o] = c0
    # The engine caps a synapse delay at 255 (spnet.py:88). EMIT is *when* the memory cell
    # spikes, not a delay -- adding it here asked for a 298-tick delay and tripped that assert.
    dmax = int(max(int(d.max()) for d in dly.values()))
    assert dmax <= 255, f"delay {dmax} exceeds the engine's 255-tick synapse limit"
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=64, _backward_group_size=64)
              for d in range(1, dmax + 1)]
    NT = len(dims)
    metas = [LIFNeuronMeta(neuron_type=0, tau=TAU_M_RAIL, threshold=1.0),           # 17 in + gate
             NeuronMeta(neuron_type=1, cf_2=0.0, cf_1=+1.0 / TAU_M_RAIL, cf_0=0.0, a=0.0,
                        b=0.0, c=0.0, d=0.0, spike_threshold=1.0),                  # 2P rails
             LIFNeuronMeta(neuron_type=2, tau=TAU_M_RAIL, threshold=1.0),           # 2P interneur
             LIFNeuronMeta(neuron_type=3, tau=TAU_MEM, threshold=1.0),              # 2P memory
             LIFNeuronMeta(neuron_type=4, tau=TAU_M_RAIL, threshold=TH_TIE),        # P tie dets
             LIFNeuronMeta(neuron_type=5, tau=200.0, threshold=1.0),                # 2048 cells
             NeuronMeta(neuron_type=6, cf_2=0.0, cf_1=+1.0 / TAU_M_OUT, cf_0=0.0, a=0.0,
                        b=0.0, c=0.0, d=0.0, spike_threshold=1.0)]                  # outputs
    counts = [18, 2 * P, 2 * P, 2 * P, P, 2048, NT]
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=counts,
                     initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(7)]
    inp, gate = ids[0][:17], ids[0][17]
    E = []
    for p, (a, b) in enumerate(pairs):
        r0, r1 = ids[1][2 * p], ids[1][2 * p + 1]
        i0, i1 = ids[2][2 * p], ids[2][2 * p + 1]
        m0, m1 = ids[3][2 * p], ids[3][2 * p + 1]
        E += [(DE, inp[a], r0, W_EXC), (DE, inp[b], r1, W_EXC),
              (DI, inp[a], i0, W_EXC), (DI, inp[b], i1, W_EXC),
              (1, i0, r1, W_INH), (1, i1, r0, W_INH),
              (1, r0, m0, W_MEM), (1, r1, m1, W_MEM),
              (1, gate, m0, W_GATE), (1, gate, m1, W_GATE)]
        if tie_break:
            td = ids[4][p]
            # fires only when BOTH rails spike (0.6+0.6 >= 1.0; one alone is 0.6)
            E += [(1, r0, td, W_TIE), (1, r1, td, W_TIE), (1, td, m0, -W_MEM)]
    # Stage 2: cells, and Stage 3 emission
    for t in range(32):
        for k in range(64):
            cell = ids[5][t * 64 + k]
            for j in range(6):
                bit = (k >> (5 - j)) & 1
                p, a_first = slot[t * 6 + j]
                # rail meaning bit==1 is "anchor_a earlier"; pairs are stored (min,max)
                r1i = 2 * p + (0 if a_first else 1)
                E.append((1, ids[3][r1i if bit else (2 * p + (1 if a_first else 0))],
                          cell, W_AND))
            for oi, o in enumerate(dims):
                E.append((int(dly[o][t, k]), cell, ids[6][oi], amps[o]))
    tri = np.array([[d - 1, s, tg] for d, s, tg, _ in E], np.int64)
    wts = np.array([w for *_, w in E], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=64,
                             max_groups_in_buffer=max(1 << 19, 4 * len(tri)))
    for i in range(7):
        ge.register_neuron_type(max_synapses=1 << 15, growth_command_list=[])
    for i in range(7):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(tt.numel()).float(),
                                                torch.zeros(tt.numel()),
                                                torch.full((tt.numel(),), float(i))], 1))
    chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32, device=device), 1,
                              weights=torch.tensor(wts, dtype=torch.float32, device=device))
    net.add_connections(chunk, 1)
    chunk.recycle()
    net.compile(shuffle_synapses_random_seed=None)
    st = (12 * TAU_M_OUT + 30) if settle is None else settle
    n_ticks = int(EMIT + max(int(d.max()) for d in dly.values()) + st)
    return net, ids, len(tri), n_ticks, sum(counts)


def run(net, ids, ticks, n_ticks, dims, device="cuda"):
    from spiky.spnet.spnet import NeuronDataType
    B = ticks.shape[0]
    va = np.zeros((B, n_ticks, 18), np.float32)
    for j in range(17):
        va[np.arange(B), ticks[:, j], j] = 1e6
    va[:, GATE_TICK, 17] = 1e6
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    out = {}
    for k in (3, 5, 6):
        oid = torch.as_tensor(np.ascontiguousarray(ids[k], dtype=np.int32), device=device)
        R = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
        R = R.reshape(B, len(ids[k]), n_ticks).ne(0)
        if k == 6:
            w = torch.arange(n_ticks, 0, -1, device=R.device, dtype=torch.float32)
            out[k] = (n_ticks - (R.float() * w).amax(-1)).cpu().numpy().astype(np.int64)
        else:
            out[k] = R.sum(-1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dims", default="0,1,2,3,4,5")
    ap.add_argument("--no-tie-break", action="store_true")
    ap.add_argument("--gate", type=int, default=None)
    ap.add_argument("--settle", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dims = [int(v) for v in a.dims.split(",")]
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    W = Z["weights"].astype(np.float64)
    ntr = len(x) - 4000
    xs = x[ntr:ntr + a.n]
    ticks = encode(xs, T_IN)
    teacher = Z["y_action_mean_f64"][ntr:ntr + a.n]
    bits_q = (ticks[:, A_] < ticks[:, B_])                # quantised-input LUT bits, tie->0
    code_q = (bits_q * (2 ** np.arange(5, -1, -1))).sum(-1)
    ws_q = W.reshape(32 * 64, 6)[code_q + (np.arange(32) * 64)[None, :]]
    lutq = TPH * TAU * (np.log(np.exp(np.clip(ws_q / TAU, -60, 60)).sum(1)) - np.log(TPH))
    pairs, slot = pair_list(Z)
    global GATE_TICK, EMIT
    if a.gate is not None:
        EMIT = EMIT - GATE_TICK + a.gate
        GATE_TICK = a.gate
    net, ids, nsyn, n_ticks, nneur = build(Z, dims, not a.no_tie_break,
                                           settle=a.settle)
    print(f"composed: {nneur} neurons, {nsyn} synapses, episode {n_ticks} ticks, "
          f"tie-break {'OFF' if a.no_tie_break else 'ON'}")
    mem_margin = W_MEM * np.exp(-(GATE_TICK - (ticks.min() + DE + 2)) / TAU_MEM) + W_GATE - 1
    print(f"memory margin {mem_margin:+.4f} (tau_mem {TAU_MEM:.0f})\n")

    M, C, T = [], [], []
    for i in range(0, a.n, a.chunk):
        r = run(net, ids, ticks[i:i + a.chunk], n_ticks, dims)
        M.append(r[3]); C.append(r[5]); T.append(r[6])
    M, C, T = np.concatenate(M), np.concatenate(C), np.concatenate(T)
    del net
    torch.cuda.empty_cache()

    # bit-level: exactly one rail per slot?
    one = np.zeros((len(xs), 192), bool); got = np.zeros((len(xs), 192), bool)
    for s, (p, a_first) in enumerate(slot):
        r1 = 2 * p + (0 if a_first else 1); r0 = 2 * p + (1 if a_first else 0)
        f1, f0 = M[:, r1] > 0, M[:, r0] > 0
        one[:, s] = f1 ^ f0
        got[:, s] = f1
    tflat = bits_q.reshape(len(xs), 192)
    # COUNTS, not just rates -- at 4000 x 192 a single bad comparison is 1.3e-6 and would
    # round to 100.0000% in a rate. Rare failures are the whole point of the larger sample.
    n_cmp = one.size
    bad_rail = int((~one).sum())
    bad_bit = int((got != tflat).sum())
    print(f"STAGE 1  exactly-one-rail {100 * one.mean():.4f}%  "
          f"({bad_rail} bad of {n_cmp} comparisons)   "
          f"bit == quantised-LUT bit {100 * (got == tflat).mean():.4f}%  ({bad_bit} wrong)")
    if bad_rail:
        s_idx = np.unique(np.where(~one)[1])
        print(f"   slots with a bad rail count: {s_idx[:12].tolist()}"
              f"{' ...' if len(s_idx) > 12 else ''}  ({len(s_idx)} distinct slots)")
    pt = C.reshape(-1, 32, 64)
    nf = (pt > 0).sum(-1)
    n_tab = nf.size
    n_none, n_multi = int((nf == 0).sum()), int((nf > 1).sum())
    wrong_cell = int(((pt.argmax(-1) != code_q) & (nf == 1)).sum())
    print(f"STAGE 2  one-hot per table {100 * float((nf == 1).mean()):.4f}%  "
          f"({n_none} none, {n_multi} multi, of {n_tab} table-instances)"
          f"  cell==quantised code {100 * float((pt.argmax(-1) == code_q)[nf == 1].mean()):.4f}%"
          f"  ({wrong_cell} wrong)")
    if n_none or n_multi or wrong_cell:
        bt = np.unique(np.where((nf != 1) | ((pt.argmax(-1) != code_q) & (nf == 1)))[1])
        print(f"   affected tables: {bt.tolist()}")
    print(f"\nSTAGE 3  per-dim decode (affine fitted on the first half, applied to the second):")
    print(f"  dim    R2 vs quantised-LUT    R2 vs TRUE teacher    MSE(q)    max|err|(q)")
    res = {}
    for oi, o in enumerate(dims):
        t = T[:, oi].astype(float)
        f = t < n_ticks
        h = f & (np.arange(len(t)) < len(t) // 2)
        e = f & ~(np.arange(len(t)) < len(t) // 2)
        cf = np.polyfit(t[h], lutq[h, o], 1)
        p = cf[0] * t + cf[1]
        mq = float(((p[e] - lutq[e, o]) ** 2).mean()); vq = float(lutq[e, o].var())
        mt = float(((p[e] - teacher[e, o]) ** 2).mean()); vt = float(teacher[e, o].var())
        sc = calib(TAU_M_OUT) / TAU
        la = EMIT + int(np.rint(-sc * W[:, :, o]
                                + np.ceil(sc * W[:, :, o].max() + 2)).max())
        res[o] = dict(r2_lutq=1 - mq / vq, r2_teacher=1 - mt / vt, mse_lutq=mq,
                      max_err=float(np.abs(p[e] - lutq[e, o]).max()),
                      frac_fired=float(f.mean()), max_out_tick=int(t[f].max()),
                      last_arrival=la, measured_settle=int(t[f].max()) - la)
        print(f"   {o}       {1 - mq / vq:8.6f}            {1 - mt / vt:8.6f}      "
              f"{mq:.6f}   {res[o]['max_err']:.4f}   out<={res[o]['max_out_tick']}"
              f"  MEASURED-settle {res[o]['measured_settle']}")
    if a.out:
        json.dump(dict(neurons=nneur, synapses=nsyn, n_ticks=n_ticks, n=a.n,
                       tie_break=not a.no_tie_break, mem_margin=float(mem_margin),
                       stage1_one_rail=float(one.mean()),
                       stage1_bit_acc=float((got == tflat).mean()),
                       stage2_onehot=float((nf == 1).mean()),
                       per_dim={str(k): v for k, v in res.items()}),
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

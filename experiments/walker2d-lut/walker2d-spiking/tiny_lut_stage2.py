"""exp012 Stage 2 (lookup) + composition with Stage 3 (output), on the real engine.

Stage 2 is two mechanisms:

  COINCIDENCE  one neuron per LUT cell (32 tables x 64 cells = 2048). Cell (t,k) receives from
               the 6 rails that spell out code k -- rail-GT of pair (t,j) when bit j of k is 1,
               rail-LT when it is 0. All Stage 1 rails emit at ONE fixed tick, so the six
               arrive together and the AND is an exact 6-way coincidence: with threshold 1.0
               and w_and = 0.18, six inputs give 1.08 (fires) and five give 0.90 (does not).

  VALUE EMISSION  the winning cell projects to each of the 6 output neurons with a FIXED
               per-(cell, dim) conduction delay encoding weights[t,k,o] on Stage 3's timing
               scale -- "which cell, all at one tick" becomes "one spike whose time is the
               stored value", which is exactly the contract Stage 3 needs.

Stage 3's anti-leaky output neuron then reads the 32 arrivals as a logsumexp.

Verified in ISOLATION: rails are driven from GROUND-TRUTH bits rather than from Stage 1, the
same way Stage 3 was verified from true w_sel. That keeps this measurement about Stage 2.
"""
import argparse
import json
import os

import numpy as np
import torch

TAU, TPH = 0.09036568, 32
TAU_M = 10.0
W_AND, TH = 0.18, 1.0
T_RAIL = 4                      # tick at which every rail fires (Stage 1 emits at one tick)


def calib(tau_m, n_euler=2, dt=0.5):
    per = (1.0 + dt / tau_m) ** n_euler
    return 1.0 / np.log(per), per


def build(Z, dim, scale, c0, amp, tau_m, device="cuda"):
    """384 rail drivers -> 2048 cell neurons -> 1 anti-leaky output for `dim`."""
    from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    W = Z["weights"].astype(np.float64)[:, :, dim]           # [32,64]
    dly = np.rint(-scale * W + c0).astype(np.int64)          # bigger value -> EARLIER
    dmax = int(max(dly.max(), 2))
    smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                          min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                          weight_decay=0.9, weight_scaling_cf=0.0,
                          _forward_group_size=64, _backward_group_size=64)
              for d in range(1, dmax + 1)]
    metas = [LIFNeuronMeta(neuron_type=0, tau=tau_m, threshold=1.0),          # 384 rails
             LIFNeuronMeta(neuron_type=1, tau=200.0, threshold=TH),           # 2048 cells
             NeuronMeta(neuron_type=2, cf_2=0.0, cf_1=+1.0 / tau_m, cf_0=0.0, a=0.0, b=0.0,
                        c=0.0, d=0.0, spike_threshold=1.0)]                   # output
    net = SpikingNet(synapse_metas=smetas, neuron_metas=metas,
                     neuron_counts=[384, 2048, 1], initial_synapse_capacity=1 << 23,
                     summation_dtype=torch.float32)
    net.to_device(device)
    ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(3)]
    edges = []
    for t in range(32):
        for k in range(64):
            cell = ids[1][t * 64 + k]
            for j in range(6):
                bit = (k >> (5 - j)) & 1
                rail = ids[0][(t * 6 + j) * 2 + (0 if bit else 1)]   # 0 = GT rail, 1 = LT
                edges.append((1, rail, cell, W_AND))
            edges.append((int(dly[t, k]), cell, ids[2][0], amp))
    tri = np.array([[d - 1, s, tg] for d, s, tg, _ in edges], np.int64)
    wts = np.array([w for *_, w in edges], np.float64)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=64,
                             max_groups_in_buffer=max(1 << 18, 4 * len(tri)))
    for i in range(3):
        ge.register_neuron_type(max_synapses=1 << 14, growth_command_list=[])
    for i in range(3):
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
    return net, ids, len(tri), dly


def run(net, ids, bits, n_ticks, device="cuda"):
    """bits [B,32,6] bool -> (cell raster counts [B,2048], output first spike [B])"""
    from spiky.spnet.spnet import NeuronDataType
    B = len(bits)
    va = np.zeros((B, n_ticks, 384), np.float32)
    flat = bits.reshape(B, 32 * 6)
    idx_gt = np.arange(192) * 2
    va[:, T_RAIL, idx_gt] = np.where(flat, 1e6, 0.0)
    va[:, T_RAIL, idx_gt + 1] = np.where(flat, 0.0, 1e6)
    sid = torch.as_tensor(np.ascontiguousarray(ids[0], dtype=np.int32),
                          device=device).reshape(1, 1, -1).expand(B, n_ticks, -1).contiguous()
    net.process_ticks(n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
                      input_values=torch.as_tensor(va, device=device), sparse_input=sid,
                      do_train=False, do_record_voltage=False, do_reset_context=True,
                      _stdp_period=32)
    cid = torch.as_tensor(np.ascontiguousarray(ids[1], dtype=np.int32), device=device)
    C = net.export_neuron_data(cid, B, NeuronDataType.Spike, 0, n_ticks - 1)
    C = C.reshape(B, 2048, n_ticks).ne(0).sum(-1).cpu().numpy()
    oid = torch.as_tensor(np.ascontiguousarray(ids[2], dtype=np.int32), device=device)
    O = net.export_neuron_data(oid, B, NeuronDataType.Spike, 0, n_ticks - 1)
    O = O.reshape(B, 1, n_ticks).ne(0)
    w = torch.arange(n_ticks, 0, -1, device=O.device, dtype=torch.float32)
    T = (n_ticks - (O.float() * w).amax(-1)).cpu().numpy()[:, 0].astype(np.int64)
    return C, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    A_, B_ = Z["anchor_a"], Z["anchor_b"]
    W = Z["weights"].astype(np.float64)
    ntr = len(x) - 4000
    xs = x[ntr:ntr + a.n]
    bits = (xs[:, A_] > xs[:, B_])                       # ground truth, strict >
    code = (bits * (2 ** np.arange(5, -1, -1))).sum(-1)  # [n,32]
    w_sel = W.reshape(32 * 64, 6)[code + (np.arange(32) * 64)[None, :]]
    tgt = TPH * TAU * (np.log(np.exp(np.clip(w_sel[:, :, a.dim] / TAU, -60, 60)).sum(1))
                       - np.log(TPH))
    tau_eff, per = calib(TAU_M)
    scale = tau_eff / TAU
    Wd = W[:, :, a.dim]
    c0 = float(np.ceil(scale * Wd.max() + 2))
    arr = np.rint(-scale * Wd + c0)
    # amplitude: theta=1 must be crossed only AFTER the last of the 32 arrivals
    Vlast = np.exp((arr.max() - arr) / tau_eff).sum()
    amp = 1.0 / (2.0 * Vlast)
    n_ticks = int(arr.max() + 12 * TAU_M + 20)
    print(f"Stage2+3, dim {a.dim}: tau_eff {tau_eff:.4f}, scale {scale:.2f} ticks/unit-w, "
          f"delays [{int(arr.min())},{int(arr.max())}], amp {amp:.3e}, episode {n_ticks}")

    net, ids, nsyn, dly = build(Z, a.dim, scale, c0, amp, TAU_M)
    print(f"assembled: {384 + 2048 + 1} neurons (384 rails, 2048 cells, 1 output), "
          f"{nsyn} synapses\n")
    C, T = [], []
    for i in range(0, a.n, a.chunk):
        c, t = run(net, ids, bits[i:i + a.chunk], n_ticks)
        C.append(c)
        T.append(t)
    C = np.concatenate(C)
    T = np.concatenate(T)
    del net
    torch.cuda.empty_cache()

    per_table = C.reshape(-1, 32, 64)
    n_fired = (per_table > 0).sum(-1)
    winner = per_table.argmax(-1)
    onehot = float((n_fired == 1).mean())
    correct = float((winner == code)[n_fired == 1].mean())
    print(f"STAGE 2 COINCIDENCE:")
    print(f"  exactly one cell fires per table: {100 * onehot:.4f}%  "
          f"(0 cells {100 * float((n_fired == 0).mean()):.4f}%, "
          f">1 cell {100 * float((n_fired > 1).mean()):.4f}%)")
    print(f"  selected cell == true code (when one-hot): {100 * correct:.4f}%")
    fired = T < n_ticks
    ok = fired
    aa = np.polyfit(T[ok].astype(float), tgt[ok], 1) if ok.sum() > 10 else (0, 0)
    pred = aa[0] * T.astype(float) + aa[1]
    mse = float(((pred[ok] - tgt[ok]) ** 2).mean())
    var = float(tgt[ok].var())
    print(f"\nSTAGE 2+3 END TO END (vs the LUT's own output on these bits):")
    print(f"  output fired {100 * fired.mean():.2f}%   held-out MSE {mse:.6f}  "
          f"max|err| {float(np.abs(pred[ok] - tgt[ok]).max()):.4f}  var {var:.6f}  "
          f"R2 {1 - mse / var:.6f}")
    if a.out:
        json.dump(dict(dim=a.dim, n=a.n, neurons=384 + 2048 + 1, synapses=nsyn,
                       n_ticks=n_ticks, onehot=onehot, cell_correct=correct,
                       frac_fired=float(fired.mean()), mse=mse, r2=1 - mse / var,
                       target_var=var), open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

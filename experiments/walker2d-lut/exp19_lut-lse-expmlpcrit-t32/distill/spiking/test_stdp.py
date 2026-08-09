"""Bisect the STDP illegal-address crash: explicit wiring vs many plastic metas.

Each case runs in its OWN PROCESS. A CUDA illegal-address poisons the context, so a single
process would report every later case as failed regardless of its own merit.

    python test_stdp.py            # run all four cases
    python test_stdp.py --case A   # run one (this is what the driver invokes)
"""
import argparse
import subprocess
import sys

import numpy as np
import torch

NE, NI = 64, 16
D_MIN, D_MAX = 1, 20
W_INH = -5.0


def make_metas(n_plastic, lr, gs, same_delay=False, no_inh=False):
    from spiky.spnet.spnet import SynapseMeta
    exc = [SynapseMeta(learning_rate=lr,
                       min_delay=D_MIN if same_delay else d,
                       max_delay=D_MIN if same_delay else d, initial_weight=0.0,
                       min_weight=0.0, max_weight=20.0, initial_noise_level=0.0,
                       weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=gs, _backward_group_size=gs)
           for d in range(D_MIN, D_MIN + n_plastic)]
    if no_inh:
        return exc
    inh = [SynapseMeta(learning_rate=0.0, min_delay=D_MIN, max_delay=D_MIN,
                       initial_weight=W_INH, min_weight=W_INH, max_weight=W_INH,
                       initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=gs, _backward_group_size=gs)]
    return exc + inh


def run_case(case, gs=8, lr=0.1, same_delay=False, n_meta=None, max_syn=NE,
             one_meta_per_source=False, sort_targets=False):
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import (SynapseGrowthEngine, GrowthCommand)
    from spiky.util.chunk_of_connections import ChunkOfConnections
    dev = "cuda"
    torch.manual_seed(1)
    rng = np.random.default_rng(0)
    explicit = case in ("A", "B", "D")
    n_plastic = n_meta if n_meta else (20 if case == "B" else 1)
    train = case != "D"

    metas = make_metas(n_plastic, lr, gs, same_delay)
    sp = SpikingNet(synapse_metas=metas,
                    neuron_metas=[NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                                  NeuronMeta(neuron_type=1, a=0.1, d=2.0)],
                    neuron_counts=[NE, NI], initial_synapse_capacity=1 << 18,
                    summation_dtype=torch.float32)
    sp.to_device(dev)
    E = sp.get_neuron_ids_by_meta(0)
    I = sp.get_neuron_ids_by_meta(1)
    ge = SynapseGrowthEngine(device=dev, synapse_group_size=gs,
                             max_groups_in_buffer=1 << 15)
    box = dict(x1=-100.0, y1=-100.0, z1=-100.0, x2=100.0, y2=100.0, z2=100.0, p=0.5)
    if explicit:
        ge.register_neuron_type(max_synapses=max_syn, growth_command_list=[])
        ge.register_neuron_type(max_synapses=max_syn, growth_command_list=[])
    else:
        ge.register_neuron_type(max_synapses=NE, growth_command_list=[
            GrowthCommand(target_type=0, synapse_meta_index=0, max_synapses=16, **box)])
        ge.register_neuron_type(max_synapses=NE, growth_command_list=[
            GrowthCommand(target_type=0, synapse_meta_index=n_plastic, max_synapses=16,
                          **box)])
    z = torch.zeros(NE)
    ge.add_neurons(neuron_type_index=0, identifiers=E,
                   coordinates=torch.stack([z, z, torch.arange(NE).float()], 1))
    z = torch.zeros(NI)
    ge.add_neurons(neuron_type_index=1, identifiers=I,
                   coordinates=torch.stack([z, z + 1, torch.arange(NI).float()], 1))

    if explicit:
        keep = {}
        for _ in range(NE * 16):
            s, t = int(E[rng.integers(NE)]), int(E[rng.integers(NE)])
            if s != t:
                # one_meta_per_source: a source's synapses ALL sit in one meta, so no
                # source ever spans two metas -- while the net still uses n_plastic metas.
                keep[(s, t)] = (s % n_plastic if one_meta_per_source
                                else int(rng.integers(0, n_plastic)))
        for _ in range(NI * 8):
            s, t = int(I[rng.integers(NI)]), int(E[rng.integers(NE)])
            keep[(s, t)] = n_plastic                      # the frozen inhibitory meta
        tri = np.array([[m, s, t] for (s, t), m in keep.items()], np.int32)
        w = np.where(tri[:, 0] == n_plastic, W_INH, 6.0).astype(np.float32)
        from es_harness import group_aligned_weights
        tri_t = torch.tensor(tri, dtype=torch.int32, device=dev)
        ch = ge._grow_explicit(tri_t, 1, do_sort_by_target_id=sort_targets)
        conn = ch.get_connections()
        ch = ChunkOfConnections(conn, gs, weights=group_aligned_weights(
            conn, tri_t, torch.tensor(w, device=dev), gs))
        sp.add_connections(ch, 1)
    else:
        ch = ge.grow(1)
        sp.add_connections(ch, 1)
    ch.recycle()
    sp.compile(shuffle_synapses_random_seed=None)

    N = NE + NI
    ids = torch.cat([E, I])
    n0 = sp.count_synapses(ids, True)
    b = [torch.zeros(n0, dtype=t, device=dev) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(ids, b[0], b[1], b[2], b[3], b[4], True)
    w_before = b[2].cpu().numpy().copy()
    meta_before = b[1].cpu().numpy().copy()

    spk = torch.randint(N, [1, 200, 1], device=dev, dtype=torch.int32)
    val = torch.ones_like(spk, dtype=torch.float32) * 20.0
    for _ in range(5):
        sp.process_ticks(n_ticks_to_process=200, batch_size=1, n_input_ticks=200,
                         input_values=val, do_train=train, sparse_input=spk,
                         do_record_voltage=False, do_reset_context=False, _stdp_period=32)
        spk.random_(0, N)
    torch.cuda.synchronize()

    sp.export_synapses(ids, b[0], b[1], b[2], b[3], b[4], True)
    w_after = b[2].cpu().numpy()
    exc = meta_before != n_plastic
    inh = ~exc
    print(f"PASS  synapses {n0}  plastic_metas {n_plastic}  explicit {explicit}  "
          f"train {train}")
    print(f"  excitatory: {int(exc.sum())} synapses, max|dw| {np.abs(w_after[exc]-w_before[exc]).max():.4f}, "
          f"moved {int((w_after[exc] != w_before[exc]).sum())}")
    print(f"  inhibitory: {int(inh.sum())} synapses, all still {W_INH}: "
          f"{bool((w_after[inh] == W_INH).all())}")


CASES = {"A": "explicit wiring + ONE plastic meta,  do_train=True",
         "B": "explicit wiring + TWENTY plastic metas, do_train=True",
         "C": "SPATIAL growth  + ONE plastic meta,  do_train=True  (arm_b control)",
         "D": "explicit wiring + ONE plastic meta,  do_train=FALSE (stage-one control)"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--gs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-meta", type=int, default=None)
    ap.add_argument("--same-delay", action="store_true")
    ap.add_argument("--max-syn", type=int, default=NE)
    ap.add_argument("--one-meta-per-source", action="store_true")
    ap.add_argument("--sort-targets", action="store_true")
    a = ap.parse_args()
    if a.case:
        run_case(a.case, gs=a.gs, lr=a.lr, same_delay=a.same_delay, n_meta=a.n_meta,
                 max_syn=a.max_syn, one_meta_per_source=a.one_meta_per_source,
                 sort_targets=a.sort_targets)
    else:
        for c, desc in CASES.items():
            print(f"\n=== CASE {c}: {desc} ===", flush=True)
            r = subprocess.run([sys.executable, __file__, "--case", c, "--gs", str(a.gs)],
                               capture_output=True, text=True)
            out = [ln for ln in (r.stdout + r.stderr).splitlines()
                   if "PASS" in ln or "excitatory:" in ln or "inhibitory:" in ln
                   or "Error" in ln or "error" in ln]
            print("\n".join(f"  {ln.strip()}" for ln in out[-4:]) or "  (no output)")

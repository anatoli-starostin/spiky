#!/usr/bin/env python3
"""REGRESSION: _grow_explicit(weights=) must put every weight on the edge it was given for,
and must give the same answer every time.

WHAT THIS CATCHES. `SynapseGrowthEngine._build_group_aligned_weights` used to recover each
connection group's owning source by carrying the last non-zero source header FORWARD IN MEMORY
ORDER, on the premise it stated in its own docstring: "the explicit build lays each source's
groups out contiguously". It does not.

  * The grow kernel gives each (synapse_meta, source) sublist its own thread and every thread
    takes its next group from one global bump allocator -- `atomicAdd(n_allocated, ...)`,
    native/spiky/synapse_growth/aux_/synapse_growth_kernels_logic.cu:1324 -- so group placement
    follows GPU scheduling order.
  * `finalize() -> merge_chains` then stitches a source's per-meta chains into one by DEMOTING
    all but one root to source_id 0 and linking the tail with a SIGNED offset to an arbitrary
    group, forwards or backwards (same file, 1400-1418).

Measured on a real 84,626-edge build at group size 128: 38,002 groups, 19,001 occupied, 1,017
roots left of ~19,000, ZERO of 17,984 chain links landing on the physically next group, about
half pointing backward, median jump 5,082 groups. The forward-fill named the right source for
5.4 % of groups; every mis-attributed group missed its lookup and silently took 0.0, so only
47.8 % of the non-zero weights and 32 % of the total weight were delivered -- a different 32 %
on every rebuild of the same input.

THE TWO DEFECTS ARE INDEPENDENT, so this test checks them separately:
  * CORRECTNESS is host-side and deterministic. It reproduces on the CPU build path, where the
    buffer is byte-identical between runs and the forward-fill is still 91.7 % wrong, because
    merge_chains rewrites the links whatever the allocator did.
  * REPRODUCIBILITY is the device-side atomic ordering. Chain-following makes the weights
    independent of layout, so they come out identical even though the layout does not.

The topology must span MANY METAS PER SOURCE or merge_chains never stitches anything and the
bug cannot appear -- that is why this builds a multi-meta net rather than reusing
test_explicit_weights.py's single-meta one.

Not pytest -- the spnet suite is standalone scripts:
    python test_explicit_weight_alignment.py            # cpu and cuda
"""
import numpy as np
import torch

from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta
from spiky.util.synapse_growth import SynapseGrowthEngine

N_SRC, N_TGT, N_METAS, FANOUT = 80, 80, 20, 40
GROUP_SIZE = 128                     # the production value; the bug needs chained groups


def _triples_and_weights(seed):
    """A multi-meta explicit edge list: every source spans most of the meta bank."""
    rng = np.random.default_rng(seed)
    seen = {}
    for s in range(N_SRC):
        for t in rng.choice(N_TGT, FANOUT, replace=False):
            seen[(s, int(t))] = int(rng.integers(0, N_METAS))     # (src,tgt) unique by dict
    edges = sorted(seen.items())
    tri = np.array([[m, s, t] for (s, t), m in edges], dtype=np.int32)
    # distinct, non-zero, both signs, so a dropped weight cannot pass as a correct 0.0
    w = rng.uniform(0.5, 5.0, len(edges)) * rng.choice([-1.0, 1.0], len(edges))
    return tri, w.astype(np.float32)


def _build(device, summation_dtype, tri, w, seed):
    metas = [SynapseMeta(learning_rate=0.0, min_delay=d + 1, max_delay=d + 1,
                         min_weight=-10.0, max_weight=10.0, initial_weight=0.0,
                         _forward_group_size=GROUP_SIZE, _backward_group_size=GROUP_SIZE)
             for d in range(N_METAS)]
    spnet = SpikingNet(synapse_metas=metas, neuron_metas=[NeuronMeta(neuron_type=0)],
                       neuron_counts=[N_SRC + N_TGT], summation_dtype=summation_dtype)
    spnet.to_device(device)
    ids = spnet.get_neuron_ids_by_meta(0).cpu().numpy()

    ge = SynapseGrowthEngine(device=device, synapse_group_size=GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(tri) + ids.size)))
    ge.register_neuron_type(max_synapses=8 * (N_SRC + N_TGT), growth_command_list=[])
    n = ids.size
    ge.add_neurons(neuron_type_index=0, identifiers=torch.tensor(ids, dtype=torch.int32),
                   coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                            torch.zeros(n)], dim=1))

    # sources are the first N_SRC neurons, targets the last N_TGT
    gtri = np.stack([tri[:, 0], ids[tri[:, 1]], ids[N_SRC + tri[:, 2]]], 1).astype(np.int32)
    chunk = ge._grow_explicit(torch.tensor(gtri, dtype=torch.int32, device=device), seed,
                              weights=torch.tensor(w, dtype=torch.float32, device=device))
    spnet.add_connections(chunk, seed)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=None)

    n_syn = spnet.n_synapses()
    bufs = [torch.zeros([n_syn], dtype=t, device=device) for t in
            (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    spnet.export_synapses(spnet.get_all_neuron_ids(), bufs[0], bufs[1], bufs[2], bufs[3],
                          bufs[4], forward_or_backward=True)
    es, em, ew, _ed, et = (x.cpu().numpy() for x in bufs)
    k = np.lexsort((em, et, es))
    kk = np.lexsort((gtri[:, 0], gtri[:, 2], gtri[:, 1]))
    return dict(got_src=es[k], got_meta=em[k], got_w=ew[k], got_tgt=et[k],
                want_src=gtri[kk, 1], want_meta=gtri[kk, 0], want_tgt=gtri[kk, 2],
                want_w=w[kk])


def _check(r, label):
    topo = (np.array_equal(r["got_src"], r["want_src"])
            and np.array_equal(r["got_tgt"], r["want_tgt"])
            and np.array_equal(r["got_meta"], r["want_meta"]))
    ok = np.isclose(r["got_w"], r["want_w"], atol=1e-5)
    delivered = np.abs(r["got_w"]).sum() / np.abs(r["want_w"]).sum()
    print(f"  {label}: {len(ok):,} edges, topology_ok={topo}, "
          f"weights correct {ok.sum():,} ({100 * ok.mean():.2f} %), "
          f"|weight| delivered {100 * delivered:.2f} %, "
          f"dropped to 0 {int(((r['got_w'] == 0) & (r['want_w'] != 0)).sum()):,}")
    return topo, ok, delivered


def test_explicit_weight_alignment(device='cpu', summation_dtype=torch.float32, seed=1):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        return None
    if summation_dtype != torch.float32:
        return None      # this compares exported weights for exact equality

    tri, w = _triples_and_weights(seed)
    print(f"  multi-meta explicit build: {len(tri):,} edges, {N_METAS} metas, "
          f"group_size {GROUP_SIZE}, "
          f"{len(np.unique(tri[tri[:, 1] == 0, 0]))} metas on source 0")

    r1 = _build(device, summation_dtype, tri, w, seed)
    topo, ok, delivered = _check(r1, "build 1")
    r2 = _build(device, summation_dtype, tri, w, seed)
    _check(r2, "build 2")
    identical = bool(np.array_equal(r1["got_w"], r2["got_w"]))
    print(f"  two builds, same seed: weight vectors byte-identical = {identical}")

    good = True
    if not topo:
        print("  FAIL: exported topology does not match the requested triples")
        good = False
    if not ok.all():
        print(f"  FAIL: {int((~ok).sum()):,} of {len(ok):,} weights landed on the wrong edge "
              f"(only {100 * ok.mean():.2f} % correct)")
        good = False
    if abs(delivered - 1.0) > 1e-6:
        print(f"  FAIL: only {100 * delivered:.2f} % of the requested |weight| was delivered")
        good = False
    if not identical:
        d = r1["got_w"] != r2["got_w"]
        print(f"  FAIL: two builds with seed {seed} differ on {int(d.sum()):,} edges")
        good = False

    # the (source, target) uniqueness guard must fire rather than silently last-one-wins
    try:
        dup = np.array([[0, 1, 2], [1, 1, 2]], dtype=np.int32)     # same (src,tgt), two metas
        ge = SynapseGrowthEngine(device=device, synapse_group_size=GROUP_SIZE,
                                 max_groups_in_buffer=4096)
        ge.register_neuron_type(max_synapses=64, growth_command_list=[])
        ge.add_neurons(neuron_type_index=0, identifiers=torch.tensor([1, 2], dtype=torch.int32),
                       coordinates=torch.zeros([2, 3]))
        ge._grow_explicit(torch.tensor(dup, dtype=torch.int32, device=device), seed,
                          weights=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device))
        print("  FAIL: duplicate (source, target) under two metas was accepted silently")
        good = False
    except ValueError as e:
        print(f"  duplicate (source, target) rejected: {str(e)[:70]}...")

    print("  passed" if good else "  FAILED")
    return good


def main():
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    rc = 0
    for device in devices:
        for seed in (1, 42):
            print(f"\ntest_explicit_weight_alignment on {device}, seed {seed}:")
            if test_explicit_weight_alignment(device, torch.float32, seed) is False:
                rc = 1
    print("\n🎉 explicit weight alignment test passed!" if rc == 0 else "\ntest FAILED")
    return rc


if __name__ == "__main__":
    exit(main())

"""Validate the chunk _grow_explicit produces, on CPU, for the configs that crash on GPU.

ChunkOfConnections documents the invariants the native code relies on -- notably
  5. "For each different source_neuron_id only one list may be present."
  7. "Groups in each list are sorted by the synapse_meta_index header field."
and ships a validator. If the explicit build violates one of these when a source spans
metas, that is the bug, visible with no CUDA and no crash.
"""
import numpy as np
import torch
from spiky.util.chunk_of_connections import ChunkOfConnectionsValidator
from spiky.util.synapse_growth import SynapseGrowthEngine

NE, NI, GS = 64, 16, 8


def build(n_meta, one_meta_per_source):
    rng = np.random.default_rng(0)
    E = list(range(1, NE + 1))
    I = list(range(NE + 1, NE + NI + 1))
    keep = {}
    for _ in range(NE * 16):
        s, t = E[rng.integers(NE)], E[rng.integers(NE)]
        if s != t:
            keep[(s, t)] = (s % n_meta if one_meta_per_source
                            else int(rng.integers(0, n_meta)))
    for _ in range(NI * 8):
        s, t = I[rng.integers(NI)], E[rng.integers(NE)]
        keep[(s, t)] = n_meta
    tri = np.array([[m, s, t] for (s, t), m in keep.items()], np.int32)

    ge = SynapseGrowthEngine(device="cpu", synapse_group_size=GS,
                             max_groups_in_buffer=1 << 15)
    for _ in range(2):
        ge.register_neuron_type(max_synapses=640, growth_command_list=[])
    for ti, ids in ((0, E), (1, I)):
        t = torch.tensor(ids, dtype=torch.int32)
        n = t.numel()
        ge.add_neurons(neuron_type_index=ti, identifiers=t,
                       coordinates=torch.stack([torch.arange(n).float(),
                                                torch.zeros(n),
                                                torch.full((n,), float(ti))], 1))
    return ge._grow_explicit(torch.tensor(tri, dtype=torch.int32), 1)


for n_meta, omps, gpu in ((1, True, "PASS"), (2, True, "CRASH"), (4, True, "PASS"),
                          (2, False, "CRASH"), (4, False, "CRASH"), (20, False, "CRASH")):
    try:
        chunk = build(n_meta, omps)
        ok, errs = ChunkOfConnectionsValidator(chunk).validate_all()
    except Exception as e:
        ok, errs = False, [f"{type(e).__name__}: {e}"]
    lbl = "one-meta-per-source" if omps else "sources SPAN metas "
    print(f"  metas {n_meta:2d}  {lbl}  gpu={gpu:5s} -> chunk valid: {ok}")
    for e in (errs or [])[:4]:
        print(f"        {e}")

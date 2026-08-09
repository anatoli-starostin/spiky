"""Does structural mutation push a target neuron past what backward_group_size=32 supports?

We measured earlier that the backward/STDP structure overflows past ~8 incoming PLASTIC
synapses per target at backward_group_size=2, and that raising it to 32 cleared the ~100
worst case at seed. Structural mutation ADDS excitatory synapses, so a target's incoming
plastic count drifts upward over rounds. If some neuron crosses a limit, that is a candidate
explanation for the hangs (K=32 at round 136, K=128 reproducibly at round 3 -- its first
round containing mutated genomes).

CPU-only, no CUDA: just count in-degrees on the genomes.
"""
import numpy as np

import steady_state as S

W_MAX = 30.0
rng = np.random.default_rng(0)


def plastic_indegree(g):
    """Incoming EXCITATORY (= plastic) synapses per target neuron, within one genome."""
    exc = g["src_pool"] != S.INH
    key = g["tgt_pool"][exc] * 100000 + g["tgt_idx"][exc]
    _, counts = np.unique(key, return_counts=True)
    return counts


g = S.seed_genome(np.random.default_rng(0), W_MAX)
c = plastic_indegree(g)
print(f"seed genome: {g['weight'].size:,} synapses, "
      f"plastic in-degree max {c.max()} mean {c.mean():.1f} p99 {np.percentile(c, 99):.0f}")

for rnd in range(1, 9):
    g = S.mutate_structural(S.clone(g), rng, W_MAX)
    c = plastic_indegree(g)
    print(f"  after {rnd} structural mutations: {g['weight'].size:,} synapses, "
          f"in-degree max {c.max():4d} mean {c.mean():5.1f} p99 {np.percentile(c, 99):3.0f}"
          f"   {'<-- OVER 32' if c.max() > 32 else ''}")

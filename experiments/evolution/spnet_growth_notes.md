# Population-in-one-SPNet — notes for the Mode-2 harness

Source: `spiky/workbooks/spnet.ipynb` (the "1000 subnets in one SpikingNet" demo)
+ `src/spiky/spnet/tests/test_izhikevitch_runtime.py` (explicit-triples build)
+ `src/spiky/util/test_utils.py` / `spnet_test_utils.py` (helpers).

## The core idea
Pack `n_subnets` independent sub-networks into ONE `SpikingNet` and run them all in
parallel in a single `process_ticks` call. Isolation is by **neuron id blocks** (each
subnet owns a contiguous range of global neuron ids) and, for *growth-based* wiring,
by **spatial coordinate shift** (subnet s placed at `+s*1000` so spatial growth
commands never reach across subnets). For *explicit* wiring the ids alone isolate —
no coordinate trick needed.

## Izhikevich dynamics (matched, from native `test_math_logic` / spnet.h)
`v` updated with **N_EULER_STEPS=2** half-steps of `EULER_DT=0.5`:
`v += 0.5*((0.04*v+5)*v + 140 - u + I)` (twice); then `u += a*(b*v - u)`; reset at
start-of-tick if `v>threshold(30)`: `v=c(-65)`, `u+=d`. Current is **+I** (native
code; the spnet.py docstring's "-I" is wrong). NeuronMeta defaults a=.02 b=.2 c=-65
d=8. Init v=c, u=b*c.

## Building a net
```
from spiky.util.synapse_growth import SynapseGrowthEngine, GrowthCommand
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType
from spiky.util.test_utils import extract_connection_map, convert_connections_to_export_format
```
1. `ge = SynapseGrowthEngine(device='cpu', synapse_group_size=32, max_groups_in_buffer=...)`
2. Register one neuron *type* per NeuronMeta: `ge.register_neuron_type(max_synapses=BIG, growth_command_list=[...])`.
   (For explicit wiring pass `growth_command_list=[]`.)
3. `spnet = SpikingNet(synapse_metas=[...], neuron_metas=[...], neuron_counts=[c0,c1,...])`
   — `neuron_counts[i]` neurons of meta i. Ids are contiguous per meta;
   `spnet.get_neuron_ids_by_meta(i)` returns them. NOTE counts are rounded up to a
   multiple of 4 internally, but `get_neuron_ids_by_meta` returns exactly the count.
4. `spnet.to_device('cpu')`
5. `ge.add_neurons(neuron_type_index=i, identifiers=<ids>, coordinates=<[n,3] float>)`
   for each type (coords matter only for spatial growth; any coords for explicit).

### Explicit arbitrary wiring (what we need)
- `triples = torch.tensor([[synapse_meta_idx, source_id, target_id], ...], dtype=int32)`
  — **global** neuron ids. The synapse's **weight & delay come from the SynapseMeta**
  the `synapse_meta_idx` points at (NOT from the triple). So to place an explicit
  (weight w, delay d) synapse, make a `SynapseMeta(initial_weight=w, min_weight=w,
  max_weight=w, min_delay=d, max_delay=d, learning_rate=0.0)` and reference its index.
  → build a **palette of metas**, one per distinct (w,d) you need.
- `chunk = ge._grow_explicit(triples, grow_seed)`  (the "backdoor")
- `spnet.add_connections(chunk, random_seed)`  then `chunk.recycle()`
- `spnet.compile(shuffle_synapses_random_seed=seed)` (finalizes forward/backward groups)
- (helper `extract_connection_map(ge, metas, seed, explicit_triples=triples, do_validate=False)`
   validates/inspects; `grow_and_add` does grow+add in one call.)

## Input injection (latency-coded spikes)
`process_ticks(..., sparse_input=S, input_values=V, n_input_ticks=T)`:
- `S`: int32 `[batch, n_ticks, K]` — each entry is a **global neuron id** to stimulate
  at that tick (K events/tick). `V`: float32 same shape — the **current** applied.
- To make an input neuron "fire" at latency tick `t_i = α+β·x_i`, inject a strong
  current (≥~15–30) into that neuron at tick `t_i` (it then Izhikevich-fires ~that
  tick and propagates through its synapses). Pad unused event slots by repeating a
  harmless id with value 0.
- Same input into every subnet: emit the event for each subnet's copy of that input
  neuron (global id = local_id + subnet_block_offset).

## Output readout
`raster = spnet.export_neuron_data(out_ids, batch, NeuronDataType.Spike, first_tick, last_tick)`
→ `[batch, len(out_ids), n_ticks]` spike raster (1.0 at spike ticks). Per subnet, read
its output-neuron ids; **first-spike tick** decodes the latency-coded output value
(value = (t_out − A)/B, same code as inputs). `NeuronDataType.Voltage` gives the v trace.

## Isolation / no-cross-talk
Different subnets share NO synapses (explicit triples only wire within a block) and
NO spatial adjacency (growth). Running N subnets packed must give **identical**
per-subnet spikes to running each alone → the batching-correctness check.

## Gotchas
- `import torch` **before** `import spiky_cuda` (libc10.so must load first).
- CPU build only (`torch.cuda.is_available()==False`); everything on `device='cpu'`.
- Weight/delay are per-meta, so continuous per-synapse weights ⇒ many metas (fine for
  small nets; quantize to a palette if the count explodes).
- Our exact 26/84 construction is **leak-free IF**; native Izhikevich is nonlinear and
  resets to −65, so the ported construction will NOT be exact — measure how close.

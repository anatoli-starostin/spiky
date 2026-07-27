"""Phase-0 harness for LUT -> spiking (issue #74).

Thin, explicit wrapper around spiky's SpNet (Izhikevich) that gives us:
  * hand-authored connectivity (no growth engine): exact (source, target, weight, delay)
  * latency-coded input injection (a current impulse at an exact tick per neuron)
  * first-spike readout from the spike raster
  * a (weight, delay) -> SynapseMeta allocator, since the engine has no
    per-synapse delay input and SpikingNet.add_connections drops per-synapse weights.

Design notes (verified against the engine source):
  - delays are integers 0..255 and are assigned round-robin over [min_delay, max_delay]
    within a (source, synapse_meta) sublist -> to pin an exact delay we register one
    SynapseMeta per (weight, delay) pair with min_delay == max_delay.
  - I (the synaptic current accumulator) is zeroed every tick -> synaptic input is a
    per-tick impulse, there is no synaptic trace.
  - at most ONE synapse per (source, target) pair (ChunkOfConnections rule 11).
"""
import torch
from collections import defaultdict

from spiky.spnet.spnet import SpikingNet, NeuronMeta, SynapseMeta, NeuronDataType
from spiky.util.chunk_of_connections import ChunkOfConnections, ChunkOfConnectionsValidator

GROUP_INTS = 6  # single_group_size = 1 -> 4 header + 2 body


class Net:
    """A hand-built spiking network with exact per-synapse weights and delays."""

    def __init__(self, n_neurons, neuron_meta=None, device="cuda:0"):
        self.n_neurons = n_neurons
        self.device = device
        self.neuron_meta = neuron_meta or NeuronMeta(neuron_type=0)
        self._syn = []                 # (src_idx, dst_idx, weight, delay)
        self._meta_key_to_idx = {}     # (weight, delay) -> meta index
        self._meta_list = []
        self._built = False

    # ---- construction -------------------------------------------------
    def meta_for(self, weight, delay):
        key = (round(float(weight), 6), int(delay))
        if key not in self._meta_key_to_idx:
            w, d = key
            self._meta_key_to_idx[key] = len(self._meta_list)
            self._meta_list.append(SynapseMeta(
                learning_rate=0.0,
                min_delay=d, max_delay=d,
                min_weight=min(0.0, w), max_weight=max(0.0, w),
                initial_weight=w,
                initial_noise_level=0.0,
                weight_decay=1.0, weight_scaling_cf=0.0,
            ))
        return self._meta_key_to_idx[key]

    def connect(self, src, dst, weight, delay):
        """src/dst are 0-based neuron indices."""
        self._syn.append((int(src), int(dst), float(weight), int(delay)))

    def n_synapses(self):
        return len(self._syn)

    def _build_chunk(self, ids):
        """Author the ChunkOfConnections buffer by hand (single_group_size=1)."""
        by_src = defaultdict(list)
        for s, t, w, d in self._syn:
            by_src[s].append((self.meta_for(w, d), t))

        groups = []
        for s in sorted(by_src):
            syns = sorted(by_src[s])                      # by (meta, target)
            # count per meta sublist
            cnt = defaultdict(int)
            for m, _ in syns:
                cnt[m] += 1
            seen = set()
            n = len(syns)
            for i, (m, t) in enumerate(syns):
                head_of_list = (i == 0)
                head_of_sub = m not in seen
                seen.add(m)
                groups.append([
                    int(ids[s]) if head_of_list else 0,    # source id (0 for continuation)
                    m,                                     # synapse meta (header)
                    cnt[m] if head_of_sub else 0,          # n targets in this sublist
                    0 if i == n - 1 else GROUP_INTS,       # shift to next group
                    m,                                     # body: meta
                    int(ids[t]),                           # body: target id
                ])
        buf = torch.tensor(groups, dtype=torch.int32).flatten()
        return ChunkOfConnections(buf, 1)

    def build(self, validate=True):
        assert not self._built
        # metas must be registered at construction time -> materialise them first
        for s, t, w, d in self._syn:
            self.meta_for(w, d)
        if not self._meta_list:
            self._meta_list.append(SynapseMeta(min_delay=1, max_delay=1))
        self.net = SpikingNet(
            synapse_metas=self._meta_list,
            neuron_metas=[self.neuron_meta],
            neuron_counts=[self.n_neurons],
            initial_synapse_capacity=max(1024, 4 * len(self._syn) + 1024),
        )
        self.ids = self.net.get_neuron_ids_by_meta(0)   # 0-based index -> engine id
        chunk = self._build_chunk(self.ids.tolist())
        if validate:
            ok, errs = ChunkOfConnectionsValidator(chunk).validate_all()
            assert ok, errs
        self.net.add_connections(chunk, random_seed=1)
        self.net.compile(shuffle_synapses_random_seed=None)
        self.net.to_device(self.device)
        self.ids = self.net.get_neuron_ids_by_meta(0)
        self._built = True
        return self

    # ---- running ------------------------------------------------------
    def run(self, spike_times, current=None, n_ticks=64, amp=200.0):
        """spike_times: [B, n_neurons] float/int tick at which to kick each neuron
        (negative / NaN / >= n_ticks means 'no input'). Returns [B, n_neurons] first
        spike tick (or -1) plus the raw raster.
        """
        st = torch.as_tensor(spike_times)
        B, N = st.shape
        assert N == self.n_neurons
        # build [B, T, N] sparse id/value pair: at tick t inject amp into neuron i
        ids = self.ids.detach().cpu().to(torch.int32)
        sp = ids.view(1, 1, N).expand(B, n_ticks, N).contiguous()
        val = torch.zeros(B, n_ticks, N, dtype=torch.float32)
        t = st.clone()
        ok = (t >= 0) & (t < n_ticks)
        idx = torch.nonzero(ok, as_tuple=False)
        if idx.numel():
            val[idx[:, 0], t[ok].long(), idx[:, 1]] = amp if current is None else 0.0
            if current is not None:
                cur = torch.as_tensor(current)
                val[idx[:, 0], t[ok].long(), idx[:, 1]] = cur[ok].float()
        dev = self.net.get_device()
        sp = sp.to(dev)
        val = val.to(dev)
        n_spikes = self.net.process_ticks(
            n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
            input_values=val, do_train=False, sparse_input=sp,
            do_reset_context=True, do_record_voltage=False,
        )
        raster = self.net.export_neuron_data(
            self.ids, B, NeuronDataType.Spike, 0, n_ticks - 1)   # [B, N, T]
        first = first_spike(raster)
        return first, raster, n_spikes

    def voltage(self, spike_times, n_ticks=64, amp=200.0):
        st = torch.as_tensor(spike_times)
        B, N = st.shape
        ids = self.ids.detach().cpu().to(torch.int32)
        sp = ids.view(1, 1, N).expand(B, n_ticks, N).contiguous()
        val = torch.zeros(B, n_ticks, N, dtype=torch.float32)
        ok = (st >= 0) & (st < n_ticks)
        idx = torch.nonzero(ok, as_tuple=False)
        if idx.numel():
            val[idx[:, 0], st[ok].long(), idx[:, 1]] = amp
        dev = self.net.get_device()
        self.net.process_ticks(
            n_ticks_to_process=n_ticks, batch_size=B, n_input_ticks=n_ticks,
            input_values=val.to(dev), do_train=False, sparse_input=sp.to(dev),
            do_reset_context=True, do_record_voltage=True,
        )
        return self.net.export_neuron_data(
            self.ids, B, NeuronDataType.Voltage, 0, n_ticks - 1)


def first_spike(raster):
    """raster [B, N, T] (0/1) -> [B, N] first spike tick, -1 if silent."""
    r = (raster > 0)
    any_ = r.any(dim=-1)
    idx = r.float().argmax(dim=-1)
    return torch.where(any_, idx, torch.full_like(idx, -1))

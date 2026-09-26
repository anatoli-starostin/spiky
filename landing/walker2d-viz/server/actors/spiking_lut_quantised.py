"""A handcrafted SPIKING network reproducing the QUANTISED Walker2d LUT policy.

A FORK of `spiking_lut.py`, which is left byte-identical and keeps its own actor name. Two
differences, both consequences of the policy this one mirrors:

  INPUT   the shared 128-bucket Gaussian-companded map.
          tick = 127 - searchsorted(in_quant_edges, x_norm)   (larger value -> earlier tick)
          Only the EDGES are needed: the network consumes tick ORDERING, never the
          dequantised value, so ties (same bucket -> same tick) reach the comparators as
          genuine ties and the existing tie detectors give the LUT's strict-'>' convention.

  STAGE 3 AMPLITUDE encoding instead of delays. Each selected-cell->output synapse has
          delay 1 and weight beta_o * exp(w/tau); all 32 selected cells land on ONE tick and
          the (exactly linear) anti-leak membrane integrates sum_t exp(w_t/tau) directly.
          Because arrivals a_t = A - (tau_eff/tau)*w_t give
              alpha * e^{t/tau_eff} * sum_t e^{-a_t/tau_eff}
                = alpha*e^{-A/tau_eff} * e^{t/tau_eff} * sum_t e^{w_t/tau},
          this is the SAME readout with the delay spread removed -- same decode slope,
          different offset. It is also strictly more faithful, because the delay path rounds
          every delay with rint() and so evaluates a quantised logsumexp.

          Consequences: Stage-3 delay span 91 -> 1, dmax 91 -> 3, episode 309 -> 234, and the
          engine's 255-tick delay cap stops being anywhere near the design. It also removes
          the dependence on cross-tick synaptic integration, which this kernel does not have.

Verified against the software quantised actor: Stage-1 address bits 100.0000% (0 bad of
98,304), Stage-2 one-hot 0 none / 0 multi of 16,384, Stage-3 100.000% within one output
level with 74.8-82.4% exact on the 22-level grid (the residual is the integer-tick
quantisation of the crossing time).
"""
import os

import numpy as np

from .base import Actor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M_RAIL, TAU_MEM, W_MEM, W_GATE = 20.0, 1200.0, 0.6, 0.6
W_TIE, W_AND = 0.6, 0.18
# Completion detector: each of the 17 inputs fires exactly once, so a perfectly leak-free
# neuron fed by all of them crosses threshold when the LAST one arrives. 16*W < 1 <= 17*W.
W_DET = 0.06
# It must then WAIT OUT the comparator pipeline before latching memory: the last rail fires
# at t_last+3, its tie detector at +4, and the tie veto reaches memory at +5. Gating earlier
# measured 99.9196% bit parity and 78 multi-selects.
D_GATE = 6
# GT-SKEW replaces the 136 tie-detector neurons. Measured on the real kernel: the
# cross-inhibition lands one tick AFTER the excitation it must cancel, so on an exact tie
# both rails see pure +1.5187 and both fire. Adding one tick to the GT rail's excitation
# makes the tie coincide with the veto (-8.5, silent -> bit 0) while leaving the 1-tick-apart
# case untouched -- exactly the software's strict `d > 0`. Structural, not tuned: the gap
# between fires (+1.519) and silent (-8.606) is 10.1 against a threshold of 1.0.
DE_GT = DE + 1


def _pairs(anchor_a, anchor_b):
    pairs, slot, index = [], [], {}
    for t in range(anchor_a.shape[0]):
        for j in range(anchor_a.shape[1]):
            a, b = int(anchor_a[t, j]), int(anchor_b[t, j])
            key = (min(a, b), max(a, b))
            if key not in index:
                index[key] = len(pairs)
                pairs.append(key)
            slot.append((index[key], a == key[0]))
    return pairs, slot


class SpikingLutQuantisedActor(Actor):
    name = "Spiking LUT quantised (handcrafted SNN)"

    def __init__(self, action_space):
        super().__init__(action_space)
        import torch
        from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
        from spiky.util.synapse_growth import SynapseGrowthEngine

        Q = np.load(os.path.join(_MODELS, "spiking_lut_quantised_actor.npz"))
        self.obs_mean = Q["obs_mean"].astype(np.float64)
        self.obs_var = Q["obs_var"].astype(np.float64)
        self.edges = Q["in_quant_edges"].astype(np.float64)
        self.affine = Q["affine"].astype(np.float64)          # (6, 2) slope, offset
        self.t_in = int(Q["t_in"])
        self.gate_tick = int(Q["gate_tick"])
        self.n_ticks = int(Q["n_ticks"])
        self.d_out = int(Q["d_out"])
        self.lv = int(Q["out_quant_levels"])
        self.clip = float(Q["out_quant_clip"])
        self.step = 2.0 * self.clip / (self.lv - 1)
        beta = Q["beta"].astype(np.float64)
        W = Q["weights"].astype(np.float64)
        tau = float(Q["tau"])
        tau_m_out = float(Q["tau_m_out"])

        A_, B_ = Q["anchor_a"], Q["anchor_b"]
        pairs, slot = _pairs(A_, B_)
        P = len(pairs)

        dmax = max(DE, DE_GT, DI, self.d_out, D_GATE)
        smetas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d, initial_weight=0.0,
                              min_weight=-1e4, max_weight=1e4, initial_noise_level=0.0,
                              weight_decay=0.9, weight_scaling_cf=0.0,
                              _forward_group_size=64, _backward_group_size=64)
                  for d in range(1, dmax + 1)]
        anti = dict(cf_2=0.0, cf_0=0.0, a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=1.0)
        metas = [LIFNeuronMeta(neuron_type=0, tau=TAU_M_RAIL, threshold=1.0),
                 NeuronMeta(neuron_type=1, cf_1=+1.0 / TAU_M_RAIL, **anti),
                 LIFNeuronMeta(neuron_type=2, tau=TAU_M_RAIL, threshold=1.0),
                 LIFNeuronMeta(neuron_type=3, tau=TAU_MEM, threshold=1.0),
                 # (tie detectors removed -- the GT skew handles ties)
                 # leak-free: the 6-way AND no longer depends on arrival phase;
                 # a wrong cell is bounded by 5*0.18 = 0.90 for ALL time.
                 NeuronMeta(neuron_type=4, cf_1=0.0, **anti),
                 NeuronMeta(neuron_type=5, cf_1=+1.0 / tau_m_out, **anti),
                 NeuronMeta(neuron_type=6, cf_1=0.0, **anti)]   # completion detector
        counts = [18, 2 * P, 2 * P, 2 * P, 2048, 6, 1]
        net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=counts,
                         initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
        net.to_device("cpu")
        ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(7)]
        ids = ids[:4] + [np.zeros(0, np.int64)] + ids[4:]   # keep downstream indices stable
        inp = ids[0][:17]
        det = ids[7][0]                       # the INTERNAL gate; no external stimulus
        E = [(1, inp[j], det, W_DET) for j in range(17)]
        for p, (a, b) in enumerate(pairs):
            r0, r1 = ids[1][2 * p], ids[1][2 * p + 1]
            i0, i1 = ids[2][2 * p], ids[2][2 * p + 1]
            m0, m1 = ids[3][2 * p], ids[3][2 * p + 1]
            E += [(DE_GT, inp[a], r0, W_EXC), (DE, inp[b], r1, W_EXC),
                  (DI, inp[a], i0, W_EXC), (DI, inp[b], i1, W_EXC),
                  (1, i0, r1, W_INH), (1, i1, r0, W_INH),
                  (1, r0, m0, W_MEM), (1, r1, m1, W_MEM),
                  (D_GATE, det, m0, W_GATE), (D_GATE, det, m1, W_GATE)]
        for t in range(32):
            for k in range(64):
                cell = ids[5][t * 64 + k]
                for j in range(6):
                    bit = (k >> (5 - j)) & 1
                    p, a_first = slot[t * 6 + j]
                    hi_i = 2 * p + (0 if a_first else 1)
                    lo_i = 2 * p + (1 if a_first else 0)
                    E.append((1, ids[3][hi_i if bit else lo_i], cell, W_AND))
                for o in range(6):
                    E.append((self.d_out, cell, ids[6][o],
                              float(beta[o] * np.exp(W[t, k, o] / tau))))
        tri = np.array([[d - 1, s, tg] for d, s, tg, _ in E], np.int64)
        wts = np.array([w for *_, w in E], np.float64)
        ge = SynapseGrowthEngine(device="cpu", synapse_group_size=64,
                                 max_groups_in_buffer=max(1 << 19, 4 * len(tri)))
        for i in range(7):
            ge.register_neuron_type(max_synapses=1 << 15, growth_command_list=[])
        for i, gi in enumerate([j for j in range(8) if len(ids[j])]):
            tt = torch.tensor(ids[gi], dtype=torch.int32)
            ge.add_neurons(neuron_type_index=i, identifiers=tt,
                           coordinates=torch.stack([torch.arange(tt.numel()).float(),
                                                    torch.zeros(tt.numel()),
                                                    torch.full((tt.numel(),), float(i))], 1))
        chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32),
                                  1, weights=torch.tensor(wts, dtype=torch.float32))
        net.add_connections(chunk, 1)
        chunk.recycle()
        net.compile(shuffle_synapses_random_seed=None)
        self._torch = torch
        self.net, self.ids = net, ids

        # ---- spike-viz interface (additive; mirrors spiking_lut.py so the raster + network panels work) ----
        # ROW order puts the internal COMPLETION gate (ids[7]) right AFTER the inputs (ids[0]); the rest
        # follow. ids[4] is the empty tie slot. Both panels (raster y-axis + network graph) group by these
        # bands, and spikes + topology share this same row space.
        #
        # Memory cells (ids[3], the green "S1 memory" band) are wired interleaved per pair:
        # ids[3][2p]=m0 (driven by r0, the greater-than / "positive" rail) and ids[3][2p+1]=m1 (the
        # "negative" rail). Exactly one of each pair ever fires, so with the pos/neg rows adjacent the
        # firing looked simultaneous. Regroup the DISPLAY rows so all positives (m0, even indices) come
        # first, then all negatives (m1, odd indices) — within each half still ordered by pair index p.
        # This is display-only: the wiring in E uses neuron IDs, and edges/spikes are remapped through
        # id2row (rebuilt from row_ids below), so the network's function, pairing and labels are unchanged.
        mem_grouped = (np.concatenate([self.ids[3][0::2], self.ids[3][1::2]])
                       if len(self.ids[3]) else self.ids[3])
        ordered = [self.ids[0], self.ids[7], self.ids[1], self.ids[2],
                   mem_grouped, self.ids[4], self.ids[5], self.ids[6]]
        row_ids = np.concatenate(ordered)
        self._all_oid = torch.as_tensor(row_ids.astype(np.int32))
        self._n_rows = int(self._all_oid.numel())
        o = np.cumsum([0] + [len(a) for a in ordered])    # o[5]==o[6] (empty tie slot)
        self._bands = [
            {"name": "S1 inputs",       "start": int(o[0]), "end": int(o[1]), "color": "#e6194B"},
            {"name": "completion gate", "start": int(o[1]), "end": int(o[2]), "color": "#42d4f4"},
            {"name": "S1 rails",        "start": int(o[2]), "end": int(o[4]), "color": "#f58231"},
            {"name": "S1 memory",       "start": int(o[4]), "end": int(o[5]), "color": "#3cb44b"},
            {"name": "S2 cells",        "start": int(o[6]), "end": int(o[7]), "color": "#4363d8"},
            {"name": "S3 outputs",      "start": int(o[7]), "end": int(o[8]), "color": "#f032e6"},
        ]
        id2row = {int(nid): r for r, nid in enumerate(row_ids)}
        self._edge_src = np.fromiter((id2row[int(s)] for _, s, _, _ in E), np.uint16, len(E))
        self._edge_tgt = np.fromiter((id2row[int(t)] for _, _, t, _ in E), np.uint16, len(E))
        self._edge_dly = np.fromiter((int(d) for d, _, _, _ in E), np.uint16, len(E))
        self._edge_exc = np.fromiter((1 if w >= 0 else 0 for _, _, _, w in E), np.uint16, len(E))

    def spike_layout(self):
        return {"n_ticks": int(self.n_ticks), "n_rows": int(self._n_rows), "bands": self._bands}

    def read_spikes(self):
        """(row uint16, tick uint16) pairs for the LAST act(), little-endian bytes."""
        torch = self._torch
        from spiky.spnet.spnet import NeuronDataType
        R = self.net.export_neuron_data(self._all_oid, 1, NeuronDataType.Spike, 0, self.n_ticks - 1)
        idx = torch.nonzero(R.reshape(self._n_rows, self.n_ticks).ne(0), as_tuple=False)  # [K,2]=(row,tick)
        if idx.numel() == 0:
            return b""
        a = idx.to(torch.int32).cpu().numpy()
        inter = np.empty(a.shape[0] * 2, dtype="<u2")
        inter[0::2] = a[:, 0].astype("<u2")
        inter[1::2] = a[:, 1].astype("<u2")
        return inter.tobytes()

    def topology_meta(self):
        return {"n_nodes": int(self._n_rows), "n_edges": int(len(self._edge_src)),
                "n_ticks": int(self.n_ticks), "bands": self._bands}

    def topology_payload(self):
        """One-time edge list: (src_row, tgt_row, delay, is_excitatory) uint16 quads, LE bytes."""
        n = len(self._edge_src)
        out = np.empty(n * 4, dtype="<u2")
        out[0::4] = self._edge_src
        out[1::4] = self._edge_tgt
        out[2::4] = self._edge_dly
        out[3::4] = self._edge_exc
        return out.tobytes()

    def _encode(self, xn):
        """shared Gaussian-companded latency code; larger value -> earlier tick"""
        g = np.searchsorted(self.edges, np.asarray(xn).ravel(), side="left")
        return np.clip((self.t_in - 1) - g.reshape(np.shape(xn)),
                       0, self.t_in - 1).astype(np.int64)

    def act(self, obs):
        torch = self._torch
        from spiky.spnet.spnet import NeuronDataType
        x = np.asarray(obs, np.float64).reshape(-1)[:17]
        # bit-identical to the pipeline: (x - mean) / sqrt(var + 1e-8), no extra eps
        ticks = self._encode((x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8))
        NT = self.n_ticks
        va = np.zeros((1, NT, 18), np.float32)
        va[0, ticks, np.arange(17)] = 1e6
        sid = torch.as_tensor(np.ascontiguousarray(self.ids[0], dtype=np.int32)
                              ).reshape(1, 1, -1).expand(1, NT, -1).contiguous()
        self.net.process_ticks(n_ticks_to_process=NT, batch_size=1, n_input_ticks=NT,
                               input_values=torch.as_tensor(va), sparse_input=sid,
                               do_train=False, do_record_voltage=False,
                               do_reset_context=True, _stdp_period=32)
        oid = torch.as_tensor(np.ascontiguousarray(self.ids[6], dtype=np.int32))
        R = self.net.export_neuron_data(oid, 1, NeuronDataType.Spike, 0, NT - 1)
        R = R.reshape(1, 6, NT).ne(0)
        w = torch.arange(NT, 0, -1, dtype=torch.float32)
        T = (NT - (R.float() * w).amax(-1)).numpy()[0].astype(np.float64)
        # Self-timed: the origin moves per observation, so the readout must be referenced to
        # the completion event rather than to absolute time. t_last is free -- we just
        # computed the input ticks. With the absolute form this scored 50-70% within-one-level.
        mu = self.affine[:, 0] * (T - float(ticks.max())) + self.affine[:, 1]
        mu = np.where(T >= NT, -self.clip, mu)          # silence = out-of-band low = -clip
        c = np.clip(mu, -self.clip, self.clip)
        q = np.round((c + self.clip) / self.step) * self.step - self.clip
        return np.clip(q, -self.clip, self.clip).astype(np.float32)

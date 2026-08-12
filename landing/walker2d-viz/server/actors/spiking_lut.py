"""A handcrafted SPIKING network that reproduces the Walker2d LUT teacher.

Three stages, all built from spnet primitives -- no training, every weight and delay derived
analytically from the teacher's own tables:

  STAGE 1  17 latency-coded inputs -> 136 dual-rail winner-take-all comparators -> 192 order
           bits. Anti-leaky rails latch the decision (V=0 is an unstable equilibrium, so the
           winner runs up and the loser runs down and stays silent); a per-pair tie detector
           reproduces the LUT's strict '>' convention (tie -> bit 0); a global gate makes every
           bit emit at one fixed tick, decoupling output timing from input timing.
  STAGE 2  2048 cell neurons, one per LUT cell. Six coincident rails fire exactly one cell per
           table (6 x 0.18 = 1.08 crosses threshold 1.0, five give 0.90 and do not). The
           winning cell emits with a fixed per-(cell, dim) delay encoding its stored value.
  STAGE 3  6 ANTI-LEAKY neurons (cf_1 = +1/tau, so the membrane grows). For a growing membrane
           the threshold-crossing time is exactly -tau*log(sum_t e^{-a_t/tau}) + const, i.e. a
           logsumexp of the arrival times -- which is precisely the teacher's readout
           out[o] = 32*tau*log((1/32) sum_t exp(w_sel[t,o]/tau)). The output spike time is
           therefore AFFINE in the teacher's output and one fitted scale+shift decodes it.

3024 neurons, 26344 synapses, 309 ticks per inference, ~16 ms on CPU. Measured R^2 vs the true
teacher: 0.898 / 0.966 / 0.966 / 0.964 / 0.948 / 0.980 across the six action dims.

NOTE ON THE RUNTIME: this actor needs torch and the `spiky_cuda` extension. That extension has
a CPU-only build (CppExtension, -DNO_CUDA, no nvcc) which setup.py selects automatically when
torch.cuda.is_available() is False -- so it MUST be built inside the GPU-less image, not on a
GPU host, or the CUDA variant gets baked in and fails at import.
"""
import os

import numpy as np

from .base import Actor

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

DE, DI, W_EXC, W_INH = 3, 2, 1.5, -10.0
TAU_M_RAIL, TAU_MEM, W_MEM, W_GATE = 20.0, 1200.0, 0.6, 0.6
W_TIE, W_AND, TAU_M_OUT = 0.6, 0.18, 10.0


def _calib(tau_m, n_euler=2, dt=0.5):
    return 1.0 / np.log((1.0 + dt / tau_m) ** n_euler)


def _pairs(anchor_a, anchor_b):
    """the distinct unordered coordinate pairs, and each (table,bit) slot's pair+orientation"""
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


class SpikingLutActor(Actor):
    name = "Spiking LUT (handcrafted SNN)"

    def __init__(self, action_space):
        super().__init__(action_space)
        import torch
        from spiky.spnet.spnet import LIFNeuronMeta, NeuronMeta, SpikingNet, SynapseMeta
        from spiky.util.synapse_growth import SynapseGrowthEngine

        Q = np.load(os.path.join(_MODELS, "spiking_lut_actor.npz"))
        self.obs_mean = Q["obs_mean"].astype(np.float64)
        self.obs_std = np.sqrt(Q["obs_var"].astype(np.float64) + 1e-8)
        self.affine = Q["affine"].astype(np.float64)
        self.lo, self.hi = float(Q["enc_lo"]), float(Q["enc_hi"])
        self.t_in = int(Q["t_in"])
        self.gate_tick = int(Q["gate_tick"])
        emit = self.gate_tick + 3
        A_, B_ = Q["anchor_a"], Q["anchor_b"]
        W = Q["weights"].astype(np.float64)
        tau = float(Q["tau"])
        pairs, slot = _pairs(A_, B_)
        P = len(pairs)
        tau_eff = _calib(TAU_M_OUT)
        scale = tau_eff / tau

        dly, amps = {}, {}
        for o in range(6):
            Wd = W[:, :, o]
            arr = np.rint(-scale * Wd + np.ceil(scale * Wd.max() + 2))
            dly[o] = arr.astype(np.int64)
            amps[o] = 1.0 / (2.0 * np.exp((arr.max() - arr) / tau_eff).sum())
        dmax = int(max(int(d.max()) for d in dly.values()))
        self.n_ticks = int(emit + dmax + int(Q["settle"]))

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
                 LIFNeuronMeta(neuron_type=4, tau=TAU_M_RAIL, threshold=1.0),
                 LIFNeuronMeta(neuron_type=5, tau=200.0, threshold=1.0),
                 NeuronMeta(neuron_type=6, cf_1=+1.0 / TAU_M_OUT, **anti)]
        counts = [18, 2 * P, 2 * P, 2 * P, P, 2048, 6]
        net = SpikingNet(synapse_metas=smetas, neuron_metas=metas, neuron_counts=counts,
                         initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
        net.to_device("cpu")
        ids = [net.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(7)]
        inp, gate = ids[0][:17], ids[0][17]
        E = []
        for p, (a, b) in enumerate(pairs):
            r0, r1 = ids[1][2 * p], ids[1][2 * p + 1]
            i0, i1 = ids[2][2 * p], ids[2][2 * p + 1]
            m0, m1 = ids[3][2 * p], ids[3][2 * p + 1]
            td = ids[4][p]
            E += [(DE, inp[a], r0, W_EXC), (DE, inp[b], r1, W_EXC),
                  (DI, inp[a], i0, W_EXC), (DI, inp[b], i1, W_EXC),
                  (1, i0, r1, W_INH), (1, i1, r0, W_INH),
                  (1, r0, m0, W_MEM), (1, r1, m1, W_MEM),
                  (1, gate, m0, W_GATE), (1, gate, m1, W_GATE),
                  (1, r0, td, W_TIE), (1, r1, td, W_TIE), (1, td, m0, -W_MEM)]
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
                    E.append((int(dly[o][t, k]), cell, ids[6][o], amps[o]))
        tri = np.array([[d - 1, s, tg] for d, s, tg, _ in E], np.int64)
        wts = np.array([w for *_, w in E], np.float64)
        ge = SynapseGrowthEngine(device="cpu", synapse_group_size=64,
                                 max_groups_in_buffer=max(1 << 19, 4 * len(tri)))
        for i in range(7):
            ge.register_neuron_type(max_synapses=1 << 15, growth_command_list=[])
        for i in range(7):
            tt = torch.tensor(ids[i], dtype=torch.int32)
            ge.add_neurons(neuron_type_index=i, identifiers=tt,
                           coordinates=torch.stack([torch.arange(tt.numel()).float(),
                                                    torch.zeros(tt.numel()),
                                                    torch.full((tt.numel(),), float(i))], 1))
        chunk = ge._grow_explicit(torch.tensor(tri, dtype=torch.int32), 1,
                                  weights=torch.tensor(wts, dtype=torch.float32))
        net.add_connections(chunk, 1)
        chunk.recycle()
        net.compile(shuffle_synapses_random_seed=None)
        self._torch = torch
        self.net, self.ids = net, ids

        # ---- spike-readout support (for the live raster viz) ----
        # Canonical ROW order = populations concatenated, so the client plots row-vs-tick without
        # knowing raw neuron ids. Bands give the stage boundaries in row space.
        self._all_oid = torch.as_tensor(np.concatenate(self.ids).astype(np.int32))
        self._n_rows = int(self._all_oid.numel())
        o = np.cumsum([0] + [len(a) for a in self.ids])   # population offsets in row space
        self._bands = [
            {"name": "S1 inputs+gate", "start": int(o[0]), "end": int(o[1]), "color": "#e6194B"},
            {"name": "S1 rails",       "start": int(o[1]), "end": int(o[3]), "color": "#f58231"},
            {"name": "S1 memory",      "start": int(o[3]), "end": int(o[4]), "color": "#3cb44b"},
            {"name": "S1 tie",         "start": int(o[4]), "end": int(o[5]), "color": "#42d4f4"},
            {"name": "S2 cells",       "start": int(o[5]), "end": int(o[6]), "color": "#4363d8"},
            {"name": "S3 outputs",     "start": int(o[6]), "end": int(o[7]), "color": "#f032e6"},
        ]

    def spike_layout(self):
        """Static metadata for the raster: tick count, row count, and stage bands (row ranges)."""
        return {"n_ticks": int(self.n_ticks), "n_rows": int(self._n_rows), "bands": self._bands}

    def read_spikes(self):
        """(row uint16, tick uint16) pairs for the LAST act(), little-endian bytes (~1 ms).
        Each neuron fires at most once, so this is the complete spike record for one inference."""
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

    def _encode(self, xn):
        u = (xn - self.lo) / max(self.hi - self.lo, 1e-9)
        t = (1.0 - np.clip(u, 0.0, 1.0)) * (self.t_in - 1)
        return np.clip(np.round(t), 0, self.t_in - 1).astype(np.int64)

    def act(self, obs):
        torch = self._torch
        from spiky.spnet.spnet import NeuronDataType
        x = np.asarray(obs, np.float64).reshape(-1)[:17]
        ticks = self._encode((x - self.obs_mean) / (self.obs_std + 1e-12))
        NT = self.n_ticks
        va = np.zeros((1, NT, 18), np.float32)
        va[0, ticks, np.arange(17)] = 1e6
        va[0, self.gate_tick, 17] = 1e6
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
        mu = self.affine[:, 0] * T + self.affine[:, 1]
        return np.clip(mu, -1.0, 1.0).astype(np.float32)

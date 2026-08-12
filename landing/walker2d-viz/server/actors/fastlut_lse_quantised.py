"""FastLutLSEQuantisedActor — the exp19 anchor-pair LUT policy, QUANTISED end to end.

Source run: experiments/walker2d-lut/exp23_qat_obs_quant (a quantization-aware PPO fine-tune
of `deploy_matched/actor_s2.pt`, the artifact this folder's parent ships). 384 updates, full
cosine 3e-4 -> 3e-5, observation normaliser frozen at the parent's statistics.

WHAT MAKES IT DIFFERENT FROM `../fastlut_lse.py`. That actor consumes a continuous
observation and emits a continuous action. This one is trained to run on a QUANTISED
datapath, and is only correct when both quantisers are applied — they were in the loop
during training, so removing either one is running a different model:

  INPUT   the normalised observation is snapped to 128 Gaussian-companded buckets through
          ONE shared monotone map over all 17 coordinates.
  OUTPUT  the action mean is clipped to [-1,1] and snapped to a 22-level uniform grid, so
          the emitted action is ALWAYS exactly one of 22 values and always inside [-1,1].

Pipeline (pure numpy, no torch, no scipy — the server image is deliberately torch-free):
  x       = (obs - obs_mean) / sqrt(obs_var + 1e-8)
  tick    = searchsorted(in_quant_edges, x)                 # 128 companded buckets, shared
  x       = in_quant_dequant[tick]
  bit_i   = 1[ x[a_i] - x[b_i] > 0 ]   over 6 FIXED anchor pairs per table, MSB-first packed
  row_t   = weights[t, addr_t]
  means   = T * tau * log( (1/T) * sum_t exp(row_t / tau) )
  action  = quantise( clip(means, -1, 1) )                  # 22 levels, step 2/21

WHY THE INPUT MAP MUST STAY SHARED. The LUT addresses by comparisons BETWEEN coordinates
(`x[a] > x[b]`). A per-coordinate scale or offset would change that comparison for every pair
spanning two different maps, and the address bit would stop meaning "coordinate a exceeds
coordinate b". One shared, strictly monotone map is the only admissible choice.

TIES. Because the map is shared and monotone, the only comparison quantisation can change is
one it collapses into a single bucket. Two coordinates in the same bucket dequantise to the
SAME value, so `d > 0` is False and the bit is 0 — deterministically. (The training-time
straight-through estimator was `x + (xq - x).detach()`, which is not value-exact in float32
and broke such ties on float noise instead; this actor dequantises exactly. Measured over
100k real observations the two disagree on 0.0018% of address rows / 0.0400% of samples, all
of it in the saturated end buckets.)
"""
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class FastLutLSEQuantisedActor(Actor):
    # A NEW name — this is an additional actor, it does not replace `fastlut_lse (exp19)`.
    name = "fastlut_lse (exp19, quantised)"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "walker2d_fastlut_lse_exp19_quantised.npz"))
        self.W = Q["weights"].astype(np.float64)             # (T, 2**NAP, 6) LUT tables
        self.a_idx = Q["anchor_a"].astype(np.int64)          # (T, NAP) fixed anchor pairs
        self.b_idx = Q["anchor_b"].astype(np.int64)          # (T, NAP)
        self.tau = float(Q["tau_actor"])                     # learned readout temperature
        self.obs_mean = Q["obs_mean"].astype(np.float64)     # (17,) frozen training stats
        self.obs_var = Q["obs_var"].astype(np.float64)       # (17,)
        # input quantiser, baked as arrays so no erf/erfinv (i.e. no scipy) is needed here
        self.in_edges = Q["in_quant_edges"].astype(np.float64)      # (127,) boundaries
        self.in_dequant = Q["in_quant_dequant"].astype(np.float64)  # (128,) bucket values
        self.n_in_ticks = self.in_dequant.shape[0]
        # output quantiser
        self.out_levels = int(Q["out_quant_levels"])
        self.out_clip = float(Q["out_quant_clip"])
        self.out_step = 2.0 * self.out_clip / (self.out_levels - 1)

        self.T, _, self.n_act = self.W.shape
        self.nap = self.a_idx.shape[1]
        self.pow2 = (1 << np.arange(self.nap - 1, -1, -1))   # MSB-first bit packing
        self.n_obs = self.obs_mean.shape[0]
        self._tables = np.arange(self.T)
        self._logT = np.log(self.T)

    def act(self, obs):
        x = np.asarray(obs, np.float64).reshape(-1)[: self.n_obs]
        x = (x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        # --- INPUT quantiser: one shared 128-bucket Gaussian-companded map ---------------
        tick = np.searchsorted(self.in_edges, x, side="left")
        x = self.in_dequant[np.clip(tick, 0, self.n_in_ticks - 1)]
        # --- the LUT ---------------------------------------------------------------------
        d = x[self.a_idx] - x[self.b_idx]                        # (T, NAP)
        addr = ((d > 0).astype(np.int64) * self.pow2).sum(-1)    # (T,) row index per table
        sel = self.W[self._tables, addr]                         # (T, 6) selected rows
        # T * tau * log( (1/T) sum_t exp(w_t / tau) ), max-subtracted so exp cannot overflow.
        z = sel / self.tau
        m = z.max(axis=0)
        lse = m + np.log(np.exp(z - m).sum(axis=0))
        means = self.T * self.tau * (lse - self._logT)           # (6,) action means
        # --- OUTPUT quantiser: clip, then snap to the 22-level uniform grid ---------------
        c = np.clip(means, -self.out_clip, self.out_clip)
        q = np.round((c + self.out_clip) / self.out_step) * self.out_step - self.out_clip
        return np.clip(q, -self.out_clip, self.out_clip).astype(np.float32)

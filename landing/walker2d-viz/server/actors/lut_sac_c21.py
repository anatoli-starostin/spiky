"""Walker2d LUT-SAC c21 — the best Walker2d LUT actor (lut_sac_c21_seed4_20k, seed 4, 20k iters).

Full-precision checkpoint pulled from gpustar (/tmp/verify_best/lut_sac_c21_seed4_20k_actor.npz).
28,032 params, n_heads=1, tables_per_head=32, 6 hyperplanes/table (nap6), hyperplane addressing.
CPU reference: 5658 ± 619 return over 100 Walker2d-v5 episodes (its cpueval.json).

Pipeline (deterministic SAC eval, pure numpy — no torch/spiky needed):
  x = (obs - obs_mean)/(obs_std+1e-6)                     # standardize the 17-dim obs (dataset stats)
  addr_t = pack( <w_t, x> + b_t > 0 ) over 6 hyperplanes  # per-table sign-tests, MSB-first -> 0..63
  means = sum_t weights[t, addr_t, :6]                    # 6-dim pre-tanh action means (first 6 of 12)
  action = tanh(means)
"""
import json
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class LUTSACc21Actor(Actor):
    name = "Walker2d LUT-SAC c21"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "lut_sac_c21_seed4_20k_actor.npz"), allow_pickle=True)
        self.W = Q["w"].astype(np.float64)              # (32,6,17) hyperplane normals
        self.B = Q["b"].astype(np.float64)              # (32,6) hyperplane biases
        self.TAB = Q["weights"].astype(np.float64)[:, :, :6]  # (32,64,6) LUT action means (first 6 of 12)
        self.NT, self.NAP, self.N = self.W.shape        # 32, 6, 17
        self.pow2 = (1 << np.arange(self.NAP - 1, -1, -1))   # [32,16,8,4,2,1] MSB-first
        S = json.load(open(os.path.join(_MODELS, "walker_dataset_stats.json")))
        self.obs_mean = np.asarray(S["obs_mean"], np.float64)
        self.obs_std = np.asarray(S["obs_std"], np.float64)

    def act(self, obs):
        obs = np.asarray(obs, np.float64).reshape(-1)
        x = (obs[: self.N] - self.obs_mean) / (self.obs_std + 1e-6)   # standardized 17-dim obs
        proj = np.einsum("tki,i->tk", self.W, x) + self.B            # (32,6) hyperplane pre-activations
        addr = (proj > 0).astype(int) @ self.pow2                    # (32,) 6-bit table addresses
        means = np.zeros(6)
        for t in range(self.NT):
            means += self.TAB[t, addr[t]]                            # sum selected LUT rows -> action means
        return np.tanh(means).astype(np.float32)                     # deterministic action in [-1,1]

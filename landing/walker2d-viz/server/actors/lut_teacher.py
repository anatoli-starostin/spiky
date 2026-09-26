"""LUTTeacherActor — the trained int4 Walker2d LUT-SAC actor (our distillation teacher).

Loads server/models/walker2d_lut_actor_int4.npz (source actor: lut_sac_c21_seed4_20k). Pipeline:
  x = (obs - obs_mean)/(obs_std+1e-6)                         # standardize the 17-dim Walker2d obs (dataset stats)
  addr_t = pack( <w_t, x> + b_t > 0 )  over 6 hyperplanes/table  # int4-dequant hyperplane sign-tests, MSB-first
  means = sum_t table[t, addr_t]  (6-dim pre-tanh action means)
  action = tanh(means)                                        # deterministic SAC policy (clip to [-1,1])
No torch needed — pure numpy.
"""
import json
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class LUTTeacherActor(Actor):
    name = "lut_teacher"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "walker2d_lut_actor_int4.npz"), allow_pickle=True)
        self.Wd = Q["w_q"].astype(np.float64) * Q["w_scale"].astype(np.float64)[:, None, None]  # (32,6,17)
        self.Bd = Q["b_q"].astype(np.float64) * Q["b_scale"].astype(np.float64)[:, None]          # (32,6)
        self.TAB = (Q["weights_q"].astype(np.float64) * Q["weights_scale"].astype(np.float64)[:, None, None])[:, :, :6]  # (32,64,6) mean-only
        self.NT, self.NAP, self.N = self.Wd.shape                                                 # 32, 6, 17
        self.pow2 = (1 << np.arange(self.NAP - 1, -1, -1))                                        # [32,16,8,4,2,1]
        S = json.load(open(os.path.join(_MODELS, "walker_dataset_stats.json")))
        self.obs_mean = np.asarray(S["obs_mean"], np.float64)
        self.obs_std = np.asarray(S["obs_std"], np.float64)

    def act(self, obs):
        obs = np.asarray(obs, np.float64).reshape(-1)
        x = (obs[: self.N] - self.obs_mean) / (self.obs_std + 1e-6)          # standardized 17-dim obs
        proj = np.einsum("tki,i->tk", self.Wd, x) + self.Bd                 # (32,6) hyperplane pre-activations
        addr = (proj > 0).astype(int) @ self.pow2                           # (32,) 6-bit table addresses
        means = np.zeros(6)
        for t in range(self.NT):
            means += self.TAB[t, addr[t]]                                    # sum selected LUT entries -> action means
        return np.tanh(means).astype(np.float32)                            # deterministic action in [-1,1]

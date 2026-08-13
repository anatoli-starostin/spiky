"""Walker2d SAC baseline — the conventional MLP continuous-control policy (the LUT distillation teacher).

Source: spiky repo `experiments/walker2d_lut/exp_c01_sac_baseline` (commit d8a2cc5, issue #75, seed 0).
Stable-Baselines3 SAC, MlpPolicy net_arch [256,256] ReLU, trained 1M steps on Walker2d-v5.
Reference: deterministic 100-episode eval = 5273.4 ± 33.9 (its config.json / summary.json).

Actor weights exported from run_seed0/sac_walker2d_final.zip on gpustar (SAC.load -> actor.state_dict).
SB3 SAC uses NO obs normalization (raw Box obs), so the forward pass is a plain 2-hidden-layer MLP:
  h1 = relu(W0 @ obs + b0)     # (256,17)
  h2 = relu(W2 @ h1 + b2)      # (256,256)
  mu = Wmu @ h2 + b_mu         # (6,256)
  action = tanh(mu)            # SB3 SquashedDiagGaussian deterministic action
Pure numpy — no torch/SB3 needed.
"""
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class SACBaselineActor(Actor):
    name = "Walker2d SAC baseline"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "sac_baseline_actor.npz"))
        self.W0 = Q["latent_pi_0_weight"].astype(np.float64)   # (256,17)
        self.b0 = Q["latent_pi_0_bias"].astype(np.float64)     # (256,)
        self.W2 = Q["latent_pi_2_weight"].astype(np.float64)   # (256,256)
        self.b2 = Q["latent_pi_2_bias"].astype(np.float64)     # (256,)
        self.Wmu = Q["mu_weight"].astype(np.float64)           # (6,256)
        self.bmu = Q["mu_bias"].astype(np.float64)             # (6,)
        self.N = self.W0.shape[1]                              # 17

    def act(self, obs):
        x = np.asarray(obs, np.float64).reshape(-1)[: self.N]  # raw obs (SB3 SAC does not normalize)
        h1 = np.maximum(self.W0 @ x + self.b0, 0.0)            # Linear(17->256) + ReLU
        h2 = np.maximum(self.W2 @ h1 + self.b2, 0.0)           # Linear(256->256) + ReLU
        mu = self.Wmu @ h2 + self.bmu                          # Linear(256->6) action means
        return np.tanh(mu).astype(np.float32)                  # deterministic squashed action in [-1,1]

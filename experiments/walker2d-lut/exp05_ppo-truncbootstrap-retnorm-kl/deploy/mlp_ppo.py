"""MLPPPOActor — the plain-MLP PPO Walker2d baseline, the first step of the LUT story.

Source run: experiments/walker2d-lut/exp05_ppo-truncbootstrap-retnorm-kl, deploy-matched
retrain (PPO, 8192 envs, 768 updates, `--obs-clip-vel 10.0 --solver-iters 100 --ls-iters 50`).
This is the reference arm every LUT experiment is measured against: same bench7 recipe, same
env, an ordinary [256,256] Tanh policy instead of a lookup table.

Pipeline (pure numpy, no torch — the server image is deliberately torch-free):
  x       = (obs - obs_mean) / sqrt(obs_var + 1e-8)        # training-time obs normalisation
  h       = tanh(W0 x + b0);  h = tanh(W1 h + b1)          # two [256,256] Tanh trunks
  means   = W2 h + b2                                      # linear readout, 6 action means
  action  = clip(means, -1, 1)

Two notes, both deliberate and both shared with `fastlut_lse.py`:
  * the action is CLIPPED, not tanh-squashed. This is a PPO Gaussian policy and the training
    env applied `action.clamp(-1, 1)`; tanh would be a different function.
  * only the ACTOR ships. The critic plays no part at inference — the demo needs actions.

The observation this expects is stock gymnasium `Walker2d-v5`, which clips qvel to [-10, 10].
The weights were trained under that same clipping (`--obs-clip-vel 10.0`), so the deployed
observation distribution is the one the normalisation statistics were fitted on. Training
without that flag is what broke exp19's first artifact — see ../deploy/README.md.
"""
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class MLPPPOActor(Actor):
    # DO NOT RENAME without updating the server: `Sim.set_actor` fails SILENTLY on an
    # unknown name (`if name in self.registry:`, no else) and a session's default is
    # `random`, so a mismatch presents exactly as "the walker falls over immediately".
    name = "mlp_ppo (exp05)"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "walker2d_mlp_ppo_exp05.npz"))
        # Layers are stored transposed for numpy (torch Linear holds (out, in)).
        self.W = [Q[f"pi_w{i}"].astype(np.float64) for i in range(3)]
        self.b = [Q[f"pi_b{i}"].astype(np.float64) for i in range(3)]
        self.obs_mean = Q["obs_mean"].astype(np.float64)    # (17,) training obs statistics
        self.obs_var = Q["obs_var"].astype(np.float64)      # (17,)
        self.n_obs = self.obs_mean.shape[0]
        self.n_act = self.b[-1].shape[0]

    def act(self, obs):
        x = np.asarray(obs, np.float64).reshape(-1)[: self.n_obs]
        x = (x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        h = np.tanh(x @ self.W[0] + self.b[0])
        h = np.tanh(h @ self.W[1] + self.b[1])
        means = h @ self.W[2] + self.b[2]
        return np.clip(means, -1.0, 1.0).astype(np.float32)

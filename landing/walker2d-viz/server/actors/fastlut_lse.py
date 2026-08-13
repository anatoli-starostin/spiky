"""FastLutLSEActor — the anchor-pair LUT Walker2d policy with a log-sum-exp table readout.

Source run: experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32 (PPO, 8192 envs, 768
updates; arm mean 5553 +- 224 over 3 seeds). Trained with a PLAIN MLP critic whose readout
was also exponential — but the critic plays no part at inference, so only the actor ships.

Pipeline (pure numpy, no torch — the server image is deliberately torch-free):
  x       = (obs - obs_mean) / sqrt(obs_var + 1e-8)        # training-time obs normalisation
  bit_i   = 1[ x[a_i] - x[b_i] > 0 ]  over 6 FIXED anchor pairs per table, MSB-first packed
  row_t   = weights[t, addr_t]                             # one 6-dim row per table (32 tables)
  means   = T * tau * log( (1/T) * sum_t exp(row_t / tau) ) # sum-scaled log-sum-exp readout
  action  = clip(means, -1, 1)

Two things differ from the other LUT actor here (`lut_teacher.py`) and both are deliberate:
  * the table reduction is a temperature-tau log-sum-exp, not a plain sum. tau is learned
    (0.05 -> ~0.085); tau->inf would recover the plain sum exactly, tau->0 gives T*max.
  * the action is CLIPPED, not tanh-squashed. This policy is PPO with a Gaussian head and
    the training env applied `action.clamp(-1, 1)`; tanh would be a different function.
"""
import os

import numpy as np

from .base import Actor

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "..", "models")


class FastLutLSEActor(Actor):
    # DO NOT RENAME. The server's default actor is configured to this exact string, so
    # changing it would leave sessions falling back to `random`.
    #
    # The name was investigated as a suspect when the first artifact failed on the server —
    # `Sim.set_actor` fails SILENTLY on an unknown name (`if name in self.registry:`, no
    # else) and a session's default is `random`, so any name mismatch presents exactly as
    # "the walker falls over immediately". It was RULED OUT: two shipped actors already use
    # spaces ("Walker2d LUT-SAC c21", "Walker2d SAC baseline"), and the client assigns the
    # option's value as a DOM property (`o.value = o.textContent = name`), so the string
    # round-trips unescaped. Spaces and parentheses are safe here.
    name = "fastlut_lse (exp19)"

    def __init__(self, action_space):
        super().__init__(action_space)
        Q = np.load(os.path.join(_MODELS, "walker2d_fastlut_lse_exp19.npz"))
        self.W = Q["weights"].astype(np.float64)            # (T, 2**NAP, 6) LUT tables
        self.a_idx = Q["anchor_a"].astype(np.int64)         # (T, NAP) fixed anchor pairs
        self.b_idx = Q["anchor_b"].astype(np.int64)         # (T, NAP)
        self.tau = float(Q["tau_actor"])                    # learned readout temperature
        self.obs_mean = Q["obs_mean"].astype(np.float64)    # (17,) training obs statistics
        self.obs_var = Q["obs_var"].astype(np.float64)      # (17,)
        self.T, _, self.n_act = self.W.shape
        self.nap = self.a_idx.shape[1]
        self.pow2 = (1 << np.arange(self.nap - 1, -1, -1))  # MSB-first bit packing
        self.n_obs = self.obs_mean.shape[0]
        self._tables = np.arange(self.T)
        self._logT = np.log(self.T)

    def act(self, obs):
        x = np.asarray(obs, np.float64).reshape(-1)[: self.n_obs]
        x = (x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        d = x[self.a_idx] - x[self.b_idx]                        # (T, NAP)
        addr = ((d > 0).astype(np.int64) * self.pow2).sum(-1)    # (T,) row index per table
        sel = self.W[self._tables, addr]                         # (T, 6) selected rows
        # T * tau * log( (1/T) sum_t exp(w_t / tau) ), max-subtracted so exp cannot overflow.
        z = sel / self.tau
        m = z.max(axis=0)
        lse = m + np.log(np.exp(z - m).sum(axis=0))
        means = self.T * self.tau * (lse - self._logT)           # (6,) action means
        return np.clip(means, -1.0, 1.0).astype(np.float32)

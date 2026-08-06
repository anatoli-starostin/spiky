"""Swappable Actor/Critic interface + registry.

A new architecture (MLP today; hyperplane-LUT / LIF-detector tomorrow) drops in by
implementing ONE method: forward(obs) -> (mean, value). The base class supplies the
Gaussian policy head (state-independent log_std) and the PPO act/evaluate helpers, so
every variant is interchangeable behind the same signature.

    from models import REGISTRY
    ac = REGISTRY["mlp"](obs_dim, act_dim, hidden=(256,256)).to("cuda")
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


class BaseActorCritic(nn.Module, ABC):
    """Contract for a swappable policy/value module.

    Subclass responsibility: forward(obs)->(mean (B,act_dim), value (B,)).
    Provided here: Gaussian head, act() (sample+logp+value), evaluate() (logp+entropy+value).
    """

    def __init__(self, obs_dim, act_dim, log_std_init=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    @abstractmethod
    def forward(self, obs):
        """Return (mean, value). mean:(B,act_dim)  value:(B,)."""

    def _dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())

    @torch.no_grad()
    def act(self, obs):
        mean, value = self(obs)
        dist = self._dist(mean)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), value

    def evaluate(self, obs, act):
        mean, value = self(obs)
        dist = self._dist(mean)
        return dist.log_prob(act).sum(-1), dist.entropy().sum(-1), value


def _mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _ortho(module, gain):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)
    return module


@register("mlp")
class MLPActorCritic(BaseActorCritic):
    """Reference: separate [256,256] Tanh MLP trunks for policy and value."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        self.pi = _ortho(_mlp([obs_dim, *hidden, act_dim]), gain=1.0)
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)
        # small final policy layer -> near-zero initial mean (standard PPO trick)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)

    def forward(self, obs):
        return self.pi(obs), self.vf(obs).squeeze(-1)


class QCritic(nn.Module):
    """SAC state-action value Q(obs, act) -> scalar. Standard MLP, independent of the
    actor architecture: the swappable/exotic part is the ACTOR (from REGISTRY); the
    critic is a plain value estimator, so LUT/LIF actors compose without a bespoke Q."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = _ortho(_mlp([obs_dim + act_dim, *hidden, 1]), gain=1.0)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


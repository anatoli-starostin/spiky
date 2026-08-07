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


@register("hyperlut")
class HyperLUTActorCritic(BaseActorCritic):
    """Hyperplane-LUT policy (the int4 LUT-SAC actor lineage) + a standard MLP critic.

    Policy: `tables_per_head` LUTs, each addressed by `nap` affine sign-tests
    1[W_k·x + b_k > 0] (MSB-first bit packing -> one of 2**nap rows), rows summed across
    tables -> action mean. Only the POLICY is a LUT; the value function stays a [256,256]
    Tanh MLP (keeps PPO value estimation stable). log_std is the base's state-independent
    parameter.

    Differentiable via the decoupled straight-through estimator (as in the LIF-detector
    work): the forward VALUE is the hard single-row lookup; the TABLE gradient follows the
    hard address (selected rows only), while the HYPERPLANE-WEIGHT gradient follows a soft
    full-2**nap product over the per-bit sigmoids (table detached on that path so the soft
    blend can't smear the weight gradient).
    """

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6, hidden=(256, 256),
                 table_std=0.001, hyp_std=0.1, temp=1.0, log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        self.T = tables_per_head
        self.nap = nap
        self.n_cells = 2 ** nap
        self.temp = temp
        # affine sign-test planes: W (T, nap, obs_dim) std 0.1 ; bias 0
        self.W_hyp = nn.Parameter(torch.randn(self.T, nap, obs_dim) * hyp_std)
        self.B_hyp = nn.Parameter(torch.zeros(self.T, nap))
        # LUT value tables: (T, 2**nap, act_dim) std 0.001 -> near-zero initial action
        self.table = nn.Parameter(torch.randn(self.T, self.n_cells, act_dim) * table_std)
        # MSB-first bit patterns per cell + packing weights (buffers)
        cells = torch.arange(self.n_cells)
        order = torch.arange(nap - 1, -1, -1)
        self.register_buffer("cell_bits", ((cells[:, None] >> order[None, :]) & 1).float())  # (C, nap)
        self.register_buffer("pow2", (2 ** order).float())                                    # (nap,)
        # standard MLP critic
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)

    def forward(self, obs):
        z = torch.einsum('bi,tki->btk', obs, self.W_hyp) + self.B_hyp     # (B, T, nap)
        hard = (z > 0).float()
        # hard address -> gather one row per table (table grad = selected rows only)
        addr = (hard * self.pow2).sum(-1).long()                           # (B, T)
        B = obs.shape[0]
        tidx = torch.arange(self.T, device=obs.device).expand(B, self.T)
        y_hard = self.table[tidx, addr].sum(1)                             # (B, act)
        # soft full-2**nap address for the hyperplane-weight gradient (table detached)
        p = torch.sigmoid(z / self.temp)                                   # (B, T, nap)
        eps = 1e-6
        lp1 = torch.log(p + eps); lp0 = torch.log(1.0 - p + eps)
        cb = self.cell_bits                                                # (C, nap)
        log_prow = (torch.einsum('ck,btk->btc', cb, lp1)
                    + torch.einsum('ck,btk->btc', 1.0 - cb, lp0))          # (B, T, C)
        prow_soft = log_prow.exp()
        y_addr = torch.einsum('btc,tco->bo', prow_soft, self.table.detach())
        mean = y_hard + y_addr - y_addr.detach()                          # forward == hard; W-grad injected
        value = self.vf(obs).squeeze(-1)
        return mean, value


class HyperLUTHead(nn.Module):
    """Reusable hyperplane-LUT block: `tables_per_head` LUTs, each addressed by `nap`
    affine sign-tests (MSB-first pack -> 2**nap rows), rows summed -> (B, n_out).
    Decoupled straight-through: hard single-row lookup forward; table grad on the hard
    rows only; hyperplane-weight grad via the soft full-2**nap product (table detached)."""

    def __init__(self, obs_dim, n_out, tables_per_head=32, nap=6,
                 table_std=0.001, hyp_std=0.1, temp=1.0):
        super().__init__()
        self.T = tables_per_head
        self.nap = nap
        self.n_cells = 2 ** nap
        self.temp = temp
        self.W_hyp = nn.Parameter(torch.randn(self.T, nap, obs_dim) * hyp_std)
        self.B_hyp = nn.Parameter(torch.zeros(self.T, nap))
        self.table = nn.Parameter(torch.randn(self.T, self.n_cells, n_out) * table_std)
        cells = torch.arange(self.n_cells)
        order = torch.arange(nap - 1, -1, -1)
        self.register_buffer("cell_bits", ((cells[:, None] >> order[None, :]) & 1).float())
        self.register_buffer("pow2", (2 ** order).float())

    def forward(self, obs):
        z = torch.einsum('bi,tki->btk', obs, self.W_hyp) + self.B_hyp
        hard = (z > 0).float()
        addr = (hard * self.pow2).sum(-1).long()
        B = obs.shape[0]
        tidx = torch.arange(self.T, device=obs.device).expand(B, self.T)
        y_hard = self.table[tidx, addr].sum(1)
        p = torch.sigmoid(z / self.temp)
        eps = 1e-6
        lp1 = torch.log(p + eps); lp0 = torch.log(1.0 - p + eps)
        cb = self.cell_bits
        log_prow = (torch.einsum('ck,btk->btc', cb, lp1)
                    + torch.einsum('ck,btk->btc', 1.0 - cb, lp0))
        prow_soft = log_prow.exp()
        y_addr = torch.einsum('btc,tco->bo', prow_soft, self.table.detach())
        return y_hard + y_addr - y_addr.detach()


@register("hyperlut2")
class HyperLUT2ActorCritic(BaseActorCritic):
    """BOTH actor and critic are hyperplane-LUTs (same structure/inits). The critic LUT
    outputs a scalar V(s) (table [T, 2**nap, 1] summed over tables)."""

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6,
                 table_std=0.001, hyp_std=0.1, temp=1.0, log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        self.actor_lut = HyperLUTHead(obs_dim, act_dim, tables_per_head, nap,
                                      table_std, hyp_std, temp)
        self.critic_lut = HyperLUTHead(obs_dim, 1, tables_per_head, nap,
                                       table_std, hyp_std, temp)

    def forward(self, obs):
        mean = self.actor_lut(obs)
        value = self.critic_lut(obs).squeeze(-1)
        return mean, value


@register("hyperlut2_t64")
class HyperLUT2T64ActorCritic(HyperLUT2ActorCritic):
    """Same fully-LUT actor+critic as `hyperlut2`, but tables_per_head=64 for both heads."""

    def __init__(self, obs_dim, act_dim, **kw):
        kw.setdefault("tables_per_head", 64)
        super().__init__(obs_dim, act_dim, **kw)


@register("hyperlut_t64")
class HyperLUTT64ActorCritic(HyperLUTActorCritic):
    """Same as `hyperlut` (LUT actor + [256,256] Tanh MLP critic), but actor
    tables_per_head=64 instead of 32."""

    def __init__(self, obs_dim, act_dim, **kw):
        kw.setdefault("tables_per_head", 64)
        super().__init__(obs_dim, act_dim, **kw)


@register("fastlut")
class FastLUTActorCritic(BaseActorCritic):
    """FastMultiHeadLut (anchor-pair) actor — REUSES the existing
    `spiky.lutorch.fast_multi_head_lut.FastMultiHeadLut` — + [256,256] Tanh MLP critic.
    Anchor pairs are FIXED (address bit = sign(x[a]-x[b]), balanced-sampled pairs of input
    coords); only the LUT table (`weights`) trains. forward_mode='hard' (discrete single-row
    lookup + soft-surrogate backward, same discipline as the hyperlut* arches), fp32."""

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6, hidden=(256, 256),
                 initial_weights_noise=0.001, log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        self.actor_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise)
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)     # (B, 1, act_dim) -> (B, act_dim)
        value = self.vf(obs).squeeze(-1)
        return mean, value


@register("fastlut2")
class FastLUT2ActorCritic(BaseActorCritic):
    """BOTH actor and critic are FastMultiHeadLut (anchor-pair) heads — same fixed-anchor
    structure/inits as `fastlut`; the critic LUT outputs a scalar V(s). This is the
    anchor-pair analogue of `hyperlut2` (fully-LUT actor+critic): address bit =
    sign(x[a]-x[b]) over balanced-sampled coordinate pairs, only the LUT tables (`weights`)
    train, forward_mode='hard', fp32. Actor and critic get independent seed-reproducible
    random anchors (torch.manual_seed(--seed) is set before construction)."""

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6,
                 initial_weights_noise=0.001, log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        self.actor_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise)
        self.critic_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=1, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)          # (B, 1, act_dim) -> (B, act_dim)
        value = self.critic_lut(obs).reshape(obs.shape[0])  # (B, 1, 1) -> (B,)
        return mean, value


"""Swappable Actor/Critic interface + registry.

A new architecture (MLP today; hyperplane-LUT / LIF-detector tomorrow) drops in by
implementing ONE method: forward(obs) -> (mean, value). The base class supplies the
Gaussian policy head (state-independent log_std) and the PPO act/evaluate helpers, so
every variant is interchangeable behind the same signature.

    from models import REGISTRY
    ac = REGISTRY["mlp"](obs_dim, act_dim, hidden=(256,256)).to("cuda")
"""
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


@register("fastlut_exp")
class FastLUTExpActorCritic(FastLUTActorCritic):
    """exp16: `fastlut` with a trainable EXPONENTIAL OUTPUT TRANSFORM on the action mean.

        mean  ->  c + exp(mean / t)

    `c` is a free trainable scalar (any sign); `t` is a trainable scalar constrained
    positive via softplus. Both are shared across action dimensions. Everything else —
    the anchor-pair LUT actor, the [256,256] Tanh MLP critic, the Gaussian head — is
    exp10's, unchanged.

    INITIALIZATION (c = -1, t = 1) is chosen so the transform starts as a first-order
    match to the identity, i.e. to exp10 itself: the LUT is initialized near zero
    (initial_weights_noise=1e-3), so at init mean ~ 0 and

        c + exp(0/t) = -1 + 1 = 0          (same near-zero initial action mean as exp10)
        d/dy [c + exp(y/t)] |_{y=0} = 1/t = 1   (same unit slope)

    so exp16 begins behaviourally identical to exp10 and the experiment measures what the
    transform does to LEARNING rather than to the starting point. A positive-constrained c
    could not do this: it would force mean >= c > 0 and saturate every actuator against the
    env's clamp(-1,1) from step one.

    The exponent is clamped at +20 purely as an fp32 overflow guard (exp(20) ~ 4.9e8, nine
    orders of magnitude beyond the action clamp). It cannot bind before the policy is
    already degenerate, but it stops an inf mean from turning the Gaussian log-prob into
    NaN and silently killing a seed.
    """

    EXP_CLAMP = 20.0

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6, hidden=(256, 256),
                 initial_weights_noise=0.001, log_std_init=0.0,
                 c_init=-1.0, t_init=1.0):
        super().__init__(obs_dim, act_dim, tables_per_head=tables_per_head, nap=nap,
                         hidden=hidden, initial_weights_noise=initial_weights_noise,
                         log_std_init=log_std_init)
        # t = softplus(t_raw) > 0 ; invert softplus so t starts exactly at t_init
        t_raw0 = math.log(math.expm1(float(t_init)))
        self.c = nn.Parameter(torch.tensor(float(c_init)))
        self.t_raw = nn.Parameter(torch.tensor(float(t_raw0)))

    @property
    def t(self):
        return F.softplus(self.t_raw)

    def forward(self, obs):
        mean, value = super().forward(obs)
        z = torch.clamp(mean / self.t, max=self.EXP_CLAMP)
        return self.c + torch.exp(z), value

    def extra_log(self):
        """Scalars ppo.py folds into each history row (see ppo.py's logging block)."""
        return dict(c=float(self.c.detach()), t=float(self.t.detach()))


@register("fastlut_lse")
class FastLUTLogSumExpActorCritic(BaseActorCritic):
    """exp17: `fastlut` with the table reduction replaced by a temperature-tau LOG-SUM-EXP.

    The anchor-pair actor normally sums the selected row across a head's tables. With
    `exp_outputs=True` on FastMultiHeadLut it instead computes, per output dimension,

        out = tau * log( sum_tables exp( w_selected / tau ) )

    with `tau` the single new trainable scalar (softplus-constrained > 0). Critic, Gaussian
    head and every PPO hyper-parameter are exp10's, unchanged. The rows selected are
    bit-identical to exp10's (verified: 16,384/16,384 indices equal); only how they are
    combined changes.

    WEIGHTS-AS-LOGARITHMS INIT (`exp_outputs_init="logspace"`, tau_init=0.05). Under this
    readout the weights sit inside exp(), so they are logarithms, not additive
    contributions, and the additive default is wrong for it. With w ~ U(-1e-3, 1e-3) every
    exp(w/tau) ~ 1, so the sum collapses to tables_per_head and

        out ~ tau * log(32) = 0.347 constant,  out std 32x too small,  d out/d w = 1/32
        uniform across all 32 tables

    -- a saturated near-constant head with tiny uniform gradients. Measured exactly:
    out mean +0.3466, std 1.02e-4 against exp10's 3.28e-3, effective tables 32.0/32.
    That is the long-warmup cause (attempt 1 reached only ~400 return by update 270,
    where exp10 is near 5000).

    The corrected init (in FastMultiHeadLut, flag-gated) fixes both terms:
      * SPREAD  sigma_log = initial_weights_noise * tables_per_head. Log-sum-exp AVERAGES
        (std ~ sigma/sqrt(T)) where the plain sum ACCUMULATES (sigma*sqrt(T)), so matching
        exp10's output spread needs a per-entry spread T times LARGER, not equal.
      * CENTRE  mu = -tau*log(T) minus the Jensen gap of the spread, computed by
        fixed-seed Monte Carlo, so the head starts at output ~ 0 rather than tau*log(T).
    At tau=0.05 this gives out mean ~0 and out std 0.98x exp10's -- a near-identity start,
    the same property exp16's c=-1, t=1 init was chosen for.

    tau_init = 0.05 balances the two things tau controls: it matches exp10's output spread
    to within 2% (tau=0.1 gives 0.99x, indistinguishable) while keeping the softmax over
    tables mildly structured (max weight 0.069 = 2.2x uniform, ~30 of 32 tables effective)
    rather than perfectly flat. tau is trainable, so this sets the starting regime only.

    What NO initialisation can fix: sum_t d(out)/d(w_t) = 1 for log-sum-exp against T = 32
    for the plain sum, so the same weight step moves the action ~T times less. That is
    averaging-vs-summing, not initialisation, and it bounds how much of the warmup gap an
    init fix can close.
    """

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6, hidden=(256, 256),
                 initial_weights_noise=0.001, log_std_init=0.0, tau_init=0.05,
                 exp_outputs_init="logspace", exp_outputs_scale="mean"):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        self.actor_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise,
            exp_outputs=True, exp_outputs_tau_init=tau_init,
            exp_outputs_init=exp_outputs_init, exp_outputs_scale=exp_outputs_scale)
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)
        value = self.vf(obs).squeeze(-1)
        return mean, value

    def extra_log(self):
        """Scalars ppo.py folds into each history row (see ppo.py's logging block)."""
        return dict(tau=float(self.actor_lut.exp_outputs_tau.detach()))


@register("fastlut_lse_sum")
class FastLUTLogSumExpSumScaledActorCritic(FastLUTLogSumExpActorCritic):
    """exp17c: the SUM-SCALED log-sum-exp readout — a smooth generalisation of exp10's sum.

        out = T * tau * log( (1/T) * sum_t exp(w_t / tau) )

    Motivation. The plain readout `tau * log(sum_t exp(w_t/tau))` is a smooth MEAN/MAX: its
    gradient sums to 1 over tables against T = 32 for the plain sum, so the same weight step
    moves the action ~32x less and the actor cannot escape its initial plateau. That is a
    property of the aggregation, not of the initialisation, which is why fixing the init
    (attempt 2) corrected the starting statistics without fixing the learning.

    Multiplying by T and subtracting tau*log(T) turns it into a smooth generalisation of the
    SUM instead:
        tau -> inf  =>  T * mean(w) = sum_t w_t   -- exactly exp10's readout
        tau -> 0    =>  T * max(w)
    with d(out)/d(w) summing to T, matching the additive path. Because it reduces to the
    sum, the ADDITIVE init is the correct one here (weights are back to being additive
    contributions at large tau), and at tau=0.05 with w ~ U(+-1e-3) the head starts
    numerically indistinguishable from exp10's.

    Still one trainable scalar (tau) and still the same bit-identical row selection.
    """

    def __init__(self, obs_dim, act_dim, **kw):
        kw.setdefault("exp_outputs_scale", "sum")
        kw.setdefault("exp_outputs_init", "additive")
        kw.setdefault("tau_init", 0.05)
        super().__init__(obs_dim, act_dim, **kw)


@register("fastlut_lse_sum_expmlpcrit")
class FastLUTLSESumExpMLPCriticActorCritic(BaseActorCritic):
    """exp19: exp17's log-sum-exp ACTOR + the exp10 MLP critic whose FINAL LINEAR READOUT
    is replaced by the same sum-scaled log-sum-exp aggregation.

    The critic backbone is untouched: obs -> [256,256] Tanh, orthogonal init gain 1.0,
    built by the identical `_ortho(_mlp([obs_dim, 256, 256, 1]))` call exp10 uses, so its
    weights and the RNG stream are bit-identical to exp10's / exp17's critic. Only how the
    final layer's per-unit contributions are pooled changes:

        plain (exp10, exp17):  value = sum_i w_i * h_i + b
        exp19:                 value = T * tau_c * log( (1/T) sum_i exp(w_i * h_i / tau_c) ) + b

    over the T = 256 penultimate units, with one trainable positive tau_c (softplus). Same
    sum-scaled form as the actor: tau->inf recovers the plain linear sum exactly, tau->0
    gives T * max_i(w_i h_i). The bias is added OUTSIDE the pooling so the tau->inf limit
    is the exact original layer, bias included.

    TAU_C INIT = 0.25, and the choice is a genuine trade-off, measured on real normalised
    observations through a real initialised critic (exp19's `design_tau_critic.py`):

        tau     shape dev vs plain    corr     effective units (of 256)
        4.0            1.3%          0.9999        256.0   <- exponential INERT
        1.0            5.3%          0.9987        256.0   <- exponential INERT
        0.25          21.1%          0.9763        255.6   <- chosen
        0.10          52.8%          0.8199        253.3
        0.05         108.4%          0.3753        244.6
        0.02         339.5%         -0.1776        175.7

    The pooled terms u_i = w_i*h_i are tiny (std ~0.0146: orthogonal gain-1 on a 256->1 row
    gives |w_i| ~ 1/16, times a Tanh output), so the readout only becomes non-linear once
    tau approaches that scale. "Deviation" is measured AFTER removing each head's mean,
    because the raw deviation is dominated by the Jensen gap ~ T*Var(u)/(2*tau), which is
    very nearly a constant and is absorbed by the layer's own trainable bias within a few
    updates.

    tau_c >= 1 would satisfy "starts near plain-linear" most comfortably but makes the
    exponential INERT (all 256 units contribute uniformly, correlation 0.999 with the
    linear head) -- a null result by construction, with no gradient to move tau_c. tau_c
    <= 0.05 starts a substantially different value function (correlation 0.38), which is
    not "near plain-linear" in any useful sense. 0.25 keeps the value function 97.6%
    correlated with exp17's while leaving the exponential measurably live, so tau_c has
    real gradient and can move either way. tau_c is trainable, so this sets only the
    starting regime.

    Everything is additive: this registers a new arch and touches nothing existing.
    """

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6, hidden=(256, 256),
                 initial_weights_noise=0.001, log_std_init=0.0, tau_init=0.05,
                 tau_critic_init=0.25, critic_clamp=60.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        self.actor_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise,
            exp_outputs=True, exp_outputs_scale="sum", exp_outputs_init="additive",
            exp_outputs_tau_init=tau_init)
        # IDENTICAL construction to exp10/exp17's critic -- same call, same RNG draw.
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)
        self.critic_clamp = float(critic_clamp)
        self.tau_c_floor = 1e-3
        self.tau_c_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(tau_critic_init))), dtype=torch.float32))

    @property
    def tau_critic(self):
        return F.softplus(self.tau_c_raw).clamp_min(self.tau_c_floor)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)
        h = self.vf[:-1](obs)                       # (B, 256) penultimate activations
        lin = self.vf[-1]
        u = h * lin.weight.view(1, -1)              # (B, 256) per-unit contributions
        tau = self.tau_critic
        T = u.shape[-1]
        # Centred form. Writing u_i = mu + d_i,
        #     T*tau*log((1/T) sum_i exp(u_i/tau)) == T*mu + T*tau*log((1/T) sum_i exp(d_i/tau))
        # exactly. The right-hand form is far better conditioned: the naive version computes
        # logsumexp(u/tau) - log(T), a difference of two ~log(T)=5.55 values whose true
        # difference is ~1e-7 once tau >> spread(u), so fp32 cancellation destroys it (at
        # tau=500 the error reached 0.069 against a value std of 0.18). Centring puts the
        # cancellation on a quantity that is genuinely ~0, and makes the tau->inf limit
        # land on the plain linear head to machine precision.
        mu = u.mean(dim=-1)
        d = torch.clamp((u - mu.unsqueeze(-1)) / tau,
                        min=-self.critic_clamp, max=self.critic_clamp)
        # log(mean(exp(d))) via log1p/expm1 rather than logsumexp(d) - log(T). Both are
        # algebraically identical, but the logsumexp form subtracts two values that are
        # BOTH ~log(T)=5.55 when tau >> spread(u), and their true difference is ~1e-7 --
        # pure fp32 cancellation (measured: 0.061 error at tau=500 even after centring).
        # expm1/log1p are exact near zero, so this lands on the plain linear head to
        # machine precision as tau->inf. The +-60 clamp keeps expm1 inside fp32 range
        # (expm1(60) ~ 1.1e26) while still reproducing the tau->0 limit T*max(u).
        gap = tau * torch.log1p(torch.expm1(d).mean(dim=-1))
        value = T * (mu + gap) + lin.bias
        return mean, value.reshape(obs.shape[0])

    def extra_log(self):
        return dict(tau_actor=float(self.actor_lut.exp_outputs_tau.detach()),
                    tau_critic=float(self.tau_critic.detach()))


class _FastLUTLSESum2Base(BaseActorCritic):
    """exp18: BOTH actor and critic are anchor-pair LUTs with the sum-scaled log-sum-exp
    readout — the exp17 actor paired with a value head carrying the same exponential
    geometry, each with its OWN trainable tau.

    This is `fastlut2`'s topology (LUT actor + LUT critic, no MLP anywhere) with exp17's
    readout on one or both heads:

        out = T * tau * log( (1/T) * sum_t exp( w_t / tau ) )

    `critic_exp_outputs` selects the arm:
      True  -> exponential critic   (the treatment: both heads exponential)
      False -> plain-sum LUT critic (the CONTROL: identical in every other respect, so the
               only difference is the critic's readout — not "LUT critic vs MLP critic",
               which exp13-15 already measured)

    Critic tph matches the actor's (32) and therefore matches exp13's config exactly, so
    exp13 (plain actor + plain critic, 2358.6 +- 878.6) is directly comparable as the
    both-plain corner of the 2x2.
    """

    CRITIC_EXP = True

    def __init__(self, obs_dim, act_dim, tables_per_head=32, nap=6,
                 initial_weights_noise=0.001, log_std_init=0.0, tau_init=0.05,
                 critic_tables_per_head=None):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        ctph = tables_per_head if critic_tables_per_head is None else critic_tables_per_head
        common = dict(input_dim=obs_dim, n_heads=1, n_anchor_pairs=nap,
                      forward_mode="hard", use_bf16=False,
                      initial_weights_noise=initial_weights_noise,
                      exp_outputs_scale="sum", exp_outputs_init="additive",
                      exp_outputs_tau_init=tau_init)
        self.actor_lut = FastMultiHeadLut(n_outputs=act_dim, tables_per_head=tables_per_head,
                                          exp_outputs=True, **common)
        self.critic_lut = FastMultiHeadLut(n_outputs=1, tables_per_head=ctph,
                                           exp_outputs=self.CRITIC_EXP, **common)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)
        value = self.critic_lut(obs).reshape(obs.shape[0])
        return mean, value

    def extra_log(self):
        d = dict(tau_actor=float(self.actor_lut.exp_outputs_tau.detach()))
        d["tau_critic"] = (float(self.critic_lut.exp_outputs_tau.detach())
                           if self.CRITIC_EXP else float("nan"))
        return d


@register("fastlut_lse_sum2")
class FastLUTLSESum2ActorCritic(_FastLUTLSESum2Base):
    """exp18 TREATMENT — exponential readout on BOTH actor and critic (two taus)."""

    CRITIC_EXP = True


@register("fastlut_lse_sum2_plaincrit")
class FastLUTLSESum2PlainCriticActorCritic(_FastLUTLSESum2Base):
    """exp18 CONTROL — exp17's exponential actor + a PLAIN-SUM LUT critic (one tau).

    Isolates the critic's exponential readout: everything else (LUT critic topology, tph,
    init, PPO recipe) is identical to the treatment arm.
    """

    CRITIC_EXP = False


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


@register("fastlut_hypcrit")
class FastLUTHyperCriticActorCritic(BaseActorCritic):
    """MIXED LUT arch: actor = anchor-pair LUT (FastMultiHeadLut, FIXED random sign(x[a]-x[b])
    anchors, only tables train) — identical to `fastlut`'s actor; critic = HYPERPLANE LUT
    (HyperLUTHead, LEARNED per-bit sign-test addressing via decoupled straight-through) with
    n_outputs=1 -> scalar V(s). Actor and critic tables_per_head are INDEPENDENT: --tables-per-head
    sets the ACTOR tph (128 here); the hyperplane critic tph is fixed at `critic_tph` (64).
    Tests whether a learned-addressing hyperplane critic beats the fixed-anchor LUT critic (exp15)."""

    def __init__(self, obs_dim, act_dim, tables_per_head=128, critic_tph=64, nap=6,
                 initial_weights_noise=0.001, table_std=0.001, hyp_std=0.1, temp=1.0,
                 log_std_init=0.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
        self.actor_lut = FastMultiHeadLut(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim, n_anchor_pairs=nap,
            tables_per_head=tables_per_head, forward_mode="hard", use_bf16=False,
            initial_weights_noise=initial_weights_noise)
        self.critic_lut = HyperLUTHead(obs_dim, 1, critic_tph, nap, table_std, hyp_std, temp)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)          # (B, 1, act_dim) -> (B, act_dim)
        value = self.critic_lut(obs).squeeze(-1)        # (B, 1) -> (B,)
        return mean, value


@register("liflut_mlpexpcrit")
class LIFLUTMlpExpCriticActorCritic(BaseActorCritic):
    """exp20: LIF-detector LUT ACTOR (LIFMultiHeadLUT, universal-timing always-on, sum-scaled
    log-sum-exp exp_outputs readout) + the exp19-style MLP critic whose final linear readout is the
    same sum-scaled log-sum-exp aggregation with one trainable tau_c.

    Actor (LIFMultiHeadLUT): n_heads=1, tables_per_head=32, n_det=1, n_buckets=64,
    freeze_temperature=False, exp_outputs=True (scale='sum', additive init, tau_init default). Input
    latency-coded -> LIF membrane -> first-spike bucketing -> mixed-radix table gather; addresses learned
    via the straight-through soft path; the tables_per_head tables combined by the sum-scaled LSE.
    Critic: IDENTICAL to exp19's FastLUTLSESumExpMLPCriticActorCritic critic — same
    _ortho(_mlp([obs,256,256,1])) backbone, and the centred exponential pooling
        value = T*tau_c*log((1/T) sum_i exp(u_i/tau_c)) + bias   (u_i = w_i*h_i over the 256 penult units)
    with the bias added OUTSIDE the pooling so tau_c->inf recovers the plain linear head exactly.
    tau_c init 0.25 (exp19's measured trade-off). Everything additive; touches nothing existing."""

    def __init__(self, obs_dim, act_dim, tables_per_head=32, n_buckets=64, n_det=1,
                 hidden=(256, 256), log_std_init=0.0, tau_critic_init=0.25, critic_clamp=60.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT
        self.actor_lut = LIFMultiHeadLUT(
            input_dim=obs_dim, n_heads=1, n_outputs=act_dim,
            tables_per_head=tables_per_head, n_det=n_det, n_buckets=n_buckets,
            freeze_temperature=False, exp_outputs=True)
        # exp19-style exponential MLP critic (identical construction to exp10/exp17/exp19's critic).
        self.vf = _ortho(_mlp([obs_dim, *hidden, 1]), gain=1.0)
        self.critic_clamp = float(critic_clamp)
        self.tau_c_floor = 1e-3
        self.tau_c_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(tau_critic_init))), dtype=torch.float32))

    @property
    def tau_critic(self):
        return F.softplus(self.tau_c_raw).clamp_min(self.tau_c_floor)

    def forward(self, obs):
        mean = self.actor_lut(obs).squeeze(1)          # (B, 1, act_dim) -> (B, act_dim)
        # exp19 centred sum-scaled log-sum-exp critic readout (tau_c->inf == plain linear head).
        h = self.vf[:-1](obs)                          # (B, 256) penultimate activations
        lin = self.vf[-1]
        u = h * lin.weight.view(1, -1)                 # (B, 256) per-unit contributions
        tau = self.tau_critic
        T = u.shape[-1]
        mu = u.mean(dim=-1)
        d = torch.clamp((u - mu.unsqueeze(-1)) / tau, min=-self.critic_clamp, max=self.critic_clamp)
        gap = tau * torch.log1p(torch.expm1(d).mean(dim=-1))
        value = T * (mu + gap) + lin.bias
        return mean, value.reshape(obs.shape[0])

    def extra_log(self):
        return dict(tau_actor=float(self.actor_lut.exp_outputs_tau.detach()),
                    tau_critic=float(self.tau_critic.detach()))


@register("liflayer_mlpexpcrit")
class LIFLayerMlpExpCriticActorCritic(BaseActorCritic):
    """STACKED LIFLayer spiking ACTOR (obs -> 32 LIF -> 64 LIF -> act_dim; exp22 wiring)
    + exp19's exact MLP exponential-head critic.

    Actor: a stack of LIFLayer (src/spiky/lutorch/lif_layer.py), each detecting patterns as spike TIMINGS
    (@torch.compile). HIDDEN layers use output_transform='rescale' (an INPUT-INDEPENDENT learnable per-
    channel affine that standardizes the raw spike times without per-sample distortion) as the inter-layer
    conditioner; the FINAL readout layer uses output_transform='log' (log-time), applied only at the end
    before the action decode. DEFAULT gain everywhere (no gain_init hack). The final layer's act_dim
    LOG-times (~mean 2.55 at init) are decoded by a LEARNABLE per-action affine centred on that init mean:
    mean = out_scale * (log_t - out_center) + out_bias, out_center=2.55, out_scale init 1, out_bias init 0
    -> ~zero-centred, modestly-scaled initial action mean (a normal PPO actor head; then trains freely).
    Critic: IDENTICAL to exp20/exp19's exp-MLP head (256x256 backbone + centred sum-scaled log-sum-exp
    readout, trainable tau_c init 0.25)."""

    def __init__(self, obs_dim, act_dim, hidden=(32, 64), t_window=32.0,
                 log_std_init=0.0, tau_critic_init=0.25, critic_clamp=60.0,
                 out_center=2.75, out_scale_init=1.0):
        super().__init__(obs_dim, act_dim, log_std_init)
        from spiky.lutorch.lif_layer import LIFLayer
        self.t_window = float(t_window)
        dims = [obs_dim, *hidden, act_dim]
        # DEFAULT gain everywhere. HIDDEN layers -> "rescale" (input-independent affine conditioner);
        # the FINAL readout layer -> "log" (log-time), decoded to the action mean below. The rescale init
        # is calibrated per in_dim from the measured raw spike-time stats (mean, std) at ~unit-variance
        # input, so each hidden layer's output starts ~standardized (learnable, adapts during training).
        RESCALE_INIT = {17: (12.4, 1.0), 32: (15.6, 2.9), 192: (15.5, 2.9)}
        n_layers = len(dims) - 1
        layers = []
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(LIFLayer(dims[i], dims[i + 1], t_window=t_window, output_transform="log"))
            else:
                m, s = RESCALE_INIT.get(dims[i], (12.0, 1.0))
                layers.append(LIFLayer(dims[i], dims[i + 1], t_window=t_window,
                                       output_transform="rescale", rescale_init_mean=m, rescale_init_std=s))
        self.actor_layers = nn.ModuleList(layers)
        # log-time -> action-mean decode: learnable per-action affine centred on the measured init mean.
        self.out_center = float(out_center)
        self.out_scale = nn.Parameter(torch.full((act_dim,), float(out_scale_init)))
        self.out_bias = nn.Parameter(torch.zeros(act_dim))
        # exp19-style exponential MLP critic (identical construction to exp10/exp17/exp19/exp20's critic).
        self.vf = _ortho(_mlp([obs_dim, 256, 256, 1]), gain=1.0)
        self.critic_clamp = float(critic_clamp)
        self.tau_c_floor = 1e-3
        self.tau_c_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(tau_critic_init))), dtype=torch.float32))

    @property
    def tau_critic(self):
        return F.softplus(self.tau_c_raw).clamp_min(self.tau_c_floor)

    def forward(self, obs):
        h = obs
        for layer in self.actor_layers:
            h = layer(h)                                   # log first-spike times; last -> (B, act_dim)
        mean = self.out_scale * (h - self.out_center) + self.out_bias   # log-time -> ~zero-centred action mean
        # exp19 centred sum-scaled log-sum-exp critic readout (tau_c->inf == plain linear head).
        hc = self.vf[:-1](obs)
        lin = self.vf[-1]
        u = hc * lin.weight.view(1, -1)
        tau = self.tau_critic
        T = u.shape[-1]
        mu = u.mean(dim=-1)
        d = torch.clamp((u - mu.unsqueeze(-1)) / tau, min=-self.critic_clamp, max=self.critic_clamp)
        gap = tau * torch.log1p(torch.expm1(d).mean(dim=-1))
        value = T * (mu + gap) + lin.bias
        return mean, value.reshape(obs.shape[0])

    def extra_log(self):
        return dict(tau_critic=float(self.tau_critic.detach()))


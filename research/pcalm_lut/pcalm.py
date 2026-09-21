"""PC / PC-ALM / BP trainers over a layer-constrained network, plus the paper's residual MLP.

Implements Algorithm 1 of "Augmented Lagrangian Predictive Coding" (Seely & Gould, arXiv 2605.31022v1) in the
form we need for the LUT study: the network is written as a chain of layer maps f_i and trained as

    min  loss(readout(h_{L-1}), y)   s.t.   h_i = f_i(h_{i-1}),  i = 1..L-1                              (1)

    L_rho(h, theta, lam) = loss + sum_i lam_i . r_i + (rho/2) sum_i ||r_i||^2,   r_i = h_i - f_i(h_{i-1})  (5)

Trainers, all sharing the same model/init/data:
  * "bp"     : ordinary autograd through the stack.
  * "pc"     : finite-inference predictive coding = PC-ALM with lam pinned to 0 (alpha = 0).
  * "pcalm"  : Algorithm 1 -- init h from a forward pass and lam = 0, then T-1 x (primal activity step, dual
               step lam_i += alpha r_i), a final primal step, then one weight step on grad_theta L_rho.

Every trainer takes ONE optimiser step per batch on the same parameters, so "steps" are comparable; wall clock
is not (PC/PC-ALM pay T inner iterations), which is why the study reports both.

The activity gradient is taken with autograd on the scalar L_rho with h as the leaf, so any layer map f_i works
-- including one whose gradient w.r.t. its input is a surrogate (LUT layers: detached address, gradient through
the confidence score). That is the point of the study: the surrogate is applied once per layer per inner step
and is never composed across layers, whereas BP composes L of them.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerChain(nn.Module):
    """A network expressed as (input map, [interior layer maps], readout) for the constrained formulation.

    Subclasses provide:
        h1(x)                -> first hidden state (the i=1 constraint target, clamped input)
        layer(i, h)          -> f_{i+1}(h) for interior layers, i = 1 .. L-2 (0-based list index)
        readout(h)           -> prediction from h_{L-1}
    `n_hidden` is L-1, the number of free hidden states.
    """

    def h1(self, x):
        raise NotImplementedError

    def layer(self, i, h):
        raise NotImplementedError

    def readout(self, h):
        raise NotImplementedError

    def forward(self, x):
        h = self.h1(x)
        for i in range(self.n_hidden - 1):
            h = self.layer(i, h)
        return self.readout(h)

    @torch.no_grad()
    def init_states(self, x):
        """Forward-pass initialisation of the free states (Algorithm 1, line 4)."""
        hs = [self.h1(x)]
        for i in range(self.n_hidden - 1):
            hs.append(self.layer(i, hs[-1]))
        return hs

    clamp_mode = 'data'        # 'data': input clamped, target enters as the loss term (the paper's setting)
                               # 'pinned': the target is clamped as the top state h_L and contributes one
                               #           more residual; the loss term is then dropped (no double count)

    def residuals(self, x, hs, y=None):
        """r_i = h_i - f_i(h_{i-1}) for i = 1..L-1 (r_1 uses the clamped input), plus, under pinned
        clamping, the top residual r_L = y - readout(h_{L-1})."""
        r = [hs[0] - self.h1(x)]
        for i in range(self.n_hidden - 1):
            r.append(hs[i + 1] - self.layer(i, hs[i]))
        if self.clamp_mode == 'pinned':
            if y is None:
                raise ValueError('clamp_mode="pinned" needs the target: the top state IS y')
            r.append(y - self.readout(hs[-1]))
        return r

    def energy(self, x, y, hs, lam, rho, loss_fn):
        """L_rho of (5), SUMMED over the batch -- inference in Algorithm 1 is per sample (line 3: "for each
        sample in B in parallel"), so the primal gradient and the dual step lam += alpha r must be in the same
        per-sample units. (Averaging the energy but not the dual step mis-scales their relative rates by |B|
        and turns the primal-dual iteration into a limit cycle.) The LEARNING step divides by |B|."""
        r = self.residuals(x, hs, y)
        # under pinned clamping the output error IS the last residual, so the loss term would count it twice
        e = 0.0 if self.clamp_mode == 'pinned' else loss_fn(self.readout(hs[-1]), y) * x.shape[0]
        for ri, li in zip(r, lam):
            if li is not None:
                e = e + (li * ri).sum()
            e = e + 0.5 * rho * ri.pow(2).sum()
        return e, r


def squared_error(pred, y):
    """1/2 ||pred - y||^2, summed over features and averaged over the batch (the paper's supervised loss)."""
    return 0.5 * (pred - y).pow(2).sum() / pred.shape[0]


def train_step(model, x, y, opt, mode, *, T=None, eta_h=0.1, alpha=1.0, rho=1.0, loss_fn=squared_error,
               grad_clip=None):
    """One optimiser step for one batch. mode in {"bp", "pc", "pcalm"}. Returns (loss_value, stats)."""
    opt.zero_grad(set_to_none=True)
    if mode == 'bp':
        loss = loss_fn(model(x), y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        return float(loss.detach()), {}
    if mode not in ('pc', 'pcalm'):
        raise ValueError(f'unknown mode {mode!r}')
    alpha = 0.0 if mode == 'pc' else alpha
    hs = [h.detach().clone().requires_grad_(True) for h in model.init_states(x)]
    with torch.no_grad():        # one multiplier per CONSTRAINT: pinned clamping adds the top residual
        lam = [torch.zeros_like(r) for r in model.residuals(x, hs, y)]
    T = T or 2 * (model.n_hidden + 1)
    for t in range(T - 1):
        e, r = model.energy(x, y, hs, lam, rho, loss_fn)
        g = torch.autograd.grad(e, hs)                       # layer-local: grad_{h_i} touches only i-1, i, i+1
        with torch.no_grad():
            for h, gi in zip(hs, g):
                h -= eta_h * gi                              # primal step (Alg. 1 line 7)
            if alpha:
                for li, ri in zip(lam, r):
                    li += alpha * ri.detach()                # dual step  (Alg. 1 line 8)
    e, r = model.energy(x, y, hs, lam, rho, loss_fn)         # final primal step (Alg. 1 line 10)
    g = torch.autograd.grad(e, hs)
    with torch.no_grad():
        for h, gi in zip(hs, g):
            h -= eta_h * gi
    e, r = model.energy(x, y, hs, lam, rho, loss_fn)         # learning step (Alg. 1 line 12)
    (e / x.shape[0]).backward()                              # 1/|B| average over the batch
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    with torch.no_grad():
        stats = {'resid_rms': float(torch.stack([ri.pow(2).mean() for ri in r]).mean().sqrt()),
                 'lam_rms': float(torch.stack([li.pow(2).mean() for li in lam]).mean().sqrt()),
                 'energy': float(e.detach())}
        stats['loss_at_h'] = float(loss_fn(model.readout(hs[-1]), y))
    return float(loss_fn(model(x), y).detach()), stats


class ResidualMLP(LayerChain):
    """The paper's residual MLP (App. B.1) at the mean-field point (App. B.2).

        h_1 = a_1 W_1 x,   h_i = h_{i-1} + a_i W_i sigma(h_{i-1}),   y_hat = a_L W_L sigma(h_{L-1})
        a_1 = 1/sqrt(D),   a_i = 1/sqrt(L N),   a_L = 1/(gamma0 N),   W ~ N(0, 1) entrywise
    """

    def __init__(self, in_dim, out_dim, width, depth, act='tanh', gamma0=1.0, device='cuda', seed=0):
        super().__init__()
        assert depth >= 2
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.depth, self.width, self.n_hidden = depth, width, depth - 1
        self.act = {'id': lambda z: z, 'tanh': torch.tanh, 'relu': F.relu}[act]
        self.W1 = nn.Parameter(torch.randn(width, in_dim, generator=g))
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(width, width, generator=g))
                                    for _ in range(depth - 2)])
        self.WL = nn.Parameter(torch.randn(out_dim, width, generator=g))
        self.a1 = 1.0 / math.sqrt(in_dim)
        self.ai = 1.0 / math.sqrt(depth * width)
        self.aL = 1.0 / (gamma0 * width)
        self.to(device)

    def h1(self, x):
        return self.a1 * F.linear(x, self.W1)

    def layer(self, i, h):
        return h + self.ai * F.linear(self.act(h), self.Wi[i])

    def readout(self, h):
        return self.aL * F.linear(self.act(h), self.WL)


def constraint_sigma_max(model, x, y=None, iters=30, eps=1e-3):
    """sigma_max of the constraint operator A = dr/dh at the current point, by power iteration on A^T A.

    The paper sets eta_h = 1 / sigma_max(A)^2 per (N, L) and dataset (App. F); this reproduces that estimate
    without forming A. A v is taken by finite differences and A^T u by one backward of (r . u): both work for
    layer maps whose forward is torch.compiled (double backward is not supported there) and for LUT layers,
    whose address is piecewise constant -- at eps this small the FD stays inside one cell almost everywhere, so
    it measures the smooth part of the map, which is what the step-size bound is about.
    """
    hs = [h.detach().clone() for h in model.init_states(x)]
    v = [torch.randn_like(h) for h in hs]
    n = math.sqrt(sum(float(vi.pow(2).sum()) for vi in v))
    v = [vi / n for vi in v]
    sigma = 0.0
    with torch.no_grad():
        r0 = model.residuals(x, hs, y)
    for _ in range(iters):
        with torch.no_grad():                                       # A v by finite differences
            hp = [h + eps * vi for h, vi in zip(hs, v)]
            rp = model.residuals(x, hp, y)
            Av = [(a - b) / eps for a, b in zip(rp, r0)]
        hg = [h.detach().clone().requires_grad_(True) for h in hs]  # A^T (A v) by one backward
        r = model.residuals(x, hg, y)
        AtAv = torch.autograd.grad(r, hg, grad_outputs=Av)
        n = math.sqrt(sum(float(t.pow(2).sum()) for t in AtAv))
        if n == 0:
            return 0.0
        v = [t / n for t in AtAv]
        sigma = math.sqrt(n)
    return sigma


def stability_ok(eta_h, sigma_max, rho, alpha):
    """PC-ALM per-mode Jury bound (eq. 14): eta_h sigma^2 (2 rho + alpha) < 4."""
    return eta_h * sigma_max ** 2 * (2 * rho + alpha) < 4.0


def run_epochs(model, loader, mode, *, epochs=1, lr=1e-3, device='cuda', log_every=100, max_steps=None,
               **step_kw):
    """Train with Adam; returns a history of (step, loss, wall time) and the total wall clock.

    `max_steps` truncates the budget (a fraction of an epoch) while keeping the batch order identical, so a
    truncated run is a strict prefix of the full one."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, step, t0 = [], 0, time.time()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss, stats = train_step(model, x, y, opt, mode, **step_kw)
            step += 1
            if step % log_every == 0 or step == 1:
                hist.append({'step': step, 'loss': loss, 'time': time.time() - t0, **stats})
            if max_steps and step >= max_steps:
                return hist, time.time() - t0
    return hist, time.time() - t0

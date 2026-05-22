"""Convergence-speed bottleneck: gradient SNR of LUT vs vanilla at bs=16.

Why does LUT-LM (exp475, 89.4M) converge to a WORSE loss than vanilla (exp476,
35.8M) at the same batch size, despite more params? Hypothesis: LUT's hard-argmax
selection makes each weight-row receive gradient from only ~1/K of tokens
(sparse, high-variance per-row gradient = low SNR = slow SGD convergence), while
vanilla Linear weights get a dense gradient from every token.

Measure, per parameter GROUP, at a fixed weight state, over M independent bs=16
microbatches:
  - grad_cosine: mean_m cos(g_m, mu), mu = mean over M. The fraction of a single
    step that points along the true descent direction. High = clean/fast,
    low = noisy/slow. This is the directional SNR that governs SGD convergence.
  - per-element |mu|/sigma SNR (median).
  - coverage (LUT only): fraction of weight-table rows that get ANY gradient in
    one microbatch (vanilla dense = 1.0 by construction).

Measured at INIT (fresh models = the actual training start) and at the trained
checkpoints (noise floor near each optimum). The LUT/vanilla RATIO is the answer.

Run: /home/starost/spiky/.venv/bin/python grad_snr.py
"""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = '/home/starost/spiky/nanochat_exps/analysis_exp475_vs_exp486'
sys.path.insert(0, ANALYSIS)
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from model_def import build_model as build_lut          # exp475/486 LUT model
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

DEVICE = 'cuda'
CKPT_LUT = '/home/starost/spiky/nanochat_exps/exp475_meanabs_nocenter/checkpoint.pt'
CKPT_VAN = '/home/starost/spiky/nanochat_exps/exp476_untied_emb_head/checkpoint.pt'
M = 24            # independent microbatches
BS, CTX, VOCAB = 16, 512, 32768


# ---- vanilla MinimalGPT (exp476), reconstructed -----------------------------
class RoPE(nn.Module):
    def __init__(self, hd, T, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(T).float()
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rot(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + _rot(q) * sin, k * cos + _rot(k) * sin


class VAttn(nn.Module):
    def __init__(self, E, H):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(E, 3 * E, bias=False)
        self.proj = nn.Linear(E, E, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, C // self.H).transpose(1, 2)
        k = k.view(B, T, self.H, C // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, C // self.H).transpose(1, 2)
        q, k = _rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class VBlock(nn.Module):
    def __init__(self, E, H):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(E), nn.LayerNorm(E)
        self.attn = VAttn(E, H)
        self.mlp = nn.Sequential(nn.Linear(E, 4 * E, bias=False), nn.GELU(),
                                 nn.Linear(4 * E, E, bias=False))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class MinimalGPT(nn.Module):
    def __init__(self, V, E, H, L, T):
        super().__init__()
        self.tok_emb = nn.Embedding(V, E)
        self.rope = RoPE(E // H, T)
        self.blocks = nn.ModuleList([VBlock(E, H) for _ in range(L)])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, V, bias=False)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight); nn.init.zeros_(b.mlp[-1].weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.rope.cos.to(x.device), self.rope.sin.to(x.device))
        logits = self.head(self.ln_f(x))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits


# ---- group classification ----------------------------------------------------
def lut_groups(model):
    """body LUT tables (ndim>=3) vs the dense head (unembedder)."""
    g = {'LUT_body(tables)': [], 'head(dense)': []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 3:
            g['LUT_body(tables)'].append((n, p))
        elif n.startswith('unembedder.'):
            g['head(dense)'].append((n, p))
    return g


def van_groups(model):
    """body dense matmuls (attn/mlp Linears) vs the dense head."""
    g = {'vanilla_body(Linears)': [], 'head(dense)': []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('blocks.') and p.ndim == 2:
            g['vanilla_body(Linears)'].append((n, p))
        elif n.startswith('head.'):
            g['head(dense)'].append((n, p))
    return g


@torch.enable_grad()
def grad_snr(model, batches, groups):
    """Two-pass, numerically-stable. pass1 -> mu = mean grad over M microbatches;
    pass2 -> var = mean (g-mu)^2 (no E[x^2]-E[x]^2 cancellation).

    Reports per group the GRADIENT NOISE SCALE = E[||g-mu||^2] / ||mu||^2: the
    per-microbatch noise-to-signal ratio. It's exactly how many microbatches you
    must average to get a fixed-quality gradient — higher = noisier = slower SGD
    convergence. Also the per-element SNR median (|mu|/sigma)."""
    params = [p for _, lst in groups.items() for _, p in lst]
    s1 = {id(p): torch.zeros_like(p) for p in params}
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        model(x, targets=y).backward()
        for p in params:
            if p.grad is not None:
                s1[id(p)] += p.grad
    mu = {id(p): s1[id(p)] / M for p in params}
    sse = {id(p): torch.zeros_like(p) for p in params}     # sum sq err
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        model(x, targets=y).backward()
        for p in params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            sse[id(p)] += (g - mu[id(p)]) ** 2
    var = {id(p): sse[id(p)] / M for p in params}
    out = {}
    for gname, lst in groups.items():
        noise = sum(var[id(p)].sum().item() for _, p in lst)        # E||g-mu||^2 (summed)
        signal = sum(mu[id(p)].pow(2).sum().item() for _, p in lst)  # ||mu||^2
        snr_meds = []
        for _, p in lst:
            sig = var[id(p)].sqrt()
            snr_meds.append((mu[id(p)].abs() / (sig + 1e-12)).median().item())
        out[gname] = dict(
            noise_to_signal=float(noise / (signal + 1e-20)),
            snr_median=float(sum(snr_meds) / len(snr_meds)),
            n_tensors=len(lst),
        )
    return out


@torch.no_grad()
def lut_coverage(model, batch):
    """Fraction of weight-table rows that receive ANY gradient in ONE microbatch,
    per LUT module type (= fraction of rows selected by >=1 token)."""
    from model_def import compute_lut_indices, LUT_NAMES
    x, y = batch
    # capture each LUT input
    store = {li: {} for li in range(len(model.layers))}
    handles = []
    for li, blk in enumerate(model.layers):
        for nm in LUT_NAMES:
            mod = getattr(blk, nm)
            def mk(li=li, nm=nm):
                def hook(m, inp):
                    store[li][nm] = inp[0].detach()
                return hook
            handles.append(mod.register_forward_pre_hook(mk()))
    model(x)
    for h in handles:
        h.remove()
    cov = {nm: [] for nm in LUT_NAMES}
    for li in range(len(model.layers)):
        for nm in LUT_NAMES:
            mod = getattr(model.layers[li], nm)
            idx = compute_lut_indices(mod, store[li][nm])     # [N, n_tables]
            n_tables = idx.shape[1]; K = mod.weights.shape[1]
            flat = (idx + torch.arange(n_tables, device=idx.device).unsqueeze(0) * K).reshape(-1)
            visited = torch.bincount(flat, minlength=n_tables * K) > 0
            cov[nm].append(visited.float().mean().item())
    return {nm: sum(v) / len(v) for nm, v in cov.items()}


@torch.no_grad()
def gradient_density(model, batch):
    """Effective gradient samples PER PARAMETER (= how many tokens contribute to
    each weight's gradient in one bs=16 microbatch of B*T tokens).

    Vanilla Linear: every weight gets all B*T tokens (dense). LUT row: only the
    tokens that select it. Returns per-LUT-module the visits-per-row distribution
    (min/p10/median) — the low tail is the rarely-updated rows that converge
    slowest. The √(visits) is the per-row gradient SNR vs vanilla's √(B*T)."""
    from model_def import compute_lut_indices, LUT_NAMES
    x, _ = batch
    BT = x.numel()
    store = {li: {} for li in range(len(model.layers))}
    handles = []
    for li, blk in enumerate(model.layers):
        for nm in LUT_NAMES:
            mod = getattr(blk, nm)
            def mk(li=li, nm=nm):
                def hook(m, inp):
                    store[li][nm] = inp[0].detach()
                return hook
            handles.append(mod.register_forward_pre_hook(mk()))
    model(x)
    for h in handles:
        h.remove()
    out = {}
    for nm in LUT_NAMES:
        vis_all = []
        for li in range(len(model.layers)):
            mod = getattr(model.layers[li], nm)
            idx = compute_lut_indices(mod, store[li][nm])
            n_tables = idx.shape[1]; K = mod.weights.shape[1]
            flat = (idx + torch.arange(n_tables, device=idx.device).unsqueeze(0) * K).reshape(-1)
            binc = torch.bincount(flat, minlength=n_tables * K).float()
            vis_all.append(binc)
        v = torch.cat(vis_all)
        out[nm] = dict(
            K=int(getattr(model.layers[0], nm).weights.shape[1]),
            visits_per_row_median=float(v.median()),
            visits_per_row_p10=float(v.quantile(0.10)),
            visits_per_row_min=float(v.min()),
            frac_rows_under_16=float((v < 16).float().mean()),
            tokens_per_microbatch=BT,
        )
    return out


def main():
    tok = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, BS, CTX, split='train', device=DEVICE)
    # NB: the loader REUSES one buffer object per call -> must clone, else all
    # M "microbatches" alias the same data (zero gradient variance).
    batches = [(x.clone(), y.clone()) for x, y in (next(loader) for _ in range(M))]

    ckL = torch.load(CKPT_LUT, map_location='cpu', weights_only=False)
    cfgL = dict(ckL['config']); cfgL['vocab_size'] = VOCAB

    results = {}
    for state in ('init', 'trained'):
        print(f'\n################ STATE = {state} ################')
        # LUT
        lut = build_lut(cfgL, device=DEVICE)
        if state == 'trained':
            lut.load_state_dict(ckL['model_state_dict'], strict=False)
        lut.train()
        rL = grad_snr(lut, batches, lut_groups(lut))
        cov = lut_coverage(lut, batches[0])
        # vanilla
        van = MinimalGPT(VOCAB, 384, 6, 6, CTX).to(DEVICE)
        if state == 'trained':
            ckV = torch.load(CKPT_VAN, map_location='cpu', weights_only=False)
            sdV = ckV['model_state_dict'] if isinstance(ckV, dict) and 'model_state_dict' in ckV else ckV
            van.load_state_dict(sdV, strict=False)
        van.train()
        rV = grad_snr(van, batches, van_groups(van))

        dens = gradient_density(lut, batches[0]) if state == 'trained' else None
        results[state] = dict(lut=rL, vanilla=rV, lut_coverage=cov, density=dens)
        if dens is not None:
            print('  --- LUT gradient density (tokens contributing per ROW; vanilla weight = all 8192) ---')
            for nm, d in dens.items():
                print(f'    {nm:14s} K={d["K"]:3d}: median={d["visits_per_row_median"]:.0f} '
                      f'p10={d["visits_per_row_p10"]:.0f} min={d["visits_per_row_min"]:.0f} '
                      f'frac<16={d["frac_rows_under_16"]:.3f}  (vanilla=8192/weight)')
        print('  --- gradient noise-to-signal  E||g-mu||^2/||mu||^2  (higher=noisier=slower) ---')
        for g, d in rL.items():
            print(f'    LUT {g:24s}: noise/signal={d["noise_to_signal"]:.2f}  snr_med={d["snr_median"]:.4f}')
        for g, d in rV.items():
            print(f'    VAN {g:24s}: noise/signal={d["noise_to_signal"]:.2f}  snr_med={d["snr_median"]:.4f}')
        print('  --- LUT per-microbatch row coverage (bs=16) ---')
        for nm, c in cov.items():
            print(f'    {nm:14s}: {c:.3f}')
        del lut, van
        torch.cuda.empty_cache()

    with open(os.path.join(HERE, 'grad_snr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote grad_snr_results.json')


if __name__ == '__main__':
    main()

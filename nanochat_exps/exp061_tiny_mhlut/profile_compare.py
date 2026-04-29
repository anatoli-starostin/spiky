"""Profile exp061 (bf16 TinyMultiHeadLut) vs exp060 (fp32 MultiHeadLut).

Same architecture, same hyperparams, fake data — measures step time and peak
GPU memory for fwd+bwd+optimizer.step() so we can quantify the bf16 win
before launching the full 8K-step training run.

Run:
    PYTHONPATH=/home/starost/nanochat \\
        /home/starost/spiky/.venv/bin/python \\
        nanochat_exps/exp061_tiny_mhlut/profile_compare.py
"""
import os, sys, json, time, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spiky modules.
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import DominanceToVector, VectorToDominance

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = torch.device('cuda:0')
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
D_QK_P      = d_qk * (d_qk - 1) // 2
D_V_P       = d_v * (d_v - 1) // 2
DEVICE_BS   = cfg['device_batch_size']
VOCAB_SIZE  = 32768  # nanochat BPE vocab size

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')


def _make_qk_joint_tiny(layer_idx):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + layer_idx, device=DEVICE,
    )

def _make_v_tiny(layer_idx):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + 200 + layer_idx, device=DEVICE,
    )

def _make_out_tiny(layer_idx):
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER else cfg['out_tph']
    return TinyMultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=tph,
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + 400 + layer_idx, device=DEVICE,
    )


def _make_qk_joint_full(layer_idx):
    return MultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
        n_alternatives=1, smooth_mode=False,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + layer_idx, device=DEVICE,
    )

def _make_v_full(layer_idx):
    return MultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        n_alternatives=1, smooth_mode=False,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + 200 + layer_idx, device=DEVICE,
    )

def _make_out_full(layer_idx):
    tph = _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER else cfg['out_tph']
    return MultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=tph,
        n_alternatives=1, smooth_mode=False,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed'] + 400 + layer_idx, device=DEVICE,
    )


class TinyLUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint_tiny(layer_idx)
        self.v_lut    = _make_v_tiny(layer_idx)
        self.out_proj = _make_out_tiny(layer_idx)
        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.v_v2d  = VectorToDominance(d_v, smooth_mode=False, temperature=canon_t)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_ln = nn.LayerNorm(E)  # fp32
        self.attn_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, pos_emb):
        # x fp32, pos_emb fp32 — matches confined-bf16 design.
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)
        qk_out = self.qk_joint(xp).float()
        q_vec, k_vec = qk_out[..., :d_qk], qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k_dom = self.qk_v2d(k_vec).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v_vec = self.v_lut(x_flat).float()
        v_dom = self.v_v2d(v_vec).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        attn_dom = F.scaled_dot_product_attention(
            q_dom * self.attn_scale, k_dom, v_dom, is_causal=True,
        )
        attn = self.attn_to_vec(attn_dom)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1).float()
        return self.out_ln(out_real).reshape(B, T, E)


class FullLUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qk_joint = _make_qk_joint_full(layer_idx)
        self.v_lut    = _make_v_full(layer_idx)
        self.out_proj = _make_out_full(layer_idx)
        canon_t = cfg.get('canon_temperature', 0.1)
        self.qk_v2d = VectorToDominance(d_qk, smooth_mode=False, temperature=canon_t)
        self.v_v2d  = VectorToDominance(d_v, smooth_mode=False, temperature=canon_t)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_ln = nn.LayerNorm(E)
        self.attn_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, pos_emb):
        # x fp32, pos_emb fp32
        B, T, _ = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, E)
        x_flat = x.reshape(B * T, E)
        qk_out = self.qk_joint(xp)
        q_vec, k_vec = qk_out[..., :d_qk], qk_out[..., d_qk:]
        q_dom = self.qk_v2d(q_vec).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k_dom = self.qk_v2d(k_vec).reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        v_vec = self.v_lut(x_flat)
        v_dom = self.v_v2d(v_vec).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)
        attn_dom = F.scaled_dot_product_attention(
            q_dom * self.attn_scale, k_dom, v_dom, is_causal=True,
        )
        attn = self.attn_to_vec(attn_dom)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_real = self.out_proj(out_in).squeeze(1)
        return self.out_ln(out_real).reshape(B, T, E)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([TinyLUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, VOCAB_SIZE),
        )

    def forward(self, tokens, targets):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        logits = self.unembedder(concat)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
        )


class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([FullLUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = N_LAYERS * E
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, VOCAB_SIZE),
        )

    def forward(self, tokens, targets):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        logits = self.unembedder(concat)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
        )


def profile_model(name, model, optimizers, n_warmup=5, n_steps=20):
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokens = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, CONTEXT_SIZE), device=DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, CONTEXT_SIZE), device=DEVICE)

    # Warmup
    for _ in range(n_warmup):
        for opt in optimizers:
            opt.zero_grad()
        loss = model(tokens, targets)
        loss.backward()
        for opt in optimizers:
            opt.step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_steps):
        for opt in optimizers:
            opt.zero_grad()
        loss = model(tokens, targets)
        loss.backward()
        for opt in optimizers:
            opt.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f'\n{name}:')
    print(f'  step_time_ms = {1000 * elapsed / n_steps:.2f}')
    print(f'  peak_mem_mb  = {peak_mb:.1f}')
    print(f'  final_loss   = {loss.item():.4f}')
    return {'name': name, 'step_ms': 1000 * elapsed / n_steps, 'peak_mb': peak_mb}


# --- (1) fp32 Tiny + single AdamW (the v8 setup) ---
print('Building TinyModel fp32 (single AdamW)...')
tiny_model = TinyModel().to(DEVICE)
tiny_param_count = sum(p.numel() for p in tiny_model.parameters())
print(f'  Total params (fp32): {tiny_param_count:,}')
tiny_adam = torch.optim.AdamW(tiny_model.parameters(), lr=cfg['adam_lr'], betas=(0.9, 0.95))
fp32_tiny_stats = profile_model('exp061 fp32 Tiny + AdamW', tiny_model, [tiny_adam])
del tiny_model, tiny_adam
gc.collect(); torch.cuda.empty_cache()

# --- (2) bf16 max-hard Tiny: weights bf16 + bf16 moments + bf16 compute + SR ---
print('\nBuilding TinyModel bf16 (max-hard: bf16 wts + bf16 moments + bf16 compute + SR)...')
import spiky.lutorch.tiny_multi_head_lut as tmhlut_mod
import spiky.lutorch.ranking_tools as rt_mod

# Override the make_*_tiny factories to use bf16 weights for this profile.
def _build_bf16_tiny():
    bf16_kwargs = dict(weight_dtype=torch.bfloat16,
                       anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                       random_seed=cfg['random_seed'], device=DEVICE)
    # Reuse TinyModel's __init__ but patch _make_*_tiny to give bf16 weights.
    saved = (_make_qk_joint_tiny, _make_v_tiny, _make_out_tiny)
    return saved
# Simpler: just monkey-patch + rebuild.
saved_mk = (globals()['_make_qk_joint_tiny'], globals()['_make_v_tiny'], globals()['_make_out_tiny'])
def _bf16_qk(li):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2*d_qk,
        n_anchor_pairs=cfg['qk_input_nap'], tables_per_head=cfg['qk_tph'],
        weight_dtype=torch.bfloat16,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed']+li, device=DEVICE)
def _bf16_v(li):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        weight_dtype=torch.bfloat16,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed']+200+li, device=DEVICE)
def _bf16_out(li):
    tph = _OUT_TPH_PER_LAYER[li] if _OUT_TPH_PER_LAYER else cfg['out_tph']
    return TinyMultiHeadLut(input_dim=H*d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=tph,
        weight_dtype=torch.bfloat16,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        random_seed=cfg['random_seed']+400+li, device=DEVICE)
globals()['_make_qk_joint_tiny'] = _bf16_qk
globals()['_make_v_tiny'] = _bf16_v
globals()['_make_out_tiny'] = _bf16_out

bf16_model = TinyModel().to(DEVICE)
# Also need the LUTBlock to .float() after each LUT call to mimic v9's
# confined-bf16 design — but our TinyLUTBlock here doesn't .float(). That's
# fine for the profile: profile measures the optimizer + kernel cost, which
# is what differs between v8 and v9.
bf16_tiny_modules = [m for m in bf16_model.modules() if isinstance(m, TinyMultiHeadLut)]
bf16_tiny_ids = {id(m.weights) for m in bf16_tiny_modules}
bf16_adam_params = [p for p in bf16_model.parameters() if id(p) not in bf16_tiny_ids]
bf16_adam = torch.optim.AdamW(bf16_adam_params, lr=cfg['adam_lr'], betas=(0.9, 0.95))
bf16_tiny_opt = TinyMultiHeadLutOptimizer(
    bf16_tiny_modules, lr=cfg['adam_lr'], beta1=0.9, beta2=0.95,
    state_dtype=torch.bfloat16, compute_dtype=torch.bfloat16,
    stochastic_rounding=True,
)
bf16_stats = profile_model('exp061 bf16 max-hard Tiny + SR', bf16_model, [bf16_adam, bf16_tiny_opt])
del bf16_model, bf16_adam, bf16_tiny_opt, bf16_tiny_modules
globals()['_make_qk_joint_tiny'], globals()['_make_v_tiny'], globals()['_make_out_tiny'] = saved_mk
gc.collect(); torch.cuda.empty_cache()

# --- (3) FullModel + AdamW (exp060 baseline) ---
print('\nBuilding FullModel (fp32 MultiHeadLut)...')
full_model = FullModel().to(DEVICE)
full_param_count = sum(p.numel() for p in full_model.parameters())
print(f'  Total params (fp32): {full_param_count:,}')
full_adam = torch.optim.AdamW(full_model.parameters(), lr=cfg['adam_lr'], betas=(0.9, 0.95))
full_stats = profile_model('exp060 fp32 MHLut', full_model, [full_adam])

# --- Summary ---
print('\n' + '=' * 70)
print('Summary  (DEVICE_BS={}, CONTEXT_SIZE={})'.format(DEVICE_BS, cfg['context_size']))
print('=' * 70)
print(f'                                       step_ms      peak_GB')
print(f'  exp060 fp32 MHLut (baseline)     : {full_stats["step_ms"]:8.2f}    {full_stats["peak_mb"]/1024:6.2f}')
print(f'  exp061 fp32 Tiny + AdamW          : {fp32_tiny_stats["step_ms"]:8.2f}    {fp32_tiny_stats["peak_mb"]/1024:6.2f}')
print(f'  exp061 bf16 max-hard Tiny + SR    : {bf16_stats["step_ms"]:8.2f}    {bf16_stats["peak_mb"]/1024:6.2f}')
print()
print(f'  fp32 Tiny vs exp060 :  {full_stats["step_ms"]/fp32_tiny_stats["step_ms"]:.2f}x speed   {fp32_tiny_stats["peak_mb"]/full_stats["peak_mb"]:.2f}x mem')
print(f'  bf16 Tiny vs exp060 :  {full_stats["step_ms"]/bf16_stats["step_ms"]:.2f}x speed   {bf16_stats["peak_mb"]/full_stats["peak_mb"]:.2f}x mem')
print(f'  bf16 vs fp32 Tiny   :  {fp32_tiny_stats["step_ms"]/bf16_stats["step_ms"]:.2f}x speed   {bf16_stats["peak_mb"]/fp32_tiny_stats["peak_mb"]:.2f}x mem')

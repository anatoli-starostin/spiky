"""exp517_tied_sparse_lut — tied unembedder via the PROVEN sparse-scatter LUT.

Simplified per the user's idea: instead of a custom row=token head + assignment STE
(exp514-516), use a STANDARD `TinyMultiHeadLut` with sparse-scatter (the exp100/body
mechanism) to map E -> VOCAB, plus an identity/inverse regularizer.

  - Embedder: nn.Embedding(V, E=64).
  - Body: exp428 block (qkv_lut + v_lut + out_proj per layer, RoPE attention).
  - Dual stream collapsed to E: residual_lut decodes R^E; summed residual = predicted
    token embedding (R^E).
  - Unembedder: TinyMHLut(input_dim=E, n_heads=1, n_outputs=unemb_n_sparse,
    n_anchor_pairs=unemb_nap, tph=unemb_tph, sparse_scatter_n_outputs=V), backward_mode
    = 'ste' (soft costs ~57GB on E->V; ste 6.6GB and is the validated sparse-scatter
    combo). Output [N, 1, V] -> squeeze -> logits.
  - MAIN loss = CE(unemb(pred_emb), next_token).
  - AUX  loss = CE(unemb(emb_table_subsample), ids)  -- identity regularizer: each
    token's own embedding must unembed to itself. (No assignment STE; the LUT's ste
    backward gives gradients to both the embedder and the unembedder.)

  evaluate_bpb() calls forward(x, y) -> main CE (mean), unchanged interface.
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
D           = cfg['residual_dim']          # == E
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
DEVICE_BS   = cfg['device_batch_size']
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']
_ROPE_BASE  = cfg.get('rope_base', 10000.0)
_NOISE_EPS  = cfg.get('argmax_noise_eps', 0.0)

UNEMB_NAP      = cfg.get('unemb_nap', 8)
UNEMB_TPH      = cfg.get('unemb_tph', 4096)
UNEMB_N_SPARSE = cfg.get('unemb_n_sparse', 8)
AUX_CE_W       = cfg.get('aux_ce_weight', 1.0)
AUX_BATCH      = cfg.get('aux_batch', 8192)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories (body: TinyMHLut soft) -------------------------------------
_TINY_SOFT_KWARGS = dict(
    weight_dtype=torch.float32,
    anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
    backward_mode='soft',
    soft_score_temp=cfg.get('soft_score_temp', 0.5),
    select_temp=cfg.get('select_temp', 0.5),
    learnable_temps=cfg.get('soft_learnable_temps', True),
    use_bf16=cfg.get('soft_use_bf16', True),
    argmax_noise_eps=_NOISE_EPS,
)

def _make_qkv_joint(layer_idx, seed_offset):
    kwargs = dict(_TINY_SOFT_KWARGS)
    kwargs['initial_weights_noise'] = cfg.get('qkv_lut_init_std', cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=2 * d_qk + d_v,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **kwargs)

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_out(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_residual_lut(layer_idx, seed_offset):
    return TinyMultiHeadLut(input_dim=E, n_heads=1, n_outputs=E,   # decode token-embedding delta (E)
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS)

def _make_unembedder():
    # Standard sparse-scatter LUT E -> V. ste backward (soft is ~57GB here).
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=UNEMB_N_SPARSE,
        n_anchor_pairs=UNEMB_NAP, tables_per_head=UNEMB_TPH,
        sparse_scatter_n_outputs=VOCAB_SIZE,
        sparse_scatter_seed=cfg['random_seed'] + 99999,
        weight_dtype=torch.float32,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        initial_weights_noise=cfg.get('mhlut_init_std', 0.001),
        backward_mode='ste',
        random_seed=cfg['random_seed'] + 88888, device=DEVICE)


# --- RoPE on (q, k) -----------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.qkv_lut      = _make_qkv_joint(layer_idx, layer_idx)
        self.v_lut        = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj     = _make_out(layer_idx, 400 + layer_idx)
        self.residual_lut = _make_residual_lut(layer_idx, 600 + layer_idx)
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_pre  = nn.LayerNorm(E)
        self.ln_post = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        x_pre = self.ln_pre(x_flat)
        qkv_out = self.qkv_lut(x_pre)
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
        v_branch = qkv_out[..., 2 * d_qk:]
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_vec = self.v_lut(x_pre) + v_branch
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)
        x_lut_next_flat = x_flat + out_e
        r_in  = self.ln_post(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)
        return x_lut_next_flat.reshape(B, T, E), r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.normal_(0, cfg.get('tok_emb_init', 0.5))
        self.unembedder = _make_unembedder()
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])

    def get_device(self):
        return self.tok_emb_E.weight.device

    def predict_embedding(self, tokens):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        return x_resid

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        pred_emb = self.predict_embedding(tokens)               # [B, T, E]
        B, T, _ = pred_emb.shape
        logits = self.unembedder(pred_emb.reshape(B * T, E)).squeeze(1)   # [B*T, V]
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1)
        return logits.view(B, T, VOCAB_SIZE)

    def consistency_loss(self, n_sample=None):
        """AUX identity regularizer: each token's own embedding unembeds to itself."""
        if n_sample is None or n_sample >= VOCAB_SIZE:
            ids = torch.arange(VOCAB_SIZE, device=self.get_device())
        else:
            ids = torch.randint(0, VOCAB_SIZE, (n_sample,), device=self.get_device())
        emb = self.tok_emb_E.weight[ids]
        logits = self.unembedder(emb).squeeze(1)
        aux_ce = F.cross_entropy(logits, ids)
        with torch.no_grad():
            aux_top1 = (logits.argmax(1) == ids).float().mean()
        return aux_ce, aux_top1


# --- Build + optimiser --------------------------------------------------------
model = Model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Total params (all fp32): {n_params:,}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

lut_params, tok_emb_params, unemb_params, decay_params, nodecay_params = [], [], [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'unembedder' in name:
        unemb_params.append(p)
    elif p.ndim >= 3:
        lut_params.append(p)
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)

_LUT_LR     = cfg.get('lut_lr', cfg['adam_lr'])
_UNEMB_LR   = cfg.get('unemb_lr', cfg['adam_lr'])
_TOK_EMB_LR = cfg.get('tok_emb_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=lut_params,    lr=_LUT_LR,     betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    dict(params=unemb_params,  lr=_UNEMB_LR,   betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    dict(params=tok_emb_params, lr=_TOK_EMB_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    dict(params=decay_params,  lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=nodecay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
]
optimizer = torch.optim.AdamW(adam_groups)
for g in optimizer.param_groups:
    g['initial_lr'] = g['lr']

print(f'optimizer groups: lut={sum(p.numel() for p in lut_params):,} (lr={_LUT_LR}) | '
      f'unemb={sum(p.numel() for p in unemb_params):,} (lr={_UNEMB_LR}) | '
      f'tok_emb={sum(p.numel() for p in tok_emb_params):,} (lr={_TOK_EMB_LR}) | '
      f'decay={sum(p.numel() for p in decay_params):,} | nodecay={sum(p.numel() for p in nodecay_params):,}')
print(f'D=residual_dim={D} (==E), E={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'TIED sparse unembedder: nap={UNEMB_NAP} (2^nap={1<<UNEMB_NAP} rows) tph={UNEMB_TPH} '
      f'n_sparse={UNEMB_N_SPARSE} -> sparse_scatter to V={VOCAB_SIZE} (ste); '
      f'coverage K={UNEMB_TPH*UNEMB_N_SPARSE/VOCAB_SIZE:.2f}, params={sum(p.numel() for p in unemb_params):,}')
print(f'losses: MAIN ce | AUX {AUX_CE_W}*ce (identity reg, aux_batch={AUX_BATCH})')


# --- Temperature tracking (body LUTs) -----------------------------------------
def collect_temperature_specs(model):
    specs = []
    for li, blk in enumerate(model.layers):
        for slut_name in ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut'):
            mod = getattr(blk, slut_name)
            if getattr(mod, 'learnable_temps', False):
                specs.append((f'L{li}.{slut_name}.T_soft',
                              (lambda m=mod: float(m.log_soft_score_temp.detach().exp()))))
                specs.append((f'L{li}.{slut_name}.T_sel',
                              (lambda m=mod: float(m.log_select_temp.detach().exp()))))
    return specs

temp_specs = collect_temperature_specs(model)
temp_f = open(os.path.join(EXP_DIR, 'temperatures.csv'), 'w', newline='')
temp_w = csv.writer(temp_f)
temp_w.writerow(['step'] + [name for name, _ in temp_specs])


# --- Training loop ------------------------------------------------------------
tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | '
      f'effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb', 'aux_ce', 'aux_top1'])

train_losses_logged, val_bpbs, val_steps = [], [], []
ema = None
best_bpb = float('inf')
t0 = time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step)
    for g in optimizer.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    optimizer.zero_grad()
    accum_main = 0.0
    last = {}
    for _ in range(grad_accum):
        x, y = next(train_loader)
        main_ce = model(x, targets=y)
        aux_ce, aux_top1 = model.consistency_loss(n_sample=AUX_BATCH)
        loss = main_ce + AUX_CE_W * aux_ce
        (loss / grad_accum).backward()
        accum_main += main_ce.item() / grad_accum
        last = dict(aux_ce=float(aux_ce), aux_top1=float(aux_top1))

    optimizer.step()
    ema = accum_main if ema is None else 0.99 * ema + 0.01 * accum_main

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | main={ema:.4f} | aux_ce={last["aux_ce"]:.3f} '
              f'| aux_top1={last["aux_top1"]*100:5.1f}% | lr={lr_scale*cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        if bpb < best_bpb:
            best_bpb = bpb
        print(f'[VAL] step {step}: bpb={bpb:.4f} | aux_ce={last["aux_ce"]:.3f} '
              f'| aux_top1={last["aux_top1"]*100:.1f}%')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}', f'{last["aux_ce"]:.6f}', f'{last["aux_top1"]:.6f}'])
        csv_f.flush()
        temp_w.writerow([step] + [f'{getter():.6f}' for _, getter in temp_specs]); temp_f.flush()
        model.train()

csv_f.close(); temp_f.close()
elapsed = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train main (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy', title='Main train loss'); ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, label='val bpb', color='red')
ax2.set(xlabel='step', ylabel='bpb', title='Validation BPB'); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120); plt.close(fig)

summary = dict(
    exp_name=cfg['exp_name'], best_val_bpb=best_bpb,
    final_val_bpb=val_bpbs[-1] if val_bpbs else float('nan'),
    n_params=n_params, training_time_hours=round(elapsed / 3600, 3),
)
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))

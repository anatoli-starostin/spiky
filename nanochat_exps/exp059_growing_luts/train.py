"""nanochat_exps/exp002_lut_e64_sumpos — LUT transformer (exp427-style) on nanochat data.

Same architecture as transformer_exps/exp427_E64_sumpos_full but adapted for
nanochat's BPE tokenizer (V=32768) and 512-token context. The unembedder is
the standard MLP (LN + Linear + ReLU + Linear) — final fan-out grows from 257
to 32768, dominating the Adam param budget.

How to launch:

    PYTHONPATH=/home/starost/nanochat \\
        /home/starost/spiky/.venv/bin/python \\
        -u nanochat_exps/exp002_lut_e64_sumpos/train.py \\
        > nanochat_exps/exp002_lut_e64_sumpos/stdout.log 2>&1 &
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

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT, BitMultiHeadLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector, VectorToDominance

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
TOTAL_BS    = cfg['total_batch_size']
N_STEPS     = cfg['n_steps']
EVAL_EVERY  = cfg['eval_every']
EVAL_STEPS  = cfg['eval_steps']
WARMUP_FRAC = cfg['lr_warmup_fraction']

_PER_LAYER_E = cfg['per_layer_embedding_dim']
assert len(_PER_LAYER_E) == N_LAYERS
E_TOK = _PER_LAYER_E[0]

def _e_in(layer_idx):
    return E_TOK if layer_idx == 0 else _PER_LAYER_E[layer_idx - 1]
def _e_out(layer_idx):
    return _PER_LAYER_E[layer_idx]

# Sum-mode positional embeddings (pos_emb_dim=0 → pos added to x, no concat).
_POS_EMB_CFG = cfg.get('pos_emb_dim', 0)
_POS_EMB_ACTIVE = isinstance(_POS_EMB_CFG, int) and _POS_EMB_CFG > 0
def _pos_emb_dim(layer_idx):
    return _POS_EMB_CFG if _POS_EMB_ACTIVE else _e_in(layer_idx)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT block helpers --------------------------------------------------------
def _qk_baseline_tph():
    return cfg['qk_tph']

def _make_qk_joint(layer_idx, seed_offset, tph=None):
    """Joint Q/K BitMultiHeadLUT, with per-LUT CLT scaling based on its tph."""
    if tph is None:
        tph = cfg['qk_tph']
    n_inputs = E + (_pos_emb_dim(layer_idx) if _POS_EMB_ACTIVE else 0)
    n_out = 2 * D_QK_P
    n_votes_per_pair = tph * cfg['qk_output_nap'] / float(n_out)
    qk_scale = 0.5 / math.sqrt(max(n_votes_per_pair, 1.0))
    return BitMultiHeadLUT(
        n_inputs=n_inputs,
        n_outputs=n_out,
        n_heads=H,
        input_nap=cfg['qk_input_nap'],
        output_nap=cfg['qk_output_nap'],
        tph=tph,
        scale=qk_scale,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        soft_backward=cfg.get('bit_lut_soft_backward', True),
        device=DEVICE,
    )


# --- GrowingLUT wrapper -------------------------------------------------------
class GrowingLUT(nn.Module):
    """Holds a ModuleList of bit-LUT modules; forward sums their outputs.

    New phases are added via add_phase(); existing modules continue training
    (no freezing).
    """
    def __init__(self):
        super().__init__()
        self.luts = nn.ModuleList([])

    def add_lut(self, lut):
        self.luts.append(lut)

    def forward(self, x):
        if len(self.luts) == 0:
            raise RuntimeError("GrowingLUT has no phases yet")
        out = self.luts[0](x)
        for lut in self.luts[1:]:
            out = out + lut(x)
        return out


def _add_lut_to_bit_opt(bit_opt, lut):
    """Hot-add a new bit-LUT to BitPermutationLUTOptimizer.
    Mirrors the optimizer's __init__ logic for one module.
    """
    state = {
        "mode": "bf16",
        "grad_out": None,
        "lookup_indices": None,
        "m_f32": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
        "v_f32": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
        "wg_buffer": torch.zeros_like(lut.latent_bf16, dtype=torch.float32),
    }
    bit_opt.modules.append(lut)
    bit_opt._state.append(state)
    bit_opt._handles.append(lut.register_forward_hook(bit_opt._make_hook(state)))

_V_TPH_PER_LAYER = cfg.get('v_tph_per_layer')
def _v_baseline_tph(layer_idx):
    return _V_TPH_PER_LAYER[layer_idx] if _V_TPH_PER_LAYER is not None else cfg['v_tph']

def _make_v(layer_idx, seed_offset, tph=None):
    if tph is None:
        tph = _v_baseline_tph(layer_idx)
    return BitPermutationLUT(
        n_inputs=_e_in(layer_idx), n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        device=DEVICE,
    )

_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
_OUT_OUTPUT_NAP_PER_LAYER = cfg.get('out_output_nap_per_layer')
def _out_baseline_tph(layer_idx):
    return _OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None else cfg['out_tph']

def _make_out(layer_idx, seed_offset, tph=None):
    if tph is None:
        tph = _out_baseline_tph(layer_idx)
    e_out_val = _e_out(layer_idx)
    n_pairs_max = e_out_val * (e_out_val - 1) // 2
    base_out_nap = (
        _OUT_OUTPUT_NAP_PER_LAYER[layer_idx]
        if _OUT_OUTPUT_NAP_PER_LAYER is not None
        else cfg['out_output_nap']
    )
    out_nap = min(base_out_nap, n_pairs_max)
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=e_out_val, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=out_nap,
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.e_in = _e_in(layer_idx)
        self.e_out = _e_out(layer_idx)
        self.layer_idx = layer_idx
        # GrowingLUTs start empty — initial phase added by Model.add_phase().
        self.qk_joint = GrowingLUT()
        self.v_perm = GrowingLUT()
        self.out_proj = GrowingLUT()

        canon_t = cfg.get('canon_temperature', 0.1)
        # Canonicalise q and k pair-dominance after splitting BitMultiHeadLUT output.
        self.q_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.k_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        # out_canon — leak fix from exp377/exp390/exp427.
        self.out_canon = DominanceCanonicalize(self.e_out, temperature=canon_t)

        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_to_vec = DominanceToVector(self.e_out)

        self.pos_dim = _pos_emb_dim(layer_idx)
        if _POS_EMB_ACTIVE:
            self.qk_input_ln = nn.LayerNorm(self.e_in + self.pos_dim)

        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, x, pos_emb):
        B, T, _E_in = x.shape
        if _POS_EMB_ACTIVE:
            pos = pos_emb.unsqueeze(0).expand(B, -1, -1)
            xp = torch.cat([x, pos], dim=-1)
            xp = self.qk_input_ln(xp).reshape(B * T, _E_in + self.pos_dim)
        else:
            xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E_in)
        x_flat = x.reshape(B * T, _E_in)

        qk_out = self.qk_joint(xp)                                     # [B*T, H, 2*P_qk]
        q_raw = qk_out[..., :D_QK_P]                                   # [B*T, H, P_qk] (raw dominance)
        k_raw = qk_out[..., D_QK_P:]
        q_dom = self.q_canon(q_raw)                                    # canonicalised pair-dominance
        k_dom = self.k_canon(k_raw)
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)

        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)

        attn_dom = F.scaled_dot_product_attention(
            q * self.attn_scale, k, v_dom,
            is_causal=True,
        )
        attn = self.attn_to_vec(attn_dom)

        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_proj(out_in)
        out_dom = self.out_canon(out_dom)
        out = self.out_to_vec(out_dom).squeeze(1).reshape(B, T, self.e_out)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E_TOK)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        _pos_dim_fn = _pos_emb_dim if _POS_EMB_ACTIVE else _e_in
        _pos_init_scale = cfg.get('pos_emb_init_scale', 0.1)
        self.pos_embs = nn.ParameterList([
            nn.Parameter(torch.randn(CONTEXT_SIZE, _pos_dim_fn(i)) * _pos_init_scale)
            for i in range(N_LAYERS)
        ])
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = sum(_PER_LAYER_E)
        # Minimal unembedder: LN + single Linear. Drops the hidden layer of
        # exp002's 2-layer MLP. Param budget: 384·32768 + 32768 ≈ 12.6M (vs
        # exp002's 51M, comparable to exp001 vanilla's 12.58M tied head).
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, VOCAB_SIZE),
        )

    def get_device(self):
        return self.token_embedder.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        # Same shape as exp427: token_embed → 6 LUT layers (sum-mode pos),
        # collect each layer's output → concat across layers → MLP unembedder.
        x = self.token_embedder(tokens)                       # [B, T, E_TOK]
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)                      # [B, T, sum(E)]
        logits = self.unembedder(concat)                      # [B, T, V]
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


# --- Build + optimisers -------------------------------------------------------
model = Model().to(DEVICE)

# Growth schedule: list of [start_step, fraction]. At each entry, a new LUT
# is added to every layer's qk_joint, v_perm, out_proj GrowingLUT, with tph =
# round(baseline_tph * fraction). Fractions ADD on top — total tph at the end
# of training = sum of all fractions × baseline.
GROWTH_SCHEDULE = cfg.get('growth_schedule', [[0, 1.0]])

def _add_growth_phase(model, frac, phase_idx):
    """Add a new LUT to each layer's three GrowingLUTs at given fraction.
    Returns a list of newly-created BitMultiHeadLUT/BitPermutationLUT modules.
    """
    new_luts = []
    for layer in model.layers:
        layer_idx = layer.layer_idx
        # qk
        qk_tph_phase = max(1, int(round(cfg['qk_tph'] * frac)))
        qk_seed = cfg['random_seed'] + 1000 * phase_idx + layer_idx
        new_qk = _make_qk_joint(layer_idx, qk_seed, tph=qk_tph_phase).to(DEVICE)
        layer.qk_joint.add_lut(new_qk)
        new_luts.append(new_qk)
        # v
        v_tph_phase = max(1, int(round(_v_baseline_tph(layer_idx) * frac)))
        v_seed = cfg['random_seed'] + 1000 * phase_idx + 200 + layer_idx
        new_v = _make_v(layer_idx, v_seed, tph=v_tph_phase).to(DEVICE)
        layer.v_perm.add_lut(new_v)
        new_luts.append(new_v)
        # out
        out_tph_phase = max(1, int(round(_out_baseline_tph(layer_idx) * frac)))
        out_seed = cfg['random_seed'] + 1000 * phase_idx + 400 + layer_idx
        new_out = _make_out(layer_idx, out_seed, tph=out_tph_phase).to(DEVICE)
        layer.out_proj.add_lut(new_out)
        new_luts.append(new_out)
    return new_luts

# Add the initial phase before building optimizers
INITIAL_FRAC = GROWTH_SCHEDULE[0][1]
print(f'Adding initial phase 0 with fraction={INITIAL_FRAC}')
_add_growth_phase(model, INITIAL_FRAC, phase_idx=0)

bit_luts = []
for layer in model.layers:
    for sub in [layer.qk_joint, layer.v_perm, layer.out_proj]:
        for lut in sub.luts:
            bit_luts.append(lut)

adam_params = list(model.parameters())
adam_param_count = sum(p.numel() for p in adam_params)
print(f'bit LUTs (initial): {len(bit_luts)}')
print(f'Adam-managed parameters: {adam_param_count:,}')

def get_lr_scale(step):
    n = N_STEPS
    w = int(WARMUP_FRAC * n)
    if step < w:
        return step / max(w, 1)
    progress = (step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

# AdamW with two groups (decay/nodecay), matching exp001 / nanochat MinimalGPT.
# Bit-LUT latents are NOT touched by this — they're owned by bit_opt below.
decay_params   = [p for p in adam_params if p.ndim >= 2]
nodecay_params = [p for p in adam_params if p.ndim < 2]
adam_groups = [
    dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=cfg.get('weight_decay', 0.0)),
    dict(params=nodecay_params, lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
         weight_decay=0.0),
]
adam_opt = torch.optim.AdamW(adam_groups)
for g in adam_opt.param_groups:
    g['initial_lr'] = g['lr']

bit_opt = BitPermutationLUTOptimizer(
    bit_luts,
    lr=cfg['bit_lut_lr'],
    beta1=cfg.get('bit_lut_beta1', 0.9),
    beta2=cfg.get('bit_lut_beta2', 0.999),
    lr_schedule_fn=get_lr_scale,
)

print(f'Q/K Joint BitMultiHeadLUT: in_nap={cfg["qk_input_nap"]} out_nap={cfg["qk_output_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} P_qk={D_QK_P} (n_outputs=2*P_qk={2*D_QK_P}, CLT-scaled, q/k-canonicalised)')
_v_tph_str = str(_V_TPH_PER_LAYER) if _V_TPH_PER_LAYER is not None else str(cfg['v_tph'])
print(f'V BitPermLUT:   in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={_v_tph_str} d_v={d_v} P_v={D_V_P}')
_tph_str = str(_OUT_TPH_PER_LAYER) if _OUT_TPH_PER_LAYER is not None else str(cfg['out_tph'])
print(f'Out BitPermLUT: in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={_tph_str}')

# --- Training loop ------------------------------------------------------------
tokens_per_step = DEVICE_BS * CONTEXT_SIZE
grad_accum = max(1, TOTAL_BS // tokens_per_step)
print(f'Tokens/micro-batch: {tokens_per_step:,} | grad_accum: {grad_accum} | effective batch: {grad_accum * tokens_per_step:,} tokens')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_bpb'])

train_losses_logged, val_bpbs, val_steps = [], [], []
ema = None
best_bpb = float('inf')
t0 = time.time()

_NEXT_PHASE_IDX = 1  # phase 0 is the initial; subsequent phases triggered in loop

model.train()
for step in range(1, N_STEPS + 1):
    # Phase-addition trigger: when step crosses a schedule boundary, add a new
    # phase to every GrowingLUT and register the new LUTs with both optimizers.
    while _NEXT_PHASE_IDX < len(GROWTH_SCHEDULE) and step >= GROWTH_SCHEDULE[_NEXT_PHASE_IDX][0]:
        sched_step, frac = GROWTH_SCHEDULE[_NEXT_PHASE_IDX]
        new_luts = _add_growth_phase(model, frac, phase_idx=_NEXT_PHASE_IDX)
        for lut in new_luts:
            _add_lut_to_bit_opt(bit_opt, lut)
        # New LUTs have no Adam params (bit-LUTs only), so adam_opt unchanged.
        total_phases = _NEXT_PHASE_IDX + 1
        n_active = len(bit_opt.modules)
        print(f'[phase {_NEXT_PHASE_IDX}] step {step}: added {len(new_luts)} new bit-LUTs at frac={frac} (total phases={total_phases}, total bit-LUTs={n_active})')
        _NEXT_PHASE_IDX += 1

    # Set LR for the AdamW path BEFORE opt.step() — matches exp001's order so
    # iteration 1 already trains at lr_scale = 1/warmup_steps (not 0).
    lr_scale = get_lr_scale(step)
    for g in adam_opt.param_groups:
        g['lr'] = g['initial_lr'] * lr_scale

    adam_opt.zero_grad()
    bit_opt.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        x, y = next(train_loader)
        loss = model(x, targets=y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum

    adam_opt.step()
    bit_opt.step()      # bit-LUT optimizer applies its own internal lr_schedule_fn

    ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        val_loader = val_loader_factory()
        bpb = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
        if bpb < best_bpb:
            best_bpb = bpb
        print(f'[VAL] step {step}: bpb={bpb:.4f}')
        train_losses_logged.append(ema)
        val_bpbs.append(bpb)
        val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}'])
        csv_f.flush()
        model.train()

csv_f.close()
elapsed = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(val_steps, train_losses_logged, label='train (ema)')
ax1.set(xlabel='step', ylabel='cross-entropy loss', title='Training Loss')
ax1.legend(); ax1.grid(True)
ax2.plot(val_steps, val_bpbs, 'o-', color='tab:orange', label='val bpb')
ax2.set(xlabel='step', ylabel='bits per byte', title='Validation BPB')
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=120)
plt.close()

summary = {
    'exp_name': cfg['exp_name'],
    'best_val_bpb': best_bpb,
    'final_val_bpb': val_bpbs[-1] if val_bpbs else None,
    'adam_params': adam_param_count,
    'n_bit_luts': len(bit_luts),
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

bit_opt.close()
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))

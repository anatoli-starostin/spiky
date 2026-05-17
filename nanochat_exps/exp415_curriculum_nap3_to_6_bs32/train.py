"""exp415 — NAP=3 → NAP=6 curriculum on the exp414 architecture.

Stage 0: NAP=3 with 2× tph everywhere (per `tph_mult` in stage spec),
         bs=4, 4000 steps, cosine LR with `warmup_fraction` (default 10%).
Stage 1: NAP=6 with target tph (exp414 base, 31.53M params),
         bs=32, 3500 steps, RESET LR, NO warmup, pure cosine anneal.

Weight transfer between stages:
  - LUT weights merged via outer-add (anchor_tree.merge_weight_tensor).
    Stage-0 has 2× tables of stage 1; we pair (2i, 2i+1) → stage-1 table i.
  - Non-LUT params (LayerNorm affines, biases, log_temps, tok_emb_E,
    unembedder) copied directly.

Adam state transfer between stages (LUT params with shape change):
  AVG merge of (exp_avg, exp_avg_sq) — same outer-add formula scaled by 0.5.
  Rationale: each child tracks the FULL upstream gradient, so the parent
  should NOT be 2× their magnitudes — hence the 0.5 factor.

Anchor mapping for curriculum modules (all four LUTs):
  Stage 1 (target NAP=6): N tables, each with 6 anchor pairs.
  Stage 0 (leaf NAP=3): 2N tables. Leaf table 2i = first 3 pairs of stage-1
  table i; leaf table 2i+1 = last 3 pairs of stage-1 table i.
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
sys.path.insert(0, EXP_DIR)
from anchor_tree import build_two_level_anchors, merge_weight_tensor

with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg['random_seed'])

CONTEXT_SIZE = cfg['context_size']
E           = cfg['embedding_dim']
D           = cfg['residual_dim']
H           = cfg['n_heads']
d_qk        = cfg['d_qk']
d_v         = cfg['d_v']
N_LAYERS    = cfg['num_layers']
_ROPE_BASE  = cfg.get('rope_base', 10000.0)
_NOISE_EPS  = cfg.get('argmax_noise_eps', 0.0)

TARGET_NAP  = cfg['target_nap']
TARGET_TPH  = cfg['target_tph']  # dict per module
STAGES      = cfg['stages']

# All four LUTs participate in the curriculum.
CURRICULUM_MODULES = ('qkv_lut', 'v_lut', 'out_proj', 'residual_lut')


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')
token_bytes = get_token_bytes(device=DEVICE)


# --- Anchor trees (2-level: target NAP=6 → leaf NAP=3) -----------------------
_MODULE_DIMS = {
    # name: (input_dim, n_heads, n_outputs)
    'qkv_lut':      (E,       H, 2 * d_qk + d_v),
    'v_lut':        (E,       H, d_v),
    'out_proj':     (H * d_v, 1, E),
    'residual_lut': (E,       1, D),
}


def build_all_anchor_trees():
    """Build 2-level anchor trees for every curriculum module."""
    trees = {}
    print(f'Building 2-level anchor trees (target NAP={TARGET_NAP} → leaf NAP={TARGET_NAP//2})...')
    seed_offset = {'qkv_lut': 1, 'v_lut': 2, 'out_proj': 3, 'residual_lut': 4}
    for name in CURRICULUM_MODULES:
        input_dim, n_heads, _ = _MODULE_DIMS[name]
        n_target_tables = n_heads * TARGET_TPH[name]
        trees[name] = build_two_level_anchors(
            input_dim=input_dim,
            n_target_tables=n_target_tables,
            target_nap=TARGET_NAP,
            n_heads=n_heads,
            random_seed=cfg['random_seed'] + seed_offset[name],
            device=DEVICE,
        )
        tgt = trees[name]['target']
        leaf = trees[name]['leaf']
        print(f'  {name}: target n_tables={tgt["n_tables"]} (NAP={tgt["nap"]})  →  leaf n_tables={leaf["n_tables"]} (NAP={leaf["nap"]})')
    return trees


# --- Soft-kwargs for TinyMHLut (constant across stages) ----------------------
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


def _make_lut(input_dim, n_heads, n_outputs, n_anchor_pairs, tph, random_seed,
              init_std_override=None):
    kwargs = dict(_TINY_SOFT_KWARGS)
    if init_std_override is not None:
        kwargs['initial_weights_noise'] = init_std_override
    return TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tph,
        random_seed=random_seed, device=DEVICE,
        **kwargs,
    )


def _override_anchor_buffers(lut_mod, anchor_a_int64, anchor_b_int64):
    """Overwrite the anchor pair buffers in a TinyMultiHeadLut with tree-derived
    pairs. Mutates the module in-place."""
    device = lut_mod.weights.device
    a = anchor_a_int64.to(device=device)
    b = anchor_b_int64.to(device=device)
    lut_mod.lookup.anchor_pairs_a.copy_(a.to(torch.int16))
    lut_mod.lookup.anchor_pairs_b.copy_(b.to(torch.int16))
    lut_mod.soft_anchor_a_long.copy_(a.to(torch.int64).contiguous())
    lut_mod.soft_anchor_b_long.copy_(b.to(torch.int64).contiguous())


# --- RoPE --------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0, device=None):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        ))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)


def _rotate_half(t):
    a, b = t.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


# --- Stage-aware Model -------------------------------------------------------
def _per_module_for_stage(stage_idx):
    """Returns dict {module: {'nap', 'tph'}} for this stage.
    tph is per-head (LUTBlock multiplies by n_heads internally)."""
    st = STAGES[stage_idx]
    cur_nap = st['curriculum_nap']
    mult    = st['tph_mult']
    return {name: {'nap': cur_nap, 'tph': TARGET_TPH[name] * mult}
            for name in CURRICULUM_MODULES}


class LUTBlock(nn.Module):
    def __init__(self, layer_idx, per_module):
        super().__init__()
        seed = cfg['random_seed']
        qkv_std = cfg.get('qkv_lut_init_std', cfg.get('mhlut_init_std', 0.001))
        self.qkv_lut = _make_lut(
            input_dim=E, n_heads=H, n_outputs=2 * d_qk + d_v,
            n_anchor_pairs=per_module['qkv_lut']['nap'],
            tph=per_module['qkv_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 1,
            init_std_override=qkv_std,
        )
        self.v_lut = _make_lut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=per_module['v_lut']['nap'],
            tph=per_module['v_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 2,
        )
        self.out_proj = _make_lut(
            input_dim=H * d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=per_module['out_proj']['nap'],
            tph=per_module['out_proj']['tph'],
            random_seed=seed + 100 * layer_idx + 3,
        )
        self.residual_lut = _make_lut(
            input_dim=E, n_heads=1, n_outputs=D,
            n_anchor_pairs=per_module['residual_lut']['nap'],
            tph=per_module['residual_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 4,
        )
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_e   = nn.LayerNorm(E)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, E)
        qkv_out = self.qkv_lut(x_flat)
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
        v_branch = qkv_out[..., 2 * d_qk:]
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        v_lut_out = self.v_lut(x_flat)
        v_vec = v_lut_out + v_branch
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)
        out_e_norm = self.ln_e(out_e)
        x_lut_next = out_e_norm.reshape(B, T, E)
        r_out = self.residual_lut(out_e_norm).squeeze(1).reshape(B, T, D)
        return x_lut_next, r_out


class Model(nn.Module):
    def __init__(self, per_module):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.uniform_(-0.1, 0.1)
        self.unembedder = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i, per_module) for i in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D)

    def get_device(self):
        return self.tok_emb_E.weight.device

    def forward(self, tokens, targets=None, loss_reduction='mean'):
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        x_resid = self.ln_final(x_resid)
        logits = self.unembedder(x_resid)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1,
            )
        return logits


def apply_anchor_trees_to_model(model, trees, stage_idx):
    """Override anchor buffers for every curriculum module using the
    appropriate level of the 2-level tree."""
    level_key = 'leaf' if stage_idx == 0 else 'target'
    for layer in model.layers:
        for name in CURRICULUM_MODULES:
            mod = getattr(layer, name)
            entry = trees[name][level_key]
            _override_anchor_buffers(mod, entry['anchor_a'], entry['anchor_b'])


def merge_weights_from_prev_stage(model, prev_state_dict, prev_nap):
    """Apply merge ONLY where shape changed (curriculum modules, all four
    LUTs); direct copy for everything else."""
    new_state = {}
    for name, p in model.named_parameters():
        if name not in prev_state_dict:
            continue
        prev_p = prev_state_dict[name]
        if p.ndim == 3 and name.endswith('.weights') and prev_p.shape != p.shape:
            merged = merge_weight_tensor(prev_p, nap_prev=prev_nap)
            new_state[name] = merged.to(p.device).to(p.dtype)
        elif prev_p.shape == p.shape:
            new_state[name] = prev_p.to(p.device).to(p.dtype)
        else:
            print(f'  WARNING: shape mismatch for {name}: prev={prev_p.shape} new={p.shape}')
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


def inject_merged_adam_state(optimizer, prev_optim_state, prev_params, prev_nap):
    """Carry Adam moments from prev stage. For LUT params (shape changed),
    apply outer-add merge with 0.5 averaging factor on (m, v). For other
    params, copy directly."""
    prev_state = prev_optim_state['state']
    new_params_flat = [p for g in optimizer.param_groups for p in g['params']]
    for idx, (new_p, prev_p) in enumerate(zip(new_params_flat, prev_params)):
        if idx not in prev_state:
            continue
        s = prev_state[idx]
        prev_m = s.get('exp_avg')
        prev_v = s.get('exp_avg_sq')
        prev_step = s.get('step')
        if prev_m is None or prev_v is None:
            continue
        if new_p.ndim == 3 and new_p.shape != prev_p.shape:
            new_m = (merge_weight_tensor(prev_m.to(DEVICE), nap_prev=prev_nap) * 0.5).to(new_p.dtype)
            new_v = (merge_weight_tensor(prev_v.to(DEVICE), nap_prev=prev_nap) * 0.5).to(new_p.dtype)
        else:
            new_m = prev_m.to(new_p.device).to(new_p.dtype)
            new_v = prev_v.to(new_p.device).to(new_p.dtype)
        optimizer.state[new_p] = {
            'step': prev_step.clone() if isinstance(prev_step, torch.Tensor) else torch.tensor(int(prev_step)),
            'exp_avg': new_m,
            'exp_avg_sq': new_v,
        }


def get_stage_lr_scale(stage_step, n_steps, warmup_fraction):
    """Per-stage LR schedule. warmup_fraction=0 means pure cosine anneal
    starting at peak LR (used for stage 2)."""
    w = int(warmup_fraction * n_steps)
    if w > 0 and stage_step < w:
        return stage_step / max(w, 1)
    progress = (stage_step - w) / max(n_steps - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def train_stage(stage_idx, model, bs, n_steps, warmup_fraction,
                csv_writer, csv_file, val_csv_writer, val_csv_file,
                prev_optim_state=None, prev_params=None, prev_nap=None):
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, bs, CONTEXT_SIZE, split='train', device=DEVICE)
    val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, bs, CONTEXT_SIZE, split='val', device=DEVICE)

    lut_params, decay_params, nodecay_params, tok_emb_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 3:
            lut_params.append(p)
        elif name.startswith('tok_emb_E.'):
            tok_emb_params.append(p)
        elif p.ndim == 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    _LUT_LR = cfg.get('lut_lr', cfg['adam_lr'])
    adam_groups = [
        dict(params=lut_params,     lr=_LUT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(params=decay_params,   lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8,
             weight_decay=cfg.get('weight_decay', 0.0)),
        dict(params=tok_emb_params + nodecay_params,
             lr=cfg['adam_lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ]
    optimizer = torch.optim.AdamW(adam_groups)
    for g in optimizer.param_groups:
        g['initial_lr'] = g['lr']

    if prev_optim_state is not None and prev_params is not None:
        print(f'  Injecting merged Adam state from prev stage (NAP={prev_nap})')
        inject_merged_adam_state(optimizer, prev_optim_state, prev_params, prev_nap)
        print(f'    state populated for {len(optimizer.state)} params')

    EVAL_EVERY = cfg['eval_every']
    EVAL_STEPS = cfg['eval_steps']

    ema = None
    model.train()
    t0 = time.time()
    for step in range(1, n_steps + 1):
        lr_scale = get_stage_lr_scale(step, n_steps, warmup_fraction)
        for g in optimizer.param_groups:
            g['lr'] = g['initial_lr'] * lr_scale
        optimizer.zero_grad()
        x, y = next(train_loader)
        loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        accum_loss = loss.item()
        ema = accum_loss if ema is None else 0.99 * ema + 0.01 * accum_loss
        if step % 100 == 0 or step == 1:
            print(f'  stage{stage_idx} step {step:6d} | loss={ema:.4f} | lr={lr_scale * cfg["adam_lr"]:.2e}')
            csv_writer.writerow([stage_idx, step, f'{ema:.6f}'])
            csv_file.flush()
        if step % EVAL_EVERY == 0 or step == n_steps:
            model.eval()
            val_loader = val_loader_factory()
            bpb = evaluate_bpb(model, val_loader, EVAL_STEPS, token_bytes)
            print(f'[VAL] stage{stage_idx} step {step}: bpb={bpb:.4f}')
            val_csv_writer.writerow([stage_idx, step, f'{bpb:.6f}'])
            val_csv_file.flush()
            model.train()
    print(f'  stage{stage_idx} done in {(time.time() - t0) / 60:.1f} min')
    return optimizer.state_dict(), [p for g in optimizer.param_groups for p in g['params']]


# ============================================================================
# Main orchestrator
# ============================================================================
print('=== NAP=3 → NAP=6 curriculum (exp415) ===')
print(f'Target arch: NAP={TARGET_NAP}, tph={TARGET_TPH}')
print(f'Curriculum modules: {CURRICULUM_MODULES}')
print(f'Stages:')
for i, st in enumerate(STAGES):
    print(f'  Stage {i}: NAP={st["curriculum_nap"]}, tph_mult={st["tph_mult"]}, bs={st["bs"]}, steps={st["n_steps"]}, warmup={st["warmup_fraction"]}')

trees = build_all_anchor_trees()

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
val_csv_path = os.path.join(EXP_DIR, 'val_metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['stage', 'step', 'train_loss'])
val_csv_f = open(val_csv_path, 'w', newline='')
val_csv_w = csv.writer(val_csv_f); val_csv_w.writerow(['stage', 'step', 'val_bpb'])

prev_optim_state = None
prev_params = None
prev_nap = None
final_bpbs = []
for stage_idx, st in enumerate(STAGES):
    cur_nap = st['curriculum_nap']
    bs = st['bs']
    n_steps = st['n_steps']
    warmup_fraction = st['warmup_fraction']
    per_module = _per_module_for_stage(stage_idx)
    print(f'\n=== Stage {stage_idx}: NAP={cur_nap}, bs={bs}, n_steps={n_steps}, warmup={warmup_fraction} ===')
    print(f'  per_module: {per_module}')

    model = Model(per_module=per_module).to(DEVICE)
    apply_anchor_trees_to_model(model, trees, stage_idx=stage_idx)
    n_params = sum(p.numel() for p in model.parameters())
    n_lut = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 3)
    print(f'  Total params: {n_params:,} (LUT: {n_lut:,})')

    if stage_idx > 0:
        prev_ckpt_path = os.path.join(EXP_DIR, f'stage{stage_idx-1}.pt')
        print(f'  Merging weights from {prev_ckpt_path} (prev NAP={prev_nap})')
        prev_ckpt = torch.load(prev_ckpt_path, map_location=DEVICE, weights_only=False)
        prev_model_state = prev_ckpt['model']
        missing, unexpected = merge_weights_from_prev_stage(model, prev_model_state, prev_nap)
        print(f'  merge load: missing={len(missing)}, unexpected={len(unexpected)}')

    new_optim_state, new_params = train_stage(
        stage_idx, model, bs=bs, n_steps=n_steps, warmup_fraction=warmup_fraction,
        csv_writer=csv_w, csv_file=csv_f,
        val_csv_writer=val_csv_w, val_csv_file=val_csv_f,
        prev_optim_state=prev_optim_state,
        prev_params=prev_params,
        prev_nap=prev_nap,
    )

    ckpt_path = os.path.join(EXP_DIR, f'stage{stage_idx}.pt')
    torch.save({'model': model.state_dict(), 'optim': new_optim_state}, ckpt_path)
    print(f'  Saved {ckpt_path}')

    prev_optim_state = new_optim_state
    prev_params = new_params
    prev_nap = cur_nap

    del model
    torch.cuda.empty_cache()

csv_f.close()
val_csv_f.close()

print('\n=== CURRICULUM DONE ===')

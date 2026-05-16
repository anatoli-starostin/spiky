"""4-stage NAP curriculum training.

Stages: NAP=1 (bs=2) → NAP=2 (bs=4) → NAP=4 (bs=8) → NAP=8 (bs=16).
Each stage trains for `stage_n_steps` (equal across stages = 4000 each).
Between stages: weight transfer via additive merge formula
  parent[bits_A, bits_B, :] = child_A[bits_A, :] + child_B[bits_B, :]

Anchor pairs are pre-built at the start via top-down derivation from the
target (NAP=8) architecture, ensuring consistent ancestry across stages.

Applies to all 4 LUT modules (qkv_lut, v_lut, out_proj, residual_lut).
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

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy

from anchor_tree import build_anchor_tree, merge_weight_tensor


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
WARMUP_FRAC = cfg['lr_warmup_fraction']

# Per-stage schedule
STAGES = cfg['stages']  # list of {'nap': int, 'bs': int, 'n_steps': int}
N_STAGES = len(STAGES)

# Target architecture (final stage, NAP=8 throughout)
TARGET_NAP = cfg['target_nap']         # 8 (max NAP across all modules in this curriculum)
TARGET_TPH = cfg['target_tph']         # baseline tph dict for non-curriculum modules
CURRICULUM_MODULES = set(cfg.get('curriculum_modules', []))  # e.g. {'out_proj'}
BASELINE_PER_MODULE = cfg.get('baseline_per_module', {})     # {'qkv': {'nap': 6, 'tph': 16}, ...}
import math
N_TREE_LEVELS = int(math.log2(TARGET_NAP)) + 1   # 4 levels for NAP=8


# --- Tokenizer ---------------------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')
token_bytes = get_token_bytes(device=DEVICE)


# Per-module fixed specs (input/output dims, n_heads)
_MODULE_SPECS = {
    'qkv_lut':      dict(input_dim=E,         n_heads=H, n_outputs=2 * d_qk + d_v),
    'v_lut':        dict(input_dim=E,         n_heads=H, n_outputs=d_v),
    'out_proj':     dict(input_dim=H * d_v,   n_heads=1, n_outputs=E),
    'residual_lut': dict(input_dim=E,         n_heads=1, n_outputs=D),
}


# --- Build anchor trees (one per LUT module) ---------------------------------
def build_all_anchor_trees():
    """Per-module anchor trees, top-down from final NAP=8 architecture.
    Trees have N_TREE_LEVELS levels: tree[0]=root (NAP=target), tree[L-1]=leaves (NAP=1).
    Stage_idx → tree_level mapping: tree_level = N_TREE_LEVELS - 1 - stage_idx.
    """
    trees = {}
    print(f'Building anchor trees (target NAP={TARGET_NAP}, {N_TREE_LEVELS} levels)...')
    trees['qkv_lut'] = build_anchor_tree(
        input_dim=E, n_target_tables=H * TARGET_TPH['qkv'],
        target_nap=TARGET_NAP, n_heads=H,
        random_seed=cfg['random_seed'] + 1, device=DEVICE)
    trees['v_lut'] = build_anchor_tree(
        input_dim=E, n_target_tables=H * TARGET_TPH['v_lut'],
        target_nap=TARGET_NAP, n_heads=H,
        random_seed=cfg['random_seed'] + 2, device=DEVICE)
    trees['out_proj'] = build_anchor_tree(
        input_dim=H * d_v, n_target_tables=TARGET_TPH['out_proj'],
        target_nap=TARGET_NAP, n_heads=1,
        random_seed=cfg['random_seed'] + 3, device=DEVICE)
    trees['residual_lut'] = build_anchor_tree(
        input_dim=E, n_target_tables=TARGET_TPH['residual_lut'],
        target_nap=TARGET_NAP, n_heads=1,
        random_seed=cfg['random_seed'] + 4, device=DEVICE)
    for name, tree in trees.items():
        print(f'  {name}: root n_tables={tree[0]["n_tables"]} (NAP={tree[0]["nap"]})  →  leaves n_tables={tree[N_TREE_LEVELS-1]["n_tables"]} (NAP=1)')
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
    argmax_noise_eps=cfg.get('argmax_noise_eps', 0.0),
)


def _make_lut(input_dim, n_heads, n_outputs, n_anchor_pairs, n_tables_total,
              random_seed):
    """Build a TinyMultiHeadLut with explicit n_tables_total = n_heads × tables_per_head.
    Caller derives tables_per_head from desired n_tables_total."""
    assert n_tables_total % n_heads == 0, \
        f"n_tables_total={n_tables_total} not divisible by n_heads={n_heads}"
    tph = n_tables_total // n_heads
    return TinyMultiHeadLut(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tph,
        random_seed=random_seed, device=DEVICE,
        **_TINY_SOFT_KWARGS,
    )


def _override_anchor_buffers(lut_mod, anchor_a_int64, anchor_b_int64):
    """Overwrite the anchor pair buffers in a TinyMultiHeadLut with tree-derived
    pairs. Mutates the module in-place."""
    # The lookup module has int16 anchor_pairs_a/_b; TinyMHLut has int64 caches
    # soft_anchor_a_long/_b_long. Overwrite both.
    device = lut_mod.weights.device
    a = anchor_a_int64.to(device=device)
    b = anchor_b_int64.to(device=device)
    # int16 buffers in lookup
    lut_mod.lookup.anchor_pairs_a.copy_(a.to(torch.int16))
    lut_mod.lookup.anchor_pairs_b.copy_(b.to(torch.int16))
    # int64 caches in TinyMHLut
    lut_mod.soft_anchor_a_long.copy_(a.to(torch.int64).contiguous())
    lut_mod.soft_anchor_b_long.copy_(b.to(torch.int64).contiguous())


# --- RoPE (unchanged from exp365) --------------------------------------------
_ROPE_BASE = cfg.get('rope_base', 10000.0)


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


# --- Model -------------------------------------------------------------------
def _per_module_for_stage(stage_idx):
    """Returns dict {module_name: {'nap': int, 'tph': int}} for this stage.

    Modules NOT in CURRICULUM_MODULES use their BASELINE_PER_MODULE
    config (same in every stage). Modules in CURRICULUM_MODULES use the
    stage's 'curriculum_nap' and tph derived from TARGET_TPH × scaling.
    """
    cur_nap = STAGES[stage_idx]['curriculum_nap']
    # Tables-multiplier so that curriculum module ends up at TARGET_TPH at
    # the final stage. mult = 2^(log2(TARGET_NAP/cur_nap)) for binary merge.
    mult = TARGET_NAP // cur_nap
    out = {}
    for name in _MODULE_SPECS:
        if name in CURRICULUM_MODULES:
            tph_target = TARGET_TPH[name if name != 'qkv_lut' else 'qkv']
            out[name] = {'nap': cur_nap, 'tph': tph_target * mult}
        else:
            spec = BASELINE_PER_MODULE[name if name != 'qkv_lut' else 'qkv']
            out[name] = {'nap': spec['nap'], 'tph': spec['tph']}
    return out


class LUTBlock(nn.Module):
    """LUT module shapes (NAP, tph) are determined PER MODULE by the
    current stage. Modules outside CURRICULUM_MODULES use their baseline
    NAP/tph in every stage."""
    def __init__(self, layer_idx, per_module):
        super().__init__()
        seed = cfg['random_seed']
        self.qkv_lut = _make_lut(
            input_dim=E, n_heads=H, n_outputs=2 * d_qk + d_v,
            n_anchor_pairs=per_module['qkv_lut']['nap'],
            n_tables_total=H * per_module['qkv_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 1,
        )
        self.v_lut = _make_lut(
            input_dim=E, n_heads=H, n_outputs=d_v,
            n_anchor_pairs=per_module['v_lut']['nap'],
            n_tables_total=H * per_module['v_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 2,
        )
        self.out_proj = _make_lut(
            input_dim=H * d_v, n_heads=1, n_outputs=E,
            n_anchor_pairs=per_module['out_proj']['nap'],
            n_tables_total=per_module['out_proj']['tph'],
            random_seed=seed + 100 * layer_idx + 3,
        )
        self.residual_lut = _make_lut(
            input_dim=E, n_heads=1, n_outputs=D,
            n_anchor_pairs=per_module['residual_lut']['nap'],
            n_tables_total=per_module['residual_lut']['tph'],
            random_seed=seed + 100 * layer_idx + 4,
        )
        self.q_norm = nn.LayerNorm(d_qk)
        self.k_norm = nn.LayerNorm(d_qk)
        self.ln_e = nn.LayerNorm(E)

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


def apply_anchor_trees_to_model(model, trees, nap_for_curriculum):
    """Override anchor buffers ONLY for curriculum modules.
    Non-curriculum modules use their default random-seeded anchor sampling
    (same across stages, so weights can be copied directly)."""
    tree_level = int(math.log2(TARGET_NAP // nap_for_curriculum))
    for layer in model.layers:
        for name in CURRICULUM_MODULES:
            mod = getattr(layer, name)
            tree = trees[name][tree_level]
            _override_anchor_buffers(mod, tree['anchor_a'], tree['anchor_b'])


def merge_weights_from_prev_stage(model, prev_state_dict, prev_nap):
    """Apply merge ONLY where shape changed (curriculum modules); direct
    copy for everything else."""
    new_state = {}
    for name, p in model.named_parameters():
        if name not in prev_state_dict:
            continue
        prev_p = prev_state_dict[name]
        if p.ndim == 3 and name.endswith('.weights') and prev_p.shape != p.shape:
            # LUT weights with shape change → merge
            merged = merge_weight_tensor(prev_p, nap_prev=prev_nap)
            new_state[name] = merged.to(p.device).to(p.dtype)
        elif prev_p.shape == p.shape:
            # Shape unchanged → copy directly
            new_state[name] = prev_p.to(p.device).to(p.dtype)
        else:
            print(f'  WARNING: shape mismatch for {name} but not a 3D LUT weight: prev={prev_p.shape} new={p.shape}')
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


# --- Per-stage trainer -------------------------------------------------------
TOTAL_N_STEPS = sum(s['n_steps'] for s in STAGES)

def get_stage_lr_scale(stage_step, stage_n_steps):
    """Per-stage cosine with 10% warmup at start of each stage, cosine anneal
    to 0.1× peak by end of stage. Each stage has its own independent schedule.
    """
    n = stage_n_steps
    w = int(WARMUP_FRAC * n)
    if stage_step < w:
        return stage_step / max(w, 1)
    progress = (stage_step - w) / max(n - w, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def inject_merged_adam_state(optimizer, prev_optim_state, prev_params, prev_nap):
    """Inject merged Adam state from prev stage's optimizer into the new one.

    For LUT params (shape [N, 2^NAP, n_out] → [N/2, 2^(2·NAP), n_out]):
      merge exp_avg and exp_avg_sq via the same additive formula as weights.
    For non-LUT params: direct copy.
    `step` counter is preserved (avoids bias-correction inflation at t=1).
    """
    # PyTorch optimizer state_dict format:
    #   prev_optim_state = {'state': {idx: {'step', 'exp_avg', 'exp_avg_sq'}, ...},
    #                       'param_groups': [...]}
    # The idx in 'state' corresponds to position in the FLATTENED parameter list
    # of the param_groups (in order). Both new and prev optimizers have params
    # in the SAME order (lut, decay, tok_emb+nodecay) so indices match.
    prev_state = prev_optim_state['state']

    # Build flat list of new optimizer's params in same order as prev
    new_params_flat = [p for g in optimizer.param_groups for p in g['params']]
    if len(new_params_flat) != len(prev_params):
        print(f'  WARNING: param count mismatch — new={len(new_params_flat)}, prev={len(prev_params)}')

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
            # LUT merge: average (not sum) for m and v — children each track
            # the FULL upstream gradient (post-merge parent does too), so
            # parent.m should have same magnitude as children.m, not 2×.
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


def train_stage(stage_idx, model, bs, n_steps, csv_writer, csv_file, val_csv_writer, val_csv_file, global_step_offset,
                prev_optim_state=None, prev_params=None, prev_nap=None):
    DEVICE_BS = bs
    TOTAL_BS = bs * CONTEXT_SIZE   # grad_accum = 1
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE)
    val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE)

    lut_params = []
    decay_params = []
    nodecay_params = []
    tok_emb_params = []
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

    # Inject merged Adam state from previous stage (if any)
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
        lr_scale = get_stage_lr_scale(step, n_steps)
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
            print(f'  [VAL] stage{stage_idx} step {step}: bpb={bpb:.4f}')
            val_csv_writer.writerow([stage_idx, step, f'{bpb:.6f}'])
            val_csv_file.flush()
            model.train()
    print(f'  stage{stage_idx} done in {(time.time() - t0) / 60:.1f} min')
    # Return ordered param list + optimizer state for next stage
    return optimizer.state_dict(), [p for g in optimizer.param_groups for p in g['params']]


# ============================================================================
# Main orchestrator
# ============================================================================
print('=== NAP curriculum training ===')
print(f'Target arch: NAP={TARGET_NAP}, tph={TARGET_TPH}')
print(f'Curriculum modules: {sorted(CURRICULUM_MODULES)}')
print(f'Baseline (non-curriculum) per-module: {BASELINE_PER_MODULE}')
print(f'Stages:')
for i, st in enumerate(STAGES):
    print(f'  Stage {i}: curriculum NAP={st["curriculum_nap"]}, bs={st["bs"]}, steps={st["n_steps"]}')

trees = build_all_anchor_trees()

# CSV setup
csv_path = os.path.join(EXP_DIR, 'metrics.csv')
val_csv_path = os.path.join(EXP_DIR, 'val_metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f); csv_w.writerow(['stage', 'step', 'train_loss'])
val_csv_f = open(val_csv_path, 'w', newline='')
val_csv_w = csv.writer(val_csv_f); val_csv_w.writerow(['stage', 'step', 'val_bpb'])

prev_optim_state = None
prev_params = None
prev_nap = None
for stage_idx, st in enumerate(STAGES):
    cur_nap = st['curriculum_nap']
    bs = st['bs']
    n_steps = st['n_steps']
    per_module = _per_module_for_stage(stage_idx)
    print(f'\n=== Stage {stage_idx}: curriculum NAP={cur_nap}, bs={bs}, n_steps={n_steps} ===')
    print(f'  per_module: {per_module}')

    model = Model(per_module=per_module).to(DEVICE)
    apply_anchor_trees_to_model(model, trees, nap_for_curriculum=cur_nap)
    n_params = sum(p.numel() for p in model.parameters())
    n_lut = sum(p.numel() for n, p in model.named_parameters() if p.ndim == 3)
    print(f'  Total params: {n_params:,} (LUT: {n_lut:,})')

    if stage_idx > 0:
        prev_ckpt_path = os.path.join(EXP_DIR, f'stage{stage_idx-1}.pt')
        print(f'  Merging weights from {prev_ckpt_path} (prev NAP={prev_nap})')
        prev_ckpt = torch.load(prev_ckpt_path, map_location=DEVICE, weights_only=False)
        prev_model_state = prev_ckpt['model']
        missing, unexpected = merge_weights_from_prev_stage(model, prev_model_state, prev_nap)
        print(f'  merge_state load: missing={len(missing)}, unexpected={len(unexpected)}')
        # Adam state will be loaded inside train_stage via prev_optim_state

    new_optim_state, new_params = train_stage(
        stage_idx, model, bs=bs, n_steps=n_steps,
        csv_writer=csv_w, csv_file=csv_f,
        val_csv_writer=val_csv_w, val_csv_file=val_csv_f,
        global_step_offset=sum(s['n_steps'] for s in STAGES[:stage_idx]),
        prev_optim_state=prev_optim_state,
        prev_params=prev_params,
        prev_nap=prev_nap,
    )

    # Save stage checkpoint (model + optimizer)
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

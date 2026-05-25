"""exp514_tied_lut_unembedder — fork of exp428.

TIED LUT unembedder instead of the Linear(D, V) head.

  - Embedder: nn.Embedding(V, E), E=64.
  - Body: exp428 block (qkv_lut + v_lut + out_proj per layer, RoPE attention).
  - DUAL STREAM COLLAPSED TO E: residual_lut decodes R^E (residual_dim == E);
    the summed residual stream x_resid IS the predicted token embedding.
  - Unembedder: RowTokenLUTUnembedder — tph hash tables, NAP=15 (2^15 == V),
    row r == token r, one learned scalar vote weight per (table, row). Inputs
    are L2-normalised so the hash and the bit-margins are scale-consistent
    (the embedder<->unembedder tie naturally because the hash is sign-based).

  - MAIN loss  = CE(unemb(pred_emb), next_token) + main_bit_weight * assignment_bit
  - AUX  loss  = CE(unemb(emb_table_subsample), ids) + aux_bit_weight * assignment_bit
    (the consistency loss: each token's own embedding must decode to itself)

  Both bit losses use the WINNER-TAKE-ALL ASSIGNMENT STE (validated in
  workbooks/tied_lut_inverse_test.py: naive Hamming-1 STE -> 2% top1, this -> 99.94%):
  route each target to the table whose current hash is closest, then push ALL its
  bits toward the target's binary pattern via a logistic-margin loss.

  evaluate_bpb() calls forward(x, y) -> main CE (mean), unchanged interface, so
  the reported val bpb is the standard next-token metric. Aux CE reported separately.
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
D           = cfg['residual_dim']          # == E in this experiment
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

# Tied-unembedder hyperparameters
UNEMB_NAP    = cfg.get('unemb_nap', 15)
UNEMB_TPH    = cfg.get('unemb_tph', 16)
UNEMB_T      = cfg.get('unemb_t_soft', 0.5)
MAIN_BIT_W   = cfg.get('main_bit_weight', 1.0)
AUX_CE_W     = cfg.get('aux_ce_weight', 1.0)
AUX_BIT_W    = cfg.get('aux_bit_weight', 1.0)
AUX_BATCH    = cfg.get('aux_batch', 8192)


# --- Tokenizer + dataloader ---------------------------------------------------
BASE_DIR = get_base_dir()
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizer')
print(f'Loading tokenizer from {TOKENIZER_DIR}')
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
VOCAB_SIZE = tokenizer.get_vocab_size()
BOS_ID = tokenizer.get_bos_token_id()
print(f'Vocab size: {VOCAB_SIZE}, BOS id: {BOS_ID}')
assert (1 << UNEMB_NAP) == VOCAB_SIZE, \
    f'row=token unembedder needs 2^unemb_nap ({1 << UNEMB_NAP}) == VOCAB_SIZE ({VOCAB_SIZE})'

train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='train', device=DEVICE
)
val_loader_factory = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, DEVICE_BS, CONTEXT_SIZE, split='val', device=DEVICE
)
token_bytes = get_token_bytes(device=DEVICE)


# --- LUT factories (all TinyMHLut soft) ---------------------------------------
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
    kwargs['initial_weights_noise'] = cfg.get('qkv_lut_init_std',
                                              cfg.get('mhlut_init_std', 0.001))
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=2 * d_qk + d_v,
        n_anchor_pairs=cfg['qkv_input_nap'], tables_per_head=cfg['qkv_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **kwargs,
    )

def _make_v(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=E, n_heads=H, n_outputs=d_v,
        n_anchor_pairs=cfg['v_input_nap'], tables_per_head=cfg['v_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS,
    )

def _make_out(layer_idx, seed_offset):
    return TinyMultiHeadLut(
        input_dim=H * d_v, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['out_input_nap'], tables_per_head=cfg['out_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS,
    )

def _make_residual_lut(layer_idx, seed_offset):
    # n_outputs = E (dual stream collapsed to E); decodes a token-embedding delta.
    return TinyMultiHeadLut(
        input_dim=E, n_heads=1, n_outputs=E,
        n_anchor_pairs=cfg['residual_input_nap'], tables_per_head=cfg['residual_tph'],
        random_seed=cfg['random_seed'] + seed_offset, device=DEVICE, **_TINY_SOFT_KWARGS,
    )


# --- Tied LUT unembedder (row = token) ----------------------------------------
class RowTokenLUTUnembedder(nn.Module):
    """Map an embedding (R^E) -> V logits via `tph` hash tables.

    Each table reads `nap` anchor-pair signs from the (L2-normalised) embedding
    -> a `nap`-bit index r in [0, 2^nap). With 2^nap == V, r IS a token id: the
    table adds a learned scalar weight[t, r] to logit[r]. `tph` tables vote.
    => only ~tph logits nonzero per embedding (sparse associative head).

    Forward: hard sign-pack (argmax-equivalent), value carried by weight[t, r].
    Backward: a soft Hamming-1 neighbour relaxation gives the embedding a (weak)
    gradient through CE; the heavy lifting for reachability is the separate
    `assignment_bit_loss` (winner-take-all multi-bit push).
    """
    def __init__(self, E, V, nap, tph, t_soft=0.5, seed=0, device='cuda'):
        super().__init__()
        assert (1 << nap) == V and 1 <= nap <= 15
        self.E, self.V, self.nap, self.tph, self.T = E, V, nap, tph, t_soft
        g = torch.Generator().manual_seed(seed)
        a = torch.randint(0, E, (tph, nap), generator=g)
        b = torch.randint(0, E, (tph, nap), generator=g)
        b = torch.where(b == a, (b + 1) % E, b)
        self.register_buffer('anchor_a', a.to(device))
        self.register_buffer('anchor_b', b.to(device))
        self.register_buffer('powers', (1 << torch.arange(nap)).long().to(device))
        self.register_buffer('bit_ar', torch.arange(nap).to(device))
        self.register_buffer('t_ar', torch.arange(tph).to(device))
        self.weight = nn.Parameter(torch.ones(tph, V, device=device))

    def forward(self, x):
        """x: [N, E] -> (logits [N, V], r [N, tph], d [N, tph, nap])."""
        # exp515: NO L2-normalize (it attenuated the embedding gradient ~1/||x||,
        # stalling the tie in exp514). Hash is sign-based so scale-invariant anyway.
        N = x.shape[0]
        d = x[:, self.anchor_a] - x[:, self.anchor_b]          # [N, tph, nap]
        bits = (d > 0).long()
        r = (bits * self.powers).sum(-1)                       # [N, tph] hard token id

        # HARD term: forward value + weight gradient (gather weight[t, r])
        w_hard = self.weight[self.t_ar[None, :], r]           # [N, tph]
        logit_hard = x.new_zeros(N, self.V).scatter_add(1, r, w_hard)

        # SOFT term: x gradient only (weights detached); spread over r and its
        # Hamming-1 neighbours with prob from per-bit confidence s = sigmoid(|d|/T).
        s = torch.sigmoid(d.abs() / self.T)                   # [N, tph, nap]
        flip_odds = (1.0 - s) / s
        Z = 1.0 + flip_odds.sum(-1, keepdim=True)
        p_stay = 1.0 / Z                                      # [N, tph, 1]
        p_flip = flip_odds / Z                                # [N, tph, nap]
        r_flip = r.unsqueeze(-1) ^ (1 << self.bit_ar)         # [N, tph, nap]
        idx_all = torch.cat([r.unsqueeze(-1), r_flip], dim=-1)        # [N, tph, nap+1]
        p_all = torch.cat([p_stay, p_flip], dim=-1)                   # [N, tph, nap+1]
        w_det = self.weight.detach()
        w_all = w_det[self.t_ar[None, :, None], idx_all]             # [N, tph, nap+1]
        logit_soft = x.new_zeros(N, self.V).scatter_add(
            1, idx_all.reshape(N, -1), (w_all * p_all).reshape(N, -1))

        logits = logit_hard + (logit_soft - logit_soft.detach())
        return logits, r, d

    def assignment_bit_loss(self, d, ids):
        """Winner-take-all reachability driver. For each target id, route to the
        table whose hard hash is closest (min Hamming), then push ALL its bits
        toward the id's binary pattern (logistic margin). Routing is non-diff;
        gradient flows through d -> embedding."""
        N = d.shape[0]
        bits_pred = (d > 0).long()                            # [N, tph, nap]
        bits_tgt = ((ids[:, None] >> self.bit_ar) & 1)        # [N, nap]
        hamming = (bits_pred != bits_tgt[:, None, :]).sum(-1) # [N, tph]
        t_star = hamming.argmin(1)                            # [N]
        d_star = d[torch.arange(N, device=d.device), t_star]  # [N, nap]
        tgt_sign = (2 * bits_tgt - 1).float()                 # [N, nap] in {-1,+1}
        return F.softplus(-tgt_sign * d_star / self.T).mean()


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

        qkv_out = self.qkv_lut(x_pre)                   # [B*T, H, 2*d_qk + d_v]
        q_vec = self.q_norm(qkv_out[..., :d_qk])
        k_vec = self.k_norm(qkv_out[..., d_qk:2 * d_qk])
        v_branch = qkv_out[..., 2 * d_qk:]
        q = q_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        k = k_vec.reshape(B, T, H, d_qk).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos[:T], sin[:T])

        v_vec = self.v_lut(x_pre) + v_branch            # [B*T, H, d_v]
        v = v_vec.reshape(B, T, H, d_v).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_e  = self.out_proj(out_in).squeeze(1)       # [B*T, E]

        x_lut_next_flat = x_flat + out_e
        x_lut_next = x_lut_next_flat.reshape(B, T, E)
        r_in  = self.ln_post(x_lut_next_flat)
        r_out = self.residual_lut(r_in).squeeze(1).reshape(B, T, D)   # D == E
        return x_lut_next, r_out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb_E = nn.Embedding(VOCAB_SIZE, E)
        self.tok_emb_E.weight.data.normal_(0, cfg.get('tok_emb_init', 0.5))
        self.unembedder = RowTokenLUTUnembedder(
            E, VOCAB_SIZE, UNEMB_NAP, UNEMB_TPH, t_soft=UNEMB_T,
            seed=cfg['random_seed'], device=DEVICE)
        self.rope = RotaryEmbedding(d_qk, max_seq_len=CONTEXT_SIZE, base=_ROPE_BASE)
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])

    def get_device(self):
        return self.tok_emb_E.weight.device

    def predict_embedding(self, tokens):
        """Run the body; return the summed residual stream = predicted embedding."""
        B, T = tokens.shape
        x_resid = torch.zeros(B, T, D, device=tokens.device, dtype=self.tok_emb_E.weight.dtype)
        x_lut = self.tok_emb_E(tokens)
        for layer in self.layers:
            x_lut, r = layer(x_lut, self.rope.cos, self.rope.sin)
            x_resid = x_resid + r
        return x_resid                                  # [B, T, E]

    def forward(self, tokens, targets=None, loss_reduction='mean', return_d=False):
        pred_emb = self.predict_embedding(tokens)       # [B, T, E]
        B, T, _ = pred_emb.shape
        logits, r, d = self.unembedder(pred_emb.reshape(B * T, E))
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1),
                reduction=loss_reduction, ignore_index=-1)
            if return_d:
                return ce, d
            return ce
        return logits.view(B, T, VOCAB_SIZE)

    def consistency_loss(self, n_sample=None):
        """AUX: each token's own embedding must decode to itself."""
        if n_sample is None or n_sample >= VOCAB_SIZE:
            ids = torch.arange(VOCAB_SIZE, device=self.get_device())
        else:
            ids = torch.randint(0, VOCAB_SIZE, (n_sample,), device=self.get_device())
        emb = self.tok_emb_E.weight[ids]                # [n, E] (grad flows to rows)
        logits, r, d = self.unembedder(emb)
        aux_ce = F.cross_entropy(logits, ids)
        aux_bit = self.unembedder.assignment_bit_loss(d, ids)
        with torch.no_grad():
            aux_top1 = (logits.argmax(1) == ids).float().mean()
        return aux_ce, aux_bit, aux_top1


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

def get_lr_scale_nowarmup(step):
    # exp515: tie groups (tok_emb, unemb) start at full LR (no warmup), cosine-decay.
    progress = step / N_STEPS
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

_TIE_NO_WARMUP = cfg.get('tie_no_warmup', False)

lut_params, tok_emb_params, unemb_params, decay_params, nodecay_params = [], [], [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'unembedder' in name:
        unemb_params.append(p)                          # the [tph, V] vote table
    elif p.ndim >= 3:
        lut_params.append(p)                            # TinyMHLut tables
    elif name.startswith('tok_emb_E.'):
        tok_emb_params.append(p)
    elif p.ndim == 2:
        decay_params.append(p)
    else:
        nodecay_params.append(p)

_LUT_LR      = cfg.get('lut_lr', cfg['adam_lr'])
_UNEMB_LR    = cfg.get('unemb_lr', cfg['adam_lr'])
_TOK_EMB_LR  = cfg.get('tok_emb_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=lut_params,    lr=_LUT_LR,     betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    dict(params=unemb_params,  lr=_UNEMB_LR,   betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
         tie_no_warmup=_TIE_NO_WARMUP),
    dict(params=tok_emb_params, lr=_TOK_EMB_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
         tie_no_warmup=_TIE_NO_WARMUP),
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
      f'decay={sum(p.numel() for p in decay_params):,} (wd={cfg.get("weight_decay",0.0)}) | '
      f'nodecay={sum(p.numel() for p in nodecay_params):,}')
print(f'D=residual_dim={D} (==E), E={E}, H={H}, d_qk={d_qk}, d_v={d_v}, L={N_LAYERS}')
print(f'TIED unembedder RowTokenLUT: nap={UNEMB_NAP} (2^nap={1<<UNEMB_NAP}=V) tph={UNEMB_TPH} '
      f'-> {UNEMB_TPH*VOCAB_SIZE:,} vote weights')
print(f'losses: MAIN ce + {MAIN_BIT_W}*bit | AUX {AUX_CE_W}*ce + {AUX_BIT_W}*bit (aux_batch={AUX_BATCH})')


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
csv_w.writerow(['step', 'train_loss', 'val_bpb', 'aux_ce', 'aux_top1', 'main_bit', 'aux_bit'])

train_losses_logged, val_bpbs, val_steps = [], [], []
ema = None
best_bpb = float('inf')
t0 = time.time()

model.train()
for step in range(1, N_STEPS + 1):
    lr_scale = get_lr_scale(step)
    tie_scale = get_lr_scale_nowarmup(step)
    for g in optimizer.param_groups:
        scale = tie_scale if g.get('tie_no_warmup') else lr_scale
        g['lr'] = g['initial_lr'] * scale

    optimizer.zero_grad()
    accum_main = 0.0
    last = {}
    for _ in range(grad_accum):
        x, y = next(train_loader)
        main_ce, d = model(x, targets=y, return_d=True)
        main_bit = model.unembedder.assignment_bit_loss(d, y.view(-1))
        aux_ce, aux_bit, aux_top1 = model.consistency_loss(n_sample=AUX_BATCH)
        loss = main_ce + MAIN_BIT_W * main_bit + AUX_CE_W * aux_ce + AUX_BIT_W * aux_bit
        (loss / grad_accum).backward()
        accum_main += main_ce.item() / grad_accum
        last = dict(main_bit=float(main_bit), aux_ce=float(aux_ce),
                    aux_bit=float(aux_bit), aux_top1=float(aux_top1))

    optimizer.step()
    ema = accum_main if ema is None else 0.99 * ema + 0.01 * accum_main

    if step % 100 == 0 or step == 1:
        print(f'step {step:6d} | main={ema:.4f} | aux_ce={last["aux_ce"]:.3f} '
              f'| aux_top1={last["aux_top1"]*100:5.1f}% | main_bit={last["main_bit"]:.4f} '
              f'| aux_bit={last["aux_bit"]:.4f} | lr={lr_scale*cfg["adam_lr"]:.2e}')

    if step % EVAL_EVERY == 0 or step == N_STEPS:
        model.eval()
        bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)
        if bpb < best_bpb:
            best_bpb = bpb
        print(f'[VAL] step {step}: bpb={bpb:.4f} | aux_ce={last["aux_ce"]:.3f} '
              f'| aux_top1={last["aux_top1"]*100:.1f}%')
        train_losses_logged.append(ema); val_bpbs.append(bpb); val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{bpb:.6f}', f'{last["aux_ce"]:.6f}',
                        f'{last["aux_top1"]:.6f}', f'{last["main_bit"]:.6f}', f'{last["aux_bit"]:.6f}'])
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

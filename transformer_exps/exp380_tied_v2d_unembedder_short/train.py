"""exp314_dom_canon_sdpa -- fork of exp312 with DominanceCanonicalize+SDPA.

Architectural changes vs exp312:
  - RankAttention is removed. Q/K dominance is cleaned with
    DominanceCanonicalize (Borda + LN + rank-projection back to ±1) and fed
    directly to F.scaled_dot_product_attention as pairwise rank features.
    The `learnable_attn_scale_init` scalar is preserved and folded into the
    SDPA scale parameter (= attn_scale / sqrt(P_qk)).
  - Manual Borda einsums + nn.LayerNorm(attn), nn.LayerNorm(E) are replaced
    with DominanceToVector modules after attention (d_v) and after out_proj
    (E) — both include Borda + LN in one module.
  - q_norm/k_norm/out_norm are dropped (their roles subsumed by the new
    modules).
"""
import sys, os, json, math, time, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spiky.util.text_snippet_sampler import TextSnippetSampler
from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector, VectorToDominance

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EXP_DIR, 'config.json')) as f:
    cfg = json.load(f)

DEVICE = 'cuda:0'
CONTEXT_SIZE = cfg['context_size']
VOCAB_SIZE = cfg['vocab_size']
BOS_ID = 256
RAW_VOCAB_SIZE = 256
TESTING_LENGTH = 10_000
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'workbooks', 'fineweb_texts.txt')
)

torch.manual_seed(cfg['random_seed'])

E = cfg['embedding_dim']
H = cfg['n_heads']
d_qk = cfg['d_qk']
d_v = cfg['d_v']
N_LAYERS = cfg['num_layers']
D_QK_P = d_qk * (d_qk - 1) // 2
D_V_P = d_v * (d_v - 1) // 2


def _make_qk(seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=cfg['qk_tph'],
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


_V_TPH_PER_LAYER = cfg.get('v_tph_per_layer')
if _V_TPH_PER_LAYER is not None:
    assert len(_V_TPH_PER_LAYER) == N_LAYERS, \
        f"v_tph_per_layer must have {N_LAYERS} entries, got {len(_V_TPH_PER_LAYER)}"


def _make_v(layer_idx, seed_offset):
    tph = (_V_TPH_PER_LAYER[layer_idx] if _V_TPH_PER_LAYER is not None
           else cfg['v_tph'])
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


# Per-layer graded out_tph: if `out_tph_per_layer` (list) is set in config,
# use it; otherwise fall back to the single `out_tph` scalar.
_OUT_TPH_PER_LAYER = cfg.get('out_tph_per_layer')
if _OUT_TPH_PER_LAYER is not None:
    assert len(_OUT_TPH_PER_LAYER) == N_LAYERS, \
        f"out_tph_per_layer must have {N_LAYERS} entries, got {len(_OUT_TPH_PER_LAYER)}"


def _make_out(layer_idx, seed_offset):
    tph = (_OUT_TPH_PER_LAYER[layer_idx] if _OUT_TPH_PER_LAYER is not None
           else cfg['out_tph'])
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'fp8'),
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.q_perm = _make_qk(layer_idx)
        self.k_perm = _make_qk(100 + layer_idx)
        self.v_perm = _make_v(layer_idx, 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, 400 + layer_idx)

        # Canonicalize ±1 bit votes for q/k into a consistent dominance before
        # SDPA. STE (smooth_mode=False by default) so downstream sees bits.
        canon_t = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.k_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        # Canonicalize out_proj's pair-dominance to ±1 before Borda+LN, so the
        # magnitude (vote count) at the out_proj output doesn't leak into the
        # unembedder. Matches the q/k pattern.
        self.out_canon = DominanceCanonicalize(E, temperature=canon_t)

        # Dominance -> vector after attention (P_v -> d_v). No LN here —
        # LN after attention was empirically harmful in prior experiments.
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        # Dominance -> vector after out_proj (P_out -> E, Borda + LN).
        # LN here replaces exp312's `out_norm`.
        self.out_to_vec = DominanceToVector(E)

        # Learnable softmax-sharpness for SDPA (replaces RankAttention's
        # attn_scale). Effective scale = attn_scale / sqrt(P_qk).
        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)

        # Q/K: bit-LUT dominance -> canonicalize -> SDPA features (±1).
        q_dom = self.q_canon(self.q_perm(xp))                          # [B*T, H, P_qk]
        k_dom = self.k_canon(self.k_perm(xp))                          # [B*T, H, P_qk]
        q = q_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)         # [B, H, T, P_qk]
        k = k_dom.reshape(B, T, H, D_QK_P).permute(0, 2, 1, 3)

        # V stays in dominance space.
        v_dom = self.v_perm(x_flat).reshape(B, T, H, D_V_P).permute(0, 2, 1, 3)  # [B, H, T, P_v]

        # Learnable softmax sharpness: multiply q by attn_scale as a tensor
        # so gradients flow into the parameter. SDPA's default scale is
        # 1/sqrt(P_qk), so the net softmax input = attn_scale * q·k / sqrt(P_qk),
        # matching RankAttention's pre-multiplication pattern in exp312.
        attn_dom = F.scaled_dot_product_attention(
            q * self.attn_scale, k, v_dom,
            is_causal=True,
        )                                                              # [B, H, T, P_v]
        attn = self.attn_to_vec(attn_dom)                              # [B, H, T, d_v]

        # Out: concat heads -> bit-LUT dominance -> Borda+LN to E.
        out_in = attn.permute(0, 2, 1, 3).reshape(B * T, H * d_v)
        out_dom = self.out_proj(out_in)                                # [B*T, 1, P_out]
        out_dom = self.out_canon(out_dom)                              # -> ±1 pairs
        out = self.out_to_vec(out_dom).squeeze(1).reshape(B, T, _E)    # [B, T, E]
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Weight-tied via V2D: single learnable token_embedder [V, E] fp32.
        # Input path: use token_embedder.weight as continuous vector.
        # Output path: V2D(token_embedder.weight) → ±1 pair-dominance [V, P_E].
        # Both paths push this one embedding toward encoding token identity.
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.token_v2d = VectorToDominance(
            E, smooth_mode=False,
            temperature=cfg.get('canon_temperature', 0.1),
        )
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        # Bit-native unembedder: concat -> big BitPermLUT -> canonicalize to ±1.
        # Logits = Kendall-tau (integer dot product) against shared token dom.
        # Partition input anchors per-layer (6 groups of E=32 dims each) so each
        # anchor pair only compares inputs within one layer's output segment.
        _FINAL_PART = [list(range(i * E, (i + 1) * E)) for i in range(N_LAYERS)] \
            if cfg.get('final_partition_by_layer', False) else None
        self.final_lut = BitPermutationLUT(
            n_inputs=concat_dim, n_outputs=E, n_heads=1,
            input_nap=cfg.get('final_input_nap', 10),
            output_nap=E,
            tph=cfg.get('final_tph', 1024),
            random_seed=cfg['random_seed'] + 500,
            initial_weights_noise=cfg['bit_lut_latent_init_std'],
            latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
            device=DEVICE,
            partition_sets=_FINAL_PART,
        )
        self.final_canon = DominanceCanonicalize(
            E, temperature=cfg.get('canon_temperature', 0.1),
        )
        # CLIP-style learnable logit scale. Init α = exp(2.0) ≈ 7.4, close
        # to the value (2.07, α≈7.9) learned from zero-init over 25k steps.
        self.log_logit_scale = nn.Parameter(torch.full((1,), 2.0))
        # Per-token learnable bias. Captures unigram frequency prior directly
        # so the LUT doesn't have to encode it through pair patterns.
        self.token_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, tokens):
        tok_dom = self.token_v2d(self.token_embedder.weight)       # [V, P_E] ±1 via STE
        x = self.token_embedder(tokens)                            # [B, T, E] fp32
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)                           # [B, T, 6*E]
        B, T, _ = concat.shape
        pair = self.final_lut(concat.reshape(B * T, -1))           # [B*T, 1, P_E]
        pair = self.final_canon(pair).squeeze(1)                   # [B*T, P_E] ±1
        # Log-odds logits with learnable scale α = exp(log_logit_scale).
        # α=1 at init → natural log-odds (CE = log(V) at uniform softmax).
        P_E = tok_dom.shape[-1]
        dot = pair @ tok_dom.T                                     # [B*T, V] in [-P_E, +P_E]
        base = 2.0 * torch.atanh((dot / P_E).clamp(-0.999, 0.999))
        logits = self.log_logit_scale.exp() * base + self.token_bias
        return logits.view(B, T, -1)


def generate_text(model, prefix, length, device):
    ctx = list(prefix.encode('utf-8'))
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            trunc = ctx[-(CONTEXT_SIZE - 1):]
            x = torch.zeros([1, CONTEXT_SIZE], dtype=torch.long, device=device)
            x[0, 0] = BOS_ID
            x[0, 1:1+len(trunc)] = torch.tensor(trunc, dtype=torch.long, device=device)
            pos = len(trunc)
            logits = model(x)
            probs = torch.softmax(logits[0, pos, :RAW_VOCAB_SIZE], dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ctx.append(next_id)
    return bytes(c for c in ctx if 0 <= c < 256).decode('utf-8', errors='replace')


def evaluate_model(model, sampler, batch_size):
    model.eval()
    losses = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), batch.long().reshape(B*T))
            losses.append(loss.item())
    gen = generate_text(model, 'Once upon a time ', length=100, device=device)
    print(f'[GEN]: {gen}')
    model.train()
    return sum(losses) / len(losses)


def get_lr_scale(step):
    n_steps = cfg['n_steps']
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * n_steps)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(n_steps - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# --- Build model + split params ---
sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=1)
model = Model().to(DEVICE)

bit_luts = []
for layer in model.layers:
    bit_luts += [layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj]
# final_lut goes to its own optimizer instance at a lower lr to compensate for
# the magnified gradient at the Kendall-tau logit boundary.
unembedder_bit_luts = [model.final_lut]

adam_params = list(model.parameters())
print(f'bit LUTs: {len(bit_luts)} main + {len(unembedder_bit_luts)} unembedder')
print(f'Adam-managed parameters: {sum(p.numel() for p in adam_params):,}')

adam_opt = torch.optim.Adam(adam_params, lr=cfg['adam_lr'])
adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)
bit_opt = BitPermutationLUTOptimizer(
    bit_luts,
    lr=cfg['bit_lut_lr'],
    beta1=cfg.get('bit_lut_beta1', 0.9),
    beta2=cfg.get('bit_lut_beta2', 0.999),
    lr_schedule_fn=get_lr_scale,
)
unembedder_bit_opt = BitPermutationLUTOptimizer(
    unembedder_bit_luts,
    lr=cfg.get('unembedder_bit_lut_lr', cfg['bit_lut_lr']),
    beta1=cfg.get('bit_lut_beta1', 0.9),
    beta2=cfg.get('bit_lut_beta2', 0.999),
    lr_schedule_fn=get_lr_scale,
)

print(f'Q/K BitPermLUT: in_nap={cfg["qk_input_nap"]} out_nap={cfg["qk_output_nap"]} tph={cfg["qk_tph"]} d_qk={d_qk} P_qk={D_QK_P}')
_v_tph_str = str(_V_TPH_PER_LAYER) if _V_TPH_PER_LAYER is not None else str(cfg['v_tph'])
print(f'V BitPermLUT:   in_nap={cfg["v_input_nap"]} out_nap={cfg["v_output_nap"]} tph={_v_tph_str} d_v={d_v} P_v={D_V_P}')
_tph_str = str(_OUT_TPH_PER_LAYER) if _OUT_TPH_PER_LAYER is not None else str(cfg['out_tph'])
print(f'Out BitPermLUT: in_nap={cfg["out_input_nap"]} out_nap={cfg["out_output_nap"]} tph={_tph_str}')
print('DominanceCanonicalize on q/k; SDPA directly on dominance features.')
print('DominanceToVector (Borda + LN) after attention and out_proj.')

csv_path = os.path.join(EXP_DIR, 'metrics.csv')
csv_f = open(csv_path, 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'train_loss', 'val_loss'])

train_losses, val_losses, val_steps = [], [], []
best_val = float('inf')
ema = None
t0 = time.time()

model.train()
for step in range(cfg['n_steps']):
    x = sampler.sample_training_batch(cfg['batch_size']).long()
    inp = torch.empty_like(x)
    inp[:, 0] = BOS_ID
    inp[:, 1:] = x[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*T, V), x.reshape(B*T))

    adam_opt.zero_grad()
    bit_opt.zero_grad()
    unembedder_bit_opt.zero_grad()
    loss.backward()
    adam_opt.step()
    adam_scheduler.step()
    bit_opt.step()
    unembedder_bit_opt.step()

    lv = loss.item()
    ema = lv if ema is None else 0.99*ema + 0.01*lv

    if step % 100 == 0:
        print(f'step {step:6d} | loss={ema:.4f} | lr={adam_scheduler.get_last_lr()[0]:.2e}')

    if step % cfg['test_every'] == 0:
        val = evaluate_model(model, sampler, cfg['test_batch_size'])
        if val < best_val:
            best_val = val
        print(f'[VAL] step {step}: {val:.4f}')
        train_losses.append(ema)
        val_losses.append(val)
        val_steps.append(step)
        csv_w.writerow([step, f'{ema:.6f}', f'{val:.6f}'])
        csv_f.flush()

csv_f.close()
elapsed = time.time() - t0

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

total_params = sum(p.numel() for p in model.parameters())
summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'adam_params': total_params,
    'n_bit_luts': len(bit_luts),
    'training_time_hours': round(elapsed / 3600, 3),
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
bit_opt.close()
unembedder_bit_opt.close()
torch.save(model.state_dict(), os.path.join(EXP_DIR, 'checkpoint.pt'))
print('\n=== DONE ===')
print(json.dumps(summary, indent=2))

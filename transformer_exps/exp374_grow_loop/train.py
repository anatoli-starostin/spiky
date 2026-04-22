"""Signal-guided iterative grow-loop training.

Algorithm
---------
1. Build tiny uniform model (per-layer tph per role from `initial_tph`).
2. For each round r in [1..n_rounds]:
   a. Train `steps_per_round` steps, capturing per-LUT-module
      grad_out_norm every step, averaged over the last `signal_window` steps.
   b. For each role in {Q, K, V, Out} independently:
        - current_total_tph = sum of tph across 6 layers
        - target_total_tph  = initial + (target - initial) * r / n_rounds
        - delta_total = target_total_tph - current_total_tph
        - signal_i = mean grad_out_norm for (layer i, role) over last window
        - normalized = signal / sum(signal)
        - delta_tph[i] = round(delta_total * normalized[i])   (integer)
        - Adjust for rounding so sum(delta_tph) = delta_total
   c. For each (layer, role) with delta_tph > 0, grow_lut(...) the module
      and replace it in the model.
   d. Rebuild BitPermutationLUTOptimizer with the new module list.

State preservation: `grow_lut` copies old latent + bit_weights + anchors into
the first `old_tph` slots per head of the new module. New slots get fresh
random state.
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
from spiky.lutorch.ranking_tools import DominanceCanonicalize, DominanceToVector

sys.path.insert(0, os.path.dirname(__file__))
from grow_lut import grow_lut


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

INITIAL_TPH = cfg['initial_tph']
TARGET_TOTALS = cfg['target_totals']
N_ROUNDS = int(cfg['n_rounds'])
STEPS_PER_ROUND = int(cfg['steps_per_round'])
SIGNAL_WINDOW = int(cfg['signal_window'])
TOTAL_STEPS = N_ROUNDS * STEPS_PER_ROUND + STEPS_PER_ROUND  # initial round + grow rounds


# --- Module factories using layer_idx-specific tph ---
def _make_qk(layer_idx, role, tph, seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_qk, n_heads=H,
        input_nap=cfg['qk_input_nap'], output_nap=cfg['qk_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        soft_backward=cfg.get('bit_lut_soft_backward', True),
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        device=DEVICE,
    )


def _make_v(layer_idx, tph, seed_offset):
    return BitPermutationLUT(
        n_inputs=E, n_outputs=d_v, n_heads=H,
        input_nap=cfg['v_input_nap'], output_nap=cfg['v_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        soft_backward=cfg.get('bit_lut_soft_backward', True),
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        device=DEVICE,
    )


def _make_out(layer_idx, tph, seed_offset):
    return BitPermutationLUT(
        n_inputs=H * d_v, n_outputs=E, n_heads=1,
        input_nap=cfg['out_input_nap'], output_nap=cfg['out_output_nap'],
        tph=tph,
        random_seed=cfg['random_seed'] + seed_offset,
        initial_weights_noise=cfg['bit_lut_latent_init_std'],
        soft_backward=cfg.get('bit_lut_soft_backward', True),
        latent_dtype=cfg.get('bit_lut_latent_dtype', 'bf16'),
        device=DEVICE,
    )


class LUTBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_perm = _make_qk(layer_idx, 'Q', INITIAL_TPH['Q'], layer_idx)
        self.k_perm = _make_qk(layer_idx, 'K', INITIAL_TPH['K'], 100 + layer_idx)
        self.v_perm = _make_v(layer_idx, INITIAL_TPH['V'], 200 + layer_idx)
        self.out_proj = _make_out(layer_idx, INITIAL_TPH['Out'], 400 + layer_idx)

        canon_t = cfg.get('canon_temperature', 0.1)
        self.q_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.k_canon = DominanceCanonicalize(d_qk, temperature=canon_t)
        self.attn_to_vec = DominanceToVector(d_v, normalise=False)
        self.out_to_vec = DominanceToVector(E)
        self.attn_scale = nn.Parameter(torch.tensor(
            float(cfg.get('learnable_attn_scale_init', 0.25))
        ))

    def forward(self, x, pos_emb):
        B, T, _E = x.shape
        xp = (x + pos_emb.unsqueeze(0)).reshape(B * T, _E)
        x_flat = x.reshape(B * T, _E)

        q_dom = self.q_canon(self.q_perm(xp))
        k_dom = self.k_canon(self.k_perm(xp))
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
        out = self.out_to_vec(out_dom).squeeze(1).reshape(B, T, _E)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedder = nn.Embedding(VOCAB_SIZE, E)
        self.token_embedder.weight.data.uniform_(-0.1, 0.1)
        self.pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(CONTEXT_SIZE, E) * 0.1) for _ in range(N_LAYERS)]
        )
        self.layers = nn.ModuleList([LUTBlock(i) for i in range(N_LAYERS)])
        concat_dim = E * N_LAYERS
        self.unembedder = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, concat_dim * 4),
            nn.ReLU(),
            nn.Linear(concat_dim * 4, VOCAB_SIZE),
        )

    def forward(self, tokens):
        x = self.token_embedder(tokens)
        outs = []
        for layer, pos_emb in zip(self.layers, self.pos_embs):
            x = layer(x, pos_emb)
            outs.append(x)
        concat = torch.cat(outs, dim=-1)
        return self.unembedder(concat)


def evaluate_model(model, sampler, batch_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(batch_size):
            inp = torch.empty(batch.shape[0], batch.shape[1], dtype=torch.long, device=batch.device)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            logits = model(inp)
            B_, T_, V_ = logits.shape
            loss = F.cross_entropy(logits.reshape(B_*T_, V_), batch.long().reshape(B_*T_))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_lr_scale(step):
    warmup = int(cfg.get('lr_warmup_fraction', 0.1) * TOTAL_STEPS)
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(TOTAL_STEPS - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# --- Build initial model + optimizers ---
sampler = TextSnippetSampler(DATA_PATH, CONTEXT_SIZE, TESTING_LENGTH, DEVICE, random_seed=1)
model = Model().to(DEVICE)


def collect_bit_luts_and_tags(m):
    luts = []
    tags = []
    for layer_idx, layer in enumerate(m.layers):
        luts += [layer.q_perm, layer.k_perm, layer.v_perm, layer.out_proj]
        tags += [(layer_idx, 'Q'), (layer_idx, 'K'), (layer_idx, 'V'), (layer_idx, 'Out')]
    return luts, tags


def build_bit_opt(luts):
    return BitPermutationLUTOptimizer(
        luts,
        lr=cfg['bit_lut_lr'],
        beta1=cfg.get('bit_lut_beta1', 0.9),
        beta2=cfg.get('bit_lut_beta2', 0.999),
        lr_schedule_fn=get_lr_scale,
    )


bit_luts, bit_lut_tags = collect_bit_luts_and_tags(model)

adam_params = list(model.parameters())
print(f'bit LUTs: {len(bit_luts)} modules')
print(f'Adam-managed parameters: {sum(p.numel() for p in adam_params):,}')

adam_opt = torch.optim.Adam(adam_params, lr=cfg['adam_lr'])
adam_scheduler = torch.optim.lr_scheduler.LambdaLR(adam_opt, get_lr_scale)
bit_opt = build_bit_opt(bit_luts)

# --- Logging ---
metrics_f = open(os.path.join(EXP_DIR, 'metrics.csv'), 'w', newline='')
metrics_w = csv.writer(metrics_f)
metrics_w.writerow(['step', 'train_loss', 'val_loss', 'phase'])

topology_log = []  # list of (step, layer, role, tph)
signal_log = []    # list of (round, layer, role, signal)


def log_topology(step, luts, tags):
    for lut, (L, role) in zip(luts, tags):
        topology_log.append((step, L, role, lut.tph))


def print_topology(luts, tags, label):
    print(f'\n=== {label} ===')
    per_role = {'Q': [0]*N_LAYERS, 'K': [0]*N_LAYERS, 'V': [0]*N_LAYERS, 'Out': [0]*N_LAYERS}
    for lut, (L, role) in zip(luts, tags):
        per_role[role][L] = lut.tph
    for role in ['Q', 'K', 'V', 'Out']:
        total = sum(per_role[role])
        print(f'  {role:3s}: {per_role[role]}  (sum={total})')


# --- Training ---
t0 = time.time()
global_step = 0
signal_accum = {tag: [] for tag in bit_lut_tags}  # per-module rolling grad_out_norm samples
best_val = float('inf')
ema = None
train_losses, val_losses, val_steps = [], [], []

log_topology(0, bit_luts, bit_lut_tags)
print_topology(bit_luts, bit_lut_tags, f'INITIAL TOPOLOGY (round 0)')


def train_for_steps(n_steps, phase_label):
    """Train `n_steps`, capturing grad_out_norms during the final
    SIGNAL_WINDOW steps into signal_accum.
    """
    global global_step, ema, best_val
    signal_start = n_steps - SIGNAL_WINDOW
    # Reset signal accumulator for this round
    for k in signal_accum:
        signal_accum[k].clear()

    for local in range(n_steps):
        x = sampler.sample_training_batch(cfg['batch_size']).long()
        inp = torch.empty_like(x)
        inp[:, 0] = BOS_ID
        inp[:, 1:] = x[:, :-1]
        logits = model(inp)
        B_, T_, V_ = logits.shape
        loss = F.cross_entropy(logits.reshape(B_*T_, V_), x.reshape(B_*T_))

        adam_opt.zero_grad()
        bit_opt.zero_grad()
        loss.backward()
        adam_opt.step()
        adam_scheduler.step()

        # Capture per-module grad_out_norm BEFORE bit_opt.step() clears it.
        if local >= signal_start:
            for (lut, state, tag) in zip(bit_luts, bit_opt._states, bit_lut_tags):
                go = state.get("grad_out")
                if go is not None:
                    signal_accum[tag].append(float(go.float().norm().item()))

        bit_opt.step()

        lv = loss.item()
        ema = lv if ema is None else 0.99*ema + 0.01*lv

        global_step += 1
        if global_step % 100 == 0:
            print(f'step {global_step:6d} [{phase_label}] | loss={ema:.4f} | lr={adam_scheduler.get_last_lr()[0]:.2e}')

        if global_step % cfg['test_every'] == 0:
            val = evaluate_model(model, sampler, cfg['test_batch_size'])
            if val < best_val:
                best_val = val
            print(f'[VAL] step {global_step} [{phase_label}]: {val:.4f}')
            train_losses.append(ema)
            val_losses.append(val)
            val_steps.append(global_step)
            metrics_w.writerow([global_step, f'{ema:.6f}', f'{val:.6f}', phase_label])
            metrics_f.flush()


def allocate_and_grow(round_idx):
    """Use signal_accum to allocate budget increments, then grow each LUT.
    Returns new (luts, tags, bit_opt)."""
    global bit_luts, bit_lut_tags, bit_opt

    # Compute per-module mean signal.
    signals = {tag: sum(v) / max(len(v), 1) for tag, v in signal_accum.items()}

    # Current tph per (layer, role)
    current = {}
    for lut, tag in zip(bit_luts, bit_lut_tags):
        current[tag] = lut.tph

    # Target total per role at end of this round
    initial_totals = {r: INITIAL_TPH[r] * N_LAYERS for r in INITIAL_TPH}
    target_now = {}
    for r in TARGET_TOTALS:
        delta = TARGET_TOTALS[r] - initial_totals[r]
        target_now[r] = initial_totals[r] + int(round(delta * round_idx / N_ROUNDS))

    # Allocate per-role across 6 layers by signal proportion.
    new_tph = {}
    for r in ['Q', 'K', 'V', 'Out']:
        role_signals = [signals[(L, r)] for L in range(N_LAYERS)]
        total_sig = sum(role_signals)
        if total_sig <= 0:
            # Fallback to uniform
            props = [1.0 / N_LAYERS] * N_LAYERS
        else:
            props = [s / total_sig for s in role_signals]
        # Each layer's new tph = round(target_now * prop), with floor at current
        # (never shrink).
        allocs = [max(int(round(target_now[r] * p)), current[(L, r)]) for L, p in enumerate(props)]
        # Normalize sum to exactly target_now[r] by adjusting the max-signal layer
        diff = target_now[r] - sum(allocs)
        if diff != 0:
            # Add/subtract from the layer with highest signal (never below current)
            order = sorted(range(N_LAYERS), key=lambda L: -role_signals[L])
            for L in order:
                if diff == 0:
                    break
                if diff > 0:
                    allocs[L] += diff
                    diff = 0
                else:  # diff < 0
                    can_reduce = allocs[L] - current[(L, r)]
                    cut = min(-diff, can_reduce)
                    allocs[L] -= cut
                    diff += cut
        for L in range(N_LAYERS):
            new_tph[(L, r)] = allocs[L]

    # Log signals + new allocations
    for L in range(N_LAYERS):
        for r in ['Q', 'K', 'V', 'Out']:
            signal_log.append((round_idx, L, r, signals[(L, r)], current[(L, r)], new_tph[(L, r)]))

    print(f'\n=== ROUND {round_idx} allocation (target totals {target_now}) ===')
    for r in ['Q', 'K', 'V', 'Out']:
        per_layer = [new_tph[(L, r)] for L in range(N_LAYERS)]
        sigs = [f'{signals[(L, r)]:.2e}' for L in range(N_LAYERS)]
        print(f'  {r:3s}: {per_layer}  (sum={sum(per_layer)})  signals={sigs}')

    # Apply grow to each LUT
    grow_seed_base = 1000 + round_idx * 1000
    new_bit_luts = []
    for i, (lut, tag) in enumerate(zip(bit_luts, bit_lut_tags)):
        L, role = tag
        new_size = new_tph[tag]
        if new_size > lut.tph:
            grown = grow_lut(lut, new_size, seed=grow_seed_base + i)
            # Replace module in the parent layer
            layer = model.layers[L]
            if role == 'Q':
                layer.q_perm = grown
            elif role == 'K':
                layer.k_perm = grown
            elif role == 'V':
                layer.v_perm = grown
            elif role == 'Out':
                layer.out_proj = grown
            new_bit_luts.append(grown)
        else:
            new_bit_luts.append(lut)

    # Re-collect and rebuild optimizer
    bit_luts, bit_lut_tags = collect_bit_luts_and_tags(model)
    # Important: close old optimizer's forward hooks, then build fresh
    bit_opt.close()
    bit_opt = build_bit_opt(bit_luts)
    print_topology(bit_luts, bit_lut_tags, f'TOPOLOGY after round {round_idx} grow')
    log_topology(global_step, bit_luts, bit_lut_tags)


model.train()

# Round 0: initial warmup training with tiny uniform model
train_for_steps(STEPS_PER_ROUND, f'round0_warmup')

# Rounds 1..N: grow then train
for r in range(1, N_ROUNDS + 1):
    allocate_and_grow(r)
    train_for_steps(STEPS_PER_ROUND, f'round{r}')

metrics_f.close()
elapsed = time.time() - t0

# Save topology + signal logs
with open(os.path.join(EXP_DIR, 'topology.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'layer', 'role', 'tph'])
    for row in topology_log:
        w.writerow(row)

with open(os.path.join(EXP_DIR, 'signals.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['round', 'layer', 'role', 'signal', 'prev_tph', 'new_tph'])
    for row in signal_log:
        w.writerow(row)

plt.figure(figsize=(8, 4))
plt.plot(val_steps, train_losses, label='train')
plt.plot(val_steps, val_losses, label='val')
plt.xlabel('steps'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'loss.png'), dpi=100)
plt.close()

# Final summary
final_luts, _ = collect_bit_luts_and_tags(model)
final_topology = {}
for lut, (L, role) in zip(final_luts, bit_lut_tags):
    final_topology.setdefault(role, [0]*N_LAYERS)
    final_topology[role][L] = lut.tph

total_params = sum(p.numel() for p in model.parameters())
summary = {
    'exp_name': cfg['exp_name'],
    'best_val_loss': best_val,
    'final_val_loss': val_losses[-1] if val_losses else None,
    'adam_params': total_params,
    'n_bit_luts': len(final_luts),
    'training_time_hours': round(elapsed / 3600, 3),
    'final_topology': final_topology,
}
with open(os.path.join(EXP_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

bit_opt.close()
print(f'\nDone. best_val={best_val:.4f}  time={elapsed/60:.1f} min')
print(f'Final topology: {final_topology}')

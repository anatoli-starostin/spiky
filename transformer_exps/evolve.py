#!/usr/bin/env python3
"""
Evolutionary search for optimal anchor topology in LUT transformers.

Fixed architecture:
  embedding_dim=32, positional_dim=16, n_layers=6, n_heads=4, d_qk=16, d_v=16
  vocab_size=257, context_size=32

Search space:
  nap       — anchor pairs per table (table_dim = 2^nap)
  tph       — tables per head (auto-set from ~3M param budget)
  qkv_a/b  — comparison graph for q/k/v LUTs (input_dim=48), freely evolved
  op_a/b   — comparison graph for out_proj LUT  (input_dim=64), freely evolved

Anchor matrices are shared across all layers and across q/k/v.
"""
import os, sys, csv, json, math, time, copy
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"  # disable compile: serialises streams
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformer_exps.common import make_sampler, BOS_ID, VOCAB_SIZE, CONTEXT_SIZE
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode, AnchorSamplingPolicy, get_balanced_anchor_pairs
from spiky.lutorch.ranking_tools import RankAttention

DEVICE = 'cuda:0'
OUT_DIR      = os.path.join(os.path.dirname(__file__), 'evolve_results')
GENOMES_DIR  = os.path.join(OUT_DIR, 'genomes')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GENOMES_DIR, exist_ok=True)

# ── Fixed architecture ─────────────────────────────────────────────────────────
EMB       = 32    # embedding_dim
POS       = 16    # positional_dim
N_LAYERS  = 1
N_HEADS   = 4
D_QK      = 16
D_V       = 16
VOCAB     = 257
TEMP      = 0.5   # rank attention temperature

INPUT_DIM_QKV = EMB + POS   # 48  (q/k/v LUT input)
INPUT_DIM_OP  = N_HEADS * D_V  # 64  (out_proj LUT input)

# ── Search / training config ───────────────────────────────────────────────────
BUDGET      = 500_000
N_STEPS     = 5_000
LR          = 1e-4
LR_UNEMB    = 1e-3
BATCH_SIZE  = 128
EVAL_EVERY  = 1_000
N_PARALLEL  = 16     # models to train simultaneously on the GPU (CUDA streams)

# ── Evolutionary config ────────────────────────────────────────────────────────
POP_SIZE      = 16    # individuals per generation (= N_PARALLEL → one round per gen)
N_SURVIVORS   = 4     # top survivors carried to next gen
N_GENERATIONS = 108
MUTATION_RATE = 0.15  # fraction of anchor indices randomly rewired per mutation

# ── Budget helpers ─────────────────────────────────────────────────────────────

def n_params(nap: int, tph_qkv: int, tph_op: int) -> int:
    td = 2 ** nap
    lut = N_LAYERS * (3 * N_HEADS * tph_qkv * td * D_QK + tph_op * td * EMB)
    return lut + 2 * VOCAB * EMB


def tph_for_budget(nap: int, budget: int = BUDGET) -> Tuple[int, int]:
    """Return (tph_qkv, tph_op) ≈ budget params, with tph_op = tph_qkv."""
    td = 2 ** nap
    budget_lut = budget - 2 * VOCAB * EMB
    cost_per_tph = N_LAYERS * td * (3 * N_HEADS * D_QK + EMB)
    tph = max(1, int(budget_lut / cost_per_tph))
    return tph, tph


# ── Genome ─────────────────────────────────────────────────────────────────────

@dataclass
class Genome:
    nap: int
    tph_qkv: int                # tables per head for q/k/v (n_heads * tph_qkv total)
    tph_op: int                 # total tables for out_proj (1 head)
    qkv_a: np.ndarray           # [N_HEADS * tph_qkv, nap] int64, indices in [0, INPUT_DIM_QKV)
    qkv_b: np.ndarray
    op_a: np.ndarray            # [tph_op, nap] int64, indices in [0, INPUT_DIM_OP)
    op_b: np.ndarray
    val_loss: float = float('inf')
    trial_id: int = -1
    origin: str = 'random'      # 'random', 'hierarchical', 'multiscale', 'mutate', 'crossover'

    def n_params(self) -> int:
        return n_params(self.nap, self.tph_qkv, self.tph_op)

    def summary(self) -> str:
        return (f"trial={self.trial_id:3d} | nap={self.nap} tph_qkv={self.tph_qkv}"
                f" params={self.n_params()/1e6:.2f}M | val={self.val_loss:.4f} | {self.origin}")


def _fix_collisions(a: np.ndarray, b: np.ndarray, input_dim: int, rng: np.random.Generator) -> None:
    col = a == b
    while col.any():
        b[col] = rng.integers(0, input_dim, size=col.sum(), dtype=np.int64)
        col = a == b


def random_genome(nap: int, rng: np.random.Generator, origin: str = 'random') -> Genome:
    tph_qkv, tph_op = tph_for_budget(nap)
    qkv_a = rng.integers(0, INPUT_DIM_QKV, size=(N_HEADS * tph_qkv, nap), dtype=np.int64)
    qkv_b = rng.integers(0, INPUT_DIM_QKV, size=(N_HEADS * tph_qkv, nap), dtype=np.int64)
    _fix_collisions(qkv_a, qkv_b, INPUT_DIM_QKV, rng)
    op_a = rng.integers(0, INPUT_DIM_OP, size=(tph_op, nap), dtype=np.int64)
    op_b = rng.integers(0, INPUT_DIM_OP, size=(tph_op, nap), dtype=np.int64)
    _fix_collisions(op_a, op_b, INPUT_DIM_OP, rng)
    return Genome(nap, tph_qkv, tph_op, qkv_a, qkv_b, op_a, op_b, origin=origin)


def _policy_genome(nap: int, policy: AnchorSamplingPolicy, origin: str) -> Genome:
    from spiky.lutorch.lut_helpers import compute_hierarchical_n_tables, compute_multiscale_n_tables
    tph_qkv_budget, tph_op_budget = tph_for_budget(nap)
    # Clamp to what the policy actually generates (policies have fixed table counts)
    if policy == AnchorSamplingPolicy.HIERARCHICAL:
        tph_qkv = min(tph_qkv_budget, compute_hierarchical_n_tables(INPUT_DIM_QKV, nap))
        tph_op  = min(tph_op_budget,  compute_hierarchical_n_tables(INPUT_DIM_OP,  nap))
    elif policy == AnchorSamplingPolicy.MULTISCALE:
        tph_qkv = min(tph_qkv_budget, compute_multiscale_n_tables(INPUT_DIM_QKV, nap))
        tph_op  = min(tph_op_budget,  compute_multiscale_n_tables(INPUT_DIM_OP,  nap))
    else:
        tph_qkv, tph_op = tph_qkv_budget, tph_op_budget
    dev = torch.device('cpu')
    a_qkv, b_qkv = get_balanced_anchor_pairs(
        N_HEADS * tph_qkv, nap, INPUT_DIM_QKV, dev,
        policy=policy, n_heads=N_HEADS, shuffle_per_head=True, random_seed=0,
    )
    a_op, b_op = get_balanced_anchor_pairs(
        tph_op, nap, INPUT_DIM_OP, dev,
        policy=policy, n_heads=1, shuffle_per_head=False, random_seed=1,
    )
    return Genome(nap, tph_qkv, tph_op,
                  a_qkv.numpy(), b_qkv.numpy(),
                  a_op.numpy(), b_op.numpy(), origin=origin)


def hierarchical_genome(nap: int) -> Genome:
    return _policy_genome(nap, AnchorSamplingPolicy.HIERARCHICAL, 'hierarchical')


def multiscale_genome(nap: int) -> Genome:
    return _policy_genome(nap, AnchorSamplingPolicy.MULTISCALE, 'multiscale')


def mutate(g: Genome, rng: np.random.Generator, rate: float = MUTATION_RATE) -> Genome:
    """Randomly rewire `rate` fraction of anchor indices."""
    def _mutate(a, b, input_dim):
        a, b = a.copy(), b.copy()
        for arr, dim in [(a, input_dim), (b, input_dim)]:
            mask = rng.random(arr.shape) < rate
            arr[mask] = rng.integers(0, dim, size=mask.sum(), dtype=np.int64)
        _fix_collisions(a, b, input_dim, rng)
        return a, b
    qkv_a, qkv_b = _mutate(g.qkv_a, g.qkv_b, INPUT_DIM_QKV)
    op_a,  op_b  = _mutate(g.op_a,  g.op_b,  INPUT_DIM_OP)
    return Genome(g.nap, g.tph_qkv, g.tph_op, qkv_a, qkv_b, op_a, op_b, origin='mutate')


def crossover(p1: Genome, p2: Genome, rng: np.random.Generator) -> Genome:
    """Row-wise cut-and-splice of anchor matrices from two same-nap parents."""
    assert p1.nap == p2.nap and p1.tph_qkv == p2.tph_qkv
    def _cross(a1, b1, a2, b2):
        cut = rng.integers(1, a1.shape[0])
        return (np.concatenate([a1[:cut], a2[cut:]]),
                np.concatenate([b1[:cut], b2[cut:]]))
    qkv_a, qkv_b = _cross(p1.qkv_a, p1.qkv_b, p2.qkv_a, p2.qkv_b)
    op_a,  op_b  = _cross(p1.op_a,  p1.op_b,  p2.op_a,  p2.op_b)
    return Genome(p1.nap, p1.tph_qkv, p1.tph_op, qkv_a, qkv_b, op_a, op_b, origin='crossover')


# ── Model ──────────────────────────────────────────────────────────────────────

def _make_lut(input_dim, n_heads, n_outputs, tph, nap, anchor_a, anchor_b):
    """Build MultiHeadLut from prebuilt anchor matrices (numpy arrays)."""
    a = torch.from_numpy(anchor_a).long()
    b = torch.from_numpy(anchor_b).long()
    return MultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        smooth_mode=False,
        n_alternatives=1,
        normalize_weights=False,
        calibrate_output=False,
        initial_weights_noise=0.001,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
        prebuilt_anchor_pairs=(a, b),
    )


class EvolveBlock(nn.Module):
    def __init__(self, genome: Genome, layer_idx: int):
        super().__init__()
        nap = genome.nap
        # q/k/v share the same anchor topology; each gets independent weights via separate LUT
        self.q_lut = _make_lut(INPUT_DIM_QKV, N_HEADS, D_QK, genome.tph_qkv, nap,
                                genome.qkv_a, genome.qkv_b)
        self.k_lut = _make_lut(INPUT_DIM_QKV, N_HEADS, D_QK, genome.tph_qkv, nap,
                                genome.qkv_a, genome.qkv_b)
        self.v_lut = _make_lut(INPUT_DIM_QKV, N_HEADS, D_V,  genome.tph_qkv, nap,
                                genome.qkv_a, genome.qkv_b)
        self.out_proj = _make_lut(INPUT_DIM_OP, 1, EMB, genome.tph_op, nap,
                                  genome.op_a, genome.op_b)
        self.rank_attn = RankAttention(D_QK, D_V, smooth_mode=False, temperature=TEMP)

    def forward(self, x, pos):
        B, T, E = x.shape
        x_pos = torch.cat([x, pos], dim=-1).reshape(B * T, INPUT_DIM_QKV)
        q = self.q_lut(x_pos).permute(1, 0, 2)   # [N_HEADS, B*T, D_QK]
        k = self.k_lut(x_pos).permute(1, 0, 2)
        v = self.v_lut(x_pos).permute(1, 0, 2)
        q = q.reshape(N_HEADS, B, T, D_QK).permute(1, 0, 2, 3)
        k = k.reshape(N_HEADS, B, T, D_QK).permute(1, 0, 2, 3)
        v = v.reshape(N_HEADS, B, T, D_V).permute(1, 0, 2, 3)
        attn = self.rank_attn(q, k, v, is_causal=True)          # [B, N_HEADS, T, D_V]
        attn = attn.permute(0, 2, 1, 3).reshape(B * T, N_HEADS * D_V)
        out = self.out_proj(attn)[:, 0, :]                       # [B*T, EMB]
        return x + out.reshape(B, T, EMB)


class EvolveTransformer(nn.Module):
    def __init__(self, genome: Genome):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB, EMB)
        self.token_emb.weight.data.uniform_(-0.1, 0.1)
        self.register_buffer('pos_emb', torch.randn(1, CONTEXT_SIZE, POS) * 0.1)
        self.layers = nn.ModuleList([EvolveBlock(genome, i) for i in range(N_LAYERS)])
        self.unemb = nn.Linear(EMB, VOCAB, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.token_emb(tokens)
        pos = self.pos_emb[:, :T].expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, pos)
        return self.unemb(x)


# ── Trial (parallel) ───────────────────────────────────────────────────────────

_sampler = None

def get_sampler():
    global _sampler
    if _sampler is None:
        _sampler = make_sampler(DEVICE, random_seed=1)
    return _sampler


def _eval_val_loss(model) -> float:
    # No model.eval() — avoids torch.compile retrace; models have no BN/Dropout
    losses = []
    sampler = get_sampler()
    with torch.no_grad():
        for batch in sampler.testing_batches_iterator(256):
            inp = torch.empty_like(batch)
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1].long()
            tgt = batch.long()
            logits = model(inp)
            B, T, V = logits.shape
            losses.append(F.cross_entropy(
                logits.reshape(B * T, V), tgt.reshape(B * T)
            ).item())
    return sum(losses) / len(losses)


def _make_optimizer(model):
    return torch.optim.Adam([
        {'params': model.unemb.parameters(), 'lr': LR_UNEMB},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('unemb')], 'lr': LR},
    ])


def run_generation(genomes: List[Genome], start_trial_id: int) -> None:
    """
    Train all genomes in parallel on the GPU using CUDA streams.
    Updates genome.val_loss and genome.trial_id in-place.
    Processes them in chunks of N_PARALLEL if len(genomes) > N_PARALLEL.
    """
    for chunk_start in range(0, len(genomes), N_PARALLEL):
        chunk = genomes[chunk_start:chunk_start + N_PARALLEL]
        _run_chunk(chunk, start_trial_id + chunk_start)


def _run_chunk(genomes: List[Genome], base_trial_id: int) -> None:
    N = len(genomes)
    for i, g in enumerate(genomes):
        g.trial_id = base_trial_id + i

    print(f"\n── Parallel chunk: {N} trials  "
          f"(ids {base_trial_id}–{base_trial_id + N - 1}) ──")
    for g in genomes:
        print(f"   [{g.trial_id}] nap={g.nap} tph_qkv={g.tph_qkv}"
              f" params={g.n_params()/1e6:.2f}M  {g.origin}")

    models     = [EvolveTransformer(g).to(DEVICE) for g in genomes]
    optimizers = [_make_optimizer(m) for m in models]
    streams    = [torch.cuda.Stream(device=DEVICE) for _ in range(N)]

    best_vals  = [float('inf')] * N
    emas       = [None] * N
    alpha      = 0.02
    sampler    = get_sampler()
    t0         = time.time()

    for m in models:
        m.train()

    for step in range(N_STEPS):
        # Draw one batch per model
        batches = [sampler.sample_training_batch(BATCH_SIZE).long() for _ in range(N)]

        # Forward + backward + optimizer step, each model on its own stream
        raw_losses = []
        for i in range(N):
            with torch.cuda.stream(streams[i]):
                x   = batches[i]
                inp = x.clone()
                inp[:, 0] = BOS_ID
                inp[:, 1:] = x[:, :-1]
                logits = models[i](inp)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), x.reshape(B * T))
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                raw_losses.append(loss)

        torch.cuda.synchronize()

        for i in range(N):
            lv = raw_losses[i].item()
            emas[i] = lv if emas[i] is None else (1 - alpha) * emas[i] + alpha * lv

        if (step + 1) % EVAL_EVERY == 0:
            elapsed = (time.time() - t0) / 3600
            vals = [_eval_val_loss(m) for m in models]
            for i in range(N):
                if vals[i] < best_vals[i]:
                    best_vals[i] = vals[i]
            row = "  ".join(
                f"[{genomes[i].trial_id}] {emas[i]:.3f}/{best_vals[i]:.4f}"
                for i in range(N)
            )
            print(f"  step {step+1:5d} | {row} | {elapsed:.2f}h")

    for i, g in enumerate(genomes):
        g.val_loss = best_vals[i]
    print(f"  → " + "  ".join(f"[{genomes[i].trial_id}]={best_vals[i]:.4f}" for i in range(N)))


# ── Leaderboard ────────────────────────────────────────────────────────────────

def print_leaderboard(population: List[Genome], gen: int):
    ranked = sorted(population, key=lambda g: g.val_loss)
    print(f"\n{'='*70}")
    print(f"  Generation {gen} leaderboard")
    print(f"{'='*70}")
    for rank, g in enumerate(ranked, 1):
        print(f"  #{rank:2d} {g.summary()}")
    print(f"{'='*70}\n")


def log_trial(writer, gen: int, g: Genome):
    writer.writerow([gen, g.trial_id, g.nap, g.tph_qkv, g.tph_op,
                     g.n_params(), f'{g.val_loss:.6f}', g.origin])
    genome_path = os.path.join(GENOMES_DIR, f'gen{gen:02d}_trial{g.trial_id:04d}.json')
    with open(genome_path, 'w') as f:
        json.dump({
            'gen': gen, 'trial_id': int(g.trial_id), 'origin': g.origin,
            'nap': int(g.nap), 'tph_qkv': int(g.tph_qkv), 'tph_op': int(g.tph_op),
            'n_params': int(g.n_params()), 'val_loss': float(g.val_loss),
            'qkv_a': g.qkv_a.tolist(), 'qkv_b': g.qkv_b.tolist(),
            'op_a': g.op_a.tolist(), 'op_b': g.op_b.tolist(),
        }, f)


# ── Initial population ─────────────────────────────────────────────────────────

def initial_population(rng: np.random.Generator) -> List[Genome]:
    """
    Diverse starting pool (16 individuals):
      - 3 random genomes per nap in {4, 5, 6}  =  9
      - hierarchical-init at nap={4, 5, 6}      =  3
      - multiscale-init   at nap={4, 5, 6}      =  3
      - 1 extra random at nap=5 (sweet-spot)    =  1
    """
    pop = []
    for nap in [4, 5, 6]:
        for _ in range(3):
            pop.append(random_genome(nap, rng, origin='random'))
        pop.append(hierarchical_genome(nap))
        pop.append(multiscale_genome(nap))
    pop.append(random_genome(5, rng, origin='random'))
    return pop  # 16 individuals


# ── Evolutionary loop ──────────────────────────────────────────────────────────

def _save_checkpoint(gen: int, trial_counter: int, population: List[Genome]):
    ckpt = {
        'gen': gen, 'trial_counter': trial_counter,
        'population': [{
            'nap': int(g.nap), 'tph_qkv': int(g.tph_qkv), 'tph_op': int(g.tph_op),
            'val_loss': float(g.val_loss), 'trial_id': int(g.trial_id), 'origin': g.origin,
            'qkv_a': g.qkv_a.tolist(), 'qkv_b': g.qkv_b.tolist(),
            'op_a': g.op_a.tolist(), 'op_b': g.op_b.tolist(),
        } for g in population],
    }
    path = os.path.join(OUT_DIR, 'checkpoint.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(ckpt, f)
    os.replace(path + '.tmp', path)


def _load_checkpoint() -> Optional[Tuple[int, int, List[Genome]]]:
    path = os.path.join(OUT_DIR, 'checkpoint.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ckpt = json.load(f)
    population = []
    for d in ckpt['population']:
        g = Genome(
            nap=d['nap'], tph_qkv=d['tph_qkv'], tph_op=d['tph_op'],
            qkv_a=np.array(d['qkv_a'], dtype=np.int64),
            qkv_b=np.array(d['qkv_b'], dtype=np.int64),
            op_a=np.array(d['op_a'],  dtype=np.int64),
            op_b=np.array(d['op_b'],  dtype=np.int64),
            val_loss=d['val_loss'], trial_id=d['trial_id'], origin=d['origin'],
        )
        population.append(g)
    return ckpt['gen'], ckpt['trial_counter'], population


def evolve():
    rng = np.random.default_rng(seed=42)
    trial_counter = [0]

    checkpoint = _load_checkpoint()
    if checkpoint is not None:
        start_gen, trial_counter[0], population = checkpoint
        start_gen += 1  # resume from next generation
        print(f"Resuming from checkpoint: gen={start_gen}, trial_counter={trial_counter[0]}")
        log_mode = 'a'  # append to existing CSV
    else:
        start_gen = 0
        population = initial_population(rng)
        log_mode = 'w'

    log_path = os.path.join(OUT_DIR, 'trials.csv')
    with open(log_path, log_mode, newline='') as log_f:
        writer = csv.writer(log_f)
        if log_mode == 'w':
            writer.writerow(['gen', 'trial_id', 'nap', 'tph_qkv', 'tph_op',
                             'n_params', 'val_loss', 'origin'])

        for gen in range(start_gen, N_GENERATIONS):
            print(f"\n{'#'*70}")
            print(f"  GENERATION {gen}  ({len(population)} individuals)")
            print(f"{'#'*70}")

            # ── Evaluate all unevaluated individuals in parallel ──
            pending = [g for g in population if g.val_loss == float('inf')]
            if pending:
                run_generation(pending, trial_counter[0])
                trial_counter[0] += len(pending)
                for g in pending:
                    log_trial(writer, gen, g)
                log_f.flush()

            print_leaderboard(population, gen)
            _save_checkpoint(gen, trial_counter[0], population)

            if gen == N_GENERATIONS - 1:
                break  # no need to breed after last generation

            # ── Selection ──
            ranked = sorted(population, key=lambda g: g.val_loss)
            survivors = ranked[:N_SURVIVORS]

            # ── Breed next generation ──
            next_pop = list(survivors)  # survivors carry forward

            # Mutate each survivor
            for s in survivors:
                next_pop.append(mutate(s, rng))

            # Crossover between survivors (round-robin)
            for i in range(len(survivors)):
                p1 = survivors[i]
                p2 = survivors[(i + 1) % len(survivors)]
                if p1.nap == p2.nap:
                    next_pop.append(crossover(p1, p2, rng))

            # Fill remaining slots with fresh random genomes
            all_naps = list({g.nap for g in survivors})
            while len(next_pop) < POP_SIZE:
                nap = rng.choice(all_naps + [4, 5, 6])
                next_pop.append(random_genome(nap, rng))

            # Trim to POP_SIZE (drop excess)
            population = next_pop[:POP_SIZE]

    print("\n\n=== EVOLUTION COMPLETE ===")
    all_genomes = sorted(population, key=lambda g: g.val_loss)
    print(f"Best: {all_genomes[0].summary()}")
    best_path = os.path.join(OUT_DIR, 'best_genome.json')
    best = all_genomes[0]
    with open(best_path, 'w') as f:
        json.dump({
            'nap': int(best.nap), 'tph_qkv': int(best.tph_qkv), 'tph_op': int(best.tph_op),
            'val_loss': float(best.val_loss), 'n_params': int(best.n_params()), 'origin': best.origin,
            'qkv_a': best.qkv_a.tolist(), 'qkv_b': best.qkv_b.tolist(),
            'op_a': best.op_a.tolist(), 'op_b': best.op_b.tolist(),
        }, f, indent=2)
    print(f"Best genome saved to {best_path}")


if __name__ == '__main__':
    evolve()

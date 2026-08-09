"""Stage one of a steady-state ("ALife tournament") neuroevolution loop.

Differs from the generational sweeps in three ways:
  * FULL GENOME — the genome is the ENTIRE synapse list (reservoir + input + output), not
    just the I/O layer over a frozen reservoir. add / prune / jitter all act on it.
  * NO GENERATIONS — K nets live in a pool, all see the same resampled batch stream, and
    each carries an EWMA of its corrected score. Every round the worst M are culled and
    replaced by mutated clones of fitness-weighted survivors.
  * CLONE = COLD START — a child is a deep copy of the parent's synapse arrays. It begins
    with empty membranes and empty delay lines, which is fine: the capability audit showed
    live state cannot be copied anyway, and with STDP off the in-memory genome IS the
    current weight state, so nothing is lost by rebuilding.

STDP is OFF everywhere (learning_rate 0 on every meta), so this stage is pure
selection+mutation. Dale's law holds throughout: excitatory sources keep weight >= 0,
inhibitory sources are pinned at RES_W_INH.

    python steady_state.py --pool 32 --rounds 60
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from harness import (D_MAX, D_MIN, GROUP_SIZE, LatencyEncoder, N_EXC, N_INH, N_IN,
                     N_OUT, N_TICKS, RES_W_INH, delay_metas, group_aligned_weights,
                     kendall_tau_b, own_null, run_episode)
from data import load, sample_batch

HERE = os.path.dirname(os.path.abspath(__file__))
# Where histories/checkpoints land. Default keeps the old behaviour; set ND_OUT to the
# experiment's own directory so each run writes beside its README instead of into src/.
OUT = os.path.abspath(os.environ.get("ND_OUT", os.path.join(HERE, "results")))

# Optional live progress bar (a local Slack bridge on the author's workstation). Entirely
# cosmetic: absent, the run just prints to its log.
_SLACK = os.environ.get("SLACK_FACADE_DIR", os.path.expanduser("~/work/slack-facade"))
if os.path.isdir(_SLACK):
    sys.path.insert(0, _SLACK)
try:
    import progress as _prog
except Exception:
    _prog = None

# source/target population codes
EXC, INH, INP, OUTP = 0, 1, 2, 3
POOL_SIZE = {EXC: N_EXC, INH: N_INH, INP: N_IN, OUTP: N_OUT}


# ----------------------------------------------------------------- genome
def seed_genome(rng, w_max, fanout_e=80, fanout_i=20, fanout_inh=100,
                fanout_in=100, fanin_out=100, res_w_exc=6.0):
    """The starting architecture, expressed as one flat synapse list.

    Same shape the generational harness used — exc->exc, exc->inh, inh->exc, input->exc,
    exc->output — but here every one of those synapses is a mutable genome entry.
    """
    sp, si, tp, ti, d, w = [], [], [], [], [], []

    def blk(spool, tpool, n_src, fan, wgen):
        s = np.repeat(np.arange(n_src), fan)
        t = rng.integers(0, POOL_SIZE[tpool], n_src * fan)
        sp.append(np.full(s.size, spool)); si.append(s)
        tp.append(np.full(s.size, tpool)); ti.append(t)
        d.append(rng.integers(D_MIN, D_MAX + 1, s.size))
        w.append(wgen(s.size))

    blk(EXC, EXC, N_EXC, fanout_e, lambda n: np.full(n, res_w_exc))
    blk(EXC, INH, N_EXC, fanout_i, lambda n: np.full(n, res_w_exc))
    blk(INH, EXC, N_INH, fanout_inh, lambda n: np.full(n, RES_W_INH))
    blk(INP, EXC, N_IN, fanout_in, lambda n: rng.uniform(0.0, w_max, n))
    # output synapses are exc -> output, so build them source-major over the output pool
    s = rng.integers(0, N_EXC, N_OUT * fanin_out)
    t = np.repeat(np.arange(N_OUT), fanin_out)
    sp.append(np.full(s.size, EXC)); si.append(s)
    tp.append(np.full(s.size, OUTP)); ti.append(t)
    d.append(rng.integers(D_MIN, D_MAX + 1, s.size))
    w.append(rng.uniform(0.0, w_max, s.size))

    g = dict(src_pool=np.concatenate(sp), src_idx=np.concatenate(si),
             tgt_pool=np.concatenate(tp), tgt_idx=np.concatenate(ti),
             delay=np.concatenate(d).astype(np.int64),
             weight=np.concatenate(w).astype(np.float64))
    return dedupe(g)


def _key(g):
    """A unique integer per (source population, target population, src idx, tgt idx)."""
    return ((g["src_pool"] * 10 + g["tgt_pool"]) * 1000 + g["src_idx"]) * 1000 + g["tgt_idx"]


def dedupe(g):
    """(src,tgt) unique — a ChunkOfConnections invariant — AND no self-loops.

    Both filters live here rather than in the add path so they hold for EVERY genome:
    the seed included. The reservoir's exc->exc block draws targets at random, so the
    seed genome contains self-loops (~1 in 800 of its 64k exc->exc synapses) unless they
    are removed here.

    A self-loop is same POOL and same INDEX — exc[7] -> exc[7]. exc[7] -> inh[7] is a
    perfectly ordinary synapse between two different neurons.
    """
    self_loop = (g["src_pool"] == g["tgt_pool"]) & (g["src_idx"] == g["tgt_idx"])
    if self_loop.any():
        g = {k: v[~self_loop] for k, v in g.items()}
    _, keep = np.unique(_key(g), return_index=True)
    keep = np.sort(keep)
    return {k: v[keep] for k, v in g.items()}


def clone(g):
    return {k: v.copy() for k, v in g.items()}


def mutate_structural(g, rng, w_max, p_add_exc=0.015, p_prune_exc=0.015,
                      p_add_inh=0.015, p_prune_inh=0.015,
                      add_weight_exc=None, weak_thresh=None):
    """PURELY STRUCTURAL mutation. Existing synapses are NEVER edited.

    No weight jitter, no delay mutation — an existing synapse's (weight, delay) is
    immutable. Evolution acts only by adding and removing synapses.

      EXCITATORY  add    random target, random delay, weight = add_weight_exc
                  remove ONLY those with |w| < weak_thresh, chosen at random among them.
                         If fewer are eligible than requested, remove only what is
                         eligible — never fall back to stronger synapses.
      INHIBITORY  add    random target, delay FIXED 1, weight FIXED RES_W_INH (-5)
                  remove uniform at random over ALL inhibitory synapses, ignoring weight.

    add_weight_exc (default 0.10*w_max) sits ABOVE weak_thresh (default 0.02*w_max), so a
    newborn excitatory synapse is not instantly prune-eligible.
    """
    add_weight_exc = 0.10 * w_max if add_weight_exc is None else add_weight_exc
    weak_thresh = 0.02 * w_max if weak_thresh is None else weak_thresh
    h = clone(g)
    is_inh = h["src_pool"] == INH
    n_exc, n_inh = int((~is_inh).sum()), int(is_inh.sum())

    # ---- removals (indices into the current arrays) --------------------------------
    drop = []
    want_e = int(p_prune_exc * n_exc)
    if want_e:
        weak = np.nonzero((~is_inh) & (np.abs(h["weight"]) < weak_thresh))[0]
        if weak.size:
            drop.append(rng.choice(weak, min(want_e, weak.size), replace=False))
    want_i = int(p_prune_inh * n_inh)
    if want_i:
        inh_i = np.nonzero(is_inh)[0]
        drop.append(rng.choice(inh_i, min(want_i, inh_i.size), replace=False))
    removed_keys = np.array([], np.int64)
    if drop:
        di = np.concatenate(drop)
        removed_keys = _key(h)[di]
        mask = np.ones(h["weight"].size, bool)
        mask[di] = False
        h = {k: v[mask] for k, v in h.items()}

    # ---- additions ------------------------------------------------------------------
    def add_block(spools, probs, n_new, inhibitory):
        if n_new <= 0:
            return None
        spool = (np.full(n_new, INH) if inhibitory
                 else rng.choice(spools, n_new, p=probs))
        tpool = (np.full(n_new, EXC) if inhibitory
                 else rng.choice([EXC, INH, OUTP], n_new, p=[0.80, 0.15, 0.05]))
        return dict(
            src_pool=spool,
            src_idx=np.array([rng.integers(0, POOL_SIZE[p]) for p in spool], np.int64),
            tgt_pool=tpool,
            tgt_idx=np.array([rng.integers(0, POOL_SIZE[p]) for p in tpool], np.int64),
            delay=(np.ones(n_new, np.int64) if inhibitory
                   else rng.integers(D_MIN, D_MAX + 1, n_new).astype(np.int64)),
            weight=np.full(n_new, RES_W_INH if inhibitory else add_weight_exc, np.float64))

    for blk in (add_block([EXC, INP], [0.85, 0.15], int(p_add_exc * n_exc), False),
                add_block(None, None, int(p_add_inh * n_inh), True)):
        if blk is not None:
            # Never re-create an address removed EARLIER IN THIS SAME CALL. Without this a
            # synapse can be pruned and immediately reborn at the same (src,tgt) with a
            # different delay, which reads — correctly — as an existing synapse's delay
            # having changed. Measured: max|delta delay| of 15 ticks on "survivors".
            keep = ~np.isin(_key(blk), removed_keys)
            blk = {k: v[keep] for k, v in blk.items()}
            # draw ONCE and reuse — see the note in the old add path
            h = dedupe({k: np.concatenate([h[k], blk[k]]) for k in h})

    # Dale, belt and braces: sources decide the sign, always.
    inh_now = h["src_pool"] == INH
    h["weight"][inh_now] = RES_W_INH
    h["weight"][~inh_now] = np.clip(h["weight"][~inh_now], 0.0, 1.5 * w_max)
    return h


def mutate(g, rng, w_max, p_add=0.015, p_prune=0.015, p_jit=0.15, w_sigma=0.15,
           target=None):
    """Dale-legal add / prune / jitter on a cloned genome.

    prune removes the WEAKEST excitatory synapses by |weight| — inhibitory ones are pinned
    at RES_W_INH so |w| carries no information about them and pruning by magnitude would
    delete them arbitrarily.
    """
    h = clone(g)
    n = h["weight"].size
    is_inh = h["src_pool"] == INH

    # ---- prune: weakest excitatory only
    n_pr = int(p_prune * n)
    if n_pr:
        exc_i = np.nonzero(~is_inh)[0]
        weak = exc_i[np.argsort(np.abs(h["weight"][exc_i]))[:n_pr]]
        mask = np.ones(n, bool)
        mask[weak] = False
        h = {k: v[mask] for k, v in h.items()}
        is_inh = h["src_pool"] == INH
        n = h["weight"].size

    # ---- jitter: excitatory weights only, clipped Dale-legal
    m = (rng.random(n) < p_jit) & (~is_inh)
    h["weight"][m] += rng.normal(0.0, w_sigma * w_max, int(m.sum()))
    h["weight"][~is_inh] = np.clip(h["weight"][~is_inh], 0.0, 1.5 * w_max)
    h["weight"][is_inh] = RES_W_INH

    # ---- add: new random synapses, source population decides the sign
    #
    # BALANCING. Asking for p_add == p_prune does NOT hold the size fixed: a freshly drawn
    # (src,tgt) can collide with an existing synapse and dedupe() then silently drops it,
    # so realised adds always undershoot. Measured drift: 97,167 -> 95,593 over 40 rounds
    # at 1.5%/1.5%. When `target` is given we therefore top up in a few passes until the
    # count is restored, instead of trusting one draw.
    def draw(k):
        spool = rng.choice([EXC, INH, INP], k, p=[0.75, 0.15, 0.10])
        tpool = np.where(spool == INH, EXC,
                         rng.choice([EXC, INH, OUTP], k, p=[0.80, 0.15, 0.05]))
        return dict(
            src_pool=spool,
            src_idx=np.array([rng.integers(0, POOL_SIZE[p]) for p in spool]),
            tgt_pool=tpool,
            tgt_idx=np.array([rng.integers(0, POOL_SIZE[p]) for p in tpool]),
            delay=rng.integers(D_MIN, D_MAX + 1, k).astype(np.int64),
            weight=np.where(spool == INH, RES_W_INH, rng.uniform(0.0, w_max, k)))

    def append(hh, k):
        # draw() ONCE and reuse it. Calling it inside the dict comprehension re-invokes it
        # per key, so every field would come from a DIFFERENT random block and the rows
        # would be silently misaligned (an inhibitory row could get an excitatory index).
        nw = draw(k)
        return dedupe({key: np.concatenate([hh[key], nw[key]]) for key in hh})

    n_ad = int(p_add * n)
    if n_ad:
        h = append(h, n_ad)
    if target is not None:
        for _ in range(6):
            short = target - h["weight"].size
            if short <= 0:
                break
            h = append(h, int(short * 1.3) + 8)
        if h["weight"].size > target:
            # trim at random, NOT by slicing: dedupe leaves the arrays sorted by key, so
            # v[:target] would drop whole (src_pool, tgt_pool) groups from one end.
            keep = np.sort(rng.choice(h["weight"].size, target, replace=False))
            h = {k: v[keep] for k, v in h.items()}
    return h


# ----------------------------------------------------------------- build
# Engine-side synapse_group_size for the explicit build. Was pinned at GROUP_SIZE (2) because
# every larger value crashed create_forward_groups; that was the input_weights sign-flip bug
# (negative shift_to_next_group promoted to unsigned), now fixed in the engine, so we can use
# the ipynb-style 128 and get ~64x fewer blocks per source.
ENGINE_GROUP_SIZE = 128


def stage2_metas(stdp_lr, w_max, group_size=8, backward_group_size=32):
    """Two meta banks: PLASTIC excitatory (indices 0..19) and FROZEN inhibitory (20..39).

    The meta index has to carry the delay AND the plasticity class, because learning_rate
    is a per-meta field. Excitatory metas get learning_rate = stdp_lr and the [0, w_max]
    bounds STDP will clip to; inhibitory metas keep learning_rate 0 and pin the weight at
    RES_W_INH, so inhibition stays structural-only exactly as stage one specified.

    group_size DEFAULTS TO GROUP_SIZE (2) AND MUST STAY THERE. It used to hardcode 8, which
    was the whole stage-two build crash: multi-meta explicit wiring is broken at every group
    size except 2 (measured -- 4, 8 and 16 all die with cudaErrorIllegalAddress in
    create_forward_groups at connections_manager.cu:126, and gs=1 is rejected as odd). Stage
    one never hit it because delay_metas() already defaults to GROUP_SIZE. Binding the
    default to the same constant the SynapseGrowthEngine uses keeps the two from drifting
    apart again.

    backward_group_size is SEPARATE and must NOT follow it down to 2. The backward structure
    is what STDP walks, and it overflows once a target neuron receives more than ~8 incoming
    PLASTIC synapses: measured on the real net, exc->exc fanout 2/4/8 train fine while
    16/32/64/80 die in process_ticks(do_train=True) at spnet_runtime.cu:1030. Here every
    excitatory neuron receives ~80 and every output neuron ~100, so 2 is far past the edge.
    Sweeping the backward size alone with forward pinned at 2: 2/4/8 crash, 16/32/64/128
    all pass. 32 leaves headroom above the ~100 worst case without padding as hard as 128.
    """
    from spiky.spnet.spnet import SynapseMeta
    exc = [SynapseMeta(learning_rate=stdp_lr, min_delay=d, max_delay=d,
                       initial_weight=0.0, min_weight=0.0, max_weight=1.5 * w_max,
                       initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=group_size,
                       _backward_group_size=backward_group_size)
           for d in range(D_MIN, D_MAX + 1)]
    inh = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                       initial_weight=RES_W_INH, min_weight=RES_W_INH,
                       max_weight=RES_W_INH, initial_noise_level=0.0,
                       weight_decay=0.9, weight_scaling_cf=0.0,
                       _forward_group_size=group_size,
                       _backward_group_size=backward_group_size)
           for d in range(D_MIN, D_MAX + 1)]
    return exc + inh


N_DELAY_METAS = D_MAX - D_MIN + 1


def build_pool(genomes, device="cuda", seed=1, stdp_lr=0.0, w_max=30.0, drives=None):
    """Pack the pool into ONE SpikingNet, one explicit chunk, chain-following weights."""
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine
    from spiky.util.chunk_of_connections import ChunkOfConnections

    K = len(genomes)
    # d_max=D_MAX is NOT redundant: delay_metas' own default is bound at import from
    # harness.D_MAX, so omitting it silently pins the bank at 20 delays and ignores --d-max.
    metas = stage2_metas(stdp_lr, w_max) if stdp_lr > 0 else delay_metas(d_max=D_MAX)
    neuron_metas = [NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=1, a=0.1, d=2.0),
                    NeuronMeta(neuron_type=2, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=3, a=0.02, d=8.0)]
    counts = [K * N_EXC, K * N_INH, K * N_IN, K * N_OUT]
    sp = SpikingNet(synapse_metas=metas, neuron_metas=neuron_metas, neuron_counts=counts,
                    initial_synapse_capacity=1 << 23, summation_dtype=torch.float32)
    sp.to_device(device)
    ids = [sp.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]
    base = {EXC: N_EXC, INH: N_INH, INP: N_IN, OUTP: N_OUT}

    tri, wts = [], []
    for c, g in enumerate(genomes):
        s = np.empty(g["weight"].size, np.int64)
        t = np.empty_like(s)
        for p in (EXC, INH, INP):
            m = g["src_pool"] == p
            if m.any():
                s[m] = ids[p][c * base[p] + g["src_idx"][m]]
        for p in (EXC, INH, OUTP):
            m = g["tgt_pool"] == p
            if m.any():
                t[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
        meta = (g["delay"] - D_MIN).copy()
        if stdp_lr > 0:                      # inhibitory synapses use the frozen bank
            meta[g["src_pool"] == INH] += N_DELAY_METAS
        tri.append(np.stack([meta, s, t], 1))
        # EVOLVABLE OUTPUT DRIVE: scale the weight of every synapse landing on this net's
        # OUTPUT neurons. Per-net neuron parameters are impossible here (all K nets share the
        # four NeuronMetas of one SpikingNet), so the drive is expressed on the readout
        # synapses instead -- lower scale = less current = later, more spread-out first
        # spikes. The genome keeps the UNSCALED weights; readback divides the scale back out.
        wg = g["weight"]
        if drives is not None and drives[c] != 1.0:
            wg = wg.copy()
            wg[g["tgt_pool"] == OUTP] *= drives[c]
        wts.append(wg)
    triples = np.concatenate(tri, 0)
    weights = np.concatenate(wts, 0)

    total = sum(counts)
    ge = SynapseGrowthEngine(device=device, synapse_group_size=ENGINE_GROUP_SIZE,
                             max_groups_in_buffer=max(4096, 8 * (len(triples) + total)))
    for i in range(4):
        ge.register_neuron_type(max_synapses=8 * (N_EXC + N_INH), growth_command_list=[])
    for i in range(4):
        tt = torch.tensor(ids[i], dtype=torch.int32)
        n = tt.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=tt,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))
    tri_t = torch.tensor(triples, dtype=torch.int32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    chunk = ge._grow_explicit(tri_t, seed, weights=w_t)
    sp.add_connections(chunk, seed)
    chunk.recycle()
    sp.compile(shuffle_synapses_random_seed=None)
    return dict(spnet=sp, ids=ids, P=K, device=device, n_syn=int(len(triples)))


GENOME_FIELDS = ("src_pool", "src_idx", "tgt_pool", "tgt_idx", "delay", "weight")


def build_eval_pool(genome, device, stdp_lr, w_max, drive=None, seed=1):
    """Build a ONE-MEMBER pool for scoring a saved/selected genome.

    THE SINGLE ENTRY POINT for every evaluation build -- main()'s final held-out score and
    eval_heldout.py both come through here -- because getting this wrong is silent.

    A genome's `delay` column is an index into the meta bank the run trained against, and
    build_pool picks that bank off `stdp_lr`: >0 gives stage2_metas' 2*(D_MAX-D_MIN+1) plastic +
    frozen banks with inhibitory synapses offset by N_DELAY_METAS, 0 gives a single unoffset
    bank. Evaluate with the wrong one and the indices mean something else.

    The original held-out line was `build_pool([genomes[best_i]], dev, seed=1)` -- no stdp_lr,
    no w_max, no drive. Two consequences, one loud and one quiet:
      * LOUD: for d_max 48 the genome carries meta indices up to 95 into a 20-meta network. The
        engine reads out of range and requests a nonsense allocation, surfacing as
        cudaErrorMemoryAllocation. That is what killed exp004's evaluation and every one of its
        40 supervisor retries -- it was never short of memory.
      * QUIET: at d_max 20 the indices stay in range, so nothing crashes, but inhibitory
        synapses land in the excitatory bank instead of the frozen one. The number that comes
        out is for a different network than the one that was selected.

    `drive` is the member's evolved output-drive scale, which the old path also dropped: it
    scored the genome at drive 1.0 rather than at the value selection actually chose it with.
    """
    return build_pool([genome], device, seed=seed, stdp_lr=stdp_lr, w_max=w_max,
                      drives=None if drive is None else np.asarray([float(drive)], np.float64))


def build_pool_retry(genomes, device, seed, stdp_lr, w_max, rnd, retries=5, drives=None):
    """build_pool, retried on the create_forward_groups race.

    THE RACE: build_pool -> add_connections -> create_forward_groups misbehaves
    nondeterministically on BYTE-IDENTICAL input -- measured ~10% of builds at K=128, ~0.7%
    at K=32 (see BUGS.md and probe_build_flaky.py). It surfaces two ways:
      (a) it RAISES "some error happened inside create_forward_groups kernel". Recoverable:
          the raise is host-side, the context is intact, and a retry on the same genomes
          succeeds ~90% of the time. That is what this function handles.
      (b) it HANGS, spinning a GPU kernel forever. NOT recoverable here, and no amount of
          Python can fix it: you cannot interrupt or kill a running CUDA kernel from the
          host, and the context is wedged even if you could. Only killing the PROCESS works
          -- which is why the round loop checkpoints every round and supervise_run.py
          restarts a stalled run with --resume. Do not add a thread-based timeout here
          expecting it to help; it would detect the stall but be unable to act on it.
    """
    for attempt in range(retries + 1):
        try:
            return build_pool(genomes, device, seed=seed, stdp_lr=stdp_lr, w_max=w_max,
                              drives=drives)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  [retry] round {rnd}: build attempt {attempt + 1}/{retries + 1} "
                  f"failed ({type(e).__name__}: {str(e)[:80]}), retrying same genomes",
                  flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(1.0)


def pool_scalars(genomes, w_max):
    """Cheap per-round structure summary, so drift is visible between snapshots.

    All numpy over the in-memory genomes -- no device work. Thresholds match the ones the
    evolution itself uses: DEAD is the prune threshold (0.02*w_max), the ceiling is the meta's
    max_weight (1.5*w_max) where STDP clips.
    """
    dead_t, ceil = 0.02 * w_max, 1.5 * w_max
    we = np.concatenate([g["weight"][g["src_pool"] != INH] for g in genomes])
    dl = np.concatenate([g["delay"][g["src_pool"] != INH] for g in genomes])
    aw = np.abs(we)
    return dict(exc_dead_frac=float((aw < dead_t).mean()),
                exc_clip_frac=float((we >= ceil - 1e-3).mean()),
                exc_w_median=float(np.median(we)), exc_w_mean=float(we.mean()),
                w_weighted_mean_delay=float((aw * dl).sum() / max(aw.sum(), 1e-9)))


def save_ckpt(path, genomes, ewma, age, newborn, hist, next_rnd, rng=None, lineage=None,
              drives=None):
    """Checkpoint every round so a supervisor can restart a wedged process without losing
    the run. Written to a temp file and renamed, so a kill mid-write cannot corrupt it."""
    d = {f"g{i}_{f}": g[f] for i, g in enumerate(genomes) for f in GENOME_FIELDS}
    d.update(ewma=ewma, age=age, newborn=newborn, next_rnd=np.array([next_rnd]),
             n_genomes=np.array([len(genomes)]))
    if lineage is not None:
        d["lineage"] = np.asarray(lineage)
    if drives is not None:
        d["drives"] = np.asarray(drives, np.float64)
    if rng is not None:
        # Without this the RNG restarts from the seed on resume, so parent selection and
        # mutation diverge from an uninterrupted run — the resumed run stays valid but is
        # no longer reproducible. Cheap to carry, so carry it.
        d["rng_state"] = np.array([json.dumps(rng.bit_generator.state)])
    np.savez_compressed(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)
    json.dump(hist, open(path + ".hist.json", "w"))


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    K = int(z["n_genomes"][0])
    genomes = [{f: z[f"g{i}_{f}"] for f in GENOME_FIELDS} for i in range(K)]
    hist = json.load(open(path + ".hist.json")) if os.path.exists(path + ".hist.json") else []
    rng_state = json.loads(str(z["rng_state"][0])) if "rng_state" in z.files else None
    # `lineage` is new; checkpoints written before it exists must still resume, so fall back
    # to "every slot is its own lineage". This is what keeps a live run restart-safe.
    lineage = z["lineage"].copy() if "lineage" in z.files else np.arange(K)
    drives = z["drives"].copy() if "drives" in z.files else np.ones(K)
    return (genomes, z["ewma"], z["age"], z["newborn"], hist, int(z["next_rnd"][0]),
            rng_state, lineage, drives)


def readback(h, genomes, drives=None):
    """LAMARCKIAN STEP: pull every candidate's CURRENT on-device weights into its genome.

    With STDP on, the device weights diverge from the in-memory genome the moment a batch
    is processed. The rebuild reconstructs the net FROM the genomes, so without this the
    rebuild would silently reset every synapse to its birth weight and throw away all STDP
    progress — and a clone would inherit its parent's birth weights rather than its matured
    ones, which is the whole point of the Lamarckian design.

    Matches on (global src, global tgt), which is unique per net, and writes back only
    excitatory weights: inhibitory ones are frozen at RES_W_INH by their meta and must not
    be touched.
    """
    sp, ids, dev = h["spnet"], h["ids"], h["device"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=dev)
    n = sp.count_synapses(all_ids, True)
    b = [torch.zeros(n, dtype=t, device=dev) for t in
         (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, b[0], b[1], b[2], b[3], b[4], True)
    s, _, w, _, t = (x.cpu().numpy() for x in b)
    live = {}
    for a_, b_, c_ in zip(s, t, w):
        live[(int(a_), int(b_))] = float(c_)

    base = {EXC: N_EXC, INH: N_INH, INP: N_IN, OUTP: N_OUT}
    changed = 0
    for c, g in enumerate(genomes):
        gs = np.empty(g["weight"].size, np.int64)
        gt = np.empty_like(gs)
        for p in (EXC, INH, INP):
            m = g["src_pool"] == p
            if m.any():
                gs[m] = ids[p][c * base[p] + g["src_idx"][m]]
        for p in (EXC, INH, OUTP):
            m = g["tgt_pool"] == p
            if m.any():
                gt[m] = ids[p][c * base[p] + g["tgt_idx"][m]]
        exc = np.nonzero(g["src_pool"] != INH)[0]
        # Output synapses were SCALED by this net's drive gene at build time, so what comes
        # back off the device is (genome weight * drive), further modified by STDP. Divide the
        # scale out, or the genome compounds it every single round.
        dsc = 1.0 if drives is None else float(drives[c])
        is_out = g["tgt_pool"] == OUTP
        for i in exc:
            v = live.get((int(gs[i]), int(gt[i])))
            if v is None:
                continue
            if dsc != 1.0 and is_out[i]:
                v = v / dsc
            if v != g["weight"][i]:
                g["weight"][i] = v
                changed += 1
    return changed


def stdp_batches(h, Xpool, Ypool, enc, batch, seed, rounds, n_batches, current=200.0):
    """Run n_batches of pure STDP through the WHOLE pool — no measurement, no selection.

    Used for newborn maturation. Maturation is counted in BATCHES, not rounds: with K=32
    and M=8 culled per round, a 32-ROUND immunity would leave fewer than M matured members
    and selection would stall outright. Batches are decoupled from the round cadence, so
    newborns settle without freezing the tournament.
    """
    for i in range(n_batches):
        Xb, _, _ = sample_batch(Xpool, Ypool, batch, seed, rounds * 1000 + i)
        run_episode(h, Xb, enc, current, train=True)


def tie_rate_per_member(first, n_out=N_OUT):
    """Fraction of (state, dim-pair) cells where two outputs share the SAME first-spike tick.

    first is [B, P, N_OUT] of integer ticks, so two outputs firing on the same tick tie
    EXACTLY. tau-B discounts such ties (it drops them from its denominator), which leaves
    evolution indifferent to collisions -- and the analytics showed the six outputs collapsing
    into only ~3.4-4.7 distinct ticks with >20% of pairs tied. Returned per member so a
    penalty can push against that.
    """
    iu, ju = np.triu_indices(n_out, 1)
    d = np.sign(-first[:, :, iu] + first[:, :, ju])      # [B, P, 15]; pred = -first
    return (d == 0).mean(axis=(0, 2))                    # [P]


def score(h, X, Y, enc, current=200.0, tie_penalty=0.0):
    """Corrected tau per pool member: raw tau minus that member's OWN null.

    With tie_penalty > 0 the fitness becomes  corrected_tau - lambda * tie_rate, so a net that
    encodes six actions in three distinguishable ticks is charged for it. tau-B itself is
    computed exactly as before; the penalty is a separate additive term.
    """
    first, _ = run_episode(h, X, enc, current)
    out = []
    for p in range(h["P"]):
        pred = -first[:, p, :]
        raw = float(kendall_tau_b(pred, Y).mean())
        nl = float(np.mean([kendall_tau_b(pred, Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
                            for k in range(40)]))
        out.append(raw - nl)
    fit = np.array(out)
    ties = tie_rate_per_member(first)
    if tie_penalty:
        fit = fit - tie_penalty * ties
    return fit, first, ties


# ----------------------------------------------------------------- the loop
def main():
    # declared up front: the architecture knobs below rebind these module-level values
    global N_EXC, N_INH, POOL_SIZE, D_MAX, N_DELAY_METAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--cull", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--grace", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--w-max", type=float, default=30.0)
    ap.add_argument("--current", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--p-add-exc", type=float, default=0.015)
    ap.add_argument("--p-prune-exc", type=float, default=0.015)
    ap.add_argument("--p-add-inh", type=float, default=0.015)
    ap.add_argument("--p-prune-inh", type=float, default=0.015)
    ap.add_argument("--add-weight-exc", type=float, default=None,
                    help="initial weight of a new excitatory synapse (default 0.10*w_max)")
    ap.add_argument("--stdp-lr", type=float, default=0.0,
                    help="STAGE TWO: >0 turns STDP on for excitatory synapses only")
    ap.add_argument("--mature-batches", type=int, default=32,
                    help="pure-STDP batches a newborn gets before being measured/culled")
    ap.add_argument("--weak-thresh", type=float, default=None,
                    help="|w| below which an excitatory synapse is prune-eligible "
                         "(default 0.02*w_max)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; written every round so a stalled run can be "
                         "restarted with --resume (see supervise_run.py)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --ckpt if it exists")
    ap.add_argument("--build-retries", type=int, default=5)
    # --- architecture knobs (defaults reproduce the 800/200 net exactly) ---
    ap.add_argument("--n-exc", type=int, default=N_EXC)
    ap.add_argument("--n-inh", type=int, default=N_INH)
    ap.add_argument("--fanout-scale", type=float, default=1.0,
                    help="divide every seed fan-out by this, e.g. 10 takes per-hidden-neuron "
                         "fan-in from ~100 to ~10")
    ap.add_argument("--d-max", type=int, default=D_MAX,
                    help="max excitatory delay; the meta banks are derived from it, so 48 "
                         "gives 48 exc + 48 inh = 96 metas instead of the default 40")
    # --- analysis instrumentation (all additive; defaults keep prior behaviour) ---
    ap.add_argument("--snapshot-every", type=int, default=25,
                    help="also keep an un-overwritten genome snapshot every N rounds "
                         "(0 disables); lets weight/delay DRIFT be reconstructed later")
    ap.add_argument("--no-fitness-log", action="store_true",
                    help="skip the per-round per-member fitness sidecar")
    ap.add_argument("--tie-penalty", type=float, default=0.0,
                    help="fitness becomes corrected_tau - LAMBDA * tie_rate. 0.0 (default) "
                         "reproduces the previous fitness exactly")
    # evolvable output-drive gene (off by default -> previous behaviour exactly)
    ap.add_argument("--drive-gene", action="store_true",
                    help="per-net evolvable scale on the synaptic drive into the 6 output "
                         "neurons; lets selection choose its own output timing")
    ap.add_argument("--drive-lo", type=float, default=0.3)
    ap.add_argument("--drive-hi", type=float, default=1.5)
    ap.add_argument("--drive-sigma", type=float, default=0.2,
                    help="mutation is drive *= exp(N(0, sigma))")
    ap.add_argument("--drive-p", type=float, default=0.5,
                    help="probability a newborn's drive gene mutates")
    ap.add_argument("--drive-clip", type=float, nargs=2, default=[0.1, 3.0])
    a = ap.parse_args()

    # Delay range drives the meta banks (stage2_metas iterates D_MIN..D_MAX) and the
    # inhibitory bank offset, so N_DELAY_METAS must be refreshed alongside it.
    D_MAX = a.d_max
    N_DELAY_METAS = D_MAX - D_MIN + 1

    # Rebind the pool sizes everywhere they are cached. N_EXC/N_INH come from harness and
    # are baked into POOL_SIZE at import, so both must be refreshed together or the genome
    # index ranges and the build will disagree.
    N_EXC, N_INH = a.n_exc, a.n_inh
    POOL_SIZE = {EXC: N_EXC, INH: N_INH, INP: N_IN, OUTP: N_OUT}
    fs = a.fanout_scale
    fanouts = dict(fanout_e=max(1, round(80 / fs)), fanout_i=max(1, round(20 / fs)),
                   fanout_inh=max(1, round(100 / fs)), fanout_in=max(1, round(100 / fs)),
                   fanin_out=max(1, round(100 / fs)))
    # Neuron counts are fixed FOR A GIVEN RUN; evolution never changes them. Input/output are
    # pinned by the dataset (17 observations -> 6 actions); the hidden pool is configurable.
    assert (N_IN, N_OUT) == (17, 6), f"input/output must be 17/6, got {(N_IN, N_OUT)}"
    print(f"architecture: {N_IN} in / {N_EXC} exc / {N_INH} inh / {N_OUT} out, "
          f"fan-outs {fanouts} (scale {fs}x)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    X, Y, Xpool, Ypool, Xval, Yval = load(a.batch, a.seed, a.n_val)
    enc = LatencyEncoder(Xpool)
    M = max(1, int(a.cull * a.pool))

    start_rnd = 0
    hist_resumed = []
    if a.resume and a.ckpt and os.path.exists(a.ckpt):
        (genomes, ewma, age, newborn, hist_resumed, start_rnd,
         rng_state, lineage, drives) = load_ckpt(a.ckpt)
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        print(f"RESUMED from {a.ckpt} at round {start_rnd} "
              f"({len(genomes)} genomes, {len(hist_resumed)} rounds of history, "
              f"rng {'restored' if rng_state else 'RESEEDED'})")
    else:
        genomes = [seed_genome(np.random.default_rng(a.seed * 100 + i), a.w_max, **fanouts)
                   for i in range(a.pool)]
        ewma = np.full(a.pool, np.nan)
        age = np.zeros(a.pool, int)
        newborn = np.zeros(a.pool, bool)
        lineage = np.arange(a.pool)     # every seed member starts its own lineage
        # output-drive gene, seeded log-uniform so the pool starts with varied output timing
        drives = (np.exp(rng.uniform(np.log(a.drive_lo), np.log(a.drive_hi), a.pool))
                  if a.drive_gene else np.ones(a.pool))
    n_changed = 0
    # per-round per-member fitness, so true pool variance is recoverable instead of the
    # (best - mean) proxy. Sidecar npz rather than the history JSON to keep that file small.
    fit_path = os.path.join(OUT, f"steady_state{a.tag}_fitness.npz")
    fit_rows = []
    if a.resume and os.path.exists(fit_path):
        try:
            fit_rows = [r for r in np.load(fit_path)["f"]]
        except Exception:
            fit_rows = []
    snap_dir = os.path.join(OUT, "snapshots")
    if a.snapshot_every:
        os.makedirs(snap_dir, exist_ok=True)
    # parent -> child events, enough to rebuild the ancestry tree offline
    lineage_events = []
    print(f"steady state: pool K={a.pool}, cull M={M}/round, alpha={a.alpha}, "
          f"grace={a.grace} rounds, {genomes[0]['weight'].size:,} synapses/genome, dev {dev}")

    bar = None
    if a.task and _prog:
        bar = _prog.progress_start(f"steady-state ALife · K={a.pool} · {a.rounds} rounds",
                                   task=a.task)
    hist, t0 = hist_resumed, time.time()
    try:
        for rnd in range(start_rnd, a.rounds):
            Xb, Yb, _ = sample_batch(Xpool, Ypool, a.batch, a.seed, rnd)
            h = build_pool_retry(genomes, dev, 1, a.stdp_lr, a.w_max, rnd,
                                 retries=a.build_retries, drives=drives)
            if a.stdp_lr > 0 and newborn.any():
                # newborns settle under pure STDP before anyone is judged
                stdp_batches(h, Xpool, Ypool, enc, a.batch, a.seed, rnd,
                             a.mature_batches, a.current)
                newborn[:] = False
            f, first_ticks, ties = score(h, Xb, Yb, enc, a.current, a.tie_penalty)
            n_syn = h["n_syn"]
            if a.stdp_lr > 0:
                n_changed = readback(h, genomes, drives)   # LAMARCKIAN: device -> genome
            del h
            ewma = np.where(np.isnan(ewma), f, (1 - a.alpha) * ewma + a.alpha * f)
            age += 1

            # cull the worst M among those past their grace period
            eligible = np.nonzero(age > a.grace)[0]
            if eligible.size >= M:
                worst = eligible[np.argsort(ewma[eligible])[:M]]
                surv = np.setdiff1d(np.arange(a.pool), worst)
                # fitness-weighted parents, tournament of 2
                for slot in worst:
                    c1, c2 = rng.choice(surv, 2, replace=False)
                    par = c1 if ewma[c1] >= ewma[c2] else c2
                    genomes[slot] = mutate_structural(
                        clone(genomes[par]), rng, a.w_max,
                        p_add_exc=a.p_add_exc, p_prune_exc=a.p_prune_exc,
                        p_add_inh=a.p_add_inh, p_prune_inh=a.p_prune_inh,
                        add_weight_exc=a.add_weight_exc, weak_thresh=a.weak_thresh)
                    ewma[slot] = ewma[par]      # inherit, so a newborn is not instantly worst
                    age[slot] = 0
                    newborn[slot] = True
                    # a child continues its PARENT'S lineage; distinct surviving lineages is
                    # then a direct measure of how much real ancestral diversity is left
                    lineage_events.append((int(rnd), int(par), int(slot),
                                           int(lineage[par])))
                    lineage[slot] = lineage[par]
                    d_new = drives[par]
                    if a.drive_gene and rng.random() < a.drive_p:
                        d_new = float(np.clip(d_new * np.exp(rng.normal(0, a.drive_sigma)),
                                              a.drive_clip[0], a.drive_clip[1]))
                    drives[slot] = d_new
            rec = dict(rnd=rnd, best=float(np.nanmax(ewma)), mean=float(np.nanmean(ewma)),
                       batch_best=float(f.max()), batch_mean=float(f.mean()),
                       n_syn=n_syn, n_exc=int(sum((g["src_pool"] != INH).sum() for g in genomes)),
                       n_inh=int(sum((g["src_pool"] == INH).sum() for g in genomes)),
                       n_changed=n_changed, wall=round(time.time() - t0, 1))
            # true pool spread, no longer a proxy, plus structure drift and lineage count
            rec.update(pool_std=float(np.nanstd(ewma)),
                       pool_min=float(np.nanmin(ewma)), pool_max=float(np.nanmax(ewma)),
                       n_lineages=int(np.unique(lineage).size))
            # tie diagnostics: does the penalty actually spread the output code over time?
            _bi = int(np.nanargmax(ewma))
            _bt = first_ticks[:, _bi, :].astype(int)
            # distinct first-spike ticks per state, for EVERY member -- sort along the output
            # axis and count the changes, so it is one vectorised pass over [B, P, N_OUT]
            # rather than B*P calls to np.unique. Cheap enough to log every round.
            _srt = np.sort(first_ticks.astype(np.int64), axis=-1)
            distinct_ticks = (1 + (np.diff(_srt, axis=-1) != 0).sum(-1)).mean(0)
            rec.update(tie_rate_mean=float(ties.mean()),
                       tie_rate_best=float(ties[_bi]),
                       distinct_ticks_best=float(np.mean([len(np.unique(r)) for r in _bt])),
                       tick_std_best=float(_bt.std()),
                       drive_best=float(drives[_bi]), drive_mean=float(np.mean(drives)),
                       drive_min=float(np.min(drives)), drive_max=float(np.max(drives)))
            # PER-MEMBER VECTORS, length K, in pool-slot order. Everything above is a scalar
            # summary, which is why no run before this one can show per-round pool variance --
            # only the final checkpoint's `ewma` array survived, so every diversity claim in
            # this chapter had to lean on (best - mean) as a proxy. These are the real thing.
            #
            # `fitness_vec` is the ARRAY SELECTION USED, not a recomputation: `f` comes straight
            # out of score() above and is untouched by the cull. `ewma_vec` is POST-cull, so a
            # newborn carries its parent's inherited value -- that is the number the next round
            # will actually judge it on, but it is not a measurement of the newborn.
            rec.update(fitness_vec=[round(float(v), 6) for v in f],
                       ewma_vec=[round(float(v), 6) for v in ewma],
                       tie_rate_vec=[round(float(v), 6) for v in ties],
                       distinct_ticks_vec=[round(float(v), 4) for v in distinct_ticks],
                       age_vec=[int(v) for v in age],
                       lineage_vec=[int(v) for v in lineage])
            rec.update(pool_scalars(genomes, a.w_max))
            hist.append(rec)
            if not a.no_fitness_log:
                fit_rows.append(np.asarray(f, np.float32))
                np.savez_compressed(fit_path, f=np.stack(fit_rows),
                                    lineage_events=np.asarray(lineage_events, np.int64)
                                    if lineage_events else np.zeros((0, 4), np.int64))
            if a.snapshot_every and (rnd % a.snapshot_every == 0
                                     or rnd == a.rounds - 1):
                save_ckpt(os.path.join(snap_dir, f"{a.tag.lstrip('_') or 'run'}_r{rnd:04d}.npz"),
                          genomes, ewma, age, newborn, [], rnd + 1, rng, lineage, drives)
            if rnd % 5 == 0 or rnd == a.rounds - 1:
                print(f"  round {rnd:3d}  batch best {f.max():+.4f} mean {f.mean():+.4f}  "
                      f"EWMA best {np.nanmax(ewma):+.4f} mean {np.nanmean(ewma):+.4f}  "
                      f"syn {n_syn//a.pool:,}/net  {rec['wall']:.0f}s", flush=True)
                if bar and _prog:
                    el = time.time() - t0
                    _prog.progress_update(
                        bar, step=rnd + 1, total=a.rounds,
                        stats=f"round {rnd+1}/{a.rounds} · EWMA best "
                              f"{np.nanmax(ewma):+.4f} · eta ~"
                              f"{el/(rnd+1)*(a.rounds-rnd-1)/60:.1f}m")
            json.dump(hist, open(os.path.join(OUT, f"steady_state{a.tag}.json"), "w"), indent=1)
            if a.ckpt:
                # next_rnd = rnd+1: genomes/ewma/age/newborn are already post-cull, so a
                # resume must start at the FOLLOWING round or it would cull twice.
                save_ckpt(a.ckpt, genomes, ewma, age, newborn, hist, rnd + 1, rng, lineage,
                          drives)
    except BaseException:
        if bar and _prog:
            _prog.progress_done(bar, ok=False, final_text="run failed — see log")
        raise

    # final: score the best genome on the held-out set. Through build_eval_pool, which carries
    # the run's OWN meta bank (stdp_lr/w_max/D_MAX) and the member's own drive -- see the note
    # there for what dropping any of them silently does.
    best_i = int(np.nanargmax(ewma))
    hb = build_eval_pool(genomes[best_i], dev, a.stdp_lr, a.w_max,
                         drive=drives[best_i] if a.drive_gene else None)
    fv, _, _ = score(hb, Xval, Yval, enc, a.current, a.tie_penalty)
    print(f"\nBEST member {best_i}: EWMA {ewma[best_i]:+.4f}  HELD-OUT corrected "
          f"{fv[0]:+.4f}  ({genomes[best_i]['weight'].size:,} synapses)")
    if bar and _prog:
        _prog.progress_done(bar, ok=True,
                            final_text=f"held-out corrected {fv[0]:+.4f} after "
                                       f"{a.rounds} rounds")


if __name__ == "__main__":
    main()

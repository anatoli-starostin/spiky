#!/usr/bin/env python3
"""
Quick benchmark: single model vs N parallel models with streams.
Helps diagnose the source of parallelism overhead.
"""
import os, sys, time
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformer_exps.evolve import (
    EvolveTransformer, random_genome, _make_optimizer,
    DEVICE, BATCH_SIZE, BOS_ID, N_STEPS, VOCAB,
)
from transformer_exps.common import make_sampler

DEVICE = 'cuda:0'
WARMUP = 50
MEASURE = 200

sampler = make_sampler(DEVICE, random_seed=1)
rng = np.random.default_rng(42)


def single_step(model, optimizer):
    batch = sampler.sample_training_batch(BATCH_SIZE).long()
    inp = batch.clone()
    inp[:, 0] = BOS_ID
    inp[:, 1:] = batch[:, :-1]
    logits = model(inp)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), batch.reshape(B * T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def bench_single(nap=4):
    g = random_genome(nap, rng)
    model = EvolveTransformer(g).to(DEVICE)
    model.train()
    opt = _make_optimizer(model)

    for _ in range(WARMUP):
        single_step(model, opt)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        single_step(model, opt)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / MEASURE * 1000
    print(f"[single nap={nap}] {ms_per_step:.2f} ms/step  ({MEASURE} steps in {elapsed:.2f}s)")
    return ms_per_step


def bench_stream_loop(N=8, nap=4):
    """Sequential python loop submitting to N streams."""
    genomes = [random_genome(nap, rng) for _ in range(N)]
    models = [EvolveTransformer(g).to(DEVICE) for g in genomes]
    opts = [_make_optimizer(m) for m in models]
    streams = [torch.cuda.Stream(device=DEVICE) for _ in range(N)]
    for m in models: m.train()

    def one_round():
        raw = []
        for i in range(N):
            with torch.cuda.stream(streams[i]):
                batch = sampler.sample_training_batch(BATCH_SIZE).long()
                inp = batch.clone()
                inp[:, 0] = BOS_ID
                inp[:, 1:] = batch[:, :-1]
                logits = models[i](inp)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), batch.reshape(B * T))
                opts[i].zero_grad()
                loss.backward()
                raw.append(loss)
        torch.cuda.synchronize()
        for i in range(N):
            opts[i].step()

    for _ in range(WARMUP):
        one_round()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        one_round()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_round = elapsed / MEASURE * 1000
    ms_per_model = ms_per_round / N
    print(f"[streams loop N={N} nap={nap}] {ms_per_round:.2f} ms/round  "
          f"= {ms_per_model:.2f} ms/model  (ratio vs single: {ms_per_model / bench_single_ms:.2f}x)")


def bench_threads(N=8, nap=4):
    """Python threading to submit work to N streams in true parallel."""
    import threading
    genomes = [random_genome(nap, rng) for _ in range(N)]
    models = [EvolveTransformer(g).to(DEVICE) for g in genomes]
    opts = [_make_optimizer(m) for m in models]
    streams = [torch.cuda.Stream(device=DEVICE) for _ in range(N)]
    for m in models: m.train()

    def worker(i, results):
        with torch.cuda.stream(streams[i]):
            batch = sampler.sample_training_batch(BATCH_SIZE).long()
            inp = batch.clone()
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1]
            logits = models[i](inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), batch.reshape(B * T))
            opts[i].zero_grad()
            loss.backward()
            results[i] = loss

    def one_round():
        results = [None] * N
        threads = [threading.Thread(target=worker, args=(i, results)) for i in range(N)]
        for t in threads: t.start()
        for t in threads: t.join()
        torch.cuda.synchronize()
        for i in range(N):
            opts[i].step()

    for _ in range(WARMUP):
        one_round()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(MEASURE):
        one_round()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_round = elapsed / MEASURE * 1000
    ms_per_model = ms_per_round / N
    print(f"[threads N={N} nap={nap}]      {ms_per_round:.2f} ms/round  "
          f"= {ms_per_model:.2f} ms/model  (ratio vs single: {ms_per_model / bench_single_ms:.2f}x)")


print("=" * 60)
print("  Parallelism benchmark")
print("=" * 60)
print()

bench_single_ms = bench_single(nap=4)
print()
bench_stream_loop(N=8, nap=4)
print()
bench_threads(N=8, nap=4)
print()

# Also check if mixing naps hurts
print("--- Mixed nap (as in actual evolve.py initial pop) ---")
rng2 = np.random.default_rng(99)
mixed_genomes = [random_genome(n, rng2) for n in [3,3,4,4,5,5,4,4]]
mixed_models = [EvolveTransformer(g).to(DEVICE) for g in mixed_genomes]
mixed_opts = [_make_optimizer(m) for m in mixed_models]
mixed_streams = [torch.cuda.Stream(device=DEVICE) for _ in range(8)]
for m in mixed_models: m.train()

def one_round_mixed():
    raw = []
    for i in range(8):
        with torch.cuda.stream(mixed_streams[i]):
            batch = sampler.sample_training_batch(BATCH_SIZE).long()
            inp = batch.clone()
            inp[:, 0] = BOS_ID
            inp[:, 1:] = batch[:, :-1]
            logits = mixed_models[i](inp)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), batch.reshape(B * T))
            mixed_opts[i].zero_grad()
            loss.backward()
            raw.append(loss)
    torch.cuda.synchronize()
    for i in range(8):
        mixed_opts[i].step()

for _ in range(WARMUP): one_round_mixed()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(MEASURE): one_round_mixed()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
ms_per_round = elapsed / MEASURE * 1000
print(f"[streams mixed-nap N=8] {ms_per_round:.2f} ms/round  = {ms_per_round/8:.2f} ms/model")

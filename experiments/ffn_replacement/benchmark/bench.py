"""Timing harness: clock burn-in, correctness check, and the interleaved A/B driver.

THE WARM-UP IS NOT OPTIONAL. A consumer GPU idles at a low SM clock and boosts under
load; timing that begins before the clock has ramped measures the ramp, not the
kernel. On an RTX 5090 (idle 1627 MHz) this artefact made exp_n_0126 read as 1.06x
vanilla when a warmed measurement puts it at 0.92x -- a wrong SIGN on the headline
result, not just a wrong magnitude. `burn_in()` runs by default before any timing
here and every entry point calls it.
"""
import statistics
import time

import torch

BURN_ITERS = 60


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def timeit(fn, iters=30, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def burn_in(models, batch=48, seq=512, iters=BURN_ITERS, vocab=32768):
    """Drive the GPU to its boost clock before any measurement. See module docstring."""
    if not torch.cuda.is_available():
        return
    dev = next(iter(models)).parameters().__next__().device \
        if not isinstance(models, dict) else \
        next(iter(models.values())).parameters().__next__().device
    ms = list(models.values()) if isinstance(models, dict) else list(models)
    idx = torch.randint(0, vocab, (batch, seq), device=dev)
    with torch.no_grad():
        for _ in range(iters):
            for m in ms:
                m(idx)
    torch.cuda.synchronize()


def check_bit_exact(reference, candidate, batches=(1, 12, 48), seq=512, vocab=32768):
    """Max |logit diff| of candidate vs reference. Returns (all_exact, {batch: diff}).

    Run this BEFORE reporting any timing: an optimization that changes the output is
    not a speedup, and the Triton gather is expected to be exactly 0.000.
    """
    dev = next(reference.parameters()).device
    out, ok = {}, True
    for B in batches:
        idx = torch.randint(0, vocab, (B, seq), device=dev)
        with torch.no_grad():
            d = (reference(idx) - candidate(idx)).abs().max().item()
        out[B] = d
        ok &= (d == 0.0)
    return ok, out


def slot_breakdown(named_models, batch=48, seq=512, n_embd=384, iters=30, reps=5,
                   dtypes=None):
    """Time block 0's FFN slot alone for each model. {name: ms}."""
    dev = next(iter(named_models.values())).parameters().__next__().device
    out = {}
    for name, m in named_models.items():
        dt = (dtypes or {}).get(name, next(m.parameters()).dtype)
        x = torch.randn(batch, seq, n_embd, device=dev, dtype=dt)
        blk = m.blocks[0]
        with torch.no_grad():
            out[name] = statistics.median(
                [timeit(lambda: blk.ffn_slot(x), iters=iters) for _ in range(reps)])
    return out


def interleaved_ab(named_models, batches=(12, 48, 96), seq=512, rounds=11, iters=30,
                   vocab=32768, warm=True):
    """Alternate every model within each round, in ONE process.

    Interleaving removes cross-process drift; the burn-in removes clock ramp. Returns
    {batch: {name: {'median','min','max'}}}.
    """
    if warm:
        burn_in(named_models, batch=max(batches), seq=seq, vocab=vocab)
    dev = next(iter(named_models.values())).parameters().__next__().device
    res = {}
    for B in batches:
        idx = torch.randint(0, vocab, (B, seq), device=dev)
        acc = {n: [] for n in named_models}
        with torch.no_grad():
            for m in named_models.values():
                m(idx)
            for _ in range(rounds):
                for n, m in named_models.items():
                    acc[n].append(timeit(lambda _m=m: _m(idx), iters=iters, warmup=2))
        res[B] = {n: dict(median=statistics.median(v), min=min(v), max=max(v))
                  for n, v in acc.items()}
    return res


def report(res, baseline, batches=None):
    """Print the end-to-end ladder with the vs-baseline ratio and separation flag."""
    batches = batches or sorted(res)
    names = [n for n in next(iter(res.values())) if n != baseline]
    head = f'{"batch":>6} ' + ''.join(f'{n:>22}' for n in names + [baseline])
    print(head)
    for B in batches:
        row = f'{B:>6} '
        for n in names + [baseline]:
            r = res[B][n]
            row += f'{r["median"]:>9.3f} [{r["min"]:.3f}-{r["max"]:.3f}]'.rjust(22)
        print(row)
    print(f'\n{"batch":>6} ' + ''.join(f'{n + " / " + baseline:>28}' for n in names))
    for B in batches:
        row = f'{B:>6} '
        for n in names:
            a, b = res[B][n], res[B][baseline]
            sep = a['min'] > b['max'] or a['max'] < b['min']
            verdict = 'faster' if a['median'] < b['median'] else 'slower'
            row += f'{a["median"] / b["median"]:>10.2f}x {verdict:>7} ' \
                   f'{"disjoint" if sep else "overlap":>9}'
        print(row)

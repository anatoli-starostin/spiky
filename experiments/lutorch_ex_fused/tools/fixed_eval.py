"""THE fixed validation protocol for ffn_replacement — one eval set, used identically for
the training-time eval curve and for final scoring, across every run.

── The bug this fixes ───────────────────────────────────────────────────────────────────
The original per-run trainers built the val loader at the *training* `device_batch_size`
and scored `eval_steps` batches:

    val_loader_factory = lambda: ...loader(tokenizer, DEVICE_BS, SEQ_LEN, split='val', ...)
    bpb = evaluate_bpb(model, val_loader_factory(), EVAL_STEPS, token_bytes)

Because the val loader walks the shard deterministically from token 0, the number of val
tokens scored was  DEVICE_BS x SEQ_LEN x EVAL_STEPS  — i.e. a function of the *training*
batch size. LUT runs (device_batch_size 12) were scored on the first ~61,440 tokens while
vanilla baselines (device_batch_size 48) were scored on the first ~245,760 — different,
nested validation slices. That coupling is the protocol confound; every conclusion drawn
by comparing those numbers was comparing models on different data.

── The fix ───────────────────────────────────────────────────────────────────────────────
Evaluation is DECOUPLED from training and fixed for all runs:
  * ALWAYS batch size 48 x 100 eval steps  -> the same 4,800-row span of the val stream,
    regardless of the training device_batch_size.
  * SKIP the leading `EVAL_SKIP_ROWS` (=12) rows: they are anomalously easy/contaminated
    (rows 0 and 8 are the two easiest 512-token spans in the shard), so the first 12-row
    block is dropped from the metric.
The scored set is therefore rows [12, 4800) of the deterministic from-token-0 val stream —
4,788 rows — identical for every model.

Correctness: with skip_rows=0 this reproduces nanochat.loss_eval.evaluate_bpb bit-for-bit
over the bs48 x 100 window (identical masking of special / zero-byte / ignore_index=-1
target tokens); the ONLY change is the leading-row skip.
"""
import math
import os
import sys

import torch

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

# The fixed protocol constants — batch-size-independent BY CONSTRUCTION. These are the
# defaults; a run may override via config keys eval_batch_size / eval_steps / eval_skip_rows,
# but eval must never be tied to device_batch_size again.
EVAL_BATCH_SIZE = 48     # fixed eval batch (NOT the training device_batch_size)
EVAL_STEPS      = 100    # fixed number of eval batches -> 48*100 = 4,800 rows spanned
EVAL_SKIP_ROWS  = 12     # drop the leading 12 val rows (contaminated / anomalously easy)


@torch.no_grad()
def evaluate_bpb_fixed(model, tokenizer, token_bytes, seq_len, device,
                       eval_batch_size=EVAL_BATCH_SIZE, eval_steps=EVAL_STEPS,
                       skip_rows=EVAL_SKIP_ROWS):
    """Bits-per-byte over the fixed eval window (bs `eval_batch_size` x `eval_steps`, with
    the leading `skip_rows` rows dropped), scored on the held-out val shard from token 0.

    Batch-size-independent by construction: `eval_batch_size` is a property of THIS window,
    never the training batch. Returns the same number whether called from the training loop
    or a standalone scorer, for any model.
    """
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, eval_batch_size, seq_len, split='val', device=device)
    it = iter(loader)
    dev = model.get_device()
    # DRAIN THEN SCORE. The loader yields views into a single reused GPU buffer that the
    # next yield overwrites via a non_blocking HtoD copy; interleaving forward passes with
    # further next() calls lets that async copy race the buffer reuse and silently corrupts
    # later batches (the reused-buffer bug). Materialising all batches with .clone() first
    # snapshots each one while valid, so scoring then runs on stable, private tensors.
    batches = []
    for _ in range(eval_steps):
        x, y = next(it)
        batches.append((x.clone(), y.clone()))
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    for step, (x, y) in enumerate(batches):
        B, T = x.shape
        loss2d = model(x, y, loss_reduction='none').view(B, T)          # (B, T)
        yb = y.view(B, T)
        # bytes for each target token; special tokens -> 0, ignore_index (-1) -> 0
        y_safe = torch.where(yb >= 0, yb, torch.zeros_like(yb))
        num_bytes = torch.where(yb >= 0, token_bytes[y_safe],
                                torch.zeros_like(yb, dtype=token_bytes.dtype))
        counts = num_bytes > 0                                          # tokens that count
        # Drop the leading `skip_rows` rows of the whole stream (they land in the first
        # step(s), since row index = step*eval_batch_size + local_row).
        row0 = step * eval_batch_size
        if row0 < skip_rows:
            drop = min(eval_batch_size, skip_rows - row0)
            counts[:drop, :] = False
        total_nats += (loss2d * counts).sum()
        total_bytes += torch.where(counts, num_bytes, torch.zeros_like(num_bytes)).sum()
    tb = total_bytes.item()
    if tb == 0:
        return float('inf')
    return (total_nats.item()) / (math.log(2) * tb)


def eval_config(cfg):
    """Resolve the fixed-eval knobs, defaulting to the fixed protocol (bs48 x100 skip12).

    Reads ONLY an optional `fixed_eval` sub-dict — never the legacy top-level `eval_steps`
    (that key meant the OLD, batch-coupled eval and is deliberately ignored here) and never
    `device_batch_size`. This is what makes the window batch-size-independent: no legacy
    config key can silently change it.

        "fixed_eval": {"eval_batch_size": 48, "eval_steps": 100, "skip_rows": 12}   # optional
    """
    fe = cfg.get('fixed_eval', {}) or {}
    return dict(
        eval_batch_size=int(fe.get('eval_batch_size', EVAL_BATCH_SIZE)),
        eval_steps=int(fe.get('eval_steps', EVAL_STEPS)),
        skip_rows=int(fe.get('skip_rows', EVAL_SKIP_ROWS)),
    )

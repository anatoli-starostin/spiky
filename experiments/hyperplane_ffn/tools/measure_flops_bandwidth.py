"""Reusable FLOPs + memory-bandwidth (bytes/token) profiler for the CompressionMHL / LUT models.

Parameterized by an experiment directory (config.json [+ checkpoint.pt]). Reports, for
BOTH regimes {decode = batch1/1-token, prefill = full seq amortized} and BOTH scopes
{whole-model, FFN/LUT-block-only}:
  * forward FLOPs/token  — ground truth from torch FlopCounterMode on a real forward,
    reconciled against an exact analytic matmul estimate; LUT routing arithmetic that the
    counter misses (anchor diffs, sign tests, bit-pack, embedding_bag sum) hand-counted and
    shown to be negligible.
  * bytes/token          — analytic, bf16 (2 B), split into dense-weight reads / LUT
    selected-rows reads / activations+KV. The LUT reads ONLY the selected rows
    (H*tph*inner_out), not the whole table — the key sparsity.
  * arithmetic intensity = FLOPs/token / bytes/token.

Usage:
  sbox .venv/bin/python experiments/hyperplane_ffn/tools/measure_flops_bandwidth.py \
       --exp experiments/hyperplane_ffn/exp_n_0079_hardclone_g0010_H16_d24_tph32_nap6_16k

Assumptions are printed inline so they can be audited/adjusted. Model classes are the stock
MinimalGPT + CompressionMultiHeadLUT (no shared-src edits). Weight VALUES don't affect
FLOP/byte counts (shape-derived), but we load the checkpoint when present to honor
"measure the actual model".
"""
import argparse, json, os, sys, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut

# ----------------------------- model (stock MinimalGPT) -----------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

def _rot_half(x):
    x1, x2 = x.chunk(2, dim=-1); return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]; sin = sin[None, None, :, :]
    return q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin

class MinimalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MinimalBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        n_embd, n_head = cfg['n_embd'], cfg['n_head']
        self.ln1 = nn.LayerNorm(n_embd); self.attn = MinimalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn_type = cfg.get('ffn_type', 'compression')
        self.lin = None
        if self.ffn_type == 'dense':
            hidden = cfg.get('mlp_hidden', 4 * n_embd)
            self.mlp = nn.Sequential(nn.Linear(n_embd, hidden, bias=False), nn.GELU(),
                                     nn.Linear(hidden, n_embd, bias=False))
        else:
            self.ffn = CompressionMultiHeadLUT(
                input_dim=n_embd, output_dim=n_embd,
                inner_in_dim=cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim')),
                inner_out_dim=cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim')),
                nap=cfg['lut_n_anchor_pairs'], tph=cfg['lut_tables_per_head'],
                n_heads=cfg.get('lut_n_heads', 1),
                joint_head_compression=cfg.get('lut_joint_head_compression', False),
                batched_multi_head_input=bool(cfg.get('lut_batched_multi_head_input', False)),
                forward_mode=cfg.get('lut_forward_mode', 'hard'),
                use_bf16=cfg.get('lut_use_bf16', False),
                initial_weights_noise=cfg.get('lut_init_weights_noise', 1e-3),
                learnable_temps=bool(cfg.get('lut_learnable_temps', False)),
                random_seed=cfg.get('lut_base_seed', 1000) + layer_idx)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        h = self.ln2(x)
        if self.ffn_type == 'dense':
            return x + self.mlp(h)
        B, T, C = h.shape
        out = self.ffn(h.reshape(B * T, C)).reshape(B, T, C).to(h.dtype)
        return x + out

class MinimalGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v, n_embd, n_head, n_layer = cfg['tokenizer_vocab_size'], cfg['n_embd'], cfg['n_head'], cfg['depth']
        self.tok_emb = nn.Embedding(v, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head, cfg['seq_len'])
        self.blocks = nn.ModuleList([MinimalBlock(cfg, i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, v, bias=False)
        if cfg.get('tie_unembedder', False):
            self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.rope.cos, self.rope.sin)
        return self.head(self.ln_f(x))

# ----------------------------- helpers -----------------------------

def fnum(x):  # human FLOPs/bytes
    for u, d in (('T', 1e12), ('G', 1e9), ('M', 1e6), ('K', 1e3)):
        if abs(x) >= d:
            return f"{x/d:.3f}{u}"
    return f"{x:.1f}"

def measure_flops(model, B, T, device):
    from torch.utils.flop_counter import FlopCounterMode
    idx = torch.zeros(B, T, dtype=torch.long, device=device)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(idx)
    total = fc.get_total_flops()
    # per-module: sum the per-block ffn subtree counts (disjoint), if the API exposes them
    ffn_total = None
    try:
        counts = fc.get_flop_counts()  # {mod_fqn: {op: flops}}
        ffn_total = 0
        for k, ops in counts.items():
            if re.search(r'blocks\.\d+\.ffn$', k):
                ffn_total += sum(ops.values())
        if ffn_total == 0:
            ffn_total = None
    except Exception:
        ffn_total = None
    return total, ffn_total

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiment dir with config.json (+ checkpoint.pt)')
    ap.add_argument('--seq', type=int, default=None, help='override seq_len / context length')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    exp = os.path.abspath(a.exp)
    cfg = json.load(open(os.path.join(exp, 'config.json')))
    if a.seq: cfg['seq_len'] = a.seq
    dev = a.device
    BYTES = 2  # bf16

    # --- dims ---
    n_embd = cfg['n_embd']; n_head = cfg['n_head']; head_dim = n_embd // n_head
    n_layer = cfg['depth']; vocab = cfg['tokenizer_vocab_size']; T = cfg['seq_len']
    ffn_type = cfg.get('ffn_type', 'compression')
    if ffn_type == 'dense':
        hidden = cfg.get('mlp_hidden', 4 * n_embd)
        H = nap = tph = inner_in = inner_out = cells = 0
        ffn_matmul_per_layer = 2 * n_embd * hidden + 2 * hidden * n_embd     # fc1 + fc2
        ffn_dense_params_per_layer = n_embd * hidden + hidden * n_embd       # weights, no bias
        lut_full_params = lut_selected_params = 0
        routing_ops_tok = 0
    else:
        hidden = None
        H = cfg.get('lut_n_heads', 1); nap = cfg['lut_n_anchor_pairs']; tph = cfg['lut_tables_per_head']
        inner_in = cfg.get('lut_inner_in_dim', cfg.get('lut_inner_dim'))
        inner_out = cfg.get('lut_inner_out_dim', cfg.get('lut_inner_dim'))
        cells = 2 ** nap
        ffn_matmul_per_layer = 2 * n_embd * (H * inner_in) + 2 * (H * inner_out) * n_embd  # compress+decompress
        ffn_dense_params_per_layer = (n_embd * (H * inner_in) + (H * inner_in)             # compress w+b
                                      + (H * inner_out) * n_embd + n_embd)                 # decompress w+b
        lut_full_params = H * tph * cells * inner_out
        lut_selected_params = H * tph * inner_out
        routing_ops_tok = (nap * tph * 4 + tph * inner_out) * H * n_layer

    model = MinimalGPT(cfg).to(dev).eval()
    ckpt = os.path.join(exp, 'checkpoint.pt')
    loaded = False
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=dev)
        miss, unexp = model.load_state_dict(sd, strict=False)
        miss = [m for m in miss if 'rope.' not in m]  # rope buffers are non-persistent
        loaded = True
    total_params = sum(p.numel() for p in model.parameters())

    # ============ FLOPs (ground truth) ============
    flops_prefill_total, flops_prefill_ffn = measure_flops(model, 1, T, dev)
    flops_decode1_total, flops_decode1_ffn = measure_flops(model, 1, 1, dev)  # projections; SDPA~0 at T=1
    fpref_tok = flops_prefill_total / T          # prefill amortized/token (incl full causal attn)

    # ---- analytic matmul FLOPs (2*m*k*n), exact, for reconciliation ----
    def lin_flops(out_f, in_f):  # per token
        return 2 * in_f * out_f
    proj_per_layer = (lin_flops(3 * n_embd, n_embd)      # qkv
                      + lin_flops(n_embd, n_embd)         # attn out proj
                      + ffn_matmul_per_layer)             # FFN (dense MLP or LUT compress+decompress)
    proj_all = proj_per_layer * n_layer
    unembed = lin_flops(vocab, n_embd)
    # attention analytic. NOTE: torch FlopCounterMode counts SDPA at the FULL T^2 cost and
    # does NOT discount the causal mask, so to reconcile with the counter we use the full
    # (undiscounted) attention term; the physically-causal cost is ~half of it.
    attn_prefill_full_tok = n_layer * 4 * n_embd * T           # counter convention (full T^2)
    attn_prefill_causal_tok = n_layer * 2 * n_embd * (T + 1)   # physical causal (~half)
    attn_decode_tok  = n_layer * 4 * n_embd * T                # 1 query vs T-key cache
    analytic_prefill_tok = proj_all + unembed + attn_prefill_full_tok
    analytic_prefill_causal_tok = proj_all + unembed + attn_prefill_causal_tok
    analytic_decode_tok  = proj_all + unembed + attn_decode_tok
    # decode FLOPs (ground truth projections from FlopCounter(1,1) + analytic decode attn)
    fdec_tok = flops_decode1_total + attn_decode_tok
    # block-only FFN matmul FLOPs/token (dense: fc1+fc2 ; LUT: compress+decompress)
    ffn_flops_tok = ffn_matmul_per_layer * n_layer   # same in decode & prefill (per-token proj)
    # (routing_ops_tok already computed above: LUT hand-count, or 0 for dense)

    # ============ Bandwidth (analytic bytes/token, bf16) ============
    # dense weight params (read in full each forward): all Linear weights+biases + LayerNorms
    dense_attn_per_layer = (3 * n_embd) * n_embd + n_embd * n_embd            # qkv + proj (no bias)
    dense_ln_per_layer   = 2 * (2 * n_embd)                                    # ln1+ln2 (w+b)
    dense_ffn_per_layer  = ffn_dense_params_per_layer                          # dense MLP or LUT compress+decompress
    dense_per_layer = dense_attn_per_layer + dense_ln_per_layer + dense_ffn_per_layer
    dense_all_params = dense_per_layer * n_layer + 2 * n_embd                 # + final ln_f
    unembed_params = vocab * n_embd                                           # tied head, read full for logits
    # lut_full_params / lut_selected_params already set above (0 for dense)

    dense_all_B = dense_all_params * BYTES
    unembed_B = unembed_params * BYTES
    lut_full_B_layer = lut_full_params * BYTES
    lut_sel_B_layer = lut_selected_params * BYTES
    ffn_dense_per_layer_B = dense_ffn_per_layer * BYTES

    # DECODE (batch1, 1 token): weights re-read every token; LUT selected rows only; KV cache read
    kv_read_B = 2 * n_layer * T * n_embd * BYTES              # read K,V for T context positions
    resid_rw_B = 4 * n_embd * BYTES                           # small residual-stream r/w (approx)
    # whole model
    dec_wm_dense = dense_all_B + unembed_B + n_embd * BYTES   # + tok_emb 1 row
    dec_wm_lut   = lut_sel_B_layer * n_layer
    dec_wm_act   = kv_read_B + resid_rw_B
    dec_wm = dec_wm_dense + dec_wm_lut + dec_wm_act
    # block only
    dec_bo_dense = ffn_dense_per_layer_B * n_layer
    dec_bo_lut   = lut_sel_B_layer * n_layer
    dec_bo_act   = 2 * n_embd * BYTES * n_layer               # compress in + decompress out (approx)
    dec_bo = dec_bo_dense + dec_bo_lut + dec_bo_act

    # PREFILL (full seq, amortized/token): weights read once & reused across T tokens
    act_prefill_tok = (2 * n_embd + n_embd) * BYTES + (2 * n_head * T / T) * BYTES  # KV write + resid (approx)
    # whole model
    pre_wm_dense = (dense_all_B + unembed_B) / T + n_embd * BYTES     # + tok_emb row/token
    pre_wm_lut   = (lut_full_B_layer * n_layer) / T                   # tables streamed once
    pre_wm_act   = act_prefill_tok
    pre_wm = pre_wm_dense + pre_wm_lut + pre_wm_act
    # block only
    pre_bo_dense = (ffn_dense_per_layer_B * n_layer) / T
    pre_bo_lut   = (lut_full_B_layer * n_layer) / T
    pre_bo_act   = 2 * n_embd * BYTES                                 # approx
    pre_bo = pre_bo_dense + pre_bo_lut + pre_bo_act

    # FLOPs/token per cell of the table
    F_wm_dec = fdec_tok
    F_wm_pre = fpref_tok
    F_bo_dec = ffn_flops_tok + routing_ops_tok
    F_bo_pre = ffn_flops_tok + routing_ops_tok

    def ai(f, b): return f / b if b else float('nan')

    # ============ OUTPUT ============
    L = []
    p = L.append
    p("="*78)
    p(f"FLOPs + memory-bandwidth profile — {os.path.basename(exp)}")
    p("="*78)
    p(f"config: depth={n_layer} n_embd={n_embd} n_head={n_head} (head_dim={head_dim}) seq_len={T} "
      f"vocab={vocab} tie_unembed={cfg.get('tie_unembedder')}")
    if ffn_type == 'dense':
        p(f"FFN: DENSE MLP  n_embd->{hidden}->n_embd (GELU, no bias) — NOT a LUT")
    else:
        p(f"FFN: LUT  n_heads(H)={H} inner_in={inner_in} inner_out={inner_out} tables_per_head(tph)={tph} "
          f"nap={nap} -> cells/table=2^{nap}={cells}  forward_mode={cfg.get('lut_forward_mode')} "
          f"path={'batched' if cfg.get('lut_batched_multi_head_input') else 'per-head-loop'}")
    p(f"total params: {total_params:,}   checkpoint_loaded={loaded}   dtype for bytes: bf16 (2 B)")
    p("")
    p("--- FLOPs ground-truth (FlopCounterMode) vs analytic matmul estimate ---")
    p(f"prefill FlopCounter total (1x{T}): {fnum(flops_prefill_total)} FLOPs  "
      f"-> {fnum(fpref_tok)}/token")
    p(f"prefill analytic matmul/token     : {fnum(analytic_prefill_tok)}  "
      f"(proj {fnum(proj_all)} + unembed {fnum(unembed)} + attn_full {fnum(attn_prefill_full_tok)})")
    gap_pre = (fpref_tok - analytic_prefill_tok) / fpref_tok * 100
    p(f"  reconcile: counter vs analytic gap = {gap_pre:+.2f}%  (matches => counter trustworthy)")
    p(f"  note: FlopCounter does NOT discount the causal mask; physically-causal attn is ~half")
    p(f"        ({fnum(attn_prefill_causal_tok)} attn => {fnum(analytic_prefill_causal_tok)}/token real compute)")
    p(f"decode FlopCounter proj (1x1)     : {fnum(flops_decode1_total)}/token "
      f"(SDPA~0 at T=1); + analytic decode-attn {fnum(attn_decode_tok)} => {fnum(fdec_tok)}/token")
    p(f"decode analytic matmul/token      : {fnum(analytic_decode_tok)}")
    p("")
    if ffn_type != 'dense':
        p("--- LUT routing arithmetic missed by the counter (hand-counted, per token, whole model) ---")
        p(f"anchor diffs+sign+bitpack: {nap}*{tph}*4*{H}*{n_layer} + embag sum {tph}*{inner_out}*{H}*{n_layer}")
        p(f"  = {routing_ops_tok:,} ops/token  vs prefill matmul {fnum(fpref_tok)} "
          f"=> {routing_ops_tok/fpref_tok*100:.3f}% (negligible; counter total trustworthy)")
    else:
        p("--- dense MLP FFN: two full matmuls per layer (n_embd<->hidden), no gather/routing ---")
    p("")
    p("="*78)
    p("RESULTS TABLE  (per token)")
    p("="*78)
    hdr = f"{'scope / regime':<24}{'FLOPs/tok':>12}{'bytes/tok':>12}{'  dense':>12}{'  LUT':>11}{'  act/KV':>11}{'AI(F/B)':>10}"
    p(hdr); p("-"*len(hdr))
    rows = [
        ("whole-model  decode", F_wm_dec, dec_wm, dec_wm_dense, dec_wm_lut, dec_wm_act),
        ("whole-model  prefill", F_wm_pre, pre_wm, pre_wm_dense, pre_wm_lut, pre_wm_act),
        ("block-only   decode", F_bo_dec, dec_bo, dec_bo_dense, dec_bo_lut, dec_bo_act),
        ("block-only   prefill", F_bo_pre, pre_bo, pre_bo_dense, pre_bo_lut, pre_bo_act),
    ]
    for name, f, b, dns, lut, act in rows:
        p(f"{name:<24}{fnum(f):>12}{fnum(b):>12}{fnum(dns):>12}{fnum(lut):>11}{fnum(act):>11}{ai(f,b):>10.2f}")
    p("-"*len(hdr))
    p("")
    p("--- key byte facts ---")
    if ffn_type != 'dense':
        p(f"LUT selected-rows/token/layer = H*tph*inner_out*2 = {H}*{tph}*{inner_out}*2 = {lut_sel_B_layer:,} B")
        p(f"LUT full table/layer          = H*tph*cells*inner_out*2 = {lut_full_B_layer:,} B "
          f"(selected = 1/{cells} = {lut_sel_B_layer/lut_full_B_layer*100:.2f}% of full)")
    else:
        p(f"dense FFN weights/layer       = 2*n_embd*hidden*2 = 2*{n_embd}*{hidden}*2 = {ffn_dense_per_layer_B:,} B "
          f"(both matmuls read in full every forward — no LUT sparsity)")
    p(f"dense weights (all layers)    = {dense_all_B:,} B ; tied unembed = {unembed_B:,} B")
    p(f"KV-cache read/token (decode)  = 2*{n_layer}*{T}*{n_embd}*2 = {kv_read_B:,} B")
    p("")
    p("--- reading ---")
    p("decode is memory-bound: low AI (weights + unembed re-read every token dominate; LUT rows tiny).")
    p("prefill is compute-bound: high AI (weights amortized over the sequence).")
    if ffn_type != 'dense':
        p("The LUT selected-rows sparsity makes LUT byte traffic ~2 orders below the dense weight reads.")
    else:
        p("Dense MLP: both FFN matmuls are full dense weight reads (no gather sparsity).")
    p("Assumptions: decode re-reads every weight per token (batch1, no reuse); prefill reads each")
    p("weight once and reuses across T tokens; act/KV terms marked approx (dense/LUT terms are exact).")
    out = "\n".join(L)
    print(out)
    with open(os.path.join(exp, 'flops_bandwidth.txt'), 'w') as f:
        f.write(out + "\n")
    print(f"\n[saved] {os.path.join(exp, 'flops_bandwidth.txt')}")

if __name__ == '__main__':
    main()

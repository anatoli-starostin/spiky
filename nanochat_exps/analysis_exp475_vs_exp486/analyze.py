"""Detailed comparison of exp475 (bs=16, 1.4962) vs exp486 (bs=48, ~1.39).

Goal: understand WHAT improves with 3x more tokens, to find a way to reach it
with fewer tokens. Both share architecture + seed, so per-row weight comparison
is meaningful.

Analyses (all on a FIXED shared validation batch so inputs are identical):
  1. Loss/bpb reproduction sanity.
  2. LUT selection coverage / visit statistics (dead rows, visit entropy, Gini).
  3. Per-table weight statistics (row norms, effective rank, row redundancy).
  4. Weight movement from shared init.
  5. Final learned temperatures (T_soft, T_sel).
  6. Selection confidence (softmax winner-coeff distribution).
  7. Attention entropy per layer/head.

Writes results.json + plots to this directory.
Run:  /home/starost/spiky/.venv/bin/python analyze.py
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

from model_def import load_checkpoint, build_model, LUT_NAMES, compute_lut_indices

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

DEVICE = 'cuda'
CKPT_475 = '/home/starost/spiky/nanochat_exps/exp475_meanabs_nocenter/checkpoint.pt'
CKPT_486 = '/home/starost/spiky/nanochat_exps/exp486_bs48_8k/checkpoint.pt'
N_BATCHES = 8          # fixed val batches collected once, reused for both models
EVAL_BPB_STEPS = 40    # bpb estimate steps


# ----------------------------------------------------------------------------
def get_val_batches(cfg, n_batches, device):
    """Collect a fixed list of (x, y) val batches once; reuse for both models."""
    base = get_base_dir()
    tok = RustBPETokenizer.from_directory(os.path.join(base, 'tokenizer'))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, cfg['device_batch_size'], cfg['context_size'], split='val', device=device)
    batches = [next(loader) for _ in range(n_batches)]
    token_bytes = get_token_bytes(device=device)
    return batches, token_bytes


@torch.no_grad()
def eval_bpb_fixed(model, batches, token_bytes):
    """bpb over the fixed batch list (mirrors evaluate_bpb's bits/byte formula)."""
    total_nats, total_bytes = 0.0, 0
    for x, y in batches:
        logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1),
                         ignore_index=-1, reduction='sum')
        valid = (y.view(-1) != -1)
        nbytes = token_bytes[y.view(-1)[valid]].sum().item()
        total_nats += nll.item()
        total_bytes += nbytes
    return total_nats / total_bytes / math.log(2)


# ----------------------------------------------------------------------------
@torch.no_grad()
def capture_lut_inputs(model, batches):
    """Run forward with pre-hooks capturing each LUT module's input tensor.
    Returns {layer_idx: {lut_name: concatenated_input [N, in_dim]}}."""
    store = {li: {n: [] for n in LUT_NAMES} for li in range(len(model.layers))}
    handles = []
    for li, blk in enumerate(model.layers):
        for n in LUT_NAMES:
            mod = getattr(blk, n)
            def mk(li=li, n=n):
                def hook(module, inp):
                    store[li][n].append(inp[0].detach())
                return hook
            handles.append(mod.register_forward_pre_hook(mk()))
    for x, _ in batches:
        model(x)
    for h in handles:
        h.remove()
    out = {}
    for li in store:
        out[li] = {n: torch.cat(store[li][n], dim=0) for n in LUT_NAMES}
    return out


@torch.no_grad()
def _chunked_select(mod, x, chunk=2048):
    """Memory-safe per-module selection over tokens. Returns:
      binc [n_tables, K] float visit counts, winner_coeff_sum (scalar), n_tokens.
    Avoids materializing the full [N, n_tables, K] score tensor (17 GB for
    out_proj at 65k tokens)."""
    T_soft = mod.log_soft_score_temp.exp().float()
    T_sel = mod.log_select_temp.exp().float()
    aa, bb, bm = mod.soft_anchor_a_long, mod.soft_anchor_b_long, mod.soft_bit_matrix.float()
    n_tables = aa.shape[0]
    K = mod.weights.shape[1]
    tbl_off = torch.arange(n_tables, device=x.device).unsqueeze(0) * K
    binc = torch.zeros(n_tables * K, device=x.device)
    coeff_sum = torch.zeros((), device=x.device)
    N = x.shape[0]
    for s in range(0, N, chunk):
        xc = x[s:s + chunk]
        d = xc[:, aa] - xc[:, bb]
        p = d / (T_soft + d.abs())
        ts = torch.einsum('btp,pk->btk', p, bm)
        idx = ts.argmax(dim=-1)                                  # [c, n_tables]
        flat = (idx + tbl_off).reshape(-1)
        binc += torch.bincount(flat, minlength=n_tables * K).float()
        sel = F.softmax(ts / T_sel, dim=-1)
        coeff_sum += sel.max(dim=-1).values.sum()
    return binc.reshape(n_tables, K), coeff_sum.item(), N * n_tables


@torch.no_grad()
def selection_stats(model, lut_inputs):
    """Per-module: row coverage, dead-row count, visit entropy, Gini, confidence."""
    res = {}
    for n in LUT_NAMES:
        cov_list, dead_list, ent_list, gini_list, conf_list = [], [], [], [], []
        for li in range(len(model.layers)):
            mod = getattr(model.layers[li], n)
            binc, coeff_sum, n_sel = _chunked_select(mod, lut_inputs[li][n])
            K = mod.weights.shape[1]
            visited = (binc > 0).float().sum(dim=1)     # [n_tables]
            cov_list.append((visited / K).mean().item())
            dead_list.append(int(((binc == 0).sum()).item()))
            prob = binc / binc.sum(dim=1, keepdim=True).clamp(min=1)
            ent = -(prob * (prob.clamp(min=1e-12)).log()).sum(dim=1) / math.log(K)
            ent_list.append(ent.mean().item())
            gini_list.append(_gini_rows(binc).mean().item())
            conf_list.append(coeff_sum / n_sel)
        res[n] = dict(
            coverage=cov_list, dead_rows=dead_list, visit_entropy=ent_list,
            gini=gini_list, winner_coeff=conf_list,
            total_rows=[int(getattr(model.layers[li], n).weights.shape[0] *
                            getattr(model.layers[li], n).weights.shape[1])
                        for li in range(len(model.layers))],
        )
    return res


def _gini_rows(binc):
    """Gini coefficient per row (table) of a [n_tables, K] count matrix."""
    n_tables, K = binc.shape
    srt, _ = binc.sort(dim=1)
    cum = torch.arange(1, K + 1, device=binc.device).float().unsqueeze(0)
    total = srt.sum(dim=1, keepdim=True).clamp(min=1e-9)
    gini = (2 * (cum * srt).sum(dim=1) / (K * total.squeeze(1))) - (K + 1) / K
    return gini


@torch.no_grad()
def _winner_coeff(mod, x):
    """Mean over tokens/tables of the winner's softmax(ts/T_sel) coefficient."""
    T_soft = mod.log_soft_score_temp.exp().float()
    T_sel = mod.log_select_temp.exp().float()
    aa, bb, bm = mod.soft_anchor_a_long, mod.soft_anchor_b_long, mod.soft_bit_matrix.float()
    d = x[:, aa] - x[:, bb]
    p = d / (T_soft + d.abs())
    ts = torch.einsum('btp,pk->btk', p, bm)
    sel = F.softmax(ts / T_sel, dim=-1)
    return sel.max(dim=-1).values.mean()


# ----------------------------------------------------------------------------
@torch.no_grad()
def weight_stats(model):
    """Per-module per-table: row-norm distribution, effective rank, redundancy."""
    res = {}
    for n in LUT_NAMES:
        erank_list, redund_list, rownorm_mean, rownorm_cv, near_zero = [], [], [], [], []
        for li in range(len(model.layers)):
            W = getattr(model.layers[li], n).weights.float()   # [n_tables, K, n_out]
            n_tables, K, n_out = W.shape
            # row norms
            rn = W.norm(dim=2)                                  # [n_tables, K]
            rownorm_mean.append(rn.mean().item())
            rownorm_cv.append((rn.std() / rn.mean().clamp(min=1e-9)).item())
            near_zero.append(int((rn < 0.01 * rn.mean()).sum().item()))
            # effective rank per table via SVD (participation ratio of singular vals)
            erank_list.append(_eff_rank(W).mean().item())
            # row redundancy: mean abs pairwise cosine among rows, per table
            redund_list.append(_row_redundancy(W).mean().item())
        res[n] = dict(eff_rank=erank_list, redundancy=redund_list,
                      rownorm_mean=rownorm_mean, rownorm_cv=rownorm_cv,
                      near_zero_rows=near_zero,
                      K=[int(getattr(model.layers[li], n).weights.shape[1])
                         for li in range(len(model.layers))])
    return res


@torch.no_grad()
def _eff_rank(W):
    """Participation-ratio effective rank per table: (sum s)^2 / sum s^2."""
    # W: [n_tables, K, n_out]; center rows per table first (DC absorbed downstream)
    Wc = W - W.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(Wc)                    # [n_tables, min(K,n_out)]
    pr = (s.sum(dim=1) ** 2) / (s.pow(2).sum(dim=1).clamp(min=1e-12))
    return pr


@torch.no_grad()
def _row_redundancy(W):
    """Mean abs off-diagonal cosine similarity among rows, per table."""
    Wc = W - W.mean(dim=1, keepdim=True)
    nrm = Wc.norm(dim=2, keepdim=True).clamp(min=1e-9)
    U = Wc / nrm
    G = torch.bmm(U, U.transpose(1, 2)).abs()        # [n_tables, K, K]
    K = G.shape[1]
    off = (G.sum(dim=(1, 2)) - K) / (K * (K - 1))    # exclude diagonal (=1)
    return off


@torch.no_grad()
def weight_movement(model, cfg):
    """Relative L2 movement of each LUT weight tensor from its seeded init."""
    init = build_model(cfg, device=DEVICE)
    res = {}
    for n in LUT_NAMES:
        mv = []
        for li in range(len(model.layers)):
            Wf = getattr(model.layers[li], n).weights.float()
            Wi = getattr(init.layers[li], n).weights.float()
            mv.append(((Wf - Wi).norm() / Wf.norm().clamp(min=1e-9)).item())
        res[n] = mv
    del init
    return res


@torch.no_grad()
def temperatures(model):
    res = {}
    for n in LUT_NAMES:
        res[n] = dict(
            T_soft=[getattr(model.layers[li], n).log_soft_score_temp.exp().item()
                    for li in range(len(model.layers))],
            T_sel=[getattr(model.layers[li], n).log_select_temp.exp().item()
                   for li in range(len(model.layers))],
        )
    return res


# ----------------------------------------------------------------------------
@torch.no_grad()
def cross_model_compare(model_a, model_b, lut_inputs_a):
    """exp475 vs exp486 share seed+anchors -> row (t,r) is the SAME bit-pattern
    in both. Directly compare per-row weights and tie selection coverage to it.

    Key question: are rows that are DEAD (never selected) in exp475 ALIVE in
    exp486? If batch scaling's win is row-coverage, exp486 should activate rows
    that bs=16 starved. Uses exp475's captured inputs to define exp475 visits;
    exp486 visits computed on the same inputs run through exp486's selection.
    """
    res = {}
    for n in LUT_NAMES:
        rel_w_change, rownorm_corr = [], []
        revived_rows, newly_dead = [], []
        for li in range(len(model_a.layers)):
            Wa = getattr(model_a.layers[li], n).weights.float()
            Wb = getattr(model_b.layers[li], n).weights.float()
            rel_w_change.append(((Wb - Wa).norm() / Wa.norm().clamp(min=1e-9)).item())
            rna = Wa.norm(dim=2).reshape(-1)
            rnb = Wb.norm(dim=2).reshape(-1)
            rownorm_corr.append(float(torch.corrcoef(torch.stack([rna, rnb]))[0, 1]))
            # selection coverage on identical inputs (exp475's captured input)
            x = lut_inputs_a[li][n]
            binc_a, _, _ = _chunked_select(getattr(model_a.layers[li], n), x)
            binc_b, _, _ = _chunked_select(getattr(model_b.layers[li], n), x)
            va = (binc_a.reshape(-1) > 0)
            vb = (binc_b.reshape(-1) > 0)
            revived_rows.append(int((vb & ~va).sum().item()))   # dead in 475, alive in 486
            newly_dead.append(int((va & ~vb).sum().item()))     # alive in 475, dead in 486
        res[n] = dict(rel_weight_change=rel_w_change, rownorm_corr=rownorm_corr,
                      revived_rows=revived_rows, newly_dead=newly_dead)
    return res


_SDPA_REAL = F.scaled_dot_product_attention
_ATTN_ENT = []


def _sdpa_capture(q, k, v, *args, **kwargs):
    """Wrapper: record per-head attention entropy (causal) then call real SDPA."""
    with torch.no_grad():
        B, H, T, d = q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)                 # [B,H,T,T]
        ent = -(attn * attn.clamp(min=1e-12).log()).sum(-1)   # [B,H,T]
        # normalize by log of valid context length (position+1)
        pos = torch.arange(1, T + 1, device=q.device).float().clamp(min=1)
        norm_ent = (ent / pos.log().clamp(min=1e-9)).mean(dim=(0, 2))   # [H]
        _ATTN_ENT.append(norm_ent.cpu())
    return _SDPA_REAL(q, k, v, *args, **kwargs)


@torch.no_grad()
def attention_entropy(model, batches):
    global _ATTN_ENT
    _ATTN_ENT = []
    F.scaled_dot_product_attention = _sdpa_capture
    try:
        for x, _ in batches:
            model(x)
    finally:
        F.scaled_dot_product_attention = _SDPA_REAL
    # _ATTN_ENT collected as n_batches * n_layers entries, each [H]
    n_layers = len(model.layers)
    per_layer = [[] for _ in range(n_layers)]
    for i, e in enumerate(_ATTN_ENT):
        per_layer[i % n_layers].append(e)
    return [torch.stack(p).mean(dim=0).tolist() for p in per_layer]   # [n_layers][H]


# ----------------------------------------------------------------------------
def run_all(model, cfg, tag, batches, token_bytes):
    print(f'\n===== {tag} =====')
    bpb = eval_bpb_fixed(model, batches, token_bytes)
    print(f'  bpb (fixed {len(batches)} batches): {bpb:.4f}')
    lut_inputs = capture_lut_inputs(model, batches)
    out = dict(
        tag=tag, bpb=bpb,
        selection=selection_stats(model, lut_inputs),
        weights=weight_stats(model),
        movement=weight_movement(model, cfg),
        temps=temperatures(model),
        attn_entropy=attention_entropy(model, batches),
    )
    return out, lut_inputs


def main():
    # use exp475 cfg for batch shape (bs=16); same val data fed to both models
    import torch as _t
    ck = _t.load(CKPT_475, map_location='cpu', weights_only=False)
    cfg475 = dict(ck['config'])
    batches, token_bytes = get_val_batches(cfg475, N_BATCHES, DEVICE)
    print(f'Collected {len(batches)} fixed val batches '
          f'(bs={cfg475["device_batch_size"]}, T={cfg475["context_size"]})')

    model475, cfg_a = load_checkpoint(CKPT_475, device=DEVICE)
    model486, cfg_b = load_checkpoint(CKPT_486, device=DEVICE)
    r475, inp475 = run_all(model475, cfg_a, 'exp475_bs16', batches, token_bytes)
    r486, _      = run_all(model486, cfg_b, 'exp486_bs48', batches, token_bytes)

    print('\n===== cross-model (shared seed/anchors) =====')
    cross = cross_model_compare(model475, model486, inp475)

    results = dict(exp475=r475, exp486=r486, cross=cross)
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote results.json')

    make_plots(r475, r486)
    print_summary(r475, r486, cross)


def print_summary(a, b, cross=None):
    print('\n================ SUMMARY (exp475 -> exp486) ================')
    print(f'bpb: {a["bpb"]:.4f} -> {b["bpb"]:.4f}  (delta {b["bpb"]-a["bpb"]:+.4f})')
    for n in LUT_NAMES:
        print(f'\n[{n}]')
        for metric, fmt in [('coverage', '{:.3f}'), ('visit_entropy', '{:.3f}'),
                            ('winner_coeff', '{:.3f}')]:
            av = np.mean(a['selection'][n][metric])
            bv = np.mean(b['selection'][n][metric])
            print(f'  {metric:16s}: {fmt.format(av)} -> {fmt.format(bv)}')
        for metric, fmt in [('eff_rank', '{:.2f}'), ('redundancy', '{:.3f}'),
                            ('rownorm_cv', '{:.3f}')]:
            av = np.mean(a['weights'][n][metric])
            bv = np.mean(b['weights'][n][metric])
            print(f'  {metric:16s}: {fmt.format(av)} -> {fmt.format(bv)}')
        if cross is not None:
            c = cross[n]
            rev = int(np.sum(c["revived_rows"]))
            nd = int(np.sum(c["newly_dead"]))
            print(f'  {"rel_w_change":16s}: {np.mean(c["rel_weight_change"]):.3f}')
            print(f'  {"rownorm_corr":16s}: {np.mean(c["rownorm_corr"]):.3f}')
            print(f'  {"revived_rows":16s}: {rev} (dead@475 -> alive@486)  newly_dead: {nd}')


def make_plots(a, b):
    # 1. selection coverage + eff_rank per module (mean over layers)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(LUT_NAMES)); w = 0.35
    for ax, key, sub, title in [
        (axes[0, 0], 'selection', 'coverage', 'Row coverage (frac rows ever selected)'),
        (axes[0, 1], 'weights', 'eff_rank', 'Effective rank per table (participation ratio)'),
        (axes[1, 0], 'weights', 'redundancy', 'Row redundancy (mean |cosine|)'),
        (axes[1, 1], 'selection', 'winner_coeff', 'Selection confidence (winner softmax coeff)'),
    ]:
        av = [np.mean(a[key][n][sub]) for n in LUT_NAMES]
        bv = [np.mean(b[key][n][sub]) for n in LUT_NAMES]
        ax.bar(x - w/2, av, w, label='exp475 bs16')
        ax.bar(x + w/2, bv, w, label='exp486 bs48')
        ax.set_xticks(x); ax.set_xticklabels(LUT_NAMES, rotation=20)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'compare_modules.png'), dpi=120)
    plt.close(fig)

    # 2. per-layer attention entropy
    fig, ax = plt.subplots(figsize=(9, 5))
    la = np.array(a['attn_entropy']).mean(axis=1)
    lb = np.array(b['attn_entropy']).mean(axis=1)
    ax.plot(la, 'o-', label='exp475 bs16')
    ax.plot(lb, 's-', label='exp486 bs48')
    ax.set(xlabel='layer', ylabel='norm. attention entropy', title='Attention entropy per layer')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'attn_entropy.png'), dpi=120)
    plt.close(fig)
    print('wrote compare_modules.png, attn_entropy.png')


if __name__ == '__main__':
    main()

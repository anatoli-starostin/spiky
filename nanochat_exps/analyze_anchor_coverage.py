#!/usr/bin/env python3
"""Static anchor-coverage analysis for tiny LUT-LM checkpoints.

For an experiment directory (must contain config.json and checkpoint.pt),
inspects the LUT anchor buffers (soft_anchor_a_long / soft_anchor_b_long)
and reports which input-stream positions are read by which LUT in which layer.

E-stream positions (input_dim = E) are read by qk_lut, v_lut, residual_lut.
out_proj reads from the post-attention concat (input_dim = H * d_v); it is
reported separately (and its "dead positions" are about the attention output
space, not the E carry).

Usage:  python analyze_anchor_coverage.py <exp_dir>
"""
import json
import re
import sys
from collections import defaultdict

import torch


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    exp_dir = sys.argv[1].rstrip('/')

    with open(f'{exp_dir}/config.json') as f:
        cfg = json.load(f)
    E = cfg['embedding_dim']
    H = cfg['n_heads']
    d_v = cfg['d_v']
    N_LAYERS = cfg['num_layers']
    out_proj_in_dim = H * d_v

    ckpt = torch.load(f'{exp_dir}/checkpoint.pt', map_location='cpu', weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and any('soft_anchor_a_long' in k for k in ckpt) \
            else ckpt.get('model_state', ckpt.get('state_dict', ckpt))

    # Match: layers.<L>.<lut_name>.soft_anchor_{a,b}_long
    pat = re.compile(r'^layers\.(\d+)\.([a-zA-Z_]+)\.soft_anchor_([ab])_long$')

    # per_lut_counts[lut_name][layer_idx] = tensor[input_dim] of read counts
    per_lut_counts: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    for key, tensor in state.items():
        m = pat.match(key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        lut_name = m.group(2)
        in_dim = out_proj_in_dim if lut_name == 'out_proj' else E
        if lut_name not in per_lut_counts or layer_idx not in per_lut_counts[lut_name]:
            per_lut_counts[lut_name][layer_idx] = torch.zeros(in_dim, dtype=torch.long)
        per_lut_counts[lut_name][layer_idx].scatter_add_(
            0, tensor.flatten().to(torch.long), torch.ones_like(tensor.flatten(), dtype=torch.long)
        )

    e_stream_luts = [n for n in ('qk_lut', 'qkv_lut', 'v_lut', 'residual_lut')
                     if n in per_lut_counts]

    print(f'=== {exp_dir} ===')
    print(f'E={E}, H={H}, d_v={d_v}, out_proj_in_dim=H*d_v={out_proj_in_dim}, layers={N_LAYERS}')
    print()

    # ----- Per-LUT-per-layer summary -----
    def fmt_stats(c: torch.Tensor) -> str:
        return (f'total={c.sum().item():>6d}  min={c.min().item():>3d}  '
                f'max={c.max().item():>3d}  mean={c.float().mean().item():>5.1f}  '
                f'zeros={(c == 0).sum().item():>3d}/{c.numel()}')

    print('Per-LUT-per-layer anchor read counts:')
    for lut_name in sorted(per_lut_counts.keys()):
        in_dim = out_proj_in_dim if lut_name == 'out_proj' else E
        print(f'  {lut_name} (in_dim={in_dim}):')
        for L in sorted(per_lut_counts[lut_name].keys()):
            print(f'    L{L}: {fmt_stats(per_lut_counts[lut_name][L])}')
    print()

    # ----- E-stream aggregate (qk_lut + v_lut + residual_lut, all layers) -----
    total_e = torch.zeros(E, dtype=torch.long)
    for lut_name in e_stream_luts:
        for L, c in per_lut_counts[lut_name].items():
            total_e += c
    dead = (total_e == 0).nonzero().flatten().tolist()
    print(f'E-stream aggregate (qk/v/residual, all layers):')
    print(f'  {fmt_stats(total_e)}')
    print(f'  STRICTLY-DEAD E positions (never referenced by any LUT in any layer): '
          f'{len(dead)}/{E}')
    if dead:
        print(f'    positions: {dead}')
    # Show 5 most/least read positions.
    sort_idx = torch.argsort(total_e)
    print(f'  bottom-5 (least-read):  '
          + ', '.join(f'pos{int(i)}={int(total_e[i])}' for i in sort_idx[:5].tolist()))
    print(f'  top-5    (most-read):   '
          + ', '.join(f'pos{int(i)}={int(total_e[i])}' for i in sort_idx[-5:].flip(0).tolist()))
    print()

    # ----- residual_lut-only (positions that DIRECTLY contribute to D) -----
    if 'residual_lut' in per_lut_counts:
        total_res = torch.zeros(E, dtype=torch.long)
        for L, c in per_lut_counts['residual_lut'].items():
            total_res += c
        res_dead = (total_res == 0).nonzero().flatten().tolist()
        print(f'residual_lut-only (positions DIRECTLY contributing to D via any layer):')
        print(f'  {fmt_stats(total_res)}')
        print(f'  E positions never read by ANY residual_lut: {len(res_dead)}/{E}')
        if res_dead:
            print(f'    positions: {res_dead}')
        print()

    # ----- Out_proj coverage (attention-output space, separate) -----
    if 'out_proj' in per_lut_counts:
        total_op = torch.zeros(out_proj_in_dim, dtype=torch.long)
        for L, c in per_lut_counts['out_proj'].items():
            total_op += c
        op_dead = (total_op == 0).nonzero().flatten().tolist()
        print(f'out_proj input-coverage (H*d_v={out_proj_in_dim} space):')
        print(f'  {fmt_stats(total_op)}')
        print(f'  attention-output positions never referenced: {len(op_dead)}/{out_proj_in_dim}')
        if op_dead:
            print(f'    positions: {op_dead}')


if __name__ == '__main__':
    main()

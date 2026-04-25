"""Profile BitPermutationLUT + BitPermutationLUTOptimizer on exp299-style shapes.

Three top-level measurements:
  fwd              — no_grad module call
  fwd + bwd        — training mode, autograd backward
  fwd + bwd + step — full training step

Plus isolated kernel timings (each built with synthetic inputs so the
measurement doesn't depend on harness plumbing)."""
import time

import torch

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT, _get_bit_permlut_native
from spiky.lutorch.bit_permutation_lut_optimizer import (
    BitPermutationLUTOptimizer,
    _project_grad_out_to_weight_grad,
    _to_fp8_per_table,
)


DEVICE = torch.device("cuda:0")


def sync():
    torch.cuda.synchronize()


def bench(name, fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    sync()
    ms = (time.time() - t0) / n * 1000.0
    print(f"  {name:50s} {ms:8.3f} ms")
    return ms


# exp299 per-layer configs (4 flavors × 6 layers = 24 modules/step).
CONFIGS = [
    ("qk",  dict(n_inputs=32, n_outputs=24, n_heads=4, input_nap=5,  output_nap=24, tph=192)),
    ("v",   dict(n_inputs=32, n_outputs=16, n_heads=4, input_nap=5,  output_nap=16, tph=128)),
    ("out", dict(n_inputs=64, n_outputs=32, n_heads=1, input_nap=10, output_nap=32, tph=1024)),
]
BATCH = 1024   # exp299 bs=8 * ctx=128 = 1024 tokens/step per layer


def profile_one(name, cfg):
    print(f"\n=== {name}: {cfg} ===")
    lut = BitPermutationLUT(random_seed=42, device=DEVICE, **cfg)
    opt = BitPermutationLUTOptimizer([lut], lr=1e-3, seed=1)
    native = _get_bit_permlut_native()

    B = BATCH
    x = torch.randn(B, cfg["n_inputs"], device=DEVICE, requires_grad=True)
    y = torch.randn(B, lut.n_heads, lut.n_pairs, device=DEVICE)

    # --- Top-level ---
    lut.eval()
    with torch.no_grad():
        t_fwd = bench("fwd (no_grad)", lambda: lut(x))
    lut.train()

    def fwd_bwd():
        opt.zero_grad()
        ((lut(x) - y) ** 2).mean().backward()
    t_fb = bench("fwd + bwd", fwd_bwd)

    def full_step():
        opt.zero_grad()
        ((lut(x) - y) ** 2).mean().backward()
        opt.step()
    t_full = bench("fwd + bwd + opt.step", full_step)

    # --- Isolated kernels ---
    # Synthetic inputs the kernels need; no snapshotting required.
    li = torch.randint(0, lut.table_dim, (B, lut.n_heads * lut.tph), dtype=torch.int16, device=DEVICE)
    lai = torch.randint(0, lut.table_dim, (B, lut.n_heads * lut.tph, 1), dtype=torch.int16, device=DEVICE)
    grad_out = torch.randn(B, lut.n_heads, lut.n_pairs, device=DEVICE)

    def _dom_fwd():
        _ = native.bit_perm_lut_dom_gather_forward(
            li, lut.bit_weights, lut.inv_idx,
            int(lut.n_heads), int(lut.tph), int(lut.output_nap), int(lut.n_pairs),
        )
    t_dom_fwd = bench("  bit_perm_lut_dom_gather_forward", _dom_fwd)

    def _dom_bwd():
        _ = native.bit_perm_lut_dom_gather_backward(
            grad_out, li, lai, lut.bit_weights, lut.output_idx_per_table,
            int(lut.n_heads), int(lut.tph), int(lut.output_nap), int(lut.n_pairs), float(lut.scale),
        )
    t_dom_bwd = bench("  bit_perm_lut_dom_gather_backward", _dom_bwd)

    # _project_grad_out_to_weight_grad (uses custom kernel for small td, index_add for big td)
    wg_buf = opt._states[0]["wg_buffer"]
    def _proj():
        _ = _project_grad_out_to_weight_grad(
            grad_out, li, lut.output_idx_per_table,
            lut.n_heads, lut.tph, lut.output_nap, lut.table_dim, lut.scale,
            wg_buffer=wg_buf,
        )
    t_proj = bench("  _project_grad_out_to_weight_grad", _proj)

    # Fused fp8 Adam kernel
    state = opt._states[0]
    wg_buf.normal_()  # fresh data, don't leak state between runs
    def _fused_adam():
        _ = native.fused_fp8_adam(
            state["latent_fp8"], state["m_fp8"], state["m_scale"],
            state["v_fp8"], state["v_scale"], wg_buf,
            0.9, 0.999, 1e-8, 0.9, 0.999, 1e-3,
        )
    t_fused = bench("  fused_fp8_adam kernel", _fused_adam)

    # Per-table m, v fp8 requantization (remaining Python op after fused kernel)
    tmp = torch.randn(lut.n_heads * lut.tph, lut.table_dim, lut.output_nap, device=DEVICE)
    def _per_table():
        _, _ = _to_fp8_per_table(tmp)
    t_per_table = bench("  _to_fp8_per_table (m or v)", _per_table)

    # Sign-packing kernel
    signs = torch.randn(lut.n_heads * lut.tph, lut.table_dim, lut.output_nap, device=DEVICE)
    def _pack():
        native.bit_pack_signs(signs, lut.bit_weights, int(lut.output_nap))
    t_pack = bench("  bit_pack_signs kernel", _pack)

    print(
        f"  SUMMARY fwd={t_fwd:.3f} | fwd+bwd={t_fb:.3f} "
        f"(bwd_only≈{t_fb - t_fwd:.3f}) | step_only≈{t_full - t_fb:.3f} | total={t_full:.3f} ms"
    )
    opt.close()


if __name__ == "__main__":
    print(f"Device: {DEVICE}, batch={BATCH}")
    for name, cfg in CONFIGS:
        profile_one(name, cfg)

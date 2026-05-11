"""Microbenchmark to localize where the soft-bwd kernel time goes."""
import torch
import lutorch_cuda
mgr = lutorch_cuda.get_lutorch_manager()


def main():
    dev = torch.device("cuda")
    # v_lut shape
    B, T, K, NAP = 4096, 6*256, 256, 8

    # d_sel_soft [B, T, K] in bf16 = 3.2 GB
    d_sel_soft = torch.randn(B, T, K, device=dev, dtype=torch.bfloat16)
    d_buf      = torch.randn(B, T, NAP, device=dev, dtype=torch.bfloat16)
    ap_a       = torch.randint(0, 96, (T, NAP), device=dev, dtype=torch.int16)
    ap_b       = torch.randint(0, 96, (T, NAP), device=dev, dtype=torch.int16)

    # Warmup
    for _ in range(8):
        gx, gts, gtx = mgr.soft_lut_backward_grad_x(d_sel_soft, d_buf, ap_a, ap_b, 96, 0.5, 0.5)
    torch.cuda.synchronize()

    n_iter = 50
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for i in range(n_iter):
        evts[i][0].record()
        gx, gts, gtx = mgr.soft_lut_backward_grad_x(d_sel_soft, d_buf, ap_a, ap_b, 96, 0.5, 0.5)
        evts[i][1].record()
    torch.cuda.synchronize()
    t = sum(s.elapsed_time(e) for s, e in evts) / n_iter
    print(f"  full kernel:                                  {t:6.2f} ms")
    print(f"  d_sel_soft size:                              {d_sel_soft.numel()*2/1e9:.2f} GB")
    print(f"  memory bandwidth read d_sel_soft @ 3 TB/s:    {d_sel_soft.numel()*2/3e12*1e3:.2f} ms")

    # Baseline: just copying d_sel_soft to itself via a kernel
    n_iter = 50
    evts = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for i in range(n_iter):
        evts[i][0].record()
        _ = d_sel_soft.sum()
        evts[i][1].record()
    torch.cuda.synchronize()
    t = sum(s.elapsed_time(e) for s, e in evts) / n_iter
    print(f"  torch.sum(d_sel_soft) [memory-bound sweep]:   {t:6.2f} ms")


if __name__ == "__main__":
    main()

"""Before/after numerics regression for fast_multi_head_lut.py (not collected by pytest).

Loads a BASE copy of fast_multi_head_lut.py as a second module beside the current one and runs every
configuration the base already supported through both, with identical seeds, state, input and
upstream gradient: forward_mode hard / backward_topk=0, hybrid_smooth / backward_topk in {0, 1, 2},
exp_outputs; forward_confidence off / "margin" / "bounded"; single- and multi-head input;
n_outputs 4 and 128 (hybrid gather vs bmm forward); CPU float64, CUDA float64, CUDA fp32 storage with
bf16 autocast. Compares with torch.equal: train output, eval (no_grad) output, x.grad and every
parameter .grad. A base-vs-base control shows the harness is deterministic.

    git show <base-rev>:src/spiky/lutorch/fast_multi_head_lut.py > /tmp/fmhl_base.py
    PYTHONPATH=src python src/spiky/lutorch/tests/regress_fast_mhl_vs_base.py /tmp/fmhl_base.py
    TORCHDYNAMO_DISABLE=1 PYTHONPATH=src python ...      # the same comparison in pure eager

torch._dynamo is reset before every module run, so each run compiles from scratch as a real training
process does. Without the reset, compile history accumulated across the configuration sweep made 5 of
892 tensors differ by 1-2 ULP between two sources with identical op sequences (identical in eager and
with the reset); HARNESS_NO_RESET=1 reproduces that sweep.
"""
import importlib.util
import itertools
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

torch._dynamo.config.suppress_errors = True
for _k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
           "accumulated_recompile_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 4096)
torch.use_deterministic_algorithms(True)

import spiky.lutorch.fast_multi_head_lut as CURRENT  # noqa: E402

RESET = os.environ.get("HARNESS_NO_RESET") != "1"

# (forward_mode, backward_topk, exp_outputs)
MODES = [("hard", 0, False), ("hybrid_smooth", 0, False), ("hybrid_smooth", 1, False),
         ("hybrid_smooth", 2, False), ("hard", 0, True)]
GATES = [None, "margin", "bounded"]
MHI = [False, True]
N_OUT = [4, 128]
PRECISIONS = [("cpu", torch.float64, False), ("cuda", torch.float64, False),
              ("cuda", torch.float32, True)]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"spiky.lutorch.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build(mod, mode, topk, exp_outputs, gate, mhi, n_out, device, wdtype, use_bf16):
    return mod.FastMultiHeadLut(
        input_dim=16, n_heads=2, n_outputs=n_out, n_anchor_pairs=4, tables_per_head=3,
        forward_mode=mode, backward_topk=topk, weight_dtype=wdtype, use_bf16=use_bf16,
        random_seed=7, initial_weights_noise=0.5, multi_head_input=mhi, device=torch.device(device),
        exp_outputs=exp_outputs, forward_confidence=gate is not None,
        confidence_form=gate or "bounded",
    )


def _run(m, x0, g):
    if RESET:
        torch._dynamo.reset()
    x = x0.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    out = m(x)
    (out * g).sum().backward()
    with torch.no_grad():
        out_eval = m(x0.clone())
    res = {"out": out.detach(), "out_eval": out_eval, "grad_x": x.grad}
    for name, p in m.named_parameters():
        res[f"grad_{name}"] = p.grad
    return res


def compare(mod_a, mod_b, label):
    n_cfg = n_tensors = 0
    bad = []
    for (mode, topk, expo), gate, mhi, n_out, (dev, wdtype, bf16) in itertools.product(
            MODES, GATES, MHI, N_OUT, PRECISIONS):
        if expo and (gate is not None or bf16 or mhi):
            continue  # exp_outputs refuses forward_confidence, use_bf16 and multi_head_input
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        torch.manual_seed(123)
        B = 9
        x0 = torch.randn(B, 32 if mhi else 16, dtype=wdtype, device=dev)
        g = torch.randn(B, 2, n_out, dtype=wdtype, device=dev)
        ma = _build(mod_a, mode, topk, expo, gate, mhi, n_out, dev, wdtype, bf16)
        mb = _build(mod_b, mode, topk, expo, gate, mhi, n_out, dev, wdtype, bf16)
        sa, sb = ma.state_dict(), mb.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), "init differs"
        ra, rb = _run(ma, x0, g), _run(mb, x0, g)
        assert ra.keys() == rb.keys()
        n_cfg += 1
        for k in ra:
            if ra[k] is None and rb[k] is None:
                continue
            n_tensors += 1
            if ra[k] is None or rb[k] is None or not torch.equal(ra[k], rb[k]):
                diff = (float((ra[k].double() - rb[k].double()).abs().max())
                        if ra[k] is not None and rb[k] is not None else float("nan"))
                bad.append((mode, topk, expo, gate, mhi, n_out, dev, str(wdtype), bf16, k, diff))
    print(f"[{label}] configs={n_cfg} tensors_compared={n_tensors} mismatches={len(bad)}")
    for b in bad:
        print("   MISMATCH", b)
    return bad


if __name__ == "__main__":
    base_path = sys.argv[1]
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "| dynamo disabled", os.environ.get("TORCHDYNAMO_DISABLE") == "1", "| reset per run", RESET)
    base, base2 = _load("_fmhl_base_a", base_path), _load("_fmhl_base_b", base_path)
    bad = compare(base, base2, "control: base vs base") + compare(base, CURRENT, "base vs current")
    sys.exit(1 if bad else 0)

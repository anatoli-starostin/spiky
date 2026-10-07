"""
[lut_nanochat] Environment smoke test: torch / CUDA / GPU / FA3 (pinned revision) / fp8 matmul.
Run from experiments/lut_nanochat/nanochat after `source runs/lut_env.sh && source .venv/bin/activate`:
    python runs/env_smoke_test.py
Exits non-zero, with a reason, on anything that would make the baseline non-comparable.
"""
import os, sys, json
import torch

fails = []
def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)

pins = json.load(open("../pins.json"))
check(torch.__version__ == pins["python_env"]["torch"], f"torch {torch.__version__} (want {pins['python_env']['torch']})")
check(torch.cuda.is_available(), "CUDA available")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"INFO GPU {name}, sm_{cap[0]}{cap[1]}, {mem:.1f} GiB, CUDA runtime {torch.version.cuda}, devices {torch.cuda.device_count()}")
    check(cap == (9, 0), "Hopper sm_90 (H100) - FA3 kernel requirement")
    check(mem > 75, "80 GB-class GPU")
    if "PCIe" in name:
        print("WARN H100 PCIe: valid, but expect longer step times and a different MFU peak; note it in the report")

rev = os.environ.get("NANOCHAT_FA3_REVISION")
check(rev == pins["fa3"]["revision"], f"NANOCHAT_FA3_REVISION={rev}")
from nanochat.flash_attention import HAS_FA3, USE_FA3, flash_attn
check(HAS_FA3 and USE_FA3, "FA3 loaded from the pinned Hub revision (if FAIL: see RUNBOOK troubleshooting 'FA3 unavailable')")
if HAS_FA3:
    q = torch.randn(2, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
    y = flash_attn.flash_attn_func(q, q, q, causal=True, window_size=(128, 0))
    check(y.shape == q.shape and torch.isfinite(y).all().item(), "FA3 sliding-window forward runs")

try:
    a = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    out = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    check(torch.isfinite(out).all().item(), "fp8 torch._scaled_mm runs")
except Exception as e:
    check(False, f"fp8 torch._scaled_mm ({type(e).__name__}: {e})")

try:
    import kernels
    print(f"INFO kernels {getattr(kernels, '__version__', '?')}")
except Exception:
    pass
print("SMOKE TEST:", "OK" if not fails else f"FAILED ({len(fails)})")
sys.exit(1 if fails else 0)

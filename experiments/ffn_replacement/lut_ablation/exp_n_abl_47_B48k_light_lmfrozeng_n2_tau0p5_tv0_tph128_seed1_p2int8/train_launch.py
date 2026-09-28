"""Launch train.py unchanged (runpy, same process) and report, once the model is built, WHICH implementation serves the
p2_int8 quantised path: the spiky_lutorch::p2_scalars CUDA op (extension loaded, op registered and enabled) or the torch
fallback (pow2_read). Pure observation: reads module flags only; no RNG, no graph, no data or optimiser interaction.

The report goes to stdout (train.log) as one line starting with "[p2_int8]". A torch fallback is printed as a loud
multi-line banner, so the run record always says which implementation produced the result (on purpose after a FAILED
kernel gate with SPIKY_P2_CUDA_DISABLE=1, or a problem otherwise).
"""
import os
import runpy
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _report():
    import torch
    from spiky.lutorch import pow2_int8 as K
    from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
    t0 = time.time()
    while time.time() - t0 < 900:                       # the op is registered while the model is built (LightMHL init)
        import gc
        luts = [o for o in gc.get_objects() if isinstance(o, LightMultiHeadLUT)] if K._tried else []
        if luts:
            break
        time.sleep(2)
    luts = [o for o in __import__("gc").get_objects() if isinstance(o, LightMultiHeadLUT)]
    n_quant = sum(getattr(m, "_quant", None) is not None for m in luts)
    probe = torch.zeros(1, device="cuda") if torch.cuda.is_available() else torch.zeros(1)
    kernel = bool(K._registered and K.op_available(probe))
    ext = K.load()
    detail = (f"quant layers {n_quant}/{len(luts)} | extension loaded={ext is not None} "
              f"({getattr(ext, '__file__', None)}) | op registered={K._registered} enabled={K._enabled} | "
              f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
              f"capability {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}")
    if n_quant and kernel:
        print(f"[p2_int8] IMPLEMENTATION: CUDA KERNEL (spiky_lutorch::p2_scalars) | {detail}", flush=True)
    else:
        bar = "!" * 100
        print(f"{bar}\n[p2_int8] IMPLEMENTATION: TORCH FALLBACK -- the CUDA kernel is NOT serving this arm "
              f"({K.available()[1]}) | {detail}\n{bar}", flush=True)


threading.Thread(target=_report, daemon=True).start()
runpy.run_path(os.path.join(HERE, "train.py"), run_name="__main__")

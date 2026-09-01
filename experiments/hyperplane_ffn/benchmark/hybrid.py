"""hybrid-v2 storage: bf16 dense weights, fp32 LUT tables and LUT input.

WHY THIS EXACT COMBINATION

  bf16 for the dense parts, STORED not autocast. Attention, the compress/decompress
  projections and above all the vocab-wide unembedder are ordinary GEMMs and want
  bf16. They must be STORED bf16, not run under torch.autocast over fp32 weights:
  autocast re-casts every dense weight on EVERY forward, and measured on the dense
  vanilla baseline -- a model with no LUT at all -- that tax is +1.43 ms at batch 48
  and +4.12 ms at 96. model.to(bfloat16) pays it once, at load.

  fp32 for the LUT tables and the LUT input. FastMultiHeadLut dispatches its
  hard-eval kernel on the INPUT dtype:

      use_native = self._native_eval_msb is not None and x.is_cuda \
                   and x.dtype in (torch.float32, torch.float64)

  so a blanket model.to(bfloat16) silently drops the native CUDA bit-pack kernel and
  falls back to the compiled path. A forward pre-hook casts the LUT's input back to
  fp32 to keep it alive. Keeping the tables fp32 costs nothing: bf16 tables were
  measured three independent ways and bought ~0%, because the gather is not bound by
  table bytes.

Do NOT set lut_use_bf16=True with fp32-stored tables: that casts the whole table to
bf16 on every forward, reading the full fp32 table and writing a bf16 copy each call.
It is slower than plain fp32.
"""
import torch


def apply(model, fp32_lut_input=True, fp32_tables=True):
    """Convert `model` to hybrid-v2 storage in place. Returns the model.

    Both LUT families are handled, and they need DIFFERENT treatment:

      FastMultiHeadLut (CompressionMHL / anchor-pair) -- tables back to fp32 AND a
        forward pre-hook casting the input back to fp32, because its native
        bit-pack kernel is selected on the input dtype and dies without it.

      TernaryHyperplaneMultiHeadLUT -- tables back to fp32, but NO input hook. Its
        addressing is a GEMM with no dtype gate, and the gather patch handles its own
        casting; hooking the input to fp32 here would make the patched slot return
        fp32 into a bf16 decompress Linear and raise.

    Missing the ternary case is not a silent 5% -- an earlier version handled only
    FastMultiHeadLut, left the ternary tables in bf16, and produced a 3.3x-slower
    "optimized" model that still looked plausible.
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    try:
        from spiky.lutorch.ternary_hyperplane_multi_head_lut import (
            TernaryHyperplaneMultiHeadLUT as _Tern)
    except Exception:                      # pragma: no cover
        _Tern = ()

    model = model.to(torch.bfloat16)
    for mod in model.modules():
        is_fast = isinstance(mod, FastMultiHeadLut)
        is_tern = bool(_Tern) and isinstance(mod, _Tern)
        if not (is_fast or is_tern):
            continue
        if fp32_tables:
            mod.weights.data = mod.weights.data.to(torch.float32)
        if fp32_lut_input and is_fast:
            mod.register_forward_pre_hook(
                lambda m, args: (args[0].float(),) + tuple(args[1:]))
    return model.eval()


def native_path_alive(model):
    """True if every LUT will take the native bit-pack kernel at eval.

    Checks the two conditions FastMultiHeadLut actually tests -- the kernel exists,
    and the input reaching it is fp32 -- rather than assuming the hook worked.
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    luts = [m for m in model.modules() if isinstance(m, FastMultiHeadLut)]
    if not luts:
        return None                      # dense baseline: not applicable
    return all(m._native_eval_msb is not None for m in luts) and \
        all(any(h is not None for h in m._forward_pre_hooks.values()) for m in luts)


def count_native_calls(model):
    """Instrument every LUT so callers can COUNT real native-kernel hits.

    Returns a dict with a mutable 'n'; run one forward and compare against the number
    of LUT modules. Cheaper than trusting a dtype rule.
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    counter = {'n': 0}
    for mod in model.modules():
        if not isinstance(mod, FastMultiHeadLut):
            continue
        orig = mod._hard_eval_native

        def wrapped(x, w, _o=orig):
            counter['n'] += 1
            return _o(x, w)

        mod._hard_eval_native = wrapped
    return counter

"""Detailed torch.profiler breakdown of exp061 vs exp060 forward/backward."""
import os, sys, gc, json
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_compare import TinyModel, FullModel, cfg, VOCAB_SIZE, DEVICE
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer

DEVICE_BS = cfg['device_batch_size']

def detailed_profile(name, model, optimizers, n_warmup=3, n_active=5):
    model.train()
    tokens = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, cfg['context_size']), device=DEVICE)
    targets = torch.randint(0, VOCAB_SIZE, (DEVICE_BS, cfg['context_size']), device=DEVICE)

    # Warmup.
    for _ in range(n_warmup):
        for opt in optimizers: opt.zero_grad()
        loss = model(tokens, targets); loss.backward()
        for opt in optimizers: opt.step()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_active):
            for opt in optimizers: opt.zero_grad()
            with torch.profiler.record_function("FORWARD"):
                loss = model(tokens, targets)
            with torch.profiler.record_function("BACKWARD"):
                loss.backward()
            with torch.profiler.record_function("OPT_STEP"):
                for opt in optimizers: opt.step()
        torch.cuda.synchronize()

    print(f'\n========== {name} ==========')
    # Top-N by CUDA time.
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# --- exp061 (bf16 Tiny) ---
print('Building TinyModel...')
tiny_model = TinyModel().to(DEVICE)
tiny_modules = [m for m in tiny_model.modules() if isinstance(m, TinyMultiHeadLut)]
ids = {id(m.weights) for m in tiny_modules}
adam_p = [p for p in tiny_model.parameters() if id(p) not in ids]
tiny_adam = torch.optim.AdamW(adam_p, lr=cfg['adam_lr'], betas=(0.9, 0.95))
tiny_opt = TinyMultiHeadLutOptimizer(tiny_modules, lr=cfg['adam_lr'], state_dtype=torch.bfloat16, compute_dtype=torch.float32)
detailed_profile('exp061 (bf16 Tiny)', tiny_model, [tiny_adam, tiny_opt])

del tiny_model, tiny_adam, tiny_opt, tiny_modules
gc.collect(); torch.cuda.empty_cache()

# --- exp060 (fp32 MHLut) ---
print('Building FullModel...')
full_model = FullModel().to(DEVICE)
full_adam = torch.optim.AdamW(full_model.parameters(), lr=cfg['adam_lr'], betas=(0.9, 0.95))
detailed_profile('exp060 (fp32 MHLut)', full_model, [full_adam])

"""Quick profile of exp053-mod-v2: 5 warmup + 5 profiled steps, breaks down
forward / backward time per major op.
"""
import sys, os, json, time
import torch
import torch.nn as nn

NANOCHAT_ROOT = os.environ.get('NANOCHAT_ROOT', '/home/starost/nanochat')
if NANOCHAT_ROOT not in sys.path:
    sys.path.insert(0, NANOCHAT_ROOT)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

# Load the exact model from train.py without entering the train loop.
# Trick: read train.py and exec until just before the loop starts.
src = open(os.path.join(EXP_DIR, 'train.py')).read()
loop_marker = 'model.train()'
header = src.split(loop_marker)[0] + 'model.train()\n'
train_py_path = os.path.join(EXP_DIR, 'train.py')
ns = {'__name__': '__main__', '__file__': train_py_path}
exec(header, ns)

model = ns['model']
adam_opt = ns['adam_opt']
bit_opt = ns['bit_opt']
train_loader = ns['train_loader']
DEVICE = ns['DEVICE']

print('--- WARMUP (5 steps, untimed) ---')
for i in range(5):
    adam_opt.zero_grad()
    bit_opt.zero_grad()
    x, y = next(train_loader)
    loss = model(x, targets=y)
    loss.backward()
    adam_opt.step()
    bit_opt.step()
torch.cuda.synchronize()

print('--- PROFILE (5 steps with detailed timing) ---')

def t():
    torch.cuda.synchronize()
    return time.perf_counter()

# coarse timing of forward / backward / step
records = []
for i in range(5):
    adam_opt.zero_grad()
    bit_opt.zero_grad()
    t0 = t()
    x, y = next(train_loader)
    t1 = t()
    loss = model(x, targets=y)
    t2 = t()
    loss.backward()
    t3 = t()
    adam_opt.step()
    bit_opt.step()
    t4 = t()
    records.append((t1-t0, t2-t1, t3-t2, t4-t3))
    print(f'  step{i}: data={t1-t0:.3f}s fwd={t2-t1:.3f}s bwd={t3-t2:.3f}s opt={t4-t3:.3f}s total={t4-t0:.3f}s')

avg = lambda i: sum(r[i] for r in records) / len(records)
print(f'\n--- AVG over 5 steps ---')
print(f'  data load: {avg(0)*1000:.1f}ms')
print(f'  forward:   {avg(1)*1000:.1f}ms')
print(f'  backward:  {avg(2)*1000:.1f}ms')
print(f'  optimizer: {avg(3)*1000:.1f}ms')
total = avg(0)+avg(1)+avg(2)+avg(3)
print(f'  TOTAL:     {total*1000:.1f}ms ({total:.3f} sec/step)')

# Now use torch.profiler for fine-grain breakdown of forward
print('\n--- TORCH PROFILER on 1 step (forward+backward) ---')
from torch.profiler import profile, ProfilerActivity, schedule
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
    with_stack=False,
) as prof:
    adam_opt.zero_grad()
    bit_opt.zero_grad()
    x, y = next(train_loader)
    loss = model(x, targets=y)
    loss.backward()
    adam_opt.step()
    bit_opt.step()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))

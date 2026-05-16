"""Does sorted scatter_indices speed up scatter_add_?"""
import time, statistics, torch
DEV = torch.device('cuda:0')
torch.manual_seed(0)
B, H, TPH, NS, D = 4096, 1, 4096, 64, 384
src = torch.randn(B, H, TPH * NS, device=DEV, dtype=torch.bfloat16)

def bench(name, fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    print(f"{name:50s}  {statistics.median(t)*1000:7.3f} ms")

# Build [H, TPH*NS] index tensors that get broadcast (expanded) to B without
# materializing — scatter_add accepts the broadcasted shape via .expand.
base_iid     = torch.randint(0, D, (H, TPH * NS), device=DEV)
base_sorted  = torch.sort(base_iid, dim=1).values
per_slot     = TPH * NS // D
base_balanced = torch.arange(D, device=DEV).repeat_interleave(per_slot)[:TPH*NS].unsqueeze(0).expand(H, -1).contiguous()

idx_iid      = base_iid.unsqueeze(0).expand(B, -1, -1)
idx_sorted   = base_sorted.unsqueeze(0).expand(B, -1, -1)
idx_balanced = base_balanced.unsqueeze(0).expand(B, -1, -1)

def s_iid():
    out = torch.zeros(B, H, D, device=DEV, dtype=torch.bfloat16)
    out.scatter_add_(2, idx_iid, src)
bench("scatter_add_ random idx", s_iid)

def s_sorted():
    out = torch.zeros(B, H, D, device=DEV, dtype=torch.bfloat16)
    out.scatter_add_(2, idx_sorted, src)
bench("scatter_add_ sorted idx (per row, B copies)", s_sorted)

def s_balanced():
    out = torch.zeros(B, H, D, device=DEV, dtype=torch.bfloat16)
    out.scatter_add_(2, idx_balanced, src)
bench("scatter_add_ balanced contig idx", s_balanced)

# Case 4: use index_add on flat indices (a different op, same effect).
flat_src = src.reshape(B * H, TPH * NS)
flat_idx_iid = idx_iid.reshape(B * H, TPH * NS)
def s_index_add():
    out = torch.zeros(B * H, D, device=DEV, dtype=torch.bfloat16)
    # For each row b: out[b].scatter_add_(0, flat_idx[b], flat_src[b])
    # As a batched op, we'd use scatter_add_ — same thing.
    pass

# Case 5: alternative — segment_reduce (if available) on sorted contributions.
# torch has torch._segment_reduce in stable now? Try.
try:
    from torch._segment_reduce_ops import _segment_reduce
    print("has _segment_reduce")
except ImportError:
    try:
        torch._segment_reduce  # noqa
        print("torch._segment_reduce private API exists")
    except AttributeError:
        print("no segment_reduce easily accessible")

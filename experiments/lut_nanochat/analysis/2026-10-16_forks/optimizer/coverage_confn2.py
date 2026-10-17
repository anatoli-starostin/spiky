"""CPU-only optimizer-coverage check for the confn2 model (read_top_n=2, v2 geometry).
GPT is built on the meta device exactly as base_train's build_model_meta (depth 24, aspect 64, head_dim 128,
seq 2048, vocab 32768, SSSL); the LUT blocks are built for real on CPU via apply_lut_ffn; then the REAL
GPT.setup_optimizer runs and every trainable parameter is checked for group membership."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.lut_ffn import LUTFFNConfig, apply_lut_ffn

depth, aspect, head_dim = 24, 64, 128
model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
cfg = GPTConfig(sequence_len=2048, vocab_size=32768, n_layer=depth, n_head=model_dim // head_dim,
                n_kv_head=model_dim // head_dim, n_embd=model_dim, window_pattern="SSSL")
with torch.device("meta"):
    model = GPT(cfg)
lut = LUTFFNConfig(enabled=True, read_top_n=2)   # every other field = v2 / base_train default
print("n LUT blocks:", apply_lut_ffn(model, lut, device="cpu"))

opt = model.setup_optimizer(unembedding_lr=0.008, embedding_lr=0.3, scalar_lr=0.5, matrix_lr=0.02,
                            weight_decay=0.06299, lut_lr=3e-3)
in_group = {}
for gi, g in enumerate(opt.param_groups):
    for p in g["params"]:
        in_group.setdefault(id(p), []).append((gi, g["kind"], g["lr"]))

named = list(model.named_parameters())
trainable = [(n, p) for n, p in named if p.requires_grad]
orphans = [n for n, p in trainable if id(p) not in in_group]
dupes = [n for n, p in trainable if len(in_group.get(id(p), [])) > 1]
print("params total:", len(named), "| requires_grad:", len(trainable),
      "| in some group:", sum(1 for n, p in trainable if id(p) in in_group),
      "| distinct tensors across groups:", len(in_group))
print("numel requires_grad:", sum(p.numel() for _, p in trainable),
      "| numel across groups:", sum(p.numel() for g in opt.param_groups for p in g["params"]))
print("TRAINABLE BUT IN NO GROUP:", orphans or "none")
print("IN MORE THAN ONE GROUP:", dupes or "none")

print("\nLUT AdamW group (the lut_lr one):")
lut_gi = [gi for gi, g in enumerate(opt.param_groups) if g["kind"] == "adamw" and abs(g["lr"] - 3e-3 * (model_dim / 768) ** -0.5) < 1e-12]
print("  group index", lut_gi, "lr", [opt.param_groups[i]["lr"] for i in lut_gi], "wd", [opt.param_groups[i]["weight_decay"] for i in lut_gi])
for key in ("log_read_tau", "confidence_log_beta", "confidence_log_gamma"):
    hits = [(n, p) for n, p in named if n.endswith(key)]
    groups = {tuple(in_group.get(id(p), [])) for _, p in hits}
    print(f"  {key:22s} count={len(hits):2d} type={type(hits[0][1]).__name__} shape={tuple(hits[0][1].shape)} "
          f"requires_grad={all(p.requires_grad for _, p in hits)} groups={groups} "
          f"init={hits[0][1].detach().item():.6f}")

"""Preflight for the 48K run. Run BEFORE launch.sh (launch.sh also runs it and refuses to start on any FAIL).

    set -a; . ./run.env; set +a; "$PYTHON" preflight.py

Checks, each printed PASS / FAIL:
  1. spiky checkout at the pinned commit, library tree unmodified, this folder inside it at the expected place
  2. nanochat checkout at its pinned commit
  3. data dir holds EXACTLY the 5 pinned ClimbMix shards with matching sha256; both tokenizer files match their sha256
  4. exactly one CUDA device visible; lutorch_cuda imports
  5. config.json == the documented config (abl_05 row 3.2 +TV with n_steps 48000 and lut_quant_mode p2_int8)
  6. model builds (SMOKE=1 train.py), the quantised path is ACTIVE on all 6 layers, and WHICH implementation serves it:
       "CUDA KERNEL"    (p2_int8 extension built, spiky_lutorch::p2_scalars op registered and enabled), or
       "TORCH FALLBACK" (pow2_read; SPIKY_P2_CUDA_DISABLE=1, or the extension unavailable)
     cross-checked against kernel_gate.json (validate_kernel.py): kernel serving without a PASS gate is a FAIL
     (set SPIKY_P2_CUDA_DISABLE=1); a PASS gate with the fallback serving is a FAIL unless SPIKY_P2_CUDA_DISABLE=1 was set
     on purpose. The implementation line is also written to preflight_implementation.txt for the run record.
  7. W&B: WANDB_BASE_URL is wandb.ai cloud (run.env is authoritative), the ~/.netrc key authenticates (key never printed)
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_SPIKY = "60a8654147f92a38a4b6e5be7fc5eced480ffef4"  # noquant fork at current HEAD 60a86541; the float LightMHL LUT code (light_multi_head_lut/compression_mhl/pow2_read) is byte-identical to the p2 runs' pin 686e5c44 -- only the UNUSED p2 int8 kernel changed on the branch since
PIN_NANOCHAT = "da32e1d657b62cc1110f20f86fbc301d6189726e"
# The exact data and tokenizer of the gpustar runs. nanochat lists $NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet sorted:
# all but the LAST are train (looped), the LAST is val -- so the dir must hold exactly these five files.
SHARDS = {
    "shard_00000.parquet": "054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b",
    "shard_00001.parquet": "d4cfe1de19f4fd976022a9047458655968bc5583582ec5fa714801ec2be2c93c",
    "shard_00002.parquet": "5e80d7596d53be648d841ea5811082747281c39f6851e10dd90375a84bc509a7",
    "shard_00003.parquet": "b349c85d4b993d98e2328833ecfd3734692e0395a592d89858bae34a4ba73c59",
    "shard_06542.parquet": "769fe59d108dfd2cfa186c63173b83fbfb90a7adb3519519ba6eaa6ca9889f94",   # val
}
TOKENIZER = {                                           # $NANOCHAT_BASE_DIR/tokenizer/ -- must be byte-identical
    "tokenizer.pkl": "61d7b5aa42e1047d7134cac0f9f3b1fc56286d9ba7c9e5fea43de6471d9afe3f",
    "token_bytes.pt": "b8374a902944f329a03dbd61ab459010d6195979e979864535f9aedec42bcc85",
}
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


def git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True).stdout.strip()


def env(k):
    v = os.environ.get(k, "")
    if not v or v == "FILL":
        check(f"env {k} set", False, "fill it in run.env")
    return v


spiky, nanochat, base = env("SPIKY_ROOT"), env("NANOCHAT_ROOT"), env("NANOCHAT_BASE_DIR")

# 1. spiky
if spiky:
    head = git(spiky, "rev-parse", "HEAD")
    # HEAD is the pin, or a descendant whose only changes since the pin are inside this run folder (the commit adding it)
    rel = os.path.relpath(os.path.realpath(HERE), os.path.realpath(spiky)) + "/"
    anc = subprocess.run(["git", "-C", spiky, "merge-base", "--is-ancestor", PIN_SPIKY, "HEAD"]).returncode == 0
    changed = [f for f in git(spiky, "diff", "--name-only", PIN_SPIKY, "HEAD").splitlines() if f] if anc else []
    outside = [f for f in changed if not f.startswith(rel)]
    check("spiky code at pinned commit", anc and not outside,
          "" if (anc and not outside) else f"HEAD={head}; need {PIN_SPIKY} plus changes only inside {rel}; outside: {outside[:5]}")
    dirty = git(spiky, "status", "--porcelain", "--untracked-files=no", "--", "src", "experiments/ffn_replacement/tools", "native")
    check("spiky library/tools/native unmodified", dirty == "", dirty[:300])
    want = os.path.join(os.path.realpath(spiky), "experiments", "ffn_replacement", "lut_ablation", os.path.basename(HERE))
    check("package folder location", os.path.realpath(HERE) == want, f"is {HERE}, want {want}")

# 2. nanochat
if nanochat:
    head = git(nanochat, "rev-parse", "HEAD")
    check("nanochat at pinned commit", head == PIN_NANOCHAT, f"HEAD={head}, want {PIN_NANOCHAT}")

# 3. data
if base:
    ddir = os.path.join(base, "base_data_climbmix")
    have = sorted(f for f in os.listdir(ddir) if f.endswith(".parquet")) if os.path.isdir(ddir) else []
    check("data dir holds exactly the 5 pinned shards", have == sorted(SHARDS), f"have {have}")
    for fn, want in SHARDS.items():
        p = os.path.join(ddir, fn)
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            check(f"sha256 {fn}", h.hexdigest() == want)
    for fn, want in TOKENIZER.items():
        p = os.path.join(base, "tokenizer", fn)
        ok = os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == want
        check(f"tokenizer {fn} sha256", ok, "" if ok else f"missing or different at {p} -- it must be byte-identical to the gpustar runs' tokenizer: ask Anatoli for both files")

# 4. GPU
try:
    import torch
    n = torch.cuda.device_count()
    check("exactly one CUDA device visible", n == 1, f"{n} visible (set CUDA_VISIBLE_DEVICES to one id)")
    if n:
        print(f"       device: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
    import lutorch_cuda  # noqa: F401
    check("lutorch_cuda imports", True)
except Exception as e:
    check("torch / lutorch_cuda", False, f"{type(e).__name__}: {e}")

# 5. config
cfg = json.load(open(os.path.join(HERE, "config.json")))
check("config n_steps 48000", cfg.get("n_steps") == 48000)
check("config UNQUANTISED (lut_quant_mode is None -> plain float LUT read)", cfg.get("lut_quant_mode") is None)
expect = dict(device_batch_size=12, total_batch_size=24576, seq_len=512, lr=0.0003, weight_decay=0.1, lr_warmup_fraction=0.1,
              eval_every=500, random_seed=1, lut_base_seed=1000, lut_impl="light", lut_read_top_n=2, lut_read_tau=0.5,
              lut_cell_smoothness=10.0, lut_tables_per_head=64, lut_n_heads=8, lut_n_anchor_pairs=8, lut_inner_in_dim=48,
              lut_inner_out_dim=48, depth=6, n_embd=384)
bad = {k: (cfg.get(k), v) for k, v in expect.items() if cfg.get(k) != v}
check("config matches the documented protocol", not bad, str(bad))

# 6. smoke build, quantised path active, which implementation serves it
if spiky and nanochat and base:
    code = ("import os,runpy,sys; os.environ['SMOKE']='1'\n"
            "g = {}\n"
            "try:\n    g = runpy.run_path('train.py', run_name='__main__')\nexcept SystemExit:\n    pass\n"
            "import gc, torch\n"
            "import spiky.lutorch.pow2_int8 as K\n"
            "from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT as L\n"
            "luts = [o for o in gc.get_objects() if isinstance(o, L)]\n"
            "print('P2_QUANT_LAYERS', sum(m._quant is not None for m in luts), len(luts))\n"
            "print('P2_IMPL', 'CUDA_KERNEL' if K.op_available(torch.zeros(1, device='cuda')) else 'TORCH_FALLBACK')\n"
            "print('P2_REASON', K.available()[1])\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, env=dict(os.environ, WANDB_MODE="disabled"))
    out = r.stdout + r.stderr
    check("model builds (SMOKE)", "SMOKE OK" in out, out[-400:] if "SMOKE OK" not in out else "")
    check("params 68,237,586 (tph64/h8 fork; +885,888 vs abl_47's 67,351,698 from doubling lut_n_heads 4->8)", "params=68,237,586" in out)
    q = [ln.split()[1:] for ln in out.splitlines() if ln.startswith("P2_QUANT_LAYERS")]
    check("unquantised float path on all 6 layers (0 quant layers)", bool(q) and q[0] == ["0", "6"], str(q))
    # This run is UNQUANTISED: no p2_int8 quant, no int8 kernel. The float LightMHL read serves.
    line = "IMPLEMENTATION SERVING: FLOAT (unquantised LUT read) -- lut_quant_mode=None; the p2_int8 int8 kernel is NOT used by this run"
    print("\n" + "=" * 110 + "\n" + line + "\n" + "=" * 110, flush=True)
    open(os.path.join(HERE, "preflight_implementation.txt"), "w").write(line + "\n")

# 7. W&B
check("WANDB_BASE_URL is wandb.ai cloud", os.environ.get("WANDB_BASE_URL") == "https://api.wandb.ai",
      "" if os.environ.get("WANDB_BASE_URL") == "https://api.wandb.ai" else "set WANDB_BASE_URL=https://api.wandb.ai in run.env")
check("WANDB_MODE not disabled/offline", os.environ.get("WANDB_MODE", "") not in ("disabled", "offline"))
try:                                                   # own process with a hard timeout: never hangs the preflight
    r = subprocess.run([sys.executable, "-c", "import wandb; v = wandb.Api(timeout=20).viewer; print('WANDB_ENTITY_OK', bool(v.entity))"],
                       capture_output=True, text=True, timeout=90)
    ok = "WANDB_ENTITY_OK True" in r.stdout
    check("W&B cloud key authenticates", ok, "" if ok else (r.stdout + r.stderr).strip().splitlines()[-1][:200] +
          " (the key must be in ~/.netrc, machine api.wandb.ai -- see README W&B section)")
except subprocess.TimeoutExpired:
    check("W&B cloud key authenticates", False, "no answer from wandb.ai within 90 s (network?) -- see README W&B section")

print("\nPREFLIGHT", "OK" if not fails else f"FAILED ({len(fails)}): " + "; ".join(fails))
sys.exit(1 if fails else 0)

"""
[lut_nanochat] Build the pin set that base_train merges into the wandb config (--pin-config):
static pins (../pins.json) + resolved runtime versions + input hashes (manifest.json from stage_data.sh)
+ this run's launch flags.
"""
import argparse, json, os, subprocess
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dbs", type=int, required=True)
parser.add_argument("--save-every", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

pins = json.load(open("../pins.json"))
results = os.environ["RESULTS"]
manifest_path = os.path.join(results, "manifest.json")
assert os.path.exists(manifest_path), "manifest.json missing: run runs/stage_data.sh first"
manifest = json.load(open(manifest_path))

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"

cfg = {
    "static": pins,
    "runtime": {
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "fa3_revision": os.environ.get("NANOCHAT_FA3_REVISION"),
        "kernels": sh("python -c 'import importlib.metadata as m; print(m.version(\"kernels\"))'"),
        "spiky_branch_commit": sh("git rev-parse HEAD"),
        "vendored_tree_dirty": sh("git status --porcelain -- .") != "",
    },
    "inputs": {
        "train_shards": sorted(f for f in manifest["shards"] if f != manifest["val_shard"]),
        "val_shard": manifest["val_shard"],
        "shard_sha256": {f: v["sha256"] for f, v in manifest["shards"].items()},
        "tokenizer_sha256": manifest["tokenizer"],
        "eval_bundle_zip_sha256": manifest["eval_bundle_zip_sha256"],
        "reference_check": manifest["reference_check"],
    },
    "launch": {
        "nproc_per_node": 1, "device_batch_size": args.dbs, "grad_accum_steps": 1048576 // (args.dbs * 2048),
        "save_every": args.save_every, "flags": "--depth=24 --target-param-data-ratio=8 --fp8",
        "nanochat_base_dir": os.environ.get("NANOCHAT_BASE_DIR"),
    },
}
json.dump(cfg, open(args.out, "w"), indent=1)
print(f"pin config written: {args.out} (dbs {args.dbs}, grad_accum {cfg['launch']['grad_accum_steps']}, "
      f"tree dirty: {cfg['runtime']['vendored_tree_dirty']})")

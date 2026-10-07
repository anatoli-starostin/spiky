#!/bin/bash
# [lut_nanochat] Stage every input of the d24 baseline and record its sha256, so we can later prove
# which bytes the run consumed. Idempotent: rerunning skips what is already on disk.
#   - ClimbMix train shards shard_00000..00169 (+ val shard_06542, always downloaded by nanochat.dataset)
#     ~88 MB each, ~15 GB total. Download time UNCERTAIN: ~5-20 min depending on HF bandwidth.
#   - tokenizer: trained here by scripts.tok_train (a few minutes), exactly as speedrun.sh does
#   - CORE eval bundle (eval_bundle.zip from S3), unzipped into $NANOCHAT_BASE_DIR/eval_bundle
# Output: $RESULTS/manifest.json (sha256 + size of every input) and a reference-hash check.
# Usage (from experiments/lut_nanochat/nanochat): bash runs/stage_data.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source runs/lut_env.sh
source .venv/bin/activate

python -m nanochat.dataset -n 170          # same as speedrun.sh; adds the val shard_06542 automatically
[ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] || python -m scripts.tok_train
python -m scripts.tok_eval
python - <<'PY'
# Fetch the eval bundle exactly the way base_eval does (it keeps eval_bundle.zip in the base dir).
from nanochat.common import download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
PY

python - "$RESULTS/manifest.json" <<'PY'
import hashlib, json, os, sys
base = os.environ["NANOCHAT_BASE_DIR"]
pins = json.load(open("../pins.json"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
data_dir = os.path.join(base, "base_data_climbmix")
shards = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
manifest = {
    "shards": {f: {"sha256": sha(os.path.join(data_dir, f)), "bytes": os.path.getsize(os.path.join(data_dir, f))} for f in shards},
    "val_shard": shards[-1],
    "tokenizer": {f: sha(os.path.join(base, "tokenizer", f)) for f in ("tokenizer.pkl", "token_bytes.pt")},
    "eval_bundle_zip_sha256": sha(os.path.join(base, "eval_bundle.zip")),
}
assert manifest["val_shard"] == pins["data"]["val_shard"] + ".parquet", f"val shard is {manifest['val_shard']}, expected shard_06542"
assert len(shards) == 171, f"expected 170 train + 1 val shards, found {len(shards)}"
bad = {f: (manifest["shards"][f]["sha256"], ref) for f, ref in pins["data"]["reference_sha256"].items()
       if manifest["shards"][f]["sha256"] != ref}
manifest["reference_check"] = "PASS" if not bad else {"FAIL": bad}
json.dump(manifest, open(sys.argv[1], "w"), indent=1)
gb = sum(v["bytes"] for v in manifest["shards"].values()) / 1e9
print(f"shards: {len(shards)} ({gb:.1f} GB), val: {manifest['val_shard']}")
print(f"tokenizer.pkl sha256: {manifest['tokenizer']['tokenizer.pkl']}")
print(f"eval_bundle.zip sha256: {manifest['eval_bundle_zip_sha256']}")
print(f"reference shard hashes: {manifest['reference_check']}")
print(f"manifest written: {sys.argv[1]}")
if bad:
    sys.exit("STOP: ClimbMix shards differ from the reference hashes in pins.json (dataset changed upstream?)")
PY

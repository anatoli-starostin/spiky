"""Backfill results/<run>/run_config.json from wandb for every lut_nanochat run (spiky-nanochat project).
Each file records what pin_config.json does not: the exact base_train argv, the full resolved config, the
launcher, the git commit, the host, the outcome, and a human-readable summary of the run's distinguishing knobs."""
import json, os, sys
import wandb

R = "/home/astarostin/projects/lut_nanochat_wt/experiments/lut_nanochat/results"
P = "anatoli-starostin-relocation/spiky-nanochat"
# results folder -> (wandb run ids, launcher, launch notes)
MAP = {
    "d24-dense-1xh100-s0": (["d24-dense-1xh100-s0-4bad20ea"], "runs/baseline_d24_1xh100.sh", ""),
    "distill_d24_from_d24_1xh100": (["distill_d24_from_d24_1xh100-dcd85603"], "runs/distill_d24_1xh100.sh", ""),
    "distill_d24_from_d24_alpha05_1xh100-s0": (["distill_d24_from_d24_alpha05_1xh100-s0-35fa2273"], "runs/distill_d24_1xh100.sh",
        "Ran on the worker VM; RUN_NAME/MODEL_TAG and the distill alpha overridden at launch (see argv)."),
    "d24-lutffn-confn1-1xh100-s0": (["d24-lutffn-confn1-1xh100-s0-84036fb8", "d24-lutffn-confn1-1xh100-s0-daf5c6f6"],
        "runs/lut_ffn_d24_1xh100.sh", "Two attempts used this folder: d24-lutffn-confn1-1xh100-s0-84036fb8 (31 steps; deleted from wandb on 2026-10-23, so not recorded here) and daf5c6f6 (crashed at 846, horizon 1000)."),
    "d24-lutffn-confn1-1xh100-s0-full-ABORTED-nolrrescale": (["d24-lutffn-confn1-1xh100-s0-full-9dac4d16"], "runs/lut_ffn_d24_1xh100.sh",
        "wandb run name d24-lutffn-confn1-1xh100-s0-full; folder renamed after the abort (LR rescale left on, 1.4142x)."),
    "d24-lutffn-confn1-1xh100-s0-full-v2": (["d24-lutffn-confn1-1xh100-s0-full-v2-9811cced"], "runs/lut_ffn_d24_1xh100.sh",
        "Launched with RUN_NAME=d24-lutffn-confn1-1xh100-s0-full-v2, MODEL_TAG=d24_lutffn_confn1_1xh100_full_v2 and "
        "NUM_ITERATIONS=5568 in the environment (that launcher does not pin the horizon)."),
    "d24-lutffn-confn1-h32-tph32-1xh100-s0": (["d24-lutffn-confn1-h32-tph32-1xh100-s0-7f4cc747"], "runs/lut_ffn_d24_h32_tph32_1xh100.sh",
        "nap=8 fork. Stopped deliberately at step 3778 to free the GPU."),
    "d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0": (["d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0-04a80c7b"],
        "runs/lut_ffn_d24_h32_tph32_nap9_1xh100.sh", "Ran on the worker VM. Stopped deliberately at step 2580."),
    "d24-lutffn-confn2-1xh100-s0": (["d24-lutffn-confn2-1xh100-s0-f1aad8a0"], "runs/lut_ffn_d24_confn2_1xh100.sh",
        "Launched with NUM_ITERATIONS=5568 in the environment. Stopped deliberately at step 2133."),
    "d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0": (["d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0-036e09af"],
        "runs/lut_ffn_d24_projmuonfp8_lrrs_1xh100.sh", "Ran on the worker VM. Stopped deliberately at step 2098."),
}

def unwrap(cfg):
    out = {}
    for k, v in cfg.items():
        if k.startswith("_") or k == "pins":
            continue
        out[k] = v["value"] if isinstance(v, dict) and set(v) >= {"value"} else v
    return out

def lr_from_log(folder):
    import re
    try:
        for line in open(os.path.join(R, folder, "train.log"), errors="ignore"):
            m = re.search(r"Scaling LRs by ([0-9.]+)", line)
            if m: return float(m.group(1))
    except OSError:
        pass
    return None

def summary_of(c, scal):
    lut = bool(c.get("lut_ffn"))
    if lut:
        if c.get("lut_ffn_proj_fp8_muon"):
            proj = "fp8 GEMMs, weights optimised by Muon (--lut-ffn-proj-fp8-muon)"
        elif c.get("lut_ffn_proj_bf16"):
            proj = "bf16 GEMMs, fp32 masters, AdamW (--lut-ffn-proj-bf16)"
        elif "lut_ffn_proj_bf16" in c or "lut_ffn_proj_fp8_muon" in c:
            proj = "fp32 island, AdamW (default)"
        else:
            proj = "fp32 island, AdamW (run predates the projection flags)"
        h, d = c.get("lut_ffn_h"), c.get("lut_ffn_d")
        ffn = (f"LUT-FFN ConfidenceLUT h={h} d={d} r={h * d if h and d else '?'} tph={c.get('lut_ffn_tph')} "
               f"nap={c.get('lut_ffn_nap')} (cells={2 ** c['lut_ffn_nap'] if c.get('lut_ffn_nap') else '?'}) "
               f"read_top_n={c.get('lut_ffn_read_top_n')}")
    else:
        proj, ffn = "n/a", "dense MLP"
    distill = c.get("distill_from")
    return {
        "ffn": ffn,
        "lut_projections": proj,
        "lut_hparams": ({k: c.get(k) for k in ("lut_ffn_beta", "lut_ffn_gamma", "lut_ffn_seed", "lut_ffn_weight_init_std",
                                               "lut_ffn_table_dropout", "lut_ffn_tv_lambda", "lut_ffn_lr")} if lut else None),
        "lr_rescale": ("OFF (--no-lr-rescale, batch_lr_scale 1.0)" if c.get("no_lr_rescale")
                       else f"ON (batch_lr_scale {scal:.4f})" if scal else "ON (sqrt(B/B_ref))"),
        "horizon": (f"num_iterations={c.get('num_iterations')} (user-provided)" if (c.get("num_iterations") or -1) > 0
                    else "auto-derived from target_param_data_ratio"),
        "fp8_backbone": c.get("fp8"),
        "device_batch_size": c.get("device_batch_size"),
        "distillation": ({"teacher": distill, "alpha": c.get("distill_alpha"), "temperature": c.get("distill_temperature")}
                         if distill else None),
    }

api = wandb.Api()
BY_ID = {r.id: r for r in api.runs(P)}          # direct api.run() misses some older runs; the listing has them all
for folder, (ids, launcher, note) in MAP.items():
    path = os.path.join(R, folder)
    if not os.path.isdir(path):
        print("SKIP (no local folder):", folder); continue
    entries = []
    for rid in ids:
        if rid not in BY_ID:
            print(f"  WARN {rid} not in project listing; ids that look similar: {[i for i in BY_ID if rid.split('-')[-1] in i]}")
            continue
        r = BY_ID[rid]
        md = r.metadata or {}
        c = unwrap(dict(r.config))
        s = dict(r.summary)
        scal = s.get("batch_lr_scale") or c.get("batch_lr_scale") or lr_from_log(folder)
        entries.append({
            "wandb_run_id": r.id, "wandb_name": r.name, "wandb_url": r.url, "wandb_group": r.group,
            "state": r.state, "created_at": r.created_at, "host": md.get("host"),
            "final_step_logged": s.get("_step"),
            "final_metrics": {k: s.get(k) for k in ("val/bpb", "core_metric", "core_standalone_base_eval",
                                                    "val_bpb_base_eval", "train/loss") if s.get(k) is not None},
            "tags": list(r.tags),
            "summary": summary_of(c, scal),
            "command": {"program": md.get("program"), "argv": md.get("args")},
            "git": md.get("git"),
            "python": md.get("python"), "gpu": md.get("gpu"), "cuda": md.get("cuda") or md.get("cudaVersion"),
            "config": dict(sorted(c.items())),
        })
    doc = {
        "_about": ("Authoritative configuration of this run, backfilled from its wandb record on 2026-10-23. "
                   "pin_config.json in this folder is the environment pin (its 'wandb' and 'launch.flags' sections are "
                   "the dense-baseline template and do NOT describe this run's LUT/optimiser flags)."),
        "results_folder": folder, "launcher": launcher, "launch_notes": note, "wandb_runs": entries,
    }
    with open(os.path.join(path, "run_config.json"), "w") as f:
        json.dump(doc, f, indent=2, default=str)
        f.write("\n")
    e = entries[-1]
    print(f"WROTE {folder}/run_config.json  [{len(entries)} run(s)]  {e['summary']['ffn']} | proj: {e['summary']['lut_projections']} | "
          f"lr: {e['summary']['lr_rescale']} | {e['summary']['horizon']} | argv words: {len(e['command']['argv'] or [])}")

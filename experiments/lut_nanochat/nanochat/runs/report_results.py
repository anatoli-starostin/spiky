"""
[lut_nanochat] Collect the reportable numbers into $RESULTS/summary.json and push the standalone CORE
to the wandb run summary (same run id, resumed). A wandb failure warns, never fails.
    python runs/report_results.py --results <results dir>
"""
import argparse, csv, glob, json, os, re

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True)
args = parser.parse_args()
R = args.results
base = os.environ["NANOCHAT_BASE_DIR"]

def grep_all(path, pattern):
    if not os.path.exists(path):
        return []
    return re.findall(pattern, open(path, errors="replace").read())

train_log, eval_log = os.path.join(R, "train.log"), os.path.join(R, "base_eval.log")
csvs = sorted(glob.glob(os.path.join(base, "base_eval", "base_model_*.csv")))
core_csv = csvs[-1] if csvs else None
tasks = {}
if core_csv:
    with open(core_csv) as f:
        for row in csv.reader(f):
            if len(row) == 3 and row[0].strip() not in ("Task", "CORE"):
                tasks[row[0].strip()] = {"accuracy": float(row[1]), "centered": float(row[2])}
core_standalone = grep_all(eval_log, r"CORE metric: ([0-9.]+)")
in_training_core = grep_all(train_log, r"Step (\d+) \| CORE metric: ([0-9.]+)")
val_bpb_train = grep_all(train_log, r"Step (\d+) \| Validation bpb: ([0-9.]+)")
resumes = open(os.path.join(R, "resumes.log")).read().splitlines() if os.path.exists(os.path.join(R, "resumes.log")) else []

summary = {
    "run": os.environ.get("RUN_NAME"),
    "core_standalone_base_eval": float(core_standalone[-1]) if core_standalone else None,
    "core_in_training_last": float(in_training_core[-1][1]) if in_training_core else None,
    "val_bpb_base_eval": float(v[-1]) if (v := grep_all(eval_log, r"val bpb: ([0-9.]+)")) else None,
    "train_bpb_base_eval": float(t[-1]) if (t := grep_all(eval_log, r"train bpb: ([0-9.]+)")) else None,
    "val_bpb_final_in_training": float(val_bpb_train[-1][1]) if val_bpb_train else None,
    "min_val_bpb_in_training": float(m[-1]) if (m := grep_all(train_log, r"Minimum validation bpb: ([0-9.]+)")) else None,
    "total_training_time_min": float(tt[-1]) if (tt := grep_all(train_log, r"Total training time: ([0-9.]+)m")) else None,
    "peak_memory_allocated_mib": float(p[-1]) if (p := grep_all(train_log, r"Peak memory usage: ([0-9.]+)MiB")) else None,
    "peak_memory_reserved_mib": float(p[-1]) if (p := grep_all(train_log, r"Peak memory reserved: ([0-9.]+)MiB")) else None,
    "grad_accum_steps": int(g[-1]) if (g := grep_all(train_log, r"gradient accumulation steps: (\d+)")) else None,
    "num_iterations": int(n[-1].replace(",", "")) if (n := grep_all(train_log, r"iterations from target data:param ratio: ([0-9,]+)")) else None,
    "resumes": resumes,
    "core_tasks": tasks,
    "core_csv": core_csv,
    "targets": {"core_record_8xh100": 0.2626, "gpt2_core_bar": 0.256525, "core_noise_per_run": 0.008},
}
c = summary["core_standalone_base_eval"]
summary["verdict"] = (None if c is None else
                      "PASS: above the GPT-2 bar and within noise of the record" if c > 0.256525 and abs(c - 0.2626) <= 0.016 else
                      "CHECK: above the GPT-2 bar but >2x noise from the record" if c > 0.256525 else
                      "FAIL: below the GPT-2 bar")
json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
if core_csv:
    import shutil
    shutil.copy(core_csv, os.path.join(R, os.path.basename(core_csv)))  # per-task CORE table, committed with the results
print(json.dumps({k: v for k, v in summary.items() if k != "core_tasks"}, indent=1))

try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT_NAME", "spiky-nanochat"), id=os.environ["WANDB_RUN_ID"], resume="must")
    for k in ("core_standalone_base_eval", "val_bpb_base_eval", "train_bpb_base_eval", "core_in_training_last", "verdict"):
        run.summary[k] = summary[k]
    run.summary["core_metric"] = c  # CORE as the summary metric: the standalone, full-set number
    run.finish()
    print("wandb summary updated")
except Exception as e:
    print(f"WARNING: could not update the wandb summary ({type(e).__name__}: {e}); summary.json is authoritative")

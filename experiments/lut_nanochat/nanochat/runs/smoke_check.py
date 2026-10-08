"""
[lut_nanochat] Mechanical PASS/FAIL check of the smoke run (runs/smoke_run_d24.sh). Prints one line per
criterion, the step-time extrapolation, and `SMOKE RUN: PASS` or `SMOKE RUN: FAIL (n)`; exit code 0/1.
Thresholds marked (choice) are ours, not upstream's.
    python runs/smoke_check.py --results ../results/d24-dense-1xh100-s0-smoke --dbs 16
"""
import argparse, json, math, os, re, statistics

parser = argparse.ArgumentParser()
parser.add_argument("--results", required=True)
parser.add_argument("--dbs", type=int, default=16)
parser.add_argument("--smoke-steps", type=int, default=40)
parser.add_argument("--resume-step", type=int, default=20)
parser.add_argument("--max-reserved-gib", type=float, default=75.0)       # (choice) same rule as the memory probe
parser.add_argument("--max-projected-hours", type=float, default=17.0)    # (choice) estimate is 13-15 h of pretraining
parser.add_argument("--usd-per-gpu-hour", type=float, default=float(os.environ.get("GPU_HOURLY_USD", 3.0)))
parser.add_argument("--skip-wandb", action="store_true", help="offline testing only; the real check must query wandb")
parser.add_argument("--distill", action="store_true", help="distillation smoke: loss is the KL (not CE from ~ln V); require the teacher to have loaded and the KL to decrease")
args = parser.parse_args()
R = args.results
read = lambda f: open(os.path.join(R, f), errors="replace").read() if os.path.exists(os.path.join(R, f)) else ""
train, evallog = read("train.log"), read("base_eval.log")
pins = json.load(open("../pins.json"))
fails = []

def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)

# 1. memory: no OOM at the chosen batch, peak reserved reported and under the limit
oom = re.search(r"out of memory|OutOfMemoryError", train + read("phaseA.out") + read("phaseB.out"), re.I)
res = [float(x) for x in re.findall(r"Peak memory reserved: ([0-9.]+)MiB", train)]
check("no OOM", not oom)
check(f"peak reserved memory reported and <= {args.max_reserved_gib} GiB",
      bool(res) and max(res) / 1024 <= args.max_reserved_gib, f"{max(res)/1024:.1f} GiB" if res else "not reported")

# 2. run shape: 2^20 global batch, grad accum 1048576/(dbs*2048), horizon overridden to the smoke steps
want_accum = 1048576 // (args.dbs * 2048)
accums = set(re.findall(r"Total batch size 1,048,576 => gradient accumulation steps: (\d+)", train))
check(f"global batch 1,048,576 with {want_accum} grad-accum steps (dbs {args.dbs})", accums == {str(want_accum)}, f"seen {sorted(accums)}")

# 3. fp8 and FA3 really active, FA3 at the pinned revision
fp8_lines = re.findall(r"FP8 training enabled \((\w+) scaling\) - converted (\d+)/(\d+)", train)
check("fp8 active (tensorwise, 145/158 Linears) in both phases",
      len(fp8_lines) >= 2 and all(l == ("tensorwise", "145", "158") for l in fp8_lines), f"{fp8_lines}")
fa3_ok = train.count("Using Flash Attention 3") >= 2 and "Flash Attention 3 not available" not in train
check("FA3 active in both phases (no SDPA fallback)", fa3_ok)
pin_cfg = json.load(open(os.path.join(R, "pin_config.json"))) if os.path.exists(os.path.join(R, "pin_config.json")) else {}
rev = pin_cfg.get("runtime", {}).get("fa3_revision")
check("FA3 revision in the run config equals the pin", rev == pins["fa3"]["revision"], f"{rev}")

# 4. loss finite and decreasing from initialisation
steps = [(int(s), float(l), float(dt)) for s, l, dt in
         re.findall(r"^step (\d+)/\d+ \([^)]*\) \| loss: ([0-9.naife+-]+) \|.*?\| dt: ([0-9.]+)ms", train, re.M)]
losses = {s: l for s, l, _ in steps}
finite = bool(losses) and all(math.isfinite(l) for l in losses.values())
first, last = losses.get(0), losses.get(args.smoke_steps - 1)
check("loss finite at every step", finite, f"{len(losses)} steps logged")
if args.distill:
    # distillation: the logged "loss" is the KL( teacher || student ), not CE from ~ln(V); just require it to drop.
    check("frozen teacher loaded (distillation path active)",
          bool(re.search(r"distillation: loading frozen teacher|teacher frozen:", train)))
    check("distillation KL decreased from step 0 to the last step",
          first is not None and last is not None and last < first, f"step0 {first} -> step{args.smoke_steps-1} {last}")
else:
    check("loss decreased by > 1 nat from step 0 to the last step (choice)",
          first is not None and last is not None and last < first - 1.0, f"step0 {first} -> step{args.smoke_steps-1} {last}")

# 5. checkpoint written, run interrupted, auto-resumed from it, and finished
resumed = re.search(rf"Resuming optimization from step {args.resume_step}\b", train)
resumes_log = read("resumes.log")
check(f"step-{args.resume_step} checkpoint written and auto-resumed from by the launcher",
      bool(resumed) and f"resuming from step {args.resume_step}" in resumes_log, resumes_log.strip()[-80:])
reached = max(losses) if losses else -1
check(f"training reached the final step {args.smoke_steps - 1}", reached == args.smoke_steps - 1, f"last logged step {reached}")
check("standalone base_eval ran (reduced) and printed a CORE value", bool(re.search(r"CORE metric: -?[0-9.]+", evallog)))

# 6. step time -> full-run extrapolation (median of steady steps: skip the first 10 after each (re)start)
steady = [dt for s, _, dt in steps if 10 < s < args.resume_step or args.resume_step + 10 < s]
if steady:
    med = statistics.median(steady) / 1000
    hours = med * 5568 / 3600
    print(f"INFO  median step time {med:.2f} s -> pretraining {hours:.1f} h for 5,568 steps; "
          f"end-to-end ~{hours*1.15:.1f}-{hours*1.25:.1f} h; ~${hours*1.2*args.usd_per_gpu_hour:.0f} at ${args.usd_per_gpu_hour}/GPU-h "
          f"(estimate was 13-15 h / 15-18 h / ~$48)")
if args.distill:
    # distillation does TWO forwards/step (~2x the baseline per-step), but converges in far fewer steps, so a
    # 5568-step wall-clock cap is not the right gate here — report the projection as INFO, don't fail on it.
    print("INFO  (distill) per-step is ~2x baseline; the full-5568 projection above is a ceiling, not the plan "
          "(distillation is expected to reach target in far fewer steps) — treat step-time as a cost input, not a gate")
else:
    check(f"projected pretraining <= {args.max_projected_hours} h (choice)",
          bool(steady) and statistics.median(steady) / 1000 * 5568 / 3600 <= args.max_projected_hours)

# 7. wandb: run in the right project/group, name, smoke tag, pin set in the config, resumed into ONE run
if args.skip_wandb:
    print("SKIP  wandb checks (--skip-wandb: offline testing only, NOT acceptable for the real smoke run)")
else:
    try:
        import wandb
        api = wandb.Api()
        project = os.environ.get("WANDB_PROJECT_NAME", "spiky-nanochat")
        run_id = open(os.path.join(R, "wandb_run_id")).read().strip()
        run = api.run(f"{api.default_entity}/{project}/{run_id}")
        cfg = run.config
        check("wandb run in project spiky-nanochat, group nanochat_baseline",
              project == "spiky-nanochat" and run.group == "nanochat_baseline", f"{project}/{run.group}")
        check("wandb run name ends in -smoke with the smoke tag",
              run.name.endswith("-smoke") and "smoke" in run.tags, f"{run.name} {run.tags}")
        check("wandb config carries the pin set (commit, inputs, derived shape)",
              cfg.get("pins", {}).get("static", {}).get("nanochat", {}).get("commit") == pins["nanochat"]["commit"]
              and "inputs" in cfg.get("pins", {}) and "derived" in cfg, "")
        check("phases A+B logged into ONE wandb run up to the last step",
              int(run.summary.get("step", -1)) >= args.smoke_steps - 1, f"summary step {run.summary.get('step')}")
        print(f"INFO  wandb run: {run.url}")
    except Exception as e:
        check("wandb run reachable via the API", False, f"{type(e).__name__}: {e}")

print("SMOKE RUN: PASS" if not fails else f"SMOKE RUN: FAIL ({len(fails)}): " + "; ".join(fails))
raise SystemExit(1 if fails else 0)

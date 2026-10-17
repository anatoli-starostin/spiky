"""Tag a deliberately-stopped wandb run + note; report its state. Usage: tag_stop.py <run_id> <step> <reason>"""
import sys, time
import wandb

ENT, PROJ = "anatoli-starostin-relocation", "spiky-nanochat"
run_id, step, reason = sys.argv[1], sys.argv[2], sys.argv[3]
api = wandb.Api()
r = api.run(f"{ENT}/{PROJ}/{run_id}")
assert r.id == run_id, f"SAFETY: got {r.id}"
print("BEFORE: state=", r.state, "tags=", list(r.tags))
tag = f"stopped-early-at-{step}"
r.tags = sorted(set(list(r.tags) + [tag]))
note = f"Stopped deliberately at step {step} ({reason}). NOT a crash."
r.notes = (r.notes + " | " if r.notes else "") + note
r.update()
time.sleep(3)
r2 = api.run(f"{ENT}/{PROJ}/{run_id}")
print("AFTER: state=", r2.state, "tags=", list(r2.tags))
print("NOTES=", r2.notes)
print("SUMMARY _step=", r2.summary.get("_step"))

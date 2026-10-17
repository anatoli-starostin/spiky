"""wandb housekeeping: mark the aborted 5280 fork run terminal + tag/note it, WITHOUT touching the live run.
Report states of the other runs of interest. Runs OUTSIDE the cage (needs network)."""
import wandb

ENT, PROJ = "anatoli-starostin-relocation", "spiky-nanochat"
ABORTED_ID = "d24-lutffn-confn1-h32-tph32-1xh100-s0-3f8b4134"
LIVE_SUF = "7f4cc747"
api = wandb.Api()
runs = list(api.runs(f"{ENT}/{PROJ}"))


def info(r):
    try:
        step = r.summary.get("_step")
    except Exception:
        step = None
    return f"id={r.id} | state={r.state} | step={step} | tags={list(r.tags)} | created={r.created_at}"


print(f"=== project has {len(runs)} runs ===")
targets = {"ABORTED(3f8b4134)": "3f8b4134", "LIVE(7f4cc747)": "7f4cc747",
           "ea2f89b5": "ea2f89b5", "9dac4d16": "9dac4d16", "v2(9811cced)": "9811cced"}
for key, suf in targets.items():
    ms = [r for r in runs if suf in r.id]
    print(f"[{key}] " + ("  ||  ".join(info(m) for m in ms) if ms else "NOT FOUND"))

# SAFETY: operate on exactly the aborted run, never the live one
ab = [r for r in runs if "3f8b4134" in r.id]
assert len(ab) == 1, f"SAFETY ABORT: expected 1 run matching 3f8b4134, got {len(ab)}"
ab = ab[0]
assert ab.id == ABORTED_ID and LIVE_SUF not in ab.id, f"SAFETY ABORT: id mismatch {ab.id}"
print(f"\nSAFETY OK -> operating ONLY on {ab.id} (state={ab.state}); live run is *{LIVE_SUF}, untouched.")

# tag + note (reliable public-API writes)
ab.tags = sorted(set(list(ab.tags) + ["aborted-wrong-horizon-5280"]))
note = "KILLED at ~4 steps: training horizon auto-derived to 5280 instead of 5568. Superseded by the pinned 5568 relaunch (run *7f4cc747)."
ab.notes = (ab.notes + " | " if ab.notes else "") + note
ab.update()
print(f"TAGS now: {ab.tags}")
print(f"NOTES now: {ab.notes!r}")

# attempt terminal state
before = ab.state
try:
    ab.state = "crashed"
    ab.update()
    after = api.run(f"{ENT}/{PROJ}/{ab.id}").state
    print(f"STATE attempt via run.update(): before={before} -> after={after} ({'OK' if after != 'running' else 'DID NOT STICK'})")
except Exception as e:
    print(f"STATE set FAILED via public API: {type(e).__name__}: {e}")
print("HOUSEKEEPING_DONE")

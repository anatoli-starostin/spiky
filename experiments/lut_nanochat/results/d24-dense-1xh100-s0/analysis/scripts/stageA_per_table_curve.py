"""[lut_nanochat] Stage A helper: per-table (no sharing) rank-r energy curve of the 12 value-embedding tables,
alongside the shared (stacked) curve from data/stageA_results.json. Prints to stdout; reads only."""
import json, os, torch, numpy as np
ANA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))             # .../results/<run>/analysis
R = json.load(open(os.path.join(ANA, "data/stageA_results.json")))
c = R["value_embeds"]["cross"]
print("stacked uncentred:", c["stacked_shared_code_variance_retained_uncentred"])
print("stacked centred:  ", c["stacked_shared_code_variance_retained_centred"])
print("shared+linear:", c["one_shared_table_plus_linear_readout_retained_uncentred"], " shared+scalar:", c["one_shared_table_plus_scalar_gate_retained"])
sd = torch.load(os.path.join(os.path.dirname(ANA), "checkpoints/model_005568.pt"), map_location="cpu", mmap=True)
keys = sorted([k for k in sd if k.startswith("value_embeds.")], key=lambda k: int(k.split(".")[1]))
RS = [32, 64, 128, 256, 512, 1024, 1536]
agg_num = {r: 0.0 for r in RS}; agg_den = 0.0; per = {}
for k in keys:
    s = torch.linalg.svdvals(sd[k].float().to("cuda" if torch.cuda.is_available() else "cpu")).double().cpu().numpy(); e = s ** 2; cum = np.cumsum(e)
    per[k.split(".")[1]] = {r: round(float(cum[r - 1] / e.sum()), 4) for r in RS}
    for r in RS: agg_num[r] += cum[r - 1]
    agg_den += e.sum()
print("per-table alone (uncentred, own SVD):", per)
print("per-table AGGREGATE (energy-weighted over all 12):", {r: round(agg_num[r] / agg_den, 4) for r in RS})
st = c["stacked_shared_code_variance_retained_uncentred"]
print("equal-code-dims comparison: per-table rank r (12r dims total) vs shared rank 12r:")
for r in (32, 64, 128, 256):
    print(f"  r={r}: per-table agg {agg_num[r]/agg_den:.4f}  vs shared rank {12*r}: {st.get(str(12*r), 'n/a')}")

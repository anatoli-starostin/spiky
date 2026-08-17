# Autonomous Night Session Journal (task 7029b478)

Branch `research/hyperplane_ffn_next`. Started ~2026-08-17 ~03:4x local. Two workstreams:
W1 = close the CompressionMHL FFN-slot line (report/commit exp_n_0005). W2 (primary) = Multi-Map
3D Unembedder (MDN head), spec at /tmp/mdn-head-experiment-spec.md. gpustar `_g_` runs + distillation
harness left untouched; all new work under `_n_` / `mdn/`.

## Decisions locked (per owner's instructions in the task)
- **Baseline unembedder W** = `exp070_compressionmhl_A5champ_6h_64-64_nap6_g0_16k/checkpoint.pt`
  (`head.weight`, [32768,384], **untied** — a genuine dense unembedder trained 16k). exp002 (1.20144,
  ideal) has NO checkpoint on disk; exp073 is tied (head==tok_emb, not an independent unembedder). So
  exp070 is the strongest untied head with weights available. Kept bias-free; head-reduction ratios
  computed vs dense 12,582,912 (note spec's 12,615,680 includes a +32768 bias we don't have, ~33k diff).
- Optimizer for MDN params (X, P, b): standard AdamW two-group (planned for E1/E2).
- Budgets: 4k probe / 16k confirm. Cold init = headline; warm (PCA-of-W) arm where spec asks.
- Defaults N=11, M=8, B=3, γ=1e-2.

## Deliverables built
- `mdn/backbone.py` — reusable MinimalGPT+RoPE (faithful copy of the flex trainer, FFN config
  parameterized), `load_pretrained(exp_dir)` loads any exp checkpoint; `.hidden(idx)` returns h=ln_f(x).
  Verified: loads exp070 with **0 missing / 0 unexpected** state-dict keys.
- `mdn/e0_localization.py` — E0 gate (PCA→3D blocks→per-map Gaussian bits + hard-intersection overlap).

## Progress log
### E0 — RESULT: **FAIL** (robust). MDN line HELD at the gate per protocol.

Baseline = exp070 untied head, 1500 real validation contexts, real top-20. (standardized PCA blocks)

| N | 3N | median Σbits (need ≥15) | median top20-in-∩ (need ≥0.50) | median \|∩\|/V |
|--:|--:|--:|--:|--:|
| 4 | 12 | 6.88 | 0.400 | 1.0e-2 |
| 8 | 24 | 10.31 | 0.150 | 1.1e-3 |
| **11** | **33** | **12.58 ❌** | **0.100 ❌** | 3.1e-4 |
| 16 | 48 | 15.87 ✓ | 0.050 ❌ | 9.2e-5 |

No-standardize sensitivity (STD=0, 1000 ctx): N=11 bits 11.35 / overlap 0.150 — also FAIL. So the
verdict is robust to the standardization choice.

**Mechanism (why it fails, and why it's partly a conservative-metric artifact):** the two gate
criteria are structurally ANTI-CORRELATED across N. More maps → higher Σbits (each 3D map tighter/more
informative) but lower hard-intersection overlap, because a token must land inside the 2σ ellipsoid of
**all** N maps simultaneously — an N-way AND that decays ~0.85^N (0.10 at N=11, matching the data). No N
reconciles ≥15 bits with ≥50% overlap: N=4 has overlap 0.40 but only 6.9 bits; N=16 clears 15 bits but
overlap collapses to 0.05. So exp070's embedding geometry does **not** admit axis-blocked 3D localization
under the hard-intersection test.

**Caveat the owner should weigh:** the all-map HARD intersection is the spec's own acknowledged-pessimistic
diagnostic — §4.1 explicitly says "hard set intersection across maps can return empty… which is why the
real head intersects SOFTLY (sum of log-densities)." A soft-product head does not require a token inside
every 2σ ellipsoid; it trades off across maps. So E0-as-specified is a stringent conservative proxy, and
its failure does not strictly prove the SOFT-intersection trained head fails — it proves the cheap
worst-case test does not clear.

**Decision (following the explicit protocol "implement only if E0 passes"):** I am NOT building/training
the MDN head this session. The MDN line is held at E0.

**Morning recommendation:** pick one —
(a) Accept the kill: exp070's geometry doesn't support cheap axis-blocked 3D localization; drop the MDN idea.
(b) Relax E0 to match the actual head: score tokens by the SOFT product (sum of per-map log-densities) and
    check whether the top-20 rank highly under the soft score / a 3σ intersection, rather than a hard 2σ
    AND. If that passes, greenlight E1.
(c) Skip straight to E1 (frozen backbone, train only X,P,b at 4k) as the direct representability test —
    it's cheap (~15 min) and is a far less pessimistic verdict than E0's hard-AND proxy. Say the word and
    I'll run it next session. (I did not run it now because the owner explicitly gated implementation on E0.)
My own lean: (c) then (a) — E1 is the honest go/no-go for a soft head and cheap enough that E0's
conservatism shouldn't kill the idea outright; but I respected the stated gate rather than override it.

Artifacts: `mdn/e0_results.json`, `mdn/e0_run.log`, `mdn/backbone.py`, `mdn/e0_localization.py`.

---

## Workstream 1 (FFN-slot line) — pending exp_n_0005 completion (still training).

_(updated as the session proceeds)_

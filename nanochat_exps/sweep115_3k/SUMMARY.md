# sweep115_3k — final tabulated results

Baseline exp115 @ step 3000: **1.8284**

## Per-run results (best_val_bpb)
| run | best_val_bpb | Δ vs baseline |
|---|---|---|
| run_out_nap10 | 1.8840 | +0.0557 |
| run_out_nap6 | 1.8689 | +0.0406 |
| run_qk_nap10 | 1.8805 | +0.0522 |
| run_qk_nap6 | 1.8615 | +0.0331 |
| run_v_nap10 | 1.8577 | +0.0293 |
| run_v_nap6 | 1.8841 | +0.0557 |

## Decisions
- stage1_qk_winner_endpoint: nap=8 (n6=1.8615079837105706, base=1.8284, n10=1.8805356661665869)
- stage1_v_winner_endpoint: nap=8 (n6=1.8840510324070348, base=1.8284, n10=1.8576748734187496)
- stage1_out_winner_endpoint: nap=8 (n6=1.8689219333523912, base=1.8284, n10=1.8840096130384474)
- joint_per_module: {'qk': 8, 'v': 8, 'out': 8}
- stage3: skipped — baseline (nap=8) wins all modules

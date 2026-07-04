# LUTorch Transformer Profiling

Wall-clock mean time per step (ms); on CUDA, `torch.cuda.synchronize()` is used so times reflect GPU work.

| backend | smooth | n_alt | forward_ms | backward_ms | optimizer_step_ms | elapsed_s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gpu_pure | True | 1 | 36.02 | 14.98 | 2.12 | 5.313 |
| gpu_pure | True | 3 | 145.26 | 28.03 | 2.13 | 17.541 |
| gpu_pure | True | all | 73.81 | 99.41 | 2.17 | 17.538 |
| gpu_pure | False | 1 | 32.10 | 12.45 | 2.13 | 4.668 |
| gpu_pure | False | 3 | 136.66 | 21.84 | 2.15 | 16.065 |
| gpu_pure | False | all | 39.25 | 70.83 | 2.16 | 11.223 |
| gpu_compiled_no_k | True | 1 | 8.80 | 26.75 | 2.10 | 3.764 |
| gpu_compiled_no_k | True | 3 | 120.00 | 28.04 | 2.15 | 15.019 |
| gpu_compiled_no_k | True | all | 60.38 | 47.36 | 2.12 | 10.987 |
| gpu_compiled_no_k | False | 1 | 8.32 | 26.98 | 2.13 | 3.743 |
| gpu_compiled_no_k | False | 3 | 116.90 | 27.78 | 2.15 | 14.683 |
| gpu_compiled_no_k | False | all | 43.56 | 31.81 | 2.14 | 7.751 |
| gpu_compiled_anchor | True | 1 | 7.49 | 9.25 | 2.11 | 1.885 |
| gpu_compiled_anchor | True | 3 | 11.66 | 13.25 | 2.15 | 2.706 |
| gpu_compiled_anchor | True | all | 60.35 | 47.17 | 2.12 | 10.963 |
| gpu_compiled_anchor | False | 1 | 5.31 | 8.26 | 2.13 | 1.570 |
| gpu_compiled_anchor | False | 3 | 8.57 | 10.22 | 2.13 | 2.092 |
| gpu_compiled_anchor | False | all | 43.56 | 31.73 | 2.13 | 7.742 |
| gpu_compiled_all | True | 1 | 5.96 | 7.81 | 2.14 | 1.591 |
| gpu_compiled_all | True | 3 | 11.35 | 13.31 | 2.14 | 2.680 |
| gpu_compiled_all | True | all | 60.41 | 47.40 | 2.15 | 10.996 |
| gpu_compiled_all | False | 1 | 5.31 | 6.73 | 2.14 | 1.417 |
| gpu_compiled_all | False | 3 | 8.56 | 10.25 | 2.13 | 2.095 |
| gpu_compiled_all | False | all | 43.56 | 31.95 | 2.12 | 7.763 |

---

## Per-run details (native profiler when present)

### Backend `gpu_pure`, smooth=True, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 36.02, backward_ms: 14.98, optimizer_step_ms: 2.12, elapsed_s: 5.313
- untrained_loss: 6.693398


### Backend `gpu_pure`, smooth=True, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 145.26, backward_ms: 28.03, optimizer_step_ms: 2.13, elapsed_s: 17.541
- untrained_loss: 6.707610


### Backend `gpu_pure`, smooth=True, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 73.81, backward_ms: 99.41, optimizer_step_ms: 2.17, elapsed_s: 17.538
- untrained_loss: 6.715455


### Backend `gpu_pure`, smooth=False, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 32.10, backward_ms: 12.45, optimizer_step_ms: 2.13, elapsed_s: 4.668
- untrained_loss: 6.687011


### Backend `gpu_pure`, smooth=False, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 136.66, backward_ms: 21.84, optimizer_step_ms: 2.15, elapsed_s: 16.065
- untrained_loss: 6.687011


### Backend `gpu_pure`, smooth=False, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 39.25, backward_ms: 70.83, optimizer_step_ms: 2.16, elapsed_s: 11.223
- untrained_loss: 6.687011


### Backend `gpu_compiled_no_k`, smooth=True, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 8.80, backward_ms: 26.75, optimizer_step_ms: 2.10, elapsed_s: 3.764
- untrained_loss: 6.693398


### Backend `gpu_compiled_no_k`, smooth=True, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 120.00, backward_ms: 28.04, optimizer_step_ms: 2.15, elapsed_s: 15.019
- untrained_loss: 6.707610


### Backend `gpu_compiled_no_k`, smooth=True, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 60.38, backward_ms: 47.36, optimizer_step_ms: 2.12, elapsed_s: 10.987
- untrained_loss: 6.715455


### Backend `gpu_compiled_no_k`, smooth=False, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 8.32, backward_ms: 26.98, optimizer_step_ms: 2.13, elapsed_s: 3.743
- untrained_loss: 6.687011


### Backend `gpu_compiled_no_k`, smooth=False, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 116.90, backward_ms: 27.78, optimizer_step_ms: 2.15, elapsed_s: 14.683
- untrained_loss: 6.687011


### Backend `gpu_compiled_no_k`, smooth=False, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 43.56, backward_ms: 31.81, optimizer_step_ms: 2.14, elapsed_s: 7.751
- untrained_loss: 6.687011


### Backend `gpu_compiled_anchor`, smooth=True, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 7.49, backward_ms: 9.25, optimizer_step_ms: 2.11, elapsed_s: 1.885
- untrained_loss: 6.693398

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 23.5796 ms / 1800 = 0.0130998 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 18.8071 ms / 1800 = 0.0104484 ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_anchor`, smooth=True, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 11.66, backward_ms: 13.25, optimizer_step_ms: 2.15, elapsed_s: 2.706
- untrained_loss: 6.707610

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 24.5059 ms / 1800 = 0.0136144 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 18.8924 ms / 1800 = 0.0104958 ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_anchor`, smooth=True, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 60.35, backward_ms: 47.17, optimizer_step_ms: 2.12, elapsed_s: 10.963
- untrained_loss: 6.715455

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_anchor`, smooth=False, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 5.31, backward_ms: 8.26, optimizer_step_ms: 2.13, elapsed_s: 1.570
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 22.5664 ms / 1800 = 0.0125369 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 18.4238 ms / 1800 = 0.0102354 ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_anchor`, smooth=False, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 8.57, backward_ms: 10.22, optimizer_step_ms: 2.13, elapsed_s: 2.092
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 22.3988 ms / 1800 = 0.0124438 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 19.0008 ms / 1800 = 0.010556 ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_anchor`, smooth=False, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 43.56, backward_ms: 31.73, optimizer_step_ms: 2.13, elapsed_s: 7.742
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_all`, smooth=True, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 5.96, backward_ms: 7.81, optimizer_step_ms: 2.14, elapsed_s: 1.591
- untrained_loss: 6.693398

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 22.158 ms / 1800 = 0.01231 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 16.1101 ms / 1800 = 0.00895007 ms
lutorch::lprojection_backward: 28.2107 ms / 1800 = 0.0156726 ms
lutorch::lprojection_forward_smooth: 21.2199 ms / 1800 = 0.0117888 ms

```


### Backend `gpu_compiled_all`, smooth=True, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 11.35, backward_ms: 13.31, optimizer_step_ms: 2.14, elapsed_s: 2.680
- untrained_loss: 6.707610

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 21.4567 ms / 1800 = 0.0119204 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 17.0849 ms / 1800 = 0.00949162 ms
lutorch::lprojection_backward: 35.5609 ms / 1800 = 0.0197561 ms
lutorch::lprojection_forward_smooth: 21.3043 ms / 1800 = 0.0118357 ms

```


### Backend `gpu_compiled_all`, smooth=True, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 60.41, backward_ms: 47.40, optimizer_step_ms: 2.15, elapsed_s: 10.996
- untrained_loss: 6.715455

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_all`, smooth=False, n_alt=1

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 5.31, backward_ms: 6.73, optimizer_step_ms: 2.14, elapsed_s: 1.417
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 22.4545 ms / 1800 = 0.0124747 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 16.2938 ms / 1800 = 0.0090521 ms
lutorch::lprojection_backward: 28.3013 ms / 1800 = 0.0157229 ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_all`, smooth=False, n_alt=3

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 8.56, backward_ms: 10.25, optimizer_step_ms: 2.13, elapsed_s: 2.095
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 22.0975 ms / 1800 = 0.0122764 ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 17.2477 ms / 1800 = 0.00958204 ms
lutorch::lprojection_backward: 35.428 ms / 1800 = 0.0196822 ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


### Backend `gpu_compiled_all`, smooth=False, n_alt=all

- device: `cuda`, batch_size: 128, profile_steps: 100
- forward_ms: 43.56, backward_ms: 31.95, optimizer_step_ms: 2.12, elapsed_s: 7.763
- untrained_loss: 6.687011

**Native LUTorchManager profiling stats:**

```
lutorch::anchor_pairs_lookup_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_eval_forward: 0 ms / 0 = -nan ms
lutorch::anchor_pairs_lookup_backward: 0 ms / 0 = -nan ms
lutorch::lprojection_backward: 45.5236 ms / 1800 = 0.0252909 ms
lutorch::lprojection_forward_smooth: 0 ms / 0 = -nan ms

```


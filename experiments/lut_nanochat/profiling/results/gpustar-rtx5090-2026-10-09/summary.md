# LUT-FFN d24 block benchmark (NVIDIA GeForce RTX 5090, cc 12.0)

torch 2.9.1+cu130 / CUDA 13.0 / Triton 3.5.1; fp32 matmul precision high; fp8: torch._scaled_mm (e4m3, tensorwise) ran on this device

## Measured peaks [HW-dep]

- gemm_high_fp32_tflops: 120.0
- gemm_bf16_tflops: 241.4
- gemm_fp8_tflops: 473.8
- copy_bandwidth_gbs: 1520.8

## Single block, median ms [HW-dep]

| variant | tokens | fwd ms (IQR) | fwd+bwd ms (IQR) | Mtok/s (f+b) | MACs/token | GEMM TFLOP/s (f+b) | 24 layers / 2^20-tok step (s) | peak alloc / reserved GiB |
|---|---|---|---|---|---|---|---|---|
| dense-fp32 | 8192 | 2.99 (0.01) | 8.96 (0.03) | 0.91 | 18.87M | 103.5 | 27.52 | 0.7 / 0.8 |
| dense-fp32 | 32768 | 11.70 (0.21) | 35.00 (0.19) | 0.94 | 18.87M | 106.0 | 26.88 | 2.4 / 3.1 |
| dense-bf16 | 8192 | 1.57 (0.00) | 4.51 (0.03) | 1.82 | 18.87M | 205.7 | 13.85 | 0.5 / 0.6 |
| dense-bf16 | 32768 | 5.80 (0.30) | 17.63 (0.19) | 1.86 | 18.87M | 210.5 | 13.54 | 1.3 / 1.6 |
| dense-fp8 | 8192 | 0.96 (0.00) | 2.78 (0.01) | 2.94 | 18.87M | 333.3 | 8.55 | 0.5 / 0.6 |
| dense-fp8 | 32768 | 3.59 (0.01) | 10.98 (0.04) | 2.98 | 18.87M | 338.0 | 8.43 | 1.8 / 2.0 |
| lut-fp32 | 8192 | 1.90 (0.00) | 8.83 (0.01) | 0.93 | 2.41M | 13.1 | 27.13 | 0.9 / 1.3 |
| lut-fp32 | 32768 | 7.16 (0.01) | 33.06 (0.03) | 0.99 | 2.41M | 14.0 | 25.39 | 3.2 / 3.9 |
| lut-bf16 | 8192 | 1.62 (0.00) | 8.16 (0.09) | 1.00 | 2.41M | 14.2 | 25.07 | 0.9 / 1.2 |
| lut-bf16 | 32768 | 6.16 (0.02) | 30.43 (0.16) | 1.08 | 2.41M | 15.2 | 23.37 | 3.0 / 3.6 |
| lut-fp8 | 8192 | 1.67 (0.00) | 8.11 (0.02) | 1.01 | 2.41M | 14.3 | 24.91 | 0.9 / 1.3 |
| lut-fp8 | 32768 | 6.44 (0.02) | 30.72 (0.04) | 1.07 | 2.41M | 15.1 | 23.59 | 3.1 / 3.6 |

## components [HW-dep]

```json
{
 "8192": {
  "cast_in bf16->fp32": {
   "fwd_ms": 0.024304000660777092,
   "fwdbwd_ms": 0.0971519984304905
  },
  "compress_fp32": {
   "fwd_ms": 0.28276799619197845,
   "fwdbwd_ms": 0.7008000016212463
  },
  "compress_bf16": {
   "fwd_ms": 0.10761599987745285,
   "fwdbwd_ms": 0.43425600230693817
  },
  "compress_fp8": {
   "fwd_ms": 0.28171199560165405,
   "fwdbwd_ms": 0.7412799894809723
  },
  "cartridge_whole (compiled, train)": {
   "fwd_ms": 1.4441439509391785,
   "fwdbwd_ms": 7.582927942276001
  },
  "  addressing (compiled)": {
   "fwd_ms": 0.417263999581337,
   "fwdbwd_ms": 2.2325119972229004
  },
  "  score (eager)": {
   "fwd_ms": 1.6672319769859314,
   "fwdbwd_ms": 4.537280082702637
  },
  "  score (compiled)": {
   "fwd_ms": 3.786911964416504,
   "fwdbwd_ms": 4.831295967102051
  },
  "  embedding_bag read": {
   "fwd_ms": 1.0756320357322693,
   "fwdbwd_ms": 6.004944086074829
  },
  "embedding_bag_bytes_fwd": 1711276032,
  "bwd-model: sort indices": {
   "fwd_ms": 1.5681759715080261,
   "fwdbwd_ms": null
  },
  "bwd-model: index_add_ grad rows": {
   "fwd_ms": 2.068015933036804,
   "fwdbwd_ms": null
  },
  "bwd-model: psw grad (gather.dot)": {
   "fwd_ms": 7.20137619972229,
   "fwdbwd_ms": null
  },
  "decompress_fp32": {
   "fwd_ms": 0.22542399913072586,
   "fwdbwd_ms": 0.6848479807376862
  },
  "decompress_bf16": {
   "fwd_ms": 0.11182400211691856,
   "fwdbwd_ms": 0.41950400173664093
  },
  "decompress_fp8": {
   "fwd_ms": 0.1708960011601448,
   "fwdbwd_ms": 0.8397440016269684
  },
  "cast_out fp32->bf16": {
   "fwd_ms": 0.014175999909639359,
   "fwdbwd_ms": 0.20948800444602966
  }
 },
 "32768": {
  "cast_in bf16->fp32": {
   "fwd_ms": 0.213919997215271,
   "fwdbwd_ms": 0.5230560004711151
  },
  "compress_fp32": {
   "fwd_ms": 0.7020800113677979,
   "fwdbwd_ms": 2.1682080030441284
  },
  "compress_bf16": {
   "fwd_ms": 0.36505600810050964,
   "fwdbwd_ms": 1.1398079991340637
  },
  "compress_fp8": {
   "fwd_ms": 1.6994240283966064,
   "fwdbwd_ms": 3.4383519887924194
  },
  "cartridge_whole (compiled, train)": {
   "fwd_ms": 5.504703998565674,
   "fwdbwd_ms": 29.655183792114258
  },
  "  addressing (compiled)": {
   "fwd_ms": 1.615231990814209,
   "fwdbwd_ms": 8.779967784881592
  },
  "  score (eager)": {
   "fwd_ms": 7.000607967376709,
   "fwdbwd_ms": 18.84585666656494
  },
  "  score (compiled)": {
   "fwd_ms": 15.069904327392578,
   "fwdbwd_ms": 19.363200187683105
  },
  "  embedding_bag read": {
   "fwd_ms": 4.272495985031128,
   "fwdbwd_ms": 23.526463508605957
  },
  "embedding_bag_bytes_fwd": 6845104128,
  "bwd-model: sort indices": {
   "fwd_ms": 6.400399923324585,
   "fwdbwd_ms": null
  },
  "bwd-model: index_add_ grad rows": {
   "fwd_ms": 8.254303932189941,
   "fwdbwd_ms": null
  },
  "bwd-model: psw grad (gather.dot)": {
   "fwd_ms": 28.758959770202637,
   "fwdbwd_ms": null
  },
  "decompress_fp32": {
   "fwd_ms": 0.7135039865970612,
   "fwdbwd_ms": 2.2838399410247803
  },
  "decompress_bf16": {
   "fwd_ms": 0.37804800271987915,
   "fwdbwd_ms": 1.1964960098266602
  },
  "decompress_fp8": {
   "fwd_ms": 1.008799970149994,
   "fwdbwd_ms": 3.547136068344116
  },
  "cast_out fp32->bf16": {
   "fwd_ms": 0.19315199553966522,
   "fwdbwd_ms": 0.47864000499248505
  }
 },
 "per_step_fixed": {
  "cell_tv_fwdbwd_ms_per_layer": 2.555087924003601,
  "cell_tv_fwdbwd_ms_24_layers": 61.322110176086426,
  "adamw_fused_step_ms_lut_params": 7.48966383934021,
  "lut_params_24_layers": 358668336
 }
}
```

## compile [HW-indep]

```json
{
 "lut-fp32": {
  "explain_graph_count": 1,
  "explain_graph_break_count": 0,
  "explain_op_count": 25,
  "break_reasons": [],
  "counters_after_3_fwdbwd": {
   "frames": {
    "total": 1,
    "ok": 1
   },
   "stats": {
    "calls_captured": 58,
    "unique_graphs": 1
   },
   "inductor": {
    "triton_bundler_read_and_emit_kernel": 208,
    "triton_bundler_load_static_autotuner": 16,
    "async_compile_cache_miss": 47,
    "async_compile_cache_hit": 29,
    "pattern_matcher_count": 1,
    "pattern_matcher_nodes": 1,
    "extern_calls": 16,
    "fxgraph_cache_hit": 2
   },
   "aot_autograd": {
    "total": 1,
    "autograd_cache_hit": 1,
    "ok": 1
   }
  }
 },
 "dense-bf16": {
  "explain_graph_count": 1,
  "explain_graph_break_count": 0,
  "explain_op_count": 3,
  "break_reasons": [],
  "counters_after_3_fwdbwd": {
   "frames": {
    "total": 1,
    "ok": 1
   },
   "stats": {
    "calls_captured": 6,
    "unique_graphs": 1
   },
   "aot_autograd": {
    "total": 1,
    "autograd_cache_miss": 1,
    "ok": 1,
    "autograd_cache_saved": 1
   },
   "inductor": {
    "pattern_matcher_count": 10,
    "pattern_matcher_nodes": 12,
    "triton_bundler_read_and_emit_kernel": 40,
    "triton_bundler_load_static_autotuner": 4,
    "async_compile_cache_miss": 12,
    "async_compile_cache_hit": 8,
    "extern_calls": 6,
    "fxgraph_cache_hit": 2
   }
  }
 }
}
```

## memory [HW-dep]

```json
{
 "dense-fp32": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 1.6875,
  "peak_alloc_fwdbwd_gib": 4.623291015625,
  "peak_reserved_fwdbwd_gib": 4.9609375,
  "saved_for_backward_total_gib": 2.5078125,
  "saved_for_backward_top": [
   {
    "shape": [
     16,
     2048,
     6144
    ],
    "dtype": "float32",
    "MiB": 768.0
   },
   {
    "shape": [
     32768,
     6144
    ],
    "dtype": "float32",
    "MiB": 768.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "float32",
    "MiB": 192.0
   },
   {
    "shape": [
     1536,
     6144
    ],
    "dtype": "float32",
    "MiB": 36.0
   },
   {
    "shape": [
     6144,
     1536
    ],
    "dtype": "float32",
    "MiB": 36.0
   }
  ]
 },
 "dense-bf16": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 0.87890625,
  "peak_alloc_fwdbwd_gib": 2.390869140625,
  "peak_reserved_fwdbwd_gib": 2.5234375,
  "saved_for_backward_total_gib": 1.25390625,
  "saved_for_backward_top": [
   {
    "shape": [
     16,
     2048,
     6144
    ],
    "dtype": "bfloat16",
    "MiB": 384.0
   },
   {
    "shape": [
     32768,
     6144
    ],
    "dtype": "bfloat16",
    "MiB": 384.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "bfloat16",
    "MiB": 96.0
   },
   {
    "shape": [
     1536,
     6144
    ],
    "dtype": "bfloat16",
    "MiB": 18.0
   },
   {
    "shape": [
     6144,
     1536
    ],
    "dtype": "bfloat16",
    "MiB": 18.0
   }
  ]
 },
 "dense-fp8": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 0.7207050323486328,
  "peak_alloc_fwdbwd_gib": 2.768801212310791,
  "peak_reserved_fwdbwd_gib": 4.3984375,
  "saved_for_backward_total_gib": 1.0019531399011612,
  "saved_for_backward_top": [
   {
    "shape": [
     16,
     2048,
     6144
    ],
    "dtype": "bfloat16",
    "MiB": 384.0
   },
   {
    "shape": [
     32768,
     6144
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 192.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 48.0
   },
   {
    "shape": [
     6144,
     1536
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 9.0
   },
   {
    "shape": [
     1536,
     6144
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 9.0
   },
   {
    "shape": [],
    "dtype": "float32",
    "MiB": 0.0
   }
  ]
 },
 "lut-fp32": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 1.3828125,
  "peak_alloc_fwdbwd_gib": 3.3416929244995117,
  "peak_reserved_fwdbwd_gib": 5.0859375,
  "saved_for_backward_total_gib": 1.3448487594723701,
  "saved_for_backward_top": [
   {
    "shape": [
     33554432
    ],
    "dtype": "int64",
    "MiB": 256.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "float32",
    "MiB": 192.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     33554432
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     32768,
     16,
     48
    ],
    "dtype": "float32",
    "MiB": 96.0
   },
   {
    "shape": [
     32768,
     768
    ],
    "dtype": "float32",
    "MiB": 96.0
   },
   {
    "shape": [
     16,
     64,
     256,
     48
    ],
    "dtype": "float32",
    "MiB": 48.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "bool",
    "MiB": 32.0
   },
   {
    "shape": [
     1536,
     768
    ],
    "dtype": "float32",
    "MiB": 4.5
   },
   {
    "shape": [
     768,
     1536
    ],
    "dtype": "float32",
    "MiB": 4.5
   },
   {
    "shape": [
     524288
    ],
    "dtype": "int64",
    "MiB": 4.0
   },
   {
    "shape": [
     16,
     64,
     8
    ],
    "dtype": "int64",
    "MiB": 0.1
   }
  ]
 },
 "lut-bf16": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 1.15283203125,
  "peak_alloc_fwdbwd_gib": 3.1563901901245117,
  "peak_reserved_fwdbwd_gib": 4.7265625,
  "saved_for_backward_total_gib": 1.1998292282223701,
  "saved_for_backward_top": [
   {
    "shape": [
     33554432
    ],
    "dtype": "int64",
    "MiB": 256.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     33554432
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "bfloat16",
    "MiB": 96.0
   },
   {
    "shape": [
     32768,
     16,
     48
    ],
    "dtype": "float32",
    "MiB": 96.0
   },
   {
    "shape": [
     16,
     64,
     256,
     48
    ],
    "dtype": "float32",
    "MiB": 48.0
   },
   {
    "shape": [
     32768,
     768
    ],
    "dtype": "bfloat16",
    "MiB": 48.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "bool",
    "MiB": 32.0
   },
   {
    "shape": [
     524288
    ],
    "dtype": "int64",
    "MiB": 4.0
   },
   {
    "shape": [
     1536,
     768
    ],
    "dtype": "bfloat16",
    "MiB": 2.2
   },
   {
    "shape": [
     768,
     1536
    ],
    "dtype": "bfloat16",
    "MiB": 2.2
   },
   {
    "shape": [
     16,
     64,
     8
    ],
    "dtype": "int64",
    "MiB": 0.1
   }
  ]
 },
 "lut-fp8": {
  "tokens": 32768,
  "activations_held_after_fwd_gib": 1.1751728057861328,
  "peak_alloc_fwdbwd_gib": 3.202655792236328,
  "peak_reserved_fwdbwd_gib": 5.1796875,
  "saved_for_backward_total_gib": 1.1273194774985313,
  "saved_for_backward_top": [
   {
    "shape": [
     33554432
    ],
    "dtype": "int64",
    "MiB": 256.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     33554432
    ],
    "dtype": "float32",
    "MiB": 128.0
   },
   {
    "shape": [
     32768,
     16,
     48
    ],
    "dtype": "float32",
    "MiB": 96.0
   },
   {
    "shape": [
     32768,
     1536
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 48.0
   },
   {
    "shape": [
     16,
     64,
     256,
     48
    ],
    "dtype": "float32",
    "MiB": 48.0
   },
   {
    "shape": [
     32768,
     16,
     64
    ],
    "dtype": "bool",
    "MiB": 32.0
   },
   {
    "shape": [
     32768,
     768
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 24.0
   },
   {
    "shape": [
     524288
    ],
    "dtype": "int64",
    "MiB": 4.0
   },
   {
    "shape": [
     768,
     1536
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 1.1
   },
   {
    "shape": [
     1536,
     768
    ],
    "dtype": "float8_e4m3fn",
    "MiB": 1.1
   },
   {
    "shape": [
     16,
     64,
     8
    ],
    "dtype": "int64",
    "MiB": 0.1
   }
  ]
 }
}
```

## profile [HW-dep]

```json
{
 "lut-fp32": {
  "tokens": 32768,
  "op_calls_per_iter_fwd": {
   "triton_poi_fused__to_copy_view_0": 1.0,
   "cuLaunchKernel": 6.0,
   "aten::addmm": 1.0,
   "aten::randint": 1.0,
   "aten::resize_": 1.0,
   "aten::random_": 1.0,
   "cudaLaunchKernel": 2.0,
   "triton_per_fused__to_copy_abs_add_arange_div_exp_expand_gather_gt_index_log_sigmoid_forward_lt_mul_rand_sub_sum_view_1": 1.0,
   "triton_poi_fused_arange_2": 1.0,
   "aten::_embedding_bag": 1.0,
   "aten::empty": 4.0,
   "aten::mm": 1.0,
   "triton_poi_fused__to_copy_addmm_view_3": 1.0
  },
  "op_calls_per_iter_fwdbwd": {
   "triton_poi_fused__to_copy_view_0": 2.0,
   "cuLaunchKernel": 24.0,
   "aten::addmm": 1.0,
   "aten::randint": 1.0,
   "aten::resize_": 2.0,
   "aten::random_": 1.0,
   "cudaLaunchKernel": 25.0,
   "triton_per_fused__to_copy_abs_add_arange_div_exp_expand_gather_gt_index_log_sigmoid_forward_lt_mul_rand_sub_sum_view_1": 1.0,
   "triton_poi_fused_arange_2": 1.0,
   "aten::_embedding_bag": 1.0,
   "aten::empty": 16.0,
   "aten::mm": 5.0,
   "triton_poi_fused__to_copy_addmm_view_3": 1.0,
   "triton_red_fused_sum_1": 1.0,
   "triton_red_fused_sum_2": 1.0,
   "aten::_embedding_bag_per_sample_weights_backward": 1.0,
   "aten::_embedding_bag_backward": 1.0,
   "aten::_embedding_bag_dense_backward": 1.0,
   "aten::empty_like": 2.0,
   "aten::arange": 2.0,
   "aten::zeros": 1.0,
   "aten::zero_": 1.0,
   "aten::fill_": 1.0,
   "triton_poi_fused__to_copy_div_exp_mul_view_3": 1.0,
   "triton_red_fused_mul_sum_4": 1.0,
   "triton_per_fused_exp_mul_sum_5": 1.0,
   "triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_sgn_view_6": 3.0,
   "triton_red_fused__to_copy_abs_add_div_exp_expand_gather_index_log_sigmoid_backward_log_sigmoid_forward_mul_neg_new_zeros_scatter_add_sgn_sub_sum_unsqueeze_view_7": 1.0,
   "cuLaunchKernelEx": 1.0,
   "triton_red_fused_abs_exp_expand_log_sigmoid_backward_log_sigmoid_forward_mul_sum_unsqueeze_8": 1.0,
   "triton_poi_fused_add_index_put_new_zeros_9": 1.0,
   "triton_red_fused_sum_view_10": 1.0,
   "triton_red_fused_sum_view_11": 1.0,
   "triton_poi_fused__to_copy_12": 1.0,
   "aten::add_": 1.0,
   "aten::detach": 7.0
  },
  "fwd": {
   "cuda_time_per_iter_ms": 7.128475666666666,
   "buckets_ms": {
    "embedding_bag fwd": 4.291,
    "fused_triton (addressing/score/casts)": 1.433,
    "gemm": 1.403,
    "elementwise/other": 0.001
   },
   "top_kernels_ms": [
    {
     "kernel": "void at::native::(anonymous namespace)::EmbeddingBag_updateOutputKernel_sum_mean<float, long>(long const*, long const*, float const*, float*, long*, long, long,",
     "ms": 4.291,
     "bucket": "embedding_bag fwd"
    },
    {
     "kernel": "triton_per_fused__to_copy_abs_add_arange_div_exp_expand_gather_gt_index_log_sigmoid_forward_lt_mul_rand_sub_sum_view_1",
     "ms": 1.053,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4::Params)",
     "ms": 0.705,
     "bucket": "gemm"
    },
    {
     "kernel": "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4::Params)",
     "ms": 0.698,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_poi_fused__to_copy_view_0",
     "ms": 0.201,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_addmm_view_3",
     "ms": 0.177,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused_arange_2",
     "ms": 0.002,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<unsigned long, 2, at::native::templates::cuda::random_from_to_kernel<at::CUD",
     "ms": 0.001,
     "bucket": "elementwise/other"
    }
   ]
  },
  "bwd": {
   "cuda_time_per_iter_ms": 26.45767666666671,
   "buckets_ms": {
    "embedding_bag bwd: per_sample_weights grad": 10.154,
    "embedding_bag bwd: weight grad (sort/unique/scatter)": 9.232,
    "fused_triton (addressing/score/casts)": 3.953,
    "gemm": 2.636,
    "elementwise/other": 0.478,
    "embedding_bag fwd": 0.004
   },
   "top_kernels_ms": [
    {
     "kernel": "void at::native::_embedding_bag_per_sample_weights_backward_kernel<float, long>(float const*, long, long, float const*, long, long, long const*, long const*, lo",
     "ms": 10.154,
     "bucket": "embedding_bag bwd: per_sample_weights grad"
    },
    {
     "kernel": "void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueTy",
     "ms": 5.677,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    },
    {
     "kernel": "void at::native::(anonymous namespace)::compute_grad_weight_bags<float, long>(long const*, float const*, long const*, long const*, long, long, int, long const*,",
     "ms": 2.696,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_add_div_exp_expand_gather_index_log_sigmoid_backward_log_sigmoid_forward_mul_neg_new_zeros_scatter_add_sgn_sub_sum_unsqueeze_view_",
     "ms": 2.572,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>(cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4::Params)",
     "ms": 1.384,
     "bucket": "gemm"
    },
    {
     "kernel": "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4>(cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4::Params)",
     "ms": 1.251,
     "bucket": "gemm"
    },
    {
     "kernel": "void at::native::(anonymous namespace)::sum_and_scatter<float, long>(long const*, float*, long, long const*, long const*, at::AccumulateType<float, true>::type ",
     "ms": 0.432,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_div_exp_mul_view_3",
     "ms": 0.342,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused_add_index_put_new_zeros_9",
     "ms": 0.257,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tenso",
     "ms": 0.243,
     "bucket": "elementwise/other"
    },
    {
     "kernel": "void at_cuda_detail::cub::detail::unique_by_key::DeviceUniqueByKeySweepKernel<at_cuda_detail::cub::detail::unique_by_key::policy_hub<long, int>::Policy1000, lon",
     "ms": 0.215,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_view_0",
     "ms": 0.199,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFl",
     "ms": 0.184,
     "bucket": "elementwise/other"
    },
    {
     "kernel": "void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortHistogramKernel<at_cuda_detail::cub::detail::radix::policy_hub<long, at::cuda::cub::detail::OpaqueT",
     "ms": 0.167,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    },
    {
     "kernel": "triton_red_fused_mul_sum_4",
     "ms": 0.152,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_12",
     "ms": 0.152,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused_expand_mul_neg_new_zeros_scatter_add_sgn_view_6",
     "ms": 0.131,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_red_fused_sum_1",
     "ms": 0.097,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_red_fused_sum_view_10",
     "ms": 0.046,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void at::native::(anonymous namespace)::krn_partial_segment_offset<long>(long*, long const*, long const*, long const*, long const*)",
     "ms": 0.032,
     "bucket": "embedding_bag bwd: weight grad (sort/unique/scatter)"
    }
   ]
  }
 },
 "dense-fp8": {
  "tokens": 32768,
  "op_calls_per_iter_fwd": {
   "triton_red_fused__to_copy_abs_max_view_0": 1.0,
   "cuLaunchKernel": 13.0,
   "triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_view_1": 1.0,
   "triton_red_fused_abs_max_2": 2.0,
   "triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_3": 2.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_4": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_5": 2.0,
   "aten::_scaled_mm": 2.0,
   "aten::t": 4.0,
   "aten::transpose": 4.0,
   "aten::alias": 2.0,
   "aten::as_strided": 2.0,
   "cudaLaunchKernelExC": 2.0,
   "triton_red_fused__to_copy_abs_max_pow_relu_view_6": 1.0,
   "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_view_7": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_pow_relu_t_view_8": 1.0,
   "triton_poi_fused__to_copy_clamp_clone_mul_reciprocal_t_view_9": 1.0,
   "aten::_unsafe_view": 1.0
  },
  "op_calls_per_iter_fwdbwd": {
   "triton_red_fused__to_copy_abs_max_view_0": 2.0,
   "cuLaunchKernel": 24.0,
   "triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_view_1": 2.0,
   "triton_red_fused_abs_max_2": 2.0,
   "triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_3": 2.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_4": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_5": 2.0,
   "aten::_scaled_mm": 6.0,
   "aten::t": 12.0,
   "aten::transpose": 12.0,
   "aten::alias": 6.0,
   "aten::as_strided": 6.0,
   "cudaLaunchKernelExC": 6.0,
   "triton_red_fused__to_copy_abs_max_pow_relu_view_6": 1.0,
   "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_view_7": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_mul_pow_relu_t_view_8": 1.0,
   "triton_poi_fused__to_copy_clamp_clone_mul_reciprocal_t_view_9": 1.0,
   "aten::_unsafe_view": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_reciprocal_t_view_2": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_reciprocal_t_view_3": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_view_4": 1.0,
   "triton_red_fused__to_copy_abs_max_mul_pow_relu_threshold_backward_view_5": 1.0,
   "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_threshold_backward_view_6": 1.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_threshold_backward_view_7": 1.0,
   "triton_poi_fused__to_copy_8": 2.0,
   "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_threshold_backward_view_9": 1.0,
   "aten::add_": 1.0,
   "cudaLaunchKernel": 1.0,
   "aten::detach": 2.0
  },
  "fwd": {
   "cuda_time_per_iter_ms": 3.549688666666665,
   "buckets_ms": {
    "gemm": 3.118,
    "fused_triton (addressing/score/casts)": 0.432
   },
   "top_kernels_ms": [
    {
     "kernel": "sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.322,
     "bucket": "gemm"
    },
    {
     "kernel": "sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.288,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_mul_pow_relu_t_view_8",
     "ms": 0.365,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_max_pow_relu_view_6",
     "ms": 0.232,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_4",
     "ms": 0.086,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_max_view_0",
     "ms": 0.071,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_clamp_clone_mul_reciprocal_t_view_9",
     "ms": 0.07,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_5",
     "ms": 0.057,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused_abs_max_2",
     "ms": 0.049,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_3",
     "ms": 0.005,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_view_7",
     "ms": 0.003,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_view_1",
     "ms": 0.002,
     "bucket": "fused_triton (addressing/score/casts)"
    }
   ]
  },
  "bwd": {
   "cuda_time_per_iter_ms": 7.290329666666672,
   "buckets_ms": {
    "gemm": 6.504,
    "fused_triton (addressing/score/casts)": 0.611,
    "elementwise/other": 0.175
   },
   "top_kernels_ms": [
    {
     "kernel": "sm89_xmma_gemm_e4m3e5m2bf16_e4m3e5m2f32_f32_tn_n_tilesize256x64x64_stage4_warpsize4x1x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.447,
     "bucket": "gemm"
    },
    {
     "kernel": "sm89_xmma_gemm_e4m3e5m2bf16_e4m3e5m2f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.32,
     "bucket": "gemm"
    },
    {
     "kernel": "sm89_xmma_gemm_e4m3e5m2bf16_e4m3e5m2f32_f32_tn_n_tilesize128x64x64_stage4_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.296,
     "bucket": "gemm"
    },
    {
     "kernel": "sm89_xmma_gemm_e4m3e5m2bf16_e4m3e5m2f32_f32_tn_n_tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 1.279,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_threshold_backward_view_7",
     "ms": 0.939,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_max_mul_pow_relu_threshold_backward_view_5",
     "ms": 0.502,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFl",
     "ms": 0.175,
     "bucket": "elementwise/other"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_reciprocal_t_view_2",
     "ms": 0.086,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_view_4",
     "ms": 0.085,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_max_view_0",
     "ms": 0.079,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_8",
     "ms": 0.026,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_reciprocal_t_view_3",
     "ms": 0.025,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_clone_mul_pow_reciprocal_relu_t_threshold_backward_view_9",
     "ms": 0.025,
     "bucket": "gemm"
    },
    {
     "kernel": "sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warpsize2x2x1_tensor16x8x32_execute_kernel__5x_cublas",
     "ms": 0.003,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_threshold_backward_view_6",
     "ms": 0.003,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_view_1",
     "ms": 0.002,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__to_copy_clamp_clone_mul_reciprocal_t_view_9",
     "ms": 0.0,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_mul_reciprocal_t_view_4",
     "ms": 0.0,
     "bucket": "gemm"
    },
    {
     "kernel": "triton_red_fused__to_copy_abs_clamp_max_mul_pow_reciprocal_relu_view_7",
     "ms": -0.0,
     "bucket": "fused_triton (addressing/score/casts)"
    },
    {
     "kernel": "triton_poi_fused__scaled_mm__to_copy_clamp_mul_pow_relu_t_view_8",
     "ms": -0.0,
     "bucket": "gemm"
    }
   ]
  }
 }
}
```

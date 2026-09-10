# Attribution and Licensing

Tuff incorporates code, architecture specs, and mathematical formulations inspired by or ported from open-source projects:

## 1. llama.cpp
- **License:** MIT License — Copyright (c) 2023-2024 The ggml authors
- **Upstream Repository:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Reference checkout:** `/home/nii/workspace/llama.cpp` @ commit `84e908c`
- **Ported Components:**
  - GGML quantization layout definitions (`block_q8_0`, `block_q4_K`, `block_q6_K`) from `ggml/src/ggml-common.h`.
  - Single-token autoregressive Gated Delta-Net forward pass math (`build_delta_net_autoregressive`) from `src/models/delta-net-base.cpp` and `src/models/qwen35moe.cpp`.
  - Quantized mat-vec CUDA kernel math (`mmvq.cu`, `vecdotq.cuh`) from `ggml/src/ggml-cuda/`.
  - **MMVQ routed-expert decode path** (`src/gpu.cu`, parity.md P3), ported from llama.cpp
    @ `84e908c`:
    - `vec_dot_q4_K_q8_1` + `vec_dot_q4_K_q8_1_impl_vmmq` (`ggml/src/ggml-cuda/vecdotq.cuh:910, 508`)
      and `vec_dot_q6_K_q8_1` + `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh:1002, 627`),
      including `VDR_Q4_K_Q8_1_MMVQ`/`VDR_Q6_K_Q8_1_MMVQ` and the `get_int_b2`/`get_int_b4`
      unaligned block readers → `vec_dot_q4_K_q8_1_c` / `vec_dot_q6_K_q8_1_c`.
    - Launch shape and reduction of `mul_mat_vec_q` (`ggml/src/ggml-cuda/mmvq.cu:480-700`),
      with `calc_nwarps`/`calc_rows_per_block` (`mmvq.cu:354, 460`) — Ampere/GENERIC at
      `ncols_dst=1`: nwarps=4, rows_per_cuda_block=1, plus the `small_k` variant used for the
      down projection → `mmvq_q4_K_expert_kernel` / `mmvq_q6_K_expert_kernel` and the
      `*_warprow_kernel` small-K forms.
    - The `ids` expert indirection `channel_x = ids[channel_dst]` (`mmvq.cu:516`), mapped onto
      Tuff's existing LFRU slot table `d_slots[e]`.
    - Activation quantization to Q8_1: `quantize_q8_1` (`ggml/src/ggml-cuda/quantize.cu:53`)
      and the `block_q8_1` layout (`ggml/src/ggml-common.h:258`) → `quantize_q8_1_kernel`,
      `block_q8_1_c`.
    - GLU fusion epilogue: `ggml_cuda_mm_fusion_args_device` (`ggml/src/ggml-cuda/common.cuh:1528`)
      and the fused `tmp`/`tmp_gate` + SwiGLU epilogue (`mmvq.cu:588, 617, 657`) →
      `mmvq_q4_K_gateup_swiglu_kernel`.
  - **AVX2 int8 CPU expert kernels (P2)** — into `src/cpu_ops.c` / `include/cpu_ops.h`:

    | ported symbol | upstream file:line | tuff symbol |
    |---|---|---|
    | `block_q8_K` layout | `ggml/src/ggml-common.h:371` | `block_q8_K` (`include/cpu_ops.h`) |
    | `quantize_row_q8_K_ref` | `ggml/src/ggml-quants.c:2768` | `quantize_row_q8_K` |
    | `nearest_int` | `ggml/src/ggml-impl.h` | `cpu_nearest_int` |
    | `hsum_float_8` | `ggml/src/ggml-cpu/arch/x86/quants.c:43` | `cpu_hsum_float_8` |
    | `get_scale_shuffle_k4` | `ggml/src/ggml-cpu/arch/x86/quants.c:527` | `cpu_get_scale_shuffle_k4` |
    | `get_scale_shuffle` | `ggml/src/ggml-cpu/arch/x86/quants.c:540` | `cpu_get_scale_shuffle` |
    | `MM256_SET_M128I` | `ggml/src/ggml-cpu/arch/x86/quants.c:26` | `MM256_SET_M128I` |
    | `ggml_vec_dot_q4_K_q8_K` (AVX2 branch) | `ggml/src/ggml-cpu/arch/x86/quants.c:2038` | `cpu_vec_dot_q4_K_q8_K` |
    | `ggml_vec_dot_q6_K_q8_K` (AVX2 branch) | `ggml/src/ggml-cpu/arch/x86/quants.c:2426` | `cpu_vec_dot_q6_K_q8_K` |

    Adapted only in mechanical ways: tuff's `uint16_t` fp16 fields and `fp16_to_fp32()`
    helper replace `ggml_half` / `GGML_CPU_FP16_TO_FP32`, the `bs`/`bx`/`by`/`nrc` strided
    signature is dropped (tuff only ever calls `nrc == 1`), and the dot products are
    wrapped in OpenMP row loops (`mat_vec_q4_K_q8K`, `mat_vec_q6_K_q8K`). The arithmetic,
    shuffle tables and accumulation order are unchanged from upstream.
  - **Fused device top-k MoE routing (H1)** (`src/gpu.cu`):
    - Ported from `ggml/src/ggml-cuda/topk-moe.cu`: `topk_moe_cuda` with warp-level `softmax_warp_inplace_f` and iterative argmax top-8 selection and weight normalization (`topk_moe_resident_kernel`).
  - **MMVQ output head (H3)** (`src/gpu.cu`):
    - Ported `vec_dot_q6_K_q8_1` / `mul_mat_vec_q` dense warp-row dot kernel from `ggml/src/ggml-cuda/mmvq.cu` and `vecdotq.cuh` for the 248320-row Q6_K output projection (`mmvq_q6_K_dense_warprow_kernel`).
  - **MMVQ dense Q8_0 projections (H9)** (`src/gpu.cu`):
    - Ported `vec_dot_q8_0_q8_1` (`ggml/src/ggml-cuda/vecdotq.cuh:170`) with `QI8_0` /
      `VDR_Q8_0_Q8_1_MMVQ`, reading the 2-byte-aligned `block_q8_0::qs` through the same
      unaligned block reader used for Q6_K → `vec_dot_q8_0_q8_1_c`, on the `mul_mat_vec_q`
      warp-row launch shape (`mmvq_q8_0_dense_warprow_kernel`), for the dense Q8_0 projections
      (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v`, `attn_output`, shared expert).
  - **Fused Gated Delta-Net (H5)** (`src/gpu.cu`):
    - Ported and fused single-token autoregressive Gated Delta-Net forward pass math from `ggml/src/ggml-cuda/gated_delta_net.cu` and `delta-net-base.cpp` (`fused_gated_delta_net_ar_kernel`, `gpu_delta_fused_gdn_dev`), fusing L2 norm, gating, recurrent state update, per-head RMSNorm, and SiLU gating into a single launch.
  - **CUDA Graph capture and executable update (H2)** (`src/gpu.cu`, `src/engine.c`):
    - Ported CUDA graph capture and executable update mechanism from `ggml/src/ggml-cuda/ggml-cuda.cu:2540–2647, 4180–4210` using `cudaStreamCaptureModeRelaxed`, `--default-stream per-thread`, and `cudaGraphExecUpdate` fallback logic (`cuda_graph_update_exec`, `gpu_capture_delta_graphs`, `gpu_launch_delta_layer_graph`).

## 2. Colibrì
- **License:** Apache License 2.0
- **Upstream Repository:** `/home/nii/workspace/colibri`
- **Ported Components:**
  - LFRU cache eviction algorithms and score decay policies (`tier.h`, `expert_store.h`).

# GPU Backend Architecture & Refactoring Plan: Modularizing `src/gpu.cu`

## 1. Executive Summary & Problem Statement

`src/gpu.cu` is currently a monolithic compilation unit of **7,258 lines** (343 KB) containing:
- 132 static/global state variables
- 64 `__global__` CUDA kernels
- Multiple distinct subsystems: VRAM LFRU cache, dense GEMV/MMVQ, Delta-Net recurrent state, FlashAttention/MHA, MoE top-k router, routed expert GEMV+SwiGLU, vocabulary argmax reduction, CUDA Graph capture, two-column speculative verify trunk, and MTP drafting.

### The Problem for LLM Agents and Developers
1. **Context Bloat**: Inspecting or editing any single GPU feature (e.g., KV cache attention or the MoE cache) requires loading up to 50k+ tokens of CUDA code, consuming context budget and causing hallucinations or accidental regressions in unrelated subsystems.
2. **High Blast Radius**: Edits to local helper functions or global scratch pointers risk silent regressions in other kernel launch paths.
3. **Slow Compilation**: The entire 7,258-line file must be recompiled by `nvcc` sequentially whenever a single line changes, bottlenecking iteration velocity.

### Refactoring Objectives
- **Semantic Modularization**: Decompose `src/gpu.cu` into focused, semantically cohesive files of **200–750 lines each**.
- **Context Economy**: Enable agents to work on an isolated subsystem (e.g. `gpu_attention.cu` or `gpu_expert_cache.cu`) without context bloat.
- **Zero Performance Degradation**: Maintain 100% of the current roofline performance (~91% bus saturation, 60.8+ tok/s).
- **Public API Stability**: Retain `include/gpu.h` verbatim; zero changes required in `engine.c`, `main.c`, `server.c`, etc.

---

## 2. Proposed Directory & File Tree

The refactored GPU backend will reside under `src/gpu/` with internal headers and focused translation units (dropping redundant `gpu_` prefixes since the directory is already `gpu/`):

```
tuf/
├── include/
│   └── gpu.h                               # Public C API (100% untouched & backward-compatible)
└── src/
    ├── gpu/
    │   ├── state.h                         # Internal shared state struct & subsystem pointer declarations
    │   ├── common.cuh                      # Device inline primitives: warp_reduce, cp_async, fast math
    │   ├── mmvq.cuh                        # Device inline dequantization dot-products (vec_dot_q4_K, q6_K, q8_0)
    │   │
    │   ├── context.cu                      # gpu_prepare, gpu_shutdown, device memory & scratch management (~450 lines)
    │   ├── expert_cache.cu                 # M4/P1 VRAM expert cache: O(1) LRU, freelist, bgfill thread (~750 lines)
    │   ├── dense_gemv.cu                   # Dense Q8_0/Q6_K/F32 GEMV & MMVQ, dual/quad/triple projections (~650 lines)
    │   ├── delta.cu                        # Gated Delta-Net: conv1d, l2norm, gate, fused GDN recurrent step (~600 lines)
    │   ├── attention.cu                    # Self-Attention: mRoPE, fused QK norm, FlashAttention & cp.async KV (~650 lines)
    │   ├── router.cu                       # MoE router: logits GEMV, warp-level softmax, top-k selection (~450 lines)
    │   ├── moe.cu                          # MoE expert GEMV: Q4_K/Q6_K, SwiGLU fusion, combine & accum (~750 lines)
    │   ├── output.cu                       # Output head: fused RMS-Norm, Q8_1 quant, argmax reduction & logits (~400 lines)
    │   ├── graphs.cu                       # CUDA Graphs: capture & replay for single-token & verify layers (~400 lines)
    │   ├── verify_n2.cu                    # H7/H15 Speculative 2-column verify trunk & state snapshot/rollback (~450 lines)
    │   └── mtp.cu                          # Native Multi-Token Prediction (MTP) speculative drafting (~200 lines)
    └── gpu.cu                              # Thin coordinator / transitional wrapper (or replaced in CMake)
```

---

## 3. Detailed Component Breakdown

### A. Shared Headers & Device Inline Primitives

#### 1. `src/gpu/common.cuh` (~150 lines)
*Header-only device inline primitives (no compilation unit).*
- **Warp Primitives**: `warp_reduce_sum`, `warp_reduce_max` using `__shfl_xor_sync` and `__shfl_down_sync`.
- **Ampere Asynchronous Copy**: `cp_async16`, `cp_async_wait_all` via PTX assembly (`cp.async.ca.shared.global`).
- **Mathematical Helpers**: fast reciprocal square roots, bounds clamps, FP16 conversions.
- *Strict Invariant*: All functions must be marked `__device__ __forceinline__` to guarantee zero device call overhead.

#### 2. `src/gpu/mmvq.cuh` (~250 lines)
*Header-only dequantization and dot-product intrinsics.*
- **Bit Extraction**: `get_int_b2_c`, `get_int_b4_c` for sub-byte quantized weights.
- **Vector Dot Products**:
  - `vec_dot_q4_K_q8_1_c`: Q4_K block dot Q8_1 activation using `__dp4a`.
  - `vec_dot_q6_K_q8_1_c`: Q6_K block dot Q8_1 activation.
  - `vec_dot_q8_0_q8_1_c`: Q8_0 block dot Q8_1 activation.
- *Strict Invariant*: Inlining these into the calling `.cu` kernel translation units avoids Relocatable Device Code (`-rdc=true`), preserving peak register optimization.

#### 3. `src/gpu/state.h` (~200 lines)
*Internal state definition shared across `src/gpu/*.cu` translation units.*
- Consolidates scattered static globals into clean, explicit structures:
  - `gpu_scratch_t`: persistent device vectors (`d_res_x`, `d_norm_x`, `d_pos`, `d_q_full`, `d_k`, `d_v`, etc.).
  - `gpu_streams_t`: dedicated streams (`upload_stream`, `expert_stream`, `d2h_stream`, `graph_stream`) and events.
  - `gpu_kv_state_t`: device KV cache allocations (`dk_cache`, `dv_cache`) and context limits.
  - `gpu_n2_state_t`: two-column verify scratch (`d_res_x2`, `d_norm_x2`, recurrent snapshots).
- Declares internal helper prototypes shared between GPU translation units (e.g. `launch_mat_vec_q8_0_accum`).

---

### B. Subsystem Translation Units

#### 1. `src/gpu/context.cu` (~450 lines)
- **Role**: Memory lifecycle, device context initialization, and buffer allocation.
- **Functions**:
  - `gpu_prepare`, `gpu_prepare_ex`, `gpu_shutdown`
  - `gpu_vram_used_mb`
  - Scratch allocation & sizing: `g_dx`, `g_dy`, `g_d_res_x`, `g_d_norm_x`, `g_d_pos`
  - Pinned host memory allocator: `gpu_host_alloc`, `gpu_host_free`
- **Dependencies**: `state.h`, `cuda_check.h`.

#### 2. `src/gpu/expert_cache.cu` (~750 lines)
- **Role**: High-performance VRAM expert cache management (M4 / P1).
- **Functions & Structs**:
  - Pool memory: `g_pool`, slot mapping, direct-map layer lookup.
  - O(1) LRU & Freelist doubly linked list: `lru_push_front`, `lru_remove`, `freelist_pop`, `select_tail_victim_slot`.
  - Background fill thread: `bgfill_thread_fn`, `bgfill_start`, `bgfill_stop`, request ring buffer `g_bg`.
  - Cache API: `gpu_expert_cache_init`, `gpu_expert_cache_free`, `gpu_expert_lookup`, `gpu_expert_cache_alloc_slot`, `gpu_expert_cache_fill*`, `gpu_expert_cache_bgfill*`.
- **Dependencies**: `state.h`, `moe_policy.h`, `pthread.h`.

#### 3. `src/gpu/dense_gemv.cu` (~650 lines)
- **Role**: Dense weight projections (attention projections, delta projections, shared expert projections).
- **Kernels**:
  - `quantize_q8_1_kernel`
  - `mmvq_q8_0_dense_warprow_kernel`, `mmvq_q8_0_dense_ncols_kernel`
  - `mmvq_q8_0_dense_dual_warprow_kernel`, `mmvq_q8_0_dense_delta_quad_warprow_kernel`, `mmvq_q8_0_dense_triple_warprow_kernel`
  - `mmvq_q8_0_dense_shexp_swiglu_kernel`
  - Fallback scalar kernels: `mul_mat_vec_q8_0_kernel`, `mul_mat_vec_f32_kernel`, etc.
- **Dispatchers**:
  - `gpu_mat_vec`, `gpu_mmvq_q8_0_dense`, `gpu_mmvq_q8_0_dense_dual`, `gpu_shared_expert_projections`
- **Dependencies**: `mmvq.cuh`, `common.cuh`, `state.h`.

#### 4. `src/gpu/delta.cu` (~600 lines)
- **Role**: Gated Delta-Net (GDN) linear recurrent attention layer.
- **Kernels**:
  - `delta_conv_kernel` (1D short convolution + SiLU over 8192 channels)
  - `delta_l2norm_dual_kernel` (per-head L2 normalization)
  - `delta_gate_kernel` (sigmoid/softplus/decay gate computation)
  - `delta_state_update_kernel` & `fused_gated_delta_net_ar_kernel` (recurrent state $S_{t} = \alpha S_{t-1} + v k^T$)
- **Host Dispatchers**:
  - `gpu_delta_pre`, `gpu_delta_state_update`, `gpu_delta_state_update_dev`, `gpu_delta_post`, `gpu_delta_post_accum`
  - `gpu_delta_fused_gdn_dev`, `gpu_delta_fused_gdn_dev_p`, `gpu_delta_fused_gdn_dev_col`
  - State persistence: `gpu_reset_delta_state`, `gpu_save_recurrent_state`, `gpu_restore_recurrent_state`
- **Dependencies**: `common.cuh`, `state.h`.

#### 5. `src/gpu/attention.cu` (~650 lines)
- **Role**: Standard MHA / GQA Attention with RoPE and KV cache.
- **Kernels**:
  - `fused_q_split_norm_rope_kernel`, `fused_k_norm_rope_kernel`, `fused_qk_split_norm_rope_kernel` (multi-head RoPE + RMSNorm fusion)
  - `flash_attn_vec_online_kernel` (online softmax vector attention)
  - `attention_kernel` (Q8_0 / F32 KV cache traversal with Ampere `cp.async` pipeline)
- **Host Dispatchers**:
  - `gpu_attention`, `gpu_attention_layer_dev`
  - `gpu_set_kv_quant`, `gpu_get_kv_quant`, `gpu_reset_kv`
  - Buffer sizing: `kv_cache_ensure`, `attn_scores_ensure`
- **Dependencies**: `common.cuh`, `state.h`.

#### 6. `src/gpu/router.cu` (~450 lines)
- **Role**: MoE gating, router logits computation, and top-k expert selection.
- **Kernels**:
  - `softmax_warp_inplace_f`
  - `topk_moe_kernel`, `topk_moe_resident_kernel` (warp-level top-8 selection and softmax normalization)
- **Host Dispatchers**:
  - `gpu_router_logits_launch`, `gpu_router_logits_wait`, `gpu_router_logits_wait_event`
  - `gpu_router_topk_launch`, `gpu_router_topk_copy_out`
- **Dependencies**: `common.cuh`, `mmvq.cuh`, `state.h`.

#### 7. `src/gpu/moe.cu` (~750 lines)
- **Role**: MoE routed expert execution and combine/residual accumulation.
- **Kernels**:
  - `mmvq_q4_K_expert_kernel`, `mmvq_q6_K_expert_kernel`
  - `mmvq_q4_K_gateup_swiglu_kernel`, `mmvq_q4_K_gateup_swiglu_warprow_kernel`
  - `mul_mat_vec_q4_K_batch_kernel`, `mul_mat_vec_q6_K_batch_kernel`
  - `swiglu_dev_kernel`, `swiglu_dev_ncols_kernel`
  - `moe_combine_resident_kernel`, `moe_combine_accum_kernel`, `moe_combine_accum_vec4_kernel`
  - `qwen_moe_swiglu_pre_gu_kernel`, `qwen_moe_combine_unified_kernel`
- **Host Dispatchers**:
  - `gpu_expert_mat_vec`, `gpu_expert_gateup_launch`, `gpu_expert_down_launch`
  - `gpu_expert_batch_gateup_launch`, `gpu_expert_batch_down_launch`
  - `gpu_expert_moe_gateup_swiglu`, `gpu_expert_moe_down`, `gpu_expert_moe_gateup_swiglu_dev`
  - `gpu_moe_combine_accum`, `gpu_qwen_moe_combine_unified`, `gpu_expert_resident_layer_dev`
- **Dependencies**: `mmvq.cuh`, `common.cuh`, `state.h`.

#### 8. `src/gpu/output.cu` (~400 lines)
- **Role**: Final layer norm, output projection head, and vocabulary argmax reduction.
- **Kernels**:
  - `fused_rms_norm_quantize_q8_1_kernel`
  - `mmvq_q6_K_dense_argmax_warprow_kernel` (direct fused Q6_K dotp + register-level argmax reduction)
  - `argmax_block_kernel`, `argmax_final_kernel`
- **Host Dispatchers**:
  - Greedy argmax: `gpu_output_argmax_dev`, `gpu_fused_output_norm_quant_argmax_dev`
  - Stochastic logits: `gpu_output_logits_dev`, `gpu_fused_output_norm_quant_logits_dev`
- **Dependencies**: `mmvq.cuh`, `common.cuh`, `state.h`.

#### 9. `src/gpu/graphs.cu` (~400 lines)
- **Role**: CUDA Graph capture and launch for deterministic decode layers.
- **Functions**:
  - Single-column capture: `gpu_capture_delta_graphs`, `gpu_launch_delta_layer_graph`, `gpu_launch_layer_graph`
  - Graph query: `gpu_has_delta_graphs`, `gpu_graph_launch_count`, `gpu_set_device_pos`
  - Two-column verify graphs: `gpu_capture_delta_graphs_n2`, `gpu_launch_delta_layer_graph_n2`, `gpu_has_delta_graphs_n2`
- **Dependencies**: `state.h`.

#### 10. `src/gpu/verify_n2.cu` (~450 lines)
- **Role**: Speculative decoding two-column verify trunk (H7 / H15).
- **Functions & Kernels**:
  - Buffer setup: `gpu_n2_prepare`, `gpu_residual_reset_n2`, `gpu_rms_norm_dev_n2`, `gpu_n2_quantize_norm_x`
  - Batched 2-column projections: `gpu_delta_projections_n2`, `gpu_shared_expert_dev_n2`
  - 2-Column MoE dispatch: `gpu_expert_moe_gateup_swiglu_dev_n2_shared`, `gpu_expert_moe_down_dev_n2_shared`
  - State rollback: `gpu_delta_snapshot_layer_n2`, `gpu_delta_rollback_n2`
  - 2-Column argmax: `gpu_output_argmax_n2`
- **Dependencies**: `state.h`, `common.cuh`, `mmvq.cuh`.

#### 11. `src/gpu/mtp.cu` (~200 lines)
- **Role**: Multi-Token Prediction (MTP) speculative drafting engine.
- **Kernels**:
  - `concat_norms_kernel`
  - `accum_inplace_kernel`
- **Host Dispatchers**:
  - `gpu_mtp_draft_dev`
- **Dependencies**: `state.h`, `common.cuh`.

---

## 4. Performance Preservation Invariants (Zero-Regression Guarantee)

CUDA performance is hypersensitive to compilation flags, instruction scheduling, and memory access patterns. This refactoring is engineered to guarantee **zero performance loss**:

### 1. Inlining & No Relocatable Device Code (`-rdc`)
- **Hazard**: Splitting device functions across `.cu` files without inlining requires `-rdc=true`, which introduces device function call overhead, register spills, and disables global register optimization.
- **Guarantee**: All low-level math routines (`warp_reduce_*`, `vec_dot_*`, `cp_async*`) remain in `.cuh` headers marked `__device__ __forceinline__`. Every kernel compiles into a self-contained translation unit without `-rdc=true`.

### 2. Launch Bounds & Register Occupancy Preservation
- **Hazard**: Inadvertent register footprint increases can trigger local DRAM spilling (e.g. H52 register limit on `fused_gated_delta_net_ar_kernel`).
- **Guarantee**: All `__launch_bounds__(threads, min_blocks)` attributes and kernel loop structures are preserved exactly.

### 3. Memory Layout & Zero-Copy Invariant
- **Hazard**: Allocating extra temporary buffers or altering pointer alignments adds PCIe/VRAM traffic.
- **Guarantee**: All persistent device scratch buffers retain their exact 128-byte aligned offsets in `state_t`. Pointer hand-offs (e.g., `dnet_out` remaining device-resident for `gpu_delta_post_accum`) remain unchanged.

### 4. CUDA Graph Replay Integrity
- **Hazard**: Graph captures fail or stall if pointers or stream dependencies change between passes.
- **Guarantee**: Kernel launch call chains captured within `graphs.cu` invoke the identical launch functions on `g_graph_stream` with device-resident arguments (`g_d_pos`).

---

## 5. Build System Integration (`CMakeLists.txt`)

In `CMakeLists.txt`, replace the single `src/gpu.cu` entry with the clean list:

```cmake
set(GPU_SOURCES
    src/gpu/context.cu
    src/gpu/expert_cache.cu
    src/gpu/dense_gemv.cu
    src/gpu/delta.cu
    src/gpu/attention.cu
    src/gpu/router.cu
    src/gpu/moe.cu
    src/gpu/output.cu
    src/gpu/graphs.cu
    src/gpu/verify_n2.cu
    src/gpu/mtp.cu
)
```

And link `${GPU_SOURCES}` into targets `tuff`, `tuff-server`, `microbench-gpu`, and `qwen-cpu-expert-test`.

### Build Velocity Benefit
Instead of one heavy 7,258-line compilation step, `make -j$(nproc)` / `ninja` compiles all 11 `.cu` files concurrently across CPU cores, reducing incremental build time by ~4–8x.

---

## 6. Phased Execution & Verification Roadmap

| Phase | Action | Verification Gate |
|---|---|---|
| **Phase 1** | Extract shared headers (`common.cuh`, `mmvq.cuh`, `state.h`). | Verify headers compile cleanly without syntax or symbol conflicts. |
| **Phase 2** | Extract memory & cache (`context.cu`, `expert_cache.cu`). | Verify `gpu_prepare` and `gpu_shutdown` allocate and free expected VRAM pool. |
| **Phase 3** | Extract compute kernels (`dense_gemv.cu`, `delta.cu`, `attention.cu`, `router.cu`, `moe.cu`, `output.cu`). | Pass `qwen-cpu-expert-test` (bit-identity vs. CPU reference). |
| **Phase 4** | Extract graphs & speculation (`graphs.cu`, `verify_n2.cu`, `batch.cu`, `mtp.cu`). | Verify CUDA graph capture replay count and token output determinism. |
| **Phase 5** | Run microbenchmark suite & full inference benchmark. | `microbench-gpu` & `tuff` report identical tok/s and zero regression. |

---

## 7. Implementation & Verification Results (PASSED)

The refactoring has been fully implemented into `src/gpu/` with clean names (no `gpu_` prefix). Both the monolithic build (`build-monolith`) and the modular build (`build`) were evaluated side-by-side:

### A. Bit-Identity & Numerical Parity Gates
1. **Golden Regression Gate (`tests/verify_qwen_golden.sh`)**:
   - `[golden] qwen35 plain: tokens 128/128 identical, logits 128/128 steps byte-identical -> PASS`
2. **Layer-by-Layer CPU vs GPU Expert Test (`qwen-cpu-expert-test`)**:
   - `287 / 287 cases byte-identical -> PASS`
3. **Speculative Verification & Batch Engine Test (`qwen-batch-test`)**:
   - `ALL TESTS PASSED (N=1, N=2, N=4, N=8, ROLLBACK) -> PASS`
4. **Intermediate Tensor Dump Parity (`TUFF_DUMP_DIR`)**:
   - All 615 tensor files across all 41 layers are 100% byte-identical between monolithic and modular builds.

### B. A/B Performance Benchmark
Generation speed and system metrics over 3 runs (32 warmup + 128 timed tokens) on Qwen3.6-35B-A3B:

| Metric | Monolithic Build (`build-monolith`) | Modular Build (`build`) | Delta |
|---|---|---|---|
| **Pipeline Generation Speed** | 47.28 +/- 0.96 tok/s | 47.30 +/- 1.24 tok/s | **+0.04% (Identical)** |
| **Host RSS Memory** | 21,715.7 MiB | 21,716.0 MiB | **0.00%** |
| **VRAM Usage** | 7,796.6 MiB | 7,796.6 MiB | **Exact Match** |
| **Expert-Cache Hit Rate** | 78.6% (134,319 / 36,561) | 78.6% (134,319 / 36,561) | **Exact Match** |
| **N-gram Speculative Alpha** | 58.8% (10/17), 1.085 tok/step | 58.8% (10/17), 1.085 tok/step | **Exact Match** |

